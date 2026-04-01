"""
llm_interpreter.py
------------------
LLM-based short-term intent interpreter for PGIM.

3-stage architecture:
  Stage 1 — Short-term interpretation (THIS FILE, interpret_with_llm):
             input:  recent window + compact persona summary
             output: deviation_reason, confidence, raw_llm_goal_concepts
             LLM focuses on semantic interpretation only.

  Stage 2 — Grounded goal selection / validation (grounded_selector.py):
             input:  Stage 1 hypothesis + candidate_concept_bank + persona prior
             output: validated_goal_concepts
             Free generation forbidden; closed concept space from backbone candidates.

  Stage 3 — Modulation (signal_builder.py):
             only validated_goal_concepts enter the signal builder.
             raw_llm_goal_concepts preserved for diagnostics/ablation only.

Interface:
    interpret_with_llm(ctx, persona_summary, intent_cfg, openai_client,
                       candidate_concept_bank=None) -> raw dict

Output schema is backward-compatible with parser.parse_intent().
New fields:
    raw_llm_goals:         original LLM goal_concepts (before Stage 2 validation)
    validated_goal_concepts: Stage 2 output (grounded to candidate bank)
    grounding_diagnostics: per-field Stage 2 audit trail

If candidate_concept_bank is None, Stage 2 is skipped and validated_goal_concepts
falls back to goal_concepts (backward-compat mode).

Design constraints:
- No open generation: goal_concepts must be drawn from a provided candidate set
- deviation_reason restricted to the 5-value taxonomy
- Structured output via JSON mode (response_format={"type": "json_object"})
- Heuristic fallback on any failure

Version: B (grounded selector integrated)
- Stage 1: semantic-primary context (goal-eligible concepts as PRIMARY signal)
- Stage 2: candidate_concept_bank activation gate + persona conflict suppression
- raw_llm_goals preserved for ablation comparison
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.intent.context_extractor import IntentContext
from src.intent.concept_roles import is_goal_eligible, is_semantic_goal, filter_non_semantic_goals
from src.intent.grounded_selector import validate_and_select_goals

logger = logging.getLogger(__name__)

# ── Taxonomy ──────────────────────────────────────────────────────────────────
VALID_REASONS = {"aligned", "exploration", "task_focus", "budget_shift", "unknown"}

# ── Versioning — bump when prompt schema or injected fields change ─────────────
LLM_PROMPT_VERSION = "v5_semantic_goal_hygiene"  # v5: strict semantic-only goal slot + v4 reason calibration
SCHEMA_VERSION     = "3.0"                       # parquet schema version for downstream compat checks

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a short-term intent classifier for an Amazon Movies & TV recommendation system.

Your task: given a user's recent watch/browse history and their long-term persona, identify
what the user is specifically interested in RIGHT NOW, and how it differs from their usual taste.

════════════════════════════════════════════════════════════
OUTPUT RULES (strictly enforced)
════════════════════════════════════════════════════════════
1. goal_concepts: select 1–4 items from candidate_concepts ONLY. Do not invent IDs.
   - Prefer SEMANTIC genre/subgenre/theme/mood concepts (e.g. category:action, category:drama)
   - NEVER put price_band:* or format:* in goal_concepts — these go into constraints only
   - NEVER put platform/container concepts in goal_concepts:
     price_band:unknown, price_band:low, price_band:high → constraints only
     format:dvd, format:blu-ray → constraints only
     category:prime_video, category:movies_&_tv, category:featured_categories → FORBIDDEN in goal slot
   - If semantic signal is absent, return goal_concepts: [] (empty list is valid)
   - If a more specific subcategory exists, prefer it over a broad category
2. constraints: dict with keys "price_band" and/or "format" if clearly present; else {}
   - price/format information MUST go here, not into goal_concepts
3. deviation_reason: exactly one of the 5 values below
4. confidence: be conservative — reserve 0.85+ for very clear cases only
5. Return ONLY valid JSON, no markdown, no free text outside JSON values
6. llm_explanation_short: max 2 sentences; phrase-level evidence only (no chain-of-thought)
7. why_not_aligned / why_exploration: fill only when semantically applicable; use "" otherwise

════════════════════════════════════════════════════════════
DEVIATION REASON DECISION RULES
════════════════════════════════════════════════════════════

Step 1 — Check budget_shift FIRST:
  → budget_shift IF recent_dominant_price_band is present AND clearly differs from
    persona_dominant_price_band (e.g. recent=price_band:low vs persona=price_band:high)
  → Do NOT use budget_shift for minor differences or when price_band is absent

Step 2 — Check task_focus SECOND (before aligned/exploration):
  → task_focus means: the user has a narrow, distinctly focused short-term objective —
    they are purposefully drilling into a specific sub-genre, subcategory, or content cluster
    that is MORE SPECIFIC than their usual persona-level preference.
  → task_focus requires BOTH of the following:
      (A) CONCENTRATION: semantic_top_dominance >= 0.45 AND the top 1–2 semantic concepts
          account for most of the semantic recent activity.
      (B) SPECIFICITY: the concentrated concept(s) represent a more specific direction
          than what the persona broadly covers. Ask: "Is this a narrowing-down beyond
          the persona's usual range, or just normal repeated consumption of what they
          always watch?"
          - If the top concept IS already one of persona's top-3 concepts AND no
            more-specific subcategory is present in recent, this is likely aligned,
            not task_focus.
          - If the top concept is a subcategory or genre the persona shows only
            weakly (not in persona top-3), this signals a purposeful drill.
  → KEY EXAMPLES OF WHAT IS NOT task_focus:
      - User's persona top concept is category:drama; recent is also mostly category:drama
        → This is aligned (normal consumption), not task_focus.
      - User always watches action; recent has category:action(6), category:movies_&_tv(3)
        → aligned if action is already the persona's dominant concept.
      - Recent is mostly category:prime_video(10) with a few sparse content concepts
        → NOT task_focus. prime_video is a platform anchor, not a content genre drill.
        → If semantic drift concepts exist (outside persona), use exploration instead.
  → KEY EXAMPLES OF WHAT IS task_focus:
      - Persona is broad (drama/action/comedy); recent is almost entirely category:horror
        → task_focus (horror is a narrowing beyond the persona's usual range).
      - Persona is drama/comedy; recent heavily clusters on category:anime + category:animation
        → task_focus (anime cluster is a distinct short-term objective).
  → goal_concepts for task_focus: pick the 1–2 dominant semantic concepts from recent.

Step 3 — Distinguish aligned vs exploration:
  → DEFAULT RULE: when in doubt between aligned and exploration, choose aligned.
    Exploration requires POSITIVE evidence of genuine semantic drift — not merely
    low overlap or a few unfamiliar concepts.

  → aligned when ANY of the following:
      (a) semantic_overlap_ratio >= 0.40 AND no repeated drift concepts outside persona
          (recent is substantially within the persona's known territory)
      (b) recent top concepts are broad/generic (drama, comedy, action, movies_&_tv)
          and the persona covers those same broad categories — broad category overlap
          is NOT exploration even at low counts
      (c) only format/price/platform differences exist but semantic categories are
          persona-consistent — format or price change alone is NOT exploration
      (d) semantic signal is sparse (≤ 3 distinct semantic concepts) AND what little
          signal exists overlaps persona — default to aligned, not exploration

  → exploration ONLY when ALL of the following are true:
      (A) SEMANTIC NOVELTY: recent contains specific genre/theme concepts that are
          NOT in persona top-10 (not just "different broad category" but genuinely
          unfamiliar territory for this user)
      (B) REPEATED SIGNAL: the novel concept(s) appear with count >= 2, OR multiple
          distinct novel concepts appear (not just a single count=1 outlier)
      (C) NOT EXPLAINABLE by task_focus or budget_shift:
          - if concentration is high (dom >= 0.45) → use task_focus instead
          - if price_band_shift is the dominant difference → use budget_shift instead
      (D) NOT just generic container drift:
          - drift concepts are NOT umbrella/platform/navigation/format_meta/promo
          - "user watched drama this week instead of action" is aligned, not exploration
          - broad-to-broad shift (drama ↔ comedy ↔ action) is aligned unless
            a genuinely specific non-persona concept (e.g. category:western,
            category:anime) appears with count >= 2

  → Do NOT use exploration for:
      - Concept count=1 outliers in an otherwise persona-consistent window
      - Users whose recent is dominated by platform anchors (prime_video, dvd, blu-ray)
        with only sparse semantic signal → prefer aligned or unknown
      - Simple rotation within known broad categories the persona already covers
      - Cases where the only "new" concepts are generic/umbrella ones

  → Do NOT classify as exploration when semantic_top_dominance >= 0.45 AND the
    concentrated concept is outside the persona — use task_focus instead.

════════════════════════════════════════════════════════════
EXPLORATION GOAL_CONCEPT SELECTION RULE (required when deviation_reason=exploration)
════════════════════════════════════════════════════════════
goal_concepts for exploration must represent the NEW or ADJACENT directions
emerging in recent behavior — NOT the user's usual persona preferences.

Selection steps:
  1. Look at "Semantic drift NOT in persona top-5" in the user prompt (concepts in recent
     but NOT in persona top-5). These are your primary candidates.
  2. Among those, pick 1–3 concepts sorted by recent semantic freq (highest first).
  3. Prefer more specific category/subcategory concepts over broad umbrella concepts.

FORBIDDEN as exploration goal_concepts:
  ✗ any concept that is already in persona top-5  (that is the persona direction, not exploration)
  ✗ UMBRELLA concepts: category:movies_&_tv, category:movies, category:tv, category:television
  ✗ PLATFORM concepts: category:prime_video
  ✗ NAVIGATION concepts: category:featured_categories, category:genre_for_featured_categories,
      category:all_titles, category:all, category:general, category:more_to_explore
  ✗ PROMO_DEAL concepts: category:studio_specials, category:today's_deals,
      category:featured_deals_&_new_releases, category:the_big_dvd_&_blu-ray_blowout,
      category:spotlight_deals, and any publisher deal label
  ✗ PUBLISHER concepts: category:warner_home_video, category:sony_pictures_home_entertainment,
      category:all_sony_pictures_titles, category:all_mgm_titles, category:all_fox_titles,
      category:independently_distributed, and any other studio/distributor label
  ✗ FORMAT_META concepts: category:blu-ray, category:dvd, category:widescreen, category:dts

  NOTE: The candidate_concepts list shown in this prompt has already been pre-filtered
  to remove most non-semantic concepts. Use concepts from that filtered list.

If recent-only semantic concepts are sparse (all count=1, few options):
  → pick the 1–2 highest-frequency ones; do NOT pick every single count=1 concept
    (that creates a too-diffuse goal with no real direction signal)
  → if all eligible concepts have count=1 and there are 3+, just pick the top 2

If recent-only semantic concepts are sparse AND platform/umbrella dominates:
  → look carefully at what little semantic signal exists
  → if no meaningful semantic drift exists outside persona, prefer aligned or unknown
    over a low-confidence exploration guess

Step 4 — Use unknown when:
  → Signals are contradictory or insufficient to classify with reasonable confidence
  → recent_concept_freq has very few semantic entries (< 3 distinct semantic concepts)
    AND browsing is not clearly persona-consistent

════════════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════════════
EXAMPLE A — task_focus:
  persona: drama, action, comedy | recent: horror(8), movies_&_tv(3) | dom=0.67 overlap=0.40
  → task_focus; goal=[category:horror]

EXAMPLE B — task_focus:
  persona: drama, comedy | recent: anime(5), animation(4), movies_&_tv(2) | dom=0.45 overlap=0.50
  → task_focus; goal=[category:anime, category:animation]

EXAMPLE C — exploration (wide drift):
  persona: drama, action, comedy, movies_&_tv, thriller | recent: documentary(2), sci-fi(2), romance(2), western(1)
  drift=[documentary, sci-fi, romance, western] dom=0.25 overlap=0.20
  → exploration; goal=[category:documentary, category:sci-fi]  ← NOT movies_&_tv (persona anchor)

EXAMPLE D — aligned:
  persona: drama, action, thriller | recent: drama(4), action(3), thriller(2) | dom=0.44 overlap=0.75
  → aligned

EXAMPLE E — unknown (platform-dominant + only count=1 drift):
  persona: drama, action, comedy, thriller, mystery | recent: prime_video(10), suspense(1), western(1)
  dom=0.71 (platform, NOT content) overlap=0.20
  → unknown; conf=0.35
  NOTE: suspense(1) and western(1) are single-occurrence outliers — NOT repeated signal.
        Do NOT call this exploration. Platform dominance + count=1 drift = insufficient evidence.
        Use unknown (weak signal) rather than forcing a low-confidence exploration guess.

EXAMPLE F — exploration (genuine specific drift with repeated signal):
  persona: drama, action, comedy | recent: anime(3), animation(2), sci-fi(1) | dom=0.38 overlap=0.10
  drift=[anime, comedy(1)] specific novel: anime(3) animation(2), both count>=2, NOT in persona top-10
  → exploration; goal=[category:anime, category:animation]; conf=0.65
  NOTE: anime and animation appear with count>=2, are specific (non-umbrella), outside persona top-10.

════════════════════════════════════════════════════════════
CONFIDENCE CALIBRATION
════════════════════════════════════════════════════════════
- 0.85–1.0 : very clear signal, strong evidence
- 0.65–0.84: reasonably clear, minor ambiguity
- 0.45–0.64: moderate confidence, some mixed signals
- 0.20–0.44: weak signal, lean toward unknown instead
- Never give high confidence (>0.80) to aligned unless semantic_overlap >= 0.60
- task_focus with mixed signals (e.g. semantic_top_dominance 0.45–0.55, borderline specificity): use 0.50–0.65
- When distinction between aligned and task_focus is borderline, use moderate confidence (0.55–0.65)

════════════════════════════════════════════════════════════
OUTPUT SCHEMA (all fields required)
════════════════════════════════════════════════════════════
{
  "goal_concepts": ["concept_id_1", ...],
  "constraints": {"price_band": [...], "format": [...]},
  "deviation_reason": "aligned|exploration|task_focus|budget_shift|unknown",
  "confidence": 0.0,
  "ttl_steps": 1,
  "persona_alignment_score": 0.0,
  "evidence_item_ids": [],

  "llm_explanation_short": "One or two sentences summarizing why you assigned this deviation_reason.",
  "why_not_aligned": "One sentence on why the recent behavior does NOT match the persona (omit / use \"\" when deviation_reason=aligned).",
  "why_exploration": "One sentence on what makes recent behavior exploratory rather than concentrated task_focus (required only when deviation_reason=exploration; use \"\" otherwise)."
}

════════════════════════════════════════════════════════════
RATIONALE SLOT RULES
════════════════════════════════════════════════════════════
llm_explanation_short   — ALWAYS fill; max 2 sentences; describe what pattern drove the label.
why_not_aligned         — Fill only when deviation_reason != "aligned"; leave "" when aligned.
why_exploration         — Fill only when deviation_reason == "exploration"; leave "" otherwise.

Keep all three slots SHORT (phrase-level evidence, not chain-of-thought).
BAD:  "The user has been watching drama and comedy films for a long time according to their persona,
       but recently their viewing history includes some elements of science fiction..."
GOOD: "Recent window mixes sci-fi and documentary with low overlap (0.22) to persona top concepts."
"""

# ── Per-call user prompt ───────────────────────────────────────────────────────

def _build_user_prompt(ctx: IntentContext, candidate_concepts: list[str], compact: bool = False) -> str:
    """
    Build a grounded, diagnostics-rich user prompt for Movies & TV intent classification.

    Version A (mainline stable):
    - Semantic-only concepts (goal-eligible) as PRIMARY classification signal
    - Raw concept signal shown as AUXILIARY only (do not classify from this)
    - Goal candidates filtered by is_goal_eligible (no PLATFORM/UMBRELLA/NAV/PROMO/PUBLISHER/FORMAT)

    compact=True: cost-reduced mode for 4o pilot runs.
    - Omits AUXILIARY raw freq section (~100 tokens saved)
    - Caps category candidates at 20 instead of 30 (~30 tokens saved)
    - Omits recent_item_ids line
    All decision signals (semantic freq, drift, overlap, dominance) are fully preserved.
    """
    # ── Persona sets ──────────────────────────────────────────────────────────
    persona_top5_set = set(ctx.persona_top_concepts[:5])
    persona_top_set  = set(ctx.persona_top_concepts[:10])
    persona_top5_str = ", ".join(ctx.persona_top_concepts[:5]) or "(none)"
    persona_top_str  = ", ".join(ctx.persona_top_concepts[:10]) or "(none)"

    # Semantic persona (goal-eligible only)
    persona_sem_top = [c for c in ctx.persona_top_concepts if is_goal_eligible(c)]
    persona_sem_top10_set = set(persona_sem_top[:10])
    persona_sem_str  = ", ".join(persona_sem_top[:10]) or "(none)"

    # ── Raw recent freq ───────────────────────────────────────────────────────
    freq_sorted_raw = sorted(ctx.recent_concept_freq.items(), key=lambda x: -x[1])[:15]
    freq_str_raw    = ", ".join(f"{cid}({cnt})" for cid, cnt in freq_sorted_raw)

    # ── Semantic-only recent freq (PRIMARY signal) ────────────────────────────
    # Semantic = goal-eligible category concepts (excludes PLATFORM/UMBRELLA/NAV/PROMO/PUBLISHER/FORMAT)
    sem_freq: dict[str, int] = {
        cid: cnt for cid, cnt in ctx.recent_concept_freq.items()
        if cid.split(":")[0] not in ("format", "price_band")
        and is_goal_eligible(cid)
    }
    sem_freq_sorted = sorted(sem_freq.items(), key=lambda x: -x[1])[:15]
    sem_freq_str    = ", ".join(f"{cid}({cnt})" for cid, cnt in sem_freq_sorted) or "(none)"
    total_sem       = sum(sem_freq.values())

    sem_top_dom  = (sem_freq_sorted[0][1] / total_sem) if sem_freq_sorted and total_sem > 0 else 0.0
    sem_n_dist   = len(sem_freq)
    sem_top2_freq = sum(cnt for _, cnt in sem_freq_sorted[:2]) if len(sem_freq_sorted) >= 2 else (sem_freq_sorted[0][1] if sem_freq_sorted else 0)
    sem_top2_frac = sem_top2_freq / total_sem if total_sem > 0 else 0.0
    sem_top1_cid  = sem_freq_sorted[0][0] if sem_freq_sorted else "(none)"
    sem_top1_in_persona = sem_top1_cid in persona_sem_top10_set

    # Semantic overlap: fraction of semantic recent concepts in semantic persona top-10
    sem_recent_set = set(sem_freq.keys())
    sem_overlap = (
        len(sem_recent_set & persona_sem_top10_set) / len(sem_recent_set)
        if sem_recent_set else 0.0
    )

    # ── Semantic drift (not in persona top-5, goal-eligible) ─────────────────
    sem_drift = sorted(
        [(cid, cnt) for cid, cnt in sem_freq.items() if cid not in persona_top5_set],
        key=lambda x: -x[1],
    )
    sem_drift_str = ", ".join(f"{cid}({cnt})" for cid, cnt in sem_drift[:8]) \
                    or "(none — all semantic concepts are in persona top-5)"

    # ── Semantic concentration signal ─────────────────────────────────────────
    if sem_n_dist == 0:
        sem_concentration = "none — no semantic concepts in recent"
    elif sem_top_dom >= 0.45:
        sem_concentration = f"HIGH ({sem_top_dom:.2f}) — consider task_focus if top-1 is outside persona"
    elif sem_top_dom >= 0.30:
        sem_concentration = f"moderate ({sem_top_dom:.2f})"
    else:
        sem_concentration = f"low ({sem_top_dom:.2f}) — spread across genres"

    # ── Sparse semantic signal flag ───────────────────────────────────────────
    # When raw top-1 is non-semantic (PLATFORM/UMBRELLA/NAV) and semantic is sparse
    from src.intent.concept_roles import get_role
    raw_top1 = freq_sorted_raw[0][0] if freq_sorted_raw else "(none)"
    _non_sem_roles = {"PLATFORM", "UMBRELLA", "NAVIGATION", "PROMO_DEAL", "PUBLISHER", "FORMAT_META"}
    raw_top1_is_nonsem = (
        raw_top1.startswith("category:") and get_role(raw_top1) in _non_sem_roles
    )
    sem_is_sparse = total_sem <= 3 or (sem_n_dist <= 2 and max(sem_freq.values(), default=0) <= 1)
    sparse_flag = raw_top1_is_nonsem and sem_is_sparse

    # ── Price band comparison ──────────────────────────────────────────────────
    r_pb = ctx.recent_dominant_price_band or "unknown"
    p_pb = ctx.persona_dominant_price_band or "unknown"
    pb_diff = (
        r_pb != p_pb
        and r_pb not in ("unknown", "price_band:unknown")
        and p_pb not in ("unknown", "price_band:unknown")
    )
    price_line = f"Recent price_band: {r_pb}  |  Persona price_band: {p_pb}  |  price_band_shift: {pb_diff}"

    # ── Format ────────────────────────────────────────────────────────────────
    r_fmt = ctx.recent_dominant_item_form or "(none)"
    p_fmt = ctx.persona_dominant_item_form or "(none)"

    # ── Candidate concepts (goal-eligible only) ───────────────────────────────
    cat_cap = 20 if compact else 30
    cat_candidates = [c for c in candidate_concepts
                      if c.split(":")[0] == "category" and is_goal_eligible(c)]
    other_candidates = [c for c in candidate_concepts if c.split(":")[0] != "category"]
    cat_str   = ", ".join(cat_candidates[:cat_cap]) or "(none)"
    other_str = ", ".join(other_candidates[:15]) or "(none)"

    lines = ["── USER CONTEXT ──────────────────────────────────────────"]
    if not compact:
        lines.append(f"Recent items ({len(ctx.recent_item_ids)} total, showing last 5): {', '.join(ctx.recent_item_ids[-5:])}")
    lines += [
        "",
        "════ PRIMARY SIGNAL: SEMANTIC CONCEPTS (use this for reason classification) ════",
        f"Semantic recent freq (genre/mood/theme only): {sem_freq_str}",
        f"Semantic distinct count : {sem_n_dist}  |  Semantic top-dominance: {sem_top_dom:.2f}  |  Semantic top-2 frac: {sem_top2_frac:.2f}",
        f"Semantic concentration  : {sem_concentration}",
        f"Semantic top-1 concept  : {sem_top1_cid}  (in semantic persona top-10: {sem_top1_in_persona})",
        f"Semantic overlap ratio  : {sem_overlap:.3f}  (semantic recent ∩ semantic persona top-10)",
        f"Semantic drift concepts : {sem_drift_str}",
        "",
        "── LONG-TERM PERSONA (semantic) ──────────────────────────",
        f"Semantic persona top-10 : {persona_sem_str}",
        f"Full persona top-5      : {persona_top5_str}",
        f"Full persona top-10     : {persona_top_str}",
        price_line,
        f"Recent format: {r_fmt}  |  Persona format: {p_fmt}",
        "",
        "── SEMANTIC ALIGNMENT GUIDANCE ───────────────────────────",
        "  → DEFAULT: choose aligned when uncertain. Exploration needs POSITIVE evidence.",
        "  → aligned:      overlap >= 0.40 OR drift is only broad/generic OR sparse signal",
        "  → task_focus:   sem_top_dom >= 0.45 AND top-1 NOT in semantic persona top-10",
        "  → budget_shift: price_band_shift=True AND price change is dominant difference",
        "  → exploration:  specific novel concept (count>=2, NOT umbrella/platform) AND",
        "                  NOT explainable by task_focus/budget_shift/simple rotation",
        "  → unknown:      truly contradictory or < 3 distinct semantic concepts and",
        "                  not clearly persona-consistent",
    ]

    if sparse_flag:
        lines += [
            "",
            "⚠ SPARSE SEMANTIC SIGNAL WARNING ──────────────────────",
            f"  Raw top-1 concept is non-semantic ({raw_top1}) and semantic signal is weak.",
            f"  Do NOT assign task_focus based on raw dominance alone.",
            f"  Do NOT assign exploration unless a specific novel concept appears with count>=2.",
            f"  Prefer aligned (low confidence) when what little signal exists overlaps persona.",
            f"  Use unknown only if signal is truly contradictory — not merely sparse.",
        ]

    if not compact:
        lines += [
            "",
            "── AUXILIARY: RAW CONCEPT SIGNAL (for context only, do NOT use for classification) ──",
            f"Raw recent freq (all types, top-15): {freq_str_raw}",
        ]

    lines += [
        "",
        "── EXPLORATION GOAL CANDIDATES (use for exploration goal_concepts) ──",
        f"Semantic drift NOT in persona top-5: [{sem_drift_str}]",
        "  → For exploration: prefer concepts from this list",
        "  → Avoid: UMBRELLA, PLATFORM, NAVIGATION, PROMO_DEAL, PUBLISHER, FORMAT_META roles",
        "  → If all drift concepts have count=1 and there are 3+, pick only top 2",
        "",
        "── CANDIDATE CONCEPTS (select goal_concepts ONLY from here) ──",
        f"category concepts: [{cat_str}]",
        f"other concepts:    [{other_str}]",
        "",
        "Classify this user's short-term intent following the decision rules in the system prompt.",
    ]
    return "\n".join(lines)


def _build_candidate_concepts(ctx: IntentContext, intent_cfg: dict) -> list[str]:
    """
    Build the restricted candidate concept set for grounded output.
    Union of: recent concepts + persona top concepts, filtered by signal_concept_types.
    Category concepts are further filtered by concept role hygiene (is_goal_eligible).
    """
    signal_types = set(
        intent_cfg.get("context", {}).get(
            "signal_concept_types", ["category", "format", "price_band"]
        )
    )
    candidates: set[str] = set()

    # All concepts seen in recent window
    for cid in ctx.recent_concept_freq:
        ctype = cid.split(":")[0]
        if ctype in signal_types:
            if ctype == "category" and not is_goal_eligible(cid):
                continue  # exclude non-semantic category concepts
            candidates.add(cid)

    # Top persona concepts
    for cid in ctx.persona_top_concepts:
        ctype = cid.split(":")[0]
        if ctype in signal_types:
            if ctype == "category" and not is_goal_eligible(cid):
                continue
            candidates.add(cid)

    # Sort by: (type_priority, frequency desc) for deterministic ordering
    priority = intent_cfg.get("context", {}).get("goal_concept_priority", {})
    def _sort_key(cid: str) -> tuple:
        ctype = cid.split(":")[0]
        prank = priority.get(ctype, 99)
        freq = ctx.recent_concept_freq.get(cid, 0)
        return (prank, -freq)

    return sorted(candidates, key=_sort_key)


def _validate_and_ground(raw: dict, candidate_concepts: list[str], ctx: IntentContext) -> dict:
    """
    Post-process LLM output:
    - Filter goal_concepts to only those in candidate_concepts
    - Clamp deviation_reason to valid taxonomy
    - Fill missing fields with safe defaults
    """
    candidate_set = set(candidate_concepts)

    # ground goal_concepts
    raw_goals = raw.get("goal_concepts", [])
    if not isinstance(raw_goals, list):
        raw_goals = []
    goal_concepts = [str(c) for c in raw_goals if str(c) in candidate_set]

    # if LLM returned nothing valid, fall back to top candidates by recent freq
    if not goal_concepts and candidate_concepts:
        freq_sorted = sorted(
            candidate_concepts,
            key=lambda c: (-ctx.recent_concept_freq.get(c, 0), c),
        )
        goal_concepts = freq_sorted[:3]

    # clamp deviation_reason
    reason = raw.get("deviation_reason", "unknown")
    if reason not in VALID_REASONS:
        reason = "unknown"

    # clamp confidence
    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # clamp ttl_steps
    try:
        ttl_steps = max(1, min(5, int(raw.get("ttl_steps", 2))))
    except (TypeError, ValueError):
        ttl_steps = 2

    # clamp alignment score
    try:
        alignment = float(raw.get("persona_alignment_score", ctx.overlap_ratio))
        alignment = max(0.0, min(1.0, alignment))
    except (TypeError, ValueError):
        alignment = ctx.overlap_ratio

    # constraints: keep only dict
    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    # evidence_item_ids — always list[str]
    evidence = raw.get("evidence_item_ids", [])
    if not isinstance(evidence, list):
        evidence = ctx.recent_item_ids[-3:]
    evidence = [str(e) for e in evidence]

    return {
        "goal_concepts": goal_concepts,
        "constraints": constraints,
        "deviation_reason": reason,
        "confidence": confidence,
        "ttl_steps": ttl_steps,
        "persona_alignment_score": alignment,
        "evidence_item_ids": evidence,
    }


def interpret_with_llm(
    ctx: IntentContext,
    persona_summary: dict,
    intent_cfg: dict,
    openai_client: Any,
    compact: bool = False,
    candidate_concept_bank: "dict[str, int] | None" = None,
) -> dict:
    """
    3-stage short-term intent interpretation.

    Stage 1 (this function):
        Call LLM to classify deviation_reason and produce raw_llm_goal_concepts.
        LLM output is grounded to the LLM prompt candidate pool (recent ∪ persona top).

    Stage 2 (grounded_selector.validate_and_select_goals):
        Validate raw_llm_goal_concepts against the backbone candidate_concept_bank.
        Applies: activation gate, persona conflict suppression, low-conf near-zero trust.
        Output stored as validated_goal_concepts.
        Only runs if candidate_concept_bank is provided; otherwise falls back to
        goal_concepts (backward-compat mode).

    Stage 3 (signal_builder — downstream):
        signal_builder consumes validated_goal_concepts, not raw LLM goals.

    compact=True: cost-reduced prompt mode for 4o pilot runs.
    candidate_concept_bank: {concept_id: activation_count} built from backbone top-K
        candidates' item concepts (see grounded_selector.build_candidate_concept_bank).
        If None, Stage 2 is skipped.
    """
    # Stage 1 — LLM interpretation
    prompt_candidates = _build_candidate_concepts(ctx, intent_cfg)
    user_prompt = _build_user_prompt(ctx, prompt_candidates, compact=compact)

    model = intent_cfg.get("llm", {}).get("model", "gpt-4o-mini")
    temperature = intent_cfg.get("llm", {}).get("temperature", 0.0)
    max_tokens = intent_cfg.get("llm", {}).get("max_tokens", 512)
    if compact:
        max_tokens = min(max_tokens, 256)

    try:
        response = openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content
        raw = json.loads(raw_text)
        # ground LLM output to prompt candidate pool (Stage 1 grounding)
        result = _validate_and_ground(raw, prompt_candidates, ctx)
        result["source_mode"] = "llm"
        result["llm_raw"] = raw_text

        # ── Rationale slots from LLM output (new in v3) ───────────────────────
        # Sanitize: must be str, max ~300 chars, empty string as fallback.
        def _clean_slot(val: Any, max_len: int = 300) -> str:
            if not val or not isinstance(val, str):
                return ""
            return val.strip()[:max_len]

        result["llm_explanation_short"] = _clean_slot(raw.get("llm_explanation_short"))
        result["why_not_aligned"]       = _clean_slot(raw.get("why_not_aligned"))
        result["why_exploration"]       = _clean_slot(raw.get("why_exploration"))

    except Exception as exc:
        logger.warning("LLM interpretation failed for user=%s: %s", ctx.user_id, exc)
        result = _heuristic_fallback(ctx, prompt_candidates)

    # Preserve Stage 1 output as raw_llm_goals (before Stage 2, before hygiene)
    result["raw_llm_goals"] = list(result.get("goal_concepts", []))

    # ── Hygiene filter on raw_llm_goals ───────────────────────────────────────
    # Remove price_band:*, format:*, and non-semantic category concepts from
    # goal_concepts BEFORE Stage 2.  Diagnostics tracked for Task 2 fields.
    _raw_before_hygiene = list(result["raw_llm_goals"])
    _kept_raw, _removed_raw, _removal_reasons_raw = filter_non_semantic_goals(_raw_before_hygiene)
    result["goal_concepts"]  = _kept_raw       # Stage 1 cleaned goals
    result["raw_llm_goals"]  = _kept_raw       # raw_llm_goals = post-hygiene (pre-Stage2)
    result["removed_non_semantic_goals"] = _removed_raw
    # semantic_signal_absent: no semantic category evidence in recent window at all
    # Use is_semantic_goal (not is_goal_eligible) to exclude price_band/format from count
    _sem_total = sum(
        cnt for cid, cnt in ctx.recent_concept_freq.items()
        if is_semantic_goal(cid)
    )
    result["semantic_signal_absent"]    = (_sem_total == 0)
    result["non_semantic_goal_leakage"] = len(_removed_raw) > 0
    # goal_hygiene_status
    if not _raw_before_hygiene:
        result["goal_hygiene_status"] = "empty_raw_goals"
    elif not _kept_raw and _removed_raw:
        result["goal_hygiene_status"] = "only_non_semantic_raw"
    elif not _kept_raw:
        result["goal_hygiene_status"] = "empty_after_semantic_filter"
    elif len(_kept_raw) == 1 and _kept_raw[0].startswith("category:") and \
            _kept_raw[0] in {"category:drama", "category:comedy", "category:action",
                             "category:movies_&_tv", "category:action_&_adventure"}:
        result["goal_hygiene_status"] = "generic_only_remaining"
    else:
        result["goal_hygiene_status"] = "ok"

    # ── Code-injected provenance fields ───────────────────────────────────────
    # These are never generated by the LLM; populated here for parquet preservation.
    sem_freq = {
        cid: cnt for cid, cnt in ctx.recent_concept_freq.items()
        if is_goal_eligible(cid)
    }
    sem_freq_sorted = sorted(sem_freq.items(), key=lambda x: -x[1])[:8]
    result["evidence_recent_concepts"]  = [f"{c}({n})" for c, n in sem_freq_sorted]
    result["evidence_persona_concepts"] = list(ctx.persona_top_concepts[:8])
    result["raw_model_response_json"]   = result.get("llm_raw")  # alias for clarity
    # pre_grounding_goal_text is set after raw_llm_goals is assigned below (same value)
    result["reason_source"]             = result.get("source_mode", "llm")
    result["llm_prompt_version"]        = LLM_PROMPT_VERSION
    result["schema_version"]            = SCHEMA_VERSION

    # Stage 2 — Grounded goal validation
    if candidate_concept_bank is not None:
        validated, grounding_diag = validate_and_select_goals(
            raw_goal_concepts=result["raw_llm_goals"],
            deviation_reason=result.get("deviation_reason", "unknown"),
            confidence=float(result.get("confidence", 0.35)),
            candidate_concept_bank=candidate_concept_bank,
            persona_top_concepts=ctx.persona_top_concepts,
            ontology_concept_pool=None,   # passive; ontology not used for active scoring
        )
        # Hygiene filter on validated goals (Stage 2 may re-introduce non-semantic
        # concepts from the bank; belt-and-suspenders cleanup)
        _val_kept, _val_removed, _ = filter_non_semantic_goals(validated)
        result["validated_goal_concepts"] = _val_kept
        grounding_diag["hygiene_removed_validated"] = _val_removed
        result["grounding_diagnostics"]   = grounding_diag
        result["has_stage2"]              = True
    else:
        # backward-compat: no bank provided, Stage 2 skipped.
        # WARNING: raw LLM goals are promoted directly to validated_goal_concepts.
        # This means modulation will use ungrounded goals — not the intended production path.
        # Always pass candidate_concept_bank in production runs.
        logger.warning(
            "Stage 2 skipped: no candidate_concept_bank provided for user=%s target_index=%s; "
            "raw LLM goals promoted to validated_goal_concepts (ungrounded path).",
            getattr(ctx, "user_id", "?"),
            getattr(ctx, "target_index", "?"),
        )
        _fb_goals = list(result.get("goal_concepts", []))
        _fb_kept, _, _ = filter_non_semantic_goals(_fb_goals)
        result["validated_goal_concepts"] = _fb_kept
        result["grounding_diagnostics"]   = {"skipped": True, "reason": "no_candidate_bank"}
        result["has_stage2"]              = False

    # pre_grounding_goal_text = raw_llm_goals (set here so Stage 2 result is already known)
    result["pre_grounding_goal_text"] = list(result["raw_llm_goals"])

    return result


def _heuristic_fallback(ctx: IntentContext, candidate_concepts: list[str]) -> dict:
    """
    Minimal heuristic fallback when LLM call fails.
    Uses overlap_ratio and recent concept entropy as proxy signals.
    """
    overlap = ctx.overlap_ratio
    entropy = ctx.recent_concept_entropy

    if overlap >= 0.6 and entropy < 0.5:
        reason = "aligned"
        confidence = 0.6
    elif overlap < 0.4 and entropy > 0.5:
        reason = "exploration"
        confidence = 0.5
    else:
        reason = "unknown"
        confidence = 0.3

    # top candidates by recent freq
    goal_concepts = sorted(
        candidate_concepts,
        key=lambda c: (-ctx.recent_concept_freq.get(c, 0), c),
    )[:3]

    return {
        "goal_concepts": goal_concepts,
        "constraints": {},
        "deviation_reason": reason,
        "confidence": confidence,
        "ttl_steps": 1,
        "persona_alignment_score": overlap,
        "evidence_item_ids": ctx.recent_item_ids[-3:],
        "source_mode": "llm_fallback",
        "llm_raw": None,
        # rationale slots — empty for fallback (LLM not called)
        "llm_explanation_short": "",
        "why_not_aligned": "",
        "why_exploration": "",
        # provenance — populated by interpret_with_llm after this returns
        "evidence_recent_concepts":  [],
        "evidence_persona_concepts": [],
        "raw_model_response_json":   None,
        "pre_grounding_goal_text":   list(goal_concepts),
        "reason_source":                "llm_fallback",
        "has_stage2":                   False,
        "llm_prompt_version":           LLM_PROMPT_VERSION,
        "schema_version":               SCHEMA_VERSION,
        # hygiene fields — fallback goals are already from goal-eligible candidates
        "removed_non_semantic_goals":   [],
        "semantic_signal_absent":       False,
        "non_semantic_goal_leakage":    False,
        "goal_hygiene_status":          "fallback",
    }
