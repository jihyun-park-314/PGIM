"""
llm_interpreter.py
------------------
LLM-based recent-context interpreter for PGIM.

Role: interpret what a user's recent watch/browse activity means relative to their
long-term persona.  NOT a next-item predictor.  NOT a simple intent classifier.

3-stage pipeline:
  Stage 1 — Recent context interpretation (THIS FILE):
             input:  IntentContext (recent window + persona summary + temporal split)
             output: structured interpretation record with 2 channels —
               SCORING: goal_concepts, context_goals, deviation_reason, confidence,
                        contrast_with_persona, temporal_cues, evidence_sources
               AUDIT:   llm_explanation_short, why_not_aligned, why_exploration

  Stage 2 — Grounded goal validation (grounded_selector.py):
             input:  Stage 1 goal_concepts + backbone candidate_concept_bank
             output: validated_goal_concepts (closed concept space; leakage-guarded)

  Stage 3 — Modulation (signal_builder.py):
             only validated_goal_concepts / context_goals enter delta computation.
             AUDIT_ONLY_FIELDS are never read by signal_builder.

Key invariants:
- No open generation: goal_concepts drawn from candidate pool only
- deviation_reason = expansion policy signal (controls modulation strength, not on/off)
- Structured output via JSON mode
- Heuristic fallback on LLM failure

Version: v6 (recent_context_interpreter frame, PR2/PR2.5)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.intent.context_extractor import IntentContext
from src.intent.concept_roles import is_goal_eligible, is_semantic_goal, filter_non_semantic_goals
from src.intent.grounded_selector import validate_and_select_goals
from src.common.schema import AUDIT_ONLY_FIELDS  # noqa: F401 — imported for contract documentation

logger = logging.getLogger(__name__)

# ── Taxonomy ──────────────────────────────────────────────────────────────────
VALID_REASONS = {"aligned", "exploration", "task_focus", "budget_shift", "unknown"}

# ── Versioning — bump when prompt schema or injected fields change ─────────────
LLM_PROMPT_VERSION = "v6_recent_context_interpreter"  # v6: frame shift to "interpret recent behavior" + temporal flow + structured output
SCHEMA_VERSION     = "3.1"                            # bumped: contrast_with_persona / temporal_cues / evidence_sources in LLM schema

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a recent-behavior interpreter for an Amazon Movies & TV recommendation system.

Interpret what a user's recent watch/browse activity means relative to their long-term
persona.  Read the temporal sequence in the user prompt and produce a structured record.

Two output channels — keep them strictly separate:
  SCORING  — structured fields used downstream to adjust recommendations.
             All concept IDs must come from candidate_concepts ONLY.
  AUDIT    — natural-language rationale for human review; never used for scoring.

════════════════════════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════════════════════════
goal_concepts      Select 1–4 from candidate_concepts. Semantic genre/theme only.
                   FORBIDDEN: price_band:*, format:*, category:prime_video,
                   category:movies_&_tv, category:featured_categories, any UMBRELLA/
                   PLATFORM/NAVIGATION/PROMO_DEAL/PUBLISHER/FORMAT_META concept.
                   Return [] if no semantic signal.

constraints        {"price_band": [...], "format": [...]} or {}

deviation_reason   EXPANSION POLICY — controls modulation strength, not on/off.
                   Decide in this order:
                   1. budget_shift  — price band clearly shifted vs persona; else skip.
                   2. task_focus    — sem_top_dom >= 0.45 AND top concept is MORE SPECIFIC
                                      than persona top-3 (a narrowing, not normal consumption).
                                      If top concept IS already persona top-3 → aligned.
                   3. aligned       — DEFAULT when uncertain. Use when overlap >= 0.40 OR
                                      drift is only broad/generic OR signal is sparse.
                   4. exploration   — ALL must be true: (A) specific novel concept NOT in
                                      persona top-10, (B) count >= 2 for that concept,
                                      (C) not explainable by task_focus/budget_shift,
                                      (D) not just UMBRELLA/PLATFORM drift.
                   5. unknown       — contradictory signals or < 3 semantic concepts AND
                                      not clearly persona-consistent.

confidence         Conservative: 0.85+ for very clear cases only.
                   aligned high conf (>0.80) requires overlap >= 0.60.
                   task_focus borderline → 0.50–0.65.

contrast_signal_concepts
                   Persona concepts NOT seen (or clearly reduced) in recent window.
                   Identifies what the user is moving away from.
                   [] if recent is mostly aligned with persona.

temporal_shift_summary
                   ≤ 12 words: dominant concept shift between first/second half.
                   Use the TEMPORAL FLOW section.  "" if no clear shift.

evidence_sources   Include whichever apply:
                     "recent_freq"      — concept frequency was the main signal
                     "temporal_shift"   — first/second half shift detected
                     "persona_contrast" — recent contrasts with persona top concepts

ttl_steps          1–5; shorter for unknown/budget_shift, longer for exploration/task_focus.

════════════════════════════════════════════════════════════
EXPLORATION goal_concepts RULE
════════════════════════════════════════════════════════════
Pick from "Semantic drift NOT in persona top-5" in the user prompt.
Sort by freq; prefer specific over broad; max 3.
If all drift concepts are count=1 and there are 3+, pick only top 2.
FORBIDDEN (same list as goal_concepts above, plus any concept in persona top-5).

════════════════════════════════════════════════════════════
EXAMPLES
════════════════════════════════════════════════════════════
A — task_focus: persona=drama/action/comedy | flow: drama(3)→horror(5) | dom=0.67 overlap=0.40
    goal=[category:horror] contrast=[category:drama,category:action]
    temporal_shift_summary="drama→horror shift in second half"

B — aligned: persona=drama/action/thriller | flow: drama(2),action(2)→drama(2),thriller(2) | overlap=0.75
    goal=[category:drama,category:action] contrast=[]
    temporal_shift_summary="drama/action consistent across both halves"

C — exploration: persona=drama/action/comedy | flow: drama(2)→anime(3),animation(2) | dom=0.38 overlap=0.10
    goal=[category:anime,category:animation] contrast=[category:drama,category:action]
    temporal_shift_summary="drama→anime/animation shift in second half"

════════════════════════════════════════════════════════════
OUTPUT SCHEMA (all fields required, valid JSON only)
════════════════════════════════════════════════════════════
{
  "goal_concepts": [],
  "constraints": {},
  "deviation_reason": "aligned|exploration|task_focus|budget_shift|unknown",
  "confidence": 0.0,
  "ttl_steps": 1,
  "persona_alignment_score": 0.0,
  "evidence_item_ids": [],
  "contrast_signal_concepts": [],
  "temporal_shift_summary": "",
  "evidence_sources": [],
  "llm_explanation_short": "≤2 sentences; phrase-level evidence only.",
  "why_not_aligned": "Fill when deviation_reason != aligned; else \"\".",
  "why_exploration": "Fill when deviation_reason == exploration; else \"\"."
}
"""

# ── Per-call user prompt ───────────────────────────────────────────────────────

def _build_user_prompt(ctx: IntentContext, candidate_concepts: list[str], compact: bool = False) -> str:
    """
    Build a grounded, diagnostics-rich user prompt for recent-behavior interpretation.

    Version B (v6 — recent context interpreter frame):
    - Semantic-only concepts (goal-eligible) as PRIMARY classification signal
    - Temporal flow section added: ordered first-half / second-half concept breakdown
    - Goal candidates filtered by is_goal_eligible
    - Instruction at bottom changed: "interpret" not "classify"

    compact=True: cost-reduced mode.
    - Omits AUXILIARY raw freq section
    - Caps category candidates at 20 instead of 30
    - Omits recent_item_ids line
    - Omits temporal flow detail (only summary line kept)
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

    # ── Temporal flow section ─────────────────────────────────────────────────
    # Built from ctx.recent_concept_temporal_split (added in PR1).
    # Shows first-half vs second-half semantic concept breakdown so the LLM can
    # populate temporal_shift_summary and detect flow changes.
    _ts = ctx.recent_concept_temporal_split
    if _ts and not compact:
        fh: dict = _ts.get("first_half", {})
        sh: dict = _ts.get("second_half", {})
        # Filter to semantic (goal-eligible) concepts only for the prompt
        fh_sem = {c: n for c, n in fh.items() if is_goal_eligible(c)}
        sh_sem = {c: n for c, n in sh.items() if is_goal_eligible(c)}
        fh_str = ", ".join(f"{c}({n})" for c, n in sorted(fh_sem.items(), key=lambda x: -x[1])[:6]) or "(none)"
        sh_str = ", ".join(f"{c}({n})" for c, n in sorted(sh_sem.items(), key=lambda x: -x[1])[:6]) or "(none)"
        fh_top = max(fh_sem, key=fh_sem.get) if fh_sem else None
        sh_top = max(sh_sem, key=sh_sem.get) if sh_sem else None
        flow_shift = fh_top and sh_top and fh_top != sh_top
        flow_summary = (
            f"{fh_top}→{sh_top} shift detected"
            if flow_shift
            else ("stable — same top concept in both halves" if fh_top else "insufficient signal")
        )
        temporal_flow_lines = [
            "",
            "── TEMPORAL FLOW (oldest→newest, split into halves) ──────",
            f"First half  (older items): {fh_str}",
            f"Second half (newer items): {sh_str}",
            f"Flow summary: {flow_summary}",
            "  → Use this to fill temporal_shift_summary in your output.",
            "  → Add \"temporal_shift\" to evidence_sources if a real concept shift is visible.",
        ]
    elif _ts and compact:
        # compact mode: one-line summary only
        fh_sem = {c: n for c, n in _ts.get("first_half", {}).items() if is_goal_eligible(c)}
        sh_sem = {c: n for c, n in _ts.get("second_half", {}).items() if is_goal_eligible(c)}
        fh_top = max(fh_sem, key=fh_sem.get) if fh_sem else None
        sh_top = max(sh_sem, key=sh_sem.get) if sh_sem else None
        flow_summary = (
            f"{fh_top}→{sh_top} shift" if (fh_top and sh_top and fh_top != sh_top)
            else ("stable" if fh_top else "insufficient signal")
        )
        temporal_flow_lines = [
            "",
            f"Temporal flow: {flow_summary}",
        ]
    else:
        temporal_flow_lines = []

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

    lines = ["── RECENT BEHAVIOR ───────────────────────────────────────"]
    if not compact:
        lines.append(f"Recent items (last 5): {', '.join(ctx.recent_item_ids[-5:])}")
    lines += [
        f"Semantic freq      : {sem_freq_str}",
        f"Overlap / dom / top2: {sem_overlap:.2f} / {sem_top_dom:.2f} / {sem_top2_frac:.2f}  |  distinct={sem_n_dist}",
        f"Semantic top-1     : {sem_top1_cid}  (in persona top-10: {sem_top1_in_persona})",
        f"Drift (not top-5)  : {sem_drift_str}",
        "",
        "── LONG-TERM PERSONA ─────────────────────────────────────",
        f"Semantic top-10    : {persona_sem_str}",
        price_line,
    ]

    if sparse_flag:
        lines += [
            f"⚠ SPARSE: raw top-1 is non-semantic ({raw_top1}). "
            "Prefer aligned/unknown; exploration needs count>=2 specific concept.",
        ]

    # Temporal flow section
    if temporal_flow_lines:
        lines += temporal_flow_lines

    if not compact:
        lines += [
            "",
            f"Raw freq (aux): {freq_str_raw}",
        ]

    lines += [
        "",
        f"Drift candidates   : [{sem_drift_str}]",
        "── CANDIDATE CONCEPTS ────────────────────────────────────",
        f"category: [{cat_str}]",
    ]
    if other_str != "(none)":
        lines.append(f"other:    [{other_str}]")
    lines += [
        "",
        "Interpret this user's recent behavior. Fill all OUTPUT SCHEMA fields.",
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

    # ── v6: parse new structured fields from LLM output ──────────────────
    # contrast_signal_concepts: persona concepts the LLM identified as conflicting
    # with recent behavior.  Must be a list of strings; sanitize aggressively.
    raw_csc = raw.get("contrast_signal_concepts", [])
    if not isinstance(raw_csc, list):
        raw_csc = []
    contrast_signal_concepts_llm: list[str] = [str(c) for c in raw_csc if isinstance(c, str)]

    # temporal_shift_summary: short phrase from LLM describing the dominant shift.
    raw_tss = raw.get("temporal_shift_summary", "")
    temporal_shift_summary_llm: str = str(raw_tss).strip()[:120] if raw_tss else ""

    # evidence_sources from LLM: list of allowed tokens only (whitelist).
    _allowed_ev = {"recent_freq", "temporal_shift", "persona_contrast"}
    raw_ev = raw.get("evidence_sources", [])
    if not isinstance(raw_ev, list):
        raw_ev = []
    evidence_sources_llm: list[str] = [s for s in raw_ev if s in _allowed_ev]

    return {
        "goal_concepts": goal_concepts,
        "constraints": constraints,
        "deviation_reason": reason,
        "confidence": confidence,
        "ttl_steps": ttl_steps,
        "persona_alignment_score": alignment,
        "evidence_item_ids": evidence,
        # ── v3/v6: structured interpretation fields ───────────────────────────
        # context_goals: pre-Stage-2 placeholder; overwritten after Stage 2.
        "context_goals": list(goal_concepts),
        # contrast_with_persona: merged from LLM output + Stage 2 upward-pass.
        # Stored as {concept_id: source} where source is "llm" or "grounding".
        # Finalized in interpret_with_llm() after Stage 2.
        "contrast_with_persona": {},
        # contrast_signal_concepts_llm: raw LLM output before merging.
        # Used in interpret_with_llm() to build the final contrast_with_persona dict.
        "_contrast_signal_concepts_llm": contrast_signal_concepts_llm,
        # temporal_cues: finalized in interpret_with_llm() from ctx + LLM summary.
        "temporal_cues": {},
        # temporal_shift_summary_llm: LLM's short phrase about the temporal shift.
        "_temporal_shift_summary_llm": temporal_shift_summary_llm,
        # evidence_sources: finalized in interpret_with_llm() merging LLM + code.
        "evidence_sources": evidence_sources_llm,
        # support_items: evidence item IDs alias.
        "support_items": list(evidence),
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
    Interpret a user's recent behavior in the context of their long-term persona.

    Returns a RecentInterpretationRecord-shaped dict with two channels:
      SCORING  — validated_goal_concepts, context_goals, deviation_reason, confidence,
                 contrast_with_persona, temporal_cues, evidence_sources, ttl_steps
      AUDIT    — llm_explanation_short, why_not_aligned, why_exploration,
                 raw_llm_goals, grounding_diagnostics, llm_raw  (see AUDIT_ONLY_FIELDS)

    Stage 1: LLM interprets recent context → structured record + audit rationale.
    Stage 2: grounded_selector validates goal_concepts against backbone candidates.
             Runs only when candidate_concept_bank is provided (production path).
    Stage 3: signal_builder (downstream) reads scoring channel only.

    compact=True — shorter prompt for cost-sensitive runs (same decision signals,
                   fewer lines, temporal flow one-line summary only).
    candidate_concept_bank — {concept_id: activation_count} from backbone top-K.
                             Pass None to skip Stage 2 (backward-compat; not recommended).
    """
    # Stage 1 — LLM interpretation
    prompt_candidates = _build_candidate_concepts(ctx, intent_cfg)
    user_prompt = _build_user_prompt(ctx, prompt_candidates, compact=compact)

    model = intent_cfg.get("llm", {}).get("model", "gpt-4o-mini")
    temperature = intent_cfg.get("llm", {}).get("temperature", 0.0)
    max_tokens = intent_cfg.get("llm", {}).get("max_tokens", 512)
    if compact:
        max_tokens = min(max_tokens, 256)

    # ── Token instrumentation ─────────────────────────────────────────────────
    # Char-level proxy for token counts (no tiktoken dependency).
    # Populated before the LLM call so we capture prompt cost even on failure.
    # Exact token counts are filled from response.usage when available.
    _sys_chars  = len(_SYSTEM_PROMPT)
    _user_chars = len(user_prompt)
    _token_usage: dict[str, int] = {
        "system_prompt_chars": _sys_chars,
        "user_prompt_chars":   _user_chars,
        "compact": int(compact),
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "token_instrumentation user=%s idx=%s  sys=%d chars  user=%d chars  compact=%s",
            ctx.user_id, ctx.target_index, _sys_chars, _user_chars, compact,
        )

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

        # ── Token counts from response.usage (exact, when available) ──────────
        _usage = getattr(response, "usage", None)
        if _usage is not None:
            _pt = getattr(_usage, "prompt_tokens", None)
            _ct = getattr(_usage, "completion_tokens", None)
            _tt = getattr(_usage, "total_tokens", None)
            if _pt is not None:
                # prompt_tokens = system + user combined (OpenAI API semantics)
                _token_usage["prompt_tokens"]   = _pt
                _token_usage["response_tokens"] = _ct or 0
                _token_usage["total_tokens"]    = _tt or (_pt + (_ct or 0))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "token_usage user=%s idx=%s  prompt=%s response=%s total=%s",
                    ctx.user_id, ctx.target_index,
                    _token_usage.get("prompt_tokens", "?"),
                    _token_usage.get("response_tokens", "?"),
                    _token_usage.get("total_tokens", "?"),
                )

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

    # token_usage is always written (may be chars-only if LLM failed or usage absent)
    result["token_usage"] = _token_usage

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

    # ── v3/v6: finalize structured interpretation fields ──────────────────────
    # Merge LLM-parsed structured output with code-injected ground-truth signals.
    # Priority: code-injected structural signals > LLM output (LLM may be wrong).
    # Final fields go into the scoring channel; LLM rationale stays audit-only.

    # context_goals: finalized alias for validated_goal_concepts (scoring path).
    result["context_goals"] = list(result.get("validated_goal_concepts", result.get("goal_concepts", [])))

    # ── contrast_with_persona ─────────────────────────────────────────────────
    # Merge two sources:
    #   (A) Stage 2 upward-pass: concepts suppressed because they matched persona top-N
    #       → these are structurally grounded; stored as {concept_id: bank_activation}
    #   (B) LLM output (contrast_signal_concepts_llm): persona concepts the LLM flagged
    #       → stored as {concept_id: "llm"} to distinguish from grounding source
    _gd = result.get("grounding_diagnostics", {})
    _suppressed = _gd.get("suppressed_by_persona", [])
    _grounding_contrast = _gd.get("contrast_signal", {})   # {concept_id: bank_activation}

    _llm_csc = result.pop("_contrast_signal_concepts_llm", [])
    _contrast: dict = {}
    for c, act in _grounding_contrast.items():
        _contrast[c] = act        # int: bank activation count (grounding-sourced)
    for c in _llm_csc:
        if c not in _contrast:
            _contrast[c] = "llm"  # str sentinel: LLM-sourced, not yet grounding-verified

    result["contrast_with_persona"] = _contrast

    # evidence_sources: start from LLM output (already whitelist-filtered),
    # then add code-injected signals that the LLM might have missed.
    _ev: list[str] = list(result.get("evidence_sources", []))

    if _suppressed and "persona_contrast" not in _ev:
        _ev.append("persona_contrast")

    # ── temporal_cues ─────────────────────────────────────────────────────────
    # Built from IntentContext.recent_concept_temporal_split (authoritative).
    # LLM temporal_shift_summary is stored inside as an audit-friendly phrase.
    _ts = ctx.recent_concept_temporal_split
    _llm_tss = result.pop("_temporal_shift_summary_llm", "")
    if _ts:
        fh: dict = _ts.get("first_half", {})
        sh: dict = _ts.get("second_half", {})
        fh_sem = {c: n for c, n in fh.items() if is_goal_eligible(c)}
        sh_sem = {c: n for c, n in sh.items() if is_goal_eligible(c)}
        fh_top = max(fh_sem, key=fh_sem.get) if fh_sem else None
        sh_top = max(sh_sem, key=sh_sem.get) if sh_sem else None
        shift_detected = bool(fh_top and sh_top and fh_top != sh_top)
        result["temporal_cues"] = {
            "shift_detected": shift_detected,
            "first_half_dominant": fh_top,
            "second_half_dominant": sh_top,
            "first_half_freq": fh_sem,
            "second_half_freq": sh_sem,
            # LLM's natural-language interpretation of the shift.
            # Kept here for offline audit; not used for scoring.
            "llm_shift_summary": _llm_tss,
        }
        if shift_detected and "temporal_shift" not in _ev:
            _ev.append("temporal_shift")
    else:
        result["temporal_cues"] = {"llm_shift_summary": _llm_tss} if _llm_tss else {}

    # evidence_sources baseline: recent_freq is always present when semantic signal exists.
    _sem_total_for_evidence = sum(
        cnt for cid, cnt in ctx.recent_concept_freq.items()
        if is_semantic_goal(cid)
    )
    if _sem_total_for_evidence > 0 and "recent_freq" not in _ev:
        _ev.insert(0, "recent_freq")

    result["evidence_sources"] = _ev

    # support_items: finalize from evidence_item_ids (scoring-safe alias).
    result["support_items"] = list(result.get("evidence_item_ids", []))

    # ── instrumentation ───────────────────────────────────────────────────────
    # Logged at DEBUG so it has zero runtime cost in production but is available
    # for offline analysis via --log-level DEBUG.
    if logger.isEnabledFor(logging.DEBUG):
        _reason   = result.get("deviation_reason", "?")
        _ev       = result.get("evidence_sources", [])
        _ct       = result.get("contrast_with_persona", {})
        _tc       = result.get("temporal_cues", {})
        _n_val    = len(result.get("validated_goal_concepts", []))
        _n_raw    = len(result.get("raw_llm_goals", []))
        _n_csc    = len([v for v in _ct.values() if v == "llm"])    # LLM-only contrast
        _n_grd    = len([v for v in _ct.values() if v != "llm"])    # grounding-verified contrast
        logger.debug(
            "interpret user=%s idx=%s  reason=%s conf=%.2f  "
            "raw_goals=%d validated=%d  contrast(llm=%d,grnd=%d)  "
            "shift=%s  evidence=%s",
            ctx.user_id, ctx.target_index,
            _reason, float(result.get("confidence", 0.0)),
            _n_raw, _n_val,
            _n_csc, _n_grd,
            _tc.get("shift_detected", False),
            _ev,
        )

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
        # v3/v6: structured interpretation fields — empty for fallback path.
        # interpret_with_llm() finalizes these via merge logic after this returns.
        "context_goals":                    list(goal_concepts),
        "contrast_with_persona":            {},
        "_contrast_signal_concepts_llm":    [],   # no LLM call → no LLM contrast output
        "temporal_cues":                    {},
        "_temporal_shift_summary_llm":      "",   # no LLM call → no LLM shift summary
        "evidence_sources":                 [],
        "support_items":                    list(ctx.recent_item_ids[-3:]),
        # token_usage: empty for fallback (no LLM call); filled by interpret_with_llm
        # after this returns using the char-level proxy captured before the call.
        "token_usage":                      {},
    }
