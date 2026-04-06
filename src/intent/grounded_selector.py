"""
grounded_selector.py
--------------------
Stage 2: Grounded goal selection / validation for PGIM short-term intent.

3-stage short-term architecture:
  Stage 1 (llm_interpreter):  LLM interprets recent deviation →
                               deviation_reason, confidence, raw_llm_goals
  Stage 2 (THIS MODULE):      Grounded goal selection / validation
                               - Input:  Stage 1 output + candidate_concept_bank
                                         + ontology pool (optional) + persona prior
                               - Output: validated_goal_concepts (closed concept space)
  Stage 3 (signal_builder):   Modulation — only validated_goal_concepts enter.
                               raw_llm_goals → debug_info only.

Key invariants
--------------
- NO free generation: every validated goal must exist in the candidate concept bank.
- GT items are NEVER used inside validate_and_select_goals() — leakage is hard-blocked.
  GT concepts may be passed to compute_grounding_diagnostics() for offline eval ONLY.
- All policy thresholds are config-driven; no hard-coded behaviour without a config key.
- Persona conflict suppression is optional and config-controlled (on/off + strength).
- Low-confidence / unknown near-zero trust is optional and config-controlled.

Candidate concept bank
----------------------
Built from backbone top-K candidates' item_concepts.
Stored as {concept_id: activation_count} where
  activation_count = number of backbone candidates carrying this concept.
This is distinct from the LLM prompt candidate pool (recent ∪ persona top).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.intent.concept_roles import is_goal_eligible, get_role

logger = logging.getLogger(__name__)

# ── Concept IDF cache (loaded once per process, keyed by idf_path) ────────────
_concept_idf_cache: dict[str, dict[str, float]] = {}


def _load_concept_idf(idf_path: str) -> dict[str, float]:
    """
    Load concept IDF weights from a parquet file.
    Expected schema: concept_id (str), idf_weight (float).
    Cached after first load.  Returns empty dict if path not found.
    """
    if idf_path not in _concept_idf_cache:
        from pathlib import Path
        p = Path(idf_path)
        if p.exists():
            import pandas as _pd
            df = _pd.read_parquet(p)
            _concept_idf_cache[idf_path] = dict(
                zip(df["concept_id"], df["idf_weight"].astype(float))
            )
            logger.info(
                "Loaded concept IDF for selector: %d entries from %s",
                len(_concept_idf_cache[idf_path]), p,
            )
        else:
            logger.warning(
                "concept IDF path not found: %s — scoring_mode count_x_idf "
                "will fall back to activation_count only.", idf_path,
            )
            _concept_idf_cache[idf_path] = {}
    return _concept_idf_cache[idf_path]


def _selector_score(
    concept_id: str,
    activation_count: int,
    scoring_mode: str,
    idf: dict[str, float],
    idf_floor: float,
) -> float:
    """
    Compute per-concept selection score for Step 6 sorting.

    scoring_mode:
      baseline_count  — activation_count only  (original behaviour)
      count_x_idf     — activation_count × idf_weight
                        IDF amplifies specific/rare concepts and down-weights
                        high-frequency generic concepts (drama, comedy, etc.)
    """
    if scoring_mode == "count_x_idf":
        w = idf.get(concept_id, idf_floor)
        return activation_count * w
    # default: baseline_count
    return float(activation_count)

# ── Default config (all overridable via selector_cfg dict) ────────────────────

_DEFAULT_SELECTOR_CFG: dict = {
    # Minimum backbone-candidate activation to keep a goal concept
    "min_activation": 1,

    # Maximum validated goals per reason
    "max_goals_per_reason": {
        "aligned":      3,
        "task_focus":   2,
        "exploration":  3,
        "budget_shift": 2,
        "unknown":      1,
    },

    # Roles that are always excluded (hard hygiene, not research-adjustable)
    "excluded_roles": [
        "PLATFORM", "UMBRELLA", "NAVIGATION", "PROMO_DEAL",
        "PUBLISHER", "FORMAT_META", "COLLECTION",
    ],

    # Persona conflict suppression
    # enabled:   whether to suppress concepts that match persona top-N for exploration
    # top_n:     persona top-N to compare against
    # reasons:   which reasons trigger suppression (default: exploration only)
    # fallback:  if suppression empties the list, keep top-activation concept anyway
    "persona_conflict": {
        "enabled": True,
        "top_n": 5,
        "reasons": ["exploration"],
        "fallback_keep_top1": True,
    },

    # Low-confidence / unknown near-zero trust
    # enabled:         whether to apply the cap
    # conf_threshold:  below this confidence → apply cap
    # also_unknown:    always apply cap for reason=unknown regardless of confidence
    # max_concepts:    maximum concepts when low-conf (default 1)
    # min_activation_required: minimum bank count for the surviving concept
    "low_conf_trust": {
        "enabled": True,
        "conf_threshold": 0.45,
        "also_unknown": True,
        "max_concepts": 1,
        "min_activation_required": 2,
    },

    # Scoring mode for Step 6 sort (and Step 5 low-conf sort)
    # baseline_count  — original: sort by activation_count only
    # count_x_idf     — sort by activation_count × idf_weight
    #                   promotes specific/rare concepts over generic high-frequency ones
    # idf_path        — path to concept IDF parquet (concept_id, idf_weight)
    # idf_floor       — fallback idf_weight for unknown concepts (default 0.1)
    "scoring": {
        "mode": "baseline_count",
        "idf_path": "",
        "idf_floor": 0.1,
    },
}


def _merge_cfg(base: dict, override: Optional[dict]) -> dict:
    """Shallow-merge override into base, returning a new dict."""
    if not override:
        return dict(base)
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


# ── Candidate Concept Bank ────────────────────────────────────────────────────

def build_candidate_concept_bank(
    candidate_item_ids: list[str],
    item_concepts: dict[str, list[str]],
    goal_eligible_only: bool = True,
) -> dict[str, int]:
    """
    Build per-ranking candidate concept activation bank.

    Counts how many backbone top-K candidates carry each concept.
    This forms the "closed concept space" for Stage 2 validation.

    Args:
        candidate_item_ids: backbone top-K item IDs for this (user, target_index).
                            Must NOT include GT items — caller responsibility.
        item_concepts:      item_id -> list[concept_id]
        goal_eligible_only: if True, filter to is_goal_eligible concepts only

    Returns:
        {concept_id: activation_count}
        activation_count = number of candidates that carry this concept

    NOTE: GT items must never be included in candidate_item_ids when building
          a bank that will be used for validate_and_select_goals().
          The bank is built from backbone candidates only.
    """
    bank: dict[str, int] = {}
    for item_id in candidate_item_ids:
        for cid in item_concepts.get(item_id, []):
            if goal_eligible_only and not is_goal_eligible(cid):
                continue
            bank[cid] = bank.get(cid, 0) + 1
    return bank


# ── Core Validator ────────────────────────────────────────────────────────────

def validate_and_select_goals(
    raw_goal_concepts: list[str],
    deviation_reason: str,
    confidence: float,
    candidate_concept_bank: dict[str, int],
    persona_top_concepts: list[str],
    ontology_concept_pool: Optional[set[str]] = None,
    selector_cfg: Optional[dict] = None,
    # LEAKAGE GUARD: GT must never enter this function
    # gt_item_concepts is intentionally absent from this signature.
    # If you need GT-based diagnostics, use compute_grounding_diagnostics() separately.
) -> tuple[list[str], dict]:
    """
    Stage 2: Validate and select grounded goal concepts.

    Pipeline (each step config-gated):
      1. Hard role exclusion    — PLATFORM/UMBRELLA/NAV/PROMO/PUBLISHER/FORMAT/COLLECTION
      2. Activation gate        — concept must appear in >= min_activation candidates
      3. Persona conflict       — optional suppression for exploration (config: persona_conflict)
      4. Ontology pool          — optional constraint (graceful fallback if pool filters all)
      5. Low-conf/unknown trust — optional near-zero trust cap (config: low_conf_trust)
      6. Sort + limit           — by scoring_mode (baseline_count or count_x_idf),
                                  then max_goals_per_reason

    LEAKAGE GUARD:
      - This function has no access to GT items.
      - candidate_concept_bank must be built from backbone candidates ONLY.
      - GT-based diagnostics belong in compute_grounding_diagnostics(), called separately.

    Args:
        raw_goal_concepts:      Stage 1 LLM output (grounded to LLM prompt candidates,
                                NOT yet grounded to backbone activation).
        deviation_reason:       one of the 5 taxonomy values.
        confidence:             float [0, 1].
        candidate_concept_bank: {concept_id: activation_count} — backbone candidates only.
        persona_top_concepts:   sorted list of persona concept_ids (highest weight first).
        ontology_concept_pool:  optional set of ontology-valid concept_ids.
        selector_cfg:           optional config overrides (merged with _DEFAULT_SELECTOR_CFG).

    Returns:
        (validated_goal_concepts, validation_diagnostics)
    """
    cfg = _merge_cfg(_DEFAULT_SELECTOR_CFG, selector_cfg)

    diag: dict = {
        "raw_goals": list(raw_goal_concepts),
        "n_raw": len(raw_goal_concepts),
        "reason": deviation_reason,
        "confidence": round(confidence, 3),
        "steps": [],
        "cfg_snapshot": {
            "min_activation": cfg["min_activation"],
            "persona_conflict_enabled": cfg["persona_conflict"]["enabled"],
            "low_conf_trust_enabled": cfg["low_conf_trust"]["enabled"],
            "scoring_mode": cfg["scoring"]["mode"],
        },
    }

    if not raw_goal_concepts:
        diag["steps"].append("empty_raw_goals")
        diag["n_validated"] = 0
        diag["validated_goals"] = []
        return [], diag

    # ── Load IDF (used in Step 5 and Step 6 if scoring_mode=count_x_idf) ─────
    scoring_cfg  = cfg["scoring"]
    scoring_mode = scoring_cfg.get("mode", "baseline_count")
    idf_floor    = float(scoring_cfg.get("idf_floor", 0.1))
    idf: dict[str, float] = {}
    if scoring_mode == "count_x_idf":
        idf_path = scoring_cfg.get("idf_path", "")
        if idf_path:
            idf = _load_concept_idf(idf_path)
        else:
            raise ValueError(
                "scoring_mode='count_x_idf' requires idf_path to be set in "
                "selector_cfg['scoring']['idf_path'], but it is empty. "
                "Either provide an idf_path or use scoring_mode='baseline_count'."
            )

    def _score(c: str) -> float:
        return _selector_score(
            c, candidate_concept_bank.get(c, 0), scoring_mode, idf, idf_floor
        )

    excluded_roles = set(cfg["excluded_roles"])

    # ── Step 1: Hard role exclusion ───────────────────────────────────────
    after_role = [c for c in raw_goal_concepts if get_role(c) not in excluded_roles]
    dropped_role = [c for c in raw_goal_concepts if c not in set(after_role)]
    if dropped_role:
        diag["steps"].append(f"role_exclusion: dropped {dropped_role}")

    if not after_role:
        diag["steps"].append("all_dropped_by_role_exclusion")
        diag["n_validated"] = 0
        diag["validated_goals"] = []
        return [], diag

    # ── Step 2: Activation gate ───────────────────────────────────────────
    # min_activation is clamped to >= 1.
    # Setting 0 would allow bank-absent concepts through, breaking the hard-gate guarantee
    # that "every validated goal exists in the candidate concept bank".
    min_act = max(1, int(cfg["min_activation"]))
    after_activation = [c for c in after_role if candidate_concept_bank.get(c, 0) >= min_act]
    dropped_act = [c for c in after_role if c not in set(after_activation)]
    if dropped_act:
        diag["steps"].append(
            f"activation_gate(min={min_act}): dropped {dropped_act} "
            f"[counts: {[candidate_concept_bank.get(c, 0) for c in dropped_act]}]"
        )

    if not after_activation:
        diag["steps"].append("all_dropped_by_activation_gate")
        diag["n_validated"] = 0
        diag["validated_goals"] = []
        return [], diag

    # ── Step 3: Persona conflict suppression (config-gated) ───────────────
    pc_cfg = cfg["persona_conflict"]
    after_persona = after_activation
    # structured upward-pass: which concepts were suppressed because they matched
    # the persona top-N, and their candidate bank activation counts.
    # Consumed by llm_interpreter to populate contrast_with_persona.
    _suppressed_by_persona: list[str] = []

    if pc_cfg["enabled"] and deviation_reason in pc_cfg.get("reasons", ["exploration"]):
        top_n = int(pc_cfg["top_n"])
        persona_top_n_set = set(persona_top_concepts[:top_n])
        suppressed = [c for c in after_activation if c in persona_top_n_set]
        survivors  = [c for c in after_activation if c not in persona_top_n_set]

        if suppressed:
            diag["steps"].append(
                f"persona_conflict_suppression(top_{top_n}, reason={deviation_reason}): "
                f"suppressed {suppressed}"
            )
            _suppressed_by_persona = suppressed

        if not survivors and pc_cfg.get("fallback_keep_top1", True):
            # Keep highest-activation concept even if it's in persona top-N
            best = max(after_activation, key=lambda c: candidate_concept_bank.get(c, 0))
            survivors = [best]
            diag["steps"].append(
                f"persona_conflict_fallback: kept {best} "
                f"(all {len(suppressed)} suppressed, fallback_keep_top1=True)"
            )
        after_persona = survivors

    # Structured upward-pass fields (consumed by llm_interpreter, not by scoring logic).
    # suppressed_by_persona: concepts dropped because they matched the persona top-N.
    # contrast_signal:       their candidate bank activation counts (proxy for how
    #                        "present" they are in the current candidate space).
    diag["suppressed_by_persona"] = _suppressed_by_persona
    diag["contrast_signal"] = {
        c: candidate_concept_bank.get(c, 0) for c in _suppressed_by_persona
    }

    if not after_persona:
        diag["steps"].append("all_dropped_by_persona_conflict (no fallback)")
        diag["n_validated"] = 0
        diag["validated_goals"] = []
        return [], diag

    # ── Step 4: Ontology pool constraint (optional, graceful fallback) ────
    after_ontology = after_persona
    if ontology_concept_pool:
        filtered = [c for c in after_persona if c in ontology_concept_pool]
        if filtered:
            dropped_ont = [c for c in after_persona if c not in set(filtered)]
            if dropped_ont:
                diag["steps"].append(f"ontology_pool_filter: dropped {dropped_ont}")
            after_ontology = filtered
        else:
            diag["steps"].append("ontology_pool_fallback: pool filtered all → reverted")

    # ── Step 5: Low-confidence / unknown near-zero trust (config-gated) ───
    lct_cfg = cfg["low_conf_trust"]
    after_conf = after_ontology

    if lct_cfg["enabled"]:
        low_conf = confidence < float(lct_cfg["conf_threshold"])
        is_unknown = lct_cfg.get("also_unknown", True) and deviation_reason == "unknown"
        if low_conf or is_unknown:
            max_c = int(lct_cfg["max_concepts"])
            min_act_req = int(lct_cfg["min_activation_required"])
            # keep only top-score concepts up to max_concepts, with activation floor check
            candidates_sorted = sorted(
                after_ontology,
                key=lambda c: (-_score(c), c),
            )
            kept = [
                c for c in candidates_sorted[:max_c]
                if candidate_concept_bank.get(c, 0) >= min_act_req
            ]
            diag["steps"].append(
                f"low_conf_trust(reason={deviation_reason}, conf={confidence:.2f}, "
                f"low_conf={low_conf}, is_unknown={is_unknown}): "
                f"cap={max_c}, min_act={min_act_req}, kept={kept}"
            )
            after_conf = kept

    # ── Step 6: Sort by score, limit to max_goals_per_reason ─────────────
    # scoring_mode=baseline_count → sort by activation_count (original behaviour)
    # scoring_mode=count_x_idf   → sort by activation_count × idf_weight
    #   IDF amplifies rare/specific concepts; down-weights generic high-frequency ones.
    max_goals = int(cfg["max_goals_per_reason"].get(deviation_reason, 3))
    validated = sorted(
        after_conf,
        key=lambda c: (-_score(c), c),
    )[:max_goals]

    diag["n_validated"] = len(validated)
    diag["validated_goals"] = validated
    diag["activation_counts"] = {c: candidate_concept_bank.get(c, 0) for c in validated}
    diag["selection_scores"]  = {c: round(_score(c), 4) for c in validated}
    diag["scoring_mode"]      = scoring_mode

    return validated, diag


# ── Grounding Diagnostics (eval-only, GT-aware) ───────────────────────────────

def compute_grounding_diagnostics(
    raw_goal_concepts: list[str],
    validated_goal_concepts: list[str],
    candidate_concept_bank: dict[str, int],
    gt_item_concepts: Optional[list[str]] = None,
) -> dict:
    """
    Compute before/after grounding diagnostics.

    This function is for OFFLINE EVALUATION ONLY.
    GT concepts may be passed here — this is safe because this function does
    NOT feed its output back into validate_and_select_goals() or modulation.

    Measures:
      - goal_to_candidate_match:    fraction of goals in candidate bank (before/after)
      - any_candidate_activated:    whether at least 1 goal has activation > 0 (before/after)
      - avg_activated_candidates:   mean activation count for matched goals (before/after)
      - activation_mass:            sum of activation counts for all goals (before/after)
      - gt_match:                   fraction of goals matching GT concepts (before/after)
                                    only computed if gt_item_concepts provided

    Args:
        raw_goal_concepts:       Stage 1 LLM output (before Stage 2).
        validated_goal_concepts: Stage 2 output (after validation).
        candidate_concept_bank:  {concept_id: activation_count}.
        gt_item_concepts:        concept_ids from GT next item(s), for eval only.
                                 MUST NOT be used in validation logic.

    Returns:
        dict with scalar before/after diagnostic values.
    """
    diag: dict = {}

    # ── Goal-to-candidate match ────────────────────────────────────────
    def _match_stats(goals: list[str]) -> dict:
        if not goals:
            return {
                "match_rate": 0.0,
                "any_activated": False,
                "avg_activation": 0.0,
                "activation_mass": 0,
                "in_bank": [],
            }
        in_bank = [c for c in goals if candidate_concept_bank.get(c, 0) > 0]
        activations = [candidate_concept_bank.get(c, 0) for c in in_bank]
        return {
            "match_rate":       round(len(in_bank) / len(goals), 4),
            "any_activated":    len(in_bank) > 0,
            "avg_activation":   round(sum(activations) / len(in_bank), 4) if in_bank else 0.0,
            "activation_mass":  sum(candidate_concept_bank.get(c, 0) for c in goals),
            "in_bank":          in_bank,
        }

    raw_stats = _match_stats(raw_goal_concepts)
    val_stats = _match_stats(validated_goal_concepts)

    diag["candidate_match_before"]      = raw_stats["match_rate"]
    diag["candidate_match_after"]       = val_stats["match_rate"]
    diag["any_activated_before"]        = raw_stats["any_activated"]
    diag["any_activated_after"]         = val_stats["any_activated"]
    diag["avg_activated_cands_before"]  = raw_stats["avg_activation"]
    diag["avg_activated_cands_after"]   = val_stats["avg_activation"]
    diag["raw_activation_mass"]         = raw_stats["activation_mass"]
    diag["val_activation_mass"]         = val_stats["activation_mass"]
    diag["activation_mass_delta"]       = val_stats["activation_mass"] - raw_stats["activation_mass"]
    diag["raw_in_bank"]                 = raw_stats["in_bank"]
    diag["val_in_bank"]                 = val_stats["in_bank"]

    # ── GT match (eval-only, safe to receive GT here) ─────────────────
    # NOTE: gt_item_concepts is used here for offline evaluation metrics ONLY.
    # It does not flow back to validation logic.
    if gt_item_concepts is not None:
        gt_set = set(gt_item_concepts)
        raw_gt  = [c for c in raw_goal_concepts       if c in gt_set]
        val_gt  = [c for c in validated_goal_concepts  if c in gt_set]
        diag["gt_match_before"] = round(len(raw_gt) / max(len(raw_goal_concepts), 1), 4)
        diag["gt_match_after"]  = round(len(val_gt) / max(len(validated_goal_concepts), 1)
                                        if validated_goal_concepts else 0.0, 4)
        diag["raw_gt_match"]    = raw_gt
        diag["val_gt_match"]    = val_gt

    return diag
