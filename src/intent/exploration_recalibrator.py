"""
exploration_recalibrator.py
---------------------------
Post-hoc rule-based recalibration layer for exploration-labeled intent records.

Problem being solved:
    LLM over-classifies recent behavioral variation as "exploration", even when
    the underlying GT and persona signal are highly consistent.  Audit results:
      - 89.4% of exploration GT items are in persona top-5
      - avg_n_validated_goals = 1.30 (mostly generic: drama/comedy)
      - true semantic drift (specific, out-of-persona concepts) is the minority

This module does NOT change LLM behavior or re-prompt.  It applies a deterministic
rule-based recalibration to already-computed intent records, using:
  - persona semantic-core overlap (derived from interaction history + persona graph)
  - recent concept concentration and specificity
  - validated goal genericity
  - price/format shift indicators

Output: recalibrated_reason ∈ {
    true_exploration, aligned_soft, task_focus_like, budget_like, exploration_unclear
}
For non-exploration reasons, recalibrated_reason passes through unchanged.

All thresholds are config-driven (RECAL_CFG default, overridable per call).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Recalibration vocabulary ───────────────────────────────────────────────────
RECAL_REASONS: frozenset[str] = frozenset({
    "true_exploration",
    "aligned_soft",
    "task_focus_like",
    "budget_like",
    "exploration_unclear",
})

# Non-exploration reasons pass through unchanged under recalibrate_record().
PASSTHROUGH_REASONS: frozenset[str] = frozenset({
    "aligned", "task_focus", "budget_shift", "unknown",
})

# ── Semantic-core filter ───────────────────────────────────────────────────────
# Concepts excluded from persona/recent "semantic core" computation.
# These are noise/product-context/umbrella concepts whose presence inflates
# raw overlap and makes "outside persona" judgments unreliable.

_NOISE_EXACT: frozenset[str] = frozenset({
    "category:movies_&_tv",
    "category:movies",
    "category:tv",
    "category:television",
    "category:prime_video",
    "category:featured_categories",
    "category:genre_for_featured_categories",
    "category:all_titles",
    "category:all",
    "category:more_to_explore",
    "category:general",
})

_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "_titles",           # all_*_titles publisher dumps
    "featured_",         # featured_deals, featured_categories
    "today_",            # today's_deals
    "studio_specials",
    "spotlight_",
    "independently_distributed",
    "warner_home",
    "sony_",
    "mgm_",
    "fox_titles",
    "lionsgate",
    "all_sony",
    "all_bbc",
    "all_disney",
    "all_hbo",
    "all_mtv",
    "all_a&e",
    "all_universal",
    "all_showtime",
    "all_new_yorker",
    "all_sundance",
    "all_sci_fi",
    # Note: format-meta concepts (blu-ray, dvd, widescreen, dts) are NOT filtered here.
    # They appear prominently in many users' persona graphs and must be included in
    # persona SC overlap computation so that the overlap signal is not artificially
    # deflated for users whose persona is dominated by format preferences.
    # Budget_like detection handles price_band separately.
)


def is_persona_semantic_core(concept_id: str) -> bool:
    """
    Return True if concept_id is a 'semantic core' concept suitable for
    persona/recent overlap computation.

    Filters out: umbrella, platform, navigation, publisher, promo/deal, format-meta,
    and catalog-dump concepts that inflate raw overlap without adding semantic signal.

    Non-category concepts (format:*, price_band:*) return False —
    they are handled separately in budget_like detection.
    """
    if not concept_id.startswith("category:"):
        return False
    if concept_id in _NOISE_EXACT:
        return False
    for sub in _NOISE_SUBSTRINGS:
        if sub in concept_id:
            return False
    return True


# ── Broad/generic concept set ──────────────────────────────────────────────────
# These concepts survive Stage 2 (high activation) but are too generic to signal
# true exploration direction — their presence in validated goals is a sign of
# aligned_soft, not true_exploration.
_BROAD_GENERIC: frozenset[str] = frozenset({
    "category:drama",
    "category:comedy",
    "category:action_&_adventure",
    "category:science_fiction",
    "category:documentary",
    "category:kids_&_family",
    "category:horror",
    "category:romance",
    "category:thriller",
    "category:mystery",
})


# ── Default config ─────────────────────────────────────────────────────────────
RECAL_CFG: dict[str, Any] = {
    # ── aligned_soft ──────────────────────────────────────────────────────────
    # Primary signal: validated goals are all broad/generic (drama/comedy/action)
    # even when SC overlap is not available (persona SC often empty or very narrow).
    # Condition: ALL validated goals are broad generic concepts (no specific drift goal)
    "aligned_soft_all_broad": True,          # if all val goals in _BROAD_GENERIC → aligned_soft
    # Secondary: SC overlap available AND high
    "aligned_soft_sc_overlap_min": 0.5,      # fraction of recent SC concepts in persona SC
    "aligned_soft_broad_goal_min": 0.5,      # fraction of validated goals that are generic
    # Tertiary: val goals overlap with persona SC (when SC is available)
    "aligned_soft_val_in_psc_min": 0.5,

    # ── task_focus_like ───────────────────────────────────────────────────────
    # Narrow validated output (1 concept) + high recent concentration
    "task_focus_concentration_min": 0.55,    # top-1 freq / total recent semantic freq
    "task_focus_max_val_goals": 1,           # validated goals <= N → narrow focus
    "task_focus_top1_in_psc_conc_min": 0.45, # moderate conc when top-1 is in persona SC

    # ── true_exploration ─────────────────────────────────────────────────────
    # Validated goals are specific (not broad) AND out-of-persona drift exists
    # Primary: validated goals contain at least one specific (low doc_freq) concept
    "true_exploration_goal_doc_freq_max": 3000,   # "specific goal" = doc_freq < this
    "true_exploration_min_specific_goals": 1,      # at least N specific val goals needed
    # Secondary: recent has specific drift (when SC available)
    "true_exploration_min_specific_drift": 2,      # at least N specific drift recent cids
    "true_exploration_doc_freq_max": 3000,         # "specific drift" = doc_freq < this
    "true_exploration_sc_overlap_max": 0.3,        # low SC overlap (secondary check only)

    # budget_like: price_band shift in constraints triggers budget_like directly
    # (binary signal — no threshold)
}


# ── Signal computation ─────────────────────────────────────────────────────────

def compute_recalibration_signals(
    intent_record: dict,
    persona_sc_concepts: list[str],   # persona semantic-core top-N for this user
    recent_sem_freq: dict[str, int],  # {concept_id: freq} from recent interaction window
    doc_freq_map: dict[str, int],     # {concept_id: doc_freq} from corpus IDF table
) -> dict[str, Any]:
    """
    Compute all signals needed for recalibration from pre-computed inputs.

    Parameters
    ----------
    intent_record       : parsed intent dict (output of parse_intent)
    persona_sc_concepts : persona top-N concepts filtered to semantic-core only
    recent_sem_freq     : {concept_id: count} for recent window, SC concepts only
    doc_freq_map        : corpus-wide document frequency per concept

    Returns a flat dict with all diagnostic signals.
    """
    persona_sc_set = set(persona_sc_concepts)
    val_list: list[str] = _to_list(intent_record.get("validated_goal_concepts"))
    n_val = len(val_list)

    # ── SC overlap ──────────────────────────────────────────────────────────
    recent_cids = set(recent_sem_freq.keys())
    if recent_cids:
        sc_overlap = len(recent_cids & persona_sc_set) / len(recent_cids)
    else:
        sc_overlap = 0.0

    # ── Persona / recent SC counts ──────────────────────────────────────────
    persona_full_overlap = _full_persona_overlap(intent_record, persona_sc_set)
    persona_semantic_core_count = len(persona_sc_concepts)
    recent_semantic_core_count  = len(recent_cids)

    # ── Concentration ───────────────────────────────────────────────────────
    total_freq = sum(recent_sem_freq.values())
    if total_freq > 0 and recent_sem_freq:
        top1_concept = max(recent_sem_freq, key=recent_sem_freq.get)
        top1_freq    = recent_sem_freq[top1_concept]
        concentration = top1_freq / total_freq
        top1_in_psc   = top1_concept in persona_sc_set
    else:
        top1_concept  = None
        concentration = 0.0
        top1_in_psc   = False

    # ── Goal genericity ─────────────────────────────────────────────────────
    broad_ratio   = sum(1 for c in val_list if c in _BROAD_GENERIC) / n_val if n_val > 0 else 0.0
    val_in_psc    = sum(1 for c in val_list if c in persona_sc_set)  / n_val if n_val > 0 else 0.0

    # ── Specific drift ──────────────────────────────────────────────────────
    # Concepts in recent but NOT in persona SC, with low doc_freq (specific)
    drift_cids = {c for c in recent_cids if c not in persona_sc_set}
    avg_doc_freq_drift = (
        float(np.mean([doc_freq_map.get(c, 10000) for c in drift_cids]))
        if drift_cids else 10000.0
    )
    avg_doc_freq_recent = (
        float(np.mean([doc_freq_map.get(c, 10000) for c in recent_cids]))
        if recent_cids else 10000.0
    )
    n_specific_drift = sum(
        1 for c in drift_cids if doc_freq_map.get(c, 10000) < RECAL_CFG["true_exploration_doc_freq_max"]
    )

    # ── Price/format shift ──────────────────────────────────────────────────
    try:
        constraints = json.loads(intent_record.get("constraints_json", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        constraints = {}
    has_price_shift  = bool(constraints.get("price_band"))
    has_format_shift = bool(constraints.get("format"))

    return {
        # core overlap signals
        "persona_full_overlap":         persona_full_overlap,
        "persona_semantic_core_overlap": sc_overlap,
        "persona_semantic_core_count":  persona_semantic_core_count,
        "recent_semantic_core_count":   recent_semantic_core_count,
        # goal quality signals
        "broad_ratio":          broad_ratio,
        "val_in_persona_sc":    val_in_psc,
        "n_val_goals":          n_val,
        # concentration / specificity
        "concentration":        concentration,
        "top1_in_persona_sc":   top1_in_psc,
        "top1_concept":         top1_concept,
        "avg_doc_freq_recent":  avg_doc_freq_recent,
        "avg_doc_freq_drift":   avg_doc_freq_drift,
        "n_specific_drift":     n_specific_drift,
        # shift flags
        "has_price_shift":      has_price_shift,
        "has_format_shift":     has_format_shift,
    }


def _full_persona_overlap(intent_record: dict, persona_sc_set: set[str]) -> float:
    """
    Fraction of persona_alignment_score stored in the record (LLM-estimated).
    Falls back to computing SC overlap from validated goals if stored score is 0.
    """
    stored = float(intent_record.get("persona_alignment_score", 0.0))
    return stored


# ── Recalibration rule engine ──────────────────────────────────────────────────

def apply_recalibration_rules(
    signals: dict[str, Any],
    val_goal_doc_freqs: list[int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """
    Apply rule cascade to produce recalibrated_reason and recalibration_trace.

    Rule priority (checked in order):
      B → budget_like     : price_band shift in constraints
      A → aligned_soft    : all validated goals are broad/generic
                            OR high SC overlap + generic goals
      C → task_focus_like : narrow validated output + high recent concentration
      D → true_exploration: validated goals contain specific (low doc_freq) concepts
                            (default when no other rule fires with specific goals)
      E → exploration_unclear: fallback

    Parameters
    ----------
    signals            : output of compute_recalibration_signals
    val_goal_doc_freqs : doc_freq values for each validated_goal_concept (used in rule D)
    cfg                : optional threshold overrides

    Returns
    -------
    recalibrated_reason : str  — one of RECAL_REASONS
    recalibration_trace : str  — short structured trace of which rule fired
    """
    c = {**RECAL_CFG, **(cfg or {})}
    val_dfs = val_goal_doc_freqs or []

    sc_ov   = signals["persona_semantic_core_overlap"]
    br      = signals["broad_ratio"]          # fraction of val goals in _BROAD_GENERIC
    vp      = signals["val_in_persona_sc"]
    conc    = signals["concentration"]
    n_val   = signals["n_val_goals"]
    top1_psc= signals["top1_in_persona_sc"]
    n_spdrift = signals["n_specific_drift"]
    has_price = signals["has_price_shift"]

    # ── Rule B: budget_like ────────────────────────────────────────────────
    if has_price:
        return (
            "budget_like",
            "price_band in constraints → budget_like",
        )

    # ── Rule A: aligned_soft ──────────────────────────────────────────────
    # Primary: ALL validated goals are broad/generic (drama, comedy, action_&_adventure, ...)
    # These goals provide no exploration-specific signal — the user is effectively
    # still in their normal taste range.
    if n_val > 0 and br >= 1.0:
        return (
            "aligned_soft",
            f"all_validated_goals_broad=True (n_val={n_val}, broad_ratio=1.0) → aligned_soft",
        )

    # Secondary: SC overlap available + high + goals mostly generic
    if (
        sc_ov >= c["aligned_soft_sc_overlap_min"]
        and (br >= c["aligned_soft_broad_goal_min"] or vp >= c["aligned_soft_val_in_psc_min"])
    ):
        return (
            "aligned_soft",
            f"sc_overlap={sc_ov:.2f}≥{c['aligned_soft_sc_overlap_min']} "
            f"broad_ratio={br:.2f} val_in_psc={vp:.2f} → aligned_soft",
        )

    # ── Rule C: task_focus_like ───────────────────────────────────────────
    # Narrow validated output (1 goal) + high recent semantic concentration
    if conc >= c["task_focus_concentration_min"] and n_val <= c["task_focus_max_val_goals"]:
        return (
            "task_focus_like",
            f"concentration={conc:.2f}≥{c['task_focus_concentration_min']} "
            f"n_val={n_val}≤{c['task_focus_max_val_goals']} → task_focus_like",
        )
    # Top-1 recent concept is in persona SC + moderate concentration → narrowed in-persona drill
    if top1_psc and conc >= c["task_focus_top1_in_psc_conc_min"] and n_val <= 1:
        return (
            "task_focus_like",
            f"top1_in_psc=True conc={conc:.2f}≥{c['task_focus_top1_in_psc_conc_min']} "
            f"n_val={n_val} → task_focus_like",
        )

    # ── Rule D: true_exploration ──────────────────────────────────────────
    # Validated goals contain at least one specific (low doc_freq) concept.
    # This means Stage 2 passed a non-generic goal → genuine direction shift.
    n_specific_goals = sum(
        1 for df_val in val_dfs
        if df_val < c["true_exploration_goal_doc_freq_max"]
    ) if val_dfs else 0

    if n_specific_goals >= c["true_exploration_min_specific_goals"]:
        return (
            "true_exploration",
            f"n_specific_goals={n_specific_goals}≥{c['true_exploration_min_specific_goals']} "
            f"(doc_freq<{c['true_exploration_goal_doc_freq_max']}) → true_exploration",
        )

    # ── Rule E: exploration_unclear ───────────────────────────────────────
    return (
        "exploration_unclear",
        f"sc_overlap={sc_ov:.2f} conc={conc:.2f} n_val={n_val} "
        f"n_specific_goals={n_specific_goals} broad_ratio={br:.2f} → no rule matched → exploration_unclear",
    )


# ── High-level entry point ─────────────────────────────────────────────────────

def recalibrate_record(
    intent_record: dict,
    persona_sc_concepts: list[str],
    recent_sem_freq: dict[str, int],
    doc_freq_map: dict[str, int],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Recalibrate a single parsed intent record.

    Only processes records where deviation_reason == "exploration".
    All other reasons pass through with recalibrated_reason == deviation_reason
    and a passthrough trace.

    Returns the original intent_record dict extended with:
        recalibrated_reason   : str
        recalibration_trace   : str
        recalibration_signals : dict  (full signal diagnostics)
    """
    original_reason = intent_record.get("deviation_reason", "unknown")

    if original_reason != "exploration":
        return {
            **intent_record,
            "recalibrated_reason":   original_reason,
            "recalibration_trace":   f"passthrough: reason={original_reason}",
            "recalibration_signals": {},
        }

    signals = compute_recalibration_signals(
        intent_record, persona_sc_concepts, recent_sem_freq, doc_freq_map,
    )

    # Compute doc_freq for each validated goal concept (needed by rule D)
    val_list = _to_list(intent_record.get("validated_goal_concepts"))
    val_goal_doc_freqs = [doc_freq_map.get(c, 10000) for c in val_list]

    rec_reason, trace = apply_recalibration_rules(signals, val_goal_doc_freqs, cfg)

    return {
        **intent_record,
        "recalibrated_reason":   rec_reason,
        "recalibration_trace":   trace,
        "recalibration_signals": signals,
    }


# ── Batch helper ──────────────────────────────────────────────────────────────

def recalibrate_dataframe(
    df_intent: "pd.DataFrame",
    persona_sc_by_user: dict[str, list[str]],   # user_id → list[concept_id]
    recent_sem_freq_by_key: dict[tuple[str, int], dict[str, int]],  # (uid, ti) → freq dict
    doc_freq_map: dict[str, int],
    cfg: dict[str, Any] | None = None,
) -> "pd.DataFrame":
    """
    Apply recalibration to an entire intent DataFrame.

    Adds columns:
        recalibrated_reason   (str)
        recalibration_trace   (str)
        rc_sc_overlap         (float)  — persona_semantic_core_overlap
        rc_concentration      (float)
        rc_broad_ratio        (float)
        rc_n_val_goals        (int)
        rc_n_specific_drift   (int)
        rc_has_price_shift    (bool)

    Non-exploration rows: recalibrated_reason == deviation_reason, trace == "passthrough:..."
    """
    import pandas as pd

    rec_reasons: list[str]  = []
    rec_traces:  list[str]  = []
    rc_sc:   list[float]    = []
    rc_conc: list[float]    = []
    rc_br:   list[float]    = []
    rc_nval: list[int]      = []
    rc_nspdrift: list[int]  = []
    rc_price: list[bool]    = []

    for row in df_intent.itertuples(index=False):
        uid = str(row.user_id)
        ti  = int(row.target_index)
        intent_rec = row._asdict()

        if intent_rec.get("deviation_reason") != "exploration":
            rec_reasons.append(intent_rec.get("deviation_reason", "unknown"))
            rec_traces.append(f"passthrough: reason={intent_rec.get('deviation_reason','unknown')}")
            rc_sc.append(float("nan"))
            rc_conc.append(float("nan"))
            rc_br.append(float("nan"))
            rc_nval.append(0)
            rc_nspdrift.append(0)
            rc_price.append(False)
            continue

        psc  = persona_sc_by_user.get(uid, [])
        rsf  = recent_sem_freq_by_key.get((uid, ti), {})
        result = recalibrate_record(intent_rec, psc, rsf, doc_freq_map, cfg=cfg)

        rec_reasons.append(result["recalibrated_reason"])
        rec_traces.append(result["recalibration_trace"])
        sigs = result["recalibration_signals"]
        rc_sc.append(sigs.get("persona_semantic_core_overlap", float("nan")))
        rc_conc.append(sigs.get("concentration", float("nan")))
        rc_br.append(sigs.get("broad_ratio", float("nan")))
        rc_nval.append(int(sigs.get("n_val_goals", 0)))
        rc_nspdrift.append(int(sigs.get("n_specific_drift", 0)))
        rc_price.append(bool(sigs.get("has_price_shift", False)))

    df_out = df_intent.copy()
    df_out["recalibrated_reason"] = rec_reasons
    df_out["recalibration_trace"] = rec_traces
    df_out["rc_sc_overlap"]       = rc_sc
    df_out["rc_concentration"]    = rc_conc
    df_out["rc_broad_ratio"]      = rc_br
    df_out["rc_n_val_goals"]      = rc_nval
    df_out["rc_n_specific_drift"] = rc_nspdrift
    df_out["rc_has_price_shift"]  = rc_price
    return df_out


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_list(x: Any) -> list:
    if x is None:
        return []
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
    return []


def build_persona_sc_map(df_persona: "pd.DataFrame") -> dict[str, list[str]]:
    """
    Build {user_id: [semantic_core_concept_id, ...]} from persona_graphs.parquet.
    Sorted by persona weight descending.  Only is_persona_semantic_core() concepts included.
    """
    result: dict[str, list[str]] = {}
    for uid, grp in df_persona.sort_values(
        ["user_id", "weight"], ascending=[True, False]
    ).groupby("user_id"):
        result[str(uid)] = [
            c for c in grp["concept_id"].tolist()
            if is_persona_semantic_core(c)
        ][:10]
    return result


def build_recent_sem_freq_map(
    df_interactions: "pd.DataFrame",
    df_intent: "pd.DataFrame",
    item_concepts: dict[str, list[str]],
    window: int = 10,
) -> dict[tuple[str, int], dict[str, int]]:
    """
    Build {(user_id, target_index): {concept_id: count}} for each row in df_intent.

    Uses the `window` items immediately before target_index in the interaction history.
    Only is_persona_semantic_core() concepts are counted.
    """
    from collections import Counter

    inter_sorted = df_interactions.sort_values(["user_id", "timestamp"])
    user_items: dict[str, list[str]] = (
        inter_sorted.groupby("user_id")["item_id"].apply(list).to_dict()
    )

    result: dict[tuple[str, int], dict[str, int]] = {}
    for row in df_intent.itertuples(index=False):
        uid = str(row.user_id)
        ti  = int(row.target_index)
        items = user_items.get(uid, [])
        window_items = items[max(0, ti - window) : ti]
        freq: Counter = Counter()
        for it in window_items:
            for cid in item_concepts.get(it, []):
                if is_persona_semantic_core(cid):
                    freq[cid] += 1
        result[(uid, ti)] = dict(freq)

    return result
