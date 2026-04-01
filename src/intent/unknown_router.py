"""
unknown_router.py
-----------------
Post-hoc soft routing layer for unknown-labeled intent records (v5+).

Problem being solved:
    v5 re-generation reduced exploration from 57.7% → 26.8%, but unknown jumped
    from 0.5% → 44.5%.  Audit shows:
      - 440/890 unknown = semantic_signal_absent=True (genuine null)
      - 380/890 unknown = has semantic evidence but LLM was too conservative
      - constraints non-empty: 8 cases (budget_like candidates)
    Unknown subtype distribution (approximate):
      - unknown_null         ~57% — no semantic signal at all
      - unknown_soft_task    ~28% — narrow 1-concept semantic evidence
      - unknown_soft_aligned ~14% — multi-concept semantic, persona-consistent
      - unknown_soft_budget   ~1% — constraints present

This module does NOT change LLM behavior or re-prompt.  It applies deterministic
rule-based routing to already-computed v5 intent records, using:
  - semantic_signal_absent flag
  - evidence_recent_concepts (semantic only)
  - validated_goal_concepts
  - constraints_json (price_band / format)
  - persona_alignment_score / evidence_persona_concepts

Output: routed_reason ∈ {
    unknown_null,
    unknown_soft_aligned,
    unknown_soft_task_focus,
    unknown_soft_budget,
}
For non-unknown reasons, routed_reason = original deviation_reason (passthrough).

signal_builder.py is extended to route:
    unknown_null           → conservative (same as unknown)
    unknown_soft_aligned   → aligned_soft branch (persona prior + soft recent boost)
    unknown_soft_task_focus → task_focus_like branch (narrow boost)
    unknown_soft_budget    → budget_like branch (price/product-context adjustment)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Routing vocabulary ─────────────────────────────────────────────────────────
UNKNOWN_SUBTYPES: frozenset[str] = frozenset({
    "unknown_null",
    "unknown_soft_aligned",
    "unknown_soft_task_focus",
    "unknown_soft_budget",
})

# Non-unknown reasons pass through unchanged.
PASSTHROUGH_REASONS: frozenset[str] = frozenset({
    "aligned", "task_focus", "budget_shift", "exploration",
    "aligned_soft", "task_focus_like", "budget_like",
    "true_exploration", "exploration_unclear",
})

# ── Config defaults ────────────────────────────────────────────────────────────
ROUTER_CFG: dict[str, Any] = {
    # Min distinct semantic evidence concepts to be considered non-null
    "min_sem_evidence_for_signal": 1,
    # Min validated goal count for soft_aligned (multi-concept)
    "min_val_goals_for_aligned": 1,
    # validated_goals count == 1 → prefer task_focus over aligned
    "single_goal_is_task_focus": True,
    # persona_alignment_score threshold — above this → soft_aligned leaning
    "alignment_score_soft_aligned_min": 0.0,  # intentionally low; any alignment counts
    # constraints non-empty → soft_budget (overrides semantic routing)
    "budget_requires_constraints": True,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
    return []


def _sem_concepts_from_evidence(evidence_list: list) -> list[str]:
    """
    Filter evidence_recent_concepts to semantic-only concept_ids.
    Evidence entries are formatted as "category:drama(3)" or bare "category:drama".
    """
    from src.intent.concept_roles import is_semantic_goal
    result = []
    for item in evidence_list:
        item = str(item)
        cid = item.split("(")[0] if "(" in item else item
        if is_semantic_goal(cid):
            result.append(cid)
    return result


def _has_constraints(constraints_json: str) -> bool:
    """Return True if constraints dict has any non-empty value."""
    try:
        c = json.loads(constraints_json) if isinstance(constraints_json, str) else {}
        return any(bool(v) for v in c.values())
    except Exception:
        return False


# ── Core routing logic ─────────────────────────────────────────────────────────

def compute_routing_signals(intent_record: dict) -> dict:
    """
    Compute routing signals for a single unknown record.

    Returns a flat dict of signals used by apply_routing_rules().
    """
    semantic_signal_absent = bool(intent_record.get("semantic_signal_absent", False))

    evidence_recent = _to_list(intent_record.get("evidence_recent_concepts", []))
    sem_evidence    = _sem_concepts_from_evidence(evidence_recent)
    n_sem_evidence  = len(sem_evidence)

    validated_goals  = _to_list(intent_record.get("validated_goal_concepts", []))
    n_val_goals      = len(validated_goals)

    constraints_json = intent_record.get("constraints_json", "{}")
    has_budget_signal = _has_constraints(str(constraints_json))

    alignment_score = float(intent_record.get("persona_alignment_score", 0.0))

    return {
        "semantic_signal_absent": semantic_signal_absent,
        "n_sem_evidence":         n_sem_evidence,
        "sem_evidence":           sem_evidence,
        "n_val_goals":            n_val_goals,
        "validated_goals":        validated_goals,
        "has_budget_signal":      has_budget_signal,
        "alignment_score":        alignment_score,
    }


def apply_routing_rules(signals: dict, cfg: dict | None = None) -> tuple[str, str]:
    """
    Apply priority-cascade routing rules to an unknown record.

    Priority order:
        1. unknown_null     — semantic_signal_absent OR n_sem_evidence == 0
        2. unknown_soft_budget — constraints present (price/format shift dominant)
        3. unknown_soft_task_focus — single validated goal (narrow focus)
        4. unknown_soft_aligned — multi-concept semantic evidence present
        5. unknown_null     — fallback (residual)

    Returns:
        (subtype, trace_string)
    """
    if cfg is None:
        cfg = ROUTER_CFG

    # Rule 1 — genuine signal absence → null
    if signals["semantic_signal_absent"] or signals["n_sem_evidence"] == 0:
        return "unknown_null", "semantic_signal_absent=True or n_sem_evidence=0"

    # Rule 2 — budget signal dominant → soft_budget
    if cfg.get("budget_requires_constraints", True) and signals["has_budget_signal"]:
        return "unknown_soft_budget", f"constraints_present, n_sem={signals['n_sem_evidence']}"

    # Rule 3 — single validated goal → soft_task_focus (narrow concentration)
    if (cfg.get("single_goal_is_task_focus", True)
            and signals["n_val_goals"] == 1):
        return "unknown_soft_task_focus", (
            f"single_validated_goal={signals['validated_goals']}, "
            f"n_sem_evidence={signals['n_sem_evidence']}"
        )

    # Rule 4 — multi-concept evidence, validated goals present → soft_aligned
    if signals["n_sem_evidence"] >= cfg.get("min_sem_evidence_for_signal", 1) \
            and signals["n_val_goals"] >= cfg.get("min_val_goals_for_aligned", 1):
        return "unknown_soft_aligned", (
            f"n_sem_evidence={signals['n_sem_evidence']}, "
            f"n_val_goals={signals['n_val_goals']}, "
            f"alignment={signals['alignment_score']:.3f}"
        )

    # Rule 5 — has sem evidence but no validated goals → task_focus_like (sparse drift)
    if signals["n_sem_evidence"] >= 1:
        return "unknown_soft_task_focus", (
            f"sem_evidence_present_but_no_validated_goals, "
            f"n_sem={signals['n_sem_evidence']}"
        )

    # Fallback
    return "unknown_null", "fallback: no qualifying signal"


def route_unknown_record(intent_record: dict, cfg: dict | None = None) -> dict:
    """
    Route a single record.

    If deviation_reason != 'unknown', returns passthrough fields unchanged.
    Otherwise computes subtype and routing.

    Returns a dict of routing fields to merge into the record:
        - unknown_subtype    (str)
        - routed_reason      (str)
        - routing_trace      (str)
        - rc_n_sem_evidence  (int)
        - rc_n_val_goals     (int)
        - rc_has_budget      (bool)
    """
    reason = intent_record.get("deviation_reason", "unknown")

    if reason != "unknown":
        return {
            "unknown_subtype":   reason,     # passthrough — subtype = original reason
            "routed_reason":     reason,
            "routing_trace":     "passthrough",
            "rc_n_sem_evidence": -1,
            "rc_n_val_goals":    -1,
            "rc_has_budget":     False,
        }

    signals = compute_routing_signals(intent_record)
    subtype, trace = apply_routing_rules(signals, cfg)

    # routed_reason: map subtype → signal_builder branch
    subtype_to_routed: dict[str, str] = {
        "unknown_null":             "unknown",
        "unknown_soft_aligned":     "aligned_soft",
        "unknown_soft_task_focus":  "task_focus_like",
        "unknown_soft_budget":      "budget_like",
    }
    routed = subtype_to_routed.get(subtype, "unknown")

    return {
        "unknown_subtype":   subtype,
        "routed_reason":     routed,
        "routing_trace":     trace,
        "rc_n_sem_evidence": signals["n_sem_evidence"],
        "rc_n_val_goals":    signals["n_val_goals"],
        "rc_has_budget":     signals["has_budget_signal"],
    }


def route_dataframe(
    df: "pd.DataFrame",
    cfg: dict | None = None,
) -> "pd.DataFrame":
    """
    Apply soft routing to all records in a DataFrame.

    Adds columns:
        unknown_subtype, routed_reason, routing_trace,
        rc_n_sem_evidence, rc_n_val_goals, rc_has_budget

    Original deviation_reason is never modified.
    """
    import pandas as pd

    routing_rows = [route_unknown_record(row.to_dict(), cfg) for _, row in df.iterrows()]
    routing_df = pd.DataFrame(routing_rows)

    out = df.copy()
    for col in routing_df.columns:
        out[col] = routing_df[col].values

    logger.info(
        "unknown_subtype distribution:\n%s",
        out[out["deviation_reason"] == "unknown"]["unknown_subtype"].value_counts().to_string(),
    )
    logger.info(
        "routed_reason distribution (unknown rows only):\n%s",
        out[out["deviation_reason"] == "unknown"]["routed_reason"].value_counts().to_string(),
    )

    return out
