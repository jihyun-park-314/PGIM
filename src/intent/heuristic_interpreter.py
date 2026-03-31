"""
heuristic_interpreter.py
-------------------------
Rule-based short-term intent extraction.
Produces a raw dict that parser.parse_intent() normalizes.

Deviation reason taxonomy:
    aligned       — recent behavior consistent with persona
    exploration   — recent behavior diverges widely from persona
    task_focus    — recent behavior concentrated on a specific attribute
    budget_shift  — recent price_band differs from persona dominant
    unknown       — no stable rule fires
"""

from __future__ import annotations

import math
from typing import Any

from src.intent.context_extractor import IntentContext


def interpret(ctx: IntentContext, intent_cfg: dict) -> dict[str, Any]:
    """
    Run heuristic rules on IntentContext and return a raw intent dict.
    """
    dev_cfg = intent_cfg.get("deviation", {})
    conf_cfg = intent_cfg.get("confidence", {})
    ttl_cfg = intent_cfg.get("ttl", {})
    ctx_cfg = intent_cfg.get("context", {})
    top_n = ctx_cfg.get("top_goal_concepts", 3)

    # ----------------------------------------------------------------
    # 1. Goal concepts: priority-aware top-N from recent window
    #
    # If goal_concept_priority is configured, concepts are sorted by
    # (priority_rank ASC, frequency DESC) so higher-priority types
    # (e.g. category) are always preferred over low-priority types
    # (e.g. price_band, format) even when less frequent.
    # Falls back to pure frequency ordering when no priority configured.
    # ----------------------------------------------------------------
    freq = ctx.recent_concept_freq
    priority_cfg: dict[str, int] = ctx_cfg.get("goal_concept_priority", {})

    if priority_cfg:
        def _sort_key(cid: str) -> tuple:
            ctype = cid.split(":")[0]
            prank = priority_cfg.get(ctype, 99)
            return (prank, -freq.get(cid, 0))
        goal_concepts = sorted(freq, key=_sort_key)[:top_n]
    else:
        goal_concepts = sorted(freq, key=freq.get, reverse=True)[:top_n]

    # ----------------------------------------------------------------
    # 2. Constraints: structured attribute constraints from recent window
    # ----------------------------------------------------------------
    constraints: dict[str, list[str]] = {}
    if ctx.recent_dominant_price_band and ctx.recent_dominant_price_band != "price_band:unknown":
        constraints["price_band"] = [ctx.recent_dominant_price_band]
    if ctx.recent_dominant_brand:
        constraints["brand"] = [ctx.recent_dominant_brand]
    if ctx.recent_dominant_item_form:
        constraints["item_form"] = [ctx.recent_dominant_item_form]
    if ctx.recent_dominant_skin_type:
        constraints["skin_type"] = [ctx.recent_dominant_skin_type]

    # ----------------------------------------------------------------
    # 3. Deviation reason detection (rules applied in priority order)
    # ----------------------------------------------------------------
    aligned_thresh = dev_cfg.get("aligned_overlap_threshold", 0.4)
    explore_entropy_thresh = dev_cfg.get("exploration_entropy_threshold", 0.6)
    task_focus_thresh = dev_cfg.get("task_focus_dominance_threshold", 0.5)
    budget_min_support = dev_cfg.get("budget_shift_min_persona_support", 2)

    reason = "unknown"

    # Rule A: budget_shift — check before aligned because price shift is explicit
    recent_pb = ctx.recent_dominant_price_band
    persona_pb = ctx.persona_dominant_price_band
    if (
        recent_pb
        and persona_pb
        and recent_pb != persona_pb
        and recent_pb != "price_band:unknown"
        and persona_pb != "price_band:unknown"
    ):
        reason = "budget_shift"

    # Rule B: task_focus — one non-category concept type dominates recent window
    # v2: source_shift to search also raises task_focus probability
    if reason == "unknown" and freq:
        total_recent = sum(freq.values())
        # group by concept type
        type_totals: dict[str, int] = {}
        for cid, cnt in freq.items():
            ctype = cid.split(":")[0]
            type_totals[ctype] = type_totals.get(ctype, 0) + cnt
        # exclude price_band:unknown from dominance signal
        type_totals_clean = {
            t: v for t, v in type_totals.items()
            if not (t == "price_band" and _all_unknown_price(freq))
        }
        if type_totals_clean:
            dominant_type = max(type_totals_clean, key=type_totals_clean.get)
            dominance = type_totals_clean[dominant_type] / total_recent
            # v2: lower threshold slightly when source shift to search observed
            effective_thresh = task_focus_thresh
            if getattr(ctx, "source_shift_flag", False) and getattr(ctx, "current_source", "unknown") == "search":
                effective_thresh = max(0.35, task_focus_thresh - 0.1)
            if dominant_type not in ("category", "price_band") and dominance >= effective_thresh:
                reason = "task_focus"

    # Rule C: aligned — high overlap with persona
    # v2: require no source shift (source shift suggests deviation)
    if reason == "unknown" and ctx.overlap_ratio >= aligned_thresh:
        if not getattr(ctx, "source_shift_flag", False):
            reason = "aligned"

    # Rule D: exploration — low overlap AND (high entropy OR source shift)
    # v2: source shift from rec→search with low overlap is a strong exploration signal
    if reason == "unknown":
        high_entropy = ctx.recent_concept_entropy >= explore_entropy_thresh
        source_shift = getattr(ctx, "source_shift_flag", False)
        current_src  = getattr(ctx, "current_source", "unknown")
        low_overlap  = ctx.overlap_ratio < aligned_thresh
        if high_entropy or (source_shift and low_overlap and current_src == "search"):
            reason = "exploration"

    # Rule E: fallback — if source shift present but overlap is moderate → exploration
    if reason == "unknown" and getattr(ctx, "source_shift_flag", False):
        reason = "exploration"

    # Rule F: fallback to aligned if overlap is moderate but nothing else fired
    if reason == "unknown" and ctx.overlap_ratio > 0.2:
        reason = "aligned"

    # ----------------------------------------------------------------
    # 4. Confidence
    # ----------------------------------------------------------------
    base_confidence = conf_cfg.get(reason, conf_cfg.get("unknown", 0.35))

    # Boost confidence when signal is strong
    if reason == "aligned" and ctx.overlap_ratio > 0.6:
        base_confidence = min(1.0, base_confidence + 0.1)
    elif reason == "task_focus" and freq:
        # boost when dominance is very high
        total = sum(freq.values())
        if total > 0:
            top_val = max(freq.values())
            if top_val / total > 0.7:
                base_confidence = min(1.0, base_confidence + 0.05)
    # v2: boost exploration confidence when source shift confirms it
    elif reason == "exploration" and getattr(ctx, "source_shift_flag", False):
        base_confidence = min(1.0, base_confidence + 0.08)

    # Penalize when recent window is very short (< 3 items)
    if len(ctx.recent_item_ids) < 3:
        base_confidence = max(0.2, base_confidence - 0.15)

    # ----------------------------------------------------------------
    # 5. TTL
    # ----------------------------------------------------------------
    ttl_steps = ttl_cfg.get(reason, ttl_cfg.get("unknown", 1))

    # ----------------------------------------------------------------
    # 6. Persona alignment score (continuous)
    # ----------------------------------------------------------------
    persona_alignment_score = ctx.overlap_ratio

    return {
        "goal_concepts": goal_concepts,
        "constraints": constraints,
        "deviation_reason": reason,
        "confidence": round(base_confidence, 4),
        "ttl_steps": ttl_steps,
        "persona_alignment_score": round(persona_alignment_score, 4),
        "evidence_item_ids": ctx.recent_item_ids[-3:],  # last 3 items as evidence
        # v2: source-aware fields passed through to intent record
        "current_source": getattr(ctx, "current_source", "unknown"),
        "source_shift_flag": getattr(ctx, "source_shift_flag", False),
        "recent_source_rec_frac": getattr(ctx, "recent_source_rec_frac", 0.5),
        "recent_source_search_frac": getattr(ctx, "recent_source_search_frac", 0.5),
    }


def _all_unknown_price(freq: dict[str, int]) -> bool:
    pb_concepts = [k for k in freq if k.startswith("price_band:")]
    return all(k == "price_band:unknown" for k in pb_concepts)
