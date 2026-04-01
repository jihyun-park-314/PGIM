"""
parser.py
---------
Validates and normalizes the raw dict output from heuristic/llm interpreters
into the canonical ShortTermIntent schema.

Output schema (one row in short_term_intents.parquet):
    user_id, target_index,
    goal_concepts, constraints_json,
    deviation_reason, confidence, ttl_steps,
    is_deviation, persona_alignment_score,
    evidence_item_ids, source_mode, parser_status,
    -- Stage 2 grounded selector fields --
    raw_llm_goals, validated_goal_concepts, grounding_diagnostics,
    -- v3 rationale + provenance fields --
    llm_explanation_short, why_not_aligned, why_exploration,
    llm_raw, evidence_recent_concepts, evidence_persona_concepts,
    raw_model_response_json, pre_grounding_goal_text,
    reason_source, has_stage2, llm_prompt_version, schema_version
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_REASONS = {"aligned", "exploration", "task_focus", "budget_shift", "proxy_use", "unknown"}


def parse_intent(
    raw: dict[str, Any],
    user_id: str,
    target_index: int,
    source_mode: str,
) -> dict:
    """
    Validate and normalize a raw interpreter output dict into a canonical record.

    raw expected keys (all optional with fallbacks):
        goal_concepts: list[str]
        constraints: dict
        deviation_reason: str
        confidence: float
        ttl_steps: int
        persona_alignment_score: float
        evidence_item_ids: list[str]

    Returns a flat dict ready to be appended to a DataFrame.
    """
    status = "ok"

    # goal_concepts
    goal_concepts = raw.get("goal_concepts", [])
    if not isinstance(goal_concepts, list):
        goal_concepts = []
        status = "partial"

    # constraints
    constraints = raw.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        status = "partial"
    try:
        constraints_json = json.dumps(constraints, ensure_ascii=False)
    except Exception:
        constraints_json = "{}"
        status = "partial"

    # deviation_reason
    reason = raw.get("deviation_reason", "unknown")
    if reason not in VALID_REASONS:
        logger.debug("Unknown deviation_reason '%s', falling back to 'unknown'", reason)
        reason = "unknown"
        status = "partial"

    # confidence
    confidence = raw.get("confidence", 0.35)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.35
        status = "partial"

    # ttl_steps
    ttl_steps = raw.get("ttl_steps", 1)
    try:
        ttl_steps = int(ttl_steps)
    except (TypeError, ValueError):
        ttl_steps = 1
        status = "partial"

    # alignment score
    alignment = raw.get("persona_alignment_score", 0.0)
    try:
        alignment = float(alignment)
    except (TypeError, ValueError):
        alignment = 0.0

    # evidence
    evidence = raw.get("evidence_item_ids", [])
    if not isinstance(evidence, list):
        evidence = []

    is_deviation = int(reason not in {"aligned", "unknown"})

    # v2: source-aware fields (optional, graceful fallback)
    current_source = raw.get("current_source", "unknown")
    if current_source not in ("rec", "search"):
        current_source = "unknown"
    source_shift_flag = bool(raw.get("source_shift_flag", False))
    recent_source_rec_frac    = float(raw.get("recent_source_rec_frac", 0.5))
    recent_source_search_frac = float(raw.get("recent_source_search_frac", 0.5))

    # Stage 2 grounded selector fields (optional; present only when LLM + bank provided)
    # raw_llm_goals: Stage 1 output before Stage 2 validation (for ablation comparison)
    # validated_goal_concepts: Stage 2 output (grounded to backbone candidate bank)
    # If not present, fall back to goal_concepts to stay backward-compatible.
    # _is_concept_list: accepts list, numpy.ndarray, tuple — anything iterable but not str.
    # parquet round-trips list columns as numpy.ndarray; isinstance(x, list) would miss those.
    def _is_concept_list(x: Any) -> bool:
        return x is not None and hasattr(x, "__iter__") and not isinstance(x, str)

    raw_llm_goals_raw = raw.get("raw_llm_goals")
    if _is_concept_list(raw_llm_goals_raw):
        raw_llm_goals = [str(c) for c in raw_llm_goals_raw]
    else:
        raw_llm_goals = list(goal_concepts)  # same as goal_concepts when Stage 2 absent

    validated_raw = raw.get("validated_goal_concepts")
    if _is_concept_list(validated_raw):
        validated_goal_concepts = [str(c) for c in validated_raw]
    else:
        validated_goal_concepts = list(goal_concepts)  # backward-compat fallback

    grounding_diagnostics = raw.get("grounding_diagnostics", {})
    if not isinstance(grounding_diagnostics, dict):
        grounding_diagnostics = {}

    # ── v3: rationale slots (LLM-generated; empty string when absent/fallback) ──
    def _clean_str(val: Any, max_len: int = 300) -> str:
        """Normalize to str, strip, clip. Never None — always "". Parquet-safe."""
        if val is None or not isinstance(val, str):
            return ""
        return val.strip()[:max_len]

    llm_explanation_short = _clean_str(raw.get("llm_explanation_short"))
    why_not_aligned       = _clean_str(raw.get("why_not_aligned"))
    why_exploration       = _clean_str(raw.get("why_exploration"))

    # Enforce conditional emptiness: slots that must be "" for certain reasons.
    if reason == "aligned":
        why_not_aligned = ""
        why_exploration = ""
    if reason != "exploration":
        why_exploration = ""

    # ── v3: provenance fields (code-injected in llm_interpreter; forwarded here) ──
    llm_raw = raw.get("llm_raw")  # raw JSON string from LLM; None for heuristic/fallback
    if not isinstance(llm_raw, str):
        llm_raw = None

    # raw_model_response_json: alias of llm_raw (prefer over None if present)
    raw_model_response_json = raw.get("raw_model_response_json")
    if not isinstance(raw_model_response_json, str):
        raw_model_response_json = llm_raw  # fall through to llm_raw

    # evidence_recent_concepts / evidence_persona_concepts: list[str]
    def _clean_str_list(val: Any) -> list[str]:
        if val is None:
            return []
        if _is_concept_list(val):
            return [str(c) for c in val]
        return []

    evidence_recent_concepts  = _clean_str_list(raw.get("evidence_recent_concepts"))
    evidence_persona_concepts = _clean_str_list(raw.get("evidence_persona_concepts"))

    # pre_grounding_goal_text: same as raw_llm_goals when present
    pre_grounding_raw = raw.get("pre_grounding_goal_text")
    if _is_concept_list(pre_grounding_raw):
        pre_grounding_goal_text = [str(c) for c in pre_grounding_raw]
    else:
        pre_grounding_goal_text = list(raw_llm_goals)  # fallback: same as raw_llm_goals

    reason_source      = _clean_str(raw.get("reason_source", source_mode), max_len=50)
    has_stage2         = bool(raw.get("has_stage2", False))
    llm_prompt_version = _clean_str(raw.get("llm_prompt_version", ""), max_len=50)
    schema_version     = _clean_str(raw.get("schema_version", ""), max_len=20)

    # ── v5: semantic goal hygiene fields ──────────────────────────────────────
    removed_non_semantic_goals = _clean_str_list(raw.get("removed_non_semantic_goals"))
    semantic_signal_absent     = bool(raw.get("semantic_signal_absent", False))
    non_semantic_goal_leakage  = bool(raw.get("non_semantic_goal_leakage", False))
    goal_hygiene_status        = _clean_str(raw.get("goal_hygiene_status", ""), max_len=50)

    return {
        "user_id": user_id,
        "target_index": target_index,
        "goal_concepts": goal_concepts,
        "constraints_json": constraints_json,
        "deviation_reason": reason,
        "confidence": confidence,
        "ttl_steps": ttl_steps,
        "is_deviation": is_deviation,
        "persona_alignment_score": alignment,
        "evidence_item_ids": evidence,
        "source_mode": source_mode,
        "parser_status": status,
        "current_source": current_source,
        "source_shift_flag": source_shift_flag,
        "recent_source_rec_frac": recent_source_rec_frac,
        "recent_source_search_frac": recent_source_search_frac,
        # Stage 2 grounded selector fields
        "raw_llm_goals": raw_llm_goals,
        "validated_goal_concepts": validated_goal_concepts,
        "grounding_diagnostics": grounding_diagnostics,
        # v3 rationale slots (LLM-generated)
        "llm_explanation_short": llm_explanation_short,
        "why_not_aligned": why_not_aligned,
        "why_exploration": why_exploration,
        # v3 provenance fields (code-injected)
        "llm_raw": llm_raw,
        "evidence_recent_concepts": evidence_recent_concepts,
        "evidence_persona_concepts": evidence_persona_concepts,
        "raw_model_response_json": raw_model_response_json,
        "pre_grounding_goal_text": pre_grounding_goal_text,
        "reason_source": reason_source,
        "has_stage2": has_stage2,
        "llm_prompt_version": llm_prompt_version,
        "schema_version": schema_version,
        # v5 semantic goal hygiene
        "removed_non_semantic_goals": removed_non_semantic_goals,
        "semantic_signal_absent": semantic_signal_absent,
        "non_semantic_goal_leakage": non_semantic_goal_leakage,
        "goal_hygiene_status": goal_hygiene_status,
    }
