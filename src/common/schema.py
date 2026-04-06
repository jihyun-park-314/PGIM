"""
schema.py
---------
Project-wide field contracts for PGIM pipeline records.

Field taxonomy (three tiers)
─────────────────────────────
VERIFIED_SCORING_FIELDS
  Structured, grounded, code-injected signals.  These are the ONLY fields
  allowed to enter delta computation in signal_builder / reranker.
  Sources: Stage 2 backbone grounding or IntentContext code-paths.

LLM_UNVERIFIED_FIELDS
  LLM-suggested structured output that has NOT been grounded against the
  backbone candidate bank.  May be used for feature engineering in a future
  PR once validated, but MUST NOT enter delta computation today.
  Treated as scoring-forbidden until promoted to VERIFIED_SCORING_FIELDS.

AUDIT_ONLY_FIELDS
  Natural language rationale, raw LLM artifacts, hygiene diagnostics,
  and provenance tags.  For human inspection / offline eval / logging ONLY.
  SCORING FOREVER FORBIDDEN — natural language must never drive a score.

CONTRACT:
  - signal_builder.build_signal() reads ONLY from VERIFIED_SCORING_FIELDS.
  - LLM_UNVERIFIED_FIELDS and AUDIT_ONLY_FIELDS are passed to debug_info only.
  - To promote a field from LLM_UNVERIFIED → VERIFIED, add a grounding step
    in llm_interpreter or grounded_selector and move the field here.
  - signal_builder enforces the audit/unverified guard at runtime via debug log.

Next-PR scoring features (delta_context + delta_deviation):
  delta_context inputs  : validated_goal_concepts, contrast_with_persona (verified only),
                          temporal_cues (verified keys), evidence_sources
  delta_deviation inputs: deviation_reason, confidence, ttl_steps
  See VERIFIED_SCORING_FIELDS for the complete list.
"""

from __future__ import annotations

# ── Tier 1: Verified scoring fields ───────────────────────────────────────────
# Grounded / code-injected.  Only these may enter delta computation.
VERIFIED_SCORING_FIELDS: frozenset[str] = frozenset([
    # Core intent
    "validated_goal_concepts",   # Stage 2 backbone-grounded goals
    "context_goals",             # alias = validated_goal_concepts post-merge
    "deviation_reason",          # expansion policy signal
    "confidence",
    "ttl_steps",
    "persona_alignment_score",
    # Structured contrast — grounding-verified entries only
    # (contrast_with_persona dict; only int-valued entries are grounded)
    "contrast_with_persona",     # {concept_id: int (grounded) | "llm" (unverified)}
    # Temporal cues — code-authoritative keys only
    # (temporal_cues dict; "llm_shift_summary" key inside is audit-only)
    "temporal_cues",             # {shift_detected, first/second_half_dominant, *_freq}
    # Evidence provenance
    "evidence_sources",          # ["recent_freq","temporal_shift","persona_contrast"]
    "evidence_item_ids",
    "support_items",
    # Constraints
    "constraints",
    # Source context (from IntentContext, not LLM)
    "current_source",
    "recent_source_rec_frac",
    "recent_source_search_frac",
    "source_shift_flag",
])

# ── Tier 2: LLM-suggested, not yet grounded ───────────────────────────────────
# Structured output from LLM that has NOT been validated against backbone bank.
# Do NOT read in delta computation.  Candidate for future promotion.
LLM_UNVERIFIED_FIELDS: frozenset[str] = frozenset([
    # LLM-suggested contrast — ungrounded entries in contrast_with_persona dict
    # are marked with value "llm"; no separate top-level field needed.
    # The "llm" sentinel in contrast_with_persona values is the unverified channel.
    #
    # LLM temporal summary — stored inside temporal_cues["llm_shift_summary"].
    # Not a top-level field; accessed via temporal_cues dict only.
    #
    # Token counts — LLM resource usage, not a signal
    # prompt_tokens = system + user combined (OpenAI API semantics)
    "token_usage",               # {system_prompt_chars, user_prompt_chars, compact,
                                 #  prompt_tokens, response_tokens, total_tokens}
])

# ── Tier 3: Audit-only fields ─────────────────────────────────────────────────
# Populated for human inspection / offline eval / logging.
# SCORING FOREVER FORBIDDEN.
AUDIT_ONLY_FIELDS: frozenset[str] = frozenset([
    # LLM natural-language rationale — never score these directly
    "llm_explanation_short",
    "why_not_aligned",
    "why_exploration",
    # Raw LLM artifacts
    "llm_raw",
    "raw_model_response_json",
    # Hygiene / provenance metadata
    "removed_non_semantic_goals",
    "goal_hygiene_status",
    "non_semantic_goal_leakage",
    "semantic_signal_absent",
    "source_mode",
    "reason_source",
    "pre_grounding_goal_text",
    "raw_llm_goals",
    # Grounding diagnostics (eval-only)
    "grounding_diagnostics",
    # Prompt/schema version tags
    "llm_prompt_version",
    "schema_version",
])

# ── Combined forbidden set (for signal_builder guard) ─────────────────────────
# signal_builder must not read any field in this set for delta computation.
SCORING_FORBIDDEN_FIELDS: frozenset[str] = AUDIT_ONLY_FIELDS | LLM_UNVERIFIED_FIELDS
