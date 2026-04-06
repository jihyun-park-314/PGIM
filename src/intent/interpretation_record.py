"""
interpretation_record.py
------------------------
Normalized schema for PGIM interpretation records.

Defines NormalizedInterpretationRecord — the canonical output of Stage 1+2
(llm_interpreter + grounded_selector) — with three explicit channels:

  VERIFIED (scoring-ready)
    Fields grounded against the backbone candidate bank or derived from
    IntentContext code-paths.  These are the ONLY inputs allowed into
    delta computation in signal_builder / reranker.

  LLM_UNVERIFIED (structured but not grounded)
    LLM-suggested structured output not yet validated against the backbone.
    Candidate for future promotion to VERIFIED once a grounding step is added.
    Must not enter delta computation today.

  AUDIT (natural language + raw artifacts)
    For human inspection / offline eval / logging only.
    Scoring forever forbidden.

Inline field contracts are authoritative — keep in sync with src/common/schema.py.

Channel separation rules
────────────────────────
contrast_with_persona dict
  int value  → grounding-verified (Stage 2 backbone suppression)
  "llm" str  → LLM-only, NOT grounded → treat as LLM_UNVERIFIED

temporal_cues dict
  keys shift_detected, first_half_dominant, second_half_dominant,
       first_half_freq, second_half_freq  → code-authoritative → VERIFIED
  key  llm_shift_summary                 → LLM-generated phrase → AUDIT

token_usage dict
  LLM resource counters — not a signal → LLM_UNVERIFIED

Next-PR scoring features (delta_context + delta_deviation):
  delta_context  ← validated_goal_concepts, contrast_with_persona (int entries),
                   temporal_cues (shift_detected, dominant keys), evidence_sources
  delta_deviation← deviation_reason, confidence, ttl_steps
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Verified channel ─────────────────────────────────────────────────────────

@dataclass
class VerifiedSignal:
    """
    Scoring-ready fields.  All entries are code-injected or backbone-grounded.

    contrast_with_persona: {concept_id: int | "llm"}
      Only int-valued entries are grounded.  "llm"-valued entries are
      LLM-suggested and are NOT used for scoring until promoted.
      Call .verified_contrast() to get the grounded-only subset.

    temporal_cues: code-authoritative temporal split from IntentContext.
      Internal key "llm_shift_summary" is an audit phrase — ignored by scoring.
      Call .verified_temporal() to get the scoring-safe subset.
    """
    # Core intent — Stage 2 grounded
    validated_goal_concepts: list[str] = field(default_factory=list)
    context_goals: list[str] = field(default_factory=list)
    deviation_reason: str = "unknown"
    confidence: float = 0.35
    ttl_steps: int = 1
    persona_alignment_score: float = 0.0

    # Structured contrast — mixed (int=grounded, "llm"=unverified)
    contrast_with_persona: dict[str, Any] = field(default_factory=dict)

    # Temporal cues — code-authoritative + embedded LLM phrase (audit)
    temporal_cues: dict[str, Any] = field(default_factory=dict)

    # Evidence provenance
    evidence_sources: list[str] = field(default_factory=list)
    evidence_item_ids: list[str] = field(default_factory=list)
    support_items: list[str] = field(default_factory=list)

    # Constraints
    constraints: dict = field(default_factory=dict)

    # Source context (from IntentContext, not LLM)
    current_source: str = "unknown"

    def verified_contrast(self) -> dict[str, int]:
        """
        Return only grounding-verified contrast entries (int-valued).
        Excludes LLM-only entries (str "llm") — those are unverified.
        """
        return {c: v for c, v in self.contrast_with_persona.items() if isinstance(v, int)}

    def llm_contrast_concepts(self) -> list[str]:
        """Return concept IDs from LLM-only contrast (unverified, not for scoring)."""
        return [c for c, v in self.contrast_with_persona.items() if v == "llm"]

    def verified_temporal(self) -> dict[str, Any]:
        """
        Return code-authoritative temporal cue keys only.
        Excludes llm_shift_summary (audit phrase).
        """
        _audit_key = "llm_shift_summary"
        return {k: v for k, v in self.temporal_cues.items() if k != _audit_key}

    def shift_detected(self) -> bool:
        return bool(self.temporal_cues.get("shift_detected", False))


# ── LLM-unverified channel ───────────────────────────────────────────────────

@dataclass
class LLMUnverifiedSignal:
    """
    LLM-suggested structured output not grounded against backbone.
    Must not enter delta computation today.

    token_usage: LLM resource counters for cost/observability logging.
    """
    # Token usage — observability only
    token_usage: dict[str, int] = field(default_factory=dict)

    # LLM-only contrast concepts (subset of VerifiedSignal.contrast_with_persona
    # where value == "llm").  Stored here as a convenience flat list.
    llm_contrast_concepts: list[str] = field(default_factory=list)

    # LLM temporal shift phrase — stored in temporal_cues["llm_shift_summary"]
    # but mirrored here for direct audit access without dict traversal.
    llm_shift_summary: str = ""

    def has_token_data(self) -> bool:
        return bool(self.token_usage)

    def total_tokens(self) -> int:
        return self.token_usage.get("total_tokens", 0)


# ── Audit channel ─────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """
    Natural language rationale + raw LLM artifacts + hygiene diagnostics.
    For human inspection / offline eval / logging ONLY.
    SCORING FOREVER FORBIDDEN.
    """
    # Natural language rationale
    llm_explanation_short: str = ""
    why_not_aligned: str = ""
    why_exploration: str = ""

    # Raw LLM output
    llm_raw: Optional[str] = None
    raw_model_response_json: Optional[str] = None

    # Hygiene diagnostics
    removed_non_semantic_goals: list[str] = field(default_factory=list)
    goal_hygiene_status: str = ""
    non_semantic_goal_leakage: bool = False
    semantic_signal_absent: bool = False

    # Provenance
    source_mode: str = "llm"
    reason_source: str = "llm"
    pre_grounding_goal_text: list[str] = field(default_factory=list)
    raw_llm_goals: list[str] = field(default_factory=list)

    # Grounding diagnostics (Stage 2 output; eval-only)
    grounding_diagnostics: dict = field(default_factory=dict)

    # Prompt/schema version tags
    llm_prompt_version: str = ""
    schema_version: str = ""


# ── Normalized record ─────────────────────────────────────────────────────────

@dataclass
class NormalizedInterpretationRecord:
    """
    Canonical output of Stage 1+2 (llm_interpreter + grounded_selector).

    Three explicit channels:
      verified   — scoring-ready; only these enter delta computation
      unverified — LLM-structured but not grounded; candidate for future promotion
      audit      — natural language + raw artifacts; scoring forever forbidden

    Identity fields
    ───────────────
    user_id, target_index — always present; required by downstream modules.
    has_stage2 — True when Stage 2 (grounded_selector) ran.
    """
    user_id: str
    target_index: int
    has_stage2: bool = False

    verified: VerifiedSignal = field(default_factory=VerifiedSignal)
    unverified: LLMUnverifiedSignal = field(default_factory=LLMUnverifiedSignal)
    audit: AuditRecord = field(default_factory=AuditRecord)

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_flat_record(cls, r: dict) -> "NormalizedInterpretationRecord":
        """
        Build from the flat dict produced by llm_interpreter.interpret_with_llm().
        Separates fields into the three channels based on schema.py contracts.
        """
        v = VerifiedSignal(
            validated_goal_concepts=list(r.get("validated_goal_concepts") or r.get("goal_concepts") or []),
            context_goals=list(r.get("context_goals") or []),
            deviation_reason=r.get("deviation_reason", "unknown"),
            confidence=float(r.get("confidence", 0.35)),
            ttl_steps=int(r.get("ttl_steps", 1)),
            persona_alignment_score=float(r.get("persona_alignment_score", 0.0)),
            contrast_with_persona=dict(r.get("contrast_with_persona") or {}),
            temporal_cues=dict(r.get("temporal_cues") or {}),
            evidence_sources=list(r.get("evidence_sources") or []),
            evidence_item_ids=list(r.get("evidence_item_ids") or []),
            support_items=list(r.get("support_items") or []),
            constraints=dict(r.get("constraints") or {}),
            current_source=r.get("current_source", "unknown"),
        )

        # Token usage — populated if token_usage dict is present in record
        tu = r.get("token_usage") or {}
        uv = LLMUnverifiedSignal(
            token_usage=dict(tu),
            llm_contrast_concepts=v.llm_contrast_concepts(),
            llm_shift_summary=v.temporal_cues.get("llm_shift_summary", ""),
        )

        a = AuditRecord(
            llm_explanation_short=r.get("llm_explanation_short", ""),
            why_not_aligned=r.get("why_not_aligned", ""),
            why_exploration=r.get("why_exploration", ""),
            llm_raw=r.get("llm_raw"),
            raw_model_response_json=r.get("raw_model_response_json"),
            removed_non_semantic_goals=list(r.get("removed_non_semantic_goals") or []),
            goal_hygiene_status=r.get("goal_hygiene_status", ""),
            non_semantic_goal_leakage=bool(r.get("non_semantic_goal_leakage", False)),
            semantic_signal_absent=bool(r.get("semantic_signal_absent", False)),
            source_mode=r.get("source_mode", "llm"),
            reason_source=r.get("reason_source", "llm"),
            pre_grounding_goal_text=list(r.get("pre_grounding_goal_text") or []),
            raw_llm_goals=list(r.get("raw_llm_goals") or []),
            grounding_diagnostics=dict(r.get("grounding_diagnostics") or {}),
            llm_prompt_version=r.get("llm_prompt_version", ""),
            schema_version=r.get("schema_version", ""),
        )

        return cls(
            user_id=r["user_id"],
            target_index=int(r["target_index"]),
            has_stage2=bool(r.get("has_stage2", False)),
            verified=v,
            unverified=uv,
            audit=a,
        )

    # ── Serialization ───────────────────────────────────────────────────────

    def to_flat_record(self) -> dict:
        """
        Flatten back to the dict format consumed by signal_builder.
        Preserves full backward compatibility.
        """
        v, uv, a = self.verified, self.unverified, self.audit
        return {
            "user_id": self.user_id,
            "target_index": self.target_index,
            "has_stage2": self.has_stage2,
            # VERIFIED
            "validated_goal_concepts": v.validated_goal_concepts,
            "context_goals": v.context_goals,
            "deviation_reason": v.deviation_reason,
            "confidence": v.confidence,
            "ttl_steps": v.ttl_steps,
            "persona_alignment_score": v.persona_alignment_score,
            "contrast_with_persona": v.contrast_with_persona,
            "temporal_cues": v.temporal_cues,
            "evidence_sources": v.evidence_sources,
            "evidence_item_ids": v.evidence_item_ids,
            "support_items": v.support_items,
            "constraints": v.constraints,
            "current_source": v.current_source,
            # LLM_UNVERIFIED
            "token_usage": uv.token_usage,
            # AUDIT
            "llm_explanation_short": a.llm_explanation_short,
            "why_not_aligned": a.why_not_aligned,
            "why_exploration": a.why_exploration,
            "llm_raw": a.llm_raw,
            "raw_model_response_json": a.raw_model_response_json,
            "removed_non_semantic_goals": a.removed_non_semantic_goals,
            "goal_hygiene_status": a.goal_hygiene_status,
            "non_semantic_goal_leakage": a.non_semantic_goal_leakage,
            "semantic_signal_absent": a.semantic_signal_absent,
            "source_mode": a.source_mode,
            "reason_source": a.reason_source,
            "pre_grounding_goal_text": a.pre_grounding_goal_text,
            "raw_llm_goals": a.raw_llm_goals,
            "grounding_diagnostics": a.grounding_diagnostics,
            "llm_prompt_version": a.llm_prompt_version,
            "schema_version": a.schema_version,
        }

    def to_audit_export(self) -> dict:
        """
        Compact export for offline interpretation audit.

        Contains all information needed to evaluate interpretation quality:
          - Identity: user_id, target_index
          - SCORING decision: deviation_reason, confidence, validated goals
          - VERIFIED contrast/temporal: grounded entries only
          - LLM contrast/temporal: unverified entries (for comparison)
          - Evidence: evidence_sources
          - AUDIT rationale: llm_explanation_short, why_not_aligned, why_exploration
          - Hygiene: goal_hygiene_status, non_semantic_goal_leakage
          - Token usage: system/user/response/total tokens
          - Versions: llm_prompt_version, schema_version
        """
        v, uv, a = self.verified, self.unverified, self.audit
        return {
            # Identity
            "user_id": self.user_id,
            "target_index": self.target_index,
            "has_stage2": self.has_stage2,
            # Scoring decision
            "deviation_reason": v.deviation_reason,
            "confidence": round(v.confidence, 4),
            "validated_goal_concepts": v.validated_goal_concepts,
            "context_goals": v.context_goals,
            # Contrast — verified vs unverified side-by-side
            "contrast_verified": v.verified_contrast(),          # {concept_id: int}
            "contrast_llm_only": uv.llm_contrast_concepts,       # [concept_id, ...]
            # Temporal — verified cues vs LLM phrase
            "temporal_shift_detected": v.shift_detected(),
            "temporal_first_half_dominant": v.temporal_cues.get("first_half_dominant"),
            "temporal_second_half_dominant": v.temporal_cues.get("second_half_dominant"),
            "temporal_llm_summary": uv.llm_shift_summary,        # audit phrase
            # Evidence
            "evidence_sources": v.evidence_sources,
            # Audit rationale
            "llm_explanation_short": a.llm_explanation_short,
            "why_not_aligned": a.why_not_aligned,
            "why_exploration": a.why_exploration,
            # Hygiene
            "goal_hygiene_status": a.goal_hygiene_status,
            "non_semantic_goal_leakage": a.non_semantic_goal_leakage,
            "semantic_signal_absent": a.semantic_signal_absent,
            "raw_llm_goals": a.raw_llm_goals,
            # Token usage
            "token_usage": uv.token_usage,
            # Versions
            "llm_prompt_version": a.llm_prompt_version,
            "schema_version": a.schema_version,
        }

    def to_audit_export_json(self) -> str:
        return json.dumps(self.to_audit_export(), ensure_ascii=False)
