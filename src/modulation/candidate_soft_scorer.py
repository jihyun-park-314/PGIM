"""
Thin candidate-aware soft scorer.

Designed as a small additive bonus over existing base delta.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.modulation.soft_config import ReasonSoftPolicy, default_reason_policies
from src.modulation.soft_features import SoftMatchFeatures


@dataclass(frozen=True)
class SoftScoreResult:
    soft_candidate_bonus: float
    soft_match_strength: float
    dominant_soft_signal_type: str
    matched_goal_count_soft: int
    matched_persona_count_soft: int


def compute_soft_features(
    candidate_semantic_core: list[str],
    candidate_semantic_anchor: list[str],
    validated_goals: list[str],
    persona_weighted_map: dict[str, float],
) -> SoftMatchFeatures:
    cand_set = set(candidate_semantic_core) | set(candidate_semantic_anchor)
    goal_set = set(validated_goals)
    matched_goal = cand_set & goal_set

    if goal_set:
        goal_overlap_ratio = len(matched_goal) / len(goal_set)
    else:
        goal_overlap_ratio = 0.0

    matched_persona = cand_set & set(persona_weighted_map.keys())
    persona_overlap_weighted = sum(persona_weighted_map[c] for c in matched_persona) if matched_persona else 0.0
    semantic_density = min(1.0, len(cand_set) / 12.0)

    return SoftMatchFeatures(
        goal_overlap_ratio=float(goal_overlap_ratio),
        persona_overlap_weighted=float(persona_overlap_weighted),
        matched_goal_count=int(len(matched_goal)),
        matched_persona_count=int(len(matched_persona)),
        semantic_density=float(semantic_density),
    )


def compute_candidate_soft_bonus(
    reason: str,
    confidence: float,
    base_delta: float,
    features: SoftMatchFeatures,
    lambda_soft: float,
    policy_override: dict[str, ReasonSoftPolicy] | None = None,
    disable_persona: bool = False,
    disable_goal: bool = False,
    exploration_only: bool = False,
) -> SoftScoreResult:
    policies = policy_override or default_reason_policies()
    policy = policies.get(reason, policies["unknown"])

    if exploration_only and reason != "exploration":
        return SoftScoreResult(0.0, 0.0, "none", 0, 0)
    if not policy.allow_soft and reason not in {"exploration", "aligned", "task_focus"}:
        return SoftScoreResult(0.0, 0.0, "none", 0, 0)

    g_comp = 0.0 if disable_goal else features.goal_overlap_ratio
    p_comp = 0.0 if disable_persona else features.persona_overlap_weighted

    raw_strength = (
        policy.goal_weight * g_comp
        + policy.persona_weight * p_comp
    )
    raw_strength *= policy.reason_prior
    raw_strength *= (0.5 + 0.5 * max(0.0, min(1.0, confidence)))
    raw_strength *= (0.75 + 0.25 * features.semantic_density)

    # Keep influence small and bounded.
    bounded_bonus = min(policy.max_bonus, max(0.0, raw_strength))

    # If baseline delta already negative, avoid counteracting with large positive soft bonus.
    if base_delta < 0:
        bounded_bonus *= 0.35

    soft_bonus = float(lambda_soft * bounded_bonus)

    if g_comp > p_comp and g_comp > 0:
        dom = "goal_match"
    elif p_comp > g_comp and p_comp > 0:
        dom = "persona_match"
    elif g_comp > 0 or p_comp > 0:
        dom = "blended"
    else:
        dom = "none"

    return SoftScoreResult(
        soft_candidate_bonus=soft_bonus,
        soft_match_strength=float(raw_strength),
        dominant_soft_signal_type=dom,
        matched_goal_count_soft=features.matched_goal_count,
        matched_persona_count_soft=features.matched_persona_count,
    )

