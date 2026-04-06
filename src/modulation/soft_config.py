"""
Configuration helpers for candidate-aware soft scorer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonSoftPolicy:
    goal_weight: float
    persona_weight: float
    reason_prior: float
    max_bonus: float
    allow_soft: bool


def default_reason_policies() -> dict[str, ReasonSoftPolicy]:
    # Small-magnitude priors by design.
    return {
        "aligned": ReasonSoftPolicy(
            goal_weight=0.35,
            persona_weight=0.65,
            reason_prior=0.85,
            max_bonus=0.045,
            allow_soft=True,
        ),
        "exploration": ReasonSoftPolicy(
            goal_weight=0.70,
            persona_weight=0.30,
            reason_prior=1.00,
            max_bonus=0.060,
            allow_soft=True,
        ),
        "task_focus": ReasonSoftPolicy(
            goal_weight=0.75,
            persona_weight=0.25,
            reason_prior=0.90,
            max_bonus=0.050,
            allow_soft=True,
        ),
        "budget_shift": ReasonSoftPolicy(
            goal_weight=0.10,
            persona_weight=0.10,
            reason_prior=0.40,
            max_bonus=0.010,
            allow_soft=False,
        ),
        "unknown": ReasonSoftPolicy(
            goal_weight=0.10,
            persona_weight=0.10,
            reason_prior=0.35,
            max_bonus=0.008,
            allow_soft=False,
        ),
    }

