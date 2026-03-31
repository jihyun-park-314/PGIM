"""
gate.py
-------
Computes gate_strength: how strongly modulation should be applied.

Formula:
    gate_strength = reason_base_weight * (confidence ^ confidence_power)
                    * max(alignment_factor, alignment_clip_min)

Where:
    alignment_factor = persona_alignment_score (0~1)
    For "exploration": alignment being LOW is expected, so we don't penalize it.
    For "aligned": alignment being HIGH boosts the gate.

All values clamped to [0, 1].
"""

from __future__ import annotations


def compute_gate_strength(
    deviation_reason: str,
    confidence: float,
    persona_alignment_score: float,
    gate_cfg: dict,
) -> float:
    """
    Returns gate_strength in [0, 1].

    gate_cfg keys (from config/modulation/default.yaml -> gate):
        reason_base_weight: dict[reason -> float]
        confidence_power: float
        alignment_clip_min: float
    """
    reason_weights: dict = gate_cfg.get("reason_base_weight", {})
    base = reason_weights.get(deviation_reason, reason_weights.get("unknown", 0.25))

    conf_power = gate_cfg.get("confidence_power", 1.0)
    conf_factor = max(0.0, min(1.0, confidence)) ** conf_power

    clip_min = gate_cfg.get("alignment_clip_min", 0.05)

    # For exploration: alignment is expected to be low — don't penalize.
    # For aligned: alignment reinforces the gate.
    # For others: use alignment as a soft factor.
    if deviation_reason == "exploration":
        alignment_factor = 1.0  # exploration is intentionally off-persona
    else:
        alignment_factor = max(clip_min, persona_alignment_score)

    gate_strength = base * conf_factor * alignment_factor
    return round(min(1.0, max(0.0, gate_strength)), 4)
