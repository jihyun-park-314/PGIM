"""
Time-based exponential decay for persona node weights.

weight_decayed = weight * 0.5 ^ (elapsed_days / half_life_days)

All timestamps are Unix seconds (int).
"""

import math
from typing import Union

_SECONDS_PER_DAY = 86_400


def decay_weight(
    weight: float,
    event_ts: int,
    reference_ts: int,
    half_life_days: float,
) -> float:
    """
    Apply exponential decay to a single weight value.

    Parameters
    ----------
    weight        : original weight
    event_ts      : timestamp of the event that created/updated this weight (seconds)
    reference_ts  : "now" reference point (seconds) — typically the cutoff timestamp
    half_life_days: days for weight to halve
    """
    if half_life_days <= 0:
        return weight
    elapsed_days = max(0, (reference_ts - event_ts)) / _SECONDS_PER_DAY
    decay_factor = math.pow(0.5, elapsed_days / half_life_days)
    return weight * decay_factor


def decay_weights_batch(
    weights: list[float],
    event_timestamps: list[int],
    reference_ts: int,
    half_life_days: float,
) -> list[float]:
    """Apply decay to a list of (weight, timestamp) pairs."""
    return [
        decay_weight(w, ts, reference_ts, half_life_days)
        for w, ts in zip(weights, event_timestamps)
    ]
