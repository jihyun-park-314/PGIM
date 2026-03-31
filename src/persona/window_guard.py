"""
WindowGuard: exclude the evaluation window from persona construction.

Two supported modes (controlled by persona config):
  "tail_n"           — exclude the last N interactions per user
  "after_timestamp"  — exclude interactions at or after a given timestamp

The guard is applied BEFORE PersonaGraphBuilder sees the sequence,
so the persona graph never leaks eval-window items.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def exclude_eval_window(
    item_sequence: list[str],
    timestamp_sequence: list[int],
    mode: str = "tail_n",
    tail_n: int = 1,
    cutoff_ts: Optional[int] = None,
) -> tuple[list[str], list[int]]:
    """
    Return (items, timestamps) with the eval window removed.

    Parameters
    ----------
    mode         : "tail_n" or "after_timestamp"
    tail_n       : used when mode="tail_n"; exclude last N items
    cutoff_ts    : used when mode="after_timestamp"; exclude ts >= cutoff_ts

    Raises
    ------
    ValueError   : if the resulting sequence is empty
    """
    if mode == "tail_n":
        if tail_n <= 0:
            return item_sequence, timestamp_sequence
        items = item_sequence[:-tail_n]
        timestamps = timestamp_sequence[:-tail_n]
    elif mode == "after_timestamp":
        if cutoff_ts is None:
            raise ValueError("cutoff_ts required for mode='after_timestamp'")
        pairs = [(i, t) for i, t in zip(item_sequence, timestamp_sequence) if t < cutoff_ts]
        if not pairs:
            items, timestamps = [], []
        else:
            items, timestamps = zip(*pairs)
            items, timestamps = list(items), list(timestamps)
    else:
        raise ValueError(f"Unknown eval_exclusion mode: {mode}")

    if len(items) == 0:
        raise ValueError("Sequence is empty after eval window exclusion.")

    return list(items), list(timestamps)


def assert_no_leakage(
    persona_item_ids: set[str],
    eval_item_ids: set[str],
) -> None:
    """
    Assert that no eval-window items appear in persona item set.
    Call this after building a persona graph to catch leakage bugs.
    """
    overlap = persona_item_ids & eval_item_ids
    if overlap:
        raise AssertionError(
            f"Eval window leakage detected! {len(overlap)} items in both "
            f"persona history and eval window: {list(overlap)[:5]}"
        )
