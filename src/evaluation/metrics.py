"""
metrics.py
----------
Ranking metrics for leave-one-out evaluation.

All functions take:
    ranked_items : list of item_ids in predicted order (index 0 = rank 1)
    ground_truth : the single correct item_id

Ground truth not in ranked_items -> all metrics 0 for that user.
"""

from __future__ import annotations

import math


def hit_at_k(ranked_items: list[str], ground_truth: str, k: int) -> float:
    return 1.0 if ground_truth in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: list[str], ground_truth: str, k: int) -> float:
    if ground_truth not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(ground_truth) + 1   # 1-based
    return 1.0 / math.log2(rank + 1)


def mrr(ranked_items: list[str], ground_truth: str) -> float:
    if ground_truth not in ranked_items:
        return 0.0
    rank = ranked_items.index(ground_truth) + 1
    return 1.0 / rank


def compute_all(
    ranked_items: list[str],
    ground_truth: str,
    k_values: list[int],
) -> dict[str, float]:
    result = {}
    for k in k_values:
        result[f"HR@{k}"]   = hit_at_k(ranked_items, ground_truth, k)
        result[f"NDCG@{k}"] = ndcg_at_k(ranked_items, ground_truth, k)
    result["MRR"] = mrr(ranked_items, ground_truth)
    return result


def aggregate(per_user_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average per-user metrics across all users."""
    if not per_user_metrics:
        return {}
    keys = per_user_metrics[0].keys()
    return {k: sum(d[k] for d in per_user_metrics) / len(per_user_metrics) for k in keys}
