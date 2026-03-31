"""
ranking_eval.py
---------------
Core evaluation logic: reranked_results + ground_truth -> per-user metric rows.

Ground truth is derived from user_sequences:
    target item = item_sequence[target_index]
"""

from __future__ import annotations

import logging

import pandas as pd

from src.evaluation.metrics import compute_all, aggregate

logger = logging.getLogger(__name__)


def build_ground_truth(
    df_sequences: pd.DataFrame,
    eval_snaps: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """
    Returns {(user_id, target_index): target_item_id}
    using eval_snaps (last snapshot per user) and full sequences.
    """
    user_seq = df_sequences.set_index("user_id")
    gt: dict[tuple[str, int], str] = {}
    for _, snap in eval_snaps.iterrows():
        uid = snap["user_id"]
        tidx = int(snap["target_index"])
        full_seq = list(user_seq.loc[uid, "item_sequence"])
        if tidx < len(full_seq):
            gt[(uid, tidx)] = full_seq[tidx]
    return gt


def evaluate_results(
    df_results: pd.DataFrame,
    ground_truth: dict[tuple[str, int], str],
    k_values: list[int],
    score_col: str = "final_score",
) -> tuple[list[dict], dict[str, float]]:
    """
    Evaluate reranked results against ground truth.

    Returns
    -------
    per_user_rows : list of per-user metric dicts (for DataFrame)
    aggregated    : mean metrics across users
    """
    # Build GT lookup as DataFrame for vectorized merge
    df_gt = pd.DataFrame(
        [(uid, tidx, iid) for (uid, tidx), iid in ground_truth.items()],
        columns=["user_id", "target_index", "_gt_item"],
    )

    # Pre-sort all results; groupby preserves sort order within groups
    df_sorted = df_results.sort_values(
        ["user_id", "target_index", score_col], ascending=[True, True, False]
    )

    # Merge GT into results once
    df_sorted = df_sorted.merge(df_gt, on=["user_id", "target_index"], how="left")

    per_user_rows = []

    for (uid, tidx), sub in df_sorted.groupby(["user_id", "target_index"], sort=False):
        target = sub["_gt_item"].iloc[0]
        if pd.isna(target):
            continue

        ranked_items = sub["candidate_item_id"].tolist()
        m = compute_all(ranked_items, target, k_values)

        gt_row = sub[sub["candidate_item_id"] == target]
        rank_after  = int(gt_row["rank_after"].iloc[0])  if (not gt_row.empty and "rank_after"  in sub.columns) else None
        rank_before = int(gt_row["rank_before"].iloc[0]) if (not gt_row.empty and "rank_before" in sub.columns) else None

        reason = sub["deviation_reason"].iloc[0] if "deviation_reason" in sub.columns else "unknown"
        gate   = float(sub["gate_strength"].iloc[0]) if "gate_strength" in sub.columns else 0.0

        row = {
            "user_id": uid, "target_index": tidx, "target_item": target,
            "deviation_reason": reason, "gate_strength": gate,
            "rank_before": rank_before, "rank_after": rank_after,
            "gt_in_candidates": target in ranked_items,
        }
        row.update(m)
        per_user_rows.append(row)

    agg = aggregate([{k: v for k, v in r.items() if isinstance(v, float)} for r in per_user_rows])
    return per_user_rows, agg
