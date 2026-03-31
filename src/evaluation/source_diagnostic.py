"""
source_diagnostic.py
--------------------
Diagnostic: split evaluation by target source_service (rec vs search).

For each backbone experiment result, compute:
    - gt_coverage, HR@K, NDCG@K, MRR
    separately for rec-target and search-target users.

Also produces a summary of target source distribution.

Output:
    source_split_metrics.csv
    source_split_diagnostics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import compute_all, aggregate
from src.evaluation.ranking_eval import build_ground_truth

logger = logging.getLogger(__name__)


def _build_target_source_map(
    df_interactions: pd.DataFrame,
    eval_snaps: pd.DataFrame,
) -> dict[tuple[str, int], str]:
    """
    Returns {(user_id, target_index): source_service}
    by looking up the interaction that matches the GT item at target_index.
    """
    user_seq_items: dict[str, list[str]] = {}
    user_seq_sources: dict[str, list[str]] = {}

    df_sorted = df_interactions.sort_values(["user_id", "timestamp"])
    for uid, grp in df_sorted.groupby("user_id"):
        user_seq_items[uid] = grp["item_id"].tolist()
        user_seq_sources[uid] = grp["source_service"].tolist() if "source_service" in grp.columns else []

    target_source: dict[tuple[str, int], str] = {}
    for _, snap in eval_snaps.iterrows():
        uid = snap["user_id"]
        tidx = int(snap["target_index"])
        sources = user_seq_sources.get(uid, [])
        if tidx < len(sources):
            target_source[(uid, tidx)] = sources[tidx]
        else:
            target_source[(uid, tidx)] = "unknown"
    return target_source


def _eval_subset(
    df_res: pd.DataFrame,
    gt: dict[tuple[str, int], str],
    user_keys: set[tuple[str, int]],
    k_values: list[int],
    score_col: str = "final_score",
) -> tuple[float, dict[str, float]]:
    """Evaluate a subset of (user_id, target_index) keys. Returns gt_coverage and aggregated metrics."""
    # Filter result df to subset keys
    df_gt_sub = pd.DataFrame(
        [(uid, tidx, iid) for (uid, tidx), iid in gt.items() if (uid, tidx) in user_keys],
        columns=["user_id", "target_index", "_gt_item"],
    )
    if df_gt_sub.empty:
        return 0.0, {}

    df_sub = df_res.merge(df_gt_sub, on=["user_id", "target_index"], how="inner")
    df_sub = df_sub.sort_values(["user_id", "target_index", score_col], ascending=[True, True, False])

    per_user_rows = []
    for (uid, tidx), group in df_sub.groupby(["user_id", "target_index"], sort=False):
        target = group["_gt_item"].iloc[0]
        if pd.isna(target):
            continue
        ranked_items = group["candidate_item_id"].tolist()
        m = compute_all(ranked_items, target, k_values)
        m["gt_in_candidates"] = float(target in ranked_items)
        per_user_rows.append(m)

    if not per_user_rows:
        return 0.0, {}

    gt_coverage = sum(r["gt_in_candidates"] for r in per_user_rows) / len(per_user_rows)
    agg = aggregate([{k: v for k, v in r.items() if k != "gt_in_candidates"} for r in per_user_rows])
    return gt_coverage, agg


def run_source_diagnostic(
    experiment_names: list[str],
    eval_dir: Path,
    df_sequences: pd.DataFrame,
    df_snaps: pd.DataFrame,
    df_interactions: pd.DataFrame,
    k_values: list[int],
    out_dir: Path,
) -> None:
    """
    Main entry point for source-split diagnostic.
    """
    # ── eval split (last snapshot per user) ──────────────────────────
    eval_snaps = df_snaps.loc[
        df_snaps.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)

    gt = build_ground_truth(df_sequences, eval_snaps)
    logger.info("Ground truth: %d users", len(gt))

    # ── target source distribution ───────────────────────────────────
    target_source = _build_target_source_map(df_interactions, eval_snaps)

    rec_keys    = {k for k, v in target_source.items() if v == "rec"}
    search_keys = {k for k, v in target_source.items() if v == "search"}
    all_keys    = set(gt.keys())
    total       = len(all_keys)

    logger.info("Target source distribution:")
    logger.info("  total   : %d", total)
    logger.info("  rec     : %d (%.1f%%)", len(rec_keys),    100 * len(rec_keys) / total)
    logger.info("  search  : %d (%.1f%%)", len(search_keys), 100 * len(search_keys) / total)
    logger.info("  unknown : %d", total - len(rec_keys) - len(search_keys))

    source_dist = {
        "total": total,
        "rec": len(rec_keys),
        "search": len(search_keys),
        "unknown": total - len(rec_keys) - len(search_keys),
        "rec_pct": round(100 * len(rec_keys) / total, 2),
        "search_pct": round(100 * len(search_keys) / total, 2),
    }

    # ── per-experiment, per-subset evaluation ────────────────────────
    metric_rows = []
    diag = {"source_distribution": source_dist, "experiments": {}}

    subsets = [
        ("all",    all_keys),
        ("rec",    rec_keys),
        ("search", search_keys),
    ]

    for name in experiment_names:
        result_path = eval_dir / f"reranked_results_{name}.parquet"
        if not result_path.exists():
            logger.warning("Missing: %s — skipping", result_path)
            continue

        logger.info("Evaluating source split: %s", name)
        df_res = pd.read_parquet(result_path)
        diag["experiments"][name] = {}

        for subset_name, keys in subsets:
            if not keys:
                continue
            cov, agg = _eval_subset(df_res, gt, keys, k_values)
            row = {
                "experiment":  name,
                "subset":      subset_name,
                "n_users":     len(keys & all_keys),
                "gt_coverage": round(cov, 4),
            }
            row.update({k: round(v, 4) for k, v in agg.items()})
            metric_rows.append(row)

            diag["experiments"][name][subset_name] = {
                "n_users":     len(keys & all_keys),
                "gt_coverage": round(cov, 4),
                "HR@10":       round(agg.get("HR@10", 0), 4),
                "NDCG@10":     round(agg.get("NDCG@10", 0), 4),
                "MRR":         round(agg.get("MRR", 0), 4),
            }
            logger.info(
                "  [%s / %s]  n=%d  gt_cov=%.3f  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f",
                name, subset_name, len(keys & all_keys), cov,
                agg.get("HR@10", 0), agg.get("NDCG@10", 0), agg.get("MRR", 0),
            )

    # ── save ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "source_split_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("saved -> %s", metrics_path)

    diag_path = out_dir / "source_split_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    logger.info("saved -> %s", diag_path)

    # ── pretty print ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SOURCE SPLIT DIAGNOSTICS")
    print("=" * 70)
    print(f"\nTarget source distribution:")
    print(f"  total   : {source_dist['total']}")
    print(f"  rec     : {source_dist['rec']}  ({source_dist['rec_pct']:.1f}%)")
    print(f"  search  : {source_dist['search']}  ({source_dist['search_pct']:.1f}%)")

    print("\nMetrics by subset:")
    cols = ["experiment", "subset", "n_users", "gt_coverage",
            "HR@10", "NDCG@10", "MRR"]
    cols = [c for c in cols if c in df_metrics.columns]
    print(df_metrics[cols].to_string(index=False))
