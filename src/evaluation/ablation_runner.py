"""
ablation_runner.py
------------------
Loads multiple experiment result files, evaluates each, and produces:
    - metrics_summary.csv       (one row per experiment)
    - per_reason_metrics.csv    (reason x experiment)
    - diagnostic_summary.json   (per-experiment diagnostics)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from src.evaluation.metrics import aggregate
from src.evaluation.ranking_eval import build_ground_truth, evaluate_results

logger = logging.getLogger(__name__)

_REASON_IMBALANCE_WARN_THRESHOLD = 0.80   # warn if one reason > 80% of users


def run_ablation(
    experiment_names: list[str],
    eval_dir: Path,
    df_sequences: pd.DataFrame,
    df_snaps: pd.DataFrame,
    k_values: list[int],
    out_dir: Path,
) -> None:
    eval_snaps = df_snaps.loc[
        df_snaps.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)

    gt = build_ground_truth(df_sequences, eval_snaps)
    logger.info("Ground truth: %d users", len(gt))

    summary_rows: list[dict] = []
    per_reason_rows: list[dict] = []
    diagnostics: dict = {}

    for name in experiment_names:
        result_path = eval_dir / f"reranked_results_{name}.parquet"
        if not result_path.exists():
            logger.warning("Missing result file: %s — skipping", result_path)
            continue

        logger.info("Evaluating: %s", name)
        df_res = pd.read_parquet(result_path)

        per_user, agg = evaluate_results(df_res, gt, k_values)
        df_user = pd.DataFrame(per_user)

        # ── summary row ──────────────────────────────────────────
        row = {"experiment": name}
        row.update({k: round(v, 4) for k, v in agg.items() if k not in ("gate_strength",)})
        row["gt_coverage"] = round(df_user["gt_in_candidates"].mean(), 4)
        row["n_users"] = len(df_user)
        summary_rows.append(row)

        # ── per-reason metrics ───────────────────────────────────
        reasons = df_user["deviation_reason"].unique()
        for reason in reasons:
            sub = df_user[df_user["deviation_reason"] == reason]
            r_agg = aggregate([
                {k: v for k, v in r.items() if isinstance(v, float) and k not in ("gate_strength",)}
                for r in sub.to_dict("records")
            ])
            r_row = {"experiment": name, "deviation_reason": reason, "n_users": len(sub)}
            r_row.update({k: round(v, 4) for k, v in r_agg.items()})
            per_reason_rows.append(r_row)

        # ── diagnostics ──────────────────────────────────────────
        has_rank = df_user["rank_before"].notna() & df_user["rank_after"].notna()
        gt_found = df_user[has_rank & df_user["gt_in_candidates"]]

        improved  = int((gt_found["rank_after"] < gt_found["rank_before"]).sum())
        same      = int((gt_found["rank_after"] == gt_found["rank_before"]).sum())
        worsened  = int((gt_found["rank_after"] > gt_found["rank_before"]).sum())

        reason_dist = df_user["deviation_reason"].value_counts().to_dict()
        dominant_reason_frac = max(reason_dist.values()) / len(df_user) if reason_dist else 0.0

        diag = {
            "n_users": len(df_user),
            "gt_coverage": round(df_user["gt_in_candidates"].mean(), 4),
            "reason_distribution": reason_dist,
            "reason_imbalance_warning": dominant_reason_frac >= _REASON_IMBALANCE_WARN_THRESHOLD,
            "dominant_reason_fraction": round(dominant_reason_frac, 4),
            "rank_movement": {
                "improved": improved,
                "same": same,
                "worsened": worsened,
                "gt_in_top100": len(gt_found),
            },
        }

        # delta stats (from result df, only GT items)
        if "modulation_delta" in df_res.columns:
            df_gt_map = pd.DataFrame(
                [(uid, tidx, iid) for (uid, tidx), iid in gt.items()],
                columns=["user_id", "target_index", "_gt_item"],
            )
            df_gt_map["target_index"] = df_gt_map["target_index"].astype(df_res["target_index"].dtype)
            df_merged = df_res.merge(df_gt_map, on=["user_id", "target_index"], how="left")
            gt_items_mask = df_merged["candidate_item_id"] == df_merged["_gt_item"]
            gt_deltas = df_res.loc[gt_items_mask.values, "modulation_delta"]
            all_deltas = df_res["modulation_delta"]
            diag["delta_stats"] = {
                "all_nonzero_frac": round((all_deltas != 0).mean(), 4),
                "all_mean": round(float(all_deltas.mean()), 6),
                "all_std": round(float(all_deltas.std()), 6),
                "gt_item_mean_delta": round(float(gt_deltas.mean()), 6) if len(gt_deltas) else None,
                "gt_item_positive_frac": round((gt_deltas > 0).mean(), 4) if len(gt_deltas) else None,
            }

        # gate stats (from signals file if available)
        sig_path = eval_dir / f"modulation_signals_{name}.parquet"
        if sig_path.exists():
            df_sig = pd.read_parquet(sig_path)
            diag["gate_stats"] = {
                "mean": round(float(df_sig["gate_strength"].mean()), 4),
                "std": round(float(df_sig["gate_strength"].std()), 4),
                "min": round(float(df_sig["gate_strength"].min()), 4),
                "max": round(float(df_sig["gate_strength"].max()), 4),
            }

        diagnostics[name] = diag

        if diag["reason_imbalance_warning"]:
            dominant = max(reason_dist, key=reason_dist.get)
            logger.warning(
                "[%s] REASON IMBALANCE: '%s' = %.1f%% of users. "
                "Metrics may be dominated by a single reason pattern.",
                name, dominant, 100 * dominant_reason_frac,
            )

    # ── save outputs ─────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("saved -> %s", summary_path)

    if per_reason_rows:
        df_reason = pd.DataFrame(per_reason_rows)
        reason_path = out_dir / "per_reason_metrics.csv"
        df_reason.to_csv(reason_path, index=False)
        logger.info("saved -> %s", reason_path)

    diag_path = out_dir / "diagnostic_summary.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("saved -> %s", diag_path)

    # ── print comparison table ────────────────────────────────────
    _print_comparison(df_summary, k_values)


def _print_comparison(df: pd.DataFrame, k_values: list[int]) -> None:
    if df.empty:
        return
    cols = ["experiment"] + [f"HR@{k}" for k in k_values] + \
           [f"NDCG@{k}" for k in k_values] + ["MRR", "gt_coverage"]
    cols = [c for c in cols if c in df.columns]
    logger.info("\n%s", df[cols].to_string(index=False))
