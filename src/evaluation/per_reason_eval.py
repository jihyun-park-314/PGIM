"""
per_reason_eval.py
------------------
Reason-sliced diagnostic for sampled evaluation results.

Reads pre-computed sampled_reranked_*.parquet files and computes
per-reason metrics + delta diagnostics for each experiment.

Output:
    data/artifacts/eval/<dataset>/sampled_per_reason_metrics.csv
    data/artifacts/eval/<dataset>/sampled_per_reason_diagnostics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import aggregate, compute_all

logger = logging.getLogger(__name__)

_REASON_ORDER = ["aligned", "exploration", "task_focus", "budget_shift", "unknown"]


def _find_cand_path(eval_dir: Path) -> Path | None:
    """Infer sampled_candidates path from eval_dir structure."""
    # eval_dir = data/artifacts/eval/<dataset>
    dataset = eval_dir.name
    p = Path("data/cache/candidate") / dataset / "sampled_candidates_k101.parquet"
    if p.exists():
        return p
    # fallback: search
    for candidate in Path("data/cache/candidate").glob("**/sampled_candidates_k101.parquet"):
        return candidate
    return None


def run_per_reason_eval(
    experiment_names: list[str],
    eval_dir: Path,
    k_values: list[int],
    out_dir: Path,
) -> None:
    """
    Load sampled_reranked_*.parquet for each experiment,
    split by deviation_reason, compute metrics.
    """
    # ── load GT mapping from sampled candidates ──────────────────
    cand_path = _find_cand_path(eval_dir)
    if cand_path is None:
        logger.error("Cannot find sampled_candidates_k101.parquet. Run --build-sampled-candidates first.")
        return

    df_cand = pd.read_parquet(cand_path)
    gt_by_key: dict[tuple[str, int], str] = {
        (row.user_id, int(row.target_index)): row.candidate_item_id
        for row in df_cand[df_cand["is_ground_truth"] == True].itertuples(index=False)
    }
    logger.info("GT map: %d entries", len(gt_by_key))

    df_gt = pd.DataFrame(
        [(uid, tidx, iid) for (uid, tidx), iid in gt_by_key.items()],
        columns=["user_id", "target_index", "_gt_item"],
    )

    # ── collect all ranked dataframes ───────────────────────────
    dfs: dict[str, pd.DataFrame] = {}
    for name in experiment_names:
        path = eval_dir / f"sampled_reranked_{name}.parquet"
        if not path.exists():
            logger.warning("Missing: %s — skipping", path)
            continue
        dfs[name] = pd.read_parquet(path)
        logger.info("Loaded %s: %d rows", name, len(dfs[name]))

    if not dfs:
        logger.error("No ranked result files found in %s", eval_dir)
        return

    # ── collect reasons ──────────────────────────────────────────
    first_df = next(iter(dfs.values()))
    if "deviation_reason" not in first_df.columns:
        logger.error("deviation_reason column missing from ranked results")
        return

    snap_df = first_df.drop_duplicates(subset=["user_id", "target_index"])
    reason_counts = snap_df.groupby("deviation_reason").size().to_dict()
    logger.info("Reason distribution: %s", reason_counts)

    all_reasons = sorted(first_df["deviation_reason"].unique().tolist())
    reasons_ordered = [r for r in _REASON_ORDER if r in all_reasons]
    reasons_ordered += [r for r in all_reasons if r not in _REASON_ORDER]

    # ── main evaluation loop ─────────────────────────────────────
    metric_rows: list[dict] = []
    diag: dict = {"reason_distribution": reason_counts, "experiments": {}}

    for name, df in dfs.items():
        diag["experiments"][name] = {}

        # Merge GT into df (inner join — only users with GT)
        df_merged = df.merge(df_gt, on=["user_id", "target_index"], how="inner")
        df_sorted = df_merged.sort_values(
            ["user_id", "target_index", "final_score"], ascending=[True, True, False]
        )

        for reason in reasons_ordered:
            df_r = df_sorted[df_sorted["deviation_reason"] == reason]
            if df_r.empty:
                continue

            per_user_rows: list[dict] = []
            for (uid, tidx), sub in df_r.groupby(["user_id", "target_index"], sort=False):
                gt_item = sub["_gt_item"].iloc[0]
                ranked_items = sub["candidate_item_id"].tolist()
                m = compute_all(ranked_items, gt_item, k_values)
                m["gt_in_candidates"] = float(gt_item in ranked_items)

                gt_row = sub[sub["candidate_item_id"] == gt_item]
                m["rank_after"]  = int(gt_row["rank_after"].iloc[0])  if not gt_row.empty else None
                m["rank_before"] = int(gt_row["rank_before"].iloc[0]) if not gt_row.empty else None
                per_user_rows.append(m)

            n_users = len(per_user_rows)
            if n_users == 0:
                continue

            agg = aggregate([
                {k: v for k, v in r.items()
                 if isinstance(v, float) and k not in ("gt_in_candidates",)}
                for r in per_user_rows
            ])

            # rank movement
            df_u = pd.DataFrame(per_user_rows)
            has_rank = df_u["rank_before"].notna() & df_u["rank_after"].notna()
            gt_found = df_u[has_rank & (df_u["gt_in_candidates"] == 1.0)]
            improved = int((gt_found["rank_after"] < gt_found["rank_before"]).sum())
            same     = int((gt_found["rank_after"] == gt_found["rank_before"]).sum())
            worsened = int((gt_found["rank_after"] > gt_found["rank_before"]).sum())

            # delta stats
            delta_stats: dict = {}
            if "modulation_delta" in df_r.columns:
                gt_mask = df_r["candidate_item_id"] == df_r["_gt_item"]
                gt_deltas = df_r.loc[gt_mask, "modulation_delta"]
                delta_stats = {
                    "all_nonzero_frac":       round(float((df_r["modulation_delta"] != 0).mean()), 4),
                    "gt_delta_mean":          round(float(gt_deltas.mean()), 6) if len(gt_deltas) else None,
                    "gt_delta_positive_frac": round(float((gt_deltas > 0).mean()), 4) if len(gt_deltas) else None,
                }

            gate_stats: dict = {}
            if "gate_strength" in df_r.columns:
                gs = df_r["gate_strength"]
                gate_stats = {
                    "mean": round(float(gs.mean()), 4),
                    "std":  round(float(gs.std()), 4),
                }

            gt_cov = sum(r["gt_in_candidates"] for r in per_user_rows) / n_users

            row = {
                "experiment":   name,
                "reason":       reason,
                "n_users":      n_users,
                "gt_coverage":  round(gt_cov, 4),
            }
            row.update({k: round(v, 4) for k, v in agg.items()})
            metric_rows.append(row)

            diag["experiments"][name][reason] = {
                "n_users":       n_users,
                "gt_coverage":   round(gt_cov, 4),
                "HR@10":         round(agg.get("HR@10", 0), 4),
                "NDCG@10":       round(agg.get("NDCG@10", 0), 4),
                "MRR":           round(agg.get("MRR", 0), 4),
                "rank_movement": {"improved": improved, "same": same, "worsened": worsened},
                "delta_stats":   delta_stats,
                "gate_stats":    gate_stats,
            }

            logger.info(
                "  [%s / %s]  n=%d  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  "
                "imp=%d  wor=%d  gt_delta=%s",
                name, reason, n_users,
                agg.get("HR@10", 0), agg.get("NDCG@10", 0), agg.get("MRR", 0),
                improved, worsened,
                delta_stats.get("gt_delta_mean"),
            )

    # ── save ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "sampled_per_reason_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info("saved -> %s", metrics_path)

    diag_path = out_dir / "sampled_per_reason_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    logger.info("saved -> %s", diag_path)

    # ── pretty print ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-REASON EVALUATION")
    print("=" * 80)
    print(f"\nReason distribution: {reason_counts}\n")

    cols = ["experiment", "reason", "n_users", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR"]
    cols = [c for c in cols if c in df_metrics.columns]

    for reason in reasons_ordered:
        sub = df_metrics[df_metrics["reason"] == reason]
        if sub.empty:
            continue
        n = int(sub["n_users"].iloc[0]) if "n_users" in sub.columns else "?"
        print(f"── reason: {reason}  (n_users={n}) ──────────────────────────")
        print(sub[cols].to_string(index=False))
        print("  Delta/gate diagnostics:")
        for _, r in sub.iterrows():
            exp = r["experiment"]
            d = diag["experiments"].get(exp, {}).get(reason, {})
            ds = d.get("delta_stats", {})
            mv = d.get("rank_movement", {})
            gs = d.get("gate_stats", {})
            gate_str  = f"gate={gs.get('mean','?'):.3f}±{gs.get('std','?'):.3f}  " if gs else ""
            delta_str = f"gt_delta={ds.get('gt_delta_mean','?')}  "
            pos_str   = f"+frac={ds.get('gt_delta_positive_frac','?')}  "
            mv_str    = f"imp={mv.get('improved',0)}  wor={mv.get('worsened',0)}"
            print(f"    [{exp:30s}]  {gate_str}{delta_str}{pos_str}{mv_str}")
        print()
