"""
tuning_sweep.py
---------------
Aggregates tuning sweep results into 4 summary tables.

Reads per-tuning-config outputs (sampled_metrics_summary.csv,
sampled_diagnostics.json, sampled_reranked_*.parquet,
concept_usage_*.csv) and produces:

  Table 1: tuning_tag | experiment | HR@10 | NDCG@10 | MRR
  Table 2: tuning_tag | reason     | experiment | HR@10 | NDCG@10 | MRR | n_users
  Table 3: tuning_tag | experiment | gt_delta_mean | gt_delta_positive_frac | improved | worsened
  Table 4: tuning_tag | experiment | concept_type  | count | fraction   (concept type usage)

Output files:
  data/artifacts/eval/<dataset>/tuning_sweep_metrics.csv
  data/artifacts/eval/<dataset>/tuning_sweep_per_reason_metrics.csv
  data/artifacts/eval/<dataset>/tuning_sweep_diagnostics.json
  data/artifacts/eval/<dataset>/granularity_concept_usage.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import aggregate, compute_all
from src.evaluation.per_reason_eval import _REASON_ORDER

logger = logging.getLogger(__name__)

_EXPERIMENTS = ["ablation_persona_only", "ablation_intent_only", "full_model"]


def aggregate_sweep(
    cfg_names: list[str],
    eval_dir: Path,
    k_values: list[int],
    out_dir: Path,
) -> None:
    """
    Read per-config output dirs and build unified sweep tables.
    """
    # ── load GT map ──────────────────────────────────────────────
    dataset = eval_dir.name
    cand_path = Path("data/cache/candidate") / dataset / "sampled_candidates_k101.parquet"
    if not cand_path.exists():
        for p in Path("data/cache/candidate").glob("**/sampled_candidates_k101.parquet"):
            cand_path = p
            break

    df_cand = pd.read_parquet(cand_path)
    gt_by_key: dict[tuple[str, int], str] = {
        (row.user_id, int(row.target_index)): row.candidate_item_id
        for row in df_cand[df_cand["is_ground_truth"] == True].itertuples(index=False)
    }
    df_gt = pd.DataFrame(
        [(uid, tidx, iid) for (uid, tidx), iid in gt_by_key.items()],
        columns=["user_id", "target_index", "_gt_item"],
    )

    # ── collect results ──────────────────────────────────────────
    table1_rows: list[dict] = []   # overall metrics
    table2_rows: list[dict] = []   # per-reason metrics
    table3_rows: list[dict] = []   # delta diagnostics
    table4_rows: list[dict] = []   # concept type usage
    sweep_diag: dict = {}

    for cfg_name in cfg_names:
        tune_dir = eval_dir / f"tuning_{cfg_name}"

        # ── Table 1: read sampled_metrics_summary.csv ────────────
        summary_path = tune_dir / "sampled_metrics_summary.csv"
        if summary_path.exists():
            df_s = pd.read_csv(summary_path)
            for _, r in df_s.iterrows():
                row = {"tuning_tag": cfg_name, "experiment": r["experiment"]}
                for col in ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR", "n_users"]:
                    if col in r:
                        row[col] = r[col]
                table1_rows.append(row)
        else:
            logger.warning("Missing: %s", summary_path)

        # ── Table 4: concept usage ───────────────────────────────
        for exp in _EXPERIMENTS:
            usage_path = tune_dir / f"concept_usage_{exp}.csv"
            if usage_path.exists():
                df_u = pd.read_csv(usage_path)
                for _, r in df_u.iterrows():
                    table4_rows.append({
                        "tuning_tag":   cfg_name,
                        "experiment":   exp,
                        "concept_type": r["concept_type"],
                        "count":        int(r["count"]),
                        "fraction":     float(r["fraction"]),
                    })

        # ── Table 3 + Table 2: read ranked parquets ──────────────
        sweep_diag[cfg_name] = {}
        for exp in _EXPERIMENTS:
            ranked_path = tune_dir / f"sampled_reranked_{exp}.parquet"
            if not ranked_path.exists():
                logger.warning("Missing: %s", ranked_path)
                continue

            df_r = pd.read_parquet(ranked_path)
            df_merged = df_r.merge(df_gt, on=["user_id", "target_index"], how="inner")
            df_sorted = df_merged.sort_values(
                ["user_id", "target_index", "final_score"], ascending=[True, True, False]
            )

            # ── Table 3: overall delta diagnostics ───────────────
            delta_stats: dict = {}
            rank_mv: dict = {}
            if "modulation_delta" in df_sorted.columns:
                gt_mask = df_sorted["candidate_item_id"] == df_sorted["_gt_item"]
                gt_deltas = df_sorted.loc[gt_mask, "modulation_delta"]
                delta_stats = {
                    "all_nonzero_frac":       round(float((df_sorted["modulation_delta"] != 0).mean()), 4),
                    "gt_delta_mean":          round(float(gt_deltas.mean()), 6) if len(gt_deltas) else None,
                    "gt_delta_positive_frac": round(float((gt_deltas > 0).mean()), 4) if len(gt_deltas) else None,
                }

            if "rank_before" in df_sorted.columns and "rank_after" in df_sorted.columns:
                gt_rows = df_sorted[df_sorted["candidate_item_id"] == df_sorted["_gt_item"]]
                has_rank = gt_rows["rank_before"].notna() & gt_rows["rank_after"].notna()
                gr = gt_rows[has_rank]
                rank_mv = {
                    "improved": int((gr["rank_after"] < gr["rank_before"]).sum()),
                    "same":     int((gr["rank_after"] == gr["rank_before"]).sum()),
                    "worsened": int((gr["rank_after"] > gr["rank_before"]).sum()),
                }

            gate_stats: dict = {}
            if "gate_strength" in df_sorted.columns:
                gs = df_sorted["gate_strength"]
                gate_stats = {"mean": round(float(gs.mean()), 4), "std": round(float(gs.std()), 4)}

            table3_rows.append({
                "tuning_tag":             cfg_name,
                "experiment":             exp,
                "gt_delta_mean":          delta_stats.get("gt_delta_mean"),
                "gt_delta_positive_frac": delta_stats.get("gt_delta_positive_frac"),
                "all_nonzero_frac":       delta_stats.get("all_nonzero_frac"),
                "improved":               rank_mv.get("improved"),
                "worsened":               rank_mv.get("worsened"),
                "gate_mean":              gate_stats.get("mean"),
            })

            sweep_diag[cfg_name][exp] = {
                "delta_stats": delta_stats,
                "rank_movement": rank_mv,
                "gate_stats": gate_stats,
                "per_reason": {},
            }

            # ── Table 2: per-reason metrics ───────────────────────
            if "deviation_reason" not in df_sorted.columns:
                continue

            all_reasons = df_sorted["deviation_reason"].unique().tolist()
            reasons_ordered = [r for r in _REASON_ORDER if r in all_reasons]
            reasons_ordered += [r for r in all_reasons if r not in _REASON_ORDER]

            for reason in reasons_ordered:
                df_reason = df_sorted[df_sorted["deviation_reason"] == reason]
                if df_reason.empty:
                    continue

                per_user_rows: list[dict] = []
                for (uid, tidx), sub in df_reason.groupby(["user_id", "target_index"], sort=False):
                    gt_item = sub["_gt_item"].iloc[0]
                    ranked_items = sub["candidate_item_id"].tolist()
                    m = compute_all(ranked_items, gt_item, k_values)
                    m["gt_in_candidates"] = float(gt_item in ranked_items)
                    per_user_rows.append(m)

                if not per_user_rows:
                    continue

                n_users = len(per_user_rows)
                agg = aggregate([
                    {k: v for k, v in r.items()
                     if isinstance(v, float) and k != "gt_in_candidates"}
                    for r in per_user_rows
                ])

                # reason-level delta
                reason_delta: dict = {}
                if "modulation_delta" in df_reason.columns:
                    gt_mask_r = df_reason["candidate_item_id"] == df_reason["_gt_item"]
                    gt_d = df_reason.loc[gt_mask_r, "modulation_delta"]
                    reason_delta = {
                        "gt_delta_mean":          round(float(gt_d.mean()), 6) if len(gt_d) else None,
                        "gt_delta_positive_frac": round(float((gt_d > 0).mean()), 4) if len(gt_d) else None,
                    }

                r2_row = {
                    "tuning_tag": cfg_name,
                    "reason":     reason,
                    "experiment": exp,
                    "n_users":    n_users,
                }
                r2_row.update({k: round(v, 4) for k, v in agg.items()})
                r2_row.update({k: v for k, v in reason_delta.items()})
                table2_rows.append(r2_row)

                sweep_diag[cfg_name][exp]["per_reason"][reason] = {
                    "n_users":  n_users,
                    "HR@10":    round(agg.get("HR@10", 0), 4),
                    "NDCG@10":  round(agg.get("NDCG@10", 0), 4),
                    "MRR":      round(agg.get("MRR", 0), 4),
                    "delta":    reason_delta,
                }

    # ── save ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    df_t1 = pd.DataFrame(table1_rows)
    df_t2 = pd.DataFrame(table2_rows)
    df_t3 = pd.DataFrame(table3_rows)
    df_t4 = pd.DataFrame(table4_rows)

    t1_path = out_dir / "tuning_sweep_metrics.csv"
    t2_path = out_dir / "tuning_sweep_per_reason_metrics.csv"
    t3_path = out_dir / "tuning_sweep_diagnostics.csv"
    t4_path = out_dir / "granularity_concept_usage.csv"
    diag_path = out_dir / "tuning_sweep_full_diagnostics.json"

    df_t1.to_csv(t1_path, index=False)
    df_t2.to_csv(t2_path, index=False)
    df_t3.to_csv(t3_path, index=False)
    df_t4.to_csv(t4_path, index=False)
    with open(diag_path, "w") as f:
        json.dump(sweep_diag, f, indent=2)

    logger.info("saved -> %s", t1_path)
    logger.info("saved -> %s", t2_path)
    logger.info("saved -> %s", t3_path)
    logger.info("saved -> %s", t4_path)
    logger.info("saved -> %s", diag_path)

    # ── pretty print ─────────────────────────────────────────────
    _print_sweep(df_t1, df_t2, df_t3, df_t4, cfg_names)


def _print_sweep(
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame,
    df_t3: pd.DataFrame,
    df_t4: pd.DataFrame,
    cfg_names: list[str],
) -> None:
    exps = ["ablation_persona_only", "ablation_intent_only", "full_model"]
    exp_short = {
        "ablation_persona_only": "persona_only",
        "ablation_intent_only":  "intent_only",
        "full_model":            "full_model",
    }

    print("\n" + "=" * 90)
    print("TUNING SWEEP — TABLE 1: Overall Metrics")
    print("=" * 90)
    cols1 = ["tuning_tag", "experiment", "HR@10", "NDCG@10", "MRR"]
    cols1 = [c for c in cols1 if c in df_t1.columns]
    # show only key experiments
    df_t1_show = df_t1[df_t1["experiment"].isin(exps)].copy()
    df_t1_show["experiment"] = df_t1_show["experiment"].map(exp_short).fillna(df_t1_show["experiment"])
    print(df_t1_show[cols1].to_string(index=False))

    print("\n" + "=" * 90)
    print("TUNING SWEEP — TABLE 3: Delta Diagnostics")
    print("=" * 90)
    cols3 = ["tuning_tag", "experiment", "gt_delta_mean", "gt_delta_positive_frac",
             "all_nonzero_frac", "improved", "worsened", "gate_mean"]
    cols3 = [c for c in cols3 if c in df_t3.columns]
    df_t3_show = df_t3[df_t3["experiment"].isin(exps)].copy()
    df_t3_show["experiment"] = df_t3_show["experiment"].map(exp_short).fillna(df_t3_show["experiment"])
    print(df_t3_show[cols3].to_string(index=False))

    print("\n" + "=" * 90)
    print("TUNING SWEEP — TABLE 2: Per-Reason Metrics (full_model only)")
    print("=" * 90)
    cols2 = ["tuning_tag", "reason", "n_users", "HR@10", "NDCG@10", "MRR", "gt_delta_mean"]
    cols2 = [c for c in cols2 if c in df_t2.columns]
    df_t2_fm = df_t2[df_t2["experiment"] == "full_model"]
    for reason in _REASON_ORDER:
        sub = df_t2_fm[df_t2_fm["reason"] == reason]
        if sub.empty:
            continue
        n = int(sub["n_users"].iloc[0]) if "n_users" in sub.columns else "?"
        print(f"\n  reason: {reason}  (n={n})")
        print(sub[cols2].to_string(index=False))

    # ── key comparison: full_model vs intent_only per tuning ──────
    print("\n" + "=" * 90)
    print("KEY COMPARISON: full_model vs intent_only (HR@10 delta)")
    print("=" * 90)
    if not df_t1.empty and "HR@10" in df_t1.columns:
        for tag in cfg_names:
            sub = df_t1[df_t1["tuning_tag"] == tag]
            fm  = sub[sub["experiment"] == "full_model"]["HR@10"].values
            io  = sub[sub["experiment"] == "ablation_intent_only"]["HR@10"].values
            po  = sub[sub["experiment"] == "ablation_persona_only"]["HR@10"].values
            fm_v  = fm[0]  if len(fm)  else float("nan")
            io_v  = io[0]  if len(io)  else float("nan")
            po_v  = po[0]  if len(po)  else float("nan")
            delta = fm_v - io_v
            sign  = "+" if delta >= 0 else ""
            print(f"  {tag:12s}  full={fm_v:.4f}  intent={io_v:.4f}  persona={po_v:.4f}  "
                  f"full-intent={sign}{delta:.4f}  {'✓ full>intent' if delta > 0 else '✗ full≤intent'}")

    # ── Table 4: concept type usage ───────────────────────────────
    if not df_t4.empty:
        print("\n" + "=" * 90)
        print("TUNING SWEEP — TABLE 4: Concept Type Usage in boost_concepts (full_model)")
        print("=" * 90)
        df_t4_fm = df_t4[df_t4["experiment"] == "full_model"]
        for tag in cfg_names:
            sub = df_t4_fm[df_t4_fm["tuning_tag"] == tag].sort_values("fraction", ascending=False)
            if sub.empty:
                continue
            print(f"\n  [{tag}]")
            print(sub[["concept_type", "count", "fraction"]].to_string(index=False))
