"""
source_split_sampled_eval.py
-----------------------------
Source-split diagnostic for sampled evaluation results.

Reads existing sampled_reranked_*.parquet files (no re-scoring needed)
and slices them by source-aware subsets defined from intent records.

Subsets:
  all                   — all eval targets
  rec_current           — current_source == "rec"
  search_current        — current_source == "search"
  source_shift          — source_shift_flag == True
  recent_search_mixed   — recent_source_search_frac > 0 (any search in recent window)
  high_rec_persona      — users whose persona is dominated by rec (source_rec_frac > 0.7 avg)

For each subset × experiment computes:
  HR@K, NDCG@K, MRR, mean_gt_rank, mean_rank_delta, improved, worsened,
  gt_delta_mean, gt_delta_positive_frac

Output:
  data/artifacts/eval/<dataset>/source_distribution_summary.json
  data/artifacts/eval/<dataset>/source_split_metrics.csv
  data/artifacts/eval/<dataset>/source_split_diagnostics.json
  data/artifacts/eval/<dataset>/source_split_per_reason_metrics.csv  (optional)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import aggregate, compute_all

logger = logging.getLogger(__name__)

_REASON_ORDER = ["aligned", "exploration", "task_focus", "budget_shift", "unknown"]


# ─────────────────────────────────────────────────────────────────────
# Source distribution analysis
# ─────────────────────────────────────────────────────────────────────

def compute_source_distribution(
    df_intents: pd.DataFrame,
    eval_keys: set[tuple[str, int]],
    out_path: Path,
) -> dict:
    """
    Compute source distribution for eval targets.
    df_intents must have: user_id, target_index, current_source,
                          source_shift_flag, recent_source_rec_frac, recent_source_search_frac
    """
    # filter to eval keys only
    df = df_intents[
        df_intents.apply(
            lambda r: (str(r["user_id"]), int(r["target_index"])) in eval_keys, axis=1
        )
    ].copy()

    total = len(df)
    if total == 0:
        logger.warning("No intent records matched eval keys")
        return {}

    has_current_source  = "current_source" in df.columns
    has_shift_flag      = "source_shift_flag" in df.columns
    has_search_frac     = "recent_source_search_frac" in df.columns

    dist: dict = {"total_eval_targets": total}

    if has_current_source:
        src_counts = df["current_source"].value_counts().to_dict()
        dist["current_source"] = {
            src: {
                "count": int(cnt),
                "fraction": round(cnt / total, 4),
            }
            for src, cnt in src_counts.items()
        }

    if has_shift_flag:
        n_shift = int(df["source_shift_flag"].sum())
        dist["source_shift_flag"] = {
            "count": n_shift,
            "fraction": round(n_shift / total, 4),
        }

    if has_search_frac:
        n_any_search = int((df["recent_source_search_frac"] > 0).sum())
        dist["recent_search_mixed"] = {
            "count": n_any_search,
            "fraction": round(n_any_search / total, 4),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dist, f, indent=2)
    logger.info("source distribution saved -> %s", out_path)

    # pretty print
    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION (eval targets)")
    print("=" * 70)
    print(f"  total eval targets : {total}")
    if has_current_source:
        for src, d in dist["current_source"].items():
            print(f"  current_source={src:8s}: {d['count']:6d}  ({d['fraction']:.1%})")
    if has_shift_flag:
        d = dist["source_shift_flag"]
        print(f"  source_shift=True  : {d['count']:6d}  ({d['fraction']:.1%})")
    if has_search_frac:
        d = dist["recent_search_mixed"]
        print(f"  recent_search_mixed: {d['count']:6d}  ({d['fraction']:.1%})")

    return dist


# ─────────────────────────────────────────────────────────────────────
# Per-subset evaluation
# ─────────────────────────────────────────────────────────────────────

def _eval_subset_from_ranked(
    df_ranked: pd.DataFrame,
    subset_keys: set[tuple[str, int]],
    gt_by_key: dict[tuple[str, int], str],
    k_values: list[int],
    exp_name: str,
    subset_name: str,
) -> tuple[dict, dict]:
    """
    Evaluate one subset from a pre-ranked DataFrame.
    Returns (metrics_row, diag_row).
    """
    keys_in_data = subset_keys & set(gt_by_key.keys())
    if not keys_in_data:
        return {}, {}

    # filter ranked df to subset
    df_sub = df_ranked[
        df_ranked.apply(
            lambda r: (str(r["user_id"]), int(r["target_index"])) in keys_in_data, axis=1
        )
    ].copy()

    if df_sub.empty:
        return {}, {}

    df_sub = df_sub.sort_values(
        ["user_id", "target_index", "final_score"], ascending=[True, True, False]
    )

    per_user: list[dict] = []
    for (uid, tidx), grp in df_sub.groupby(["user_id", "target_index"], sort=False):
        gt_item = gt_by_key.get((str(uid), int(tidx)))
        if gt_item is None:
            continue
        ranked_items = grp["candidate_item_id"].tolist()
        m = compute_all(ranked_items, gt_item, k_values)
        m["gt_in_candidates"] = float(gt_item in ranked_items)

        # rank info
        gt_rows = grp[grp["candidate_item_id"] == gt_item]
        if not gt_rows.empty:
            m["rank_after"]  = int(gt_rows["rank_after"].iloc[0])  if "rank_after"  in gt_rows.columns else None
            m["rank_before"] = int(gt_rows["rank_before"].iloc[0]) if "rank_before" in gt_rows.columns else None
            m["modulation_delta"] = float(gt_rows["modulation_delta"].iloc[0]) if "modulation_delta" in gt_rows.columns else None
        else:
            m["rank_after"] = m["rank_before"] = m["modulation_delta"] = None

        m["deviation_reason"] = grp["deviation_reason"].iloc[0] if "deviation_reason" in grp.columns else "unknown"
        per_user.append(m)

    if not per_user:
        return {}, {}

    n = len(per_user)
    gt_cov = sum(r["gt_in_candidates"] for r in per_user) / n

    agg = aggregate([
        {k: v for k, v in r.items()
         if isinstance(v, float) and k not in ("gt_in_candidates", "modulation_delta")}
        for r in per_user
    ])

    # rank movement
    has_rank = [(r["rank_before"], r["rank_after"]) for r in per_user
                if r.get("rank_before") is not None and r.get("rank_after") is not None]
    improved  = sum(1 for rb, ra in has_rank if ra < rb)
    same      = sum(1 for rb, ra in has_rank if ra == rb)
    worsened  = sum(1 for rb, ra in has_rank if ra > rb)

    # mean GT rank
    valid_ranks = [r["rank_after"] for r in per_user if r.get("rank_after") is not None]
    mean_gt_rank = round(sum(valid_ranks) / len(valid_ranks), 2) if valid_ranks else None

    valid_deltas = [(r["rank_before"] - r["rank_after"])
                    for r in per_user
                    if r.get("rank_before") is not None and r.get("rank_after") is not None]
    mean_rank_delta = round(sum(valid_deltas) / len(valid_deltas), 4) if valid_deltas else None

    # GT modulation delta
    gt_deltas = [r["modulation_delta"] for r in per_user if r.get("modulation_delta") is not None]
    gt_delta_mean = round(sum(gt_deltas) / len(gt_deltas), 6) if gt_deltas else None
    gt_delta_pos_frac = round(sum(1 for d in gt_deltas if d > 0) / len(gt_deltas), 4) if gt_deltas else None

    metrics_row = {
        "subset":      subset_name,
        "experiment":  exp_name,
        "n_users":     n,
        "gt_coverage": round(gt_cov, 4),
    }
    metrics_row.update({k: round(v, 4) for k, v in agg.items()})
    metrics_row["mean_gt_rank"]   = mean_gt_rank
    metrics_row["mean_rank_delta"] = mean_rank_delta

    diag_row = {
        "subset":                  subset_name,
        "experiment":              exp_name,
        "n_users":                 n,
        "gt_delta_mean":           gt_delta_mean,
        "gt_delta_positive_frac":  gt_delta_pos_frac,
        "improved":                improved,
        "same":                    same,
        "worsened":                worsened,
        "net_improvement":         improved - worsened,
    }

    return metrics_row, diag_row


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def run_source_split_sampled_eval(
    experiment_names: list[str],
    ranked_dir: Path,           # directory containing sampled_reranked_*.parquet
    df_intents: pd.DataFrame,   # v2 intent records (with current_source etc.)
    df_cands: pd.DataFrame,     # sampled_candidates_k101.parquet
    df_persona: pd.DataFrame,   # persona_graphs_v2.parquet (for persona source profile)
    k_values: list[int],
    out_dir: Path,
) -> None:
    """
    Source-split evaluation of existing sampled_reranked_*.parquet files.

    ranked_dir: where sampled_reranked_{exp}.parquet files are
    out_dir:    where to write output CSVs/JSONs
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── GT map ──────────────────────────────────────────────────────
    gt_by_key: dict[tuple[str, int], str] = {
        (str(row.user_id), int(row.target_index)): str(row.candidate_item_id)
        for row in df_cands[df_cands["is_ground_truth"] == True].itertuples(index=False)
    }
    eval_keys = set(gt_by_key.keys())
    logger.info("eval_keys: %d", len(eval_keys))

    # ── intent index ─────────────────────────────────────────────────
    intent_by_key: dict[tuple[str, int], dict] = {
        (str(r["user_id"]), int(r["target_index"])): r
        for r in df_intents.to_dict("records")
    }

    # ── source distribution & subset building ───────────────────────
    dist_path = out_dir / "source_distribution_summary.json"
    dist = compute_source_distribution(df_intents, eval_keys, dist_path)

    has_v2_fields = "current_source" in df_intents.columns

    # Define subsets as sets of (user_id, target_index)
    # Always include "all"
    subset_defs: list[tuple[str, set[tuple[str, int]]]] = [
        ("all", set(eval_keys)),
    ]

    if has_v2_fields:
        # current_source == rec / search
        rec_keys = {
            (str(r["user_id"]), int(r["target_index"]))
            for r in df_intents.to_dict("records")
            if r.get("current_source") == "rec"
            and (str(r["user_id"]), int(r["target_index"])) in eval_keys
        }
        search_keys = {
            (str(r["user_id"]), int(r["target_index"]))
            for r in df_intents.to_dict("records")
            if r.get("current_source") == "search"
            and (str(r["user_id"]), int(r["target_index"])) in eval_keys
        }
        shift_keys = {
            (str(r["user_id"]), int(r["target_index"]))
            for r in df_intents.to_dict("records")
            if r.get("source_shift_flag") == True
            and (str(r["user_id"]), int(r["target_index"])) in eval_keys
        }
        recent_search_mixed_keys = {
            (str(r["user_id"]), int(r["target_index"]))
            for r in df_intents.to_dict("records")
            if float(r.get("recent_source_search_frac", 0)) > 0
            and (str(r["user_id"]), int(r["target_index"])) in eval_keys
        }

        if rec_keys:
            subset_defs.append(("rec_current", rec_keys))
        if search_keys:
            subset_defs.append(("search_current", search_keys))
        if shift_keys:
            subset_defs.append(("source_shift", shift_keys))
        if recent_search_mixed_keys:
            subset_defs.append(("recent_search_mixed", recent_search_mixed_keys))

    # ── persona source profile subsets ───────────────────────────────
    # high_rec_persona: users whose avg source_rec_frac > 0.7 across their persona nodes
    if "source_rec_frac" in df_persona.columns:
        persona_avg_rec = (
            df_persona.groupby("user_id")["source_rec_frac"].mean()
        )
        high_rec_users = set(persona_avg_rec[persona_avg_rec > 0.7].index.astype(str))
        high_rec_keys = {k for k in eval_keys if k[0] in high_rec_users}
        if high_rec_keys:
            subset_defs.append(("high_rec_persona", high_rec_keys))

        high_search_users = set(persona_avg_rec[persona_avg_rec < 0.3].index.astype(str))
        high_search_keys = {k for k in eval_keys if k[0] in high_search_users}
        if high_search_keys:
            subset_defs.append(("high_search_persona", high_search_keys))

    logger.info("Subsets defined:")
    for sname, skeys in subset_defs:
        logger.info("  %-25s  n=%d (%.1f%%)", sname, len(skeys), 100 * len(skeys) / len(eval_keys))

    # ── per-experiment evaluation ────────────────────────────────────
    metric_rows: list[dict] = []
    diag_rows:   list[dict] = []
    per_reason_rows: list[dict] = []

    for exp_name in experiment_names:
        ranked_path = ranked_dir / f"sampled_reranked_{exp_name}.parquet"
        if not ranked_path.exists():
            logger.warning("Missing: %s — skipping", ranked_path)
            continue

        logger.info("Loading ranked results: %s", ranked_path.name)
        df_ranked = pd.read_parquet(ranked_path)

        for subset_name, subset_keys in subset_defs:
            m_row, d_row = _eval_subset_from_ranked(
                df_ranked, subset_keys, gt_by_key, k_values, exp_name, subset_name
            )
            if m_row:
                metric_rows.append(m_row)
            if d_row:
                diag_rows.append(d_row)

        # ── per-reason within "all" subset ───────────────────────────
        if "deviation_reason" in df_ranked.columns:
            for reason in _REASON_ORDER:
                reason_keys = {
                    (str(r["user_id"]), int(r["target_index"]))
                    for r in df_intents.to_dict("records")
                    if r.get("deviation_reason") == reason
                    and (str(r["user_id"]), int(r["target_index"])) in eval_keys
                }
                if not reason_keys:
                    continue
                m_row, d_row = _eval_subset_from_ranked(
                    df_ranked, reason_keys, gt_by_key, k_values,
                    exp_name, f"reason_{reason}"
                )
                if m_row:
                    m_row["reason"] = reason
                    per_reason_rows.append(m_row)

    # ── save ─────────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(metric_rows)
    df_diag    = pd.DataFrame(diag_rows)
    df_pr      = pd.DataFrame(per_reason_rows)

    metrics_path = out_dir / "source_split_metrics.csv"
    diag_path    = out_dir / "source_split_diagnostics.json"
    pr_path      = out_dir / "source_split_per_reason_metrics.csv"

    df_metrics.to_csv(metrics_path, index=False)
    df_pr.to_csv(pr_path, index=False)

    # build JSON diagnostics
    diag_json: dict = {
        "source_distribution": dist,
        "subsets": {},
    }
    for _, row in df_diag.iterrows():
        sn = row["subset"]
        en = row["experiment"]
        if sn not in diag_json["subsets"]:
            diag_json["subsets"][sn] = {}
        diag_json["subsets"][sn][en] = {
            k: v for k, v in row.items() if k not in ("subset", "experiment")
        }

    with open(diag_path, "w") as f:
        json.dump(diag_json, f, indent=2, default=str)

    logger.info("saved -> %s", metrics_path)
    logger.info("saved -> %s", diag_path)
    logger.info("saved -> %s", pr_path)

    # ── pretty print ─────────────────────────────────────────────────
    _print_results(df_metrics, df_diag, df_pr, experiment_names)


def _print_results(
    df_metrics: pd.DataFrame,
    df_diag: pd.DataFrame,
    df_pr: pd.DataFrame,
    experiment_names: list[str],
) -> None:
    exp_short = {
        "ablation_persona_only": "persona_only",
        "ablation_intent_only":  "intent_only",
        "full_model":            "full_model",
        "ablation_backbone_only": "backbone_only",
    }

    print("\n" + "=" * 100)
    print("SOURCE-SPLIT SAMPLED EVALUATION")
    print("=" * 100)

    # Table 1: metrics
    print("\nTABLE 1: HR@10 / NDCG@10 / MRR / mean_gt_rank by subset × experiment")
    print("-" * 100)
    cols1 = ["subset", "experiment", "n_users", "HR@10", "NDCG@10", "MRR",
             "mean_gt_rank", "mean_rank_delta"]
    cols1 = [c for c in cols1 if c in df_metrics.columns]
    df_m = df_metrics.copy()
    df_m["experiment"] = df_m["experiment"].map(exp_short).fillna(df_m["experiment"])

    # print subset by subset, all experiments side by side
    subsets_ordered = df_m["subset"].unique().tolist()
    for sname in subsets_ordered:
        sub = df_m[df_m["subset"] == sname]
        n = sub["n_users"].iloc[0] if len(sub) > 0 else "?"
        print(f"\n  [{sname}]  n={n}")
        print(sub[cols1].to_string(index=False))

    # Table 2: delta diagnostics
    print("\n" + "=" * 100)
    print("TABLE 2: GT delta / rank movement by subset × experiment")
    print("-" * 100)
    cols2 = ["subset", "experiment", "n_users", "gt_delta_mean",
             "gt_delta_positive_frac", "improved", "worsened", "net_improvement"]
    cols2 = [c for c in cols2 if c in df_diag.columns]
    df_d = df_diag.copy()
    df_d["experiment"] = df_d["experiment"].map(exp_short).fillna(df_d["experiment"])

    for sname in subsets_ordered:
        sub = df_d[df_d["subset"] == sname]
        if sub.empty:
            continue
        print(f"\n  [{sname}]")
        print(sub[cols2].to_string(index=False))

    # Table 3: key comparison — full_model vs intent_only per subset
    print("\n" + "=" * 100)
    print("KEY COMPARISON: full_model vs intent_only (HR@10 delta, NDCG@10 delta)")
    print("-" * 100)
    if not df_metrics.empty and "HR@10" in df_metrics.columns:
        for sname in subsets_ordered:
            sub = df_metrics[df_metrics["subset"] == sname]
            fm = sub[sub["experiment"] == "full_model"]["HR@10"].values
            io = sub[sub["experiment"] == "ablation_intent_only"]["HR@10"].values
            fm_n = sub[sub["experiment"] == "full_model"]["NDCG@10"].values
            io_n = sub[sub["experiment"] == "ablation_intent_only"]["NDCG@10"].values
            fm_r = sub[sub["experiment"] == "full_model"]["mean_rank_delta"].values
            io_r = sub[sub["experiment"] == "ablation_intent_only"]["mean_rank_delta"].values
            n    = sub["n_users"].iloc[0] if len(sub) > 0 else "?"

            if len(fm) == 0 or len(io) == 0:
                continue

            dhr   = fm[0] - io[0]
            dndcg = fm_n[0] - io_n[0] if len(fm_n) and len(io_n) else float("nan")
            shr   = "+" if dhr  >= 0 else ""
            snd   = "+" if dndcg >= 0 else ""

            fm_rd = f"{fm_r[0]:+.3f}" if len(fm_r) and fm_r[0] is not None else "N/A"
            io_rd = f"{io_r[0]:+.3f}" if len(io_r) and io_r[0] is not None else "N/A"

            print(f"  {sname:25s}  n={n:6}  "
                  f"full-intent HR@10={shr}{dhr:.4f}  NDCG={snd}{dndcg:.4f}  "
                  f"mean_rank_delta: full={fm_rd} intent={io_rd}  "
                  f"{'✓' if dhr > 0 else ('~' if abs(dhr) < 0.001 else '✗')}")

    # Per-reason (full_model only)
    if not df_pr.empty:
        print("\n" + "=" * 100)
        print("TABLE 4: Per-reason metrics (full_model only, all targets)")
        print("-" * 100)
        df_pr_fm = df_pr[df_pr["experiment"] == "full_model"].copy()
        if not df_pr_fm.empty:
            pr_cols = ["reason", "n_users", "HR@10", "NDCG@10", "MRR", "mean_gt_rank", "mean_rank_delta"]
            pr_cols = [c for c in pr_cols if c in df_pr_fm.columns]
            print(df_pr_fm[pr_cols].to_string(index=False))
