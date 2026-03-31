"""
intent_debug.py
---------------
Diagnostic module for intent signal quality analysis.

Answers:
  1. goal_concepts vs GT item concepts overlap (per reason)
  2. suppress/filter penalty hitting GT items (per reason)
  3. reason-sliced HR@10 / NDCG@10 / MRR / delta stats across experiments
  4. concept type distribution of goal_concepts vs GT items
  5. case studies: what non-GT items get boosted over GT

Outputs:
  data/artifacts/eval/<dataset>/intent_debug_summary.json
  data/artifacts/eval/<dataset>/intent_goal_gt_overlap.csv
  data/artifacts/eval/<dataset>/intent_suppress_on_gt.csv
  data/artifacts/eval/<dataset>/intent_reason_metrics.csv
  data/artifacts/eval/<dataset>/intent_concept_type_analysis.csv
  data/artifacts/eval/<dataset>/intent_case_studies.csv
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPERIMENTS = [
    "ablation_backbone_only",
    "ablation_persona_only",
    "ablation_intent_only",
    "full_model",
]
REASONS = ["aligned", "task_focus", "exploration", "budget_shift", "unknown"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _concept_type(concept_id: str) -> str:
    return concept_id.split(":")[0]


def _parse_goal(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, np.ndarray)):
        return [str(c) for c in raw]
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def _parse_constraints(raw) -> dict:
    if not raw or raw == "{}":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _violates_constraints(item_concept_set: set, filter_constraints: dict) -> bool:
    """Same logic as CandidateReranker._violates_constraints."""
    for concept_type, allowed_list in filter_constraints.items():
        if not allowed_list:
            continue
        allowed_set = set(allowed_list)
        item_type_concepts = {c for c in item_concept_set if c.startswith(concept_type + ":")}
        if item_type_concepts and not item_type_concepts.intersection(allowed_set):
            return True
    return False


# ---------------------------------------------------------------------------
# Section 1: goal_concepts vs GT item concepts overlap
# ---------------------------------------------------------------------------

def _compute_goal_gt_overlap(
    df_eval_intent: pd.DataFrame,
    item_concepts: dict[str, list[str]],
    gt_by_key: dict[tuple[str, int], str],
) -> pd.DataFrame:
    """
    For each eval user, compute:
      - goal_concepts (from intent)
      - gt_item_concepts (from item_concepts map)
      - overlap_count, overlap_ratio, has_overlap
    Returns per-user DataFrame.
    """
    rows = []
    for _, rec in df_eval_intent.iterrows():
        uid = rec["user_id"]
        tidx = int(rec["target_index"])
        key = (uid, tidx)
        gt_item = gt_by_key.get(key)
        if gt_item is None:
            continue

        goal = set(_parse_goal(rec["goal_concepts"]))
        gt_concepts = set(item_concepts.get(gt_item, []))
        overlap = goal & gt_concepts
        overlap_count = len(overlap)
        goal_size = len(goal)
        gt_size = len(gt_concepts)

        rows.append({
            "user_id": uid,
            "target_index": tidx,
            "deviation_reason": rec["deviation_reason"],
            "goal_size": goal_size,
            "gt_concept_size": gt_size,
            "overlap_count": overlap_count,
            "has_overlap": int(overlap_count > 0),
            "overlap_ratio_goal": overlap_count / goal_size if goal_size > 0 else 0.0,
            "overlap_ratio_gt": overlap_count / gt_size if gt_size > 0 else 0.0,
            "goal_concepts_list": list(goal),
            "gt_concepts_list": list(gt_concepts),
            "overlap_concepts": list(overlap),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 2: suppress / filter penalty on GT
# ---------------------------------------------------------------------------

def _compute_suppress_on_gt(
    df_eval_intent: pd.DataFrame,
    item_concepts: dict[str, list[str]],
    gt_by_key: dict[tuple[str, int], str],
    modulation_cfg: dict,
) -> pd.DataFrame:
    """
    For each eval user, check whether the GT item:
      - has any concepts in suppress_concepts (task_focus / budget_shift suppress logic)
      - violates filter_constraints (filter_active reasons)
    """
    reason_policy = modulation_cfg.get("reason_policy", {})

    # Reproduce suppress_concepts logic from signal_builder for each reason
    # suppress_concepts is built from persona top, but we can approximate using
    # the intent record itself — for task_focus: suppress non-goal concepts in persona
    # For a clean diagnostic, we check what signal_builder would produce.
    # We need persona_nodes; but we can do a partial check using constraints only.

    rows = []
    for _, rec in df_eval_intent.iterrows():
        uid = rec["user_id"]
        tidx = int(rec["target_index"])
        key = (uid, tidx)
        gt_item = gt_by_key.get(key)
        if gt_item is None:
            continue

        reason = rec["deviation_reason"]
        policy = reason_policy.get(reason, reason_policy.get("unknown", {}))
        filter_active = policy.get("filter_active", False)

        goal = set(_parse_goal(rec["goal_concepts"]))
        constraints = _parse_constraints(rec.get("constraints_json", "{}"))
        gt_concepts = set(item_concepts.get(gt_item, []))

        # filter constraint violation
        filter_violation = False
        if filter_active and constraints:
            filter_violation = _violates_constraints(gt_concepts, constraints)

        # goal mismatch: GT has NO overlap with goal_concepts at all
        goal_mismatch = int(len(goal & gt_concepts) == 0) if goal else 0

        # which constraint types the GT violates
        violated_types = []
        if filter_active and constraints:
            for ctype, allowed_list in constraints.items():
                if not allowed_list:
                    continue
                item_type_concepts = {c for c in gt_concepts if c.startswith(ctype + ":")}
                if item_type_concepts and not item_type_concepts.intersection(set(allowed_list)):
                    violated_types.append(ctype)

        rows.append({
            "user_id": uid,
            "target_index": tidx,
            "deviation_reason": reason,
            "filter_active": int(filter_active),
            "gt_violates_filter": int(filter_violation),
            "gt_goal_mismatch": goal_mismatch,
            "violated_constraint_types": ",".join(violated_types),
            "n_constraints": len(constraints),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 3: reason-sliced metrics
# ---------------------------------------------------------------------------

def _compute_reason_metrics(
    ranked_path: Path,
    gt_by_key: dict[tuple[str, int], str],
    k_values: list[int],
    exp_name: str,
) -> pd.DataFrame:
    """
    Load sampled_reranked_{exp}.parquet and compute per-reason HR@k, NDCG@k, MRR, delta stats.
    """
    df = pd.read_parquet(ranked_path)

    # Build gt_map
    gt_rows = [
        {"user_id": uid, "target_index": tidx, "gt_item": gt_item}
        for (uid, tidx), gt_item in gt_by_key.items()
    ]
    df_gt = pd.DataFrame(gt_rows)

    # Assign rank by final_score
    df = df.sort_values(["user_id", "target_index", "final_score"], ascending=[True, True, False])
    df["_rank"] = df.groupby(["user_id", "target_index"]).cumcount() + 1

    # Merge GT
    df = df.merge(df_gt, on=["user_id", "target_index"], how="left")
    gt_mask = df["candidate_item_id"] == df["gt_item"]
    df_gt_rows = df[gt_mask].copy()

    results = []
    for reason in REASONS:
        sub = df_gt_rows[df_gt_rows["deviation_reason"] == reason]
        if len(sub) == 0:
            continue

        ranks = sub["_rank"].values
        n = len(ranks)

        row: dict = {
            "experiment": exp_name,
            "deviation_reason": reason,
            "n_users": n,
        }
        for k in k_values:
            row[f"HR@{k}"]   = float((ranks <= k).mean())
            row[f"NDCG@{k}"] = float(np.mean([
                1.0 / math.log2(r + 1) if r <= k else 0.0 for r in ranks
            ]))
        row["MRR"] = float(np.mean(1.0 / ranks))

        # delta stats (if available)
        if "modulation_delta" in sub.columns:
            deltas = sub["modulation_delta"].values
            row["gt_delta_mean"]          = round(float(deltas.mean()), 6)
            row["gt_delta_positive_frac"] = round(float((deltas > 0).mean()), 4)
        else:
            row["gt_delta_mean"]          = 0.0
            row["gt_delta_positive_frac"] = 0.0

        # rank movement (need rank_before / rank_after)
        if "rank_before" in sub.columns and "rank_after" in sub.columns:
            has_both = sub["rank_before"].notna() & sub["rank_after"].notna()
            sub_mv = sub[has_both]
            row["improved"] = int((sub_mv["rank_after"] < sub_mv["rank_before"]).sum())
            row["worsened"] = int((sub_mv["rank_after"] > sub_mv["rank_before"]).sum())
            row["same"]     = int((sub_mv["rank_after"] == sub_mv["rank_before"]).sum())
        else:
            row["improved"] = row["worsened"] = row["same"] = 0

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Section 4: concept type analysis
# ---------------------------------------------------------------------------

def _compute_concept_type_analysis(
    df_eval_intent: pd.DataFrame,
    item_concepts: dict[str, list[str]],
    gt_by_key: dict[tuple[str, int], str],
) -> pd.DataFrame:
    """
    For each concept type appearing in goal_concepts:
      - how often it appears
      - what fraction of the time the GT item has a concept of that type that matches
    """
    type_goal_count: dict[str, int] = defaultdict(int)
    type_gt_match: dict[str, int] = defaultdict(int)

    for _, rec in df_eval_intent.iterrows():
        uid = rec["user_id"]
        tidx = int(rec["target_index"])
        key = (uid, tidx)
        gt_item = gt_by_key.get(key)
        if gt_item is None:
            continue

        goal = _parse_goal(rec["goal_concepts"])
        gt_concepts = set(item_concepts.get(gt_item, []))

        for cid in goal:
            ctype = _concept_type(cid)
            type_goal_count[ctype] += 1
            if cid in gt_concepts:
                type_gt_match[ctype] += 1

    rows = []
    for ctype in sorted(type_goal_count, key=lambda x: -type_goal_count[x]):
        cnt = type_goal_count[ctype]
        match = type_gt_match[ctype]
        rows.append({
            "concept_type": ctype,
            "goal_count": cnt,
            "gt_match_count": match,
            "gt_match_rate": round(match / cnt, 4) if cnt > 0 else 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 5: case studies
# ---------------------------------------------------------------------------

def _compute_case_studies(
    df_intent_only: pd.DataFrame,
    df_eval_intent: pd.DataFrame,
    item_concepts: dict[str, list[str]],
    gt_by_key: dict[tuple[str, int], str],
    n_cases: int = 30,
    min_rank_drop: int = 5,
) -> pd.DataFrame:
    """
    Find cases where intent_only caused GT rank to drop significantly.
    For each such case, show:
      - GT item + concepts + rank_before/after + delta
      - top-3 non-GT items that jumped over GT + their concepts
      - goal_concepts vs GT overlap
    """
    # GT rows where rank worsened a lot
    df_gt_map = pd.DataFrame(
        [{"user_id": uid, "target_index": tidx, "gt_item": gt_item}
         for (uid, tidx), gt_item in gt_by_key.items()]
    )
    df_gt_rows = df_intent_only.merge(df_gt_map, on=["user_id", "target_index"], how="inner")
    df_gt_rows = df_gt_rows[df_gt_rows["candidate_item_id"] == df_gt_rows["gt_item"]].copy()

    # Cases where rank dropped
    df_gt_rows["rank_drop"] = df_gt_rows["rank_after"] - df_gt_rows["rank_before"]
    worst = df_gt_rows[df_gt_rows["rank_drop"] >= min_rank_drop].nlargest(n_cases, "rank_drop")

    if len(worst) == 0:
        logger.warning("No cases with rank_drop >= %d found", min_rank_drop)
        return pd.DataFrame()

    # Build intent lookup
    intent_lookup = {
        (r["user_id"], int(r["target_index"])): r
        for _, r in df_eval_intent.iterrows()
    }

    case_rows = []
    for _, gt_row in worst.iterrows():
        uid = gt_row["user_id"]
        tidx = int(gt_row["target_index"])
        key = (uid, tidx)
        gt_item = gt_row["gt_item"]
        intent_rec = intent_lookup.get(key, {})

        goal = _parse_goal(intent_rec.get("goal_concepts", []))
        gt_concepts = item_concepts.get(gt_item, [])
        overlap = set(goal) & set(gt_concepts)

        # top items that beat GT (rank_after < gt_row.rank_after, rank_before >= gt_row.rank_before)
        sub = df_intent_only[
            (df_intent_only["user_id"] == uid) &
            (df_intent_only["target_index"] == tidx) &
            (df_intent_only["candidate_item_id"] != gt_item) &
            (df_intent_only["rank_after"] < gt_row["rank_after"])
        ].sort_values("rank_after").head(3)

        boosted_items = []
        for _, br in sub.iterrows():
            bid = br["candidate_item_id"]
            b_concepts = item_concepts.get(bid, [])
            b_overlap = set(goal) & set(b_concepts)
            boosted_items.append(f"{bid}[rank:{int(br['rank_after'])},goal_overlap:{list(b_overlap)}]")

        case_rows.append({
            "user_id": uid,
            "target_index": tidx,
            "deviation_reason": gt_row["deviation_reason"],
            "gt_item": gt_item,
            "rank_before": int(gt_row["rank_before"]),
            "rank_after": int(gt_row["rank_after"]),
            "rank_drop": int(gt_row["rank_drop"]),
            "gt_delta": round(float(gt_row["modulation_delta"]), 6),
            "goal_concepts": goal,
            "gt_concepts": gt_concepts[:10],
            "goal_gt_overlap": list(overlap),
            "boosted_non_gt": boosted_items,
        })

    return pd.DataFrame(case_rows)


# ---------------------------------------------------------------------------
# Main entry: run_intent_debug
# ---------------------------------------------------------------------------

def run_intent_debug(
    eval_dir: Path,
    dataset: str,
    df_snaps: pd.DataFrame,
    item_concepts: dict[str, list[str]],
    intent_by_key: dict[tuple[str, int], dict],
    modulation_cfg: dict,
    k_values: list[int],
    experiment_names: list[str] | None = None,
) -> None:
    """
    Run all intent diagnostic analyses and write output files.
    """
    if experiment_names is None:
        experiment_names = EXPERIMENTS

    out_dir = eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── build eval intent DataFrame ───────────────────────────────────
    eval_snaps = df_snaps.loc[
        df_snaps.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)

    # Build gt_by_key from candidate parquet (single GT per eval user)
    cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k101.parquet")
    df_cands = pd.read_parquet(cand_path)
    gt_by_key: dict[tuple[str, int], str] = {
        (row.user_id, int(row.target_index)): row.candidate_item_id
        for row in df_cands[df_cands["is_ground_truth"]].itertuples(index=False)
    }
    logger.info("GT map built: %d users", len(gt_by_key))

    # Build eval intent DataFrame from intent_by_key
    intent_rows = [
        {**v, "user_id": uid, "target_index": tidx}
        for (uid, tidx), v in intent_by_key.items()
        if (uid, tidx) in gt_by_key
    ]
    df_eval_intent = pd.DataFrame(intent_rows)
    logger.info("Eval intent rows: %d", len(df_eval_intent))

    # ── Section 1: goal-GT overlap ────────────────────────────────────
    logger.info("Computing goal-GT overlap...")
    df_overlap = _compute_goal_gt_overlap(df_eval_intent, item_concepts, gt_by_key)

    overlap_path = out_dir / "intent_goal_gt_overlap.csv"
    df_overlap.drop(columns=["goal_concepts_list", "gt_concepts_list", "overlap_concepts"]).to_csv(
        overlap_path, index=False
    )
    logger.info("saved -> %s", overlap_path)

    # Table 1: per-reason overlap summary
    overlap_by_reason = (
        df_overlap.groupby("deviation_reason")
        .agg(
            n_users=("user_id", "count"),
            overlap_rate=("has_overlap", "mean"),
            mean_overlap_count=("overlap_count", "mean"),
            mean_overlap_ratio_goal=("overlap_ratio_goal", "mean"),
            mean_overlap_ratio_gt=("overlap_ratio_gt", "mean"),
        )
        .reset_index()
        .round(4)
    )
    logger.info("\n=== TABLE 1: goal-GT overlap by reason ===\n%s", overlap_by_reason.to_string(index=False))

    # ── Section 2: suppress on GT ─────────────────────────────────────
    logger.info("Computing suppress/filter on GT...")
    df_suppress = _compute_suppress_on_gt(df_eval_intent, item_concepts, gt_by_key, modulation_cfg)

    suppress_path = out_dir / "intent_suppress_on_gt.csv"
    df_suppress.to_csv(suppress_path, index=False)
    logger.info("saved -> %s", suppress_path)

    # Table 2: per-reason suppress summary
    suppress_by_reason = (
        df_suppress.groupby("deviation_reason")
        .agg(
            n_users=("user_id", "count"),
            gt_filter_rate=("gt_violates_filter", "mean"),
            gt_goal_mismatch_rate=("gt_goal_mismatch", "mean"),
            filter_active_frac=("filter_active", "mean"),
        )
        .reset_index()
        .round(4)
    )
    logger.info("\n=== TABLE 2: suppress/filter on GT by reason ===\n%s", suppress_by_reason.to_string(index=False))

    # ── Section 3: reason-sliced metrics ─────────────────────────────
    logger.info("Computing reason-sliced metrics...")
    all_reason_metrics = []
    for exp_name in experiment_names:
        ranked_path = out_dir / f"sampled_reranked_{exp_name}.parquet"
        if not ranked_path.exists():
            logger.warning("Missing: %s — skipping", ranked_path)
            continue
        df_rm = _compute_reason_metrics(ranked_path, gt_by_key, k_values, exp_name)
        all_reason_metrics.append(df_rm)

    df_reason_metrics = pd.concat(all_reason_metrics, ignore_index=True) if all_reason_metrics else pd.DataFrame()
    reason_metrics_path = out_dir / "intent_reason_metrics.csv"
    df_reason_metrics.to_csv(reason_metrics_path, index=False)
    logger.info("saved -> %s", reason_metrics_path)

    # Table 3: pretty print
    if not df_reason_metrics.empty:
        display_cols = ["deviation_reason", "experiment", "n_users", "HR@10", "NDCG@10", "MRR",
                        "gt_delta_mean", "gt_delta_positive_frac", "improved", "worsened"]
        display_cols = [c for c in display_cols if c in df_reason_metrics.columns]
        logger.info("\n=== TABLE 3: reason × experiment metrics ===\n%s",
                    df_reason_metrics[display_cols].sort_values(["deviation_reason", "experiment"]).to_string(index=False))

    # ── Section 4: concept type analysis ─────────────────────────────
    logger.info("Computing concept type analysis...")
    df_ctype = _compute_concept_type_analysis(df_eval_intent, item_concepts, gt_by_key)
    ctype_path = out_dir / "intent_concept_type_analysis.csv"
    df_ctype.to_csv(ctype_path, index=False)
    logger.info("saved -> %s", ctype_path)
    logger.info("\n=== TABLE 4: concept type analysis ===\n%s", df_ctype.to_string(index=False))

    # ── Section 5: case studies ───────────────────────────────────────
    intent_only_path = out_dir / "sampled_reranked_ablation_intent_only.parquet"
    if intent_only_path.exists():
        logger.info("Computing case studies (intent_only worsened cases)...")
        df_intent_only_ranked = pd.read_parquet(intent_only_path)
        df_cases = _compute_case_studies(
            df_intent_only_ranked, df_eval_intent, item_concepts, gt_by_key,
            n_cases=50, min_rank_drop=10,
        )
        if not df_cases.empty:
            # Stringify list columns for CSV
            for col in ["goal_concepts", "gt_concepts", "goal_gt_overlap", "boosted_non_gt"]:
                if col in df_cases.columns:
                    df_cases[col] = df_cases[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            cases_path = out_dir / "intent_case_studies.csv"
            df_cases.to_csv(cases_path, index=False)
            logger.info("saved -> %s  (%d cases)", cases_path, len(df_cases))
    else:
        logger.warning("intent_only reranked parquet not found — skipping case studies")

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "dataset": dataset,
        "n_eval_users": len(gt_by_key),
        "table1_goal_gt_overlap": overlap_by_reason.to_dict(orient="records"),
        "table2_suppress_on_gt": suppress_by_reason.to_dict(orient="records"),
        "table3_reason_metrics": df_reason_metrics.round(4).to_dict(orient="records") if not df_reason_metrics.empty else [],
        "table4_concept_type_analysis": df_ctype.to_dict(orient="records"),
    }
    summary_path = out_dir / "intent_debug_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved -> %s", summary_path)

    # ── Final print ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTENT DEBUG SUMMARY")
    print("=" * 70)

    print("\n[TABLE 1] goal-GT overlap by reason")
    print(overlap_by_reason.to_string(index=False))

    print("\n[TABLE 2] suppress / filter hitting GT by reason")
    print(suppress_by_reason.to_string(index=False))

    print("\n[TABLE 4] concept type analysis (goal_concepts)")
    print(df_ctype.to_string(index=False))

    if not df_reason_metrics.empty:
        print("\n[TABLE 3] reason × experiment metrics")
        t3_pivot = df_reason_metrics[display_cols].sort_values(["deviation_reason", "experiment"])
        print(t3_pivot.to_string(index=False))
