"""
run_selector_scoring_eval.py
----------------------------
Compare baseline_count vs count_x_idf selector scoring modes.

Runs full ablation:
  A. LLM + baseline_count selector  (current default)
  B. LLM + count_x_idf selector     (specificity-weighted)
  C. heuristic control baseline

For each: intent_only_rerank + graph_conditioned_full.

Diagnostics per branch:
  - selected concept avg doc_freq / avg specificity
  - GT match before/after
  - candidate match
  - generic concept fraction (doc_freq > threshold)
  - reason breakdown (especially exploration)

Outputs (to --out-dir):
  selector_scoring_comparison.csv
  selector_scoring_exploration.csv
  selector_scoring_report.md
"""
from __future__ import annotations

import copy
import json
import logging
import math
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _to_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    if hasattr(x, "__iter__"):
        return list(x)
    return []


def _load_intent(path: Path, shared_keys: set) -> dict[tuple, dict]:
    df = pd.read_parquet(path)
    result = {}
    for r in df.to_dict("records"):
        k = (str(r["user_id"]), int(r["target_index"]))
        if k in shared_keys:
            result[k] = r
    return result


# ── concept diagnostics ───────────────────────────────────────────────

def _concept_diag(
    concepts: list[str],
    concept_doc_freq: dict[str, int],
    total_items: int,
    persona_top_set: set[str],
    generic_df_threshold: int = 5000,
) -> dict:
    if not concepts:
        return {
            "avg_doc_freq": 0.0, "avg_specificity": 0.0,
            "generic_frac": 0.0, "persona_overlap_frac": 0.0,
        }
    specs = []
    dfs   = []
    generic_n = 0
    persona_n = 0
    for c in concepts:
        df_val = concept_doc_freq.get(c, 1)
        dfs.append(df_val)
        specs.append(math.log(total_items / max(df_val, 1)))
        if df_val >= generic_df_threshold:
            generic_n += 1
        if c in persona_top_set:
            persona_n += 1
    return {
        "avg_doc_freq":          round(sum(dfs) / len(dfs), 2),
        "avg_specificity":       round(sum(specs) / len(specs), 4),
        "generic_frac":          round(generic_n / len(concepts), 4),
        "persona_overlap_frac":  round(persona_n / len(concepts), 4),
    }


# ── eval loop ─────────────────────────────────────────────────────────

def _run_eval(
    branch_name: str,
    intent_by_key: dict[tuple, dict],
    cand_by_key: dict[tuple, list],
    is_gt_by_key: dict[tuple, set],
    backbone_scores: dict[tuple, dict],
    persona_nodes_by_user: dict[str, list],
    item_concepts: dict[str, list],
    modulation_cfg: dict,
    k_values: list[int],
    concept_doc_freq: dict[str, int],
    total_items: int,
    persona_top_by_user: dict[str, set],
) -> dict[str, dict]:
    from src.modulation.gate import compute_gate_strength
    from src.modulation.reranker import CandidateReranker
    from src.modulation.signal_builder import build_signal

    reranker = CandidateReranker(modulation_cfg, item_concepts)
    experiment_modes = {
        "intent_only":  "intent_only_rerank",
        "full_model":   "graph_conditioned_full",
    }
    results: dict[str, dict] = {}

    for exp_name, mode in experiment_modes.items():
        all_ranked: list[dict] = []

        for (uid, tidx), candidate_ids in cand_by_key.items():
            scores = backbone_scores[(uid, tidx)]
            candidate_tuples = sorted(
                [(iid, scores[iid]) for iid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )
            intent_record = dict(intent_by_key[(uid, tidx)])
            reason     = intent_record.get("deviation_reason", "unknown")
            confidence = float(intent_record.get("confidence", 0.5))
            persona_nodes = persona_nodes_by_user.get(uid, [])

            gate_strength = compute_gate_strength(
                deviation_reason=reason,
                confidence=confidence,
                persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_record, persona_nodes, gate_strength, modulation_cfg, mode=mode)
            ranked = reranker.rerank(candidate_tuples, signal, mode=mode)

            gt_items = is_gt_by_key.get((uid, tidx), set())
            for r in ranked:
                rec = r.to_record()
                rec["is_ground_truth"] = int(rec["candidate_item_id"] in gt_items)
                rec["deviation_reason"] = reason
                all_ranked.append(rec)

        df_r = pd.DataFrame(all_ranked)
        df_r = df_r.sort_values(
            ["user_id", "target_index", "final_score"], ascending=[True, True, False]
        )
        df_r["_rank"] = df_r.groupby(["user_id", "target_index"]).cumcount() + 1
        gt_df = df_r[df_r["is_ground_truth"] == 1].copy()

        if gt_df.empty:
            results[exp_name] = {}
            continue

        ranks = gt_df["_rank"].values.astype(int)
        n = len(ranks)
        metrics: dict = {"branch": branch_name, "experiment": exp_name, "n_users": n}
        for k in k_values:
            metrics[f"HR@{k}"]   = round(float((ranks <= k).mean()), 4)
            metrics[f"NDCG@{k}"] = round(
                float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n), 4
            )
        metrics["MRR"] = round(float((1.0 / ranks).mean()), 4)
        if "modulation_delta" in gt_df.columns:
            gt_deltas = gt_df["modulation_delta"]
            metrics["gt_delta_pos_frac"]  = round(float((gt_deltas > 0).mean()), 4)
            metrics["gt_delta_zero_frac"] = round(float((gt_deltas == 0).mean()), 4)

        # per-reason HR@10
        if "deviation_reason" in gt_df.columns:
            for rsn, grp in gt_df.groupby("deviation_reason"):
                r_arr = grp["_rank"].values.astype(int)
                metrics[f"HR@10_{rsn}"] = round(float((r_arr <= 10).mean()), 4)

        results[exp_name] = metrics
        logger.info(
            "[%s / %s]  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  gt_delta+frac=%s  gt_zero=%s",
            branch_name, exp_name,
            metrics.get("HR@10", 0), metrics.get("NDCG@10", 0), metrics.get("MRR", 0),
            metrics.get("gt_delta_pos_frac", "n/a"), metrics.get("gt_delta_zero_frac", "n/a"),
        )

    return results


# ── concept diagnostics aggregation ──────────────────────────────────

def _aggregate_concept_diag(
    intent_by_key: dict,
    goal_field: str,
    concept_doc_freq: dict,
    total_items: int,
    persona_top_by_user: dict,
    gt_concepts_by_key: dict,
    reason_filter: str | None = None,
) -> dict:
    rows = []
    for (uid, tidx), rec in intent_by_key.items():
        reason = rec.get("deviation_reason", "unknown")
        if reason_filter and reason != reason_filter:
            continue
        goals = _to_list(rec.get(goal_field))
        pt_set = persona_top_by_user.get(uid, set())
        d = _concept_diag(goals, concept_doc_freq, total_items, pt_set)
        gt_c = gt_concepts_by_key.get((uid, tidx), set())
        gt_match = len(set(goals) & gt_c) / max(len(goals), 1) if goals else 0.0
        d["gt_match"] = round(gt_match, 4)
        d["n_goals"] = len(goals)
        d["reason"] = reason
        rows.append(d)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        "avg_doc_freq":         round(df["avg_doc_freq"].mean(), 2),
        "avg_specificity":      round(df["avg_specificity"].mean(), 4),
        "generic_frac":         round(df["generic_frac"].mean(), 4),
        "persona_overlap_frac": round(df["persona_overlap_frac"].mean(), 4),
        "gt_match":             round(df["gt_match"].mean(), 4),
        "avg_n_goals":          round(df["n_goals"].mean(), 3),
        "n_records":            len(df),
    }


# ── selector rescore helper ───────────────────────────────────────────

def _apply_idf_rescore(
    intent_by_key: dict[tuple, dict],
    cand_by_key: dict[tuple, list],
    item_concepts: dict[str, list],
    idf: dict[str, float],
    idf_floor: float = 0.1,
    max_goals_per_reason: dict | None = None,
) -> dict[tuple, dict]:
    """
    Re-apply Stage 2 with count_x_idf scoring in-memory (no parquet I/O).
    Reads raw_llm_goals from each record and re-selects using IDF-weighted score.
    This lets us compare baseline_count vs count_x_idf on the same user set
    without re-running run_apply_grounded_selector.
    """
    from src.intent.grounded_selector import (
        build_candidate_concept_bank,
        validate_and_select_goals,
    )

    if max_goals_per_reason is None:
        max_goals_per_reason = {
            "aligned": 3, "task_focus": 2,
            "exploration": 3, "budget_shift": 2, "unknown": 1,
        }

    result: dict[tuple, dict] = {}
    for (uid, tidx), rec in intent_by_key.items():
        _rg = rec.get("raw_llm_goals")
        raw_goals = _to_list(_rg if _rg is not None else rec.get("goal_concepts"))
        reason    = str(rec.get("deviation_reason", "unknown"))
        conf      = float(rec.get("confidence", 0.5))

        cands = cand_by_key.get((uid, tidx), [])
        bank  = build_candidate_concept_bank(cands, item_concepts)

        # Build a temporary IDF parquet path is not needed —
        # pass idf dict directly via a closure in selector_cfg scoring.
        # We use a trick: write a temp parquet for the idf dict.
        # Actually cleaner: call with scoring dict that has in-memory idf.
        # Since _load_concept_idf caches by path, we write once globally.
        validated, _ = validate_and_select_goals(
            raw_goal_concepts=raw_goals,
            deviation_reason=reason,
            confidence=conf,
            candidate_concept_bank=bank,
            persona_top_concepts=[],
            selector_cfg={
                "scoring": {
                    "mode": "count_x_idf",
                    "idf_path": _IDF_PATH,
                    "idf_floor": idf_floor,
                },
                "max_goals_per_reason": max_goals_per_reason,
            },
        )
        new_rec = dict(rec)
        new_rec["validated_goal_concepts"] = validated
        result[(uid, tidx)] = new_rec
    return result


_IDF_PATH = ""   # set in main()


# ── main ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",       default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--eval-config",       default="config/evaluation/default.yaml")
    parser.add_argument("--mod-config",        default="config/modulation/amazon_movies_tv.yaml")
    parser.add_argument("--llm-intent-path",   default=None)
    parser.add_argument("--heur-intent-path",  default=None)
    parser.add_argument("--idf-path",          default=None,
                        help="Concept IDF parquet for count_x_idf scoring. "
                             "Default: data/cache/concept_idf/<dataset>/concept_idf.parquet")
    parser.add_argument("--max-users",  type=int, default=2000)
    parser.add_argument("--out-dir",    default="results/selector_scoring_eval")
    args = parser.parse_args()

    data_cfg = _load_yaml(args.data_config)
    eval_cfg = _load_yaml(args.eval_config)
    mod_cfg  = _load_yaml(args.mod_config)

    dataset       = data_cfg.get("dataset", "amazon_movies_tv")
    k_values      = eval_cfg.get("k_values", [5, 10, 20])
    processed_dir = Path(data_cfg["paths"]["processed_dir"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── IDF ───────────────────────────────────────────────────────────
    global _IDF_PATH
    _IDF_PATH = args.idf_path or f"data/cache/concept_idf/{dataset}/concept_idf.parquet"
    if not Path(_IDF_PATH).exists():
        logger.error("IDF path not found: %s — run build_concept_idf first.", _IDF_PATH)
        return
    logger.info("IDF path: %s", _IDF_PATH)

    # ── load item_concepts ────────────────────────────────────────────
    logger.info("Loading item_concepts...")
    df_ic = pd.read_parquet(processed_dir / "item_concepts.parquet")
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    concept_doc_freq: dict[str, int] = (
        df_ic.groupby("concept_id")["item_id"].nunique().to_dict()
    )
    total_items = df_ic["item_id"].nunique()
    logger.info("item_concepts: %d  concepts: %d  items: %d",
                len(item_concepts), len(concept_doc_freq), total_items)

    # ── load candidates ───────────────────────────────────────────────
    cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k101.parquet")
    logger.info("Loading candidates...")
    df_cands = pd.read_parquet(cand_path)
    cand_keys = set(zip(df_cands["user_id"].astype(str),
                        df_cands["target_index"].astype(int)))

    # ── load intent paths ─────────────────────────────────────────────
    llm_path  = Path(
        args.llm_intent_path
        or f"data/cache/intent/{dataset}/short_term_intents_llm_subset_2000_validated.parquet"
    )
    heur_path = Path(
        args.heur_intent_path
        or f"data/cache/intent/{dataset}/short_term_intents_heuristic.parquet"
    )
    if not llm_path.exists():
        logger.error("LLM intent not found: %s", llm_path); return

    # ── shared keys ───────────────────────────────────────────────────
    df_llm_keys = pd.read_parquet(llm_path, columns=["user_id", "target_index"])
    llm_keys    = set(zip(df_llm_keys["user_id"].astype(str),
                          df_llm_keys["target_index"].astype(int)))
    llm_users   = sorted({k[0] for k in llm_keys})[:args.max_users]
    llm_users_s = set(llm_users)
    shared_keys = {k for k in (cand_keys & llm_keys) if k[0] in llm_users_s}
    logger.info("shared_keys: %d", len(shared_keys))

    # ── build candidate index + GT ────────────────────────────────────
    df_cands = df_cands[df_cands["user_id"].astype(str).isin(llm_users_s)]
    cand_by_key:   dict[tuple, list[str]] = {}
    is_gt_by_key:  dict[tuple, set[str]]  = {}
    gt_concepts_by_key: dict[tuple, set[str]] = {}
    for row in df_cands.itertuples(index=False):
        key = (str(row.user_id), int(row.target_index))
        if key not in shared_keys:
            continue
        if key not in cand_by_key:
            cand_by_key[key]  = []
            is_gt_by_key[key] = set()
        cand_by_key[key].append(row.candidate_item_id)
        if row.is_ground_truth:
            is_gt_by_key[key].add(row.candidate_item_id)
            gt_concepts_by_key[key] = set(item_concepts.get(row.candidate_item_id, []))
    logger.info("Candidate index: %d keys", len(cand_by_key))

    # ── backbone scores ───────────────────────────────────────────────
    bs_path = Path(f"data/cache/backbone/{dataset}/backbone_scores.parquet")
    logger.info("Loading backbone scores...")
    df_bs = pd.read_parquet(bs_path)
    backbone_scores: dict[tuple, dict[str, float]] = {}
    for row in df_bs.itertuples(index=False):
        key = (str(row.user_id), int(row.target_index))
        if key not in shared_keys:
            continue
        if key not in backbone_scores:
            backbone_scores[key] = {}
        backbone_scores[key][row.candidate_item_id] = float(row.backbone_score)
    del df_bs
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    # ── persona ───────────────────────────────────────────────────────
    logger.info("Loading persona...")
    df_persona = pd.read_parquet(
        f"data/cache/persona/{dataset}/persona_graphs.parquet"
    )
    persona_nodes_by_user: dict[str, list[dict]] = {
        uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
    }
    # persona top-10 concept set per user (for generic overlap diagnostic)
    persona_top_by_user: dict[str, set[str]] = {}
    for uid, grp in df_persona.sort_values(
        ["user_id", "weight"], ascending=[True, False]
    ).groupby("user_id"):
        persona_top_by_user[uid] = set(grp["concept_id"].head(10).tolist())

    # ── load intent caches ────────────────────────────────────────────
    logger.info("Loading LLM intent cache (baseline_count validated)...")
    intent_baseline = _load_intent(llm_path, shared_keys)
    logger.info("LLM intent loaded: %d", len(intent_baseline))

    heur_intent: dict[tuple, dict] = {}
    if heur_path.exists():
        logger.info("Loading heuristic intent...")
        heur_intent = _load_intent(heur_path, shared_keys)
        logger.info("Heuristic intent loaded: %d", len(heur_intent))

    # ── re-score with count_x_idf in-memory ───────────────────────────
    logger.info("Re-scoring with count_x_idf...")
    intent_idf = _apply_idf_rescore(
        intent_baseline, cand_by_key, item_concepts,
        idf={},   # not used directly; _IDF_PATH is used via _load_concept_idf
        idf_floor=0.1,
    )
    logger.info("count_x_idf rescoring done: %d records", len(intent_idf))

    # ── concept diagnostics ───────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("CONCEPT DIAGNOSTICS — all reasons")
    print(SEP)
    for label, field, intent_d in [
        ("raw_llm_goals",       "raw_llm_goals",            intent_baseline),
        ("baseline_count val",  "validated_goal_concepts",  intent_baseline),
        ("count_x_idf val",     "validated_goal_concepts",  intent_idf),
    ]:
        d = _aggregate_concept_diag(
            intent_d, field, concept_doc_freq, total_items,
            persona_top_by_user, gt_concepts_by_key,
        )
        print(f"  {label:22s}  avg_doc_freq={d.get('avg_doc_freq',0):8.1f}  "
              f"avg_spec={d.get('avg_specificity',0):.4f}  "
              f"generic_frac={d.get('generic_frac',0):.4f}  "
              f"persona_overlap={d.get('persona_overlap_frac',0):.4f}  "
              f"gt_match={d.get('gt_match',0):.4f}")

    print(f"\n{SEP}")
    print("CONCEPT DIAGNOSTICS — exploration only")
    print(SEP)
    exp_diag_rows = []
    for label, field, intent_d in [
        ("raw_llm_goals",       "raw_llm_goals",            intent_baseline),
        ("baseline_count val",  "validated_goal_concepts",  intent_baseline),
        ("count_x_idf val",     "validated_goal_concepts",  intent_idf),
    ]:
        d = _aggregate_concept_diag(
            intent_d, field, concept_doc_freq, total_items,
            persona_top_by_user, gt_concepts_by_key,
            reason_filter="exploration",
        )
        print(f"  {label:22s}  avg_doc_freq={d.get('avg_doc_freq',0):8.1f}  "
              f"avg_spec={d.get('avg_specificity',0):.4f}  "
              f"generic_frac={d.get('generic_frac',0):.4f}  "
              f"persona_overlap={d.get('persona_overlap_frac',0):.4f}  "
              f"gt_match={d.get('gt_match',0):.4f}")
        exp_diag_rows.append({"selector": label, **d})
    pd.DataFrame(exp_diag_rows).to_csv(
        out_dir / "selector_scoring_exploration.csv", index=False
    )

    # ── run eval ─────────────────────────────────────────────────────
    all_metric_rows: list[dict] = []

    branches = [
        ("LLM+baseline_count", intent_baseline),
        ("LLM+count_x_idf",    intent_idf),
    ]
    if heur_intent:
        branches.append(("heuristic_control", heur_intent))

    for branch_name, intent_d in branches:
        logger.info("%s\nBranch: %s\n%s", SEP, branch_name, SEP)
        # filter intent_d to shared_keys (heuristic may differ)
        intent_filtered = {k: v for k, v in intent_d.items() if k in shared_keys}
        cand_filtered   = {k: v for k, v in cand_by_key.items()   if k in intent_filtered}
        gt_filtered     = {k: v for k, v in is_gt_by_key.items()  if k in intent_filtered}
        bs_filtered     = {k: v for k, v in backbone_scores.items() if k in intent_filtered}

        results = _run_eval(
            branch_name=branch_name,
            intent_by_key=intent_filtered,
            cand_by_key=cand_filtered,
            is_gt_by_key=gt_filtered,
            backbone_scores=bs_filtered,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            modulation_cfg=mod_cfg,
            k_values=k_values,
            concept_doc_freq=concept_doc_freq,
            total_items=total_items,
            persona_top_by_user=persona_top_by_user,
        )
        for exp_name, m in results.items():
            if m:
                all_metric_rows.append(m)

    # ── save + print results ──────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metric_rows)
    csv_path = out_dir / "selector_scoring_comparison.csv"
    df_metrics.to_csv(csv_path, index=False)
    logger.info("saved -> %s", csv_path)

    print(f"\n{SEP}")
    print("SELECTOR SCORING COMPARISON — intent_only")
    print(SEP)
    cols = ["branch", "HR@10", "NDCG@10", "MRR", "gt_delta_pos_frac", "gt_delta_zero_frac"]
    sub = df_metrics[df_metrics["experiment"] == "intent_only"][
        [c for c in cols if c in df_metrics.columns]
    ]
    print(sub.to_string(index=False))

    print(f"\n{SEP}")
    print("SELECTOR SCORING COMPARISON — full_model")
    print(SEP)
    sub = df_metrics[df_metrics["experiment"] == "full_model"][
        [c for c in cols if c in df_metrics.columns]
    ]
    print(sub.to_string(index=False))

    # per-reason HR@10 if available
    reason_cols = [c for c in df_metrics.columns if c.startswith("HR@10_")]
    if reason_cols:
        print(f"\n{SEP}")
        print("PER-REASON HR@10 — intent_only")
        print(SEP)
        sub_r = df_metrics[df_metrics["experiment"] == "intent_only"][
            ["branch"] + reason_cols
        ]
        print(sub_r.to_string(index=False))

    # ── markdown report ───────────────────────────────────────────────
    _write_report(df_metrics, exp_diag_rows, out_dir)
    logger.info("Output: %s", out_dir)


def _write_report(df_metrics: pd.DataFrame, exp_diag_rows: list[dict], out_dir: Path):
    lines = [
        "# Selector Scoring Eval Report — baseline_count vs count_x_idf\n",
        "이번 수정은 grounded selector의 hard activation gate를 유지한 채, "
        "selection score를 specificity-aware하게 바꾸어, "
        "LLM short-term exploration signal이 generic backbone concept에 흡수되지 않고 "
        "persona와 구별되는 discriminative grounded signal이 되도록 만드는 1차 시도이다.\n",
    ]

    # concept diagnostics
    lines.append("## Exploration Concept Diagnostics\n")
    lines.append("| selector | avg_doc_freq | avg_specificity | generic_frac | persona_overlap | gt_match |")
    lines.append("|---|---|---|---|---|---|")
    for d in exp_diag_rows:
        lines.append(
            f"| {d.get('selector','')} "
            f"| {d.get('avg_doc_freq',0):.1f} "
            f"| {d.get('avg_specificity',0):.4f} "
            f"| {d.get('generic_frac',0):.4f} "
            f"| {d.get('persona_overlap_frac',0):.4f} "
            f"| {d.get('gt_match',0):.4f} |"
        )
    lines.append("")

    # metrics table
    for exp in ["intent_only", "full_model"]:
        lines.append(f"## {exp} Metrics\n")
        sub = df_metrics[df_metrics["experiment"] == exp]
        if sub.empty:
            continue
        cols = ["branch", "HR@10", "NDCG@10", "MRR", "gt_delta_pos_frac", "gt_delta_zero_frac"]
        cols = [c for c in cols if c in sub.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in sub[cols].iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        lines.append("")

    # Q&A
    lines.append("## 해석 질문 답변\n")

    # Q1: generic concept 감소?
    raw_gf   = next((d["generic_frac"] for d in exp_diag_rows if "raw" in d["selector"]), None)
    base_gf  = next((d["generic_frac"] for d in exp_diag_rows if "baseline" in d["selector"]), None)
    idf_gf   = next((d["generic_frac"] for d in exp_diag_rows if "idf" in d["selector"]), None)
    if all(v is not None for v in [base_gf, idf_gf]):
        q1 = "감소" if idf_gf < base_gf else "감소하지 않음"
        lines.append(
            f"**Q1. count_x_idf가 exploration generic concept 비중을 줄였는가?** {q1}\n"
            f"- baseline_count generic_frac={base_gf:.4f} → count_x_idf={idf_gf:.4f} "
            f"(delta={idf_gf-base_gf:+.4f})\n"
        )

    # Q2: GT match 개선?
    base_gt = next((d["gt_match"] for d in exp_diag_rows if "baseline" in d["selector"]), None)
    idf_gt  = next((d["gt_match"] for d in exp_diag_rows if "idf" in d["selector"]), None)
    if all(v is not None for v in [base_gt, idf_gt]):
        q2 = "개선됨" if idf_gt > base_gt + 0.001 else "미미하거나 없음"
        lines.append(
            f"**Q2. changed cases에서 GT match가 의미 있게 올랐는가?** {q2}\n"
            f"- baseline_count gt_match={base_gt:.4f} → count_x_idf={idf_gt:.4f}\n"
        )

    # Q3/Q4: HR 개선?
    def _get(branch, exp, metric):
        sub = df_metrics[
            (df_metrics["branch"] == branch) & (df_metrics["experiment"] == exp)
        ]
        return sub.iloc[0].get(metric) if not sub.empty else None

    hr_base_io  = _get("LLM+baseline_count", "intent_only", "HR@10")
    hr_idf_io   = _get("LLM+count_x_idf",   "intent_only", "HR@10")
    hr_base_fm  = _get("LLM+baseline_count", "full_model",  "HR@10")
    hr_idf_fm   = _get("LLM+count_x_idf",   "full_model",  "HR@10")
    hr_heur_io  = _get("heuristic_control",  "intent_only", "HR@10")
    hr_heur_fm  = _get("heuristic_control",  "full_model",  "HR@10")

    if hr_base_io and hr_idf_io:
        q3 = "개선" if hr_idf_io > hr_base_io + 0.001 else ("동일" if abs(hr_idf_io - hr_base_io) <= 0.001 else "하락")
        lines.append(
            f"**Q3. intent_only 성능이 개선되는가?** {q3}\n"
            f"- baseline_count HR@10={hr_base_io} → count_x_idf={hr_idf_io} "
            f"(delta={hr_idf_io-hr_base_io:+.4f})\n"
        )
    if hr_base_fm and hr_idf_fm:
        q4 = "감소" if hr_idf_fm < hr_base_fm - 0.001 else ("동일" if abs(hr_idf_fm - hr_base_fm) <= 0.001 else "개선")
        lines.append(
            f"**Q4. full_model에서 persona와의 충돌이 줄어드는가?** {q4}\n"
            f"- baseline_count HR@10={hr_base_fm} → count_x_idf={hr_idf_fm} "
            f"(delta={hr_idf_fm-hr_base_fm:+.4f})\n"
        )
    if hr_heur_fm and hr_idf_fm:
        gap_base = (hr_heur_fm or 0) - (hr_base_fm or 0)
        gap_idf  = (hr_heur_fm or 0) - (hr_idf_fm or 0)
        q5 = "줄었음" if gap_idf < gap_base - 0.001 else "줄지 않음"
        lines.append(
            f"**Q5. full_model heuristic과의 격차가 줄어드는가?** {q5}\n"
            f"- baseline gap (heur-LLM)={gap_base:+.4f} → idf gap={gap_idf:+.4f}\n"
        )

    out_path = out_dir / "selector_scoring_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report saved -> %s", out_path)


if __name__ == "__main__":
    main()
