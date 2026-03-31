"""
run_idf_selector_eval.py
------------------------
Compares baseline_count vs count_x_idf selector scoring modes.

Produces:
  idf_selector_eval.csv            — per-branch metrics
  idf_selector_grounding.csv       — per-reason grounding stats
  idf_selector_report.md           — markdown report

Branches evaluated:
  llm_baseline_count   — LLM + baseline_count selector (current default)
  llm_count_x_idf      — LLM + count_x_idf selector    (new)
  raw_llm_goals        — LLM raw goals, no Stage 2
  heuristic_control    — heuristic intent baseline

Usage:
  python3 -m src.evaluation.run_idf_selector_eval \\
    --data-config config/data/amazon_movies_tv.yaml \\
    --eval-config  config/evaluation/default.yaml \\
    --mod-config   config/modulation/amazon_movies_tv.yaml \\
    --llm-intent-path  data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet \\
    --idf-intent-path  data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_idf.parquet \\
    --heur-intent-path data/cache/intent/amazon_movies_tv/short_term_intents_heuristic.parquet \\
    --out-dir results/idf_selector_eval
"""
from __future__ import annotations

import argparse
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
    shared_users = {k[0] for k in shared_keys}
    df = df[df["user_id"].isin(shared_users)]
    result = {}
    for r in df.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            result[k] = r
    return result


def _patch_goals(rec: dict, goal_field: str) -> dict:
    """Return shallow copy with goal_concepts overwritten by goal_field."""
    patched = dict(rec)
    src = rec.get(goal_field)
    if src is None:
        return patched
    goals = _to_list(src)
    patched["goal_concepts"] = goals
    patched["validated_goal_concepts"] = goals
    return patched


# ── concept specificity helpers ───────────────────────────────────────

def _build_doc_freq(item_concepts: dict[str, list[str]]) -> dict[str, int]:
    doc_freq: dict[str, int] = {}
    for concepts in item_concepts.values():
        for c in set(concepts):
            doc_freq[c] = doc_freq.get(c, 0) + 1
    return doc_freq


def _avg_specificity(
    concepts: list[str],
    doc_freq: dict[str, int],
    total_items: int,
) -> float:
    if not concepts:
        return 0.0
    return sum(
        math.log(total_items / max(doc_freq.get(c, 1), 1))
        for c in concepts
    ) / len(concepts)


# ── per-branch grounding stats (no GT leak) ───────────────────────────

def _grounding_stats(
    intent_by_key: dict[tuple, dict],
    goal_field: str,
    item_concepts: dict[str, list[str]],
    cand_by_key: dict[tuple, list[str]],
    doc_freq: dict[str, int],
    total_items: int,
) -> pd.DataFrame:
    from src.intent.grounded_selector import build_candidate_concept_bank

    rows = []
    for key, rec in intent_by_key.items():
        goals = _to_list(rec.get(goal_field) or rec.get("goal_concepts"))
        reason = rec.get("deviation_reason", "unknown")
        cands = cand_by_key.get(key, [])
        bank = build_candidate_concept_bank(cands, item_concepts)
        n_active = sum(1 for c in goals if bank.get(c, 0) >= 1)
        avg_act = (sum(bank.get(c, 0) for c in goals) / len(goals)) if goals else 0.0
        spec = _avg_specificity(goals, doc_freq, total_items)
        rows.append({
            "user_id": key[0], "target_index": key[1],
            "reason": reason,
            "n_goals": len(goals),
            "n_activated": n_active,
            "any_activated": int(n_active > 0),
            "avg_activation": round(avg_act, 4),
            "avg_specificity": round(spec, 4),
        })
    return pd.DataFrame(rows)


# ── GT-based diagnostics (eval only, no leakage into scoring path) ────

def _gt_stats(
    intent_by_key: dict[tuple, dict],
    goal_field: str,
    item_concepts: dict[str, list[str]],
    is_gt_by_key: dict[tuple, set[str]],
) -> pd.DataFrame:
    rows = []
    for key, rec in intent_by_key.items():
        goals = set(_to_list(rec.get(goal_field) or rec.get("goal_concepts")))
        gt_items = is_gt_by_key.get(key, set())
        gt_concepts: set[str] = set()
        for iid in gt_items:
            gt_concepts.update(item_concepts.get(iid, []))
        gt_match = len(goals & gt_concepts) / max(len(goals), 1)
        rows.append({
            "user_id": key[0], "target_index": key[1],
            "reason": rec.get("deviation_reason", "unknown"),
            "gt_match": round(gt_match, 4),
            "zero_gt": int(gt_match == 0.0),
        })
    return pd.DataFrame(rows)


# ── core eval loop ────────────────────────────────────────────────────

def _run_eval(
    branch_name: str,
    intent_by_key: dict[tuple, dict],
    goal_field: str,
    cand_by_key: dict[tuple, list[str]],
    is_gt_by_key: dict[tuple, set[str]],
    backbone_scores: dict[tuple, dict[str, float]],
    persona_nodes_by_user: dict[str, list[dict]],
    item_concepts: dict[str, list[str]],
    modulation_cfg: dict,
    k_values: list[int],
    experiment_modes: dict[str, str],
) -> dict[str, dict]:
    from src.modulation.gate import compute_gate_strength
    from src.modulation.reranker import CandidateReranker
    from src.modulation.signal_builder import build_signal

    reranker = CandidateReranker(modulation_cfg, item_concepts)
    results: dict[str, dict] = {}

    for exp_name, mode in experiment_modes.items():
        all_ranked: list[dict] = []

        for (uid, tidx), cand_ids in cand_by_key.items():
            if (uid, tidx) not in intent_by_key:
                continue
            scores = backbone_scores.get((uid, tidx), {})
            candidate_tuples = sorted(
                [(iid, scores[iid]) for iid in cand_ids if iid in scores],
                key=lambda x: x[1], reverse=True,
            )
            intent_record = _patch_goals(intent_by_key[(uid, tidx)], goal_field)
            reason     = intent_record.get("deviation_reason", "unknown")
            confidence = float(intent_record.get("confidence", 0.5))
            persona_nodes = persona_nodes_by_user.get(uid, [])
            gate_strength = compute_gate_strength(
                deviation_reason=reason, confidence=confidence,
                persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_record, persona_nodes, gate_strength, modulation_cfg, mode=mode)
            ranked = reranker.rerank(candidate_tuples, signal, mode=mode)
            gt_items = is_gt_by_key.get((uid, tidx), set())
            for r in ranked:
                rec = r.to_record()
                rec["is_ground_truth"] = int(rec["candidate_item_id"] in gt_items)
                all_ranked.append(rec)

        df = pd.DataFrame(all_ranked)
        if df.empty or "is_ground_truth" not in df.columns:
            results[exp_name] = {}
            continue

        df = df.sort_values(
            ["user_id", "target_index", "final_score"], ascending=[True, True, False]
        )
        df["_rank"] = df.groupby(["user_id", "target_index"]).cumcount() + 1
        gt_df = df[df["is_ground_truth"] == 1].copy()
        if gt_df.empty:
            results[exp_name] = {}
            continue

        ranks = gt_df["_rank"].values.astype(int)
        n = len(ranks)
        m: dict = {"branch": branch_name, "experiment": exp_name, "n_users": n}
        for k in k_values:
            m[f"HR@{k}"]   = round(float((ranks <= k).mean()), 4)
            m[f"NDCG@{k}"] = round(
                float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n), 4
            )
        m["MRR"] = round(float((1.0 / ranks).mean()), 4)

        if "modulation_delta" in gt_df.columns:
            deltas = gt_df["modulation_delta"]
            m["gt_delta_pos_frac"]  = round(float((deltas > 0).mean()), 4)
            m["gt_delta_zero_frac"] = round(float((deltas == 0).mean()), 4)
        else:
            m["gt_delta_pos_frac"] = m["gt_delta_zero_frac"] = None

        results[exp_name] = m
        logger.info(
            "[%s / %s]  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  gt_nz=%.3f",
            branch_name, exp_name,
            m.get("HR@10", 0), m.get("NDCG@10", 0), m.get("MRR", 0),
            m.get("gt_delta_pos_frac", 0) or 0,
        )
    return results


# ── report ────────────────────────────────────────────────────────────

def _write_report(
    df_metrics: pd.DataFrame,
    df_grounding: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = ["# IDF Selector Eval Report\n"]
    lines.append(
        "이번 수정은 grounded selector의 hard activation gate를 유지한 채, "
        "selection score를 specificity-aware하게 바꾸어, "
        "LLM short-term exploration signal이 generic backbone concept에 흡수되지 않고 "
        "persona와 구별되는 discriminative grounded signal이 되도록 만드는 1차 시도이다.\n"
    )

    lines.append("## Metrics\n")
    for exp in df_metrics["experiment"].unique():
        sub = df_metrics[df_metrics["experiment"] == exp]
        lines.append(f"### {exp}\n")
        cols = [c for c in ["branch", "HR@10", "NDCG@10", "MRR",
                             "gt_delta_pos_frac", "gt_delta_zero_frac", "n_users"]
                if c in sub.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in sub[cols].iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        lines.append("")

    lines.append("## Grounding Stats (per reason)\n")
    for branch in df_grounding["branch"].unique():
        lines.append(f"### {branch}\n")
        sub = df_grounding[df_grounding["branch"] == branch]
        grp = sub.groupby("reason").agg(
            n=("n_goals", "count"),
            avg_n_goals=("n_goals", "mean"),
            any_activated=("any_activated", "mean"),
            avg_activation=("avg_activation", "mean"),
            avg_specificity=("avg_specificity", "mean"),
            gt_zero_frac=("zero_gt", "mean") if "zero_gt" in sub.columns else ("any_activated", "mean"),
        ).round(4)
        lines.append(grp.to_string())
        lines.append("")

    lines.append("## Q&A\n")
    # Q1: generic concept 비중 감소
    base = df_grounding[df_grounding["branch"] == "llm_baseline_count"]
    idf  = df_grounding[df_grounding["branch"] == "llm_count_x_idf"]
    if not base.empty and not idf.empty:
        base_exp = base[base["reason"] == "exploration"]["avg_specificity"].mean()
        idf_exp  = idf[idf["reason"]  == "exploration"]["avg_specificity"].mean()
        q1 = "Yes" if idf_exp > base_exp else "No"
        lines.append(f"**Q1. count_x_idf가 exploration에서 generic concept 비중을 줄였는가?** {q1}")
        lines.append(f"- baseline avg_specificity(exp): {base_exp:.4f}")
        lines.append(f"- count_x_idf avg_specificity(exp): {idf_exp:.4f}\n")

    # Q3/Q4: intent-only / full_model
    for exp_label, exp_key in [("Q3. intent-only 성능", "ablation_intent_only"),
                                ("Q4. full_model 성능", "graph_conditioned_full")]:
        sub = df_metrics[df_metrics["experiment"] == exp_key]
        if sub.empty:
            continue
        lines.append(f"**{exp_label}**")
        for _, row in sub.iterrows():
            lines.append(f"- {row['branch']}: HR@10={row.get('HR@10', '?')}  NDCG@10={row.get('NDCG@10', '?')}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("saved -> %s", out_path)


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",      default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--eval-config",      default="config/evaluation/default.yaml")
    parser.add_argument("--mod-config",       default="config/modulation/amazon_movies_tv.yaml")
    parser.add_argument("--llm-intent-path",  default=None,
                        help="LLM intent with baseline_count Stage 2 "
                             "(short_term_intents_llm_subset_2000_validated.parquet)")
    parser.add_argument("--idf-intent-path",  default=None,
                        help="LLM intent with count_x_idf Stage 2 "
                             "(short_term_intents_llm_subset_2000_idf.parquet). "
                             "If omitted, count_x_idf branch is applied on-the-fly.")
    parser.add_argument("--heur-intent-path", default=None)
    parser.add_argument("--idf-path",
                        default="data/cache/concept_idf/amazon_movies_tv/concept_idf.parquet")
    parser.add_argument("--max-users",  type=int, default=2000)
    parser.add_argument("--out-dir",    default="results/idf_selector_eval")
    args = parser.parse_args()

    data_cfg = _load_yaml(args.data_config)
    eval_cfg = _load_yaml(args.eval_config)
    mod_cfg  = _load_yaml(args.mod_config)

    dataset       = data_cfg.get("dataset", "amazon_movies_tv")
    k_values      = eval_cfg.get("k_values", [5, 10, 20])
    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve paths ────────────────────────────────────────────────
    llm_path = Path(
        args.llm_intent_path or
        f"data/cache/intent/{dataset}/short_term_intents_llm_subset_2000_validated.parquet"
    )
    cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k101.parquet")
    bs_path   = Path(
        f"data/cache/backbone/{dataset}/backbone_scores.parquet"
    )

    for p, label in [(llm_path, "llm_intent"), (cand_path, "candidates"), (bs_path, "backbone_scores")]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            return

    # ── shared keys ──────────────────────────────────────────────────
    logger.info("Computing shared keys...")
    df_cid = pd.read_parquet(cand_path, columns=["user_id", "target_index"])
    cand_keys = set(zip(df_cid["user_id"], df_cid["target_index"].astype(int)))
    df_lid = pd.read_parquet(llm_path, columns=["user_id", "target_index"])
    llm_keys = set(zip(df_lid["user_id"], df_lid["target_index"].astype(int)))
    users_sorted = sorted({k[0] for k in llm_keys})[:args.max_users]
    users_set = set(users_sorted)
    shared_keys = {k for k in (cand_keys & llm_keys) if k[0] in users_set}
    logger.info("shared_keys: %d  users: %d", len(shared_keys), len(users_set))

    # ── load intent caches ───────────────────────────────────────────
    logger.info("Loading LLM intent (baseline_count)...")
    llm_by_key = _load_intent(llm_path, shared_keys)

    # count_x_idf intent: load from precomputed file or apply on-the-fly
    idf_by_key: dict[tuple, dict] = {}
    if args.idf_intent_path and Path(args.idf_intent_path).exists():
        logger.info("Loading LLM intent (count_x_idf) from %s...", args.idf_intent_path)
        idf_by_key = _load_intent(Path(args.idf_intent_path), shared_keys)
    else:
        logger.info("count_x_idf intent not precomputed — applying on-the-fly...")
        idf_by_key = _apply_idf_scoring_on_the_fly(llm_by_key, args.idf_path, cand_path)

    heur_by_key: dict[tuple, dict] = {}
    if args.heur_intent_path and Path(args.heur_intent_path).exists():
        logger.info("Loading heuristic intent...")
        heur_by_key = _load_intent(Path(args.heur_intent_path), shared_keys)

    # ── load shared data ─────────────────────────────────────────────
    logger.info("Loading item_concepts, persona, candidates...")
    df_ic = pd.read_parquet(processed_dir / "item_concepts.parquet")
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    doc_freq = _build_doc_freq(item_concepts)
    total_items = len(item_concepts)

    df_persona = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")
    persona_nodes_by_user = {
        uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
    }

    df_cands = pd.read_parquet(cand_path)
    df_cands = df_cands[df_cands["user_id"].isin(users_set)]
    cand_by_key: dict[tuple, list[str]] = {}
    is_gt_by_key: dict[tuple, set[str]] = {}
    for row in df_cands.itertuples(index=False):
        key = (str(row.user_id), int(row.target_index))
        if key not in shared_keys:
            continue
        cand_by_key.setdefault(key, []).append(row.candidate_item_id)
        is_gt_by_key.setdefault(key, set())
        if row.is_ground_truth:
            is_gt_by_key[key].add(row.candidate_item_id)
    logger.info("Candidate index: %d keys", len(cand_by_key))

    logger.info("Loading backbone scores...")
    df_bs = pd.read_parquet(bs_path)
    df_bs = df_bs[df_bs["user_id"].isin(users_set)]
    backbone_scores: dict[tuple, dict[str, float]] = {}
    for row in df_bs.itertuples(index=False):
        key = (str(row.user_id), int(row.target_index))
        if key not in shared_keys:
            continue
        backbone_scores.setdefault(key, {})[row.candidate_item_id] = float(row.backbone_score)
    del df_bs
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    experiment_modes = {
        "ablation_intent_only": "intent_only_rerank",
        "graph_conditioned_full": "graph_conditioned_full",
    }

    # ── define branches ───────────────────────────────────────────────
    # (label, intent_dict, goal_field)
    branches: list[tuple[str, dict, str]] = [
        ("llm_baseline_count",  llm_by_key,  "validated_goal_concepts"),
        ("llm_count_x_idf",     idf_by_key,  "validated_goal_concepts"),
        ("raw_llm_goals",       llm_by_key,  "raw_llm_goals"),
    ]
    if heur_by_key:
        branches.append(("heuristic_control", heur_by_key, "goal_concepts"))

    # ── run eval ─────────────────────────────────────────────────────
    all_metrics: list[dict] = []
    all_grounding: list[pd.DataFrame] = []

    for branch_name, intent_dict, goal_field in branches:
        if not intent_dict:
            logger.warning("  %s: empty intent — skipping", branch_name)
            continue
        logger.info("=" * 60)
        logger.info("Branch: %s  (goal_field=%s)", branch_name, goal_field)
        logger.info("=" * 60)

        results = _run_eval(
            branch_name=branch_name,
            intent_by_key=intent_dict,
            goal_field=goal_field,
            cand_by_key=cand_by_key,
            is_gt_by_key=is_gt_by_key,
            backbone_scores=backbone_scores,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            modulation_cfg=mod_cfg,
            k_values=k_values,
            experiment_modes=experiment_modes,
        )
        for m in results.values():
            if m:
                all_metrics.append(m)

        # grounding stats (no GT)
        gdf = _grounding_stats(
            intent_dict, goal_field, item_concepts, cand_by_key, doc_freq, total_items
        )
        gdf["branch"] = branch_name
        # GT stats — eval only
        gtdf = _gt_stats(intent_dict, goal_field, item_concepts, is_gt_by_key)
        gtdf["branch"] = branch_name
        merged = gdf.merge(gtdf[["user_id", "target_index", "gt_match", "zero_gt"]],
                           on=["user_id", "target_index"], how="left")
        all_grounding.append(merged)

    df_metrics   = pd.DataFrame(all_metrics)
    df_grounding = pd.concat(all_grounding, ignore_index=True) if all_grounding else pd.DataFrame()

    # ── save ─────────────────────────────────────────────────────────
    df_metrics.to_csv(out_dir / "idf_selector_eval.csv", index=False)
    logger.info("saved -> %s", out_dir / "idf_selector_eval.csv")

    if not df_grounding.empty:
        df_grounding.to_csv(out_dir / "idf_selector_grounding.csv", index=False)
        logger.info("saved -> %s", out_dir / "idf_selector_grounding.csv")

    # ── console summary ───────────────────────────────────────────────
    SEP = "=" * 72
    for exp in ["ablation_intent_only", "graph_conditioned_full"]:
        sub = df_metrics[df_metrics["experiment"] == exp] if not df_metrics.empty else pd.DataFrame()
        if sub.empty:
            continue
        print(f"\n{SEP}")
        print(f"{'INTENT-ONLY' if 'intent' in exp else 'FULL MODEL'} — {exp}")
        print(SEP)
        cols = [c for c in ["branch", "HR@10", "NDCG@10", "MRR",
                             "gt_delta_pos_frac", "gt_delta_zero_frac"] if c in sub.columns]
        print(sub[cols].to_string(index=False))

    print(f"\n{SEP}")
    print("GROUNDING STATS — exploration (avg_specificity comparison)")
    print(SEP)
    if not df_grounding.empty:
        exp_grp = df_grounding[df_grounding["reason"] == "exploration"]
        if not exp_grp.empty:
            summary = exp_grp.groupby("branch").agg(
                n=("n_goals", "count"),
                avg_n_goals=("n_goals", "mean"),
                avg_specificity=("avg_specificity", "mean"),
                avg_activation=("avg_activation", "mean"),
                any_activated=("any_activated", "mean"),
                gt_zero_frac=("zero_gt", "mean"),
            ).round(4)
            print(summary.to_string())

    _write_report(df_metrics, df_grounding, out_dir / "idf_selector_report.md")
    print(f"\nOutput: {out_dir}")


def _apply_idf_scoring_on_the_fly(
    llm_by_key: dict[tuple, dict],
    idf_path: str,
    cand_path: Path,
) -> dict[tuple, dict]:
    """
    Apply count_x_idf scoring on-the-fly to existing LLM intent records.
    Re-ranks the Stage-2-surviving concept set using IDF-weighted scores.
    Does NOT re-run the full Stage 2 pipeline — only changes the ordering/selection
    among already-activated concepts.
    """
    import math as _math
    import pandas as _pd

    logger.info("Building on-the-fly count_x_idf reranking...")

    # Load IDF
    idf: dict[str, float] = {}
    idf_floor = 0.1
    if idf_path and Path(idf_path).exists():
        df_idf = _pd.read_parquet(idf_path)
        idf = dict(zip(df_idf["concept_id"], df_idf["idf_weight"].astype(float)))
        logger.info("  Loaded IDF: %d entries", len(idf))
    else:
        logger.warning("  IDF path not found: %s — using idf_floor only", idf_path)

    # Load candidate concept banks (needed for activation_count)
    from src.intent.grounded_selector import build_candidate_concept_bank

    # Load item_concepts
    # We'll use already-loaded item_concepts from the caller's scope indirectly:
    # but here we need to rebuild. Load directly.
    ic_path = Path("data/processed/amazon_movies_tv/item_concepts.parquet")
    df_ic = _pd.read_parquet(ic_path)
    item_concepts_local: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )

    df_cands = _pd.read_parquet(cand_path)
    bank_by_key: dict[tuple, dict[str, int]] = {}
    for (uid, tidx), grp in df_cands.groupby(["user_id", "target_index"]):
        key = (str(uid), int(tidx))
        bank_by_key[key] = build_candidate_concept_bank(
            grp["candidate_item_id"].tolist(), item_concepts_local
        )
    logger.info("  Banks built: %d keys", len(bank_by_key))

    def _to_list_local(x):
        if x is None: return []
        if isinstance(x, str):
            try: return json.loads(x)
            except: return []
        if hasattr(x, "__iter__"): return list(x)
        return []

    result: dict[tuple, dict] = {}
    max_goals_map = {"aligned": 3, "task_focus": 2, "exploration": 3, "budget_shift": 2, "unknown": 1}

    for key, rec in llm_by_key.items():
        # Start from already-validated (activation-gated) concepts
        validated = _to_list_local(rec.get("validated_goal_concepts") or rec.get("goal_concepts"))
        bank = bank_by_key.get(key, {})
        reason = rec.get("deviation_reason", "unknown")
        max_goals = max_goals_map.get(reason, 3)

        # Re-sort by count_x_idf score
        reranked = sorted(
            validated,
            key=lambda c: (-(bank.get(c, 0) * idf.get(c, idf_floor)), c),
        )[:max_goals]

        new_rec = dict(rec)
        new_rec["validated_goal_concepts"] = reranked
        result[key] = new_rec

    logger.info("  On-the-fly IDF reranking complete: %d records", len(result))
    return result


if __name__ == "__main__":
    main()
