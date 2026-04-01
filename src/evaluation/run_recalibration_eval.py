"""
run_recalibration_eval.py
-------------------------
Ablation evaluation for exploration recalibration.

Compares four branches:
  A. current_baseline         — original LLM deviation_reason, no recalibration
  B. recalibrated_reason_only — recalibrated_reason replaces deviation_reason;
                                same raw/validated goals
  C. recalibrated_full        — recalibrated_reason + validated goals (main candidate)
  D. heuristic_control        — heuristic intent baseline

All branches use the same backbone candidates and GT set.

Metrics: HR@10, NDCG@10, MRR, gt_delta_pos_frac, gt_delta_zero_frac
Per-reason breakdown included.

Usage:
  python -m src.evaluation.run_recalibration_eval \\
    --data-config   config/data/amazon_movies_tv.yaml \\
    --eval-config   config/evaluation/default.yaml \\
    --mod-config    config/modulation/amazon_movies_tv.yaml \\
    --llm-intent-path  data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet \\
    --heur-intent-path data/cache/intent/amazon_movies_tv/short_term_intents_heuristic.parquet \\
    --backbone-candidates-path data/cache/candidate/amazon_movies_tv/sampled_candidates_k101.parquet \\
    --out-dir results/recalibration_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _to_list(x) -> list:
    if x is None: return []
    if hasattr(x, "__iter__") and not isinstance(x, str): return list(x)
    return []


def _intent_keys(path: Path) -> set[tuple[str, int]]:
    df = pd.read_parquet(path, columns=["user_id", "target_index"])
    return set(zip(df["user_id"], df["target_index"].astype(int)))


def _load_intent_filtered(
    path: Path, shared_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict]:
    df = pd.read_parquet(path)
    result = {}
    for r in df.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            result[k] = r
    return result


def _compute_metrics(ranks: np.ndarray, deltas: np.ndarray | None, k_values: list[int]) -> dict:
    n = len(ranks)
    if n == 0:
        return {}
    m: dict = {"n": n}
    for k in k_values:
        m[f"HR@{k}"]   = round(float((ranks <= k).mean()), 4)
        m[f"NDCG@{k}"] = round(
            float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n), 4
        )
    m["MRR"] = round(float((1.0 / ranks).mean()), 4)
    if deltas is not None and len(deltas) > 0:
        m["gt_delta_mean"]      = round(float(deltas.mean()), 6)
        m["gt_delta_pos_frac"]  = round(float((deltas > 0).mean()), 4)
        m["gt_delta_zero_frac"] = round(float((deltas == 0).mean()), 4)
        m["gt_delta_neg_frac"]  = round(float((deltas < 0).mean()), 4)
    return m


def _run_eval_branch(
    branch_name: str,
    intent_by_key: dict[tuple[str, int], dict],
    cand_by_key: dict[tuple[str, int], list[str]],
    backbone_scores: dict[tuple[str, int], dict[str, float]],
    persona_nodes_by_user: dict[str, list[dict]],
    item_concepts: dict[str, list[str]],
    gt_items_by_key: dict[tuple[str, int], set[str]],
    modulation_cfg: dict,
    k_values: list[int],
    experiment_modes: dict[str, str],
) -> dict[str, dict]:
    """Run eval for one branch across all experiment modes."""
    from src.modulation.gate import compute_gate_strength
    from src.modulation.reranker import CandidateReranker
    from src.modulation.signal_builder import build_signal

    reranker = CandidateReranker(modulation_cfg, item_concepts)
    results_by_mode: dict[str, dict] = {}

    for mode_name, mode in experiment_modes.items():
        all_ranked: list[dict] = []

        for (uid, tidx), cand_ids in cand_by_key.items():
            intent_rec = intent_by_key.get((uid, tidx))
            if intent_rec is None:
                continue
            scores = backbone_scores.get((uid, tidx), {})
            candidate_tuples = sorted(
                [(iid, scores.get(iid, 0.0)) for iid in cand_ids],
                key=lambda x: x[1], reverse=True,
            )

            reason    = intent_rec.get("recalibrated_reason") or intent_rec.get("deviation_reason", "unknown")
            confidence = float(intent_rec.get("confidence", 0.5))
            alignment  = float(intent_rec.get("persona_alignment_score", 0.0))

            persona_nodes = persona_nodes_by_user.get(uid, [])
            gate_strength = compute_gate_strength(
                deviation_reason=reason,
                confidence=confidence,
                persona_alignment_score=alignment,
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_rec, persona_nodes, gate_strength, modulation_cfg, mode=mode)
            ranked = reranker.rerank(candidate_tuples, signal, mode=mode)

            gt_items = gt_items_by_key.get((uid, tidx), set())
            for r in ranked:
                rec = r.to_record()
                rec["is_gt"]          = int(rec["candidate_item_id"] in gt_items)
                rec["deviation_reason"] = intent_rec.get("deviation_reason", "unknown")
                rec["recalibrated_reason"] = intent_rec.get("recalibrated_reason",
                                             intent_rec.get("deviation_reason", "unknown"))
                all_ranked.append(rec)

        df_r = pd.DataFrame(all_ranked)
        if df_r.empty:
            results_by_mode[mode_name] = {}
            continue

        df_r = df_r.sort_values(
            ["user_id", "target_index", "final_score"], ascending=[True, True, False]
        )
        df_r["_rank"] = df_r.groupby(["user_id", "target_index"]).cumcount() + 1
        gt_df = df_r[df_r["is_gt"] == 1].copy()

        if gt_df.empty:
            results_by_mode[mode_name] = {}
            continue

        ranks  = gt_df["_rank"].values.astype(int)
        deltas = gt_df["modulation_delta"].values if "modulation_delta" in gt_df.columns else None

        metrics = _compute_metrics(ranks, deltas, k_values)
        metrics["branch"]     = branch_name
        metrics["mode"]       = mode_name

        # per-reason breakdown
        reason_col = "recalibrated_reason"
        for rsn, rg in gt_df.groupby(reason_col):
            rranks  = rg["_rank"].values.astype(int)
            rdeltas = rg["modulation_delta"].values if "modulation_delta" in rg.columns else None
            rm = _compute_metrics(rranks, rdeltas, k_values)
            metrics[f"reason_{rsn}"] = rm

        results_by_mode[mode_name] = metrics
        logger.info(
            "[%s/%s] HR@10=%.4f NDCG@10=%.4f MRR=%.4f gt_delta_zero=%.3f",
            branch_name, mode_name,
            metrics.get("HR@10", 0), metrics.get("NDCG@10", 0),
            metrics.get("MRR", 0), metrics.get("gt_delta_zero_frac", 0),
        )

    return results_by_mode


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config",   required=True)
    ap.add_argument("--eval-config",   required=True)
    ap.add_argument("--mod-config",    required=True)
    ap.add_argument("--llm-intent-path",  required=True)
    ap.add_argument("--heur-intent-path", default=None,
                    help="Heuristic intent parquet for branch D. Optional.")
    ap.add_argument("--backbone-candidates-path", required=True)
    ap.add_argument("--recal-cfg-path", default=None,
                    help="YAML with recalibration threshold overrides (optional).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--interaction-window", type=int, default=10,
                    help="Recent interaction window size for recalibration signals (default: 10).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(args.data_config)
    eval_cfg = _load_yaml(args.eval_config)
    mod_cfg  = _load_yaml(args.mod_config)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    dataset       = data_cfg.get("dataset", "unknown")

    k_values = eval_cfg.get("k_values", [5, 10, 20])
    experiment_modes: dict[str, str] = eval_cfg.get("experiment_modes", {
        "full_model":    "graph_conditioned_full",
        "intent_only":   "intent_only_rerank",
        "persona_only":  "persona_only_rerank",
    })

    recal_cfg_override: dict = {}
    if args.recal_cfg_path:
        recal_cfg_override = _load_yaml(args.recal_cfg_path)
        logger.info("Loaded recalibration config overrides: %s", recal_cfg_override)

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading item concepts...")
    df_ic = pd.read_parquet(processed_dir / "item_concepts.parquet")
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )

    logger.info("Loading persona...")
    df_persona = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")
    persona_nodes_by_user: dict[str, list[dict]] = {
        uid: g.to_dict("records")
        for uid, g in df_persona.groupby("user_id")
    }

    logger.info("Loading interactions for recalibration signals...")
    df_inter = pd.read_parquet(processed_dir / "interactions.parquet")

    logger.info("Loading IDF for recalibration...")
    idf_path = f"data/cache/concept_idf/{dataset}/concept_idf.parquet"
    df_idf = pd.read_parquet(idf_path)
    doc_freq_map: dict[str, int] = dict(zip(df_idf["concept_id"], df_idf["doc_freq"]))

    logger.info("Loading backbone candidates...")
    df_cands = pd.read_parquet(args.backbone_candidates_path)

    logger.info("Loading LLM intent...")
    df_llm = pd.read_parquet(args.llm_intent_path)
    if args.max_users:
        users = sorted(df_llm["user_id"].unique())[: args.max_users]
        df_llm = df_llm[df_llm["user_id"].isin(users)].reset_index(drop=True)
    logger.info("LLM intent: %d records", len(df_llm))

    # ── Build shared keys ────────────────────────────────────────────
    llm_keys: set[tuple[str, int]] = set(
        zip(df_llm["user_id"], df_llm["target_index"].astype(int))
    )

    # ── Build backbone candidate/score lookups ────────────────────────
    cand_by_key: dict[tuple[str, int], list[str]] = {}
    backbone_scores: dict[tuple[str, int], dict[str, float]] = {}

    if "candidate_item_ids" in df_cands.columns:
        for row in df_cands.itertuples(index=False):
            k = (str(row.user_id), int(row.target_index))
            if k not in llm_keys: continue
            cand_by_key[k] = _to_list(row.candidate_item_ids)
    else:
        item_col = next((c for c in ("item_id","candidate_item_id") if c in df_cands.columns), None)
        if item_col is None:
            raise ValueError(f"Cannot find item column in candidates: {list(df_cands.columns)}")
        for (uid, tidx), grp in df_cands.groupby(["user_id","target_index"]):
            k = (str(uid), int(tidx))
            if k not in llm_keys: continue
            cand_by_key[k] = grp[item_col].tolist()

    # Load backbone scores
    backbone_path = f"data/cache/backbone/{dataset}/backbone_scores.parquet"
    if Path(backbone_path).exists():
        df_bb = pd.read_parquet(backbone_path)
        for (uid, tidx), grp in df_bb.groupby(["user_id","target_index"]):
            k = (str(uid), int(tidx))
            if k in cand_by_key:
                backbone_scores[k] = dict(zip(grp["candidate_item_id"], grp["backbone_score"]))
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    # ── GT items lookup ───────────────────────────────────────────────
    inter_sorted = df_inter.sort_values(["user_id","timestamp"])
    user_items_list: dict[str, list[str]] = (
        inter_sorted.groupby("user_id")["item_id"].apply(list).to_dict()
    )
    gt_items_by_key: dict[tuple[str, int], set[str]] = {}
    for (uid, tidx) in cand_by_key:
        items = user_items_list.get(uid, [])
        if tidx < len(items):
            gt_items_by_key[(uid, tidx)] = {items[tidx]}

    shared_keys = {k for k in cand_by_key if k in gt_items_by_key and k in backbone_scores}
    logger.info("Shared eval keys: %d", len(shared_keys))

    # Filter intent to shared keys
    llm_intent_by_key: dict[tuple[str, int], dict] = {}
    for r in df_llm.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            llm_intent_by_key[k] = r
    logger.info("LLM intent after shared key filter: %d", len(llm_intent_by_key))

    # ── Build recalibration signals ───────────────────────────────────
    from src.intent.exploration_recalibrator import (
        build_persona_sc_map,
        build_recent_sem_freq_map,
        recalibrate_dataframe,
        RECAL_CFG,
    )

    logger.info("Building persona semantic-core map...")
    persona_sc_by_user = build_persona_sc_map(df_persona)

    logger.info("Building recent semantic freq map (window=%d)...", args.interaction_window)
    df_llm_filtered = df_llm[df_llm.apply(
        lambda r: (r["user_id"], int(r["target_index"])) in shared_keys, axis=1
    )].copy()
    recent_sem_freq_by_key = build_recent_sem_freq_map(
        df_inter, df_llm_filtered, item_concepts, window=args.interaction_window,
    )

    # ── Apply recalibration → create branch B/C intent ───────────────
    logger.info("Applying recalibration rules...")
    cfg_to_use = {**RECAL_CFG, **recal_cfg_override}
    df_recal = recalibrate_dataframe(
        df_llm_filtered, persona_sc_by_user, recent_sem_freq_by_key, doc_freq_map, cfg=cfg_to_use,
    )

    # Log recalibrated reason distribution
    exp_mask = df_recal["deviation_reason"] == "exploration"
    logger.info("=== Recalibrated reason distribution (exploration rows only) ===")
    for reason, cnt in df_recal.loc[exp_mask, "recalibrated_reason"].value_counts().items():
        frac = cnt / exp_mask.sum()
        logger.info("  %-25s %4d  (%.1f%% of exploration)", reason, cnt, 100 * frac)

    logger.info("=== Overall reason distribution after recalibration ===")
    for reason, cnt in df_recal["recalibrated_reason"].value_counts().items():
        logger.info("  %-25s %4d  (%.1f%%)", reason, cnt, 100 * cnt / len(df_recal))

    # ── Branch intent dicts ───────────────────────────────────────────
    # Branch A: original LLM (no recalibration — recalibrated_reason = deviation_reason)
    branch_a_by_key: dict[tuple[str, int], dict] = {}
    for r in df_llm_filtered.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            # Ensure recalibrated_reason == deviation_reason for branch A
            branch_a_by_key[k] = {**r, "recalibrated_reason": r.get("deviation_reason", "unknown")}

    # Branch B/C: recalibrated
    branch_bc_by_key: dict[tuple[str, int], dict] = {}
    for r in df_recal.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            branch_bc_by_key[k] = r

    # ── Branch D: heuristic control ───────────────────────────────────
    branch_d_by_key: dict[tuple[str, int], dict] = {}
    if args.heur_intent_path:
        df_heur = pd.read_parquet(args.heur_intent_path)
        for r in df_heur.to_dict("records"):
            k = (r["user_id"], int(r["target_index"]))
            if k in shared_keys:
                branch_d_by_key[k] = {**r, "recalibrated_reason": r.get("deviation_reason","unknown")}
        logger.info("Heuristic intent loaded: %d keys", len(branch_d_by_key))
    else:
        logger.info("No heuristic intent path provided — branch D skipped.")

    # ── Run eval branches ─────────────────────────────────────────────
    all_results: list[dict] = []

    branches = [
        ("A_current_baseline",     branch_a_by_key),
        ("B_recalibrated",         branch_bc_by_key),
    ]
    if branch_d_by_key:
        branches.append(("D_heuristic_control", branch_d_by_key))

    for branch_name, intent_by_key in branches:
        logger.info("=== Running branch: %s ===", branch_name)
        res_by_mode = _run_eval_branch(
            branch_name=branch_name,
            intent_by_key=intent_by_key,
            cand_by_key={k: v for k,v in cand_by_key.items() if k in intent_by_key},
            backbone_scores=backbone_scores,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            gt_items_by_key=gt_items_by_key,
            modulation_cfg=mod_cfg,
            k_values=k_values,
            experiment_modes=experiment_modes,
        )
        for mode, metrics in res_by_mode.items():
            flat = {"branch": branch_name, "mode": mode, **{
                k: v for k, v in metrics.items()
                if not isinstance(v, dict)
            }}
            all_results.append(flat)

    # ── Save results ──────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    results_path = out_dir / "recalibration_eval_results.csv"
    df_results.to_csv(results_path, index=False)
    logger.info("Saved results -> %s", results_path)

    # Save recalibrated intent parquet
    recal_parquet_path = out_dir / "intent_recalibrated.parquet"
    save_cols = [c for c in df_recal.columns
                 if not isinstance(df_recal[c].iloc[0] if len(df_recal) > 0 else None, dict)]
    df_recal[save_cols].to_parquet(recal_parquet_path, index=False)
    logger.info("Saved recalibrated intent -> %s", recal_parquet_path)

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECALIBRATION EVAL — SUMMARY")
    print("=" * 70)

    for mode in experiment_modes:
        print(f"\n--- Mode: {mode} ---")
        mode_rows = df_results[df_results["mode"] == mode]
        cols = ["branch"] + [f"HR@{k}" for k in k_values] + \
               [f"NDCG@{k}" for k in k_values] + ["MRR",
                "gt_delta_pos_frac", "gt_delta_zero_frac"]
        available = [c for c in cols if c in mode_rows.columns]
        print(mode_rows[available].to_string(index=False))

    # Reason distribution change table
    print("\n--- Reason distribution: original vs recalibrated (exploration rows) ---")
    before = df_recal.loc[exp_mask, "deviation_reason"].value_counts().to_dict()
    after  = df_recal.loc[exp_mask, "recalibrated_reason"].value_counts().to_dict()
    all_r  = sorted(set(list(before.keys()) + list(after.keys())))
    print(f"{'reason':<28} {'before':>8} {'after':>8}")
    for r in all_r:
        print(f"  {r:<26} {before.get(r,0):>8} {after.get(r,0):>8}")
    print(f"  {'TOTAL':<26} {sum(before.values()):>8} {sum(after.values()):>8}")

    # Write markdown report
    _write_report(df_results, df_recal, exp_mask, before, after, k_values, out_dir)


def _write_report(
    df_results: "pd.DataFrame",
    df_recal: "pd.DataFrame",
    exp_mask: "pd.Series",
    reason_before: dict,
    reason_after: dict,
    k_values: list[int],
    out_dir: Path,
) -> None:
    lines = [
        "# Exploration Recalibration Eval — Report",
        "",
        "이번 수정은 short-term LLM이 recent 변화를 감지하는 능력 자체를 없애는 것이 아니라,",
        "그 변화를 너무 쉽게 exploration으로 해석하던 문제를 교정하여,",
        "대부분의 사례를 원래 취향(persona) 위에 최근 의도를 얹는",
        "aligned_soft / task_focus_like / budget_like 상태로 재해석하는 작업이다.",
        "",
        "---",
        "",
        "## 1. Recalibrated Reason Distribution (exploration rows only)",
        "",
        "| reason | before | after | delta |",
        "|---|---|---|---|",
    ]
    all_r = sorted(set(list(reason_before.keys()) + list(reason_after.keys())))
    for r in all_r:
        b = reason_before.get(r, 0)
        a = reason_after.get(r, 0)
        lines.append(f"| {r} | {b} | {a} | {a-b:+d} |")
    lines += [
        "",
        "## 2. HR@10 — Branch Comparison",
        "",
        "| branch | mode | HR@10 | NDCG@10 | MRR | gt_delta_pos | gt_delta_zero |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df_results.iterrows():
        lines.append(
            f"| {row.get('branch','')} | {row.get('mode','')} "
            f"| {row.get('HR@10','')} | {row.get('NDCG@10','')} "
            f"| {row.get('MRR','')} | {row.get('gt_delta_pos_frac','')} "
            f"| {row.get('gt_delta_zero_frac','')} |"
        )

    # aligned_soft / task_focus_like movement
    moved = df_recal[exp_mask & (df_recal["recalibrated_reason"] != "exploration")].copy()
    n_moved = len(moved)
    n_total_exp = exp_mask.sum()
    lines += [
        "",
        f"## 3. Exploration Redistribution",
        "",
        f"- Total exploration records: {n_total_exp}",
        f"- Recalibrated away from exploration: {n_moved} ({100*n_moved/max(n_total_exp,1):.1f}%)",
        "",
        "| recalibrated_reason | count | % of exploration |",
        "|---|---|---|",
    ]
    for r, cnt in df_recal.loc[exp_mask, "recalibrated_reason"].value_counts().items():
        lines.append(f"| {r} | {cnt} | {100*cnt/max(n_total_exp,1):.1f}% |")

    lines += [
        "",
        "## 4. Semantic-Core Overlap Distribution",
        "",
    ]
    sc_vals = df_recal.loc[exp_mask, "rc_sc_overlap"].dropna()
    if len(sc_vals) > 0:
        lines.append(f"- mean sc_overlap (exploration rows): {sc_vals.mean():.3f}")
        lines.append(f"- fraction with sc_overlap == 0.0: {(sc_vals == 0).mean():.3f}")
        lines.append(f"- fraction with sc_overlap >= 0.4: {(sc_vals >= 0.4).mean():.3f}")

    lines += [
        "",
        "## 5. Conclusion",
        "",
        "현재 PGIM의 문제는 exploration 신호 자체가 아니라,",
        "exploration으로 과대 해석되는 reason calibration 문제였는가?",
        "",
        "> **결론**: 위 비교 결과를 기반으로 판단 (B vs A HR@10 delta 참고)",
    ]

    report_path = out_dir / "recalibration_report.md"
    report_path.write_text("\n".join(lines))
    logger.info("Saved report -> %s", report_path)


if __name__ == "__main__":
    main()
