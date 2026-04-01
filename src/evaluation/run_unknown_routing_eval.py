"""
run_unknown_routing_eval.py
---------------------------
Ablation evaluation for v5 unknown soft routing.

Branches:
  A. v5_baseline          — v5 cache as-is (no routing)
  B. v5_unknown_routed    — unknown soft routing applied
  C. heuristic_control    — heuristic intent baseline
  D. v3_baseline          — v3 cache (reference comparison, optional)

Outputs:
  - unknown_audit.csv         Task 1: per-record unknown subtype audit
  - unknown_subtype_summary.csv
  - unknown_routing_eval_results.csv
  - unknown_routing_report.md

Usage:
  python -m src.evaluation.run_unknown_routing_eval \\
    --data-config   config/data/amazon_movies_tv.yaml \\
    --eval-config   config/evaluation/default.yaml \\
    --mod-config    config/modulation/amazon_movies_tv.yaml \\
    --v5-intent-path  data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_v5_2000.parquet \\
    --heur-intent-path data/cache/intent/amazon_movies_tv/short_term_intents_heuristic.parquet \\
    --backbone-candidates-path data/cache/candidate/amazon_movies_tv/sampled_candidates_k101.parquet \\
    --out-dir results/unknown_routing_eval \\
    [--v3-intent-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _to_list(x) -> list:
    if x is None:
        return []
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return list(x)
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


def _compute_metrics(ranks: np.ndarray, deltas, k_values: list[int]) -> dict:
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
        m["gt_delta_mean"]      = round(float(np.array(deltas).mean()), 6)
        m["gt_delta_pos_frac"]  = round(float((np.array(deltas) > 0).mean()), 4)
        m["gt_delta_zero_frac"] = round(float((np.array(deltas) == 0).mean()), 4)
        m["gt_delta_neg_frac"]  = round(float((np.array(deltas) < 0).mean()), 4)
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

            # routed_reason takes priority in signal_builder (already patched)
            eff_reason = (
                intent_rec.get("routed_reason")
                or intent_rec.get("recalibrated_reason")
                or intent_rec.get("deviation_reason", "unknown")
            )
            confidence = float(intent_rec.get("confidence", 0.5))
            alignment  = float(intent_rec.get("persona_alignment_score", 0.0))

            persona_nodes = persona_nodes_by_user.get(uid, [])
            gate_strength = compute_gate_strength(
                deviation_reason=eff_reason,
                confidence=confidence,
                persona_alignment_score=alignment,
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_rec, persona_nodes, gate_strength, modulation_cfg, mode=mode)
            ranked = reranker.rerank(candidate_tuples, signal, mode=mode)

            gt_items = gt_items_by_key.get((uid, tidx), set())
            for r in ranked:
                rec = r.to_record()
                rec["is_gt"]            = int(rec["candidate_item_id"] in gt_items)
                rec["deviation_reason"] = intent_rec.get("deviation_reason", "unknown")
                rec["routed_reason"]    = intent_rec.get("routed_reason", eff_reason)
                rec["unknown_subtype"]  = intent_rec.get("unknown_subtype", "")
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
        metrics["branch"] = branch_name
        metrics["mode"]   = mode_name

        # per-routed_reason breakdown
        for rsn, rg in gt_df.groupby("routed_reason"):
            rranks  = rg["_rank"].values.astype(int)
            rdeltas = rg["modulation_delta"].values if "modulation_delta" in rg.columns else None
            metrics[f"reason_{rsn}"] = _compute_metrics(rranks, rdeltas, k_values)

        # per-unknown_subtype breakdown (routing branch only)
        if "unknown_subtype" in gt_df.columns:
            for stype, sg in gt_df.groupby("unknown_subtype"):
                if not stype:
                    continue
                stranks  = sg["_rank"].values.astype(int)
                sdeltas  = sg["modulation_delta"].values if "modulation_delta" in sg.columns else None
                metrics[f"subtype_{stype}"] = _compute_metrics(stranks, sdeltas, k_values)

        results_by_mode[mode_name] = metrics
        logger.info(
            "[%s/%s] HR@10=%.4f NDCG@10=%.4f MRR=%.4f gt_delta_zero=%.3f",
            branch_name, mode_name,
            metrics.get("HR@10", 0), metrics.get("NDCG@10", 0),
            metrics.get("MRR", 0), metrics.get("gt_delta_zero_frac", 0),
        )

    return results_by_mode


def _write_report(
    out_dir: Path,
    df_audit: pd.DataFrame,
    df_subtype_summary: pd.DataFrame,
    all_results: list[dict],
    k_values: list[int],
) -> None:
    lines = [
        "# Unknown Soft Routing Eval — Report",
        "",
        "이번 수정은 unknown 전체를 억지로 해석하는 것이 아니라,",
        "v5에서 보수적으로 처리된 unknown 중 semantic evidence가 남아 있는 soft states만 persona 기반 recent-intent 경로로 재라우팅하여,",
        "exploration over-correction 이후 숨은 유효 신호를 복구할 수 있는지 검증하는 작업이다.",
        "",
        "---",
        "",
        "## 1. Unknown Subtype Distribution",
        "",
        df_subtype_summary.to_markdown(index=False),
        "",
        "## 2. HR@10 — Branch Comparison (full_model)",
        "",
    ]

    # metrics table
    rows = []
    for res in all_results:
        if res.get("mode") != "full_model":
            continue
        rows.append({
            "branch":       res["branch"],
            "HR@10":        res.get("HR@10", ""),
            "NDCG@10":      res.get("NDCG@10", ""),
            "MRR":          res.get("MRR", ""),
            "gt_delta_pos": res.get("gt_delta_pos_frac", ""),
            "gt_delta_zero":res.get("gt_delta_zero_frac", ""),
        })
    if rows:
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")

    lines += [
        "## 3. intent_only",
        "",
    ]
    rows2 = []
    for res in all_results:
        if res.get("mode") != "intent_only":
            continue
        rows2.append({
            "branch":       res["branch"],
            "HR@10":        res.get("HR@10", ""),
            "NDCG@10":      res.get("NDCG@10", ""),
            "MRR":          res.get("MRR", ""),
            "gt_delta_pos": res.get("gt_delta_pos_frac", ""),
            "gt_delta_zero":res.get("gt_delta_zero_frac", ""),
        })
    if rows2:
        lines.append(pd.DataFrame(rows2).to_markdown(index=False))
        lines.append("")

    lines += [
        "## 4. Unknown Subtype × HR@10 (routed branch, full_model)",
        "",
    ]
    for res in all_results:
        if res.get("branch") != "B_v5_unknown_routed" or res.get("mode") != "full_model":
            continue
        subtype_rows = []
        for k, v in res.items():
            if k.startswith("subtype_") and isinstance(v, dict):
                stype = k.replace("subtype_", "")
                subtype_rows.append({
                    "subtype": stype,
                    "n":       v.get("n", ""),
                    "HR@10":   v.get("HR@10", ""),
                    "NDCG@10": v.get("NDCG@10", ""),
                    "MRR":     v.get("MRR", ""),
                    "gt_delta_zero": v.get("gt_delta_zero_frac", ""),
                })
        if subtype_rows:
            lines.append(pd.DataFrame(subtype_rows).to_markdown(index=False))
            lines.append("")

    (out_dir / "unknown_routing_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved report -> %s", out_dir / "unknown_routing_report.md")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config",   required=True)
    ap.add_argument("--eval-config",   required=True)
    ap.add_argument("--mod-config",    required=True)
    ap.add_argument("--v5-intent-path",   required=True)
    ap.add_argument("--heur-intent-path", default=None)
    ap.add_argument("--v3-intent-path",   default=None, help="Optional v3 reference baseline")
    ap.add_argument("--backbone-candidates-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-users", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(args.data_config)
    eval_cfg  = _load_yaml(args.eval_config)
    mod_cfg   = _load_yaml(args.mod_config)

    k_values = eval_cfg.get("k_values", [5, 10, 20])
    dataset  = data_cfg.get("dataset", "amazon_movies_tv")

    experiment_modes: dict[str, str] = eval_cfg.get("experiment_modes", {
        "full_model":   "graph_conditioned_full",
        "intent_only":  "intent_only_rerank",
        "persona_only": "persona_only_rerank",
    })

    processed_dir = Path(data_cfg["paths"]["processed_dir"])

    # ── Load data (same pattern as run_recalibration_eval.py) ─────────────────
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

    logger.info("Loading interactions...")
    df_inter = pd.read_parquet(processed_dir / "interactions.parquet")

    logger.info("Loading backbone candidates...")
    df_cands = pd.read_parquet(args.backbone_candidates_path)

    logger.info("Loading v5 intent...")
    df_v5_raw = pd.read_parquet(args.v5_intent_path)
    if args.max_users:
        users = sorted(df_v5_raw["user_id"].unique())[: args.max_users]
        df_v5_raw = df_v5_raw[df_v5_raw["user_id"].isin(users)].reset_index(drop=True)
    logger.info("v5 intent: %d records", len(df_v5_raw))

    # ── Build candidate/score lookups ──────────────────────────────────────────
    v5_keys: set[tuple[str, int]] = set(
        zip(df_v5_raw["user_id"], df_v5_raw["target_index"].astype(int))
    )

    cand_by_key: dict[tuple[str, int], list[str]] = {}
    backbone_scores: dict[tuple[str, int], dict[str, float]] = {}

    if "candidate_item_ids" in df_cands.columns:
        for row in df_cands.itertuples(index=False):
            k = (str(row.user_id), int(row.target_index))
            if k not in v5_keys:
                continue
            cand_by_key[k] = _to_list(row.candidate_item_ids)
    else:
        item_col = next((c for c in ("item_id", "candidate_item_id") if c in df_cands.columns), None)
        if item_col is None:
            raise ValueError(f"Cannot find item column in candidates: {list(df_cands.columns)}")
        for (uid, tidx), grp in df_cands.groupby(["user_id", "target_index"]):
            k = (str(uid), int(tidx))
            if k not in v5_keys:
                continue
            cand_by_key[k] = grp[item_col].tolist()

    backbone_path = f"data/cache/backbone/{dataset}/backbone_scores.parquet"
    if Path(backbone_path).exists():
        df_bb = pd.read_parquet(backbone_path)
        for (uid, tidx), grp in df_bb.groupby(["user_id", "target_index"]):
            k = (str(uid), int(tidx))
            if k in cand_by_key:
                backbone_scores[k] = dict(zip(grp["candidate_item_id"], grp["backbone_score"]))
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    # ── GT lookup ──────────────────────────────────────────────────────────────
    inter_sorted = df_inter.sort_values(["user_id", "timestamp"])
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

    # ── Filter v5 intent to shared keys ───────────────────────────────────────
    df_v5 = df_v5_raw[df_v5_raw.apply(
        lambda r: (r["user_id"], int(r["target_index"])) in shared_keys, axis=1
    )].reset_index(drop=True)
    logger.info("v5 intent after shared key filter: %d records", len(df_v5))

    # ── Task 1: Unknown Audit ──────────────────────────────────────────────────
    logger.info("Running unknown subtype audit (Task 1)...")
    from src.intent.unknown_router import route_dataframe

    df_v5_routed = route_dataframe(df_v5)

    # Audit CSV: unknown rows only
    unk_mask = df_v5_routed["deviation_reason"] == "unknown"
    audit_cols = [
        "user_id", "target_index", "confidence",
        "raw_llm_goals", "validated_goal_concepts",
        "llm_explanation_short", "why_not_aligned",
        "evidence_recent_concepts", "evidence_persona_concepts",
        "unknown_subtype", "routing_trace",
        "rc_n_sem_evidence", "rc_n_val_goals", "rc_has_budget",
        "semantic_signal_absent", "non_semantic_goal_leakage",
        "constraints_json",
    ]
    avail_cols = [c for c in audit_cols if c in df_v5_routed.columns]
    df_audit = df_v5_routed[unk_mask][avail_cols].copy()
    df_audit.to_csv(out_dir / "unknown_audit.csv", index=False)
    logger.info("Saved unknown audit -> %s (%d rows)", out_dir / "unknown_audit.csv", len(df_audit))

    # Subtype summary
    subtype_counts = df_v5_routed[unk_mask]["unknown_subtype"].value_counts()
    total_unk = unk_mask.sum()
    total_all = len(df_v5_routed)
    subtype_rows = []
    for stype, cnt in subtype_counts.items():
        subtype_rows.append({
            "unknown_subtype":     stype,
            "count":               cnt,
            "pct_of_unknown":      round(cnt / total_unk * 100, 1),
            "pct_of_all":          round(cnt / total_all * 100, 1),
            "routed_to":           df_v5_routed[
                                       (unk_mask) & (df_v5_routed["unknown_subtype"] == stype)
                                   ]["routed_reason"].iloc[0] if cnt > 0 else "",
        })
    df_subtype_summary = pd.DataFrame(subtype_rows)
    df_subtype_summary.to_csv(out_dir / "unknown_subtype_summary.csv", index=False)
    logger.info("Unknown subtype summary:\n%s", df_subtype_summary.to_markdown(index=False))

    # ── Build intent dicts ─────────────────────────────────────────────────────
    # Branch A: v5 as-is (no routing fields)
    v5_baseline_by_key: dict[tuple[str, int], dict] = {}
    for r in df_v5.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            v5_baseline_by_key[k] = r

    # Branch B: v5 with routing applied
    v5_routed_by_key: dict[tuple[str, int], dict] = {}
    for r in df_v5_routed.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            v5_routed_by_key[k] = r

    # ── Task 3: Ablation ───────────────────────────────────────────────────────
    all_results: list[dict] = []

    logger.info("=== Branch A: v5_baseline ===")
    res_a = _run_eval_branch(
        "A_v5_baseline", v5_baseline_by_key, cand_by_key, backbone_scores,
        persona_nodes_by_user, item_concepts, gt_items_by_key, mod_cfg, k_values, experiment_modes,
    )
    for mode, metrics in res_a.items():
        all_results.append(metrics)

    logger.info("=== Branch B: v5_unknown_routed ===")
    res_b = _run_eval_branch(
        "B_v5_unknown_routed", v5_routed_by_key, cand_by_key, backbone_scores,
        persona_nodes_by_user, item_concepts, gt_items_by_key, mod_cfg, k_values, experiment_modes,
    )
    for mode, metrics in res_b.items():
        all_results.append(metrics)

    if args.heur_intent_path:
        logger.info("=== Branch C: heuristic_control ===")
        heur_keys = _intent_keys(Path(args.heur_intent_path))
        heur_shared = shared_keys & heur_keys
        heur_by_key = _load_intent_filtered(Path(args.heur_intent_path), heur_shared)
        res_c = _run_eval_branch(
            "C_heuristic_control", heur_by_key, cand_by_key, backbone_scores,
            persona_nodes_by_user, item_concepts, gt_items_by_key, mod_cfg, k_values, experiment_modes,
        )
        for mode, metrics in res_c.items():
            all_results.append(metrics)

    if args.v3_intent_path:
        logger.info("=== Branch D: v3_baseline (reference) ===")
        v3_keys = _intent_keys(Path(args.v3_intent_path))
        v3_shared = shared_keys & v3_keys
        v3_by_key = _load_intent_filtered(Path(args.v3_intent_path), v3_shared)
        res_d = _run_eval_branch(
            "D_v3_baseline", v3_by_key, cand_by_key, backbone_scores,
            persona_nodes_by_user, item_concepts, gt_items_by_key, mod_cfg, k_values, experiment_modes,
        )
        for mode, metrics in res_d.items():
            all_results.append(metrics)

    # ── Save results ───────────────────────────────────────────────────────────
    flat_rows = []
    for res in all_results:
        row = {k: v for k, v in res.items() if not isinstance(v, dict)}
        flat_rows.append(row)
    df_results = pd.DataFrame(flat_rows)
    df_results.to_csv(out_dir / "unknown_routing_eval_results.csv", index=False)
    logger.info("Saved results -> %s", out_dir / "unknown_routing_eval_results.csv")

    # Save routed intent
    df_v5_routed.to_parquet(out_dir / "intent_v5_routed.parquet", index=False)
    logger.info("Saved routed intent -> %s", out_dir / "intent_v5_routed.parquet")

    # ── Report ─────────────────────────────────────────────────────────────────
    _write_report(out_dir, df_audit, df_subtype_summary, all_results, k_values)

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("UNKNOWN ROUTING EVAL — SUMMARY")
    print("=" * 70)
    for mode in ["full_model", "intent_only"]:
        print(f"\n--- Mode: {mode} ---")
        mode_rows = [r for r in all_results if r.get("mode") == mode and "HR@10" in r]
        if mode_rows:
            df_m = pd.DataFrame(mode_rows)[
                ["branch", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR",
                 "gt_delta_pos_frac", "gt_delta_zero_frac"]
            ]
            print(df_m.to_markdown(index=False))

    print("\n--- Unknown subtype distribution ---")
    print(df_subtype_summary.to_markdown(index=False))


if __name__ == "__main__":
    main()
