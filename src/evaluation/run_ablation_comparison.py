"""
run_ablation_comparison.py
--------------------------
Four-branch ablation for PGIM short-term signal validation.

Comparison matrix:
  B_raw   — LLM reason   + raw_llm_goals        (Stage 1 only, no grounding)
  B_valid — LLM reason   + validated_goal_concepts (Stage 1+2, grounded)
  D_raw   — Heuristic reason + heuristic goals   (strong control baseline)
  D_mix   — LLM reason   + heuristic goals       (reason-only effect isolation)

Modulation modes evaluated per branch:
  intent_only_rerank       — short-term signal effect in isolation
  graph_conditioned_full   — full PGIM (persona + short-term)

Diagnostics output (before/after per branch):
  - goal-to-candidate match rate
  - any_candidate_activated rate
  - avg activated candidates
  - GT match rate
  - activation mass
  - intent-only nonzero delta rate / mean delta
  - intent-only HR@K / NDCG@K / MRR
  - full_model HR@K / NDCG@K / MRR
  - per-reason breakdown

Usage:
  python -m src.evaluation.run_ablation_comparison \\
    --data-config    config/data/amazon_movies_tv.yaml \\
    --eval-config    config/evaluation/default.yaml \\
    --mod-config     config/modulation/amazon_movies_tv.yaml \\
    --llm-intent-path  data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet \\
    --heur-intent-path data/cache/intent/amazon_movies_tv/short_term_intents.parquet \\
    --max-users 2000 \\
    --out-dir results/ablation_comparison

Success criteria logged:
  1. B_valid intent-only nonzero delta > B_raw
  2. B_valid goal-to-candidate match   > B_raw
  3. B_valid full_model HR@10          >= B_raw (improvement not from persona only)
  4. B_valid vs D_mix isolates validated goal effect
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Optional

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
    """Safely convert parquet field (list / numpy array / JSON string / None) to list."""
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


def _load_intent(path: Path, max_users: Optional[int] = None) -> dict[tuple[str, int], dict]:
    """Load intent parquet → {(user_id, target_index): record_dict}."""
    df = pd.read_parquet(path)
    if max_users is not None:
        users = sorted(df["user_id"].unique())[:max_users]
        df = df[df["user_id"].isin(users)]
    result: dict[tuple[str, int], dict] = {}
    for r in df.to_dict("records"):
        result[(r["user_id"], int(r["target_index"]))] = r
    return result


def _load_intent_filtered(
    path: Path,
    shared_keys: set[tuple[str, int]],
) -> dict[tuple[str, int], dict]:
    df = pd.read_parquet(path)
    shared_users = {k[0] for k in shared_keys}
    df = df[df["user_id"].isin(shared_users)]
    result = {}
    for r in df.to_dict("records"):
        k = (r["user_id"], int(r["target_index"]))
        if k in shared_keys:
            result[k] = r
    return result


def _compute_metrics(
    ranked_rows: list[dict],
    is_gt_by_key: dict[tuple[str, int], set[str]],
    k_values: list[int],
) -> dict:
    """Compute HR@K, NDCG@K, MRR, rank movement, delta stats from ranked rows."""
    df = pd.DataFrame(ranked_rows)
    if df.empty:
        return {}

    df["is_gt"] = df.apply(
        lambda r: int(r["candidate_item_id"] in is_gt_by_key.get(
            (r["user_id"], int(r["target_index"])), set()
        )),
        axis=1,
    )
    df_s = df.sort_values(["user_id", "target_index", "final_score"],
                          ascending=[True, True, False])
    df_s["_rank"] = df_s.groupby(["user_id", "target_index"]).cumcount() + 1
    gt_df = df_s[df_s["is_gt"] == 1].copy()

    if gt_df.empty:
        return {}

    ranks = gt_df["_rank"].values.astype(int)
    n = len(ranks)
    metrics: dict = {"n_users": n}
    for k in k_values:
        metrics[f"HR@{k}"]   = round(float((ranks <= k).mean()), 4)
        metrics[f"NDCG@{k}"] = round(
            float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n), 4
        )
    metrics["MRR"] = round(float((1.0 / ranks).mean()), 4)

    if "modulation_delta" in gt_df.columns:
        deltas = gt_df["modulation_delta"].dropna()
        metrics["gt_delta_mean"]     = round(float(deltas.mean()), 6)
        metrics["gt_delta_pos_frac"] = round(float((deltas > 0).mean()), 4)
        metrics["gt_delta_neg_frac"] = round(float((deltas < 0).mean()), 4)
        metrics["gt_nonzero_frac"]   = round(float((deltas != 0).mean()), 4)

    return metrics


def _compute_metrics_per_reason(
    ranked_rows: list[dict],
    is_gt_by_key: dict[tuple[str, int], set[str]],
    intent_by_key: dict[tuple[str, int], dict],
    k_values: list[int],
) -> dict[str, dict]:
    """Compute metrics broken down by deviation_reason."""
    by_reason: dict[str, list[dict]] = {}
    for row in ranked_rows:
        key = (row["user_id"], int(row["target_index"]))
        reason = intent_by_key.get(key, {}).get("deviation_reason", "unknown")
        by_reason.setdefault(reason, []).append(row)

    return {
        reason: _compute_metrics(rows, is_gt_by_key, k_values)
        for reason, rows in by_reason.items()
    }


# ── Branch-specific intent record patching ───────────────────────────────────

def _make_branch_record(base_record: dict, branch: str) -> dict:
    """
    Return a copy of intent_record with goal_concepts set according to branch.

    B_raw   — goal_concepts = raw_llm_goals   (Stage 1 LLM output)
    B_valid — goal_concepts = validated_goal_concepts (Stage 2 grounded)
    D_raw   — goal_concepts = heuristic goals  (passed as separate dict)
    D_mix   — goal_concepts = heuristic goals, deviation_reason = LLM reason
              (goal field comes from heuristic record, reason from LLM record)

    For D_raw / D_mix the caller must pass the heuristic record as 'heur_record' in base_record.
    """
    rec = dict(base_record)

    if branch == "B_raw":
        # Stage 1: raw LLM goals — use goal_concepts as-is (already raw LLM output)
        # Also set validated_goal_concepts = raw goals so signal_builder uses them
        _g = rec.get("raw_llm_goals")
        raw = _to_list(_g if _g is not None else rec.get("goal_concepts"))
        rec["goal_concepts"] = raw
        rec["validated_goal_concepts"] = raw

    elif branch == "B_valid":
        # Stage 2: use validated_goal_concepts (grounded)
        _g = rec.get("validated_goal_concepts")
        val = _to_list(_g if _g is not None else rec.get("goal_concepts"))
        rec["goal_concepts"] = val
        rec["validated_goal_concepts"] = val

    elif branch == "D_raw":
        # Heuristic reason + heuristic goals
        heur = rec.pop("_heur_record", {})
        heur_goals = _to_list(heur.get("goal_concepts"))
        rec["goal_concepts"] = heur_goals
        rec["validated_goal_concepts"] = heur_goals
        rec["deviation_reason"] = heur.get("deviation_reason", rec.get("deviation_reason", "unknown"))
        rec["confidence"] = heur.get("confidence", rec.get("confidence", 0.5))

    elif branch == "D_mix":
        # LLM reason + heuristic goals (isolates reason effect from goal effect)
        heur = rec.pop("_heur_record", {})
        heur_goals = _to_list(heur.get("goal_concepts"))
        rec["goal_concepts"] = heur_goals
        rec["validated_goal_concepts"] = heur_goals
        # Keep LLM deviation_reason and confidence
    else:
        raise ValueError(f"Unknown branch: {branch}")

    return rec


# ── Grounding diagnostics ─────────────────────────────────────────────────────

def _compute_branch_grounding_diag(
    branch: str,
    intent_by_key: dict[tuple[str, int], dict],
    heur_by_key: dict[tuple[str, int], dict],
    cand_by_key: dict[tuple[str, int], list[str]],
    is_gt_by_key: dict[tuple[str, int], set[str]],
    item_concepts: dict[str, list[str]],
) -> dict:
    """
    Compute grounding diagnostics for a branch.

    GT is used ONLY here (eval metric), not in modulation.
    """
    from src.intent.grounded_selector import (
        build_candidate_concept_bank,
        compute_grounding_diagnostics,
    )

    rows: list[dict] = []
    for key, cand_ids in cand_by_key.items():
        llm_rec  = intent_by_key.get(key, {})
        heur_rec = heur_by_key.get(key, {})

        # Select goals for this branch
        if branch == "B_raw":
            _g = llm_rec.get("raw_llm_goals")
            goals = _to_list(_g if _g is not None else llm_rec.get("goal_concepts"))
        elif branch == "B_valid":
            _g = llm_rec.get("validated_goal_concepts")
            goals = _to_list(_g if _g is not None else llm_rec.get("goal_concepts"))
        elif branch in ("D_raw", "D_mix"):
            goals = _to_list(heur_rec.get("goal_concepts"))
        else:
            goals = []

        # Build bank from backbone candidates (NO GT — leakage guard)
        bank = build_candidate_concept_bank(cand_ids, item_concepts)

        # GT concepts for eval only
        gt_items = is_gt_by_key.get(key, set())
        gt_concepts: list[str] = []
        for iid in gt_items:
            gt_concepts.extend(item_concepts.get(iid, []))

        gd = compute_grounding_diagnostics(
            raw_goal_concepts=goals,
            validated_goal_concepts=goals,   # already the target goals; no before/after here
            candidate_concept_bank=bank,
            gt_item_concepts=gt_concepts,
        )
        reason = llm_rec.get("deviation_reason", "unknown") if branch != "D_raw" else heur_rec.get("deviation_reason", "unknown")
        gd["user_id"] = key[0]
        gd["target_index"] = key[1]
        gd["deviation_reason"] = reason
        gd["n_goals"] = len(goals)
        rows.append(gd)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    scalar_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], list)]
    df_s = df[scalar_cols]

    agg: dict = {
        "n_total":                    len(df_s),
        "candidate_match_rate":       round(df_s["candidate_match_before"].mean(), 4),
        "any_activated_rate":         round(df_s["any_activated_before"].mean(), 4),
        "avg_activated_candidates":   round(df_s["avg_activated_cands_before"].mean(), 4),
        "activation_mass_mean":       round(df_s["raw_activation_mass"].mean(), 4),
    }
    if "gt_match_before" in df_s.columns:
        agg["gt_match_rate"] = round(df_s["gt_match_before"].mean(), 4)

    # Per-reason breakdown
    by_reason: dict[str, dict] = {}
    for reason, grp in df_s.groupby("deviation_reason"):
        entry: dict = {
            "n":                    len(grp),
            "candidate_match_rate": round(grp["candidate_match_before"].mean(), 4),
            "any_activated_rate":   round(grp["any_activated_before"].mean(), 4),
            "avg_activated_cands":  round(grp["avg_activated_cands_before"].mean(), 4),
        }
        if "gt_match_before" in grp.columns:
            entry["gt_match_rate"] = round(grp["gt_match_before"].mean(), 4)
        by_reason[str(reason)] = entry
    agg["by_reason"] = by_reason

    return agg


# ── Core eval runner ──────────────────────────────────────────────────────────

def _run_branch_eval(
    branch: str,
    llm_by_key: dict[tuple[str, int], dict],
    heur_by_key: dict[tuple[str, int], dict],
    cand_by_key: dict[tuple[str, int], list[str]],
    is_gt_by_key: dict[tuple[str, int], set[str]],
    backbone_scores: dict[tuple[str, int], dict[str, float]],
    persona_nodes_by_user: dict[str, list[dict]],
    item_concepts: dict[str, list[str]],
    modulation_cfg: dict,
    k_values: list[int],
    modes: dict[str, str],
) -> dict[str, dict]:
    """
    Run eval for one branch across all modulation modes.
    Returns {mode_name: metrics_dict}.
    """
    from src.modulation.gate import compute_gate_strength
    from src.modulation.reranker import CandidateReranker
    from src.modulation.signal_builder import build_signal

    reranker = CandidateReranker(modulation_cfg, item_concepts)
    results: dict[str, dict] = {}

    for mode_name, mode in modes.items():
        all_ranked: list[dict] = []

        for (uid, tidx), candidate_ids in cand_by_key.items():
            scores = backbone_scores.get((uid, tidx), {})
            if not scores:
                continue

            candidate_tuples = sorted(
                [(iid, scores[iid]) for iid in candidate_ids if iid in scores],
                key=lambda x: x[1], reverse=True,
            )
            if not candidate_tuples:
                continue

            llm_rec  = dict(llm_by_key.get((uid, tidx), {}))
            heur_rec = dict(heur_by_key.get((uid, tidx), {}))

            # Inject heuristic record for D branches
            if branch in ("D_raw", "D_mix"):
                llm_rec["_heur_record"] = heur_rec

            intent_record = _make_branch_record(llm_rec, branch)

            reason     = str(intent_record.get("deviation_reason", "unknown"))
            confidence = float(intent_record.get("confidence", 0.5))
            persona_nodes = persona_nodes_by_user.get(uid, [])

            gate_strength = compute_gate_strength(
                deviation_reason=reason,
                confidence=confidence,
                persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_record, persona_nodes, gate_strength,
                                  modulation_cfg, mode=mode)
            ranked = reranker.rerank(candidate_tuples, signal, mode=mode)

            for r in ranked:
                rec = r.to_record()
                all_ranked.append(rec)

        metrics = _compute_metrics(all_ranked, is_gt_by_key, k_values)
        metrics["branch"] = branch
        metrics["mode"]   = mode_name

        # Per-reason metrics
        intent_map = llm_by_key if branch not in ("D_raw",) else heur_by_key
        per_reason = _compute_metrics_per_reason(
            all_ranked, is_gt_by_key, intent_map, k_values
        )
        metrics["by_reason"] = per_reason

        results[mode_name] = metrics
        logger.info(
            "[%s / %s]  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  gt_delta+frac=%s",
            branch, mode_name,
            metrics.get("HR@10", 0), metrics.get("NDCG@10", 0), metrics.get("MRR", 0),
            metrics.get("gt_delta_pos_frac", "N/A"),
        )

    return results


# ── Report writer ─────────────────────────────────────────────────────────────

def _write_report(
    branch_results: dict[str, dict[str, dict]],
    grounding_diags: dict[str, dict],
    out_path: Path,
    k_values: list[int],
) -> None:
    """Write markdown comparison report."""
    lines: list[str] = [
        "# PGIM Short-term Branch Ablation Comparison\n",
        "이번 수정은 reason source를 LLM으로 유지한 채, "
        "raw short-term goal signal의 candidate grounding failure를 해결하기 위해 "
        "grounded goal selection/validation layer를 추가한 것이다. "
        "즉 PGIM의 방향성(장기 ontology persona + 단기 LLM interpretation)은 유지하되, "
        "LLM output을 candidate-effective한 modulation signal로 변환하는 위치로 재배치한다.\n",
        "---\n",
        "## Branch Definitions\n",
        "| Branch  | Reason Source | Goal Source                          |",
        "|---------|---------------|--------------------------------------|",
        "| B_raw   | LLM           | raw_llm_goals (Stage 1, no grounding)|",
        "| B_valid | LLM           | validated_goal_concepts (Stage 2)    |",
        "| D_raw   | Heuristic     | heuristic goals (control baseline)   |",
        "| D_mix   | LLM           | heuristic goals (reason-only effect) |",
        "",
    ]

    # ── Grounding diagnostics ─────────────────────────────────────────
    lines.append("## 1. Grounding Diagnostics\n")
    lines.append("| Branch | cand_match_rate | any_activated_rate | avg_activated_cands | gt_match_rate |")
    lines.append("|--------|-----------------|--------------------|---------------------|---------------|")
    for branch in ["B_raw", "B_valid", "D_raw", "D_mix"]:
        gd = grounding_diags.get(branch, {})
        lines.append(
            f"| {branch} "
            f"| {gd.get('candidate_match_rate', 'N/A')} "
            f"| {gd.get('any_activated_rate', 'N/A')} "
            f"| {gd.get('avg_activated_candidates', 'N/A')} "
            f"| {gd.get('gt_match_rate', 'N/A')} |"
        )
    lines.append("")

    # ── intent_only metrics ────────────────────────────────────────────
    lines.append("## 2. intent_only_rerank Metrics\n")
    lines.append("| Branch | HR@10 | NDCG@10 | MRR | gt_delta+frac | gt_nonzero_frac |")
    lines.append("|--------|-------|---------|-----|---------------|-----------------|")
    for branch in ["B_raw", "B_valid", "D_raw", "D_mix"]:
        m = branch_results.get(branch, {}).get("ablation_intent_only", {})
        lines.append(
            f"| {branch} "
            f"| {m.get('HR@10', 'N/A')} "
            f"| {m.get('NDCG@10', 'N/A')} "
            f"| {m.get('MRR', 'N/A')} "
            f"| {m.get('gt_delta_pos_frac', 'N/A')} "
            f"| {m.get('gt_nonzero_frac', 'N/A')} |"
        )
    lines.append("")

    # ── full_model metrics ─────────────────────────────────────────────
    lines.append("## 3. graph_conditioned_full Metrics\n")
    lines.append("| Branch | HR@10 | NDCG@10 | MRR |")
    lines.append("|--------|-------|---------|-----|")
    for branch in ["B_raw", "B_valid", "D_raw", "D_mix"]:
        m = branch_results.get(branch, {}).get("full_model", {})
        lines.append(
            f"| {branch} "
            f"| {m.get('HR@10', 'N/A')} "
            f"| {m.get('NDCG@10', 'N/A')} "
            f"| {m.get('MRR', 'N/A')} |"
        )
    lines.append("")

    # ── per-reason breakdown (intent_only) ────────────────────────────
    lines.append("## 4. Per-Reason Breakdown (intent_only)\n")
    reasons = ["aligned", "exploration", "task_focus", "budget_shift", "unknown"]
    for reason in reasons:
        lines.append(f"### {reason}\n")
        lines.append("| Branch | n | HR@10 | NDCG@10 | MRR | gt_delta+frac |")
        lines.append("|--------|---|-------|---------|-----|---------------|")
        for branch in ["B_raw", "B_valid", "D_raw", "D_mix"]:
            rm = branch_results.get(branch, {}).get("ablation_intent_only", {}).get("by_reason", {}).get(reason, {})
            lines.append(
                f"| {branch} "
                f"| {rm.get('n_users', 'N/A')} "
                f"| {rm.get('HR@10', 'N/A')} "
                f"| {rm.get('NDCG@10', 'N/A')} "
                f"| {rm.get('MRR', 'N/A')} "
                f"| {rm.get('gt_delta_pos_frac', 'N/A')} |"
            )
        lines.append("")

    # ── Success criteria assessment ────────────────────────────────────
    lines.append("## 5. Success Criteria Assessment\n")

    b_raw_io  = branch_results.get("B_raw",   {}).get("ablation_intent_only", {})
    b_val_io  = branch_results.get("B_valid", {}).get("ablation_intent_only", {})
    b_raw_fm  = branch_results.get("B_raw",   {}).get("full_model", {})
    b_val_fm  = branch_results.get("B_valid", {}).get("full_model", {})

    raw_nz  = b_raw_io.get("gt_nonzero_frac", 0) or 0
    val_nz  = b_val_io.get("gt_nonzero_frac", 0) or 0
    raw_hr  = b_raw_io.get("HR@10", 0) or 0
    val_hr  = b_val_io.get("HR@10", 0) or 0
    raw_gd  = grounding_diags.get("B_raw",   {}).get("candidate_match_rate", 0) or 0
    val_gd  = grounding_diags.get("B_valid", {}).get("candidate_match_rate", 0) or 0
    raw_fm  = b_raw_fm.get("HR@10", 0) or 0
    val_fm  = b_val_fm.get("HR@10", 0) or 0

    def _check(passed: bool, label: str) -> str:
        return f"{'✓' if passed else '✗'} {label}"

    lines += [
        _check(val_nz > raw_nz,
               f"B_valid intent-only nonzero delta > B_raw  "
               f"({val_nz:.4f} vs {raw_nz:.4f})"),
        _check(val_gd >= raw_gd,
               f"B_valid goal-to-candidate match >= B_raw  "
               f"({val_gd:.4f} vs {raw_gd:.4f})"),
        _check(val_hr >= raw_hr,
               f"B_valid intent-only HR@10 >= B_raw  "
               f"({val_hr:.4f} vs {raw_hr:.4f})"),
        _check(val_fm >= raw_fm,
               f"B_valid full_model HR@10 >= B_raw  "
               f"({val_fm:.4f} vs {raw_fm:.4f})"),
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Report saved -> %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",       default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--eval-config",       default="config/evaluation/default.yaml")
    parser.add_argument("--mod-config",        default="config/modulation/amazon_movies_tv.yaml")
    parser.add_argument("--llm-intent-path",   default=None,
                        help="LLM intent cache (with validated_goal_concepts). "
                             "Default: short_term_intents_llm_subset_2000_validated.parquet")
    parser.add_argument("--heur-intent-path",  default=None,
                        help="Heuristic intent cache. Default: short_term_intents.parquet")
    parser.add_argument("--cand-path",         default=None,
                        help="Backbone candidate parquet. "
                             "Default: sampled_candidates_k101.parquet")
    parser.add_argument("--backbone-scores",   default=None,
                        help="Backbone scores parquet. Default: backbone_scores.parquet")
    parser.add_argument("--max-users",         type=int, default=2000)
    parser.add_argument("--out-dir",           default="results/ablation_comparison")
    args = parser.parse_args()

    data_cfg  = _load_yaml(args.data_config)
    eval_cfg  = _load_yaml(args.eval_config)
    mod_cfg   = _load_yaml(args.mod_config)

    dataset      = data_cfg.get("dataset", "amazon_movies_tv")
    k_values     = eval_cfg.get("k_values", [5, 10, 20])
    interim_dir  = Path(data_cfg["paths"]["interim_dir"])
    processed_dir= Path(data_cfg["paths"]["processed_dir"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── paths ─────────────────────────────────────────────────────────
    llm_path  = Path(args.llm_intent_path  or
                     f"data/cache/intent/{dataset}/short_term_intents_llm_subset_2000_validated.parquet")
    heur_path = Path(args.heur_intent_path or
                     f"data/cache/intent/{dataset}/short_term_intents.parquet")
    cand_path = Path(args.cand_path or
                     f"data/cache/candidate/{dataset}/sampled_candidates_k101.parquet")
    bs_path   = Path(args.backbone_scores  or
                     f"data/cache/backbone/{dataset}/backbone_scores.parquet")

    for p, name in [(llm_path, "LLM intent"), (cand_path, "candidates"), (bs_path, "backbone scores")]:
        if not p.exists():
            logger.error("%s not found: %s", name, p)
            return

    # ── shared keys (LLM ∩ candidates) ───────────────────────────────
    logger.info("Computing shared keys...")
    df_llm_ids  = pd.read_parquet(llm_path,  columns=["user_id", "target_index"])
    df_cand_ids = pd.read_parquet(cand_path, columns=["user_id", "target_index"])
    l_keys = set(zip(df_llm_ids["user_id"], df_llm_ids["target_index"].astype(int)))
    c_keys = set(zip(df_cand_ids["user_id"], df_cand_ids["target_index"].astype(int)))

    llm_users   = sorted({k[0] for k in l_keys})[:args.max_users]
    llm_user_set= set(llm_users)
    shared_keys = {k for k in (l_keys & c_keys) if k[0] in llm_user_set}
    if not shared_keys:
        logger.error("No shared keys — check intent cache and candidate files.")
        return
    shared_users = {k[0] for k in shared_keys}
    logger.info("shared_keys: %d  users: %d", len(shared_keys), len(shared_users))

    # ── load intents ──────────────────────────────────────────────────
    logger.info("Loading LLM intent cache...")
    llm_by_key = _load_intent_filtered(llm_path, shared_keys)
    logger.info("LLM intent: %d records", len(llm_by_key))

    heur_by_key: dict[tuple[str, int], dict] = {}
    if heur_path.exists():
        logger.info("Loading heuristic intent cache...")
        heur_by_key = _load_intent_filtered(heur_path, shared_keys)
        logger.info("Heuristic intent: %d records", len(heur_by_key))
    else:
        logger.warning("Heuristic intent not found: %s — D_raw / D_mix branches skipped", heur_path)

    # ── load shared data ──────────────────────────────────────────────
    logger.info("Loading item_concepts, persona, candidates...")
    df_ic = pd.read_parquet(processed_dir / "item_concepts.parquet")
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )

    df_persona = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")
    df_persona = df_persona[df_persona["user_id"].isin(shared_users)]
    persona_nodes_by_user: dict[str, list[dict]] = {
        uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
    }

    logger.info("Loading candidates...")
    df_cands = pd.read_parquet(cand_path)
    df_cands = df_cands[df_cands["user_id"].isin(shared_users)]
    key_series = list(zip(df_cands["user_id"], df_cands["target_index"].astype(int)))
    df_cands = df_cands[[k in shared_keys for k in key_series]].reset_index(drop=True)

    cand_by_key:  dict[tuple[str, int], list[str]]  = {}
    is_gt_by_key: dict[tuple[str, int], set[str]]   = {}
    for row in df_cands.itertuples(index=False):
        key = (row.user_id, int(row.target_index))
        cand_by_key.setdefault(key, []).append(row.candidate_item_id)
        if row.is_ground_truth:
            is_gt_by_key.setdefault(key, set()).add(row.candidate_item_id)
    logger.info("Candidate index: %d keys", len(cand_by_key))

    logger.info("Loading backbone scores...")
    df_bs = pd.read_parquet(bs_path)
    df_bs = df_bs[df_bs["user_id"].isin(shared_users)]
    backbone_scores: dict[tuple[str, int], dict[str, float]] = {}
    for row in df_bs.itertuples(index=False):
        key = (row.user_id, int(row.target_index))
        if key not in shared_keys:
            continue
        backbone_scores.setdefault(key, {})[row.candidate_item_id] = float(row.backbone_score)
    del df_bs
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    modes = {
        "ablation_intent_only": "intent_only_rerank",
        "full_model":           "graph_conditioned_full",
    }

    # ── decide which branches to run ─────────────────────────────────
    all_branches = ["B_raw", "B_valid"]
    if heur_by_key:
        all_branches += ["D_raw", "D_mix"]
    else:
        logger.info("Skipping D_raw / D_mix (no heuristic intent cache)")

    # ── grounding diagnostics ─────────────────────────────────────────
    logger.info("Computing grounding diagnostics...")
    grounding_diags: dict[str, dict] = {}
    for branch in all_branches:
        logger.info("  grounding diag: %s", branch)
        grounding_diags[branch] = _compute_branch_grounding_diag(
            branch=branch,
            intent_by_key=llm_by_key,
            heur_by_key=heur_by_key,
            cand_by_key=cand_by_key,
            is_gt_by_key=is_gt_by_key,
            item_concepts=item_concepts,
        )

    # ── eval per branch ───────────────────────────────────────────────
    branch_results: dict[str, dict[str, dict]] = {}
    for branch in all_branches:
        logger.info("=" * 60)
        logger.info("Branch: %s", branch)
        logger.info("=" * 60)
        branch_results[branch] = _run_branch_eval(
            branch=branch,
            llm_by_key=llm_by_key,
            heur_by_key=heur_by_key,
            cand_by_key=cand_by_key,
            is_gt_by_key=is_gt_by_key,
            backbone_scores=backbone_scores,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            modulation_cfg=mod_cfg,
            k_values=k_values,
            modes=modes,
        )

    # ── save outputs ──────────────────────────────────────────────────
    # Flat CSV
    rows: list[dict] = []
    for branch, mode_results in branch_results.items():
        for mode_name, metrics in mode_results.items():
            row = {"branch": branch, "mode": mode_name}
            row.update({k: v for k, v in metrics.items() if k not in ("by_reason", "branch", "mode")})
            rows.append(row)
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "ablation_comparison.csv"
    df_out.to_csv(csv_path, index=False)
    logger.info("saved -> %s", csv_path)

    # Full JSON
    json_path = out_dir / "ablation_comparison_full.json"
    with open(json_path, "w") as f:
        json.dump({
            "branch_results":  branch_results,
            "grounding_diags": grounding_diags,
        }, f, indent=2, default=str)
    logger.info("saved -> %s", json_path)

    # Grounding diags CSV
    gd_rows = []
    for branch, gd in grounding_diags.items():
        flat = {"branch": branch}
        for k, v in gd.items():
            if k != "by_reason":
                flat[k] = v
        gd_rows.append(flat)
    pd.DataFrame(gd_rows).to_csv(out_dir / "grounding_diagnostics_summary.csv", index=False)
    logger.info("saved -> %s", out_dir / "grounding_diagnostics_summary.csv")

    # Markdown report
    _write_report(branch_results, grounding_diags, out_dir / "ablation_comparison_report.md", k_values)

    # ── console summary ───────────────────────────────────────────────
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("ABLATION COMPARISON — intent_only_rerank")
    print(SEP)
    print(f"{'Branch':<10} {'HR@10':>7} {'NDCG@10':>9} {'MRR':>7} "
          f"{'gt_nz_frac':>11} {'cand_match':>11} {'gt_match':>9}")
    print("-" * 80)
    for branch in all_branches:
        m  = branch_results.get(branch, {}).get("ablation_intent_only", {})
        gd = grounding_diags.get(branch, {})
        print(
            f"{branch:<10} "
            f"{m.get('HR@10', 'N/A'):>7} "
            f"{m.get('NDCG@10', 'N/A'):>9} "
            f"{m.get('MRR', 'N/A'):>7} "
            f"{m.get('gt_nonzero_frac', 'N/A'):>11} "
            f"{gd.get('candidate_match_rate', 'N/A'):>11} "
            f"{gd.get('gt_match_rate', 'N/A'):>9}"
        )

    print(f"\n{SEP}")
    print("ABLATION COMPARISON — graph_conditioned_full")
    print(SEP)
    print(f"{'Branch':<10} {'HR@10':>7} {'NDCG@10':>9} {'MRR':>7}")
    print("-" * 40)
    for branch in all_branches:
        m = branch_results.get(branch, {}).get("full_model", {})
        print(f"{branch:<10} {m.get('HR@10','N/A'):>7} {m.get('NDCG@10','N/A'):>9} {m.get('MRR','N/A'):>7}")

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
