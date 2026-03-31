"""
run_exploration_ablation.py
---------------------------
Focused diagnostic + ablation study on exploration-reason modulation behavior.

Goal: determine whether the LLM intent_only weakness for exploration is due to
  (a) label quality (exploration assigned where it hurts)
  (b) modulation policy (boost_scale / blend weights too aggressive for exploration)

Ablation variants (exploration policy only; all other reasons unchanged):
  baseline   — current config (boost_scale=0.5, intent_weight=0.4)
  zero       — exploration delta = 0 (treat exploration like backbone_only)
  half       — boost_scale = 0.25 (half of current)
  conf_gate  — exploration delta only when confidence >= 0.65
  spread_gate— exploration delta only when top_dominance < 0.35 (wide spread, low concentration)

Outputs (to --out-dir):
  exploration_diagnostic.json   — per-bin breakdown
  exploration_ablation.csv      — metrics per variant × experiment
  exploration_ablation_report.md — markdown summary answering the 4 key questions

Usage:
  python -m src.evaluation.run_exploration_ablation \\
    --data-config config/data/amazon_movies_tv.yaml \\
    --evaluation-config config/evaluation/default.yaml \\
    --backbone-config config/backbone/amazon_movies_tv_sasrec.yaml \\
    --modulation-config config/modulation/amazon_movies_tv.yaml \\
    --llm-intent-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet \\
    --max-users 2000 \\
    --out-dir /home/jhpark/PGIM/results/exploration_ablation
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

# ── ablation variant definitions ─────────────────────────────────────────────
# Each entry: name -> callable(intent_record, modulation_cfg) -> (patched_cfg, skip_delta)
# We patch modulation_cfg per-call for exploration rows.
# skip_delta=True means we zero out delta inside the reranker for this row.

_VARIANTS = {
    "baseline":   {"boost_scale": 0.5,  "conf_threshold": None, "spread_threshold": None},
    "zero":       {"boost_scale": 0.0,  "conf_threshold": None, "spread_threshold": None},
    "half":       {"boost_scale": 0.25, "conf_threshold": None, "spread_threshold": None},
    "conf_gate":  {"boost_scale": 0.5,  "conf_threshold": 0.65, "spread_threshold": None},
    "spread_gate":{"boost_scale": 0.5,  "conf_threshold": None, "spread_threshold": 0.35},
}

# Goal-source variants for Stage 2 grounding ablation.
# raw_llm_goals  — uses goal_concepts (Stage 1 output, before validation)
# validated_llm_goals — uses validated_goal_concepts (Stage 2 output, grounded)
# These are layered on top of _VARIANTS at baseline boost_scale only.
_GOAL_SOURCE_VARIANTS = {
    "raw_llm_goals":       "goal_concepts",           # Stage 1 output field name
    "validated_llm_goals": "validated_goal_concepts", # Stage 2 output field name
}


def _patch_goal_source(intent_record: dict, goal_field: str) -> dict:
    """
    Return a shallow copy of intent_record with goal_concepts overwritten
    by the specified goal_field value.  Preserves all other fields.

    goal_field: "goal_concepts" (raw LLM) or "validated_goal_concepts" (Stage 2)

    If goal_field is absent (old cache without Stage 2), falls back to goal_concepts.
    """
    patched = dict(intent_record)
    src = intent_record.get(goal_field)
    if src is None:
        pass  # keep original goal_concepts
    elif isinstance(src, str):
        # parquet sometimes serializes list as JSON string
        import json as _json
        try:
            patched["goal_concepts"] = _json.loads(src)
        except Exception:
            pass
    elif hasattr(src, '__iter__'):
        # list, numpy.ndarray, or other iterable
        patched["goal_concepts"] = list(src)
    # validated_goal_concepts field must always be consistent with what enters signal_builder
    patched["validated_goal_concepts"] = patched["goal_concepts"]
    return patched


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _intent_keys(path: Path) -> set[tuple[str, int]]:
    df = pd.read_parquet(path, columns=["user_id", "target_index"])
    return set(zip(df["user_id"], df["target_index"].astype(int)))


def _load_intent_filtered(
    path: Path, shared_keys: set[tuple[str, int]]
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


def _patched_cfg(base_cfg: dict, boost_scale: float) -> dict:
    """Return a deep-copied modulation_cfg with exploration boost_scale overridden."""
    cfg = copy.deepcopy(base_cfg)
    if "reason_policy" not in cfg:
        cfg["reason_policy"] = {}
    if "exploration" not in cfg["reason_policy"]:
        cfg["reason_policy"]["exploration"] = {}
    cfg["reason_policy"]["exploration"]["boost_scale"] = boost_scale
    return cfg


def _run_variant_eval(
    variant_name: str,
    variant_cfg: dict,
    intent_by_key: dict[tuple[str, int], dict],
    cand_by_key: dict[tuple[str, int], list[str]],
    gt_by_key: dict[tuple[str, int], str],
    is_gt_by_key: dict[tuple[str, int], set[str]],
    backbone_scores: dict[tuple[str, int], dict[str, float]],
    persona_nodes_by_user: dict[str, list[dict]],
    item_concepts: dict[str, list[str]],
    base_modulation_cfg: dict,
    k_values: list[int],
    experiment_modes: dict[str, str],
    goal_source: str = "validated_goal_concepts",
) -> dict[str, dict]:
    """Run eval for one ablation variant. Returns {exp_name: metrics_dict}.

    goal_source: which field to use as goal_concepts for modulation.
        "validated_goal_concepts" (default) — Stage 2 grounded output
        "goal_concepts"                      — Stage 1 raw LLM output (for ablation)
    """
    from src.modulation.gate import compute_gate_strength
    from src.modulation.reranker import CandidateReranker
    from src.modulation.signal_builder import build_signal

    conf_threshold  = variant_cfg.get("conf_threshold")
    spread_threshold = variant_cfg.get("spread_threshold")
    boost_scale     = variant_cfg.get("boost_scale", 0.5)

    # Pre-patch cfg for exploration
    mod_cfg_patched = _patched_cfg(base_modulation_cfg, boost_scale)
    reranker = CandidateReranker(mod_cfg_patched, item_concepts)

    results_by_exp: dict[str, dict] = {}

    for exp_name, mode in experiment_modes.items():
        all_ranked: list[dict] = []

        for (uid, tidx), candidate_ids in cand_by_key.items():
            scores = backbone_scores[(uid, tidx)]
            candidate_tuples = sorted(
                [(iid, scores[iid]) for iid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )

            # Apply goal source selection (Stage 1 raw vs Stage 2 validated)
            intent_record = _patch_goal_source(
                intent_by_key[(uid, tidx)], goal_source
            )
            reason = intent_record.get("deviation_reason", "unknown")
            confidence = float(intent_record.get("confidence", 0.5))

            # Determine effective boost_scale for this row
            eff_boost_scale = boost_scale
            if reason == "exploration":
                if conf_threshold is not None and confidence < conf_threshold:
                    eff_boost_scale = 0.0
                if spread_threshold is not None:
                    # top_dominance not stored in intent; infer from goal_concepts count
                    # Use confidence as proxy: low confidence exploration = wide spread
                    # Better: store top_dominance in intent record if available
                    # For now use the intent record's persona_alignment_score as spread proxy
                    alignment = float(intent_record.get("persona_alignment_score", 0.5))
                    # wide spread = low alignment AND low confidence
                    if not (alignment < 0.35 and confidence < 0.70):
                        eff_boost_scale = 0.0

                if eff_boost_scale != boost_scale:
                    # Re-patch cfg for this row
                    row_cfg = _patched_cfg(base_modulation_cfg, eff_boost_scale)
                    row_reranker = CandidateReranker(row_cfg, item_concepts)
                else:
                    row_reranker = reranker
            else:
                row_reranker = reranker

            # Use per-row reranker cfg if needed
            active_cfg = row_reranker._cfg if row_reranker is not reranker else mod_cfg_patched

            persona_nodes = persona_nodes_by_user.get(uid, [])
            gate_strength = compute_gate_strength(
                deviation_reason=reason,
                confidence=confidence,
                persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                gate_cfg=base_modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_record, persona_nodes, gate_strength, active_cfg, mode=mode)
            ranked = row_reranker.rerank(candidate_tuples, signal, mode=mode)

            gt_items = is_gt_by_key.get((uid, tidx), set())
            for r in ranked:
                rec = r.to_record()
                rec["is_ground_truth"] = int(rec["candidate_item_id"] in gt_items)
                all_ranked.append(rec)

        df_ranked = pd.DataFrame(all_ranked)

        # Compute metrics
        gt_rows = df_ranked[df_ranked["is_ground_truth"] == 1].copy()
        gt_rows = gt_rows.sort_values(["user_id", "target_index", "final_score"], ascending=[True, True, False])
        gt_rows["_rank"] = gt_rows.groupby(["user_id", "target_index"]).cumcount() + 1

        # rank_after from ranked output
        df_ranked_s = df_ranked.sort_values(
            ["user_id", "target_index", "final_score"], ascending=[True, True, False]
        ).copy()
        df_ranked_s["_rank"] = df_ranked_s.groupby(["user_id", "target_index"]).cumcount() + 1
        gt_mask = df_ranked_s["is_ground_truth"] == 1
        gt_df = df_ranked_s[gt_mask].copy()

        if gt_df.empty:
            results_by_exp[exp_name] = {}
            continue

        ranks = gt_df["_rank"].values.astype(int)
        n = len(ranks)
        metrics = {"n_users": n, "experiment": exp_name, "variant": variant_name}
        for k in k_values:
            metrics[f"HR@{k}"]   = round(float((ranks <= k).mean()), 4)
            metrics[f"NDCG@{k}"] = round(
                float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= k) / n), 4
            )
        metrics["MRR"] = round(float((1.0 / ranks).mean()), 4)

        # rank movement (need rank_before from ranked)
        if "rank_before" in gt_df.columns and "rank_after" in gt_df.columns:
            improved = int((gt_df["rank_after"] < gt_df["rank_before"]).sum())
            worsened = int((gt_df["rank_after"] > gt_df["rank_before"]).sum())
        else:
            improved = worsened = -1
        metrics["improved"] = improved
        metrics["worsened"] = worsened

        # GT delta stats
        if "modulation_delta" in gt_df.columns:
            gt_deltas = gt_df["modulation_delta"]
            metrics["gt_delta_mean"]    = round(float(gt_deltas.mean()), 6)
            metrics["gt_delta_pos_frac"] = round(float((gt_deltas > 0).mean()), 4)
        else:
            metrics["gt_delta_mean"] = metrics["gt_delta_pos_frac"] = None

        results_by_exp[exp_name] = metrics
        logger.info(
            "[%s/%s]  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  improved=%d  worsened=%d  gt_delta+frac=%.3f",
            variant_name, exp_name,
            metrics.get("HR@10", 0), metrics.get("NDCG@10", 0), metrics.get("MRR", 0),
            improved, worsened, metrics.get("gt_delta_pos_frac", 0),
        )

    return results_by_exp


def _build_exploration_diagnostic(
    intent_by_key: dict[tuple[str, int], dict],
    ranked_parquet: Path,
    k_values: list[int],
) -> dict:
    """
    Detailed breakdown for exploration rows only.
    Uses the baseline reranked parquet (already computed).
    """
    if not ranked_parquet.exists():
        return {}

    df = pd.read_parquet(ranked_parquet)
    exp_df = df[df["deviation_reason"] == "exploration"].copy()
    if exp_df.empty:
        return {"n_exploration": 0}

    n_total = exp_df["user_id"].nunique()

    # GT rows only
    gt_df = exp_df[exp_df["is_ground_truth"] == 1].copy() if "is_ground_truth" in exp_df.columns else pd.DataFrame()

    diag: dict = {"n_exploration_users": n_total}

    if gt_df.empty:
        diag["note"] = "no is_ground_truth column or no GT rows"
        return diag

    # Overall stats
    ranks = gt_df["rank_after"].values.astype(int) if "rank_after" in gt_df.columns else None
    if ranks is not None:
        n = len(ranks)
        diag["overall"] = {
            "n_users": n,
            "HR@10": round(float((ranks <= 10).mean()), 4),
            "NDCG@10": round(float(sum(1.0 / math.log2(r + 1) for r in ranks if r <= 10) / n), 4),
            "MRR": round(float((1.0 / ranks).mean()), 4),
        }

    if "modulation_delta" in gt_df.columns:
        gt_deltas = gt_df["modulation_delta"]
        diag["gt_delta"] = {
            "mean":     round(float(gt_deltas.mean()), 6),
            "pos_frac": round(float((gt_deltas > 0).mean()), 4),
            "neg_frac": round(float((gt_deltas < 0).mean()), 4),
            "zero_frac": round(float((gt_deltas == 0).mean()), 4),
        }

    if "rank_before" in gt_df.columns and "rank_after" in gt_df.columns:
        improved = int((gt_df["rank_after"] < gt_df["rank_before"]).sum())
        same     = int((gt_df["rank_after"] == gt_df["rank_before"]).sum())
        worsened = int((gt_df["rank_after"] > gt_df["rank_before"]).sum())
        diag["rank_movement"] = {"improved": improved, "same": same, "worsened": worsened}

    # Confidence bins
    conf_bins: dict[str, dict] = {}
    for label, (lo, hi) in [("low", (0.0, 0.55)), ("mid", (0.55, 0.70)), ("high", (0.70, 1.01))]:
        # match intent confidence via user_id + target_index
        bin_users = {
            (r["user_id"], int(r["target_index"]))
            for r in intent_by_key.values()
            if r.get("deviation_reason") == "exploration"
            and lo <= float(r.get("confidence", 0)) < hi
        }
        bin_gt = gt_df[
            gt_df.apply(lambda row: (row["user_id"], int(row["target_index"])) in bin_users, axis=1)
        ]
        if not bin_gt.empty:
            r_arr = bin_gt["rank_after"].values.astype(int)
            nb = len(r_arr)
            conf_bins[label] = {
                "n": nb,
                "HR@10": round(float((r_arr <= 10).mean()), 4),
                "gt_delta_pos_frac": round(
                    float((bin_gt["modulation_delta"] > 0).mean()), 4
                ) if "modulation_delta" in bin_gt.columns else None,
            }
    diag["by_confidence"] = conf_bins

    # Alignment score bins
    align_bins: dict[str, dict] = {}
    for label, (lo, hi) in [("low", (0.0, 0.30)), ("mid", (0.30, 0.55)), ("high", (0.55, 1.01))]:
        bin_users = {
            (r["user_id"], int(r["target_index"]))
            for r in intent_by_key.values()
            if r.get("deviation_reason") == "exploration"
            and lo <= float(r.get("persona_alignment_score", 0)) < hi
        }
        bin_gt = gt_df[
            gt_df.apply(lambda row: (row["user_id"], int(row["target_index"])) in bin_users, axis=1)
        ]
        if not bin_gt.empty:
            r_arr = bin_gt["rank_after"].values.astype(int)
            nb = len(r_arr)
            align_bins[label] = {
                "n": nb,
                "HR@10": round(float((r_arr <= 10).mean()), 4),
                "gt_delta_pos_frac": round(
                    float((bin_gt["modulation_delta"] > 0).mean()), 4
                ) if "modulation_delta" in bin_gt.columns else None,
            }
    diag["by_alignment"] = align_bins

    return diag


def _write_report(
    diag: dict,
    ablation_rows: list[dict],
    out_path: Path,
    k_values: list[int],
    heur_control_rows: list[dict] | None = None,
) -> None:
    """Write markdown diagnostic report."""
    lines = ["# Exploration Ablation Report — Amazon Movies & TV\n"]

    # ── Section 1: Diagnostic ────────────────────────────────────────
    lines.append("## 1. Exploration Diagnostic (baseline intent_only)\n")
    lines.append(f"- n_exploration_users: {diag.get('n_exploration_users', 'N/A')}\n")

    ov = diag.get("overall", {})
    if ov:
        lines.append(f"- HR@10: {ov.get('HR@10', 'N/A')}")
        lines.append(f"- NDCG@10: {ov.get('NDCG@10', 'N/A')}")
        lines.append(f"- MRR: {ov.get('MRR', 'N/A')}\n")

    gd = diag.get("gt_delta", {})
    if gd:
        lines.append("### GT Delta Stats")
        lines.append(f"- mean delta: {gd.get('mean', 'N/A')}")
        lines.append(f"- positive fraction: {gd.get('pos_frac', 'N/A')}")
        lines.append(f"- negative fraction: {gd.get('neg_frac', 'N/A')}")
        lines.append(f"- zero fraction: {gd.get('zero_frac', 'N/A')}\n")

    rm = diag.get("rank_movement", {})
    if rm:
        lines.append("### Rank Movement (exploration rows)")
        lines.append(f"- improved: {rm.get('improved', 'N/A')}")
        lines.append(f"- same: {rm.get('same', 'N/A')}")
        lines.append(f"- worsened: {rm.get('worsened', 'N/A')}\n")

    lines.append("### Breakdown by Confidence Bin")
    for label, stats in diag.get("by_confidence", {}).items():
        lines.append(
            f"- {label}: n={stats.get('n')}  HR@10={stats.get('HR@10')}  gt_delta+frac={stats.get('gt_delta_pos_frac')}"
        )
    lines.append("")

    lines.append("### Breakdown by Alignment Score Bin")
    for label, stats in diag.get("by_alignment", {}).items():
        lines.append(
            f"- {label}: n={stats.get('n')}  HR@10={stats.get('HR@10')}  gt_delta+frac={stats.get('gt_delta_pos_frac')}"
        )
    lines.append("")

    # ── Section 2: Ablation Table ────────────────────────────────────
    lines.append("## 2. Ablation Results\n")
    if ablation_rows:
        df_abl = pd.DataFrame(ablation_rows)

        cols = ["variant", "HR@10", "NDCG@10", "MRR", "improved", "worsened", "gt_delta_pos_frac"]
        for exp in df_abl["experiment"].unique():
            lines.append(f"### Experiment: {exp}")
            sub = df_abl[df_abl["experiment"] == exp][cols].reset_index(drop=True)
            # manual markdown table (no tabulate dependency)
            header = "| " + " | ".join(cols) + " |"
            sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
            lines.append(header)
            lines.append(sep)
            for _, row in sub.iterrows():
                lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
            lines.append("")

    # ── Section 3: Key Questions ─────────────────────────────────────
    lines.append("## 3. Analysis\n")

    # Q1: label quality vs modulation policy?
    gd_pos = gd.get("pos_frac", 0.5) if gd else 0.5
    gd_neg = gd.get("neg_frac", 0.5) if gd else 0.5
    if gd_pos < 0.40:
        q1 = (
            "**Modulation policy** is the primary problem. GT delta +frac is very low "
            f"({gd_pos:.2f}), meaning the exploration boost actively hurts GT items. "
            "The label itself may be reasonable, but the boost_concepts selected for "
            "exploration are pointing away from GT."
        )
    elif gd_neg > 0.30:
        q1 = (
            "**Both label quality and modulation policy** contribute. Some exploration "
            "labels are misclassified (label quality), and the modulation signal for "
            "exploration regularly produces negative GT deltas (policy problem)."
        )
    else:
        q1 = (
            "**Label quality** may be the dominant issue. GT delta +frac is not strongly "
            "negative, suggesting the boost direction is roughly correct but the label "
            "assignment is noisy — exploration is assigned to cases that would benefit "
            "more from aligned or task_focus treatment."
        )
    lines.append(f"### Q1: Label quality vs modulation policy?\n{q1}\n")

    # Q2: best variant
    best_variant = "baseline"
    best_hr = 0.0
    for row in ablation_rows:
        if row.get("experiment") == "ablation_intent_only":
            hr = row.get("HR@10", 0.0) or 0.0
            if hr > best_hr:
                best_hr = hr
                best_variant = row["variant"]
    lines.append(
        f"### Q2: Best exploration policy (intent_only)\n"
        f"Best variant: **{best_variant}** (HR@10={best_hr:.4f})\n"
    )

    # Q3: intent_only recovery
    baseline_intent = next(
        (r for r in ablation_rows if r["variant"] == "baseline" and r["experiment"] == "ablation_intent_only"), {}
    )
    best_intent = next(
        (r for r in ablation_rows if r["variant"] == best_variant and r["experiment"] == "ablation_intent_only"), {}
    )
    baseline_full = next(
        (r for r in ablation_rows if r["variant"] == "baseline" and r["experiment"] == "full_model"), {}
    )
    best_full = next(
        (r for r in ablation_rows if r["variant"] == best_variant and r["experiment"] == "full_model"), {}
    )

    delta_intent = (best_intent.get("HR@10") or 0) - (baseline_intent.get("HR@10") or 0)
    delta_full   = (best_full.get("HR@10") or 0) - (baseline_full.get("HR@10") or 0)
    lines.append(
        f"### Q3: intent_only recovery without hurting full_model?\n"
        f"- intent_only HR@10 delta: {delta_intent:+.4f}  ({best_variant} vs baseline)\n"
        f"- full_model HR@10 delta:  {delta_full:+.4f}  ({best_variant} vs baseline)\n"
    )
    if delta_intent > 0 and delta_full >= -0.002:
        q3 = "Yes — the best variant improves intent_only without meaningfully hurting full_model."
    elif delta_intent > 0 and delta_full < -0.002:
        q3 = "Partial — intent_only improves but full_model is hurt. Tradeoff needs consideration."
    else:
        q3 = "No clear recovery in intent_only from exploration policy changes alone."
    lines.append(f"{q3}\n")

    # Q4: Phase 3 recommendation
    lines.append("### Q4: Phase 3 (reason-conditioned fusion) vs Phase 2 (persona graph strengthening)?\n")
    if delta_intent < 0.003 and gd_pos < 0.45:
        q4 = (
            "**Recommend Phase 3 next.** The exploration modulation problem is structural: "
            "policy changes alone cannot sufficiently fix intent_only. The core issue is that "
            "exploration's boost direction (goal_concepts) does not reliably correlate with GT. "
            "Reason-conditioned fusion (Phase 3) would allow the model to down-weight intent "
            "signal for exploration cases dynamically, rather than applying a fixed policy.\n\n"
            "Phase 2 (persona graph strengthening) is less urgent because full_model already "
            "recovers via persona signal — the persona graph is functioning as intended."
        )
    elif delta_intent > 0.005:
        q4 = (
            "**Policy fix is sufficient for now.** The best ablation variant meaningfully "
            "improves intent_only, suggesting exploration modulation policy is the bottleneck. "
            "Apply the best variant as the new config baseline before committing to Phase 3."
        )
    else:
        q4 = (
            "**Phase 3 is recommended** as the next major step. Marginal policy improvements "
            "exist but are insufficient. Reason-conditioned fusion would enable the model to "
            "learn when exploration intent signal is trustworthy vs. when to fall back to persona."
        )
    lines.append(q4)
    lines.append("")

    # ── Section 4: LLM reason utility (heuristic control) ────────────
    if heur_control_rows:
        lines.append("## 4. LLM Reason Utility vs Heuristic Control\n")
        lines.append(
            "이번 수정은 reason source를 LLM으로 유지한 채 candidate grounding 실패를 "
            "Stage 2 activation gate로 해결하는 방향으로 진행됨. "
            "아래는 LLM reason baseline vs heuristic reason control 비교 결과.\n"
        )
        llm_baseline_intent = next(
            (r for r in ablation_rows if r.get("variant") == "baseline" and r.get("experiment") == "ablation_intent_only"), {}
        )
        llm_baseline_full = next(
            (r for r in ablation_rows if r.get("variant") == "baseline" and r.get("experiment") == "full_model"), {}
        )
        hc_intent = next(
            (r for r in heur_control_rows if r.get("experiment") == "ablation_intent_only"), {}
        )
        hc_full = next(
            (r for r in heur_control_rows if r.get("experiment") == "full_model"), {}
        )
        lines.append("| Source | Mode | HR@10 | NDCG@10 | MRR |")
        lines.append("| --- | --- | --- | --- | --- |")
        for label, row in [
            ("LLM baseline", llm_baseline_intent),
            ("Heuristic control", hc_intent),
            ("LLM baseline (full)", llm_baseline_full),
            ("Heuristic control (full)", hc_full),
        ]:
            if row:
                lines.append(
                    f"| {label} | {row.get('experiment', '')} "
                    f"| {row.get('HR@10', 'N/A')} | {row.get('NDCG@10', 'N/A')} | {row.get('MRR', 'N/A')} |"
                )
        lines.append("")
        try:
            delta_hr_intent = float(llm_baseline_intent.get("HR@10", 0)) - float(hc_intent.get("HR@10", 0))
            delta_hr_full   = float(llm_baseline_full.get("HR@10", 0))   - float(hc_full.get("HR@10", 0))
            verdict_intent = "LLM reason이 heuristic보다 유리" if delta_hr_intent > 0.001 else \
                             ("차이 없음" if abs(delta_hr_intent) <= 0.001 else "heuristic reason이 더 유리")
            verdict_full   = "LLM reason이 heuristic보다 유리" if delta_hr_full > 0.001 else \
                             ("차이 없음" if abs(delta_hr_full) <= 0.001 else "heuristic reason이 더 유리")
            lines.append(f"- intent_only HR@10 delta (LLM - heur): {delta_hr_intent:+.4f}  → {verdict_intent}")
            lines.append(f"- full_model HR@10 delta (LLM - heur): {delta_hr_full:+.4f}  → {verdict_full}")
        except (TypeError, ValueError):
            pass
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("saved -> %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",       default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--evaluation-config", default="config/evaluation/default.yaml")
    parser.add_argument("--backbone-config",   default="config/backbone/amazon_movies_tv_sasrec.yaml")
    parser.add_argument("--modulation-config", default="config/modulation/amazon_movies_tv.yaml")
    parser.add_argument("--llm-intent-path",   default=None)
    parser.add_argument("--heur-intent-path",  default=None,
                        help="Heuristic intent cache for control comparison. "
                             "If provided, runs an extra heuristic-reason control branch "
                             "alongside LLM branches to assess LLM reason utility.")
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--out-dir",   default="/home/jhpark/PGIM/results/exploration_ablation")
    parser.add_argument("--backbone-scores-cache", default=None)
    args = parser.parse_args()

    data_cfg       = _load_yaml(args.data_config)
    eval_cfg       = _load_yaml(args.evaluation_config)
    backbone_cfg   = _load_yaml(args.backbone_config)
    modulation_cfg = _load_yaml(args.modulation_config)

    dataset      = data_cfg.get("dataset", "amazon_movies_tv")
    k_values     = eval_cfg.get("k_values", [5, 10, 20])
    interim_dir  = Path(data_cfg["paths"]["interim_dir"])
    processed_dir= Path(data_cfg["paths"]["processed_dir"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── intent cache ──────────────────────────────────────────────────
    llm_path = Path(
        args.llm_intent_path
        or f"data/cache/intent/{dataset}/short_term_intents_llm_subset_2000_validated.parquet"
    )
    if not llm_path.exists():
        logger.error("LLM intent cache not found: %s", llm_path)
        return

    # ── shared_keys ───────────────────────────────────────────────────
    cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k101.parquet")
    if not cand_path.exists():
        logger.error("Candidates not found: %s", cand_path)
        return

    logger.info("Computing shared_keys...")
    df_cands_ids = pd.read_parquet(cand_path, columns=["user_id", "target_index"])
    cand_keys = set(zip(df_cands_ids["user_id"], df_cands_ids["target_index"].astype(int)))
    l_keys    = _intent_keys(llm_path)

    llm_users_sorted = sorted({k[0] for k in l_keys})[:args.max_users]
    llm_users_set    = set(llm_users_sorted)
    shared_keys = {k for k in (cand_keys & l_keys) if k[0] in llm_users_set}

    if not shared_keys:
        logger.error("No shared_keys found.")
        return
    logger.info("shared_keys: %d", len(shared_keys))

    shared_users_set = {k[0] for k in shared_keys}

    # ── load intent ───────────────────────────────────────────────────
    intent_by_key = _load_intent_filtered(llm_path, shared_keys)
    n_exploration = sum(1 for v in intent_by_key.values() if v.get("deviation_reason") == "exploration")
    logger.info("intent loaded: %d keys  exploration=%d (%.1f%%)",
                len(intent_by_key), n_exploration, 100 * n_exploration / max(1, len(intent_by_key)))

    # ── load heuristic intent (optional, for control comparison) ──────
    heur_intent_by_key: dict[tuple[str, int], dict] = {}
    if args.heur_intent_path:
        heur_path = Path(args.heur_intent_path)
        if not heur_path.exists():
            logger.warning("--heur-intent-path not found: %s — control comparison skipped", heur_path)
        else:
            heur_intent_by_key = _load_intent_filtered(heur_path, shared_keys)
            n_heur_exp = sum(1 for v in heur_intent_by_key.values() if v.get("deviation_reason") == "exploration")
            logger.info(
                "heuristic intent loaded: %d keys  exploration=%d (%.1f%%)",
                len(heur_intent_by_key), n_heur_exp, 100 * n_heur_exp / max(1, len(heur_intent_by_key)),
            )
    else:
        logger.info("--heur-intent-path not provided; control comparison disabled.")

    # ── load shared data ──────────────────────────────────────────────
    logger.info("Loading shared data...")
    df_sequences     = pd.read_parquet(interim_dir / "user_sequences.parquet")
    df_snaps         = pd.read_parquet(interim_dir / "recent_context_snapshots.parquet")
    df_interactions  = pd.read_parquet(processed_dir / "interactions.parquet")
    df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")
    df_persona       = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")

    item_concepts: dict[str, list[str]] = (
        df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    persona_nodes_by_user: dict[str, list[dict]] = {
        uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
    }

    # ── build candidate index ──────────────────────────────────────────
    logger.info("Building candidate index...")
    df_cands = pd.read_parquet(cand_path)
    df_cands = df_cands[df_cands["user_id"].isin(shared_users_set)]
    key_series = list(zip(df_cands["user_id"], df_cands["target_index"].astype(int)))
    df_cands = df_cands[[k in shared_keys for k in key_series]].reset_index(drop=True)

    cand_by_key: dict[tuple[str, int], list[str]] = {}
    gt_by_key:   dict[tuple[str, int], str] = {}
    is_gt_by_key: dict[tuple[str, int], set[str]] = {}
    for row in df_cands.itertuples(index=False):
        key = (row.user_id, int(row.target_index))
        if key not in cand_by_key:
            cand_by_key[key] = []
            is_gt_by_key[key] = set()
        cand_by_key[key].append(row.candidate_item_id)
        if row.is_ground_truth:
            gt_by_key[key] = row.candidate_item_id
            is_gt_by_key[key].add(row.candidate_item_id)
    logger.info("Candidate index: %d keys", len(cand_by_key))

    # ── backbone scores ───────────────────────────────────────────────
    bs_cache = Path(
        args.backbone_scores_cache
        or f"data/cache/backbone/{dataset}/backbone_scores.parquet"
    )
    logger.info("Loading backbone scores...")
    df_bs = pd.read_parquet(bs_cache)
    df_bs = df_bs[df_bs["user_id"].isin(shared_users_set)]
    backbone_scores: dict[tuple[str, int], dict[str, float]] = {}
    for row in df_bs.itertuples(index=False):
        key = (row.user_id, int(row.target_index))
        if key not in shared_keys:
            continue
        if key not in backbone_scores:
            backbone_scores[key] = {}
        backbone_scores[key][row.candidate_item_id] = float(row.backbone_score)
    del df_bs
    logger.info("Backbone scores: %d keys", len(backbone_scores))

    experiment_modes = {
        "ablation_intent_only": "intent_only_rerank",
        "full_model":           "graph_conditioned_full",
    }

    # Detect whether intent cache has Stage 2 fields
    sample_intent = next(iter(intent_by_key.values()), {})
    _vgc = sample_intent.get("validated_goal_concepts")
    has_stage2 = (
        "validated_goal_concepts" in sample_intent
        and _vgc is not None
        and hasattr(_vgc, '__iter__')
        and not isinstance(_vgc, str)
    )
    if has_stage2:
        logger.info("Stage 2 fields detected in intent cache — goal-source ablation enabled.")
    else:
        logger.info(
            "No validated_goal_concepts in intent cache — "
            "goal-source ablation will compare identical sources (raw only). "
            "Re-run run_build_intent with --backbone-candidates-path to enable Stage 2."
        )

    # ── run ablation variants ─────────────────────────────────────────
    all_ablation_rows: list[dict] = []
    baseline_intent_parquet = out_dir / "baseline_reranked_ablation_intent_only.parquet"

    for variant_name, variant_cfg in _VARIANTS.items():
        logger.info("=" * 60)
        logger.info("Variant: %s", variant_name)
        logger.info("=" * 60)
        results = _run_variant_eval(
            variant_name=variant_name,
            variant_cfg=variant_cfg,
            intent_by_key=intent_by_key,
            cand_by_key=cand_by_key,
            gt_by_key=gt_by_key,
            is_gt_by_key=is_gt_by_key,
            backbone_scores=backbone_scores,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            base_modulation_cfg=modulation_cfg,
            k_values=k_values,
            experiment_modes=experiment_modes,
            goal_source="validated_goal_concepts",  # Stage 2 output (default)
        )

        # Save baseline reranked parquet for diagnostic
        if variant_name == "baseline":
            # Re-run to save full parquet for diagnostic
            from src.modulation.gate import compute_gate_strength
            from src.modulation.reranker import CandidateReranker
            from src.modulation.signal_builder import build_signal

            reranker = CandidateReranker(modulation_cfg, item_concepts)
            all_ranked_baseline: list[dict] = []
            for (uid, tidx), candidate_ids in cand_by_key.items():
                scores = backbone_scores[(uid, tidx)]
                candidate_tuples = sorted(
                    [(iid, scores[iid]) for iid in candidate_ids],
                    key=lambda x: x[1], reverse=True,
                )
                intent_record = dict(intent_by_key[(uid, tidx)])
                reason = intent_record.get("deviation_reason", "unknown")
                confidence = float(intent_record.get("confidence", 0.5))
                persona_nodes = persona_nodes_by_user.get(uid, [])
                gate_strength = compute_gate_strength(
                    deviation_reason=reason,
                    confidence=confidence,
                    persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                    gate_cfg=modulation_cfg.get("gate", {}),
                )
                signal = build_signal(intent_record, persona_nodes, gate_strength, modulation_cfg, mode="intent_only_rerank")
                ranked = reranker.rerank(candidate_tuples, signal, mode="intent_only_rerank")
                gt_items = is_gt_by_key.get((uid, tidx), set())
                for r in ranked:
                    rec = r.to_record()
                    rec["is_ground_truth"] = int(rec["candidate_item_id"] in gt_items)
                    all_ranked_baseline.append(rec)
            pd.DataFrame(all_ranked_baseline).to_parquet(baseline_intent_parquet, index=False)
            logger.info("saved baseline parquet -> %s", baseline_intent_parquet)

        for exp_name, metrics in results.items():
            all_ablation_rows.append(metrics)

    # ── heuristic control comparison ──────────────────────────────────
    # Control branch: run baseline eval using heuristic intent (heur_intent_by_key).
    # This lets us compare LLM reason vs heuristic reason on the same user set,
    # quantifying the actual utility of LLM-derived deviation reasons.
    heur_control_rows: list[dict] = []
    if heur_intent_by_key:
        # Filter heur_intent to shared_keys that also exist in LLM intent
        heur_shared = {k: v for k, v in heur_intent_by_key.items() if k in intent_by_key}
        if heur_shared:
            logger.info("=" * 60)
            logger.info("Heuristic control: baseline eval on %d shared keys", len(heur_shared))
            logger.info("=" * 60)
            heur_results = _run_variant_eval(
                variant_name="heur_baseline",
                variant_cfg=_VARIANTS["baseline"],
                intent_by_key=heur_shared,
                cand_by_key={k: v for k, v in cand_by_key.items() if k in heur_shared},
                gt_by_key={k: v for k, v in gt_by_key.items() if k in heur_shared},
                is_gt_by_key={k: v for k, v in is_gt_by_key.items() if k in heur_shared},
                backbone_scores={k: v for k, v in backbone_scores.items() if k in heur_shared},
                persona_nodes_by_user=persona_nodes_by_user,
                item_concepts=item_concepts,
                base_modulation_cfg=modulation_cfg,
                k_values=k_values,
                experiment_modes=experiment_modes,
                goal_source="goal_concepts",  # heuristic intent has no Stage 2 fields
            )
            for exp_name, metrics in heur_results.items():
                if metrics:
                    metrics["intent_source"] = "heuristic"
                heur_control_rows.append(metrics)
            # Save heuristic control CSV
            heur_path_out = out_dir / "heuristic_control.csv"
            pd.DataFrame(heur_control_rows).to_csv(heur_path_out, index=False)
            logger.info("saved -> %s", heur_path_out)

    # ── goal-source ablation: raw_llm_goals vs validated_llm_goals ────
    # Compares Stage 1 raw LLM output vs Stage 2 grounded output at baseline boost_scale.
    # Only exploration rows are affected (other reasons have identical raw vs validated
    # unless Stage 2 changes something — persona conflict, activation gate, conf cap).
    logger.info("=" * 60)
    logger.info("Goal-source ablation: raw_llm_goals vs validated_llm_goals")
    logger.info("=" * 60)

    goal_source_rows: list[dict] = []
    baseline_variant_cfg = _VARIANTS["baseline"]

    for gs_name, gs_field in _GOAL_SOURCE_VARIANTS.items():
        logger.info("  goal_source=%s  (field=%s)", gs_name, gs_field)
        gs_results = _run_variant_eval(
            variant_name=f"gs_{gs_name}",
            variant_cfg=baseline_variant_cfg,
            intent_by_key=intent_by_key,
            cand_by_key=cand_by_key,
            gt_by_key=gt_by_key,
            is_gt_by_key=is_gt_by_key,
            backbone_scores=backbone_scores,
            persona_nodes_by_user=persona_nodes_by_user,
            item_concepts=item_concepts,
            base_modulation_cfg=modulation_cfg,
            k_values=k_values,
            experiment_modes=experiment_modes,
            goal_source=gs_field,
        )
        for exp_name, metrics in gs_results.items():
            if metrics:
                metrics["goal_source"] = gs_name
            goal_source_rows.append(metrics)

    # Save goal-source ablation CSV
    gs_path = out_dir / "goal_source_ablation.csv"
    pd.DataFrame(goal_source_rows).to_csv(gs_path, index=False)
    logger.info("saved -> %s", gs_path)

    # ── grounding diagnostics (per-record Stage 2 audit) ──────────────
    # Computes: validated vs raw goal-to-candidate match, GT match,
    # nonzero delta potential, intent-only HR@10 change.
    grounding_diag_rows: list[dict] = []
    if has_stage2:
        from src.intent.grounded_selector import compute_grounding_diagnostics
        from src.intent.grounded_selector import build_candidate_concept_bank

        for (uid, tidx), rec in intent_by_key.items():
            def _to_list(x):
                if x is None:
                    return []
                if isinstance(x, str):
                    import json as _json
                    try:
                        return _json.loads(x)
                    except Exception:
                        return []
                if hasattr(x, '__iter__'):
                    return list(x)
                return []

            _rg = rec.get("raw_llm_goals")
            raw_goals = _to_list(_rg if _rg is not None else rec.get("goal_concepts"))
            val_goals = _to_list(rec.get("validated_goal_concepts"))

            candidate_ids = cand_by_key.get((uid, tidx), [])
            bank = build_candidate_concept_bank(candidate_ids, item_concepts)

            # GT item concepts for this user/target
            gt_items = list(is_gt_by_key.get((uid, tidx), set()))
            gt_concepts: list[str] = []
            for iid in gt_items:
                gt_concepts.extend(item_concepts.get(iid, []))

            diag_entry = compute_grounding_diagnostics(
                raw_goal_concepts=raw_goals,
                validated_goal_concepts=val_goals,
                candidate_concept_bank=bank,
                gt_item_concepts=gt_concepts if gt_concepts else None,
            )
            diag_entry["user_id"] = uid
            diag_entry["target_index"] = tidx
            diag_entry["deviation_reason"] = rec.get("deviation_reason", "unknown")
            diag_entry["confidence"] = float(rec.get("confidence", 0.0))
            diag_entry["n_raw_goals"] = len(raw_goals)
            diag_entry["n_validated_goals"] = len(val_goals)
            grounding_diag_rows.append(diag_entry)

        if grounding_diag_rows:
            df_gd = pd.DataFrame(grounding_diag_rows)
            gd_path = out_dir / "grounding_diagnostics.parquet"
            # drop list-valued columns for parquet (keep scalar summaries only)
            scalar_cols = [c for c in df_gd.columns
                           if not isinstance(df_gd[c].iloc[0], list)]
            df_gd[scalar_cols].to_parquet(gd_path, index=False)
            logger.info("saved -> %s  (%d rows)", gd_path, len(df_gd))

            # Print per-reason summary
            SEP = "=" * 80
            print(f"\n{SEP}")
            print("GROUNDING DIAGNOSTICS — per deviation_reason")
            print(SEP)
            for reason in sorted(df_gd["deviation_reason"].unique()):
                sub = df_gd[df_gd["deviation_reason"] == reason]
                line = (
                    f"  {reason:15s}  n={len(sub):4d}"
                    f"  cand_match_before={sub['candidate_match_before'].mean():.3f}"
                    f"  cand_match_after={sub['candidate_match_after'].mean():.3f}"
                )
                if "any_activated_before" in sub.columns:
                    line += (
                        f"  any_act_before={sub['any_activated_before'].mean():.3f}"
                        f"  any_act_after={sub['any_activated_after'].mean():.3f}"
                    )
                if "avg_activated_cands_before" in sub.columns:
                    line += (
                        f"  avg_act_cands_before={sub['avg_activated_cands_before'].mean():.3f}"
                        f"  avg_act_cands_after={sub['avg_activated_cands_after'].mean():.3f}"
                    )
                if "gt_match_before" in sub.columns:
                    line += (
                        f"  gt_match_before={sub['gt_match_before'].mean():.3f}"
                        f"  gt_match_after={sub['gt_match_after'].mean():.3f}"
                    )
                print(line)

    # ── exploration diagnostic ────────────────────────────────────────
    logger.info("Building exploration diagnostic...")
    diag = _build_exploration_diagnostic(intent_by_key, baseline_intent_parquet, k_values)
    diag_path = out_dir / "exploration_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    logger.info("saved -> %s", diag_path)

    # ── save ablation CSV ─────────────────────────────────────────────
    df_abl = pd.DataFrame(all_ablation_rows)
    abl_path = out_dir / "exploration_ablation.csv"
    df_abl.to_csv(abl_path, index=False)
    logger.info("saved -> %s", abl_path)

    # ── print table ───────────────────────────────────────────────────
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("EXPLORATION ABLATION — intent_only")
    print(SEP)
    sub = df_abl[df_abl["experiment"] == "ablation_intent_only"][
        ["variant", "HR@10", "NDCG@10", "MRR", "improved", "worsened", "gt_delta_pos_frac"]
    ]
    print(sub.to_string(index=False))

    print(f"\n{SEP}")
    print("EXPLORATION ABLATION — full_model")
    print(SEP)
    sub = df_abl[df_abl["experiment"] == "full_model"][
        ["variant", "HR@10", "NDCG@10", "MRR", "improved", "worsened", "gt_delta_pos_frac"]
    ]
    print(sub.to_string(index=False))

    print(f"\n{SEP}")
    print("EXPLORATION DIAGNOSTIC (baseline intent_only)")
    print(SEP)
    print(json.dumps(diag, indent=2))

    # ── write markdown report ─────────────────────────────────────────
    _write_report(
        diag, all_ablation_rows, out_dir / "exploration_ablation_report.md", k_values,
        heur_control_rows=heur_control_rows if heur_control_rows else None,
    )

    # ── goal-source comparison table ──────────────────────────────────
    if goal_source_rows:
        df_gs = pd.DataFrame(goal_source_rows)
        print(f"\n{SEP}")
        print("GOAL-SOURCE ABLATION — raw_llm_goals vs validated_llm_goals (baseline boost_scale)")
        print(SEP)
        for exp in ["ablation_intent_only", "full_model"]:
            sub = df_gs[df_gs.get("experiment", pd.Series(dtype=str)) == exp] if "experiment" in df_gs.columns else pd.DataFrame()
            if sub.empty:
                # experiment field embedded in metrics dict
                sub = df_gs[df_gs.apply(lambda r: r.get("experiment") == exp, axis=1)]
            if not sub.empty:
                print(f"\n  {exp}:")
                cols = [c for c in ["goal_source", "HR@10", "NDCG@10", "MRR",
                                     "gt_delta_pos_frac", "n_users"] if c in sub.columns]
                print(sub[cols].to_string(index=False))

    if heur_control_rows:
        df_hc = pd.DataFrame(heur_control_rows)
        print(f"\n{SEP}")
        print("HEURISTIC CONTROL — LLM reason utility (baseline vs heuristic)")
        print(SEP)
        # Compare LLM baseline vs heuristic baseline side by side
        llm_baseline_rows = [r for r in all_ablation_rows if r.get("variant") == "baseline"]
        df_llm_bl = pd.DataFrame(llm_baseline_rows)
        for exp in ["ablation_intent_only", "full_model"]:
            llm_sub = df_llm_bl[df_llm_bl.apply(lambda r: r.get("experiment") == exp, axis=1)] if not df_llm_bl.empty else pd.DataFrame()
            hc_sub  = df_hc[df_hc.apply(lambda r: r.get("experiment") == exp, axis=1)] if not df_hc.empty else pd.DataFrame()
            print(f"\n  {exp}:")
            if not llm_sub.empty:
                hr_llm  = llm_sub.iloc[0].get("HR@10", "?")
                ndcg_llm = llm_sub.iloc[0].get("NDCG@10", "?")
                mrr_llm  = llm_sub.iloc[0].get("MRR", "?")
                print(f"    LLM (baseline):       HR@10={hr_llm}  NDCG@10={ndcg_llm}  MRR={mrr_llm}")
            if not hc_sub.empty:
                hr_hc   = hc_sub.iloc[0].get("HR@10", "?")
                ndcg_hc  = hc_sub.iloc[0].get("NDCG@10", "?")
                mrr_hc   = hc_sub.iloc[0].get("MRR", "?")
                print(f"    Heuristic (control):  HR@10={hr_hc}  NDCG@10={ndcg_hc}  MRR={mrr_hc}")
            if not llm_sub.empty and not hc_sub.empty:
                try:
                    delta_hr = float(llm_sub.iloc[0].get("HR@10", 0)) - float(hc_sub.iloc[0].get("HR@10", 0))
                    print(f"    Delta (LLM - heur):   HR@10={delta_hr:+.4f}")
                except (TypeError, ValueError):
                    pass

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
