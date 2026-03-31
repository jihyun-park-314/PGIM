"""
sampled_eval.py
---------------
Sampled evaluation protocol for reranking-level assessment.

Separates retrieval difficulty from reranking quality by constructing
guaranteed-GT candidate sets: 1 ground-truth + N random negatives.

Two public functions:
    build_sampled_candidates()  — build + save the candidate sets (run once)
    run_sampled_eval()          — score + rerank + evaluate all experiments

Output files:
    data/cache/candidate/<dataset>/sampled_candidates_k101.parquet
    data/cache/backbone/<dataset>/backbone_scores.parquet          ← NEW (skip on rerun)
    data/artifacts/eval/<dataset>/sampled_reranked_{experiment}.parquet
    data/artifacts/eval/<dataset>/sampled_metrics_summary.csv
    data/artifacts/eval/<dataset>/sampled_diagnostics.json
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.metrics import aggregate, compute_all
from src.evaluation.ranking_eval import build_ground_truth
from src.modulation.gate import compute_gate_strength
from src.modulation.reranker import CandidateReranker
from src.modulation.signal_builder import build_signal

logger = logging.getLogger(__name__)

_NEUTRAL_INTENT = {
    "goal_concepts": [],
    "constraints_json": "{}",
    "deviation_reason": "unknown",
    "confidence": 0.35,
    "ttl_steps": 1,
    "persona_alignment_score": 0.0,
    "is_deviation": 0,
}


# ---------------------------------------------------------------------------
# Step 1: build candidate sets
# ---------------------------------------------------------------------------

def build_sampled_candidates(
    df_sequences: pd.DataFrame,
    eval_snaps: pd.DataFrame,
    all_item_ids: list[str],
    n_negatives: int = 100,
    random_seed: int = 42,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build 1 GT + n_negatives random negatives per eval user.

    Parameters
    ----------
    df_sequences  : user_sequences.parquet (user_id, item_sequence)
    eval_snaps    : last snapshot per user (user_id, target_index, recent_item_ids)
    all_item_ids  : sorted list of all valid item IDs
    n_negatives   : number of negatives per user (default 100 → 101-item sets)
    random_seed   : fixed seed for reproducibility
    out_path      : save path; if None, don't save

    Returns
    -------
    DataFrame with columns:
        user_id, target_index, candidate_item_id, is_ground_truth
    """
    gt = build_ground_truth(df_sequences, eval_snaps)
    all_items_set = set(all_item_ids)

    # Sort for deterministic seed advance order
    sorted_keys = sorted(gt.keys())
    rng = random.Random(random_seed)

    # User history for exclusion (items before target)
    user_seq_map: dict[str, list[str]] = {}
    for _, row in df_sequences.iterrows():
        user_seq_map[row["user_id"]] = list(row["item_sequence"])

    rows: list[dict] = []
    skipped = 0

    for uid, tidx in tqdm(sorted_keys, desc="building sampled candidates"):
        gt_item = gt[(uid, tidx)]
        full_seq = user_seq_map.get(uid, [])
        seen = set(full_seq[:tidx])  # items seen before target

        # Pool = all items except GT and seen history
        pool = [iid for iid in all_item_ids if iid != gt_item and iid not in seen]

        if len(pool) < n_negatives:
            # Very short pool — take all available negatives
            negatives = pool
            skipped += 1
        else:
            negatives = rng.sample(pool, n_negatives)

        # GT row
        rows.append({
            "user_id":           uid,
            "target_index":      tidx,
            "candidate_item_id": gt_item,
            "is_ground_truth":   True,
        })
        # Negative rows
        for neg in negatives:
            rows.append({
                "user_id":           uid,
                "target_index":      tidx,
                "candidate_item_id": neg,
                "is_ground_truth":   False,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Sampled candidates: %d users × ~%d candidates = %d rows  (skipped_small_pool=%d)",
        len(sorted_keys), n_negatives + 1, len(df), skipped,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("saved -> %s", out_path)

    return df


# ---------------------------------------------------------------------------
# Step 2: score + rerank + evaluate
# ---------------------------------------------------------------------------

def _compute_metrics_vectorized(
    df_ranked: pd.DataFrame,
    gt_by_key: dict[tuple[str, int], str],
    k_values: list[int],
) -> pd.DataFrame:
    """
    Vectorized metrics computation. Replaces the 494K × 49.9M row per-user loop.

    Strategy:
    1. Build a gt_map DataFrame and merge onto df_ranked to mark GT rows.
    2. Sort by (user_id, target_index, final_score desc) → assign rank.
    3. For each GT row, rank gives HR/NDCG/MRR directly via numpy.
    """
    # ── build gt lookup DataFrame ────────────────────────────────────
    gt_rows = [
        {"user_id": uid, "target_index": tidx, "gt_item": gt_item}
        for (uid, tidx), gt_item in gt_by_key.items()
    ]
    df_gt = pd.DataFrame(gt_rows)

    # ── sort once by score to assign rank ────────────────────────────
    df_s = df_ranked.sort_values(
        ["user_id", "target_index", "final_score"],
        ascending=[True, True, False],
    ).copy()
    df_s["_rank"] = (
        df_s.groupby(["user_id", "target_index"]).cumcount() + 1
    )  # 1-based rank within each (user, target)

    # ── merge GT info ─────────────────────────────────────────────────
    df_s = df_s.merge(df_gt, on=["user_id", "target_index"], how="left")
    gt_mask = df_s["candidate_item_id"] == df_s["gt_item"]
    df_gt_rows = df_s[gt_mask].copy()

    # ── compute per-user metrics from GT rank ─────────────────────────
    ranks = df_gt_rows["_rank"].values.astype(np.int32)

    result_rows = []
    for (uid, tidx), grp in df_gt_rows.groupby(["user_id", "target_index"], sort=False):
        rank = int(grp["_rank"].iloc[0])
        gt_in_cands = 1.0
        m: dict = {"user_id": uid, "target_index": tidx, "gt_in_candidates": gt_in_cands}
        for k in k_values:
            m[f"HR@{k}"]   = 1.0 if rank <= k else 0.0
            m[f"NDCG@{k}"] = (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
        m["MRR"] = 1.0 / rank
        # rank_before / rank_after / deviation_reason from the GT row
        m["rank_after"]        = int(grp["rank_after"].iloc[0])  if "rank_after"  in grp.columns else rank
        m["rank_before"]       = int(grp["rank_before"].iloc[0]) if "rank_before" in grp.columns else rank
        m["deviation_reason"]  = grp["deviation_reason"].iloc[0]  if "deviation_reason" in grp.columns else "unknown"
        result_rows.append(m)

    # Users whose GT was NOT found (shouldn't happen in sampled eval, but guard)
    found_keys = {(r["user_id"], r["target_index"]) for r in result_rows}
    for (uid, tidx), gt_item in gt_by_key.items():
        if (uid, tidx) not in found_keys:
            m = {"user_id": uid, "target_index": tidx, "gt_in_candidates": 0.0,
                 "MRR": 0.0, "rank_after": None, "rank_before": None, "deviation_reason": "unknown"}
            for k in k_values:
                m[f"HR@{k}"] = 0.0
                m[f"NDCG@{k}"] = 0.0
            result_rows.append(m)

    return pd.DataFrame(result_rows)


def run_sampled_eval(
    experiment_modes: dict[str, str],       # {experiment_name: modulation_mode}
    eval_dir: Path,
    df_sequences: pd.DataFrame,
    df_snaps: pd.DataFrame,
    df_interactions: pd.DataFrame,
    backbone_cfg: dict,
    modulation_cfg: dict,
    item_concepts: dict[str, list[str]],
    persona_nodes_by_user: dict[str, list[dict]],
    intent_by_key: dict[tuple[str, int], dict],
    k_values: list[int],
    cand_path: Path,
    out_dir: Path,
    precomputed_backbone_scores: "dict | None" = None,
    backbone_scores_cache_path: "Path | None" = None,
    restrict_to_intent_keys: bool = False,
) -> "dict":
    """
    Score sampled candidates with backbone, apply modulation per experiment,
    evaluate with reranking metrics.

    All experiments share the same 101-item candidate sets and backbone scores
    (scored once), then each experiment applies its own modulation mode.

    Parameters
    ----------
    backbone_scores_cache_path : optional path to save/load backbone scores parquet.
        If the file exists, scores are loaded instead of recomputed (saves ~3.5h).
        Pass explicitly, or it defaults to data/cache/backbone/<dataset>/backbone_scores.parquet
        (inferred from cand_path location).

    Returns backbone_scores dict so callers can reuse across tuning configs.
    Pass precomputed_backbone_scores to skip the scoring step entirely.
    """
    from src.backbone.sasrec_wrapper import SASRecWrapper

    # ── infer default backbone cache path ────────────────────────────
    if backbone_scores_cache_path is None:
        dataset = cand_path.parts[-2]  # .../candidate/<dataset>/sampled_candidates_k101.parquet
        backbone_scores_cache_path = Path(f"data/cache/backbone/{dataset}/backbone_scores.parquet")

    # ── load sampled candidates ──────────────────────────────────────
    df_cands = pd.read_parquet(cand_path)
    logger.info("Loaded sampled candidates: %d rows", len(df_cands))

    # ── restrict to intent keys if requested ─────────────────────────
    # Computes shared_keys = cand_keys ∩ intent_keys, then filters cand.
    # Missing intent keys raise an error (no silent neutral fallback).
    if restrict_to_intent_keys and intent_by_key:
        intent_keys = set(intent_by_key.keys())
        cand_keys   = set(zip(df_cands["user_id"], df_cands["target_index"].astype(int)))
        shared_keys = cand_keys & intent_keys
        if len(shared_keys) == 0:
            raise ValueError(
                "restrict_to_intent_keys=True but no shared keys between "
                "candidates and intent_by_key. Check user_id / target_index alignment."
            )
        missing_in_intent = cand_keys - intent_keys
        if missing_in_intent:
            logger.info(
                "restrict_to_intent_keys: dropping %d keys not in intent cache "
                "(keeping %d shared keys)",
                len(missing_in_intent), len(shared_keys),
            )
        shared_users  = {k[0] for k in shared_keys}
        shared_tidx   = {k[1] for k in shared_keys}
        df_cands = df_cands[
            df_cands["user_id"].isin(shared_users) &
            df_cands["target_index"].astype(int).isin(shared_tidx)
        ].reset_index(drop=True)
        # Fine-grained filter: drop rows whose (user_id, target_index) pair is not in shared_keys
        # (needed when same tidx appears for different users outside shared_keys)
        key_series = list(zip(df_cands["user_id"], df_cands["target_index"].astype(int)))
        df_cands = df_cands[[k in shared_keys for k in key_series]].reset_index(drop=True)
        logger.info("Restricted candidates: %d rows (%d keys)", len(df_cands), len(shared_keys))

    # Index: (user_id, target_index) -> [item_id, ...]
    cand_by_key: dict[tuple[str, int], list[str]] = {}
    gt_by_key:   dict[tuple[str, int], str] = {}
    # gt_item_by_key for carrying is_ground_truth into ranked output
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

    # ── eval snapshots ───────────────────────────────────────────────
    eval_snaps = df_snaps.loc[
        df_snaps.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)
    snap_by_key: dict[tuple[str, int], list[str]] = {
        (row["user_id"], int(row["target_index"])): list(row["recent_item_ids"])
        for _, row in eval_snaps.iterrows()
    }

    # ── backbone scoring: load cache or compute ───────────────────────
    backbone_scores: dict[tuple[str, int], dict[str, float]]

    if precomputed_backbone_scores is not None:
        backbone_scores = precomputed_backbone_scores
        logger.info("Using precomputed backbone scores (%d keys)", len(backbone_scores))

    elif backbone_scores_cache_path.exists():
        logger.info("Loading backbone scores from cache: %s", backbone_scores_cache_path)
        df_bs = pd.read_parquet(backbone_scores_cache_path)
        backbone_scores = {}
        for row in df_bs.itertuples(index=False):
            key = (row.user_id, int(row.target_index))
            if key not in backbone_scores:
                backbone_scores[key] = {}
            backbone_scores[key][row.candidate_item_id] = float(row.backbone_score)
        logger.info("Backbone scores loaded: %d keys", len(backbone_scores))

    else:
        logger.info("Building backbone scorer...")
        backbone = SASRecWrapper(df_interactions, backbone_cfg)

        logger.info("Scoring %d sampled candidate sets with backbone...", len(cand_by_key))
        backbone_scores = {}
        for (uid, tidx), candidate_ids in tqdm(cand_by_key.items(), desc="backbone scoring"):
            item_sequence = snap_by_key.get((uid, tidx), [])
            all_scores = backbone.get_all_scores(uid, item_sequence)
            backbone_scores[(uid, tidx)] = {
                iid: all_scores.get(iid, 0.0) for iid in candidate_ids
            }

        # ── save backbone scores to cache ─────────────────────────────
        logger.info("Saving backbone scores to cache: %s", backbone_scores_cache_path)
        backbone_scores_cache_path.parent.mkdir(parents=True, exist_ok=True)
        bs_rows = [
            {"user_id": uid, "target_index": tidx, "candidate_item_id": iid, "backbone_score": score}
            for (uid, tidx), scores in backbone_scores.items()
            for iid, score in scores.items()
        ]
        pd.DataFrame(bs_rows).to_parquet(backbone_scores_cache_path, index=False)
        logger.info("Backbone scores cached -> %s  (%d rows)", backbone_scores_cache_path, len(bs_rows))

    # ── reranker (shared structure, mode varies per experiment) ──────
    reranker = CandidateReranker(modulation_cfg, item_concepts)

    # ── per-experiment evaluation ─────────────────────────────────────
    summary_rows: list[dict] = []
    diagnostics: dict = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, mode in experiment_modes.items():
        logger.info("Sampled eval: %s  (mode=%s)", exp_name, mode)
        all_ranked: list[dict] = []
        ctype_counts: dict[str, int] = {}
        total_boost_slots = 0

        for (uid, tidx), candidate_ids in tqdm(
            cand_by_key.items(), desc=exp_name, leave=False
        ):
            scores = backbone_scores[(uid, tidx)]
            candidate_tuples = sorted(
                [(iid, scores[iid]) for iid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )

            intent_record = intent_by_key.get((uid, tidx), None)
            if intent_record is None:
                if restrict_to_intent_keys:
                    # Should never happen after filtering — guard
                    raise KeyError(
                        f"restrict_to_intent_keys=True but intent missing for "
                        f"user={uid} target_index={tidx}"
                    )
                intent_record = {"user_id": uid, "target_index": tidx, **_NEUTRAL_INTENT}
            else:
                intent_record = dict(intent_record)
                intent_record.setdefault("user_id", uid)
                intent_record.setdefault("target_index", tidx)

            persona_nodes = persona_nodes_by_user.get(uid, [])
            gate_strength = compute_gate_strength(
                deviation_reason=intent_record.get("deviation_reason", "unknown"),
                confidence=float(intent_record.get("confidence", 0.35)),
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

            if signal.debug_info:
                boost_ctypes = signal.debug_info.get("boost_ctypes", [])
            else:
                from src.modulation.signal_builder import _concept_type
                boost_ctypes = [_concept_type(c) for c in signal.boost_concepts]
            for ctype in boost_ctypes:
                ctype_counts[ctype] = ctype_counts.get(ctype, 0) + 1
            total_boost_slots += len(boost_ctypes)

        # ── build ranked DataFrame ────────────────────────────────────
        df_ranked = pd.DataFrame(all_ranked)

        ranked_path = out_dir / f"sampled_reranked_{exp_name}.parquet"
        df_ranked.to_parquet(ranked_path, index=False)
        logger.info("saved -> %s  (%d rows)", ranked_path, len(df_ranked))

        if total_boost_slots > 0:
            usage_rows = [
                {"concept_type": ct, "count": cnt, "fraction": round(cnt / total_boost_slots, 4)}
                for ct, cnt in sorted(ctype_counts.items(), key=lambda x: -x[1])
            ]
            pd.DataFrame(usage_rows).to_csv(out_dir / f"concept_usage_{exp_name}.csv", index=False)
            logger.info("concept_usage saved -> %s", out_dir / f"concept_usage_{exp_name}.csv")

        # ── vectorized metrics ────────────────────────────────────────
        logger.info("Computing metrics (vectorized)...")
        df_user = _compute_metrics_vectorized(df_ranked, gt_by_key, k_values)

        gt_coverage = float(df_user["gt_in_candidates"].mean())
        metric_cols = [c for c in df_user.columns
                       if c.startswith(("HR@", "NDCG@", "MRR"))
                       and pd.api.types.is_numeric_dtype(df_user[c])]
        agg = {col: float(df_user[col].mean()) for col in metric_cols}

        row = {"experiment": exp_name, "gt_coverage": round(gt_coverage, 4), "n_users": len(df_user)}
        row.update({k: round(v, 4) for k, v in agg.items()})
        summary_rows.append(row)

        # ── rank movement ─────────────────────────────────────────────
        has_rank = df_user["rank_before"].notna() & df_user["rank_after"].notna()
        gt_found = df_user[has_rank & (df_user["gt_in_candidates"] == 1.0)]
        improved = int((gt_found["rank_after"] < gt_found["rank_before"]).sum())
        same     = int((gt_found["rank_after"] == gt_found["rank_before"]).sum())
        worsened = int((gt_found["rank_after"] > gt_found["rank_before"]).sum())

        # ── delta stats (vectorized merge instead of row-by-row apply) ─
        delta_stats: dict = {}
        if "modulation_delta" in df_ranked.columns:
            df_gt_lookup = pd.DataFrame(
                [{"user_id": uid, "target_index": tidx, "gt_item": gt_item}
                 for (uid, tidx), gt_item in gt_by_key.items()]
            )
            df_merged = df_ranked.merge(df_gt_lookup, on=["user_id", "target_index"], how="inner")
            gt_delta_mask = df_merged["candidate_item_id"] == df_merged["gt_item"]
            gt_deltas = df_merged.loc[gt_delta_mask, "modulation_delta"]
            delta_stats = {
                "all_nonzero_frac":      round(float((df_ranked["modulation_delta"] != 0).mean()), 4),
                "gt_item_mean_delta":    round(float(gt_deltas.mean()), 6) if len(gt_deltas) else None,
                "gt_item_positive_frac": round(float((gt_deltas > 0).mean()), 4) if len(gt_deltas) else None,
            }

        reason_dist = df_user["deviation_reason"].value_counts().to_dict()
        diagnostics[exp_name] = {
            "gt_coverage":   round(gt_coverage, 4),
            "n_users":       len(df_user),
            "reason_dist":   reason_dist,
            "rank_movement": {"improved": improved, "same": same, "worsened": worsened},
            "delta_stats":   delta_stats,
        }

        logger.info(
            "[%s]  gt_cov=%.3f  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f  improved=%d  worsened=%d",
            exp_name, gt_coverage,
            agg.get("HR@10", 0), agg.get("NDCG@10", 0), agg.get("MRR", 0),
            improved, worsened,
        )

    # ── save summary ─────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "sampled_metrics_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("saved -> %s", summary_path)

    diag_path = out_dir / "sampled_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("saved -> %s", diag_path)

    # ── pretty print ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLED EVALUATION RESULTS  (1 GT + 100 random negatives)")
    print("=" * 70)
    cols = ["experiment", "gt_coverage", "HR@5", "HR@10", "NDCG@5", "NDCG@10", "MRR", "n_users"]
    cols = [c for c in cols if c in df_summary.columns]
    print(df_summary[cols].to_string(index=False))

    print("\nDiagnostics:")
    for exp, d in diagnostics.items():
        mv = d["rank_movement"]
        ds = d.get("delta_stats", {})
        print(f"\n  [{exp}]")
        print(f"    gt_coverage  : {d['gt_coverage']:.1%}")
        print(f"    reason_dist  : {d['reason_dist']}")
        print(f"    rank_movement: improved={mv['improved']}  same={mv['same']}  worsened={mv['worsened']}")
        if ds:
            print(f"    delta nonzero: {ds.get('all_nonzero_frac', 0):.1%}")
            print(f"    GT delta mean: {ds.get('gt_item_mean_delta', 'N/A')}")
            print(f"    GT delta +frac: {ds.get('gt_item_positive_frac', 'N/A')}")

    return backbone_scores
