"""
run_apply_grounded_selector.py
------------------------------
Post-hoc Stage 2 application: adds validated_goal_concepts and raw_llm_goals
to an existing intent cache parquet WITHOUT re-calling the LLM.

Use this when you have an existing short_term_intents_llm_subset*.parquet
and want to apply the grounded selector retroactively.

Usage:
    python -m src.intent.run_apply_grounded_selector \\
        --intent-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_candidate.parquet \\
        --data-config config/data/amazon_movies_tv.yaml \\
        --backbone-candidates-path data/cache/candidate/amazon_movies_tv/sampled_candidates_k101.parquet \\
        --out-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet

Output: same schema as input, with these columns added/updated:
    raw_llm_goals             — copy of original goal_concepts (Stage 1 output)
    validated_goal_concepts   — Stage 2 grounded output
    grounding_diagnostics     — JSON string with per-record audit trail
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.intent.grounded_selector import (
    build_candidate_concept_bank,
    validate_and_select_goals,
    compute_grounding_diagnostics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-path",   required=True,
                        help="Input intent cache parquet (LLM output)")
    parser.add_argument("--data-config",   default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--backbone-candidates-path", required=True,
                        help="Backbone top-K candidates parquet "
                             "(layout A: candidate_item_ids column, or "
                             " layout B: long format with item_id column)")
    parser.add_argument("--persona-path",  default=None,
                        help="Persona parquet for persona_top_concepts. "
                             "If omitted, persona conflict suppression uses empty list.")
    parser.add_argument("--out-path",      default=None,
                        help="Output path. Default: input file with '_validated' suffix.")
    parser.add_argument("--min-activation", type=int, default=1,
                        help="Minimum candidate bank activation to accept a goal concept.")
    parser.add_argument("--scoring-mode", default="baseline_count",
                        choices=["baseline_count", "count_x_idf"],
                        help="Step 6 scoring mode. "
                             "baseline_count: sort by activation_count (default). "
                             "count_x_idf: sort by activation_count × idf_weight "
                             "(promotes specific concepts over generic ones).")
    parser.add_argument("--idf-path", default=None,
                        help="Path to concept IDF parquet (concept_id, idf_weight). "
                             "Required when --scoring-mode=count_x_idf.")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])

    # ── Load item_concepts ────────────────────────────────────────────
    logger.info("Loading item_concepts...")
    df_ic = pd.read_parquet(processed_dir / "item_concepts.parquet")
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    logger.info("item_concepts: %d items", len(item_concepts))

    # ── Load persona (optional) ───────────────────────────────────────
    persona_nodes_by_user: dict[str, list[str]] = {}
    if args.persona_path:
        logger.info("Loading persona from %s", args.persona_path)
        df_p = pd.read_parquet(args.persona_path)
        df_p_sorted = df_p.sort_values(["user_id", "weight"], ascending=[True, False])
        persona_nodes_by_user = {
            uid: grp["concept_id"].tolist()
            for uid, grp in df_p_sorted.groupby("user_id")
        }
        logger.info("Loaded persona for %d users", len(persona_nodes_by_user))
    else:
        logger.info("No --persona-path: persona conflict suppression disabled (empty prior)")

    # ── Load backbone candidates → build bank_by_key ─────────────────
    logger.info("Building candidate concept banks...")
    df_cands = pd.read_parquet(args.backbone_candidates_path)
    bank_by_key: dict[tuple[str, int], dict[str, int]] = {}

    # Detect layout:
    #   A: candidate_item_ids (list column, one row per user/target)
    #   B: item_id or candidate_item_id (long format, one row per candidate)
    if "candidate_item_ids" in df_cands.columns:
        # Layout A: one row per (user_id, target_index), candidate_item_ids is list
        for row in tqdm(df_cands.itertuples(index=False), total=len(df_cands),
                        desc="building banks (layout A)"):
            key = (str(row.user_id), int(row.target_index))
            cands = list(row.candidate_item_ids) if row.candidate_item_ids else []
            bank_by_key[key] = build_candidate_concept_bank(cands, item_concepts)
    else:
        # Layout B: long format — find the item_id column
        item_col = None
        for col in ("item_id", "candidate_item_id"):
            if col in df_cands.columns:
                item_col = col
                break
        if item_col is None:
            raise ValueError(
                "backbone-candidates-path parquet must have one of: "
                "'candidate_item_ids', 'item_id', 'candidate_item_id' column. "
                f"Found columns: {list(df_cands.columns)}"
            )
        logger.info("Layout B detected: grouping by user_id+target_index on column '%s'", item_col)
        for (uid, tidx), grp in tqdm(df_cands.groupby(["user_id", "target_index"]),
                                     desc="building banks (layout B)"):
            key = (str(uid), int(tidx))
            cands = grp[item_col].tolist()
            bank_by_key[key] = build_candidate_concept_bank(cands, item_concepts)
    logger.info("Built %d candidate banks", len(bank_by_key))

    # ── Load intent cache ─────────────────────────────────────────────
    logger.info("Loading intent cache from %s", args.intent_path)
    df = pd.read_parquet(args.intent_path)
    logger.info("Loaded %d intent records", len(df))

    # ── Apply Stage 2 per record ──────────────────────────────────────
    raw_llm_goals_col: list[list[str]] = []
    validated_col: list[list[str]] = []
    grounding_diag_col: list[str] = []

    n_improved = 0
    n_worsened = 0
    n_same = 0
    n_no_bank = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="applying Stage 2"):
        key = (str(row.user_id), int(row.target_index))
        bank = bank_by_key.get(key)

        # raw_llm_goals: preserve original goal_concepts
        # goal_concepts may be stored as list, numpy.ndarray, or None
        gc = row.goal_concepts
        raw_goals = list(gc) if gc is not None and hasattr(gc, '__iter__') and not isinstance(gc, str) else []
        raw_llm_goals_col.append(raw_goals)

        if bank is None:
            # No bank available — validated falls back to raw
            validated_col.append(list(raw_goals))
            grounding_diag_col.append(json.dumps({"skipped": True, "reason": "no_bank"}))
            n_no_bank += 1
            continue

        persona_top = persona_nodes_by_user.get(str(row.user_id), [])
        reason = str(row.deviation_reason) if hasattr(row, "deviation_reason") else "unknown"
        confidence = float(row.confidence) if hasattr(row, "confidence") else 0.5

        _sel_cfg: dict = {}
        if args.min_activation != 1:
            _sel_cfg["min_activation"] = args.min_activation
        if args.scoring_mode != "baseline_count" or args.idf_path:
            _sel_cfg["scoring"] = {
                "mode": args.scoring_mode,
                "idf_path": args.idf_path or "",
                "idf_floor": 0.1,
            }
        validated, val_diag = validate_and_select_goals(
            raw_goal_concepts=raw_goals,
            deviation_reason=reason,
            confidence=confidence,
            candidate_concept_bank=bank,
            persona_top_concepts=persona_top,
            selector_cfg=_sel_cfg if _sel_cfg else None,
        )
        validated_col.append(validated)
        grounding_diag_col.append(json.dumps(val_diag, ensure_ascii=False))

        # Track coverage changes
        n_raw_in_bank = sum(1 for c in raw_goals if bank.get(c, 0) > 0)
        n_val_in_bank = sum(1 for c in validated if bank.get(c, 0) > 0)
        if n_val_in_bank > n_raw_in_bank:
            n_improved += 1
        elif n_val_in_bank < n_raw_in_bank:
            n_worsened += 1
        else:
            n_same += 1

    # ── Write output ──────────────────────────────────────────────────
    df = df.copy()
    df["raw_llm_goals"]           = raw_llm_goals_col
    df["validated_goal_concepts"] = validated_col
    df["grounding_diagnostics"]   = grounding_diag_col

    out_path = args.out_path
    if out_path is None:
        p = Path(args.intent_path)
        out_path = str(p.parent / (p.stem + "_validated" + p.suffix))

    df.to_parquet(out_path, index=False)
    logger.info("Saved -> %s  (%d rows)", out_path, len(df))

    # ── Summary ───────────────────────────────────────────────────────
    n_total = len(df)
    logger.info("=" * 60)
    logger.info("Stage 2 Application Summary")
    logger.info("=" * 60)
    logger.info("total records       : %d", n_total)
    logger.info("no bank (skipped)   : %d (%.1f%%)", n_no_bank, 100 * n_no_bank / max(n_total, 1))
    logger.info("activation improved : %d (%.1f%%)", n_improved, 100 * n_improved / max(n_total, 1))
    logger.info("activation same     : %d (%.1f%%)", n_same, 100 * n_same / max(n_total, 1))
    logger.info("activation worsened : %d (%.1f%%)", n_worsened, 100 * n_worsened / max(n_total, 1))

    # validated empty rate per reason
    df["_val_empty"] = df["validated_goal_concepts"].apply(
        lambda x: len(x) == 0 if isinstance(x, list) else True
    )
    df["_raw_empty"] = df["raw_llm_goals"].apply(
        lambda x: len(x) == 0 if isinstance(x, list) else True
    )
    logger.info("raw goals empty     : %d (%.1f%%)",
                df["_raw_empty"].sum(), 100 * df["_raw_empty"].mean())
    logger.info("validated empty     : %d (%.1f%%)",
                df["_val_empty"].sum(), 100 * df["_val_empty"].mean())

    if "deviation_reason" in df.columns:
        logger.info("\nPer-reason validated empty rate:")
        for reason, grp in df.groupby("deviation_reason"):
            n_r = len(grp)
            n_empty = grp["_val_empty"].sum()
            raw_avg = grp["raw_llm_goals"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            val_avg = grp["validated_goal_concepts"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            logger.info(
                "  %-15s n=%4d  val_empty=%.1f%%  avg_raw=%.2f  avg_val=%.2f",
                reason, n_r, 100 * n_empty / max(n_r, 1), raw_avg, val_avg,
            )

    logger.info("=" * 60)
    logger.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
