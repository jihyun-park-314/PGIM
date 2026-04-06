"""
CLI entry point: build short_term_intents.parquet from snapshots + persona cache.

Usage (from project root):

  Heuristic (default, full dataset):
    python -m src.intent.run_build_intent \
        --data-config config/data/amazon_movies_tv.yaml \
        --intent-config config/intent/amazon_movies_tv.yaml \
        --persona-config config/persona/default.yaml

  LLM, eval subset only (2000 users) with grounded Stage 2 validation:
    python -m src.intent.run_build_intent \
        --data-config config/data/amazon_movies_tv.yaml \
        --intent-config config/intent/amazon_movies_tv.yaml \
        --persona-config config/persona/default.yaml \
        --use-llm \
        --subset-eval-only \
        --max-users 2000 \
        --backbone-candidates-path data/cache/backbone/amazon_movies_tv/backbone_top100.parquet

  Stage 2 (grounded selector) is activated when --backbone-candidates-path is provided.
  Without it, validated_goal_concepts falls back to goal_concepts (backward-compat).

Output files:
  Heuristic  : data/cache/intent/<dataset>/short_term_intents[_v2].parquet
  LLM full   : data/cache/intent/<dataset>/short_term_intents_llm.parquet
  LLM subset : data/cache/intent/<dataset>/short_term_intents_llm_subset.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env from project root if present (sets OPENAI_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import pandas as pd
import yaml
from tqdm import tqdm

from src.intent.cache_resolver import IntentCacheResolver
from src.intent.context_extractor import ContextExtractor
from src.persona.builder import PersonaGraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",    default="config/data/amazon_beauty.yaml")
    parser.add_argument("--persona-config", default="config/persona/default.yaml")
    parser.add_argument("--intent-config",  default="config/intent/default.yaml")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="v2 mode: load source index, use persona_graphs_v2.parquet, "
             "output short_term_intents_v2.parquet",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Activate LLM-based interpreter (requires OPENAI_API_KEY env var). "
             "Output saved to short_term_intents_llm[_subset].parquet.",
    )
    parser.add_argument(
        "--subset-eval-only",
        action="store_true",
        help="Process only the last snapshot per user (eval subset). "
             "Use with --max-users to limit API cost.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Max number of users to process (applied after --subset-eval-only). "
             "Users are taken from the top of the snapshot ordering (by user_id sort). "
             "Default: all users.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=8,
        help="Number of parallel threads for LLM API calls (LLM mode only). Default: 8.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to the output filename before .parquet, "
             "e.g. '--tag onto_v1' produces short_term_intents_llm_subset_onto_v1.parquet.",
    )
    parser.add_argument(
        "--backbone-candidates-path",
        type=str,
        default=None,
        help="Path to backbone top-K candidates parquet "
             "(columns: user_id, target_index, candidate_item_ids as list or "
             "multiple rows per user/target_index with item_id column). "
             "When provided, activates Stage 2 grounded selector: "
             "validated_goal_concepts is grounded to backbone candidate concept activation. "
             "Without this, validated_goal_concepts falls back to goal_concepts.",
    )
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.intent_config) as f:
        intent_cfg = yaml.safe_load(f)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir   = Path(data_cfg["paths"]["interim_dir"])
    dataset       = data_cfg.get("dataset", "amazon_beauty")

    # ----------------------------------------------------------------
    # Load inputs
    # ----------------------------------------------------------------
    logger.info("Loading inputs...")
    df_snapshots = pd.read_parquet(interim_dir / "recent_context_snapshots.parquet")

    # v2: prefer v2 persona if available
    persona_suffix = "_v2" if args.v2 else ""
    persona_path = f"data/cache/persona/{dataset}/persona_graphs{persona_suffix}.parquet"
    if args.v2 and not Path(persona_path).exists():
        logger.warning("v2 persona not found at %s — falling back to v1", persona_path)
        persona_path = f"data/cache/persona/{dataset}/persona_graphs.parquet"
    df_persona = pd.read_parquet(persona_path)
    logger.info("Loaded persona from %s", persona_path)

    df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")

    # item_id -> list[concept_id]
    item_concepts: dict[str, list[str]] = (
        df_item_concepts.groupby("item_id")["concept_id"]
        .apply(list)
        .to_dict()
    )

    # user_id -> list of persona node dicts
    persona_nodes: dict[str, list[dict]] = {
        uid: g.to_dict("records")
        for uid, g in df_persona.groupby("user_id")
    }

    signal_types = intent_cfg.get("context", {}).get(
        "signal_concept_types", ["brand", "price_band", "item_form", "skin_type"]
    )
    top_goal = intent_cfg.get("context", {}).get("top_goal_concepts", 3)

    # v2: build source index from interactions
    source_by_key = None
    if args.v2:
        interactions_path = processed_dir / "interactions.parquet"
        if interactions_path.exists():
            logger.info("v2 mode: building source index from interactions...")
            df_interactions = pd.read_parquet(interactions_path)
            source_by_key = PersonaGraphBuilder.build_source_index(df_interactions)
        else:
            logger.warning("v2 mode: interactions.parquet not found — source context disabled")

    extractor = ContextExtractor(
        item_concepts=item_concepts,
        persona_nodes=persona_nodes,
        signal_concept_types=signal_types,
        top_goal_concepts=top_goal,
        source_by_key=source_by_key,
    )

    # ----------------------------------------------------------------
    # Stage 2: candidate concept bank (backbone candidates → per-user/target bank)
    # Activated only when --backbone-candidates-path is provided.
    # bank_by_key: {(user_id, target_index): {concept_id: activation_count}}
    # ----------------------------------------------------------------
    bank_by_key: "dict[tuple[str, int], dict[str, int]]" = {}
    if args.backbone_candidates_path:
        logger.info("Loading backbone candidates for Stage 2 grounded selector...")
        from src.intent.grounded_selector import build_candidate_concept_bank
        df_cands = pd.read_parquet(args.backbone_candidates_path)

        # Support two parquet layouts:
        #   Layout A: one row per (user_id, target_index) with candidate_item_ids as list
        #   Layout B: one row per (user_id, target_index, item_id) — long format
        if "candidate_item_ids" in df_cands.columns:
            for row in df_cands.itertuples(index=False):
                key = (str(row.user_id), int(row.target_index))
                cands = list(row.candidate_item_ids) if row.candidate_item_ids else []
                bank_by_key[key] = build_candidate_concept_bank(cands, item_concepts)
        else:
            item_col = next(
                (c for c in ("item_id", "candidate_item_id") if c in df_cands.columns), None
            )
            if item_col is None:
                logger.warning(
                    "backbone-candidates-path parquet has no recognized item column "
                    "(tried: candidate_item_ids, item_id, candidate_item_id) — Stage 2 disabled. "
                    "Found columns: %s", list(df_cands.columns)
                )
                bank_by_key = {}
            else:
                for (uid, tidx), grp in df_cands.groupby(["user_id", "target_index"]):
                    key = (str(uid), int(tidx))
                    cands = grp[item_col].tolist()
                    bank_by_key[key] = build_candidate_concept_bank(cands, item_concepts)

        logger.info(
            "Stage 2 candidate banks built: %d (user, target_index) pairs", len(bank_by_key)
        )
    else:
        logger.info(
            "No --backbone-candidates-path: Stage 2 grounded selector disabled; "
            "validated_goal_concepts will equal goal_concepts."
        )

    # ----------------------------------------------------------------
    # Subset filtering
    # ----------------------------------------------------------------
    if args.subset_eval_only:
        # Keep only the last snapshot per user (highest target_index)
        df_snapshots = df_snapshots.loc[
            df_snapshots.groupby("user_id")["target_index"].idxmax()
        ].reset_index(drop=True)
        logger.info("subset_eval_only: reduced to %d snapshots (1 per user)", len(df_snapshots))

    if args.max_users is not None:
        # Deterministic: sort by user_id.
        # Keep a larger pool (2× target) so that failed LLM calls can be replaced
        # by the next user in line, guaranteeing exactly max_users successful records.
        pool_size = args.max_users * 2 if args.use_llm else args.max_users
        unique_users = sorted(df_snapshots["user_id"].unique())[:pool_size]
        df_snapshots = df_snapshots[df_snapshots["user_id"].isin(unique_users)].reset_index(drop=True)
        logger.info("max_users=%d (pool=%d): %d snapshots loaded",
                    args.max_users, pool_size, len(df_snapshots))

    # ----------------------------------------------------------------
    # LLM client setup
    # ----------------------------------------------------------------
    openai_client = None
    if args.use_llm:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "--use-llm requires OPENAI_API_KEY environment variable to be set."
            )
        try:
            import openai
            openai_client = openai.OpenAI(api_key=api_key)
            llm_model = intent_cfg.get("llm", {}).get("model", "gpt-4o-mini")
            logger.info("LLM client ready (model=%s)", llm_model)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Force use_llm in config for cache_resolver
        if "llm" not in intent_cfg:
            intent_cfg["llm"] = {}
        intent_cfg["llm"]["use_llm"] = True

    resolver = IntentCacheResolver(intent_cfg, use_llm=args.use_llm)

    # ----------------------------------------------------------------
    # Process snapshots
    # ----------------------------------------------------------------
    import time
    from src.intent.llm_interpreter import _build_candidate_concepts, interpret_with_llm
    from src.intent.heuristic_interpreter import interpret as heuristic_interpret
    from src.intent.parser import parse_intent

    logger.info(
        "Processing %d snapshots (interpreter=%s, workers=%s)...",
        len(df_snapshots),
        "llm" if args.use_llm else "heuristic",
        args.llm_workers if args.use_llm else "1 (heuristic)",
    )

    # Pre-extract all contexts (fast, CPU-only)
    rows_list = [row.to_dict() for _, row in df_snapshots.iterrows()]
    contexts = [extractor.extract(r) for r in rows_list]
    candidate_counts: list[int] = [len(_build_candidate_concepts(c, intent_cfg)) for c in contexts]

    n_llm_ok = 0
    n_fallback = 0
    latencies: list[float] = []

    if args.use_llm and args.llm_workers > 1:
        # Parallel LLM calls — each worker calls interpret_with_llm directly.
        # Results collected in submission order via index.
        raw_results: list[tuple[int, dict, float]] = []  # (idx, record, elapsed)

        def _call_one(idx: int, ctx) -> tuple[int, dict, float]:
            t0 = time.perf_counter()
            bank = bank_by_key.get((ctx.user_id, ctx.target_index))
            raw = interpret_with_llm(
                ctx, "", intent_cfg, openai_client,
                candidate_concept_bank=bank,
            )
            elapsed = time.perf_counter() - t0
            # Preserve source_mode from interpret_with_llm:
            # "llm" on success, "llm_fallback" on exception.
            actual_source = raw.get("source_mode", "llm")
            record = parse_intent(raw, ctx.user_id, ctx.target_index, actual_source)
            return idx, record, elapsed

        with ThreadPoolExecutor(max_workers=args.llm_workers) as pool:
            futures = {pool.submit(_call_one, i, ctx): i for i, ctx in enumerate(contexts)}
            with tqdm(total=len(contexts)) as pbar:
                for future in as_completed(futures):
                    try:
                        idx, record, elapsed = future.result()
                        raw_results.append((idx, record, elapsed))
                    except Exception as exc:
                        logger.error("Worker thread raised unexpected exception: %s", exc)
                    pbar.update(1)

        # Sort by original index, then filter out fallbacks
        raw_results.sort(key=lambda x: x[0])
        accepted = [(idx, r, e) for idx, r, e in raw_results if r.get("source_mode") == "llm"]
        fallbacks = [(idx, r, e) for idx, r, e in raw_results if r.get("source_mode") != "llm"]
        n_fallback = len(fallbacks)

        # Trim to max_users if we got more successes than needed
        target = args.max_users if args.max_users is not None else len(accepted)
        accepted = accepted[:target]

        intent_records = [r for _, r, _ in accepted]
        accepted_idxs = {idx for idx, _, _ in accepted}
        contexts = [ctx for i, ctx in enumerate(contexts) if i in accepted_idxs]

        for _, record, elapsed in accepted:
            n_llm_ok += 1
            latencies.append(elapsed)

        if len(intent_records) < target:
            logger.warning(
                "Only %d/%d succeeded — pool exhausted. "
                "Increase pool or check quota.", len(intent_records), target
            )

    else:
        # Sequential path (heuristic or llm_workers=1)
        intent_records = []
        accepted_contexts = []
        target = args.max_users if (args.use_llm and args.max_users is not None) else len(contexts)

        with tqdm(total=target) as pbar:
            for ctx in contexts:
                if args.use_llm and len(intent_records) >= target:
                    break
                t0 = time.perf_counter()
                if args.use_llm:
                    bank = bank_by_key.get((ctx.user_id, ctx.target_index))
                    raw = interpret_with_llm(
                        ctx, "", intent_cfg, openai_client,
                        candidate_concept_bank=bank,
                    )
                else:
                    raw = None
                elapsed = time.perf_counter() - t0

                if args.use_llm:
                    actual_source = raw.get("source_mode", "llm_fallback")
                    if actual_source == "llm_fallback":
                        # LLM failed for this user — skip, try next in pool
                        n_fallback += 1
                        logger.debug("Skipping user=%s (llm_fallback)", ctx.user_id)
                        continue
                    record = parse_intent(raw, ctx.user_id, ctx.target_index, actual_source)
                    n_llm_ok += 1
                    latencies.append(elapsed)
                else:
                    record = resolver.resolve(ctx, openai_client=openai_client)

                intent_records.append(record)
                accepted_contexts.append(ctx)
                pbar.update(1)

        if args.use_llm and len(intent_records) < target:
            logger.warning(
                "Only %d/%d succeeded — pool exhausted. "
                "Increase pool or check quota.", len(intent_records), target
            )
        contexts = accepted_contexts  # align feature_records with accepted contexts

    feature_records = [ctx.to_record() for ctx in contexts]

    # ----------------------------------------------------------------
    # Determine output paths
    # ----------------------------------------------------------------
    intent_cache_dir = Path(f"data/cache/intent/{dataset}")
    intent_cache_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    if args.use_llm:
        suffix = "_llm_subset" if (args.subset_eval_only or args.max_users) else "_llm"
    else:
        suffix = "_v2" if args.v2 else ""

    tag_str = f"_{args.tag}" if args.tag else ""
    intent_path = intent_cache_dir / f"short_term_intents{suffix}{tag_str}.parquet"

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    df_intents = pd.DataFrame(intent_records)
    # Ensure list columns are stored as object (list-of-str), not inferred as scalar/int
    _list_cols = (
        "goal_concepts", "evidence_item_ids", "raw_llm_goals", "validated_goal_concepts",
        "evidence_recent_concepts", "evidence_persona_concepts", "pre_grounding_goal_text",
        "removed_non_semantic_goals",
        # PR3-1 list fields
        "context_goals", "evidence_sources", "support_items",
    )
    for _list_col in _list_cols:
        if _list_col in df_intents.columns:
            df_intents[_list_col] = df_intents[_list_col].apply(
                lambda x: list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [str(x)])
            )
    # JSON-string columns: store nested dicts as JSON strings to avoid parquet issues
    import json as _json
    _json_cols = ("grounding_diagnostics", "contrast_with_persona", "temporal_cues", "token_usage")
    for _jcol in _json_cols:
        if _jcol in df_intents.columns:
            df_intents[_jcol] = df_intents[_jcol].apply(
                lambda x: x if isinstance(x, str) else (
                    _json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else "{}"
                )
            )
    df_intents.to_parquet(intent_path, index=False)
    logger.info("saved -> %s  (%d rows)", intent_path, len(df_intents))

    # Only save intent_features for full runs (subset is just for eval comparison)
    if not (args.subset_eval_only or args.max_users):
        df_features = pd.DataFrame(feature_records)
        features_path = interim_dir / "intent_features.parquet"
        df_features.to_parquet(features_path, index=False)
        logger.info("saved -> %s", features_path)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    n_total = len(df_intents)
    sep = "=" * 60

    logger.info(sep)
    logger.info("=== Intent Build Summary ===")
    logger.info(sep)
    logger.info("total rows          : %d", n_total)

    # source_mode distribution
    if "source_mode" in df_intents.columns:
        logger.info("source_mode distribution:")
        for src, cnt in df_intents["source_mode"].value_counts().items():
            logger.info("  %-20s %d (%.1f%%)", src, cnt, 100 * cnt / n_total)
    if args.use_llm:
        logger.info(
            "  llm_ok=%-6d  fallback=%-6d  fallback_rate=%.1f%%",
            n_llm_ok, n_fallback, 100 * n_fallback / max(1, n_total),
        )

    # parser_status
    logger.info("parser_status distribution:")
    for s, c in df_intents["parser_status"].value_counts().items():
        logger.info("  %-12s %d (%.1f%%)", s, c, 100 * c / n_total)

    # goal_concepts / validated_goal_concepts empty rates
    if "goal_concepts" in df_intents.columns:
        n_empty = df_intents["goal_concepts"].apply(
            lambda x: len(x) == 0 if isinstance(x, list) else True
        ).sum()
        logger.info(
            "goal_concepts empty : %d (%.1f%%)", n_empty, 100 * n_empty / n_total
        )
    if "validated_goal_concepts" in df_intents.columns:
        n_val_empty = df_intents["validated_goal_concepts"].apply(
            lambda x: len(x) == 0 if isinstance(x, list) else True
        ).sum()
        logger.info(
            "validated_goals empty: %d (%.1f%%) [Stage 2 activation gate]",
            n_val_empty, 100 * n_val_empty / n_total,
        )
        if bank_by_key:
            # Compare raw vs validated sizes
            raw_sizes = df_intents["raw_llm_goals"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            ) if "raw_llm_goals" in df_intents.columns else None
            val_sizes = df_intents["validated_goal_concepts"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            if raw_sizes is not None:
                logger.info(
                    "avg goal count raw=%.2f  validated=%.2f  (reduction=%.2f)",
                    raw_sizes.mean(), val_sizes.mean(),
                    raw_sizes.mean() - val_sizes.mean(),
                )

    # deviation_reason distribution
    logger.info("deviation_reason distribution:")
    for reason, cnt in df_intents["deviation_reason"].value_counts().items():
        logger.info("  %-15s %d (%.1f%%)", reason, cnt, 100 * cnt / n_total)
    logger.info("is_deviation rate   : %.1f%%", 100 * df_intents["is_deviation"].mean())
    logger.info("avg confidence      : %.3f", df_intents["confidence"].mean())
    logger.info("avg alignment score : %.3f", df_intents["persona_alignment_score"].mean())

    # candidate_concepts stats
    if candidate_counts:
        import statistics
        logger.info(
            "candidate_concepts  : avg=%.1f  min=%d  max=%d  median=%.0f",
            sum(candidate_counts) / len(candidate_counts),
            min(candidate_counts),
            max(candidate_counts),
            statistics.median(candidate_counts),
        )

    # API latency (LLM only)
    if latencies:
        import statistics
        logger.info(
            "API latency (s)     : avg=%.3f  p50=%.3f  p95=%.3f  max=%.3f",
            sum(latencies) / len(latencies),
            statistics.median(latencies),
            sorted(latencies)[int(len(latencies) * 0.95)],
            max(latencies),
        )
        logger.info(
            "estimated total API time: %.1f s  (%.1f min)",
            sum(latencies), sum(latencies) / 60,
        )

    if "current_source" in df_intents.columns:
        logger.info("current_source distribution:")
        for src, cnt in df_intents["current_source"].value_counts().items():
            logger.info("  %-10s %d (%.1f%%)", src, cnt, 100 * cnt / n_total)

    logger.info(sep)
    logger.info("Output: %s", intent_path)


if __name__ == "__main__":
    main()
