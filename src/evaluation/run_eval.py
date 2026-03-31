"""
CLI entry point for evaluation + diagnostics.

Usage:
    # Standard ablation eval (full-ranking)
    python -m src.evaluation.run_eval \
        --data-config config/data/kuaisar.yaml \
        --evaluation-config config/evaluation/default.yaml

    # Source-split diagnostic (rec vs search targets)
    python -m src.evaluation.run_eval \
        --data-config config/data/kuaisar.yaml \
        --evaluation-config config/evaluation/default.yaml \
        --source-split \
        --experiment-names ablation_backbone_only full_model

    # Build sampled candidate sets (run once per dataset)
    python -m src.evaluation.run_eval \
        --data-config config/data/kuaisar.yaml \
        --evaluation-config config/evaluation/default.yaml \
        --build-sampled-candidates

    # Sampled evaluation (reranking-level, GT always in candidates)
    python -m src.evaluation.run_eval \
        --data-config config/data/kuaisar.yaml \
        --evaluation-config config/evaluation/default.yaml \
        --backbone-config config/backbone/kuaisar_sasrec.yaml \
        --modulation-config config/modulation/default.yaml \
        --sampled-eval \
        --experiment-names ablation_backbone_only ablation_persona_only \
                           ablation_intent_only full_model
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.ablation_runner import run_ablation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# experiment name -> modulation_mode mapping (matches experiment config files)
_EXPERIMENT_MODES = {
    "ablation_backbone_only":  "backbone_only",
    "ablation_persona_only":   "persona_only_rerank",
    "ablation_intent_only":    "intent_only_rerank",
    "full_model":              "graph_conditioned_full",
}


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",       default="config/data/amazon_beauty.yaml")
    parser.add_argument("--evaluation-config", default="config/evaluation/default.yaml")
    parser.add_argument("--backbone-config",   default="config/backbone/sasrec.yaml",
                        help="Backbone config (needed for --sampled-eval).")
    parser.add_argument("--modulation-config", default="config/modulation/default.yaml",
                        help="Modulation config (needed for --sampled-eval).")
    parser.add_argument("--experiment-names",  nargs="+",
                        default=["ablation_backbone_only", "ablation_persona_only",
                                 "ablation_intent_only", "full_model"])
    parser.add_argument("--eval-dir", default=None,
                        help="defaults to data/artifacts/eval/<dataset>")
    parser.add_argument("--source-split", action="store_true",
                        help="Run source-split diagnostic (rec vs search targets).")
    parser.add_argument("--build-sampled-candidates", action="store_true",
                        help="Build 1 GT + 100 random negative candidates per user. "
                             "Fixed seed; all experiments share the same negatives.")
    parser.add_argument("--sampled-eval", action="store_true",
                        help="Sampled evaluation: GT guaranteed in candidates. "
                             "Measures reranking quality isolated from retrieval difficulty.")
    parser.add_argument("--n-negatives", type=int, default=100,
                        help="Number of random negatives per user (default 100 → 101-item sets).")
    parser.add_argument("--sampled-seed", type=int, default=42,
                        help="Random seed for negative sampling (default 42).")
    parser.add_argument("--per-reason", action="store_true",
                        help="Reason-sliced diagnostic on existing sampled_reranked_*.parquet files.")
    parser.add_argument("--sampled-source-split", action="store_true",
                        help="Source-split diagnostic on existing sampled_reranked_*.parquet files. "
                             "Slices by current_source/source_shift from v2 intent. "
                             "Use --v2 to load v2 intent/persona. "
                             "Use --eval-dir to specify which ranked_dir to read from.")
    parser.add_argument("--tuning-configs", nargs="+", default=None,
                        help="Paths to modulation config YAMLs for tuning sweep. "
                             "Each config is run as a separate sampled eval variant.")
    parser.add_argument("--v2", action="store_true",
                        help="Use v2 persona/intent files: persona_graphs_v2.parquet, "
                             "short_term_intents_v2.parquet (if available).")
    parser.add_argument("--intent-debug", action="store_true",
                        help="Run intent signal diagnostic on existing sampled_reranked_*.parquet files. "
                             "Requires --data-config, --modulation-config. "
                             "Outputs intent_debug_summary.json and related CSVs.")
    parser.add_argument("--backbone-scores-cache", default=None,
                        help="Path to backbone scores parquet cache. "
                             "If the file exists, backbone scoring is skipped entirely. "
                             "Defaults to data/cache/backbone/<dataset>/backbone_scores.parquet.")
    parser.add_argument("--intent-cache-path", default=None,
                        help="Explicit path to short_term_intents parquet to use for eval. "
                             "Overrides the default path derived from --v2 flag. "
                             "Use this to compare heuristic vs LLM intent without file renaming. "
                             "Example: data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset.parquet")
    parser.add_argument("--restrict-to-intent-cache", action="store_true",
                        help="Restrict evaluation to keys present in the intent cache. "
                             "Disables _NEUTRAL_INTENT fallback: missing intent keys are excluded "
                             "rather than silently filled with unknown. "
                             "Required for fair heuristic vs LLM comparison on a subset.")
    args = parser.parse_args()

    data_cfg = _load_yaml(args.data_config)
    eval_cfg = _load_yaml(args.evaluation_config)

    k_values: list[int] = eval_cfg.get("k_values", [5, 10, 20])
    dataset       = data_cfg.get("dataset", "amazon_beauty")
    eval_dir      = Path(args.eval_dir) if args.eval_dir else Path(f"data/artifacts/eval/{dataset}")
    interim_dir   = Path(data_cfg["paths"]["interim_dir"])
    processed_dir = Path(data_cfg["paths"]["processed_dir"])

    df_sequences = pd.read_parquet(interim_dir / "user_sequences.parquet")
    df_snaps     = pd.read_parquet(interim_dir / "recent_context_snapshots.parquet")

    # ── build sampled candidates ─────────────────────────────────────
    if args.build_sampled_candidates:
        from src.evaluation.sampled_eval import build_sampled_candidates

        df_interactions = pd.read_parquet(processed_dir / "interactions.parquet")
        all_item_ids = sorted(df_interactions["item_id"].unique().tolist())

        eval_snaps_last = df_snaps.loc[
            df_snaps.groupby("user_id")["target_index"].idxmax()
        ].reset_index(drop=True)

        cand_dir = Path(f"data/cache/candidate/{dataset}")
        out_path = cand_dir / f"sampled_candidates_k{args.n_negatives + 1}.parquet"

        build_sampled_candidates(
            df_sequences=df_sequences,
            eval_snaps=eval_snaps_last,
            all_item_ids=all_item_ids,
            n_negatives=args.n_negatives,
            random_seed=args.sampled_seed,
            out_path=out_path,
        )
        return

    # ── sampled evaluation ───────────────────────────────────────────
    if args.sampled_eval:
        backbone_cfg   = _load_yaml(args.backbone_config)
        modulation_cfg = _load_yaml(args.modulation_config)

        cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k{args.n_negatives + 1}.parquet")
        if not cand_path.exists():
            logger.error(
                "Sampled candidates not found: %s\n"
                "Run with --build-sampled-candidates first.", cand_path,
            )
            return

        df_interactions  = pd.read_parquet(processed_dir / "interactions.parquet")
        df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")

        # v2: prefer v2 persona/intent if --v2 flag and files exist
        persona_suffix = "_v2" if args.v2 else ""
        intent_suffix  = "_v2" if args.v2 else ""
        persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs{persona_suffix}.parquet")
        if args.intent_cache_path:
            intent_path = Path(args.intent_cache_path)
        else:
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents{intent_suffix}.parquet")
        if args.v2 and not persona_path.exists():
            logger.warning("v2 persona not found (%s), falling back to v1", persona_path)
            persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs.parquet")
        if not intent_path.exists():
            if args.intent_cache_path:
                logger.error("--intent-cache-path not found: %s", intent_path)
                return
            logger.warning("intent not found (%s), falling back to v1", intent_path)
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents.parquet")
        df_persona  = pd.read_parquet(persona_path)
        df_intents  = pd.read_parquet(intent_path)
        logger.info("persona: %s  intent: %s", persona_path.name, intent_path.name)
        if "source_mode" in df_intents.columns:
            sm = df_intents["source_mode"].value_counts().to_dict()
            logger.info("intent source_mode: %s", sm)

        item_concepts: dict[str, list[str]] = (
            df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
        )
        persona_nodes_by_user: dict[str, list[dict]] = {
            uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
        }
        intent_by_key: dict[tuple[str, int], dict] = {
            (r["user_id"], int(r["target_index"])): r
            for r in df_intents.to_dict("records")
        }

        # Build experiment_name -> mode mapping for requested experiments
        experiment_modes: dict[str, str] = {}
        for name in args.experiment_names:
            mode = _EXPERIMENT_MODES.get(name)
            if mode is None:
                # Try to infer from experiment config file
                exp_cfg_path = Path(f"config/experiment/{name}.yaml")
                if exp_cfg_path.exists():
                    exp_cfg = _load_yaml(str(exp_cfg_path))
                    mode = exp_cfg.get("modulation_mode", "graph_conditioned_full")
                else:
                    logger.warning("Unknown experiment '%s', defaulting to graph_conditioned_full", name)
                    mode = "graph_conditioned_full"
            experiment_modes[name] = mode

        backbone_scores_cache_path = (
            Path(args.backbone_scores_cache)
            if args.backbone_scores_cache
            else Path(f"data/cache/backbone/{dataset}/backbone_scores.parquet")
        )

        from src.evaluation.sampled_eval import run_sampled_eval
        run_sampled_eval(
            experiment_modes=experiment_modes,
            eval_dir=eval_dir,
            df_sequences=df_sequences,
            df_snaps=df_snaps,
            df_interactions=df_interactions,
            backbone_cfg=backbone_cfg,
            modulation_cfg=modulation_cfg,
            item_concepts=item_concepts,
            persona_nodes_by_user=persona_nodes_by_user,
            intent_by_key=intent_by_key,
            k_values=k_values,
            cand_path=cand_path,
            out_dir=eval_dir,
            backbone_scores_cache_path=backbone_scores_cache_path,
            restrict_to_intent_keys=args.restrict_to_intent_cache,
        )
        return

    # ── per-reason diagnostic (reads existing parquets) ─────────────
    if args.per_reason:
        from src.evaluation.per_reason_eval import run_per_reason_eval
        run_per_reason_eval(
            experiment_names=args.experiment_names,
            eval_dir=eval_dir,
            k_values=k_values,
            out_dir=eval_dir,
        )
        return

    # ── intent debug diagnostic ───────────────────────────────────────
    if args.intent_debug:
        modulation_cfg = _load_yaml(args.modulation_config)

        df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")
        item_concepts: dict[str, list[str]] = (
            df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
        )

        intent_suffix = "_v2" if args.v2 else ""
        intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents{intent_suffix}.parquet")
        if args.v2 and not intent_path.exists():
            logger.warning("v2 intent not found (%s), falling back to v1", intent_path)
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents.parquet")
        df_intents = pd.read_parquet(intent_path)
        intent_by_key: dict[tuple[str, int], dict] = {
            (r["user_id"], int(r["target_index"])): r
            for r in df_intents.to_dict("records")
        }
        logger.info("intent_debug: loaded %d intent records from %s", len(intent_by_key), intent_path.name)

        from src.evaluation.intent_debug import run_intent_debug
        run_intent_debug(
            eval_dir=eval_dir,
            dataset=dataset,
            df_snaps=df_snaps,
            item_concepts=item_concepts,
            intent_by_key=intent_by_key,
            modulation_cfg=modulation_cfg,
            k_values=k_values,
            experiment_names=args.experiment_names,
        )
        return

    # ── sampled source-split diagnostic (reads existing parquets) ───
    if args.sampled_source_split:
        from src.evaluation.source_split_sampled_eval import run_source_split_sampled_eval

        # ranked_dir: where sampled_reranked_*.parquet files live
        # If --eval-dir explicitly set, use that; otherwise default eval_dir
        ranked_dir = eval_dir

        # Load v2 intent (required for current_source/source_shift_flag)
        intent_suffix  = "_v2" if args.v2 else ""
        persona_suffix = "_v2" if args.v2 else ""
        intent_path  = Path(f"data/cache/intent/{dataset}/short_term_intents{intent_suffix}.parquet")
        persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs{persona_suffix}.parquet")
        if args.v2 and not intent_path.exists():
            logger.warning("v2 intent not found (%s) — falling back to v1", intent_path)
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents.parquet")
        if args.v2 and not persona_path.exists():
            logger.warning("v2 persona not found (%s) — falling back to v1", persona_path)
            persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs.parquet")

        df_intents_ss = pd.read_parquet(intent_path)
        df_persona_ss = pd.read_parquet(persona_path)
        logger.info("source-split: intent=%s  persona=%s", intent_path.name, persona_path.name)

        cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k{args.n_negatives + 1}.parquet")
        if not cand_path.exists():
            logger.error("Sampled candidates not found: %s", cand_path)
            return
        df_cands = pd.read_parquet(cand_path)

        # filter intent to last snapshot per user (eval targets only)
        eval_snaps = df_snaps.loc[
            df_snaps.groupby("user_id")["target_index"].idxmax()
        ].reset_index(drop=True)
        eval_snap_keys = set(
            zip(eval_snaps["user_id"].astype(str), eval_snaps["target_index"].astype(int))
        )
        df_intents_ss = df_intents_ss[
            df_intents_ss.apply(
                lambda r: (str(r["user_id"]), int(r["target_index"])) in eval_snap_keys, axis=1
            )
        ].reset_index(drop=True)
        logger.info("Filtered intents to eval targets: %d rows", len(df_intents_ss))

        run_source_split_sampled_eval(
            experiment_names=args.experiment_names,
            ranked_dir=ranked_dir,
            df_intents=df_intents_ss,
            df_cands=df_cands,
            df_persona=df_persona_ss,
            k_values=k_values,
            out_dir=eval_dir,
        )
        return

    # ── tuning sweep (multiple modulation configs) ────────────────────
    if args.tuning_configs:
        backbone_cfg   = _load_yaml(args.backbone_config)

        cand_path = Path(f"data/cache/candidate/{dataset}/sampled_candidates_k{args.n_negatives + 1}.parquet")
        if not cand_path.exists():
            logger.error(
                "Sampled candidates not found: %s\n"
                "Run with --build-sampled-candidates first.", cand_path,
            )
            return

        df_interactions  = pd.read_parquet(processed_dir / "interactions.parquet")
        df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")

        persona_suffix = "_v2" if args.v2 else ""
        intent_suffix  = "_v2" if args.v2 else ""
        persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs{persona_suffix}.parquet")
        if args.intent_cache_path:
            intent_path = Path(args.intent_cache_path)
        else:
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents{intent_suffix}.parquet")
        if args.v2 and not persona_path.exists():
            logger.warning("v2 persona not found (%s), falling back to v1", persona_path)
            persona_path = Path(f"data/cache/persona/{dataset}/persona_graphs.parquet")
        if not intent_path.exists():
            if args.intent_cache_path:
                logger.error("--intent-cache-path not found: %s", intent_path)
                return
            logger.warning("intent not found (%s), falling back to v1", intent_path)
            intent_path = Path(f"data/cache/intent/{dataset}/short_term_intents.parquet")
        df_persona  = pd.read_parquet(persona_path)
        df_intents  = pd.read_parquet(intent_path)
        logger.info("persona: %s  intent: %s", persona_path.name, intent_path.name)
        if "source_mode" in df_intents.columns:
            sm = df_intents["source_mode"].value_counts().to_dict()
            logger.info("intent source_mode: %s", sm)

        item_concepts: dict[str, list[str]] = (
            df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
        )
        persona_nodes_by_user: dict[str, list[dict]] = {
            uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")
        }
        intent_by_key: dict[tuple[str, int], dict] = {
            (r["user_id"], int(r["target_index"])): r
            for r in df_intents.to_dict("records")
        }

        # Tuning experiments: intent_only + persona_only + full_model per config
        tune_exp_modes = {
            "ablation_persona_only":  "persona_only_rerank",
            "ablation_intent_only":   "intent_only_rerank",
            "full_model":             "graph_conditioned_full",
        }

        tune_backbone_cache = (
            Path(args.backbone_scores_cache)
            if args.backbone_scores_cache
            else Path(f"data/cache/backbone/{dataset}/backbone_scores.parquet")
        )

        from src.evaluation.sampled_eval import run_sampled_eval
        from src.evaluation.tuning_sweep import aggregate_sweep
        cached_backbone_scores = None
        cfg_names = []
        for cfg_path in args.tuning_configs:
            mod_cfg = _load_yaml(cfg_path)
            cfg_name = Path(cfg_path).stem  # e.g. "tuning_B"
            cfg_names.append(cfg_name)
            tune_out_dir = eval_dir / f"tuning_{cfg_name}"
            logger.info("=== Tuning config: %s ===", cfg_path)

            # ── log key config values for verification ──────────────
            _v2 = mod_cfg.get("v2_blend", {})
            _rp = mod_cfg.get("reason_policy", {})
            _delta = mod_cfg.get("delta", {})
            def _pw(reason):
                b = _v2.get(reason, {})
                return f"p={b.get('persona_weight','?')} i={b.get('intent_weight','?')}"
            def _bs(reason):
                return _rp.get(reason, {}).get("boost_scale", "?")
            logger.info(
                "  [%s] delta: max_boost=%.2f max_suppress=%.2f",
                cfg_name,
                _delta.get("max_boost", float("nan")),
                _delta.get("max_suppress", float("nan")),
            )
            logger.info(
                "  [%s] v2_blend:  aligned(%s)  exploration(%s)  task_focus(%s)  budget_shift(%s)  unknown(%s)",
                cfg_name, _pw("aligned"), _pw("exploration"), _pw("task_focus"), _pw("budget_shift"), _pw("unknown"),
            )
            logger.info(
                "  [%s] boost_scale: aligned=%s  exploration=%s  task_focus=%s  budget_shift=%s  unknown=%s",
                cfg_name, _bs("aligned"), _bs("exploration"), _bs("task_focus"),
                _bs("budget_shift"), _bs("unknown"),
            )
            logger.info("  [%s] out_dir: %s", cfg_name, tune_out_dir)
            cached_backbone_scores = run_sampled_eval(
                experiment_modes=tune_exp_modes,
                eval_dir=tune_out_dir,
                df_sequences=df_sequences,
                df_snaps=df_snaps,
                df_interactions=df_interactions,
                backbone_cfg=backbone_cfg,
                modulation_cfg=mod_cfg,
                item_concepts=item_concepts,
                persona_nodes_by_user=persona_nodes_by_user,
                intent_by_key=intent_by_key,
                k_values=k_values,
                cand_path=cand_path,
                out_dir=tune_out_dir,
                precomputed_backbone_scores=cached_backbone_scores,
                backbone_scores_cache_path=tune_backbone_cache,
            )

        # ── aggregate sweep results into summary tables ───────────
        aggregate_sweep(
            cfg_names=cfg_names,
            eval_dir=eval_dir,
            k_values=k_values,
            out_dir=eval_dir,
        )
        return

    # ── source-split diagnostic ──────────────────────────────────────
    if args.source_split:
        inter_path = processed_dir / "interactions.parquet"
        if not inter_path.exists():
            logger.error("interactions.parquet not found at %s", inter_path)
            return
        df_interactions = pd.read_parquet(inter_path)
        if "source_service" not in df_interactions.columns:
            logger.error("interactions.parquet has no source_service column")
            return

        from src.evaluation.source_diagnostic import run_source_diagnostic
        run_source_diagnostic(
            experiment_names=args.experiment_names,
            eval_dir=eval_dir,
            df_sequences=df_sequences,
            df_snaps=df_snaps,
            df_interactions=df_interactions,
            k_values=k_values,
            out_dir=eval_dir,
        )
        return

    # ── standard full-ranking ablation eval ──────────────────────────
    run_ablation(
        experiment_names=args.experiment_names,
        eval_dir=eval_dir,
        df_sequences=df_sequences,
        df_snaps=df_snaps,
        k_values=k_values,
        out_dir=eval_dir,
    )

    summary_path = eval_dir / "metrics_summary.csv"
    diag_path    = eval_dir / "diagnostic_summary.json"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))

    if diag_path.exists():
        diag = json.load(open(diag_path))
        print("\n" + "="*60)
        print("DIAGNOSTICS")
        print("="*60)
        for exp, d in diag.items():
            print(f"\n[{exp}]")
            print(f"  gt_coverage     : {d['gt_coverage']:.1%}")
            print(f"  reason_dist     : {d['reason_distribution']}")
            if d.get("reason_imbalance_warning"):
                print(f"  ⚠ IMBALANCE     : dominant={d['dominant_reason_fraction']:.1%}")
            mv = d.get("rank_movement", {})
            print(f"  rank_movement   : improved={mv.get('improved',0)}  "
                  f"same={mv.get('same',0)}  worsened={mv.get('worsened',0)}")
            ds = d.get("delta_stats", {})
            if ds:
                print(f"  delta nonzero   : {ds.get('all_nonzero_frac',0):.1%}")
                print(f"  GT delta mean   : {ds.get('gt_item_mean_delta','N/A')}")
                print(f"  GT delta +frac  : {ds.get('gt_item_positive_frac','N/A')}")
            gs = d.get("gate_stats", {})
            if gs:
                print(f"  gate_strength   : mean={gs['mean']:.3f}  std={gs['std']:.3f}")


if __name__ == "__main__":
    main()
