"""
CLI entry point: backbone candidate retrieval + modulation reranking.

Usage (from project root):
    python -m src.modulation.run_rerank \
        --data-config config/data/amazon_beauty.yaml \
        --backbone-config config/backbone/sasrec.yaml \
        --modulation-config config/modulation/default.yaml \
        --experiment-config config/experiment/full_model.yaml

    # Skip backbone retrieval (reuse existing backbone_topk.parquet):
    python -m src.modulation.run_rerank \
        --data-config config/data/kuaisar.yaml \
        --experiment-config config/experiment/full_model.yaml \
        --skip-backbone
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.modulation.gate import compute_gate_strength
from src.modulation.reranker import CandidateReranker
from src.modulation.signal_builder import build_signal
from src.pipeline.online import OnlinePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",       default="config/data/amazon_beauty.yaml")
    parser.add_argument("--backbone-config",   default="config/backbone/sasrec.yaml")
    parser.add_argument("--persona-config",    default="config/persona/default.yaml")
    parser.add_argument("--intent-config",     default="config/intent/default.yaml")
    parser.add_argument("--modulation-config", default="config/modulation/default.yaml")
    parser.add_argument("--experiment-config", default="config/experiment/full_model.yaml")
    parser.add_argument(
        "--skip-backbone",
        action="store_true",
        help="Skip backbone retrieval and reuse existing backbone_topk.parquet.",
    )
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=None,
        help="Override top_k in backbone config (e.g. 500, 1000). "
             "Output files are tagged with _k<N> suffix for sweep experiments.",
    )
    args = parser.parse_args()

    data_cfg       = load_yaml(args.data_config)
    backbone_cfg   = load_yaml(args.backbone_config)
    modulation_cfg = load_yaml(args.modulation_config)
    experiment_cfg = load_yaml(args.experiment_config)

    # Apply top-K override
    if args.candidate_topk is not None:
        backbone_cfg["top_k"] = args.candidate_topk
        logger.info("top_k overridden -> %d", args.candidate_topk)

    mode = experiment_cfg.get("modulation_mode", modulation_cfg.get("mode", "graph_conditioned_full"))
    experiment_name = experiment_cfg.get("experiment_name", "experiment")
    # Tag experiment name with K value when overriding for sweep clarity
    top_k_tag = f"_k{backbone_cfg['top_k']}" if args.candidate_topk is not None else ""
    experiment_name_tagged = f"{experiment_name}{top_k_tag}"
    logger.info("mode=%s  experiment=%s  top_k=%d", mode, experiment_name_tagged, backbone_cfg["top_k"])

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir   = Path(data_cfg["paths"]["interim_dir"])
    dataset       = data_cfg.get("dataset", "amazon_beauty")

    eval_dir = Path(f"data/artifacts/eval/{dataset}")
    cand_dir = Path(f"data/cache/candidate/{dataset}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    backbone_path = cand_dir / f"backbone_topk{top_k_tag}.parquet"

    # ----------------------------------------------------------------
    # Load shared data (needed regardless of skip-backbone)
    # ----------------------------------------------------------------
    logger.info("Loading data...")
    df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")
    df_persona       = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")
    df_intents       = pd.read_parquet(f"data/cache/intent/{dataset}/short_term_intents.parquet")

    item_concepts: dict[str, list[str]] = (
        df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    persona_nodes_by_user: dict[str, list[dict]] = {
        uid: g.to_dict("records")
        for uid, g in df_persona.groupby("user_id")
    }
    intent_by_key: dict[tuple[str, int], dict] = {
        (r["user_id"], int(r["target_index"])): r
        for r in df_intents.to_dict("records")
    }

    df_snapshots = pd.read_parquet(interim_dir / "recent_context_snapshots.parquet")
    eval_snapshots = df_snapshots.loc[
        df_snapshots.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)

    reranker = CandidateReranker(modulation_cfg, item_concepts)

    # ----------------------------------------------------------------
    # Backbone retrieval (skip if backbone_topk.parquet exists and --skip-backbone)
    # ----------------------------------------------------------------
    if args.skip_backbone and backbone_path.exists():
        logger.info("--skip-backbone: loading existing %s", backbone_path)
        df_backbone = pd.read_parquet(backbone_path)

        # Build ranked results by applying modulation to saved backbone candidates
        top_k = backbone_cfg.get("top_k", 100)
        all_ranked: list[dict] = []
        signal_rows: list[dict] = []

        # Index backbone candidates: (user_id, target_index) -> list of (item_id, base_score, rank_before)
        backbone_by_key: dict[tuple[str, int], list[tuple[str, float, int]]] = {}
        for row in df_backbone.itertuples(index=False):
            key = (row.user_id, int(row.target_index))
            if key not in backbone_by_key:
                backbone_by_key[key] = []
            backbone_by_key[key].append((row.candidate_item_id, float(row.base_score), int(row.rank_before)))

        logger.info("Applying modulation to %d users...", len(eval_snapshots))
        for _, snap in tqdm(eval_snapshots.iterrows(), total=len(eval_snapshots)):
            user_id      = snap["user_id"]
            target_index = int(snap["target_index"])
            key          = (user_id, target_index)

            candidates = backbone_by_key.get(key, [])
            if not candidates:
                continue

            intent_record = intent_by_key.get(key, {})
            persona_nodes = persona_nodes_by_user.get(user_id, [])

            # Compute modulation signal (use neutral if no intent record)
            if not intent_record:
                intent_record = {
                    "user_id": user_id,
                    "target_index": target_index,
                    "goal_concepts": [],
                    "constraints_json": "{}",
                    "deviation_reason": "unknown",
                    "confidence": 0.35,
                    "ttl_steps": 1,
                    "persona_alignment_score": 0.0,
                    "is_deviation": 0,
                }
            gate_strength = compute_gate_strength(
                deviation_reason=intent_record.get("deviation_reason", "unknown"),
                confidence=float(intent_record.get("confidence", 0.35)),
                persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                gate_cfg=modulation_cfg.get("gate", {}),
            )
            signal = build_signal(intent_record, persona_nodes, gate_strength, modulation_cfg, mode=mode)
            if intent_record.get("deviation_reason", "unknown") != "unknown":
                signal_rows.append(signal.to_record())

            # Apply reranker: pass (item_id, base_score) tuples sorted by rank_before
            sorted_candidates = sorted(candidates, key=lambda x: x[2])  # sort by rank_before
            candidate_tuples = [(item_id, base_score) for item_id, base_score, _ in sorted_candidates]

            ranked = reranker.rerank(
                candidates=candidate_tuples,
                signal=signal,
                mode=mode,
            )
            for r in ranked:
                all_ranked.append(r.to_record())

    else:
        # Full backbone build + inference
        if args.skip_backbone:
            logger.warning("--skip-backbone specified but %s not found; running full backbone.", backbone_path)

        logger.info("Loading interactions and sequences...")
        df_interactions = pd.read_parquet(processed_dir / "interactions.parquet")
        df_sequences    = pd.read_parquet(interim_dir / "user_sequences.parquet")

        logger.info("Building backbone scorer...")
        from src.backbone.sasrec_wrapper import SASRecWrapper
        backbone = SASRecWrapper(df_interactions, backbone_cfg)

        pipeline = OnlinePipeline(
            backbone=backbone,
            reranker=reranker,
            persona_nodes_by_user=persona_nodes_by_user,
            intent_by_key=intent_by_key,
            backbone_cfg=backbone_cfg,
            modulation_cfg=modulation_cfg,
            mode=mode,
        )

        top_k = backbone_cfg.get("top_k", 100)
        all_ranked = []
        backbone_rows: list[dict] = []
        signal_rows   = []

        logger.info("Evaluating %d users (leave-one-out)...", len(eval_snapshots))
        for _, snap in tqdm(eval_snapshots.iterrows(), total=len(eval_snapshots)):
            user_id      = snap["user_id"]
            target_index = int(snap["target_index"])
            item_sequence = list(snap["recent_item_ids"])

            ranked = pipeline.run(user_id, item_sequence, target_index, top_k)

            for r in ranked:
                all_ranked.append(r.to_record())
                backbone_rows.append({
                    "user_id":           r.user_id,
                    "target_index":      r.target_index,
                    "candidate_item_id": r.candidate_item_id,
                    "base_score":        r.base_score,
                    "rank_before":       r.rank_before,
                })

            intent_record = intent_by_key.get((user_id, target_index), {})
            persona_nodes = persona_nodes_by_user.get(user_id, [])
            if intent_record:
                gs = compute_gate_strength(
                    deviation_reason=intent_record.get("deviation_reason", "unknown"),
                    confidence=float(intent_record.get("confidence", 0.35)),
                    persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
                    gate_cfg=modulation_cfg.get("gate", {}),
                )
                sig = build_signal(intent_record, persona_nodes, gs, modulation_cfg, mode=mode)
                signal_rows.append(sig.to_record())

        df_backbone = pd.DataFrame(backbone_rows).drop_duplicates(
            subset=["user_id", "target_index", "candidate_item_id"]
        )
        df_backbone.to_parquet(backbone_path, index=False)
        logger.info("saved -> %s  (%d rows)", backbone_path, len(df_backbone))

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    df_ranked = pd.DataFrame(all_ranked)
    ranked_path = eval_dir / f"reranked_results_{experiment_name_tagged}.parquet"
    df_ranked.to_parquet(ranked_path, index=False)
    logger.info("saved -> %s  (%d rows)", ranked_path, len(df_ranked))

    if signal_rows:
        df_signals = pd.DataFrame(signal_rows)
        sig_path = eval_dir / f"modulation_signals_{experiment_name_tagged}.parquet"
        df_signals.to_parquet(sig_path, index=False)
        logger.info("saved -> %s", sig_path)

    # ----------------------------------------------------------------
    # Quick sanity summary
    # ----------------------------------------------------------------
    logger.info("=== Rerank Summary ===")
    logger.info("users evaluated: %d", df_ranked["user_id"].nunique())
    logger.info("avg candidates per user: %.1f", len(df_ranked) / df_ranked["user_id"].nunique())

    nonzero_delta = (df_ranked["modulation_delta"] != 0).sum()
    logger.info("candidates with nonzero delta: %d / %d (%.1f%%)",
                nonzero_delta, len(df_ranked), 100 * nonzero_delta / len(df_ranked))

    rank_change = (df_ranked["rank_after"] != df_ranked["rank_before"]).sum()
    logger.info("candidates with rank change: %d (%.1f%%)",
                rank_change, 100 * rank_change / len(df_ranked))

    if "deviation_reason" in df_ranked.columns:
        reasons = df_ranked.drop_duplicates(["user_id", "target_index"])["deviation_reason"].value_counts()
        logger.info("reason distribution:")
        for r, c in reasons.items():
            logger.info("  %-15s %d", r, c)


if __name__ == "__main__":
    main()
