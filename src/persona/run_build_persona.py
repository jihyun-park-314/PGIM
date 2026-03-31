"""
CLI entry point: ontology grounding + persona graph build.

Usage (from project root):
    # Amazon Beauty
    python -m src.persona.run_build_persona \
        --data-config config/data/amazon_beauty.yaml \
        --ontology-config config/ontology/category_v1.yaml \
        --persona-config config/persona/default.yaml

    # KuaiSAR
    python -m src.persona.run_build_persona \
        --data-config config/data/kuaisar.yaml \
        --ontology-config config/ontology/kuaisar_v1.yaml \
        --persona-config config/persona/default.yaml
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.ontology.store import OntologyStore
from src.persona.builder import PersonaGraphBuilder
from src.persona.cache_io import PersonaCacheIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="config/data/amazon_beauty.yaml")
    parser.add_argument("--ontology-config", default="config/ontology/category_v1.yaml")
    parser.add_argument("--persona-config", default="config/persona/default.yaml")
    parser.add_argument(
        "--skip-grounding",
        action="store_true",
        help="skip grounding step (use existing item_concepts.parquet)",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="build v2 persona with source-aware profile (source_rec_frac/source_search_frac). "
             "Requires interactions.parquet with source_service column.",
    )
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.ontology_config) as f:
        ontology_cfg = yaml.safe_load(f)
    with open(args.persona_config) as f:
        persona_cfg = yaml.safe_load(f)

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir = Path(data_cfg["paths"]["interim_dir"])

    # ------------------------------------------------------------------
    # Step 1: ontology grounding (dispatch by dataset)
    # ------------------------------------------------------------------
    if not args.skip_grounding:
        logger.info("=== Step 1: ontology grounding ===")
        dataset = data_cfg.get("dataset", "amazon_beauty")
        if dataset == "kuaisar":
            from src.ontology.kuaisar_grounding import run as run_grounding
        else:
            from src.ontology.grounding import run as run_grounding
        df_item_concepts, df_nodes = run_grounding(data_cfg, ontology_cfg)
    else:
        logger.info("=== Step 1: skipping grounding, loading existing files ===")
        df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")
        df_nodes = pd.read_parquet(interim_dir / "ontology_nodes.parquet")

    store = OntologyStore.from_dataframe(df_nodes)
    logger.info("OntologyStore ready: %d concepts", len(store))

    # ------------------------------------------------------------------
    # Step 2: persona graph build
    # ------------------------------------------------------------------
    logger.info("=== Step 2: building persona graphs ===")
    df_sequences = pd.read_parquet(interim_dir / "user_sequences.parquet")

    # v2: load source index from interactions
    source_by_key = None
    if args.v2:
        interactions_path = processed_dir / "interactions.parquet"
        if interactions_path.exists():
            logger.info("v2 mode: loading interactions for source index...")
            df_interactions = pd.read_parquet(interactions_path)
            source_by_key = PersonaGraphBuilder.build_source_index(df_interactions)
        else:
            logger.warning("v2 mode: interactions.parquet not found at %s — source fracs will be 0.5", interactions_path)

    builder = PersonaGraphBuilder(df_item_concepts, persona_cfg, source_by_key=source_by_key)
    graphs = builder.build_all(df_sequences)

    # ------------------------------------------------------------------
    # Step 3: save cache
    # ------------------------------------------------------------------
    dataset = data_cfg.get("dataset", "amazon_beauty")
    suffix = "_v2" if args.v2 else ""
    cache_path = Path(f"data/cache/persona/{dataset}/persona_graphs{suffix}.parquet")
    cache_io = PersonaCacheIO(cache_path)
    cache_io.save_all(graphs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    df_summary = cache_io.load_all_as_df()
    logger.info("=== Summary ===")
    logger.info("total users: %d", df_summary["user_id"].nunique())
    logger.info("total persona nodes: %d", len(df_summary))
    logger.info("avg nodes/user: %.1f", len(df_summary) / df_summary["user_id"].nunique())
    logger.info("avg weight: %.4f", df_summary["weight"].mean())
    logger.info("avg stability_score: %.4f", df_summary["stability_score"].mean())
    logger.info("concept_type distribution:")
    type_counts = (
        df_summary["concept_id"]
        .str.split(":")
        .str[0]
        .value_counts()
    )
    for t, c in type_counts.items():
        logger.info("  %s: %d", t, c)


if __name__ == "__main__":
    main()
