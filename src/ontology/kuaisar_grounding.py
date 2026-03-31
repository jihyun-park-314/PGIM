"""
kuaisar_grounding.py
--------------------
KuaiSAR items.parquet -> item_concepts.parquet + ontology_nodes.parquet

Concept sources (KuaiSAR-specific):
    1. category hierarchy  -> cat_l1 / cat_l2 / cat_l3 / cat_l4
    2. source_service      -> service:rec / service:search  (from interactions)
    3. author_id           -> author:<id>  (optional, off by default)

KuaiSAR has no price, brand, or detail fields, so those concept types are skipped.
The category hierarchy is the primary semantic signal (4 levels, English names).

Output schema is identical to Amazon grounding:
    item_concepts.parquet  : item_id, concept_id, concept_type, level, concept_source
    ontology_nodes.parquet : concept_id, concept_type, display_name,
                             parent_concept_id, level
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip().lower())


# ---------------------------------------------------------------------------
# Core grounding
# ---------------------------------------------------------------------------

def ground_items_kuaisar(
    df_items: pd.DataFrame,
    ontology_cfg: dict,
    df_interactions: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ground KuaiSAR items to concept rows.

    Parameters
    ----------
    df_items        : items.parquet (from kuaisar_preprocessor)
    ontology_cfg    : config/ontology/kuaisar_v1.yaml contents
    df_interactions : optional — used to attach source_service concepts per item

    Returns
    -------
    df_item_concepts, df_nodes
    """
    delimiter   = ontology_cfg.get("category_delimiter", " > ")
    normalize   = ontology_cfg.get("normalize", True)
    max_depth   = ontology_cfg.get("max_category_depth", 4)
    skip_brand  = ontology_cfg.get("skip_brand", True)
    skip_price  = ontology_cfg.get("skip_price_band", True)
    add_service = ontology_cfg.get("add_source_service_concept", True)
    level_prefix: dict = ontology_cfg.get("category_level_prefix", {
        0: "cat_l1", 1: "cat_l2", 2: "cat_l3", 3: "cat_l4"
    })

    item_concept_rows: list[dict] = []
    nodes: dict[str, dict] = {}

    def _add_node(cid: str, ctype: str, display: str,
                  parent: Optional[str], level: int) -> None:
        if cid not in nodes:
            nodes[cid] = {
                "concept_id": cid, "concept_type": ctype,
                "display_name": display, "parent_concept_id": parent, "level": level,
            }

    def _add_ic(item_id: str, cid: str, ctype: str, level: int, src: str) -> None:
        item_concept_rows.append({
            "item_id": item_id, "concept_id": cid,
            "concept_type": ctype, "level": level, "concept_source": src,
        })

    # ── service concept index: item_id -> set of services ──────────────
    service_by_item: dict[str, set[str]] = {}
    if add_service and df_interactions is not None and "source_service" in df_interactions.columns:
        for iid, grp in df_interactions.groupby("item_id"):
            service_by_item[str(iid)] = set(grp["source_service"].dropna().unique())

    # ── ground each item ────────────────────────────────────────────────
    for _, row in tqdm(df_items.iterrows(), total=len(df_items), desc="grounding KuaiSAR"):
        item_id = str(row["item_id"])
        meta = json.loads(row["raw_meta_json"]) if isinstance(row.get("raw_meta_json"), str) else {}

        # ---- 1. category hierarchy ----
        _cp = row.get("category_path")
        cat_path: str = "" if (not _cp or (isinstance(_cp, float) and pd.isna(_cp))) else str(_cp)
        if cat_path:
            parts = [p.strip() for p in cat_path.split(delimiter) if p.strip()]
            parent_id: Optional[str] = None
            for depth, part in enumerate(parts[:max_depth]):
                ctype = level_prefix.get(depth, f"cat_l{depth+1}")
                norm = _normalize(part) if normalize else part
                cid = f"{ctype}:{norm}"
                _add_node(cid, ctype, part, parent_id, depth)
                _add_ic(item_id, cid, ctype, depth, "category_path")
                parent_id = cid

        # ---- 2. source_service concept ----
        if add_service:
            services = service_by_item.get(item_id, set())
            for svc in services:
                cid = f"service:{svc}"
                _add_node(cid, "service", svc, None, 0)
                _add_ic(item_id, cid, "service", 0, "source_service")

    df_item_concepts = pd.DataFrame(item_concept_rows)
    df_nodes = pd.DataFrame(list(nodes.values()))

    if not df_item_concepts.empty:
        logger.info(
            "item_concepts: %d rows across %d items",
            len(df_item_concepts), df_item_concepts["item_id"].nunique(),
        )
    logger.info("ontology_nodes: %d unique concepts", len(df_nodes))
    return df_item_concepts, df_nodes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    data_cfg: dict,
    ontology_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load items.parquet (and optionally interactions.parquet),
    ground, and save item_concepts + ontology_nodes.
    """
    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir   = Path(data_cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    df_items = pd.read_parquet(processed_dir / "items.parquet")

    # load interactions for source_service concept enrichment
    inter_path = processed_dir / "interactions.parquet"
    df_interactions = pd.read_parquet(inter_path) if inter_path.exists() else None

    df_item_concepts, df_nodes = ground_items_kuaisar(
        df_items, ontology_cfg, df_interactions=df_interactions
    )

    ic_path    = processed_dir / "item_concepts.parquet"
    nodes_path = interim_dir   / "ontology_nodes.parquet"

    df_item_concepts.to_parquet(ic_path, index=False)
    df_nodes.to_parquet(nodes_path, index=False)
    logger.info("saved -> %s", ic_path)
    logger.info("saved -> %s", nodes_path)

    return df_item_concepts, df_nodes
