"""
Grounding: items.parquet -> item_concepts.parquet + ontology_nodes.parquet

Concept sources per item:
  1. category_path  -> category hierarchy (ancestor chain)
  2. brand          -> brand:<normalized>
  3. price          -> price_band:low / mid / high
  4. details fields -> item_form:<val>, skin_type:<val>  (configurable)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lowercase, strip, replace whitespace runs with underscore."""
    return re.sub(r"\s+", "_", s.strip().lower())


def _price_band(price: Optional[float], low_max: float, mid_max: float) -> str:
    if price is None or pd.isna(price):
        return "unknown"
    if price <= low_max:
        return "low"
    if price <= mid_max:
        return "mid"
    return "high"


# ---------------------------------------------------------------------------
# Core grounding
# ---------------------------------------------------------------------------

def ground_items(
    df_items: pd.DataFrame,
    ontology_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ground every item in df_items to concept rows.

    Returns
    -------
    df_item_concepts : item_id x concept_id mapping (one row per concept per item)
    df_nodes         : unique ontology nodes (concept registry)
    """
    delimiter = ontology_cfg.get("category_delimiter", " > ")
    detail_fields: dict[str, str] = ontology_cfg.get("detail_concept_fields", {})
    price_bands = ontology_cfg.get("price_bands", {})
    low_max = price_bands.get("low_max", 15.0)
    mid_max = price_bands.get("mid_max", 40.0)
    normalize: bool = ontology_cfg.get("normalize", True)

    item_concept_rows: list[dict] = []
    # concept_id -> node dict (deduped)
    nodes: dict[str, dict] = {}

    def _add_node(
        concept_id: str,
        concept_type: str,
        display_name: str,
        parent_id: Optional[str],
        level: int,
    ) -> None:
        if concept_id not in nodes:
            nodes[concept_id] = {
                "concept_id": concept_id,
                "concept_type": concept_type,
                "display_name": display_name,
                "parent_concept_id": parent_id,
                "level": level,
            }

    def _add_item_concept(
        item_id: str,
        concept_id: str,
        concept_type: str,
        level: int,
        source: str,
    ) -> None:
        item_concept_rows.append({
            "item_id": item_id,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "level": level,
            "concept_source": source,
        })

    for _, row in tqdm(df_items.iterrows(), total=len(df_items), desc="grounding items"):
        item_id = row["item_id"]
        meta = json.loads(row["raw_meta_json"]) if isinstance(row.get("raw_meta_json"), str) else {}
        details: dict = meta.get("details") or {}

        # ---- 1. category_path -> ancestor chain ----
        cat_path: str = row.get("category_path") or ""
        if cat_path:
            parts = [p.strip() for p in cat_path.split(delimiter) if p.strip()]
            parent_id: Optional[str] = None
            for level, part in enumerate(parts):
                cid = f"category:{_normalize(part) if normalize else part}"
                _add_node(cid, "category", part, parent_id, level)
                _add_item_concept(item_id, cid, "category", level, "category_path")
                parent_id = cid

        # ---- 2. brand ----
        brand = row.get("brand") or details.get("Brand") or details.get("Manufacturer")
        if brand and isinstance(brand, str) and brand.strip():
            b_norm = _normalize(brand) if normalize else brand.strip()
            cid = f"brand:{b_norm}"
            _add_node(cid, "brand", brand.strip(), None, 0)
            _add_item_concept(item_id, cid, "brand", 0, "brand")

        # ---- 3. price band ----
        price = row.get("price")
        band = _price_band(price, low_max, mid_max)
        cid = f"price_band:{band}"
        _add_node(cid, "price_band", band, None, 0)
        _add_item_concept(item_id, cid, "price_band", 0, "price")

        # ---- 4. detail concept fields (Item Form, Skin Type, …) ----
        for detail_key, concept_type in detail_fields.items():
            val = details.get(detail_key)
            if not val or not isinstance(val, str):
                continue
            # some values are comma-separated (e.g. "Sensitive,All Skin Types")
            for v in val.split(","):
                v = v.strip()
                if not v:
                    continue
                v_norm = _normalize(v) if normalize else v
                cid = f"{concept_type}:{v_norm}"
                _add_node(cid, concept_type, v, None, 0)
                _add_item_concept(item_id, cid, concept_type, 0, f"detail:{detail_key}")

    df_item_concepts = pd.DataFrame(item_concept_rows)
    df_nodes = pd.DataFrame(list(nodes.values()))

    logger.info("item_concepts: %d rows across %d items",
                len(df_item_concepts), df_item_concepts["item_id"].nunique())
    logger.info("ontology_nodes: %d unique concepts", len(df_nodes))
    return df_item_concepts, df_nodes


def run(
    data_cfg: dict,
    ontology_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load items.parquet, ground, and save item_concepts + ontology_nodes.
    Returns (df_item_concepts, df_nodes).
    """
    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir = Path(data_cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    df_items = pd.read_parquet(processed_dir / "items.parquet")
    df_item_concepts, df_nodes = ground_items(df_items, ontology_cfg)

    ic_path = processed_dir / "item_concepts.parquet"
    nodes_path = interim_dir / "ontology_nodes.parquet"

    df_item_concepts.to_parquet(ic_path, index=False)
    df_nodes.to_parquet(nodes_path, index=False)
    logger.info("saved -> %s", ic_path)
    logger.info("saved -> %s", nodes_path)

    return df_item_concepts, df_nodes
