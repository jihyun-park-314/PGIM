"""
OntologyStore: in-memory concept registry built from ontology_nodes.parquet.

Built once during offline pipeline, read-only at inference time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ontology.schema import OntologyConcept

logger = logging.getLogger(__name__)


class OntologyStore:
    def __init__(self) -> None:
        self._concepts: dict[str, OntologyConcept] = {}

    # ------------------------------------------------------------------
    # Build from DataFrame (called by grounding after nodes are created)
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df_nodes: pd.DataFrame) -> "OntologyStore":
        store = cls()
        for _, row in df_nodes.iterrows():
            c = OntologyConcept(
                concept_id=row["concept_id"],
                concept_type=row["concept_type"],
                display_name=row["display_name"],
                parent_concept_id=row["parent_concept_id"] if pd.notna(row["parent_concept_id"]) else None,
                level=int(row["level"]),
            )
            store._concepts[c.concept_id] = c
        logger.info("OntologyStore loaded: %d concepts", len(store._concepts))
        return store

    @classmethod
    def from_parquet(cls, path: str | Path) -> "OntologyStore":
        df = pd.read_parquet(path)
        return cls.from_dataframe(df)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, concept_id: str) -> Optional[OntologyConcept]:
        return self._concepts.get(concept_id)

    def get_ancestors(self, concept_id: str) -> list[OntologyConcept]:
        """Return ancestor chain from root to (not including) this concept."""
        ancestors = []
        current = self._concepts.get(concept_id)
        while current and current.parent_concept_id:
            parent = self._concepts.get(current.parent_concept_id)
            if parent:
                ancestors.append(parent)
            current = parent
        return list(reversed(ancestors))

    def all_concepts(self) -> list[OntologyConcept]:
        return list(self._concepts.values())

    def __len__(self) -> int:
        return len(self._concepts)
