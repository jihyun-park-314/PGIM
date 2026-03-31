"""
PersonaCacheIO: read/write persona graphs as a flat parquet table.

Storage format: one row per (user_id, concept_id) node.
All graphs for a dataset are stored in a single parquet file for easy analysis.
User lookup is done by filtering on user_id.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.persona.graph import PersonaGraph

logger = logging.getLogger(__name__)


class PersonaCacheIO:
    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self._df: Optional[pd.DataFrame] = None  # lazy load

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_all(self, graphs: list[PersonaGraph]) -> None:
        """Save all persona graphs to a single parquet file."""
        if not graphs:
            raise ValueError("No persona graphs to save. Check that builder.build_all() returned results.")
        records = []
        for g in graphs:
            records.extend(g.to_records())
        df = pd.DataFrame(records)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        self._df = df
        logger.info("saved %d persona nodes (%d users) -> %s",
                    len(df), df["user_id"].nunique(), self.cache_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(self.cache_path)
        return self._df

    def load_user(self, user_id: str) -> Optional[PersonaGraph]:
        """Load a single user's PersonaGraph. Returns None if not found."""
        df = self._load()
        user_df = df[df["user_id"] == user_id]
        if user_df.empty:
            return None
        return PersonaGraph.from_records(user_df.to_dict("records"))

    def load_all_as_df(self) -> pd.DataFrame:
        """Return the full node table as a DataFrame (for analysis/eval)."""
        return self._load()

    def exists(self) -> bool:
        return self.cache_path.exists()
