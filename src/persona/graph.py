"""
PersonaGraph: lightweight hypothesis graph of a user's long-term preference.

Stored as a flat dict of concept_id -> PersonaNode.
Not a full graph library — edges are implicit via ontology hierarchy.

Serializes to/from a list of dicts (one per node) for parquet storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonaNode:
    user_id: str
    concept_id: str
    weight: float               # decayed cumulative affinity score
    support_count: int          # number of interactions that activated this concept
    contradiction_count: int    # times this concept conflicted with recent behavior
    stability_score: float      # fraction of time windows in which concept appeared
    last_confirmed_ts: int      # unix seconds of most recent supporting interaction
    # v2: source-aware profile (fraction of activations from rec vs search)
    source_rec_frac: float = 0.5    # fraction of activations via recommendation
    source_search_frac: float = 0.5 # fraction of activations via search


class PersonaGraph:
    """
    Container for a single user's persona nodes.
    Thin wrapper over a dict for O(1) node lookup.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._nodes: dict[str, PersonaNode] = {}

    # ------------------------------------------------------------------
    # Mutations (used only during offline build / periodic update)
    # ------------------------------------------------------------------

    def add_or_update(self, node: PersonaNode) -> None:
        existing = self._nodes.get(node.concept_id)
        if existing is None:
            self._nodes[node.concept_id] = node
        else:
            existing.weight += node.weight
            existing.support_count += node.support_count
            existing.contradiction_count += node.contradiction_count
            existing.stability_score = node.stability_score  # overwrite with latest
            existing.last_confirmed_ts = max(existing.last_confirmed_ts, node.last_confirmed_ts)

    def prune(self, min_weight: float = 0.01, top_k: Optional[int] = None) -> None:
        """Remove low-weight nodes; optionally keep only top-k by weight."""
        self._nodes = {
            cid: n for cid, n in self._nodes.items() if n.weight >= min_weight
        }
        if top_k and len(self._nodes) > top_k:
            sorted_nodes = sorted(self._nodes.values(), key=lambda n: n.weight, reverse=True)
            self._nodes = {n.concept_id: n for n in sorted_nodes[:top_k]}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, concept_id: str) -> Optional[PersonaNode]:
        return self._nodes.get(concept_id)

    def top_nodes(self, k: int) -> list[PersonaNode]:
        return sorted(self._nodes.values(), key=lambda n: n.weight, reverse=True)[:k]

    def all_nodes(self) -> list[PersonaNode]:
        return list(self._nodes.values())

    def concept_ids(self) -> set[str]:
        return set(self._nodes.keys())

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_records(self) -> list[dict]:
        return [
            {
                "user_id": n.user_id,
                "concept_id": n.concept_id,
                "weight": n.weight,
                "support_count": n.support_count,
                "contradiction_count": n.contradiction_count,
                "stability_score": n.stability_score,
                "last_confirmed_ts": n.last_confirmed_ts,
                "source_rec_frac": n.source_rec_frac,
                "source_search_frac": n.source_search_frac,
            }
            for n in self._nodes.values()
        ]

    @classmethod
    def from_records(cls, records: list[dict]) -> "PersonaGraph":
        if not records:
            raise ValueError("Cannot build PersonaGraph from empty records")
        user_id = records[0]["user_id"]
        graph = cls(user_id)
        for r in records:
            # backward compat: v1 parquet has no source fields
            node = PersonaNode(
                user_id=r["user_id"],
                concept_id=r["concept_id"],
                weight=r["weight"],
                support_count=r["support_count"],
                contradiction_count=r["contradiction_count"],
                stability_score=r["stability_score"],
                last_confirmed_ts=r["last_confirmed_ts"],
                source_rec_frac=float(r.get("source_rec_frac", 0.5)),
                source_search_frac=float(r.get("source_search_frac", 0.5)),
            )
            graph._nodes[r["concept_id"]] = node
        return graph

    @classmethod
    def uniform(cls, user_id: str) -> "PersonaGraph":
        """Empty graph used as ablation baseline (no persona)."""
        return cls(user_id)
