"""
BackboneInterface: abstract base for all backbone recommenders.

Contract:
    Input  : user_id + item_sequence (history before target)
    Output : list of (item_id, base_score) sorted by score desc, length = top_k

The user modeling layer sees ONLY this interface.
Backbone internals (embeddings, attention weights) are not exposed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BackboneInterface(ABC):
    @abstractmethod
    def get_top_k(
        self,
        user_id: str,
        item_sequence: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Return top-K (item_id, base_score) candidates, sorted by score descending.
        item_ids in item_sequence should not appear in the output (already seen).
        """
        ...

    @abstractmethod
    def get_all_scores(
        self,
        user_id: str,
        item_sequence: list[str],
    ) -> dict[str, float]:
        """
        Return scores for ALL candidate items (unseen by user).
        Used for evaluation where we need full ranking.
        """
        ...
