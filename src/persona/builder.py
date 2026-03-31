"""
PersonaGraphBuilder: user_sequences + item_concepts -> PersonaGraph per user.

Algorithm per user
------------------
1. Exclude eval window via window_guard.
2. For each remaining interaction, look up its concepts in item_concepts.
3. Apply time decay to each concept activation (weight = decay(1.0, event_ts)).
4. Accumulate: weight, support_count.
5. Compute stability_score: fraction of stability windows in which concept appeared.
6. Compute contradiction_count: how many recent interactions activate concepts
   NOT in user's top-K long-term set.
7. Prune below min_weight, keep top_k.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.persona.decay import decay_weight
from src.persona.graph import PersonaGraph, PersonaNode
from src.persona.window_guard import exclude_eval_window, assert_no_leakage

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400


class PersonaGraphBuilder:
    def __init__(
        self,
        df_item_concepts: pd.DataFrame,
        persona_cfg: dict,
        source_by_key: "dict[tuple[str,str], str] | None" = None,
    ) -> None:
        """
        source_by_key: optional dict (user_id, item_id) -> source_service ("rec"/"search")
        When provided, persona nodes will include source_rec_frac / source_search_frac.
        """
        # item_id -> list of concept_ids
        self._item_concepts: dict[str, list[str]] = (
            df_item_concepts.groupby("item_id")["concept_id"]
            .apply(list)
            .to_dict()
        )
        self._cfg = persona_cfg
        # (user_id, item_id) -> dominant source; None means v1 (no source info)
        self._source_by_key = source_by_key

    def build(
        self,
        user_id: str,
        item_sequence: list[str],
        timestamp_sequence: list[int],
    ) -> Optional[PersonaGraph]:
        """
        Build a PersonaGraph for one user.
        Returns None if the sequence is too short after exclusion.
        """
        excl_cfg = self._cfg.get("eval_exclusion", {})
        try:
            items, timestamps = exclude_eval_window(
                item_sequence,
                timestamp_sequence,
                mode=excl_cfg.get("mode", "tail_n"),
                tail_n=excl_cfg.get("tail_n", 1),
            )
        except ValueError:
            logger.warning("User %s: empty sequence after eval exclusion, skipped", user_id)
            return None

        decay_cfg = self._cfg.get("decay", {})
        half_life_days: float = decay_cfg.get("half_life_days", 30)
        min_weight: float = decay_cfg.get("min_weight", 0.01)

        p_cfg = self._cfg.get("persona", {})
        min_support: int = p_cfg.get("min_support", 2)
        top_k: int = p_cfg.get("top_k_concepts", 50)
        stability_window_days: float = p_cfg.get("stability_window_days", 14)

        contradiction_cfg = self._cfg.get("contradiction", {})
        top_k_ref: int = contradiction_cfg.get("top_k_reference", 10)
        recent_n: int = contradiction_cfg.get("recent_window_size", 5)

        reference_ts = timestamps[-1]  # most recent interaction as "now"

        # ------------------------------------------------------------------
        # Pass 1: accumulate decayed weights and support counts per concept
        # ------------------------------------------------------------------
        concept_weights: dict[str, float] = defaultdict(float)
        concept_support: dict[str, int] = defaultdict(int)
        concept_last_ts: dict[str, int] = defaultdict(int)
        # v2: source counts per concept
        concept_source_rec: dict[str, int] = defaultdict(int)
        concept_source_search: dict[str, int] = defaultdict(int)
        # for stability: track which windows each concept appeared in
        # window index = floor(elapsed_days / stability_window_days) from oldest ts
        concept_windows: dict[str, set[int]] = defaultdict(set)
        total_windows = max(
            1,
            math.ceil((reference_ts - timestamps[0]) / (_SECONDS_PER_DAY * stability_window_days))
        )

        for item_id, ts in zip(items, timestamps):
            concepts = self._item_concepts.get(item_id, [])
            w = decay_weight(1.0, ts, reference_ts, half_life_days)
            elapsed_days = max(0, (reference_ts - ts)) / _SECONDS_PER_DAY
            window_idx = int(elapsed_days / stability_window_days)

            # v2: look up source for this (user, item) interaction
            src = None
            if self._source_by_key is not None:
                src = self._source_by_key.get((user_id, item_id))

            for cid in concepts:
                concept_weights[cid] += w
                concept_support[cid] += 1
                concept_last_ts[cid] = max(concept_last_ts[cid], ts)
                concept_windows[cid].add(window_idx)
                if src == "rec":
                    concept_source_rec[cid] += 1
                elif src == "search":
                    concept_source_search[cid] += 1

        # ------------------------------------------------------------------
        # Pass 2: contradiction — compare top long-term vs recent behavior
        # ------------------------------------------------------------------
        # Long-term reference: top_k_ref concepts by raw support (pre-decay)
        top_longterm = set(
            sorted(concept_support, key=concept_support.get, reverse=True)[:top_k_ref]
        )
        # Recent concepts: from last recent_n interactions
        recent_concept_set: set[str] = set()
        for item_id in items[-recent_n:]:
            recent_concept_set.update(self._item_concepts.get(item_id, []))

        # contradiction_count per concept: how many recent items did NOT activate it
        # while it is in top long-term set
        concept_contradictions: dict[str, int] = defaultdict(int)
        for item_id in items[-recent_n:]:
            item_concepts_set = set(self._item_concepts.get(item_id, []))
            for cid in top_longterm:
                if cid not in item_concepts_set:
                    concept_contradictions[cid] += 1

        # ------------------------------------------------------------------
        # Assemble PersonaGraph
        # ------------------------------------------------------------------
        graph = PersonaGraph(user_id)
        for cid, w in concept_weights.items():
            if concept_support[cid] < min_support:
                continue
            windows_seen = len(concept_windows[cid])
            stability = windows_seen / total_windows

            # v2: compute source fracs
            n_rec    = concept_source_rec.get(cid, 0)
            n_search = concept_source_search.get(cid, 0)
            n_total  = n_rec + n_search
            if n_total > 0:
                src_rec_frac    = round(n_rec / n_total, 4)
                src_search_frac = round(n_search / n_total, 4)
            else:
                # no source info available (v1 compat or item not matched)
                src_rec_frac    = 0.5
                src_search_frac = 0.5

            node = PersonaNode(
                user_id=user_id,
                concept_id=cid,
                weight=w,
                support_count=concept_support[cid],
                contradiction_count=concept_contradictions.get(cid, 0),
                stability_score=round(stability, 4),
                last_confirmed_ts=concept_last_ts[cid],
                source_rec_frac=src_rec_frac,
                source_search_frac=src_search_frac,
            )
            graph.add_or_update(node)

        graph.prune(min_weight=min_weight, top_k=top_k)

        # Leakage check: only meaningful for after_timestamp mode.
        # For tail_n, the slice is non-overlapping by construction.
        if excl_cfg.get("mode", "tail_n") == "after_timestamp":
            eval_items = set(item_sequence) - set(items)
            assert_no_leakage(set(items), eval_items)

        return graph

    def build_all(
        self,
        df_sequences: pd.DataFrame,
    ) -> list[PersonaGraph]:
        graphs = []
        for _, row in tqdm(df_sequences.iterrows(), total=len(df_sequences), desc="building persona graphs"):
            g = self.build(
                user_id=row["user_id"],
                item_sequence=row["item_sequence"],
                timestamp_sequence=row["timestamp_sequence"],
            )
            if g is not None:
                graphs.append(g)
        logger.info("built %d persona graphs", len(graphs))
        return graphs

    @staticmethod
    def build_source_index(df_interactions: pd.DataFrame) -> "dict[tuple[str,str], str]":
        """
        Build (user_id, item_id) -> source_service lookup from interactions DataFrame.
        When a user interacted with the same item via multiple sources, keep the last one.
        Used to populate source_rec_frac / source_search_frac in PersonaNode.
        """
        if "source_service" not in df_interactions.columns:
            logger.warning("interactions has no source_service column — source index empty")
            return {}
        # sort by timestamp so last interaction wins on duplicate keys
        df_sorted = df_interactions.sort_values("timestamp")
        index: dict[tuple[str, str], str] = {}
        for row in df_sorted.itertuples(index=False):
            src = getattr(row, "source_service", None)
            if src in ("rec", "search"):
                index[(str(row.user_id), str(row.item_id))] = src
        logger.info("source index built: %d (user,item) pairs", len(index))
        return index
