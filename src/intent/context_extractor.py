"""
context_extractor.py
--------------------
Builds IntentContext from a snapshot row + persona graph nodes + item concepts.

IntentContext is the intermediate feature object consumed by heuristic/llm interpreters.
It is also saved as intent_features.parquet for debugging.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntentContext:
    user_id: str
    target_index: int
    recent_item_ids: list[str]

    # concept frequency over recent items: concept_id -> count
    recent_concept_freq: dict[str, int] = field(default_factory=dict)

    # dominant values per concept type in recent window
    recent_dominant_brand: Optional[str] = None
    recent_dominant_price_band: Optional[str] = None
    recent_dominant_item_form: Optional[str] = None
    recent_dominant_skin_type: Optional[str] = None

    # persona summary
    persona_top_concepts: list[str] = field(default_factory=list)   # sorted by weight
    persona_dominant_price_band: Optional[str] = None
    persona_dominant_brand: Optional[str] = None
    persona_dominant_item_form: Optional[str] = None
    persona_dominant_skin_type: Optional[str] = None

    # derived alignment signal
    overlap_ratio: float = 0.0        # fraction of recent signal concepts in persona top
    recent_concept_entropy: float = 0.0

    # v2: source-aware context
    current_source: str = "unknown"          # dominant source in recent window ("rec"/"search"/"unknown")
    recent_source_rec_frac: float = 0.5      # fraction of recent items from rec
    recent_source_search_frac: float = 0.5   # fraction of recent items from search
    source_shift_flag: bool = False          # True if source composition shifted notably

    # v3: temporal split — concept freq in first vs second half of recent window
    # {"first_half": {concept_id: count}, "second_half": {concept_id: count}}
    # Used by llm_interpreter to populate temporal_cues in the interpretation record.
    recent_concept_temporal_split: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "user_id": self.user_id,
            "target_index": self.target_index,
            "recent_item_ids_json": json.dumps(self.recent_item_ids),
            "recent_concept_freq_json": json.dumps(self.recent_concept_freq),
            "recent_dominant_brand": self.recent_dominant_brand,
            "recent_dominant_price_band": self.recent_dominant_price_band,
            "recent_dominant_item_form": self.recent_dominant_item_form,
            "recent_dominant_skin_type": self.recent_dominant_skin_type,
            "persona_top_concepts_json": json.dumps(self.persona_top_concepts),
            "persona_dominant_price_band": self.persona_dominant_price_band,
            "persona_dominant_brand": self.persona_dominant_brand,
            "persona_dominant_item_form": self.persona_dominant_item_form,
            "persona_dominant_skin_type": self.persona_dominant_skin_type,
            "overlap_ratio": self.overlap_ratio,
            "recent_concept_entropy": self.recent_concept_entropy,
            "current_source": self.current_source,
            "recent_source_rec_frac": self.recent_source_rec_frac,
            "recent_source_search_frac": self.recent_source_search_frac,
            "source_shift_flag": self.source_shift_flag,
            "recent_concept_temporal_split_json": json.dumps(self.recent_concept_temporal_split),
        }


def _dominant(concept_freq: dict[str, int], prefix: str) -> Optional[str]:
    """Return most frequent concept_id with given prefix, or None."""
    filtered = {k: v for k, v in concept_freq.items() if k.startswith(prefix + ":")}
    if not filtered:
        return None
    return max(filtered, key=filtered.get)


def _entropy(freq: dict[str, int]) -> float:
    """Normalized entropy of a frequency distribution (0=uniform, 1=concentrated)."""
    import math
    total = sum(freq.values())
    if total == 0 or len(freq) <= 1:
        return 0.0
    ent = -sum((v / total) * math.log2(v / total) for v in freq.values() if v > 0)
    max_ent = math.log2(len(freq))
    # invert: high entropy = high dispersion = closer to 1
    return round(ent / max_ent, 4) if max_ent > 0 else 0.0


class ContextExtractor:
    def __init__(
        self,
        item_concepts: dict[str, list[str]],   # item_id -> list[concept_id]
        persona_nodes: dict[str, list[dict]],   # user_id -> list of node dicts
        signal_concept_types: list[str],
        top_goal_concepts: int = 3,
        source_by_key: "dict[tuple[str,str], str] | None" = None,
    ) -> None:
        self._item_concepts = item_concepts
        self._persona_nodes = persona_nodes
        self._signal_types = set(signal_concept_types)
        self._top_goal_concepts = top_goal_concepts
        # v2: (user_id, item_id) -> source_service
        self._source_by_key = source_by_key or {}

    def extract(self, snapshot_row: dict) -> IntentContext:
        user_id = snapshot_row["user_id"]
        target_index = int(snapshot_row["target_index"])
        recent_items = list(snapshot_row["recent_item_ids"])

        # --- recent concept frequency ---
        freq: Counter = Counter()
        for item_id in recent_items:
            for cid in self._item_concepts.get(item_id, []):
                ctype = cid.split(":")[0]
                if ctype in self._signal_types:
                    freq[cid] += 1

        recent_concept_freq = dict(freq)

        # --- persona summary ---
        p_nodes = self._persona_nodes.get(user_id, [])
        # sort by weight desc
        p_nodes_sorted = sorted(p_nodes, key=lambda n: n["weight"], reverse=True)
        persona_top = [n["concept_id"] for n in p_nodes_sorted
                       if n["concept_id"].split(":")[0] in self._signal_types]

        persona_freq: Counter = Counter()
        for n in p_nodes:
            cid = n["concept_id"]
            if cid.split(":")[0] in self._signal_types:
                persona_freq[cid] = n["support_count"]

        # --- overlap: what fraction of recent signal concepts are in persona top ---
        recent_signal_concepts = set(recent_concept_freq.keys())
        persona_top_set = set(persona_top[:10])  # top-10 as reference
        if recent_signal_concepts:
            overlap = len(recent_signal_concepts & persona_top_set) / len(recent_signal_concepts)
        else:
            overlap = 0.0

        # v2: source-aware context
        current_source, recent_rec_frac, recent_search_frac, source_shift = \
            self._compute_source_context(user_id, recent_items)

        # v3: temporal split — first half vs second half of recent window
        temporal_split = self._compute_temporal_split(recent_items)

        ctx = IntentContext(
            user_id=user_id,
            target_index=target_index,
            recent_item_ids=recent_items,
            recent_concept_freq=recent_concept_freq,
            recent_dominant_brand=_dominant(recent_concept_freq, "brand"),
            recent_dominant_price_band=_dominant(recent_concept_freq, "price_band"),
            recent_dominant_item_form=_dominant(recent_concept_freq, "item_form"),
            recent_dominant_skin_type=_dominant(recent_concept_freq, "skin_type"),
            persona_top_concepts=persona_top[:self._top_goal_concepts * 2],
            persona_dominant_price_band=_dominant(dict(persona_freq), "price_band"),
            persona_dominant_brand=_dominant(dict(persona_freq), "brand"),
            persona_dominant_item_form=_dominant(dict(persona_freq), "item_form"),
            persona_dominant_skin_type=_dominant(dict(persona_freq), "skin_type"),
            overlap_ratio=round(overlap, 4),
            recent_concept_entropy=_entropy(recent_concept_freq),
            current_source=current_source,
            recent_source_rec_frac=recent_rec_frac,
            recent_source_search_frac=recent_search_frac,
            source_shift_flag=source_shift,
            recent_concept_temporal_split=temporal_split,
        )
        return ctx

    def _compute_source_context(
        self, user_id: str, recent_items: list[str]
    ) -> "tuple[str, float, float, bool]":
        """
        Returns (current_source, rec_frac, search_frac, source_shift_flag).

        current_source: "rec" / "search" / "unknown"
        source_shift_flag: True if the last half of recent window has a very different
                           source mix than the first half (shift > 0.4).
        """
        if not self._source_by_key or not recent_items:
            return "unknown", 0.5, 0.5, False

        sources = [self._source_by_key.get((user_id, iid), "unknown") for iid in recent_items]

        n_rec    = sources.count("rec")
        n_search = sources.count("search")
        n_known  = n_rec + n_search
        if n_known == 0:
            return "unknown", 0.5, 0.5, False

        rec_frac    = round(n_rec / n_known, 4)
        search_frac = round(n_search / n_known, 4)
        current_source = "rec" if rec_frac >= 0.5 else "search"

        # source shift: compare first half vs second half of recent window
        mid = max(1, len(recent_items) // 2)
        first_half  = sources[:mid]
        second_half = sources[mid:]

        def _rec_frac(src_list: list[str]) -> float:
            n_r = src_list.count("rec")
            n_k = n_r + src_list.count("search")
            return n_r / n_k if n_k > 0 else 0.5

        shift = abs(_rec_frac(first_half) - _rec_frac(second_half))
        source_shift = shift > 0.4

        return current_source, rec_frac, search_frac, source_shift

    def _compute_temporal_split(self, recent_items: list[str]) -> dict:
        """
        Split recent window into first half / second half and compute
        goal-eligible concept frequency for each half.

        Returns:
            {
                "first_half":  {concept_id: count, ...},
                "second_half": {concept_id: count, ...},
            }

        Uses only goal-eligible concepts (no PLATFORM/UMBRELLA/NAV/PROMO/FORMAT),
        matching the same filter applied in the main recent_concept_freq.
        Returns empty dict when recent_items has fewer than 2 items (no split possible).
        """
        if len(recent_items) < 2:
            return {}

        mid = max(1, len(recent_items) // 2)
        first_half_items  = recent_items[:mid]
        second_half_items = recent_items[mid:]

        def _freq_for(items: list[str]) -> dict[str, int]:
            freq: Counter = Counter()
            for item_id in items:
                for cid in self._item_concepts.get(item_id, []):
                    ctype = cid.split(":")[0]
                    if ctype in self._signal_types:
                        freq[cid] += 1
            return dict(freq)

        return {
            "first_half":  _freq_for(first_half_items),
            "second_half": _freq_for(second_half_items),
        }
