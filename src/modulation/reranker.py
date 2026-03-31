"""
reranker.py
-----------
Applies modulation signal to backbone top-K candidates.

Final score = base_score + modulation_delta
delta >= 0 for boost, delta < 0 for suppress/filter penalty.

Modes (controlled by experiment config):
    backbone_only         — delta = 0 for all candidates
    persona_only_rerank   — signal built from persona only (no intent)
    intent_only_rerank    — signal built from intent only (no persona)
    weighted_sum_baseline — simple weighted combination of persona + intent scores
    hard_switch_baseline  — if is_deviation: use intent score, else use persona score
    graph_conditioned_full — full reason-conditioned modulation (main model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.modulation.signal_builder import ModulationSignal, _load_idf

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    user_id: str
    target_index: int
    candidate_item_id: str
    base_score: float
    modulation_delta: float
    final_score: float
    rank_before: int
    rank_after: int
    deviation_reason: str
    gate_strength: float

    def to_record(self) -> dict:
        return self.__dict__


class CandidateReranker:
    def __init__(self, modulation_cfg: dict, item_concepts: dict[str, list[str]]) -> None:
        """
        item_concepts: item_id -> list[concept_id]
        """
        self._cfg = modulation_cfg
        self._item_concepts = item_concepts

        # concept IDF weights (optional): concept_id -> idf_weight in [idf_floor, 1.0]
        idf_cfg = modulation_cfg.get("concept_idf", {})
        if idf_cfg.get("enabled", False):
            idf_path = idf_cfg.get("idf_path", "")
            self._idf: dict[str, float] = _load_idf(idf_path) if idf_path else {}
        else:
            self._idf = {}

    def rerank(
        self,
        candidates: list[tuple[str, float]],   # (item_id, base_score), sorted desc
        signal: ModulationSignal,
        mode: str = "graph_conditioned_full",
    ) -> list[RankedCandidate]:
        """
        Rerank backbone candidates using modulation signal.

        Returns list sorted by final_score descending.
        """
        if mode == "backbone_only":
            return self._identity(candidates, signal)

        delta_cfg = self._cfg.get("delta", {})
        reason_policy = self._cfg.get("reason_policy", {})
        policy = reason_policy.get(signal.reason_type, reason_policy.get("unknown", {}))

        boost_scale: float = policy.get("boost_scale", 0.3)
        suppress_scale: float = policy.get("suppress_scale", 0.1)
        filter_active: bool = policy.get("filter_active", False)

        max_boost: float = delta_cfg.get("max_boost", 0.30)
        max_suppress: float = delta_cfg.get("max_suppress", 0.15)
        filter_penalty: float = delta_cfg.get("filter_penalty", 0.50)

        boost_set = set(signal.boost_concepts)
        suppress_set = set(signal.suppress_concepts)

        results = []
        for rank_before, (item_id, base_score) in enumerate(candidates, start=1):
            item_concept_set = set(self._item_concepts.get(item_id, []))
            delta = 0.0

            if mode == "backbone_only":
                delta = 0.0

            elif mode == "weighted_sum_baseline":
                # simple overlap score, weighted sum with base
                ws = self._cfg.get("weighted_sum", {})
                persona_w = ws.get("persona_weight", 0.4)
                intent_w = ws.get("intent_weight", 0.6)
                persona_overlap = len(item_concept_set & set(signal.boost_concepts)) / max(1, len(item_concept_set))
                intent_overlap = len(item_concept_set & boost_set) / max(1, len(item_concept_set))
                delta = persona_w * persona_overlap + intent_w * intent_overlap
                delta *= signal.gate_strength

            elif mode == "hard_switch_baseline":
                # hard switch: deviation -> intent-only, else -> persona-only
                if signal.reason_type not in ("aligned", "unknown"):
                    overlap = len(item_concept_set & boost_set) / max(1, len(item_concept_set))
                    delta = overlap * max_boost
                else:
                    # persona overlap
                    overlap = len(item_concept_set & boost_set) / max(1, len(item_concept_set))
                    delta = overlap * max_boost * 0.5

            elif mode in ("persona_only_rerank", "intent_only_rerank", "graph_conditioned_full"):
                # --- boost ---
                matched_boost = item_concept_set & boost_set

                if signal.concept_signals:
                    # v2: per-concept weighted delta
                    # delta_boost = Σ combined_score(c) for c in item_concepts ∩ boost_signals
                    # Normalized by sum of all boost signal scores so delta stays in [0, max_boost]
                    matched_signals = {
                        c: signal.concept_signals[c]
                        for c in item_concept_set
                        if c in signal.concept_signals
                    }
                    if matched_signals:
                        total_signal_mass = sum(
                            s.combined_score for s in signal.concept_signals.values()
                        )
                        if total_signal_mass > 0:
                            matched_mass = sum(s.combined_score for s in matched_signals.values())
                            boost_ratio = matched_mass / total_signal_mass
                        else:
                            boost_ratio = 0.0
                        delta += boost_ratio * max_boost * boost_scale * signal.gate_strength

                elif matched_boost and boost_set:
                    # v1 fallback: IDF-weighted overlap ratio
                    if self._idf:
                        idf_floor = self._cfg.get("concept_idf", {}).get("idf_floor", 0.1)
                        matched_w = sum(self._idf.get(c, idf_floor) for c in matched_boost)
                        total_w   = sum(self._idf.get(c, idf_floor) for c in boost_set)
                        boost_ratio = matched_w / total_w if total_w > 0 else 0.0
                    else:
                        boost_ratio = len(matched_boost) / len(boost_set)
                    delta += boost_ratio * max_boost * boost_scale * signal.gate_strength

                # --- suppress ---
                suppress_overlap = len(item_concept_set & suppress_set)
                if suppress_overlap > 0 and suppress_set:
                    suppress_ratio = suppress_overlap / len(suppress_set)
                    delta -= suppress_ratio * max_suppress * suppress_scale * signal.gate_strength

                # --- filter penalty ---
                if filter_active and signal.filter_constraints:
                    if self._violates_constraints(item_concept_set, signal.filter_constraints):
                        delta -= filter_penalty * signal.gate_strength

            final_score = base_score + delta
            results.append(RankedCandidate(
                user_id=signal.user_id,
                target_index=signal.target_index,
                candidate_item_id=item_id,
                base_score=round(base_score, 6),
                modulation_delta=round(delta, 6),
                final_score=round(final_score, 6),
                rank_before=rank_before,
                rank_after=0,  # filled below
                deviation_reason=signal.reason_type,
                gate_strength=signal.gate_strength,
            ))

        # assign rank_after
        results.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(results, start=1):
            r.rank_after = i

        return results

    def _identity(
        self,
        candidates: list[tuple[str, float]],
        signal: ModulationSignal,
    ) -> list[RankedCandidate]:
        return [
            RankedCandidate(
                user_id=signal.user_id,
                target_index=signal.target_index,
                candidate_item_id=item_id,
                base_score=round(score, 6),
                modulation_delta=0.0,
                final_score=round(score, 6),
                rank_before=i + 1,
                rank_after=i + 1,
                deviation_reason=signal.reason_type,
                gate_strength=0.0,
            )
            for i, (item_id, score) in enumerate(candidates)
        ]

    def _violates_constraints(
        self,
        item_concept_set: set[str],
        filter_constraints: dict,
    ) -> bool:
        """
        Returns True if item violates any constraint in filter_constraints.
        filter_constraints example: {"price_band": ["price_band:high"]}
        -> item must NOT have price_band:high

        Interpretation: constraints specify DESIRED values.
        Violation = item has a concept in the SAME type but NOT in allowed list.
        """
        for concept_type, allowed_list in filter_constraints.items():
            if not allowed_list:
                continue
            allowed_set = set(allowed_list)
            item_type_concepts = {c for c in item_concept_set if c.startswith(concept_type + ":")}
            if item_type_concepts and not item_type_concepts.intersection(allowed_set):
                return True
        return False
