"""
online.py
---------
Online inference pipeline for a single user/step.

Flow:
    1. backbone.get_top_k() -> candidates
    2. persona_cache.load_user() -> persona_nodes
    3. intent_cache lookup -> intent_record
    4. gate.compute_gate_strength()
    5. signal_builder.build_signal()
    6. reranker.rerank(candidates, signal, mode)
    7. return ranked list
"""

from __future__ import annotations

import logging
from typing import Optional

from src.backbone.interface import BackboneInterface
from src.modulation.gate import compute_gate_strength
from src.modulation.reranker import CandidateReranker, RankedCandidate
from src.modulation.signal_builder import build_signal

logger = logging.getLogger(__name__)


class OnlinePipeline:
    def __init__(
        self,
        backbone: BackboneInterface,
        reranker: CandidateReranker,
        persona_nodes_by_user: dict[str, list[dict]],   # user_id -> list of node dicts
        intent_by_key: dict[tuple[str, int], dict],     # (user_id, target_index) -> intent record
        backbone_cfg: dict,
        modulation_cfg: dict,
        mode: str = "graph_conditioned_full",
    ) -> None:
        self._backbone = backbone
        self._reranker = reranker
        self._persona = persona_nodes_by_user
        self._intents = intent_by_key
        self._backbone_cfg = backbone_cfg
        self._modulation_cfg = modulation_cfg
        self._mode = mode

    def run(
        self,
        user_id: str,
        item_sequence: list[str],
        target_index: int,
        top_k: Optional[int] = None,
    ) -> list[RankedCandidate]:
        k = top_k or self._backbone_cfg.get("top_k", 100)

        # 1. backbone candidates
        candidates = self._backbone.get_top_k(user_id, item_sequence, k)

        if self._mode == "backbone_only":
            # build a neutral signal just for record-keeping
            dummy_intent = {
                "user_id": user_id,
                "target_index": target_index,
                "goal_concepts": [],
                "constraints_json": "{}",
                "deviation_reason": "unknown",
                "confidence": 0.0,
                "ttl_steps": 1,
                "persona_alignment_score": 0.0,
                "is_deviation": 0,
            }
            signal = build_signal(dummy_intent, [], 0.0, self._modulation_cfg)
            return self._reranker.rerank(candidates, signal, mode="backbone_only")

        # 2. persona nodes
        persona_nodes = self._persona.get(user_id, [])

        # 3. intent record
        intent_record = self._intents.get((user_id, target_index))
        if intent_record is None:
            logger.debug("No intent for user=%s idx=%d, using neutral", user_id, target_index)
            intent_record = {
                "user_id": user_id,
                "target_index": target_index,
                "goal_concepts": [],
                "constraints_json": "{}",
                "deviation_reason": "unknown",
                "confidence": 0.35,
                "ttl_steps": 1,
                "persona_alignment_score": 0.0,
                "is_deviation": 0,
            }

        # 4. gate
        gate_strength = compute_gate_strength(
            deviation_reason=intent_record.get("deviation_reason", "unknown"),
            confidence=float(intent_record.get("confidence", 0.35)),
            persona_alignment_score=float(intent_record.get("persona_alignment_score", 0.0)),
            gate_cfg=self._modulation_cfg.get("gate", {}),
        )

        # 5. signal — mode propagated so builder selects correct source
        signal = build_signal(intent_record, persona_nodes, gate_strength,
                              self._modulation_cfg, mode=self._mode)

        # 6. rerank
        return self._reranker.rerank(candidates, signal, mode=self._mode)
