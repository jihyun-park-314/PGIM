"""
cache_resolver.py
-----------------
Offline-first intent cache: snapshot -> ShortTermIntent record.

Flow:
    1. Check in-memory cache by (user_id, target_index)
    2. Hit  -> return cached record (source_mode="cache")
    3. Miss -> run interpreter (heuristic or LLM) -> parse -> store -> return

LLM path is activated when:
    - IntentCacheResolver is constructed with use_llm=True (runtime override), OR
    - intent_cfg["llm"]["use_llm"] is True AND openai_client is provided

Heuristic fallback fires automatically when LLM fails or is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.intent.context_extractor import IntentContext
from src.intent.heuristic_interpreter import interpret as heuristic_interpret
from src.intent.llm_interpreter import interpret_with_llm
from src.intent.parser import parse_intent

logger = logging.getLogger(__name__)


class IntentCacheResolver:
    def __init__(
        self,
        intent_cfg: dict,
        use_llm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        intent_cfg : full intent config dict
        use_llm    : runtime override — set True to activate LLM path regardless
                     of intent_cfg["llm"]["use_llm"]. If False, config value is used.
        """
        self._cfg = intent_cfg
        self._cache: dict[tuple[str, int], dict] = {}   # (user_id, target_index) -> record
        # Runtime flag takes precedence over config
        cfg_use_llm = intent_cfg.get("llm", {}).get("use_llm", False)
        self._use_llm: bool = use_llm or cfg_use_llm

    def resolve(
        self,
        ctx: IntentContext,
        persona_summary: str = "",
        openai_client: Optional[Any] = None,
    ) -> dict:
        """
        Return a parsed intent record for the given context.
        Uses cache if available, otherwise runs interpreter.

        When LLM path is active but LLM call fails, llm_interpreter falls back
        internally to a heuristic-style unknown intent (never raises).
        """
        key = (ctx.user_id, ctx.target_index)

        if key in self._cache:
            record = dict(self._cache[key])
            record["source_mode"] = "cache"
            return record

        # Cache miss: run interpreter
        if self._use_llm and openai_client is not None:
            raw = interpret_with_llm(ctx, persona_summary, self._cfg, openai_client)
            source = "llm"
        else:
            raw = heuristic_interpret(ctx, self._cfg)
            source = "heuristic"

        record = parse_intent(raw, ctx.user_id, ctx.target_index, source)
        self._cache[key] = record
        return record

    def load_from_records(self, records: list[dict]) -> None:
        """Pre-populate cache from saved parquet records."""
        for r in records:
            key = (r["user_id"], int(r["target_index"]))
            self._cache[key] = r
        logger.info("IntentCacheResolver: loaded %d cached intents", len(self._cache))

    def all_records(self) -> list[dict]:
        return list(self._cache.values())

    def __len__(self) -> int:
        return len(self._cache)
