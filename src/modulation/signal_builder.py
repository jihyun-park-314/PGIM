"""
signal_builder.py
-----------------
Builds a ModulationSignal from persona graph nodes + short-term intent.

Output is a structured signal dict — NOT final scores.
Reranker consumes this signal to compute deltas.

v1 (B-lite):
  - concept_priority config: KuaiSAR cat_l4 > cat_l3 > cat_l2 > cat_l1 > service
  - concept_exclude_types: drop service:* from boost
  - finer persona blend: pull cat_l3/l4 from persona into aligned boost

v2 additions:
  - ContextualConceptSignal: per-concept weighted score
    - persona_score  = weight × source_alignment(node, current_source)
    - intent_score   = goal_weight (uniform 1.0 per goal concept)
    - combined_score = reason-conditioned blend(persona_score, intent_score)
  - Reranker uses combined_score × idf(c) instead of uniform overlap ratio
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src.intent.concept_roles import get_ontology_zone

logger = logging.getLogger(__name__)

# ── IDF cache (loaded once per process) ──────────────────────────────
_idf_cache: dict[str, dict[str, float]] = {}   # path -> {concept_id: idf_weight}


def _load_idf(idf_path: str) -> dict[str, float]:
    if idf_path not in _idf_cache:
        p = Path(idf_path)
        if p.exists():
            df = pd.read_parquet(p)
            _idf_cache[idf_path] = dict(zip(df["concept_id"], df["idf_weight"]))
            logger.info("Loaded concept IDF: %d entries from %s", len(_idf_cache[idf_path]), p)
        else:
            logger.warning("concept IDF not found: %s — IDF downweight disabled", p)
            _idf_cache[idf_path] = {}
    return _idf_cache[idf_path]


def build_concept_idf(
    item_concepts_path: str | Path,
    out_path: str | Path,
    idf_floor: float = 0.1,
) -> pd.DataFrame:
    """
    Precompute IDF weights for all concepts and save to parquet.
    idf_weight = log(N / df) / log(N), clipped to [idf_floor, 1.0]
    """
    df = pd.read_parquet(item_concepts_path)
    N = df["item_id"].nunique()
    df_count = df.groupby("concept_id")["item_id"].nunique().reset_index()
    df_count.columns = ["concept_id", "doc_freq"]
    df_count["idf_weight"] = df_count["doc_freq"].apply(
        lambda df_val: max(idf_floor, min(1.0, math.log(N / df_val) / math.log(N)))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_count.to_parquet(out_path, index=False)
    logger.info(
        "Saved concept IDF: %d concepts  idf range [%.3f, %.3f] -> %s",
        len(df_count), df_count["idf_weight"].min(), df_count["idf_weight"].max(), out_path,
    )
    return df_count


# ── v2: ContextualConceptSignal ───────────────────────────────────────

@dataclass
class ContextualConceptSignal:
    """
    Per-concept weighted signal for v2 reranking.

    persona_score:  weight × source_alignment — how well this concept
                    matches the user's long-term preference AND current source context.
    intent_score:   1.0 if concept is in intent goal_concepts, 0.0 otherwise.
    combined_score: reason-conditioned blend used by reranker for delta.
    """
    concept_id: str
    persona_score: float
    intent_score: float
    combined_score: float


def _source_alignment(
    source_rec_frac: float,
    source_search_frac: float,
    current_source: str,
) -> float:
    """
    Alignment between this concept's source profile and the current context source.
    Returns value in [0.5, 1.0]:
      - perfect alignment (current_source == dominant source): 1.0
      - unknown / neutral: 0.75
      - opposite source: 0.5
    """
    if current_source == "rec":
        # how much this concept is associated with rec
        return 0.5 + 0.5 * source_rec_frac
    elif current_source == "search":
        return 0.5 + 0.5 * source_search_frac
    else:
        return 0.75  # unknown source → neutral alignment


def _get_zone_multiplier(
    concept_id: str,
    reason: str,
    zone_weights_cfg: dict,
) -> float:
    """
    Return the ontology-zone-based multiplier for a concept's persona_score.

    Looks up: zone_weights_cfg[reason][zone] -> float multiplier.
    Falls back to 1.0 if reason or zone not found in config.
    NoiseMeta always returns 0.0 regardless of config (hard block).
    """
    zone = get_ontology_zone(concept_id)
    if zone == "NoiseMeta":
        return 0.0
    reason_cfg = zone_weights_cfg.get(reason, zone_weights_cfg.get("unknown", {}))
    return float(reason_cfg.get(zone, 1.0))


def build_contextual_signals(
    persona_nodes: list[dict],
    goal_concepts: list[str],
    current_source: str,
    reason: str,
    modulation_cfg: dict,
    exclude_types: list[str],
    also_exclude: set[str],
    idf: dict[str, float],
    idf_floor: float,
) -> dict[str, ContextualConceptSignal]:
    """
    Build per-concept ContextualConceptSignal for all concepts in
    persona top-K union intent goal_concepts.

    Returns dict[concept_id -> ContextualConceptSignal].
    """
    # reason-conditioned blend weights
    blend_cfg: dict = modulation_cfg.get("v2_blend", {})
    reason_blend = blend_cfg.get(reason, blend_cfg.get("unknown", {}))
    persona_weight = float(reason_blend.get("persona_weight", 0.5))
    intent_weight  = float(reason_blend.get("intent_weight", 0.5))

    # ontology zone weights (optional; disabled if not in config)
    zone_cfg = modulation_cfg.get("zone_weights", {})
    zone_weighting_enabled = bool(zone_cfg.get("enabled", False))

    # gather all candidate concepts
    persona_sorted = sorted(persona_nodes, key=lambda n: n["weight"], reverse=True)
    top_k = int(modulation_cfg.get("max_boost_concepts", 10))
    persona_top = persona_sorted[:top_k * 2]  # take extra for filtering

    goal_set = set(goal_concepts)
    ex_set   = set(exclude_types)

    signals: dict[str, ContextualConceptSignal] = {}

    # --- from persona ---
    for node in persona_top:
        cid = node["concept_id"]
        ctype = cid.split(":")[0]
        if ctype in ex_set or cid in also_exclude:
            continue

        src_rec_frac    = float(node.get("source_rec_frac", 0.5))
        src_search_frac = float(node.get("source_search_frac", 0.5))
        src_align = _source_alignment(src_rec_frac, src_search_frac, current_source)

        p_score = float(node["weight"]) * src_align

        # ── ontology zone multiplier (applied to persona score only) ──
        if zone_weighting_enabled:
            p_score = p_score * _get_zone_multiplier(cid, reason, zone_cfg)

        i_score = 1.0 if cid in goal_set else 0.0

        combined = persona_weight * p_score + intent_weight * i_score
        # scale by IDF specificity
        idf_w = idf.get(cid, idf_floor) if idf else 1.0
        combined = combined * idf_w

        signals[cid] = ContextualConceptSignal(
            concept_id=cid,
            persona_score=round(p_score, 6),
            intent_score=i_score,
            combined_score=round(combined, 6),
        )

    # --- from intent (concepts not already in persona top) ---
    for cid in goal_concepts:
        if cid in signals:
            # update intent_score if it was persona-only before
            s = signals[cid]
            if s.intent_score == 0.0:
                i_score = 1.0
                idf_w = idf.get(cid, idf_floor) if idf else 1.0
                combined = persona_weight * s.persona_score + intent_weight * i_score
                combined = combined * idf_w
                signals[cid] = ContextualConceptSignal(
                    concept_id=cid,
                    persona_score=s.persona_score,
                    intent_score=i_score,
                    combined_score=round(combined, 6),
                )
            continue

        ctype = cid.split(":")[0]
        if ctype in ex_set or cid in also_exclude:
            continue

        idf_w = idf.get(cid, idf_floor) if idf else 1.0
        i_score = 1.0
        # intent-only concept: zone multiplier not applied
        # (zone weighting targets persona signal, not pure intent signal)
        combined = intent_weight * i_score * idf_w

        signals[cid] = ContextualConceptSignal(
            concept_id=cid,
            persona_score=0.0,
            intent_score=i_score,
            combined_score=round(combined, 6),
        )

    return signals


# ── ModulationSignal ──────────────────────────────────────────────────

@dataclass
class ModulationSignal:
    user_id: str
    target_index: int
    boost_concepts: list[str]        # concept_ids to boost (v1 compat + v2 top-K)
    suppress_concepts: list[str]     # concept_ids to suppress
    filter_constraints: dict         # parsed from constraints_json
    gate_strength: float             # 0~1, computed by gate.py
    reason_type: str                 # deviation_reason
    confidence: float
    ttl_steps: int
    # v2: per-concept weighted signals (empty dict = v1 mode)
    concept_signals: dict[str, ContextualConceptSignal] = field(default_factory=dict)
    # debug: which concepts came from which source
    debug_info: dict | None = None

    def to_record(self) -> dict:
        return {
            "user_id": self.user_id,
            "target_index": self.target_index,
            "boost_concepts": self.boost_concepts,
            "suppress_concepts": self.suppress_concepts,
            "filter_constraints": json.dumps(self.filter_constraints),
            "gate_strength": self.gate_strength,
            "deviation_reason": self.reason_type,
            "confidence": self.confidence,
            "ttl_steps": self.ttl_steps,
        }


# ── Concept priority helpers ──────────────────────────────────────────

def _concept_type(concept_id: str) -> str:
    return concept_id.split(":")[0]


def _concept_priority_fn(priority_cfg: dict) -> "callable[[str], int]":
    """Return a sort-key function: lower = higher priority."""
    def _priority(concept_id: str) -> int:
        ctype = _concept_type(concept_id)
        return priority_cfg.get(ctype, 50)
    return _priority


def _filter_excluded(
    concepts: list[str],
    exclude_types: list[str],
    also_exclude: set[str] | None = None,
) -> list[str]:
    """Remove concepts whose type is in exclude_types or id is in also_exclude."""
    ex_set = set(exclude_types)
    also = also_exclude or set()
    return [c for c in concepts if _concept_type(c) not in ex_set and c not in also]


def _sort_by_priority(concepts: list[str], priority_cfg: dict) -> list[str]:
    fn = _concept_priority_fn(priority_cfg)
    return sorted(concepts, key=fn)


def _finer_persona_concepts(
    persona_top_ids: list[str],
    existing_boost_set: set[str],
    exclude_types: list[str],
    also_exclude: set[str],
    fine_types: tuple[str, ...],
    top_n: int,
) -> list[str]:
    """
    Pull top_n persona concepts of fine_types not already in boost.
    """
    fine = [
        c for c in persona_top_ids
        if _concept_type(c) in fine_types
        and c not in existing_boost_set
        and c not in also_exclude
        and _concept_type(c) not in exclude_types
    ][:top_n]
    return fine


# ── Main build_signal ─────────────────────────────────────────────────

def build_signal(
    intent_record: dict,
    persona_nodes: list[dict],
    gate_strength: float,
    modulation_cfg: dict,
    mode: str = "graph_conditioned_full",
) -> ModulationSignal:
    """
    Build modulation signal from intent + persona.

    v2 mode: if modulation_cfg has 'v2_blend' section, builds ContextualConceptSignal
    per concept (source-aware, reason-conditioned weighted blend).
    Falls back to v1 behavior if 'v2_blend' absent.

    mode controls which sources are active:
        persona_only_rerank  — boost from persona top concepts only; intent zeroed
        intent_only_rerank   — boost from intent goal_concepts only; persona zeroed
        graph_conditioned_full — both sources, reason-conditioned combination
        (backbone_only / others) — empty signal, handled upstream
    """
    # Priority: routed_reason (unknown_router) > recalibrated_reason (exploration_recalibrator)
    # > deviation_reason (original LLM output). Each layer is backward-compatible.
    reason = (
        intent_record.get("routed_reason")
        or intent_record.get("recalibrated_reason")
        or intent_record.get("deviation_reason", "unknown")
    )

    # Stage 3: prefer validated_goal_concepts (Stage 2 output) over raw goal_concepts.
    # validated_goal_concepts is set by grounded_selector and is grounded to the
    # backbone candidate activation bank.  Raw LLM goals are preserved only for
    # diagnostics/ablation — they must NOT enter modulation directly.
    _validated = intent_record.get("validated_goal_concepts")
    _raw_goal  = intent_record.get("goal_concepts")
    if _validated is not None:
        goal_concepts: list[str] = list(_validated)
        _raw_llm_goals_raw = intent_record.get("raw_llm_goals")
        _raw_llm_goals: list[str] = (
            list(_raw_llm_goals_raw) if _raw_llm_goals_raw is not None
            else (list(_raw_goal) if _raw_goal is not None else [])
        )
    else:
        # backward-compat: no Stage 2 present (heuristic or old cache)
        goal_concepts = list(_raw_goal) if _raw_goal is not None else []
        _raw_llm_goals = list(goal_concepts)

    constraints_raw = intent_record.get("constraints_json", "{}")
    try:
        filter_constraints = json.loads(constraints_raw) if isinstance(constraints_raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        filter_constraints = {}

    persona_sorted = sorted(persona_nodes, key=lambda n: n["weight"], reverse=True)
    persona_top_ids = [n["concept_id"] for n in persona_sorted[:10]]

    # ── config reads ──────────────────────────────────────────────────
    priority_cfg: dict = modulation_cfg.get("concept_priority", {})
    exclude_types: list[str] = modulation_cfg.get("concept_exclude_types", [])
    max_boost: int = int(modulation_cfg.get("max_boost_concepts", 10))

    persona_blend_cfg = modulation_cfg.get("persona_blend", {})
    aligned_persona_top_n   = int(persona_blend_cfg.get("aligned_top_n", 0))
    unknown_persona_top_n   = int(persona_blend_cfg.get("unknown_top_n", 0))
    aligned_finer_top_n     = int(persona_blend_cfg.get("aligned_finer_top_n", 0))
    exploration_finer_top_n = int(persona_blend_cfg.get("exploration_finer_top_n", 0))

    _ALWAYS_EXCLUDE = {"price_band:unknown"}
    _FINE_TYPES = ("cat_l3", "cat_l4")

    # ── v2: current_source from intent record ─────────────────────────
    current_source: str = intent_record.get("current_source", "unknown")
    use_v2 = "v2_blend" in modulation_cfg

    # ── IDF (shared for v1 and v2) ───────────────────────────────────
    idf: dict[str, float] = {}
    idf_floor: float = 0.1
    idf_cfg = modulation_cfg.get("concept_idf", {})
    if idf_cfg.get("enabled", False):
        idf_path = idf_cfg.get("idf_path", "")
        idf_floor = float(idf_cfg.get("idf_floor", 0.1))
        if idf_path:
            idf = _load_idf(idf_path)

    # ── MODE: persona_only ────────────────────────────────────────────
    # TRUE NEUTRAL PATH: must not use any intent-derived fields.
    # deviation_reason, current_source, source_shift_flag, recent_source_*
    # are all intent outputs and must be excluded here to keep heuristic
    # vs LLM ablation comparable.
    if mode == "persona_only_rerank":
        candidates = _filter_excluded(persona_top_ids, exclude_types, _ALWAYS_EXCLUDE)
        if priority_cfg:
            candidates = _sort_by_priority(candidates, priority_cfg)
        boost_concepts = candidates[:5]

        concept_signals: dict[str, ContextualConceptSignal] = {}
        if use_v2:
            # Explicit bypass: persona_weight=1.0, intent_weight=0.0,
            # source alignment fixed to neutral (0.75) — no intent fields used.
            persona_sorted_v2 = sorted(persona_nodes, key=lambda n: n["weight"], reverse=True)
            top_k = int(modulation_cfg.get("max_boost_concepts", 10))
            ex_set = set(exclude_types)
            for node in persona_sorted_v2[:top_k * 2]:
                cid = node["concept_id"]
                ctype = cid.split(":")[0]
                if ctype in ex_set or cid in _ALWAYS_EXCLUDE:
                    continue
                p_score = float(node["weight"]) * 0.75  # neutral source alignment
                idf_w = idf.get(cid, idf_floor) if idf else 1.0
                concept_signals[cid] = ContextualConceptSignal(
                    concept_id=cid,
                    persona_score=round(p_score, 6),
                    intent_score=0.0,
                    combined_score=round(p_score * idf_w, 6),
                )

        debug = {"source": "persona", "raw": persona_top_ids[:5], "filtered": boost_concepts}
        return ModulationSignal(
            user_id=intent_record["user_id"],
            target_index=int(intent_record["target_index"]),
            boost_concepts=boost_concepts,
            suppress_concepts=[],
            filter_constraints={},
            gate_strength=1.0,  # fixed neutral — ignore intent-derived gate_strength
            reason_type="persona_only",  # neutral sentinel — not intent-derived
            confidence=float(intent_record.get("confidence", 0.5)),
            ttl_steps=int(intent_record.get("ttl_steps", 1)),
            concept_signals=concept_signals,
            debug_info=debug,
        )

    # ── MODE: intent_only ─────────────────────────────────────────────
    if mode == "intent_only_rerank":
        def _legacy_priority(concept_id: str) -> int:
            ctype = _concept_type(concept_id)
            if ctype in ("item_form", "skin_type"): return 0
            if ctype == "price_band" and concept_id != "price_band:unknown": return 1
            return 99

        candidates = _filter_excluded(goal_concepts, exclude_types, _ALWAYS_EXCLUDE)
        candidates = [c for c in candidates if _concept_type(c) != "brand"]
        if priority_cfg:
            candidates = _sort_by_priority(candidates, priority_cfg)
        else:
            candidates = sorted(candidates, key=_legacy_priority)

        boost_concepts = candidates[:max_boost]
        if not boost_concepts:
            for ctype in ("item_form", "skin_type", "price_band"):
                for v in filter_constraints.get(ctype, []):
                    if v != "price_band:unknown" and v not in boost_concepts:
                        boost_concepts.append(v)

        concept_signals = {}
        if use_v2:
            # intent_only: persona_score zeroed, intent_score only
            for cid in boost_concepts:
                idf_w = idf.get(cid, idf_floor) if idf else 1.0
                concept_signals[cid] = ContextualConceptSignal(
                    concept_id=cid,
                    persona_score=0.0,
                    intent_score=1.0,
                    combined_score=round(1.0 * idf_w, 6),
                )

        debug = {
            "source": "intent",
            "raw_llm_goals": _raw_llm_goals,          # Stage 1 output
            "validated_goal_concepts": goal_concepts,  # Stage 2 output (entered modulation)
            "excluded": [c for c in goal_concepts if c not in boost_concepts],
            "boost": boost_concepts,
        }
        return ModulationSignal(
            user_id=intent_record["user_id"],
            target_index=int(intent_record["target_index"]),
            boost_concepts=boost_concepts,
            suppress_concepts=[],
            filter_constraints=filter_constraints,
            gate_strength=gate_strength,
            reason_type=reason,
            confidence=float(intent_record.get("confidence", 0.35)),
            ttl_steps=int(intent_record.get("ttl_steps", 1)),
            concept_signals=concept_signals,
            debug_info=debug,
        )

    # ── MODE: graph_conditioned_full ──────────────────────────────────

    # v2: build contextual signals first (used below)
    concept_signals = {}
    if use_v2:
        concept_signals = build_contextual_signals(
            persona_nodes=persona_nodes,
            goal_concepts=goal_concepts,
            current_source=current_source,
            reason=reason,
            modulation_cfg=modulation_cfg,
            exclude_types=exclude_types,
            also_exclude=_ALWAYS_EXCLUDE,
            idf=idf,
            idf_floor=idf_floor,
        )

    # v1 boost_concepts (kept for backward compat and as top-K list for reranker)
    def _legacy_boost_priority(concept_id: str) -> int:
        ctype = _concept_type(concept_id)
        if ctype in ("item_form", "skin_type"): return 0
        if ctype == "price_band" and concept_id != "price_band:unknown": return 1
        if ctype == "category": return 2
        return 99

    candidates = _filter_excluded(goal_concepts, exclude_types, _ALWAYS_EXCLUDE)
    candidates = [c for c in candidates if _concept_type(c) != "brand"]
    if priority_cfg:
        candidates = _sort_by_priority(candidates, priority_cfg)
    else:
        candidates = sorted(candidates, key=_legacy_boost_priority)

    boost_concepts = candidates[:max_boost]
    if not boost_concepts:
        for ctype in ("item_form", "skin_type", "price_band"):
            for v in filter_constraints.get(ctype, []):
                if v != "price_band:unknown" and v not in boost_concepts:
                    boost_concepts.append(v)

    intent_contributed = list(boost_concepts)
    persona_contributed: list[str] = []

    # ── reason-conditioned persona blend ─────────────────────────────
    suppress_concepts: list[str] = []

    if reason in ("aligned", "aligned_soft"):
        # aligned_soft: treat like aligned — persona prior dominates,
        # recent intent is a mild same-taste flavor (no contrastive boost)
        existing = set(boost_concepts)
        if aligned_finer_top_n > 0:
            finer = _finer_persona_concepts(
                persona_top_ids, existing, exclude_types, _ALWAYS_EXCLUDE,
                _FINE_TYPES, aligned_finer_top_n,
            )
            boost_concepts = boost_concepts + finer
            persona_contributed.extend(finer)
            existing = set(boost_concepts)
        if aligned_persona_top_n > 0:
            persona_extra = _filter_excluded(persona_top_ids, exclude_types, _ALWAYS_EXCLUDE)
            persona_extra = [c for c in persona_extra if c not in existing][:aligned_persona_top_n]
            boost_concepts = boost_concepts + persona_extra
            persona_contributed.extend(persona_extra)

    elif reason in ("task_focus", "task_focus_like"):
        # task_focus_like: narrow boost, TTL short, persona override suppressed
        goal_set = set(boost_concepts)
        suppress_concepts = [
            c for c in persona_top_ids
            if c not in goal_set and not c.startswith("category:")
        ][:3]

    elif reason in ("budget_shift", "budget_like"):
        # budget_like: price/format modulation; suppress conflicting price_band
        recent_price = next(
            (c for c in goal_concepts
             if c.startswith("price_band:") and c != "price_band:unknown"), None,
        )
        suppress_concepts = [
            c for c in persona_top_ids
            if c.startswith("price_band:") and c != "price_band:unknown" and c != recent_price
        ]

    elif reason == "exploration":
        # legacy exploration (no recalibration applied): keep existing behavior
        if exploration_finer_top_n > 0:
            existing = set(boost_concepts)
            finer = _finer_persona_concepts(
                persona_top_ids, existing, exclude_types, _ALWAYS_EXCLUDE,
                _FINE_TYPES, exploration_finer_top_n,
            )
            boost_concepts = boost_concepts + finer
            persona_contributed.extend(finer)

    elif reason == "true_exploration":
        # true_exploration: contrastive recent concepts allowed.
        # Moderate trust — persona not fully overridden.
        # Finer persona concepts added to preserve some persona anchor.
        if exploration_finer_top_n > 0:
            existing = set(boost_concepts)
            finer = _finer_persona_concepts(
                persona_top_ids, existing, exclude_types, _ALWAYS_EXCLUDE,
                _FINE_TYPES, exploration_finer_top_n,
            )
            boost_concepts = boost_concepts + finer
            persona_contributed.extend(finer)

    elif reason in ("exploration_unclear", "unknown"):
        # exploration_unclear: conservative — near-aligned handling
        if unknown_persona_top_n > 0:
            existing = set(boost_concepts)
            persona_extra = _filter_excluded(persona_top_ids, exclude_types, _ALWAYS_EXCLUDE)
            persona_extra = [c for c in persona_extra if c not in existing][:unknown_persona_top_n]
            boost_concepts = boost_concepts + persona_extra
            persona_contributed.extend(persona_extra)

    boost_concepts = boost_concepts[:max_boost]

    # v2: if concept_signals built, ensure boost_concepts are covered
    # and sort by combined_score (highest first)
    if use_v2 and concept_signals:
        # add any boost_concept not yet in signals (intent-only fallback)
        for cid in boost_concepts:
            if cid not in concept_signals:
                idf_w = idf.get(cid, idf_floor) if idf else 1.0
                concept_signals[cid] = ContextualConceptSignal(
                    concept_id=cid,
                    persona_score=0.0,
                    intent_score=1.0,
                    combined_score=round(1.0 * idf_w, 6),
                )
        # re-rank boost_concepts by combined_score descending
        boost_concepts = sorted(
            boost_concepts,
            key=lambda c: concept_signals.get(c, ContextualConceptSignal(c, 0, 0, 0)).combined_score,
            reverse=True,
        )[:max_boost]

    debug = {
        "reason": reason,
        "current_source": current_source,
        "raw_llm_goals": _raw_llm_goals,          # Stage 1 output (before Stage 2)
        "validated_goal_concepts": goal_concepts,  # Stage 2 output (what entered modulation)
        "intent_contributed": intent_contributed,
        "persona_contributed": persona_contributed,
        "final_boost": boost_concepts,
        "suppress": suppress_concepts,
        "boost_ctypes": [_concept_type(c) for c in boost_concepts],
        "v2_top_signals": [
            {"concept_id": c, "combined": round(concept_signals[c].combined_score, 4),
             "persona": round(concept_signals[c].persona_score, 4),
             "intent": concept_signals[c].intent_score}
            for c in boost_concepts[:5] if c in concept_signals
        ] if use_v2 else [],
    }

    return ModulationSignal(
        user_id=intent_record["user_id"],
        target_index=int(intent_record["target_index"]),
        boost_concepts=boost_concepts,
        suppress_concepts=suppress_concepts,
        filter_constraints=filter_constraints,
        gate_strength=gate_strength,
        reason_type=reason,
        confidence=float(intent_record.get("confidence", 0.35)),
        ttl_steps=int(intent_record.get("ttl_steps", 1)),
        concept_signals=concept_signals,
        debug_info=debug,
    )
