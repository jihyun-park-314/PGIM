"""
Feature extraction utilities for candidate-aware soft scorer.
"""

from __future__ import annotations

from dataclasses import dataclass


def split_candidate_semantic_signature(
    item_concepts: list[str],
    get_ontology_zone_fn,
) -> dict[str, list[str]]:
    raw = list(dict.fromkeys(item_concepts))
    core, anchor, pctx, noise = [], [], [], []
    for c in raw:
        z = get_ontology_zone_fn(c)
        if z == "SemanticCore":
            core.append(c)
        elif z == "SemanticAnchor":
            anchor.append(c)
        elif z == "ProductContext":
            pctx.append(c)
        elif z == "NoiseMeta":
            noise.append(c)
    return {
        "semantic_core_concepts": core,
        "semantic_anchor_concepts": anchor,
        "product_context_concepts": pctx,
        "noise_meta_concepts": noise,
        "raw_item_concepts": raw,
    }


def top_persona_concepts(persona_nodes: list[dict], top_n: int = 10) -> list[str]:
    ordered = sorted(persona_nodes, key=lambda x: float(x.get("weight", 0.0)), reverse=True)
    return [str(n.get("concept_id", "")) for n in ordered[:top_n] if n.get("concept_id")]


def weighted_persona_map(persona_nodes: list[dict], top_n: int = 20) -> dict[str, float]:
    ordered = sorted(persona_nodes, key=lambda x: float(x.get("weight", 0.0)), reverse=True)[:top_n]
    out = {}
    total = 0.0
    for n in ordered:
        cid = str(n.get("concept_id", ""))
        w = float(n.get("weight", 0.0))
        if not cid:
            continue
        out[cid] = max(out.get(cid, 0.0), w)
        total += w
    if total <= 0:
        return out
    return {k: v / total for k, v in out.items()}


@dataclass(frozen=True)
class SoftMatchFeatures:
    goal_overlap_ratio: float
    persona_overlap_weighted: float
    matched_goal_count: int
    matched_persona_count: int
    semantic_density: float

