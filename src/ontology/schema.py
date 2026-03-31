"""
Ontology dataclasses.
These are plain dataclasses — no DB, no graph engine dependency.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OntologyConcept:
    concept_id: str          # e.g. "category:skin_care"
    concept_type: str        # "category" | "brand" | "price_band" | "item_form" | "skin_type"
    display_name: str        # human-readable label
    parent_concept_id: Optional[str] = None   # None for root concepts
    level: int = 0           # depth from root (root = 0)


@dataclass
class OntologyRelation:
    src_concept_id: str
    dst_concept_id: str
    relation_type: str       # "parent_of" | "sibling_of"
