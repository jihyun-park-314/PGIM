"""
pgim_ontology.py
----------------
Lightweight ontology loader and query interface for PGIM.

Loads the YAML ontology skeleton and exposes zone/subzone lookups, policy
checks, and relation queries. Does NOT implement ontology reasoning — it is
purely a config-driven query layer.

Usage:
    from src.ontology.pgim_ontology import PGIMOntology
    o = PGIMOntology("config/ontology/pgim_movies_tv_v0_2.yaml")
    o.get_zone("category:drama")        # "SemanticCore"
    o.get_subzone("category:drama")     # "Genre"
    o.is_allowed("category:drama", "long_term_persona")  # True
"""

from __future__ import annotations

import yaml
from src.intent.concept_roles import get_role

# ── Role → (Zone, Subzone) static mapping ─────────────────────────────────────
# FRANCHISE and EDITION are fully resolved roles (no disambiguation needed).
# Remaining COLLECTION entries (all_*_titles, editorial picks, etc.) still go
# through _disambiguate_collection() for NoiseMeta/StorefrontTag assignment.

_ROLE_TO_ZONE_SUBZONE: dict[str, tuple[str, str]] = {
    "STRONG_SEMANTIC":  ("SemanticCore",   "Genre"),
    "WEAK_DESCRIPTOR":  ("SemanticCore",   "MoodTone"),
    "PLATFORM":         ("ProductContext", "AvailabilityChannel"),
    "NAVIGATION":       ("NoiseMeta",      "NavigationTag"),
    "PROMO_DEAL":       ("NoiseMeta",      "PromoTag"),
    "PUBLISHER":        ("ProductContext", "PublisherStudio"),
    "FORMAT_META":      ("ProductContext", "Format"),
    "FRANCHISE":        ("SemanticAnchor", "Franchise"),
    "EDITION":          ("ProductContext", "Edition"),
    "UMBRELLA":         ("NoiseMeta",      "NavigationTag"),
    "PERSON":           ("SemanticAnchor", "Creator"),
    "TEMPORAL":         ("SemanticAnchor", "EraPeriod"),
    "GEO_LANG":         ("SemanticAnchor", "Language"),
    "AGE_DEMO":         ("SemanticCore",   "Audience"),
    "AWARD":            ("SemanticAnchor", "EraPeriod"),
    "NETWORK_CHANNEL":  ("SemanticAnchor", "Creator"),
}

# ── COLLECTION disambiguation keyword sets ────────────────────────────────────
_FRANCHISE_KEYWORDS = frozenset({
    "star_wars", "harry_potter", "x-men", "batman", "superman",
    "terminator", "alien_saga", "wall-e", "twilight_zone",
    "blues_brothers",
})

_EDITION_KEYWORDS = frozenset({
    "collector", "edition", "special_edition", "box_set", "two-disc",
    "ultimate", "legacy_collection", "treasures", "vista_series",
})

_STOREFRONT_KEYWORDS = frozenset({
    "all_", "_titles", "_deals", "_store", "top_sellers",
})

_PUBLISHER_STUDIO_KEYWORDS = frozenset({
    "bbc", "hbo", "mtv", "fx_shows", "showtime", "sundance",
    "sci_fi_channel", "a&e", "new_yorker", "sony", "mgm", "fox",
    "lionsgate", "universal", "disney", "warner_archive",
    "american_film_institute",
})


def _disambiguate_collection(concept_id: str) -> tuple[str, str]:
    """
    Apply COLLECTION disambiguation rules in order.
    Returns (zone, subzone).
    """
    # Strip 'category:' prefix for matching
    name = concept_id.removeprefix("category:")

    # Rule 1: franchise
    for kw in _FRANCHISE_KEYWORDS:
        if kw in name:
            return ("SemanticAnchor", "Franchise")

    # Rule 2: edition
    for kw in _EDITION_KEYWORDS:
        if kw in name:
            return ("ProductContext", "Edition")

    # Rule 3: storefront patterns (substring match for prefix/suffix patterns)
    if (name.startswith("all_") or name.endswith("_titles") or
            name.endswith("_deals") or name.endswith("_store") or
            "top_sellers" in name):
        return ("NoiseMeta", "StorefrontTag")

    # Rule 4: studio/network names
    for kw in _PUBLISHER_STUDIO_KEYWORDS:
        if kw in name:
            return ("ProductContext", "PublisherStudio")

    # Rule 5: default — conservative
    return ("NoiseMeta", "StorefrontTag")


class PGIMOntology:
    """
    Lightweight loader and query interface for the PGIM YAML ontology.

    Parameters
    ----------
    yaml_path : str
        Path to the ontology YAML file (e.g. 'config/ontology/pgim_movies_tv_v0_2.yaml').
    """

    def __init__(self, yaml_path: str) -> None:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            self._data: dict = yaml.safe_load(fh)

        # Pre-build flat policy lookup: context -> {primary, secondary, tertiary, excluded}
        self._policy: dict[str, dict[str, set[str]]] = {}
        self._load_policy()

        # Propagation policy: relation -> {policy, strength, note}
        self._propagation: dict[str, dict] = self._load_propagation()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_policy(self) -> None:
        usage = self._data.get("usage_policy", {})

        # long_term_persona is a top-level key
        for ctx in ("long_term_persona",):
            if ctx in usage:
                self._policy[ctx] = self._parse_policy_block(usage[ctx])

        # short_term_reason_usage contains the rest
        short_term = usage.get("short_term_reason_usage", {})
        for ctx in ("aligned", "exploration", "task_focus", "budget_shift", "unknown"):
            if ctx in short_term:
                self._policy[ctx] = self._parse_policy_block(short_term[ctx])

    def _load_propagation(self) -> dict[str, dict]:
        """Load propagation_policy as relation-name → {policy, strength, note}."""
        result: dict[str, dict] = {}
        for entry in self._data.get("propagation_policy", []) or []:
            rel = entry.get("relation")
            if rel:
                result[rel] = {
                    "policy":   entry.get("policy", ""),
                    "strength": float(entry.get("strength", 0.0)),
                    "note":     entry.get("note", ""),
                }
        return result

    @staticmethod
    def _parse_policy_block(block: dict) -> dict[str, set[str]]:
        # primary_allowed — also absorb v0.2 task_focus split keys
        primary: set[str] = set(block.get("primary_allowed", []) or [])
        primary |= set(block.get("content_focus_primary", []) or [])
        primary |= set(block.get("purchase_focus_primary", []) or [])
        return {
            "primary":   primary,
            "secondary": set(block.get("secondary_allowed", []) or []),
            "tertiary":  set(block.get("tertiary_allowed", []) or []),
            "excluded":  set(block.get("excluded", []) or []),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_zone(self, concept_id: str) -> str:
        """
        Returns the top-level ontology zone for a concept.

        Returns
        -------
        str
            One of: "SemanticCore" | "SemanticAnchor" | "ProductContext" |
                    "NoiseMeta" | "Unknown"
        """
        role = get_role(concept_id)
        if role == "COLLECTION":
            zone, _ = _disambiguate_collection(concept_id)
            return zone
        entry = _ROLE_TO_ZONE_SUBZONE.get(role)
        if entry is None:
            return "Unknown"
        return entry[0]

    def get_subzone(self, concept_id: str) -> str:
        """
        Returns the finer-grained subzone for a concept.

        Returns
        -------
        str
            One of: "Genre" | "Subgenre" | "Theme" | "MoodTone" | "Audience" |
                    "Franchise" | "Creator" | "Language" | "CountryRegion" |
                    "EraPeriod" | "Format" | "Edition" | "PriceBand" |
                    "AvailabilityChannel" | "PublisherStudio" | "NavigationTag" |
                    "PromoTag" | "StorefrontTag" | "Unknown"
        """
        role = get_role(concept_id)
        if role == "COLLECTION":
            _, subzone = _disambiguate_collection(concept_id)
            return subzone
        entry = _ROLE_TO_ZONE_SUBZONE.get(role)
        if entry is None:
            return "Unknown"
        return entry[1]

    def is_allowed(self, concept_id: str, context: str) -> bool:
        """
        Return True if concept_id is allowed for a given usage context.

        Policy logic
        ------------
        - If subzone is in primary_allowed or secondary_allowed for this context → True
        - If subzone is in excluded → False
        - If not mentioned:
            - False for: long_term_persona, exploration, unknown
            - True  for: aligned, task_focus, budget_shift

        Parameters
        ----------
        concept_id : str
        context : str
            One of: "long_term_persona" | "aligned" | "exploration" |
                    "task_focus" | "budget_shift" | "unknown"
        """
        subzone = self.get_subzone(concept_id)
        policy = self._policy.get(context)

        if policy is None:
            # Unknown context: conservative — only allow if subzone is known semantic core
            return False

        if subzone in policy["excluded"]:
            return False
        if (subzone in policy["primary"]
                or subzone in policy["secondary"]
                or subzone in policy.get("tertiary", set())):
            return True

        # Not mentioned — context-dependent default
        _STRICT_CONTEXTS = {"long_term_persona", "exploration", "unknown"}
        return context not in _STRICT_CONTEXTS

    def get_relations(self) -> list[dict]:
        """
        Returns all relation definitions from the YAML as a flat list.

        Each dict has at minimum: 'name', 'domain', 'range'.
        Some have 'description'.
        """
        relations_section = self._data.get("relations", {})
        result: list[dict] = []
        for _group, entries in relations_section.items():
            if isinstance(entries, list):
                result.extend(entries)
        return result

    def get_relation_def(self, relation_name: str) -> dict | None:
        """
        Returns the definition dict for a named relation, or None if not found.

        Parameters
        ----------
        relation_name : str
            E.g. 'has_genre', 'subgenre_of', 'manifestation_of'
        """
        for rel in self.get_relations():
            if rel.get("name") == relation_name:
                return rel
        return None

    def propagation_strength(self, relation_name: str) -> float:
        """
        Return the propagation strength for a relation (0.0 = no propagation).

        Defined in propagation_policy in the YAML.
        Returns 0.0 for unknown relations.
        """
        entry = self._propagation.get(relation_name)
        return entry["strength"] if entry else 0.0

    def propagation_policy(self, relation_name: str) -> str:
        """
        Return the propagation policy label for a relation.

        E.g. 'upward_propagation', 'semantic_projection_allowed',
             'no_core_propagation', 'anchor_propagation', etc.
        Returns empty string for unknown relations.
        """
        entry = self._propagation.get(relation_name)
        return entry["policy"] if entry else ""

    def get_validation_rules(self) -> list[dict]:
        """
        Returns the list of validation rule dicts from the YAML.

        Each dict has 'rule' and 'description' keys.
        """
        return list(self._data.get("validation_rules", []) or [])

    def is_allowed_in_persona_core(self, concept_id: str) -> bool:
        """
        Return True if concept is allowed as primary or secondary in long_term_persona.
        Tertiary (Actor) and excluded return False.
        """
        subzone = self.get_subzone(concept_id)
        policy = self._policy.get("long_term_persona", {})
        if subzone in policy.get("excluded", set()):
            return False
        return (subzone in policy.get("primary", set())
                or subzone in policy.get("secondary", set()))
