"""
usage_policy.py
---------------
Thin wrappers that make the PGIM ontology usage policy concrete and callable.

Usage:
    from src.ontology.pgim_ontology import PGIMOntology
    from src.ontology.usage_policy import filter_concepts_by_context, get_allowed_zones

    o = PGIMOntology("config/ontology/pgim_movies_tv_v0_2.yaml")
    allowed = filter_concepts_by_context(["category:drama", "category:prime_video"], "exploration", o)
    zones   = get_allowed_zones("exploration", o)
"""

from __future__ import annotations

from src.ontology.pgim_ontology import PGIMOntology


def filter_concepts_by_context(
    concept_ids: list[str],
    context: str,
    ontology: PGIMOntology,
    tier: str = "primary",
) -> list[str]:
    """
    Filter concept_ids to those allowed for the given usage context.

    Parameters
    ----------
    concept_ids : list[str]
        Candidate concept IDs to filter (e.g. ["category:drama", "category:prime_video"]).
    context : str
        One of: "long_term_persona" | "aligned" | "exploration" |
                "task_focus" | "budget_shift" | "unknown"
    ontology : PGIMOntology
        Loaded ontology instance.
    tier : str
        "primary"   — keep only concepts in primary_allowed.
        "both"      — keep concepts in primary_allowed OR secondary_allowed.
        "all_tiers" — keep concepts in primary, secondary, OR tertiary_allowed.

    Returns
    -------
    list[str]
        Filtered list preserving original order.
    """
    if tier not in ("primary", "both", "all_tiers"):
        raise ValueError(f"tier must be 'primary', 'both', or 'all_tiers', got {tier!r}")

    result: list[str] = []
    for cid in concept_ids:
        subzone = ontology.get_subzone(cid)
        policy  = ontology._policy.get(context)

        if policy is None:
            # Unknown context: skip
            continue

        # Always skip if explicitly excluded
        if subzone in policy["excluded"]:
            continue

        if tier == "primary":
            if subzone in policy["primary"]:
                result.append(cid)
        elif tier == "both":
            if subzone in policy["primary"] or subzone in policy["secondary"]:
                result.append(cid)
        else:  # "all_tiers"
            if (subzone in policy["primary"]
                    or subzone in policy["secondary"]
                    or subzone in policy.get("tertiary", set())):
                result.append(cid)

    return result


def get_allowed_zones(
    context: str,
    ontology: PGIMOntology,
    tier: str = "primary",
) -> set[str]:
    """
    Return the set of allowed subzone names for a given context and tier.

    Parameters
    ----------
    context : str
        One of: "long_term_persona" | "aligned" | "exploration" |
                "task_focus" | "budget_shift" | "unknown"
    ontology : PGIMOntology
        Loaded ontology instance.
    tier : str
        "primary"   — only primary_allowed subzones.
        "both"      — primary_allowed + secondary_allowed subzones.
        "all_tiers" — primary + secondary + tertiary_allowed subzones.

    Returns
    -------
    set[str]
        Set of subzone names (e.g. {"Genre", "Subgenre", "Theme"}).
    """
    if tier not in ("primary", "both", "all_tiers"):
        raise ValueError(f"tier must be 'primary', 'both', or 'all_tiers', got {tier!r}")

    policy = ontology._policy.get(context)
    if policy is None:
        return set()

    if tier == "primary":
        return set(policy["primary"])
    elif tier == "both":
        return set(policy["primary"]) | set(policy["secondary"])
    else:  # "all_tiers"
        return (set(policy["primary"])
                | set(policy["secondary"])
                | set(policy.get("tertiary", set())))
