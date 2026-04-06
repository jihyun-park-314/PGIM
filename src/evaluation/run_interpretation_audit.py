"""
run_interpretation_audit.py
---------------------------
PR3-2 / PR3-2b: offline interpretation quality audit.

Two modes
─────────
reconstruct mode (default, --native not set):
  Reads pre-PR3-1 caches (schema_version=3.0, prompt=v5).
  Reconstructs contrast_with_persona, temporal_cues, evidence_sources, token_usage
  from grounding_diagnostics + llm_raw + interactions.
  Use for: v5 historical baseline.

native mode (--native):
  Reads v6+ caches that already contain all PR3-1 fields natively.
  contrast_with_persona, temporal_cues, evidence_sources, token_usage
  are read directly — no reconstruction needed.
  Use for: PR3-2b v6 quality audit.

Outputs (both modes)
────────────────────
  1. aggregate_report.json  — quantitative distribution report
  2. qualitative_sample.json — 30-50 case deep-dive records
  3. (stdout)               — summary table

Usage — reconstruct (v5 baseline):
  python -m src.evaluation.run_interpretation_audit \\
      --intent-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_v5_2000.parquet \\
      --persona-path data/cache/persona/amazon_movies_tv/persona_graphs_v2.parquet \\
      --interactions-path data/processed/amazon_movies_tv/interactions.parquet \\
      --item-concepts-path data/processed/amazon_movies_tv/item_concepts.parquet \\
      --out-dir results/interpretation_audit_v5 \\
      --n-sample 200 --n-qualitative 40

Usage — native (v6 PR3-2b):
  python -m src.evaluation.run_interpretation_audit \\
      --intent-path data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_v6_500.parquet \\
      --persona-path data/cache/persona/amazon_movies_tv/persona_graphs_v2.parquet \\
      --interactions-path data/processed/amazon_movies_tv/interactions.parquet \\
      --item-concepts-path data/processed/amazon_movies_tv/item_concepts.parquet \\
      --out-dir results/interpretation_audit_v6 \\
      --native \\
      --n-sample 200 --n-qualitative 40
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Field reconstruction helpers ─────────────────────────────────────────────

def _reconstruct_contrast(row: dict, persona_top10: list[str]) -> dict[str, Any]:
    """
    Reconstruct contrast_with_persona from grounding_diagnostics.

    Grounding-verified source: concepts whose activation_count > 0 and are
    in persona_top10 but were NOT selected as validated_goal_concepts.
    These are the concepts that *could* have been goals but were overridden
    by persona conflict suppression.

    LLM-suggested source: llm_raw.contrast_signal_concepts (if present in
    the raw JSON — v5 schema did not expose this field, so this will be empty
    for most v5 records).
    """
    diag = row.get("grounding_diagnostics") or {}
    if isinstance(diag, str):
        try:
            diag = json.loads(diag)
        except Exception:
            diag = {}

    validated = set(row.get("validated_goal_concepts") or [])
    activation_counts: dict[str, int] = diag.get("activation_counts", {})
    steps: list[str] = diag.get("steps", [])

    # Grounding-verified contrast: activated concepts suppressed because they
    # matched persona top-N (v5 grounded_selector: "persona_conflict" step).
    grounded_contrast: dict[str, int] = {}
    persona_top10_set = set(persona_top10)
    for cid, cnt in activation_counts.items():
        if cid in persona_top10_set and cid not in validated:
            # This concept was in the bank AND persona, but not selected.
            # Could be suppressed by persona_conflict step.
            # Only include if a suppression step exists for this concept.
            suppressed_by_step = any(
                "persona_conflict" in step and cid in step
                for step in steps
            )
            if suppressed_by_step:
                grounded_contrast[cid] = cnt

    # LLM-suggested contrast: from llm_raw if it has contrast_signal_concepts
    llm_raw_str = row.get("llm_raw") or ""
    llm_contrast: dict[str, str] = {}
    if llm_raw_str:
        try:
            raw_parsed = json.loads(llm_raw_str)
            csc = raw_parsed.get("contrast_signal_concepts", [])
            if isinstance(csc, list):
                for c in csc:
                    if isinstance(c, str) and c not in grounded_contrast:
                        llm_contrast[c] = "llm"
        except Exception:
            pass

    return {**grounded_contrast, **llm_contrast}


def _infer_evidence_sources(
    row: dict,
    validated: list[str],
    contrast: dict,
) -> list[str]:
    """Infer evidence_sources from existing signals (v5 records lack this field)."""
    ev = []

    # recent_freq: always present when validated_goal_concepts non-empty
    if validated:
        ev.append("recent_freq")

    # temporal_shift: inferred from temporal_cues if available;
    # for v5 records, check if first/second half concept split is detectable
    # via evidence_recent_concepts field (rough proxy).
    temporal_shift = row.get("_temporal_shift_detected", False)
    if temporal_shift:
        ev.append("temporal_shift")

    # persona_contrast: grounded contrast entries exist
    grounded_n = sum(1 for v in contrast.values() if isinstance(v, int))
    if grounded_n > 0:
        ev.append("persona_contrast")
    elif any(v == "llm" for v in contrast.values()):
        # llm-only contrast: still evidence of contrast, but unverified
        ev.append("persona_contrast")

    return ev


def _reconstruct_token_usage(row: dict) -> dict[str, int]:
    """
    Reconstruct token_usage from llm_raw length (char-based proxy).
    Real token counts unavailable for v5 cache — marked with proxy=True.
    """
    llm_raw = row.get("llm_raw") or ""
    raw_model = row.get("raw_model_response_json") or llm_raw
    # rough char→token approximation: English ~4 chars/token
    response_chars = len(raw_model)
    response_tokens_est = max(1, response_chars // 4)
    return {
        "response_chars": response_chars,
        "response_tokens_est": response_tokens_est,
        # system/user prompt chars unavailable in v5 cache
        "proxy": 1,
    }


# ── IntentContext re-derivation ───────────────────────────────────────────────

def _derive_temporal_split(
    user_id: str,
    target_index: int,
    interactions_by_user: dict[str, list[str]],
    item_concepts_by_item: dict[str, list[str]],
    signal_types: set[str],
    window_size: int = 8,
) -> dict:
    """
    Re-derive temporal split from interactions (recent window first/second half).
    Uses the same logic as ContextExtractor._compute_temporal_split().
    Returns {} when insufficient data.
    """
    items_sorted = interactions_by_user.get(user_id, [])
    recent = items_sorted[max(0, target_index - window_size): target_index]
    if len(recent) < 2:
        return {}

    mid = max(1, len(recent) // 2)
    first_half = recent[:mid]
    second_half = recent[mid:]

    def _freq(item_list: list[str]) -> dict[str, int]:
        c: Counter = Counter()
        for iid in item_list:
            for cid in item_concepts_by_item.get(iid, []):
                ctype = cid.split(":")[0]
                if ctype in signal_types:
                    c[cid] += 1
        return dict(c)

    fh = _freq(first_half)
    sh = _freq(second_half)
    fh_top = max(fh, key=fh.get) if fh else None
    sh_top = max(sh, key=sh.get) if sh else None
    shift = bool(fh_top and sh_top and fh_top != sh_top)

    return {
        "shift_detected": shift,
        "first_half_dominant": fh_top,
        "second_half_dominant": sh_top,
        "first_half_freq": fh,
        "second_half_freq": sh,
        "llm_shift_summary": "",  # not available in v5 cache
    }


# ── Build NormalizedInterpretationRecord-like dicts from v5 rows ─────────────

def build_normalized_row(
    row: dict,
    persona_top10: list[str],
    temporal_cues: dict,
) -> dict:
    """
    Build a flat normalized dict equivalent to NormalizedInterpretationRecord.to_audit_export()
    from a v5 cache row + reconstructed fields.
    """
    def _to_list(val: Any) -> list:
        if val is None:
            return []
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return []
        except Exception:
            pass
        try:
            return list(val)
        except Exception:
            return []

    validated_raw = row.get("validated_goal_concepts")
    if not _to_list(validated_raw):
        validated_raw = row.get("goal_concepts")
    validated = _to_list(validated_raw)

    contrast = _reconstruct_contrast(row, persona_top10)

    # temporal: use re-derived if available, else empty
    tc = temporal_cues if temporal_cues else {}
    shift_detected = bool(tc.get("shift_detected", False))
    row["_temporal_shift_detected"] = shift_detected

    evidence_sources = _infer_evidence_sources(row, validated, contrast)
    token_usage = _reconstruct_token_usage(row)

    # Verified contrast (int-valued = grounding-verified)
    verified_contrast = {c: v for c, v in contrast.items() if isinstance(v, int)}
    # LLM-only contrast (str-valued)
    llm_contrast = [c for c, v in contrast.items() if v == "llm"]

    def _to_str(val: Any, max_len: int = 300) -> str:
        if not val or not isinstance(val, str):
            return ""
        return val[:max_len]

    return {
        # identity
        "user_id": row["user_id"],
        "target_index": int(row["target_index"]),
        "has_stage2": bool(row.get("has_stage2", False)),
        # scoring decision
        "deviation_reason": row.get("deviation_reason", "unknown"),
        "confidence": float(row.get("confidence", 0.35)),
        "validated_goal_concepts": validated,
        "context_goals": validated,
        # contrast
        "contrast_verified": verified_contrast,
        "contrast_llm_only": llm_contrast,
        "n_contrast_verified": len(verified_contrast),
        "n_contrast_llm": len(llm_contrast),
        "n_contrast_total": len(contrast),
        # temporal
        "temporal_shift_detected": shift_detected,
        "temporal_first_half_dominant": tc.get("first_half_dominant"),
        "temporal_second_half_dominant": tc.get("second_half_dominant"),
        "temporal_llm_summary": tc.get("llm_shift_summary", ""),
        # evidence
        "evidence_sources": evidence_sources,
        # audit rationale
        "llm_explanation_short": _to_str(row.get("llm_explanation_short"), 200),
        "why_not_aligned": _to_str(row.get("why_not_aligned"), 150),
        "why_exploration": _to_str(row.get("why_exploration"), 150),
        # hygiene
        "goal_hygiene_status": row.get("goal_hygiene_status", ""),
        "non_semantic_goal_leakage": bool(row.get("non_semantic_goal_leakage", False)),
        "semantic_signal_absent": bool(row.get("semantic_signal_absent", False)),
        "raw_llm_goals": _to_list(row.get("raw_llm_goals")),
        # token usage
        "token_usage": token_usage,
        # versions
        "llm_prompt_version": row.get("llm_prompt_version", ""),
        "schema_version": row.get("schema_version", ""),
        # extra for qualitative review
        "persona_top10": persona_top10,
        "recent_concepts": _to_list(row.get("evidence_recent_concepts")),
    }


# ── Native mode: read PR3-1 fields directly from v6 cache ────────────────────

def build_normalized_row_native(row: dict, persona_top10: list[str]) -> dict:
    """
    Build normalized dict from a v6+ cache row that natively carries all PR3-1 fields.

    Reads contrast_with_persona, temporal_cues, evidence_sources, token_usage
    directly from the row — no reconstruction.

    contrast_with_persona may be stored as a JSON string (parquet serialization
    of nested dict) or as a plain dict; both are handled.
    temporal_cues likewise.
    """
    def _load_json_field(val: Any, default: Any) -> Any:
        """Load a field that may be a dict/list, a JSON string, or None/NaN."""
        if val is None:
            return default
        # numpy scalar NaN check (parquet null → float NaN)
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return default
        except Exception:
            pass
        if isinstance(val, (dict, list)):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except Exception:
                return default
        return default

    def _to_list(val: Any) -> list:
        """Convert a parquet column value (list, ndarray, None, NaN) to plain list[str]."""
        if val is None:
            return []
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return []
        except Exception:
            pass
        try:
            return [str(x) for x in val]
        except Exception:
            return []

    def _to_str(val: Any, max_len: int = 300) -> str:
        if not val or not isinstance(val, str):
            return ""
        return val[:max_len]

    validated_raw = row.get("validated_goal_concepts")
    if not _to_list(validated_raw):
        validated_raw = row.get("goal_concepts")
    validated = _to_list(validated_raw)

    contrast: dict[str, Any] = _load_json_field(row.get("contrast_with_persona"), {})
    tc: dict[str, Any] = _load_json_field(row.get("temporal_cues"), {})
    ev: list[str] = _to_list(row.get("evidence_sources"))
    token_usage: dict = _load_json_field(row.get("token_usage"), {})

    context_goals_raw = row.get("context_goals")
    context_goals = _to_list(context_goals_raw) if _to_list(context_goals_raw) else validated

    # verified/llm split from contrast_with_persona
    verified_contrast = {c: v for c, v in contrast.items() if isinstance(v, int)}
    llm_contrast = [c for c, v in contrast.items() if v == "llm"]
    shift_detected = bool(tc.get("shift_detected", False))

    return {
        # identity
        "user_id": row["user_id"],
        "target_index": int(row["target_index"]),
        "has_stage2": bool(row.get("has_stage2", False)),
        # scoring decision
        "deviation_reason": row.get("deviation_reason", "unknown"),
        "confidence": float(row.get("confidence", 0.35)),
        "validated_goal_concepts": validated,
        "context_goals": context_goals,
        # contrast
        "contrast_verified": verified_contrast,
        "contrast_llm_only": llm_contrast,
        "n_contrast_verified": len(verified_contrast),
        "n_contrast_llm": len(llm_contrast),
        "n_contrast_total": len(contrast),
        # temporal
        "temporal_shift_detected": shift_detected,
        "temporal_first_half_dominant": tc.get("first_half_dominant"),
        "temporal_second_half_dominant": tc.get("second_half_dominant"),
        "temporal_llm_summary": tc.get("llm_shift_summary", ""),
        # evidence
        "evidence_sources": ev,
        # audit rationale
        "llm_explanation_short": _to_str(row.get("llm_explanation_short"), 200),
        "why_not_aligned": _to_str(row.get("why_not_aligned"), 150),
        "why_exploration": _to_str(row.get("why_exploration"), 150),
        # hygiene
        "goal_hygiene_status": row.get("goal_hygiene_status", ""),
        "non_semantic_goal_leakage": bool(row.get("non_semantic_goal_leakage", False)),
        "semantic_signal_absent": bool(row.get("semantic_signal_absent", False)),
        "raw_llm_goals": _to_list(row.get("raw_llm_goals")),
        # token usage — native (may include prompt_tokens, response_tokens, total_tokens)
        "token_usage": token_usage,
        # versions
        "llm_prompt_version": row.get("llm_prompt_version", ""),
        "schema_version": row.get("schema_version", ""),
        # extra for qualitative review
        "persona_top10": persona_top10,
        "recent_concepts": _to_list(row.get("evidence_recent_concepts")),
    }


# ── Aggregate report ──────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total > 0 else "0.0%"


def build_aggregate_report(rows: list[dict], reconstruct: bool) -> dict:
    """Compute quantitative distributions from normalized rows."""
    n = len(rows)

    # 1. deviation_reason distribution
    reason_counter: Counter = Counter(r["deviation_reason"] for r in rows)

    # 2. confidence distribution (buckets)
    conf_buckets = {"<0.4": 0, "0.4-0.6": 0, "0.6-0.75": 0, ">=0.75": 0}
    for r in rows:
        c = r["confidence"]
        if c < 0.4:
            conf_buckets["<0.4"] += 1
        elif c < 0.6:
            conf_buckets["0.4-0.6"] += 1
        elif c < 0.75:
            conf_buckets["0.6-0.75"] += 1
        else:
            conf_buckets[">=0.75"] += 1
    conf_values = [r["confidence"] for r in rows]
    conf_mean = sum(conf_values) / n
    conf_p25 = sorted(conf_values)[n // 4]
    conf_p75 = sorted(conf_values)[3 * n // 4]
    conf_p50 = sorted(conf_values)[n // 2]

    # 3. verified_contrast fill-rate
    n_has_verified_contrast = sum(1 for r in rows if r["n_contrast_verified"] > 0)
    n_has_llm_contrast      = sum(1 for r in rows if r["n_contrast_llm"] > 0)
    n_has_any_contrast      = sum(1 for r in rows if r["n_contrast_total"] > 0)

    # 4. verified vs llm contrast agreement
    # Agreement: concept appears in BOTH verified_contrast AND llm_contrast_only
    # (Note: v5 cache has almost no llm contrast since the field didn't exist)
    n_agreement = sum(
        1 for r in rows
        if r["n_contrast_verified"] > 0
        and r["n_contrast_llm"] > 0
        and set(r["contrast_verified"].keys()) & set(r["contrast_llm_only"])
    )
    n_both = sum(1 for r in rows if r["n_contrast_verified"] > 0 and r["n_contrast_llm"] > 0)

    # 5. temporal shift_detected rate (by reason)
    shift_by_reason: dict[str, dict] = {}
    for r in rows:
        reason = r["deviation_reason"]
        if reason not in shift_by_reason:
            shift_by_reason[reason] = {"shift": 0, "total": 0}
        shift_by_reason[reason]["total"] += 1
        if r["temporal_shift_detected"]:
            shift_by_reason[reason]["shift"] += 1

    # 6. evidence_sources distribution
    ev_counter: Counter = Counter()
    ev_combo_counter: Counter = Counter()
    for r in rows:
        ev = sorted(r["evidence_sources"])
        for e in ev:
            ev_counter[e] += 1
        ev_combo_counter[tuple(ev)] += 1

    # 7. token distribution
    # Native v6: uses prompt_tokens, response_tokens, total_tokens from response.usage.
    # Reconstruct v5: uses response_tokens_est (char proxy).
    def _tok(r: dict, key: str, fallback_key: str) -> int:
        return r["token_usage"].get(key) or r["token_usage"].get(fallback_key, 0)

    prompt_vals   = [_tok(r, "prompt_tokens", "system_prompt_chars") for r in rows]
    response_vals = [_tok(r, "response_tokens", "response_tokens_est") for r in rows]
    total_vals    = [_tok(r, "total_tokens", "response_tokens_est") for r in rows]
    token_mean_prompt   = sum(prompt_vals) / n if n > 0 else 0
    token_mean_response = sum(response_vals) / n if n > 0 else 0
    token_mean_total    = sum(total_vals) / n if n > 0 else 0
    # backward compat: keep response_tokens_est for reconstruct path
    token_vals = response_vals
    token_mean = token_mean_response

    # 8. n_validated_goals distribution
    n_goals_counter: Counter = Counter(len(r["validated_goal_concepts"]) for r in rows)

    # 9. hygiene status distribution
    hygiene_counter: Counter = Counter(r.get("goal_hygiene_status", "") for r in rows)

    return {
        "meta": {
            "n_records": n,
            "reconstruct_pr3_fields": reconstruct,
            "note": (
                "contrast_with_persona and temporal_cues reconstructed from v5 cache. "
                "token_usage is char-based proxy (no real prompt stored in v5)."
                if reconstruct else "all fields from cache"
            ),
        },
        "deviation_reason": {
            "counts": dict(reason_counter),
            "pct": {k: _pct(v, n) for k, v in reason_counter.items()},
        },
        "confidence": {
            "mean": round(conf_mean, 4),
            "p25": conf_p25,
            "p50": conf_p50,
            "p75": conf_p75,
            "buckets": conf_buckets,
            "buckets_pct": {k: _pct(v, n) for k, v in conf_buckets.items()},
        },
        "contrast_fill_rate": {
            "n_has_verified_contrast": n_has_verified_contrast,
            "pct_verified": _pct(n_has_verified_contrast, n),
            "n_has_llm_contrast": n_has_llm_contrast,
            "pct_llm": _pct(n_has_llm_contrast, n),
            "n_has_any_contrast": n_has_any_contrast,
            "pct_any": _pct(n_has_any_contrast, n),
            "n_both_sources": n_both,
            "n_agreement_when_both": n_agreement,
            "agreement_rate_when_both": _pct(n_agreement, n_both) if n_both > 0 else "N/A",
            "note": (
                "v5 cache: llm_raw.contrast_signal_concepts absent → "
                "llm_contrast nearly all empty; verified_contrast from grounding_diagnostics only"
            ),
        },
        "temporal_shift": {
            "overall_shift_rate": _pct(sum(r["temporal_shift_detected"] for r in rows), n),
            "by_reason": {
                reason: {
                    "shift_rate": _pct(d["shift"], d["total"]),
                    "n": d["total"],
                }
                for reason, d in sorted(shift_by_reason.items())
            },
            "note": "temporal_cues re-derived from interactions (requires interactions+item_concepts)",
        },
        "evidence_sources": {
            "individual_counts": dict(ev_counter),
            "individual_pct": {k: _pct(v, n) for k, v in ev_counter.items()},
            "combination_counts": {
                str(list(k)): v for k, v in ev_combo_counter.most_common(10)
            },
        },
        "token_usage": {
            "prompt_tokens_mean":   round(token_mean_prompt, 1),
            "response_tokens_mean": round(token_mean_response, 1),
            "total_tokens_mean":    round(token_mean_total, 1),
            "note": (
                "native: real token counts from response.usage"
                if not reconstruct
                else "reconstruct: response_tokens_est = response_chars/4 (proxy); "
                     "prompt_tokens from system_prompt_chars (char proxy)"
            ),
        },
        "n_validated_goals": {
            "counts": dict(n_goals_counter),
            "pct": {str(k): _pct(v, n) for k, v in n_goals_counter.items()},
        },
        "goal_hygiene_status": {
            "counts": dict(hygiene_counter),
            "pct": {k: _pct(v, n) for k, v in hygiene_counter.items()},
        },
    }


# ── Qualitative sample ────────────────────────────────────────────────────────

def build_qualitative_record(norm: dict) -> dict:
    """
    Compact qualitative record for one interpretation instance.
    Shows recent flow, persona, goals, contrast, temporal, reason — in one view.
    """
    tc_first = norm.get("temporal_first_half_dominant") or "(no data)"
    tc_second = norm.get("temporal_second_half_dominant") or "(no data)"
    temporal_flow = (
        f"{tc_first} → {tc_second}"
        if norm.get("temporal_shift_detected")
        else f"{tc_first} (stable)"
    )

    return {
        "user_id": norm["user_id"],
        "target_index": norm["target_index"],
        # Scoring decision
        "deviation_reason": norm["deviation_reason"],
        "confidence": norm["confidence"],
        # Goals
        "validated_goal_concepts": norm["validated_goal_concepts"],
        "raw_llm_goals": norm["raw_llm_goals"],
        "goal_hygiene": norm["goal_hygiene_status"],
        # Context signals
        "persona_top5": norm["persona_top10"][:5],
        "recent_concepts": norm["recent_concepts"][:6],
        # Temporal
        "temporal_flow": temporal_flow,
        "temporal_shift_detected": norm["temporal_shift_detected"],
        # Contrast
        "verified_contrast": norm["contrast_verified"],
        "llm_contrast_only": norm["contrast_llm_only"],
        "n_contrast_verified": norm["n_contrast_verified"],
        "n_contrast_llm": norm["n_contrast_llm"],
        # Evidence
        "evidence_sources": norm["evidence_sources"],
        # Audit rationale
        "llm_explanation": norm["llm_explanation_short"],
        "why_not_aligned": norm["why_not_aligned"],
        "why_exploration": norm["why_exploration"],
        # Flags
        "semantic_signal_absent": norm["semantic_signal_absent"],
    }


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(
    df: pd.DataFrame,
    n: int,
    reason_col: str = "deviation_reason",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n rows stratified by deviation_reason.
    Each reason gets proportional representation (min 1 per stratum).
    """
    reasons = df[reason_col].value_counts()
    total = len(df)
    sampled_indices = []
    random.seed(seed)

    for reason, count in reasons.items():
        target_n = max(1, round(n * count / total))
        subset = df[df[reason_col] == reason]
        take = min(target_n, len(subset))
        sampled_indices.extend(subset.sample(n=take, random_state=seed).index.tolist())

    # trim or pad to exactly n
    random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:n]
    return df.loc[sampled_indices].reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PR3-2: interpretation audit")
    parser.add_argument(
        "--intent-path",
        default="data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_v5_2000.parquet",
    )
    parser.add_argument(
        "--persona-path",
        default="data/cache/persona/amazon_movies_tv/persona_graphs_v2.parquet",
    )
    parser.add_argument(
        "--interactions-path",
        default="data/processed/amazon_movies_tv/interactions.parquet",
    )
    parser.add_argument(
        "--item-concepts-path",
        default="data/processed/amazon_movies_tv/item_concepts.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="results/interpretation_audit",
    )
    parser.add_argument("--n-sample",      type=int, default=200)
    parser.add_argument("--n-qualitative", type=int, default=40)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument(
        "--signal-types",
        default="category,format,price_band",
        help="Comma-separated concept types for temporal split re-derivation (reconstruct mode only)",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help=(
            "Native mode: read PR3-1 fields (contrast_with_persona, temporal_cues, "
            "evidence_sources, token_usage) directly from cache (v6+ required). "
            "Skips reconstruction — interactions/item-concepts still loaded for "
            "persona_top10 lookup only."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_types = set(args.signal_types.split(","))

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading intent cache: %s", args.intent_path)
    df_intent = pd.read_parquet(args.intent_path)
    logger.info("  %d records loaded", len(df_intent))

    logger.info("Loading persona graph: %s", args.persona_path)
    df_persona = pd.read_parquet(args.persona_path)

    logger.info("Loading interactions: %s", args.interactions_path)
    df_int = pd.read_parquet(args.interactions_path)

    logger.info("Loading item concepts: %s", args.item_concepts_path)
    df_ic = pd.read_parquet(args.item_concepts_path)

    # ── Build lookup tables ───────────────────────────────────────────────────
    logger.info("Building lookup tables...")

    # persona top-10 per user (semantic-first, then all signal types)
    _SEMANTIC_TYPES = {"category"}
    persona_top10_by_user: dict[str, list[str]] = {}
    for uid, grp in df_persona.groupby("user_id"):
        sorted_grp = grp.sort_values("weight", ascending=False)
        # semantic concepts first, then others
        sem = sorted_grp[sorted_grp["concept_id"].str.startswith("category:")]
        others = sorted_grp[~sorted_grp["concept_id"].str.startswith("category:")]
        top = pd.concat([sem, others])["concept_id"].tolist()[:10]
        persona_top10_by_user[uid] = top

    # interactions sorted by timestamp per user
    interactions_by_user: dict[str, list[str]] = {}
    for uid, grp in df_int.sort_values("timestamp").groupby("user_id"):
        interactions_by_user[uid] = grp["item_id"].tolist()

    # item concepts lookup
    item_concepts_by_item: dict[str, list[str]] = defaultdict(list)
    for row in df_ic.itertuples(index=False):
        item_concepts_by_item[row.item_id].append(row.concept_id)
    item_concepts_by_item = dict(item_concepts_by_item)

    # ── Stratified sample ─────────────────────────────────────────────────────
    logger.info("Stratified sampling %d from %d...", args.n_sample, len(df_intent))
    df_sample = stratified_sample(df_intent, args.n_sample, seed=args.seed)
    logger.info("  Sample reason dist: %s", df_sample["deviation_reason"].value_counts().to_dict())

    # ── Build normalized rows ─────────────────────────────────────────────────
    mode_label = "native (v6)" if args.native else "reconstruct (v5)"
    logger.info("Building normalized interpretation records [mode=%s]...", mode_label)
    normalized_rows: list[dict] = []

    for _, row in df_sample.iterrows():
        row_dict = row.to_dict()
        uid = row_dict["user_id"]
        tidx = int(row_dict["target_index"])
        persona_top10 = persona_top10_by_user.get(uid, [])

        if args.native:
            norm = build_normalized_row_native(row_dict, persona_top10)
        else:
            temporal_cues = _derive_temporal_split(
                user_id=uid,
                target_index=tidx,
                interactions_by_user=interactions_by_user,
                item_concepts_by_item=item_concepts_by_item,
                signal_types=signal_types,
            )
            norm = build_normalized_row(row_dict, persona_top10, temporal_cues)

        normalized_rows.append(norm)

    # ── Aggregate report ──────────────────────────────────────────────────────
    logger.info("Computing aggregate report...")
    report = build_aggregate_report(normalized_rows, reconstruct=not args.native)

    report_path = out_dir / "aggregate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved aggregate report: %s", report_path)

    # ── Qualitative sample ────────────────────────────────────────────────────
    logger.info("Building qualitative sample (%d records)...", args.n_qualitative)

    # Stratified pick for qualitative: ~10 per reason
    qual_reasons = ["aligned", "task_focus", "exploration", "unknown"]
    qual_indices: list[int] = []
    per_reason = max(1, args.n_qualitative // len(qual_reasons))
    reason_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(normalized_rows):
        reason_groups[r["deviation_reason"]].append(i)

    random.seed(args.seed)
    for reason in qual_reasons:
        idxs = reason_groups.get(reason, [])
        random.shuffle(idxs)
        qual_indices.extend(idxs[:per_reason])

    # fill remaining from other reasons
    all_remaining = [i for i in range(len(normalized_rows)) if i not in set(qual_indices)]
    random.shuffle(all_remaining)
    qual_indices.extend(all_remaining[:max(0, args.n_qualitative - len(qual_indices))])
    qual_indices = qual_indices[:args.n_qualitative]

    qual_records = [build_qualitative_record(normalized_rows[i]) for i in qual_indices]

    qual_path = out_dir / "qualitative_sample.json"
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump(qual_records, f, indent=2, ensure_ascii=False)
    logger.info("Saved qualitative sample: %s", qual_path)

    # ── Stdout summary ────────────────────────────────────────────────────────
    _print_summary(report, normalized_rows, args)


def _print_summary(report: dict, rows: list[dict], args: argparse.Namespace) -> None:
    """Print human-readable summary to stdout."""
    n = report["meta"]["n_records"]
    sep = "─" * 62

    print()
    mode_str = "native (v6)" if getattr(args, "native", False) else "reconstruct (v5)"
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PGIM PR3-2: Interpretation Audit Report                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  n_records        : {n}")
    print(f"  source           : {Path(args.intent_path).name}")
    print(f"  mode             : {mode_str}")
    print()

    # 1. deviation_reason
    print(sep)
    print("  1. DEVIATION REASON DISTRIBUTION")
    print(sep)
    dr = report["deviation_reason"]
    for reason in ["aligned", "task_focus", "exploration", "budget_shift", "unknown"]:
        cnt = dr["counts"].get(reason, 0)
        pct = dr["pct"].get(reason, "0.0%")
        bar = "█" * (cnt * 20 // n) if n > 0 else ""
        print(f"  {reason:<14} {cnt:>4}  {pct:>6}  {bar}")
    print()

    # 2. confidence
    print(sep)
    print("  2. CONFIDENCE DISTRIBUTION")
    print(sep)
    cc = report["confidence"]
    print(f"  mean={cc['mean']:.3f}  p25={cc['p25']:.2f}  p50={cc['p50']:.2f}  p75={cc['p75']:.2f}")
    for bucket, cnt in cc["buckets"].items():
        print(f"  {bucket:<10} {cnt:>4}  {cc['buckets_pct'][bucket]:>6}")
    print()

    # 3. contrast fill rate
    print(sep)
    print("  3. CONTRAST FILL-RATE  (verified vs llm-only)")
    print(sep)
    cf = report["contrast_fill_rate"]
    print(f"  verified contrast (grounding)  : {cf['n_has_verified_contrast']:>4}  {cf['pct_verified']}")
    print(f"  llm-only contrast              : {cf['n_has_llm_contrast']:>4}  {cf['pct_llm']}")
    print(f"  any contrast                   : {cf['n_has_any_contrast']:>4}  {cf['pct_any']}")
    print(f"  agreement when both present    : {cf['agreement_rate_when_both']}")
    print(f"  note: {cf['note'][:70]}")
    print()

    # 4. temporal shift
    print(sep)
    print("  4. TEMPORAL SHIFT RATE  (by reason)")
    print(sep)
    ts = report["temporal_shift"]
    print(f"  overall: {ts['overall_shift_rate']}")
    for reason, d in ts["by_reason"].items():
        print(f"  {reason:<14}  {d['shift_rate']:>6}  (n={d['n']})")
    print()

    # 5. evidence_sources
    print(sep)
    print("  5. EVIDENCE SOURCES DISTRIBUTION")
    print(sep)
    es = report["evidence_sources"]
    for src, cnt in sorted(es["individual_counts"].items()):
        print(f"  {src:<20} {cnt:>4}  {es['individual_pct'][src]}")
    print("  Top combinations:")
    for combo, cnt in list(es["combination_counts"].items())[:5]:
        print(f"    {combo:<35} {cnt:>4}  {_pct(cnt, n)}")
    print()

    # 6. token usage
    print(sep)
    print("  6. TOKEN USAGE")
    print(sep)
    tu = report["token_usage"]
    print(f"  prompt_tokens   mean : {tu['prompt_tokens_mean']}")
    print(f"  response_tokens mean : {tu['response_tokens_mean']}")
    print(f"  total_tokens    mean : {tu['total_tokens_mean']}")
    print(f"  note: {tu['note'][:80]}")
    print()

    # 7. n_goals
    print(sep)
    print("  7. N_VALIDATED_GOALS DISTRIBUTION")
    print(sep)
    ng = report["n_validated_goals"]
    for k, cnt in sorted(ng["counts"].items()):
        print(f"  {k} goal(s)    {cnt:>4}  {ng['pct'].get(str(k),'?')}")
    print()

    # 8. hygiene
    print(sep)
    print("  8. GOAL HYGIENE STATUS")
    print(sep)
    hg = report["goal_hygiene_status"]
    for k, cnt in sorted(hg["counts"].items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {cnt:>4}  {hg['pct'].get(k,'?')}")
    print()

    # 9. qualitative preview (3 per reason)
    print(sep)
    print("  9. QUALITATIVE PREVIEW (1 per reason)")
    print(sep)
    shown_reasons: set[str] = set()
    for r in rows:
        reason = r["deviation_reason"]
        if reason in shown_reasons:
            continue
        shown_reasons.add(reason)
        print(f"\n  ── {reason.upper()} ──")
        print(f"  user: {r['user_id'][:20]}  idx={r['target_index']}")
        print(f"  persona_top5  : {r['persona_top10'][:5]}")
        print(f"  recent_sem    : {r['recent_concepts'][:5]}")
        tc1 = r.get("temporal_first_half_dominant", "?")
        tc2 = r.get("temporal_second_half_dominant", "?")
        print(f"  temporal_flow : {tc1} → {tc2}  (shift={r['temporal_shift_detected']})")
        print(f"  validated     : {r['validated_goal_concepts']}")
        print(f"  verified_ctrs : {r['contrast_verified']}")
        print(f"  llm_ctrs      : {r['contrast_llm_only']}")
        print(f"  evidence      : {r['evidence_sources']}")
        print(f"  confidence    : {r['confidence']:.2f}")
        if r.get("llm_explanation_short"):
            print(f"  explanation   : {r['llm_explanation_short'][:120]}")


if __name__ == "__main__":
    main()
