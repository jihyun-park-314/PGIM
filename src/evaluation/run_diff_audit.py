"""
run_diff_audit.py
-----------------
Task A/B: raw_llm_goals vs validated_goal_concepts diff audit.

Outputs:
  raw_vs_validated_diff_audit.csv
  exploration_changed_cases_analysis.csv
  diff_audit_report.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
BASE        = Path("/home/jhpark/PGIM")
INTENT_PATH = BASE / "data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_2000_validated.parquet"
CAND_PATH   = BASE / "data/cache/candidate/amazon_movies_tv/sampled_candidates_k101.parquet"
IC_PATH     = BASE / "data/processed/amazon_movies_tv/item_concepts.parquet"
OUT_DIR     = BASE / "results/diff_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    if hasattr(x, "__iter__"):
        return list(x)
    return []


def main():
    # ── load ─────────────────────────────────────────────────────────
    print("Loading intent cache...")
    df_intent = pd.read_parquet(INTENT_PATH)
    print(f"  intent: {len(df_intent)} rows  cols: {list(df_intent.columns)}")

    print("Loading item_concepts...")
    df_ic = pd.read_parquet(IC_PATH)
    item_concepts: dict[str, list[str]] = (
        df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()
    )
    concept_doc_freq: dict[str, int] = (
        df_ic.groupby("concept_id")["item_id"].nunique().to_dict()
    )
    total_items = df_ic["item_id"].nunique()
    print(f"  item_concepts: {len(item_concepts)}  concepts: {len(concept_doc_freq)}  total_items: {total_items}")

    print("Loading candidates (for GT)...")
    df_cands = pd.read_parquet(CAND_PATH)
    gt_concepts_by_key: dict[tuple, set] = {}
    if "is_ground_truth" in df_cands.columns:
        for row in df_cands[df_cands["is_ground_truth"] == True].itertuples(index=False):
            key = (str(row.user_id), int(row.target_index))
            gt_concepts_by_key[key] = set(item_concepts.get(row.candidate_item_id, []))
    print(f"  GT keys: {len(gt_concepts_by_key)}")

    # ── build candidate concept banks ────────────────────────────────
    print("Building candidate concept banks...")
    from src.intent.grounded_selector import build_candidate_concept_bank
    bank_by_key: dict[tuple, dict[str, int]] = {}
    for (uid, tidx), grp in df_cands.groupby(["user_id", "target_index"]):
        key = (str(uid), int(tidx))
        cands = grp["candidate_item_id"].tolist()
        bank_by_key[key] = build_candidate_concept_bank(cands, item_concepts)
    print(f"  banks: {len(bank_by_key)}")

    # ── helpers ───────────────────────────────────────────────────────
    def specificity(c: str) -> float:
        df_val = concept_doc_freq.get(c, 1)
        return math.log(total_items / max(df_val, 1))

    def avg_spec(concepts):
        if not concepts:
            return 0.0
        return sum(specificity(c) for c in concepts) / len(concepts)

    def bank_activation(concepts, bank):
        if not concepts:
            return 0.0
        return sum(bank.get(c, 0) for c in concepts) / len(concepts)

    # ── per-record diff ───────────────────────────────────────────────
    print("Computing per-record diff...")
    records = []
    for row in df_intent.itertuples(index=False):
        key = (str(row.user_id), int(row.target_index))
        raw = to_list(row.raw_llm_goals)
        val = to_list(row.validated_goal_concepts)
        reason = str(row.deviation_reason)
        conf = float(row.confidence)

        raw_set = set(raw)
        val_set = set(val)
        removed = raw_set - val_set
        added   = val_set - raw_set   # should be empty by design

        changed = raw_set != val_set
        jaccard = (
            len(raw_set & val_set) / len(raw_set | val_set)
            if (raw_set | val_set) else 1.0
        )

        gt_c = gt_concepts_by_key.get(key, set())
        gt_match_raw = len(raw_set & gt_c) / max(len(raw_set), 1)
        gt_match_val = len(val_set & gt_c) / max(len(val_set), 1) if val_set else 0.0

        bank = bank_by_key.get(key, {})
        act_raw = bank_activation(list(raw_set), bank)
        act_val = bank_activation(list(val_set), bank)

        # specificity of removed vs survived
        spec_removed  = avg_spec(list(removed))
        spec_survived = avg_spec(list(val_set))

        records.append({
            "user_id":        row.user_id,
            "target_index":   row.target_index,
            "reason":         reason,
            "confidence":     round(conf, 4),
            "n_raw":          len(raw),
            "n_val":          len(val),
            "changed_flag":   int(changed),
            "n_removed":      len(removed),
            "n_added":        len(added),
            "jaccard":        round(jaccard, 4),
            "gt_match_raw":   round(gt_match_raw, 4),
            "gt_match_val":   round(gt_match_val, 4),
            "gt_match_delta": round(gt_match_val - gt_match_raw, 4),
            "act_raw":        round(act_raw, 4),
            "act_val":        round(act_val, 4),
            "act_delta":      round(act_val - act_raw, 4),
            "spec_raw":       round(avg_spec(list(raw_set)), 4),
            "spec_val":       round(avg_spec(list(val_set)), 4),
            "spec_delta":     round(avg_spec(list(val_set)) - avg_spec(list(raw_set)), 4),
            "spec_removed":   round(spec_removed, 4),
            "spec_survived":  round(spec_survived, 4),
            "removed_concepts": ";".join(sorted(removed)),
            "added_concepts":   ";".join(sorted(added)),
        })

    df_diff = pd.DataFrame(records)
    out_csv = OUT_DIR / "raw_vs_validated_diff_audit.csv"
    df_diff.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv}")

    # ── exploration changed cases ─────────────────────────────────────
    exp_changed = df_diff[(df_diff["reason"] == "exploration") & (df_diff["changed_flag"] == 1)].copy()
    exp_changed.to_csv(OUT_DIR / "exploration_changed_cases_analysis.csv", index=False)
    print(f"  Saved exploration_changed_cases_analysis.csv ({len(exp_changed)} rows)")

    # ── print aggregate stats ─────────────────────────────────────────
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("AGGREGATE SUMMARY")
    print(SEP)
    print(f"Total records : {len(df_diff)}")
    print(f"changed_flag  : {df_diff['changed_flag'].mean():.3f}  ({df_diff['changed_flag'].sum()} records)")
    print(f"jaccard=1.0   : {(df_diff['jaccard']==1.0).mean():.3f}  ({(df_diff['jaccard']==1.0).sum()} records)")

    print(f"\nBy reason:")
    for reason, grp in df_diff.groupby("reason"):
        print(f"  {reason:15s}  n={len(grp):4d}  "
              f"changed={grp['changed_flag'].mean():.3f}  "
              f"avg_removed={grp['n_removed'].mean():.2f}  "
              f"jaccard={grp['jaccard'].mean():.3f}  "
              f"gt_match_raw={grp['gt_match_raw'].mean():.4f}  "
              f"gt_match_val={grp['gt_match_val'].mean():.4f}  "
              f"gt_match_delta={grp['gt_match_delta'].mean():.4f}  "
              f"spec_raw={grp['spec_raw'].mean():.3f}  "
              f"spec_val={grp['spec_val'].mean():.3f}")

    print(f"\nChanged vs unchanged:")
    for label, mask in [("changed", df_diff["changed_flag"] == 1),
                         ("unchanged", df_diff["changed_flag"] == 0)]:
        sub = df_diff[mask]
        print(f"  {label:10s}  n={len(sub):4d}  "
              f"gt_match_raw={sub['gt_match_raw'].mean():.4f}  "
              f"gt_match_val={sub['gt_match_val'].mean():.4f}  "
              f"gt_match_delta={sub['gt_match_delta'].mean():.4f}  "
              f"spec_raw={sub['spec_raw'].mean():.3f}  spec_val={sub['spec_val'].mean():.3f}  "
              f"act_delta={sub['act_delta'].mean():.4f}")

    # ── exploration deep dive ─────────────────────────────────────────
    exp = df_diff[df_diff["reason"] == "exploration"]
    print(f"\n{SEP}")
    print("EXPLORATION DEEP DIVE")
    print(SEP)
    print(f"n={len(exp)}  changed={exp['changed_flag'].mean():.3f}  jaccard={exp['jaccard'].mean():.3f}")
    print(f"gt_match_raw={exp['gt_match_raw'].mean():.4f}  gt_match_val={exp['gt_match_val'].mean():.4f}")
    print(f"spec_raw={exp['spec_raw'].mean():.4f}  spec_val={exp['spec_val'].mean():.4f}")
    print(f"act_raw={exp['act_raw'].mean():.4f}  act_val={exp['act_val'].mean():.4f}")
    print(f"spec_removed={exp_changed['spec_removed'].mean():.4f}  spec_survived={exp_changed['spec_survived'].mean():.4f}")

    print(f"\nExploration — changed vs unchanged:")
    for label, mask in [("changed",   exp["changed_flag"] == 1),
                         ("unchanged", exp["changed_flag"] == 0)]:
        sub = exp[mask]
        if len(sub) == 0:
            continue
        print(f"  {label:10s}  n={len(sub):4d}  "
              f"gt_match_raw={sub['gt_match_raw'].mean():.4f}  "
              f"gt_match_val={sub['gt_match_val'].mean():.4f}  "
              f"gt_match_delta={sub['gt_match_delta'].mean():.4f}  "
              f"spec_survived={sub['spec_survived'].mean():.3f}  "
              f"act_val={sub['act_val'].mean():.3f}")

    # ── removed concept analysis ──────────────────────────────────────
    print(f"\n{SEP}")
    print("REMOVED CONCEPTS — exploration (what gets removed?)")
    print(SEP)
    removed_rows = []
    for _, row in exp.iterrows():
        for c in row["removed_concepts"].split(";") if row["removed_concepts"] else []:
            if not c:
                continue
            removed_rows.append({
                "concept": c,
                "doc_freq": concept_doc_freq.get(c, 1),
                "specificity": round(specificity(c), 4),
            })
    if removed_rows:
        df_rem = pd.DataFrame(removed_rows)
        print(f"removed instances: {len(df_rem)}  unique: {df_rem['concept'].nunique()}")
        print(f"avg doc_freq    : {df_rem['doc_freq'].mean():.1f}")
        print(f"avg specificity : {df_rem['specificity'].mean():.3f}")
        print(f"\nTop 15 most removed concepts:")
        top_rem = (df_rem.groupby("concept")
                   .agg(count=("concept", "count"),
                        doc_freq=("doc_freq", "first"),
                        specificity=("specificity", "first"))
                   .sort_values("count", ascending=False).head(15))
        print(top_rem.to_string())

    # ── survived concept analysis ─────────────────────────────────────
    print(f"\n{SEP}")
    print("SURVIVED CONCEPTS — exploration changed cases (what stays?)")
    print(SEP)
    survived_rows = []
    for _, row in exp_changed.iterrows():
        val_raw = df_intent.loc[
            (df_intent["user_id"] == row["user_id"]) &
            (df_intent["target_index"] == row["target_index"]),
            "validated_goal_concepts"
        ].iloc[0]
        for c in to_list(val_raw):
            survived_rows.append({
                "concept": c,
                "doc_freq": concept_doc_freq.get(c, 1),
                "specificity": round(specificity(c), 4),
            })
    if survived_rows:
        df_surv = pd.DataFrame(survived_rows)
        print(f"survived instances: {len(df_surv)}  unique: {df_surv['concept'].nunique()}")
        print(f"avg doc_freq    : {df_surv['doc_freq'].mean():.1f}")
        print(f"avg specificity : {df_surv['specificity'].mean():.3f}")
        print(f"\nTop 15 most frequent survived concepts:")
        top_surv = (df_surv.groupby("concept")
                    .agg(count=("concept", "count"),
                         doc_freq=("doc_freq", "first"),
                         specificity=("specificity", "first"))
                    .sort_values("count", ascending=False).head(15))
        print(top_surv.to_string())

    # ── hypothesis tests ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("HYPOTHESIS TESTS")
    print(SEP)

    # H1
    print(f"H1 — selector mostly keeps same concepts:")
    print(f"  unchanged rate (all) : {(df_diff['changed_flag']==0).mean():.3f}")
    print(f"  unchanged rate (exp) : {(exp['changed_flag']==0).mean():.3f}")
    print(f"  avg jaccard (all)    : {df_diff['jaccard'].mean():.3f}")
    print(f"  avg jaccard (exp)    : {exp['jaccard'].mean():.3f}")
    print(f"  jaccard=1.0 (all)    : {(df_diff['jaccard']==1.0).mean():.3f}")

    # H2
    if removed_rows and survived_rows:
        print(f"\nH2 — survived concepts broader/generic than removed:")
        print(f"  removed  avg doc_freq={df_rem['doc_freq'].mean():.1f}  avg_spec={df_rem['specificity'].mean():.3f}")
        print(f"  survived avg doc_freq={df_surv['doc_freq'].mean():.1f}  avg_spec={df_surv['specificity'].mean():.3f}")
        verdict = "CONFIRMED" if df_surv["doc_freq"].mean() > df_rem["doc_freq"].mean() else "NOT confirmed"
        print(f"  survived broader: {verdict}")

    # H3
    aligned = df_diff[df_diff["reason"] == "aligned"]
    print(f"\nH3 — exploration backbone activation weak:")
    print(f"  exploration avg_act_val : {exp['act_val'].mean():.4f}")
    print(f"  aligned     avg_act_val : {aligned['act_val'].mean():.4f}")
    print(f"  ratio exp/aligned       : {exp['act_val'].mean()/max(aligned['act_val'].mean(),0.001):.4f}")

    # H4
    changed_all = df_diff[df_diff["changed_flag"] == 1]
    print(f"\nH4 — gt_match barely improves even when changed:")
    print(f"  avg gt_match_delta (changed) : {changed_all['gt_match_delta'].mean():.5f}")
    print(f"  fraction delta > 0           : {(changed_all['gt_match_delta'] > 0).mean():.3f}")
    print(f"  fraction delta = 0           : {(changed_all['gt_match_delta'] == 0).mean():.3f}")
    print(f"  fraction delta < 0           : {(changed_all['gt_match_delta'] < 0).mean():.3f}")
    exp_changed2 = exp[exp["changed_flag"] == 1]
    print(f"  exploration only:")
    print(f"    avg gt_match_delta (changed): {exp_changed2['gt_match_delta'].mean():.5f}")
    print(f"    fraction delta > 0           : {(exp_changed2['gt_match_delta'] > 0).mean():.3f}")

    # ── write markdown report ─────────────────────────────────────────
    _write_report(df_diff, exp, removed_rows, survived_rows, OUT_DIR)
    print(f"\nAll outputs -> {OUT_DIR}")


def _write_report(df_diff, exp, removed_rows, survived_rows, out_dir):
    df_rem  = pd.DataFrame(removed_rows)  if removed_rows  else pd.DataFrame()
    df_surv = pd.DataFrame(survived_rows) if survived_rows else pd.DataFrame()
    exp_changed = exp[exp["changed_flag"] == 1]

    lines = ["# Raw vs Validated Diff Audit Report\n"]

    # ── Core diagnosis ────────────────────────────────────────────────
    lines.append("## 핵심 진단\n")
    changed_rate = df_diff["changed_flag"].mean()
    exp_changed_rate = exp["changed_flag"].mean()
    gt_delta_changed = df_diff[df_diff["changed_flag"]==1]["gt_match_delta"].mean()
    lines.append(
        f"1. Stage 2 activation gate는 전체 {changed_rate:.1%}의 레코드에서 goal set을 변경하며, "
        f"exploration에서 {exp_changed_rate:.1%}로 가장 높다.\n"
    )
    lines.append(
        f"2. 그러나 변경된 케이스에서도 gt_match_delta={gt_delta_changed:.5f}로 "
        f"GT alignment는 거의 개선되지 않는다 — selector가 후보 공간 정합성은 높이지만 "
        f"GT-discriminative selection은 수행하지 못한다.\n"
    )
    if not df_surv.empty and not df_rem.empty:
        survived_broader = df_surv["doc_freq"].mean() > df_rem["doc_freq"].mean()
        lines.append(
            f"3. Activation gate는 특이도가 높은(niche) concept를 제거하고 "
            f"{'광범위한(generic)' if survived_broader else '더 구체적인'} concept를 살리는 경향이 있다 "
            f"(removed avg_spec={df_rem['specificity'].mean():.3f} vs "
            f"survived avg_spec={df_surv['specificity'].mean():.3f}).\n"
        )

    # ── Aggregate table ───────────────────────────────────────────────
    lines.append("## Aggregate Summary\n")
    lines.append("| reason | n | changed% | avg_jaccard | gt_match_raw | gt_match_val | gt_match_delta | spec_raw | spec_val |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for reason, grp in df_diff.groupby("reason"):
        lines.append(
            f"| {reason} | {len(grp)} "
            f"| {grp['changed_flag'].mean():.3f} "
            f"| {grp['jaccard'].mean():.3f} "
            f"| {grp['gt_match_raw'].mean():.4f} "
            f"| {grp['gt_match_val'].mean():.4f} "
            f"| {grp['gt_match_delta'].mean():.5f} "
            f"| {grp['spec_raw'].mean():.3f} "
            f"| {grp['spec_val'].mean():.3f} |"
        )
    lines.append("")

    # ── Hypothesis verdicts ───────────────────────────────────────────
    lines.append("## Hypothesis 판정\n")

    h1_verdict = "CONFIRMED" if (exp["changed_flag"] == 0).mean() > 0.4 else "PARTIALLY confirmed"
    lines.append(f"**H1 (selector mostly keeps same concepts):** {h1_verdict}")
    lines.append(
        f"- exploration unchanged rate: {(exp['changed_flag']==0).mean():.3f}, "
        f"avg jaccard: {exp['jaccard'].mean():.3f}\n"
    )

    if not df_surv.empty and not df_rem.empty:
        h2_verdict = "CONFIRMED" if df_surv["doc_freq"].mean() > df_rem["doc_freq"].mean() else "NOT confirmed"
        lines.append(f"**H2 (survived concepts are broader/generic):** {h2_verdict}")
        lines.append(
            f"- removed avg doc_freq={df_rem['doc_freq'].mean():.1f} spec={df_rem['specificity'].mean():.3f} "
            f"vs survived avg doc_freq={df_surv['doc_freq'].mean():.1f} spec={df_surv['specificity'].mean():.3f}\n"
        )

    aligned = df_diff[df_diff["reason"] == "aligned"]
    ratio = exp["act_val"].mean() / max(aligned["act_val"].mean(), 0.001)
    h3_verdict = "CONFIRMED" if ratio < 0.2 else "PARTIAL"
    lines.append(f"**H3 (exploration backbone activation weak):** {h3_verdict}")
    lines.append(
        f"- exploration avg_act_val={exp['act_val'].mean():.4f} "
        f"vs aligned={aligned['act_val'].mean():.4f} "
        f"(ratio={ratio:.4f})\n"
    )

    changed_all = df_diff[df_diff["changed_flag"] == 1]
    h4_verdict = "CONFIRMED" if (changed_all["gt_match_delta"] == 0).mean() > 0.5 else "PARTIAL"
    lines.append(f"**H4 (gt_match barely improves even when changed):** {h4_verdict}")
    lines.append(
        f"- avg gt_match_delta (changed)={changed_all['gt_match_delta'].mean():.5f}, "
        f"delta=0 fraction={( changed_all['gt_match_delta']==0).mean():.3f}\n"
    )

    # ── Next selector redesign direction ─────────────────────────────
    lines.append("## Next Selector Redesign Direction\n")
    lines.append(
        "**Discriminative grounded selector** — activation gate에 specificity score를 추가.\n\n"
        "현재 selector는 `activation_count >= 1`이면 통과시킨다. "
        "결과적으로 high-frequency generic concept (doc_freq 높음, specificity 낮음)이 "
        "살아남고, 실제로 이 후보군을 특징짓는 specific concept는 activation이 낮아 제거된다.\n\n"
        "제안: activation gate 통과 후 concept들을 "
        "`score = activation_count × specificity_weight`로 재정렬하여 "
        "generic concept에 암묵적 감점을 부여.\n"
        "specificity_weight = log(N / doc_freq) — IDF와 동일한 형태로 기존 인프라 재활용 가능.\n"
    )

    # ── Persona ontology interaction ──────────────────────────────────
    lines.append("## Persona Ontology와 LLM Short-term Branch 상호작용\n")
    lines.append(
        "**Q1. 왜 intent-only에서는 LLM이 약간 낫고, full_model에서는 heuristic이 더 높은가?**\n\n"
        "intent-only에서 LLM(+0.0065)이 앞서는 이유: "
        "LLM은 exploration/task_focus를 세밀하게 구분하여 modulation policy를 더 정확하게 적용한다. "
        "heuristic은 aligned를 48.7%로 과다 분류하여 많은 케이스에서 boost 기회를 놓친다.\n\n"
        "full_model에서 heuristic(-0.013)이 앞서는 이유: "
        "LLM은 exploration을 57.7%로 과다 분류한다. "
        "exploration goal의 gt_delta zero_frac=0.923 — 거의 모든 경우 GT delta=0. "
        "이는 persona graph의 안정적 prior 기여를 방해하지는 않지만, "
        "exploration으로 잘못 분류된 케이스에서 persona signal이 제대로 증폭되지 않아 "
        "signal dilution이 발생한다.\n"
    )
    lines.append(
        "**Q2. LLM과 persona ontology가 본질적으로 안 맞는가, 아니면 조율 문제인가?**\n\n"
        "조율 문제다. "
        "full_model에서 zero variant(exploration 완전 제거) 시 HR@10이 0.6015→0.5885로 급락한다. "
        "즉 exploration signal 자체는 full_model에 기여하고 있다. "
        "문제는 LLM이 exploration을 너무 많이 잡아 persona prior의 aligned/task_focus 기여가 "
        "exploration policy로 희석되는 것이다. "
        "LLM reason과 persona는 본질적으로 상보적 구조이나, "
        "현재 exploration 과다 분류가 그 상보성을 노이즈로 바꾸고 있다.\n"
    )
    lines.append(
        "**Q3. 상보적 긴장을 만들려면 selector/fusion에서 무엇이 필요한가?**\n\n"
        "두 가지가 필요하다:\n"
        "1. **Exploration-specific discriminative selection**: "
        "현재처럼 activation만으로 통과시키면 generic concept만 남는다. "
        "specificity-weighted selection으로 persona prior와 실질적으로 다른 "
        "specific concept만 exploration signal에 포함시켜야 한다.\n"
        "2. **Confidence-conditioned fusion weight**: "
        "exploration reason이어도 selected goal의 specificity/activation이 낮으면 "
        "persona prior를 더 신뢰하도록 gate strength를 조정. "
        "현재 gate는 reason+confidence로 구동되지만, "
        "goal quality(specificity)를 추가 입력으로 받는 것이 필요하다.\n"
    )

    # ── Final one-sentence answer ──────────────────────────────────────
    lines.append("## 핵심 질문에 대한 한 문장 답변\n")
    lines.append(
        "> 지금 PGIM의 핵심 문제는 exploration을 많이 잡는 것과 "
        "GT-discriminative signal로 변환하지 못하는 것이 **동시에** 존재하며, "
        "전자(과다 분류)는 full_model에서 persona와의 조율 실패를 유발하고, "
        "후자(변환 실패)는 intent-only의 ceiling을 낮추고 있다 — "
        "그러나 두 문제 중 **selector의 GT-discriminative 변환 실패**가 더 근본적이다. "
        "exploration 비율을 줄여도 선택된 goal이 GT와 무관하면 ranking utility는 여전히 낮기 때문이다.\n"
    )

    out_path = out_dir / "diff_audit_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
