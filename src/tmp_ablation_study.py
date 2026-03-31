"""
Mixed-source ablation study + activation + delta/rank diagnostics
Amazon Movies_and_TV

NOTE: This is a cache-constrained 1000-user diagnostic subset study.
The onto_v2 LLM intent cache covers only 1000 users.
MAX_USERS_REQUESTED is set to 2000 but actual shared_keys is bounded by LLM cache coverage (~1000).
Do NOT interpret results as a reproduction of the trusted 2000-user benchmark.

Variants:
  A: heuristic reason + heuristic goals  (pure heuristic — upper ref)
  B: llm reason     + llm goals          (pure llm)
  C: heuristic reason + llm goals        (isolates goal-side)
  D: llm reason     + heuristic goals    (isolates reason-side)
  B_cons: B + conservative gate          (fusion sanity check on pure LLM)
  D_cons: D + conservative gate          (fusion sanity check on llm reason + heur goals)

Interpretation notes are embedded in the report. Do not draw conclusions
from any single table — always interpret mixed ablation + activation +
delta/rank diagnostics together.
"""

import sys, json, copy, subprocess
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, "/pgim")

import pandas as pd
import numpy as np
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CFG_PATH = "config/data/amazon_movies_tv.yaml"
EVAL_CFG_PATH = "config/evaluation/default.yaml"
MOD_CFG_PATH  = "config/modulation/amazon_movies_tv.yaml"
BB_CFG_PATH   = "config/backbone/amazon_movies_tv_sasrec.yaml"
DATASET       = "amazon_movies_tv"

# NOTE: 2000 requested but actual shared_keys bounded by LLM cache coverage.
# The true N will be asserted and reported explicitly.
MAX_USERS_REQUESTED = 2000

HEURISTIC_PATH     = "data/cache/intent/amazon_movies_tv/short_term_intents.parquet"
LLM_PATH           = "data/cache/intent/amazon_movies_tv/short_term_intents_llm_subset_onto_v2.parquet"
ITEM_CONCEPTS_PATH = "data/processed/amazon_movies_tv/item_concepts.parquet"
CANDIDATES_PATH    = "data/cache/candidate/amazon_movies_tv/sampled_candidates_k101.parquet"

OUT_DIR   = Path("data/artifacts/eval/amazon_movies_tv/ablation_study")
CACHE_DIR = Path("data/cache/intent/amazon_movies_tv")
MANIFEST_PATH = OUT_DIR / "shared_keys_manifest.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REASONS     = ["aligned", "exploration", "task_focus", "budget_shift", "unknown"]
EXPERIMENTS = ["ablation_intent_only", "full_model"]

# ── Load configs ──────────────────────────────────────────────────────────────
with open(DATA_CFG_PATH)  as f: data_cfg = yaml.safe_load(f)
with open(EVAL_CFG_PATH)  as f: eval_cfg = yaml.safe_load(f)
with open(MOD_CFG_PATH)   as f: mod_cfg  = yaml.safe_load(f)

# Conservative gate config (B_cons and D_cons only)
mod_cfg_conservative = copy.deepcopy(mod_cfg)
mod_cfg_conservative["gate"]["reason_base_weight"] = {
    "aligned":      0.5,
    "exploration":  0.3,
    "task_focus":   0.6,
    "budget_shift": 0.5,
    "unknown":      0.10,
}
cons_mod_path = str(OUT_DIR / "mod_cfg_conservative.yaml")
with open(cons_mod_path, "w") as f:
    yaml.dump(mod_cfg_conservative, f)

# ── Load intent caches ────────────────────────────────────────────────────────
print("Loading intent caches...")
df_h = pd.read_parquet(HEURISTIC_PATH)
df_l = pd.read_parquet(LLM_PATH)

def to_key_dict(df: pd.DataFrame) -> dict:
    d = {}
    for _, row in df.iterrows():
        k = (row["user_id"], int(row["target_index"]))
        d[k] = row.to_dict()
    return d

h_dict = to_key_dict(df_h)
l_dict = to_key_dict(df_l)

shared_keys_both = set(h_dict.keys()) & set(l_dict.keys())
print(f"  heuristic={len(h_dict)}  llm={len(l_dict)}  shared={len(shared_keys_both)}")

# ── Build eval_keys manifest (single source of truth) ────────────────────────
print("Loading candidates for manifest construction...")
df_cands_full = pd.read_parquet(CANDIDATES_PATH)
cand_keys = set(zip(df_cands_full["user_id"], df_cands_full["target_index"].astype(int)))

eval_keys_full = shared_keys_both & cand_keys
all_users_sorted = sorted({k[0] for k in eval_keys_full})

# Apply MAX_USERS_REQUESTED cap — actual N bounded by LLM cache
actual_users = all_users_sorted[:MAX_USERS_REQUESTED]
eval_keys = {k for k in eval_keys_full if k[0] in set(actual_users)}

ACTUAL_N = len({k[0] for k in eval_keys})
print(f"\n  MAX_USERS_REQUESTED={MAX_USERS_REQUESTED}  actual_N={ACTUAL_N}")
print(f"  NOTE: actual_N < MAX_USERS_REQUESTED because LLM cache covers only {len(l_dict)} users")

assert ACTUAL_N <= len(l_dict), \
    f"eval_keys has {ACTUAL_N} users but LLM cache only has {len(l_dict)}"
assert ACTUAL_N >= 500, \
    f"Too few eval users ({ACTUAL_N}) — check LLM cache or candidate intersection"

# Save manifest
df_manifest = pd.DataFrame(
    [{"user_id": k[0], "target_index": k[1]} for k in sorted(eval_keys)]
)
df_manifest.to_csv(MANIFEST_PATH, index=False)
print(f"  Manifest saved: {MANIFEST_PATH}  ({len(eval_keys)} keys, {ACTUAL_N} users)")

# ── Build hybrid intent variants ──────────────────────────────────────────────
def build_variant_dict(variant: str) -> dict:
    result = {}
    for key in eval_keys:
        h = h_dict.get(key)
        l = l_dict.get(key)
        assert h is not None, f"Missing heuristic record for key {key}"
        assert l is not None, f"Missing LLM record for key {key}"
        if variant == "A":
            result[key] = copy.deepcopy(h)
        elif variant == "B":
            result[key] = copy.deepcopy(l)
        elif variant == "C":
            rec = copy.deepcopy(h)
            rec["goal_concepts"]    = l["goal_concepts"]
            rec["constraints_json"] = l["constraints_json"]
            result[key] = rec
        elif variant == "D":
            rec = copy.deepcopy(l)
            rec["goal_concepts"]    = h["goal_concepts"]
            rec["constraints_json"] = h["constraints_json"]
            result[key] = rec
    return result

print("\nBuilding variants A/B/C/D...")
variants = {v: build_variant_dict(v) for v in ["A", "B", "C", "D"]}

# ── Save hybrid parquets ──────────────────────────────────────────────────────
def save_intent_parquet(intent_dict: dict, path: str) -> int:
    records = list(intent_dict.values())
    df = pd.DataFrame(records)
    for col in ("goal_concepts", "evidence_item_ids"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: list(x) if isinstance(x, (list, np.ndarray))
                else ([] if x is None else [str(x)])
            )
    df.to_parquet(path, index=False)
    return len(df)

cache_paths = {}
for var in ["A", "B", "C", "D"]:
    p = str(CACHE_DIR / f"ablation_variant_{var}.parquet")
    n = save_intent_parquet(variants[var], p)
    cache_paths[var] = p
    print(f"  variant {var}: {n} rows -> {p}")

# ── Print reason distributions ────────────────────────────────────────────────
print("\nReason distributions:")
for var in ["A", "B", "C", "D"]:
    reasons = [variants[var][k]["deviation_reason"] for k in eval_keys if k in variants[var]]
    dist = Counter(reasons)
    print(f"  {var}: {dict(dist.most_common())}")

# ── Part 1: Run eval subprocesses ─────────────────────────────────────────────
print("\n" + "="*70)
print(f"PART 1: MIXED ABLATION EVAL  (N={ACTUAL_N} users, cache-constrained)")
print("="*70)

eval_configs = [
    ("A", cache_paths["A"], MOD_CFG_PATH,  "A_heur+heur"),
    ("B", cache_paths["B"], MOD_CFG_PATH,  "B_llm+llm"),
    ("C", cache_paths["C"], MOD_CFG_PATH,  "C_heur_reason+llm_goals"),
    ("D", cache_paths["D"], MOD_CFG_PATH,  "D_llm_reason+heur_goals"),
    ("B", cache_paths["B"], cons_mod_path, "B_cons"),
    ("D", cache_paths["D"], cons_mod_path, "D_cons"),
]

eval_rows = []

for var, llm_path, mod_path, label in eval_configs:
    print(f"\n  [{label}]")
    out_subdir = str(OUT_DIR / label)
    cmd = [
        "python3", "-m", "src.evaluation.run_llm_vs_heuristic_eval",
        "--data-config",        DATA_CFG_PATH,
        "--evaluation-config",  EVAL_CFG_PATH,
        "--backbone-config",    BB_CFG_PATH,
        "--modulation-config",  mod_path,
        "--llm-intent-path",    llm_path,
        "--max-users",          str(MAX_USERS_REQUESTED),
        "--out-dir",            out_subdir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/pgim")
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[-600:]}")
        continue

    mpath = Path(out_subdir) / "llm_vs_heuristic_metrics.csv"
    if not mpath.exists():
        print(f"  metrics not found: {mpath}")
        continue

    df_m = pd.read_csv(mpath)
    df_m["variant"] = label
    eval_rows.append(df_m)

    llm_n = df_m[df_m["intent_source"] == "llm"]["n_users"].iloc[0] \
            if "n_users" in df_m.columns else "?"
    print(f"  n_users={llm_n}  (manifest={ACTUAL_N})")

    for _, r in df_m[df_m["intent_source"] == "llm"].iterrows():
        print(f"    {r['experiment']:30s}  HR@10={r['HR@10']:.4f}  NDCG={r['NDCG@10']:.4f}  "
              f"MRR={r['MRR']:.4f}  imp={int(r['improved'])}  wor={int(r['worsened'])}  "
              f"gt_delta_+frac={r.get('gt_delta_positive_frac', float('nan')):.3f}")

df_eval_all = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
if not df_eval_all.empty:
    df_eval_all.to_csv(OUT_DIR / "ablation_metrics_all.csv", index=False)

# ── Part 3: Delta/rank diagnostics ───────────────────────────────────────────
print("\n" + "="*70)
print("PART 3: DELTA / RANK-CHANGE DIAGNOSTICS")
print("="*70)

rank_diag_rows = []

for var, _, _, label in eval_configs:
    for exp in EXPERIMENTS:
        rpath = OUT_DIR / label / "llm" / f"sampled_reranked_{exp}.parquet"
        if not rpath.exists():
            continue
        df_r = pd.read_parquet(rpath)

        if "is_ground_truth" in df_r.columns:
            df_gt = df_r[df_r["is_ground_truth"] == True].copy()
        elif "label" in df_r.columns:
            df_gt = df_r[df_r["label"] == 1].copy()
        else:
            print(f"  WARNING: no GT column in {rpath} — skipping")
            continue

        assert len(df_gt) > 0, f"No GT rows found in {rpath}"

        if "modulation_delta" in df_gt.columns:
            delta = df_gt["modulation_delta"]
        elif "final_score" in df_gt.columns and "backbone_score" in df_gt.columns:
            delta = df_gt["final_score"] - df_gt["backbone_score"]
        else:
            print(f"  WARNING: no delta column in {rpath} — skipping")
            continue

        nonzero_frac   = (delta.abs() > 1e-6).mean()
        avg_abs_delta  = delta.abs().mean()
        avg_delta      = delta.mean()
        delta_pos_frac = (delta > 0).mean()

        if "rank_before" in df_gt.columns and "rank_after" in df_gt.columns:
            rank_improved   = (df_gt["rank_after"] < df_gt["rank_before"]).mean()
            rank_worsened   = (df_gt["rank_after"] > df_gt["rank_before"]).mean()
            in_top10_after  = (df_gt["rank_after"]  <= 10).mean()
            in_top10_before = (df_gt["rank_before"] <= 10).mean()
            top10_changed   = float(abs(in_top10_after - in_top10_before))
        else:
            rank_improved = rank_worsened = top10_changed = float("nan")

        rank_diag_rows.append({
            "variant": label, "experiment": exp,
            "n_gt": len(df_gt),
            "nonzero_delta": nonzero_frac,
            "avg_abs_delta": avg_abs_delta,
            "avg_delta":     avg_delta,
            "delta_pos_frac": delta_pos_frac,
            "rank_improved": rank_improved,
            "rank_worsened": rank_worsened,
            "top10_changed": top10_changed,
        })
        print(f"  [{label}] {exp}: nonzero={nonzero_frac:.3f}  avg|delta|={avg_abs_delta:.4f}  "
              f"delta+={delta_pos_frac:.3f}  rank_imp={rank_improved:.3f}  rank_wor={rank_worsened:.3f}")

df_rank_diag = pd.DataFrame(rank_diag_rows)
if not df_rank_diag.empty:
    df_rank_diag.to_csv(OUT_DIR / "rank_delta_diagnostics.csv", index=False)

# ── Part 2: Candidate Activation Diagnostics ─────────────────────────────────
print("\n" + "="*70)
print("PART 2: CANDIDATE ACTIVATION DIAGNOSTICS")
print("="*70)

print("Loading item concepts...")
df_ic = pd.read_parquet(ITEM_CONCEPTS_PATH)
item_concepts_map: dict[str, list] = df_ic.groupby("item_id")["concept_id"].apply(list).to_dict()

concept_to_items: dict[str, set] = defaultdict(set)
for iid, cids in item_concepts_map.items():
    for cid in cids:
        concept_to_items[cid].add(iid)

def string_prefix_relaxed_match_items(goals: list[str]) -> set:
    """
    String-prefix relaxed match (NOT ontology-aware).
    category:action matches category:action_adventure by prefix only.
    Prefix = first token before '_' in the value part.
    """
    matched: set = set()
    prefixes: set = set()
    for g in goals:
        parts = g.split(":", 1)
        if len(parts) == 2:
            ns, val = parts
            prefix = val.split("_")[0]
            if prefix:
                prefixes.add(f"{ns}:{prefix}")
    for cid, items in concept_to_items.items():
        for pfx in prefixes:
            if cid.startswith(pfx):
                matched |= items
                break
    return matched

# Build candidate lookup from manifest (strict — no GT fallback)
print("Building candidate lookup from manifest...")
manifest_keys = set(zip(df_manifest["user_id"], df_manifest["target_index"].astype(int)))
assert manifest_keys == eval_keys, \
    "Manifest keys do not match eval_keys — regenerate manifest"

cand_by_key: dict = {}
for _, row in df_cands_full.iterrows():
    k = (row["user_id"], int(row["target_index"]))
    if k not in eval_keys:
        continue
    iid   = str(row["candidate_item_id"])
    is_gt = bool(row.get("is_ground_truth", False))
    if k not in cand_by_key:
        cand_by_key[k] = {"gt": None, "items": set()}
    if is_gt:
        cand_by_key[k]["gt"] = iid
    else:
        cand_by_key[k]["items"].add(iid)

# Strict: every eval key must have GT
missing_gt = [k for k in eval_keys if cand_by_key.get(k, {}).get("gt") is None]
assert len(missing_gt) == 0, \
    f"{len(missing_gt)} eval keys have no GT item in candidate pool"
print(f"  candidate keys: {len(cand_by_key)}  all have GT: True")

act_rows = []

for var, _, _, label in eval_configs[:4]:  # A-D only
    intent_d = variants.get(var)
    if intent_d is None:
        continue
    print(f"\n  [{label}]")

    for k in eval_keys:
        rec = intent_d.get(k)
        assert rec is not None, f"Missing variant {var} record for key {k}"

        reason = rec.get("deviation_reason", "unknown")
        goals  = rec.get("goal_concepts", [])
        if goals is None:
            goals = []
        if isinstance(goals, np.ndarray):
            goals = list(goals)
        goals = [str(g) for g in goals if g]

        exact_activated: set = set()
        for g in goals:
            exact_activated |= concept_to_items.get(g, set())

        relaxed_activated = string_prefix_relaxed_match_items(goals)

        cand_info  = cand_by_key[k]
        cand_items = cand_info["items"]
        gt_item    = cand_info["gt"]

        n_exact_cand   = len(exact_activated & cand_items)
        gt_exact       = gt_item in exact_activated
        n_relaxed_cand = len(relaxed_activated & cand_items)
        gt_relaxed     = gt_item in relaxed_activated

        act_rows.append({
            "variant":          label,
            "reason":           reason,
            "goals_nonempty":   len(goals) > 0,
            "n_goals":          len(goals),
            "any_cand_exact":   n_exact_cand > 0,
            "n_cand_exact":     n_exact_cand,
            "gt_exact":         gt_exact,
            "any_cand_relaxed": n_relaxed_cand > 0,
            "n_cand_relaxed":   n_relaxed_cand,
            "gt_relaxed":       gt_relaxed,
        })

    sub = pd.DataFrame([r for r in act_rows if r["variant"] == label])
    print(f"    n={len(sub)}  goals_ne={sub['goals_nonempty'].mean():.3f}  "
          f"gt_exact={sub['gt_exact'].mean():.3f}  gt_relaxed={sub['gt_relaxed'].mean():.3f}  "
          f"any_cand_exact={sub['any_cand_exact'].mean():.3f}  avg_cand_exact={sub['n_cand_exact'].mean():.1f}")
    for reason in REASONS:
        r_sub = sub[sub["reason"] == reason]
        if len(r_sub) == 0:
            continue
        print(f"      [{reason:12s}] n={len(r_sub):3d}  "
              f"gt_exact={r_sub['gt_exact'].mean():.3f}  gt_relaxed={r_sub['gt_relaxed'].mean():.3f}  "
              f"any_cand_exact={r_sub['any_cand_exact'].mean():.3f}")

df_act = pd.DataFrame(act_rows)
if not df_act.empty:
    df_act.to_csv(OUT_DIR / "activation_diagnostics.csv", index=False)

# ── Part 5: Markdown Report ───────────────────────────────────────────────────
print("\n" + "="*70)
print("PART 5: MARKDOWN REPORT")
print("="*70)

lines = [
    "# PGIM Ablation Study Report",
    "",
    "## Study Scope",
    "",
    f"- **Dataset**: Amazon Movies_and_TV",
    f"- **N (actual)**: {ACTUAL_N} users (cache-constrained diagnostic subset)",
    f"- **N (requested)**: {MAX_USERS_REQUESTED}",
    f"- **Constraint**: onto_v2 LLM cache covers only {len(l_dict)} users.",
    f"  Results reflect a diagnostic subset study, NOT the trusted 2000-user benchmark.",
    f"- **Manifest**: `shared_keys_manifest.csv` (single source of truth for all analyses)",
    f"- **Relaxed match**: string-prefix relaxed (NOT ontology-aware).",
    f"  e.g. `category:action` matches `category:action_adventure` by string prefix only.",
    "",
]

# Reason distributions
lines += ["## Reason Distributions per Variant", ""]
lines.append("| Variant | aligned | exploration | task_focus | budget_shift | unknown |")
lines.append("|---|---|---|---|---|---|")
for var, _, _, label in eval_configs[:4]:
    intent_d = variants.get(var)
    if not intent_d:
        continue
    dist = Counter(intent_d[k]["deviation_reason"] for k in eval_keys if k in intent_d)
    total = sum(dist.values())
    lines.append(
        f"| {label} "
        f"| {dist.get('aligned',0)/total:.3f} "
        f"| {dist.get('exploration',0)/total:.3f} "
        f"| {dist.get('task_focus',0)/total:.3f} "
        f"| {dist.get('budget_shift',0)/total:.3f} "
        f"| {dist.get('unknown',0)/total:.3f} |"
    )

# Table 1
lines += ["", "## Table 1: Mixed Ablation — LLM Branch Metrics", ""]
if not df_eval_all.empty:
    lines.append("| Variant | Experiment | HR@10 | NDCG@10 | MRR | improved | worsened | gt_delta_+frac |")
    lines.append("|---|---|---|---|---|---|---|---|")
    order = ["A_heur+heur","B_llm+llm","C_heur_reason+llm_goals","D_llm_reason+heur_goals","B_cons","D_cons"]
    for v in order:
        for exp in EXPERIMENTS:
            sub = df_eval_all[
                (df_eval_all["variant"] == v) &
                (df_eval_all["intent_source"] == "llm") &
                (df_eval_all["experiment"] == exp)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| {v} | {exp} "
                f"| {r['HR@10']:.4f} | {r['NDCG@10']:.4f} | {r['MRR']:.4f} "
                f"| {int(r['improved'])} | {int(r['worsened'])} "
                f"| {r.get('gt_delta_positive_frac', float('nan')):.3f} |"
            )

# Table 2: Activation overall
lines += ["", "## Table 2: Candidate Activation Diagnostics (overall)", ""]
if not df_act.empty:
    lines.append("| Variant | goals_ne | gt_exact | gt_relaxed | any_cand_exact | avg_cand_exact |")
    lines.append("|---|---|---|---|---|---|")
    for v in df_act["variant"].unique():
        s = df_act[df_act["variant"] == v]
        lines.append(
            f"| {v} | {s['goals_nonempty'].mean():.3f} "
            f"| {s['gt_exact'].mean():.3f} | {s['gt_relaxed'].mean():.3f} "
            f"| {s['any_cand_exact'].mean():.3f} | {s['n_cand_exact'].mean():.1f} |"
        )

# Table 3: GT activation by reason (A/B/C/D)
lines += ["", "## Table 3: GT Activation by Reason (A / B / C / D)", ""]
if not df_act.empty:
    var_labels = ["A_heur+heur","B_llm+llm","C_heur_reason+llm_goals","D_llm_reason+heur_goals"]
    lines.append("| Reason | n(A) | A gt_ex | A gt_rel | n(B) | B gt_ex | B gt_rel | n(C) | C gt_ex | n(D) | D gt_ex |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for reason in REASONS:
        row = f"| {reason} |"
        for v in var_labels:
            sub = df_act[(df_act["variant"] == v) & (df_act["reason"] == reason)]
            if len(sub) == 0:
                if v in var_labels[:2]:
                    row += " 0 | — | — |"
                else:
                    row += " 0 | — |"
            else:
                if v in var_labels[:2]:
                    row += f" {len(sub)} | {sub['gt_exact'].mean():.3f} | {sub['gt_relaxed'].mean():.3f} |"
                else:
                    row += f" {len(sub)} | {sub['gt_exact'].mean():.3f} |"
        lines.append(row)

# Table 4: Delta/rank
lines += ["", "## Table 4: Delta / Rank-Change Diagnostics", ""]
if not df_rank_diag.empty:
    lines.append("| Variant | Experiment | nonzero_delta | avg|delta| | delta+_frac | rank_improved | rank_worsened |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df_rank_diag.sort_values(["variant","experiment"]).iterrows():
        lines.append(
            f"| {r['variant']} | {r['experiment']} "
            f"| {r['nonzero_delta']:.3f} | {r['avg_abs_delta']:.4f} "
            f"| {r['delta_pos_frac']:.3f} | {r['rank_improved']:.3f} | {r['rank_worsened']:.3f} |"
        )

# Interpretation
lines += [
    "", "## Interpretation Notes",
    "",
    "**Do not draw conclusions from any single table. Interpret together.**",
    "",
    "### Mixed ablation (Table 1):",
    "- C weak → goal-side bottleneck (extraction, grounding, OR fusion — not yet decomposed)",
    "- D recovers → reason classification is NOT the primary bottleneck",
    "- C weak AND D strong → goal-side is primary; reason is secondary",
    "- Both weak → multiple bottlenecks; decompose with Tables 2-4",
    "",
    "### Activation (Tables 2–3):",
    "- gt_exact low AND any_cand_exact low → goals not connected to candidate space",
    "  (extraction failure OR concept coverage gap — cannot distinguish without ontology analysis)",
    "- gt_exact low BUT any_cand_exact high → grounding mismatch (plausible goals, wrong items)",
    "- gt_exact(B) ≈ gt_exact(A) but HR@10(B) << HR@10(A) → activation similar, fusion is the issue",
    "",
    "### Delta/rank (Table 4):",
    "- nonzero_delta low → gate suppresses signal (fusion bottleneck)",
    "- nonzero_delta high, rank_improved low → signal direction wrong",
    "- delta+_frac high (>0.85) → directionally correct; rank failure is magnitude/threshold issue",
    "",
    "### Conservative trust (B_cons, D_cons — sanity check only):",
    "- B_cons improves over B → over-trust in LLM fusion is part of the problem",
    "- D_cons improves over D → LLM reason produces noisy gate inputs",
    "",
    f"> **Study constraint**: N={ACTUAL_N} (LLM cache limit, not 2000-user benchmark).",
    "> Do not generalize without replication on full user set.",
]

report_path = OUT_DIR / "ablation_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Report saved: {report_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"SUMMARY  (N={ACTUAL_N} cache-constrained users)")
print("="*70)

if not df_eval_all.empty:
    print("\nMixed Ablation (LLM branch, full_model HR@10):")
    for v in ["A_heur+heur","B_llm+llm","C_heur_reason+llm_goals","D_llm_reason+heur_goals","B_cons","D_cons"]:
        sub = df_eval_all[
            (df_eval_all["variant"] == v) &
            (df_eval_all["intent_source"] == "llm") &
            (df_eval_all["experiment"] == "full_model")
        ]
        if len(sub):
            r = sub.iloc[0]
            print(f"  {v:<42}  HR@10={r['HR@10']:.4f}  NDCG={r['NDCG@10']:.4f}  MRR={r['MRR']:.4f}")

if not df_act.empty:
    print("\nActivation (gt_exact / gt_relaxed):")
    for v in df_act["variant"].unique():
        s = df_act[df_act["variant"] == v]
        print(f"  {v:<42}  gt_exact={s['gt_exact'].mean():.3f}  gt_relaxed={s['gt_relaxed'].mean():.3f}")

print(f"\nAll outputs in: {OUT_DIR}")
print("Done.")
