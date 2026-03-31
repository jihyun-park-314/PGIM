"""
run_build_intent_4o_pilot.py
----------------------------
Cost-safe, interruption-safe GPT-4o pilot runner for intent classification.

Design constraints:
- Forces --subset-eval-only (last snapshot per user only — 1 call per user)
- Call-level JSONL checkpointing: partial progress is never lost on interruption
- Compact prompt mode: ~30-40% token reduction vs full prompt
- Small pilot by default (--max-users 100)
- On restart: skips already completed (user_id, target_index) keys automatically

Usage:
  # First pilot (100 users):
  python -m src.intent.run_build_intent_4o_pilot \\
      --data-config config/data/amazon_movies_tv.yaml \\
      --intent-config config/intent/amazon_movies_tv_4o.yaml \\
      --max-users 100 \\
      --tag 4o_pilot_100

  # Resume after interruption (same command — skips completed rows):
  python -m src.intent.run_build_intent_4o_pilot \\
      --data-config config/data/amazon_movies_tv.yaml \\
      --intent-config config/intent/amazon_movies_tv_4o.yaml \\
      --max-users 100 \\
      --tag 4o_pilot_100

  # Expand to 200 users (reuses first 100 from checkpoint if same tag):
  python -m src.intent.run_build_intent_4o_pilot \\
      --data-config config/data/amazon_movies_tv.yaml \\
      --intent-config config/intent/amazon_movies_tv_4o.yaml \\
      --max-users 200 \\
      --tag 4o_pilot_200

Output:
  data/cache/intent/<dataset>/short_term_intents_llm_subset_<tag>.parquet
  data/cache/intent/<dataset>/checkpoint_<tag>.jsonl  (intermediate, kept for resumability)

Go/no-go rule (check after pilot):
  - If 4o intent_only HR@10 > mini intent_only HR@10 by >= 0.010: model capacity IS a bottleneck → expand
  - If delta < 0.010: model capacity is NOT the main bottleneck → stop, investigate prompt/data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import pandas as pd
import yaml
from tqdm import tqdm

from src.intent.cache_resolver import IntentCacheResolver
from src.intent.context_extractor import ContextExtractor
from src.intent.parser import parse_intent
from src.persona.builder import PersonaGraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_checkpoint(ckpt_path: Path) -> dict[tuple[str, int], dict]:
    """Load completed (user_id, target_index) -> record from JSONL checkpoint."""
    completed: dict[tuple[str, int], dict] = {}
    if not ckpt_path.exists():
        return completed
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["user_id"], int(rec["target_index"]))
                completed[key] = rec
            except Exception:
                pass
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-safe GPT-4o pilot: last-snapshot-only + checkpointing"
    )
    parser.add_argument("--data-config",   default="config/data/amazon_movies_tv.yaml")
    parser.add_argument("--intent-config", default="config/intent/amazon_movies_tv_4o.yaml")
    parser.add_argument(
        "--max-users",
        type=int,
        default=100,
        help="Number of users to process (default: 100 for pilot). Always uses last snapshot only.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=2,
        help="Parallel LLM threads. Keep low (2-3) to avoid 429 rate limits. Default: 2.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Output tag, e.g. '4o_pilot_100'. Determines output parquet and checkpoint filenames.",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable compact prompt mode (uses full prompt — higher cost, for ablation only).",
    )
    args = parser.parse_args()

    compact = not args.no_compact

    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.intent_config) as f:
        intent_cfg = yaml.safe_load(f)

    if "llm" not in intent_cfg:
        intent_cfg["llm"] = {}
    intent_cfg["llm"]["use_llm"] = True

    processed_dir = Path(data_cfg["paths"]["processed_dir"])
    interim_dir   = Path(data_cfg["paths"]["interim_dir"])
    dataset       = data_cfg.get("dataset", "amazon_movies_tv")
    intent_cache_dir = Path(f"data/cache/intent/{dataset}")
    intent_cache_dir.mkdir(parents=True, exist_ok=True)

    out_path  = intent_cache_dir / f"short_term_intents_llm_subset_{args.tag}.parquet"
    ckpt_path = intent_cache_dir / f"checkpoint_{args.tag}.jsonl"

    # ── Load inputs ──────────────────────────────────────────────────────────
    logger.info("Loading inputs...")
    df_snapshots     = pd.read_parquet(interim_dir / "recent_context_snapshots.parquet")
    df_persona       = pd.read_parquet(f"data/cache/persona/{dataset}/persona_graphs.parquet")
    df_item_concepts = pd.read_parquet(processed_dir / "item_concepts.parquet")

    item_concepts = df_item_concepts.groupby("item_id")["concept_id"].apply(list).to_dict()
    persona_nodes = {uid: g.to_dict("records") for uid, g in df_persona.groupby("user_id")}
    signal_types  = intent_cfg.get("context", {}).get("signal_concept_types", ["category"])
    top_goal      = intent_cfg.get("context", {}).get("top_goal_concepts", 5)

    extractor = ContextExtractor(
        item_concepts=item_concepts,
        persona_nodes=persona_nodes,
        signal_concept_types=signal_types,
        top_goal_concepts=top_goal,
    )

    # ── FORCE last snapshot per user ─────────────────────────────────────────
    df_last = df_snapshots.loc[
        df_snapshots.groupby("user_id")["target_index"].idxmax()
    ].reset_index(drop=True)

    # Deterministic user ordering
    unique_users = sorted(df_last["user_id"].unique())[:args.max_users]
    df_last = df_last[df_last["user_id"].isin(unique_users)].reset_index(drop=True)

    n_snapshots = len(df_last)
    logger.info(
        "Pilot plan: max_users=%d  last-snapshot-only=True  snapshots=%d  compact=%s",
        args.max_users, n_snapshots, compact,
    )
    logger.info("Expected LLM calls: %d  (1 per user)", n_snapshots)
    logger.info("Estimated cost @ $5/1M tokens, ~3500 tok/call: ~$%.2f", n_snapshots * 3500 / 1e6 * 5)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    completed = _load_checkpoint(ckpt_path)
    if completed:
        logger.info("Checkpoint loaded: %d already completed rows (will skip)", len(completed))

    # ── LLM client ───────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    try:
        import openai
        openai_client = openai.OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    llm_model = intent_cfg.get("llm", {}).get("model", "gpt-4o")
    logger.info("LLM client ready (model=%s)", llm_model)

    from src.intent.llm_interpreter import _build_candidate_concepts, interpret_with_llm

    # ── Process ───────────────────────────────────────────────────────────────
    rows_list = [row.to_dict() for _, row in df_last.iterrows()]
    contexts  = [extractor.extract(r) for r in rows_list]

    # Identify which contexts still need processing
    pending_idxs = [
        i for i, ctx in enumerate(contexts)
        if (ctx.user_id, int(ctx.target_index)) not in completed
    ]
    logger.info(
        "Pending: %d / %d  (skipping %d already completed)",
        len(pending_idxs), n_snapshots, n_snapshots - len(pending_idxs),
    )

    n_ok = 0
    n_fallback = 0
    latencies: list[float] = []

    ckpt_fh = open(ckpt_path, "a", encoding="utf-8")  # append mode

    def _process_one(idx: int) -> tuple[bool, float]:
        """Process single context, write to checkpoint. Returns (success, elapsed)."""
        ctx = contexts[idx]
        t0 = time.perf_counter()
        raw = interpret_with_llm(ctx, "", intent_cfg, openai_client, compact=compact)
        elapsed = time.perf_counter() - t0
        source = raw.get("source_mode", "llm_fallback")
        record = parse_intent(raw, ctx.user_id, ctx.target_index, source)
        # Write to checkpoint immediately
        ckpt_fh.write(json.dumps(record, default=str) + "\n")
        ckpt_fh.flush()
        return source == "llm", elapsed

    if args.llm_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.llm_workers) as pool:
            futures = {pool.submit(_process_one, i): i for i in pending_idxs}
            with tqdm(total=len(pending_idxs), desc="LLM calls") as pbar:
                for future in as_completed(futures):
                    try:
                        ok, elapsed = future.result()
                        if ok:
                            n_ok += 1
                            latencies.append(elapsed)
                        else:
                            n_fallback += 1
                    except Exception as exc:
                        logger.error("Worker raised: %s", exc)
                        n_fallback += 1
                    pbar.update(1)
    else:
        with tqdm(total=len(pending_idxs), desc="LLM calls") as pbar:
            for idx in pending_idxs:
                try:
                    ok, elapsed = _process_one(idx)
                    if ok:
                        n_ok += 1
                        latencies.append(elapsed)
                    else:
                        n_fallback += 1
                except Exception as exc:
                    logger.error("Call failed: %s", exc)
                    n_fallback += 1
                pbar.update(1)

    ckpt_fh.close()

    # ── Finalize: merge checkpoint into parquet ───────────────────────────────
    all_records = _load_checkpoint(ckpt_path)  # reload to get all rows including pre-existing
    # Keep only users in current run's user set
    target_users = set(unique_users)
    final_records = [
        rec for (uid, _), rec in all_records.items() if uid in target_users
    ]

    if not final_records:
        logger.error("No records to save — all calls failed.")
        return

    df_out = pd.DataFrame(final_records)
    for col in ("goal_concepts", "evidence_item_ids"):
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [str(x)])
            )
    df_out.to_parquet(out_path, index=False)
    logger.info("saved -> %s  (%d rows)", out_path, len(df_out))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total = len(df_out)
    logger.info("=" * 60)
    logger.info("4o Pilot Summary")
    logger.info("=" * 60)
    logger.info("total rows     : %d", n_total)
    logger.info("llm_ok         : %d  fallback: %d  fallback_rate: %.1f%%",
                n_ok, n_fallback, 100 * n_fallback / max(1, n_ok + n_fallback))
    if latencies:
        import statistics
        logger.info("latency (s)    : avg=%.2f  p50=%.2f  p95=%.2f",
                    sum(latencies)/len(latencies),
                    statistics.median(latencies),
                    sorted(latencies)[int(len(latencies)*0.95)])
    if "deviation_reason" in df_out.columns:
        logger.info("reason dist    : %s", df_out["deviation_reason"].value_counts().to_dict())
    if "source_mode" in df_out.columns:
        logger.info("source_mode    : %s", df_out["source_mode"].value_counts().to_dict())
    logger.info("checkpoint     : %s", ckpt_path)
    logger.info("output parquet : %s", out_path)
    logger.info("=" * 60)
    logger.info("NEXT: run eval with --llm-intent-path %s", out_path)


if __name__ == "__main__":
    main()
