"""
run_train.py
------------
CLI entry point for SASRec training.

Usage (from project root):
    python -m src.backbone.run_train \
        --data-config config/data/amazon_beauty.yaml \
        --backbone-config config/backbone/sasrec.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.backbone.train_sasrec import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_csv_list(raw: str | None) -> list[int]:
    vals = _parse_csv_list(raw)
    out: list[int] = []
    for v in vals:
        out.append(int(v))
    return out


def _summary_row(label: str, out: dict, split_mode: str, max_seq_len: int) -> dict:
    return {
        "label": label,
        "split_mode": split_mode,
        "max_seq_len": max_seq_len,
        "best_epoch": out.get("best_epoch"),
        "best_full_valid_ndcg@10": out.get("best_full_valid_ndcg@10"),
        "full_test_hr@10": out.get("full_test_hr@10"),
        "full_test_ndcg@10": out.get("full_test_ndcg@10"),
        "full_test_mrr": out.get("full_test_mrr"),
        "checkpoint_path": str(out.get("checkpoint_path")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",     default="config/data/amazon_beauty.yaml")
    parser.add_argument("--backbone-config", default="config/backbone/sasrec.yaml")
    parser.add_argument(
        "--split-modes",
        default=None,
        help="Optional robustness split comparison list, e.g. leave_one_out,chrono_8_2",
    )
    parser.add_argument(
        "--seq-lens",
        default=None,
        help="Optional robustness max_seq_len comparison list, e.g. 30,50,100",
    )
    parser.add_argument(
        "--skip-robustness-train",
        action="store_true",
        help="Only run official training once, skip extra split/seq_len comparisons.",
    )
    args = parser.parse_args()

    data_cfg     = load_yaml(args.data_config)
    backbone_cfg = load_yaml(args.backbone_config)

    interim_dir = Path(data_cfg["paths"]["interim_dir"])
    df_sequences = pd.read_parquet(interim_dir / "user_sequences.parquet")

    sequences: list[list[str]] = [list(s) for s in df_sequences["item_sequence"]]
    logger.info("Loaded %d user sequences", len(sequences))

    # checkpoint output dir: derived from interim_dir parent (e.g. amazon_beauty)
    dataset_name = interim_dir.name
    out_dir = Path("data/checkpoints") / dataset_name

    train_out = train(sequences, backbone_cfg, out_dir)
    ckpt_path = Path(str(train_out["checkpoint_path"]))
    logger.info("Checkpoint saved: %s", ckpt_path)

    # Update backbone config in-place so run_rerank picks it up automatically
    cfg_path = Path(args.backbone_config)
    cfg_raw = load_yaml(str(cfg_path))
    cfg_raw["checkpoint_path"] = str(ckpt_path)
    cfg_raw["use_trained_model"] = True
    # force trained path explicitly; avoids remaining in popularity mode
    cfg_raw["backbone_mode"] = "trained_sasrec"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_raw, f, sort_keys=False, allow_unicode=False)
    logger.info(
        "Updated %s: backbone_mode=trained_sasrec, use_trained_model=true, checkpoint_path=%s",
        cfg_path, ckpt_path,
    )

    split_modes = _parse_csv_list(args.split_modes)
    if not split_modes:
        split_modes = [str(backbone_cfg.get("split_mode", "leave_one_out"))]
    seq_lens = _parse_int_csv_list(args.seq_lens)
    if not seq_lens:
        seq_lens = [int(backbone_cfg.get("max_seq_len", 50))]

    split_rows = [
        _summary_row(
            label="official",
            out=train_out,
            split_mode=str(backbone_cfg.get("split_mode", "leave_one_out")),
            max_seq_len=int(backbone_cfg.get("max_seq_len", 50)),
        )
    ]
    seq_rows = [
        _summary_row(
            label="official",
            out=train_out,
            split_mode=str(backbone_cfg.get("split_mode", "leave_one_out")),
            max_seq_len=int(backbone_cfg.get("max_seq_len", 50)),
        )
    ]

    if not args.skip_robustness_train:
        official_split = str(backbone_cfg.get("split_mode", "leave_one_out"))
        official_len = int(backbone_cfg.get("max_seq_len", 50))

        for sm in split_modes:
            if sm == official_split:
                continue
            cfg_tmp = copy.deepcopy(backbone_cfg)
            cfg_tmp["split_mode"] = sm
            logger.info("Robustness train (split_mode=%s, max_seq_len=%d) start", sm, official_len)
            out_sm = train(sequences, cfg_tmp, out_dir / f"robust_split_{sm}")
            split_rows.append(_summary_row(label=f"robust_split_{sm}", out=out_sm, split_mode=sm, max_seq_len=official_len))

        for sl in seq_lens:
            if sl == official_len:
                continue
            cfg_tmp = copy.deepcopy(backbone_cfg)
            cfg_tmp["max_seq_len"] = int(sl)
            logger.info("Robustness train (max_seq_len=%d, split_mode=%s) start", sl, official_split)
            out_sl = train(sequences, cfg_tmp, out_dir / f"robust_seq_len_{sl}")
            seq_rows.append(_summary_row(label=f"robust_seq_len_{sl}", out=out_sl, split_mode=official_split, max_seq_len=int(sl)))

    split_cmp_path = out_dir / "split_mode_comparison.csv"
    seq_cmp_path = out_dir / "seq_len_comparison.csv"
    pd.DataFrame(split_rows).to_csv(split_cmp_path, index=False)
    pd.DataFrame(seq_rows).to_csv(seq_cmp_path, index=False)
    logger.info("Saved split comparison: %s", split_cmp_path)
    logger.info("Saved seq_len comparison: %s", seq_cmp_path)

    final_report_path = out_dir / "final_backbone_selection_report.md"
    official_ndcg = train_out.get("full_test_ndcg@10")
    official_hr = train_out.get("full_test_hr@10")
    official_mrr = train_out.get("full_test_mrr")
    split_note = (
        "chrono_8_2 robustness rows generated"
        if any(r["split_mode"] == "chrono_8_2" for r in split_rows)
        else "chrono_8_2 robustness row not generated in this run"
    )
    seq_note = (
        "30/50/100 robustness rows generated"
        if {30, 50, 100}.issubset({int(r["max_seq_len"]) for r in seq_rows})
        else "30/50/100 robustness rows not fully generated in this run"
    )
    final_report_lines = [
        "# Final Backbone Selection Report",
        "",
        "## Official Role",
        "- SASRec backbone role: recent continuation modeling for candidate generation/base scoring.",
        "- Long-term stable preference remains in persona branch, not backbone.",
        "",
        "## Official Protocol",
        f"- split_mode: `{backbone_cfg.get('split_mode', 'leave_one_out')}`",
        "- checkpoint selection: `full-ranking valid NDCG@10`",
        "- official report metric: `full-ranking test`",
        f"- official checkpoint: `{ckpt_path}`",
        "",
        "## Official Metrics (Full Test)",
        f"- HR@10: {official_hr}",
        f"- NDCG@10: {official_ndcg}",
        f"- MRR: {official_mrr}",
        "",
        "## Robustness Artifacts",
        f"- split comparison: `{split_cmp_path}` ({split_note})",
        f"- seq_len comparison: `{seq_cmp_path}` ({seq_note})",
        "",
        "## Required Answers",
        "1) SASRec official role: recent continuation backbone (not long-term persona model).",
        "2) leave_one_out official adoption: yes, it aligns with current PGIM pipeline.",
        f"3) chrono_8_2 trend check: see `{split_cmp_path}` rows.",
        f"4) max_seq_len 30/50/100 comparison: see `{seq_cmp_path}` rows; default official value remains 50.",
        f"5) official checkpoint: `{ckpt_path}`.",
        "6) PGIM fixed backbone setting: trained_sasrec + sasrec_best.pt + leave_one_out + full-ranking test reporting.",
        "",
        "현재 sampled valid는 quick monitor로만 사용하고, PGIM 본 실험에서는 sasrec_best.pt + full-ranking test 기준 backbone을 고정해야 한다.",
    ]
    final_report_path.write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    logger.info("Saved final backbone selection report: %s", final_report_path)

    checkpoint_audit_path = out_dir / "checkpoint_audit.txt"
    lines = [
        "SASRec checkpoint audit",
        f"backbone_config_path: {cfg_path}",
        f"checkpoint_path_selected: {ckpt_path}",
        f"backbone_mode_after_train: {cfg_raw.get('backbone_mode')}",
        f"use_trained_model_after_train: {cfg_raw.get('use_trained_model')}",
        f"best_epoch: {train_out.get('best_epoch')}",
        f"best_full_valid_ndcg@10: {train_out.get('best_full_valid_ndcg@10')}",
        f"sampled_valid_metrics_csv: {train_out.get('sampled_valid_metrics_csv')}",
        f"full_valid_metrics_csv: {train_out.get('full_valid_metrics_csv')}",
        f"full_test_metrics_csv: {train_out.get('full_test_metrics_csv')}",
        f"epoch_curve_csv: {train_out.get('epoch_curve_csv')}",
        f"split_protocol_report_md: {train_out.get('split_protocol_report_md')}",
        f"final_backbone_selection_report_md: {final_report_path}",
        f"split_mode_comparison_csv: {split_cmp_path}",
        f"seq_len_comparison_csv: {seq_cmp_path}",
    ]
    checkpoint_audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved checkpoint audit: %s", checkpoint_audit_path)


if __name__ == "__main__":
    main()
