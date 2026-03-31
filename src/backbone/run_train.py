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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config",     default="config/data/amazon_beauty.yaml")
    parser.add_argument("--backbone-config", default="config/backbone/sasrec.yaml")
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

    ckpt_path = train(sequences, backbone_cfg, out_dir)
    logger.info("Checkpoint saved: %s", ckpt_path)

    # Update backbone config in-place so run_rerank picks it up automatically
    cfg_path = Path(args.backbone_config)
    raw = cfg_path.read_text()
    updated = False
    if "checkpoint_path: null" in raw:
        raw = raw.replace("checkpoint_path: null", f"checkpoint_path: {ckpt_path}")
        updated = True
    if "use_trained_model: false" in raw:
        raw = raw.replace("use_trained_model: false", "use_trained_model: true")
        updated = True
    if updated:
        cfg_path.write_text(raw)
        logger.info("Updated %s: checkpoint_path -> %s", cfg_path, ckpt_path)


if __name__ == "__main__":
    main()
