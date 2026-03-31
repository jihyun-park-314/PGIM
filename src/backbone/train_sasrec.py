"""
train_sasrec.py
---------------
Training loop — faithful to pmixer/SASRec.pytorch.

Loss    : BCE with logits on (positive, sampled-negative) pairs
          (identical to official main.py)
Eval    : NDCG@10, HR@10 on 101-item sampled ranking
          (1 ground-truth + 100 random negatives, identical to official evaluate_valid)
Optimizer: Adam with optional L2 regularization on embeddings

Checkpoint:
    data/checkpoints/<dataset>/sasrec_best.pt
    {model_state, item2idx, idx2item, config, best_ndcg, epoch}
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.backbone.dataset import (
    SASRecEvalDataset,
    SASRecTrainDataset,
    build_item_vocab,
    data_partition,
)
from src.backbone.model import SASRec

logger = logging.getLogger(__name__)


def _evaluate_sampled(
    model: SASRec,
    eval_ds: SASRecEvalDataset,
    item_num: int,
    k: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    NDCG@k and HR@k using 101-item sampled evaluation
    (1 ground-truth + 100 random negatives) — identical to official evaluate_valid.
    """
    model.eval()
    ndcg_sum = hr_sum = 0.0
    n = 0

    with torch.no_grad():
        for sample in eval_ds.samples:
            seq_np, tgt, rated = sample

            # 100 random negatives not in user history
            neg_items: list[int] = []
            while len(neg_items) < 100:
                t = random.randint(1, item_num)
                if t not in rated:
                    neg_items.append(t)

            item_idx = [tgt] + neg_items          # [101]
            item_tensor = torch.tensor(item_idx, dtype=torch.long, device=device)
            seq_tensor  = torch.tensor(seq_np,   dtype=torch.long, device=device).unsqueeze(0)
            uid_tensor  = torch.zeros(1, dtype=torch.long, device=device)

            logits = model.predict(uid_tensor, seq_tensor, item_tensor)  # (1, 101)
            # rank of ground truth (index 0) among 101 items (desc)
            scores = logits[0]                                            # (101,)
            rank = (scores > scores[0]).sum().item() + 1                 # 1-indexed

            if rank <= k:
                ndcg_sum += 1.0 / math.log2(rank + 1)
                hr_sum   += 1.0
            n += 1

    ndcg = ndcg_sum / n if n > 0 else 0.0
    hr   = hr_sum   / n if n > 0 else 0.0
    return ndcg, hr


def train(
    sequences: list[list[str]],
    backbone_cfg: dict,
    out_dir: Path,
) -> Path:
    """
    Train SASRec on string item sequences.

    Parameters
    ----------
    sequences    : list of full item_id string sequences, one per user
    backbone_cfg : config/backbone/sasrec.yaml contents
    out_dir      : directory to save the checkpoint

    Returns
    -------
    Path to saved checkpoint.
    """
    hidden_units  = backbone_cfg.get("model_dim",   64)
    num_blocks    = backbone_cfg.get("num_blocks",   2)
    num_heads     = backbone_cfg.get("num_heads",    2)
    dropout_rate  = backbone_cfg.get("dropout",    0.2)
    maxlen        = backbone_cfg.get("max_seq_len", 50)
    batch_size    = backbone_cfg.get("batch_size", 128)
    num_epochs    = backbone_cfg.get("num_epochs", 100)
    lr            = backbone_cfg.get("learning_rate", 1e-3)
    l2_emb        = backbone_cfg.get("l2_emb", 0.0)
    patience      = backbone_cfg.get("early_stopping_patience", 10)
    eval_k        = backbone_cfg.get("eval_ndcg_k", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # ── vocabulary ──────────────────────────────────────────────────────
    item2idx, idx2item = build_item_vocab(sequences)
    item_num   = len(item2idx)
    user_num   = len(sequences)
    logger.info("Users: %d  |  Items: %d", user_num, item_num)

    idx_sequences = [
        [item2idx[item] for item in seq if item in item2idx]
        for seq in sequences
    ]

    # ── data split (official leave-one-out) ─────────────────────────────
    user_train, user_valid, user_test = data_partition(idx_sequences)

    train_ds = SASRecTrainDataset(user_train, item_num, maxlen=maxlen)
    valid_ds = SASRecEvalDataset(user_train, user_valid, maxlen=maxlen)
    logger.info(
        "Train windows: %d  |  Valid users: %d",
        len(train_ds), len(valid_ds),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0,
    )

    # ── model ───────────────────────────────────────────────────────────
    model = SASRec(
        user_num=user_num,
        item_num=item_num,
        hidden_units=hidden_units,
        maxlen=maxlen,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        norm_first=False,   # Post-LN (original paper default)
        device=device,
    ).to(device)

    # ── optimizer (identical to official: Adam, no scheduler) ───────────
    # official uses Adam without scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98)
    )

    bce = nn.BCEWithLogitsLoss()

    # ── training loop ───────────────────────────────────────────────────
    best_ndcg   = -1.0
    best_epoch  = 0
    no_improve  = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path   = out_dir / "sasrec_best.pt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)   # (B, T)
            pos = batch["pos"].to(device)   # (B, T)
            neg = batch["neg"].to(device)   # (B, T)

            # dummy user_ids (not used in model)
            uid = torch.zeros(seq.shape[0], dtype=torch.long, device=device)

            pos_logits, neg_logits = model(uid, seq, pos, neg)

            # mask PAD positions (pos == 0)
            mask = (pos != 0).float()

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            loss = bce(pos_logits * mask, pos_labels * mask) + \
                   bce(neg_logits * mask, neg_labels * mask)

            # L2 regularization on embedding weights (official comment mentions this)
            if l2_emb > 0:
                loss += l2_emb * (
                    model.item_emb.weight.norm(2) +
                    model.pos_emb.weight.norm(2)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches if n_batches else float("nan")

        # ── validation every 5 epochs ───────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            ndcg, hr = _evaluate_sampled(model, valid_ds, item_num, eval_k, device)
            logger.info(
                "epoch %3d  loss=%.4f  NDCG@%d=%.4f  HR@%d=%.4f",
                epoch, avg_loss, eval_k, ndcg, eval_k, hr,
            )

            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "item2idx":    item2idx,
                    "idx2item":    idx2item,
                    "config": {
                        "user_num":     user_num,
                        "item_num":     item_num,
                        "hidden_units": hidden_units,
                        "num_blocks":   num_blocks,
                        "num_heads":    num_heads,
                        "maxlen":       maxlen,
                        "dropout_rate": dropout_rate,
                    },
                    "best_ndcg": best_ndcg,
                    "epoch":     epoch,
                }, ckpt_path)
                logger.info(
                    "  -> saved best  NDCG@%d=%.4f  (%s)",
                    eval_k, best_ndcg, ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stopping: no improvement for %d evals (epoch %d)",
                        no_improve, epoch,
                    )
                    break
        else:
            if epoch % 20 == 0:
                logger.info("epoch %3d  loss=%.4f", epoch, avg_loss)

    logger.info(
        "Training done. Best NDCG@%d=%.4f at epoch %d",
        eval_k, best_ndcg, best_epoch,
    )
    return ckpt_path
