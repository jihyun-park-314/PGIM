"""
train_sasrec.py
---------------
Training loop — faithful to pmixer/SASRec.pytorch.

Loss    : BCE with logits on (positive, sampled-negative) pairs
          (identical to official main.py)
Eval    : sampled valid (quick monitor) + full-ranking valid/test (official)
Optimizer: Adam with optional L2 regularization on embeddings

Checkpoint:
    data/checkpoints/<dataset>/sasrec_best.pt
    {model_state, item2idx, idx2item, config, best_ndcg_full_valid, epoch}
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    max_eval_users: int | None = None,
) -> tuple[float, float]:
    """
    NDCG@k and HR@k using 101-item sampled evaluation
    (1 ground-truth + 100 random negatives) — identical to official evaluate_valid.
    """
    model.eval()
    ndcg_sum = hr_sum = 0.0
    n = 0

    samples = eval_ds.samples
    if max_eval_users is not None and max_eval_users > 0:
        samples = samples[:max_eval_users]
        logger.info("Validation subset enabled: %d users", len(samples))
    else:
        logger.info("Validation users: %d", len(samples))

    it = tqdm(
        samples,
        total=len(samples),
        desc="valid",
        leave=False,
        disable=not sys.stdout.isatty(),
    )

    with torch.no_grad():
        for sample in it:
            seq_np, tgt, rated = sample

            # 100 random negatives not in user history
            neg_items: list[int] = []
            while len(neg_items) < 100:
                t = random.randint(1, item_num)
                if t not in rated:
                    neg_items.append(t)

            item_idx = [tgt] + neg_items          # [101]
            item_tensor = torch.tensor(item_idx, dtype=torch.long, device=device)
            seq_tensor = torch.tensor(seq_np, dtype=torch.long, device=device).unsqueeze(0)
            uid_tensor = torch.zeros(1, dtype=torch.long, device=device)

            logits = model.predict(uid_tensor, seq_tensor, item_tensor)  # (1, 101)
            # rank of ground truth (index 0) among 101 items (desc)
            scores = logits[0]                                            # (101,)
            rank = (scores > scores[0]).sum().item() + 1                 # 1-indexed

            if rank <= k:
                ndcg_sum += 1.0 / math.log2(rank + 1)
                hr_sum += 1.0
            n += 1

    ndcg = ndcg_sum / n if n > 0 else 0.0
    hr = hr_sum / n if n > 0 else 0.0
    return ndcg, hr


def _metrics_from_rank(rank: int) -> dict[str, float]:
    return {
        "HR@5": 1.0 if rank <= 5 else 0.0,
        "HR@10": 1.0 if rank <= 10 else 0.0,
        "NDCG@5": (1.0 / math.log2(rank + 1)) if rank <= 5 else 0.0,
        "NDCG@10": (1.0 / math.log2(rank + 1)) if rank <= 10 else 0.0,
        "MRR": 1.0 / rank,
    }


def _evaluate_full_ranking(
    model: SASRec,
    eval_ds: SASRecEvalDataset,
    item_num: int,
    device: torch.device,
    phase: str,
    max_eval_users: int | None = None,
) -> dict[str, float]:
    """
    Full-ranking evaluation over all unseen items.
    Candidate set per user: all items not in rated history (plus GT).
    """
    model.eval()
    samples = eval_ds.samples
    if max_eval_users is not None and max_eval_users > 0:
        samples = samples[:max_eval_users]
        logger.info("%s full-ranking subset enabled: %d users", phase, len(samples))
    else:
        logger.info("%s full-ranking users: %d", phase, len(samples))

    if len(samples) == 0:
        return {
            "n_users": 0.0,
            "HR@5": 0.0,
            "HR@10": 0.0,
            "NDCG@5": 0.0,
            "NDCG@10": 0.0,
            "MRR": 0.0,
            "mean_rank": float("nan"),
        }

    all_item_tensor = torch.arange(1, item_num + 1, dtype=torch.long, device=device)
    metric_sum = {"HR@5": 0.0, "HR@10": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "MRR": 0.0}
    rank_sum = 0.0

    it = tqdm(
        samples,
        total=len(samples),
        desc=f"{phase}_full",
        leave=False,
        disable=not sys.stdout.isatty(),
    )

    with torch.no_grad():
        for seq_np, tgt, rated in it:
            seq_tensor = torch.tensor(seq_np, dtype=torch.long, device=device).unsqueeze(0)
            uid_tensor = torch.zeros(1, dtype=torch.long, device=device)

            scores = model.predict(uid_tensor, seq_tensor, all_item_tensor).squeeze(0)

            # Exclude rated history from ranking (except GT itself).
            mask_ids = [iid for iid in rated if iid != 0 and iid != tgt]
            if mask_ids:
                mask_idx = torch.tensor(mask_ids, dtype=torch.long, device=device) - 1
                scores[mask_idx] = float("-inf")

            gt_score = scores[tgt - 1]
            rank = int((scores > gt_score).sum().item() + 1)
            rank_sum += rank

            m = _metrics_from_rank(rank)
            for k, v in m.items():
                metric_sum[k] += v

    n = float(len(samples))
    return {
        "n_users": n,
        "HR@5": metric_sum["HR@5"] / n,
        "HR@10": metric_sum["HR@10"] / n,
        "NDCG@5": metric_sum["NDCG@5"] / n,
        "NDCG@10": metric_sum["NDCG@10"] / n,
        "MRR": metric_sum["MRR"] / n,
        "mean_rank": rank_sum / n,
    }


def evaluate_full_ranking_valid(
    model: SASRec,
    valid_ds: SASRecEvalDataset,
    item_num: int,
    device: torch.device,
    max_eval_users: int | None = None,
) -> dict[str, float]:
    return _evaluate_full_ranking(
        model=model,
        eval_ds=valid_ds,
        item_num=item_num,
        device=device,
        phase="valid",
        max_eval_users=max_eval_users,
    )


def evaluate_full_ranking_test(
    model: SASRec,
    test_ds: SASRecEvalDataset,
    item_num: int,
    device: torch.device,
    max_eval_users: int | None = None,
) -> dict[str, float]:
    return _evaluate_full_ranking(
        model=model,
        eval_ds=test_ds,
        item_num=item_num,
        device=device,
        phase="test",
        max_eval_users=max_eval_users,
    )


def train(
    sequences: list[list[str]],
    backbone_cfg: dict,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Train SASRec on string item sequences.

    Parameters
    ----------
    sequences    : list of full item_id string sequences, one per user
    backbone_cfg : config/backbone/sasrec.yaml contents
    out_dir      : directory to save the checkpoint

    Returns
    -------
    Dict with checkpoint path and evaluation artifact paths.
    """
    hidden_units = backbone_cfg.get("model_dim", 64)
    num_blocks = backbone_cfg.get("num_blocks", 2)
    num_heads = backbone_cfg.get("num_heads", 2)
    dropout_rate = backbone_cfg.get("dropout", 0.2)
    maxlen = backbone_cfg.get("max_seq_len", 50)
    batch_size = backbone_cfg.get("batch_size", 128)
    num_epochs = backbone_cfg.get("num_epochs", 100)
    lr = backbone_cfg.get("learning_rate", 1e-3)
    l2_emb = backbone_cfg.get("l2_emb", 0.0)
    patience = backbone_cfg.get("early_stopping_patience", 10)
    eval_k = backbone_cfg.get("eval_ndcg_k", 10)
    eval_every = int(backbone_cfg.get("eval_every_n_epochs", 1))
    full_eval_every = int(backbone_cfg.get("full_eval_every_n_epochs", 1))
    save_every_ep = bool(backbone_cfg.get("save_every_epoch", True))
    save_opt_state = bool(backbone_cfg.get("save_optimizer_state", False))
    max_eval_users = backbone_cfg.get("eval_max_users", None)
    if max_eval_users is not None:
        max_eval_users = int(max_eval_users)
    max_full_eval_users = backbone_cfg.get("full_eval_max_users", None)
    if max_full_eval_users is not None:
        max_full_eval_users = int(max_full_eval_users)
    split_mode = str(backbone_cfg.get("split_mode", "leave_one_out"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # ── vocabulary ──────────────────────────────────────────────────────
    item2idx, idx2item = build_item_vocab(sequences)
    item_num = len(item2idx)
    user_num = len(sequences)
    logger.info("Users: %d  |  Items: %d", user_num, item_num)

    idx_sequences = [
        [item2idx[item] for item in seq if item in item2idx]
        for seq in sequences
    ]

    # ── data split (default: official leave-one-out) ───────────────────
    user_train, user_valid, user_test = data_partition(
        idx_sequences,
        split_mode=split_mode,
    )

    train_ds = SASRecTrainDataset(user_train, item_num, maxlen=maxlen)
    valid_ds = SASRecEvalDataset(user_train, user_valid, maxlen=maxlen)
    user_train_plus_valid = [
        (tr + [va]) if va != 0 else tr[:]
        for tr, va in zip(user_train, user_valid)
    ]
    test_ds = SASRecEvalDataset(user_train_plus_valid, user_test, maxlen=maxlen)

    logger.info(
        "Split mode=%s  |  Train windows=%d  |  Valid users=%d  |  Test users=%d",
        split_mode,
        len(train_ds),
        len(valid_ds),
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
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

    # ── optimizer (identical to official: Adam, no scheduler) ──────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    bce = nn.BCEWithLogitsLoss()

    # ── training loop ───────────────────────────────────────────────────
    best_full_valid_ndcg = -1.0
    best_epoch = 0
    no_improve = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sasrec_best.pt"
    epoch_ckpt_dir = out_dir / "epochs"
    if save_every_ep:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)

    sampled_valid_rows: list[dict[str, Any]] = []
    full_valid_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        logger.info("epoch %3d/%d start", epoch, num_epochs)

        pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch}/{num_epochs}",
            leave=False,
            disable=not sys.stdout.isatty(),
        )
        for batch in pbar:
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

            loss = bce(pos_logits * mask, pos_labels * mask) + bce(neg_logits * mask, neg_labels * mask)

            # L2 regularization on embedding weights (official comment mentions this)
            if l2_emb > 0:
                loss += l2_emb * (
                    model.item_emb.weight.norm(2) + model.pos_emb.weight.norm(2)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if n_batches % 10 == 0:
                pbar.set_postfix(loss=f"{(total_loss / n_batches):.4f}")

        avg_loss = total_loss / n_batches if n_batches else float("nan")

        sampled_ndcg = sampled_hr = None
        full_valid = None

        # Save train-end snapshot before potentially long validation.
        if save_every_ep:
            ep_train_path = epoch_ckpt_dir / f"sasrec_epoch_{epoch:03d}_trainend.pt"
            payload_train = {
                "model_state": model.state_dict(),
                "item2idx": item2idx,
                "idx2item": idx2item,
                "config": {
                    "user_num": user_num,
                    "item_num": item_num,
                    "hidden_units": hidden_units,
                    "num_blocks": num_blocks,
                    "num_heads": num_heads,
                    "maxlen": maxlen,
                    "dropout_rate": dropout_rate,
                },
                "epoch": epoch,
                "avg_loss": avg_loss,
                "phase": "train_end_pre_eval",
                "best_ndcg_so_far": best_full_valid_ndcg,
            }
            if save_opt_state:
                payload_train["optimizer_state"] = optimizer.state_dict()
            torch.save(payload_train, ep_train_path)
            logger.info("  -> saved pre-eval epoch ckpt: %s", ep_train_path)

        # quick monitor (sampled valid)
        if (epoch % eval_every == 0) or (epoch == 1):
            logger.info("epoch %3d sampled-valid start", epoch)
            sampled_ndcg, sampled_hr = _evaluate_sampled(
                model=model,
                eval_ds=valid_ds,
                item_num=item_num,
                k=eval_k,
                device=device,
                max_eval_users=max_eval_users,
            )
            logger.info(
                "epoch %3d sampled-valid  loss=%.4f  NDCG@%d=%.4f  HR@%d=%.4f",
                epoch,
                avg_loss,
                eval_k,
                sampled_ndcg,
                eval_k,
                sampled_hr,
            )
            sampled_valid_rows.append(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "NDCG@10_sampled_valid": sampled_ndcg,
                    "HR@10_sampled_valid": sampled_hr,
                    "k": eval_k,
                }
            )

        # official selection signal (full-ranking valid)
        if (epoch % full_eval_every == 0) or (epoch == 1):
            logger.info("epoch %3d full-valid start", epoch)
            full_valid = evaluate_full_ranking_valid(
                model=model,
                valid_ds=valid_ds,
                item_num=item_num,
                device=device,
                max_eval_users=max_full_eval_users,
            )
            logger.info(
                "epoch %3d full-valid  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f",
                epoch,
                full_valid["HR@10"],
                full_valid["NDCG@10"],
                full_valid["MRR"],
            )
            full_valid_rows.append({"epoch": epoch, **full_valid})
            current_full_ndcg = float(full_valid["NDCG@10"])

            if current_full_ndcg > best_full_valid_ndcg:
                best_full_valid_ndcg = current_full_ndcg
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "item2idx": item2idx,
                        "idx2item": idx2item,
                        "config": {
                            "user_num": user_num,
                            "item_num": item_num,
                            "hidden_units": hidden_units,
                            "num_blocks": num_blocks,
                            "num_heads": num_heads,
                            "maxlen": maxlen,
                            "dropout_rate": dropout_rate,
                        },
                        "best_ndcg_full_valid": best_full_valid_ndcg,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                logger.info(
                    "  -> saved best (full-valid) NDCG@10=%.4f  (%s)",
                    best_full_valid_ndcg,
                    ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(
                        "Early stopping: no improvement for %d full-valid evals (epoch %d)",
                        no_improve,
                        epoch,
                    )
                    # Save final epoch snapshot before stopping (if enabled)
                    if save_every_ep:
                        ep_path = epoch_ckpt_dir / f"sasrec_epoch_{epoch:03d}.pt"
                        payload = {
                            "model_state": model.state_dict(),
                            "item2idx": item2idx,
                            "idx2item": idx2item,
                            "config": {
                                "user_num": user_num,
                                "item_num": item_num,
                                "hidden_units": hidden_units,
                                "num_blocks": num_blocks,
                                "num_heads": num_heads,
                                "maxlen": maxlen,
                                "dropout_rate": dropout_rate,
                            },
                            "epoch": epoch,
                            "avg_loss": avg_loss,
                            "valid_ndcg_sampled": sampled_ndcg,
                            "valid_hr_sampled": sampled_hr,
                            "valid_ndcg_full": (full_valid["NDCG@10"] if full_valid else None),
                            "best_ndcg_so_far": best_full_valid_ndcg,
                        }
                        if save_opt_state:
                            payload["optimizer_state"] = optimizer.state_dict()
                        torch.save(payload, ep_path)
                        logger.info("  -> saved epoch ckpt: %s", ep_path)
                    break
        else:
            logger.info("epoch %3d  loss=%.4f", epoch, avg_loss)

        # Save every epoch checkpoint (default enabled).
        if save_every_ep:
            ep_path = epoch_ckpt_dir / f"sasrec_epoch_{epoch:03d}.pt"
            payload = {
                "model_state": model.state_dict(),
                "item2idx": item2idx,
                "idx2item": idx2item,
                "config": {
                    "user_num": user_num,
                    "item_num": item_num,
                    "hidden_units": hidden_units,
                    "num_blocks": num_blocks,
                    "num_heads": num_heads,
                    "maxlen": maxlen,
                    "dropout_rate": dropout_rate,
                },
                "epoch": epoch,
                "avg_loss": avg_loss,
                "valid_ndcg_sampled": sampled_ndcg,
                "valid_hr_sampled": sampled_hr,
                "valid_ndcg_full": (full_valid["NDCG@10"] if full_valid else None),
                "best_ndcg_so_far": best_full_valid_ndcg,
            }
            if save_opt_state:
                payload["optimizer_state"] = optimizer.state_dict()
            torch.save(payload, ep_path)
            logger.info("  -> saved epoch ckpt: %s", ep_path)

        curve_rows.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "sampled_valid_ndcg@10": sampled_ndcg,
                "sampled_valid_hr@10": sampled_hr,
                "full_valid_ndcg@10": (full_valid["NDCG@10"] if full_valid else None),
                "full_valid_hr@10": (full_valid["HR@10"] if full_valid else None),
                "full_valid_mrr": (full_valid["MRR"] if full_valid else None),
            }
        )

    logger.info(
        "Training done. Best full-valid NDCG@10=%.4f at epoch %d",
        best_full_valid_ndcg,
        best_epoch,
    )

    if ckpt_path.exists():
        best_payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        model.eval()

    logger.info("Running final full-ranking test with best checkpoint...")
    full_test = evaluate_full_ranking_test(
        model=model,
        test_ds=test_ds,
        item_num=item_num,
        device=device,
        max_eval_users=max_full_eval_users,
    )
    logger.info(
        "Final full-test  HR@10=%.4f  NDCG@10=%.4f  MRR=%.4f",
        full_test["HR@10"],
        full_test["NDCG@10"],
        full_test["MRR"],
    )

    sampled_csv = out_dir / "sampled_valid_metrics.csv"
    full_valid_csv = out_dir / "full_valid_metrics.csv"
    full_test_csv = out_dir / "full_test_metrics.csv"
    curve_csv = out_dir / "epoch_level_training_curve.csv"
    split_report_md = out_dir / "split_protocol_report.md"
    final_report_md = out_dir / "final_backbone_selection_report.md"

    pd.DataFrame(sampled_valid_rows).to_csv(sampled_csv, index=False)
    pd.DataFrame(full_valid_rows).to_csv(full_valid_csv, index=False)
    pd.DataFrame([{"best_epoch": best_epoch, **full_test}]).to_csv(full_test_csv, index=False)
    pd.DataFrame(curve_rows).to_csv(curve_csv, index=False)

    split_report_md.write_text(
        "\n".join(
            [
                "# Split Protocol Report",
                "",
                f"- split_mode: `{split_mode}`",
                "- random split: `disabled`",
                "- sampled valid: `1 GT + 100 random negatives` (quick monitor only)",
                "- official model selection: `full-ranking valid NDCG@10`",
                "- official report metric: `full-ranking test`",
            ]
        ),
        encoding="utf-8",
    )
    final_report_md.write_text(
        "\n".join(
            [
                "# Final Backbone Selection Report",
                "",
                f"- best checkpoint: `{ckpt_path}`",
                f"- best epoch (full-valid NDCG@10): `{best_epoch}`",
                f"- best full-valid NDCG@10: `{best_full_valid_ndcg:.6f}`",
                (
                    f"- full-test metrics: HR@5={full_test['HR@5']:.6f}, "
                    f"HR@10={full_test['HR@10']:.6f}, "
                    f"NDCG@5={full_test['NDCG@5']:.6f}, "
                    f"NDCG@10={full_test['NDCG@10']:.6f}, "
                    f"MRR={full_test['MRR']:.6f}"
                ),
            ]
        ),
        encoding="utf-8",
    )

    return {
        "checkpoint_path": ckpt_path,
        "best_epoch": best_epoch,
        "best_full_valid_ndcg@10": best_full_valid_ndcg,
        "full_test_hr@10": float(full_test["HR@10"]),
        "full_test_ndcg@10": float(full_test["NDCG@10"]),
        "full_test_mrr": float(full_test["MRR"]),
        "split_mode": split_mode,
        "max_seq_len": int(maxlen),
        "sampled_valid_metrics_csv": sampled_csv,
        "full_valid_metrics_csv": full_valid_csv,
        "full_test_metrics_csv": full_test_csv,
        "epoch_curve_csv": curve_csv,
        "split_protocol_report_md": split_report_md,
        "final_backbone_selection_report_md": final_report_md,
    }
