"""
model.py
--------
SASRec model — faithful port of the official pmixer/SASRec.pytorch implementation.

Source: https://github.com/pmixer/SASRec.pytorch/blob/main/python/model.py
Changes from original (interface only, architecture unchanged):
    - `log2feats` accepts a torch.Tensor directly (no internal np.array conversion)
      so the wrapper can pass tensors without extra round-trips.
    - `forward` / `predict` accept tensors (not numpy arrays).
    - `args` namespace replaced by explicit keyword arguments for clarity.
    - device stored as torch.device, not string.

Architecture (identical to paper):
    - Item embedding  : Embedding(item_num+1, hidden_units, padding_idx=0)
    - Pos embedding   : Embedding(maxlen+1,   hidden_units, padding_idx=0)
                        position index = 0 at PAD positions (masked out)
    - N x blocks      : Pre-LN or Post-LN (norm_first flag), MHA + Conv1d FFN
    - Loss (training) : BCE with logits on (positive, sampled-negative) pairs
    - Inference       : dot-product of last-position feature vs item embeddings
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """Identical to official: Conv1d → ReLU → Conv1d (with dropout)."""

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1    = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu     = nn.ReLU()
        self.conv2    = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, B, C) or (B, T, C) — Conv1d needs (N, C, L)
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        return outputs.transpose(-1, -2)


class SASRec(nn.Module):
    """
    SASRec (official architecture).

    Parameters
    ----------
    user_num    : total number of users (not used in forward, kept for API compat)
    item_num    : total number of items (vocab size, 1-indexed)
    hidden_units: embedding / hidden dimension
    maxlen      : maximum sequence length
    num_blocks  : number of transformer blocks
    num_heads   : number of attention heads
    dropout_rate: dropout probability
    norm_first  : Pre-LN (True) or Post-LN (False, original paper default)
    device      : torch.device
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        hidden_units: int  = 64,
        maxlen: int        = 50,
        num_blocks: int    = 2,
        num_heads: int     = 2,
        dropout_rate: float = 0.2,
        norm_first: bool   = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.user_num   = user_num
        self.item_num   = item_num
        self.dev        = device or torch.device("cpu")
        self.norm_first = norm_first

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen  + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers     = nn.ModuleList()
        self.forward_layernorms   = nn.ModuleList()
        self.forward_layers       = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout_rate)
            )
            self.forward_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of item sequences into contextual features.

        Parameters
        ----------
        log_seqs : (B, T) long tensor, 0 = PAD (right-padded OR left-padded,
                   but official training uses RIGHT-padded / tail-truncated seqs)

        Returns
        -------
        (B, T, hidden_units)
        """
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5   # scale as in original

        # position indices: 1-indexed, 0 at PAD positions
        # official: np.tile(np.arange(1, T+1), [B, 1]) * (log_seqs != 0)
        B, T = log_seqs.shape
        poss = torch.arange(1, T + 1, device=self.dev).unsqueeze(0).expand(B, T)
        poss = poss * (log_seqs.to(self.dev) != 0).long()   # zero out PAD positions
        seqs = seqs + self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # causal mask: bool, True = blocked  (upper-triangular excluding diagonal)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            # MHA expects (T, B, C) when batch_first=False (default)
            seqs = seqs.transpose(0, 1)

            if self.norm_first:
                # Pre-LN
                x = self.attention_layernorms[i](seqs)
                mha_out, _ = self.attention_layers[i](
                    x, x, x, attn_mask=attention_mask
                )
                seqs = seqs + mha_out
                seqs = seqs.transpose(0, 1)
                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs)
                )
            else:
                # Post-LN (original paper default)
                mha_out, _ = self.attention_layers[i](
                    seqs, seqs, seqs, attn_mask=attention_mask
                )
                seqs = self.attention_layernorms[i](seqs + mha_out)
                seqs = seqs.transpose(0, 1)
                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs)
                )

        log_feats = self.last_layernorm(seqs)   # (B, T, C)
        return log_feats

    # ------------------------------------------------------------------
    # Training forward: BCE on positive / negative pairs
    # ------------------------------------------------------------------
    def forward(
        self,
        user_ids: torch.Tensor,   # (B,)   — unused but kept for API compat
        log_seqs: torch.Tensor,   # (B, T)
        pos_seqs: torch.Tensor,   # (B, T)  positive next-items
        neg_seqs: torch.Tensor,   # (B, T)  sampled negatives
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_feats = self.log2feats(log_seqs)   # (B, T, C)

        pos_embs = self.item_emb(pos_seqs.to(self.dev))   # (B, T, C)
        neg_embs = self.item_emb(neg_seqs.to(self.dev))   # (B, T, C)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)   # (B, T)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)   # (B, T)

        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    # Inference: score a list of candidate items for each user
    # ------------------------------------------------------------------
    def predict(
        self,
        user_ids: torch.Tensor,       # (B,)   — unused
        log_seqs: torch.Tensor,       # (B, T)
        item_indices: torch.Tensor,   # (B, I) or (I,)
    ) -> torch.Tensor:
        """
        Returns (B, I) logits.
        Uses only the last-position feature (as in official code).
        """
        log_feats  = self.log2feats(log_seqs)          # (B, T, C)
        final_feat = log_feats[:, -1, :]               # (B, C)

        item_embs = self.item_emb(item_indices.to(self.dev))  # (I, C) or (B, I, C)

        if item_embs.dim() == 2:
            # shared item list for all users: (B, C) x (C, I) -> (B, I)
            logits = final_feat @ item_embs.T
        else:
            # per-user item list: (B, I, C) · (B, C, 1) -> (B, I)
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
