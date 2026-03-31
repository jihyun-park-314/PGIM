"""
dataset.py
----------
Data preparation for SASRec training — faithful to pmixer/SASRec.pytorch.

Split (leave-one-out, identical to official):
    train : all items except last two
    valid : second-to-last item
    test  : last item  (= the eval target used by ranking_eval.py)

Sampling (identical to official utils.py sample_function):
    For each training step, given a user sequence, slide a window of length
    `maxlen` and for each position t:
        seq[t]   = context ending at t
        pos[t]   = item at t+1  (positive)
        neg[t]   = random item not in user's history  (negative)

All item indices are 1-based (0 = PAD).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset


def build_item_vocab(sequences: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """1-indexed vocab. Index 0 reserved for PAD."""
    all_items = sorted({item for seq in sequences for item in seq})
    item2idx  = {item: i + 1 for i, item in enumerate(all_items)}
    idx2item  = {v: k for k, v in item2idx.items()}
    return item2idx, idx2item


def data_partition(
    idx_sequences: list[list[int]],
) -> tuple[list[list[int]], list[int], list[int]]:
    """
    Split each integer sequence into train / valid / test.

    Returns
    -------
    user_train : list[list[int]]   sequences for training (all but last 2)
    user_valid : list[int]         one validation target per user (item at -2)
    user_test  : list[int]         one test target per user      (item at -1)
    """
    user_train: list[list[int]] = []
    user_valid: list[int] = []
    user_test:  list[int] = []

    for seq in idx_sequences:
        if len(seq) < 4:
            # too short for proper split — put everything in train
            user_train.append(seq[:])
            user_valid.append(0)
            user_test.append(0)
        else:
            user_train.append(seq[:-2])
            user_valid.append(seq[-2])
            user_test.append(seq[-1])

    return user_train, user_valid, user_test


class SASRecTrainDataset(Dataset):
    """
    Training dataset using the same sliding-window negative-sampling
    strategy as the official WarpSampler.

    For each user, produces (seq, pos, neg) arrays of length `maxlen`:
        seq[t] = t-th history item (right-padded with 0 if shorter)
        pos[t] = item at position t+1 in user history
        neg[t] = random item not in user history
    PAD positions have seq=pos=neg=0.
    """

    def __init__(
        self,
        user_train: list[list[int]],
        item_num: int,
        maxlen: int = 50,
    ) -> None:
        self.maxlen    = maxlen
        self.item_num  = item_num
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for seq in user_train:
            if len(seq) <= 1:
                continue
            ts = set(seq)

            seq_arr = np.zeros(maxlen, dtype=np.int32)
            pos_arr = np.zeros(maxlen, dtype=np.int32)
            neg_arr = np.zeros(maxlen, dtype=np.int32)

            nxt = seq[-1]
            idx = maxlen - 1

            for item in reversed(seq[:-1]):
                seq_arr[idx] = item
                pos_arr[idx] = nxt
                neg_arr[idx] = self._random_neg(ts)
                nxt = item
                idx -= 1
                if idx == -1:
                    break

            self.samples.append((seq_arr, pos_arr, neg_arr))

    def _random_neg(self, ts: set) -> int:
        t = random.randint(1, self.item_num)
        while t in ts:
            t = random.randint(1, self.item_num)
        return t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq, pos, neg = self.samples[idx]
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "pos": torch.tensor(pos, dtype=torch.long),
            "neg": torch.tensor(neg, dtype=torch.long),
        }


class SASRecEvalDataset(Dataset):
    """
    Evaluation dataset (valid or test split).

    For each user, produces:
        seq     : history of length `maxlen` (right-padded, includes valid item for test)
        target  : single target item index
        rated   : set of items in user history (for negative sampling in 100-item eval)
    """

    def __init__(
        self,
        user_train: list[list[int]],
        targets: list[int],
        maxlen: int = 50,
    ) -> None:
        self.maxlen  = maxlen
        self.samples: list[tuple[np.ndarray, int, set[int]]] = []

        for train_seq, tgt in zip(user_train, targets):
            if tgt == 0 or len(train_seq) < 1:
                continue
            seq = np.zeros(maxlen, dtype=np.int32)
            idx = maxlen - 1
            for item in reversed(train_seq):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break
            rated = set(train_seq)
            rated.add(0)
            self.samples.append((seq, tgt, rated))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        seq, tgt, rated = self.samples[idx]
        return {
            "seq":   torch.tensor(seq, dtype=torch.long),
            "target": tgt,
            "rated":  rated,
        }
