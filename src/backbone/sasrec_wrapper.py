"""
SASRecWrapper
-------------
Unified backbone wrapper. Mode is controlled by backbone_cfg["backbone_mode"]:

  popularity     (default for large datasets)
      Global popularity top-K. Always fast, good gt_coverage on large item spaces.
      Filter by source_service if popularity_source != "all".

  cooccurrence
      Recency-weighted co-occurrence + popularity blend.
      Works well on small datasets (Amazon Beauty). OOM-prone on large ones.

  trained_sasrec
      Loads a trained SASRec checkpoint.
      Falls back to popularity mode on cold sequences or missing checkpoint.

All modes expose the same BackboneInterface (get_top_k / get_all_scores).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from src.backbone.interface import BackboneInterface

logger = logging.getLogger(__name__)

# Co-occurrence index parameters (only used in cooccurrence mode)
_COOCCUR_TOPN   = 200   # keep only top-200 co-occurring neighbors per item
_COOCCUR_WINDOW = 20    # sliding window size (avoids O(L^2) on long sequences)

# lazy torch import — only imported when backbone_mode=trained_sasrec
_torch = None
_SASRec = None


def _lazy_imports() -> None:
    global _torch, _SASRec
    if _torch is None:
        import torch as _t
        _torch = _t
    if _SASRec is None:
        from src.backbone.model import SASRec as _S
        _SASRec = _S


class SASRecWrapper(BackboneInterface):
    """
    Unified backbone wrapper.

    Parameters
    ----------
    df_interactions : pd.DataFrame
        columns: user_id, item_id, timestamp, (optionally source_service)
    backbone_cfg : dict
        Contents of a backbone config yaml.
    """

    def __init__(
        self,
        df_interactions: pd.DataFrame,
        backbone_cfg: dict,
    ) -> None:
        self._cfg = backbone_cfg
        self._all_item_ids: list[str] = sorted(df_interactions["item_id"].unique().tolist())
        self._item_id_set: set[str] = set(self._all_item_ids)

        mode = backbone_cfg.get("backbone_mode", None)

        # Back-compat: old configs used use_trained_model=true with no backbone_mode
        if mode is None:
            if backbone_cfg.get("use_trained_model", False):
                mode = "trained_sasrec"
            else:
                mode = "cooccurrence"

        self._mode = mode
        logger.info("Backbone mode: %s", mode)

        # ── Popularity index (always built — cheap, used by popularity mode
        #   and as fallback for trained_sasrec) ──────────────────────────
        pop_source = backbone_cfg.get("popularity_source", "all")
        if pop_source != "all" and "source_service" in df_interactions.columns:
            df_pop = df_interactions[df_interactions["source_service"] == pop_source]
            logger.info("Popularity source filtered to '%s': %d rows", pop_source, len(df_pop))
        else:
            df_pop = df_interactions

        counts = df_pop["item_id"].value_counts()
        max_count = counts.max()
        self._popularity: dict[str, float] = {
            iid: math.log(1 + cnt) / math.log(1 + max_count)
            for iid, cnt in counts.items()
        }
        # Pre-sorted popularity list for fast top-K
        self._popularity_sorted: list[tuple[str, float]] = sorted(
            self._popularity.items(), key=lambda x: x[1], reverse=True
        )

        # ── Co-occurrence index (only built in cooccurrence mode) ────────
        self._cooccur: dict[str, Counter] = {}
        if mode == "cooccurrence":
            logger.info("Building co-occurrence index (sparse, top-%d neighbors)...", _COOCCUR_TOPN)
            _raw_cooccur: dict[str, Counter] = {}
            for _, group in df_interactions.groupby("user_id"):
                items = group.sort_values("timestamp")["item_id"].tolist()
                for i, item in enumerate(items):
                    window = items[max(0, i - _COOCCUR_WINDOW): i] + items[i + 1: i + _COOCCUR_WINDOW + 1]
                    if item not in _raw_cooccur:
                        _raw_cooccur[item] = Counter()
                    _raw_cooccur[item].update(window)
            self._cooccur = {
                item: Counter(dict(cnt.most_common(_COOCCUR_TOPN)))
                for item, cnt in _raw_cooccur.items()
            }
            del _raw_cooccur
            logger.info("Co-occurrence index built: %d items", len(self._cooccur))

        # ── Trained SASRec (only loaded in trained_sasrec mode) ──────────
        self._model = None
        self._item2idx: Optional[dict[str, int]] = None
        self._idx2item: Optional[dict[int, str]] = None
        self._device = None
        self._all_item_tensor = None
        self._max_seq_len: int = backbone_cfg.get("max_seq_len", 50)
        self._min_seq_for_trained: int = (
            backbone_cfg.get("fallback", {}).get("min_sequence_length", 2)
        )

        if mode == "trained_sasrec":
            ckpt_path_raw = backbone_cfg.get("checkpoint_path", None)
            if ckpt_path_raw and ckpt_path_raw != "null":
                self._load_checkpoint(Path(ckpt_path_raw))
            else:
                logger.warning("trained_sasrec mode but no checkpoint_path — falling back to popularity")
                self._mode = "popularity"

    # ------------------------------------------------------------------
    # Checkpoint loading (trained_sasrec mode)
    # ------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s — falling back to popularity", ckpt_path)
            self._mode = "popularity"
            return

        _lazy_imports()
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

        logger.info("Loading SASRec checkpoint: %s  (device=%s)", ckpt_path, device)
        ckpt = _torch.load(ckpt_path, map_location=device, weights_only=False)

        cfg = ckpt["config"]
        model = _SASRec(
            user_num=cfg.get("user_num", 0),
            item_num=cfg["item_num"],
            hidden_units=cfg["hidden_units"],
            maxlen=cfg["maxlen"],
            num_blocks=cfg["num_blocks"],
            num_heads=cfg["num_heads"],
            dropout_rate=cfg.get("dropout_rate", 0.0),
            norm_first=False,
            device=device,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        self._model = model
        self._item2idx = ckpt["item2idx"]
        self._idx2item = ckpt["idx2item"]
        self._device = device
        self._max_seq_len = cfg["maxlen"]

        item_num = cfg["item_num"]
        self._all_item_tensor = _torch.arange(
            1, item_num + 1, dtype=_torch.long, device=device
        )
        logger.info(
            "SASRec loaded: vocab=%d  dim=%d  blocks=%d  best_NDCG=%.4f",
            cfg["item_num"], cfg["hidden_units"], cfg["num_blocks"],
            ckpt.get("best_ndcg", 0.0),
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_popularity(
        self,
        seen: set[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Return popularity-sorted (item_id, score) excluding seen items."""
        result = [(iid, s) for iid, s in self._popularity_sorted if iid not in seen]
        return result[:top_k] if top_k else result

    def _score_cooccurrence(
        self,
        item_sequence: list[str],
        candidate_item_id: str,
    ) -> float:
        if not item_sequence:
            return self._popularity.get(candidate_item_id, 0.0)

        n = len(item_sequence)
        co_score = 0.0
        weight_sum = 0.0
        for i, hist_item in enumerate(item_sequence):
            w = (i + 1) / n
            weight_sum += w
            co_count = self._cooccur.get(hist_item, Counter()).get(candidate_item_id, 0)
            hist_total = sum(self._cooccur.get(hist_item, Counter()).values()) or 1
            co_score += w * (co_count / hist_total)

        co_score /= weight_sum or 1.0
        pop_score = self._popularity.get(candidate_item_id, 0.0)
        return co_score + 0.3 * pop_score

    def _score_all_trained(self, item_sequence: list[str]) -> Optional[dict[str, float]]:
        if self._model is None:
            return None
        if len(item_sequence) < self._min_seq_for_trained:
            return None

        try:
            _lazy_imports()
            idxs = [self._item2idx[item] for item in item_sequence if item in self._item2idx]
            if not idxs:
                return None

            seq = _torch.zeros(self._max_seq_len, dtype=_torch.long)
            idx = self._max_seq_len - 1
            for item_idx in reversed(idxs):
                seq[idx] = item_idx
                idx -= 1
                if idx == -1:
                    break

            seq_tensor = seq.unsqueeze(0).to(self._device)
            uid_tensor = _torch.zeros(1, dtype=_torch.long, device=self._device)

            with _torch.no_grad():
                scores_tensor = self._model.predict(
                    uid_tensor, seq_tensor, self._all_item_tensor
                ).squeeze(0)

            scores_np = scores_tensor.cpu().float().numpy()
            return {
                self._idx2item[i + 1]: float(s)
                for i, s in enumerate(scores_np)
                if (i + 1) in self._idx2item
            }

        except Exception as e:
            logger.debug("Trained model inference failed (%s), using popularity fallback", e)
            return None

    # ------------------------------------------------------------------
    # BackboneInterface
    # ------------------------------------------------------------------

    def get_all_scores(
        self,
        user_id: str,
        item_sequence: list[str],
    ) -> dict[str, float]:
        seen = set(item_sequence)

        if self._mode == "popularity":
            return {iid: s for iid, s in self._popularity_sorted if iid not in seen}

        if self._mode == "trained_sasrec":
            trained = self._score_all_trained(item_sequence)
            if trained is not None:
                return {iid: s for iid, s in trained.items() if iid not in seen}
            # fallback to popularity
            return {iid: s for iid, s in self._popularity_sorted if iid not in seen}

        # cooccurrence mode
        return {
            iid: self._score_cooccurrence(item_sequence, iid)
            for iid in self._all_item_ids
            if iid not in seen
        }

    def get_top_k(
        self,
        user_id: str,
        item_sequence: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        seen = set(item_sequence)

        if self._mode == "popularity":
            return self._score_popularity(seen, top_k)

        if self._mode == "trained_sasrec":
            trained = self._score_all_trained(item_sequence)
            if trained is not None:
                sorted_items = sorted(
                    ((iid, s) for iid, s in trained.items() if iid not in seen),
                    key=lambda x: x[1], reverse=True,
                )
                return sorted_items[:top_k]
            return self._score_popularity(seen, top_k)

        # cooccurrence mode
        scores = {
            iid: self._score_cooccurrence(item_sequence, iid)
            for iid in self._all_item_ids
            if iid not in seen
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
