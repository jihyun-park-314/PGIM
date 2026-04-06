"""
Learnable tiny gating module for candidate-aware modulation.

The gate is intentionally small and interpretable:
  - input: structured scalar features
  - output: mixture weights over persona / goal signals plus a bounded scale
  - final bonus: bounded additive delta on top of existing modulation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TinyGateConfig:
    input_dim: int
    hidden_dim: int = 16
    dropout: float = 0.0
    max_bonus: float = 0.06
    unknown_cap_factor: float = 0.20
    budget_cap_factor: float = 0.15


class TinyGateNet(nn.Module):
    """
    Small feature MLP that predicts:
      - persona / goal mixture weights
      - a bounded activation scale

    bonus = capped_scale * (w_p * persona_score + w_g * goal_score)
    """

    def __init__(self, cfg: TinyGateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.mix_head = nn.Linear(cfg.hidden_dim, 2)
        self.scale_head = nn.Linear(cfg.hidden_dim, 1)
        self._init_conservative()

    def _init_conservative(self) -> None:
        # Start close to a neutral 50/50 blend with small positive scale.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.mix_head.weight)
        nn.init.zeros_(self.mix_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, -1.25)

    def forward(
        self,
        features: torch.Tensor,
        persona_score: torch.Tensor,
        goal_score: torch.Tensor,
        reason_ids: torch.Tensor,
        cap_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(features)
        mix = torch.softmax(self.mix_head(h), dim=-1)
        scale = torch.sigmoid(self.scale_head(h)).squeeze(-1)

        raw_signal = mix[:, 0] * persona_score + mix[:, 1] * goal_score
        cap = float(self.cfg.max_bonus if cap_override is None else cap_override)

        # Conservative hard safety for reasons that historically overfire.
        reason_cap_factor = torch.ones_like(scale)
        reason_cap_factor = torch.where(reason_ids == 3, self.cfg.budget_cap_factor, reason_cap_factor)
        reason_cap_factor = torch.where(reason_ids == 4, self.cfg.unknown_cap_factor, reason_cap_factor)

        bonus = torch.clamp(scale * raw_signal, min=0.0, max=cap)
        bonus = bonus * reason_cap_factor

        return {
            "bonus": bonus,
            "gate_persona": mix[:, 0],
            "gate_goal": mix[:, 1],
            "gate_scale": scale,
            "raw_signal": raw_signal,
        }
