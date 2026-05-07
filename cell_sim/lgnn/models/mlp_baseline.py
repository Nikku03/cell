"""Plain residual MLP delta-predictor — the M0 baseline.

Predicts Δsigned-log1p(x) given x. Two residual blocks of (LN -> Linear ->
SiLU -> Dropout -> Linear), no skip from input to output.

Known weaknesses (measured on Luthey-Schulten data, 2 epochs):
  * median R² over top-100 high-variance species: -0.502
  * catastrophic rollout drift: |signed-log1p| ≈ 1300 by t=7200s
Both are expected for this architecture and are what motivates moving
to a graph + heteroscedastic + multi-step model. This file is kept for
ablations and as the reference floor in EXPERIMENTS.md.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class CountDynamicsMLP(nn.Module):
    """Input: (B, S) signed-log1p of counts.  Output: (B, S) Δsigned-log1p."""

    def __init__(self, n_species: int, hidden: int = 1024,
                 n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_species = n_species
        self.input_proj = nn.Linear(n_species, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            ))
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, n_species)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(self.out_norm(h))


@torch.no_grad()
def predict_step(model: nn.Module, x_count: np.ndarray,
                 device: Optional[torch.device] = None) -> np.ndarray:
    """One forward step: counts (or signed fluxes) -> next-second values."""
    if device is None:
        device = next(model.parameters()).device
    x = x_count.astype(np.float32)
    x_log = np.sign(x) * np.log1p(np.abs(x))
    t = torch.from_numpy(x_log).to(device).unsqueeze(0)
    dx = model(t).squeeze(0).cpu().numpy()
    next_log = x_log + dx
    next_val = np.sign(next_log) * np.expm1(np.abs(next_log))
    return np.clip(np.round(next_val), 0, None).astype(np.int64)
