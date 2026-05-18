"""M8: stochastic output head producing (μ, log_σ) per species row.

Companion to PINNHead. PINNHead provides the *mean* prediction with
mass-conservation guarantees; this head adds a learned per-species
*log_σ* so the model can express "this transition is irreducibly
noisy" instead of being penalised for not predicting variance.

Together they enable an NLL training objective:
  L = 0.5 * ((target - μ) / σ)² + log σ

which lets val_count drop below the deterministic 0.0716 noise floor
that every M7.* variant has hit. Sigma calibration is the actual
unlock — the model learns to put high σ on stochastic rows (e.g.
metabolite count fluctuations) and low σ on rows that are
deterministically explained by S·v.

Inference:
  - Point prediction:  use μ directly (same as deterministic M7)
  - Stochastic sample: sample from Normal(μ, σ)
  - Uncertainty:        σ itself (per-row epistemic+aleatoric estimate)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StochasticHead(nn.Module):
    """Predict per-species log_σ from the encoder's hidden state.

    The mean μ is supplied externally (from PINNHead + residual head).
    This module only produces log_σ. Total output (μ, σ) is reassembled
    by the calling code.

    Why a separate head: keeps the mass-conservation invariant of
    PINNHead intact. The PINN-mapped rows still get Δx = S·v as their
    mean; only the σ branch is added.

    Args
    ----
    hidden : int           encoder hidden dim
    log_sigma_init: float  initial bias for log_σ; -2.0 → σ ≈ 0.135
                            which matches signed-log1p noise scale of
                            ~0.07 at the observed val_count floor.
    log_sigma_clip: tuple  (min, max) clip on log_σ output. Default
                            (-5, 3) → σ in [0.0067, 20]. Wide enough
                            for any plausible row.
    """

    def __init__(
        self,
        hidden: int,
        log_sigma_init: float = -2.0,
        log_sigma_clip: tuple = (-5.0, 3.0),
    ):
        super().__init__()
        self.log_sigma_head = nn.Linear(hidden, 1)
        self.log_sigma_min, self.log_sigma_max = log_sigma_clip
        with torch.no_grad():
            self.log_sigma_head.weight.normal_(std=0.01)
            self.log_sigma_head.bias.fill_(log_sigma_init)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, hidden)  ->  log_sigma: (B, N)."""
        log_sigma = self.log_sigma_head(h).squeeze(-1)
        return torch.clamp(
            log_sigma, min=self.log_sigma_min, max=self.log_sigma_max,
        )


def nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    sigma_floor: float = 1e-3,
    mean_dim: tuple | None = None,
) -> torch.Tensor:
    """Gaussian negative log-likelihood per-element, reduced.

    L = 0.5 * ((target - μ) / σ)² + log σ + const

    The constant 0.5·log(2π) is dropped (doesn't affect optimisation).
    `sigma_floor` prevents division by zero in cases where the network
    drives σ to 0 to game the loss.
    """
    sigma = log_sigma.exp().clamp(min=sigma_floor)
    z = (target - mu) / sigma
    per_element = 0.5 * z.pow(2) + log_sigma
    if mean_dim is None:
        return per_element.mean()
    return per_element.mean(dim=mean_dim)


def nll_loss_masked(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma_floor: float = 1e-3,
) -> torch.Tensor:
    """NLL loss over a boolean row-mask only.

    Used for per-prefix breakdown (count / flux / cum). Mask is
    1-D (S,); broadcasts over the batch dim of mu/log_sigma/target.
    """
    sigma = log_sigma.exp().clamp(min=sigma_floor)
    z = (target - mu) / sigma
    per_element = 0.5 * z.pow(2) + log_sigma   # (B, S)
    return per_element[:, mask].mean()
