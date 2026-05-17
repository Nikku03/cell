"""Ensemble-based uncertainty for the multi-mode potential.

Trains K models with different random seeds, then at inference returns:
  - mean predicted energy
  - std across the ensemble (the "epistemic" uncertainty signal)
  - mean predicted forces

This is the simplest robust uncertainty source - cheap, well-understood,
no calibration tricks. Variance is high when the K models disagree, which
correlates well with out-of-distribution inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .multimode import MultiModeConfig, MultiModePotential


@dataclass
class EnsembleConfig:
    n_members: int = 3
    base_cfg: MultiModeConfig = None
    seeds: tuple[int, ...] = (0, 1, 2)


class PotentialEnsemble:
    """Independent K-member ensemble; train each member; predict mean+std."""

    def __init__(self, cfg: EnsembleConfig | None = None):
        self.cfg = cfg or EnsembleConfig()
        if self.cfg.base_cfg is None:
            self.cfg.base_cfg = MultiModeConfig()
        self.members: list[MultiModePotential] = []
        for seed in self.cfg.seeds[: self.cfg.n_members]:
            torch.manual_seed(seed)
            self.members.append(MultiModePotential(self.cfg.base_cfg))

    def parameters_per_member(self) -> int:
        return sum(p.numel() for p in self.members[0].parameters())

    @torch.no_grad()
    def _predict_one(self, member: MultiModePotential, sample) -> tuple[float, torch.Tensor]:
        # The energy needs grad-enabled positions even at "no_grad" wrap
        # because we want forces. Reenable grad inside.
        with torch.enable_grad():
            pos = sample["positions"].clone().requires_grad_(True)
            kw = {k: v for k, v in sample.items() if k not in ("positions", "energy", "forces")}
            E, _ = member.forward(pos, **kw)
            (g,) = torch.autograd.grad(E, pos, allow_unused=True)
            if g is None:
                g = torch.zeros_like(pos)
        return E.item(), (-g).detach()

    def predict(self, sample) -> dict:
        Es = []
        Fs = []
        for m in self.members:
            m.eval()
            E, F = self._predict_one(m, sample)
            Es.append(E)
            Fs.append(F)
        Et = torch.tensor(Es)
        Fs_stack = torch.stack(Fs, dim=0)
        return {
            "energy_mean": Et.mean().item(),
            "energy_std": Et.std(unbiased=False).item(),
            "energies": Es,
            "forces_mean": Fs_stack.mean(dim=0),
            "forces_std": Fs_stack.std(dim=0, unbiased=False),
        }
