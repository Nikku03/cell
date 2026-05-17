"""Force-driven pose relaxation.

Given a configuration, take steps along the predicted forces with an
adaptive step size, optionally enforcing a hard minimum pair distance
(the only "physics cage" we include here - prevents the relaxation from
walking into the steep repulsive wall in regions the network has not seen).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .network import ArtificialAtomPotential


@dataclass
class RelaxConfig:
    max_steps: int = 200
    step: float = 0.02            # max position change (A) per atom per step
    force_tol: float = 1e-2       # |F|_inf below which we stop
    min_pair_dist: float = 1.8    # hard floor to keep relaxation safe
    line_search: bool = True


def _min_pair_distance(positions: torch.Tensor) -> float:
    N = positions.shape[0]
    if N < 2:
        return float("inf")
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    r = torch.linalg.norm(diff, dim=-1)
    eye = torch.eye(N, device=r.device).bool()
    r = r.masked_fill(eye, float("inf"))
    return r.min().item()


def relax(
    model: ArtificialAtomPotential,
    positions: torch.Tensor,
    Z: torch.Tensor,
    cfg: RelaxConfig | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Settle the configuration along the predicted forces.

    Returns (final positions, energy history, max-force history).
    """
    cfg = cfg or RelaxConfig()
    pos = positions.detach().clone()
    e_hist: list[float] = []
    f_hist: list[float] = []
    for step_idx in range(cfg.max_steps):
        E, F = model.energy_and_forces(pos, Z)
        e_hist.append(E.item())
        f_max = F.abs().max().item()
        f_hist.append(f_max)
        if f_max < cfg.force_tol:
            break
        # Normalise so largest single-atom step equals cfg.step.
        scale = cfg.step / max(f_max, 1e-9)
        proposed = pos + F * scale
        if cfg.line_search:
            best = proposed
            best_E = float("inf")
            for s in (1.0, 0.5, 0.25):
                trial = pos + F * scale * s
                # Reject moves that violate the steric floor.
                if _min_pair_distance(trial) < cfg.min_pair_dist:
                    continue
                E_trial, _ = model.energy_and_forces(trial, Z)
                if E_trial.item() < best_E:
                    best_E = E_trial.item()
                    best = trial
            pos = best
        else:
            if _min_pair_distance(proposed) >= cfg.min_pair_dist:
                pos = proposed
            else:
                break
    return pos, e_hist, f_hist
