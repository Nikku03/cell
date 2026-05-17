"""End-to-end verification of a trained potential.

Checks (each prints PASS / FAIL with a numeric margin):

  1. Test-set energy R^2 >= threshold       (accuracy)
  2. Test-set force MAE <= threshold        (accuracy)
  3. Translation invariance to ~machine eps (symmetry)
  4. Rotation invariance to ~machine eps    (symmetry)
  5. Permutation invariance (same-element swap) (symmetry)
  6. Newton's 3rd law: forces sum to zero   (symmetry consequence)
  7. Relaxation: random C2 dimer settles to ~LJ minimum 2^(1/6)*sigma
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import Configuration
from .network import ArtificialAtomPotential
from .physics import SIGMA, pair_params
from .relax import RelaxConfig, relax


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def _accuracy(
    model: ArtificialAtomPotential, configs: list[Configuration]
) -> tuple[float, float]:
    E_true, E_pred = [], []
    f_abs_sum = 0.0
    f_count = 0
    for c in configs:
        pos = c.positions.clone().requires_grad_(True)
        E = model(pos, c.Z)
        (g,) = torch.autograd.grad(E, pos, allow_unused=True)
        if g is None:
            g = torch.zeros_like(pos)
        F = -g
        E_true.append(c.energy.item())
        E_pred.append(E.item())
        f_abs_sum += (F - c.forces).abs().sum().item()
        f_count += c.forces.numel()
    Et = torch.tensor(E_true)
    Ep = torch.tensor(E_pred)
    ss_res = ((Et - Ep) ** 2).sum().item()
    ss_tot = ((Et - Et.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, f_abs_sum / max(1, f_count)


def _symmetries(model: ArtificialAtomPotential) -> tuple[float, float, float, float, float]:
    """Returns: (translation, rotation, permutation, force-rotation, sum-forces)."""
    torch.manual_seed(42)
    N = 6
    pos = torch.randn(N, 3) * 1.2 + torch.arange(N).float().unsqueeze(-1) * 0.8
    pos[:, 1] *= 0.3
    pos[:, 2] *= 0.3
    Z = torch.tensor([6, 8, 6, 8, 6, 8], dtype=torch.long)

    E = model(pos, Z).item()
    shift = torch.tensor([3.7, -1.2, 0.4])
    dE_t = abs(model(pos + shift, Z).item() - E)

    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                      [math.sin(theta),  math.cos(theta), 0.0],
                      [0.0, 0.0, 1.0]])
    dE_r = abs(model(pos @ R.T, Z).item() - E)

    perm = torch.tensor([2, 1, 0, 3, 4, 5])    # swap atoms 0,2 (both Z=6)
    dE_p = abs(model(pos[perm], Z[perm]).item() - E)

    _, F = model.energy_and_forces(pos, Z)
    _, F_rot = model.energy_and_forces(pos @ R.T, Z)
    dF_rot = (F_rot @ R - F).abs().max().item()
    sum_F = F.sum(dim=0).abs().max().item()

    return dE_t, dE_r, dE_p, dF_rot, sum_F


def _relaxation_lj_minimum(model: ArtificialAtomPotential) -> tuple[float, float]:
    """For a C-C dimer, predicted minimum should be at r=2^(1/6)*sigma_CC.

    Returns (relaxed_r, target_r).
    """
    sigma_cc, _ = pair_params(6, 6)
    target = sigma_cc * 2 ** (1 / 6)
    pos = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])  # r=3.0
    Z = torch.tensor([6, 6], dtype=torch.long)
    pos_relaxed, _, _ = relax(model, pos, Z, RelaxConfig(max_steps=400, step=0.02))
    r = torch.linalg.norm(pos_relaxed[1] - pos_relaxed[0]).item()
    return r, target


def run_all_checks(
    model: ArtificialAtomPotential,
    test_set: list[Configuration],
    r2_threshold: float = 0.9,
    force_mae_threshold: float = 0.20,
    sym_eps: float = 1e-3,
    relax_tol_frac: float = 0.10,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    r2, f_mae = _accuracy(model, test_set)
    results.append(CheckResult("test-set energy R^2", r2 >= r2_threshold, r2, r2_threshold))
    results.append(CheckResult("test-set force MAE", f_mae <= force_mae_threshold, f_mae, force_mae_threshold,
                               note="(lower is better)"))

    dE_t, dE_r, dE_p, dF_rot, sum_F = _symmetries(model)
    results.append(CheckResult("translation invariance", dE_t < sym_eps, dE_t, sym_eps, note="(lower is better)"))
    results.append(CheckResult("rotation invariance",    dE_r < sym_eps, dE_r, sym_eps, note="(lower is better)"))
    results.append(CheckResult("same-element permutation invariance", dE_p < sym_eps, dE_p, sym_eps, note="(lower is better)"))
    results.append(CheckResult("force rotation equivariance", dF_rot < sym_eps, dF_rot, sym_eps, note="(lower is better)"))
    results.append(CheckResult("Newton 3rd law (sum F = 0)", sum_F < sym_eps, sum_F, sym_eps, note="(lower is better)"))

    r_pred, r_target = _relaxation_lj_minimum(model)
    rel_err = abs(r_pred - r_target) / r_target
    results.append(CheckResult("C-C dimer relaxes to LJ minimum",
                                rel_err < relax_tol_frac, rel_err, relax_tol_frac,
                                note=f"(r_pred={r_pred:.3f}, r_target={r_target:.3f}; lower is better)"))
    return results


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<46}  {r.value:.4f}  / threshold {r.threshold:.4f}  {r.note}")
    return "\n".join(lines)
