"""Phase 1 verification: add a diffusing concentration field.

Adds to Phase 0:
  - a concentration field c(x, t) governed by partial_t c = D nabla^2 c + S
  - particles that act as sources / sinks (deposit / consume)
  - re-runs every Phase 0 check on the electric field (regression guard)

New gates:
  1. mass conservation under sourceless diffusion
  2. Gaussian-spread test: a localised gaussian spreads exactly as predicted
     by the analytic diffusion equation, sigma^2(t) = sigma_0^2 + 2 D t
  3. particle-source consistency: total integrated c grows at the rate
     specified by the particles' deposition rate
  4. exact-in-spectrum stability: stepping at very large dt does not blow up
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .coupling import deposit_cic
from .grid import Grid, diffusion_step_spectral
from .particle import ParticleState
from .verify_phase0 import CheckResult, run_all as run_phase0_checks


# ----------------------- mass conservation under diffusion -----------------------

def check_mass_conservation_pure_diffusion(grid: Grid, seed: int = 0,
                                             n_steps: int = 50, dt: float = 0.1,
                                             D: float = 0.5) -> CheckResult:
    """A periodic concentration field with no sources should preserve total mass."""
    torch.manual_seed(seed)
    c = torch.rand(grid.shape) + 0.1   # strictly positive
    M0 = (c.sum() * grid.voxel_volume).item()
    for _ in range(n_steps):
        c = diffusion_step_spectral(c, grid, D=D, dt=dt, source=None)
    M_final = (c.sum() * grid.voxel_volume).item()
    err = abs(M_final - M0) / max(abs(M0), 1e-12)
    return CheckResult(
        "mass conservation under sourceless diffusion (50 steps)",
        err < 1e-6, err, 1e-6,
        note=f"(M0={M0:.4f}, M_final={M_final:.4f}, rel err={err:.2e})",
    )


# ----------------------- Gaussian spreading against analytic -----------------------

def check_gaussian_spread(grid: Grid, D: float = 0.5, t_final: float = 1.0,
                           sigma_0: float = 0.5) -> CheckResult:
    """A Gaussian concentration field should spread as sigma^2(t) = sigma_0^2 + 2*D*t.

    We initialise c(x, 0) = exp(-r^2 / (2 sigma_0^2)) centred in the box,
    diffuse for t_final, and fit a Gaussian to extract sigma_pred.
    Compare to analytic sigma_expected.
    """
    N = grid.N
    h = grid.h
    x = (torch.arange(N, dtype=torch.float64) + 0.5) * h
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    centre = grid.L / 2
    r2 = (X - centre) ** 2 + (Y - centre) ** 2 + (Z - centre) ** 2
    c = torch.exp(-r2 / (2 * sigma_0 ** 2))

    # Step once with the full t_final.
    c_final = diffusion_step_spectral(c, grid, D=D, dt=t_final, source=None)

    # Variance of the spread distribution (moments).
    M0 = (c_final.sum() * grid.voxel_volume).item()
    Mx2 = ((c_final * (X - centre) ** 2).sum() * grid.voxel_volume).item()
    sigma_meas_sq = Mx2 / M0
    sigma_expected_sq = sigma_0 ** 2 + 2 * D * t_final
    rel_err = abs(sigma_meas_sq - sigma_expected_sq) / sigma_expected_sq
    return CheckResult(
        "Gaussian spread matches sigma^2 = sigma_0^2 + 2 D t",
        rel_err < 0.05, rel_err, 0.05,
        note=f"(sigma^2_pred={sigma_meas_sq:.4f}, "
              f"sigma^2_exp={sigma_expected_sq:.4f}, rel err={rel_err*100:.1f}%)",
    )


# ----------------------- particle-source consistency -----------------------

def check_particle_source(grid: Grid, n_steps: int = 20, dt: float = 0.05,
                            D: float = 0.5, deposition_rate: float = 2.0,
                            seed: int = 0) -> CheckResult:
    """A single particle deposits substrate at a constant rate.

    Total mass in the field should grow linearly:
      M(t) = M_0 + deposition_rate * t

    Test: drive a particle's deposition for n_steps * dt total time, then
    check the integral matches the expected gain.
    """
    torch.manual_seed(seed)
    pos = torch.tensor([[grid.L / 2 + 0.37 * grid.h,
                          grid.L / 2 + 0.41 * grid.h,
                          grid.L / 2 + 0.23 * grid.h]])
    P = pos.shape[0]
    state = ParticleState(positions=pos, velocities=torch.zeros(P, 3),
                           charges=torch.zeros(P), masses=torch.ones(P),
                           box_size=grid.L)
    c = torch.zeros(grid.shape)

    M0 = (c.sum() * grid.voxel_volume).item()
    expected_growth = deposition_rate * n_steps * dt

    for _ in range(n_steps):
        # Build the source field as the particle's deposition (per-unit-time).
        S = deposit_cic(state.positions,
                         torch.full((P,), deposition_rate, dtype=torch.float32),
                         grid, as_density=True)
        c = diffusion_step_spectral(c, grid, D=D, dt=dt, source=S)

    M_final = (c.sum() * grid.voxel_volume).item()
    actual_growth = M_final - M0
    rel_err = abs(actual_growth - expected_growth) / expected_growth
    return CheckResult(
        "particle-source consistency: dM/dt = deposition_rate",
        rel_err < 0.01, rel_err, 0.01,
        note=f"(expected dM={expected_growth:.4f}, "
              f"measured dM={actual_growth:.4f}, rel err={rel_err*100:.3f}%)",
    )


# ----------------------- stability at large dt -----------------------

def check_diffusion_stability(grid: Grid, n_steps: int = 50,
                                D: float = 0.5, dt_huge: float = 100.0,
                                seed: int = 0) -> CheckResult:
    """The spectral exact-in-mode step is unconditionally stable.

    Take a wild initial condition, step with a ridiculous dt, and confirm
    no values blow up. After diffusion saturates, c should be approximately
    uniform at its mean value.
    """
    torch.manual_seed(seed)
    c0 = (torch.rand(grid.shape) - 0.5) * 100.0
    c = c0.clone()
    mean0 = c.mean().item()
    for _ in range(n_steps):
        c = diffusion_step_spectral(c, grid, D=D, dt=dt_huge, source=None)
    if not torch.isfinite(c).all():
        return CheckResult(
            "diffusion stability at huge dt", False, float("inf"), 1e-3,
            note="(produced non-finite values)",
        )
    # After many steps with huge dt, should be approximately uniform at mean0.
    max_dev = (c - mean0).abs().max().item()
    return CheckResult(
        "diffusion stability + relaxation to mean at huge dt",
        max_dev < 1e-3, max_dev, 1e-3,
        note=f"(initial range {c0.min().item():.1f}..{c0.max().item():.1f}, "
              f"final max deviation from mean = {max_dev:.2e})",
    )


# ----------------------- runner -----------------------

def run_all(grid: Grid) -> list[CheckResult]:
    # Re-run Phase 0 first as a regression guard
    phase0 = run_phase0_checks(grid)
    phase1 = [
        check_mass_conservation_pure_diffusion(grid),
        check_gaussian_spread(grid),
        check_particle_source(grid),
        check_diffusion_stability(grid),
    ]
    return phase0 + phase1


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<60}  {r.value:.4e}  / threshold {r.threshold:.4e}  {r.note}")
    return "\n".join(lines)
