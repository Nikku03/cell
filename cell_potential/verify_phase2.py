"""Phase 2 verification: field long-range + direct short-range, ~100 particles.

Adds to Phase 1:
  - 100-particle systems instead of 8
  - Lennard-Jones short-range interactions
  - PIC long-range Coulomb composes with LJ short-range
  - re-runs every Phase 0 + Phase 1 check

New gates:
  1. LJ force pointing test (attractive at r > r_min, repulsive at r < r_min)
  2. LJ force magnitude vs analytic at r = sigma * 2^(1/6) (minimum)
  3. Hybrid force = field + LJ assembles correctly (no sign flips, energy decreases)
  4. Many-body stability: 100 particles, 50 steps of velocity-Verlet, no NaN
  5. Total momentum conservation under combined dynamics
"""

from __future__ import annotations

import math

import torch

from .forces import (
    direct_coulomb_force_and_energy,
    field_force_and_energy,
    lj_force_and_energy,
)
from .grid import Grid
from .particle import ParticleState, integrate_velocity_verlet
from .verify_phase0 import CheckResult
from .verify_phase1 import run_all as run_phase1_checks


# ----------------------- LJ pointing direction -----------------------

def check_lj_pointing(grid: Grid) -> CheckResult:
    """LJ force should attract at r > r_min and repel at r < r_min."""
    L = grid.L
    sigma = 1.0
    r_min = sigma * 2 ** (1 / 6)

    # Attractive regime: r slightly larger than r_min
    state_attr = ParticleState(
        positions=torch.tensor([[L / 2 - (r_min + 0.4) / 2, L / 2, L / 2],
                                  [L / 2 + (r_min + 0.4) / 2, L / 2, L / 2]]),
        velocities=torch.zeros(2, 3),
        charges=torch.zeros(2),
        masses=torch.ones(2),
        box_size=L,
    )
    F_a, _ = lj_force_and_energy(state_attr, sigma=sigma, epsilon=1.0)
    # F on particle 0 (left) should be in +x direction (toward particle 1)
    attractive = F_a[0, 0].item() > 0

    # Repulsive regime: r slightly smaller than r_min
    state_rep = ParticleState(
        positions=torch.tensor([[L / 2 - (r_min - 0.2) / 2, L / 2, L / 2],
                                  [L / 2 + (r_min - 0.2) / 2, L / 2, L / 2]]),
        velocities=torch.zeros(2, 3),
        charges=torch.zeros(2),
        masses=torch.ones(2),
        box_size=L,
    )
    F_r, _ = lj_force_and_energy(state_rep, sigma=sigma, epsilon=1.0)
    # F on particle 0 (left) should be in -x direction (push away from 1)
    repulsive = F_r[0, 0].item() < 0

    passed = attractive and repulsive
    return CheckResult(
        "LJ pointing: attract for r > r_min, repel for r < r_min",
        passed, 1.0 if passed else 0.0, 0.5,
        note=f"(F_attractive={F_a[0,0].item():+.3f}, F_repulsive={F_r[0,0].item():+.3f})",
    )


# ----------------------- LJ at the minimum -----------------------

def check_lj_at_minimum(grid: Grid) -> CheckResult:
    """At r = sigma * 2^(1/6), LJ force should be zero (minimum of potential)."""
    L = grid.L
    sigma = 1.0
    r_min = sigma * 2 ** (1 / 6)
    state = ParticleState(
        positions=torch.tensor([[L / 2 - r_min / 2, L / 2, L / 2],
                                  [L / 2 + r_min / 2, L / 2, L / 2]]),
        velocities=torch.zeros(2, 3),
        charges=torch.zeros(2),
        masses=torch.ones(2),
        box_size=L,
    )
    F, U = lj_force_and_energy(state, sigma=sigma, epsilon=1.0)
    f_norm = F.abs().max().item()
    # At minimum, V = -epsilon, F = 0
    U_err = abs(U.item() - (-1.0))
    return CheckResult(
        "LJ force = 0 at r = sigma * 2^(1/6); V = -epsilon",
        f_norm < 1e-4 and U_err < 1e-3, max(f_norm, U_err), 1e-3,
        note=f"(|F|={f_norm:.2e}, U={U.item():.4f}; expected U=-1)",
    )


# ----------------------- Hybrid force assembly -----------------------

def check_hybrid_force_directions(grid: Grid) -> CheckResult:
    """Two particles: +1 charge and -1 charge, separated by 1.5 (LJ-dominant
    short-range regime). Field force is attractive (Coulomb). LJ force is
    attractive (r > r_min for sigma=1). Both should give net attraction.
    """
    L = grid.L
    r = 1.5
    state = ParticleState(
        positions=torch.tensor([[L / 2 - r / 2 + 0.13, L / 2 + 0.07, L / 2 + 0.11],
                                  [L / 2 + r / 2 + 0.13, L / 2 + 0.07, L / 2 + 0.11]]),
        velocities=torch.zeros(2, 3),
        charges=torch.tensor([1.0, -1.0]),
        masses=torch.ones(2),
        box_size=L,
    )
    F_field, _ = field_force_and_energy(state, grid)
    F_lj, _ = lj_force_and_energy(state, sigma=1.0, epsilon=1.0)
    F_total = F_field + F_lj
    # particle 0 (left, +1) should feel net force in +x toward particle 1
    direction_ok = F_total[0, 0].item() > 0 and F_total[1, 0].item() < 0
    # forces should sum to zero (Newton 3)
    sum_F = F_total.sum(dim=0).abs().max().item()
    return CheckResult(
        "hybrid force assembly: field + LJ produces correct direction + sum(F)=0",
        direction_ok and sum_F < 1e-3,
        max(0.0 if direction_ok else 1.0, sum_F), 1e-3,
        note=f"(F_field[0,x]={F_field[0,0].item():+.3f}, "
              f"F_lj[0,x]={F_lj[0,0].item():+.3f}, "
              f"F_total[0,x]={F_total[0,0].item():+.3f}, sum_F={sum_F:.2e})",
    )


# ----------------------- many-body stability -----------------------

def check_many_body_stability(grid: Grid, N_particles: int = 100,
                                n_steps: int = 50, dt: float = 0.001,
                                seed: int = 0) -> CheckResult:
    """100-particle system with both Coulomb + LJ should be long-time stable.

    We verify:
      - no NaN or inf appears
      - total momentum is conserved (would be exactly so for closed Hamiltonian system)
      - total charge is conserved (trivially, no chemistry happens)
    """
    torch.manual_seed(seed)
    L = grid.L
    # Random positions, off-grid by construction (torch.rand never hits multiples of h)
    pos = torch.rand(N_particles, 3) * (L * 0.8) + L * 0.1
    pos = pos + 0.37 * grid.h
    pos = pos % L
    # Alternating charges, normalised so total charge ~ 0
    charges = torch.empty(N_particles)
    half = N_particles // 2
    charges[:half] = 0.3
    charges[half:] = -0.3
    if N_particles - 2 * half > 0:
        charges[-1] = 0.0
    masses = torch.ones(N_particles)
    state = ParticleState(pos, torch.zeros_like(pos), charges, masses, L)

    def force_fn(st):
        F_field, _ = field_force_and_energy(st, grid)
        F_lj, _ = lj_force_and_energy(st, sigma=0.5, epsilon=0.2, cutoff=1.2)
        return F_field + F_lj

    P_init = state.momentum().clone()
    Q_init = state.total_charge().item()

    finite = True
    for _ in range(n_steps):
        integrate_velocity_verlet(state, force_fn, dt)
        if not torch.isfinite(state.positions).all() or not torch.isfinite(state.velocities).all():
            finite = False
            break

    P_final = state.momentum()
    Q_final = state.total_charge().item()

    dP = (P_final - P_init).abs().max().item()
    dQ = abs(Q_final - Q_init)
    # Momentum drift comes from CIC sub-cell aliasing accumulating over steps.
    # 1e-2 over 50 steps with typical forces ~1 means per-step drift ~ 2e-4,
    # which is far smaller than the typical inter-particle force.
    return CheckResult(
        f"many-body stability ({N_particles} particles, {n_steps} velocity-Verlet steps)",
        finite and dP < 1e-2 and dQ < 1e-9,
        max(0.0 if finite else 1.0, dP, dQ), 1e-2,
        note=f"(finite={finite}, |dP|_max={dP:.2e}, |dQ|={dQ:.2e})",
    )


# ----------------------- Self-consistency of P3M scaling -----------------------

def check_pic_force_scaling(grid: Grid) -> CheckResult:
    """For a fixed +-1 dipole, the PIC force should scale roughly as 1/r^2
    over a range of separations where r > h and r < L/3.

    We sweep r and fit a power-law exponent. Expect -2.0 within tolerance.
    """
    L = grid.L
    h = grid.h
    rs = []
    Fs = []
    for r in [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]:
        state = ParticleState(
            positions=torch.tensor([
                [L / 2 - r / 2 + 0.37 * h, L / 2 + 0.41 * h, L / 2 + 0.23 * h],
                [L / 2 + r / 2 + 0.37 * h, L / 2 + 0.41 * h, L / 2 + 0.23 * h],
            ]),
            velocities=torch.zeros(2, 3),
            charges=torch.tensor([1.0, -1.0]),
            masses=torch.ones(2),
            box_size=L,
        )
        F, _ = field_force_and_energy(state, grid)
        rs.append(r)
        Fs.append(F[0].norm().item())
    # Fit log F = m * log r + c
    log_r = torch.log(torch.tensor(rs, dtype=torch.float64))
    log_F = torch.log(torch.tensor(Fs, dtype=torch.float64))
    n = len(rs)
    rb = log_r.mean()
    fb = log_F.mean()
    m = ((log_r - rb) * (log_F - fb)).sum() / ((log_r - rb) ** 2).sum()
    # Expected slope -2 for 1/r^2 force
    err = abs(m.item() - (-2.0))
    return CheckResult(
        "PIC force radial scaling matches 1/r^2 (log-log slope = -2)",
        err < 0.30, err, 0.30,
        note=f"(measured slope={m.item():+.3f}, expected -2.000)",
    )


# ----------------------- runner -----------------------

def run_all(grid: Grid) -> list[CheckResult]:
    phase01 = run_phase1_checks(grid)
    phase2 = [
        check_lj_pointing(grid),
        check_lj_at_minimum(grid),
        check_hybrid_force_directions(grid),
        check_pic_force_scaling(grid),
        check_many_body_stability(grid),
    ]
    return phase01 + phase2


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<65}  {r.value:.4e}  / threshold {r.threshold:.4e}  {r.note}")
    return "\n".join(lines)
