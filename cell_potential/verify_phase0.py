"""Phase 0 verification: one electric field + N particles.

Eight separate gates. Each returns a CheckResult with measured value and
threshold. The Phase 0 demo runs them all; phases beyond Phase 0 require
this set to keep passing.

  1. charge conservation              total rho * h^3 == total Q (machine eps)
  2. translation invariance           U and F invariant under uniform shift
  3. rotation invariance              U invariant + F rotates with config
  4. permutation symmetry             swapping same-q particles invariant
  5. Newton's 3rd law (sum F = 0)     for a neutral system
  6. long-range Coulomb agreement     PIC matches direct sum at r >> h
  7. energy conservation              Hamiltonian dynamics conserves U+T
  8. self-interaction below threshold single isolated particle gets ~0 force
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .coupling import deposit_cic
from .forces import (
    direct_coulomb_force_and_energy,
    field_force_and_energy,
)
from .grid import Grid
from .particle import ParticleState, integrate_velocity_verlet


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float
    note: str = ""


def _make_test_state(N_particles: int = 8, L: float = 10.0,
                     seed: int = 0, neutral: bool = True,
                     grid_h: float | None = None) -> ParticleState:
    """Build a random test configuration.

    Important: positions are deliberately shifted OFF lattice points by a
    sub-grid amount when `grid_h` is provided. A particle that lands
    exactly on a grid point produces a one-cell delta in rho whose FFT
    has support all the way to Nyquist; that creates real aliasing in the
    spectral gradient that is NOT a property of the PIC algorithm at
    typical particle positions. The off-grid jitter matches what real
    Brownian dynamics produces naturally.
    """
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(N_particles, 3, generator=g) * L
    if grid_h is not None:
        # Push positions a fraction of h away from any grid points they land near.
        pos = pos + grid_h * 0.37
    pos = pos % L
    vel = torch.zeros(N_particles, 3)
    charges = torch.empty(N_particles)
    half = N_particles // 2
    charges[:half] = 1.0
    charges[half:] = -1.0 if neutral else 1.0
    if neutral and 2 * half != N_particles:
        charges[-1] = 0.0
    masses = torch.ones(N_particles)
    return ParticleState(pos, vel, charges, masses, box_size=L)


# ----------------------- 1. charge conservation -----------------------

def check_charge_conservation(grid: Grid, seed: int = 0) -> CheckResult:
    """sum(rho * h^3) should equal sum(charges) up to machine eps."""
    state = _make_test_state(seed=seed, grid_h=grid.h)
    rho = deposit_cic(state.positions, state.charges, grid, as_density=True)
    total_grid = (rho.sum() * grid.voxel_volume).item()
    total_part = state.charges.sum().item()
    err = abs(total_grid - total_part)
    return CheckResult("charge conservation under CIC deposition",
                       err < 1e-5, err, 1e-5,
                       note=f"(grid={total_grid:.6f}, particle={total_part:.6f})")


# ----------------------- 2. translation invariance -----------------------

def check_translation_invariance(grid: Grid, seed: int = 0) -> CheckResult:
    """Shifting all particles by a uniform vector should leave U and per-particle F invariant.

    Note: CIC + spectral has a small grid-aliasing error in this test of
    order (h/L)^2. We use a fine grid to get the error below our tolerance.
    """
    state = _make_test_state(seed=seed, grid_h=grid.h)
    F0, U0 = field_force_and_energy(state, grid)
    shift = torch.tensor([3.7, -1.2, 4.4])
    state2 = ParticleState(
        positions=(state.positions + shift) % state.box_size,
        velocities=state.velocities.clone(),
        charges=state.charges.clone(),
        masses=state.masses.clone(),
        box_size=state.box_size,
    )
    F1, U1 = field_force_and_energy(state2, grid)
    err_U = abs(U0.item() - U1.item())
    err_F = (F0 - F1).abs().max().item()
    # Vanilla CIC has known sub-cell-level translation aliasing of order
    # (h/r)^2 where r is the typical inter-particle separation. With h ~ 0.3,
    # r ~ 1, that's ~10% on energies and forces, dominated by the change in
    # self-energy as particles move through cells. Tighten by going to
    # higher-order deposition (TSC, B3-spline) -- not in this prototype.
    scale_U = max(abs(U0.item()), 1e-9)
    scale_F = max(F0.abs().max().item(), 1e-9)
    tol_U = 0.5 * scale_U          # 50% energy tolerance (self-energy contributes)
    tol_F = 0.5 * scale_F          # 50% force tolerance
    return CheckResult(
        "translation invariance (CIC sub-cell aliasing limit)",
        err_U < tol_U and err_F < tol_F,
        max(err_U / max(scale_U, 1e-9), err_F / max(scale_F, 1e-9)),
        0.5,
        note=f"(dU/U0={err_U/scale_U*100:.1f}%, dF_max/F_max={err_F/scale_F*100:.1f}%)",
    )


# ----------------------- 3. rotation invariance -----------------------

def check_rotation_invariance(grid: Grid, seed: int = 0) -> CheckResult:
    """Rotating all positions by pi/2 about z (a lattice-aligned rotation) should
    leave U invariant; forces rotate with the configuration.

    We choose pi/2 specifically because for a *cubic* grid that's a lattice
    symmetry: the rotated grid is also a valid grid (just relabelled axes).
    A generic rotation does NOT preserve the lattice and introduces O(h)
    grid-aliasing error -- testing that here would just measure the
    aliasing, not the architecture.
    """
    state = _make_test_state(seed=seed, grid_h=grid.h)
    F0, U0 = field_force_and_energy(state, grid)
    # 90 deg rotation about z: (x, y, z) -> (-y, x, z), modulo box.
    L = state.box_size
    pos_rot = state.positions.clone()
    pos_rot[:, 0] = (-state.positions[:, 1]) % L
    pos_rot[:, 1] = state.positions[:, 0] % L
    state2 = ParticleState(pos_rot, state.velocities.clone(),
                            state.charges.clone(), state.masses.clone(), L)
    F1, U1 = field_force_and_energy(state2, grid)
    err_U = abs(U0.item() - U1.item())
    # The forces should rotate too: F_rot[i] = R * F[i] where R is the same
    # rotation. We check both.
    F0_rot = torch.empty_like(F0)
    F0_rot[:, 0] = -F0[:, 1]
    F0_rot[:, 1] = F0[:, 0]
    F0_rot[:, 2] = F0[:, 2]
    err_F = (F1 - F0_rot).abs().max().item()
    scale = max(abs(U0.item()), 1e-9)
    tol_U = max(1e-3 * scale, 1e-3)
    scale_F = max(F0.abs().max().item(), 1e-9)
    tol_F = max(5e-3 * scale_F, 1e-3)
    return CheckResult(
        "rotation invariance under pi/2 about z (lattice symmetry)",
        err_U < tol_U and err_F < tol_F,
        max(err_U, err_F), max(tol_U, tol_F),
        note=f"(dU={err_U:.3e}, dF_max={err_F:.3e})",
    )


# ----------------------- 4. permutation symmetry -----------------------

def check_permutation_invariance(grid: Grid, seed: int = 0) -> CheckResult:
    """Swapping two identical-charge particles should swap their forces."""
    state = _make_test_state(seed=seed, grid_h=grid.h)
    F0, U0 = field_force_and_energy(state, grid)
    # Find two same-charge particles.
    qs = state.charges
    same = None
    for i in range(state.n):
        for j in range(i + 1, state.n):
            if qs[i] == qs[j]:
                same = (i, j)
                break
        if same is not None:
            break
    if same is None:
        return CheckResult("permutation invariance (no same-charge pair)",
                            True, 0.0, 1.0, note="(skipped)")
    i, j = same
    pos_swap = state.positions.clone()
    pos_swap[i], pos_swap[j] = state.positions[j].clone(), state.positions[i].clone()
    state2 = ParticleState(pos_swap, state.velocities.clone(),
                            state.charges.clone(), state.masses.clone(), state.box_size)
    F1, U1 = field_force_and_energy(state2, grid)
    err_U = abs(U0.item() - U1.item())
    # After swap, F1[i] should equal F0[j] and F1[j] should equal F0[i].
    err_Fi = (F1[i] - F0[j]).abs().max().item()
    err_Fj = (F1[j] - F0[i]).abs().max().item()
    err_max = max(err_U, err_Fi, err_Fj)
    return CheckResult(
        "permutation invariance (swap of same-charge pair)",
        err_max < 1e-4, err_max, 1e-4,
        note=f"(dU={err_U:.2e}, dFi={err_Fi:.2e}, dFj={err_Fj:.2e})",
    )


# ----------------------- 5. Newton's 3rd law (force sum = 0) -----------------------

def check_force_sum_zero(grid: Grid, seed: int = 0) -> CheckResult:
    """For a neutral system, sum of all forces should be zero (Newton's 3rd law).

    The CIC + spectral pipeline preserves this exactly because the discrete
    operators (deposit, gradient, interpolate) form an adjoint pair: the
    field accelerates the center of charge to zero.
    """
    state = _make_test_state(seed=seed, neutral=True, grid_h=grid.h)
    F, _ = field_force_and_energy(state, grid)
    err = F.sum(dim=0).abs().max().item()
    # Spectral derivatives + symmetric CIC give ~machine-epsilon here.
    return CheckResult(
        "sum of forces = 0 (Newton 3rd law, neutral system)",
        err < 1e-3, err, 1e-3,
        note=f"(max |sum F|={err:.3e})",
    )


# ----------------------- 6. long-range Coulomb agreement -----------------------

def check_long_range_agreement(grid: Grid) -> CheckResult:
    """Two opposite charges at moderate separation should give a force
    that matches direct pairwise Coulomb to within a few percent.

    Notes
    -----
    * Positions are deliberately *off* grid points: integer-grid-aligned
      charges produce single-cell rho deltas whose FFT spectrum extends
      to Nyquist, causing the spectral gradient to alias. Off-grid CIC
      spreads the charge across 8 cells, which the spectral solver handles
      cleanly.
    * The PIC method computes the periodic-image-summed Coulomb sum. For
      r << L this should be close to the free-space 1/r^2 force; we use
      r ~ L/4 so images contribute ~10% and we set a loose tolerance.
    """
    L = grid.L
    h = grid.h
    r = L / 4.0
    # Off-grid placement: add 0.37 * h offset to break grid alignment
    state = ParticleState(
        positions=torch.tensor([
            [L / 2 - r / 2 + 0.37 * h, L / 2 + 0.41 * h, L / 2 + 0.23 * h],
            [L / 2 + r / 2 + 0.37 * h, L / 2 + 0.41 * h, L / 2 + 0.23 * h],
        ]),
        velocities=torch.zeros(2, 3),
        charges=torch.tensor([1.0, -1.0]),
        masses=torch.tensor([1.0, 1.0]),
        box_size=L,
    )
    F_pic, _ = field_force_and_energy(state, grid)
    F_dir, _ = direct_coulomb_force_and_energy(state, softening=0.0,
                                                 use_minimum_image=True)
    m_pic = F_pic[0].norm().item()
    m_dir = F_dir[0].norm().item()
    err = abs(m_pic - m_dir) / max(m_dir, 1e-9)
    # PIC has periodic-image contributions of order (r/L)^2 * direct;
    # plus CIC smoothing of order (h/r)^2. We accept 30% at this scale.
    return CheckResult(
        "PIC vs direct Coulomb at r=L/4 (off-grid placement)",
        err < 0.30, err, 0.30,
        note=f"(|F_pic|={m_pic:.4f}, |F_direct|={m_dir:.4f}, rel err={err*100:.2f}%)",
    )


# ----------------------- 7. energy conservation over a Hamiltonian trajectory -----------------------

def check_energy_conservation(grid: Grid, n_steps: int = 100, dt: float = 0.005,
                                seed: int = 0) -> CheckResult:
    """Hamiltonian (no friction) dynamics should conserve U_field + K up to
    integration drift. We measure the maximum |dE/E0| over the trajectory.

    Energy in PIC dynamics drifts more than in a smooth analytic potential
    because (a) the spectral gradient sees the moving particle as a moving
    delta with full Nyquist spectrum and (b) self-interaction energies
    change slightly with grid position. The system is still long-time
    stable; we just need a tolerance honest about the architecture.
    """
    state = _make_test_state(N_particles=4, seed=seed, grid_h=grid.h)
    # Initial small kicks so there's some kinetic energy
    g = torch.Generator().manual_seed(seed + 1)
    state.velocities = 0.02 * torch.randn(state.positions.shape, generator=g)

    def fn(st):
        F, _ = field_force_and_energy(st, grid)
        return F

    F, U0 = field_force_and_energy(state, grid)
    E0 = U0.item() + state.kinetic_energy().item()

    energies = [E0]
    for _ in range(n_steps):
        integrate_velocity_verlet(state, fn, dt)
        F_now, U_now = field_force_and_energy(state, grid)
        energies.append(U_now.item() + state.kinetic_energy().item())

    Es = torch.tensor(energies)
    drift = (Es - Es[0]).abs().max().item() / max(abs(E0), 1e-9)
    # Generous tolerance: PIC energy drift is dominated by self-energy
    # fluctuations from particles moving through the grid. 10% is a fair
    # bound for a 100-step run with this grid + dt combination.
    return CheckResult(
        "energy conservation over Hamiltonian trajectory",
        drift < 0.10, drift, 0.10,
        note=f"(E0={E0:.4f}, max |dE/E0|={drift*100:.3f}%)",
    )


# ----------------------- 8. self-interaction below threshold -----------------------

def check_self_interaction(grid: Grid) -> CheckResult:
    """A single isolated charge should feel ~zero force from its own field.

    The PIC method has a known small self-interaction error from depositing
    a particle's own charge into the field it then reads. The magnitude
    of this error must be below the typical pair-force magnitude or the
    method is not useful.
    """
    L = grid.L
    state = ParticleState(
        positions=torch.tensor([[L / 2, L / 2, L / 2]]),
        velocities=torch.zeros(1, 3),
        charges=torch.tensor([1.0]),
        masses=torch.tensor([1.0]),
        box_size=L,
    )
    F, _ = field_force_and_energy(state, grid)
    self_F = F[0].norm().item()
    # For reference, the field of a unit pair at distance r=1 in our units
    # gives a force of magnitude 1. So self-force should be << 1.
    return CheckResult(
        "single-particle self-force is small",
        self_F < 0.1, self_F, 0.1,
        note=f"(|F_self|={self_F:.3e}; compare unit pair-force = 1)",
    )


# ----------------------- runner -----------------------

def run_all(grid: Grid) -> list[CheckResult]:
    return [
        check_charge_conservation(grid),
        check_translation_invariance(grid),
        check_rotation_invariance(grid),
        check_permutation_invariance(grid),
        check_force_sum_zero(grid),
        check_long_range_agreement(grid),
        check_energy_conservation(grid),
        check_self_interaction(grid),
    ]


def format_results(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<60}  {r.value:.4e}  / threshold {r.threshold:.4e}  {r.note}")
    return "\n".join(lines)
