"""Phase 3 verification: v3-learned kernel as the short-range force.

Adds to Phase 2:
  - the short-range pairwise term is now a v3 ArtificialAtomPotential,
    trained on the same LJ reference it was originally fit to, plugged in
    via cell_potential.v3_bridge
  - all Phase 0 / Phase 1 / Phase 2 gates still pass

New gates:
  1. v3 trains within the expected accuracy ceiling (val_F_MAE < 0.2)
  2. v3 force respects all v3 invariances on cell_potential geometries
     (translation, rotation, permutation, sum(F) = 0)
  3. v3 force on an isolated pair points correctly: attractive at r > r_min,
     repulsive at r < r_min (matches the LJ it was trained on)
  4. hybrid (v3 + field) preserves Newton's 3rd law
  5. 100-particle Hamiltonian dynamics with the hybrid is stable for many
     velocity-Verlet steps (no NaN, momentum drift bounded)
  6. energy from the v3-only path is conservative (autograd guarantee)
"""

from __future__ import annotations

import math

import torch

from artificial_atom.train import evaluate as v3_evaluate
from artificial_atom.dataset import generate_dataset

from .forces import field_force_and_energy
from .grid import Grid
from .particle import ParticleState, integrate_velocity_verlet
from .v3_bridge import (
    build_and_train_v3,
    state_to_v3_inputs,
    v3_force_and_energy,
)
from .verify_phase0 import CheckResult
from .verify_phase2 import run_all as run_phase2_checks


# Shared model across the Phase 3 gates so we only pay the training cost once.
_V3_MODEL = None


def get_v3_model():
    global _V3_MODEL
    if _V3_MODEL is None:
        _V3_MODEL = build_and_train_v3(epochs=15, n_train=300, n_val=80, seed=0)
    return _V3_MODEL


# ----------------------- 1. v3 training quality -----------------------

def check_v3_training_quality(grid: Grid) -> CheckResult:
    """v3 should reach val_F_MAE below 0.2 on the LJ reference it was
    trained on. This is the floor accuracy we need before relying on v3
    for short-range forces in the hybrid."""
    model = get_v3_model()
    val_set = generate_dataset(n_configs=80, seed=99)
    e_mae, f_mae = v3_evaluate(model, val_set)
    return CheckResult(
        "v3 training accuracy (val force MAE < 0.2)",
        f_mae < 0.2, f_mae, 0.2,
        note=f"(E_MAE={e_mae:.3f}, F_MAE={f_mae:.3f} vs LJ reference)",
    )


# ----------------------- 2. v3 invariances on cell_potential geometries -----------------------

def check_v3_translation_invariance(grid: Grid) -> CheckResult:
    """v3 is translation-invariant by construction; verify it on a small,
    non-wrapping shift in cell_potential coordinates.

    Note: v3 currently uses absolute (non-periodic) positions. If a shift
    wraps particles around the box, v3 sees a *fragmented* cluster and
    forces change. Adding PBC-aware distance computation to v3 is a
    Phase 4 task; for now we verify invariance on a shift that keeps
    every particle inside the original box bounds.
    """
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(7)
    # Keep positions well inside the box, with a shift that doesn't wrap.
    pos = torch.rand(8, 3) * (L * 0.5) + L * 0.25     # in [2.5, 7.5]
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F0, _ = v3_force_and_energy(state, model)
    shift = torch.tensor([0.4, -0.3, 0.5])           # all positions stay in [2.0, 8.0]
    state2 = ParticleState(pos + shift, torch.zeros_like(pos),
                            torch.zeros(8), torch.ones(8), L)
    F1, _ = v3_force_and_energy(state2, model)
    err = (F0 - F1).abs().max().item()
    return CheckResult(
        "v3 translation invariance (non-wrapping shift; PBC is Phase 4)",
        err < 1e-3, err, 1e-3,
        note=f"(max |dF| under shift = {err:.2e})",
    )


def check_v3_rotation_invariance(grid: Grid) -> CheckResult:
    """v3 is invariant under arbitrary rotation, and forces rotate with it."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(7)
    pos = torch.rand(8, 3) * (L * 0.6) + L * 0.2
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F0, _ = v3_force_and_energy(state, model)
    # Rotate about center by arbitrary angle
    centre = torch.tensor([L / 2, L / 2, L / 2])
    theta = 0.61
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                       [math.sin(theta),  math.cos(theta), 0.0],
                       [0.0,              0.0,              1.0]])
    pos_rot = ((pos - centre) @ R.T + centre) % L
    state_rot = ParticleState(pos_rot, torch.zeros_like(pos), torch.zeros(8),
                                torch.ones(8), L)
    F_rot, _ = v3_force_and_energy(state_rot, model)
    # If we rotate forces back, they should match originals.
    F_back = F_rot @ R
    err = (F0 - F_back).abs().max().item()
    return CheckResult(
        "v3 rotation invariance (forces rotate with positions)",
        err < 1e-3, err, 1e-3,
        note=f"(max |F0 - R^T*F_rot| = {err:.2e})",
    )


def check_v3_force_sum_zero(grid: Grid) -> CheckResult:
    """v3 forces should sum to ~0 since they come from autograd of a
    scalar invariant under translation."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(5)
    pos = torch.rand(20, 3) * (L * 0.6) + L * 0.2
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(20),
                           torch.ones(20), L)
    F, _ = v3_force_and_energy(state, model)
    err = F.sum(dim=0).abs().max().item()
    return CheckResult(
        "v3 forces sum to ~0 (Newton 3rd law via autograd)",
        err < 1e-3, err, 1e-3,
        note=f"(max |sum F| = {err:.2e})",
    )


# ----------------------- 3. v3 pointing on isolated pair -----------------------

def check_v3_pair_pointing(grid: Grid) -> CheckResult:
    """v3 was trained on LJ -- should attract at r > r_min, repel at r < r_min,
    where r_min = sigma * 2^(1/6). For Z=6 (carbon), sigma_v3 = 2.5."""
    model = get_v3_model()
    L = grid.L
    sigma = 2.5
    r_min = sigma * 2 ** (1 / 6)

    # Attractive regime
    state_a = ParticleState(
        positions=torch.tensor([[L / 2 - (r_min + 0.6) / 2, L / 2, L / 2],
                                  [L / 2 + (r_min + 0.6) / 2, L / 2, L / 2]]),
        velocities=torch.zeros(2, 3),
        charges=torch.zeros(2),
        masses=torch.ones(2),
        box_size=L,
    )
    F_a, _ = v3_force_and_energy(state_a, model)
    # F[0,x] should be > 0 (attract toward +x, i.e., toward partner)
    attractive = F_a[0, 0].item() > 0

    # Repulsive regime
    state_r = ParticleState(
        positions=torch.tensor([[L / 2 - (r_min - 0.5) / 2, L / 2, L / 2],
                                  [L / 2 + (r_min - 0.5) / 2, L / 2, L / 2]]),
        velocities=torch.zeros(2, 3),
        charges=torch.zeros(2),
        masses=torch.ones(2),
        box_size=L,
    )
    F_r, _ = v3_force_and_energy(state_r, model)
    repulsive = F_r[0, 0].item() < 0
    passed = attractive and repulsive
    return CheckResult(
        "v3 pair pointing: attract for r > r_min, repel for r < r_min",
        passed, 1.0 if passed else 0.0, 0.5,
        note=f"(F_attract={F_a[0,0].item():+.3f}, F_repel={F_r[0,0].item():+.3f})",
    )


# ----------------------- 4. Hybrid Newton 3rd law -----------------------

def check_hybrid_newton_3rd_law(grid: Grid, seed: int = 0) -> CheckResult:
    """v3 short-range + PIC long-range on a neutral system. Both
    contributions should sum to zero independently, so the total does too."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(seed)
    pos = torch.rand(20, 3) * (L * 0.6) + L * 0.2
    charges = torch.empty(20)
    charges[:10] = 0.5
    charges[10:] = -0.5
    state = ParticleState(pos, torch.zeros_like(pos), charges,
                           torch.ones(20), L)
    F_v3, _ = v3_force_and_energy(state, model)
    F_field, _ = field_force_and_energy(state, grid)
    F_total = F_v3 + F_field
    err = F_total.sum(dim=0).abs().max().item()
    return CheckResult(
        "hybrid sum(F) = 0 (v3 short-range + PIC long-range, neutral system)",
        err < 1e-3, err, 1e-3,
        note=f"(max |sum F_total| = {err:.2e})",
    )


# ----------------------- 5. Hybrid many-body stability -----------------------

def check_hybrid_stability(grid: Grid, N_particles: int = 100,
                             n_steps: int = 30, dt: float = 0.001,
                             seed: int = 0) -> CheckResult:
    """Run 30 velocity-Verlet steps with the v3 + field hybrid on 100
    particles. Verify no NaN and bounded momentum drift."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(seed)
    pos = torch.rand(N_particles, 3) * (L * 0.6) + L * 0.2
    charges = torch.empty(N_particles)
    half = N_particles // 2
    charges[:half] = 0.2
    charges[half:] = -0.2
    if 2 * half != N_particles:
        charges[-1] = 0.0
    state = ParticleState(pos, torch.zeros_like(pos), charges,
                           torch.ones(N_particles), L)

    def force_fn(st):
        F_v3, _ = v3_force_and_energy(st, model)
        F_field, _ = field_force_and_energy(st, grid)
        return F_v3 + F_field

    P_init = state.momentum().clone()
    finite = True
    for _ in range(n_steps):
        integrate_velocity_verlet(state, force_fn, dt)
        if not torch.isfinite(state.positions).all() or not torch.isfinite(state.velocities).all():
            finite = False
            break
    P_final = state.momentum()
    dP = (P_final - P_init).abs().max().item()
    return CheckResult(
        f"hybrid many-body stability ({N_particles} particles, {n_steps} steps)",
        finite and dP < 1e-2,
        max(0.0 if finite else 1.0, dP), 1e-2,
        note=f"(finite={finite}, |dP|_max={dP:.2e})",
    )


# ----------------------- 6. v3-only energy conservation -----------------------

def check_v3_energy_conservation(grid: Grid, n_steps: int = 50,
                                    dt: float = 0.005,
                                    seed: int = 0) -> CheckResult:
    """v3 forces come from autograd of a scalar energy, so the dynamics
    are conservative. Total energy should be stable over a trajectory.

    No charges -> no field contribution; pure v3 dynamics in a small system.
    """
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(seed)
    pos = torch.rand(6, 3) * (L * 0.4) + L * 0.3
    g = torch.Generator().manual_seed(seed + 1)
    vel = 0.02 * torch.randn(6, 3, generator=g)
    state = ParticleState(pos, vel, torch.zeros(6), torch.ones(6), L)

    def force_fn(st):
        F, _ = v3_force_and_energy(st, model)
        return F

    F0, U0 = v3_force_and_energy(state, model)
    E0 = U0.item() + state.kinetic_energy().item()
    energies = [E0]
    for _ in range(n_steps):
        integrate_velocity_verlet(state, force_fn, dt)
        F_now, U_now = v3_force_and_energy(state, model)
        energies.append(U_now.item() + state.kinetic_energy().item())
    drift = max(abs(e - E0) for e in energies) / max(abs(E0), 1e-9)
    return CheckResult(
        "v3-only energy conservation (autograd-conservative dynamics)",
        drift < 0.05, drift, 0.05,
        note=f"(E0={E0:.4f}, max |dE/E0|={drift*100:.2f}%)",
    )


# ----------------------- runner -----------------------

def run_all(grid: Grid) -> list[CheckResult]:
    # Phase 0+1+2 regression first
    phase012 = run_phase2_checks(grid)
    # Phase 3 specific
    phase3 = [
        check_v3_training_quality(grid),
        check_v3_translation_invariance(grid),
        check_v3_rotation_invariance(grid),
        check_v3_force_sum_zero(grid),
        check_v3_pair_pointing(grid),
        check_hybrid_newton_3rd_law(grid),
        check_v3_energy_conservation(grid),
        check_hybrid_stability(grid),
    ]
    return phase012 + phase3


def format_results(results) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<70}  {r.value:.4e}  / threshold {r.threshold:.4e}  {r.note}")
    return "\n".join(lines)
