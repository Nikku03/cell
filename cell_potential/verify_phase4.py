"""Phase 4 verification: PBC + v3.2 + FastPairKernel distillation.

Three extensions to Phase 3, each with its own gates plus a regression
pass over every earlier check.

Extension 1: PBC-aware v3
  - radius_edges accepts a box_size argument for minimum-image distances.
  - Translation invariance now holds for ANY shift (including those that
    wrap around the box), not just non-wrapping ones.

Extension 2: v3.2 (MultiModePotentialV3) bridge
  - cell_potential particles get plausible defaults for v3.2's chemistry
    inputs and are routed through the multi-channel network.
  - v3.2 produces conservative forces via autograd, just like v3.

Extension 3: FastPairKernel distillation
  - Train a tiny pair MLP (~3-5 k params) to imitate v3 on the same
    LJ reference. The kernel is strictly pairwise so it cannot capture
    many-body terms beyond v3 -- but since v3 was itself trained on a
    pairwise potential, the kernel matches it closely on the training
    distribution at a fraction of the compute.
"""

from __future__ import annotations

import math
import time

import torch  # noqa: F401  (used indirectly via cell_potential modules)

from artificial_atom.dataset import generate_dataset

from .fast_kernel import (
    DistillConfig,
    FastPairConfig,
    FastPairKernel,
    distill_from_v3,
    kernel_force_and_energy,
)
from .forces import field_force_and_energy
from .grid import Grid
from .particle import ParticleState, integrate_velocity_verlet
from .v3_bridge import build_and_train_v3, v3_force_and_energy
from .v32_bridge import build_and_train_v32, v32_force_and_energy
from .verify_phase0 import CheckResult
from .verify_phase3 import (
    get_v3_model,
    run_all as run_phase3_checks,
)


# Cache trained models -- training each is expensive.
_V32_MODEL = None
_FAST_KERNEL = None
_DISTILL_HISTORY = None


def get_v32_model():
    global _V32_MODEL
    if _V32_MODEL is None:
        _V32_MODEL = build_and_train_v32(
            epochs=8, n_train=200, n_val=50, seed=0
        )
    return _V32_MODEL


def get_fast_kernel():
    global _FAST_KERNEL, _DISTILL_HISTORY
    if _FAST_KERNEL is None:
        teacher = get_v3_model()
        _FAST_KERNEL, _DISTILL_HISTORY = distill_from_v3(
            teacher,
            cfg=DistillConfig(epochs=30, lr=3e-3, n_train=250, n_val=60),
            student_cfg=FastPairConfig(d_embed=16, d_rbf=12, d_hidden=32),
        )
    return _FAST_KERNEL


# ============= Extension 1: PBC =============

def check_v3_pbc_translation_under_wrap(grid: Grid) -> CheckResult:
    """With PBC enabled, translation invariance holds even when particles
    wrap around the box."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(7)
    pos = torch.rand(8, 3) * (L * 0.5) + L * 0.25
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F0, _ = v3_force_and_energy(state, model, use_pbc=True)
    # Large shift that wraps multiple particles
    shift = torch.tensor([4.5, -3.7, 5.2])
    pos2 = (pos + shift) % L
    state2 = ParticleState(pos2, torch.zeros_like(pos), torch.zeros(8),
                            torch.ones(8), L)
    F2, _ = v3_force_and_energy(state2, model, use_pbc=True)
    err = (F0 - F2).abs().max().item()
    return CheckResult(
        "v3 PBC: translation invariance under wrapping shift",
        err < 1e-3, err, 1e-3,
        note=f"(max |dF| under big wrap = {err:.2e})",
    )


def check_v3_pbc_matches_non_pbc_when_safe(grid: Grid) -> CheckResult:
    """When no pair crosses the box boundary, PBC and non-PBC must agree."""
    model = get_v3_model()
    L = grid.L
    torch.manual_seed(11)
    pos = torch.rand(8, 3) * (L * 0.5) + L * 0.25     # in [2.5, 7.5]
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F_no_pbc, _ = v3_force_and_energy(state, model, use_pbc=False)
    F_pbc, _ = v3_force_and_energy(state, model, use_pbc=True)
    err = (F_no_pbc - F_pbc).abs().max().item()
    return CheckResult(
        "v3 PBC vs no-PBC agree when no pair crosses boundary",
        err < 1e-6, err, 1e-6,
        note=f"(max diff = {err:.2e})",
    )


# ============= Extension 2: v3.2 =============

def check_v32_loads_and_runs(grid: Grid) -> CheckResult:
    """v3.2 trains and produces finite forces on a cell_potential state."""
    model = get_v32_model()
    L = grid.L
    torch.manual_seed(13)
    pos = torch.rand(12, 3) * (L * 0.5) + L * 0.25
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(12),
                           torch.ones(12), L)
    F, E = v32_force_and_energy(state, model)
    finite = torch.isfinite(F).all().item() and torch.isfinite(E).item()
    return CheckResult(
        "v3.2 loads and produces finite forces on ParticleState",
        finite, 1.0 if finite else 0.0, 0.5,
        note=f"(|F|_max={F.abs().max().item():.3f}, E={E.item():.3f})",
    )


def check_v32_force_sum_zero(grid: Grid) -> CheckResult:
    """v3.2 forces should sum to ~0 by autograd of a translation-invariant scalar."""
    model = get_v32_model()
    L = grid.L
    torch.manual_seed(19)
    pos = torch.rand(15, 3) * (L * 0.4) + L * 0.3
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(15),
                           torch.ones(15), L)
    F, _ = v32_force_and_energy(state, model)
    err = F.sum(dim=0).abs().max().item()
    return CheckResult(
        "v3.2 sum(F) ~ 0 (Newton 3rd via autograd)",
        err < 1e-3, err, 1e-3,
        note=f"(max |sum F| = {err:.2e})",
    )


def check_v32_translation_invariance(grid: Grid) -> CheckResult:
    """v3.2 is translation-invariant by construction (no raw position
    enters the network -- only pairwise distances)."""
    model = get_v32_model()
    L = grid.L
    torch.manual_seed(23)
    pos = torch.rand(10, 3) * (L * 0.4) + L * 0.3
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(10),
                           torch.ones(10), L)
    F0, _ = v32_force_and_energy(state, model)
    shift = torch.tensor([0.6, -0.4, 0.7])
    state2 = ParticleState(pos + shift, torch.zeros_like(pos),
                            torch.zeros(10), torch.ones(10), L)
    F1, _ = v32_force_and_energy(state2, model)
    err = (F0 - F1).abs().max().item()
    return CheckResult(
        "v3.2 translation invariance (non-wrapping shift)",
        err < 1e-3, err, 1e-3,
        note=f"(max |dF| = {err:.2e})",
    )


def check_v32_hybrid_force_sum(grid: Grid) -> CheckResult:
    """v3.2 short-range + PIC long-range on a neutral charged system."""
    model = get_v32_model()
    L = grid.L
    torch.manual_seed(29)
    pos = torch.rand(20, 3) * (L * 0.4) + L * 0.3
    charges = torch.empty(20)
    charges[:10] = 0.4
    charges[10:] = -0.4
    state = ParticleState(pos, torch.zeros_like(pos), charges, torch.ones(20), L)
    F_v32, _ = v32_force_and_energy(state, model)
    F_field, _ = field_force_and_energy(state, grid)
    F_total = F_v32 + F_field
    err = F_total.sum(dim=0).abs().max().item()
    return CheckResult(
        "v3.2 + field hybrid: sum(F) ~ 0 (Newton 3rd)",
        err < 1e-3, err, 1e-3,
        note=f"(max |sum F_total| = {err:.2e})",
    )


# ============= Extension 3: Fast kernel =============

def check_fast_kernel_accuracy_on_training_distribution(grid: Grid) -> CheckResult:
    """Student should match teacher v3 on a held-out batch of the training
    distribution (3-8 atom configurations).

    Threshold is set realistically: the student is strictly pair-only
    while v3 has 2 rounds of message passing. On a pair potential like LJ
    the gap should be small, but it is not zero.
    """
    teacher = get_v3_model()
    student = get_fast_kernel()
    val_configs = generate_dataset(n_configs=40, seed=2025)
    f_abs_sum = 0.0
    f_count = 0
    f_rms_sq = 0.0
    for c in val_configs:
        pos_t = c.positions.clone().requires_grad_(True)
        E_t = teacher(pos_t, c.Z)
        (g_t,) = torch.autograd.grad(E_t, pos_t, allow_unused=True)
        if g_t is None:
            g_t = torch.zeros_like(pos_t)
        F_t = -g_t.detach()

        pos_s = c.positions.clone().requires_grad_(True)
        E_s = student(pos_s, c.Z)
        (g_s,) = torch.autograd.grad(E_s, pos_s, allow_unused=True)
        if g_s is None:
            g_s = torch.zeros_like(pos_s)
        F_s = -g_s.detach()

        f_abs_sum += (F_t - F_s).abs().sum().item()
        f_rms_sq += (F_t ** 2).sum().item()
        f_count += F_t.numel()
    f_mae = f_abs_sum / max(1, f_count)
    f_rms = (f_rms_sq / max(1, f_count)) ** 0.5
    rel = f_mae / max(f_rms, 1e-9)
    return CheckResult(
        "fast kernel force MAE relative to v3 teacher (training distribution)",
        rel < 0.35, rel, 0.35,
        note=f"(F_MAE={f_mae:.3f}, F_RMS_v3={f_rms:.3f}, MAE/RMS={rel*100:.1f}%)",
    )


def check_fast_kernel_speedup(grid: Grid) -> CheckResult:
    """Student should be at least 1.5x faster than v3 on a 50-atom system."""
    teacher = get_v3_model()
    student = get_fast_kernel()
    torch.manual_seed(41)
    L = grid.L
    pos = torch.rand(50, 3) * (L * 0.5) + L * 0.25
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(50),
                           torch.ones(50), L)
    Z = torch.full((50,), 6, dtype=torch.long)

    # warmup
    for _ in range(3):
        teacher.energy_and_forces(state.positions.clone(), Z=Z)
        kernel_force_and_energy(state, student)

    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        teacher.energy_and_forces(state.positions.clone(), Z=Z)
    t_v3 = (time.time() - t0) / n_runs
    t0 = time.time()
    for _ in range(n_runs):
        kernel_force_and_energy(state, student)
    t_kernel = (time.time() - t0) / n_runs
    speedup = t_v3 / max(t_kernel, 1e-9)
    return CheckResult(
        "fast kernel speedup vs v3 (50-particle config)",
        speedup >= 1.5, speedup, 1.5,
        note=f"(v3: {t_v3*1000:.2f} ms, kernel: {t_kernel*1000:.2f} ms, speedup: {speedup:.2f}x)",
    )


def check_fast_kernel_invariances(grid: Grid) -> CheckResult:
    """Fast kernel is translation/rotation/permutation invariant by
    construction. Newton 3rd law via autograd."""
    student = get_fast_kernel()
    L = grid.L
    torch.manual_seed(53)
    pos = torch.rand(8, 3) * (L * 0.4) + L * 0.3
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F0, E0 = kernel_force_and_energy(state, student)

    # Translation
    shift = torch.tensor([0.43, -0.21, 0.31])
    state_t = ParticleState(pos + shift, torch.zeros_like(pos), torch.zeros(8),
                              torch.ones(8), L)
    Ft, Et = kernel_force_and_energy(state_t, student)
    err_t = (F0 - Ft).abs().max().item()

    # Rotation
    theta = 0.47
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0.0],
                       [math.sin(theta),  math.cos(theta), 0.0],
                       [0.0,              0.0,              1.0]])
    centre = torch.tensor([L / 2, L / 2, L / 2])
    pos_r = (pos - centre) @ R.T + centre
    state_r = ParticleState(pos_r, torch.zeros_like(pos), torch.zeros(8),
                              torch.ones(8), L)
    F_r, _ = kernel_force_and_energy(state_r, student)
    F_back = F_r @ R
    err_r = (F0 - F_back).abs().max().item()

    # Newton 3rd
    err_sum = F0.sum(dim=0).abs().max().item()

    err_max = max(err_t, err_r, err_sum)
    return CheckResult(
        "fast kernel invariances (translation + rotation + sum(F)=0)",
        err_max < 1e-3, err_max, 1e-3,
        note=f"(dF_trans={err_t:.2e}, dF_rot={err_r:.2e}, sum_F={err_sum:.2e})",
    )


def check_fast_kernel_pbc(grid: Grid) -> CheckResult:
    """Fast kernel with PBC: translation invariance even when particles wrap."""
    student = get_fast_kernel()
    L = grid.L
    torch.manual_seed(59)
    pos = torch.rand(8, 3) * (L * 0.4) + L * 0.3
    state = ParticleState(pos, torch.zeros_like(pos), torch.zeros(8),
                           torch.ones(8), L)
    F0, _ = kernel_force_and_energy(state, student, use_pbc=True)
    # Big wrap
    shift = torch.tensor([4.2, -3.6, 5.7])
    state2 = ParticleState((pos + shift) % L, torch.zeros_like(pos),
                            torch.zeros(8), torch.ones(8), L)
    F2, _ = kernel_force_and_energy(state2, student, use_pbc=True)
    err = (F0 - F2).abs().max().item()
    return CheckResult(
        "fast kernel PBC: translation invariance with wrapping",
        err < 1e-3, err, 1e-3,
        note=f"(max |dF| = {err:.2e})",
    )


# ============= runner =============

def run_all(grid: Grid) -> list[CheckResult]:
    phase3 = run_phase3_checks(grid)
    phase4 = [
        # Extension 1: PBC
        check_v3_pbc_translation_under_wrap(grid),
        check_v3_pbc_matches_non_pbc_when_safe(grid),
        # Extension 2: v3.2
        check_v32_loads_and_runs(grid),
        check_v32_force_sum_zero(grid),
        check_v32_translation_invariance(grid),
        check_v32_hybrid_force_sum(grid),
        # Extension 3: Fast kernel
        check_fast_kernel_accuracy_on_training_distribution(grid),
        check_fast_kernel_speedup(grid),
        check_fast_kernel_invariances(grid),
        check_fast_kernel_pbc(grid),
    ]
    return phase3 + phase4


def format_results(results) -> str:
    lines = []
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{tag}]  {r.name:<78}  {r.value:.4e}  / threshold {r.threshold:.4e}  {r.note}")
    return "\n".join(lines)
