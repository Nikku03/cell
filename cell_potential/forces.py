"""Force / energy assembly.

Two ways to compute electrostatic forces in this package:

  field_force_and_energy(state, grid)         particle-in-cell (PIC):
                                              charges -> CIC -> Poisson -> grad -> interp
  direct_coulomb_force_and_energy(state)      O(N^2) pairwise Coulomb,
                                              ignoring periodic images (so it
                                              agrees with PIC only at r << L).

The verification suite uses the direct sum as the ground truth that the
PIC method must agree with at long range, modulo finite-grid and
self-interaction corrections.

There's also a short-range Lennard-Jones helper used in Phase 2 (P3M-style
combination where short-range is direct + long-range is field).
"""

from __future__ import annotations

import math

import torch

from .coupling import deposit_cic, interpolate_scalar, interpolate_vector
from .grid import Grid, grid_gradient_spectral, solve_poisson
from .particle import ParticleState


# ----------------------- Particle-in-cell electrostatic forces -----------------------

def field_force_and_energy(
    state: ParticleState,
    grid: Grid,
    *,
    return_field: bool = False,
):
    """PIC electrostatic force on each particle + total field energy.

    Implementation:
      1. Deposit charges via CIC to get rho on the grid.
      2. Solve Poisson nabla^2 phi = -4 pi rho (Gaussian units).
      3. Compute E = -grad phi on the grid via spectral derivative.
      4. Interpolate E back to particle positions: F_i = q_i * E(x_i).
      5. Field energy: U = (1/(8 pi)) * sum(|E|^2) * h^3.

    The PIC force has a small self-interaction error at very short
    distances (the particle's own charge contributes to the field it
    reads). The error vanishes as the grid is refined. See verify.py
    for explicit accuracy bounds.
    """
    rho = deposit_cic(state.positions, state.charges, grid, as_density=True)
    phi = solve_poisson(rho, grid, units="gaussian")
    E = -grid_gradient_spectral(phi, grid)        # (3, N, N, N)
    E_at_particle = interpolate_vector(E, state.positions, grid)
    force = state.charges.unsqueeze(-1) * E_at_particle

    # Field energy U_field = integral of |E|^2 / (8 pi) dV
    # We use Gaussian units (4 pi eps = 1), so U = (1/(8 pi)) * sum|E|^2 * dV.
    e2 = (E ** 2).sum(dim=0)
    U_field = (e2.sum() * grid.voxel_volume) / (8.0 * math.pi)

    if return_field:
        return force, U_field, {"rho": rho, "phi": phi, "E": E}
    return force, U_field


# ----------------------- Direct (O(N^2)) pairwise Coulomb baseline -----------------------

def direct_coulomb_force_and_energy(
    state: ParticleState,
    *,
    softening: float = 1e-3,
    use_minimum_image: bool = False,
):
    """O(N^2) direct pairwise Coulomb. F_i = sum_{j != i} q_i q_j (r_i - r_j) / |r_ij|^3.

    `softening` adds a tiny constant to |r_ij|^2 to avoid singularities;
    set to 0 if you have a guaranteed minimum separation.

    `use_minimum_image`: if True, applies the minimum-image convention so the
    pair separation respects the periodic box (clipping to a single image).
    PIC sums over *all* periodic images, so the two will not agree unless
    the box is large compared to the inter-particle separation.
    """
    P = state.n
    pos = state.positions
    q = state.charges
    L = state.box_size

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (P, P, 3): r_j - r_i (from i to j)
    if use_minimum_image:
        diff = diff - L * torch.round(diff / L)
    r2 = (diff * diff).sum(dim=-1) + softening
    r2.fill_diagonal_(1.0)                              # avoid self-term blowing up
    r3 = r2 ** 1.5

    # qiqj = q.unsqueeze(0) * q.unsqueeze(1)
    inv_r3 = 1.0 / r3
    inv_r3.fill_diagonal_(0.0)                          # exclude self
    # F on i from all j: F_i = q_i * sum_j q_j (r_i - r_j) / r_ij^3
    #                       = -q_i * sum_j q_j * diff / r_ij^3
    F = -(q.unsqueeze(0) * inv_r3).unsqueeze(-1) * diff
    F = q.unsqueeze(-1) * F.sum(dim=1)

    # Energy: U = (1/2) sum_{i != j} q_i q_j / r_ij  (Gaussian units)
    inv_r = 1.0 / torch.sqrt(r2)
    inv_r.fill_diagonal_(0.0)
    U = 0.5 * (q.unsqueeze(0) * q.unsqueeze(1) * inv_r).sum()

    return F, U


# ----------------------- Short-range Lennard-Jones (for Phase 2) -----------------------

def lj_force_and_energy(
    state: ParticleState,
    *,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    cutoff: float = 2.5,
    use_minimum_image: bool = True,
):
    """Pairwise Lennard-Jones with a hard cutoff.

    V(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6],   r < cutoff
         = 0,                                      r >= cutoff

    Returns (force (P, 3), total energy scalar).
    """
    P = state.n
    pos = state.positions
    L = state.box_size

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (P, P, 3) j - i
    if use_minimum_image:
        diff = diff - L * torch.round(diff / L)
    r2 = (diff * diff).sum(dim=-1)
    r2.fill_diagonal_(1.0)
    mask = (r2 < cutoff * cutoff) & (r2 > 0)
    r2 = torch.where(mask, r2, torch.full_like(r2, cutoff * cutoff * 2))   # outside contributes 0
    inv_r2 = 1.0 / r2
    s2 = sigma * sigma
    sr2 = s2 * inv_r2
    sr6 = sr2 ** 3
    sr12 = sr6 * sr6
    V_pair = 4.0 * epsilon * (sr12 - sr6)
    # dV/dr = -24 eps/r [(2*(sigma/r)^12) - ((sigma/r)^6)] -> magnitude
    # Force on i from j is - dV/dr * r_hat where r_hat points i->j.
    f_factor = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6)     # (P, P), magnitude per unit vector
    f_factor = torch.where(mask, f_factor, torch.zeros_like(f_factor))
    # F_i = sum_j (f_factor[i, j]) * (r_i - r_j)   (note sign: dV/dr in r direction j->i)
    # Our `diff` is r_j - r_i. So F_i should use (-diff). Equivalently, F = -sum f_factor * diff.
    F = -(f_factor.unsqueeze(-1) * diff).sum(dim=1)

    V_pair = torch.where(mask, V_pair, torch.zeros_like(V_pair))
    V_pair.fill_diagonal_(0.0)
    U = 0.5 * V_pair.sum()
    return F, U


# ----------------------- P3M-style: field for long-range, direct for short -----------------------

def hybrid_coulomb_force_and_energy(
    state: ParticleState,
    grid: Grid,
    *,
    short_range_cutoff: float = 0.0,
    softening: float = 1e-3,
):
    """Combine: field gives the long-range Coulomb force; direct pairs
    inside `short_range_cutoff` are added (and the corresponding field
    contribution is *not* subtracted here -- this is a simple hybrid,
    not a proper Ewald split). For Phase 2 verification we use this with
    `short_range_cutoff=0` (pure field) and compare to direct.

    Returns (force, U_field). The direct-pair energy is not returned
    separately to keep the interface simple; the verification suite
    computes its own pair energy when needed.
    """
    F_field, U_field = field_force_and_energy(state, grid)
    if short_range_cutoff <= 0.0:
        return F_field, U_field

    # Add direct-pair contribution for the close neighbours only.
    P = state.n
    pos = state.positions
    q = state.charges
    L = state.box_size
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    diff = diff - L * torch.round(diff / L)
    r2 = (diff * diff).sum(dim=-1) + softening
    r2.fill_diagonal_(float("inf"))
    mask = r2 < short_range_cutoff * short_range_cutoff
    inv_r3 = torch.where(mask, 1.0 / r2 ** 1.5, torch.zeros_like(r2))
    F_short = -(q.unsqueeze(0) * inv_r3).unsqueeze(-1) * diff
    F_short = q.unsqueeze(-1) * F_short.sum(dim=1)

    return F_field + F_short, U_field
