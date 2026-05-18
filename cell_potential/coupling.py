"""Particle <-> grid coupling: cloud-in-cell deposition and trilinear interpolation.

Both operations are linear interpolation in 3D between the 8 corners of
the grid cell containing the particle. Used in both directions:

  particle -> grid    deposit_cic(positions, weights, grid)
                       deposits per-particle weights (eg charges or sources)
                       onto the grid such that sum(grid * h^3) = sum(weights).

  grid -> particle    interpolate_scalar(field, positions, grid)
                       interpolate a per-cell scalar field to particle positions.
                      interpolate_vector(field, positions, grid)
                       same for a (3, N, N, N) vector field.

Both use periodic boundaries (positions are wrapped into [0, L)).

The deposit and interpolate operations are adjoint up to a volume factor:
  deposit_cic distributes a unit weight over a cell's 8 corners with the
  same trilinear weights that interpolate_scalar uses to read back.
  This adjointness is what makes the coupled particle-field dynamics
  energy-conserving in the leap-frog limit.
"""

from __future__ import annotations

import torch

from .grid import Grid


def _cic_indices_and_weights(
    positions: torch.Tensor, grid: Grid
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute integer corner indices (8, P, 3) and weights (8, P) for CIC.

    Each particle contributes to 8 grid corners; this returns those corner
    indices (wrapped modulo N for periodicity) and the trilinear weights.
    Sum of weights over the 8 corners equals 1 per particle, exactly.
    """
    P = positions.shape[0]
    N = grid.N
    h = grid.h

    # Wrap into the box, then express positions in cell-coordinate units.
    pos_wrapped = positions % grid.L
    x = pos_wrapped / h                          # in [0, N)
    i0 = torch.floor(x).long() % N               # lower corner index, periodic
    i1 = (i0 + 1) % N                            # upper corner index, periodic
    frac = x - torch.floor(x)                    # in [0, 1)
    # frac has shape (P, 3)

    # Build the 8 corners as bit triples (sx, sy, sz) in {0, 1}^3.
    corners_idx = torch.empty(8, P, 3, dtype=torch.long, device=positions.device)
    corners_w = torch.empty(8, P, device=positions.device, dtype=positions.dtype)
    for c, (sx, sy, sz) in enumerate(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    ):
        ix = i1[:, 0] if sx else i0[:, 0]
        iy = i1[:, 1] if sy else i0[:, 1]
        iz = i1[:, 2] if sz else i0[:, 2]
        wx = frac[:, 0] if sx else (1.0 - frac[:, 0])
        wy = frac[:, 1] if sy else (1.0 - frac[:, 1])
        wz = frac[:, 2] if sz else (1.0 - frac[:, 2])
        corners_idx[c, :, 0] = ix
        corners_idx[c, :, 1] = iy
        corners_idx[c, :, 2] = iz
        corners_w[c] = wx * wy * wz
    return corners_idx, corners_w


def deposit_cic(
    positions: torch.Tensor,    # (P, 3)
    weights: torch.Tensor,      # (P,) per-particle weight to deposit
    grid: Grid,
    *,
    as_density: bool = True,
) -> torch.Tensor:
    """Deposit per-particle weights onto a grid via cloud-in-cell.

    If `as_density=True` (default), returns weights per unit volume, so
    sum(grid) * h^3 == sum(weights).  For charge density rho where weights
    are charges, this is what Poisson expects.

    If `as_density=False`, returns weights per cell (not divided by h^3).
    """
    N = grid.N
    h = grid.h
    corners_idx, corners_w = _cic_indices_and_weights(positions, grid)
    out = torch.zeros((N, N, N), dtype=weights.dtype, device=positions.device)
    # Each corner gets accumulated.
    for c in range(8):
        ix, iy, iz = corners_idx[c, :, 0], corners_idx[c, :, 1], corners_idx[c, :, 2]
        out.index_put_(
            (ix, iy, iz),
            weights * corners_w[c],
            accumulate=True,
        )
    if as_density:
        out = out / (h ** 3)
    return out


def interpolate_scalar(
    field: torch.Tensor,        # (N, N, N) scalar field
    positions: torch.Tensor,    # (P, 3) particle positions
    grid: Grid,
) -> torch.Tensor:
    """Trilinear interpolation of a scalar grid field to particle positions.

    Returns shape (P,).
    """
    corners_idx, corners_w = _cic_indices_and_weights(positions, grid)
    out = torch.zeros(positions.shape[0], dtype=field.dtype, device=field.device)
    for c in range(8):
        ix, iy, iz = corners_idx[c, :, 0], corners_idx[c, :, 1], corners_idx[c, :, 2]
        out = out + corners_w[c] * field[ix, iy, iz]
    return out


def interpolate_vector(
    field: torch.Tensor,        # (3, N, N, N) vector field
    positions: torch.Tensor,    # (P, 3)
    grid: Grid,
) -> torch.Tensor:
    """Trilinear interpolation of a (3, N, N, N) vector field to (P, 3)."""
    fx = interpolate_scalar(field[0], positions, grid)
    fy = interpolate_scalar(field[1], positions, grid)
    fz = interpolate_scalar(field[2], positions, grid)
    return torch.stack([fx, fy, fz], dim=-1)
