"""The field substrate: a 3D periodic grid + spectral PDE solvers.

Conventions
-----------

  * The box is a cubic [0, L)^3 with periodic boundary conditions.
  * Grid has shape (N, N, N) with spacing h = L / N.
  * Position indexing is (x, y, z), matching the natural torch layout.
  * Charge density rho is "charge per unit volume". For a point charge q
    deposited via CIC into one cell, that cell's rho is q / h^3.
  * Units: we adopt 4 pi epsilon = 1 so the Poisson equation reads
    nabla^2 phi = -4 pi rho     (Gaussian-style)
    and the resulting potential for a single charge q is phi(r) = q / r.
    Forces and energies derived from phi are then directly comparable to
    direct Coulomb F = q1 q2 / r^2, V = q1 q2 / r.

Solvers
-------

  solve_poisson         spectral Poisson on the grid
  grid_gradient_spectral spectral derivative for -nabla phi -> E-field
  diffusion_step_spectral exact-in-spectrum diffusion step c <- c + dt*(D*nabla^2 c + S)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ----------------------- the grid object -----------------------

@dataclass
class Grid:
    """A periodic cubic grid of N^3 cells in a box of side length L.

    The grid stores no data of its own; it's just a description used by
    the field operators below. Field tensors live wherever the caller
    keeps them.
    """

    N: int
    L: float

    @property
    def h(self) -> float:
        return self.L / self.N

    @property
    def voxel_volume(self) -> float:
        return (self.L / self.N) ** 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.N, self.N, self.N)

    def k_vectors(self, device=None, dtype=torch.float32):
        """Return (kx, ky, kz, k2) wave-number tensors of shape (N, N, N).

        Uses torch.fft.fftfreq so the layout matches torch.fft.fftn output.
        """
        N, L = self.N, self.L
        k1d = 2 * math.pi * torch.fft.fftfreq(N, d=L / N, device=device).to(dtype)
        kx, ky, kz = torch.meshgrid(k1d, k1d, k1d, indexing="ij")
        k2 = kx * kx + ky * ky + kz * kz
        return kx, ky, kz, k2


# ----------------------- Poisson -----------------------

def solve_poisson(rho: torch.Tensor, grid: Grid, *, units: str = "gaussian") -> torch.Tensor:
    """Solve nabla^2 phi = -alpha * rho on a periodic cubic grid.

    units="gaussian"  : alpha = 4 pi    (so point charge -> phi = q/r)
    units="si"        : alpha = 1       (so equation is nabla^2 phi = -rho/eps_0;
                                          caller divides rho by eps_0 if needed)

    Returns phi with the same shape as rho. The k=0 (zero-mean) component
    is set to zero, which is required for solvability under periodic BCs.
    """
    if units == "gaussian":
        alpha = 4.0 * math.pi
    elif units == "si":
        alpha = 1.0
    else:
        raise ValueError(f"unknown units '{units}'")

    _, _, _, k2 = grid.k_vectors(device=rho.device, dtype=rho.dtype)
    # Avoid division by zero at k=0; we set the zero-mean component to 0 afterwards.
    safe_k2 = torch.where(k2 > 0, k2, torch.ones_like(k2))

    rho_hat = torch.fft.fftn(rho)
    phi_hat = alpha * rho_hat / safe_k2
    phi_hat[0, 0, 0] = 0.0       # zero mean (required by periodic Poisson)
    phi = torch.fft.ifftn(phi_hat).real
    return phi


# ----------------------- gradient -----------------------

def grid_gradient_spectral(field: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Compute (d/dx, d/dy, d/dz) of a periodic scalar field via FFT.

    Returns a (3, N, N, N) tensor of gradient components.
    Spectral derivatives are exact up to round-off for periodic data,
    far cleaner than finite differences for our verification suite.
    """
    kx, ky, kz, _ = grid.k_vectors(device=field.device, dtype=field.dtype)
    f_hat = torch.fft.fftn(field)
    dfdx = torch.fft.ifftn(1j * kx * f_hat).real
    dfdy = torch.fft.ifftn(1j * ky * f_hat).real
    dfdz = torch.fft.ifftn(1j * kz * f_hat).real
    return torch.stack([dfdx, dfdy, dfdz], dim=0)


def grid_laplacian_spectral(field: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Compute nabla^2 of a periodic scalar field via FFT."""
    _, _, _, k2 = grid.k_vectors(device=field.device, dtype=field.dtype)
    f_hat = torch.fft.fftn(field)
    lap_hat = -k2 * f_hat
    return torch.fft.ifftn(lap_hat).real


# ----------------------- diffusion -----------------------

def diffusion_step_spectral(
    c: torch.Tensor,
    grid: Grid,
    D: float,
    dt: float,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance c(x, t) by dt under dc/dt = D nabla^2 c + S, exactly in
    spectrum (unconditionally stable for any dt).

    Closed-form per mode k:
        c_hat(k, t+dt) = c_hat(k, t) * exp(-D k^2 dt)
                       + S_hat(k) * (1 - exp(-D k^2 dt)) / (D k^2)
    with the k=0 mode treated separately as a linear-in-time accumulation.
    """
    _, _, _, k2 = grid.k_vectors(device=c.device, dtype=c.dtype)
    decay = torch.exp(-D * k2 * dt)

    c_hat = torch.fft.fftn(c)
    if source is not None:
        s_hat = torch.fft.fftn(source)
    else:
        s_hat = torch.zeros_like(c_hat)

    # General modes
    safe_k2 = torch.where(k2 > 0, k2, torch.ones_like(k2))
    c_hat_new = c_hat * decay + s_hat * (1.0 - decay) / (D * safe_k2)
    # k=0 mode (uniform offset): the analytic limit of (1-exp(-x))/x as x->0 is 1,
    # so the k=0 contribution from sources over dt is just S_hat[0] * dt.
    c_hat_new[0, 0, 0] = c_hat[0, 0, 0] + s_hat[0, 0, 0] * dt

    return torch.fft.ifftn(c_hat_new).real
