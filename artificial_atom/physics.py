"""Synthetic reference potential.

A two-species Lennard-Jones potential acts as the ground truth: we know its
energies and forces in closed form, so we can verify the learned model end
to end.

  V(r) = 4 * eps_ij * ( (sigma_ij/r)^12 - (sigma_ij/r)^6 )

Pair parameters use Lorentz-Berthelot mixing for the cross term (arithmetic
mean for sigma, geometric mean for eps) - the textbook default.

We map atomic numbers (Z) to two "species": Z=6 (carbon) and Z=8 (oxygen)
purely so the element embedding in the network has two distinct labels to
learn. The physics is not real C-O LJ; the model is just learning whatever
analytic potential we hand it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Pair parameters (sigma in Angstroms, eps in eV-equivalents - units are
# arbitrary; we only need them to be self-consistent across energy and
# force computations).
SIGMA = {6: 2.5, 8: 3.0}      # diagonal terms
EPS = {6: 1.0, 8: 1.5}


def pair_params(Z_i: int, Z_j: int) -> tuple[float, float]:
    """Lorentz-Berthelot mixing for unlike-pair parameters."""
    sigma_i = SIGMA[Z_i]
    sigma_j = SIGMA[Z_j]
    eps_i = EPS[Z_i]
    eps_j = EPS[Z_j]
    sigma = 0.5 * (sigma_i + sigma_j)
    eps = (eps_i * eps_j) ** 0.5
    return sigma, eps


@dataclass
class LennardJonesReference:
    """Pairwise LJ energy + autograd-compatible forces.

    The energy is computed in torch so the same forward pass can supply
    analytic forces (via the closed form below) AND autograd-derived
    forces (which we check match, as a sanity test of the reference
    itself).
    """

    r_cutoff: float = 6.0   # Angstroms; matches the network cutoff
    smooth: bool = True

    def _cutoff_factor(self, r: torch.Tensor) -> torch.Tensor:
        if not self.smooth:
            return (r < self.r_cutoff).float()
        # Cosine cutoff identical to the one used in the network so the
        # learned potential can fit the reference exactly at long range.
        x = torch.clamp(r / self.r_cutoff, max=1.0)
        c = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        return c * (r < self.r_cutoff).float()

    def energy(self, positions: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Total energy of the configuration.

        positions: (N, 3) float
        Z: (N,) long
        returns: scalar tensor
        """
        N = positions.shape[0]
        if N < 2:
            return positions.sum() * 0.0  # zero, but keeps autograd graph

        diff = positions.unsqueeze(0) - positions.unsqueeze(1)   # (N, N, 3)
        r = torch.linalg.norm(diff, dim=-1)                       # (N, N)
        # Avoid self-pair singularity: mask later, but stuff a 1.0 in the
        # diagonal so the pow() doesn't generate inf.
        r_safe = r + torch.eye(N, device=r.device) * 1.0

        # Build per-pair sigma and eps from Z.
        Z_i = Z.unsqueeze(1).expand(N, N)
        Z_j = Z.unsqueeze(0).expand(N, N)
        sigma = torch.zeros_like(r)
        eps = torch.zeros_like(r)
        for za, sa in SIGMA.items():
            for zb, sb in SIGMA.items():
                mask = (Z_i == za) & (Z_j == zb)
                if mask.any():
                    s, e = pair_params(za, zb)
                    sigma = torch.where(mask, torch.full_like(sigma, s), sigma)
                    eps = torch.where(mask, torch.full_like(eps, e), eps)

        sr6 = (sigma / r_safe) ** 6
        sr12 = sr6 ** 2
        V_pair = 4.0 * eps * (sr12 - sr6)
        V_pair = V_pair * self._cutoff_factor(r_safe)

        # Zero out the diagonal and double-counts.
        diag = torch.eye(N, device=r.device).bool()
        V_pair = V_pair.masked_fill(diag, 0.0)
        E = 0.5 * V_pair.sum()    # 1/2 for double-counted pairs
        return E

    def forces(self, positions: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Analytic forces. F_i = -dE/dx_i."""
        positions = positions.detach().clone().requires_grad_(True)
        E = self.energy(positions, Z)
        (grad,) = torch.autograd.grad(E, positions, create_graph=False)
        return -grad
