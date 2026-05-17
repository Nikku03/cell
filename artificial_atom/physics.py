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


# ----------------- Bond-aware reference (Morse + LJ) -----------------

# Morse parameters for "bonded" pairs - one set per (Z_i, Z_j) symmetric pair.
# D_e is bond dissociation depth, a is width, r_e is equilibrium distance.
MORSE = {
    (6, 6): {"D_e": 3.0, "a": 1.8, "r_e": 1.54},   # C-C
    (6, 8): {"D_e": 3.5, "a": 1.9, "r_e": 1.43},   # C-O
    (8, 8): {"D_e": 2.5, "a": 2.0, "r_e": 1.48},   # O-O (chemically rare; placeholder)
}


def morse_params(Z_i: int, Z_j: int) -> dict:
    key = (min(Z_i, Z_j), max(Z_i, Z_j))
    return MORSE[key]


@dataclass
class BondAwareReference:
    """Reference potential that distinguishes bonded and non-bonded pairs.

    For every atom pair (i, j):
      - if (i, j) appears in the bond list -> Morse potential
      - else                               -> Lennard-Jones (with cutoff)

    Bonds are passed as a (B, 2) long tensor of atom-index pairs. The order
    within a pair does not matter; we canonicalise (i < j) internally.
    """

    r_cutoff: float = 6.0
    smooth: bool = True

    def _cutoff_factor(self, r: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(r / self.r_cutoff, max=1.0)
        return 0.5 * (torch.cos(torch.pi * x) + 1.0) * (r < self.r_cutoff).float()

    def _bond_mask(self, N: int, bonds: torch.Tensor, device) -> torch.Tensor:
        """Returns an (N, N) bool mask, True where (i, j) is a bond."""
        mask = torch.zeros((N, N), dtype=torch.bool, device=device)
        if bonds is None or bonds.numel() == 0:
            return mask
        i = bonds[:, 0]
        j = bonds[:, 1]
        mask[i, j] = True
        mask[j, i] = True
        return mask

    def energy(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        bonds: torch.Tensor | None,
    ) -> torch.Tensor:
        N = positions.shape[0]
        if N < 2:
            return positions.sum() * 0.0

        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        r = torch.linalg.norm(diff, dim=-1)
        r_safe = r + torch.eye(N, device=r.device) * 1.0
        diag = torch.eye(N, device=r.device).bool()

        # ---- Lennard-Jones term for non-bonded pairs ----
        Z_i = Z.unsqueeze(1).expand(N, N)
        Z_j = Z.unsqueeze(0).expand(N, N)
        sigma = torch.zeros_like(r)
        eps = torch.zeros_like(r)
        for za, sa in SIGMA.items():
            for zb, sb in SIGMA.items():
                mask_pair = (Z_i == za) & (Z_j == zb)
                if mask_pair.any():
                    s, e = pair_params(za, zb)
                    sigma = torch.where(mask_pair, torch.full_like(sigma, s), sigma)
                    eps = torch.where(mask_pair, torch.full_like(eps, e), eps)
        sr6 = (sigma / r_safe) ** 6
        sr12 = sr6 ** 2
        V_lj = 4.0 * eps * (sr12 - sr6) * self._cutoff_factor(r_safe)

        # ---- Morse term for bonded pairs ----
        bond_mask = self._bond_mask(N, bonds, r.device)
        V_morse = torch.zeros_like(r)
        if bond_mask.any():
            # Build per-pair Morse parameters.
            D_e = torch.zeros_like(r)
            a = torch.zeros_like(r)
            r_e = torch.zeros_like(r)
            for (za, zb), pp in MORSE.items():
                pair_mask = ((Z_i == za) & (Z_j == zb)) | ((Z_i == zb) & (Z_j == za))
                pair_mask = pair_mask & bond_mask
                if pair_mask.any():
                    D_e = torch.where(pair_mask, torch.full_like(D_e, pp["D_e"]), D_e)
                    a = torch.where(pair_mask, torch.full_like(a, pp["a"]), a)
                    r_e = torch.where(pair_mask, torch.full_like(r_e, pp["r_e"]), r_e)
            x = 1.0 - torch.exp(-a * (r_safe - r_e))
            # Morse "well": V = D_e * (1 - exp(-a*(r-r_e)))^2 - D_e
            # The -D_e shift makes the bottom of the well V = -D_e at r=r_e.
            V_morse_full = D_e * x * x - D_e
            V_morse = torch.where(bond_mask, V_morse_full, V_morse)

        V_pair = torch.where(bond_mask, V_morse, V_lj)
        V_pair = V_pair.masked_fill(diag, 0.0)
        return 0.5 * V_pair.sum()

    def forces(
        self,
        positions: torch.Tensor,
        Z: torch.Tensor,
        bonds: torch.Tensor | None,
    ) -> torch.Tensor:
        positions = positions.detach().clone().requires_grad_(True)
        E = self.energy(positions, Z, bonds)
        (g,) = torch.autograd.grad(E, positions, create_graph=False)
        return -g
