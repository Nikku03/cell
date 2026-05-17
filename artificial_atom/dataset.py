"""Generate training / test data from the analytic LJ reference.

`generate_dataset` produces simple LJ-only configurations.
`generate_rich_dataset` produces v2 configurations with bond labels,
hybridisation, formal charge, and partial charges, with energies and
forces from `BondAwareReference` (Morse on bonded pairs, LJ on the rest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .physics import BondAwareReference, LennardJonesReference, MORSE
from .unit import (
    BOND_AROMATIC, BOND_DOUBLE, BOND_SINGLE, BOND_TRIPLE,
    HYB_AROMATIC, HYB_NONE, HYB_SP, HYB_SP2, HYB_SP3,
)


@dataclass
class Configuration:
    positions: torch.Tensor   # (N, 3) float32
    Z: torch.Tensor           # (N,) long
    energy: torch.Tensor      # scalar float32
    forces: torch.Tensor      # (N, 3) float32

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]


def _sample_positions(
    n_atoms: int, box: float, rng: torch.Generator, min_dist: float
) -> torch.Tensor:
    """Rejection-sample n_atoms uniform-in-box positions with minimum separation."""
    pts: list[torch.Tensor] = []
    max_tries = 5000
    for _ in range(max_tries):
        if len(pts) == n_atoms:
            break
        x = (torch.rand(3, generator=rng) - 0.5) * box
        ok = True
        for q in pts:
            if torch.linalg.norm(x - q).item() < min_dist:
                ok = False
                break
        if ok:
            pts.append(x)
    if len(pts) < n_atoms:
        raise RuntimeError(f"could not place {n_atoms} atoms in box={box} "
                           f"with min_dist={min_dist}")
    return torch.stack(pts, dim=0)


def generate_dataset(
    n_configs: int,
    min_atoms: int = 3,
    max_atoms: int = 8,
    box: float = 8.0,
    min_dist: float = 2.8,        # stay clear of the steep repulsive wall
    elements: tuple[int, ...] = (6, 8),
    seed: int = 0,
    r_cutoff: float = 6.0,
    energy_cap: float = 5.0,      # reject samples above this total energy
    force_cap: float = 15.0,      # reject samples above this max force
) -> list[Configuration]:
    """Random configurations of LJ atoms with their energies and forces."""
    rng = torch.Generator().manual_seed(seed)
    ref = LennardJonesReference(r_cutoff=r_cutoff, smooth=True)
    out: list[Configuration] = []
    while len(out) < n_configs:
        n = int(torch.randint(min_atoms, max_atoms + 1, (1,), generator=rng).item())
        pos = _sample_positions(n, box, rng, min_dist)
        Z = torch.tensor(
            [elements[int(torch.randint(0, len(elements), (1,), generator=rng).item())]
             for _ in range(n)],
            dtype=torch.long,
        )
        # Compute energy and forces in float64 for numerical precision, store float32.
        pos64 = pos.double().requires_grad_(True)
        E = ref.energy(pos64, Z)
        (g,) = torch.autograd.grad(E, pos64)
        forces = -g
        if not torch.isfinite(E).item() or not torch.isfinite(forces).all().item():
            continue
        if E.item() > energy_cap or forces.abs().max().item() > force_cap:
            continue  # keep samples near the minimum / attractive region
        out.append(Configuration(
            positions=pos.float().detach(),
            Z=Z,
            energy=E.float().detach(),
            forces=forces.float().detach(),
        ))
    return out


@dataclass
class RichConfiguration:
    """v2 sample: positions + Z + per-atom chemistry context + bond list."""
    positions: torch.Tensor       # (N, 3) float32
    Z: torch.Tensor               # (N,) long
    hybridisation: torch.Tensor   # (N,) long
    formal_charge: torch.Tensor   # (N,) long
    aromatic: torch.Tensor        # (N,) bool
    in_ring: torch.Tensor         # (N,) bool
    h_donor: torch.Tensor         # (N,) bool
    h_acceptor: torch.Tensor      # (N,) bool
    partial_charge: torch.Tensor  # (N,) float32
    bonds: torch.Tensor           # (B, 2) long, each row (i, j) with i < j
    bond_type: torch.Tensor       # (B,) long
    energy: torch.Tensor          # scalar float32
    forces: torch.Tensor          # (N, 3) float32

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]


def _sample_chain_with_bonds(
    n_atoms: int,
    Z: torch.Tensor,
    rng: torch.Generator,
    min_dist: float,
    p_chain_extends: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place atoms as a partly-bonded chain: walk a random path in 3D,
    placing each new atom near the previous one at typical bond length when
    we "extend the bond", or in a random direction further away when we don't.

    Returns (positions (N,3), bonds (B,2)).
    """
    # Pick a typical bond length per element pair
    pos = [torch.zeros(3)]
    bonds: list[tuple[int, int]] = []
    for k in range(1, n_atoms):
        extends = torch.rand((), generator=rng).item() < p_chain_extends
        # Random direction
        direction = torch.randn(3, generator=rng)
        direction = direction / torch.linalg.norm(direction).clamp_min(1e-8)
        if extends:
            za, zb = sorted([int(Z[k - 1]), int(Z[k])])
            if (za, zb) in MORSE:
                r_e = MORSE[(za, zb)]["r_e"]
            else:
                r_e = 1.5
            d = r_e + (torch.rand((), generator=rng).item() - 0.5) * 0.2
            new_pos = pos[-1] + direction * d
            # Make sure it doesn't clash with earlier atoms (other than k-1).
            ok = all(torch.linalg.norm(new_pos - p).item() >= min_dist
                      for j, p in enumerate(pos) if j != k - 1)
            if ok:
                pos.append(new_pos)
                bonds.append((k - 1, k))
                continue
        # Otherwise place at non-bond distance.
        for _ in range(50):
            d = 2.5 + torch.rand((), generator=rng).item() * 2.0  # 2.5-4.5 A
            new_pos = pos[-1] + direction * d
            ok = all(torch.linalg.norm(new_pos - p).item() >= min_dist for p in pos)
            if ok:
                pos.append(new_pos)
                break
        else:
            # Could not place safely; restart this atom from the origin region
            new_pos = (torch.rand(3, generator=rng) - 0.5) * 4.0
            pos.append(new_pos)
    positions = torch.stack(pos, dim=0)
    if bonds:
        bond_t = torch.tensor(bonds, dtype=torch.long)
    else:
        bond_t = torch.empty(0, 2, dtype=torch.long)
    return positions, bond_t


def _random_bonds(
    positions: torch.Tensor, max_bond_distance: float, rng: torch.Generator
) -> torch.Tensor:
    """Sample a small set of plausible covalent bonds.

    We pick atom pairs whose distance is within a chemically reasonable
    bond range; each such candidate becomes a bond with probability 0.5
    (sampled with the provided RNG). This is not real chemistry, but it
    gives the model a learnable signal: bonded pairs at short distance
    have very different energies from non-bonded pairs at the same
    distance, by construction of `BondAwareReference`.
    """
    N = positions.shape[0]
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    r = torch.linalg.norm(diff, dim=-1)
    # Upper triangle of pairs within bond range.
    candidates = []
    for i in range(N):
        for j in range(i + 1, N):
            if r[i, j].item() <= max_bond_distance:
                candidates.append((i, j))
    if not candidates:
        return torch.empty(0, 2, dtype=torch.long)
    keep = [
        torch.rand((), generator=rng).item() < 0.5
        for _ in candidates
    ]
    chosen = [c for c, k in zip(candidates, keep) if k]
    if not chosen:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(chosen, dtype=torch.long)


def generate_rich_dataset(
    n_configs: int,
    min_atoms: int = 3,
    max_atoms: int = 6,
    min_dist: float = 1.3,
    elements: tuple[int, ...] = (6, 8),
    seed: int = 0,
    r_cutoff: float = 6.0,
    energy_cap: float = 12.0,
    force_cap: float = 30.0,
    p_chain_extends: float = 0.6,
) -> list[RichConfiguration]:
    """Configurations with bonds + chemistry context + Morse/LJ ground truth.

    Atoms are placed as a partial chain so bonded pairs land near their
    Morse equilibrium distance and non-bonded pairs land further apart.
    """
    rng = torch.Generator().manual_seed(seed)
    ref = BondAwareReference(r_cutoff=r_cutoff, smooth=True)
    out: list[RichConfiguration] = []
    attempts = 0
    while len(out) < n_configs and attempts < n_configs * 200:
        attempts += 1
        n = int(torch.randint(min_atoms, max_atoms + 1, (1,), generator=rng).item())
        Z = torch.tensor(
            [elements[int(torch.randint(0, len(elements), (1,), generator=rng).item())]
             for _ in range(n)],
            dtype=torch.long,
        )
        pos, bonds = _sample_chain_with_bonds(n, Z, rng, min_dist, p_chain_extends)
        # Keep only bonds we have Morse parameters for.
        ok_bonds = []
        for b in bonds.tolist():
            za, zb = sorted([int(Z[b[0]]), int(Z[b[1]])])
            if (za, zb) in MORSE:
                ok_bonds.append(b)
        bonds = (
            torch.tensor(ok_bonds, dtype=torch.long)
            if ok_bonds else torch.empty(0, 2, dtype=torch.long)
        )

        # Chemistry context: randomised per-atom labels (not learned by LJ;
        # the v2 network has them as inputs and we just need *some* labels).
        hyb = torch.tensor(
            [int(torch.randint(HYB_SP, HYB_AROMATIC + 1, (1,), generator=rng).item())
             for _ in range(n)],
            dtype=torch.long,
        )
        formal_charge = torch.zeros(n, dtype=torch.long)
        aromatic = torch.zeros(n, dtype=torch.bool)
        in_ring = torch.zeros(n, dtype=torch.bool)
        # H-bond donor/acceptor flags: O can accept; we won't use H explicitly
        h_donor = torch.zeros(n, dtype=torch.bool)
        h_acceptor = (Z == 8)
        # Partial charges: small, sum to zero
        raw = torch.randn(n, generator=rng) * 0.2
        partial_charge = raw - raw.mean()

        # Energy + forces (Morse for bonded pairs, LJ otherwise)
        pos64 = pos.double().requires_grad_(True)
        E = ref.energy(pos64, Z, bonds)
        if not torch.isfinite(E).item():
            continue
        (g,) = torch.autograd.grad(E, pos64)
        forces = -g
        if (not torch.isfinite(forces).all().item()
                or E.item() > energy_cap
                or forces.abs().max().item() > force_cap):
            continue

        bond_type = torch.full(
            (bonds.shape[0],), BOND_SINGLE, dtype=torch.long
        )
        out.append(RichConfiguration(
            positions=pos.float().detach(),
            Z=Z,
            hybridisation=hyb,
            formal_charge=formal_charge,
            aromatic=aromatic,
            in_ring=in_ring,
            h_donor=h_donor,
            h_acceptor=h_acceptor,
            partial_charge=partial_charge.float().detach(),
            bonds=bonds,
            bond_type=bond_type,
            energy=E.float().detach(),
            forces=forces.float().detach(),
        ))
    if len(out) < n_configs:
        raise RuntimeError(f"only generated {len(out)} / {n_configs}; loosen filters")
    return out


def rich_dataset_stats(configs: list[RichConfiguration]) -> dict:
    Es = torch.stack([c.energy for c in configs])
    F_all = torch.cat([c.forces.reshape(-1) for c in configs])
    n_bonds_per = [c.bonds.shape[0] for c in configs]
    return {
        "n_configs": len(configs),
        "energy_mean": float(Es.mean()),
        "energy_std": float(Es.std()),
        "energy_min": float(Es.min()),
        "energy_max": float(Es.max()),
        "force_rms": float(F_all.pow(2).mean().sqrt()),
        "n_atoms_total": sum(c.n_atoms for c in configs),
        "bonds_per_config_mean": sum(n_bonds_per) / len(configs),
        "bonds_per_config_max": max(n_bonds_per),
    }


def dataset_stats(configs: list[Configuration]) -> dict:
    Es = torch.stack([c.energy for c in configs])
    F_all = torch.cat([c.forces.reshape(-1) for c in configs])
    return {
        "n_configs": len(configs),
        "energy_mean": float(Es.mean()),
        "energy_std": float(Es.std()),
        "energy_min": float(Es.min()),
        "energy_max": float(Es.max()),
        "force_rms": float(F_all.pow(2).mean().sqrt()),
        "force_abs_max": float(F_all.abs().max()),
        "n_atoms_total": sum(c.n_atoms for c in configs),
    }
