"""Generate training / test data from the analytic LJ reference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .physics import LennardJonesReference


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
