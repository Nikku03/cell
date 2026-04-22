"""Atom-soup seed builder for the reaction demo.

Place ``n`` atoms of selected elements in a spherical confinement
volume with Maxwell-Boltzmann velocities at temperature ``T``. No
pre-existing bonds. A minimum-distance rejection ensures no pair
starts closer than ``min_separation_nm`` to avoid LJ explosions on
step 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .atom_unit import AtomUnit
from .element import Element, mass

_K_B_KJ_PER_MOL_K = 0.00831446


@dataclass
class SoupSpec:
    # Element -> count. E.g. {Element.H: 40, Element.C: 10, ...}
    composition: dict[Element, int] = field(default_factory=dict)
    radius_nm: float = 2.0              # confinement / seed radius
    temperature_K: float = 1500.0
    min_separation_nm: float = 0.18     # rejection radius when placing atoms
    parent_molecule: str = "soup"
    seed: Optional[int] = 42


def _mb_velocity(mass_da: float, T_K: float,
                 rng: np.random.Generator) -> np.ndarray:
    sigma = math.sqrt(_K_B_KJ_PER_MOL_K * T_K / mass_da)
    return rng.normal(0.0, sigma, size=3)


def _rejection_sample(n: int, radius: float, min_sep: float,
                      rng: np.random.Generator,
                      max_tries: int = 2000) -> np.ndarray:
    """Return (n, 3) positions inside a sphere of ``radius`` with every
    pair separated by ``min_sep``. Raises if the box is too crowded."""
    pts = np.empty((n, 3), dtype=np.float64)
    min_sep2 = min_sep * min_sep
    placed = 0
    for _ in range(max_tries * n):
        # Uniform in sphere: sample cube, reject outside sphere
        v = rng.uniform(-radius, radius, size=3)
        if v @ v > radius * radius:
            continue
        ok = True
        for j in range(placed):
            d = pts[j] - v
            if d @ d < min_sep2:
                ok = False
                break
        if ok:
            pts[placed] = v
            placed += 1
            if placed == n:
                return pts
    raise RuntimeError(
        f"could not place {n} atoms in sphere r={radius} with "
        f"min_sep={min_sep} (placed {placed}); try a larger radius "
        f"or fewer atoms"
    )


def build_soup(spec: SoupSpec) -> list[AtomUnit]:
    """Place atoms of the given composition in a sphere at temperature T."""
    rng = np.random.default_rng(spec.seed)
    total = sum(spec.composition.values())
    if total <= 0:
        return []
    pos = _rejection_sample(total, spec.radius_nm, spec.min_separation_nm, rng)
    atoms: list[AtomUnit] = []
    idx = 0
    for element, count in spec.composition.items():
        for _ in range(count):
            p = pos[idx]
            v = _mb_velocity(mass(element), spec.temperature_K, rng)
            atoms.append(AtomUnit.create(
                element=element,
                position=(float(p[0]), float(p[1]), float(p[2])),
                velocity=(float(v[0]), float(v[1]), float(v[2])),
                parent_molecule=spec.parent_molecule,
            ))
            idx += 1
    _zero_net_momentum(atoms)
    return atoms


def _zero_net_momentum(atoms: list[AtomUnit]) -> None:
    total = sum(a.mass_da for a in atoms)
    if total <= 0.0:
        return
    px = sum(a.mass_da * a.velocity[0] for a in atoms) / total
    py = sum(a.mass_da * a.velocity[1] for a in atoms) / total
    pz = sum(a.mass_da * a.velocity[2] for a in atoms) / total
    for a in atoms:
        a.velocity[0] -= px
        a.velocity[1] -= py
        a.velocity[2] -= pz
