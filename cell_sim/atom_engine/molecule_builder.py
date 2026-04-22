"""Small-molecule templates for the chemistry demo.

A ``MoleculeTemplate`` carries:
  - the chemical formula (for classifier output)
  - relative atom positions (nm, centered on the heavy atom)
  - a list of intra-molecule bonds with their equilibrium distances
    and spring constants

``build_mixture(counts, radius, ...)`` drops ``n`` randomly-rotated,
non-overlapping copies of each template into a sphere of the given
radius and returns ``(atoms, bonds)``. Atoms carry a
``parent_molecule`` tag of the form ``"water#3"`` so the classifier
can tell which molecule each atom started in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .atom_unit import AtomUnit, Bond, BondType
from .element import Element, mass

_K_B_KJ_PER_MOL_K = 0.00831446

# --- Typical covalent bond lengths (nm). Used as equilibrium distances. ---
_BOND_LENGTH_NM = {
    frozenset([Element.H, Element.H]): 0.074,
    frozenset([Element.C, Element.H]): 0.109,
    frozenset([Element.O, Element.H]): 0.096,
    frozenset([Element.N, Element.H]): 0.101,
    frozenset([Element.S, Element.H]): 0.134,
    frozenset([Element.C, Element.C]): 0.154,
    frozenset([Element.C, Element.N]): 0.147,
    frozenset([Element.C, Element.O]): 0.143,
    frozenset([Element.C, Element.S]): 0.182,
    frozenset([Element.N, Element.N]): 0.125,   # N=N / N#N average
    frozenset([Element.O, Element.O]): 0.121,   # O=O
    frozenset([Element.N, Element.O]): 0.121,   # N=O
    frozenset([Element.S, Element.S]): 0.205,
}

_DEFAULT_BOND_K = 3.0e5   # kJ/mol/nm^2 — stiff enough to stay bound at
                           # 3000 K, loose enough for 1 fs timesteps to
                           # be stable.


def _bond_length(a: Element, b: Element, override: Optional[float] = None) -> float:
    if override is not None:
        return override
    key = frozenset([a, b])
    return _BOND_LENGTH_NM.get(key, 0.15)


@dataclass
class MoleculeTemplate:
    name: str
    formula: str
    atoms: list[tuple[Element, tuple[float, float, float]]]
    # Each bond entry: (i, j, kind, r0 override, k override). ``None`` for
    # r0/k uses the default bond length / _DEFAULT_BOND_K.
    bonds: list[tuple[int, int, BondType, Optional[float], Optional[float]]]

    @property
    def radius_nm(self) -> float:
        """Max distance from origin of any atom in the template."""
        return max(math.sqrt(x * x + y * y + z * z)
                   for _, (x, y, z) in self.atoms)


# ---------- Molecule library ------------------------------------------
# Heavy atom at origin when sensible. All geometries are approximate —
# good enough for an MD toy, not crystallography.

_S = BondType.COVALENT_SINGLE

H2 = MoleculeTemplate(
    name="hydrogen", formula="H2",
    atoms=[
        (Element.H, (0.0, 0.0, 0.0)),
        (Element.H, (0.074, 0.0, 0.0)),
    ],
    bonds=[(0, 1, _S, None, None)],
)

O2 = MoleculeTemplate(
    name="oxygen", formula="O2",
    atoms=[
        (Element.O, (0.0, 0.0, 0.0)),
        (Element.O, (0.121, 0.0, 0.0)),
    ],
    bonds=[(0, 1, BondType.COVALENT_DOUBLE, None, None)],
)

N2 = MoleculeTemplate(
    name="nitrogen", formula="N2",
    atoms=[
        (Element.N, (0.0, 0.0, 0.0)),
        (Element.N, (0.110, 0.0, 0.0)),
    ],
    bonds=[(0, 1, BondType.COVALENT_TRIPLE, 0.110, None)],
)

# Water: O at origin, two H's at 104.5° with O-H = 0.096 nm
_theta = math.radians(104.5 / 2)
_oh = 0.096
H2O = MoleculeTemplate(
    name="water", formula="H2O",
    atoms=[
        (Element.O, (0.0, 0.0, 0.0)),
        (Element.H, (_oh * math.sin(_theta),  _oh * math.cos(_theta), 0.0)),
        (Element.H, (-_oh * math.sin(_theta), _oh * math.cos(_theta), 0.0)),
    ],
    bonds=[(0, 1, _S, None, None), (0, 2, _S, None, None)],
)

# Methane: C at origin, 4 H at tetrahedral vertices with C-H = 0.109 nm
_ch = 0.109
_t = _ch / math.sqrt(3)
CH4 = MoleculeTemplate(
    name="methane", formula="CH4",
    atoms=[
        (Element.C, (0.0, 0.0, 0.0)),
        (Element.H, ( _t,  _t,  _t)),
        (Element.H, ( _t, -_t, -_t)),
        (Element.H, (-_t,  _t, -_t)),
        (Element.H, (-_t, -_t,  _t)),
    ],
    bonds=[(0, 1, _S, None, None), (0, 2, _S, None, None),
           (0, 3, _S, None, None), (0, 4, _S, None, None)],
)

# Ammonia: N at apex, 3 H's below at ~107°. H-N-H 107°; N-H = 0.101 nm
_nh = 0.101
_nh_xy = _nh * math.sin(math.radians(68))
_nh_z = _nh * math.cos(math.radians(68))
NH3 = MoleculeTemplate(
    name="ammonia", formula="NH3",
    atoms=[
        (Element.N, (0.0, 0.0, 0.0)),
        (Element.H, ( _nh_xy,                      0.0,          -_nh_z)),
        (Element.H, (-_nh_xy * 0.5,   _nh_xy * math.sqrt(3) / 2, -_nh_z)),
        (Element.H, (-_nh_xy * 0.5,  -_nh_xy * math.sqrt(3) / 2, -_nh_z)),
    ],
    bonds=[(0, 1, _S, None, None), (0, 2, _S, None, None), (0, 3, _S, None, None)],
)

# CO: C-O with triple-ish bond (treat as single in our valence model so
# C still has 3 slots and O has 1)
CO = MoleculeTemplate(
    name="carbon_monoxide", formula="CO",
    atoms=[
        (Element.C, (0.0, 0.0, 0.0)),
        (Element.O, (0.113, 0.0, 0.0)),
    ],
    bonds=[(0, 1, _S, 0.113, None)],
)

# CO2: O=C=O linear
CO2 = MoleculeTemplate(
    name="carbon_dioxide", formula="CO2",
    atoms=[
        (Element.C, (0.0, 0.0, 0.0)),
        (Element.O, ( 0.116, 0.0, 0.0)),
        (Element.O, (-0.116, 0.0, 0.0)),
    ],
    bonds=[(0, 1, BondType.COVALENT_DOUBLE, 0.116, None),
           (0, 2, BondType.COVALENT_DOUBLE, 0.116, None)],
)


LIBRARY: dict[str, MoleculeTemplate] = {
    "H2": H2, "O2": O2, "N2": N2,
    "H2O": H2O, "CH4": CH4, "NH3": NH3,
    "CO": CO, "CO2": CO2,
}


# ---------- Mixture builder -------------------------------------------


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3D rotation (quaternion method)."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array([
        [1 - 2 * (q3 * q3 + q4 * q4),   2 * (q2 * q3 - q4 * q1),     2 * (q2 * q4 + q3 * q1)],
        [    2 * (q2 * q3 + q4 * q1),   1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q2 * q1)],
        [    2 * (q2 * q4 - q3 * q1),   2 * (q3 * q4 + q2 * q1),     1 - 2 * (q2 * q2 + q3 * q3)],
    ], dtype=np.float64)


def _mb_velocity(mass_da: float, T_K: float,
                 rng: np.random.Generator) -> np.ndarray:
    sigma = math.sqrt(_K_B_KJ_PER_MOL_K * T_K / mass_da)
    return rng.normal(0.0, sigma, size=3)


def build_mixture(
    composition: dict[str, int],
    radius_nm: float,
    temperature_K: float = 300.0,
    seed: Optional[int] = 42,
    min_center_separation_nm: float = 0.4,
    bond_k_kj_per_nm2: Optional[float] = None,
) -> tuple[list[AtomUnit], list[Bond]]:
    """Place ``count`` copies of each named template in a sphere.

    ``bond_k_kj_per_nm2`` overrides the default bond spring constant
    (3e5 kJ/mol/nm^2) for every newly-built bond. Lower values make the
    toy more reactive at fixed T, which is useful for demonstrating
    chemistry that would otherwise require unrealistic temperatures.
    """
    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    min2 = min_center_separation_nm ** 2
    max_tries = 2000

    # Resolve composition into a flat list of templates.
    plan: list[MoleculeTemplate] = []
    for formula, count in composition.items():
        if formula not in LIBRARY:
            raise ValueError(f"unknown molecule '{formula}'; known: "
                             f"{sorted(LIBRARY.keys())}")
        for _ in range(count):
            plan.append(LIBRARY[formula])
    rng.shuffle(plan)

    # Precompute available max molecule radius for the rejection sampler
    # (conservative: treat every molecule as its own bounding sphere).
    max_mol_r = max((t.radius_nm for t in plan), default=0.0)
    r_place = max(0.0, radius_nm - max_mol_r)

    # Place centers.
    for _ in range(len(plan)):
        ok = False
        for _ in range(max_tries):
            v = rng.uniform(-r_place, r_place, size=3)
            if v @ v > r_place * r_place:
                continue
            close = False
            for c in centers:
                if (c - v) @ (c - v) < min2:
                    close = True
                    break
            if not close:
                centers.append(v)
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"could not place {len(plan)} molecules in sphere "
                f"r={radius_nm}; already placed {len(centers)}. Try a "
                f"larger radius or smaller molecule count."
            )

    # Build atoms + bonds.
    atoms: list[AtomUnit] = []
    bonds: list[Bond] = []
    for idx, (template, center) in enumerate(zip(plan, centers)):
        tag = f"{template.name}#{idx}"
        R = _random_rotation_matrix(rng)
        com_vel = _mb_velocity(
            sum(mass(e) for e, _ in template.atoms),
            temperature_K, rng,
        )
        atom_objs: list[AtomUnit] = []
        for elem, rel_pos in template.atoms:
            p = R @ np.array(rel_pos) + center
            v = com_vel + 0.3 * _mb_velocity(mass(elem), temperature_K, rng)
            atom = AtomUnit.create(
                element=elem,
                position=(float(p[0]), float(p[1]), float(p[2])),
                velocity=(float(v[0]), float(v[1]), float(v[2])),
                parent_molecule=tag,
            )
            atoms.append(atom)
            atom_objs.append(atom)
        for i, j, kind, r0_override, k_override in template.bonds:
            ai = atom_objs[i]
            aj = atom_objs[j]
            r0 = _bond_length(ai.element, aj.element, r0_override)
            if k_override is not None:
                k = k_override
            elif bond_k_kj_per_nm2 is not None:
                k = bond_k_kj_per_nm2
            else:
                k = _DEFAULT_BOND_K
            bond = ai.form_bond(
                aj, kind=kind, t_ps=0.0,
                equilibrium_length_nm=r0,
                spring_constant_kj_per_nm2=k,
            )
            bonds.append(bond)

    _zero_net_momentum(atoms)
    return atoms, bonds


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


# ---------- Molecule classifier ---------------------------------------


def _connected_components_by_live_bonds(
    atoms: list[AtomUnit],
) -> list[list[int]]:
    """Return list of atom-index groups linked by any LIVE bond."""
    n = len(atoms)
    parent = list(range(n))
    id_to_idx = {id(a): i for i, a in enumerate(atoms)}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for a in atoms:
        for b in a.bonds:
            if b.death_time_ps is not None:
                continue
            i = id_to_idx.get(id(b.a))
            j = id_to_idx.get(id(b.b))
            if i is not None and j is not None:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


_HILL_ORDER = [Element.C, Element.H, Element.N, Element.O, Element.P,
               Element.S, Element.F, Element.Cl, Element.Na, Element.K,
               Element.Mg, Element.Ca, Element.Fe, Element.Cu, Element.Zn]


def canonical_formula(atom_indices: list[int],
                      atoms: list[AtomUnit]) -> str:
    """Return a Hill-system formula string ('C2H6O', 'H2O', ...)."""
    counts: dict[Element, int] = {}
    for i in atom_indices:
        e = atoms[i].element
        counts[e] = counts.get(e, 0) + 1
    parts: list[str] = []
    for e in _HILL_ORDER:
        c = counts.pop(e, 0)
        if c == 1:
            parts.append(e.name)
        elif c > 1:
            parts.append(f"{e.name}{c}")
    # Any leftover non-canonical elements, in enum order.
    for e in sorted(counts.keys(), key=lambda x: x.value):
        c = counts[e]
        parts.append(e.name if c == 1 else f"{e.name}{c}")
    return "".join(parts) if parts else "?"


def classify_molecules(atoms: list[AtomUnit]) -> dict[str, int]:
    """Return a dict mapping canonical formula -> count of such molecules
    based on the current live-bond graph. Lone atoms are counted as their
    element (e.g. ``H``).
    """
    comps = _connected_components_by_live_bonds(atoms)
    out: dict[str, int] = {}
    for group in comps:
        f = canonical_formula(group, atoms)
        out[f] = out.get(f, 0) + 1
    return out
