"""AtomUnit — the fundamental atom-as-primitive object.

Each AtomUnit carries its own identity (element), kinematic state
(position + velocity), bonding state (list of bonds with finite valence),
and a per-atom event history log. Bonds form when two atoms satisfy a
compatibility test and enough valence remains; bonds break when strained
past a force threshold or explicitly broken by a chemistry event.

Design goals:
- Conservation: mass, charge, valence are enforced by construction.
- Provenance: every bond-forming or bond-breaking event is logged with a
  timestamp and the participating atoms so the full trajectory is
  reconstructible from the event list alone.
- Event-driven: no fixed time grid required for the bond state. The MD
  integrator can run at fs timesteps for kinematics, but the bond graph
  records only the discrete change-points.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .element import Element, default_valence, mass, pair_is_bondable, props


class BondType(Enum):
    """Categorical bond type. Order is the covalent bond order where
    meaningful; for non-covalent, order is the effective valence usage
    (hydrogen bonds use up 0.5 slot, ionic 1, disulfide 1)."""
    COVALENT_SINGLE = "covalent_single"
    COVALENT_DOUBLE = "covalent_double"
    COVALENT_TRIPLE = "covalent_triple"
    HYDROGEN = "hydrogen"
    IONIC = "ionic"
    DISULFIDE = "disulfide"
    COARSE_CHAIN = "coarse_chain"   # synthetic chain connectivity in toys


_ORDER = {
    BondType.COVALENT_SINGLE: 1,
    BondType.COVALENT_DOUBLE: 2,
    BondType.COVALENT_TRIPLE: 3,
    BondType.HYDROGEN: 0,           # occupies no covalent valence slot
    BondType.IONIC: 0,
    BondType.DISULFIDE: 1,
    BondType.COARSE_CHAIN: 1,
}


def bond_valence_cost(bt: BondType) -> int:
    return _ORDER[bt]


_global_atom_id = itertools.count()


@dataclass(eq=False, slots=True)
class Bond:
    """A bond between two AtomUnit instances."""
    a: "AtomUnit"
    b: "AtomUnit"
    kind: BondType
    birth_time_ps: float
    equilibrium_length_nm: float
    spring_constant_kj_per_nm2: float
    energy_kj_per_mol: float = 0.0      # filled at formation
    death_time_ps: Optional[float] = None  # set when bond breaks

    def other(self, atom: "AtomUnit") -> "AtomUnit":
        if atom is self.a:
            return self.b
        if atom is self.b:
            return self.a
        raise ValueError("atom not in bond")

    @property
    def order(self) -> int:
        return _ORDER[self.kind]


@dataclass(eq=False, slots=True)
class DihedralBond:
    """4-body periodic dihedral around the j-k bond (i-j-k-l).

    ``U = k_phi * (1 + cos(n*phi - phi_0))``

    Matches the standard OPLS / AMBER "proper" dihedral form. ``n``
    is the multiplicity (1, 2, or 3 typical), ``phi_0`` the phase
    (0 or pi), ``k_phi`` in kJ/mol. The backbone psi (N-Ca-C-N) and
    phi (C-N-Ca-C) dihedrals use this with multiplicity 1 or 2.
    """
    i: "AtomUnit"
    j: "AtomUnit"
    k: "AtomUnit"
    l: "AtomUnit"
    n: int
    phi_0_rad: float
    k_phi_kj_per_mol: float


@dataclass(eq=False, slots=True)
class AngleBond:
    """3-body harmonic angle on the i-j-k bend (j is the vertex).

    ``U = 0.5 * k_theta * (theta - theta_0)^2`` where
    ``cos(theta) = (r_ji . r_jk) / (|r_ji| |r_jk|)``.

    ``theta_0_rad`` is in radians; ``k_theta_kj_per_mol_rad2`` in
    kJ/(mol·rad²). Typical organic values: k ~ 400-700, theta_0 =
    109.47 deg (sp3 carbon), 104.5 deg (water O), 107 deg (NH3 N),
    120 deg (sp2 trigonal).
    """
    i: "AtomUnit"
    j: "AtomUnit"                 # central atom (vertex)
    k: "AtomUnit"
    theta_0_rad: float
    k_theta_kj_per_mol_rad2: float


@dataclass(eq=False, slots=True)
class AtomEvent:
    """An event in an atom's history. Timestamp is simulation time in ps."""
    kind: str                         # "bond_formed" | "bond_broken" | "reaction" | "created"
    t_ps: float
    atom_ids: tuple[int, ...]         # (self_id, partner_id, ...)
    bond_kind: Optional[BondType] = None
    notes: str = ""


@dataclass(eq=False, slots=True)
class AtomUnit:
    """A single atom-as-particle.

    Identity (element, parent_molecule, atom_id) is immutable. Kinematics
    (position, velocity) are mutable. Bonds form/break during the sim and
    every change is appended to `history`.
    """
    element: Element
    atom_id: int
    parent_molecule: str

    # Kinematics (nm, nm/ps)
    position: list[float]   # length 3 (mutable for perf; use list to avoid overhead)
    velocity: list[float]   # length 3

    # Electronic coarse state
    formal_charge: int = 0
    partial_charge: float = 0.0

    # Bonding
    bonds: list[Bond] = field(default_factory=list)

    # Provenance
    birth_time_ps: float = 0.0
    history: list[AtomEvent] = field(default_factory=list)

    # --- construction helpers -------------------------------------------
    @classmethod
    def create(
        cls,
        element: Element,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        parent_molecule: str = "free",
        formal_charge: int = 0,
        birth_time_ps: float = 0.0,
    ) -> "AtomUnit":
        pc = float(props(element).partial_charge) + float(formal_charge)
        aid = next(_global_atom_id)
        atom = cls(
            element=element,
            atom_id=aid,
            parent_molecule=parent_molecule,
            position=list(position),
            velocity=list(velocity),
            formal_charge=int(formal_charge),
            partial_charge=pc,
            birth_time_ps=birth_time_ps,
        )
        atom.history.append(AtomEvent(
            kind="created", t_ps=birth_time_ps, atom_ids=(aid,),
            notes=f"element={element.name}",
        ))
        return atom

    # --- accessors ------------------------------------------------------
    @property
    def mass_da(self) -> float:
        return mass(self.element)

    @property
    def valence_used(self) -> int:
        return sum(bond_valence_cost(b.kind) for b in self.bonds)

    @property
    def valence_remaining(self) -> int:
        return default_valence(self.element) - self.valence_used

    # --- bonding --------------------------------------------------------
    def is_bonded_to(self, other: "AtomUnit") -> bool:
        for b in self.bonds:
            if b.a is other or b.b is other:
                return True
        return False

    def can_bond_to(
        self,
        other: "AtomUnit",
        kind: BondType = BondType.COVALENT_SINGLE,
    ) -> bool:
        """Check: same-element rule respected, pair allowed, both atoms
        have valence remaining, not already bonded."""
        if other is self:
            return False
        if self.is_bonded_to(other):
            return False
        cost = bond_valence_cost(kind)
        if self.valence_remaining < cost:
            return False
        if other.valence_remaining < cost:
            return False
        if kind in (BondType.COVALENT_SINGLE, BondType.COVALENT_DOUBLE,
                    BondType.COVALENT_TRIPLE, BondType.COARSE_CHAIN):
            return pair_is_bondable(self.element, other.element)
        if kind == BondType.DISULFIDE:
            return self.element == Element.S and other.element == Element.S
        if kind == BondType.HYDROGEN:
            # Donor must be H bonded to N/O/F; acceptor must be N/O/F with lone pair
            return self.element == Element.H or other.element == Element.H
        if kind == BondType.IONIC:
            return self.partial_charge * other.partial_charge < 0
        return False

    def form_bond(
        self,
        other: "AtomUnit",
        kind: BondType,
        t_ps: float,
        equilibrium_length_nm: float,
        spring_constant_kj_per_nm2: float,
        energy_kj_per_mol: float = 0.0,
    ) -> Bond:
        if not self.can_bond_to(other, kind):
            raise ValueError(
                f"can't form {kind.name} bond between "
                f"{self.element.name}[{self.atom_id}] and "
                f"{other.element.name}[{other.atom_id}]"
            )
        bond = Bond(
            a=self, b=other, kind=kind,
            birth_time_ps=t_ps,
            equilibrium_length_nm=equilibrium_length_nm,
            spring_constant_kj_per_nm2=spring_constant_kj_per_nm2,
            energy_kj_per_mol=energy_kj_per_mol,
        )
        self.bonds.append(bond)
        other.bonds.append(bond)
        ev = AtomEvent(
            kind="bond_formed", t_ps=t_ps,
            atom_ids=(self.atom_id, other.atom_id),
            bond_kind=kind,
        )
        self.history.append(ev)
        other.history.append(ev)
        return bond

    def break_bond(self, bond: Bond, t_ps: float, reason: str = "") -> None:
        if bond not in self.bonds:
            raise ValueError("bond not on this atom")
        other = bond.other(self)
        self.bonds.remove(bond)
        other.bonds.remove(bond)
        bond.death_time_ps = t_ps
        ev = AtomEvent(
            kind="bond_broken", t_ps=t_ps,
            atom_ids=(self.atom_id, other.atom_id),
            bond_kind=bond.kind, notes=reason,
        )
        self.history.append(ev)
        other.history.append(ev)

    # --- diagnostics ----------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"AtomUnit({self.element.name}#{self.atom_id} "
            f"pos=({self.position[0]:.3f},{self.position[1]:.3f},{self.position[2]:.3f}) "
            f"bonds={len(self.bonds)}/{default_valence(self.element)})"
        )
