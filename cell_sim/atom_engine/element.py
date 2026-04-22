"""Elements + atomic data used by the AtomUnit primitive.

Source: CODATA 2018 atomic masses, OPLS-AA typical Lennard-Jones parameters.
Valence values are typical ground-state bonding count; a few elements (S, P,
Fe, Cu) accept a range but the :data:`VALENCE` table picks the most common
for Syn3A-relevant chemistry.

For a toy MD simulator we don't need DFT-level quantum accuracy; we need
the categorical identity (element -> mass / valence / size) to be right so
that bonding rules and conservation laws hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Element(Enum):
    """Elements relevant to Syn3A + generic biochemistry.

    Assignments for `COARSE_*` are pseudo-elements for lipid-like particles
    in the fission demo. They are NOT chemistry; they carry just enough
    state (mass, LJ size) to build a membrane and watch it pinch.
    """
    # Real elements (Z = atomic number)
    H = 1
    C = 6
    N = 7
    O = 8
    F = 9
    Na = 11
    Mg = 12
    P = 15
    S = 16
    Cl = 17
    K = 19
    Ca = 20
    Fe = 26
    Cu = 29
    Zn = 30
    # Pseudo-elements for coarse demos (distinct valence / size / stickiness)
    COARSE_HEAD = 100   # hydrophilic head (like choline + phosphate lumped)
    COARSE_TAIL = 101   # hydrophobic tail (like acyl chain lumped)
    COARSE_SOLVENT = 102  # bulk water coarse bead


@dataclass(frozen=True, slots=True)
class ElementProps:
    mass_da: float              # atomic mass, Da
    default_valence: int        # typical bond count in ground state
    lj_sigma_nm: float          # Lennard-Jones sigma, nm
    lj_epsilon_kj: float        # Lennard-Jones epsilon, kJ/mol
    partial_charge: float       # default formal charge
    is_coarse: bool = False     # pseudo-element flag


# Derived primarily from OPLS-AA; coarse values chosen so the MD integrator
# stays numerically stable at 5 fs timesteps in the fission demo.
_PROPS: dict[Element, ElementProps] = {
    Element.H:  ElementProps(mass_da=1.008,  default_valence=1, lj_sigma_nm=0.25,  lj_epsilon_kj=0.125,  partial_charge=0.0),
    Element.C:  ElementProps(mass_da=12.011, default_valence=4, lj_sigma_nm=0.34,  lj_epsilon_kj=0.457,  partial_charge=0.0),
    Element.N:  ElementProps(mass_da=14.007, default_valence=3, lj_sigma_nm=0.325, lj_epsilon_kj=0.711,  partial_charge=0.0),
    Element.O:  ElementProps(mass_da=15.999, default_valence=2, lj_sigma_nm=0.296, lj_epsilon_kj=0.879,  partial_charge=0.0),
    Element.F:  ElementProps(mass_da=18.998, default_valence=1, lj_sigma_nm=0.295, lj_epsilon_kj=0.314,  partial_charge=0.0),
    Element.Na: ElementProps(mass_da=22.990, default_valence=1, lj_sigma_nm=0.333, lj_epsilon_kj=0.012,  partial_charge=1.0),
    Element.Mg: ElementProps(mass_da=24.305, default_valence=2, lj_sigma_nm=0.210, lj_epsilon_kj=0.370,  partial_charge=2.0),
    Element.P:  ElementProps(mass_da=30.974, default_valence=5, lj_sigma_nm=0.374, lj_epsilon_kj=0.836,  partial_charge=0.0),
    Element.S:  ElementProps(mass_da=32.065, default_valence=2, lj_sigma_nm=0.356, lj_epsilon_kj=1.046,  partial_charge=0.0),
    Element.Cl: ElementProps(mass_da=35.453, default_valence=1, lj_sigma_nm=0.441, lj_epsilon_kj=0.492,  partial_charge=-1.0),
    Element.K:  ElementProps(mass_da=39.098, default_valence=1, lj_sigma_nm=0.493, lj_epsilon_kj=0.001,  partial_charge=1.0),
    Element.Ca: ElementProps(mass_da=40.078, default_valence=2, lj_sigma_nm=0.241, lj_epsilon_kj=1.880,  partial_charge=2.0),
    Element.Fe: ElementProps(mass_da=55.845, default_valence=3, lj_sigma_nm=0.260, lj_epsilon_kj=0.063,  partial_charge=2.0),
    Element.Cu: ElementProps(mass_da=63.546, default_valence=2, lj_sigma_nm=0.339, lj_epsilon_kj=0.021,  partial_charge=2.0),
    Element.Zn: ElementProps(mass_da=65.380, default_valence=2, lj_sigma_nm=0.196, lj_epsilon_kj=1.046,  partial_charge=2.0),
    # Coarse pseudo-elements: tuned for a stable toy bilayer at 300 K.
    Element.COARSE_HEAD:    ElementProps(mass_da=72.0, default_valence=2, lj_sigma_nm=0.47, lj_epsilon_kj=1.0, partial_charge=0.0, is_coarse=True),
    Element.COARSE_TAIL:    ElementProps(mass_da=72.0, default_valence=2, lj_sigma_nm=0.47, lj_epsilon_kj=3.5, partial_charge=0.0, is_coarse=True),
    Element.COARSE_SOLVENT: ElementProps(mass_da=72.0, default_valence=0, lj_sigma_nm=0.47, lj_epsilon_kj=0.5, partial_charge=0.0, is_coarse=True),
}


def props(e: Element) -> ElementProps:
    return _PROPS[e]


def mass(e: Element) -> float:
    return _PROPS[e].mass_da


def default_valence(e: Element) -> int:
    return _PROPS[e].default_valence


def is_coarse(e: Element) -> bool:
    return _PROPS[e].is_coarse


# Pairs that are allowed to form covalent bonds (for bond-forming kernels).
# Non-exhaustive: tuned for the molecules that appear in Syn3A + the fission
# demo. ('A','B') and ('B','A') both count.
_ALLOWED_COVALENT_PAIRS: set[tuple[Element, Element]] = {
    (Element.C, Element.C), (Element.C, Element.H), (Element.C, Element.N),
    (Element.C, Element.O), (Element.C, Element.S), (Element.C, Element.P),
    (Element.N, Element.H), (Element.N, Element.O), (Element.N, Element.N),
    (Element.O, Element.H), (Element.O, Element.P), (Element.O, Element.S),
    (Element.P, Element.O), (Element.P, Element.H),
    (Element.S, Element.H), (Element.S, Element.S),
    # Coarse tail-tail "bonds" (not chemistry; express chain connectivity).
    (Element.COARSE_HEAD, Element.COARSE_TAIL),
    (Element.COARSE_TAIL, Element.COARSE_TAIL),
}


def pair_is_bondable(a: Element, b: Element) -> bool:
    return (a, b) in _ALLOWED_COVALENT_PAIRS or (b, a) in _ALLOWED_COVALENT_PAIRS
