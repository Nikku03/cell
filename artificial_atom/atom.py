"""Atom-level data structures.

`ArtificialAtom` is the per-atom record (element, position, mass, learned
state). It corresponds to the "unit" in the user-facing spec - state plus
local physics plus learned features plus an uncertainty slot. The fields
that the model actually reads during inference are:

  - `Z` (atomic number; the rest of element identity is looked up)
  - `position` (3D coordinates)

Other fields (partial charge, hybridisation, donor/acceptor, vdW radius,
local energy, uncertainty) are stored as optional metadata that the
network's output head can populate, and that a downstream docking layer
would consume. They are not used by the LJ-toy training in this module.

`Element` is a tiny periodic-table-lookup wrapper. The model handles all
elements through a learned embedding table indexed by Z, so isotopes are
*not* distinguishable to the model - they would only enter dynamics
through `mass`, which the energy network does not see. This matches how
real MLIPs (SchNet, MACE, etc.) treat isotopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Atomic symbols by Z (index 0 unused; use Z=1..118).
_SYMBOLS = (
    "_,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,"
    "Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,"
    "Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,"
    "Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,"
    "Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,"
    "Nh,Fl,Mc,Lv,Ts,Og"
).split(",")
SYMBOL_TO_Z = {sym: Z for Z, sym in enumerate(_SYMBOLS) if Z > 0}

# Most common isotopic mass (in atomic mass units) for a handful of elements
# used in this demo. Real applications would use a full table.
_DEFAULT_MASS = {
    1: 1.008, 2: 4.003, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 15: 30.974, 16: 32.06, 17: 35.45,
}


@dataclass
class Element:
    Z: int
    symbol: str
    mass: float  # default isotope mass; isotope-specific values can override

    @staticmethod
    def from_symbol(sym: str) -> "Element":
        Z = SYMBOL_TO_Z[sym]
        return Element(Z=Z, symbol=sym, mass=_DEFAULT_MASS.get(Z, float(Z) * 2))

    @staticmethod
    def from_Z(Z: int) -> "Element":
        return Element(Z=Z, symbol=_SYMBOLS[Z], mass=_DEFAULT_MASS.get(Z, float(Z) * 2))


@dataclass
class ArtificialAtom:
    """One atom in the system.

    The fields the energy network reads are `Z` and `position`. Everything
    else is metadata that downstream code (docking, dynamics, analysis)
    may populate; the LJ training in this module does not need them.
    """

    Z: int                          # element atomic number (1..118)
    position: tuple[float, float, float]
    mass: float | None = None       # if None, use default for Z

    # Optional chemistry metadata (populated by featurisers, not by the toy potential)
    partial_charge: float = 0.0
    hybridisation: str = ""         # "sp", "sp2", "sp3", "aromatic", ...
    aromatic: bool = False
    donor: bool = False
    acceptor: bool = False
    vdw_radius: float = 0.0

    # Per-atom outputs populated by the network's output head
    local_energy: float = 0.0
    uncertainty: float = 0.0

    # Reserved slot for the network's hidden state at the last forward
    # pass (kept as a Python list so this dataclass stays torch-free).
    learned_features: list[float] = field(default_factory=list)

    @property
    def element(self) -> Element:
        return Element.from_Z(self.Z)

    @property
    def effective_mass(self) -> float:
        return self.mass if self.mass is not None else self.element.mass
