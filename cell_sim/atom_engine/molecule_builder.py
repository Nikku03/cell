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

from .atom_unit import AngleBond, AtomUnit, Bond, BondType, DihedralBond
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
    # Optional per-atom partial charges (same length as atoms). ``None``
    # means "use element default partial_charge". Used by the Coulomb
    # force (Physics Upgrade 1).
    partial_charges: Optional[list[float]] = None
    # Optional 3-body angle constraints: (i, j, k, theta_0_rad, k_theta).
    # j is the vertex. Used by the angle force (Physics Upgrade 2).
    angles: Optional[list[tuple[int, int, int, float, float]]] = None
    # Optional 4-body periodic dihedrals:
    # (i, j, k, l, n, phi_0_rad, k_phi_kj_per_mol).
    dihedrals: Optional[
        list[tuple[int, int, int, int, int, float, float]]
    ] = None

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

# Water: O at origin, two H's at 104.5 deg with O-H = 0.096 nm.
# TIP3P-style charges (O = -0.834 e, H = +0.417 e) + H-O-H angle
# constraint at 104.52 deg. Enables realistic H-bonding via the
# Coulomb + angle physics upgrades.
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
    partial_charges=[-0.834, 0.417, 0.417],
    angles=[(1, 0, 2, math.radians(104.52), 8000.0)],   # stiff H-O-H bend
                                                         # (8000 chosen to
                                                         # win against the
                                                         # 200 kJ/mol LJ
                                                         # repulsion
                                                         # between H-H on
                                                         # the same water)
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


# Glycine (neutral form, NH2-CH2-COOH): the simplest amino acid.
# 10 atoms. The zwitterionic NH3+/COO- form would be more biological
# but would require a formal-charge > default_valence for N, which the
# current AtomUnit valence model doesn't support cleanly.
#
# Indices and connectivity:
#   0  N   (amine nitrogen)
#   1  H   (on N)
#   2  H   (on N)
#   3  C   (alpha carbon)
#   4  H   (alpha-H)
#   5  H   (alpha-H)
#   6  C   (carbonyl C)
#   7  O   (=O, carbonyl)
#   8  O   (-O-H, hydroxyl)
#   9  H   (hydroxyl H)
_gN = (+0.147, 0.0, 0.0)
_gCa = (0.0, 0.0, 0.0)
_gC = (-0.153, 0.0, 0.0)
_gO_carbonyl = (-0.153 - 0.123, 0.104, 0.0)
_gO_hydroxyl = (-0.153 - 0.123, -0.104, 0.0)
_gOH = (_gO_hydroxyl[0] + 0.096, _gO_hydroxyl[1] - 0.05, 0.0)
# Two H's on amine N (sp3, ~109 deg)
_nh = 0.101
_gNH1 = (_gN[0] + 0.05, 0.087, 0.0)
_gNH2 = (_gN[0] + 0.05, -0.043, 0.075)
# Alpha-C hydrogens (sp3): up and down, ~0.109 nm.
_gCaH1 = (0.0, 0.089, 0.063)
_gCaH2 = (0.0, -0.089, 0.063)

GLYCINE = MoleculeTemplate(
    name="glycine", formula="C2H5NO2",
    atoms=[
        (Element.N, _gN),           # 0 amine N
        (Element.H, _gNH1),         # 1
        (Element.H, _gNH2),         # 2
        (Element.C, _gCa),          # 3 alpha-C
        (Element.H, _gCaH1),        # 4
        (Element.H, _gCaH2),        # 5
        (Element.C, _gC),           # 6 carbonyl C
        (Element.O, _gO_carbonyl),  # 7 =O
        (Element.O, _gO_hydroxyl),  # 8 -O-
        (Element.H, _gOH),          # 9 O-H
    ],
    bonds=[
        (0, 1, _S, 0.101, None),       # N-H
        (0, 2, _S, 0.101, None),
        (0, 3, _S, 0.147, None),       # N-Ca
        (3, 4, _S, 0.109, None),       # Ca-H
        (3, 5, _S, 0.109, None),
        (3, 6, _S, 0.153, None),       # Ca-C
        (6, 7, BondType.COVALENT_DOUBLE, 0.123, None),   # C=O
        (6, 8, _S, 0.134, None),       # C-O
        (8, 9, _S, 0.096, None),       # O-H hydroxyl
    ],
    partial_charges=[
        -0.35, +0.18, +0.18,           # N, 2x H on N
        +0.02, +0.05, +0.05,            # Ca, 2x H on alpha
        +0.50, -0.45, -0.50, +0.42,     # C, =O, -O-, O-H
    ],
    angles=[
        # H-N-H, H-N-Ca (sp3 at N, ~109 deg)
        (1, 0, 2, math.radians(107.0), 400.0),
        (1, 0, 3, math.radians(109.5), 400.0),
        (2, 0, 3, math.radians(109.5), 400.0),
        # sp3 at Ca
        (0, 3, 4, math.radians(109.5), 400.0),
        (0, 3, 5, math.radians(109.5), 400.0),
        (0, 3, 6, math.radians(109.5), 500.0),     # backbone N-Ca-C
        (4, 3, 5, math.radians(109.5), 400.0),
        (4, 3, 6, math.radians(109.5), 400.0),
        (5, 3, 6, math.radians(109.5), 400.0),
        # sp2 trigonal at carbonyl C
        (3, 6, 7, math.radians(120.0), 500.0),
        (3, 6, 8, math.radians(110.0), 500.0),
        (7, 6, 8, math.radians(125.0), 500.0),
        # sp3 at hydroxyl O (C-O-H)
        (6, 8, 9, math.radians(105.0), 500.0),
    ],
)


# ---- Alanine (neutral): H2N-CH(CH3)-COOH. 13 atoms.
_aNH1 = (_gN[0] + 0.05, 0.087, 0.0)
_aNH2 = (_gN[0] + 0.05, -0.043, 0.075)
_aCaH = (0.0, -0.089, 0.063)
_aCB = (0.0, 0.089, 0.120)
_aCBH1 = (0.089, 0.110, 0.170)
_aCBH2 = (-0.089, 0.110, 0.170)
_aCBH3 = (0.0, 0.020, 0.210)

ALANINE = MoleculeTemplate(
    name="alanine", formula="C3H7NO2",
    atoms=[
        (Element.N, _gN),        # 0
        (Element.H, _aNH1),      # 1
        (Element.H, _aNH2),      # 2
        (Element.C, _gCa),       # 3 alpha-C
        (Element.H, _aCaH),      # 4 alpha-H
        (Element.C, _aCB),       # 5 beta-C (methyl)
        (Element.H, _aCBH1),     # 6
        (Element.H, _aCBH2),     # 7
        (Element.H, _aCBH3),     # 8
        (Element.C, _gC),        # 9 carbonyl C
        (Element.O, _gO_carbonyl),  # 10 =O
        (Element.O, _gO_hydroxyl),  # 11 -O-
        (Element.H, _gOH),       # 12 hydroxyl H
    ],
    bonds=[
        (0, 1, _S, 0.101, None), (0, 2, _S, 0.101, None),
        (0, 3, _S, 0.147, None),
        (3, 4, _S, 0.109, None),
        (3, 5, _S, 0.154, None),
        (5, 6, _S, 0.109, None), (5, 7, _S, 0.109, None),
        (5, 8, _S, 0.109, None),
        (3, 9, _S, 0.153, None),
        (9, 10, BondType.COVALENT_DOUBLE, 0.123, None),
        (9, 11, _S, 0.134, None),
        (11, 12, _S, 0.096, None),
    ],
    partial_charges=[
        -0.35, +0.18, +0.18,
         0.02, +0.05,
         0.00, +0.03, +0.03, +0.03,
        +0.50, -0.45, -0.50, +0.42,
    ],
    angles=[
        (1, 0, 2, math.radians(107.0), 400.0),
        (1, 0, 3, math.radians(109.5), 400.0),
        (2, 0, 3, math.radians(109.5), 400.0),
        (0, 3, 4, math.radians(109.5), 400.0),
        (0, 3, 5, math.radians(109.5), 400.0),
        (0, 3, 9, math.radians(109.5), 500.0),
        (4, 3, 5, math.radians(109.5), 400.0),
        (4, 3, 9, math.radians(109.5), 400.0),
        (5, 3, 9, math.radians(109.5), 400.0),
        (3, 5, 6, math.radians(109.5), 400.0),
        (3, 5, 7, math.radians(109.5), 400.0),
        (3, 5, 8, math.radians(109.5), 400.0),
        (6, 5, 7, math.radians(109.5), 400.0),
        (6, 5, 8, math.radians(109.5), 400.0),
        (7, 5, 8, math.radians(109.5), 400.0),
        (3, 9, 10, math.radians(120.0), 500.0),
        (3, 9, 11, math.radians(110.0), 500.0),
        (10, 9, 11, math.radians(125.0), 500.0),
        (9, 11, 12, math.radians(105.0), 500.0),
    ],
)


# ---- Serine (neutral): H2N-CH(CH2OH)-COOH. 14 atoms.
_sCB = (0.0, 0.089, 0.120)
_sCBH1 = (0.089, 0.110, 0.170)
_sCBH2 = (-0.089, 0.110, 0.170)
_sOG = (0.0, 0.020, 0.250)
_sOGH = (0.060, -0.030, 0.290)

SERINE = MoleculeTemplate(
    name="serine", formula="C3H7NO3",
    atoms=[
        (Element.N, _gN),        # 0
        (Element.H, _aNH1),      # 1
        (Element.H, _aNH2),      # 2
        (Element.C, _gCa),       # 3 alpha-C
        (Element.H, _aCaH),      # 4 alpha-H
        (Element.C, _sCB),       # 5 beta-C
        (Element.H, _sCBH1),     # 6
        (Element.H, _sCBH2),     # 7
        (Element.O, _sOG),       # 8 side-chain O
        (Element.H, _sOGH),      # 9 side-chain O-H
        (Element.C, _gC),        # 10 carbonyl C
        (Element.O, _gO_carbonyl), # 11 =O
        (Element.O, _gO_hydroxyl), # 12 -O-
        (Element.H, _gOH),       # 13 carboxyl O-H
    ],
    bonds=[
        (0, 1, _S, 0.101, None), (0, 2, _S, 0.101, None),
        (0, 3, _S, 0.147, None),
        (3, 4, _S, 0.109, None),
        (3, 5, _S, 0.154, None),
        (5, 6, _S, 0.109, None), (5, 7, _S, 0.109, None),
        (5, 8, _S, 0.143, None),
        (8, 9, _S, 0.096, None),
        (3, 10, _S, 0.153, None),
        (10, 11, BondType.COVALENT_DOUBLE, 0.123, None),
        (10, 12, _S, 0.134, None),
        (12, 13, _S, 0.096, None),
    ],
    partial_charges=[
        -0.35, +0.18, +0.18,
         0.05, +0.05,
         0.05, +0.03, +0.03,
        -0.60, +0.42,
        +0.50, -0.45, -0.50, +0.42,
    ],
    angles=[
        (1, 0, 2, math.radians(107.0), 400.0),
        (1, 0, 3, math.radians(109.5), 400.0),
        (2, 0, 3, math.radians(109.5), 400.0),
        (0, 3, 4, math.radians(109.5), 400.0),
        (0, 3, 5, math.radians(109.5), 400.0),
        (0, 3, 10, math.radians(109.5), 500.0),
        (4, 3, 5, math.radians(109.5), 400.0),
        (4, 3, 10, math.radians(109.5), 400.0),
        (5, 3, 10, math.radians(109.5), 400.0),
        (3, 5, 6, math.radians(109.5), 400.0),
        (3, 5, 7, math.radians(109.5), 400.0),
        (3, 5, 8, math.radians(109.5), 400.0),
        (6, 5, 8, math.radians(109.5), 400.0),
        (7, 5, 8, math.radians(109.5), 400.0),
        (5, 8, 9, math.radians(105.0), 500.0),
        (3, 10, 11, math.radians(120.0), 500.0),
        (3, 10, 12, math.radians(110.0), 500.0),
        (11, 10, 12, math.radians(125.0), 500.0),
        (10, 12, 13, math.radians(105.0), 500.0),
    ],
)


# ---- Gly-Gly dipeptide (pre-formed peptide bond). 17 atoms.
_dCa1 = (0.0, 0.0, 0.0)
_dN1 = (0.147, 0.0, 0.0)
_dNH1a = (_dN1[0] + 0.05, 0.087, 0.0)
_dNH1b = (_dN1[0] + 0.05, -0.043, 0.075)
_dCaH1a = (0.0, 0.089, 0.063)
_dCaH1b = (0.0, -0.089, 0.063)
_dC1 = (-0.153, 0.0, 0.0)
_dO1 = (-0.153, 0.123, 0.0)
_dN2 = (-0.153 - 0.133, -0.020, 0.0)
_dN2H = (_dN2[0] - 0.05, 0.080, 0.0)
_dCa2 = (_dN2[0] - 0.147, 0.0, 0.0)
_dCaH2a = (_dCa2[0], 0.089, 0.063)
_dCaH2b = (_dCa2[0], -0.089, 0.063)
_dC2 = (_dCa2[0] - 0.153, 0.0, 0.0)
_dO2a = (_dC2[0] - 0.123, 0.104, 0.0)
_dO2b = (_dC2[0] - 0.123, -0.104, 0.0)
_dO2H = (_dO2b[0] - 0.05, _dO2b[1] - 0.05, 0.0)

GLYCYL_GLYCINE = MoleculeTemplate(
    name="glycyl_glycine", formula="C4H8N2O3",
    atoms=[
        (Element.N, _dN1),    # 0
        (Element.H, _dNH1a),  # 1
        (Element.H, _dNH1b),  # 2
        (Element.C, _dCa1),   # 3
        (Element.H, _dCaH1a), # 4
        (Element.H, _dCaH1b), # 5
        (Element.C, _dC1),    # 6 carbonyl of residue 1
        (Element.O, _dO1),    # 7
        (Element.N, _dN2),    # 8 peptide N
        (Element.H, _dN2H),   # 9 peptide H
        (Element.C, _dCa2),   # 10
        (Element.H, _dCaH2a), # 11
        (Element.H, _dCaH2b), # 12
        (Element.C, _dC2),    # 13
        (Element.O, _dO2a),   # 14
        (Element.O, _dO2b),   # 15
        (Element.H, _dO2H),   # 16
    ],
    bonds=[
        (0, 1, _S, 0.101, None), (0, 2, _S, 0.101, None),
        (0, 3, _S, 0.147, None),
        (3, 4, _S, 0.109, None), (3, 5, _S, 0.109, None),
        (3, 6, _S, 0.153, None),
        (6, 7, BondType.COVALENT_DOUBLE, 0.123, None),
        (6, 8, _S, 0.133, None),         # peptide bond
        (8, 9, _S, 0.101, None),
        (8, 10, _S, 0.147, None),
        (10, 11, _S, 0.109, None), (10, 12, _S, 0.109, None),
        (10, 13, _S, 0.153, None),
        (13, 14, BondType.COVALENT_DOUBLE, 0.123, None),
        (13, 15, _S, 0.134, None),
        (15, 16, _S, 0.096, None),
    ],
    partial_charges=[
        -0.35, +0.18, +0.18,
         0.02, +0.05, +0.05,
        +0.50, -0.50,
        -0.40, +0.25,
         0.02, +0.05, +0.05,
        +0.50, -0.45, -0.50, +0.42,
    ],
    angles=[
        (1, 0, 2, math.radians(107.0), 400.0),
        (1, 0, 3, math.radians(109.5), 400.0),
        (2, 0, 3, math.radians(109.5), 400.0),
        (0, 3, 4, math.radians(109.5), 400.0),
        (0, 3, 5, math.radians(109.5), 400.0),
        (0, 3, 6, math.radians(109.5), 500.0),
        (4, 3, 5, math.radians(109.5), 400.0),
        (4, 3, 6, math.radians(109.5), 400.0),
        (5, 3, 6, math.radians(109.5), 400.0),
        (3, 6, 7, math.radians(120.0), 500.0),
        (3, 6, 8, math.radians(116.0), 500.0),
        (7, 6, 8, math.radians(124.0), 500.0),
        (6, 8, 9, math.radians(120.0), 500.0),
        (6, 8, 10, math.radians(122.0), 500.0),
        (9, 8, 10, math.radians(118.0), 500.0),
        (8, 10, 11, math.radians(109.5), 400.0),
        (8, 10, 12, math.radians(109.5), 400.0),
        (8, 10, 13, math.radians(109.5), 500.0),
        (11, 10, 12, math.radians(109.5), 400.0),
        (11, 10, 13, math.radians(109.5), 400.0),
        (12, 10, 13, math.radians(109.5), 400.0),
        (10, 13, 14, math.radians(120.0), 500.0),
        (10, 13, 15, math.radians(110.0), 500.0),
        (14, 13, 15, math.radians(125.0), 500.0),
        (13, 15, 16, math.radians(105.0), 500.0),
    ],
    # Backbone psi (N-Ca1-C-N-peptide) and phi (C-N-peptide-Ca2-C)
    # dihedrals + the peptide omega (Ca1-C-N-Ca2) held at 180 deg
    # so the peptide bond stays planar. Standard AMBER/OPLS forms
    # with multiplicity 2 and small barriers.
    dihedrals=[
        # psi: N(0)-Ca1(3)-C1(6)-N2(8)
        (0, 3, 6, 8, 2, math.radians(0.0), 2.0),
        # omega: Ca1(3)-C1(6)-N2(8)-Ca2(10) at 180 deg, stiff
        (3, 6, 8, 10, 2, math.radians(180.0), 30.0),
        # phi: C1(6)-N2(8)-Ca2(10)-C2(13)
        (6, 8, 10, 13, 2, math.radians(0.0), 2.0),
    ],
)


LIBRARY: dict[str, MoleculeTemplate] = {
    "H2": H2, "O2": O2, "N2": N2,
    "H2O": H2O, "CH4": CH4, "NH3": NH3,
    "CO": CO, "CO2": CO2,
    "glycine": GLYCINE, "Gly": GLYCINE, "C2H5NO2": GLYCINE,
    "alanine": ALANINE, "Ala": ALANINE, "C3H7NO2": ALANINE,
    "serine": SERINE, "Ser": SERINE, "C3H7NO3": SERINE,
    "glycyl_glycine": GLYCYL_GLYCINE, "GlyGly": GLYCYL_GLYCINE,
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
) -> tuple[list[AtomUnit], list[Bond], list[AngleBond], list[DihedralBond]]:
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
    angles_out: list[AngleBond] = []
    dihedrals_out: list[DihedralBond] = []
    for idx, (template, center) in enumerate(zip(plan, centers)):
        tag = f"{template.name}#{idx}"
        R = _random_rotation_matrix(rng)
        com_vel = _mb_velocity(
            sum(mass(e) for e, _ in template.atoms),
            temperature_K, rng,
        )
        atom_objs: list[AtomUnit] = []
        for a_i, (elem, rel_pos) in enumerate(template.atoms):
            p = R @ np.array(rel_pos) + center
            v = com_vel + 0.3 * _mb_velocity(mass(elem), temperature_K, rng)
            atom = AtomUnit.create(
                element=elem,
                position=(float(p[0]), float(p[1]), float(p[2])),
                velocity=(float(v[0]), float(v[1]), float(v[2])),
                parent_molecule=tag,
            )
            # Apply template partial charge if given.
            if template.partial_charges is not None:
                atom.partial_charge = float(template.partial_charges[a_i])
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
        # Angles (3-body). Template indices are per-molecule and we
        # resolve them to the atom objects just created.
        if template.angles:
            for i, j, k_ang, theta_0, k_theta in template.angles:
                angles_out.append(AngleBond(
                    i=atom_objs[i], j=atom_objs[j], k=atom_objs[k_ang],
                    theta_0_rad=float(theta_0),
                    k_theta_kj_per_mol_rad2=float(k_theta),
                ))
        if template.dihedrals:
            for i, j, k_dh, l, n, phi_0, k_phi in template.dihedrals:
                dihedrals_out.append(DihedralBond(
                    i=atom_objs[i], j=atom_objs[j],
                    k=atom_objs[k_dh], l=atom_objs[l],
                    n=int(n), phi_0_rad=float(phi_0),
                    k_phi_kj_per_mol=float(k_phi),
                ))

    _zero_net_momentum(atoms)
    return atoms, bonds, angles_out, dihedrals_out


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
