"""PDB importer for the atom engine.

Reads standard PDB ATOM / HETATM records and produces the same
(atoms, bonds, angles, dihedrals) tuple that :func:`build_mixture`
returns. Heuristic bonding + angle generation means no external
residue database is required.

Rules applied:
  - Bonds: any two atoms closer than their (element-pair-specific
    equilibrium length) * 1.25 are bonded. Uses
    ``molecule_builder._BOND_LENGTH_NM`` for the equilibria.
  - Angles: for every A-B-C where both (A, B) and (B, C) are bonds,
    emit an AngleBond at the standard equilibrium for B's hybridisation
    (sp3 = 109.5, sp2 trigonal = 120, sp = 180). Hybridisation is
    guessed from the heavy-atom degree of B and the presence of any
    double/triple bond on B (we currently only know "single" unless a
    CONECT record tagged otherwise, so everything defaults to sp3 on
    C/N and bent/tetrahedral on O — a first-pass heuristic, not a
    production force-field setup).
  - Partial charges: coarse assignment by element + local environment
    (amide N, hydroxyl O, carboxyl O, etc.) — sufficient for a
    Coulomb-aware demo but not a production charge model.
  - Dihedrals: skipped in this first pass. The angle + bond stack
    alone is enough to hold peptide geometry for short runs.

Honest caveat: this is an MD-toy importer, not a biomolecule-quality
one. For published-grade simulations use a parameter file (AMBER
prmtop, CHARMM psf, GROMACS top) via a real importer. Our output is
"geometry + rough forces to let MD evolve the system", which is all
the atom engine can honestly do with classical LJ+harmonic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .atom_unit import AngleBond, AtomUnit, Bond, BondType, DihedralBond
from .element import Element, mass
from .molecule_builder import _BOND_LENGTH_NM, _DEFAULT_BOND_K

_K_B_KJ_PER_MOL_K = 0.00831446


# Partial charge heuristic: (element, neighbor_elements_multiset) -> charge.
# Falls back to element-default if no rule matches.
def _charge_heuristic(element: Element, neighbors: list[Element],
                      residue: str, atom_name: str) -> float:
    name = atom_name.strip().upper()
    res = residue.strip().upper()
    # Backbone amide N
    if element is Element.N and any(n is Element.C for n in neighbors) \
            and any(n is Element.H for n in neighbors):
        return -0.35 if name in ("N",) else -0.30
    # Backbone amide H
    if element is Element.H and len(neighbors) == 1 \
            and neighbors[0] is Element.N:
        return +0.20
    # Hydroxyl O
    if element is Element.O and any(n is Element.H for n in neighbors) \
            and any(n is Element.C for n in neighbors):
        return -0.60
    # Hydroxyl H
    if element is Element.H and len(neighbors) == 1 \
            and neighbors[0] is Element.O:
        return +0.40
    # Carbonyl =O (only one heavy neighbor, which is C)
    if element is Element.O and len(neighbors) == 1 \
            and neighbors[0] is Element.C:
        return -0.50
    # Carboxyl O (-O- with C and H would hit hydroxyl first; bare -O-)
    if element is Element.O and len(neighbors) == 1 \
            and neighbors[0] is Element.C:
        return -0.50
    # Alpha carbon (bonded to N and to C of carbonyl)
    if element is Element.C:
        n_types = sorted(n.name for n in neighbors)
        if "N" in n_types and n_types.count("C") >= 1:
            return +0.05
        return 0.0
    if element is Element.H:
        return +0.05
    if element is Element.N:
        return -0.30
    if element is Element.O:
        return -0.40
    return 0.0


@dataclass
class ImportedStructure:
    atoms: list[AtomUnit]
    bonds: list[Bond]
    angles: list[AngleBond]
    dihedrals: list[DihedralBond]


def _parse_element(atom_name: str, element_field: str) -> Element:
    """Try the PDB element column first, fall back to the atom-name
    first character. PDB atom names are right-aligned on columns 13-16
    for heavy atoms; the element goes in 77-78."""
    e = element_field.strip().upper()
    if e:
        try:
            return Element[e.capitalize() if len(e) == 2 else e]
        except KeyError:
            pass
    # Fall back to atom-name first letter.
    first = atom_name.strip().upper()[:1]
    try:
        return Element[first]
    except KeyError:
        return Element.C


def load_pdb(
    path: str | Path,
    *,
    temperature_K: float = 300.0,
    parent_molecule: Optional[str] = None,
    bond_scale_factor: float = 1.25,
    bond_k_kj_per_nm2: float = 3.0e5,
    default_angle_k_kj_per_mol_rad2: float = 400.0,
    seed: int = 42,
) -> ImportedStructure:
    """Load a PDB file and return an ImportedStructure.

    Bonds are inferred by proximity: atom pair (i, j) is bonded iff
    ||r_i - r_j|| < bond_scale_factor * r0(element_i, element_j) with
    r0 taken from ``molecule_builder._BOND_LENGTH_NM`` (or 0.17 nm for
    unknown pairs).

    Angles are auto-generated: for every heavy atom B with neighbors
    A and C, emit an AngleBond at 109.5 deg (sp3) or 120 deg (when B
    has >= 3 heavy neighbors).
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    text = path.read_text()

    # Parse ATOM / HETATM.
    raw_atoms: list[dict] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # Columns per the PDB specification (fixed-width).
        atom_name = line[12:16]
        resname = line[17:20]
        try:
            x = float(line[30:38]) * 0.1   # PDB angstrom -> nm
            y = float(line[38:46]) * 0.1
            z = float(line[46:54]) * 0.1
        except ValueError:
            continue
        element_field = line[76:78] if len(line) >= 78 else ""
        elem = _parse_element(atom_name, element_field)
        raw_atoms.append({
            "atom_name": atom_name,
            "resname": resname,
            "pos": (x, y, z),
            "element": elem,
        })

    if not raw_atoms:
        raise ValueError(f"no ATOM/HETATM records in {path}")

    # Create AtomUnit objects.
    atom_objs: list[AtomUnit] = []
    for i, r in enumerate(raw_atoms):
        tag = parent_molecule or f"{r['resname'].strip()}#{i}"
        sigma_v = math.sqrt(_K_B_KJ_PER_MOL_K * temperature_K /
                            mass(r["element"]))
        v = tuple(rng.normal(0.0, sigma_v, size=3))
        atom = AtomUnit.create(
            element=r["element"],
            position=r["pos"],
            velocity=v,
            parent_molecule=tag,
        )
        atom_objs.append(atom)

    # Bond inference by proximity.
    pos_arr = np.array([a.position for a in atom_objs], dtype=np.float64)
    n = len(atom_objs)
    bonds: list[Bond] = []
    bonded_nbrs: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        ai = atom_objs[i]
        for j in range(i + 1, n):
            aj = atom_objs[j]
            key = frozenset([ai.element, aj.element])
            r0 = _BOND_LENGTH_NM.get(key, 0.15)
            d = pos_arr[j] - pos_arr[i]
            if d @ d < (r0 * bond_scale_factor) ** 2:
                # Check valence remaining on both.
                if ai.valence_remaining <= 0 or aj.valence_remaining <= 0:
                    continue
                try:
                    b = ai.form_bond(
                        aj, kind=BondType.COVALENT_SINGLE, t_ps=0.0,
                        equilibrium_length_nm=r0,
                        spring_constant_kj_per_nm2=bond_k_kj_per_nm2,
                    )
                    bonds.append(b)
                    bonded_nbrs[i].append(j)
                    bonded_nbrs[j].append(i)
                except ValueError:
                    pass

    # Assign partial charges heuristically.
    for i, atom in enumerate(atom_objs):
        nbrs = [atom_objs[j].element for j in bonded_nbrs[i]]
        atom.partial_charge = _charge_heuristic(
            atom.element, nbrs, raw_atoms[i]["resname"],
            raw_atoms[i]["atom_name"],
        )

    # Auto-angle generation: for every B with >= 2 bonded neighbours
    # (A, C), emit A-B-C.
    angles: list[AngleBond] = []
    for b in range(n):
        nbrs = bonded_nbrs[b]
        if len(nbrs) < 2:
            continue
        # Choose equilibrium angle based on center element + degree.
        center = atom_objs[b].element
        deg = len(nbrs)
        if deg >= 4:
            theta_0 = math.radians(109.5)       # sp3 tetrahedral
        elif deg == 3:
            theta_0 = math.radians(120.0)       # sp2 trigonal
        elif deg == 2 and center is Element.O:
            theta_0 = math.radians(104.5)       # bent (water, ether)
        elif deg == 2 and center is Element.N:
            theta_0 = math.radians(120.0)       # sp2 (amide) or
                                                  # can be sp3 depending on
                                                  # environment; 120 is a
                                                  # decent compromise.
        elif deg == 2 and center is Element.C:
            theta_0 = math.radians(180.0)       # sp linear (rare; CO2)
        else:
            theta_0 = math.radians(109.5)
        for ai_idx in range(len(nbrs)):
            for aj_idx in range(ai_idx + 1, len(nbrs)):
                a_atom = atom_objs[nbrs[ai_idx]]
                c_atom = atom_objs[nbrs[aj_idx]]
                angles.append(AngleBond(
                    i=a_atom, j=atom_objs[b], k=c_atom,
                    theta_0_rad=float(theta_0),
                    k_theta_kj_per_mol_rad2=default_angle_k_kj_per_mol_rad2,
                ))

    return ImportedStructure(
        atoms=atom_objs,
        bonds=bonds,
        angles=angles,
        dihedrals=[],
    )


# ---------- mini-PDB strings for quick standard residues ------------
# Programmatic fallback when no external PDB file is available. Each
# entry uses PDB-style ATOM records so the importer processes them
# through exactly the same path. Geometries from ideal peptide
# backbone + extended side chains (not minimized; MD will relax).

STANDARD_RESIDUES_PDB: dict[str, str] = {
    # Glycine (neutral form)
    "GLY": """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  GLY A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  GLY A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  GLY A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA1 GLY A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  HA2 GLY A   1       1.821   0.509  -0.895  1.00  0.00           H
ATOM      7  C   GLY A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM      8  O   GLY A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM      9  OXT GLY A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     10  HXT GLY A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Alanine
    "ALA": """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ALA A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  ALA A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  ALA A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  ALA A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 ALA A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 ALA A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  HB3 ALA A   1       3.077   0.770  -1.218  1.00  0.00           H
ATOM     10  C   ALA A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     11  O   ALA A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     12  OXT ALA A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     13  HXT ALA A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Water (TIP3P-like geometry in a PDB ATOM block)
    "HOH": """\
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.076   0.059   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.019   0.094   0.000  1.00  0.00           H
END
""",
    # Serine: Ala + beta-hydroxyl
    "SER": """\
ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  SER A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  SER A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  SER A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  SER A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  SER A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 SER A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 SER A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  OG  SER A   1       3.417   0.820  -1.220  1.00  0.00           O
ATOM     10  HG  SER A   1       3.720   1.730  -1.220  1.00  0.00           H
ATOM     11  C   SER A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     12  O   SER A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     13  OXT SER A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     14  HXT SER A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Valine (branched aliphatic)
    "VAL": """\
ATOM      1  N   VAL A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  VAL A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  VAL A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  VAL A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  VAL A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  VAL A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB  VAL A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  CG1 VAL A   1       1.493   0.066  -2.490  1.00  0.00           C
ATOM      9  HG11 VAL A   1       1.877   0.593  -3.367  1.00  0.00           H
ATOM     10  HG12 VAL A   1       1.877  -0.958  -2.490  1.00  0.00           H
ATOM     11  HG13 VAL A   1       0.403   0.066  -2.490  1.00  0.00           H
ATOM     12  CG2 VAL A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     13  HG21 VAL A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     14  HG22 VAL A   1       3.887   0.258  -2.107  1.00  0.00           H
ATOM     15  HG23 VAL A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     16  C   VAL A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     17  O   VAL A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     18  OXT VAL A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     19  HXT VAL A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Leucine
    "LEU": """\
ATOM      1  N   LEU A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  LEU A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  LEU A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  LEU A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  LEU A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  LEU A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 LEU A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 LEU A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  LEU A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG  LEU A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  CD1 LEU A   1       4.047   0.066  -2.490  1.00  0.00           C
ATOM     12  HD11 LEU A   1       4.431   0.593  -3.367  1.00  0.00           H
ATOM     13  HD12 LEU A   1       4.431  -0.958  -2.490  1.00  0.00           H
ATOM     14  HD13 LEU A   1       2.957   0.066  -2.490  1.00  0.00           H
ATOM     15  CD2 LEU A   1       4.047   0.066   0.054  1.00  0.00           C
ATOM     16  HD21 LEU A   1       4.431   0.593   0.931  1.00  0.00           H
ATOM     17  HD22 LEU A   1       4.431  -0.958   0.054  1.00  0.00           H
ATOM     18  HD23 LEU A   1       5.137   0.066   0.054  1.00  0.00           H
ATOM     19  C   LEU A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     20  O   LEU A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     21  OXT LEU A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     22  HXT LEU A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Threonine: Ala + beta-hydroxyl + beta-methyl
    "THR": """\
ATOM      1  N   THR A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  THR A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  THR A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  THR A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  THR A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  THR A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB  THR A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  OG1 THR A   1       3.417   0.820  -1.220  1.00  0.00           O
ATOM      9  HG1 THR A   1       3.720   1.730  -1.220  1.00  0.00           H
ATOM     10  CG2 THR A   1       1.493   0.066  -2.490  1.00  0.00           C
ATOM     11  HG21 THR A   1       1.877   0.593  -3.367  1.00  0.00           H
ATOM     12  HG22 THR A   1       1.877  -0.958  -2.490  1.00  0.00           H
ATOM     13  HG23 THR A   1       0.403   0.066  -2.490  1.00  0.00           H
ATOM     14  C   THR A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     15  O   THR A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     16  OXT THR A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     17  HXT THR A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Cysteine (S-H)
    "CYS": """\
ATOM      1  N   CYS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  CYS A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  CYS A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  CYS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  CYS A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  CYS A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 CYS A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 CYS A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  SG  CYS A   1       3.817   0.820  -1.220  1.00  0.00           S
ATOM     10  HG  CYS A   1       4.170   1.950  -1.220  1.00  0.00           H
ATOM     11  C   CYS A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     12  O   CYS A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     13  OXT CYS A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     14  HXT CYS A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Methionine (C-C-C-S-C thioether)
    "MET": """\
ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  MET A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  MET A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  MET A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  MET A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  MET A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 MET A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 MET A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  MET A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG1 MET A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  HG2 MET A   1       3.887   0.258  -2.107  1.00  0.00           H
ATOM     12  SD  MET A   1       4.247   0.000   0.260  1.00  0.00           S
ATOM     13  CE  MET A   1       6.077   0.000   0.000  1.00  0.00           C
ATOM     14  HE1 MET A   1       6.440   0.509   0.895  1.00  0.00           H
ATOM     15  HE2 MET A   1       6.440   0.509  -0.895  1.00  0.00           H
ATOM     16  HE3 MET A   1       6.440  -1.024   0.000  1.00  0.00           H
ATOM     17  C   MET A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     18  O   MET A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     19  OXT MET A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     20  HXT MET A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Asparagine: CH2-C(=O)-NH2 side chain
    "ASN": """\
ATOM      1  N   ASN A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ASN A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  ASN A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  ASN A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  ASN A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  ASN A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 ASN A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 ASN A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  ASN A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  OD1 ASN A   1       4.147   1.800  -1.218  1.00  0.00           O
ATOM     11  ND2 ASN A   1       4.217  -0.400  -1.218  1.00  0.00           N
ATOM     12  HD21 ASN A   1       5.227  -0.400  -1.218  1.00  0.00           H
ATOM     13  HD22 ASN A   1       3.677  -1.250  -1.218  1.00  0.00           H
ATOM     14  C   ASN A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     15  O   ASN A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     16  OXT ASN A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     17  HXT ASN A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Aspartate (neutral COOH form — simplest protonation)
    "ASP": """\
ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ASP A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  ASP A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  ASP A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  ASP A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  ASP A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 ASP A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 ASP A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  ASP A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  OD1 ASP A   1       4.147   1.800  -1.218  1.00  0.00           O
ATOM     11  OD2 ASP A   1       4.117  -0.400  -1.218  1.00  0.00           O
ATOM     12  HD2 ASP A   1       5.107  -0.400  -1.218  1.00  0.00           H
ATOM     13  C   ASP A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     14  O   ASP A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     15  OXT ASP A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     16  HXT ASP A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Note on nucleotides (A/T/G/C/U): aromatic ring geometry needs
    # careful coordinate tabulation (planar 5/6-membered fused rings
    # with bond lengths ~0.14 nm). Out of scope for this first
    # importer pass; future work is to parse a real PDB file of a
    # solved nucleotide (e.g. the RCSB ATP ligand record) rather
    # than hand-coding.
}


def load_residue(
    resname: str,
    *,
    temperature_K: float = 300.0,
    parent_molecule: Optional[str] = None,
) -> ImportedStructure:
    """Shortcut: load a small standard residue from the built-in PDB
    strings. Usage:  load_residue('ALA')."""
    key = resname.upper()
    if key not in STANDARD_RESIDUES_PDB:
        raise KeyError(
            f"no standard residue PDB for '{resname}'. Available: "
            f"{sorted(STANDARD_RESIDUES_PDB.keys())}"
        )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
        fh.write(STANDARD_RESIDUES_PDB[key])
        tmp = fh.name
    try:
        return load_pdb(tmp, temperature_K=temperature_K,
                        parent_molecule=parent_molecule)
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass
