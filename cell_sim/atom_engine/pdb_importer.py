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
    # The strict PDB column layout (atom name at cols 13-16, coords at
    # 31-38/39-46/47-54) breaks when hand-written templates misalign a
    # 4-character atom name (HD21, HE21, HG11, ...) by one column,
    # which shifts the coord field and causes float() to fail silently.
    # We fall back to whitespace-split parsing: for well-formed ATOM
    # records (our templates + most real PDB files) the tokens are
    # [ATOM, serial, atom_name, resname, chain, resseq, x, y, z,
    #  occupancy, tempfactor, element?]. That parse is robust to the
    # column-alignment bug without giving up on real PDB files.
    raw_atoms: list[dict] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # Try strict fixed-width first; fall back to split-based on any
        # parse error so hand-written templates work.
        atom_name = line[12:16] if len(line) >= 16 else ""
        resname = line[17:20] if len(line) >= 20 else ""
        x = y = z = None
        try:
            x = float(line[30:38]) * 0.1
            y = float(line[38:46]) * 0.1
            z = float(line[46:54]) * 0.1
        except (ValueError, IndexError):
            pass
        element_field = line[76:78] if len(line) >= 78 else ""

        if x is None or y is None or z is None:
            # Whitespace-split fallback.
            toks = line.split()
            # toks: ATOM, serial, name, resname, chain?, resseq, x, y, z, occ, temp, [element]
            if len(toks) < 9:
                continue
            try:
                # x,y,z are the 3rd-to-last coord triple before occupancy
                # and tempfactor. Walk back from the end.
                x = float(toks[-6]) * 0.1
                y = float(toks[-5]) * 0.1
                z = float(toks[-4]) * 0.1
            except (ValueError, IndexError):
                continue
            atom_name = toks[2]
            resname = toks[3]
            if len(toks) >= 12 and all(c.isalpha() for c in toks[-1]):
                element_field = toks[-1]
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
    # Water (TIP3P-like geometry in a PDB ATOM block).
    # PDB coords are in Angstroms; |O-H| = 0.96 A, HOH angle = 104.5 deg.
    # H1, H2 placed symmetrically so |H1-H2| = 2*0.96*sin(52.25°) = 1.518 A.
    "HOH": """\
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.759   0.588   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.759   0.588   0.000  1.00  0.00           H
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
    # Glutamate (neutral COOH form) — ASP + extra CH2 in side chain.
    "GLU": """\
ATOM      1  N   GLU A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  GLU A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  GLU A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  GLU A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  GLU A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  GLU A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 GLU A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 GLU A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  GLU A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG1 GLU A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  HG2 GLU A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     12  CD  GLU A   1       4.047   0.000  -2.436  1.00  0.00           C
ATOM     13  OE1 GLU A   1       3.247  -0.938  -2.436  1.00  0.00           O
ATOM     14  OE2 GLU A   1       5.327  -0.208  -2.436  1.00  0.00           O
ATOM     15  HE2 GLU A   1       5.636  -1.129  -2.436  1.00  0.00           H
ATOM     16  C   GLU A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     17  O   GLU A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     18  OXT GLU A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     19  HXT GLU A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Glutamine — ASN + extra CH2 in side chain (same shape as GLU
    # but with NH2 in place of OH).
    "GLN": """\
ATOM      1  N   GLN A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  GLN A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  GLN A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  GLN A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  GLN A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  GLN A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 GLN A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 GLN A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  GLN A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG1 GLN A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  HG2 GLN A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     12  CD  GLN A   1       4.047   0.000  -2.436  1.00  0.00           C
ATOM     13  OE1 GLN A   1       3.247  -0.938  -2.436  1.00  0.00           O
ATOM     14  NE2 GLN A   1       5.397  -0.208  -2.436  1.00  0.00           N
ATOM     15  HE21 GLN A   1       5.696  -1.158  -2.436  1.00  0.00           H
ATOM     16  HE22 GLN A   1       6.097   0.510  -2.436  1.00  0.00           H
ATOM     17  C   GLN A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     18  O   GLN A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     19  OXT GLN A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     20  HXT GLN A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Lysine (neutral NH2 form — simpler than the +1 protonated NH3+).
    # Side chain: CB-CG-CD-CE-NZ (H2), 4 carbons then primary amine.
    "LYS": """\
ATOM      1  N   LYS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  LYS A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  LYS A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  LYS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  LYS A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  LYS A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 LYS A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 LYS A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  LYS A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG1 LYS A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  HG2 LYS A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     12  CD  LYS A   1       4.047   0.000  -2.436  1.00  0.00           C
ATOM     13  HD1 LYS A   1       3.677  -1.024  -2.436  1.00  0.00           H
ATOM     14  HD2 LYS A   1       3.677   0.512  -3.325  1.00  0.00           H
ATOM     15  CE  LYS A   1       5.577   0.000  -2.436  1.00  0.00           C
ATOM     16  HE1 LYS A   1       5.947   1.024  -2.436  1.00  0.00           H
ATOM     17  HE2 LYS A   1       5.947  -0.512  -1.547  1.00  0.00           H
ATOM     18  NZ  LYS A   1       6.107  -0.770  -3.654  1.00  0.00           N
ATOM     19  HZ1 LYS A   1       5.737  -1.694  -3.654  1.00  0.00           H
ATOM     20  HZ2 LYS A   1       7.117  -0.770  -3.654  1.00  0.00           H
ATOM     21  C   LYS A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     22  O   LYS A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     23  OXT LYS A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     24  HXT LYS A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Arginine (neutral form — NH rather than NH2+). Side chain
    # CB-CG-CD-NE-CZ(=NH1)(-NH2). Coordinates place the guanidinium
    # roughly planar around CZ.
    "ARG": """\
ATOM      1  N   ARG A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ARG A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  ARG A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  ARG A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  ARG A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  ARG A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 ARG A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 ARG A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  ARG A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  HG1 ARG A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     11  HG2 ARG A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     12  CD  ARG A   1       4.047   0.000  -2.436  1.00  0.00           C
ATOM     13  HD1 ARG A   1       3.677  -1.024  -2.436  1.00  0.00           H
ATOM     14  HD2 ARG A   1       3.677   0.512  -3.325  1.00  0.00           H
ATOM     15  NE  ARG A   1       5.547   0.000  -2.436  1.00  0.00           N
ATOM     16  HE  ARG A   1       5.997  -0.896  -2.436  1.00  0.00           H
ATOM     17  CZ  ARG A   1       6.287   1.100  -2.436  1.00  0.00           C
ATOM     18 NH1  ARG A   1       5.697   2.287  -2.436  1.00  0.00           N
ATOM     19 HH11 ARG A   1       4.697   2.287  -2.436  1.00  0.00           H
ATOM     20 HH12 ARG A   1       6.187   3.167  -2.436  1.00  0.00           H
ATOM     21 NH2  ARG A   1       7.617   1.100  -2.436  1.00  0.00           N
ATOM     22 HH21 ARG A   1       8.127   1.970  -2.436  1.00  0.00           H
ATOM     23 HH22 ARG A   1       8.127   0.230  -2.436  1.00  0.00           H
ATOM     24  C   ARG A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     25  O   ARG A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     26  OXT ARG A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     27  HXT ARG A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Histidine (HIS-E tautomer: H on NE2). Imidazole ring:
    # CG-ND1-CE1-NE2-CD2 back to CG. Ring is planar (z all equal);
    # C-N / C-C bonds ~1.34 A.
    "HIS": """\
ATOM      1  N   HIS A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  HIS A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  HIS A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  HIS A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  HIS A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  HIS A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 HIS A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 HIS A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  HIS A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  ND1 HIS A   1       4.587   1.550  -1.218  1.00  0.00           N
ATOM     11  CE1 HIS A   1       5.887   0.950  -1.218  1.00  0.00           C
ATOM     12  HE1 HIS A   1       6.767   1.580  -1.218  1.00  0.00           H
ATOM     13  NE2 HIS A   1       5.697  -0.430  -1.218  1.00  0.00           N
ATOM     14  HE2 HIS A   1       6.377  -1.180  -1.218  1.00  0.00           H
ATOM     15  CD2 HIS A   1       4.367  -0.470  -1.218  1.00  0.00           C
ATOM     16  HD2 HIS A   1       3.887  -1.440  -1.218  1.00  0.00           H
ATOM     17  C   HIS A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     18  O   HIS A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     19  OXT HIS A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     20  HXT HIS A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Phenylalanine. Benzene ring side chain: CG-CD1-CE1-CZ-CE2-CD2.
    # Regular hexagon, C-C ~1.40 A, C-H 1.09 A. Ring planar in xy plane.
    "PHE": """\
ATOM      1  N   PHE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  PHE A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  PHE A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  PHE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  PHE A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  PHE A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 PHE A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 PHE A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  PHE A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  CD1 PHE A   1       4.217   1.970  -1.218  1.00  0.00           C
ATOM     11  HD1 PHE A   1       3.657   2.899  -1.218  1.00  0.00           H
ATOM     12  CE1 PHE A   1       5.617   1.970  -1.218  1.00  0.00           C
ATOM     13  HE1 PHE A   1       6.167   2.900  -1.218  1.00  0.00           H
ATOM     14  CZ  PHE A   1       6.317   0.770  -1.218  1.00  0.00           C
ATOM     15  HZ  PHE A   1       7.417   0.770  -1.218  1.00  0.00           H
ATOM     16  CE2 PHE A   1       5.617  -0.430  -1.218  1.00  0.00           C
ATOM     17  HE2 PHE A   1       6.167  -1.360  -1.218  1.00  0.00           H
ATOM     18  CD2 PHE A   1       4.217  -0.430  -1.218  1.00  0.00           C
ATOM     19  HD2 PHE A   1       3.657  -1.360  -1.218  1.00  0.00           H
ATOM     20  C   PHE A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     21  O   PHE A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     22  OXT PHE A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     23  HXT PHE A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Tyrosine. PHE + hydroxyl on CZ. Same hexagon, replace HZ with OH.
    "TYR": """\
ATOM      1  N   TYR A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  TYR A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  TYR A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  TYR A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  TYR A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  TYR A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 TYR A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 TYR A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  TYR A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  CD1 TYR A   1       4.217   1.970  -1.218  1.00  0.00           C
ATOM     11  HD1 TYR A   1       3.657   2.899  -1.218  1.00  0.00           H
ATOM     12  CE1 TYR A   1       5.617   1.970  -1.218  1.00  0.00           C
ATOM     13  HE1 TYR A   1       6.167   2.900  -1.218  1.00  0.00           H
ATOM     14  CZ  TYR A   1       6.317   0.770  -1.218  1.00  0.00           C
ATOM     15  OH  TYR A   1       7.677   0.770  -1.218  1.00  0.00           O
ATOM     16  HH  TYR A   1       7.977   1.668  -1.218  1.00  0.00           H
ATOM     17  CE2 TYR A   1       5.617  -0.430  -1.218  1.00  0.00           C
ATOM     18  HE2 TYR A   1       6.167  -1.360  -1.218  1.00  0.00           H
ATOM     19  CD2 TYR A   1       4.217  -0.430  -1.218  1.00  0.00           C
ATOM     20  HD2 TYR A   1       3.657  -1.360  -1.218  1.00  0.00           H
ATOM     21  C   TYR A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     22  O   TYR A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     23  OXT TYR A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     24  HXT TYR A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Isoleucine. Beta-branched with both a methyl (CG2) and ethyl
    # (CG1-CD1) extension from CB.
    "ILE": """\
ATOM      1  N   ILE A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  ILE A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  ILE A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  ILE A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  ILE A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  ILE A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB  ILE A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  CG1 ILE A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM      9  HG11 ILE A   1       3.887   1.794  -1.218  1.00  0.00           H
ATOM     10  HG12 ILE A   1       3.887   0.258  -0.329  1.00  0.00           H
ATOM     11  CD1 ILE A   1       4.047   0.000  -2.436  1.00  0.00           C
ATOM     12  HD11 ILE A   1       5.137   0.000  -2.436  1.00  0.00           H
ATOM     13  HD12 ILE A   1       3.677  -1.024  -2.436  1.00  0.00           H
ATOM     14  HD13 ILE A   1       3.677   0.512  -3.325  1.00  0.00           H
ATOM     15  CG2 ILE A   1       1.487   0.000  -2.550  1.00  0.00           C
ATOM     16  HG21 ILE A   1       1.857   1.024  -2.550  1.00  0.00           H
ATOM     17  HG22 ILE A   1       0.397   0.000  -2.550  1.00  0.00           H
ATOM     18  HG23 ILE A   1       1.857  -0.512  -3.439  1.00  0.00           H
ATOM     19  C   ILE A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     20  O   ILE A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     21  OXT ILE A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     22  HXT ILE A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Tryptophan. Indole = benzene fused to pyrrole. Bicyclic planar.
    # Pyrrole: CG-CD1-NE1-CE2-CD2-CG. Benzene: CD2-CE2-CZ2-CH2-CZ3-CE3-CD2.
    # Shared edge is CD2-CE2.
    "TRP": """\
ATOM      1  N   TRP A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H1  TRP A   1      -0.340   0.940   0.000  1.00  0.00           H
ATOM      3  H2  TRP A   1      -0.340  -0.470   0.814  1.00  0.00           H
ATOM      4  CA  TRP A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      5  HA  TRP A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      6  CB  TRP A   1       1.987   0.770  -1.218  1.00  0.00           C
ATOM      7  HB1 TRP A   1       1.617   1.794  -1.218  1.00  0.00           H
ATOM      8  HB2 TRP A   1       1.617   0.258  -2.107  1.00  0.00           H
ATOM      9  CG  TRP A   1       3.517   0.770  -1.218  1.00  0.00           C
ATOM     10  CD1 TRP A   1       4.407   1.770  -1.218  1.00  0.00           C
ATOM     11  HD1 TRP A   1       4.167   2.830  -1.218  1.00  0.00           H
ATOM     12  NE1 TRP A   1       5.717   1.230  -1.218  1.00  0.00           N
ATOM     13  HE1 TRP A   1       6.527   1.820  -1.218  1.00  0.00           H
ATOM     14  CE2 TRP A   1       5.667  -0.150  -1.218  1.00  0.00           C
ATOM     15  CD2 TRP A   1       4.337  -0.470  -1.218  1.00  0.00           C
ATOM     16  CE3 TRP A   1       3.897  -1.820  -1.218  1.00  0.00           C
ATOM     17  HE3 TRP A   1       2.827  -2.000  -1.218  1.00  0.00           H
ATOM     18  CZ3 TRP A   1       4.847  -2.830  -1.218  1.00  0.00           C
ATOM     19  HZ3 TRP A   1       4.527  -3.873  -1.218  1.00  0.00           H
ATOM     20  CH2 TRP A   1       6.217  -2.500  -1.218  1.00  0.00           C
ATOM     21  HH2 TRP A   1       6.947  -3.310  -1.218  1.00  0.00           H
ATOM     22  CZ2 TRP A   1       6.637  -1.170  -1.218  1.00  0.00           C
ATOM     23  HZ2 TRP A   1       7.697  -0.940  -1.218  1.00  0.00           H
ATOM     24  C   TRP A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     25  O   TRP A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     26  OXT TRP A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     27  HXT TRP A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # Proline. 5-ring closes from N through CA-CB-CG-CD back to N.
    # Secondary amine: only ONE H on the ring N. Uses pyrrolidine
    # geometry with ~75 deg bond angles forcing ring closure.
    "PRO": """\
ATOM      1  N   PRO A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H   PRO A   1      -0.870   0.500   0.000  1.00  0.00           H
ATOM      3  CA  PRO A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      4  HA  PRO A   1       1.821   0.509   0.895  1.00  0.00           H
ATOM      5  CB  PRO A   1       1.987   1.414   0.000  1.00  0.00           C
ATOM      6  HB1 PRO A   1       2.780   1.560   0.740  1.00  0.00           H
ATOM      7  HB2 PRO A   1       2.447   1.560  -0.980  1.00  0.00           H
ATOM      8  CG  PRO A   1       0.930   2.500   0.000  1.00  0.00           C
ATOM      9  HG1 PRO A   1       1.280   3.290   0.670  1.00  0.00           H
ATOM     10  HG2 PRO A   1       0.830   2.930  -1.000  1.00  0.00           H
ATOM     11  CD  PRO A   1      -0.320   1.700   0.500  1.00  0.00           C
ATOM     12  HD1 PRO A   1      -1.100   2.350   0.900  1.00  0.00           H
ATOM     13  HD2 PRO A   1      -0.690   1.080   1.320  1.00  0.00           H
ATOM     14  C   PRO A   1       1.988  -1.418   0.000  1.00  0.00           C
ATOM     15  O   PRO A   1       1.188  -2.356   0.000  1.00  0.00           O
ATOM     16  OXT PRO A   1       3.268  -1.626   0.000  1.00  0.00           O
ATOM     17  HXT PRO A   1       3.577  -2.547   0.000  1.00  0.00           H
END
""",
    # --- Nucleobases (free bases, no sugar/phosphate) ---
    # Purines: bicyclic, imidazole fused to pyrimidine ring.
    # Pyrimidines: monocyclic 6-membered ring.
    # All planar in the xy plane. Standard-PDB free-base tautomers.

    # Adenine. Purine with NH2 at C6. Imidazole (N9-C8-N7-C5-C4) fused
    # with pyrimidine (N1-C2-N3-C4-C5-C6) at the C4-C5 edge. Planar.
    "ADE": """\
HETATM    1  N9  ADE A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  H9  ADE A   1      -0.710  -0.720   0.000  1.00  0.00           H
HETATM    3  C8  ADE A   1       1.080  -0.680   0.000  1.00  0.00           C
HETATM    4  H8  ADE A   1       1.140  -1.760   0.000  1.00  0.00           H
HETATM    5  N7  ADE A   1       2.300  -0.100   0.000  1.00  0.00           N
HETATM    6  C5  ADE A   1       2.000   1.300   0.000  1.00  0.00           C
HETATM    7  C4  ADE A   1       0.620   1.260   0.000  1.00  0.00           C
HETATM    8  N3  ADE A   1      -0.380   2.160   0.000  1.00  0.00           N
HETATM    9  C2  ADE A   1       0.060   3.460   0.000  1.00  0.00           C
HETATM   10  H2  ADE A   1      -0.760   4.170   0.000  1.00  0.00           H
HETATM   11  N1  ADE A   1       1.400   3.500   0.000  1.00  0.00           N
HETATM   12  C6  ADE A   1       2.420   2.580   0.000  1.00  0.00           C
HETATM   13  N6  ADE A   1       3.720   2.620   0.000  1.00  0.00           N
HETATM   14  HN61 ADE A   1       4.230   3.490   0.000  1.00  0.00           H
HETATM   15  HN62 ADE A   1       4.230   1.750   0.000  1.00  0.00           H
END
""",

    # Guanine. Purine with =O at C6 and NH2 at C2.
    "GUA": """\
HETATM    1  N9  GUA A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  H9  GUA A   1      -0.040  -1.000   0.000  1.00  0.00           H
HETATM    3  C8  GUA A   1       1.290   0.100   0.000  1.00  0.00           C
HETATM    4  H8  GUA A   1       1.930  -0.770   0.000  1.00  0.00           H
HETATM    5  N7  GUA A   1       1.780   1.300   0.000  1.00  0.00           N
HETATM    6  C5  GUA A   1       0.770   2.200   0.000  1.00  0.00           C
HETATM    7  C6  GUA A   1       0.810   3.610   0.000  1.00  0.00           C
HETATM    8  O6  GUA A   1       1.820   4.320   0.000  1.00  0.00           O
HETATM    9  N1  GUA A   1      -0.470   4.180   0.000  1.00  0.00           N
HETATM   10  H1  GUA A   1      -0.510   5.200   0.000  1.00  0.00           H
HETATM   11  C2  GUA A   1      -1.650   3.430   0.000  1.00  0.00           C
HETATM   12  N2  GUA A   1      -2.830   4.090   0.000  1.00  0.00           N
HETATM   13  HN21 GUA A   1      -3.710   3.580   0.000  1.00  0.00           H
HETATM   14  HN22 GUA A   1      -2.830   5.110   0.000  1.00  0.00           H
HETATM   15  N3  GUA A   1      -1.720   2.090   0.000  1.00  0.00           N
HETATM   16  C4  GUA A   1      -0.510   1.560   0.000  1.00  0.00           C
END
""",

    # Cytosine. Pyrimidine: 6-ring with N1-C2(=O)-N3-C4(NH2)-C5-C6.
    "CYT": """\
HETATM    1  N1  CYT A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  H1  CYT A   1      -0.040  -1.000   0.000  1.00  0.00           H
HETATM    3  C2  CYT A   1      -1.220   0.690   0.000  1.00  0.00           C
HETATM    4  O2  CYT A   1      -2.310   0.130   0.000  1.00  0.00           O
HETATM    5  N3  CYT A   1      -1.170   2.040   0.000  1.00  0.00           N
HETATM    6  C4  CYT A   1       0.050   2.640   0.000  1.00  0.00           C
HETATM    7  N4  CYT A   1       0.100   3.990   0.000  1.00  0.00           N
HETATM    8  HN41 CYT A   1       1.010   4.430   0.000  1.00  0.00           H
HETATM    9  HN42 CYT A   1      -0.780   4.480   0.000  1.00  0.00           H
HETATM   10  C5  CYT A   1       1.300   1.900   0.000  1.00  0.00           C
HETATM   11  H5  CYT A   1       2.260   2.410   0.000  1.00  0.00           H
HETATM   12  C6  CYT A   1       1.270   0.590   0.000  1.00  0.00           C
HETATM   13  H6  CYT A   1       2.190   0.030   0.000  1.00  0.00           H
END
""",

    # Thymine. Pyrimidine with methyl at C5 (distinguishes from U).
    "THY": """\
HETATM    1  N1  THY A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  H1  THY A   1      -0.040  -1.000   0.000  1.00  0.00           H
HETATM    3  C2  THY A   1      -1.220   0.690   0.000  1.00  0.00           C
HETATM    4  O2  THY A   1      -2.310   0.130   0.000  1.00  0.00           O
HETATM    5  N3  THY A   1      -1.170   2.040   0.000  1.00  0.00           N
HETATM    6  H3  THY A   1      -2.070   2.530   0.000  1.00  0.00           H
HETATM    7  C4  THY A   1       0.050   2.640   0.000  1.00  0.00           C
HETATM    8  O4  THY A   1       0.100   3.870   0.000  1.00  0.00           O
HETATM    9  C5  THY A   1       1.300   1.900   0.000  1.00  0.00           C
HETATM   10  C5M THY A   1       2.620   2.610   0.000  1.00  0.00           C
HETATM   11  HM51 THY A   1       2.530   3.690   0.000  1.00  0.00           H
HETATM   12  HM52 THY A   1       3.180   2.280   0.880  1.00  0.00           H
HETATM   13  HM53 THY A   1       3.180   2.280  -0.880  1.00  0.00           H
HETATM   14  C6  THY A   1       1.270   0.590   0.000  1.00  0.00           C
HETATM   15  H6  THY A   1       2.190   0.030   0.000  1.00  0.00           H
END
""",

    # Uracil. Thymine without the methyl (H instead of CH3 at C5).
    "URA": """\
HETATM    1  N1  URA A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  H1  URA A   1      -0.040  -1.000   0.000  1.00  0.00           H
HETATM    3  C2  URA A   1      -1.220   0.690   0.000  1.00  0.00           C
HETATM    4  O2  URA A   1      -2.310   0.130   0.000  1.00  0.00           O
HETATM    5  N3  URA A   1      -1.170   2.040   0.000  1.00  0.00           N
HETATM    6  H3  URA A   1      -2.070   2.530   0.000  1.00  0.00           H
HETATM    7  C4  URA A   1       0.050   2.640   0.000  1.00  0.00           C
HETATM    8  O4  URA A   1       0.100   3.870   0.000  1.00  0.00           O
HETATM    9  C5  URA A   1       1.300   1.900   0.000  1.00  0.00           C
HETATM   10  H5  URA A   1       2.260   2.410   0.000  1.00  0.00           H
HETATM   11  C6  URA A   1       1.270   0.590   0.000  1.00  0.00           C
HETATM   12  H6  URA A   1       2.190   0.030   0.000  1.00  0.00           H
END
""",
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
