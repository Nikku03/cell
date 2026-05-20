"""Protein-ligand binding: data I/O layer (Milestone 1 of the PDBbind harness).

PDBbind ships one directory per complex:

    <root>/<pdb_id>/<pdb_id>_protein.pdb     full protein
    <root>/<pdb_id>/<pdb_id>_pocket.pdb      protein atoms within ~10 A of ligand
    <root>/<pdb_id>/<pdb_id>_ligand.sdf      the bound ligand (3D)
    <root>/INDEX_refined_data.2020           affinity table

This module parses that layout and also provides a SYNTHETIC generator
that writes the same layout with made-up complexes, so the whole
pipeline (load -> featurise -> train -> verify) is testable without the
~3 GB real download (which is registration-gated and blocked from the
build container; the real run happens on Colab).

The synthetic affinity label is a LEARNABLE function of complex geometry
(contact count + a polarity term), so verification gates can check that
a trained model actually recovers structure-affinity signal rather than
memorising noise.

Public surface:
  parse_pdb_atoms(path)              -> list[PdbAtom]
  parse_pdbbind_index(path)          -> dict[pdb_id -> neg_log_affinity]
  crop_pocket(prot, lig, radius)     -> list[PdbAtom]   (distance crop)
  BindingComplex                     -> one (pocket, ligand, affinity) record
  load_pdbbind(root, ...)            -> list[BindingComplex]
  generate_synthetic_pdbbind(out, n) -> writes a synthetic PDBbind tree
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

import torch

from .atom import SYMBOL_TO_Z


# ---------------- atom record ----------------

@dataclass
class PdbAtom:
    element: str          # 'C', 'N', 'O', ...
    z: int                # atomic number
    x: float
    y: float
    z_coord: float
    name: str = ""        # PDB atom name, e.g. 'CA'
    res_name: str = ""    # residue name, e.g. 'ALA'
    res_seq: int = 0
    chain: str = ""
    is_hetatm: bool = False

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z_coord)


# ---------------- PDB parser (fixed-width, no BioPython) ----------------

def _element_from_pdb_line(line: str) -> str:
    """PDB columns 77-78 hold the element symbol; fall back to the atom
    name's leading letters if that field is blank (common in old files)."""
    elem = line[76:78].strip()
    if elem:
        return elem[0].upper() + elem[1:].lower()
    # Fallback: derive from atom name (cols 13-16)
    name = line[12:16].strip()
    # strip leading digits, take first 1-2 alpha chars
    alpha = "".join(c for c in name if c.isalpha())
    if not alpha:
        return "C"
    cand2 = alpha[:2].capitalize()
    if cand2 in SYMBOL_TO_Z:
        return cand2
    return alpha[0].upper()


def parse_pdb_atoms(path: str, include_hetatm: bool = True,
                     drop_hydrogens: bool = True,
                     drop_water: bool = True) -> list[PdbAtom]:
    """Parse ATOM / HETATM records from a PDB file into PdbAtom records."""
    atoms: list[PdbAtom] = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            is_het = rec == "HETATM"
            if is_het and not include_hetatm:
                continue
            res_name = line[17:20].strip()
            if drop_water and res_name in ("HOH", "WAT", "DOD"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            elem = _element_from_pdb_line(line)
            if drop_hydrogens and elem == "H":
                continue
            zno = SYMBOL_TO_Z.get(elem, SYMBOL_TO_Z.get(elem.capitalize(), 6))
            try:
                res_seq = int(line[22:26])
            except ValueError:
                res_seq = 0
            atoms.append(PdbAtom(
                element=elem, z=zno, x=x, y=y, z_coord=z,
                name=line[12:16].strip(), res_name=res_name,
                res_seq=res_seq, chain=line[21:22].strip(),
                is_hetatm=is_het,
            ))
    return atoms


# ---------------- affinity index parser ----------------

def parse_pdbbind_index(path: str) -> dict[str, float]:
    """Parse a PDBbind INDEX_*_data.* file.

    Each data line looks like:
        1a1e  2.50  1997  6.66  Kd=219nM  // ...
    columns: pdb_id, resolution, year, -logKd/Ki, raw_measurement // ref

    Returns {pdb_id: neg_log_affinity}. Comment lines (start with '#') skipped.
    """
    out: dict[str, float] = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0].lower()
            try:
                neg_log = float(parts[3])
            except ValueError:
                continue
            out[pdb_id] = neg_log
    return out


# ---------------- pocket cropping ----------------

def _min_dist_to_set(a: PdbAtom, others: list[PdbAtom]) -> float:
    best = float("inf")
    ax, ay, az = a.x, a.y, a.z_coord
    for o in others:
        d = (ax - o.x) ** 2 + (ay - o.y) ** 2 + (az - o.z_coord) ** 2
        if d < best:
            best = d
    return math.sqrt(best) if best < float("inf") else best


def crop_pocket(protein_atoms: list[PdbAtom], ligand_atoms: list[PdbAtom],
                 radius: float = 8.0) -> list[PdbAtom]:
    """Return protein atoms whose nearest ligand atom is within `radius` A.

    PDBbind also ships a precomputed _pocket.pdb (10 A); use this when you
    only have the full protein, or to re-crop to a tighter radius."""
    if not ligand_atoms:
        return []
    keep: list[PdbAtom] = []
    for pa in protein_atoms:
        if _min_dist_to_set(pa, ligand_atoms) <= radius:
            keep.append(pa)
    return keep


# ---------------- binding complex record ----------------

@dataclass
class BindingComplex:
    pdb_id: str
    pocket: list[PdbAtom]              # protein atoms near the ligand
    ligand: list[PdbAtom]             # ligand atoms
    neg_log_affinity: float           # -logKd/Ki  (PDBbind label; ~2-12)
    meta: dict = field(default_factory=dict)

    @property
    def n_pocket(self) -> int:
        return len(self.pocket)

    @property
    def n_ligand(self) -> int:
        return len(self.ligand)

    def positions(self) -> torch.Tensor:
        """(n_pocket + n_ligand, 3) coordinates, pocket first then ligand."""
        rows = [[a.x, a.y, a.z_coord] for a in self.pocket]
        rows += [[a.x, a.y, a.z_coord] for a in self.ligand]
        return torch.tensor(rows, dtype=torch.float32)

    def atomic_numbers(self) -> torch.Tensor:
        zs = [a.z for a in self.pocket] + [a.z for a in self.ligand]
        return torch.tensor(zs, dtype=torch.long)

    def is_ligand_mask(self) -> torch.Tensor:
        """(n_pocket + n_ligand,) bool: True for ligand atoms."""
        m = torch.zeros(self.n_pocket + self.n_ligand, dtype=torch.bool)
        m[self.n_pocket:] = True
        return m


# ---------------- real-data loader ----------------

def load_pdbbind(root: str, index_file: str | None = None,
                  pocket_radius: float = 8.0, max_complexes: int | None = None,
                  use_provided_pocket: bool = True) -> list[BindingComplex]:
    """Load complexes from a real PDBbind directory tree.

    root: directory containing per-complex subdirs + an INDEX file.
    use_provided_pocket: if True and a <id>_pocket.pdb exists, use it;
      otherwise crop from the full protein with `pocket_radius`.
    """
    from rdkit import Chem  # local import: only needed for real data

    if index_file is None:
        cands = [f for f in os.listdir(root) if f.startswith("INDEX") and "data" in f]
        if not cands:
            raise FileNotFoundError(f"no INDEX_*_data.* file in {root}")
        index_file = os.path.join(root, sorted(cands)[-1])
    affinity = parse_pdbbind_index(index_file)

    complexes: list[BindingComplex] = []
    for pdb_id in sorted(affinity):
        cdir = os.path.join(root, pdb_id)
        if not os.path.isdir(cdir):
            continue
        lig_sdf = os.path.join(cdir, f"{pdb_id}_ligand.sdf")
        pocket_pdb = os.path.join(cdir, f"{pdb_id}_pocket.pdb")
        protein_pdb = os.path.join(cdir, f"{pdb_id}_protein.pdb")
        if not os.path.exists(lig_sdf):
            continue

        supplier = Chem.SDMolSupplier(lig_sdf, removeHs=True, sanitize=True)
        mol = next((m for m in supplier if m is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        ligand_atoms = []
        for at in mol.GetAtoms():
            p = conf.GetAtomPosition(at.GetIdx())
            sym = at.GetSymbol()
            ligand_atoms.append(PdbAtom(
                element=sym, z=SYMBOL_TO_Z.get(sym, 6),
                x=p.x, y=p.y, z_coord=p.z, name=sym, is_hetatm=True,
            ))

        if use_provided_pocket and os.path.exists(pocket_pdb):
            pocket = parse_pdb_atoms(pocket_pdb)
        elif os.path.exists(protein_pdb):
            protein = parse_pdb_atoms(protein_pdb)
            pocket = crop_pocket(protein, ligand_atoms, radius=pocket_radius)
        else:
            continue

        complexes.append(BindingComplex(
            pdb_id=pdb_id, pocket=pocket, ligand=ligand_atoms,
            neg_log_affinity=affinity[pdb_id],
        ))
        if max_complexes is not None and len(complexes) >= max_complexes:
            break
    return complexes


# ---------------- synthetic PDBbind-format generator ----------------

_AA = ["ALA", "GLY", "SER", "VAL", "LEU", "THR", "ASP", "GLU", "LYS", "PHE"]
_PROT_ELEMENTS = ["C", "C", "C", "N", "O", "S"]   # rough protein composition


def _write_pdb(path: str, atoms: list[PdbAtom]) -> None:
    with open(path, "w") as fh:
        for i, a in enumerate(atoms, start=1):
            rec = "HETATM" if a.is_hetatm else "ATOM  "
            fh.write(
                f"{rec}{i:>5} {a.name:<4}{'':1}{a.res_name:>3} "
                f"{a.chain or 'A':1}{a.res_seq:>4}{'':4}"
                f"{a.x:>8.3f}{a.y:>8.3f}{a.z_coord:>8.3f}"
                f"{1.0:>6.2f}{0.0:>6.2f}{'':10}{a.element:>2}\n"
            )
        fh.write("END\n")


def _synthetic_affinity(pocket: list[PdbAtom], ligand: list[PdbAtom]) -> float:
    """A LEARNABLE structure->affinity function.

    Driven by two geometric signals the model can see:
      - per-ligand-atom contact count (how buried the ligand is)
      - polar-contact fraction (N/O <-> N/O pairs among the contacts)
    Plus small noise. Calibrated to spread across PDBbind's ~3-11
    -logKd/Ki band rather than saturating at the clamp."""
    contacts = 0
    polar = 0
    for la in ligand:
        for pa in pocket:
            d2 = ((la.x - pa.x) ** 2 + (la.y - pa.y) ** 2
                  + (la.z_coord - pa.z_coord) ** 2)
            if d2 <= 12.25:                       # within 3.5 A
                contacts += 1
                if pa.element in ("N", "O") and la.element in ("N", "O"):
                    polar += 1
    n_lig = max(len(ligand), 1)
    per_lig = contacts / n_lig                    # ~1-8 depending on burial
    polar_frac = polar / max(contacts, 1)         # 0-~0.3
    base = 2.8 + 0.78 * per_lig + 3.5 * polar_frac
    base += random.gauss(0.0, 0.35)
    return max(2.0, min(12.0, base))


def generate_synthetic_pdbbind(out_dir: str, n_complexes: int = 60,
                                 seed: int = 0) -> str:
    """Write a synthetic PDBbind-format tree (per-complex dirs + INDEX file).

    Pockets: 80-160 random C/N/O/S atoms in a spherical shell.
    Ligands: 12-28 random C/N/O atoms in a small blob at the pocket centre.
    Affinity: _synthetic_affinity() - a learnable function of geometry.

    Returns the path to the written root directory.
    """
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    index_rows: list[str] = []

    for ci in range(n_complexes):
        pdb_id = f"syn{ci:04d}"
        cdir = os.path.join(out_dir, pdb_id)
        os.makedirs(cdir, exist_ok=True)

        # Ligand: small blob near origin. Vary compactness per complex so
        # the burial / contact count (and thus affinity) genuinely spreads.
        n_lig = rng.randint(10, 32)
        lig_spread = rng.uniform(1.4, 3.4)        # tight -> buried, loose -> exposed
        # Vary the polar (N/O) fraction per complex too
        polar_p = rng.uniform(0.10, 0.55)
        ligand = []
        for j in range(n_lig):
            elem = (rng.choice(["N", "O"]) if rng.random() < polar_p else "C")
            ligand.append(PdbAtom(
                element=elem, z=SYMBOL_TO_Z[elem],
                x=rng.gauss(0, lig_spread), y=rng.gauss(0, lig_spread),
                z_coord=rng.gauss(0, lig_spread),
                name=elem, res_name="LIG", is_hetatm=True,
            ))

        # Pocket: shell of protein atoms around the ligand
        n_pock = rng.randint(80, 160)
        pocket = []
        for j in range(n_pock):
            # random direction, radius 4-11 A
            theta = rng.uniform(0, math.pi)
            phi = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(4.0, 11.0)
            elem = rng.choice(_PROT_ELEMENTS)
            pocket.append(PdbAtom(
                element=elem, z=SYMBOL_TO_Z[elem],
                x=r * math.sin(theta) * math.cos(phi),
                y=r * math.sin(theta) * math.sin(phi),
                z_coord=r * math.cos(theta),
                name=elem, res_name=rng.choice(_AA),
                res_seq=j // 8, chain="A",
            ))

        aff = _synthetic_affinity(pocket, ligand)

        _write_pdb(os.path.join(cdir, f"{pdb_id}_pocket.pdb"), pocket)
        _write_pdb(os.path.join(cdir, f"{pdb_id}_protein.pdb"), pocket)
        _write_pdb(os.path.join(cdir, f"{pdb_id}_ligand_atoms.pdb"), ligand)
        index_rows.append(f"{pdb_id}  2.00  2020  {aff:.2f}  Kd=synthetic  // synthetic")

    with open(os.path.join(out_dir, "INDEX_refined_data.2020"), "w") as fh:
        fh.write("# synthetic PDBbind-format index\n")
        fh.write("# PDB_code  resolution  year  -logKd/Ki  Kd/Ki  // reference\n")
        fh.write("\n".join(index_rows) + "\n")
    return out_dir


def load_synthetic_pdbbind(root: str,
                            max_complexes: int | None = None) -> list[BindingComplex]:
    """Loader for the synthetic tree (ligands stored as a plain PDB, not SDF,
    so no RDKit dependency for the synthetic smoke path)."""
    index_file = os.path.join(root, "INDEX_refined_data.2020")
    affinity = parse_pdbbind_index(index_file)
    out: list[BindingComplex] = []
    for pdb_id in sorted(affinity):
        cdir = os.path.join(root, pdb_id)
        lig_pdb = os.path.join(cdir, f"{pdb_id}_ligand_atoms.pdb")
        pocket_pdb = os.path.join(cdir, f"{pdb_id}_pocket.pdb")
        if not (os.path.exists(lig_pdb) and os.path.exists(pocket_pdb)):
            continue
        out.append(BindingComplex(
            pdb_id=pdb_id,
            pocket=parse_pdb_atoms(pocket_pdb),
            ligand=parse_pdb_atoms(lig_pdb),
            neg_log_affinity=affinity[pdb_id],
        ))
        if max_complexes is not None and len(out) >= max_complexes:
            break
    return out
