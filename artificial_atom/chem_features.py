"""Extract real chemistry features from RDKit molecules.

For a given RDKit Mol with a single conformer, produce the per-atom and
per-bond feature tensors that the v3 network consumes:

  positions       (N, 3)   float32
  Z               (N,)     long
  hybridisation   (N,)     long      (HYB_* enum)
  formal_charge   (N,)     long
  aromatic        (N,)     bool
  in_ring         (N,)     bool
  h_donor         (N,)     bool
  h_acceptor      (N,)     bool
  partial_charge  (N,)     float32   (Gasteiger)
  vdw_radius      (N,)     float32   (Angstroms; from RDKit periodic table)

  bonds           (B, 2)   long      (i, j) with i < j
  bond_order      (B,)     float32   (1.0, 1.5 aromatic, 2.0, 3.0)
  bond_in_ring    (B,)     bool
  bond_conjugated (B,)     bool
  bond_rotatable  (B,)     bool

Donor / acceptor heuristics (no full SMARTS rules - sufficient for our
small-molecule training distribution):
  donor:    N or O with at least one H
  acceptor: N (any), O with no H or with lone pair (treated as any O)
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges

from .unit import (
    HYB_AROMATIC, HYB_NONE, HYB_SP, HYB_SP2, HYB_SP3,
)


_HYB_MAP = {
    Chem.HybridizationType.UNSPECIFIED: HYB_NONE,
    Chem.HybridizationType.S: HYB_NONE,
    Chem.HybridizationType.SP: HYB_SP,
    Chem.HybridizationType.SP2: HYB_SP2,
    Chem.HybridizationType.SP3: HYB_SP3,
    Chem.HybridizationType.SP3D: HYB_SP3,
    Chem.HybridizationType.SP3D2: HYB_SP3,
}


_PTABLE = Chem.GetPeriodicTable()


def _hyb_of(atom) -> int:
    if atom.GetIsAromatic():
        return HYB_AROMATIC
    return _HYB_MAP.get(atom.GetHybridization(), HYB_NONE)


def _donor_acceptor(atom) -> tuple[bool, bool]:
    sym = atom.GetSymbol()
    n_h = atom.GetTotalNumHs()
    donor = sym in {"N", "O"} and n_h > 0
    acceptor = sym in {"N", "O"}
    return donor, acceptor


def featurise(mol, conformer_id: int = 0) -> dict:
    """Returns a dict of feature tensors for a single conformer of `mol`."""
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformer; embed it first")
    conf = mol.GetConformer(conformer_id)

    # Compute Gasteiger charges once.
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass  # rare failures on weird structures; leave charges at zero

    n = mol.GetNumAtoms()
    positions = torch.zeros((n, 3), dtype=torch.float32)
    Z = torch.zeros(n, dtype=torch.long)
    hyb = torch.zeros(n, dtype=torch.long)
    formal_charge = torch.zeros(n, dtype=torch.long)
    aromatic = torch.zeros(n, dtype=torch.bool)
    in_ring = torch.zeros(n, dtype=torch.bool)
    h_donor = torch.zeros(n, dtype=torch.bool)
    h_acceptor = torch.zeros(n, dtype=torch.bool)
    partial_charge = torch.zeros(n, dtype=torch.float32)
    vdw_radius = torch.zeros(n, dtype=torch.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        p = conf.GetAtomPosition(i)
        positions[i] = torch.tensor([p.x, p.y, p.z])
        Z[i] = atom.GetAtomicNum()
        hyb[i] = _hyb_of(atom)
        formal_charge[i] = atom.GetFormalCharge()
        aromatic[i] = atom.GetIsAromatic()
        in_ring[i] = atom.IsInRing()
        d, a = _donor_acceptor(atom)
        h_donor[i] = d
        h_acceptor[i] = a
        if atom.HasProp("_GasteigerCharge"):
            q = atom.GetDoubleProp("_GasteigerCharge")
            if q != q or q == float("inf") or q == float("-inf"):
                q = 0.0
            partial_charge[i] = q
        vdw_radius[i] = _PTABLE.GetRvdw(int(Z[i].item()))

    # Bond features
    bond_pairs: list[tuple[int, int]] = []
    bond_order: list[float] = []
    bond_in_ring: list[bool] = []
    bond_conjugated: list[bool] = []
    bond_rotatable: list[bool] = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        bond_pairs.append((i, j))
        bond_order.append(b.GetBondTypeAsDouble())
        bond_in_ring.append(bool(b.IsInRing()))
        bond_conjugated.append(bool(b.GetIsConjugated()))
        # Rotatable: single bond, not in ring, both endpoints non-terminal.
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        rot = (
            b.GetBondType() == Chem.BondType.SINGLE
            and not b.IsInRing()
            and a1.GetDegree() > 1
            and a2.GetDegree() > 1
            and not a1.GetIsAromatic()
            and not a2.GetIsAromatic()
        )
        bond_rotatable.append(rot)

    if bond_pairs:
        bonds_t = torch.tensor(bond_pairs, dtype=torch.long)
        bo_t = torch.tensor(bond_order, dtype=torch.float32)
        bir_t = torch.tensor(bond_in_ring, dtype=torch.bool)
        bcon_t = torch.tensor(bond_conjugated, dtype=torch.bool)
        brot_t = torch.tensor(bond_rotatable, dtype=torch.bool)
    else:
        bonds_t = torch.empty(0, 2, dtype=torch.long)
        bo_t = torch.empty(0, dtype=torch.float32)
        bir_t = torch.empty(0, dtype=torch.bool)
        bcon_t = torch.empty(0, dtype=torch.bool)
        brot_t = torch.empty(0, dtype=torch.bool)

    return {
        "positions": positions,
        "Z": Z,
        "hybridisation": hyb,
        "formal_charge": formal_charge,
        "aromatic": aromatic,
        "in_ring": in_ring,
        "h_donor": h_donor,
        "h_acceptor": h_acceptor,
        "partial_charge": partial_charge,
        "vdw_radius": vdw_radius,
        "bonds": bonds_t,
        "bond_order": bo_t,
        "bond_in_ring": bir_t,
        "bond_conjugated": bcon_t,
        "bond_rotatable": brot_t,
    }
