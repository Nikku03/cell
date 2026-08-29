"""Docking Benchmark 5.5 loader, difficulty classification, and RMSD metrics.

THE BENCHMARK'S OWN WARNING, quoted from its README because it is a leakage trap:

    "Each unbound structure was superimposed onto the corresponding bound complex. This
     fact may cause biased docking results if the docking algorithm is sensitive to
     initial position of receptor/ligand. Users can avoid the possible biased docking
     results by randomizing the unbound structure positions."

The unbound structures ship ALREADY SITTING ON THE ANSWER. Docking them as-provided
measures nothing and reports a fake success. `randomize_pose()` exists for this and the
benchmark driver must call it; `leakage_check()` measures how good the as-provided starting
pose already is, so the size of the trap is on the record rather than assumed.

DIFFICULTY IS COMPUTED, NOT TRANSCRIBED. Vreven et al. (2015) classify by interface RMSD
between the bound and unbound components: rigid-body <= 1.5 A, medium 1.5-2.2 A,
difficult > 2.2 A. DB5.5 does not ship that table, and I-RMSD is the defining quantity, so
it is measured here from the structures themselves. That also exercises the RMSD code every
accuracy number in this suite depends on.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
DB5_DIR = os.path.join(SCRATCH, "db5", "benchmark5.5", "structures")

BACKBONE = ("N", "CA", "C", "O")
RIGID_MAX, MEDIUM_MAX = 1.5, 2.2          # Vreven et al. 2015 I-RMSD thresholds


@dataclass
class Structure:
    coords: np.ndarray            # (n_atoms, 3)
    atom_names: np.ndarray        # (n_atoms,)
    res_ids: np.ndarray           # (n_atoms,) "chain:resseq:icode"
    res_names: np.ndarray
    elements: np.ndarray

    def __len__(self):
        return len(self.coords)

    def select(self, mask) -> "Structure":
        return Structure(self.coords[mask], self.atom_names[mask], self.res_ids[mask],
                         self.res_names[mask], self.elements[mask])

    def backbone(self) -> "Structure":
        return self.select(np.isin(self.atom_names, BACKBONE))

    def residues(self) -> List[str]:
        seen, out = set(), []
        for r in self.res_ids:
            if r not in seen:
                seen.add(r); out.append(r)
        return out


def read_pdb(path: str) -> Structure:
    """Minimal fixed-column PDB parser. Skips hydrogens and altloc != ' '/'A'."""
    xyz, an, ri, rn, el = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            name = line[12:16].strip()
            elem = line[76:78].strip() or name[:1]
            if elem == "H" or name.startswith("H"):
                continue
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            an.append(name)
            ri.append(f"{line[21]}:{line[22:26].strip()}:{line[26].strip()}")
            rn.append(line[17:20].strip())
            el.append(elem)
    if not xyz:
        raise ValueError(f"no ATOM records parsed from {path}")
    return Structure(np.asarray(xyz, float), np.asarray(an), np.asarray(ri),
                     np.asarray(rn), np.asarray(el))


def list_complexes(db5_dir: str = DB5_DIR) -> List[str]:
    ids = set()
    for p in glob.glob(os.path.join(db5_dir, "*_r_b.pdb")):
        cid = os.path.basename(p)[:4]
        if all(os.path.exists(os.path.join(db5_dir, f"{cid}_{a}_{b}.pdb"))
               for a in "rl" for b in "ub"):
            ids.add(cid)
    return sorted(ids)


def load_case(cid: str, db5_dir: str = DB5_DIR) -> Dict[str, Structure]:
    return {f"{a}_{b}": read_pdb(os.path.join(db5_dir, f"{cid}_{a}_{b}.pdb"))
            for a in "rl" for b in "ub"}


# ------------------------------------------------------------------ geometry
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R P + t - Q|. Handles reflection."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.sqrt(((P - Q) ** 2).sum(1).mean()))


def superimposed_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = kabsch(P, Q)
    return rmsd((R @ P.T).T + t, Q)


def interface_residues(a: Structure, b: Structure, cutoff: float = 10.0) -> List[str]:
    """Residues of `a` with any heavy atom within `cutoff` of any heavy atom of `b`."""
    d2 = ((a.coords[:, None, :] - b.coords[None, :, :]) ** 2).sum(-1)
    close = (d2 <= cutoff * cutoff).any(1)
    ids = _canonical_ids(a)                        # canonical, so it keys against unbound
    seen, out = set(), []
    for r in ids[close]:
        if r not in seen:
            seen.add(r); out.append(r)
    return out


def chain_order(s: Structure) -> List[str]:
    """Chain IDs in order of first appearance."""
    seen, out = set(), []
    for r in s.res_ids:
        c = r.split(":")[0]
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def _canonical_ids(s: Structure) -> np.ndarray:
    """Residue keys as (chain_INDEX, resseq, icode) rather than (chain_LETTER, ...).

    DB5 RENAMES CHAINS between bound and unbound: 1ACB's receptor is chain E bound and B
    unbound, and its ligand is I bound and blank unbound. Keying on the letter silently
    matched zero atoms and produced NaN for most of the benchmark -- a failure that looks
    like missing data rather than a bug. Chain ORDER is preserved, so the index is the
    stable key. `residue_name_agreement` below is what proves the mapping is right rather
    than merely non-empty."""
    idx = {c: i for i, c in enumerate(chain_order(s))}
    return np.array([f"{idx[r.split(':')[0]]}:{':'.join(r.split(':')[1:])}"
                     for r in s.res_ids])


def residue_name_agreement(s1: Structure, s2: Structure) -> Tuple[float, int]:
    """Fraction of co-keyed residues whose 3-letter names agree, and how many were compared.

    A mapping can be non-empty and still be wrong. If two structures of the same protein
    are aligned correctly their residue names must agree; a low rate means the keys are
    lining up different residues and any RMSD computed from them is meaningless."""
    c1, c2 = _canonical_ids(s1), _canonical_ids(s2)
    n1 = {k: v for k, v in zip(c1, s1.res_names)}
    n2 = {k: v for k, v in zip(c2, s2.res_names)}
    shared = sorted(set(n1) & set(n2))
    if not shared:
        return 0.0, 0
    agree = sum(1 for k in shared if n1[k] == n2[k])
    return agree / len(shared), len(shared)


def _matched_backbone(s1: Structure, s2: Structure,
                      res_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Backbone atoms present in BOTH structures, keyed by (chain index, resseq, atom)."""
    b1, b2 = s1.backbone(), s2.backbone()
    c1, c2 = _canonical_ids(b1), _canonical_ids(b2)
    k1 = {(r, a): i for i, (r, a) in enumerate(zip(c1, b1.atom_names))}
    k2 = {(r, a): i for i, (r, a) in enumerate(zip(c2, b2.atom_names))}
    keys = sorted(set(k1) & set(k2))
    if res_subset is not None:
        want = set(res_subset)                    # res_subset arrives as canonical ids
        keys = [k for k in keys if k[0] in want]
    if not keys:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return (b1.coords[[k1[k] for k in keys]], b2.coords[[k2[k] for k in keys]])


def interface_rmsd(case: Dict[str, Structure], cutoff: float = 10.0) -> Dict[str, float]:
    """I-RMSD between bound and unbound components, the quantity that defines difficulty.

    The interface is defined on the BOUND complex, then the same residues are compared
    between bound and unbound after superposition."""
    rb, lb = case["r_b"], case["l_b"]
    out = {}
    for tag, bound, unbound, partner in (("receptor", rb, case["r_u"], lb),
                                         ("ligand", lb, case["l_u"], rb)):
        res = interface_residues(bound, partner, cutoff)
        P, Q = _matched_backbone(unbound, bound, res)
        out[tag] = superimposed_rmsd(P, Q) if len(P) >= 3 else float("nan")
        out[f"{tag}_n_atoms"] = float(len(P))
    both_b = np.vstack([_matched_backbone(case["r_u"], rb, interface_residues(rb, lb, cutoff))[1],
                        _matched_backbone(case["l_u"], lb, interface_residues(lb, rb, cutoff))[1]])
    both_u = np.vstack([_matched_backbone(case["r_u"], rb, interface_residues(rb, lb, cutoff))[0],
                        _matched_backbone(case["l_u"], lb, interface_residues(lb, rb, cutoff))[0]])
    out["combined"] = superimposed_rmsd(both_u, both_b) if len(both_u) >= 3 else float("nan")
    return out


def classify(irmsd_combined: float) -> str:
    if not np.isfinite(irmsd_combined):
        return "unknown"
    if irmsd_combined <= RIGID_MAX:
        return "rigid"
    if irmsd_combined <= MEDIUM_MAX:
        return "medium"
    return "difficult"
