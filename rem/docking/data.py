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


def _matched_backbone_aligned(s_from: Structure, s_to: Structure, mapping: Dict[str, str],
                              to_subset: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Backbone atoms paired through an explicit residue mapping (from sequence alignment)."""
    bf, bt = s_from.backbone(), s_to.backbone()
    kf = {(r, a): i for i, (r, a) in enumerate(zip(_canonical_ids(bf), bf.atom_names))}
    kt = {(r, a): i for i, (r, a) in enumerate(zip(_canonical_ids(bt), bt.atom_names))}
    P, Q = [], []
    for (rf, a), i in sorted(kf.items()):
        rt = mapping.get(rf)
        if rt is None or (to_subset is not None and rt not in to_subset):
            continue
        j = kt.get((rt, a))
        if j is not None:
            P.append(bf.coords[i]); Q.append(bt.coords[j])
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(P), np.asarray(Q)


def interface_rmsd(case: Dict[str, Structure], cutoff: float = 10.0) -> Dict[str, float]:
    """I-RMSD between bound and unbound components, the quantity that defines difficulty.

    The interface is defined on the BOUND complex, then the same residues are compared
    between bound and unbound after superposition."""
    rb, lb = case["r_b"], case["l_b"]
    out, parts = {}, []
    for tag, bound, unbound, partner in (("receptor", rb, case["r_u"], lb),
                                         ("ligand", lb, case["l_u"], rb)):
        iface = set(interface_residues(bound, partner, cutoff))
        mapping = align_residues(unbound, bound)
        P, Q = _matched_backbone_aligned(unbound, bound, mapping, iface)
        out[tag] = superimposed_rmsd(P, Q) if len(P) >= 3 else float("nan")
        out[f"{tag}_n_atoms"] = float(len(P))
        out[f"{tag}_n_mapped"] = float(len(mapping))
        if len(P) >= 3:
            parts.append((P, Q))
    if parts:
        U = np.vstack([p for p, _ in parts]); B = np.vstack([q for _, q in parts])
        out["combined"] = superimposed_rmsd(U, B)
    else:
        out["combined"] = float("nan")
    return out


def classify(irmsd_combined: float) -> str:
    if not np.isfinite(irmsd_combined):
        return "unknown"
    if irmsd_combined <= RIGID_MAX:
        return "rigid"
    if irmsd_combined <= MEDIUM_MAX:
        return "medium"
    return "difficult"


# --------------------------------------------------------------- sequence-based matching
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "HSD": "H", "HSE": "H", "HSP": "H",
}


def chain_sequences(s: Structure) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Per chain, in file order: (chain_id, [(canonical_residue_key, one_letter), ...])."""
    ids = _canonical_ids(s)
    out, seen = [], {}
    for key, name in zip(ids, s.res_names):
        ch = key.split(":")[0]
        if ch not in seen:
            seen[ch] = []
            out.append((ch, seen[ch]))
        if not seen[ch] or seen[ch][-1][0] != key:
            seen[ch].append((key, _THREE_TO_ONE.get(name, "X")))
    return out


def align_residues(s_from: Structure, s_to: Structure,
                   min_identity: float = 0.7) -> Dict[str, str]:
    """Map residue keys of s_from -> s_to by SEQUENCE ALIGNMENT, per chain in order.

    WHY THIS EXISTS. Keying residues by number recovered only 200 of 271 DB5 cases. The
    excluded 71 shared a signature: one component matched at 100% while the other matched
    0% or 4%. Four percent is not a failure to match -- it is matching the WRONG residues,
    which means the bound and unbound numbering are OFFSET, not merely on renamed chains.
    A residue number is not a stable key across depositions; the sequence is. Aligning also
    handles the insertions and deletions that an offset alone cannot.

    Returns only mappings whose chain alignment reaches min_identity, so a bad alignment
    produces no mapping rather than a plausible wrong one."""
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -10.0, -0.5

    ca, cb = chain_sequences(s_from), chain_sequences(s_to)

    def pair_chain(sa, sb):
        """Align two chains; return (residue pairs, sequence identity) or (None, 0)."""
        if not sa or not sb:
            return None, 0.0
        A, B = "".join(x[1] for x in sa), "".join(x[1] for x in sb)
        try:
            aln = al.align(A, B)[0]
        except Exception:
            return None, 0.0
        ia, ib = aln.aligned
        matched, ident = [], 0
        for (a0, a1), (b0, b1) in zip(ia, ib):
            for k in range(a1 - a0):
                pa, pb = a0 + k, b0 + k
                matched.append((sa[pa][0], sb[pb][0]))
                if A[pa] == B[pb]:
                    ident += 1
        if not matched:
            return None, 0.0
        return matched, ident / len(matched)

    # THE MULTI-COPY TRAP. DB5 unbound files often contain the CRYSTALLOGRAPHIC OLIGOMER --
    # 2VIS ships 3 copies of its ligand chain, 1K4C ships 4 (a tetramer) -- while the bound
    # file has one. Pairing each unbound chain with its best-identity partner maps EVERY copy
    # onto the same bound chain, and a dict update lets the LAST one processed win. That copy
    # sits elsewhere in the lattice, so the interface RMSD came out at 25 A for structures of
    # the same protein. The alignment was never wrong; the wrong COPY was chosen.
    # Correct choice is an assignment, scored by the geometry rather than by sequence: among
    # equally-identical copies, the corresponding one is the one that superimposes best.
    bf, bt = s_from.backbone(), s_to.backbone()
    kf = {(r, a): i for i, (r, a) in enumerate(zip(_canonical_ids(bf), bf.atom_names))}
    kt = {(r, a): i for i, (r, a) in enumerate(zip(_canonical_ids(bt), bt.atom_names))}

    cand = {}
    cost = np.full((len(ca), len(cb)), 1e6)
    for i, (_, sa) in enumerate(ca):
        for j, (_, sb) in enumerate(cb):
            matched, ident = pair_chain(sa, sb)
            if matched is None or ident < min_identity:
                continue
            P, Q = [], []
            for rf, rt in matched:
                for a in BACKBONE:
                    x, y = kf.get((rf, a)), kt.get((rt, a))
                    if x is not None and y is not None:
                        P.append(bf.coords[x]); Q.append(bt.coords[y])
            if len(P) < 3:
                continue
            cand[(i, j)] = matched
            # SCORE THE ASSIGNMENT BY DIRECT RMSD, NOT SUPERIMPOSED RMSD. Superimposed RMSD
            # is rotation-invariant, so every copy of a homo-oligomer scores almost the same
            # and the choice among them is arbitrary -- which left 1K4C (a tetramer) at 17 A
            # and 1FC2 at 24 A even after the assignment was made one-to-one. DB5 ships every
            # unbound structure ALREADY SUPERIMPOSED onto the bound complex, so the
            # corresponding copy is simply the one sitting in the right place. Direct RMSD
            # says which. This uses the pre-superposition ONLY to identify chains; the
            # docking benchmark must still randomize poses before searching, or it inherits
            # the leakage the README warns about.
            cost[i, j] = rmsd(np.asarray(P), np.asarray(Q))

    if not cand:
        return {}
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        chosen = [(i, j) for i, j in zip(rows, cols) if (i, j) in cand]
    except ImportError:                      # greedy fallback, still one-to-one
        chosen, usedf, usedt = [], set(), set()
        for (i, j) in sorted(cand, key=lambda k: cost[k]):
            if i not in usedf and j not in usedt:
                chosen.append((i, j)); usedf.add(i); usedt.add(j)

    out: Dict[str, str] = {}
    for i, j in chosen:
        out.update(dict(cand[(i, j)]))
    return out
