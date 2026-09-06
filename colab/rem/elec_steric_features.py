"""Electrostatic and steric feature blocks, from the cached AlphaFold monomers. A cache, no gates.

KEPT AS TWO SEPARATE BLOCKS, and separate from the 64 geometric descriptors already in
struct_enzymes.npz, because loop 163c's M2 showed that what decides whether a new block is worth
anything is whether it is INDEPENDENT of the blocks already present -- sequence and structure
correlate at Spearman +0.5858 and the merge could only recover +0.0058. A block that re-derives
geometry under a new name will correlate with it and buy nothing, and keeping them separate is what
makes that measurable rather than assumed.

ELECTROSTATICS. Formal charges at pH 7.4 -- Asp and Glu -1, Lys and Arg +1, His +0.1 -- placed at
side-chain centroids, then:
  net charge, and net charge of the SURFACE separately from the core, since a ligand sees the surface
  the dipole: magnitude of the charge-weighted position sum, and the separation between the positive
    and negative centroids normalised by the radius of gyration, which says whether the molecule is
    a dipole or merely charged
  a screened Coulomb self-energy, sum over charged pairs of q_i q_j exp(-r/lambda)/r at a Debye
    length of 10 A, which is the actual electrostatic quantity rather than a residue count
  the largest connected SURFACE PATCH of like charge, found by connected components over charged
    surface residues within 8 A -- a binding site for a charged ligand is a patch, not an average
  charge asymmetry projected on the three principal axes

STERICS. Residue volumes from the standard table, then:
  side-chain bulk, its variance, and the bulky (FWYLIM) and small (GAS) fractions
  packing density: atoms per A^3 inside the radius of gyration, and the fraction of residue pairs in
    tight CB-CB contact under 5 A
  the free volume of the largest internal cavity, which is how much room a ligand actually has
  crowding around the most buried decile of residues, where a buried site would have to sit
  glycine and proline content as a backbone-flexibility proxy, and pLDDT variance as a second one

-> colab/data/ml/elecster_enzymes.npz
"""
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loop_replication as LR  # noqa: E402

AF = LR.SC / "af"
ACCS = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/enz_accs.json")
OUT = Path("colab/data/ml/elecster_enzymes.npz")
THREE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
         "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
         "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
# residue volumes, A^3 (Zamyatnin 1972)
VOL = {"A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9, "G": 60.1, "H": 153.2,
       "I": 166.7, "K": 168.6, "L": 166.7, "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8,
       "R": 173.4, "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6}
BULKY, SMALL = set("FWYLIM"), set("GAS")
DEBYE = 10.0


def parse(path):
    """CA and side-chain centroid per residue, plus residue name and pLDDT."""
    cur, out = None, []
    ca, sc, res, pl = [], [], [], []
    side = []
    for ln in open(path, errors="replace"):
        if not ln.startswith("ATOM"):
            continue
        ri = int(ln[22:26])
        at = ln[12:16].strip()
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        if ri != cur:
            if cur is not None:
                sc.append(np.mean(side, 0) if side else ca[-1])
            cur, side = ri, []
            res.append(ln[17:20].strip())
            pl.append(float(ln[60:66]))
            ca.append(xyz)
        if at == "CA":
            ca[-1] = xyz
        elif at not in ("N", "C", "O"):
            side.append(xyz)
    if cur is not None:
        sc.append(np.mean(side, 0) if side else ca[-1])
    if len(ca) < 30:
        return None
    return np.array(ca), np.array(sc), np.array(res), np.array(pl)


def features(ca, sc, res, pl):
    n = len(ca)
    letters = np.array([THREE.get(r, "X") for r in res])
    q = np.array([CHARGE.get(c, 0.0) for c in letters])
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    cen = ca.mean(0)
    rg = float(np.sqrt(((ca - cen) ** 2).sum(1).mean()))
    nb = (d < 10).sum(1) - 1.0
    surface = nb < 18

    E, En = [], []

    def add(v, name, into):
        into.append(float(v) if np.isfinite(v) else 0.0)
        (Ename if into is E else Sname).append(name)

    Ename, Sname = [], []
    # ---- electrostatics
    E.append(float(q.sum())); Ename.append("net_charge")
    E.append(float(q.sum() / n)); Ename.append("net_charge_per_res")
    E.append(float(q[surface].sum())); Ename.append("surface_net_charge")
    E.append(float(q[~surface].sum())); Ename.append("core_net_charge")
    E.append(float((q > 0).mean())); Ename.append("frac_pos")
    E.append(float((q < 0).mean())); Ename.append("frac_neg")
    E.append(float((np.abs(q) > 0).mean())); Ename.append("frac_charged")
    E.append(float((np.abs(q[surface]) > 0).mean() if surface.any() else 0)); Ename.append("frac_surface_charged")
    dip = (q[:, None] * (sc - cen)).sum(0)
    E.append(float(np.linalg.norm(dip) / max(rg, 1e-6))); Ename.append("dipole_over_rg")
    pos, neg = q > 0, q < 0
    if pos.any() and neg.any():
        sep = np.linalg.norm(sc[pos].mean(0) - sc[neg].mean(0)) / max(rg, 1e-6)
    else:
        sep = 0.0
    E.append(sep); Ename.append("pos_neg_centroid_sep_over_rg")
    ch = np.where(np.abs(q) > 0)[0]
    if len(ch) > 1:
        dq = d[np.ix_(ch, ch)]
        qq = np.outer(q[ch], q[ch])
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.where(dq > 0, qq * np.exp(-dq / DEBYE) / np.maximum(dq, 1e-6), 0.0)
        E.append(float(np.triu(U, 1).sum() / n)); Ename.append("screened_coulomb_per_res")
        E.append(float(np.abs(np.triu(U, 1)).sum() / n)); Ename.append("abs_coulomb_per_res")
    else:
        E += [0.0, 0.0]; Ename += ["screened_coulomb_per_res", "abs_coulomb_per_res"]
    # largest like-charge surface patch
    for sign, nm in ((1, "pos"), (-1, "neg")):
        sel = np.where(surface & (q * sign > 0))[0]
        best = 0
        if len(sel):
            adj = {int(i): [int(j) for j in sel if j != i and d[i, j] < 8.0] for i in sel}
            seen = set()
            for s0 in sel:
                if int(s0) in seen:
                    continue
                comp, dq2 = 0, deque([int(s0)])
                seen.add(int(s0))
                while dq2:
                    u = dq2.popleft()
                    comp += 1
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            dq2.append(v)
                best = max(best, comp)
        E.append(best / n); Ename.append(f"largest_{nm}_patch_frac")
    c = ca - cen
    G = (c.T @ c) / n
    ev, evec = np.linalg.eigh(G)
    proj = (sc - cen) @ evec
    for k in range(3):
        E.append(float((q * proj[:, k]).sum() / (n * max(rg, 1e-6))))
        Ename.append(f"charge_asym_axis{k}")

    # ---- sterics
    vol = np.array([VOL.get(c, 120.0) for c in letters])
    S = []
    S.append(float(vol.mean())); Sname.append("mean_res_volume")
    S.append(float(vol.std())); Sname.append("sd_res_volume")
    S.append(float(np.isin(letters, list(BULKY)).mean())); Sname.append("frac_bulky")
    S.append(float(np.isin(letters, list(SMALL)).mean())); Sname.append("frac_small")
    S.append(float((letters == "G").mean())); Sname.append("frac_gly")
    S.append(float((letters == "P").mean())); Sname.append("frac_pro")
    S.append(float(vol.sum() / (4 / 3 * np.pi * rg ** 3))); Sname.append("packing_vol_over_sphere")
    dsc = np.linalg.norm(sc[:, None, :] - sc[None, :, :], axis=-1)
    np.fill_diagonal(dsc, 1e9)
    S.append(float((dsc < 5.0).sum() / (2.0 * n))); Sname.append("tight_contacts_per_res")
    S.append(float(np.median(dsc.min(1)))); Sname.append("median_nearest_sidechain")
    S.append(float(np.percentile(dsc.min(1), 10))); Sname.append("p10_nearest_sidechain")
    # free volume of internal cavities at 2 A
    g = 2.0
    qgrid = np.floor((sc - sc.min(0)) / g).astype(int)
    occ = np.zeros(qgrid.max(0) + 1, bool)
    occ[qgrid[:, 0], qgrid[:, 1], qgrid[:, 2]] = True
    enc = np.ones_like(occ)
    for ax in range(3):
        f1 = np.cumsum(occ, axis=ax) > 0
        f2 = np.flip(np.cumsum(np.flip(occ, ax), axis=ax), ax) > 0
        enc &= f1 & f2
    cav = enc & ~occ
    S.append(float(cav.sum() * g ** 3)); Sname.append("cavity_volume")
    S.append(float(cav.sum() / max(occ.sum(), 1))); Sname.append("cavity_over_occupied")
    S.append(float(np.log1p(cav.sum() * g ** 3))); Sname.append("log_cavity_volume")
    deep = nb >= np.percentile(nb, 90)
    S.append(float(vol[deep].mean() if deep.any() else 0)); Sname.append("volume_at_buried_decile")
    S.append(float(nb[deep].mean() if deep.any() else 0)); Sname.append("crowding_at_buried_decile")
    S.append(float(pl.std())); Sname.append("plddt_sd")
    S.append(float((pl < 70).mean())); Sname.append("frac_low_plddt")
    return np.array(E, np.float32), Ename, np.array(S, np.float32), Sname


def main():
    t0 = time.time()
    want = json.load(open(ACCS))
    accs, Es, Ss, en, sn = [], [], [], None, None
    for i, a in enumerate(want):
        p = AF / f"{a}.pdb"
        if not p.exists():
            continue
        got = parse(p)
        if got is None:
            continue
        e, en, s, sn = features(*got)
        if not (np.isfinite(e).all() and np.isfinite(s).all()):
            continue
        accs.append(a)
        Es.append(e)
        Ss.append(s)
        if len(accs) % 400 == 0:
            print(f"  {len(accs):,}/{len(want):,} [{time.time()-t0:.0f}s]", flush=True)
    E = np.array(Es, np.float32)
    S = np.array(Ss, np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, accs=np.array(accs), elec=E, steric=S,
                        elec_names=np.array(en), steric_names=np.array(sn))
    print(f"wrote elec {E.shape} ({len(en)} features), steric {S.shape} ({len(sn)}) "
          f"-> {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
