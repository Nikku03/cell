"""Structure features for the enzyme set, from cached AlphaFold monomers. A cache, not a test.

DELIBERATELY NOT DOCKING. The nexus arm spent 92% of its compute on FFT shape-complementarity
docking and measured the result: every docking feature landed in AUC [0.450, 0.549] against a
size-only control at 0.532, true-catalyst mean rank 5.47 against a chance of 5.5, and in the
enumerated feature-block design space every docking block had regret 0.0000. Repeating that would
be repeating a measured null.

What is extracted instead is the geometry a FOLD has independent of any partner -- shape, compactness,
surface chemistry, contact topology and pocket-like concavity. If structure carries information
about what an enzyme does, it should be here, and if it is not here it is unlikely to be in a more
expensive version of the same idea.

  shape         radius of gyration, asphericity and acylindricity from the gyration tensor,
                normalised principal-axis ratios -- 7 numbers that say what shape the fold is
  compactness   atoms per residue, radius of gyration against the N^0.38 globular expectation,
                fraction of residues buried at 3 burial thresholds
  surface       amino-acid composition of the SURFACE residues separately from the core, because
                what an enzyme touches is its surface and its core is mostly packing (40 numbers)
  contacts      contact-order, contact-density, and the degree distribution of the CA contact graph
                at 8 A -- the fold's topology rather than its shape
  concavity     the count and volume of grid cells that are enclosed by protein but empty, at 2 A
                spacing -- a pocket proxy that needs no ligand and no docking
  plddt         mean and the fraction of residues above 70, since a low-confidence region is a
                statement about disorder and disorder is functional information

-> colab/data/ml/struct_enzymes.npz
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loop_replication as LR  # noqa: E402

AF = LR.SC / "af"
ACCS = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/enz_accs.json")
OUT = Path("colab/data/ml/struct_enzymes.npz")
AA = list("ACDEFGHIKLMNPQRSTVWY")
THREE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
         "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
         "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}


def parse(path):
    co, res, plddt, isca = [], [], [], []
    for ln in open(path, errors="replace"):
        if not ln.startswith("ATOM"):
            continue
        co.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        res.append(ln[17:20].strip())
        plddt.append(float(ln[60:66]))
        isca.append(ln[12:16].strip() == "CA")
    if not co:
        return None
    return (np.array(co), np.array(res), np.array(plddt), np.array(isca, bool))


def features(co, res, plddt, isca):
    f, nm = [], []

    def add(v, n):
        f.append(float(v))
        nm.append(n)

    ca = co[isca]
    n = len(ca)
    c = ca - ca.mean(0)
    G = (c.T @ c) / n
    ev = np.sort(np.linalg.eigvalsh(G))[::-1]
    rg = float(np.sqrt(ev.sum()))
    add(np.log(rg), "log_rg")
    add(np.log(n), "log_n")
    add(rg / (2.2 * n ** 0.38), "rg_over_globular")
    add(len(co) / n, "atoms_per_res")
    tot = ev.sum()
    add(ev[0] / tot, "gyr_ax1")
    add(ev[1] / tot, "gyr_ax2")
    add(ev[0] - 0.5 * (ev[1] + ev[2]), "asphericity")
    add(ev[1] - ev[2], "acylindricity")

    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    contact = (d < 8.0) & (seq_sep > 2)
    deg = contact.sum(1)
    add(contact.sum() / (2.0 * n), "contact_density")
    add((contact * seq_sep).sum() / max(contact.sum(), 1) / n, "relative_contact_order")
    add(deg.mean(), "deg_mean")
    add(deg.std(), "deg_sd")
    add(np.percentile(deg, 90), "deg_p90")
    add((deg == 0).mean(), "frac_isolated")
    long_c = (d < 8.0) & (seq_sep > 24)
    add(long_c.sum() / (2.0 * n), "long_contact_density")

    # burial: neighbours within 10 A of each CA
    nb = ((d < 10.0).sum(1) - 1).astype(float)
    for t in (12, 18, 24):
        add((nb >= t).mean(), f"frac_buried_{t}")
    surface = nb < 18
    letters = np.array([THREE.get(r, "X") for r in res[isca]])
    for a in AA:
        sel = letters == a
        add(sel.mean(), f"comp_{a}")
        add((sel & surface).sum() / max(surface.sum(), 1), f"surf_{a}")

    # concavity: empty grid cells enclosed by protein along all three axes
    g = 2.0
    q = np.floor((co - co.min(0)) / g).astype(int)
    shape = q.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[q[:, 0], q[:, 1], q[:, 2]] = True
    enclosed = np.ones_like(occ)
    for ax in range(3):
        fwd = np.cumsum(occ, axis=ax) > 0
        bwd = np.flip(np.cumsum(np.flip(occ, ax), axis=ax), ax) > 0
        enclosed &= fwd & bwd
    pocket = enclosed & ~occ
    add(pocket.sum() * g ** 3 / max(occ.sum() * g ** 3, 1), "pocket_vol_ratio")
    add(np.log1p(pocket.sum() * g ** 3), "log_pocket_vol")
    add(occ.sum() * g ** 3 / (4 / 3 * np.pi * rg ** 3), "packing_vs_sphere")

    p = plddt[isca]
    add(p.mean(), "plddt_mean")
    add((p >= 70).mean(), "plddt_frac70")
    add((p < 50).mean(), "plddt_frac_disordered")
    return np.array(f, np.float32), nm


def main():
    t0 = time.time()
    want = json.load(open(ACCS))
    accs, rows, names = [], [], None
    for i, a in enumerate(want):
        p = AF / f"{a}.pdb"
        if not p.exists():
            continue
        got = parse(p)
        if got is None or got[3].sum() < 30:
            continue
        v, nm = features(*got)
        if not np.isfinite(v).all():
            continue
        accs.append(a)
        rows.append(v)
        names = nm
        if len(accs) % 250 == 0:
            print(f"  {len(accs):,}/{len(want):,} [{time.time()-t0:.0f}s]", flush=True)
    X = np.array(rows, np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, accs=np.array(accs), X=X, names=np.array(names))
    print(f"wrote {X.shape} ({len(names)} features) -> {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
