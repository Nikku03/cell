"""Binding-site labels and local structural environments. A cache, not a test -- no gates.

WHY THIS EXISTS INSTEAD OF DOCKING. Docking asks "does this whole protein fit this whole partner",
which the nexus arm measured at AUC [0.450, 0.549] against a size control at 0.532 for 92% of its
compute. This asks a smaller and much better-posed question: which RESIDUES form a site, and what
NON-PROTEIN COMPOUND sits in it. The compound is a 448-way label rather than a 5,000-way partner
search, and UniProt annotates it at residue resolution with a ChEBI identifier.

THE CHEBI IDENTIFIER MATTERS BEYOND THIS LOOP. Loop's earlier expression join matched Reactome
entities to Human-GEM species BY NAME and got 2.2%. UniProt gives ligand_id="ChEBI:CHEBI:57540",
and Human-GEM annotates its species with ChEBI too, so this is the key that join actually needed.

ALPHAFOLD MODELS CONTAIN NO LIGANDS. That is why the labels come from UniProt and not from the
structures: an AlphaFold monomer has no HETATM records at all, so a site is only knowable here
because somebody annotated it. The structures supply geometry, the annotations supply truth, and
neither can supply the other.

WHAT IS EXTRACTED PER RESIDUE, from the CA/CB geometry of the cached monomer:
  burial       neighbour counts at 8/10/12 A, and the rank of that count within the protein
  environment amino-acid composition of the 10 A neighbourhood (20 numbers) -- what chemistry is
               presented at this spot, which is the thing a ligand actually sees
  concavity    how far the residue sits inside the convex hull direction, plus local sphere
               occupancy at 6 and 9 A -- a pocket lives where protein surrounds empty space
  geometry     distance from the centroid over the radius of gyration, and the local CA density
  confidence   the residue's own pLDDT and the mean pLDDT of its neighbourhood

-> colab/data/ml/sites.npz
"""
import gzip
import json
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loop_replication as LR  # noqa: E402

SITES = Path("colab/data/uniprot_sites.tsv.gz")
AF = LR.SC / "af"
OUT = Path("colab/data/ml/sites.npz")
AA = list("ACDEFGHIKLMNPQRSTVWY")
THREE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
         "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
         "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}
BIND_RE = re.compile(
    r'BINDING (\d+)(?:\.\.(\d+))?;[^;]*?/ligand="([^"]+)";\s*/ligand_id="ChEBI:(CHEBI:\d+)"')


def parse_annotations(hum):
    """UniProt BINDING (with a ChEBI ligand) and ACT_SITE, restricted to human."""
    out = {}
    with gzip.open(SITES, "rt", errors="replace") as f:
        f.readline()
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] not in hum:
                continue
            b = [(int(a), lig, ch) for a, _, lig, ch in BIND_RE.findall(p[3])]
            a = [int(x) for x in re.findall(r"ACT_SITE (\d+)", p[2])]
            if b or a:
                out[p[0]] = {"binding": b, "act": a, "ec": p[4], "seq": p[1]}
    return out


def parse_struct(path):
    ca, cb, res, pl = [], [], [], []
    seen = {}
    for ln in open(path, errors="replace"):
        if not ln.startswith("ATOM"):
            continue
        at = ln[12:16].strip()
        if at not in ("CA", "CB"):
            continue
        ri = int(ln[22:26])
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        if at == "CA":
            seen[ri] = len(ca)
            ca.append(xyz)
            res.append(ln[17:20].strip())
            pl.append(float(ln[60:66]))
            cb.append(xyz)
        elif ri in seen:
            cb[seen[ri]] = xyz
    if len(ca) < 30:
        return None
    return np.array(ca), np.array(cb), np.array(res), np.array(pl), sorted(seen)


def residue_features(ca, cb, res, pl):
    n = len(ca)
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    cen = ca.mean(0)
    rg = float(np.sqrt(((ca - cen) ** 2).sum(1).mean()))
    letters = np.array([THREE.get(r, "X") for r in res])
    feats = []
    nb8 = (d < 8).sum(1) - 1.0
    nb10 = (d < 10).sum(1) - 1.0
    nb12 = (d < 12).sum(1) - 1.0
    rank10 = np.argsort(np.argsort(nb10)) / max(n - 1, 1)
    dcen = np.linalg.norm(ca - cen, axis=1) / max(rg, 1e-6)
    # concavity: how much of the sphere around the residue is occupied, near and far
    occ6 = (d < 6).sum(1) / (4 / 3 * np.pi * 6 ** 3) * 1000.0
    occ9 = (d < 9).sum(1) / (4 / 3 * np.pi * 9 ** 3) * 1000.0
    # direction test: does the CA->CB vector point INTO the protein (buried pocket) or out
    v = cb - ca
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    v = np.divide(v, np.maximum(nv, 1e-6))
    out = (ca - cen) / np.maximum(np.linalg.norm(ca - cen, axis=1, keepdims=True), 1e-6)
    cbdir = (v * out).sum(1)
    plm = np.array([pl[d[i] < 10].mean() for i in range(n)])
    comp = np.zeros((n, 20))
    for i in range(n):
        m = d[i] < 10
        c = Counter(letters[m])
        tot = max(m.sum(), 1)
        for k, a in enumerate(AA):
            comp[i, k] = c.get(a, 0) / tot
    base = np.column_stack([nb8, nb10, nb12, rank10, dcen, occ6, occ9, cbdir, pl, plm,
                            np.log(n) * np.ones(n)])
    names = (["nb8", "nb10", "nb12", "rank_nb10", "dist_centroid_over_rg", "occ6", "occ9",
              "cb_points_in", "plddt", "plddt_nbhd10", "log_len"]
             + [f"nbhd_{a}" for a in AA] + [f"self_{a}" for a in AA])
    selfaa = np.zeros((n, 20))
    for k, a in enumerate(AA):
        selfaa[:, k] = (letters == a).astype(float)
    return np.hstack([base, comp, selfaa]).astype(np.float32), names


def main():
    t0 = time.time()
    hum = set()
    a2g = {}
    with gzip.open(LR.SC / "human_proteome.fasta.gz", "rt", errors="replace") as f:
        for ln in f:
            if ln.startswith(">"):
                m = re.match(r">\w\w\|([^|]+)\|", ln)
                g = re.search(r"GN=(\S+)", ln)
                if m:
                    hum.add(m.group(1))
                    if g:
                        a2g[m.group(1)] = g.group(1)
    ann = parse_annotations(hum)
    print(f"{len(ann):,} human proteins with a BINDING or ACT_SITE annotation "
          f"[{time.time()-t0:.0f}s]", flush=True)

    have = {p.stem for p in AF.glob("*.pdb")}
    todo = sorted(set(ann) & have)
    print(f"{len(todo):,} of them have a cached AlphaFold monomer", flush=True)

    accs, ridx, X, ybind, yact, ylig, names = [], [], [], [], [], [], None
    for k, a in enumerate(todo):
        got = parse_struct(AF / f"{a}.pdb")
        if got is None:
            continue
        ca, cb, res, pl, resnums = got
        F, names = residue_features(ca, cb, res, pl)
        pos = {r: i for i, r in enumerate(resnums)}
        bset, ligof = set(), {}
        for r, lig, ch in ann[a]["binding"]:
            if r in pos:
                bset.add(pos[r])
                ligof[pos[r]] = ch
        aset = {pos[r] for r in ann[a]["act"] if r in pos}
        for i in range(len(ca)):
            accs.append(a)
            ridx.append(i)
            X.append(F[i])
            ybind.append(1 if i in bset else 0)
            yact.append(1 if i in aset else 0)
            ylig.append(ligof.get(i, ""))
        if (k + 1) % 200 == 0:
            print(f"  {k+1:,}/{len(todo):,} proteins, {len(X):,} residues "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    X = np.array(X, np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, accs=np.array(accs), residue=np.array(ridx, np.int32), X=X,
        is_binding=np.array(ybind, np.int8), is_active=np.array(yact, np.int8),
        ligand=np.array(ylig), names=np.array(names))
    print(f"wrote {X.shape} residues, {int(np.sum(ybind)):,} binding, "
          f"{int(np.sum(yact)):,} active -> {OUT} [{time.time()-t0:.0f}s]", flush=True)
    json.dump({a: {"binding": [[r, l, c] for r, l, c in v["binding"]], "act": v["act"],
                   "ec": v["ec"], "gene": a2g.get(a, "")} for a, v in ann.items()},
              open("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/"
                   "scratchpad/uniprot_binding.json", "w"))


if __name__ == "__main__":
    main()
