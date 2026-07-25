"""build_rpe1_profiles -- turn the per-cell RPE1 Perturb-seq matrix into per-knockout z-scored profiles.

WHY. pair_retrieval_audit showed that 88% of the gap between the retrieval model (0.385) and its architecture ceiling (0.548) is RANKING:
the genuinely most-similar train knockout lands at median rank 10 of ~235 and reaches the top 10 only half the time. Improving how
retrieved profiles are blended is worth 12%; improving the SIMILARITY MODEL is worth the rest. The similarity model is trained on pairs,
and pairs scale quadratically, so the cheapest way to improve it is more knockouts to pair up -- and RPE1 is already on disk.

WHAT THIS DOES AND DOES NOT NEED TO MATCH. The retrieval step must stay over K562 profiles, because averaging RPE1 responses to predict a
K562 response would need cell-type conditioning this project does not have. RPE1 is used ONLY to generate additional PAIR LABELS: "these
two genes, knocked out in RPE1, produced responses this similar". The label is a CORRELATION between two profiles within one cell line, and
correlation is invariant to per-gene scaling, so RPE1 only has to be internally consistent -- it does not have to sit on K562's scale. That
removes most of the normalisation risk in reusing a differently-processed dataset.

METHOD. Cells are grouped by their targeted gene, summed in chunks (the matrix is 247,914 x 8,749 and will not fit in memory), divided to
per-knockout mean profiles, then each gene column is robustly standardised across knockouts (median / MAD * 1.4826) so a value is "how
unusual is this gene's level in this knockout relative to all knockouts". Knockouts with fewer than MINCELLS cells are dropped as too
noisy to give a trustworthy pair label. Deterministic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SRC = SP / "rpe1.h5ad"
DST = SP / "rpe1_ko_zscores.parquet"
CHUNK = 8000
MINCELLS = 25


def cat(h, key):
    g = h["obs"][key]
    if isinstance(g, h5py.Dataset):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    codes = g["codes"][:]
    return np.array([cats[c] if 0 <= c < len(cats) else "?" for c in codes])


def main():
    h = h5py.File(SRC, "r")
    X = h["X"]; ncell, ngene = X.shape
    vg = h["var"]["gene_name"]
    if isinstance(vg, h5py.Group):
        vcats = [x.decode() if isinstance(x, bytes) else str(x) for x in vg["categories"][:]]
        gnames = [vcats[c] for c in vg["codes"][:]]
    else:
        gnames = [x.decode() if isinstance(x, bytes) else str(x) for x in vg[:]]
    pert = cat(h, "perturbation") if "perturbation" in h["obs"] else cat(h, "gene")
    print(f"RPE1: {ncell} cells x {ngene} genes | {len(set(pert))} distinct perturbation labels")

    labs = sorted(set(pert)); li = {l: i for i, l in enumerate(labs)}
    code = np.array([li[p] for p in pert], dtype=np.int32)
    S = np.zeros((len(labs), ngene), dtype=np.float64)
    C = np.zeros(len(labs), dtype=np.int64)
    for a in range(0, ncell, CHUNK):
        b = min(a + CHUNK, ncell)
        blk = np.asarray(X[a:b], dtype=np.float64)
        blk[~np.isfinite(blk)] = 0.0
        np.add.at(S, code[a:b], blk)
        np.add.at(C, code[a:b], 1)
        if (a // CHUNK) % 5 == 0:
            print(f"  {b}/{ncell}", flush=True)
    keepl = [i for i in range(len(labs)) if C[i] >= MINCELLS]
    M = S[keepl] / C[keepl][:, None]
    kept = [labs[i] for i in keepl]
    print(f"kept {len(kept)} perturbations with >= {MINCELLS} cells (of {len(labs)})")

    # robust per-gene standardisation ACROSS knockouts: "how unusual is this gene here, relative to every other knockout"
    med = np.median(M, axis=0)
    mad = np.median(np.abs(M - med), axis=0) * 1.4826
    good = mad > 1e-9
    Zs = np.zeros_like(M)
    Zs[:, good] = (M[:, good] - med[good]) / mad[good]
    Zs = np.clip(Zs, -50, 50)
    df = pd.DataFrame(Zs[:, good].astype(np.float32),
                      index=kept, columns=[gnames[i] for i in np.where(good)[0]])
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_parquet(DST)
    nz = (np.abs(df.values) >= 4.2).sum(1)
    print(f"wrote {DST.name}: {df.shape[0]} knockouts x {df.shape[1]} genes")
    print(f"  movers at |z|>=4.2 per knockout: median {int(np.median(nz))}, "
          f">=20 movers: {int((nz >= 20).sum())}, >=5: {int((nz >= 5).sum())}")
    json.dump({"n_knockouts": int(df.shape[0]), "n_genes": int(df.shape[1]),
               "median_movers": int(np.median(nz)), "n_with_20_movers": int((nz >= 20).sum()),
               "min_cells": MINCELLS}, open(SP / "rpe1_profiles_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
