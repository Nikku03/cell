"""build_uncensored -- rebuild DENSE, uncensored knockout z-score matrices for the donor cell lines from their raw single-cell data.

Why this matters: every cross-line result so far (multiline_network's 4.6% conservation and 0.546 lookup, edge_transformer's gene descriptions)
used the harness pickle, which stores only the TOP ~250 GENES PER KNOCKOUT. When K562 was uncensored, its GATA1 row went from 250 movers to 637
-- the store was showing ~39% of the response. The donor lines were never uncensored, and edge_transformer's ablation showed those donor profiles
carry half the model's signal, so they are worth rebuilding properly.

PIPELINE (per line, from the raw counts):
  1. accumulate per-knockout summed counts by streaming the cell x gene matrix in chunks (never materialising it)
  2. CP10k-normalise then log1p each pseudobulk
  3. z-score against the CONTROL cells -- and crucially estimate the control standard deviation AT MATCHED CELL DEPTH: a knockout pseudobulked
     from 72 cells is noisier than one from 500, so control pseudobulks are resampled at the same n (binned) to give a per-gene, per-depth sigma.
     Without this, low-cell knockouts get systematically inflated z-scores and look like hubs.
  4. write a dense (knockouts x genes) float32 matrix

Only knockouts with >= MINCELL cells are kept. Deterministic (fixed seed). Usage: python build_uncensored.py <line> [<line> ...]"""
import sys, json, collections
from pathlib import Path
import numpy as np
import h5py
import os

SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
FILES = {"RPE1": "rpe1.h5ad", "HepG2": "nadig_hepg2.h5ad", "Jurkat": "nadig_jurkat.h5ad"}
MINCELL = 50
NBOOT = 30          # control pseudobulks per depth bin
CHUNK = 40000
RNG = np.random.RandomState(0)


def decode(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def build(line):
    f = SP / FILES[line]
    with h5py.File(f, "r") as h:
        pert = h["obs"]["perturbation"]
        cats = decode(pert["categories"][:]); codes = pert["codes"][:]
        genes = decode(h["var"]["gene_name"][:]) if "gene_name" in h["var"] else decode(h["var"]["_index"][:])
        X = h["X"]; ncell, ngene = X.shape
        ctrl_code = next((i for i, c in enumerate(cats) if c.lower() == "control"), None)
        if ctrl_code is None:
            print(f"{line}: no control group"); return None
        counts = collections.Counter(codes.tolist())
        keep = [i for i in range(len(cats)) if i != ctrl_code and counts[i] >= MINCELL]
        print(f"[{line}] {ncell:,} cells x {ngene:,} genes | {len(keep):,} knockouts with >={MINCELL} cells | "
              f"{counts[ctrl_code]:,} control cells")

        # scatter-add per perturbation via a sparse indicator matmul: one BLAS call per chunk instead of ~2400 masked copies
        from scipy.sparse import csr_matrix
        S = np.zeros((len(cats), ngene), np.float64)
        ctrl_rows = []
        for start in range(0, ncell, CHUNK):
            end = min(start + CHUNK, ncell)
            blk = X[start:end].astype(np.float32)
            cb = codes[start:end]; m = end - start
            ind = csr_matrix((np.ones(m, np.float32), (cb.astype(np.int64), np.arange(m))), shape=(len(cats), m))
            S += ind.dot(blk)
            mc = cb == ctrl_code
            if mc.any():
                ctrl_rows.append(blk[mc].copy())
        C = np.vstack(ctrl_rows); del ctrl_rows
        print(f"  control cell matrix {C.shape}")

    def norm(v):
        t = v.sum()
        return np.log1p(v / (t if t > 0 else 1.0) * 1e4)

    ctrl_mean = norm(C.sum(0).astype(np.float64))
    # control sigma at MATCHED depth: resample n control cells, pseudobulk, per-gene sd -- binned by n
    ns = sorted({int(counts[i]) for i in keep})
    bins = np.unique(np.clip(np.round(np.logspace(np.log10(max(MINCELL, min(ns))), np.log10(max(ns)), 12)), 1, None).astype(int))
    sigma = {}
    for n in bins:
        n_use = int(min(n, C.shape[0]))
        boots = np.stack([norm(C[RNG.choice(C.shape[0], n_use, replace=False)].sum(0).astype(np.float64))
                          for _ in range(NBOOT)])
        sigma[n] = boots.std(0) + 1e-6
    def sig_for(n):
        return sigma[bins[np.argmin(np.abs(bins - n))]]

    kos = [cats[i] for i in keep]
    Z = np.zeros((len(keep), len(genes)), np.float32)
    for r, i in enumerate(keep):
        Z[r] = ((norm(S[i]) - ctrl_mean) / sig_for(counts[i])).astype(np.float32)
    out = SP / f"unc_{line}.npz"
    np.savez_compressed(out, kos=np.array(kos), genes=np.array(genes), Z=Z,
                        ncells=np.array([counts[i] for i in keep]))
    mv = (np.abs(Z) >= 4.0).sum(1)
    print(f"  -> {out.name}  {Z.shape}  | movers/KO at |z|>=4: median {int(np.median(mv))}, "
          f"KOs with >=20: {(mv>=20).sum()} ({100*(mv>=20).mean():.0f}%)")
    return out


def main():
    lines = sys.argv[1:] or list(FILES)
    for ln in lines:
        if ln in FILES:
            build(ln)


if __name__ == "__main__":
    main()
