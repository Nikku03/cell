"""Emit per-gene chromatin features for EVERY gene with an hg19 TSS, and cache them.

Loops 95 and 96 each streamed the GM12878 Hi-C and computed A/B compartment, TSS insulation and
local contact density, then threw the per-gene values away and kept only summary statistics. Loop
95 took 1,223 s to do it. This writes them to disk once so any downstream loop can join to them
without re-streaming, and so the feature vector a later loop tests is the SAME one, byte for byte.

Nothing is fitted here and no gate is declared -- it is a cache, not a test.

-> scratchpad/_chromatin_features.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_replication as LR   # noqa: E402
import loop_second as L77       # noqa: E402
import loop_genome as L86       # noqa: E402
from loop_hic_target import insulation  # noqa: E402

SC = LR.SC
BIN = L77.BIN
OUTF = SC / "_chromatin_features.json"


def compartment_pc1(M, mask):
    """A/B from the map's own correlation-matrix PC1 -- the standard call, no GC involved."""
    A = M[np.ix_(mask, mask)]
    A = np.log1p(A)
    A = A - np.nanmean(A, axis=0, keepdims=True)
    A = np.nan_to_num(A)
    Cm = np.corrcoef(A)
    Cm = np.nan_to_num(Cm)
    w, v = np.linalg.eigh(Cm)
    pc = v[:, -1]
    out = np.full(len(mask), np.nan)
    out[mask] = pc
    return out


def main():
    t0 = time.time()
    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    mrna = {}
    S = json.load(open(SC / "_schwan2011.json"))
    for g, v in S.items():
        if v.get("mrna_copies"):
            mrna[g] = v["mrna_copies"]

    tss = {}
    for ln in open(SC / "_tss_hg19.bed"):
        f = ln.split()
        i = int(f[3][1:])
        if i < len(names):
            tss[names[i]] = (f[0], int(f[1]))
    print(f"{len(tss):,} genes with an hg19 TSS", flush=True)

    rows = {}
    for ch in L86.CHROMS:
        n = L86.HG19[ch] // BIN + 1
        gl = [g for g in tss if tss[g][0] == ch and tss[g][1] // BIN < n]
        if len(gl) < 5:
            continue
        try:
            M = L86.fetch_hic(ch, n)
        except Exception as e:
            print(f"  {ch:6s} FETCH FAILED {repr(e)[:60]}", flush=True)
            continue
        M[M == 0] = np.nan
        mask = np.isfinite(M).sum(1) > 50
        pc = compartment_pc1(M, mask)
        ins = insulation(M)
        w = int(1e6 // BIN)
        dens = np.full(n, np.nan)
        for b in range(n):
            if not mask[b]:
                continue
            lo, hi = max(0, b - w), min(n, b + w + 1)
            seg = M[b, lo:hi]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 5:
                dens[b] = float(seg.sum())
        got = 0
        for g in gl:
            b = tss[g][1] // BIN
            if not mask[b]:
                continue
            rows[g] = {"chrom": ch, "bin": int(b), "pc1": float(pc[b]),
                       "ins": float(ins[b]) if np.isfinite(ins[b]) else None,
                       "dens": float(dens[b]) if np.isfinite(dens[b]) else None}
            got += 1
        print(f"  {ch:6s} {len(gl):5d} genes, {got:5d} on mappable bins   "
              f"[{time.time() - t0:.0f}s]", flush=True)
        del M

    # PC1's sign is arbitrary per chromosome; orient it so A (higher mRNA) is positive.
    # This is the same convention loop 95 used, and it is a convention rather than a fit.
    for ch in sorted({rows[g]["chrom"] for g in rows}):
        sub = [g for g in rows if rows[g]["chrom"] == ch and g in mrna]
        if len(sub) < 10:
            continue
        x = np.array([rows[g]["pc1"] for g in sub], float)
        y = np.log1p(np.array([mrna[g] for g in sub], float))
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean()
        ry -= ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        r = float((rx * ry).sum() / den) if den > 0 else 0.0
        if r < 0:
            for g in [g for g in rows if rows[g]["chrom"] == ch]:
                rows[g]["pc1"] = -rows[g]["pc1"]

    json.dump({"n": len(rows), "bin": BIN, "source": "GM12878 in situ primary MAPQ30, hg19",
               "features": rows, "seconds": time.time() - t0},
              open(OUTF, "w"))
    print(f"wrote {len(rows):,} genes -> {OUTF}   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
