"""DENSE PER-CELL RESPONSE MATRICES -- the prerequisite the contrast tests are blocked on.

WHY. `colab/contrast_context.py` is void on the nlz benches: those store only each knockout's top ~250 genes,
so 96.5% of every response matrix is exactly zero and a zero means NOT RECORDED rather than "no response".
Measured on the shared four-line rectangle, the number of genes jointly observed in ALL FOUR cells for the
same perturbation has a median of 3, 26% of perturbations have no overlap at all, and the whole rectangle
yields ~9,700 usable points. A cross-cell contrast Y[a,p,g] - Y[b,p,g] cannot be built from that at any
number of cell lines, because the recorded gene sets barely intersect.

The full data was never missing -- it was thrown away by the bench format. Three of the four deposits are
CELL-LEVEL and can be pseudobulked to a complete perturbation x gene matrix, and the fourth is already one.

    K562     gwps.h5ad            ALREADY a dense pseudobulk log fold change, 11,258 x 8,248
    RPE1     rpe1.h5ad            247,914 cells -> pseudobulk
    HepG2    nadig_hepg2.h5ad     145,473 cells -> pseudobulk
    Jurkat   nadig_jurkat.h5ad    262,956 cells -> pseudobulk

CONSTRUCTION, identical for the three cell-level deposits and matching what colab/causal_reg.py already does
for RPE1 so the two agree: sum raw counts per perturbation, CPM-normalise, log2(1+x), subtract the
non-targeting control profile. Then a per-gene robust z, (x - median) / (1.4826 * MAD), which is the
project's convention everywhere else -- median and MAD rather than mean and sd because the column is exactly
where a few enormous effects live and letting them set the scale would shrink them toward the noise.

K562 IS NOT RE-DERIVED and that is a deliberate asymmetry, recorded rather than hidden. Its deposit is
already a pseudobulk log fold change built by its authors; re-deriving it from a cell-level file this project
does not have would be impossible, and treating the two pipelines as interchangeable is the error this
project files as `err-provenance-on-the-axis`. What is checked instead is that the resulting matrices behave
alike -- the report prints per-line density, robust-z spread and the correlation of gene-level response
variance across lines, so a K562 that does not look like the others is visible rather than assumed away.

THE NUMBER THIS MODULE EXISTS TO MOVE is density: 3.5% on the benches. Anything below ~50% here and the
contrast target is still mostly absent and the tests stay blocked.

OUTPUT is the shared rectangle only -- perturbations and genes present in all four lines -- as float16,
because the full four-line tensor is not something to carry around and the contrast tests need exactly the
intersection.
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DENSE = SP / "dense_response.npz"
CHUNK = 20_000
MIN_CELLS = 25
DENSITY_BAR = 0.50

SOURCES = {
    "K562": ("gwps.h5ad", "pseudobulk"),
    "RPE1": ("rpe1.h5ad", "cells"),
    "HepG2": ("nadig_hepg2.h5ad", "cells"),
    "Jurkat": ("nadig_jurkat.h5ad", "cells"),
}


def gene_names(f):
    import h5py
    v = f["var"]
    o = v["gene_name"]
    if isinstance(o, h5py.Group) and "categories" in o:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        return np.array([cats[i] for i in o["codes"][:]])
    arr = o[:]
    if arr.dtype.kind in "iu" and "__categories" in v:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in v["__categories"]["gene_name"][:]]
        return np.array([cats[i] for i in arr])
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def robust_z(M):
    med = np.nanmedian(M, axis=0)
    mad = np.nanmedian(np.abs(M - med), axis=0) * 1.4826
    mad = np.where(mad < 1e-6, np.nan, mad)
    return (M - med) / mad


def build_pseudobulk(path, report, label):
    """Sum raw counts per perturbation, CPM, log2, minus the non-targeting profile."""
    import h5py
    with h5py.File(path, "r") as f:
        o = f["obs"]["gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        codes = o["codes"][:]
        gn = gene_names(f)
        n, G = f["X"].shape
        S = np.zeros((len(cats), G), dtype=np.float64)
        C = np.zeros(len(cats), dtype=np.int64)
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            blk = f["X"][i0:i1]
            np.add.at(S, codes[i0:i1], blk)
            np.add.at(C, codes[i0:i1], 1)
            if (i0 // CHUNK) % 5 == 0:
                report(f"      {label} pseudobulk {i1:,}/{n:,} cells")
    lg = np.log2(1.0 + S / np.maximum(S.sum(1, keepdims=True), 1) * 1e6)
    ctrl = [i for i, c in enumerate(cats) if "non-targeting" in c.lower() or c.lower() == "control"]
    if not ctrl:
        raise SystemExit(f"{label}: no non-targeting category; cannot form a log fold change")
    base = lg[ctrl].mean(0)
    keep = (C >= MIN_CELLS) & ~np.isin(np.arange(len(cats)), ctrl)
    lfc = (lg[keep] - base).astype(np.float32)
    perts = np.array(cats)[keep]
    report(f"    {label:8s} {lfc.shape[0]:,} perturbations x {lfc.shape[1]:,} genes "
           f"(>= {MIN_CELLS} cells; {len(ctrl)} control categories, {C[ctrl].sum():,} control cells)")
    return lfc, perts, gn


def load_gwps(report):
    import h5py
    with h5py.File(SP / "gwps.h5ad", "r") as f:
        X = f["X"][:]
        pert = [s.decode() if isinstance(s, bytes) else str(s) for s in f["obs"]["gene_transcript"][:]]
        gn = gene_names(f)
    psym = np.array([p.split("_")[1] if len(p.split("_")) > 2 else p for p in pert])
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    report(f"    K562     {X.shape[0]:,} perturbations x {X.shape[1]:,} genes "
           f"(deposit is ALREADY a pseudobulk log fold change; not re-derived)")
    return X, psym, gn


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("DENSE RESPONSE MATRICES -- rebuilding what the top-250 bench format threw away")
    report("=" * 100)
    report("  the benches are 3.5% dense; below ~50% here the contrast target is still absent")

    mats, perts, gnames = {}, {}, {}
    for line, (fn, kind) in SOURCES.items():
        p = SP / fn
        if not p.exists():
            report(f"    {line:8s} ABSENT: {fn} -- excluded, not substituted")
            continue
        if kind == "pseudobulk":
            m, pr, gn = load_gwps(report)
        else:
            m, pr, gn = build_pseudobulk(p, report, line)
        mats[line], perts[line], gnames[line] = m, pr, gn

    lines = sorted(mats)
    if len(lines) < 3:
        raise SystemExit(f"only {len(lines)} lines built; the contrast tests need at least 3")

    shared_p = sorted(set.intersection(*[set(perts[l]) for l in lines]))
    shared_g = sorted(set.intersection(*[set(gnames[l]) for l in lines]))
    report(f"\n  shared rectangle: {len(lines)} lines x {len(shared_p):,} perturbations x "
           f"{len(shared_g):,} genes")
    if not shared_p or not shared_g:
        raise SystemExit("empty intersection; the deposits do not share a common grid")

    Z = np.zeros((len(lines), len(shared_p), len(shared_g)), dtype=np.float32)
    dens = {}
    for i, l in enumerate(lines):
        pi = {s: j for j, s in enumerate(perts[l])}
        gj = {s: j for j, s in enumerate(gnames[l])}
        rows = np.array([pi[s] for s in shared_p])
        cols = np.array([gj[s] for s in shared_g])
        sub = mats[l][np.ix_(rows, cols)]
        z = robust_z(sub)
        Z[i] = z
        dens[l] = float(np.isfinite(z).mean())
        report(f"    {l:8s} density {100*dens[l]:5.1f}%   |z| median {np.nanmedian(np.abs(z)):.3f}   "
               f"p99 {np.nanpercentile(np.abs(z), 99):.2f}")

    # THE CHECK THAT DECIDES WHETHER THIS UNBLOCKS ANYTHING.
    fin = np.isfinite(Z).all(0)
    joint = fin.sum(1)
    report(f"\n  JOINTLY OBSERVED IN ALL {len(lines)} LINES, per perturbation: "
           f"median {np.median(joint):,.0f}, mean {joint.mean():,.1f}, min {joint.min():,}")
    report(f"  total usable observations {int(fin.sum()):,}  "
           f"(the top-250 benches gave ~9,700 over the same four lines)")
    ok = float(np.mean(list(dens.values()))) >= DENSITY_BAR

    # K562 came from a different pipeline; check it behaves like the others rather than assuming it.
    gv = np.array([np.nanvar(Z[i], axis=0) for i in range(len(lines))])
    C = np.corrcoef(np.nan_to_num(gv))
    report(f"\n  cross-line correlation of per-gene response VARIANCE (K562 is the odd pipeline out):")
    for i, l in enumerate(lines):
        report("    " + l.ljust(8) + " " + "  ".join(f"{C[i, j]:+.3f}" for j in range(len(lines))))

    SP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DENSE, Z=Z.astype(np.float16), lines=np.array(lines),
                        perts=np.array(shared_p), genes=np.array(shared_g))
    size = DENSE.stat().st_size / 1e6
    report(f"\n  -> {DENSE}  ({size:.0f} MB, float16)")

    R = {"model": "dense-response-v1",
         "purpose": "unblock colab/contrast_context.py, which is void on the 3.5%-dense nlz benches",
         "lines": lines, "n_perturbations": len(shared_p), "n_genes": len(shared_g),
         "density_per_line": dens, "mean_density": float(np.mean(list(dens.values()))),
         "jointly_observed_per_perturbation": {"median": float(np.median(joint)),
                                               "mean": float(joint.mean()), "min": int(joint.min())},
         "total_usable_observations": int(fin.sum()),
         "benches_gave": 9739,
         "gain_factor": float(fin.sum() / 9739.0),
         "unblocked": bool(ok),
         "cache": str(DENSE), "cache_mb": size,
         "cross_line_variance_correlation": {lines[i]: {lines[j]: float(C[i, j])
                                                        for j in range(len(lines))}
                                             for i in range(len(lines))},
         "limits": [
             "K562 is NOT re-derived: its deposit is already a pseudobulk log fold change from its own "
             "authors' pipeline, while the other three are pseudobulked here. That asymmetry is real and "
             "the cross-line variance correlation above is what makes it inspectable rather than assumed",
             "the three rebuilt lines use a single pooled non-targeting profile per deposit as the "
             "reference, so batch structure within a deposit is not removed",
             "perturbations with fewer than 25 cells are dropped, which removes the weakest-powered rows "
             "and therefore makes the retained set easier than the full deposit",
             "the cache is written to the scratchpad, which this container restores from a snapshot on "
             "restart. It is regenerable from this module in one run and is deliberately NOT committed: "
             "at float16 the shared rectangle is still far too large to version",
             "robust z is computed per gene WITHIN each line, so a gene that is uniformly unresponsive in "
             "one line gets a NaN column there and drops out of the joint-observation count",
         ],
         "verdict": (f"{'UNBLOCKED' if ok else 'STILL BLOCKED'}: mean density "
                     f"{100*np.mean(list(dens.values())):.1f}% against the benches' 3.5%, and "
                     f"{int(fin.sum()):,} observations jointly measured in all {len(lines)} lines against "
                     f"~9,700 before -- a factor of {fin.sum()/9739.0:.0f}. Genes jointly observed per "
                     f"perturbation: median {np.median(joint):,.0f} against 3."),
         "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "dense_response.json", "w"), indent=1)
    report(f"\n  VERDICT: {R['verdict']}")
    report(f"  -> {OUT/'dense_response.json'}")


if __name__ == "__main__":
    main()
