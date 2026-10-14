"""BASELINE MOLECULAR STATE PER CELL LINE -- the input the context tests need, from the right source.

WHY NOT DEPMAP. The obvious source for per-cell-line baseline RNA is DepMap's expression matrix, and it is
the wrong one here for two reasons. RPE1 is not a cancer line and is not in that panel, which would cost one
of only four usable cell lines. And it is a different assay on a different platform from the Perturb-seq
readout the responses come from, so conditioning a Perturb-seq model on a bulk RNA-seq baseline would put an
assay boundary exactly along the axis being tested -- the same defect the environment audit found running
through the mRNA->protein ceiling and the sign-transfer arms.

WHAT THIS USES INSTEAD. Every Perturb-seq deposit here carries its own NON-TARGETING control cells. Averaged,
they are the unperturbed transcriptome of that line, measured on the same instrument, in the same pipeline,
in the same units, in the same experiment as the responses. That is the baseline state, and it is strictly
better than a cross-platform substitute.

WHAT THIS IS NOT ALLOWED TO CONTAIN, because the context tests it feeds are a leave-one-cell-line-out task:
nothing derived from perturbation RESPONSES. The tide vector, per-gene mean |z|, mover frequency and
response-similarity embeddings are all response-derived and all forbidden as context, however convenient. The
only thing extracted here is the control-cell expression profile.

OUTPUT is cached into the repository rather than the scratchpad. This container has restarted five times in
one session and the scratchpad is restored from a snapshot each time, so anything written there during a
session is lost; a 4 x ~8,000 float table is small enough to version and is then immune.
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
CACHE = OUT / "baseline_state.json.gz"
CHUNK = 20_000

# cell line -> h5ad. K562 uses the small Replogle deposit rather than the pseudobulk gwps file, because
# gwps.h5ad is ALREADY a log fold change against control and therefore contains no baseline at all.
# cell line -> h5ad. K562's deposits are PSEUDOBULK (a log fold change per perturbation) and contain no
# control CELLS at all, so its baseline can only come from the deposit's own per-gene mean expression in
# var/mean. The other three are cell-level and do have control cells.
#
# THAT IS A DERIVATION MISMATCH AND IT IS NOT PAPERED OVER. Using control-cell means for three lines and an
# all-cell mean for the fourth would put a processing difference exactly along the axis being tested -- the
# error this project files as `err-provenance-on-the-axis`. So the PRIMARY baseline is the ALL-CELL mean for
# every line, which is identically derived everywhere; where control cells also exist the control-only mean
# is computed as well and the two are correlated, so the substitution is measured rather than assumed.
SOURCES = {
    "K562": ("gwps.h5ad", "pseudobulk"),
    "RPE1": ("rpe1.h5ad", "cells"),
    "HepG2": ("nadig_hepg2.h5ad", "cells"),
    "Jurkat": ("nadig_jurkat.h5ad", "cells"),
}
CONTROL_TOKENS = ("non-targeting", "control")


def cat_codes(f, key):
    import h5py
    o = f["obs"][key]
    if isinstance(o, h5py.Group) and "categories" in o:
        cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
        return cats, o["codes"][:]
    return None, None


def gene_names(f):
    import h5py
    v = f["var"]
    if "gene_name" in v:
        o = v["gene_name"]
        if isinstance(o, h5py.Group) and "categories" in o:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in o["categories"][:]]
            return np.array([cats[i] for i in o["codes"][:]])
        arr = o[:]
        if arr.dtype.kind in "iu" and "__categories" in v:
            cats = [c.decode() if isinstance(c, bytes) else str(c)
                    for c in v["__categories"]["gene_name"][:]]
            return np.array([cats[i] for i in arr])
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    raise SystemExit("no var/gene_name in this deposit")


def extract(line, spec, report):
    """Returns (all_cell_profile, control_only_profile_or_None, note). The first is the primary."""
    import h5py
    fname, kind = spec
    p = SP / fname
    if not p.exists():
        report(f"  {line:8s} ABSENT: {fname} -- reported as unavailable, not substituted")
        return None, None, "absent"
    with h5py.File(p, "r") as f:
        gn = gene_names(f)
        if kind == "pseudobulk":
            if "mean" not in f["var"]:
                report(f"  {line:8s} pseudobulk deposit without var/mean -- unavailable")
                return None, None, "no var/mean"
            mean_expr = f["var"]["mean"][:].astype(np.float64)
            report(f"  {line:8s} PSEUDOBULK: baseline from var/mean over {len(gn):,} genes "
                   f"(no control cells exist in this deposit)")
            prof = np.log2(1.0 + mean_expr / max(mean_expr.sum(), 1e-9) * 1e6)
            return ({g: float(v) for g, v in zip(gn, prof) if v > 0}, None, "var/mean, all cells")

        n, G = f["X"].shape
        cats, codes = cat_codes(f, "gene")
        cmask = None
        if cats is not None:
            ctrl = [i for i, c in enumerate(cats) if any(t in c.lower() for t in CONTROL_TOKENS)]
            if ctrl:
                cmask = np.isin(codes, np.array(ctrl))
        acc_all = np.zeros(G, dtype=np.float64)
        acc_ctl = np.zeros(G, dtype=np.float64) if cmask is not None else None
        n_ctl = 0
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            blk = f["X"][i0:i1]
            acc_all += blk.sum(0)
            if cmask is not None:
                m = cmask[i0:i1]
                if m.any():
                    acc_ctl += blk[m].sum(0)
                    n_ctl += int(m.sum())

    def to_prof(acc):
        cpm = acc / max(acc.sum(), 1.0) * 1e6
        return np.log2(1.0 + cpm)

    pa = to_prof(acc_all)
    pc = to_prof(acc_ctl) if acc_ctl is not None and n_ctl >= 200 else None
    r = float(np.corrcoef(pa, pc)[0, 1]) if pc is not None else float("nan")
    report(f"  {line:8s} CELLS: {n:,} total, {n_ctl:,} non-targeting; all-cell vs control-only "
           f"baseline correlate r {r:.4f}"
           + ("" if pc is not None else "  (too few controls for the control-only arm)"))
    return ({g: float(v) for g, v in zip(gn, pa) if v > 0},
            ({g: float(v) for g, v in zip(gn, pc) if v > 0} if pc is not None else None),
            f"all-cell mean over {n:,} cells; control-only r {r:.4f}")


def main():
    log = []

    def report(s):
        print(s, flush=True)
        log.append(s)

    report("=" * 100)
    report("BASELINE MOLECULAR STATE per cell line, from each deposit's own non-targeting control cells")
    report("=" * 100)
    report("  RESPONSE-DERIVED QUANTITIES ARE FORBIDDEN HERE. The tide, per-gene mean |z| and mover")
    report("  frequency are all functions of the perturbation responses, which is exactly what a")
    report("  leave-one-cell-line-out task must not leak. Only control-cell expression is extracted.")

    prof, ctl, notes = {}, {}, {}
    for line, spec in SOURCES.items():
        a, c, note = extract(line, spec, report)
        notes[line] = note
        if a:
            prof[line] = a
        if c:
            ctl[line] = c

    if len(prof) < 2:
        raise SystemExit(f"only {len(prof)} cell line(s) yielded a baseline; the context tests need at "
                         f"least two and this is reported as a blocked input rather than worked around")

    shared = set.intersection(*[set(v) for v in prof.values()])
    report(f"\n  {len(prof)} cell lines; {len(shared):,} genes with a baseline in ALL of them")
    # A SANITY CHECK THAT THE PROFILES ARE ACTUALLY DIFFERENT CELLS. If two lines' baselines correlate at
    # ~1.0 the extraction has collapsed to one profile and every context test downstream would be measuring
    # nothing while appearing to run.
    sh = sorted(shared)
    M = np.array([[prof[l][g] for g in sh] for l in prof])
    C = np.corrcoef(M)
    names = list(prof)
    report("  pairwise correlation of baseline profiles (must NOT be ~1.0, or the lines are not distinct):")
    for i, a in enumerate(names):
        report("    " + a.ljust(8) + " " + "  ".join(f"{C[i, j]:+.3f}" for j in range(len(names))))
    off = [C[i, j] for i in range(len(names)) for j in range(len(names)) if i != j]
    if min(off) > 0.995:
        raise SystemExit(f"every pair of baselines correlates above 0.995 (min {min(off):.4f}); these are "
                         f"not distinct cell states and the context tests would be vacuous")
    report(f"  off-diagonal range {min(off):+.3f}..{max(off):+.3f} -- distinct")

    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt") as fh:
        json.dump({"source": "non-targeting control cells of each Perturb-seq deposit",
                   "units": "log2(CPM+1) over pooled control cells",
                   "forbidden": "nothing response-derived is included; this is baseline state only",
                   "n_lines": len(prof), "n_shared_genes": len(shared),
                   "derivation_per_line": notes,
                   "control_only_profiles": ctl,
                   "pairwise_corr": {names[i]: {names[j]: float(C[i, j]) for j in range(len(names))}
                                     for i in range(len(names))},
                   "profiles": prof, "log": log}, fh)
    report(f"\n  -> {CACHE}  ({CACHE.stat().st_size/1e6:.1f} MB, versioned so a container restart cannot "
           f"cost it)")


if __name__ == "__main__":
    main()
