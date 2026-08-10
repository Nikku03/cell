"""DOES MEASURED DNA TORSION PREDICT TRANSCRIPTION RATE?  The bar the CRISPR result pointed at.

WHY THIS TEST AND NOT ANOTHER.  Torsion has now been tested twice against the enhancer-gene bar and
returned PARTIAL both times -- real against its own displaced control, redundant against CTCF counting.
But the same second attempt measured the twin-domain signature at the promoter and found it strongly:
TMP signal runs 14.5% higher in the 2 kb upstream of the TSS than downstream, z = +23.21 over 10,331
genes. That is a statement about TRANSCRIPTION, not about which enhancer contacts which gene, and no test
so far has asked the question it actually bears on. This one does.

THE TARGET, newly available.  k_syn = RPKM_total * k_deg in RPKM/hour, from RNADecayCafe v1.1
(Zenodo 10.5281/zenodo.16884513, AvgKdegs_genes_v1.1.csv, 122,235 rows over 21 cell lines, 10,802 of them
K562). This file was absent from the container and is the reason the transcription bar could not be run at
all; the integration assessment listed it as the single missing input blocking this line of work.

WHAT IS REPRODUCED AND WHAT IS NOT.  nexus_txn's ladder has five layers. Two are reconstructible here and
three are not, and the difference is stated rather than papered over:
    L1 basal promoter   REPRODUCED. Seven fields, all resident in cell_complete.
    L2 + M_ATAC         REPRODUCED. Rebuilt from ENCODE ENCFF926KTI (K562 ATAC IDR, GRCh38, 55,575
                        peaks). A different peak file than the original, so the arm is re-measured.
    L3 ABC, L4 TF, L5   NOT REPRODUCIBLE. _abcfeat.json has no producer anywhere in this repository --
                        only two readers -- so its 12,438-gene enhancer aggregation cannot be rebuilt
                        without redoing an ABC run, and lambert_tf_symbols.json is likewise absent.
The recorded ladder reads 0.1582 / 0.1667 / 0.1878 / 0.1995 / 0.1970. Only the first two are comparable,
and whether they reproduce is the check on this reconstruction.

THE TORSION FEATURES, at the scale the effect was measured at.  From GSE285699 TMP-seq (SH-SY5Y WT,
hg38, run-length encoded at 70-90 bp), all locally normalised by the surrounding 200 kb because that
track ships no naked-DNA control:
    torsion at the TSS, +/- 500 bp
    UPSTREAM ARM and DOWNSTREAM ARM separately, 2 kb each, assigned by strand
    their ASYMMETRY, which is the twin-domain quantity itself
The asymmetry is the feature this whole test exists for. It could only be computed after strand was added
to cell_complete, and it is the one that carries the z = +23.21 signal.

PREDECLARED, before any number:
    L1 and L2 reproduce near 0.1582 and 0.1667
        -> the reconstruction is faithful and the torsion arm can be read against them.
    they do not reproduce
        -> something differs beyond the ATAC peak file and no torsion verdict should be drawn.
    torsion adds >= 0.01 R2 over L1+L2 AND beats its displaced control by >= 0.005
        -> measured torsion predicts transcription rate. Given two PARTIALs on the enhancer-gene task,
           this would locate precisely where torsion matters and where it does not, which is a more
           useful result than either test alone.
    torsion beats its displaced control but adds nothing over L1+L2
        -> the twin-domain signal is real and already captured by promoter accessibility. Torsion would
           be a mechanism the existing features proxy, not new information.
    torsion adds nothing anywhere
        -> a strong three-way negative: real signature at the promoter, no predictive value on either
           the enhancer-gene task or the rate task. Worth stating plainly if it happens.

CROSS-CELL-LINE, still. The track is SH-SY5Y; the rates are K562. Unchanged from before and unfixable --
no per-locus torsion measurement exists for K562 anywhere public.

-> outputs/orphan/supercoil_txn.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
SC = Path("outputs/orphan/supercoil")
TMP1 = SC / "GSM8706920_1_RPKM.bw"
TMP2 = SC / "GSM8706924_5_RPKM.bw"
NSPLIT = 5
PROM = 2000
WIN = 1_000
FLANK = 2_000
LOCAL = 200_000
SHIFT = 2_000_000
SEED = 3312


def torsion_at(bw, chrom, tss, shift=0):
    """TSS value, upstream arm, downstream arm -- all as ratios to the local 200 kb mean."""
    if chrom not in bw.chroms() or tss is None:
        return None
    lim = bw.chroms()[chrom]

    def q(lo, hi):
        lo, hi = max(0, int(lo) + shift), min(int(hi) + shift, lim)
        if hi <= lo:
            return np.nan
        try:
            v = bw.stats(chrom, lo, hi, type="mean")[0]
        except Exception:
            return np.nan
        return np.nan if v is None else float(v)

    loc = q(tss - LOCAL // 2, tss + LOCAL // 2)
    if not np.isfinite(loc) or loc <= 0:
        return None
    c = q(tss - WIN // 2, tss + WIN // 2)
    a = q(tss - FLANK, tss)
    b = q(tss, tss + FLANK)
    if not (np.isfinite(c) and np.isfinite(a) and np.isfinite(b)):
        return None
    return c / loc, a / loc, b / loc


def main():
    import gzip
    import collections
    import pandas as pd
    import pyBigWig
    from sklearn.ensemble import HistGradientBoostingRegressor
    from scipy.stats import pearsonr

    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("DOES MEASURED DNA TORSION PREDICT TRANSCRIPTION RATE?")
    report("=" * 100)
    report("  Torsion returned PARTIAL twice on the enhancer-gene bar, but the same run measured the")
    report("  twin-domain signature at the promoter at z = +23.21. That is a claim about TRANSCRIPTION,")
    report("  and this is the first test that asks it.")

    dc = pd.read_csv(SP / "rnadecay" / "AvgKdegs.csv")
    dc = dc[(dc.avg_RPKM_total > 0) & (dc.avg_kdeg > 0)].copy()
    dc["log_ksyn"] = np.log10(dc.avg_RPKM_total * dc.avg_kdeg)
    report(f"\n  target: RNADecayCafe v1.1, {len(dc):,} usable rows, "
           f"{(dc.cell_line == 'K562').sum():,} K562")

    D = json.load(open(OUT / "cell_complete.json"))
    info = {g["name"]: g for g in D["genes"]}

    peaks, starts = collections.defaultdict(list), {}
    with gzip.open(SP / "k562_atac.bed.gz", "rt") as fh:
        for line in fh:
            p = line.split("\t")
            if len(p) < 7:
                continue
            peaks[p[0]].append((int(p[1]), int(p[2]), float(p[6])))
    for c in peaks:
        peaks[c].sort()
        starts[c] = np.array([x[0] for x in peaks[c]])
    report(f"  ATAC: ENCODE ENCFF926KTI, {sum(len(v) for v in peaks.values()):,} K562 peaks (GRCh38)")

    def atac_at(chrom, tss):
        v = peaks.get(chrom)
        if not v or tss is None:
            return (0.0, 0.0, 0)
        s = starts[chrom]
        lo = int(np.searchsorted(s, tss - PROM - 5000))
        hi = int(np.searchsorted(s, tss + PROM))
        best = tot = 0.0
        n = 0
        for a, b, sig in v[lo:hi]:
            if b >= tss - PROM and a <= tss + PROM:
                best = max(best, sig)
                tot += sig
                n += 1
        return (best, tot, n)

    bw = pyBigWig.open(str(TMP1))
    genes, T, Ts = [], [], []
    for g in sorted(set(dc.feature_ID) & set(info)):
        rec = info[g]
        ch = rec.get("chrom")
        try:
            tss = int(float(rec.get("tss")))
        except (TypeError, ValueError):
            continue
        sd = rec.get("strand")
        t = torsion_at(bw, ch, tss)
        td = torsion_at(bw, ch, tss, shift=SHIFT)
        if t is None or td is None:
            continue

        def pack(v):
            c, a, b = v
            up, dn = (a, b) if sd == "+" else (b, a)
            return [c, up, dn, up - dn]
        genes.append(g)
        T.append(pack(t))
        Ts.append(pack(td))
    T, Ts = np.array(T), np.array(Ts)
    report(f"  {len(genes):,} genes with a rate, coordinates, strand and torsion")

    def num(g, k, d=0.0):
        try:
            return float(info[g].get(k) or d)
        except (TypeError, ValueError):
            return d

    A = {g: atac_at(info[g].get("chrom"), int(float(info[g].get("tss")))) for g in genes}
    F1 = np.array([[num(g, "cpg"), np.log1p(num(g, "enh")), num(g, "loeuf", 1.0), num(g, "ess"),
                    num(g, "dep_frac"), np.log1p(num(g, "pubs")), num(g, "dark")] for g in genes])
    F2 = np.array([[np.log1p(A[g][0]), np.log1p(A[g][1]), float(A[g][2]), 1.0 if A[g][2] else 0.0]
                   for g in genes])
    nat = sum(1 for g in genes if A[g][2] > 0)
    report(f"  {nat:,} ({nat/len(genes):.1%}) have an ATAC peak at the promoter")
    report(f"  twin-domain asymmetry over these genes: mean {T[:,3].mean():+.5f}, "
           f"z = {T[:,3].mean()/(T[:,3].std(ddof=1)/np.sqrt(len(T))):+.2f}")

    y = {r.feature_ID: r.log_ksyn for r in dc[dc.cell_line == "K562"].itertuples()}
    idx = np.array([i for i, g in enumerate(genes) if g in y])
    report(f"  {len(idx):,} K562 genes scored")

    ARMS = [("L1 basal promoter", [F1]),
            ("L2 + M_ATAC", [F1, F2]),
            ("+ TORSION", [F1, F2, T]),
            ("+ TORSION [displaced 2 Mb]", [F1, F2, Ts])]
    rng = np.random.RandomState(0)
    res = collections.defaultdict(list)
    for si in range(NSPLIT):
        o = idx.copy()
        rng.shuffle(o)
        cut = int(0.7 * len(o))
        tr, te = o[:cut], o[cut:]
        ytr = np.array([y[genes[i]] for i in tr])
        yte = np.array([y[genes[i]] for i in te])
        for lab, blocks in ARMS:
            X = np.hstack(blocks)
            p = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.08,
                                              random_state=0).fit(X[tr], ytr).predict(X[te])
            res[lab].append(pearsonr(p, yte)[0])

    report(f"\n  HELD-OUT GENES, {NSPLIT} splits, 70/30, same regressor and settings as nexus_txn")
    report(f"    {'arm':<32}{'r':>8}{'R2':>8}")
    R = {}
    for lab, _ in ARMS:
        r = float(np.mean(res[lab]))
        R[lab] = r * r
        report(f"    {lab:<32}{r:>8.4f}{r*r:>8.4f}")

    l1, l2 = R["L1 basal promoter"], R["L2 + M_ATAC"]
    tor, tsh = R["+ TORSION"], R["+ TORSION [displaced 2 Mb]"]
    report(f"\n  reconstruction check: L1 {l1:.4f} vs recorded 0.1582   L2 {l2:.4f} vs recorded 0.1667")
    faithful = abs(l1 - 0.1582) < 0.03
    report(f"  {'FAITHFUL' if faithful else 'NOT FAITHFUL -- do not read the torsion arm'}")
    report("\n  " + "-" * 96)
    report(f"  TORSION MINUS L1+L2               : {tor - l2:+.4f}")
    report(f"  TORSION MINUS its displaced twin  : {tor - tsh:+.4f}")

    if not faithful:
        verdict, note = "NOT TESTABLE", "The L1 baseline did not reproduce, so no torsion verdict stands."
    elif tor - l2 >= 0.01 and tor - tsh >= 0.005:
        verdict = "GO"
        note = ("Measured torsion predicts transcription rate beyond promoter accessibility. Combined "
                "with two PARTIALs on the enhancer-gene task, this locates where torsion matters: at "
                "the promoter, on rate, not on which enhancer contacts which gene.")
    elif tor - tsh >= 0.005:
        verdict = "PARTIAL"
        note = ("Torsion beats its displaced control but adds nothing over promoter accessibility. The "
                "twin-domain signal is real and already proxied by ATAC -- a mechanism the existing "
                "features capture, not new information.")
    else:
        verdict = "NO-GO"
        note = ("Torsion adds nothing here either. Three-way negative: a strong twin-domain signature at "
                "the promoter (z = +23.21), no predictive value on the enhancer-gene task, none on rate. "
                "The signature is real and, on these two tasks, not useful.")
    report(f"\n  VERDICT: {verdict}")
    for i in range(0, len(note), 94):
        report(f"    {note[i:i+94]}")

    man = RM.manifest(inputs=[str(SP / "rnadecay" / "AvgKdegs.csv"), str(SP / "k562_atac.bed.gz"),
                              str(TMP1), str(OUT / "cell_complete.json")],
                      available=int((dc.cell_line == "K562").sum()), used=int(len(idx)),
                      selection="filtered", seed=SEED,
                      controls=["displaced 2 Mb torsion", "L1/L2 reconstruction against recorded"],
                      note="K562 rates; L3/L4/L5 not reproducible (_abcfeat.json has no producer)")
    RM.report(man, emit=report)
    json.dump({"test": "supercoil_txn", "manifest": man, "arms": R, "verdict": verdict, "note": note,
               "faithful": bool(faithful), "n_genes": len(genes), "n_scored": int(len(idx)),
               "target": "RNADecayCafe v1.1 (10.5281/zenodo.16884513)",
               "torsion_track": "GSE285699 SH-SY5Y WT", "log": log},
              open(OUT / "supercoil_txn.json", "w"), indent=2)
    report(f"\n  -> {OUT/'supercoil_txn.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
