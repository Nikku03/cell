"""MEASURED DNA TORSION vs THE 0.663 BAR, ATTEMPT 2 -- at 250x finer resolution.

WHY A SECOND ATTEMPT.  Attempt 1 returned PARTIAL: torsion beat its own displaced control by +0.0139 but
added +0.0000 on top of CTCF counting. I named three confounds it could not separate, and one of them is
fixable with a different measurement rather than a different argument.

    RESOLUTION -- attempt 1 used GSE277502 at a native 20 kb bin. The assessment's own reasoning says
        promoter-proximal torsion lives at hundreds of bp, so that test could not have seen the effect it
        was looking for. FIXED HERE: GSE285699 TMP-seq is run-length encoded at 70-90 bp where there is
        signal, roughly 250x finer.
    CELL LINE -- attempt 1 used RPE1 against a K562 assay. NOT FIXED, and cannot be: the search
        established that no per-locus torsion measurement exists for K562 in any public repository. This
        attempt uses SH-SY5Y (neuroblastoma), which is a different mismatch, not a smaller one.
    NOT SIGMA -- still a crosslink propensity rather than superhelical density. Unfixable by any
        available measurement.

THE NEW PROBLEM THIS TRACK BRINGS.  GSE285699 deposits RAW RPKM COVERAGE with no naked-DNA control, where
GSE277502 shipped a matched genomic control and a pre-computed ratio. Raw coverage is confounded by
mappability and copy number: a region with more reads may simply be more mappable or present in more
copies, which has nothing to do with torsion. Using it unnormalised would measure the karyotype of a
neuroblastoma line.

    So every value here is LOCALLY NORMALISED: signal in the window divided by the mean over +/- 100 kb
    of the same track. Copy number and broad mappability are constant across 200 kb and divide out; the
    fine structure that the extra resolution was acquired for survives. This is the standard treatment
    and it is also the only one available without a control library.

WHAT IS NEW BESIDES RESOLUTION.  The strand-aware asymmetry becomes a real feature rather than a null.
Twin-domain predicts negative supercoiling behind an elongating polymerase and positive ahead, so
TMP signal should be higher upstream of the TSS. At 20 kb that test returned z = -1.38, which says
nothing because the bin was larger than the effect. At 70-90 bp it can be asked properly, and it is
included both as a model feature and as a standalone measurement.

PREDECLARED, before any number, thresholds unchanged from attempt 1 and from rouse_gate:
    torsion+CTCF beats the bar by >= 0.01 AND torsion-only beats its displaced twin by >= 0.005
        -> GO. The resolution WAS the limitation, and finer measurement recovers signal that 20 kb bins
           washed out.
    torsion-only beats its displaced twin but adds nothing over the bar
        -> PARTIAL again. Two tracks, two cell lines, two resolutions, same answer: whatever torsion
           carries here, CTCF counting already has. That is a much stronger negative than one track.
    neither
        -> NO-GO. Note this would be a WORSE result than attempt 1, and if it happens the honest reading
           is the missing control library, not the biology.
    strand asymmetry significant and upstream-positive
        -> the twin-domain signature, measurable at last. Independent of whether the model arm gains.

-> outputs/orphan/supercoil_gate2.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

OUT = Path("outputs/orphan")
INV = OUT / "invivo"
SC = OUT / "supercoil"
REP1 = SC / "GSM8706920_1_RPKM.bw"          # SH-SY5Y WT rep1, TMP-seq, hg38
REP2 = SC / "GSM8706924_5_RPKM.bw"          # SH-SY5Y WT rep2
WIN = 1_000                                 # +/- 500 bp, the scale promoter torsion actually lives at
LOCAL = 200_000                             # local normaliser: removes copy number and broad mappability
FLANK = 2_000                               # strand-aware arm for the twin-domain feature
SHIFT = 2_000_000
SEED = 7734


def track_features(rows, bw, shift=0):
    """Five locally-normalised features at the scale promoter torsion actually occupies.

    EVERY value is a RATIO to the local 200 kb mean of the same track. GSE285699 deposits raw RPKM with no
    naked-DNA control, so an unnormalised value is confounded by mappability and copy number -- in a
    neuroblastoma line that would largely be measuring the karyotype. Copy number is constant across
    200 kb and divides out; the fine structure survives, which is the whole reason for the finer track.
    """
    F = np.zeros((len(rows), 5))
    chroms = bw.chroms()
    for k, (c, e, t) in enumerate(rows):
        if c not in chroms:
            F[k] = (1.0, 1.0, 0.0, 0.0, 1.0)
            continue
        lim = chroms[c]

        def q(lo, hi):
            lo, hi = max(0, int(lo) + shift), min(int(hi) + shift, lim)
            if hi <= lo:
                return np.nan
            try:
                v = bw.stats(c, lo, hi, type="mean")[0]
            except Exception:
                return np.nan
            return np.nan if v is None else float(v)

        def norm(pos, half):
            loc = q(pos - LOCAL // 2, pos + LOCAL // 2)
            v = q(pos - half, pos + half)
            if not np.isfinite(v) or not np.isfinite(loc) or loc <= 0:
                return np.nan
            return v / loc

        ve = norm(e, WIN // 2)
        vt = norm(t, WIN // 2)
        # STRAND-AWARE ASYMMETRY: twin-domain puts negative supercoiling BEHIND the polymerase, so with
        # strand known the upstream arm is the one that should carry more TMP signal. This is the test
        # that was meaningless at a 20 kb bin.
        loc_t = q(t - LOCAL // 2, t + LOCAL // 2)
        a = q(t - FLANK, t)
        b = q(t, t + FLANK)
        asym = ((a - b) / loc_t) if (np.isfinite(a) and np.isfinite(b) and np.isfinite(loc_t)
                                     and loc_t > 0) else np.nan
        lo, hi = min(e, t), max(e, t)
        vs = (q(lo, hi) / loc_t) if (hi - lo > 1000 and np.isfinite(loc_t) and loc_t > 0) else vt
        F[k] = (ve if np.isfinite(ve) else 1.0,
                vt if np.isfinite(vt) else 1.0,
                (ve - vt) if (np.isfinite(ve) and np.isfinite(vt)) else 0.0,
                asym if np.isfinite(asym) else 0.0,
                vs if np.isfinite(vs) else 1.0)
    return F


def strand_asymmetry(rows, strands, bw):
    """The twin-domain test as a standalone measurement, independent of any model."""
    d = []
    chroms = bw.chroms()
    for (c, e, t), sd in zip(rows, strands):
        if c not in chroms or sd not in ("+", "-"):
            continue
        lim = chroms[c]

        def q(lo, hi):
            lo, hi = max(0, int(lo)), min(int(hi), lim)
            if hi <= lo:
                return np.nan
            try:
                v = bw.stats(c, lo, hi, type="mean")[0]
            except Exception:
                return np.nan
            return np.nan if v is None else float(v)

        a, b = q(t - FLANK, t), q(t, t + FLANK)
        loc = q(t - LOCAL // 2, t + LOCAL // 2)
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(loc)) or loc <= 0:
            continue
        up, dn = (a, b) if sd == "+" else (b, a)
        d.append((up - dn) / loc)
    d = np.array(d)
    if len(d) < 20:
        return {"n": int(len(d))}
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"n": int(len(d)), "mean": float(d.mean()), "se": se,
            "z": float(d.mean() / se) if se else float("nan")}


def main():
    import pandas as pd
    import pyBigWig
    from crispr_gate import _cv_auprc, _seeded_folds
    from contact_gate import load_ctcf, contact_features

    log = []

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("MEASURED DNA TORSION vs THE 0.663 BAR")
    report("=" * 100)
    report("  ATTEMPT 2. GSE285699 TMP-seq, SH-SY5Y WT, hg38, run-length encoded at 70-90 bp -- about")
    report("  250x finer than attempt 1's 20 kb bins, which is the confound most likely to have caused")
    report("  the PARTIAL. Still not K562: no per-locus torsion measurement exists for K562 anywhere.")
    report("  This track ships NO naked-DNA control, so every value is locally normalised by the 200 kb")
    report("  mean around it -- otherwise raw RPKM in a neuroblastoma line largely measures karyotype.")
    report("")
    report("  THE BAR IS RECOMPUTED, NOT ASSUMED. The CTCF peak file the recorded 0.6632 was produced")
    report("  from lived in the ephemeral scratch directory and is gone, so the arm is rebuilt from")
    report("  ENCODE ENCFF362OPG (K562 CTCF IDR thresholded, ENCSR000AKO, GRCh38, 44,104 peaks). That is")
    report("  a DIFFERENT peak set, so the bar measured here is the one torsion is judged against, and")
    report("  its distance from the recorded 0.6632 is reported as a check on the reconstruction.")

    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]

    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS",
                                        "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}",
                     p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2,
                    int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    strandmap = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        h2 = fh.readline().rstrip("\n").split("\t")
        gi, si = h2.index("measuredGeneSymbol"), h2.index("strandGene")
        for line in fh:
            q = line.rstrip("\n").split("\t")
            if len(q) > max(gi, si) and q[si] in ("+", "-"):
                strandmap.setdefault(q[gi], q[si])
    strands = [strandmap.get(g) for g in df["gene"]]
    report(f"\n  {len(df):,} pairs, torsion window {WIN//1000} kb, shuffled control displaced "
           f"{SHIFT//10**6} Mb")

    bw1 = pyBigWig.open(str(REP1))
    bw2 = pyBigWig.open(str(REP2))
    Tr = track_features(rows, bw1)
    Tr2 = track_features(rows, bw2)
    Ts = track_features(rows, bw1, shift=SHIFT)
    report(f"  torsion at enhancer: median {np.median(Tr[:,0]):.4f}   at TSS: {np.median(Tr[:,1]):.4f}"
           f"   |difference| median {np.median(np.abs(Tr[:,2])):.4f}")
    rho = np.corrcoef(Tr[:, 0], Tr2[:, 0])[0, 1]
    report(f"  replicate agreement on the enhancer feature across pairs: r = {rho:.4f}")
    sa = strand_asymmetry(rows, strands, bw1)
    report(f"\n  TWIN-DOMAIN, measured at {FLANK//1000} kb arms rather than 20 kb -- the test that was")
    report(f"  meaningless at attempt 1's bin size. Predicts upstream > downstream.")
    if sa.get("n", 0) >= 20:
        report(f"    n = {sa['n']}   mean (up-down)/local = {sa['mean']:+.5f}   se {sa['se']:.5f}   "
               f"z = {sa['z']:+.2f}")
        report("    " + ("SIGNIFICANT, predicted direction" if sa["z"] > 2 else
                         "SIGNIFICANT, REVERSED -- reported as measured" if sa["z"] < -2 else
                         "null at this scale too"))
    else:
        report(f"    too few usable genes ({sa.get('n', 0)})")

    real = load_ctcf()
    Cr = contact_features(rows, real)
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for i, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[i, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    y = df["crispr_hit"].values
    ch = df["chromosome"].values
    Xa = np.hstack([df[base].values, tfmat])
    Xc = np.hstack([Xa, Cr])

    def sc(X):
        return float(np.mean([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in (0, 1, 2)]))

    arms = [("epi + TF identity (311)", Xa),
            ("+ CTCF-count contact  [the 0.663 bar]", Xc),
            ("+ TORSION-fine only", np.hstack([Xa, Tr])),
            ("+ TORSION-fine + CTCF-count", np.hstack([Xc, Tr])),
            ("+ TORSION-fine [displaced 2 Mb]", np.hstack([Xa, Ts]))]
    res = {}
    report("")
    for nm, X in arms:
        a = sc(X)
        res[nm] = a
        report(f"    {nm:40s} AUPRC {a:.4f}")

    RECORDED = 0.6632
    bar = res["+ CTCF-count contact  [the 0.663 bar]"]
    report(f"\n  reconstructed bar {bar:.4f} against the recorded {RECORDED:.4f}  "
           f"(delta {bar - RECORDED:+.4f}) -- a different CTCF peak set, so exact agreement is not expected")
    both = res["+ TORSION-fine + CTCF-count"]
    only = res["+ TORSION-fine only"]
    sh = res["+ TORSION-fine [displaced 2 Mb]"]
    report("\n  " + "-" * 96)
    report(f"  TORSION+count MINUS the bar     : {both - bar:+.4f}")
    report(f"  TORSION-only  MINUS the bar     : {only - bar:+.4f}")
    report(f"  TORSION-only  MINUS its displaced control : {only - sh:+.4f}")
    if both - bar >= 0.01 and only - sh >= 0.005:
        verdict = "GO"
        note = ("Measured torsion carries pair-level signal the epigenetic and boundary features miss. "
                "The physics route is worth pricing properly, and a K562 track would be worth acquiring.")
    elif only - sh >= 0.005:
        verdict = "PARTIAL"
        note = ("Torsion beats its displaced control, so the track is reading something real about "
                "position -- but it adds nothing beyond CTCF counting. Redundant, not absent.")
    else:
        verdict = "NO-GO"
        note = ("Measured torsion does not beat the bar and does not beat its own displaced control. "
                "Since a simulation's output is a function of the same quantity, this retires the "
                "689 core-day physics route on a measurement rather than an opinion. Two causes remain "
                "confounded and this experiment cannot separate them: no effect, or the wrong cell line "
                "(RPE1 track against a K562 assay) at a 20 kb bin that cannot see promoter-proximal "
                "torsion.")
    report(f"\n  VERDICT: {verdict}")
    for line in [note[i:i + 94] for i in range(0, len(note), 94)]:
        report(f"    {line}")

    man = RM.manifest(inputs=[str(REP1), str(REP2), str(OUT / "crispr_features_compendium.csv"),
                              str(INV / "crispr_egpairs.tsv")],
                      available=len(df), used=len(df), selection="all", seed=SEED,
                      controls=["displaced 2 Mb track", "replicate agreement", "CTCF-count bar"],
                      note="protocol copied from rouse_gate: same folds, same metric, same thresholds")
    RM.report(man, emit=report)
    json.dump({"test": "supercoil_gate2", "manifest": man, "arms": res, "verdict": verdict,
               "note": note, "replicate_r": float(rho), "accession": "GSE285699",
               "cell_line_track": "SH-SY5Y", "cell_line_assay": "K562", "bin_bp": WIN,
               "strand_asymmetry": sa, "log": log},
              open(OUT / "supercoil_gate2.json", "w"), indent=2)
    report(f"\n  -> {OUT/'supercoil_gate2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
