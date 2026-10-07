"""MEASURED DNA TORSION vs THE 0.663 BAR -- the test the integration assessment said to run first.

THE ARGUMENT FOR RUNNING THIS BEFORE ANY PHYSICS.  The assessment priced two routes to a torsion-aware
cell model. A full supercoiling simulation is 689 core-days per configuration at one sigma and one length,
extrapolated from the repo's own MC timings (0.1244 s/sweep at N=61, 0.7591 s/sweep at N=121, N^2.64 fit).
Adding a DIRECTLY MEASURED torsion track as one column to a harness that already runs costs minutes. And
the cheap test dominates the expensive one logically: if measured torsion cannot beat the bar, a simulated
multiplier derived from torsion cannot either, because the simulation's whole output is a function of the
same quantity. A negative here retires the expensive route without spending it.

THE PROTOCOL IS COPIED, NOT REINVENTED.  Same compendium, same 8 epigenetic columns plus 311 TF identity
one-hots, same _cv_auprc over _seeded_folds(chromosome, seed) averaged across seeds 0, 1, 2, same arms and
the same GO/PARTIAL/NO-GO thresholds as rouse_gate. If the numbers are to be compared with 0.6076, 0.6632
and 0.5976, they have to be produced the same way.

THE FEATURES.  Four, from GSE277502 bTMP-seq (hTERT-RPE1, hg38, 20 kb bins):
    torsion at the ENHANCER
    torsion at the TSS
    their DIFFERENCE          -- a torsional step between the two elements
    the MEAN ACROSS THE SPAN  -- the torsional environment the pair sits in
The difference is the one that could carry something a per-gene value cannot: it is a property of the
PAIR, which is what the task actually asks about.

THE CONTROL.  The same shape as the gate's shuffled-CTCF twin: query the identical track at coordinates
displaced 2 Mb along the same chromosome. Distribution, chromosome and bin structure are preserved; only
the correspondence to the actual locus is destroyed. A torsion arm that cannot beat that is reading
genomic density, not torsion.

THE CAVEAT THAT CANNOT BE ENGINEERED AWAY.  The track is RPE1; the CRISPR assay is K562. This is a
cross-cell-line transfer and supercoiling is transcription-dependent, so the tissue mismatch attacks the
signal directly. A null result therefore has two possible causes -- no effect, or the wrong cell line --
and this experiment cannot separate them. Said here, before the number.

PREDECLARED, before any number, thresholds identical to rouse_gate:
    torsion+CTCF beats the bar by >= 0.01 AND torsion-only beats its shuffled twin by >= 0.005
        -> GO. Measured torsion carries pair-level signal the epigenetic and boundary features miss, and
           the physics route is worth pricing properly.
    torsion-only beats its shuffled twin but adds nothing over the bar
        -> PARTIAL. The track is real but redundant with what CTCF counting already provides.
    neither
        -> NO-GO. Retires the 689 core-day route on a measurement, not an opinion. Given a 20 kb bin and
           a cross-cell-line transfer this is the expected outcome, and expecting it is not a reason to
           skip the test -- it is the reason the test is cheap enough to be worth running.

-> outputs/orphan/supercoil_gate.json
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
REP1 = SC / "GSM8523629_supercoil_rep1_hg38.bw"
REP2 = SC / "GSM8523630_supercoil_rep2_hg38.bw"
WIN = 20_000
SHIFT = 2_000_000
SEED = 6021


def track_features(rows, bw, shift=0):
    """torsion at enhancer, at TSS, their difference, and the mean across the span."""
    F = np.zeros((len(rows), 4))
    chroms = bw.chroms()
    for k, (c, e, t) in enumerate(rows):
        if c not in chroms:
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

        ve = q(e - WIN // 2, e + WIN // 2)
        vt = q(t - WIN // 2, t + WIN // 2)
        lo, hi = min(e, t), max(e, t)
        vs = q(lo, hi) if hi - lo > 1000 else vt
        F[k] = (ve if np.isfinite(ve) else 1.0,
                vt if np.isfinite(vt) else 1.0,
                (ve - vt) if (np.isfinite(ve) and np.isfinite(vt)) else 0.0,
                vs if np.isfinite(vs) else 1.0)
    return F


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
    report("  GSE277502 bTMP-seq, hTERT-RPE1, hg38, 20 kb bins. The CRISPR assay is K562, so this is a")
    report("  CROSS-CELL-LINE transfer and a null has two possible causes this test cannot separate.")
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
            ("+ TORSION only", np.hstack([Xa, Tr])),
            ("+ TORSION + CTCF-count", np.hstack([Xc, Tr])),
            ("+ TORSION [displaced 2 Mb]", np.hstack([Xa, Ts]))]
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
    both = res["+ TORSION + CTCF-count"]
    only = res["+ TORSION only"]
    sh = res["+ TORSION [displaced 2 Mb]"]
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
    json.dump({"test": "supercoil_gate", "manifest": man, "arms": res, "verdict": verdict,
               "note": note, "replicate_r": float(rho), "accession": "GSE277502",
               "cell_line_track": "hTERT-RPE1", "cell_line_assay": "K562", "bin_bp": WIN,
               "log": log}, open(OUT / "supercoil_gate.json", "w"), indent=2)
    report(f"\n  -> {OUT/'supercoil_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
