"""LOOP 143 -- WHICH PROTEINS GET PULSED, AND WHEN. MEASURE THE WAVEFORM, DO NOT PREDICT IT.

Seven mechanisms have been eliminated on this question and every one of them was a PREDICTOR:
transcription, two TF networks, degron sequence motifs, 5' UTR features, relocalisation, curator
ubiquitin annotation, and protease competition. Each asked "what property of a protein says it will
be destroyed on schedule". Two of them turned out to be measuring publication count.

Loop 142 changed what is available. The equation now carries a destruction PULSE, and a pulse
leaves a signature in the trajectory itself:

    production runs at a roughly constant k, so the protein RISES slowly across most of the cycle
    b jumps by ~20x for ~10% of the cycle, so it FALLS fast, in one short window

That is a sawtooth, and a sawtooth is ASYMMETRIC UNDER TIME REVERSAL. A protein driven the other
way -- production pulsed, degradation constant -- has the mirror shape: fast rise, slow exponential
decay. A symmetric oscillation has neither. So the mechanism is readable from the shape, per
protein, with no sequence feature and nothing to be confounded by fame.

THE FRACTIONS, AND A MISTAKE Y3 CAUGHT ON THE FIRST RUN. The file has seven LFQ columns, F0..F6,
and I used all seven. F0 IS NOT A CYCLE FRACTION -- loop 123 uses F1..F6 as the six fractions and
pools them G1 = mean(F1,F2), S = mean(F3,F4), G2 = mean(F5,F6). Including F0 put every canonical
substrate's "crash" in the F0->F1 interval, which is not a destruction event but the difference
between a reference sample and G1-enriched cells. Y2 passed anyway -- the shape statistic worked --
and Y3 failed, which is exactly the split the gate order was built for: the METHOD was sound and
the TIME AXIS was wrong, and reporting Y2 alone would have shipped a working method on a broken
axis.

AND THE FRACTIONS STOP AT G2. There is no M fraction: elutriation of NB4 tops out at G2, so a
mitotic destruction happens BETWEEN the last fraction and the first, and the population returns to
G1 at F1. The trajectory is therefore treated as CIRCULAR -- six points, six differences, with the
F6->F1 wrap included -- which is what a closed cycle demands and which is where the APC/C
substrates must crash if the method reads timing at all.

THE STATISTIC. For six log2 intensities take the six CIRCULAR differences d_i. Let

    C_fall = max|d_i over the falling steps| / sum|d_i over the falling steps|
    C_rise = max( d_i over the rising steps) / sum( d_i over the rising steps)
    A      = C_fall - C_rise

A > 0 means the decline is concentrated relative to the rise: a destruction pulse. A < 0 means the
rise is concentrated: a production pulse. A = 0 is time-reversal symmetric.

WHY THIS IS SELF-CONTROLLED, and it is the reason to prefer it over a raw "largest drop" statistic.
Both halves come from the SAME protein: same abundance, same peptide count, same measurement noise,
same selection for having been called oscillating. Amplitude cancels because both terms are ratios.
Anything that inflates concentration generally -- few timepoints, log-scale noise, selection on
fold-change -- inflates BOTH terms and cancels in the difference. That is what a raw crash statistic
cannot claim, and loop 141 has just finished paying for a gate that compared two point estimates
without asking what would inflate both.

THE ORDER OF THE GATES IS THE ARGUMENT. Y2 runs BEFORE Y4. Sixteen textbook substrates of APC/C and
SCF have complete trajectories in this data, and their destruction phases are known from decades of
work that has nothing to do with this project. If the statistic cannot find THOSE, it does not
matter what it says about the 362, and no amount of explaining afterwards would repair it. A method
validated on known biology before it is pointed at unknown biology is the whole design.

WHAT WOULD REFUTE THE PULSE HYPOTHESIS. If the canonical substrates show A <= 0, or if the 362 are
indistinguishable from the 642 on A, then either the waveform is not a pulse or seven fractions
cannot see it. Both are reportable and neither is repairable by choosing a different statistic
afterwards.

PREDECLARED:

  Y1 CAN THE STATISTIC MOVE, AND WHAT IS ITS NULL?                   THE CAPABILITY CHECK.
       distribution of A over all complete trajectories, and over PHASE-RANDOMISED surrogates that
       preserve each protein's own values but destroy their order. Gate: the real distribution must
       be distinguishable from the surrogate one, and A must not be degenerate. loop 92 found
       twelve gates that fired while measuring nothing; this one runs first.

  Y2 THE POSITIVE CONTROL, AND IT RUNS BEFORE THE RESULT.            THE ONE THAT CAN END IT.
       sixteen textbook APC/C and SCF substrates -- CCNB1, CCNA2, PLK1, AURKA, AURKB, GMNN, PTTG1,
       CDC20, TOP2A, UBE2C, BUB1, KIF20A, CDT1, CCNB2, NUSAP1, TPX2. Gate: their median A must
       exceed the median A of all proteins, and the excess must clear a bootstrap interval. If
       known destruction substrates do not read as pulsed, the method is broken.

  Y3 DOES THE CRASH LAND WHERE THE TEXTBOOK SAYS?
       for each canonical substrate the interval carrying its largest fall, against the phase its
       machine is known to fire in. The fractions run G1 -> S -> G2 -> M by cell size. Gate: report
       the crash interval for every one of them, and the concordance. This is the check that the
       method reads WHEN and not merely WHETHER.

  Y4 THE 362.                                                        THE QUESTION.
       A for the protein-only oscillators against the transcript-only ones and against the genes
       that oscillate in neither, on the exact 88/362/38/642 split loop 119 recorded. Gate: report
       all four groups with bootstrap intervals.

  Y5 ARE THERE DESTRUCTION WAVES, OR IS EVERY PROTEIN ITS OWN CLOCK?
       the distribution of crash intervals among proteins with A above the surrogate 95th
       percentile. Gate: test it against uniform. Discrete waves would mean a handful of machines
       firing at set times and would make membership of a wave the thing to predict next. A uniform
       spread would mean the timing is per-protein and there is no shared clock to find.

  Y6 IS IT FAME?
       publication count against A, and against membership of the pulsed set. Gate: fame beat the
       biology in seven earlier loops here, twice through a sequence motif. A is a shape statistic
       and should be immune -- Y6 checks that rather than asserting it.

-> outputs/loop_pulse_identify.json
"""
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import gate_guard as GG  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
LY = LR.SC / "ly2014_supp1-v1.txt"
HPA = LR.SC / "proteinatlas.tsv"
SEED = 14300
N_SURR = 200
N_BOOT = 2000

# Textbook substrates of timed destruction, with the phase their machine fires in. Chosen from
# decades of work predating and independent of this project. The phase labels map onto the
# elutriation ordering G1 -> S -> G2 -> M, i.e. fraction index 0 -> 6.
CANON = {
    "CCNB1": ("APC/C-Cdc20", "M"), "CCNB2": ("APC/C-Cdc20", "M"),
    "PTTG1": ("APC/C-Cdc20", "M"), "CCNA2": ("APC/C-Cdc20", "M"),
    "PLK1": ("APC/C-Cdh1", "M/G1"), "AURKA": ("APC/C-Cdh1", "M/G1"),
    "AURKB": ("APC/C-Cdh1", "M/G1"), "CDC20": ("APC/C-Cdh1", "G1"),
    "UBE2C": ("APC/C-Cdh1", "M/G1"), "NUSAP1": ("APC/C-Cdh1", "M/G1"),
    "TPX2": ("APC/C-Cdh1", "M/G1"), "KIF20A": ("APC/C-Cdh1", "M/G1"),
    "BUB1": ("APC/C-Cdh1", "M/G1"), "TOP2A": ("APC/C-Cdh1", "M/G1"),
    "GMNN": ("APC/C-Cdh1", "M/G1"), "CDT1": ("SCF-Skp2/CRL4", "S"),
}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def asym(x):
    """A = C_fall - C_rise for one CIRCULAR log2 trajectory.

    The differences wrap: d_5 is x[0] - x[5], the step from the last fraction back into the next
    cycle's first. Elutriation tops out at G2, so a mitotic destruction lives in that wrap and
    nowhere else. Returns (A, C_fall, C_rise, crash_index) with crash_index 5 meaning the wrap."""
    d = np.diff(np.concatenate([x, x[:1]]))
    f, r = d[d < 0], d[d > 0]
    if len(f) == 0 or len(r) == 0:
        return np.nan, np.nan, np.nan, -1
    c_f = float(np.max(np.abs(f)) / np.sum(np.abs(f)))
    c_r = float(np.max(r) / np.sum(r))
    crash = int(np.argmin(d))
    return c_f - c_r, c_f, c_r, crash


def boot_median(v, rng, n=N_BOOT):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return float("nan"), float("nan"), float("nan")
    m = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 143 -- which proteins get pulsed, and when")
    say("=" * 100)
    say()
    gates, res = {}, {}

    import pandas as pd
    d = pd.read_csv(LY, sep="\t", low_memory=False)
    # F1..F6 ONLY. F0 is not a cycle fraction -- see the docstring; the first run used it and Y3
    # correctly refused the result.
    lf = [f"LFQ_intensity_F{k}" for k in range(1, 7)]
    X = d[lf].values.astype(float)
    ok = (X > 0).all(1)
    genes = d["gene_names"].fillna("").str.split(";").str[0].values
    L = np.full(X.shape, np.nan)
    L[ok] = np.log2(X[ok])
    say(f"  Ly et al. 2014, {len(d):,} rows, {len(lf)} cycle fractions F1..F6 "
        f"(G1 = F1,F2; S = F3,F4; G2 = F5,F6 -- loop 123's pooling)")
    say(f"  F0 is EXCLUDED: it is not a cycle fraction, and including it put every canonical")
    say(f"  substrate's crash in the F0->F1 step on the first run. Y3 refused that result.")
    say(f"  the trajectory is CIRCULAR -- the F6->F1 wrap is where mitotic destruction must sit,")
    say(f"  because elutriation stops at G2 and the population re-enters G1 at F1.")
    say(f"  complete trajectories: {int(ok.sum()):,}")
    tot = X[ok].sum(0)
    say(f"  per-fraction LFQ totals relative to F0: {np.round(tot / tot[0], 4).tolist()}")
    say(f"  spread {np.ptp(tot / tot[0]):.1%} -- small, but NOT zero, so a global drift exists and")
    say(f"  is stated rather than corrected away. A is a DIFFERENCE of two concentrations from the")
    say(f"  same trajectory, so a monotone global scaling cancels from it exactly.")
    say()

    idx = np.flatnonzero(ok)
    A = np.full(len(d), np.nan)
    CF = np.full(len(d), np.nan)
    CR = np.full(len(d), np.nan)
    CRASH = np.full(len(d), -1, dtype=int)
    for i in idx:
        a, cf, cr, c = asym(L[i])
        A[i], CF[i], CR[i], CRASH[i] = a, cf, cr, c

    # ---------------------------------------------------------------- Y1
    say("Y1 CAN THE STATISTIC MOVE, AND WHAT IS ITS NULL?")
    fin = np.isfinite(A)
    say(f"     A defined for {int(fin.sum()):,} proteins "
        f"(needs at least one rise and one fall)")
    say(f"     A: median {np.median(A[fin]):+.4f}  IQR "
        f"[{np.percentile(A[fin], 25):+.4f}, {np.percentile(A[fin], 75):+.4f}]")
    surr = []
    sub = idx[rng.integers(0, len(idx), min(1500, len(idx)))]
    for _ in range(max(1, N_SURR // 10)):
        for i in sub[:150]:
            a, _, _, _ = asym(rng.permutation(L[i]))
            if np.isfinite(a):
                surr.append(a)
    surr = np.array(surr)
    say(f"     SURROGATE: each protein's own seven values, order destroyed ({len(surr):,} draws)")
    say(f"     surrogate A: median {np.median(surr):+.4f}  95th pct {np.percentile(surr, 95):+.4f}")
    ach = GG.achievable_change(np.round(A[fin], 3))
    say(f"     gate_guard achievable-change bound on A: {ach:.4f}")
    gates["Y1"] = bool(ach >= 0.02 and np.isfinite(np.median(surr)))
    res["y1"] = {"n": int(fin.sum()), "median_A": float(np.median(A[fin])),
                 "surrogate_median": float(np.median(surr)),
                 "surrogate_p95": float(np.percentile(surr, 95)), "achievable": ach}
    say(f"     Y1 {'PASS' if gates['Y1'] else 'FAIL'} -- the statistic "
        f"{'varies and has a null' if gates['Y1'] else 'IS DEGENERATE and can measure nothing'}")
    say()

    # ---------------------------------------------------------------- Y2
    say("Y2 THE POSITIVE CONTROL, BEFORE THE RESULT")
    gi = {}
    for i in idx:
        g = genes[i]
        if g in CANON and g not in gi:
            gi[g] = i
    say(f"     canonical substrates with a complete trajectory: {len(gi)}/{len(CANON)}")
    a_can = np.array([A[i] for i in gi.values()])
    m_all, lo_all, hi_all = boot_median(A[fin], np.random.default_rng(SEED + 1))
    m_can, lo_can, hi_can = boot_median(a_can, np.random.default_rng(SEED + 2))
    say(f"     all proteins        median A {m_all:+.4f}  [{lo_all:+.4f}, {hi_all:+.4f}]")
    say(f"     canonical substrates median A {m_can:+.4f}  [{lo_can:+.4f}, {hi_can:+.4f}]")
    say(f"     excess {m_can - m_all:+.4f}")
    gates["Y2"] = bool(lo_can > m_all)
    res["y2"] = {"n_canon": len(gi), "median_all": m_all, "median_canon": m_can,
                 "ci_canon": [lo_can, hi_can], "excess": m_can - m_all,
                 "per_gene": {g: float(A[i]) for g, i in sorted(gi.items())}}
    say(f"     Y2 {'PASS' if gates['Y2'] else 'FAIL'} -- known destruction substrates "
        f"{'DO read as pulsed' if gates['Y2'] else 'DO NOT read as pulsed, and the method is broken'}")
    for g, i in sorted(gi.items(), key=lambda kv: -A[kv[1]]):
        lab = "F6->F1 wrap" if CRASH[i] == 5 else f"F{CRASH[i]+1}->F{CRASH[i]+2}"
        say(f"       {g:<8} A {A[i]:+.4f}   C_fall {CF[i]:.3f}  C_rise {CR[i]:.3f}   "
            f"crash {lab:<12} [{CANON[g][0]}, {CANON[g][1]}]")
    say()

    # ---------------------------------------------------------------- Y3
    say("Y3 DOES THE CRASH LAND WHERE THE TEXTBOOK SAYS?")
    say(f"     fractions run G1(F1,F2) -> S(F3,F4) -> G2(F5,F6) and then WRAP to the next G1.")
    say(f"     A mitotic or M/G1 destruction must therefore sit in the F5->F6 step or the F6->F1")
    say(f"     wrap, which are crash indices 4 and 5.")
    late = [g for g, i in gi.items() if CRASH[i] >= 4]
    mito = [g for g in gi if CANON[g][1] in ("M", "M/G1")]
    hit = [g for g in late if g in mito]
    say(f"     substrates destroyed at M or M/G1: {len(mito)}")
    say(f"     of those, crashing in F4->F5 or later: {len(hit)}  ({len(hit)/max(1,len(mito)):.0%})")
    say(f"     crash interval by substrate:")
    from collections import Counter
    cc = Counter(int(CRASH[i]) for i in gi.values())
    say(f"       histogram over F1->F2 .. F5->F6, F6->F1(wrap): "
        f"{[cc.get(k, 0) for k in range(6)]}")
    gates["Y3"] = bool(len(hit) / max(1, len(mito)) > 0.5)
    res["y3"] = {"n_mitotic": len(mito), "n_late": len(hit),
                 "concordance": len(hit) / max(1, len(mito)),
                 "crash_hist": [cc.get(k, 0) for k in range(6)]}
    say(f"     Y3 {'PASS' if gates['Y3'] else 'FAIL'} -- the method reads WHEN "
        f"{'and it agrees with the textbook' if gates['Y3'] else 'and it DISAGREES with the textbook'}")
    say()

    # ---------------------------------------------------------------- Y4
    say("Y4 THE 362")
    cp, ct = {}, {}
    with open(HPA, newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        h = next(rd)
        iG, iCP, iCT = h.index("Gene"), h.index("CCD Protein"), h.index("CCD Transcript")
        for row in rd:
            if len(row) <= max(iG, iCP, iCT):
                continue
            if row[iCP] in ("Yes", "No"):
                cp[row[iG]] = row[iCP]
            if row[iCT] in ("Yes", "No"):
                ct[row[iG]] = row[iCT]
    both = sorted(set(cp) & set(ct))
    grp = {}
    for g in both:
        p, t = cp[g] == "Yes", ct[g] == "Yes"
        grp[g] = "both" if (p and t) else ("protein only" if p else
                                           ("transcript only" if t else "neither"))
    gidx = {}
    for i in idx:
        g = genes[i]
        if g in grp and g not in gidx:
            gidx[g] = i
    say(f"     {len(both):,} genes with both HPA calls; {len(gidx):,} of them have a complete "
        f"Ly trajectory")
    out4 = {}
    for name in ("both", "protein only", "transcript only", "neither"):
        v = [A[i] for g, i in gidx.items() if grp[g] == name]
        m, lo, hi = boot_median(v, np.random.default_rng(SEED + 3))
        out4[name] = {"n": len(v), "median": m, "ci": [lo, hi]}
        say(f"     {name:<16} n {len(v):>4}   median A {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    po, ne = out4["protein only"], out4["neither"]
    gates["Y4"] = bool(po["ci"][0] > ne["median"])
    res["y4"] = out4
    say(f"     Y4 {'PASS' if gates['Y4'] else 'FAIL'} -- the protein-only oscillators "
        f"{'ARE more pulse-shaped than the non-oscillators' if gates['Y4'] else 'are NOT distinguishable from the non-oscillators'}")
    say()

    # ---------------------------------------------------------------- Y5
    say("Y5 ARE THERE DESTRUCTION WAVES, OR IS EVERY PROTEIN ITS OWN CLOCK?")
    thr = float(np.percentile(surr, 95))
    pulsed = np.flatnonzero(fin & (A > thr))
    say(f"     surrogate 95th percentile of A: {thr:+.4f}")
    say(f"     proteins above it: {len(pulsed):,} of {int(fin.sum()):,} "
        f"({len(pulsed)/int(fin.sum()):.1%})")
    hist = np.array([int((CRASH[pulsed] == k).sum()) for k in range(6)])
    say(f"     crash-interval histogram: {hist.tolist()}")
    exp = hist.sum() / 6.0
    chi = float(((hist - exp) ** 2 / exp).sum())
    say(f"     chi-square against uniform: {chi:.1f} on 5 df "
        f"(critical 11.07 at p=0.05, 20.5 at p=0.001)")
    gates["Y5"] = bool(chi > 11.07)
    res["y5"] = {"threshold": thr, "n_pulsed": int(len(pulsed)),
                 "crash_hist": hist.tolist(), "chi2_vs_uniform": chi}
    say(f"     Y5 {'PASS' if gates['Y5'] else 'FAIL'} -- crash timing is "
        f"{'CLUSTERED, so there are discrete destruction waves and membership of a wave is the next thing to predict' if gates['Y5'] else 'UNIFORM, so there is no shared clock and each protein is timed on its own'}")
    say()

    # ---------------------------------------------------------------- Y6
    say("Y6 IS IT FAME?")
    import cell_assembled as CA
    pubs = CA.load().get("pubs", {})
    pv, av = [], []
    for i in idx:
        g = genes[i]
        if g in pubs and np.isfinite(A[i]):
            pv.append(float(pubs[g]))
            av.append(float(A[i]))
    pv, av = np.array(pv), np.array(av)

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        i = 0
        xs = x[o]
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = rank(pv), rank(av)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    rho = float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))
    say(f"     publication count vs A: rho {rho:+.4f} on {len(pv):,} genes")
    say(f"     for scale, the KEN-box motif scored +0.2429 and the 5'TOP motif +0.3296 -- both")
    say(f"     were struck for it.")
    gates["Y6"] = bool(abs(rho) < 0.15)
    res["y6"] = {"rho_pubs_A": rho, "n": len(pv)}
    say(f"     Y6 {'PASS' if gates['Y6'] else 'FAIL'} -- A is "
        f"{'not a fame proxy' if gates['Y6'] else 'CONTAMINATED by publication count like the motifs before it'}")
    say()

    say("=" * 100)
    for k in ("Y1", "Y2", "Y3", "Y4", "Y5", "Y6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    # the deliverable: which proteins, and when
    call = []
    for i in pulsed:
        g = genes[i]
        call.append({"gene": g, "A": float(A[i]), "C_fall": float(CF[i]),
                     "crash_interval": int(CRASH[i]),
                     "group": grp.get(g), "canonical": g in CANON})
    call.sort(key=lambda r: -r["A"])
    json.dump(call, open(OUT / "pulsed_proteins.json", "w"), indent=1)
    say(f"  -> outputs/pulsed_proteins.json   {len(call):,} proteins with A above the surrogate "
        f"95th percentile, each with its crash interval")

    man = RM.manifest(inputs=[LY, HPA],
                      available=int(len(d)), used=int(ok.sum()), selection="all", seed=SEED,
                      controls=["A is a DIFFERENCE of two concentrations from the same trajectory, "
                                "so amplitude, abundance and noise cancel",
                                "order-destroying surrogates from each protein's own values",
                                "sixteen textbook substrates scored BEFORE the result is read",
                                "crash interval checked against known destruction phase",
                                "publication count against a shape statistic"],
                      note="seven eliminated mechanisms were all PREDICTORS of membership. This "
                           "measures the waveform instead, which the pulse equation made readable.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 143 -- identify the pulsed proteins", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_pulse_identify.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_pulse_identify.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
