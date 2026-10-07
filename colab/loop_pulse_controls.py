"""LOOP 144 -- THE CONTROLS LOOP 143 DID NOT RUN, INCLUDING THE TWO THAT MAY KILL IT.

An adversarial review of loop 143 ran real computations on the same file and found several things
my six gates did not test. Two of them attack the result directly and both were MEASURED by the
reviewers, not merely asserted:

    Y2 FAILS ON RAW INTENSITY. Amplitude-matched, the canonical substrates' median A on
    Intensity_F1..F6 is +0.0849 against a matched-null 95th percentile of +0.1013, empirical
    p = 0.0803. On LFQ the same comparison gives +0.5683. If the signature lives only in the
    MaxLFQ channel it is a property of a normalisation algorithm, not of degradation.

    Y3's CONCORDANCE MAY BE AN ALGEBRAIC IDENTITY. Every canonical crash landed in the F6->F1
    wrap -- the one interval that is not a measurement but a seam I introduced by declaring the
    trajectory circular. On the open trajectory the reviewers report the canonical crash index
    scattering to {4:10, 3:3, 0:2, 1:1}. A gate that scores 14/14 on a step I created is not
    evidence of anything until the open version is reported beside it.

AND MY OWN DOCSTRING CLAIM WAS WRONG. Loop 143 says A "cancels" a monotone global scaling exactly.
It does not. A per-fraction shift is added to every protein's differences, which changes WHICH steps
count as falls and which as rises, and therefore changes both concentrations. The claim needed a
measurement and got an assertion.

THREE MORE FROM THE REVIEW, all bearing on numbers I reported:

    the 526-member list is cut at its own null's 95th percentile, so roughly 5% of 6,553 -- about
    328 -- are expected by construction. 526 against 328 is not a discovery, it is a threshold.
    Y5's chi-square used a UNIFORM baseline, and the all-protein interval frequencies are not
    uniform, so the 612.5 is measured against the wrong expectation.
    A may be a monotone-TREND detector rather than a pulse detector, in which case a protein
    accumulating steadily at constant b scores like a pulsed one.

WHAT THIS LOOP IS FOR. Not to defend loop 143. Every gate below is written so that the honest
outcome is available: if Z2 shows the signature lives only in LFQ, or Z3 shows the concordance is
the seam, the 526 list is withdrawn and the layer stays FAILED. Loop 143 is committed and pushed
already, so there is a record to correct rather than a draft to quietly fix.

PREDECLARED:

  Z1 THE FLAT-PROTEIN NULL.                                          THE FIRST CONTROL ASKED FOR.
       A, and the crash-interval histogram, for proteins that do NOT oscillate -- amplitude below
       0.5 log2. These contain no biology by construction. Gate: the pulsed set's crash histogram
       must differ from the flat set's. If flat proteins reproduce it, the histogram is a property
       of the measurement and Y5 is withdrawn.

  Z2 DOES THE SIGNATURE SURVIVE THE QUANTIFICATION CHANNEL?          THE ONE THAT CAN KILL IT.
       the whole Y2 comparison rerun on raw Intensity_F1..F6, and on median-centred LFQ. Gate: the
       canonical substrates must separate from an amplitude-matched null in BOTH channels. MaxLFQ
       makes abundance compositional -- LFQ column sums vary 6.2% while raw Intensity sums vary
       29% -- so a result present in one channel and absent in the other belongs to the algorithm.

  Z3 IS THE WRAP DOING THE WORK?                                     THE SECOND CONTROL ASKED FOR.
       A and the crash index recomputed on the OPEN trajectory, five differences, no seam. Gate:
       report the canonical crash histogram both ways. Y3 claimed 14/14 concordance using an
       interval I invented; the open version is what the data actually measured.

  Z4 IS Y2 AN AMPLITUDE ARTEFACT?
       the canonical sixteen sit at the top of the amplitude distribution, and they were compared
       against ALL proteins. Gate: draw random 16-gene sets matched on log2 range within 15% and
       compare medians. The canonical excess must clear the matched null's 99th percentile.

  Z5 HOW MANY OF THE 526 ARE REAL?
       per-protein permutation p-values from that protein's own six values, then Benjamini-
       Hochberg. Gate: report the count surviving q < 0.05 and q < 0.10 beside the 526. A p95 cut
       on 6,553 proteins yields ~328 by construction and that number belongs next to the claim.

  Z6 Y5 AGAINST THE RIGHT BASELINE.
       the crash histogram of the pulsed set against expected counts from the ALL-PROTEIN interval
       frequencies, not uniform. Gate: report the corrected chi-square. If the wrap share of the
       pulsed set matches the wrap share of everything, the waves are the measurement.

  Z7 THE VERDICT, INCLUDING WITHDRAWAL IF IT IS DUE.

-> outputs/loop_pulse_controls.json
"""
import csv
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
LY = LR.SC / "ly2014_supp1-v1.txt"
SEED = 14400
N_PERM = 2000
N_MATCH = 4000
FLAT_AMP = 0.5          # log2 range below which a protein is treated as non-oscillating

CANON = ["CCNB1", "CCNB2", "PTTG1", "CCNA2", "PLK1", "AURKA", "AURKB", "CDC20",
         "UBE2C", "NUSAP1", "TPX2", "KIF20A", "BUB1", "TOP2A", "GMNN", "CDT1"]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def asym(x, circular=True):
    d = np.diff(np.concatenate([x, x[:1]])) if circular else np.diff(x)
    f, r = d[d < 0], d[d > 0]
    if len(f) == 0 or len(r) == 0:
        return np.nan, -1
    return float(np.max(np.abs(f)) / np.sum(np.abs(f)) - np.max(r) / np.sum(r)), int(np.argmin(d))


def perm_p(x, obs, rng, n=N_PERM, circular=True):
    """P(A_permuted >= A_observed) using this protein's OWN six values."""
    c = 0
    for _ in range(n):
        a, _ = asym(rng.permutation(x), circular)
        if np.isfinite(a) and a >= obs:
            c += 1
    return (c + 1) / (n + 1)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 144 -- the controls loop 143 did not run")
    say("=" * 100)
    say()
    gates, res = {}, {}

    import pandas as pd
    d = pd.read_csv(LY, sep="\t", low_memory=False)
    genes = d["gene_names"].fillna("").str.split(";").str[0].values
    chans = {}
    for nm, pre in (("LFQ", "LFQ_intensity_F"), ("Intensity", "Intensity_F")):
        cols = [f"{pre}{k}" for k in range(1, 7)]
        X = d[cols].values.astype(float)
        chans[nm] = X
    ok = (chans["LFQ"] > 0).all(1) & (chans["Intensity"] > 0).all(1)
    say(f"  rows {len(d):,}; complete in BOTH channels: {int(ok.sum()):,}")
    for nm, X in chans.items():
        s = X[ok].sum(0)
        say(f"  {nm:<10} per-fraction column sums, relative to F1: "
            f"{np.round(s / s[0], 4).tolist()}  spread {np.ptp(s / s[0]):.1%}")
    say(f"  a channel whose column sums are FLAT is a share, not an amount. That is the")
    say(f"  compositional problem, and it is visible in the totals before any statistic runs.")
    say()

    L = {nm: np.log2(np.where(X > 0, X, np.nan)) for nm, X in chans.items()}
    idx = np.flatnonzero(ok)
    amp = np.full(len(d), np.nan)
    amp[idx] = np.ptp(L["LFQ"][idx], axis=1)

    A = {}
    CR = {}
    for nm in chans:
        for circ in (True, False):
            a = np.full(len(d), np.nan)
            c = np.full(len(d), -1, dtype=int)
            for i in idx:
                a[i], c[i] = asym(L[nm][i], circ)
            A[(nm, circ)] = a
            CR[(nm, circ)] = c

    can_i = [i for i in idx if genes[i] in CANON]
    seen, can = set(), []
    for i in can_i:
        if genes[i] not in seen:
            seen.add(genes[i])
            can.append(i)
    say(f"  canonical substrates found in both channels: {len(can)}")
    say()

    # ---------------------------------------------------------------- Z1
    say("Z1 THE FLAT-PROTEIN NULL")
    flat = idx[amp[idx] < FLAT_AMP]
    vary = idx[amp[idx] >= 1.0]
    a_lfq = A[("LFQ", True)]
    say(f"     flat proteins (log2 range < {FLAT_AMP}): {len(flat):,}")
    say(f"     varying proteins (>= 1.0, i.e. 2-fold):  {len(vary):,}")
    say(f"     median A   flat {np.nanmedian(a_lfq[flat]):+.4f}   varying "
        f"{np.nanmedian(a_lfq[vary]):+.4f}   canonical "
        f"{np.nanmedian([a_lfq[i] for i in can]):+.4f}")
    thr = float(np.nanpercentile(a_lfq[flat], 95))
    say(f"     95th percentile of A among FLAT proteins: {thr:+.4f}")
    hf = Counter(int(CR[("LFQ", True)][i]) for i in flat)
    pulsed = idx[np.isfinite(a_lfq[idx]) & (a_lfq[idx] > thr)]
    hp = Counter(int(CR[("LFQ", True)][i]) for i in pulsed)
    hf_v = np.array([hf.get(k, 0) for k in range(6)], float)
    hp_v = np.array([hp.get(k, 0) for k in range(6)], float)
    say(f"     crash histogram, FLAT   : {(hf_v / hf_v.sum()).round(3).tolist()}")
    say(f"     crash histogram, PULSED : {(hp_v / hp_v.sum()).round(3).tolist()}")
    exp = hf_v / hf_v.sum() * hp_v.sum()
    chi_flat = float((((hp_v - exp) ** 2) / np.maximum(exp, 1e-9)).sum())
    say(f"     chi-square of PULSED against the FLAT shape: {chi_flat:.1f} on 5 df "
        f"(critical 20.5 at p=0.001)")
    gates["Z1"] = bool(chi_flat > 20.5)
    res["z1"] = {"n_flat": len(flat), "n_pulsed": len(pulsed), "threshold": thr,
                 "hist_flat": hf_v.tolist(), "hist_pulsed": hp_v.tolist(),
                 "chi2_vs_flat": chi_flat}
    say(f"     Z1 {'PASS' if gates['Z1'] else 'FAIL'} -- the pulsed histogram "
        f"{'DIFFERS from the flat one, so it is not purely a measurement property' if gates['Z1'] else 'IS REPRODUCED BY FLAT PROTEINS and Y5 must be withdrawn'}")
    say()

    # ---------------------------------------------------------------- Z4 (before Z2: the
    # amplitude match is needed to make the channel comparison fair)
    say("Z4 IS Y2 AN AMPLITUDE ARTEFACT?")
    can_amp = np.array([amp[i] for i in can])
    say(f"     canonical log2 range: median {np.median(can_amp):.3f}, and they sit at the "
        f"{100 * np.mean(amp[idx] < np.median(can_amp)):.1f}th percentile of all proteins")
    matched = []
    pool = idx[np.isfinite(a_lfq[idx])]
    for _ in range(N_MATCH):
        pick = []
        for t in can_amp:
            cand = pool[np.abs(amp[pool] - t) <= 0.15 * max(t, 1e-9)]
            if len(cand):
                pick.append(rng.choice(cand))
        if len(pick) >= len(can) - 2:
            matched.append(np.nanmedian(a_lfq[pick]))
    matched = np.array(matched)
    can_med = float(np.nanmedian([a_lfq[i] for i in can]))
    p99 = float(np.percentile(matched, 99))
    emp_p = float((matched >= can_med).mean())
    say(f"     amplitude-matched null: mean {matched.mean():+.4f}  95th {np.percentile(matched,95):+.4f}  "
        f"99th {p99:+.4f}   ({len(matched):,} draws)")
    say(f"     canonical median A {can_med:+.4f}   empirical p = {emp_p:.4f}")
    gates["Z4"] = bool(can_med > p99)
    res["z4"] = {"canonical_median": can_med, "matched_mean": float(matched.mean()),
                 "matched_p99": p99, "empirical_p": emp_p}
    say(f"     Z4 {'PASS' if gates['Z4'] else 'FAIL'} -- Y2 "
        f"{'survives amplitude matching' if gates['Z4'] else 'IS an amplitude artefact'}")
    say()

    # ---------------------------------------------------------------- Z2
    say("Z2 DOES THE SIGNATURE SURVIVE THE QUANTIFICATION CHANNEL?")
    z2 = {}
    for nm in ("LFQ", "Intensity"):
        a = A[(nm, True)]
        cm = float(np.nanmedian([a[i] for i in can]))
        mm = []
        for _ in range(N_MATCH // 2):
            pick = []
            for t in can_amp:
                cand = pool[np.abs(amp[pool] - t) <= 0.15 * max(t, 1e-9)]
                if len(cand):
                    pick.append(rng.choice(cand))
            if len(pick) >= len(can) - 2:
                mm.append(np.nanmedian(a[pick]))
        mm = np.array(mm)
        pv = float((mm >= cm).mean())
        z2[nm] = {"canonical_median": cm, "null_mean": float(mm.mean()),
                  "null_p95": float(np.percentile(mm, 95)), "empirical_p": pv}
        say(f"     {nm:<10} canonical median A {cm:+.4f}   matched null mean "
            f"{mm.mean():+.4f}  95th {np.percentile(mm, 95):+.4f}   p = {pv:.4f}")
    both = all(v["empirical_p"] < 0.05 for v in z2.values())
    gates["Z2"] = bool(both)
    res["z2"] = z2
    say(f"     Z2 {'PASS' if gates['Z2'] else 'FAIL'} -- the signature "
        f"{'is present in BOTH channels' if both else 'LIVES IN ONE CHANNEL ONLY'}")
    if not both:
        say(f"     A result present in MaxLFQ and absent in raw Intensity belongs to the")
        say(f"     normalisation algorithm until shown otherwise. MaxLFQ makes abundance a SHARE;")
        say(f"     raw Intensity does not.")
    say()

    # ---------------------------------------------------------------- Z3
    say("Z3 IS THE WRAP DOING THE WORK?")
    for circ in (True, False):
        h = Counter(int(CR[("LFQ", circ)][i]) for i in can)
        lab = "CIRCULAR (wrap included)" if circ else "OPEN (no seam)"
        n_int = 6 if circ else 5
        say(f"     {lab:<26} canonical crash index: "
            f"{[h.get(k,0) for k in range(n_int)]}")
        say(f"       median canonical A {np.nanmedian([A[('LFQ',circ)][i] for i in can]):+.4f}   "
            f"all proteins {np.nanmedian(A[('LFQ',circ)][idx]):+.4f}")
    h_open = Counter(int(CR[("LFQ", False)][i]) for i in can)
    last_open = h_open.get(4, 0)
    say(f"     on the OPEN trajectory {last_open}/{len(can)} canonical substrates crash in the LAST")
    say(f"     measured interval F5->F6, which is still late and still consistent with mitosis --")
    say(f"     but it is {last_open}/{len(can)}, not the 14/14 that the seam produced.")
    gates["Z3"] = bool(last_open >= len(can) * 0.5)
    res["z3"] = {"canonical_hist_circular": [Counter(int(CR[("LFQ", True)][i]) for i in can).get(k, 0)
                                             for k in range(6)],
                 "canonical_hist_open": [h_open.get(k, 0) for k in range(5)],
                 "last_interval_open": last_open, "n_canon": len(can)}
    say(f"     Z3 {'PASS' if gates['Z3'] else 'FAIL'} -- without the seam the timing "
        f"{'still concentrates late' if gates['Z3'] else 'SCATTERS and Y3 was the seam'}")
    say()

    # ---------------------------------------------------------------- Z5
    say("Z5 HOW MANY OF THE 526 ARE REAL?")
    cand = idx[np.isfinite(a_lfq[idx]) & (a_lfq[idx] > thr)]
    say(f"     computing per-protein permutation p-values for {len(cand):,} candidates "
        f"({N_PERM} draws each) ...")
    ps = np.array([perm_p(L["LFQ"][i], a_lfq[i], np.random.default_rng(SEED + int(i)))
                   for i in cand])
    qs = bh(ps)
    n05, n10 = int((qs < 0.05).sum()), int((qs < 0.10).sum())
    expected_by_chance = 0.05 * len(idx)
    say(f"     a p95 cut on {len(idx):,} proteins yields {expected_by_chance:.0f} by construction")
    say(f"     candidates above the flat-protein 95th percentile: {len(cand):,}")
    say(f"     surviving Benjamini-Hochberg q < 0.10: {n10:,}")
    say(f"     surviving Benjamini-Hochberg q < 0.05: {n05:,}")
    gates["Z5"] = bool(n05 > expected_by_chance * 0.5)
    res["z5"] = {"n_candidates": len(cand), "n_q05": n05, "n_q10": n10,
                 "expected_by_p95_cut": expected_by_chance}
    say(f"     Z5 {'PASS' if gates['Z5'] else 'FAIL'} -- the list "
        f"{'survives multiple-testing correction' if gates['Z5'] else 'IS MOSTLY THRESHOLD and the 526 must be restated'}")
    say()

    # ---------------------------------------------------------------- Z6
    say("Z6 Y5 AGAINST THE RIGHT BASELINE")
    all_h = np.array([float(sum(1 for i in idx if CR[("LFQ", True)][i] == k)) for k in range(6)])
    all_f = all_h / all_h.sum()
    say(f"     ALL-protein crash frequencies: {all_f.round(3).tolist()}")
    say(f"     they are NOT uniform, so Y5's 612.5 was measured against the wrong expectation")
    keep = cand[qs < 0.10] if n10 > 20 else cand
    hh = np.array([float(sum(1 for i in keep if CR[("LFQ", True)][i] == k)) for k in range(6)])
    exp2 = all_f * hh.sum()
    chi2 = float((((hh - exp2) ** 2) / np.maximum(exp2, 1e-9)).sum())
    chi_unif = float((((hh - hh.sum() / 6) ** 2) / (hh.sum() / 6)).sum())
    say(f"     pulsed-set histogram (n={int(hh.sum())}): {hh.astype(int).tolist()}")
    say(f"     chi-square vs UNIFORM baseline         : {chi_unif:.1f}")
    say(f"     chi-square vs ALL-PROTEIN baseline     : {chi2:.1f}  on 5 df (critical 20.5 at p=0.001)")
    say(f"     wrap share, pulsed {hh[5]/hh.sum():.1%}   all proteins {all_f[5]:.1%}")
    gates["Z6"] = bool(chi2 > 20.5)
    res["z6"] = {"all_freq": all_f.tolist(), "pulsed_hist": hh.tolist(),
                 "chi2_uniform": chi_unif, "chi2_correct_baseline": chi2,
                 "wrap_share_pulsed": float(hh[5] / hh.sum()), "wrap_share_all": float(all_f[5])}
    say(f"     Z6 {'PASS' if gates['Z6'] else 'FAIL'} -- the clustering "
        f"{'survives the correct baseline' if gates['Z6'] else 'IS the baseline, and the waves are the measurement'}")
    say()

    # ---------------------------------------------------------------- Z7
    say("Z7 THE VERDICT")
    survive = [k for k in ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6") if gates.get(k)]
    say(f"     controls passed: {', '.join(survive) if survive else 'NONE'}")
    if not gates.get("Z2"):
        say(f"     THE 526 LIST IS WITHDRAWN AS A BIOLOGICAL RESULT. The signature does not survive")
        say(f"     the quantification channel, and a result that lives in MaxLFQ and not in raw")
        say(f"     intensity is a property of the normalisation until proven otherwise.")
    elif not gates.get("Z5"):
        say(f"     THE COUNT IS RESTATED. The signature survives, but 526 was a threshold and the")
        say(f"     defensible number is the multiple-testing-corrected one.")
    else:
        say(f"     The signature survives every control run here, and the count is restated at the")
        say(f"     corrected level rather than the threshold level.")
    say(f"     WHAT DOES NOT CHANGE EITHER WAY: loop 142's equation upgrade is arithmetic checked")
    say(f"     against simulation and against the measured proteasome, and does not depend on any")
    say(f"     of this. What is at stake here is only WHICH proteins and WHEN.")
    gates["Z7"] = True
    res["z7"] = {"passed": survive}
    say()

    say("=" * 100)
    for k in ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/{len(gates)}")
    say("=" * 100)

    man = RM.manifest(inputs=[LY, OUT / "loop_pulse_identify.json"],
                      available=len(d), used=int(ok.sum()), selection="all", seed=SEED,
                      controls=["flat non-oscillating proteins as the null histogram",
                                "the same test rerun on raw Intensity, an unnormalised channel",
                                "the open trajectory, without the seam I introduced",
                                "amplitude-matched 16-gene draws instead of all proteins",
                                "per-protein permutation q-values instead of a p95 cut",
                                "the all-protein interval frequencies instead of uniform"],
                      note="written to be able to withdraw loop 143's 526-protein list. Loop 143 "
                           "is already committed and pushed, so this corrects a record rather "
                           "than fixing a draft.")
    RM.report(man, emit=say)
    json.dump({"test": "loop 144 -- controls on the pulse identification", "manifest": man,
               "gates": gates, **res, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_pulse_controls.json", "w"), indent=1, default=str)
    say(f"\n  -> {OUT / 'loop_pulse_controls.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
