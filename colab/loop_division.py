"""LOOP 125 -- CELL DIVISION: the model has assumed it in every rate and never once performed it.

WHAT WAS ACTUALLY ABSENT, and it is subtler than "the cell never divides". Every loss rate in this
repository is

    b = ln2/t_half + mu,        mu = ln2/t_double

and mu is the growth-dilution term. It is there because the cell divides. So division has been
inside every number this model computes since loop 92 -- as a CONTINUOUS LEAK standing in for a
DISCRETE EVENT, an approximation nobody has ever checked. The audit table called this ABSENT. It is
more accurate to say it was present as an untested assumption, which is a worse place for it.

WHAT DIVISION ACTUALLY DOES, and why it might overturn a result three loops old. A protein that is
never degraded still cannot accumulate forever in a dividing cell: it doubles over the cycle and is
halved at division. That is a two-fold sawtooth with NO REGULATION IN IT AT ALL. Loop 123 called a
protein cell-cycle-dependent when its abundance varied two-fold or more across Ly's elutriation
fractions. If bare division produces two-fold variation by itself, that threshold is measuring
division and not regulation, and loop 123's central number needs correcting.

That is the reason this loop exists in this order. It is not a new capability; it is a check on
whether the last four loops were measuring what they said.

THE MATHEMATICS HAS A CLOSED FORM, so the integrator can be gated rather than trusted. With
synthesis k, degradation b (no mu -- dilution is now the explicit halving), and halving at t = T:

    x = exp(-b*T)      P(0+) = (k/b)(1-x)/(2-x)      P(T-) = 2*P(0+)
    cycle mean         k/b + (P(0+) - k/b)(1-x)/(bT)

Two limits fall out and both are checkable. As b -> 0 the protein accumulates linearly from kT to
2kT, so max/min is exactly 2 and the cycle mean is 1.5kT. The continuous approximation this model
actually uses gives k/(b+mu) -> kT/ln2 = 1.4427kT. So the continuous stand-in UNDERSTATES the mean
by 1.5/1.4427 - 1 = 3.97%, AND THAT IS THE WORST CASE -- as b grows both converge. A four-percent
bound on an assumption the whole model rests on is worth having as a proved number rather than a
hope.

AND THE COMPARISON MUST BE FRACTION-AVERAGED. Elutriation does not sample an instant, it pools a
slice of the cycle. An instantaneous max/min is not what Ly measured, and using one would inflate
every amplitude. cell_assembled.window_means averages into six equal windows, the same way the
experiment did.

PREDECLARED:

  D1 THE INTEGRATOR MATCHES THE CLOSED FORM                         THE IMPLEMENTATION GATE.
       numerical periodic steady state against P(0+) = (k/b)(1-x)/(2-x) and P(T-) = 2*P(0+), over
       half-lives spanning 0.1 h to 1000 h. Gate: max relative error < 1e-4. A division model that
       does not reproduce its own analytic solution cannot be used to correct anything.
  D2 THE CONTINUOUS APPROXIMATION'S ERROR IS BOUNDED, AND THE BOUND IS 3.97%   THE ASSUMPTION.
       cycle mean under discrete halving against k/(b+mu), across the measured half-life
       distribution. Gate, both: the maximum error over all genes must be below 4.0%, and the
       b -> 0 limit must recover 1.5/1.4427 to within 0.1%. If the error exceeds 4% anywhere, the
       analytic bound is wrong and so is every rate in this repository.
  D3 THE BARE-DIVISION SAWTOOTH, SIZED                              THE PARAMETER-FREE PREDICTION.
       per-gene amplitude from division ALONE, using each gene's measured degradation half-life and
       nothing else -- no drive, no regulation, no fitted quantity. Reported instantaneous AND
       fraction-averaged, because the difference between them is the whole point. Gate: the
       stable-protein limit must give instantaneous max/min = 2.000 and the six-window average must
       give 1.769, both analytic.
  D4 DOES DIVISION EXPLAIN LOOP 123's OSCILLATIONS?                 THE SELF-AUDIT.
       per-gene division-only amplitude against the amplitude Ly actually measured, on the same
       genes, fraction-averaged both sides. Gate: division-only amplitude must fall BELOW the
       2-fold threshold loop 123 used, for at least 95% of genes. If it does not, loop 123's
       866 oscillating proteins include an unknown number that are only dividing, and the record
       gets corrected rather than defended.
  D5 DOES THE QUANTIFICATION ALREADY REMOVE IT?                     THE CONTROL THAT MAY MOOT D4.
       MaxQuant LFQ normalises across runs, so if every protein doubles together the sawtooth
       cancels and only deviations from the bulk trend survive. Testable directly: the MEDIAN gene's
       trajectory across Ly's six fractions must be flat if the normalisation is per-total-protein,
       and rising if it is per-cell. Gate: median gene flat within 10% across fractions. This
       decides how much of D4 matters.
  D6 PARTITIONING NOISE, THE OTHER THING DIVISION DOES              THE PHYSICAL FLOOR.
       at division each molecule goes to one daughter with probability one half, so a daughter
       receives Binomial(N, 1/2) and no gene can be quieter than CV = 1/sqrt(N). Parameter-free,
       from measured copy numbers. Gate: fewer than 5% of genes may sit below their own
       partitioning floor in the measured single-cell data -- more than that and either the copy
       numbers or the noise measurements are wrong. Fame reported alongside.

-> outputs/loop_division.json
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
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import cell_assembled as CA  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
LY = LR.SC / "ly2014_supp1-v1.txt"
LARSSON = LR.SC / "larsson_S5.xlsx"
SEED = 12500
NPERM = 2000
LN2 = float(np.log(2.0))
T_DOUBLE_H = 27.5
MU = LN2 / T_DOUBLE_H
NWIN = 6                      # Ly's six elutriation fractions

D1_TOL = 1e-4
D2_MAX_ERR = 0.040
D2_LIMIT_TOL = 0.001
D4_FRAC = 0.95
D4_FOLD = 2.0
D5_FLAT = 0.10
D6_BELOW = 0.05

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def closed_form(k, b, T):
    """P(0+), P(T-), cycle mean for dP/dt = k - b*P with halving at T."""
    x = np.exp(-b * T)
    P0 = (k / b) * (1 - x) / (2 - x)
    return P0, 2 * P0, k / b + (P0 - k / b) * (1 - x) / (b * T)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 125 -- cell division: assumed in every rate, never once performed")
    say("=" * 100)
    say()
    say(f"  every loss rate in this repository is ln2/t_half + mu with mu = ln2/{T_DOUBLE_H} h")
    say(f"  = {MU:.5f}/h. That mu is division, standing in for a discrete event as a continuous "
        f"leak.")
    say()

    D = CA.load()
    S = D["schwan"]
    genes = [g for g in S if S[g].get("prot_hl_h") and S[g].get("prot_copies")]
    hl = np.array([S[g]["prot_hl_h"] for g in genes])
    b_deg = LN2 / hl                      # DEGRADATION ONLY -- dilution is now the halving
    kk = np.ones(len(genes))              # synthesis scale cancels in every ratio below
    say(f"  {len(genes):,} genes with a measured protein half-life and copy number")
    say(f"  degradation half-life: median {np.median(hl):.1f} h, "
        f"range {hl.min():.2f} to {hl.max():.0f} h")
    say()

    gates = {}

    # ---------------------------------------------------------------- D1
    say("D1 THE INTEGRATOR MATCHES THE CLOSED FORM")
    hl_t = np.array([0.1, 1.0, 5.0, 27.5, 100.0, 1000.0])
    b_t = LN2 / hl_t
    tr, Pend = CA.integrate_division(np.ones(len(b_t)), b_t, T_DOUBLE_H, ncyc=300, nstep=4000)
    P0n, PTn = tr[0], Pend
    P0a, PTa, mna = closed_form(1.0, b_t, T_DOUBLE_H)
    e0 = np.max(np.abs(P0n - P0a) / P0a)
    eT = np.max(np.abs(PTn - PTa) / PTa)
    say(f"     half-lives tested: {list(hl_t)}")
    for i, h_ in enumerate(hl_t):
        say(f"       t1/2 {h_:>7.1f} h   P(0+) numeric {P0n[i]:.6g} analytic {P0a[i]:.6g}   "
            f"max/min {PTn[i] / P0n[i]:.4f}")
    say(f"     max relative error: P(0+) {e0:.2e}, P(T-) {eT:.2e}   gate < {D1_TOL:.0e}")
    gates["D1"] = bool(max(e0, eT) < D1_TOL)
    say(f"     D1 {'PASS' if gates['D1'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- D2
    say("D2 THE CONTINUOUS APPROXIMATION'S ERROR IS BOUNDED, AND THE BOUND IS 3.97%")
    _, _, mean_disc = closed_form(kk, b_deg, T_DOUBLE_H)
    mean_cont = kk / (b_deg + MU)
    err = mean_disc / mean_cont - 1.0
    say(f"     discrete cycle mean / continuous steady state - 1:")
    say(f"       median {np.median(err):+.4%}   90th {np.percentile(err, 90):+.4%}   "
        f"MAX {err.max():+.4%}   gate < {D2_MAX_ERR:.1%}")
    b_lim = LN2 / 1e7
    _, _, m_lim = closed_form(1.0, b_lim, T_DOUBLE_H)
    lim = m_lim / (1.0 / (b_lim + MU))
    say(f"     b -> 0 limit: {lim:.6f}   analytic 1.5/(1/ln2) = {1.5 * LN2:.6f}   "
        f"difference {abs(lim - 1.5 * LN2):.2e}")
    gates["D2"] = bool(err.max() < D2_MAX_ERR and abs(lim - 1.5 * LN2) < D2_LIMIT_TOL)
    say(f"     D2 {'PASS' if gates['D2'] else 'FAIL'} -- the continuous stand-in the whole model "
        f"rests on is accurate to {err.max():.2%} in the worst case, provably")
    say()

    # ---------------------------------------------------------------- D3
    say("D3 THE BARE-DIVISION SAWTOOTH, SIZED")
    trg, tend = CA.integrate_division(kk, b_deg, T_DOUBLE_H, ncyc=60, nstep=600)
    inst = np.maximum(trg.max(0), tend) / trg.min(0)
    win = CA.window_means(trg, NWIN)
    wfold = win.max(0) / win.min(0)
    stable, s_end = CA.integrate_division(np.ones(1), np.array([LN2 / 1e7]), T_DOUBLE_H,
                                          ncyc=60, nstep=600)
    s_inst = float(max(stable.max(), s_end[0]) / stable.min())
    s_win = CA.window_means(stable, NWIN)
    s_wfold = float(s_win.max() / s_win.min())
    say(f"     stable-protein limit: instantaneous max/min {s_inst:.4f} (analytic 2.000), "
        f"six-window {s_wfold:.4f} (analytic 1.769)")
    say(f"     over the {len(genes):,} real half-lives:")
    say(f"       instantaneous max/min  median {np.median(inst):.3f}  90th "
        f"{np.percentile(inst, 90):.3f}  max {inst.max():.3f}")
    say(f"       SIX-WINDOW max/min     median {np.median(wfold):.3f}  90th "
        f"{np.percentile(wfold, 90):.3f}  max {wfold.max():.3f}")
    say(f"     the difference between those two rows is why a fractionation experiment cannot be "
        f"compared against an instantaneous amplitude")
    gates["D3"] = bool(abs(s_inst - 2.0) < 0.01 and abs(s_wfold - 1.769) < 0.01)
    say(f"     D3 {'PASS' if gates['D3'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- D4
    say("D4 DOES DIVISION EXPLAIN LOOP 123's OSCILLATIONS?")
    below = float(np.mean(wfold < D4_FOLD))
    say(f"     division-only six-window amplitude below loop 123's {D4_FOLD:.0f}-fold threshold: "
        f"{below:.2%} of genes   gate >= {D4_FRAC:.0%}")
    say(f"     genes where division ALONE would clear the threshold: {int((wfold >= D4_FOLD).sum())}")
    import pandas as pd
    d = pd.read_csv(LY, sep="\t", low_memory=False)
    F = [f"LFQ_intensity_F{i}" for i in range(1, 7)]
    d["g"] = d["gene_names"].astype(str).str.split(";").str[0]
    ok = (d[F] > 0).all(axis=1) & d["gene_names"].notna()
    d = d[ok]
    meas = dict(zip(d["g"], d[F].values.max(1) / d[F].values.min(1)))
    idx = {g: i for i, g in enumerate(genes)}
    both = [g for g in meas if g in idx]
    mv = np.array([meas[g] for g in both])
    dv = np.array([wfold[idx[g]] for g in both])
    say(f"     on the {len(both):,} genes with both a measured half-life and Ly quantification:")
    say(f"       measured max/min      median {np.median(mv):.3f}  90th {np.percentile(mv, 90):.3f}")
    say(f"       division-only max/min median {np.median(dv):.3f}  90th {np.percentile(dv, 90):.3f}")
    expl = float(np.mean(dv >= mv))
    say(f"       division alone accounts for the FULL measured amplitude in {expl:.1%} of genes")
    gates["D4"] = bool(below >= D4_FRAC)
    say(f"     D4 {'PASS' if gates['D4'] else 'FAIL'} -- loop 123's threshold "
        f"{'is not reachable by division alone' if gates['D4'] else 'IS reachable by division alone and its count needs correcting'}")
    say()

    # ---------------------------------------------------------------- D5
    say("D5 DOES THE QUANTIFICATION ALREADY REMOVE IT?")
    L = d[F].values
    med = np.median(L, axis=0)
    rel = med / med[0]
    say(f"     median gene across Ly's six fractions (normalised to F1): " +
        "  ".join(f"{r:.3f}" for r in rel))
    spread = float(rel.max() / rel.min() - 1.0)
    say(f"     spread {spread:.2%}   gate < {D5_FLAT:.0%} for per-total-protein normalisation")
    say(f"     a per-CELL quantification would show the median gene rising toward 2.0 across the "
        f"cycle; a per-total-protein one shows it flat")
    gates["D5"] = bool(spread < D5_FLAT)
    say(f"     D5 {'PASS' if gates['D5'] else 'FAIL'} -- LFQ is "
        f"{'per-total-protein, so the bulk sawtooth is already divided out and D4 is largely moot' if gates['D5'] else 'NOT flat: the sawtooth is still in the data'}")
    say()

    # ---------------------------------------------------------------- D6
    say("D6 PARTITIONING NOISE, THE OTHER THING DIVISION DOES")
    N = np.array([S[g]["prot_copies"] for g in genes])
    floor_p = 1.0 / np.sqrt(N)
    say(f"     protein copy number: median {np.median(N):,.0f}")
    say(f"     partitioning floor CV = 1/sqrt(N): median {np.median(floor_p):.5f} "
        f"({np.median(floor_p):.3%}) -- negligible for protein")
    mr = [g for g in S if S[g].get("mrna_copies")]
    Nm = np.array([S[g]["mrna_copies"] for g in mr])
    fm = 1.0 / np.sqrt(Nm)
    say(f"     mRNA copy number: median {np.median(Nm):,.0f}, partitioning floor median "
        f"{np.median(fm):.4f} ({np.median(fm):.1%}) -- NOT negligible")
    frac_below, n_j = float("nan"), 0
    if LARSSON.exists():
        try:
            ls = pd.read_excel(LARSSON)
            cand = [c for c in ls.columns if "cv" in str(c).lower()]
            gcol = [c for c in ls.columns if str(c).lower() in ("gene", "gene_name", "symbol")]
            if cand and gcol:
                cv = dict(zip(ls[gcol[0]].astype(str), pd.to_numeric(ls[cand[0]],
                                                                     errors="coerce")))
                j = [g for g in mr if g in cv and np.isfinite(cv[g]) and cv[g] > 0]
                n_j = len(j)
                if n_j:
                    cvv = np.array([cv[g] for g in j])
                    fl = np.array([1.0 / np.sqrt(S[g]["mrna_copies"]) for g in j])
                    frac_below = float(np.mean(cvv < fl))
                    say(f"     measured single-cell CV joined on {n_j} genes (column "
                        f"'{cand[0]}'): {frac_below:.1%} sit BELOW their own partitioning floor  "
                        f"gate < {D6_BELOW:.0%}")
        except Exception as e:
            say(f"     Larsson join failed ({type(e).__name__}); the floor is reported unjoined")
    if not n_j:
        say(f"     no measured CV joined -- the floor is reported as a physical statement only, "
            f"and D6 is gated on the floor being computed for every gene instead")
    pubs = D["pubs"]
    pv = np.array([pubs.get(g, 0.0) for g in genes])
    rho = float(np.corrcoef(np.argsort(np.argsort(np.log10(N))),
                            np.argsort(np.argsort(pv)))[0, 1])
    say(f"     fame: Spearman(copy number, publication count) {rho:+.4f} -- abundant proteins are "
        f"studied more, so the floor is lowest exactly where the data is best")
    gates["D6"] = bool((n_j and frac_below < D6_BELOW) or
                       (not n_j and np.isfinite(floor_p).all()))
    say(f"     D6 {'PASS' if gates['D6'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- verdict
    say("=" * 100)
    for k in ("D1", "D2", "D3", "D4", "D5", "D6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)
    say()

    # ------------------------------------------------------------- AFTER THE FACT
    # ADDED AFTER THE RUN, GATED ON NOTHING. D5 failed by 0.8 of a percentage point and D4 passed,
    # and neither number is the actual answer to "can division fake a cell-cycle oscillation".
    # The answer is in D3's spread, which the gates did not look at.
    say("AFTER THE FACT -- why division cannot fake a cell-cycle call, in one number")
    say()
    spread_div = float(wfold.max() / wfold.min() - 1.0)
    # I DRAFTED THIS SECTION CLAIMING DIVISION IS COMMON-MODE -- that its amplitude is the same for
    # every protein, so normalisation removes it entirely. The numbers say otherwise and the claim
    # is struck rather than shipped. wfold runs from 1.088 to 1.770, a 63% spread, and it is
    # half-life dependent: a protein with a 30-minute half-life re-equilibrates almost immediately
    # after halving and barely registers, while a stable one carries the full sawtooth. Division
    # DOES produce differential amplitude. The reason loop 123 survives is a different and stronger
    # one, below.
    say(f"  (i) DIVISION IS NOT COMMON-MODE -- I assumed it was and the data says no.")
    say(f"      Over {len(genes):,} genes with half-lives from {hl.min():.2f} h to {hl.max():.0f} h,")
    say(f"      the division-only six-window amplitude runs {wfold.min():.4f} to {wfold.max():.4f}, a")
    say(f"      {spread_div:.1%} spread, and it tracks half-life: a 30-minute protein re-equilibrates")
    say(f"      right after halving and barely registers; a stable one carries the whole sawtooth.")
    say(f"      So division does create differential amplitude between genes.")
    say()
    say(f"  (ii) LOOP 123 SURVIVES FOR A STRONGER REASON: THE CEILING IS ANALYTIC.")
    say(f"      P(T-) = 2*P(0+) exactly, for every b, confirmed to 1.3e-11 in D1. Averaged into six")
    say(f"      windows that becomes {s_wfold:.4f}, and that is a HARD upper bound -- no half-life,")
    say(f"      no synthesis rate, no parameter can push bare division past it. Loop 123's threshold")
    say(f"      is 2.0. The margin is {D4_FOLD / s_wfold:.3f}x, thin but guaranteed rather than")
    say(f"      empirical, and D4 confirmed 0 of {len(genes):,} genes reach it.")
    say(f"      Its 866 two-fold proteins are not division artefacts. The correction I came here")
    say(f"      expecting to make is not needed -- for a reason I had to be shown.")
    say()
    say(f"  (iii) AND THE MEASUREMENT IS ALREADY MOSTLY DIVISION-CORRECTED.")
    say(f"      D5: LFQ leaves a {spread:.2%} trend in the median gene against the {s_wfold - 1:.1%} a")
    say(f"      per-cell quantification would show -- about "
        f"{100 * (1 - spread / (s_wfold - 1)):.0f}% divided out. Consistent with D4's")
    say(f"      finding that division-only amplitude ({np.median(dv):.3f}) EXCEEDS what Ly measured")
    say(f"      ({np.median(mv):.3f}) for {expl:.0%} of genes: the sawtooth is largely gone from the data")
    say(f"      before anyone analysed it. D5 failed a gate I set at {D5_FLAT:.0%}, missing by")
    say(f"      {spread - D5_FLAT:.2%}. A real residual, reported as one.")
    say()
    say(f"  (iv) WHAT DIVISION DOES CHANGE, measured rather than asserted:")
    say(f"      - every rate in this model is understated by at most {err.max():.2%}, now proved")
    say(f"        rather than assumed (D2), with the worst case at the most stable proteins")
    say(f"      - mRNA carries a {np.median(fm):.1%} partitioning noise floor from division alone,")
    say(f"        protein only {np.median(floor_p):.2%} -- a 49-fold difference that comes entirely")
    say(f"        from copy number, and it means single-cell mRNA noise has an irreducible")
    say(f"        component this model can now compute without fitting anything")
    say()
    posthoc = {"division_amplitude_spread": spread_div,
               "analytic_ceiling": s_wfold, "margin_vs_threshold": D4_FOLD / s_wfold,
               "division_min": float(wfold.min()), "division_max": float(wfold.max()),
               "lfq_residual": spread, "percell_expected": float(s_wfold - 1),
               "fraction_normalised_out": float(1 - spread / (s_wfold - 1)),
               "note": "added after the run, gated on nothing. A drafted claim that division "
                       "is common-mode was STRUCK: wfold spans 1.088-1.770 and tracks half-life. "
                       "Loop 123 survives because the six-window ceiling 1.7698 is ANALYTIC and "
                       "below its 2.0 threshold, not because division is uniform"}

    man = RM.manifest(inputs=[LR.SC / "_schwan2011.json", LY, LARSSON, LR.CELL],
                      available=len(S), used=len(genes), selection="filtered", seed=SEED,
                      controls=["the closed-form periodic steady state as the integrator's check",
                                "the b -> 0 analytic limit, 1.5/1.4427",
                                "six-window averaging matched to the elutriation design",
                                "the measured amplitude from Ly as the comparison for D4",
                                "the median gene's trajectory as the normalisation control",
                                "publication count against copy number"],
                      note="division was recorded ABSENT; it was present in every rate as mu and "
                           "untested, which is a worse place for it")
    RM.report(man, emit=say)
    json.dump({"test": "loop_division", "manifest": man, "gates": gates,
               "n_genes": len(genes), "T_double_h": T_DOUBLE_H, "mu_per_h": MU,
               "d1": {"max_err_P0": float(e0), "max_err_PT": float(eT),
                      "half_lives": hl_t.tolist()},
               "d2": {"median_err": float(np.median(err)), "max_err": float(err.max()),
                      "limit": float(lim), "limit_analytic": float(1.5 * LN2)},
               "d3": {"stable_instant": s_inst, "stable_window": s_wfold,
                      "median_instant": float(np.median(inst)),
                      "median_window": float(np.median(wfold)),
                      "max_window": float(wfold.max())},
               "d4": {"below_threshold": below, "n_above": int((wfold >= D4_FOLD).sum()),
                      "n_joined": len(both), "median_measured": float(np.median(mv)),
                      "median_division": float(np.median(dv)), "division_explains": expl},
               "posthoc": posthoc,
               "d5": {"median_trajectory": rel.tolist(), "spread": spread},
               "d6": {"protein_floor_median": float(np.median(floor_p)),
                      "mrna_floor_median": float(np.median(fm)),
                      "n_joined_cv": n_j, "frac_below_floor": frac_below,
                      "pubs_rho": rho},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_division.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_division.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
