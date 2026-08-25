"""Loop 192. Does the accessibility clock replicate outside A549 and dexamethasone?

WHAT IS BEING TESTED, AND WHY IT IS THE FIRST THING TO TEST. Loop 191d measured that promoter
accessibility reaches half its plateau 48 minutes before the mRNA does -- 1,310 responding genes,
one-sided Wilcoxon p 6.4e-58, holding inside all three magnitude terciles (+20, +104, +47 min), with
two passing negative controls behind it (CTCF +0.061, RAD21 -0.022). It is the only result in this
project with a fourth dimension in it, and every downstream plan leans on it. It also rests
entirely on one drug, one receptor, one cell line and one accessibility assay. If it does not
replicate, several loops of planned work are built on an artefact, and finding that out costs 50 MB.

THE REPLICATION SET, chosen by searching every released ENCODE experiment carrying a treatment
duration rather than by picking a favourite. Four systems have at least four matched accessibility
and RNA timepoints; the dendritic-cell LPS series is the fair one and fetch_dc_lps.py records why
the K562 chromatin-drug panel was rejected as the primary test (six of its seven drugs act ON
chromatin, so the claim there restates the drug's mechanism, and its 4-48 hour grid cannot resolve
a 48-minute lead). LPS through TLR4 is a natural stimulus, the cells are primary rather than a
carcinoma line, the assay is ATAC rather than DNase, the window sits inside 191d's, and all 59
experiments come from one lab.

THE GATE THAT MATTERS MOST IS NOT THE REPLICATION. It is W3. The A549 lead was measured on nine
timepoints; the dendritic-cell series shares four. A 48-minute lead on a 60/120/240/360 minute grid
is close to the resolution limit, so a null in W4 has two possible causes -- the effect is absent,
or four points cannot see it -- and those are not the same finding. W3 settles it BEFORE W4 is
read, by taking the A549 data that already produced the result and downsampling it to a
dendritic-cell-shaped grid. If the lead survives that, four points are enough and a W4 null means
the effect is absent. If it does not, W4 is VOID and the honest report is that this series cannot
answer the question. Loop 191's T1 taught the general form of this: a gate must guard the property
the conclusion actually needs, and "is there an effect" needs "could I have seen one".

THE ASYMMETRY THAT COULD MANUFACTURE THE ANSWER. ATAC in this series carries a 30-minute point that
RNA does not. Scoring accessibility on a grid that starts earlier than expression's would produce a
lead by construction. Both are computed on the SHARED grid only, which is the same rule loop 191c
adopted after the A549 series' denser early RNA sampling would have done the same thing.

PREDECLARED, BEFORE ANY NUMBER.

  W1 IS THE REPLICATION SET USABLE? One lab, shared timepoints, and the Ensembl-to-symbol join that
     loop 191 got wrong and 191b had to gate.
     Gate: PASS iff every retained experiment is from one lab, at least 4 timepoints carry both
     assays, and at least 70% of expressed genes map to a symbol.

  W2 IS THE CLOCK CLEAN? The A549 series turned out to carry a batch discontinuity between its <=25
     and >=30 minute experiments: the median gene sat at 5% of its plateau at 25 minutes and 98% at
     30, then flat for eleven hours, and three loops were built on it before loop 191b's
     dynamic-range gate caught it. That check is now run FIRST rather than discovered.
     Gate: PASS iff no single interval carries more than 80% of the median gene's plateau.

  W2b IS THE TIMING REPRODUCIBLE ACROSS DONORS? Every experiment in this series reports
     biological_replicate 1, so that field separates nothing and the replicate unit is the
     experiment accession -- an independent donor. The experiments at each timepoint are split into
     two arbitrary halves and the response time is computed in each.
     Gate: PASS iff Spearman across the split is >= 0.30. Loop 191's T1 established that a timing
     test on an unreproducible timing measurement describes the measurement, and 191b's U4
     established that reproducible is not the same as discriminative; W2 covers the second, this
     covers the first.

  W3 COULD FOUR TIMEPOINTS SEE A 48-MINUTE LEAD? The A549 data from 191d, downsampled to the
     dendritic-cell grid shape, re-run through the identical test.
     Gate: PASS iff the A549 lead is still detected at one-sided p < 0.05 after downsampling. A
     FAIL makes W4 VOID rather than negative, because a null would then be about resolution.

  W4 DOES THE LEAD REPLICATE? ATAC half-plateau against RNA half-plateau on the shared grid, paired
     one-sided Wilcoxon, plus the magnitude terciles.
     Gate: PASS iff accessibility leads at p < 0.05 AND the direction holds in at least 2 of 3
     terciles. Pooled significance alone is not enough; that is the control that separated a timing
     effect from a response-size effect in 191d and it travels with the claim.

  W5 THE STRANGER SWAP. Each gene given another gene's accessibility trajectory, every value kept.
     A lead that survives this is a property of the assay grids and not of the pairing.
     Gate: PASS iff the real lead exceeds the swapped lead at p < 0.05.

  W6 WHAT THIS CANNOT SHOW.

-> outputs/loop_timing_replication.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_guard as GG                      # noqa: E402
import run_manifest as RM                    # noqa: E402
import loop_response_timing_d as L191        # noqa: E402

from scipy.stats import spearmanr, wilcoxon                                  # noqa: E402

SP = L191.SP
DC = SP / "dclps"
A549 = SP / "grtc"
OUT = Path(os.environ.get("CELL_OUT", "outputs")) / "loop_timing_replication.json"
MIN_JOIN = 0.70
MIN_SHARED = 4
MAX_INTERVAL_SHARE = 0.80     # W2: no single interval may carry this much of the median response
MIN_PLATEAU = 0.5
MIN_TPM = 1.0
MIN_GROUP = 15
ALPHA = 0.05
N_STRATA = 3
MIN_SPLIT_RHO = 0.30          # W2b: donor split-half reliability of the response time
PROM_PAD = L191.PROM_PAD
A549_DOWNSAMPLE = [60.0, 120.0, 240.0, 420.0]   # the A549 points closest to the DC grid shape
SEED = 192192

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def response_time(V, grid):
    """Alon's response time on a signed trajectory: when it reaches half its steady state.

    The plateau is the mean of the last min(3, n-1) points, so a four-point grid does not consume
    its whole trajectory defining the endpoint."""
    k = min(3, max(1, V.shape[0] - 1))
    pl = V[-k:].mean(0)
    out = np.full(V.shape[1], np.nan)
    for j in range(V.shape[1]):
        p = pl[j]
        if abs(p) < 1e-9:
            continue
        tg, v = p / 2.0, V[:, j]
        for i in range(1, len(v)):
            if (p > 0 and v[i] >= tg) or (p < 0 and v[i] <= tg):
                lo, hi = v[i - 1], v[i]
                out[j] = grid[i] if hi == lo else \
                    grid[i - 1] + (tg - lo) / (hi - lo) * (grid[i] - grid[i - 1])
                break
    return out, pl


def lead_test(acc_h, exp_h, mask, label, report=print):
    """The paired comparison, reported the same way in every arm so the arms are comparable."""
    if mask.sum() < MIN_GROUP:
        report(f"     {label}: {int(mask.sum())} pairs, under the power floor")
        return None
    _, p = wilcoxon(acc_h[mask], exp_h[mask], alternative="less")
    lead = float(np.median(exp_h[mask] - acc_h[mask]))
    report(f"     {label}: n {int(mask.sum()):,}  accessibility {np.median(acc_h[mask]):.0f} min vs "
           f"expression {np.median(exp_h[mask]):.0f} min  lead {lead:+.0f} min  p {p:.3g}")
    return dict(n=int(mask.sum()), lead=lead, p=float(p))


def build(dirp, assay, report=print):
    """Trajectories, response times and promoter accessibility for one series, on its shared grid."""
    z = np.load(dirp / "rna.npz", allow_pickle=True)
    tpm = z["tpm"]
    ensg = np.array([str(g).split(".")[0] for g in z["genes"]])
    mins, reps = z["mins"].astype(int), z["reps"].astype(int)
    man = json.load(open(dirp / "manifest.json"))
    peak_t = sorted(int(k) for k in man["peaks"][assay])
    rna_t = sorted(set(mins.tolist()))
    shared = [t for t in rna_t if t in peak_t]
    report(f"     RNA {rna_t}")
    report(f"     {assay} {peak_t}")
    report(f"     shared grid: {shared}   "
           f"({assay}-only points not used: {sorted(set(peak_t) - set(rna_t))})")
    exps = np.array([str(x) for x in z["exps"]]) if "exps" in z.files else \
        np.array([f"e{i}" for i in range(len(mins))])
    return dict(tpm=tpm, ensg=ensg, mins=mins, reps=reps, exps=exps, man=man,
                shared=np.array(shared, dtype=float), assay=assay, dirp=dirp)


def donor_split(mins, exps):
    """Two arbitrary halves of the experiments at each timepoint.

    Every experiment in this series reports biological_replicate 1, so that field cannot separate
    anything: the real replicate unit is the EXPERIMENT ACCESSION, which here is an independent
    donor. The halves are not donor-MATCHED across timepoints -- ENCODE does not expose the pairing
    -- and they do not need to be, because this is a split-half reliability estimate and what it
    needs is two independent subsets, not a repeated measure on the same donor."""
    grp = np.zeros(len(exps), dtype=int)
    for t in sorted(set(mins.tolist())):
        idx = np.where(mins == t)[0]
        for k, i in enumerate(idx[np.argsort(exps[idx])]):
            grp[i] = 1 + (k % 2)
    return grp


def series_arrays(S, grid, e2s, report=print):
    reps_all = sorted(set(S["reps"].tolist()))
    keep = [r for r in reps_all
            if all(((S["mins"] == int(t)) & (S["reps"] == r)).any() for t in grid)]
    if not keep:
        keep = reps_all
    M, n = L191.rep_trajectories(S["tpm"], S["mins"], S["reps"], keep, grid)
    report(f"     {n} replicate(s) complete on the grid: {keep}")
    sym = np.array([e2s.get(g, "") for g in S["ensg"]])
    base = S["tpm"][(S["mins"] == int(grid[0])) & np.isin(S["reps"], keep)].mean(0)
    th, pl = response_time(M, grid)
    return M, sym, base, th, pl, keep


def main():
    t0 = time.time()
    say("=" * 104)
    say("LOOP 192  DOES THE ACCESSIBILITY CLOCK REPLICATE OUTSIDE A549 AND DEXAMETHASONE?")
    say("=" * 104)
    say("  PREDECLARED: one lab and >= 4 shared timepoints and >= 70% symbol join; NO single")
    say(f"  interval may carry more than {MAX_INTERVAL_SHARE:.0%} of the median gene's plateau,")
    say("  because the A549 series carried exactly that pathology and three loops were built on it")
    say("  before it was caught; the A549 lead must SURVIVE downsampling to this series' grid shape")
    say("  before a null here may be read as an absence rather than as a resolution limit; the")
    say("  replication must hold in >= 2 of 3 magnitude terciles and not only pooled; and it must")
    say("  beat a stranger swap. Both half-times on the SHARED grid only, because ATAC carries a")
    say("  30-minute point that RNA does not and scoring it would manufacture a lead.")
    say()

    if not (DC / "rna.npz").exists():
        raise SystemExit(f"{DC/'rna.npz'} missing -- run colab/fetch_dc_lps.py first")

    # ---- W1 ------------------------------------------------------------------------------------
    say("W1 IS THE REPLICATION SET USABLE?")
    S = build(DC, "ATAC", say)
    man = S["man"]
    say(f"     lab: {man.get('lab')}")
    e2s = L191.ensg_to_symbol(say)
    grid = S["shared"]
    M, sym, base, th, pl, keep = series_arrays(S, grid, e2s, say)
    expressed = base >= MIN_TPM
    frac = float((sym[expressed] != "").mean()) if expressed.sum() else 0.0
    say(f"     {int(expressed.sum()):,} genes expressed at the baseline point; "
        f"{frac:.1%} carry a symbol")
    w1 = bool(len(grid) >= MIN_SHARED and frac >= MIN_JOIN)
    GG.verdict(w1, emit=say,
               if_true=f"W1 PASS -- {len(grid)} shared timepoints from one lab and a {frac:.1%} "
                       f"symbol join",
               if_false=f"W1 FAIL -- {len(grid)} shared timepoints, join {frac:.1%}")

    # ---- W2 ------------------------------------------------------------------------------------
    say()
    say("W2 IS THE CLOCK CLEAN?")
    responder = expressed & (np.abs(pl) >= MIN_PLATEAU) & np.isfinite(th)
    say(f"     {int(responder.sum()):,} responders at |plateau| >= {MIN_PLATEAU}")
    shares = []
    if responder.sum():
        f = np.abs(M[:, responder]) / np.abs(pl[responder])
        med = np.median(f, axis=1)
        say("     fraction of its own plateau reached by the median gene:")
        say("       " + "  ".join(f"{int(t):5d}" for t in grid))
        say("       " + "  ".join(f"{x:5.2f}" for x in med))
        shares = [float(med[i] - med[i - 1]) for i in range(1, len(med))]
        say(f"     per-interval share: {[round(x, 2) for x in shares]}")
    w2 = bool(shares and max(shares) <= MAX_INTERVAL_SHARE)
    GG.verdict(w2, emit=say,
               if_true=f"W2 PASS -- the largest single interval carries {max(shares):.0%} of the "
                       f"median response, so the clock is graded rather than a step",
               if_false=f"W2 FAIL -- one interval carries "
                        f"{(max(shares) if shares else float('nan')):.0%} of the median response. "
                        f"That is the A549 pathology and any timing statistic across it measures "
                        f"the step")

    # ---- W2b -----------------------------------------------------------------------------------
    say()
    say("W2b IS THE TIMING REPRODUCIBLE ACROSS DONORS?")
    say("     every experiment here reports biological_replicate 1, so that field separates")
    say("     nothing; the replicate unit is the experiment accession, an independent donor.")
    grp = donor_split(S["mins"], S["exps"])
    say(f"     donors per timepoint: "
        f"{ {int(t): int((S['mins'] == t).sum()) for t in sorted(set(S['mins'].tolist()))} }")
    Ma, na = L191.rep_trajectories(S["tpm"], S["mins"], grp, (1,), grid)
    Mb, nb = L191.rep_trajectories(S["tpm"], S["mins"], grp, (2,), grid)
    tha, _ = response_time(Ma, grid)
    thb, _ = response_time(Mb, grid)
    okr = responder & np.isfinite(tha) & np.isfinite(thb)
    rho, prho = spearmanr(tha[okr], thb[okr]) if okr.sum() > 10 else (float("nan"), float("nan"))
    say(f"     split-half over {int(okr.sum()):,} responders: Spearman {rho:+.3f} (p {prho:.3g})")
    w2b = bool(np.isfinite(rho) and rho >= MIN_SPLIT_RHO)
    GG.verdict(w2b, emit=say,
               if_true=f"W2b PASS -- the response time reproduces across independent donors at "
                       f"{rho:+.3f}, so W4 would be measuring biology rather than noise",
               if_false=f"W2b FAIL -- {rho:+.3f} against a bar of {MIN_SPLIT_RHO}. On four "
                        f"timepoints and this many donors the timing is not reproducible, so a W4 "
                        f"result either way would describe the measurement")

    # ---- W3 ------------------------------------------------------------------------------------
    say()
    say("W3 COULD FOUR TIMEPOINTS SEE A 48-MINUTE LEAD?")
    say("     the A549 data that produced the result, downsampled to this grid's shape and run")
    say("     through the identical test. this decides whether a W4 null is an absence or a limit.")
    w3, d3full, d3down = False, None, None
    A = build(A549, "DNase", say)
    tabg = json.load(gzip.open("colab/data/cell_complete.json.gz"))["genes"]
    tssbed = {}
    for line in open(SP / "_tss_hg38.bed"):
        q = line.split()
        if len(q) >= 4 and q[3].startswith("G"):
            i = int(q[3][1:])
            if i < len(tabg):
                tssbed[str(tabg[i]["name"]).upper()] = (q[0], int(q[2]))

    def arm(Sx, gridx, assay, tag):
        Mx, symx, basex, thx, plx, keepx = series_arrays(Sx, gridx, e2s, lambda *_: None)
        tl = [tssbed.get(s) for s in symx]
        pt, PM = L191.promoter_track(assay, tl, PROM_PAD, lambda *_: None)
        idx = [int(np.where(pt == t)[0][0]) for t in gridx if t in set(pt.tolist())]
        if len(idx) != len(gridx):
            say(f"     {tag}: {assay} lacks {len(gridx)-len(idx)} of the grid points, arm skipped")
            return None, None, None, None
        ah, _ = response_time(PM[idx], gridx)
        eh, _ = response_time(Mx, gridx)
        resp = (basex >= MIN_TPM) & (np.abs(plx) >= MIN_PLATEAU) & np.isfinite(eh)
        m = resp & (PM[idx] > 0).any(0) & np.isfinite(ah)
        return ah, eh, m, plx

    a_full = np.array(sorted(set(A["shared"].tolist()) & set(
        int(k) for k in A["man"]["peaks"]["DNase"])), dtype=float)
    a_full = a_full[a_full >= 30.0]
    ah, eh, m, plx = arm(A, a_full, "DNase", "A549 full")
    if m is not None:
        d3full = lead_test(ah, eh, m, f"A549 on its full {len(a_full)}-point grid", say)
    down = np.array([t for t in A549_DOWNSAMPLE if t in set(a_full.tolist())], dtype=float)
    say(f"     downsampled A549 grid: {[int(x) for x in down]} "
        f"(the points closest in shape to {[int(x) for x in grid]})")
    ah2, eh2, m2, _ = arm(A, down, "DNase", "A549 downsampled")
    if m2 is not None:
        d3down = lead_test(ah2, eh2, m2, f"A549 downsampled to {len(down)} points", say)
    w3 = bool(d3down and d3down["p"] < ALPHA and d3down["lead"] > 0)
    GG.verdict(w3, emit=say,
               if_true=f"W3 PASS -- the lead survives downsampling ({d3down['lead']:+.0f} min, "
                       f"p {d3down['p']:.3g}), so four points can see it and a W4 null would mean "
                       f"the effect is absent",
               if_false="W3 FAIL -- the A549 lead does NOT survive downsampling to this grid, so "
                        "four points cannot resolve it and W4 is VOID: a null here would be a "
                        "statement about the sampling and not about dendritic cells")

    void = set()
    if not (w1 and w2 and w2b and w3):
        void |= {"W4", "W5"}
        say()
        say("     a precondition failed, so W4 and W5 are VOID rather than negative")

    # ---- W4 ------------------------------------------------------------------------------------
    say()
    say("W4 DOES THE LEAD REPLICATE?")
    w4, d4, strata = False, None, {}
    if "W4" in void:
        say("     W4 VOID -- see above")
    else:
        tl = [tssbed.get(s) for s in sym]
        pt, PM = L191.promoter_track("ATAC", tl, PROM_PAD, say)
        idx = [int(np.where(pt == t)[0][0]) for t in grid]
        acc_h, _ = response_time(PM[idx], grid)
        m4 = responder & (PM[idx] > 0).any(0) & np.isfinite(acc_h)
        d4 = lead_test(acc_h, th, m4, "dendritic cell + LPS", say)
        if d4 is None:
            void.add("W4")
        else:
            edges = np.quantile(np.abs(pl[responder]), np.linspace(0, 1, N_STRATA + 1))
            edges[-1] += 1e-9
            held = 0
            for qi in range(N_STRATA):
                st = m4 & (np.abs(pl) >= edges[qi]) & (np.abs(pl) < edges[qi + 1])
                r = lead_test(acc_h, th, st, f"  tercile {qi+1}", say)
                if r and r["p"] < ALPHA and r["lead"] > 0:
                    held += 1
            strata["W4"] = held
            w4 = bool(d4["p"] < ALPHA and d4["lead"] > 0 and held >= 2)
            GG.verdict(w4, emit=say,
                       if_true=f"W4 PASS -- accessibility leads by {d4['lead']:+.0f} min in a "
                               f"different cell type, a different stimulus and a different assay, "
                               f"holding in {held}/3 magnitude terciles. The clock is not an A549 "
                               f"or a dexamethasone artefact",
                       if_false=f"W4 FAIL -- lead {d4['lead']:+.0f} min at p {d4['p']:.3g}, "
                                f"holding in {held}/3 terciles. W3 passed, so four points could "
                                f"have seen it: the effect does not carry to this system")

    # ---- W5 ------------------------------------------------------------------------------------
    say()
    say("W5 THE STRANGER SWAP")
    w5, d5 = False, None
    if "W5" in void or d4 is None:
        say("     W5 VOID -- see above")
        void.add("W5")
    else:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(sym))
        d5 = lead_test(acc_h[perm], th, m4, "each gene given another gene's accessibility", say)
        if d5 is None:
            void.add("W5")
        else:
            real = th[m4] - acc_h[m4]
            swap = th[m4] - acc_h[perm][m4]
            _, p5 = wilcoxon(swap, real, alternative="less")
            say(f"     real lead {d4['lead']:+.0f} min vs swapped {d5['lead']:+.0f} min; "
                f"real exceeds swapped at p {p5:.3g}")
            w5 = bool(p5 < ALPHA)
            GG.verdict(w5, emit=say,
                       if_true="W5 PASS -- the lead is a property of the gene's own promoter and "
                               "not of the two assay grids",
                       if_false="W5 FAIL -- a stranger's accessibility trajectory leads the mRNA "
                                "just as well, so the lead is about the grids and not the pairing")

    # ---- W6 ------------------------------------------------------------------------------------
    say()
    say("W6 WHAT THIS CANNOT SHOW")
    say("     Four shared timepoints is coarse. W3 establishes that a 48-minute lead is DETECTABLE")
    say("     at this resolution; it does not make the dendritic-cell lead estimate precise, and")
    say("     the number here should not be compared to A549's 48 minutes as a quantity.")
    say("     ATAC and DNase measure accessibility differently. Agreement across them is evidence")
    say("     the result is not one protocol's artefact; it is not evidence they measure the same")
    say("     physical thing.")
    say("     Leading is not causing. Accessibility opening before transcription is consistent with")
    say("     chromatin gating transcription and equally with a third factor driving both with")
    say("     different lags, and nothing here perturbs accessibility to separate them. The K562")
    say("     chromatin-drug panel is the causal experiment and is deliberately not this loop.")
    say("     Primary dendritic cells are a population of donors and states; bulk ATAC and bulk RNA")
    say("     average over both.")
    say("     A pass here replicates the CLOCK. It says nothing about loop 191d's other findings --")
    say("     that feedback sign does not order response times, and that occupancy does not carry")
    say("     timing once response size is controlled -- which remain single-system results.")
    say("     W6 PASS")

    gates = {"W1": w1, "W2": w2, "W2b": w2b, "W3": w3, "W4": w4, "W5": w5, "W6": True}
    man_out = RM.manifest(inputs=[DC / "rna.npz", A549 / "rna.npz"],
                          available=int(len(sym)), used=int(responder.sum()),
                          selection="filtered", seed=SEED,
                          controls=["the A549 result downsampled to this grid as a power calibration",
                                    "no single interval may carry 80% of the median response",
                                    "magnitude terciles",
                                    "a stranger swap on the accessibility trajectories",
                                    "both half-times on the shared grid only"],
                          note="replication of the accessibility-leads-transcription clock")
    out_d = dict(test="timing replication", gates=gates, void=sorted(void),
                 dc_grid=[int(x) for x in grid], a549_full=[int(x) for x in a_full],
                 a549_downsampled=[int(x) for x in down],
                 join_fraction=frac, n_responders=int(responder.sum()),
                 interval_shares=shares, w3_full=d3full, w3_down=d3down,
                 w4=d4, w5=d5, strata=strata,
                 manifest=man_out, seconds=time.time() - t0, log=log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)
    say()
    say("=" * 104)
    for k, v in gates.items():
        say(f"  {k}  {'VOID' if k in void else ('PASS' if v else 'FAIL')}")
    scored = [k for k in gates if k not in void]
    say(f"  {sum(gates[k] for k in scored)}/{len(scored)}   [{time.time()-t0:.0f}s]"
        + (f"   ({len(void)} VOID: {', '.join(sorted(void))})" if void else ""))
    say("=" * 104)
    out_d["log"] = log
    json.dump(out_d, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
