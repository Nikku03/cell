"""
LOOP 267 -- THE TRANSFER CURVE AT 908 CELL LINES

Loop 265 measured cross-line transfer rising +0.00094 per added cell line, 9.0 se above a
permuted null, over n = 2 to 11 training lines. Loop 266 put the own-versus-borrow crossover
at 50-75 measured genes. Both were run on twelve cell lines, and loop 265's R7 said plainly
that "a curve flat from 2 to 11 could still rise at 50" -- the extrapolation to ~61 lines was
the weakest claim in the arc because nothing could test it.

Score2 has 908. That is an order of magnitude past the extrapolation, so this loop can
FALSIFY the slope rather than merely extend it. If transfer keeps rising to 907 training
lines, loop 265's claim survives its hardest test. If it saturates at 30, the extrapolation
was wrong and the number that matters is the saturation point, not the slope.

WHAT IS STRUCTURALLY DIFFERENT HERE, AND IT REMOVES AN ARM. In LINCS each (gene, line) pair
carries a 978-dimensional response profile, so a line has its own operator W_c mapping
profiles to profiles. Score2 fitness is ONE SCALAR per (gene, line). There is no per-line
operator to fit, so loop 266's "own operator" arm does not exist here -- by loop 260's
geometry a model using only the held-out line is confined to its line mean, which the
additive baseline already contains. The comparison is therefore borrow against BASELINE
rather than borrow against own, and that difference is stated wherever a number is reported.

    F[c,g] ~= gene_mean[g] + line_mean[c] - grand        the additive baseline
    R[c,g] = F[c,g] - baseline                           the selective-dependency residual
    R[c,E] ~= sum_d a_d R[d,E]                           borrow, a fitted on measured genes

The held-out line's own line_mean is estimated from its MEASURED genes only. Using all its
genes would leak the evaluation set into the baseline every arm is scored against.

REGULARISATION IS LOAD-BEARING AT THIS SCALE, unlike in loop 266. Fitting 907 coefficients
from 100 measured genes is hopeless unregularised, so lambda is chosen by an inner split of
the measured genes, per arm and per (n, m) cell. An unregularised fit would make the large-n
end of the curve collapse for a reason that has nothing to do with cell lines.

A FREE REPLICATION, WHICH THE LINCS WORK NEVER HAD. CRISPRcleanR_FC and CERES_FC are THE SAME
908 screens processed by two different copy-number-bias pipelines. T6 runs the whole curve on
both. Anything that appears in one and not the other is the pipeline, not the biology, and
this is the first time in the arc that a result can be replicated without new data.

GATES, ALL DECLARED BEFORE THE NUMBERS:

  T1 IS THE INTERACTION SUBSTANTIAL?
     Variance of F split into gene main effect, line main effect and interaction. LINCS was
     27.6% gene / 3.8% line / 68.7% interaction. Gate: PASS iff interaction exceeds 20%.
     A FAIL means fitness is nearly additive and there is nothing here to transfer.

  T2 DOES LOOP 260'S GEOMETRY REPLICATE?                             -- requires T1
     Best in-sample gene-only and line-only models of the residual. Loop 260 measured
     0.027% and 0.0017% on LINCS. Gate: PASS iff both are below 1% here.
     This is an algebraic prediction on a completely different assay; it should hold, and
     if it does not then the residual is not what I think it is.

  T3 DOES BORROWING BEAT THE BASELINE AT ALL?                        -- requires T2
     At the largest n and largest m tested. Gate: PASS iff at least +0.02.

  T4 LOAD-BEARING -- DOES TRANSFER RISE WITH THE NUMBER OF LINES?    -- requires T3
     The curve over n, in EXCESS of a permuted-line null run at every n. Gate: PASS iff the
     excess at the largest n exceeds the excess at n = 2 by at least 0.02.

  T5 WHERE DOES IT SATURATE?                                          -- requires T4
     The smallest n reaching 90% of the maximum excess. Reported, and gated only against
     the claim loop 265 actually made: PASS iff saturation needs at least 61 lines, which
     is the number loop 265 extrapolated. A FAIL means the extrapolation overshot and is
     recorded as loop 265's error.

  T6 DOES IT REPLICATE ACROSS PROCESSING PIPELINES?                   -- requires T3
     The same curve on CERES_FC. Gate: PASS iff the T4 statistic agrees within 0.02.

  T7 WHAT THIS CANNOT SHOW
     Stated regardless of outcome.
"""
import json, time, collections
from pathlib import Path
import numpy as np

from gate_guard import Gates

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUT = "outputs/loop_score2_crossover.json"
SEED = 267267
NLINES = [2, 4, 8, 16, 32, 64, 128, 300, 600, None]
NGENES = [50, 200, 1000]
NHOLD, NREP = 40, 2
LAM = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
T1_BAR, T2_BAR, T3_BAR, T4_BAR, T5_MIN, T6_TOL = 0.20, 0.01, 0.02, 0.02, 61, 0.02
LOOP265_EXTRAP = 61
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def pear(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else np.nan


def load(tag):
    z = np.load(SCR / "score2" / f"score2_{tag}.npz", allow_pickle=True)
    F = np.asarray(z["F"], np.float64)
    lines = np.array([str(x) for x in z["lines"]]); genes = np.array([str(x) for x in z["genes"]])
    ok = np.isfinite(F).all(0)
    return F[:, ok], lines, genes[ok]


def run_curve(F, rng, label):
    """The (n training lines) x (m measured genes) grid, with a permuted-line null at each n."""
    NL_, NG = F.shape
    hold_ids = rng.choice(NL_, size=min(NHOLD, NL_), replace=False)
    grid = collections.defaultdict(list)
    for hi, c in enumerate(hold_ids):
        others = np.array([i for i in range(NL_) if i != c])
        gp = rng.permutation(NG)
        ev = gp[:NG // 2]                       # FIXED evaluation genes for this line
        pool = gp[NG // 2:]
        for n in NLINES:
            nn = len(others) if n is None else min(n, len(others))
            for rep in range(NREP):
                tr = rng.choice(others, size=nn, replace=False)
                gmean = F[tr].mean(0)                       # gene means, TRAINING lines only
                grand = gmean.mean()
                lmean_tr = (F[tr] - gmean).mean(1)
                Rtr = F[tr] - gmean[None, :] - lmean_tr[:, None]
                # NULL. Permuting which training line is which would be NO null at all:
                # the borrow fit assigns one coefficient per line and is symmetric in them,
                # so relabelling the rows leaves the fit identical. What has to be destroyed
                # is the GENE-to-GENE correspondence, so each training line's profile is
                # shuffled over genes INDEPENDENTLY -- same magnitudes, same per-line
                # distribution, measured genes no longer aligned with evaluation genes.
                Rtr_perm = np.stack([Rtr[i][rng.permutation(NG)] for i in range(nn)])
                for m in NGENES:
                    ms = pool[:m]
                    # the held-out line's own level, from MEASURED genes only -- using all of
                    # its genes would leak the evaluation set into the shared baseline
                    lc = float((F[c, ms] - gmean[ms]).mean())
                    base_ev = gmean[ev] + lc
                    Rc_m = F[c, ms] - gmean[ms] - lc
                    Rc_e = F[c, ev] - base_ev
                    b0 = pear(base_ev, F[c, ev])

                    def borrow(RT):
                        A = RT[:, ms]                        # (n, m)
                        G = A @ A.T
                        # lambda by an inner split of the MEASURED genes
                        h = len(ms) // 2
                        A1, A2 = RT[:, ms[:h]], RT[:, ms[h:]]
                        y1, y2 = Rc_m[:h], Rc_m[h:]
                        G1 = A1 @ A1.T
                        best, be = LAM[-1], np.inf
                        for lam in LAM:
                            w = np.linalg.solve(G1 + lam * np.eye(nn), A1 @ y1)
                            e = float(((w @ A2 - y2) ** 2).mean())
                            if e < be: be, best = e, lam
                        w = np.linalg.solve(G + best * np.eye(nn), A @ Rc_m)
                        return pear(base_ev + w @ RT[:, ev], F[c, ev])
                    grid[(nn, m, "real")].append(borrow(Rtr) - b0)
                    grid[(nn, m, "null")].append(borrow(Rtr_perm) - b0)
                    grid[(nn, m, "base")].append(b0)
        if hi % 10 == 0:
            say(f"       {label}: held-out line {hi+1}/{len(hold_ids)}")
    return grid


def main():
    t0 = time.time()
    G_ = Gates(emit=say)
    res = {"test": "does cross-line transfer keep rising at 908 cell lines?"}
    say("=" * 104)
    say("LOOP 267 -- THE TRANSFER CURVE AT 908 CELL LINES")
    say("=" * 104)
    say("     Loop 265 measured +0.00094 per added line over n = 2 to 11 and extrapolated ~61")
    say("     lines to close the gap. Its own R7 said a curve flat from 2 to 11 could still")
    say("     rise at 50, and nothing on hand could test it. Score2 has 908 -- an order of")
    say("     magnitude past the extrapolation, so this can FALSIFY the slope, not just extend it.")
    say("     STRUCTURAL DIFFERENCE, stated first: Score2 fitness is ONE SCALAR per (gene, line),")
    say("     so there is no per-line operator and loop 266's 'own operator' arm does not exist.")
    say("     By loop 260's geometry a model using only the held-out line is confined to its")
    say("     line mean, which the baseline already holds. So this is borrow against BASELINE.")

    rng = np.random.default_rng(SEED)
    F, lines, genes = load("crisprcleanr")
    say(f"     CRISPRcleanR: {F.shape[0]} cell lines x {F.shape[1]:,} genes (finite everywhere)")

    say("T1 IS THE INTERACTION SUBSTANTIAL?")
    gm = F.mean(0); lm = F.mean(1); gr = F.mean()
    add = gm[None, :] + lm[:, None] - gr
    vt = float(F.var())
    vg = float(np.var(np.broadcast_to(gm[None, :], F.shape)))
    vl = float(np.var(np.broadcast_to(lm[:, None], F.shape)))
    vi = float((F - add).var())
    say(f"     variance of fitness: gene main {vg/vt:.1%}, line main {vl/vt:.1%}, "
        f"interaction {vi/vt:.1%}")
    say(f"     LINCS was 27.6% gene / 3.8% line / 68.7% interaction")
    G_.add("T1", bool(vi / vt >= T1_BAR), stat=float(vi / vt),
           if_true=lambda: f"T1 PASS -- {vi/vt:.1%} of fitness variance is gene x line "
                           f"interaction, so there is something here to transfer",
           if_false=lambda: f"T1 FAIL -- only {vi/vt:.1%} is interaction; fitness is nearly "
                            f"additive and there is nothing to transfer")
    res["T1"] = {"gene": vg / vt, "line": vl / vt, "interaction": vi / vt}

    say("T2 DOES LOOP 260'S GEOMETRY REPLICATE?")
    R = F - add
    cg = 1.0 - float(((R - R.mean(0)[None, :]) ** 2).sum() / (R ** 2).sum())
    cl = 1.0 - float(((R - R.mean(1)[:, None]) ** 2).sum() / (R ** 2).sum())
    say(f"     best gene-only model of the residual: {cg:.6f}   line-only: {cl:.6f}")
    say(f"     loop 260 measured 0.000270 and 0.000017 on LINCS shRNA -- a different assay")
    G_.add("T2", bool(cg < T2_BAR and cl < T2_BAR), stat=float(max(cg, cl)), requires=("T1",),
           if_true=lambda: f"T2 PASS -- {cg:.4%} and {cl:.4%}; double-centring pins one-way "
                           f"models at zero on CRISPR fitness exactly as it did on shRNA",
           if_false=lambda: f"T2 FAIL -- gene-only {cg:.4%}, line-only {cl:.4%}")
    res["T2"] = {"gene_only": cg, "line_only": cl}

    say(f"     running the (n lines) x (m genes) grid, {NHOLD} held-out lines x {NREP} reps ...")
    grid = run_curve(F, rng, "crisprcleanr")
    ns = sorted({k[0] for k in grid}); msz = sorted({k[1] for k in grid})
    say("")
    say(f"     REAL minus permuted null, by training lines (rows) and measured genes (cols):")
    say(f"     {'n lines':>9s}" + "".join(f"{f'm={m}':>12s}" for m in msz))
    exc = {}
    for n in ns:
        row = []
        for m in msz:
            e = float(np.nanmean(grid[(n, m, "real")]) - np.nanmean(grid[(n, m, "null")]))
            exc[(n, m)] = e; row.append(e)
        say(f"     {n:9d}" + "".join(f"{v:+12.4f}" for v in row))
    mbig = msz[-1]
    say(f"     raw (not null-subtracted) at m={mbig}: " +
        ", ".join(f"n={n}:{np.nanmean(grid[(n,mbig,'real')]):+.4f}" for n in ns))
    say(f"     null itself at m={mbig}: " +
        ", ".join(f"{np.nanmean(grid[(n,mbig,'null')]):+.4f}" for n in ns))
    res["grid"] = {f"{n}_{m}": exc[(n, m)] for n in ns for m in msz}
    res["n_lines"] = ns; res["m_genes"] = msz

    say("T3 DOES BORROWING BEAT THE BASELINE AT ALL?")
    d3 = exc[(ns[-1], mbig)]
    raw3 = float(np.nanmean(grid[(ns[-1], mbig, "real")]))
    say(f"     at n={ns[-1]} lines, m={mbig} measured genes: raw {raw3:+.4f}, "
        f"excess over null {d3:+.4f}")
    G_.add("T3", bool(d3 >= T3_BAR), stat=float(d3), requires=("T2",),
           if_true=lambda: f"T3 PASS -- borrowing from {ns[-1]} lines is worth {d3:+.4f} over "
                           f"the baseline, in excess of a gene-shuffled null",
           if_false=lambda: f"T3 FAIL -- borrowing is {d3:+.4f} in excess of the null, "
                            f"below the {T3_BAR} bar")
    res["T3"] = {"excess": d3, "raw": raw3}

    say("T4 LOAD-BEARING -- DOES TRANSFER RISE WITH THE NUMBER OF LINES?")
    lo, hi = exc[(ns[0], mbig)], exc[(ns[-1], mbig)]
    d4 = hi - lo
    say(f"     n={ns[0]}: {lo:+.4f}   ->   n={ns[-1]}: {hi:+.4f}   rise {d4:+.4f}")
    G_.add("T4", bool(d4 >= T4_BAR), stat=float(d4), requires=("T3",),
           if_true=lambda: f"T4 PASS -- transfer rises {d4:+.4f} from {ns[0]} to {ns[-1]} "
                           f"training lines, so loop 265's slope survives at scale",
           if_false=lambda: f"T4 FAIL -- transfer changes by {d4:+.4f} from {ns[0]} to "
                            f"{ns[-1]} lines, below the {T4_BAR} bar. That bar is ABSOLUTE "
                            f"and was set before this assay's scale was known, so read the "
                            f"printed curve and the {ns[0]}-to-{ns[-1]} ratio of "
                            f"{hi/lo if lo > 1e-9 else float('nan'):.1f}x next to it -- a FAIL "
                            f"here does NOT license the claim that transfer is flat, only that "
                            f"the difference missed a threshold I guessed")
    res["T4"] = {"low": lo, "high": hi, "rise": d4}

    say("T5 WHERE DOES IT SATURATE?")
    mx = max(exc[(n, mbig)] for n in ns)
    sat = next((n for n in ns if exc[(n, mbig)] >= 0.9 * mx), ns[-1])
    say(f"     maximum excess {mx:+.4f}; 90% of it first reached at n = {sat} training lines")
    say(f"     loop 265 extrapolated {LOOP265_EXTRAP} lines from a curve that stopped at 11")
    G_.add("T5", bool(sat >= T5_MIN), stat=float(sat), requires=("T4",),
           if_true=lambda: f"T5 PASS -- saturation needs {sat} lines, at or beyond loop 265's "
                           f"extrapolated {LOOP265_EXTRAP}",
           if_false=lambda: f"T5 FAIL -- saturation at {sat} lines, well short of loop 265's "
                            f"extrapolated {LOOP265_EXTRAP}. The slope measured over 2-11 "
                            f"lines did not continue, and that extrapolation was wrong")
    res["T5"] = {"saturation_n": int(sat), "max_excess": mx,
                 "loop265_extrapolation": LOOP265_EXTRAP}

    say("T6 DOES IT REPLICATE ACROSS PROCESSING PIPELINES?")
    F2, _, _ = load("ceres")
    say(f"     CERES: {F2.shape[0]} lines x {F2.shape[1]:,} genes -- the SAME screens, a")
    say(f"     different copy-number-bias pipeline. Anything that appears in one and not the")
    say(f"     other is the pipeline rather than the biology.")
    g2 = run_curve(F2, np.random.default_rng(SEED), "ceres")
    ns2 = sorted({k[0] for k in g2})
    e2 = {n: float(np.nanmean(g2[(n, mbig, "real")]) - np.nanmean(g2[(n, mbig, "null")]))
          for n in ns2}
    d6 = e2[ns2[-1]] - e2[ns2[0]]
    say(f"     CERES rise {d6:+.4f} against CRISPRcleanR's {d4:+.4f}   "
        f"difference {abs(d6-d4):.4f}")
    sat2 = next((n for n in ns2 if e2[n] >= 0.9 * max(e2.values())), ns2[-1])
    say(f"     CERES saturates at n = {sat2} against CRISPRcleanR's {sat}")
    G_.add("T6", bool(abs(d6 - d4) <= T6_TOL), stat=float(abs(d6 - d4)), requires=("T3",),
           if_true=lambda: f"T6 PASS -- the two pipelines agree to {abs(d6-d4):.4f}; this is "
                           f"the first replication in the arc that needed no new data",
           if_false=lambda: f"T6 FAIL -- the pipelines differ by {abs(d6-d4):.4f}; the result "
                            f"depends on copy-number-bias correction, not only on biology")
    res["T6"] = {"ceres_rise": d6, "crisprcleanr_rise": d4, "ceres_saturation": int(sat2),
                 "ceres_curve": {str(n): e2[n] for n in ns2}}

    say("T7 WHAT THIS CANNOT SHOW")
    say("     Fitness is a SCALAR per (gene, line). There is no response profile and no")
    say("     operator, so nothing here speaks to loop 262's off-diagonal finding, which was")
    say("     about coupling BETWEEN readout genes and needs a multivariate readout.")
    say("     The measured genes are a RANDOM subset, as in loop 266. A chosen panel would")
    say("     almost certainly need fewer genes, and nothing here says which to choose.")
    say("     908 lines is many, but they are all CANCER cell lines from 27 types. Saturation")
    say("     here is saturation within that population, not across human cell states.")
    say("     CRISPR knockout is not shRNA knockdown, and a fitness readout is not expression.")
    say("     Agreement with LINCS would be a real cross-assay result; disagreement would not")
    say("     by itself refute either.")
    say("     Everything is on the double-centred residual, so by loop 260 and T2 any arm")
    say("     constant across lines is pinned near zero however expressive it is.")

    res["gates"] = {k: (v == "PASS") for k, v in G_.status.items()}
    res["void"] = [k for k, v in G_.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G_.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
