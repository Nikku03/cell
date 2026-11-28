"""Loop 247. Memory: does the cell carry a decaying history, and does it need more than one clock?

THE PROPOSAL. Give the cell a hidden memory state

    dM/dt = -Lambda M + Phi(S,G,E,t)      so      M_t = e^{-Lambda t} M_0 + int e^{-Lambda(t-tau)} Phi dtau

with several timescales, M = sum_k M^(k), each obeying dM^(k)/dt = -M^(k)/tau_k + Phi_k, standing for
fast signalling, transcriptional, chromatin and structural memory.

WHY THIS DATASET IS THE RIGHT INSTRUMENT AND SCI-PLEX WAS NOT. GSE144662 is a dexamethasone
WITHDRAWAL series in A549: the stimulus is applied, then REMOVED at t = 0, and the transcriptome is
sampled at 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10 and 12 hours in triplicate. After withdrawal the
driving term Phi is gone, so the equation makes a bare prediction with nothing else in it:

    M_t = e^{-Lambda t} M_0        -- pure relaxation, no forcing

That is a decay curve, and a decay curve is a thing one can hold out timepoints from. Loop 246
could not test M at all because sci-Plex is a single timepoint, and said so in advance.

WHAT THE DATA CAN AND CANNOT RESOLVE, MEASURED BEFORE THE GATES WERE SET. Replicate 1 against
replicate 2 at matched timepoints correlates at 0.9888 over the 2,000 most time-variable genes, so
the measurement floor is low and there is room for dynamics to show. But the profile at t = 0
already correlates with t = 12 at 0.8987 over those same genes: most of the transcriptome does not
move over this window, so every gate below runs on a screened set where across-time variation
exceeds across-replicate variation, and the screen is declared here rather than chosen later.

THE PARAMETER-COUNTING TRAP, WHICH LOOP 244 ALREADY WALKED INTO ONCE. A two-timescale model has
five free parameters per gene against three for one timescale, and more parameters always fit
better in sample. Loop 244's Hill curve had exactly this shape and LOST to a straight line once the
dose was held out. So W2 holds out TIMEPOINTS, and the second exponential has to earn its
parameters on data the fit never saw.

FOUR MODELS, all fitted per gene, all scored on a held-out timepoint.
    CONST     y = C                                    1 parameter. No memory.
    EXP1      y = C + A exp(-t/tau)                    3 parameters. One timescale.
    EXP2      y = C + A1 exp(-t/tau1) + A2 exp(-t/tau2) 5 parameters. Two timescales.
    LIN       y = C + B t                              2 parameters. The honest twin: a decay
                                                       measured over 12 hours can look linear, and
                                                       a model that only beats a CONSTANT has not
                                                       shown that anything is exponential.

PREDECLARED, BEFORE ANY NUMBER.

  W1 IS THERE A RELAXATION TO FIT AT ALL?
     EXP1 against CONST, leave-one-timepoint-out, on the screened genes.
     Gate: PASS iff EXP1 reduces held-out squared error by at least 10%. Everything requires this.

  W2 DOES MEMORY NEED MORE THAN ONE CLOCK?      -- requires W1
     EXP2 against EXP1 on held-out timepoints. This is the M = sum_k M^(k) claim.
     Gate: PASS iff EXP2 reduces held-out squared error by at least 5% relative to EXP1.

  W2b IS IT EVEN EXPONENTIAL?      -- requires W1
     EXP1 against LIN, held out. Stated separately because W1 passing against a CONSTANT would
     not distinguish an exponential from any other monotone relaxation.
     Gate: PASS iff EXP1 beats LIN by at least 5% of held-out squared error.

  W3 IS THE PROCESS NON-MARKOVIAN?      -- requires W1
     The deeper claim is that the present cell contains its history, not merely that it relaxes.
     If the state is Markov in S, then S(t-2) tells you nothing once you know S(t-1). Replicate 3
     at time t_i is predicted from replicates 1 and 2 at t_{i-1} alone, against t_{i-1} together
     with t_{i-2} and t_{i-3}. Features and target come from DIFFERENT replicates so their
     measurement noise is independent -- loop 231's L2 confound, where features and target shared
     terms by construction, is what this design exists to avoid.
     Gate: PASS iff the lagged model beats the one-step model by at least 0.02, held out by gene.

  W4 ARE THE FITTED TIMESCALES REAL OR GRID ARTEFACTS?      -- requires W1
     Loop 244's S5 found 80% of its Hill constants pinned at the edge of the search grid, which is
     a fit running out of road rather than a measured constant, and that was the mechanism behind
     its failure. The same check runs here.
     Gate: PASS iff at least half of the screened genes have tau strictly inside the grid, and the
     interquartile range of log10(tau) spans at least half a decade. A single narrow tau shared by
     every gene would mean one clock, not many.

  W5 CONTROL: TIME LABELS PERMUTED.      -- requires W1, VOID if W1's improvement is under 2%
     The same profiles with the timepoints reassigned.
     Gate: PASS iff W1's improvement collapses to under 25%.

  W6 WHAT THIS CANNOT SHOW -- written before the run.
     Twelve hours. A chromatin or structural memory with a timescale of days is not merely hard to
     fit here, it is outside the window entirely and would appear as an offset C, not as a decay.
     W4 bounds how far the fitted taus can be trusted; it cannot invent range the data lacks.
     Bulk RNA. A population relaxing coherently and a population of cells switching at different
     times give the same mean curve, and this design cannot separate them.
     One stimulus in one cell line. Dexamethasone withdrawal is a receptor-mediated response and
     nothing here says the same timescales govern other perturbations.
     Withdrawal is not the reverse of addition: removing a ligand leaves receptor, chromatin and
     transcript pools in states that the approach to steady state never passed through.
"""
import os, sys, json, time, gzip, re, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_memory_decay.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DEDEX = SCR / "dedex.npz"

SEED, NFOLD = 247247, 10
TAU_GRID = np.exp(np.linspace(np.log(0.15), np.log(60.0), 40))
SCREEN_RATIO = 2.0
W1_BAR, W2_BAR, W2B_BAR, W3_BAR, W4_FRAC, W4_IQR, W5_MAX = 0.10, 0.05, 0.05, 0.02, 0.50, 0.5, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if d.size < 3: return float("nan"), float("nan"), float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    mu = float(np.mean(d))
    return mu, se, (mu / se if se > 0 else float("nan"))


def lstsq_rows(B, Y):
    """Least squares of every gene's series on a shared design B (times x k). Y is genes x times."""
    coef, *_ = np.linalg.lstsq(B, Y.T, rcond=None)
    return coef.T, (Y - (B @ coef).T)


def fit_predict(tr_t, tr_Y, te_t, model, taus=TAU_GRID):
    """Fit on the training timepoints, predict the held-out one. Amplitudes are linear given tau,
    so tau is searched on a grid with a closed form inside -- no optimiser that can fail."""
    n = tr_Y.shape[0]
    if model == "CONST":
        c = tr_Y.mean(1)
        return np.repeat(c[:, None], len(te_t), 1), None
    if model == "LIN":
        B = np.stack([np.ones_like(tr_t), tr_t], 1)
        co, _ = lstsq_rows(B, tr_Y)
        return co @ np.stack([np.ones_like(te_t), te_t], 1).T, None
    if model == "EXP1":
        best = np.full(n, np.inf); bc = np.zeros((n, 2)); bt = np.zeros(n)
        for tau in taus:
            B = np.stack([np.ones_like(tr_t), np.exp(-tr_t / tau)], 1)
            co, r = lstsq_rows(B, tr_Y)
            ss = (r ** 2).sum(1)
            m = ss < best
            best[m], bc[m], bt[m] = ss[m], co[m], tau
        P = np.empty((n, len(te_t)))
        for i, tau in enumerate(np.unique(bt)):
            m = bt == tau
            P[m] = bc[m] @ np.stack([np.ones_like(te_t), np.exp(-te_t / tau)], 1).T
        return P, bt
    # EXP2
    best = np.full(n, np.inf); bc = np.zeros((n, 3)); bt = np.zeros((n, 2))
    sub = taus[::3]
    for i, t1 in enumerate(sub):
        for t2 in sub[i + 1:]:
            B = np.stack([np.ones_like(tr_t), np.exp(-tr_t / t1), np.exp(-tr_t / t2)], 1)
            co, r = lstsq_rows(B, tr_Y)
            ss = (r ** 2).sum(1)
            m = ss < best
            best[m], bc[m], bt[m] = ss[m], co[m], (t1, t2)
    P = np.empty((n, len(te_t)))
    for pair in np.unique(bt, axis=0):
        m = (bt == pair).all(1)
        P[m] = bc[m] @ np.stack([np.ones_like(te_t), np.exp(-te_t / pair[0]),
                                 np.exp(-te_t / pair[1])], 1).T
    return P, bt


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "memory decay on the dexamethasone withdrawal timecourse"}
    say("=" * 104)
    say("LOOP 247 -- MEMORY, ON A WITHDRAWAL TIMECOURSE WHERE THE FORCING TERM IS GONE")
    say("=" * 104)
    say("     After withdrawal Phi = 0, so the equation predicts M_t = exp(-Lambda t) M_0 with")
    say("     nothing else in it -- a bare decay curve, and a curve one can hold timepoints out of.")
    say("     A two-timescale model has 5 parameters per gene against 3. Loop 244's Hill curve had")
    say("     the same shape and LOST to a straight line once dose was held out, so W2 holds out")
    say("     timepoints and W2b asks separately whether the relaxation is exponential at all.")

    z = np.load(DEDEX, allow_pickle=True)
    V, genes, times, reps = z["V"], z["genes"], z["times"], z["reps"]
    T = np.array(sorted(set(times)))
    say(f"     {V.shape[0]:,} genes, {len(T)} timepoints {list(T)} h, "
        f"{len(set(reps))} replicates")

    # per-timepoint mean, and the two variance components the screen compares
    Ymean = np.stack([V[:, times == t].mean(1) for t in T], 1)
    v_time = Ymean.var(1, ddof=1)
    v_rep = np.mean([V[:, times == t].var(1, ddof=1) for t in T], 0)
    keep = np.where(v_time > SCREEN_RATIO * np.maximum(v_rep, 1e-12))[0]
    say(f"     screen declared in advance: across-time variance > {SCREEN_RATIO}x across-replicate")
    say(f"     variance -> {len(keep):,} genes of {V.shape[0]:,}")
    Y = Ymean[keep]
    res["n_screened"] = int(len(keep))

    # ---------------------------------------------------------------- leave-one-timepoint-out
    say("     leave-one-timepoint-out, all four models ...")
    SSE = {m: np.zeros(len(keep)) for m in ("CONST", "LIN", "EXP1", "EXP2")}
    taus_last = None
    for hi in range(len(T)):
        tr = np.array([j for j in range(len(T)) if j != hi])
        for m in SSE:
            P, tt = fit_predict(T[tr], Y[:, tr], T[hi:hi + 1], m)
            SSE[m] += (P[:, 0] - Y[:, hi]) ** 2
            if m == "EXP1" and hi == 0: taus_last = tt
    tot = {m: float(SSE[m].sum()) for m in SSE}
    for m in ("CONST", "LIN", "EXP1", "EXP2"):
        say(f"       {m:<6} held-out sum of squares {tot[m]:.4f}   "
            f"({1 - tot[m] / tot['CONST']:+.1%} vs CONST)")
    res["sse"] = tot

    # ---------------------------------------------------------------- W1
    say("W1 IS THERE A RELAXATION TO FIT AT ALL?")
    imp1 = 1 - tot["EXP1"] / tot["CONST"]
    say(f"     EXP1 reduces held-out squared error by {imp1:.1%} against a constant")
    G.add("W1", bool(imp1 >= W1_BAR), stat=float(imp1),
          if_true=lambda: f"W1 PASS -- there is a relaxation: {imp1:.1%} of held-out error removed",
          if_false=lambda: f"W1 FAIL -- {imp1:.1%} against a {W1_BAR:.0%} bar; nothing is relaxing "
                           f"on this timescale")
    res["W1"] = {"improvement": imp1}

    # ---------------------------------------------------------------- W2
    say("W2 DOES MEMORY NEED MORE THAN ONE CLOCK?")
    imp2 = 1 - tot["EXP2"] / tot["EXP1"]
    d2, se2, z2 = paired(SSE["EXP1"], SSE["EXP2"])
    say(f"     EXP2 against EXP1 on held-out timepoints: {imp2:+.1%} of squared error")
    say(f"     per-gene paired difference EXP1 minus EXP2: {d2:+.6f} +/- {se2:.6f} ({z2:+.1f} se)")
    G.add("W2", bool(imp2 >= W2_BAR), stat=float(imp2), requires=("W1",),
          if_true=lambda: f"W2 PASS -- a second timescale earns its parameters: {imp2:+.1%}",
          if_false=lambda: f"W2 FAIL -- a second timescale buys {imp2:+.1%} on held-out "
                           f"timepoints, against a {W2_BAR:.0%} bar; five parameters per gene do "
                           f"not beat three")
    res["W2"] = {"improvement": imp2, "paired": d2, "se": se2, "z": z2}

    # ---------------------------------------------------------------- W2b
    say("W2b IS IT EVEN EXPONENTIAL?")
    impb = 1 - tot["EXP1"] / tot["LIN"]
    say(f"     EXP1 against a straight line in time: {impb:+.1%} of held-out squared error")
    G.add("W2b", bool(impb >= W2B_BAR), stat=float(impb), requires=("W1",),
          if_true=lambda: f"W2b PASS -- the relaxation is exponential rather than merely monotone: "
                          f"{impb:+.1%} over a straight line",
          if_false=lambda: f"W2b FAIL -- EXP1 beats a straight line by {impb:+.1%}; over 12 hours "
                           f"the decay is not distinguishable from linear")
    res["W2b"] = {"improvement": impb}

    # ---------------------------------------------------------------- W3
    say("W3 IS THE PROCESS NON-MARKOVIAN?")
    say("     replicate 3 at t_i predicted from replicates 1 and 2 at t_{i-1} alone, against")
    say("     t_{i-1} with t_{i-2} and t_{i-3}. Different replicates, so the measurement noise in")
    say("     the features is independent of the noise in the target -- loop 231's L2 confound.")
    A = np.stack([V[np.ix_(keep, (times == t) & (reps != 3))].mean(1) for t in T], 1)
    B = np.stack([V[np.ix_(keep, (times == t) & (reps == 3))].mean(1) for t in T], 1)
    rows_m, rows_l, ys = [], [], []
    for i in range(3, len(T)):
        rows_m.append(np.stack([A[:, i - 1]], 1))
        rows_l.append(np.stack([A[:, i - 1], A[:, i - 2], A[:, i - 3]], 1))
        ys.append(B[:, i])
    Xm = np.concatenate(rows_m, 0); Xl = np.concatenate(rows_l, 0); yv = np.concatenate(ys)
    gid = np.tile(np.arange(len(keep)), len(T) - 3)
    ug = np.arange(len(keep)); rng.shuffle(ug)
    fold = {g: i % NFOLD for i, g in enumerate(ug)}
    fo = np.array([fold[g] for g in gid])

    def cv(F):
        p = np.full(len(yv), np.nan)
        for k in range(NFOLD):
            tr, te = fo != k, fo == k
            Z = np.concatenate([F[tr], np.ones((tr.sum(), 1))], 1)
            mu_, sd_ = Z[:, :-1].mean(0), Z[:, :-1].std(0) + 1e-9
            Zs = np.concatenate([(F[tr] - mu_) / sd_, np.ones((tr.sum(), 1))], 1)
            M = Zs.T @ Zs + 1e-3 * tr.sum() * np.eye(Zs.shape[1])
            b = np.linalg.solve(M, Zs.T @ yv[tr])
            p[te] = np.concatenate([(F[te] - mu_) / sd_, np.ones((te.sum(), 1))], 1) @ b
        return np.array([pear(p[fo == k], yv[fo == k]) for k in range(NFOLD)])
    sm, sl = cv(Xm), cv(Xl)
    d3, se3, z3 = paired(sl, sm)
    say(f"     one step back only: {np.nanmean(sm):.4f}   with three lags: {np.nanmean(sl):.4f}")
    say(f"     paired over {NFOLD} gene folds: {d3:+.4f} +/- {se3:.4f}  ({z3:+.1f} se)")
    G.add("W3", bool(d3 >= W3_BAR), stat=float(d3), requires=("W1",),
          if_true=lambda: f"W3 PASS -- the past adds {d3:+.4f} beyond the present; the process is "
                          f"not Markov in S",
          if_false=lambda: f"W3 FAIL -- knowing t-2 and t-3 adds {d3:+.4f} once t-1 is known, "
                           f"against a {W3_BAR} bar; S(t-1) is a sufficient statistic here")
    res["W3"] = {"markov": float(np.nanmean(sm)), "lagged": float(np.nanmean(sl)),
                 "delta": d3, "se": se3, "z": z3}

    # ---------------------------------------------------------------- W4
    say("W4 ARE THE FITTED TIMESCALES REAL OR GRID ARTEFACTS?")
    _, tau_all = fit_predict(T, Y, T[:1], "EXP1")
    inside = (tau_all > TAU_GRID[0] * 1.001) & (tau_all < TAU_GRID[-1] * 0.999)
    frac = float(inside.mean())
    lt = np.log10(tau_all[inside])
    iqr = float(np.percentile(lt, 75) - np.percentile(lt, 25)) if inside.sum() > 10 else float("nan")
    say(f"     tau strictly inside the {TAU_GRID[0]:.2f}-{TAU_GRID[-1]:.0f} h grid: {frac:.1%}")
    if inside.sum() > 10:
        say(f"     median tau {10 ** np.median(lt):.2f} h, "
            f"interquartile range of log10(tau) {iqr:.2f} decades "
            f"({10 ** np.percentile(lt, 25):.2f} - {10 ** np.percentile(lt, 75):.2f} h)")
    G.add("W4", bool(frac >= W4_FRAC and np.isfinite(iqr) and iqr >= W4_IQR), stat=float(frac),
          requires=("W1",),
          if_true=lambda: f"W4 PASS -- {frac:.0%} of taus are off the grid edge and they span "
                          f"{iqr:.2f} decades",
          if_false=lambda: f"W4 FAIL -- {frac:.0%} inside the grid, log10(tau) spread {iqr:.2f} "
                           f"decades; the timescales are pinned or all the same")
    res["W4"] = {"fraction_inside": frac, "iqr_decades": iqr,
                 "median_tau_h": float(10 ** np.median(lt)) if inside.sum() > 10 else None}

    # ---------------------------------------------------------------- W5
    say("W5 CONTROL: TIME LABELS PERMUTED")
    if imp1 < 0.02:
        G.add("W5", False, stat=float(imp1), requires=("W1",), void_if=True,
              void_reason=f"W1's improvement is {imp1:.1%}; there is nothing to collapse")
    else:
        Tp = T[rng.permutation(len(T))]
        s_const = s_exp = 0.0
        for hi in range(len(T)):
            tr = np.array([j for j in range(len(T)) if j != hi])
            Pc, _ = fit_predict(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "CONST")
            Pe, _ = fit_predict(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "EXP1")
            s_const += float(((Pc[:, 0] - Y[:, hi]) ** 2).sum())
            s_exp += float(((Pe[:, 0] - Y[:, hi]) ** 2).sum())
        imp_s = 1 - s_exp / s_const
        f5 = imp_s / imp1
        say(f"     timepoints reassigned: EXP1 improves {imp_s:+.1%} against a real {imp1:+.1%} "
            f"({f5:.0%})")
        G.add("W5", bool(f5 <= W5_MAX), stat=float(f5), requires=("W1",),
              if_true=lambda: f"W5 PASS -- collapses to {f5:.0%} with the clock scrambled",
              if_false=lambda: f"W5 FAIL -- {f5:.0%} survives permuting time; the fit is not using "
                               f"the time axis")
        res["W5"] = {"real": imp1, "shuffled": imp_s, "fraction": f5}

    say("W6 WHAT THIS CANNOT SHOW")
    say("     Twelve hours. A chromatin or structural memory with a timescale of days is outside")
    say("     the window entirely and would appear as the offset C, not as a decay.")
    say("     Bulk RNA. A population relaxing coherently and a population switching at different")
    say("     times give the same mean curve; this design cannot separate them.")
    say("     One stimulus, one cell line. Nothing here says other perturbations share these")
    say("     timescales.")
    say("     Withdrawal is not the reverse of addition: removing a ligand leaves receptor,")
    say("     chromatin and transcript pools in states the approach to steady state never passed")
    say("     through.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary(seconds=res["seconds"])
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
