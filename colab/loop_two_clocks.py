"""Loop 248. The multi-timescale claim, with the identifiability defect that voided it repaired.

WHAT WENT WRONG IN LOOP 247, MEASURED RATHER THAN GUESSED. Loop 247's EXP2 arm returned a
held-out sum of squares of 461,317 against the constant model's 1,899 -- 243 times worse -- and
its W2 gate recorded that as "five parameters per gene do not beat three". That reading was not
available from the number. The first hypothesis was collinearity between two nearby exponentials;
it was WRONG, and the basis condition numbers say so: 26 to 60. The measured cause:

    held-out fold t = 0     SSE 460,623        99.85% of the total
    every other fold        SSE  under 500
    exp(-t/0.15h) over the twelve timepoints = 1.0, 0.036, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0

With tau as short as 0.15 h that basis column is essentially zero at EVERY TRAINING timepoint once
t = 0 is held out. Its amplitude is then unconstrained -- any value fits the training data equally
well -- least squares returned amplitudes up to 337, and the prediction exploded at the single
point where the basis is not zero. So loop 247 did not measure whether two timescales help. It
measured that a timescale shorter than the sampling grid can resolve cannot be fitted by an
unconstrained least squares.

THE REPAIR, AND WHY IT IS A DEFINITION RATHER THAN A PATCH. An amplitude is identifiable only if
its basis column actually varies across the data used to fit it. So a (tau) is admitted to a fold
only if, over that fold's TRAINING timepoints,

    max_j exp(-t_j/tau) - min_j exp(-t_j/tau)  >=  RANGE_MIN

which excludes the too-fast end (a spike at t=0, flat elsewhere) and the too-slow end (flat
everywhere, indistinguishable from the constant) by the same criterion and without naming either.
For two components the pair must additionally be distinguishable from each other: the correlation
between the two basis columns over the training timepoints must be below 0.95. Amplitudes are
ridge-regularised. Every model is refitted under the same rule so EXP1 and EXP2 are compared on
equal footing rather than one of them being handicapped.

The admitted band is fold-dependent, and that is correct rather than awkward: when t = 0 is held
out, a component only visible at t = 0 is genuinely unmeasurable, and a model that claims to
predict it is claiming something the data cannot support.

PREDECLARED, BEFORE ANY NUMBER.

  X1 DOES THE REPAIR ACTUALLY REPAIR IT?
     A self-check on the fix before any scientific claim rests on it.
     Gate: PASS iff EXP2's total held-out error is now BELOW the constant model's, and the t = 0
     fold contributes under 40% of it. Everything requires this. A FAIL means the repair failed
     and loop 247's W2 stays untested rather than being replaced by another broken number.

  X2 DOES A SECOND TIMESCALE EARN ITS PARAMETERS?      -- requires X1
     EXP2 against EXP1, both under the identifiability rule, leave-one-timepoint-out.
     Gate: PASS iff EXP2 reduces held-out squared error by at least 5%.

  X3 IS THE RELAXATION EXPONENTIAL AT ALL?      -- requires X1
     Loop 247's W2b found a straight line beating the exponential by 31% of held-out error, but
     that EXP1 was fitted without the identifiability rule. Both exponentials are re-compared
     against the line here.
     Gate: PASS iff the better exponential beats LIN by at least 5% of held-out squared error.

  X4 IF TWO CLOCKS ARE NEEDED, ARE THEY DISTINCT?      -- requires X2
     Among the genes where EXP2 actually wins on held-out data, the ratio of the two fitted taus.
     Gate: PASS iff the median tau2/tau1 is at least 3. A pair of near-equal taus is one clock
     fitted twice, whatever the error says.

  X5 HOW MANY GENES NEED TWO CLOCKS?      -- requires X1
     The fraction of screened genes whose held-out error is lower under EXP2 than EXP1.
     Gate: PASS iff at least 20%. A second timescale that helps 3% of genes is a statement about
     three percent of genes, and reporting it as "memory has multiple timescales" would be the
     kind of overstatement this project exists to avoid.

  X6 CONTROL: TIME LABELS PERMUTED.      -- requires X1
     Gate: PASS iff EXP2's improvement over the constant collapses to under 25%.

  X7 WHAT THIS CANNOT SHOW -- written before the run.
     The identifiability rule bounds what can be fitted; it does not create resolution. A genuine
     sub-30-minute component exists or does not exist independently of whether this sampling grid
     can see it, and this loop cannot decide that question. It can only stop the model from
     claiming to have decided it.
     Twelve hours still caps the slow end. A day-scale chromatin memory appears as the offset C.
     Bulk RNA still cannot distinguish a coherently relaxing population from cells switching at
     different times.
     Two exponentials fitting better than one does not establish that the cell contains two
     physical memory pools. A stretched exponential, a power law, or a distribution of rates
     across genes would all produce the same improvement.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_two_clocks.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DEDEX = SCR / "dedex.npz"

SEED = 248248
TAU_GRID = np.exp(np.linspace(np.log(0.15), np.log(120.0), 48))
RANGE_MIN, MAX_BASIS_CORR, RIDGE = 0.10, 0.95, 1e-4
SCREEN_RATIO = 2.0
X1_T0_MAX, X2_BAR, X3_BAR, X4_RATIO, X5_FRAC, X6_MAX = 0.40, 0.05, 0.05, 3.0, 0.20, 0.25

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def admissible(taus, tr_t):
    """A tau is admitted only if its basis column VARIES across the training timepoints.
    Excludes the too-fast end (a spike at one point, flat elsewhere) and the too-slow end (flat
    everywhere, indistinguishable from the intercept) by one criterion, naming neither."""
    out = []
    for tau in taus:
        b = np.exp(-tr_t / tau)
        if b.max() - b.min() >= RANGE_MIN: out.append(tau)
    return np.array(out)


def solve(B, Y):
    """Ridge-regularised amplitudes for every gene on a shared design, and the residuals."""
    A = B.T @ B + RIDGE * np.trace(B.T @ B) / B.shape[1] * np.eye(B.shape[1])
    co = np.linalg.solve(A, B.T @ Y.T).T
    return co, Y - co @ B.T


def fit_fold(tr_t, tr_Y, te_t, model):
    n = tr_Y.shape[0]
    ones = np.ones_like(tr_t)
    if model == "CONST":
        return np.repeat(tr_Y.mean(1)[:, None], len(te_t), 1), None
    if model == "LIN":
        co, _ = solve(np.stack([ones, tr_t], 1), tr_Y)
        return co @ np.stack([np.ones_like(te_t), te_t], 1).T, None
    adm = admissible(TAU_GRID, tr_t)
    if len(adm) == 0:
        return np.repeat(tr_Y.mean(1)[:, None], len(te_t), 1), None
    if model == "EXP1":
        best = np.full(n, np.inf); bc = np.zeros((n, 2)); bt = np.zeros(n)
        for tau in adm:
            co, r = solve(np.stack([ones, np.exp(-tr_t / tau)], 1), tr_Y)
            ss = (r ** 2).sum(1); m = ss < best
            best[m], bc[m], bt[m] = ss[m], co[m], tau
        P = np.empty((n, len(te_t)))
        for tau in np.unique(bt):
            m = bt == tau
            P[m] = bc[m] @ np.stack([np.ones_like(te_t), np.exp(-te_t / tau)], 1).T
        return P, bt
    best = np.full(n, np.inf); bc = np.zeros((n, 3)); bt = np.zeros((n, 2))
    for i, t1 in enumerate(adm):
        b1 = np.exp(-tr_t / t1)
        for t2 in adm[i + 1:]:
            b2 = np.exp(-tr_t / t2)
            c = np.corrcoef(b1, b2)[0, 1]
            if not np.isfinite(c) or abs(c) >= MAX_BASIS_CORR: continue
            co, r = solve(np.stack([ones, b1, b2], 1), tr_Y)
            ss = (r ** 2).sum(1); m = ss < best
            best[m], bc[m], bt[m] = ss[m], co[m], (t1, t2)
    if not np.isfinite(best).any() or (bt == 0).all():
        return np.repeat(tr_Y.mean(1)[:, None], len(te_t), 1), None
    P = np.empty((n, len(te_t)))
    for pair in np.unique(bt, axis=0):
        m = (bt == pair).all(1)
        if pair[0] == 0:
            P[m] = tr_Y[m].mean(1)[:, None]
        else:
            P[m] = bc[m] @ np.stack([np.ones_like(te_t), np.exp(-te_t / pair[0]),
                                     np.exp(-te_t / pair[1])], 1).T
    return P, bt


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "two-timescale memory with an identifiability constraint"}
    say("=" * 104)
    say("LOOP 248 -- TWO CLOCKS, WITH THE IDENTIFIABILITY DEFECT REPAIRED")
    say("=" * 104)
    say("     Loop 247's EXP2 scored 461,317 against a constant model's 1,899. The cause, measured:")
    say("     the t=0 fold alone contributed 460,623 of it -- 99.85%. With tau = 0.15 h the basis")
    say("     is 1.0 at t=0 and under 0.04 everywhere else, so with t=0 held out its amplitude was")
    say("     unconstrained, least squares returned values up to 337, and the prediction exploded")
    say("     at the one point where the basis is not zero. Loop 247 did not test two timescales.")
    say(f"     The rule: a tau is admitted to a fold only if its basis RANGE over that fold's")
    say(f"     training timepoints is at least {RANGE_MIN}, and a pair must have basis correlation")
    say(f"     below {MAX_BASIS_CORR}. Every model is refitted under it, so nothing is handicapped.")

    z = np.load(DEDEX, allow_pickle=True)
    V, times, reps = z["V"], z["times"], z["reps"]
    T = np.array(sorted(set(times)))
    Ymean = np.stack([V[:, times == t].mean(1) for t in T], 1)
    v_time = Ymean.var(1, ddof=1)
    v_rep = np.mean([V[:, times == t].var(1, ddof=1) for t in T], 0)
    keep = np.where(v_time > SCREEN_RATIO * np.maximum(v_rep, 1e-12))[0]
    Y = Ymean[keep]
    say(f"     {len(keep):,} screened genes, {len(T)} timepoints {list(T)} h")

    for hi in (0, 6):
        tr = np.array([j for j in range(len(T)) if j != hi])
        adm = admissible(TAU_GRID, T[tr])
        say(f"     with t={T[hi]:g}h held out, admissible taus: {len(adm)} of {len(TAU_GRID)} "
            f"({adm.min():.2f}-{adm.max():.1f} h)" if len(adm) else
            f"     with t={T[hi]:g}h held out: NO admissible tau")

    say("     leave-one-timepoint-out, all four models under the same rule ...")
    SSE = {m: np.zeros(len(keep)) for m in ("CONST", "LIN", "EXP1", "EXP2")}
    per_fold = {m: [] for m in SSE}
    TAU2 = None
    for hi in range(len(T)):
        tr = np.array([j for j in range(len(T)) if j != hi])
        for m in SSE:
            P, tt = fit_fold(T[tr], Y[:, tr], T[hi:hi + 1], m)
            e = (P[:, 0] - Y[:, hi]) ** 2
            SSE[m] += e; per_fold[m].append(float(e.sum()))
            if m == "EXP2" and hi == 6: TAU2 = tt
    tot = {m: float(SSE[m].sum()) for m in SSE}
    say(f"     {'model':<7}{'held-out SSE':>16}{'vs CONST':>12}")
    for m in ("CONST", "LIN", "EXP1", "EXP2"):
        say(f"     {m:<7}{tot[m]:>16.2f}{1 - tot[m] / tot['CONST']:>+12.1%}")
    res["sse"] = tot
    res["per_fold"] = {m: per_fold[m] for m in per_fold}

    # ---------------------------------------------------------------- X1
    say("X1 DOES THE REPAIR ACTUALLY REPAIR IT?")
    t0share = per_fold["EXP2"][0] / max(tot["EXP2"], 1e-12)
    say(f"     EXP2 held-out SSE {tot['EXP2']:.2f} against the constant model's {tot['CONST']:.2f}")
    say(f"     the t=0 fold now contributes {t0share:.1%} of EXP2's total "
        f"(it was 99.85% in loop 247)")
    G.add("X1", bool(tot["EXP2"] < tot["CONST"] and t0share <= X1_T0_MAX), stat=float(t0share),
          if_true=lambda: f"X1 PASS -- EXP2 is now below the constant model and the t=0 fold is "
                          f"{t0share:.0%} of its error; the fix holds",
          if_false=lambda: f"X1 FAIL -- EXP2 is {tot['EXP2']:.1f} against CONST {tot['CONST']:.1f} "
                           f"with {t0share:.0%} from t=0; the repair did not work and loop 247's "
                           f"W2 stays untested rather than being replaced by another broken number")
    res["X1"] = {"exp2": tot["EXP2"], "const": tot["CONST"], "t0_share": t0share}

    # ---------------------------------------------------------------- X2
    say("X2 DOES A SECOND TIMESCALE EARN ITS PARAMETERS?")
    imp = 1 - tot["EXP2"] / tot["EXP1"]
    say(f"     EXP2 against EXP1, both identifiability-constrained: {imp:+.1%} of held-out error")
    G.add("X2", bool(imp >= X2_BAR), stat=float(imp), requires=("X1",),
          if_true=lambda: f"X2 PASS -- a second clock earns its parameters: {imp:+.1%}",
          if_false=lambda: f"X2 FAIL -- a second clock buys {imp:+.1%} on held-out timepoints "
                           f"against a {X2_BAR:.0%} bar")
    res["X2"] = {"improvement": imp}

    # ---------------------------------------------------------------- X3
    say("X3 IS THE RELAXATION EXPONENTIAL AT ALL?")
    bestexp = "EXP1" if tot["EXP1"] <= tot["EXP2"] else "EXP2"
    impl = 1 - tot[bestexp] / tot["LIN"]
    say(f"     best exponential is {bestexp} ({tot[bestexp]:.2f}) against LIN ({tot['LIN']:.2f}): "
        f"{impl:+.1%}")
    say(f"     loop 247 found -31.0% here with an UNCONSTRAINED EXP1")
    G.add("X3", bool(impl >= X3_BAR), stat=float(impl), requires=("X1",),
          if_true=lambda: f"X3 PASS -- the relaxation is exponential rather than merely monotone: "
                          f"{impl:+.1%} over a straight line",
          if_false=lambda: f"X3 FAIL -- the best exponential beats a straight line by {impl:+.1%}; "
                           f"over 12 hours the decay is not distinguishable from linear")
    res["X3"] = {"best_exp": bestexp, "improvement": impl}

    # ---------------------------------------------------------------- X5 (before X4, which needs it)
    say("X5 HOW MANY GENES NEED TWO CLOCKS?")
    wins = SSE["EXP2"] < SSE["EXP1"]
    frac = float(wins.mean())
    say(f"     EXP2 has lower held-out error than EXP1 for {int(wins.sum()):,} of {len(keep):,} "
        f"genes ({frac:.1%})")
    G.add("X5", bool(frac >= X5_FRAC), stat=float(frac), requires=("X1",),
          if_true=lambda: f"X5 PASS -- {frac:.0%} of genes are better fitted with two clocks",
          if_false=lambda: f"X5 FAIL -- only {frac:.0%} of genes benefit, against a {X5_FRAC:.0%} "
                           f"bar; a second timescale that helps this few genes is a statement "
                           f"about those genes, not about memory")
    res["X5"] = {"fraction": frac, "n_win": int(wins.sum()), "n": int(len(keep))}

    # ---------------------------------------------------------------- X4
    say("X4 IF TWO CLOCKS ARE NEEDED, ARE THEY DISTINCT?")
    if TAU2 is None or not np.isfinite(TAU2).any():
        G.add("X4", False, stat=float("nan"), requires=("X2",), void_if=True,
              void_reason="no EXP2 fit was admitted in the reference fold")
    else:
        ok = wins & (TAU2[:, 0] > 0)
        if ok.sum() < 20:
            G.add("X4", False, stat=float(ok.sum()), requires=("X2",), void_if=True,
                  void_reason=f"only {int(ok.sum())} genes have both an EXP2 win and a fitted pair")
        else:
            ratio = TAU2[ok, 1] / np.maximum(TAU2[ok, 0], 1e-9)
            med = float(np.median(ratio))
            say(f"     among the {int(ok.sum()):,} genes where EXP2 wins: median tau1 "
                f"{np.median(TAU2[ok, 0]):.2f} h, median tau2 {np.median(TAU2[ok, 1]):.2f} h, "
                f"median ratio {med:.2f}")
            G.add("X4", bool(med >= X4_RATIO), stat=float(med), requires=("X2",),
                  if_true=lambda: f"X4 PASS -- the two clocks are {med:.1f}x apart",
                  if_false=lambda: f"X4 FAIL -- median tau ratio {med:.2f}, under {X4_RATIO}; two "
                                   f"near-equal taus are one clock fitted twice")
            res["X4"] = {"median_ratio": med, "median_tau1": float(np.median(TAU2[ok, 0])),
                         "median_tau2": float(np.median(TAU2[ok, 1])), "n": int(ok.sum())}

    # ---------------------------------------------------------------- X6
    say("X6 CONTROL: TIME LABELS PERMUTED")
    imp_real = 1 - tot["EXP2"] / tot["CONST"]
    if imp_real < 0.02:
        G.add("X6", False, stat=float(imp_real), requires=("X1",), void_if=True,
              void_reason=f"EXP2's improvement over the constant is {imp_real:.1%}; nothing to "
                          f"collapse")
    else:
        Tp = T[rng.permutation(len(T))]
        sc = se = 0.0
        for hi in range(len(T)):
            tr = np.array([j for j in range(len(T)) if j != hi])
            Pc, _ = fit_fold(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "CONST")
            Pe, _ = fit_fold(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "EXP2")
            sc += float(((Pc[:, 0] - Y[:, hi]) ** 2).sum())
            se += float(((Pe[:, 0] - Y[:, hi]) ** 2).sum())
        imp_s = 1 - se / sc
        f6 = imp_s / imp_real
        say(f"     timepoints reassigned: EXP2 improves {imp_s:+.1%} against a real "
            f"{imp_real:+.1%}  ({f6:.0%})")
        G.add("X6", bool(f6 <= X6_MAX), stat=float(f6), requires=("X1",),
              if_true=lambda: f"X6 PASS -- collapses to {f6:.0%} with the clock scrambled",
              if_false=lambda: f"X6 FAIL -- {f6:.0%} survives permuting time")
        res["X6"] = {"real": imp_real, "shuffled": imp_s, "fraction": f6}

    say("X7 WHAT THIS CANNOT SHOW")
    say("     The identifiability rule bounds what can be FITTED; it does not create resolution.")
    say("     A genuine sub-30-minute component exists or does not independently of whether this")
    say("     grid can see it. This loop only stops the model claiming to have decided that.")
    say("     Twelve hours caps the slow end; a day-scale memory appears as the offset C.")
    say("     Bulk RNA cannot separate a coherently relaxing population from cells switching at")
    say("     different times.")
    say("     Two exponentials fitting better than one does NOT establish two physical memory")
    say("     pools. A stretched exponential, a power law, or a distribution of rates across")
    say("     genes would produce the same improvement.")

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
