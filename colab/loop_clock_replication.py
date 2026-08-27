"""Loop 249. The two-clock result, separated from the boundary fold and tested where it was blind.

WHERE THIS STANDS, STATED PLAINLY BECAUSE IT AFFECTS HOW THE NUMBERS BELOW MUST BE READ.

Loop 247 tested whether memory needs more than one timescale and its EXP2 arm returned nonsense
(461,317 held-out error against a constant model's 1,899). Loop 248 diagnosed the cause -- a basis
column essentially zero at every training timepoint has an unidentifiable amplitude -- applied an
identifiability rule, cut the error 114-fold to 4,057, and STILL failed its own X1 self-check
because 93% of what remained was one fold: t = 0.

The reason is structural and not a tuning problem. Every decaying basis exp(-t/tau) attains its
MAXIMUM at t = 0. Holding out t = 0 therefore asks every decay model to extrapolate its basis
beyond the range it saw while fitting, and no threshold on training-range prevents that. t = 0 is
the boundary of the time domain, and for a decay model it is an extrapolation fold by construction.
Loop 244 met the same distinction on the dose axis and reported interior and extrapolation
separately, as S2 and S4.

AND THEN, AFTER LOOP 248's X1 FAILED, THE INTERIOR FOLDS WERE COMPUTED DIAGNOSTICALLY:

    interior only, ten folds        CONST 1318.13   LIN 457.83   EXP1 326.11   EXP2 258.91
                                    EXP2 over EXP1  +20.6%       EXP1 over LIN  +28.8%

Those numbers were seen BEFORE this file was written. They are therefore reported here as a
diagnostic, not as a blind test, and Y2 is labelled post-hoc in its own gate text. Presenting them
as a fresh predeclared result would be exactly the move this project's ledger exists to prevent.

WHAT IS ACTUALLY BLIND HERE. Three things that could not be read off the numbers above:

  Y3  the permuted-time control restricted to interior folds -- never computed.
  Y4  a GENE holdout: the two taus chosen on one half of the genes and applied, frozen, to the
      other half. If two clocks are a property of the relaxation rather than of per-gene curve
      fitting, taus fitted on other genes should still help.
  Y5  replication in an INDEPENDENT experiment running the opposite direction. GSE144662 is
      dexamethasone WITHDRAWAL; the matched OE_ctrl arm is dexamethasone ADDITION in the same
      cell line, sampled at 0, 1, 4, 8 and 12 h. Five timepoints cannot fit five parameters, so
      the taus are FROZEN from the withdrawal fit and only amplitudes are estimated. A two-clock
      structure that is real should transfer; one that is curve-fitting should not.

PREDECLARED, BEFORE ANY NUMBER THAT IS NOT ALREADY QUOTED ABOVE.

  Y1 IS THE BOUNDARY DIAGNOSIS CORRECT?
     Confirmatory, and cheap: for every admissible tau, compare the basis value at t = 0 against
     the largest value it reaches over the other eleven timepoints.
     Gate: PASS iff every admissible tau has basis(0) strictly greater than its maximum over the
     remaining timepoints -- that is, t = 0 is an extrapolation fold for EVERY decay model, not
     merely for the fast ones. Everything requires this, because if it is false then dropping
     t = 0 is a convenience rather than a necessity.

  Y2 THE INTERIOR COMPARISON -- POST-HOC, LABELLED AS SUCH.      -- requires Y1
     EXP2 against EXP1 on the ten interior folds.
     Gate: PASS iff EXP2 reduces held-out squared error by at least 5%. This gate cannot be
     evidence on its own, because its number was already known when the gate was written. It is
     here so the value enters the record with its status attached.

  Y3 DOES THE INTERIOR RESULT SURVIVE SCRAMBLING THE CLOCK?      -- requires Y1, blind
     Timepoints permuted, interior folds only.
     Gate: PASS iff EXP2's improvement over the constant collapses to under 25%.

  Y4 DO THE TAUS TRANSFER TO GENES THEY WERE NOT FITTED ON?      -- requires Y1, blind
     The best (tau1, tau2) pair is chosen on a random half of the genes by total training error,
     then FROZEN and applied to the held-out half, where only amplitudes are fitted. Compared
     against the same procedure with one frozen tau.
     Gate: PASS iff the frozen two-tau model beats the frozen one-tau model by at least 5% of
     held-out squared error on genes that had no say in choosing the taus.

  Y5 DOES IT REPLICATE IN THE OPPOSITE DIRECTION?      -- requires Y1, blind
     Taus frozen from withdrawal, applied to dexamethasone ADDITION (OE_ctrl, 0/1/4/8/12 h),
     amplitudes free, leave-one-timepoint-out over interior timepoints.
     Gate: PASS iff the frozen two-tau model beats the frozen one-tau model by at least 5% there.
     A FAIL means the two-clock structure is specific to relaxation, or to this experiment, and
     the distinction matters more than the pooled number.

  Y6 WHAT THIS CANNOT SHOW -- written before the run.
     Dropping t = 0 removes the only fold that could reveal a sub-30-minute component, so this
     loop is structurally blind to the fastest clock. That is a limit of a grid whose second
     sample is at 30 minutes, and no analysis recovers it.
     Two exponentials beating one does not establish two physical memory pools. A stretched
     exponential, a power law, or a distribution of single rates across genes all produce the
     same improvement, and Y4 tests transferability rather than mechanism.
     Addition and withdrawal are not time-reverses of each other, so a Y5 failure has a
     biological reading as well as a statistical one.
     Bulk RNA throughout: a coherently relaxing population and cells switching at different times
     give the same mean curve.
"""
import os, sys, json, time, gzip, re, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_clock_replication.json"
SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
DEDEX = SCR / "dedex.npz"
OECTRL = SCR / "matched" / "OE_ctrl.txt.gz"

SEED = 249249
TAU_GRID = np.exp(np.linspace(np.log(0.15), np.log(120.0), 48))
RANGE_MIN, MAX_BASIS_CORR, RIDGE, SCREEN_RATIO = 0.10, 0.95, 1e-4, 2.0
Y2_BAR, Y3_MAX, Y4_BAR, Y5_BAR = 0.05, 0.25, 0.05, 0.05

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def admissible(taus, tr_t):
    return np.array([t for t in taus
                     if (np.exp(-tr_t / t).max() - np.exp(-tr_t / t).min()) >= RANGE_MIN])


def solve(B, Y):
    A = B.T @ B + RIDGE * np.trace(B.T @ B) / B.shape[1] * np.eye(B.shape[1])
    co = np.linalg.solve(A, B.T @ Y.T).T
    return co, Y - co @ B.T


def fit_fold(tr_t, tr_Y, te_t, model):
    n = tr_Y.shape[0]; ones = np.ones_like(tr_t)
    if model == "CONST":
        return np.repeat(tr_Y.mean(1)[:, None], len(te_t), 1), None
    if model == "LIN":
        co, _ = solve(np.stack([ones, tr_t], 1), tr_Y)
        return co @ np.stack([np.ones_like(te_t), te_t], 1).T, None
    adm = admissible(TAU_GRID, tr_t)
    if len(adm) == 0: return np.repeat(tr_Y.mean(1)[:, None], len(te_t), 1), None
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
    P = np.empty((n, len(te_t)))
    for pair in np.unique(bt, axis=0):
        m = (bt == pair).all(1)
        P[m] = (tr_Y[m].mean(1)[:, None] if pair[0] == 0 else
                bc[m] @ np.stack([np.ones_like(te_t), np.exp(-te_t / pair[0]),
                                  np.exp(-te_t / pair[1])], 1).T)
    return P, bt


def frozen_sse(t_all, Y, taus_fixed, interior):
    """Leave-one-timepoint-out with the taus FIXED: only amplitudes are estimated per gene."""
    s = 0.0
    for hi in interior:
        tr = np.array([j for j in range(len(t_all)) if j != hi])
        cols = [np.ones(len(tr))] + [np.exp(-t_all[tr] / tt) for tt in taus_fixed]
        co, _ = solve(np.stack(cols, 1), Y[:, tr])
        pc = [1.0] + [float(np.exp(-t_all[hi] / tt)) for tt in taus_fixed]
        s += float((((co @ np.array(pc)) - Y[:, hi]) ** 2).sum())
    return s


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "two-clock memory: boundary separated, gene holdout, opposite-direction replication"}
    say("=" * 104)
    say("LOOP 249 -- THE TWO-CLOCK RESULT, SEPARATED FROM THE BOUNDARY AND TESTED WHERE IT IS BLIND")
    say("=" * 104)
    say("     The interior numbers below (EXP2 over EXP1 +20.6%) were computed diagnostically")
    say("     AFTER loop 248's X1 failed. They enter this record as a diagnostic with that status")
    say("     attached, not as a blind test. Y3, Y4 and Y5 are the blind ones.")

    z = np.load(DEDEX, allow_pickle=True)
    V, times = z["V"], z["times"]
    T = np.array(sorted(set(times)))
    Ym = np.stack([V[:, times == t].mean(1) for t in T], 1)
    vt = Ym.var(1, ddof=1)
    vr = np.mean([V[:, times == t].var(1, ddof=1) for t in T], 0)
    keep = np.where(vt > SCREEN_RATIO * np.maximum(vr, 1e-12))[0]
    Y = Ym[keep]
    say(f"     {len(keep):,} screened genes, withdrawal timepoints {list(T)} h")

    # ---------------------------------------------------------------- Y1
    say("Y1 IS THE BOUNDARY DIAGNOSIS CORRECT?")
    adm_all = admissible(TAU_GRID, T)
    viol = [tau for tau in adm_all if np.exp(-0.0 / tau) <= np.exp(-T[1:] / tau).max()]
    say(f"     {len(adm_all)} admissible taus; for each, basis(t=0) against its maximum over the")
    say(f"     other eleven timepoints. Counterexamples (basis(0) NOT the strict maximum): "
        f"{len(viol)}")
    G.add("Y1", bool(len(viol) == 0), stat=float(len(viol)),
          if_true=lambda: f"Y1 PASS -- basis(0) is the strict maximum for all {len(adm_all)} "
                          f"admissible taus, so t=0 is an extrapolation fold for EVERY decay "
                          f"model, not a convenience to drop",
          if_false=lambda: f"Y1 FAIL -- {len(viol)} taus do not peak at t=0; dropping the fold is "
                           f"not structurally forced")
    res["Y1"] = {"n_admissible": int(len(adm_all)), "n_violations": int(len(viol))}

    interior = list(range(1, len(T) - 1))
    say(f"     interior folds: {[float(T[i]) for i in interior]} h "
        f"(t=0 and t={T[-1]:g} are the boundaries)")

    say("     leave-one-timepoint-out on interior folds, all four models ...")
    tot = {}
    for m in ("CONST", "LIN", "EXP1", "EXP2"):
        s = 0.0
        for hi in interior:
            tr = np.array([j for j in range(len(T)) if j != hi])
            P, _ = fit_fold(T[tr], Y[:, tr], T[hi:hi + 1], m)
            s += float(((P[:, 0] - Y[:, hi]) ** 2).sum())
        tot[m] = s
        say(f"       {m:<6} {s:10.2f}   vs CONST {1 - s / tot['CONST']:+.1%}")
    res["interior_sse"] = tot

    # ---------------------------------------------------------------- Y2
    say("Y2 THE INTERIOR COMPARISON -- POST-HOC, LABELLED AS SUCH")
    imp = 1 - tot["EXP2"] / tot["EXP1"]
    say(f"     EXP2 against EXP1 on interior folds: {imp:+.1%}")
    say(f"     THIS NUMBER WAS KNOWN BEFORE THIS FILE WAS WRITTEN. It is not evidence on its own.")
    G.add("Y2", bool(imp >= Y2_BAR), stat=float(imp), requires=("Y1",),
          if_true=lambda: f"Y2 PASS (post-hoc) -- {imp:+.1%}, a value already seen when the gate "
                          f"was written; Y3-Y5 carry the evidence",
          if_false=lambda: f"Y2 FAIL (post-hoc) -- {imp:+.1%}")
    res["Y2"] = {"improvement": imp, "status": "post-hoc"}

    # ---------------------------------------------------------------- Y3
    say("Y3 DOES THE INTERIOR RESULT SURVIVE SCRAMBLING THE CLOCK?")
    Tp = T[rng.permutation(len(T))]
    sc = se = 0.0
    for hi in interior:
        tr = np.array([j for j in range(len(T)) if j != hi])
        Pc, _ = fit_fold(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "CONST")
        Pe, _ = fit_fold(Tp[tr], Y[:, tr], Tp[hi:hi + 1], "EXP2")
        sc += float(((Pc[:, 0] - Y[:, hi]) ** 2).sum())
        se += float(((Pe[:, 0] - Y[:, hi]) ** 2).sum())
    real = 1 - tot["EXP2"] / tot["CONST"]
    shuf = 1 - se / sc
    f3 = shuf / real if abs(real) > 1e-9 else float("nan")
    say(f"     timepoints permuted: EXP2 improves {shuf:+.1%} against a real {real:+.1%} ({f3:.0%})")
    G.add("Y3", bool(np.isfinite(f3) and f3 <= Y3_MAX), stat=float(f3), requires=("Y1",),
          if_true=lambda: f"Y3 PASS -- collapses to {f3:.0%} with the clock scrambled",
          if_false=lambda: f"Y3 FAIL -- {f3:.0%} survives permuting time")
    res["Y3"] = {"real": real, "shuffled": shuf, "fraction": f3}

    # ---------------------------------------------------------------- Y4
    say("Y4 DO THE TAUS TRANSFER TO GENES THEY WERE NOT FITTED ON?")
    perm = rng.permutation(len(keep)); half = len(keep) // 2
    A, B = perm[:half], perm[half:]
    adm = admissible(TAU_GRID, T)
    best1, bt1 = np.inf, None
    for tau in adm:
        _, r = solve(np.stack([np.ones_like(T), np.exp(-T / tau)], 1), Y[A])
        s = float((r ** 2).sum())
        if s < best1: best1, bt1 = s, (tau,)
    best2, bt2 = np.inf, None
    for i, t1 in enumerate(adm):
        b1 = np.exp(-T / t1)
        for t2 in adm[i + 1:]:
            b2 = np.exp(-T / t2)
            c = np.corrcoef(b1, b2)[0, 1]
            if not np.isfinite(c) or abs(c) >= MAX_BASIS_CORR: continue
            _, r = solve(np.stack([np.ones_like(T), b1, b2], 1), Y[A])
            s = float((r ** 2).sum())
            if s < best2: best2, bt2 = s, (t1, t2)
    say(f"     taus chosen on {len(A):,} genes: one clock {bt1[0]:.2f} h; "
        f"two clocks {bt2[0]:.2f} h and {bt2[1]:.2f} h (ratio {bt2[1] / bt2[0]:.2f})")
    s1 = frozen_sse(T, Y[B], bt1, interior)
    s2 = frozen_sse(T, Y[B], bt2, interior)
    imp4 = 1 - s2 / s1
    say(f"     applied FROZEN to the other {len(B):,} genes: one clock {s1:.2f}, two clocks "
        f"{s2:.2f}  ({imp4:+.1%})")
    G.add("Y4", bool(imp4 >= Y4_BAR), stat=float(imp4), requires=("Y1",),
          if_true=lambda: f"Y4 PASS -- taus fitted on other genes still help by {imp4:+.1%}; two "
                          f"clocks are a property of the relaxation, not per-gene curve fitting",
          if_false=lambda: f"Y4 FAIL -- frozen taus transfer at {imp4:+.1%}, under {Y4_BAR:.0%}; "
                           f"the second clock is fitted per gene rather than shared")
    res["Y4"] = {"tau1": [float(x) for x in bt1], "tau2": [float(x) for x in bt2],
                 "sse_one": s1, "sse_two": s2, "improvement": imp4}

    # ---------------------------------------------------------------- Y5
    say("Y5 DOES IT REPLICATE IN THE OPPOSITE DIRECTION?")
    with gzip.open(OECTRL, "rt") as fh:
        cols = fh.readline().rstrip("\n").split("\t")[1:]
        idx, vals = [], []
        for ln in fh:
            q = ln.rstrip("\n").split("\t")
            idx.append(q[0].split(".")[0])
            vals.append([float(x) if x else np.nan for x in q[1:]])
    Vo = np.asarray(vals, np.float32)
    ho = np.array([float(re.search(r"dex\.(\d+)h", c).group(1)) for c in cols])
    To = np.array(sorted(set(ho)))
    Yo_all = np.stack([np.nanmean(Vo[:, ho == t], 1) for t in To], 1)
    gi = {g: i for i, g in enumerate(np.array(idx))}
    gsym = z["genes"]
    rows = [gi[gsym[g]] for g in keep if gsym[g] in gi]
    Yo = Yo_all[rows]
    vto = Yo.var(1, ddof=1)
    Yo = Yo[vto > np.percentile(vto, 50)]
    say(f"     ADDITION timecourse {list(To)} h, {Yo.shape[0]:,} of the withdrawal-screened genes "
        f"matched and above median time-variance")
    int_o = list(range(1, len(To) - 1))
    say(f"     interior folds there: {[float(To[i]) for i in int_o]} h")
    o1 = frozen_sse(To, Yo, bt1, int_o)
    o2 = frozen_sse(To, Yo, bt2, int_o)
    imp5 = 1 - o2 / o1
    say(f"     taus FROZEN from withdrawal, amplitudes free: one clock {o1:.2f}, two clocks "
        f"{o2:.2f}  ({imp5:+.1%})")
    G.add("Y5", bool(imp5 >= Y5_BAR), stat=float(imp5), requires=("Y1",),
          if_true=lambda: f"Y5 PASS -- the two timescales measured on withdrawal transfer to "
                          f"ADDITION at {imp5:+.1%}",
          if_false=lambda: f"Y5 FAIL -- withdrawal taus transfer to addition at {imp5:+.1%}; the "
                           f"two-clock structure is specific to relaxation or to that experiment")
    res["Y5"] = {"sse_one": o1, "sse_two": o2, "improvement": imp5,
                 "n_genes": int(Yo.shape[0]), "timepoints": [float(x) for x in To]}

    say("Y6 WHAT THIS CANNOT SHOW")
    say("     Dropping t=0 removes the only fold that could reveal a sub-30-minute component, so")
    say("     this loop is structurally blind to the fastest clock. No analysis recovers it from")
    say("     a grid whose second sample is at 30 minutes.")
    say("     Two exponentials beating one does not establish two physical memory pools. A")
    say("     stretched exponential, a power law, or a distribution of single rates across genes")
    say("     all produce the same improvement; Y4 tests transferability, not mechanism.")
    say("     Addition and withdrawal are not time-reverses, so a Y5 failure has a biological")
    say("     reading as well as a statistical one.")
    say("     Bulk RNA throughout: coherent relaxation and staggered switching give the same mean.")

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
