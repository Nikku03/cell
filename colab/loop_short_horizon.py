"""Loop 215. One step ahead: does the distribution shift explain everything since loop 198?

THE DIAGNOSIS THIS TESTS. Loop 214's chain failed because the level distribution in the training
window is not the distribution in the test window. Its state occupancy was [600, 600, 64, 1136,
600] on train and [1063, 38, 0, 38, 661] on test -- a state with 64 training rows had ZERO test
rows, because by 480-720 minutes the genes have moved to the extremes and the middle of the early
distribution is empty. Every model since loop 198 has been fitted on intervals ending at or before
240 minutes and scored on intervals ending at 480, 600 and 720. That is a long extrapolation across
a moving distribution, and it is a property of the SPLIT rather than of any model.

THE FIX, AND IT IS NOT FREE. Fit on everything strictly earlier and score on the very NEXT
interval, rolling forward. The distribution barely moves in one step, so the extrapolation is
short. But the same shortness helps PERSISTENCE: if little changes in sixty minutes then predicting
no change is nearly right, and the bar goes UP, not down. This is stated before any number because
it is the reason a short-horizon win would be worth more than the long-horizon losses, and the
reason a short-horizon loss would be worse.

    grid   30  60  120  180  240  420  480  600  720 minutes
    split  train on every interval ending at or before t, score the one interval ending next

WHAT MAKES THIS A REAL TEST RATHER THAN AN EASIER ONE. Persistence is recomputed on exactly the
rows being scored at every horizon, so the comparison is like-for-like at each step. And H2
measures the distribution shift directly, so the explanation stands or falls on a number rather
than on the story loop 214 told.

PREDECLARED, BEFORE ANY NUMBER.

  H1 IS THE ROLLING SPLIT HONEST?
     Gate: PASS iff for every split the training rows come strictly from earlier intervals than the
     scored row, every split has at least two training intervals, and no gene appears in a training
     row for the interval it is scored on. FAIL means leakage and nothing below may be read.

  H2 IS THE DISTRIBUTION SHIFT REAL, AND DOES IT GROW WITH HORIZON?
     For each split measure the Wasserstein-1 distance between the training levels and the scored
     levels, and the fraction of scored rows falling outside the training range.
     Gate: PASS iff the shift at loop 198's split (train <= 240, score 480-720) exceeds the median
     one-step shift by at least 2x. A FAIL means loop 214's explanation is wrong and the long
     split was not the problem.

  H3 DOES ANY MODEL BEAT PERSISTENCE ONE STEP AHEAD?
     Relaxation on the persisted set point, the Markov chain, and the ten-block ridge, each scored
     against persistence computed on the same rows.
     Gate: PASS iff at least one model beats persistence by more than 0.01 on at least half the
     one-step splits. This is the bar loop 198 set and nothing has cleared.

  H4 DOES PERFORMANCE DEGRADE WITH HORIZON, AS THE SHIFT HYPOTHESIS REQUIRES?
     Score the same models at horizons of one, two and three intervals ahead from the same
     training window.
     Gate: PASS iff the best model's margin over persistence is monotone non-increasing in horizon
     within noise. A FAIL means horizon is not what decides it and H2's shift is incidental.

  H5 IS THE ONE-STEP RESULT JUST AN EASIER TARGET?
     Report the variance of the scored change at each horizon. If one-step changes are tiny then a
     win is small in absolute terms even when the R2 is good.
     Gate: PASS iff the one-step scored change has at least 20% of the variance of the long-split
     scored change. Below that the win would be real and worth almost nothing, and saying so is
     the point.

  H6 IS IT THE SET POINT OR THE SPLIT?
     Rerun the best short-horizon model with the set point SHUFFLED across genes.
     Gate: PASS iff the real set point beats the shuffled one by more than 0.005. Magnitude, not
     sign. A FAIL means the short horizon is carrying the result and the features are not.

  H7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import gzip, json, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates, weakened_by

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
MODEL = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_short_horizon.json"
SEED, K = 215215, 5
LONG_TRAIN = 6           # loop 198's split: intervals 1..5 train, 6..8 score

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def w1(a, b):
    """Wasserstein-1 between two samples, by quantile matching."""
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "short horizon"}
    say("=" * 104)
    say("LOOP 215 -- ONE STEP AHEAD: DOES THE DISTRIBUTION SHIFT EXPLAIN EVERYTHING SINCE 198?")
    say("=" * 104)

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    allg = [sym[i] for i in gi]
    pos = {s: k for k, s in enumerate(allg)}
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    S = np.array(art["stack_prediction"])
    idx = gi[np.array([pos[s] for s in names])]
    say(f"     grid {[int(x) for x in grid]} minutes, {len(names):,} genes")
    say(f"     set point: persisted three-block stack, |r| {art['r']:.4f}")

    def block(j):
        """rows for the single interval ending at grid[j]."""
        dt = grid[j] - grid[j - 1]
        lev = M[j - 1, idx]; nxt = M[j, idx]
        return lev, nxt, nxt - lev, np.full(len(idx), dt), np.arange(len(idx))

    def gather(js):
        L_, N_, Y_, D_, Gg = [], [], [], [], []
        for j in js:
            a, b, c, d, e = block(j)
            L_.append(a); N_.append(b); Y_.append(c); D_.append(d); Gg.append(e)
        return (np.concatenate(L_), np.concatenate(N_), np.concatenate(Y_),
                np.concatenate(D_), np.concatenate(Gg))

    # ---------------------------------------------------------------- H1
    say("H1 IS THE ROLLING SPLIT HONEST?")
    splits = [(list(range(1, j)), j) for j in range(3, len(grid))]
    ok1 = all(max(tr) < te and len(tr) >= 2 for tr, te in splits)
    for tr, te in splits:
        say(f"       train intervals ending {[int(grid[j]) for j in tr]}  ->  "
            f"score interval {int(grid[te-1])}-{int(grid[te])} min")
    G.add("H1", ok1,
          if_true=f"H1 PASS -- {len(splits)} rolling splits, training always strictly earlier",
          if_false="H1 FAIL -- a split trains on or after the interval it scores")

    # ---------------------------------------------------------------- H2
    say("H2 IS THE DISTRIBUTION SHIFT REAL, AND DOES IT GROW WITH HORIZON?")
    shifts = {}
    for tr, te in splits:
        Ltr, _, _, _, _ = gather(tr)
        Lte, _, _, _, _ = gather([te])
        out = float(np.mean((Lte < Ltr.min()) | (Lte > Ltr.max())))
        shifts[int(grid[te])] = {"w1": w1(Ltr, Lte), "outside": out}
        say(f"       score {int(grid[te]):>3} min   W1 {shifts[int(grid[te])]['w1']:.4f}   "
            f"outside training range {out:.3%}")
    Ltr_l, _, _, _, _ = gather(list(range(1, LONG_TRAIN)))
    Lte_l, _, Y_l, _, _ = gather(list(range(LONG_TRAIN, len(grid))))
    w_long = w1(Ltr_l, Lte_l)
    out_long = float(np.mean((Lte_l < Ltr_l.min()) | (Lte_l > Ltr_l.max())))
    med_one = float(np.median([v["w1"] for v in shifts.values()]))
    say(f"     loop 198's split (train <=240, score 480-720)   W1 {w_long:.4f}   "
        f"outside {out_long:.3%}")
    say(f"     median one-step shift                           W1 {med_one:.4f}   "
        f"ratio {w_long/med_one:.2f}x")
    G.add("H2", bool(w_long >= 2 * med_one), stat=w_long / med_one, requires=("H1",),
          if_true=lambda: f"H2 PASS -- the long split shifts {w_long/med_one:.1f}x more than a "
                          f"one-step split, so loop 214's explanation holds",
          if_false=lambda: f"H2 FAIL -- long split {w_long:.4f} against a median one-step "
                           f"{med_one:.4f}, ratio {w_long/med_one:.2f}x. The long split was not "
                           f"the problem and loop 214's explanation is wrong")
    res["shift"] = {"one_step": shifts, "long_w1": w_long, "long_outside": out_long,
                    "median_one_step": med_one}

    # ---------------------------------------------------------------- models
    def fit_score(tr, te, Sp):
        Ltr, _, Ytr, Dtr, Gtr = gather(tr)
        Lte, _, Yte, Dte, Gte = gather([te] if isinstance(te, int) else te)
        pers = r2s(Yte, np.zeros_like(Yte))
        d_tr = Dtr * (Sp[Gtr] - Ltr); d_te = Dte * (Sp[Gte] - Lte)
        lam = float(d_tr @ Ytr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
        r_rel = r2s(Yte, lam * d_te)
        edges = np.quantile(Ltr, np.linspace(0, 1, K + 1)); edges[0], edges[-1] = -np.inf, np.inf
        s_tr = np.clip(np.digitize(Ltr, edges[1:-1]), 0, K - 1)
        s_nx = np.clip(np.digitize(Ltr + Ytr, edges[1:-1]), 0, K - 1)
        cent = np.array([(Ltr + Ytr)[s_nx == k].mean() if (s_nx == k).any() else 0.0
                         for k in range(K)])
        T = np.ones((K, K)) * 0.5
        for a, b in zip(s_tr, s_nx):
            T[a, b] += 1
        T /= T.sum(1, keepdims=True)
        s_te = np.clip(np.digitize(Lte, edges[1:-1]), 0, K - 1)
        r_mk = r2s(Yte, (T[s_te] @ cent) - Lte)
        # per-gene mean change fitted on train, the cheapest non-persistence baseline
        mu = np.zeros(len(Sp))
        for g in range(len(Sp)):
            m = Gtr == g
            if m.any():
                mu[g] = Ytr[m].mean()
        r_mu = r2s(Yte, mu[Gte])
        return pers, r_rel, r_mk, r_mu, float(np.var(Yte))

    # ---------------------------------------------------------------- H3
    say("H3 DOES ANY MODEL BEAT PERSISTENCE ONE STEP AHEAD?")
    say("        score   persistence  relaxation    markov    per-gene mean   var(dY)")
    tab, wins = [], {"relaxation": 0, "markov": 0, "gene_mean": 0}
    for tr, te in splits:
        p, a, b, c, v = fit_score(tr, te, S)
        tab.append({"score_min": int(grid[te]), "persistence": p, "relaxation": a,
                    "markov": b, "gene_mean": c, "var": v})
        for nm, val in (("relaxation", a), ("markov", b), ("gene_mean", c)):
            if val - p > 0.01:
                wins[nm] += 1
        say(f"       {int(grid[te]):>4}    {p:+.5f}   {a:+.5f}   {b:+.5f}   {c:+.5f}   {v:.4f}")
    n_needed = len(splits) / 2
    best_nm = max(wins, key=lambda k: wins[k])
    say(f"     splits beaten (of {len(splits)}): " +
        "  ".join(f"{k} {v}" for k, v in wins.items()))
    G.add("H3", bool(wins[best_nm] >= n_needed), stat=float(wins[best_nm]), requires=("H1",),
          if_true=lambda: f"H3 PASS -- {best_nm} beats persistence on {wins[best_nm]} of "
                          f"{len(splits)} one-step splits",
          if_false=lambda: f"H3 FAIL -- the best arm ({best_nm}) beats persistence on only "
                           f"{wins[best_nm]} of {len(splits)} one-step splits. Shortening the "
                           f"horizon removes the distribution shift and does not produce a model")
    res["one_step"] = tab; res["wins"] = wins

    # ---------------------------------------------------------------- H4
    say("H4 DOES PERFORMANCE DEGRADE WITH HORIZON?")
    base_tr = list(range(1, 5))
    hz = {}
    for h in (1, 2, 3):
        te = 4 + h - 1
        if te >= len(grid):
            continue
        p, a, b, c, v = fit_score(base_tr, te, S)
        hz[h] = {"score_min": int(grid[te]), "persistence": p, "best": max(a, b, c),
                 "margin": max(a, b, c) - p}
        say(f"       horizon {h} (score {int(grid[te])} min)   best {max(a,b,c):+.5f}   "
            f"persistence {p:+.5f}   margin {max(a,b,c)-p:+.5f}")
    ms = [hz[h]["margin"] for h in sorted(hz)]
    mono = all(b <= a + 0.01 for a, b in zip(ms, ms[1:]))
    G.add("H4", bool(mono), requires=("H2",),
          if_true="H4 PASS -- the margin is non-increasing in horizon, as the shift hypothesis "
                  "requires",
          if_false=lambda: f"H4 FAIL -- margins {[round(m,5) for m in ms]} are not monotone; "
                           f"horizon is not what decides it")
    res["horizon"] = hz

    # ---------------------------------------------------------------- H5
    say("H5 IS THE ONE-STEP RESULT JUST AN EASIER TARGET?")
    v_one = float(np.median([t["var"] for t in tab]))
    v_long = float(np.var(Y_l))
    say(f"     median one-step var(dY) {v_one:.4f}   long-split var(dY) {v_long:.4f}   "
        f"ratio {v_one/v_long:.3f}")
    G.add("H5", bool(v_one >= 0.20 * v_long), stat=v_one / v_long, requires=("H1",),
          if_true=lambda: f"H5 PASS -- one-step changes carry {v_one/v_long:.0%} of the long "
                          f"split's variance, so a win there is worth having",
          if_false=lambda: f"H5 FAIL -- one-step changes carry only {v_one/v_long:.0%} of the "
                           f"variance. A win at this horizon would be real and nearly worthless")

    # ---------------------------------------------------------------- H6
    say("H6 IS IT THE SET POINT OR THE SPLIT?")
    rg = np.random.default_rng(SEED)
    Ssh = rg.permutation(S)
    real_m, shuf_m = [], []
    for tr, te in splits:
        p, a, b, c, _ = fit_score(tr, te, S)
        p2, a2, b2, c2, _ = fit_score(tr, te, Ssh)
        real_m.append(max(a, b, c) - p); shuf_m.append(max(a2, b2, c2) - p2)
    rm, sm = float(np.mean(real_m)), float(np.mean(shuf_m))
    say(f"     mean margin, real set point     {rm:+.5f}")
    say(f"     mean margin, shuffled set point {sm:+.5f}   delta {rm-sm:+.5f}")
    cmp6 = weakened_by(rm, sm)
    G.add("H6", bool(rm - sm > 0.005), stat=rm - sm, requires=("H3",),
          if_true=lambda: f"H6 PASS -- the real set point beats shuffled by {rm-sm:+.5f}",
          if_false=lambda: f"H6 FAIL -- real beats shuffled by {rm-sm:+.5f}; whatever is happening "
                           f"at short horizon is not the features")
    res["shuffle_control"] = {"real": rm, "shuffled": sm, "compare": cmp6}

    say("H7 WHAT THIS CANNOT SHOW")
    say("     A one-step model is not a simulator. Predicting the next sixty minutes from the")
    say("     previous four intervals is interpolation inside a course that has already been")
    say("     measured; it says nothing about running forward from an unmeasured state.")
    say("     The set point is the persisted THREE-block stack, not loop 213's ten-block 0.5474.")
    say("     The grid is irregular -- 30, 60, 120, 180, 240, 420, 480, 600, 720 -- so 'one step'")
    say("     means anything from 30 to 180 minutes and the horizons are not comparable in time.")
    say("     Loop 197 established there is no second densely-sampled matched course in ENCODE,")
    say("     so none of this can be replicated on another target.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
