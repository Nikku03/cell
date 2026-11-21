"""Loop 214. A Markov chain instead of a relaxation: is the FUNCTIONAL FORM the problem?

WHAT LOOP 213 FORCED. Three set points have now been scored through the same first-order
relaxation, and their forward performance runs BACKWARDS against their quality:

    set point |r| 0.3902   ->  margin over persistence  +0.00154
    set point |r| 0.4761   ->                           +0.00046
    set point |r| 0.5474   ->                           -0.00467

Getting better at predicting where a gene is heading has not made the forward model better; the
best set point this project has produced made it worse. Every one of those runs used the same
update law, dM/dt = k - a.M, so the law itself is the thing that has never been varied. Loop 210's
C4 was read as "the form is right and the set point is missing"; loop 212 retracted the shrinkage
half of that, and loop 213's F6 puts the rest in doubt.

THE ALTERNATIVE FORM THE ARCHITECTURE ALREADY SPECIFIES. REM-Cell's Stage-6 table maps molecular
switching to a Markov chain, p_{t+1} = p_t P. That is a genuinely different object from a
relaxation: the state is DISCRETE, so a prediction is a probability over states rather than a
magnitude, and the expected change is bounded by construction. A relaxation with a bad set point
can push a gene arbitrarily far the wrong way -- which is exactly the pathology loop 210's C4
found and mislabelled -- while a chain can at worst assign the wrong state and move by one step.

THE GUARD THIS DESIGN NEEDS ABOVE ALL OTHERS. A transition matrix close to the identity IS
persistence wearing different notation. Loop 212 already produced one vacuous result this way: r^2
shrinkage made every predictor numerically identical to predicting no change, and the gate that
should have caught it instead reported a crossover. So G1 refuses to let the chain be scored until
its off-diagonal mass is measured and shown to be substantial, and every comparison below is
against persistence computed on the same rows.

PREDECLARED, BEFORE ANY NUMBER.

  G1 IS THE CHAIN A CHAIN?
     Discretise each gene's expression level into K quantile states and fit the transition matrix
     on training intervals only.
     Gate: PASS iff every state is occupied in both train and test, the rows sum to 1 within 1e-9,
     AND the mean off-diagonal mass exceeds 0.15. A near-identity matrix is persistence in
     disguise and must not be scored as a model.

  G2 DOES AN UNCONDITIONED CHAIN BEAT PERSISTENCE?
     One global transition matrix, no features at all, predicting the expected next level.
     Gate: PASS iff held-out-in-time R2 exceeds persistence by more than 0.01.
     This is the cheapest possible dynamic model and nothing in this project has cleared that bar.

  G3 DOES CONDITIONING ON THE SET POINT HELP?
     Separate transition matrices for genes the loop-213 stack predicts will rise, stay, or fall,
     fitted on training intervals only.
     Gate: PASS iff the conditioned chain beats the unconditioned one by more than 0.01 in R2.
     A FAIL means the best set point this project has -- |r| 0.5474 -- does not inform even a
     three-way choice of transition matrix, which would be a stronger negative than F6's.

  G4 CHAIN AGAINST RELAXATION, SAME SET POINT, SAME ROWS.
     Gate: PASS iff the better of the two chains beats the relaxation's R2. This is the direct
     test of whether the functional form was the problem.

  G5 DOES THE STATE COUNT DECIDE THE ANSWER?
     Sweep K over 3, 5, 7 and 9.
     Gate: PASS iff the verdict of G2 -- beats persistence or does not -- is the same at every K.
     If it flips inside the sweep then K is doing the work and no conclusion may be drawn.

  G6 IS THE CONDITIONING REAL?
     Refit G3 with the set point SHUFFLED across genes.
     Gate: PASS iff the real conditioning beats the shuffled conditioning by more than 0.005 in
     R2. Magnitude, not sign. Requires G3 to have produced a number.

  G7 WHAT THIS CANNOT SHOW.
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
FULL = "outputs/loop_full_stack.json"
OUT = "outputs/loop_markov.json"
N_TRAIN, SEED = 6, 214214
KS = (3, 5, 7, 9)

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "markov forward model"}
    say("=" * 104)
    say("LOOP 214 -- A MARKOV CHAIN INSTEAD OF A RELAXATION: IS THE FUNCTIONAL FORM THE PROBLEM?")
    say("=" * 104)

    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    allg = [sym[i] for i in gi]
    pos = {s: k for k, s in enumerate(allg)}
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    S_stack = np.array(art["stack_prediction"])
    say(f"     reloaded {MODEL.name}: {len(names):,} genes, three-block set point |r| {art['r']:.4f}")
    # the loop-213 ten-block set point, recomputed is not available here, so the persisted
    # three-block stack is used and the difference is stated rather than hidden
    say(f"     NOTE the persisted artefact is the THREE-block stack. Loop 213's ten-block set")
    say(f"     point (|r| 0.5474) was not persisted, so G3/G6 condition on the three-block one")
    say(f"     and G4's relaxation arm is recomputed from it for a like-for-like comparison.")

    rows_g = np.array([pos[s] for s in names])
    idx = gi[rows_g]

    def rows(lo, hi):
        lev, nxt, gg, dts = [], [], [], []
        for j in range(1, len(grid)):
            if not (lo <= j < hi):
                continue
            dt = grid[j] - grid[j - 1]
            for kk, i in enumerate(idx):
                lev.append(M[j - 1, i]); nxt.append(M[j, i]); gg.append(kk); dts.append(dt)
        return (np.array(lev), np.array(nxt), np.array(gg), np.array(dts))
    ltr, ntr, gtr, dtr = rows(1, N_TRAIN)
    lte, nte, gte, dte = rows(N_TRAIN, len(grid))
    ytr, yte = ntr - ltr, nte - lte
    pers = r2s(yte, np.zeros_like(yte))
    say(f"     harness: train {len(ytr):,} rows, score {len(yte):,}, persistence {pers:+.5f}")

    def build_chain(K, lev_tr, nxt_tr, sub=None):
        edges = np.quantile(lev_tr, np.linspace(0, 1, K + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        s_tr = np.clip(np.digitize(lev_tr, edges[1:-1]), 0, K - 1)
        s_nx = np.clip(np.digitize(nxt_tr, edges[1:-1]), 0, K - 1)
        centres = np.array([nxt_tr[s_nx == k].mean() if (s_nx == k).any() else 0.0
                            for k in range(K)])
        if sub is None:
            sub = np.zeros(len(lev_tr), int)
        P = {}
        for c in np.unique(sub):
            m = sub == c
            T = np.ones((K, K)) * 0.5           # Laplace prior, so no row is empty
            for a, b in zip(s_tr[m], s_nx[m]):
                T[a, b] += 1
            P[int(c)] = T / T.sum(1, keepdims=True)
        return edges, centres, P

    def predict(K, edges, centres, P, lev, sub=None):
        s = np.clip(np.digitize(lev, edges[1:-1]), 0, K - 1)
        if sub is None:
            sub = np.zeros(len(lev), int)
        out = np.empty(len(lev))
        for i, (si, ci) in enumerate(zip(s, sub)):
            T = P.get(int(ci), P[int(list(P)[0])])
            out[i] = float(T[si] @ centres)
        return out - lev

    # ---------------------------------------------------------------- G1
    say("G1 IS THE CHAIN A CHAIN?")
    K = 5
    edges, centres, P0 = build_chain(K, ltr, ntr)
    T = P0[0]
    offdiag = float(1.0 - np.mean(np.diag(T)))
    s_tr = np.clip(np.digitize(ltr, edges[1:-1]), 0, K - 1)
    s_te = np.clip(np.digitize(lte, edges[1:-1]), 0, K - 1)
    occ_tr = np.array([int((s_tr == k).sum()) for k in range(K)])
    occ_te = np.array([int((s_te == k).sum()) for k in range(K)])
    rowsum = float(np.max(np.abs(T.sum(1) - 1)))
    say(f"     K = {K}, state occupancy train {occ_tr.tolist()}  test {occ_te.tolist()}")
    say(f"     row sums deviate by at most {rowsum:.2e}")
    say(f"     mean off-diagonal mass {offdiag:.4f}  (a near-identity chain is persistence)")
    say(f"     transition matrix diagonal {np.round(np.diag(T),3).tolist()}")
    ok1 = bool((occ_tr > 0).all() and (occ_te > 0).all() and rowsum < 1e-9 and offdiag > 0.15)
    G.add("G1", ok1, stat=offdiag,
          if_true=lambda: f"G1 PASS -- all states occupied, rows normalised, off-diagonal mass "
                          f"{offdiag:.3f} so the chain actually moves",
          if_false=lambda: f"G1 FAIL -- occupancy {occ_tr.tolist()}/{occ_te.tolist()}, rowsum "
                           f"{rowsum:.1e}, off-diagonal {offdiag:.4f}. A chain that does not move "
                           f"is persistence in different notation")

    # ---------------------------------------------------------------- G2
    say("G2 DOES AN UNCONDITIONED CHAIN BEAT PERSISTENCE?")
    p_un = predict(K, edges, centres, P0, lte)
    r2_un = r2s(yte, p_un)
    say(f"     unconditioned chain R2 {r2_un:+.5f}   persistence {pers:+.5f}   "
        f"margin {r2_un-pers:+.5f}")
    G.add("G2", bool(r2_un - pers > 0.01), stat=r2_un, requires=("G1",),
          if_true=lambda: f"G2 PASS -- a global transition matrix with NO features clears "
                          f"persistence by {r2_un-pers:+.5f}",
          if_false=lambda: f"G2 FAIL -- {r2_un-pers:+.5f}")

    # ---------------------------------------------------------------- G3
    say("G3 DOES CONDITIONING ON THE SET POINT HELP?")
    q = np.quantile(S_stack, [1 / 3, 2 / 3])
    cls = np.digitize(S_stack, q)
    say(f"     genes split by predicted set point: "
        f"{[int((cls==c).sum()) for c in range(3)]} (fall / stay / rise)")
    edges_c, centres_c, Pc = build_chain(K, ltr, ntr, sub=cls[gtr])
    p_c = predict(K, edges_c, centres_c, Pc, lte, sub=cls[gte])
    r2_c = r2s(yte, p_c)
    say(f"     conditioned chain R2 {r2_c:+.5f}   unconditioned {r2_un:+.5f}   "
        f"gain {r2_c-r2_un:+.5f}")
    G.add("G3", bool(r2_c - r2_un > 0.01), stat=r2_c, requires=("G1",),
          if_true=lambda: f"G3 PASS -- conditioning buys {r2_c-r2_un:+.5f}",
          if_false=lambda: f"G3 FAIL -- conditioning buys {r2_c-r2_un:+.5f}. The best set point "
                           f"this project has does not inform even a three-way choice of "
                           f"transition matrix")

    # ---------------------------------------------------------------- G4
    say("G4 CHAIN AGAINST RELAXATION, SAME SET POINT, SAME ROWS")
    d_tr = dtr * (S_stack[gtr] - ltr); d_te = dte * (S_stack[gte] - lte)
    lam = float(d_tr @ ytr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
    r2_rel = r2s(yte, lam * d_te)
    best_chain = max(r2_un, r2_c)
    say(f"     relaxation (lambda {lam:+.6f})  R2 {r2_rel:+.5f}")
    say(f"     best chain                     R2 {best_chain:+.5f}")
    say(f"     persistence                    R2 {pers:+.5f}")
    G.add("G4", bool(best_chain > r2_rel), stat=best_chain, requires=("G1",),
          if_true=lambda: f"G4 PASS -- the chain beats the relaxation by "
                          f"{best_chain-r2_rel:+.5f}; the functional form WAS part of the problem",
          if_false=lambda: f"G4 FAIL -- the chain scores {best_chain:+.5f} against the "
                           f"relaxation's {r2_rel:+.5f}. Changing the functional form does not "
                           f"rescue it")
    res["scores"] = {"persistence": pers, "unconditioned": r2_un, "conditioned": r2_c,
                     "relaxation": r2_rel, "lambda": lam, "offdiag": offdiag}

    # ---------------------------------------------------------------- G5
    say("G5 DOES THE STATE COUNT DECIDE THE ANSWER?")
    sweep = {}
    for k in KS:
        e, c, p = build_chain(k, ltr, ntr)
        v = r2s(yte, predict(k, e, c, p, lte))
        od = float(1.0 - np.mean(np.diag(p[0])))
        sweep[k] = {"r2": v, "offdiag": od, "beats": bool(v - pers > 0.01)}
        say(f"       K={k}  R2 {v:+.5f}   off-diagonal {od:.3f}   "
            f"beats persistence {v-pers>0.01}")
    verdicts = {v["beats"] for v in sweep.values()}
    G.add("G5", bool(len(verdicts) == 1), requires=("G1",),
          if_true=lambda: f"G5 PASS -- the verdict is {list(verdicts)[0]} at every K, so the "
                          f"state count is not doing the work",
          if_false=lambda: f"G5 FAIL -- the verdict flips inside the sweep; K decides the answer "
                           f"and no conclusion may be drawn from G2")
    res["k_sweep"] = sweep

    # ---------------------------------------------------------------- G6
    say("G6 IS THE CONDITIONING REAL?")
    rg = np.random.default_rng(SEED)
    cls_sh = rg.permutation(cls)
    e2, c2, Ps = build_chain(K, ltr, ntr, sub=cls_sh[gtr])
    r2_sh = r2s(yte, predict(K, e2, c2, Ps, lte, sub=cls_sh[gte]))
    say(f"     real conditioning     R2 {r2_c:+.5f}")
    say(f"     shuffled conditioning R2 {r2_sh:+.5f}   delta {r2_c-r2_sh:+.5f}")
    cmp6 = weakened_by(r2_c - pers, r2_sh - pers)
    G.add("G6", bool(r2_c - r2_sh > 0.005), stat=r2_c - r2_sh, requires=("G3",),
          if_true=lambda: f"G6 PASS -- real conditioning beats shuffled by {r2_c-r2_sh:+.5f}",
          if_false=lambda: f"G6 FAIL -- real beats shuffled by only {r2_c-r2_sh:+.5f}; the "
                           f"conditioning is not reading the set point")
    res["conditioning_control"] = {"real": r2_c, "shuffled": r2_sh, "compare": cmp6}

    say("G7 WHAT THIS CANNOT SHOW")
    say("     The chain is fitted on LEVELS discretised into quantile states, which throws away")
    say("     magnitude within a state. That is the price of the discreteness that bounds it, and")
    say("     it means a chain cannot express a large move even when a large move is correct.")
    say("     The persisted artefact is the three-block set point, not loop 213's ten-block")
    say("     0.5474, so G3 and G6 condition on the weaker one. If conditioning fails here it")
    say("     could still succeed on the better set point, and that is not tested.")
    say("     A time-homogeneous chain assumes the transition matrix does not change with time")
    say("     since the drug. The A549 grid spans 30 to 720 minutes and the biology plainly does")
    say("     change over it; loop 191c already found a batch discontinuity at 25-30 minutes.")
    say("     Still one cell line, one perturbation, one channel, 600 genes.")

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
