"""Loop 231. Testing the framework itself: is dS/dt = F(S,G,E,H) identifiable from this data?

THE PROPOSAL, AND WHAT IN IT IS ACTUALLY TESTABLE. The framework says a cell is a stochastic,
adaptive, information-controlled dynamical system:

    dS = F_theta(S, G, E, H) dt + Sigma(S) dW        with   theta_{t+1} = A(theta_t, S_t, E_t)

and that the goal is not to predict which genes move but to INFER THE VECTOR FIELD F. That is a
sharper and more falsifiable claim than it first appears, because it makes four commitments that
can each be measured against data already on disk, and this loop measures them rather than
discussing them.

  AUTONOMY.      F is written as a function of STATE. If that is right, knowing where the system
                 IS should predict where it is GOING better than knowing what time it is. If the
                 clock beats the state, the system is not autonomous in these coordinates and F(S)
                 is the wrong object regardless of how it is fitted.
  STATIONARITY.  theta is allowed to adapt, which means F is NOT fixed. That is testable directly:
                 fit the transition on early intervals and test on late ones, against fitting on a
                 random half of intervals and testing on the other half. If theta adapts, the
                 time-ordered split must be worse than the random split. If they are equal, the
                 adaptive layer is not doing anything measurable here and the simpler static F is
                 sufficient.
  GEOMETRY.      dS/dt = -grad U(S) + F_active + xi says the flow is a gradient field plus a
                 driven part. A gradient field is curl-free, which for a linearised flow means the
                 Jacobian is SYMMETRIC. The antisymmetric part of the estimated Jacobian is
                 therefore a direct measurement of how much of the dynamics the potential picture
                 cannot express.
  IDENTIFIABILITY. All of the above presuppose that dS/dt is measurable. Loop 216 measured the
                 replicate ceiling on the per-interval change at -0.54028 against the plateau's
                 +0.83380. If the derivative is not reproducible between replicates of the same
                 experiment, F cannot be inferred from this data by any method, and that is a
                 statement about the data rather than about the framework.

L1 THEREFORE RUNS FIRST AND EVERYTHING REQUIRES IT. This is not a formality. Five loops in this
project have now failed by measuring something on a quantity that could not carry the answer, and
the vector field is exactly that kind of object: it is a derivative, derivatives amplify noise, and
this series has four replicates.

WHAT IS NOT TESTED HERE, and why. The evolution layer needs generations and we have one cell line
in one experiment; loops 225 to 227 already tested the one evolutionary signal that is fetchable
and found protein-identity co-evolution at AUC 0.5151 against a 0.60 bar. The organism layer needs
many coupled cells. The physical constraint layer is partly tested already: loop 206 computed
equilibrium occupancy at r -0.0133 against measured occupancy at +0.2932, and loop 228 found the
driven generalisation's effect inside its own baseline's split noise.

PREDECLARED, BEFORE ANY NUMBER.

  L1 IS THE DERIVATIVE MEASURABLE AT ALL?  -- everything requires it
     Replicate agreement on the per-interval change, against the same statistic for the level.
     Gate: PASS iff the change's cross-replicate Pearson exceeds 0.30, which is the floor at
     which a fitted vector field could carry signal rather than noise. Loop 222 measured the
     per-interval reliabilities as 30->60 +0.202, 60->120 +0.292, 120->180 +0.036, 180->240
     +0.014, 240->420 +0.317, 420->480 +0.145, 480->600 +0.127, 600->720 -0.064.

  L2 IS dS/dt A FUNCTION OF STATE, OR OF THE CLOCK?  -- the autonomy test
     Three arms predicting the per-interval change, held out BY GENE so no gene appears in both
     train and test: STATE (the gene's current level and its recent history), CLOCK (interval
     identity alone, one-hot), and BOTH.
     Gate: PASS iff STATE beats CLOCK by at least 0.05 in held-out |r|, paired across 20 splits.
     A FAIL means that in these coordinates the system is better described by elapsed time than
     by where it is, and F(S) is the wrong object.

  L3 IS F STATIONARY, OR DOES theta ADAPT?
     Fit the transition on the first four intervals and test on the last four, against fitting on
     a random half of intervals and testing on the other half, everything else identical.
     Gate: PASS iff the two differ by more than 2 standard errors. The DIRECTION is reported and
     not gated: time-ordered worse than random means theta adapts; equal means the static F is
     sufficient; and gating on the direction I expect is a defect this project has committed.

  L4 IS THE FLOW GRADIENT-LIKE?
     Project the trajectories onto their leading components, estimate the linear map J from state
     to derivative, and split J into symmetric and antisymmetric parts. A pure -grad U flow has a
     symmetric Jacobian and zero antisymmetric part.
     Gate: PASS iff J is estimable with a condition number below 1e6, so the decomposition means
     something. The curl fraction ||J_anti|| / ||J|| is REPORTED, not gated -- there is no
     principled bar and inventing one would be the same error as gating on a direction.

  L5 CONTROL: DOES ANY OF THIS SURVIVE SHUFFLING?
     Repeat L2's best arm with the gene labels of the target permuted.
     Gate: PASS iff the real arm exceeds the shuffled arm by at least 0.10 on every split.

  L6 WHAT THIS CANNOT SHOW -- written before the run.
     A negative on L2 is a statement about THESE coordinates -- 1,336 genes' log2 levels -- not
     about the framework. The state the theory refers to includes protein, PTM, metabolite,
     localisation and mechanics, none of which this series measures. Failing to find an autonomous
     vector field in a projection is expected if the projection omits the variables that make it
     autonomous, and that is the most likely reading of a FAIL.
     L4's Jacobian is LINEAR. A curl-free nonlinear flow can look rotational under a linear fit,
     so a large antisymmetric part bounds how well a linear potential describes the flow, not how
     well any potential does.
     L3 cannot separate theta adapting from the environment changing: dexamethasone exposure is
     itself time-varying in effect, so a non-stationary F is consistent with a static controller
     driven by a changing input.
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = "outputs/loop_vector_field.json"
SP = L191.SP
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
MIN_TPM, SEED, NSPLIT, KPC = 1.0, 231231, 20, 6
L1_BAR, L2_BAR, CTRL_BAR = 0.30, 0.05, 0.10

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def pear(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    R = lam * np.eye(A.shape[1]); R[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ ytr)
    return np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))]) @ w


def main():
    t0 = time.time()
    G = Gates(emit=say)
    rng = np.random.default_rng(SEED)
    res = {"test": "is dS/dt = F(S) identifiable from this data"}
    say("=" * 104)
    say("LOOP 231 -- TESTING THE FRAMEWORK: IS THE VECTOR FIELD IDENTIFIABLE?")
    say("=" * 104)
    say("     The proposal makes four measurable commitments: that F is a function of STATE")
    say("     (autonomy), that theta adapts (non-stationarity), that the flow is a gradient plus")
    say("     a driven part (symmetric Jacobian), and -- underneath all of them -- that dS/dt is")
    say("     measurable at all. L1 tests the last one first because it gates the rest.")

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    g = np.array(GRID, float)
    base = {r: tpm[(mins == 30) & (reps == r)].mean(0) for r in (1, 2, 3, 4)}
    sel = np.where(np.all([base[r] >= MIN_TPM for r in (1, 2, 3, 4)], axis=0))[0]
    V = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        V[r] = Mi[:, sel]
    D = {r: np.array([V[r][j] - V[r][j - 1] for j in range(1, len(g))]) for r in (1, 2, 3, 4)}
    NG, NI = len(sel), len(g) - 1
    say(f"     {NG:,} genes, {len(g)} timepoints, {NI} intervals, 4 replicates")

    # ---------------------------------------------------------------- L1
    say("L1 IS THE DERIVATIVE MEASURABLE AT ALL?")
    PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    r_change = np.array([pear(D[a], D[b]) for a, b in PAIRS])
    r_level = np.array([pear(V[a][1:], V[b][1:]) for a, b in PAIRS])
    say("     cross-replicate agreement, all six pairs:")
    for (a, b), rc, rl in zip(PAIRS, r_change, r_level):
        say(f"       {a} vs {b}   change {rc:+.4f}   level {rl:+.4f}")
    mc, ml = float(np.mean(r_change)), float(np.mean(r_level))
    say(f"     mean over pairs: change {mc:+.4f}   level {ml:+.4f}")
    say("     loop 216 measured the plateau ceiling +0.83380 and the change ceiling -0.54028")
    G.add("L1", bool(mc > L1_BAR), stat=float(mc),
          if_true=lambda: f"L1 PASS -- the derivative reproduces at {mc:+.4f} across replicates, "
                          f"enough for a fitted vector field to carry signal",
          if_false=lambda: f"L1 FAIL -- the derivative reproduces at {mc:+.4f} against a "
                           f"{L1_BAR:.2f} bar while the LEVEL reproduces at {ml:+.4f}. dS/dt is "
                           f"not measurable in this data, so F cannot be inferred from it by any "
                           f"method. That is a fact about the data, not about the framework")
    res["identifiability"] = {"change_mean": mc, "level_mean": ml,
                              "change_pairs": [float(x) for x in r_change],
                              "level_pairs": [float(x) for x in r_level]}

    # ---------------------------------------------------------------- L2
    say("L2 IS dS/dt A FUNCTION OF STATE, OR OF THE CLOCK?")
    Vm = np.mean([V[r] for r in (1, 2, 3, 4)], axis=0)
    Dm = np.mean([D[r] for r in (1, 2, 3, 4)], axis=0)
    rows_g, rows_i, X_state, X_clock, yv = [], [], [], [], []
    for j in range(1, NI):
        lvl = Vm[j]
        prev = Dm[j - 1]
        hist = Vm[j] - Vm[0]
        for gidx in range(NG):
            rows_g.append(gidx); rows_i.append(j)
            X_state.append([lvl[gidx], prev[gidx], hist[gidx]])
            oh = np.zeros(NI); oh[j] = 1.0
            X_clock.append(oh)
            yv.append(Dm[j][gidx])
    rows_g = np.array(rows_g); X_state = np.array(X_state)
    X_clock = np.array(X_clock); yv = np.array(yv)
    X_both = np.hstack([X_state, X_clock])
    say(f"     {len(yv):,} gene-by-interval observations; held out BY GENE so no gene appears in "
        f"both train and test")
    sc = {"STATE": [], "CLOCK": [], "BOTH": []}
    for i in range(NSPLIT):
        rg = np.random.default_rng(SEED + i)
        gperm = rg.permutation(NG)
        test_g = set(gperm[: NG // 4].tolist())
        te = np.array([k for k, gg in enumerate(rows_g) if gg in test_g])
        tr = np.setdiff1d(np.arange(len(yv)), te)
        for nm, X in (("STATE", X_state), ("CLOCK", X_clock), ("BOTH", X_both)):
            sc[nm].append(abs(pear(yv[te], ridge(X[tr], yv[tr], X[te]))))
    for nm in sc:
        sc[nm] = np.array(sc[nm])
        say(f"       {nm:<6} held-out |r| {sc[nm].mean():.4f} +/- {sc[nm].std(ddof=1):.4f}")
    d2 = sc["STATE"] - sc["CLOCK"]
    se2 = d2.std(ddof=1) / np.sqrt(len(d2))
    say(f"     PAIRED STATE - CLOCK {d2.mean():+.4f} +/- {se2:.4f} "
        f"({d2.mean()/se2 if se2>0 else np.inf:+.1f} standard errors)")
    G.add("L2", bool(d2.mean() >= L2_BAR), stat=float(d2.mean()), requires=("L1",),
          if_true=lambda: f"L2 PASS -- state beats the clock by {d2.mean():+.4f}; the dynamics "
                          f"are autonomous in these coordinates",
          if_false=lambda: f"L2 FAIL -- state beats the clock by only {d2.mean():+.4f} against a "
                           f"{L2_BAR:.2f} bar; in these coordinates elapsed time describes the "
                           f"change at least as well as position does")
    res["autonomy"] = {k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
                       for k, v in sc.items()}
    res["autonomy"]["paired_state_minus_clock"] = float(d2.mean())

    # ---------------------------------------------------------------- L3
    say("L3 IS F STATIONARY, OR DOES theta ADAPT?")
    early = np.isin(rows_i, [1, 2, 3])
    late = np.isin(rows_i, [4, 5, 6, 7])
    ordered, random_ = [], []
    for i in range(NSPLIT):
        rg = np.random.default_rng(SEED + 500 + i)
        gp2 = rg.permutation(NG); tg = set(gp2[: NG // 4].tolist())
        gm = np.array([gg in tg for gg in rows_g])
        tr_o = np.where(early & ~gm)[0]; te_o = np.where(late & gm)[0]
        ordered.append(abs(pear(yv[te_o], ridge(X_both[tr_o], yv[tr_o], X_both[te_o]))))
        iv = rg.permutation(np.arange(1, NI))
        ha, hb = set(iv[: len(iv) // 2].tolist()), set(iv[len(iv) // 2:].tolist())
        ma = np.isin(rows_i, list(ha)); mb = np.isin(rows_i, list(hb))
        tr_r = np.where(ma & ~gm)[0]; te_r = np.where(mb & gm)[0]
        random_.append(abs(pear(yv[te_r], ridge(X_both[tr_r], yv[tr_r], X_both[te_r]))))
    ordered, random_ = np.array(ordered), np.array(random_)
    d3 = random_ - ordered
    se3 = d3.std(ddof=1) / np.sqrt(len(d3))
    z3 = d3.mean() / se3 if se3 > 0 else np.inf
    say(f"     fit early intervals, test late:      {ordered.mean():.4f} +/- "
        f"{ordered.std(ddof=1):.4f}")
    say(f"     fit random half, test the other:     {random_.mean():.4f} +/- "
        f"{random_.std(ddof=1):.4f}")
    say(f"     PAIRED random - ordered {d3.mean():+.4f} +/- {se3:.4f} ({z3:+.1f} standard errors)")
    say("     direction is REPORTED, not gated: positive means theta adapts, zero means a static")
    say("     F is sufficient")
    G.add("L3", bool(abs(z3) > 2.0), stat=float(z3), requires=("L1",),
          if_true=lambda: f"L3 PASS -- the two splits differ by {abs(z3):.1f} standard errors; "
                          f"{'the transition is NON-stationary, theta adapts' if d3.mean() > 0 else 'the time-ordered split is BETTER, which no adaptive account predicts'}",
          if_false=lambda: f"L3 FAIL -- {abs(z3):.1f} standard errors; stationary and adaptive "
                           f"accounts are indistinguishable here")
    res["stationarity"] = {"ordered": float(ordered.mean()), "random": float(random_.mean()),
                           "delta": float(d3.mean()), "se": float(se3), "z": float(z3)}

    # ---------------------------------------------------------------- L4
    say("L4 IS THE FLOW GRADIENT-LIKE?")
    Xc = Vm - Vm.mean(0, keepdims=True)
    U, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:KPC].T
    dP = np.array([P[j] - P[j - 1] for j in range(1, len(g))])
    Pm = P[:-1]
    A = np.hstack([Pm, np.ones((len(Pm), 1))])
    cond = float(np.linalg.cond(A.T @ A))
    J = np.linalg.solve(A.T @ A + 1e-8 * np.eye(A.shape[1]), A.T @ dP)[:KPC].T
    Jsym = 0.5 * (J + J.T); Janti = 0.5 * (J - J.T)
    curl = float(np.linalg.norm(Janti) / (np.linalg.norm(J) + 1e-12))
    say(f"     state projected onto {KPC} leading components, capturing "
        f"{np.sum(S_[:KPC]**2)/np.sum(S_**2):.1%} of trajectory variance")
    say(f"     linear map J from state to derivative, condition number {cond:.2e}")
    say(f"     ||J_antisymmetric|| / ||J|| = {curl:.4f}")
    say("     a pure -grad U flow has a SYMMETRIC Jacobian and a curl fraction of 0")
    say("     REPORTED, not gated: there is no principled bar and inventing one would be the")
    say("     same error as gating on a direction")
    G.add("L4", bool(np.isfinite(cond) and cond < 1e6), stat=float(cond),
          if_true=lambda: f"L4 PASS -- J is estimable (condition {cond:.1e}), so the "
                          f"decomposition is meaningful; curl fraction {curl:.3f}",
          if_false=lambda: f"L4 FAIL -- condition number {cond:.1e}; J is not estimable and the "
                           f"curl fraction would be an artefact of the inversion")
    res["geometry"] = {"curl_fraction": curl, "cond": cond,
                       "var_captured": float(np.sum(S_[:KPC] ** 2) / np.sum(S_ ** 2))}

    # ---------------------------------------------------------------- L5
    say("L5 CONTROL: DOES ANY OF THIS SURVIVE SHUFFLING?")
    best = max(sc, key=lambda k: sc[k].mean())
    Xb = {"STATE": X_state, "CLOCK": X_clock, "BOTH": X_both}[best]
    real, shuf = [], []
    for i in range(NSPLIT):
        rg = np.random.default_rng(SEED + 700 + i)
        gp3 = rg.permutation(NG); tg = set(gp3[: NG // 4].tolist())
        te = np.array([k for k, gg in enumerate(rows_g) if gg in tg])
        tr = np.setdiff1d(np.arange(len(yv)), te)
        real.append(abs(pear(yv[te], ridge(Xb[tr], yv[tr], Xb[te]))))
        gmap = rg.permutation(NG)
        ysh = yv.copy()
        idx = np.argsort(rows_g, kind="stable")
        ysh = yv[idx][np.argsort(np.argsort(gmap[rows_g[idx]], kind="stable"), kind="stable")]
        shuf.append(abs(pear(ysh[te], ridge(Xb[tr], ysh[tr], Xb[te]))))
    real, shuf = np.array(real), np.array(shuf)
    marg = real - shuf
    say(f"     best arm {best}: real {real.mean():.4f}, gene-shuffled {shuf.mean():.4f}")
    say(f"     per-split margin: min {marg.min():+.4f}, mean {marg.mean():+.4f}")
    G.add("L5", bool(marg.min() >= CTRL_BAR), stat=float(marg.min()), requires=("L1",),
          if_true=lambda: f"L5 PASS -- margin at least {marg.min():.4f} on every split",
          if_false=lambda: f"L5 FAIL -- worst-split margin {marg.min():+.4f}")
    res["control"] = {"arm": best, "real": float(real.mean()), "shuffled": float(shuf.mean()),
                      "min_margin": float(marg.min())}

    # ---------------------------------------------------------------- L6
    say("L6 WHAT THIS CANNOT SHOW")
    say("     A negative on L2 is about THESE coordinates -- 1,336 genes' log2 levels -- not about")
    say("     the framework. The state the theory names includes protein, PTM, metabolite,")
    say("     localisation and mechanics, none of which this series measures. Failing to find an")
    say("     autonomous field in a projection is EXPECTED if the projection omits the variables")
    say("     that make it autonomous, and that is the most likely reading of a FAIL.")
    say("     L4's Jacobian is LINEAR. A curl-free nonlinear flow can look rotational under a")
    say("     linear fit, so the antisymmetric part bounds how well a linear potential describes")
    say("     the flow, not how well any potential does.")
    say("     L3 cannot separate theta adapting from the environment changing: dexamethasone's")
    say("     effect is itself time-varying, so a non-stationary F is consistent with a static")
    say("     controller driven by a changing input.")

    res["gates"] = {k: (v == "PASS") for k, v in G.status.items()}
    res["void"] = [k for k, v in G.status.items() if v == "VOID"]
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    G.summary()
    Path("outputs").mkdir(exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=1, default=float))
    say(f"     written {OUT}")


if __name__ == "__main__":
    main()
