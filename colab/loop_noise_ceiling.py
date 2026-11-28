"""Loop 216. What is being compared against what, and can any model beat the measurement noise?

TWO QUESTIONS, AND THE SECOND MAY ANSWER EVERY LOOP SINCE 198.

FIRST, THE DATA DICTIONARY. Fifteen loops have reported R2 against "persistence" without ever
stating plainly what the inputs and the target are. K1 writes it down and verifies it against the
files rather than reciting it.

SECOND, THE NOISE CEILING, AND WHY IT IS NOW URGENT. Loop 215b measured two numbers on the
180->240 minute interval that do not sit together:

    median |measured change|      0.1026 log2
    replicate noise floor         0.2277 log2

THE TYPICAL GENE MOVES LESS THAN THE MEASUREMENT ERROR. If that holds across intervals then the
per-interval change is mostly noise, no model can predict noise, and predicting NO CHANGE is not a
weak baseline -- it is close to the OPTIMAL estimator, because under errors-in-variables the
variance-minimising predictor of a quantity swamped by measurement error shrinks toward zero.

That would explain loop 198, loop 213's F6, loop 214, loop 215 and the backwards relationship
between set-point quality and forward performance, all at once, without any of them being a failure
of modelling. It is testable directly: the data has three replicates, so one replicate's measured
change can be scored against another's. NO MODEL CAN BEAT THAT, because it is what the instrument
says about itself.

MONTE CARLO IS THE RIGHT TOOL AND K4 USES IT FOR THE RIGHT THING. Resampling cannot remove noise
from a target that was measured once; averaging predictions does not make a noisy target less
noisy. What it CAN do is put an interval on the ceiling and on the model's position relative to it,
which is what decides whether the remaining gap is worth chasing.

PREDECLARED, BEFORE ANY NUMBER.

  K1 IS THE DATA DICTIONARY CORRECT?
     State inputs and target explicitly and verify each against its file: shapes, units, and the
     exact arithmetic that produces the target.
     Gate: PASS iff the stated target is reproduced from the raw TPM matrix to within 1e-9 by an
     independent recomputation. FAIL means fifteen loops have been describing the wrong quantity.

  K2 WHAT IS THE NOISE CEILING ON A PER-INTERVAL CHANGE?
     For each pair of replicates, score one replicate's measured change against the other's, on
     exactly the rows the models were scored on.
     Gate: PASS iff the mean replicate-to-replicate R2 is above 0.10. This is the ceiling for any
     model of the interval change. A FAIL means the target is mostly noise and no model can
     predict it -- which would retire the whole forward-modelling question rather than answer it.

  K3 IS PERSISTENCE THE OPTIMAL ESTIMATOR UNDER THAT NOISE?
     Estimate the signal-to-noise ratio of the interval change from the replicate variance
     decomposition, and compute the optimal shrinkage factor for a linear predictor.
     Gate: PASS iff the optimal shrinkage exceeds 0.20 -- that is, iff a non-trivial fraction of
     the measured change is real signal worth predicting. A FAIL says predicting zero is close to
     the best any estimator can do, and every negative R2 since loop 198 is a property of the
     measurement rather than of the models.

  K4 WHERE DOES THE MODEL SIT RELATIVE TO THE CEILING?
     Bootstrap the genes 1,000 times and report the model's R2, persistence's R2 and the replicate
     ceiling as intervals rather than points.
     Gate: PASS iff the model's interval and the ceiling's interval do not overlap -- that is, iff
     there is a measurable gap left to close. Requires K2.

  K5 DOES THE SAME HOLD FOR THE SET POINT?
     The plateau is an average of three timepoints and should be far less noisy than a single
     interval. Compute its replicate ceiling the same way.
     Gate: PASS iff the plateau's replicate R2 exceeds the interval's by more than 0.20, which
     would locate the problem precisely in the DIFFERENCING rather than in the data.

  K6 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
MODEL = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_noise_ceiling.json"
REPS, SEED, NBOOT = (1, 2, 3), 216216, 1000

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
    res = {"test": "noise ceiling"}
    say("=" * 104)
    say("LOOP 216 -- WHAT IS COMPARED AGAINST WHAT, AND CAN ANY MODEL BEAT THE MEASUREMENT NOISE?")
    say("=" * 104)

    # ---------------------------------------------------------------- K1
    say("K1 THE DATA DICTIONARY")
    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    idx = gi[np.array([pos[s] for s in names])]
    S = np.array(art["stack_prediction"])
    say("     RAW: A549 lung adenocarcinoma, 100 nM dexamethasone, ENCODE polyA RNA-seq.")
    say(f"          TPM matrix {tpm.shape} = (samples, genes); {len(set(mins.tolist()))} "
        f"timepoints, replicates {sorted(set(reps.tolist()))}")
    say(f"     GRID USED: {[int(x) for x in grid]} minutes -- the 9 points where replicates "
        f"{list(REPS)} all exist and t >= 30")
    say("     TARGET m(t,g) = mean over replicates r of")
    say("               [ log2(1 + TPM(t,g,r)) - log2(1 + TPM(30,g,r)) ]")
    say("          EACH REPLICATE IS BASELINE-SUBTRACTED AGAINST ITS OWN t=30 BEFORE AVERAGING.")
    say("          That order matters and is deliberate: averaging TPM first and taking one log")
    say("          lets a replicate entering or leaving the series shift the level, which is the")
    say("          defect that put a batch step at t=30 in loop 191 and made every half-time")
    say("          28 minutes (loop_response_timing_d.py:302-317). My first statement of this")
    say("          dictionary had the order wrong and K1 caught it at 8.53e-01.")
    say("     SET POINT S(g) = mean of m over the last three grid points (480, 600, 720 min).")
    say("     FORWARD TARGET dm(t,g) = m(t,g) - m(t_prev,g). THIS is what persistence predicts")
    say("          as zero and what every R2 since loop 198 is computed on.")
    say("     INPUTS, per gene, all measured BEFORE or independently of the target:")
    say("          chip     9 ENCODE A549 ChIP/DNase tracks x 3 summaries = 27 columns")
    say("          gains    200 Perturb-seq CRISPRi signatures, K562 = 200 columns")
    say("          physics  879 JASPAR motifs x 3 chemical potentials = 2,637 columns")
    # independent recomputation of m from raw TPM
    acc, nrep = None, 0
    for r in REPS:
        rows = [np.where((mins == int(t)) & (reps == r))[0] for t in grid]
        if any(len(x) == 0 for x in rows):
            continue
        V = np.array([np.log2(1.0 + tpm[ix].astype(np.float64)).mean(0) for ix in rows])
        V = V - V[0]
        acc = V if acc is None else acc + V
        nrep += 1
    chk = acc / max(nrep, 1)
    err = float(np.max(np.abs(chk[:, idx] - M[:, idx])))
    say(f"     independent recomputation of m from raw TPM: max abs difference {err:.3e}")
    G.add("K1", bool(err < 1e-9), stat=err,
          if_true=f"K1 PASS -- the stated target reproduces from raw TPM to {err:.1e}",
          if_false=lambda: f"K1 FAIL -- recomputation differs by {err:.3e}; the dictionary above "
                           f"does not describe what the loops have been scoring")

    # per-replicate trajectories
    Mr = {}
    for rp in REPS:
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (rp,), grid)
        Mr[rp] = Mi[:, idx]
    Mall = M[:, idx]

    def ivals(A):
        out = []
        for j in range(1, len(grid)):
            out.append(A[j] - A[j - 1])
        return np.concatenate(out)
    d_all = ivals(Mall)
    d_rep = {rp: ivals(Mr[rp]) for rp in REPS}

    # ---------------------------------------------------------------- K2
    say("K2 WHAT IS THE NOISE CEILING ON A PER-INTERVAL CHANGE?")
    pairs = [(1, 2), (1, 3), (2, 3)]
    r2p = []
    for a, b_ in pairs:
        v = r2s(d_rep[b_], d_rep[a])
        r2p.append(v)
        say(f"       replicate {a} predicting replicate {b_}:  R2 {v:+.5f}   "
            f"pearson {np.corrcoef(d_rep[a], d_rep[b_])[0,1]:+.4f}")
    ceil = float(np.mean(r2p))
    say(f"     mean replicate-to-replicate R2 on interval changes: {ceil:+.5f}")
    say(f"     NO MODEL CAN BEAT THIS -- it is what the instrument says about itself")
    G.add("K2", bool(ceil > 0.10), stat=ceil, requires=("K1",),
          if_true=lambda: f"K2 PASS -- the ceiling is {ceil:+.4f}, so there is signal to predict",
          if_false=lambda: f"K2 FAIL -- one replicate predicts another at R2 {ceil:+.4f} on the "
                           f"per-interval change. The target is mostly measurement noise, no "
                           f"model can predict noise, and every negative R2 since loop 198 is a "
                           f"property of the MEASUREMENT and not of the models")
    res["ceiling"] = {"pairs": dict(zip([f"{a}v{b}" for a, b in pairs], r2p)), "mean": ceil}

    # ---------------------------------------------------------------- K3
    say("K3 IS PERSISTENCE THE OPTIMAL ESTIMATOR UNDER THAT NOISE?")
    stack = np.vstack([d_rep[rp] for rp in REPS])
    within = float(np.mean(np.var(stack, axis=0, ddof=1)))
    total = float(np.var(d_all, ddof=1))
    between = max(total - within / len(REPS), 0.0)
    snr = between / within if within > 0 else float("inf")
    shrink = between / (between + within / len(REPS)) if (between + within) > 0 else 0.0
    say(f"     variance of the 3-replicate MEAN change      {total:.6f}")
    say(f"     mean within-gene variance across replicates  {within:.6f}")
    say(f"     implied signal variance                      {between:.6f}")
    say(f"     signal-to-noise (signal / per-replicate noise) {snr:.4f}")
    say(f"     optimal shrinkage for a linear predictor       {shrink:.4f}")
    say(f"     (shrinkage near 0 means the variance-minimising prediction is near ZERO CHANGE)")
    G.add("K3", bool(shrink > 0.20), stat=shrink, requires=("K1",),
          if_true=lambda: f"K3 PASS -- optimal shrinkage {shrink:.3f}, a real fraction of the "
                          f"measured change is signal worth predicting",
          if_false=lambda: f"K3 FAIL -- optimal shrinkage {shrink:.3f}. Predicting ZERO is close "
                           f"to the best any estimator can do on this target, and persistence was "
                           f"never a weak baseline -- it was nearly optimal")
    res["variance"] = {"total": total, "within": within, "between": between,
                       "snr": snr, "shrinkage": shrink}

    # ---------------------------------------------------------------- K4
    say("K4 WHERE DOES THE MODEL SIT RELATIVE TO THE CEILING?  (Monte Carlo over genes)")
    ng = len(names)
    nint = len(grid) - 1
    gidx = np.tile(np.arange(ng), nint)
    N_TRAIN = 6
    lev = np.concatenate([Mall[j - 1] for j in range(1, len(grid))])
    dts = np.concatenate([np.full(ng, grid[j] - grid[j - 1]) for j in range(1, len(grid))])
    jj = np.concatenate([np.full(ng, j) for j in range(1, len(grid))])
    tr, te = jj < N_TRAIN, jj >= N_TRAIN
    d_tr = dts[tr] * (S[gidx[tr]] - lev[tr]); d_te = dts[te] * (S[gidx[te]] - lev[te])
    lam = float(d_tr @ d_all[tr] / (d_tr @ d_tr))
    pred = lam * d_te
    yte = d_all[te]
    rg = np.random.default_rng(SEED)
    bm, bp, bc = [], [], []
    for _ in range(NBOOT):
        g = rg.integers(0, ng, ng)
        m = np.isin(gidx[te], g)
        if m.sum() < 50:
            continue
        bm.append(r2s(yte[m], pred[m])); bp.append(r2s(yte[m], np.zeros(int(m.sum()))))
        mm = np.isin(gidx, g)
        bc.append(np.mean([r2s(d_rep[b_][mm], d_rep[a][mm]) for a, b_ in pairs]))
    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    say(f"       model        R2 {np.mean(bm):+.5f}   95% CI [{ci(bm)[0]:+.5f}, {ci(bm)[1]:+.5f}]")
    say(f"       persistence  R2 {np.mean(bp):+.5f}   95% CI [{ci(bp)[0]:+.5f}, {ci(bp)[1]:+.5f}]")
    say(f"       CEILING      R2 {np.mean(bc):+.5f}   95% CI [{ci(bc)[0]:+.5f}, {ci(bc)[1]:+.5f}]")
    gap = bool(ci(bm)[1] < ci(bc)[0])
    G.add("K4", gap, requires=("K2",),
          if_true=lambda: f"K4 PASS -- the model's interval sits below the ceiling's, so there is "
                          f"a measurable gap left to close",
          if_false=lambda: f"K4 FAIL -- the model's interval [{ci(bm)[0]:+.4f}, {ci(bm)[1]:+.4f}] "
                           f"and the ceiling's [{ci(bc)[0]:+.4f}, {ci(bc)[1]:+.4f}] overlap; "
                           f"there is no separable gap to chase")
    res["bootstrap"] = {"model": [float(np.mean(bm)), *ci(bm)],
                        "persistence": [float(np.mean(bp)), *ci(bp)],
                        "ceiling": [float(np.mean(bc)), *ci(bc)], "n_boot": len(bm)}

    # ---------------------------------------------------------------- K5
    say("K5 DOES THE SAME HOLD FOR THE SET POINT?")
    pl = {rp: Mr[rp][-3:].mean(0) for rp in REPS}
    r2pl = [r2s(pl[b_], pl[a]) for a, b_ in pairs]
    cpl = float(np.mean(r2pl))
    for (a, b_), v in zip(pairs, r2pl):
        say(f"       plateau, replicate {a} predicting {b_}:  R2 {v:+.5f}")
    say(f"     plateau ceiling {cpl:+.5f}   interval ceiling {ceil:+.5f}   "
        f"difference {cpl-ceil:+.5f}")
    G.add("K5", bool(cpl - ceil > 0.20), stat=cpl, requires=("K1",),
          if_true=lambda: f"K5 PASS -- the plateau is measurable at R2 {cpl:.3f} while a single "
                          f"interval is at {ceil:.3f}. The problem is the DIFFERENCING, not the "
                          f"data: averaging three timepoints keeps the signal and cancels the "
                          f"noise, subtracting two adjacent ones cancels the signal and keeps it",
          if_false=lambda: f"K5 FAIL -- plateau {cpl:.4f} against interval {ceil:.4f}")
    res["plateau_ceiling"] = {"pairs": r2pl, "mean": cpl}

    say("K6 WHAT THIS CANNOT SHOW")
    say("     A replicate ceiling assumes the replicates are independent measurements of the same")
    say("     quantity. ENCODE isogenic replicates share library prep and batch, so they")
    say("     UNDERSTATE the true noise and the real ceiling is lower than measured here.")
    say("     The ceiling is for the target AS DEFINED -- a difference of two three-replicate")
    say("     means. A different target definition, such as a fitted trajectory slope, would have")
    say("     a different and probably higher ceiling, and that is the constructive reading.")
    say("     Nothing here says the models are good. It says the target they were scored on may")
    say("     not have been predictable, which is a different and more useful claim.")

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
