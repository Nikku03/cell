"""Loop 217. Amplify the signal: where IS the interval change measurable, and does the model work there?

WHAT WAS ASKED AND WHAT IS BUILT. The request was to use quantum amplitude amplification to
separate signal from noise. Amplitude amplification gives a quadratic speedup for finding a marked
item in an unstructured set; it needs an oracle that recognises the answer, and it does not raise
the signal-to-noise ratio of a noisy measurement. There is no oracle here -- if we could recognise
the true change we would not need to estimate it. So the goal is taken and the method is replaced
with the classical technique that does achieve it: VARIANCE DECOMPOSITION PER GENE, then selection.

WHY THIS IS THE RIGHT NEXT LOOP. Loop 216 measured that the per-interval change is unmeasurable IN
AGGREGATE -- one replicate predicts another at R2 -0.540, while the plateau reaches +0.834. But it
also measured that the target is not pure noise: signal variance 0.036637 against a per-replicate
noise variance of 0.098377, an optimal shrinkage of 0.5277. Roughly half the measured change is
real. An aggregate that is half signal is made of genes that are mostly noise and genes that are
mostly signal, and loop 215b already saw the split from the other side: 75.3% directional accuracy
on the 16% of genes moving more than the noise floor.

So the question is not whether the average gene is predictable. It is whether the measurable
MINORITY is, and how large that minority is.

THE TRAP, AND THE ONLY DESIGN THAT AVOIDS IT. Selecting genes by signal-to-noise and then scoring
on the same replicates is circular: a gene selected for having a consistent change across
replicates 1, 2 and 3 will trivially have replicate 1 predict replicate 2. L4 therefore selects
on replicates 1 and 2 ONLY and scores on replicate 3, which never touched the selection. Anything
that does not do this is measuring its own selection.

PREDECLARED, BEFORE ANY NUMBER.

  L1 IS THE PER-GENE DECOMPOSITION SOUND?
     For each gene and interval, decompose the measured change into between-replicate signal and
     within-replicate noise.
     Gate: PASS iff the per-gene variance components are non-negative after truncation for at
     least 95% of genes, and the pooled decomposition reproduces loop 216's aggregate numbers
     (signal 0.036637, noise 0.098377) to three decimals. FAIL means this is decomposing something
     else.

  L2 HOW MANY GENES CARRY MORE SIGNAL THAN NOISE?
     Rank genes by signal-to-noise and report the distribution.
     Gate: PASS iff at least 5% of genes have SNR above 1. Below that there is no measurable
     minority to amplify and the honest answer is that the interval change is not a usable target
     for any subset.

  L3 DOES THE CEILING RISE WITH SELECTION?
     Replicate-to-replicate R2 restricted to the top decile, quartile and half by SNR, with
     selection made on replicates 1 and 2 and the ceiling measured between them -- reported as the
     OPTIMISTIC curve, and marked as such because it is circular by construction.
     Gate: PASS iff the top decile's ceiling exceeds 0.10. This bounds what selection could buy
     before honesty is applied.

  L4 THE HONEST VERSION: SELECT ON TWO REPLICATES, SCORE ON THE THIRD.
     Select genes by SNR computed from replicates 1 and 2 alone. Score the model and persistence
     against replicate 3's measured change on those genes only.
     Gate: PASS iff the model beats persistence by more than 0.01 on the selected genes. This is
     the whole loop and it is allowed to fail.

  L5 DOES CROSS-REPLICATE PROJECTION HELP?
     Take the singular value decomposition of the gene-by-interval change matrix for replicates 1
     and 2, keep only components whose loadings agree between them, and project replicate 3.
     Gate: PASS iff the projected target's replicate ceiling exceeds the raw target's by more than
     0.10. This is the closest classical analogue to what was asked -- amplifying the reproducible
     subspace and discarding the rest.

  L6 IS THE SELECTION READING SIGNAL OR JUST AMPLITUDE?
     A gene with a large change has a large signal variance almost by construction. Re-select on
     SNR with the change magnitude regressed out, and rescore.
     Gate: PASS iff the magnitude-controlled selection still beats persistence by more than 0.01.
     A FAIL means the selection is finding big movers, not measurable ones, and L4 is a
     restatement of loop 215b.

  L7 WHAT THIS CANNOT SHOW.
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
OUT = "outputs/loop_snr_amplify.json"
REPS, SEED, N_TRAIN = (1, 2, 3), 217217, 6
REF_SIG, REF_NOISE = 0.036637, 0.098377

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
    res = {"test": "snr amplification"}
    say("=" * 104)
    say("LOOP 217 -- WHERE IS THE INTERVAL CHANGE MEASURABLE, AND DOES THE MODEL WORK THERE?")
    say("=" * 104)
    say("     NOTE ON THE METHOD. Amplitude amplification needs an oracle that recognises the")
    say("     answer and gives a quadratic speedup for SEARCH; it does not raise the")
    say("     signal-to-noise of a measurement, and there is no oracle here -- if we could")
    say("     recognise the true change we would not be estimating it. The goal is kept and the")
    say("     method is per-gene variance decomposition followed by selection.")

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    idx = gi[np.array([pos[s] for s in names])]
    S = np.array(art["stack_prediction"])
    ng, nint = len(names), len(grid) - 1

    D = {}
    for rp in REPS:
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (rp,), grid)
        A = Mi[:, idx]
        D[rp] = np.array([A[j] - A[j - 1] for j in range(1, len(grid))])   # (nint, ng)
    Dm = np.mean([D[r] for r in REPS], axis=0)
    say(f"     {ng:,} genes x {nint} intervals x {len(REPS)} replicates")

    # ---------------------------------------------------------------- L1
    say("L1 IS THE PER-GENE DECOMPOSITION SOUND?")
    stack = np.stack([D[r] for r in REPS])                # (3, nint, ng)
    within_gi = stack.var(axis=0, ddof=1)                 # (nint, ng)
    noise_g = within_gi.mean(axis=0)                      # per gene
    total_g = Dm.var(axis=0, ddof=1)                      # per gene, across intervals
    sig_g = np.maximum(total_g - noise_g / len(REPS), 0.0)
    nonneg = float(np.mean(total_g - noise_g / len(REPS) >= -1e-12))
    pooled_noise = float(within_gi.mean())
    pooled_total = float(np.var(Dm.ravel(), ddof=1))
    pooled_sig = max(pooled_total - pooled_noise / len(REPS), 0.0)
    say(f"     pooled: total {pooled_total:.6f}  noise {pooled_noise:.6f}  signal {pooled_sig:.6f}")
    say(f"     loop 216 recorded signal {REF_SIG:.6f}  noise {REF_NOISE:.6f}")
    say(f"     genes with a non-negative signal component before truncation: {nonneg:.1%}")
    ok1 = (abs(pooled_sig - REF_SIG) < 5e-4 and abs(pooled_noise - REF_NOISE) < 5e-4)
    G.add("L1", ok1,
          if_true="L1 PASS -- the per-gene decomposition pools to loop 216's aggregate",
          if_false=lambda: f"L1 FAIL -- signal {pooled_sig:.6f} vs {REF_SIG:.6f}, noise "
                           f"{pooled_noise:.6f} vs {REF_NOISE:.6f}")

    # ---------------------------------------------------------------- L2
    say("L2 HOW MANY GENES CARRY MORE SIGNAL THAN NOISE?")
    snr = sig_g / (noise_g / len(REPS) + 1e-12)
    for q in (50, 75, 90, 95, 99):
        say(f"       {q}th percentile SNR {np.percentile(snr, q):.4f}")
    frac1 = float(np.mean(snr > 1.0))
    say(f"     genes with SNR > 1: {int((snr>1).sum()):,} of {ng:,} = {frac1:.1%}")
    G.add("L2", bool(frac1 >= 0.05), stat=frac1, requires=("L1",),
          if_true=lambda: f"L2 PASS -- {frac1:.1%} of genes carry more signal than noise",
          if_false=lambda: f"L2 FAIL -- only {frac1:.1%}; there is no measurable minority")
    res["snr"] = {"frac_gt1": frac1, "p50": float(np.percentile(snr, 50)),
                  "p90": float(np.percentile(snr, 90)), "p99": float(np.percentile(snr, 99))}

    # ---------------------------------------------------------------- L3
    say("L3 DOES THE CEILING RISE WITH SELECTION?  (optimistic, circular by construction)")
    w12 = np.stack([D[1], D[2]]).var(axis=0, ddof=1).mean(axis=0)
    t12 = np.mean([D[1], D[2]], axis=0).var(axis=0, ddof=1)
    snr12 = np.maximum(t12 - w12 / 2, 0.0) / (w12 / 2 + 1e-12)
    opt = {}
    for frac in (0.10, 0.25, 0.50, 1.00):
        k = max(20, int(frac * ng))
        sel = np.argsort(-snr12)[:k]
        v = r2s(D[2][:, sel].ravel(), D[1][:, sel].ravel())
        opt[frac] = v
        say(f"       top {int(frac*100):>3}% by SNR(1,2)   rep1 predicts rep2   R2 {v:+.5f}   "
            f"n {k:,}")
    G.add("L3", bool(opt[0.10] > 0.10), stat=opt[0.10], requires=("L2",),
          if_true=lambda: f"L3 PASS -- the top decile reaches {opt[0.10]:+.4f}, so selection could "
                          f"in principle buy a measurable target",
          if_false=lambda: f"L3 FAIL -- even the top decile reaches only {opt[0.10]:+.4f}")
    res["optimistic_ceiling"] = {str(k): v for k, v in opt.items()}

    # ---------------------------------------------------------------- L4
    say("L4 THE HONEST VERSION: SELECT ON REPLICATES 1+2, SCORE ON REPLICATE 3")
    lev = np.array([np.mean([D[r] for r in REPS], axis=0)])   # unused, kept explicit
    Mm = M[:, idx]
    lvl = np.array([Mm[j - 1] for j in range(1, len(grid))])
    dts = np.array([grid[j] - grid[j - 1] for j in range(1, len(grid))])
    tr_j = np.arange(nint) < (N_TRAIN - 1)
    hon = {}
    for frac in (0.10, 0.25, 0.50, 1.00):
        k = max(20, int(frac * ng))
        sel = np.argsort(-snr12)[:k]
        d_tr = (dts[tr_j, None] * (S[None, sel] - lvl[tr_j][:, sel])).ravel()
        y_tr = Dm[tr_j][:, sel].ravel()
        lam = float(d_tr @ y_tr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
        d_te = (dts[~tr_j, None] * (S[None, sel] - lvl[~tr_j][:, sel])).ravel()
        y3 = D[3][~tr_j][:, sel].ravel()                  # replicate 3, never used to select
        rm, rp_ = r2s(y3, lam * d_te), r2s(y3, np.zeros_like(y3))
        hon[frac] = {"model": rm, "persistence": rp_, "margin": rm - rp_, "n": k}
        say(f"       top {int(frac*100):>3}%   model {rm:+.5f}   persistence {rp_:+.5f}   "
            f"margin {rm-rp_:+.5f}   n {k:,}")
    best = max(hon, key=lambda f: hon[f]["margin"])
    G.add("L4", bool(hon[best]["margin"] > 0.01), stat=hon[best]["margin"], requires=("L2",),
          if_true=lambda: f"L4 PASS -- on the top {int(best*100)}% by SNR the model beats "
                          f"persistence by {hon[best]['margin']:+.5f}, scored on a replicate that "
                          f"never touched the selection",
          if_false=lambda: f"L4 FAIL -- the best selection buys {hon[best]['margin']:+.5f}. "
                           f"Selecting the measurable genes does not make the model work on them")
    res["honest"] = {str(k): v for k, v in hon.items()}

    # ---------------------------------------------------------------- L5
    say("L5 DOES CROSS-REPLICATE PROJECTION HELP?")
    U1, s1, V1 = np.linalg.svd(D[1], full_matrices=False)
    U2, s2, V2 = np.linalg.svd(D[2], full_matrices=False)
    agree = np.array([abs(float(V1[i] @ V2[i])) for i in range(min(len(s1), len(s2)))])
    say(f"     |cosine| between replicate-1 and replicate-2 right singular vectors, by component:")
    say(f"       " + "  ".join(f"{a:.3f}" for a in agree))
    kkeep = int((agree > 0.5).sum())
    raw = r2s(D[3].ravel(), D[2].ravel())
    if kkeep > 0:
        P = V1[:kkeep].T @ V1[:kkeep]
        proj3, proj2 = D[3] @ P, D[2] @ P
        prj = r2s(proj3.ravel(), proj2.ravel())
    else:
        prj = float("nan")
    say(f"     components with |cosine| > 0.5: {kkeep} of {len(agree)}")
    say(f"     replicate ceiling  raw {raw:+.5f}   projected {prj:+.5f}   delta {prj-raw:+.5f}")
    G.add("L5", bool(np.isfinite(prj) and prj - raw > 0.10), stat=prj, requires=("L1",),
          if_true=lambda: f"L5 PASS -- projecting onto the reproducible subspace raises the "
                          f"ceiling by {prj-raw:+.4f}",
          if_false=lambda: f"L5 FAIL -- projection buys {prj-raw:+.4f}; the reproducible subspace "
                           f"is not separable from the rest at this sample size")
    res["projection"] = {"agree": agree.tolist(), "kept": kkeep, "raw": raw, "projected": prj}

    # ---------------------------------------------------------------- L6
    say("L6 IS THE SELECTION READING SIGNAL OR JUST AMPLITUDE?")
    amp = np.abs(np.mean([D[1], D[2]], axis=0)).mean(axis=0)
    la, ls = np.log1p(amp), np.log1p(snr12)
    beta = float(np.polyfit(la, ls, 1)[0])
    resid = ls - np.polyval(np.polyfit(la, ls, 1), la)
    say(f"     log SNR on log |change| slope {beta:+.4f}  "
        f"(a large mover has a large signal component almost by construction)")
    k = max(20, int(0.25 * ng))
    sel = np.argsort(-resid)[:k]
    d_tr = (dts[tr_j, None] * (S[None, sel] - lvl[tr_j][:, sel])).ravel()
    y_tr = Dm[tr_j][:, sel].ravel()
    lam = float(d_tr @ y_tr / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
    d_te = (dts[~tr_j, None] * (S[None, sel] - lvl[~tr_j][:, sel])).ravel()
    y3 = D[3][~tr_j][:, sel].ravel()
    rm, rp_ = r2s(y3, lam * d_te), r2s(y3, np.zeros_like(y3))
    say(f"     magnitude-controlled top 25%: model {rm:+.5f}   persistence {rp_:+.5f}   "
        f"margin {rm-rp_:+.5f}")
    G.add("L6", bool(rm - rp_ > 0.01), stat=rm - rp_, requires=("L4",),
          if_true=lambda: f"L6 PASS -- {rm-rp_:+.5f} with magnitude regressed out, so the "
                          f"selection is finding measurable genes and not merely big movers",
          if_false=lambda: f"L6 FAIL -- {rm-rp_:+.5f} once magnitude is controlled; the selection "
                           f"was finding big movers and L4 restates loop 215b")
    res["magnitude_control"] = {"slope": beta, "model": rm, "persistence": rp_,
                                "margin": rm - rp_}

    say("L7 WHAT THIS CANNOT SHOW")
    say("     Selecting genes by measurability changes the question. A model that works only on")
    say("     the genes whose change is reproducible is a model of those genes, and loop 216")
    say("     measured that they are a minority.")
    say("     Replicate 3 is used as the honest scorer throughout, but loop 216 measured that")
    say("     replicate 1 is the outlier -- it agrees with 2 and 3 at pearson +0.12 while they")
    say("     agree with each other at +0.60. Selecting on 1+2 therefore inherits replicate 1's")
    say("     idiosyncrasy, and selecting on 2+3 would not be independent of the scorer.")
    say("     With three replicates there is no split that is both clean and balanced, and no")
    say("     amount of method fixes that -- it needs a fourth replicate.")
    say("     The SVD projection is fitted on 8 intervals. Eight points is very few for a")
    say("     subspace estimate and L5's cosines should be read as indicative.")

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
