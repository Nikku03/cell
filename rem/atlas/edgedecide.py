"""The decider: does the channel tree beat the DEEP MLP on sample efficiency, or not?

WHERE sampleedge.py LEFT IT. The channel tree's sample-efficiency edge replicated on fresh seeds,
and against a structurally identical tanh tree it held on 5 of 5 seeds -- the channel dynamics do
work. But against the deep MLP, the best conventional network, the conservative per-seed analysis
gave 4 of 5 seeds and a sign-test p of 0.188, and the deep MLP ended 5 of its 30 runs at the
10,000-step cap without fitting its training set. Two weaknesses: too few independent units, and a
budget that may have handicapped the control.

THIS REMOVES BOTH, AND IS PREDECLARED IN FULL.
  TEN FRESH SEEDS, 20-29 -- disjoint from limits.py (0-2) and sampleedge.py (10-14).
  BUDGET 30,000 steps for both models, three times larger, so the deep MLP is not capped.
  PER-SEED CLUSTERING IS THE PRIMARY TEST FROM THE START, not a re-analysis afterwards.
  Same k = 12, same n grid 1500..4000, same protocol: disjoint train/test, shared parity subset,
  identical optimiser and early stopping.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

E1  THE BUDGET CONFOUND MUST BE GONE. Count runs of each model that end at the cap without fitting
    the training set. PREDECLARED: if the deep MLP still has capped unfitted runs, the verdict
    below is reported WITH that caveat attached, not as clean.

E0  THE DECISION. For each seed, the mean test accuracy over the six n values is one unit.
    PRIMARY: the channel tree beats the deep MLP if it is higher on at least 9 of 10 seeds
             (one-sided sign test, p = 11/1024 = 0.011). 8 of 10 is p = 0.055 and does NOT pass.
    SECONDARY: paired t over the 10 seed differences, one-sided, df 9, t > 1.833.
    The verdict is BEATS only if the primary passes. If the primary fails but the secondary passes,
    it is SUGGESTIVE, NOT ESTABLISHED. If both fail, NO ADVANTAGE OVER THE DEEP MLP.
    A deep-MLP win on 9 of 10 seeds would be reported as the deep MLP beating the channel tree.

E2  WHAT THIS IS AND IS NOT.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_edgedecide.txt")
ART = R("outputs", "edgedecide.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("limits", os.path.join(HERE, "limits.py"))
lim = importlib.util.module_from_spec(_s); _s.loader.exec_module(lim)
ch = lim.ch

K = 12
N_GRID = (1500, 2000, 2500, 3000, 3500, 4000)
SEEDS = tuple(range(20, 30))
MAXSTEPS = 30000
MODELS = ("channel", "mlpdeep")


def job(args):
    name, k, seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(k, seed, ntr)
    m = lim.build(name, np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps, t0 = {}, 0, time.time()
    while steps < MAXSTEPS:
        for _ in range(lim.CHECK):
            i = rng.integers(0, len(Xtr), lim.BS)
            z = m.forward(Xtr[i])
            ch.adam(m.ps, m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / lim.BS), st, lim.LR)
        steps += lim.CHECK
        if lim.acc(m, Xtr, ytr) == 1.0:
            break
    return dict(model=name, k=k, seed=seed, ntr=ntr, steps=steps,
                train=lim.acc(m, Xtr, ytr), test=lim.acc(m, Xte, yte),
                seconds=round(time.time() - t0, 1))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("THE DECIDER: CHANNEL TREE VERSUS THE DEEP MLP, TEN FRESH SEEDS, NO STEP CAP"); P_(RULE)
    P_(f"  k = {K}; n = {N_GRID}; seeds {SEEDS[0]}-{SEEDS[-1]}; budget {MAXSTEPS:,} steps for both")
    pool = mp.get_context("fork").Pool(4)
    runs = pool.map(job, [(m, K, s, n) for s in SEEDS for n in N_GRID for m in MODELS])
    pool.close()

    # ---- E1 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("E1  IS THE BUDGET CONFOUND GONE?"); P_(RULE)
    capped = {}
    for m in MODELS:
        rs = [r for r in runs if r["model"] == m]
        capped[m] = sum(1 for r in rs if r["steps"] >= MAXSTEPS and r["train"] < 1.0)
        P_(f"  {m:<8} runs ending at the cap unfitted: {capped[m]}/{len(rs)}"
           f"   max steps used {max(r['steps'] for r in rs):,}")
    clean = capped["mlpdeep"] == 0
    P_(f"\n  E1: {'PASS -- no deep-MLP run was cut off before fitting' if clean else 'NOT CLEAN -- the verdict carries this caveat'}")

    # ---- per n, for reading -------------------------------------------------------------------
    P_("\n  mean test accuracy over 10 seeds [min - max]")
    P_(f"    {'n':>5} {'channel tree':>24} {'MLP deep':>24}")
    for n in N_GRID:
        row = f"    {n:>5} "
        for m in MODELS:
            v = [r["test"] for r in runs if r["model"] == m and r["ntr"] == n]
            row += f"{np.mean(v):>12.3f} [{min(v):.2f}-{max(v):.2f}]"
        P_(row)

    # ---- E0 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("E0  THE DECISION, ONE UNIT PER SEED"); P_(RULE)
    unit = lambda m, s: float(np.mean([r["test"] for r in runs if r["model"] == m and r["seed"] == s]))
    P_(f"    {'seed':>4} {'channel':>9} {'MLP deep':>9} {'diff':>8}")
    diffs = []
    for s in SEEDS:
        a, b = unit("channel", s), unit("mlpdeep", s)
        diffs.append(a - b)
        P_(f"    {s:>4} {a:>9.3f} {b:>9.3f} {a - b:>+8.3f}")
    w = sum(d > 0 for d in diffs); l = sum(d < 0 for d in diffs); n = len(diffs)
    p_ch = sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n
    p_mlp = sum(math.comb(n, i) for i in range(l, n + 1)) / 2 ** n
    md, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    P_(f"\n  channel tree higher on {w}/{n} seeds   one-sided sign p = {p_ch:.4f}   (pass needs >= 9/10)")
    P_(f"  mean difference {md:+.3f} +/- {sd:.3f}   paired t = {t:.2f}   (secondary passes at t > 1.833)")
    if w >= 9:
        verdict = "BEATS"
    elif l >= 9:
        verdict = "DEEP MLP BEATS IT"
    elif t > 1.833:
        verdict = "SUGGESTIVE, NOT ESTABLISHED"
    elif t < -1.833:
        verdict = "DEEP MLP SUGGESTIVELY BETTER, NOT ESTABLISHED"
    else:
        verdict = "NO ADVANTAGE OVER THE DEEP MLP"
    P_(f"\n  E0: {verdict}{'' if clean else '   (with the E1 budget caveat)'}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"k": K, "n_grid": N_GRID, "seeds": SEEDS, "max_steps": MAXSTEPS,
               "capped_unfitted": capped, "seed_diffs": diffs, "wins": w, "losses": l,
               "sign_p": p_ch, "paired_t": t, "verdict": verdict, "runs": runs},
              open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/edgedecide.json ({len(runs)} runs)")

    P_("\n" + RULE); P_("E2  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  1. One task family -- parity at k = 12 over 16 inputs -- and one tree shape. A result here")
    P_("     is about sample efficiency on this task, not reasoning and not ARC.")
    P_("  2. The deep MLP is ONE conventional baseline at matched parameters. Others -- wider,")
    P_("     differently regularised, convolutional, transformer -- were not tested.")
    P_("  3. Ten seeds is ten independent units; the 9/10 bar is strict on purpose.")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
