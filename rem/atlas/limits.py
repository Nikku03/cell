"""Testing the biological neuron to its limit -- difficulty and sample efficiency until things break.

WHY THIS EXISTS. Three runs in a row could not answer the neuron question, for three different
reasons, all recorded. dendrite.py's task was degenerate and then its one valid row hit a symmetry
defect. channeltree.py's first two sweeps were void, the second because train and test used
different parity subsets. Its valid sweep was SATURATED: channel tree, shallow MLP and deep MLP all
scored 1.000 at every parity order, so it showed a tie it could not have avoided showing. The
uploaded Reasoning-Engine-v2 package has the same defect: biological network, MLP and a neuron-free
symbolic system all at 100%. A saturated benchmark cannot rank anything.

So this module does not ask "who passes". It asks WHERE EACH MODEL BREAKS, on two axes:
  DIFFICULTY  parity order k over 16 inputs, k = 2..12, at a fixed training set
  SAMPLES     training-set size at a fixed k, down to 250 examples
The second axis is the one that matters most: the uploaded package's one hard test, ARC, scored
0/400 because the missing capability is learning from FEW examples.

AND IT ADDS THE CONTROL WHOSE ABSENCE LEFT THE LAST RESULT UNATTRIBUTABLE. channeltree.py measured
that the plain tanh tree is EXACTLY odd (no internal biases, tanh odd), so it cannot represent
even-order parity at all, and the channel tree's wins over it were explained by SYMMETRY BREAKING,
not shown to come from channel dynamics. Here a BIASED tanh tree -- one bias per node, nothing
else -- competes at matched parameters. If it matches the channel tree, the Hodgkin-Huxley driving
force adds nothing beyond a bias.

MODELS, all at ~37.5k parameters:
  channel tree       per-node weights + gated conductances with opposite reversals   (U=8)
  biased tanh tree   the old tanh tree plus one bias per node                       (U=9, to match)
  MLP shallow        one tanh hidden layer
  MLP deep           four tanh hidden layers

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

L1  THE NEW CONTROL MUST BE CORRECT, BLOCKING. Gradient check on the biased tanh tree, max
    relative error < 1e-5 on every parameter group, AND it must NOT be odd -- a "biased" control
    that is still odd would silently repeat the symmetry defect it exists to remove.

L2  THE HARNESS MUST WORK, BLOCKING. A shallow MLP must reach >= 0.99 on k = 2. The last time this
    control failed it located a bug that had invalidated every void row in two modules.

L3  THE BENCHMARK MUST NOT BE SATURATED. Each sweep must contain at least one cell where at least
    one model fails (mean test accuracy < 0.90). PREDECLARED: if every model clears every cell, the
    run reports "STILL SATURATED, no ranking possible" -- it does NOT report a tie as a finding.

L0  THE LIMIT, AND WHAT COUNTS AS A WIN, fixed now.
      difficulty limit  = the largest k with mean test accuracy >= 0.90 over 3 seeds
      sample limit      = the smallest training-set size at the sample-sweep k with mean >= 0.90
    The channel tree WINS only if, on at least one axis, its limit is strictly better than ALL
    THREE controls by at least one grid step, and it is worse than none of them on either axis.
    Anything else is NO ADVANTAGE. If the biased tanh tree matches it, the channel dynamics
    specifically add nothing.
    The sample-sweep k is fixed by a rule stated now, not chosen after seeing it: the largest k at
    which EVERY model reached >= 0.90 in the difficulty sweep (k = 2 if none). That uses only the
    difficulty sweep, never the sample sweep's own outcome.

L5  FITTING VERSUS GENERALISING. Train accuracy is recorded beside test accuracy in every cell, so
    a failure can be read as "cannot fit" (train low) or "fits but does not generalise" (train
    high, test low). These are different limits and must not be merged.

L4  WHAT THIS IS AND IS NOT.

PROTOCOL. Input width 16; train and test DISJOINT (no test input appears in training); the parity
subset is drawn once per (k, seed) and shared by train and test -- the bug fixed in channeltree.py.
Every model gets the same data, seed, optimiser (Adam, lr 0.01, batch 128) and budget: at most
10,000 steps, stopping early -- identically for every model -- once training accuracy is 1.0.
"Breaks at k" therefore means "breaks at k under this budget", and the budget is the same for all.
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
OUT = os.path.join(HERE, "RESULTS_limits.txt")
ART = R("outputs", "limits.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("channeltree", os.path.join(HERE, "channeltree.py"))
ch = importlib.util.module_from_spec(_s); _s.loader.exec_module(ch)

D, B, LV = 16, 4, 4                    # inputs; tree branching and depth from Blue Brain morphology
MAXSTEPS, CHECK, LR, BS = 10000, 500, 0.01, 128
SEEDS = (0, 1, 2)
K_GRID = (2, 4, 6, 8, 10, 12)
N_GRID = (250, 500, 1000, 2000, 4000)
NTR_DIFF, NTE = 8000, 4000
MODELS = ("channel", "btanh", "mlp", "mlpdeep")
LABEL = {"channel": "channel tree", "btanh": "biased tanh tree",
         "mlp": "MLP shallow", "mlpdeep": "MLP deep"}


class BiasedTanhTree(ch.TanhTree):
    """dendrite.py's tanh tree plus ONE bias per internal node. Nothing else changes."""

    def __init__(self, D, U, b, L, rng):
        super().__init__(D, U, b, L, rng)
        self.beta, M = [], self.K
        for _ in range(L):
            M //= b
            self.beta.append(np.zeros((U, M)))
        self.ps = [self.W, self.g] + self.beta + [self.wo, self.bo]

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.W)
        self.pre, self.X = [], X
        for l in range(self.L):
            pre = h.reshape(n, self.U, -1, self.b).sum(-1) + self.beta[l][None]
            self.pre.append(pre)
            h = np.tanh(pre) * self.g[:, l][None, :, None]
        self.s = h[:, :, 0]
        return self.s @ self.wo + self.bo

    def backward(self, dz):
        gwo = self.s.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]
        gg = np.zeros_like(self.g); gb = [None] * self.L
        n = self.X.shape[0]
        for l in range(self.L - 1, -1, -1):
            t = np.tanh(self.pre[l])
            gg[:, l] = (dh * t).sum(axis=(0, 2))
            dpre = dh * self.g[:, l][None, :, None] * (1 - t ** 2)
            gb[l] = dpre.sum(0)
            dh = np.repeat(dpre[:, :, :, None], self.b, axis=3).reshape(n, self.U, -1)
        return [np.einsum('nuk,nd->ukd', dh, self.X), gg] + gb + [gwo, gbo]


def build(name, rng):
    if name == "channel":
        return ch.ChannelTree(D, 8, B, LV, rng)
    if name == "btanh":
        return BiasedTanhTree(D, 9, B, LV, rng)
    P = ch.nparams(ch.ChannelTree(D, 8, B, LV, np.random.default_rng(0)))
    if name == "mlp":
        return ch.MLP(D, [int(round((P - 1) / (D + 2)))], rng)
    hd = int(round((math.sqrt((D + LV) ** 2 + 4 * (LV - 1) * P) - (D + LV)) / (2 * (LV - 1))))
    return ch.MLP(D, [hd] * LV, rng)


def data(k, seed, ntr):
    """Train and test DISJOINT; parity subset drawn once and shared."""
    rng = np.random.default_rng(100000 * k + 1000 * seed + 7)
    sub = rng.choice(D, size=k, replace=False)
    pool = rng.integers(0, 2, size=(ntr + NTE + 4000, D))
    keys = pool @ (1 << np.arange(D))
    _, first = np.unique(keys, return_index=True)
    pool = pool[np.sort(first)]
    tr, te = pool[:ntr], pool[ntr:ntr + NTE]
    lab = lambda Z: (Z[:, sub].sum(1) % 2).astype(np.float64)
    return tr * 2.0 - 1, lab(tr), te * 2.0 - 1, lab(te)


def acc(m, X, y):
    out = []
    for i in range(0, len(X), 2000):
        out.append(m.forward(X[i:i + 2000]))
    return float(((np.concatenate(out) > 0) == (y > 0.5)).mean())


def job(args):
    name, k, seed, ntr = args
    Xtr, ytr, Xte, yte = data(k, seed, ntr)
    m = build(name, np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps = {}, 0
    t0 = time.time()
    while steps < MAXSTEPS:
        for _ in range(CHECK):
            i = rng.integers(0, len(Xtr), BS)
            z = m.forward(Xtr[i])
            ch.adam(m.ps, m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / BS), st, LR)
        steps += CHECK
        if acc(m, Xtr, ytr) == 1.0:
            break
    return dict(model=name, k=k, seed=seed, ntr=ntr, steps=steps,
                train=acc(m, Xtr, ytr), test=acc(m, Xte, yte),
                params=ch.nparams(m), seconds=round(time.time() - t0, 1))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("TESTING THE BIOLOGICAL NEURON TO ITS LIMIT"); P_(RULE)
    for nm in MODELS:
        P_(f"  {LABEL[nm]:<18} {ch.nparams(build(nm, np.random.default_rng(0))):>7,} parameters")

    # ---- L1 BLOCKING ----------------------------------------------------------------------
    P_("\n" + RULE); P_("L1  THE NEW CONTROL: GRADIENT CHECK AND SYMMETRY (BLOCKING)"); P_(RULE)
    rng = np.random.default_rng(0)
    net = BiasedTanhTree(6, 3, 2, 2, rng)
    for p in net.beta:
        p[...] = rng.normal(0, 0.5, p.shape)
    X = rng.normal(size=(7, 6)); y = rng.integers(0, 2, 7).astype(float)
    z = net.forward(X); gs = net.backward((1 / (1 + np.exp(-z)) - y) / len(y))
    worst = 0.0
    for p, g in zip(net.ps, gs):
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = ch.loss_of(net, X, y)
            p[idx] = o - 1e-6; lm = ch.loss_of(net, X, y)
            p[idx] = o
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[idx]) / max(abs(num), abs(g[idx]), 1e-9))
    Xs = rng.choice([-1.0, 1.0], size=(400, 6))
    odd = float(np.abs((net.forward(-Xs) - net.bo) + (net.forward(Xs) - net.bo)).max())
    P_(f"  gradient check worst relative error   {worst:.2e}   (bar 1e-5)")
    P_(f"  max |f(-x) + f(x)|                    {odd:.2e}   (must be > 0: not odd)")
    if worst >= 1e-5 or odd < 1e-9:
        P_("\n  L1: FAIL -- the control is wrong. Nothing reported.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("\n  L1: PASS -- the biased tanh tree is correctly differentiated and is not odd")

    pool = mp.get_context("fork").Pool(4)

    # ---- DIFFICULTY SWEEP --------------------------------------------------------------------
    P_("\n" + RULE); P_("SWEEP 1  DIFFICULTY: parity order k over 16 inputs, 8,000 training examples"); P_(RULE)
    jobs = [(m, k, s, NTR_DIFF) for k in K_GRID for m in MODELS for s in SEEDS]
    r1 = list(pool.imap_unordered(job, jobs))
    cell = lambda rs, m, key, val: [r for r in rs if r["model"] == m and r[key] == val]
    mean = lambda rs, f: float(np.mean([r[f] for r in rs]))

    ctrl = cell(r1, "mlp", "k", 2)
    harness = mean(ctrl, "test")
    P_(f"  L2 harness control, shallow MLP at k=2: {harness:.3f}   (bar 0.99)")
    if harness < 0.99:
        P_("  L2: FAIL -- the harness is broken. Nothing reported.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("  L2: PASS")

    P_(f"\n  test accuracy (train accuracy in brackets), mean of {len(SEEDS)} seeds")
    P_("    " + f"{'k':>3} " + "".join(f"{LABEL[m]:>24}" for m in MODELS))
    lim1 = {}
    for k in K_GRID:
        row = f"    {k:>3} "
        for m in MODELS:
            c = cell(r1, m, "k", k)
            c = [r for r in c]
            te, tr = mean(c, "test"), mean(c, "train")
            row += f"{te:>15.3f} ({tr:.2f})  "
            if te >= 0.90:
                lim1[m] = max(lim1.get(m, 0), k)
        P_(row)
    fail1 = any(mean(cell([r for r in r1 if r["k"] == k], m, "k", k), "test") < 0.90
                for k in K_GRID for m in MODELS)
    P_("\n  DIFFICULTY LIMIT (largest k with mean test >= 0.90):")
    for m in MODELS:
        P_(f"    {LABEL[m]:<18} k = {lim1.get(m, 'none'):>4}")

    # ---- SAMPLE SWEEP -----------------------------------------------------------------------
    allpass = [k for k in K_GRID
               if all(mean(cell(r1, m, "k", k), "test") >= 0.90 for m in MODELS)]
    ks = max(allpass) if allpass else 2
    P_("\n" + RULE)
    P_(f"SWEEP 2  SAMPLES: training-set size at k = {ks}   (rule: largest k every model passed)")
    P_(RULE)
    jobs = [(m, ks, s, n) for n in N_GRID for m in MODELS for s in SEEDS]
    r2 = list(pool.imap_unordered(job, jobs))
    pool.close()
    P_("    " + f"{'n':>5} " + "".join(f"{LABEL[m]:>24}" for m in MODELS))
    lim2 = {}
    for n in N_GRID:
        row = f"    {n:>5} "
        for m in MODELS:
            c = cell(r2, m, "ntr", n)
            te, tr = mean(c, "test"), mean(c, "train")
            row += f"{te:>15.3f} ({tr:.2f})  "
            if te >= 0.90 and m not in lim2:
                lim2[m] = n
        P_(row)
    lim2 = {m: min([n for n in N_GRID if mean(cell(r2, m, "ntr", n), "test") >= 0.90],
                   default=None) for m in MODELS}
    fail2 = any(mean(cell(r2, m, "ntr", n), "test") < 0.90 for n in N_GRID for m in MODELS)
    P_("\n  SAMPLE LIMIT (fewest training examples with mean test >= 0.90):")
    for m in MODELS:
        P_(f"    {LABEL[m]:<18} n = {lim2[m] if lim2[m] else 'none'}")

    # ---- L3 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("L3  IS THE BENCHMARK UNSATURATED?"); P_(RULE)
    P_(f"  difficulty sweep has a failing cell: {fail1}")
    P_(f"  sample sweep has a failing cell:     {fail2}")
    if not (fail1 or fail2):
        P_("\n  L3: STILL SATURATED -- every model cleared every cell. No ranking is possible and")
        P_("  none is reported. The tie is not a finding.")
        verdict = "SATURATED"
    else:
        P_("\n  L3: PASS -- the benchmark breaks models, so limits can be ranked")
        # ---- L0 -----------------------------------------------------------------------------
        P_("\n" + RULE); P_("L0  THE VERDICT"); P_(RULE)
        ctrls = [m for m in MODELS if m != "channel"]
        kstep = K_GRID[1] - K_GRID[0]
        d_ch = lim1.get("channel", 0)
        s_ch = lim2["channel"] or 10 ** 9
        better_d = all(d_ch >= lim1.get(c, 0) + kstep for c in ctrls)
        better_s = all(s_ch < (lim2[c] or 10 ** 9) for c in ctrls)
        worse_d = any(d_ch < lim1.get(c, 0) for c in ctrls)
        worse_s = any(s_ch > (lim2[c] or 10 ** 9) for c in ctrls)
        win = (better_d or better_s) and not (worse_d or worse_s)
        P_(f"  channel tree strictly better than ALL controls on difficulty: {better_d}")
        P_(f"  channel tree strictly better than ALL controls on samples:    {better_s}")
        P_(f"  channel tree worse than SOME control on difficulty:           {worse_d}")
        P_(f"  channel tree worse than SOME control on samples:              {worse_s}")
        verdict = "WIN" if win else ("LOSES" if (worse_d or worse_s) else "NO ADVANTAGE")
        P_(f"\n  L0: {verdict}")
        if lim1.get("btanh") == d_ch and lim2["btanh"] == lim2["channel"]:
            P_("  The BIASED TANH TREE matches the channel tree on both limits: whatever the")
            P_("  channel tree achieves, a single bias per node achieves too. The Hodgkin-Huxley")
            P_("  driving force adds nothing measurable here.")

    # ---- L5 ---------------------------------------------------------------------------------
    P_("\n" + RULE); P_("L5  WHERE THE FAILURES ARE: CANNOT FIT, OR FITS BUT DOES NOT GENERALISE?"); P_(RULE)
    for rs, key in ((r1, "k"), (r2, "ntr")):
        for r in sorted(rs, key=lambda r: (r["model"], r[key], r["seed"])):
            pass
    bad = [r for r in r1 + r2 if r["test"] < 0.90]
    nofit = sum(1 for r in bad if r["train"] < 0.95)
    over = sum(1 for r in bad if r["train"] >= 0.95)
    P_(f"  failing runs: {len(bad)}   of which CANNOT FIT (train < 0.95): {nofit}"
       f"   FIT BUT DID NOT GENERALISE: {over}")
    for m in MODELS:
        bm = [r for r in bad if r["model"] == m]
        P_(f"    {LABEL[m]:<18} failing {len(bm):>3}   cannot fit {sum(1 for r in bm if r['train'] < 0.95):>3}"
           f"   overfit {sum(1 for r in bm if r['train'] >= 0.95):>3}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"difficulty_limit": lim1, "sample_limit": lim2, "sample_sweep_k": ks,
               "verdict": verdict, "runs_difficulty": r1, "runs_samples": r2,
               "params": {m: ch.nparams(build(m, np.random.default_rng(0))) for m in MODELS}},
              open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/limits.json ({len(r1) + len(r2)} training runs)")

    P_("\n" + RULE); P_("L4  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  1. Parity is a clean test of compositional learning. It is not reasoning, and a model's")
    P_("     limit on parity says nothing direct about ARC.")
    P_("  2. Limits are at a fixed 10,000-step budget, identical for every model. A model that")
    P_("     breaks at k might pass with more steps; so might its controls.")
    P_("  3. The trees take their branching (4) and depth (4) from Blue Brain morphology. Other")
    P_("     morphologies were not swept.")
    P_("  4. Three seeds per cell; the 0.90 threshold is applied to the mean.")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
