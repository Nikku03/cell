"""Is the channel tree's sample-efficiency edge real, and is it the channel dynamics?

WHERE limits.py LEFT IT. Its predeclared verdict was NO ADVANTAGE: the channel tree tied the deep
MLP at the sample limit (4,000 examples on a doubling grid) and every model tied on difficulty.
But it recorded, labelled as EXPLORATORY, that at 2,000 examples every channel-tree seed
(0.651-0.788) beat every control seed (max 0.600). The doubling grid could not see a limit between
2,000 and 4,000, and a pattern found after looking is a hypothesis, not a result.

It also left the attribution confounded. The biased tanh tree was the worst model on both axes,
but it lacked the channel tree's PER-NODE WEIGHTS, so "channel tree beats biased tanh tree" could
be the weights rather than the channel dynamics.

THIS MODULE TESTS BOTH, PREDECLARED, ON DATA THE HYPOTHESIS HAS NEVER SEEN.
  REPLICATION  five FRESH seeds (10-14; limits.py used 0-2), a finer grid n = 1500..4000 in steps
               of 500, same k = 12 as limits.py's rule selected. Same protocol throughout.
  ISOLATION    a WEIGHTED tanh tree: per-node weights, per-node biases, per-node gains --
               structurally identical to the channel tree, differing ONLY in the node
               nonlinearity: tanh versus the gated conductance g * m(v) * (E_rev - v).

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

S1  THE ISOLATING CONTROL MUST BE CORRECT, BLOCKING. Gradient check < 1e-5 on every group; NOT odd
    once its biases are nonzero; parameter count within 2% of the channel tree.

S2  THE HARNESS MUST WORK, BLOCKING. Shallow MLP >= 0.99 at k = 2.

S0  THE REPLICATION. The exploratory claim was that the channel tree is more sample-efficient than
    every control. It REPLICATES only if BOTH hold on the fresh seeds:
      (a) its sample limit (smallest n with mean test >= 0.90 over 5 seeds) is strictly smaller
          than the deep MLP's and the shallow MLP's, AND
      (b) a one-sided sign test over all (n, seed) pairs, ties excluded, favours it against each
          of those two controls at p < 0.05.
    If either fails, the exploratory signal did NOT replicate and is reported as such.

S3  ISOLATION, by the same two criteria against the WEIGHTED tanh tree. If the channel tree does
    not beat it, any edge is TREE STRUCTURE, not channel dynamics, and the Hodgkin-Huxley term
    adds nothing. If it does, the channel dynamics specifically contribute.

S4  A THRESHOLD-FREE SUMMARY, secondary: mean test accuracy across the whole n grid per model,
    with the spread over seeds. Reported so the verdict does not rest on one threshold alone.

S5  WHAT THIS IS AND IS NOT.
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
OUT = os.path.join(HERE, "RESULTS_sampleedge.txt")
ART = R("outputs", "sampleedge.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("limits", os.path.join(HERE, "limits.py"))
lim = importlib.util.module_from_spec(_s); _s.loader.exec_module(lim)
ch = lim.ch

K = 12
N_GRID = (1500, 2000, 2500, 3000, 3500, 4000)
SEEDS = (10, 11, 12, 13, 14)
MODELS = ("channel", "wtanh", "mlpdeep", "mlp")
LABEL = {"channel": "channel tree", "wtanh": "weighted tanh tree",
         "mlpdeep": "MLP deep", "mlp": "MLP shallow"}


class WeightedTanhTree:
    """The channel tree with its node replaced by g * tanh(sum_c w_c child_c + beta). Everything
    else -- leaves, per-node weights, readout -- is identical, so the comparison isolates the
    nonlinearity."""

    def __init__(self, D, U, b, L, rng):
        self.D, self.U, self.b, self.L = D, U, b, L
        self.K = b ** L
        self.Wl = rng.normal(0, 1 / math.sqrt(D), (U, self.K, D))
        self.W, self.beta, self.g = [], [], []
        M = self.K
        for _ in range(L):
            M //= b
            self.W.append(rng.normal(0, 1 / math.sqrt(b), (U, M, b)))
            self.beta.append(np.zeros((U, M)))
            self.g.append(np.ones((U, M)))
        self.wo = rng.normal(0, 1 / math.sqrt(U), U)
        self.bo = np.zeros(1)
        self.ps = [self.Wl] + self.W + self.beta + self.g + [self.wo, self.bo]

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.Wl)
        self.cX, self.cH, self.cT = X, [h], []
        for l in range(self.L):
            chn = h.reshape(n, self.U, -1, self.b)
            v = (chn * self.W[l][None]).sum(-1) + self.beta[l][None]
            t = np.tanh(v)
            h = self.g[l][None] * t
            self.cT.append(t); self.cH.append(h)
        self.s = h[:, :, 0]
        return self.s @ self.wo + self.bo

    def backward(self, dz):
        n = self.cX.shape[0]
        gwo = self.s.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]
        gW, gB, gG = [None] * self.L, [None] * self.L, [None] * self.L
        for l in range(self.L - 1, -1, -1):
            t = self.cT[l]
            gG[l] = (dh * t).sum(0)
            dv = dh * self.g[l][None] * (1 - t ** 2)
            gB[l] = dv.sum(0)
            chn = self.cH[l].reshape(n, self.U, -1, self.b)
            gW[l] = (dv[:, :, :, None] * chn).sum(0)
            dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, self.U, -1)
        return [np.einsum('nuk,nd->ukd', dh, self.cX)] + gW + gB + gG + [gwo, gbo]


def build(name, rng):
    if name == "wtanh":
        return WeightedTanhTree(lim.D, 8, lim.B, lim.LV, rng)
    return lim.build(name, rng)


def job(args):
    name, k, seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(k, seed, ntr)
    m = build(name, np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps, t0 = {}, 0, time.time()
    while steps < lim.MAXSTEPS:
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


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("IS THE SAMPLE-EFFICIENCY EDGE REAL, AND IS IT THE CHANNEL DYNAMICS?"); P_(RULE)
    pc = {m: ch.nparams(build(m, np.random.default_rng(0))) for m in MODELS}
    for m in MODELS:
        P_(f"  {LABEL[m]:<20} {pc[m]:>7,} parameters")

    # ---- S1 --------------------------------------------------------------------------------
    P_("\n" + RULE); P_("S1  THE ISOLATING CONTROL (BLOCKING)"); P_(RULE)
    rng = np.random.default_rng(0)
    net = WeightedTanhTree(6, 3, 2, 2, rng)
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
    gap = abs(pc["wtanh"] - pc["channel"]) / pc["channel"]
    P_(f"  gradient check worst relative error   {worst:.2e}   (bar 1e-5)")
    P_(f"  max |f(-x) + f(x)| with biases set    {odd:.2e}   (must be > 0)")
    P_(f"  parameter gap to channel tree         {100*gap:.1f}%     (bar 2%)")
    if worst >= 1e-5 or odd < 1e-9 or gap > 0.02:
        P_("\n  S1: FAIL -- the isolating control is not valid. Nothing reported.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("\n  S1: PASS")

    pool = mp.get_context("fork").Pool(4)
    hc = pool.map(job, [("mlp", 2, s, 4000) for s in SEEDS[:2]])
    hv = float(np.mean([r["test"] for r in hc]))
    P_(f"\n  S2 harness, shallow MLP at k=2: {hv:.3f} (bar 0.99)  -> {'PASS' if hv >= 0.99 else 'FAIL'}")
    if hv < 0.99:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    # ---- the sweep ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_(f"FRESH-SEED SWEEP: k = {K}, seeds {SEEDS[0]}-{SEEDS[-1]} (limits.py used 0-2), n = {N_GRID[0]}..{N_GRID[-1]}")
    P_(RULE)
    runs = pool.map(job, [(m, K, s, n) for n in N_GRID for m in MODELS for s in SEEDS])
    pool.close()
    get = lambda m, n: sorted([r for r in runs if r["model"] == m and r["ntr"] == n],
                              key=lambda r: r["seed"])
    P_("  mean test accuracy over 5 seeds [min - max]")
    P_("    " + f"{'n':>5} " + "".join(f"{LABEL[m]:>25}" for m in MODELS))
    lim_n = {}
    for n in N_GRID:
        row = f"    {n:>5} "
        for m in MODELS:
            v = [r["test"] for r in get(m, n)]
            row += f"{np.mean(v):>12.3f} [{min(v):.2f}-{max(v):.2f}]"
            if np.mean(v) >= 0.90 and m not in lim_n:
                lim_n[m] = n
        P_(row)
    P_("\n  SAMPLE LIMIT (smallest n with mean test >= 0.90):")
    for m in MODELS:
        P_(f"    {LABEL[m]:<20} {lim_n.get(m, 'none')}")

    def compare(ctrl):
        w = l = 0
        for n in N_GRID:
            for a, b in zip(get("channel", n), get(ctrl, n)):
                if a["test"] > b["test"] + 1e-9:
                    w += 1
                elif b["test"] > a["test"] + 1e-9:
                    l += 1
        lc, lo = lim_n.get("channel", 10 ** 9), lim_n.get(ctrl, 10 ** 9)
        p = sign_p(w, l)
        return dict(wins=w, losses=l, p=p, limit_better=lc < lo, ok=(lc < lo and p < 0.05))

    # ---- S0 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("S0  DOES THE EXPLORATORY SIGNAL REPLICATE?"); P_(RULE)
    c0 = {c: compare(c) for c in ("mlpdeep", "mlp")}
    for c, v in c0.items():
        P_(f"  vs {LABEL[c]:<14} limit strictly better: {str(v['limit_better']):<5}  "
           f"paired wins/losses {v['wins']}/{v['losses']}  sign p = {v['p']:.2e}  -> "
           f"{'beats it' if v['ok'] else 'does NOT beat it'}")
    rep = all(v["ok"] for v in c0.values())
    P_(f"\n  S0: {'REPLICATED -- the channel tree is more sample-efficient than both MLPs on fresh seeds' if rep else 'DID NOT REPLICATE -- the exploratory signal does not survive fresh seeds and a finer grid'}")

    # ---- S3 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("S3  IS IT THE CHANNEL DYNAMICS? (vs the structurally identical tanh tree)"); P_(RULE)
    c3 = compare("wtanh")
    P_(f"  vs {LABEL['wtanh']:<20} limit strictly better: {str(c3['limit_better']):<5}  "
       f"paired wins/losses {c3['wins']}/{c3['losses']}  sign p = {c3['p']:.2e}")
    if c3["ok"]:
        P_("\n  S3: THE CHANNEL DYNAMICS CONTRIBUTE -- with everything else identical, the gated")
        P_("  conductance beats tanh on both criteria.")
    else:
        P_("\n  S3: NOT THE CHANNEL DYNAMICS -- the structurally identical tanh tree is not beaten,")
        P_("  so any edge belongs to the TREE STRUCTURE and the Hodgkin-Huxley term adds nothing")
        P_("  measurable here.")

    # ---- S4 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("S4  THRESHOLD-FREE: MEAN TEST ACCURACY OVER THE WHOLE n GRID"); P_(RULE)
    auc = {}
    for m in MODELS:
        per_seed = [np.mean([r["test"] for r in runs if r["model"] == m and r["seed"] == s])
                    for s in SEEDS]
        auc[m] = (float(np.mean(per_seed)), float(np.std(per_seed)))
        P_(f"    {LABEL[m]:<20} {auc[m][0]:.3f} +/- {auc[m][1]:.3f}")
    fit = [r for r in runs if r["test"] < 0.90]
    P_(f"\n  failing runs {len(fit)}: perfect train fit in {sum(1 for r in fit if r['train'] >= 0.95)},"
       f" could not fit in {sum(1 for r in fit if r['train'] < 0.95)}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"k": K, "seeds": SEEDS, "n_grid": N_GRID, "params": pc, "sample_limit": lim_n,
               "replication": c0, "isolation": c3, "replicated": rep,
               "threshold_free": auc, "runs": runs}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/sampleedge.json ({len(runs)} runs)")

    P_("\n" + RULE); P_("S5  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  1. Parity at k = 12 over 16 inputs. Sample efficiency on parity is not reasoning, and does")
    P_("     not transfer to ARC by itself.")
    P_("  2. Fixed 10,000-step budget, identical early stopping for all.")
    P_("  3. One tree shape (branching 4, depth 4, from Blue Brain morphology).")
    P_("  4. Five seeds per cell. The sign test pools n and seed and treats pairs as independent;")
    P_("     cells at the same seed share a parity subset, so p is somewhat optimistic.")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
