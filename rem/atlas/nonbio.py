"""Is the channel tree's edge BIOLOGICAL, or just non-monotonicity? Two non-biological twins decide.

WHERE edgedecide.py LEFT IT. The channel tree beat the deep MLP on sample efficiency (9/10 fresh
seeds, predeclared) and beat a structurally identical tanh tree (5/5). But its node expands to
    out = v * (1 - (g_dep + g_hyp) * m(v)) + (g_dep - g_hyp) * m(v),   m = sigmoid(v - theta)
a LEARNABLE NON-MONOTONE activation, and non-monotone units are known to make parity easier. So
"the gated conductance beats tanh" was established; "because it is biological" was not.

THE CONTROL. Two twins, each IDENTICAL to the channel tree -- leaves, per-node weights, readout,
three free parameters per node, EXACTLY the same parameter count -- with the node replaced by a
generic, non-biological, non-monotone function:
    sine tree    out = v + a * sin(w * v + phi)                    periodic, non-monotone
    bump tree    out = v + a * exp(-(v - mu)^2 / (s^2 + 0.01))     one localised bump
The tanh tree (monotone) is carried as the reference that the earlier runs already beat.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

B1  THE TWINS MUST BE VALID, BLOCKING. Gradient check < 1e-5 on every group for both; parameter
    count EXACTLY equal to the channel tree's; neither odd once parameters are nonzero.

B2  THE HARNESS MUST WORK, BLOCKING. Shallow MLP >= 0.99 at k = 2.

B0  THE ATTRIBUTION. Ten FRESH seeds, 30-39, disjoint from every earlier run. k = 12, n = 1500..4000,
    budget 30,000 steps, identical early stopping. One unit per seed (mean test over the n grid).
    Against EACH twin:
        channel higher on >= 9/10 seeds   -> channel BEATS that twin       (sign p = 0.011)
        twin higher on >= 9/10 seeds      -> that twin BEATS the channel tree
        otherwise                         -> NO DIFFERENCE at this resolution (paired t reported)
    THE ATTRIBUTION VERDICT, fixed now:
        channel beats BOTH twins  -> the gated-conductance form contributes BEYOND generic
                                     non-monotonicity; the biological form is doing something.
        otherwise                 -> NOT SPECIFICALLY BIOLOGICAL: a generic non-monotone
                                     activation does at least as well.

B3  SANITY ON THE TWINS THEMSELVES. A twin that does worse than the MONOTONE tanh tree is not
    evidence about biology -- it is evidence that twin trained badly. Reported per twin.

B4  BUDGET. Runs ending at the cap without fitting are counted per model; a capped model's verdict
    carries that caveat.

B5  WHAT THIS IS AND IS NOT.
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
OUT = os.path.join(HERE, "RESULTS_nonbio.txt")
ART = R("outputs", "nonbio.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("limits", os.path.join(HERE, "limits.py"))
lim = importlib.util.module_from_spec(_s); _s.loader.exec_module(lim)
_s2 = importlib.util.spec_from_file_location("sampleedge", os.path.join(HERE, "sampleedge.py"))
se = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(se)
ch = lim.ch

K = 12
N_GRID = (1500, 2000, 2500, 3000, 3500, 4000)
SEEDS = tuple(range(30, 40))
MAXSTEPS = 30000
MODELS = ("channel", "sine", "bump", "wtanh")
LABEL = {"channel": "channel tree", "sine": "sine tree", "bump": "bump tree",
         "wtanh": "tanh tree (monotone)"}


class NodeTree:
    """The channel tree's skeleton with a pluggable three-parameter node."""

    def __init__(self, D, U, b, L, rng):
        self.D, self.U, self.b, self.L = D, U, b, L
        self.K = b ** L
        self.Wl = rng.normal(0, 1 / math.sqrt(D), (U, self.K, D))
        self.W, self.p1, self.p2, self.p3 = [], [], [], []
        M = self.K
        for _ in range(L):
            M //= b
            self.W.append(rng.normal(0, 1 / math.sqrt(b), (U, M, b)))
            a, c, d = self.init_node((U, M), rng)
            self.p1.append(a); self.p2.append(c); self.p3.append(d)
        self.wo = rng.normal(0, 1 / math.sqrt(U), U)
        self.bo = np.zeros(1)
        self.ps = [self.Wl] + self.W + self.p1 + self.p2 + self.p3 + [self.wo, self.bo]

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.Wl)
        self.cX, self.cH, self.cC = X, [h], []
        for l in range(self.L):
            chn = h.reshape(n, self.U, -1, self.b)
            v = (chn * self.W[l][None]).sum(-1)
            h, c = self.node_f(v, self.p1[l][None], self.p2[l][None], self.p3[l][None])
            self.cC.append(c); self.cH.append(h)
        self.s = h[:, :, 0]
        return self.s @ self.wo + self.bo

    def backward(self, dz):
        n = self.cX.shape[0]
        gwo = self.s.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]
        gW, g1, g2, g3 = [None] * self.L, [None] * self.L, [None] * self.L, [None] * self.L
        for l in range(self.L - 1, -1, -1):
            dv, d1, d2, d3 = self.node_b(dh, self.cC[l],
                                         self.p1[l][None], self.p2[l][None], self.p3[l][None])
            g1[l], g2[l], g3[l] = d1.sum(0), d2.sum(0), d3.sum(0)
            chn = self.cH[l].reshape(n, self.U, -1, self.b)
            gW[l] = (dv[:, :, :, None] * chn).sum(0)
            dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, self.U, -1)
        return [np.einsum('nuk,nd->ukd', dh, self.cX)] + gW + g1 + g2 + g3 + [gwo, gbo]


class SineTree(NodeTree):
    """out = v + a * sin(w * v + phi)."""

    def init_node(self, shape, rng):
        return np.full(shape, 0.5), np.ones(shape), rng.normal(0, 1, shape)

    def node_f(self, v, a, w, phi):
        u = w * v + phi
        return v + a * np.sin(u), (v, u)

    def node_b(self, dh, c, a, w, phi):
        v, u = c
        cu = np.cos(u)
        return dh * (1 + a * w * cu), dh * np.sin(u), dh * a * cu * v, dh * a * cu


class BumpTree(NodeTree):
    """out = v + a * exp(-(v - mu)^2 / (s^2 + 0.01))."""

    def init_node(self, shape, rng):
        return np.full(shape, 0.5), np.zeros(shape), np.ones(shape)

    def node_f(self, v, a, mu, s):
        Dn = s * s + 0.01
        q = (v - mu) ** 2 / Dn
        e = np.exp(-q)
        return v + a * e, (v, e, q, Dn)

    def node_b(self, dh, c, a, mu, s):
        v, e, q, Dn = c
        k = a * e * 2 * (v - mu) / Dn
        return dh * (1 - k), dh * e, dh * k, dh * a * e * q * 2 * s / Dn


def build(name, rng):
    if name == "sine":
        return SineTree(lim.D, 8, lim.B, lim.LV, rng)
    if name == "bump":
        return BumpTree(lim.D, 8, lim.B, lim.LV, rng)
    return se.build(name, rng)


def job(args):
    name, k, seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(k, seed, ntr)
    m = build(name, np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps, t0 = {}, 0, time.time()
    while steps < MAXSTEPS:
        for _ in range(lim.CHECK):
            i = rng.integers(0, len(Xtr), lim.BS)
            z = m.forward(Xtr[i])
            ch.adam(m.ps, m.backward((1 / (1 + np.exp(-z)) - ytr[i]) / lim.BS), st, lim.LR)
        steps += lim.CHECK
        if not np.all(np.isfinite(m.forward(Xtr[:64]))):
            break
        if lim.acc(m, Xtr, ytr) == 1.0:
            break
    return dict(model=name, k=k, seed=seed, ntr=ntr, steps=steps,
                train=lim.acc(m, Xtr, ytr), test=lim.acc(m, Xte, yte),
                seconds=round(time.time() - t0, 1))


def decide(units, a, b, seeds):
    diffs = [units[(a, s)] - units[(b, s)] for s in seeds]
    w = sum(d > 0 for d in diffs); l = sum(d < 0 for d in diffs); n = len(diffs)
    p = sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n
    md, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    v = ("BEATS" if w >= 9 else "IS BEATEN BY" if l >= 9 else "NO DIFFERENCE AT THIS RESOLUTION")
    return dict(wins=w, losses=l, sign_p=p, mean_diff=md, paired_t=t, verdict=v, diffs=diffs)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("IS THE EDGE BIOLOGICAL? TWO NON-BIOLOGICAL TWINS AT EXACTLY MATCHED SIZE"); P_(RULE)
    pc = {m: ch.nparams(build(m, np.random.default_rng(0))) for m in MODELS}
    for m in MODELS:
        P_(f"  {LABEL[m]:<22} {pc[m]:>7,} parameters")

    # ---- B1 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("B1  THE TWINS MUST BE VALID (BLOCKING)"); P_(RULE)
    ok = True
    for cls, nm in ((SineTree, "sine"), (BumpTree, "bump")):
        rng = np.random.default_rng(1)
        net = cls(6, 3, 2, 2, rng)
        for p in net.p1 + net.p2 + net.p3:
            p[...] = rng.normal(0.3, 0.4, p.shape)
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
        same = pc[nm] == pc["channel"]
        P_(f"  {LABEL[nm]:<12} gradient {worst:.2e}   not odd {odd:.2e}   params equal to channel: {same}")
        ok = ok and worst < 1e-5 and odd > 1e-9 and same
    if not ok:
        P_("\n  B1: FAIL -- a twin is invalid. Nothing reported.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("\n  B1: PASS")

    pool = mp.get_context("fork").Pool(4)
    hv = float(np.mean([r["test"] for r in pool.map(lim.job, [("mlp", 2, s, 4000) for s in (30, 31)])]))
    P_(f"\n  B2 harness, shallow MLP at k=2: {hv:.3f}  -> {'PASS' if hv >= 0.99 else 'FAIL'}")
    if hv < 0.99:
        open(OUT, "w").write("\n".join(out) + "\n"); return

    P_("\n" + RULE)
    P_(f"FRESH-SEED SWEEP: k = {K}, seeds {SEEDS[0]}-{SEEDS[-1]}, n = {N_GRID[0]}..{N_GRID[-1]}, budget {MAXSTEPS:,}")
    P_(RULE)
    runs = pool.map(job, [(m, K, s, n) for s in SEEDS for n in N_GRID for m in MODELS])
    pool.close()

    P_("  mean test accuracy over 10 seeds [min - max]")
    P_("    " + f"{'n':>5} " + "".join(f"{LABEL[m]:>26}" for m in MODELS))
    for n in N_GRID:
        row = f"    {n:>5} "
        for m in MODELS:
            v = [r["test"] for r in runs if r["model"] == m and r["ntr"] == n]
            row += f"{np.mean(v):>13.3f} [{min(v):.2f}-{max(v):.2f}]"
        P_(row)

    units = {(m, s): float(np.mean([r["test"] for r in runs if r["model"] == m and r["seed"] == s]))
             for m in MODELS for s in SEEDS}

    # ---- B4 -----------------------------------------------------------------------------------
    P_("\n  B4 budget: runs ending at the cap unfitted, and max steps used")
    capped = {}
    for m in MODELS:
        rs = [r for r in runs if r["model"] == m]
        capped[m] = sum(1 for r in rs if r["steps"] >= MAXSTEPS and r["train"] < 1.0)
        P_(f"    {LABEL[m]:<22} {capped[m]:>2}/{len(rs)}   max steps {max(r['steps'] for r in rs):,}")

    # ---- B0 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("B0  THE ATTRIBUTION, ONE UNIT PER SEED"); P_(RULE)
    P_("    seed " + "".join(f"{LABEL[m]:>22}" for m in MODELS))
    for s in SEEDS:
        P_(f"    {s:>4} " + "".join(f"{units[(m, s)]:>22.3f}" for m in MODELS))
    res = {}
    for tw in ("sine", "bump", "wtanh"):
        r = decide(units, "channel", tw, SEEDS)
        res[tw] = r
        P_(f"\n  channel tree vs {LABEL[tw]:<22} channel higher on {r['wins']}/10   "
           f"sign p {r['sign_p']:.4f}   mean diff {r['mean_diff']:+.3f}   t {r['paired_t']:.2f}")
        P_(f"    -> channel tree {r['verdict']} the {LABEL[tw]}"
           f"{'' if capped.get(tw, 0) == 0 else '   (that twin has capped runs: caveat)'}")
    beats_both = res["sine"]["verdict"] == "BEATS" and res["bump"]["verdict"] == "BEATS"
    P_("\n  ATTRIBUTION VERDICT:")
    if beats_both:
        P_("    THE GATED-CONDUCTANCE FORM CONTRIBUTES BEYOND GENERIC NON-MONOTONICITY.")
        P_("    Both non-biological twins of identical size are beaten on fresh seeds.")
    else:
        P_("    NOT SPECIFICALLY BIOLOGICAL. At least one generic non-monotone activation of")
        P_("    identical size is not beaten by the channel tree, so the edge established earlier")
        P_("    belongs to non-monotonicity, not to the biological form.")

    # ---- B3 -----------------------------------------------------------------------------------
    P_("\n" + RULE); P_("B3  ARE THE TWINS THEMSELVES HEALTHY? (vs the monotone tanh tree)"); P_(RULE)
    for tw in ("sine", "bump"):
        r = decide(units, tw, "wtanh", SEEDS)
        healthy = r["mean_diff"] > 0
        P_(f"  {LABEL[tw]:<12} vs monotone tanh tree: higher on {r['wins']}/10, mean diff {r['mean_diff']:+.3f}"
           f"  -> {'healthy' if healthy else 'WORSE THAN MONOTONE: evidence it trained badly, not about biology'}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"k": K, "seeds": SEEDS, "n_grid": N_GRID, "params": pc, "capped": capped,
               "units": {f"{m}|{s}": v for (m, s), v in units.items()},
               "decisions": res, "beats_both": beats_both, "runs": runs},
              open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/nonbio.json ({len(runs)} runs)")

    P_("\n" + RULE); P_("B5  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  1. Two non-biological twins, not every possible activation. If both lose, the conductance")
    P_("     form beats THESE TWO; a third generic family could still match it.")
    P_("  2. Initial values for the twins' node parameters are a design choice and were not swept.")
    P_("  3. One task (12-bit parity over 16 inputs), one tree shape.")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
