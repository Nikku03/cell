"""Does the optogenetic data repair the weak node? Separate gates, ordered by real neurons.

THE WEAK PART. The channel node opens its depolarising and hyperpolarising conductances with ONE
shared gate; it lost to generic bends by ~0.17 on 10/10 seeds (nonbio.py). optounits.py measures, in
4,335 ALM units under graded channelrhodopsin drive, whether suppression is recruited at higher or
lower drive than excitation, and writes ORDER (+1, -1 or 0) by a rule fixed before it ran.

THE REPAIR, applied mechanically from outputs/opto_units.json:
    out = v + g_dep sigma(v - theta_dep)(1 - v) + g_hyp sigma(v - theta_hyp)(-1 - v)
    init  g_dep = g_hyp = 0.5,  theta_dep = 0,  theta_hyp = ORDER * 1.0
With ORDER = 0 the node starts EXACTLY as the channel node (v - SiLU(v)) with its gates untied.
Four parameters per node instead of three: 680 more than the channel tree (+1.8%), disclosed.

=================================================================================================
GATES, PREDECLARED (committed with optounits.py, before either ran)
=================================================================================================
G1 VALID, BLOCKING. Gradient |num - ana| <= 1e-5 max(|num|,|ana|) + 1e-9 on sampled entries; a
   deliberately broken copy (theta_dep gradient missing its sigma' factor) MUST fail; not odd.
G2 HARNESS, BLOCKING. Shallow MLP >= 0.99 at k = 2.
Fresh seeds 50-59, k = 12, n = 1500..4000, budget 30,000; one unit per seed (mean over n).
G0 PRIMARY -- DOES IT HELP THE WEAK PART? two-gate vs channel tree:
       >= 9/10 higher -> HELPS;  >= 9/10 lower -> HARMS;  else NO CLEAR EFFECT.
G3 DOES IT CLOSE THE GAP? two-gate vs bump tree (the generic bar), same three-way rule.
G4 DID THE DATA'S ORDERING MATTER? Only if ORDER != 0: data-ordered vs reversed initialisation,
   same rule. If ORDER = 0 the data supplied no ordering and G4 is not run.
G5 Runs capped unfitted, per model.
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
OUT = os.path.join(HERE, "RESULTS_twogate.txt")
ART = R("outputs", "twogate.json")
UNITS = R("outputs", "opto_units.json")
RULE = "=" * 97

_s = importlib.util.spec_from_file_location("nonbio", os.path.join(HERE, "nonbio.py"))
nb = importlib.util.module_from_spec(_s); _s.loader.exec_module(nb)
lim, ch = nb.lim, nb.ch

K = 12
N_GRID = (1500, 2000, 2500, 3000, 3500, 4000)
SEEDS = tuple(range(50, 60))
MAXSTEPS = 30000
ORDER = None


class TwoGateTree:
    """The channel tree with separate gates for its two conductances."""

    def __init__(self, D, U, b, L, rng, order=0):
        self.D, self.U, self.b, self.L = D, U, b, L
        self.K = b ** L
        self.Wl = rng.normal(0, 1 / math.sqrt(D), (U, self.K, D))
        self.W, self.gd, self.gh, self.td, self.th = [], [], [], [], []
        M = self.K
        for _ in range(L):
            M //= b
            self.W.append(rng.normal(0, 1 / math.sqrt(b), (U, M, b)))
            self.gd.append(np.full((U, M), 0.5)); self.gh.append(np.full((U, M), 0.5))
            self.td.append(np.zeros((U, M))); self.th.append(np.full((U, M), float(order)))
        self.wo = rng.normal(0, 1 / math.sqrt(U), U)
        self.bo = np.zeros(1)
        self.ps = [self.Wl] + self.W + self.gd + self.gh + self.td + self.th + [self.wo, self.bo]

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.Wl)
        self.cX, self.cH, self.cC = X, [h], []
        for l in range(self.L):
            v = (h.reshape(n, self.U, -1, self.b) * self.W[l][None]).sum(-1)
            md = 1 / (1 + np.exp(-(v - self.td[l][None])))
            mh = 1 / (1 + np.exp(-(v - self.th[l][None])))
            h = v + self.gd[l][None] * md * (1 - v) + self.gh[l][None] * mh * (-1 - v)
            self.cC.append((v, md, mh)); self.cH.append(h)
        self.s = h[:, :, 0]
        return self.s @ self.wo + self.bo

    def dtheta_d(self, dh, gd, v, md):
        return dh * gd * (1 - v) * (-md * (1 - md))

    def backward(self, dz):
        n = self.cX.shape[0]
        gwo = self.s.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]
        L = self.L
        gW, ggd, ggh, gtd, gth = [None] * L, [None] * L, [None] * L, [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            v, md, mh = self.cC[l]
            gd, gh = self.gd[l][None], self.gh[l][None]
            ggd[l] = (dh * md * (1 - v)).sum(0)
            ggh[l] = (dh * mh * (-1 - v)).sum(0)
            gtd[l] = self.dtheta_d(dh, gd, v, md).sum(0)
            gth[l] = (dh * gh * (-1 - v) * (-mh * (1 - mh))).sum(0)
            dv = dh * (1 - gd * md - gh * mh + gd * (1 - v) * md * (1 - md) + gh * (-1 - v) * mh * (1 - mh))
            chn = self.cH[l].reshape(n, self.U, -1, self.b)
            gW[l] = (dv[:, :, :, None] * chn).sum(0)
            dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, self.U, -1)
        return [np.einsum('nuk,nd->ukd', dh, self.cX)] + gW + ggd + ggh + gtd + gth + [gwo, gbo]


class BrokenTwoGate(TwoGateTree):
    """G1 negative control: theta_dep gradient loses its sigma' factor."""

    def dtheta_d(self, dh, gd, v, md):
        return dh * gd * (1 - v) * (-md)


def gradcheck(cls, seed=1):
    rng = np.random.default_rng(seed)
    net = cls(6, 3, 2, 2, rng, order=1)
    for p in net.gd + net.gh + net.td + net.th:
        p[...] = rng.normal(0.3, 0.4, p.shape)
    X = rng.normal(size=(7, 6)); y = rng.integers(0, 2, 7).astype(float)
    z = net.forward(X); gs = net.backward((1 / (1 + np.exp(-z)) - y) / len(y))
    ratio = 0.0
    for p, g in zip(net.ps, gs):
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = ch.loss_of(net, X, y)
            p[idx] = o - 1e-6; lm = ch.loss_of(net, X, y)
            p[idx] = o
            num = (lp - lm) / 2e-6
            ratio = max(ratio, abs(num - g[idx]) / (1e-5 * max(abs(num), abs(g[idx])) + 1e-9))
    Xs = np.random.default_rng(seed + 1).choice([-1.0, 1.0], size=(400, 6))
    odd = float(np.abs((net.forward(-Xs) - net.bo) + (net.forward(Xs) - net.bo)).max())
    return ratio, odd


def build(name, rng):
    if name == "twogate":
        return TwoGateTree(lim.D, 8, lim.B, lim.LV, rng, order=ORDER)
    if name == "twogate_rev":
        return TwoGateTree(lim.D, 8, lim.B, lim.LV, rng, order=-ORDER)
    return nb.build(name, rng)


def job(args):
    name, k, seed, ntr = args
    Xtr, ytr, Xte, yte = lim.data(k, seed, ntr)
    m = build(name, np.random.default_rng(200 + seed))
    rng = np.random.default_rng(300 + seed)
    st, steps = {}, 0
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
                train=lim.acc(m, Xtr, ytr), test=lim.acc(m, Xte, yte))


def harness_job(args):
    return lim.job(args)


def main():
    global ORDER
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    if not os.path.exists(UNITS):
        print("outputs/opto_units.json missing: run optounits.py first."); return
    ORDER = int(json.load(open(UNITS))["order"])
    models = ["twogate"] + (["twogate_rev"] if ORDER != 0 else []) + ["channel", "bump"]
    label = {"twogate": f"two-gate (ORDER {ORDER:+d})", "twogate_rev": f"two-gate reversed ({-ORDER:+d})",
             "channel": "channel tree", "bump": "bump tree"}
    t0 = time.time()
    P_(RULE); P_("DOES THE OPTOGENETIC DATA REPAIR THE WEAK NODE?"); P_(RULE)
    P_(f"  ORDER read from outputs/opto_units.json: {ORDER:+d}")
    for m in models:
        P_(f"  {label[m]:<26} {ch.nparams(build(m, np.random.default_rng(0))):>7,} parameters")
    ratio, odd = gradcheck(TwoGateTree); bratio, _ = gradcheck(BrokenTwoGate)
    g1 = ratio <= 1.0 and odd > 1e-9 and bratio > 1.0
    P_(f"  G1 gradient / tolerance {ratio:.3f}   broken copy {bratio:.2e} ({'caught' if bratio > 1 else 'NOT CAUGHT'})"
       f"   not odd {odd:.2e}   -> {'PASS' if g1 else 'FAIL'}")
    if not g1:
        open(OUT, "w").write("\n".join(out) + "\n"); return
    pool = mp.get_context("fork").Pool(4)
    hv = float(np.mean([r["test"] for r in pool.map(harness_job, [("mlp", 2, s, 4000) for s in (50, 51)])]))
    P_(f"  G2 harness {hv:.3f} -> {'PASS' if hv >= 0.99 else 'FAIL'}")
    if hv < 0.99:
        open(OUT, "w").write("\n".join(out) + "\n"); pool.close(); return
    runs = pool.map(job, [(m, K, s, n) for s in SEEDS for n in N_GRID for m in models])
    pool.close()
    P_("\n  mean test accuracy over 10 seeds [min - max]")
    P_("    " + f"{'n':>5} " + "".join(f"{label[m]:>28}" for m in models))
    for n in N_GRID:
        row = f"    {n:>5} "
        for m in models:
            v = [r["test"] for r in runs if r["model"] == m and r["ntr"] == n]
            row += f"{np.mean(v):>15.3f} [{min(v):.2f}-{max(v):.2f}]"
        P_(row)
    units = {(m, s): float(np.mean([r["test"] for r in runs if r["model"] == m and r["seed"] == s]))
             for m in models for s in SEEDS}
    capped = {m: sum(1 for r in runs if r["model"] == m and r["steps"] >= MAXSTEPS and r["train"] < 1.0) for m in models}
    P_("  G5 capped unfitted: " + ", ".join(f"{label[m]} {capped[m]}/60" for m in models))
    P_("\n    seed " + "".join(f"{label[m]:>28}" for m in models))
    for s in SEEDS:
        P_(f"    {s:>4} " + "".join(f"{units[(m, s)]:>28.3f}" for m in models))

    def rule(r):
        return "HELPS" if r["wins"] >= 9 else "HARMS" if r["losses"] >= 9 else "NO CLEAR EFFECT"

    res = {}
    for key, b_, nm in (("G0", "channel", "vs channel tree (does it help the weak part?)"),
                        ("G3", "bump", "vs bump tree (does it close the gap?)")) + (
                       (("G4", "twogate_rev", "vs reversed ordering (did the data's ordering matter?)"),)
                       if ORDER != 0 else ()):
        r = nb.decide(units, "twogate", b_, SEEDS)
        r["verdict"] = rule(r)
        res[key] = r
        P_(f"\n  {key} two-gate {nm}")
        P_(f"     higher on {r['wins']}/10, lower on {r['losses']}/10, sign p {r['sign_p']:.4f}, "
           f"mean {r['mean_diff']:+.3f}, t {r['paired_t']:.2f}  -> {r['verdict']}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"order": ORDER, "decisions": res, "capped": capped,
               "units": {f"{m}|{s}": v for (m, s), v in units.items()}, "runs": runs}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/twogate.json   runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
