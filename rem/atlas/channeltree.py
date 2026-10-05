"""Channel dynamics and per-node parameters inside the dendritic tree -- the diagnosed fix, tested.

WHY THIS EXISTS. dendrite.py built a tree whose branching and depth came from Blue Brain
morphology, and it lost to a plain MLP by 0.496 at matched parameters. The diagnosis was specific:
24,576 of its 24,617 parameters sat at the LEAVES and the whole four-level hierarchy carried 32
gains, so the tree was a FIXED pooling structure. Depth that costs nothing also does nothing. And
its node nonlinearity was tanh-of-a-sum, which is MONOTONE in the local sum, while parity is not a
function of any sum.

THE TWO FIXES, AND THE SECOND IS REAL BIOPHYSICS RATHER THAN A CHOICE OF ACTIVATION.

  (1) PARAMETERS INSIDE THE TREE. Every internal node gets its own weights over its children
      instead of one shared gain per level, so the hierarchy is learnable rather than fixed.

  (2) CHANNEL DYNAMICS, WHICH SUPPLY NON-MONOTONICITY FOR FREE. A voltage-gated conductance
      contributes current g * m(v) * (E_rev - v): an activation gate m(v) = sigmoid((v - theta)/k)
      times a DRIVING FORCE that CHANGES SIGN when v crosses the reversal potential. That sign
      flip is the thing a tanh cannot do, and it is not a modelling liberty -- it is the
      Hodgkin-Huxley current equation. Two channel populations per node with fixed, opposite
      reversals (depolarising at +1, hyperpolarising at -1, in normalised units) and learnable
      conductances give a node whose output is genuinely non-monotone in its input.

      per node:  v   = sum_c w_c * child_c
                 m   = sigmoid(v - theta)
                 out = v + g_dep * m * (+1 - v) + g_hyp * m * (-1 - v)
      learnable: w (one per child), theta, g_dep, g_hyp.

AND THE FIRST SWEEP CAME BACK ENTIRELY VOID, WHICH IS RECORDED RATHER THAN QUIETLY RETUNED.
At input width 16 with parity order k in 4/6/8/10, NO model learnt ANY row -- channel tree, tanh
tree and both MLPs all sat at 0.488-0.503, and the channel tree agreed with the MLP to three
decimals INCLUDING the spread, which is the signature of every model emitting a constant. The task
coupled two hard problems: finding which k of 16 bits matter AND computing parity on them. The
learnability gate correctly returned VOID instead of a comparison.

TWO CHANGES, AND THE SECOND IS A CONTROL I SHOULD HAVE HAD FROM THE START.
  Input width drops 16 -> 8, so feature selection is near-trivial and the COMPOSITION difficulty
  that the dendrite question is actually about is what the sweep varies.
  And k = 2 is added as a HARNESS POSITIVE CONTROL: XOR of 2 of 8 bits must be learnable by a
  plain MLP. If it is not, the optimiser or the budget is broken and nothing about architectures
  can be concluded -- a distinction the first run could not make, because with every row void
  there was no way to tell an unlearnable task from a broken harness.

AND THE SWEEP DESIGN IS FIXED TOO. dendrite.py's "depth" did not order difficulty: its subset size
was min(2^depth, 12), so the deepest point used all 12 inputs and was the EASIEST, which is why the
only learnable row was the last one. Here the input width is fixed at 16 and the parity subset size
k sweeps 4, 6, 8, 10, so every point requires finding k of 16 bits and difficulty rises with k.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

C1  THE GRADIENT CHECK, BLOCKING, FIRST. The backward pass is hand-derived through a sign-flipping
    driving force and a per-node gate; that is exactly where a sign error hides and still trains to
    something plausible. Every parameter group is checked against a central finite difference.
    PREDECLARED: max relative error < 1e-5 on every group, or NO training number may be reported.
    dendrite.py had no such check, and three separate defects survived its first run.

C0  THE CEILING GATE, unchanged in spirit from dendrite.py so the two are comparable. The channel
    tree must beat BOTH an equal-parameter shallow MLP and an equal-parameter, equal-depth MLP.
    PREDECLARED: a pass on one control is not a pass. And it must beat the TANH TREE from
    dendrite.py at matched parameters, or the two fixes did not do anything.

C2  THE LEARNABILITY GATE, carried over because its absence invalidated dendrite.py's first run.
    At each k the best model must clear the majority-class baseline by 0.05 or that row is VOID and
    may not be used in any comparison.

C3  THE PARAMETER SPLIT, REPORTED. State what fraction of parameters now sit inside the hierarchy
    versus at the leaves. PREDECLARED: if the split is still leaf-dominated the first fix was not
    applied in substance, whatever the architecture diagram says.

C4  RANK BY THE SWEEP, not by one k. And report the spread over seeds.

C5  WHAT THIS IS AND IS NOT.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import json
import math
import time
import numpy as np

HERE = os.path.dirname(__file__)
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
MORPH = os.environ.get("MARKRAM_MORPH", os.path.join(HERE, "_cache", "markram_morph.json"))
OUT = os.path.join(HERE, "RESULTS_channeltree.txt")
ART = R("outputs", "channeltree.json")
RULE = "=" * 97
E_DEP, E_HYP = 1.0, -1.0


def make_task(n_bits, k, n, rng, sub=None):
    """Parity over a FIXED k-subset of n_bits, shared between train and test.

    THE BUG THIS SIGNATURE EXISTS TO FIX, RECORDED RATHER THAN QUIETLY PATCHED. The previous
    version drew the subset INSIDE the function, so the two calls that built the train and test
    sets drew DIFFERENT subsets. Every model learnt parity over one set of bits and was scored on
    parity over another, which is chance by construction. That single defect explains every void
    row in both this module and dendrite.py -- and it explains the one row that was NOT void:
    dendrite.py used k = min(2**depth, 12), so at depth 4 the subset was ALL 12 bits, the one case
    where the two draws cannot disagree. That row is therefore the only valid comparison either
    module has produced so far.

    The harness positive control is what caught it: a plain MLP with 21,151 parameters scoring
    0.504 on XOR is impossible for working code, which is a statement about the harness and not
    about any architecture. Without that control the run would have read as four more void rows."""
    X = rng.integers(0, 2, size=(n, n_bits)).astype(np.float64)
    if sub is None:
        sub = rng.choice(n_bits, size=k, replace=False)
    y = (X[:, sub].sum(axis=1) % 2).astype(np.float64)
    return X * 2 - 1, y, sub


def adam(ps, gs, st, lr):
    for i, (p, g) in enumerate(zip(ps, gs)):
        if i not in st:
            st[i] = [np.zeros_like(p), np.zeros_like(p), 0]
        m, v, t = st[i]
        t += 1
        m[...] = 0.9 * m + 0.1 * g
        v[...] = 0.999 * v + 0.001 * g * g
        st[i][2] = t
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


class ChannelTree:
    """U neurons, each a b-ary tree of depth L with per-node weights and per-node channels."""

    def __init__(self, D, U, b, L, rng):
        self.D, self.U, self.b, self.L = D, U, b, L
        self.K = b ** L
        self.Wl = rng.normal(0, 1 / math.sqrt(D), (U, self.K, D))      # leaves (synapses)
        self.W, self.th, self.gd, self.gh = [], [], [], []
        M = self.K
        for _ in range(L):
            M //= b
            self.W.append(rng.normal(0, 1 / math.sqrt(b), (U, M, b)))
            self.th.append(np.zeros((U, M)))
            self.gd.append(np.full((U, M), 0.5))
            self.gh.append(np.full((U, M), 0.5))
        self.wo = rng.normal(0, 1 / math.sqrt(U), U)
        self.bo = np.zeros(1)
        self.ps = [self.Wl] + self.W + self.th + self.gd + self.gh + [self.wo, self.bo]

    def split(self):
        leaf = self.Wl.size
        inner = sum(x.size for x in self.W + self.th + self.gd + self.gh)
        return leaf, inner, self.wo.size + self.bo.size

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.Wl)
        self.cX, self.cH, self.cV, self.cM = X, [h], [], []
        for l in range(self.L):
            ch = h.reshape(n, self.U, -1, self.b)
            v = (ch * self.W[l][None]).sum(-1)                          # (n,U,M)
            m = 1.0 / (1.0 + np.exp(-(v - self.th[l][None])))
            h = v + self.gd[l][None] * m * (E_DEP - v) + self.gh[l][None] * m * (E_HYP - v)
            self.cV.append(v); self.cM.append(m); self.cH.append(h)
        s = h[:, :, 0]
        self.s = s
        return s @ self.wo + self.bo

    def backward(self, dz):
        n = self.cX.shape[0]
        gwo = self.s.T @ dz
        gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]                          # (n,U,1)
        gW = [None] * self.L; gth = [None] * self.L
        ggd = [None] * self.L; ggh = [None] * self.L
        for l in range(self.L - 1, -1, -1):
            v, m = self.cV[l], self.cM[l]
            gd, gh = self.gd[l][None], self.gh[l][None]
            ggd[l] = (dh * m * (E_DEP - v)).sum(0)
            ggh[l] = (dh * m * (E_HYP - v)).sum(0)
            dm = dh * (gd * (E_DEP - v) + gh * (E_HYP - v))
            mp = m * (1 - m)
            gth[l] = (dm * (-mp)).sum(0)
            dv = dh * (1 - gd * m - gh * m) + dm * mp
            ch = self.cH[l].reshape(n, self.U, -1, self.b)
            gW[l] = (dv[:, :, :, None] * ch).sum(0)
            dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, self.U, -1)
        gWl = np.einsum('nuk,nd->ukd', dh, self.cX)
        return [gWl] + gW + gth + ggd + ggh + [gwo, gbo]


class TanhTree:
    """dendrite.py's unit, carried over verbatim in spirit as the third control."""

    def __init__(self, D, U, b, L, rng):
        self.D, self.U, self.b, self.L = D, U, b, L
        self.K = b ** L
        self.W = rng.normal(0, 1 / math.sqrt(D), (U, self.K, D))
        self.g = np.ones((U, L))
        self.wo = rng.normal(0, 1 / math.sqrt(U), U)
        self.bo = np.zeros(1)
        self.ps = [self.W, self.g, self.wo, self.bo]

    def forward(self, X):
        n = X.shape[0]
        h = np.einsum('nd,ukd->nuk', X, self.W)
        self.c = [h]; self.X = X
        for l in range(self.L):
            h = np.tanh(h.reshape(n, self.U, -1, self.b).sum(-1)) * self.g[:, l][None, :, None]
            self.c.append(h)
        self.s = h[:, :, 0]
        return self.s @ self.wo + self.bo

    def backward(self, dz):
        n = self.X.shape[0]
        gwo = self.s.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)[:, :, None]
        gg = np.zeros_like(self.g)
        for l in range(self.L - 1, -1, -1):
            pre = self.c[l].reshape(n, self.U, -1, self.b).sum(-1)
            t = np.tanh(pre)
            gg[:, l] = (dh * t).sum(axis=(0, 2))
            dpre = dh * self.g[:, l][None, :, None] * (1 - t ** 2)
            dh = np.repeat(dpre[:, :, :, None], self.b, axis=3).reshape(n, self.U, -1)
        return [np.einsum('nuk,nd->ukd', dh, self.X), gg, gwo, gbo]


class MLP:
    def __init__(self, D, hs, rng):
        self.Ws, self.bs = [], []
        prev = D
        for h in hs:
            self.Ws.append(rng.normal(0, 1 / math.sqrt(prev), (prev, h)))
            self.bs.append(np.zeros(h)); prev = h
        self.Ws.append(rng.normal(0, 1 / math.sqrt(prev), (prev, 1)))
        self.bs.append(np.zeros(1))
        self.ps = self.Ws + self.bs

    def forward(self, X):
        self.a = [X]; h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.tanh(h @ W + b); self.a.append(h)
        return (h @ self.Ws[-1] + self.bs[-1])[:, 0]

    def backward(self, dz):
        gW = [None] * len(self.Ws); gb = [None] * len(self.bs)
        d = dz[:, None]
        gW[-1] = self.a[-1].T @ d; gb[-1] = d.sum(0)
        d = d @ self.Ws[-1].T
        for i in range(len(self.Ws) - 2, -1, -1):
            d = d * (1 - self.a[i + 1] ** 2)
            gW[i] = self.a[i].T @ d; gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.Ws[i].T
        return gW + gb


def nparams(m):
    return sum(p.size for p in m.ps)


def loss_of(m, X, y):
    z = m.forward(X)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def train(m, Xtr, ytr, Xte, yte, steps, lr, bs, rng):
    st = {}; n = Xtr.shape[0]
    for _ in range(steps):
        i = rng.integers(0, n, bs)
        z = m.forward(Xtr[i])
        dz = (1 / (1 + np.exp(-z)) - ytr[i]) / bs
        adam(m.ps, m.backward(dz), st, lr)
    return float(((m.forward(Xte) > 0).astype(float) == yte).mean())


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    P_(RULE); P_("CHANNEL DYNAMICS AND PER-NODE PARAMETERS INSIDE THE DENDRITIC TREE"); P_(RULE)
    M = json.load(open(MORPH))
    by = collections.defaultdict(list)
    for m in M:
        by[m["label"]].append(m)
    pyr = by["pyramidal"]
    stems = float(np.median([x["n_stems"] for x in pyr]))
    branch = float(np.median([x["n_branch"] for x in pyr]))
    b = max(2, int(round(stems / 2)))
    L = max(2, min(4, int(round(math.log(branch, max(b, 2))))))
    P_(f"  morphology: {len(M)} Markram-lab cells; pyramidal median n_stems {stems:.0f},"
       f" n_branch {branch:.0f}")
    P_(f"  tree from morphology: branching b = {b}, depth L = {L}, leaves {b**L}")

    # ---- C1  BLOCKING gradient check -------------------------------------------------------
    P_("\n" + RULE); P_("C1  GRADIENT CHECK, HAND-DERIVED BACKWARD (BLOCKING)"); P_(RULE)
    rng = np.random.default_rng(0)
    net = ChannelTree(6, 3, 2, 2, rng)
    X = rng.normal(size=(7, 6)); y = rng.integers(0, 2, 7).astype(float)
    z = net.forward(X)
    gs = net.backward((1 / (1 + np.exp(-z)) - y) / len(y))
    names = (["Wl"] + [f"W[{i}]" for i in range(L and 2)] + [f"th[{i}]" for i in range(2)]
             + [f"gd[{i}]" for i in range(2)] + [f"gh[{i}]" for i in range(2)] + ["wo", "bo"])
    worst = 0.0
    for nm, p, g in zip(names, net.ps, gs):
        idx = tuple(rng.integers(0, s) for s in p.shape)
        e = 1e-6
        o = p[idx]
        p[idx] = o + e; lp = loss_of(net, X, y)
        p[idx] = o - e; lm = loss_of(net, X, y)
        p[idx] = o
        num = (lp - lm) / (2 * e)
        rel = abs(num - g[idx]) / max(abs(num), abs(g[idx]), 1e-9)
        worst = max(worst, rel)
        P_(f"    {nm:<8} analytic {g[idx]:+.6e}  numeric {num:+.6e}  rel {rel:.2e}")
    P_(f"\n  worst relative error {worst:.2e}")
    if worst >= 1e-5:
        P_("\n  C1: FAIL -- the hand-derived backward pass is wrong. Nothing reported.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("\n  C1: PASS -- every parameter group matches a central finite difference")

    # ---- C3  parameter split ----------------------------------------------------------------
    P_("\n" + RULE); P_("C3  WHERE THE PARAMETERS NOW SIT"); P_(RULE)
    D, U, STEPS, LR, BS, NTR, NTE = 8, 8, 12000, 0.01, 128, 8000, 4000
    probe = ChannelTree(D, U, b, L, np.random.default_rng(1))
    lf, inn, hd = probe.split()
    tot = lf + inn + hd
    P_(f"  leaves (synapses)      {lf:>8,}  {100*lf/tot:>5.1f}%")
    P_(f"  INSIDE the hierarchy   {inn:>8,}  {100*inn/tot:>5.1f}%   (dendrite.py: 32 = 0.1%)")
    P_(f"  readout                {hd:>8,}  {100*hd/tot:>5.1f}%")
    P_(f"  total                  {tot:>8,}")
    P_(f"\n  C3: the hierarchy holds {100*inn/tot:.1f}% of parameters against 0.1% before."
       f" {'Applied in substance.' if inn/tot > 0.05 else 'STILL leaf-dominated -- fix (1) not applied in substance.'}")

    # ---- C0 / C2 / C4 ------------------------------------------------------------------------
    P_("\n" + RULE); P_("C0 + C2 + C4  THE CEILING GATE, SWEPT OVER PARITY ORDER k"); P_(RULE)
    P_(f"  inputs {D} (fixed), neurons {U}, steps {STEPS}, {NTR} train / {NTE} test, 3 seeds")
    P_(f"\n    {'k':>3} {'channel tree':>17} {'tanh tree':>17} {'MLP eq-param':>17}"
       f" {'MLP eq-par+depth':>17} {'learnt':>7}")
    rows = []
    for k in (2, 3, 4, 5, 6):
        acc = collections.defaultdict(list)
        for seed in range(3):
            r1 = np.random.default_rng(100 + seed)
            Xtr, ytr, sub = make_task(D, k, NTR, r1)
            Xte, yte, _ = make_task(D, k, NTE, r1, sub=sub)   # SAME subset, which was the bug
            ct = ChannelTree(D, U, b, L, np.random.default_rng(200 + seed))
            P0 = nparams(ct)
            tt = TanhTree(D, U, b, L, np.random.default_rng(200 + seed))
            h1 = max(2, int(round((P0 - 1) / (D + 2))))
            m1 = MLP(D, [h1], np.random.default_rng(200 + seed))
            hd2 = max(2, int(round((math.sqrt((D + L) ** 2 + 4 * (L - 1) * P0) - (D + L))
                                   / (2 * (L - 1)))))
            m2 = MLP(D, [hd2] * L, np.random.default_rng(200 + seed))
            for nm, mdl in (("ct", ct), ("tt", tt), ("m1", m1), ("m2", m2)):
                acc[nm].append(train(mdl, Xtr, ytr, Xte, yte, STEPS, LR, BS,
                                     np.random.default_rng(300 + seed)))
        maj = float(max(ytr.mean(), 1 - ytr.mean()))
        best = max(np.mean(acc[x]) for x in acc)
        learnt = best > maj + 0.05
        f = lambda x: f"{np.mean(acc[x]):.3f}+/-{np.std(acc[x]):.3f}"
        P_(f"    {k:>3} {f('ct'):>17} {f('tt'):>17} {f('m1'):>17} {f('m2'):>17}"
           f" {'yes' if learnt else 'NO':>7}")
        rows.append(dict(k=k, ct=np.mean(acc["ct"]), tt=np.mean(acc["tt"]),
                         m1=np.mean(acc["m1"]), m2=np.mean(acc["m2"]), learnt=learnt,
                         params=dict(ct=nparams(ct), tt=nparams(tt),
                                     m1=nparams(m1), m2=nparams(m2))))
    pp = rows[0]["params"]
    P_(f"\n  parameters: channel tree {pp['ct']:,}  tanh tree {pp['tt']:,}"
       f"  MLP {pp['m1']:,}  MLP deep {pp['m2']:,}")
    ctrl = [r for r in rows if r["k"] == 2]
    harness_ok = bool(ctrl) and ctrl[0]["m1"] > 0.90
    P_(f"\n  HARNESS POSITIVE CONTROL (k=2, XOR of 2 of {D} bits, plain MLP):"
       f" {ctrl[0]['m1']:.3f}" if ctrl else "")
    P_(f"  {'PASS -- the optimiser and budget can learn a 2-bit composition, so a void row above'  if harness_ok else 'FAIL -- a plain MLP cannot even learn XOR here, so the HARNESS is broken'}")
    P_(f"  {'means the task is hard, not that training is broken.' if harness_ok else 'and no statement about architectures can be made from this run at all.'}")
    if not harness_ok:
        P_("\n  C0: VOID -- harness control failed. Untested, and the fault is mine, not the models'.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    live = [r for r in rows if r["learnt"] and r["k"] > 2]
    P_(f"\n  C2 LEARNABILITY: {len(live)} of {len(rows)-1} non-control k values learnt by any model."
       f" {'Only those are used.' if live else ''}")
    if not live:
        P_("\n  C0: VOID -- nothing was learnt at any k. Untested, not answered.")
        open(OUT, "w").write("\n".join(out) + "\n"); return
    wt = sum(1 for r in live if r["ct"] > r["tt"] + 0.01)
    w1 = sum(1 for r in live if r["ct"] > r["m1"] + 0.01)
    w2 = sum(1 for r in live if r["ct"] > r["m2"] + 0.01)
    P_(f"\n  channel tree beats the TANH TREE at      {wt}/{len(live)} learnt k"
       f"   <- did the two fixes do anything?")
    P_(f"  channel tree beats equal-param MLP at    {w1}/{len(live)}")
    P_(f"  channel tree beats equal-param+depth MLP {w2}/{len(live)}")
    passed = w1 == len(live) and w2 == len(live)
    P_(f"\n  C0: {'PASS' if passed else 'FAIL'} -- "
       f"{'the channel tree beats both conventional controls everywhere it was tested' if passed else 'it does NOT beat both conventional controls, so the biological neuron still is not carrying the result'}")
    if wt == len(live):
        P_("  BUT THE FIXES DID WORK: the channel tree beats dendrite.py's tanh tree at every")
        P_("  learnt k, so parameters-inside-the-tree plus a sign-flipping driving force is a real")
        P_("  improvement over the previous unit -- just not enough to beat a plain MLP.")
    elif wt:
        P_(f"  The fixes helped at {wt} of {len(live)} k values but not uniformly.")
    else:
        P_("  AND THE FIXES DID NOT HELP EITHER: the channel tree does not beat the tanh tree it")
        P_("  was built to replace, so the diagnosis behind it was wrong.")
    P_("\n  C4 THE SWEEP, which ranks the approach rather than one point:")
    for r in rows:
        tag = "" if r["learnt"] else "   (VOID, nothing learnt)"
        P_(f"    k={r['k']}: ct-tanh {r['ct']-r['tt']:+.3f}   ct-MLP {r['ct']-r['m1']:+.3f}"
           f"   ct-MLPdeep {r['ct']-r['m2']:+.3f}{tag}")

    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"morphology": {"b": b, "L": L, "leaves": b ** L, "n_cells": len(M)},
               "param_split": {"leaves": lf, "hierarchy": inn, "readout": hd},
               "sweep": rows, "verdict": "PASS" if passed else "FAIL"}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/channeltree.json")

    P_("\n" + RULE); P_("C5  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  1. Still not reasoning. Parity is compositional, not cognitive.")
    P_("  2. The channel model is the HH current equation with two fixed reversals and a learnable")
    P_("     gate. It has no TIME: no activation kinetics, no inactivation, no calcium. A real")
    P_("     NMDA plateau is a temporal event and nothing here has a time axis at all.")
    P_("  3. Morphology sets only branching and depth. Real dendrites differ in diameter, length")
    P_("     and channel density along the tree, none of which is used.")
    P_("  4. Rat somatosensory cortex. Nothing about human is claimed.")
    P_("  5. Four seeds and four k values. The spread is printed so the reader can see whether the")
    P_("     gaps exceed it.")
    P_(f"\n  runtime {time.time()-t0:.1f}s")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
