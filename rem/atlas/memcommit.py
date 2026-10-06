"""Give the network memory, then test it against the mouse's optogenetic commitment curve.

WHY. Reasoning needs a held intermediate conclusion that later irrelevant input cannot overwrite.
optocommit.py measured how a mouse does this (DANDI 000060, 9 mice, 26,746 trials): a light pulse in
touch cortex identical to the decision cue fools the animal when it arrives early in the memory delay
and much less when it arrives late. That falling curve is the target.

THE MEMORY NETWORK. N = 32 rate units in a loop, dt = 0.1 s, leak alpha = 0.5 (tau = 0.2 s),
    v_t = W h_{t-1} + U x_t + b,   h_t = (1 - alpha) h_{t-1} + alpha f(v_t) + noise (sd 0.1)
read out at the go cue. f is one of the node types already tested in the tree:
    tanh       conventional, no node parameters
    channel    the gated-conductance node   v + g_dep m (1 - v) + g_hyp m (-1 - v), m = sigmoid(v - theta)
    bump       the generic bend that won     v + a exp(-(v - mu)^2 / (s^2 + 0.01))
Disclosed: point units, not dendritic trees (a one-bit memory does not need the branching); tanh has
96 fewer parameters (1,185 vs 1,281).

THE TRIAL, from the dataset's own event times (seconds relative to the go cue): trial start -4.7,
sample-start chirp -3.00, sample light pulse -2.5 (0.4 s), sample-end chirp -2.15, delay to 0, go 0.
Inputs: [light amplitude (Full 1, Mini 1/3 -- the power ratio), chirp, go]. Distractors as the mice
received them: -3.8 (before the sample window), -2.5 Mini, -1.6 (early delay), -0.8 (late delay).

THE MOUSE TARGET (outputs/opto_commitment.json, committed before this file): P(lick right) on
no-sample trials: none 0.186; Full -3.8 0.610, -1.6 0.578, -0.8 0.322; Mini -3.8 0.214, -2.5 0.337,
-1.6 0.317, -0.8 0.179. COMMITMENT INDEX = fooling(-1.6 Full) - fooling(-0.8 Full) = +0.256.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
H0 VALID, BLOCKING. For each node type: BPTT gradient |num - ana| <= 1e-5 max + 1e-9 on sampled
   entries; a deliberately broken copy (input-weight gradient dropped at the first step) MUST fail.
H1 HARNESS, BLOCKING. The same probe code on two hand-built memories, 10 noise seeds each:
   a COMMITTING latch (pulse effectiveness falls linearly to 0 across the delay) must give COMMITS;
   a PURE latch (any pulse, any time, sets the memory) must NOT.
H2 LEARNABILITY. A run that does not reach its accuracy bar within 4,000 steps is counted and
   excluded from its model's verdict.

REGIMES (10 seeds per model each, seeds 0-9 for init, distinct noise streams):
   A  ZERO-SHOT: trained only on sample-vs-none trials (never a distractor) to basic accuracy >= 0.95.
      Asks whether the memory COMMITS ON ITS OWN.
   B  MOUSE EXPERIENCE: trial types drawn in the mice's proportions (kept trials of DANDI 000060),
      labels = the dataset's trial_instruction; training stops when basic accuracy reaches the mice's
      0.82. Asks what a network trained like the mouse, to the mouse's skill, does.
   Probe: 2,000 noisy trials per condition, the mouse's 14 conditions (no-sample and sample trials).

C1 PER MODEL AND REGIME. COMMITS if the commitment index > 0 on >= 9/10 seeds (sign p 0.011);
   ANTI-COMMITS if < 0 on >= 9/10; else NO COMMITMENT. A model fooled by nothing (mean fooling at
   -1.6 Full < 0.05) is reported as IGNORES DISTRACTORS, whatever its index.
C2 MOUSE-LIKENESS. Per seed, mean absolute error between the model's 14 P(right) values and the
   mouse's. Between node types, paired per seed: closer on >= 9/10 -> CLOSER TO THE MOUSE.
C3 DESCRIPTIVE. The full fooling curve on a fine grid of pulse times (-4.0 to -0.4 s), and fooling
   by the pre-sample pulse (the mouse: +0.42 at Full).
C4 WHAT THIS IS AND IS NOT. One task, one noise level, one leak, chosen once and not swept. Matching
   a mouse is not the same as reasoning well: a network that ignores every distractor is a BETTER
   memory than the mouse and a WORSE model of it. Both are reported; neither is hidden.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import importlib.util
import json
import math
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_memcommit.txt")
ART = R("outputs", "memcommit.json")
MOUSE = R("outputs", "opto_commitment.json")
RULE = "=" * 97

DT, T0, ALPHA, NOISE, N = 0.1, -4.7, 0.5, 0.1, 32
T = 48                                  # steps k = 0..47, time T0 + 0.1 k; go cue at k = 47
LR, BS, CAP, EVERY = 0.003, 128, 4000, 50
CLIP = 1.0
MINI = 1.0 / 3.0
SEEDS = tuple(range(10))
MODELS = ("tanh", "channel", "bump")
NEVAL = 2000
CONDS = [(None, None), (3.8, "Full"), (3.8, "Mini"), (2.5, "Mini"), (1.6, "Full"), (1.6, "Mini"),
         (0.8, "Full"), (0.8, "Mini")]


def step_of(t):
    return int(math.floor((t - T0) / DT + 0.5))


def trial_inputs(sample, dt=None, inten=None):
    """One trial's input sequence (T, 3)."""
    x = np.zeros((T, 3))
    for k in (step_of(-3.00), step_of(-2.15)):
        x[k, 1] = 1.0
    x[step_of(0.0), 2] = 1.0
    if sample:
        k = step_of(-2.5); x[k:k + 4, 0] += 1.0
    if dt is not None:
        k = step_of(-dt); x[k:k + 4, 0] += 1.0 if inten == "Full" else MINI
    return x


class MemNet:
    def __init__(self, node, rng, n=N):
        self.node, self.n = node, n
        self.W = rng.normal(0, 0.9 / math.sqrt(n), (n, n))
        self.U = rng.normal(0, 1.0, (n, 3))
        self.b = np.zeros(n)
        if node == "channel":
            self.q = [np.full(n, 0.5), np.full(n, 0.5), np.zeros(n)]
        elif node == "bump":
            self.q = [np.full(n, 0.5), np.zeros(n), np.ones(n)]
        else:
            self.q = []
        self.wo = rng.normal(0, 1 / math.sqrt(n), n)
        self.bo = np.zeros(1)
        self.ps = [self.W, self.U, self.b] + self.q + [self.wo, self.bo]

    def f(self, v):
        if self.node == "tanh":
            a = np.tanh(v); return a, (a,)
        if self.node == "channel":
            gd, gh, th = self.q
            m = 1 / (1 + np.exp(-(v - th)))
            return v + gd * m * (1 - v) + gh * m * (-1 - v), (v, m)
        A, mu, s = self.q
        Dn = s * s + 0.01
        q = (v - mu) ** 2 / Dn
        e = np.exp(-q)
        return v + A * e, (v, e, q, Dn)

    def fb(self, da, c):
        if self.node == "tanh":
            (a,) = c; return da * (1 - a * a), []
        if self.node == "channel":
            gd, gh, th = self.q
            v, m = c
            mp_ = m * (1 - m)
            dv = da * (1 - (gd + gh) * m + ((gd - gh) - (gd + gh) * v) * mp_)
            return dv, [(da * m * (1 - v)).sum(0), (da * m * (-1 - v)).sum(0),
                        (-da * mp_ * (gd * (1 - v) + gh * (-1 - v))).sum(0)]
        A, mu, s = self.q
        v, e, q, Dn = c
        k = A * e * 2 * (v - mu) / Dn
        return da * (1 - k), [(da * e).sum(0), (da * k).sum(0), (da * A * e * q * 2 * s / Dn).sum(0)]

    def forward(self, X, noise=None):
        B = X.shape[0]
        h = np.zeros((B, self.n))
        self.cache = []
        for t in range(T):
            v = h @ self.W.T + X[:, t] @ self.U.T + self.b
            a, c = self.f(v)
            hn = (1 - ALPHA) * h + ALPHA * a
            if noise is not None:
                hn = hn + noise[:, t]
            self.cache.append((h, X[:, t], c))
            h = hn
        self.hT = h
        return h @ self.wo + self.bo

    def first_step_scale(self):
        return 1.0

    def backward(self, dz):
        gwo = self.hT.T @ dz; gbo = dz.sum(keepdims=True)
        dh = np.outer(dz, self.wo)
        gW = np.zeros_like(self.W); gU = np.zeros_like(self.U); gb = np.zeros_like(self.b)
        gq = [np.zeros_like(p) for p in self.q]
        for t in range(T - 1, -1, -1):
            hp, xt, c = self.cache[t]
            dv, dq = self.fb(ALPHA * dh, c)
            gW += dv.T @ hp
            gU += (dv.T @ xt) * (self.first_step_scale() if t == 0 else 1.0)
            gb += dv.sum(0)
            for i, d in enumerate(dq):
                gq[i] += d
            dh = (1 - ALPHA) * dh + dv @ self.W
        return [gW, gU, gb] + gq + [gwo, gbo]


class BrokenMemNet(MemNet):
    """H0 negative control: the input-weight gradient of the first step is dropped."""

    def first_step_scale(self):
        return 0.0


def gradcheck(node, broken=False, seed=3):
    rng = np.random.default_rng(seed)
    net = (BrokenMemNet if broken else MemNet)(node, rng, n=5)
    for p in net.q:
        p[...] = rng.normal(0.4, 0.3, p.shape)
    X = np.stack([trial_inputs(bool(i % 2), *( (1.6, "Full") if i % 3 == 0 else (None, None))) for i in range(6)])
    X[:, 0, :] += rng.normal(0, 0.5, (6, 3))          # make the first step's input gradient non-trivial
    y = rng.integers(0, 2, 6).astype(float)

    def loss():
        z = net.forward(X)
        return float(np.mean(np.logaddexp(0, z) - y * z))

    z = net.forward(X); gs = net.backward((1 / (1 + np.exp(-z)) - y) / len(y))
    ratio = 0.0
    for p, g in zip(net.ps, gs):
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = loss()
            p[idx] = o - 1e-6; lm = loss()
            p[idx] = o
            num = (lp - lm) / 2e-6
            ratio = max(ratio, abs(num - g[idx]) / (1e-5 * max(abs(num), abs(g[idx])) + 1e-9))
    return ratio


def adam_clip(ps, gs, st, lr):
    nrm = math.sqrt(sum(float((g * g).sum()) for g in gs))
    if nrm > CLIP:
        gs = [g * (CLIP / nrm) for g in gs]
    for i, (p, g) in enumerate(zip(ps, gs)):
        if i not in st:
            st[i] = [np.zeros_like(p), np.zeros_like(p), 0]
        m, v, t = st[i]
        t += 1
        m[...] = 0.9 * m + 0.1 * g
        v[...] = 0.999 * v + 0.001 * g * g
        st[i][2] = t
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


def mouse_mix():
    """Kept-trial proportions by (instruction, distractor time, intensity), as optocommit kept them."""
    s = importlib.util.spec_from_file_location("optocommit", os.path.join(HERE, "optocommit.py"))
    oc = importlib.util.module_from_spec(s); s.loader.exec_module(oc)
    man = json.load(open(os.path.join(oc.CACHE, "manifest.json")))
    rows, _, _ = oc.load(man)
    c = collections.Counter()
    for r in rows:
        want = sorted(([2.5] if r["side"] == "r" else []) + ([r["dt"]] if r["dt"] is not None else []))
        got = sorted(r["onsets"])
        if not (len(want) == len(got) and all(abs(a - b) <= 0.02 for a, b in zip(want, got))):
            continue
        if r["outcome"] in ("hit", "miss") and r["early"] == "no early" and r["instr"] in ("left", "right"):
            c[(r["instr"], r["dt"], r["inten"])] += 1
    keys = sorted(c, key=str)
    p = np.array([c[k] for k in keys], dtype=float)
    return keys, p / p.sum()


MIX = None


def batch(rng, regime, bs=BS):
    if regime == "A":
        ys = rng.integers(0, 2, bs)
        X = np.stack([trial_inputs(bool(y)) for y in ys])
        return X, ys.astype(float)
    keys, p = MIX
    idx = rng.choice(len(keys), size=bs, p=p)
    X, ys = [], []
    for i in idx:
        instr, dt, inten = keys[i]
        X.append(trial_inputs(instr == "right", dt, inten)); ys.append(1.0 if instr == "right" else 0.0)
    return np.stack(X), np.array(ys)


class Latch:
    """Hand-built memories for H1. committing: pulse weight falls linearly from 1 at -2.0 to 0 at 0."""

    def __init__(self, committing):
        self.committing = committing

    def forward(self, X, noise=None):
        tt = T0 + DT * np.arange(T)
        w = np.ones(T)
        if self.committing:
            w = np.where(tt < -2.0, 1.0, np.clip(-tt / 2.0, 0, 1))
        drive = (X[:, :, 0] * w[None]).sum(1) / 4.0
        nz = 0.0 if noise is None else noise[:, :, 0].sum(1) * 0.5
        return 6 * (np.minimum(drive, 1.0) + nz - 0.5)


def probe(m, rng, fine=False):
    """P(lick right) on the mouse's conditions (and optionally a fine grid of Full pulse times)."""
    out = {}
    for sample in (False, True):
        for dt, it in CONDS:
            if sample and dt == 2.5:
                continue
            X = np.repeat(trial_inputs(sample, dt, it)[None], NEVAL, 0)
            z = m.forward(X, rng.normal(0, NOISE, (NEVAL, T, N)))
            out[(sample, dt, it)] = float((z > 0).mean())
    if fine:
        grid = {}
        for t in np.round(np.arange(4.0, 0.39, -0.2), 2):
            X = np.repeat(trial_inputs(False, float(t), "Full")[None], NEVAL, 0)
            z = m.forward(X, rng.normal(0, NOISE, (NEVAL, T, N)))
            grid[float(t)] = float((z > 0).mean()) - out[(False, None, None)]
        out["grid"] = grid
    return out


def basic_acc(m, rng, n=1000):
    ys = np.repeat([0, 1], n // 2)
    X = np.stack([trial_inputs(bool(y)) for y in ys])
    z = m.forward(X, rng.normal(0, NOISE, (n, T, N)))
    return float(((z > 0) == (ys > 0)).mean())


def train(args):
    node, seed, regime = args
    rng = np.random.default_rng(1000 * (1 + MODELS.index(node)) + seed + (0 if regime == "A" else 50000))
    m = MemNet(node, np.random.default_rng(seed))
    bar = 0.95 if regime == "A" else 0.82
    st, steps, acc = {}, 0, 0.0
    while steps < CAP:
        X, y = batch(rng, regime)
        z = m.forward(X, rng.normal(0, NOISE, (BS, T, N)))
        if not np.all(np.isfinite(z)):
            break
        adam_clip(m.ps, m.backward((1 / (1 + np.exp(-z)) - y) / BS), st, LR)
        steps += 1
        if steps % EVERY == 0:
            acc = basic_acc(m, np.random.default_rng(seed + 777))
            if acc >= bar:
                break
    learnt = acc >= bar
    pr = probe(m, np.random.default_rng(seed + 999), fine=True) if learnt else None
    return dict(node=node, seed=seed, regime=regime, steps=steps, basic_acc=acc, learnt=learnt, probe=pr)


def mouse_target():
    d = json.load(open(MOUSE))["curve"]
    tgt = {(False, None, None): d["left|none"]["pooled"], (True, None, None): d["right|none"]["pooled"]}
    for dt, it in CONDS[1:]:
        e = d[f"by_intensity|-{dt:.1f} s|{it}"]
        tgt[(False, dt, it)] = e["left"][0]
        if dt != 2.5:
            tgt[(True, dt, it)] = e["right"][0]
    return tgt


def index_of(pr):
    b = pr[(False, None, None)]
    return (pr[(False, 1.6, "Full")] - b) - (pr[(False, 0.8, "Full")] - b)


def verdict(vals, fooled):
    w = sum(v > 0 for v in vals); l = sum(v < 0 for v in vals)
    v = "COMMITS" if w >= 9 else "ANTI-COMMITS" if l >= 9 else "NO COMMITMENT"
    if fooled < 0.05:
        v = "IGNORES DISTRACTORS (" + v + " by the index)"
    return w, l, v


def main():
    global MIX
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    tgt = mouse_target()
    P_(RULE); P_("MEMORY FOR THE NETWORK, TESTED AGAINST THE MOUSE'S OPTOGENETIC COMMITMENT CURVE"); P_(RULE)
    ok = True
    for nd in MODELS:
        g, bg = gradcheck(nd), gradcheck(nd, broken=True)
        P_(f"  H0 {nd:<8} gradient / tolerance {g:.3f}   broken copy {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
        ok &= g <= 1.0 and bg > 1.0
    if not ok:
        P_("  H0: FAIL"); open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("  H0: PASS")
    h1 = {}
    for nm, lt in (("committing latch", Latch(True)), ("pure latch", Latch(False))):
        vals = [index_of(probe(lt, np.random.default_rng(s))) for s in range(10)]
        w, l, v = verdict(vals, 1.0)
        h1[nm] = v
        P_(f"  H1 {nm:<17} index mean {np.mean(vals):+.3f}, positive on {w}/10 -> {v}")
    if h1["committing latch"] != "COMMITS" or h1["pure latch"] == "COMMITS":
        P_("  H1: FAIL -- the probe cannot tell commitment from its absence."); open(OUT, "w").write("\n".join(out) + "\n"); return
    P_("  H1: PASS")

    MIX = mouse_mix()
    P_(f"\n  regime B trial mix from the mice ({len(MIX[0])} trial types): " +
       ", ".join(f"{k[0][0]}{'' if k[1] is None else ' -%.1f %s' % (k[1], k[2])}:{p:.3f}" for k, p in zip(*MIX)))
    pool = mp.get_context("fork").Pool(4)
    runs = pool.map(train, [(nd, s, rg) for rg in ("A", "B") for nd in MODELS for s in SEEDS])
    pool.close()

    keys14 = sorted(tgt, key=str)
    res = {}
    for rg in ("A", "B"):
        P_("\n" + RULE)
        P_(f"REGIME {rg}: " + ("ZERO-SHOT (never trained on a distractor), bar 0.95"
                              if rg == "A" else "MOUSE EXPERIENCE (mice's trial mix and labels), stopped at the mice's 0.82"))
        P_(RULE)
        P_("  P(lick right), no-sample trials, mean over learnt seeds")
        P_(f"    {'':<22}" + "".join(f"{('none' if dt is None else '-%.1f %s' % (dt, it)):>12}" for dt, it in CONDS))
        P_(f"    {'MOUSE':<22}" + "".join(f"{tgt[(False, dt, it)]:>12.3f}" for dt, it in CONDS))
        for nd in MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd and r["learnt"]]
            if not rs:
                P_(f"    {nd:<22} no seed learnt"); continue
            P_(f"    {nd + ' (' + str(len(rs)) + '/10 learnt)':<22}" + "".join(
                f"{np.mean([r['probe'][(False, dt, it)] for r in rs]):>12.3f}" for dt, it in CONDS))
        P_("\n  C1 COMMITMENT INDEX = fooling(-1.6 Full) - fooling(-0.8 Full)   (mouse +0.256)")
        for nd in MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd]
            lr_ = [r for r in rs if r["learnt"]]
            if len(lr_) < 9:
                res[(rg, nd)] = dict(verdict="NOT ENOUGH LEARNT SEEDS", learnt=len(lr_))
                P_(f"    {nd:<8} learnt {len(lr_)}/10 -> NOT ENOUGH LEARNT SEEDS (H2)"); continue
            vals = [index_of(r["probe"]) for r in lr_]
            fooled = float(np.mean([r["probe"][(False, 1.6, "Full")] - r["probe"][(False, None, None)] for r in lr_]))
            w, l, v = verdict(vals, fooled)
            mae = [float(np.mean([abs(r["probe"][k] - tgt[k]) for k in keys14])) for r in lr_]
            res[(rg, nd)] = dict(verdict=v, wins=w, losses=l, index_mean=float(np.mean(vals)), fooled_early=fooled,
                                 mae=mae, mae_mean=float(np.mean(mae)), learnt=len(lr_),
                                 steps_median=float(np.median([r["steps"] for r in lr_])),
                                 presample_full=float(np.mean([r["probe"][(False, 3.8, "Full")] - r["probe"][(False, None, None)] for r in lr_])))
            P_(f"    {nd:<8} index {np.mean(vals):+.3f} (positive {w}/{len(lr_)})  early fooling {fooled:+.3f}  "
               f"pre-sample fooling {res[(rg, nd)]['presample_full']:+.3f}  MAE to mouse {np.mean(mae):.3f}  -> {v}")
        P_("\n  C2 CLOSER TO THE MOUSE? (paired per seed, lower MAE on >= 9/10)")
        for a_, b_ in (("bump", "channel"), ("bump", "tanh"), ("channel", "tanh")):
            A_ = {r["seed"]: r for r in runs if r["regime"] == rg and r["node"] == a_ and r["learnt"]}
            B_ = {r["seed"]: r for r in runs if r["regime"] == rg and r["node"] == b_ and r["learnt"]}
            sd = sorted(set(A_) & set(B_))
            if len(sd) < 9:
                P_(f"    {a_} vs {b_}: fewer than 9 shared learnt seeds"); continue
            ma = lambda r: float(np.mean([abs(r["probe"][k] - tgt[k]) for k in keys14]))
            d = [ma(A_[s]) - ma(B_[s]) for s in sd]
            w = sum(x < 0 for x in d); l = sum(x > 0 for x in d)
            P_(f"    {a_} vs {b_}: {a_} closer on {w}/{len(sd)}, mean MAE diff {np.mean(d):+.3f}  -> "
               + (f"{a_.upper()} CLOSER" if w >= 9 else f"{b_.upper()} CLOSER" if l >= 9 else "NO CLEAR DIFFERENCE"))
        P_("\n  C3 FINE FOOLING CURVE, Full pulse on no-sample trials (mean over learnt seeds), seconds before go:")
        grid = sorted(next(r for r in runs if r["learnt"])["probe"]["grid"], reverse=True)
        P_("    " + f"{'':<10}" + "".join(f"{t:>6.1f}" for t in grid))
        for nd in MODELS:
            rs = [r for r in runs if r["regime"] == rg and r["node"] == nd and r["learnt"]]
            if rs:
                P_(f"    {nd:<10}" + "".join(f"{np.mean([r['probe']['grid'][t] for r in rs]):>+6.2f}" for t in grid))

    for r in runs:
        if r["probe"]:
            r["probe"] = {(str(k) if not isinstance(k, str) else k): v for k, v in r["probe"].items()}
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"mouse_target": {str(k): v for k, v in tgt.items()},
               "results": {f"{rg}|{nd}": v for (rg, nd), v in res.items()}, "runs": runs},
              open(ART, "w"), indent=1, default=str)
    P_(f"\n  artifact: outputs/memcommit.json   runtime {time.time() - t0:.0f}s")
    P_("\n" + RULE); P_("C4  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  One task, one noise level (0.1), one leak (0.5), not swept. A network that ignores every distractor")
    P_("  is a better memory than the mouse and a worse model of it; both readings are reported.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
