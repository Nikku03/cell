"""Multi-step reasoning: follow a chain of facts for more steps than were ever trained.

THE COMBINED NETWORK. The record's best parts in one model: a dendritic TREE (per-node weights) of
BUMP nodes (the generic bend that beat the biological node, nonbio.py) as the update of a MEMORY
loop (memcommit.py / memmatch.py). Each step:  h_t = tanh( F( [x_t, h_{t-1}] ) ).
    combined   F = a bump-node tree: U = 24 units, branching 4, depth 2 (16 leaves per unit)
    bumprnn    F = one layer of bump point-neurons      (memory, no tree)
    tanhrnn    F = one linear layer                     (memory, conventional baseline)
The RNN widths are solved so every model has the combined network's parameter count to within 1%.
The tanh shell on the state is common to all three, so they differ only in F.

THE TASK -- POINTER CHASING. Every trial draws a fresh random permutation f of 6 items and a start
item s; the answer after k hops is f^k(s). The table f (36 one-hot bits) is presented at every step;
s only at step 1. The network runs exactly k steps and answers from h_k (6-way). Nothing about f can
be memorised across trials: each hop must be computed from the facts given.
TRAIN on k in {1, 2, 3}. TEST on k = 1..8. Chance = 1/6.
A network that learnt "one step = one hop" keeps working at k = 4..8; one that learnt the trained
depths does not. That gap is the test of multi-step reasoning.

BUDGET. Adam lr 0.003, batch 128, gradient norm clipped at 1, up to MAXSTEPS steps, stopping early
when in-distribution accuracy >= 0.99 (checked every 250 steps). MAXSTEPS was set by ONE pilot of
the BASELINE only (tanhrnn, seed 999, outside the evaluation seeds): it reached 0.994 in
distribution at 2,750 steps (unseen depths 0.263), so MAXSTEPS = 6,000, about twice that.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
Q0 VALID, BLOCKING. BPTT gradient |num - ana| <= 1e-5 max + 1e-9 for all three models; a broken
   copy (the gradient into the previous state dropped) MUST fail.
Q1 HARNESS, BLOCKING. The evaluation code scores a hand-built exact solver 1.000 at every k and a
   constant guesser within 0.03 of 1/6.
Q2 LEARNABILITY. A seed counts if its in-distribution accuracy (k = 1-3) >= 0.90. A model with
   fewer than 9/10 learnt seeds gets no verdict.
Seeds 0-9. Unit = per seed GENERALISATION SCORE = mean accuracy over the unseen depths k = 4..8.
R1 PRIMARY. combined vs tanhrnn: higher on >= 9/10 seeds -> THE COMBINED NETWORK REASONS FURTHER;
   lower on >= 9/10 -> THE BASELINE REASONS FURTHER; else NO DIFFERENCE AT THIS RESOLUTION.
R2 DOES THE TREE ADD ANYTHING? combined vs bumprnn, same rule.
R3 DESCRIPTIVE. Accuracy at every k; the deepest k still >= 0.5 per model.
R4 WHAT THIS IS AND IS NOT. One synthetic multi-step task with 6 items and the facts re-shown every
   step. Following chains is a component of reasoning, not reasoning in general.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import multiprocessing as mp
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_chainreason.txt")
ART = R("outputs", "chainreason.json")
RULE = "=" * 97

M = 6
DX = M * M + M
U, B, LV = 24, 4, 2
LR, BS, CLIP, EVERY = 0.003, 128, 1.0, 250
MAXSTEPS = int(os.environ.get("CHAIN_MAXSTEPS", "6000"))
TRAIN_K = (1, 2, 3)
TEST_K = tuple(range(1, 9))
SEEDS = tuple(range(10))
MODELS = ("combined", "bumprnn", "tanhrnn")
NEVAL = 2000


def batch(rng, n, k):
    perm = np.argsort(rng.random((n, M)), axis=1)
    T = np.zeros((n, M, M)); T[np.arange(n)[:, None], np.arange(M)[None, :], perm] = 1.0
    s = rng.integers(0, M, n)
    y = s.copy()
    for _ in range(k):
        y = perm[np.arange(n), y]
    X = np.zeros((n, k, DX)); X[:, :, :M * M] = T.reshape(n, -1)[:, None, :]
    X[np.arange(n), 0, M * M + s] = 1.0
    return X, y, perm, s


def bump_f(v, a, mu, s):
    Dn = s * s + 0.01
    q = (v - mu) ** 2 / Dn
    e = np.exp(-q)
    return v + a * e, (v, e, q, Dn)


def bump_b(dout, c, a, mu, s):
    v, e, q, Dn = c
    k = a * e * 2 * (v - mu) / Dn
    return dout * (1 - k), dout * e, dout * k, dout * a * e * q * 2 * s / Dn


class Net:
    """Recurrent shell h_t = tanh(F([x_t, h_{t-1}])) with a 6-way readout."""

    broken = False

    def __init__(self, kind, rng, H=None, u=U):
        self.kind = kind
        self.H = u if kind == "combined" else H
        D = DX + self.H
        if kind == "combined":
            self.K = B ** LV
            self.Wl = rng.normal(0, 1 / math.sqrt(D), (u, self.K, D))
            self.W, self.a, self.mu, self.s = [], [], [], []
            Mn = self.K
            for _ in range(LV):
                Mn //= B
                self.W.append(rng.normal(0, 1 / math.sqrt(B), (u, Mn, B)))
                self.a.append(np.full((u, Mn), 0.5)); self.mu.append(np.zeros((u, Mn))); self.s.append(np.ones((u, Mn)))
            self.F = [self.Wl] + self.W + self.a + self.mu + self.s
        else:
            self.Wr = rng.normal(0, 1 / math.sqrt(D), (self.H, D)); self.br = np.zeros(self.H)
            self.F = [self.Wr, self.br]
            if kind == "bumprnn":
                self.q = [np.full(self.H, 0.5), np.zeros(self.H), np.ones(self.H)]
                self.F += self.q
        self.Wo = rng.normal(0, 1 / math.sqrt(self.H), (self.H, M)); self.bo = np.zeros(M)
        self.ps = self.F + [self.Wo, self.bo]

    # ---- F forward / backward --------------------------------------------------------------
    def f_fwd(self, z):
        n = z.shape[0]
        if self.kind == "combined":
            h = np.einsum('nd,ukd->nuk', z, self.Wl)
            cache = [h]
            for l in range(LV):
                chn = h.reshape(n, self.H, -1, B)
                v = (chn * self.W[l][None]).sum(-1)
                h, c = bump_f(v, self.a[l][None], self.mu[l][None], self.s[l][None])
                cache.append((chn, c))
            return h[:, :, 0], cache
        v = z @ self.Wr.T + self.br
        if self.kind == "tanhrnn":
            return v, None
        out, c = bump_f(v, self.q[0], self.q[1], self.q[2])
        return out, c

    def f_bwd(self, dout, z, cache):
        n = z.shape[0]
        if self.kind == "combined":
            dh = dout[:, :, None]
            gW, ga, gm, gs = [None] * LV, [None] * LV, [None] * LV, [None] * LV
            for l in range(LV - 1, -1, -1):
                chn, c = cache[l + 1]
                dv, da, dm, ds = bump_b(dh, c, self.a[l][None], self.mu[l][None], self.s[l][None])
                ga[l], gm[l], gs[l] = da.sum(0), dm.sum(0), ds.sum(0)
                gW[l] = (dv[:, :, :, None] * chn).sum(0)
                dh = (dv[:, :, :, None] * self.W[l][None]).reshape(n, self.H, -1)
            gWl = np.einsum('nuk,nd->ukd', dh, z)
            dz = np.einsum('nuk,ukd->nd', dh, self.Wl)
            return dz, [gWl] + gW + ga + gm + gs
        if self.kind == "tanhrnn":
            dv, extra = dout, []
        else:
            dv, da, dm, ds = bump_b(dout, cache, self.q[0], self.q[1], self.q[2])
            extra = [da.sum(0), dm.sum(0), ds.sum(0)]
        return dv @ self.Wr, [dv.T @ z, dv.sum(0)] + extra

    # ---- recurrence -----------------------------------------------------------------------
    def forward(self, X):
        n, k, _ = X.shape
        h = np.zeros((n, self.H))
        self.cache = []
        for t in range(k):
            z = np.concatenate([X[:, t], h], 1)
            pre, c = self.f_fwd(z)
            h = np.tanh(pre)
            self.cache.append((z, c, h))
        self.hk = h
        return h @ self.Wo + self.bo

    def backward(self, dlog):
        gWo = self.hk.T @ dlog; gbo = dlog.sum(0)
        dh = dlog @ self.Wo.T
        gF = [np.zeros_like(p) for p in self.F]
        for t in range(len(self.cache) - 1, -1, -1):
            z, c, h = self.cache[t]
            dz, g = self.f_bwd(dh * (1 - h * h), z, c)
            for i, gi in enumerate(g):
                gF[i] += gi
            dh = dz[:, DX:] * (0.0 if self.broken else 1.0)
        return gF + [gWo, gbo]


class BrokenNet(Net):
    broken = True


def nparams(m):
    return sum(p.size for p in m.ps)


def rnn_width(kind):
    target = nparams(Net("combined", np.random.default_rng(0)))
    best = min(range(8, 400), key=lambda H: abs(nparams(Net(kind, np.random.default_rng(0), H=H)) - target))
    return best


WIDTH = {}


def build(kind, rng):
    return Net(kind, rng) if kind == "combined" else Net(kind, rng, H=WIDTH[kind])


def softmax_grad(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1
    return p / len(y)


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))


def gradcheck(kind, broken=False, seed=5):
    rng = np.random.default_rng(seed)
    cls = BrokenNet if broken else Net
    m = cls(kind, rng, H=7, u=3) if kind != "combined" else cls(kind, rng, u=3)
    if kind == "combined":
        for p in m.a + m.mu + m.s:
            p[...] = rng.normal(0.4, 0.3, p.shape)
    elif kind == "bumprnn":
        for p in m.q:
            p[...] = rng.normal(0.4, 0.3, p.shape)
    X, y, _, _ = batch(rng, 5, 3)
    lg = m.forward(X); gs = m.backward(softmax_grad(lg, y))
    ratio = 0.0
    for p, g in zip(m.ps, gs):
        for _ in range(4):
            idx = tuple(rng.integers(0, s_) for s_ in p.shape)
            o = p[idx]
            p[idx] = o + 1e-6; lp = xent(m.forward(X), y)
            p[idx] = o - 1e-6; lm = xent(m.forward(X), y)
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
        m_, v_, t = st[i]
        t += 1
        m_[...] = 0.9 * m_ + 0.1 * g
        v_[...] = 0.999 * v_ + 0.001 * g * g
        st[i][2] = t
        p -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)


def accuracy(predict, k, seed, n=NEVAL):
    X, y, perm, s = batch(np.random.default_rng(10_000 + 97 * seed + k), n, k)
    return float((predict(X, perm, s) == y).mean())


def model_predict(m):
    def f(X, perm, s):
        lg = m.forward(X)
        return np.where(np.all(np.isfinite(lg), 1), lg.argmax(1), -1)
    return f


def train(args):
    kind, seed = args
    m = build(kind, np.random.default_rng(seed))
    rng = np.random.default_rng(500 + seed)
    st, steps, ind = {}, 0, 0.0
    t0 = time.time()
    while steps < MAXSTEPS:
        k = int(rng.choice(TRAIN_K))
        X, y, _, _ = batch(rng, BS, k)
        lg = m.forward(X)
        if not np.all(np.isfinite(lg)):
            break
        adam_clip(m.ps, m.backward(softmax_grad(lg, y)), st, LR)
        steps += 1
        if steps % EVERY == 0:
            ind = float(np.mean([accuracy(model_predict(m), k_, 7 + seed, 600) for k_ in TRAIN_K]))
            if ind >= 0.99:
                break
    acc = {k_: accuracy(model_predict(m), k_, seed) for k_ in TEST_K}
    return dict(model=kind, seed=seed, steps=steps, ind=float(np.mean([acc[k_] for k_ in TRAIN_K])),
                acc=acc, gen=float(np.mean([acc[k_] for k_ in TEST_K if k_ > 3])),
                params=nparams(m), seconds=round(time.time() - t0, 1))


def decide(units, a, b):
    d = [units[(a, s)] - units[(b, s)] for s in SEEDS if (a, s) in units and (b, s) in units]
    w = sum(x > 0 for x in d); l = sum(x < 0 for x in d); n = len(d)
    p = sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n if n else 1.0
    return w, l, n, p, float(np.mean(d)) if d else float("nan")


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    for kd in ("bumprnn", "tanhrnn"):
        WIDTH[kd] = rnn_width(kd)
    P_(RULE); P_("MULTI-STEP REASONING: FOLLOW A CHAIN FOR MORE HOPS THAN EVER TRAINED"); P_(RULE)
    P_(f"  budget MAXSTEPS {MAXSTEPS:,} (set by one baseline-only pilot, seed 999)")
    for kd in MODELS:
        P_(f"  {kd:<9} {nparams(build(kd, np.random.default_rng(0))):>7,} parameters"
           + (f"  (width {WIDTH[kd]})" if kd in WIDTH else f"  ({U} tree units, {B ** LV} leaves each)"))
    ok = True
    for kd in MODELS:
        g, bg = gradcheck(kd), gradcheck(kd, broken=True)
        P_(f"  Q0 {kd:<9} gradient / tolerance {g:.3f}   broken copy {bg:.2e} ({'caught' if bg > 1 else 'NOT CAUGHT'})")
        ok &= g <= 1.0 and bg > 1.0
    exact = lambda X, perm, s: np.array([_hop(perm[i], s[i], X.shape[1]) for i in range(len(s))])
    const = lambda X, perm, s: np.zeros(len(s), int)
    ex = [accuracy(exact, k_, 0, 500) for k_ in TEST_K]; co = float(np.mean([accuracy(const, k_, 0) for k_ in TEST_K]))
    q1 = min(ex) == 1.0 and abs(co - 1 / 6) <= 0.03
    P_(f"  Q1 harness: exact solver min accuracy {min(ex):.3f}; constant guesser {co:.3f}  -> {'PASS' if q1 else 'FAIL'}")
    if not (ok and q1):
        P_("  BLOCKED."); open(OUT, "w").write("\n".join(out) + "\n"); return
    pool = mp.get_context("fork").Pool(4)
    runs = pool.map(train, [(kd, s) for kd in MODELS for s in SEEDS])
    pool.close()

    P_("\n  accuracy by number of hops (mean over seeds that learnt; chance 0.167); TRAINED on 1-3")
    P_("    " + f"{'':<22}" + "".join(f"{'k=' + str(k_):>7}" for k_ in TEST_K) + "   learnt   steps(median)")
    units = {}
    for kd in MODELS:
        rs = [r for r in runs if r["model"] == kd]
        lr_ = [r for r in rs if r["ind"] >= 0.90]
        for r in lr_:
            units[(kd, r["seed"])] = r["gen"]
        if lr_:
            P_(f"    {kd:<22}" + "".join(f"{np.mean([r['acc'][k_] for r in lr_]):>7.3f}" for k_ in TEST_K)
               + f"   {len(lr_):>2}/10    {np.median([r['steps'] for r in rs]):,.0f}")
        else:
            P_(f"    {kd:<22} no seed reached 0.90 in distribution (best {max(r['ind'] for r in rs):.3f})")
    P_("\n  per-seed generalisation score (mean accuracy at unseen k = 4..8):")
    P_("    seed " + "".join(f"{kd:>12}" for kd in MODELS))
    for s in SEEDS:
        P_(f"    {s:>4} " + "".join(f"{units[(kd, s)]:>12.3f}" if (kd, s) in units else f"{'(not learnt)':>12}" for kd in MODELS))
    learnt = {kd: sum(1 for s in SEEDS if (kd, s) in units) for kd in MODELS}
    res = {}
    for key, a_, b_, nm in (("R1", "combined", "tanhrnn", "PRIMARY: combined vs conventional memory"),
                            ("R2", "combined", "bumprnn", "does the tree add anything?")):
        if learnt[a_] < 9 or learnt[b_] < 9:
            res[key] = dict(verdict="NO VERDICT (Q2: fewer than 9/10 learnt)", learnt={a_: learnt[a_], b_: learnt[b_]})
            P_(f"\n  {key} {nm}: NO VERDICT -- learnt {a_} {learnt[a_]}/10, {b_} {learnt[b_]}/10"); continue
        w, l, n, p, md = decide(units, a_, b_)
        v = ("THE COMBINED NETWORK REASONS FURTHER" if w >= 9 else
             ("THE BASELINE REASONS FURTHER" if key == "R1" else "THE TREE-LESS BUMP MEMORY REASONS FURTHER") if l >= 9
             else "NO DIFFERENCE AT THIS RESOLUTION")
        res[key] = dict(wins=w, losses=l, n=n, sign_p=p, mean_diff=md, verdict=v)
        P_(f"\n  {key} {nm}: higher on {w}/{n}, lower on {l}/{n}, sign p {p:.4f}, mean {md:+.3f}  -> {v}")
    P_("\n  R3 deepest k with accuracy >= 0.5 (per model, mean over learnt seeds):")
    for kd in MODELS:
        lr_ = [r for r in runs if r["model"] == kd and r["ind"] >= 0.90]
        if lr_:
            deep = [k_ for k_ in TEST_K if np.mean([r["acc"][k_] for r in lr_]) >= 0.5]
            P_(f"    {kd:<9} {max(deep) if deep else 'none'}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump({"maxsteps": MAXSTEPS, "width": WIDTH, "decisions": res,
               "runs": [{**r, "acc": {str(k_): v for k_, v in r["acc"].items()}} for r in runs]},
              open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/chainreason.json   runtime {time.time() - t0:.0f}s")
    P_("\n" + RULE); P_("R4  WHAT THIS IS AND IS NOT"); P_(RULE)
    P_("  One synthetic multi-step task (6 items, facts re-shown every step). Following a chain is one")
    P_("  component of reasoning, not reasoning in general.")
    open(OUT, "w").write("\n".join(out) + "\n")


def _hop(perm, s, k):
    for _ in range(k):
        s = perm[s]
    return s


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        WIDTH["tanhrnn"] = rnn_width("tanhrnn")
        MAXSTEPS = 12000
        t = time.time(); r = train(("tanhrnn", 999))
        print("PILOT tanhrnn seed 999:", {k: r[k] for k in ("steps", "ind", "gen", "params", "seconds")})
    else:
        main()
