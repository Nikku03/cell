"""Do brain-like higher-dimensional cliques help a network reason?

THE QUESTION. Blue Brain's reconstruction of neocortex found an abundance of DIRECTED CLIQUES: groups of
neurons all wired to each other in one direction, bound into cavities that shape correlated activity
(Reimann et al., Front Comput Neurosci 2017, doi 10.3389/fncom.2017.00048; press coverage: 'up to 11
dimensions'). A directed clique of n neurons has dimension n - 1. The reasoning engine's update function is
a dendritic TREE (branching 4, depth 2): it has no all-connected groups at all, i.e. dimension 1. Does
higher-dimensional wiring help multi-step reasoning when everything else is held equal?

THE CONTRAST, EVERYTHING ELSE EQUAL. The update function h_t = tanh(F([x_t, (e_{t-1},) h_{t-1}])) is built
from 24 blocks of L bump units (bump: v + a exp(-(v - mu)^2 / (s^2 + 0.01)), the node that beat the
ion-channel node). Every unit reads the input; the block's last unit (its sink) is the block's output, so
the state is 24 numbers, as in the tree. Within a block:
   CHAIN-L   unit p reads unit p-1 only                -> clique dimension 1
   CLIQUE-L  unit p reads every earlier unit q < p     -> one directed clique of dimension L - 1
Same units, same depth (L sequential units), same inputs; the clique has L(L-1)/2 - (L-1) more within-block
weights per block, a few percent of the parameters (reported).
   L = 5   dimension 4      L = 12   dimension 11 (the brain's number)
   TREE    the reasoning engine's update function (chainexact 'combined': 24 units x 16 leaves) -- reference,
           not parameter-matched.

TASKS (data, splits, seeds and evaluation code from chainexact.py; trained on 1-3 hops, fixed 3,000 steps,
Adam lr 0.003, batch 128, clip 1.0, state noise sd 0.2 in training, a loss at every step):
   H1S  12 items, facts every step, WITH the snap (chainsnap.py): test 1, 2, 3, 4, 6, 8, 12, 16; seeds 20-29.
        Here only the tree learnt before (10/10 vs tanh 1/10); none was exact (tree 0.892 at 16).
   P1N  6 items, facts every step, NO snap (chainexact.py C2): test 1-4, 6, 8, 12, 16, 24, 32; seeds 10-19.
        Here the tree reached 0.904 at 16, not exact.
Implemented in PyTorch (autograd; the BPTT of earlier modules was hand-written and gradient-checked).

=================================================================================================
GATES, PREDECLARED
=================================================================================================
H  HARNESS (blocking): through the same evaluation code, an exact solver scores 1.000 at every depth and a
   constant guesser within 0.03 of 1/items.
LEARNT: in-distribution (1-3 hops) accuracy >= 0.90. A comparison needs >= 9/10 learnt seeds in both arms,
else NO VERDICT. LONG SCORE: mean accuracy over test depths > 3. Paired by seed:
   HELPS if the first arm is higher on >= 9/10 seeds, HURTS if lower on >= 9/10, else NO DIFFERENCE.
K1 PRIMARY, per task: CLIQUE-12 vs CHAIN-12 -- do 11-dimensional cliques help at equal size and depth?
K2 per task: CLIQUE-5 vs CHAIN-5 -- 4-dimensional.
K3 per task: CLIQUE-12 vs CLIQUE-5, read beside CHAIN-12 vs CHAIN-5 (the depth-only change).
K4 descriptive: the best clique arm vs TREE; EXACT (>= 0.99 at the deepest test on >= 9/10 seeds) per arm.
X  NOT: one family of synthetic reasoning tasks; cliques placed by design inside blocks, not grown; the
   brain's cliques overlap and are bound into cavities, which this does not model.
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
OUT = os.path.join(HERE, "RESULTS_cliquenet.txt")
ART = R("outputs", "cliquenet.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("chainexact", os.path.join(HERE, "chainexact.py"))
ce = importlib.util.module_from_spec(_s); _s.loader.exec_module(ce)

U, P = 24, 16
TASKS = {"H1S": dict(m=12, snap=True, test=(1, 2, 3, 4, 6, 8, 12, 16), seeds=tuple(range(20, 30))),
         "P1N": dict(m=6, snap=False, test=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32), seeds=tuple(range(10, 20)))}
ARMS = {"TREE": None, "CHAIN-5": (5, False), "CLIQUE-5": (5, True), "CHAIN-12": (12, False), "CLIQUE-12": (12, True)}


def build(arm, D, m, snap):
    import torch
    import torch.nn as nn

    def bump(v, a, mu, s):
        return v + a * torch.exp(-(v - mu) ** 2 / (s ** 2 + 0.01))

    class TreeF(nn.Module):
        def __init__(self, B=4, LV=2):
            super().__init__()
            self.B, self.LV, K = B, LV, B ** LV
            self.leaf = nn.Linear(D, U * K, bias=False)
            nn.init.normal_(self.leaf.weight, 0, 1 / math.sqrt(D))
            self.w, self.a, self.mu, self.s = (nn.ParameterList() for _ in range(4))
            nodes = K
            for _ in range(LV):
                nodes //= B
                self.w.append(nn.Parameter(torch.randn(U, nodes, B) / math.sqrt(B)))
                self.a.append(nn.Parameter(torch.full((U, nodes), 0.5)))
                self.mu.append(nn.Parameter(torch.zeros(U, nodes)))
                self.s.append(nn.Parameter(torch.ones(U, nodes)))

        def forward(self, z):
            h = self.leaf(z).view(z.shape[0], U, -1)
            for l in range(self.LV):
                v = (h.view(h.shape[0], U, -1, self.B) * self.w[l]).sum(-1)
                h = bump(v, self.a[l], self.mu[l], self.s[l])
            return h.squeeze(-1)

    class BlockF(nn.Module):
        def __init__(self, L, clique):
            super().__init__()
            self.L = L
            self.win = nn.Linear(D, U * L)
            nn.init.normal_(self.win.weight, 0, 1 / math.sqrt(D)); nn.init.zeros_(self.win.bias)
            mask = torch.tril(torch.ones(L, L), -1) if clique else torch.diag(torch.ones(L - 1), -1)
            self.register_buffer("mask", mask)
            fan = mask.sum(1).clamp(min=1)
            self.wl = nn.Parameter(torch.randn(U, L, L) / torch.sqrt(fan)[None, :, None])
            self.a = nn.Parameter(torch.full((U, L), 0.5)); self.mu = nn.Parameter(torch.zeros(U, L))
            self.s = nn.Parameter(torch.ones(U, L))

        def n_within(self):
            return int(self.mask.sum().item()) * U

        def forward(self, z):
            pre = self.win(z).view(z.shape[0], U, self.L)
            acts = []
            for p in range(self.L):
                v = pre[:, :, p]
                if p > 0:
                    A = torch.stack(acts, -1)
                    v = v + (A * (self.wl[:, p, :p] * self.mask[p, :p])).sum(-1)
                acts.append(bump(v, self.a[:, p], self.mu[:, p], self.s[:, p]))
            return acts[-1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.F = TreeF() if arm == "TREE" else BlockF(*ARMS[arm])
            self.out = nn.Linear(U, m)
            self.emb = nn.Embedding(m, P) if snap else None

        def forward(self, X, Y=None, noise=0.0):
            n, T, _ = X.shape
            h = X.new_zeros(n, U); e = X.new_zeros(n, P)
            logits = []
            for t in range(T):
                z = torch.cat([X[:, t], e, h], 1) if snap else torch.cat([X[:, t], h], 1)
                h = torch.tanh(self.F(z))
                if noise > 0:
                    h = h + noise * torch.randn_like(h)
                lg = self.out(h); logits.append(lg)
                if snap:
                    e = self.emb(Y[:, t] if Y is not None else lg.argmax(1))
            return torch.stack(logits, 1)

    return Net()


def job(args):
    task, arm, seed = args
    import torch
    torch.set_num_threads(1)
    V = TASKS[task]; m, snap = V["m"], V["snap"]
    dx = ce.dx_of(m, "every"); D = dx + U + (P if snap else 0)
    torch.manual_seed(seed)
    net = build(arm, D, m, snap)
    opt = torch.optim.Adam(net.parameters(), lr=ce.LR)
    rng = np.random.default_rng(700 + seed)
    t0, ok = time.time(), True
    for _ in range(ce.STEPS):
        k = int(rng.choice(ce.TRAIN_K))
        X, Y = ce.batch(rng, ce.BS, k, m, "every")
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)
        lg = net(Xt, Yt if snap else None, ce.NOISE)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, m), Yt.reshape(-1))
        if not torch.isfinite(loss):
            ok = False; break
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), ce.CLIP); opt.step()
    net.eval()

    def pred(X):
        with torch.no_grad():
            lg = net(torch.tensor(X, dtype=torch.float32))[:, -1]
        return lg.argmax(1).numpy()

    acc = {k_: ce.evaluate(pred, k_, seed, m, "every") for k_ in V["test"]}
    nwithin = net.F.n_within() if hasattr(net.F, "n_within") else 0
    return dict(task=task, arm=arm, seed=seed, acc=acc, ok=ok, params=sum(q.numel() for q in net.parameters()),
                within=nwithin, ind=float(np.mean([acc[k_] for k_ in ce.TRAIN_K])),
                long=float(np.mean([acc[k_] for k_ in V["test"] if k_ > 3])), seconds=round(time.time() - t0, 1))


def harness():
    out = {}
    for task, V in TASKS.items():
        m = V["m"]

        def exact(X, m=m):
            n, T, _ = X.shape
            perm = X[:, 0, :m * m].reshape(n, m, m).argmax(2)
            cur = X[:, 0, m * m:].argmax(1)
            for _ in range(T):
                cur = perm[np.arange(n), cur]
            return cur

        ex = min(ce.evaluate(exact, k, 0, m, "every") for k in V["test"])
        const = float(np.mean([ce.evaluate(lambda X: np.zeros(len(X), int), k, 0, m, "every") for k in V["test"]]))
        out[task] = dict(exact_min=ex, constant=const, passed=bool(ex == 1.0 and abs(const - 1 / m) <= 0.03))
    return out


def compare(res, task, a, b):
    ra = {r["seed"]: r for r in res if r["task"] == task and r["arm"] == a}
    rb = {r["seed"]: r for r in res if r["task"] == task and r["arm"] == b}
    la = sum(r["ind"] >= 0.90 for r in ra.values()); lb = sum(r["ind"] >= 0.90 for r in rb.values())
    if la < 9 or lb < 9:
        return f"NO VERDICT (learnt {la}/10 vs {lb}/10)", None
    d = [ra[s]["long"] - rb[s]["long"] for s in ra if s in rb]
    hi, lo = sum(x > 0 for x in d), sum(x < 0 for x in d)
    v = "HELPS" if hi >= 9 else "HURTS" if lo >= 9 else "NO DIFFERENCE"
    return f"{a} higher on {hi}/10, lower on {lo}/10, mean {np.mean(d):+.3f} -> {v}", v


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("DO BRAIN-LIKE HIGHER-DIMENSIONAL CLIQUES HELP A NETWORK REASON?"); P_(RULE)
    h = harness()
    for task, v in h.items():
        P_(f"  H {task}: exact solver worst depth {v['exact_min']:.3f}, constant guesser {v['constant']:.3f} -> {'PASS' if v['passed'] else 'FAIL'}")
    if not all(v["passed"] for v in h.values()):
        P_("  HARNESS FAILED: nothing trained."); open(OUT, "w").write("\n".join(out) + "\n"); return
    jobs = [(t, a, s) for t in TASKS for a in ARMS for s in TASKS[t]["seeds"]]
    with mp.Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    verdicts = {}
    for task, V in TASKS.items():
        P_(); P_(RULE); P_(f"{task} -- {V['m']} items, {'with' if V['snap'] else 'no'} snap; trained on 1-3 hops; mean accuracy by hops"); P_(RULE)
        P_("     arm         dim  params (within-block)   " + "  ".join(f"k={k:<3d}" for k in V["test"]) + "   learnt  exact")
        for arm in ARMS:
            rs = [r for r in res if r["task"] == task and r["arm"] == arm]
            dim = 1 if arm in ("TREE",) or arm.startswith("CHAIN") else ARMS[arm][0] - 1
            mean = {k: np.mean([r["acc"][k] for r in rs]) for k in V["test"]}
            learnt = sum(r["ind"] >= 0.90 for r in rs)
            exact = sum(r["acc"][V["test"][-1]] >= 0.99 for r in rs)
            P_(f"     {arm:10s} {dim:4d}  {rs[0]['params']:7,} ({rs[0]['within']:5,})    " + "  ".join(f"{mean[k]:.3f}" for k in V["test"])
               + f"   {learnt:2d}/10  {exact:2d}/10" + ("  EXACT" if exact >= 9 else ""))
        for tag, a, b in (("K1 11-dim cliques vs chains (PRIMARY)", "CLIQUE-12", "CHAIN-12"), ("K2 4-dim cliques vs chains", "CLIQUE-5", "CHAIN-5"),
                          ("K3 11-dim vs 4-dim cliques", "CLIQUE-12", "CLIQUE-5"), ("   (depth only: chain 12 vs chain 5)", "CHAIN-12", "CHAIN-5"),
                          ("K4 11-dim cliques vs tree (not matched)", "CLIQUE-12", "TREE"), ("   4-dim cliques vs tree (not matched)", "CLIQUE-5", "TREE")):
            txt, v = compare(res, task, a, b)
            verdicts[f"{task} {tag.strip()}"] = v
            P_(f"  {tag:42s} {txt}")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(dict(harness=h, runs=res, verdicts=verdicts), open(ART, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
