"""Do cliques add anything to the best design? The engine's tree with each sibling group wired as a clique.

FROM cliquenet.py: inside a deep sequence of units, directed-clique wiring turned a network that could not
learn into one that could, but the engine's shallow, wide TREE still beat every clique arm (9/10 seeds).
So the fair question is whether cliques improve the tree itself.

THE CONTRAST. The reasoning engine's update function (24 units x 16 linear leaves; each level sums groups of
4 children, then a bump), in two wirings, otherwise identical (same leaves, levels, inputs, training):
   TREE        siblings do not see each other (clique dimension 1)
   CLIQUETREE  inside each group of 4 siblings, sibling p also reads every earlier sibling q < p before its
               bump (a 3-dimensional clique per group; with the parent reading all four, dimension 4)
The lateral weights add 6 per sibling group (about 1-2% of parameters; reported) and start at 0, so the
clique tree starts as the tree and has to learn its lateral wiring.
Tasks, seeds, data, evaluation, training: exactly cliquenet.py (H1S: 12 items with the snap; P1N: 6 items
without), 10 seeds each.

GATES, PREDECLARED
H   harness as cliquenet.py (blocking).
C1  PRIMARY, per task: CLIQUETREE vs TREE on the long score, paired by seed: HELPS if higher on >= 9/10,
    HURTS if lower on >= 9/10, else NO DIFFERENCE; both need >= 9/10 learnt seeds, else NO VERDICT.
C2  EXACT (>= 0.99 at the deepest test on >= 9/10 seeds), per arm (descriptive).
X   NOT: one placement of cliques (inside sibling groups); the dimension/skip-connection confound of
    cliquenet.py applies here too.
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
OUT = os.path.join(HERE, "RESULTS_cliquetree.txt")
ART = R("outputs", "cliquetree.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("cliquenet", os.path.join(HERE, "cliquenet.py"))
cn = importlib.util.module_from_spec(_s); _s.loader.exec_module(cn)
ce, U, P, TASKS = cn.ce, cn.U, cn.P, cn.TASKS
ARMS = ("TREE", "CLIQUETREE")


def build(arm, D, m, snap):
    if arm == "TREE":
        return cn.build("TREE", D, m, snap)
    import torch
    import torch.nn as nn
    net = cn.build("TREE", D, m, snap)

    class CliqueTreeF(nn.Module):
        def __init__(self, base, B=4):
            super().__init__()
            self.base, self.B = base, B
            self.lat = nn.ParameterList()
            nodes = B ** base.LV
            for _ in range(base.LV):
                nodes //= B
                self.lat.append(nn.Parameter(torch.zeros(U, nodes, B, B)))     # lateral weights, start at 0
            self.register_buffer("mask", torch.tril(torch.ones(B, B), -1))

        def forward(self, z):
            b = self.base
            h = b.leaf(z).view(z.shape[0], U, -1)
            for l in range(b.LV):
                kids = h.view(h.shape[0], U, -1, self.B)                   # (n, U, groups, B)
                lat = self.lat[l] * self.mask                              # sibling p reads siblings q < p
                outs = []
                for p in range(self.B):
                    c = kids[..., p]
                    if p > 0:
                        c = c + (torch.stack(outs, -1) * lat[:, :, p, :p]).sum(-1)
                    outs.append(c)
                kids = torch.stack(outs, -1)
                v = (kids * b.w[l]).sum(-1)
                h = v + b.a[l] * torch.exp(-(v - b.mu[l]) ** 2 / (b.s[l] ** 2 + 0.01))
            return h.squeeze(-1)

    net.F = CliqueTreeF(net.F)
    return net


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
    t0 = time.time()
    for _ in range(ce.STEPS):
        k = int(rng.choice(ce.TRAIN_K))
        X, Y = ce.batch(rng, ce.BS, k, m, "every")
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)
        lg = net(Xt, Yt if snap else None, ce.NOISE)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, m), Yt.reshape(-1))
        if not torch.isfinite(loss):
            break
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), ce.CLIP); opt.step()
    net.eval()

    def pred(X):
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32))[:, -1].argmax(1).numpy()

    acc = {k_: ce.evaluate(pred, k_, seed, m, "every") for k_ in V["test"]}
    return dict(task=task, arm=arm, seed=seed, acc=acc, params=sum(q.numel() for q in net.parameters()),
                ind=float(np.mean([acc[k_] for k_ in ce.TRAIN_K])),
                long=float(np.mean([acc[k_] for k_ in V["test"] if k_ > 3])), seconds=round(time.time() - t0, 1))


def main():
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("DO CLIQUES ADD ANYTHING TO THE BEST DESIGN? THE TREE WITH CLIQUE-WIRED SIBLINGS"); P_(RULE)
    h = cn.harness()
    for task, v in h.items():
        P_(f"  H {task}: exact solver worst depth {v['exact_min']:.3f}, constant guesser {v['constant']:.3f} -> {'PASS' if v['passed'] else 'FAIL'}")
    if not all(v["passed"] for v in h.values()):
        P_("  HARNESS FAILED"); open(OUT, "w").write("\n".join(out) + "\n"); return
    jobs = [(t, a, s) for t in TASKS for a in ARMS for s in TASKS[t]["seeds"]]
    with mp.Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    verdicts = {}
    for task, V in TASKS.items():
        P_(); P_(f"{task} -- {V['m']} items, {'with' if V['snap'] else 'no'} snap; trained on 1-3 hops")
        P_("     arm          params   " + "  ".join(f"k={k:<3d}" for k in V["test"]) + "   learnt  exact")
        for arm in ARMS:
            rs = [r for r in res if r["task"] == task and r["arm"] == arm]
            mean = {k: np.mean([r["acc"][k] for r in rs]) for k in V["test"]}
            learnt = sum(r["ind"] >= 0.90 for r in rs); exact = sum(r["acc"][V["test"][-1]] >= 0.99 for r in rs)
            P_(f"     {arm:11s} {rs[0]['params']:7,}   " + "  ".join(f"{mean[k]:.3f}" for k in V["test"])
               + f"   {learnt:2d}/10  {exact:2d}/10" + ("  EXACT" if exact >= 9 else ""))
        txt, v = cn.compare(res, task, "CLIQUETREE", "TREE")
        verdicts[task] = v
        P_(f"  C1 clique-wired tree vs tree (PRIMARY): {txt}")
    P_(f"\n  runtime {time.time() - t0:.0f}s")
    open(OUT, "w").write("\n".join(out) + "\n")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(dict(harness=h, runs=res, verdicts=verdicts), open(ART, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
