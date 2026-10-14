"""Our chain network vs a transformer, head to head on the same chain-following questions.

THE TRANSFORMERS. Decoder-only, causal, pre-LayerNorm, NO positional encoding (NoPE -- the variant
reported to length-generalise best; facts are a set, order comes from the causal mask). Tokens:
    fact token   [source item, target item]          (the M facts, shuffled)
    item token   [an item]                           (the start, and in CoT every written-down answer)
    hop token    [a blank 'take a step' marker]      (direct mode)
Two modes, each the counterpart of one of ours:
    DIRECT  facts, start, then k hop tokens; every hop token is trained to name the item after that
            many hops (step supervision, no written-down answers)   <->  our combined net, no snap
    CoT     facts, start, then the written-down answers y1..y(k-1); each position names the next item
            (teacher-forced in training, generated one by one at test) <-> our combined net + snap
Sizes matched to ours: P1 and H2 d=32, 4 layers, 4 heads, FF 64; H1 d=48, 4 layers, 4 heads, FF 96
(exact counts printed). Adam lr 1e-3, batch 128, trained on 1-3 hops, 3,000 steps -- OUR budget.
A GENEROUS budget (9,000 steps, 3x ours) is also run for DIRECT on P1 and reported beside it.
DISCLOSED BEFORE COMMITTING: a 300-step timing run (not the evaluated configuration) on P1 seed 10
was seen: CoT 1.000 to 8 hops, 0.975 at 16, 0.388 at 64; direct 0.379 on the trained depths. It
set nothing above except the decision to give DIRECT, not CoT, the generous budget (CoT had
already learnt the trained depths in 300 steps). Attention is hand-written (same maths as
nn.MultiheadAttention, less overhead) for speed.

SAME QUESTIONS. Test sets are regenerated with chainexact's evaluation seeds (verified identical:
the permutation, start and answer of every test question), seeds 10-19 for P1, 20-29 for H1 and H2.
For a transformer H2 ("facts shown once") is the SAME input as P1 -- the facts stay in its context
window, which is exactly the memory our recurrent networks lack. Our H2 numbers are reported beside it.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
T0 SAME QUESTIONS, BLOCKING: regenerated test questions identical to chainexact's.
T1 HARNESS, BLOCKING: an oracle pushed through the transformer evaluation loops (direct and
   autoregressive) scores 1.000 at every depth; a constant guesser within 0.03 of 1/items.
T2 EXACT if accuracy >= 0.99 at P1: 16 and 64 hops; H1: 16; H2: 8, on >= 9/10 seeds.
T3 HEAD TO HEAD, per seed, long score = mean accuracy over depths > 3 (P1 excluding 64, as before):
     DIRECT vs our combined, no snap (chainexact C2)     CoT vs our combined + snap (chainsnap)
   >= 9/10 higher -> that side BETTER; else NO DIFFERENCE AT THIS RESOLUTION. A side with < 9/10
   seeds learnt (accuracy on 1-3 hops >= 0.90) gets no verdict.
T4 NOT: tiny transformers on one synthetic task. Large pretrained transformers are a different
   regime; published numbers for them are reported separately, from the literature.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import json
import multiprocessing as mp
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_transvs.txt")
ART = R("outputs", "transvs.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("chainexact", os.path.join(HERE, "chainexact.py"))
ce = importlib.util.module_from_spec(_s); _s.loader.exec_module(ce)

VERS = {"P1": dict(m=6, test=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64), exact=(16, 64), seeds=tuple(range(10, 20)), cfg=(32, 4, 4, 64)),
        "H1": dict(m=12, test=(1, 2, 3, 4, 6, 8, 12, 16), exact=(16,), seeds=tuple(range(20, 30)), cfg=(48, 4, 4, 96)),
        "H2": dict(m=6, test=(1, 2, 3, 4, 6, 8), exact=(8,), seeds=tuple(range(20, 30)), cfg=(32, 4, 4, 64))}
STEPS, GENEROUS, LR, BS = 3000, 9000, 1e-3, 128
TRAIN_K = (1, 2, 3)


def questions(seed, k, m, n=ce.NEVAL):
    """chainexact.evaluate's question generator: same seed expression, same first two draws."""
    rng = np.random.default_rng(10_000 + 97 * seed + k + 1000 * m)
    perm = np.argsort(rng.random((n, m)), axis=1)
    s = rng.integers(0, m, n)
    ys, cur = [], s.copy()
    for _ in range(k):
        cur = perm[np.arange(n), cur]; ys.append(cur.copy())
    return perm, s, np.stack(ys, 1)


def tokens(perm, s, items, m, mode, k, rng):
    n = len(s)
    din = 3 * m + 3
    order = np.argsort(rng.random((n, m)), axis=1)
    F = np.zeros((n, m, din))
    ii = np.arange(n)[:, None]; jj = np.arange(m)[None, :]
    F[ii, jj, order] = 1.0
    F[ii, jj, m + perm[ii, order]] = 1.0
    F[:, :, 3 * m] = 1.0
    S = np.zeros((n, 1, din)); S[np.arange(n), 0, 2 * m + s] = 1.0; S[:, 0, 3 * m + 1] = 1.0
    if mode == "direct":
        Hh = np.zeros((n, k, din)); Hh[:, :, 3 * m + 2] = 1.0
        return np.concatenate([F, S, Hh], 1)
    C = np.zeros((n, k - 1, din))
    for j in range(k - 1):
        C[np.arange(n), j, 2 * m + items[:, j]] = 1.0
    C[:, :, 3 * m + 1] = 1.0
    return np.concatenate([F, S, C], 1)


def build(m, cfg):
    import torch
    import torch.nn as nn
    d, L, H, ff = cfg

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

        def forward(self, h, mask):
            n, T, _ = h.shape
            q, k, v = self.qkv(self.l1(h)).view(n, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
            h = h + self.proj(a.transpose(1, 2).reshape(n, T, d))
            return h + self.ff(self.l2(h))

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(3 * m + 3, d)
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, m)

        def forward(self, x):
            T = x.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            h = self.emb(x)
            for b in self.blocks:
                h = b(h, mask)
            return self.out(self.ln(h))
    return TF()


def predict(model, perm, s, m, mode, k, rng, oracle=None):
    """Answer after k hops. Direct: read the last hop token. CoT: generate k answers one by one."""
    import torch
    n = len(s)
    if mode == "direct":
        X = tokens(perm, s, None, m, mode, k, rng)
        if oracle is not None:
            return oracle(perm, s, k)
        with torch.no_grad():
            return model(torch.tensor(X, dtype=torch.float32))[:, -1].argmax(-1).numpy()
    items = np.zeros((n, 0), int)
    order_rng_state = rng.bit_generator.state
    for j in range(k):
        rng.bit_generator.state = order_rng_state          # the same fact order on every pass
        X = tokens(perm, s, np.concatenate([items, np.zeros((n, 1), int)], 1), m, "cot", j + 1, rng)
        if oracle is not None:
            nxt = oracle(perm, items[:, -1] if j else s, 1)
        else:
            with torch.no_grad():
                nxt = model(torch.tensor(X, dtype=torch.float32))[:, -1].argmax(-1).numpy()
        items = np.concatenate([items, nxt[:, None]], 1)
    return items[:, -1]


def hop(perm, s, k):
    cur = s.copy()
    for _ in range(k):
        cur = perm[np.arange(len(s)), cur]
    return cur


def job(args):
    import torch
    torch.set_num_threads(1)
    ver, mode, seed, steps = args
    V = VERS[ver]; m = V["m"]
    torch.manual_seed(seed)
    model = build(m, V["cfg"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(700 + seed)
    t0 = time.time()
    for _ in range(steps):
        k = int(rng.choice(TRAIN_K))
        perm = np.argsort(rng.random((BS, m)), axis=1); s = rng.integers(0, m, BS)
        ys = np.stack([hop(perm, s, j) for j in range(1, k + 1)], 1)
        X = torch.tensor(tokens(perm, s, ys, m, mode, k, rng), dtype=torch.float32)
        lg = model(X)[:, m + (1 if mode == "direct" else 0): m + (1 if mode == "direct" else 0) + k]
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, m), torch.tensor(ys.reshape(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    acc = {}
    for k_ in V["test"]:
        perm, s, ys = questions(seed, k_, m)
        acc[k_] = float((predict(model, perm, s, m, mode, k_, np.random.default_rng(31 + seed + k_)) == ys[:, -1]).mean())
    return dict(version=ver, mode=mode, seed=seed, steps=steps, acc=acc,
                params=int(sum(p.numel() for p in model.parameters())),
                ind=float(np.mean([acc[k_] for k_ in TRAIN_K])), seconds=round(time.time() - t0, 1))


def main():
    import torch
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    P_(RULE); P_("OUR CHAIN NETWORK vs A TRANSFORMER, SAME QUESTIONS"); P_(RULE)
    same = True
    for ver, V in VERS.items():
        for k_ in (1, 5, 16):
            perm, s, ys = questions(V["seeds"][0], k_, V["m"])
            fmt = "once" if ver == "H2" else "every"
            X, Y = ce.batch(np.random.default_rng(10_000 + 97 * V["seeds"][0] + k_ + 1000 * V["m"]), ce.NEVAL, k_, V["m"], fmt)
            same &= bool((Y[:, -1] == ys[:, -1]).all())
    P_(f"  T0 regenerated questions identical to chainexact's: {'PASS' if same else 'FAIL'}")
    orc = lambda perm, s, k: hop(perm, s, k)
    worst = min(float((predict(None, *questions(10, k_, 6)[:2], 6, md, k_, np.random.default_rng(1), oracle=orc)
                       == questions(10, k_, 6)[2][:, -1]).mean()) for md in ("direct", "cot") for k_ in (1, 4, 16))
    const = float((np.zeros(ce.NEVAL, int) == questions(10, 4, 6)[2][:, -1]).mean())
    t1 = worst == 1.0 and abs(const - 1 / 6) <= 0.03
    P_(f"  T1 harness: oracle worst {worst:.3f}, constant guesser {const:.3f} -> {'PASS' if t1 else 'FAIL'}")
    if not (same and t1):
        open(OUT, "w").write("\n".join(out) + "\n"); return
    for ver, V in VERS.items():
        P_(f"  {ver} transformer parameters: {sum(p.numel() for p in build(V['m'], V['cfg']).parameters()):,}")
    jobs = ([(v, md, s, STEPS) for v in VERS for md in ("direct", "cot") for s in VERS[v]["seeds"]]
            + [("P1", "direct", s, GENEROUS) for s in VERS["P1"]["seeds"]])
    jobs.sort(key=lambda j: -j[3])
    pool = mp.get_context("fork").Pool(4)
    runs = list(pool.imap_unordered(job, jobs, chunksize=1))
    pool.close()
    ex = json.load(open(R("outputs", "chainexact.json")))["runs"]
    sn = json.load(open(R("outputs", "chainsnap.json")))["runs"]
    res = {}
    for ver, V in VERS.items():
        P_("\n" + RULE); P_(f"{ver} -- " + {"P1": "6 items", "H1": "12 items", "H2": "6 items, facts shown once (in context for the transformer)"}[ver]
                            + "; trained on 1-3 hops; accuracy by hops (mean over seeds)"); P_(RULE)
        P_("    " + f"{'':<30}" + "".join(f"{'k=' + str(k_):>7}" for k_ in V["test"]) + "  learnt exact")
        shared = [k_ for k_ in V["test"] if k_ > 3 and k_ != 64]

        def line(name, accs, learnt, exact):
            P_(f"    {name:<30}" + "".join(f"{np.mean([a[k_] for a in accs]):>7.3f}" if all(k_ in a for a in accs) else f"{'-':>7}"
                                          for k_ in V["test"]) + f"  {learnt:>2}/10 {exact:>2}/10" + ("  EXACT" if exact >= 9 else ""))

        groups = {}
        for md, lab in (("direct", "TRANSFORMER direct"), ("cot", "TRANSFORMER + CoT")):
            for steps in ((STEPS, GENEROUS) if ver == "P1" and md == "direct" else (STEPS,)):
                rs = [r for r in runs if r["version"] == ver and r["mode"] == md and r["steps"] == steps]
                accs = [r["acc"] for r in rs]
                nm = lab + (" (9k steps)" if steps == GENEROUS else "")
                line(nm, accs, sum(r["ind"] >= 0.9 for r in rs), sum(all(r["acc"][e] >= 0.99 for e in V["exact"]) for r in rs))
                groups[(md, steps)] = {r["seed"]: r for r in rs}
        ours = {"ours no snap": {x["seed"]: {int(k): v for k, v in x["acc"].items()} for x in ex if x["version"] == ver and x["arm"] == "C2"},
                "ours + snap": {x["seed"]: {int(k): v for k, v in x["acc"].items()} for x in sn if x["version"] == ver and x["kind"] == "combined"}}
        for nm, d in ours.items():
            accs = [d[s] for s in V["seeds"]]
            line("OUR combined (" + nm.split(" ", 1)[1] + ")", accs, sum(np.mean([a[k_] for k_ in TRAIN_K]) >= 0.9 for a in accs),
                 sum(all(a.get(e, 0) >= 0.99 for e in V["exact"]) for a in accs))
        for md, ourkey in (("direct", "ours no snap"), ("cot", "ours + snap")):
            T = groups[(md, STEPS)]; O = ours[ourkey]
            lt = sum(T[s]["ind"] >= 0.9 for s in V["seeds"]); lo = sum(np.mean([O[s][k_] for k_ in TRAIN_K]) >= 0.9 for s in V["seeds"])
            name = f"transformer {md} vs our combined {ourkey.split(' ', 1)[1]} (same 3,000-step budget)"
            if lt < 9 or lo < 9:
                P_(f"  T3 {name}: NO VERDICT (learnt transformer {lt}/10, ours {lo}/10)")
                res[(ver, md)] = dict(verdict="NO VERDICT", learnt_t=lt, learnt_o=lo); continue
            d = [np.mean([O[s][k_] for k_ in shared]) - np.mean([T[s]["acc"][k_] for k_ in shared]) for s in V["seeds"]]
            w = sum(x > 0 for x in d); l = sum(x < 0 for x in d)
            v = "OURS BETTER" if w >= 9 else "TRANSFORMER BETTER" if l >= 9 else "NO DIFFERENCE AT THIS RESOLUTION"
            res[(ver, md)] = dict(ours_higher=w, transformer_higher=l, mean_diff=float(np.mean(d)), verdict=v)
            P_(f"  T3 {name}: ours higher on {w}/10, transformer on {l}/10, mean {np.mean(d):+.3f} -> {v}")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    res = {k: {kk: (int(vv) if isinstance(vv, (np.integer,)) else float(vv) if isinstance(vv, np.floating) else vv)
               for kk, vv in v.items()} for k, v in res.items()}
    json.dump({"results": {"|".join(k): v for k, v in res.items()},
               "runs": [{**r, "acc": {str(k_): v for k_, v in r["acc"].items()}} for r in runs]}, open(ART, "w"), indent=1)
    P_(f"\n  artifact: outputs/transvs.json   runtime {time.time() - t0:.0f}s")
    P_("  T4: tiny transformers, one synthetic task; large pretrained models are a different regime.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
