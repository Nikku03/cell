"""A LEARNED TIE-BREAKER: does set attention beat counting on the missing-piece ranking problem?

WHERE THIS SITS.  `cell_orphan_ties` measured a real, unclaimed opportunity: the chemistry score puts the true
catalyst inside the top tie group 48.8% of the time and cannot order within it. Hand-built tie-breakers claimed
31% of that headroom, and the winner was the crudest one available -- raw catalysis frequency.

    random   0.294  (floor)      freq  0.355      oracle  0.488  (ceiling)      69% still unclaimed

GEOCONV IS NOT APPLICABLE AND IS NOT INCLUDED.  It consumes a surface point cloud plus a KNN graph per entity.
A tie group is a set of GENE NAMES with no geometry attached, and the catalyst vocabulary is 5,282 genes
against 311 cached structures -- an upper bound of 5.9% coverage even if every structure were a distinct
catalyst. Running it here would be a category error, not an experiment.

THE TRANSFORMER IS APPLICABLE, and for a specific reason rather than because it is bigger.  Ranking within a
candidate set is a SET problem, which is what the ADRN line's set encoders were built for. Every hand-built
tie-breaker scores each candidate INDEPENDENTLY -- freq(gene), degree(gene) -- so it cannot condition on WHO
ELSE is in the group. Self-attention can: "most frequent among these three" is a different question from
"frequent in general", and a group of five kinases should be resolved differently from a mixed group. That is
the mechanism, and if it buys nothing the mechanism was not there.

FEATURES, per candidate, all already measured and none requiring structure:
    log frequency | log PPI degree | n_pathways | pathway-context overlap | compartment overlap
plus, computed WITHIN the group so the model can use relative standing:
    each feature's z-score inside its own tie group

ARMS:
    random                  uniform order. The floor.
    freq                    the hand-built winner to beat, 0.355.
    logistic                fitted linear model over the SAME feature vectors, including the within-group
                            z-scores. So it does get group context -- but only as summary statistics (where
                            this candidate sits relative to the group mean), never candidate-to-candidate.
                            This separates "a fitted combination of group statistics helps" from "pairwise
                            attention helps", which is the sharper question.
    set_transformer         MultiheadAttention over the tie group; each candidate attends to its competitors.
    set_transformer_untrained   random init, identical architecture. THE control -- it has decided several
                            results today, twice by revealing a below-chance floor that exposed a bug.
    oracle                  perfect tie-break. The ceiling, 0.488.

Split by REACTION, held out, so no reaction contributes to both training and evaluation. That makes the eval
set a 35% subset of the 1,500 `cell_orphan_ties` used, so the numbers here are NOT directly comparable to
0.294/0.355/0.488 -- random, freq and oracle are all re-measured on this exact subset, and every comparison
below is internal to this run.

PREDECLARED, before any number:
    set_transformer beats freq AND beats its untrained control
                        -> set context carries information no per-candidate score can, and the remaining 69%
                           of headroom is partly reachable.
    set_transformer ~ logistic
                        -> a fitted combination is what helps; attention over the group adds nothing.
    set_transformer ~ untrained
                        -> capacity does not help, and with a median tie group of 3 that would be entirely
                           unsurprising: there is very little set to attend over.
"""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import cell_orphan_context as C

OUT = A.OUT
N_EVAL = 1500
KS = (1, 5, 20)
EPOCHS = 40
D_MODEL = 32
N_HEADS = 4
SEED = 0
FEATS = ("log_freq", "log_deg", "n_pw", "pw_ctx", "comp")


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    import propagate as pr
    torch.set_num_threads(4)

    report("=" * 100)
    report("A LEARNED TIE-BREAKER -- does set attention beat counting?")
    report("=" * 100)
    report("  Headroom: truth is in the top tie group 48.8% of the time; hand-built tie-breakers claimed 31%")
    report("  of it and the winner was raw frequency. 69% unclaimed.")
    report("  GeoConv is NOT included: it needs a surface point cloud per entity, and a tie group is gene")
    report("  names -- 311 structures against 5,282 catalysts, 5.9% coverage at best. Category error.")
    report("  The transformer IS applicable: every hand-built tie-breaker scores candidates INDEPENDENTLY and")
    report("  cannot condition on who else is in the group. Attention can. That is the mechanism under test.")
    report("  PREDECLARED: beats freq AND its untrained control => set context is real. ~ logistic => a fitted")
    report("  combination is what helps. ~ untrained => capacity does not, which a median group of 3 predicts.")

    net = json.load(open(OUT / "reaction_network.json"))
    steps = net["steps"]
    currency = set(str(x) for x in (net.get("currency") or []))

    def side(s, w):
        return {str(p.get("id") or p.get("name") or "") for p in (s.get(w) or [])} - currency - {""}

    P = [side(s, "in") | side(s, "out") for s in steps]
    CMP = [{C.norm_comp(p.get("compartment"))
            for w in ("in", "out") for p in (s.get(w) or [])} - {None} for s in steps]
    cats = {i: list(s["catalysts"]) for i, s in enumerate(steps) if s.get("catalysts")}
    cat_of_part = defaultdict(set)
    for i in cats:
        for p in P[i]:
            cat_of_part[p].add(i)

    G = pr._load()
    idx, ppi, g2pw = G["idx"], G["ppi"], G.get("g2pw") or {}
    freq = Counter(c for cs in cats.values() for c in cs)
    deg = {g: len(ppi.get(idx.get(g), []) or []) for g in freq}
    pw = {g: set(g2pw.get(idx.get(g), []) or []) for g in freq}
    g2c = defaultdict(Counter)
    for i, cs in cats.items():
        for g in cs:
            g2c[g].update(CMP[i])

    rng = np.random.default_rng(A.SEED)
    pool = [i for i in cats if P[i]]
    pick = rng.choice(len(pool), min(N_EVAL, len(pool)), replace=False)
    ev_all = [pool[int(x)] for x in pick]

    # ---------------- build one example per reaction: the tie group with per-candidate features ----------------
    examples = []
    for i in ev_all:
        pi = P[i]
        sc = defaultdict(float)
        seen, neigh = set(), []
        for p in sorted(pi):
            for j in sorted(cat_of_part.get(p, ())):
                if j in seen or j == i:
                    continue
                seen.add(j)
                neigh.append(j)
                u = len(pi | P[j])
                v = len(pi & P[j]) / u if u else 0.0
                for c in cats[j]:
                    if v > sc[c]:
                        sc[c] = v
        if not sc:
            continue
        truth = set(cats[i])
        top = max(sc.values())
        tied = sorted([c for c, v in sc.items() if v == top])
        rest = sorted([c for c, v in sc.items() if v < top], key=lambda c: -sc[c])
        ctx = Counter()
        for j in neigh:
            for g in cats[j]:
                ctx.update(pw.get(g, ()))
        want_c = CMP[i]

        def comp_of(g):
            c = g2c.get(g)
            if not c:
                return set()
            if g in cats.get(i, ()):
                c = Counter(c); c.subtract(CMP[i])
                return {k for k, v in c.items() if v > 0}
            return set(c)

        X = np.array([[np.log1p(freq.get(g, 0)), np.log1p(deg.get(g, 0)), len(pw.get(g, ())),
                       sum(ctx.get(x, 0) for x in pw.get(g, ())), len(comp_of(g) & want_c)]
                      for g in tied], float)
        y = np.array([1 if g in truth else 0 for g in tied])
        examples.append({"rx": i, "tied": tied, "rest": rest, "X": X, "y": y,
                         "has_truth": bool(y.sum())})
    n = len(examples)
    tie = np.array([len(e["tied"]) for e in examples])
    report(f"\n  {n:,} reactions with candidates | tie median {np.median(tie):.0f} mean {tie.mean():.1f} "
           f"p90 {np.percentile(tie,90):.0f} | truth in tie group {sum(e['has_truth'] for e in examples)/n:.3f}")

    rs = np.random.default_rng(SEED)
    perm = rs.permutation(n)
    cut = int(n * 0.65)
    tr_i, ev_i = perm[:cut], perm[cut:]
    report(f"  split by REACTION: train {len(tr_i):,} | eval {len(ev_i):,} (disjoint)")

    # within-group z-scores, so the model can express relative standing
    for e in examples:
        Z = e["X"] - e["X"].mean(0)
        s = e["X"].std(0)
        e["Xz"] = np.hstack([e["X"], np.divide(Z, s, out=np.zeros_like(Z), where=s > 1e-9)])
    FD = examples[0]["Xz"].shape[1]

    class SetRanker(nn.Module):
        """each candidate attends to its competitors, so the score can depend on the group's composition --
        the one thing a per-candidate feature cannot express."""
        def __init__(s, fd, d=D_MODEL, h=N_HEADS):
            super().__init__()
            s.proj = nn.Linear(fd, d)
            s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, h, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            s.out = nn.Linear(d, 1)

        def forward(s, X):
            h = s.proj(X).unsqueeze(0)
            a, _ = s.attn(s.ln1(h), s.ln1(h), s.ln1(h), need_weights=False)
            h = h + a
            h = h + s.ff(s.ln2(h))
            return s.out(h.squeeze(0)).squeeze(-1)

    def train_set(trained, seed=SEED):
        torch.manual_seed(seed)
        net_ = SetRanker(FD)
        npar = sum(p.numel() for p in net_.parameters())
        curve = []
        if trained:
            opt = torch.optim.Adam(net_.parameters(), lr=3e-3)
            use = [examples[k] for k in tr_i if examples[k]["has_truth"] and len(examples[k]["tied"]) > 1]
            shuf = np.random.default_rng(seed)          # NOT the global RNG -- runs must reproduce
            for _ in range(EPOCHS):
                shuf.shuffle(use)
                net_.train()
                tot = 0.0
                for e in use:
                    opt.zero_grad()
                    s_ = net_(torch.tensor(e["Xz"], dtype=torch.float32))
                    # listwise softmax cross-entropy: rank the true catalyst first within its own group
                    loss = -torch.log_softmax(s_, 0)[torch.tensor(e["y"]).bool()].mean()
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                curve.append(tot / max(len(use), 1))
        net_.eval()
        return net_, npar, curve

    # ---------------- scoring ----------------
    Xtr = np.vstack([examples[k]["Xz"] for k in tr_i if examples[k]["has_truth"]])
    ytr = np.concatenate([examples[k]["y"] for k in tr_i if examples[k]["has_truth"]])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    lr = LogisticRegression(max_iter=3000).fit((Xtr - mu) / sd, ytr)

    net_t, npar, curve = train_set(True)
    net_u, _, _ = train_set(False, seed=1234)
    # LIVENESS: if the loss never moved, every number below is about an untrained net wearing a trained label
    d_loss = curve[0] - curve[-1]
    n_fit = sum(1 for k in tr_i if examples[k]["has_truth"] and len(examples[k]["tied"]) > 1)
    report(f"\n  training: {n_fit:,} usable groups (truth present, >1 candidate) | listwise loss "
           f"{curve[0]:.4f} -> {curve[-1]:.4f}  (delta {d_loss:+.4f})")
    if d_loss < 0.01:
        report("  *** LOSS DID NOT MOVE -- the 'trained' arm is untrained in all but name. Numbers below are")
        report("      about optimisation failure, not about whether set context exists.")

    def order_of(e, arm):
        t = e["tied"]
        if arm == "random":
            k = {g: rs.random() for g in t}
        elif arm == "freq":
            k = {g: freq.get(g, 0) for g in t}
        elif arm == "logistic":
            p = lr.predict_proba((e["Xz"] - mu) / sd)[:, 1]
            k = dict(zip(t, p))
        elif arm in ("set_transformer", "set_transformer_untrained"):
            m = net_t if arm == "set_transformer" else net_u
            with torch.no_grad():
                v = m(torch.tensor(e["Xz"], dtype=torch.float32)).numpy()
            k = dict(zip(t, v))
        else:                                     # oracle
            k = {g: float(y) for g, y in zip(t, e["y"])}
        return sorted(t, key=lambda g: (-k[g], hashlib.md5(g.encode()).hexdigest())) + e["rest"]

    ARMS = ("random", "freq", "logistic", "set_transformer", "set_transformer_untrained", "oracle")
    hits = {a: {k: 0 for k in KS} for a in ARMS}
    ne = 0
    for j in ev_i:
        e = examples[j]
        truth = set(cats[e["rx"]])
        ne += 1
        for a in ARMS:
            o = order_of(e, a)
            for k in KS:
                hits[a][k] += bool(truth & set(o[:k]))

    report(f"\n  RECALL@k on {ne:,} held-out reactions, differing ONLY in tie-group order")
    report(f"    {'arm':<28}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    rec = {}
    for a in ARMS:
        rec[a] = {str(k): hits[a][k] / ne for k in KS}
        mark = ("   <- floor" if a == "random" else "   <- ceiling" if a == "oracle" else "")
        report(f"    {a:<28}" + "".join(f"{rec[a][str(k)]:>9.3f}" for k in KS) + mark)
    report(f"    set transformer parameters: {npar:,}")

    report("\n  READING")
    base, ceil = rec["random"]["1"], rec["oracle"]["1"]
    st, su = rec["set_transformer"]["1"], rec["set_transformer_untrained"]["1"]
    fq, lg = rec["freq"]["1"], rec["logistic"]["1"]
    def share(v):
        return (v - base) / max(ceil - base, 1e-9)
    if st > fq + 0.02 and st > su + 0.02:
        report(f"  SET CONTEXT IS REAL. The transformer reaches {st:.3f} against freq {fq:.3f} and its own")
        report(f"  untrained control {su:.3f} -- {share(st):.0%} of the headroom versus {share(fq):.0%}.")
        report("  Conditioning on WHO ELSE is in the tie group carries information no per-candidate score can.")
    elif st <= su + 0.02:
        report(f"  CAPACITY DOES NOT HELP. transformer {st:.3f} vs its untrained control {su:.3f}. With a")
        report(f"  median tie group of {np.median(tie):.0f} there is almost no set to attend over, which is the")
        report("  most likely reason -- attention needs competitors to compare against.")
    elif abs(st - lg) <= 0.02:
        report(f"  A FITTED COMBINATION IS WHAT HELPS, not attention. transformer {st:.3f} vs logistic")
        report(f"  {lg:.3f} vs freq {fq:.3f}. The set context adds nothing over weighting the same features.")
    else:
        report(f"  MIXED: transformer {st:.3f}, logistic {lg:.3f}, freq {fq:.3f}, untrained {su:.3f}, ceiling")
        report(f"  {ceil:.3f}. Read the per-arm numbers rather than a single verdict.")

    json.dump({"test": "cell_orphan_ties_nn", "n_eval": ne, "n_train": int(len(tr_i)), "ks": list(KS),
               "recall": rec, "tie_median": float(np.median(tie)), "params": npar,
               "loss_first": curve[0], "loss_last": curve[-1], "n_fit_groups": n_fit,
               "geoconv_excluded": "needs a surface point cloud per entity; 311 structures vs 5,282 catalysts",
               "log": log}, open(OUT / "cell_orphan_ties_nn.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_ties_nn.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
