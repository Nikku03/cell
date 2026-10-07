"""ANCHOR-SET SCREEN DESIGN: does WHICH pairs you measure change what you can predict, at equal cost?

THE CLAIM TO TEST.  Gate 2 at proper power showed the interaction is predictable when ONE of the two genes is
novel (+0.0816 / +0.0906, RESOLVED) and not when BOTH are (-0.0215).  That implies a design consequence rather
than a verdict: a screen that pairs many genes against a shared ANCHOR SET leaves most unmeasured pairs with one
familiar gene, while a screen of disjoint pairs leaves them cold on both sides.

If that reasoning is right, then for the SAME NUMBER OF MEASURED PAIRS -- the same screen cost -- an anchor design
should predict the unmeasured pairs better than a random selection.  If it is wrong, the two designs tie and only
the count matters.

CONSTRUCTION.  All designs are simulated on Norman's 131 measured pairs over 73 genes, so the "unmeasured" pairs
are really measured and the prediction can be scored.
    anchor_hub     the k highest-degree genes in the pair graph become the anchor; training = every pair touching
                   it. This is the realistic choice: you would anchor on well-characterised hubs.
    anchor_random  k random genes as the anchor. Separates "anchoring" from "anchoring on hubs".
    random_pairs   a uniformly drawn set of pairs of the SAME SIZE. The cost-matched control.
Every design is scored on the pairs it did not train on, with the same ridge, the same features and the same
K=16 residual basis fitted on its own training pairs only.

REPORTED ALONGSIDE: the fraction of each design's test pairs that have at least one gene appearing somewhere in
its training pairs. That is the proposed mechanism, so it has to be shown moving.

PREDECLARED: the design effect is real only if anchor_hub beats random_pairs at matched training size, past a
bootstrap CI, at more than one anchor size.
"""
import collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
OUT = A.OUT
KC, BOOT, SEEDS = 16, 10_000, 25
ANCHORS = (4, 6, 8, 10, 12)


def main():
    log = []
    def report(t):
        print(t, flush=True); log.append(t)
    report("=" * 100)
    report("ANCHOR-SET SCREEN DESIGN: does WHICH pairs you measure matter, at equal cost?")
    report("=" * 100)
    report("  PREDECLARED: real only if anchor_hub beats cost-matched random_pairs at >1 anchor size.")

    z = np.load(SP / "pseudobulk.npz", allow_pickle=True)
    groups = [str(g) for g in z["groups"]]; prof = z["profile"]
    idx = {g: i for i, g in enumerate(groups)}
    c = idx["control"]
    singles = {g: idx[g] for g in groups if "_" not in g and g != "control"}
    pairs = {g: (idx[g], *g.split("_", 1)) for g in groups
             if "_" in g and "NegCtrl" not in g
             and g.split("_", 1)[0] in singles and g.split("_", 1)[1] in singles}
    keys = sorted(pairs)
    D = prof - prof[c][None, :]
    R = np.stack([D[pairs[k][0]] - D[singles[pairs[k][1]]] - D[singles[pairs[k][2]]] for k in keys])

    zk = np.load(A.SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    uni = sorted(set(kos) | set(singles))
    base, _ = A.build_channels(uni, set(kos), report)
    extra, _, tb = C.extra_channels(uni, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(uni)}
    def feat(k):
        _, a, b = pairs[k]
        xa = chan[urow[a]] if a in urow else np.zeros(chan.shape[1], np.float32)
        xb = chan[urow[b]] if b in urow else np.zeros(chan.shape[1], np.float32)
        return np.concatenate([xa + xb, np.abs(xa - xb), xa * xb])
    X = np.stack([feat(k) for k in keys]).astype(np.float32)

    deg = collections.Counter()
    for k in keys:
        deg[pairs[k][1]] += 1; deg[pairs[k][2]] += 1
    genes = sorted(deg)
    hubs = [g for g, _ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))]
    report(f"  {len(keys)} pairs over {len(genes)} genes; top hub degrees "
           f"{[deg[g] for g in hubs[:6]]}")

    def score(tr, te):
        if len(tr) < 15 or len(te) < 5: return None, None
        u, s, vt = np.linalg.svd(R[tr] - R[tr].mean(0), full_matrices=False)
        B = vt[:min(KC, len(tr) - 1)].astype(np.float32); mu = R[tr].mean(0)
        Y = (R[tr] - mu) @ B.T
        P = A.ridge_apply(X[te], A.ridge_fit(X[tr], Y, 10.0))
        cs = [float(np.corrcoef(P[r] @ B + mu, R[i])[0, 1]) for r, i in enumerate(te)]
        seen = {g for i in tr for g in pairs[keys[i]][1:]}
        warm = np.mean([any(g in seen for g in pairs[keys[i]][1:]) for i in te])
        return np.array(cs), float(warm)

    rng = np.random.default_rng(A.SEED)
    summary = {}
    report(f"\n  {'anchor':>7} {'design':<16} {'n_train':>8} {'n_test':>7} {'corr':>8} "
           f"{'>=1 gene warm':>14}")
    for k in ANCHORS:
        anc = set(hubs[:k])
        tr = np.array([i for i, key in enumerate(keys) if set(pairs[key][1:]) & anc])
        te = np.array([i for i in range(len(keys)) if i not in set(tr.tolist())])
        hub_c, hub_w = score(tr, te)
        if hub_c is None:
            report(f"  {k:>7} anchor_hub       (too few pairs)"); continue
        n_tr = len(tr)
        rnd_means, rnd_all, rnd_w = [], [], []
        for s in range(SEEDS):
            rg = np.random.default_rng(2000 + s)
            trr = rg.choice(len(keys), size=n_tr, replace=False)
            ter = np.array([i for i in range(len(keys)) if i not in set(trr.tolist())])
            cs, w = score(trr, ter)
            if cs is None: continue
            rnd_means.append(np.nanmean(cs)); rnd_all.append(cs); rnd_w.append(w)
        arnd_means, arnd_w = [], []
        for s in range(SEEDS):
            rg = np.random.default_rng(3000 + s)
            a2 = set(rg.choice(genes, size=k, replace=False))
            tr2 = np.array([i for i, key in enumerate(keys) if set(pairs[key][1:]) & a2])
            te2 = np.array([i for i in range(len(keys)) if i not in set(tr2.tolist())])
            cs, w = score(tr2, te2)
            if cs is not None:
                arnd_means.append(np.nanmean(cs)); arnd_w.append(w)
        report(f"  {k:>7} {'anchor_hub':<16} {n_tr:>8} {len(te):>7} {np.nanmean(hub_c):>8.4f} "
               f"{100*hub_w:>13.1f}%")
        report(f"  {'':>7} {'anchor_random':<16} {'~'+str(n_tr):>8} {'':>7} "
               f"{np.mean(arnd_means):>8.4f} {100*np.mean(arnd_w):>13.1f}%")
        report(f"  {'':>7} {'random_pairs':<16} {n_tr:>8} {len(keys)-n_tr:>7} "
               f"{np.mean(rnd_means):>8.4f} {100*np.mean(rnd_w):>13.1f}%")
        d = np.nanmean(hub_c) - np.array(rnd_means)
        br = np.random.default_rng(0)
        bs = d[br.integers(0, len(d), size=(BOOT, len(d)))].mean(1)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        report(f"  {'':>7} anchor_hub - random_pairs  {d.mean():>+7.4f} [{lo:+.4f}, {hi:+.4f}]  "
               f"{'RESOLVED' if (lo>0 or hi<0) else 'within noise'}")
        summary[str(k)] = {"n_train": int(n_tr), "hub": float(np.nanmean(hub_c)),
                           "hub_warm": hub_w, "anchor_random": float(np.mean(arnd_means)),
                           "random_pairs": float(np.mean(rnd_means)),
                           "random_warm": float(np.mean(rnd_w)),
                           "gap": float(d.mean()), "ci": [lo, hi]}
    json.dump({"test": "norman_anchor_design", "n_pairs": len(keys), "n_genes": len(genes),
               "anchors": list(ANCHORS), "summary": summary, "log": log},
              open(OUT / "norman_anchor_design.json", "w"), indent=2)
    report(f"\n  -> {OUT/'norman_anchor_design.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
