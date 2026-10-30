"""perturb_recall — the honest test of the user's idea: can a model that fuses ALL the data (graph + regulons + measured
co-response) get RECALL on which genes move when you knock a gene out, on HELD-OUT knockouts? Trains a gradient-boosted
classifier on measured Perturb-seq (K562, 9,871 knockouts x 8,202 genes) with features per (knockout K, gene T) pair:
  base       — T's response-proneness (fraction of TRAIN knockouts where T moves)   [the "mean-response" signal]
  transfer   — how often T moves when K's GRAPH NEIGHBOURS are knocked out (train)   [co-response, graph-guided]
  is_target  — T is a reg / SIGNOR regulatory target of K
  is_ppi     — T physically interacts with K
Evaluated held-out BY KNOCKOUT (train on 80% of KOs, predict the unseen 20%), against baselines.

MEASURED RESULT (545 held-out knockouts with >=5 movers), mean AUPRC / recall@50:
  random 0.017          graph-neighbour 0.019 (~useless alone)   base-rate 0.152 / 15%
  transfer 0.155 / 15%  MODEL (all data) 0.184 / ~18%
HONEST READING: the model WORKS — ~11x random, and it beats every baseline — but the recall is PARTIAL (top-50 captures
~18% of movers) and the lift OVER the simple base-rate baseline is modest (+0.03 AUPRC). Most predictable recall is
"which genes are response-prone" (base rate) + "what do my network-neighbour knockouts do" (transfer); the graph edges
ALONE are ~random, and knockout-SPECIFIC prediction stays hard. This matches the field (foundation models barely beat
simple baselines on perturbation prediction) and this project's earlier latent-bridge (+0.018 over mean). More data /
more screens raise the base-rate quality but do not, on this evidence, break the wall — the missing signal is measured,
context-specific function, not model capacity. So: a real recall LIFT and a useful triage ranker, not a solved cascade.
"""
import json, collections
import numpy as np


def _load():
    import dependency as D
    dep = D.Dependency()
    M, syms, pidx, thr = dep.M, dep.syms, dep.pidx, dep.thr
    moved = (np.abs(M) > thr)
    symidx = {s: i for i, s in enumerate(syms)}
    Dj = json.load(open("outputs/orphan/cell_complete.json"))
    names = [g["name"] for g in Dj["genes"]]
    adj, reg = collections.defaultdict(set), collections.defaultdict(set)
    for a, b in Dj["ppi"]:
        if a < len(names) and b < len(names):
            adj[names[a]].add(names[b]); adj[names[b]].add(names[a])
    for s, t, _sg in Dj["reg"]:
        if s < len(names) and t < len(names):
            reg[names[s]].add(names[t]); adj[names[s]].add(names[t])
    try:
        for s, t, _sg, _src in json.load(open("outputs/orphan/causal_edges.json"))["edges"]:
            if s < len(names) and t < len(names):
                reg[names[s]].add(names[t]); adj[names[s]].add(names[t])
    except OSError:
        pass
    kos = [g for g in dep.dual if g in pidx]
    return dict(moved=moved, syms=syms, pidx=pidx, symidx=symidx, adj=adj, reg=reg, kos=kos)


def _features(K, C, base, train_set):
    moved, syms, pidx, adj, reg = C["moved"], C["syms"], C["pidx"], C["adj"], C["reg"]
    nb = [pidx[x] for x in adj.get(K, ()) if x in train_set]
    transfer = moved[np.array(nb)].mean(axis=0) if nb else np.zeros(len(syms))
    tgt, ppin = reg.get(K, set()), adj.get(K, set())
    is_tgt = np.fromiter((1.0 if s in tgt else 0.0 for s in syms), float, len(syms))
    is_ppi = np.fromiter((1.0 if s in ppin else 0.0 for s in syms), float, len(syms))
    return np.column_stack([base, transfer, is_tgt, is_ppi])


def run(seed=0, train_cap=1500):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    C = _load()
    moved, syms, pidx, kos = C["moved"], C["syms"], C["pidx"], C["kos"]
    rng.shuffle(kos)
    ntr = int(len(kos) * 0.8)
    train_kos, test_kos = kos[:ntr], kos[ntr:]
    train_set = set(train_kos)
    base = moved[np.array([pidx[g] for g in train_kos])].mean(axis=0)
    X, y = [], []
    for K in train_kos[:train_cap]:
        mv = np.where(moved[pidx[K]])[0]
        if len(mv) < 5:
            continue
        neg = rng.choice(len(syms), size=min(len(mv) * 3, len(syms)), replace=False)
        F = _features(K, C, base, train_set)
        for j in np.concatenate([mv, neg]):
            X.append(F[j]); y.append(int(moved[pidx[K], j]))
    clf = HistGradientBoostingClassifier(max_iter=250, max_depth=4).fit(np.array(X), np.array(y))
    ap = lambda s, t: average_precision_score(t, s) if t.sum() else np.nan
    r50 = lambda s, t: t[np.argsort(-s)[:50]].sum() / t.sum() if t.sum() else np.nan
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for K in test_kos:
        mv = moved[pidx[K]]
        if mv.sum() < 5:
            continue
        F = _features(K, C, base, train_set)
        scores = {"random": np.full(len(syms), mv.mean()), "graph": F[:, 2] + F[:, 3],
                  "base-rate": F[:, 0], "transfer": F[:, 1], "MODEL": clf.predict_proba(F)[:, 1]}
        for k, s in scores.items():
            R[k]["ap"].append(ap(s, mv)); R[k]["r50"].append(r50(s, mv))
    n = len(R["base-rate"]["ap"])
    score = {k: {"auprc": round(float(np.nanmean(R[k]["ap"])), 3),
                 "recall_at50": round(float(np.nanmean(R[k]["r50"])), 3)} for k in R}
    return {"n_heldout_kos": n, "scorecard": score,
            "note": "Held-out-by-knockout recall of measured movers. MODEL fuses base-rate + graph-neighbour transfer + "
                    "reg/SIGNOR targets + PPI. Beats every baseline (AUPRC ~0.18 vs base-rate 0.15 vs graph 0.02 vs "
                    "random 0.02) but recall is PARTIAL (top-50 ~18% of movers) and the lift over base-rate is modest. "
                    "Graph edges alone ~random; recall is base-rate + co-response. Real triage lift, not a solved cascade."}


def main():
    print("=" * 84)
    print("PERTURB-RECALL — can a model on ALL the data predict which genes move? (held-out knockouts)")
    print("=" * 84)
    out = run()
    print(f"\n  held-out knockouts (>=5 movers): {out['n_heldout_kos']}")
    print(f"  {'method':16s}  AUPRC   recall@50")
    for k in ["random", "graph", "base-rate", "transfer", "MODEL"]:
        s = out["scorecard"][k]
        print(f"    {k:14s}  {s['auprc']:.3f}   {s['recall_at50']:.1%}")
    print("\n  -> MODEL beats every baseline but recall is PARTIAL (~18% @50); the graph edges alone are ~random.")
    print("     The wall is softened (11x random), not broken — knockout-specific recall needs measured function.")
    from pathlib import Path
    Path("outputs/orphan").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("outputs/orphan/perturb_recall.json", "w"), indent=1)
    print("\n  -> outputs/orphan/perturb_recall.json")
    return out


if __name__ == "__main__":
    main()
