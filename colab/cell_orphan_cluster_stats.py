"""RESTATE EVERY OBSCURE NUMBER CLUSTER-WEIGHTED -- because the bin has 80 rows and far fewer facts.

WHY THIS COMES BEFORE ANYTHING ELSE.  The headline of this line is 0.688 top-1 on the obscure bin, and it is
about to be fed into a whole-cell model as if it were a measured constant. But the obscure bin's 80
reactions are not 80 independent trials: many share a catalyst, and reactions with the SAME answer are one
fact seen several times. A model that knows one gene gets every row in its cluster right or every row wrong
together, so treating them as independent overstates precision -- not the estimate, the CONFIDENCE in it.

That distinction matters differently for the two claims this line makes:

    the GAIN (examples over closed-book) is a paired, within-reaction comparison and is robust to this
    the LEVEL (0.688) is an average over correlated rows and its interval is far wider than it looks

Anything built on top needs the second number carried as an interval, not a point.

WHAT IS COMPUTED
    reaction-weighted   the headline as reported: every reaction counts once. This is the DEPLOYMENT-
                        relevant number, because the 9,186 real gaps also contain near-duplicate families
                        and a curator does face each row.
    cluster-weighted    every distinct catalyst SET counts once, regardless of how many reactions carry it.
                        The METHOD-relevant number: how often the model gets a distinct fact right.
    cluster bootstrap   resample CLUSTERS with replacement, not reactions. The honest interval, because it
                        respects the unit at which the answers are actually correlated.

Both are reported for every arm and every subset. Neither is "the" right one -- they answer different
questions and quoting one without the other is what created the problem.

THE PAIRED GAIN IS BOOTSTRAPPED THE SAME WAY, and that is the number that has to survive: if the interval
on (examples - closed_book) excludes zero under cluster resampling, the in-context result is real
regardless of what the level's interval does.

CONCENTRATION IS REPORTED EXPLICITLY, not left implicit in the CI. The largest cluster's share of the bin
is printed, along with what the headline becomes when that cluster is removed. A result that changes a lot
when one gene is dropped is a result about one gene, and the reader should be told which.

PREDECLARED, before any number:
    the paired gain's cluster-bootstrap CI excludes zero
        -> the in-context effect is real at the level of distinct facts, and the ledger keeps it.
    the level's CI is much wider than the reaction-weighted standard error suggests
        -> 0.688 must be carried downstream as an interval; any model consuming it inherits that width.
    one cluster dominates the bin
        -> name it and report the headline with and without it, since a single gene's behaviour is not a
           property of the method.

-> outputs/orphan/cell_orphan_cluster_stats.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_orphan_ties_nn as T  # noqa: E402
import cell_orphan_obscurity as O  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
NBOOT = 20000
SEED = 97


def boot_ci(rng, vals_by_cluster, nboot=NBOOT, q=(2.5, 97.5)):
    """Cluster bootstrap: resample whole CLUSTERS with replacement and recompute.

    Resampling reactions would treat correlated rows as independent, which is exactly the error being
    corrected. Each draw takes the same NUMBER of clusters as observed, so cluster-size variation enters
    the interval rather than being averaged away.
    """
    ks = list(vals_by_cluster)
    m = len(ks)
    if m < 2:
        return float("nan"), float("nan")
    arrs = [np.asarray(vals_by_cluster[k], float) for k in ks]
    out = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, m, m)
        cat = np.concatenate([arrs[i] for i in pick])
        out[b] = cat.mean() if len(cat) else np.nan
    return float(np.nanpercentile(out, q[0])), float(np.nanpercentile(out, q[1]))


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("OBSCURE NUMBERS, RESTATED CLUSTER-WEIGHTED")
    report("=" * 100)
    report("  The obscure bin's 80 reactions are not 80 independent trials: reactions sharing a catalyst")
    report("  are one fact seen several times, and a model gets a whole cluster right or wrong together.")
    report("  That does not move the estimate; it widens the interval, and the level is about to be fed")
    report("  into a cell model as a constant. The paired GAIN is robust to this. The LEVEL is not.")

    B = T.build()
    cats = B["cats"]
    sel = json.load(open(O.PUZZLES))["pids"]
    ans = json.load(open(OUT / "incontext_answers.json"))
    papers = O._lit()
    pool = O._pool(B, papers)
    n = len(sel)
    truth = [{g.upper() for g in cats[i]} for i in sel]
    bidx = np.array([O._binof(pool[i]) for i in sel])
    strict = OUT / "incontext_shown_strict.json"
    src = json.load(open(strict)) if strict.exists() else json.load(open(OUT / "incontext_shown.json"))
    vis = np.array([bool(src.get(str(p), False)) for p in range(n)])

    hits = {a: np.array([str(d.get(str(p), "")).upper() in truth[p] for p in range(n)])
            for a, d in ans.items()}
    ob = json.load(open(OUT / "obscurity_answers.json"))
    hits["closed_book"] = np.array([str((ob.get(str(p)) or {}).get("open", "")).upper() in truth[p]
                                    for p in range(n)])
    ARMS = [a for a in ("closed_book", "examples", "shuffled_examples") if a in hits]

    # the cluster key is the CATALYST SET: two reactions with the same answer are one fact
    ckey = ["|".join(sorted(truth[p])) for p in range(n)]
    rng = np.random.default_rng(SEED)

    SUBSETS = [("ALL", np.ones(n, bool)),
               ("obscure", bidx == 0),
               ("answer NOT shown", ~vis),
               ("obscure AND not shown", (bidx == 0) & ~vis)]

    res = {"n": n, "arms": {}, "subsets": {}}
    report("")
    for lab, m in SUBSETS:
        idx = np.where(m)[0]
        cl = {}
        for p in idx:
            cl.setdefault(ckey[p], []).append(p)
        sizes = sorted(((len(v), k) for k, v in cl.items()), reverse=True)
        big_n, big_k = sizes[0] if sizes else (0, "")
        report(f"  {lab}: {len(idx)} reactions over {len(cl)} distinct catalyst sets "
               f"| largest cluster {big_n} ({big_k[:28]})")
        sub = {"n_reactions": int(len(idx)), "n_clusters": len(cl),
               "largest_cluster": int(big_n), "largest_key": big_k, "arms": {}}
        report(f"    {'arm':<20}{'reaction-wtd':>14}{'cluster-wtd':>13}{'cluster 95% CI':>22}")
        for a in ARMS:
            h = hits[a]
            rw = float(h[m].mean())
            per = {k: h[v] for k, v in cl.items()}
            cw = float(np.mean([h[v].mean() for v in cl.values()]))
            lo, hi = boot_ci(rng, per)
            sub["arms"][a] = {"reaction_weighted": rw, "cluster_weighted": cw, "ci": [lo, hi]}
            report(f"    {a:<20}{rw:>14.3f}{cw:>13.3f}      [{lo:.3f}, {hi:.3f}]")
        # the paired gain, bootstrapped on the same clusters
        if "examples" in hits and "closed_book" in hits:
            d = hits["examples"].astype(float) - hits["closed_book"].astype(float)
            per = {k: d[v] for k, v in cl.items()}
            g = float(d[m].mean())
            lo, hi = boot_ci(rng, per)
            w = int((d[m] > 0).sum())
            l = int((d[m] < 0).sum())
            sub["gain"] = {"mean": g, "ci": [lo, hi], "wins": w, "losses": l}
            star = "excludes 0" if lo > 0 else "INCLUDES 0 -- not resolvable"
            report(f"    {'GAIN ex - closed':<20}{g:>14.3f}{'':>13}      [{lo:.3f}, {hi:.3f}]  "
                   f"{w}w/{l}l  {star}")
        res["subsets"][lab] = sub
        report("")

    # concentration: what the obscure headline becomes without its largest cluster
    m = bidx == 0
    idx = np.where(m)[0]
    cl = {}
    for p in idx:
        cl.setdefault(ckey[p], []).append(p)
    big_k = max(cl, key=lambda k: len(cl[k]))
    keep = np.array([p for p in idx if ckey[p] != big_k])
    report("  CONCENTRATION -- a result that moves when one gene is dropped is a result about that gene.")
    report(f"    largest obscure cluster: {big_k[:40]} ({len(cl[big_k])} of {len(idx)} reactions, "
           f"{len(cl[big_k])/len(idx):.1%})")
    report(f"    {'arm':<20}{'obscure':>10}{'without it':>13}{'delta':>9}")
    for a in ARMS:
        h = hits[a]
        report(f"    {a:<20}{h[m].mean():>10.3f}{h[keep].mean():>13.3f}"
               f"{h[keep].mean()-h[m].mean():>+9.3f}")
    res["concentration"] = {"cluster": big_k, "n": len(cl[big_k]), "of": len(idx),
                            "without": {a: float(hits[a][keep].mean()) for a in ARMS}}

    report("\n  READING")
    o = res["subsets"]["obscure"]
    g = o.get("gain", {})
    lvl = o["arms"]["examples"]
    report(f"    LEVEL: 0.688 reaction-weighted is really {lvl['cluster_weighted']:.3f} per distinct fact,")
    report(f"    with a cluster-bootstrap interval of [{lvl['ci'][0]:.3f}, {lvl['ci'][1]:.3f}] -- far wider")
    report(f"    than the +-0.05 a 80-row standard error implies, because there are only "
           f"{o['n_clusters']} facts.")
    report("    ANY MODEL CONSUMING THIS NUMBER INHERITS THAT WIDTH and should carry it as an interval.")
    if g and g["ci"][0] > 0:
        report(f"    GAIN: {g['mean']:+.3f} with CI [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}], excluding zero.")
        report("    The in-context effect is real at the level of distinct facts, not just distinct rows.")
        report("    That is the claim worth keeping, and it is the one that survives.")
    elif g:
        report(f"    GAIN: CI [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}] includes zero -- the effect is NOT")
        report("    resolvable once correlated rows are counted once, and must not be quoted as measured.")

    OUT.mkdir(parents=True, exist_ok=True)
    res["log"] = log
    json.dump(res, open(OUT / "cell_orphan_cluster_stats.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_cluster_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
