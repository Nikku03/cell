"""BREAKING THE TIES: the missing-piece ranking problem, where the headroom actually is.

THE OPPORTUNITY, measured before anything was built.  The chemistry-neighbour score assigns candidates a
max-Jaccard value, and that value COLLIDES constantly:

    tie-group size at the top score     median 3, mean 8.3, p90 21
    truth is IN the top tie group       638/1,308 = 0.488
    current recall@1 (md5 tie-break)    0.249
    HEADROOM from tie-breaking alone    +0.239

**The score already puts the right enzyme in the top tie group half the time and simply cannot order within
it.** Recall@1 would nearly double if ties were broken perfectly. This is the opposite of the docking
situation, where the answer sat at rank ~1,791 of 9,600 and no reordering could have rescued it.

So the whole question is: does anything order a tie group better than a coin flip?

TIE-BREAKERS, each a per-gene property that the chemistry score does not see:
    freq            how many reactions this gene already catalyses. Promiscuous enzymes are prior-likely.
    ppi_degree      interaction-network connectivity. A hub is a different kind of prior-likely.
    pathway_ctx     THE ONE WITH A MECHANISM. The orphan's chemical neighbours have known catalysts; those
                    catalysts belong to pathways. A tied candidate sharing those pathways is doing related
                    biology. Not circular -- the orphan has no catalyst, so its pathway context is inferred
                    from NEIGHBOURS, and the held-out reaction is excluded throughout.
    compartment     shares a compartment with the reaction. Hurt as a global filter in cell_orphan_context;
                    within a tie group it is a different question and gets a second, fair hearing.
    n_pathways      how many pathways the gene is in at all -- a generality prior, and the control for
                    pathway_ctx: if plain generality does as well, sharing THIS reaction's pathways is not
                    what is working.

CONTROLS:
    random          break ties uniformly. The floor, and it is exactly the current md5 behaviour in
                    expectation.
    oracle          break ties perfectly. The ceiling, 0.488 by construction.

Every arm is evaluated ONLY on the tie group -- the rest of the ranking is identical across arms, so any
difference is attributable to the tie-break and nothing else.

PREDECLARED, before any number:
    a tie-breaker lifts recall@1 materially toward 0.488   -> ranking the missing piece is improvable now, and
                                                              by a lot. The chemistry retrieval was never the
                                                              bottleneck; ordering within it was.
    all tie-breakers ~ random                              -> the tie group is genuinely unordered by anything
                                                              available, and 0.488 is a ceiling we cannot
                                                              approach with these signals.
    pathway_ctx works but n_pathways works equally         -> it is a generality prior, not pathway relevance.
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


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import propagate as pr

    report("=" * 100)
    report("BREAKING THE TIES -- the missing-piece ranking problem, where the headroom actually is")
    report("=" * 100)
    report("  Measured first: the top chemistry score is shared by a median of 3 candidates (p90 21), and the")
    report("  TRUTH is inside that tie group 48.8% of the time against a current recall@1 of 0.249.")
    report("  Perfect tie-breaking would nearly double recall@1. Every arm below differs ONLY in how the tie")
    report("  group is ordered; the rest of the ranking is identical, so any difference is the tie-break.")
    report("  PREDECLARED: a tie-breaker moving materially toward 0.488 => ordering was the bottleneck, not")
    report("  retrieval. All ~ random => the tie group is unordered by anything we have.")

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
    # gene -> compartments, from its OWN catalysed reactions, held-out contribution subtracted at use time
    g2c = defaultdict(Counter)
    for i, cs in cats.items():
        for g in cs:
            g2c[g].update(CMP[i])
    report(f"\n  {len(freq):,} catalysts | pathway annotation for "
           f"{sum(1 for g in freq if pw.get(g)):,} | PPI degree for {sum(1 for g in freq if deg.get(g)):,}")

    rng = np.random.default_rng(A.SEED)
    ev_pool = [i for i in cats if P[i]]
    pick = rng.choice(len(ev_pool), min(N_EVAL, len(ev_pool)), replace=False)
    ev = [ev_pool[int(x)] for x in pick]

    ARMS = ("random", "freq", "ppi_degree", "pathway_ctx", "n_pathways", "compartment", "oracle")
    hits = {a: {k: 0 for k in KS} for a in ARMS}
    tie_sizes, in_tie, n = [], 0, 0

    for i in ev:
        pi = P[i]
        sc = defaultdict(float)
        seen = set()
        neigh = []
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
        n += 1
        truth = set(cats[i])
        top = max(sc.values())
        tied = sorted([c for c, v in sc.items() if v == top])
        rest = sorted([c for c, v in sc.items() if v < top], key=lambda c: -sc[c])
        tie_sizes.append(len(tied))
        in_tie += bool(truth & set(tied))

        # pathway context: pathways of the catalysts of this orphan's chemical NEIGHBOURS (never its own)
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

        keyed = {
            "random": lambda g: rng.random(),
            "freq": lambda g: freq.get(g, 0),
            "ppi_degree": lambda g: deg.get(g, 0),
            "pathway_ctx": lambda g: sum(ctx.get(x, 0) for x in pw.get(g, ())),
            "n_pathways": lambda g: len(pw.get(g, ())),
            "compartment": lambda g: len(comp_of(g) & want_c),
            "oracle": lambda g: 1 if g in truth else 0,
        }
        for a in ARMS:
            f = keyed[a]
            order = sorted(tied, key=lambda g: (-f(g), hashlib.md5(g.encode()).hexdigest())) + rest
            for k in KS:
                hits[a][k] += bool(truth & set(order[:k]))

    tie_sizes = np.array(tie_sizes)
    report(f"\n  {n:,} reactions with candidates | tie-group median {np.median(tie_sizes):.0f}, "
           f"mean {tie_sizes.mean():.1f}, p90 {np.percentile(tie_sizes, 90):.0f}")
    report(f"  truth inside the top tie group: {in_tie:,}/{n:,} = {in_tie/n:.3f}  <- the ceiling for @1")

    report(f"\n  RECALL@k, differing ONLY in how the top tie group is ordered")
    report(f"    {'arm':<14}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    rec = {}
    for a in ARMS:
        rec[a] = {str(k): hits[a][k] / n for k in KS}
        mark = "   <- floor" if a == "random" else ("   <- ceiling" if a == "oracle" else "")
        report(f"    {a:<14}" + "".join(f"{rec[a][str(k)]:>9.3f}" for k in KS) + mark)

    report("\n  READING")
    base, ceil = rec["random"]["1"], rec["oracle"]["1"]
    cands = {a: rec[a]["1"] for a in ARMS if a not in ("random", "oracle")}
    best = max(cands, key=lambda a: cands[a])
    bv = cands[best]
    frac = (bv - base) / max(ceil - base, 1e-9)
    if bv > base + 0.02:
        report(f"  '{best}' lifts recall@1 from {base:.3f} to {bv:.3f} against a ceiling of {ceil:.3f} -- "
               f"{frac:.0%} of the")
        report("  available headroom. Ordering within the tie group was a real and unexploited bottleneck.")
        if best == "pathway_ctx" and cands.get("n_pathways", 0) > bv - 0.02:
            report(f"  BUT n_pathways reaches {cands['n_pathways']:.3f} on its own, so this is a generality")
            report("  prior -- being in many pathways -- rather than being in THIS reaction's pathways.")
    else:
        report(f"  NOTHING ORDERS THE TIE GROUP. Best arm '{best}' at {bv:.3f} against a random tie-break of")
        report(f"  {base:.3f} and a ceiling of {ceil:.3f}. The truth is inside the tie group half the time and")
        report("  none of frequency, degree, pathway context, generality or compartment can find it there.")
        report("  The +0.239 headroom is real but not reachable with these signals.")

    json.dump({"test": "cell_orphan_ties", "n_eval": n, "ks": list(KS), "recall": rec,
               "tie_median": float(np.median(tie_sizes)), "tie_mean": float(tie_sizes.mean()),
               "tie_p90": float(np.percentile(tie_sizes, 90)), "truth_in_tie": in_tie / n,
               "headroom": ceil - base, "log": log},
              open(OUT / "cell_orphan_ties.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'cell_orphan_ties.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
