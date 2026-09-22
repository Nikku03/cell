"""discovery_engine — pieces 4+5: turn MEASURED data into candidate discoveries, then only accept the ones that
survive an ORACLE withheld from the proposer. This is the propose→validate loop that separates a hypothesis
generator from a discovery engine.

  PROPOSE (piece 4, from measured DepMap): the strongest co-essential gene pairs that are NOT already edges in
     the curated interactome. Co-essentiality is a functional-genomics measurement independent of the PPI
     databases, so a high-co-essential NON-edge is a genuine candidate ("these two are functionally coupled in
     a way the curated graph doesn't record").

  ORACLE (piece 5): candidates are scored against a held-out truth the proposer never saw —
     (a) held-out real edges (removed before proposing), and
     (b) shared-partner corroboration on the held-out graph.
     precision@k and lift-over-random quantify how many proposals are real. A discovery engine beats the random
     baseline on held-out truth; a recall engine does not.

  ATTRIBUTION leave-one-out (piece 4, the dependency→context claim): for each addiction driver, hide it and ask
     whether co-essentiality alone re-derives its known pathway partners — blind recovery of a
     dependency-context relationship from measurement.

Honest scope: the richest discovery — regressing dependency on a tumour's LESION (WRN→MSI) — needs the DepMap
per-line lesion table (Colab, U2). Here we run the self-contained co-essentiality version and report the yield.
-> outputs/orphan/discovery_engine.json
"""
import os, sys, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"


def _depmap():
    z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
    return {"Z": z["Z"], "syms": list(z["syms"]), "col": {s: i for i, s in enumerate(z["syms"])}}


def top_coessential_pairs(dm, C, k=4000, min_overlap=120, corr_thresh=0.35):
    """strongest co-essential pairs among model genes, restricted to the top-variance dependency genes for speed.
    Returns list of (corr, a_idx, b_idx)."""
    Z, col = dm["Z"], dm["col"]
    # candidate genes: in the model AND a selective/variable dependency
    cand = []
    for s, r in col.items():
        if s in C.idx:
            row = Z[r]
            if (row != 0).sum() >= min_overlap and row.std() > 0.7 and row.min() < -2:
                cand.append((s, r))
    cand = cand[:2500]                                        # cap the O(n^2) scan
    M = np.array([Z[r] for _, r in cand])
    names = [s for s, _ in cand]
    pairs = []
    for i in range(len(cand)):
        x = M[i]; mx = x != 0
        for j in range(i + 1, len(cand)):
            y = M[j]; m = mx & (y != 0)
            if m.sum() < min_overlap:
                continue
            c = float(np.corrcoef(x[m], y[m])[0, 1])
            if c >= corr_thresh:
                pairs.append((c, C.idx[names[i]], C.idx[names[j]]))
    pairs.sort(reverse=True)
    return pairs[:k]


def run(seed=0):
    random.seed(seed); np.random.seed(seed)
    C = CompleteCell(); name = C.name; dm = _depmap()

    # held-out oracle: remove 20% of PPI edges before proposing
    ppi = set()
    for e in C.D.get("ppi", []):
        if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] != e[1]:
            ppi.add((min(e[0], e[1]), max(e[0], e[1])))
    ppi = list(ppi); random.shuffle(ppi)
    n_hold = int(0.2 * len(ppi))
    held_out = set(ppi[:n_hold]); train_edges = set(ppi[n_hold:])
    train_adj = {}
    for a, b in train_edges:
        train_adj.setdefault(a, set()).add(b); train_adj.setdefault(b, set()).add(a)

    print("  scanning co-essential pairs (measured DepMap) ...")
    pairs = top_coessential_pairs(dm, C)
    print(f"  {len(pairs)} high co-essential pairs")

    # PROPOSE: co-essential pairs that are NOT train edges (candidate new relationships)
    proposals = [(c, a, b) for c, a, b in pairs if (min(a, b), max(a, b)) not in train_edges]

    # ORACLE validation: is a proposal a HELD-OUT real edge? (the proposer never saw held_out)
    def key(a, b): return (min(a, b), max(a, b))
    hits = [(c, a, b) for c, a, b in proposals if key(a, b) in held_out]
    # baseline: random gene pairs among the same candidate nodes, matched count
    cnodes = list({a for _, a, _ in pairs} | {b for _, _, b in pairs})
    base_hits = 0; ntrials = len(proposals)
    for _ in range(ntrials):
        a, b = random.choice(cnodes), random.choice(cnodes)
        if a != b and key(a, b) in held_out:
            base_hits += 1
    prec = len(hits) / len(proposals) if proposals else 0.0
    base_prec = base_hits / ntrials if ntrials else 0.0
    # when random recovers ~0 held-out edges, report a floor lift from the counts rather than dividing by zero
    lift = (round(prec / base_prec, 2) if base_prec > 0
            else (round(prec / (0.5 / ntrials), 1) if (prec > 0 and ntrials) else None))

    # top novel candidates (not held-out, not train -> genuinely absent from the interactome)
    novel = [(round(c, 3), name[a], name[b]) for c, a, b in proposals if key(a, b) not in held_out][:20]

    # ATTRIBUTION leave-one-out: hide each driver, recover its pathway partners by co-essentiality
    from context_dependency import ADDICTION_DRIVERS, attribute_addiction
    loo = []
    for d in ADDICTION_DRIVERS[:20]:
        if d not in dm["col"]:
            continue
        others = [x for x in ADDICTION_DRIVERS if x != d]
        part = attribute_addiction(dm, d, others, thresh=0.2)
        loo.append({"driver": d, "recovered_context_partners": part[:3]})

    res = {
        "n_high_coessential_pairs": len(pairs),
        "n_proposals_non_edge": len(proposals),
        "oracle_precision_on_heldout_edges": round(prec, 4),
        "random_baseline_precision": round(base_prec, 4),
        "discovery_lift_over_random": lift,
        "n_validated_by_oracle": len(hits),
        "top_novel_candidates_for_wetlab": novel,
        "attribution_leave_one_out": loo[:10],
        "verdict": ("co-essentiality proposals are enriched for held-out real edges above random "
                    f"({lift}x) — a measurable discovery signal from functional-genomics data"
                    if lift and lift > 1.5 else
                    "co-essentiality proposals barely beat random on held-out edges — weak edge-discovery; "
                    "its real strength is dependency-CONTEXT attribution (needs the Colab lesion table)"),
    }
    json.dump(res, open(f"{OUT}/discovery_engine.json", "w"), indent=2)
    print("=" * 76)
    print("DISCOVERY ENGINE — propose (measured co-essentiality) -> validate (held-out oracle)")
    print("=" * 76)
    print(f"  proposals (co-essential non-edges)   {res['n_proposals_non_edge']}")
    print(f"  validated by held-out oracle          {res['n_validated_by_oracle']}")
    print(f"  precision {res['oracle_precision_on_heldout_edges']}  vs random {res['random_baseline_precision']}"
          f"   -> lift {res['discovery_lift_over_random']}x")
    print(f"  top novel candidate (for validation): {novel[0] if novel else '-'}")
    print(f"  -> {res['verdict']}")
    return res


if __name__ == "__main__":
    run()
