"""struct_discovery_test — the ONE honest test the user asked for: does adding a STRUCTURE-derived feature
(domain-domain interaction compatibility) lift the DISCOVERY (HARD-split) AUC off 0.5 — computed the SAME way
for positives and negatives, with NO co-complex / label shortcut?

Why domains, not ESM interface-conservation: computing an interface-complementarity feature needs a predicted 3D
complex per pair (AlphaFold-Multimer) — infeasible here (no GPU) and, if only done for pairs with a solved
co-complex, it LEAKS (having the structure IS the label). Domain-domain interaction compatibility is the
feasible, leak-CONTROLLABLE structural signal: it is an intrinsic per-protein property (domain architecture),
orthogonal to the network graph, and its enrichment can be learned on the TRAIN split only.

Harness = the established discovery holdout (discovery_validation.py):
  - split PPI edges 80/20 train/test; rebuild the graph with TEST edges removed.
  - HARD held-out edge = endpoints share ZERO train neighbours (triadic closure useless = the discovery regime).
  - baseline score = topology (shared/jaccard) + co-essentiality (the current discovery_validation score).
  - + domain score = domain_compat, with domain-pair enrichment learned on TRAIN positives vs TRAIN negatives.

Two confound controls the original domain_ppi.py LACKED (it used random negatives + reported blind-pos-vs-ALL-neg):
  (1) MATCHED negatives — for each HARD positive (a,b), a decoy (a,b') with the SAME anchor a and b' matched on
      #domains and degree. Kills the "multi-domain / hub proteins score higher by max-over-pairs" inflation.
  (2) n_domains-only baseline — AUC of just (#domains(a) x #domains(b)); shows how much apparent signal is merely
      protein complexity, not specific complementarity.

Decisive number: domain AUC on HARD positives vs MATCHED negatives. Off 0.5 -> structural features genuinely add
discovery signal (integrate them). Back near 0.5 -> the 0.856 was confound, and structure does not rescue discovery.
-> outputs/orphan/struct_discovery_test.json
"""
import os, sys, json, math, random
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
ALPHA = 1e-4


def _auc(pos, neg):
    """rank-based AUC = P(score(pos) > score(neg)), ties at 0.5."""
    if not pos or not neg:
        return None
    xs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank = 0.0; i = 0; n = len(xs)
    while i < n:
        j = i
        while j < n and xs[j][0] == xs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if xs[k][1] == 1:
                rank += avg
        i = j
    npos, nneg = len(pos), len(neg)
    return (rank - npos * (npos + 1) / 2.0) / (npos * nneg)


def _matched_auc(pairs):
    """pairs = [(score_true, score_decoy)]; matched-pair AUC = P(true > decoy), ties 0.5."""
    if not pairs:
        return None
    w = sum(1.0 for t, d in pairs if t > d) + 0.5 * sum(1.0 for t, d in pairs if t == d)
    return w / len(pairs)


def _depmap():
    p = f"{OUT}/depmap_vecs.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {"Z": z["Z"], "col": {s: i for i, s in enumerate(z["syms"])}}


def run(frac=0.2, n_hard=3000, seed=0):
    random.seed(seed); np.random.seed(seed)
    C = CompleteCell(); name = C.name
    dom = {g: set(v) for g, v in getattr(C, "domains", {}).items() if v}
    dm = _depmap()

    # ---- split edges ----
    E = set()
    for e in C.D.get("ppi", []):
        if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] != e[1]:
            E.add((min(e[0], e[1]), max(e[0], e[1])))
    if not E:  # fall back to adjacency if D['ppi'] absent
        for u, ps in C.ppi_adj.items():
            for v in ps:
                if u != v:
                    E.add((min(u, v), max(u, v)))
    E = sorted(E); random.shuffle(E)
    n_hold = int(len(E) * frac)
    test, train = E[:n_hold], E[n_hold:]
    Eset = set(E)

    # TRAIN adjacency (leak-free graph)
    adj = defaultdict(set)
    for a, b in train:
        adj[a].add(b); adj[b].add(a)
    deg = {g: len(adj[g]) for g in adj}

    # ---- learn domain-pair enrichment on TRAIN only (leak-free) ----
    def pairs_of(a, b):
        for di in dom.get(a, ()):
            for dj in dom.get(b, ()):
                yield (di, dj) if di <= dj else (dj, di)

    tr_pos = [(a, b) for a, b in train if a in dom and b in dom]
    domnodes = [g for g in dom if g in deg]  # domain-bearing nodes that appear in train graph
    tr_neg = []
    while len(tr_neg) < len(tr_pos) and len(tr_neg) < 200000:
        a, b = random.choice(domnodes), random.choice(domnodes)
        if a != b and (min(a, b), max(a, b)) not in Eset:
            tr_neg.append((a, b))
    pc, nc = defaultdict(int), defaultdict(int)
    for a, b in tr_pos:
        for dp in set(pairs_of(a, b)):
            pc[dp] += 1
    for a, b in tr_neg:
        for dp in set(pairs_of(a, b)):
            nc[dp] += 1
    Np, Nn = len(tr_pos), len(tr_neg)
    enr = {dp: math.log2((pc.get(dp, 0) / Np + ALPHA) / (nc.get(dp, 0) / Nn + ALPHA))
           for dp in set(pc) | set(nc)}

    def compat(a, b):
        best = 0.0
        for dp in pairs_of(a, b):
            e = enr.get(dp)
            if e is not None and e > best:
                best = e
        return best

    def ndom_prod(a, b):
        return len(dom.get(a, ())) * len(dom.get(b, ()))

    def coess(a, b):
        if not dm:
            return 0.0
        ca, cb = dm["col"].get(name[a]), dm["col"].get(name[b])
        if ca is None or cb is None:
            return 0.0
        x, y = dm["Z"][ca], dm["Z"][cb]; m = (x != 0) & (y != 0)
        if m.sum() < 60:
            return 0.0
        return float(np.corrcoef(x[m], y[m])[0, 1])

    def topo(a, b):
        na, nb = adj.get(a, set()), adj.get(b, set())
        shared = len(na & nb); union = len(na | nb) or 1
        return 0.5 * (shared / union) + 0.1 * min(shared, 10) / 10 + 0.6 * max(0.0, coess(a, b))

    # ---- HARD held-out positives: 0 shared TRAIN neighbours, both endpoints have domains ----
    hard_pos = [(a, b) for a, b in test
                if a in dom and b in dom and len(adj.get(a, set()) & adj.get(b, set())) == 0]
    random.shuffle(hard_pos); hard_pos = hard_pos[:n_hard]

    # ---- negatives ----
    # N1 random non-edges (both have domains) — what domain_ppi used
    n1 = []
    while len(n1) < len(hard_pos):
        a, b = random.choice(domnodes), random.choice(domnodes)
        if a != b and (min(a, b), max(a, b)) not in Eset:
            n1.append((a, b))

    # N2 MATCHED decoys: same anchor a, decoy b' with SAME #domains as true b and degree within a bin; not an edge
    by_ndom_deg = defaultdict(list)
    for g in domnodes:
        by_ndom_deg[(len(dom[g]), min(deg.get(g, 0), 50) // 5)].append(g)
    matched = []  # (anchor, true_b, decoy_b)
    for a, b in hard_pos:
        key = (len(dom[b]), min(deg.get(b, 0), 50) // 5)
        cands = by_ndom_deg.get(key, [])
        bp = None
        for _ in range(20):
            c = random.choice(cands) if cands else random.choice(domnodes)
            if c != a and c != b and (min(a, c), max(a, c)) not in Eset:
                bp = c; break
        if bp is not None:
            matched.append((a, b, bp))

    # ---- score everything ----
    def sc_pool(fn, pos, neg):
        return _auc([fn(a, b) for a, b in pos], [fn(a, b) for a, b in neg])

    res = {
        "n_test_edges": len(test), "n_hard_pos": len(hard_pos),
        "n_domain_pairs_scored": len(enr),
        "HARD_pos_vs_random_neg": {
            "topology+coess": round(sc_pool(topo, hard_pos, n1) or 0, 3),
            "domain_compat":   round(sc_pool(compat, hard_pos, n1) or 0, 3),
            "n_domains_only":  round(sc_pool(ndom_prod, hard_pos, n1) or 0, 3),
        },
        "HARD_pos_vs_MATCHED_neg (decoy = same anchor, #domains+degree matched)": {
            "n_matched_pairs": len(matched),
            "topology+coess": round(_matched_auc([(topo(a, b), topo(a, c)) for a, b, c in matched]) or 0, 3),
            "domain_compat":   round(_matched_auc([(compat(a, b), compat(a, c)) for a, b, c in matched]) or 0, 3),
            "n_domains_only":  round(_matched_auc([(ndom_prod(a, b), ndom_prod(a, c)) for a, b, c in matched]) or 0, 3),
        },
    }
    # verdict keyed on the DECISIVE number: domain_compat vs matched negatives
    dm_matched = res["HARD_pos_vs_MATCHED_neg (decoy = same anchor, #domains+degree matched)"]["domain_compat"]
    dm_random = res["HARD_pos_vs_random_neg"]["domain_compat"]
    res["decisive"] = {
        "domain_compat_vs_random": dm_random,
        "domain_compat_vs_matched": dm_matched,
        "confound_inflation": round(dm_random - dm_matched, 3),
        "verdict": (
            "STRUCTURE ADDS DISCOVERY SIGNAL: domain compatibility separates true partners from domain/degree-"
            "matched decoys on zero-shared-neighbour edges — integrate it." if dm_matched >= 0.62 else
            "MOSTLY CONFOUND: the vs-random AUC was protein-complexity/hubness; against matched decoys the "
            "structural feature is near chance — it does NOT rescue discovery." if dm_matched < 0.57 else
            "WEAK: a little real signal survives matching, but far below the vs-random headline."),
    }
    json.dump(res, open(f"{OUT}/struct_discovery_test.json", "w"), indent=2)

    print("=" * 78)
    print("STRUCT-FEATURE DISCOVERY TEST  (domain compatibility on the HARD/zero-shared split)")
    print("=" * 78)
    print(f"  held-out HARD positives (0 shared train neighbours): {len(hard_pos)}")
    print("\n  HARD positives vs RANDOM negatives (what domain_ppi reported):")
    for k, v in res["HARD_pos_vs_random_neg"].items():
        print(f"      {k:18} AUC {v}")
    print("\n  HARD positives vs MATCHED negatives (same anchor, #domains+degree matched):")
    mm = res["HARD_pos_vs_MATCHED_neg (decoy = same anchor, #domains+degree matched)"]
    for k in ("topology+coess", "domain_compat", "n_domains_only"):
        print(f"      {k:18} AUC {mm[k]}")
    print(f"\n  DECISIVE  domain_compat: {dm_random} (vs random)  ->  {dm_matched} (vs matched)   "
          f"[confound inflation {res['decisive']['confound_inflation']}]")
    print(f"  -> {res['decisive']['verdict']}")
    print("=" * 78)
    return res


if __name__ == "__main__":
    run()
