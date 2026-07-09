"""domain_ppi — wire the DEAD-END domain layer into the PPI network.

Domains mediate physical interaction (an SH3 domain binds a proline-rich motif, etc.), yet our 16,492-gene
domain architecture feeds nothing — it is folded into .gene() and stops there. This learns, self-supervised
from the KNOWN PPI edges, which InterPro domain PAIRS are enriched among interacting proteins, and turns that
into a per-pair 'domain_compatibility' score that can (a) become a new PPI-combiner feature and (b) predict
edges the structural/co-essentiality features are blind to.

Method (no external data — only domains + our own PPI):
  1. positives = known PPI edges; negatives = random non-edges (matched count). Split 50/50 train/test.
  2. on TRAIN only, for every domain pair (di,dj) seen in a pair, tally positive vs negative occurrence and score
        enrichment(di,dj) = log2( (p/Np + a) / (n/Nn + a) )        # log-odds, a = Laplace smoothing
  3. score a protein pair (A,B):  domain_compat = max over di in dom(A), dj in dom(B) of enrichment(di,dj)
  4. evaluate on the held-out TEST split: AUC of domain_compat alone, and — the real point — how well it scores
     the test edges that have ZERO shared PPI partners (triadic closure is structurally blind to those).

Honest: enrichment is fit on TRAIN and scored on TEST, so the AUC is out-of-sample, not circular.
-> outputs/orphan/domain_ppi_validation.json
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
from complete_cell import CompleteCell

OUT = "outputs/orphan"
ALPHA = 1e-4                                   # Laplace smoothing on the odds


def _auc(pos_scores, neg_scores):
    """rank-based AUC = P(score(pos) > score(neg))."""
    xs = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores])
    rank = 0.0; npos = len(pos_scores); nneg = len(neg_scores)
    i = 0; n = len(xs)
    while i < n:                                # average ranks over ties
        j = i
        while j < n and xs[j][0] == xs[i][0]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            if xs[k][1] == 1:
                rank += avg
        i = j
    if npos == 0 or nneg == 0:
        return None
    return (rank - npos * (npos + 1) / 2.0) / (npos * nneg)


def run():
    C = CompleteCell()
    dom = {g: set(v) for g, v in getattr(C, "domains", {}).items() if v}     # gene idx -> {InterPro ids}
    adj = C.ppi_adj
    n = len(C.name)
    print(f"  genes with domains: {len(dom):,} | PPI adjacency over {n:,} nodes")

    # ---- build positive edges (deterministic sample) and matched random negatives ----
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v and u in dom and v in dom:            # both endpoints must have domains
                edges.append((u, v))
    edges.sort()
    # deterministic pseudo-shuffle (no RNG): interleave by a hash of the pair
    edges.sort(key=lambda e: (e[0] * 2654435761 + e[1]) & 0xFFFFFFFF)
    CAP = 40000
    edges = edges[:CAP]
    have = set(edges) | {(min(u, v), max(u, v)) for u in adj for v in adj[u]}
    domlist = [g for g in dom]
    negs = []
    i = 0
    # deterministic negatives: walk domlist pairs by a strided index, skip real edges
    while len(negs) < len(edges) and i < len(edges) * 50:
        a = domlist[(i * 7919) % len(domlist)]
        b = domlist[(i * 104729 + 17) % len(domlist)]
        i += 1
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in have:
            continue
        negs.append(key)
    print(f"  positives: {len(edges):,} | negatives: {len(negs):,}")

    half_p, half_n = len(edges) // 2, len(negs) // 2
    tr_pos, te_pos = edges[:half_p], edges[half_p:]
    tr_neg, te_neg = negs[:half_n], negs[half_n:]

    # ---- learn domain-pair enrichment on TRAIN only ----
    from collections import defaultdict
    pc = defaultdict(int); nc = defaultdict(int)

    def pairs_of(a, b):
        for di in dom[a]:
            for dj in dom[b]:
                yield (di, dj) if di <= dj else (dj, di)

    for (a, b) in tr_pos:
        for dp in set(pairs_of(a, b)):
            pc[dp] += 1
    for (a, b) in tr_neg:
        for dp in set(pairs_of(a, b)):
            nc[dp] += 1
    Np, Nn = len(tr_pos), len(tr_neg)
    enr = {}
    for dp in set(pc) | set(nc):
        p = pc.get(dp, 0) / Np
        nneg = nc.get(dp, 0) / Nn
        enr[dp] = math.log2((p + ALPHA) / (nneg + ALPHA))
    print(f"  scored domain pairs: {len(enr):,}")

    def compat(a, b):
        best = None
        for dp in pairs_of(a, b):
            e = enr.get(dp)
            if e is not None and (best is None or e > best):
                best = e
        return best if best is not None else 0.0

    # ---- evaluate on TEST ----
    te_pos_s = [compat(a, b) for (a, b) in te_pos]
    te_neg_s = [compat(a, b) for (a, b) in te_neg]
    auc = _auc(te_pos_s, te_neg_s)

    # complementarity: test edges with ZERO shared PPI neighbours (triadic-blind)
    def shared(a, b):
        return len(adj.get(a, set()) & adj.get(b, set()))
    blind = [(a, b) for (a, b) in te_pos if shared(a, b) == 0]
    blind_s = [compat(a, b) for (a, b) in blind]
    blind_neg_s = te_neg_s
    auc_blind = _auc(blind_s, blind_neg_s)

    # ---- MARGINAL GAIN: does domain_compat add over the structural + co-essentiality features? ----
    D = C.D
    codep = {}
    for g, lst in D.get("codep", {}).items():
        gi = int(g)
        for p, s in lst:
            key = (min(gi, p), max(gi, p))
            codep[key] = max(codep.get(key, 0.0), s)
    g2c = {int(k): set(v) for k, v in D.get("gene2cplx", {}).items()}
    diso = getattr(C, "disorder", {}) or {}                 # #4 disorder->PPI (present only on the Colab run)
    has_diso = len(diso) > 0

    def feats(a, b, with_domain, with_disorder=False):
        ua, ub = adj.get(a, set()), adj.get(b, set())
        sp = len(ua & ub)
        jac = sp / (len(ua | ub) or 1)
        sc = 1.0 if (g2c.get(a, set()) & g2c.get(b, set())) else 0.0
        ce = codep.get((min(a, b), max(a, b)), 0.0)
        row = [sp, jac, sc, ce]
        if with_domain:
            row.append(compat(a, b))
        if with_disorder:
            row.append(diso.get(a, 0.0) * diso.get(b, 0.0))    # both disordered -> promiscuous interaction
        return row

    marg = None
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        def design(pairs_pos, pairs_neg, wd, wdis=False):
            X = [feats(a, b, wd, wdis) for (a, b) in pairs_pos] + [feats(a, b, wd, wdis) for (a, b) in pairs_neg]
            y = [1] * len(pairs_pos) + [0] * len(pairs_neg)
            return np.array(X, float), np.array(y)

        def fit_auc(wd, wdis=False):
            Xtr, ytr = design(tr_pos, tr_neg, wd, wdis)
            Xte, yte = design(te_pos, te_neg, wd, wdis)
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
            pte = m.predict_proba(sc.transform(Xte))[:, 1]
            return _auc([pte[i] for i in range(len(yte)) if yte[i] == 1],
                        [pte[i] for i in range(len(yte)) if yte[i] == 0])

        auc_struct = fit_auc(False)
        auc_both = fit_auc(True)
        marg = {"auc_structural_only": round(auc_struct, 4), "auc_with_domain": round(auc_both, 4),
                "delta_auc": round(auc_both - auc_struct, 4)}
        if has_diso:                                         # #4 disorder->PPI (Colab only)
            auc_dd = fit_auc(True, True)
            marg["auc_with_domain_and_disorder"] = round(auc_dd, 4)
            marg["delta_auc_disorder"] = round(auc_dd - auc_both, 4)
        else:
            marg["disorder"] = "absent in this run (Colab-generated) — #4 validates there"
    except Exception as e:
        marg = {"error": str(e)[:80]}

    # top interacting domain pairs (sanity)
    top = sorted(enr.items(), key=lambda kv: -kv[1])[:8]
    res = {
        "n_genes_with_domains": len(dom),
        "n_positive_edges": len(edges), "n_negative": len(negs),
        "n_domain_pairs_scored": len(enr),
        "auc_domain_compat": round(auc, 3) if auc else None,
        "test_edges_triadic_blind": len(blind),
        "auc_on_triadic_blind": round(auc_blind, 3) if auc_blind else None,
        "top_interacting_domain_pairs": [{"pair": list(dp), "log2_enrich": round(e, 2)} for dp, e in top],
        "marginal_gain_vs_structural": marg,
        "verdict": None,
    }
    res["verdict"] = ("domain compatibility predicts held-out PPI out-of-sample"
                      if auc and auc > 0.6 else
                      "weak/no signal — domains do not add to PPI here")
    json.dump(res, open(f"{OUT}/domain_ppi_validation.json", "w"))
    print("=" * 70)
    print("DOMAINS -> PPI  (self-supervised domain-pair interaction feature)")
    print("=" * 70)
    print(f"  AUC domain_compat (out-of-sample)   {res['auc_domain_compat']}")
    print(f"  AUC on triadic-BLIND edges          {res['auc_on_triadic_blind']}  "
          f"({res['test_edges_triadic_blind']:,} edges with 0 shared partners)")
    print(f"  domain pairs scored                 {res['n_domain_pairs_scored']:,}")
    if marg and "delta_auc" in marg:
        print(f"  combiner AUC: structural {marg['auc_structural_only']} -> +domain {marg['auc_with_domain']}"
              f"  (Δ {marg['delta_auc']:+})")
    print(f"  -> {res['verdict']}")
    print("=" * 70)
    return res


if __name__ == "__main__":
    run()
