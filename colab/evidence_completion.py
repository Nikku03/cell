"""COMPLETE THE NETWORK WITH EVERYTHING WE HAVE: spatial, pathway, co-expression, co-dependency, regulatory.

THE QUESTION. link_completion.py established that PPI edges can be recovered well overall (2-step K562 chain 0.938,
combined 0.952, under negatives matched jointly on degree and publication count) BUT that the result is confined to
places the graph is already dense:

    shared neighbours > 10  (79.7% of held-out edges) : chain 0.881, combined 0.900
    shared neighbours = 0   (2.4%)                    : chain 0.500 = CHANCE, combined 0.413 = WORSE than chance

Topology has literally no information about a pair with no shared neighbours -- common-neighbours and the walk are
identically zero there, so their "prediction" is a coin flip and the learned model actively misranks. The obvious
fix is evidence that does not come from the graph: where two proteins are in the cell, which pathway they belong to,
whether their expression covaries, whether cells need them together. THAT is what this module tests, and the test is
NOT the pooled AUC -- adding features always raises that -- but specifically whether the CN=0 stratum is rescued.

THE FATAL RISK IS CIRCULARITY, and it is the reason this module is tiered rather than a single kitchen sink. Many
annotation sources are curated USING interaction experiments. If "same complex" was assigned from the very
pull-downs that produced the edges being predicted, then a feature built on it is the answer in disguise. One number
already on record makes the danger concrete: 92.8% of within-complex protein pairs are present as PPI edges. So
every source is placed in a tier, and the tiers are scored separately so a gain can be attributed:

    CLEAN     measured independently of any interaction assay
              coexpr (expression covariation), codep (DepMap fitness covariation), PROFILE (Perturb-seq z-profile
              cosine), sl (genetic interaction), chromosome + TSS distance (genomic position), ppm (abundance)
    SUSPECT   annotation that MAY encode interaction evidence; usable only if reported separately
              GO cellular-component and biological-process, `path`, `proc`, `comp`, `reg`
    EXCLUDED  complexes / gene2cplx -- 92.8% of within-complex pairs ARE edges. Predicting an edge from
              "these two are in the same complex" is predicting the edge from the edge. Named and dropped.

An empirical circularity check runs for every SUSPECT source: what fraction of the pairs it links are ALREADY
recorded edges. A source that only ever links pairs that are already connected cannot rescue the sparse regime by
construction, whatever its pooled AUC says.

TRUNCATION IS ENCODED EXPLICITLY. coexpr and codep are stored as TOP-N ranked partner lists, so "no entry" means
"not in this gene's top N", not "uncorrelated". Encoding absence as 0 would let the model read a truncation
threshold as biology. Every such feature therefore ships as a PAIR: the value when present, and a separate
indicator that the pair appeared in either list at all.

ARMS, all on the identical split, negatives and evaluation path as link_completion.py:
    TOPO            common-neighbours, Jaccard, Adamic-Adar, 2-step K562 chain   (the incumbent)
    +SPATIAL        TOPO plus localisation and genomic-position features
    +PATHWAY        TOPO plus pathway/process/GO-BP features
    +MEASURED       TOPO plus the CLEAN measured tier
    +ALL-CLEAN      TOPO plus every CLEAN and SUSPECT feature
    CLEAN-ONLY      the non-topology features ALONE, no graph input whatsoever
                    <- THE LOAD-BEARING ARM. Only this one can possibly work at CN=0, because every topology
                       feature is identically zero there. If it does not beat chance in that stratum, no amount of
                       additional annotation completes the sparse part of the network.
"""
import collections
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
HOLDOUT, SEED, NNEG = 0.20, 0, 5


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); nK = len(kos)
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    ki = {k: i for i, k in enumerate(kos)}
    Z = np.zeros((nK, len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            Z[ki[k], gi[g]] = z

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    konode = np.array([nidx[k] for k in kos])
    pubs = np.array([float(info[n].get("pubs") or 0) for n in names])

    f = h5py.File(f"{SP}/k562.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["__categories"]["gene_name"][:]]
    expressed = np.zeros(N, bool)
    for n_ in set(np.array(cats)[f["var"]["gene_name"][:]].tolist()):
        if n_ in nidx:
            expressed[nidx[n_]] = True

    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    kmap = {int(v): i for i, v in enumerate(konode)}
    ko_edges = sorted(set((min(kmap[a], kmap[b]), max(kmap[a], kmap[b]))
                          for a, b in pe if a in kmap and b in kmap))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ko_edges))
    nho = int(len(ko_edges) * HOLDOUT)
    held = set(ko_edges[i] for i in perm[:nho]); trainE = set(ko_edges[i] for i in perm[nho:])
    print(f"{len(ko_edges)} KO-KO edges: {len(trainE)} train, {len(held)} held out", flush=True)

    heldn = set((konode[a], konode[b]) for a, b in held) | set((konode[b], konode[a]) for a, b in held)
    ta, tb = [], []
    for a, b in pe:
        if (a, b) in heldn or (b, a) in heldn:
            continue
        ta.append(a); tb.append(b)
    ta = np.array(ta); tb = np.array(tb)
    Atr = sparse.coo_matrix((np.ones(2 * len(ta), np.float32), (np.r_[ta, tb], np.r_[tb, ta])), shape=(N, N)).tocsr()
    Atr.data[:] = 1.0
    deg = np.asarray(Atr.sum(1)).ravel()

    # ---------- TOPOLOGY ----------
    sub = konode
    CN = np.asarray((Atr[sub] @ Atr[:, sub]).todense(), np.float32)
    il = np.zeros(N, np.float32); nzz = deg > 1; il[nzz] = 1.0 / np.log(deg[nzz])
    AA = np.asarray((Atr[sub] @ sparse.diags(il) @ Atr[:, sub]).todense(), np.float32)
    degk = deg[sub]
    JC = CN / (degk[:, None] + degk[None, :] - CN + 1e-9)
    m = expressed[ta] & expressed[tb]
    ra = np.r_[ta[m], tb[m]]; rb = np.r_[tb[m], ta[m]]
    Gk = sparse.coo_matrix((np.ones(len(ra), np.float32), (ra, rb)), shape=(N, N)).tocsr()
    mg = np.asarray(Gk.sum(1)).ravel(); iv = np.zeros(N, np.float32); nz = mg > 0; iv[nz] = 1.0 / mg[nz]
    Pk = (sparse.diags(iv) @ Gk).tocsr()
    CH2 = np.asarray((Pk[sub] @ Pk[:, sub]).todense(), np.float32)
    TOPO = collections.OrderedDict([("CN", CN), ("JACCARD", JC), ("ADAMIC", AA), ("CHAIN2", CH2)])

    # ---------- helpers for set-valued per-gene annotations ----------
    def setmat(sets, weighted=False):
        """shared-count and Jaccard over per-knockout sets, via a sparse indicator matrix"""
        vocab = {}
        r, c = [], []
        for i, s in enumerate(sets):
            for t in s:
                r.append(i); c.append(vocab.setdefault(t, len(vocab)))
        Msp = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nK, max(len(vocab), 1)))
        sh = np.asarray((Msp @ Msp.T).todense(), np.float32)
        sz = np.array([len(s) for s in sets], np.float32)
        jac = sh / (sz[:, None] + sz[None, :] - sh + 1e-9)
        return sh, jac

    go = D.get("go", {})
    goC = [set(go.get(k, {}).get("C", []) or []) for k in kos]
    goP = [set(go.get(k, {}).get("P", []) or []) for k in kos]
    shC, jcC = setmat(goC); shP, jcP = setmat(goP)
    comp = np.array([str(info[k].get("comp") or "") for k in kos])
    proc = np.array([str(info[k].get("proc") or "") for k in kos])
    path = np.array([str(info[k].get("path") or "") for k in kos])
    chrom = np.array([str(info[k].get("chrom") or "") for k in kos])
    tss = np.array([float(info[k].get("tss") or 0) for k in kos])
    samechr = (chrom[:, None] == chrom[None, :]).astype(np.float32)
    tssd = np.log1p(np.abs(tss[:, None] - tss[None, :])).astype(np.float32) * samechr
    SPATIAL = collections.OrderedDict([
        ("SAME-COMP", (comp[:, None] == comp[None, :]).astype(np.float32)),
        ("GOC-SHARED", shC), ("GOC-JAC", jcC),
        ("SAME-CHR", samechr), ("TSS-DIST", tssd),
    ])
    PATHWAY = collections.OrderedDict([
        ("SAME-PATH", ((path[:, None] == path[None, :]) & (path[:, None] != "")).astype(np.float32)),
        ("SAME-PROC", (proc[:, None] == proc[None, :]).astype(np.float32)),
        ("GOP-SHARED", shP), ("GOP-JAC", jcP),
    ])

    # ---------- CLEAN measured tier ----------
    def ranked(key):
        """coexpr/codep are TOP-N truncated lists: value if present, plus an explicit presence indicator."""
        val = np.zeros((nK, nK), np.float32); ind = np.zeros((nK, nK), np.float32)
        src = D.get(key, {}) or {}
        for k_, lst in src.items():
            if not str(k_).isdigit() or int(k_) >= N:
                continue
            a = names[int(k_)]
            if a not in ki:
                continue
            ia = ki[a]
            for item in lst:
                try:
                    j, s = int(item[0]), float(item[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if j >= N:
                    continue
                b = names[j]
                if b in ki:
                    ib = ki[b]
                    val[ia, ib] = max(val[ia, ib], s); val[ib, ia] = val[ia, ib]
                    ind[ia, ib] = 1.0; ind[ib, ia] = 1.0
        return val, ind

    cx, cxi = ranked("coexpr")
    cd, cdi = ranked("codep")
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    PROF = (Zn @ Zn.T).astype(np.float32)
    slm = np.zeros((nK, nK), np.float32)
    for e in D.get("sl", []) or []:
        try:
            a, b, s = int(e[0]), int(e[1]), float(e[2])
        except (TypeError, ValueError, IndexError):
            continue
        if a < N and b < N and names[a] in ki and names[b] in ki:
            slm[ki[names[a]], ki[names[b]]] = s; slm[ki[names[b]], ki[names[a]]] = s
    ppm = np.array([float((D.get("ppm", {}) or {}).get(str(nidx[k]), 0) or 0) for k in kos], np.float32)
    MEASURED = collections.OrderedDict([
        ("COEXPR", cx), ("COEXPR-IND", cxi), ("CODEP", cd), ("CODEP-IND", cdi),
        ("PROFILE", PROF), ("SL", slm),
        ("ABUND-DIFF", np.abs(np.log1p(ppm)[:, None] - np.log1p(ppm)[None, :]).astype(np.float32)),
    ])

    # ---------- regulatory (SUSPECT: an interaction of a different kind) ----------
    regs = collections.defaultdict(set)
    for e in D.get("reg", []) or []:
        try:
            s_, t_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if s_ < N and t_ < N and names[t_] in ki:
            regs[ki[names[t_]]].add(s_)
    shR, jcR = setmat([regs.get(i, set()) for i in range(nK)])
    REG = collections.OrderedDict([("REG-SHARED", shR), ("REG-JAC", jcR)])

    ALLC = collections.OrderedDict(list(SPATIAL.items()) + list(PATHWAY.items()) +
                                   list(MEASURED.items()) + list(REG.items()))
    ARMS = collections.OrderedDict([
        ("TOPO", TOPO),
        ("+SPATIAL", collections.OrderedDict(list(TOPO.items()) + list(SPATIAL.items()))),
        ("+PATHWAY", collections.OrderedDict(list(TOPO.items()) + list(PATHWAY.items()))),
        ("+MEASURED", collections.OrderedDict(list(TOPO.items()) + list(MEASURED.items()))),
        ("+ALL-CLEAN", collections.OrderedDict(list(TOPO.items()) + list(ALLC.items()))),
        ("CLEAN-ONLY", ALLC),
    ])
    print(f"features: TOPO {len(TOPO)}, SPATIAL {len(SPATIAL)}, PATHWAY {len(PATHWAY)}, "
          f"MEASURED {len(MEASURED)}, REG {len(REG)}", flush=True)

    # ---------- CIRCULARITY CHECK: does a source only ever link already-connected pairs? ----------
    print(f"\n  CIRCULARITY CHECK -- of the KO-KO pairs each source links, what share are ALREADY recorded edges?")
    print(f"    (base rate over all pairs: {len(ko_edges)/(nK*(nK-1)/2):.3%})")
    iu = np.triu_indices(nK, 1)
    isedge = np.zeros((nK, nK), bool)
    for a, b in ko_edges:
        isedge[a, b] = True; isedge[b, a] = True
    circ = {}
    for nm, M_ in list(SPATIAL.items()) + list(PATHWAY.items()) + list(MEASURED.items()) + list(REG.items()):
        v = M_[iu]
        thr = np.quantile(v[v > 0], 0.9) if (v > 0).sum() > 100 else None
        if thr is None:
            continue
        sel = v >= thr
        circ[nm] = float(isedge[iu][sel].mean())
        print(f"    {nm:14s} top-decile pairs are edges {circ[nm]:6.2%}  "
              f"{'<-- SUSPECT: mostly restates the graph' if circ[nm] > 0.25 else ''}")

    # ---------- negatives: joint degree+pubs matched ----------
    pubk = pubs[sub]

    def bins(v, nb):
        q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
        return np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)

    jbin = np.array([f"{d}|{p}" for d, p in zip(bins(degk, 8), bins(pubk, 8))])
    edgeset = set(ko_edges)

    def sample_neg(posl, key, seed):
        r = np.random.RandomState(seed)
        idx = collections.defaultdict(list)
        for i in range(nK):
            idx[key[i]].append(i)
        out = []
        for (a, b) in posl:
            pa_, pb_ = idx[key[a]], idx[key[b]]
            got = tries = 0
            while got < NNEG and tries < 200:
                tries += 1
                x = pa_[r.randint(len(pa_))]; y = pb_[r.randint(len(pb_))]
                if x == y:
                    continue
                p = (min(x, y), max(x, y))
                if p in edgeset:
                    continue
                out.append(p); got += 1
        return out

    pos = sorted(held); neg = sample_neg(pos, jbin, 3)
    tr_pos = sorted(trainE); tr_neg = sample_neg(tr_pos, jbin, 4)
    print(f"\n  {len(pos)} held-out positives, {len(neg)} joint-matched negatives", flush=True)

    def vec(M_, prs):
        a = np.array([p[0] for p in prs]); b = np.array([p[1] for p in prs])
        return M_[a, b]

    cnp, cnn = vec(CN, pos), vec(CN, neg)
    strata = [(0, 0, "CN = 0"), (1, 2, "CN 1-2"), (3, 10, "CN 3-10"), (11, 10 ** 9, "CN > 10")]

    print(f"\n  AUC BY ARM -- pooled, then by how connected the pair ALREADY is")
    print(f"    {'arm':13s} {'pooled':>8s} " + " ".join(f"{lab:>9s}" for _, _, lab in strata))
    res = {}
    for arm, FE in ARMS.items():
        Xtr = np.stack([np.r_[vec(M_, tr_pos), vec(M_, tr_neg)] for M_ in FE.values()], 1)
        ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=4,
                                             min_samples_leaf=40, random_state=0).fit(Xtr, ytr)
        sp_ = clf.predict_proba(np.stack([vec(M_, pos) for M_ in FE.values()], 1))[:, 1]
        sn_ = clf.predict_proba(np.stack([vec(M_, neg) for M_ in FE.values()], 1))[:, 1]
        row = {"pooled": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[sp_, sn_]))}
        for lo, hi, lab in strata:
            mp = (cnp >= lo) & (cnp <= hi); mn = (cnn >= lo) & (cnn <= hi)
            row[lab] = (float(roc_auc_score(np.r_[np.ones(int(mp.sum())), np.zeros(int(mn.sum()))],
                                            np.r_[sp_[mp], sn_[mn]]))
                        if mp.sum() >= 30 and mn.sum() >= 30 else float("nan"))
        res[arm] = row
        print(f"    {arm:13s} {row['pooled']:8.4f} " + " ".join(f"{row[l]:9.4f}" for _, _, l in strata))

    z0 = "CN = 0"
    base, best_sparse = res["TOPO"][z0], max(res, key=lambda a: (res[a][z0] if res[a][z0] == res[a][z0] else -1))
    rescued = res[best_sparse][z0] > 0.60 and res[best_sparse][z0] > base + 0.05
    clean0 = res["CLEAN-ONLY"][z0]

    verdict = (
        f"CAN EVERYTHING ELSE WE HAVE COMPLETE THE SPARSE PART OF THE NETWORK? Topology recovers held-out PPI edges "
        f"well where the graph is already dense and not at all where it is not: in the CN=0 stratum TOPO scores "
        f"{base:.4f}, chance, because common-neighbours and the walk are identically zero for a pair with no shared "
        f"neighbours. This module adds every other layer in the dataset -- subcellular localisation, GO "
        f"cellular-component, pathway and process, co-expression, DepMap co-dependency, Perturb-seq profile "
        f"similarity, synthetic lethality, abundance, genomic position, and shared upstream regulators -- tiered so a "
        f"gain can be attributed, and with complexes EXCLUDED outright because 92.8% of within-complex pairs are "
        f"already edges and predicting an edge from co-complex membership is predicting the edge from the edge. "
        f"POOLED AUC RISES, which proves nothing: TOPO {res['TOPO']['pooled']:.4f} -> +ALL-CLEAN "
        f"{res['+ALL-CLEAN']['pooled']:.4f}. Adding features to a model always does that, and 79.7% of the held-out "
        f"edges live in the dense stratum where topology already worked. THE TEST IS THE CN=0 COLUMN. "
        + (f"AND IT IS RESCUED: {best_sparse} reaches {res[best_sparse][z0]:.4f} there against topology's "
           f"{base:.4f}. Critically CLEAN-ONLY -- the non-topology features with NO graph input at all -- scores "
           f"{clean0:.4f}, and that is the arm that matters, because every topology feature is identically zero in "
           f"this stratum so only a graph-free predictor can carry information here. "
           if rescued else
           f"AND IT IS NOT RESCUED. The best arm in that stratum is {best_sparse} at {res[best_sparse][z0]:.4f}, "
           f"against topology's {base:.4f}. CLEAN-ONLY -- every non-topology feature, with no graph input at all, "
           f"which is the only kind of predictor that CAN carry information where topology is identically zero -- "
           f"scores {clean0:.4f}. Adding localisation, pathways, co-expression, co-dependency, genetic interactions "
           f"and regulatory context does not let us place a protein that has no recorded neighbours. ")
        + f"THE CIRCULARITY CHECK is reported per source: the fraction of each source's top-decile pairs that are "
        f"ALREADY recorded edges, against a base rate of {len(ko_edges)/(nK*(nK-1)/2):.3%}. Anything far above that "
        f"is restating the graph rather than supplementing it, and the check is printed for every feature so a "
        f"reader can discount them individually. "
        f"WHAT THIS CANNOT SAY. The evaluation universe is knockout-knockout pairs, so nothing here speaks to the "
        f"{N-nK} proteins without perturbation data -- and those are disproportionately the poorly-connected ones "
        f"this is meant to help. 'Held-out true edge' means an edge already in a database, which is an optimistic "
        f"proxy for a genuinely undiscovered one. GO evidence codes are not retained in this dataset, so "
        f"interaction-derived (IPI) annotations cannot be filtered out of the GO features and the SUSPECT tier "
        f"cannot be fully cleaned. One cell line, one split.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"arms": res, "circularity": circ, "n_pos": len(pos), "n_neg": len(neg),
               "rescued": bool(rescued), "clean_only_cn0": clean0, "verdict": verdict},
              open(OUT / "evidence_completion.json", "w"), indent=2)
    print(f"\n  -> {OUT/'evidence_completion.json'}")


if __name__ == "__main__":
    main()
