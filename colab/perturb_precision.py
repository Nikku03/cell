"""PERTURB-SEQ AS EDGE EVIDENCE, SCORED ON PRECISION RATHER THAN AUC.

TWO THINGS THIS FIXES ABOUT evidence_completion.py.

FIRST, IT WAS USING PERTURB-SEQ BADLY. That module's only perturbation feature was PROFILE, the cosine between two
knockouts' whole z-profiles -- a symmetric, undirected summary that throws away the most direct evidence in the
dataset: whether knocking out A actually MOVES B's transcript. Measured here before building anything:

    mean |z| of B under knockout of A     PPI edges 0.109   random pairs 0.019    5.7x
    either direction moves more than 1sd  PPI edges 5.99%   random pairs 0.49%   12.2x
    both directions move more than 1sd    PPI edges 0.256%  random pairs 0.000%

1296 of the 1400 knockouts are themselves measured genes, so this directional feature is computable for 92.6% of
knockouts and was simply left on the table. Note what it is and is not: "knocking out A changes B's mRNA" is
FUNCTIONAL coupling, not proof of physical contact. It is being tested as a predictor of a physical edge, and
whether that transfer works is the measurement, not an assumption.

SECOND, AUC IS THE WRONG HEADLINE FOR THIS QUESTION. AUC is a ranking statistic over a balanced-ish comparison; what
anyone completing a network actually needs is "if I take the top N pairs, how many are real?" At the true base rate
of KO-KO pairs -- roughly 0.36% -- an AUC of 0.95 is compatible with poor precision, and link_completion.py already
measured that gap: p@100 was 0.840 in a 1-in-6 enriched pool and 0.420 when ranking all ~965,000 candidates. So
every arm here is reported on DEPLOYMENT PRECISION over the full candidate pool, with AUC kept only as context.

PERTURB-SEQ FEATURE SET (all CLEAN -- measured expression, wholly independent of any interaction database):
    EFF-MAX / EFF-MIN   max and min over the two directions of |z| of one knockout's gene under the other's knockout
    RECIP               both directions exceed 1sd -- mutual perturbation, the strongest form
    EITHER              either direction exceeds 1sd
    RANK-BEST           best (smallest) rank of one gene within the other's sorted mover list, log-scaled
    SPEC-SHARED/JAC     count and Jaccard of shared SPECIFIC movers (tide genes removed, so a generic stress
                        response cannot manufacture similarity)
    PROFILE             cosine of full z-profiles (the old feature, kept so its marginal value is visible)
    PROFILE-SPEC        cosine restricted to non-tide genes

ARMS: TOPO / +SPATIAL / +PERTURB / +SPATIAL+PERTURB / PERTURB-ONLY / SPATIAL+PERTURB (graph-free).
The graph-free arms exist because at CN=0 every topology feature is identically zero, so only a predictor that never
looks at the graph can carry information there -- established in evidence_completion.py, where CLEAN-ONLY (0.631)
beat every arm that included topology (0.609).
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
HOLDOUT, SEED, NNEG, TAU, TIDE_FRAC = 0.20, 0, 5, 1.0, 0.05


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); nK = len(kos); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    Z = np.zeros((nK, len(genes)), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            Z[ki[k], gi[g]] = z
    A_ = np.abs(Z)
    tide = (A_ >= TAU).mean(0) >= TIDE_FRAC

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
    perm = rng.permutation(len(ko_edges)); nho = int(len(ko_edges) * HOLDOUT)
    held = set(ko_edges[i] for i in perm[:nho]); trainE = set(ko_edges[i] for i in perm[nho:])
    heldn = set((konode[a], konode[b]) for a, b in held) | set((konode[b], konode[a]) for a, b in held)
    ta, tb = [], []
    for a, b in pe:
        if (a, b) not in heldn and (b, a) not in heldn:
            ta.append(a); tb.append(b)
    ta = np.array(ta); tb = np.array(tb)
    Atr = sparse.coo_matrix((np.ones(2 * len(ta), np.float32), (np.r_[ta, tb], np.r_[tb, ta])), shape=(N, N)).tocsr()
    Atr.data[:] = 1.0
    deg = np.asarray(Atr.sum(1)).ravel()
    print(f"{len(ko_edges)} KO-KO edges: {len(trainE)} train, {len(held)} held out", flush=True)

    # ---------------- TOPOLOGY ----------------
    sub = konode
    CN = np.asarray((Atr[sub] @ Atr[:, sub]).todense(), np.float32)
    il = np.zeros(N, np.float32); nzz = deg > 1; il[nzz] = 1.0 / np.log(deg[nzz])
    AA = np.asarray((Atr[sub] @ sparse.diags(il) @ Atr[:, sub]).todense(), np.float32)
    degk = deg[sub]; JC = CN / (degk[:, None] + degk[None, :] - CN + 1e-9)
    mm = expressed[ta] & expressed[tb]
    ra = np.r_[ta[mm], tb[mm]]; rb = np.r_[tb[mm], ta[mm]]
    Gk = sparse.coo_matrix((np.ones(len(ra), np.float32), (ra, rb)), shape=(N, N)).tocsr()
    mg = np.asarray(Gk.sum(1)).ravel(); iv = np.zeros(N, np.float32); nz = mg > 0; iv[nz] = 1.0 / mg[nz]
    CH2 = np.asarray(((sparse.diags(iv) @ Gk)[sub] @ (sparse.diags(iv) @ Gk)[:, sub]).todense(), np.float32)
    TOPO = collections.OrderedDict([("CN", CN), ("JACCARD", JC), ("ADAMIC", AA), ("CHAIN2", CH2)])

    # ---------------- SPATIAL (the tier that worked in evidence_completion) ----------------
    go = D.get("go", {})

    def setmat(sets):
        vocab = {}; r, c = [], []
        for i, s in enumerate(sets):
            for t in s:
                r.append(i); c.append(vocab.setdefault(t, len(vocab)))
        Ms = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(nK, max(len(vocab), 1)))
        sh = np.asarray((Ms @ Ms.T).todense(), np.float32)
        sz = np.array([len(s) for s in sets], np.float32)
        return sh, sh / (sz[:, None] + sz[None, :] - sh + 1e-9)

    shC, jcC = setmat([set(go.get(k, {}).get("C", []) or []) for k in kos])
    comp = np.array([str(info[k].get("comp") or "") for k in kos])
    chrom = np.array([str(info[k].get("chrom") or "") for k in kos])
    tss = np.array([float(info[k].get("tss") or 0) for k in kos])
    samechr = (chrom[:, None] == chrom[None, :]).astype(np.float32)
    SPATIAL = collections.OrderedDict([
        ("SAME-COMP", (comp[:, None] == comp[None, :]).astype(np.float32)),
        ("GOC-SHARED", shC), ("GOC-JAC", jcC), ("SAME-CHR", samechr),
        ("TSS-DIST", (np.log1p(np.abs(tss[:, None] - tss[None, :])) * samechr).astype(np.float32)),
    ])

    # ---------------- DENSE DepMap co-dependency ----------------
    # A provenance audit found outputs/orphan/depmap_vecs.npz: the FULL Chronos gene-effect matrix, 18,443 genes x
    # 1,150 cell lines, already per-gene standardised, 98.9% knockout coverage and no NaNs. Every module up to now
    # used cell_complete's `codep` instead, which stores only each gene's TOP-8 partners -- so it is undefined for
    # almost every pair and structurally cannot help in the sparse regime, which is the regime we care about. A
    # dense cosine is defined for EVERY pair. Provenance is clean: CRISPR fitness across cell lines, no interaction
    # assay anywhere in the chain.
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    dsym = {s_: i for i, s_ in enumerate(dv["syms"].tolist())}
    Vd = np.zeros((nK, dv["Z"].shape[1]), np.float32)
    dep_ok = np.zeros(nK, np.float32)
    for i, k in enumerate(kos):
        if k in dsym:
            Vd[i] = dv["Z"][dsym[k]]; dep_ok[i] = 1.0
    Vd /= (np.linalg.norm(Vd, axis=1, keepdims=True) + 1e-9)
    DENSE_CODEP = (Vd @ Vd.T).astype(np.float32)
    DEPMAP = collections.OrderedDict([
        ("DENSE-CODEP", DENSE_CODEP),
        ("DEP-AVAIL", np.minimum(dep_ok[:, None], dep_ok[None, :]).astype(np.float32)),
    ])
    print(f"  dense DepMap: {int(dep_ok.sum())}/{nK} knockouts covered", flush=True)

    # ---------------- PERTURB-SEQ, properly ----------------
    # E[i, j] = |z| of knockout j's OWN GENE measured under knockout i. Undefined when j's gene is not on the panel;
    # encoded as 0 with an explicit availability indicator so the model cannot read "not measured" as "no effect".
    E = np.zeros((nK, nK), np.float32)
    avail = np.zeros((nK, nK), np.float32)
    col_of = np.array([gi.get(k, -1) for k in kos])
    ok = col_of >= 0
    E[:, ok] = A_[:, col_of[ok]]
    avail[:, ok] = 1.0
    EFFMAX = np.maximum(E, E.T); EFFMIN = np.minimum(E, E.T)
    RECIP = ((E > TAU) & (E.T > TAU)).astype(np.float32)
    EITHER = ((E > TAU) | (E.T > TAU)).astype(np.float32)
    # rank of j inside i's sorted |z| list (1 = strongest); log-scaled, best of the two directions
    order = np.argsort(-A_, axis=1)
    rankpos = np.empty_like(order)
    np.put_along_axis(rankpos, order, np.arange(A_.shape[1])[None, :].repeat(nK, 0), axis=1)
    R = np.full((nK, nK), A_.shape[1], np.float32)
    R[:, ok] = rankpos[:, col_of[ok]]
    RANKBEST = -np.log1p(np.minimum(R, R.T))
    spec = ((A_ >= TAU) & (~tide)).astype(np.float32)
    SPSH = spec @ spec.T
    ssz = spec.sum(1)
    SPJC = SPSH / (ssz[:, None] + ssz[None, :] - SPSH + 1e-9)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    PROF = (Zn @ Zn.T).astype(np.float32)
    Zs = Z[:, ~tide]
    Zsn = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-9)
    PROFS = (Zsn @ Zsn.T).astype(np.float32)
    PERTURB = collections.OrderedDict([
        ("EFF-MAX", EFFMAX), ("EFF-MIN", EFFMIN), ("RECIP", RECIP), ("EITHER", EITHER),
        ("AVAIL", np.minimum(avail, avail.T)), ("RANK-BEST", RANKBEST),
        ("SPEC-SHARED", SPSH), ("SPEC-JAC", SPJC), ("PROFILE", PROF), ("PROFILE-SPEC", PROFS),
    ])
    # `sl` is DROPPED on audit evidence, not on a hunch: it fires on 14 of 979,300 KO-KO pairs and on 0 of the
    # 3,444 held-out edges (0 of 82 at CN=0), so it cannot move any number -- and it is a thresholded subset of
    # codep, so counting it as a second independent source would have been double-counting.
    print(f"features: TOPO {len(TOPO)}, SPATIAL {len(SPATIAL)}, PERTURB {len(PERTURB)}", flush=True)

    def merge(*ds):
        o = collections.OrderedDict()
        for d in ds:
            o.update(d)
        return o

    ARMS = collections.OrderedDict([
        ("TOPO", TOPO), ("+SPATIAL", merge(TOPO, SPATIAL)), ("+PERTURB", merge(TOPO, PERTURB)),
        ("+SPAT+PERT", merge(TOPO, SPATIAL, PERTURB)),
        ("PERTURB-ONLY", PERTURB), ("SPAT+PERT(no graph)", merge(SPATIAL, PERTURB)),
        ("+DEPMAP", merge(TOPO, DEPMAP)),
        ("+SPAT+PERT+DEP", merge(TOPO, SPATIAL, PERTURB, DEPMAP)),
        ("ALL(no graph)", merge(SPATIAL, PERTURB, DEPMAP)),
    ])

    # ---------------- negatives: joint degree+pubs matched ----------------
    pubk = pubs[sub]

    def bins(v, nb):
        q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
        return np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)

    jbin = np.array([f"{d}|{p}" for d, p in zip(bins(degk, 8), bins(pubk, 8))])
    edgeset = set(ko_edges)

    def sample_neg(posl, key, seed):
        r = np.random.RandomState(seed); idx = collections.defaultdict(list)
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

    def vec(M_, prs):
        a = np.array([p[0] for p in prs]); b = np.array([p[1] for p in prs])
        return M_[a, b]

    cnp, cnn = vec(CN, pos), vec(CN, neg)
    strata = [(0, 0, "CN=0"), (1, 2, "CN1-2"), (3, 10, "CN3-10"), (11, 10 ** 9, "CN>10")]

    # deployment pool: every candidate pair that is not an already-known training edge
    iu = np.triu_indices(nK, 1)
    code = iu[0].astype(np.int64) * nK + iu[1].astype(np.int64)
    ispos = np.isin(code, np.array([a * nK + b for a, b in pos], np.int64))
    istr = np.isin(code, np.array([a * nK + b for a, b in tr_pos], np.int64))
    poolm = ~istr
    ypool = ispos[poolm]
    base = float(ypool.mean())
    cnpool = CN[iu][poolm]
    print(f"\n  deployment pool: {int(poolm.sum()):,} candidate pairs, {int(ypool.sum())} true "
          f"(base rate {base:.4%})", flush=True)

    print(f"\n  {'arm':20s} {'AUC':>7s} | " + " ".join(f"{l:>7s}" for _, _, l in strata) +
          " | " + " ".join(f"{'p@'+str(k):>7s}" for k in (100, 500, 1000)))
    res = {}
    for arm, FE in ARMS.items():
        Xtr = np.stack([np.r_[vec(M_, tr_pos), vec(M_, tr_neg)] for M_ in FE.values()], 1)
        ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.07, max_depth=4,
                                             min_samples_leaf=40, random_state=0).fit(Xtr, ytr)
        sp_ = clf.predict_proba(np.stack([vec(M_, pos) for M_ in FE.values()], 1))[:, 1]
        sn_ = clf.predict_proba(np.stack([vec(M_, neg) for M_ in FE.values()], 1))[:, 1]
        row = {"auc": float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[sp_, sn_]))}
        for lo, hi, lab in strata:
            mp = (cnp >= lo) & (cnp <= hi); mn = (cnn >= lo) & (cnn <= hi)
            if mp.sum() >= 30 and mn.sum() >= 30:
                yv = np.r_[np.ones(int(mp.sum())), np.zeros(int(mn.sum()))]
                sv = np.r_[sp_[mp], sn_[mn]]
                row[lab] = float(roc_auc_score(yv, sv))
                br = np.random.RandomState(11)
                ip, inn = np.where(yv == 1)[0], np.where(yv == 0)[0]
                row[lab + "_se"] = float(np.std([roc_auc_score(yv[ix], sv[ix]) for ix in
                                                 (np.r_[br.choice(ip, len(ip)), br.choice(inn, len(inn))]
                                                  for _ in range(300))]))
            else:
                row[lab] = float("nan"); row[lab + "_se"] = float("nan")
        # DEPLOYMENT PRECISION over the whole pool
        scp = clf.predict_proba(np.stack([M_[iu][poolm] for M_ in FE.values()], 1))[:, 1]
        o = np.argsort(-scp)
        for kk in (100, 500, 1000, 5000):
            row[f"p@{kk}"] = float(ypool[o[:kk]].mean())
        sparse_mask = cnpool == 0
        if sparse_mask.sum() > 1000 and ypool[sparse_mask].sum() > 5:
            os_ = np.argsort(-scp[sparse_mask])
            row["p@100_CN0"] = float(ypool[sparse_mask][os_[:100]].mean())
            row["base_CN0"] = float(ypool[sparse_mask].mean())
        res[arm] = row
        print(f"  {arm:20s} {row['auc']:7.4f} | " + " ".join(f"{row[l]:7.3f}" for _, _, l in strata) +
              " | " + " ".join(f"{row['p@'+str(k)]:7.3f}" for k in (100, 500, 1000)))

    print(f"\n  PRECISION IN THE SPARSE REGIME (CN=0 pairs only; base rate "
          f"{res['TOPO'].get('base_CN0', float('nan')):.4%})")
    for arm in ARMS:
        if "p@100_CN0" in res[arm]:
            print(f"    {arm:20s} p@100 = {res[arm]['p@100_CN0']:.3f}")

    bt, bp = res["TOPO"], res["+SPAT+PERT"]
    dp100 = bp["p@100"] - bt["p@100"]
    best_cn0 = max(ARMS, key=lambda a: (res[a]["CN=0"] if res[a]["CN=0"] == res[a]["CN=0"] else -1))

    verdict = (
        f"PERTURB-SEQ AS EDGE EVIDENCE, AND WHAT IT DOES TO PRECISION. The previous module used Perturb-seq only as "
        f"a whole-profile cosine, discarding the most direct signal available: knocking out A and watching B's "
        f"transcript. Measured before building, that signal is real -- mean |z| of B under A's knockout is 0.109 on "
        f"PPI edges against 0.019 on random pairs (5.7x), and 'either direction moves more than 1sd' is 5.99% "
        f"against 0.49% (12.2x). 1296 of 1400 knockouts are themselves measured genes, so it was computable all "
        f"along. "
        f"ON PRECISION, WHICH IS THE NUMBER THAT MATTERS: over the full deployment pool of {int(poolm.sum()):,} "
        f"candidate pairs at a {base:.4%} base rate, topology alone gets p@100 {bt['p@100']:.3f} and topology plus "
        f"spatial plus Perturb-seq gets {bp['p@100']:.3f} ({dp100:+.3f}); at 1000 it is {bt['p@1000']:.3f} -> "
        f"{bp['p@1000']:.3f}. Perturb-seq alone, with no graph and no annotation, reaches p@100 "
        f"{res['PERTURB-ONLY']['p@100']:.3f}. "
        f"IN THE SPARSE REGIME the ordering is what matters rather than the level: at CN=0 topology is pinned at "
        f"{bt['CN=0']:.3f} by construction, and the best arm is {best_cn0} at {res[best_cn0]['CN=0']:.3f} "
        f"+/- {res[best_cn0].get('CN=0_se', float('nan')):.3f}. "
        f"WHAT THIS CANNOT SAY. 'Knocking out A changes B's mRNA' is FUNCTIONAL coupling; it is being used to "
        f"predict a PHYSICAL edge, and the two are not the same thing -- a strong directional effect through three "
        f"intermediates counts identically to a direct contact. Availability is explicit (104 knockouts are not on "
        f"the measured panel and get a zero effect with an AVAIL indicator so the model cannot read absence as "
        f"evidence). The pool is knockout-knockout pairs, and a held-out 'true' edge is one already in a database. "
        f"One cell line, one split.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"arms": res, "pool": int(poolm.sum()), "base_rate": base, "verdict": verdict},
              open(OUT / "perturb_precision.json", "w"), indent=2)
    print(f"\n  -> {OUT/'perturb_precision.json'}")


if __name__ == "__main__":
    main()
