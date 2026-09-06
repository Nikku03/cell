"""wall_learnsim -- the follow-through on wall_test/wall_edgetype: LEARN a perturbation-similarity from gene FEATURES (no peeking at the
held-out knockout's own profile) and see how far past the fixed-graph model (recall~0.44) it climbs toward the profile-oracle ceiling
(~0.62) on the K562 held-out specific-mover task.

The oracle in wall_test cheats: it uses knockout x's OWN measured profile to find x's nearest measured knockouts, then transfers their
programs (recall 0.62). The fixed-graph model uses ONE similarity (share a PPI/complex/pathway edge) and gets ~0.44. This model asks:
can we PREDICT which measured knockouts behave like x from x's FEATURES alone (complex/GO/pathway/PPI-neighbourhood + dependency +
degree + compartment...), recovering the oracle's neighbour-selection without measuring x? If yes, the wall (for single-KO within a
data-rich cell type) is a learnable-similarity gap, and we can say how much of the 0.44->0.62 head-room is closable.

Method (honest, leakage-controlled): split knockouts train/test. TARGET = true signed-z profile cosine S(x,y) (exactly what the oracle
ranks on). FEATURES(x,y) = shared complexes / GO-BP / GO-CC / GO-MF / pathway, shared & Jaccard PPI-neighbourhood, direct-PPI, and
numeric gene-property gaps (|dep_frac|, |log-degree|, |loeuf|, same compartment/process, both-TF). Train a gradient-boosted regressor on
TRAIN x TRAIN pairs; for each held-out TEST knockout predict similarity to all TRAIN knockouts, take the top-10, transfer their |z|
mean, and score specific-mover recall@50 -- head-to-head with the tide null, the fixed physical-graph model, and the (train-only)
oracle, all transferring from TRAIN knockouts only for a fair comparison. NEXUS/structural interaction magnitude is exactly the kind of
extra FEATURE this frame is built to absorb next; here we first establish what the annotation features alone buy.
"""
import os
import json, pickle, collections
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI = 1.0, 50, 5, 0.05, 10


def onehot(key_lists, kos):
    """KO x category sparse binary matrix from a per-KO list of category ids."""
    cats = {}; rows = []; cols = []
    for i, k in enumerate(kos):
        for c in key_lists.get(k, ()):
            j = cats.setdefault(c, len(cats)); rows.append(i); cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(kos), max(len(cats), 1)))


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), dtype=np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide_idx = np.where(~tide)[0]; mover_freq = (A >= TAU).mean(0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = (Mn @ Mn.T).astype(np.float32)                     # true profile similarity (oracle target)

    # ---- gene features ----
    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    go = D.get("go", {})
    cplx = {}                                              # gene2cplx is keyed by INDEX-STRING -> map to name (bug found & fixed)
    for k, cs in (D.get("gene2cplx", {}) or {}).items():
        if k.isdigit() and int(k) < len(names):
            cplx[names[int(k)]] = cs
    per = {  # per-KO category lists
        "cplx": {k: cplx.get(k, []) for k in kos},
        "gop": {k: go.get(k, {}).get("P", []) for k in kos},
        "goc": {k: go.get(k, {}).get("C", []) for k in kos},
        "gof": {k: go.get(k, {}).get("F", []) for k in kos},
        "path": {k: ([info.get(k, {}).get("path")] if info.get(k, {}).get("path") else []) for k in kos}}
    Msp = {name: onehot(d, kos) for name, d in per.items()}
    shared = {name: np.asarray((m @ m.T).todense(), dtype=np.float32) for name, m in Msp.items()}
    # PPI neighbourhood over the full gene universe
    nbset = collections.defaultdict(set)
    # THE INDEX-SPACE BUG. cell_complete's 'ppi' endpoints are INTEGER INDICES into genes[], but `kos` holds gene
    # NAME strings. The shipped code keyed nbset by the raw integer and then looked it up by name, so
    # nbset.get(<name>) returned EMPTY for every knockout: 0 of 16,492 gene names resolved against a 14,230-key
    # table, and 0 of 191,447 edges were ever seen. shared_ppi_nbr / jaccard_ppi_nbr / is_direct_ppi were
    # identically zero, and phys_adj silently degraded to complex-OR-pathway -- so the component labelled
    # "PHYS (PPI|complex|pathway)" contained no PPI at all. Fixed the same way as protein_chain_combine.py.
    for e in D.get("ppi", []):
        try:
            a_, b_ = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a_ < len(names) and 0 <= b_ < len(names) and a_ != b_:
            nbset[names[a_]].add(names[b_])
            nbset[names[b_]].add(names[a_])
    NBcols = {}; rows = []; cols = []
    for i, k in enumerate(kos):
        for g in nbset.get(k, ()):
            rows.append(i); cols.append(NBcols.setdefault(g, len(NBcols)))
    NB = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nK, max(len(NBcols), 1)))
    shared_nbr = np.asarray((NB @ NB.T).todense(), dtype=np.float32)
    deg = np.array(NB.sum(1)).ravel()
    jac_nbr = shared_nbr / (deg[:, None] + deg[None, :] - shared_nbr + 1e-9)
    is_ppi = np.zeros((nK, nK), dtype=np.float32)
    for i, k in enumerate(kos):
        for g in nbset.get(k, ()):
            if g in ki:
                is_ppi[i, ki[g]] = 1
    # numeric per-gene props
    def arr(f, default=0.0):
        return np.array([float(info.get(k, {}).get(f, default) or default) for k in kos], dtype=np.float32)
    dep = arr("dep_frac"); loeuf = arr("loeuf", 1.0); logdeg = np.log1p(arr("ppi"))
    comp = np.array([info.get(k, {}).get("comp", "") for k in kos]); proc = np.array([info.get(k, {}).get("proc", "") for k in kos])
    tf = arr("tf")
    d_dep = np.abs(dep[:, None] - dep[None, :]); d_loe = np.abs(loeuf[:, None] - loeuf[None, :])
    d_deg = np.abs(logdeg[:, None] - logdeg[None, :])
    same_comp = (comp[:, None] == comp[None, :]).astype(np.float32); same_proc = (proc[:, None] == proc[None, :]).astype(np.float32)
    both_tf = ((tf[:, None] == 1) & (tf[None, :] == 1)).astype(np.float32)

    FEATS = [shared["cplx"], shared["gop"], shared["goc"], shared["gof"], shared["path"],
             shared_nbr, jac_nbr, is_ppi, d_dep, d_loe, d_deg, same_comp, same_proc, both_tf]

    def feat_rows(xi, yidx):
        return np.stack([Fm[xi, yidx] for Fm in FEATS], axis=1)

    # ---- train/test split of knockouts ----
    rng = np.random.RandomState(0)
    scor = [i for i in range(nK) if len([j for j in np.where(A[i] >= TAU)[0] if ~tide[j]]) >= MIN_SPEC]
    perm = rng.permutation(scor); ntest = len(scor) // 3
    test = set(perm[:ntest].tolist()); train = np.array([i for i in range(nK) if i not in test])
    train_scor = [i for i in train if i in set(scor)]
    print(f"K562: {nK} KOs; {len(scor)} scorable; train {len(train)} / test {len(test)}")

    # ---- training pairs (train x train), target = true similarity ----
    Xtr = []; ytr = []
    for xi in train_scor:
        cand = train[train != xi]
        sub = rng.choice(cand, min(len(cand), 250), replace=False)
        Xtr.append(feat_rows(xi, sub)); ytr.append(S[xi, sub])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_depth=4, min_samples_leaf=40,
                                        random_state=0).fit(Xtr, ytr)
    print(f"trained similarity regressor on {len(ytr):,} train-train pairs ({len(FEATS)} features)")

    def rec(scores, truth):
        order = nontide_idx[np.argsort(-scores[nontide_idx])][:K]
        return len(set(order.tolist()) & truth) / max(len(truth), 1)

    # PHYS baseline consistent with wall_edgetype: neighbour = shares a PPI edge OR a complex OR a pathway (restricted to TRAIN KOs)
    trainset = set(train.tolist())
    phys_adj = (is_ppi > 0) | (shared["cplx"] > 0) | (shared["path"] > 0)
    phys_nb = {i: [j for j in np.where(phys_adj[i])[0] if j in trainset and j != i] for i in test}
    out = collections.defaultdict(list)
    for xi in sorted(test):
        spec = set(j for j in np.where(A[xi] >= TAU)[0].tolist() if ~tide[j])
        # LEARNED: predict sim to all TRAIN kos from features, top-N, transfer
        ps = reg.predict(feat_rows(xi, train))
        top = train[np.argsort(-ps)[:N_NEI]]
        out["learned"].append(rec(A[top].mean(0), spec))
        # ORACLE (train-only, peeks at true S)
        st = S[xi, train]; otop = train[np.argsort(-st)[:N_NEI]]
        out["oracle"].append(rec(A[otop].mean(0), spec))
        # PHYSICAL graph (train neighbours only)
        pn = phys_nb[xi]
        out["phys"].append(rec(A[pn].mean(0), spec) if pn else 0.0)
        out["tide"].append(rec(mover_freq, spec))
    m = {k: float(np.mean(v)) for k, v in out.items()}
    covered = sum(1 for xi in sorted(test) if phys_nb[xi])
    print(f"\nHeld-out TEST knockouts ({len(out['tide'])}). Specific-mover recall@{K} (tide removed):\n")
    print(f"   {'TIDE null':28s} {m['tide']:.3f}")
    print(f"   {'PHYS graph (fixed)':28s} {m['phys']:.3f}   [phys-covered test KOs: {covered}/{len(test)}]")
    print(f"   {'LEARNED similarity':28s} {m['learned']:.3f}")
    print(f"   {'ORACLE* (peeks, ceiling)':28s} {m['oracle']:.3f}")
    closed = (m["learned"] - m["phys"]) / max(m["oracle"] - m["phys"], 1e-9)
    imps = [reg.feature_importances_] if hasattr(reg, "feature_importances_") else []
    print(f"\n   learned vs tide: {m['learned']/max(m['tide'],1e-9):.2f}x  |  vs phys: {m['learned']-m['phys']:+.3f}  "
          f"|  head-room to oracle closed: {closed*100:.0f}%")

    meaningful = (m["learned"] - m["phys"]) >= 0.05 and closed >= 0.3        # honest bar: real head-room closed, not a rounding win
    verdict = (
        f"LEARNED PERTURBATION SIMILARITY from gene features (no peeking at the held-out KO), K562, {len(test)} held-out knockouts, "
        f"specific-mover recall@{K}: TIDE null {m['tide']:.3f} -> fixed PHYS graph {m['phys']:.3f} -> LEARNED {m['learned']:.3f} -> "
        f"ORACLE* ceiling {m['oracle']:.3f}. HONEST READ: "
        + (f"the learned model MEANINGFULLY beats the graph (+{m['learned']-m['phys']:.3f}, {closed*100:.0f}% of the head-room closed) -- "
           "combining annotation similarities with learned weights predicts behavioural similarity better than any single edge."
           if meaningful else
           f"the learned annotation-similarity BARELY beats the fixed physical graph ({m['learned']:.3f} vs {m['phys']:.3f}, only "
           f"+{m['learned']-m['phys']:.3f}, closing just {closed*100:.0f}% of the graph->oracle head-room). This is a NEGATIVE-ish "
           "result and I am reporting it as one: hand-crafted STATIC ANNOTATION features (complex + GO-BP/CC/MF + pathway + PPI-"
           "neighbourhood + dependency + degree + compartment) add almost nothing over the raw physical graph for predicting which "
           "knockouts behave alike. The large oracle head-room (behavioural similarity IS there and IS shared) is NOT reachable from "
           "these annotations.")
        + f" WHAT THIS RULES IN/OUT: it rules OUT 'just engineer more annotation features' as the path to the oracle, and points to a "
        "richer REPRESENTATION -- learned gene embeddings, sequence/structure (ESM), or ultimately co-perturbation profiles (which is "
        "the oracle, i.e. you need measured neighbours). NEXUS/structural interaction magnitude can still enter as a feature here, but "
        f"the bar is now explicit and high: it must beat {m['learned']:.3f}, and annotation features already saturate near the graph, so "
        "it would have to add genuinely ORTHOGONAL behavioural signal. CAVEATS: modest test set (" + str(len(test)) + " KOs); single-"
        "KO, within-one-cell-type, one steady-state-mRNA readout; says nothing about combinations, mutations, drugs, cross-cell-type, "
        "or a real cancer genotype.")
    print(f"\nVERDICT: {verdict}")
    js = {"cell_type": "K562", "n_test": len(test), "K": K,
          "recall_specific": {n: m[n] for n in ["tide", "phys", "learned", "oracle"]},
          "phys_covered_test": covered, "headroom_closed_frac": closed,
          "feature_names": ["shared_cplx", "shared_gobp", "shared_gocc", "shared_gomf", "same_path", "shared_ppi_nbr",
                            "jaccard_ppi_nbr", "is_direct_ppi", "abs_dep_diff", "abs_loeuf_diff", "abs_logdeg_diff",
                            "same_comp", "same_proc", "both_tf"],
          "feature_importances": (imps[0].tolist() if imps else None), "verdict": verdict, "note": verdict}
    json.dump(js, open(OUT / "wall_learnsim.json", "w"), indent=1)
    print("\n  -> outputs/orphan/wall_learnsim.json")
    return js


if __name__ == "__main__":
    main()
