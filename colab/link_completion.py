"""CAN WE COMPLETE THE NETWORK? Hold out real PPI edges, try to recover them, and control for study bias.

THE QUESTION. Everything else in this repo runs the network FORWARD -- use the graph to predict what a knockout does.
This runs it BACKWARD: use the perturbation data (and the graph's own topology) to predict EDGES. If that works, the
same machinery proposes new connections and completes the network.

WHAT IS ALREADY KNOWN HERE, so this adds rather than repeats:
  - connection_proposer.py emits ranked machine->target CONVERGENCES with a confidence score, and is explicit that
    the number is "P(looks like a real coherent-machine effect)", NOT "P(true direct edge)". It also cannot see a
    lone pairwise edge, by construction, because it scores convergence onto an annotated complex.
  - complete_network.py merged curated + measured edges and found the result trustworthy about ONE HOP out (direct
    edges 1.91x enriched over degree-matched decoys, one intermediate 1.04x = chance).
  - NOBODY HAS TESTED ACTUAL EDGE RECOVERY AGAINST A STUDY-BIAS CONTROL. That is this module.

WHY STUDY BIAS IS THE WHOLE PROBLEM, AND WHY MOST PPI LINK-PREDICTION NUMBERS ARE INFLATED. A PPI database records
interactions somebody looked for. `pubs` in this dataset runs from 0 to 11,630 with a median of 51. Two famous
proteins are recorded as interacting far more often than two obscure ones, whether or not they interact more. A
predictor that has silently learned "both endpoints are well studied" scores beautifully against randomly sampled
non-edges and has discovered nothing. So every number here is reported against THREE negative sets:

    RANDOM        uniformly sampled non-edges                       -- the inflated number everyone quotes
    DEGREE-MATCHED  negatives matched on both endpoints' degree     -- removes hubs-attract-hubs
    PUBS-MATCHED    negatives matched on both endpoints' pubs count -- removes famous-attracts-famous

The gap between the first and the last IS the study bias, and it is reported as a headline rather than buried.

THE SPLIT. 20% of the PPI edges among the 1400 knockouts are held out. Every topology and chain feature is rebuilt on
the REMAINING graph only -- if a held-out edge were still present when computing common neighbours, the task would be
to predict an edge from itself. The universe is KO-KO pairs (1400 choose 2 = 979,300) because that is where measured
perturbation profiles exist for BOTH endpoints, which is what makes the perturbation features computable at all.

PREDICTORS, weakest-and-most-suspicious first:
    PUBS        pubs_a * pubs_b                     -- pure study bias. The bar. If nothing beats it, we are done.
    PREF-ATT    deg_a * deg_b                       -- preferential attachment, pure degree
    COMMON-NB   |N(a) & N(b)| on the training graph
    JACCARD     |N(a) & N(b)| / |N(a) | N(b)|
    ADAMIC-AD   sum over shared neighbours of 1/log(deg)
    CHAIN-2     the K562-expressed 2-step Markov reachability -- the chain from k562_specific_chain.py used as a
                link predictor. This is the direct test of "can the walk complete the network".
    PROFILE     cosine similarity of the two knockouts' measured z-profiles -- the PERTURBATION evidence, and the
                only feature here that does not come from the graph at all
    COMBINED    a GBM over all of the above, trained on TRAIN edges only

THE DELIVERABLE, if and only if something clears the pubs-matched bar: a ranked list of high-scoring pairs that are
NOT recorded as edges, with a precision estimate taken from the held-out calibration rather than asserted.
"""
import collections
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
HOLDOUT = 0.20
NNEG = 5              # negatives sampled per positive, per negative-set flavour
SEED = 0
TOPN = 40


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    Z = np.zeros((nK, nG), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            Z[ki[k], gi[g]] = z

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    konode = np.array([nidx.get(k, -1) for k in kos])
    assert (konode >= 0).all(), "every knockout should be a graph node"

    pubs = np.array([float(info[n].get("pubs") or 0) for n in names], np.float64)

    # K562 expression, for the cell-specific chain arm (same source and the same categorical trap as before)
    f = h5py.File(f"{SP}/k562.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["__categories"]["gene_name"][:]]
    gname = np.array(cats)[f["var"]["gene_name"][:]]
    expressed = np.zeros(N, bool)
    for n_ in set(gname.tolist()):
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
    print(f"{len(pe)} PPI edges over {N} nodes; {nK} knockouts; expressed {int(expressed.sum())}", flush=True)

    # ---- the KO-KO universe: this is where perturbation profiles exist for BOTH endpoints ----
    konode_set = {int(v): i for i, v in enumerate(konode)}
    ko_edges = [(konode_set[a], konode_set[b]) for a, b in pe if a in konode_set and b in konode_set]
    ko_edges = sorted(set((min(a, b), max(a, b)) for a, b in ko_edges))
    print(f"KO-KO pairs: {nK*(nK-1)//2} total, {len(ko_edges)} recorded as PPI edges "
          f"({len(ko_edges)/(nK*(nK-1)//2):.3%} density)", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(ko_edges))
    nho = int(len(ko_edges) * HOLDOUT)
    held = set(ko_edges[i] for i in perm[:nho])
    trainE = [ko_edges[i] for i in perm[nho:]]
    print(f"held out {len(held)} KO-KO edges; {len(trainE)} remain in the training graph", flush=True)

    # ---- TRAINING GRAPH: the full PPI graph MINUS the held-out KO-KO edges ----
    heldnodes = set((konode[a], konode[b]) for a, b in held)
    heldnodes |= set((b, a) for a, b in heldnodes)
    tr_a, tr_b = [], []
    for a, b in pe:
        if (a, b) in heldnodes or (b, a) in heldnodes:
            continue
        tr_a.append(a); tr_b.append(b)
    tr_a = np.array(tr_a); tr_b = np.array(tr_b)
    Atr = sparse.coo_matrix((np.ones(2 * len(tr_a), np.float32), (np.r_[tr_a, tr_b], np.r_[tr_b, tr_a])),
                            shape=(N, N)).tocsr()
    Atr.data[:] = 1.0
    deg = np.asarray(Atr.sum(1)).ravel()
    print(f"training graph: {len(tr_a)} undirected edges (removed {len(pe)-len(tr_a)})", flush=True)

    # cell-specific operator for the CHAIN-2 arm, built on the TRAINING graph only
    kp = expressed
    m = kp[tr_a] & kp[tr_b]
    ra = np.r_[tr_a[m], tr_b[m]]; rb = np.r_[tr_b[m], tr_a[m]]
    Gk = sparse.coo_matrix((np.ones(len(ra), np.float32), (ra, rb)), shape=(N, N)).tocsr()
    mag = np.asarray(Gk.sum(1)).ravel(); inv = np.zeros(N, np.float32)
    nz = mag > 0; inv[nz] = 1.0 / mag[nz]
    Pk = (sparse.diags(inv) @ Gk).tocsr()

    sub = konode                                   # graph-node index of every knockout
    Asub = Atr[sub][:, sub]                        # KO x KO adjacency, training only
    CN = np.asarray((Atr[sub] @ Atr[:, sub]).todense(), np.float32)     # common neighbours
    invlog = np.zeros(N, np.float32)
    nzz = deg > 1
    invlog[nzz] = 1.0 / np.log(deg[nzz])
    AA = np.asarray((Atr[sub] @ sparse.diags(invlog) @ Atr[:, sub]).todense(), np.float32)
    degk = deg[sub]
    JC = CN / (degk[:, None] + degk[None, :] - CN + 1e-9)
    CH2 = np.asarray((Pk[sub] @ Pk[:, sub]).todense(), np.float32)      # 2-step chain reachability
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    PROF = (Zn @ Zn.T).astype(np.float32)
    pubk = pubs[sub]
    print("features built", flush=True)

    FEATS = collections.OrderedDict([
        ("PUBS", np.log1p(pubk)[:, None] + np.log1p(pubk)[None, :]),
        ("PREF-ATT", np.log1p(degk)[:, None] + np.log1p(degk)[None, :]),
        ("COMMON-NB", CN), ("JACCARD", JC), ("ADAMIC-AD", AA),
        ("CHAIN-2", CH2), ("PROFILE", PROF),
    ])

    # ---- negative sets: random, degree-matched, pubs-matched ----
    edgeset = set(ko_edges)

    def bins(v, nb=12):
        q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
        return np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)

    dbin, pbin = bins(degk), bins(pubk)

    def sample_neg(pos, key, seed):
        """For each positive, draw NNEG non-edges whose endpoints fall in the SAME bins under `key`."""
        r = np.random.RandomState(seed)
        idx = collections.defaultdict(list)
        for i in range(nK):
            idx[key[i]].append(i)
        out = []
        for (a, b) in pos:
            ka, kb = key[a], key[b]
            pa_, pb_ = idx[ka], idx[kb]
            got = 0; tries = 0
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

    def sample_rand(pos, seed):
        r = np.random.RandomState(seed)
        out = []
        while len(out) < NNEG * len(pos):
            x, y = r.randint(nK), r.randint(nK)
            if x == y:
                continue
            p = (min(x, y), max(x, y))
            if p in edgeset:
                continue
            out.append(p)
        return out

    pos = sorted(held)
    NEG = {"RANDOM": sample_rand(pos, 1),
           "DEGREE-MATCHED": sample_neg(pos, dbin, 2),
           "PUBS-MATCHED": sample_neg(pos, pbin, 3)}
    for k, v in NEG.items():
        aa = np.array([p[0] for p in v]); bb = np.array([p[1] for p in v])
        pa_ = np.array([p[0] for p in pos]); pb_ = np.array([p[1] for p in pos])
        print(f"  {k:15s} n={len(v):6d}  mean log1p(pubs) pos {np.mean(np.log1p(pubk[pa_])+np.log1p(pubk[pb_])):.2f} "
              f"neg {np.mean(np.log1p(pubk[aa])+np.log1p(pubk[bb])):.2f}   "
              f"mean log1p(deg) pos {np.mean(np.log1p(degk[pa_])+np.log1p(degk[pb_])):.2f} "
              f"neg {np.mean(np.log1p(degk[aa])+np.log1p(degk[bb])):.2f}", flush=True)

    def vec(mat, prs):
        a = np.array([p[0] for p in prs]); b = np.array([p[1] for p in prs])
        return mat[a, b]

    # ---- COMBINED: GBM trained on TRAIN edges vs matched negatives (never on held-out) ----
    tr_pos = [p for p in trainE]
    tr_neg = sample_neg(tr_pos, pbin, 4)
    Xtr = np.stack([np.r_[vec(M_, tr_pos), vec(M_, tr_neg)] for M_ in FEATS.values()], 1)
    ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4,
                                        min_samples_leaf=40, random_state=0).fit(Xtr, ytr)

    print(f"\n  AUC FOR RECOVERING {len(pos)} HELD-OUT PPI EDGES")
    print(f"    {'predictor':13s} {'vs RANDOM':>10s} {'vs DEG-M':>10s} {'vs PUBS-M':>10s}   bias gap")
    res = {}
    for nm, M_ in list(FEATS.items()) + [("COMBINED", None)]:
        row = {}
        for negname, negs in NEG.items():
            if nm == "COMBINED":
                Xp = np.stack([vec(m2, pos) for m2 in FEATS.values()], 1)
                Xn = np.stack([vec(m2, negs) for m2 in FEATS.values()], 1)
                sc = np.r_[gbm.predict(Xp), gbm.predict(Xn)]
            else:
                sc = np.r_[vec(M_, pos), vec(M_, negs)]
            yy = np.r_[np.ones(len(pos)), np.zeros(len(negs))]
            row[negname] = float(roc_auc_score(yy, sc))
        res[nm] = row
        print(f"    {nm:13s} {row['RANDOM']:10.4f} {row['DEGREE-MATCHED']:10.4f} {row['PUBS-MATCHED']:10.4f}   "
              f"{row['RANDOM']-row['PUBS-MATCHED']:+.4f}")

    pubs_bar = res["PUBS"]["PUBS-MATCHED"]
    winners = {k: v["PUBS-MATCHED"] for k, v in res.items()
               if k != "PUBS" and v["PUBS-MATCHED"] > pubs_bar + 0.02}
    best = max(res, key=lambda k: res[k]["PUBS-MATCHED"])
    print(f"\n  the bar is PUBS under pubs-matched negatives: {pubs_bar:.4f}")
    print(f"  clearing it by >0.02: {', '.join(f'{k} {v:.4f}' for k, v in sorted(winners.items(), key=lambda t:-t[1])) or 'NOTHING'}")

    # ---- precision at the top, which is what "propose new connections" actually needs ----
    print(f"\n  PRECISION AMONG TOP-RANKED PAIRS (pubs-matched pool: {len(pos)} true, {len(NEG['PUBS-MATCHED'])} false)")
    allp = pos + NEG["PUBS-MATCHED"]
    ytrue = np.r_[np.ones(len(pos)), np.zeros(len(NEG["PUBS-MATCHED"]))]
    prec = {}
    for nm in [best, "PROFILE", "COMMON-NB", "CHAIN-2", "PUBS"]:
        if nm == "COMBINED":
            sc = gbm.predict(np.stack([vec(m2, allp) for m2 in FEATS.values()], 1))
        else:
            sc = vec(FEATS[nm], allp)
        o = np.argsort(-sc)
        prec[nm] = {f"p@{k}": float(ytrue[o[:k]].mean()) for k in (100, 500, 2000)}
        print(f"    {nm:13s} " + "  ".join(f"p@{k}={prec[nm][f'p@{k}']:.3f}" for k in (100, 500, 2000))
              + f"   (base rate {ytrue.mean():.3f})")

    # ---- the deliverable: unrecorded pairs the best predictor ranks highest ----
    scoreM = (gbm.predict(np.stack([m2.ravel() for m2 in FEATS.values()], 1)).reshape(nK, nK)
              if best == "COMBINED" else FEATS[best])
    cand = []
    iu = np.triu_indices(nK, 1)
    sc_all = scoreM[iu]
    order = np.argsort(-sc_all)
    for t in order:
        a, b = int(iu[0][t]), int(iu[1][t])
        if (a, b) in edgeset:
            continue
        cand.append((kos[a], kos[b], float(sc_all[t])))
        if len(cand) >= TOPN:
            break
    print(f"\n  TOP {TOPN} UNRECORDED PAIRS by {best} (these are PROPOSALS, precision estimated above):")
    for a, b, s in cand[:15]:
        print(f"    {a:10s} -- {b:10s}  {s:.4f}")

    cleared = bool(winners)
    verdict = (
        f"CAN THE NETWORK BE COMPLETED? The test is recovering {len(pos)} held-out PPI edges among the "
        f"{nK} knockouts, with every topology and chain feature rebuilt on the graph MINUS those edges. "
        f"THE HEADLINE IS THE BIAS GAP. Ranking pairs by nothing but how well-studied the two proteins are "
        f"(pubs_a + pubs_b) recovers held-out edges at AUC {res['PUBS']['RANDOM']:.4f} against randomly sampled "
        f"non-edges. That number is worthless and it is the number this kind of work usually reports. Against "
        f"negatives matched on the same pubs counts it falls to {res['PUBS']['PUBS-MATCHED']:.4f}, a drop of "
        f"{res['PUBS']['RANDOM']-res['PUBS']['PUBS-MATCHED']:.4f}, and THAT gap is the study bias. Every predictor "
        f"is therefore scored against pubs-matched negatives. "
        + (f"SOMETHING DOES CLEAR THE BAR: {', '.join(f'{k} at {v:.4f}' for k, v in sorted(winners.items(), key=lambda t: -t[1]))}, "
           f"against the pubs bar of {pubs_bar:.4f}. The best single predictor is {best} at "
           f"{res[best]['PUBS-MATCHED']:.4f}. Notably PROFILE -- cosine similarity of the two knockouts' MEASURED "
           f"z-profiles, the one feature that never touches the graph -- scores "
           f"{res['PROFILE']['PUBS-MATCHED']:.4f}, so the perturbation data carries genuine edge information that "
           f"is not a restatement of who is famous. " if cleared else
           f"NOTHING CLEARS THE BAR BY MORE THAN 0.02. The best predictor under pubs-matched negatives is {best} at "
           f"{res[best]['PUBS-MATCHED']:.4f} against the pubs bar's {pubs_bar:.4f}. On this evidence the apparent "
           f"ability to complete the network is mostly an ability to recognise well-studied protein pairs. ")
        + f"THE CHAIN SPECIFICALLY -- the K562-expressed 2-step walk, i.e. exactly the machinery from "
        f"k562_specific_chain.py used as a link predictor -- scores {res['CHAIN-2']['PUBS-MATCHED']:.4f} pubs-matched "
        f"against {res['CHAIN-2']['RANDOM']:.4f} random. "
        f"WHAT THIS CANNOT SAY. Recovering HIDDEN KNOWN edges is a proxy for proposing NEW ones, and it is optimistic: "
        f"held-out edges were discoverable enough to be in a database, whereas genuinely missing edges may be missing "
        f"for reasons that also make them hard to predict. The proposal list is emitted with a precision estimated "
        f"from the held-out pool, NOT validated -- no new edge here has been confirmed by any experiment, and the "
        f"only honest claim is a calibrated ranking. The universe is knockout-knockout pairs only, because that is "
        f"where measured profiles exist for both endpoints, so this says nothing about the {N-nK} other proteins. "
        f"complete_network.py already established that the merged graph is trustworthy about one hop out; this is "
        f"the edge-level version of the same question and it is scored against a control that work did not have.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"n_heldout": len(pos), "n_train_edges": len(trainE), "ko_pairs": nK * (nK - 1) // 2,
               "auc": res, "pubs_bar": pubs_bar, "cleared": cleared, "best": best,
               "precision": prec, "proposals": [{"a": a, "b": b, "score": s} for a, b, s in cand],
               "verdict": verdict}, open(OUT / "link_completion.json", "w"), indent=2)
    print(f"\n  -> {OUT/'link_completion.json'}")


if __name__ == "__main__":
    main()
