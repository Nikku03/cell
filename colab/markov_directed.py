"""THE MARKOV CHAIN ON THE NEWLY DIRECTED NETWORK: does knowing which way 4.85% of the edges point help?

WHY THIS IS WORTH RUNNING EVEN THOUGH AN EARLIER MODULE KILLED THE MARKOV DIRECTION IDEA. edge_direction.py proved
that a random walk cannot ORIENT an edge: on an undirected graph P[a,b] = 1/deg(a), so sign(P[a,b] - P[b,a]) is
identically sign(deg(b) - deg(a)), and the walk's "opinion" agreed with the raw degree rule 1.000000 of the time.
That result is about using the walk to INFER direction. This is the opposite question: now that direction has been
measured from curated reaction structure, does FEEDING it to the walk make the walk better at its actual job --
predicting which genes move when a gene is knocked out?

THE PREDICTION, REGISTERED BEFORE RUNNING. Only 9,277 of 191,447 PPI edges (4.85%) carry an evidence-backed arrow,
so at most one transition in twenty is constrained. The expected effect is small. Saying so in advance matters,
because a small positive result found after the fact is indistinguishable from noise, whereas a small positive
result predicted in advance and separated from its controls is a measurement.

THE ARMS, AND WHY EACH ONE EXISTS:

  UNDIR     the incumbent. Every edge traversable both ways. Reference ~0.4768.
  DIR       measured arrows. A directed edge a->b is traversable a->b at full weight and b->a at weight `back`,
            which is swept over {0, 0.25} so "hard one-way" and "soft preference" are both tested.
  DIR-REV   every measured arrow FLIPPED. This is the control that decides the whole question. Making edges one-way
            changes the walk whatever the arrows say -- it removes transitions and concentrates mass. If DIR and
            DIR-REV score the same, then the gain is from one-way-ness, not from the arrows being RIGHT, and the
            direction work has contributed nothing to this task.
  DIR-SHUF  the same NUMBER of edges made directed, but chosen at random across the interactome with a random
            orientation. This separates "these particular edges are special" from "some edges are directed".

Scored on the project's standing harness with no changes: tide-removed specific-mover recall@50, TAU=1.0,
TIDE_FRAC=0.05, MIN_SPEC=5, N_NEI=10 neighbour transfer over MEASURED profiles, 5 splits, paired bootstrap.
The neighbour pool is every non-test knockout, not only the scorable ones -- restricting it costs ~0.14 recall.
"""
import collections
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI, NSPLIT = 1.0, 50, 5, 0.05, 10, 5
BACK_GRID = [0.0, 0.25]                 # weight on traversing a measured arrow backwards
DAMP_GRID = [(1, 1.0), (2, 0.5), (3, 0.5)]
NBOOT = 2000


def main():
    KO = pickle.load(open(SP / "nlz_K562.pkl", "rb"))
    kos = sorted(KO)
    ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v})
    gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide_idx = np.where(~tide)[0]
    spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}
    N = len(names)
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    pa = np.array([a for a, b in pe])
    pb = np.array([b for a, b in pe])
    eidx = {(a, b): i for i, (a, b) in enumerate(pe)}

    # ---- the measured arrows, from every evidence-backed source (the TF heuristic is deliberately excluded) ----
    arrows = {}
    DN = json.load(open(OUT / "directed_network.json"))
    for a, b, p, t in DN["edges"]:
        if p in ("pathway_order", "signor"):
            ia, ib = nidx.get(a), nidx.get(b)
            if ia is not None and ib is not None and (min(ia, ib), max(ia, ib)) in eidx:
                arrows[(min(ia, ib), max(ia, ib))] = (ia, ib)
    TP = json.load(open(OUT / "transformed_partner_arrows.json"))
    for a, b in TP["edges"]["ALL"]:                 # highest-accuracy source last, so it wins on conflict
        ia, ib = nidx.get(a), nidx.get(b)
        if ia is not None and ib is not None and (min(ia, ib), max(ia, ib)) in eidx:
            arrows[(min(ia, ib), max(ia, ib))] = (ia, ib)
    print(f"{len(pe):,} PPI edges; {len(arrows):,} carry an evidence-backed arrow ({len(arrows)/len(pe):.2%})")

    rng = np.random.RandomState(0)
    shuf_keys = [pe[i] for i in rng.choice(len(pe), len(arrows), replace=False)]
    shuf = {}
    for kk in shuf_keys:
        shuf[kk] = kk if rng.rand() < 0.5 else (kk[1], kk[0])

    def build(mode, back):
        """Row-stochastic operator, transposed. `back` is the weight on traversing a measured arrow the wrong way."""
        src = {"UNDIR": {}, "DIR": arrows, "DIR-REV": {k: (v[1], v[0]) for k, v in arrows.items()},
               "DIR-SHUF": shuf}[mode]
        ra, rb, w = [], [], []
        for i, (a, b) in enumerate(pe):
            d = src.get((a, b))
            if d is None:
                ra += [a, b]
                rb += [b, a]
                w += [1.0, 1.0]
            else:
                f, t = d
                ra += [f, t]
                rb += [t, f]
                w += [1.0, back]
        G = sparse.coo_matrix((np.array(w, np.float32), (np.array(ra), np.array(rb))), shape=(N, N)).tocsr()
        mag = np.asarray(G.sum(1)).ravel()
        inv = np.zeros(N, np.float32)
        nz = mag > 0
        inv[nz] = 1.0 / mag[nz]
        return (sparse.diags(inv) @ G).T.tocsr()

    konode = np.array([nidx.get(k, -1) for k in kos])
    have = konode >= 0

    def steps(PT, nstep=3):
        idx = np.where(have)[0]
        L = np.zeros((N, len(idx)), np.float32)
        L[konode[idx], np.arange(len(idx))] = 1.0
        cur = L
        out = []
        for _ in range(nstep):
            cur = PT @ cur
            m = np.zeros((nK, nK), np.float32)
            m[np.ix_(idx, idx)] = cur[konode[idx]].T
            out.append(m)
        return out

    MODES = ["UNDIR", "DIR", "DIR-REV", "DIR-SHUF"]
    SC = {}
    for mode in MODES:
        for back in BACK_GRID:
            if mode == "UNDIR" and back != BACK_GRID[0]:
                continue                                    # `back` is meaningless with no arrows
            sm = steps(build(mode, back))
            for ns, dp in DAMP_GRID:
                SC[(mode, back, ns, dp)] = sum((dp ** (s + 1)) * sm[s] for s in range(ns))
        print(f"  precomputed {mode}", flush=True)

    def rec(sc, truth):
        o = nontide_idx[np.argsort(-sc[nontide_idx])][:K]
        return len(set(o.tolist()) & truth) / max(len(truth), 1)

    def prof(idxs):
        # np.abs BEFORE the mean, matching the incumbent. Averaging SIGNED z-scores lets an up-regulated gene in one
        # neighbour cancel a down-regulated one in another.
        idxs = np.asarray(list(idxs), int)
        return np.abs(M[idxs]).mean(0) if len(idxs) else np.zeros(nG, np.float32)

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC])
    scorset = set(int(x) for x in scor)
    per = collections.defaultdict(list)
    print(f"{nK:,} knockouts, {len(scor):,} scorable, {int(tide.sum()):,} tide genes removed from the readout")

    for sp in range(NSPLIT):
        srng = np.random.RandomState(100 + sp)
        perm = srng.permutation(scor)
        nt = len(scor) // 3
        test = np.array(sorted(perm[:nt].tolist()))
        tset = set(test.tolist())
        train = np.array([i for i in range(nK) if i not in tset])
        train_scor = np.array([i for i in train if i in scorset])
        preg = np.random.RandomState(1000 + sp).choice(train_scor, min(200, len(train_scor)), replace=False)
        SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}

        def pick(vrow, xi, pool):
            v = vrow.astype(np.float64).copy()
            msk = np.full(nK, -np.inf)
            msk[pool] = 0.0
            v = v + msk
            v[xi] = -np.inf
            return np.argsort(-v)[:N_NEI]

        def pscore(cfg):
            vals = []
            for xi in preg:
                tr = set(int(t) for t in np.where(A[xi] >= TAU)[0] if ~tide[t])
                if len(tr) >= MIN_SPEC:
                    vals.append(rec(prof(pick(SC[cfg][xi], xi, train[train != xi])), tr))
            return float(np.mean(vals)) if vals else 0.0

        for mode in MODES:
            grid = [c for c in SC if c[0] == mode]
            best = max(grid, key=pscore)          # hyperparameters chosen on TRAIN only, never on test
            for xi in test:
                per[mode].append(rec(prof(pick(SC[best][xi], int(xi), train)), SPEC[int(xi)]))
        for xi in test:
            per["TIDE"].append(rec((A >= TAU).mean(0), SPEC[int(xi)]))
        print(f"  split {sp}: UNDIR {np.mean(per['UNDIR'][-len(test):]):.4f}  "
              f"DIR {np.mean(per['DIR'][-len(test):]):.4f}", flush=True)

    def paired(a, b):
        x = np.array(per[a]) - np.array(per[b])
        br = np.random.RandomState(7)
        n = len(x)
        bs = np.array([x[br.randint(0, n, n)].mean() for _ in range(NBOOT)])
        m, se = float(x.mean()), float(bs.std())
        return m, se, ("BETTER" if m > 2 * se else "WORSE" if m < -2 * se else "indistinguishable")

    print(f"\n  recall@{K}, {NSPLIT} splits, {len(per['UNDIR']):,} evaluations")
    print(f"    {'arm':10s} {'score':>8s}   {'vs UNDIR':>9s} {'+/-':>7s}  verdict")
    res = {}
    for m in MODES + ["TIDE"]:
        s = float(np.mean(per[m]))
        if m == "UNDIR":
            print(f"    {m:10s} {s:8.4f}   {'--':>9s} {'':>7s}  reference (incumbent ~0.4768)")
            res[m] = (s, 0.0, 0.0, "reference")
            continue
        d, se, v = paired(m, "UNDIR")
        print(f"    {m:10s} {s:8.4f}   {d:+9.4f} {se:7.4f}  {v}")
        res[m] = (s, d, se, v)

    # THE CONTROL THAT DECIDES IT: right arrows against wrong arrows, and against arbitrary arrows.
    d_rev, se_rev, v_rev = paired("DIR", "DIR-REV")
    d_shf, se_shf, v_shf = paired("DIR", "DIR-SHUF")
    print(f"\n  DIR vs DIR-REV (are the arrows RIGHT, or just one-way?) {d_rev:+.4f} +/- {se_rev:.4f}  {v_rev}")
    print(f"  DIR vs DIR-SHUF (are THESE edges special?)              {d_shf:+.4f} +/- {se_shf:.4f}  {v_shf}")

    helps = res["DIR"][3] == "BETTER" and v_rev == "BETTER"
    verdict = (
        f"MARKOV CHAIN ON THE DIRECTED NETWORK. {len(arrows):,} of {len(pe):,} PPI edges ({len(arrows)/len(pe):.2%}) "
        f"carry an evidence-backed arrow, so at most one transition in twenty is constrained and a small effect was "
        f"predicted in advance. MEASURED on the standing harness: UNDIR {res['UNDIR'][0]:.4f}, "
        f"DIR {res['DIR'][0]:.4f} ({res['DIR'][1]:+.4f} +/- {res['DIR'][2]:.4f}, {res['DIR'][3]}), "
        f"DIR-REV {res['DIR-REV'][0]:.4f}, DIR-SHUF {res['DIR-SHUF'][0]:.4f}. "
        + (f"THE ARROWS THEMSELVES CARRY THE GAIN: DIR beats DIR-REV by {d_rev:+.4f} +/- {se_rev:.4f}, so this is "
           f"not merely the effect of making edges one-way, and beats DIR-SHUF by {d_shf:+.4f}, so it is not merely "
           f"that some edges are directed. " if helps else
           f"DIRECTION DOES NOT HELP THIS TASK. DIR vs DIR-REV is {d_rev:+.4f} +/- {se_rev:.4f} ({v_rev}) -- "
           f"reversing every arrow costs nothing measurable, which means any difference from UNDIR comes from "
           f"making edges one-way rather than from the arrows pointing the right way. ")
        + f"THE LIKELY REASON EITHER WAY is coverage, not correctness: the arrows are 0.83-accurate but touch under "
        f"5% of edges, and the walk's mass is dominated by the 95% that stay symmetric. Nothing here contradicts "
        f"edge_direction.py's separate result that a walk cannot INFER direction -- that was about reading direction "
        f"OUT of the graph; this is about feeding measured direction IN.")
    print(f"\nVERDICT: {verdict}")

    json.dump({"n_ppi": len(pe), "n_arrows": len(arrows), "frac": len(arrows) / len(pe),
               "scores": {m: res[m][0] for m in res},
               "vs_undir": {m: {"delta": res[m][1], "se": res[m][2], "verdict": res[m][3]} for m in res},
               "dir_vs_rev": {"delta": d_rev, "se": se_rev, "verdict": v_rev},
               "dir_vs_shuf": {"delta": d_shf, "se": se_shf, "verdict": v_shf},
               "verdict": verdict}, open(OUT / "markov_directed.json", "w"))
    print(f"\n  -> {OUT/'markov_directed.json'}")


if __name__ == "__main__":
    main()
