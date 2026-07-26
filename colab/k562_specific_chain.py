"""A CELL-SPECIFIC, MEMORYLESS, STEP-BY-STEP MARKOV CHAIN: delete the proteins K562 does not make.

THE IDEA, AND WHY IT IS NOT LIKE THE PREVIOUS ATTEMPTS. Every chain in this repo so far has walked the generic human
PPI graph -- 16,492 proteins, most of which K562 never expresses. A protein the cell does not make cannot transmit a
perturbation, so every path routed through one is fictional. Measured here: only 7,223 of the 16,492 network genes
(43.8%) are expressed in K562, so 9,269 nodes (56.2%) are ghosts. This module deletes them and re-runs the chain.

That is a change to the STATE SPACE ITSELF rather than to the transition rule, which is what makes it different from
interaction_states.py (six state spaces, all tied) -- those all kept the same node set and rearranged how mass moved
through it. Here the node set shrinks by more than half on a biological criterion.

"MEMORYLESS, STEP BY STEP" is taken literally. A Markov chain is memoryless by definition, so what is new is the
READOUT: instead of the incumbent's damped accumulation tot = sum_s damp^s P^s, each step is reported ALONE.

    S1 = P      the knocked-out protein's direct partners
    S2 = P^2    partners of partners (a plain walk, so mass returns to the seed too)
    S3 = P^3

This has never been run here. Every previous number blends the steps, so if one step carries everything and the
others add noise, the blend hides it. The step-alone arms answer that directly.

WHERE "EXPRESSED IN K562" COMES FROM, because this is the load-bearing input. Not the emask in cell_complete.json:
that is a cell-type MARKER mask flagging only 634 genes for erythroid progenitor (8.5%), which is specificity, not
presence. Instead the K562 Perturb-seq panel itself (k562.h5ad var, with per-gene mean expression), which is a
MEASURED property of these exact cells. Note the gene names there are an anndata categorical -- var/gene_name holds
integer CODES and var/__categories/gene_name holds the strings -- and decoding the codes as names silently yields a
0% overlap with the network instead of 43.8%. That trap cost one debugging round and is recorded so it is not
repeated.

THE CONTROL THAT DECIDES EVERYTHING. Pruning 56% of nodes changes the graph in two ways at once: it removes the
biologically absent proteins (the hypothesis) and it makes the graph smaller and differently connected (a confound).
So a RANDOM prune to the SAME NUMBER OF NODES is run alongside, and a DEGREE-MATCHED random prune on top of that,
because expressed genes are better studied and therefore higher-degree, and a plain random prune would not control
for it. If the K562 prune does not beat its degree-matched twin, the answer is "shrinking the graph helps, biology
had nothing to do with it".

ARMS, all sharing one identical evaluation path:
  FULL-S1/S2/S3      full 16,492-node graph, each step alone
  FULL-DAMP          full graph, damped sum -- this is the incumbent, and it must reproduce ~0.4768
  K562-S1/S2/S3      expressed-only graph, each step alone
  K562-DAMP          expressed-only, damped sum
  K562-EXPR          expressed-only AND transitions weighted by the target's abundance, so a perturbation
                     preferentially propagates into proteins the cell actually has a lot of
CONTROLS:
  RAND-PRUNE         a uniformly random node set of the same size
  DEG-PRUNE          a random node set of the same size, matched to the expressed set's degree distribution
  EXPR-SHUF          K562-EXPR with the expression values shuffled across the surviving genes

Harness constants are untouched (TAU=1.0, K=50, MIN_SPEC=5, TIDE_FRAC=0.05, N_NEI=10, 5 splits). Neighbour pool is
every non-test knockout. Profiles are averaged as abs-then-mean. Configs are pre-registered on TRAIN only.
"""
import collections
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI, NSPLIT = 1.0, 50, 5, 0.05, 10, 5
CW_GRID = [0.0, 0.1, 1.0]
DAMP_GRID = [(1, 1.0), (2, 0.5), (3, 0.5)]
NBOOT = 2000


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
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

    # ---- THE CELL STATE: which proteins does K562 actually make, and how much ----
    # anndata categorical: var/gene_name are integer CODES into var/__categories/gene_name.
    f = h5py.File(f"{SP}/k562.h5ad", "r")
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["__categories"]["gene_name"][:]]
    gname = np.array(cats)[f["var"]["gene_name"][:]]
    gmean = np.asarray(f["var"]["mean"][:], float)
    expr = {}
    for n_, m_ in zip(gname.tolist(), gmean.tolist()):
        expr[n_] = max(expr.get(n_, 0.0), float(m_))
    expressed = np.zeros(N, bool)
    exprv = np.zeros(N, np.float32)
    for i, n_ in enumerate(names):
        if n_ in expr:
            expressed[i] = True
            exprv[i] = expr[n_]
    print(f"network {N} genes; expressed in K562 {int(expressed.sum())} ({expressed.mean():.1%}); "
          f"ghosts removed {int((~expressed).sum())}", flush=True)

    # ---- graph ----
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    pa = np.array([a for a, b in pe]); pb = np.array([b for a, b in pe])
    memb = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if g.isdigit() and int(g) < N:
            for c in cs:
                memb[c].append(int(g))
    cmem = [sorted(set(v)) for v in memb.values() if 2 <= len(set(v)) <= 60]

    deg_full = np.zeros(N)
    np.add.at(deg_full, pa, 1); np.add.at(deg_full, pb, 1)
    print(f"{len(pe)} PPI edges; both endpoints expressed: "
          f"{int((expressed[pa] & expressed[pb]).sum())} ({(expressed[pa] & expressed[pb]).mean():.1%})", flush=True)

    konode = np.array([nidx.get(k, -1) for k in kos])
    have = konode >= 0
    ko_expressed = np.array([konode[i] >= 0 and expressed[konode[i]] for i in range(nK)])
    print(f"{int(ko_expressed.sum())}/{nK} knockouts are expressed; "
          f"{int((spec_n >= MIN_SPEC).sum())} scorable, of which "
          f"{int((ko_expressed & (spec_n >= MIN_SPEC)).sum())} expressed", flush=True)

    rng = np.random.RandomState(0)

    def rand_keep(seed, n_keep):
        r = np.random.RandomState(seed)
        m = np.zeros(N, bool); m[r.choice(N, n_keep, replace=False)] = True
        return m

    def deg_matched_keep(seed, ref_mask):
        """Random node set of the same size, matched to ref_mask's DEGREE distribution. Expressed genes are better
        studied and so higher-degree; a uniform random prune would not control for that."""
        r = np.random.RandomState(seed)
        bins = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 10 ** 9])
        b = np.digitize(deg_full, bins)
        keep = np.zeros(N, bool)
        for bb in np.unique(b):
            pool = np.where(b == bb)[0]
            want = int((b[ref_mask] == bb).sum())
            if want > 0:
                keep[r.choice(pool, min(want, len(pool)), replace=False)] = True
        return keep

    def build(keep, cw, wexpr=False, exprvec=None):
        """Row-stochastic operator restricted to `keep`, optionally weighting each transition by the TARGET's
        abundance. Returned transposed for x <- P^T x."""
        ev = exprv if exprvec is None else exprvec
        m = keep[pa] & keep[pb]
        ra = np.r_[pa[m], pb[m]]; rb = np.r_[pb[m], pa[m]]
        w = np.ones(len(ra), np.float32)
        if cw > 0:
            xa, xb = [], []
            for mem in cmem:
                mm = [x for x in mem if keep[x]]
                for x in mm:
                    for y in mm:
                        if x != y:
                            xa.append(x); xb.append(y)
            if xa:
                ra = np.r_[ra, np.array(xa)]; rb = np.r_[rb, np.array(xb)]
                w = np.r_[w, np.full(len(xa), cw, np.float32)]
        if wexpr:
            w = w * np.log1p(ev[rb]).astype(np.float32)      # prefer abundant TARGETS
        G = sparse.coo_matrix((w, (ra, rb)), shape=(N, N)).tocsr()
        mag = np.asarray(abs(G).sum(1)).ravel()
        inv = np.zeros(N, np.float32); nz = mag > 0; inv[nz] = 1.0 / mag[nz]
        return (sparse.diags(inv) @ G).T.tocsr()

    def steps(PT, nstep):
        """Return the PURE s-step distributions, s = 1..nstep, each ALONE. No damping, no accumulation."""
        idx = np.where(have)[0]
        L = np.zeros((N, len(idx)), np.float32); L[konode[idx], np.arange(len(idx))] = 1.0
        cur = L; out = []
        for _ in range(nstep):
            cur = PT @ cur
            m = np.zeros((nK, nK), np.float32)
            m[np.ix_(idx, idx)] = cur[konode[idx]].T
            out.append(m)
        return out

    def damped(stepmats, nstep, damp):
        return sum((damp ** (s + 1)) * stepmats[s] for s in range(nstep))

    keep_full = np.ones(N, bool)
    keep_k562 = expressed.copy()
    nkeep = int(keep_k562.sum())
    keep_rand = rand_keep(11, nkeep)
    keep_deg = deg_matched_keep(12, keep_k562)
    shuf_expr = exprv.copy()
    sh_idx = np.where(keep_k562)[0]
    shuf_expr[sh_idx] = shuf_expr[np.random.RandomState(13).permutation(sh_idx)]
    for nm, kp in [("FULL", keep_full), ("K562", keep_k562), ("RAND-PRUNE", keep_rand), ("DEG-PRUNE", keep_deg)]:
        m = kp[pa] & kp[pb]
        print(f"   {nm:11s} nodes {int(kp.sum()):6d}  surviving PPI edges {int(m.sum()):7d}", flush=True)

    SC = {}
    GRAPHS = [("FULL", keep_full, False, None), ("K562", keep_k562, False, None),
              ("RAND-PRUNE", keep_rand, False, None), ("DEG-PRUNE", keep_deg, False, None),
              ("K562-EXPR", keep_k562, True, None), ("EXPR-SHUF", keep_k562, True, shuf_expr)]
    for gname_, kp, we, ev in GRAPHS:
        for cw in CW_GRID:
            sm = steps(build(kp, cw, we, ev), 3)
            for s in range(3):
                SC[(f"{gname_}-S{s+1}", cw)] = sm[s]
            for (ns, dp) in DAMP_GRID:
                SC[(f"{gname_}-DAMP", cw, ns, dp)] = damped(sm, ns, dp)
        print(f"  precomputed {gname_}", flush=True)

    STEP_ARMS = [f"{g}-S{s}" for g, _, _, _ in GRAPHS for s in (1, 2, 3)]
    DAMP_ARMS = [f"{g}-DAMP" for g, _, _, _ in GRAPHS]
    ALL = STEP_ARMS + DAMP_ARMS

    def rec(sc, truth):
        o = nontide_idx[np.argsort(-sc[nontide_idx])][:K]
        return len(set(o.tolist()) & truth) / max(len(truth), 1)

    def prof(idxs):
        idxs = np.asarray(list(idxs), int)
        return np.abs(M[idxs]).mean(0) if len(idxs) else np.zeros(nG, np.float32)

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC]); scorset = set(int(x) for x in scor)
    per = collections.defaultdict(list)
    picks = collections.defaultdict(list)

    for sp in range(NSPLIT):
        srng = np.random.RandomState(100 + sp)
        perm = srng.permutation(scor); nt = len(scor) // 3
        test = np.array(sorted(perm[:nt].tolist())); tset = set(test.tolist())
        train = np.array([i for i in range(nK) if i not in tset])
        train_scor = np.array([i for i in train if i in scorset])
        preg = np.random.RandomState(1000 + sp).choice(train_scor, min(200, len(train_scor)), replace=False)
        SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}

        def pick(vrow, xi, pool):
            v = vrow.astype(np.float64).copy()
            msk = np.full(nK, -np.inf); msk[pool] = 0.0
            v = v + msk; v[xi] = -np.inf
            return np.argsort(-v)[:N_NEI]

        def pscore(cfg):
            v = []
            for xi in preg:
                tr = set(int(t) for t in np.where(A[xi] >= TAU)[0] if ~tide[t])
                if len(tr) >= MIN_SPEC:
                    v.append(rec(prof(pick(SC[cfg][xi], xi, train[train != xi])), tr))
            return float(np.mean(v)) if v else 0.0

        for arm in ALL:
            grid = ([(arm, cw) for cw in CW_GRID] if arm in STEP_ARMS else
                    [(arm, cw, ns, dp) for cw in CW_GRID for (ns, dp) in DAMP_GRID])
            best = max(grid, key=pscore)
            picks[arm].append(best[1:])
            for xi in test:
                per[arm].append(rec(prof(pick(SC[best][xi], int(xi), train)), SPEC[int(xi)]))
        for xi in test:
            per["TIDE"].append(rec((A >= TAU).mean(0), SPEC[int(xi)]))
        print(f"  split {sp}: FULL-DAMP {np.mean(per['FULL-DAMP'][-len(test):]):.4f}  "
              f"K562-DAMP {np.mean(per['K562-DAMP'][-len(test):]):.4f}", flush=True)

    def paired(a, b):
        x = np.array(per[a]) - np.array(per[b])
        br = np.random.RandomState(7); n = len(x)
        bs = np.array([x[br.randint(0, n, n)].mean() for _ in range(NBOOT)])
        m, se = float(x.mean()), float(bs.std())
        return m, se, ("BETTER" if m > 2 * se else "WORSE" if m < -2 * se else "indistinguishable")

    print(f"\n  STEP-BY-STEP, EACH STEP ALONE (recall@{K}, {NSPLIT} splits, {len(per['FULL-S1'])} evaluations)")
    print(f"    {'arm':16s} {'step1':>8s} {'step2':>8s} {'step3':>8s}")
    for g, _, _, _ in GRAPHS:
        print(f"    {g:16s} " + " ".join(f"{np.mean(per[f'{g}-S{s}']):8.4f}" for s in (1, 2, 3)))

    print(f"\n  DAMPED SUMS, and the comparison that matters")
    print(f"    {'arm':16s} {'score':>8s}   {'vs FULL':>9s} {'+/-':>7s}  verdict")
    rows = {}
    for a in DAMP_ARMS + ["TIDE"]:
        s = float(np.mean(per[a]))
        if a == "FULL-DAMP":
            print(f"    {a:16s} {s:8.4f}   {'--':>9s} {'':>7s}  reference (incumbent ~0.4768)")
            rows[a] = (s, 0.0, 0.0, "reference"); continue
        d, se, v = paired(a, "FULL-DAMP")
        print(f"    {a:16s} {s:8.4f}   {d:+9.4f} {se:7.4f}  {v}")
        rows[a] = (s, d, se, v)
    rows["FULL-DAMP"] = (float(np.mean(per["FULL-DAMP"])), 0.0, 0.0, "reference")

    print(f"\n  DOES THE BIOLOGY MATTER, OR ONLY THE SHRINKING? K562 against its size-matched twins")
    ctrl = {}
    for c in ["RAND-PRUNE-DAMP", "DEG-PRUNE-DAMP"]:
        d, se, v = paired("K562-DAMP", c)
        ctrl[c] = (d, se, v)
        print(f"    K562-DAMP vs {c:18s} {d:+.4f} +/- {se:.4f}  {v}")
    d_ex, se_ex, v_ex = paired("K562-EXPR-DAMP", "EXPR-SHUF-DAMP")
    print(f"    K562-EXPR-DAMP vs EXPR-SHUF-DAMP  {d_ex:+.4f} +/- {se_ex:.4f}  {v_ex}")

    d_k, se_k, v_k = paired("K562-DAMP", "FULL-DAMP")
    real_biology = (ctrl["DEG-PRUNE-DAMP"][2] == "BETTER")
    best_step = max([(f"{g}-S{s}", float(np.mean(per[f'{g}-S{s}']))) for g, _, _, _ in GRAPHS for s in (1, 2, 3)],
                    key=lambda t: t[1])

    verdict = (
        f"A CELL-SPECIFIC MARKOV CHAIN: DELETE THE PROTEINS K562 DOES NOT MAKE. Of the 16,492 proteins in the generic "
        f"PPI graph every previous chain in this repo has walked, only {int(expressed.sum())} ({expressed.mean():.1%}) "
        f"are expressed in K562, so {int((~expressed).sum())} nodes are ghosts and every path through one is "
        f"fictional. Pruning them removes {len(pe) - int((expressed[pa] & expressed[pb]).sum())} of {len(pe)} PPI "
        f"edges. THE MEMORYLESS CHAIN IS ALSO REPORTED STEP BY STEP for the first time here -- every previous number "
        f"blended the steps into a damped sum, which hides whether one step carries everything. Step alone, on the "
        f"full graph: S1 {np.mean(per['FULL-S1']):.4f}, S2 {np.mean(per['FULL-S2']):.4f}, "
        f"S3 {np.mean(per['FULL-S3']):.4f}; on the K562 graph: S1 {np.mean(per['K562-S1']):.4f}, "
        f"S2 {np.mean(per['K562-S2']):.4f}, S3 {np.mean(per['K562-S3']):.4f}. The best single step anywhere is "
        f"{best_step[0]} at {best_step[1]:.4f}, against the damped incumbent's {rows['FULL-DAMP'][0]:.4f}. "
        f"THE HEADLINE: K562-pruned scores {rows['K562-DAMP'][0]:.4f} against the full graph's "
        f"{rows['FULL-DAMP'][0]:.4f}, {d_k:+.4f} +/- {se_k:.4f} ({v_k}). "
        + (f"AND THE BIOLOGY IS WHAT DOES IT, not the shrinking: against a uniformly random prune to the SAME node "
           f"count it is {ctrl['RAND-PRUNE-DAMP'][0]:+.4f} ({ctrl['RAND-PRUNE-DAMP'][2]}), and against a random prune "
           f"MATCHED ON THE DEGREE DISTRIBUTION -- the control that matters, because expressed genes are better "
           f"studied and therefore higher-degree -- it is {ctrl['DEG-PRUNE-DAMP'][0]:+.4f} "
           f"({ctrl['DEG-PRUNE-DAMP'][2]}). " if real_biology else
           f"BUT THE BIOLOGY IS NOT WHAT DOES IT. Against a random prune matched on the DEGREE DISTRIBUTION, K562 is "
           f"{ctrl['DEG-PRUNE-DAMP'][0]:+.4f} +/- {ctrl['DEG-PRUNE-DAMP'][1]:.4f} ({ctrl['DEG-PRUNE-DAMP'][2]}). "
           f"Expressed genes are better studied and so higher-degree, and once that is matched the expression "
           f"criterion adds nothing measurable -- so whatever pruning does here, it does by changing the degree "
           f"structure, not by removing biologically absent proteins. ")
        + f"Weighting transitions by the target's abundance gives {rows['K562-EXPR-DAMP'][0]:.4f}, and against the "
        f"same graph with expression values SHUFFLED across surviving genes that is {d_ex:+.4f} ({v_ex}). "
        f"WHAT THIS CANNOT SAY: 'expressed' is defined by presence in the K562 Perturb-seq panel, which conflates "
        f"'not expressed' with 'not measured', and the panel is itself an expression-based selection, so the "
        f"criterion is circular to an unknown degree. {int((~ko_expressed).sum())} of {nK} knockouts are not in the "
        f"expressed set and get an all-zero score row in the pruned arms. Five splits of one cell line, one "
        f"pre-registered config grid, neighbour pool = every non-test knockout.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"expressed": int(expressed.sum()), "network": N, "frac": float(expressed.mean()),
               "ppi_edges": len(pe), "ppi_edges_expressed": int((expressed[pa] & expressed[pb]).sum()),
               "steps": {f"{g}-S{s}": float(np.mean(per[f"{g}-S{s}"])) for g, _, _, _ in GRAPHS for s in (1, 2, 3)},
               "damped": {a: {"score": rows[a][0], "delta": rows[a][1], "se": rows[a][2], "verdict": rows[a][3]}
                          for a in rows},
               "controls": {c: {"delta": ctrl[c][0], "se": ctrl[c][1], "verdict": ctrl[c][2]} for c in ctrl},
               "expr_shuffle": {"delta": d_ex, "se": se_ex, "verdict": v_ex},
               "ko_not_expressed": int((~ko_expressed).sum()),
               "verdict": verdict}, open(OUT / "k562_specific_chain.json", "w"), indent=2)
    print(f"\n  -> {OUT/'k562_specific_chain.json'}")


if __name__ == "__main__":
    main()
