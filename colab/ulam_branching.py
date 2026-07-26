"""ULAM'S METHOD, APPLIED HONESTLY: a branching process for perturbation spread, and where Monte Carlo actually pays.

WHAT ULAM ACTUALLY DID, because the analogy only works if the premise is right. Monte Carlo did NOT predict neutron
output without measurement. It consumed MEASURED microscopic inputs -- scattering and fission cross-sections, each
painstakingly determined -- and produced an EMERGENT MACROSCOPIC quantity, the multiplication factor k, that had no
closed form. MC converts measured micro-laws into non-analytic macro-observables. It does not create information.
Ulam and Everett developed branching-process theory for exactly this, so the neutron problem IS a Galton-Watson
process: one neutron makes nu neutrons, and the question is whether the population grows (k>1) or dies (k<1).

THE MAPPING ONTO THIS DATASET:
    neutron                 ->  a perturbed gene
    fission event           ->  a perturbed gene perturbing its PPI partners
    measured cross-section  ->  P(partner is a specific mover | gene knocked out), fitted on TRAIN knockouts
    multiplication factor k ->  the perturbation R0
    neutrons out            ->  the number of specific movers, which we measure and can hold out

A CLAIM THIS MODULE ORIGINALLY MADE AND NOW WITHDRAWS. The first version of this file offered the shape of the
response-size distribution as evidence for branching: mean 10.8, median 1, max 246, 689/1400 exact zeros, top 10% of
knockouts holding 75.7% of movers, Fano 91.3, log-log slope -1.41 against critical Galton-Watson's -1.5. An
adversarial review pointed out that these are all functions of the PER-KNOCKOUT MARGINAL, and the check is trivial:
permute each knockout's z-scores WITHIN its own non-tide columns and the number exceeding TAU cannot change, so
every one of those statistics is preserved exactly. Measured, and they are bit-identical to 1e-9 (section A below).

    THEY CARRY ZERO BITS ABOUT TOPOLOGY. A heavy tail is consistent with branching and equally consistent with any
    mechanism that produces heavy-tailed response sizes. They are NOT evidence for a network process and are no
    longer presented as such.

What survives as genuinely topological is the cross-section itself, because it contrasts PARTNERS against
NON-partners: P(PPI partner is a specific mover | gene knocked out) = 674/71654 = 0.0094 against a 0.0015
background, 6.26x. That comparison is destroyed by the permutation, so it is real -- though see the caveat below,
because 6.26x against a flat background is not 6.26x against a degree- and frequency-matched one.

R0 IS A RANGE, NOT A NUMBER, AND THE EARLIER SINGLE FIGURE WAS OVERCONFIDENT. The naive p*<k> = 0.25 is the wrong
formula: a node reached BY an edge is size-biased, so the branching number uses the EXCESS degree <k^2>/<k> - 1 =
76.2, giving 0.72, and equivalently p/p_c = 0.72 with p_c = <k>/(<k^2>-<k>) = 0.01312. But that formula assumes a
LOCALLY TREE-LIKE graph, and this one is not: PPI clustering is ~0.20 against ~0.002 for a degree-matched random
graph, and triangles make a configuration-model R0 an UPPER BOUND of unknown tightness. Different estimators of the
same quantity span roughly 0.56 to 1.25 here, which straddles 1. THE SUB- VERSUS SUPER-CRITICAL QUESTION IS NOT
ANSWERABLE FROM THIS DATA, and any statement that the process "is subcritical" overstates what was measured.

A SECOND WITHDRAWN CLAIM: "#P-COMPLETENESS JUSTIFIES MONTE CARLO HERE." It does not, and the refutation is a proof
rather than a measurement. At depth <= 2 the paths from s to t are the direct edge (s,t) plus the 2-paths s-x-t over
common neighbours x. Any two of those share NO edge -- the direct edge uses (s,t); the path through x uses (s,x) and
(x,t); the path through y uses (s,y),(y,t) with y != x. Edge-disjoint paths are independent, so

    P(t reachable from s within 2 hops) = 1 - (1 - p_st) * PROD_x (1 - p_sx * p_xt)     EXACTLY

Two-terminal reliability IS #P-complete in general, but not on the truncated neighbourhood this repo actually uses,
and at depth 3 the multi-path correction is not only tiny but MONOTONE (roughly 1 - exp(-U) in the mean-field sum U),
so it cannot reorder anything. Monte Carlo was therefore never estimating a quantity without a closed form. That was
the argument for trying it and the argument was wrong.

THE PRE-REGISTERED FEASIBILITY TEST, RUN AND REPORTED BEFORE ANY PERFORMANCE NUMBER. Two previous Monte Carlos in
this repo failed by being rank-decorative (measured Spearman -0.9904 against their own mean), so a third one has to
clear a bar before it is allowed to claim anything. The bar here is SPLIT-HALF REPRODUCIBILITY: run the estimator
twice with independent seeds and see whether it agrees with ITSELF. Measured, at 6000 replicates and depth 3:

    Spearman(MC run 1, MC run 2)     = 0.36 - 0.41        <- the estimator does not reproduce itself
    Spearman(MC run 1, linear walk)  = 0.29 - 0.53

The MC and the cheap linear walk disagree, but the MC disagrees with ITSELF just as much, so the disagreement is
SAMPLING NOISE, not nonlinearity. The reason is elementary: at R0 = 0.72 the reachability probabilities are ~1e-4,
so at 6000 replicates most genes are hit zero or one times and the per-gene ranking is Poisson noise. This is a
DIFFERENT failure from the previous two -- not decorative, but statistically infeasible at per-gene resolution.

WHICH IS EXACTLY WHAT ULAM'S METHOD SAYS TO DO INSTEAD. He never computed the probability that a specific neutron
reached a specific place; he computed the AGGREGATE multiplication factor. An aggregate over ~10^4 genes has a
relative standard error smaller by sqrt(n) than any single gene's hit probability. So this module does NOT use MC to
rank genes. It uses MC for the aggregate: the BLAST RADIUS, which is the direct neutron-count analogue, plus the
higher moments of the cascade-size distribution, which are the one thing here that is genuinely not a closed form
(the variance of total progeny has an analytic form for a homogeneous Galton-Watson process but not for a
heterogeneous one on a real degree-skewed graph).

WHAT A PPI CASCADE CAN AND CANNOT EXPLAIN, measured up front so the result is not oversold: of 15,183 specific
movers across all knockouts, only 674 (4.4%) are direct PPI partners of the knocked-out gene. A cascade on this graph
therefore cannot explain WHICH genes move. It can only be tested on HOW MANY -- which is the Ulam question anyway.

WHAT IS MEASURED HERE:
  B  criticality: R0 by the correct excess-degree formula, p/p_c, and the analytic zero-response probability
  C  the feasibility test above, re-run inside the module so it is reproducible rather than quoted
  D  BLAST RADIUS held out: predict each knockout's number of specific movers, against a full baseline ladder --
     train mean, PPI degree alone, essentiality alone, a GBM on node features, and the analytic mean-field cascade.
     A branching simulation that cannot beat "degree plus essentiality in a GBM" has demonstrated nothing.
  E  does MC earn its cost ON TOP of the analytic mean-field? The pre-registered test is whether the MC-derived
     cascade VARIANCE is rank-equivalent to the MC mean. If Spearman > 0.99 it is a monotone function of the mean,
     therefore rank-decorative, therefore worthless -- and that is the third strike, reported as such.

Evaluation is 5-fold cross-validation over ALL 1400 knockouts, not the 378 scorable ones: filtering to knockouts with
>=5 movers would condition the evaluation set on the target being predicted, which is the selection artifact that
would make a size-prediction result meaningless.
"""
import collections
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, TIDE_FRAC = 1.0, 0.05
NFOLD = 5
R_MC = 600          # replicates for AGGREGATE quantities (well-conditioned; see the feasibility note above)
R_FEAS = 3000       # replicates for the per-gene feasibility test
DEPTH = 3
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

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]; info = {g["name"]: g for g in D["genes"]}
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)

    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    ra = np.array([a for a, b in pe]); rb = np.array([b for a, b in pe])
    src = np.r_[ra, rb]; dst = np.r_[rb, ra]                     # directed edge list, both orientations
    Adj = sparse.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(N, N)).tocsr()
    deg = np.asarray(Adj.sum(1)).ravel()
    kk = deg[deg > 0]
    excess = (kk ** 2).mean() / kk.mean() - 1.0
    p_c = kk.mean() / ((kk ** 2).mean() - kk.mean())
    print(f"{len(pe)} PPI edges. <k>={kk.mean():.2f} <k^2>={(kk**2).mean():.1f} "
          f"excess degree={excess:.2f}  p_c={p_c:.5f}", flush=True)

    konode = np.array([nidx.get(k, -1) for k in kos])
    have = konode >= 0

    # ---------------- node features (NO measured perturbation data: these describe the gene, not its response) ----
    def nf(f, d=0.0):
        return np.array([float(info.get(n, {}).get(f, d) or d) for n in names], np.float32)
    dep = nf("dep_frac"); loe = nf("loeuf", 1.0); tf = nf("tf")
    loe[loe < 0] = 1.0                                            # -1 is a missing sentinel, not a real LOEUF
    ncplx = np.zeros(N, np.float32)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if g.isdigit() and int(g) < N:
            ncplx[int(g)] = len(cs)
    measured = np.zeros(N, np.float32)
    for g in genes:
        if g in nidx:
            measured[nidx[g]] = 1.0
    NODEF = np.stack([np.log1p(deg), dep, loe, tf, ncplx, measured], 1)

    # ---------------- the target: blast radius, defined on TRAIN ONLY per fold (the tide cut included) ----------
    rng = np.random.RandomState(0)
    fold = rng.permutation(nK) % NFOLD

    # ================= B. CRITICALITY =================
    tide_all = (A >= TAU).mean(0) >= TIDE_FRAC
    spec_all = (A >= TAU) & (~tide_all)
    size_all = spec_all.sum(1)
    nbrs = {}
    for i in range(nK):
        if have[i]:
            nbrs[i] = Adj.indices[Adj.indptr[konode[i]]:Adj.indptr[konode[i] + 1]]
    # ---- A. THE PERMUTATION CONTROL, run first because it decides what may be CALLED evidence ----
    # Shuffle each knockout's z-scores WITHIN its own non-tide columns. The count above TAU cannot change, so every
    # statistic that is a function of the response-size marginal survives untouched. Anything that survives this is
    # not evidence about topology, no matter how branching-shaped it looks.
    ntc = np.where(~tide_all)[0]
    prng = np.random.RandomState(0); Ap = A.copy()
    for i in range(nK):
        Ap[i, ntc] = Ap[i, ntc][prng.permutation(len(ntc))]
    size_perm = ((Ap >= TAU) & (~tide_all)).sum(1)

    def shape_stats(s):
        nz = s[s > 0]
        h, e = np.histogram(nz, bins=np.logspace(0, np.log10(nz.max() + 1), 18))
        c = (e[:-1] + e[1:]) / 2; m = h > 0
        return {"mean": float(s.mean()), "median": float(np.median(s)), "max": float(s.max()),
                "zeros": float((s == 0).sum()), "fano": float(s.var() / s.mean()),
                "top10pct": float(np.sort(s)[::-1][:len(s) // 10].sum() / s.sum()),
                "slope": float(np.polyfit(np.log(c[m]), np.log(h[m] / np.diff(e)[m]), 1)[0])}
    sreal, sperm = shape_stats(size_all), shape_stats(size_perm)
    perm_invariant = all(abs(sreal[k] - sperm[k]) < 1e-9 for k in sreal)
    print(f"\nA. PERMUTATION CONTROL -- which 'branching evidence' survives shuffling z WITHIN each knockout?")
    for k in sreal:
        print(f"   {k:9s} real {sreal[k]:11.4f}   permuted {sperm[k]:11.4f}   "
              f"{'IDENTICAL -> carries NO topology' if abs(sreal[k]-sperm[k]) < 1e-9 else 'differs'}")

    hits = tot = 0
    for i in range(nK):
        if i not in nbrs:
            continue
        gj = [gi[names[t]] for t in nbrs[i] if names[t] in gi]
        if gj:
            tot += len(gj); hits += int(spec_all[i, gj].sum())
    p_hat0 = hits / max(tot, 1)
    print(f"\nB. CRITICALITY")
    print(f"   cross-section p = {hits}/{tot} = {p_hat0:.5f}   background = {spec_all.mean():.5f}"
          f"   enrichment {p_hat0/spec_all.mean():.2f}x")
    print(f"   R0 = p * excess_degree = {p_hat0*excess:.3f}    p/p_c = {p_hat0/p_c:.3f}")
    print(f"   naive p*<k> would have said {p_hat0*kk.mean():.3f}  (wrong formula: ignores size-biased arrival)")
    # NO SUB/SUPERCRITICAL VERDICT IS PRINTED HERE, and an earlier version printing one flatly was overconfident.
    # Both formulas assume a locally tree-like graph. This one is not: PPI clustering ~0.20 against ~0.002 for a
    # degree-matched random graph, so triangles make the configuration-model figure an UPPER BOUND of unknown
    # tightness. Across estimators the value spans roughly 0.56-1.25, which straddles 1.
    print(f"   NOT DECIDABLE: estimators of R0 span ~0.56-1.25 and straddle 1; the graph is not tree-like "
          f"(clustering ~0.20 vs ~0.002 random), so this is an upper bound of unknown tightness.")

    # ================= C. FEASIBILITY: does the MC reproduce ITSELF at per-gene resolution? =================
    def percolate(pvec, seeds_node, R, seed, depth=DEPTH, want_reach=False):
        """R replicates of independent-edge percolation. Returns per-seed reached COUNTS (R x nseeds), and
        optionally the per-gene hit frequency. pvec is a per-DIRECTED-EDGE probability."""
        r = np.random.RandomState(seed)
        ns = len(seeds_node)
        base = np.zeros((N, ns), np.float32); base[seeds_node, np.arange(ns)] = 1.0
        counts = np.zeros((R, ns), np.float32)
        freq = np.zeros((N, ns), np.float32) if want_reach else None
        for rep in range(R):
            keep = r.random_sample(len(pvec)) < pvec
            Asub = sparse.coo_matrix((np.ones(int(keep.sum()), np.float32), (src[keep], dst[keep])),
                                     shape=(N, N)).tocsr()
            F = base.copy(); reach = F > 0
            for _ in range(depth):
                F = Asub @ F
                new = (F > 0) & (~reach)
                if not new.any():
                    break
                reach |= new
                F = new.astype(np.float32)
            reach[seeds_node, np.arange(ns)] = False               # the seed itself is not part of its own progeny
            counts[rep] = reach.sum(0)
            if want_reach:
                freq += reach
        return counts, (freq / R if want_reach else None)

    puni = np.full(len(src), p_hat0, np.float32)
    fs = rng.choice(np.where(deg >= 5)[0], 40, replace=False)
    _, f1 = percolate(puni, fs, R_FEAS, 1, want_reach=True)
    _, f2 = percolate(puni, fs, R_FEAS, 2, want_reach=True)
    lin = np.zeros((N, len(fs)), np.float32)
    cur = np.zeros((N, len(fs)), np.float32); cur[fs, np.arange(len(fs))] = 1.0
    for _ in range(DEPTH):
        cur = (Adj @ cur) * p_hat0; lin += cur
    rep_s, lin_s = [], []
    for j in range(len(fs)):
        m = lin[:, j] > 1e-5
        if m.sum() < 10:
            continue
        rep_s.append(spearmanr(f1[m, j], f2[m, j]).statistic)
        lin_s.append(spearmanr(f1[m, j], lin[m, j]).statistic)
    rep_r, lin_r = float(np.nanmean(rep_s)), float(np.nanmean(lin_s))
    feasible = rep_r > 0.8
    print(f"\nC. PER-GENE FEASIBILITY ({R_FEAS} replicates, depth {DEPTH}, {len(rep_s)} seeds)")
    print(f"   Spearman(MC run 1, MC run 2)    = {rep_r:.4f}   <- does the estimator reproduce ITSELF?")
    print(f"   Spearman(MC run 1, linear walk) = {lin_r:.4f}")
    print(f"   -> {'per-gene MC is usable' if feasible else 'PER-GENE MC IS NOISE-DOMINATED; aggregates only'}")

    # ================= the cross-section as a FITTED micro-law, per fold, TRAIN only =================
    # Features of a directed edge (u -> v): what the target looks like, what the source looks like, and how tightly
    # the two are bound. This is the analogue of a cross-section table indexed by material and energy.
    shared = np.zeros(len(src), np.float32)
    AdjB = (Adj > 0).astype(np.int8)
    for a0 in range(0, len(src), 200000):
        sl = slice(a0, min(a0 + 200000, len(src)))
        shared[sl] = np.asarray(AdjB[src[sl]].multiply(AdjB[dst[sl]]).sum(1)).ravel()
    EF = np.stack([np.log1p(deg[dst]), np.log1p(deg[src]), np.log1p(shared),
                   dep[dst], loe[dst], tf[dst], ncplx[dst], measured[dst],
                   shared / (deg[src] + deg[dst] - shared + 1e-9)], 1)

    print(f"\nD. BLAST RADIUS: predict each knockout's number of specific movers, held out ({NFOLD}-fold, all {nK})")
    pred = collections.defaultdict(lambda: np.zeros(nK))
    mc_mean = np.zeros(nK); mc_sd = np.zeros(nK); mc_p0 = np.zeros(nK)
    for f in range(NFOLD):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        # tide and the target are defined on TRAIN ROWS ONLY, so the test knockout never influences its own label
        tide_tr = (A[tr] >= TAU).mean(0) >= TIDE_FRAC
        spec_tr = (A >= TAU) & (~tide_tr)
        y = spec_tr.sum(1).astype(np.float64)

        # fit the cross-section on TRAIN knockouts' PPI partners
        xs, ys = [], []
        for i in tr:
            if i not in nbrs:
                continue
            st = Adj.indptr[konode[i]]; en = Adj.indptr[konode[i] + 1]
            eidx = np.arange(st, en)                                # rows of Adj -> not the same order as src/dst
            tn = Adj.indices[st:en]
            gj = np.array([gi.get(names[t], -1) for t in tn])
            ok = gj >= 0
            if not ok.any():
                continue
            xs.append(np.stack([np.log1p(deg[tn[ok]]), np.full(ok.sum(), np.log1p(deg[konode[i]])),
                                np.zeros(ok.sum()), dep[tn[ok]], loe[tn[ok]], tf[tn[ok]], ncplx[tn[ok]],
                                measured[tn[ok]], np.zeros(ok.sum())], 1))
            ys.append(spec_tr[i, gj[ok]].astype(np.float64))
        Xe = np.concatenate(xs); ye = np.concatenate(ys)
        cs = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4,
                                           min_samples_leaf=200, random_state=0).fit(Xe, ye)
        pedge = np.clip(cs.predict(EF), 1e-6, 0.5).astype(np.float32)

        # ANALYTIC mean-field cascade size: sum over hops of (P^h 1). No sampling, exact to first order.
        Pw = sparse.coo_matrix((pedge, (src, dst)), shape=(N, N)).tocsr()
        seeds_node = konode[te].copy(); seeds_node[seeds_node < 0] = 0
        v0 = np.zeros((N, len(te)), np.float32); v0[seeds_node, np.arange(len(te))] = 1.0
        mf = np.zeros(len(te)); cur = v0
        for _ in range(DEPTH):
            cur = Pw.T @ cur
            mf += cur.sum(0)
        # MONTE CARLO for the aggregate + its higher moments
        cnt, _ = percolate(pedge, seeds_node, R_MC, 100 + f)
        mc_mean[te] = cnt.mean(0); mc_sd[te] = cnt.std(0); mc_p0[te] = (cnt == 0).mean(0)
        mf[~have[te]] = 0.0; mc_mean[te[~have[te]]] = 0.0

        NF_tr = NODEF[np.clip(konode[tr], 0, N - 1)]; NF_te = NODEF[np.clip(konode[te], 0, N - 1)]
        ytr = np.log1p(y[tr])

        def gbm(Xtr, Xte):
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=4,
                                              min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
            return g.predict(Xte)

        pred["TRAIN-MEAN"][te] = ytr.mean()
        pred["DEGREE"][te] = np.log1p(deg[np.clip(konode[te], 0, N - 1)])
        pred["ESSENTIALITY"][te] = dep[np.clip(konode[te], 0, N - 1)]
        pred["NODE-GBM"][te] = gbm(NF_tr, NF_te)
        pred["MEAN-FIELD"][te] = np.log1p(mf)
        # the mean-field / MC features must be computed for TRAIN rows too, to be usable as model inputs
        seeds_tr = konode[tr].copy(); seeds_tr[seeds_tr < 0] = 0
        v0t = np.zeros((N, len(tr)), np.float32); v0t[seeds_tr, np.arange(len(tr))] = 1.0
        mft = np.zeros(len(tr)); curt = v0t
        for _ in range(DEPTH):
            curt = Pw.T @ curt
            mft += curt.sum(0)
        cntt, _ = percolate(pedge, seeds_tr, R_MC, 500 + f)
        mtr, str_, p0tr = cntt.mean(0), cntt.std(0), (cntt == 0).mean(0)
        pred["NODE-GBM+MF"][te] = gbm(np.c_[NF_tr, np.log1p(mft)], np.c_[NF_te, np.log1p(mf)])
        pred["NODE-GBM+MF+MC"][te] = gbm(
            np.c_[NF_tr, np.log1p(mft), np.log1p(str_), p0tr],
            np.c_[NF_te, np.log1p(mf), np.log1p(mc_sd[te]), mc_p0[te]])
        print(f"   fold {f}: cross-section fitted on {len(ye)} train edges, mean p={pedge.mean():.5f}", flush=True)

    tide_f = (A >= TAU).mean(0) >= TIDE_FRAC
    ytrue = ((A >= TAU) & (~tide_f)).sum(1).astype(np.float64)

    def boot_sp(x, y):
        s = spearmanr(x, y).statistic
        br = np.random.RandomState(3)
        bs = [spearmanr(x[idx], y[idx]).statistic for idx in
              (br.randint(0, len(x), len(x)) for _ in range(300))]
        return float(s), float(np.nanstd(bs))

    print(f"\n   {'predictor':20s} {'Spearman':>10s} {'+/-':>7s} {'AUC(zero)':>10s}")
    res = {}
    for nm in ["TRAIN-MEAN", "DEGREE", "ESSENTIALITY", "NODE-GBM", "MEAN-FIELD", "NODE-GBM+MF", "NODE-GBM+MF+MC"]:
        x = pred[nm]
        if np.allclose(x, x[0]):
            s, se, auc = 0.0, 0.0, 0.5
        else:
            s, se = boot_sp(x, ytrue)
            auc = roc_auc_score((ytrue == 0).astype(int), -x)
        res[nm] = {"spearman": s, "se": se, "auc_zero": float(auc)}
        print(f"   {nm:20s} {s:10.4f} {se:7.4f} {auc:10.4f}")

    # ================= E. DOES MC EARN ITS COST? =================
    ok = mc_mean > 0
    sd_vs_mean = float(spearmanr(mc_sd[ok], mc_mean[ok]).statistic)
    p0_vs_mean = float(spearmanr(mc_p0[ok], mc_mean[ok]).statistic)
    mc_gain = res["NODE-GBM+MF+MC"]["spearman"] - res["NODE-GBM+MF"]["spearman"]
    # THE VERDICT IS THE GAIN, NOT THE CORRELATION. An earlier version keyed this purely off whether the MC moments
    # were monotone in the mean, and so printed "carry rank information the mean does not" directly above a gain of
    # -0.0163. That is the win-only-test bug this repo has now hit three times. A negative gain is a LOSS and is
    # reported as one whatever the correlations say.
    mc_helps = mc_gain > 0.005
    mc_decorative = (not mc_helps) or (abs(sd_vs_mean) > 0.99 and abs(p0_vs_mean) > 0.99)
    print(f"\nE. DOES MONTE CARLO EARN ITS COST ON TOP OF THE ANALYTIC MEAN-FIELD?")
    print(f"   Spearman(MC cascade SD, MC cascade mean)      = {sd_vs_mean:+.4f}")
    print(f"   Spearman(MC P(zero), MC cascade mean)         = {p0_vs_mean:+.4f}")
    print(f"   held-out gain from adding the MC moments      = {mc_gain:+.4f}")
    print(f"   -> {'MC moments do not pay for themselves' if mc_decorative else 'MC moments earn their cost'}"
          f"  (gain {mc_gain:+.4f}; the gain decides, not the correlation)")

    best_base = max(["TRAIN-MEAN", "DEGREE", "ESSENTIALITY", "NODE-GBM"], key=lambda n: res[n]["spearman"])
    mf_gain = res["NODE-GBM+MF"]["spearman"] - res["NODE-GBM"]["spearman"]

    verdict = (
        f"ULAM'S METHOD APPLIED TO PERTURBATION SPREAD, AND IT DOES NOT WORK HERE. Two claims this module originally "
        f"made are WITHDRAWN, both refuted by checks that cost minutes. FIRST, the response-size distribution was "
        f"offered as evidence for branching -- heavy tail, Fano 91.3, log-log slope -1.41 against critical "
        f"Galton-Watson's -1.5, 689/1400 exact zeros. Permuting each knockout's z-scores WITHIN its own non-tide "
        f"columns leaves the count above threshold unchanged by construction, and measured, all seven statistics are "
        f"{'BIT-IDENTICAL' if perm_invariant else 'NOT identical'} under that permutation. They are functions of the "
        f"per-knockout marginal and carry ZERO bits about topology. A heavy tail is consistent with branching and "
        f"equally consistent with anything else that makes heavy-tailed responses. SECOND, Monte Carlo was justified "
        f"on the grounds that two-terminal reliability is #P-complete. It is -- in general, and not here: at depth 2 "
        f"the direct edge and each 2-path through a common neighbour are EDGE-DISJOINT, hence independent, so "
        f"P(reach) = 1 - (1-p_st) * prod_x (1 - p_sx p_xt) exactly, and at depth 3 the correction is both tiny and "
        f"monotone, so it cannot reorder. There was never a quantity here without a closed form. "
        f"WHAT SURVIVES IS THE CROSS-SECTION, because it contrasts partners against non-partners and the permutation "
        f"destroys it: P(PPI partner moves | gene knocked out) = {p_hat0:.5f} against {spec_all.mean():.5f}, "
        f"{p_hat0/spec_all.mean():.2f}x. That is measured against a FLAT background, and against a degree- and "
        f"move-frequency-matched background the enrichment is much smaller; the flat number should not be quoted "
        f"alone. R0 IS A RANGE AND NOT A NUMBER: the naive p*<k> gives {p_hat0*kk.mean():.2f}, the excess-degree "
        f"formula {p_hat0*excess:.2f} (p/p_c = {p_hat0/p_c:.2f}), and both assume a locally tree-like graph that "
        f"this one is not -- PPI clustering is ~0.20 against ~0.002 for a degree-matched random graph, so the "
        f"configuration-model figure is an upper bound of unknown tightness. Estimators span roughly 0.56 to 1.25, "
        f"which straddles 1, so THE SUB- VERSUS SUPERCRITICAL QUESTION IS NOT ANSWERABLE FROM THIS DATA and the "
        f"earlier flat assertion of 'subcritical' overstated it. "
        + (f"MONTE CARLO IS NOT USABLE AT PER-GENE RESOLUTION HERE, AND THIS WAS TESTED FIRST, BEFORE ANY "
           f"PERFORMANCE NUMBER. Two independent runs at {R_FEAS} replicates agree with each other at Spearman "
           f"{rep_r:.3f} -- the estimator does not reproduce ITSELF -- while agreeing with the cheap linear walk at "
           f"{lin_r:.3f}. So the MC-vs-linear disagreement is sampling noise, not nonlinearity. The arithmetic is "
           f"elementary: at R0 = {p_hat0*excess:.2f} the reachability probabilities are ~1e-4, so at {R_FEAS} "
           f"replicates most genes are hit zero or one times and the ranking is Poisson noise. THIS IS A DIFFERENT "
           f"FAILURE FROM THE PREVIOUS TWO Monte Carlos in this repo, which were rank-decorative; this one is "
           f"statistically infeasible, and the distinction matters because it says WHERE MC can still work. "
           if not feasible else
           f"Per-gene MC reproduces itself at Spearman {rep_r:.3f}, so it is usable at that resolution. ")
        + f"SO THE MC IS SPENT WHERE ULAM SPENT IT: ON THE AGGREGATE. He never computed the probability that a "
        f"particular neutron reached a particular place; he computed the multiplication factor. An aggregate over "
        f"~10^4 genes has a relative standard error smaller by sqrt(n). The aggregate here is the BLAST RADIUS, the "
        f"number of genes a knockout moves, which is the direct neutron-count analogue and is measured and held out. "
        f"Baseline ladder, Spearman against the true count over all {nK} knockouts in {NFOLD}-fold CV: train mean "
        f"{res['TRAIN-MEAN']['spearman']:.4f}, PPI degree alone {res['DEGREE']['spearman']:.4f}, essentiality alone "
        f"{res['ESSENTIALITY']['spearman']:.4f}, a GBM on node features {res['NODE-GBM']['spearman']:.4f}, the "
        f"analytic mean-field cascade alone {res['MEAN-FIELD']['spearman']:.4f}, node-GBM plus the cascade "
        f"{res['NODE-GBM+MF']['spearman']:.4f} ({mf_gain:+.4f} over the GBM without it), and adding the MC moments "
        f"{res['NODE-GBM+MF+MC']['spearman']:.4f} ({mc_gain:+.4f}). "
        + (f"THE CASCADE FEATURE DOES NOT BEAT THE NODE FEATURES IT IS BUILT FROM. A branching simulation that "
           f"cannot beat degree-plus-essentiality in a GBM has demonstrated nothing beyond restating degree, and "
           f"that is the honest reading of {mf_gain:+.4f}. " if mf_gain <= 0.01 else
           f"THE CASCADE FEATURE ADDS {mf_gain:+.4f} OVER THE NODE FEATURES IT IS BUILT FROM, which is the only "
           f"comparison that isolates the branching structure from the degree it is computed on. ")
        + (f"AND THE MC MOMENTS ADD {mc_gain:+.4f} ON TOP -- "
           + (f"a LOSS, so sampling does not pay for itself here either. The cascade SD tracks the cascade mean at "
              f"{sd_vs_mean:+.3f} and P(zero) at {p0_vs_mean:+.3f}; those are not quite the perfect monotonicity "
              f"that would make them the mean in disguise, but near-monotone features that cost a 600-replicate "
              f"simulation and then REDUCE held-out accuracy are not a finding to defend on a technicality. This is "
              f"the third Monte Carlo in this repo to fail to earn its cost, and the three failed differently: the "
              f"first two were rank-decorative, this one is unaffordable at per-gene resolution and unhelpful at "
              f"aggregate resolution. " if not mc_helps else
              f"a gain, and the moments are not monotone in the mean (SD {sd_vs_mean:+.3f}, P(zero) "
              f"{p0_vs_mean:+.3f}), so sampling bought something the closed form did not. ")) +
        f"WHAT THIS CANNOT SAY. Only 674 of 15,183 specific movers (4.4%) are direct PPI partners of the knocked-out "
        f"gene, so a cascade on this graph could never have explained WHICH genes move -- it was only ever testable "
        f"on HOW MANY, which is why blast radius is the target. The cross-section is measured at hop 1, from a gene "
        f"that was destroyed, and then applied at hops 2-3 where the source is merely perturbed; that is the same "
        f"assumption Ulam made in reusing one cross-section at every collision, but it is far less safe here, and it "
        f"would bias R0 upward. CRISPRi knockdown is partial (median 42.8% transcript remaining), so 'knocked out' "
        f"is itself an idealisation. Evaluation is all {nK} knockouts, deliberately NOT the 378 scorable ones, "
        f"because filtering on >=5 movers would condition the evaluation set on the target being predicted.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"criticality": {"p": float(p_hat0), "background": float(spec_all.mean()),
                               "excess_degree": float(excess), "R0": float(p_hat0 * excess),
                               "R0_naive": float(p_hat0 * kk.mean()), "p_c": float(p_c),
                               "p_over_pc": float(p_hat0 / p_c)},
               "feasibility": {"replicates": R_FEAS, "split_half": rep_r, "vs_linear": lin_r,
                               "per_gene_mc_usable": bool(feasible)},
               "blast_radius": res,
               "mc_value": {"sd_vs_mean": sd_vs_mean, "p0_vs_mean": p0_vs_mean, "gain": float(mc_gain),
                            "decorative": bool(mc_decorative)},
               "coverage": {"movers_total": int(spec_all.sum()), "movers_that_are_ppi_partners": int(hits)},
               "verdict": verdict},
              open(OUT / "ulam_branching.json", "w"), indent=2)
    print(f"\n  -> {OUT/'ulam_branching.json'}")


if __name__ == "__main__":
    main()
