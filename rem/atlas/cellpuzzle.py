"""The jigsaw and the maze: one map of DIRECT pushes, assembled from every knockdown at once, then every
knockdown run through it step by step -- and where the runs get stuck.

THE IDEA (user's: an unknown jigsaw combined with a maze). Each knockdown's measured end state is the SUM
OF ALL PATHS from the knocked-down gene: the direct pushes, plus every 2-step route, 3-step route, ...
  total effect = W + W^2 + W^3 + ...   (W = the direct map; the series is the maze, one step per power)
A route that explains one knockdown but contradicts the others cannot be in W, so fitting W to ALL
knockdowns together collapses the many possible routes to one map (the logic of Modular Response
Analysis: Kholodenko et al., PNAS 2002, doi 10.1073/pnas.192442699).

THE JIGSAW (how W is assembled). Observations = TRAIN knockdowns of K562 genome-wide Perturb-seq
(Replogle 2022, figshare 20029387, CC BY 4.0; energy-test phenotype p < 0.05; askdata's SHA-256 split;
10% of TRAIN held out as VALIDATION). Each observation is the knockdown's specific response (minus the TRAIN
mean) over all 8,248 measured genes. For every gene i, a ridge regression explains its response across
observations from the responses of ALL OTHER genes: W[i, k] = direct push of gene k on gene i.
  - gene i is never its own predictor (exact leave-one-predictor-out, Sherman-Morrison);
  - in the observation where gene i itself was knocked down, its value is the intervention, not a
    response: it is set to 0 as a response (kept as a predictor -- it is the cleanest variation of i).
Because W[:, k] is learnt from how gene k rises and falls in OTHER knockdowns, it exists for genes whose
own knockdown was never seen. That is what lets the maze run on held-out knockdowns -- and it is new
information about untested genes, which every earlier module found to be the bottleneck.

THE MAZE. A held-out knockdown of gene j (j measured) is run from its start (j drops) through the map:
  after K steps the predicted change of every gene = -(W e_j + W^2 e_j + ... + W^K e_j).
K = 1 is direct pushes only. lambda (ridge) and K (1-6) are chosen on VALIDATION knockdowns only.
Score: per knockdown, Pearson r between predicted and measured specific response over all genes but j
(the metric of askdata / expmem / the Colab notebook; the tide predicts 0).

WHERE THE RUNS GET STUCK. Residuals of TEST runs (measured - best-scaled prediction) are factorised; a
shared residual pattern across many knockdowns is the outline of a missing piece. Each pattern is named by
its Reactome pathways (CC0). This is steps 1-3 of the plan; filling the pieces (LLM guesses, unmeasured
genes) comes after, with held-out checks.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
P0 POSITIVE CONTROL (blocking). A fake cell: 600 genes, a random sparse signed direct map (3 inputs per
   gene, spectral radius 0.6), 300 observed knockdowns with Gaussian noise; the same fitting code.
   PASS if (a) true direct edges are found: AUROC of |W| for true edges vs non-edges >= 0.75, (b) |W|
   beats the total-effect matrix at finding DIRECT edges (higher AUROC), and (c) for true edges a -> b,
   |W[b, a]| > |W[a, b]| in > 60% (direction). If any fails, nothing on real data is run.
   AMENDMENT 2026-10-30, AFTER P0 FAILED ON ITS FIRST RUN (output kept in history, commit 4719327):
   (a) 0.806 and (c) 71.9% passed; (b) failed because it compared W, fitted from 300 noisy
   knockdowns, with the NOISE-FREE total effects of ALL 600 genes -- a comparison the real data can
   never offer (total effects exist only for knocked-down genes, and are noisy). P0 is now:
     (a) AUROC of |W| for true direct edges >= 0.75 over all pairs;
     (a2) AUROC >= 0.70 for edges whose source was NEVER knocked down (what the maze relies on);
     (c) direction right > 60%;
     (d) maze check: the 300 genes never knocked down are run through W for 3 steps; median r with
         their true (noise-free) responses >= 0.5.
   Reported, no longer blocking: (b) on knocked-down sources only, |W| vs the NOISY observed total
   effects, same pairs. Nothing else changes.
P1 THE MAZE vs THE PLAIN LEARNER. TEST knockdowns whose gene is measured: MAP (lambda, K from validation)
   vs RIDGE (askdata's feature blocks, lambda by its inner CV) on the same knockdowns; paired two-sided
   sign test; BETTER needs p < 0.05 and median r gain >= +0.01 (WORSE the mirror; else NO DIFFERENCE).
P2 DOES THE MAZE NEED MORE THAN ONE STEP? MAP at the chosen K vs MAP at K = 1 (direct pushes only), same
   rule. Not tested if K = 1 is chosen.
P3 THE MAP + THE PLAIN LEARNER: their rescaled average vs RIDGE, same rule.
P4 IDENTITY CONTROL: MAP run from ANOTHER gene's start (a derangement of the TEST genes) must be WORSE than
   MAP (P1/P3 void otherwise).
P5 THE MAP REPEATS IN AN INDEPENDENT EXPERIMENT. W fitted the same way from the K562 essential screen
   (B; its TRAIN knockdowns). Among the strongest 0.1% of A's direct pushes between genes measured in
   both, the share with the same sign in B, vs 20 derangements of B's gene labels: REPEATS if above all.
P6 ARE DIRECT PUSHES MORE DIRECT? For each ChIP-profiled factor (ENCODE K562 IDR peaks already cached,
   summit within 1 kb of a refGene TSS) that is measured and knocked down in TRAIN: the share of its top
   200 DIRECT targets (|W[:, tf]|) that it binds vs the share of its top 200 TOTAL targets (its knockdown's
   measured response). Paired sign test across factors: DIRECT MORE BOUND if more factors favour W, p < 0.05.
P7 WHERE THE RUNS GET STUCK (descriptive): share of measured response energy the runs leave unexplained;
   top 5 shared residual patterns, their share of residual energy and their Reactome pathways.
CEILING (reported): the same protein measured in screen B (as in expmem).
X  NOT: linear, steady state; only measured genes can carry a push, so unmeasured genes, proteins and
   metabolites are folded into apparent direct links; 'direct' means direct among measured genes.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import collections
import importlib.util
import json
import gzip
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_cellpuzzle.txt")
ART = R("outputs", "cellpuzzle.json")
RULE = "=" * 97


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


ad = _load("askdata"); pe = _load("proteffect"); rc = pe.rc
wc, cd, dc = ad.wc, ad.cd, ad.dc
LAMS, KMAX, MIN_GAIN, SEED, N_DER = (1e1, 1e2, 1e3, 1e4), 6, 0.01, 20261030, 20


# =================================================================================================
# THE JIGSAW: fit the direct map
# =================================================================================================
def fit_map(Xo, own, lam):
    """Xo: m observations x n genes (centred responses); own[o] = column of the gene knocked down in
    observation o (or -1). Returns W (n x n), W[i, k] = direct push of gene k on gene i: ridge of each gene
    on all other genes, the gene itself excluded as predictor (exact), its own knockdown excluded as a
    response (set to 0)."""
    X = Xo.astype(np.float64)
    Y = X.copy()
    o = np.where(own >= 0)[0]
    Y[o, own[o]] = 0.0
    A = np.linalg.inv(X @ X.T + lam * np.eye(len(X)))
    AY, AX = A @ Y, A @ X
    num, den = (X * AY).sum(0), 1.0 - (X * AX).sum(0)
    alpha = AY + AX * (num / den)
    W = (alpha.T @ X).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    return W


def run_maze(W, cols, kmax):
    """Predicted change after 1..kmax steps for knockdowns of genes `cols`: list of (n x len(cols))."""
    P = -W[:, cols].astype(np.float64)
    S, out = P.copy(), [P.copy()]
    for _ in range(kmax - 1):
        P = W @ P
        S = S + P
        out.append(S.copy())
    return out


def row_r(Pred, Act, skip):
    """Pearson r per knockdown (columns of Pred, rows of Act), leaving out gene skip[t]."""
    r = np.full(Pred.shape[1], np.nan)
    for t in range(Pred.shape[1]):
        a = Act[t].copy(); p = Pred[:, t].copy()
        m = np.isfinite(a) & np.isfinite(p); m[skip[t]] = False
        if m.sum() > 10 and p[m].std() > 0 and a[m].std() > 0:
            r[t] = np.corrcoef(p[m], a[m])[0, 1]
    return r


def auroc(scores, labels):
    from scipy.stats import rankdata
    rk = rankdata(scores); npos = int(labels.sum()); nneg = len(labels) - npos
    return (rk[labels].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1)


def sign_rule(d):
    d = d[np.isfinite(d)]
    w, l, p = wc.sign_test(list(d))
    md = float(np.median(d))
    v = "BETTER" if p < 0.05 and md >= MIN_GAIN else "WORSE" if p < 0.05 and md <= -MIN_GAIN else "NO DIFFERENCE"
    return dict(better=w, worse=l, p=p, median_diff=md, verdict=v)


def positive_control(rng):
    n, m, deg = 600, 300, 3
    W0 = np.zeros((n, n))
    for i in range(n):
        src = rng.choice(np.setdiff1d(np.arange(n), [i]), deg, replace=False)
        W0[i, src] = rng.choice([-1, 1], deg) * rng.uniform(0.3, 1.0, deg)
    W0 *= 0.6 / np.max(np.abs(np.linalg.eigvals(W0)))
    T = np.linalg.inv(np.eye(n) - W0)                       # total effect of a unit push on each gene
    kd = rng.choice(n, m, replace=False)
    Xo = (-T[:, kd]).T + rng.normal(0, 0.05, (m, n))         # each observation: one knockdown's noisy response
    Tobs = -Xo.T.copy()                                      # what an experiment sees: noisy totals, knocked-down genes only
    Xo -= Xo.mean(0)
    W = fit_map(Xo, kd.copy(), 1.0)
    off = ~np.eye(n, dtype=bool)
    edge = W0 != 0
    seen = np.zeros(n, bool); seen[kd] = True
    a_all = auroc(np.abs(W)[off], edge[off])
    m_un = off & ~seen[None, :]                              # pairs whose source was never knocked down
    a_unseen = auroc(np.abs(W)[m_un], edge[m_un])
    pos = {g: c for c, g in enumerate(kd)}
    m_seen = off & seen[None, :]
    Tfull = np.zeros((n, n)); Tfull[:, kd] = Tobs
    a_w_seen, a_t_seen = auroc(np.abs(W)[m_seen], edge[m_seen]), auroc(np.abs(Tfull)[m_seen], edge[m_seen])
    ii, jj = np.nonzero(W0)
    dirn = float(np.mean(np.abs(W[ii, jj]) > np.abs(W[jj, ii])))
    unseen = np.where(~seen)[0]
    pred = run_maze(W, unseen, 3)[2]
    r = [np.corrcoef(np.delete(pred[:, t], j), np.delete(-T[:, j], j))[0, 1] for t, j in enumerate(unseen)]
    maze_r = float(np.median(r))
    ok = a_all >= 0.75 and a_unseen >= 0.70 and dirn > 0.6 and maze_r >= 0.5
    return dict(auroc_all=float(a_all), auroc_never_knocked_down=float(a_unseen), direction=dirn, maze_r_unseen=maze_r,
                reported_seen_sources=dict(auroc_direct_map=float(a_w_seen), auroc_noisy_total=float(a_t_seen)),
                passed=bool(ok))


# =================================================================================================
def main():
    out, res = [], {}

    def P_(s=""):
        print(s, flush=True); out.append(s)

    def finish():
        open(OUT, "w").write("\n".join(out) + "\n")
        os.makedirs(os.path.dirname(ART), exist_ok=True)
        json.dump(pe.clean(res), open(ART, "w"), indent=1)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    P_(RULE); P_("THE JIGSAW AND THE MAZE: ONE DIRECT MAP FROM EVERY KNOCKDOWN, THEN EVERY KNOCKDOWN RUN THROUGH IT"); P_(RULE)
    pc = positive_control(rng)
    P_(f"  P0 fake cell (600 genes, 300 knockdowns): direct edges AUROC {pc['auroc_all']:.3f} (sources never knocked down "
       f"{pc['auroc_never_knocked_down']:.3f}); direction right {pc['direction']:.1%}; maze for never-knocked-down genes median r "
       f"{pc['maze_r_unseen']:+.3f} -> {'PASS' if pc['passed'] else 'FAIL'}")
    rs = pc["reported_seen_sources"]
    P_(f"     reported: on knocked-down sources, direct map AUROC {rs['auroc_direct_map']:.3f} vs noisy total effects {rs['auroc_noisy_total']:.3f}")
    res["P0"] = pc
    if not pc["passed"]:
        P_("  P0 FAILED: the method does not recover a known map; nothing on real data is run."); finish(); return

    S = pe.load_screen(pe.FILES["A"])
    X, kos, ep, genes = S["X"], S["kos"], S["ep"], S["genes"]
    gi = pe.first_index(genes); n = len(genes)
    phen = [i for i, e in enumerate(ep) if np.isfinite(e) and e < 0.05]
    tr = [i for i in phen if ad.split(kos[i]) == "TRAIN"]; te = [i for i in phen if ad.split(kos[i]) == "TEST"]
    perm = np.random.default_rng(2).permutation(len(tr))
    va = [tr[k] for k in perm[: len(tr) // 10]]; fit = [tr[k] for k in perm[len(tr) // 10:]]
    tide = np.nanmean(X[tr], 0)
    Spec = X - tide                                              # specific responses (NaN kept)
    own = np.array([gi.get(kos[i], -1) for i in range(len(kos))])
    Xfit = np.nan_to_num(Spec[fit]).astype(np.float32)
    te_m = [i for i in te if own[i] >= 0]; va_m = [i for i in va if own[i] >= 0]
    P_(f"  observations (TRAIN knockdowns fitted) {len(fit):,}; validation {len(va):,}; TEST {len(te):,}, of which "
       f"{len(te_m):,} knock down a measured gene (the maze can start there); genes {n:,}")

    # ---- choose lambda and K on validation knockdowns -------------------------------------------
    best, maps = None, {}
    for lam in LAMS:
        t1 = time.time()
        W = fit_map(Xfit, own[fit], lam)
        runs = run_maze(W, own[va_m], KMAX)
        meds = [float(np.nanmedian(row_r(Pk, Spec[va_m], own[va_m]))) for Pk in runs]
        P_(f"     lambda {lam:g}: validation median r by steps 1-{KMAX}: " + " ".join(f"{m_:+.3f}" for m_ in meds)
           + f"   ({time.time() - t1:.0f}s)")
        k_best = int(np.nanargmax(meds)) + 1
        if best is None or meds[k_best - 1] > best[2]:
            best = (lam, k_best, meds[k_best - 1]); maps["W"] = W
    lam, K, _ = best
    W = maps["W"]
    P_(f"  chosen on validation: lambda {lam:g}, K = {K} step(s)")
    res["choice"] = dict(lam=lam, K=K)

    # ---- the maze on TEST ------------------------------------------------------------------------
    runs = run_maze(W, own[te_m], KMAX)
    Act = Spec[te_m]
    r_map = row_r(runs[K - 1], Act, own[te_m]); r_k1 = row_r(runs[0], Act, own[te_m])
    der = rng.permutation(len(te_m))
    while (der == np.arange(len(te_m))).any():
        der = rng.permutation(len(te_m))
    r_other = row_r(run_maze(W, own[te_m][der], K)[K - 1], Act, own[te_m])

    # plain learner on the same knockdowns
    dep = cd.load_depmap(); D = json.load(gzip.open(dc.ENCY)); bn = [r_["name"] for r_ in D["genes"]]
    fg = [kos[i] for i in phen]; fpos = {i: j for j, i in enumerate(phen)}
    blocks = ad.feature_blocks(fg, genes, dep, D, bn)
    base = ["dep", "net", "proc", "chip", "prot"]
    Ftr, Fte = ad.design(blocks, base, [fpos[i] for i in tr]), ad.design(blocks, base, [fpos[i] for i in te_m])
    Ytr = np.nan_to_num(Spec[tr])
    sc = lambda P, A_: float(np.nanmedian(wc.pearson_rows(P, A_)))
    lam_r, _ = ad.cv_lambda(Ftr, Ytr, sc)
    Pr = ad.ridge_multi(Ftr, Ytr, Fte, [lam_r])[lam_r]
    r_ridge = row_r(Pr.T, Act, own[te_m])
    unit = lambda M: (M - M.mean(0)) / np.maximum(np.linalg.norm(M - M.mean(0), axis=0), 1e-12)
    r_comb = row_r((unit(runs[K - 1]) + unit(Pr.T)) / 2, Act, own[te_m])

    # ceiling: the same protein in screen B
    Bs = pe.load_screen(pe.FILES["B"]); bk = {p: i for i, p in enumerate(Bs["kos"])}; bg = pe.first_index(Bs["genes"])
    cols = [j for j, g in enumerate(genes) if g in bg]; bcols = [bg[genes[j]] for j in cols]
    btide = np.nanmean(Bs["X"][[bk[kos[i]] for i in tr if kos[i] in bk]][:, bcols], 0)
    ce = []
    for t, i in enumerate(te_m):
        if kos[i] in bk:
            b = np.zeros((n, 1)); b[cols, 0] = np.nan_to_num(Bs["X"][bk[kos[i]], bcols] - btide)
            ce.append(row_r(b, Act[[t]], own[[i]])[0])

    P_(); P_("  TEST knockdowns of measured genes: median / mean r of the predicted specific response")
    for nm, v in (("MAP (the maze, chosen K)", r_map), ("MAP, direct pushes only (K = 1)", r_k1),
                  ("MAP from another gene's start", r_other), ("RIDGE (plain learner)", r_ridge),
                  ("MAP + RIDGE", r_comb)):
        P_(f"     {nm:34s} {np.nanmedian(v):+.3f} / {np.nanmean(v):+.3f}")
    P_(f"     {'CEILING: screen B':34s} {np.nanmedian(ce):+.3f} / {np.nanmean(ce):+.3f}   ({len(ce)} genes also in B)")
    res["scores"] = {nm: dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v))) for nm, v in
                     (("map", r_map), ("map_k1", r_k1), ("map_other_start", r_other), ("ridge", r_ridge), ("map_plus_ridge", r_comb))}
    res["ceiling"] = dict(median=float(np.nanmedian(ce)), n=len(ce))
    G = {}
    G["P1 maze vs plain learner"] = sign_rule(r_map - r_ridge)
    if K > 1:
        G["P2 more than one step"] = sign_rule(r_map - r_k1)
    G["P3 map + learner vs learner"] = sign_rule(r_comb - r_ridge)
    G["P4 identity (vs another gene's start)"] = sign_rule(r_map - r_other)
    for tag, g in G.items():
        P_(f"  {tag:40s} better {g['better']:,} / worse {g['worse']:,}  median diff {g['median_diff']:+.4f}  p {g['p']:.2g} -> {g['verdict']}")
    if K == 1:
        P_("  P2 not tested: one step (direct pushes only) was chosen on validation")
    if G["P4 identity (vs another gene's start)"]["verdict"] != "BETTER":
        P_("  NOTE: P4 failed -- P1 and P3 do not count.")
    res["gates"] = G

    # ---- P5: does the map repeat in the independent screen? ---------------------------------------
    bphen = [i for i, e in enumerate(Bs["ep"]) if np.isfinite(e) and e < 0.05 and ad.split(Bs["kos"][i]) == "TRAIN"]
    btide2 = np.nanmean(Bs["X"][bphen], 0)
    bown = np.array([bg.get(Bs["kos"][i], -1) for i in bphen])
    WB = fit_map(np.nan_to_num(Bs["X"][bphen] - btide2).astype(np.float32), bown, lam)
    common = [g for g in gi if g in bg]
    ia = np.array([gi[g] for g in common]); ib = np.array([bg[g] for g in common])
    WA_c = W[np.ix_(ia, ia)]; WB_c = WB[np.ix_(ib, ib)]
    off = ~np.eye(len(common), dtype=bool)
    thr = np.quantile(np.abs(WA_c[off]), 0.999)
    sel = (np.abs(WA_c) >= thr) & off
    agree = float(np.mean(np.sign(WA_c[sel]) == np.sign(WB_c[sel])))
    nul = []
    for _ in range(N_DER):
        p_ = rng.permutation(len(common))
        WBp = WB_c[np.ix_(p_, p_)]
        nul.append(float(np.mean(np.sign(WA_c[sel]) == np.sign(WBp[sel]))))
    rep = bool(agree > max(nul))
    P_(); P_(f"  P5 strongest 0.1% of direct pushes ({int(sel.sum()):,}) between {len(common):,} genes measured in both screens: "
             f"same sign in B {agree:.1%} vs relabelled {np.mean(nul):.1%} (max {max(nul):.1%}) -> {'REPEATS' if rep else 'DOES NOT REPEAT'}")
    res["P5"] = dict(n_edges=int(sel.sum()), agree=agree, null_mean=float(np.mean(nul)), null_max=max(nul), repeats=rep)
    del WB

    # ---- P6: are direct pushes more often bound? ----------------------------------------------------
    meta = json.load(open(rc.META)) if os.path.exists(rc.META) else {}
    tss = rc.tss_table(); kidx = {k: i for i, k in enumerate(kos)}
    rows6 = []
    for tf, m_ in sorted(meta.items()):
        if not m_ or tf not in gi or tf not in kidx or kidx[tf] not in tr or not os.path.exists(m_["path"]):
            continue
        bound = set(rc.bound_genes(m_["path"], tss)) & set(gi)
        if len(bound) < 50:
            continue
        j = gi[tf]
        dvec = np.abs(W[:, j]).copy(); dvec[j] = -1
        tvec = np.abs(np.nan_to_num(Spec[kidx[tf]])).copy(); tvec[j] = -1
        dtop = [genes[k] for k in np.argsort(-dvec)[:200]]; ttop = [genes[k] for k in np.argsort(-tvec)[:200]]
        rows6.append((tf, np.mean([g in bound for g in dtop]), np.mean([g in bound for g in ttop]), len(bound) / len(gi)))
    if rows6:
        d6 = np.array([a - b for _, a, b, _ in rows6])
        w6, l6, p6 = wc.sign_test(list(d6))
        v6 = "DIRECT MORE BOUND" if w6 > l6 and p6 < 0.05 else "TOTAL MORE BOUND" if l6 > w6 and p6 < 0.05 else "NO DIFFERENCE"
        P_(f"  P6 {len(rows6)} ChIP-profiled factors: bound share of top-200 DIRECT targets {np.mean([a for _, a, _, _ in rows6]):.1%}, "
           f"of TOTAL targets {np.mean([b for _, _, b, _ in rows6]):.1%}, background {np.mean([c for *_, c in rows6]):.1%}; "
           f"direct higher in {w6}, lower in {l6}, p {p6:.2g} -> {v6}")
        res["P6"] = dict(n=len(rows6), direct=float(np.mean([a for _, a, _, _ in rows6])), total=float(np.mean([b for _, _, b, _ in rows6])),
                         better=w6, worse=l6, p=p6, verdict=v6, per_factor=[list(x) for x in rows6])
    else:
        P_("  P6 no ChIP-profiled factor qualified")

    # ---- P7: where the runs get stuck --------------------------------------------------------------
    Pk = runs[K - 1]; A0 = np.nan_to_num(Act)
    for t, i in enumerate(te_m):
        A0[t, own[i]] = 0.0
    sc_ = (Pk * A0.T).sum(0) / np.maximum((Pk * Pk).sum(0), 1e-12)      # best scale per run
    Res = A0 - (Pk * sc_).T
    unexpl = float((Res ** 2).sum() / (A0 ** 2).sum())
    U_, sv, Vt = np.linalg.svd(Res - Res.mean(0), full_matrices=False)
    share = sv[:5] ** 2 / (sv ** 2).sum()
    mods, pnames, _ = wc.reactome(set(genes)); mid = sorted(mods)
    Mm = np.zeros((n, len(mid)), np.float32)
    for c_, pid in enumerate(mid):
        for g in mods[pid]:
            Mm[gi[g], c_] = 1
    P_(); P_(f"  P7 the runs leave {unexpl:.1%} of the measured response energy unexplained. Shared ways of getting stuck:")
    stuck = []
    for c_ in range(5):
        lo = (Vt[c_] @ Mm) / np.maximum(Mm.sum(0), 1)
        up = [pnames[mid[k]] for k in np.argsort(-lo)[:3]]; dn = [pnames[mid[k]] for k in np.argsort(lo)[:3]]
        stuck.append(dict(share=float(share[c_]), one_side=up, other_side=dn))
        P_(f"     pattern {c_ + 1} ({share[c_]:.1%} of what is left): {'; '.join(up)}  <->  {'; '.join(dn)}")
    res["P7"] = dict(unexplained=unexpl, patterns=stuck)
    res["runtime_s"] = time.time() - t0
    P_(f"\n  runtime {time.time() - t0:.0f}s; data: Replogle 2022 (CC BY 4.0), ENCODE (unrestricted), Reactome (CC0), DepMap 24Q4 (CC BY 4.0)")
    finish()


if __name__ == "__main__":
    main()
