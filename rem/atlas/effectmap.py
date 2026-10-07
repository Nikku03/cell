"""The network of effects: what does each protein do to every pathway -- measured, learnable, and
filled in for proteins nobody has knocked down?

THE MAP. For every K562 knockdown with a phenotype (Replogle gwps, CC BY 4.0), the effect on each of
981 Reactome processes (CC0): the mean response of the process's genes, minus the TRAIN average over
all knockdowns (the tide), so only what is SPECIFIC to that protein remains.
THE CEILING. 102 genes were knocked down twice, by independent guides at two promoters (P1, P2), with
a phenotype both times; how well two independent knockdowns of the SAME protein agree on its specific
pathway effects bounds what any prediction can reach. (A lower bound on true reproducibility: the
two promoters of a gene are not always equally used.)
THE QUESTION, no reasoning supplied (askdata.py's data blocks and learners, chosen by inner CV on
TRAIN): "what is this protein's specific effect on every pathway?" -- for proteins never seen.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
N1 CEILING (descriptive): median Pearson r between the two independent knockdowns' specific
   pathway-effect vectors.
N2 DOES IT LEARN PROTEIN-SPECIFIC EFFECTS? Per TEST protein, r between predicted and measured specific
   effects over the 981 pathways (the tide predicts 0 specific effect: r = 0). One-sample sign test
   of r > 0, p < 0.05. Reported as a fraction of the ceiling.
N3 LEARNER vs the hand-designed process analogy (W2 in pathway space), paired sign test, p < 0.05.
N4 TOP-10 PATHWAYS: overlap of predicted and measured 10 most-affected pathways, against a null that
   shuffles predictions across TEST proteins (200 permutations).
N5 WHICH PARTS OF THE NETWORK ARE LEARNABLE: per pathway, Spearman across TEST proteins between
   predicted and measured effect; Benjamini-Hochberg FDR < 0.05 -> LEARNABLE.
FILL. The learner retrained on ALL measured proteins predicts the specific pathway effects of every
   gene with data that has NO measured knockdown phenotype; only effects in LEARNABLE pathways are kept,
   each with that pathway's TEST Spearman as its reliability. Written to the memory bank beside the
   measured map (top effects per protein).
X  NOT: one cell line, steady state; effects on pathways are averages over their genes.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import collections
import gzip
import importlib.util
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_effectmap.txt")
ART = R("outputs", "effectmap.json")
BANK = R("memory_bank", "cell_v3")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("askdata", os.path.join(HERE, "askdata.py"))
ad = importlib.util.module_from_spec(_s); _s.loader.exec_module(ad)
wc, cd, dc = ad.wc, ad.cd, ad.dc
SCALE = 6.85


def replicate_rows(genes_meas):
    """All perturbations (not collapsed) for genes with >= 2 independent promoter knockdowns."""
    import h5py
    f = h5py.File(os.path.join(cd.CA, "perturbseq", "K562_gwps_normalized_bulk_01.h5ad"), "r")
    gt = [s.decode() for s in f["obs/gene_transcript"][()]]; core = f["obs/core_control"][()]; ep = f["obs/energy_test_p_value"][()]
    by = collections.defaultdict(list)
    for i, g in enumerate(gt):
        p = g.split("_")
        if core[i] or "non-targeting" in g.lower():
            continue
        by[p[1]].append(i)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    both = {k: v for k, v in multi.items() if sum(np.isfinite(ep[i]) and ep[i] < 0.05 for i in v) >= 2}
    idx = sorted({i for v in both.values() for i in v})
    Xr = f["X"][idx].astype(np.float32); Xr[~np.isfinite(Xr)] = np.nan
    pos = {i: r for r, i in enumerate(idx)}
    pairs = {k: [pos[i] for i in v if np.isfinite(ep[i]) and ep[i] < 0.05][:2] for k, v in both.items()}
    return Xr, pairs, len(multi)


def main():
    from scipy.stats import spearmanr
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(20261010)
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    mods, rnames, rtop = wc.reactome(set(genes)); mid = sorted(mods); M = len(mid)
    Mm = np.zeros((len(genes), M), np.float32)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            Mm[gidx[g], c] = 1
    score = lambda A: (np.where(np.isfinite(A), A, 0) @ Mm) / np.maximum(np.isfinite(A).astype(np.float32) @ Mm, 1)
    S = score(X)
    phen = [k for k, e in zip(kos, ep) if np.isfinite(e) and e < 0.05]
    tr_k = [k for k in phen if ad.split(k) == "TRAIN"]; te_k = [k for k in phen if ad.split(k) == "TEST"]
    tideS = S[[kidx[k] for k in tr_k]].mean(0)
    E = S - tideS                                              # specific pathway effects, all knockdowns
    P_(RULE); P_("THE NETWORK OF EFFECTS: EVERY PROTEIN ON EVERY PATHWAY"); P_(RULE)
    P_(f"  measured map: {len(phen):,} proteins with a phenotype x {M} pathways (TRAIN {len(tr_k):,}, TEST {len(te_k):,})")

    # ---- N1 ceiling ------------------------------------------------------------------------------
    Xr, pairs, nmulti = replicate_rows(genes)
    Er = score(Xr) - tideS
    rc = [np.corrcoef(Er[a], Er[b])[0, 1] for a, b in pairs.values()]
    ceil = float(np.median(rc))
    P_(f"  N1 CEILING: {len(pairs)} genes knocked down twice independently (of {nmulti} with two promoters), "
       f"agreement of specific pathway effects median r {ceil:.3f} (IQR {np.percentile(rc, 25):.3f}-{np.percentile(rc, 75):.3f})")

    # ---- learner ---------------------------------------------------------------------------------
    dep = cd.load_depmap(); k562 = dep.loc["ACH-000551"]
    D = json.load(gzip.open(dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    feat_genes = sorted(set(kos) | {g for g in dep.columns if np.isfinite(k562[g])})
    fi = {g: i for i, g in enumerate(feat_genes)}
    blocks = ad.feature_blocks(feat_genes, genes, dep, D, bn)
    names_ = ["dep", "net", "proc", "chip", "prot"]
    tr = np.array([fi[k] for k in tr_k]); te = np.array([fi[k] for k in te_k])
    Ytr = E[[kidx[k] for k in tr_k]].astype(np.float64); Yte = E[[kidx[k] for k in te_k]].astype(np.float64)
    sc = lambda P, A: float(np.nanmedian(wc.pearson_rows(P, A)))
    Ftr, Fte = ad.design(blocks, names_, tr), ad.design(blocks, names_, te)
    lam, cvk = ad.cv_lambda(Ftr, Ytr, sc)
    Pk = ad.ridge_multi(Ftr, Ytr, Fte, [lam])[lam]
    Fall = np.concatenate([blocks[b] for b in names_], 1)
    folds = np.array_split(np.random.default_rng(2).permutation(len(tr)), 5); s_nn = []
    for f_ in folds:
        ti = np.setdiff1d(np.arange(len(tr)), f_)
        s_nn.append(sc(ad.nn_fit_predict(Fall[tr[ti]], Ytr[ti], Fall[tr[f_]], seed=3), Ytr[f_]))
    cvn = float(np.mean(s_nn))
    learner = "ridge" if cvk >= cvn else "neural net"
    P_(f"  inner-CV median r: ridge {cvk:.3f} (lambda {lam:g})  neural net {cvn:.3f} -> LEARNER = {learner}")
    Pl = Pk if learner == "ridge" else ad.nn_fit_predict(Fall[tr], Ytr, Fall[te], seed=3)
    r_l = wc.pearson_rows(Pl, Yte); r_l = np.where(np.isfinite(r_l), r_l, 0.0)
    w = int((r_l > 0).sum()); l = int((r_l < 0).sum())
    p2 = wc.sign_test(list(r_l))[2]
    P_(f"  N2 protein-specific effects on unseen proteins: median r {np.median(r_l):.3f} = {np.median(r_l) / ceil:.0%} of the ceiling; "
       f"r > 0 for {w}, < 0 for {l}, p {p2:.3g} -> {'LEARNS SPECIFIC EFFECTS' if p2 < 0.05 and w > l else 'DOES NOT'}")
    # hand-designed W2 in pathway space
    gene_mods = collections.defaultdict(set)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            gene_mods[g].add(c)
    mod_tr = collections.defaultdict(list)
    for k in tr_k:
        for c in gene_mods.get(k, ()):
            mod_tr[c].append(kidx[k])
    W2 = np.zeros_like(Yte)
    for r_, k in enumerate(te_k):
        cs = [c for c in gene_mods.get(k, ()) if c in mod_tr]
        if cs:
            W2[r_] = np.mean([E[mod_tr[c]].mean(0) for c in cs], 0)
    r_w = wc.pearson_rows(W2, Yte); r_w = np.where(np.isfinite(r_w), r_w, 0.0)
    w3, l3, p3 = wc.sign_test(list(r_l - r_w))
    P_(f"  N3 learner vs hand-designed process analogy: median r {np.median(r_l):.3f} vs {np.median(r_w):.3f}; better {w3}, worse {l3}, p {p3:.3g} -> "
       + ("BETTER" if p3 < 0.05 and w3 > l3 else "WORSE" if p3 < 0.05 else "NO DIFFERENCE"))
    top = lambda A: [set(np.argsort(-np.abs(a))[:10]) for a in A]
    tp, tm = top(Pl), top(Yte)
    ov = float(np.mean([len(a & b) for a, b in zip(tp, tm)]))
    nul = [float(np.mean([len(tp[i] & tm[j]) for i, j in enumerate(rng.permutation(len(tm)))])) for _ in range(200)]
    P_(f"  N4 top-10 most-affected pathways: predicted vs measured overlap {ov:.2f} of 10; shuffled null {np.mean(nul):.2f} "
       f"(95th pct {np.percentile(nul, 95):.2f}) -> {'ABOVE CHANCE' if ov > np.percentile(nul, 95) else 'AT CHANCE'}")
    rho = np.array([spearmanr(Pl[:, c], Yte[:, c]).correlation for c in range(M)])
    from scipy.stats import t as tdist
    n_ = len(te_k)
    tt = rho * np.sqrt((n_ - 2) / np.maximum(1 - rho ** 2, 1e-12)); pv = 2 * tdist.sf(np.abs(tt), n_ - 2)
    o = np.argsort(pv); q = np.empty(M); q[o] = np.minimum.accumulate((pv[o] * M / np.arange(1, M + 1))[::-1])[::-1]
    learnable = (q < 0.05) & (rho > 0)
    P_(f"  N5 LEARNABLE pathways (effect across unseen proteins predicted, BH FDR < 0.05): {int(learnable.sum())} of {M}")
    sysc = collections.Counter(rtop[mid[c]][0] for c in np.where(learnable)[0]); tot = collections.Counter(rtop[mid[c]][0] for c in range(M))
    P_("     by system: " + ", ".join(f"{s_} {sysc[s_]}/{tot[s_]}" for s_, _ in tot.most_common(12)))
    P_("     most learnable: " + "; ".join(f"{rnames[mid[c]][:40]} ({rho[c]:.2f})" for c in np.argsort(-rho)[:8]))
    P_("     least learnable: " + "; ".join(f"{rnames[mid[c]][:40]} ({rho[c]:.2f})" for c in np.argsort(rho)[:5]))

    # ---- fill the map ---------------------------------------------------------------------------
    allm = np.array([fi[k] for k in phen]); Yall = E[[kidx[k] for k in phen]].astype(np.float64)
    unseen = [g for g in feat_genes if g not in set(phen)]
    Fa, Fu = ad.design(blocks, names_, allm), ad.design(blocks, names_, np.array([fi[g] for g in unseen]))
    if learner == "ridge":
        Pu = ad.ridge_multi(Fa, Yall, Fu, [lam])[lam]
    else:
        Pu = ad.nn_fit_predict(Fall[allm], Yall, Fall[[fi[g] for g in unseen]], seed=3)
    keep = np.where(learnable)[0]
    pred_map = {}
    for i, g in enumerate(unseen):
        e = Pu[i, keep]; o_ = np.argsort(-np.abs(e))[:8]
        pred_map[g] = [[rnames[mid[keep[j]]], round(float(e[j]) * SCALE, 3), round(float(rho[keep[j]]), 3)] for j in o_]
    meas_map = {}
    for k in phen:
        e = E[kidx[k]]; o_ = np.argsort(-np.abs(e))[:10]
        meas_map[k] = [[rnames[mid[c]], round(float(e[c]) * SCALE, 3)] for c in o_]
    os.makedirs(BANK, exist_ok=True)
    with gzip.open(os.path.join(BANK, "effect_network_measured.json.gz"), "wt") as fh:
        json.dump({"what": "protein -> top-10 specific pathway effects (z-equivalent, tide removed), K562 CRISPRi, measured",
                   "source": "Replogle 2022 gwps (CC BY 4.0); Reactome (CC0)", "map": meas_map}, fh)
    with gzip.open(os.path.join(BANK, "effect_network_predicted.json.gz"), "wt") as fh:
        json.dump({"what": "protein -> top-8 predicted specific pathway effects for genes with no measured phenotype; "
                           "only LEARNABLE pathways; third value = that pathway's TEST Spearman (reliability)",
                   "learner": learner, "map": pred_map}, fh)
    P_(f"  FILL: measured map {len(meas_map):,} proteins; predicted map {len(pred_map):,} proteins never measured "
       f"(effects restricted to the {len(keep)} learnable pathways)")
    for g in ("TP53", "BRCA1", "KRAS", "PTEN", "EGFR"):
        if g in pred_map:
            P_(f"     e.g. {g} (predicted): " + "; ".join(f"{a[:34]} {b:+.2f} (rel {c:.2f})" for a, b, c in pred_map[g][:3]))
        elif g in meas_map:
            P_(f"     e.g. {g} (measured): " + "; ".join(f"{a[:34]} {b:+.2f}" for a, b in meas_map[g][:3]))
    res = dict(ceiling=ceil, n_pairs=len(pairs), learner=learner, cv_ridge=cvk, cv_nn=cvn, n2_median=float(np.median(r_l)),
               n2_fraction_of_ceiling=float(np.median(r_l) / ceil), n2_p=p2, n3=dict(better=w3, worse=l3, p=p3, w2_median=float(np.median(r_w))),
               n4=dict(overlap=ov, null=float(np.mean(nul)), null95=float(np.percentile(nul, 95))),
               n5_learnable=int(learnable.sum()), n_pathways=M, n_measured=len(meas_map), n_predicted=len(pred_map))
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(res, open(ART, "w"), indent=1, default=float)
    P_(f"\n  maps in memory_bank/cell_v3/ ; artifact outputs/effectmap.json ; runtime {time.time() - t0:.0f}s")
    P_("  X: one cell line, steady state; pathway effects are averages over their genes.")
    open(OUT, "w").write("\n".join(out) + "\n")


def posthoc():
    """POST-RUN CHECK (added after the run, committed before running it). The run showed red flags:
    the learner (median r 0.322) beat the replicate ceiling (0.098) by 3x, all 981 pathways came out
    'learnable', and shuffled predictions nearly matched the top-10 overlap. Suspected cause: the
    'specific' effects still share one dominant cell-wide axis (how hard a knockdown hits the cell),
    which the learner predicts from essentiality. Test: project out the top k principal components of
    TRAIN specific effects (k = 1 and k = 5) from every effect vector, then recompute the ceiling, N2,
    N4 and N5 on what remains."""
    from scipy.stats import spearmanr, t as tdist
    out = ["", RULE, "POST-RUN CHECK: REMOVE THE SHARED CELL-WIDE AXES, THEN ASK AGAIN", RULE]
    rng = np.random.default_rng(7)
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    mods, rnames, rtop = wc.reactome(set(genes)); mid = sorted(mods); M = len(mid)
    Mm = np.zeros((len(genes), M), np.float32)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            Mm[gidx[g], c] = 1
    score = lambda A: (np.where(np.isfinite(A), A, 0) @ Mm) / np.maximum(np.isfinite(A).astype(np.float32) @ Mm, 1)
    S = score(X)
    phen = [k for k, e in zip(kos, ep) if np.isfinite(e) and e < 0.05]
    tr_k = [k for k in phen if ad.split(k) == "TRAIN"]; te_k = [k for k in phen if ad.split(k) == "TEST"]
    tideS = S[[kidx[k] for k in tr_k]].mean(0)
    E = (S - tideS).astype(np.float64)
    Etr = E[[kidx[k] for k in tr_k]]
    U, Sv, Vt = np.linalg.svd(Etr, full_matrices=False)
    var = Sv ** 2 / (Sv ** 2).sum()
    out.append(f"  variance of TRAIN specific effects explained by PC1 {var[0]:.1%}, PCs 1-5 {var[:5].sum():.1%}")
    Xr, pairs, _ = replicate_rows(genes)
    Er = (score(Xr) - tideS).astype(np.float64)
    dep = cd.load_depmap(); k562 = dep.loc["ACH-000551"]
    D = json.load(gzip.open(dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    feat_genes = sorted(set(kos) | {g for g in dep.columns if np.isfinite(k562[g])})
    fi = {g: i for i, g in enumerate(feat_genes)}
    blocks = ad.feature_blocks(feat_genes, genes, dep, D, bn)
    names_ = ["dep", "net", "proc", "chip", "prot"]
    tr = np.array([fi[k] for k in tr_k]); te = np.array([fi[k] for k in te_k])
    Ftr, Fte = ad.design(blocks, names_, tr), ad.design(blocks, names_, te)
    sc = lambda P, A: float(np.nanmedian(wc.pearson_rows(P, A)))
    res = {"pc1_var": float(var[0]), "pc5_var": float(var[:5].sum())}
    for k in (1, 5):
        V = Vt[:k]
        proj = lambda A: A - (A @ V.T) @ V
        Ytr, Yte, Erk = proj(Etr), proj(E[[kidx[x] for x in te_k]]), proj(Er)
        rc = [np.corrcoef(Erk[a], Erk[b])[0, 1] for a, b in pairs.values()]
        lam, _ = ad.cv_lambda(Ftr, Ytr, sc)
        Pl = ad.ridge_multi(Ftr, Ytr, Fte, [lam])[lam]
        r_l = wc.pearson_rows(Pl, Yte); r_l = np.where(np.isfinite(r_l), r_l, 0.0)
        w = int((r_l > 0).sum()); l = int((r_l < 0).sum()); p2 = wc.sign_test(list(r_l))[2]
        top = lambda A: [set(np.argsort(-np.abs(a))[:10]) for a in A]
        tp, tm = top(Pl), top(Yte)
        ov = float(np.mean([len(a & b) for a, b in zip(tp, tm)]))
        nul = [float(np.mean([len(tp[i] & tm[j]) for i, j in enumerate(rng.permutation(len(tm)))])) for _ in range(200)]
        rho = np.array([spearmanr(Pl[:, c], Yte[:, c]).correlation for c in range(M)])
        n_ = len(te_k)
        tt = rho * np.sqrt((n_ - 2) / np.maximum(1 - rho ** 2, 1e-12)); pv = 2 * tdist.sf(np.abs(tt), n_ - 2)
        o = np.argsort(pv); q = np.empty(M); q[o] = np.minimum.accumulate((pv[o] * M / np.arange(1, M + 1))[::-1])[::-1]
        learn = (q < 0.05) & (rho > 0)
        out.append(f"  k = {k} axes removed: ceiling median r {np.median(rc):.3f}; learner median r {np.median(r_l):.3f} "
                   f"(r > 0 {w} / < 0 {l}, p {p2:.2g}); top-10 overlap {ov:.2f} vs null {np.mean(nul):.2f} (95th {np.percentile(nul, 95):.2f}); "
                   f"learnable pathways {int(learn.sum())}/{M}")
        out.append("      most learnable: " + "; ".join(f"{rnames[mid[c]][:38]} ({rho[c]:.2f})" for c in np.argsort(-rho)[:6]))
        res[f"k{k}"] = dict(ceiling=float(np.median(rc)), learner=float(np.median(r_l)), p=p2, top10=ov, null=float(np.mean(nul)),
                            null95=float(np.percentile(nul, 95)), learnable=int(learn.sum()),
                            top_learnable=[[rnames[mid[c]], float(rho[c])] for c in np.argsort(-rho)[:20]])
    print("\n".join(out))
    open(OUT, "a").write("\n".join(out) + "\n")
    a = json.load(open(ART)); a["posthoc"] = res; json.dump(a, open(ART, "w"), indent=1, default=float)


if __name__ == "__main__":
    import sys
    if "--posthoc" in sys.argv:
        posthoc()
    else:
        main()
