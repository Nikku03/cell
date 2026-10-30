"""Don't tell it how to reason: give a general learner the data and a question, nothing else.

WHY. Every earlier module hand-designed the reasoning -- sign propagation, hop decay, process
averaging (wholecell.py's W2), the tide. The instruction now: supply the DATA and the QUESTION, and
let a generic learner find its own way from one to the other.

THE DATA, per gene, as raw feature blocks (none used as a rule):
    dep    DepMap 24Q4 gene effect in 1,177 cancer lines (K562 REMOVED -- it is a question's answer)
    net    literature network: signed out- and in-links of the gene to every measured gene (bank +
           OmniPath commercial subset), compressed by an unsupervised SVD (256 dims, no labels used)
    proc   Reactome memberships (CC0), binary, every human pathway with >= 5 member genes
    chip   which ENCODE K562 factors bind its promoter region (2015 gene sets), binary
    prot   protein abundance (log ppm) and a 'has protein record' flag
THE QUESTIONS (answers never enter the features):
    Q1 WHOLE CELL   "knock this gene down: what happens to all 8,248 measured genes?" (K562 gwps
                    Perturb-seq; knockdowns with a phenotype; TRAIN / TEST = the SHA-256 parity split)
    Q2 SURVIVAL     "can K562 live without this gene?" (DepMap K562 gene effect; TRAIN / TEST split)
THE LEARNERS (generic; chosen per question on TRAIN only by 5-fold inner CV):
    KR   ridge regression (kernel ridge with one trace-normalised linear kernel per block, summed --
         run in its equivalent primal form), with an intercept so it shrinks toward the TRAIN mean;
         lambda from 5-fold CV on TRAIN
    NN   a small neural network (2 hidden layers of 512, dropout 0.2, Adam, early stopping on a TRAIN
         hold-out) -- for Q1 it predicts the top 64 principal components of TRAIN responses (PCA fitted
         on TRAIN), for Q2 the gene effect
    The one with the better inner-CV score is the LEARNER for that question; both are reported.

=================================================================================================
GATES, PREDECLARED
=================================================================================================
H0 NO LEAKAGE, BLOCKING. KR trained on TRAIN answers SHUFFLED across TRAIN genes must NOT beat the
   tide on TEST (Q1: per-knockdown sign test p >= 0.05 or worse). If it does, something leaks.
Q1 per TEST knockdown, Pearson r between predicted and measured whole-cell response (knocked-down gene
   excluded, missing values excluded). Paired sign tests, two-sided p < 0.05:
     A1 LEARNER vs TIDE (mean TRAIN response)          -- does it learn anything?
     A2 LEARNER vs W2 (wholecell.py's hand-designed process reasoning) -- does free beat designed?
     A3 ABLATION: KR without each block vs KR with all -> which data it relies on.
Q2 TEST genes: Spearman with the true gene effect, and AUROC for essential (<= -0.5). Compared with
   wholecell.py's F2 process prior (AUROC 0.780), and ablated by block as A3.
X  NOT: one cell line, steady state; 'reasoning' here is whatever a kernel or a small network finds
   -- it is not inspected for mechanism, only for which data it needs.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import collections
import gzip
import hashlib
import importlib.util
import json
import math
import time
import zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
CA = os.path.join(HERE, "_cache")
OUT = os.path.join(HERE, "RESULTS_askdata.txt")
ART = R("outputs", "askdata.json")
RULE = "=" * 97
_s = importlib.util.spec_from_file_location("wholecell", os.path.join(HERE, "wholecell.py"))
wc = importlib.util.module_from_spec(_s); _s.loader.exec_module(wc)
cd, cb, dc = wc.cd, wc.cb, wc.dc
LAMS = [10 ** x for x in (-2, -1, 0, 1, 2, 3, 4)]
split = lambda k: "TRAIN" if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 2 == 0 else "TEST"


def feature_blocks(all_genes, genes_meas, dep, D, bn):
    """Raw per-gene data blocks for every gene in all_genes."""
    gi = {g: i for i, g in enumerate(all_genes)}; n = len(all_genes)
    blocks = {}
    dcols = [c for c in dep.index if c != "ACH-000551"]
    Dm = dep.loc[dcols].T                                           # genes x lines
    Fd = np.zeros((n, len(dcols)), np.float32); has_d = np.zeros(n, np.float32)
    for g in all_genes:
        if g in Dm.index:
            v = Dm.loc[g].values.astype(np.float32)
            Fd[gi[g]] = np.where(np.isfinite(v), v, np.nanmean(v) if np.isfinite(v).any() else 0.0); has_d[gi[g]] = 1
    Fd = (Fd - Fd.mean(0)) / (Fd.std(0) + 1e-6)
    blocks["dep"] = np.concatenate([Fd, has_d[:, None]], 1)
    from scipy.sparse import lil_matrix
    mi = {g: j for j, g in enumerate(genes_meas)}
    A = lil_matrix((n, 2 * len(genes_meas)), dtype=np.float32)
    rows = cb.facts_table(D, bn)[0] + [(s, t, sg, 0) for s, t, sg, _, _ in cd.load_omnipath()]
    for s, t, sg, _ in rows:
        if s in gi and t in mi:
            A[gi[s], mi[t]] = sg
        if t in gi and s in mi:
            A[gi[t], len(genes_meas) + mi[s]] = sg
    from scipy.sparse.linalg import svds
    A = A.tocsr()
    u, sv, _ = svds(A.astype(np.float64), k=256)
    blocks["net"] = (u * sv).astype(np.float32)
    z = zipfile.ZipFile(os.path.join(wc.RE, "ReactomePathways.gmt.zip"))
    pw = []
    for line in z.read(z.namelist()[0]).decode().splitlines():
        p = line.split("\t")
        mem = [x for x in p[2:] if x in gi]
        if len(mem) >= 5:
            pw.append(mem)
    Fp = np.zeros((n, len(pw)), np.float32)
    for c, mem in enumerate(pw):
        for g in mem:
            Fp[gi[g], c] = 1
    blocks["proc"] = Fp
    enc = cd.load_encode(); tfs = sorted(enc)
    Fc = np.zeros((n, len(tfs)), np.float32)
    for c, tf in enumerate(tfs):
        for g in enc[tf]:
            if g in gi:
                Fc[gi[g], c] = 1
    blocks["chip"] = Fc
    protein = {bn[int(k)]: v for k, v in D["ppm"].items() if k.isdigit()}
    Fq = np.array([[math.log1p(protein.get(g, 0.0)), float(g in protein)] for g in all_genes], np.float32)
    blocks["prot"] = (Fq - Fq.mean(0)) / (Fq.std(0) + 1e-6)
    return blocks


def design(blocks, names_, rows):
    """Concatenate blocks, each scaled so its linear kernel has unit mean diagonal -- the primal form
    of 'one trace-normalised linear kernel per block, summed' (same model, far less memory)."""
    out = []
    for b in names_:
        F = blocks[b][rows].astype(np.float64)
        c = float(np.mean(np.sum(blocks[b].astype(np.float64) ** 2, 1))) + 1e-12
        out.append(F / math.sqrt(c))
    return np.concatenate(out, 1)


def ridge_multi(Ftr, Ytr, Fte, lams):
    """Ridge with an intercept (features and answers centred on TRAIN), all lambdas from one SVD."""
    fm, ym = Ftr.mean(0), Ytr.mean(0)
    U, S, Vt = np.linalg.svd(Ftr - fm, full_matrices=False)
    UtY = U.T @ (Ytr - ym); FV = (Fte - fm) @ Vt.T
    return {lam: FV @ ((S / (S ** 2 + lam))[:, None] * UtY) + ym for lam in lams}


def cv_lambda(F, Y, score, seed=0):
    rng = np.random.default_rng(seed); n = len(F); folds = np.array_split(rng.permutation(n), 5)
    acc = collections.defaultdict(list)
    for f in folds:
        ti = np.setdiff1d(np.arange(n), f)
        for lam, P in ridge_multi(F[ti], Y[ti], F[f], LAMS).items():
            acc[lam].append(score(P, Y[f]))
    best = max(LAMS, key=lambda l: np.mean(acc[l]))
    return best, float(np.mean(acc[best]))


def rows_r(P, A):
    return wc.pearson_rows(P, A)


def nn_fit_predict(Ftr, Ytr, Fte, seed, epochs=400):
    import torch
    torch.manual_seed(seed); torch.set_num_threads(4)
    n = len(Ftr); rng = np.random.default_rng(seed); idx = rng.permutation(n); v = idx[: n // 10]; t = idx[n // 10:]
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-6
    Yt = torch.tensor((Ytr - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(Ftr.shape[1], 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, Ytr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Xv = f(Ftr[t]), f(Ftr[v]); best, bstate, bad = 1e18, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(t))
        for i in range(0, len(t), 128):
            b = perm[i:i + 128]
            loss = ((net(Xt[b]) - Yt[t][b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(Xv) - Yt[v]) ** 2).mean())
        if vl < best - 1e-5:
            best, bstate, bad = vl, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad():
        return net(f(Fte)).numpy() * ys + ym


def main():
    from scipy.stats import spearmanr, rankdata
    out = []

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    X, kos, genes, ep, fold = cd.load_gwps()
    gidx = {g: j for j, g in enumerate(genes)}; kidx = {k: i for i, k in enumerate(kos)}
    dep = cd.load_depmap()
    k562 = dep.loc["ACH-000551"]
    D = json.load(gzip.open(dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    phen = [k for k, e in zip(kos, ep) if np.isfinite(e) and e < 0.05]
    surv = [g for g in dep.columns if np.isfinite(k562[g])]
    all_genes = sorted(set(phen) | set(surv))
    gi = {g: i for i, g in enumerate(all_genes)}
    P_(RULE); P_("GIVE IT THE DATA AND A QUESTION -- NO REASONING SUPPLIED"); P_(RULE)
    blocks = feature_blocks(all_genes, genes, dep, D, bn)
    names_ = ["dep", "net", "proc", "chip", "prot"]
    P_("  data blocks per gene: " + ", ".join(f"{b} {blocks[b].shape[1]:,}" for b in names_) + f"   genes {len(all_genes):,}")
    res = {}

    # =================================================================== Q1 whole cell ===========
    tr_k = [k for k in phen if split(k) == "TRAIN"]; te_k = [k for k in phen if split(k) == "TEST"]
    tr = np.array([gi[k] for k in tr_k]); te = np.array([gi[k] for k in te_k])
    Y = np.where(np.isfinite(X[[kidx[k] for k in tr_k]]), X[[kidx[k] for k in tr_k]], 0.0).astype(np.float64)
    actual = X[[kidx[k] for k in te_k]].astype(np.float64)
    for r, k in enumerate(te_k):
        if k in gidx:
            actual[r, gidx[k]] = np.nan
    tide = np.nanmean(X[[kidx[k] for k in tr_k]], 0)
    r_tide = rows_r(np.tile(tide, (len(te_k), 1)), actual)
    sc1 = lambda Pp, A: float(np.nanmedian(rows_r(Pp, A)))
    P_("\n" + RULE); P_(f"Q1 WHOLE CELL: TRAIN {len(tr_k):,} knockdowns, TEST {len(te_k):,}"); P_(RULE)
    Ftr, Fte = design(blocks, names_, tr), design(blocks, names_, te)
    rng = np.random.default_rng(1)
    Ysh = Y[rng.permutation(len(Y))]
    lam_s, _ = cv_lambda(Ftr, Ysh, sc1)
    r_sh = rows_r(ridge_multi(Ftr, Ysh, Fte, [lam_s])[lam_s], actual)
    w, l, p = wc.sign_test(r_sh - r_tide)
    h0 = not (p < 0.05 and w > l)
    P_(f"  H0 shuffled-answer learner vs tide: median r {np.nanmedian(r_sh):.3f} vs {np.nanmedian(r_tide):.3f}; better {w}, worse {l}, p {p:.3g} "
       f"-> {'PASS (no leakage)' if h0 else 'FAIL (leakage)'}")
    if not h0:
        open(OUT, "w").write("\n".join(out) + "\n"); return
    lam1, cv_kr = cv_lambda(Ftr, Y, sc1)
    P_kr = ridge_multi(Ftr, Y, Fte, [lam1])[lam1]; r_kr = rows_r(P_kr, actual)
    Fall = np.concatenate([blocks[b] for b in names_], 1)
    U, Sv, Vt = np.linalg.svd(Y - Y.mean(0), full_matrices=False)
    comps = Vt[:64]
    # NN inner score: 5-fold on TRAIN
    rng2 = np.random.default_rng(2); folds = np.array_split(rng2.permutation(len(tr)), 5); s_nn = []
    for f_ in folds:
        ti = np.setdiff1d(np.arange(len(tr)), f_)
        Uc, Sc, Vc = np.linalg.svd(Y[ti] - Y[ti].mean(0), full_matrices=False)
        cc = Vc[:64]
        pr = nn_fit_predict(Fall[tr[ti]], (Y[ti] - Y[ti].mean(0)) @ cc.T, Fall[tr[f_]], seed=3) @ cc + Y[ti].mean(0)
        s_nn.append(sc1(pr, Y[f_]))
    cv_nn = float(np.mean(s_nn))
    P_nn = nn_fit_predict(Fall[tr], (Y - Y.mean(0)) @ comps.T, Fall[te], seed=3) @ comps + Y.mean(0)
    r_nn = rows_r(P_nn, actual)
    learner = "KR" if cv_kr >= cv_nn else "NN"
    r_L = r_kr if learner == "KR" else r_nn
    P_(f"  inner-CV median r: KR {cv_kr:.3f} (lambda {lam1:g})   NN {cv_nn:.3f}   -> LEARNER = {learner}")
    # hand-designed W2 from wholecell, recomputed on the same TEST set
    mods, rnames, rtop = wc.reactome(set(genes))
    gene_mods = collections.defaultdict(set); mid = sorted(mods)
    for c, pid in enumerate(mid):
        for g in mods[pid]:
            gene_mods[g].add(c)
    mod_tr = collections.defaultdict(list)
    for k in tr_k:
        for c in gene_mods.get(k, ()):
            mod_tr[c].append(kidx[k])
    W2 = np.tile(tide, (len(te_k), 1))
    for r, k in enumerate(te_k):
        cs = [c for c in gene_mods.get(k, ()) if c in mod_tr]
        if cs:
            W2[r] = np.mean([np.nanmean(X[mod_tr[c]], 0) for c in cs], 0)
    r_w2 = rows_r(W2, actual)
    for nm, rr in (("tide", r_tide), ("W2 hand-designed process reasoning", r_w2), ("KR (free)", r_kr), ("NN (free)", r_nn)):
        P_(f"     {nm:<36} median r {np.nanmedian(rr):.3f}   mean r {np.nanmean(rr):.3f}")
    for key, a, b, nm in (("A1", r_L, r_tide, f"{learner} vs tide"), ("A2", r_L, r_w2, f"{learner} vs hand-designed W2")):
        w, l, p = wc.sign_test(a - b)
        v = "BETTER" if p < 0.05 and w > l else "WORSE" if p < 0.05 else "NO DIFFERENCE"
        res[key] = dict(better=w, worse=l, p=p, verdict=v, median_a=float(np.nanmedian(a)), median_b=float(np.nanmedian(b)))
        P_(f"  {key} {nm:<32} better {w}, worse {l}, p {p:.3g} -> {v}")
    P_("  A3 what it relies on -- KR with one block removed (lambda re-chosen on TRAIN):")
    abl = {}
    for b in names_:
        keep = [x for x in names_ if x != b]
        Fa, Fb = design(blocks, keep, tr), design(blocks, keep, te)
        lb, _ = cv_lambda(Fa, Y, sc1)
        rb = rows_r(ridge_multi(Fa, Y, Fb, [lb])[lb], actual)
        w, l, p = wc.sign_test(r_kr - rb)
        abl[b] = dict(median=float(np.nanmedian(rb)), full_better=w, full_worse=l, p=p)
        P_(f"     without {b:<5} median r {np.nanmedian(rb):.3f} (full {np.nanmedian(r_kr):.3f}); full better on {w}, worse on {l}, p {p:.3g}"
           + ("  <- NEEDED" if p < 0.05 and w > l else ""))
    for b in names_:
        Fa, Fb = design(blocks, [b], tr), design(blocks, [b], te)
        lb, _ = cv_lambda(Fa, Y, sc1)
        rb = rows_r(ridge_multi(Fa, Y, Fb, [lb])[lb], actual)
        abl[b]["alone_median"] = float(np.nanmedian(rb))
    P_("     each block ALONE: " + ", ".join(f"{b} {abl[b]['alone_median']:.3f}" for b in names_))
    res["Q1"] = dict(learner=learner, cv_kr=cv_kr, cv_nn=cv_nn, lam=lam1, median_kr=float(np.nanmedian(r_kr)),
                     median_nn=float(np.nanmedian(r_nn)), median_tide=float(np.nanmedian(r_tide)),
                     median_w2=float(np.nanmedian(r_w2)), shuffled=float(np.nanmedian(r_sh)), ablation=abl)

    # =================================================================== Q2 survival =============
    P_("\n" + RULE); P_("Q2 SURVIVAL: CAN K562 LIVE WITHOUT THIS GENE? (K562 column removed from the data)"); P_(RULE)
    trs = np.array([gi[g] for g in surv if split(g) == "TRAIN"]); tes = np.array([gi[g] for g in surv if split(g) == "TEST"])
    ys = np.array([k562[g] for g in surv if split(g) == "TRAIN"]); yt = np.array([k562[g] for g in surv if split(g) == "TEST"])

    def auc(sc, y):
        rk = rankdata(sc); pos = y
        return float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (len(pos) - pos.sum())))
    sc2 = lambda Pp, A: float(spearmanr(Pp.ravel(), A.ravel()).correlation)
    Fs_tr, Fs_te = design(blocks, names_, trs), design(blocks, names_, tes)
    lam2, cvk = cv_lambda(Fs_tr, ys[:, None], sc2)
    pk = ridge_multi(Fs_tr, ys[:, None], Fs_te, [lam2])[lam2].ravel()
    s_nn2 = []
    rng3 = np.random.default_rng(4); folds = np.array_split(rng3.permutation(len(trs)), 5)
    for f_ in folds:
        ti = np.setdiff1d(np.arange(len(trs)), f_)
        s_nn2.append(sc2(nn_fit_predict(Fall[trs[ti]], ys[ti, None], Fall[trs[f_]], seed=5), ys[f_, None]))
    cvn = float(np.mean(s_nn2))
    pn = nn_fit_predict(Fall[trs], ys[:, None], Fall[tes], seed=5).ravel()
    learner2 = "KR" if cvk >= cvn else "NN"
    pl = pk if learner2 == "KR" else pn
    ess = yt <= -0.5
    P_(f"  inner-CV Spearman: KR {cvk:.3f}  NN {cvn:.3f} -> LEARNER = {learner2}")
    P_(f"  TEST genes {len(tes):,} (essential {int(ess.sum()):,}): Spearman {spearmanr(pl, yt).correlation:.3f}; "
       f"AUROC essential {auc(-pl, ess):.3f}   (wholecell F2 hand-designed process prior: 0.780)")
    abl2 = {}
    for b in names_:
        keep = [x for x in names_ if x != b]
        Fa, Fb = design(blocks, keep, trs), design(blocks, keep, tes)
        lb, _ = cv_lambda(Fa, ys[:, None], sc2)
        abl2[b] = auc(-ridge_multi(Fa, ys[:, None], Fb, [lb])[lb].ravel(), ess)
    P_("  without each block, AUROC: " + ", ".join(f"{b} {abl2[b]:.3f}" for b in names_))
    res["Q2"] = dict(learner=learner2, cv_kr=cvk, cv_nn=cvn, spearman=float(spearmanr(pl, yt).correlation),
                     auroc=auc(-pl, ess), auroc_kr=auc(-pk, ess), auroc_nn=auc(-pn, ess), ablation_auroc=abl2)
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(res, open(ART, "w"), indent=1, default=float)
    P_(f"\n  artifact: outputs/askdata.json   runtime {time.time() - t0:.0f}s")
    P_("  X: one cell line, steady state; the learners' 'reasoning' is judged only by what they predict and which data they need.")
    open(OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
