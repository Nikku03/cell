"""A cell model whose memory is MEASURED EXPERIMENTS, not literature facts -- and does the Human Protein
Atlas help it?

WHY. The literature memory bank's signs agree with K562 about half the time, and a plain learner given
raw per-gene data (askdata.py) beat every reasoning rule built on it. A design panel (four independent
designs, two of three judges finished, both chose this one) proposed keeping the reasoning engine's two
load-bearing ideas and feeding them real data:
  ADDRESSABLE MEMORY: every TRAIN knockdown's measured whole-cell response stays an individual item
    (transformer-style memory; chainsnap/transvs: lookup beat compression).
  SNAPPED HOPS: each hop commits to the k most similar REAL experiments (a hard top-k snap, as chainsnap
    writes down a discrete pointer) and combines them; a second hop re-retrieves from the result.
The engine's bump-node tree is NOT used here (its advantage was measured on synthetic chains only);
that is the next step only if memory + hops earn their place.

THE QUESTION (askdata's, sharpened). For an UNSEEN knockdown in K562 (genome-wide screen, energy-test
phenotype p < 0.05; askdata's SHA-256 TRAIN/TEST split), predict its SPECIFIC whole-cell response: the
measured response minus the TRAIN mean (the tide), over 8,248 genes, the knocked-down gene itself left
out of scoring. Score per TEST knockdown: Pearson r(predicted, measured). The tide predicts 0 specific
response (r = 0 by definition). No Perturb-seq measurement of the query protein (B, RPE1) is an input.

INPUTS, IDENTICAL FOR EVERY ARM: askdata's blocks (DepMap without K562, literature-network SVD, Reactome,
ENCODE, protein abundance) + three Human Protein Atlas v25 blocks (CC BY 4.0; Thul et al. 2017, doi
10.1126/science.aal3321; Uhlen et al. 2015, doi 10.1126/science.1260419), keyed by Ensembl id:
  hloc  multi-hot subcellular main + additional locations
  hcls  multi-hot protein classes, minus any class built from DepMap essentiality (would leak survival)
  hexp  log1p K-562 nTPM (HPA cell-line RNA) + a 'measured' flag
ARMS
  RIDGE-base   askdata's ridge (lambda by its inner CV) on askdata's blocks only
  RIDGE        the same on all blocks                                   <- THE BAR
  MEM-F        1 hop: retrieve the k nearest TRAIN knockdowns by feature cosine; softmax(sim / tau) average
  MEM-R        guess, then snap: RIDGE's prediction retrieves the k TRAIN knockdowns whose MEASURED
               responses are most similar; output (1 - alpha) * guess + alpha * retrieved average
  MEM-RR       as MEM-R with a second hop: the retrieved average retrieves again (same k, tau)
  CONTROLS     MEM-R with shuffled memory (values permuted against keys): must lose;
               RIDGE trained on shuffled answers: median r must be < 0.02 (no leak);
               CEILING (reported only): the same protein measured in the independent essential screen B
  k in (5, 10, 25, 50, 100), tau in (0.02, 0.05, 0.1), alpha in (0.25, 0.5, 0.75, 1.0), chosen per arm by
  5-fold CV on TRAIN only (mean of fold-median r). Ridge's lambda is chosen once on all of TRAIN by
  askdata.cv_lambda and reused inside the folds (the same small optimism for every memory arm).

=================================================================================================
GATES, PREDECLARED
=================================================================================================
H0 HARNESS (blocking): shuffled-answer RIDGE median r < 0.02; RIDGE-base reproduces askdata's level
   (median r > 0 on TEST).
Paired comparisons over TEST knockdowns, two-sided sign test on r differences. BETTER requires p < 0.05
AND median difference >= +0.01 (a minimum meaningful gain, fixed now); WORSE the mirror; else NO
DIFFERENCE.
E1 PRIMARY -- DOES THE EXPERIMENT MEMORY BEAT THE PLAIN LEARNER? MEM-R vs RIDGE.
E2 DOES A SECOND HOP ADD? MEM-RR vs MEM-R.
E3 DOES MEMORY CONTENT MATTER? MEM-R vs shuffled-memory MEM-R (must be BETTER, else E1 is void).
E4 DOES THE PROTEIN ATLAS HELP? RIDGE vs RIDGE-base.
E5 (descriptive) MEM-F vs RIDGE: retrieval by features alone.
X  NOT: one cell line, steady state; r over all genes rewards getting the shared axes right; the ceiling
   is a different day with the same reagents.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import csv
import gzip
import importlib.util
import io
import json
import math
import time
import zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
OUT = os.path.join(HERE, "RESULTS_expmem.txt")
ART = R("outputs", "expmem.json")
HPA = os.path.join(HERE, "_cache", "hpa")
RULE = "=" * 97


def _load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


ad = _load("askdata"); pe = _load("proteffect")
wc, cd, dc = ad.wc, ad.cd, ad.dc
KS, TAUS, ALPHAS = (5, 10, 25, 50, 100), (0.02, 0.05, 0.1), (0.25, 0.5, 0.75, 1.0)
MIN_GAIN, SEED = 0.01, 20261013


def hpa_blocks(ens):
    """HPA per-gene blocks for a list of Ensembl ids (None allowed)."""
    z = zipfile.ZipFile(os.path.join(HPA, "proteinatlas.tsv.zip"))
    loc, cls = {}, {}
    with z.open(z.namelist()[0]) as fh:
        for r in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"), delimiter="\t"):
            e = r["Ensembl"]
            loc[e] = {x.strip() for k in ("Subcellular main location", "Subcellular additional location")
                      for x in (r.get(k) or "").split(",") if x.strip()}
            cls[e] = {x.strip() for x in (r.get("Protein class") or "").split(",")
                      if x.strip() and "essential" not in x.lower()}
    z = zipfile.ZipFile(os.path.join(HPA, "rna_celline.tsv.zip"))
    tpm = {}
    with z.open(z.namelist()[0]) as fh:
        next(fh)
        for line in fh:
            p = line.decode().rstrip("\n").split("\t")
            if p[2] == "K-562":
                tpm[p[0]] = float(p[5])
    vl = sorted({x for v in loc.values() for x in v}); vc = sorted({x for v in cls.values() for x in v})
    il, ic = {x: j for j, x in enumerate(vl)}, {x: j for j, x in enumerate(vc)}
    n = len(ens)
    L = np.zeros((n, len(vl) + 1), np.float32); Cm = np.zeros((n, len(vc) + 1), np.float32)
    Ex = np.zeros((n, 2), np.float32)
    for i, e in enumerate(ens):
        if e in loc and loc[e]:
            L[i, [il[x] for x in loc[e]]] = 1
        else:
            L[i, -1] = 1
        if e in cls and cls[e]:
            Cm[i, [ic[x] for x in cls[e]]] = 1
        else:
            Cm[i, -1] = 1
        if e in tpm:
            Ex[i] = (math.log1p(tpm[e]), 1.0)
    Ex = (Ex - Ex.mean(0)) / (Ex.std(0) + 1e-6)
    cov = dict(location=int((L[:, -1] == 0).sum()), protein_class=int((Cm[:, -1] == 0).sum()),
               k562_rna=int(sum(e in tpm for e in ens)), n=n, n_locations=len(vl), n_classes=len(vc))
    return dict(hloc=L, hcls=Cm, hexp=Ex), cov


def unit(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def retrieve(S, V, k, tau):
    """Snap to the k most similar stored items (rows of similarity matrix S), softmax(sim / tau) average of V."""
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(S, idx, 1)
    w = np.exp((s - s.max(1, keepdims=True)) / tau); w /= w.sum(1, keepdims=True)
    out = np.zeros((S.shape[0], V.shape[1]), np.float32)
    for j in range(k):
        out += w[:, j:j + 1].astype(np.float32) * V[idx[:, j]]
    return out


def med_r(P, A):
    r = wc.pearson_rows(P, A)
    return float(np.nanmedian(r))


def sign_rule(d):
    w, l, p = wc.sign_test(list(d))
    md = float(np.median(d))
    v = "BETTER" if p < 0.05 and md >= MIN_GAIN else "WORSE" if p < 0.05 and md <= -MIN_GAIN else "NO DIFFERENCE"
    return w, l, p, md, v


def main():
    out, res = [], {}

    def P_(s=""):
        print(s, flush=True); out.append(s)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    P_(RULE); P_("A CELL MODEL WHOSE MEMORY IS MEASURED EXPERIMENTS -- AND DOES THE PROTEIN ATLAS HELP?"); P_(RULE)
    S = pe.load_screen(pe.FILES["A"])
    X, kos, ens, ep, genes = S["X"], S["kos"], S["kos_ens"], S["ep"], S["genes"]
    gi = pe.first_index(genes)
    phen = [i for i, e in enumerate(ep) if np.isfinite(e) and e < 0.05]
    tr = [i for i in phen if ad.split(kos[i]) == "TRAIN"]; te = [i for i in phen if ad.split(kos[i]) == "TEST"]
    Act = X.copy()
    for i in phen:
        if kos[i] in gi:
            Act[i, gi[kos[i]]] = np.nan
    tide = np.nanmean(Act[tr], 0)
    Ytr = np.nan_to_num(Act[tr] - tide).astype(np.float32); Ate = (Act[te] - tide).astype(np.float32)
    P_(f"  knockdowns with a phenotype {len(phen):,}: TRAIN {len(tr):,}, TEST {len(te):,}; genes {len(genes):,}")

    dep = cd.load_depmap(); D = json.load(gzip.open(dc.ENCY)); bn = [r["name"] for r in D["genes"]]
    fg = [kos[i] for i in phen]; fpos = {i: j for j, i in enumerate(phen)}
    blocks = ad.feature_blocks(fg, genes, dep, D, bn)
    hb, cov = hpa_blocks([ens[i] for i in phen])
    blocks.update(hb)
    P_(f"  HPA coverage of these knockdowns: location {cov['location']:,}, protein class {cov['protein_class']:,}, "
       f"K-562 RNA {cov['k562_rna']:,} of {cov['n']:,} ({cov['n_locations']} locations, {cov['n_classes']} classes)")
    base = ["dep", "net", "proc", "chip", "prot"]; full = base + ["hloc", "hcls", "hexp"]
    rtr = [fpos[i] for i in tr]; rte = [fpos[i] for i in te]
    sc = lambda P, A: med_r(P, A)
    preds, lam = {}, {}
    for name, bl in (("RIDGE-base", base), ("RIDGE", full)):
        Ftr, Fte = ad.design(blocks, bl, rtr), ad.design(blocks, bl, rte)
        lam[name], _ = ad.cv_lambda(Ftr, Ytr, sc)
        preds[name] = ad.ridge_multi(Ftr, Ytr, Fte, [lam[name]])[lam[name]].astype(np.float32)
    Ftr, Fte = ad.design(blocks, full, rtr), ad.design(blocks, full, rte)
    perm = rng.permutation(len(tr))
    preds["shuffled-answer RIDGE"] = ad.ridge_multi(Ftr, Ytr[perm], Fte, [lam["RIDGE"]])[lam["RIDGE"]].astype(np.float32)

    # ---- inner CV for the memory arms (TRAIN only) -------------------------------------------------
    folds = np.array_split(np.random.default_rng(2).permutation(len(tr)), 5)
    grid = {a: {} for a in ("MEM-F", "MEM-R", "MEM-RR")}
    Kf_all = unit(Ftr)
    for f in folds:
        ti = np.setdiff1d(np.arange(len(tr)), f)
        g = ad.ridge_multi(Ftr[ti], Ytr[ti], Ftr[f], [lam["RIDGE"]])[lam["RIDGE"]].astype(np.float32)
        Kr = unit(Ytr[ti]); Sf = Kf_all[f] @ Kf_all[ti].T; Sr = unit(g) @ Kr.T
        A_f = Ytr[f]
        for k in KS:
            for tau in TAUS:
                grid["MEM-F"].setdefault((k, tau, 1.0), []).append(sc(retrieve(Sf, Ytr[ti], k, tau), A_f))
                y1 = retrieve(Sr, Ytr[ti], k, tau)
                y2 = retrieve(unit(y1) @ Kr.T, Ytr[ti], k, tau)
                for a in ALPHAS:
                    grid["MEM-R"].setdefault((k, tau, a), []).append(sc((1 - a) * g + a * y1, A_f))
                    grid["MEM-RR"].setdefault((k, tau, a), []).append(sc((1 - a) * g + a * y2, A_f))
    best = {a: max(gd, key=lambda h: np.mean(gd[h])) for a, gd in grid.items()}
    P_("  inner CV (TRAIN): " + "; ".join(f"{a} k {h[0]} tau {h[1]} alpha {h[2]} (cv median r {np.mean(grid[a][h]):.3f})"
                                         for a, h in best.items()))

    # ---- TEST predictions ------------------------------------------------------------------------
    g = preds["RIDGE"]; Kr = unit(Ytr)
    k, tau, _ = best["MEM-F"]
    preds["MEM-F"] = retrieve(unit(Fte) @ Kf_all.T, Ytr, k, tau)
    k, tau, a = best["MEM-R"]
    Sr = unit(g) @ Kr.T
    y1 = retrieve(Sr, Ytr, k, tau)
    preds["MEM-R"] = (1 - a) * g + a * y1
    preds["MEM-R shuffled memory"] = (1 - a) * g + a * retrieve(Sr, Ytr[rng.permutation(len(tr))], k, tau)
    k, tau, a = best["MEM-RR"]
    y1 = retrieve(Sr, Ytr, k, tau)
    preds["MEM-RR"] = (1 - a) * g + a * retrieve(unit(y1) @ Kr.T, Ytr, k, tau)

    r = {n: np.nan_to_num(wc.pearson_rows(P, Ate)) for n, P in preds.items()}
    # ceiling: the same protein in the independent essential screen B
    Bs = pe.load_screen(pe.FILES["B"]); bk = {p: i for i, p in enumerate(Bs["kos"])}; bg = pe.first_index(Bs["genes"])
    btr = [bk[kos[i]] for i in tr if kos[i] in bk]
    cols = [j for j, gname in enumerate(genes) if gname in bg]; bcols = [bg[genes[j]] for j in cols]
    btide = np.nanmean(Bs["X"][btr][:, bcols], 0)
    ce = []
    for i in te:
        if kos[i] in bk:
            b = np.nan_to_num(Bs["X"][bk[kos[i]], bcols] - btide)[None]
            ce.append(float(np.nan_to_num(wc.pearson_rows(b, Ate[[te.index(i)]][:, cols])[0])))
    P_(); P_("  TEST knockdowns: median / mean Pearson r of the predicted SPECIFIC response")
    for n in ("RIDGE-base", "RIDGE", "MEM-F", "MEM-R", "MEM-RR", "MEM-R shuffled memory", "shuffled-answer RIDGE"):
        P_(f"     {n:24s} {np.median(r[n]):+.3f} / {np.mean(r[n]):+.3f}")
    P_(f"     {'CEILING: screen B':24s} {np.median(ce):+.3f} / {np.mean(ce):+.3f}   ({len(ce)} TEST proteins also in B)")
    P_(f"     {'tide':24s} +0.000 (predicts no specific response)")
    res["scores"] = {n: dict(median=float(np.median(v)), mean=float(np.mean(v))) for n, v in r.items()}
    res["ceiling"] = dict(median=float(np.median(ce)), mean=float(np.mean(ce)), n=len(ce))
    res["best"] = {a: list(h) for a, h in best.items()}; res["lambda"] = lam; res["hpa_coverage"] = cov

    h0 = np.median(r["shuffled-answer RIDGE"]) < 0.02 and np.median(r["RIDGE-base"]) > 0
    P_(f"  H0 shuffled-answer median r {np.median(r['shuffled-answer RIDGE']):+.3f} < 0.02 and RIDGE-base > 0 -> {'PASS' if h0 else 'FAIL'}")
    res["H0"] = bool(h0)
    for tag, a_, b_ in (("E1 experiment memory vs plain learner (MEM-R vs RIDGE)", "MEM-R", "RIDGE"),
                        ("E2 second hop (MEM-RR vs MEM-R)", "MEM-RR", "MEM-R"),
                        ("E3 memory content (MEM-R vs shuffled memory)", "MEM-R", "MEM-R shuffled memory"),
                        ("E4 Protein Atlas (RIDGE vs RIDGE-base)", "RIDGE", "RIDGE-base"),
                        ("E5 retrieval by features alone (MEM-F vs RIDGE)", "MEM-F", "RIDGE")):
        w, l, p, md, v = sign_rule(r[a_] - r[b_])
        P_(f"  {tag}: better {w:,} / worse {l:,}, median diff {md:+.4f}, p {p:.2g} -> {v}" + ("" if h0 else "  [H0 FAILED]"))
        res[tag.split()[0]] = dict(better=w, worse=l, median_diff=md, p=p, verdict=v)
    res["runtime_s"] = time.time() - t0
    P_(f"  runtime {time.time() - t0:.0f}s; data: Replogle 2022 (CC BY 4.0), HPA v25 (CC BY 4.0), DepMap 24Q4 (CC BY 4.0)")
    open(OUT, "w").write("\n".join(out) + "\n")
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    json.dump(pe.clean(res), open(ART, "w"), indent=1)


if __name__ == "__main__":
    main()
