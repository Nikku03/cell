"""CELLFIX -- fixes for the cell-map limits, each tested before it is used.

Asked by the user: "first add public available info, then the yes things, then complete the list". The data came in
with fixbank.py (ProCan and CCLE proteomics, HPA tissue immunohistochemistry and single-cell-type RNA, BioPlex in two
cell lines, PROGENy, HPA cell-cycle dependence). Lines and genes DEV / EXAM by sha256('cellbrain-c1:' + id) parity.
BETTER (house rule) = one-sided sign test p < 0.05 AND median gain >= +0.01.

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G8-*):
 G8-P1  PROTEIN FROM RNA, CANCER LINES (limit 1). ProCan lines with DepMap RNA. Per gene, offset = median over DEV
        lines of (protein - RNA) (log2). On each EXAM line, Spearman across genes of predicted protein (RNA + offset)
        vs measured protein is BETTER than of RNA alone. Negative: offsets shuffled between genes (must not be
        BETTER). Planted: protein replaced by RNA + offset + noise (must be BETTER).
 G8-P2  PROTEIN FROM RNA, BODY CELL TYPES (the same offsets carried to tissues). In the 21 HPA single-cell types whose
        name matches an immunohistochemistry cell type: Spearman across genes of log2(nCPM+1) + offset vs the protein
        staining level (max over tissues; reliability approved or better) is BETTER than of RNA alone over the 21.
        Negative: shuffled offsets.
 G8-D1  DEEPER RNA FINDS REAL PROTEINS (limit 2). In the same 21 cell types, genes made by the deep pseudobulk RNA
        (nCPM >= 1) but outside the cell type's top 1,500 (what single-cell detection sees) have their protein stained
        more often than genes not made (nCPM < 1): Mantel-Haenszel odds ratio over cell types >= 2, CMH p < 0.05, and
        Monte Carlo p < 0.05 against 200 draws with RNA shuffled between genes within each cell type.
 G8-I1  PARTNERS DEPEND ON THE CELL (limit 3; the "both made" rule with interactions measured in two cell types).
        BioPlex pairs whose two proteins appear in both the HEK293T and the HCT116 network, seen in only one line.
        delta = RNA percentile of the less-made partner in HCT116 minus in HEK293T (HCT116: DepMap RNA; HEK293T:
        OpenCell RNA). CLAIM: median delta(HCT116-only) - median delta(HEK293T-only) >= 0.02, one-sided Mann-Whitney
        p < 0.05, Monte Carlo p < 0.05 (line labels permuted 1,000 times). Negative: one permutation. Planted: +0.05.
 G8-A1  ON / OFF FROM TARGETS (limit 5). Activity of a transcription factor in a profile = mean over its CollecTRI
        targets of sign x (lfc / lfcSE). In the K562 genome-wide screen, for each knocked-down TF with >= 10 signed
        targets measured, the share of other TFs whose activity is higher than its own (AUROC; own expected lowest).
        CLAIM: mean AUROC >= 0.60 and Monte Carlo p < 0.05 against 200 draws with each regulon's targets replaced by
        random measured genes (same number and signs). Negative: the first draw. Planted: own activity set to -3.
 G8-U1  JOBS OF UNKNOWN PROTEINS FROM NEIGHBOURS (limit 6). Similarity of two genes = mean Fisher-z of their
        correlations in each evidence both have: DepMap gene effects, ProCan protein levels, HPA single-cell-type RNA,
        K562 knockdown patterns. Score of an EXAM gene for a Reactome pathway (10-300 members, >= 5 DEV members) =
        mean of its 5 highest similarities to the pathway's DEV members. CLAIM: median AUROC over pathways with >= 5
        EXAM members >= 0.70 and Monte Carlo p < 0.05 (membership labels permuted among EXAM genes, 200 draws).
        Negative: the first permutation. Planted: a pathway's members replaced by the genes it scores highest.
 G8-M1  MACHINES, WITH PROTEIN (limit 7). G7-M1 repeated with ProCan: a part counts as made where its protein is
        detected. Lines with ProCan and DepMap gene effects; same statistic, thresholds and Monte Carlo (stand-ins
        matched on detection-rate decile).
 G8-M2  MACHINES MOVE TOGETHER AS PROTEIN. Complexes (Complex Portal, 3-60 parts, >= 3 measured): median pairwise
        correlation across lines of the parts' protein levels minus that of their RNA levels is BETTER over complexes,
        AND the protein value beats random gene sets of the same size (matched on detection decile) in 200 draws
        (median over complexes; p_MC < 0.05).
Limit 4 (timing) is descriptive here: cell-cycle-dependent proteins (HPA) are flagged in the maps; true timing needs
time-course data per cell type (see the closing list in RESULTS).

Run: python3 -I rem/atlas/cellfix.py declare | test [--trial] | apply
"""

from __future__ import annotations
import collections
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
TRIAL = "--trial" in sys.argv


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]] + (["--trial"] if TRIAL else [])
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


pp = _load("generegppi"); pa = _load("generegpara")
gx = pp.gx; cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "cellfix")
CLAIMS = ["G8-P1", "G8-P2", "G8-D1", "G8-I1", "G8-A1", "G8-U1", "G8-M1", "G8-M2"]
SEED = 20261128
N200, N1000 = (20, 50) if TRIAL else (200, 1000)
H_ = lambda e: int(hashlib.sha256(("cellbrain-c1:" + e).encode()).hexdigest(), 16) % 2
LAYERS = ["PROT_PROCAN", "PROT_CCLE", "PROT_HPA_IHC", "EXPR_HPA_SCT", "PPI_BIOPLEX", "PATH_PROGENY", "TIMING_HPA_CCD", "EXPR_DEPMAP", "DEP_DEPMAP",
          "EXPR_OPENCELL_HEK293T", "REG_COLLECTRI", "SETS_PATHWAY", "SETS_COMPLEX", "XCELL_K562GW"]


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "cellfix.py"), "rb").read())
    for m in (pp, pa):
        h.update(m.code_sha().encode())
    for n in LAYERS:
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def better(gain):
    g = gx.binom_greater(np.asarray(gain, float)); g["passed"] = bool(g["p"] < 0.05 and g["median"] >= 0.01); return g


def mc_p(real, nulls, greater=True):
    nulls = np.asarray(nulls, float); return float((1 + ((nulls >= real) if greater else (nulls <= real)).sum()) / (1 + len(nulls)))


def spearman_cols(A, B):
    """Spearman per column over rows finite in both."""
    from scipy.stats import rankdata
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        m = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        if m.sum() >= 50:
            a = rankdata(A[m, j]); b = rankdata(B[m, j]); out[j] = np.corrcoef(a, b)[0, 1]
    return out


def matrix(name, dedupe=True):
    z = gx.npz(name); X = z["X"].astype(np.float32); rows = z["rows"].astype(str); cols = z["cols"].astype(str)
    if dedupe:
        seen, keep = set(), []
        for i, g in enumerate(rows):
            if g not in seen:
                seen.add(g); keep.append(i)
        X, rows = X[keep], rows[keep]
    return X, rows, cols


def align2(X1, r1, c1, X2, r2, c2):
    rs = sorted(set(r1) & set(r2)); cs = sorted(set(c1) & set(c2))
    i1 = {g: i for i, g in enumerate(r1)}; i2 = {g: i for i, g in enumerate(r2)}; j1 = {c: j for j, c in enumerate(c1)}; j2 = {c: j for j, c in enumerate(c2)}
    return X1[np.ix_([i1[g] for g in rs], [j1[c] for c in cs])], X2[np.ix_([i2[g] for g in rs], [j2[c] for c in cs])], np.array(rs), np.array(cs)


# ------------------------------------------------------------------------------------------------ P1, P2, D1
def offsets(rng):
    Xp, rp, cp = matrix("PROT_PROCAN"); Xr, rr, cr = matrix("EXPR_DEPMAP")
    P, Rn, genes, lines = align2(Xp, rp, cp, Xr, rr, cr)
    dev = np.array([H_(c) == 0 for c in lines]); exm = ~dev
    o = np.nanmedian(np.where(np.isfinite(P[:, dev]), P[:, dev] - Rn[:, dev], np.nan), 1)
    return P, Rn, genes, lines, dev, exm, o


def run_p1(rng, OFF):
    P, Rn, genes, lines, dev, exm, o = OFF
    ok = np.isfinite(o); Pe, Re, oe = P[ok][:, exm], Rn[ok][:, exm], o[ok]
    r0 = spearman_cols(Re, Pe); r1 = spearman_cols(Re + oe[:, None], Pe)
    sh = rng.permutation(len(oe)); rn = spearman_cols(Re + oe[sh][:, None], Pe)
    resid = np.nanstd(Pe - (Re + oe[:, None]))
    Pp = np.where(np.isfinite(Pe), Re + oe[:, None] + rng.normal(0, resid, Pe.shape), np.nan); rp0 = spearman_cols(Re, Pp); rp1 = spearman_cols(Re + oe[:, None], Pp)
    real, neg, pl = better(r1 - r0), better(rn - r0), better(rp1 - rp0)
    out = dict(lines_dev=int(dev.sum()), lines_exam=int(exm.sum()), genes=int(ok.sum()), median_r_rna=float(np.nanmedian(r0)),
               median_r_rna_plus_offset=float(np.nanmedian(r1)), better=real, negative=neg, planted=pl)
    out["gate"] = dict(real=real["passed"], controls_ok=bool(not neg["passed"] and pl["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


def hpa_matched():
    S = gx.npz("EXPR_HPA_SCT"); X = S["X"].astype(np.float32); rows = S["rows"].astype(str); cols = S["cols"].astype(str)
    I = gx.npz("PROT_HPA_IHC"); norm = lambda x: str(x).lower().replace("-", " ").rstrip("s")
    ct = np.array([norm(c) for c in I["cell_type"]]); ok = I["reliability"] >= 1
    by = collections.defaultdict(dict)
    for g, c, l in zip(I["gene"][ok].astype(str), ct[ok], I["level"][ok]):
        if l > by[c].get(g, -1):
            by[c][g] = int(l)
    matched = [(j, c) for j, c in enumerate(cols) if norm(c) in by]
    return X, rows, cols, by, matched, norm


def run_p2(rng, OFF, HM):
    _, _, genes, _, _, _, o = OFF; X, rows, cols, by, matched, norm = HM
    og = dict(zip(genes, o)); ri = {g: i for i, g in enumerate(rows)}
    r0, r1, rn = [], [], []
    vals = np.array([v for v in og.values() if np.isfinite(v)]); sh = dict(zip([g for g, v in og.items() if np.isfinite(v)], rng.permutation(vals)))
    from scipy.stats import spearmanr
    for j, c in matched:
        lv = by[norm(c)]; gs = [g for g in lv if g in ri and g in og and np.isfinite(og[g])]
        x = np.log2(X[[ri[g] for g in gs], j] + 1); y = np.array([lv[g] for g in gs]); off = np.array([og[g] for g in gs]); offs = np.array([sh[g] for g in gs])
        r0.append(spearmanr(x, y)[0]); r1.append(spearmanr(x + off, y)[0]); rn.append(spearmanr(x + offs, y)[0])
    r0, r1, rn = map(np.array, (r0, r1, rn))
    real, neg = better(r1 - r0), better(rn - r0)
    out = dict(cell_types=[c for _, c in matched], median_r_rna=float(np.median(r0)), median_r_rna_plus_offset=float(np.median(r1)),
               per_cell_type={c: [round(float(a), 3), round(float(b), 3)] for (_, c), a, b in zip(matched, r0, r1)}, better=real, negative=neg)
    out["gate"] = dict(real=real["passed"], controls_ok=bool(not neg["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


def run_d1(rng, HM):
    X, rows, cols, by, matched, norm = HM; ri = {g: i for i, g in enumerate(rows)}
    flag, out_, strat, rates = [], [], [], {}

    def cats(x):
        rank = np.argsort(np.argsort(-x)); made = x >= 1
        return made & (rank >= 1500), ~made, made & (rank < 1500)
    data = []
    for k, (j, c) in enumerate(matched):
        lv = by[norm(c)]; gs = [g for g in lv if g in ri]; x = X[[ri[g] for g in gs], j]; det = np.array([lv[g] >= 1 for g in gs])
        data.append((x, det, k)); deep, notm, busy = cats(x)
        rates[c] = dict(busy=float(det[busy].mean()) if busy.any() else None, deep_only=float(det[deep].mean()) if deep.any() else None,
                        not_made=float(det[notm].mean()) if notm.any() else None, n_deep_only=int(deep.sum()))

    def odds(perm):
        F, Y, S_ = [], [], []
        for x, det, k in data:
            xx = x[rng.permutation(len(x))] if perm else x; deep, notm, _ = cats(xx); sel = deep | notm
            F.append(deep[sel]); Y.append(det[sel]); S_.append(np.full(int(sel.sum()), k))
        return gx.cmh(np.concatenate(F), np.concatenate(Y), np.concatenate(S_))
    OR, p = odds(False); nulls = [odds(True)[0] for _ in range(N200)]
    pm = mc_p(OR, nulls); neg = nulls[0]; neg_p = mc_p(neg, nulls[1:])
    out = dict(cell_types=len(matched), detection_rates=rates, mh_or=OR, cmh_p=p, null_or_95=float(np.percentile(nulls, 95)), p_mc=pm,
               passed=bool(OR >= 2 and p < 0.05 and pm < 0.05), negative=dict(or_=neg, p_mc=neg_p, passed=bool(neg >= 2 and neg_p < 0.05)))
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ I1
def run_i1(rng):
    import pandas as pd
    from scipy.stats import mannwhitneyu, rankdata
    B = gx.npz("PPI_BIOPLEX"); ln = B["line"].astype(str); a, b = B["a"].astype(str), B["b"].astype(str)
    m = pd.read_csv(os.path.join(HERE, "_cache", "paralog", "depmap", "raw", "Model.csv"), dtype=str)
    hct = m.ModelID[m.StrippedCellLineName == "HCT116"].tolist()[0]
    Xe, re_, ce = matrix("EXPR_DEPMAP"); j = list(ce).index(hct)
    oc = gx.npz("EXPR_OPENCELL_HEK293T"); okc = np.isfinite(oc["X"][:, 0].astype(float))
    h_rna = dict(zip(oc["rows"].astype(str)[okc], np.log2(oc["X"][:, 0].astype(float)[okc] + 1)))
    c_rna = dict(zip(re_, Xe[:, j]))
    genes = sorted(set(h_rna) & set(c_rna))
    hp = dict(zip(genes, rankdata([h_rna[g] for g in genes]) / len(genes))); cp = dict(zip(genes, rankdata([c_rna[g] for g in genes]) / len(genes)))
    nodes = {L_: set(a[ln == L_]) | set(b[ln == L_]) for L_ in ("HEK293T", "HCT116")}; both = nodes["HEK293T"] & nodes["HCT116"] & set(genes)
    prs = {L_: {frozenset(p) for p in zip(a[ln == L_], b[ln == L_]) if all(x in both for x in p)} for L_ in ("HEK293T", "HCT116")}
    only_h, only_c, shared = prs["HEK293T"] - prs["HCT116"], prs["HCT116"] - prs["HEK293T"], prs["HEK293T"] & prs["HCT116"]
    dl = lambda p: min(cp[x] for x in p) - min(hp[x] for x in p)
    dh = np.array([dl(p) for p in only_h]); dc = np.array([dl(p) for p in only_c]); ds = np.array([dl(p) for p in shared])
    real = float(np.median(dc) - np.median(dh)); p_mw = float(mannwhitneyu(dc, dh, alternative="greater").pvalue)
    allv = np.concatenate([dc, dh]); nc = len(dc); nulls = []
    for _ in range(N1000):
        q = rng.permutation(allv); nulls.append(float(np.median(q[:nc]) - np.median(q[nc:])))
    pm = mc_p(real, nulls); neg = nulls[0]
    out = dict(pairs_hek293t_only=len(dh), pairs_hct116_only=len(dc), pairs_shared=len(ds), median_delta_hct116_only=float(np.median(dc)),
               median_delta_hek293t_only=float(np.median(dh)), median_delta_shared=float(np.median(ds)), diff=real, p_mannwhitney=p_mw,
               null_95=float(np.percentile(nulls, 95)), p_mc=pm, passed=bool(real >= 0.02 and p_mw < 0.05 and pm < 0.05),
               negative=dict(diff=neg, passed=bool(neg >= 0.02 and mc_p(neg, nulls[1:]) < 0.05)),
               planted=dict(diff=real + 0.05, passed=bool(real + 0.05 >= 0.02 and mc_p(real + 0.05, nulls) < 0.05)))
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ A1
def regulons(rows_set, min_n=10):
    CT = gx.npz("REG_COLLECTRI"); reg = collections.defaultdict(list)
    for t, g, s in zip(CT["tf"].astype(str), CT["target"].astype(str), CT["sign"]):
        if s != 0 and g in rows_set and g != t:
            reg[t].append((g, int(s)))
    return {t: v for t, v in reg.items() if len(v) >= min_n}


def reg_matrix(reg, ri):
    from scipy import sparse
    tfs = sorted(reg); rr, cc, vv = [], [], []
    for i, t in enumerate(tfs):
        for g, s in reg[t]:
            rr.append(i); cc.append(ri[g]); vv.append(s / len(reg[t]))
    return tfs, sparse.csr_matrix((vv, (rr, cc)), shape=(len(tfs), len(ri)))


def run_a1(rng):
    T = gx.tss(); keep = None
    if TRIAL:
        keep = set(sorted({str(e) for e in gx.npz("XCELL_K562GW")["cols_ens"] if str(e)})[::7])
    G = gx.load("K562GW", T, keep); rows = list(G["rows"]); ri = {g: i for i, g in enumerate(rows)}
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.nan_to_num(G["lfc"] / G["se"]).astype(np.float32)
    reg = regulons(set(rows)); tfs, M = reg_matrix(reg, ri)
    kd = [t for t in tfs if t in G["ci"]]; kc = np.array([G["ci"][t] for t in kd]); ti = {t: i for i, t in enumerate(tfs)}
    def auroc(Mx):
        A = np.asarray(Mx.dot(Z[:, kc]))                                         # TFs x knockdowns
        own = np.array([A[ti[t], j] for j, t in enumerate(kd)])
        return float(np.mean([(A[:, j] > own[j]).sum() / (A.shape[0] - 1) for j in range(len(kd))])), A, own
    real, A, own = auroc(M)
    genes_all = np.arange(len(rows)); nulls = []
    for b in range(N200):
        rnd = {t: [(rows[g], s) for g, (_, s) in zip(rng.choice(genes_all, len(v), replace=False), v)] for t, v in reg.items()}
        _, Mn = reg_matrix(rnd, ri); nulls.append(auroc(Mn)[0])
    pm = mc_p(real, nulls); neg = nulls[0]
    own_pl = np.full(len(kd), -3.0); pl = float(np.mean([(A[:, j] > own_pl[j]).sum() / (A.shape[0] - 1) for j in range(len(kd))]))
    H = cbb.hgnc(); sym = lambda e: H["e2s"].get(e, e); rk = [(float((A[:, j] > own[j]).sum() / (A.shape[0] - 1)), sym(t)) for j, t in enumerate(kd)]
    out = dict(tfs_with_regulons=len(tfs), knocked_down_tfs=len(kd), mean_auroc=real, null_mean=float(np.mean(nulls)), null_95=float(np.percentile(nulls, 95)),
               p_mc=pm, passed=bool(real >= 0.60 and pm < 0.05), negative=dict(auroc=neg, passed=bool(neg >= 0.60 and mc_p(neg, nulls[1:]) < 0.05)),
               planted=dict(auroc=pl, passed=bool(pl >= 0.60 and mc_p(pl, nulls) < 0.05)),
               best_detected=[dict(tf=s, auroc=round(v, 3)) for v, s in sorted(rk, reverse=True)[:15]])
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ U1
def evidences():
    """per evidence: rows (genes) and row-standardised features (NaN -> 0) so that dot / n = correlation."""
    ev = {}

    def std(X):
        mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True); Zs = np.nan_to_num((X - mu) / np.where(sd > 0, sd, np.nan))
        n = np.isfinite(X).sum(1); return Zs.astype(np.float32), n
    X, r, _ = matrix("DEP_DEPMAP"); ev["depmap"] = (r, *std(X))
    X, r, _ = matrix("PROT_PROCAN"); ok = np.isfinite(X).sum(1) >= 100; ev["procan"] = (r[ok], *std(X[ok]))
    X, r, _ = matrix("EXPR_HPA_SCT"); X = np.log2(X + 1); ok = (X > 0).sum(1) >= 5; ev["hpa_rna"] = (r[ok], *std(X[ok]))
    z = gx.npz("XCELL_K562GW"); ens = z["cols_ens"].astype(str); lfc = z["lfc"].astype(np.float32); se = z["se"].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = (lfc / se).T                                                                 # knockdowns x measured genes
    seen, keep = set(), []
    for i, e in enumerate(ens):
        if e and e not in seen:
            seen.add(e); keep.append(i)
    ev["k562_knockdown"] = (ens[keep], *std(K[keep]))
    return ev


def combined_sim(ev, A, B):
    """genes A x genes B: mean Fisher-z over evidences both have (NaN where none)."""
    S = np.zeros((len(A), len(B)), np.float32); C = np.zeros((len(A), len(B)), np.float32)
    for name, (rows, Zs, n) in ev.items():
        ri = {g: i for i, g in enumerate(rows)}
        ia = np.array([ri.get(g, -1) for g in A]); ib = np.array([ri.get(g, -1) for g in B]); ma, mb = ia >= 0, ib >= 0
        if not ma.any() or not mb.any():
            continue
        r = (Zs[ia[ma]] @ Zs[ib[mb]].T) / Zs.shape[1]
        S[np.ix_(ma, mb)] += np.arctanh(np.clip(r, -0.999, 0.999)); C[np.ix_(ma, mb)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(C > 0, S / C, np.nan), C


def pathway_scores(sim, members_idx, k=5):
    sub = np.nan_to_num(sim[:, members_idx], nan=-9)
    kk = min(k, sub.shape[1]); top = -np.partition(-sub, kk - 1, axis=1)[:, :kk]
    return top.mean(1)


def run_u1(rng):
    from scipy.stats import rankdata
    ev = evidences(); pw = {k: (v["name"], v["members"]) for k, v in json.load(open(gx.L("SETS_PATHWAY"))).items()}
    have = collections.Counter()
    for name, (rows, _, _) in ev.items():
        for g in rows:
            have[g] += 1
    annotated = {g for _, m in pw.values() for g in m}
    genes = sorted(g for g in annotated if have[g] >= 2)
    if TRIAL:
        genes = genes[::4]
    dev = [g for g in genes if H_(g) == 0]; exm = [g for g in genes if H_(g) == 1]; di = {g: i for i, g in enumerate(dev)}; ei = {g: i for i, g in enumerate(exm)}
    sim, _ = combined_sim(ev, exm, dev)
    aucs, ranks, pos = [], [], []
    names = []
    for k, (nm, m) in pw.items():
        if not (10 <= len(m) <= 300):
            continue
        md = [di[g] for g in m if g in di]; me = np.array([ei[g] for g in m if g in ei])
        if len(md) < 5 or len(me) < 5:
            continue
        s = pathway_scores(sim, np.array(md)); rk = rankdata(s)
        n1 = len(me); n0 = len(exm) - n1; auc = (rk[me].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        aucs.append(auc); ranks.append(rk); pos.append(n1); names.append(nm)
    aucs = np.array(aucs); real = float(np.median(aucs)); nulls = []; nE = len(exm)
    for _ in range(N200):
        a_ = [(rk[rng.choice(nE, n1, replace=False)].sum() - n1 * (n1 + 1) / 2) / (n1 * (nE - n1)) for rk, n1 in zip(ranks, pos)]
        nulls.append(float(np.median(a_)))
    pm = mc_p(real, nulls); neg = nulls[0]
    pl = float(np.median([(np.sort(rk)[-n1:].sum() - n1 * (n1 + 1) / 2) / (n1 * (nE - n1)) for rk, n1 in zip(ranks, pos)]))
    o = np.argsort(-aucs)
    out = dict(genes_dev=len(dev), genes_exam=len(exm), pathways=len(aucs), median_auroc=real, share_auroc_ge_0_7=float((aucs >= 0.7).mean()),
               null_median=float(np.median(nulls)), null_95=float(np.percentile(nulls, 95)), p_mc=pm, passed=bool(real >= 0.70 and pm < 0.05),
               negative=dict(median_auroc=neg, passed=bool(neg >= 0.70 and mc_p(neg, nulls[1:]) < 0.05)), planted=dict(median_auroc=pl, passed=bool(pl >= 0.70 and mc_p(pl, nulls) < 0.05)),
               best_predicted=[dict(pathway=names[i], auroc=round(float(aucs[i]), 3)) for i in o[:12]],
               worst_predicted=[dict(pathway=names[i], auroc=round(float(aucs[i]), 3)) for i in o[-6:]])
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ M1, M2
def complexes(min_parts=3, max_parts=60):
    return [(c, m) for c, d in json.load(open(gx.L("SETS_COMPLEX"))).items() for m in [d["members"]] if min_parts <= len(m) <= max_parts]


def run_m1(rng):
    Xp, rp, cp = matrix("PROT_PROCAN"); Xd, rd, cd = matrix("DEP_DEPMAP")
    lines = sorted(set(cp) & set(cd)); jp = {c: j for j, c in enumerate(cp)}; jd = {c: j for j, c in enumerate(cd)}
    P = Xp[:, [jp[c] for c in lines]]; Dm = Xd[:, [jd[c] for c in lines]]; MADE = np.isfinite(P)
    pi = {g: i for i, g in enumerate(rp)}; di = {g: i for i, g in enumerate(rd)}
    dec = pa.deciles(MADE.mean(1)); pool = {d: np.nonzero(dec == d)[0] for d in range(10)}
    comps = [(c, [x for x in m if x in pi and x in di]) for c, m in complexes()]; comps = [(c, m) for c, m in comps if len(m) >= 3]
    if TRIAL:
        comps = comps[:120]

    def dvals(stand):
        out = []
        for (c, mm), Mr in zip(comps, stand):
            for a, x in enumerate(mm):
                aok = MADE[pi[x]]; others = np.delete(Mr, a, axis=0).all(0); cl, il = aok & others, aok & ~others
                if cl.sum() >= 20 and il.sum() >= 20:
                    e = Dm[di[x]]; out.append(float(np.nanmean(e[cl]) - np.nanmean(e[il])))
                else:
                    out.append(np.nan)
        return np.array(out)
    d = dvals([MADE[[pi[x] for x in mm]] for _, mm in comps]); v = d[np.isfinite(d)]; nz = v[v != 0]
    from scipy.stats import binomtest
    real = float(np.median(v)) if len(v) else float("nan")
    ps = float(binomtest(int((nz < 0).sum()), int(len(nz)), 0.5, alternative="greater").pvalue) if len(nz) else 1.0
    nulls = []
    for b in range(N200):
        st = [MADE[[pool[dec[pi[x]]][rng.integers(0, len(pool[dec[pi[x]]]))] for x in mm]] for _, mm in comps]
        dn = dvals(st); nulls.append(float(np.nanmedian(dn)))
        if b == 0:
            v0 = dn[np.isfinite(dn)]
    pm = mc_p(real, nulls, greater=False); neg = float(np.median(v0)) if len(v0) else float("nan")
    out = dict(lines=len(lines), complexes=len(comps), items=int(len(v)), median_d=real, share_negative=float((nz < 0).mean()) if len(nz) else None, p_sign=ps,
               null_median=float(np.median(nulls)), p_mc=pm, passed=bool(real <= -0.02 and ps < 0.05 and pm < 0.05),
               negative=dict(median_d=neg, passed=bool(neg <= -0.02 and mc_p(neg, nulls[1:], greater=False) < 0.05)),
               planted=dict(median_d=real - 0.05, passed=bool(real - 0.05 <= -0.02 and mc_p(real - 0.05, nulls, greater=False) < 0.05)))
    out["gate"] = dict(real=out["passed"], controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out


def run_m2(rng):
    Xp, rp, cp = matrix("PROT_PROCAN"); Xr, rr, cr = matrix("EXPR_DEPMAP")
    P, Rn, genes, lines = align2(Xp, rp, cp, Xr, rr, cr); gi = {g: i for i, g in enumerate(genes)}

    def zrow(X):
        mu = np.nanmean(X, 1, keepdims=True); sd = np.nanstd(X, 1, keepdims=True); return np.nan_to_num((X - mu) / np.where(sd > 0, sd, np.nan)), np.isfinite(X)
    ZP, FP = zrow(P); ZR, FR = zrow(Rn)

    def med_corr(Z, F, idx):
        """median pairwise correlation (rows z-scored over each gene's measured lines; pairs with >= 30 shared lines)."""
        idx = np.asarray(idx); Ff = F[idx].astype(np.float32); Zs = Z[idx] * Ff
        num = Zs @ Zs.T; cnt = Ff @ Ff.T; iu = np.triu_indices(len(idx), 1); ok = cnt[iu] >= 30
        return float(np.median(num[iu][ok] / cnt[iu][ok])) if ok.any() else np.nan
    comps = [(c, [gi[x] for x in m if x in gi]) for c, m in complexes()]; comps = [(c, idx) for c, idx in comps if len(idx) >= 3]
    if TRIAL:
        comps = comps[:80]
    cp_, cr_ = np.array([med_corr(ZP, FP, idx) for _, idx in comps]), np.array([med_corr(ZR, FR, idx) for _, idx in comps])
    g = better(cp_ - cr_); real = float(np.nanmedian(cp_))
    dec = pa.deciles(FP.mean(1)); pool = {d: np.nonzero(dec == d)[0] for d in range(10)}; nulls = []
    for _ in range(N200 if not TRIAL else 5):
        nulls.append(float(np.nanmedian([med_corr(ZP, FP, [pool[dec[i]][rng.integers(0, len(pool[dec[i]]))] for i in idx]) for _, idx in comps])))
    pm = mc_p(real, nulls)
    out = dict(complexes=len(comps), lines=len(lines), median_protein_covariation=real, median_rna_covariation=float(np.nanmedian(cr_)),
               protein_minus_rna=g, random_protein_median=float(np.median(nulls)), p_mc=pm, passed=bool(g["passed"] and pm < 0.05))
    out["gate"] = dict(real=out["passed"], controls_ok=True, verdict="PASS" if out["passed"] else "FAIL")
    return out


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G8-")[0].split("Limit 4")[0].strip()[:1200], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G8 already tested")
    t0 = time.time(); rng = np.random.default_rng(SEED); out = {}
    OFF = offsets(rng); HM = hpa_matched()
    for cid, fn in (("G8-P1", lambda: run_p1(rng, OFF)), ("G8-P2", lambda: run_p2(rng, OFF, HM)), ("G8-D1", lambda: run_d1(rng, HM)),
                    ("G8-I1", lambda: run_i1(rng)), ("G8-A1", lambda: run_a1(rng)), ("G8-U1", lambda: run_u1(rng)),
                    ("G8-M1", lambda: run_m1(rng)), ("G8-M2", lambda: run_m2(rng))):
        out[cid] = fn(); print(cid, "done", f"({time.time() - t0:.0f}s)", flush=True)
    if TRIAL:
        print("trial controls ok:", {c: out[c]["gate"]["controls_ok"] for c in CLAIMS}, "; real outcomes not shown"); return
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "fix_tests.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(cbb.LAY, "protein_offset_procan.npz"), genes=OFF[2], offset=OFF[6].astype(np.float32))
    t = cbb.now(); src = {"name": "cellfix:G8", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "fix_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], json.dumps({k: v for k, v in out[cid].items() if not isinstance(v, (dict, list))}, default=float)[:600], flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
