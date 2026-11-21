"""GENEREGX -- how one gene's influence on another changes with cell type, and which gene influences which, at what level.

Asked by the user: "we have 2 major goals with this type of comparison: how influence of one gene on other changes with
cell type, and which gene influences which and at what level."
Data (cellbrain bank, sha-checked layers): the same ~2,400 CRISPRi knockdowns read out gene by gene in four cell types --
K562 (blood cancer), RPE1 (retina epithelium), HepG2 (liver cancer), Jurkat (T-cell leukaemia) -- and the K562
genome-wide screen (~10,000 knockdowns; an independent experiment in K562), all as DESeq2 log2 fold change, lfcSE and p
from one pipeline (Nadig 2025, CC BY 4.0; layers XCELL_*). HPA cell-line RNA (EXPR_HPA_CELLLINES). CollecTRI curated
TF -> target interactions (REG_COLLECTRI, local only). HPA transcription factors. GENCODE 46 TSS.

LINK r -> t (knocking down gene r changes gene t) in a screen: BH q < 0.05 over every pair measured in that screen, on
the Wald p of lfc / lfcSE (data check before any claim: in the four essential screens the published p equals that Wald p,
median ratio 1.00; in the K562 genome-wide file it does not -- no p above 0.32, |lfc / lfcSE| a median 0.43 of the z
its p implies -- so one rule, Wald from lfc and lfcSE, is used for all five);
the knocked-down gene itself and genes whose TSS is within 50 kb of its TSS are excluded (CRISPRi reaches neighbours).
Knockdown lowers t -> r ACTIVATES t; raises t -> r REPRESSES t. OWN KNOCKDOWN of r = log2 fold change of r's own mRNA
in its knockdown (usable when measured and <= -0.5).

CLAIMS (declared before running; rem/atlas/cellbrain_ledger.jsonl, ids G2-*; the brain's rules):
 G2-X1  INFLUENCE CHANGES WITH CELL TYPE, BEYOND EXPERIMENT NOISE. Knockdowns in all five screens; targets measured
        in all five. Per knockdown, agreement of two screens = Pearson r of its target log2 fold changes, corrected for
        noise: rho = r / sqrt(rel_A x rel_B), rel = 1 - mean(lfcSE^2) / var(lfc) over the same targets. Knockdowns
        with rel >= 0.2 in all five screens. Reference: rho(K562 essential, K562 genome-wide), same cell type, two
        experiments. CLAIM, for each of RPE1, HepG2, Jurkat: gap = rho_ref - rho(K562 essential, that cell type) is
        positive (one-sided sign test over knockdowns p < 0.05) with median gap >= 0.05. Negative (must not pass): the
        other cell type replaced by K562 genome-wide plus noise drawn from that cell type's lfcSE. Planted (must
        pass): the same, with half of each knockdown's target effects shuffled.
 G2-X2  A GENE INFLUENCES MORE WHERE IT IS MORE EXPRESSED. Reach of a knockdown in a cell type = share of measured
        genes it changes (links); expression = HPA cell-line RNA (K-562, hTERT-RPE1, Hep-G2, mean of the two Jurkat
        lines; log1p nTPM). Both as percentiles among the knockdowns common to the four essential screens, within each
        cell type. Knockdowns whose expression varies (range of log1p nTPM >= 1). CLAIM: mean over knockdowns of the
        Spearman correlation across the four cell types between expression and reach >= 0.10, permutation p < 0.05
        (expression's cell-type labels permuted within each knockdown, 2,000 draws). Negative (must not pass): each
        knockdown given another knockdown's expression. Planted (must pass): reach := expression + noise (sd 0.2).
 G2-L1  AT WHAT LEVEL -- DOSE. Links significant with the same sign in >= 2 of the four cell types where the
        knockdown's own knockdown is usable. Per link, across those cell types: d_dose = SD of log2(|effect| /
        |own knockdown|); d_rand = the same with the own knockdown of one random other knockdown usable in those cell
        types (keeps each cell type's overall knockdown strength). CLAIM (BETTER rule): gain = d_rand - d_dose, sign
        test p < 0.05 and median gain >= +0.01 (up to 300,000 links drawn at random, seeded), i.e. a deeper knockdown
        of the same gene gives a bigger effect, so "effect per unit of knockdown" is the transferable level.
        Negative (must not pass): both arms random knockdowns. Planted (must pass): effect := own knockdown x the
        link's mean ratio x noise (sd 0.3 log2).
 G2-L2  DIRECT OR THROUGH ANOTHER GENE (K562 genome-wide). For a transcription factor r (HPA) with CollecTRI targets
        measured, each link r -> t is INDIRECT if some mediator m explains it: m knocked down in the screen (own
        knockdown usable), r -> m and m -> t both links, and the predicted effect lfc(m | r) x lfc(t | m) / own(m) has
        the same sign as lfc(t | r) and >= half its size; otherwise DIRECT. CLAIM: direct links are more often curated
        TF -> target interactions (CollecTRI) than indirect ones, Mantel-Haenszel odds ratio over deciles of |lfc/lfcSE|
        >= 1.5, CMH p < 0.05. Negative (must stay within 0.67-1.5): mediators' effects on targets shuffled between
        mediators. Planted (>= 90%): 200 links set to a mediator's predicted effect are called indirect.
DESCRIPTIVE: per link, the pattern across the four cell types (everywhere / shared / one cell type only / flips sign /
only measurable in one) with the K562 two-experiment flip rate as the noise baseline; per knockdown, reach in each
cell type with its expression; per link, strength (lfc, z), level (effect per unit of own knockdown) and, in K562,
direct or via which mediator; a TF hierarchy from direct TF -> TF links. All written to the bank.
NOT: four cell lines, not the body's cell types; the knockdowns are essential genes (few cell-type-specific TFs);
HepG2 and Jurkat were screened later by another team, so their differences include lab and protocol, which RPE1 (same
lab and time as K562) does not; DESeq2 calls many small effects; "direct" means "not explained by a measured
mediator", not proof of binding (mediators not knocked down in the screen cannot be seen).

Run: python3 -I rem/atlas/generegx.py declare | test [--trial]
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
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("cellbrain_bank", os.path.join(HERE, "cellbrain_bank.py"))
cbb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cbb)
from brain.bank import Fact  # noqa: E402

cbb.SRC["nadig2025_de"] = ("Nadig et al. 2025 Nat Genet: transcriptome-wide differential expression for K562, RPE1, HepG2, Jurkat essential and "
                           "K562 genome-wide CRISPRi Perturb-seq (TRADE; figshare 29498366)", "https://doi.org/10.6084/m9.figshare.29498366", "CC BY 4.0")

TRIAL = "--trial" in sys.argv
LEDGER = os.path.join(cbb.HOME, "trial_ledger.jsonl") if TRIAL else os.path.join(HERE, "cellbrain_ledger.jsonl")
RUN = os.path.join(cbb.HOME, "generegx_trial" if TRIAL else "generegx")
OUTD = os.path.join(RUN, "out") if TRIAL else R("outputs", "genereg")
RES = os.path.join(RUN, "RESULTS.txt") if TRIAL else os.path.join(HERE, "RESULTS_generegx.txt")
CLAIMS = ["G2-X1", "G2-X2", "G2-L1", "G2-L2"]
ESS = ["K562ESS", "RPE1ESS", "HEPG2ESS", "JURKATESS"]
GW = "K562GW"
CELL = {"K562ESS": "K562", "RPE1ESS": "RPE1", "HEPG2ESS": "HepG2", "JURKATESS": "Jurkat", "K562GW": "K562 genome-wide"}
HPA = {"K562ESS": ["K-562"], "RPE1ESS": ["hTERT-RPE1"], "HEPG2ESS": ["Hep-G2"], "JURKATESS": ["JURKAT", "Jurkat E6.1"]}
ALPHA, CIS, KD_MIN, REL_MIN, NPERM, SEED, L1_MAX = 0.05, 50_000, -0.5, 0.2, 2000, 20261121, 300_000
TRIAL_GENES = 700
USED = set()


def L(name):
    f, p = cbb.layer(name); USED.add(f.ref); return p


def npz(name):
    return np.load(L(name), allow_pickle=False)


def code_sha():
    h = hashlib.sha256()
    for f in ("generegx.py", "cellbrain_bank.py", "xcellbank.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    B = cbb.bank()
    for n in [f"XCELL_{s}" for s in ESS + [GW]] + ["REG_COLLECTRI", "EXPR_HPA_CELLLINES", "TSS_GENCODE46", "HPA_PROTEINATLAS"]:
        h.update(B.facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def sym_map():
    H = cbb.hgnc(); return lambda e: H["e2s"].get(e, e)


# ------------------------------------------------------------------------------------------------ screens
def bh_mask(nlp):
    """BH q < ALPHA over every finite pair, on -log10 p."""
    v = nlp[np.isfinite(nlp)]; m = v.size
    if not m:
        return np.zeros(nlp.shape, bool)
    s = np.sort(v)[::-1]
    ok = s >= -np.log10(ALPHA * np.arange(1, m + 1, dtype=np.float64) / m)
    if not ok.any():
        return np.zeros(nlp.shape, bool)
    thr = s[np.nonzero(ok)[0].max()]
    return np.isfinite(nlp) & (nlp >= thr)


def tss():
    t = json.load(open(L("TSS_GENCODE46"))); return {g[0]: (g[3], g[4]) for g in t}


def load(s, T, keep=None):
    z = npz(f"XCELL_{s}")
    rows = z["rows"].astype(str); ens = z["cols_ens"].astype(str); seen, kc = set(), []
    for j, e in enumerate(ens):
        if e and e not in seen and (keep is None or e in keep):
            seen.add(e); kc.append(j)
    kc = np.array(kc); cols = ens[kc]
    lfc = z["lfc"][:, kc].astype(np.float32); se = z["se"][:, kc].astype(np.float32)
    from scipy.special import log_ndtr
    nlp = np.full(lfc.shape, np.nan, np.float32)
    for a in range(0, lfc.shape[1], 500):          # two-sided Wald p of lfc / lfcSE, as -log10 p
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = np.abs(lfc[:, a:a + 500].astype(np.float64) / se[:, a:a + 500])
        ok = np.isfinite(zz) & (se[:, a:a + 500] > 0)
        nlp[:, a:a + 500] = np.where(ok, -(log_ndtr(-np.where(ok, zz, 0)) + np.log(2)) / np.log(10), np.nan)
    ri = {g: i for i, g in enumerate(rows)}
    own = np.array([lfc[ri[c], j] if c in ri else np.nan for j, c in enumerate(cols)], np.float32)
    sig = bh_mask(nlp); del nlp
    # cis: the knocked-down gene and genes with TSS within 50 kb of its TSS
    by = collections.defaultdict(list)
    for i, g in enumerate(rows):
        if g in T:
            by[T[g][0]].append((T[g][1], i))
    by = {c: (np.array([p for p, _ in v]), np.array([i for _, i in v])) for c, v in by.items()}
    ncis = 0
    for j, c in enumerate(cols):
        idx = [ri[c]] if c in ri else []
        if c in T and T[c][0] in by:
            pos, ii = by[T[c][0]]; idx += list(ii[np.abs(pos - T[c][1]) <= CIS])
        if idx:
            lfc[idx, j] = np.nan; se[idx, j] = np.nan; sig[idx, j] = False; ncis += len(idx)
    print(f"  {s}: {len(rows):,} genes x {len(cols):,} knockdowns, links {int(sig.sum()):,}, cis/self masked {ncis:,}", flush=True)
    return dict(name=s, rows=rows, cols=cols, lfc=lfc, se=se, sig=sig, own=own, ri=ri, ci={c: j for j, c in enumerate(cols)})


def align(S, rows, cols, what):
    """S[what] on the given rows x cols (NaN / False where the screen lacks them)."""
    a = S[what]; fill = False if a.dtype == bool else np.nan
    out = np.full((len(rows), len(cols)), fill, dtype=a.dtype)
    ri = np.array([S["ri"].get(g, -1) for g in rows]); ci = np.array([S["ci"].get(c, -1) for c in cols])
    r_ok, c_ok = ri >= 0, ci >= 0
    out[np.ix_(r_ok, c_ok)] = a[np.ix_(ri[r_ok], ci[c_ok])]
    return out


def binom_greater(x):
    from scipy.stats import binomtest
    x = x[np.isfinite(x)]; nz = x[x != 0]
    p = float(binomtest(int((nz > 0).sum()), int(len(nz)), 0.5, alternative="greater").pvalue) if len(nz) else 1.0
    return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else float("nan"), share_positive=float((nz > 0).mean()) if len(nz) else float("nan"), p=p)


# ------------------------------------------------------------------------------------------------ X1
def colstats(a, b, va, vb):
    am = a - np.nanmean(a, 0); bm = b - np.nanmean(b, 0)
    r = np.nansum(am * bm, 0) / np.sqrt(np.nansum(am ** 2, 0) * np.nansum(bm ** 2, 0))
    rel = lambda x, v: np.clip(1 - np.nanmean(v, 0) / np.nanvar(x, 0), 0, None)
    return r, rel(a, va), rel(b, vb)


def run_x1(SS, rng):
    scr = ESS + [GW]
    T_ = sorted(set.intersection(*(set(SS[s]["rows"]) for s in scr))); K = sorted(set.intersection(*(set(SS[s]["cols"]) for s in scr)))
    A = {s: align(SS[s], T_, K, "lfc") for s in scr}; V = {s: align(SS[s], T_, K, "se") ** 2 for s in scr}
    F = np.ones((len(T_), len(K)), bool)
    for s in scr:
        F &= np.isfinite(A[s]) & np.isfinite(V[s])
    for s in scr:
        A[s][~F] = np.nan; V[s][~F] = np.nan
    rels = {s: np.clip(1 - np.nanmean(V[s], 0) / np.nanvar(A[s], 0), 0, None) for s in scr}
    elig = np.all([rels[s] >= REL_MIN for s in scr], 0)
    ref = "K562ESS"

    def rho(a, b, va, vb):
        r, ra, rb = colstats(a[:, elig], b[:, elig], va[:, elig], vb[:, elig])
        with np.errstate(invalid="ignore", divide="ignore"):
            return r / np.sqrt(ra * rb), r
    rho_ref, r_ref = rho(A[ref], A[GW], V[ref], V[GW])
    out = dict(knockdowns_common=len(K), targets_common=len(T_), knockdowns_eligible=int(elig.sum()),
               rho_ref_median=float(np.nanmedian(rho_ref)), r_ref_median=float(np.nanmedian(r_ref)), per_cell={}, negative={}, planted={})
    for s in ESS[1:]:
        rx, rr = rho(A[ref], A[s], V[ref], V[s]); g = binom_greater(rho_ref - rx)
        out["per_cell"][CELL[s]] = dict(rho_median=float(np.nanmedian(rx)), r_median=float(np.nanmedian(rr)), gap=g,
                                        passed=bool(g["p"] < 0.05 and g["median"] >= 0.05))
        noise = rng.standard_normal(A[GW].shape).astype(np.float32) * np.sqrt(V[s])
        Xn = A[GW] + noise; Vn = V[GW] + V[s]
        rn, _ = rho(A[ref], Xn, V[ref], Vn); gn = binom_greater(rho_ref - rn)
        out["negative"][CELL[s]] = dict(gap=gn, passed=bool(gn["p"] < 0.05 and gn["median"] >= 0.05))
        Xp = A[GW].copy()
        for j in range(Xp.shape[1]):
            ok = np.nonzero(np.isfinite(Xp[:, j]))[0]; h = rng.choice(ok, len(ok) // 2, replace=False); Xp[h, j] = Xp[rng.permutation(h), j]
        Xp = Xp + noise
        rp, _ = rho(A[ref], Xp, V[ref], Vn); gp = binom_greater(rho_ref - rp)
        out["planted"][CELL[s]] = dict(gap=gp, passed=bool(gp["p"] < 0.05 and gp["median"] >= 0.05))
    # descriptive: median rho for every pair of screens
    M = {}
    for i, a in enumerate(scr):
        for b in scr[i + 1:]:
            rx, _ = rho(A[a], A[b], V[a], V[b]); M[f"{CELL[a]} ~ {CELL[b]}"] = float(np.nanmedian(rx))
    out["rho_matrix"] = M
    ok_real = all(v["passed"] for v in out["per_cell"].values())
    ok_ctrl = (not any(v["passed"] for v in out["negative"].values())) and all(v["passed"] for v in out["planted"].values())
    out["gate"] = dict(real=ok_real, controls_ok=ok_ctrl, verdict="PASS" if ok_real and ok_ctrl else "FAIL")
    per_kd = {K[j]: dict(rel={CELL[s]: float(rels[s][j]) for s in scr}) for j in range(len(K))}
    for s in ESS[1:]:
        rx, _ = rho(A[ref], A[s], V[ref], V[s])
        for j, v in zip(np.nonzero(elig)[0], rx):
            per_kd[K[j]].setdefault("rho_vs_K562", {})[CELL[s]] = None if not np.isfinite(v) else float(v)
    for j, v in zip(np.nonzero(elig)[0], rho_ref):
        per_kd[K[j]].setdefault("rho_vs_K562", {})["K562 genome-wide"] = None if not np.isfinite(v) else float(v)
    return out, per_kd


# ------------------------------------------------------------------------------------------------ X2
def hpa_expr(K):
    z = npz("EXPR_HPA_CELLLINES"); rows = {str(g): i for i, g in enumerate(z["rows"])}; cols = [str(c) for c in z["cols"]]
    X = z["X"]; E = np.full((len(K), len(ESS)), np.nan, np.float32)
    for k, s in enumerate(ESS):
        ci = [cols.index(c) for c in HPA[s]]
        for i, g in enumerate(K):
            if g in rows:
                E[i, k] = float(np.mean(X[rows[g], ci].astype(np.float32)))
    return E


def pct(M):
    from scipy.stats import rankdata
    return np.column_stack([rankdata(M[:, k]) / M.shape[0] for k in range(M.shape[1])])


def rowspear(a, b):
    from scipy.stats import rankdata
    ra = rankdata(a, axis=1); rb = rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


def x2_stat(Ep, Rp, var, rng, nperm=NPERM):
    rs = rowspear(Ep[var], Rp[var]); ok = np.isfinite(rs); S = float(rs[ok].mean())
    from scipy.stats import rankdata
    ra = rankdata(Ep[var][ok], axis=1); rb = rankdata(Rp[var][ok], axis=1)
    rb = rb - rb.mean(1, keepdims=True); nb = (rb ** 2).sum(1)
    ge = 0
    for _ in range(nperm):
        perm = np.argsort(rng.random(ra.shape), axis=1); x = np.take_along_axis(ra, perm, 1); x = x - x.mean(1, keepdims=True)
        sp = (x * rb).sum(1) / np.sqrt((x ** 2).sum(1) * nb)
        ge += sp.mean() >= S
    return dict(n=int(ok.sum()), mean_spearman=S, p=(1 + ge) / (1 + nperm), passed=bool(S >= 0.10 and (1 + ge) / (1 + nperm) < 0.05))


def run_x2(SS, K4, rng):
    reach = np.column_stack([SS[s]["sig"][:, [SS[s]["ci"][k] for k in K4]].sum(0) /
                             np.maximum(1, np.isfinite(SS[s]["lfc"][:, [SS[s]["ci"][k] for k in K4]]).sum(0)) for s in ESS])
    E = hpa_expr(K4); has = np.all(np.isfinite(E), 1)
    Kx = [k for k, h in zip(K4, has) if h]; E, Rch = E[has], reach[has]
    Ep, Rp = pct(E), pct(Rch); var = (E.max(1) - E.min(1)) >= 1.0
    real = x2_stat(Ep, Rp, var, rng)
    sh = rng.permutation(len(Kx)); Es = E[sh]; neg = x2_stat(pct(Es), Rp, (Es.max(1) - Es.min(1)) >= 1.0, rng)
    plant = x2_stat(Ep, np.clip(Ep + rng.normal(0, 0.2, Ep.shape), 0, 1), var, rng)
    out = dict(knockdowns=len(Kx), variable=int(var.sum()), real=real, negative=neg, planted=plant,
               gate=dict(controls_ok=bool(not neg["passed"] and plant["passed"])))
    out["gate"]["verdict"] = "PASS" if real["passed"] and out["gate"]["controls_ok"] else "FAIL"
    per = {k: dict(reach={CELL[s]: float(Rch[i, j]) for j, s in enumerate(ESS)}, expression={CELL[s]: float(E[i, j]) for j, s in enumerate(ESS)})
           for i, k in enumerate(Kx)}
    return out, per


# ------------------------------------------------------------------------------------------------ L1 and the per-link catalogue
def stack4(SS, K4):
    T4 = sorted(set().union(*(set(SS[s]["rows"]) for s in ESS)))
    lfc = np.stack([align(SS[s], T4, K4, "lfc") for s in ESS]); se = np.stack([align(SS[s], T4, K4, "se") for s in ESS])
    sig = np.stack([align(SS[s], T4, K4, "sig") for s in ESS])
    own = np.stack([np.array([SS[s]["own"][SS[s]["ci"][k]] for k in K4], np.float32) for s in ESS])   # 4 x K
    return T4, lfc, se, sig, own


def masked_sd(x, m):
    n = m.sum(1); mu = np.where(m, x, 0).sum(1) / n
    return np.sqrt(np.where(m, (x - mu[:, None]) ** 2, 0).sum(1) / n)


def run_l1(T4, lfc, sig, own, rng):
    usable = np.isfinite(own) & (own <= KD_MIN)                                  # 4 x K
    S = sig & usable[:, None, :]
    pos = (S & (lfc > 0)).sum(0); neg_ = (S & (lfc < 0)).sum(0); n = S.sum(0)
    link = (n >= 2) & ((pos == 0) | (neg_ == 0))
    ti, ki = np.nonzero(link); total = len(ti)
    if total > L1_MAX:
        pick = np.sort(rng.choice(total, L1_MAX, replace=False)); ti, ki = ti[pick], ki[pick]
    M = S[:, ti, ki].T                                                           # links x 4
    lE = np.log2(np.abs(lfc[:, ti, ki].T)); lK = np.log2(np.abs(own[:, ki].T))
    U = usable.T                                                                 # K x 4

    def rand_k(mask):
        rk = rng.integers(0, U.shape[0], len(mask))
        for _ in range(50):
            bad = ~np.all(U[rk] | ~mask, 1) | (rk == ki)
            if not bad.any():
                break
            rk[bad] = rng.integers(0, U.shape[0], int(bad.sum()))
        good = np.all(U[rk] | ~mask, 1) & (rk != ki)
        return rk, good
    r1, g1 = rand_k(M); r2, g2 = rand_k(M); ok = g1 & g2
    lK1 = np.log2(np.abs(own[:, r1].T)); lK2 = np.log2(np.abs(own[:, r2].T))
    d_raw = masked_sd(lE, M); d_dose = masked_sd(lE - lK, M); d_rand = masked_sd(lE - lK1, M); d_rand2 = masked_sd(lE - lK2, M)
    beta = np.where(M, lE - lK, 0).sum(1) / M.sum(1)
    lEp = lK + beta[:, None] + rng.normal(0, 0.3, lK.shape)
    real = binom_greater((d_rand - d_dose)[ok]); negc = binom_greater((d_rand2 - d_rand)[ok])
    plant = binom_greater((masked_sd(lEp - lK1, M) - masked_sd(lEp - lK, M))[ok])
    ps = lambda g: bool(g["p"] < 0.05 and g["median"] >= 0.01)
    out = dict(links_total=int(total), links_tested=int(ok.sum()), d_raw_median=float(np.median(d_raw[ok])), d_dose_median=float(np.median(d_dose[ok])),
               d_rand_median=float(np.median(d_rand[ok])), real=dict(**real, passed=ps(real)), negative=dict(**negc, passed=ps(negc)),
               planted=dict(**plant, passed=ps(plant)))
    out["gate"] = dict(controls_ok=bool(not out["negative"]["passed"] and out["planted"]["passed"]))
    out["gate"]["verdict"] = "PASS" if out["real"]["passed"] and out["gate"]["controls_ok"] else "FAIL"
    return out


def catalogue(T4, K4, lfc, se, sig, own, SS):
    """Pattern of every link across the four cell types."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = lfc / se
    meas = np.isfinite(lfc); nm = meas.sum(0); ns = sig.sum(0)
    sp = (sig & (lfc > 0)).sum(0); sn = (sig & (lfc < 0)).sum(0)
    off = meas & ~sig & (np.abs(z) < 1)
    everywhere = (nm >= 3) & (ns == nm) & ((sp == 0) | (sn == 0))
    shared = (ns >= 2) & ((sp == 0) | (sn == 0)) & ~everywhere
    flips = (sp >= 1) & (sn >= 1)
    only = (ns == 1) & (nm >= 3) & ((off.sum(0) + sig.sum(0)) == nm)
    only_meas = (ns == 1) & (nm == 1)
    counts = dict(pairs_measured_anywhere=int((nm >= 1).sum()), links_anywhere=int((ns >= 1).sum()), everywhere=int(everywhere.sum()),
                  shared_some=int(shared.sum()), flips_sign=int(flips.sum()), one_cell_type_only=int(only.sum()),
                  only_measurable_in_one=int(only_meas.sum()))
    counts["one_cell_type_only_by_cell"] = {CELL[s]: int((only & sig[i]).sum()) for i, s in enumerate(ESS)}
    # flip baseline: two experiments in K562 (essential vs genome-wide), links in both
    A, G = SS["K562ESS"], SS[GW]
    Kb = sorted(set(A["cols"]) & set(G["cols"])); Tb = sorted(set(A["rows"]) & set(G["rows"]))
    sa, sg = align(A, Tb, Kb, "sig"), align(G, Tb, Kb, "sig"); la, lg = align(A, Tb, Kb, "lfc"), align(G, Tb, Kb, "lfc")
    both = sa & sg; counts["flip_rate_k562_two_experiments"] = float((both & (np.sign(la) != np.sign(lg))).sum() / max(1, both.sum()))
    fl = []
    for i in range(4):
        for j in range(i + 1, 4):
            b = sig[i] & sig[j]; fl.append(float((b & (np.sign(lfc[i]) != np.sign(lfc[j]))).sum() / max(1, b.sum())))
    counts["flip_rate_across_cell_types_mean"] = float(np.mean(fl))
    sym = sym_map(); ex = {}

    def rows_of(mask, score, n=20000):
        t, k = np.nonzero(mask); sc = score[t, k]; o = np.argsort(-sc)[:n]
        return [dict(regulator=sym(K4[k[i]]), target=sym(T4[t[i]]), regulator_id=K4[k[i]], target_id=T4[t[i]],
                     lfc={CELL[s]: (None if not np.isfinite(lfc[c, t[i], k[i]]) else round(float(lfc[c, t[i], k[i]]), 3)) for c, s in enumerate(ESS)},
                     z={CELL[s]: (None if not np.isfinite(z[c, t[i], k[i]]) else round(float(z[c, t[i], k[i]]), 1)) for c, s in enumerate(ESS)},
                     linked=[CELL[s] for c, s in enumerate(ESS) if sig[c, t[i], k[i]]],
                     per_unit_knockdown={CELL[s]: (round(float(lfc[c, t[i], k[i]] / own[c, k[i]]), 3)
                                                   if sig[c, t[i], k[i]] and np.isfinite(own[c, k[i]]) and own[c, k[i]] <= KD_MIN else None)
                                         for c, s in enumerate(ESS)}) for i in o]
    az = np.where(np.isfinite(z), np.abs(z), 0)
    ex["everywhere"] = rows_of(everywhere, np.where(sig, az, np.inf).min(0), 10000)
    ex["flips_sign"] = rows_of(flips, np.where(sig, az, 0).max(0), 2000)
    for i, s in enumerate(ESS):
        ex[f"only_{CELL[s]}"] = rows_of(only & sig[i], az[i], 5000)
    return counts, ex


# ------------------------------------------------------------------------------------------------ L2 direct or indirect
def tf_list():
    import pandas as pd
    import zipfile
    with zipfile.ZipFile(L("HPA_PROTEINATLAS")) as zf:
        d = pd.read_csv(zf.open(zf.namelist()[0]), sep="\t", usecols=["Gene", "Ensembl", "Protein class"])
    return set(d[d["Protein class"].fillna("").str.contains("Transcription factors")].Ensembl)


def cmh(direct, inct, strata):
    from scipy.stats import chi2
    num = den = sa = sea = sva = 0.0
    for k in np.unique(strata):
        m = strata == k; a = float((direct & inct & m).sum()); b = float((direct & ~inct & m).sum())
        c = float((~direct & inct & m).sum()); d = float((~direct & ~inct & m).sum()); n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n; den += b * c / n; sa += a; sea += (a + b) * (a + c) / n
        sva += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    OR = num / den if den > 0 else float("inf")
    stat = (abs(sa - sea) - 0.5) ** 2 / sva if sva > 0 else 0.0
    return float(OR), float(chi2.sf(stat, 1))


def run_l2(G, rng):
    CT = npz("REG_COLLECTRI"); ct = set(zip(CT["tf"].astype(str), CT["target"].astype(str)))
    ctsign = {(a, b): int(s) for a, b, s in zip(CT["tf"].astype(str), CT["target"].astype(str), CT["sign"])}
    ct_tfs = set(CT["tf"].astype(str)); rows = G["rows"]; rowset = set(rows)
    tfs = [r for r in sorted(tf_list() & set(G["cols"]) & ct_tfs) if any((r, t) in ct for t in rowset)]
    lfc, sig, own, ci, ri = G["lfc"], G["sig"], G["own"], G["ci"], G["ri"]
    med_cols = np.array([ci[m] for m in G["cols"] if m in ri and np.isfinite(own[ci[m]]) and own[ci[m]] <= KD_MIN])
    med_rows = np.array([ri[G["cols"][j]] for j in med_cols])
    recs = []; sh = rng.permutation(len(med_cols)); plant_tot = plant_ok = 0

    def explain(e, tt, j, perm):
        sel = sig[med_rows, j] & (med_cols != j)
        if not sel.any():
            return np.zeros(len(tt), bool), np.full(len(tt), -1)
        mc, mr = med_cols[sel], med_rows[sel]
        mc_eff = med_cols[perm][sel] if perm is not None else mc
        scale = lfc[mr, j] / own[mc]
        Ep = lfc[np.ix_(tt, mc_eff)]; Sg = sig[np.ix_(tt, mc_eff)]
        P = np.where(Sg, Ep * scale[None, :], 0)
        hit = (np.sign(P) == np.sign(e)[:, None]) & (np.abs(P) >= 0.5 * np.abs(e)[:, None])
        best = np.where(hit.any(1), mc[np.argmax(np.where(hit, np.abs(P), -1), 1)], -1)
        return hit.any(1), best
    for r in tfs:
        j = ci[r]; tt = np.nonzero(sig[:, j])[0]
        if not len(tt):
            continue
        e = lfc[tt, j]; zz = np.abs(e / G["se"][tt, j])
        ind, best = explain(e, tt, j, None); ind_n, _ = explain(e, tt, j, sh)
        for t, x, b, bn, zv, ev in zip(tt, ind, best, ind_n, zz, e):
            recs.append((r, rows[t], bool(x), bool(bn), float(zv), float(ev), G["cols"][b] if b >= 0 else ""))
        # planted: links set to a mediator's predicted effect must be called indirect
        sel = np.nonzero(sig[med_rows, j] & (med_cols != j))[0]
        if len(sel) and plant_tot < 200:
            k = rng.choice(sel); mc, mr = med_cols[k], med_rows[k]
            cand = np.nonzero(sig[:, mc] & np.isfinite(lfc[:, mc]))[0]
            cand = cand[(cand != mr) & (cand != ri.get(r, -1))]
            if len(cand):
                t = rng.choice(cand); pe = lfc[t, mc] * lfc[mr, j] / own[mc]
                if np.isfinite(pe) and pe != 0:
                    x, _ = explain(np.array([pe], np.float32), np.array([t]), j, None); plant_tot += 1; plant_ok += int(x[0])
    regs = np.array([x[0] for x in recs]); tg = np.array([x[1] for x in recs])
    ind = np.array([x[2] for x in recs]); ind_n = np.array([x[3] for x in recs]); zz = np.array([x[4] for x in recs])
    inct = np.array([(a, b) in ct for a, b in zip(regs, tg)])
    strata = np.searchsorted(np.quantile(zz, np.linspace(0.1, 0.9, 9)), zz)
    OR, p = cmh(~ind, inct, strata); ORn, pn = cmh(~ind_n, inct, strata)
    ev = np.array([x[5] for x in recs]); cs = np.array([ctsign.get((a, b), 0) for a, b in zip(regs, tg)])

    def agree(m):     # curated activation <-> knockdown lowers the target
        k = m & (cs != 0); return [int(((cs[k] > 0) == (ev[k] < 0)).sum()), int(k.sum())]
    out = dict(tfs=len(tfs), links=int(len(recs)), direct=int((~ind).sum()), indirect=int(ind.sum()),
               collectri_share_direct=float(inct[~ind].mean()) if (~ind).any() else None,
               collectri_share_indirect=float(inct[ind].mean()) if ind.any() else None,
               mh_odds_ratio=OR, cmh_p=p, negative_mh_odds_ratio=ORn, negative_p=pn,
               planted_indirect=plant_ok / max(1, plant_tot), planted_n=plant_tot,
               sign_agreement_direct=agree(~ind), sign_agreement_indirect=agree(ind))
    out["gate"] = dict(real=bool(OR >= 1.5 and p < 0.05), controls_ok=bool(0.67 <= ORn <= 1.5 and out["planted_indirect"] >= 0.9))
    out["gate"]["verdict"] = "PASS" if out["gate"]["real"] and out["gate"]["controls_ok"] else "FAIL"
    return out, recs


def hierarchy(recs):
    """Direct TF -> TF links: who sits above whom."""
    tfs = {r for r, *_ in recs}; down = collections.defaultdict(set); up = collections.defaultdict(set)
    for r, t, ind, _, z, e, m in recs:
        if not ind and t in tfs and t != r:
            down[r].add(t); up[t].add(r)
    sym = sym_map(); rows = []
    for g in tfs:
        rows.append(dict(tf=sym(g), id=g, directly_controls_tfs=len(down[g]), directly_controlled_by_tfs=len(up[g]),
                         level=len(down[g]) - len(up[g]), controls=[sym(x) for x in sorted(down[g])][:30], controlled_by=[sym(x) for x in sorted(up[g])][:30]))
    return sorted(rows, key=lambda x: -x["level"])


# ------------------------------------------------------------------------------------------------ ledger
def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def _git(*a):
    import subprocess
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=_git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G2-")[0].split("DESCRIPTIVE")[0].strip()[:1200], trial=TRIAL),
                               sort_keys=True) + "\n")
            print(cid, "declared")


def show(cid, r):
    """In a trial only the controls are shown (the real outcome stays unseen until the declared run)."""
    if not TRIAL:
        return json.dumps(r, default=float)[:1500]
    keep = {k: v for k, v in r.items() if k in ("negative", "planted", "negative_mh_odds_ratio", "negative_p", "planted_indirect", "planted_n",
                                                "knockdowns_common", "targets_common", "knockdowns_eligible", "knockdowns", "variable", "links_total",
                                                "links_tested", "tfs", "links")}
    keep["controls_ok"] = r["gate"]["controls_ok"]
    return json.dumps(keep, default=float)[:1500]


def test():
    if not TRIAL:
        head = _git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G2 already tested")
    t0 = time.time(); os.makedirs(RUN, exist_ok=True); os.makedirs(OUTD, exist_ok=True); rng = np.random.default_rng(SEED)
    T = tss(); keep = None
    if TRIAL:
        tr = np.random.default_rng(1)
        ess = sorted({str(e) for e in npz("XCELL_K562ESS")["cols_ens"] if str(e)}); gw = sorted({str(e) for e in npz(f"XCELL_{GW}")["cols_ens"] if str(e)})
        gtf = sorted(tf_list() & set(gw))
        keep = set(tr.choice(ess, TRIAL_GENES, replace=False)) | set(tr.choice(gtf, 300, replace=False)) | set(tr.choice(gw, 400, replace=False))
    print("loading screens", flush=True)
    SS = {s: load(s, T, keep) for s in ESS + [GW]}
    print(f"  loaded ({time.time() - t0:.0f}s)", flush=True)
    x1, per_x1 = run_x1(SS, rng); print("G2-X1", show("G2-X1", x1), f"({time.time() - t0:.0f}s)", flush=True)
    K4 = sorted(set.intersection(*(set(SS[s]["cols"]) for s in ESS)))
    x2, per_x2 = run_x2(SS, K4, rng); print("G2-X2", show("G2-X2", x2), f"({time.time() - t0:.0f}s)", flush=True)
    T4, lfc, se, sig, own = stack4(SS, K4)
    l1 = run_l1(T4, lfc, sig, own, rng); print("G2-L1", show("G2-L1", l1), f"({time.time() - t0:.0f}s)", flush=True)
    counts, ex = catalogue(T4, K4, lfc, se, sig, own, SS)
    if not TRIAL:
        print("catalogue", json.dumps(counts), flush=True)
    del lfc, se, sig
    l2, recs = run_l2(SS[GW], rng); print("G2-L2", show("G2-L2", l2), f"({time.time() - t0:.0f}s)", flush=True)
    H = hierarchy(recs)
    if TRIAL:
        print(f"trial done ({time.time() - t0:.0f}s); real outcomes not shown"); return
    sym = sym_map()
    regs = {}
    for k in set(per_x1) | set(per_x2):
        regs[k] = dict(symbol=sym(k), **per_x2.get(k, {}), **per_x1.get(k, {}))
    l2rows = [dict(tf=sym(r), target=sym(t), tf_id=r, target_id=t, kind="indirect" if ind else "direct", via=sym(m) if m else None,
                   lfc=round(e, 3), z=round(z, 1)) for r, t, ind, _, z, e, m in recs]
    with gzip.open(os.path.join(OUTD, "xcell_links.json.gz"), "wt") as f:
        json.dump(dict(counts=counts, examples=ex), f, default=float)
    with gzip.open(os.path.join(OUTD, "k562_direct_indirect.json.gz"), "wt") as f:
        json.dump(l2rows, f, default=float)
    json.dump(dict(x1=x1, x2=x2, l1=l1, l2=l2, catalogue=counts), open(os.path.join(OUTD, "xcell_summary.json"), "w"), indent=1, default=float)
    json.dump(regs, open(os.path.join(OUTD, "xcell_regulators.json"), "w"), default=float)
    json.dump(H, open(os.path.join(OUTD, "tf_hierarchy_k562.json"), "w"), default=float)
    facts = bank_facts(regs, ex, l2rows, H)
    n = cbb.bank().add_many(facts)
    for cid, r in (("G2-X1", x1), ("G2-X2", x2), ("G2-L1", l1), ("G2-L2", l2)):
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(USED), **r), sort_keys=True, default=float) + "\n")
    pc = x1["per_cell"]
    Ls = ["=" * 97, "GENEREGX -- how gene-to-gene influence changes with cell type, and at what level", "=" * 97,
          f"  G2-X1 influence changes with cell type: {x1['knockdowns_eligible']} knockdowns, noise-corrected agreement with K562 "
          f"(K562 second experiment {x1['rho_ref_median']:.2f}; " + ", ".join(f"{c} {v['rho_median']:.2f}" for c, v in pc.items()) +
          f"); negative passed {sum(v['passed'] for v in x1['negative'].values())}/3, planted {sum(v['passed'] for v in x1['planted'].values())}/3 -> {x1['gate']['verdict']}",
          f"  G2-X2 more expressed -> more influence: {x2['real']['n']} knockdowns, mean Spearman {x2['real']['mean_spearman']:.3f}, p {x2['real']['p']:.3g} "
          f"(negative {x2['negative']['mean_spearman']:.3f}, planted {x2['planted']['mean_spearman']:.3f}) -> {x2['gate']['verdict']}",
          f"  G2-L1 level = effect per unit of knockdown: {l1['links_tested']:,} links, spread across cell types raw {l1['d_raw_median']:.3f}, per unit "
          f"{l1['d_dose_median']:.3f}, random dose {l1['d_rand_median']:.3f}; gain median {l1['real']['median']:.3f}, p {l1['real']['p']:.3g} -> {l1['gate']['verdict']}",
          f"  G2-L2 direct vs via another gene (K562): {l2['links']:,} links from {l2['tfs']} TFs, {l2['direct']:,} direct, {l2['indirect']:,} indirect; "
          f"CollecTRI share direct {l2['collectri_share_direct']:.3f} vs indirect {l2['collectri_share_indirect']:.3f}, MH OR {l2['mh_odds_ratio']:.2f}, "
          f"p {l2['cmh_p']:.3g} (shuffled mediators OR {l2['negative_mh_odds_ratio']:.2f}; planted {l2['planted_indirect']:.0%}) -> {l2['gate']['verdict']}",
          f"  Links across the four cell types: everywhere {counts['everywhere']:,}; shared by some {counts['shared_some']:,}; one cell type only "
          f"{counts['one_cell_type_only']:,} " + json.dumps(counts['one_cell_type_only_by_cell']) + f"; flips {counts['flips_sign']:,} "
          f"(flip rate across cell types {counts['flip_rate_across_cell_types_mean']:.3f} vs K562 two experiments {counts['flip_rate_k562_two_experiments']:.3f})",
          f"  Facts written {n:,}"]
    open(RES, "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))


def bank_facts(regs, ex, l2rows, H):
    t = cbb.now(); src = {"name": "generegx:G2", "code_sha": code_sha()}; inp = tuple(sorted(USED)); F = []
    for g, d in regs.items():
        F.append(Fact(g, "xcell/influence_by_cell_type", d, t[:10], t, "derived", src, inputs=inp,
                      context={"reach": "share of measured genes the knockdown changes", "expression": "HPA log1p nTPM",
                               "rho_vs_K562": "noise-corrected agreement of its target profile with K562 essential"}))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for cls, rows in ex.items():
        for x in rows:
            by[x["regulator_id"]][f"targets_{cls}"].append([x["target"], x["lfc"], x["per_unit_knockdown"]])
            by[x["target_id"]][f"regulators_{cls}"].append([x["regulator"], x["lfc"]])
    for x in l2rows:
        by[x["tf_id"]]["k562_" + x["kind"] + "_targets"].append([x["target"], x["lfc"], x["z"], x["via"]])
        by[x["target_id"]]["k562_" + x["kind"] + "_tfs"].append([x["tf"], x["lfc"], x["z"], x["via"]])
    for g, d in by.items():
        for k, v in d.items():
            F.append(Fact(g, f"xcell/{k}", v[:60], t[:10], t, "derived", src, inputs=inp, context={"n": len(v)}))
    for h in H:
        F.append(Fact(h["id"], "xcell/tf_hierarchy_k562", {k: v for k, v in h.items() if k != "id"}, t[:10], t, "derived", src, inputs=inp))
    return F


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
