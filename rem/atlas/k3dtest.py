"""K3DTEST (G16) -- when the 3D shape of a chromosome changes, do the genes there change with it, and which comes first?
K562 cells pushed toward megakaryocyte with PMA (Bond et al. 2023 Genome Research; GEO GSE213909), 3D map (Hi-C) at
0, 6 and 72 h and RNA at 0, 30 min, 90 min, 3, 6, 24, 48 and 72 h (layer K562_PMA_3D, built by k3dbank.py).

Asked by the user: "build G16 with K562 first" -- goal: "look at the chromosome and its 3d map at that time, and compare it
with that time's gene expression so we can link how change in chromosome links to gene expression".
Genes: protein-coding (GENCODE 46, chr1-22 + X), EXPRESSED = mean TPM >= 1 at 0 h or 72 h. Expression e = log2(TPM + 1),
mean of the 2 replicates; change = e(72 h) - e(0 h). Baseline strata: ten equal-count bins of e(0 h).
Halves: DEV = sha256('cellbrain-c1:' + Ensembl) parity 0, EXAM = parity 1; the trial used DEV genes and showed only
controls; the declared run scores EXAM genes once. Gain rule (house rule): one-sided sign test p < 0.05 AND median >= +0.01.

CLAIMS (declared before running; ids G16-*):
 G16-C1 OPEN/CLOSED NEIGHBOURHOOD. Each 100-kb bin has a compartment value E1 (open 'A' positive, closed 'B' negative;
        per chromosome, mean 0, SD 1). SHIFTED bins = the 10% of bins with the largest |E1(72 h) - E1(0 h)|; STABLE =
        the half with the smallest. For each expressed gene whose start site lies in a shifted bin: score =
        sign(E1 change) x (its expression change - mean expression change of stable-bin genes in the same half with the
        same baseline stratum and the same 0 h E1 quintile). PASS: gain rule on EXAM genes AND the median score beats
        95% of 1,000 circular shifts of the E1-change track along each chromosome (keeps neighbouring bins together;
        p_MC < 0.05).
 G16-L1 LOOPS AT THE GENE START. Loops from the authors' table (33,914 at 10 kb; 'gained' / 'lost' = their differential
        calls); direction = sign of the loop's log2 fold change at 72 h. A gene belongs to a loop when its start site
        is within 10 kb of either anchor. Genes with only changing loops of one direction get sign +1 (gained) or -1
        (lost); mixed genes are left out. Score = sign x (expression change - mean change of genes in the same half
        and baseline stratum that sit only at static loops). PASS: gain rule on EXAM genes AND the median beats 95% of
        1,000 shuffles of gained / lost labels among the changing loops (p_MC < 0.05).
 G16-T1 THE 3D CHANGE COMES FIRST. Timing is read against transcription (intron RNA, short-lived) rather than mature
        mRNA, which lags. Hi-C halves: biological replicates 1-2 vs 3-4; RNA replicates 1 vs 2. Cross-fitted: a
        loop-gene pair (gene start within 10 kb of an anchor, any loop) is CHOSEN on one Hi-C half + one RNA replicate
        (|loop change 0 -> 72 h| >= 1 and |intron change 0 -> 72 h| >= 1, same sign) and MEASURED on the other half +
        other replicate: share of the 72 h change already reached at 6 h, for the loop (fL) and for the intron signal
        (fY), each clipped to [-0.5, 1.5]; d = fL - fY; per gene the median d over its pairs (both folds). PASS: gain
        rule on EXAM genes (d > 0).
 Controls: C1 and L1: 50 planted runs (an effect of +0.2 log2 in the predicted direction added on a shifted / shuffled
        track) must PASS the full rule in >= 90% (else VOID); 50 null runs (shifted / shuffled track, no effect) must
        PASS in <= 10% (else VOID). T1: 200 simulated runs with EQUAL timing for loop and intron (each pair's own
        changes and the measured noise of each assay) must pass in <= 10%, and 200 with the loop 0.25 ahead and the
        intron 0.25 behind must pass in >= 90% (else VOID).
 REPORTED: the same at 6 h (C1, L1); C1 and L1 with intron signal; T1 with mRNA instead of intron signal; share of
        pairs where expression moves first; rank correlations of E1 change and of loop change with expression change;
        numbers of bins flipping A <-> B; per-chromosome compartment checks.
NOT: bulk cells (an average); PMA also stops division and changes many pathways at once; one cell line; only one Hi-C
time point between 0 and 72 h; loop calls and their differential test are the authors'; association, not cause.

Run: python3 -I rem/atlas/k3dtest.py declare | test [--trial]
"""

from __future__ import annotations
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
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


gx = _load("generegx")
cbb, Fact = gx.cbb, gx.Fact
LEDGER = gx.LEDGER
OUT = R("outputs", "k562_3d")
CLAIMS = ["G16-C1", "G16-L1", "G16-T1"]
SEED = 20261128
N_MC = 100 if TRIAL else 1000
N_CTRL, N_CTRL_MC = (10, 100) if TRIAL else (50, 200)
N_SIM = 50 if TRIAL else 200
PLANT = 0.2
H_ = lambda e: int(hashlib.sha256(("cellbrain-c1:" + e).encode()).hexdigest(), 16) % 2
LAYER = "K562_PMA_3D"


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "k3dtest.py"), "rb").read())
    h.update(cbb.bank().facts(f"LAYER-{LAYER}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def signtest(s):
    from scipy.stats import binom
    s = np.asarray(s); s = s[s != 0]; n = len(s); k = int((s > 0).sum())
    return (float(binom.sf(k - 1, n, 0.5)) if n else 1.0), (float(np.median(s)) if n else 0.0), n


def gain(s):
    p, med, n = signtest(s); return p < 0.05 and med >= 0.01, p, med, n


# ---------------------------------------------------------------- data
def load():
    Z = np.load(os.path.join(cbb.LAY, "k562_pma_3d.npz")); gx.USED.add(LAYER)
    D = {k: Z[k] for k in Z.files}
    cols = [tuple(map(int, c.split("_"))) for c in D["rna_cols"]]
    e = np.log2(D["tpm"].astype(np.float64) + 1)
    D["e"] = {(t, r): e[:, k] for k, (t, r) in enumerate(cols)}
    D["em"] = {t: np.mean([D["e"][(t, r)] for r in (1, 2)], 0) for t in sorted({t for t, _ in cols})}
    tpm_m = {t: np.mean([D["tpm"][:, k] for k, (tt, r) in enumerate(cols) if tt == t], 0) for t in (0, 4320)}
    D["expressed"] = (tpm_m[0] >= 1) | (tpm_m[4320] >= 1)
    D["half"] = np.array([H_(g) for g in D["gene"]])
    D["stratum"] = np.full(len(D["gene"]), -1); ex = D["expressed"]
    q = np.quantile(D["em"][0][ex], np.linspace(0, 1, 11)[1:-1]); D["stratum"][ex] = np.searchsorted(q, D["em"][0][ex], side="right")
    # intron signal: median-ratio size factors over genes with all values > 0, then log2(x / s + p)
    icols = [tuple(map(int, c.split("_"))) for c in D["int_cols"]]; X = D["intron"].astype(np.float64)
    good = np.isfinite(X).all(1) & (X > 0).all(1) & ex
    ref = np.exp(np.log(X[good]).mean(1)); sf = np.median(X[good] / ref[:, None], 0); Xn = X / sf
    p = 0.05 * np.median(Xn[good][:, [k for k, (t, r) in enumerate(icols) if t == 0]])
    D["y"] = {(t, r): np.log2(Xn[:, k] + p) for k, (t, r) in enumerate(icols)}
    D["y_ok"] = np.isfinite(X).all(1) & ex & (np.nanmean(Xn, 1) > 2 * p)
    # gene -> 100 kb bin
    bkey = {(c, s): k for k, (c, s) in enumerate(zip(D["bin_chrom"], D["bin_start"]))}
    D["gbin"] = np.array([bkey.get((c, (t // 100000) * 100000), -1) for c, t in zip(D["chrom"], D["tss"])])
    return D


# ---------------------------------------------------------------- C1
class C1:
    def __init__(self, D, t1=2, t0=0):
        self.D = D; E = D["e1"].astype(np.float64)
        self.valid = np.isfinite(E[:, t0]) & np.isfinite(E[:, t1])
        self.d = np.where(self.valid, E[:, t1] - E[:, t0], np.nan)
        b0 = E[:, t0]; q = np.nanquantile(b0[self.valid], [0.2, 0.4, 0.6, 0.8])
        self.bq = np.where(self.valid, np.searchsorted(q, np.nan_to_num(b0), side="right"), -1)
        self.idx = {c: np.nonzero(self.valid & (D["bin_chrom"] == c))[0] for c in np.unique(D["bin_chrom"])}
        g = D["gbin"]; self.gok = (g >= 0) & D["expressed"]; self.gok[self.gok] &= self.valid[g[self.gok]]

    def roll(self, rng):
        d = self.d.copy()
        for c, ix in self.idx.items():
            n = len(ix)
            if n > 2:
                k = int(rng.integers(50, n - 50)) if n > 120 else int(rng.integers(1, n)); d[ix] = np.roll(self.d[ix], k)
        return d

    def scores(self, d, lfc, half):
        D = self.D; a = np.abs(d[self.valid]); hi, lo = np.quantile(a, 0.9), np.quantile(a, 0.5)
        g = D["gbin"]; m = self.gok & (D["half"] == half); gd = np.where(m, d[np.where(g >= 0, g, 0)], np.nan)
        shifted = m & (np.abs(gd) >= hi); stable = m & (np.abs(gd) <= lo)
        cell = D["stratum"] * 5 + self.bq[np.where(g >= 0, g, 0)]
        nC = np.bincount(cell[stable], minlength=50); sC = np.bincount(cell[stable], weights=lfc[stable], minlength=50)
        ctrl = np.where(nC >= 5, sC / np.maximum(nC, 1), np.nan)
        s = np.sign(gd[shifted]) * (lfc[shifted] - ctrl[cell[shifted]])
        return s[np.isfinite(s)]

    def full(self, d, lfc, half, rng, n_mc):
        s = self.scores(d, lfc, half); ok, p, med, n = gain(s)
        null = np.array([np.median(self.scores(self.roll_from(d, rng), lfc, half)) for _ in range(n_mc)])
        p_mc = float((1 + (null >= med).sum()) / (n_mc + 1))
        return dict(genes=n, sign_p=p, median=med, share_positive=float((s > 0).mean()) if n else None, p_mc=p_mc, null_median_95=float(np.quantile(null, 0.95)),
                    passed=bool(ok and p_mc < 0.05))

    def roll_from(self, d, rng):
        save = self.d; self.d = d; out = self.roll(rng); self.d = save; return out


# ---------------------------------------------------------------- L1
class L1:
    def __init__(self, D, col=1, M=None):
        from scipy.sparse import csr_matrix
        self.D = D; A = D["loop_anchor"]; lfc = D["loop_lfc"][:, col]; cl = D["loop_cluster"]
        if M is None:
            rows, cols_ = [], []
            for side in (0, 3):
                ch = A[:, side]; s = A[:, side + 1].astype(np.int64); e = A[:, side + 2].astype(np.int64); W = int((e - s).max())
                for c in np.unique(ch):
                    li = np.nonzero(ch == c)[0]; li = li[np.argsort(s[li])]; ss = s[li]
                    for g in np.nonzero(D["chrom"] == c)[0]:
                        t = int(D["tss"][g])
                        for k in range(np.searchsorted(ss, t - 10000 - W, side="right"), np.searchsorted(ss, t + 10000, side="right")):
                            L = li[k]
                            if s[L] - 10000 <= t < e[L] + 10000:
                                rows.append(g); cols_.append(L)
            M = csr_matrix((np.ones(len(rows)), (rows, cols_)), shape=(len(D["gene"]), len(A))); M.sum_duplicates(); M.data[:] = 1
        self.M = M; self.diff = np.isin(cl, ["gained", "lost"]) & (lfc != 0); self.static = cl == "static"
        self.dir = np.where(self.diff, np.sign(lfc), 0.0)
        self.on_static = np.asarray(M @ self.static.astype(float)).ravel() > 0
        self.on_diff = np.asarray(M @ self.diff.astype(float)).ravel() > 0
        self.cluster_agree = float((np.sign(lfc[self.diff]) == np.where(cl[self.diff] == "gained", 1, -1)).mean())

    def gene_sign(self, dirv):
        pos = np.asarray(self.M @ (dirv > 0).astype(float)).ravel(); neg = np.asarray(self.M @ (dirv < 0).astype(float)).ravel()
        return np.where((pos > 0) & (neg == 0), 1.0, np.where((neg > 0) & (pos == 0), -1.0, 0.0))

    def scores(self, dirv, lfc, half):
        D = self.D; m = D["expressed"] & (D["half"] == half); sg = self.gene_sign(dirv)
        test = m & (sg != 0); ctrl_pool = m & self.on_static & ~self.on_diff
        nC = np.bincount(D["stratum"][ctrl_pool], minlength=10); sC = np.bincount(D["stratum"][ctrl_pool], weights=lfc[ctrl_pool], minlength=10)
        ctrl = np.where(nC >= 5, sC / np.maximum(nC, 1), np.nan)
        s = sg[test] * (lfc[test] - ctrl[D["stratum"][test]]); return s[np.isfinite(s)]

    def shuffle(self, rng):
        d = self.dir.copy(); ix = np.nonzero(self.diff)[0]; d[ix] = rng.permutation(d[ix]); return d

    def full(self, dirv, lfc, half, rng, n_mc):
        s = self.scores(dirv, lfc, half); ok, p, med, n = gain(s)
        null = np.array([np.median(self.scores(self.shuffle(rng), lfc, half)) for _ in range(n_mc)])
        p_mc = float((1 + (null >= med).sum()) / (n_mc + 1))
        return dict(genes=n, sign_p=p, median=med, share_positive=float((s > 0).mean()) if n else None, p_mc=p_mc, null_median_95=float(np.quantile(null, 0.95)),
                    passed=bool(ok and p_mc < 0.05))


def controls(obj, null_track, lfc, half, rng, plant_genes_sign):
    pl = ng = 0
    for _ in range(N_CTRL):
        tr = null_track(rng); sg = plant_genes_sign(tr)
        pl += obj.full(tr, lfc + PLANT * sg, half, rng, N_CTRL_MC)["passed"]
        ng += obj.full(null_track(rng), lfc, half, rng, N_CTRL_MC)["passed"]
    return dict(planted_pass_rate=pl / N_CTRL, null_pass_rate=ng / N_CTRL, runs=N_CTRL, ok=bool(pl / N_CTRL >= 0.9 and ng / N_CTRL <= 0.10))


def c1_plant_sign(c1):
    def f(d):
        D = c1.D; a = np.abs(d[c1.valid]); hi = np.quantile(a, 0.9); g = D["gbin"]
        gd = np.where(c1.gok, d[np.where(g >= 0, g, 0)], 0.0); return np.where(np.abs(gd) >= hi, np.sign(gd), 0.0)
    return f


# ---------------------------------------------------------------- T1
def t1_data(D):
    libs = [str(x) for x in D["loop_libs"]]; C = D["loop_counts"].astype(np.float64); tot = C.sum(0)
    half = {"A": (1, 2), "B": (3, 4)}; L, N = {}, {}
    for h, reps in half.items():
        for t in (0, 360, 4320):
            k = [i for i, n in enumerate(libs) if n.split("_")[4] == str(t) and int(n.split("_")[5]) in reps]
            cnt = C[:, k].sum(1); L[(h, t)] = np.log2(cnt / tot[k].sum() * 1e6 + 1); N[(h, t)] = cnt
    return L, N


def t1_pairs(D, l1):
    M = l1.M.tocoo(); m = D["y_ok"][M.row] & D["expressed"][M.row]; return M.row[m], M.col[m]


def frac(d6, d72):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(np.where(d72 != 0, d6 / d72, 0.0), -0.5, 1.5)


def t1_scores(D, L, gpair, lpair, Y, half, folds=(("A", 1, "B", 2), ("B", 2, "A", 1)), keep=None):
    per = {}
    for hs, rs, hm, rm in folds:
        dLs = L[(hs, 4320)][lpair] - L[(hs, 0)][lpair]; dYs = Y[(4320, rs)][gpair] - Y[(0, rs)][gpair]
        ch = (np.abs(dLs) >= 1) & (np.abs(dYs) >= 1) & (np.sign(dLs) == np.sign(dYs)) & (D["half"][gpair] == half)
        fL = frac(L[(hm, 360)][lpair] - L[(hm, 0)][lpair], L[(hm, 4320)][lpair] - L[(hm, 0)][lpair])
        fY = frac(Y[(360, rm)][gpair] - Y[(0, rm)][gpair], Y[(4320, rm)][gpair] - Y[(0, rm)][gpair])
        for k in np.nonzero(ch)[0]:
            per.setdefault(int(gpair[k]), []).append(fL[k] - fY[k])
            if keep is not None:
                keep.append((hs, rs, hm, rm, int(gpair[k]), int(lpair[k]), float(fL[k]), float(fY[k])))
    return np.array([np.median(v) for v in per.values()])


def t1_sim(D, L, N, Y, keep, rng, shift):
    """re-measure every chosen pair with simulated noise; equal timing (shift 0) or loop ahead by `shift`."""
    diff = np.concatenate([(Y[(t, 1)] - Y[(t, 2)])[D["y_ok"]] for t in (0, 360, 4320)])
    mean_lv = np.concatenate([((Y[(t, 1)] + Y[(t, 2)]) / 2)[D["y_ok"]] for t in (0, 360, 4320)])
    qs = np.quantile(mean_lv, np.linspace(0, 1, 11)); sdb = np.array([diff[(mean_lv >= qs[i]) & (mean_lv <= qs[i + 1])].std() / np.sqrt(2) for i in range(10)])
    per = {}
    for hs, rs, hm, rm, g, l, fL, fY in keep:
        f = float(np.clip((fL + fY) / 2, 0, 1)); dL = L[(hm, 4320)][l] - L[(hm, 0)][l]; dY = Y[(4320, rm)][g] - Y[(0, rm)][g]
        sL = [1 / (np.log(2) * np.sqrt(N[(hm, t)][l] + 1)) for t in (0, 360, 4320)]
        sY = [sdb[min(9, max(0, np.searchsorted(qs, Y[(t, rm)][g]) - 1))] for t in (0, 360, 4320)]
        fl, fy = np.clip(f + shift, 0, 1), np.clip(f - shift, 0, 1)
        Ls = np.array([0, fl * dL, dL]) + rng.normal(0, sL); Ys = np.array([0, fy * dY, dY]) + rng.normal(0, sY)
        per.setdefault(g, []).append(frac(Ls[1] - Ls[0], Ls[2] - Ls[0]) - frac(Ys[1] - Ys[0], Ys[2] - Ys[0]))
    return np.array([np.median(v) for v in per.values()])


# ---------------------------------------------------------------- run
def run(rng, half):
    D = load(); out, rep, ctl = {}, {}, {}
    lfc72 = D["em"][4320] - D["em"][0]; lfc6 = D["em"][360] - D["em"][0]
    ylfc72 = np.mean([D["y"][(4320, r)] - D["y"][(0, r)] for r in (1, 2)], 0)
    c1 = C1(D); res = c1.full(c1.d, lfc72, half, rng, N_MC)
    ctl["G16-C1"] = controls(c1, lambda r: c1.roll(r), lfc72, half, rng, c1_plant_sign(c1))
    out["G16-C1"] = res
    c16 = C1(D, t1=1); rep["C1_at_6h"] = c16.full(c16.d, lfc6, half, rng, N_MC)
    yl = np.where(D["y_ok"], ylfc72, np.nan); D_ = dict(D); D_["expressed"] = D["expressed"] & D["y_ok"]; c1y = C1(D_)
    rep["C1_intron_signal"] = c1y.full(c1y.d, np.nan_to_num(yl), half, rng, N_MC)
    E = D["e1"]; v = np.isfinite(E[:, 0]) & np.isfinite(E[:, 2])
    rep["bins"] = dict(valid=int(v.sum()), flip_B_to_A=int(((E[:, 0] < 0) & (E[:, 2] > 0) & v).sum()), flip_A_to_B=int(((E[:, 0] > 0) & (E[:, 2] < 0) & v).sum()),
                       corr_E1_0h_72h=float(np.corrcoef(E[v, 0], E[v, 2])[0, 1]))
    from scipy.stats import spearmanr
    gm = c1.gok & (D["half"] == half)
    rep["rank_corr_E1_change_vs_expression_change"] = float(spearmanr(c1.d[D["gbin"][gm]], lfc72[gm])[0])

    l1 = L1(D); out["G16-L1"] = l1.full(l1.dir, lfc72, half, rng, N_MC)
    ctl["G16-L1"] = controls(l1, lambda r: l1.shuffle(r), lfc72, half, rng, lambda d: l1.gene_sign(d))
    l16 = L1(D, col=0, M=l1.M); rep["L1_at_6h"] = l16.full(l16.dir, lfc6, half, rng, N_MC)
    D2 = dict(D); D2["expressed"] = D["expressed"] & D["y_ok"]; l1y = L1(D2, M=l1.M); rep["L1_intron_signal"] = l1y.full(l1y.dir, np.nan_to_num(yl), half, rng, N_MC)
    rep["loops"] = dict(changing=int(l1.diff.sum()), gained_72h=int((l1.dir > 0).sum()), lost_72h=int((l1.dir < 0).sum()), cluster_vs_72h_sign_agree=l1.cluster_agree)
    sg = l1.gene_sign(l1.dir); mm = D["expressed"] & (D["half"] == half) & (sg != 0)
    rep["rank_corr_loop_sign_vs_expression_change"] = float(spearmanr(sg[mm], lfc72[mm])[0]) if mm.sum() > 2 else None

    L, N = t1_data(D); gp, lp = t1_pairs(D, l1); keep = []
    d = t1_scores(D, L, gp, lp, D["y"], half, keep=keep); ok, p, med, n = gain(d)
    out["G16-T1"] = dict(genes=n, pairs=len(keep), sign_p=p, median=med, share_loop_first=float((d > 0).mean()) if n else None,
                         share_expression_first=float((d < 0).mean()) if n else None, passed=bool(ok))
    eq = [gain(t1_sim(D, L, N, D["y"], keep, rng, 0.0))[0] for _ in range(N_SIM)]
    pl = [gain(t1_sim(D, L, N, D["y"], keep, rng, 0.25))[0] for _ in range(N_SIM)]
    ctl["G16-T1"] = dict(equal_timing_pass_rate=float(np.mean(eq)), planted_lead_pass_rate=float(np.mean(pl)), runs=N_SIM,
                         ok=bool(np.mean(eq) <= 0.10 and np.mean(pl) >= 0.9))
    Ye = {k: v for k, v in D["e"].items()}
    dm = t1_scores(D, L, gp, lp, Ye, half); okm, pm, medm, nm = gain(dm)
    rep["T1_mRNA_instead_of_intron"] = dict(genes=nm, sign_p=pm, median=medm, share_loop_first=float((dm > 0).mean()) if nm else None)
    for cid in CLAIMS:
        out[cid]["controls"] = ctl[cid]
        out[cid]["gate"] = dict(real=out[cid]["passed"], controls_ok=ctl[cid]["ok"],
                                verdict=("PASS" if out[cid]["passed"] else "FAIL") if ctl[cid]["ok"] else "VOID (controls)")
    rep["genes_expressed_in_half"] = int((D["expressed"] & (D["half"] == half)).sum())
    return out, rep


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G16-")[0].split(" Controls:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G16 already tested")
    t0 = time.time(); out, rep = run(np.random.default_rng(SEED), 0 if TRIAL else 1)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]), "units:", out[cid]["genes"])
        print("trial ran on DEV genes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(dict(out, reported=rep), open(os.path.join(OUT, "k562_3d_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "k3dtest:G16", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "k562_3d_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
