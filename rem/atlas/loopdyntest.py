"""LOOPDYNTEST (G20) -- what GUIDES THE CHANGES in chromosome folding over time, and how a loop passes activity to a
gene. K562 cells pushed toward megakaryocyte with PMA: loops, zones and RNA at 0, 6 h, 72 h (layer K562_PMA_3D) and, at the
same times, per-site counts of CTCF, cohesin (RAD21; 0 and 72 h), the activity mark H3K27ac, the switched-on factor JUN
(AP-1) and open DNA (ATAC), 2 replicates (layer LOOPCODE_K562; Bond et al. 2023).

Asked by the user: "crack the loop mechanism ... what guides them, how?" (G19 tests the static rules; G16-G18 found that
gene activity changes before loops do).
Measures. Per assay and library: counts / median-ratio size factor; a WINDOW's value = log2(sum over sites centred in it
+ 1). A loop anchor's window = its 10 kb bin +- 5 kb. Change = value(72 h) - value(0 h), mean of the 2 replicates.
Loop change = the authors' log2 fold change at 72 h; for timing, loop strength per Hi-C half (biological replicates 1-2 /
3-4) as in G16. Gene change = log2(TPM + 1) at 72 h - 0 h (expressed genes: mean TPM >= 1 at 0 h or 72 h).
DNA halves (house rule): DEV = odd autosomes, EXAM = even autosomes; models fitted on DEV, scored on EXAM; the trial used
DEV only and showed only controls. Gain rule: one-sided sign test p < 0.05 AND median gain >= +0.01.

CLAIMS (declared before running; ids G20-*):
 G20-D1 WHAT MOVES A LOOP: ACTIVITY AT ITS ENDS, BEYOND CTCF AND COHESIN. Ridge models (alpha 1) of the loop change, fitted
        on DEV loops. ARCH = loop size + change of CTCF and of cohesin at the anchors (mean of the two anchors, and the
        anchor with the larger change). FULL = ARCH + the same for H3K27ac, JUN and ATAC + change of H3K27ac inside the
        loop. Score: Spearman r of prediction with loop change per EXAM chromosome. PASS: median r(FULL) >= 0.15 AND
        gain rule on r(FULL) - r(ARCH) over the 11 EXAM chromosomes.
        Controls: activity features permuted among the loops of each chromosome (10 runs) must pass <= 10%; planted
        (loop change replaced by its permutation + 0.5 x z-scored mean H3K27ac change of the anchors; 5 runs) >= 90%.
 G20-D2 ORDER OF EVENTS: A LOOP'S ENDS LIGHT UP BEFORE THE LOOP CHANGES. Cross-fitted as G16-T1: a loop is CHOSEN on one
        Hi-C half + one replicate (|loop change| >= 0.5 and |mean anchor H3K27ac change| >= 0.5, log2, 0 -> 72 h, same
        sign) and MEASURED on the other half + other replicate. Pooled over chosen EXAM loops (both folds; turned by the
        sign of the chosen change): share of the 72 h change reached at 6 h, F_act for anchor H3K27ac and F_loop for the
        loop; d = F_act - F_loop. PASS: d >= +0.01 AND d > 0 in >= 95% of 2,000 bootstrap resamples of loops.
        Controls: the loops' measured 6 h change replaced by F_act x (their 72 h change) + counting noise (equal timing;
        200 runs) must pass <= 10%; replaced by (F_act - 0.25) x (72 h change) + noise (planted lag) must pass >= 90%.
 G20-E1 A LOOP CARRIES ITS FAR END'S ACTIVITY TO THE GENE. Pairs: an expressed gene whose start site is within 10 kb of
        one anchor, and the loop's other anchor (its window, >= 1 H3K27ac site, > 20 kb from the gene start). Matched
        control for each pair: the H3K27ac site whose distance from the gene start is closest to the looped distance
        (within 25%), not within 15 kb of any anchor of a loop at that gene (window = site +- 10 kb). Statistic:
        Spearman r(H3K27ac change at the looped end, gene change) - Spearman r(H3K27ac change at the control, gene
        change), over EXAM pairs. PASS: difference >= +0.01 AND > 0 in >= 95% of 2,000 bootstrap resamples of genes.
        Controls: looped-end changes permuted among pairs (10 runs) must pass <= 10%; planted (gene change + 0.5 x
        z-scored looped-end change; 5 runs with the gene changes permuted first) must pass >= 90%.
 G20-M2 A ZONE CHANGES WHERE ACTIVITY CHANGES. Ridge (alpha 1) fitted on DEV 100 kb bins: change of E1 (72 h - 0 h) from
        the bin's change of H3K27ac, ATAC, JUN, CTCF (window = the bin) and the mean change of its expressed genes.
        Score: Spearman r per EXAM chromosome. PASS: median r >= 0.10 AND gain rule on r over the 11 EXAM chromosomes.
        Controls: features shifted 100 bins along each chromosome must not pass; planted (E1 change replaced by its
        shifted copy + 0.5 x z-scored H3K27ac change; 5 runs, each with its own shift) must pass >= 90%.
 G20-M3 ACTIVITY CHANGES BEFORE THE ZONE DOES. Bins CHOSEN on one replicate (|H3K27ac change 0 -> 72 h| >= 0.5) and
        MEASURED on the other (both folds, EXAM chromosomes; turned by the sign of the chosen change): F_act = share of
        the 72 h H3K27ac change reached at 6 h; F_zone = the same for E1 (merged maps; the choice does not use E1, so
        its noise cannot bias F_zone). d = F_act - F_zone. PASS: the E1 change at 72 h goes with the chosen sign
        (pooled sum > 0), d >= +0.01 AND d > 0 in >= 95% of 2,000 bootstrap resamples of bins.
        Controls: E1 change at 6 h replaced by F_act x (72 h change) + noise (SD of the 6 h E1 change among the half
        of bins with the smallest H3K27ac change; 200 runs) must pass <= 10%; replaced by (F_act - 0.25) x (72 h change)
        + noise must pass >= 90%.
 REPORTED: D1 weights and r of single features; D2 shares at 6 h for JUN, ATAC, CTCF as well; E1 with promoter H3K27ac
        change partialled out; numbers of loops / pairs / bins used.
NOT: one cell line, one stimulus (PMA); three time points (6 h is the only in-between); sites and loops are the authors'
calls; CUT&RUN and ATAC counts are relative (size-factor normalised), so a global rise of a mark is not seen;
correlation in time is not proof of cause.

Run: python3 -I rem/atlas/loopdyntest.py declare | test [--trial]
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


kt = _load("k3dtest")
gx, cbb, Fact, gain = kt.gx, kt.cbb, kt.Fact, kt.gain
LEDGER = gx.LEDGER
OUT = R("outputs", "loopcode")
CLAIMS = ["G20-D1", "G20-D2", "G20-E1", "G20-M2", "G20-M3"]
LAYERS = ["LOOPCODE_K562", "K562_PMA_3D"]
SEED = 20261202
DEV = [str(c) for c in range(1, 23, 2)]; EXAM = [str(c) for c in range(2, 23, 2)]
NNEG, NPL, NSIM = (3, 2, 20) if TRIAL else (10, 5, 200)
N_BOOT = 500 if TRIAL else 2000
WIN, SEL = 5000, 0.5
ASSAYS = ["ctcf", "rad21", "h3k27ac", "jun", "atac"]
T0, T6, T72 = 0, 360, 4320


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "loopdyntest.py"), "rb").read()); h.update(open(os.path.join(HERE, "k3dtest.py"), "rb").read())
    for n in LAYERS:
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else 0.0


z = lambda v: (v - v.mean()) / (v.std() + 1e-12)


# ---------------------------------------------------------------- data
class Sites:
    """one assay's sites: normalised counts with prefix sums per chromosome for window sums."""
    def __init__(self, A, a):
        C = np.nan_to_num(A[f"{a}_counts"].astype(float)); good = (C > 0).all(1)
        sf = np.median(C[good] / np.exp(np.log(C[good]).mean(1))[:, None], 0); C = C / sf
        self.cols = [tuple(map(int, c.split("_"))) for c in A[f"{a}_cols"]]; self.reps = sorted({r for _, r in self.cols})
        ch = A[f"{a}_chrom"]; ce = (A[f"{a}_start"] + A[f"{a}_stop"]) // 2; self.by = {}
        for c in np.unique(ch):
            m = np.nonzero(ch == c)[0]; o = np.argsort(ce[m]); m = m[o]
            self.by[c] = (ce[m], np.concatenate([np.zeros((1, C.shape[1])), np.cumsum(C[m], 0)]))

    def window(self, chrom, lo, hi):
        """log2(sum + 1) per window and column; and the number of sites."""
        out = np.zeros((len(chrom), len(self.cols))); n = np.zeros(len(chrom), int)
        for c in np.unique(chrom):
            if c not in self.by:
                continue
            m = np.nonzero(chrom == c)[0]; ce, cs = self.by[c]; i0, i1 = np.searchsorted(ce, lo[m]), np.searchsorted(ce, hi[m])
            out[m] = cs[np.maximum(i1, i0)] - cs[i0]; n[m] = np.maximum(i1 - i0, 0)
        return np.log2(out + 1), n

    def col(self, t, k):
        """column index of time t, replicate number k (0 / 1)."""
        return self.cols.index((t, self.reps[k]))

    def change(self, W, t=T72):
        return np.mean([W[:, self.col(t, k)] - W[:, self.col(T0, k)] for k in (0, 1)], 0)


def load():
    A = np.load(os.path.join(cbb.LAY, "loopcode_k562.npz")); Z = np.load(os.path.join(cbb.LAY, "k562_pma_3d.npz")); gx.USED.update(LAYERS)
    S = {a: Sites(A, a) for a in ASSAYS}
    LA = Z["loop_anchor"]; L = dict(chrom=LA[:, 0], a0=LA[:, 1].astype(np.int64), a1=LA[:, 2].astype(np.int64), b0=LA[:, 4].astype(np.int64), b1=LA[:, 5].astype(np.int64),
                                    lfc=Z["loop_lfc"][:, 1].astype(float))
    sw = L["a0"] > L["b0"]                                # a few loops list their ends right-to-left: put the left end first
    for x, y in (("a0", "b0"), ("a1", "b1")):
        L[x], L[y] = np.where(sw, L[y], L[x]), np.where(sw, L[x], L[y])
    W = {a: (S[a].window(L["chrom"], L["a0"] - WIN, L["a1"] + WIN)[0], S[a].window(L["chrom"], L["b0"] - WIN, L["b1"] + WIN)[0]) for a in ASSAYS}
    WI = S["h3k27ac"].window(L["chrom"], L["a1"] + WIN, L["b0"] - WIN)[0]
    cols = [tuple(map(int, c.split("_"))) for c in Z["rna_cols"]]; e = np.log2(Z["tpm"].astype(float) + 1)
    em = {t: np.mean([e[:, k] for k, (tt, r) in enumerate(cols) if tt == t], 0) for t in (T0, T6, T72)}
    tp = {t: np.mean([Z["tpm"][:, k] for k, (tt, r) in enumerate(cols) if tt == t], 0) for t in (T0, T72)}
    G = dict(chrom=Z["chrom"], tss=Z["tss"].astype(np.int64), expressed=(tp[T0] >= 1) | (tp[T72] >= 1), change=em[T72] - em[T0], gene=Z["gene"])
    return dict(A=A, Z=Z, S=S, L=L, W=W, WI=WI, G=G)


def ridge_r(Xd, yd, Xe, ye, ce, chroms):
    from sklearn.linear_model import Ridge
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9; mod = Ridge(alpha=1.0).fit((Xd - mu) / sd, yd); pr = mod.predict((Xe - mu) / sd)
    return np.array([spearman(pr[ce == c], ye[ce == c]) for c in chroms]), mod.coef_


# ---------------------------------------------------------------- D1
def d1(D, dev, exam, rng):
    L, S, W = D["L"], D["S"], D["W"]; ok = np.isfinite(L["lfc"]) & np.isin(L["chrom"], dev + exam)
    size = np.log10(L["b0"] - L["a0"]); feats, names = {}, {}
    for a in ASSAYS:
        dl, dr = S[a].change(W[a][0]), S[a].change(W[a][1]); big = np.where(np.abs(dl) >= np.abs(dr), dl, dr)
        feats[a] = np.stack([(dl + dr) / 2, big], 1); names[a] = [f"{a} mean of anchors", f"{a} larger anchor"]
    inside = S["h3k27ac"].change(D["WI"])
    arch = np.concatenate([size[:, None], feats["ctcf"], feats["rad21"]], 1); act = np.concatenate([feats["h3k27ac"], feats["jun"], feats["atac"], inside[:, None]], 1)
    nm = ["loop size"] + names["ctcf"] + names["rad21"] + names["h3k27ac"] + names["jun"] + names["atac"] + ["h3k27ac inside the loop"]
    d = ok & np.isin(L["chrom"], dev); e = ok & np.isin(L["chrom"], exam); ce = L["chrom"][e]

    def run(act, y):
        full = np.concatenate([arch, act], 1); rf, coef = ridge_r(full[d], y[d], full[e], y[e], ce, exam); ra, _ = ridge_r(arch[d], y[d], arch[e], y[e], ce, exam)
        g_ok, p, med, n = gain(rf - ra)
        return dict(loops_exam=int(e.sum()), median_r_full=float(np.median(rf)), median_r_arch=float(np.median(ra)), sign_p=p, median_gain=med, chromosomes=n,
                    passed=bool(np.median(rf) >= 0.15 and g_ok)), coef
    res, coef = run(act, L["lfc"]); res["weights"] = dict(zip(nm, map(float, coef)))
    res["single_feature_r_exam"] = {n_: spearman(np.concatenate([arch, act], 1)[e][:, k], L["lfc"][e]) for k, n_ in enumerate(nm)}
    neg = pl = 0
    for _ in range(NNEG):
        sh = act.copy()
        for c in np.unique(L["chrom"]):
            m = np.nonzero(L["chrom"] == c)[0]; sh[m] = act[rng.permutation(m)]
        neg += run(sh, L["lfc"])[0]["passed"]
    sd = float(np.std(L["lfc"][ok]))
    for _ in range(NPL):
        y = np.full(len(ok), np.nan); y[ok] = L["lfc"][ok][rng.permutation(int(ok.sum()))] + 0.5 * sd * z(feats["h3k27ac"][ok, 0])
        pl += run(act, y)[0]["passed"]
    res["controls"] = dict(permuted_pass_rate=neg / NNEG, planted_pass_rate=pl / NPL, runs=[NNEG, NPL], ok=bool(neg / NNEG <= 0.10 and pl / NPL >= 0.9))
    return res


# ---------------------------------------------------------------- D2 / M3 timing machinery
def pooled(sg, a6, a72, b6, b72, unit, rng):
    """F_a = sum(sg a6) / sum(sg a72), F_b likewise; d = F_a - F_b; bootstrap over units."""
    u, ui = np.unique(unit, return_inverse=True); S_ = np.stack([np.bincount(ui, weights=sg * x, minlength=len(u)) for x in (a6, a72, b6, b72)], 1)
    tot = S_.sum(0)
    if len(u) < 5 or tot[1] <= 0 or tot[3] <= 0:
        return dict(units=int(len(u)), rows=int(len(sg)), F_a=None, F_b=None, d=0.0, p_boot=1.0, denominators_positive=False, passed=False)
    Fa, Fb = tot[0] / tot[1], tot[2] / tot[3]; Wt = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=N_BOOT); B = Wt @ S_
    with np.errstate(divide="ignore", invalid="ignore"):
        db = B[:, 0] / B[:, 1] - B[:, 2] / B[:, 3]
    p = float((1 + np.sum(~(db > 0))) / (N_BOOT + 1))
    return dict(units=int(len(u)), rows=int(len(sg)), F_a=float(Fa), F_b=float(Fb), d=float(Fa - Fb), d_boot_5=float(np.nanquantile(db, 0.05)), p_boot=p,
                denominators_positive=True, passed=bool(p < 0.05 and Fa - Fb >= 0.01))


def d2(D, exam, rng):
    L, S, W, Z = D["L"], D["S"], D["W"], D["Z"]; LH, N = kt.t1_data({k: Z[k] for k in ("loop_libs", "loop_counts")})
    inex = np.isin(L["chrom"], exam)

    def anchors(a, t, k):
        return (W[a][0][:, S[a].col(t, k)] + W[a][1][:, S[a].col(t, k)]) / 2

    def rows(a):
        out = []
        for f, (hs, ks, hm, km) in enumerate((("A", 0, "B", 1), ("B", 1, "A", 0))):
            dls = LH[(hs, T72)] - LH[(hs, T0)]; dys = anchors(a, T72, ks) - anchors(a, T0, ks)
            ch = inex & (np.abs(dls) >= SEL) & (np.abs(dys) >= SEL) & (np.sign(dls) == np.sign(dys)); k = np.nonzero(ch)[0]
            sl = np.stack([1 / (np.log(2) * np.sqrt(N[(hm, t)][k] + 1)) for t in (T0, T6)], 1)
            out.append(np.stack([k, np.sign(dls[k]), anchors(a, T6, km)[k] - anchors(a, T0, km)[k], anchors(a, T72, km)[k] - anchors(a, T0, km)[k],
                                 LH[(hm, T6)][k] - LH[(hm, T0)][k], LH[(hm, T72)][k] - LH[(hm, T0)][k], np.sqrt((sl ** 2).sum(1))], 1))
        return np.concatenate(out)
    Rw = rows("h3k27ac"); st = lambda Rw, l6=None: pooled(Rw[:, 1], Rw[:, 2], Rw[:, 3], Rw[:, 4] if l6 is None else l6, Rw[:, 5], Rw[:, 0].astype(int), rng)
    res = st(Rw); res["F_act"], res["F_loop"] = res.pop("F_a"), res.pop("F_b")
    eq = pl = 0
    if res["denominators_positive"]:
        for _ in range(NSIM):
            eq += st(Rw, res["F_act"] * Rw[:, 5] + rng.normal(0, Rw[:, 6]))["passed"]; pl += st(Rw, (res["F_act"] - 0.25) * Rw[:, 5] + rng.normal(0, Rw[:, 6]))["passed"]
    res["controls"] = dict(equal_timing_pass_rate=eq / NSIM, planted_lag_pass_rate=pl / NSIM, runs=NSIM, ok=bool(eq / NSIM <= 0.10 and pl / NSIM >= 0.9))
    res["other_marks_share_at_6h"] = {}
    for a in ("jun", "atac", "ctcf"):
        r_ = rows(a); x = st(r_); res["other_marks_share_at_6h"][a] = dict(loops=x["units"], F_mark=x["F_a"], F_loop=x["F_b"], d=x["d"], p_boot=x["p_boot"])
    return res


# ---------------------------------------------------------------- E1
def e1(D, exam, rng):
    L, S, G = D["L"], D["S"], D["G"]; H = S["h3k27ac"]; pairs = []
    for c in exam:
        gi = np.nonzero((G["chrom"] == c) & G["expressed"])[0]; li = np.nonzero(L["chrom"] == c)[0]
        if not len(gi) or not len(li):
            continue
        ce = H.by[c][0]
        for side, (s0, s1, o0, o1) in enumerate(((L["a0"], L["a1"], L["b0"], L["b1"]), (L["b0"], L["b1"], L["a0"], L["a1"]))):
            o = np.argsort(s0[li]); lo = li[o]; st = s0[lo]
            for g in gi:
                t = G["tss"][g]
                for k in lo[np.searchsorted(st, t - 20000 - 10000):np.searchsorted(st, t + 10000, side="right")]:
                    if s0[k] - 10000 <= t < s1[k] + 10000:
                        pairs.append((g, k, side, (o0[k] + o1[k]) // 2))
    P = np.array(pairs, np.int64); g, k, far = P[:, 0], P[:, 1], P[:, 3]; ch = G["chrom"][g]; tss = G["tss"][g]
    Wd, nd = H.window(ch, far - 5000 - WIN, far + 5000 + WIN); dist = np.abs(far - tss); keep = (nd >= 1) & (dist > 20000)
    # matched control: H3K27ac site at about the same distance, away from the anchors of the gene's loops
    banned = {}
    for gg, kk in zip(g, k):
        banned.setdefault(int(gg), []).extend([(L["a0"][kk] + L["a1"][kk]) // 2, (L["b0"][kk] + L["b1"][kk]) // 2])
    ctrl = np.full(len(g), -1, np.int64)
    for i in np.nonzero(keep)[0]:
        ce = H.by[ch[i]][0]; cand = np.concatenate([ce[np.searchsorted(ce, tss[i] + 0.75 * dist[i]):np.searchsorted(ce, tss[i] + 1.25 * dist[i])],
                                                    ce[np.searchsorted(ce, tss[i] - 1.25 * dist[i]):np.searchsorted(ce, tss[i] - 0.75 * dist[i])]])
        if len(cand):
            b = np.array(banned[int(g[i])]); cand = cand[(np.abs(cand[:, None] - b[None, :]) > 15000).all(1)]
        if len(cand):
            ctrl[i] = cand[np.argmin(np.abs(np.abs(cand - tss[i]) - dist[i]))]
    keep &= ctrl >= 0; g, k, ch, tss, far, ctrl = g[keep], k[keep], ch[keep], tss[keep], far[keep], ctrl[keep]
    dl = H.change(Wd[keep]); dc = H.change(H.window(ch, ctrl - 10000, ctrl + 10000)[0]); dp = H.change(H.window(ch, tss - 2500, tss + 2500)[0]); y = G["change"][g]

    def stat(dl, y):
        r1_, r2_ = spearman(dl, y), spearman(dc, y); u, ui = np.unique(g, return_inverse=True); idx = [np.nonzero(ui == q)[0] for q in range(len(u))]; ds = []
        for _ in range(N_BOOT if not TRIAL else 200):
            b = np.concatenate([idx[q] for q in rng.integers(0, len(u), len(u))]); ds.append(spearman(dl[b], y[b]) - spearman(dc[b], y[b]))
        p = float((1 + np.sum(np.array(ds) <= 0)) / (len(ds) + 1))
        return dict(pairs=int(len(g)), genes=int(len(u)), r_looped=r1_, r_control=r2_, difference=r1_ - r2_, p_boot=p, passed=bool(r1_ - r2_ >= 0.01 and p < 0.05))
    res = stat(dl, y)
    from sklearn.linear_model import LinearRegression
    resid = lambda v: v - LinearRegression().fit(dp[:, None], v).predict(dp[:, None])
    res["with_promoter_change_removed"] = dict(r_looped=spearman(resid(dl), resid(y)), r_control=spearman(resid(dc), resid(y)))
    res["median_distance_kb"] = float(np.median(np.abs(far - tss)) / 1000)
    neg = sum(stat(dl[rng.permutation(len(dl))], y)["passed"] for _ in range(NNEG)); pl = 0
    for _ in range(NPL):
        yp = y[rng.permutation(len(y))]; pl += stat(dl, yp + 0.5 * np.std(y) * z(dl))["passed"]
    res["controls"] = dict(permuted_pass_rate=neg / NNEG, planted_pass_rate=pl / NPL, runs=[NNEG, NPL], ok=bool(neg / NNEG <= 0.10 and pl / NPL >= 0.9))
    return res


# ---------------------------------------------------------------- M2 / M3 (zones)
def zones(D, dev, exam, rng):
    A, Z, S, G = D["A"], D["Z"], D["S"], D["G"]; bc, bs = A["bin_chrom"], A["bin_start"].astype(np.int64); E = Z["e1"].astype(float)
    ok = np.isfinite(E).all(1) & np.isin(bc, dev + exam); dE72, dE6 = E[:, 2] - E[:, 0], E[:, 1] - E[:, 0]
    Wb = {a: S[a].window(bc, bs, bs + 100000)[0] for a in ("h3k27ac", "atac", "jun", "ctcf")}
    bkey = {(c, s): k for k, (c, s) in enumerate(zip(bc, bs))}; gs = np.zeros(len(bc)); gn = np.zeros(len(bc))
    for c, t, x, ex in zip(G["chrom"], G["tss"], G["change"], G["expressed"]):
        k = bkey.get((c, int(t) // 100000 * 100000))
        if k is not None and ex:
            gs[k] += x; gn[k] += 1
    X = np.stack([S[a].change(Wb[a]) for a in ("h3k27ac", "atac", "jun", "ctcf")] + [gs / np.maximum(gn, 1)], 1)
    names = ["H3K27ac", "open DNA (ATAC)", "JUN", "CTCF", "mean gene change"]; d = ok & np.isin(bc, dev); e = ok & np.isin(bc, exam)

    def m2(X, y):
        r, coef = ridge_r(X[d], y[d], X[e], y[e], bc[e], exam); g_ok, p, med, n = gain(r)
        return dict(bins_exam=int(e.sum()), median_r=float(np.median(r)), sign_p=p, chromosomes=n, passed=bool(np.median(r) >= 0.10 and g_ok)), coef

    def shift(v, k=100):
        out = v.copy()
        for c in np.unique(bc):
            ix = np.nonzero(bc == c)[0]; out[ix] = np.roll(v[ix], k, axis=0)
        return out
    M2, coef = m2(X, np.nan_to_num(dE72)); M2["weights"] = dict(zip(names, map(float, coef)))
    M2["single_feature_r_exam"] = {n_: spearman(X[e][:, k], dE72[e]) for k, n_ in enumerate(names)}
    neg = m2(shift(X), np.nan_to_num(dE72))[0]["passed"]; pl = 0; sdy = np.nanstd(dE72[ok])
    for _ in range(NPL):
        y = shift(np.nan_to_num(dE72), int(rng.integers(50, 200))) + 0.5 * sdy * z(X[:, 0]); pl += m2(X, y)[0]["passed"]
    M2["controls"] = dict(negative_passed=bool(neg), planted_pass_rate=pl / NPL, ok=bool(not neg and pl / NPL >= 0.9))

    H = S["h3k27ac"]; W = Wb["h3k27ac"]; rows = []
    for ks, km in ((0, 1), (1, 0)):
        ds = W[:, H.col(T72, ks)] - W[:, H.col(T0, ks)]; k = np.nonzero(e & (np.abs(ds) >= SEL))[0]
        rows.append(np.stack([k, np.sign(ds[k]), W[k, H.col(T6, km)] - W[k, H.col(T0, km)], W[k, H.col(T72, km)] - W[k, H.col(T0, km)], dE6[k], dE72[k]], 1))
    Rw = np.concatenate(rows); st = lambda z6=None: pooled(Rw[:, 1], Rw[:, 2], Rw[:, 3], Rw[:, 4] if z6 is None else z6, Rw[:, 5], Rw[:, 0].astype(int), rng)
    M3 = st(); M3["F_act"], M3["F_zone"] = M3.pop("F_a"), M3.pop("F_b")
    quiet = ok & (np.abs(X[:, 0]) <= np.median(np.abs(X[ok, 0]))); sd6 = float(np.std(dE6[quiet])); eq = pl = 0
    if M3["denominators_positive"]:
        for _ in range(NSIM):
            eq += st(M3["F_act"] * Rw[:, 5] + rng.normal(0, sd6, len(Rw)))["passed"]; pl += st((M3["F_act"] - 0.25) * Rw[:, 5] + rng.normal(0, sd6, len(Rw)))["passed"]
    M3["noise_sd_E1_6h"] = sd6
    M3["controls"] = dict(equal_timing_pass_rate=eq / NSIM, planted_lag_pass_rate=pl / NSIM, runs=NSIM, ok=bool(eq / NSIM <= 0.10 and pl / NSIM >= 0.9))
    return M2, M3


# ---------------------------------------------------------------- run
def run(rng, exam):
    D = load(); dev = DEV
    if TRIAL:                                         # trial: fit on six DEV chromosomes, score on the other five (no EXAM data touched)
        dev, exam = DEV[:6], DEV[6:]
    out = {"G20-D1": d1(D, dev, exam, rng), "G20-D2": d2(D, exam, rng), "G20-E1": e1(D, exam, rng)}
    out["G20-M2"], out["G20-M3"] = zones(D, dev, exam, rng)
    for cid in CLAIMS:
        ok = out[cid]["controls"]["ok"]
        out[cid]["gate"] = dict(real=out[cid]["passed"], controls_ok=ok, verdict=("PASS" if out[cid]["passed"] else "FAIL") if ok else "VOID (controls)")
    return out


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def declare():
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(),
                                    claim=__doc__.split(" " + cid)[1].split(" G20-")[0].split(" REPORTED:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G20 already tested")
    t0 = time.time(); out = run(np.random.default_rng(SEED), EXAM)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]),
                  "units:", out[cid].get("units", out[cid].get("pairs", out[cid].get("loops_exam", out[cid].get("bins_exam")))))
        print("trial ran on DEV chromosomes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "loopdyn_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "loopdyntest:G20", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "loopdyn_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
