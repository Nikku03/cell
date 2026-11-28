"""LOOPCODETEST (G19) -- the RULES that fold a chromosome, tested one by one (static rules; what changes them is G20).

Asked by the user: "crack the loop mechanism which is guiding the dynamic chromosome folding ... what guides them, how?"
The model under test (the field's loop-extrusion + compartment picture): a ring-shaped motor (cohesin) reels DNA in from
both sides until each side meets a CTCF protein whose DNA site points INTO the loop; separately, chromatin of the same
activity state sticks together and makes the open / closed zones.
Data (layers LOOPCODE_K562, K562_PMA_3D, LOOPCODE_HCT116; built by loopcodebank.py):
  K562: 66,168 CTCF sites (CUT&RUN, 0 h) with the direction of their DNA motif (JASPAR MA0139.1, best hit, relative score
    >= 0.8 = a BOUND MOTIF SITE; position = motif centre; strength = log2(mean 0 h count + 1)); cohesin (RAD21) count
    at each site (overlapping RAD21 site, 0 h; log2(mean + 1)); 33,914 loops (10 kb anchors); compartment value E1 per
    100 kb bin at 0 h; six histone marks (share of each bin covered by ENCODE peaks); gene positions and RNA.
  HCT116 with a degron on the cohesin subunit RAD21 (ENCODE; Rao et al. 2017): Hi-C untreated, after 6 h auxin (cohesin
    destroyed), and 20 / 40 / 60 / 180 min after auxin wash-out (cohesin returning); 7,304 loops called in untreated cells.
    Loop strength of a set of loops in a condition = E = sum of counts in the 3 x 3 pixels (10 kb) at the loops / sum at
    distance-matched control positions (both ends shifted 100 kb left, and right; mean of the two). E = 1 means no loop.
DNA halves (house rule): DEV = odd autosomes, EXAM = even autosomes; models are fitted on DEV and scored on EXAM; the
trial showed only controls. Gain rule (house rule): one-sided sign test p < 0.05 AND median gain >= +0.01.

CLAIMS (declared before running; ids G19-*):
 G19-R1 CTCF SITES AT THE TWO ENDS OF A LOOP POINT AT EACH OTHER (K562). An anchor's window = its 10 kb bin +- 5 kb. Loops
        on EXAM chromosomes with exactly one bound motif site in each anchor window: CONVERGENT = left site on the +
        strand and right site on the - strand (4 arrangements, chance 0.25). PASS: convergent share >= 0.70 and
        one-sided binomial p < 0.05 against 0.25.
        Controls: strands shuffled among the bound sites of each chromosome (20 runs) must pass <= 10%; planted (in each
        counted loop the left site set to +, the right to -, with probability 0.8; 20 runs) must pass >= 90%.
 G19-R2 THE CTCF MAP PREDICTS WHICH SITES LOOP, AND DIRECTION IS PART OF THE CODE (K562). Candidates: all pairs of bound
        motif sites on one chromosome 20 kb - 2 Mb apart; a pair is a LOOP when the two sites lie in the two anchor
        windows of a called loop. Logistic models fitted on DEV chromosomes: M0 distance only (log distance and its
        square); M1 = M0 + both strengths and their product, cohesin at both sites, and the number and summed strength
        of bound sites in between (direction-blind); M2 = M1 + the arrangement of the pair (convergent / both + / both -
        / divergent) and the summed strength of in-between sites split by direction. Score: average precision per EXAM
        chromosome. PASS: gain rule on AP(M2) - AP(M1) over the 11 EXAM chromosomes.
        Controls: strands shuffled within chromosome, M2 refitted (10 runs) must pass <= 10%; planted truth (each + site
        loops to its nearest downstream - site in range; 20% of these moved to random pairs; 5 runs) must pass >= 90%.
 G19-K1 REMOVE THE MOTOR AND THE LOOPS GO (HCT116, loops on EXAM chromosomes). PASS: E(untreated) >= 1.5; share of
        the loop signal lost after 6 h auxin, 1 - (E(auxin) - 1) / (E(untreated) - 1), >= 0.80; and gain rule on the
        per-loop drop log2((obs_u + 1) / (ctrl_u + 1)) - log2((obs_a + 1) / (ctrl_a + 1)).
        Controls: the left control position treated as the 'loop' (right one as its control) must not pass; planted
        (auxin counts at the loops replaced by its control counts) must pass.
 G19-K2 RETURN THE MOTOR AND THE LOOPS COME BACK (HCT116, EXAM loops). Recovered share at time t after wash-out,
        R(t) = (E(t) - E(auxin)) / (E(untreated) - E(auxin)). PASS: R(180 min) >= 0.50 AND R(180) > R(20) in >= 95% of
        2,000 bootstrap resamples of loops.
        Controls: the auxin map used as both wash-out maps must not pass; planted (untreated as 180 min, auxin as
        20 min) must pass.
 G19-K3 THE ZONES DO NOT NEED THE MOTOR (HCT116). Correlation between E1 of untreated and auxin-treated cells over the
        100 kb bins of each EXAM chromosome. PASS: median r >= 0.90 AND on >= 9 of the 11 chromosomes the matched
        pattern is the auxin map's own first eigenvector.
        Controls: the auxin track shifted 50 bins along each chromosome must not pass; planted (untreated + noise of
        SD 0.2) must pass the r rule.
 G19-M1 THE ZONES FOLLOW THE CHROMATIN MARKS (K562). Ridge regression (alpha 1) fitted on DEV bins, E1(0 h) from: share of
        the bin covered by H3K27ac, H3K4me1, H3K4me3, H3K36me3, H3K27me3, H3K9me3; number of open (ATAC) sites, CTCF
        sites; H3K27ac counts; protein-coding gene starts, expressed gene starts (TPM >= 1), summed log2(TPM + 1).
        Baseline: gene starts only. PASS: median r on EXAM chromosomes >= 0.60 AND gain rule on r(model) - r(baseline)
        over the 11 EXAM chromosomes.
        Controls: model features shifted 100 bins along each chromosome must not pass; planted target
        (z(H3K27ac) + z(H3K36me3) - z(H3K9me3) + noise SD 1; 5 runs) must pass >= 90%.
 REPORTED: the four arrangements; share of loops with bound motif sites at both / one / no anchor; AP of M0, M1, M2,
        base rate, precision among the top-N pairs (N = number of true pairs); M2 weights; E per condition and R(t);
        ridge weights of M1.
NOT: K562 and HCT116 are two cancer lines; loops are those the callers found (the weakest are missed); 'bound motif
site' depends on the 0.8 cut; the motor test removes cohesin for 6 h (cells keep dividing); the zone model is a
correlation with marks, not a proof that marks cause zones.

Run: python3 -I rem/atlas/loopcodetest.py declare | test [--trial]
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
CLAIMS = ["G19-R1", "G19-R2", "G19-K1", "G19-K2", "G19-K3", "G19-M1"]
LAYERS = ["LOOPCODE_K562", "K562_PMA_3D", "LOOPCODE_HCT116"]
SEED = 20261201
DEV = [str(c) for c in range(1, 23, 2)]; EXAM = [str(c) for c in range(2, 23, 2)]
N1 = 5 if TRIAL else 20
N2N, N2P = (3, 2) if TRIAL else (10, 5)
N_BOOT = 500 if TRIAL else 2000
WIN, DMIN, DMAX, REL = 5000, 20_000, 2_000_000, 0.8


def code_sha():
    h = hashlib.sha256(open(os.path.join(HERE, "loopcodetest.py"), "rb").read())
    for n in LAYERS:
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- data
def load():
    A = np.load(os.path.join(cbb.LAY, "loopcode_k562.npz")); Z = np.load(os.path.join(cbb.LAY, "k562_pma_3d.npz")); H = np.load(os.path.join(cbb.LAY, "loopcode_hct116.npz"))
    gx.USED.update(LAYERS)
    rel = (A["motif_score"] - A["motif_min"]) / (A["motif_max"] - A["motif_min"]); ok = rel >= REL
    t0 = np.array([c.split("_")[0] == "0" for c in A["ctcf_cols"]])
    S = dict(chrom=A["ctcf_chrom"][ok], pos=A["motif_pos"][ok] + 9, strand=np.where(A["motif_strand"][ok] == "+", 1, -1),
             s=np.log2(A["ctcf_counts"][ok][:, t0].mean(1) + 1), start=A["ctcf_start"][ok], stop=A["ctcf_stop"][ok])
    r0 = np.array([c.split("_")[0] == "0" for c in A["rad21_cols"]]); rad = np.zeros(len(S["pos"]))
    for c in np.unique(S["chrom"]):
        m = np.nonzero(A["rad21_chrom"] == c)[0]; o = np.argsort(A["rad21_start"][m]); m = m[o]
        st, en, val = A["rad21_start"][m], A["rad21_stop"][m], A["rad21_counts"][m][:, r0].mean(1)
        for i in np.nonzero(S["chrom"] == c)[0]:
            k = np.searchsorted(st, S["stop"][i]) - 1
            if k >= 0 and en[k] > S["start"][i]:
                rad[i] = np.log2(val[k] + 1)
    S["rad"] = rad
    o = np.lexsort((S["pos"], S["chrom"])); S = {k: v[o] for k, v in S.items()}
    LA = Z["loop_anchor"]; L = dict(chrom=LA[:, 0], a0=LA[:, 1].astype(np.int64), a1=LA[:, 2].astype(np.int64), b0=LA[:, 4].astype(np.int64), b1=LA[:, 5].astype(np.int64))
    return dict(A=A, Z=Z, H=H, S=S, L=L)


def loop_sites(S, L, chroms):
    """per loop on `chroms`: indices (into S) of bound motif sites in the left and in the right anchor window."""
    out = []
    for c in chroms:
        si = np.nonzero(S["chrom"] == c)[0]; p = S["pos"][si]
        for k in np.nonzero(L["chrom"] == c)[0]:
            le = si[np.searchsorted(p, L["a0"][k] - WIN):np.searchsorted(p, L["a1"][k] + WIN)]
            ri = si[np.searchsorted(p, L["b0"][k] - WIN):np.searchsorted(p, L["b1"][k] + WIN)]
            out.append((k, le, ri))
    return out


# ---------------------------------------------------------------- R1
def r1_stat(strand, LS):
    from scipy.stats import binomtest
    u = [(le[0], ri[0]) for _, le, ri in LS if len(le) == 1 and len(ri) == 1 and le[0] != ri[0]]
    a = np.array([strand[i] for i, _ in u]); b = np.array([strand[j] for _, j in u]); n = len(u)
    tab = dict(convergent=int(((a > 0) & (b < 0)).sum()), both_plus=int(((a > 0) & (b > 0)).sum()), both_minus=int(((a < 0) & (b < 0)).sum()),
               divergent=int(((a < 0) & (b > 0)).sum()))
    share = tab["convergent"] / max(n, 1); p = float(binomtest(tab["convergent"], n, 0.25, alternative="greater").pvalue) if n else 1.0
    return dict(loops=n, share_convergent=share, p=p, table=tab, passed=bool(share >= 0.70 and p < 0.05)), u


def r1(D, chroms, rng):
    S = D["S"]; LS = loop_sites(S, D["L"], chroms); res, u = r1_stat(S["strand"], LS)
    neg = pl = 0
    for _ in range(N1):
        st = S["strand"].copy()
        for c in np.unique(S["chrom"]):
            m = np.nonzero(S["chrom"] == c)[0]; st[m] = rng.permutation(st[m])
        neg += r1_stat(st, LS)[0]["passed"]
        st = S["strand"].copy()
        for i, j in u:
            if rng.random() < 0.8:
                st[i], st[j] = 1, -1
            else:
                st[i], st[j] = rng.choice([1, -1]), rng.choice([1, -1])
        pl += r1_stat(st, LS)[0]["passed"]
    n_all = len(LS); both = sum(1 for _, le, ri in LS if len(le) and len(ri)); one = sum(1 for _, le, ri in LS if (len(le) > 0) != (len(ri) > 0))
    res["anchors"] = dict(loops=n_all, sites_at_both=both / n_all, at_one=one / n_all, at_none=1 - (both + one) / n_all)
    res["controls"] = dict(shuffled_pass_rate=neg / N1, planted_pass_rate=pl / N1, runs=N1, ok=bool(neg / N1 <= 0.10 and pl / N1 >= 0.9))
    return res


# ---------------------------------------------------------------- R2
def pairs(S, strand, chroms):
    """candidate pairs with features; returns dict of arrays (+ site indices I, J and chromosome)."""
    out = {k: [] for k in ("I", "J", "chrom", "logd", "si", "sj", "ri", "rj", "nb", "sb", "pb", "mb", "oi", "oj")}
    for c in chroms:
        g = np.nonzero(S["chrom"] == c)[0]; p = S["pos"][g]; s = S["s"][g]; st = strand[g]
        lo = np.searchsorted(p, p + DMIN, side="left"); hi = np.searchsorted(p, p + DMAX, side="right"); cnt = np.maximum(hi - lo, 0)
        I = np.repeat(np.arange(len(g)), cnt); J = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a]) if cnt.sum() else np.zeros(0, int)
        cs, cp, cm = (np.concatenate([[0], np.cumsum(x)]) for x in (s, s * (st > 0), s * (st < 0)))
        between = lambda q: q[J] - q[I + 1]
        out["I"].append(g[I]); out["J"].append(g[J]); out["chrom"].append(np.full(len(I), c)); out["logd"].append(np.log10(p[J] - p[I]))
        out["si"].append(s[I]); out["sj"].append(s[J]); out["ri"].append(S["rad"][g][I]); out["rj"].append(S["rad"][g][J])
        out["nb"].append(np.log1p(J - I - 1)); out["sb"].append(between(cs)); out["pb"].append(between(cp)); out["mb"].append(between(cm))
        out["oi"].append(st[I]); out["oj"].append(st[J])
    return {k: np.concatenate(v) for k, v in out.items()}


def design(P):
    m1 = np.stack([P["logd"], P["logd"] ** 2, P["si"], P["sj"], P["si"] * P["sj"], P["ri"], P["rj"], P["nb"], P["sb"]], 1)
    conv = (P["oi"] > 0) & (P["oj"] < 0); pp = (P["oi"] > 0) & (P["oj"] > 0); mm = (P["oi"] < 0) & (P["oj"] < 0)
    m2 = np.concatenate([m1, np.stack([conv, pp, mm, P["pb"], P["mb"], conv * P["si"], conv * P["sj"]], 1).astype(float)], 1)
    return m1[:, :2], m1, m2


M2_NAMES = ["log distance", "log distance^2", "strength left", "strength right", "strength product", "cohesin left", "cohesin right", "sites between (log)",
            "strength between", "convergent", "both +", "both -", "+ strength between", "- strength between", "convergent x strength left", "convergent x strength right"]


def fit_ap(Xd, yd, Xe, ye, ce):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=300).fit((Xd - mu) / sd, yd); pr = clf.predict_proba((Xe - mu) / sd)[:, 1]
    ap = {c: float(average_precision_score(ye[ce == c], pr[ce == c])) for c in np.unique(ce) if ye[ce == c].sum() > 0}
    return ap, pr, clf.coef_[0]


def r2_run(S, strand, y_of, dev, exam):
    Pd, Pe = pairs(S, strand, dev), pairs(S, strand, exam); yd, ye = y_of(Pd), y_of(Pe)
    Xd, Xe = design(Pd), design(Pe); res = {}
    for name, k in (("M0", 0), ("M1", 1), ("M2", 2)):
        res[name] = fit_ap(Xd[k], yd, Xe[k], ye, Pe["chrom"])
    g = np.array([res["M2"][0][c] - res["M1"][0][c] for c in sorted(res["M1"][0], key=int)]); ok, p, med, n = gain(g)
    return dict(passed=bool(ok), sign_p=p, median_gain=med, chromosomes=n), res, (Pe, ye)


def r2(D, dev, exam, rng):
    from sklearn.metrics import average_precision_score
    S = D["S"]; LS = loop_sites(S, D["L"], dev + exam); N = len(S["pos"])
    keys = np.array(sorted({int(i) * N + int(j) for _, le, ri in LS for i in le for j in ri if i < j}), np.int64)
    truth = lambda P: np.isin(P["I"].astype(np.int64) * N + P["J"], keys)
    out, res, (Pe, ye) = r2_run(S, S["strand"], truth, dev, exam)
    pr = res["M2"][1]; top = np.argsort(-pr)[:int(ye.sum())]
    out.update(pairs_exam=int(len(ye)), true_pairs_exam=int(ye.sum()), base_rate=float(ye.mean()),
               ap_pooled={k: float(average_precision_score(ye, res[k][1])) for k in ("M0", "M1", "M2")},
               ap_median_chromosome={k: float(np.median(list(res[k][0].values()))) for k in ("M0", "M1", "M2")},
               precision_top_n=float(ye[top].mean()), m2_weights=dict(zip(M2_NAMES, map(float, res["M2"][2]))))
    neg = pl = 0
    for _ in range(N2N):
        st = S["strand"].copy()
        for c in np.unique(S["chrom"]):
            m = np.nonzero(S["chrom"] == c)[0]; st[m] = rng.permutation(st[m])
        neg += r2_run(S, st, truth, dev, exam)[0]["passed"]

    def planted(P):
        y = np.zeros(len(P["I"]), bool); conv = (P["oi"] > 0) & (P["oj"] < 0); ix = np.nonzero(conv)[0]
        pos = ix[np.unique(P["I"][ix], return_index=True)[1]]                 # nearest downstream - site (pairs are ordered by J within I)
        keep = rng.random(len(pos)) >= 0.2; y[pos[keep]] = True; y[rng.integers(0, len(y), int((~keep).sum()))] = True
        return y
    for _ in range(N2P):
        pl += r2_run(S, S["strand"], planted, dev, exam)[0]["passed"]
    out["controls"] = dict(shuffled_pass_rate=neg / N2N, planted_pass_rate=pl / N2P, runs=[N2N, N2P], ok=bool(neg / N2N <= 0.10 and pl / N2P >= 0.9))
    return out


# ---------------------------------------------------------------- K1-K3 (HCT116)
def k_tests(D, chroms, rng):
    H = D["H"]; cond = [str(c) for c in H["conditions"]]; ci = {c: k for k, c in enumerate(cond)}
    m = np.isin(H["loop_chrom"], chroms); O = H["obs"][m].astype(float); CL, CR = H["ctrl_left"][m].astype(float), H["ctrl_right"][m].astype(float)
    C = (CL + CR) / 2; n = int(m.sum())
    E = lambda o, c, ix=slice(None): float(o[ix].sum() / max(c[ix].sum(), 1e-9))

    def k1(ou, cu, oa, ca):
        eu, ea = E(ou, cu), E(oa, ca); lost = 1 - (ea - 1) / (eu - 1) if eu > 1 else 0.0
        drop = np.log2((ou + 1) / (cu + 1)) - np.log2((oa + 1) / (ca + 1)); ok, p, med, k = gain(drop)
        return dict(loops=n, E_untreated=eu, E_auxin=ea, share_lost=float(lost), sign_p=p, median_drop_log2=med, passed=bool(eu >= 1.5 and lost >= 0.80 and ok))
    u, a = ci["untreated"], ci["auxin_6h"]
    K1 = k1(O[:, u], C[:, u], O[:, a], C[:, a])
    negc = k1(CL[:, u], CR[:, u], CL[:, a], CR[:, a]); plc = k1(O[:, u], C[:, u], C[:, a], C[:, a])
    K1["controls"] = dict(negative_passed=negc["passed"], planted_passed=plc["passed"], ok=bool(not negc["passed"] and plc["passed"]))
    K1["E_by_condition"] = {c: E(O[:, ci[c]], C[:, ci[c]]) for c in cond}

    def k2(o20, c20, o180, c180):
        def rr(ix):
            eu, ea = E(O[:, u], C[:, u], ix), E(O[:, a], C[:, a], ix); d = eu - ea
            return (E(o20, c20, ix) - ea) / d, (E(o180, c180, ix) - ea) / d
        r20, r180 = rr(slice(None)); B = [rr(rng.integers(0, n, n)) for _ in range(N_BOOT)]
        pb = float((1 + sum(1 for x, y in B if not y > x)) / (N_BOOT + 1))
        return dict(R_20=float(r20), R_180=float(r180), p_boot=pb, passed=bool(r180 >= 0.5 and pb < 0.05))
    w = lambda t: (O[:, ci[f"washout_{t}"]], C[:, ci[f"washout_{t}"]])
    K2 = k2(*w(20), *w(180)); eu, ea = E(O[:, u], C[:, u]), E(O[:, a], C[:, a])
    K2["R_by_time"] = {str(t): float((E(*w(t)) - ea) / (eu - ea)) for t in (20, 40, 60, 180)}
    K2["controls"] = dict(negative_passed=k2(O[:, a], C[:, a], O[:, a], C[:, a])["passed"], planted_passed=k2(O[:, a], C[:, a], O[:, u], C[:, u])["passed"])
    K2["controls"]["ok"] = bool(not K2["controls"]["negative_passed"] and K2["controls"]["planted_passed"])

    e1 = H["e1"].astype(float); bc = H["bin_chrom"]

    def k3(x, y, need_pc=True):
        rs = []
        for c in chroms:
            mm = (bc == c) & np.isfinite(x) & np.isfinite(y); rs.append(float(np.corrcoef(x[mm], y[mm])[0, 1]))
        pcs = [int(H["pc_auxin"][list(H["pc_chrom"]).index(c)]) for c in chroms]; own = sum(1 for q in pcs if q == 1)
        return dict(median_r=float(np.median(rs)), min_r=float(min(rs)), own_first_eigenvector=own, chromosomes=len(chroms),
                    passed=bool(np.median(rs) >= 0.90 and (own >= len(chroms) - 2 or not need_pc)))
    K3 = k3(e1[:, 0], e1[:, 1]); sh = e1[:, 1].copy()
    for c in chroms:
        ix = np.nonzero(bc == c)[0]; sh[ix] = np.roll(e1[ix, 1], 50)
    K3["controls"] = dict(negative_passed=k3(e1[:, 0], sh)["passed"], planted_passed=k3(e1[:, 0], e1[:, 0] + rng.normal(0, 0.2, len(e1)), need_pc=False)["passed"])
    K3["controls"]["ok"] = bool(not K3["controls"]["negative_passed"] and K3["controls"]["planted_passed"])
    return K1, K2, K3


# ---------------------------------------------------------------- M1 (K562 zones from marks)
def m1(D, dev, exam, rng):
    from sklearn.linear_model import Ridge
    A, Z = D["A"], D["Z"]; bc, bs = A["bin_chrom"], A["bin_start"]; nb = len(bc); e1 = Z["e1"][:, 0].astype(float)
    bkey = {(c, s): k for k, (c, s) in enumerate(zip(bc, bs))}

    def per_bin(chrom, pos, w=None):
        v = np.zeros(nb)
        for c, p_, ww in zip(chrom, pos, np.ones(len(pos)) if w is None else w):
            k = bkey.get((c, int(p_) // 100000 * 100000))
            if k is not None:
                v[k] += ww
        return v
    t0 = lambda cols: np.array([c.split("_")[0] == "0" for c in cols])
    tpm = Z["tpm"][:, t0(Z["rna_cols"])].mean(1)
    feats = [A["mark_cover"][:, k] for k in range(A["mark_cover"].shape[1])]
    feats += [np.log1p(per_bin(A["atac_chrom"], (A["atac_start"] + A["atac_stop"]) // 2)), np.log1p(per_bin(A["ctcf_chrom"], (A["ctcf_start"] + A["ctcf_stop"]) // 2)),
              np.log1p(per_bin(A["h3k27ac_chrom"], (A["h3k27ac_start"] + A["h3k27ac_stop"]) // 2, A["h3k27ac_counts"][:, t0(A["h3k27ac_cols"])].mean(1)))]
    dens = per_bin(Z["chrom"], Z["tss"]); feats += [dens, per_bin(Z["chrom"][tpm >= 1], Z["tss"][tpm >= 1]), per_bin(Z["chrom"], Z["tss"], np.log2(tpm + 1))]
    names = [str(x) for x in A["mark_names"]] + ["open sites (ATAC)", "CTCF sites", "H3K27ac counts", "gene starts", "expressed gene starts", "summed expression"]
    X = np.stack(feats, 1); ok = np.isfinite(e1); d = ok & np.isin(bc, dev); e = ok & np.isin(bc, exam)

    def run(X, y):
        mu, sd = X[d].mean(0), X[d].std(0) + 1e-9; Xs = (X - mu) / sd
        mod = Ridge(alpha=1.0).fit(Xs[d], y[d]); base = Ridge(alpha=1.0).fit(dens[d][:, None], y[d]); pm, pb = mod.predict(Xs), base.predict(dens[:, None])
        rm = np.array([np.corrcoef(pm[e & (bc == c)], y[e & (bc == c)])[0, 1] for c in exam]); rb = np.array([np.corrcoef(pb[e & (bc == c)], y[e & (bc == c)])[0, 1] for c in exam])
        g_ok, p, med, n = gain(rm - rb)
        return dict(median_r=float(np.median(rm)), median_r_baseline=float(np.median(rb)), sign_p=p, median_gain=med, chromosomes=n,
                    passed=bool(np.median(rm) >= 0.60 and g_ok)), mod.coef_
    res, coef = run(X, e1); res["weights"] = dict(zip(names, map(float, coef))); res["bins_exam"] = int(e.sum())
    Xs = X.copy()
    for c in np.unique(bc):
        ix = np.nonzero(bc == c)[0]; Xs[ix] = np.roll(X[ix], 100, axis=0)
    neg = run(Xs, e1)[0]["passed"]; z = lambda v: (v - v[ok].mean()) / v[ok].std(); pl = 0; NP = 2 if TRIAL else 5
    for _ in range(NP):
        y = np.where(ok, z(X[:, 0]) + z(X[:, 3]) - z(X[:, 5]) + rng.normal(0, 1, nb), np.nan); pl += run(X, y)[0]["passed"]
    res["controls"] = dict(negative_passed=bool(neg), planted_pass_rate=pl / NP, ok=bool(not neg and pl / NP >= 0.9))
    return res


# ---------------------------------------------------------------- run
def run(rng, exam):
    D = load(); dev = DEV
    if TRIAL:                                         # trial: fit on the first six DEV chromosomes, score on the other five (no EXAM data touched)
        dev, exam = DEV[:6], DEV[6:]
    out = {"G19-R1": r1(D, exam, rng), "G19-R2": r2(D, dev, exam, rng)}
    out["G19-K1"], out["G19-K2"], out["G19-K3"] = k_tests(D, exam, rng)
    out["G19-M1"] = m1(D, dev, exam, rng)
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
                                    claim=__doc__.split(" " + cid)[1].split(" G19-")[0].split(" REPORTED:")[0].strip()[:1500], trial=TRIAL), sort_keys=True) + "\n")
            print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G19 already tested")
    t0 = time.time(); out = run(np.random.default_rng(SEED), EXAM)
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]))
        print("trial ran on DEV chromosomes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "loopcode_tests.json"), "w"), indent=1, default=float)
    t = cbb.now(); src = {"name": "loopcodetest:G19", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "loopcode_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"declare": declare, "test": test}[a[0]]()
