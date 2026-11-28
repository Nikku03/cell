"""FOLDSIMTEST (G21) -- does the folding simulator (foldsim.py), built only from the rules of G19 / G20, reproduce what
is measured in real cells on DNA it was not tuned on?

Asked by the user: "Do whatever is needed. Give me final product."
What the simulator is given: the stop-sign map of a cell line (CTCF sites with a clear motif: position, direction,
strength rank q in 0-1), chromosome lengths, and four numbers fitted ONCE on DEV regions of K562 (`fit`): processivity
and separation of the motors, capture scale of a stop sign, exponent alpha of contact with distance; for changes over
time a fifth number beta (how strongly motors prefer to land where H3K27ac is). It is never given a contact.
Data: K562 sites, loops, zones, marks (LOOPCODE_K562, K562_PMA_3D); measured maps of 88 fixed 3 Mb regions at 25 kb in
K562 (FOLDSIM_K562_MAPS) and in HCT116 with and without cohesin, with HCT116's own CTCF sites (FOLDSIM_HCT116).
Halves (house rule): DEV = odd autosomes (all fitting, and the trial), EXAM = even autosomes (scored once).
Map score of a region: Spearman r between simulated and measured distance-normalised maps (both balanced, then each
diagonal divided by its mean), over bin pairs 50 kb - 1.5 Mb apart. A region is USED when >= 108 of its 120 bins
have coverage and it holds >= 15 stop signs. Gain rule (house rule): one-sided sign test p < 0.05 AND median >= +0.01.

CLAIMS (declared before the EXAM run; thresholds were set after the DEV fit; ids G21-*):
 G21-S1 THE SIMULATED MAP LOOKS LIKE THE MEASURED MAP (K562, EXAM regions). PASS: median map score >= {T_S1} AND gain rule
        on score(real stop signs) - score(the same number of stop signs at random places on the chromosome) over regions.
        Controls: each simulated map compared with ANOTHER region's measured map must not pass; planted (measured map
        replaced by the simulated one times log-normal noise SD 0.05) must pass.
 G21-S2 THE SIMULATOR FINDS THE LOOPS (K562). Candidate stop-sign pairs and truth as in G19-R2. Score of a pair = share
        of snapshots with a motor held between the two signs (left leg within 2 sites inside the left sign, right leg
        within 2 sites inside the right sign). PASS: gain rule on average precision (simulator) - average precision
        (distance only, logistic fitted on DEV) over the 11 EXAM chromosomes.
        Controls: the same run with stop-sign directions shuffled must gain less over distance than the real run on
        >= 9 of the 11 chromosomes; planted truth (the pairs the simulator scores highest) must pass.
 G21-S3 THE SIMULATOR PREDICTS WHICH LOOPS CHANGE (K562, 0 -> 72 h of PMA). Two runs: stop-sign strengths and H3K27ac
        landing preference of 0 h, and of 72 h. Predicted change of a called loop = log2((occupancy_72 + 0.002) /
        (occupancy_0 + 0.002)), occupancy = share of snapshots with a motor held between its two anchor windows; loops
        with occupancy > 0 in either run. PASS: median over EXAM chromosomes of Spearman r(predicted, measured log2 fold
        change) >= {T_S3} AND gain rule on r over the 11 chromosomes.
        Controls: 'change' between two independent 0 h runs must not pass; planted (measured change replaced by the
        prediction + noise of the same SD) must pass.
 G21-S4 THE ZONE RULE DRAWS THE LARGE-SCALE PATTERN (K562). Zone value of each 100 kb bin predicted from chromatin marks
        (the G19-M1 model, fitted on DEV); predicted contact pattern = product of the two bins' zone values (like with
        like). Score per EXAM chromosome: Spearman r with the measured distance-normalised 100 kb map over bin pairs
        2 - 50 Mb apart. PASS: median r >= {T_S4} AND gain rule on r over the 11 chromosomes.
        Controls: zone values shifted 100 bins along the chromosome must not pass; planted (measured pattern replaced
        by exp(0.5 x product) times log-normal noise SD 1) must pass.
 G21-S5 ANOTHER CELL LINE, AND TAKING THE MOTOR AWAY (HCT116, EXAM regions). The simulator keeps the K562 numbers and gets
        HCT116's own stop signs. PASS: median map score against the untreated measured map >= {T_S5} AND gain rule on
        score(vs untreated map) - score(vs the map measured after cohesin was destroyed) over regions.
        Controls: stop signs at random places must not pass; planted (untreated map replaced by simulated x noise) must
        pass (noise SD 0.05).
 REPORTED: fitted numbers and DEV scores; map score with shuffled directions; AP of the G19 logistic models next to the
        simulator's; S3 with stop signs only and with landing preference only; recovery of loops after cohesin returns,
        by loop size; share of regions used.
NOT: a map score of about 0.15 arises even with stop signs at random places (balancing a 3 Mb window treats its edge
bins alike in both maps), which is why S1 and S5 also require a gain over a comparison; a 1D simulator with a
distance rule for contacts, not a 3D physical model; one set of numbers for the whole genome;
motors land uniformly except in S3; stop signs without a clear motif are ignored; enhancer-promoter contacts made
without CTCF are not modelled; two cancer cell lines.

Run: python3 -I rem/atlas/foldsimtest.py fit | declare | test [--trial]
"""

from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
TRIAL = "--trial" in sys.argv
T_S1, T_S3, T_S4, T_S5 = 0.20, 0.15, 0.25, 0.20    # set after the DEV fit and preview (DEV values 0.32, 0.24, 0.40, 0.34), before declaring


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


lt = _load("loopcodetest"); ld = _load("loopdyntest"); fs = _load("foldsim")
kt, gx, cbb, Fact, gain = lt.kt, lt.gx, lt.cbb, lt.Fact, lt.gain
LEDGER = gx.LEDGER
OUT = R("outputs", "foldsim")
FIT = os.path.join(OUT, "fit.json")
CLAIMS = ["G21-S1", "G21-S2", "G21-S3", "G21-S4", "G21-S5"]
LAYERS = ["LOOPCODE_K562", "K562_PMA_3D", "LOOPCODE_HCT116", "FOLDSIM_K562_MAPS", "FOLDSIM_HCT116"]
SEED = 20261203
DEV, EXAM = lt.DEV, lt.EXAM
NB, SITE, WIN = 120, fs.SITE, 5000
GRID = dict(processivity_kb=(100, 200, 400), separation_kb=(75, 150, 300), c=(0.5, 1.0), alpha=(0.75, 1.25))
BETAS = (0.0, 2.0, 8.0)
spearman = ld.spearman


def code_sha():
    h = hashlib.sha256()
    for f in ("foldsimtest.py", "foldsim.py", "loopcodetest.py", "loopdyntest.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    h.update(open(FIT, "rb").read())
    for n in LAYERS:
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- data
def load():
    D = lt.load(); A, S = D["A"], D["S"]; gx.USED.update(LAYERS)
    C = ld.Sites(A, "ctcf"); W = C.window(S["chrom"], S["start"], S["stop"])[0]
    v0 = np.mean([W[:, C.col(ld.T0, k)] for k in (0, 1)], 0); v72 = np.mean([W[:, C.col(ld.T72, k)] for k in (0, 1)], 0); srt = np.sort(v0)
    S["q0"] = np.searchsorted(srt, v0, side="right") / len(srt); S["q72"] = np.clip(np.searchsorted(srt, v72, side="right") / len(srt), 0, 1)
    FK = np.load(os.path.join(cbb.LAY, "foldsim_k562_maps.npz")); FH = np.load(os.path.join(cbb.LAY, "foldsim_hct116.npz"))
    D["len"] = dict(zip(map(str, FK["chrom_names"]), map(int, FK["chrom_len"]))); D["FK"], D["FH"] = FK, FH
    rel = (FH["ctcf_score"] - FH["ctcf_min"]) / (FH["ctcf_max"] - FH["ctcf_min"]); ok = rel >= lt.REL; sig = FH["ctcf_signal"][ok]
    D["SH"] = dict(chrom=FH["ctcf_chrom"][ok], pos=FH["ctcf_pos"][ok] + 9, strand=np.where(FH["ctcf_strand"][ok] == "+", 1, -1), q0=(np.argsort(np.argsort(sig)) + 1) / len(sig))
    D["H"] = ld.Sites(A, "h3k27ac")
    return D


def simulate(D, S, c, P, rng, q="q0", load=None, shuffle=None, R_=10, snaps=100):
    m = S["chrom"] == c; pos, strand = S["pos"][m].copy(), S["strand"][m].copy()
    if shuffle == "places":
        pos = np.sort(rng.integers(0, D["len"][c], len(pos)))
    elif shuffle == "directions":
        strand = rng.permutation(strand)
    return fs.chromosome(D["len"][c], pos, strand, S[q][m], P["processivity_kb"], P["separation_kb"], P["c"], load=load, R=R_, snaps=snaps, every=5, rng=rng)


def region_score(P, M):
    """simulated (P) and measured (M) maps go through the same steps: balance, divide each diagonal by its mean, rank-correlate."""
    Mb, valid = fs.balance(M)
    if valid.sum() < 108:
        return None
    Pz = np.where(valid[:, None] & valid[None, :], P, 0.0); Pb, v2 = fs.balance(Pz); valid = valid & v2
    Om, Os = fs.obs_over_exp(Mb, valid), fs.obs_over_exp(Pb, valid); iu = np.triu_indices(NB, 2); sep = iu[1] - iu[0]
    m = (sep <= 60) & np.isfinite(Om[iu]) & np.isfinite(Os[iu])
    return spearman(Os[iu][m], Om[iu][m])


def regions_of(F, S, chroms):
    out = []
    for k, (c, b0) in enumerate(zip(map(str, F["chrom"]), F["bin0"])):
        if c in chroms:
            lo, hi = int(b0) * 25000, (int(b0) + NB) * 25000
            if int(((S["chrom"] == c) & (S["pos"] >= lo) & (S["pos"] < hi)).sum()) >= 15:
                out.append((k, c, int(b0)))
    return out


def sim_maps(D, S, chroms, P, rng, regs, alphas, shuffle=None, max_snaps=1500, R_=20, snaps=150):
    """simulated maps {region index: [map per alpha]} for the regions on `chroms`."""
    out = {}
    for c in chroms:
        rc = [r for r in regs if r[1] == c]
        if not rc:
            continue
        L, Rr, n = simulate(D, S, c, P, rng, shuffle=shuffle, R_=R_, snaps=snaps)
        for k, _, b0 in rc:
            out[k] = fs.contact_map(L, Rr, b0 * 5, b0 * 5 + NB * 5, 5, alphas, max_snaps=max_snaps, rng=rng)
    return out


# ---------------------------------------------------------------- S1 / S5 (maps)
def map_claim(sim, ctl, measured, other, thr, rng, alt=None):
    """sim / ctl: {region: map}; measured: {region: raw map}; other: comparison scores to beat (ctl maps or alt measured)."""
    ks = [k for k in sim if k in measured and region_score(sim[k], measured[k]) is not None]
    r = np.array([region_score(sim[k], measured[k]) for k in ks])
    rc = np.array([region_score(ctl[k], measured[k]) if alt is None else (region_score(sim[k], alt[k]) or 0.0) for k in ks])
    ok, p, med, n = gain(r - rc)
    res = dict(regions=len(ks), median_score=float(np.median(r)) if len(r) else 0.0, median_comparison=float(np.median(rc)) if len(r) else 0.0, sign_p=p, median_gain=med,
               passed=bool(len(r) and np.median(r) >= thr and ok))
    return res, ks, r


def s1(D, exam, P, rng):
    S, FK = D["S"], D["FK"]; regs = regions_of(FK, S, exam); a = (P["alpha"],)
    sim = {k: v[0] for k, v in sim_maps(D, S, exam, P, rng, regs, a).items()}; ctl = {k: v[0] for k, v in sim_maps(D, S, exam, P, rng, regs, a, shuffle="places").items()}
    dirs = {k: v[0] for k, v in sim_maps(D, S, exam, P, rng, regs, a, shuffle="directions").items()}; meas = {k: FK["maps"][k].astype(float) for k in sim}
    res, ks, r = map_claim(sim, ctl, meas, None, T_S1, rng)
    res["median_score_shuffled_directions"] = float(np.median([region_score(dirs[k], meas[k]) for k in ks])) if ks else None
    res["regions_all"] = int(sum(1 for c in map(str, FK["chrom"]) if c in exam))
    rot = dict(zip(ks, ks[1:] + ks[:1])); neg = map_claim(sim, ctl, {k: meas[rot[k]] for k in ks}, None, T_S1, rng)[0]["passed"] if len(ks) > 1 else False
    pl = map_claim(sim, ctl, {k: sim[k] * np.exp(rng.normal(0, 0.05, sim[k].shape)) * 1000 for k in ks}, None, T_S1, rng)[0]["passed"]
    res["controls"] = dict(negative_passed=bool(neg), planted_passed=bool(pl), ok=bool(not neg and pl)); res["scores"] = dict(zip(map(str, ks), map(float, r)))
    return res, sim, meas


def s5(D, exam, P, rng):
    S, FH = D["SH"], D["FH"]; regs = regions_of(FH, S, exam); a = (P["alpha"],)
    sim = {k: v[0] for k, v in sim_maps(D, S, exam, P, rng, regs, a).items()}; ctl = {k: v[0] for k, v in sim_maps(D, S, exam, P, rng, regs, a, shuffle="places").items()}
    mu = {k: FH["maps_untreated"][k].astype(float) for k in sim}; ma = {k: FH["maps_auxin"][k].astype(float) for k in sim}
    res, ks, r = map_claim(sim, None, mu, None, T_S5, rng, alt=ma)
    res["median_score_vs_no_cohesin_map"] = res.pop("median_comparison")
    neg = map_claim(ctl, None, mu, None, T_S5, rng, alt=ma)[0]["passed"]
    pl = map_claim(sim, None, {k: sim[k] * np.exp(rng.normal(0, 0.05, sim[k].shape)) * 1000 for k in ks}, None, T_S5, rng, alt=ma)[0]["passed"]
    res["controls"] = dict(negative_passed=bool(neg), planted_passed=bool(pl), ok=bool(not neg and pl)); res["scores"] = dict(zip(map(str, ks), map(float, r)))
    return res, sim, mu, ma


# ---------------------------------------------------------------- S2 (loops)
def s2(D, dev, exam, P, rng):
    from sklearn.metrics import average_precision_score
    S = D["S"]; LS = lt.loop_sites(S, D["L"], dev + exam); N = len(S["pos"])
    keys = np.array(sorted({int(i) * N + int(j) for _, le, ri in LS for i in le for j in ri if i < j}), np.int64)
    truth = lambda Q: np.isin(Q["I"].astype(np.int64) * N + Q["J"], keys)
    _, res, (Pe, ye) = lt.r2_run(S, S["strand"], truth, dev, exam); x = S["pos"] // SITE

    def sim_scores(shuffle=None):
        sc = np.zeros(len(ye))
        for c in exam:
            L, Rr, n = simulate(D, S, c, P, rng, shuffle=shuffle, R_=20, snaps=150); tab = fs.loop_table(L, Rr, n); m = np.nonzero(Pe["chrom"] == c)[0]
            xi, xj = x[Pe["I"][m]], x[Pe["J"][m]]; sc[m] = fs.occupancy(tab, xi, xi + 2, xj - 2, xj)
        return sc

    def stat(sc, y):
        ap = np.array([average_precision_score(y[Pe["chrom"] == c], sc[Pe["chrom"] == c]) for c in exam]); a0 = np.array([res["M0"][0][c] for c in exam])
        return ap, a0
    sc = sim_scores(); ap, a0 = stat(sc, ye); ok, p, med, n = gain(ap - a0); top = np.argsort(-sc)[:int(ye.sum())]
    out = dict(pairs=int(len(ye)), true_pairs=int(ye.sum()), ap_simulator=float(average_precision_score(ye, sc)), ap_distance=float(average_precision_score(ye, res["M0"][1])),
               ap_logistic_no_direction=float(average_precision_score(ye, res["M1"][1])), ap_logistic_with_direction=float(average_precision_score(ye, res["M2"][1])),
               precision_top_n=float(ye[top].mean()), share_of_true_pairs_with_any_occupancy=float((sc[ye] > 0).mean()), sign_p=p, median_gain=med, passed=bool(ok))
    scd = sim_scores("directions"); apd, _ = stat(scd, ye); worse = int((apd - a0 < ap - a0).sum())
    ypl = np.zeros(len(ye), bool); ypl[np.argsort(-sc)[:int(ye.sum())]] = True; app, _ = stat(sc, ypl)
    out["ap_simulator_shuffled_directions"] = float(average_precision_score(ye, scd))
    out["controls"] = dict(shuffled_directions_worse_on=worse, of=len(exam), planted_passed=bool(gain(app - a0)[0]), ok=bool(worse >= len(exam) - 2 and gain(app - a0)[0]))
    return out


# ---------------------------------------------------------------- S3 (change over time)
def landing(D, c, n, t, beta):
    H = D["H"]; lo = np.arange(n) * SITE; W = H.window(np.full(n, c), lo, lo + SITE)[0]; h = np.mean([2.0 ** W[:, H.col(t, k)] - 1 for k in (0, 1)], 0)
    h0 = np.mean([2.0 ** W[:, H.col(ld.T0, k)] - 1 for k in (0, 1)], 0)
    return 1.0 + beta * h / max(h0.mean(), 1e-9)


def s3_predict(D, chroms, P, beta, rng, ctcf=True, null=False):
    """per chromosome: (measured lfc, predicted change, mask occupancy > 0) for called loops with a measured change."""
    S, L = D["S"], D["L"]; lfc = D["Z"]["loop_lfc"][:, 1].astype(float); out = {}
    for c in chroms:
        n = D["len"][c] // SITE + 1; ld0 = landing(D, c, n, ld.T0, beta) if beta else None; ld72 = landing(D, c, n, ld.T72, beta) if beta else None
        A0 = simulate(D, S, c, P, rng, q="q0", load=ld0, R_=20, snaps=150)
        A1 = simulate(D, S, c, P, rng, q="q0" if (null or not ctcf) else "q72", load=ld0 if null else ld72, R_=20, snaps=150)
        k = np.nonzero((L["chrom"] == c) & np.isfinite(lfc))[0]; win = lambda v0, v1: ((v0 - WIN) // SITE, (v1 + WIN) // SITE)
        la, lb = win(L["a0"][k], L["a1"][k]); ra, rb = win(L["b0"][k], L["b1"][k])
        o0 = fs.occupancy(fs.loop_table(*A0), la, lb, ra, rb); o1 = fs.occupancy(fs.loop_table(*A1), la, lb, ra, rb)
        out[c] = (lfc[k], np.log2((o1 + 0.002) / (o0 + 0.002)), (o0 + o1) > 0)
    return out


def s3_stat(pred, thr, y_of=None):
    rs, n_used, n_all = [], 0, 0
    for c, (y, p, m) in pred.items():
        yy = y if y_of is None else y_of(y, p); rs.append(spearman(p[m], yy[m])); n_used += int(m.sum()); n_all += len(m)
    ok, pv, med, n = gain(np.array(rs))
    return dict(loops_used=n_used, loops_with_measured_change=n_all, median_r=float(np.median(rs)), sign_p=pv, passed=bool(np.median(rs) >= thr and ok))


def s3(D, exam, P, beta, rng):
    pred = s3_predict(D, exam, P, beta, rng); res = s3_stat(pred, T_S3); res["beta"] = beta
    res["stop_signs_only"] = s3_stat(s3_predict(D, exam, P, 0.0, rng), T_S3)["median_r"]
    res["landing_only"] = s3_stat(s3_predict(D, exam, P, beta, rng, ctcf=False), T_S3)["median_r"] if beta else None
    neg = s3_stat(s3_predict(D, exam, P, beta, rng, null=True), T_S3)["passed"]
    pl = s3_stat(pred, T_S3, y_of=lambda y, p: p + rng.normal(0, max(np.std(p), 1e-6), len(p)))["passed"]
    res["controls"] = dict(negative_passed=bool(neg), planted_passed=bool(pl), ok=bool(not neg and pl))
    return res


# ---------------------------------------------------------------- S4 (zones)
def zone_values(D, dev):
    """predicted zone value per 100 kb bin: ridge on chromatin marks fitted on DEV bins (features as G19-M1)."""
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
    t0 = lambda cols: np.array([c.split("_")[0] == "0" for c in cols]); tpm = Z["tpm"][:, t0(Z["rna_cols"])].mean(1)
    feats = [A["mark_cover"][:, k] for k in range(A["mark_cover"].shape[1])]
    feats += [np.log1p(per_bin(A["atac_chrom"], (A["atac_start"] + A["atac_stop"]) // 2)), np.log1p(per_bin(A["ctcf_chrom"], (A["ctcf_start"] + A["ctcf_stop"]) // 2)),
              np.log1p(per_bin(A["h3k27ac_chrom"], (A["h3k27ac_start"] + A["h3k27ac_stop"]) // 2, A["h3k27ac_counts"][:, t0(A["h3k27ac_cols"])].mean(1)))]
    feats += [per_bin(Z["chrom"], Z["tss"]), per_bin(Z["chrom"][tpm >= 1], Z["tss"][tpm >= 1]), per_bin(Z["chrom"], Z["tss"], np.log2(tpm + 1))]
    X = np.stack(feats, 1); d = np.isfinite(e1) & np.isin(bc, dev); mu, sd = X[d].mean(0), X[d].std(0) + 1e-9
    return Ridge(alpha=1.0).fit((X[d] - mu) / sd, e1[d]).predict((X - mu) / sd), bc, bs


def s4(D, dev, exam, rng):
    zv, bc, bs = zone_values(D, dev); H = np.load(os.path.join(lt.kt.HERE, "_cache", "k562pma", "hic", "raw", "intra100k_0.npz")); gx.USED.add("K562PMA_RAW_HIC")

    def run(zv, noise=None):
        rs = []
        for c in exam:
            ix = np.nonzero(bc == c)[0]; n = int(H[f"{c}_j"].max()) + 1; M = np.zeros((n, n)); np.add.at(M, (H[f"{c}_i"], H[f"{c}_j"]), H[f"{c}_v"]); M = M + M.T - np.diag(np.diag(M))
            Mb, valid = fs.balance(M, iters=100); z = np.zeros(n); m_ = min(n, len(ix)); z[:m_] = zv[ix][:m_]
            i = rng.integers(0, n, 400_000); j = i + rng.integers(20, 500, 400_000); ok = (j < n); i, j = i[ok], j[ok]; ok = valid[i] & valid[j]; i, j = i[ok], j[ok]
            d = j - i; obs = Mb[i, j]
            if noise is not None:
                obs = np.exp(0.5 * z[i] * z[j] + rng.normal(0, 1, len(i))) * (d.astype(float) ** -1.0)
            mu = np.bincount(d, weights=obs) / np.maximum(np.bincount(d), 1); rs.append(spearman(z[i] * z[j], obs / np.maximum(mu[d], 1e-12)))
        ok_, p, med, n_ = gain(np.array(rs))
        return dict(median_r=float(np.median(rs)), sign_p=p, chromosomes=n_, passed=bool(np.median(rs) >= T_S4 and ok_))
    res = run(zv); sh = zv.copy()
    for c in np.unique(bc):
        ix = np.nonzero(bc == c)[0]; sh[ix] = np.roll(zv[ix], 100)
    neg = run(sh)["passed"]; pl = run(zv, noise=True)["passed"]
    res["controls"] = dict(negative_passed=bool(neg), planted_passed=bool(pl), ok=bool(not neg and pl))
    return res


def recovery_by_size(D, chroms):
    H = D["H_"] if "H_" in D else np.load(os.path.join(cbb.LAY, "loopcode_hct116.npz")); cond = [str(c) for c in H["conditions"]]; ci = {c: k for k, c in enumerate(cond)}
    m = np.isin(H["loop_chrom"], chroms); O = H["obs"][m].astype(float); C = (H["ctrl_left"][m] + H["ctrl_right"][m]) / 2.0; size = (H["loop_b"] - H["loop_a"])[m]
    q = np.quantile(size, [1 / 3, 2 / 3]); grp = np.searchsorted(q, size); out = {}
    for g, name in enumerate(("short", "middle", "long")):
        ix = grp == g; E = {c: float(O[ix, ci[c]].sum() / C[ix, ci[c]].sum()) for c in cond}
        out[name] = dict(loops=int(ix.sum()), median_size_kb=float(np.median(size[ix]) / 1000), loop_strength=E,
                         share_lost_without_cohesin=float(1 - (E["auxin_6h"] - 1) / (E["untreated"] - 1)))
    return out


# ---------------------------------------------------------------- fit (DEV only)
def fit():
    t0 = time.time(); D = load(); rng = np.random.default_rng(SEED + 1); S, FK = D["S"], D["FK"]; regs = regions_of(FK, S, DEV); rows = []
    for lam, sep, c in itertools.product(GRID["processivity_kb"], GRID["separation_kb"], GRID["c"]):
        P = dict(processivity_kb=lam, separation_kb=sep, c=c); sim = sim_maps(D, S, DEV, P, rng, regs, GRID["alpha"], max_snaps=300, R_=10, snaps=100)
        for q, al in enumerate(GRID["alpha"]):
            r = [region_score(sim[k][q], FK["maps"][k].astype(float)) for k in sim]; r = [x for x in r if x is not None]
            rows.append(dict(P, alpha=al, median_score=float(np.median(r)), regions=len(r)))
        print(f"fit {lam} {sep} {c}: " + " ".join(f"{x['median_score']:.3f}" for x in rows[-len(GRID['alpha']):]), f"({time.time() - t0:.0f}s)", flush=True)
    best = max(rows, key=lambda x: x["median_score"]); P = {k: best[k] for k in ("processivity_kb", "separation_kb", "c", "alpha")}
    betas = {}
    for b in BETAS:
        betas[str(b)] = s3_stat(s3_predict(D, DEV, P, b, rng), -1)["median_r"]; print("beta", b, round(betas[str(b)], 3), flush=True)
    beta = float(max(betas, key=betas.get)); os.makedirs(OUT, exist_ok=True)
    json.dump(dict(params=P, beta=beta, grid=rows, beta_scores=betas, fitted_on="DEV (odd autosomes)", seed=SEED + 1), open(FIT, "w"), indent=1)
    print("chosen", P, "beta", beta)


def evaluate(dev, exam, rng):
    D = load(); F = json.load(open(FIT)); P, beta = F["params"], F["beta"]; out = {}
    out["G21-S1"], sim, meas = s1(D, exam, P, rng); out["G21-S2"] = s2(D, dev, exam, P, rng); out["G21-S3"] = s3(D, exam, P, beta, rng)
    out["G21-S4"] = s4(D, dev, exam, rng); out["G21-S5"], simh, mu, ma = s5(D, exam, P, rng)
    for cid in CLAIMS:
        ok = out[cid]["controls"]["ok"]
        out[cid]["gate"] = dict(real=out[cid]["passed"], controls_ok=ok, verdict=("PASS" if out[cid]["passed"] else "FAIL") if ok else "VOID (controls)")
    rep = dict(params=P, beta=beta, recovery_by_loop_size=recovery_by_size(D, exam))
    return out, rep, dict(sim=simh, untreated=mu, auxin=ma, D=D, P=P)


def preview():
    """DEV only: six DEV chromosomes for fitting the helper models, the other five scored -- to set the thresholds."""
    global T_S1, T_S3, T_S4, T_S5
    T_S1 = T_S3 = T_S4 = T_S5 = -1.0
    out, rep, _ = evaluate(DEV[:6], DEV[6:], np.random.default_rng(SEED + 2))
    for cid in CLAIMS:
        print(cid, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[cid].items() if k not in ("scores", "gate")}, default=float)[:700])
    print(json.dumps(rep["recovery_by_loop_size"], default=float)[:900])


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def claim_text(cid):
    doc = __doc__.format(T_S1=T_S1, T_S3=T_S3, T_S4=T_S4, T_S5=T_S5)
    return doc.split(" " + cid)[1].split(" G21-")[0].split(" REPORTED:")[0].strip()[:1500]


def declare():
    assert None not in (T_S1, T_S3, T_S4, T_S5), "set the thresholds first"
    have = {r["id"] for r in _read() if r["op"] == "declare"}
    with open(LEDGER, "a") as f:
        for cid in CLAIMS:
            if cid in have:
                print(cid, "already declared"); continue
            f.write(json.dumps(dict(op="declare", id=cid, at=cbb.now(), code_sha=code_sha(), commit=gx._git("rev-parse", "HEAD").stdout.strip(), claim=claim_text(cid), trial=TRIAL),
                               sort_keys=True) + "\n"); print(cid, "declared")


def test():
    if not TRIAL:
        head = gx._git("show", "HEAD:rem/atlas/cellbrain_ledger.jsonl").stdout
        dec = {json.loads(x)["id"]: json.loads(x) for x in head.splitlines() if x.strip() and json.loads(x)["op"] == "declare"}
        for cid in CLAIMS:
            if cid not in dec or dec[cid]["code_sha"] != code_sha():
                raise SystemExit(f"{cid}: not declared in HEAD, or the code changed since")
        if any(r["op"] == "result" and r["id"] in CLAIMS for r in _read()):
            raise SystemExit("G21 already tested")
    t0 = time.time(); out, rep, extra = evaluate(*((DEV[:6], DEV[6:]) if TRIAL else (DEV, EXAM)), np.random.default_rng(SEED))
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]))
        print("trial ran on DEV chromosomes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    os.makedirs(OUT, exist_ok=True); json.dump(dict(out, reported=rep), open(os.path.join(OUT, "foldsim_tests.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, "hct116_example_maps.npz"), keys=np.array(sorted(extra["sim"])), sim=np.stack([extra["sim"][k] for k in sorted(extra["sim"])]).astype(np.float32),
                        untreated=np.stack([extra["untreated"][k] for k in sorted(extra["sim"])]).astype(np.float32), auxin=np.stack([extra["auxin"][k] for k in sorted(extra["sim"])]).astype(np.float32),
                        chrom=np.array([str(extra["D"]["FH"]["chrom"][k]) for k in sorted(extra["sim"])]), bin0=np.array([int(extra["D"]["FH"]["bin0"][k]) for k in sorted(extra["sim"])]))
    t = cbb.now(); src = {"name": "foldsimtest:G21", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "foldsim_test", {k: v for k, v in out[cid].items() if k != "scores"}, t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **{k: v for k, v in out[cid].items() if k != "scores"}),
                               sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"fit": fit, "preview": preview, "declare": declare, "test": test}[a[0]]()
