"""FOLDSIM2TEST (G22) -- five ways to make the folding simulator better, each tested.

Asked by the user: "Well do all 5": (1) measure the best possible score, (2) add the missing stop signs and anchors,
(3) test on 'break one part' experiments, (4) make it truly 3D and check against microscope distances, (5) more cells,
including single cells. Also: repair the zone test that came back void in G21.
Simulator versions (foldsim.py). V1 = G21: stop signs with a clear CTCF motif only. V2 adds, with numbers fitted on DEV
chromosomes of K562 (`fit`): weak stop signs (CTCF sites without a clear motif; capture = w x strength rank), leaky signs
(a stalled leg restarts with probability `release` per step), anchors without direction at the stronger half of H3K27ac
sites (capture = e x rank), motors landing more often on active DNA (beta). 3D = the V2 motors on a chain of beads
joined by springs, each motor a spring between its legs, bead positions solved exactly (no volume exclusion; numbers
k_loop, contact radius, optional zone springs k_zone, fitted on DEV).
Data: layers of G19-G21 plus FOLDSIM2_LOCAL (K562 repeat-half maps; microscope traces of single chromosomes, Bintu et al.
2018) and FOLDSIM2_ENCODE (HCT116 CTCF / RNA polymerase II degron maps; GM12878, IMR-90, HepG2 maps and sites).
Halves (house rule): DEV = odd autosomes (all fitting, previews, trial), EXAM = even autosomes (scored once here; the
EXAM regions of K562 and HCT116 were already scored once for V1 in G21, which is the comparison). Map score, USED
regions and the gain rule (one-sided sign test p < 0.05 AND median gain >= +0.01) as in G21.

CLAIMS (declared before the EXAM run; thresholds set after the DEV fit and preview; ids G22-*):
 G22-U1 V2 DRAWS BETTER MAPS THAN V1 (K562, EXAM regions). PASS: gain rule on score(V2) - score(V1) over regions.
        Controls: V2 with all its sites at random places must not pass; planted (measured map replaced by the V2 map x
        log-normal noise SD 0.05) must pass.
 G22-U2 V2 FINDS LOOPS BETTER THAN V1, INCLUDING LOOPS WITHOUT STOP SIGNS (K562). For every called loop: occupancy (share of
        snapshots with a motor held between its two anchor windows) against the same windows moved 100 kb left, and
        right. AUC per chromosome = chance that the loop beats its control (ties half). PASS: gain rule on AUC(V2) -
        AUC(V1) over the 11 EXAM chromosomes.
        Controls: V2 with sites at random places must not pass; planted (V1 with 0.005 added to the occupancy of a
        random fifth of the loops, against V1) must pass.
 G22-B1 TAKE THE STOP-SIGN PROTEIN AWAY AND THE MAP STOPS MATCHING (HCT116 CTCF degron, EXAM regions). The V2 simulator
        with HCT116's own sites. PASS: median score against the untreated map >= {T_B1} AND gain rule on score(vs
        untreated) - score(vs the map after 6 h without CTCF) over regions.
        Controls: sites at random places must not pass; planted (untreated map replaced by simulated x noise) must pass.
 G22-B2 TAKING THE GENE-READING MACHINE AWAY CHANGES THE FOLDING LESS (HCT116, RNA polymerase II degron). Per region: drop_CTCF
        = score(vs CTCF-line untreated) - score(vs CTCF removed); drop_Pol2 likewise for the polymerase line. PASS: gain
        rule on drop_CTCF - drop_Pol2 over EXAM regions.
        Controls: the two lines swapped must not pass; planted (polymerase-removed maps replaced by its untreated maps)
        must pass.
 G22-B3 THE 3D MODEL PREDICTS HOW CONTACT-WITH-DISTANCE CHANGES WHEN A PART IS REMOVED (HCT116 degrons). Curve = log of
        [mean contact at each distance (50 kb - 2.5 Mb) with the part / without], mean over EXAM regions, centred.
        Predicted: 3D model with motors / the same chain with none; and with stop signs / with motors but no CTCF stop
        signs. Measured: untreated / 6 h auxin maps of the cohesin line and of the CTCF line (each map divided by its
        total). PASS: Pearson r(predicted, measured cohesin curve) >= {T_B3} AND the CTCF curve is flatter than the
        cohesin curve (SD ratio < 0.5) in the measurement AND in the prediction.
        Controls: the predicted cohesin curve reversed along distance must not pass; planted (measured curves replaced
        by the predicted ones + noise of a fifth of the cohesin curve's SD) must pass.
 G22-D1 THE 3D MODEL'S MAPS LOOK LIKE THE MEASURED MAPS (K562, EXAM regions). PASS: median score >= {T_D1} AND gain rule on
        score(real sites) - score(sites at random places).
        Controls: each simulated map compared with ANOTHER region's measured map must not pass; planted must pass.
 G22-D2 THE 3D MODEL PREDICTS DISTANCES SEEN UNDER THE MICROSCOPE (K562, chr21:28-30 Mb, 65 segments of 30 kb, about
        10,000 chromosomes). Model: 2,000 sampled shapes; its median distance for every two segments. Both matrices
        divided by the mean of each diagonal; Spearman r over pairs >= 2 segments apart. PASS: r >= 0.01 AND r above
        95% of {N_MC} simulations with the sites at random places (p_MC < 0.05).
        Controls: the model compared with the measured matrix reversed along the region must not reach p_MC < 0.05;
        the model compared with a second run of itself (planted) must.
 G22-D3 UNDER THE MICROSCOPE TOO, THE MATCH NEEDS THE MOTOR (HCT116 cohesin degron, chr21:34.6-37.1 Mb, 83 segments). As
        D2 with HCT116's sites: r_untreated - r_auxin. PASS: difference >= 0.01 AND above 95% of the differences from
        {N_MC} random-place simulations.
        Controls: both measured matrices reversed must not reach p_MC < 0.05; planted (untreated = a second model run,
        auxin = the model without motors) must.
 G22-E1 IT WORKS IN THREE MORE CELL TYPES (GM12878, IMR-90, HepG2; K562's numbers, each cell's own sites; EXAM regions). Per
        cell type: median score >= {T_E1} AND gain rule on score(real) - score(random places). PASS: all three.
        Controls (per cell type): maps compared with another region's measured map must not pass; planted must pass.
 G22-E2 SINGLE CHROMOSOMES PUT THEIR BORDERS WHERE THE MODEL DOES (K562 traces, chr21:28-30 Mb). Border share per position
        (foldsim.border_calls) in the measured single chromosomes against the same in 2,000 sampled model shapes:
        Spearman r over positions. PASS: r >= 0.01 AND p_MC < 0.05 against {N_MC} random-place simulations.
        Controls: the measured profile reversed must not reach p_MC < 0.05; a second model run (planted) must.
 G22-Z1 THE ZONE RULE DRAWS THE LARGE-SCALE PATTERN (repair of G21-S4, whose planted control was too noisy; its outcome,
        r 0.49, is already known). Same statistic and bar (median r >= 0.25 and gain rule over the 11 EXAM chromosomes);
        planted noise SD 0.5 instead of 1.
 REPORTED: (1) THE CEILING: score between the two K562 half maps on EXAM regions, the ceiling for one full map
        sqrt(2r / (1 + r)), and each model's score as a share of it; fitted numbers; AUC for loops without clear stop
        signs; scores per cell type; curves; r values.
NOT: the 3D model is a chain of springs without volume exclusion; zones are not in the scored 3D maps unless k_zone was
chosen > 0; traces are of one 2 Mb region per cell line, on a DEV chromosome (a different kind of measurement, not used
in fitting); the EXAM regions are scored a second time (for V2); cancer or immortalised cell lines except IMR-90.

Run: python3 -I rem/atlas/foldsim2test.py fit | preview | declare | test [--trial]
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
T_B1 = T_B3 = T_D1 = T_E1 = None                 # set after the DEV fit and preview, before declaring


def _load(name):
    argv = sys.argv; sys.argv = [sys.argv[0]]
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.argv = argv; return m


ft = _load("foldsimtest")
lt, ld, fs, kt, gx, cbb, Fact, gain, spearman = ft.lt, ft.ld, ft.fs, ft.kt, ft.gx, ft.cbb, ft.Fact, ft.gain, ft.spearman
LEDGER = gx.LEDGER
OUT = R("outputs", "foldsim")
FIT2 = os.path.join(OUT, "fit2.json")
CLAIMS = ["G22-U1", "G22-U2", "G22-B1", "G22-B2", "G22-B3", "G22-D1", "G22-D2", "G22-D3", "G22-E1", "G22-E2", "G22-Z1"]
LAYERS = ft.LAYERS + ["FOLDSIM2_LOCAL", "FOLDSIM2_ENCODE"]
SEED = 20261204
DEV, EXAM, NB, SITE, WIN = ft.DEV, ft.EXAM, ft.NB, ft.SITE, ft.WIN
N_MC = 20 if TRIAL else 100
V1 = dict(w=0.0, release=0.0, e=0.0, beta=0.0)
GRID2 = [("w", (0.3, 0.6)), ("release", (0.01, 0.03)), ("e", (0.1, 0.25)), ("beta", (2.0,))]
GRID3 = dict(k_loop=(0.3, 1.0, 3.0), radius=(0.7, 1.5, 3.0), k_zone=(0.0, 0.001))
OTHER = ("gm12878", "imr90", "hepg2")


def code_sha():
    h = hashlib.sha256()
    for f in ("foldsim2test.py", "foldsimtest.py", "foldsim.py", "loopcodetest.py", "loopdyntest.py"):
        h.update(open(os.path.join(HERE, f), "rb").read())
    h.update(open(FIT2, "rb").read()); h.update(open(ft.FIT, "rb").read())
    for n in LAYERS:
        h.update(cbb.bank().facts(f"LAYER-{n}", "layer")[-1].value["sha256"].encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- data
def rank(v):
    return (np.argsort(np.argsort(v)) + 1) / max(len(v), 1)


def cell(chrom, pos, strand, signal, score, mn, mx, ac_chrom, ac_pos, ac_signal):
    ok = np.isin(chrom, DEV + EXAM); rel = (score[ok] - mn) / (mx - mn); st = np.where(strand[ok] == "+", 1, -1); sig = signal[ok].astype(float); strong = rel >= lt.REL
    q = np.zeros(len(sig)); q[strong] = rank(sig[strong]); aok = np.isin(ac_chrom, DEV + EXAM)
    return dict(chrom=chrom[ok], pos=pos[ok] + 9, strand=st, strong=strong, q=q, q_all=rank(sig), ac_chrom=ac_chrom[aok], ac_pos=ac_pos[aok], ac_q=rank(ac_signal[aok].astype(float)),
                ac_signal=ac_signal[aok].astype(float))


def load():
    D = ft.load(); A, FH = D["A"], D["FH"]; E = np.load(os.path.join(cbb.LAY, "foldsim2_encode.npz")); Lc = np.load(os.path.join(cbb.LAY, "foldsim2_local.npz")); gx.USED.update(LAYERS)
    t0 = np.array([c.split("_")[0] == "0" for c in A["ctcf_cols"]]); a0 = np.array([c.split("_")[0] == "0" for c in A["h3k27ac_cols"]]); C = {}
    C["k562"] = cell(A["ctcf_chrom"], A["motif_pos"], A["motif_strand"], np.nan_to_num(A["ctcf_counts"][:, t0]).mean(1), A["motif_score"], float(A["motif_min"]), float(A["motif_max"]),
                     A["h3k27ac_chrom"], (A["h3k27ac_start"] + A["h3k27ac_stop"]) // 2, np.nan_to_num(A["h3k27ac_counts"][:, a0]).mean(1))
    C["hct116"] = cell(FH["ctcf_chrom"], FH["ctcf_pos"], FH["ctcf_strand"], FH["ctcf_signal"], FH["ctcf_score"], float(FH["ctcf_min"]), float(FH["ctcf_max"]),
                       E["ac_hct116_chrom"], E["ac_hct116_pos"], E["ac_hct116_signal"])
    for k in OTHER:
        C[k] = cell(E[f"ctcf_{k}_chrom"], E[f"ctcf_{k}_pos"], E[f"ctcf_{k}_strand"], E[f"ctcf_{k}_signal"], E[f"ctcf_{k}_score"], float(E[f"ctcf_{k}_min"]), float(E[f"ctcf_{k}_max"]),
                    E[f"ac_{k}_chrom"], E[f"ac_{k}_pos"], E[f"ac_{k}_signal"])
    D.update(C=C, E=E, Lc=Lc, reg=list(zip(map(str, D["FK"]["chrom"]), map(int, D["FK"]["bin0"]))))
    assert D["reg"] == list(zip(map(str, FH["chrom"]), map(int, FH["bin0"]))) and all(len(E[f"maps_{k}"]) == len(D["reg"]) for k in OTHER), "region lists differ between map sets"
    return D


def simulate(D, cl, c, P, rng, shuffle=None, R_=10, snaps=100, motors=True):
    C = D["C"][cl]; m = (C["chrom"] == c) & C["strong"]; pos, st, p = [C["pos"][m]], [C["strand"][m]], [P["c"] * C["q"][m]]
    if P["w"] > 0:
        m = (C["chrom"] == c) & ~C["strong"]; pos.append(C["pos"][m]); st.append(C["strand"][m]); p.append(P["w"] * C["q_all"][m])
    am = (C["ac_chrom"] == c)
    if P["e"] > 0:
        m = am & (C["ac_q"] >= 0.5); pos.append(C["ac_pos"][m]); st.append(np.zeros(int(m.sum()), int)); p.append(P["e"] * C["ac_q"][m])
    pos, st, p = np.concatenate(pos), np.concatenate(st), np.concatenate(p); n = D["len"][c] // SITE + 1
    if shuffle == "places":
        pos = rng.integers(0, D["len"][c], len(pos))
    load = None
    if P["beta"] > 0:
        apos = C["ac_pos"][am] if shuffle != "places" else rng.integers(0, D["len"][c], int(am.sum()))
        h = np.bincount(np.minimum(apos // SITE, n - 1), weights=C["ac_signal"][am], minlength=n); load = 1.0 + P["beta"] * h / max(h.mean(), 1e-9)
    return fs.chromosome(D["len"][c], pos, st, None, P["processivity_kb"], P["separation_kb"] if motors else 1e15, P["c"], load=load, R=R_, snaps=snaps, every=5, rng=rng,
                         release=P["release"], p=p)


def regs(D, cl, chroms, strong_only=True):
    C = D["C"][cl]; out = []
    for k, (c, b0) in enumerate(D["reg"]):
        if c in chroms and int(((C["chrom"] == c) & C["strong"] & (C["pos"] >= b0 * 25000) & (C["pos"] < (b0 + NB) * 25000)).sum()) >= 15:
            out.append((k, c, b0))
    return out


def maps(D, cl, chroms, P, rng, shuffle=None, kind="1d", zone=None, R_=20, snaps=150, max_snaps=1500):
    """simulated maps {region index: map} of the regions of a cell line on `chroms` (1D rule or 3D chain model)."""
    out = {}; rg = regs(D, cl, chroms)
    for c in chroms:
        rc = [r for r in rg if r[1] == c]
        if not rc:
            continue
        L, Rr, n = simulate(D, cl, c, P, rng, shuffle=shuffle, R_=R_, snaps=snaps)
        for k, _, b0 in rc:
            if kind == "1d":
                out[k] = fs.contact_map(L, Rr, b0 * 5, b0 * 5 + NB * 5, 5, (P["alpha"],), max_snaps=max_snaps, rng=rng)[0]
            else:
                z = None if zone is None else zone[k]
                out[k] = fs.chain3d(L, Rr, b0 * 5, b0 * 5 + NB * 5, 5, P["k_loop"], P["radius"], zone=z, k_zone=P["k_zone"], max_snaps=min(max_snaps, 600), rng=rng)[0]
    return out


def scores(sim, meas):
    """{region: score} for regions with a usable measured map."""
    out = {}
    for k in sim:
        r = ft.region_score(sim[k], meas[k].astype(float))
        if r is not None:
            out[k] = r
    return out


def paired(a, b, thr=None):
    ks = sorted(set(a) & set(b)); x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks]); ok, p, med, n = gain(x - y)
    return dict(regions=len(ks), median_score=float(np.median(x)) if len(ks) else 0.0, median_comparison=float(np.median(y)) if len(ks) else 0.0, sign_p=p, median_gain=med,
                passed=bool(len(ks) and ok and (thr is None or np.median(x) >= thr)))


def planted(sim, rng):
    return {k: v * np.exp(rng.normal(0, 0.05, v.shape)) * 1000 for k, v in sim.items()}


def rotated(meas, ks):
    ks = sorted(ks); return {k: meas[ks[(i + 1) % len(ks)]] for i, k in enumerate(ks)}


def zones25(D, dev):
    zv, bc, bs = ft.zone_values(D, dev); key = {(c, int(s)): v for c, s, v in zip(bc, bs, zv)}
    return {k: np.array([key.get((c, (b0 + j) * 25000 // 100000 * 100000), 0.0) for j in range(NB)]) for k, (c, b0) in enumerate(D["reg"])}


# ---------------------------------------------------------------- loops (U2)
def loop_occ(D, chroms, P, rng, shuffle=None):
    """per chromosome: occupancy of every called loop, of its two shifted controls, and which loops have no clear stop sign."""
    L, S = D["L"], D["S"]; out = {}; LS = {k: (len(le), len(ri)) for k, le, ri in lt.loop_sites(S, L, chroms)}
    win = lambda v0, v1, sh: ((v0 - WIN) // SITE + sh, (v1 + WIN) // SITE + sh)
    for c in chroms:
        tab = fs.loop_table(*simulate(D, "k562", c, P, rng, shuffle=shuffle, R_=20, snaps=150)); k = np.nonzero(L["chrom"] == c)[0]
        occ = lambda sh: fs.occupancy(tab, *win(L["a0"][k], L["a1"][k], sh), *win(L["b0"][k], L["b1"][k], sh))
        out[c] = (occ(0), occ(-20), occ(20), np.array([LS.get(i, (0, 0)) == (0, 0) for i in k]))
    return out


def auc_of(o, cl, cr):
    w = lambda x, y: (x > y) + 0.5 * (x == y); return (w(o, cl) + w(o, cr)) / 2


def loop_auc(occ):
    return {c: float(auc_of(o, cl, cr).mean()) for c, (o, cl, cr, _) in occ.items()}


def bare_auc(occ):
    a = np.concatenate([auc_of(o, cl, cr)[bare] for o, cl, cr, bare in occ.values()]); return float(a.mean()) if len(a) else 0.0, int(len(a))


# ---------------------------------------------------------------- compaction curve (B3)
def curve(maps_, ks, raw=True):
    out = []
    for k in ks:
        M = np.asarray(maps_[k], float); tot = M.sum() if raw else 1.0
        out.append(np.array([np.diag(M, s).mean() for s in range(2, 101)]) / max(tot, 1e-12))
    return np.mean(out, 0)


# ---------------------------------------------------------------- microscope traces (D2, D3, E2)
def trace_model(D, cl, start, nseg, P, rng, shuffle=None, motors=True, n_s=2000):
    L, Rr, n = simulate(D, cl, "21", P, rng, shuffle=shuffle, R_=20, snaps=100, motors=motors); lo = start // SITE
    X = fs.chain3d(L, Rr, lo, lo + nseg * 6, 6, P["k_loop"], P["radius"], max_snaps=n_s, rng=rng, sample=True)[2]
    M = np.zeros((nseg, nseg))
    for i in range(nseg):
        M[i] = np.median(np.sqrt(((X[:, i:i + 1] - X) ** 2).sum(2)), 0)
    return M, fs.border_calls(X)[0]


def dist_r(Ms, Mm):
    a, b = [], []
    for k in range(2, len(Ms) - 1):
        x, y = np.diag(Ms, k), np.diag(Mm, k); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2:
            a.append(x[ok] / x[ok].mean()); b.append(y[ok] / y[ok].mean())
    return spearman(np.concatenate(a), np.concatenate(b))


def border_r(bs, bm):
    ok = np.isfinite(bs) & np.isfinite(bm); return spearman(bs[ok], bm[ok])


def traces(D, P, rng):
    Lc = D["Lc"]; out = {}
    Mk, Bk = Lc["trace_k562_28_median_nm"], Lc["trace_k562_28_border_share"]; n = len(Mk); st = int(Lc["trace_k562_28_start"])
    Ms, Bs = trace_model(D, "k562", st, n, P, rng); nullM, nullB, nullRev = [], [], []
    for _ in range(N_MC):
        m_, b_ = trace_model(D, "k562", st, n, P, rng, shuffle="places", n_s=1000); nullM.append(dist_r(m_, Mk)); nullB.append(border_r(b_, Bk))
    M2, B2 = trace_model(D, "k562", st, n, P, rng)
    pmc = lambda obs, null: float((1 + np.sum(np.array(null) >= obs)) / (len(null) + 1))

    def claim(obs, null, rev, pl):
        res = dict(r=float(obs), p_mc=pmc(obs, null), null_95=float(np.quantile(null, 0.95)), passed=bool(obs >= 0.01 and pmc(obs, null) < 0.05))
        res["controls"] = dict(reversed_p_mc=pmc(rev, null), planted_p_mc=pmc(pl, null), ok=bool(pmc(rev, null) >= 0.05 and pmc(pl, null) < 0.05)); return res
    scale = float(np.nanmedian(np.diag(Mk, 1)) / np.median(np.diag(Ms, 1)))
    out["G22-D2"] = claim(dist_r(Ms, Mk), nullM, dist_r(Ms, Mk[::-1, ::-1]), dist_r(Ms, M2)); out["G22-D2"].update(segments=n, nm_per_unit=scale)
    out["G22-E2"] = claim(border_r(Bs, Bk), nullB, border_r(Bs, Bk[::-1]), border_r(Bs, B2))
    out["G22-E2"].update(measured_border_share_range=[float(np.nanmin(Bk)), float(np.nanmax(Bk))], model_border_share_range=[float(np.nanmin(Bs)), float(np.nanmax(Bs))])
    Mu, Ma = Lc["trace_hct116_34_untreated_median_nm"], Lc["trace_hct116_34_auxin_median_nm"]; n = len(Mu); st = int(Lc["trace_hct116_34_untreated_start"])
    Mh, Bh = trace_model(D, "hct116", st, n, P, rng); dobs = dist_r(Mh, Mu) - dist_r(Mh, Ma); null = []
    for _ in range(N_MC):
        m_, _b = trace_model(D, "hct116", st, n, P, rng, shuffle="places", n_s=1000); null.append(dist_r(m_, Mu) - dist_r(m_, Ma))
    Mh2, _ = trace_model(D, "hct116", st, n, P, rng); M0, _ = trace_model(D, "hct116", st, n, P, rng, motors=False)
    res = dict(r_untreated=float(dist_r(Mh, Mu)), r_auxin=float(dist_r(Mh, Ma)), difference=float(dobs), p_mc=pmc(dobs, null), segments=n, passed=bool(dobs >= 0.01 and pmc(dobs, null) < 0.05))
    rev = dist_r(Mh, Mu[::-1, ::-1]) - dist_r(Mh, Ma[::-1, ::-1]); pl = dist_r(Mh, Mh2) - dist_r(Mh, M0)
    res["controls"] = dict(reversed_p_mc=pmc(rev, null), planted_p_mc=pmc(pl, null), ok=bool(pmc(rev, null) >= 0.05 and pmc(pl, null) < 0.05))
    res["border_share_r"] = dict(untreated=border_r(Bh, Lc["trace_hct116_34_untreated_border_share"]), auxin=border_r(Bh, Lc["trace_hct116_34_auxin_border_share"]))
    out["G22-D3"] = res
    return out, dict(k562_model_nm=Ms * scale, k562_model_border=Bs, hct116_model_nm=Mh * float(np.nanmedian(np.diag(Mu, 1)) / np.median(np.diag(Mh, 1))), hct116_model_border=Bh)


# ---------------------------------------------------------------- zone repair (Z1)
def z1(D, dev, exam, rng):
    zv, bc, bs = ft.zone_values(D, dev); H = np.load(os.path.join(kt.HERE, "_cache", "k562pma", "hic", "raw", "intra100k_0.npz")); gx.USED.add("K562PMA_RAW_HIC")

    def run(zv, plant=False):
        rs = []
        for c in exam:
            ix = np.nonzero(bc == c)[0]; n = int(H[f"{c}_j"].max()) + 1; M = np.zeros((n, n)); np.add.at(M, (H[f"{c}_i"], H[f"{c}_j"]), H[f"{c}_v"]); M = M + M.T - np.diag(np.diag(M))
            Mb, valid = fs.balance(M, iters=100); z = np.zeros(n); m_ = min(n, len(ix)); z[:m_] = zv[ix][:m_]
            i = rng.integers(0, n, 400_000); j = i + rng.integers(20, 500, 400_000); ok = j < n; i, j = i[ok], j[ok]; ok = valid[i] & valid[j]; i, j = i[ok], j[ok]; d = j - i
            obs = np.exp(0.5 * z[i] * z[j] + rng.normal(0, 0.5, len(i))) * (d.astype(float) ** -1.0) if plant else Mb[i, j]
            mu = np.bincount(d, weights=obs) / np.maximum(np.bincount(d), 1); rs.append(spearman(z[i] * z[j], obs / np.maximum(mu[d], 1e-12)))
        ok_, p, med, n_ = gain(np.array(rs)); return dict(median_r=float(np.median(rs)), sign_p=p, chromosomes=n_, passed=bool(np.median(rs) >= 0.25 and ok_))
    res = run(zv); sh = zv.copy()
    for c in np.unique(bc):
        ix = np.nonzero(bc == c)[0]; sh[ix] = np.roll(zv[ix], 100)
    res["controls"] = dict(negative_passed=run(sh)["passed"], planted_passed=run(zv, plant=True)["passed"]); res["controls"]["ok"] = bool(not res["controls"]["negative_passed"] and res["controls"]["planted_passed"])
    return res


# ---------------------------------------------------------------- fit (DEV only)
def objective(D, P, rng, chroms):
    sim = maps(D, "k562", chroms, P, rng, R_=10, snaps=100, max_snaps=300); sc = scores(sim, D["FK"]["maps"]); auc = loop_auc(loop_occ(D, chroms, P, rng))
    return float(np.median(list(sc.values()))), float(np.median(list(auc.values())))


def fit():
    t0 = time.time(); D = load(); rng = np.random.default_rng(SEED + 1); base = dict(json.load(open(ft.FIT))["params"]); P = dict(base, **V1); log = []
    best = objective(D, P, rng, DEV); log.append(dict(P, map=best[0], auc=best[1])); print("fit v1", best, f"({time.time() - t0:.0f}s)", flush=True)
    for name, vals in GRID2:
        for v in vals:
            Q = dict(P, **{name: v}); o = objective(D, Q, rng, DEV); log.append(dict(Q, map=o[0], auc=o[1])); print("fit", name, v, o, f"({time.time() - t0:.0f}s)", flush=True)
            if sum(o) > sum(best) + 0.005:
                best, P = o, Q
    zone = zones25(D, DEV); FK = D["FK"]["maps"]; rows3 = []; rg = regs(D, "k562", DEV); snaps = {}
    for c in DEV:
        if any(r[1] == c for r in rg):
            snaps[c] = simulate(D, "k562", c, P, rng, R_=20, snaps=150)
    for kl in GRID3["k_loop"]:
        for rad in GRID3["radius"]:
            for kz in GRID3["k_zone"]:
                sim = {k: fs.chain3d(snaps[c][0], snaps[c][1], b0 * 5, b0 * 5 + NB * 5, 5, kl, rad, zone=zone[k], k_zone=kz, max_snaps=300, rng=rng)[0] for k, c, b0 in rg}
                sc = scores(sim, FK); rows3.append(dict(k_loop=kl, radius=rad, k_zone=kz, map=float(np.median(list(sc.values())))))
        print("fit3d", kl, [round(r["map"], 3) for r in rows3[-len(GRID3["radius"]) * len(GRID3["k_zone"]):]], f"({time.time() - t0:.0f}s)", flush=True)
    b3 = max(rows3, key=lambda r: r["map"]); P.update(k_loop=b3["k_loop"], radius=b3["radius"], k_zone=b3["k_zone"])
    json.dump(dict(params=P, log=log, grid3d=rows3, fitted_on="DEV (odd autosomes), K562", seed=SEED + 1), open(FIT2, "w"), indent=1); print("chosen", P)


# ---------------------------------------------------------------- evaluate
def ctl(res, neg, pl):
    res["controls"] = dict(negative_passed=bool(neg), planted_passed=bool(pl), ok=bool(not neg and pl)); return res


def evaluate(dev, exam, rng):
    D = load(); P = json.load(open(FIT2))["params"]; P1 = dict(P, **V1); out, rep = {}, dict(params=P); FK, FH, E, Lc = D["FK"]["maps"], D["FH"], D["E"], D["Lc"]
    # ---- K562 maps: V1, V2, V2 with sites at random places
    m1 = maps(D, "k562", exam, P1, rng); m2 = maps(D, "k562", exam, P, rng); s1, s2 = scores(m1, FK), scores(m2, FK); s2r = scores(maps(D, "k562", exam, P, rng, shuffle="places"), FK)
    u1 = paired(s2, s1); u1.update(median_v1=u1.pop("median_comparison"), median_v2=u1.pop("median_score"), median_random_places=float(np.median(list(s2r.values()))))
    pl = planted(m2, rng); out["G22-U1"] = ctl(u1, paired(s2r, s1)["passed"], paired(scores(m2, pl), scores(m1, pl))["passed"])
    hA, hB = Lc["maps_k562_halfA"], Lc["maps_k562_halfB"]; half = [ft.region_score(hA[k].astype(float), hB[k].astype(float)) for k in s2]
    rAB = float(np.median([v for v in half if v is not None])); ceil = float(np.sqrt(2 * rAB / (1 + rAB)))
    # ---- loops
    o1, o2, orr = loop_occ(D, exam, P1, rng), loop_occ(D, exam, P, rng), loop_occ(D, exam, P, rng, shuffle="places"); a1, a2, ar = loop_auc(o1), loop_auc(o2), loop_auc(orr)
    ok, p, med, n = gain(np.array([a2[c] - a1[c] for c in exam])); anyocc = lambda oc: float(np.mean(np.concatenate([o > 0 for o, _, _, _ in oc.values()])))
    u2 = dict(auc_v1=float(np.median(list(a1.values()))), auc_v2=float(np.median(list(a2.values()))), auc_random_places=float(np.median(list(ar.values()))), sign_p=p, median_gain=med,
              auc_loops_without_clear_signs=dict(v1=bare_auc(o1)[0], v2=bare_auc(o2)[0], loops=bare_auc(o2)[1]), share_of_loops_with_any_occupancy=dict(v1=anyocc(o1), v2=anyocc(o2)), passed=bool(ok))
    opl = {c: (o + 0.005 * (rng.random(len(o)) < 0.2), cl, cr, b) for c, (o, cl, cr, b) in o1.items()}; apl = loop_auc(opl)
    out["G22-U2"] = ctl(u2, gain(np.array([ar[c] - a1[c] for c in exam]))[0], gain(np.array([apl[c] - a1[c] for c in exam]))[0])
    # ---- 3D maps (K562)
    zone = zones25(D, dev) if P["k_zone"] > 0 else None
    m3 = maps(D, "k562", exam, P, rng, kind="3d", zone=zone); s3 = scores(m3, FK); s3r = scores(maps(D, "k562", exam, P, rng, shuffle="places", kind="3d", zone=zone), FK)
    d1 = paired(s3, s3r, T_D1); d1["median_random_places"] = d1.pop("median_comparison"); d1["median_1d_v2"] = u1["median_v2"]
    out["G22-D1"] = ctl(d1, paired(scores(m3, rotated(FK, list(s3))), s3r, T_D1)["passed"], paired(scores(m3, planted(m3, rng)), s3r, T_D1)["passed"])
    rep["ceiling"] = dict(half_vs_half=rAB, ceiling_full_map=ceil, v1_share=u1["median_v1"] / ceil, v2_share=u1["median_v2"] / ceil, model3d_share=d1["median_score"] / ceil)
    # ---- HCT116: break one part
    mh = maps(D, "hct116", exam, P, rng); mhr = maps(D, "hct116", exam, P, rng, shuffle="places"); names = ("ctcf_untreated", "ctcf_auxin", "pol2_untreated", "pol2_auxin")
    sc = {k: scores(mh, E[f"maps_hct116_{k}"]) for k in names}; scr = {k: scores(mhr, E[f"maps_hct116_{k}"]) for k in names[:2]}
    b1 = paired(sc["ctcf_untreated"], sc["ctcf_auxin"], T_B1); b1["median_after_ctcf_removed"] = b1.pop("median_comparison")
    out["G22-B1"] = ctl(b1, paired(scr["ctcf_untreated"], scr["ctcf_auxin"], T_B1)["passed"], paired(scores(mh, planted(mh, rng)), sc["ctcf_auxin"], T_B1)["passed"])
    ks = sorted(set.intersection(*(set(v) for v in sc.values()))); drop = lambda a, b: np.array([sc[a][k] - sc[b][k] for k in ks])
    dc, dp = drop("ctcf_untreated", "ctcf_auxin"), drop("pol2_untreated", "pol2_auxin"); ok, p, med, n = gain(dc - dp)
    b2 = dict(regions=len(ks), median_drop_ctcf=float(np.median(dc)), median_drop_pol2=float(np.median(dp)), median_score_pol2_untreated=float(np.median([sc["pol2_untreated"][k] for k in ks])),
              median_score_pol2_removed=float(np.median([sc["pol2_auxin"][k] for k in ks])), sign_p=p, median_gain=med, passed=bool(ok))
    out["G22-B2"] = ctl(b2, gain(dp - dc)[0], gain(dc)[0])
    su, sa = scores(mh, FH["maps_untreated"]), scores(mh, FH["maps_auxin"]); rep["v2_hct116_cohesin_line"] = dict(untreated=float(np.median(list(su.values()))), no_cohesin=float(np.median(list(sa.values()))))
    # ---- contact with distance (3D model)
    m3h = maps(D, "hct116", exam, P, rng, kind="3d"); m0 = maps(D, "hct116", exam, dict(P, separation_kb=1e15), rng, kind="3d", R_=2, snaps=5)
    m3n = maps(D, "hct116", exam, dict(P, c=0.0, w=0.0), rng, kind="3d"); ks = sorted(set(m3h) & set(sc["ctcf_untreated"]) & set(su)); cen = lambda v: v - v.mean()
    pc, pn = cen(np.log(curve(m3h, ks, raw=False) / curve(m0, ks, raw=False))), cen(np.log(curve(m3h, ks, raw=False) / curve(m3n, ks, raw=False)))
    mc, mn = cen(np.log(curve(FH["maps_untreated"], ks) / curve(FH["maps_auxin"], ks))), cen(np.log(curve(E["maps_hct116_ctcf_untreated"], ks) / curve(E["maps_hct116_ctcf_auxin"], ks)))
    b3f = lambda pc_, mc_, mn_: bool(np.corrcoef(pc_, mc_)[0, 1] >= T_B3 and mn_.std() / mc_.std() < 0.5 and pn.std() / pc.std() < 0.5)
    b3 = dict(r=float(np.corrcoef(pc, mc)[0, 1]), sd_ratio_measured=float(mn.std() / mc.std()), sd_ratio_predicted=float(pn.std() / pc.std()), amplitude_predicted_over_measured=float(pc.std() / mc.std()),
              regions=len(ks), passed=b3f(pc, mc, mn), curve_kb=[int(x * 25) for x in range(2, 101, 7)], predicted_cohesin=[float(x) for x in pc[::7]], predicted_ctcf=[float(x) for x in pn[::7]],
              measured_cohesin=[float(x) for x in mc[::7]], measured_ctcf=[float(x) for x in mn[::7]])
    nz = lambda v: v + rng.normal(0, pc.std() / 5, len(v)); out["G22-B3"] = ctl(b3, b3f(pc[::-1], mc, mn), b3f(pc, nz(pc), nz(pn)))
    # ---- more cell types
    e1 = dict(cells={}); allok, cok = True, True; other_maps = {}
    for k in OTHER:
        mk = maps(D, k, exam, P, rng); Mk = E[f"maps_{k}"]; sk = scores(mk, Mk); skr = scores(maps(D, k, exam, P, rng, shuffle="places"), Mk); r = paired(sk, skr, T_E1); r["median_random_places"] = r.pop("median_comparison")
        neg = paired(scores(mk, rotated(Mk, list(sk))), skr, T_E1)["passed"]; pl = paired(scores(mk, planted(mk, rng)), skr, T_E1)["passed"]
        e1["cells"][k] = dict(r, negative_passed=bool(neg), planted_passed=bool(pl)); allok &= r["passed"]; cok &= (not neg) and pl; other_maps[k] = (mk, sk)
    e1["passed"] = bool(allok); e1["controls"] = dict(ok=bool(cok)); out["G22-E1"] = e1
    tr, tx = traces(D, P, rng); out.update(tr); out["G22-Z1"] = z1(D, dev, exam, rng)
    for cid in CLAIMS:
        ok = out[cid]["controls"]["ok"]; out[cid]["gate"] = dict(real=out[cid]["passed"], controls_ok=ok, verdict=("PASS" if out[cid]["passed"] else "FAIL") if ok else "VOID (controls)")
    return out, rep, dict(D=D, P=P, mh=mh, m3h=m3h, sc=sc, other=other_maps, traces=tx)


def preview():
    global T_B1, T_B3, T_D1, T_E1
    T_B1 = T_B3 = T_D1 = T_E1 = -1.0
    out, rep, _ = evaluate(DEV[:6], DEV[6:], np.random.default_rng(SEED + 2))
    for cid in CLAIMS:
        print(cid, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cid].items() if k not in ("gate", "curve_kb", "predicted", "measured_cohesin", "measured_ctcf")}, default=float)[:520])
    print(json.dumps(rep, default=float)[:600])


def save_page(x):
    """what the page shows: example regions (best and typical) per map set, simulated next to measured (ENCODE maps: free
    use), and the microscope summaries (median distance per pair of segments; Bintu et al. 2018) next to the model's."""
    D, E, Lc = x["D"], x["D"]["E"], x["D"]["Lc"]; Z = {}; pick = lambda sc: [max(sc, key=sc.get), sorted(sc, key=sc.get)[len(sc) // 2]]
    f32 = lambda M: np.asarray(M, np.float32)
    for name, sim, sc, meas in [("hct116", x["mh"], x["sc"]["ctcf_untreated"], ("hct116_ctcf_untreated", "hct116_ctcf_auxin"))] + [(k, x["other"][k][0], x["other"][k][1], (k,)) for k in OTHER]:
        ks = pick(sc); Z[f"{name}_keys"] = np.array(ks); Z[f"{name}_score"] = np.array([sc[k] for k in ks]); Z[f"{name}_sim"] = f32([sim[k] for k in ks])
        Z[f"{name}_chrom"] = np.array([D["reg"][k][0] for k in ks]); Z[f"{name}_bin0"] = np.array([D["reg"][k][1] for k in ks])
        for m in meas:
            Mm = E[f"maps_{m}"]; Z[f"{m}_measured"] = f32([Mm[k] for k in ks])
        if name == "hct116":
            Z["hct116_sim3d"] = f32([x["m3h"][k] for k in ks])
    for k, v in x["traces"].items():
        Z["trace_" + k] = f32(v)
    for t in ("k562_28", "hct116_34_untreated", "hct116_34_auxin"):
        Z[f"trace_{t}_measured_nm"] = f32(Lc[f"trace_{t}_median_nm"]); Z[f"trace_{t}_measured_border"] = f32(Lc[f"trace_{t}_border_share"])
    np.savez_compressed(os.path.join(OUT, "foldsim2_page.npz"), **Z)


def _read():
    return [json.loads(x) for x in open(LEDGER)] if os.path.exists(LEDGER) else []


def claim_text(cid):
    doc = __doc__.format(T_B1=T_B1, T_B3=T_B3, T_D1=T_D1, T_E1=T_E1, N_MC=100)
    return doc.split(" " + cid)[1].split(" G22-")[0].split(" REPORTED:")[0].strip()[:1500]


def declare():
    assert None not in (T_B1, T_B3, T_D1, T_E1), "set the thresholds first"
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
            raise SystemExit("G22 already tested")
    t0 = time.time(); out, rep, extra = evaluate(*((DEV[:6], DEV[6:]) if TRIAL else (DEV, EXAM)), np.random.default_rng(SEED))
    if TRIAL:
        for cid in CLAIMS:
            print(cid, "controls", "ok" if out[cid]["gate"]["controls_ok"] else "NOT ok", json.dumps(out[cid]["controls"]))
        print("trial ran on DEV chromosomes; outcomes not shown", f"({time.time() - t0:.0f}s)"); return
    json.dump(dict(out, reported=rep), open(os.path.join(OUT, "foldsim2_tests.json"), "w"), indent=1, default=float); save_page(extra)
    t = cbb.now(); src = {"name": "foldsim2test:G22", "code_sha": code_sha()}; inp = tuple(sorted(gx.USED))
    cbb.bank().add_many([Fact(f"CLAIM-{cid}", "foldsim2_test", out[cid], t[:10], t, "derived", src, inputs=inp) for cid in CLAIMS])
    for cid in CLAIMS:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(op="result", id=cid, at=cbb.now(), code_sha=code_sha(), inputs=sorted(gx.USED), **out[cid]), sort_keys=True, default=float) + "\n")
        print(cid, out[cid]["gate"]["verdict"], flush=True)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--trial"]
    {"fit": fit, "preview": preview, "declare": declare, "test": test}[a[0]]()
