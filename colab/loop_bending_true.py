"""LOOP 83 -- THE TERM I CALLED PERSISTENCE WAS NOT PERSISTENCE. HEAD-TO-HEAD WITH THE REAL ONE.

WHAT LOOP 82 ACTUALLY ADDED. I added second-neighbour SPRINGS, (i, i+2) at weight kappa, and called
them a persistence length. They are not. A spring PULLS i and i+2 together, which folds the chain back
on itself; stiffness requires an ANGULAR penalty on |r_{i+1} - 2 r_i + r_{i-1}|^2, which is a different
operator. Measured, by the textbook definition -- the bond-vector correlation <b_i . b_j>:

    loop 82's spring, kappa = 4     -0.610  +0.372  -0.227  -0.084  +0.019     alternating: ZIGZAG
    true bending,     kappa = 8     +0.703  +0.495  +0.348  +0.172  +0.060     decaying: STIFF

A stiff chain has POSITIVE, monotonically decaying bond correlations -- consecutive bonds pointing the
same way. Loop 82's term gives strongly NEGATIVE, sign-alternating correlations: consecutive bonds
point in opposite directions. It added local COMPACTION, not rigidity, and loop 82's D3 gate -- which
was supposed to catch exactly this by reporting an implied persistence length -- was broken and
returned 25.0 kb (one bin) for every kappa including zero.

SO LOOP 82's EMPIRICAL RESULT IS REAL AND ITS EXPLANATION IS WRONG. It produced the best map
correlation in this arc (+0.8588, 26.5% of headroom), matched both contact-decay bands for the first
time, and transferred to chr22 (+0.8779). None of that is in doubt. What is in doubt is whether it
generalises, because a result whose mechanism is mislabelled cannot be reasoned about -- and local
compaction and rigidity make different predictions everywhere else.

THE TRUE OPERATOR, and now the literature check is meaningful. E = kappa * sum_i |r_{i+1} - 2r_i +
r_{i-1}|^2 = r^T (kappa D2^T D2) r, positive semi-definite by construction. Its measured persistence
lengths: kappa 2 -> 36 kb, kappa 8 -> 71 kb, kappa 32 -> 142 kb, i.e. 1.4 to 5.7 bins. Effective
coarse-grained chromatin persistence at this resolution is usually quoted at a few bins, so the low
end is defensible and the high end is not.

PREDECLARED, before any number:

  P1 THE TWO OPERATORS ARE DISTINGUISHED BY MEASUREMENT, NOT BY ASSERTION
       bond-vector correlation for both terms across kappa. Gate: the true bending operator must give
       POSITIVE correlations at s = 1..3 and the loop 82 spring must give a NEGATIVE one at s = 1.
       If they are not distinguishable this way the whole premise is wrong.
  P2 THE IMPLIED PERSISTENCE LENGTH IS PHYSICAL AND IS NOW MEASURABLE
       lp in kb per kappa, from the exponential decay of the bond correlation -- the measurement loop
       82's D3 failed to make. Reported with the literature range, not gated.
  P3 EVERY BASE LAPLACIAN IS POSITIVE DEFINITE AND THE MAP IS STILL AN IDENTITY
       as loops 80-82. Gate: max relative error <= 1e-6 against a fresh full inversion.
  P4 DOES TRUE BENDING REPRODUCE LOOP 82's RESULT                    THE GATE.
       same grid, same compartment sweep, same everything except the operator. Gate: the best
       true-bending point must reach a map correlation within 0.01 of loop 82's +0.8588 AND match
       both bands. If it does, loop 82's finding survives with a corrected mechanism. If it does not,
       loop 82's success came specifically from local compaction, and that must be said plainly
       rather than left as a mislabelled win.
  P5 HEAD-TO-HEAD ON EVERY OBSERVABLE
       both operators, best point each: bands, map correlation, checkerboard, orientation signature.
       Whichever wins, the comparison is the deliverable.
  P6 HELD OUT: CHROMOSOME 22
       the winning operator applied unchanged, chr22's own GC track and compartment mass.

-> outputs/loop_bending_true.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_second as L77  # noqa: E402
import loop_map_score as L79  # noqa: E402
import loop_bending as L80  # noqa: E402
import loop_compartment_attract as L81  # noqa: E402
import loop_persistence as L82  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BIN = L77.BIN
K_LOOP = L80.K_DERIVED
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)
BAND_TOL = 0.12
DIST_NULL, CEILING = 0.8280, 0.9441
L82_RHO = 0.8588                      # loop 82's best, the number to reproduce
MATCH_TOL = 0.01                      # P4: how close true bending must come

KAPPA_TRUE = [0.0, 0.5, 2.0, 8.0]     # true bending; lp 0, 19, 36, 71 kb (measured below)
KAPPA_SPRING = [0.0, 1.0, 4.0, 16.0]  # loop 82's second-neighbour spring, for the head-to-head
ALPHA_SWEEP = [0.0, 1e-4, 3e-4, 1e-3]
SEPARATION_KB = [200.0, 400.0]
RESIDENCE_S = [600.0, 1500.0]
SPEED_KB_S = [0.75]
DT_SWEEP, DT_FINAL = 3.0, 1.0
NCFG_SWEEP, NCFG_FINAL = 12, 50
P3_TOL = 1e-6
SEED = 8301

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def bending_op(n, kappa):
    """TRUE bending: kappa * D2^T D2 with D2 the second-difference operator. PSD by construction."""
    if kappa <= 0:
        return np.zeros((n, n))
    D2 = np.zeros((n - 2, n))
    i = np.arange(n - 2)
    D2[i, i] = 1.0
    D2[i, i + 1] = -2.0
    D2[i, i + 2] = 1.0
    return kappa * (D2.T @ D2)


def spring_op(n, kappa):
    """Loop 82's term: a second-neighbour SPRING. Retained only for the head-to-head."""
    L = np.zeros((n, n))
    if kappa > 0:
        i = np.arange(n - 2)
        L[i, i + 2] -= kappa
        L[i + 2, i] -= kappa
        L[i, i] += kappa
        L[i + 2, i + 2] += kappa
    return L


def base_laplacian(n, kappa, c, eps, mode, confine=L77.CONFINE):
    from loop_polymer import laplacian
    L = laplacian(n, loops=[], confine=confine)
    L = L + (bending_op(n, kappa) if mode == "bend" else spring_op(n, kappa))
    if eps > 0:
        p = np.maximum(c, 0.0)
        m = np.maximum(-c, 0.0)
        L = L + 2.0 * eps * (np.diag(p * p.sum() + m * m.sum()) - np.outer(p, p) - np.outer(m, m))
    return L


def bond_corr(n, kappa, mode, smax=10):
    """<b_i . b_j> normalised, from the covariance. The textbook persistence measurement."""
    from loop_polymer import laplacian
    m = min(n, 300)
    L = laplacian(m, loops=[], confine=0.0)
    L = L + (bending_op(m, kappa) if mode == "bend" else spring_op(m, kappa))
    G = np.linalg.pinv(L + 1e-10 * np.eye(m))
    lo, hi = m // 4, 3 * m // 4
    out = []
    for s in range(smax):
        v = [G[i + 1, i + s + 1] - G[i + 1, i + s] - G[i, i + s + 1] + G[i, i + s]
             for i in range(lo, hi)]
        out.append(float(np.mean(v)))
    if abs(out[0]) < 1e-12:
        return [float("nan")] * smax, float("nan")
    nb = [x / out[0] for x in out]
    ss = [s for s in range(1, smax) if nb[s] > 1e-4]
    lp = float(-1.0 / np.polyfit(ss, np.log([nb[s] for s in ss]), 1)[0]) if len(ss) >= 2 else float("nan")
    return nb, lp


def contact_map_exact(n, configs, kappa, c, eps, mode, k, confine=L77.CONFINE):
    from loop_polymer import r2_matrix
    acc = np.zeros((n, n))
    for cfg in configs:
        L = base_laplacian(n, kappa, c, eps, mode, confine)
        for a, b in cfg:
            a, b = int(a), int(b)
            if a == b:
                continue
            L[a, b] -= k
            L[b, a] -= k
            L[a, a] += k
            L[b, b] += k
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return acc / max(len(configs), 1)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 83 -- the term I called persistence was not persistence. Head-to-head with the real one.")
    say("=" * 100)
    say()

    C21 = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    n, mask, H = C21["n"], C21["mask"], C21["H"]
    w = int(L77.BAND_BP // BIN)
    bf, br = L79.landscape(C21, C21["orients"])
    fs, rs = L79.sites(C21, C21["orients"])
    c = L81.comp_score(L81.gc_track(L77.FASTA, n), mask)
    cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))
    rho_dist = L77.band_rho(L79.distance_null(C21), H, mask, n, w)[0]
    meas_s = L80.ps_band(H, mask, *SHORT_BAND)
    meas_l = L80.ps_band(H, mask, *LONG_BAND)
    say(f"  TARGET short {meas_s:+.4f}  long {meas_l:+.4f};  null {rho_dist:+.4f}; "
        f"loop 82 best {L82_RHO:+.4f}; ceiling {CEILING:+.4f}")
    say()

    say("P1 THE TWO OPERATORS ARE DISTINGUISHED BY MEASUREMENT")
    lps = {}
    ok_bend = ok_spring = True
    for mode, sweep in (("bend", KAPPA_TRUE), ("spring", KAPPA_SPRING)):
        for kap in sweep:
            if kap == 0:
                continue
            nb, lp = bond_corr(n, kap, mode)
            lps[(mode, kap)] = lp
            say(f"     {mode:6s} kappa {kap:5.1f}  <b.b> " +
                " ".join(f"{nb[s]:+.3f}" for s in (1, 2, 3, 5)) +
                f"   lp {lp:5.2f} bins = {lp*BIN/1e3:6.0f} kb")
            if mode == "bend" and not all(nb[s] > 0 for s in (1, 2, 3)):
                ok_bend = False
            if mode == "spring" and nb[1] >= 0:
                ok_spring = False
    p1 = ok_bend and ok_spring
    say(f"     true bending gives POSITIVE decaying correlations: {ok_bend}")
    say(f"     the loop 82 spring gives a NEGATIVE s=1 correlation (zigzag): {ok_spring}")
    say(f"     P1 {'PASS' if p1 else 'FAIL'}")
    say()

    say("P2 THE IMPLIED PERSISTENCE LENGTH, THE MEASUREMENT LOOP 82's D3 FAILED TO MAKE")
    for kap in KAPPA_TRUE[1:]:
        say(f"     true bending kappa {kap:5.1f}  ->  lp {lps[('bend',kap)]*BIN/1e3:6.0f} kb "
            f"({lps[('bend',kap)]:.2f} bins)")
    say(f"     effective coarse-grained chromatin persistence at 25 kb bins is usually quoted at a")
    say(f"     few bins, so the low end here is defensible and the high end is not")
    say(f"     [loop 82's D3 returned 25.0 kb -- exactly one bin -- for EVERY kappa including zero,")
    say(f"      because its crossover detector found the slope already below 1.5 at s=1 and returned")
    say(f"      the first bin unconditionally. It was labelled 'reported, not judged' so it could")
    say(f"      not fail, and it printed a constant carrying no information.]")
    say(f"     P2 PASS (reported)")
    say()

    say("P3 POSITIVE DEFINITE, AND THE MAP IS STILL AN IDENTITY")
    G0 = {}
    for mode, sweep in (("bend", KAPPA_TRUE), ("spring", KAPPA_SPRING)):
        for kap in sweep:
            for al in ALPHA_SWEEP:
                L = base_laplacian(n, kap, c, al / cmass, mode)
                lam = float(np.linalg.eigvalsh(L).min())
                assert lam > 0, f"indefinite: {mode} kappa={kap} alpha={al} (min eig {lam})"
                G0[(mode, kap, al)] = np.linalg.inv(L)
    say(f"     {len(G0)} base Laplacians built, all positive definite")
    cf, _, _ = L77.simulate(n, bf, br, np.random.default_rng(SEED), DT_FINAL, n_config=3)
    worst = 0.0
    for mode, kap, al in (("bend", 2.0, 3e-4), ("bend", 8.0, 1e-3), ("spring", 4.0, 1e-3)):
        A = L80.contact_map_k(n, cf, G0[(mode, kap, al)], K_LOOP)
        B = contact_map_exact(n, cf, kap, c, al / cmass, mode, K_LOOP)
        f = np.isfinite(A) & np.isfinite(B) & (B > 0)
        e = float(np.max(np.abs(A[f] - B[f]) / np.abs(B[f])))
        worst = max(worst, e)
        say(f"     {mode:6s} kappa {kap:4.1f} alpha {al:.0e}   max relative difference {e:.3e}")
    p3 = worst <= P3_TOL
    say(f"     P3 {'PASS' if p3 else 'FAIL'}  (gate {P3_TOL:.0e})")
    say()

    say("P4/P5 DOES TRUE BENDING REPRODUCE LOOP 82's RESULT")
    rng = np.random.default_rng(SEED)
    sh = list(rng.permutation(C21["orients"]))
    bfs, brs = L79.landscape(C21, sh)
    fss, rss = L79.sites(C21, sh)
    results = {}
    for mode, sweep in (("bend", KAPPA_TRUE), ("spring", KAPPA_SPRING)):
        rows = []
        for sep in SEPARATION_KB:
            for res in RESIDENCE_S:
                for spd in SPEED_KB_S:
                    for kap in sweep:
                        for al in ALPHA_SWEEP:
                            R = L82.run_point(C21, bf, br, sep, res, spd,
                                              G0[(mode, kap, al)], DT_SWEEP, NCFG_SWEEP, SEED)
                            bs = L80.ps_band(R["M"], mask, *SHORT_BAND)
                            bl = L80.ps_band(R["M"], mask, *LONG_BAND)
                            rho = L77.band_rho(R["M"], H, mask, n, w)[0]
                            rows.append({"sep_kb": sep, "res_s": res, "v_kb_s": spd, "kappa": kap,
                                         "alpha": al, "band_short": bs, "band_long": bl,
                                         "rho_map": rho,
                                         "bands_match": (abs(bs - meas_s) <= BAND_TOL
                                                         and abs(bl - meas_l) <= BAND_TOL),
                                         "beats_dist": rho > rho_dist})
        results[mode] = rows
        nm = sum(1 for x in rows if x["bands_match"])
        nb_ = sum(1 for x in rows if x["bands_match"] and x["beats_dist"])
        say(f"     {mode:6s}: {nm} of {len(rows)} match both bands, {nb_} also beat the null")

    summary = {}
    for mode in ("bend", "spring"):
        cands = [x for x in results[mode] if x["bands_match"]] or results[mode]
        best = max(cands, key=lambda x: x["rho_map"])
        B = L82.run_point(C21, bf, br, best["sep_kb"], best["res_s"], best["v_kb_s"],
                          G0[(mode, best["kappa"], best["alpha"])], DT_FINAL, NCFG_FINAL, SEED)
        rho = L77.band_rho(B["M"], H, mask, n, w)[0]
        bs = L80.ps_band(B["M"], mask, *SHORT_BAND)
        bl = L80.ps_band(B["M"], mask, *LONG_BAND)
        o, _ = L77.orientation_effect(B["M"], B["exp"], fs, rs, mask, n)
        Bs = L82.run_point(C21, bfs, brs, best["sep_kb"], best["res_s"], best["v_kb_s"],
                           G0[(mode, best["kappa"], best["alpha"])], DT_FINAL, NCFG_FINAL, SEED)
        osh, _ = L77.orientation_effect(Bs["M"], Bs["exp"], fss, rss, mask, n)
        cb, _ = L81.checkerboard(B["M"], c, mask, n, w)
        summary[mode] = {"best": best, "rho": rho, "short": bs, "long": bl,
                         "orient": o, "orient_shuf": osh, "checkerboard": cb,
                         "lp_kb": (lps.get((mode, best["kappa"]), float("nan")) * BIN / 1e3)}
        say(f"     {mode:6s} best: kappa {best['kappa']:4.1f} alpha {best['alpha']:.0e} "
            f"sep {best['sep_kb']:.0f} res {best['res_s']:.0f}")
        say(f"            bands {bs:+.4f}/{bl:+.4f}  rho {rho:+.4f}  "
            f"orient {o:+.4f}->{osh:+.4f}  checkerboard {cb:+.4f}")
    cbm, _ = L81.checkerboard(H, c, mask, n, w)
    say(f"     measured: bands {meas_s:+.4f}/{meas_l:+.4f}  checkerboard {cbm:+.4f}")
    p4 = (summary["bend"]["rho"] >= L82_RHO - MATCH_TOL
          and abs(summary["bend"]["short"] - meas_s) <= BAND_TOL
          and abs(summary["bend"]["long"] - meas_l) <= BAND_TOL)
    say(f"     P4 {'PASS' if p4 else 'FAIL'} -- true bending "
        f"{'REPRODUCES' if p4 else 'does NOT reproduce'} loop 82's result "
        f"({summary['bend']['rho']:+.4f} vs {L82_RHO:+.4f})")
    say(f"     P5 PASS (head-to-head reported)")
    say()

    say("P6 HELD OUT: CHROMOSOME 22")
    win = "bend" if summary["bend"]["rho"] >= summary["spring"]["rho"] else "spring"
    wb = summary[win]["best"]
    say(f"     carrying forward the {win} operator (higher chr21 rho)")
    C22 = L79.build_chrom("chr22", "hg19_chr22.fa.gz")
    n22 = C22["n"]
    c22 = L81.comp_score(L81.gc_track(L77.SC / "hg19_chr22.fa.gz", n22), C22["mask"])
    cm22 = max(float(np.maximum(c22, 0).sum()), float(np.maximum(-c22, 0).sum()))
    bf22, br22 = L79.landscape(C22, C22["orients"])
    L22 = base_laplacian(n22, wb["kappa"], c22, wb["alpha"] / cm22, win)
    assert float(np.linalg.eigvalsh(L22).min()) > 0, "chr22 base indefinite"
    G22 = np.linalg.inv(L22)
    rho_d22 = L77.band_rho(L79.distance_null(C22), C22["H"], C22["mask"], n22, w)[0]
    T = L82.run_point(C22, bf22, br22, wb["sep_kb"], wb["res_s"], wb["v_kb_s"], G22,
                      DT_FINAL, NCFG_FINAL, SEED)
    rho22 = L77.band_rho(T["M"], C22["H"], C22["mask"], n22, w)[0]
    s22 = L80.ps_band(T["M"], C22["mask"], *SHORT_BAND)
    l22 = L80.ps_band(T["M"], C22["mask"], *LONG_BAND)
    ms22 = L80.ps_band(C22["H"], C22["mask"], *SHORT_BAND)
    ml22 = L80.ps_band(C22["H"], C22["mask"], *LONG_BAND)
    say(f"     chr22 {rho22:+.4f} vs its null {rho_d22:+.4f}   (loop 82 got +0.8779)")
    say(f"     chr22 bands simulated {s22:+.4f}/{l22:+.4f}  measured {ms22:+.4f}/{ml22:+.4f}")
    p6 = rho22 > rho_d22
    say(f"     P6 {'PASS' if p6 else 'FAIL'}")
    say()

    gates = {"P1 operators distinguished by measurement": bool(p1),
             "P2 persistence length measured": True,
             "P3 PSD and map is an identity": bool(p3),
             "P4 true bending reproduces loop 82": bool(p4),
             "P5 head-to-head reported": True,
             "P6 transfers to chr22": bool(p6)}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L77.HIC), str(L77.SC / "hic_chr22_25kb.npy"), str(L77.FASTA),
                              str(L77.SC / "hg19_chr22.fa.gz"), str(L77.CTCF), str(L77.PFM)],
                      available=len(results["bend"]) + len(results["spring"]),
                      used=len(results["bend"]) + len(results["spring"]),
                      selection="all", seed=SEED,
                      controls=["bond-vector correlation measured for both operators",
                                "persistence length from exponential decay, not a crossover guess",
                                "every base Laplacian asserted positive definite",
                                "Woodbury checked against a fresh full inversion for both operators",
                                "identical grid, compartment sweep and seeds for both arms",
                                "orientation shuffle control at both best points",
                                "chr22 held out with its own GC track and compartment mass"],
                      note="loop 82's term was a second-neighbour SPRING (zigzag), not a bending "
                           "penalty; its empirical result stands but its mechanism was mislabelled")
    RM.report(man, emit=say)
    json.dump({"test": "loop_bending_true", "manifest": man, "gates": gates,
               "persistence_lengths_kb": {f"{m}_{k}": v * BIN / 1e3 for (m, k), v in lps.items()},
               "measured_short": meas_s, "measured_long": meas_l,
               "checkerboard_measured": cbm, "loop82_rho": L82_RHO,
               "summary": {m: {kk: vv for kk, vv in s.items() if kk != "best"} |
                           {"best": s["best"]} for m, s in summary.items()},
               "grid_bend": results["bend"], "grid_spring": results["spring"],
               "winner": win, "chr22": {"rho": rho22, "null": rho_d22, "short": s22, "long": l22},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_bending_true.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_bending_true.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
