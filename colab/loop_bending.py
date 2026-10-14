"""LOOP 80 -- THE BENDING LIMIT: HOW HARD CAN CHROMATIN BE FORCED TO TURN, AND DOES IT RESOLVE THE
THREE-WAY INCOMPATIBILITY?

FIRST, A DEFECT I INTRODUCED AND HAVE TO NAME. Loop 36 measured something loops 34/35 had assumed
silently: that a cohesin loop bond is as stiff as a backbone bond. It is not. Cohesin is a ring
holding two loci at its own diameter, ~40 nm, while a 25 kb bead is ~79 nm across, so the loop bond
is STIFFER than the backbone by (79/40)^2 = 3.9, confirmed by a second route (s/(1+ks) = (40/79)^2 at
s = 50 gives k = 3.8). Loop 36 called that k_loop and used it.

Loops 77, 78 and 79 -- mine -- did not. The Woodbury update is L = L0 + U U^T, and the implicit unit
weight on U silently restores exactly the assumption loop 36 had refuted. Every number in those three
loops was computed at k_loop = 1. That is not a small detail here: loop 36 reported the orientation
signature going from +0.0272 to +0.2278 when the stiffness was corrected, an eightfold change in the
observable that loop 79's W4 gate then failed on.

WHAT BENDING MEANS AT THIS RESOLUTION, stated carefully because the intuitive version is wrong. At
25 kb per bead each bead already contains far more than one persistence length of chromatin, so the
BACKBONE is fully flexible and a bead-to-bead bending penalty would be physics that has already been
coarse-grained away. The bending constraint that survives coarse-graining is the one at the LOOP
ANCHOR: the ring forces two 79 nm beads to sit 40 nm apart, and that is a geometric limit on how
sharply the chain can be made to turn. So "how much can it bend before it breaks" enters this model
as k_loop, and it enters nowhere else.

THE HYPOTHESIS WORTH TESTING, and it is why this is worth a loop rather than a patch. Loop 79 found
that no literature-admissible parameter point satisfies all three measured observables: the point
matching contact decay places contacts worse than a distance null, and the point matching the map
loses the CTCF orientation signature. Those three observables live at DIFFERENT SCALES -- P(s) is
fitted over 100 kb to 10 Mb, map correlation is dominated by the near-diagonal, and the orientation
effect is measured at CTCF pairs under 2 Mb. A loop-anchor stiffness acts specifically at the anchor,
i.e. at short range. So it is exactly the kind of term that could decouple them. If it cannot, the
incompatibility is structural and no amount of extra local physics will fix it.

PREDECLARED, before any number:

  K1 THE WEIGHTED FAST MAP IS STILL AN IDENTITY
       Woodbury with W = k*I against a fresh weighted Laplacian inversion, on the same
       configurations. Gate: max relative error <= 1e-6. Loop 77's V1 established this at k = 1; a
       weighted update is different arithmetic and gets its own check rather than an assumption.
  K2 k_loop IS DERIVED, NOT FITTED, AND THE SWEEP IS BOUNDED BY GEOMETRY
       the derivation above fixes k = 3.9 from ring and bead diameters. The sweep spans
       k in {1, 2, 3.9, 8, 16} so that the derived value sits inside a range covering a rigid ring
       (large k) and loop 35's silent assumption (k = 1). Reported, not judged.
  K3 BENDING ACTS WHERE IT SHOULD, OR IT IS NOT DOING WHAT IT CLAIMS
       P(s) measured separately in a SHORT band (100-500 kb) and a LONG band (1-10 Mb). A loop-anchor
       term must move the short band more than the long band. If it moves them equally it is acting
       as a global rescaling, not as a bending constraint, and the physical story is wrong.
  K4 DOES BENDING RESOLVE THE THREE-WAY INCOMPATIBILITY          THE GATE.
       across the full grid x k_loop, count points that satisfy ALL THREE: P(s) inside
       (-1.16, -0.76), map correlation above the distance-only null, and a convergent-CTCF signature
       that collapses below half under motif shuffling. Loop 79 found 0 of 45 satisfying all three.
       Gate: at least one point must. If none does with the bending term restored, the failure is
       structural rather than a missing local force.
  K5 HELD OUT: CHROMOSOME 22
       the best all-three point (or, if none, the best two-of-three) applied unchanged to chr22 and
       scored against chr22's own distance-only null, exactly as loop 79 did.
  K6 WHAT k_loop = 1 COST LOOPS 77-79
       every headline from those three loops recomputed at the derived k, so the size of my own
       omission is a number in the record rather than an apology.

-> outputs/loop_bending.json
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

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BIN = L77.BIN
PS_WINDOW = L77.PS_WINDOW
MEASURED_PS = -0.9636

RING_NM, BEAD_NM = 40.0, 79.0
K_DERIVED = (BEAD_NM / RING_NM) ** 2          # 3.90
K_SWEEP = [1.0, 2.0, K_DERIVED, 8.0, 16.0]

SEPARATION_KB = [100.0, 200.0, 300.0, 400.0]
RESIDENCE_S = [600.0, 900.0, 1500.0]
SPEED_KB_S = [0.5, 0.75, 1.0]
DT_SWEEP, DT_FINAL = 3.0, 1.0
NCFG_SWEEP, NCFG_FINAL = 15, 50
SHORT_BAND = (1e5, 5e5)
LONG_BAND = (1e6, 1e7)
K1_TOL = 1e-6
SEED = 8001

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def r2_woodbury_k(G0, cfg, k):
    """<R^2> with loop bonds at stiffness k. L = L0 + U (k I) U^T, so the Woodbury core is
    (k^-1 I + U^T G0 U)^-1 rather than (I + U^T G0 U)^-1."""
    loops = [(int(a), int(b)) for a, b in cfg if b > a]
    n = G0.shape[0]
    if loops:
        m = len(loops)
        U = np.zeros((n, m))
        for i, (a, b) in enumerate(loops):
            U[a, i] = 1.0
            U[b, i] = -1.0
        GU = G0 @ U
        M = np.eye(m) / k + U.T @ GU
        G = G0 - GU @ np.linalg.solve(M, GU.T)
    else:
        G = G0
    d = np.diag(G)
    return d[:, None] + d[None, :] - 2.0 * G


def contact_map_k(n, configs, G0, k):
    acc = np.zeros((n, n))
    for cfg in configs:
        R2 = r2_woodbury_k(G0, cfg, k)
        np.fill_diagonal(R2, np.inf)
        acc += np.maximum(R2, 1e-12) ** -1.5
    return acc / max(len(configs), 1)


def contact_map_exact_k(n, configs, k, confine=L77.CONFINE):
    """Fresh weighted Laplacian inversion, for K1 only."""
    from loop_polymer import r2_matrix
    acc = np.zeros((n, n))
    for cfg in configs:
        L = np.zeros((n, n))
        idx = np.arange(n - 1)
        L[idx, idx + 1] -= 1.0
        L[idx + 1, idx] -= 1.0
        L[idx, idx] += 1.0
        L[idx + 1, idx + 1] += 1.0
        for a, b in cfg:
            a, b = int(a), int(b)
            if a == b:
                continue
            L[a, b] -= k
            L[b, a] -= k
            L[a, a] += k
            L[b, b] += k
        L = L + confine * np.eye(n)
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return acc / max(len(configs), 1)


def ps_band(M, mask, lo, hi):
    from loop_hic_target import expected
    e = expected(M, mask)
    d = np.arange(len(M)) * BIN
    s = np.isfinite(e) & (d >= lo) & (d <= hi) & (e > 0)
    if s.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log10(d[s]), np.log10(e[s]), 1)[0])


def run_point(C, bf, br, sep, res, spd, k, dt, ncfg, seed):
    old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
    L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = sep, res, spd
    try:
        cfgs, ncoh, _ = L77.simulate(C["n"], bf, br, np.random.default_rng(seed), dt, n_config=ncfg)
    finally:
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    M = contact_map_k(C["n"], cfgs, C["G0"], k)
    ps, exp = L77.ps_slope(M, C["mask"])
    return {"M": M, "exp": exp, "ps": ps, "cfgs": cfgs, "n_coh": ncoh}


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 80 -- the bending limit at the loop anchor, and whether it resolves the")
    say("  three-way incompatibility loop 79 measured")
    say("=" * 100)
    say()

    C21 = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    n, mask, H = C21["n"], C21["mask"], C21["H"]
    w = int(L77.BAND_BP // BIN)
    bf, br = L79.landscape(C21, C21["orients"])
    fs, rs = L79.sites(C21, C21["orients"])
    rho_dist = L77.band_rho(L79.distance_null(C21), H, mask, n, w)[0]
    say(f"  chr21 {n:,} bins;  distance-only null {rho_dist:+.4f};  replicate ceiling +0.9441")
    say(f"  ring {RING_NM:.0f} nm, bead {BEAD_NM:.0f} nm  ->  k_loop = (bead/ring)^2 = "
        f"{K_DERIVED:.2f}")
    say()

    say("K1 THE WEIGHTED FAST MAP IS STILL AN IDENTITY")
    cfgs, _, _ = L77.simulate(n, bf, br, np.random.default_rng(SEED), DT_FINAL, n_config=4)
    worst = 0.0
    for k in (1.0, K_DERIVED, 16.0):
        A = contact_map_k(n, cfgs, C21["G0"], k)
        B = contact_map_exact_k(n, cfgs, k)
        f = np.isfinite(A) & np.isfinite(B) & (B > 0)
        e = float(np.max(np.abs(A[f] - B[f]) / np.abs(B[f])))
        worst = max(worst, e)
        say(f"     k = {k:5.2f}   max relative difference {e:.3e}")
    k1 = worst <= K1_TOL
    say(f"     K1 {'PASS' if k1 else 'FAIL'}  (gate {K1_TOL:.0e})")
    say()

    say("K2 k_loop IS DERIVED, AND THE SWEEP IS BOUNDED BY GEOMETRY")
    say(f"     sweep k in {[round(x,2) for x in K_SWEEP]};  derived {K_DERIVED:.2f}; "
        f"k = 1 is loop 35's silent assumption and what loops 77-79 actually ran")
    say(f"     K2 PASS (declared)")
    say()

    say("K3 BENDING ACTS AT SHORT RANGE, OR IT IS NOT A BENDING TERM")
    ref = run_point(C21, bf, br, 200.0, 900.0, 0.75, 1.0, DT_FINAL, NCFG_FINAL, SEED)
    base_s = ps_band(ref["M"], mask, *SHORT_BAND)
    base_l = ps_band(ref["M"], mask, *LONG_BAND)
    say(f"     at k = 1:  short band (100-500 kb) {base_s:+.4f}   long band (1-10 Mb) {base_l:+.4f}")
    dS = dL = 0.0
    for k in (K_DERIVED, 16.0):
        R = run_point(C21, bf, br, 200.0, 900.0, 0.75, k, DT_FINAL, NCFG_FINAL, SEED)
        s_, l_ = ps_band(R["M"], mask, *SHORT_BAND), ps_band(R["M"], mask, *LONG_BAND)
        say(f"     at k = {k:5.2f}: short {s_:+.4f} (d {s_-base_s:+.4f})   "
            f"long {l_:+.4f} (d {l_-base_l:+.4f})")
        if abs(k - K_DERIVED) < 1e-9:
            dS, dL = abs(s_ - base_s), abs(l_ - base_l)
    k3 = dS > dL
    say(f"     at the derived k the short band moves {dS:.4f} and the long band {dL:.4f}")
    say(f"     K3 {'PASS' if k3 else 'FAIL'} -- it {'is' if k3 else 'is NOT'} acting as a local "
        f"anchor constraint")
    say()

    say("K4 DOES BENDING RESOLVE THE THREE-WAY INCOMPATIBILITY")
    rng = np.random.default_rng(SEED)
    sh = list(rng.permutation(C21["orients"]))
    bfs, brs = L79.landscape(C21, sh)
    fss, rss = L79.sites(C21, sh)
    grid = [(a, b, c, k) for a in SEPARATION_KB for b in RESIDENCE_S for c in SPEED_KB_S
            for k in K_SWEEP]
    rows = []
    for i, (sep, res, spd, k) in enumerate(grid, 1):
        R = run_point(C21, bf, br, sep, res, spd, k, DT_SWEEP, NCFG_SWEEP, SEED)
        rho = L77.band_rho(R["M"], H, mask, n, w)[0]
        inw = PS_WINDOW[0] <= R["ps"] <= PS_WINDOW[1]
        beats = rho > rho_dist
        row = {"sep_kb": sep, "res_s": res, "v_kb_s": spd, "k_loop": k, "ps": R["ps"],
               "rho_map": rho, "ps_in_window": inw, "beats_dist": beats,
               "orient": None, "orient_shuf": None, "all_three": False}
        if inw and beats:                      # only then is the orientation test worth running
            o, _ = L77.orientation_effect(R["M"], R["exp"], fs, rs, mask, n)
            Rs = run_point(C21, bfs, brs, sep, res, spd, k, DT_SWEEP, NCFG_SWEEP, SEED)
            os_, _ = L77.orientation_effect(Rs["M"], Rs["exp"], fss, rss, mask, n)
            row["orient"], row["orient_shuf"] = o, os_
            row["all_three"] = bool(np.isfinite(o) and o > 0
                                    and (not np.isfinite(os_) or os_ < 0.5 * o))
            say(f"       sep {sep:5.0f} res {res:6.0f} v {spd:4.2f} k {k:5.2f}  "
                f"P(s) {R['ps']:+.4f}  rho {rho:+.4f}  orient {o:+.4f}->{os_:+.4f}"
                f"{'   ALL THREE' if row['all_three'] else ''}")
        rows.append(row)
        if i % 30 == 0:
            say(f"       ... {i}/{len(grid)}")
    n_two = sum(1 for x in rows if x["ps_in_window"] and x["beats_dist"])
    n_all = sum(1 for x in rows if x["all_three"])
    say(f"     {n_two} of {len(rows)} points satisfy P(s)-in-window AND beat the distance null")
    say(f"     {n_all} of {len(rows)} satisfy ALL THREE  (loop 79, without bending: 0 of 45)")
    k4 = n_all > 0
    say(f"     K4 {'PASS' if k4 else 'FAIL'} -- the incompatibility is "
        f"{'resolvable with the anchor constraint restored' if k4 else 'STRUCTURAL'}")
    say()

    say("K5 HELD OUT: CHROMOSOME 22")
    cands = [x for x in rows if x["all_three"]] or [x for x in rows
                                                    if x["ps_in_window"] and x["beats_dist"]]
    best = max(cands, key=lambda x: x["rho_map"]) if cands else max(rows, key=lambda x: x["rho_map"])
    say(f"     carrying forward: sep {best['sep_kb']:.0f} kb, res {best['res_s']:.0f} s, "
        f"v {best['v_kb_s']:.2f} kb/s, k {best['k_loop']:.2f}"
        f"{' (all three)' if best['all_three'] else ' (best available)'}")
    C22 = L79.build_chrom("chr22", "hg19_chr22.fa.gz")
    bf22, br22 = L79.landscape(C22, C22["orients"])
    rho_d22 = L77.band_rho(L79.distance_null(C22), C22["H"], C22["mask"], C22["n"], w)[0]
    T = run_point(C22, bf22, br22, best["sep_kb"], best["res_s"], best["v_kb_s"],
                  best["k_loop"], DT_FINAL, NCFG_FINAL, SEED)
    rho22 = L77.band_rho(T["M"], C22["H"], C22["mask"], C22["n"], w)[0]
    say(f"     chr22 model {rho22:+.4f}   chr22 distance null {rho_d22:+.4f}   "
        f"(loop 79 got +0.8710 vs +0.8517)")
    k5 = rho22 > rho_d22
    say(f"     K5 {'PASS' if k5 else 'FAIL'}")
    say()

    say("K6 WHAT k_loop = 1 COST LOOPS 77-79")
    pairs = [("loop 79 best-by-map", 100.0, 600.0, 0.5), ("loop 78 best-by-P(s)", 300.0, 1500.0, 1.0),
             ("loop 35 default", 150.0, 900.0, 0.75)]
    cost = []
    for nm, sep, res, spd in pairs:
        a = run_point(C21, bf, br, sep, res, spd, 1.0, DT_FINAL, NCFG_FINAL, SEED)
        b = run_point(C21, bf, br, sep, res, spd, K_DERIVED, DT_FINAL, NCFG_FINAL, SEED)
        ra = L77.band_rho(a["M"], H, mask, n, w)[0]
        rb = L77.band_rho(b["M"], H, mask, n, w)[0]
        oa, _ = L77.orientation_effect(a["M"], a["exp"], fs, rs, mask, n)
        ob, _ = L77.orientation_effect(b["M"], b["exp"], fs, rs, mask, n)
        cost.append({"point": nm, "ps_k1": a["ps"], "ps_kd": b["ps"], "rho_k1": ra, "rho_kd": rb,
                     "orient_k1": oa, "orient_kd": ob})
        say(f"     {nm:22s}  P(s) {a['ps']:+.4f} -> {b['ps']:+.4f}   "
            f"rho {ra:+.4f} -> {rb:+.4f}   orient {oa:+.4f} -> {ob:+.4f}")
    say(f"     K6 PASS (reported)")
    say()

    gates = {"K1 weighted fast map is an identity": bool(k1),
             "K2 k_loop derived and sweep bounded": True,
             "K3 bending acts at short range": bool(k3),
             "K4 bending resolves the three-way incompatibility": bool(k4),
             "K5 transfers to chr22": bool(k5),
             "K6 cost of k_loop=1 reported": True}
    for kk, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {kk}")

    man = RM.manifest(inputs=[str(L77.HIC), str(L77.SC / "hic_chr22_25kb.npy"), str(L77.CTCF),
                              str(L77.FASTA), str(L77.PFM)],
                      available=len(grid), used=len(rows), selection="all", seed=SEED,
                      controls=["weighted Woodbury checked against a fresh weighted inversion",
                                "k_loop derived from ring and bead geometry, not fitted",
                                "short-band vs long-band P(s) to test that bending is local",
                                "orientation shuffle control on every candidate point",
                                "all three observables required simultaneously",
                                "chr22 held out entirely",
                                "the cost of the k_loop=1 omission in loops 77-79 quantified"],
                      note="loops 77-79 silently ran at k_loop = 1, restoring the assumption "
                           "loop 36 had refuted")
    RM.report(man, emit=say)
    json.dump({"test": "loop_bending", "manifest": man, "gates": gates,
               "k_derived": K_DERIVED, "k_sweep": K_SWEEP,
               "k1_max_rel_err": worst, "rho_dist_chr21": rho_dist,
               "short_band_shift": dS, "long_band_shift": dL,
               "grid": rows, "n_two_of_three": n_two, "n_all_three": n_all,
               "best": best, "chr22_rho": rho22, "chr22_dist_null": rho_d22,
               "k1_cost": cost, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_bending.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_bending.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
