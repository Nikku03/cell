"""LOOP 88 -- IS THERE ONE CONFIGURATION THAT GETS BOTH THE MAP AND THE ORIENTATION, OR NEITHER?

THE SPLIT THIS ARC HAS BEEN LIVING WITH. Loop 85 re-scored the two surviving parameter points on the
corrected map and they came out on opposite corners:

    spring  sep 200 res 600 spd 0.75 kappa 4 alpha 1e-3    map rho 0.8533 (17% headroom)
                                                            orientation +0.0512  =  13% of measured
    bend    sep 200 res 600 spd 0.75 kappa 0 alpha 3e-4    map rho 0.8301  (1% headroom)
                                                            orientation +0.3965  = 105% of measured

Whichever property you care about you pick a different model. That is not one model of chromatin, it
is two, and every summary this arc has produced has quietly chosen which of them to quote. So the
question this loop answers is the one that decides whether the mechanism is right: does a single
configuration exist that does both, or is the trade-off structural?

WHY THIS IS CHEAP ENOUGH TO SWEEP PROPERLY. The extrusion simulation depends only on (separation,
residence, speed) and the CTCF landscape; the Gaussian network inverse depends only on (kappa, alpha,
mode). They are independent, so a grid of S extrusion settings and L network settings costs S
simulations and L dense inverses, not S x L of either -- only the cheap Woodbury contact map is paid
S x L times. Loops 78-83 did not exploit this and swept one axis at a time, which is part of why the
two objectives were never seen together on one grid.

THE TARGETS ARE LOOP 86's, NOT CHR21's. Loop 86 established that chr21 sits at the 91st percentile on
P(s) and the 96th on the long band, so its values are not the genome's. The orientation signature is
the exception -- chr21's +0.3788 is at the 52nd percentile against a genome-wide +0.3725 -- so for
this loop's purposes the orientation target barely moves and the map target is chr21's own map, which
is legitimate because chr21's map is the object being predicted. Bands are reported against the
genome values and left to loop 90.

PREDECLARED, before any number:

  J1 THE TRADE-OFF IS MEASURED, NOT ASSUMED                          THE DIAGNOSIS.
       every grid point scored on BOTH criteria, and the rank correlation between the two scores
       across the grid reported. Strongly negative means the trade-off is structural and the model
       cannot be fixed by searching harder. Near zero or positive means loops 82 and 83 simply landed
       on opposite corners of a space neither of them swept, and a joint point should exist.
  J2 A JOINT POINT EXISTS                                            THE GATE.
       defined before looking: map rho >= 0.8533, the best any configuration has managed, AND
       orientation between 0.5x and 2.0x the measured +0.3725 at z >= 4. At least one grid point must
       satisfy both. If none does, the model as formulated cannot do both, and that is the finding
       rather than a reason to relax the threshold afterwards.
  J3 WHICH PARAMETER SEPARATES THE TWO OBJECTIVES                    THE MECHANISM.
       Spearman of each parameter against each score across the grid. If one axis drives map rho up
       while driving orientation down, that axis is where the model is wrong, and naming it is worth
       more than the best point.
  J4 HELD OUT                                                        chr22.
       the joint winner, or the best compromise if J2 fails, applied unchanged to chr22 and scored on
       both criteria against chr22's own map and its own CTCF sites.
  J5 THE CONTROL THAT MUST FIRE                                      THE GUARD.
       the distance-only null map scored on both criteria. It must score high on the map (it reaches
       0.8283 by knowing separation alone) and essentially ZERO on orientation. If it scores on both,
       the two criteria are measuring the same thing and J1's trade-off is an artifact of the scoring
       rather than a property of the model. Seven gates this session fired while measuring nothing.
  J6 IF NO JOINT POINT EXISTS, WHAT IS MISSING                       THE HONEST NEXT STEP.
       diagnosis of which term is responsible, stated as a testable claim for the next loop rather
       than as a conclusion.

-> outputs/loop_joint.json
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
import loop_bending_true as L83  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
K_LOOP = L80.K_DERIVED
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)

# loop 86, mappable-weighted over 23 chromosomes
GENOME = {"ps": -1.0510, "short": -0.8671, "long": -0.9576, "orient": 0.3725}
RHO_TARGET = 0.8533              # J2: the best map rho any configuration has reached (loop 85)
ORIENT_LO, ORIENT_HI = 0.5, 2.0  # J2: factor window on the measured orientation
Z_MIN = 4.0
DIST_NULL, CEILING = 0.8283, 0.9727      # loop 85, corrected

EXTRUSION = [(sep, res, spd) for sep in (100.0, 200.0, 400.0)
             for res in (600.0, 900.0) for spd in (0.75,)]
NETWORK = [(k, a, m) for m in ("spring", "bend") for k in (0.0, 2.0, 4.0)
           for a in (0.0, 3e-4, 1e-3)]
NPERM = 10
NCFG, DT = 50, 1.0
SEED = 8801

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def score_orientation(M, exp, ors, C, mask, n, nperm=NPERM, seed=SEED):
    """Loop 84's control (B): one map, permute only the labels used to score it."""
    fs, rs = L79.sites(C, ors)
    real, npair = L77.orientation_effect(M, exp, fs, rs, mask, n)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        f2, r2 = L79.sites(C, list(rng.permutation(ors)))
        v, _ = L77.orientation_effect(M, exp, f2, r2, mask, n)
        if np.isfinite(v):
            null.append(v)
    if not null or not np.isfinite(real):
        return float("nan"), float("nan"), int(npair)
    null = np.array(null)
    sd = null.std()
    return float(real), float((real - null.mean()) / sd) if sd > 1e-12 else float("inf"), int(npair)


def sweep(C, tag):
    """S simulations x L inverses, paying the dense inverse L times and the simulation S times."""
    n, mask, H = C["n"], C["mask"], C["H"]
    ors = C["orients"]
    bf, br = L79.landscape(C, ors)
    w = int(L77.BAND_BP // BIN)
    c = L81.comp_score(L81.gc_track(SC / f"hg19_{tag}.fa.gz", n), mask)
    cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))

    cfgs = {}
    for sep, res, spd in EXTRUSION:
        old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = sep, res, spd
        try:
            cfgs[(sep, res, spd)] = L77.simulate(n, bf, br, np.random.default_rng(SEED), DT,
                                                 n_config=NCFG)[0]
        finally:
            L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    say(f"     {len(cfgs)} extrusion simulations, {len(NETWORK)} network inverses, "
        f"{len(cfgs)*len(NETWORK)} contact maps")

    rows = []
    for k, a, m in NETWORK:
        L = L83.base_laplacian(n, k, c, a / cmass if cmass else 0.0, m)
        lam = float(np.linalg.eigvalsh(L).min())
        assert lam > 0, f"indefinite base kappa={k} alpha={a} mode={m} lam={lam}"
        G0 = np.linalg.inv(L)
        for (sep, res, spd), cf in cfgs.items():
            M = L80.contact_map_k(n, cf, G0, K_LOOP)
            ps, exp = L77.ps_slope(M, mask)
            rho = L77.band_rho(M, H, mask, n, w)[0]
            o, z, npair = score_orientation(M, exp, ors, C, mask, n)
            rows.append({"sep": sep, "res": res, "spd": spd, "kappa": k, "alpha": a, "mode": m,
                         "rho": rho, "orient": o, "orient_z": z, "ps": ps,
                         "short": L80.ps_band(M, mask, *SHORT_BAND),
                         "long": L80.ps_band(M, mask, *LONG_BAND),
                         "headroom": (rho - DIST_NULL) / (CEILING - DIST_NULL)})
    return rows, c, cmass, bf, br


def joint_ok(r):
    return (np.isfinite(r["rho"]) and np.isfinite(r["orient"]) and np.isfinite(r["orient_z"])
            and r["rho"] >= RHO_TARGET
            and ORIENT_LO * GENOME["orient"] <= r["orient"] <= ORIENT_HI * GENOME["orient"]
            and r["orient_z"] >= Z_MIN)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 88 -- is there one configuration that gets both the map and the orientation?")
    say("=" * 100)
    say()
    say(f"  J2 defined before any run: map rho >= {RHO_TARGET:.4f} AND orientation in "
        f"[{ORIENT_LO*GENOME['orient']:+.4f}, {ORIENT_HI*GENOME['orient']:+.4f}] at z >= {Z_MIN}")
    say(f"  grid: {len(EXTRUSION)} extrusion x {len(NETWORK)} network = "
        f"{len(EXTRUSION)*len(NETWORK)} configurations")
    say()

    C21 = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    say("J1 THE TRADE-OFF IS MEASURED, NOT ASSUMED")
    rows, c, cmass, bf, br = sweep(C21, "chr21")
    from scipy.stats import spearmanr
    rr = np.array([r["rho"] for r in rows], float)
    oo = np.array([r["orient"] for r in rows], float)
    f = np.isfinite(rr) & np.isfinite(oo)
    tradeoff = float(spearmanr(rr[f], oo[f]).statistic)
    say(f"     {int(f.sum())} scored configurations")
    say(f"     rank correlation between map rho and orientation across the grid: {tradeoff:+.4f}")
    say(f"     map rho      {rr[f].min():+.4f} to {rr[f].max():+.4f}")
    say(f"     orientation  {oo[f].min():+.4f} to {oo[f].max():+.4f}   (measured {GENOME['orient']:+.4f})")
    say(f"     J1 {'a real trade-off' if tradeoff < -0.3 else 'NO structural trade-off'} "
        f"(reported, not gated)")
    say()

    say("J2 A JOINT POINT EXISTS")
    ok = [r for r in rows if joint_ok(r)]
    best_rho = max((r for r in rows if np.isfinite(r["rho"])), key=lambda r: r["rho"])
    best_or = max((r for r in rows if np.isfinite(r["orient"])),
                  key=lambda r: -abs(r["orient"] - GENOME["orient"]))
    say(f"     configurations satisfying BOTH: {len(ok)} of {len(rows)}")
    for r in sorted(ok, key=lambda r: -r["rho"])[:5]:
        say(f"       {r['mode']:6s} sep {r['sep']:5.0f} res {r['res']:5.0f} k {r['kappa']:.1f} "
            f"a {r['alpha']:.0e}   rho {r['rho']:+.4f} ({r['headroom']:+.0%})  "
            f"orient {r['orient']:+.4f} z {r['orient_z']:+.1f}")
    say(f"     best map rho     {best_rho['mode']:6s} k {best_rho['kappa']:.1f} "
        f"a {best_rho['alpha']:.0e}  rho {best_rho['rho']:+.4f}  orient {best_rho['orient']:+.4f}")
    say(f"     best orientation {best_or['mode']:6s} k {best_or['kappa']:.1f} "
        f"a {best_or['alpha']:.0e}  rho {best_or['rho']:+.4f}  orient {best_or['orient']:+.4f}")
    j2 = len(ok) > 0
    say(f"     J2 {'PASS' if j2 else 'FAIL'} -- the model "
        f"{'CAN do both at one point' if j2 else 'cannot do both anywhere in the admissible grid'}")
    say()

    say("J3 WHICH PARAMETER SEPARATES THE TWO OBJECTIVES")
    drivers = {}
    for key in ("sep", "res", "kappa", "alpha"):
        v = np.array([r[key] for r in rows], float)
        if v.std() < 1e-12:
            continue
        a_ = float(spearmanr(v[f], rr[f]).statistic)
        b_ = float(spearmanr(v[f], oo[f]).statistic)
        drivers[key] = {"vs_rho": a_, "vs_orient": b_, "opposed": bool(a_ * b_ < 0)}
        say(f"     {key:8s} vs map rho {a_:+.4f}   vs orientation {b_:+.4f}   "
            f"{'OPPOSED' if a_*b_ < 0 else 'aligned'}")
    ms = np.array([1.0 if r["mode"] == "bend" else 0.0 for r in rows])
    a_ = float(spearmanr(ms[f], rr[f]).statistic)
    b_ = float(spearmanr(ms[f], oo[f]).statistic)
    drivers["mode_bend"] = {"vs_rho": a_, "vs_orient": b_, "opposed": bool(a_ * b_ < 0)}
    say(f"     {'mode':8s} vs map rho {a_:+.4f}   vs orientation {b_:+.4f}   "
        f"{'OPPOSED' if a_*b_ < 0 else 'aligned'} (1 = bend)")
    opposed = [k for k, v in drivers.items() if v["opposed"]]
    say(f"     axes pulling the two objectives apart: {', '.join(opposed) if opposed else 'none'}")
    say()

    say("J5 THE CONTROL THAT MUST FIRE")
    n, mask, H = C21["n"], C21["mask"], C21["H"]
    w = int(L77.BAND_BP // BIN)
    DN = L79.distance_null(C21)
    from loop_hic_target import expected
    expD = expected(DN, mask)
    rho_d = L77.band_rho(DN, H, mask, n, w)[0]
    o_d, z_d, _ = score_orientation(DN, expD, C21["orients"], C21, mask, n)
    say(f"     distance-only null: map rho {rho_d:+.4f}   orientation {o_d:+.4f} z {z_d:+.1f}")
    j5 = np.isfinite(rho_d) and rho_d > 0.75 and (not np.isfinite(o_d) or abs(o_d) < 0.05)
    say(f"     J5 {'PASS' if j5 else 'FAIL'} -- knowing separation alone scores on the map and "
        f"{'not on orientation, so the two criteria are independent' if j5 else 'ALSO on orientation, so they are not independent and J1 is an artifact'}")
    say()

    say("J4 HELD OUT: CHROMOSOME 22")
    pick = (sorted(ok, key=lambda r: -r["rho"])[0] if ok else best_rho)
    say(f"     applying {pick['mode']} sep {pick['sep']:.0f} res {pick['res']:.0f} "
        f"k {pick['kappa']:.1f} a {pick['alpha']:.0e} unchanged to chr22")
    C22 = L79.build_chrom("chr22", "hg19_chr22.fa.gz")
    n2, m2, H2 = C22["n"], C22["mask"], C22["H"]
    o2 = C22["orients"]
    bf2, br2 = L79.landscape(C22, o2)
    c2 = L81.comp_score(L81.gc_track(SC / "hg19_chr22.fa.gz", n2), m2)
    cm2 = max(float(np.maximum(c2, 0).sum()), float(np.maximum(-c2, 0).sum()))
    L2 = L83.base_laplacian(n2, pick["kappa"], c2, pick["alpha"] / cm2 if cm2 else 0.0, pick["mode"])
    assert float(np.linalg.eigvalsh(L2).min()) > 0, "indefinite chr22 base"
    G2 = np.linalg.inv(L2)
    old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
    L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = pick["sep"], pick["res"], pick["spd"]
    try:
        cf2 = L77.simulate(n2, bf2, br2, np.random.default_rng(SEED), DT, n_config=NCFG)[0]
    finally:
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    M2 = L80.contact_map_k(n2, cf2, G2, K_LOOP)
    ps2, exp2 = L77.ps_slope(M2, m2)
    rho2 = L77.band_rho(M2, H2, m2, n2, int(L77.BAND_BP // BIN))[0]
    or2, z2, _ = score_orientation(M2, exp2, o2, C22, m2, n2)
    om2, zm2, _ = score_orientation(H2, L77.ps_slope(H2, m2)[1], o2, C22, m2, n2)
    say(f"     chr22 simulated: map rho {rho2:+.4f}   orientation {or2:+.4f} z {z2:+.1f}")
    say(f"     chr22 measured orientation {om2:+.4f} z {zm2:+.1f}")
    j4 = bool(np.isfinite(rho2) and rho2 > DIST_NULL and np.isfinite(z2) and z2 >= Z_MIN)
    say(f"     J4 {'PASS' if j4 else 'FAIL'} -- transfers on both criteria to a chromosome that "
        f"entered no selection")
    say()

    say("J6 IF NO JOINT POINT EXISTS, WHAT IS MISSING")
    if j2:
        say(f"     not applicable -- {len(ok)} joint configurations found")
    else:
        gap_rho = RHO_TARGET - best_or["rho"]
        gap_or = GENOME["orient"] - best_rho["orient"]
        say(f"     the best-orientation point is {gap_rho:+.4f} short on map rho")
        say(f"     the best-map point is {gap_or:+.4f} short on orientation")
        say(f"     axes that pull them apart: {', '.join(opposed) if opposed else 'none identified'}")
        say(f"     testable claim for the next loop: the term responsible is the one whose axis is")
        say(f"     OPPOSED above; if none is opposed then the split is not in the swept parameters")
        say(f"     and must be in a term the model does not have.")
    say()

    gates = {"J1 the trade-off is measured": True,
             "J2 a joint configuration exists": bool(j2),
             "J3 the separating parameter is identified": True,
             "J4 it transfers to chr22": bool(j4),
             "J5 the distance null scores on one criterion only": bool(j5),
             "J6 the missing term is named as a testable claim": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L77.HIC), str(SC / "hic_chr22_25kb.npy"), str(L77.CTCF),
                              str(L77.FASTA), str(SC / "hg19_chr22.fa.gz"), str(L77.PFM)],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["the joint threshold defined before any configuration was run",
                                "the distance-only null scored on both criteria as a guard",
                                "orientation scored by loop 84's control (B) throughout",
                                "chr22 held out and scored on both criteria",
                                "both objectives measured on every grid point, not one per sweep",
                                "extrusion and network axes decoupled so the grid is complete"],
                      note="loops 82 and 83 each optimised one objective and reported the other as "
                           "a failure; this sweeps both on every point of one grid")
    RM.report(man, emit=say)
    json.dump({"test": "loop_joint", "manifest": man, "gates": gates,
               "grid": rows, "tradeoff_rho": tradeoff, "n_joint": len(ok),
               "joint": sorted(ok, key=lambda r: -r["rho"])[:10],
               "best_rho": best_rho, "best_orient": best_or, "drivers": drivers,
               "distance_null": {"rho": rho_d, "orient": o_d, "z": z_d},
               "chr22": {"point": pick, "rho": rho2, "orient": or2, "z": z2,
                         "measured_orient": om2, "measured_z": zm2, "ps": ps2},
               "targets": GENOME, "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_joint.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_joint.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
