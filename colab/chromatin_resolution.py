"""IS THE WRITHE OVERSHOOT A RESOLUTION ARTEFACT? -- testing my own excuse instead of asserting it.

THE SITUATION.  chromatin_supercoil produced a rod that buckles for genuinely torsional reasons -- the
C = 0 control is unambiguous, Wr = 0.050 against -19.8 -- but it puts 96.6% of the excess linking number
into writhe where real DNA puts 70-75%. I proposed an explanation immediately: at 50 bp per bead each
segment is 17 nm, while a plectonemic superhelix has a radius of 5-9 nm, so the chain physically cannot
form the tight coil and instead makes wide smooth loops that cost almost no bending energy. Writhe is too
cheap, so it takes everything.

That is a comfortable explanation and comfortable explanations are exactly the ones to test. If it is
right, Wr/dLk must FALL toward 0.7 as the segment shrinks through the superhelical radius, and must do so
for the reason claimed -- the bending energy per unit writhe must RISE. If Wr/dLk is flat in resolution,
the discretisation is not the problem and something in the model is.

THE DESIGN.  Fixed chain (2000 bp, one sigma), varying only the bead count, so nothing changes but the
segment length. Cost goes as N^3 -- N^2 pairs in the writhe times N attempts per sweep -- so the sweep is
short and deliberately brackets the superhelical radius rather than going as fine as possible:

    N =  40   50.0 bp/bead   17.0 nm   segment much LARGER than a superhelix radius
    N =  80   25.0 bp/bead    8.5 nm   comparable
    N = 120   16.7 bp/bead    5.7 nm   comparable to SMALLER

PREDECLARED, before any number:
    Wr/dLk falls monotonically toward ~0.75 and bending energy per unit writhe rises
        -> the overshoot was discretisation. The model is right and the coarse level of the multi-scale
           scheme simply cannot be trusted for the twist/writhe partition, which is a concrete constraint
           on the architecture rather than a defect in the physics.
    Wr/dLk is flat across resolution
        -> my explanation was wrong. The overshoot is in the model -- most likely the excluded-volume
           diameter or the bending discretisation -- and it has to be found before anything is built on top.
    Wr/dLk falls but nowhere near 0.75
        -> resolution is part of it and not all of it; both effects are real and the remaining gap needs
           its own explanation rather than an extrapolation.

-> outputs/chromatin_resolution.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chromatin_supercoil as SC  # noqa: E402
from chromatin_rod import RISE, KT, A_BEND  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BP = float(os.environ.get("RES_BP", 2000))
SIGMA = float(os.environ.get("RES_SIGMA", -0.05))
SWEEPS = int(os.environ.get("RES_SWEEPS", 900))
BURN = int(os.environ.get("RES_BURN", 300))
NS = tuple(int(x) for x in os.environ.get("RES_N", "40,80,120").split(","))


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("IS THE WRITHE OVERSHOOT A RESOLUTION ARTEFACT? -- testing the excuse, not asserting it")
    report("=" * 100)
    lk0 = BP / 10.5
    dlk = SIGMA * lk0
    report(f"  {BP:.0f} bp, sigma {SIGMA}, dLk {dlk:.2f}, contour {BP*RISE*1e9:.0f} nm, "
           f"{SWEEPS} sweeps ({BURN} burn-in)")
    report(f"  Only the bead count varies. A plectonemic superhelix radius is ~5-9 nm, so the sweep")
    report(f"  brackets it rather than chasing the finest possible mesh.")
    report(f"\n  {'N':>5}{'bp/bead':>10}{'nm/bead':>10}{'<Wr>':>9}{'Wr/dLk':>9}{'<Tw>':>9}"
           f"{'E_bend/kT':>12}{'bend/|Wr|':>11}{'Rg nm':>8}{'acc':>6}{'s':>7}")
    rows = []
    for n in NS:
        t1 = time.time()
        r = SC.run(dlk, SC.C_TWIST, sweeps=SWEEPS, burn=BURN, n=n)
        ch = r["final"]
        eb = ch.e_bend() / KT
        frac = r["wr_mean"] / dlk
        per = eb / max(abs(r["wr_mean"]), 1e-9)
        rows.append({"n": n, "bp_bead": BP / n, "nm_bead": BP / n * RISE * 1e9,
                     "wr": r["wr_mean"], "frac": frac, "tw": dlk - r["wr_mean"],
                     "e_bend_kT": eb, "bend_per_wr": per, "rg_nm": r["rg"] * 1e9,
                     "acc": r["acc"], "sec": time.time() - t1})
        report(f"  {n:>5}{BP/n:>10.1f}{BP/n*RISE*1e9:>10.1f}{r['wr_mean']:>9.3f}{frac:>9.3f}"
               f"{dlk-r['wr_mean']:>9.3f}{eb:>12.1f}{per:>11.2f}{r['rg']*1e9:>8.1f}"
               f"{r['acc']:>6.2f}{time.time()-t1:>7.0f}")

    f0, f1 = rows[0]["frac"], rows[-1]["frac"]
    b0, b1 = rows[0]["bend_per_wr"], rows[-1]["bend_per_wr"]
    report("\n  READING")
    report(f"    Wr/dLk        {f0:.3f} at {rows[0]['nm_bead']:.1f} nm  ->  {f1:.3f} at "
           f"{rows[-1]['nm_bead']:.1f} nm   change {f1-f0:+.3f}")
    report(f"    bending cost per unit writhe   {b0:.2f} -> {b1:.2f} kT   ratio {b1/max(b0,1e-9):.2f}x")
    falls = (f1 < f0 - 0.03)
    costlier = (b1 > b0 * 1.2)
    near = abs(f1 - 0.725) < 0.15
    if falls and costlier and near:
        report("  THE EXPLANATION HOLDS, and for the stated reason. Shrinking the segment below the")
        report("  superhelical radius makes writhe genuinely expensive in bending, and the partition moves")
        report("  to where real DNA sits. The physics was right and the mesh was wrong.")
    elif falls and costlier:
        report("  RESOLUTION IS PART OF IT. The partition moves the right way and the bending cost rises")
        report("  as claimed, but it does not reach the measured value, so a second effect is present and")
        report("  extrapolating this trend to 0.72 would be wishful.")
    elif not falls:
        report("  MY EXPLANATION WAS WRONG. The partition does not care about resolution, so the overshoot")
        report("  lives in the model -- excluded-volume diameter and the bending discretisation are the")
        report("  first two places to look. Nothing should be layered on top until it is found.")
    else:
        report("  Mixed: the partition moves without the bending cost rising, which is not the mechanism")
        report("  claimed. Whatever is changing with resolution, it is not what I said it was.")
    report(f"\n  COST NOTE: {rows[-1]['sec']/max(rows[0]['sec'],1e-9):.0f}x from N={NS[0]} to N={NS[-1]}, "
           f"against N^3 = {(NS[-1]/NS[0])**3:.0f}x predicted.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "chromatin_resolution", "bp": BP, "sigma": SIGMA, "dlk": dlk,
               "sweeps": SWEEPS, "rows": rows,
               "gates": {"falls": bool(falls), "bending_costlier": bool(costlier),
                         "reaches_literature": bool(near)}, "log": log},
              open(OUT / "chromatin_resolution.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'chromatin_resolution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
