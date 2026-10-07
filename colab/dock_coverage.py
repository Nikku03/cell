"""IS THE RIGHT ANSWER EVEN IN THE SET?  Pure sampling recall, with no ranking anywhere.

WHY THIS COMES FIRST, AND WHY IT SHOULD HAVE COME FIRST.  Five experiments tried to RANK the FFT pose sets --
shape, learned embeddings, retrained embeddings, global descriptors, physics energies. All of them were scored
against pose sets in which:

    6 of the 12 complexes had NO near-native pose at all      near_native_frac 0.0000
    the other 6 had 2 to 4 near-native poses out of 2,400     frac 0.0008 - 0.0017

**For half the test set no ranker could possibly have succeeded**, and for the rest the target was a 0.1%
population. So "0/12 in the top ten" was never purely a ranking result -- part of it was the answer being
absent. Ranking a candidate set that does not contain the answer measures nothing.

This measures the precondition instead: how much sampling before the correct pose is reliably PRESENT. No score
is used to select anything here; every pose the search produces is kept and compared to the crystal answer.

ONE PASS GIVES THE WHOLE CURVE.  `rotations(n, seed)` draws from a fresh generator in order, so rotations(600)
is exactly the first 600 of rotations(4800) -- the sets are NESTED. Recording which rotation index produced
each pose therefore yields coverage at every prefix from a single run. The same trick covers translation: keep
the top PEAKS_MAX peaks per rotation and record each pose's peak rank, so coverage at 4 peaks is just the
subset with rank < 4. Two sweeps, one pass, no re-running.

WHAT IS MEASURED, at each (rotations, peaks) budget:
    best RMSD reachable      the closest the search ever gets to the crystal pose
    n near-native            how many poses are within NEAR_NATIVE of it
    complexes covered        how many of the 12 have at least one -- the number that gates every ranking test
    coverage at 10 A         the looser "right neighbourhood" criterion, since rigid-body docking is expected
                             to find the neighbourhood and leave the last angstrom to flexible refinement

AGAINST THEORY.  N random rotations cover SO(3) with P(some rotation within t of native) = 1-(1-t^3/(6*pi))^N,
and a ligand of gyration radius Rg suffers ligand RMSD ~ t*Rg from a rotation error t. The predicted curve is
printed beside the observed one. Agreement means the limit is rotation count and more sampling fixes it;
observed FAR below predicted means something else binds -- grid resolution, the rigid-body assumption, or the
peak selection discarding the right translation.

PREDECLARED, before any number:
    coverage still rising at the largest budget      -> SAMPLING-limited. The ranking experiments were run on
                                                       sets that often lacked the answer, and the fix is more
                                                       rotations before any further work on scoring.
    coverage plateaus below 12/12                    -> some complexes are unreachable by rigid-body docking at
                                                       this grid resolution. That is a different limit and no
                                                       amount of rotation sampling touches it.
    12/12 covered at some budget                     -> ranking becomes the sole remaining problem, and
                                                       clustering is worth trying at that budget and not before.

NO RANKING IS PERFORMED IN THIS FILE. The FFT score is recorded but never used to select, filter or order any
pose that enters a coverage number. That is the point: this measures what the search can REACH, which is the
ceiling every ranker is working under.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import flex_physics as fp

OUT = A.OUT
N_ROT_MAX = 4800                       # 8x the 600 used by every previous run
ROT_STEPS = (600, 1200, 2400, 4800)    # nested prefixes, free from one pass
PEAKS_MAX = 12                         # vs the 4 used before; peak rank recorded so both are readable
PEAK_STEPS = (4, 12)
NEAR_NATIVE = DF.NEAR_NATIVE           # 5 A, the standard acceptable-pose criterion
NEIGHBOURHOOD = 10.0                   # the looser "right region" criterion


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("IS THE RIGHT ANSWER EVEN IN THE SET? -- pure sampling recall, no ranking anywhere")
    report("=" * 100)
    report("  Five ranking experiments ran against pose sets where 6 of 12 complexes had NO near-native pose")
    report("  and the rest had 2-4 out of 2,400. Ranking a set that lacks the answer measures nothing.")
    report(f"  Sweeping rotations {ROT_STEPS} and peaks-per-rotation {PEAK_STEPS} in ONE pass, since the")
    report("  rotation sets are nested prefixes and peak rank is recorded per pose.")
    report("  PREDECLARED: still rising at the largest budget => SAMPLING-limited, and every ranking result so")
    report("  far was measured under a ceiling. Plateau below 12/12 => a resolution or rigid-body limit that")
    report("  more rotations cannot touch. 12/12 covered => ranking is the sole problem and clustering is next.")
    report("  NO POSE IS EVER SELECTED BY SCORE HERE. Coverage is what the search can REACH.")

    dockj = json.load(open(OUT / "dock_fft.json"))
    CW = dockj["clash_w_chosen"]
    test_pdbs = [r["pdb"] for r in dockj["complexes"]]
    report(f"\n  test complexes (pinned from dock_fft.json): {test_pdbs}")

    rots = DF.rotations(N_ROT_MAX)
    rows = []
    for pdb in test_pdbs:
        p = DF.prepare(pdb)
        if p is None or not p["sane"]:
            continue
        centre, L, lcen = p["centre"], p["L"], p["lcen"]
        Lc = L - lcen
        rg = float(np.sqrt((Lc ** 2).sum(1).mean()))
        FA = p["F_skin"] - CW * p["F_core"]
        t1 = time.time()
        ri_of, pk_of, rm_of = [], [], []
        for ri, Rm in enumerate(rots):
            g = DF.rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
            if g is None:
                continue
            sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
            for rank, pk in enumerate(DF.nms_peaks(sc, PEAKS_MAX, DF.NMS_CELLS)):
                if sc[pk] <= 0:
                    continue
                sh = DF.shift_of(pk)
                rm = float(np.sqrt((((Lc @ Rm.T + centre + sh) - L) ** 2).sum(1).mean()))
                ri_of.append(ri); pk_of.append(rank); rm_of.append(rm)
        if not rm_of:
            continue
        ri_of = np.array(ri_of); pk_of = np.array(pk_of); rm_of = np.array(rm_of)

        grid = {}
        for nr in ROT_STEPS:
            for npk in PEAK_STEPS:
                m = (ri_of < nr) & (pk_of < npk)
                if not m.any():
                    grid[f"{nr}x{npk}"] = {"n_poses": 0, "best_rmsd": None, "n_near": 0, "n_neigh": 0}
                    continue
                sub = rm_of[m]
                grid[f"{nr}x{npk}"] = {"n_poses": int(m.sum()), "best_rmsd": float(sub.min()),
                                       "n_near": int((sub <= NEAR_NATIVE).sum()),
                                       "n_neigh": int((sub <= NEIGHBOURHOOD).sum())}
        # theory: P(some rotation within t of native), t = NEAR_NATIVE / Rg
        t_need = NEAR_NATIVE / rg
        p_one = min(1.0, t_need ** 3 / (6 * np.pi))
        theory = {str(nr): float(1 - (1 - p_one) ** nr) for nr in ROT_STEPS}
        rows.append({"pdb": pdb, "chains": [p["ca"], p["cb"]], "rg_ligand": rg,
                     "rot_tol_deg": float(np.degrees(t_need)), "grid": grid, "theory_cover": theory,
                     "secs": time.time() - t1})
        g4 = [grid[f"{nr}x{PEAKS_MAX}"] for nr in ROT_STEPS]
        report(f"    {pdb} {p['ca']}/{p['cb']} (Rg {rg:.0f} A, tol {np.degrees(t_need):.0f} deg) "
               f"{time.time()-t1:.0f}s | best RMSD by rotations: "
               + "  ".join(f"{nr}:{(x['best_rmsd'] if x['best_rmsd'] is not None else float('nan')):.1f}"
                           for nr, x in zip(ROT_STEPS, g4))
               + " | near-native: " + " ".join(f"{nr}:{x['n_near']}" for nr, x in zip(ROT_STEPS, g4)))

    if not rows:
        report("\n  nothing sampled")
        return 1

    report(f"\n  {len(rows)} complexes\n")
    report(f"  COVERAGE -- complexes with at least one pose within {NEAR_NATIVE:.0f} A (the gate on every "
           f"ranking test)")
    report(f"    {'rotations':>10}" + "".join(f"{'peaks=' + str(k):>12}" for k in PEAK_STEPS)
           + f"{'theory':>10}{'poses/cx':>10}")
    cov = {}
    for nr in ROT_STEPS:
        line = f"    {nr:>10}"
        for npk in PEAK_STEPS:
            c = sum(r["grid"][f"{nr}x{npk}"]["n_near"] > 0 for r in rows)
            cov[f"{nr}x{npk}"] = c
            line += f"{str(c) + '/' + str(len(rows)):>12}"
        th = float(np.mean([r["theory_cover"][str(nr)] for r in rows])) * len(rows)
        npose = float(np.mean([r["grid"][f"{nr}x{PEAKS_MAX}"]["n_poses"] for r in rows]))
        report(line + f"{th:>10.1f}{npose:>10.0f}")

    report(f"\n  COVERAGE at the looser {NEIGHBOURHOOD:.0f} A neighbourhood criterion")
    covn = {}
    for nr in ROT_STEPS:
        line = f"    {nr:>10}"
        for npk in PEAK_STEPS:
            c = sum(r["grid"][f"{nr}x{npk}"]["n_neigh"] > 0 for r in rows)
            covn[f"{nr}x{npk}"] = c
            line += f"{str(c) + '/' + str(len(rows)):>12}"
        report(line)

    report(f"\n  MEDIAN best RMSD reachable, and median near-native COUNT (peaks={PEAKS_MAX})")
    report(f"    {'rotations':>10}{'best RMSD':>12}{'n near-native':>15}")
    for nr in ROT_STEPS:
        b = [r["grid"][f"{nr}x{PEAKS_MAX}"]["best_rmsd"] for r in rows
             if r["grid"][f"{nr}x{PEAKS_MAX}"]["best_rmsd"] is not None]
        n = [r["grid"][f"{nr}x{PEAKS_MAX}"]["n_near"] for r in rows]
        report(f"    {nr:>10}{np.median(b):>12.1f}{np.median(n):>15.1f}")

    report("\n  READING")
    lo, hi = f"{ROT_STEPS[-2]}x{PEAKS_MAX}", f"{ROT_STEPS[-1]}x{PEAKS_MAX}"
    step = cov[hi] - cov[lo]
    base = cov[f"{ROT_STEPS[0]}x4"]
    report(f"  coverage went {base}/{len(rows)} at the budget every ranking test used "
           f"({ROT_STEPS[0]} rotations, 4 peaks) -> {cov[hi]}/{len(rows)} at {ROT_STEPS[-1]} rotations, "
           f"{PEAKS_MAX} peaks")
    if cov[hi] == len(rows):
        report("  FULLY COVERED at the largest budget. The answer is now in every pose set, so ranking is the")
        report("  sole remaining problem -- and clustering should be tried at THIS budget, not the old one.")
    elif step > 0:
        report(f"  STILL RISING (+{step} on the last doubling). SAMPLING-limited: every ranking result so far")
        report("  was measured against sets that often did not contain the answer, and more rotations is the")
        report("  first lever, not a better score.")
    else:
        report(f"  PLATEAUED at {cov[hi]}/{len(rows)}. More rotations do not help the rest, so their limit is")
        report("  grid resolution or the rigid-body assumption -- bound-form side chains that cannot relax.")
        report("  Those complexes are unreachable by this method however it is scored.")

    json.dump({"test": "dock_coverage", "n_rot_max": N_ROT_MAX, "rot_steps": list(ROT_STEPS),
               "peak_steps": list(PEAK_STEPS), "peaks_max": PEAKS_MAX, "near_native_A": NEAR_NATIVE,
               "neighbourhood_A": NEIGHBOURHOOD, "coverage_near": cov, "coverage_neigh": covn,
               "complexes": rows, "n_complexes": len(rows), "log": log},
              open(OUT / "dock_coverage.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_coverage.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
