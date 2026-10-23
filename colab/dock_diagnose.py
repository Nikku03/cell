"""IS THE SHAPE SCORE BROKEN, OR MERELY WEAK?  The check that has to come before any claim about docking.

WHAT PROVOKED THIS.  `dock_fft` found that with the EXACTLY CORRECT ligand orientation handed to it for free,
the shape score still places the ligand a median 21.3 A from the crystal position in 5 of 6 complexes. Rotation
sampling was not the constraint (coverage 0.80). So either the score is genuinely weak, or my grid is wrong --
and those two have completely different consequences, so guessing between them is not allowed.

THE DISCRIMINATING FACT.  In a crystal structure the two chains DO NOT interpenetrate. So at the native pose:

    clash must be ~0        if it is large, the dilation has made the bodies too fat and the score is
                            PENALISING THE TRUE ANSWER. That is my bug, and every dock_fft number is void.
    interlock must be high  the native interface is, by construction, a well-packed one.

If clash is ~0 and interlock is high and the native STILL ranks poorly, then the score is not broken: there are
simply many placements on a protein surface with more raw contact area than the real site has, and shape alone
cannot pick between them. That is a real and known property, not a coding error.

THE TUNING SWEEP, so the conclusion cannot be dismissed as a badly chosen constant.  score = interlock - w*clash
is LINEAR in w, and both fields come from the same two transforms, so every w is free once the FFTs are done.
The native percentile is reported across w from 0 (pure contact area, no clash penalty at all) to 100. If the
native pose ranks poorly at EVERY w, no choice of the penalty weight rescues it and the limitation is the
representation, not the parameter.

CONTROLS:
    random shift    what an arbitrary placement scores, so "high interlock" has a scale.
    w = 0           pure contact-area maximisation, the degenerate case. If w=0 is as good as the tuned w,
                    the clash term is decorative.

PREDECLARED, before any number:
    native clash large                        -> MY BUG. Grid too fat. dock_fft is void and gets re-run.
    native clash ~0, native percentile poor
      at every w                              -> the score is WEAK, not broken. Shape complementarity cannot
                                                 rank, and the fix is a better energy on top of FFT sampling,
                                                 not a better FFT.
    native percentile good at some w          -> the weight was simply wrong. Re-run dock_fft with it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import dock_fft as DF
import flex_physics as fp

OUT = A.OUT
GRID, SPACING = DF.GRID, DF.SPACING
WEIGHTS = (0.0, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0)


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("IS THE SHAPE SCORE BROKEN OR MERELY WEAK? -- native-pose components + clash-weight sweep")
    report("=" * 100)
    report("  PREDECLARED: native clash large => MY BUG, grid too fat, dock_fft is void.")
    report("               native clash ~0 and percentile poor at EVERY w => score is weak, not broken.")
    report("               percentile good at some w => the weight was wrong; re-run dock_fft with it.")

    pdbs = [r["pdb"] for r in json.load(open(OUT / "dock_fft.json"))["complexes"]]
    rng = np.random.default_rng(0)
    rows = []
    for pdb in pdbs:
        t = fp.table(pdb)
        chains = sorted(set(t["ch"].tolist()))
        ca, cb = chains[0], chains[1]
        R = t["co"][t["ch"] == ca]
        L = t["co"][t["ch"] == cb]
        centre = R.mean(0)
        gR = DF.gridify(R, centre)
        lcen = L.mean(0)
        gL = DF.gridify(L - lcen + centre, centre)          # NATIVE orientation, centred
        if gR is None or gL is None:
            continue
        _sR, surf_R, core_R = gR
        solid_L = gL[0]

        FL = np.conj(np.fft.rfftn(solid_L.astype(np.float32)))
        inter = np.fft.irfftn(np.fft.rfftn(surf_R.astype(np.float32)) * FL, surf_R.shape)
        clash = np.fft.irfftn(np.fft.rfftn(core_R.astype(np.float32)) * FL, surf_R.shape)

        # the native translation, as a correlation index (positive shift, wrapped)
        k = np.round((lcen - centre) / SPACING).astype(int) % GRID
        kn = tuple(int(x) for x in k)
        # a random in-contact-range shift, as the scale reference
        kr = tuple(int(x) for x in (rng.integers(-25, 26, 3) % GRID))

        # how much true cross-interface contact exists, independent of any grid, for scale
        d = np.linalg.norm(R[:, None, :] - L[None, :, :], axis=2)
        n_contacts = int((d < 5.0).sum())

        sweep = {}
        for w in WEIGHTS:
            sc = inter - w * clash
            nat = float(sc[kn])
            sweep[str(w)] = {"native_score": nat,
                             "pct": float((sc < nat).mean()),
                             "n_better": int((sc > nat).sum()),
                             "max": float(sc.max())}
        best_w = max(WEIGHTS, key=lambda w: sweep[str(w)]["pct"])
        rows.append({"pdb": pdb, "chains": [ca, cb],
                     "native_interlock": float(inter[kn]), "native_clash": float(clash[kn]),
                     "rand_interlock": float(inter[kr]), "rand_clash": float(clash[kr]),
                     "max_interlock": float(inter.max()), "max_clash_at_argmax_inter":
                         float(clash[np.unravel_index(np.argmax(inter), inter.shape)]),
                     "true_atom_contacts_5A": n_contacts,
                     "sweep": sweep, "best_w": best_w, "best_pct": sweep[str(best_w)]["pct"]})
        report(f"    {pdb} {ca}/{cb}: native interlock {inter[kn]:8.0f} clash {clash[kn]:8.0f} | "
               f"random-shift interlock {inter[kr]:6.0f} clash {clash[kr]:8.0f} | "
               f"max interlock anywhere {inter.max():8.0f} | true 5A atom contacts {n_contacts}")

    if not rows:
        report("\n  nothing to diagnose")
        return 1

    report("\n  IS THE NATIVE POSE BEING CHARGED A CLASH PENALTY?")
    report(f"    {'pdb':<6} {'native clash':>13} {'native interlock':>17} {'clash/interlock':>16}")
    for r in rows:
        ratio = r["native_clash"] / max(r["native_interlock"], 1e-9)
        report(f"    {r['pdb']:<6} {r['native_clash']:>13.0f} {r['native_interlock']:>17.0f} {ratio:>16.3f}")
    med_ratio = float(np.median([r["native_clash"] / max(r["native_interlock"], 1e-9) for r in rows]))

    report("\n  CLASH-WEIGHT SWEEP -- percentile of the native pose among all 884,736 placements")
    hdr = "    " + f"{'pdb':<6}" + "".join(f"{('w=' + str(int(w))):>9}" for w in WEIGHTS)
    report(hdr)
    for r in rows:
        report("    " + f"{r['pdb']:<6}" + "".join(f"{r['sweep'][str(w)]['pct']:>9.4f}" for w in WEIGHTS))
    med_by_w = {w: float(np.median([r["sweep"][str(w)]["pct"] for r in rows])) for w in WEIGHTS}
    report("    " + f"{'MEDIAN':<6}" + "".join(f"{med_by_w[w]:>9.4f}" for w in WEIGHTS))
    bw = max(WEIGHTS, key=lambda w: med_by_w[w])
    report(f"\n    best weight by median percentile: w={bw:g} at {med_by_w[bw]:.4f}   "
           f"(w=0, no clash penalty at all: {med_by_w[0.0]:.4f})")
    report(f"    placements scoring BETTER than the native at w={bw:g}: "
           f"{[r['sweep'][str(bw)]['n_better'] for r in rows]}")

    report("\n  READING")
    if med_ratio > 0.5:
        report(f"  The native pose is charged a clash of {med_ratio:.2f}x its own interlock. In a crystal the")
        report("  chains do not interpenetrate, so this is MY GRID, not the concept: dilation has fattened the")
        report("  bodies past the real contact distance and the score penalises the true answer. dock_fft is")
        report("  VOID and must be re-run with a thinner representation.")
    elif med_by_w[bw] > 0.9999:
        report(f"  At w={bw:g} the native pose sits in the top 0.01% of placements. The weight was simply wrong")
        report("  in dock_fft; the score can rank after all, and dock_fft gets re-run with this weight.")
    else:
        report(f"  Native clash is {med_ratio:.3f}x interlock -- the true pose is NOT being penalised, so the")
        report("  grid is fine. And the native percentile never exceeds "
               f"{med_by_w[bw]:.4f} at ANY weight from 0 to 100,")
        report(f"  with {int(np.median([r['sweep'][str(bw)]['n_better'] for r in rows])):,} placements scoring")
        report("  higher at the best one. So the score is WEAK, not broken, and no tuning of the clash penalty")
        report("  rescues it: a protein surface simply has many spots with more raw contact area than the real")
        report("  binding site. FFT sampling is sound; the ranking has to come from an energy, not a shape.")

    json.dump({"test": "dock_diagnose", "weights": list(WEIGHTS), "complexes": rows,
               "median_clash_over_interlock": med_ratio,
               "median_pct_by_weight": {str(w): med_by_w[w] for w in WEIGHTS},
               "best_weight": bw, "log": log}, open(OUT / "dock_diagnose.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_diagnose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
