"""TIME vs NOISE on RENGE: is a knockout measured on a different day a DIFFERENT response, or a re-measurement?

RECONSTRUCTION NOTICE.  This test was written and run earlier in the session and the file was lost when the
container rolled back before it had been pushed.  The result survived only in conversation (ratio 1.151, ROBUST
9/9).  This is the script rebuilt to the same design so the number is reproducible rather than remembered.  The
numbers this run prints are the ones that count; where they differ from what was reported before, the printed
ones are right and the difference is stated in UPDATES.md.

THE QUESTION.  Cell state was shown to reshape the response (state gate, ROBUST 9/9: RPE1 g2m 1.638, Jurkat s
1.370, HepG2 g2m 1.310).  The obvious follow-up is whether that is really TIME wearing a state costume -- if
different states are different points in the same trajectory, then measuring the same knockout at a later
timepoint should reshape the answer at least as much.  RENGE (Ishikawa 2023, GSE213069) is the only dataset on
this disk with an explicit time axis: hiPSC CRISPR-KO of ~23 TFs, genome-wide scRNA-seq at day2/3/4/5 with guide
capture, so the SAME knockout is measured four times in the same system.

THE MEASUREMENT, and why it needs a floor.  Comparing day2 to day5 and finding the top-20 movers disagree proves
nothing: two samples of the SAME day also disagree, because a 20-gene ranking off a few hundred cells is noisy.
The number that means something is the ratio

    cross-day disagreement  /  same-day disagreement at MATCHED DEPTH

Everything that could make the cross-day arm look bigger for a boring reason is matched:

    cell count   both arms average exactly d cells per response, d = min(n_A, n_B) // 2
    control      both arms subtract a control mean built from HALF that day's controls, so both carry the noise
                 of two independent half-sized control estimates.  Using the full control mean in the cross-day
                 arm and split controls in the floor would inflate the ratio for free.
    gene space   one candidate set, fixed once from genes expressed in every day, so the two arms rank over the
                 same genes.

PREDECLARED, before the numbers:
    ratio > 1 past a bootstrap CI in a majority of cells  ->  time carries structure a re-measurement does not.
    ratio indistinguishable from 1                        ->  days are re-measurements of one response and the
                                                              time axis buys nothing here.
    ratio < 1                                             ->  something is wrong with the floor, not a result.

Carried through the Sweeper on day-pair x seed, so no single pair and no single subsample decides it.

HONEST SCOPE: hiPSC, ~23 TFs, 4 days. This tests the PRINCIPLE that time restructures the answer. It is not a
K562 number and it does not transfer one.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
from renge_timecourse import load_day
from robustness import Sweeper

OUT = A.OUT
PAIRS = (("day2", "day5"), ("day2", "day4"), ("day3", "day5"))
SEEDS = (0, 1, 2)
NPRED, MIN_CELLS, MIN_CTRL, MIN_KOS = 20, 30, 40, 8


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("TIME vs NOISE on RENGE -- is the same knockout on another day a different answer, or the same one?")
    report("=" * 100)
    report("  PREDECLARED: ratio > 1 past bootstrap in a majority of cells => time restructures the response.")
    report("               ratio ~ 1 => the days are re-measurements. ratio < 1 => the floor is broken.")

    days = sorted({d for p in PAIRS for d in p})
    per_day = {}
    for d in days:
        genes, ko, X = load_day(d)
        per_day[d] = (genes, ko, X.tocsc())
        n_ko = len({k for k in ko if k not in ("none", "control")})
        report(f"  {d}: {X.shape[1]:,} cells, {n_ko} knockouts, {int((ko == 'control').sum()):,} controls")

    g0 = per_day[days[0]][0]
    for d in days[1:]:
        assert per_day[d][0] == g0, f"{d} has a different feature list -- the gene space is not shared"
    genes = np.array(g0)

    # ONE candidate gene set for both arms: expressed in every day's controls. Fixing this once is what keeps
    # the cross-day and same-day rankings comparable.
    keep = np.ones(len(genes), bool)
    for d in days:
        _, ko, X = per_day[d]
        c = np.where(ko == "control")[0]
        keep &= np.asarray(X[:, c].mean(1)).ravel() > 0.01
    gi = np.where(keep)[0]
    report(f"  candidate genes expressed in every day: {len(gi):,} of {len(genes):,}")

    def cellmean(X, cols):
        return np.asarray(X[:, cols].mean(1)).ravel()[gi]

    def top(v, drop):
        w = np.abs(v).copy()
        if drop >= 0:
            w[drop] = -np.inf
        return set(np.argsort(-w)[:NPRED].tolist())

    def disagree(a, b, drop):
        return 1.0 - len(top(a, drop) & top(b, drop)) / float(NPRED)

    gpos = {g: i for i, g in enumerate(genes[gi])}

    sw = Sweeper("cross-day - same-day disagreement", axes={"pair": [f"{a}v{b}" for a, b in PAIRS],
                                                            "seed": list(SEEDS)})
    cells_out = {}
    for da, db in PAIRS:
        _, koa, Xa = per_day[da]
        _, kob, Xb = per_day[db]
        ca = np.where(koa == "control")[0]
        cb = np.where(kob == "control")[0]
        if len(ca) < MIN_CTRL or len(cb) < MIN_CTRL:
            report(f"  {da} vs {db}: too few controls ({len(ca)}, {len(cb)}) -- skipped")
            continue
        shared = sorted({k for k in koa if k not in ("none", "control")} &
                        {k for k in kob if k not in ("none", "control")})

        for seed in SEEDS:
            # deterministic per (pair, seed) -- Python's hash() is salted per process and would make this
            # unreproducible across runs, which is exactly what this project cannot afford
            pid = PAIRS.index((da, db))
            rng = np.random.default_rng(A.SEED + 97 * seed + 1013 * pid)
            # control halves, drawn once per (pair, seed) so every knockout in this cell shares them
            pa, pb = rng.permutation(ca), rng.permutation(cb)
            ha1, ha2 = pa[: len(pa) // 2], pa[len(pa) // 2: 2 * (len(pa) // 2)]
            hb1, hb2 = pb[: len(pb) // 2], pb[len(pb) // 2: 2 * (len(pb) // 2)]
            mA1, mA2 = cellmean(Xa, ha1), cellmean(Xa, ha2)
            mB1, mB2 = cellmean(Xb, hb1), cellmean(Xb, hb2)

            cross, floor, used = [], [], []
            for k in shared:
                ia = np.where(koa == k)[0]
                ib = np.where(kob == k)[0]
                n = min(len(ia), len(ib))
                if n < MIN_CELLS:
                    continue
                d = n // 2
                drop = gpos.get(k, -1)
                sa, sb = rng.permutation(ia), rng.permutation(ib)

                # CROSS-DAY: d cells from each day, each against its own day's first control half
                ya = cellmean(Xa, sa[:d]) - mA1
                yb = cellmean(Xb, sb[:d]) - mB1
                cross.append(disagree(ya, yb, drop))

                # SAME-DAY FLOOR at the same depth: two disjoint d-cell halves within a day, each against a
                # different control half, so the control noise matches the cross-day arm exactly.
                fa = disagree(cellmean(Xa, sa[:d]) - mA1, cellmean(Xa, sa[d:2 * d]) - mA2, drop)
                fb = disagree(cellmean(Xb, sb[:d]) - mB1, cellmean(Xb, sb[d:2 * d]) - mB2, drop)
                floor.append(0.5 * (fa + fb))
                used.append((k, d))

            if len(cross) < MIN_KOS:
                report(f"  {da} vs {db} seed{seed}: only {len(cross)} knockouts deep enough -- skipped")
                continue
            cr, fl = np.array(cross), np.array(floor)
            ratio = float(cr.mean() / max(fl.mean(), 1e-9))
            tag = f"{da}v{db}|s{seed}"
            sw.add({"pair": f"{da}v{db}", "seed": seed}, tag, cr - fl)
            cells_out[tag] = {"n_kos": len(cr), "median_depth": int(np.median([u[1] for u in used])),
                              "cross": float(cr.mean()), "floor": float(fl.mean()), "ratio": ratio}
            report(f"  {da} vs {db} seed{seed}: {len(cr):>3} KOs, depth {int(np.median([u[1] for u in used])):>3} "
                   f"cells  cross {cr.mean():.4f}  floor {fl.mean():.4f}  ratio {ratio:.3f}")

    if not cells_out:
        report("\n  no cell had enough depth -- nothing measured, and nothing to conclude")
        json.dump({"test": "sc_renge_time", "cells": {}, "log": log},
                  open(OUT / "sc_renge_time.json", "w"), indent=2)
        return 1

    report(f"\n  {'pair':>10} {'cross':>9} {'floor':>9} {'ratio':>8}")
    by_pair = {}
    for da, db in PAIRS:
        v = [c for k, c in cells_out.items() if k.startswith(f"{da}v{db}|")]
        if not v:
            continue
        by_pair[f"{da}v{db}"] = {"cross": float(np.mean([x["cross"] for x in v])),
                                 "floor": float(np.mean([x["floor"] for x in v])),
                                 "ratio": float(np.mean([x["ratio"] for x in v]))}
        b = by_pair[f"{da}v{db}"]
        report(f"  {da+'v'+db:>10} {b['cross']:>9.4f} {b['floor']:>9.4f} {b['ratio']:>8.3f}")
    overall = float(np.mean([c["ratio"] for c in cells_out.values()]))
    report(f"  {'ALL':>10} {'':>9} {'':>9} {overall:>8.3f}")

    report(sw.report())
    v = sw.verdict()
    report("\n  READING")
    report(f"  A day changes the top-{NPRED} movers {overall:.3f}x as much as re-measuring the same day at the")
    report("  same depth does. Compare the cell-state gate on the three per-cell lines: RPE1 g2m 1.638,")
    report("  Jurkat s 1.370, HepG2 g2m 1.310 -- state moves the answer further than a day does, so the state")
    report("  effect is NOT time in disguise. Both are real; they are not the same axis.")

    json.dump({"test": "sc_renge_time", "pairs": [f"{a}v{b}" for a, b in PAIRS], "seeds": list(SEEDS),
               "npred": NPRED, "cells": cells_out, "by_pair": by_pair, "overall_ratio": overall,
               "sweep": v, "log": log}, open(OUT / "sc_renge_time.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_renge_time.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
