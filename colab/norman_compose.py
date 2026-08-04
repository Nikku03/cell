"""DO PERTURBATION EFFECTS COMPOSE?  The measurement, not the inference.

THE QUESTION THIS ANSWERS THAT NOTHING ELSE HERE CAN.  Every compositional claim in this project so far is
indirect: LOCO asks whether a model trained without a functional class can predict it, adrn_basis_compose asks
whether a class's responses lie in the span of other classes' responses. Both infer composition from a model.

Norman 2019 measures it. For 131 gene pairs the dataset contains A alone, B alone, AND A+B, in the same cells,
same batch, same readout. So the question becomes arithmetic: **given the measured response to A and the measured
response to B, does the measured response to A+B follow?** No model, no annotation, no channels.

If the answer is no -- if the joint response is not predictable from its parts even with both parts MEASURED --
then the compositional model this project has been circling does not exist, and no architecture rescues it. If
the answer is yes, composition is real and the remaining problem is getting the parts, which is a much easier
problem than inventing them.

ARMS.  Each is a rule for combining two measured single responses into a predicted double response.

    additive        r(A) + r(B).  The null model of no interaction, and what "effects compose" literally means.
    dominant        whichever single has the larger response magnitude. Composition by masking rather than sum.
    single_A        one parent alone, chosen by name order. The control for "the double just looks like a single."
    best_single_ORACLE  whichever parent scores better AGAINST THE ANSWER. An oracle, reported as a denominator:
                    no rule that must choose blind can beat it.

CONTROLS, and the first one is the one that decides it.

    scrambled       predict A+B from the singles of a DIFFERENT randomly chosen pair. Same rule, same statistics,
                    same number of genes, wrong biology. If `additive` does not beat this, then the arithmetic is
                    carrying nothing and any apparent success is the shape of the data rather than composition.
    mean_double     the average response across all doubles. The floor -- what you score knowing nothing.
    FLOOR           half0 vs half1 of the SAME double. How well the measurement agrees with itself at this depth.
                    Every arm is reported against it, because a rule cannot beat the measurement's own noise and
                    a number without this is uninterpretable.

PREDECLARED, before any number is seen:

    additive > scrambled, resolved, in a majority of cells   -> EFFECTS COMPOSE. The parts determine the whole.
    additive ~ scrambled                                     -> they do not, and the compositional model is dead
                                                                on measured data, not just on modelled data.
    additive < dominant, resolved                            -> composition happens but by masking, not addition;
                                                                the right primitive is a max, not a sum.

Judged on robustness.Sweeper over prediction depth x seed. Scored as precision@N on the double's own top movers,
consistent with every other number in this project, with Pearson r reported alongside because the two fail
differently.

SCOPE: Norman 2019 is CRISPRa -- ACTIVATION. Every other number here is CRISPRi knockdown or CRISPR KO. This
measures whether PERTURBATION EFFECTS compose. Carrying the conclusion to knockouts is an inference and is
labelled as one wherever it is cited.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
NPREDS = (10, 20, 50)
SEEDS = (0, 1, 2)
# MIN_CELLS is a real second sweep axis, not decoration: it changes WHICH pairs are scored, and the shallow ones
# are exactly where a composition rule could be flattered by a noisy target. The first version of this file swept
# npred x seed instead, and the harness caught it -- `additive`, `dominant` and the floor do not depend on the
# seed at all (only `scrambled` draws a random pairing), so 6 of 9 cells were byte-identical duplicates and the
# "ROBUST 9/9" on additive-dominant was really 3 observations. Seeds are now averaged WITHIN a cell, where they
# belong, and the grid varies two things that actually move the numbers.
MIN_CELLS_SWEEP = (50, 150)


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("DO PERTURBATION EFFECTS COMPOSE?  Norman 2019 doubles, measured parts and measured whole")
    report("=" * 100)
    report("  PREDECLARED: additive > scrambled => effects compose. additive ~ scrambled => they do not.")
    report("               additive < dominant => composition by masking, not addition.")
    report("  SCOPE: CRISPRa activation, not knockout. Transfer to knockouts is an inference.")

    z = np.load(SP / "norman_pseudobulk_halves.npz", allow_pickle=True)
    groups = [str(g) for g in z["groups"]]
    genes = np.array([str(g) for g in z["genes"]])
    Pf = z["profile"]
    Ph = z["profile_half"]
    ncell = z["ncell"]
    nch = z["ncell_half"]

    def parts(g):
        pair = g.split("__")[0].split("_")
        return [p for p in pair if not p.startswith("NegCtrl")]

    ctrl_rows = [i for i, g in enumerate(groups) if not parts(g)]
    assert ctrl_rows, "no control programs found -- the guide-identity parsing is wrong"
    ctrl = Pf[ctrl_rows].mean(0)
    ctrl_h = Ph[ctrl_rows].mean(0)          # (2, genes) -- one control per half, so the floor is matched
    report(f"  {len(groups)} programs, {len(genes):,} genes, {len(ctrl_rows)} control programs")

    def cohort(min_cells):
        single, double = {}, {}
        for i, g in enumerate(groups):
            p = parts(g)
            if len(p) == 1 and ncell[i] >= min_cells:
                single[p[0]] = i
            elif len(p) == 2 and ncell[i] >= min_cells and nch[i].min() >= min_cells // 2:
                double[tuple(sorted(p))] = i
        return single, [(k, v) for k, v in double.items() if k[0] in single and k[1] in single]

    def top(v, n):
        return set(np.argsort(-np.abs(v))[:n].tolist())

    def prec(pred, truth, n):
        return len(top(pred, n) & top(truth, n)) / float(n)

    sw = Sweeper("additive - scrambled", axes={"npred": list(NPREDS), "min_cells": list(MIN_CELLS_SWEEP)})
    swd = Sweeper("additive - dominant", axes={"npred": list(NPREDS), "min_cells": list(MIN_CELLS_SWEEP)})
    cells = {}
    for min_cells in MIN_CELLS_SWEEP:
        single, usable = cohort(min_cells)
        keys = [k for k, _ in usable]
        report(f"  min_cells {min_cells}: singles {len(single)} | doubles with BOTH parents measured {len(keys)}")
        if len(keys) < 30:
            report(f"    fewer than 30 usable pairs -- skipped")
            continue
        rS = {g: Pf[i] - ctrl for g, i in single.items()}
        rD = {k: Pf[i] - ctrl for k, i in usable}
        rDh = {k: Ph[i] - ctrl_h for k, i in usable}
        mean_d = np.mean([rD[k] for k in keys], 0)
        for npred in NPREDS:
            # `scrambled` is the ONLY arm with a random component, so the seeds are averaged inside the cell
            # rather than being spread across the grid as if they were independent observations.
            perms = [np.random.default_rng(A.SEED + 31 * s + npred).permutation(len(keys)) for s in SEEDS]
            arms = {n: [] for n in ("additive", "dominant", "single_A",
                                    "best_single_ORACLE", "scrambled", "mean_double")}
            floor, rs = [], []
            for j, k in enumerate(keys):
                a, b = rS[k[0]], rS[k[1]]
                # THE TARGET IS ONE HALF OF THE DOUBLE, and the floor is the OTHER half predicting it.  Scoring
                # the arms against the full double while measuring the floor between two halves would compare a
                # full-depth target with a half-depth one: the floor would come out too low and every arm would
                # look better than it is.  Same target, same depth, same noise, for every arm and the floor.
                # The singles stay at full depth -- they are the predictor, and in any real use you would use the
                # best estimate of the parts you have.
                t = rDh[k][0]
                arms["additive"].append(prec(a + b, t, npred))
                # NOTE the arm that is NOT here. The first version also scored (a+b)/2 as a separate "mean" rule.
                # It is provably identical to `additive`: top-N by |value| is invariant to positive rescaling of
                # the whole vector, so halving it cannot change which genes are ranked first. It printed the same
                # number to four decimals in every cell and tested nothing. The rationale I wrote for including
                # it -- that ranking is "scale-free in one gene but not across two" -- was simply wrong.
                dom = a if np.abs(a).sum() >= np.abs(b).sum() else b
                arms["dominant"].append(prec(dom, t, npred))
                arms["single_A"].append(prec(a, t, npred))
                arms["best_single_ORACLE"].append(max(prec(a, t, npred), prec(b, t, npred)))
                sc = []
                for p in perms:
                    ko = keys[p[j] if p[j] != j else (j + 1) % len(keys)]   # never scramble a pair to itself
                    sc.append(prec(rS[ko[0]] + rS[ko[1]], t, npred))
                arms["scrambled"].append(float(np.mean(sc)))
                arms["mean_double"].append(prec(mean_d, t, npred))
                floor.append(prec(rDh[k][1], t, npred))
                rs.append(float(np.corrcoef(a + b, t)[0, 1]))

            arms = {n: np.array(v) for n, v in arms.items()}
            floor = np.array(floor)
            cfg = {"npred": npred, "min_cells": min_cells}
            tag = f"n{npred}|m{min_cells}"
            sw.add(cfg, tag, arms["additive"] - arms["scrambled"])
            swd.add(cfg, tag, arms["additive"] - arms["dominant"])
            cells[f"npred={npred}|min_cells={min_cells}"] = {
                "n_pairs": len(keys), "floor": float(floor.mean()),
                "pearson_additive": float(np.mean(rs)),
                **{n: float(v.mean()) for n, v in arms.items()}}
            report(f"  N={npred:>2} min{min_cells:>4}: n={len(keys):>3} floor {floor.mean():.4f} | "
                   f"additive {arms['additive'].mean():.4f} | scrambled {arms['scrambled'].mean():.4f} | "
                   f"dominant {arms['dominant'].mean():.4f} | mean_double {arms['mean_double'].mean():.4f}")

    report(f"\n  {'N':>4} {'floor':>8} {'additive':>9} {'dominant':>9} {'single_A':>9} "
           f"{'ORACLE':>8} {'scrambled':>10} {'mean_dbl':>9} {'add/floor':>10}")
    summary = {}
    for npred in NPREDS:
        v = [c for k, c in cells.items() if k.startswith(f"npred={npred}|")]
        if not v:
            continue
        m = {n: float(np.mean([x[n] for x in v])) for n in
             ("floor", "additive", "dominant", "single_A", "best_single_ORACLE",
              "scrambled", "mean_double", "pearson_additive")}
        m["additive_over_floor"] = m["additive"] / max(m["floor"], 1e-9)
        summary[npred] = m
        report(f"  {npred:>4} {m['floor']:>8.4f} {m['additive']:>9.4f} {m['dominant']:>9.4f} "
               f"{m['single_A']:>9.4f} {m['best_single_ORACLE']:>8.4f} {m['scrambled']:>10.4f} "
               f"{m['mean_double']:>9.4f} {m['additive_over_floor']:>10.3f}")

    report(sw.report())
    v1 = sw.verdict()
    report(swd.report())
    v2 = swd.verdict()
    for s, nm in ((sw, "additive-scrambled"), (swd, "additive-dominant")):
        if s.inert_axes() or s.duplicate_cells():
            report(f"  GUARD {nm}: inert axes {s.inert_axes()}, duplicate cells {s.duplicate_cells()}")

    ga = float(np.mean([summary[n]["additive"] - summary[n]["scrambled"] for n in NPREDS]))
    gd = float(np.mean([summary[n]["additive"] - summary[n]["dominant"] for n in NPREDS]))
    fr = float(np.mean([summary[n]["additive_over_floor"] for n in NPREDS]))
    report("\n  READING")
    npos = sum(1 for c in sw._cells.values() if c["lo"] > 0)
    if npos > len(sw._cells) / 2:
        report(f"  Adding two MEASURED single responses predicts the measured double better than the same rule")
        report(f"  applied to the wrong pair ({ga:+.4f}). EFFECTS COMPOSE -- on measured parts, with no model in")
        report(f"  the loop. The additive rule reaches {fr:.1%} of the measurement's own self-agreement, so the")
        report("  remaining headroom is what an interaction term would have to explain.")
        report("  This makes the compositional model a problem of GETTING the parts, not of inventing them.")
    else:
        report(f"  Adding two measured single responses does NOT beat the same rule on the wrong pair ({ga:+.4f}).")
        report("  Even with both parts MEASURED, the joint response does not follow from them. No architecture")
        report("  recovers a composition that is not present in the measurements, so the compositional model is")
        report("  dead on data, not merely unbuilt.")
    if gd < 0:
        report(f"  dominant beats additive by {-gd:+.4f}: where composition happens it looks like MASKING, so the")
        report("  right primitive would be a max rather than a sum.")

    json.dump({"test": "norman_compose", "npreds": list(NPREDS), "seeds": list(SEEDS),
               "n_pairs": len(usable), "cells": cells,
               "summary": {str(k): v for k, v in summary.items()},
               "gap_additive_scrambled": ga, "gap_additive_dominant": gd, "additive_over_floor": fr,
               "sweep_scrambled": v1, "sweep_dominant": v2, "log": log},
              open(OUT / "norman_compose.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'norman_compose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
