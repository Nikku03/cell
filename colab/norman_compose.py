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
    mean            (r(A) + r(B)) / 2.  Additive up to scale; separated because top-20 ranking is scale-free in
                    one gene but not across two, so the two rules differ in which genes they promote.
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
MIN_CELLS = 50


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

    single = {}
    double = {}
    for i, g in enumerate(groups):
        p = parts(g)
        if len(p) == 1 and ncell[i] >= MIN_CELLS:
            single[p[0]] = i
        elif len(p) == 2 and ncell[i] >= MIN_CELLS and nch[i].min() >= MIN_CELLS // 2:
            double[tuple(sorted(p))] = i
    usable = [(k, v) for k, v in double.items() if k[0] in single and k[1] in single]
    report(f"  singles {len(single)} | doubles {len(double)} | doubles with BOTH parents measured "
           f"{len(usable)}")
    if len(usable) < 30:
        report("  too few usable pairs -- nothing to conclude")
        return 1

    # responses relative to control, computed once
    rS = {g: Pf[i] - ctrl for g, i in single.items()}
    rD = {k: Pf[i] - ctrl for k, i in usable}
    rDh = {k: Ph[i] - ctrl_h for k, i in usable}          # (2, genes)

    def top(v, n):
        return set(np.argsort(-np.abs(v))[:n].tolist())

    def prec(pred, truth, n):
        return len(top(pred, n) & top(truth, n)) / float(n)

    keys = [k for k, _ in usable]
    sw = Sweeper("additive - scrambled", axes={"npred": list(NPREDS), "seed": list(SEEDS)})
    swd = Sweeper("additive - dominant", axes={"npred": list(NPREDS), "seed": list(SEEDS)})
    cells = {}
    for npred in NPREDS:
        for seed in SEEDS:
            rng = np.random.default_rng(A.SEED + 31 * seed + npred)
            perm = rng.permutation(len(keys))
            arms = {n: [] for n in ("additive", "mean", "dominant", "single_A",
                                    "best_single_ORACLE", "scrambled", "mean_double")}
            floor, rs = [], []
            mean_d = np.mean([rD[k] for k in keys], 0)
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
                arms["mean"].append(prec(0.5 * (a + b), t, npred))
                dom = a if np.abs(a).sum() >= np.abs(b).sum() else b
                arms["dominant"].append(prec(dom, t, npred))
                arms["single_A"].append(prec(a, t, npred))
                arms["best_single_ORACLE"].append(max(prec(a, t, npred), prec(b, t, npred)))
                ko = keys[perm[j]]
                arms["scrambled"].append(prec(rS[ko[0]] + rS[ko[1]], t, npred))
                arms["mean_double"].append(prec(mean_d, t, npred))
                floor.append(prec(rDh[k][1], t, npred))
                v1, v2 = a + b, t
                rs.append(float(np.corrcoef(v1, v2)[0, 1]))

            arms = {n: np.array(v) for n, v in arms.items()}
            floor = np.array(floor)
            sw.add({"npred": npred, "seed": seed}, f"n{npred}|s{seed}",
                   arms["additive"] - arms["scrambled"])
            swd.add({"npred": npred, "seed": seed}, f"n{npred}|s{seed}",
                    arms["additive"] - arms["dominant"])
            cells[f"npred={npred}|seed={seed}"] = {
                "n_pairs": len(keys), "floor": float(floor.mean()),
                "pearson_additive": float(np.mean(rs)),
                **{n: float(v.mean()) for n, v in arms.items()}}
            report(f"  N={npred:>2} seed{seed}: floor {floor.mean():.4f} | additive {arms['additive'].mean():.4f} "
                   f"| scrambled {arms['scrambled'].mean():.4f} | dominant {arms['dominant'].mean():.4f} "
                   f"| mean_double {arms['mean_double'].mean():.4f}")

    report(f"\n  {'N':>4} {'floor':>8} {'additive':>9} {'mean':>8} {'dominant':>9} {'single_A':>9} "
           f"{'ORACLE':>8} {'scrambled':>10} {'mean_dbl':>9} {'add/floor':>10}")
    summary = {}
    for npred in NPREDS:
        v = [c for k, c in cells.items() if k.startswith(f"npred={npred}|")]
        m = {n: float(np.mean([x[n] for x in v])) for n in
             ("floor", "additive", "mean", "dominant", "single_A", "best_single_ORACLE",
              "scrambled", "mean_double", "pearson_additive")}
        m["additive_over_floor"] = m["additive"] / max(m["floor"], 1e-9)
        summary[npred] = m
        report(f"  {npred:>4} {m['floor']:>8.4f} {m['additive']:>9.4f} {m['mean']:>8.4f} {m['dominant']:>9.4f} "
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
