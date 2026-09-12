"""STEP 3: the (knockout x cell-state) interaction gate, run in memory on the cohort tensor.

Same science as adrn_cellstate_gate.py, which was killed after 23 minutes: that version issued scattered row
reads against a dense 8.7 GB HDF5 matrix, twelve times per knockout, and would have taken hours while starving
the dataset build of I/O. The cohort tensor is already the right object -- 200 perturbations x 3 lines, in RAM --
so the same measurement costs seconds. Nothing about the design changed, only where the bytes come from.

THE QUESTION.  Every model here trains on pseudobulk, averaging a knockout over its ~100 cells. If a knockout's
effect DEPENDS on the state the cell was in, pseudobulk destroys that -- and cell state is exactly the input
whose absence explains the cross-line and LOCO failures.

    interaction_g = [mean_g(top state third) - mean_g(bottom state third)]
                  - [mean_ntc(top) - mean_ntc(bottom)]        <- subtracts the STATE MAIN EFFECT

Without the control term this would "discover" the cell cycle: every cell's expression changes with phase,
perturbed or not. Controls are taken from the SAME BATCHES as the perturbation, so batch is not confounded in.

    noise_g       = the identical statistic on a RANDOM split of the same 2n cells.

Same sample size, same depth distribution, same control correction, and no state contrast by construction. So
ratio = RMS(interaction) / RMS(noise) is the whole answer:

    ~= 1   pseudobulk discards only sampling noise. The single-cell direction is CLOSED and averaging is right.
    >> 1   knockout effects are state-dependent, and conditioning the model on state is worth building.

AXES.  `g2m` and `s` are the two cell-cycle scores, kept separate because a knockout can arrest in one phase
without touching the other. `umi` is sequencing depth, included as a SEMI-NEGATIVE CONTROL: if depth shows the
same interaction as cycle, the effect is technical rather than biological and the gate should not be trusted.
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
AXES = ("g2m", "s", "umi")
MIN_CELLS, N_SPLITS, MAX_PERT = 45, 5, 200


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 3 -- CELL-STATE GATE on the cohort tensor")
    report("=" * 100)
    report("  PREDECLARED: ratio ~= 1 closes the single-cell direction; ratio >> 1 justifies conditioning.")

    z = np.load(SP / "sc_cohort_cells.npz", allow_pickle=True)
    X = z["X"]
    line = np.array([str(x) for x in z["line"]])
    pert = np.array([str(x) for x in z["pert"]])
    batch = z["batch"]
    ntc = z["is_ntc"]
    state_all = {"g2m": z["g2m_score"].astype(np.float64), "s": z["s_score"].astype(np.float64),
                 "umi": z["umi"].astype(np.float64)}
    report(f"  tensor {X.shape}; {int((~ntc).sum()):,} perturbed, {int(ntc.sum()):,} NTC cells")

    sw = Sweeper("RMS(interaction) - RMS(random split)", axes={"line": sorted(set(line)), "axis": list(AXES)})
    detail = {}

    for ln in sorted(set(line)):
        lm = line == ln
        for ax in AXES:
            st = state_all[ax]
            rng = np.random.default_rng(A.SEED)
            inter, noise, used = [], [], 0
            for g in sorted(set(pert[lm & ~ntc])):
                cm = lm & (pert == g) & ~ntc
                n_tot = int(cm.sum())
                if n_tot < MIN_CELLS:
                    continue
                cells = np.where(cm)[0]
                bs = set(batch[cells].tolist())
                ctl = np.where(lm & ntc & np.isin(batch, list(bs)))[0]
                if len(ctl) < 30:
                    continue
                n = n_tot // 3
                order = np.argsort(st[cells])
                bot, top = cells[order[:n]], cells[order[-n:]]
                co = np.argsort(st[ctl])
                m = min(len(ctl) // 3, 2000)
                c_delta = X[ctl[co[-m:]]].mean(0) - X[ctl[co[:m]]].mean(0)
                d_state = (X[top].mean(0) - X[bot].mean(0)) - c_delta
                pooled = np.concatenate([bot, top])
                rs = []
                for _ in range(N_SPLITS):
                    p = rng.permutation(len(pooled))
                    cp = rng.permutation(len(ctl))
                    c_r = X[ctl[cp[:m]]].mean(0) - X[ctl[cp[m:2 * m]]].mean(0)
                    rs.append(np.sqrt(np.mean(((X[pooled[p[:n]]].mean(0) - X[pooled[p[n:2 * n]]].mean(0))
                                               - c_r) ** 2)))
                inter.append(float(np.sqrt(np.mean(d_state ** 2))))
                noise.append(float(np.mean(rs)))
                used += 1
                if used >= MAX_PERT:
                    break
            if used < 10:
                report(f"    {ln} {ax}: only {used} usable perturbations -- skipped")
                continue
            inter, noise = np.array(inter), np.array(noise)
            sw.add({"line": ln, "axis": ax}, f"{ln}|{ax}", inter - noise)
            ratio = float(inter.mean() / max(noise.mean(), 1e-12))
            detail[f"{ln}|{ax}"] = {"n": used, "rms_interaction": float(inter.mean()),
                                    "rms_noise": float(noise.mean()), "ratio": ratio,
                                    "frac_above": float((inter > noise).mean())}
            report(f"    {ln:<7} {ax:<4} n={used:>3} | interaction {inter.mean():.4f} | noise {noise.mean():.4f}"
                   f" | ratio {ratio:.3f} | {100*(inter>noise).mean():.0f}% of perts above noise")

    report(sw.report())
    v = sw.verdict()
    cyc = [d["ratio"] for k, d in detail.items() if k.endswith(("g2m", "|s"))]
    umi = [d["ratio"] for k, d in detail.items() if k.endswith("umi")]
    report("\n  READING")
    if cyc and umi:
        report(f"  cycle ratios {[round(x,3) for x in cyc]}  vs  umi ratios {[round(x,3) for x in umi]}")
        if max(umi) >= max(cyc) * 0.9:
            report("  DEPTH matches CYCLE -- the interaction is technical, not biological. Do not condition on")
            report("  state on the strength of this: fix normalisation first.")
        else:
            report("  cycle exceeds depth, so the interaction is not merely a depth artefact.")
    if v["verdict"] in ("ROBUST", "DIRECTIONAL"):
        report("  State-dependent knockout effects are REAL. Pseudobulk averages over signal and conditioning")
        report("  the model on cell state is justified -- proceed to the conditioned models.")
    else:
        report("  No state-dependence beyond the matched noise floor: pseudobulk discards only sampling noise.")
        report("  Conditioning on state has nothing to condition on, and this direction is CLOSED.")

    json.dump({"test": "sc_state_gate", "axes": list(AXES), "min_cells": MIN_CELLS,
               "detail": detail, "sweep": v, "log": log},
              open(OUT / "sc_state_gate.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_state_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
