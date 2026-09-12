"""GATE: is there a real (KNOCKOUT x CELL-STATE) interaction, or does pseudobulking discard only noise?

THE IDEA BEING TESTED.  Every model in this project is trained on PSEUDOBULK -- one profile per knockout, averaged
over its ~100 cells.  But those cells are not replicates of one condition: they differ in cell-cycle phase, size,
depth and baseline state.  If a knockout's effect DEPENDS on the state the cell was in, then pseudobulk is
averaging over a real interaction, and the single-cell data contains (perturbation x context) observations that
the model has never seen.  That interaction is exactly the missing input behind today's cross-line and LOCO
failures -- the model has no representation of cell state at all.

THE CORRECTION TO THE PREMISE, because it changes what can be claimed.  ~965,000 single cells exist here, but
~104 cells sharing a guide are NOT 104 independent perturbation conditions.  They are 104 observations of ONE
knockout in different states.  The gain, if real, is not "2 million perturbations" -- it is the ability to
estimate a knockout's effect CONDITIONAL on state, which pseudobulk cannot do at any sample size.

THE MEASUREMENT.  For each knockout with enough cells, order its cells along a state axis, take the extreme
n at each end, and compute

    interaction_g = [mean_g(top n) - mean_g(bottom n)] - [mean_ctrl(top) - mean_ctrl(bottom)]

The control term removes the STATE MAIN EFFECT -- cell-cycle changes expression for every cell, perturbed or not,
and without subtracting it this test would "discover" the cell cycle rather than any interaction.

THE NOISE FLOOR, which is what makes it a test.  The same 2n cells are split at RANDOM into two groups of n and
the identical statistic is computed.  That difference has the same sample size, the same depth distribution and
the same control correction, and contains no state contrast by construction.  So:

    ratio = RMS(interaction) / RMS(random split)

    ratio ~= 1   pseudobulk discards only sampling noise. The single-cell direction is CLOSED, and every model
                 here is right to average.
    ratio >> 1   knockout effects are genuinely state-dependent, pseudobulk destroys that, and conditioning on
                 state is worth building.

PREDECLARED: judged by robustness.Sweeper on (RMS_interaction - RMS_random) per knockout, swept over the state
axis and the cell line, because "which state axis" and "which line" are exactly the undeliberated choices this
project has been burned by. ROBUST means real state-dependence. NULL means the pseudobulk was right.

STATE AXES.  `cycle` = summed expression of S and G2/M markers, the dominant known axis of cell-to-cell
variation. `umi` = total UMI count, a technical/size axis included deliberately as a semi-negative control: if
`umi` shows the same interaction as `cycle`, the effect is depth artefact rather than biology.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
LINES = {"RPE1": "rpe1.h5ad", "Jurkat": "nadig_jurkat.h5ad"}
AXES = ("cycle", "umi")
MIN_CELLS, N_SPLITS, BLOCK = 60, 5, 8192

S_GENES = ("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 HELLS RFC2 RPA2 NASP "
           "RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 "
           "BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8").split()
G2M_GENES = ("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF TACC3 SMC4 "
             "CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP CDCA3 CDC20 TTK CDC25C "
             "KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE "
             "CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA").split()


def _dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def load_line(path, report):
    h = h5py.File(path, "r")
    gn = _dec(h["var"]["gene_name"][:])
    gidx = {g: i for i, g in enumerate(gn)}
    pc = h["obs"]["gene"]
    cats = _dec(pc["categories"][:])
    codes = pc["codes"][:]
    umi = np.asarray(h["obs"]["UMI_count"][:], np.float64)
    ncell, ngene = h["X"].shape
    cyc_cols = sorted({gidx[g] for g in S_GENES + G2M_GENES if g in gidx})
    report(f"    {ncell:,} cells x {ngene:,} genes; {len(cyc_cols)} cycle markers found; "
           f"{len(cats)} perturbation labels")
    cyc = np.zeros(ncell)
    for i0 in range(0, ncell, BLOCK):
        blk = h["X"][i0:i0 + BLOCK, :]
        cyc[i0:i0 + blk.shape[0]] = blk[:, cyc_cols].sum(1)
    cyc = cyc / np.maximum(umi, 1) * 1e4
    return h, gn, cats, codes, umi, cyc


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("CELL-STATE GATE: is the (knockout x state) interaction real, or does pseudobulk discard only noise?")
    report("=" * 100)
    report("  PREDECLARED: ratio ~= 1 closes the single-cell direction; ratio >> 1 justifies conditioning.")

    sw = Sweeper("RMS(interaction) - RMS(random split)", axes={"line": list(LINES), "axis": list(AXES)})
    detail = {}

    for line, fn in LINES.items():
        report(f"\n  === {line} ===")
        h, gn, cats, codes, umi, cyc = load_line(SP / fn, report)
        ctrl_names = [c for c in cats if c.lower() in ("control", "non-targeting", "nt", "ntc", "safe-harbor")]
        cidx = {cats.index(c) for c in ctrl_names} if ctrl_names else set()
        if not cidx:
            lc = [i for i, c in enumerate(cats) if "control" in c.lower() or "non-target" in c.lower()]
            cidx = set(lc)
        report(f"    control labels: {[cats[i] for i in cidx][:3]} ({sum((codes == i).sum() for i in cidx):,} cells)")
        if not cidx:
            report("    SKIP: no control population identified")
            h.close()
            continue

        is_ctrl = np.isin(codes, list(cidx))
        for axname in AXES:
            state = cyc if axname == "cycle" else umi.astype(np.float64)
            rng = np.random.default_rng(A.SEED)
            # control state contrast, computed once per axis on control cells only
            cs = state[is_ctrl]
            lo_q, hi_q = np.quantile(cs, [0.25, 0.75])
            ctrl_lo = np.where(is_ctrl & (state <= lo_q))[0]
            ctrl_hi = np.where(is_ctrl & (state >= hi_q))[0]
            nctrl = min(len(ctrl_lo), len(ctrl_hi), 4000)
            ctrl_lo = rng.choice(ctrl_lo, nctrl, replace=False)
            ctrl_hi = rng.choice(ctrl_hi, nctrl, replace=False)

            def mean_rows(idx):
                idx = np.sort(idx)
                acc = np.zeros(h["X"].shape[1])
                tot = 0.0
                for j0 in range(0, len(idx), 2048):
                    sub = idx[j0:j0 + 2048]
                    blk = h["X"][sub, :].astype(np.float64)
                    blk = blk / np.maximum(umi[sub][:, None], 1) * 1e4
                    acc += blk.sum(0)
                    tot += len(sub)
                return acc / max(tot, 1)

            c_hi, c_lo = mean_rows(ctrl_hi), mean_rows(ctrl_lo)
            c_delta = c_hi - c_lo
            # random control split, for the noise arm's matched control correction
            allc = np.concatenate([ctrl_hi, ctrl_lo])
            perm = rng.permutation(len(allc))
            c_rdelta = mean_rows(allc[perm[:nctrl]]) - mean_rows(allc[perm[nctrl:]])

            inter, noise, used = [], [], 0
            for gi, gname in enumerate(cats):
                if gi in cidx:
                    continue
                cells = np.where(codes == gi)[0]
                if len(cells) < MIN_CELLS:
                    continue
                s = state[cells]
                order = np.argsort(s)
                n = len(cells) // 3
                bot, top = cells[order[:n]], cells[order[-n:]]
                d_state = (mean_rows(top) - mean_rows(bot)) - c_delta
                pooled = np.concatenate([bot, top])
                dr = []
                for _ in range(N_SPLITS):
                    p = rng.permutation(len(pooled))
                    dr.append((mean_rows(pooled[p[:n]]) - mean_rows(pooled[p[n:2 * n]])) - c_rdelta)
                d_rand = np.mean([np.sqrt(np.mean(x ** 2)) for x in dr])
                inter.append(float(np.sqrt(np.mean(d_state ** 2))))
                noise.append(float(d_rand))
                used += 1
                if used >= 150:
                    break
            inter, noise = np.array(inter), np.array(noise)
            sw.add({"line": line, "axis": axname}, f"{line}|{axname}", inter - noise)
            detail[f"{line}|{axname}"] = {"n_knockouts": int(used), "rms_interaction": float(inter.mean()),
                                          "rms_random": float(noise.mean()),
                                          "ratio": float(inter.mean() / max(noise.mean(), 1e-12))}
            report(f"    {axname:<6} {used} knockouts | RMS interaction {inter.mean():.4f} "
                   f"| RMS random {noise.mean():.4f} | ratio {inter.mean()/max(noise.mean(),1e-12):.3f}")
        h.close()

    report(sw.report())
    v = sw.verdict()
    report("\n  READING")
    ratios = {k: d["ratio"] for k, d in detail.items()}
    cyc_r = [r for k, r in ratios.items() if k.endswith("cycle")]
    umi_r = [r for k, r in ratios.items() if k.endswith("umi")]
    if cyc_r and umi_r:
        report(f"  cycle ratios {[round(x,3) for x in cyc_r]} vs umi ratios {[round(x,3) for x in umi_r]}")
        report("  If `umi` matches `cycle`, the interaction is a depth artefact rather than biology.")
    if v["verdict"] in ("ROBUST", "DIRECTIONAL"):
        report("  State-dependent knockout effects are REAL -- pseudobulk is averaging over signal, and")
        report("  conditioning the model on cell state is justified.")
    else:
        report("  No state-dependence beyond the matched noise floor: pseudobulk discards only sampling noise,")
        report("  and the single-cell direction is CLOSED for this question.")

    json.dump({"test": "adrn_cellstate_gate", "lines": list(LINES), "axes": list(AXES),
               "min_cells": MIN_CELLS, "detail": detail, "sweep": v, "log": log},
              open(OUT / "adrn_cellstate_gate.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'adrn_cellstate_gate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
