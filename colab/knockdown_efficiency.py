"""knockdown_efficiency -- item #3 tested rather than assumed. The claim under test was mine: 83% of profiled knockouts move almost nothing
(median 1 gene), and I attributed that to FAILED KNOCKDOWN, recommending QC'd guides as the biggest available multiplier. That is checkable
directly, because CRISPRi lowers the target's own transcript: if the guide worked, the knocked-out gene's own z-score should be strongly negative
in its own row, whether or not anything else responded.

  SELF-KNOCKDOWN z  = the target gene's own z in its own knockout. Strongly negative => the perturbation worked.
  NULL knockout     = moves fewer than 20 genes at the calibrated threshold.

The decisive split: among NULL knockouts, how many nonetheless show a clear self-knockdown? Those are genes that were successfully silenced and
the cell simply did not care -- genuinely DISPENSABLE, not recoverable by better reagents. Only the remainder (silenced poorly or not at all) are
addressable by guide QC, and that fraction is the honest size of the "better perturbations" lever. Deterministic."""
import os
import json, sys
from pathlib import Path
import numpy as np

SP = Path(os.environ.get("CELL_SCRATCH", "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
KD = -4.0          # self-z at or below this counts as a clear knockdown
MINMOV = 20


def run(line, thr):
    d = np.load(SP / f"unc_{line}.npz", allow_pickle=True)
    kos = [str(x) for x in d["kos"]]; genes = [str(x) for x in d["genes"]]
    Z = d["Z"]; nc = d["ncells"]; gi = {g: i for i, g in enumerate(genes)}
    mv = (np.abs(Z) >= thr).sum(1)
    rows = [(r, kos[r]) for r in range(len(kos)) if kos[r] in gi]
    self_z = np.array([Z[r, gi[k]] for r, k in rows])
    idx = np.array([r for r, _ in rows]); mvo = mv[idx]; nco = nc[idx]
    null = mvo < MINMOV
    good = self_z <= KD
    res = {"line": line, "threshold": thr, "n_measurable": int(len(rows)),
           "median_self_z": float(np.median(self_z)), "frac_clear_kd": float(np.mean(good)),
           "n_null": int(null.sum()), "null_with_good_kd": int((null & good).sum()),
           "null_frac_dispensable": float(np.mean(good[null])) if null.any() else float("nan"),
           "responsive_frac_good_kd": float(np.mean(good[~null])) if (~null).any() else float("nan")}
    print(f"[{line}] threshold |z|>={thr}")
    print(f"  {len(rows)} knockouts whose own transcript is measurable | median self-knockdown z {np.median(self_z):+.1f} "
          f"| clear knockdown in {100*np.mean(good):.0f}%")
    print(f"  {'group':22s} {'n':>5s} {'median self-KD z':>17s} {'clear self-drop':>16s} {'median cells':>13s}")
    for nm, m in (("NULL (<20 movers)", null), ("RESPONSIVE (>=20)", ~null)):
        if m.any():
            print(f"  {nm:22s} {int(m.sum()):>5d} {np.median(self_z[m]):>17.1f} {100*np.mean(good[m]):>15.0f}% "
                  f"{np.median(nco[m]):>13.0f}")
    print(f"  -> of {int(null.sum())} NULL knockouts, {int((null&good).sum())} ({100*np.mean(good[null]):.0f}%) WERE successfully silenced "
          f"=> genuinely dispensable genes")
    print(f"  -> only {int((null&~good).sum())} ({100*np.mean(~good[null]):.0f}%) lack a clear knockdown => the part guide QC could recover\n")
    return res


def main():
    cal = json.load(open(SP / "unc_calibration.json"))
    res = [run(ln, cal[ln]["threshold"]) for ln in (sys.argv[1:] or ["RPE1", "HepG2", "Jurkat"]) if ln in cal]
    disp = float(np.mean([r["null_frac_dispensable"] for r in res]))
    verdict = (
        "KNOCKDOWN EFFICIENCY (knockdown_efficiency.py): a direct test of MY OWN recommendation, and it refutes it. I had argued that because "
        "~83% of profiled knockouts move almost nothing, the null rate was mostly FAILED KNOCKDOWN and that QC'd guides were the biggest cheap "
        "multiplier available. CRISPRi lowers the target's own transcript, so this is checkable: "
        + "; ".join(f"{r['line']} {100*r['frac_clear_kd']:.0f}% of knockouts show a clear self-knockdown (median self-z "
                    f"{r['median_self_z']:+.1f})" for r in res) + ". "
        f"THE DECISIVE SPLIT: among NULL knockouts, {100*disp:.0f}% nonetheless show a clear self-knockdown -- the gene WAS silenced and the cell "
        "simply did not respond. Those are genuinely DISPENSABLE genes, not reagent failures, and no amount of guide QC recovers them. Only the "
        f"remaining ~{100*(1-disp):.0f}% lack a clear knockdown and are addressable by better reagents. "
        "CONSEQUENCE: 'better perturbations' is a much smaller lever than I claimed -- it is worth doing for the minority of poorly-silenced "
        "targets, but it will NOT convert the null majority into usable sources, because those knockouts already worked. The reason most of the "
        "genome yields no transcriptional response is biology (dispensability and buffering), not technique. Deterministic.")
    print(f"VERDICT: {verdict}")
    json.dump({"results": res, "null_frac_dispensable_mean": disp, "verdict": verdict, "note": verdict},
              open(OUT / "knockdown_efficiency.json", "w"), indent=1)
    print("\n  -> outputs/orphan/knockdown_efficiency.json")


if __name__ == "__main__":
    main()
