"""WHY THE SLOPE IS 1.3 -- diagnosing the one thing loop 22 got away with.

WHERE THIS COMES FROM.  loop_quantity passed Y2 by 0.0023: the log-log slope of protein against mRNA is
1.2977 against a declared ceiling of 1.3. That gate was written to test whether protein is PROPORTIONAL
to transcript, which is what cell_loop's balance asserts, and 1.3 is not proportional -- a tenfold
change in transcript goes with a twentyfold change in protein. The module said so rather than banking
the pass. This finds out why.

WHAT THE MODEL ASSUMES, and it is the suspect.  cell_loop declares ONE protein degradation constant for
every gene: K_DP = ln(2)/24h, uniform, flagged in its own source as an assumption. Under that
assumption P = k_tl * m / (k_dp + mu) is strictly proportional to m and the slope MUST be 1. A measured
slope of 1.3 therefore falsifies something, and the uniform constant is the obvious candidate: if
abundant-mRNA genes also make longer-lived proteins, protein accumulates faster than transcript and the
slope exceeds 1 for a reason the model has assumed away.

THAT IS TESTABLE WITHOUT PROTEIN HALF-LIFE DATA, which this container does not have.  If the deviation
is driven by a systematic property, the RESIDUAL from a slope-1 fit will correlate with it. Four
candidates, each a different explanation:
    mRNA stability     long-lived transcripts making long-lived proteins would produce exactly this
    gene function      structural and metabolic proteins turn over differently from regulators
    abundance itself   pure measurement compression -- mass-spectrometry under-reports the extremes,
                       which would make this an artefact of the assay and not biology at all
    gene length        a nuisance variable that should NOT explain it

PREDECLARED, before any number:

    A1 THE SLOPE IS REALLY ABOVE 1   the 95% bootstrap interval on the slope excludes 1.0.
                fails -> 1.2977 is noise, loop 22's Y2 caveat was over-stated, and I should say so.
    A2 IT IS NOT THE ASSAY   the residual's correlation with protein abundance is weaker than with the
                best biological candidate. fails -> the amplification is a measurement artefact and the
                model's assumption is not implicated.
    A3 SOMETHING NAMEABLE EXPLAINS IT   at least one candidate correlates with the residual above the
                95th percentile of 200 shuffles.
    A4 IT IS THE UNIFORM CONSTANT   mRNA stability specifically explains part of it, which is the
                mechanism the model assumed away.
                fails -> the deviation is real and comes from somewhere else; name what.

CONTROLS: 2,000 bootstrap resamples for the slope; 200 shuffles per candidate; gene length as a
nuisance variable that should not explain it; and protein abundance as the artefact control.

-> outputs/loop_amplify.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import KDEG, CELL, MIN_LINES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_BOOT = 2000
N_SHUFFLE = 200


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("WHY THE SLOPE IS 1.3")
    say("=" * 100)
    say("  loop 22 passed its proportionality gate by 0.0023 and said so. cell_loop assumes ONE protein")
    say("  degradation constant for every gene, under which the slope MUST be 1. A measured 1.2977")
    say("  falsifies something; this finds out what.")

    D = json.load(open(CELL))
    genes = D["genes"]
    ppm = {int(k): float(v) for k, v in D["ppm"].items()}
    k = pd.read_csv(KDEG)
    rp = (k[k.avg_RPKM_total > 0].groupby("feature_ID")
          .agg(rpkm=("avg_RPKM_total", "median"), n=("cell_line", "size")))
    kd = k.groupby("feature_ID").agg(kdeg=("avg_kdeg", "median"), n=("cell_line", "size"))
    rp, kd = rp[rp.n >= MIN_LINES], kd[kd.n >= MIN_LINES]

    rows = []
    for i, g in enumerate(genes):
        s = g["name"]
        p = ppm.get(i)
        ln = (g.get("gene_end") or 0) - (g.get("gene_start") or 0)
        if p is None or p <= 0 or s not in rp.index or s not in kd.index or ln <= 0:
            continue
        rows.append({"sym": s, "protein": float(p), "mrna": float(rp.rpkm[s]),
                     "kdeg": float(kd.kdeg[s]), "len": float(ln),
                     "path": g.get("path") or ""})
    T = pd.DataFrame(rows)
    y = np.log10(T.protein.to_numpy())
    m = np.log10(T.mrna.to_numpy())
    say(f"\n  {len(T):,} genes with protein, mRNA and decay rate")

    slope, intercept = np.polyfit(m, y, 1)
    boot = np.array([np.polyfit(m[i], y[i], 1)[0]
                     for i in (rng.integers(0, len(m), len(m)) for _ in range(N_BOOT))])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    a1 = bool(lo > 1.0)
    say(f"\n  A1 THE SLOPE IS REALLY ABOVE 1 -- {slope:.4f}, 95% bootstrap [{lo:.4f}, {hi:.4f}] "
        f"over {N_BOOT:,} resamples   A1 {'PASS' if a1 else 'FAIL'}")
    if a1:
        say(f"    Proportionality is excluded. A tenfold change in transcript goes with a "
            f"{10**slope:.0f}-fold change in protein, and cell_loop's uniform k_dp cannot produce that.")

    resid = y - (slope * m + intercept)
    cands = {"mRNA stability (-log kdeg)": -np.log10(T.kdeg.to_numpy()),
             "protein abundance (ARTEFACT CONTROL)": y,
             "gene length (NUISANCE)": np.log10(T["len"].to_numpy()),
             "mRNA abundance": m}
    say(f"\n  what the residual from a slope-{slope:.2f} fit correlates with:")
    res = {}
    for name, v in cands.items():
        r = float(pd.Series(resid).corr(pd.Series(v), method="spearman"))
        null = np.array([float(pd.Series(resid).corr(pd.Series(rng.permutation(v)),
                                                     method="spearman"))
                         for _ in range(N_SHUFFLE)])
        p95 = float(np.percentile(np.abs(null), 95))
        res[name] = {"rho": r, "null_p95": p95, "beats": bool(abs(r) > p95)}
        say(f"    {name:<40}{r:>+9.4f}   |null| 95th {p95:.4f}   "
            f"{'SIGNIFICANT' if abs(r) > p95 else 'no'}")

    r_assay = abs(res["protein abundance (ARTEFACT CONTROL)"]["rho"])
    bio = {n: abs(v["rho"]) for n, v in res.items() if "ARTEFACT" not in n and "NUISANCE" not in n}
    best_bio = max(bio, key=bio.get)
    a2 = bool(r_assay < bio[best_bio])
    say(f"\n  A2 IT IS NOT THE ASSAY -- abundance {r_assay:.4f} vs best biological "
        f"({best_bio}) {bio[best_bio]:.4f}   A2 {'PASS' if a2 else 'FAIL'}")
    if not a2:
        say("    The residual tracks protein abundance more strongly than anything biological. That is")
        say("    what mass-spectrometry compression looks like: the assay under-reports both extremes,")
        say("    which inflates a log-log slope without any biology being involved. The model's uniform")
        say("    constant is NOT implicated by this evidence.")

    a3 = bool(any(v["beats"] for n, v in res.items() if "NUISANCE" not in n))
    say(f"  A3 SOMETHING NAMEABLE EXPLAINS IT -- {'PASS' if a3 else 'FAIL'}")
    a4 = bool(res["mRNA stability (-log kdeg)"]["beats"])
    say(f"  A4 IT IS THE UNIFORM CONSTANT -- mRNA stability rho "
        f"{res['mRNA stability (-log kdeg)']['rho']:+.4f}   A4 {'PASS' if a4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"A1 slope really above 1": a1, "A2 not the assay": a2,
             "A3 something nameable explains it": a3, "A4 the uniform constant is implicated": a4}
    for kk, vv in gates.items():
        say(f"  {kk:<42}{'PASS' if vv else 'FAIL'}")
    if a1 and not a2:
        say("  THE AMPLIFICATION IS MOSTLY MEASUREMENT.  The slope really is above 1, and the residual")
        say("  tracks protein abundance more strongly than any biological candidate -- the signature of")
        say("  assay compression rather than of a wrong degradation constant. loop 22's Y2 caveat stands")
        say("  as a caveat about the DATA, and cell_loop's uniform k_dp is not convicted by it.")
    elif a1 and a4:
        say("  THE UNIFORM DEGRADATION CONSTANT IS IMPLICATED.  The slope is above 1, it is not an assay")
        say("  artefact, and the residual tracks mRNA stability -- genes with long-lived transcripts")
        say("  accumulate more protein than proportionality allows. cell_loop's declared assumption of")
        say("  one k_dp for every gene is the specific thing to fix, and this is the evidence.")
    elif a1:
        say("  THE SLOPE IS REAL AND UNEXPLAINED by any candidate tested here. Recorded as open.")
    else:
        say("  THE SLOPE IS NOT DISTINGUISHABLE FROM 1. loop 22's Y2 caveat was over-stated and this")
        say("  retracts it: proportionality is not excluded by this data.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(KDEG)], available=len(genes), used=len(T),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_BOOT} bootstrap resamples of the slope",
                                f"{N_SHUFFLE} shuffles per candidate",
                                "protein abundance as the assay-artefact control",
                                "gene length as a nuisance that should not explain it"],
                      note="tests a specific declared assumption of cell_loop -- one k_dp for every "
                           "gene -- rather than the model as a whole")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_amplify", "manifest": man, "gates": gates,
               "slope": float(slope), "slope_ci": [float(lo), float(hi)],
               "residual_correlations": res, "n": len(T), "log": log},
              open(OUT / "loop_amplify.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_amplify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
