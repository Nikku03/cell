"""DOES THE CELL HAVE SLACK -- and was the gate that said no even asking the right question?

WHERE THIS COMES FROM.  loop_medium's M3 failed: 2,619 of 2,848 knockouts (92.0%) moved growth, against
a gate of 90%, and the module concluded "a cell in which 92% of single deletions change growth is still
not a cell". That conclusion may be right. But the threshold behind it was `abs(v - mu_med) > 1e-9`,
which is a FLOATING-POINT threshold, not a biological one. It counts a knockout that costs a
ten-billionth of a percent of growth as "moving growth", identically to one that kills the cell.

So before accepting that the model has no slack, the gate itself gets audited. This is the philosopher
turned on my own instrument rather than on the repository.

WHAT SLACK ACTUALLY MEANS.  A real cell tolerates most single gene deletions. DepMap measures this
directly: `dep_frac`, the fraction of cell lines in which a gene scores as a dependency. If the model
has realistic slack, the DISTRIBUTION of its knockout costs should look like the distribution of real
dependencies -- mostly zero, with a small tail that matters -- not merely differ from 100% at a
threshold chosen by numerical noise.

PREDECLARED, before any number:

    P1 THE GATE WAS THE PROBLEM   at a biologically meaningful threshold -- a knockout must cost at
                  least 1% of growth to count -- fewer than 50% of deletions move growth.
                  passes -> M3's failure was an artefact of a 1e-9 cutoff and should be restated.
                  fails  -> the model really is saturated and M3's conclusion stands as written.
    P2 SHAPE      the model's cost distribution and DepMap's dependency distribution agree on what
                  fraction of genes matter, within 15 percentage points, at the 1% / 5% / 10% marks.
                  fails -> the model has slack but not the right amount, which is a different and
                  more precise complaint than "no slack".
    P3 RANK       among genes that DO cost something, the model's cost ranks agree with DepMap's
                  dep_frac above the 95th percentile of 200 shuffles.
                  fails -> the slack is in the wrong places even if the amount is right.

CONTROLS: the threshold swept across five orders of magnitude so the reader can see where the answer
changes; DepMap's own dependency distribution as the external comparator; 200 label shuffles for P3.

WHAT HAPPENED, written after the run against the gates above, unedited.

    P1 THE GATE WAS THE PROBLEM   PASS, and the mechanism was found rather than inferred.
        loop_medium reported 92.0% of knockouts moving growth. The stored scores say 14.4% have any
        positive cost at all and 11.7% cost more than 1% of growth. The discrepancy is fully explained:
        1,525 of 1,784 knockouts carry a cost of about -7.2e-07 -- NEGATIVE, which is impossible, since
        deleting a gene cannot make this model grow faster. loop_medium calibrated the uptake cap by
        bisection and then used the bisection's last iterate as mu_WT (0.029999790) rather than
        re-solving at the final bounds. Every unaffected knockout then solved to a marginally higher mu,
        and M3's `abs(diff) > 1e-9` counted all of them.
        THE 92% WAS AN OFF-BY-A-BISECTION-TOLERANCE ERROR IN MY OWN GATE.

    P2 THE RIGHT AMOUNT OF SLACK   PASS.  Against DepMap's own dependency fractions on 1,775 shared
        genes: 11.6% of model knockouts cost more than 1% of growth against 24.3% of genes being
        dependencies in more than 10% of lines; 11.3% vs 18.2%; 11.0% vs 13.2%. Largest gap 12.7 points,
        inside the 15-point gate. The model is somewhat MORE tolerant than real cells, and converges
        with them as the threshold rises.

    P3 THE SLACK IS IN THE RIGHT PLACES   FAIL.  Among the 206 genes costing more than 1%, the model's
        cost ranking agrees with DepMap at Spearman +0.2889 against a |null| 95th percentile of 0.4284 --
        inside the null. The model has about the right AMOUNT of slack and cannot yet say WHICH genes
        should be in the tail.

WHAT THIS CHANGES.  loop_medium's stated conclusion -- "a cell in which 92% of single deletions change
growth is still not a cell" -- is WRONG and is corrected in that file. The correct statement is that
11.7% of deletions cost more than 1% of growth, which is close to the real distribution, and that the
remaining defect is the ordering of the tail rather than its size. That is a materially different and
more tractable problem, and it was only visible because the gate was audited instead of believed.

-> outputs/loop_slack.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
CELL = ROOT / "outputs" / "orphan" / "cell_complete.json"
SEED = 1904
N_SHUFFLE = 200
THRESHOLDS = (1e-9, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1)
BIO_THRESHOLD = 1e-2
GATE_SLACK = 0.50
GATE_SHAPE = 0.15


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("DOES THE CELL HAVE SLACK -- and was the gate that said no asking the right question?")
    say("=" * 100)
    say("  loop_medium's M3 counted a knockout as 'moving growth' if it changed mu by more than 1e-9.")
    say("  That is a floating-point threshold, not a biological one. This audits my own gate first.")

    lm = OUT / "loop_medium.json"
    if not lm.exists():
        say("  loop_medium.json missing -- nothing to audit. Recorded as not testable.")
        return 1
    M = json.load(open(lm))
    S = pd.DataFrame(M["scores"])
    cost = S["medium"].astype(float)
    cost = cost[np.isfinite(cost)]
    say(f"\n  {len(cost):,} scored knockouts from loop 4's defined-medium model")

    # THE MECHANISM OF M3's FAILURE, diagnosed rather than guessed.
    neg = int((cost < 0).sum())
    if neg:
        say(f"\n  DIAGNOSIS: {neg:,} of {len(cost):,} knockouts have a NEGATIVE cost, clustered at "
            f"{float(cost[cost < 0].median()):.3g}.")
        say(f"    A knockout cannot improve growth in this model, so that is arithmetic, not biology.")
        say(f"    loop_medium calibrated the uptake cap by BISECTION and then used the bisection's last")
        say(f"    iterate as mu_WT ({M['mu_medium']:.9f}) instead of re-solving at the final bounds.")
        say(f"    Every unaffected knockout then solved to a very slightly HIGHER mu, so 1 - mu_KO/mu_WT")
        say(f"    came out at about -7e-07 -- and M3's `abs(diff) > 1e-9` test counted all {neg:,} of")
        say(f"    them as 'moving growth'. That is where 92% came from.")
        say(f"    The fix for any future run is one line: re-solve mu_WT at the final bounds, or test")
        say(f"    against a relative threshold rather than an absolute one.")

    # ---- P1: sweep the threshold ---------------------------------------------------------------------
    say(f"\n  P1 THE THRESHOLD SWEEP -- where does the answer change?")
    say(f"    {'cost >':>10}   {'fraction moving growth':>24}")
    sweep = {}
    for t in THRESHOLDS:
        f = float((cost > t).mean())
        sweep[t] = f
        mark = "   <- loop_medium used this" if t == 1e-9 else (
               "   <- 1% of growth, a real effect" if t == BIO_THRESHOLD else "")
        say(f"    {t:>10.0e}   {f:>23.1%}{mark}")
    frac_bio = sweep[BIO_THRESHOLD]
    p1 = bool(frac_bio < GATE_SLACK)
    say(f"\n    at a 1% cost threshold, {frac_bio:.1%} of deletions move growth   "
        f"(gate < {GATE_SLACK:.0%})   P1 {'PASS' if p1 else 'FAIL'}")
    if p1:
        say("    M3's failure WAS an artefact of the cutoff. The model does have slack; the gate was")
        say("    counting numerical dust as biology. loop_medium's conclusion is restated accordingly.")
    else:
        say("    The model really is saturated. M3's conclusion stands as written and the 1e-9 threshold,")
        say("    while badly chosen, was not what produced it.")

    # ---- P2: shape against DepMap --------------------------------------------------------------------
    D = json.load(open(CELL))
    genes = D["genes"]
    dep = {g["name"]: float(g["dep_frac"]) for g in genes
           if isinstance(g.get("dep_frac"), (int, float))}
    S = S[S["sym"].isin(dep)].copy()
    S["dep"] = [dep[s] for s in S["sym"]]
    say(f"\n  P2 SHAPE -- against DepMap's own dependency fractions ({len(S):,} genes with both)")
    say(f"    {'threshold':>12}{'model':>10}{'DepMap':>10}{'gap':>9}")
    gaps = {}
    for t, dt in ((1e-2, 0.10), (5e-2, 0.25), (1e-1, 0.50)):
        fm = float((S["medium"].astype(float) > t).mean())
        fd = float((S["dep"] > dt).mean())
        gaps[t] = abs(fm - fd)
        say(f"    {t:>12.0e}{fm:>10.1%}{fd:>10.1%}{abs(fm-fd):>9.1%}")
    p2 = bool(max(gaps.values()) < GATE_SHAPE)
    say(f"    largest gap {max(gaps.values()):.1%}  (gate < {GATE_SHAPE:.0%})   "
        f"P2 {'PASS' if p2 else 'FAIL'}")

    # ---- P3: are the costly genes the right ones? ----------------------------------------------------
    hit = S[S["medium"].astype(float) > BIO_THRESHOLD]
    if len(hit) >= 30:
        r = float(pd.Series(hit["medium"].astype(float)).corr(pd.Series(hit["dep"]),
                                                              method="spearman"))
        null = np.array([float(pd.Series(hit["medium"].astype(float)).corr(
            pd.Series(rng.permutation(hit["dep"].to_numpy())), method="spearman"))
            for _ in range(N_SHUFFLE)])
        p95 = float(np.percentile(np.abs(null), 95))
        p3 = bool(abs(r) > p95)
        say(f"\n  P3 RANK -- among the {len(hit):,} genes costing more than 1%, does the model rank them")
        say(f"    like DepMap does?  Spearman {r:+.4f} vs |null| 95th pct {p95:.4f}   "
            f"P3 {'PASS' if p3 else 'FAIL'}")
    else:
        r, p95, p3 = float("nan"), float("nan"), False
        say(f"\n  P3 RANK -- only {len(hit)} genes cost more than 1%; too few to rank. P3 FAIL")

    say("\n" + "=" * 100)
    gates = {"P1 the gate was the problem": p1, "P2 the right amount of slack": p2,
             "P3 the slack is in the right places": p3}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if p1 and not p2:
        say("  THE MODEL HAS SLACK AND THE WRONG AMOUNT OF IT. That is a sharper complaint than M3's,")
        say("  and it is actionable: the quantity to fix is the size of the tail, not its existence.")
    elif p1:
        say("  M3 WAS MEASURING NUMERICAL DUST. The model tolerates most single deletions at any")
        say("  biologically meaningful threshold, and loop_medium's conclusion is corrected.")
    else:
        say("  M3 STANDS. The model is genuinely saturated and the badly chosen threshold was not what")
        say("  produced that verdict.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(lm), str(CELL)], available=len(cost), used=len(S),
                      selection="filtered", seed=SEED,
                      controls=["threshold swept over five orders of magnitude",
                                "DepMap dependency distribution as external comparator",
                                f"{N_SHUFFLE} label shuffles for the rank test"],
                      note="audits loop_medium's own M3 gate rather than the repository")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_slack", "manifest": man, "gates": gates,
               "sweep": {str(k): v for k, v in sweep.items()},
               "frac_at_1pct": frac_bio, "shape_gaps": {str(k): v for k, v in gaps.items()},
               "rank_spearman": r, "rank_null_p95": p95, "n_costly": int(len(hit)), "log": log},
              open(OUT / "loop_slack.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_slack.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
