"""PREDICTING A QUANTITY -- the structural gap, tested on the loop's own equation.

WHERE THIS COMES FROM.  loop_statement's third limit, written after twenty loops: "Nothing here predicts
a QUANTITY. Every passing row separates classes -- driver from not, cytotoxic from not, high torsion from
low. None of them says how much." That is a structural complaint about the whole project, not about one
module, and it has never been tested. This tests it.

WHAT IS PREDICTED, and it is deliberately the model's OWN equation rather than a convenient target.
cell_loop's protein balance is

        dP/dt = k_tl * m - (k_dp + mu) * P     ->     P = k_tl * m / (k_dp + mu)

Every term is measurable here. m is mRNA abundance (RNADecayCafe RPKM), k_dp is the declared uniform
protein degradation constant, mu is the growth rate, and P is protein abundance (cell_complete's `ppm`,
16,015 genes). So the loop makes a QUANTITATIVE, falsifiable claim about a number nobody has checked:
protein abundance should be proportional to mRNA abundance, with the constant of proportionality fixed.

THE SCORE IS R-SQUARED ON HELD-OUT GENES, not AUC. A class separation cannot fail the way a quantity
can: predicting the mean gives R-squared of exactly zero, and a bad model gives a NEGATIVE one. That is
a much harder bar than 0.5 on an AUC and it is the point of running this.

THE THREE THINGS THIS CAN DISTINGUISH, which the class-separation work could not:
    the SHAPE is right     mRNA predicts protein at all
    the SCALE is right     the fitted slope is near 1 in log space, i.e. P really is PROPORTIONAL to m
                           rather than merely correlated with it
    the CORRECTION helps   dividing by (k_dp + mu) and by (k_deg + mu), which is the loop's actual
                           content, beats using raw mRNA

PREDECLARED, before any number:

    Y1 BEATS THE MEAN   held-out R-squared > 0.05 for log protein from log mRNA, 5-fold CV.
                fails -> mRNA does not predict protein here at all and the loop's protein balance has
                no quantitative content in this data.
    Y2 THE SLOPE IS NEAR 1   the fitted log-log slope lies in [0.7, 1.3].
                fails -> protein is not proportional to mRNA; the relationship is real but the balance
                equation's FORM is wrong, which is a sharper finding than a poor R-squared.
    Y3 THE CORRECTION EARNS ITS PLACE   adding the loop's turnover terms improves held-out R-squared by
                >= 0.02 over raw mRNA alone.
                fails -> the equation's distinctive content -- that turnover and dilution set the ratio
                -- contributes nothing, and what remains is "protein tracks mRNA", which needed no model.
    Y4 BEATS A SHUFFLE   R-squared above the 95th percentile of 200 gene shuffles.

CONTROLS: 200 gene-label shuffles; the mean as the zero-R-squared baseline; raw mRNA as the comparator
for Y3; and the residual checked against protein abundance itself, because a model that is right only
for abundant proteins is right about detection, not biology.

WHAT HAPPENED, written after the run against the gates above, unedited.

    Y1 BEATS THE MEAN   PASS.  Held-out R-squared 0.3783 for log protein from log mRNA over 10,136
        genes. The first held-out R-squared in this project; twenty loops of AUCs came before it.
    Y2 THE SLOPE IS NEAR 1   PASS, AND BY 0.0023.  The fitted log-log slope is 1.2977 against a declared
        ceiling of 1.3. That is not a vindication of proportionality -- it means a tenfold change in
        transcript goes with a TWENTYFOLD change in protein. Protein AMPLIFIES mRNA differences here,
        which is a real and well-documented effect, and my band was simply wide enough to admit it.
        The honest reading is that the balance equation's form is approximately right and measurably
        not exactly right.
    Y3 THE CORRECTION EARNS ITS PLACE   PASS.  0.3783 -> 0.4332, a gain of +0.0549 against a +0.02 gate.
        THIS IS THE FIRST TIME THE LOOP'S DISTINCTIVE CONTENT HAS ADDED ANYTHING TO ANYTHING. In loop 1
        the feedback bought -0.0074 AUC on essentiality; here the same turnover-and-dilution term buys
        +0.055 R-squared on protein abundance. Same mechanism, different question, opposite outcome.
    Y4 BEATS A SHUFFLE   PASS.  Null R-squared -0.0003 +/- 0.0003, 95th percentile 0.0000, against
        0.3783. R-squared is a bar a shuffle cannot clear by luck the way an AUC sometimes can.

    residual check: |error| against protein abundance, Spearman -0.0460 -- essentially flat, so the fit
    is not carried by abundant, easy-to-detect proteins.

WHY THIS MATTERS MORE THAN THE NUMBER.  Twenty loops produced class separations, and loop_statement
recorded that as a structural limit. It was testable and it is now tested: the model does predict a
quantity, at R-squared 0.43, and the part of it that is genuinely the LOOP -- dividing by (k_deg + mu)
and (k_dp + mu) -- is what took it from 0.38 to 0.43. The mechanism that failed to earn its place
against essentiality earns it against abundance. That is a statement about which QUESTION suits the
model, which is the same lesson loops 12 and 18 taught with side effects and cytotoxicity.

-> outputs/loop_quantity.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from cell_loop import KDEG, CELL, K_DP, MIN_LINES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SEED = 1904
N_SHUFFLE = 200
MU = 0.030
GATE_R2 = 0.05
GATE_SLOPE = (0.7, 1.3)
GATE_ADDS = 0.02


def cv_r2(X, y, folds=5, seed=SEED):
    """Out-of-fold R-squared. Predicting the mean scores exactly 0; a bad model scores negative."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    oof = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        oof[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    ss_res = float(np.sum((y - oof) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot, oof


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("PREDICTING A QUANTITY -- the structural gap, tested on the loop's own equation")
    say("=" * 100)
    say("  After twenty loops nothing here predicts a number, only a class. AUC cannot fail the way")
    say("  R-squared can: the mean scores exactly zero and a bad model scores negative.")

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
        if p is None or p <= 0 or s not in rp.index or s not in kd.index:
            continue
        rows.append({"sym": s, "protein": float(p), "mrna": float(rp.rpkm[s]),
                     "kdeg": float(kd.kdeg[s])})
    T = pd.DataFrame(rows)
    say(f"\n  {len(T):,} genes with measured protein abundance, mRNA abundance and mRNA decay rate")

    y = np.log10(T.protein.to_numpy())
    m = np.log10(T.mrna.to_numpy())
    # the loop's own prediction: P = k_tl * k_syn / ((k_deg + mu)(k_dp + mu)), and k_syn = m (k_deg + mu),
    # so P = k_tl * m / (k_dp + mu). With uniform k_tl and k_dp that is a CONSTANT times m -- which is
    # why Y3 has to test something the constant cannot absorb.
    turnover = np.log10(1.0 / ((T.kdeg.to_numpy() + MU) * (K_DP + MU)))

    r2_raw, oof_raw = cv_r2(m, y)
    r2_full, oof_full = cv_r2(np.c_[m, turnover], y)
    slope = float(np.polyfit(m, y, 1)[0])
    say(f"\n  {'model':<44}{'held-out R2':>13}")
    say(f"  {'the mean (baseline, exactly zero by construction)':<44}{0.0:>13.4f}")
    say(f"  {'log mRNA alone':<44}{r2_raw:>13.4f}")
    say(f"  {'log mRNA + the loop turnover term':<44}{r2_full:>13.4f}")
    say(f"\n  fitted log-log slope: {slope:.4f}   (1.0 would mean protein is strictly proportional "
        f"to mRNA)")

    y1 = bool(r2_raw > GATE_R2)
    say(f"\n  Y1 BEATS THE MEAN -- {r2_raw:.4f} > {GATE_R2}   Y1 {'PASS' if y1 else 'FAIL'}")
    y2 = bool(GATE_SLOPE[0] <= slope <= GATE_SLOPE[1])
    say(f"  Y2 THE SLOPE IS NEAR 1 -- {slope:.4f} in {GATE_SLOPE}   Y2 {'PASS' if y2 else 'FAIL'}")
    if not y2:
        say(f"    Protein is NOT proportional to mRNA in this data: a tenfold change in transcript goes")
        say(f"    with a {10**slope:.1f}-fold change in protein. The relationship is real and the")
        say(f"    balance equation's FORM -- P proportional to m -- is wrong, which is a sharper and")
        say(f"    more useful finding than a poor R-squared would have been.")
    d3 = r2_full - r2_raw
    y3 = bool(d3 >= GATE_ADDS)
    say(f"  Y3 THE CORRECTION EARNS ITS PLACE -- {r2_full:.4f} - {r2_raw:.4f} = {d3:+.4f}   "
        f"Y3 {'PASS' if y3 else 'FAIL'}")
    if not y3:
        say(f"    Dividing by turnover and dilution -- the loop's distinctive content -- adds nothing.")
        say(f"    What survives is 'protein tracks mRNA', which needed no model to say.")

    null = np.array([cv_r2(m, rng.permutation(y))[0] for _ in range(N_SHUFFLE)])
    p95 = float(np.percentile(null, 95))
    y4 = bool(r2_raw > p95)
    say(f"  Y4 BEATS A SHUFFLE -- null {null.mean():+.4f} +/- {null.std():.4f}, 95th {p95:+.4f}   "
        f"Y4 {'PASS' if y4 else 'FAIL'}")

    # is the model only right about abundant proteins?
    resid = y - oof_raw
    r_ab = float(pd.Series(np.abs(resid)).corr(pd.Series(y), method="spearman"))
    say(f"\n  residual check: |error| vs protein abundance, Spearman {r_ab:+.4f}")
    if r_ab < -0.2:
        say("    Errors shrink as protein abundance rises, so the fit is carried by abundant proteins.")
        say("    A model that is right about what is easy to detect is right about detection.")

    say("\n" + "=" * 100)
    gates = {"Y1 beats the mean": y1, "Y2 slope near 1": y2,
             "Y3 correction earns its place": y3, "Y4 beats a shuffle": y4}
    for kk, vv in gates.items():
        say(f"  {kk:<36}{'PASS' if vv else 'FAIL'}")
    if y1 and y4 and not y2:
        say("  A QUANTITY IS PREDICTED, AND THE EQUATION'S FORM IS WRONG.  mRNA predicts protein well")
        say(f"  above chance, so this is the first held-out R-squared in the project. But the slope is")
        say(f"  {slope:.2f}, not 1, so protein is NOT proportional to transcript -- the loop's protein")
        say("  balance is the wrong shape, and that is a concrete, fixable defect rather than a vague")
        say("  shortfall.")
    elif y1 and y2 and y4:
        margin = min(slope - GATE_SLOPE[0], GATE_SLOPE[1] - slope)
        say("  A QUANTITY IS PREDICTED. First held-out R-squared in this project, and the first time the")
        say("  loop's distinctive content -- turnover and dilution -- has added anything to anything.")
        if margin < 0.05:
            say(f"  BUT READ Y2 CAREFULLY: the slope is {slope:.4f} against a ceiling of "
                f"{GATE_SLOPE[1]}, which is {margin:.4f} from failing. A slope of {slope:.2f} means a")
            say(f"  tenfold change in transcript goes with a {10**slope:.0f}-fold change in protein.")
            say("  That is AMPLIFICATION, not proportionality. The gate passed because the band was")
            say("  wide, not because the equation's form was vindicated, and calling this")
            say("  'proportional' would be the overstatement this project exists to avoid.")
    elif not y1:
        say("  NO QUANTITY IS PREDICTED. mRNA does not predict protein above the mean here, so the")
        say("  protein balance has no quantitative content in this data. Recorded.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(CELL), str(KDEG)], available=len(genes), used=len(T),
                      selection="filtered", seed=SEED,
                      controls=[f"{N_SHUFFLE} gene-label shuffles", "the mean as a zero-R2 baseline",
                                "raw mRNA as the comparator for the turnover term",
                                "residual checked against protein abundance"],
                      note="scored by held-out R-squared, which the mean fails at exactly 0 and a bad "
                           "model fails below it")
    RM.report(man, emit=say)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"test": "loop_quantity", "manifest": man, "gates": gates,
               "r2_mrna": r2_raw, "r2_with_turnover": r2_full, "slope": slope,
               "null": {"mean": float(null.mean()), "sd": float(null.std()), "p95": p95},
               "resid_vs_abundance": r_ab, "n": len(T), "log": log},
              open(OUT / "loop_quantity.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_quantity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
