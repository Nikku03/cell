"""Cascade agreement test: does requiring INDEPENDENT streams to agree lift
precision at expanded coverage, as the synthesis claims?

The load-bearing claim of the whole drug-target synthesis is:
  "two independent ~0.7 estimators agreeing -> ~0.9 precision on the
   agreement subset, because their errors are uncorrelated."

We test it with the two genuinely independent streams we have locally:
  STREAM A = conservation  (family_frac_essential_leakfree, Phase-0 signal)
  STREAM B = metabolic flux (fba_essential, from FBA on a genome-scale model)
  GROUND TRUTH = experimental essential label

corr(A,B) measured at 0.395 -> partly independent, the right regime.

For each stream we pick a threshold, then evaluate four decision rules
against the true essential labels on the joint-covered genes:
  A-alone, B-alone, AGREE (both call essential), UNION (either).

If the synthesis is right:
  - AGREE has HIGHER precision than either stream alone
  - the errors of A and B are largely on DIFFERENT genes (low error overlap)
  - UNION has higher recall (the streams catch different essentials)

If the synthesis is wrong:
  - AGREE precision ~= max(A,B) alone (streams not independent enough)
  - errors overlap heavily (redundant streams)

Output: outputs/cascade_agreement_test.md
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data" / "drive_import" / "labels"


def metrics(y, pred):
    import numpy as np
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    cov = (pred == 1).mean()
    return dict(precision=prec, recall=rec, coverage=cov,
                tp=tp, fp=fp, fn=fn, tn=tn, n_called=tp + fp)


def main() -> int:
    import pandas as pd
    import numpy as np

    lab = pd.read_csv(D / "labels.csv")
    fba = pd.read_csv(D / "fba_features.csv")
    orth = pd.read_csv(D / "orthology_features.csv",
                       usecols=["organism", "locus_tag",
                                "family_frac_essential_leakfree"])
    d = (lab.merge(fba, on=["organism", "locus_tag"], how="left")
            .merge(orth, on=["organism", "locus_tag"], how="left"))
    d = d[(d.fba_in_model == 1) &
          d.fba_essential.notna() &
          d.family_frac_essential_leakfree.notna()].copy()
    y = d.essential.values.astype(int)
    A = d.family_frac_essential_leakfree.values.astype(float)   # conservation
    B = d.fba_essential.values.astype(float)                     # flux (0/1 or prob)
    base = y.mean()
    print(f"joint-covered genes: {len(d):,}  essential base rate: {base:.3f}")
    print(f"corr(conservation, FBA) = {np.corrcoef(A, B)[0,1]:.3f}\n")

    # --- pick thresholds: conservation at the point matching FBA's call rate,
    #     and also sweep to report at matched-precision for fairness ---
    # FBA is essentially binary (essential / not). Use B>=0.5.
    b_pred = (B >= 0.5).astype(int)
    # conservation: sweep threshold, report the rule at a comparable call-rate
    # AND at the threshold that maximises agreement precision.
    print("=== individual streams ===")
    # conservation at a few thresholds
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a_pred = (A >= t).astype(int)
        m = metrics(y, a_pred)
        rows.append(("conservation@%.1f" % t, m))
    mB = metrics(y, b_pred)
    rows.append(("FBA(flux)", mB))
    print(f"  {'rule':<22s} {'prec':>6s} {'recall':>7s} {'cov':>6s} {'n_called':>9s}")
    for name, m in rows:
        print(f"  {name:<22s} {m['precision']:>6.3f} {m['recall']:>7.3f} "
              f"{m['coverage']:>6.3f} {m['n_called']:>9d}")

    # --- the agreement test: pick conservation threshold near FBA precision ---
    # use conservation@0.5 as the comparable operating point
    a_pred = (A >= 0.5).astype(int)
    mA = metrics(y, a_pred)

    agree = ((a_pred == 1) & (b_pred == 1)).astype(int)
    union = ((a_pred == 1) | (b_pred == 1)).astype(int)
    mAgree = metrics(y, agree)
    mUnion = metrics(y, union)

    print(f"\n=== THE AGREEMENT TEST (conservation@0.5 x FBA) ===")
    print(f"  {'rule':<14s} {'prec':>6s} {'recall':>7s} {'cov':>6s} {'n_called':>9s}")
    print(f"  {'conservation':<14s} {mA['precision']:>6.3f} {mA['recall']:>7.3f} "
          f"{mA['coverage']:>6.3f} {mA['n_called']:>9d}")
    print(f"  {'FBA':<14s} {mB['precision']:>6.3f} {mB['recall']:>7.3f} "
          f"{mB['coverage']:>6.3f} {mB['n_called']:>9d}")
    print(f"  {'AGREE (both)':<14s} {mAgree['precision']:>6.3f} {mAgree['recall']:>7.3f} "
          f"{mAgree['coverage']:>6.3f} {mAgree['n_called']:>9d}   <-- should be HIGHEST precision")
    print(f"  {'UNION (either)':<14s} {mUnion['precision']:>6.3f} {mUnion['recall']:>7.3f} "
          f"{mUnion['coverage']:>6.3f} {mUnion['n_called']:>9d}   <-- should be HIGHEST recall")

    # --- error-independence: do A and B err on DIFFERENT genes? ---
    a_err = (a_pred != y)
    b_err = (b_pred != y)
    both_err = (a_err & b_err).sum()
    a_only = (a_err & ~b_err).sum()
    b_only = (b_err & ~a_err).sum()
    # if errors were independent: P(both) = P(a)*P(b)*n
    exp_both = a_err.mean() * b_err.mean() * len(y)
    print(f"\n=== ERROR INDEPENDENCE ===")
    print(f"  conservation errors: {a_err.sum()}   FBA errors: {b_err.sum()}")
    print(f"  errors on SAME gene (both wrong): {both_err}")
    print(f"  expected if independent: {exp_both:.0f}")
    print(f"  -> {'errors are ~independent (agreement helps)' if both_err <= 1.4*exp_both else 'errors overlap (agreement helps less)'}")
    print(f"  conservation-only errors: {a_only}  FBA-only errors: {b_only}")
    print(f"  (high A-only + B-only = the streams fail on DIFFERENT genes = "
          f"agreement filters them out)")

    # --- verdict ---
    lift = mAgree["precision"] - max(mA["precision"], mB["precision"])
    print(f"\n=== VERDICT ===")
    print(f"  best single-stream precision: {max(mA['precision'], mB['precision']):.3f}")
    print(f"  agreement precision:          {mAgree['precision']:.3f}")
    print(f"  precision LIFT from agreement: {lift:+.3f}")
    if lift > 0.05:
        print(f"  *** AGREEMENT LIFTS PRECISION -- the cascade principle HOLDS ***")
        print(f"  (at the cost of lower recall/coverage -- the abstain trade)")
    elif lift > 0.0:
        print(f"  marginal lift -- streams partly independent")
    else:
        print(f"  NO lift -- streams too correlated OR one dominates; "
              f"cascade buys nothing here")

    # --- write report ---
    lines = ["# Cascade agreement test: conservation x FBA\n",
             f"{len(d):,} genes (2 organisms with FBA models), essential base "
             f"rate {base:.3f}, corr(streams)={np.corrcoef(A,B)[0,1]:.3f}.\n",
             "## Decision rules vs true essentiality\n",
             "| rule | precision | recall | coverage | n_called |",
             "|---|---:|---:|---:|---:|"]
    for name, m in [("conservation@0.5", mA), ("FBA", mB),
                    ("AGREE (both)", mAgree), ("UNION (either)", mUnion)]:
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['coverage']:.3f} | {m['n_called']} |")
    lines.append(f"\n## Error independence\n")
    lines.append(f"- conservation errors: {a_err.sum()}, FBA errors: {b_err.sum()}")
    lines.append(f"- both-wrong-on-same-gene: {both_err} (expected if independent: {exp_both:.0f})")
    lines.append(f"- conservation-only errors: {a_only}, FBA-only errors: {b_only}")
    lines.append(f"\n## Verdict\n")
    lines.append(f"- precision lift from agreement: **{lift:+.3f}**")
    lines.append(f"- agreement precision {mAgree['precision']:.3f} vs best single "
                 f"{max(mA['precision'],mB['precision']):.3f}")
    (REPO / "outputs" / "cascade_agreement_test.md").write_text("\n".join(lines))
    print(f"\nwrote outputs/cascade_agreement_test.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
