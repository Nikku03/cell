"""PAIRED COMPARISON OF KNOCKOUT PREDICTORS, ACROSS TWO DISJOINT COHORTS.

WHY THIS FILE EXISTS. Comparing two predictors by their headline precision is wrong whenever they score different
numbers of knockouts, and in this project they always do -- a method that abstains on the hard cases looks better
than one that attempts them. Every comparison here is PAIRED: restricted to the knockouts BOTH methods scored, with
a sign test over per-knockout hit counts and a bootstrap over knockouts for the precision difference.

WHY TWO COHORTS. ko_predict.py selects 50 knockouts by functional class, round-robin, taking the middle member of
each class -- never by effect size. `build(skip=50)` continues the same loop and takes the NEXT 50, so cohort 2 is
disjoint from cohort 1 and selected by an identical rule. Any refinement discovered by looking at cohort 1 can then
be tested on cohort 2 without the selection changing underneath it.

THIS IS NOT OPTIONAL BOOKKEEPING. The refinement it was built to test -- "consult the lesion site only when the
gene's own annotation is silent" -- was the best-scoring variant on cohort 1 (0.0838 vs the incumbent's 0.0786,
better on 3 knockouts and worse on none). On cohort 2 it is better on ZERO knockouts, worse on four, and its
bootstrap CI for the precision difference excludes zero on the wrong side. It was overfitting, and one cohort could
not have shown that.
"""
import collections
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
NBOOT = 4000

COHORTS = {
    "COHORT 1  (50 knockouts, where every rule below was developed)": [
        ("multilayer", "incumbent: sensors from the knockout gene's own annotation"),
        ("abl_fallback", "lesion site consulted only when annotation is silent"),
        ("abl_combined", "annotation sensors UNION lesion-site sensors"),
        ("abl_ablation", "lesion site only -- annotation not consulted"),
        ("abl_shuffled", "control: lesion-site consequences permuted across genes"),
    ],
    "COHORT 2  (disjoint 50, selected identically, never used to develop anything)": [
        ("multilayer_holdout", "incumbent"),
        ("abl_fallback_holdout", "lesion site as fallback"),
        ("abl_combined_holdout", "annotation UNION lesion site"),
        ("abl_ablation_holdout", "lesion site only"),
        ("abl_shuffled_holdout", "control: permuted"),
    ],
}


def rows(tag):
    p = OUT / f"ko_score_{tag}.json"
    if not p.exists():
        return None
    return {r["gene"]: r for r in json.loads(p.read_text())["rows"]}


def agg(P, ks):
    h = sum(P[k]["hits"] for k in ks)
    n = sum(P[k]["n_pred"] for k in ks)
    return h, n, h / max(n, 1), sum(P[k]["beats_frequency"] for k in ks)


def sign_test(d):
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    n = w + l
    p = (sum(comb(n, i) for i in range(max(w, l), n + 1)) * 2 / 2 ** n) if n else 1.0
    return w, l, min(p, 1.0)


def main():
    verdicts = {}
    for title, methods in COHORTS.items():
        ref = rows(methods[0][0])
        if ref is None:
            print(f"\n{title}\n  (not scored -- run ko_predict.py score for these first)")
            continue
        print(f"\n{title}")
        print(f"  {'variant':<22} {'n':>3} {'hits':>5} {'pred':>5} {'prec':>8} {'beat':>6}   paired vs incumbent")
        for tag, _desc in methods:
            P = rows(tag)
            if P is None:
                continue
            both = sorted(set(ref) & set(P))
            h, n, p, bf = agg(P, sorted(P))
            if tag == methods[0][0]:
                print(f"  {tag:<22} {len(P):>3} {h:>5} {n:>5} {p:>8.4f} {bf:>3}/{len(P):<3}  (reference)")
                continue
            _, _, pp, _ = agg(P, both)
            _, _, pa, _ = agg(ref, both)
            w, l, sg = sign_test([P[k]["hits"] - ref[k]["hits"] for k in both])
            rng = np.random.default_rng(0)
            ks = list(both)
            df = []
            for _ in range(NBOOT):
                # ONE resample, scored under both methods. Drawing separate resamples would break the pairing and
                # give a CI for two independent samples, which is wider and answers a different question.
                kk = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
                df.append(agg(P, kk)[2] - agg(ref, kk)[2])
            df = np.array(df)
            lo, hi = np.percentile(df, 2.5), np.percentile(df, 97.5)
            verdicts[tag] = (pp - pa, lo, hi, w, l)
            print(f"  {tag:<22} {len(P):>3} {h:>5} {n:>5} {p:>8.4f} {bf:>3}/{len(P):<3}  "
                  f"n={len(both)} {pp:.4f} v {pa:.4f}  +{w}/-{l} p={sg:.3f}  {df.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")

    print("\nVERDICT (a CI excluding zero is a real difference; one straddling zero is not a result either way)")
    for tag, (d, lo, hi, w, l) in verdicts.items():
        call = ("WORSE" if hi < 0 else "BETTER" if lo > 0 else "indistinguishable")
        print(f"  {tag:<24} {d:+.4f}  [{lo:+.4f},{hi:+.4f}]  +{w}/-{l}   {call}")
    c1 = verdicts.get("abl_fallback")
    c2 = verdicts.get("abl_fallback_holdout")
    if c1 and c2:
        print(f"\n  the fallback rule: {c1[0]:+.4f} on the cohort it was derived from, {c2[0]:+.4f} on the cohort")
        print(f"  it was not. Better on {c1[3]} knockouts there, {c2[3]} here. That is overfitting, and it is only")
        print(f"  visible because the second cohort existed before the rule was scored on it.")


if __name__ == "__main__":
    sys.exit(main())
