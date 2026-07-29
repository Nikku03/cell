"""WHY 0.6881 AND NOT MORE? Decomposing the gap into label noise, marginal biology, data volume and features.

Four candidate explanations, each with a test that can distinguish it from the others:

  1 LABEL NOISE          the benchmark's negatives include real regulatory pairs the screen missed
  2 MARGINAL BIOLOGY     many positives are small effects near the detection threshold, so they sit in a
                         regime where "regulatory" and "not" are separated by measurement noise, not by
                         anything a model could read off chromatin
  3 DATA VOLUME          569 positives is few; the model is still learning and more pairs would help
  4 FEATURE CEILING      the features carry only so much and no amount of data or capacity extracts more

They imply different next moves, which is the point of separating them. 1 says fix the labels; 2 says the
ceiling is real and 0.6881 is closer to done than it looks; 3 says get more CRISPR data; 4 says get more
feature types.

A NOTE ON THE BENCHMARK'S OWN STRUCTURE, found while checking claim 1. Of the 166 pairs with
PowerAtEffectSize25 < 0.8, ALL 166 are positives. That is not a coincidence -- it is how the benchmark is
assembled: a pair is retained as a testable NEGATIVE only where the screen had power to detect an effect,
while a SIGNIFICANT pair is retained regardless of nominal power. So low power implies positive by
construction. Any feature correlated with statistical power (gene expression, most obviously) can pick up
part of that for free. It is a small stratum (1.6%) but it inflates rather than caps the score, and it
belongs in the ledger next to everything else.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SEEDS = (0, 1, 2)


def build():
    from contact_gate import load_ctcf, contact_features
    from competition import competition_features, load as load_comp
    from ubiquitous import load_ubiquitous
    df, Xid, Cr = load_comp()
    CF = competition_features(df)
    U = np.isin(df["gene"].values, list(load_ubiquitous())).astype(float)[:, None]
    X = np.hstack([Xid, Cr, CF, U])
    return df, X


def bench_cols(df):
    """EffectSize and power, joined on (element, gene) exactly as everything else in this project."""
    d = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            k = (f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", row["measuredGeneSymbol"])
            try:
                es = float(row["EffectSize"])
            except (ValueError, KeyError):
                es = np.nan
            try:
                pw = float(row["PowerAtEffectSize25"])
            except (ValueError, KeyError):
                pw = np.nan
            d[k] = (es, pw)
    out = np.array([d.get((e, g), (np.nan, np.nan)) for e, g in zip(df["element"], df["gene"])])
    assert np.isfinite(out[:, 0]).mean() > 0.95, "EffectSize join failed"
    return out[:, 0], out[:, 1]


def oof(X, y, ch, seed):
    import xgboost as xgb
    from crispr_gate import _seeded_folds
    folds = _seeded_folds(ch, seed)
    p = np.zeros(len(y))
    tr_scores = []
    from sklearn.metrics import average_precision_score
    for k in range(5):
        tr, te = folds != k, folds == k
        c = xgb.XGBClassifier(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), max_depth=4,
                              n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
        c.fit(X[tr], y[tr])
        p[te] = c.predict_proba(X[te])[:, 1]
        tr_scores.append(average_precision_score(y[tr], c.predict_proba(X[tr])[:, 1]))
    return p, float(np.mean(tr_scores))


def main():
    from sklearn.metrics import average_precision_score
    from crispr_gate import _seeded_folds
    import xgboost as xgb

    df, X = build()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    es, pw = bench_cols(df)
    R = {}

    print("=" * 100)
    print("WHY 0.6881 AND NOT MORE?")
    print("=" * 100)

    ps, trs = zip(*[oof(X, y, ch, s) for s in SEEDS])
    p = np.mean(ps, 0)
    full = average_precision_score(y, p)
    print(f"  held-out AUPRC {full:.4f}   train AUPRC {np.mean(trs):.4f}   "
          f"gap {np.mean(trs)-full:+.4f}")
    R["heldout"], R["train"] = float(full), float(np.mean(trs))

    # ---- 2. MARGINAL BIOLOGY: are the misses the small-effect positives? ----
    print("\n  [2] MARGINAL BIOLOGY -- are missed positives the weak-effect ones?")
    order = np.argsort(-p)
    rank = np.empty(len(y), int)
    rank[order] = np.arange(len(y))
    found = (y == 1) & (rank < 500)
    miss = (y == 1) & (rank >= 500)
    print(f"      found (top-500):  n={found.sum():3d}  median |effect| {np.nanmedian(np.abs(es[found])):.4f}")
    print(f"      missed:           n={miss.sum():3d}  median |effect| {np.nanmedian(np.abs(es[miss])):.4f}")
    print(f"      ratio {np.nanmedian(np.abs(es[found]))/np.nanmedian(np.abs(es[miss])):.2f}x stronger among found")
    R["effect_found"] = float(np.nanmedian(np.abs(es[found])))
    R["effect_missed"] = float(np.nanmedian(np.abs(es[miss])))

    print("\n      AUPRC restricted to positives above an effect-size floor (negatives all retained):")
    for thr in (0.0, 0.05, 0.10, 0.15, 0.20):
        keep = (y == 0) | (np.abs(es) >= thr)
        if (y[keep] == 1).sum() < 30:
            continue
        a = average_precision_score(y[keep], p[keep])
        print(f"        |effect| >= {thr:.2f}: {int((y[keep]==1).sum()):3d} positives kept, AUPRC {a:.4f}")
        R[f"auprc_effect_{thr}"] = float(a)

    # ---- 3. DATA VOLUME: learning curve ----
    print("\n  [3] DATA VOLUME -- is the model still learning?")
    folds = _seeded_folds(ch, 0)
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 0.75, 1.0):
        pp = np.zeros(len(y))
        for k in range(5):
            tr, te = np.where(folds != k)[0], folds == k
            sub = rng.choice(tr, int(len(tr) * frac), replace=False)
            c = xgb.XGBClassifier(scale_pos_weight=(y[sub] == 0).sum() / max((y[sub] == 1).sum(), 1),
                                  max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                                  eval_metric="aucpr", n_jobs=4)
            c.fit(X[sub], y[sub])
            pp[te] = c.predict_proba(X[te])[:, 1]
        a = average_precision_score(y, pp)
        print(f"      {int(frac*100):3d}% of training pairs ({int(len(y)*0.8*frac):5,d}): AUPRC {a:.4f}")
        R[f"curve_{frac}"] = float(a)

    # ---- 4. WHERE THE LOSS SITS ----
    print("\n  [4] WHERE THE LOSS SITS -- AUPRC within distance strata")
    dist = 10 ** df["log_dist"].values if df["log_dist"].max() < 20 else df["log_dist"].values
    tssd = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        for row in r:
            k = (f"{row['chrom']}:{row['chromStart']}-{row['chromEnd']}", row["measuredGeneSymbol"])
            try:
                tssd[k] = abs((int(row["chromStart"]) + int(row["chromEnd"])) // 2 - int(row["startTSS"]))
            except ValueError:
                pass
    dist = np.array([tssd.get((e, g), np.nan) for e, g in zip(df["element"], df["gene"])])
    for lo, hi, lab in ((0, 5e4, "<50 kb"), (5e4, 2.5e5, "50-250 kb"),
                        (2.5e5, 1e6, "250kb-1Mb"), (1e6, 1e9, ">1 Mb")):
        m = (dist >= lo) & (dist < hi)
        if (y[m] == 1).sum() < 5:
            continue
        a = average_precision_score(y[m], p[m])
        print(f"      {lab:12s} {m.sum():5,d} pairs, {int(y[m].sum()):3d} pos, base {y[m].mean():.4f} "
              f"-> AUPRC {a:.4f}  ({a/y[m].mean():.1f}x its own base rate)")
        R[f"auprc_{lab}"] = float(a)

    # ---- 1. the power artifact, quantified ----
    lowp = np.isfinite(pw) & (pw < 0.8)
    print(f"\n  [1] BENCHMARK STRUCTURE -- pairs with power<0.8: {int(lowp.sum())}, "
          f"of which positive: {int(y[lowp].sum())} ({100*y[lowp].mean():.0f}%)")
    keep = ~lowp
    a = average_precision_score(y[keep], p[keep])
    print(f"      AUPRC excluding that stratum: {a:.4f} (vs {full:.4f})")
    # THE DROP IS NOT ITSELF EVIDENCE. Removing those 166 pairs removes 166 of 569 positives, so the base
    # rate falls 0.0551 -> 0.0396, and AUPRC is base-rate bound: any 166 positives would lower it. The
    # control removes 166 RANDOM positives, matching the base-rate change exactly, and asks whether the
    # low-power stratum cost MORE than its arithmetic share.
    pos = np.where(y == 1)[0]
    rnd = []
    for s in range(20):
        drop = set(np.random.default_rng(s).choice(pos, int(lowp.sum()), replace=False).tolist())
        k = np.array([i not in drop for i in range(len(y))])
        rnd.append(average_precision_score(y[k], p[k]))
    rnd = np.array(rnd)
    excess = a - rnd.mean()
    print(f"      control, drop 166 RANDOM positives: {rnd.mean():.4f} +/- {rnd.std(ddof=1):.4f} "
          f"(same base rate {y[keep].mean():.4f})")
    print(f"      low-power drop MINUS random drop: {excess:+.4f} "
          f"({excess/rnd.std(ddof=1):+.1f} sd)")
    print(f"      -> that stratum carries {-excess:+.4f} of AUPRC BEYOND its base-rate share: it is "
          f"positive by benchmark construction, and the model can find it from gene expression.")
    R.update({"auprc_excl_lowpower": float(a), "auprc_drop_random_positives": float(rnd.mean()),
              "auprc_drop_random_sd": float(rnd.std(ddof=1)),
              "lowpower_excess_over_baserate": float(excess),
              "lowpower_excess_sd": float(excess / rnd.std(ddof=1))})

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "ceiling.json", "w"), indent=1)
    print(f"\n  -> {OUT/'ceiling.json'}")


if __name__ == "__main__":
    main()
