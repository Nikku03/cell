"""CROSS-CELL-TYPE ENHANCER-GENE EVALUATION: does the K562 contact verdict survive where it never trained?

(Named crosscell_eg to avoid colliding with crosscell.py, which is the Perturb-seq cross-cell-line work.)

TWO DESIGNS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND ONLY ONE IS OPEN.

    POOLED   train on all cell types minus a chromosome-grouped fold, test on that fold. The model HAS
             seen every test cell type. Answers: how good is this on cell types we have data for?

    LOCO     leave-one-cell-type-out: train on the others, test on the one held out entirely. The model has
             NEVER seen the test cell type. Answers: does it transfer to a NEW cell type?

The open question is the second. Every entry in the contact ledger -- Rouse +0.005, extrusion -0.0027,
measured Hi-C -0.0031, Borzoi +0.0054 -- was measured inside K562, and `data_scarce.py` showed contact's
value GROWS as K562's 311 TF tracks are stripped. Pooled CV cannot test that: a model that saw WTC11 in
training is not being asked to transfer to WTC11.

Pooled is not wasted. **POOLED minus LOCO is the transfer penalty** -- the cost of cell-type novelty, never
measured in this project, and the number that decides whether the model is usable outside K562 at all.

LOCO also delivers what pooling was wanted for: training on three cell types instead of one uses the
diversity, while holding the fourth out entirely keeps the property that makes the answer mean anything.

LEAKAGE, CHECKED RATHER THAN ASSUMED. The Engreitz split is disjoint where it matters: 0 shared
(element, gene) pairs and 0 shared elements between the K562 training file and the non-K562 held-out rows.
But 54 of the 131 non-K562 held-out genes DO appear in the training file (41%). Gene-level features --
expression, competition, candidate counts -- would let a model memorise "this gene is easy". Chromosome-
grouped folds subsume gene grouping, since a gene sits on one chromosome. Stated because it is not obvious
and would be invisible if wrong.

ONE FEATURE SPACE FOR ALL CELL TYPES. Features come from narrowPeak files, K562 included. Reusing K562's
bigWig-derived signal for training while held-out cell types get peak-derived features would make any
transfer number a measurement of the feature mismatch, so K562 is rebuilt here rather than imported from
crispr_features_compendium.csv.

SAMPLE SIZES ARE SMALL AND PRINTED, NOT SMOOTHED OVER. GM12878 has 68 pairs and 21 positives; HCT116 396
and 40; WTC11 1,921 and 35. A LOCO score on 68 pairs carries an enormous interval and is reported with its
positive count so it cannot be read as equivalent to the others.
"""
import bisect
import collections
import csv
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CT = SP / "celltypes"
CR = SP / "crispr"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
SEEDS = (0, 1, 2, 3, 4)
NEAR = 500          # bp: a peak this close to a point counts as overlapping it


def load_peaks(cell, name):
    """chrom -> (sorted starts, ends, signalValue). narrowPeak column 7 is signalValue."""
    p = CT / cell / f"{name}.bed.gz"
    if not p.exists():
        return None
    d = collections.defaultdict(list)
    with gzip.open(p, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            try:
                d[f[0]].append((int(f[1]), int(f[2]), float(f[6]) if len(f) > 6 else 1.0))
            except ValueError:
                continue
    out = {}
    for c, v in d.items():
        v.sort()
        out[c] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                  np.array([x[2] for x in v]))
    return out or None


def load_loops(cell):
    p = CT / cell / "hic_loops.bedpe.gz"
    if not p.exists():
        return None
    d = collections.defaultdict(list)
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0] != f[3]:
                continue
            try:
                x = (int(f[1]) + int(f[2])) // 2
                y = (int(f[4]) + int(f[5])) // 2
            except ValueError:
                continue
            d[f[0]].append((min(x, y), max(x, y)))
    return {c: np.array(sorted(v), float) for c, v in d.items() if v} or None


def point_signal(pk, chrom, pos):
    """(overlaps a peak, log1p signal of that peak, log10 distance to the nearest peak)."""
    if pk is None or chrom not in pk:
        return 0.0, 0.0, 6.0
    s, e, v = pk[chrom]
    i = bisect.bisect_right(s, pos) - 1
    best_d, best_v, hit = 1e9, 0.0, 0.0
    for j in (i, i + 1):
        if 0 <= j < len(s):
            if s[j] - NEAR <= pos <= e[j] + NEAR:
                hit, best_v, best_d = 1.0, max(best_v, v[j]), 0.0
            else:
                best_d = min(best_d, abs(pos - s[j]), abs(pos - e[j]))
    return hit, float(np.log1p(best_v)), float(np.log10(max(best_d, 1.0)))


def between_count(pk, chrom, lo, hi):
    """(peaks strictly between, log1p summed signal) -- the CTCF-between feature, per cell type."""
    if pk is None or chrom not in pk:
        return 0.0, 0.0
    s, e, v = pk[chrom]
    a, b = bisect.bisect_right(s, lo), bisect.bisect_left(s, hi)
    if b <= a:
        return 0.0, 0.0
    return float(b - a), float(np.log1p(v[a:b].sum()))


def loop_connects(lp, chrom, lo, hi, tol=25_000):
    if lp is None or chrom not in lp:
        return 0.0
    a = lp[chrom]
    m = ((np.abs(a[:, 0] - lo) < tol) & (np.abs(a[:, 1] - hi) < tol)) | \
        ((np.abs(a[:, 0] - hi) < tol) & (np.abs(a[:, 1] - lo) < tol))
    return float(m.any())


FEATNAMES = ["log_dist",
             "acc_elem_hit", "acc_elem_sig", "acc_elem_dist",
             "k27_elem_hit", "k27_elem_sig", "k27_elem_dist",
             "pol_elem_hit", "pol_elem_sig", "pol_elem_dist",
             "acc_tss_hit", "acc_tss_sig", "acc_tss_dist",
             "pol_tss_hit", "pol_tss_sig", "pol_tss_dist",
             "n_candidates", "dist_rank", "dist_excess", "act_share"]
CONTACT_FEATS = ["ctcf_between", "ctcf_between_sig", "ctcf_elem_hit", "ctcf_tss_hit", "loop_connects"]


def build(rows, cell, tracks):
    acc, k27 = tracks[cell]["acc"], tracks[cell]["k27"]
    pol, ctcf, lp = tracks[cell]["pol"], tracks[cell]["ctcf"], tracks[cell]["loops"]
    n = len(rows)
    X = np.zeros((n, len(FEATNAMES)))
    C = np.zeros((n, len(CONTACT_FEATS)))
    act = np.zeros(n)
    for i, r in enumerate(rows):
        ch = r["chrom"]
        em = (int(r["chromStart"]) + int(r["chromEnd"])) // 2
        t = int(r["startTSS"])
        lo, hi = (em, t) if em <= t else (t, em)
        X[i, 0] = np.log10(max(abs(t - em), 1))
        X[i, 1:4] = point_signal(acc, ch, em)
        X[i, 4:7] = point_signal(k27, ch, em)
        X[i, 7:10] = point_signal(pol, ch, em)
        X[i, 10:13] = point_signal(acc, ch, t)
        X[i, 13:16] = point_signal(pol, ch, t)
        act[i] = X[i, 2] * X[i, 5]
        C[i, 0], C[i, 1] = between_count(ctcf, ch, lo, hi)
        C[i, 2] = point_signal(ctcf, ch, em)[0]
        C[i, 3] = point_signal(ctcf, ch, t)[0]
        C[i, 4] = loop_connects(lp, ch, lo, hi)
    # competition computed WITHIN cell type: a gene's candidate set is cell-type specific
    genes = np.array([r["measuredGeneSymbol"] for r in rows])
    for g in np.unique(genes):
        m = np.where(genes == g)[0]
        d, a = X[m, 0], act[m]
        X[m, 16] = len(m)
        X[m, 17] = np.argsort(np.argsort(d)) + 1
        X[m, 18] = d - d.min()
        X[m, 19] = a / max(a.sum(), 1e-9)
    return X, C


def main():
    from sklearn.metrics import average_precision_score
    from crispr_gate import _seeded_folds
    import xgboost as xgb

    def rd(f):
        return list(csv.DictReader(gzip.open(CR / f, "rt"), delimiter="\t"))

    tr, ho = rd(TRAINING), rd(HELDOUT)
    for r in tr:
        r["CellType"] = "K562"
    allrows = tr + ho
    print("=" * 100)
    print("CROSS-CELL-TYPE ENHANCER-GENE EVALUATION -- pooled vs leave-one-cell-type-out")
    print("=" * 100)

    tracks, usable = {}, []
    for c in sorted({r["CellType"] for r in allrows}):
        t = {"acc": load_peaks(c, "accessibility") or load_peaks(c, "accessibility_atac"),
             "k27": load_peaks(c, "h3k27ac"), "pol": load_peaks(c, "polr2a"),
             "ctcf": load_peaks(c, "ctcf"), "loops": load_loops(c)}
        tracks[c] = t
        n = sum(1 for r in allrows if r["CellType"] == c)
        p = sum(1 for r in allrows if r["CellType"] == c
                and r["Significant"] in ("TRUE", "True", "true"))
        ok = all(t[k] for k in ("acc", "k27", "pol", "ctcf"))
        print(f"  {c:9s} {n:6,d} pairs {p:4d} pos   have: "
              f"{','.join(k for k, v in t.items() if v) or 'NONE':34s} "
              f"{'USABLE' if ok else 'SKIPPED (missing 1D tracks)'}")
        if ok:
            usable.append(c)

    Xs, Cs, ys, chs, cts = [], [], [], [], []
    for c in usable:
        rc = [r for r in allrows if r["CellType"] == c]
        Xc, Cc = build(rc, c, tracks)
        Xs.append(Xc); Cs.append(Cc)
        ys += [1 if r["Significant"] in ("TRUE", "True", "true") else 0 for r in rc]
        chs += [r["chrom"] for r in rc]
        cts += [c] * len(rc)
    X, C = np.vstack(Xs), np.vstack(Cs)
    y, ch, ct = np.array(ys), np.array(chs), np.array(cts)
    print(f"\n  {len(y):,} pairs x {X.shape[1]} base + {C.shape[1]} contact features, "
          f"{int(y.sum())} positives, base rate {y.mean():.4f}")

    def fit_predict(Xtr, ytr, Xte):
        m = xgb.XGBClassifier(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1), max_depth=4,
                              n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

    R = {"n_pairs": int(len(y)), "n_positives": int(y.sum()), "cells": usable,
         "pooled": {}, "loco": {}}

    print("\n  POOLED (model HAS seen every test cell type; chromosome-grouped folds)")
    for tag, M in (("base", X), ("base+contact", np.hstack([X, C]))):
        sc = []
        for s in SEEDS:
            f = _seeded_folds(ch, s)
            p = np.zeros(len(y))
            for k in range(5):
                p[f == k] = fit_predict(M[f != k], y[f != k], M[f == k])
            sc.append(average_precision_score(y, p))
        R["pooled"][tag] = float(np.mean(sc))
        print(f"    {tag:16s} {np.mean(sc):.4f}  [{' '.join(f'{x:.3f}' for x in sc)}]")
    R["pooled"]["delta_contact"] = R["pooled"]["base+contact"] - R["pooled"]["base"]
    print(f"    {'contact delta':16s} {R['pooled']['delta_contact']:+.4f}")

    print("\n  LOCO (model has NEVER seen the test cell type)")
    print(f"    {'test cell':10s} {'pairs':>6s} {'pos':>5s} {'base rate':>10s} {'base':>8s} "
          f"{'+contact':>9s} {'delta':>8s}")
    XC = np.hstack([X, C])
    for c in usable:
        te = ct == c
        if y[te].sum() < 5:
            print(f"    {c:10s} {te.sum():6,d} {int(y[te].sum()):5d}   too few positives, skipped")
            continue
        ab = average_precision_score(y[te], fit_predict(X[~te], y[~te], X[te]))
        ac = average_precision_score(y[te], fit_predict(XC[~te], y[~te], XC[te]))
        R["loco"][c] = {"n": int(te.sum()), "pos": int(y[te].sum()), "base_rate": float(y[te].mean()),
                        "base": float(ab), "plus_contact": float(ac), "delta": float(ac - ab)}
        print(f"    {c:10s} {te.sum():6,d} {int(y[te].sum()):5d} {y[te].mean():10.4f} {ab:8.4f} "
              f"{ac:9.4f} {ac-ab:+8.4f}")

    if R["loco"]:
        lb = float(np.mean([v["base"] for v in R["loco"].values()]))
        ld = float(np.mean([v["delta"] for v in R["loco"].values()]))
        R.update({"loco_mean_base": lb, "loco_mean_delta": ld,
                  "transfer_penalty": R["pooled"]["base"] - lb})
        print(f"\n  LOCO mean base {lb:.4f}   LOCO mean contact delta {ld:+.4f}")
        print(f"  TRANSFER PENALTY (pooled - LOCO): {R['transfer_penalty']:+.4f}")
        print("    ^ the cost of the test cell type being novel, which pooled CV cannot show")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "crosscell_eg.json", "w"), indent=1)
    print(f"\n  -> {OUT/'crosscell_eg.json'}")


if __name__ == "__main__":
    main()
