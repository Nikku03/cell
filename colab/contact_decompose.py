"""WHICH PART OF THE +0.0340 IS IT? CTCF counting (one ChIP track) or measured Hi-C loops (a library)?

`crosscell_eg.py` found contact worth +0.0340 pooled and +0.0774/+0.0771 LOCO on HCT116/K562 once the 311
K562 TF tracks are gone -- which overturned the K562-internal null. But the contact block is five features
at once:

    ctcf_between        peaks strictly between element and TSS      \\
    ctcf_between_sig    their summed signal                          |  ONE ChIP experiment, then a bisect
    ctcf_elem_hit       a peak at the element                        |
    ctcf_tss_hit        a peak at the promoter                      /
    loop_connects       a called Hi-C loop joining the two          <- a sequencing library

Those differ in cost by orders of magnitude. The recommendation for any new cell type turns entirely on
which one carries the gain, so quoting +0.0340 for "contact" is not actionable.

THE DECIDING QUESTION is not "is Hi-C worth something" but **is Hi-C worth anything ON TOP OF CTCF
counting**. If CTCF carries most of it, a new cell type needs one CTCF ChIP track and no Hi-C at all.

GROUP ABLATION, NOT DROP-ONE. The four CTCF features are near-duplicates -- between-count and
between-signal encode almost the same thing. Removing any one moves the score by ~0, and a drop-one table
would conclude that nothing matters while the block carries the gain. This project already inverted one
conclusion that way in `competition.py`. Drop-one is printed anyway, labelled as misleading, as evidence.

LOOP COVERAGE IS REPORTED PER CELL TYPE. If a cell type has no loop calls, `loop_connects` is a constant
zero there and cannot carry signal. A null for Hi-C must not be confused with absent Hi-C.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEEDS = (0, 1, 2, 3, 4)
CTCF_IDX = [0, 1, 2, 3]          # ctcf_between, ctcf_between_sig, ctcf_elem_hit, ctcf_tss_hit
LOOP_IDX = [4]                   # loop_connects


def assemble():
    """Rebuild exactly the matrices crosscell_eg builds, from the same peak files."""
    import csv
    import gzip
    from crosscell_eg import (CR, HELDOUT, TRAINING, load_peaks, load_loops, build, CONTACT_FEATS)

    def rd(f):
        return list(csv.DictReader(gzip.open(CR / f, "rt"), delimiter="\t"))

    tr, ho = rd(TRAINING), rd(HELDOUT)
    for r in tr:
        r["CellType"] = "K562"
    allrows = tr + ho
    tracks, usable = {}, []
    for c in sorted({r["CellType"] for r in allrows}):
        t = {"acc": load_peaks(c, "accessibility") or load_peaks(c, "accessibility_atac"),
             "k27": load_peaks(c, "h3k27ac"), "pol": load_peaks(c, "polr2a"),
             "ctcf": load_peaks(c, "ctcf"), "loops": load_loops(c)}
        tracks[c] = t
        if all(t[k] for k in ("acc", "k27", "pol", "ctcf")):
            usable.append(c)
    Xs, Cs, ys, chs, cts, cov = [], [], [], [], [], {}
    for c in usable:
        rc = [r for r in allrows if r["CellType"] == c]
        Xc, Cc = build(rc, c, tracks)
        cov[c] = float((Cc[:, LOOP_IDX[0]] > 0).mean())
        Xs.append(Xc); Cs.append(Cc)
        ys += [1 if r["Significant"] in ("TRUE", "True", "true") else 0 for r in rc]
        chs += [r["chrom"] for r in rc]
        cts += [c] * len(rc)
    return (np.vstack(Xs), np.vstack(Cs), np.array(ys), np.array(chs), np.array(cts),
            usable, cov, CONTACT_FEATS)


def main():
    from sklearn.metrics import average_precision_score
    from crispr_gate import _seeded_folds
    import xgboost as xgb

    X, C, y, ch, ct, cells, cov, names = assemble()
    print("=" * 100)
    print("DECOMPOSING THE CONTACT GAIN -- CTCF counting vs measured Hi-C loops")
    print("=" * 100)
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")
    print("  fraction of pairs with a connecting Hi-C loop, per cell type:")
    for c, f in cov.items():
        note = "   <- no loops here, so loop_connects is a constant" if f == 0 else ""
        print(f"    {c:9s} {f:.4f}{note}")

    def fp(Xtr, ytr, Xte):
        m = xgb.XGBClassifier(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1), max_depth=4,
                              n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

    ARMS = {"base": X,
            "+CTCF only": np.hstack([X, C[:, CTCF_IDX]]),
            "+Hi-C only": np.hstack([X, C[:, LOOP_IDX]]),
            "+both": np.hstack([X, C])}
    R = {"n_pairs": int(len(y)), "n_positives": int(y.sum()), "loop_coverage": cov,
         "pooled": {}, "loco": {}}

    print("\n  POOLED (chromosome-grouped folds; all cell types present in training)")
    for tag, M in ARMS.items():
        sc = []
        for s in SEEDS:
            f = _seeded_folds(ch, s)
            p = np.zeros(len(y))
            for k in range(5):
                p[f == k] = fp(M[f != k], y[f != k], M[f == k])
            sc.append(average_precision_score(y, p))
        R["pooled"][tag] = float(np.mean(sc))
        print(f"    {tag:14s} {np.mean(sc):.4f}  [{' '.join(f'{x:.3f}' for x in sc)}]")
    pb, pc, pt = R["pooled"]["base"], R["pooled"]["+CTCF only"], R["pooled"]["+both"]
    print(f"\n    CTCF alone over base   {pc-pb:+.4f}")
    print(f"    Hi-C alone over base   {R['pooled']['+Hi-C only']-pb:+.4f}")
    print(f"    both over base         {pt-pb:+.4f}")
    print(f"    ** Hi-C ON TOP OF CTCF {pt-pc:+.4f}   <- decides whether to buy a Hi-C library")
    R["pooled"]["hic_over_ctcf"] = float(pt - pc)

    print("\n  LOCO (test cell type never seen in training)")
    print(f"    {'test cell':10s} {'pos':>4s}" + "".join(f"{t:>14s}" for t in ARMS)
          + f"{'HiC|CTCF':>11s}")
    for c in cells:
        te = ct == c
        if y[te].sum() < 5:
            continue
        vals, row = {}, f"    {c:10s} {int(y[te].sum()):4d}"
        for tag, M in ARMS.items():
            a = float(average_precision_score(y[te], fp(M[~te], y[~te], M[te])))
            vals[tag] = a
            row += f"{a:14.4f}"
        row += f"{vals['+both']-vals['+CTCF only']:+11.4f}"
        print(row)
        R["loco"][c] = vals
    if R["loco"]:
        mean = {t: float(np.mean([v[t] for v in R["loco"].values()])) for t in ARMS}
        print(f"    {'MEAN':10s} {'':4s}" + "".join(f"{mean[t]:14.4f}" for t in ARMS)
              + f"{mean['+both']-mean['+CTCF only']:+11.4f}")
        print(f"\n    CTCF alone over base   {mean['+CTCF only']-mean['base']:+.4f}")
        print(f"    Hi-C alone over base   {mean['+Hi-C only']-mean['base']:+.4f}")
        print(f"    both over base         {mean['+both']-mean['base']:+.4f}")
        print(f"    ** Hi-C ON TOP OF CTCF {mean['+both']-mean['+CTCF only']:+.4f}")
        R["loco_summary"] = {**mean,
                             "ctcf_over_base": mean["+CTCF only"] - mean["base"],
                             "hic_over_base": mean["+Hi-C only"] - mean["base"],
                             "both_over_base": mean["+both"] - mean["base"],
                             "hic_over_ctcf": mean["+both"] - mean["+CTCF only"]}

    print("\n  drop-one inside the block -- printed BECAUSE it misleads on correlated features:")
    f0 = _seeded_folds(ch, 0)
    full = np.hstack([X, C])

    def one(M):
        p = np.zeros(len(y))
        for k in range(5):
            p[f0 == k] = fp(M[f0 != k], y[f0 != k], M[f0 == k])
        return average_precision_score(y, p)
    fs = one(full)
    for i in CTCF_IDX + LOOP_IDX:
        keep = [j for j in range(C.shape[1]) if j != i]
        print(f"    without {names[i]:20s} delta vs full {one(np.hstack([X, C[:, keep]]))-fs:+.4f}")
    print("    -> near-duplicates; only the block-level numbers above are meaningful.")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "contact_decompose.json", "w"), indent=1)
    print(f"\n  -> {OUT/'contact_decompose.json'}")


if __name__ == "__main__":
    main()
