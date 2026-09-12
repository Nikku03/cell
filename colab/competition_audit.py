"""AUDITING THE COMPETITION BLOCK -- the largest contributor, and the least examined.

WHY THIS BLOCK NEEDS ATTACKING. Competition features (n_candidates, dist_rank, activity_rank, activity_share,
dist_excess) were measured at +0.0278 pooled AUPRC when first built, and the short-range analysis found them
the LARGEST leave-one-out block in every distance stratum -- +0.0620, +0.0592, +0.1142, +0.0709 AUROC --
bigger than element activity, promoter occupancy and CTCF combined. A block that dominates every stratum and
has never been audited is the first place to look for an artefact.

FOUR SPECIFIC THREATS

 1 GROUP-SIZE LEAKAGE. `n_candidates` is the size of the (gene, cell type) group. If groups vary in how many
   positives they contain, group size encodes the per-group base rate, and the model can exploit that without
   learning anything about elements. Tested by correlating group size with group positive RATE, and by
   scoring n_candidates alone.

 2 CANDIDATE-SET DEPENDENCE, WHICH IS ALSO A LIVE DEFECT IN A SHIPPED ARTEFACT. Every competition feature is
   defined relative to a candidate set. In the benchmark that set is "elements someone chose to TEST near this
   gene" -- an experimental design decision. In `eg_layer.py`'s genome-wide scoring it is "every accessible
   peak in a 250 kb window", a different and much larger set. So the shipped layer computes these features on
   a distribution the model never trained on. This module recomputes the benchmark pairs' competition features
   under the genome-wide candidate definition and measures how far they move and what it costs.

 3 DEGENERATE GROUPS. In a group of one, dist_rank and activity_rank are both 1 and activity_share is 1.0 by
   construction. If singletons are enriched for positives, those features encode "this gene had one candidate"
   -- a design artefact wearing a biological label. Tested by re-scoring with small groups excluded.

 4 THE METRIC IS MATCHED TO THE FEATURE. Within-gene ranking rewards whatever orders elements inside a group,
   and competition features ARE within-group orderings. So the ranking gain is partly definitional. The pooled
   AUPRC arm is the one that is not, and both are reported.

CONTROLS. Two permutations, which separate different claims: shuffling competition values WITHIN a group
destroys the ordering while preserving every group's marginal distribution, so it isolates the ordering
itself; shuffling ACROSS groups destroys both. A block whose gain survives the within-group shuffle is not
carrying ordering information at all.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEEDS = (0, 1, 2, 3, 4)
WINDOW = 250_000
COMP = ["n_candidates", "dist_rank", "activity_rank", "activity_share", "dist_excess"]


def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ranking_gate import load, folds_chrom, fit_oof, groups_of, rank_metrics, summarise
    from eg_layer import load_peaks, featurise, FEATNAMES, TRACKS
    print("=" * 100)
    print("COMPETITION BLOCK AUDIT -- leakage, candidate-set dependence, degenerate groups")
    print("=" * 100)
    rows = load()
    y = np.array([1 if r.get("Significant") in ("TRUE", "True", "true") else 0 for r in rows])
    ct = np.array([r["CellType"] for r in rows])
    ch = np.array([r["chrom"] for r in rows])
    peaks = {}
    for c in sorted(set(ct)):
        got = {t: load_peaks(c, t) for t in TRACKS}
        if all(v is not None for v in got.values()):
            peaks[c] = got
    pairs = [(r["chrom"], (int(r["chromStart"]) + int(r["chromEnd"])) // 2,
              int(float(r["startTSS"])), (r["measuredGeneSymbol"], c)) for r, c in zip(rows, ct)]
    X = np.zeros((len(rows), len(FEATNAMES)))
    for c in peaks:
        m = np.where(ct == c)[0]
        X[m] = featurise([pairs[i] for i in m], peaks[c])
    keep = np.array([c in peaks for c in ct])
    X, y, ch = X[keep], y[keep], ch[keep]
    key = [pairs[i][3] for i in np.where(keep)[0]]
    kp = [pairs[i] for i in np.where(keep)[0]]
    ix = {n: i for i, n in enumerate(FEATNAMES)}
    CI = [ix[n] for n in COMP]
    OTHER = [i for i in range(len(FEATNAMES)) if i not in CI]
    print(f"  {len(y):,} scoreable pairs, {int(y.sum())} positives")

    R = {"n": int(len(y)), "n_pos": int(y.sum()), "competition_features": COMP}

    # ---- 1 GROUP-SIZE LEAKAGE ----
    gid = {}
    for i, k in enumerate(key):
        gid.setdefault(k, []).append(i)
    sizes = np.array([len(gid[k]) for k in key])
    grate = np.array([y[np.array(gid[k])].mean() for k in key])
    gsz = np.array([len(v) for v in gid.values()])
    gpr = np.array([y[np.array(v)].mean() for v in gid.values()])
    rho = stats.spearmanr(gsz, gpr)
    print(f"\n  [1] GROUP-SIZE LEAKAGE ({len(gid):,} groups)")
    print(f"    group size vs group positive RATE: Spearman {rho[0]:+.4f} (p {rho[1]:.3g})")
    print(f"    group size distribution: 1 -> {int((gsz==1).sum()):,}, 2 -> {int((gsz==2).sum()):,}, "
          f">=3 -> {int((gsz>=3).sum()):,};  median {int(np.median(gsz))}")
    au_n = roc_auc_score(y, sizes)
    print(f"    n_candidates ALONE, AUROC {au_n:.4f}   "
          f"{'<- encodes the per-group base rate' if abs(au_n-0.5) > 0.05 else '<- no standalone signal'}")
    R["group_size"] = {"n_groups": len(gid), "spearman_size_vs_rate": float(rho[0]),
                       "p": float(rho[1]), "auroc_n_candidates_alone": float(au_n),
                       "singletons": int((gsz == 1).sum())}

    # ---- 2 CANDIDATE-SET DEPENDENCE (audits the shipped layer) ----
    print(f"\n  [2] CANDIDATE-SET DEPENDENCE -- benchmark-tested set vs genome-wide peaks")
    Xg = X.copy()
    moved = {n: [] for n in COMP}
    for c in peaks:
        acc = peaks[c].get("accessibility")
        idx = [i for i, k in enumerate(key) if k[1] == c]
        if acc is None or not idx:
            continue
        # rebuild each pair's group as ALL accessible peaks within the window of its TSS -- the definition
        # eg_layer uses genome-wide -- then recompute the same five features on that set
        bygene = {}
        for i in idx:
            bygene.setdefault(key[i], []).append(i)
        for k, mem in bygene.items():
            chrom, _mid, tss, _k = kp[mem[0]]
            if chrom not in acc:
                continue
            st, en, sg = acc[chrom]
            mid = (st + en) // 2
            sel = np.where((np.abs(mid - tss) <= WINDOW) & (np.abs(mid - tss) > 500))[0]
            if len(sel) < 2:
                continue
            dd = np.abs(mid[sel] - tss).astype(float)
            aa = np.log1p(sg[sel])
            for i in mem:
                d_i = 10 ** X[i, ix["log_dist"]]
                a_i = X[i, ix["elem_accessibility"]] + X[i, ix["elem_h3k27ac"]]
                n_c = len(sel) + 1
                dr = 1 + int((dd < d_i).sum())
                ar = 1 + int((aa > a_i).sum())
                sh = a_i / max(aa.sum() + a_i, 1e-9)
                de = np.log10(max(d_i, 1)) - np.median(np.log10(np.maximum(dd, 1)))
                new = [n_c, dr, ar, sh, de]
                for j, (nm, v) in enumerate(zip(COMP, new)):
                    moved[nm].append(abs(v - X[i, CI[j]]))
                    Xg[i, CI[j]] = v
    print(f"    {'feature':16s} {'benchmark median':>17s} {'genome-wide median':>19s} {'median |shift|':>15s}")
    for j, nm in enumerate(COMP):
        if not moved[nm]:
            continue
        print(f"    {nm:16s} {np.median(X[:, CI[j]]):17.3f} {np.median(Xg[:, CI[j]]):19.3f} "
              f"{np.median(moved[nm]):15.3f}")
    R["candidate_set_shift"] = {nm: {"benchmark_median": float(np.median(X[:, CI[j]])),
                                     "genomewide_median": float(np.median(Xg[:, CI[j]])),
                                     "median_abs_shift": float(np.median(moved[nm])) if moved[nm] else None}
                                for j, nm in enumerate(COMP)}

    def score(Xa, tag, m=None):
        m = np.ones(len(y), bool) if m is None else m
        ap, au = [], []
        for s in SEEDS:
            f = folds_chrom(ch[m], s)
            p = fit_oof(Xa[m], y[m], f)
            ap.append(average_precision_score(y[m], p))
            au.append(roc_auc_score(y[m], p))
        return float(np.mean(ap)), float(np.mean(au))

    ap_b, au_b = score(X, "benchmark competition")
    ap_g, au_g = score(Xg, "genome-wide competition")
    ap_n, au_n2 = score(X[:, OTHER], "no competition")
    print(f"    trained+scored with BENCHMARK competition : AUPRC {ap_b:.4f}  AUROC {au_b:.4f}")
    print(f"    trained+scored with GENOME-WIDE competition: AUPRC {ap_g:.4f}  AUROC {au_g:.4f}")
    print(f"    no competition block at all               : AUPRC {ap_n:.4f}  AUROC {au_n2:.4f}")
    print(f"    -> competition worth {ap_b-ap_n:+.4f} AUPRC as defined on the benchmark, "
          f"{ap_g-ap_n:+.4f} under the genome-wide definition")
    R["definitions"] = {"benchmark": {"auprc": ap_b, "auroc": au_b},
                        "genomewide": {"auprc": ap_g, "auroc": au_g},
                        "none": {"auprc": ap_n, "auroc": au_n2},
                        "gain_benchmark": ap_b - ap_n, "gain_genomewide": ap_g - ap_n}

    # cross-definition: train on one, score the other -- what the shipped layer actually does
    import xgboost as xgb
    cross = []
    for s in SEEDS:
        f = folds_chrom(ch, s)
        p = np.zeros(len(y))
        for k in range(5):
            tr, te = f != k, f == k
            m = xgb.XGBClassifier(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
                                  max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                                  eval_metric="aucpr", n_jobs=4)
            m.fit(X[tr], y[tr])                 # trained on BENCHMARK competition
            p[te] = m.predict_proba(Xg[te])[:, 1]   # scored on GENOME-WIDE competition
        cross.append(average_precision_score(y, p))
    print(f"    TRAIN benchmark -> SCORE genome-wide (what eg_layer does): AUPRC {np.mean(cross):.4f}"
          f"   ({np.mean(cross)-ap_b:+.4f} vs matched training)")
    R["definitions"]["cross_train_bench_score_genomewide"] = float(np.mean(cross))

    # ---- 3 DEGENERATE GROUPS ----
    print(f"\n  [3] DEGENERATE GROUPS")
    for lo in (1, 2, 3, 5):
        m = sizes >= lo
        if y[m].sum() < 30:
            continue
        a1, _ = score(X, "", m)
        a0, _ = score(X[:, OTHER], "", m)
        print(f"    groups with >={lo} candidates: {int(m.sum()):6,d} pairs, {int(y[m].sum()):4d} pos   "
              f"competition worth {a1-a0:+.4f} AUPRC")
        R.setdefault("by_group_size", {})[f">={lo}"] = {"n": int(m.sum()), "gain": a1 - a0}

    # ---- 4 PERMUTATION CONTROLS ----
    print(f"\n  [4] PERMUTATION CONTROLS")
    rng = np.random.default_rng(0)
    Xw = X.copy()
    for k, mem in gid.items():
        mem = np.array(mem)
        if len(mem) > 1:
            Xw[np.ix_(mem, CI)] = X[np.ix_(mem[rng.permutation(len(mem))], CI)]
    Xa = X.copy()
    Xa[:, CI] = X[rng.permutation(len(y))][:, CI]
    ap_w, _ = score(Xw, "")
    ap_a, _ = score(Xa, "")
    print(f"    real competition                        AUPRC {ap_b:.4f}")
    print(f"    shuffled WITHIN group (ordering killed) AUPRC {ap_w:.4f}   ({ap_w-ap_b:+.4f})")
    print(f"    shuffled ACROSS groups (all killed)     AUPRC {ap_a:.4f}   ({ap_a-ap_b:+.4f})")
    print(f"    no block at all                         AUPRC {ap_n:.4f}")
    print(f"    -> ordering contributes {ap_b-ap_w:+.4f}; the rest of the block's gain "
          f"({ap_w-ap_n:+.4f}) survives having its ordering destroyed")
    R["permutations"] = {"real": ap_b, "within_group": ap_w, "across_groups": ap_a, "none": ap_n,
                         "ordering_contribution": ap_b - ap_w,
                         "survives_within_shuffle": ap_w - ap_n}

    # ---- verdict ----
    print("\n" + "=" * 100)
    lk = abs(au_n - 0.5) > 0.05
    shift = R["definitions"]["gain_benchmark"] - R["definitions"]["gain_genomewide"]
    crossloss = R["definitions"]["cross_train_bench_score_genomewide"] - ap_b
    order_frac = (ap_b - ap_w) / max(ap_b - ap_n, 1e-9)
    parts = []
    parts.append(f"n_candidates alone scores AUROC {au_n:.4f}"
                 + (f", so group size DOES encode the per-group base rate and part of the block is "
                    f"design leakage" if lk else ", so group size carries no standalone signal"))
    parts.append(f"the block is worth {R['definitions']['gain_benchmark']:+.4f} AUPRC under the benchmark's "
                 f"candidate definition and {R['definitions']['gain_genomewide']:+.4f} under the "
                 f"genome-wide one (difference {shift:+.4f})")
    parts.append(f"training on one definition and scoring the other -- which is what the shipped edge layer "
                 f"does -- costs {crossloss:+.4f} AUPRC")
    parts.append(f"{100*order_frac:.0f}% of the gain is the within-group ORDERING (destroyed by a "
                 f"within-group shuffle); the remainder survives it and is therefore not ordering "
                 f"information at all")
    R["verdict"] = ". ".join(parts) + "."
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "competition_audit.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'competition_audit.json'}")


if __name__ == "__main__":
    main()
