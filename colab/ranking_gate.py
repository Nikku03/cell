"""THE EVALUATION FRAMEWORK, RUN ON THE EXISTING MODEL BEFORE ANY NEW FEATURE.

WHY THIS ORDER. `adversary.py` established a coordinate-only floor: distance alone reaches within-gene
Recall@1 0.6466 and MRR 0.7521, against a pooled AUPRC of only 0.3305. Pooled AUPRC and within-gene ranking
are not the same difficulty, and this project has only ever reported the former. So the biological model's
ranking numbers have to be measured NOW -- otherwise every later feature gets judged against a target nobody
has established.

Ranking is not a mechanism. It is the lens that says whether new biology improves the decision anyone
actually makes: given this promoter and its candidate enhancers, which one regulates it.

THE POWER REGIMES COLLAPSE, AND THAT IS A FINDING. The plan was three label regimes -- strict (powered
negatives), inclusive (all), and power-weighted (w=power for negatives). Checking the benchmark first:

    power < 0.8 : 243 pairs, ALL 243 of them POSITIVES
    power >= 0.8: 14,491 pairs, of which 13,914 negatives -- i.e. 100% of negatives are powered

Every negative is already adequately powered at effect size 25. That is forced by construction: a pair is
only CALLED negative when the test had power to detect an effect and found none. So strict and inclusive are
identical on negatives, and weighting negatives by power multiplies almost everything by ~1.

The real asymmetry runs the other way: 243 of 820 positives (30%) are UNDERPOWERED. They were called
significant despite low power. So the regime worth running is STRICT POSITIVES -- keep only the 577
well-powered positives -- which asks whether the low-power 243 are noise the model is being penalised for
missing. `ceiling.py` already found this stratum sits 0.0881 (9.1 sd) below its base-rate share.

FEATURES COME FROM THE BENCHMARK'S OWN COLUMNS, NOT FROM MY PEAK FILES. The TSV ships quantitative
DHS.RPM/percentile, H3K27ac.RPM/percentile, H3K4me1.RPM, H3K27me3.RPM and CTCF.RPM per pair, per cell type.
Those are what ENCODE-rE2G is scored on, so using them makes the comparison apples-to-apples and removes a
whole class of my own approximation error: my earlier features were peak-overlap and distance-to-peak
derived, which is a lossy proxy for signal. H3K4me1 and repressive H3K27me3 were both on the "missing
features" list and are in the file.

THE DECIDING STATISTIC IS PAIRED, NOT A DIFFERENCE OF AVERAGES. MRR_model > MRR_distance over 279 groups can
come from a handful of groups. The test is the per-group delta with a sign test and a paired bootstrap, plus:

    Rescue@1 = P(model ranks a positive first | distance-only does NOT)

which measures directly whether biology fixes the coordinate model's mistakes. Its converse -- Break@1,
where distance was right and the model breaks it -- is reported beside it, because a rescue rate quoted
without the breakage rate is half a result.

STRATIFY BY DIFFICULTY. Distance already solves the easy groups, so a modest overall gain can hide a large
gain where it matters. Groups are split by the distance-rank of their positive: easy (nearest), medium (2-3),
hard (4+).
"""
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
CR = SP / "crispr"
HELDOUT = "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
TRAINING = "EPCrisprBenchmark_combined_data.training_K562.GRCh38.tsv.gz"
SEEDS = (0, 1, 2, 3, 4)
POWER_COL = "PowerAtEffectSize25"
MIN_POWER = 0.8

ACT = ["DHS.RPM", "DHS.percentile", "H3K27ac.RPM", "H3K27ac.percentile",
       "H3K27ac.RPM.expandedRegion", "H3K4me1.RPM", "H3K27me3.RPM",
       "H3K27ac_peak_overlap", "H3K4me1_peak_overlap", "H3K27me3_peak_overlap"]
CTCFC = ["CTCF.RPM", "CTCF_peak_overlap"]


def load():
    def rd(f):
        p = CR / f
        if not p.exists():
            raise SystemExit(f"benchmark absent at {p}; run expand_cells.py first")
        return list(csv.DictReader(gzip.open(p, "rt"), delimiter="\t"))
    tr, ho = rd(TRAINING), rd(HELDOUT)
    for r in tr:
        r["CellType"] = "K562"
    return tr + ho


def fnum(r, c):
    try:
        v = float(r[c])
        return v if np.isfinite(v) else 0.0
    except (ValueError, KeyError, TypeError):
        return 0.0


def build(rows):
    y = np.array([1 if r.get("Significant") in ("TRUE", "True", "true") else 0 for r in rows])
    ch = np.array([r["chrom"] for r in rows])
    ct = np.array([r["CellType"] for r in rows])
    key = [(r["measuredGeneSymbol"], r["CellType"]) for r in rows]
    power = np.array([fnum(r, POWER_COL) for r in rows])
    eff = np.array([fnum(r, "EffectSize") for r in rows])
    dist = np.array([max(abs(fnum(r, "distanceToTSS")), 1.0) for r in rows])
    D = np.log10(dist).reshape(-1, 1)
    A = np.array([[fnum(r, c) for c in ACT] for r in rows])
    C = np.array([[fnum(r, c) for c in CTCFC] for r in rows])

    # competition, recomputed here so the arm is self-contained: within each (gene, cell type) group, where
    # does this candidate sit against its rivals. `competition.py` measured this block at +0.0278.
    grp = {}
    for i, k in enumerate(key):
        grp.setdefault(k, []).append(i)
    act1 = A[:, ACT.index("H3K27ac.RPM")] * A[:, ACT.index("DHS.RPM")]
    P = np.zeros((len(rows), 5))
    for k, idx in grp.items():
        idx = np.array(idx)
        d, a = dist[idx], act1[idx]
        P[idx, 0] = len(idx)
        P[idx, 1] = np.argsort(np.argsort(d)) + 1                      # distance rank
        P[idx, 2] = np.argsort(np.argsort(-a)) + 1                     # activity rank
        P[idx, 3] = a / max(a.sum(), 1e-9)                             # activity share
        P[idx, 4] = np.log10(d) - np.log10(np.median(d))               # distance excess
    pn = ["n_candidates", "dist_rank", "activity_rank", "activity_share", "dist_excess"]
    return dict(y=y, ch=ch, ct=ct, key=key, power=power, eff=eff, dist=dist,
                D=D, A=A, C=C, P=P, pnames=pn)


def folds_chrom(ch, seed):
    u = sorted(set(ch))
    a = np.random.default_rng(seed).permutation(len(u))
    m = {c: a[i] % 5 for i, c in enumerate(u)}
    return np.array([m[c] for c in ch])


def fit_oof(X, y, f, w=None):
    import xgboost as xgb
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = f != k, f == k
        m = xgb.XGBClassifier(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1),
                              max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                              eval_metric="aucpr", n_jobs=4)
        m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def fit_loco(X, y, ct):
    """Macro-average over cell types, so K562's 717 positives cannot dominate the mean."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    out = {}
    for c in sorted(set(ct)):
        te = ct == c
        if y[te].sum() < 5:
            continue
        m = xgb.XGBClassifier(scale_pos_weight=(y[~te] == 0).sum() / max((y[~te] == 1).sum(), 1),
                              max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                              eval_metric="aucpr", n_jobs=4)
        m.fit(X[~te], y[~te])
        out[c] = float(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))
    return out


def groups_of(key, y, min_cand=2):
    g = {}
    for i, k in enumerate(key):
        g.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in g.items()
            if len(v) >= min_cand and y[np.array(v)].sum() > 0}


def rank_metrics(scores, y, groups):
    """Per-group rank statistics, returned per group so they can be paired against a baseline."""
    from sklearn.metrics import average_precision_score
    per = {}
    for k, idx in groups.items():
        order = idx[np.argsort(-scores[idx])]
        yy = y[order]
        first = int(np.argmax(yy == 1)) + 1
        per[k] = {"r1": float(yy[0]), "r3": float(yy[:3].max()), "mrr": 1.0 / first,
                  "rank": float(first), "nrank": float((first - 1) / max(len(idx) - 1, 1)),
                  "ap": float(average_precision_score(y[idx], scores[idx]))}
    return per


def summarise(per):
    ks = ("r1", "r3", "mrr", "rank", "nrank", "ap")
    return {k: float(np.mean([v[k] for v in per.values()])) for k in ks}


def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score
    print("=" * 100)
    print("WITHIN-GENE RANKING OF THE EXISTING MODEL -- the target every later feature is judged against")
    print("=" * 100)
    rows = load()
    d = build(rows)
    y, ch, ct, key, power = d["y"], d["ch"], d["ct"], d["key"], d["power"]
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, base rate {y.mean():.4f}")

    print("\n  POWER REGIMES -- checked before being used")
    lo = power < MIN_POWER
    print(f"    power <{MIN_POWER}: {int(lo.sum())} pairs, {int(y[lo].sum())} positives "
          f"({100*y[lo].mean():.0f}% positive)")
    print(f"    power>={MIN_POWER}: {int((~lo).sum()):,} pairs, {int(y[~lo].sum())} positives")
    nneg_pow = int(((y == 0) & (~lo)).sum())
    print(f"    negatives with adequate power: {nneg_pow:,}/{int((y == 0).sum()):,} "
          f"= {100*nneg_pow/(y == 0).sum():.1f}%")
    print("    -> strict and inclusive are IDENTICAL on negatives, and power-weighting negatives is a "
          "no-op. The underpowered pairs are all POSITIVES.")

    REGIMES = {"inclusive (all pairs)": np.ones(len(y), bool),
               f"strict positives (power>={MIN_POWER})": ~(lo & (y == 1))}
    ARMS = {"M0 distance": d["D"],
            "M1 +activity": np.hstack([d["D"], d["A"]]),
            "M2 +CTCF": np.hstack([d["D"], d["A"], d["C"]]),
            "M3 +competition": np.hstack([d["D"], d["A"], d["C"], d["P"]])}
    R = {"n_pairs": int(len(y)), "n_pos": int(y.sum()), "power_col": POWER_COL,
         "n_lowpower": int(lo.sum()), "n_lowpower_positive": int(y[lo].sum()),
         "frac_negatives_powered": nneg_pow / float((y == 0).sum()),
         "activity_cols": ACT, "ctcf_cols": CTCFC, "competition_cols": d["pnames"], "regimes": {}}

    for rname, keep in REGIMES.items():
        yy, chh, ctt = y[keep], ch[keep], ct[keep]
        kk = [key[i] for i in np.where(keep)[0]]
        groups = groups_of(kk, yy)
        print("\n" + "-" * 100)
        print(f"  REGIME: {rname}   {int(keep.sum()):,} pairs, {int(yy.sum())} positives, "
              f"{len(groups)} evaluable gene groups")
        rr = {"n_pairs": int(keep.sum()), "n_pos": int(yy.sum()), "n_groups": len(groups), "arms": {}}
        base_per = None
        print(f"    {'arm':17s} {'pooled':>7s} {'LOCO':>7s} {'R@1':>6s} {'R@3':>6s} {'MRR':>6s} "
              f"{'rank':>6s} {'Rescue@1':>9s} {'Break@1':>8s} {'dMRR p':>8s}")
        for aname, M in ARMS.items():
            X = M[keep]
            ap, pers = [], []
            for s in SEEDS:
                f = folds_chrom(chh, s)
                p = fit_oof(X, yy, f)
                ap.append(average_precision_score(yy, p))
                pers.append(rank_metrics(p, yy, groups))
            # average each group's metrics across seeds before comparing arms
            per = {k: {m: float(np.mean([pp[k][m] for pp in pers]))
                       for m in ("r1", "r3", "mrr", "rank", "nrank", "ap")} for k in groups}
            sm = summarise(per)
            loco = fit_loco(X, yy, ctt)
            macro = float(np.mean(list(loco.values()))) if loco else np.nan
            if base_per is None:
                base_per, resc, brk, pv = per, np.nan, np.nan, np.nan
            else:
                miss = [k for k in groups if base_per[k]["r1"] < 0.5]
                hit = [k for k in groups if base_per[k]["r1"] >= 0.5]
                resc = float(np.mean([per[k]["r1"] for k in miss])) if miss else np.nan
                brk = float(1 - np.mean([per[k]["r1"] for k in hit])) if hit else np.nan
                dm = np.array([per[k]["mrr"] - base_per[k]["mrr"] for k in groups])
                nz = dm[dm != 0]
                pv = (stats.binomtest(int((nz > 0).sum()), len(nz), 0.5,
                                      alternative="greater").pvalue if len(nz) else np.nan)
            print(f"    {aname:17s} {np.mean(ap):7.4f} {macro:7.4f} {sm['r1']:6.3f} {sm['r3']:6.3f} "
                  f"{sm['mrr']:6.3f} {sm['rank']:6.2f} "
                  f"{resc if np.isfinite(resc) else float('nan'):9.3f} "
                  f"{brk if np.isfinite(brk) else float('nan'):8.3f} "
                  f"{pv if np.isfinite(pv) else float('nan'):8.4f}")
            rr["arms"][aname] = {"pooled_auprc": float(np.mean(ap)), "loco_macro": macro,
                                 "loco": loco, **sm,
                                 "rescue_at_1": resc, "break_at_1": brk, "dmrr_sign_p": pv}
        # difficulty strata on the best arm vs distance
        best = list(ARMS)[-1]
        print(f"\n    DIFFICULTY STRATA ({best} vs M0 distance), by the distance-rank of the positive:")
        Xb, Xd = ARMS[best][keep], ARMS["M0 distance"][keep]
        pb = np.mean([fit_oof(Xb, yy, folds_chrom(chh, s)) for s in SEEDS], 0)
        pd_ = np.mean([fit_oof(Xd, yy, folds_chrom(chh, s)) for s in SEEDS], 0)
        perb, perd = rank_metrics(pb, yy, groups), rank_metrics(pd_, yy, groups)
        dv = d["dist"][keep]
        strat = {"easy (positive nearest)": [], "medium (rank 2-3)": [], "hard (rank 4+)": []}
        for k, idx in groups.items():
            dr = int(np.argsort(np.argsort(dv[idx]))[np.argmax(yy[idx] == 1)]) + 1
            strat["easy (positive nearest)" if dr == 1 else
                  "medium (rank 2-3)" if dr <= 3 else "hard (rank 4+)"].append(k)
        rr["strata"] = {}
        for sname, keys in strat.items():
            if not keys:
                continue
            m_b = float(np.mean([perb[k]["mrr"] for k in keys]))
            m_d = float(np.mean([perd[k]["mrr"] for k in keys]))
            r_b = float(np.mean([perb[k]["r1"] for k in keys]))
            r_d = float(np.mean([perd[k]["r1"] for k in keys]))
            print(f"      {sname:26s} {len(keys):4d} groups   MRR {m_d:.3f} -> {m_b:.3f} "
                  f"({m_b-m_d:+.3f})   R@1 {r_d:.3f} -> {r_b:.3f} ({r_b-r_d:+.3f})")
            rr["strata"][sname] = {"n": len(keys), "mrr_distance": m_d, "mrr_model": m_b,
                                   "r1_distance": r_d, "r1_model": r_b}
        R["regimes"][rname] = rr

    print("\n" + "=" * 100)
    inc = R["regimes"]["inclusive (all pairs)"]["arms"]
    b = inc["M3 +competition"]
    print(f"  THE TARGET FOR EVERY LATER FEATURE, on {R['regimes']['inclusive (all pairs)']['n_groups']} "
          f"evaluable gene groups:")
    print(f"    distance floor   R@1 {inc['M0 distance']['r1']:.4f}  MRR {inc['M0 distance']['mrr']:.4f}")
    print(f"    current model    R@1 {b['r1']:.4f}  MRR {b['mrr']:.4f}   "
          f"Rescue@1 {b['rescue_at_1']:.4f}  Break@1 {b['break_at_1']:.4f}  "
          f"paired dMRR sign p {b['dmrr_sign_p']:.4g}")
    R["target"] = {"distance_r1": inc["M0 distance"]["r1"], "distance_mrr": inc["M0 distance"]["mrr"],
                   "model_r1": b["r1"], "model_mrr": b["mrr"], "rescue_at_1": b["rescue_at_1"],
                   "break_at_1": b["break_at_1"], "dmrr_sign_p": b["dmrr_sign_p"]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "ranking_gate.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'ranking_gate.json'}")


if __name__ == "__main__":
    main()
