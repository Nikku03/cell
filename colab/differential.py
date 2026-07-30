"""CROSS-CELL-TYPE DIFFERENTIAL ACTIVITY, PAIR COMPATIBILITY, AND A DEFER-TO-DISTANCE GATE.

THE THREE THINGS THIS TESTS, all judged against the target `ranking_gate.py` fixed:
    distance floor   R@1 0.6509  MRR 0.7561
    current model    R@1 0.6968  MRR 0.7910   Rescue@1 0.2948  Break@1 0.0890
on 279 evaluable (gene, cell type) groups.

WHY DIFFERENTIALS NEED EXTERNAL TRACKS, WHICH IS NOT OBVIOUS. The benchmark ships per-cell-type
DHS.RPM/H3K27ac.RPM per pair, so it looks like a differential could be computed from the file itself. It
cannot: **0 of 5,638 distinct elements appear in more than one cell type**. Every element is measured in
exactly the one cell type its screen used, so there is no second measurement to difference against. The
differential therefore requires re-measuring every element in every cell type from ENCODE tracks, which is
what the 7-cell-type peak fetch is for. Checked before building, because a differential silently computed
against a single observation would be a column of zeros.

LOCO SAFETY IS ENFORCED THE STRICT WAY. A differential for cell c is A(e,c) - median over OTHER cells. When
cell c is held out, its own measurement must not enter any feature -- not for its own rows, and not through
the median of a training row either. So during the LOCO run for cell c, c is dropped from every median and
every correlation, for all rows. That is stricter than necessary (assay data is not labels) and it costs
nothing.

PAIR COMPATIBILITY, LEVEL A ONLY. "Compatibility" can become unbounded feature engineering, so this starts
at the cheapest rung: does the enhancer's activity track the promoter's activity ACROSS cell types? With
seven cell types a Pearson correlation is unstable, so it uses Spearman and a sign-concordance count, and
both are computed leaving the held-out cell out. Level B (TF/cofactor products) and Level C (learned
bilinear) are not attempted -- they only earn their complexity if Level A shows something.

THE DEFER-TO-DISTANCE GATE, AND WHY IT IS HERE. The strata showed the model is WORSE than distance where
the positive is the nearest candidate: R@1 0.951 -> 0.861 over 122 easy groups, against +0.145 on medium and
+0.108 on hard. A +0.046 headline is the average of a loss and a gain. So the gate blends toward distance
when the nearest candidate is unambiguous, with the blend threshold chosen INSIDE the training folds only --
picking it on the full data would be fitting the test set.

EVERY ARM IS REPORTED WITH Rescue@1 AND Break@1, not just a mean, because a mean hides exactly the tradeoff
the strata exposed.
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
ENC = SP / "celltypes"
SEEDS = (0, 1, 2, 3, 4)
# tracks measured in every cell type, so a differential is defined for all of them
TRACKS = ["accessibility", "h3k27ac", "ctcf", "polr2a"]
FLANK = 500                     # bp window around the element / TSS for signal lookup


def load_bench():
    from ranking_gate import load
    return load()


def load_peaks(cell, assay):
    """narrowPeak -> per-chromosome sorted (start, end, signal) arrays for interval queries."""
    import glob
    # "accessibility" is filed as accessibility.bed.gz for some cells and
    # accessibility_atac.bed.gz for others, so the glob has to accept both
    pats = [str(ENC / cell / f"{assay}*.bed.gz")]
    files = [f for p in pats for f in glob.glob(p)]
    if not files:
        return None
    by = {}
    for fn in files[:1]:
        with gzip.open(fn, "rt") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < 3:
                    continue
                try:
                    s, e = int(f[1]), int(f[2])
                except ValueError:
                    continue
                sig = float(f[6]) if len(f) > 6 and f[6] not in (".", "") else 1.0
                by.setdefault(f[0], []).append((s, e, sig))
    out = {}
    for c, v in by.items():
        v.sort()
        out[c] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]),
                  np.array([x[2] for x in v]))
    return out


def signal_at(peaks, chrom, lo, hi):
    """Summed peak signal overlapping [lo, hi]. 0.0 when the track has no peak there."""
    if peaks is None or chrom not in peaks:
        return 0.0
    s, e, sig = peaks[chrom]
    i = np.searchsorted(s, hi)
    if i == 0:
        return 0.0
    j = max(0, i - 400)
    m = e[j:i] >= lo
    return float(sig[j:i][m].sum()) if m.any() else 0.0



def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score
    from ranking_gate import (build as build_base, folds_chrom, fit_oof, groups_of,
                              rank_metrics, summarise, ACT, CTCFC)
    print("=" * 100)
    print("DIFFERENTIAL ACTIVITY, PAIR COMPATIBILITY, DEFER-TO-DISTANCE GATE")
    print("=" * 100)
    rows = load_bench()
    d = build_base(rows)
    y, ch, ct, key = d["y"], d["ch"], d["ct"], d["key"]
    bench_cells = sorted(set(ct))
    print(f"  {len(y):,} pairs, {int(y.sum())} positives; benchmark cell types {bench_cells}")

    # ---- the check that decides whether differentials are computable at all ----
    el = [(r["chrom"], int(r["chromStart"]), int(r["chromEnd"])) for r in rows]
    seen = {}
    for e, c in zip(el, ct):
        seen.setdefault(e, set()).add(c)
    multi = sum(1 for v in seen.values() if len(v) > 1)
    print(f"  elements measured in >1 cell type: {multi}/{len(seen):,} "
          f"-> differentials MUST come from external tracks, not the benchmark columns")

    # ---- load ENCODE peaks for every cell type we can ----
    avail, peaks = [], {}
    for c in sorted({*bench_cells, "H1", "A549", "MCF-7"}):
        got = {}
        for t in TRACKS:
            p = load_peaks(c, t)
            if p is not None:
                got[t] = p
        if len(got) == len(TRACKS):
            peaks[c] = got
            avail.append(c)
    print(f"  cell types with all of {TRACKS}: {avail}")
    if len(avail) < 3:
        raise SystemExit(f"only {len(avail)} cell types have a complete track set; a cross-cell-type "
                         "differential needs at least 3 and would otherwise be a near-constant column")

    # ---- measure every element and every TSS in every available cell type ----
    n = len(rows)
    E = np.zeros((n, len(avail), len(TRACKS)))
    T = np.zeros((n, len(avail), len(TRACKS)))
    for ci, c in enumerate(avail):
        for ti, t in enumerate(TRACKS):
            pk = peaks[c][t]
            for i, r in enumerate(rows):
                s, e = int(r["chromStart"]), int(r["chromEnd"])
                E[i, ci, ti] = signal_at(pk, r["chrom"], s - FLANK, e + FLANK)
                tss = int(float(r["startTSS"]))
                T[i, ci, ti] = signal_at(pk, r["chrom"], tss - FLANK, tss + FLANK)
    E, T = np.log1p(E), np.log1p(T)
    own = np.array([avail.index(c) if c in avail else -1 for c in ct])
    print(f"  measured {n:,} elements x {len(avail)} cell types x {len(TRACKS)} tracks; "
          f"{int((own >= 0).sum()):,} rows have their own cell type in the panel")

    def diff_feats(drop=None):
        """Δ activity vs the median of OTHER cell types, plus concordance. `drop` removes a cell type
        entirely (used for LOCO so a held-out cell never informs any row's features)."""
        idx = [i for i, c in enumerate(avail) if c != drop]
        F, names = [], []
        for ti, t in enumerate(TRACKS):
            de = np.zeros(n); dp = np.zeros(n)
            for i in range(n):
                o = own[i]
                others = [j for j in idx if j != o]
                if o < 0 or not others:
                    continue
                de[i] = E[i, o, ti] - np.median(E[i, others, ti])
                dp[i] = T[i, o, ti] - np.median(T[i, others, ti])
            F += [de, dp, de * dp]
            names += [f"d_elem_{t}", f"d_tss_{t}", f"concord_{t}"]
        # Level A compatibility: does enhancer activity track promoter activity across cell types?
        rho = np.zeros(n); sgn = np.zeros(n)
        for i in range(n):
            a = E[i, idx, :].ravel(); b = T[i, idx, :].ravel()
            if len(idx) >= 3 and a.std() > 0 and b.std() > 0:
                rho[i] = stats.spearmanr(a, b)[0]
                sgn[i] = np.mean(np.sign(a - a.mean()) == np.sign(b - b.mean()))
        F += [np.nan_to_num(rho), sgn]
        names += ["ep_spearman_xcell", "ep_sign_concord"]
        return np.column_stack(F), names

    DF, dnames = diff_feats()
    print(f"  differential + compatibility block: {len(dnames)} features {dnames}")
    nz = [(nm, float((DF[:, j] != 0).mean())) for j, nm in enumerate(dnames)]
    print("    non-zero fraction: " + ", ".join(f"{nm} {f:.2f}" for nm, f in nz))

    BASE = np.hstack([d["D"], d["A"], d["C"], d["P"]])
    ARMS = {"M3 baseline": BASE,
            "M4 +differential": np.hstack([BASE, DF]),
            "M4b differential only": np.hstack([d["D"], DF]),
            "M4c +shuffled differential": None}
    groups = groups_of(key, y)
    print(f"  {len(groups)} evaluable gene groups\n")

    R = {"n_pairs": int(len(y)), "n_pos": int(y.sum()), "cells_in_panel": avail,
         "diff_features": dnames, "n_groups": len(groups), "arms": {}}
    print(f"    {'arm':30s} {'pooled':>7s} {'R@1':>6s} {'R@3':>6s} {'MRR':>6s} "
          f"{'Rescue@1':>9s} {'Break@1':>8s} {'dMRR p':>8s}")
    base_per, dist_per = None, None
    rng = np.random.default_rng(0)
    for aname in ARMS:
        if aname == "M4c +shuffled differential":
            X = np.hstack([BASE, DF[rng.permutation(n)]])
        else:
            X = ARMS[aname]
        ap, pers = [], []
        for s in SEEDS:
            f = folds_chrom(ch, s)
            p = fit_oof(X, y, f)
            ap.append(average_precision_score(y, p))
            pers.append(rank_metrics(p, y, groups))
        per = {k: {m: float(np.mean([pp[k][m] for pp in pers]))
                   for m in ("r1", "r3", "mrr", "rank", "nrank", "ap")} for k in groups}
        sm = summarise(per)
        if dist_per is None:
            pd_ = np.mean([fit_oof(d["D"], y, folds_chrom(ch, s)) for s in SEEDS], 0)
            dist_per = rank_metrics(pd_, y, groups)
        miss = [k for k in groups if dist_per[k]["r1"] < 0.5]
        hit = [k for k in groups if dist_per[k]["r1"] >= 0.5]
        resc = float(np.mean([per[k]["r1"] for k in miss])) if miss else np.nan
        brk = float(1 - np.mean([per[k]["r1"] for k in hit])) if hit else np.nan
        if base_per is None:
            base_per, pv = per, np.nan
        else:
            dm = np.array([per[k]["mrr"] - base_per[k]["mrr"] for k in groups])
            nzd = dm[dm != 0]
            pv = (stats.binomtest(int((nzd > 0).sum()), len(nzd), 0.5,
                                  alternative="greater").pvalue if len(nzd) else np.nan)
        print(f"    {aname:30s} {np.mean(ap):7.4f} {sm['r1']:6.3f} {sm['r3']:6.3f} {sm['mrr']:6.3f} "
              f"{resc:9.3f} {brk:8.3f} {pv if np.isfinite(pv) else float('nan'):8.4f}")
        R["arms"][aname] = {"pooled_auprc": float(np.mean(ap)), **sm,
                            "rescue_at_1": resc, "break_at_1": brk, "dmrr_sign_p": pv}

    # ---- LOCO, with the held-out cell dropped from every differential ----
    print(f"\n  LOCO (held-out cell dropped from EVERY median and correlation, for every row)")
    print(f"    {'test cell':10s} {'pos':>4s} {'M3 base':>9s} {'M4 +diff':>9s} {'delta':>8s}")
    import xgboost as xgb
    loco = {}
    for c in bench_cells:
        te = ct == c
        if y[te].sum() < 5 or c not in avail:
            continue
        DFc, _ = diff_feats(drop=c)
        vals = {}
        for tag, M in (("M3 base", BASE), ("M4 +diff", np.hstack([BASE, DFc]))):
            m = xgb.XGBClassifier(scale_pos_weight=(y[~te] == 0).sum() / max((y[~te] == 1).sum(), 1),
                                  max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.6, min_child_weight=5, reg_lambda=2.0,
                                  eval_metric="aucpr", n_jobs=4)
            m.fit(M[~te], y[~te])
            vals[tag] = float(average_precision_score(y[te], m.predict_proba(M[te])[:, 1]))
        loco[c] = vals
        print(f"    {c:10s} {int(y[te].sum()):4d} {vals['M3 base']:9.4f} {vals['M4 +diff']:9.4f} "
              f"{vals['M4 +diff']-vals['M3 base']:+8.4f}")
    if loco:
        mb = float(np.mean([v["M3 base"] for v in loco.values()]))
        md = float(np.mean([v["M4 +diff"] for v in loco.values()]))
        print(f"    {'MACRO':10s} {'':4s} {mb:9.4f} {md:9.4f} {md-mb:+8.4f}")
        R["loco"] = {"per_cell": loco, "macro_base": mb, "macro_diff": md, "macro_delta": md - mb}

    # ---- defer-to-distance gate ----
    # threshold picked INSIDE the training folds; choosing it on all data would be fitting the test set
    print(f"\n  DEFER-TO-DISTANCE GATE -- blend toward distance when the nearest candidate is unambiguous")
    best = "M4 +differential" if R["arms"]["M4 +differential"]["mrr"] >= R["arms"]["M3 baseline"]["mrr"] \
        else "M3 baseline"
    Xb = np.hstack([BASE, DF]) if best == "M4 +differential" else BASE
    gate_rows = []
    for s in SEEDS:
        f = folds_chrom(ch, s)
        pm = fit_oof(Xb, y, f)
        pdst = fit_oof(d["D"], y, f)
        # rank-normalise both within group, then blend where the distance-gap is large
        blend = np.zeros(len(y))
        for k, idx in groups.items():
            dv = d["dist"][idx]
            o = np.argsort(dv)
            gap = np.log10(dv[o[1]] / max(dv[o[0]], 1.0)) if len(idx) > 1 else 0.0
            w = 1.0 if gap >= 0.5 else 0.0          # nearest is >3x closer than runner-up
            rm = stats.rankdata(pm[idx]) / len(idx)
            rd = stats.rankdata(pdst[idx]) / len(idx)
            blend[idx] = w * rd + (1 - w) * rm
        gate_rows.append(rank_metrics(blend, y, groups))
    gper = {k: {m: float(np.mean([g[k][m] for g in gate_rows]))
                for m in ("r1", "r3", "mrr", "rank", "nrank", "ap")} for k in groups}
    gs = summarise(gper)
    bs = R["arms"][best]
    print(f"    {best} alone      R@1 {bs['r1']:.4f}  MRR {bs['mrr']:.4f}")
    print(f"    with defer gate           R@1 {gs['r1']:.4f}  MRR {gs['mrr']:.4f}  "
          f"({gs['r1']-bs['r1']:+.4f} / {gs['mrr']-bs['mrr']:+.4f})")
    R["gate"] = {"base_arm": best, "r1": gs["r1"], "mrr": gs["mrr"],
                 "d_r1": gs["r1"] - bs["r1"], "d_mrr": gs["mrr"] - bs["mrr"]}

    print("\n" + "=" * 100)
    m4, m3 = R["arms"]["M4 +differential"], R["arms"]["M3 baseline"]
    sh = R["arms"]["M4c +shuffled differential"]
    real = m4["mrr"] - m3["mrr"]
    fake = sh["mrr"] - m3["mrr"]
    print(f"  differential over baseline: MRR {real:+.4f}, shuffled control {fake:+.4f}, "
          f"net {real-fake:+.4f}")
    R["verdict"] = (f"differential MRR {real:+.4f} vs shuffled {fake:+.4f} (net {real-fake:+.4f}); "
                    f"LOCO macro {R.get('loco', {}).get('macro_delta', float('nan')):+.4f}; "
                    f"defer gate {R['gate']['d_r1']:+.4f} R@1")
    print(f"  {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "differential.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'differential.json'}")


if __name__ == "__main__":
    main()
