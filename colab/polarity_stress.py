"""STRESS-TESTING THE POLARITY RESULT AGAINST POWER AND LOCUS LEAKAGE.

THE CLAIM UNDER TEST. `effectsize.py` reported that activating and repressive elements are distinguishable:
AUPRC 0.6507 against a 0.194 base rate (3.36x lift), AUROC 0.8584, permuted control 0.174, on 820
CRISPR-validated pairs of which 159 (19.4%) are repressive. It was called the one thing in several rounds that
cleared its controls decisively. That makes it exactly the result most worth attacking.

WHAT IS ALREADY SUSPICIOUS IN THAT TABLE. Distance ALONE reached AUROC 0.8038 and AUPRC 0.5152. So the large
majority of the headline discrimination is available from genomic distance with no regulatory information at
all, and the activity/CTCF features add 0.5152 -> 0.6507 AUPRC on top. Any honest reading of "sign is
predictable" has to separate those.

THREAT 1 -- POWER. The same benchmark's effect MAGNITUDE turned out to be predictable only from detection
power (R2 +0.0881 with power, every biological arm net-negative), while the marginal correlation of
|EffectSize| with power was a misleading -0.058. If repressive elements systematically have smaller absolute
effects, they sit in a different power regime, and any feature correlated with power becomes a polarity
predictor without carrying polarity information. Tested by: a power-only arm, and a matched-subsample test
where repressive and activating pairs are matched on power, |EffectSize| and distance.

THREAT 2 -- LOCUS LEAKAGE AND EFFECTIVE SAMPLE SIZE. Chromosome-grouped folds are gene-disjoint (verified 0 of
2,344 genes span a fold), so a gene cannot be memorised across folds. But that is not the whole risk:
  - one element can be tested against several genes, and one gene against several elements, so the 820 rows
    are not 820 independent observations;
  - if the 159 repressive calls concentrate in a few genes or loci, the EFFECTIVE n is the number of clusters,
    not the number of pairs, and a pair-level permutation p-value is anticonservative by that factor.
Tested by: counting distinct elements, genes and 1-Mb loci among the repressive set; re-scoring under
element-held-out and gene-held-out splits; and replacing the pair-level permutation with a CLUSTER permutation
that shuffles sign within gene and within (cell type x distance decile), which preserves the confound
structure and destroys only the element-specific signal.

THREAT 3 -- CELL TYPE. K562 supplies 717 of 820 pairs. If the repressive fraction differs by cell type, a
model can reach the base-rate lift by learning cell type. Tested by scoring within K562 alone.

A RESULT THAT SURVIVES ALL OF THIS is worth keeping. One that does not should be downgraded in the record,
which is the point of running it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEEDS = (0, 1, 2, 3, 4)
LOCUS = 1_000_000


def folds_by(key, seed):
    """Group folds on an arbitrary key (element, gene, locus), so a whole group is held out together."""
    u = sorted(set(key))
    a = np.random.default_rng(seed).permutation(len(u))
    m = {k: a[i] % 5 for i, k in enumerate(u)}
    return np.array([m[k] for k in key])


def score(X, t, f, seeds=SEEDS, folder=None, key=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from ranking_gate import fit_oof
    ap, au = [], []
    for s in seeds:
        ff = folder(key, s) if folder is not None else f
        p = fit_oof(X, t, ff)
        ap.append(average_precision_score(t, p))
        au.append(roc_auc_score(t, p))
    return float(np.mean(ap)), float(np.mean(au))


def main():
    from scipy import stats
    from sklearn.metrics import average_precision_score
    from ranking_gate import load, build, folds_chrom, fit_oof, fnum, ACT, CTCFC
    print("=" * 100)
    print("POLARITY STRESS TEST -- power, locus leakage, effective n, cell type")
    print("=" * 100)
    rows = load()
    d = build(rows)
    y = d["y"]
    sig = y == 1
    eff = np.array([fnum(r, "EffectSize") for r in rows])[sig]
    pw = np.array([fnum(r, "PowerAtEffectSize25") for r in rows])[sig]
    dist = d["dist"][sig]
    ct = d["ct"][sig]
    ch = d["ch"][sig]
    srows = [r for r, k in zip(rows, sig) if k]
    elem = np.array([f"{r['chrom']}:{r['chromStart']}-{r['chromEnd']}" for r in srows])
    gene = np.array([f"{r['measuredGeneSymbol']}|{r['CellType']}" for r in srows])
    locus = np.array([f"{r['chrom']}:{int(r['chromStart'])//LOCUS}" for r in srows])
    rep = (eff > 0).astype(int)
    ae = np.abs(eff)
    BASE = np.hstack([d["D"], d["A"], d["C"], d["P"]])[sig]
    br = rep.mean()
    print(f"  {len(rep)} causal pairs, {int(rep.sum())} repressive, base rate {br:.4f}")

    R = {"n": int(len(rep)), "n_repressive": int(rep.sum()), "base_rate": float(br)}

    # ---- descriptive: what actually differs between the two classes ----
    print("\n  [A] WHAT DIFFERS BETWEEN REPRESSIVE AND ACTIVATING")
    print(f"    {'quantity':22s} {'repressive':>12s} {'activating':>12s} {'MWU p':>10s}")
    desc = {}
    for nm, v in (("|EffectSize|", ae), ("distance (bp)", dist), ("power@ES25", pw),
                  ("log10 distance", np.log10(dist))):
        a, b = v[rep == 1], v[rep == 0]
        p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
        print(f"    {nm:22s} {np.median(a):12.4g} {np.median(b):12.4g} {p:10.3g}")
        desc[nm] = {"repressive_median": float(np.median(a)), "activating_median": float(np.median(b)),
                    "mwu_p": float(p)}
    R["descriptive"] = desc
    print("    repressive fraction by cell type:")
    for c in sorted(set(ct)):
        m = ct == c
        print(f"      {c:9s} n={int(m.sum()):4d}  repressive {rep[m].mean():.3f}")
    R["repressive_frac_by_cell"] = {c: float(rep[ct == c].mean()) for c in sorted(set(ct))}

    # ---- effective sample size ----
    print("\n  [B] EFFECTIVE SAMPLE SIZE -- are the 820 rows independent?")
    for nm, k in (("elements", elem), ("genes (x cell)", gene), ("1-Mb loci", locus)):
        tot = len(set(k))
        rk = len(set(k[rep == 1]))
        print(f"    {nm:16s} {tot:5d} distinct overall; {rk:4d} distinct among the "
              f"{int(rep.sum())} repressive  -> {int(rep.sum())/max(rk,1):.2f} pairs per cluster")
        R.setdefault("clusters", {})[nm] = {"distinct": tot, "distinct_repressive": rk}

    # ---- arms, under three split schemes ----
    print("\n  [C] AUPRC / AUROC UNDER PROGRESSIVELY STRICTER SPLITS")
    ARMS = {"power only": pw.reshape(-1, 1),
            "distance only": d["D"][sig],
            "distance + power": np.hstack([d["D"][sig], pw.reshape(-1, 1)]),
            "full biology": BASE,
            "full + power": np.hstack([BASE, pw.reshape(-1, 1)])}
    SPLITS = {"chromosome": (folds_chrom, ch), "element-held-out": (folds_by, elem),
              "gene-held-out": (folds_by, gene), "1-Mb locus-held-out": (folds_by, locus)}
    print(f"    {'arm':18s}" + "".join(f"{s:>22s}" for s in SPLITS))
    R["arms"] = {}
    for tag, X in ARMS.items():
        row = f"    {tag:18s}"
        R["arms"][tag] = {}
        for sname, (folder, key) in SPLITS.items():
            ap, au = score(X, rep, None, folder=folder, key=key)
            row += f"{ap:11.4f}/{au:.4f}"
            R["arms"][tag][sname] = {"auprc": ap, "auroc": au}
        print(row)
    print(f"    ^ base rate {br:.4f}. If 'distance only' is close to 'full biology', polarity is geometry.")

    # ---- cluster permutation: destroy element-specific signal, keep the confounds ----
    print("\n  [D] CLUSTER PERMUTATION -- shuffle sign WITHIN a stratum, preserving its confounds")
    f0 = folds_chrom(ch, 0)
    real_ap = score(BASE, rep, f0, seeds=(0, 1, 2))[0]
    print(f"    real (full biology, chromosome folds): AUPRC {real_ap:.4f}")
    dec = np.digitize(np.log10(dist), np.quantile(np.log10(dist), np.linspace(0, 1, 11))[1:-1])
    STRAT = {"unrestricted (as before)": np.zeros(len(rep), int),
             "within gene": np.unique(gene, return_inverse=True)[1],
             "within cell x distance decile": np.array([hash((c, int(x))) % 10**6
                                                        for c, x in zip(ct, dec)]),
             "within power decile": np.digitize(pw, np.quantile(pw, np.linspace(0, 1, 11))[1:-1])}
    R["cluster_permutation"] = {"real_auprc": real_ap}
    for sname, st in STRAT.items():
        vals = []
        for i in range(12):
            g = np.random.default_rng(500 + i)
            perm = rep.copy()
            for s_ in np.unique(st):
                m = np.where(st == s_)[0]
                if len(m) > 1:
                    perm[m] = rep[m][g.permutation(len(m))]
            vals.append(average_precision_score(perm, fit_oof(BASE, perm, f0)))
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        z = (real_ap - mu) / max(sd, 1e-9)
        print(f"    {sname:32s} permuted AUPRC {mu:.4f} +/- {sd:.4f}   z {z:+6.2f}")
        R["cluster_permutation"][sname] = {"mean": mu, "sd": sd, "z": z}

    # ---- matched subsample: equalise power, |effect| and distance ----
    print("\n  [E] MATCHED SUBSAMPLE -- repressive vs activating equalised on power, |effect|, distance")
    rng = np.random.default_rng(0)
    lg = np.log10(dist)
    key3 = np.column_stack([
        np.digitize(pw, np.quantile(pw, [0.5])),
        np.digitize(ae, np.quantile(ae, [0.25, 0.5, 0.75])),
        np.digitize(lg, np.quantile(lg, [0.25, 0.5, 0.75]))])
    ks = [tuple(k) for k in key3]
    pool_r = {}
    pool_a = {}
    for i, k in enumerate(ks):
        (pool_r if rep[i] else pool_a).setdefault(k, []).append(i)
    keep = []
    for k, ri in pool_r.items():
        ai = pool_a.get(k, [])
        n = min(len(ri), len(ai))
        if n == 0:
            continue
        keep += list(ri[:n]) + list(rng.choice(ai, n, replace=False))
    keep = np.array(sorted(keep))
    print(f"    matched set: {len(keep)} pairs, {int(rep[keep].sum())} repressive "
          f"(base rate {rep[keep].mean():.4f})")
    R["matched"] = {"n": int(len(keep)), "base_rate": float(rep[keep].mean())}
    if len(keep) >= 100:
        for nm, v in (("|EffectSize|", ae), ("power", pw), ("log10 dist", lg)):
            a, b = v[keep][rep[keep] == 1], v[keep][rep[keep] == 0]
            print(f"      residual imbalance {nm:14s} MWU p "
                  f"{stats.mannwhitneyu(a, b, alternative='two-sided')[1]:.3g}")
        for tag, X in (("distance only", d["D"][sig]), ("full biology", BASE)):
            ap, au = score(X[keep], rep[keep], None, folder=folds_by, key=ch[keep], seeds=(0, 1, 2))
            lift = ap / max(rep[keep].mean(), 1e-9)
            print(f"      {tag:16s} AUPRC {ap:.4f} (base {rep[keep].mean():.4f}, {lift:.2f}x)  "
                  f"AUROC {au:.4f}")
            R["matched"][tag] = {"auprc": ap, "auroc": au, "lift": lift}
    else:
        print("      too few matched pairs to score")

    # ---- verdict ----
    print("\n" + "=" * 100)
    strict = min(R["arms"]["full biology"][s]["auprc"] for s in SPLITS)
    dist_only = R["arms"]["distance only"]["chromosome"]["auprc"]
    full_chr = R["arms"]["full biology"]["chromosome"]["auprc"]
    pw_only = R["arms"]["power only"]["chromosome"]["auprc"]
    zg = R["cluster_permutation"].get("within gene", {}).get("z", np.nan)
    zp = R["cluster_permutation"].get("within power decile", {}).get("z", np.nan)
    mfull = R["matched"].get("full biology", {})
    geometry_share = dist_only / max(full_chr, 1e-9)
    parts = [f"distance alone supplies {100*geometry_share:.0f}% of the headline AUPRC "
             f"({dist_only:.4f} of {full_chr:.4f}), so most of the discrimination is GEOMETRY"]
    if pw_only > 1.3 * br:
        parts.append(f"power ALONE reaches AUPRC {pw_only:.4f} ({pw_only/br:.2f}x base rate), so detection "
                     f"power is itself a polarity predictor in this benchmark")
    else:
        parts.append(f"power alone is not a predictor (AUPRC {pw_only:.4f} vs base {br:.4f})")
    parts.append(f"the biology survives the strictest split at AUPRC {strict:.4f}" if strict > 1.3 * br
                 else f"the biology does NOT survive the strictest split (AUPRC {strict:.4f} vs base {br:.4f})")
    if np.isfinite(zg):
        parts.append(f"within-gene cluster permutation z {zg:+.2f}" +
                     (" (survives)" if zg > 3 else " (DOES NOT survive -- the signal is gene-level, "
                      "not element-level)"))
    if mfull:
        parts.append(f"after matching on power, |effect| and distance the lift falls to "
                     f"{mfull['lift']:.2f}x (AUPRC {mfull['auprc']:.4f}, AUROC {mfull['auroc']:.4f})")
    R["verdict"] = ". ".join(parts) + "."
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "polarity_stress.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'polarity_stress.json'}")


if __name__ == "__main__":
    main()
