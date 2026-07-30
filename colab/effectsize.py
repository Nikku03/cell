"""STAGE 2: do(enhancer) -> dRNA. THREE QUESTIONS THE BINARY LABEL CANNOT ASK.

WHY THIS IS STAGE 2 AND NOT A SIDE-QUEST. The revised staging said the decisive remaining test is
do(ATAC_e down) -> dRNA_g, because observational predictability is neither necessary nor sufficient for
perturbational causality -- two observational nulls (pseudotime multiome, real-time A549 lag curve) do not
predict a null under intervention. That perturbation already exists on disk and this project has never used it:
the EPCrisprBenchmark ships **EffectSize**, the measured change in the gene when the element is silenced. Every
result in this project so far has thrown that away and used only the binary `Significant` flag.

WHAT THE QUANTITY LOOKS LIKE, checked before any model:

    820 significant pairs, |EffectSize| quartiles 0.0735 / 0.1165 / 0.2292
    159 of 820 (19.4%) have EffectSize > 0 -- silencing the element RAISES the gene, i.e. it is REPRESSIVE
    |EffectSize| vs power       Spearman -0.058   (so magnitude is not a detection-power artefact)
    |EffectSize| vs distance    Spearman -0.120   (closer elements act slightly harder, weakly)

THREE QUESTIONS

 1 MAGNITUDE. Among pairs where a causal effect demonstrably exists, can |EffectSize| be predicted? A model
   that finds edges but cannot rank their strength is a classifier, not a quantitative causal model -- and the
   ABC formula has been argued to be better suited to classification than to effect-size prediction. This
   tests that directly on 820 perturbations.

 2 SIGN. Can an activating element be told from a repressive one? 19.4% are repressive, so this is a real
   subclass the binary label erases, and a model that calls every element activating gets 80.6% "right"
   while being useless. Scored by AUPRC against that 0.194 base rate, not by accuracy.

 3 DOES THE EDGE-FINDER RANK STRENGTH? The existence classifier's own out-of-fold score is correlated with
   |EffectSize| on the same pairs. If a model tuned to find edges cannot order them by how hard they act,
   that is a specific, reportable limitation rather than a vague one.

CONTROLS. Held out by chromosome, which is gene-disjoint (verified 0 of 2,344 genes span a fold in
`adversary.py`). Permuted-target arm for every regression, because 20 features against 820 rows will fit
something. Power is included as a FEATURE in a dedicated arm rather than assumed irrelevant: |EffectSize|
correlates with it at only -0.058, but the check is cheap and the mRNA-protein gate in this project was nearly
derailed by an unexamined assay-quality confound.

n = 820, so this is a small-sample result and every number is reported with its permuted control beside it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SEEDS = (0, 1, 2, 3, 4)


def main():
    from scipy import stats
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, average_precision_score, roc_auc_score
    from ranking_gate import load, build, folds_chrom, fit_oof, fnum, ACT, CTCFC
    print("=" * 100)
    print("STAGE 2 -- do(enhancer) -> dRNA: magnitude, sign, and whether the edge-finder ranks strength")
    print("=" * 100)
    rows = load()
    d = build(rows)
    y = d["y"]
    eff = np.array([fnum(r, "EffectSize") for r in rows])
    pw = np.array([fnum(r, "PowerAtEffectSize25") for r in rows])
    sig = y == 1
    print(f"  {len(rows):,} pairs; {int(sig.sum())} with a demonstrated causal effect")
    print(f"  |EffectSize| median {np.median(np.abs(eff[sig])):.4f}; "
          f"repressive (KO raises gene) {int((eff[sig] > 0).sum())}/{int(sig.sum())} "
          f"= {100*(eff[sig] > 0).mean():.1f}%")

    BASE = np.hstack([d["D"], d["A"], d["C"], d["P"]])
    names = (["log_dist"] + ACT + CTCFC + d["pnames"])
    Xs = BASE[sig]
    chs = d["ch"][sig]
    es = eff[sig]
    ae = np.abs(es)
    pws = pw[sig]
    R = {"n_sig": int(sig.sum()), "median_abs_effect": float(np.median(ae)),
         "frac_repressive": float((es > 0).mean()),
         "abs_effect_vs_power_spearman": float(stats.spearmanr(ae, pws)[0]),
         "features": names}

    def loco_reg(X, t, ch, seeds=SEEDS):
        """Ridge held out by chromosome. Ridge and not a tree: 820 rows against 20+ columns is where a
        boosted tree starts fitting the fold rather than the biology."""
        sc = []
        for s in seeds:
            f = folds_chrom(ch, s)
            pred = np.full(len(t), np.nan)
            for k in range(5):
                te = f == k
                if te.sum() < 10 or (~te).sum() < 40:
                    continue
                mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit((X[~te] - mu) / sd, t[~te])
                pred[te] = m.predict((X[te] - mu) / sd)
            ok = np.isfinite(pred)
            if ok.sum() > 50:
                sc.append(r2_score(t[ok], pred[ok]))
        return (float(np.mean(sc)) if sc else np.nan), pred

    # ---- 1 MAGNITUDE ----
    print(f"\n  [1] MAGNITUDE -- predicting |EffectSize| on {int(sig.sum())} causal pairs, "
          f"held out by chromosome")
    ARMS = {"distance only": d["D"][sig],
            "+activity": np.hstack([d["D"][sig], d["A"][sig]]),
            "+CTCF": np.hstack([d["D"][sig], d["A"][sig], d["C"][sig]]),
            "+competition (full)": Xs,
            "CTRL +power as feature": np.hstack([Xs, pws.reshape(-1, 1)])}
    print(f"    {'arm':26s} {'R2':>8s} {'permuted':>9s} {'net':>8s}   {'Spearman':>9s}")
    rng = np.random.default_rng(0)
    R["magnitude"] = {}
    for tag, X in ARMS.items():
        r2, pred = loco_reg(X, ae, chs)
        perm = np.mean([loco_reg(X, ae[rng.permutation(len(ae))], chs, seeds=(0, 1))[0]
                        for _ in range(3)])
        ok = np.isfinite(pred)
        rho = float(stats.spearmanr(pred[ok], ae[ok])[0]) if ok.sum() > 50 else np.nan
        print(f"    {tag:26s} {r2:+8.4f} {perm:+9.4f} {r2-perm:+8.4f}   {rho:+9.4f}")
        R["magnitude"][tag] = {"r2": r2, "permuted": float(perm), "net": r2 - float(perm),
                               "spearman": rho}

    # ---- 2 SIGN ----
    rep = (es > 0).astype(int)
    print(f"\n  [2] SIGN -- activating vs repressive ({int(rep.sum())} repressive, "
          f"base rate {rep.mean():.3f})")
    print(f"    {'arm':26s} {'AUPRC':>8s} {'AUROC':>8s} {'permuted AUPRC':>15s}")
    R["sign"] = {}
    for tag, X in list(ARMS.items())[:4]:
        ap, au = [], []
        for s in SEEDS:
            f = folds_chrom(chs, s)
            p = fit_oof(X, rep, f)
            ap.append(average_precision_score(rep, p))
            au.append(roc_auc_score(rep, p))
        pm = []
        for i in range(3):
            rp = rep[np.random.default_rng(100 + i).permutation(len(rep))]
            pmp = fit_oof(X, rp, folds_chrom(chs, 0))
            pm.append(average_precision_score(rp, pmp))
        print(f"    {tag:26s} {np.mean(ap):8.4f} {np.mean(au):8.4f} {np.mean(pm):15.4f}")
        R["sign"][tag] = {"auprc": float(np.mean(ap)), "auroc": float(np.mean(au)),
                          "permuted_auprc": float(np.mean(pm)), "base_rate": float(rep.mean())}

    # ---- 3 DOES THE EDGE-FINDER RANK STRENGTH? ----
    print(f"\n  [3] DOES THE EXISTENCE CLASSIFIER RANK STRENGTH?")
    pfull = np.zeros(len(y))
    for s in SEEDS:
        pfull += fit_oof(BASE, y, folds_chrom(d["ch"], s)) / len(SEEDS)
    ps = pfull[sig]
    r_all = stats.spearmanr(ps, ae)
    print(f"    classifier OOF score vs |EffectSize| on causal pairs: Spearman {r_all[0]:+.4f} "
          f"(p {r_all[1]:.3g}, n={len(ps)})")
    # and within cell type, since K562 dominates and could drive a pooled correlation on its own
    for c in sorted(set(d["ct"][sig])):
        m = d["ct"][sig] == c
        if m.sum() >= 20:
            rr = stats.spearmanr(ps[m], ae[m])
            print(f"      {c:9s} n={int(m.sum()):4d}  Spearman {rr[0]:+.4f} (p {rr[1]:.3g})")
    R["classifier_ranks_strength"] = {"spearman": float(r_all[0]), "p": float(r_all[1]),
                                      "n": int(len(ps))}

    # ---- verdict ----
    print("\n" + "=" * 100)
    best = max(R["magnitude"], key=lambda k: R["magnitude"][k]["net"])
    mb = R["magnitude"][best]
    sb = max(R["sign"], key=lambda k: R["sign"][k]["auprc"])
    sv = R["sign"][sb]
    lift = sv["auprc"] / max(sv["base_rate"], 1e-9)
    parts = []
    if mb["net"] > 0.02:
        parts.append(f"MAGNITUDE IS PARTLY PREDICTABLE ({best}: R2 {mb['r2']:+.4f} vs permuted "
                     f"{mb['permuted']:+.4f}, Spearman {mb['spearman']:+.4f})")
    else:
        parts.append(f"MAGNITUDE IS NOT PREDICTABLE (best {best}: R2 {mb['r2']:+.4f} vs permuted "
                     f"{mb['permuted']:+.4f}, net {mb['net']:+.4f})")
    if sv["auprc"] > 1.3 * sv["base_rate"]:
        parts.append(f"SIGN IS PARTLY PREDICTABLE ({sb}: AUPRC {sv['auprc']:.4f} vs base rate "
                     f"{sv['base_rate']:.3f}, {lift:.2f}x lift, AUROC {sv['auroc']:.4f})")
    else:
        parts.append(f"SIGN IS NOT PREDICTABLE (best AUPRC {sv['auprc']:.4f} against a base rate of "
                     f"{sv['base_rate']:.3f}, only {lift:.2f}x)")
    cr = R["classifier_ranks_strength"]
    if cr["p"] < 0.05 and abs(cr["spearman"]) > 0.1:
        parts.append(f"the edge-finder DOES rank strength (Spearman {cr['spearman']:+.4f}, p {cr['p']:.3g})")
    else:
        parts.append(f"the edge-finder does NOT rank strength (Spearman {cr['spearman']:+.4f}, "
                     f"p {cr['p']:.3g}) -- it finds edges without ordering them by how hard they act")
    R["verdict"] = ". ".join(parts) + (f". n={int(sig.sum())} causal pairs, so this is a small-sample "
                                       f"result and 19.4% of those elements are repressive, a subclass the "
                                       f"binary label erases entirely.")
    print(f"  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "effectsize.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'effectsize.json'}")


if __name__ == "__main__":
    main()
