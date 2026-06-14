"""FLUX-FIELD LINCHPIN (v0) -- does gene gain/loss RATE carry essentiality
information ORTHOGONAL to static conservation?

This adjudicates the panel's "Dispensability Response Field" thesis (Expert E):
static conservation (prevalence / family_frac) is the TIME-INTEGRAL of the
gene gain/loss process; integrating destroys the RATE. The rate -- how often a
gene is independently gained/lost across the tree -- is claimed to be a
separate, orthogonal coordinate that still predicts essentiality.

We cannot run full Panstripe over AllTheBacteria (2.4M genomes) locally, so
this is the LOW-POWER proxy on our 48 organisms. Asymmetric evidence:
  - a POSITIVE result (flux adds over conservation on held-out clades) is
    STRONG -- signal detectable even at n=48.
  - a NULL is INCONCLUSIVE -- 48 orgs may be underpowered; escalate to the
    AllTheBacteria/Panstripe version before concluding E is wrong.

METRIC (tree-free, non-circular):
  For each ortholog group (OG), using the EXTERNAL clade phylogeny
  (clade_splits.csv, 34 clades over 48 orgs):
    prevalence       = (# orgs with the OG) / 48          [static conservation]
    n_clades_present = # distinct clades the OG appears in [dispersion]
  An OG present in n orgs concentrated in few clades was lost/gained RARELY
  (low flux); the same n scattered across many clades implies MANY independent
  gain/loss events (high flux). We residualize n_clades_present on a flexible
  function of prevalence -> FLUX_RESID = "more/fewer independent clade
  occurrences than expected for this prevalence" = the rate orthogonal to the
  integral. (Tree-free avoids the circularity of building a tree from the same
  gene content then measuring flux on it.)

TEST (leave-one-clade-out, the panel's kill-gate #1):
  Does [conservation + flux_resid] beat [conservation] at predicting per-gene
  essentiality on a held-out clade? Plus:
   - ORTHOGONALITY: corr(flux_resid, family_frac) should be low.
   - PHYLOGENY-SHUFFLE NULL: shuffle org->clade labels, recompute flux; the
     lift must VANISH (else it's an artifact, not phylogenetic structure).

VERDICT: flux ALIVE if it adds AUPRC over conservation on held-out clades AND
the lift collapses under phylogeny-shuffle. Else flux was conservation-in-
disguise (or underpowered at n=48).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "data" / "drive_import" / "labels"
OUT = REPO / "outputs"


def auprc(y, s):
    import numpy as np
    y = np.asarray(y).astype(int); s = np.asarray(s).astype(float)
    m = ~np.isnan(s); y, s = y[m], s[m]
    if y.sum() < 5 or len(np.unique(y)) < 2:
        return None
    order = np.argsort(-s, kind="stable"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(y.sum(), 1)
    return float(np.trapezoid(prec, rec))


def logreg_fit(X, y, iters=50, l2=1e-3):
    """Tiny IRLS logistic regression (no sklearn dependency). X has a bias col."""
    import numpy as np
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = p * (1 - p) + 1e-6
        grad = X.T @ (p - y) + l2 * w
        H = (X * W[:, None]).T @ X + l2 * np.eye(d)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def logreg_pred(X, w):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def standardize(train, *others):
    import numpy as np
    mu = np.nanmean(train, axis=0); sd = np.nanstd(train, axis=0) + 1e-9
    out = [(train - mu) / sd]
    for o in others:
        out.append((o - mu) / sd)
    return out


def compute_flux(orth, org2clade, shuffle_seed=None):
    """Per-OG prevalence + n_clades_present + flux_resid. If shuffle_seed is
    set, permute org->clade (the phylogeny-shuffle null)."""
    import numpy as np, pandas as pd
    o2c = dict(org2clade)
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        orgs = list(o2c.keys()); clades = list(o2c.values())
        rng.shuffle(clades)
        o2c = dict(zip(orgs, clades))
    df = orth[["organism", "og_id"]].drop_duplicates().copy()
    df["clade"] = df.organism.map(o2c)
    g = df.groupby("og_id").agg(
        prevalence=("organism", "nunique"),
        n_clades=("clade", "nunique")).reset_index()
    g["prevalence"] = g.prevalence / orth.organism.nunique()
    # residualize n_clades on a quadratic in prevalence -> flux orthogonal to
    # the static conservation integral
    x = g.prevalence.values; yv = g.n_clades.values.astype(float)
    A = np.vstack([np.ones_like(x), x, x**2]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    g["flux_resid"] = yv - A @ coef
    return g[["og_id", "prevalence", "n_clades", "flux_resid"]]


def evaluate(per_gene, feature_sets, clades):
    """Leave-one-clade-out AUPRC for each named feature set."""
    import numpy as np
    results = {name: [] for name in feature_sets}
    for held in clades:
        tr = per_gene[per_gene.clade != held]
        te = per_gene[per_gene.clade == held]
        if te.essential.sum() < 5 or tr.essential.sum() < 20:
            continue
        ytr = tr.essential.values.astype(float)
        yte = te.essential.values.astype(int)
        for name, cols in feature_sets.items():
            Xtr = tr[cols].values.astype(float)
            Xte = te[cols].values.astype(float)
            Xtr_s, Xte_s = standardize(Xtr, Xte)
            # add bias
            Xtr_b = np.hstack([np.ones((len(Xtr_s), 1)), Xtr_s])
            Xte_b = np.hstack([np.ones((len(Xte_s), 1)), Xte_s])
            w = logreg_fit(Xtr_b, ytr)
            s = logreg_pred(Xte_b, w)
            ap = auprc(yte, s)
            if ap is not None:
                results[name].append(ap)
    return {k: (float(np.mean(v)) if v else None, len(v))
            for k, v in results.items()}


def main() -> int:
    import numpy as np, pandas as pd
    print("=" * 70)
    print("FLUX-FIELD LINCHPIN (v0) -- is gain/loss RATE orthogonal to")
    print("conservation AND predictive of essentiality?  (n=48 low-power proxy)")
    print("=" * 70)

    orth = pd.read_csv(LAB / "orthology_features.csv",
                       usecols=["organism", "locus_tag", "og_id",
                                "family_frac_essential_leakfree"])
    lab = pd.read_csv(LAB / "labels.csv",
                      usecols=["organism", "locus_tag", "essential"])
    cs = pd.read_csv(LAB / "clade_splits.csv", usecols=["organism", "clade"])
    org2clade = dict(zip(cs.organism, cs.clade))
    print(f"  orgs={orth.organism.nunique()}  OGs={orth.og_id.nunique()}  "
          f"clades={cs.clade.nunique()}")

    # --- flux field (real) ---
    flux = compute_flux(orth, org2clade)
    print(f"\n[1] FLUX FIELD computed for {len(flux):,} OGs")
    print(f"    prevalence median {flux.prevalence.median():.3f}; "
          f"n_clades median {flux.n_clades.median():.0f}; "
          f"flux_resid sd {flux.flux_resid.std():.3f}")

    # --- per-gene table ---
    pg = (lab.merge(orth, on=["organism", "locus_tag"], how="inner")
             .merge(flux, on="og_id", how="left"))
    pg["clade"] = pg.organism.map(org2clade)
    pg = pg.dropna(subset=["essential", "family_frac_essential_leakfree",
                           "flux_resid", "prevalence", "clade"])
    pg = pg.rename(columns={"family_frac_essential_leakfree": "family_frac"})
    print(f"  per-gene rows: {len(pg):,}  essential: {int(pg.essential.sum()):,}")

    # --- orthogonality check (kill-gate: flux must NOT be conservation) ---
    r_ff = float(np.corrcoef(pg.flux_resid, pg.family_frac)[0, 1])
    r_prev = float(np.corrcoef(pg.flux_resid, pg.prevalence)[0, 1])
    print(f"\n[2] ORTHOGONALITY  corr(flux_resid, family_frac)={r_ff:+.3f}  "
          f"corr(flux_resid, prevalence)={r_prev:+.3f}")
    print(f"    (near 0 => genuinely orthogonal to conservation)")

    # --- leave-one-clade-out: does flux ADD over conservation? ---
    clades = sorted(pg.clade.unique())
    fsets = {
        "conservation (family_frac)": ["family_frac"],
        "conservation+prevalence": ["family_frac", "prevalence"],
        "flux_resid alone": ["flux_resid"],
        "conservation + flux_resid": ["family_frac", "flux_resid"],
        "conservation+prev+flux": ["family_frac", "prevalence", "flux_resid"],
    }
    print(f"\n[3] LEAVE-ONE-CLADE-OUT AUPRC (over {len(clades)} clades):")
    res = evaluate(pg, fsets, clades)
    base = res["conservation (family_frac)"][0]
    for name, (ap, n) in res.items():
        if ap is None:
            print(f"    {name:<32s} n/a"); continue
        delta = (ap - base) if base else 0.0
        flag = ""
        if name == "conservation + flux_resid":
            flag = ("  <<< flux lift" if delta > 0 else "  (no lift)")
        print(f"    {name:<32s} AUPRC={ap:.4f}  Δvs_cons={delta:+.4f}"
              f"  ({n} clades){flag}")

    full = res["conservation + flux_resid"][0]
    lift = (full - base) if (full and base) else None

    # --- phylogeny-shuffle null: lift must vanish ---
    print(f"\n[4] PHYLOGENY-SHUFFLE NULL (shuffle org->clade, recompute flux):")
    null_lifts = []
    for seed in range(5):
        flux_s = compute_flux(orth, org2clade, shuffle_seed=seed)
        pgs = (lab.merge(orth, on=["organism", "locus_tag"], how="inner")
                  .merge(flux_s, on="og_id", how="left"))
        pgs["clade"] = pgs.organism.map(org2clade)
        pgs = pgs.dropna(subset=["essential", "family_frac_essential_leakfree",
                                 "flux_resid", "prevalence", "clade"])
        pgs = pgs.rename(columns={"family_frac_essential_leakfree": "family_frac"})
        rs = evaluate(pgs, {"cons": ["family_frac"],
                            "cons+flux": ["family_frac", "flux_resid"]}, clades)
        b = rs["cons"][0]; f = rs["cons+flux"][0]
        if b and f:
            null_lifts.append(f - b)
    null_mean = float(np.mean(null_lifts)) if null_lifts else None
    print(f"    real flux lift:    {lift:+.4f}" if lift is not None else
          "    real flux lift: n/a")
    print(f"    shuffled flux lift (mean of 5): "
          f"{null_mean:+.4f}" if null_mean is not None else
          "    shuffled flux lift: n/a")

    # --- verdict ---
    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    alive = (lift is not None and lift > 0.005 and
             abs(r_ff) < 0.5 and
             (null_mean is None or lift > 2 * abs(null_mean)))
    if alive:
        print("  >>> FLUX ALIVE.  Gain/loss-rate dispersion adds essentiality")
        print("      signal over conservation on held-out clades, the signal is")
        print("      orthogonal to family_frac, and it collapses under")
        print("      phylogeny-shuffle. The Dispensability-Response-Field thesis")
        print("      survives even at n=48 -> escalate to AllTheBacteria/Panstripe.")
    elif lift is not None and lift > 0:
        print("  >>> WEAK/INCONCLUSIVE.  Small positive lift but not decisive at")
        print("      n=48 (underpowered, as Expert E warned). The full-power test")
        print("      needs gain/loss RATES over AllTheBacteria (Panstripe), not")
        print("      clade-dispersion over 48 orgs. Do not reject the thesis.")
    else:
        print("  >>> NO LIFT at n=48.  Either flux is conservation-in-disguise OR")
        print("      the 48-org proxy is too underpowered to see the rate signal.")
        print("      INCONCLUSIVE per E's asymmetry -- the verdict requires the")
        print("      AllTheBacteria/Panstripe escalation before rejecting.")

    out = OUT / "flux_linchpin_results.json"
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "orthogonality": {"corr_family_frac": r_ff, "corr_prevalence": r_prev},
        "loco_auprc": {k: {"auprc": v[0], "n_clades": v[1]}
                       for k, v in res.items()},
        "flux_lift_over_conservation": lift,
        "phylogeny_shuffle_null_lift": null_mean,
        "verdict_alive": bool(alive),
        "note": "n=48 low-power proxy; positive=strong, null=inconclusive "
                "(needs AllTheBacteria/Panstripe for full power).",
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
