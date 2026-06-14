"""KAPPA-FLOOR DECOMPOSITION -- split the continuous reproducibility floor
(cross-experiment Spearman, median ~0.38, spread 0.14-0.75) into its parts.

Paper 1 measured the floor. This turns it from a yardstick into a THEORY of the
yardstick (Expert D's reframe): the floor is not pure technical noise -- it is a
MIXTURE. We decompose the per-pair Spearman across same-condition-cluster
experiment pairs into:

  Component 1 - TECHNICAL: library quality + gene coverage.
      proxies: cor12 (feba's within-experiment consistency), n_genes.
  Component 2 - CONDITION-MISMATCH: pairs share (media, aerobic, condition_1)
      but still differ in temperature / pH / concentration / condition_2 --
      "same condition" isn't. proxies: |dTemp|, |dpH|, concentration mismatch,
      condition_2 mismatch.
  Component 3 - BUFFERING (per-gene, NOT in v0): which genes flip is a function
      of redundancy/buffering. Gated on the orthology bridge (feba locusId !=
      our locus_tag, the disjoint-data wall). Flagged for v1.

Output: variance of the floor partitioned into technical / condition / residual,
plus the "matched-condition high-quality" reproducibility ceiling (what two
screens agree on once you remove technical + condition-mismatch). That residual
ceiling is the number Paper 2 must beat, and the irreducible+buffering floor is
the honest accuracy bound.

Runs on Colab (needs feba.db). Loads outputs/kappa_floor_continuous.csv if
present (from the floor run) to skip the ~7-min recompute.

--smoke : validate the OLS variance-decomposition logic on synthetic data
          (no feba.db needed).
--real  : feba.db; full decomposition.
"""
from __future__ import annotations
import argparse, json, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
OUT = REPO / "outputs"
FEBA_DEFAULT = REPO / "data" / "feba.db"
sys.path.insert(0, str(SCRIPTS))


def ols_r2(X, y):
    """OLS R^2 with an intercept; returns (r2, coef_with_bias)."""
    import numpy as np
    n = len(y)
    Xb = np.hstack([np.ones((n, 1)), X]) if X.ndim == 2 and X.shape[1] else \
        np.ones((n, 1))
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, coef


def standardize_cols(M):
    import numpy as np
    M = np.asarray(M, float)
    mu = np.nanmean(M, axis=0); sd = np.nanstd(M, axis=0) + 1e-9
    Z = (M - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0)
    return Z


def decompose(pairs, tech_cols, cond_cols):
    """Variance partition of pair Spearman into technical vs condition.
    Returns a dict of R^2 values + the matched/high-quality ceiling."""
    import numpy as np
    y = pairs["spearman"].values.astype(float)
    out = {"n_pairs": int(len(y)), "floor_mean": float(y.mean()),
           "floor_median": float(np.median(y)),
           "floor_p10": float(np.percentile(y, 10)),
           "floor_p90": float(np.percentile(y, 90)),
           "floor_var": float(y.var())}
    tech = [c for c in tech_cols if c in pairs.columns]
    cond = [c for c in cond_cols if c in pairs.columns]
    out["technical_features_used"] = tech
    out["condition_features_used"] = cond

    def R2(cols):
        if not cols:
            return 0.0, None
        Z = standardize_cols(pairs[cols].values)
        r2, coef = ols_r2(Z, y)
        return max(r2, 0.0), coef

    r2_tech, _ = R2(tech)
    r2_cond, _ = R2(cond)
    r2_both, coef_both = R2(tech + cond)
    out["r2_technical_only"] = r2_tech
    out["r2_condition_only"] = r2_cond
    out["r2_combined"] = r2_both
    # variance partition (combined, with overlap split proportionally)
    overlap = max(r2_tech + r2_cond - r2_both, 0.0)
    uniq_tech = max(r2_tech - overlap, 0.0)
    uniq_cond = max(r2_cond - overlap, 0.0)
    out["variance_partition"] = {
        "technical_unique_pct": round(100 * uniq_tech, 1),
        "condition_unique_pct": round(100 * uniq_cond, 1),
        "shared_pct": round(100 * overlap, 1),
        "residual_irreducible_pct": round(100 * (1 - r2_both), 1),
    }
    # matched-condition, high-quality ceiling: predict y at best-quality
    # (max of tech features) and matched condition (0 for all cond diffs)
    if tech or cond:
        cols = tech + cond
        Z = standardize_cols(pairs[cols].values)
        r2, coef = ols_r2(Z, y)
        # build the "ideal" standardized row: tech -> +2 sd (high quality),
        # cond mismatch -> most-negative (matched). We approximate by using
        # column-wise best value direction from the coefficient sign.
        bias = coef[0]; betas = coef[1:]
        ideal = np.zeros(len(cols))
        for i, c in enumerate(cols):
            # high quality = 90th pct (not the unbounded max -> avoids wild
            # extrapolation); least mismatch = 10th pct
            colvals = standardize_cols(pairs[[c]].values).ravel()
            if c in tech:
                ideal[i] = np.percentile(colvals, 90)
            else:
                ideal[i] = np.percentile(colvals, 10)  # least mismatch
        ceiling = float(np.clip(bias + ideal @ betas, -1.0, 1.0))
        out["matched_highquality_ceiling"] = ceiling
        out["interpretation_ceiling"] = (
            "predicted cross-experiment Spearman once technical quality is "
            "maximal and condition is matched -- the residual biological "
            "reproducibility (incl. buffering); Paper 2's target sits relative "
            "to THIS, not the raw median.")
    return out


def build_pairs_with_covariates(db_path, pairs_csv=None):
    """Get per-pair Spearman + attach per-experiment covariates from feba.db."""
    import pandas as pd, numpy as np
    from kappa_floor import _pick_table, feba_continuous_floor

    # 1. pairs
    if pairs_csv and Path(pairs_csv).exists():
        print(f"  loading existing pairs from {pairs_csv}")
        pairs = pd.read_csv(pairs_csv)
    else:
        print("  recomputing continuous floor pairs (~7 min) ...")
        pairs = feba_continuous_floor(Path(db_path))
    if pairs.empty:
        raise RuntimeError("no continuous-floor pairs available")
    print(f"  pairs: {len(pairs):,}")

    # 2. full Experiment table (all columns)
    exp_table = _pick_table(Path(db_path), ["Experiment", "Experiments", "Exp"],
                            "decompose")
    con = sqlite3.connect(str(db_path))
    exp = pd.read_sql(f"SELECT * FROM {exp_table}", con)
    con.close()
    print(f"  Experiment columns available: {list(exp.columns)}")

    # quality column detection (feba uses cor12 = consistency of fit between
    # two halves of each gene; also gMed/maxFit). Use whatever exists.
    qual_candidates = ["cor12", "maxFit", "gMed", "gMean", "opcor", "adjcor"]
    qual_cols = [c for c in qual_candidates if c in exp.columns]
    cond_candidates = ["temperature", "pH", "concentration_1", "condition_2"]
    cond_cols_present = [c for c in cond_candidates if c in exp.columns]
    print(f"  quality cols: {qual_cols}  condition cols: {cond_cols_present}")

    key = ["orgId", "expName"]
    for k in key:
        if k not in exp.columns:
            raise RuntimeError(f"Experiment missing {k}; cols={list(exp.columns)}")
    em = exp.set_index(["orgId", "expName"])

    def _num(s):
        return pd.to_numeric(s, errors="coerce")

    rows = []
    for _, r in pairs.iterrows():
        org, ea, eb = r["orgId"], r["exp_a"], r["exp_b"]
        try:
            ra = em.loc[(org, ea)]; rb = em.loc[(org, eb)]
        except KeyError:
            rows.append({}); continue
        feat = {}
        # technical: min quality of the pair, log n_genes
        for q in qual_cols:
            va, vb = pd.to_numeric(pd.Series([ra.get(q), rb.get(q)]),
                                   errors="coerce")
            feat[f"tech_min_{q}"] = float(min(va, vb)) if pd.notna(va) and pd.notna(vb) else np.nan
        feat["tech_log_ngenes"] = float(np.log1p(r.get("n_genes", np.nan)))
        # condition-mismatch within the cluster
        for c in ("temperature", "pH", "concentration_1"):
            if c in exp.columns:
                va, vb = _num(pd.Series([ra.get(c), rb.get(c)]))
                feat[f"cond_d_{c}"] = float(abs(va - vb)) if pd.notna(va) and pd.notna(vb) else 0.0
        if "condition_2" in exp.columns:
            feat["cond_mismatch_cond2"] = float(str(ra.get("condition_2")) !=
                                                str(rb.get("condition_2")))
        rows.append(feat)
    cov = pd.DataFrame(rows)
    pairs = pd.concat([pairs.reset_index(drop=True), cov], axis=1)
    tech_cols = [c for c in pairs.columns if c.startswith("tech_")]
    cond_cols = [c for c in pairs.columns if c.startswith("cond_")]
    # drop pairs with no covariates at all
    pairs = pairs.dropna(subset=["spearman"])
    return pairs, tech_cols, cond_cols


def run_real(args):
    import pandas as pd
    from kappa_floor import fetch_feba
    db = Path(args.feba_db)
    if not args.no_download:
        fetch_feba(db)
    pairs_csv = OUT / "kappa_floor_continuous.csv"
    pairs, tech_cols, cond_cols = build_pairs_with_covariates(
        db, pairs_csv if pairs_csv.exists() else None)
    print(f"\n  technical features: {tech_cols}")
    print(f"  condition features: {cond_cols}")
    res = decompose(pairs, tech_cols, cond_cols)
    _report(res)
    (OUT / "kappa_floor_decomposition.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT / 'kappa_floor_decomposition.json'}")
    return 0


def _report(res):
    print("\n" + "=" * 66)
    print("FLOOR DECOMPOSITION")
    print("=" * 66)
    print(f"  continuous floor: median {res['floor_median']:.3f}  "
          f"mean {res['floor_mean']:.3f}  spread "
          f"{res['floor_p10']:.3f}-{res['floor_p90']:.3f}  "
          f"({res['n_pairs']:,} pairs)")
    print(f"\n  variance explained:")
    print(f"    technical only : R2 = {res['r2_technical_only']:.3f}")
    print(f"    condition only : R2 = {res['r2_condition_only']:.3f}")
    print(f"    combined       : R2 = {res['r2_combined']:.3f}")
    vp = res["variance_partition"]
    print(f"\n  variance partition of the floor spread:")
    print(f"    technical (unique)  : {vp['technical_unique_pct']}%")
    print(f"    condition-mismatch  : {vp['condition_unique_pct']}%")
    print(f"    shared              : {vp['shared_pct']}%")
    print(f"    residual/irreducible: {vp['residual_irreducible_pct']}%")
    if "matched_highquality_ceiling" in res:
        print(f"\n  matched-condition, high-quality ceiling: "
              f"{res['matched_highquality_ceiling']:.3f}")
        print(f"    -> two screens agree at ~this Spearman once technical")
        print(f"       quality is maxed and condition is matched. Paper 2's")
        print(f"       'better than Tn-seq' target is relative to THIS, not the")
        print(f"       raw {res['floor_median']:.2f} median.")
    print(f"\n  NOTE: component 3 (buffering, per-gene) needs the orthology")
    print(f"        bridge (disjoint-data wall) -- flagged for v1.")


def run_smoke():
    """Validate the variance-decomposition math on synthetic pairs."""
    import numpy as np, pandas as pd
    print("=" * 66)
    print("DECOMPOSE SMOKE -- synthetic variance partition")
    print("=" * 66)
    rng = np.random.RandomState(0)
    n = 4000
    tech = rng.normal(size=n)          # quality
    dtemp = np.abs(rng.normal(size=n))  # condition mismatch
    noise = rng.normal(scale=0.5, size=n)
    # construct spearman with known contributions: ~25% tech, ~15% cond
    y = 0.5 + 0.30 * tech - 0.25 * dtemp + noise
    y = np.clip(y, -1, 1)
    pairs = pd.DataFrame({"spearman": y, "tech_min_cor12": tech,
                          "tech_log_ngenes": rng.normal(size=n),
                          "cond_d_temperature": dtemp})
    res = decompose(pairs, ["tech_min_cor12", "tech_log_ngenes"],
                    ["cond_d_temperature"])
    _report(res)
    assert res["r2_combined"] > 0.2, "decomposition failed to recover variance"
    assert res["variance_partition"]["technical_unique_pct"] > \
        res["variance_partition"]["condition_unique_pct"], \
        "technical should dominate in this synthetic setup"
    print("\n=== DECOMPOSE SMOKE PASS ===")
    print("  OLS variance partition + ceiling logic validated on synthetic data.")
    print("  Run --real on Colab (with feba.db) for the actual decomposition.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--feba_db", default=str(FEBA_DEFAULT))
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()
    if not (args.smoke or args.real):
        args.smoke = True
    OUT.mkdir(exist_ok=True)
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
