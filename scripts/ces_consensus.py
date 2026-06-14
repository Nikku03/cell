"""STEP 1 of the road to the rho=0.77 ceiling: CES consensus label.

The kappa-floor decomposition showed ~48% of cross-experiment disagreement is
TECHNICAL (library quality), not biology. Training a predictor on a single
noisy screen therefore caps it at the raw floor (rho~0.38). The fix, borrowed
from cancer-genomics Combined Essentiality Scoring (CES): for each
(organism x condition-cluster x gene), average the fitness across the screens
of that cluster, WEIGHTED BY LIBRARY QUALITY (feba's cor12), so technical noise
is averaged down before the model ever sees it. This is the single biggest
lever toward the matched-quality ceiling.

Condition cluster = (media, aerobic, condition_1) -- same keying as the floor.

OUTPUT (outputs/ces_consensus.parquet), one row per (organism, locusId,
condition_cluster):
  consensus_fitness        cor12-weighted mean of per-screen fitness
  consensus_essential      cor12-weighted fraction of screens calling essential
                           (fit < -2 AND |t| >= 4)  -> in [0,1]
  n_screens                # screens in the cluster that measured this gene
  mean_cor12               mean library quality of those screens  (abstention
                           signal -- conservative: abstain when low)
  total_weight             sum of cor12 weights (inverse-variance proxy)
  within_disagreement      weighted SD of fitness across screens (the
                           per-gene reproducibility; high => buffered/noisy)

This consensus is Paper 2's training label. The abstention head (built next)
keys on mean_cor12 / n_screens. Conservative bias: abstain on the bottom
~15-20% by quality, not 40%.

--smoke : validate the weighted-consensus math on synthetic data (no feba.db)
--real  : compute the consensus from feba.db on Colab
"""
from __future__ import annotations
import argparse, json, sqlite3, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
OUT = REPO / "outputs"
FEBA_DEFAULT = REPO / "data" / "feba.db"
sys.path.insert(0, str(SCRIPTS))

ESS_FIT_T = -2.0
ESS_T_T = 4.0


def _weighted_consensus(M, E, W):
    """M (genes x screens) fitness with NaN where unmeasured; E same shape
    binary-essential; W (screens,) quality weights >= 0. Returns dict of
    per-gene arrays."""
    import numpy as np
    M = np.asarray(M, float); E = np.asarray(E, float); W = np.asarray(W, float)
    mask = ~np.isnan(M)                      # measured cells
    Wb = np.where(mask, W[None, :], 0.0)     # weight only where measured
    wsum = Wb.sum(axis=1)
    safe = wsum > 0
    cons = np.full(M.shape[0], np.nan)
    cons[safe] = np.nansum(np.where(mask, M, 0.0) * Wb, axis=1)[safe] / wsum[safe]
    # weighted fraction essential
    cons_ess = np.full(M.shape[0], np.nan)
    cons_ess[safe] = (np.nansum(np.where(mask, E, 0.0) * Wb, axis=1)[safe]
                      / wsum[safe])
    n_screens = mask.sum(axis=1)
    mean_cor12 = np.full(M.shape[0], np.nan)
    cnt = mask.sum(axis=1)
    has = cnt > 0
    mean_cor12[has] = (np.where(mask, W[None, :], 0.0).sum(axis=1)[has]
                       / cnt[has])
    # weighted SD across screens (reproducibility)
    disagree = np.full(M.shape[0], np.nan)
    multi = (n_screens >= 2) & safe
    if multi.any():
        diff2 = np.where(mask, (M - cons[:, None]) ** 2, 0.0)
        var = np.nansum(diff2 * Wb, axis=1) / wsum
        disagree[multi] = np.sqrt(np.maximum(var[multi], 0))
    return {"consensus_fitness": cons, "consensus_essential": cons_ess,
            "n_screens": n_screens, "mean_cor12": mean_cor12,
            "total_weight": wsum, "within_disagreement": disagree}


def _read_experiments(db_path):
    import pandas as pd
    from kappa_floor import _pick_table
    exp_table = _pick_table(db_path, ["Experiment", "Experiments", "Exp"],
                            "ces consensus")
    con = sqlite3.connect(str(db_path))
    exp = pd.read_sql(f"SELECT * FROM {exp_table}", con)
    con.close()
    for c in ("orgId", "expName"):
        if c not in exp.columns:
            raise RuntimeError(f"Experiment missing {c}: {list(exp.columns)}")
    # quality weight: cor12 if present else maxFit-derived else 1
    qcol = next((c for c in ("cor12", "opcor", "adjcor") if c in exp.columns),
                None)
    if qcol is None:
        print("  WARNING: no cor12-like quality column; using equal weights")
        exp["_w"] = 1.0
    else:
        exp["_w"] = pd.to_numeric(exp[qcol], errors="coerce").clip(lower=0)
        exp["_w"] = exp["_w"].fillna(exp["_w"].median())
        print(f"  quality weight column: {qcol} "
              f"(median {exp['_w'].median():.2f})")
    for c in ("media", "aerobic", "condition_1"):
        if c not in exp.columns:
            exp[c] = "-"
    exp["cluster"] = exp.apply(
        lambda r: f"{r.media or '-'}|{r.aerobic or '-'}|{r.condition_1 or '-'}",
        axis=1)
    return exp, exp_table


def run_real(args):
    import pandas as pd, numpy as np
    from kappa_floor import fetch_feba, _pick_table
    db = Path(args.feba_db)
    if not args.no_download:
        fetch_feba(db)
    print("=" * 64)
    print("CES CONSENSUS -- cor12-weighted fitness per (org, cluster, gene)")
    print("=" * 64)
    exp, _ = _read_experiments(db)
    fit_table = _pick_table(db, ["GeneFitness", "Fitness", "GeneFit"], "ces")
    orgs = sorted(exp.orgId.unique())
    print(f"  {len(exp):,} experiments over {len(orgs)} orgs; "
          f"{exp.cluster.nunique():,} (org-agnostic) cluster keys")

    con = sqlite3.connect(str(db))
    out_rows = []
    for oi, org in enumerate(orgs, 1):
        eg = exp[exp.orgId == org]
        sub = pd.read_sql(
            f"SELECT locusId, expName, fit, t FROM {fit_table} WHERE orgId=?",
            con, params=(org,))
        if sub.empty:
            continue
        sub["ess"] = ((sub.fit < ESS_FIT_T) &
                      (sub.t.abs() >= ESS_T_T)).astype(int)
        for cluster, cg in eg.groupby("cluster"):
            exps = cg.expName.tolist()
            ss = sub[sub.expName.isin(exps)]
            if ss.empty:
                continue
            fitw = ss.pivot_table(index="locusId", columns="expName",
                                  values="fit", aggfunc="first")
            essw = ss.pivot_table(index="locusId", columns="expName",
                                  values="ess", aggfunc="first")
            essw = essw.reindex(columns=fitw.columns)
            wmap = cg.set_index("expName")["_w"]
            W = wmap.reindex(fitw.columns).fillna(0).values
            if W.sum() <= 0:
                W = np.ones(len(fitw.columns))
            res = _weighted_consensus(fitw.values, essw.values, W)
            d = pd.DataFrame({"locusId": fitw.index, **res})
            d["organism"] = org
            d["condition_cluster"] = cluster
            d = d.dropna(subset=["consensus_fitness"])
            out_rows.append(d)
        if oi % 10 == 0:
            print(f"    ...{oi}/{len(orgs)} orgs, "
                  f"{sum(len(x) for x in out_rows):,} rows")
    con.close()
    df = pd.concat(out_rows, ignore_index=True)
    cols = ["organism", "locusId", "condition_cluster", "consensus_fitness",
            "consensus_essential", "n_screens", "mean_cor12", "total_weight",
            "within_disagreement"]
    df = df[cols]
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "ces_consensus.parquet", index=False)

    # diagnostics
    multi = df[df.n_screens >= 2]
    print(f"\n  consensus rows: {len(df):,}  "
          f"({df.organism.nunique()} orgs, {df.condition_cluster.nunique()} "
          f"distinct clusters)")
    print(f"  multi-screen cells (n>=2): {len(multi):,} "
          f"({100*len(multi)/len(df):.0f}%)")
    print(f"  consensus_fitness: median {df.consensus_fitness.median():.2f}  "
          f"essential(>=0.5): {int((df.consensus_essential>=0.5).sum()):,}")
    q = df.mean_cor12.quantile([0.15, 0.5, 0.85])
    print(f"  mean_cor12 quantiles 15/50/85: "
          f"{q.iloc[0]:.2f}/{q.iloc[1]:.2f}/{q.iloc[2]:.2f}")
    print(f"  CONSERVATIVE abstention threshold (15th pct cor12): "
          f"{q.iloc[0]:.3f}  -> abstain below this")
    summary = {
        "n_rows": int(len(df)), "n_orgs": int(df.organism.nunique()),
        "n_clusters": int(df.condition_cluster.nunique()),
        "pct_multiscreen": round(100*len(multi)/len(df), 1),
        "consensus_essential_count": int((df.consensus_essential>=0.5).sum()),
        "mean_cor12_p15": float(q.iloc[0]), "mean_cor12_p50": float(q.iloc[1]),
        "abstain_threshold_cor12_p15": float(q.iloc[0]),
    }
    (OUT / "ces_consensus_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'ces_consensus.parquet'} + summary")
    return 0


def run_smoke():
    import numpy as np, pandas as pd
    print("=" * 64)
    print("CES CONSENSUS SMOKE -- weighted-consensus math on synthetic data")
    print("=" * 64)
    # cluster A: 3 screens, cor12 weights [0.9, 0.6, 0.1]; cluster B: 1 screen.
    # gene g1 fitness across A screens: [-3.0, -2.0, 5.0] (3rd is junky, low w)
    M = np.array([[-3.0, -2.0, 5.0],     # g1: clean screens say essential
                  [0.1, np.nan, 0.0]])   # g2: one screen missing
    E = np.array([[1, 1, 0],
                  [0, np.nan, 0]])
    W = np.array([0.9, 0.6, 0.1])
    r = _weighted_consensus(M, E, W)
    # g1 weighted mean = (0.9*-3 + 0.6*-2 + 0.1*5)/(0.9+0.6+0.1)
    exp_g1 = (0.9*-3 + 0.6*-2 + 0.1*5) / 1.6
    print(f"  g1 consensus={r['consensus_fitness'][0]:.3f} "
          f"(expected {exp_g1:.3f})  n_screens={r['n_screens'][0]}")
    assert abs(r["consensus_fitness"][0] - exp_g1) < 1e-9, "weighted mean wrong"
    assert r["n_screens"][0] == 3 and r["n_screens"][1] == 2, "n_screens wrong"
    # g2: screen2 missing -> weights 0.9,0.1 over [0.1,0.0]
    exp_g2 = (0.9*0.1 + 0.1*0.0) / (0.9 + 0.1)
    print(f"  g2 consensus={r['consensus_fitness'][1]:.3f} "
          f"(expected {exp_g2:.3f}, screen2 NaN excluded)")
    assert abs(r["consensus_fitness"][1] - exp_g2) < 1e-9, "NaN handling wrong"
    # consensus_essential g1 = (0.9*1+0.6*1+0.1*0)/1.6
    exp_ess = (0.9*1 + 0.6*1 + 0.1*0) / 1.6
    print(f"  g1 consensus_essential={r['consensus_essential'][0]:.3f} "
          f"(expected {exp_ess:.3f})")
    assert abs(r["consensus_essential"][0] - exp_ess) < 1e-9
    # within_disagreement for g1 should be > 0 (screens disagree)
    print(f"  g1 within_disagreement={r['within_disagreement'][0]:.3f} (>0)")
    assert r["within_disagreement"][0] > 0
    # single-screen cell: disagreement is NaN
    r2 = _weighted_consensus(np.array([[-3.0]]), np.array([[1]]),
                             np.array([0.8]))
    assert np.isnan(r2["within_disagreement"][0]), "single-screen disagree"
    assert r2["n_screens"][0] == 1
    print(f"  single-screen: consensus={r2['consensus_fitness'][0]:.2f} "
          f"disagree=NaN  n=1  OK")
    # all-zero weights fallback handled at caller (equal weight) -- not here
    print("\n=== CES CONSENSUS SMOKE PASS ===")
    print("  cor12-weighting, NaN exclusion, fraction-essential, disagreement,")
    print("  and single-screen handling all validated. Run --real on Colab.")
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
    return run_real(args) if args.real else run_smoke()


if __name__ == "__main__":
    sys.exit(main())
