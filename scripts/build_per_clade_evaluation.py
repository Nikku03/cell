"""Per-organism leave-one-out evaluation, grouped by clade.

Motivated by the codon-subset diagnostic finding:
  within-clade (ralstonia) random 5-fold MCC = 0.72
  cross-clade LOO mean MCC                   = 0.49
The bottleneck is cross-clade transfer, not features. This
script measures the DEPLOYMENT-REALISTIC number:

  "Given every other labelled organism (including this organism's
   clade-mates), how well do I predict THIS organism's essential
   genes?"

That is organism-level leave-one-out (LOO-organism). The key
difference from the existing clade-level LOO: family_frac here
EXCLUDES ONLY the held-out organism, so an organism's clade-mates
DO contribute to its family_frac prior. This is leakage-free
(never uses the test organism's own labels) yet realistic (you
really do have clade-mates labelled at deployment time).

The clade effect emerges directly:
  - organisms in well-sampled clades (many clade-mates) get a
    strong within-clade conservation signal -> high MCC
  - singleton-clade organisms get only cross-clade priors -> low MCC

Output:
  per_organism_loo.csv     : one row per organism (MCC, P, R, n, ...)
  per_clade_summary.csv     : aggregated per clade (mean MCC, n_orgs)
  per_clade_evaluation.json : full structured report + the
                               clade-size-vs-MCC relationship

Note on family_frac recomputation:
  We rebuild family_frac per held-out organism from the raw
  OG membership (organism, locus_tag, og_id) joined to labels.
  For OG X and held-out org O:
    frac_excl_O(X) = (essential members of X in orgs != O)
                     / (total members of X in orgs != O)
  NaN if X has no members outside O.

Runtime: ~one XGBoost fit per BERIL organism (~45 fits on
~170k rows). Expect 15-30 min. n_estimators kept at 300 to
bound cost.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",     type=Path, default=DATA_DIR)
    p.add_argument("--n-estimators", type=int,   default=300)
    p.add_argument("--max-depth",    type=int,   default=4)
    p.add_argument("--learning-rate",type=float, default=0.05)
    p.add_argument("--min-test-ess", type=int,   default=10,
                   help="skip organisms with fewer essential genes than "
                        "this (MCC unstable). Default 10.")
    p.add_argument("--out-org",      type=Path, default=DATA_DIR / "per_organism_loo.csv")
    p.add_argument("--out-clade",    type=Path, default=DATA_DIR / "per_clade_summary.csv")
    p.add_argument("--out-json",     type=Path, default=DATA_DIR / "per_clade_evaluation.json")
    p.add_argument("--out-pred",     type=Path, default=DATA_DIR / "loo_organism_predictions.csv",
                   help="per-row predictions across LOO-org loop (for calibration)")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import matthews_corrcoef
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_clade_splits import is_bacterial_clade

    D = args.data_dir
    print("Loading ...")
    labels = pd.read_csv(D / "labels.csv")
    splits = pd.read_csv(D / "clade_splits.csv")
    orth   = pd.read_csv(D / "orthology_features.csv")

    def opt(name):
        pth = D / name
        return pd.read_csv(pth) if pth.exists() else None
    cooc = opt("cooccurrence_features.csv")
    reg  = opt("regulator_features.csv")
    fba  = opt("fba_features.csv")
    syn  = opt("synteny_features.csv")
    blo  = opt("blo_features.csv")

    # base frame
    j = labels.merge(splits[["organism","clade","fold"]], on="organism", how="left")
    j = j[j.clade.notna() & j.clade.apply(is_bacterial_clade)].copy()

    feat_base = ["n_paralogs_in_genome","family_size_total",
                 "family_n_organisms","is_orphan"]
    # bring og_id + base features
    j = j.merge(orth[["organism","locus_tag","og_id"] + feat_base],
                  on=["organism","locus_tag"], how="left")

    # ---- precompute OG essentiality aggregates for LOO-organism family_frac ----
    # need (og_id, organism) essential-count and member-count
    print("Precomputing OG aggregates for per-organism LOO family_frac ...")
    ess_src = j[["organism","og_id","essential"]].dropna(subset=["og_id"]).copy()
    ess_src["essential"] = ess_src["essential"].astype(int)
    og_total_ess = ess_src.groupby("og_id")["essential"].sum()
    og_total_n   = ess_src.groupby("og_id")["essential"].size()
    og_org = ess_src.groupby(["og_id","organism"])["essential"].agg(["sum","size"])
    print(f"  {len(og_total_n):,} OGs, "
          f"{ess_src.organism.nunique()} organisms contributing")

    def family_frac_excl_org(org: str) -> "pd.Series":
        """Return Series indexed by og_id: essential-frac of OG among all
        organisms EXCEPT `org`. NaN where no members remain."""
        # subtract org's contribution
        try:
            sub = og_org.xs(org, level="organism")  # index=og_id, cols sum,size
            e_sub = sub["sum"]; n_sub = sub["size"]
        except KeyError:
            e_sub = pd.Series(0, index=og_total_ess.index)
            n_sub = pd.Series(0, index=og_total_n.index)
        e = og_total_ess.sub(e_sub, fill_value=0)
        n = og_total_n.sub(n_sub, fill_value=0)
        frac = e / n.where(n > 0)
        return frac  # NaN where n==0

    # ---- optional feature joins (static across the LOO loop) ----
    cooc_cols = []
    if cooc is not None:
        cooc_cols = [c for c in cooc.columns if c.startswith("cooccur_")]
        j = j.merge(cooc[["organism","locus_tag"] + cooc_cols],
                      on=["organism","locus_tag"], how="left")
    reg_cols = []
    if reg is not None:
        reg_cols = [c for c in reg.columns if c.startswith("is_") and c != "is_orphan"]
        j = j.merge(reg[["organism","locus_tag"] + reg_cols],
                      on=["organism","locus_tag"], how="left")
    fba_cols = []
    if fba is not None:
        fba_cols = [c for c in fba.columns if c.startswith("fba_")]
        j = j.merge(fba[["organism","locus_tag"] + fba_cols],
                      on=["organism","locus_tag"], how="left")
    syn_static = []
    if syn is not None:
        syn_static = [c for c in ["synteny_count_prev","synteny_count_next",
                                    "max_synteny_count","prev_same_og","next_same_og"]
                      if c in syn.columns]
        j = j.merge(syn[["organism","locus_tag","prev_og_id","next_og_id"] + syn_static],
                      on=["organism","locus_tag"], how="left")
    blo_cols = []
    if blo is not None:
        blo_cols = [c for c in ["function_essentiality_universal","log_solution_diversity"]
                    if c in blo.columns]
        j = j.merge(blo[["organism","locus_tag"] + blo_cols],
                      on=["organism","locus_tag"], how="left")

    # only rows with base features
    j = j[j.n_paralogs_in_genome.notna()].reset_index(drop=True)
    print(f"  evaluable rows: {len(j):,}  ({int(j.essential.sum())} essential)")

    syn_neigh_cols = ["prev_family_frac","next_family_frac","max_neighbor_family_frac"] if syn is not None else []
    feat_cols = (feat_base + ["family_frac_essential"] + cooc_cols + reg_cols +
                 fba_cols + syn_static + syn_neigh_cols + blo_cols)

    # organisms to evaluate: BERIL orgs with enough essentials
    org_clade = dict(zip(splits.organism, splits.clade))
    clade_sizes = splits.groupby("clade").organism.nunique().to_dict()
    eval_orgs = []
    for org, g in j.groupby("organism"):
        n_ess = int(g.essential.sum())
        if n_ess >= args.min_test_ess:
            eval_orgs.append(org)
    print(f"  organisms to evaluate (>= {args.min_test_ess} essentials): {len(eval_orgs)}")

    # ---- LOO-organism loop ----
    per_org = []
    per_row: list[dict] = []
    for oi, O in enumerate(sorted(eval_orgs), 1):
        t0 = time.time()
        frac = family_frac_excl_org(O)
        # map family_frac onto every row (train + test) -- excludes O
        j["family_frac_essential"] = j["og_id"].map(frac)
        if syn is not None:
            j["prev_family_frac"] = j["prev_og_id"].map(frac)
            j["next_family_frac"] = j["next_og_id"].map(frac)
            j["max_neighbor_family_frac"] = j[["prev_family_frac",
                                                 "next_family_frac"]].max(axis=1)
        train = j[j.organism != O]
        test  = j[j.organism == O]
        if test.essential.nunique() < 2:
            print(f"  [{oi}/{len(eval_orgs)}] {O}: degenerate, skip"); continue

        X_tr = train[feat_cols]; y_tr = train["essential"].astype(int).values
        X_te = test[feat_cols];  y_te = test["essential"].astype(int).values
        spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            learning_rate=args.learning_rate, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=spw,
            objective="binary:logistic", tree_method="hist",
            eval_metric="logloss", verbosity=0, random_state=0)
        clf.fit(X_tr, y_tr, verbose=False)
        prob = clf.predict_proba(X_te)[:,1]
        pred = (prob > 0.5).astype(int)
        mcc = float(matthews_corrcoef(y_te, pred))
        tp = int(((pred==1)&(y_te==1)).sum()); fp = int(((pred==1)&(y_te==0)).sum())
        tn = int(((pred==0)&(y_te==0)).sum()); fn = int(((pred==0)&(y_te==1)).sum())
        P = tp/max(tp+fp,1); R = tp/max(tp+fn,1)
        clade = org_clade.get(O, "?")
        csize = clade_sizes.get(clade, 1)
        per_org.append({
            "organism": O, "clade": clade, "clade_n_orgs": csize,
            "n_genes": len(test), "n_essential": int(y_te.sum()),
            "ess_rate": float(y_te.mean()),
            "MCC": mcc, "precision": P, "recall": R,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
        # per-row predictions for downstream calibration
        test_locus = test["locus_tag"].astype(str).values
        for lt, yt, pr in zip(test_locus, y_te, prob):
            per_row.append({
                "organism": O, "clade": clade, "locus_tag": lt,
                "y_true":   int(yt), "prob": float(pr),
            })
        print(f"  [{oi}/{len(eval_orgs)}] {O:<30s} clade={clade:<16s} "
              f"(n_orgs={csize})  MCC={mcc:+.3f}  P={P:.2f} R={R:.2f}  "
              f"ess={int(y_te.sum())}/{len(test)}  {time.time()-t0:.0f}s")

    if not per_org:
        print("ERROR: no organisms evaluated", file=sys.stderr); return 1

    po = pd.DataFrame(per_org)
    po.to_csv(args.out_org, index=False)
    print(f"\nwrote {args.out_org}")
    pr_df = pd.DataFrame(per_row)
    pr_df.to_csv(args.out_pred, index=False)
    print(f"wrote {args.out_pred}  ({len(pr_df):,} per-row predictions)")

    # ---- per-clade aggregation ----
    clade_rows = []
    for clade, g in po.groupby("clade"):
        clade_rows.append({
            "clade": clade,
            "n_orgs_evaluated": len(g),
            "clade_n_orgs_total": int(g.clade_n_orgs.iloc[0]),
            "mean_MCC": float(g.MCC.mean()),
            "std_MCC": float(g.MCC.std(ddof=0)),
            "min_MCC": float(g.MCC.min()),
            "max_MCC": float(g.MCC.max()),
            "mean_precision": float(g.precision.mean()),
            "mean_recall": float(g.recall.mean()),
            "total_genes": int(g.n_genes.sum()),
            "total_essential": int(g.n_essential.sum()),
        })
    cs = pd.DataFrame(clade_rows).sort_values("mean_MCC", ascending=False)
    cs.to_csv(args.out_clade, index=False)
    print(f"wrote {args.out_clade}")

    # ---- the headline: clade size vs MCC ----
    print(f"\n=== PER-CLADE MCC (LOO-organism, sorted) ===")
    print(f"{'clade':<20s} {'n_orgs':>6s} {'mean_MCC':>9s} {'P':>5s} {'R':>5s} {'genes':>7s}")
    for _, r in cs.iterrows():
        print(f"  {r.clade:<18s} {r.n_orgs_evaluated:>6d} {r.mean_MCC:>+9.3f} "
              f"{r.mean_precision:>5.2f} {r.mean_recall:>5.2f} {r.total_genes:>7d}")

    # well-sampled vs singleton split
    multi = po[po.clade_n_orgs >= 3]
    singl = po[po.clade_n_orgs == 1]
    pair  = po[po.clade_n_orgs == 2]
    print(f"\n=== CLADE-SIZE EFFECT (the deployment story) ===")
    print(f"  well-sampled clades (>=3 orgs): {len(multi):>3d} orgs  "
          f"mean MCC = {multi.MCC.mean():+.4f}" if len(multi) else "  (none)")
    print(f"  paired clades (2 orgs):         {len(pair):>3d} orgs  "
          f"mean MCC = {pair.MCC.mean():+.4f}" if len(pair) else "  (none)")
    print(f"  singleton clades (1 org):       {len(singl):>3d} orgs  "
          f"mean MCC = {singl.MCC.mean():+.4f}" if len(singl) else "  (none)")
    # correlation
    import numpy as np
    if len(po) >= 5:
        x = np.log2(po.clade_n_orgs.to_numpy().astype(float))
        y = po.MCC.to_numpy()
        if x.std() > 0:
            corr = float(np.corrcoef(x, y)[0,1])
            print(f"  corr(log2 clade_n_orgs, MCC) = {corr:+.3f}")

    overall = {
        "n_orgs_evaluated":     len(po),
        "overall_mean_MCC":     float(po.MCC.mean()),
        "overall_median_MCC":   float(po.MCC.median()),
        "macro_clade_mean_MCC": float(cs.mean_MCC.mean()),
        "by_clade_size": {
            "ge3_orgs":   {"n": int(len(multi)), "mean_MCC": float(multi.MCC.mean()) if len(multi) else None},
            "eq2_orgs":   {"n": int(len(pair)),  "mean_MCC": float(pair.MCC.mean())  if len(pair)  else None},
            "eq1_org":    {"n": int(len(singl)), "mean_MCC": float(singl.MCC.mean()) if len(singl) else None},
        },
        "per_clade":    clade_rows,
        "per_organism": per_org,
        "config": {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "features": feat_cols},
    }
    with open(args.out_json, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\nwrote {args.out_json}")
    print(f"\n=== HEADLINE ===")
    print(f"  LOO-organism overall mean MCC:   {po.MCC.mean():+.4f}")
    print(f"  (vs cross-clade LOO baseline:    +0.4864)")
    print(f"  macro per-clade mean MCC:        {cs.mean_MCC.mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
