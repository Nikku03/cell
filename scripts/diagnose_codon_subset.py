"""Subset diagnostic: is CAI starved by coverage or absorbed by family_frac?

If CAI moves MCC meaningfully ON THE 13K ROWS WHERE IT EXISTS,
the global ~0 movement is due to NaN-coverage (CAI missing in
92% of training rows) and we should fix the failed-org accessions
to scale coverage. If MCC doesn't move on the subset either,
family_frac is absorbing the codon signal and scaling won't help.

Runs LOO over the clades represented in the subset, two models:
  Model A : all features INCLUDING codon
  Model B : all features EXCEPT codon (same subset, same folds)

Compares mean MCC delta and per-fold delta.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"


def main() -> int:
    try:
        import pandas as pd
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import matthews_corrcoef
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_clade_splits import is_bacterial_clade

    print("Loading ...")
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    splits = pd.read_csv(DATA_DIR / "clade_splits.csv")
    orth   = pd.read_csv(DATA_DIR / "orthology_features.csv")
    cooc   = pd.read_csv(DATA_DIR / "cooccurrence_features.csv") if (DATA_DIR / "cooccurrence_features.csv").exists() else None
    reg    = pd.read_csv(DATA_DIR / "regulator_features.csv") if (DATA_DIR / "regulator_features.csv").exists() else None
    fba    = pd.read_csv(DATA_DIR / "fba_features.csv") if (DATA_DIR / "fba_features.csv").exists() else None
    syn    = pd.read_csv(DATA_DIR / "synteny_features.csv") if (DATA_DIR / "synteny_features.csv").exists() else None
    blo    = pd.read_csv(DATA_DIR / "blo_features.csv") if (DATA_DIR / "blo_features.csv").exists() else None
    cod    = pd.read_csv(DATA_DIR / "codon_features.csv")

    # Build training frame
    j = labels.merge(splits[["organism","clade","fold"]], on="organism", how="left")
    j = j[j.clade.notna() & j.clade.apply(is_bacterial_clade)]

    feat_cols_base = ["n_paralogs_in_genome","family_size_total",
                       "family_n_organisms","is_orphan"]
    fold_cols = [c for c in orth.columns if c.startswith("family_frac_essential_fold")]
    n_folds = len(fold_cols)
    orth_full = orth[["og_id"] + fold_cols].drop_duplicates("og_id").set_index("og_id")
    j = j.merge(orth[["organism","locus_tag"] + feat_cols_base + fold_cols],
                  on=["organism","locus_tag"], how="left")

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
    syn_cols_static = []
    if syn is not None:
        ss = [c for c in ["synteny_count_prev","synteny_count_next",
                            "max_synteny_count","prev_same_og","next_same_og"]
              if c in syn.columns]
        j = j.merge(syn[["organism","locus_tag","prev_og_id","next_og_id"] + ss],
                      on=["organism","locus_tag"], how="left")
        syn_cols_static = ss
    syn_cols = ["prev_family_frac","next_family_frac","max_neighbor_family_frac"] + syn_cols_static
    blo_cols = []
    if blo is not None:
        bs = [c for c in ["function_essentiality_universal","log_solution_diversity"]
              if c in blo.columns]
        j = j.merge(blo[["organism","locus_tag"] + bs],
                      on=["organism","locus_tag"], how="left")
        blo_cols = bs

    codon_cols = ["cai","gc","gc3","cds_length","intergenic_prev","intergenic_next",
                  "same_strand_prev","same_strand_next"]
    codon_cols = [c for c in codon_cols if c in cod.columns]
    j = j.merge(cod[["organism","locus_tag"] + codon_cols],
                  on=["organism","locus_tag"], how="left")

    # Drop rows without basic features
    j = j[j.n_paralogs_in_genome.notna()].reset_index(drop=True)
    print(f"  full set: {len(j):,} rows  "
          f"({int(j.essential.sum())} essentials)")

    # Subset to CAI-non-null
    sub = j[j.cai.notna()].reset_index(drop=True)
    print(f"  CAI subset: {len(sub):,} rows  "
          f"({int(sub.essential.sum())} essentials)  "
          f"orgs: {sub.organism.unique().tolist()}")
    print(f"  clades in subset: {sorted(sub.clade.unique().tolist())}")
    print(f"  folds in subset:  {sorted(sub.fold.unique().tolist())}")

    def train_eval(feat_cols_extra: list[str], label: str):
        per_fold = []
        for k in sorted(sub.fold.unique()):
            train = sub[sub.fold != k].copy()
            test  = sub[sub.fold == k].copy()
            if len(test) == 0 or len(train) == 0: continue
            if test.essential.nunique() < 2:
                print(f"    [{label}] fold {k}: degenerate test, skip"); continue

            ff_col = f"family_frac_essential_fold{k}"
            # fill synteny neighbor family_frac for this fold
            og2f = orth_full[ff_col]
            for df in (train, test):
                df["prev_family_frac"] = df["prev_og_id"].map(og2f)
                df["next_family_frac"] = df["next_og_id"].map(og2f)
                df["max_neighbor_family_frac"] = df[
                    ["prev_family_frac","next_family_frac"]].max(axis=1)

            feat_cols = feat_cols_base + [ff_col] + cooc_cols + reg_cols + \
                        fba_cols + syn_cols + blo_cols + feat_cols_extra
            X_tr = train[feat_cols].rename(columns={ff_col:"family_frac_essential"})
            y_tr = train["essential"].astype(int).values
            X_te = test[feat_cols].rename(columns={ff_col:"family_frac_essential"})
            y_te = test["essential"].astype(int).values
            spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
            clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                scale_pos_weight=spw, objective="binary:logistic",
                tree_method="hist", eval_metric="logloss", verbosity=0,
                random_state=0)
            clf.fit(X_tr, y_tr, verbose=False)
            prob = clf.predict_proba(X_te)[:,1]
            pred = (prob > 0.5).astype(int)
            mcc  = matthews_corrcoef(y_te, pred)
            tp = int(((pred==1) & (y_te==1)).sum())
            fp = int(((pred==1) & (y_te==0)).sum())
            tn = int(((pred==0) & (y_te==0)).sum())
            fn = int(((pred==0) & (y_te==1)).sum())
            P = tp / max(tp+fp,1); R = tp / max(tp+fn,1)
            print(f"    [{label}] fold {k}: MCC={mcc:+.4f}  P={P:.3f}  R={R:.3f}  "
                  f"n_test={len(test)}  clades={sorted(test.clade.unique())}")
            per_fold.append(mcc)
        return per_fold

    print(f"\n=== Model A: ALL features INCLUDING codon (on CAI subset) ===")
    mcc_with = train_eval(codon_cols, "WITH-CODON")
    print(f"\n=== Model B: ALL features EXCEPT codon (same rows, same folds) ===")
    mcc_without = train_eval([], "NO-CODON")

    print(f"\n=== HEAD-TO-HEAD on CAI subset ({len(sub):,} rows) ===")
    print(f"  Model A (with codon):  mean MCC = {np.mean(mcc_with):+.4f}  "
          f"({len(mcc_with)} folds)")
    print(f"  Model B (no codon):    mean MCC = {np.mean(mcc_without):+.4f}  "
          f"({len(mcc_without)} folds)")
    delta = np.mean(mcc_with) - np.mean(mcc_without)
    print(f"  delta = {delta:+.4f}")
    print(f"\nVerdict:")
    if delta > 0.03:
        print(f"  *** CAI IS HELPING ON ITS COVERED ROWS ***")
        print(f"  Scale-up worthwhile: fix failed-org accessions to extend coverage.")
    elif delta > 0.005:
        print(f"  Small but positive effect. Marginal scaling case.")
    else:
        print(f"  *** CAI IS ABSORBED BY family_frac, even on its native rows ***")
        print(f"  Scaling coverage won't help. Move on to ESM-2 or accept ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
