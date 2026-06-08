"""ESM-2 pilot evaluation: do embeddings add MCC over the existing
feature stack, on the organisms where we have them?

LOO-organism among the embedded organisms only. Two arms, identical
folds/rows:
  A: base feature stack (orth + family_frac_loo + cooccur + reg +
     fba + synteny + blo)
  B: base + ESM-2 embedding columns

family_frac is recomputed per held-out organism (LOO-organism,
clade-inclusive, leakage-free) exactly as build_per_clade_evaluation.py
does. The only difference between arms is the embedding block, so the
mean MCC delta is the embedding's marginal value.

Verdict:
  delta > 0.03  : embeddings help -> worth bridging the rest + scaling
  delta > 0.005 : marginal
  delta <= 0.005: embeddings absorbed too -> ESM-2 dead, stop
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "memory_bank" / "data" / "multiorg_essentiality"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--emb",      type=Path, required=True,
                   help="esm2_embeddings.parquet")
    p.add_argument("--n-estimators", type=int, default=300)
    args = p.parse_args()

    try:
        import pandas as pd, numpy as np, xgboost as xgb
        from sklearn.metrics import matthews_corrcoef
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr); return 1
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_clade_splits import is_bacterial_clade

    D = args.data_dir
    labels = pd.read_csv(D / "labels.csv")
    splits = pd.read_csv(D / "clade_splits.csv")
    orth   = pd.read_csv(D / "orthology_features.csv")
    emb    = pd.read_parquet(args.emb)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    embedded_orgs = sorted(emb.organism.unique())
    print(f"embedded organisms: {len(embedded_orgs)} ({emb_cols.__len__()}-dim)")
    print(f"  {embedded_orgs}")

    j = labels.merge(splits[["organism","clade","fold"]], on="organism", how="left")
    j = j[j.clade.notna() & j.clade.apply(is_bacterial_clade)].copy()
    feat_base = ["n_paralogs_in_genome","family_size_total",
                 "family_n_organisms","is_orphan"]
    j = j.merge(orth[["organism","locus_tag","og_id"]+feat_base],
                on=["organism","locus_tag"], how="left")

    # OG aggregates for per-organism LOO family_frac
    ess_src = j[["organism","og_id","essential"]].dropna(subset=["og_id"]).copy()
    ess_src["essential"] = ess_src["essential"].astype(int)
    og_tot_e = ess_src.groupby("og_id")["essential"].sum()
    og_tot_n = ess_src.groupby("og_id")["essential"].size()
    og_org = ess_src.groupby(["og_id","organism"])["essential"].agg(["sum","size"])

    def frac_excl(org):
        try:
            s = og_org.xs(org, level="organism")
            e, n = s["sum"], s["size"]
        except KeyError:
            e = pd.Series(0, index=og_tot_e.index); n = pd.Series(0, index=og_tot_n.index)
        E = og_tot_e.sub(e, fill_value=0); N = og_tot_n.sub(n, fill_value=0)
        return E / N.where(N > 0)

    # optional features
    def opt(name, pref):
        pth = D / name
        if not pth.exists(): return None, []
        d = pd.read_csv(pth)
        cols = [c for c in d.columns if c.startswith(pref)]
        return d, cols
    cooc, cooc_cols = opt("cooccurrence_features.csv","cooccur_")
    reg,  reg_all   = opt("regulator_features.csv","is_")
    reg_cols = [c for c in reg_all if c != "is_orphan"] if reg is not None else []
    fba,  fba_cols  = opt("fba_features.csv","fba_")
    blo,  blo_all   = opt("blo_features.csv","function_") if (D/"blo_features.csv").exists() else (None,[])
    blo_cols = []
    if blo is not None:
        blo_cols = [c for c in ["function_essentiality_universal","log_solution_diversity"] if c in blo.columns]
    for d, cols in [(cooc,cooc_cols),(reg,reg_cols),(fba,fba_cols),(blo,blo_cols)]:
        if d is not None and cols:
            j = j.merge(d[["organism","locus_tag"]+cols], on=["organism","locus_tag"], how="left")
    # synteny static
    syn = pd.read_csv(D/"synteny_features.csv") if (D/"synteny_features.csv").exists() else None
    syn_static = []
    if syn is not None:
        syn_static = [c for c in ["synteny_count_prev","synteny_count_next",
                                  "max_synteny_count","prev_same_og","next_same_og"] if c in syn.columns]
        j = j.merge(syn[["organism","locus_tag","prev_og_id","next_og_id"]+syn_static],
                    on=["organism","locus_tag"], how="left")
    syn_neigh = ["prev_family_frac","next_family_frac","max_neighbor_family_frac"] if syn is not None else []

    # join embeddings
    j = j.merge(emb[["organism","locus_tag"]+emb_cols], on=["organism","locus_tag"], how="left")
    j = j[j.n_paralogs_in_genome.notna()].reset_index(drop=True)

    base_cols = (feat_base + ["family_frac_essential"] + cooc_cols + reg_cols
                 + fba_cols + syn_static + syn_neigh + blo_cols)

    def run(use_emb: bool):
        per = []
        for O in embedded_orgs:
            n_ess = int(j[j.organism==O].essential.sum())
            if n_ess < 10: continue
            fr = frac_excl(O)
            j["family_frac_essential"] = j["og_id"].map(fr)
            if syn is not None:
                j["prev_family_frac"] = j["prev_og_id"].map(fr)
                j["next_family_frac"] = j["next_og_id"].map(fr)
                j["max_neighbor_family_frac"] = j[["prev_family_frac","next_family_frac"]].max(axis=1)
            cols = base_cols + (emb_cols if use_emb else [])
            tr = j[j.organism != O]; te = j[j.organism == O]
            if te.essential.nunique() < 2: continue
            ytr = tr.essential.astype(int).values; yte = te.essential.astype(int).values
            spw = (len(ytr)-ytr.sum())/max(ytr.sum(),1)
            clf = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=4,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=2, scale_pos_weight=spw, tree_method="hist",
                eval_metric="logloss", verbosity=0, random_state=0)
            clf.fit(tr[cols], ytr)
            pred = (clf.predict_proba(te[cols])[:,1] > 0.5).astype(int)
            mcc = matthews_corrcoef(yte, pred)
            per.append((O, mcc))
        return per

    print("\n=== Arm A: base (no embeddings) ===")
    a = run(False)
    for O,m in a: print(f"  {O:<30s} MCC={m:+.3f}")
    print("\n=== Arm B: base + ESM-2 ===")
    b = run(True)
    for O,m in b: print(f"  {O:<30s} MCC={m:+.3f}")

    import numpy as np
    am = {o:m for o,m in a}; bm = {o:m for o,m in b}
    common = [o for o in am if o in bm]
    ma = np.mean([am[o] for o in common]); mb = np.mean([bm[o] for o in common])
    print(f"\n=== HEAD-TO-HEAD ({len(common)} organisms) ===")
    print(f"  base mean MCC:        {ma:+.4f}")
    print(f"  base + ESM-2 mean MCC:{mb:+.4f}")
    print(f"  delta:                {mb-ma:+.4f}")
    print(f"\n  per-organism delta:")
    for o in common:
        print(f"    {o:<30s} {bm[o]-am[o]:+.4f}")
    d = mb - ma
    print(f"\nVerdict: ", end="")
    if d > 0.03:   print("ESM-2 HELPS -> worth bridging remaining orgs + scaling")
    elif d > 0.005:print("marginal embedding effect")
    else:          print("ESM-2 ABSORBED -> embeddings don't add over the prior; stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
