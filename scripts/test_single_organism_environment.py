"""Single-organism test: conservation-only vs conservation + environment
(isozyme-backup) feature, on the MOST DISTANT organism in the dataset.

'Most distant' = the genome-having organism whose gene families are least
covered by the rest of the dataset (lowest mean family_n_organisms),
i.e. the hardest case for the conservation prior. Archaea (Methanococcus)
are expected to win this.

Two methods, identical train/test split (leave-this-organism-out):
  A. EARLIER METHOD  : base orthology features + leave-O-out family_frac
  B. + ENVIRONMENT   : A plus func_copies / func_unique (does the genome
                        encode an isozyme backup for this gene's function?)

family_frac is recomputed excluding the test organism (clade-inclusive,
leakage-free). The environment feature comes from the test organism's
own genome (GFF product redundancy), so it IS available at deployment
time for a novel organism (you have its genome, you just haven't done
the knockout screen).

Reports MCC / precision / recall / accuracy / confusion for both,
plus how many of the test organism's rogue-essentials (low family_frac)
the environment feature recovers.
"""
from __future__ import annotations
import argparse, math, re, sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_flanking_regions import parse_gff_cds

GENERIC = {"hypothetical protein","uncharacterized protein","membrane protein",
           "conserved hypothetical protein","putative membrane protein",
           "conserved protein","domain-containing protein","putative protein"}


def norm_product(p):
    p = p.lower().strip()
    p = re.sub(r"\b(putative|probable|possible|predicted)\b", "", p).strip()
    return re.sub(r"\s+", " ", p)


def isozyme_features(gff_path, lt2e):
    """Returns {locus_tag: (func_copies, func_unique)} for labeled genes."""
    feats = parse_gff_cds(gff_path)
    prods = [norm_product(f["product"]) for f in feats if f["product"]]
    cnt = Counter(prods)
    out = {}
    for f in feats:
        lt = f["locus_tag"]
        key = lt if lt in lt2e else (f["old_locus_tag"] if f["old_locus_tag"] in lt2e else None)
        if key is None: continue
        npd = norm_product(f["product"]) if f["product"] else ""
        generic = (npd in GENERIC) or (npd == "") or ("hypothetical" in npd) or ("uncharacterized" in npd)
        copies = cnt.get(npd, 1)
        out[key] = (copies if not generic else float("nan"),
                    int(copies == 1) if not generic else float("nan"))
    return out


def confusion(y, pred):
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    tn = int(((pred==0)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    P = tp/max(tp+fp,1); R = tp/max(tp+fn,1)
    acc = (tp+tn)/max(tp+fp+tn+fn,1)
    mcc_den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn-fp*fn)/mcc_den if mcc_den else 0.0
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,precision=P,recall=R,accuracy=acc,mcc=mcc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--organism", default=None, help="force a specific test org")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT/"data/drive_import/genome_cache")
    p.add_argument("--n-estimators", type=int, default=400)
    args = p.parse_args()

    import pandas as pd, numpy as np, xgboost as xgb

    D = REPO_ROOT/"data/drive_import/labels"
    labels = pd.read_csv(D/"labels.csv")
    orth = pd.read_csv(D/"orthology_features.csv")
    mf = pd.read_csv(D/"genome_cache_manifest.csv")
    elig = mf[(mf.kind=="ours") & mf.accession.notna() &
              mf.join_rate.notna() & (mf.join_rate>=0.3)]
    genome_orgs = dict(zip(elig.organism, elig.accession))

    base_feats = ["n_paralogs_in_genome","family_size_total","family_n_organisms","is_orphan"]
    j = labels.merge(orth[["organism","locus_tag","og_id"]+base_feats],
                     on=["organism","locus_tag"], how="inner")
    j = j[j.og_id.notna()].copy()

    # ---- pick most-distant org: lowest mean family_n_organisms among genome orgs ----
    dist = (j[j.organism.isin(genome_orgs)]
            .groupby("organism").family_n_organisms.mean().sort_values())
    print("=== distance ranking (lower mean family breadth = more distant) ===")
    for o, v in dist.items():
        print(f"  {o:<28s} mean family_n_organisms = {v:.1f}")
    test_org = args.organism or dist.index[0]
    print(f"\n>>> TEST ORGANISM: {test_org}  (accession {genome_orgs[test_org]}) <<<\n")

    # ---- leave-O-out family_frac (clade-inclusive) ----
    es = j[["organism","og_id","essential"]].copy()
    es.essential = es.essential.astype(int)
    tot_e = es.groupby("og_id").essential.sum()
    tot_n = es.groupby("og_id").essential.size()
    oo = es[es.organism==test_org].groupby("og_id").essential.agg(["sum","size"])
    E = tot_e.sub(oo["sum"], fill_value=0); N = tot_n.sub(oo["size"], fill_value=0)
    ffrac = (E / N.where(N>0))
    j["family_frac"] = j.og_id.map(ffrac)

    # ---- isozyme (environment) features for genome orgs ----
    iso_map = {}  # (org, lt) -> (copies, unique)
    for org, acc in genome_orgs.items():
        ol = labels[labels.organism==org]
        lt2e = dict(zip(ol.locus_tag.astype(str), ol.essential.astype(int)))
        feats = isozyme_features(args.cache_dir/acc/"genomic.gff", lt2e)
        for lt,(c,u) in feats.items():
            iso_map[(org,lt)] = (c,u)
    j["func_copies"] = j.apply(lambda r: iso_map.get((r.organism,str(r.locus_tag)),(np.nan,np.nan))[0], axis=1)
    j["func_unique"] = j.apply(lambda r: iso_map.get((r.organism,str(r.locus_tag)),(np.nan,np.nan))[1], axis=1)

    train = j[j.organism != test_org]
    test  = j[j.organism == test_org]
    print(f"train: {len(train):,} genes ({train.organism.nunique()} orgs)   "
          f"test: {len(test):,} genes  ({int(test.essential.sum())} essential, "
          f"{test.essential.mean()*100:.1f}%)")

    def run(cols, name):
        ytr = train.essential.astype(int).values
        yte = test.essential.astype(int).values
        spw = (len(ytr)-ytr.sum())/max(ytr.sum(),1)
        clf = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, scale_pos_weight=spw, tree_method="hist",
            eval_metric="logloss", verbosity=0, random_state=0)
        clf.fit(train[cols], ytr)
        prob = clf.predict_proba(test[cols])[:,1]
        pred = (prob>0.5).astype(int)
        cm = confusion(yte, pred)
        print(f"\n=== {name} ===  features: {cols}")
        print(f"  MCC={cm['mcc']:+.4f}  precision={cm['precision']:.3f}  "
              f"recall={cm['recall']:.3f}  accuracy={cm['accuracy']:.3f}")
        print(f"  TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
        return prob, pred, cm

    A_cols = base_feats + ["family_frac"]
    B_cols = A_cols + ["func_copies","func_unique"]
    probA, predA, cmA = run(A_cols, "METHOD A: conservation only")
    probB, predB, cmB = run(B_cols, "METHOD B: conservation + environment(isozyme)")

    print(f"\n=== HEAD-TO-HEAD on {test_org} ===")
    print(f"  {'metric':<12s} {'A (cons)':>10s} {'B (+env)':>10s} {'delta':>9s}")
    for k in ["mcc","precision","recall","accuracy"]:
        print(f"  {k:<12s} {cmA[k]:>10.4f} {cmB[k]:>10.4f} {cmB[k]-cmA[k]:>+9.4f}")

    # ---- rogue-essential recovery: low family_frac but truly essential ----
    yte = test.essential.astype(int).values
    rogue = (test.family_frac.fillna(0).values < 0.3) & (yte==1)
    n_rogue = int(rogue.sum())
    recA = int((predA[rogue]==1).sum()); recB = int((predB[rogue]==1).sum())
    print(f"\n=== rogue-essential recovery (truly essential, family_frac<0.3) ===")
    print(f"  rogue essentials in {test_org}: {n_rogue}")
    print(f"  recovered by A (conservation):       {recA}  ({100*recA/max(n_rogue,1):.0f}%)")
    print(f"  recovered by B (+environment):       {recB}  ({100*recB/max(n_rogue,1):.0f}%)")
    print(f"  environment feature rescued: {recB-recA} additional rogue essentials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
