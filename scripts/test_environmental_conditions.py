"""Test the ENVIRONMENT hypothesis using a STANDARD CONDITION (rich medium),
encoded as known biochemistry -- no Fitness Browser matrix required.

LOGIC: essentiality is measured on (approximately) rich medium. A rich
medium PROVIDES amino acids, nucleobases/nucleosides, and B-vitamins.
So a gene whose job is to SYNTHESISE one of those is rescuable by the
medium -> should be LESS essential. A gene doing core machinery
(translation/replication/energy/envelope) makes products the medium
does NOT provide -> essential regardless.

If ENVIRONMENT drives the conditional zone (the ~47% ambiguous flips),
then this single biochemistry-derived feature -- "does standard rich
medium provide this gene's product?" -- should:
  (1) strongly predict essentiality (univariate),
  (2) be concentrated in the conditional zone (that's where flips live),
  (3) add usable signal to the fused model there.

LEAK CONTROL: env_substitutability is derived ONLY from the gene's
product annotation (functional category) -- available at deployment,
no essentiality labels used. family_frac stays per-fold leak-free.
"""
from __future__ import annotations
import re, sys, time
from collections import defaultdict, Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
D = REPO / "data/drive_import"

# ---------- STANDARD CONDITION = rich medium ----------
# What a standard rich medium (e.g. LB / rich defined) PROVIDES, so the
# corresponding biosynthesis is rescuable (-> less essential):
RESCUABLE = {
  "amino_acid": ["amino acid biosynthe","aminotransferase",
     "histidin","tryptophan synth","anthranilate","chorismate","shikimat",
     "prephenate","threonine synth","homoserine","cystathion","methionine synth",
     "ornithine","argininosuccinate","diaminopimelate","aspartate kinase",
     "branched-chain amino","leucine","valine","isoleucine","glutamate synth",
     "asparagine synth","serine biosynth","glycine hydroxymethyl","proline",
     "acetylglutamate","dihydrodipicolinate","ketol-acid reductoisomerase"],
  "nucleotide": ["purine biosynth","pyrimidine biosynth","nucleotide biosynth",
     "phosphoribosyl","orotidine","orotate","carbamoyl-phosphate",
     "adenylosuccinate","dihydroorotate","amidophosphoribosyl","inosine",
     "thymidylate synth","GMP synth","IMP "],
  "vitamin_cofactor": ["biotin synth","biotin biosynth","thiamine","thiamin ",
     "riboflavin","folate","dihydrofolate","dihydropteroate","tetrahydrofolate",
     "pantothenate","coenzyme a biosynth","pyridoxal","pyridoxine","cobalamin",
     "vitamin b12","nicotinate","nicotinamide","quinolinate","pyridoxamine",
     "lipoate","lipoyl","molybdopterin","menaquinone","ubiquinone biosynth"],
}
# Core functions the medium does NOT provide (essential regardless of medium):
CORE_NOT_PROVIDED = ["ribosomal protein","trna ligase","trna synthetase",
   "rna polymerase","dna polymerase","dna primase","dna gyrase","helicase",
   "topoisomerase","elongation factor","initiation factor","release factor",
   "atp synthase","f0f1","cell division","ftsz","peptidoglycan","mur ",
   "lipid a","fatty acid synth","acyl carrier","preprotein translocase",
   "signal recognition","ribosome","16s","23s","5s ","rrna"]


def classify_env(product):
    p = (product or "").lower()
    for cat, kws in RESCUABLE.items():
        if any(k in p for k in kws):
            return 2, cat            # 2 = rescuable by rich medium
    if any(k in p for k in CORE_NOT_PROVIDED):
        return 0, "core"            # 0 = medium can't help -> essential regardless
    return 1, "other"               # 1 = neither clearly


# ---------- reuse parsing ----------
def parse_gff(path):
    gene_attrs={}; cds=[]
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        p=line.rstrip("\n").split("\t")
        if len(p)<9: continue
        d={}
        for kv in p[8].split(";"):
            if "=" in kv: k,v=kv.split("=",1); d[k]=v
        if p[2]=="gene" and "ID" in d:
            gene_attrs[d["ID"]]={"locus_tag":d.get("locus_tag","")}
        elif p[2]=="CDS": cds.append((p,d))
    feats=[]
    for p,d in cds:
        lt=d.get("locus_tag","")
        if not lt and d.get("Parent","") in gene_attrs:
            lt=gene_attrs[d["Parent"]]["locus_tag"]
        feats.append({"locus_tag":lt,"old_locus_tag":d.get("old_locus_tag",""),
                       "product":d.get("product","")})
    return feats


def main():
    import pandas as pd, numpy as np, xgboost as xgb
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    print("="*70)
    print("ENVIRONMENT TEST via STANDARD CONDITION (rich medium biochemistry)")
    print("="*70)

    labels=pd.read_csv(D/"labels/labels.csv")
    orth=pd.read_csv(D/"labels/orthology_features.csv")
    mf=pd.read_csv(D/"labels/genome_cache_manifest.csv")
    elig=mf[(mf.kind=="ours")&mf.join_rate.notna()&(mf.join_rate>=0.3)]
    accs=dict(zip(elig.organism,elig.accession))

    # leak-free family_frac machinery
    j=labels.merge(orth[["organism","locus_tag","og_id"]],on=["organism","locus_tag"],how="inner")
    j=j[j.og_id.notna()].copy(); j.essential=j.essential.astype(int)
    og_tot_e=j.groupby("og_id").essential.sum(); og_tot_n=j.groupby("og_id").essential.size()
    og_org=j.groupby(["og_id","organism"]).essential.agg(["sum","size"])
    og_map=dict(zip(zip(j.organism,j.locus_tag),j.og_id))
    lab_map=dict(zip(zip(labels.organism,labels.locus_tag),labels.essential.astype(int)))
    def ff_excl(org):
        try:
            s=og_org.xs(org,level="organism"); e,n=s["sum"],s["size"]
        except KeyError:
            e=pd.Series(0,index=og_tot_e.index); n=pd.Series(0,index=og_tot_n.index)
        E=og_tot_e.sub(e,fill_value=0); N=og_tot_n.sub(n,fill_value=0)
        return E/N.where(N>0)

    # build per-gene table with env feature
    print("\n[build] classifying genes by rich-medium substitutability...")
    rows=[]
    for org in sorted(accs):
        gff=D/"genome_cache"/accs[org]/"genomic.gff"
        if not gff.exists(): continue
        for f in parse_gff(gff):
            lt=f["locus_tag"]
            e=lab_map.get((org,lt))
            if e is None and f["old_locus_tag"]: e=lab_map.get((org,f["old_locus_tag"]))
            if e is None: continue
            ogid=og_map.get((org,lt)) or og_map.get((org,f["old_locus_tag"]))
            env,cat=classify_env(f["product"])
            rows.append({"organism":org,"locus_tag":lt,"og_id":ogid,"essential":e,
                          "env_substitutable":env,"env_cat":cat})
    df=pd.DataFrame(rows)
    print(f"[data] {len(df):,} genes, {int(df.essential.sum()):,} essential\n")

    # ---------- (1) UNIVARIATE: essentiality by env category ----------
    print("="*70); print("(1) Essentiality rate by env-substitutability (univariate)")
    print("="*70)
    print(f"  {'category':<18s} {'n':>7s} {'ess_rate':>9s}")
    for cat in ["core","other","amino_acid","nucleotide","vitamin_cofactor"]:
        sub=df[df.env_cat==cat]
        if len(sub): print(f"  {cat:<18s} {len(sub):>7,d} {sub.essential.mean()*100:>8.1f}%")
    print(f"\n  env_substitutable code:  0=core(not provided)  1=other  2=rescuable")
    for v in (0,1,2):
        sub=df[df.env_substitutable==v]
        print(f"    code {v}: {len(sub):>6,d} genes, {sub.essential.mean()*100:.1f}% essential")
    # the smoking gun: rescuable should be MUCH less essential than core
    core_rate=df[df.env_substitutable==0].essential.mean()
    resc_rate=df[df.env_substitutable==2].essential.mean()
    print(f"\n  >>> core essential rate {core_rate*100:.0f}%  vs  "
          f"rescuable {resc_rate*100:.0f}%  (ratio {core_rate/max(resc_rate,1e-9):.1f}x) <<<")

    # ---------- (2) is the env signal concentrated in the CONDITIONAL zone? ----------
    print("\n"+"="*70)
    print("(2) Where does the env signal live? (need leak-free family_frac)")
    print("="*70)
    # use full-data family_frac for this descriptive split (organism-pooled);
    # the predictive test below uses per-fold leak-free.
    ff_all=(og_tot_e/og_tot_n)
    df["ff_pooled"]=df.og_id.map(ff_all)
    cond=df[(df.ff_pooled>0.2)&(df.ff_pooled<0.8)]
    print(f"  conditional zone (family_frac 0.2-0.8): {len(cond):,} genes")
    print(f"  {'category':<18s} {'n':>6s} {'ess_rate':>9s}")
    for cat in ["core","other","amino_acid","nucleotide","vitamin_cofactor"]:
        sub=cond[cond.env_cat==cat]
        if len(sub)>20: print(f"  {cat:<18s} {len(sub):>6,d} {sub.essential.mean()*100:>8.1f}%")

    # ---------- (3) PREDICTIVE: does env feature add over family_frac? ----------
    print("\n"+"="*70)
    print("(3) Predictive lift, leak-free LOO-organism")
    print("="*70)
    XGB=dict(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.85,
             colsample_bytree=0.8,min_child_weight=2,tree_method="hist",
             eval_metric="logloss",verbosity=0,n_jobs=-1,random_state=0)
    y_all=[]; p_base=[]; p_env=[]; ff_keep=[]
    for O in sorted(df.organism.unique()):
        ff=ff_excl(O); df["family_frac"]=df.og_id.map(ff)
        tr=df[df.organism!=O]; te=df[df.organism==O]
        if te.essential.nunique()<2: continue
        ytr=tr.essential.values; yte=te.essential.values
        spw=(len(ytr)-ytr.sum())/max(ytr.sum(),1)
        # baseline: family_frac only
        mb=xgb.XGBClassifier(scale_pos_weight=spw,**XGB)
        mb.fit(tr[["family_frac"]].fillna(0),ytr)
        pb=mb.predict_proba(te[["family_frac"]].fillna(0))[:,1]
        # + env feature
        me=xgb.XGBClassifier(scale_pos_weight=spw,**XGB)
        me.fit(tr[["family_frac","env_substitutable"]].fillna(0),ytr)
        pe=me.predict_proba(te[["family_frac","env_substitutable"]].fillna(0))[:,1]
        y_all.extend(yte); p_base.extend(pb); p_env.extend(pe)
        ff_keep.extend(te.family_frac.fillna(0).values)
    y_all=np.array(y_all); p_base=np.array(p_base); p_env=np.array(p_env); ff_keep=np.array(ff_keep)

    def auc(yv,pv): return roc_auc_score(yv,pv) if yv.sum() and yv.sum()<len(yv) else float("nan")
    def mcc(yv,pv,t=0.5):
        pr=(pv>t).astype(int); return matthews_corrcoef(yv,pr) if yv.sum() and yv.sum()<len(yv) else float("nan")
    print(f"\n  OVERALL ({len(y_all):,} genes):")
    print(f"    family_frac only      AUC={auc(y_all,p_base):.4f}  MCC={mcc(y_all,p_base):+.4f}")
    print(f"    + env(standard rich)  AUC={auc(y_all,p_env):.4f}  MCC={mcc(y_all,p_env):+.4f}")
    cmask=(ff_keep>0.2)&(ff_keep<0.8)
    print(f"\n  CONDITIONAL ZONE ({int(cmask.sum()):,} genes, family_frac 0.2-0.8):")
    print(f"    family_frac only      AUC={auc(y_all[cmask],p_base[cmask]):.4f}  MCC={mcc(y_all[cmask],p_base[cmask]):+.4f}")
    print(f"    + env(standard rich)  AUC={auc(y_all[cmask],p_env[cmask]):.4f}  MCC={mcc(y_all[cmask],p_env[cmask]):+.4f}")
    d_overall=auc(y_all,p_env)-auc(y_all,p_base)
    d_cond=auc(y_all[cmask],p_env[cmask])-auc(y_all[cmask],p_base[cmask])
    print(f"\n  >>> AUC lift from standard-condition feature: "
          f"overall {d_overall:+.4f}, CONDITIONAL ZONE {d_cond:+.4f} <<<")
    print(f"  (if conditional lift >> overall lift, environment IS the driver)")

    out=REPO/"outputs/environment_standard_condition.md"
    with open(out,"w") as f:
        f.write("# Environment test via standard condition (rich medium)\n\n")
        f.write(f"core essential {core_rate*100:.0f}% vs rescuable {resc_rate*100:.0f}% "
                f"({core_rate/max(resc_rate,1e-9):.1f}x)\n\n")
        f.write(f"AUC lift from env feature: overall {d_overall:+.4f}, conditional zone {d_cond:+.4f}\n")
    print(f"\nwrote {out}")
    return 0

if __name__=="__main__":
    sys.exit(main())
