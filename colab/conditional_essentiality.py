"""Genome-wide CONDITIONAL essentiality: how the essential-gene set shifts with
growth condition (the core 'conditional genes' question), via FBA across media.

iJO1366, single-gene-deletion under several sole-carbon minimal media. Partition:
  CORE-essential  : essential in EVERY medium (unconditional backbone)
  CONDITIONAL     : essential in some media, dispensable in others
  never-essential : never required
Validate CORE against Keio (Baba) rich-medium truth. Quantifies the magnitude of
condition-dependence the regulatory/closed-loop layer must handle.
"""
import csv, json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz"))
CARBON={"glucose":"EX_glc__D_e","arabinose":"EX_arab__L_e","glycerol":"EX_glyc_e",
        "maltose":"EX_malt_e","galactose":"EX_gal_e","succinate":"EX_succ_e","acetate":"EX_ac_e"}
exset={r.id for r in m.reactions}
media=[c for c in CARBON if CARBON[c] in exset]
print(f"media tested: {media}")
allg=[g.id for g in m.genes]
ess_by_med={}
for c in media:
    with m as mm:
        med=mm.medium.copy()
        for cs,ex in CARBON.items():
            if ex in med or ex in exset: med[ex]= (10.0 if cs==c else 0.0)
        mm.medium=med
        wt=mm.slim_optimize()
        if not wt or wt<1e-6:
            print(f"  {c}: no growth, skip"); continue
        df=single_gene_deletion(mm,processes=1)
        ess=set()
        for _,row in df.iterrows():
            ids=row["ids"]; gid=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
            gr=row["growth"]
            if gr is None or (isinstance(gr,float) and np.isnan(gr)) or gr<0.01*wt: ess.add(gid)
        ess_by_med[c]=ess
        print(f"  {c}: WT={wt:.2f}  essential={len(ess)}")
media=list(ess_by_med)
cnt=defaultdict(int)
for c in media:
    for g in ess_by_med[c]: cnt[g]+=1
core=set(g for g in allg if cnt[g]==len(media))
cond=set(g for g in allg if 0<cnt[g]<len(media))
never=set(g for g in allg if cnt[g]==0)
print(f"\n=== conditional essentiality across {len(media)} media ({len(allg)} model genes) ===")
print(f"  CORE-essential (all media):   {len(core)}")
print(f"  CONDITIONAL (some media):     {len(cond)}")
print(f"  never-essential:              {len(never)}")
print(f"  -> {100*len(cond)/max(len(core)+len(cond),1):.0f}% of ever-essential genes are CONDITIONAL")

# validate CORE vs Keio truth
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ess_b={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"])
        if b: ess_b[b]=int(r["essential"])
uni=[g for g in allg if g in ess_b]; truth=set(g for g in uni if ess_b[g]==1)
def prf(pred):
    P=pred&set(uni); tp=len(P&truth); fp=len(P-truth); fn=len(truth-P)
    return dict(n=len(P),tp=tp,precision=round(tp/max(tp+fp,1),3),recall=round(tp/max(tp+fn,1),3))
print(f"\nvs Keio truth (n={len(uni)} mapped, {len(truth)} essential):")
print(f"  CORE-essential:        {prf(core)}")
print(f"  CORE+CONDITIONAL:      {prf(core|cond)}")
# example conditional genes
ex=[]
for g in list(cond)[:200]:
    meds=[c for c in media if g in ess_by_med[c]]
    if 1<=len(meds)<=3: ex.append((m.genes.get_by_id(g).name or g, meds))
json.dump(dict(media=media,n_genes=len(allg),core=len(core),conditional=len(cond),never=len(never),
               core_vs_keio=prf(core),core_cond_vs_keio=prf(core|cond),
               example_conditional=[{"gene":n,"essential_in":meds} for n,meds in ex[:25]]),
          open(OUT/"conditional_essentiality.json","w"),indent=2)
print("\nexample conditional-essential genes (essential only in some media):")
for n,meds in ex[:12]: print(f"  {n:<8} essential in: {meds}")
print(f"\nwrote {OUT}/conditional_essentiality.json")
