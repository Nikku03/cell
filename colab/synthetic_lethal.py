"""Synthetic lethality / redundant-functional-essential genes.

syn3A keeps 'non-essential' genes partly because single-gene deletion is blind
to redundancy: gene A with backup B is individually dispensable, but {A,B} is
essential. Our Wheel 3 (single_gene_deletion) AND our ground truth (Keio single
knockouts) share this blind spot.

This quantifies the blind spot on iML1515 (rich medium):
  1. single-gene-deletion -> individually essential vs non-essential
  2. double-gene-deletion among redundancy candidates -> synthetic-lethal pairs
     (both individually non-essential, pair lethal/sick)
  3. SL genes = 'individually dispensable but functionally essential'
  4. cross-check: are they correctly NON-essential in Keio truth? (shared blind
     spot) and does our paralog feature already flag them?

Output: outputs/orphan/synthetic_lethal.json
"""
import csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion, double_gene_deletion
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); MODEL="iML1515"

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L",
      "leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L",
      "ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R",
      "pydx","4abz","btn","fol"]
print(f"loading {MODEL}...", flush=True)
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))
for x in RICH:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
g0=M.slim_optimize()
print(f"  rich growth g0={g0:.3f}", flush=True)

# ---- single-gene deletion ----
print("single-gene deletion...", flush=True)
sd=single_gene_deletion(M)
single_ess=set(); single_growth={}
for _,row in sd.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    single_growth[gid]=row["growth"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): single_ess.add(gid)
noness=[g.id for g in M.genes if g.id not in single_ess]
print(f"  individually essential: {len(single_ess)}  non-essential: {len(noness)}", flush=True)

# ---- redundancy candidates: non-essential genes in an isozyme reaction
#      (a reaction catalysable by >=2 genes) -- the cleanest SL mechanism ----
iso_genes=set()
for R in M.reactions:
    if len(R.genes)>=2:
        for g in R.genes: iso_genes.add(g.id)
cands=sorted((set(noness) & iso_genes))
print(f"  redundancy candidates (non-ess & in isozyme rxn): {len(cands)}", flush=True)

# ---- double-gene deletion among candidates ----
print(f"double-gene deletion on {len(cands)} candidates ({len(cands)*(len(cands)-1)//2} pairs)...", flush=True)
t0=time.time()
dd=double_gene_deletion(M, gene_list1=cands, gene_list2=cands)
print(f"  ({time.time()-t0:.0f}s)", flush=True)
sl_pairs=[]; sick_pairs=[]
for _,row in dd.iterrows():
    ids=list(row["ids"])
    if len(ids)!=2: continue
    a,b=ids
    if a in single_ess or b in single_ess: continue   # need BOTH individually non-ess
    gr=row["growth"]
    if gr<0.01*g0 or np.isnan(gr): sl_pairs.append((a,b,gr))
    elif gr<0.5*g0: sick_pairs.append((a,b,gr))
sl_genes=set()
for a,b,_ in sl_pairs: sl_genes.add(a); sl_genes.add(b)
print(f"\n  SYNTHETIC-LETHAL pairs: {len(sl_pairs)}  (synthetic-sick: {len(sick_pairs)})", flush=True)
print(f"  genes that are individually dispensable but functionally essential: {len(sl_genes)}", flush=True)

# ---- cross-check vs Keio truth + paralog feature ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
def keio(b):
    lt=b_to_keio.get(b); return truth.get(lt) if lt else None
paralogs={}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r["organism"]=="beril_Keio" and r.get("n_paralogs_in_genome"):
        # key by b-number via reverse bridge
        pass
# paralog count keyed by our keio locus
par_by_lt={}
for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")):
    if r["organism"]=="beril_Keio" and r.get("n_paralogs_in_genome"):
        par_by_lt[r["locus_tag"]]=float(r["n_paralogs_in_genome"])

sl_join=[(g, keio(g)) for g in sl_genes if keio(g) is not None]
truth_ess_among_sl=sum(1 for g,t in sl_join if t==1)
# paralog enrichment: SL genes vs all non-essential genes
def paralog_of(b):
    lt=b_to_keio.get(b); return par_by_lt.get(lt) if lt else None
sl_par=[paralog_of(g) for g in sl_genes]; sl_par=[p for p in sl_par if p is not None]
all_ne_par=[paralog_of(g) for g in noness]; all_ne_par=[p for p in all_ne_par if p is not None]
print(f"\n  of {len(sl_join)} SL genes joinable to Keio: {truth_ess_among_sl} are truth-ESSENTIAL "
      f"({truth_ess_among_sl/max(len(sl_join),1)*100:.0f}%)  -> rest are correctly individually non-ess (shared blind spot)")
print(f"  mean paralogs: SL genes={np.mean(sl_par) if sl_par else 0:.2f}  vs all non-ess={np.mean(all_ne_par) if all_ne_par else 0:.2f}")
frac_sl_with_paralog=np.mean([p>0 for p in sl_par]) if sl_par else 0
frac_ne_with_paralog=np.mean([p>0 for p in all_ne_par]) if all_ne_par else 0
print(f"  fraction with >=1 paralog: SL={frac_sl_with_paralog:.2f}  vs all non-ess={frac_ne_with_paralog:.2f}")

# example SL pairs with names
print(f"\n  example synthetic-lethal pairs:")
for a,b,gr in sl_pairs[:8]:
    na=M.genes.get_by_id(a).name or a; nb=M.genes.get_by_id(b).name or b
    print(f"    {a}({na}) + {b}({nb})  -> growth {gr:.3f}")

json.dump(dict(model=MODEL,
   single_essential=len(single_ess), single_nonessential=len(noness),
   redundancy_candidates=len(cands),
   synthetic_lethal_pairs=len(sl_pairs), synthetic_sick_pairs=len(sick_pairs),
   functionally_essential_genes=len(sl_genes),
   sl_genes_truth_essential=truth_ess_among_sl, sl_genes_joinable=len(sl_join),
   mean_paralogs_sl=float(np.mean(sl_par)) if sl_par else 0,
   mean_paralogs_all_noness=float(np.mean(all_ne_par)) if all_ne_par else 0,
   frac_paralog_sl=float(frac_sl_with_paralog), frac_paralog_noness=float(frac_ne_with_paralog),
   example_pairs=[(a,b,float(gr)) for a,b,gr in sl_pairs[:25]]),
   open(OUT/"synthetic_lethal.json","w"),indent=2)
print(f"\nwrote outputs/orphan/synthetic_lethal.json", flush=True)
