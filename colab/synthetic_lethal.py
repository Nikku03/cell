"""Redundant-functional-essential genes via ISOZYME reactions (fast, no pairwise).

syn3A keeps 'non-essential' genes partly because single-gene deletion is blind
to redundancy. The dominant mechanism is isozymes: a reaction catalysed by >=2
genes. Each gene is individually dispensable (the other covers the reaction),
but the reaction -- and thus the function -- is essential.

Find these directly (no 400k-pair double deletion):
  1. single-gene-deletion  -> which genes are individually essential
  2. single-REACTION-deletion -> which reactions are essential
  3. ISOZYME-redundant-essential reaction = essential reaction with >=2 genes,
     none of which is individually essential. Its genes are 'individually
     dispensable but functionally essential' -- the blind spot.
  4. cross-check vs Keio truth (should be non-essential -> shared blind spot)
     and vs our paralog feature (do we already flag them?).

Output: outputs/orphan/synthetic_lethal.json
"""
import csv, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion, single_reaction_deletion
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

t0=time.time()
sd=single_gene_deletion(M)
single_ess=set()
for _,row in sd.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): single_ess.add(gid)
print(f"  individually essential genes: {len(single_ess)} / {len(M.genes)}", flush=True)

rd=single_reaction_deletion(M)
rxn_ess=set()
for _,row in rd.iterrows():
    rid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): rxn_ess.add(rid)
print(f"  essential reactions: {len(rxn_ess)}  ({time.time()-t0:.0f}s)", flush=True)

# isozyme-redundant essential reactions
redundant=[]   # (rxn_id, [genes]) essential rxn, >=2 genes, none individually essential
func_ess_genes=set()
for R in M.reactions:
    if R.id not in rxn_ess: continue
    genes=[g.id for g in R.genes]
    if len(genes)>=2 and not any(g in single_ess for g in genes):
        redundant.append((R.id, R.name, genes))
        for g in genes: func_ess_genes.add(g)
print(f"\n  ISOZYME-redundant essential reactions: {len(redundant)}", flush=True)
print(f"  genes individually dispensable BUT functionally essential: {len(func_ess_genes)}", flush=True)

# cross-check vs Keio truth + paralog feature
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
par_by_lt={r["locus_tag"]:float(r["n_paralogs_in_genome"])
           for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv"))
           if r["organism"]=="beril_Keio" and r.get("n_paralogs_in_genome")}
def keio(b):
    lt=b_to_keio.get(b); return truth.get(lt) if lt else None
def paralog(b):
    lt=b_to_keio.get(b); return par_by_lt.get(lt) if lt else None

join=[(g,keio(g)) for g in func_ess_genes if keio(g) is not None]
truth_ess=sum(1 for g,t in join if t==1)
fe_par=[paralog(g) for g in func_ess_genes if paralog(g) is not None]
noness_genes=[g.id for g in M.genes if g.id not in single_ess]
ne_par=[paralog(g) for g in noness_genes if paralog(g) is not None]
print(f"\n  of {len(join)} functionally-essential genes joinable to Keio:")
print(f"    truth-ESSENTIAL: {truth_ess} ({truth_ess/max(len(join),1)*100:.0f}%)  "
      f"-> {len(join)-truth_ess} correctly individually NON-essential = SHARED BLIND SPOT")
print(f"  paralogs: functionally-essential genes mean={np.mean(fe_par) if fe_par else 0:.2f}  "
      f"vs all non-essential mean={np.mean(ne_par) if ne_par else 0:.2f}")
print(f"  fraction with >=1 paralog: func-ess={np.mean([p>0 for p in fe_par]) if fe_par else 0:.2f}  "
      f"vs all non-ess={np.mean([p>0 for p in ne_par]) if ne_par else 0:.2f}")

print(f"\n  example redundant essential reactions (function essential, genes individually safe):")
for rid,name,genes in redundant[:10]:
    gn=[M.genes.get_by_id(g).name or g for g in genes]
    print(f"    {rid:<10} {name[:42]:<42} genes={gn}")

json.dump(dict(model=MODEL, g0=float(g0),
   individually_essential=len(single_ess), essential_reactions=len(rxn_ess),
   isozyme_redundant_essential_reactions=len(redundant),
   functionally_essential_genes=len(func_ess_genes),
   func_ess_joinable=len(join), func_ess_truth_essential=truth_ess,
   shared_blind_spot_genes=len(join)-truth_ess,
   mean_paralogs_funcess=float(np.mean(fe_par)) if fe_par else 0,
   mean_paralogs_noness=float(np.mean(ne_par)) if ne_par else 0,
   frac_paralog_funcess=float(np.mean([p>0 for p in fe_par])) if fe_par else 0,
   frac_paralog_noness=float(np.mean([p>0 for p in ne_par])) if ne_par else 0,
   examples=[(rid,name,[M.genes.get_by_id(g).name or g for g in genes]) for rid,name,genes in redundant[:30]]),
   open(OUT/"synthetic_lethal.json","w"),indent=2)
print(f"\nwrote outputs/orphan/synthetic_lethal.json", flush=True)
