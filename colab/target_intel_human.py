"""Same structure as target_intel.py, but on a HUMAN cell — the cross-kingdom
generalization test. Does the FBA metabolic-necessity wheel that generalized across
BACTERIA also predict HUMAN gene essentiality? Reframed as oncology target discovery:
a cell's essential genes are the drug targets.

Truth: Hart CEGv2 (core-essential) vs NEGv1 (non-essential) reference sets (the human
analog of Keio essential/non-essential). Model: Human-GEM (Chalmers).
Honest headline metric: what FRACTION of human essentiality is even metabolic?
"""
import csv, json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

# ---- gene id map + truth ----
ensg2sym={}
for r in csv.DictReader(open(H/"genes.tsv"),delimiter="\t"):
    if r.get("genes") and r.get("geneSymbols"): ensg2sym[r["genes"]]=r["geneSymbols"].split(";")[0]
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i==0: continue
        g=l.split("\t")[0].strip().strip('"')
        if g: s.add(g)
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
print(f"truth: {len(CEG)} core-essential, {len(NEG)} non-essential (Hart reference sets)")

# ---- load Human-GEM ----
print("loading Human-GEM (43MB) ...",flush=True)
m=cobra.io.read_sbml_model(str(H/"Human-GEM.xml"))
print(f"  {len(m.genes)} genes, {len(m.reactions)} reactions, {len(m.metabolites)} metabolites")
# objective = the ENABLED biomass reaction (MAR04413 ships disabled ub=0; MAR13082 is the
# active generic biomass; exclude maintenance/exchange variants)
rids={r.id for r in m.reactions}
if "MAR13082" in rids:
    bio=m.reactions.get_by_id("MAR13082")
else:
    cand=[r for r in m.reactions if "biomass" in (r.name or "").lower() and r.upper_bound>0
          and "maintenance" not in (r.name or "").lower() and "exchange" not in (r.name or "").lower()]
    bio=cand[0] if cand else None
m.objective=bio; print(f"  objective -> {bio.id} ({bio.name})")
wt=m.slim_optimize()
print(f"  WT biomass flux = {wt}")

# model gene symbols
gsym={g.id: ensg2sym.get(g.id, (g.name or g.id)) for g in m.genes}
sym2ensg=defaultdict(list)
for e,s in gsym.items(): sym2ensg[s].append(e)
model_syms=set(gsym.values())

# ---- metabolic COVERAGE of human essentiality (the honest headline) ----
ceg_in_model=CEG & model_syms; neg_in_model=NEG & model_syms
print(f"\n=== metabolic coverage of essentiality ===")
print(f"  core-essential genes that are metabolic (in Human-GEM): {len(ceg_in_model)}/{len(CEG)} "
      f"({100*len(ceg_in_model)/len(CEG):.0f}%)")
print(f"  non-essential genes that are metabolic: {len(neg_in_model)}/{len(NEG)} ({100*len(neg_in_model)/len(NEG):.0f}%)")

# ---- FBA single-gene deletion -> essential metabolic genes ----
print("\nrunning Human-GEM single-gene deletion (may take several min) ...",flush=True)
df=single_gene_deletion(m,processes=1)
ess_ensg=set()
for _,row in df.iterrows():
    ids=row["ids"]; e=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
    gr=row["growth"]; gr=0.0 if gr is None or gr!=gr else gr
    if wt>1e-6 and gr<0.01*wt: ess_ensg.add(e)
ess_sym={gsym[e] for e in ess_ensg if e in gsym}
print(f"  FBA-essential metabolic genes: {len(ess_sym)}")

# ---- benchmark vs CEG/NEG over metabolic genes ----
def metrics(pred_sym):
    uni=(CEG|NEG) & model_syms
    truth=CEG & model_syms
    P=pred_sym & uni
    tp=len(P&truth); fp=len(P-truth); fn=len(truth-P); tn=len(uni-truth-P)
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1
    mcc=(tp*tn-fp*fn)/den
    return dict(precision=round(prec,3),recall=round(rec,3),mcc=round(float(mcc),3),
                tp=tp,fp=fp,fn=fn,tn=tn,n_eval=len(uni))
mb=metrics(ess_sym)
print(f"\n=== HUMAN FBA essentiality vs Hart CEG/NEG (metabolic genes only) ===")
print(f"  {mb}")
print(f"  (bacterial reference: E. coli FBA precision 0.455 recall 0.163)")

# ---- condition dependence: Warburg (glucose vs glutamine) ----
def ess_without(exch_substr):
    with m as mm:
        for r in mm.reactions:
            if r.boundary and any(s in (r.name or "").lower() or s in r.id.lower() for s in exch_substr):
                r.lower_bound=0.0
        w=mm.slim_optimize()
        if not w or w!=w or w<1e-6: return None,0.0
        d=single_gene_deletion(mm,processes=1); es=set()
        for _,row in d.iterrows():
            ids=row["ids"]; e=list(ids)[0] if isinstance(ids,(set,frozenset)) else ids
            gr=row["growth"]; gr=0.0 if gr is None or gr!=gr else gr
            if gr<0.01*w: es.add(gsym.get(e))
        return es,float(w)
print("\n=== condition dependence (cancer metabolism) ===")
glc_es,wg=ess_without(["glucose"]); gln_es,wq=ess_without(["glutamine"])
print(f"  glucose removed:   wt={wg:.3f}  new essentials={len(glc_es-ess_sym) if glc_es else 'no-growth'}")
print(f"  glutamine removed: wt={wq:.3f}  new essentials={len(gln_es-ess_sym) if gln_es else 'no-growth'}")

# ---- druggability proxy + target table over metabolic genes ----
b2sub={}
for rx in m.reactions:
    sub=(rx.subsystem or "").strip()
    for gg in rx.genes:
        if sub and gg.id not in b2sub: b2sub[gg.id]=sub
rows=[]
for e in gsym:
    s=gsym[e]; is_ess=int(e in ess_ensg)
    truth = 1 if s in CEG else (0 if s in NEG else None)
    rows.append(dict(gene=s,ensg=e,fba_essential=is_ess,hart_truth=truth,
        subsystem=b2sub.get(e,"")[:40], druggability=0.9))
rows.sort(key=lambda r:(-r["fba_essential"], r["gene"]))
with open(OUT/"target_intel_human.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- known ONCOLOGY metabolic target recovery ----
ONCO={"DHFR","TYMS","GART","ATIC","PPAT","IMPDH1","IMPDH2","RRM1","RRM2","DHODH",
      "TYMP","TK1","SHMT1","SHMT2","PHGDH","MTHFD1","MTHFD2","ODC1","HPRT1","TYMS",
      "NAMPT","PNP","ADA","GLS","LDHA","HK2","PKM"}
onco_ess=[g for g in ONCO if g in ess_sym]
onco_in_model=[g for g in ONCO if g in model_syms]
print(f"\n=== known oncology METABOLIC targets ===")
print(f"  in model: {len(onco_in_model)}/{len(ONCO)}; FBA-essential: {sorted(onco_ess)}")

summary=dict(model="Human-GEM", n_genes=len(m.genes), n_reactions=len(m.reactions),
    truth=dict(CEG=len(CEG),NEG=len(NEG)),
    metabolic_coverage=dict(
        ceg_in_model=len(ceg_in_model), ceg_total=len(CEG), ceg_frac=round(len(ceg_in_model)/len(CEG),3),
        neg_in_model=len(neg_in_model), neg_total=len(NEG)),
    fba_essential_metabolic=len(ess_sym),
    benchmark_metabolic_only=mb,
    bacterial_reference=dict(ecoli_fba_precision=0.455, ecoli_fba_recall=0.163),
    warburg=dict(glucose_removed_new_ess=(len(glc_es-ess_sym) if glc_es else None),
                 glutamine_removed_new_ess=(len(gln_es-ess_sym) if gln_es else None)),
    oncology_targets=dict(in_model=onco_in_model, fba_essential=sorted(onco_ess)))
json.dump(summary,open(OUT/"target_intel_human.json","w"),indent=2)
print("\nwrote target_intel_human.csv + target_intel_human.json")
