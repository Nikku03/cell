"""Wheel 2 the way it was meant to be: NO FBA. Pure connectivity reasoning.

The cell looks at itself as a dependency graph (metabolite -> reaction ->
metabolite). A gene is NEEDED when removing its reaction(s) makes some
non-currency metabolite DANGLE -- lose all its producers (can no longer be
made) or all its consumers (piles up with nowhere to go). That is the literal
"I have glucose but nothing to metabolise it" hole. No biomass objective, no
medium, no linear program -- just who-connects-to-what.

Definition (graph necessity):
  gene g is GRAPH-ESSENTIAL  <=>  g is the SOLE producer or SOLE consumer of at
  least one non-currency metabolite (remove g and that metabolite is orphaned).

We then:
  1. validate graph-essential vs truth (Keio) on iML1515 -- precision/recall
  2. show it recovers FBA-essentials WITHOUT running FBA (overlap), and where
     the two differ
  3. one worked gap: a dangling metabolite -> the function needed to fill it
     -> the truck (same-class soup proteins) -> dimension filter -> the gene

Output: outputs/orphan/wheel2_graph.json
"""
import csv, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion   # ONLY for the comparison column
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")
MODEL="iML1515"

CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump",
          "ctp","cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2",
          "coa","co2","o2","nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2",
          "fe3","ca2","cobalt2","cu2","zn2","mobd","h2","q8","q8h2","mqn8","mql8",
          "2dmmql8","2dmmq8","h2o2","amet","ahcys","pi","ppi","nadp","accoa","acp",
          "thf","mlthf","methf","5mthf","10fthf","dhf","glu__L","akg","pep","pyr","atp"}
def base(mid): return mid.rsplit("_",1)[0]

print(f"loading {MODEL} (as a GRAPH, no FBA)...", flush=True)
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))

# ---- build connectivity: per metabolite, the set of producer/consumer reactions ----
# (reversible reactions count as both; exchange/demand/sink and biomass are kept so
#  exchangeables and biomass precursors have a legitimate consumer/producer)
producers=defaultdict(set); consumers=defaultdict(set)
metabolic=[]
for R in M.reactions:
    is_boundary = R.id.startswith(("EX_","DM_","SK_"))
    for met,coeff in R.metabolites.items():
        if R.reversibility:
            producers[met.id].add(R.id); consumers[met.id].add(R.id)
        elif coeff>0: producers[met.id].add(R.id)
        else: consumers[met.id].add(R.id)
    if not is_boundary and "BIOMASS" not in R.id.upper() and R.genes:
        metabolic.append(R)

def noncurrency(mid): return base(mid) not in CURRENCY

# ---- graph-essential genes: sole producer/consumer of a non-currency metabolite ----
print("scanning for sole-connection (graph-essential) genes -- pure graph, no LP...", flush=True)
t0=time.time()
gene_rxn=defaultdict(set)
for R in metabolic:
    for g in R.genes: gene_rxn[g.id].add(R.id)

graph_ess={}     # gene_id -> example orphaned metabolite (name) or None
for gid, rxns in gene_rxn.items():
    orphaned=None
    # metabolites touched by this gene's reactions
    mets=set()
    for rid in rxns:
        for m in M.reactions.get_by_id(rid).metabolites: mets.add(m.id)
    for mid in mets:
        if not noncurrency(mid): continue
        # would removing all of g's reactions orphan this metabolite?
        prod_left = producers[mid] - rxns
        cons_left = consumers[mid] - rxns
        # metabolite is "real" in the cell: had both a producer and consumer
        if producers[mid] and consumers[mid]:
            if not prod_left or not cons_left:
                orphaned=M.metabolites.get_by_id(mid).name
                break
    if orphaned is not None:
        graph_ess[gid]=orphaned
print(f"  graph-essential genes: {len(graph_ess)} / {len(gene_rxn)} ({time.time()-t0:.0f}s)", flush=True)

# ---- truth bridge (Keio b-number) ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
# model gene (b-number) -> our keio locus -> truth
def model_truth(bnum):
    lt=b_to_keio.get(bnum); return truth.get(lt) if lt else None

joinable={g:model_truth(g) for g in gene_rxn if model_truth(g) is not None}
print(f"  genes joinable to Keio truth: {len(joinable)}", flush=True)

def prec_rec(callset, universe):
    tp=sum(1 for g in callset if universe.get(g)==1)
    called=sum(1 for g in callset if g in universe)
    pos=sum(1 for g in universe if universe[g]==1)
    P=tp/max(called,1); R=tp/max(pos,1)
    return round(P,3), round(R,3), tp, called, pos

gP,gR,gtp,gca,gpos = prec_rec(set(graph_ess)&set(joinable), joinable)
print(f"\n=== GRAPH-ESSENTIAL vs truth (NO FBA) ===")
print(f"  precision={gP}  recall={gR}  (TP={gtp} of {gca} called; {gpos} true essentials)")

# ---- comparison column: FBA-essential (only to show graph matches it without LP) ----
print("\n(for comparison only) running FBA single-gene-deletion ...", flush=True)
for x in ["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
          "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
          "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
          "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
g0=M.slim_optimize()
de=single_gene_deletion(M)
fba_ess=set()
for _,row in de.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): fba_ess.add(gid)
fP,fR,ftp,fca,fpos = prec_rec(fba_ess&set(joinable), joinable)
print(f"  FBA-essential vs truth: precision={fP} recall={fR}")
# overlap graph vs FBA
gset=set(graph_ess)&set(joinable); fset=fba_ess&set(joinable)
both=gset&fset
print(f"\n  graph∩FBA={len(both)}  graph-only={len(gset-fset)}  FBA-only={len(fset-gset)}")
print(f"  of FBA-essentials, graph recovers {len(both)}/{len(fset)} ({len(both)/max(len(fset),1)*100:.0f}%) WITHOUT running FBA")

# ---- one worked gap (dangling metabolite -> needed function -> truck -> dims) ----
def ec_of(R):
    e=R.annotation.get("ec-code"); return (e[0] if isinstance(e,list) else e) if e else None
example=None
for gid,met in graph_ess.items():
    if joinable.get(gid)==1:
        rid=list(gene_rxn[gid])[0]; R=M.reactions.get_by_id(rid)
        example=dict(gene=gid, orphan_metabolite=met, reaction=R.id, eq=R.reaction,
                     ec=ec_of(R), name=R.name); break

print(f"\n=== worked gap (pure graph) ===")
if example:
    print(f"  cell sees: '{example['orphan_metabolite']}' would be orphaned without a gene")
    print(f"  needed function: {example['name']}  [EC {example['ec']}]")
    print(f"  reaction: {example['eq']}")
    print(f"  -> shop the soup for an enzyme of this class + matching substrate/cofactor/direction")
    print(f"  truth: gene {example['gene']} IS essential (Keio)")

json.dump(dict(model=MODEL, no_fba=True,
               n_graph_essential=len(graph_ess), n_joinable=len(joinable),
               graph_precision=gP, graph_recall=gR,
               fba_precision=fP, fba_recall=fR,
               graph_recovers_fba_frac=round(len(both)/max(len(fset),1),3),
               graph_only=len(gset-fset), fba_only=len(fset-gset), overlap=len(both),
               worked_example=example),
          open(OUT/"wheel2_graph.json","w"), indent=2)
print(f"\nwrote outputs/orphan/wheel2_graph.json", flush=True)
