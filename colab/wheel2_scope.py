"""FBA-free Wheel 2, completed: NETWORK EXPANSION (reachability), no LP.

The cell asks: 'starting from what the medium gives me, which molecules can I
build, step by step?' That is network expansion -- pure graph reachability, no
objective, no flux. A gene is NEEDED when removing its reaction(s) makes some
required building block (a biomass precursor) UN-buildable.

This replaces FBA's growth objective with reachability:
  scope(blocked) = fixpoint of 'a reaction fires if all its inputs are already
                   available; its outputs then become available'
  gene g is SCOPE-ESSENTIAL  <=>  some biomass precursor reachable in the intact
                   cell is NOT reachable once g's reactions are removed.

Candidates are the sole-connection genes (only those can break producibility);
we then keep the ones that actually disconnect a required output.

Validate on iML1515 vs Keio truth, and compare to FBA (run only for the
comparison column) to show graph reachability matches it without LP.

Output: outputs/orphan/wheel2_scope.json
"""
import csv, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_gene_deletion   # comparison only
cobra.Configuration().solver="glpk"
OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models"); MODEL="iML1515"
CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump",
          "ctp","cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2","coa",
          "co2","o2","nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2","fe3","ca2",
          "cobalt2","cu2","zn2","mobd","h2","q8","q8h2","mqn8","mql8","2dmmql8","2dmmq8",
          "h2o2","amet","ahcys","accoa","acp","thf","mlthf","methf","5mthf","10fthf","dhf",
          "glu__L","akg","pep","pyr"}
def base(mid): return mid.rsplit("_",1)[0]

print(f"loading {MODEL} as a graph...", flush=True)
M=cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))

# rich medium uptake (defines what the cell is GIVEN)
RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L","ile__L",
      "leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L","tyr__L","val__L",
      "ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm","ribflv","nac","pnto__R",
      "pydx","4abz","btn","fol"]
for x in RICH:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10

# ---- seed: currency (recycled) + medium inputs ----
seed=set()
for m in M.metabolites:
    if base(m.id) in CURRENCY: seed.add(m.id)
for R in M.reactions:
    if R.id.startswith("EX_") and R.lower_bound<0:
        for m in R.metabolites: seed.add(m.id)
print(f"  seed metabolites (currency+medium): {len(seed)}", flush=True)

# ---- directed reaction list + consumer index for worklist expansion ----
dirs=[]   # (reactants_frozenset, products_tuple, rid)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")): continue
    subs=tuple(m.id for m,c in R.metabolites.items() if c<0)
    pros=tuple(m.id for m,c in R.metabolites.items() if c>0)
    if subs and pros:
        dirs.append((frozenset(subs),pros,R.id))
        if R.reversibility: dirs.append((frozenset(pros),subs,R.id))
consumers_of=defaultdict(list)   # metabolite -> indices of dirs needing it
for i,(subs,pros,rid) in enumerate(dirs):
    for s in subs: consumers_of[s].append(i)

def expand(blocked):
    avail=set(seed)
    # initial fireable
    queue=[i for i,(subs,pros,rid) in enumerate(dirs) if dirs[i][2] not in blocked and subs<=avail]
    fired=set()
    # worklist
    changed=True
    while changed:
        changed=False
        for i,(subs,pros,rid) in enumerate(dirs):
            if i in fired or rid in blocked: continue
            if subs<=avail:
                newp=[p for p in pros if p not in avail]
                if newp:
                    avail.update(newp); changed=True
                fired.add(i)
    return avail

# biomass precursors
bio=[R for R in M.reactions if "BIOMASS" in R.id.upper()]
bio_pre=set()
for R in bio:
    for m,c in R.metabolites.items():
        if c<0 and base(m.id) not in CURRENCY: bio_pre.add(m.id)
print(f"  biomass precursors (non-currency): {len(bio_pre)}", flush=True)

t0=time.time()
base_scope=expand(set())
reach0={p for p in bio_pre if p in base_scope}
print(f"  baseline reachable precursors: {len(reach0)}/{len(bio_pre)}  (expand {time.time()-t0:.1f}s)", flush=True)

# ---- candidate genes = sole producer/consumer of a non-currency met ----
producers=defaultdict(set); consumers=defaultdict(set)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")): continue
    for m,c in R.metabolites.items():
        if R.reversibility: producers[m.id].add(R.id); consumers[m.id].add(R.id)
        elif c>0: producers[m.id].add(R.id)
        else: consumers[m.id].add(R.id)
gene_rxn=defaultdict(set)
for R in M.reactions:
    if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper(): continue
    for g in R.genes: gene_rxn[g.id].add(R.id)
cand=set()
for gid,rxns in gene_rxn.items():
    mets=set()
    for rid in rxns:
        for m in M.reactions.get_by_id(rid).metabolites: mets.add(m.id)
    for mid in mets:
        if base(mid) in CURRENCY: continue
        if producers[mid] and consumers[mid] and (not (producers[mid]-rxns) or not (consumers[mid]-rxns)):
            cand.add(gid); break
print(f"  sole-connection candidate genes: {len(cand)}", flush=True)

# ---- scope-essential: removing the gene (GPR-respecting; isozymes rescue) un-reaches
#      a required precursor. Block a reaction ONLY if the knockout actually disables it. ----
print("  testing scope-essentiality (pure reachability + GPR, no LP)...", flush=True)
base_zero={r.id for r in M.reactions if r.lower_bound==0 and r.upper_bound==0}
t0=time.time()
scope_ess={}
for k,gid in enumerate(sorted(cand)):
    with M:
        M.genes.get_by_id(gid).knock_out()
        blocked={r.id for r in M.reactions if r.lower_bound==0 and r.upper_bound==0} - base_zero
    if not blocked: continue
    sc=expand(blocked)
    dropped=[p for p in reach0 if p not in sc]
    if dropped:
        scope_ess[gid]=M.metabolites.get_by_id(dropped[0]).name
print(f"  scope-essential genes: {len(scope_ess)} ({time.time()-t0:.0f}s)", flush=True)

# ---- truth + validation ----
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv")) if r["organism"]=="beril_Keio"}
def mt(b):
    lt=b_to_keio.get(b); return truth.get(lt) if lt else None
joinable={g:mt(g) for g in gene_rxn if mt(g) is not None}
def pr(callset):
    s=callset&set(joinable)
    tp=sum(1 for g in s if joinable[g]==1); pos=sum(1 for g in joinable if joinable[g]==1)
    return round(tp/max(len(s),1),3), round(tp/max(pos,1),3), tp, len(s), pos

sP,sR,stp,sca,spos=pr(set(scope_ess))
cP,cR,ctp,cca,_=pr(cand)
print(f"\n=== SCOPE-ESSENTIAL (FBA-free reachability) vs truth ===")
print(f"  sole-connection only : P={cP} R={cR} ({cca} called)")
print(f"  + reachability filter: P={sP} R={sR} ({sca} called, {stp} true of {spos})")

# FBA comparison
print("\n(comparison only) FBA single-gene-deletion ...", flush=True)
g0=M.slim_optimize(); de=single_gene_deletion(M); fba=set()
for _,row in de.iterrows():
    gid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): fba.add(gid)
fP,fR,ftp,fca,_=pr(fba)
sset=set(scope_ess)&set(joinable); fset=fba&set(joinable)
print(f"  FBA vs truth: P={fP} R={fR}")
print(f"  scope∩FBA={len(sset&fset)}  scope-only={len(sset-fset)}  FBA-only={len(fset-sset)}")
print(f"  scope recovers {len(sset&fset)}/{len(fset)} ({len(sset&fset)/max(len(fset),1)*100:.0f}%) of FBA-essentials WITHOUT LP")

json.dump(dict(model=MODEL,no_fba=True,
   sole_connection={"precision":cP,"recall":cR,"called":cca},
   scope_essential={"precision":sP,"recall":sR,"called":sca,"tp":stp,"pos":spos},
   fba={"precision":fP,"recall":fR},
   scope_recovers_fba=round(len(sset&fset)/max(len(fset),1),3),
   scope_only=len(sset-fset), fba_only=len(fset-sset), overlap=len(sset&fset)),
   open(OUT/"wheel2_scope.json","w"),indent=2)
print(f"\nwrote outputs/orphan/wheel2_scope.json", flush=True)
