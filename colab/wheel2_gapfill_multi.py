"""Wheel 2 gap-directed funnel across all 4 native metabolic models.
Runs the same identify-the-doorway-by-dimensions experiment on each model,
prints per-model funnels, and aggregates."""
import csv, json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_reaction_deletion
cobra.Configuration().solver = "glpk"
OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")

RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
      "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
      "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
      "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
CURRENCY={"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump",
          "ctp","cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2",
          "coa","co2","o2","nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2",
          "fe3","ca2","cobalt2","cu2","zn2","mobd","h2","q8","q8h2","mqn8","mql8",
          "2dmmql8","2dmmq8","thf","mlthf","methf","5mthf","10fthf","dhf","h2o2","no",
          "no2","no3","n2o","amet","ahcys","glu__L"}
REDUCED={"nadh","nadph","fadh2","fmnh2","q8h2","mql8","2dmmql8"}
def base(mid): return mid.rsplit("_",1)[0]
def carbons(m):
    f=m.formula or ""; mt=re.search(r"C(\d+)",f); return int(mt.group(1)) if mt else 0
def cofactor(bases):
    if {"nadh","nad"}&bases: return "NAD"
    if {"nadph","nadp"}&bases: return "NADP"
    if {"fadh2","fad","fmn","fmnh2"}&bases: return "FAD/FMN"
    if {"atp","adp","amp","gtp","gdp"}&bases: return "ATP/NTP"
    if {"coa"}&bases: return "CoA"
    return "none"
def direction(R):
    if R.reversibility: return "reversible"
    for m,c in R.metabolites.items():
        if base(m.id) in REDUCED: return "oxidation" if c>0 else "reduction"
    return "n/a"
def size_bucket(R):
    subs=[carbons(m) for m in R.metabolites if base(m.id) not in CURRENCY]
    mx=max(subs) if subs else 0
    return "small(<=4C)" if mx<=4 else "med(5-10C)" if mx<=10 else "large(>10C)"
def ec_of(R):
    e=R.annotation.get("ec-code")
    if not e: return None
    return e[0] if isinstance(e,list) else e
def ec1(R):
    e=ec_of(R); return e.split(".")[0] if e else None
def ec3(R):
    e=ec_of(R);
    if not e: return None
    p=e.split("."); return ".".join(p[:3]) if len(p)>=3 else e
def shape(R):
    bases={base(m.id) for m in R.metabolites}
    return dict(ec1=ec1(R), ec3=ec3(R), cof=cofactor(bases),
                dirn=direction(R), size=size_bucket(R))

TIERS=[("T0 truck (EC-1 only)",  []),
       ("T1 +cofactor",          ["cof"]),
       ("T2 +direction",         ["cof","dirn"]),
       ("T3 +substrate size",    ["cof","dirn","size"]),
       ("T4 +EC 3-level",        ["cof","dirn","size","ec3"])]
def filt(cands, s, gene_rxns, tiers):
    out=set()
    for g in cands:
        for (rid,gs) in gene_rxns[g]:
            if gs["ec1"]!=s["ec1"]: continue
            ok=True
            if "cof"  in tiers and gs["cof"]!=s["cof"]: ok=False
            if "dirn" in tiers and not (gs["dirn"]==s["dirn"] or "reversible" in (gs["dirn"],s["dirn"])): ok=False
            if "size" in tiers and gs["size"]!=s["size"]: ok=False
            if "ec3"  in tiers and gs["ec3"]!=s["ec3"]: ok=False
            if ok: out.add(g); break
    return out

def run_model(model_id, label):
    print(f"\n=== {label} ({model_id}) ===", flush=True)
    t0=time.time()
    M=cobra.io.read_sbml_model(str(MODELS/f"{model_id}.xml.gz"))
    for x in RICH:
        rid=f"EX_{x}_e"
        if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
    g0=M.slim_optimize()
    print(f"  growth={g0:.3f}/h  rxns={len(M.reactions)} genes={len(M.genes)}", flush=True)
    rxn_shape={}; metabolic=[]
    for R in M.reactions:
        if R.id.startswith(("EX_","DM_","SK_")) or "BIOMASS" in R.id.upper() or not R.genes: continue
        rxn_shape[R.id]=shape(R); metabolic.append(R)
    gene_rxns=defaultdict(list); gene_ec1=defaultdict(set)
    for R in metabolic:
        s=rxn_shape[R.id]
        for g in R.genes:
            gene_rxns[g.id].append((R.id,s))
            if s["ec1"]: gene_ec1[g.id].add(s["ec1"])
    truck_of_ec1=defaultdict(set)
    for g,ecs in gene_ec1.items():
        for e in ecs: truck_of_ec1[e].add(g)
    de=single_reaction_deletion(M, reaction_list=[R.id for R in metabolic])
    ess_rxn=set()
    for _,row in de.iterrows():
        rid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
        if row["growth"]<0.01*g0 or np.isnan(row["growth"]): ess_rxn.add(rid)
    holes=[(R, rxn_shape[R.id], {g.id for g in R.genes}) for R in metabolic
           if R.id in ess_rxn and rxn_shape[R.id]["ec1"]]
    print(f"  essential rxns={len(ess_rxn)}  holes with EC class={len(holes)}  ({time.time()-t0:.0f}s)", flush=True)
    agg={name:dict(pool=[],retained=0) for name,_ in TIERS}
    for (R,s,true_genes) in holes:
        truck=set(truck_of_ec1[s["ec1"]])
        if not truck: continue
        for name,tiers in TIERS:
            keep=truck if not tiers else filt(truck, s, gene_rxns, tiers)
            agg[name]["pool"].append(len(keep))
            if true_genes & keep: agg[name]["retained"]+=1
    n=len(holes)
    print(f"  {'tier':<22}{'median':>8}{'mean':>8}{'true-gene retained':>22}")
    rows=[]
    for name,_ in TIERS:
        pools=agg[name]["pool"]; ret=agg[name]["retained"]
        med=int(np.median(pools)) if pools else 0
        mean=float(np.mean(pools)) if pools else 0
        rows.append(dict(tier=name, median=med, mean=mean, retained=ret))
        print(f"    {name:<20}{med:>8}{mean:>8.1f}{ret:>14}/{n} ({ret/max(n,1)*100:.0f}%)")
    med0=int(np.median(agg[TIERS[0][0]]["pool"])) if agg[TIERS[0][0]]["pool"] else 0
    med4=int(np.median(agg[TIERS[-1][0]]["pool"])) if agg[TIERS[-1][0]]["pool"] else 0
    shrink=med0/max(med4,1)
    ret_final=agg[TIERS[-1][0]]["retained"]
    print(f"  shrink T0->T4: {shrink:.0f}x   "
          f"(true gene kept {ret_final}/{n} = {ret_final/max(n,1)*100:.0f}%)")
    return dict(model=model_id, label=label, n_holes=n,
                shrink=float(shrink), tiers=rows)

CONFIG=[("iJN1463","P. putida"),
        ("iML1515","E. coli (Keio)"),
        ("iEK1008","M. tuberculosis"),
        ("iYL1228","K. oxytoca")]
results=[run_model(mid,lab) for mid,lab in CONFIG]

print("\n=== AGGREGATE ===")
print(f"{'model':<22}{'holes':>7}{'truck':>8}{'+cof':>7}{'+dir':>7}{'+size':>7}{'+EC3':>7}{'shrink':>9}{'kept':>9}")
for r in results:
    m=[t['median'] for t in r['tiers']]; k=r['tiers'][-1]['retained']
    print(f"  {r['label']:<20}{r['n_holes']:>7}{m[0]:>8}{m[1]:>7}{m[2]:>7}{m[3]:>7}{m[4]:>7}"
          f"{r['shrink']:>7.0f}x{k:>5}/{r['n_holes']}")
tot_holes=sum(r['n_holes'] for r in results); tot_kept=sum(r['tiers'][-1]['retained'] for r in results)
all_pools={name:[m for r in results for m in [r['tiers'][i]['median']]*r['n_holes']] for i,(name,_) in enumerate(TIERS)}
print(f"\n  TOTAL holes across 4 models: {tot_holes}  "
      f"true gene kept at T4: {tot_kept}/{tot_holes} ({tot_kept/tot_holes*100:.0f}%)")
print(f"  median candidate-pool by tier:  "
      + "  ".join(f"{name.split(' ',1)[0]}={int(np.median(all_pools[name]))}"
                  for name,_ in TIERS))

json.dump(results, open(OUT/"wheel2_gapfill_multi.json","w"), indent=2)
print(f"\nwrote outputs/orphan/wheel2_gapfill_multi.json", flush=True)
