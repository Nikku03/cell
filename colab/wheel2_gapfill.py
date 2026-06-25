"""Wheel 2 (proper): gap-directed gene identification with a within-family
DIMENSION filter.

The analogy, made literal on a genome-scale model (iML1515 E. coli):

  1. ASSEMBLE the cell from confident essentials -> some reactions are still
     required for growth but their gene is "in the soup" (uncertain). Each such
     required reaction is a HOLE -- a doorway in the wall.

  2. The assembled network can only describe the hole COARSELY: "I need an
     oxidoreductase here" (EC level-1 class). It cannot name the gene. That's
     the "I need a roof / a door, but not the brand or style" limit.

  3. TRUCKS arrive from the soup: every gene whose enzyme is the same coarse
     class (EC-1). Lots of candidates -- most are the wrong door.

  4. DIMENSIONS narrow within the truck. The doorway's measurements come from
     the metabolites the assembled cell ALREADY HAS flanking the gap -- not from
     knowing the answer:
        - cofactor   (NAD vs NADP vs FAD vs ATP vs CoA)   "the power source"
        - direction  (oxidation vs reduction)             "which way it swings"
        - substrate  carbon-size bucket                   "the size"
        - EC 3-level sub-subclass                          "the exact style"
     Keep only candidates whose own chemistry fits the doorway.

  5. MEASURE: how much does each dimension shrink the wet-lab candidate list
     (= knockouts you'd have to try), and is the TRUE gene always retained?

This turns "hit-and-trial over the whole soup" into "test the 5 doors that
actually fit the frame."
"""
import csv, json, time, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import cobra, cobra.io
from cobra.flux_analysis import single_reaction_deletion
cobra.Configuration().solver = "glpk"

OUT = Path("outputs/orphan"); MODELS = Path("data/external_data/bigg_models")
MODEL = "iML1515"

# ---- rich medium (match the screen) ----
RICH=["ala__L","arg__L","asn__L","asp__L","cys__L","gln__L","glu__L","gly","his__L",
      "ile__L","leu__L","lys__L","met__L","phe__L","pro__L","ser__L","thr__L","trp__L",
      "tyr__L","val__L","ade","gua","ura","thymd","cytd","ins","adn","gsn","uri","thm",
      "ribflv","nac","pnto__R","pydx","4abz","btn","fol"]
print(f"loading {MODEL} ...", flush=True)
M = cobra.io.read_sbml_model(str(MODELS/f"{MODEL}.xml.gz"))
for x in RICH:
    rid=f"EX_{x}_e"
    if rid in M.reactions: M.reactions.get_by_id(rid).lower_bound=-10
g0 = M.slim_optimize()
print(f"  rich growth g0 = {g0:.3f}/h", flush=True)

# ---- dimension extraction ----
CURRENCY = {"h2o","h","pi","ppi","atp","adp","amp","gtp","gdp","gmp","utp","udp","ump",
            "ctp","cdp","cmp","nad","nadh","nadp","nadph","fad","fadh2","fmn","fmnh2",
            "coa","co2","o2","nh4","hco3","so4","h2s","na1","k","cl","mg2","mn2","fe2",
            "fe3","ca2","cobalt2","cu2","zn2","mobd","h2","q8","q8h2","mqn8","mql8",
            "2dmmql8","2dmmq8","thf","mlthf","methf","5mthf","10fthf","dhf","pi","ppi",
            "h2o2","no","no2","no3","n2o","nadp","amet","ahcys","glu__L"}
REDUCED = {"nadh","nadph","fadh2","fmnh2","q8h2","mql8","2dmmql8"}
def base(mid): return mid.rsplit("_",1)[0]
def carbons(m):
    f=m.formula or ""
    mt=re.search(r"C(\d+)", f)
    return int(mt.group(1)) if mt else 0
def cofactor(bases):
    if {"nadh","nad"} & bases: return "NAD"
    if {"nadph","nadp"} & bases: return "NADP"
    if {"fadh2","fad","fmn","fmnh2"} & bases: return "FAD/FMN"
    if {"atp","adp","amp","gtp","gdp"} & bases: return "ATP/NTP"
    if {"coa"} & bases: return "CoA"
    return "none"
def direction(R):
    if R.reversibility: return "reversible"
    # in stored direction, reactant coeff<0, product coeff>0
    for m,c in R.metabolites.items():
        if base(m.id) in REDUCED:
            return "oxidation" if c>0 else "reduction"  # reduced form produced => substrate oxidized
    return "n/a"
def size_bucket(R):
    subs=[carbons(m) for m in R.metabolites if base(m.id) not in CURRENCY]
    mx=max(subs) if subs else 0
    return "small(<=4C)" if mx<=4 else "med(5-10C)" if mx<=10 else "large(>10C)"
def ec_of(R):
    e=R.annotation.get("ec-code")
    if not e: return None
    if isinstance(e,list): e=e[0]
    return e
def ec1(R):
    e=ec_of(R); return e.split(".")[0] if e else None
def ec3(R):
    e=ec_of(R)
    if not e: return None
    p=e.split("."); return ".".join(p[:3]) if len(p)>=3 else e
def shape(R):
    bases={base(m.id) for m in R.metabolites}
    return dict(ec1=ec1(R), ec3=ec3(R), cof=cofactor(bases),
               dirn=direction(R), size=size_bucket(R))

# precompute shapes + per-gene coarse-class membership
EC1_NAME={"1":"oxidoreductase","2":"transferase","3":"hydrolase","4":"lyase",
          "5":"isomerase","6":"ligase","7":"translocase"}
rxn_shape={}
metabolic=[]
for R in M.reactions:
    if R.id.startswith("EX_") or R.id.startswith("DM_") or R.id.startswith("SK_"): continue
    if "BIOMASS" in R.id.upper(): continue
    if not R.genes: continue
    s=shape(R); rxn_shape[R.id]=s
    metabolic.append(R)

# gene -> list of (rxn_id, shape) ; gene -> set of ec1
gene_rxns=defaultdict(list); gene_ec1=defaultdict(set)
for R in metabolic:
    s=rxn_shape[R.id]
    for g in R.genes:
        gene_rxns[g.id].append((R.id,s))
        if s["ec1"]: gene_ec1[g.id].add(s["ec1"])
# truck index: ec1 -> set of genes of that coarse class
truck_of_ec1=defaultdict(set)
for g,ecs in gene_ec1.items():
    for e in ecs: truck_of_ec1[e].add(g)

# ---- Step 1: HOLES = essential gene-associated metabolic reactions ----
print("Step 1: single-reaction deletion to find holes (essential rxns)...", flush=True)
t0=time.time()
de = single_reaction_deletion(M, reaction_list=[R.id for R in metabolic])
ess_rxn=set()
for _,row in de.iterrows():
    rid=list(row["ids"])[0] if isinstance(row["ids"],(set,frozenset)) else row["ids"]
    if row["growth"]<0.01*g0 or np.isnan(row["growth"]): ess_rxn.add(rid)
print(f"  {len(ess_rxn)} essential gene-associated reactions ({time.time()-t0:.0f}s)", flush=True)

# keep holes that have an EC-1 class (so a truck exists) and >=1 true gene
holes=[]
for R in metabolic:
    if R.id not in ess_rxn: continue
    s=rxn_shape[R.id]
    if not s["ec1"]: continue
    true_genes={g.id for g in R.genes}
    holes.append((R, s, true_genes))
print(f"  {len(holes)} holes with a coarse EC class + known gene\n", flush=True)

# ---- Steps 2-5: truck -> dimension filter, measure shrinkage + retention ----
def filt(cands, s, tiers):
    """keep gene g if it has a same-ec1 reaction matching the requested dims."""
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

TIERS=[("T0 truck (EC-1 only)", []),
       ("T1 +cofactor",         ["cof"]),
       ("T2 +direction",        ["cof","dirn"]),
       ("T3 +substrate size",   ["cof","dirn","size"]),
       ("T4 +EC 3-level",       ["cof","dirn","size","ec3"])]

agg={name:dict(pool=[],retained=0) for name,_ in TIERS}
examples=[]
for (R,s,true_genes) in holes:
    truck = set(truck_of_ec1[s["ec1"]])           # all soup-family candidates
    if not truck: continue
    row=dict(rxn=R.id, name=R.name, ec=ec_of(R), cof=s["cof"], dirn=s["dirn"],
             size=s["size"], class_=EC1_NAME.get(s["ec1"],s["ec1"]))
    for name,tiers in TIERS:
        keep = truck if not tiers else filt(truck, s, tiers)
        agg[name]["pool"].append(len(keep))
        if true_genes & keep: agg[name]["retained"]+=1
        row[name]=len(keep)
    examples.append(row)

n=len(examples)
print(f"=== Funnel over {n} holes: median candidate-pool size per dimension ===")
print(f"{'tier':<24}{'median':>8}{'mean':>8}{'true-gene retained':>22}")
for name,_ in TIERS:
    pools=agg[name]["pool"]; ret=agg[name]["retained"]
    print(f"  {name:<22}{int(np.median(pools)):>8}{np.mean(pools):>8.1f}"
          f"{ret:>14}/{n} ({ret/n*100:.0f}%)")
shrink = np.median(agg[TIERS[0][0]]["pool"]) / max(np.median(agg[TIERS[-1][0]]["pool"]),1)
print(f"\n  median shrink T0 -> T4: {shrink:.0f}x   "
      f"(true gene retained {agg[TIERS[-1][0]]['retained']/n*100:.0f}% of the time)")

# wet-lab interpretation: expected knockouts to hit the right gene = pool/2
exp0=np.median(agg[TIERS[0][0]]["pool"])/2
exp4=np.median(agg[TIERS[-1][0]]["pool"])/2
print(f"  expected knockouts-to-hit:  blind ~{exp0:.0f}   ->   dimension-filtered ~{exp4:.0f}")

# show 6 worked examples (largest blind truck first)
examples.sort(key=lambda r:-r[TIERS[0][0]])
print(f"\n=== Worked examples (doorway -> dimensions -> candidates) ===")
for r in examples[:6]:
    print(f"  {r['rxn']:<10} {r['name'][:40]:<40} [{r['class_']}]")
    print(f"     dims: cofactor={r['cof']}, dir={r['dirn']}, size={r['size']}, EC={r['ec']}")
    print(f"     candidates:  T0={r[TIERS[0][0]]}  ->  +cof={r[TIERS[1][0]]}  "
          f"+dir={r[TIERS[2][0]]}  +size={r[TIERS[3][0]]}  +EC3={r[TIERS[4][0]]}")

json.dump(dict(model=MODEL, n_holes=n,
               tiers={name:dict(median=float(np.median(agg[name]["pool"])),
                                mean=float(np.mean(agg[name]["pool"])),
                                retained=agg[name]["retained"]) for name,_ in TIERS},
               median_shrink=float(shrink),
               examples=examples[:40]),
          open(OUT/"wheel2_gapfill.json","w"), indent=2)
print(f"\nwrote outputs/orphan/wheel2_gapfill.json", flush=True)
