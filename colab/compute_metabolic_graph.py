"""Parse Human-GEM into a metabolite-reaction graph for alternative-pathway finding + isozymes.
-> metabolic_graph.json {mets:[names], rxns:[{s,p,rev,g,eq,id}], hubs:[met ids], gene2rxn}."""
import re, json, gzip, csv
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
# ENSG -> symbol (same map the builder uses)
ensp2sym={}; ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<3: continue
    if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
    elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
ENSG=re.compile(r"ENSG\d+"); COEF=re.compile(r"^(\d+(?:\.\d+)?)\s+(.+)$")
def parse_side(side):
    outs=[]
    for tok in side.split(" + "):
        tok=tok.strip()
        if not tok: continue
        m=COEF.match(tok)
        if m: tok=m.group(2)
        if "[" in tok: name=tok[:tok.rfind("[")].strip()
        else: name=tok
        if name: outs.append(name)          # compartment-collapsed metabolite name
    return outs
# hub / currency metabolites to exclude from route connectivity (classic metabolic-pathfinding fix)
_nuc=[b+s for b in ["A","G","U","C","I","X","dA","dG","dC","dT","dU","dI"] for s in ["TP","DP","MP"]]
HUB=set(_nuc)|{
 "H2O","H+","H","e-","CO2","O2","NH3","NH4+","H2O2","O2-","OH-",
 "Pi","PPi","Pi(2-)","phosphate","diphosphate","PAP","PAPS","APS",
 "NAD+","NADH","NADP+","NADPH","FAD","FADH2","FMN","FMNH2","riboflavin",
 "CoA","ACP","[ACP]","holo-[ACP]","apo-[ACP]",
 "thioredoxin","oxidized thioredoxin","GSH","GSSG","glutathione","ferredoxin",
 "Na+","K+","Cl-","Ca2+","Mg2+","Mn2+","Fe2+","Fe3+","Zn2+","Cu2+","Co2+","HCO3-",
 "sulfate","sulfite","H2S","H2S2O3","SO4","thiosulfate","S",
 "SAM","SAH","S-adenosyl-L-methionine","S-adenosyl-L-homocysteine"}
mets={}; def_id=lambda n: mets.setdefault(n,len(mets))
rxns=[]; gene2rxn=defaultdict(set)
for l in open(H/"human_gem.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<3 or p[0]=="Rxn name": continue
    rid,formula,gpr=p[0],p[1],p[2]
    rev="<=>" in formula
    sep=" <=> " if rev else (" => " if " => " in formula else None)
    if not sep: continue
    L,R=formula.split(sep,1)
    subs=[def_id(n) for n in parse_side(L)]; prods=[def_id(n) for n in parse_side(R)]
    if not subs or not prods: continue
    syms=sorted({ensg2sym[e] for e in set(ENSG.findall(gpr)) if e in ensg2sym})
    eq=formula.replace(" <=> "," ⇌ ").replace(" => "," → ")
    ri=len(rxns); rxns.append(dict(s=subs,p=prods,rev=rev,g=syms,eq=(eq[:120]),id=rid))
    for sy in syms: gene2rxn[sy].add(ri)
metnames=[None]*len(mets)
for n,i in mets.items(): metnames[i]=n
hub_ids=[mets[n] for n in HUB if n in mets]
DATA=dict(mets=metnames,rxns=rxns,hubs=hub_ids,gene2rxn={k:sorted(v) for k,v in gene2rxn.items()})
json.dump(DATA,open(OUT/"metabolic_graph.json","w"),separators=(",",":"))
# isozyme stats: reactions with >1 enzyme = substitutable
iso=sum(1 for r in rxns if len(r["g"])>1)
print("metabolites:",len(mets),"| reactions:",len(rxns),"| with genes:",sum(1 for r in rxns if r["g"]))
print("reactions with isozymes (>1 enzyme):",iso,"| genes:",len(gene2rxn),"| hubs excluded:",len(hub_ids))
print("size:",len(json.dumps(DATA))//1024,"KB")
