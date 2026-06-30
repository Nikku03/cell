"""Augment cell_app_data.json for the visual cell map: add each gene's genomic
position (for the chromosome ring) and metabolic-pathway neighbours (genes whose
reactions share a metabolite) -> the 'pipes' connecting metabolism.
"""
import csv, json, warnings
from collections import defaultdict
from pathlib import Path
import cobra, cobra.io
warnings.filterwarnings("ignore"); cobra.Configuration().solver="glpk"
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan"); MODELS=Path("data/external_data/bigg_models")
data=json.load(open(OUT/"cell_app_data.json"))
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
# genomic position + strand + product per gene name
pos={}; GENOME=0
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit():
        pos[p[1].lower()]=(int(p[3]),int(p[4]),1 if p[5]=="forward" else -1)
        GENOME=max(GENOME,int(p[4]))
# membrane / compartment hint from product keywords
def compartment(g):
    pr=(g.get("product","") or "").lower()
    if any(k in pr for k in["transport","permease","abc ","porin","channel","symport","antiport","efflux","membrane"]): return "membrane"
    if g.get("is_tf"): return "nucleoid"
    if g.get("core"): return "ribosome"
    if g.get("metabolic"): return "metabolism"
    return "cytoplasm"
for g in data["genes"]:
    c=pos.get(g["name"]); g["pos"]=round(c[0]/GENOME,5) if c else None; g["strand"]=(c[2] if c else 0)
    g["comp"]=compartment(g)
data["genome_len"]=GENOME
# metabolic adjacency from iJO1366 (genes sharing a non-currency metabolite)
m=cobra.io.read_sbml_model(str(MODELS/"iJO1366.xml.gz"))
b2name={b:n for n,b in name2b.items()}
met2genes=defaultdict(set); deg=defaultdict(int)
for rx in m.reactions:
    gs=[g.id for g in rx.genes]
    if not gs: continue
    for met in rx.metabolites: deg[met.id]+=1
for rx in m.reactions:
    gs=[g.id for g in rx.genes]
    if not gs: continue
    for met in rx.metabolites:
        if deg[met.id]<=25:  # skip currency mets (high degree)
            for gid in gs: met2genes[met.id].add(gid)
adj=defaultdict(lambda: defaultdict(int))
for met,gs in met2genes.items():
    gl=list(gs)
    for i in range(len(gl)):
        for j in range(i+1,len(gl)):
            adj[gl[i]][gl[j]]+=1; adj[gl[j]][gl[i]]+=1
# store as name->[neighbour names] capped
metab_adj={}
for gid,nb in adj.items():
    nm=b2name.get(gid)
    if not nm: continue
    top=sorted(nb.items(),key=lambda x:-x[1])[:8]
    metab_adj[nm]=[b2name.get(b) for b,_ in top if b2name.get(b)]
data["metab_adj"]=metab_adj
json.dump(data,open(OUT/"cell_map_data.json","w"))
print(f"genome {GENOME} bp; genes with pos {sum(1 for g in data['genes'] if g['pos'] is not None)}; "
      f"metabolic adjacency for {len(metab_adj)} genes")
print("compartments:", {c:sum(1 for g in data['genes'] if g['comp']==c) for c in ['membrane','nucleoid','ribosome','metabolism','cytoplasm']})
print("wrote outputs/orphan/cell_map_data.json")
