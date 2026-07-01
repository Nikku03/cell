"""Physical/signaling + pathway + metabolic layer: STRING physical PPI + Reactome pathways,
anchored/validated by disease (mutation -> disease -> pathway). Integrate with essentiality
and the disease data we already have."""
import gzip, csv, json, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); rng=np.random.default_rng(0)
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hh=f.readline().split("\t"); gi,li=hh.index("gene"),hh.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
g2dis=defaultdict(set)
for i,l in enumerate(open(H/"genes_to_phenotype.txt")):
    if i==0: continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6: g2dis[p[1]].add(p[5])
TFs=set()
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    if r["source_genesymbol"]: TFs.add(r["source_genesymbol"])
# STRING id maps
ensp2sym={}; ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    if l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)<3: continue
    if p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
    elif p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
print(f"STRING id map: {len(ensp2sym)} proteins -> symbol")
# STRING physical PPI (high confidence >=700)
ppi=defaultdict(set); nedge=0
for l in gzip.open(H/"string_physical.txt.gz","rt"):
    if l.startswith("protein1"): continue
    a,b,s=l.split()
    if int(s)<700: continue
    ga=ensp2sym.get(a); gb=ensp2sym.get(b)
    if ga and gb and ga!=gb: ppi[ga].add(gb); ppi[gb].add(ga); nedge+=1
print(f"STRING physical high-conf (>=700): {len(ppi)} proteins, {nedge} edges")
# Reactome pathways (human) symbol-based
path_genes=defaultdict(set); gene_paths=defaultdict(set); pname={}
for l in open(H/"reactome_human.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4 or not p[0].startswith("ENSG"): continue
    sym=ensg2sym.get(p[0])
    if not sym: continue
    path_genes[p[1]].add(sym); gene_paths[sym].add(p[1]); pname[p[1]]=p[3]
print(f"Reactome: {len(path_genes)} pathways, {len(gene_paths)} genes annotated")

# --- 1. essential/disease genes are PPI hubs ---
def deg(gs): v=[len(ppi[g]) for g in gs if g in ppi]; return round(float(np.mean(v)),1) if v else None
allppi=set(ppi)
disease_genes=set(g2dis)
print(f"\n=== PPI connectivity (mean interactions) ===")
print(f"  essential {deg(CEG)} | non-essential {deg(NEG)} | disease {deg(disease_genes)} | all {deg(allppi)} | TFs {deg(TFs)}")

# --- 2. physical interaction <-> shared disease (validation) ---
pairs=[(a,b) for a in ppi for b in ppi[a] if a<b and a in g2dis and b in g2dis]
sp=rng.choice(len(pairs),size=min(4000,len(pairs)),replace=False)
share_ppi=np.mean([len(g2dis[pairs[k][0]]&g2dis[pairs[k][1]])>0 for k in sp])
dg=[g for g in disease_genes if g in allppi]
rand=[(dg[rng.integers(len(dg))],dg[rng.integers(len(dg))]) for _ in range(4000)]
share_rand=np.mean([len(g2dis[a]&g2dis[b])>0 for a,b in rand if a!=b])
print(f"\n=== physical interaction <-> same disease ===")
print(f"  STRING-interacting disease-gene pairs share a disease: {share_ppi:.2f} vs random {share_rand:.2f} ({share_ppi/max(share_rand,1e-6):.0f}x)")

# --- 3. metabolic layer (Reactome Metabolism) ---
metab_paths=[pid for pid,nm in pname.items() if "metabolism" in nm.lower() or "metabolic" in nm.lower()]
metab_genes=set()
for pid in metab_paths: metab_genes|=path_genes[pid]
mg_hgem=set()
try:
    for r in csv.DictReader(open(H/"genes.tsv"),delimiter="\t"):
        if r.get("geneSymbols"): mg_hgem.add(r["geneSymbols"].split(";")[0])
except: pass
print(f"\n=== metabolic layer ===")
print(f"  Reactome metabolic genes: {len(metab_genes)} | Human-GEM genes: {len(mg_hgem)} | union: {len(metab_genes|mg_hgem)}")
print(f"  essential genes that are metabolic (Reactome): {len(CEG&metab_genes)}/{len(CEG)}")

# --- 4. mutation -> disease -> pathway (examples) ---
DIS_GENES=["LDLR","PAH","TP53","CFTR","HBB","BRCA1","MLH1","PTEN","G6PD","GATA4","RET","VHL"]
print(f"\n=== mutation -> disease -> pathway ===")
d2p=[]
for g in DIS_GENES:
    ps=[pname[pid] for pid in gene_paths.get(g,[])]
    ps=sorted(set(ps),key=len)[:2]
    d2p.append(dict(gene=g,pathways=ps))
    print(f"  {g:7s}: {ps}")
# pathway disease modules: pathways whose genes converge on one disease
mod=[]
for pid,genes in path_genes.items():
    dgs=[g for g in genes if g in g2dis]
    if len(dgs)>=4:
        from collections import Counter
        dc=Counter(d for g in dgs for d in g2dis[g])
        top,cnt=dc.most_common(1)[0]
        if cnt>=3: mod.append((pname[pid],cnt,len(dgs)))
mod.sort(key=lambda x:-x[1])
print(f"\n  pathway-level disease modules (>=3 genes share a disease): {len(mod)}; e.g.:")
for nm,cnt,n in mod[:6]: print(f"    {nm[:48]:48s} {cnt}/{n} genes share a disease")

json.dump(dict(ppi_proteins=len(ppi),ppi_edges=nedge,
    ppi_degree=dict(essential=deg(CEG),nonessential=deg(NEG),disease=deg(disease_genes),all=deg(allppi),TF=deg(TFs)),
    ppi_disease_enrichment=round(float(share_ppi/max(share_rand,1e-6)),1),
    reactome_pathways=len(path_genes),reactome_genes=len(gene_paths),
    metabolic_genes_reactome=len(metab_genes),metabolic_union=len(metab_genes|mg_hgem),
    disease_to_pathway=d2p,pathway_disease_modules=len(mod),top_modules=[{"pathway":n,"shared":c,"genes":g} for n,c,g in mod[:15]]),
    open(OUT/"physical_pathway_layer.json","w"),indent=2)
print("\nwrote physical_pathway_layer.json")
