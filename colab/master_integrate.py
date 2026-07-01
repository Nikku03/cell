"""Integrate every layer into one per-gene 'cell object', anchored on genome coordinates.
Output: integrated_cell_human.csv/json = the human cell laid out (kinetics-free)."""
import gzip, csv, json, bisect, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
# constraint
loeuf={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    hh=f.readline().split("\t"); gi,li=hh.index("gene"),hh.index("oe_lof_upper")
    for l in f:
        p=l.split("\t")
        try: v=float(p[li])
        except: continue
        if p[gi] not in loeuf or v<loeuf[p[gi]]: loeuf[p[gi]]=v
# regulatory network (CollecTRI+TRRUST) out/in degree
outdeg=defaultdict(set); indeg=defaultdict(set); TFs=set()
for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
    a,b=r["source_genesymbol"],r["target_genesymbol"]
    if a and b: outdeg[a].add(b); indeg[b].add(a); TFs.add(a)
for l in open(H/"trrust_human.tsv"):
    p=l.split("\t")
    if len(p)>=3: outdeg[p[0]].add(p[1]); indeg[p[1]].add(p[0]); TFs.add(p[0])
# STRING PPI degree
ensp2sym={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3 and p[2]=="Ensembl_HGNC_symbol": ensp2sym[p[0]]=p[1]
ppideg=defaultdict(set)
for l in gzip.open(H/"string_physical.txt.gz","rt"):
    if l.startswith("protein1"): continue
    a,b,s=l.split()
    if int(s)<700: continue
    ga=ensp2sym.get(a); gb=ensp2sym.get(b)
    if ga and gb and ga!=gb: ppideg[ga].add(gb); ppideg[gb].add(ga)
# Reactome pathway count + ensg2sym
ensg2sym={ensp2ensg:ensp2sym[e] for e in () for ensp2ensg in ()}  # placeholder
ensp2ensg={}
for l in gzip.open(H/"string_aliases.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=3 and p[2]=="Ensembl_HGNC_ensembl_gene_id": ensp2ensg[p[0]]=p[1]
ensg2sym={ensp2ensg[e]:ensp2sym[e] for e in ensp2sym if e in ensp2ensg}
npath=defaultdict(int)
for l in open(H/"reactome_human.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<4 or not p[0].startswith("ENSG"): continue
    sym=ensg2sym.get(p[0])
    if sym: npath[sym]+=1
# HPO disease count
ndis=defaultdict(set)
for i,l in enumerate(open(H/"genes_to_phenotype.txt")):
    if i==0: continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6: ndis[p[1]].add(p[5])
# lineage master TFs (differentiation)
LIN={"pluripotent":["POU5F1","SOX2","NANOG","KLF4"],"erythroid":["GATA1","KLF1","TAL1"],
 "myeloid":["SPI1","CEBPA"],"lymphoid":["PAX5","EBF1","GATA3","TCF7"],"neuron":["NEUROG2","NEUROD1","ASCL1"],
 "muscle":["MYOD1","MYF5","MYOG"],"cardiac":["NKX2-5","GATA4","TBX5","MEF2C"],"hepatocyte":["HNF4A","HNF1A","FOXA2"],
 "pancreas":["PDX1","NKX6-1","NEUROG3"],"osteoblast":["RUNX2","SP7"],"melanocyte":["MITF","SOX10"],"retina":["CRX","NRL"]}
master={m:lin for lin,ms in LIN.items() for m in ms}
# genome coordinates (refGene, longest tx / gene)
tss={}
for l in gzip.open(H/"refGene.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)<13: continue
    sym=p[12]; chrom=p[2]; strand=p[3]; txs=int(p[4]); txe=int(p[5])
    if "_" in chrom: continue
    t=txs if strand=="+" else txe
    if sym not in tss or (txe-txs)>tss[sym][3]:
        tss[sym]=(chrom,t,strand,txe-txs)
# epigenetics per gene (CpG promoter + enhancer count)
cpg=defaultdict(list)
for l in gzip.open(H/"cpgIslandExt.txt.gz","rt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=4: cpg[p[1]].append(int(p[2]))
for c in cpg: cpg[c].sort()
enh=defaultdict(list)
for l in open(H/"cCREs.bed"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and "ELS" in p[5]: enh[p[0]].append((int(p[1])+int(p[2]))//2)
for c in enh: enh[c].sort()
def has_cpg(ch,pos,pad=1000):
    v=cpg.get(ch); 
    if not v: return 0
    i=bisect.bisect_left(v,pos-pad); return 1 if i<len(v) and v[i]<=pos+pad else 0
def n_enh(ch,pos,w=50000):
    a=enh.get(ch); 
    if not a: return 0
    return bisect.bisect_right(a,pos+w)-bisect.bisect_left(a,pos-w)
# assemble
allg=set(tss)|set(loeuf)|set(outdeg)|set(indeg)|set(ppideg)|set(npath)|set(ndis)|CEG|NEG
rows=[]
for g in sorted(allg):
    c=tss.get(g); chrom=c[0] if c else None; pos=c[1] if c else None
    rows.append(dict(gene=g,chrom=chrom,tss=pos,
        essential=(1 if g in CEG else (0 if g in NEG else None)),
        loeuf=round(loeuf[g],3) if g in loeuf else None,
        is_tf=int(g in TFs), regulon_out=len(outdeg.get(g,())), regulators_in=len(indeg.get(g,())),
        ppi_degree=len(ppideg.get(g,())), n_pathways=npath.get(g,0),
        cpg_promoter=(has_cpg(chrom,pos) if c else None), enhancers=(n_enh(chrom,pos) if c else None),
        n_diseases=len(ndis.get(g,())), lineage_master=master.get(g,"")))
import csv as _csv
with open(OUT/"integrated_cell_human.csv","w",newline="") as f:
    w=_csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
chrom_sizes={l.split()[0]:int(l.split()[1]) for l in open(H/"hg38.chrom.sizes") if l.split()[0] in {f"chr{i}" for i in list(range(1,23))+['X','Y']}}
json.dump(dict(n_genes=len(rows),chrom_sizes=chrom_sizes,
    layers=["coords","essentiality","constraint","TF_network","PPI","pathways","epigenetics","disease","differentiation"],
    coverage={k:sum(1 for r in rows if r[k] not in (None,0,"")) for k in ["tss","loeuf","is_tf","ppi_degree","n_pathways","cpg_promoter","n_diseases","lineage_master"]}),
    open(OUT/"integrated_cell_human.json","w"),indent=2)
print(f"integrated {len(rows)} genes across 9 layers")
print("layer coverage:", {k:sum(1 for r in rows if r[k] not in (None,0,'')) for k in ['tss','loeuf','is_tf','ppi_degree','n_pathways','cpg_promoter','n_diseases']})
print("wrote integrated_cell_human.csv + .json")
