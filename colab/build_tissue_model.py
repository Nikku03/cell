"""TISSUE MODEL (separate from the cell model): multiple cell types per tissue, wired by cell-cell
communication (ligand in sender -> receptor in receiver). Data: HPA single-cell expression (154 cell
types) + Omnipath ligand-receptor (6,658 pairs). -> tissue_model.json"""
import json, csv, io, zipfile
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
# --- HPA single-cell expression: cell type -> {gene: nCPM} ---
expr=defaultdict(dict)
z=zipfile.ZipFile(H/"hpa_sc.tsv.zip")
f=io.TextIOWrapper(z.open("rna_single_cell_type.tsv"),encoding="utf-8")
rd=csv.reader(f,delimiter="\t"); next(rd)
for g_ensg,gene,ct,ncpm in rd:
    try: v=float(ncpm)
    except: continue
    if v>=1: expr[ct][gene]=v
print("HPA cell types:",len(expr))
gsum=defaultdict(float); NT=len(expr)
for ct in expr:
    for g,v in expr[ct].items(): gsum[g]+=v
gmean={g:s/NT for g,s in gsum.items()}
def expressed(ct,g,thr=1): return expr.get(ct,{}).get(g,0)>=thr
def spec(ct,g): return expr.get(ct,{}).get(g,0)/(gmean.get(g,0)+0.5)   # enrichment vs mean across cell types
def is_ligand_of(ct,g): return expr.get(ct,{}).get(g,0)>=3 and spec(ct,g)>=2   # specific + expressed
def top_markers(ct,n=8):
    out=[(g,spec(ct,g)) for g,v in expr[ct].items() if v>5 and spec(ct,g)>=3]
    return [g for g,_ in sorted(out,key=lambda x:-x[1])[:n]]
# --- Omnipath ligand-receptor ---
LR=[]
for r in csv.DictReader(open(H/"omnipath_ligrec.tsv"),delimiter="\t"):
    l=r.get("source_genesymbol"); rec=r.get("target_genesymbol")
    if l and rec and l!=rec:
        s=1 if r.get("is_stimulation")=="1" else (-1 if r.get("is_inhibition")=="1" else 0)
        LR.append((l,rec,s))
LR=list({(l,rec,s) for l,rec,s in LR})
print("ligand-receptor pairs:",len(LR))
# --- curated tissues (HPA cell type names) ---
TISSUES={
 "Liver":["hepatocytes","cholangiocytes","kupffer cells","hepatic stellate cells","vascular endothelial cells"],
 "Heart":["cardiomyocytes","fibroblasts","vascular endothelial cells","macrophages","epicardial cells"],
 "Lung":["alveolar cells type 1","alveolar cells type 2","macrophages","vascular endothelial cells","fibroblasts"],
 "Brain":["brain excitatory neurons","brain inhibitory neurons","astrocytes","oligodendrocytes","vascular endothelial cells"],
 "Kidney":["proximal tubule cells","distal convoluted tubule cells","podocytes","vascular endothelial cells","macrophages"],
 "Intestine":["enterocytes","colonocytes","enteric stem cells","fibroblasts","t-cells"],
 "Skin":["basal keratinocytes","suprabasal keratinocytes","melanocytes","fibroblasts","vascular endothelial cells"],
 "Immune (blood)":["t-cells","b-cells","nk-cells","macrophages","cdc"],
}
avail=set(expr)
model={}
for tis,cts in TISSUES.items():
    cts=[c for c in cts if c in avail]
    comm=[]   # (sender, receiver, [ [ligand, receptor, sign], ... ])
    for s in cts:
        for r in cts:
            pairs=[]
            for l,rec,sg in LR:
                if is_ligand_of(s,l) and expressed(r,rec,1):
                    score=spec(s,l)*max(1,spec(r,rec))
                    pairs.append([l,rec,sg,round(score,1)])
            if pairs:
                pairs.sort(key=lambda x:-x[3])
                comm.append([s,r,pairs[:25]])
    cells={c:dict(markers=top_markers(c),
                  ligands=sorted({l for l,rec,sg in LR if is_ligand_of(c,l)}),
                  receptors=sorted({rec for l,rec,sg in LR if expressed(c,rec,1)})) for c in cts}
    model[tis]=dict(cells=cts,cellinfo=cells,comm=comm)
    print(f"  {tis}: {len(cts)} cell types, {len(comm)} communication channels")
DATA=dict(tissues=model,n_lr=len(LR))
json.dump(DATA,open(OUT/"tissue_model.json","w"),separators=(",",":"))
print("wrote tissue_model.json (%d KB)"%(len(json.dumps(DATA))//1024))
