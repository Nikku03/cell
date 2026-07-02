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
# LOOP 1: signaling pathway families (CellChat via Omnipath)
pathway={}
try:
    for l in open(H/"omnipath_cellchat.tsv",encoding="utf-8",errors="ignore"):
        p=l.rstrip("\n").split("\t")
        if len(p)>=6 and p[4]=="pathway" and p[1] and p[1] not in pathway: pathway[p[1]]=p[5]
except Exception as e: print("cellchat:",e)
# LOOP 2: signaling MODE (secreted=paracrine vs transmembrane=juxtacrine) from CellPhoneDB
u2s={}
try:
    for r in csv.DictReader(open(H/"cpdb_gene.csv")):
        if r.get("uniprot") and r.get("hgnc_symbol"): u2s[r["uniprot"]]=r["hgnc_symbol"]
except Exception as e: print("cpdb_gene:",e)
secreted=set(); membrane=set()
try:
    for r in csv.DictReader(open(H/"cpdb_protein.csv")):
        s=u2s.get(r.get("uniprot"))
        if not s: continue
        if (r.get("secreted") or "").upper()=="TRUE": secreted.add(s)
        if (r.get("transmembrane") or "").upper()=="TRUE": membrane.add(s)
except Exception as e: print("cpdb_protein:",e)
def lmode(l): return "paracrine" if l in secreted else ("juxtacrine" if l in membrane else "signal")
def lr_path(l,rec): return pathway.get(l) or pathway.get(rec) or ""
print("pathway families:",len(set(pathway.values())),"| secreted ligands:",len(secreted),"| membrane:",len(membrane))
# LOOP 3: druggable communication — DGIdb drugs on ligands/receptors
lrgenes=set(l for l,rec,sg in LR)|set(rec for l,rec,sg in LR)
drugs=defaultdict(list)
try:
    rd=csv.reader(open(H/"dgidb.tsv"),delimiter="\t"); next(rd)
    seen=set()
    for row in rd:
        if len(row)>9 and row[2] in lrgenes and row[9] and not row[9].lower().startswith("chembl"):
            k=(row[2],row[9].lower())
            if k in seen or len(drugs[row[2]])>=5: continue
            seen.add(k); drugs[row[2]].append(row[9].title())
except Exception as e: print("dgidb:",e)
print("druggable LR genes:",len(drugs))
print("ligand-receptor pairs:",len(LR))
# --- curated tissues (HPA cell type names) ---
TISSUES={
 "Liver":["hepatocytes","cholangiocytes","kupffer cells","hepatic stellate cells","vascular endothelial cells","t-cells"],
 "Heart":["cardiomyocytes","fibroblasts","vascular endothelial cells","macrophages","pericytes","t-cells"],
 "Lung":["alveolar cells type 1","alveolar cells type 2","macrophages","vascular endothelial cells","fibroblasts","t-cells"],
 "Brain":["brain excitatory neurons","brain inhibitory neurons","astrocytes","oligodendrocytes","microglia","vascular endothelial cells"],
 "Kidney":["proximal tubule cells","distal convoluted tubule cells","podocytes","vascular endothelial cells","macrophages","fibroblasts"],
 "Intestine":["enterocytes","colonocytes","goblet cells","paneth cells","enteric stem cells","t-cells","macrophages"],
 "Skin":["basal keratinocytes","suprabasal keratinocytes","melanocytes","fibroblasts","vascular endothelial cells","t-cells"],
 "Pancreas":["pancreatic acinar cells","pancreatic duct cells","pancreatic islet cells","fibroblasts","vascular endothelial cells","macrophages"],
 "Immune (blood)":["t-cells","b-cells","nk-cells","macrophages","cdc","plasma cells"],
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
                    md="autocrine" if s==r else lmode(l)
                    pairs.append([l,rec,sg,round(score,1),lr_path(l,rec),md])
            if pairs:
                pairs.sort(key=lambda x:-x[3])
                comm.append([s,r,pairs[:25]])
    cells={c:dict(markers=top_markers(c),
                  ligands=sorted({l for l,rec,sg in LR if is_ligand_of(c,l)}),
                  receptors=sorted({rec for l,rec,sg in LR if expressed(c,rec,1)})) for c in cts}
    pwcount=defaultdict(int)
    for s,r,pairs in comm:
        for p in pairs:
            if len(p)>4 and p[4]: pwcount[p[4]]+=1
    links=[[s,r,round(sum(p[3] for p in pairs)),len(pairs)] for s,r,pairs in comm if s!=r]
    links.sort(key=lambda x:-x[2])
    model[tis]=dict(cells=cts,cellinfo=cells,comm=comm,
                    pathways=dict(sorted(pwcount.items(),key=lambda x:-x[1])[:15]),
                    toplinks=links[:8])
    print(f"  {tis}: {len(cts)} cell types, {len(comm)} channels, top pathways: {list(dict(sorted(pwcount.items(),key=lambda x:-x[1])[:4]))}")
# --- downstream effect INSIDE the receiver (reuse the CELL MODEL's shared network data:
#     SIGNOR receptor->signaling + CollecTRI TF->target). Baked into the tissue file; no cell files touched. ---
from collections import defaultdict as dd
sig_out=dd(list)
try:
    for l in open(H/"signor.tsv",encoding="utf-8",errors="ignore"):
        p=l.split("\t")
        if len(p)>=9 and p[0] and p[4] and p[0]!=p[4]: sig_out[p[0]].append(p[4])
except Exception as e: print("signor:",e)
reg_out=dd(list); tfset=set()
try:
    for r in csv.DictReader(open(H/"collectri.tsv"),delimiter="\t"):
        a=r["source_genesymbol"]; b=r["target_genesymbol"]
        if a and b and a!=b: reg_out[a].append(b); tfset.add(a)
except Exception as e: print("collectri:",e)
receptors=set()
for tis in model.values():
    for s,r,pairs in tis["comm"]:
        for p in pairs: receptors.add(p[1])
downstream={}
for rec in receptors:
    seen=set([rec]); q=[(rec,0)]
    while q:
        x,d=q.pop()
        if d>=2: continue
        for y in sig_out.get(x,[])[:25]:
            if y not in seen: seen.add(y); q.append((y,d+1))
    seen.discard(rec)
    tfs=[g for g in seen if g in tfset][:6]
    targets=set()
    for tf in tfs: targets.update(reg_out.get(tf,[])[:15])
    if seen or targets:
        downstream[rec]=dict(sig=sorted(seen)[:10],tfs=tfs,targets=sorted(targets)[:18])
print("receptors with downstream response wired:",len(downstream),"of",len(receptors))
# LOOP 6: cross-tissue ENDOCRINE axes — organs signalling systemically by hormones (curated, textbook)
ENDOCRINE=[
 ["INS","Pancreas","Liver","INSR","lowers blood glucose (uptake/storage)"],
 ["GCG","Pancreas","Liver","GCGR","raises blood glucose (glycogenolysis)"],
 ["GCG","Intestine","Pancreas","GLP1R","GLP-1 stimulates insulin secretion (incretin)"],
 ["FGF19","Intestine","Liver","FGFR4","bile-acid synthesis feedback"],
 ["HAMP","Liver","Intestine","SLC40A1","hepcidin blocks iron export (iron homeostasis)"],
 ["EPO","Kidney","Immune (blood)","EPOR","drives red-blood-cell production"],
 ["REN","Kidney","Liver","AGT","renin-angiotensin: blood pressure"],
 ["IGF1","Liver","Heart","IGF1R","systemic growth signal"],
 ["FGF21","Liver","Pancreas","FGFR1","metabolic/fasting hormone"],
 ["IL6","Immune (blood)","Liver","IL6R","acute-phase response (CRP)"],
 ["ANGPT2","Liver","Heart","TEK","vascular remodeling"],
 ["NPPB","Heart","Kidney","NPR1","natriuretic peptide: fluid balance"],
]
DATA=dict(tissues=model,n_lr=len(LR),downstream=downstream,drugs={k:v for k,v in drugs.items()},
          endocrine=ENDOCRINE)
print("endocrine cross-tissue axes:",len(ENDOCRINE))
json.dump(DATA,open(OUT/"tissue_model.json","w"),separators=(",",":"))
print("wrote tissue_model.json (%d KB)"%(len(json.dumps(DATA))//1024))
