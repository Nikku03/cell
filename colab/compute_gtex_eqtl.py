"""GTEx trans-eQTLs -> candidate cross-gene regulatory edges.

A trans-eQTL is a variant that changes a DISTANT gene's expression. We map the variant to the
nearest gene's TSS (the 'source') and pair it with the affected eGene (the 'target'), giving
source_gene -> eGene genetic-regulation candidates -> gtex_trans_edges.tsv (loaded by
build_cell_complete into the regulatory layer). Reads the GTEx trans_qtl_pairs file + refGene
(TSS) + gene_info (ENSG->symbol). Skips if absent (runs on Colab).

First-pass: association, not proven causal direction; treat as candidate edges. -> gtex_trans_edges.tsv
"""
import gzip, bisect
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
def _find(*names):
    for n in names:
        p=H/n
        if p.exists() and p.stat().st_size>500: return p
    return None
trans=_find("gtex_trans_pairs.txt","gtex_trans_pairs.txt.gz","GTEx_Analysis_v8_trans_eGenes_fdr05.txt","GTEx_Analysis_v8.trans_qtl_pairs.txt.gz")
rg=H/"refGene.txt.gz"; gi=H/"gene_info.gz"
if not trans or not rg.exists() or not gi.exists():
    print("GTEx trans / refGene / gene_info absent -> skipping (runs on Colab)"); raise SystemExit(0)

def _open(p): return gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)
# ENSG -> symbol
ensg2sym={}
with gzip.open(gi,"rt") as f:
    next(f)
    for l in f:
        p=l.split("\t")
        if len(p)>5 and p[0]=="9606" and "Ensembl:ENSG" in p[5]:
            for x in p[5].split("|"):
                if x.startswith("Ensembl:ENSG"): ensg2sym[x.split(":")[1]]=p[2]
# TSS index per chrom
tss=defaultdict(list)
with gzip.open(rg,"rt") as f:
    for l in f:
        p=l.rstrip("\n").split("\t")
        if len(p)<13 or "_" in p[2] or not p[12]: continue
        try: pos=int(p[4]) if p[3]=="+" else int(p[5])
        except: continue
        tss[p[2]].append((pos,p[12]))
for c in tss: tss[c].sort()
posidx={c:[x[0] for x in v] for c,v in tss.items()}
def nearest(chrom,pos):
    arr=posidx.get(chrom)
    if not arr: return None
    k=bisect.bisect_left(arr,pos)
    best=None
    for j in (k-1,k):
        if 0<=j<len(arr):
            d=abs(arr[j]-pos)
            if best is None or d<best[0]: best=(d,tss[chrom][j][1])
    return best[1] if best and best[0]<1_000_000 else None   # within 1 Mb

edges=set(); hdr=None
with _open(trans) as f:
    for l in f:
        p=l.rstrip("\n").split("\t")
        if hdr is None and ("variant_id" in p or "gene_id" in p): hdr={c:i for i,c in enumerate(p)}; continue
        if hdr:
            vid=p[hdr.get("variant_id",0)] if hdr.get("variant_id",0)<len(p) else p[0]
            gid=p[hdr.get("gene_id",1)] if hdr.get("gene_id",1)<len(p) else p[1]
        else:
            vid,gid=(p+["",""])[0],(p+["",""])[1]
        eg=ensg2sym.get(gid.split(".")[0])
        if not eg or not vid: continue
        vp=vid.split("_")
        if len(vp)<2: continue
        chrom=vp[0] if vp[0].startswith("chr") else "chr"+vp[0]
        try: pos=int(vp[1])
        except: continue
        src=nearest(chrom,pos)
        if src and src!=eg: edges.add((src,eg))
with open(OUT/"gtex_trans_edges.tsv","w") as o:
    for a,b in sorted(edges): o.write(f"{a}\t{b}\n")
print(f"GTEx trans-eQTL: {len(edges)} source-gene -> eGene candidate regulatory edges")
