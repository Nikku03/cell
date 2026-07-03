"""PHASE 1 — co-expression network from ARCHS4 (the 8th independent functional lens).

ARCHS4 is ~1M+ uniformly-processed human RNA-seq samples (gene x sample). Genes whose expression
co-varies across this huge, diverse compendium tend to be functionally related — an assay
INDEPENDENT of physical interaction and genetic co-essentiality, so agreement across them is strong
evidence (feeds the convergence engine + dark-gene function).

Method (robust at scale): subsample samples, CPM+log1p per sample, restrict to our genes, rank each
gene across samples (-> Spearman), then block-wise gene x gene correlation keeping only the top-K
partners per gene. Skips gracefully if the h5 is absent (the 59 GB file lives on Colab/Drive).
-> coexpr_neighbors.json  {gene:[[neighbor, r], ...]}

The h5 internal layout varies by ARCHS4 version, so the dataset/gene/orientation are auto-detected.
"""
import json, numpy as np
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
N_SAMPLES=int(__import__("os").environ.get("COEXPR_SAMPLES","80000"))  # subsample size
TOPK=50; MIN_R=0.30; BLOCK=2000

def _find_h5():
    import os
    env=os.environ.get("COEXPR_H5","")
    if env and Path(env).exists(): return Path(env)
    names=["archs4_human_gene.h5","archs4_gene_human.h5","human_gene_v2.h5","archs4_human.h5"]
    # local data dir + the Drive expression_geo folder (read the 59 GB file in place, no copy)
    dirs=[H, Path("/content/drive/MyDrive/virtual_cell_data/expression_geo"),
          Path("/content/drive/MyDrive/virtual_cell_data/human_raw")]
    for d in dirs:
        for n in names:
            p=d/n
            if p.exists() and p.stat().st_size>1e7: return p
        if d.exists():
            hits=sorted(d.glob("archs4*.h5"))+sorted(d.glob("*human*gene*.h5"))
            if hits: return hits[0]
    return None

def _pick(h5, *cands):
    for c in cands:
        if c in h5: return c
    return None

def load_expression(f, our_genes):
    """Return (genes_kept, X[genes x samples]) as float32, restricted to our_genes.
    Reads evenly-spaced CONTIGUOUS sample chunks (FUSE-friendly on a mounted 59 GB h5, representative
    across the whole compendium, bounded memory) rather than scattered fancy-indexing."""
    import h5py
    h5=h5py.File(f,"r")
    dpath=_pick(h5,"data/expression","data/expression/counts","expression","data/counts")
    if dpath is None:
        for k in h5.get("data",{}):
            if getattr(h5["data"][k],"ndim",0)==2: dpath=f"data/{k}"; break
    dset=h5[dpath]
    gpath=_pick(h5,"meta/genes/symbol","meta/genes/gene_symbol","meta/genes/genes","meta/gene/symbol","meta/genes")
    graw=h5[gpath][:]
    genes=np.array([g.decode() if isinstance(g,bytes) else str(g) for g in graw])
    nG=len(genes)
    rows,cols=dset.shape
    genes_axis=0 if rows==nG else (1 if cols==nG else (0 if abs(rows-nG)<abs(cols-nG) else 1))
    n_samples=cols if genes_axis==0 else rows
    gi=np.arange(nG) if not our_genes else np.where(np.array([g in our_genes for g in genes]))[0]
    kept_genes=genes[gi]
    # evenly-spaced contiguous chunks whose total ~= N_SAMPLES
    csize=1000
    n_chunks=max(1, min(n_samples//csize, N_SAMPLES//csize))
    starts=np.linspace(0, max(0,n_samples-csize), n_chunks).astype(int)
    cols_out=[]
    for st in starts:
        en=min(st+csize, n_samples)
        block = dset[:, st:en] if genes_axis==0 else dset[st:en, :].T   # (allGenes x chunk)
        cols_out.append(np.asarray(block[gi], dtype=np.float32))
    X=np.concatenate(cols_out, axis=1)
    h5.close()
    return kept_genes, X

def normalize_rank(X):
    """CPM+log1p per sample, then rank each gene across samples (=> Spearman via Pearson-on-ranks)."""
    col=X.sum(0,keepdims=True); col[col==0]=1.0
    X=np.log1p(X/col*1e6)
    # drop genes with ~no variance
    v=X.var(1); keep=v>1e-8
    X=X[keep]
    # rank per gene across samples
    R=np.empty_like(X)
    for i in range(X.shape[0]):
        R[i]=np.argsort(np.argsort(X[i]))
    R=(R-R.mean(1,keepdims=True))
    R/= (np.linalg.norm(R,axis=1,keepdims=True)+1e-9)
    return R, keep

def topk_neighbors(genes, R, k=TOPK, min_r=MIN_R):
    """Block-wise cosine on rank-normalised rows (=Spearman) -> top-k partners per gene."""
    out={}
    n=len(genes)
    for s in range(0,n,BLOCK):
        C=R[s:s+BLOCK]@R.T            # (block x n) Spearman correlations
        for r in range(C.shape[0]):
            i=s+r; row=C[r]; row[i]=-9
            order=np.argsort(-row)[:k]
            part=[[genes[j],round(float(row[j]),3)] for j in order if row[j]>=min_r]
            if part: out[genes[i]]=part
    return out

def main():
    f=_find_h5()
    if f is None:
        print("ARCHS4 h5 absent -> skipping (the 59 GB file runs on Colab). No coexpr_neighbors.json."); return
    # our gene universe (so we only keep relevant genes) — from the backbone or the assembled model
    our=set()
    cc=OUT/"cell_complete.json"; bb=OUT/"integrated_cell_human.csv"
    if cc.exists():
        our=set(g["name"] for g in json.load(open(cc))["genes"])
    elif bb.exists():
        import csv as _csv
        our=set(r["gene"] for r in _csv.DictReader(open(bb)) if r.get("gene"))
    print(f"ARCHS4: {f.name} | restricting to {len(our) or 'ALL'} model genes | subsample {N_SAMPLES} samples")
    genes,X=load_expression(f, our if our else None)
    R,keep=normalize_rank(X); genes=genes[keep]
    print(f"ARCHS4: {len(genes)} genes x {X.shape[1]} samples after normalization")
    nb=topk_neighbors(genes,R)
    json.dump(nb,open(OUT/"coexpr_neighbors.json","w"))
    print(f"co-expression: top-{TOPK} partners (r>={MIN_R}) for {len(nb)} genes -> coexpr_neighbors.json")

if __name__=="__main__":
    main()
