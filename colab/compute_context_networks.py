"""CONTEXT NETWORKS — data-driven per-condition co-expression from ARCHS4.

The condition ontology (compute_conditions.py) is curated. This is the DATA-DRIVEN version: split ARCHS4's
~1M samples by condition (keyword-mined from sample metadata) and compute co-expression WITHIN each
condition, so the model shows how the network rewires under cancer / hypoxia / inflammation / etc. — and
which links (and dark genes) are condition-specific. -> context_networks.json {condition:{n,neighbors}}

Reuses the ARCHS4 reader; runs on Colab where the 59 GB h5 lives. Skips gracefully if absent.
"""
import json, re, os, numpy as np
from collections import defaultdict
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
TOPK=25; MIN_R=0.35; MIN_SAMPLES=300; MAX_PER=8000

# condition -> regex over free-text sample metadata
CONDITIONS = {
 "cancer/tumor":   r"tumor|tumour|carcinoma|cancer|malignan|metasta|adenocarc|glioma|leukemia|lymphoma",
 "hypoxia":        r"hypoxia|hypoxic|low oxygen|anoxi",
 "inflammation":   r"\blps\b|inflammat|cytokine|tnf|interferon|il-?1|il-?6|sepsis",
 "infection/virus":r"infect|virus|viral|sars|influenza|hiv|bacteri",
 "stem/pluripotent":r"stem cell|ipsc|pluripoten|embryonic| esc ",
 "drug-treated":   r"treated|treatment|drug|inhibitor|compound|dose|\bµm\b|\bum\b",
 "heat/stress":    r"heat shock|heat stress|thermal|hyperthermi",
 "healthy/control":r"healthy|normal|control|untreated|wild.?type|baseline",
}

def _find_h5():
    env=os.environ.get("COEXPR_H5","")
    if env and Path(env).exists(): return Path(env)
    for d in [H, Path("/content/drive/MyDrive/virtual_cell_data/expression_geo")]:
        for n in ["archs4_human_gene.h5","archs4_gene_human.h5"]:
            p=d/n
            if p.exists() and p.stat().st_size>1e7: return p
        if d.exists():
            hits=sorted(d.glob("archs4*.h5"))
            if hits: return hits[0]
    return None

def _pick(h5,*c):
    for x in c:
        if x in h5: return x
    return None

def coexpr_for(dset, genes_axis, gi, cols):
    """co-expression (top-k Spearman) for model genes over the given sample columns, chunked."""
    csz=1500; parts=[]
    cols=np.sort(cols)
    for s in range(0,len(cols),csz):
        cc=cols[s:s+csz]; lo,hi=cc.min(),cc.max()+1
        block = dset[:,lo:hi] if genes_axis==0 else dset[lo:hi,:].T
        parts.append(np.asarray(block[np.ix_(gi, cc-lo)],dtype=np.float32))
    X=np.concatenate(parts,axis=1)
    col=X.sum(0,keepdims=True); col[col==0]=1; X=np.log1p(X/col*1e6)
    v=X.var(1); keep=v>1e-8
    R=np.empty((keep.sum(),X.shape[1]),np.float32)
    for k,i in enumerate(np.where(keep)[0]): R[k]=np.argsort(np.argsort(X[i]))
    R=R-R.mean(1,keepdims=True); R/=(np.linalg.norm(R,axis=1,keepdims=True)+1e-9)
    return R, keep

def main():
    f=_find_h5()
    if f is None:
        print("ARCHS4 absent -> skipping context networks (Colab only)"); return
    our=set()
    cc=OUT/"cell_complete.json"
    if cc.exists(): our=set(g["name"] for g in json.load(open(cc))["genes"])
    import h5py
    h5=h5py.File(f,"r")
    dpath=_pick(h5,"data/expression","expression"); dset=h5[dpath]
    gpath=_pick(h5,"meta/genes/symbol","meta/genes/gene_symbol","meta/genes/genes")
    genes=np.array([g.decode() if isinstance(g,bytes) else str(g) for g in h5[gpath][:]])
    nG=len(genes); rows,cols=dset.shape
    genes_axis=0 if rows==nG else 1; n=cols if genes_axis==0 else rows
    gi=np.arange(nG) if not our else np.where(np.array([g in our for g in genes]))[0]
    kept_genes=genes[gi]
    # metadata free-text
    meta=[]
    for mp in ["meta/samples/characteristics_ch1","meta/samples/source_name_ch1","meta/samples/title"]:
        p=_pick(h5,mp)
        if p: meta.append(np.array([str(x) for x in h5[p][:]]))
    if not meta: print("no ARCHS4 sample metadata -> skipping"); h5.close(); return
    txt=[" ".join(m[i] for m in meta).lower() for i in range(n)]
    out={}
    for cond,rgx in CONDITIONS.items():
        pat=re.compile(rgx)
        idxs=np.array([i for i in range(n) if pat.search(txt[i])])
        if len(idxs)<MIN_SAMPLES: continue
        if len(idxs)>MAX_PER: idxs=np.random.default_rng(0).choice(idxs,MAX_PER,replace=False)
        R,keep=coexpr_for(dset,genes_axis,gi,idxs); cg=kept_genes[keep]
        nb={}
        for s in range(0,len(cg),2000):
            C=R[s:s+2000]@R.T
            for r in range(C.shape[0]):
                i=s+r; row=C[r]; row[i]=-9; order=np.argsort(-row)[:TOPK]
                part=[[cg[j],round(float(row[j]),3)] for j in order if row[j]>=MIN_R]
                if part: nb[cg[i]]=part
        out[cond]=dict(n_samples=int(len(idxs)), n_genes=len(nb), neighbors={k:v for k,v in list(nb.items())})
        print(f"  {cond:20} {len(idxs)} samples -> co-expression for {len(nb)} genes")
    h5.close()
    json.dump(out, open(OUT/"context_networks.json","w"))
    print(f"context networks: {len(out)} condition-specific co-expression networks -> context_networks.json")

if __name__=="__main__":
    main()
