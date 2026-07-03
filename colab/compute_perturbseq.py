"""MEASURED dark-gene function from Perturb-seq (Replogle 2022 genome-wide K562 pseudobulk).
Each perturbed gene has a transcriptional signature = what changes when you delete it. Genes whose
signatures correlate share function. -> perturbseq_neighbors.json {gene:[[neighbor,corr],...]}
(same shape as depmap_codep, so the cell uses it as a MEASURED functional-partner layer for dark genes)."""
import json, numpy as np
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
# every Perturb-seq dataset present (K562 genome-wide, RPE1, Norman combinatorial, ...)
files=[f for f in sorted(H.glob("perturbseq_*.h5ad")) if f.stat().st_size>1_000_000]  # ignore empty/truncated
if not files:
    print("no valid perturbseq_*.h5ad present -> skipping (runs on Colab where figshare is reachable)"); raise SystemExit(0)
import anndata as ad, scipy.sparse as sp

def neighbors_from(f):
    """One dataset -> {gene:[[neighbor,corr],...]} from perturbation-response similarity."""
    A=ad.read_h5ad(f); obs=A.obs
    gcol=next((c for c in ["gene","gene_symbol","target_gene","perturbation","perturbed_gene"] if c in obs.columns), None)
    perts=(obs[gcol] if gcol else obs.index).astype(str).values
    X=A.X; X=np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    bad=np.array([any(k in p.lower() for k in ["non-targeting","control","ntc","scramble","safe"]) or p in("","nan") for p in perts])
    X=X[~bad]; perts=perts[~bad]
    uniq={}
    for i,g in enumerate(perts): uniq.setdefault(g,[]).append(i)
    genes=sorted(uniq); M=np.vstack([X[uniq[g]].mean(0) for g in genes])
    M=np.nan_to_num(M); M=(M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-6)
    C=(M@M.T)/M.shape[1]; np.fill_diagonal(C,-9)
    out={}
    for i,g in enumerate(genes):
        order=np.argsort(-C[i])[:10]
        part=[[genes[j],round(float(C[i,j]),3)] for j in order if C[i,j]>0.15]
        if part: out[g]=part[:8]
    print(f"  {f.name}: {len(genes)} perturbed genes -> neighbors for {len(out)}")
    return out

# merge neighbor lists across datasets (keep the strongest correlation per neighbor)
merged={}
for f in files:
    try:
        parts_by_gene=neighbors_from(f)
    except Exception as e:
        print(f"  !! {f.name} unreadable ({repr(e)[:80]}) - skipping this file"); continue
    for g,parts in parts_by_gene.items():
        d=merged.setdefault(g,{})
        for nb,r in parts:
            if r>d.get(nb,-9): d[nb]=r
if not merged:
    print("no readable Perturb-seq data -> skipping"); raise SystemExit(0)
nb={g:sorted(([n,r] for n,r in d.items()),key=lambda x:-x[1])[:8] for g,d in merged.items()}
json.dump(nb,open(OUT/"perturbseq_neighbors.json","w"))
print(f"Perturb-seq ({len(files)} dataset(s)): merged functional neighbors for {len(nb)} genes")
