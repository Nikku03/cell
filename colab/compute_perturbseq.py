"""MEASURED dark-gene function from Perturb-seq (Replogle 2022 genome-wide K562 pseudobulk).
Each perturbed gene has a transcriptional signature = what changes when you delete it. Genes whose
signatures correlate share function. -> perturbseq_neighbors.json {gene:[[neighbor,corr],...]}
(same shape as depmap_codep, so the cell uses it as a MEASURED functional-partner layer for dark genes)."""
import json, numpy as np
from pathlib import Path
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
f=H/"perturbseq_gwps_bulk.h5ad"
if not f.exists():
    print("perturbseq h5ad absent -> skipping (runs on Colab where figshare is reachable)"); raise SystemExit(0)
import anndata as ad
A=ad.read_h5ad(f)
obs=A.obs
# perturbation identity: the targeted gene symbol (defensive across column names)
gcol=next((c for c in ["gene","gene_symbol","target_gene","perturbation","perturbed_gene"] if c in obs.columns), None)
perts=(obs[gcol] if gcol else obs.index).astype(str).values
# readout gene names
vcol=next((c for c in ["gene_name","feature_name","symbol"] if c in A.var.columns), None)
vgenes=(A.var[vcol] if vcol else A.var.index).astype(str).values
X=A.X
import scipy.sparse as sp
X=np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
# drop non-targeting / control rows
bad=np.array([any(k in p.lower() for k in ["non-targeting","control","ntc","scramble","safe"]) or p in("","nan") for p in perts])
X=X[~bad]; perts=perts[~bad]
# collapse duplicate perturbations of the same gene (mean)
uniq={}
for i,g in enumerate(perts): uniq.setdefault(g,[]).append(i)
genes=sorted(uniq); M=np.vstack([X[uniq[g]].mean(0) for g in genes])
# z-score each signature, correlate perturbation-vs-perturbation
M=np.nan_to_num(M); M=(M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-6)
n=M.shape[1]; C=(M@M.T)/n
np.fill_diagonal(C,-9)
nb={}
for i,g in enumerate(genes):
    order=np.argsort(-C[i])[:10]
    part=[[genes[j],round(float(C[i,j]),3)] for j in order if C[i,j]>0.15]
    if part: nb[g]=part[:8]
json.dump(nb,open(OUT/"perturbseq_neighbors.json","w"))
print("Perturb-seq: signatures for",len(genes),"perturbed genes | functional neighbors for",len(nb))
