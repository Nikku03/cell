"""LINCS L1000 (CMap) -> perturbation functional neighbors, same format as Perturb-seq.

Level-5 signatures: each is the transcriptional response to a perturbation (shRNA knockdown /
CRISPR KO / over-expression) of one gene. We aggregate signatures per perturbed gene, z-score,
correlate perturbation-vs-perturbation, and emit the top functional neighbors -> lincs_neighbors.json
(same shape as perturbseq_neighbors.json, so build_cell_complete merges it into the dark-gene /
Model-4 neighbor layer). Reads a GCTX (HDF5) matrix + siginfo table. Skips if absent (runs on Colab).

First-pass: defensive across GCTX/siginfo column layouts; validate the count on the real files.
-> lincs_neighbors.json
"""
import json, gzip, numpy as np
from pathlib import Path
from collections import defaultdict
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")

def _find(*names):
    for n in names:
        p=H/n
        if p.exists() and p.stat().st_size>1000: return p
    return None

gctx=_find("lincs_level5.gctx","lincs.gctx")
sinfo=_find("lincs_siginfo.txt","lincs_siginfo.txt.gz","siginfo_beta.txt","GSE92742_Broad_LINCS_sig_info.txt.gz")
if not gctx or not sinfo:
    print("LINCS gctx/siginfo absent -> skipping (runs on Colab)"); raise SystemExit(0)
import h5py

# siginfo: sig_id -> (perturbed gene symbol, pert_type)
def _open(p): return gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)
hdr=None; sig2gene={}
KEEP={"trt_sh","trt_xpr","trt_oe","trt_sh.cgs","trt_oe.mut"}   # knockdown / CRISPR / over-expression
with _open(sinfo) as f:
    for line in f:
        p=line.rstrip("\n").split("\t")
        if hdr is None:
            hdr={c:i for i,c in enumerate(p)}; continue
        def col(*names):
            for n in names:
                if n in hdr and hdr[n]<len(p): return p[hdr[n]]
            return ""
        sid=col("sig_id","distil_id"); gene=col("cmap_name","pert_iname","pert_mfc_desc"); pt=col("pert_type")
        if sid and gene and (not pt or pt in KEEP): sig2gene[sid]=gene

# GCTX matrix (HDF5): genes x signatures + row/col ids
with h5py.File(gctx,"r") as f:
    M=f["0/DATA/0/matrix"][:]                     # shape (n_sig, n_gene) in CMap GCTX
    rid=[x.decode() if isinstance(x,bytes) else str(x) for x in f["0/META/ROW/id"][:]]
    cid=[x.decode() if isinstance(x,bytes) else str(x) for x in f["0/META/COL/id"][:]]
# orient so rows = signatures, cols = genes (readout)
if M.shape[0]==len(rid) and M.shape[1]==len(cid): M=M.T; rid,cid=cid,rid   # make rid=signatures
# map readout gene ids (entrez) -> symbol for interpretability is not needed for correlation
# aggregate signatures per perturbed gene
byg=defaultdict(list)
for i,sid in enumerate(rid):
    g=sig2gene.get(sid)
    if g: byg[g].append(i)
genes=sorted(byg)
if len(genes)<5:
    print("LINCS: <5 perturbed genes mapped -> check siginfo columns; skipping"); raise SystemExit(0)
S=np.vstack([M[byg[g]].mean(0) for g in genes]).astype(float)
S=np.nan_to_num(S); S=(S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-6)
C=(S@S.T)/S.shape[1]; np.fill_diagonal(C,-9)
nb={}
for i,g in enumerate(genes):
    order=np.argsort(-C[i])[:10]
    part=[[genes[j],round(float(C[i,j]),3)] for j in order if C[i,j]>0.2]
    if part: nb[g]=part[:8]
json.dump(nb,open(OUT/"lincs_neighbors.json","w"))
print(f"LINCS L1000: signatures for {len(genes)} perturbed genes -> functional neighbors for {len(nb)}")
