"""GRAPH DIFFUSION: zero-parameter essentiality prediction by label propagation.

The shock test. Everything that worked in this project was a form of borrowing a
measured label from a related gene (ortholog vote, family-rate, conservation).
A transformer is just LEARNED graph diffusion. So: replace the trained
transformer with PARAMETER-FREE label diffusion on the gene graph and see if it
matches/beats it.

Graph (all 6 bacteria's genes as nodes, plus hub nodes):
  gene  --(w=1.0)--  its orthologous-group hub      (the ortholog-vote edge)
  gene  --(w=0.4)--  its protein-family hub          (the family-reasoning edge)
  gene  --(w=0.6)--  its operon neighbour gene       (the 'siblings' edge)

Personalized-PageRank label propagation (no parameters, no training):
  F <- alpha * S @ F + (1-alpha) * Y
  Y = +1 on training-clade essential genes, -1 on non-essential, 0 elsewhere
  S = symmetric-normalized adjacency
  alpha (restart) directly controls effective HOP COUNT: high alpha = many hops.

Leave-one-CLADE-out: held-clade genes are unlabeled (Y=0); read their final F.
Sweep alpha to answer the open question: does multi-hop beat 1-hop?

Compares two graphs (OG-only vs OG+family+operon) against the transformer's
~0.85 AUC / 78% coverage baseline.

Outputs:
  outputs/orphan/graph_diffusion_results.json
  outputs/orphan/graph_diffusion.png
"""
import re, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")
ORGS = ["beril_RalstoniaGMI1000","beril_RalstoniaBSBF1503","beril_RalstoniaPSI07",
        "beril_Dda3937","beril_HerbieS","beril_Magneto"]

# ---- load labels, clade, OG, synteny, family ----
labels={};
for r in csv.DictReader(open(DATA/"labels"/"labels.csv")):
    if r["organism"] in ORGS: labels[(r["organism"],r["locus_tag"])]=int(r["essential"])
clade={}
for r in csv.DictReader(open(DATA/"labels"/"clade_splits.csv")): clade[r["organism"]]=r.get("clade","?")
gene_og={}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["organism"] in ORGS and r["og_id"]: gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
syn_prev={}; syn_next={}
for r in csv.DictReader(open(DATA/"labels"/"synteny_features.csv")):
    if r["organism"] in ORGS:
        k=(r["organism"],r["locus_tag"])
        syn_prev[k]=(r["organism"],r["prev_locus_tag"]) if r.get("prev_locus_tag") else None
        syn_next[k]=(r["organism"],r["next_locus_tag"]) if r.get("next_locus_tag") else None

# family class from GFF products
acc={}
for r in csv.DictReader(open(DATA/"labels"/"genome_cache_manifest.csv")):
    if r.get("accession"): acc[r["organism"]]=r["accession"]
CLASSES=[("ribo",r"ribosom|tRNA|rRNA|translation|elongation factor|aminoacyl"),
 ("transport",r"transport|permease|efflux|ABC|MFS |channel|porin|TonB"),
 ("regulator",r"regulator|sigma factor|repressor|TetR|LysR|response regulator|two-component"),
 ("signaling",r"\bkinase|diguanylate|GGDEF|phosphatase|chemotaxis|sensor histidine"),
 ("secretion",r"secretion system|effector|T3SS|T6SS|pilus|flagell|\bXop|\bAvr|toxin"),
 ("metabolic",r"synthase|synthetase|dehydrogenase|reductase|transferase|hydrolase|oxidase|lyase|isomerase|ligase|carboxylase|oxidoreductase"),
 ("envelope",r"peptidoglycan|murein|cell wall|lipopolysaccharide|outer membrane|penicillin-binding|cell division|Fts"),
 ("dna",r"transposase|integrase|recombinase|helicase|nuclease|DNA polymerase|topoisomerase|phage|gyrase|primase"),
 ("dark",r"hypothetical|DUF\d|uncharacterized|unknown function")]
def classify(p):
    p=p or ""
    for n,pat in CLASSES:
        if re.search(pat,p,re.I): return n
    return "other"
gene_fam={}
for org in ORGS:
    a=acc.get(org);
    if not a: continue
    gff=DATA/"genome_cache"/a/"genomic.gff"
    if not gff.exists(): continue
    for line in gff.read_text().splitlines():
        if "\tCDS\t" not in line: continue
        m=re.search(r"locus_tag=([^;\n]+)",line); pr=re.search(r"product=([^;\n]+)",line)
        if m: gene_fam[(org,m.group(1))]=classify(pr.group(1) if pr else "")

genes=[k for k in labels if k in gene_og]
gidx={g:i for i,g in enumerate(genes)}; NG=len(genes)
y=np.array([labels[g] for g in genes])
cl=np.array([clade[g[0]] for g in genes])
print(f"{NG} gene nodes, {len(set(gene_og[g] for g in genes))} OGs, base ess {y.mean():.3f}")

def build_graph(use_family, use_operon):
    # hub node ids start after gene nodes
    og_ids={}; fam_ids={}; rows=[]; cols=[]; vals=[]
    def hub(d, key, base):
        if key not in d: d[key]=base+len(d)
        return d[key]
    nxt=NG
    og_base=nxt
    # OG edges
    for g,i in gidx.items():
        h=hub(og_ids, gene_og[g], og_base)
        rows+= [i,h]; cols+=[h,i]; vals+=[1.0,1.0]
    nxt=og_base+len(og_ids)
    if use_family:
        fam_base=nxt
        for g,i in gidx.items():
            f=gene_fam.get(g,"other")
            if f in ("dark","other"): continue   # don't propagate through uninformative hubs
            h=hub(fam_ids, f, fam_base)
            rows+=[i,h]; cols+=[h,i]; vals+=[0.4,0.4]
        nxt=fam_base+len(fam_ids)
    if use_operon:
        for g,i in gidx.items():
            for nb in (syn_prev.get(g), syn_next.get(g)):
                if nb and nb in gidx:
                    j=gidx[nb]; rows+=[i,j]; cols+=[j,i]; vals+=[0.6,0.6]
    n=nxt
    A=sp.csr_matrix((vals,(rows,cols)), shape=(n,n))
    d=np.asarray(A.sum(1)).ravel(); d[d==0]=1; dinv=1/np.sqrt(d)
    Dinv=sp.diags(dinv); S=Dinv@A@Dinv
    return S, n

def diffuse(S, n, seed_mask, yv, alpha, iters=60):
    Y=np.zeros(n); Y[:NG]=np.where(seed_mask, np.where(yv==1,1.0,-1.0), 0.0)
    F=Y.copy()
    for _ in range(iters):
        F = alpha*(S@F) + (1-alpha)*Y
    return F[:NG]

def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
def cov_at_p(p, yy, t, mn=50):
    # two-sided: confidence = |p|, call sign; precision among most-confident
    conf=np.abs(p); call=(p>0).astype(int); corr=(call==yy).astype(float)
    o=np.argsort(-conf); cc=corr[o]; k=np.arange(1,len(cc)+1); pr=np.cumsum(cc)/k
    ok=(pr>=t)&(k>=mn); return float(k[np.where(ok)[0].max()]/len(cc)) if ok.any() else 0.0

CONFIGS={"OG_only":(False,False), "OG+family":(True,False), "OG+family+operon":(True,True)}
ALPHAS=[0.50,0.70,0.85,0.90,0.95,0.99]
MAJOR=ORGS  # leave-one-clade-out by clade
clades=sorted(set(cl))
results={}
print(f"\nclades: {clades}\n")
for cfg,(uf,uo) in CONFIGS.items():
    S,n=build_graph(uf,uo)
    results[cfg]={}
    for alpha in ALPHAS:
        aucs=[]; covs=[]
        for held in clades:
            test=cl==held; train=~test
            if test.sum()<50 or y[test].sum()<5: continue
            F=diffuse(S,n,train,y,alpha)
            aucs.append(auc(F[test], y[test]))
            covs.append(cov_at_p(F[test], y[test], 0.90))
        results[cfg][alpha]=dict(auc=round(float(np.nanmean(aucs)),4),
                                 cov90=round(float(np.mean(covs)),4))
        print(f"  {cfg:<18} alpha={alpha:.2f}  AUC={results[cfg][alpha]['auc']:.4f}  cov@P90={results[cfg][alpha]['cov90']:.3f}")
    print()

# best
best=max(((c,a,results[c][a]['auc']) for c in results for a in results[c]), key=lambda x:x[2])
print(f"BEST: {best[0]} alpha={best[1]} -> AUC {best[2]:.4f}  (transformer baseline 0.85)")
bc=results[best[0]][best[1]]
print(f"      coverage@P90 = {bc['cov90']:.3f}  (transformer baseline 0.78)")

json.dump({"n_genes":NG,"configs":results,
           "best":{"config":best[0],"alpha":best[1],"auc":best[2],"cov90":bc['cov90']},
           "transformer_baseline":{"auc":0.85,"cov90":0.78}},
          open(OUT/"graph_diffusion_results.json","w"), indent=2)

# figure: AUC vs alpha (hop count) per config
fig,ax=plt.subplots(1,2,figsize=(15,5.5))
for cfg in CONFIGS:
    ax[0].plot(ALPHAS,[results[cfg][a]['auc'] for a in ALPHAS],"o-",lw=2,label=cfg)
    ax[1].plot(ALPHAS,[results[cfg][a]['cov90'] for a in ALPHAS],"o-",lw=2,label=cfg)
ax[0].axhline(0.85,color="k",ls="--",lw=1,label="trained transformer (0.85)")
ax[0].set_xlabel("alpha (restart -> more hops ->)"); ax[0].set_ylabel("AUC (leave-one-clade-out)")
ax[0].set_title("Zero-parameter diffusion vs trained transformer"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].axhline(0.78,color="k",ls="--",lw=1,label="transformer coverage (0.78)")
ax[1].set_xlabel("alpha (more hops ->)"); ax[1].set_ylabel("coverage @ P>=0.90")
ax[1].set_title("Coverage vs hop count"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.suptitle("GRAPH DIFFUSION: parameter-free label propagation for essentiality",fontweight="bold",fontsize=13)
fig.tight_layout(); fig.savefig(OUT/"graph_diffusion.png",dpi=140,bbox_inches="tight")
print("\nwrote graph_diffusion_results.json + graph_diffusion.png")
