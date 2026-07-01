"""Human essentiality FUSION: population constraint (gnomAD) + W1 (cross-species
conservation via BLAST to yeast+E.coli, and ESM-2 sequence embedding). 5-fold
cross-validated, with ablations so we see honestly what each layer adds. Truth: Hart
CEG/NEG. Target: clear the LOEUF-only 0.864.
"""
import gzip, csv, json, subprocess, warnings, time
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
SC=Path("/tmp/claude-0/-home-user-cell/558875a8-e44f-59af-a869-11202bf854d3/scratchpad")
np.random.seed(0)

def readset(fn):
    s=set()
    for i,l in enumerate(open(H/fn)):
        if i>0 and l.split("\t")[0].strip().strip('"'): s.add(l.split("\t")[0].strip().strip('"'))
    return s
CEG=readset("CEGv2.txt"); NEG=readset("NEGv1.txt")
TRUTH={**{g:1 for g in CEG},**{g:0 for g in NEG}}

# ---- gnomAD constraint (multi-metric) ----
def fv(x):
    try: return float(x)
    except: return np.nan
con={}
with gzip.open(H/"gnomad_constraint.txt.bgz","rt") as f:
    h=f.readline().rstrip("\n").split("\t"); I={c:h.index(c) for c in
      ["gene","pLI","oe_lof","oe_lof_upper","mis_z","lof_z","syn_z"]}
    for l in f:
        p=l.rstrip("\n").split("\t"); g=p[I["gene"]]
        rec=dict(pLI=fv(p[I["pLI"]]),loeuf=fv(p[I["oe_lof_upper"]]),oe_lof=fv(p[I["oe_lof"]]),
                 mis_z=fv(p[I["mis_z"]]),lof_z=fv(p[I["lof_z"]]),syn_z=fv(p[I["syn_z"]]))
        if g not in con or (not np.isnan(rec["loeuf"]) and (np.isnan(con[g]["loeuf"]) or rec["loeuf"]<con[g]["loeuf"])):
            con[g]=rec

# ---- human protein sequences per gene symbol (longest isoform) ----
def parse_faa(path):
    seqs={}; sym=None; buf=[]
    def flush():
        if sym and buf:
            s="".join(buf)
            if sym not in seqs or len(s)>len(seqs[sym]): seqs[sym]=s
    for l in open(path):
        if l.startswith(">"):
            flush(); buf=[]; sym=None
            for tok in l.split():
                if tok.startswith("GN="): sym=tok[3:]
        else: buf.append(l.strip())
    flush(); return seqs
hsym=parse_faa(SC/"human.faa")
genes=[g for g in TRUTH if g in hsym and g in con]   # need seq + constraint + label
print(f"genes usable (truth + seq + constraint): {len(genes)} "
      f"(ess {sum(TRUTH[g] for g in genes)}, non {sum(1-TRUTH[g] for g in genes)})")
# subset FASTA
sub=SC/"human_truth.faa"
with open(sub,"w") as f:
    for g in genes: f.write(f">{g}\n{hsym[g][:1000]}\n")

# ---- cross-species conservation via BLAST (yeast=eukaryote, E.coli=deep) ----
def fetch_yeast():
    yf=SC/"yeast.faa"
    if not yf.exists() or yf.stat().st_size<500000:
        subprocess.run(["curl","-sS","--max-time","120","-o",str(yf),
          "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:559292+AND+reviewed:true&format=fasta"],check=True)
    return yf
def blast_bits(query,dbfaa,tag):
    db=SC/f"{tag}db"
    subprocess.run(["makeblastdb","-in",str(dbfaa),"-dbtype","prot","-out",str(db)],check=True,capture_output=True)
    outf=SC/f"human_vs_{tag}.tsv"
    subprocess.run(["blastp","-query",str(query),"-db",str(db),"-out",str(outf),
      "-outfmt","6 qseqid sseqid pident bitscore evalue","-evalue","1e-3","-max_target_seqs","1","-num_threads","4"],check=True)
    best={}
    for l in open(outf):
        q,s,pid,bs,ev=l.rstrip("\n").split("\t"); bs=float(bs)
        if q not in best or bs>best[q]: best[q]=bs
    return best
print("blast vs yeast (eukaryote conservation)...",flush=True); t0=time.time()
yeast=fetch_yeast(); ybits=blast_bits(sub,yeast,"yeast")
print(f"  yeast hits {len(ybits)} ({time.time()-t0:.0f}s)")
print("blast vs E. coli (deep conservation)...",flush=True)
ebits=blast_bits(sub,SC/"ecoli_prot.faa","ecoli"); print(f"  ecoli hits {len(ebits)}")

# ---- ESM-2 embedding (W1 sequence) ----
print("ESM-2 8M embedding of truth proteins ...",flush=True); t0=time.time()
import torch
from transformers import AutoTokenizer, AutoModel
tok=AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").eval(); torch.set_num_threads(8)
@torch.no_grad()
def embed(ss):
    x=tok(ss,return_tensors="pt",padding=True,truncation=True,max_length=402)
    o=esm(**x).last_hidden_state; mm=x["attention_mask"].unsqueeze(-1).float()
    return ((o*mm).sum(1)/mm.sum(1).clamp(min=1)).numpy().astype(np.float32)
E={}; B=16
for i in range(0,len(genes),B):
    gs=genes[i:i+B]; em=embed([hsym[g][:400] for g in gs])
    for g,v in zip(gs,em): E[g]=v
    if (i//B)%15==0: print(f"  {i}/{len(genes)} ({time.time()-t0:.0f}s)",flush=True)
ESMD=len(next(iter(E.values()))); print(f"  ESM dim {ESMD} ({time.time()-t0:.0f}s)")

# ---- feature blocks ----
def cons_feats(g):
    return [np.log1p(ybits.get(g,0.0)),1.0 if g in ybits else 0.0,
            np.log1p(ebits.get(g,0.0)),1.0 if g in ebits else 0.0]
def con_feats(g):
    c=con[g]; return [c["loeuf"],c["mis_z"],c["pLI"],c["oe_lof"],c["lof_z"],c["syn_z"]]
y=np.array([TRUTH[g] for g in genes])
BLOCKS={
 "constraint":  np.array([con_feats(g) for g in genes],float),
 "conservation":np.array([cons_feats(g) for g in genes],float),
 "esm":         np.array([E[g] for g in genes],float),
}
for k in BLOCKS: BLOCKS[k]=np.nan_to_num(BLOCKS[k],nan=np.nanmedian(BLOCKS[k]) if np.isnan(BLOCKS[k]).any() else 0.0)

# ---- 5-fold CV logistic regression (L2), OOF AUC ----
def auc(p,yy):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=yy==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
def logit_fit(X,yy,l2,it=300,lr=0.1):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd; Xn=np.hstack([Xn,np.ones((len(Xn),1))])
    w=np.zeros(Xn.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xn@w)); g=Xn.T@(p-yy)/len(yy)+l2*np.r_[w[:-1],0]/len(yy); w-=lr*g
    return mu,sd,w
def logit_pred(m,X):
    mu,sd,w=m; Xn=(X-mu)/sd; Xn=np.hstack([Xn,np.ones((len(Xn),1))]); return 1/(1+np.exp(-Xn@w))
def oof_auc(cols,l2):
    X=np.hstack([BLOCKS[c] for c in cols]); n=len(y)
    idx=np.random.permutation(n); folds=np.array_split(idx,5); oof=np.zeros(n)
    for k in range(5):
        te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
        m=logit_fit(X[tr],y[tr],l2); oof[te]=logit_pred(m,X[te])
    return auc(oof,y)
print("\n=== 5-fold OOF AUC (human essentiality) ===")
res={}
res["LOEUF alone (ref)"]=auc(-BLOCKS["constraint"][:,0],y)
res["constraint (6 metrics)"]=oof_auc(["constraint"],1.0)
res["conservation (yeast+ecoli)"]=oof_auc(["conservation"],1.0)
res["ESM-2 8M alone"]=oof_auc(["esm"],5.0)
res["constraint+conservation"]=oof_auc(["constraint","conservation"],1.0)
res["constraint+conservation+ESM (FUSION)"]=oof_auc(["constraint","conservation","esm"],3.0)
for k,v in res.items(): print(f"  {k:42s}: {v:.3f}")
best=max(res.values())
json.dump(dict(n_genes=len(genes),n_ess=int(y.sum()),results={k:round(v,3) for k,v in res.items()},
    esm_dim=ESMD),open(OUT/"human_fusion.json","w"),indent=2)
print(f"\nbest AUC {best:.3f}  (LOEUF-only baseline {res['LOEUF alone (ref)']:.3f})")
print("wrote human_fusion.json")
