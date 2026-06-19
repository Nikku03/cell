"""Two-representation AlphaFold-style model: MSA track + PAIR track, fused.

The single-track model used only the ortholog-MSA. AlphaFold carries TWO
representations that are combined for the output. Here:

  TRACK 1 (MSA / ortholog set)  -- attention over the gene's OG members across
            organisms (as before): "what does this family look like everywhere".
  TRACK 2 (PAIR / genomic context) -- the gene's operon neighbours and
            co-occurrence partners, encoded by whether THOSE families are
            essential across OTHER organisms: "what company does this gene keep".
            This is the residue-pair analog (gene x gene relationships).

Both tracks are concatenated and read by a shared head + a pLDDT-style
confidence head (the brightness). Leakage control is identical and applies to
BOTH tracks: any label used (MSA rows or neighbour-family fractions) from the
held-out clade is masked, computed per fold via per-clade OG count tables.

Evaluated leave-one-CLADE-out on the 5 major clades; reports 1-track vs 2-track.

Outputs:
  outputs/orphan/af_2track_results.json
  outputs/orphan/af_2track.png
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")
rng = np.random.default_rng(0)

Z = np.load(OUT/"af_msa_cache.npz", allow_pickle=True)
foc=Z["foc"]; msa=Z["msa"]; msa_org=Z["msa_org"]; mask=Z["msa_mask"]; y=Z["y"]
clade=Z["clade"]; orgs=list(Z["orgs"]); meta_org=Z["meta_org"]; lt=Z["lt"]
N,K,Dm=msa.shape; Df0=foc.shape[1]
orgname=[orgs[i] for i in meta_org]
keys=list(zip(orgname, [str(x) for x in lt]))
keyidx={k:i for i,k in enumerate(keys)}
print(f"N={N} genes")

# --- per-OG per-clade essentiality count tables (for leak-clean neighbour frac) ---
labels={}
for r in csv.DictReader(open(DATA/"labels"/"labels.csv")):
    labels[(r["organism"],r["locus_tag"])]=int(r["essential"])
cl_of={}
for r in csv.DictReader(open(DATA/"labels"/"clade_splits.csv")):
    cl_of[r["organism"]]=r.get("clade","?")
gene_og={}
for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
    if r["og_id"]: gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
# og -> total (ess,n) and per-clade (ess,n)
og_tot=defaultdict(lambda:[0,0]); og_cl=defaultdict(lambda:defaultdict(lambda:[0,0]))
for (o,l),yy in labels.items():
    og=gene_og.get((o,l));
    if not og: continue
    c=cl_of.get(o,"?")
    og_tot[og][0]+=yy; og_tot[og][1]+=1
    og_cl[og][c][0]+=yy; og_cl[og][c][1]+=1

def og_essfrac(og, held):
    if og is None or og not in og_tot: return (0.0, 0.0)   # frac, known
    e,n=og_tot[og]; he,hn=og_cl[og].get(held,[0,0])
    e-=he; n-=hn
    return (e/n, 1.0) if n>0 else (0.0,0.0)

# --- synteny: prev/next OG per gene; cooccur jaccard ---
prev_og=[None]*N; next_og=[None]*N
for r in csv.DictReader(open(DATA/"labels"/"synteny_features.csv")):
    i=keyidx.get((r["organism"], r["locus_tag"]))
    if i is None: continue
    prev_og[i]=r["prev_og_id"] or None; next_og[i]=r["next_og_id"] or None
jac=np.zeros(N,np.float32)
for r in csv.DictReader(open(DATA/"labels"/"cooccurrence_features.csv")):
    i=keyidx.get((r["organism"], r["locus_tag"]))
    if i is None: continue
    try: jac[i]=float(r["cooccur_max_jaccard"] or 0)
    except: pass
print("pair features loaded (synteny prev/next OG + cooccur jaccard)")

def pair_feats(held):
    """[N,5]: prev_essfrac, prev_known, next_essfrac, next_known, jaccard."""
    pf=np.zeros((N,5),np.float32)
    for i in range(N):
        pe,pk=og_essfrac(prev_og[i], held); ne,nk=og_essfrac(next_og[i], held)
        pf[i]=[pe,pk,ne,nk,jac[i]]
    return pf

# --- model (reuse single-head attention + MLP + confidence) ---
H=12; HID=24
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def init(Df):
    s=0.3
    return dict(Wq=rng.normal(0,s,(Df,H)).astype(np.float32),
        Wk=rng.normal(0,s,(Dm,H)).astype(np.float32),
        Wv=rng.normal(0,s,(Dm,H)).astype(np.float32),
        W1=rng.normal(0,s,(Df+H,HID)).astype(np.float32), b1=np.zeros(HID,np.float32),
        w2=rng.normal(0,s,HID).astype(np.float32), b2=np.float32(0.0),
        wc=rng.normal(0,s,HID).astype(np.float32), bc=np.float32(0.0))
def forward(P,fb,mb,kb):
    q=fb@P["Wq"]; Kx=mb@P["Wk"]; Vx=mb@P["Wv"]
    s=np.einsum("bkh,bh->bk",Kx,q)/np.sqrt(H); s=np.where(kb>0,s,-1e9)
    s=s-s.max(1,keepdims=True); e=np.exp(s)*(kb>0); a=e/(e.sum(1,keepdims=True)+1e-9)
    ctx=np.einsum("bk,bkh->bh",a,Vx); z=np.concatenate([fb,ctx],1)
    h1=z@P["W1"]+P["b1"]; hid=np.maximum(h1,0)
    return hid@P["w2"]+P["b2"], hid@P["wc"]+P["bc"], dict(fb=fb,mb=mb,kb=kb,q=q,Kx=Kx,Vx=Vx,a=a,z=z,h1=h1,hid=hid)
def backward(P,c,dl,dc):
    g={k:np.zeros_like(v) for k,v in P.items()}; hid=c["hid"]; z=c["z"]
    g["w2"]=hid.T@dl; g["b2"]=dl.sum(); g["wc"]=hid.T@dc; g["bc"]=dc.sum()
    dhid=np.outer(dl,P["w2"])+np.outer(dc,P["wc"]); dh1=dhid*(c["h1"]>0)
    g["W1"]=z.T@dh1; g["b1"]=dh1.sum(0); dz=dh1@P["W1"].T
    Df=c["fb"].shape[1]; dfb=dz[:,:Df]; dctx=dz[:,Df:]
    a=c["a"]; Vx=c["Vx"]; Kx=c["Kx"]; q=c["q"]; mb=c["mb"]
    da=np.einsum("bh,bkh->bk",dctx,Vx); dVx=np.einsum("bk,bh->bkh",a,dctx)
    ds=(a*(da-(a*da).sum(1,keepdims=True)))/np.sqrt(H)
    dKx=np.einsum("bk,bh->bkh",ds,q); dq=np.einsum("bk,bkh->bh",ds,Kx)
    g["Wq"]=c["fb"].T@dq; g["Wk"]=np.einsum("bkd,bkh->dh",mb,dKx); g["Wv"]=np.einsum("bkd,bkh->dh",mb,dVx)
    return g
def mask_msa(held_orgs):
    out=msa.copy(); bad=np.isin(msa_org,list(held_orgs))
    out[...,0]=np.where(bad,0,out[...,0]); out[...,1]=np.where(bad,0,out[...,1]); return out
def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
def auprc(s,yy):
    o=np.argsort(-s); ys=np.asarray(yy)[o]; tp=np.cumsum(ys); k=np.arange(1,len(ys)+1)
    return float(np.sum((tp/k)*np.diff(np.concatenate([[0],tp/max(1,ys.sum())]))))
def cov_at_p(s,yy,t,mn=20):
    o=np.argsort(-s); ys=np.asarray(yy)[o].astype(float); k=np.arange(1,len(ys)+1)
    pr=np.cumsum(ys)/k; ok=(pr>=t)&(k>=mn); return float(k[np.where(ok)[0].max()]/len(ys)) if ok.any() else 0.0

def run(fb_all, train_idx, test_idx, held_orgs, epochs=8, bs=2048, lr=3e-3):
    Df=fb_all.shape[1]; P=init(Df)
    m={k:np.zeros_like(v) for k,v in P.items()}; v={k:np.zeros_like(v) for k,v in P.items()}
    msa_m=mask_msa(held_orgs); ytr=y[train_idx]; pos=ytr.mean() or 1e-6; t=0
    for ep in range(epochs):
        perm=rng.permutation(train_idx)
        for i in range(0,len(perm),bs):
            bi=perm[i:i+bs]
            lo,clo,cache=forward(P,fb_all[bi],msa_m[bi],mask[bi]); p=sigmoid(lo); yb=y[bi]
            cw=np.where(yb==1,0.5/pos,0.5/(1-pos)); dl=(p-yb)*cw/len(bi)
            corr=((p>0.5).astype(np.float32)==yb).astype(np.float32); cc=sigmoid(clo); dc=(cc-corr)/len(bi)
            g=backward(P,cache,dl,dc); t+=1
            for kk in P:
                m[kk]=0.9*m[kk]+0.1*g[kk]; v[kk]=0.999*v[kk]+0.001*(g[kk]**2)
                P[kk]-=lr*(m[kk]/(1-0.9**t))/(np.sqrt(v[kk]/(1-0.999**t))+1e-8)
    lo,clo,_=forward(P,fb_all[test_idx],msa_m[test_idx],mask[test_idx])
    return sigmoid(lo), sigmoid(clo)

MAJOR=["pseudomonas","ralstonia","shewanella","dickeya","burkholderia"]
res={"1track":{},"2track":{}}
allp1=np.full(N,np.nan); allp2=np.full(N,np.nan); allc2=np.full(N,np.nan)
for cl in MAJOR:
    test_idx=np.where(clade==cl)[0]
    if len(test_idx)<100: continue
    train_idx=np.where(clade!=cl)[0]
    held=set(int(o) for o in np.unique(meta_org[test_idx]))
    pf=pair_feats(cl)
    foc_aug=np.concatenate([foc,pf],1)
    # 1-track (MSA only)
    p1,_=run(foc,train_idx,test_idx,held)
    # 2-track (MSA + pair)
    p2,c2=run(foc_aug,train_idx,test_idx,held)
    allp1[test_idx]=p1; allp2[test_idx]=p2; allc2[test_idx]=c2
    yt=y[test_idx]
    res["1track"][cl]=dict(auc=round(auc(p1,yt),3),auprc=round(auprc(p1,yt),3),
        ne_cov_p90=round(cov_at_p(1-p1,1-yt,0.9),3),ess_cov_p70=round(cov_at_p(p1,yt,0.7),3))
    res["2track"][cl]=dict(auc=round(auc(p2,yt),3),auprc=round(auprc(p2,yt),3),
        ne_cov_p90=round(cov_at_p(1-p2,1-yt,0.9),3),ess_cov_p70=round(cov_at_p(p2,yt,0.7),3))
    print(f"{cl:<13} 1trk AUC={res['1track'][cl]['auc']:.3f} neP90={res['1track'][cl]['ne_cov_p90']:.2f} "
          f"essP70={res['1track'][cl]['ess_cov_p70']:.2f}  ||  "
          f"2trk AUC={res['2track'][cl]['auc']:.3f} neP90={res['2track'][cl]['ne_cov_p90']:.2f} "
          f"essP70={res['2track'][cl]['ess_cov_p70']:.2f}")

ev=~np.isnan(allp1)
def pool(p):
    return dict(auc=round(auc(p[ev],y[ev]),3),auprc=round(auprc(p[ev],y[ev]),3),
        ne_cov_p90=round(cov_at_p(1-p[ev],1-y[ev],0.9),3),
        ess_cov_p70=round(cov_at_p(p[ev],y[ev],0.7),3),
        ess_cov_p90=round(cov_at_p(p[ev],y[ev],0.9),3))
res["pooled_1track"]=pool(allp1); res["pooled_2track"]=pool(allp2)
print(f"\nPOOLED 1-track: {res['pooled_1track']}")
print(f"POOLED 2-track: {res['pooled_2track']}")

json.dump(res, open(OUT/"af_2track_results.json","w"), indent=2)

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(17,5.5))
cls=[c for c in MAJOR if c in res["1track"]]
x=np.arange(len(cls)); w=0.38
ax[0].bar(x-w/2,[res["1track"][c]["auc"] for c in cls],w,label="1-track (MSA only)",color="#7F8C8D")
ax[0].bar(x+w/2,[res["2track"][c]["auc"] for c in cls],w,label="2-track (MSA+pair)",color="#C0392B")
ax[0].set_xticks(x); ax[0].set_xticklabels(cls,rotation=30,ha="right",fontsize=8); ax[0].set_ylim(0.5,1)
ax[0].set_ylabel("AUC (leave-one-clade-out)"); ax[0].legend(fontsize=8); ax[0].set_title("A. 1-track vs 2-track AUC")
ax[1].bar(x-w/2,[res["1track"][c]["ne_cov_p90"] for c in cls],w,label="1-track",color="#7F8C8D")
ax[1].bar(x+w/2,[res["2track"][c]["ne_cov_p90"] for c in cls],w,label="2-track",color="#16A085")
ax[1].set_xticks(x); ax[1].set_xticklabels(cls,rotation=30,ha="right",fontsize=8); ax[1].set_ylim(0,1)
ax[1].set_ylabel("non-ess coverage @ P>=0.9"); ax[1].legend(fontsize=8); ax[1].set_title("B. Non-essential coverage gain")
ax[2].bar(x-w/2,[res["1track"][c]["ess_cov_p70"] for c in cls],w,label="1-track",color="#7F8C8D")
ax[2].bar(x+w/2,[res["2track"][c]["ess_cov_p70"] for c in cls],w,label="2-track",color="#8E44AD")
ax[2].set_xticks(x); ax[2].set_xticklabels(cls,rotation=30,ha="right",fontsize=8); ax[2].set_ylim(0,0.6)
ax[2].set_ylabel("essential coverage @ P>=0.7"); ax[2].legend(fontsize=8); ax[2].set_title("C. Essential coverage gain")
fig.suptitle("Two-representation model (MSA track + genomic-context PAIR track) vs single-track",
             fontweight="bold",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"af_2track.png",dpi=140,bbox_inches="tight")
print("\nwrote af_2track_results.json + af_2track.png")
