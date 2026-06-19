"""Faithful port of AlphaFold's Evoformer cross-talk, molded to genes-as-residues.

Mapping (genome = protein, gene = residue, organism = MSA sequence):
  AF MSA [N_seq, N_res, c]   ->  ours [N_org, W_genes, c]   (ortholog essentiality)
  AF OuterProductMean        ->  gene-gene COEVOLUTION across organisms -> pair
  AF pair [N_res,N_res,c]    ->  ours [W,W,c]               (gene-gene relations)
  AF MSARowAttentionWithPairBias -> pair biases which genes mix in the MSA
  AF recycling               ->  iterate MSA->pair->MSA

OuterProductMean is ported verbatim from vendor/alphafold/modules.py
(Jumper et al. 2021, Suppl. Alg. 10), the exact einsums
'acb,ade->dceb' then 'dceb,cef->dbf', normalized by the mask outer product.

Window = [prev_gene, focal_gene, next_gene] (W=3) via synteny; the MSA rows are
all 48 organisms, entry = essentiality(0/1)+known-bit of that window-gene's OG in
that organism. Leakage control: rows of held-out-clade organisms are masked.

Pure NumPy, manual backprop, gradient-checked. Trains a single Evoformer block
with R recycles + a readout head, leave-one-clade-out, comparing recycle depth.

Outputs:
  outputs/orphan/af_evoformer_windows.npz   (cached windows)
  outputs/orphan/af_evoformer_results.json
  outputs/orphan/af_evoformer.png
"""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path("/home/user/cell/outputs/orphan")
DATA = Path("/home/user/cell/data/drive_import")
rng = np.random.default_rng(0)
W = 3                      # window: prev, focal, next
CACHE = OUT/"af_evoformer_windows.npz"

# ---------------- build windowed MSA tensors ----------------
def build():
    labels={}
    for r in csv.DictReader(open(DATA/"labels"/"labels.csv")):
        labels[(r["organism"],r["locus_tag"])]=int(r["essential"])
    cl_of={}
    for r in csv.DictReader(open(DATA/"labels"/"clade_splits.csv")):
        cl_of[r["organism"]]=r.get("clade","?")
    gene_og={}
    for r in csv.DictReader(open(DATA/"labels"/"orthology_features.csv")):
        if r["og_id"]: gene_og[(r["organism"],r["locus_tag"])]=r["og_id"]
    # og -> {org: label}
    og2={}
    for (o,l),y in labels.items():
        og=gene_og.get((o,l))
        if og: og2.setdefault(og,{})[o]=y
    # synteny prev/next OG per gene
    prev_og={}; next_og={}
    for r in csv.DictReader(open(DATA/"labels"/"synteny_features.csv")):
        prev_og[(r["organism"],r["locus_tag"])]=r["prev_og_id"] or None
        next_og[(r["organism"],r["locus_tag"])]=r["next_og_id"] or None
    orgs=sorted({o for (o,_) in labels}); oidx={o:i for i,o in enumerate(orgs)}
    S=len(orgs)
    org_clade=np.array([cl_of.get(o,"?") for o in orgs])
    # sample focal genes from the 5 major clades (tractable)
    MAJOR={"pseudomonas","ralstonia","shewanella","dickeya","burkholderia"}
    focal=[(o,l) for (o,l) in labels if cl_of.get(o,"?") in MAJOR and (o,l) in gene_og]
    rng.shuffle(focal); focal=focal[:60000]
    N=len(focal); print(f"building {N} windows over S={S} organisms")
    msa=np.zeros((N,S,W,2),np.float32); yv=np.zeros(N,np.float32)
    cl=np.empty(N,object)
    for i,(o,l) in enumerate(focal):
        ogs=[prev_og.get((o,l)), gene_og.get((o,l)), next_og.get((o,l))]
        for w,og in enumerate(ogs):
            if og is None or og not in og2: continue
            d=og2[og]
            for org,lab in d.items():
                msa[i,oidx[org],w,0]=lab; msa[i,oidx[org],w,1]=1.0
        yv[i]=labels[(o,l)]; cl[i]=cl_of.get(o,"?")
    np.savez_compressed(CACHE, msa=msa, y=yv, clade=cl.astype(str),
                        org_clade=org_clade)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)")

if "--build" in sys.argv or not CACHE.exists():
    build()
Z=np.load(CACHE, allow_pickle=True)
MSA=Z["msa"]; Y=Z["y"]; CLADE=Z["clade"]; ORGCL=Z["org_clade"]
N,S,Wc,Cin=MSA.shape
print(f"windows {MSA.shape}, ess_rate {Y.mean():.3f}")

# ---------------- Evoformer block (NumPy, faithful ops) ----------------
CM=8; CO=6; CZ=6
def relu(x): return np.maximum(x,0)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def init():
    s=0.3
    return dict(
        emb=rng.normal(0,s,(Cin,CM)).astype(np.float32),          # input proj to c_m
        # OuterProductMean (MSA -> pair)
        opm_l=rng.normal(0,s,(CM,CO)).astype(np.float32),
        opm_r=rng.normal(0,s,(CM,CO)).astype(np.float32),
        opm_w=rng.normal(0,s,(CO,CO,CZ)).astype(np.float32),
        opm_b=np.zeros(CZ,np.float32),
        # pair -> MSA bias (scalar bias per gene-pair)
        pair2bias=rng.normal(0,s,(CZ,)).astype(np.float32),
        # MSA value mix
        msa_v=rng.normal(0,s,(CM,CM)).astype(np.float32),
        # readout from focal (pos 1) consensus + its two pair edges
        rw=rng.normal(0,s,(CM+2*CZ,)).astype(np.float32), rb=np.float32(0.0),
    )

def outer_product_mean(msa, mask, P):
    # msa [S,W,CM], mask [S,W] -> pair [W,W,CZ]  (AF Alg.10, exact einsums)
    m=mask[...,None]
    left = m*(msa@P["opm_l"])          # [S,W,CO]
    right= m*(msa@P["opm_r"])          # [S,W,CO]
    lt = np.transpose(left,(0,2,1))    # [S,CO,W]   (their 'acb')
    act = np.einsum('acb,ade->dceb', lt, right)        # [W,CO,CO,W]
    act = np.einsum('dceb,cef->dbf', act, P["opm_w"]) + P["opm_b"]  # [W,W,CZ]
    norm = np.einsum('abc,adc->bdc', m, m)             # [W,W,1]
    pair = act/(1e-3+norm)
    cache=dict(msa=msa,m=m,left=left,right=right,lt=lt,act_pre=act,norm=norm)
    return pair, cache

def block(msa0, mask, P, recycles=1):
    """one Evoformer block, R recycles: MSA->pair->(bias)->MSA."""
    h = msa0@P["emb"]                      # [S,W,CM]
    caches=[]
    for _ in range(recycles):
        pair,opm_cache = outer_product_mean(h, mask, P)     # [W,W,CZ]
        bias = pair@P["pair2bias"]                          # [W,W] gene-pair bias
        # pair->MSA: each gene position mixes others weighted by softmax(bias)
        a = bias - bias.max(1,keepdims=True)
        a = np.exp(a); a = a/a.sum(1,keepdims=True)         # [W,W] rows sum 1
        v = h@P["msa_v"]                                    # [S,W,CM]
        mixed = np.einsum('ij,sjc->sic', a, v)             # [S,W,CM]
        h = relu(h + mixed)                                 # residual update
        caches.append((opm_cache,pair,bias,a,v,mixed,h))
    return h, pair, caches

def forward(msa0, mask, P, recycles=1):
    h,pair,caches = block(msa0, mask, P, recycles)
    # focal = window pos 1; consensus over organisms (masked mean)
    fm = mask[:,1]                                  # [S]
    cons = (h[:,1,:]*fm[:,None]).sum(0)/(fm.sum()+1e-6)   # [CM]
    edge_prev = pair[1,0,:]; edge_next = pair[1,2,:]      # [CZ] each
    feat = np.concatenate([cons, edge_prev, edge_next])   # [CM+2CZ]
    logit = feat@P["rw"]+P["rb"]
    return logit, dict(h=h,pair=pair,caches=caches,fm=fm,cons=cons,feat=feat,
                       mask=mask,msa0=msa0)

# ------- numeric gradient check (small) -------
def loss_of(P, idx, recycles=1):
    tot=0
    for i in idx:
        lo,_=forward(MSA[i],MSA[i,:,:,1],P,recycles)  # mask = known-bit
        p=sigmoid(lo); y=Y[i]
        tot+=-(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
    return tot/len(idx)

# analytic backward (reverse through readout, recycles, OPM)
def backward(P, cache, dlogit, recycles=1):
    g={k:np.zeros_like(v) for k,v in P.items()}
    feat=cache["feat"]
    g["rw"]+=dlogit*feat; g["rb"]+=dlogit
    dfeat=dlogit*P["rw"]
    dcons=dfeat[:CM]; dprev=dfeat[CM:CM+CZ]; dnext=dfeat[CM+2*CZ-CZ:]
    # cons grad -> h[:,1,:]
    fm=cache["fm"]; h=cache["h"]
    dh=np.zeros_like(h)
    dh[:,1,:]+=np.outer(fm,dcons)/(fm.sum()+1e-6)
    # pair edge grads accumulate onto last pair
    dpair_last=np.zeros_like(cache["pair"])
    dpair_last[1,0,:]+=dprev; dpair_last[1,2,:]+=dnext
    # walk recycles backward
    caches=cache["caches"]; mask=cache["mask"]
    dpair_extra=dpair_last
    for r in range(recycles-1,-1,-1):
        opm_cache,pair,bias,a,v,mixed,h_out=caches[r]
        # h_out = relu(h_in + mixed); dh is grad wrt h_out
        relu_mask=(h_out>0).astype(np.float32)
        d_pre=dh*relu_mask                     # grad wrt (h_in+mixed)
        dh_in=d_pre.copy()                     # to h_in
        dmixed=d_pre.copy()
        # mixed = einsum('ij,sjc->sic', a, v)
        da=np.einsum('sic,sjc->ij', dmixed, v)
        dv=np.einsum('ij,sic->sjc', a, dmixed)
        g["msa_v"]+=np.einsum('swc,swd->cd', opm_cache["msa"] if False else h_in_for(caches,r), dv) \
            if False else _msav_grad(caches,r,dv,P,g)
        dh_from_v=dv@P["msa_v"].T
        dh_in+=dh_from_v
        # da -> bias via softmax
        dbias=a*(da-(a*da).sum(1,keepdims=True))
        dpair = dpair_extra + np.einsum('ij,c->ijc', dbias, P["pair2bias"])
        g["pair2bias"]+=np.einsum('ijc,ij->c', pair, dbias)
        # backprop OuterProductMean
        dh_opm=_opm_backward(opm_cache,dpair,P,g)
        dh = dh_in + dh_opm
        dpair_extra=np.zeros_like(pair)
    # emb
    g["emb"]+=np.einsum('swc,swd->cd', cache["msa0"], dh)
    return g

def _msav_grad(caches,r,dv,P,g):
    h_in = caches[r][6] if False else None
    return 0.0  # handled below via h cache; placeholder

def _opm_backward(c, dpair, P, g):
    # forward: left=m*(msa@opm_l); right=m*(msa@opm_r); lt=T(left)
    # act=einsum('acb,ade->dceb',lt,right); act2=einsum('dceb,cef->dbf',act,opm_w)+b
    # pair=act2/(1e-3+norm)
    m=c["m"]; norm=c["norm"]; left=c["left"]; right=c["right"]; lt=c["lt"]; msa=c["msa"]
    dact2=dpair/(1e-3+norm)
    g["opm_b"]+=dact2.sum((0,1))
    # act2=einsum('dceb,cef->dbf',act,opm_w): recompute act
    act=np.einsum('acb,ade->dceb', lt, right)   # [W,CO,CO,W]
    g["opm_w"]+=np.einsum('dceb,dbf->cef', act, dact2)
    dact=np.einsum('dbf,cef->dceb', dact2, P["opm_w"])   # [W,CO,CO,W]
    # act=einsum('acb,ade->dceb',lt,right)
    dlt=np.einsum('dceb,ade->acb', dact, right)
    dright=np.einsum('dceb,acb->ade', dact, lt)
    dleft=np.transpose(dlt,(0,2,1))
    dleft=dleft*m; dright=dright*m
    g["opm_l"]+=np.einsum('swc,swo->co', msa, dleft)
    g["opm_r"]+=np.einsum('swc,swo->co', msa, dright)
    dmsa=dleft@P["opm_l"].T + dright@P["opm_r"].T
    return dmsa

# fix msa_v grad cleanly: redo backward storing h_in
def forward_train(msa0,mask,P,recycles):
    h=msa0@P["emb"]; caches=[]
    for _ in range(recycles):
        pair,opm_cache=outer_product_mean(h,mask,P)
        bias=pair@P["pair2bias"]
        a=bias-bias.max(1,keepdims=True); a=np.exp(a); a=a/a.sum(1,keepdims=True)
        v=h@P["msa_v"]; mixed=np.einsum('ij,sjc->sic',a,v)
        h_in=h; h=relu(h+mixed)
        caches.append(dict(opm=opm_cache,pair=pair,bias=bias,a=a,v=v,h_in=h_in,h_out=h))
    fm=mask[:,1]; cons=(h[:,1,:]*fm[:,None]).sum(0)/(fm.sum()+1e-6)
    feat=np.concatenate([cons,pair[1,0,:],pair[1,2,:]])
    logit=feat@P["rw"]+P["rb"]
    return logit, dict(caches=caches,fm=fm,cons=cons,feat=feat,pair=pair,h=h,mask=mask,msa0=msa0)

def backward_train(P,cache,dlogit):
    g={k:np.zeros_like(v) for k,v in P.items()}
    feat=cache["feat"]; g["rw"]+=dlogit*feat; g["rb"]+=dlogit
    dfeat=dlogit*P["rw"]
    dcons=dfeat[:CM]; dprev=dfeat[CM:CM+CZ]; dnext=dfeat[CM+CZ:CM+2*CZ]
    fm=cache["fm"]
    dh=np.zeros_like(cache["h"]); dh[:,1,:]+=np.outer(fm,dcons)/(fm.sum()+1e-6)
    dpair_extra=np.zeros_like(cache["pair"])
    dpair_extra[1,0,:]+=dprev; dpair_extra[1,2,:]+=dnext
    for r in range(len(cache["caches"])-1,-1,-1):
        c=cache["caches"][r]
        relu_mask=(c["h_out"]>0).astype(np.float32); d_pre=dh*relu_mask
        dh_in=d_pre.copy(); dmixed=d_pre
        a=c["a"]; v=c["v"]
        da=np.einsum('sic,sjc->ij',dmixed,v); dv=np.einsum('ij,sic->sjc',a,dmixed)
        g["msa_v"]+=np.einsum('swc,swd->cd', c["h_in"], dv)
        dh_in+=dv@P["msa_v"].T
        dbias=a*(da-(a*da).sum(1,keepdims=True))
        g["pair2bias"]+=np.einsum('ijc,ij->c', c["pair"], dbias)
        dpair=dpair_extra+np.einsum('ij,c->ijc',dbias,P["pair2bias"])
        dh_opm=_opm_backward(c["opm"],dpair,P,g)
        dh=dh_in+dh_opm; dpair_extra=np.zeros_like(cache["pair"])
    g["emb"]+=np.einsum('swc,swd->cd', cache["msa0"], dh)
    return g

if "--gradcheck" in sys.argv:
    P=init(); idx=[0,1,2]
    # analytic
    gsum={k:np.zeros_like(v) for k,v in P.items()}
    for i in idx:
        lo,cache=forward_train(MSA[i],MSA[i,:,:,1],P,recycles=2)
        p=sigmoid(lo); g=backward_train(P,cache,(p-Y[i])/len(idx))
        for k in g: gsum[k]+=g[k]
    for kk in ["rw","opm_l","opm_w","pair2bias","msa_v","emb"]:
        flat=P[kk].ravel(); ng=np.zeros_like(flat); eps=1e-4
        for j in range(min(5,flat.size)):
            old=flat[j]; flat[j]=old+eps; lp=loss_of(P,idx,2); flat[j]=old-eps; lm=loss_of(P,idx,2); flat[j]=old
            ng[j]=(lp-lm)/(2*eps)
        print(f"  {kk}: analytic {gsum[kk].ravel()[:3]}  numeric {ng[:3]}")
    sys.exit(0)

# ---------------- train LOCO, compare recycle depth ----------------
def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1; return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
def cov_at_p(s,yy,t,mn=20):
    o=np.argsort(-s); ys=np.asarray(yy)[o].astype(float); k=np.arange(1,len(ys)+1)
    pr=np.cumsum(ys)/k; ok=(pr>=t)&(k>=mn); return float(k[np.where(ok)[0].max()]/len(ys)) if ok.any() else 0.0

def train_eval(recycles, held, epochs=3, bs=512, lr=3e-3):
    tr=np.where(CLADE!=held)[0]; te=np.where(CLADE==held)[0]
    # mask held-clade organism rows
    held_rows=np.where(ORGCL==held)[0]
    msaT=MSA.copy(); msaT[:,held_rows,:,:]=0.0
    P=init(); m={k:np.zeros_like(v) for k,v in P.items()}; v={k:np.zeros_like(v) for k,v in P.items()}; t=0
    pos=Y[tr].mean() or 1e-6
    for ep in range(epochs):
        perm=rng.permutation(tr)
        for i in perm[:20000]:   # cap per epoch for speed
            lo,cache=forward_train(msaT[i],msaT[i,:,:,1],P,recycles)
            p=sigmoid(lo); y=Y[i]; cw=(0.5/pos) if y==1 else (0.5/(1-pos))
            g=backward_train(P,cache,(p-y)*cw); t+=1
            for k in P:
                m[k]=0.9*m[k]+0.1*g[k]; v[k]=0.999*v[k]+0.001*(g[k]**2)
                P[k]-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
    pr=np.array([sigmoid(forward_train(msaT[i],msaT[i,:,:,1],P,recycles)[0]) for i in te])
    return pr, Y[te]

if __name__=="__main__":
    MAJOR=["pseudomonas","ralstonia","shewanella","dickeya"]   # 4 for speed
    res={}
    for R in [1,3]:
        aucs=[]; ne=[]
        for cl in MAJOR:
            pr,yt=train_eval(R, cl)
            a=auc(pr,yt); c=cov_at_p(1-pr,1-yt,0.9)
            aucs.append(a); ne.append(c)
            print(f"recycles={R} {cl:<12} AUC={a:.3f} nonEss_cov@P90={c:.2f}")
        res[f"recycles_{R}"]=dict(mean_auc=round(float(np.nanmean(aucs)),3),
            mean_ne_cov_p90=round(float(np.mean(ne)),3),
            per_clade={cl:round(a,3) for cl,a in zip(MAJOR,aucs)})
        print(f"  -> recycles={R} MEAN AUC {res[f'recycles_{R}']['mean_auc']:.3f} "
              f"nonEss cov@P90 {res[f'recycles_{R}']['mean_ne_cov_p90']:.2f}\n")
    json.dump(res, open(OUT/"af_evoformer_results.json","w"), indent=2)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    Rs=[1,3]; ax.plot(Rs,[res[f"recycles_{R}"]["mean_auc"] for R in Rs],"o-",lw=2,color="#C0392B",label="mean AUC")
    ax.plot(Rs,[res[f"recycles_{R}"]["mean_ne_cov_p90"] for R in Rs],"o-",lw=2,color="#16A085",label="non-ess cov@P90")
    ax.set_xlabel("recycles (MSA<->pair iterations)"); ax.set_xticks(Rs); ax.set_ylim(0,1)
    ax.set_title("Ported AlphaFold Evoformer (OuterProductMean+recycle), genes-as-residues\nleave-one-clade-out")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"af_evoformer.png",dpi=140,bbox_inches="tight")
    print("wrote af_evoformer_results.json + af_evoformer.png")
