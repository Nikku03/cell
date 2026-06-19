"""AlphaFold-style essentiality model: attention over the OG-MSA + confidence head.

Architecture (single-head MSA row-attention -> prediction + pLDDT-like confidence):
  q   = Wq . focal                              (h,)
  Kx  = msa @ Wk ; Vx = msa @ Wv                (K,h)
  s   = (Kx . q)/sqrt(h) , masked , softmax     (K,)
  ctx = sum_k s_k Vx_k                          (h,)
  z   = [focal ; ctx]
  hid = relu(z @ W1 + b1)
  p_ess  = sigmoid(hid @ w2 + b2)               essentiality
  conf   = sigmoid(hid @ wc + bc)               predicted correctness (brightness)

Leakage control: per held-out clade, ortholog rows whose organism is in that
clade have their label + label_known zeroed -> the model cannot peek at held-out
labels and must generalize the 'conservation' concept.

Evaluation: leave-one-CLADE-out (the project's strict gold standard) for the
major clades. Reports AUC/AUPRC + coverage@precision for BOTH classes, plus
confidence calibration (does brightness predict correctness?).

Outputs:
  outputs/orphan/af_model_results.json
  outputs/orphan/af_model.png
"""
import json
from pathlib import Path
import numpy as np

OUT = Path("/home/user/cell/outputs/orphan")
rng = np.random.default_rng(0)
Z = np.load(OUT/"af_msa_cache.npz", allow_pickle=True)
foc=Z["foc"]; msa=Z["msa"]; msa_org=Z["msa_org"]; mask=Z["msa_mask"]; y=Z["y"]
clade=Z["clade"]; orgs=Z["orgs"]
org_clade = None
# map org index -> clade string via meta (rebuild from any gene): use msa_org cross ref
# we have clade per gene (focal). Build org->clade from focal meta:
meta_org=Z["meta_org"]
o2c={}
for oi,c in zip(meta_org, clade):
    o2c[int(oi)]=c
NUM_ORG=len(orgs)
org_clade_arr=np.array([o2c.get(i,"?") for i in range(NUM_ORG)])
N,K,Dm=msa.shape; Df=foc.shape[1]
print(f"N={N} genes, K={K}, Df={Df}, Dm={Dm}, ess_rate={y.mean():.3f}")

H=12; HID=24
def init():
    s=0.3
    return dict(
        Wq=rng.normal(0,s,(Df,H)).astype(np.float32),
        Wk=rng.normal(0,s,(Dm,H)).astype(np.float32),
        Wv=rng.normal(0,s,(Dm,H)).astype(np.float32),
        W1=rng.normal(0,s,(Df+H,HID)).astype(np.float32), b1=np.zeros(HID,np.float32),
        w2=rng.normal(0,s,HID).astype(np.float32), b2=np.float32(0.0),
        wc=rng.normal(0,s,HID).astype(np.float32), bc=np.float32(0.0),
    )

def forward(P, fb, mb, kb):
    # fb[B,Df], mb[B,K,Dm], kb[B,K] mask
    q = fb @ P["Wq"]                      # B,H
    Kx = mb @ P["Wk"]                     # B,K,H
    Vx = mb @ P["Wv"]                     # B,K,H
    s = np.einsum("bkh,bh->bk", Kx, q)/np.sqrt(H)   # B,K
    s = np.where(kb>0, s, -1e9)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)*(kb>0); a = e/(e.sum(1,keepdims=True)+1e-9)   # B,K
    ctx = np.einsum("bk,bkh->bh", a, Vx)            # B,H
    z = np.concatenate([fb, ctx],1)                 # B,Df+H
    h1 = z @ P["W1"] + P["b1"]; hid=np.maximum(h1,0)
    logit = hid @ P["w2"] + P["b2"]
    clogit= hid @ P["wc"] + P["bc"]
    cache=dict(fb=fb,mb=mb,kb=kb,q=q,Kx=Kx,Vx=Vx,a=a,ctx=ctx,z=z,h1=h1,hid=hid)
    return logit, clogit, cache

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def backward(P, cache, dlogit, dclogit):
    g={k:np.zeros_like(v) for k,v in P.items()}
    hid=cache["hid"]; z=cache["z"]
    g["w2"]=hid.T@dlogit; g["b2"]=dlogit.sum()
    g["wc"]=hid.T@dclogit; g["bc"]=dclogit.sum()
    dhid=np.outer(dlogit,P["w2"])+np.outer(dclogit,P["wc"])
    dh1=dhid*(cache["h1"]>0)
    g["W1"]=z.T@dh1; g["b1"]=dh1.sum(0)
    dz=dh1@P["W1"].T
    dfb=dz[:,:cache["fb"].shape[1]]; dctx=dz[:,cache["fb"].shape[1]:]
    a=cache["a"]; Vx=cache["Vx"]; Kx=cache["Kx"]; q=cache["q"]; mb=cache["mb"]; kb=cache["kb"]
    # ctx = sum_k a_k Vx_k
    da=np.einsum("bh,bkh->bk", dctx, Vx)
    dVx=np.einsum("bk,bh->bkh", a, dctx)
    # softmax backward
    ds = a*(da - (a*da).sum(1,keepdims=True))
    ds=ds/np.sqrt(H)
    dKx=np.einsum("bk,bh->bkh", ds, q)
    dq=np.einsum("bk,bkh->bh", ds, Kx)
    g["Wq"]=cache["fb"].T@dq; dfb=dfb+dq@P["Wq"].T
    g["Wk"]=np.einsum("bkd,bkh->dh", mb, dKx)
    g["Wv"]=np.einsum("bkd,bkh->dh", mb, dVx)
    return g

def mask_heldout(mb_org_block, mb_block, held_org_set):
    """zero label(col0)+known(col1) of ortholog rows from held-out orgs."""
    out=mb_block.copy()
    bad=np.isin(mb_org_block, list(held_org_set))
    out[...,0]=np.where(bad,0.0,out[...,0])
    out[...,1]=np.where(bad,0.0,out[...,1])
    return out

def auc(s,yy):
    yy=np.asarray(yy); n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float("nan")
    r=np.argsort(np.argsort(s))+1
    return float((r[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
def auprc(s,yy):
    o=np.argsort(-s); ys=np.asarray(yy)[o]; tp=np.cumsum(ys); k=np.arange(1,len(ys)+1)
    return float(np.sum((tp/k)*np.diff(np.concatenate([[0],tp/max(1,ys.sum())]))))
def cov_at_p(s,yy,t,mn=20):
    o=np.argsort(-s); ys=np.asarray(yy)[o].astype(float); k=np.arange(1,len(ys)+1)
    pr=np.cumsum(ys)/k; ok=(pr>=t)&(k>=mn)
    return float(k[np.where(ok)[0].max()]/len(ys)) if ok.any() else 0.0

def train_eval(train_idx, test_idx, held_org_set, epochs=8, bs=2048, lr=3e-3):
    P=init()
    m={k:np.zeros_like(v) for k,v in P.items()}; v={k:np.zeros_like(v) for k,v in P.items()}
    # precompute masked msa for train (mask held-out labels everywhere)
    msa_tr=mask_heldout(msa_org, msa, held_org_set)
    ytr=y[train_idx]; pos=ytr.mean() or 1e-6
    t=0
    for ep in range(epochs):
        perm=rng.permutation(train_idx)
        for i in range(0,len(perm),bs):
            bi=perm[i:i+bs]
            logit,clogit,cache=forward(P, foc[bi], msa_tr[bi], mask[bi])
            p=sigmoid(logit); yb=y[bi]
            cw=np.where(yb==1,0.5/pos,0.5/(1-pos))
            dlogit=(p-yb)*cw/len(bi)
            # confidence target: was the (rounded) prediction correct?
            correct=((p>0.5).astype(np.float32)==yb).astype(np.float32)
            c=sigmoid(clogit); dclogit=(c-correct)/len(bi)
            g=backward(P,cache,dlogit,dclogit)
            t+=1
            for kk in P:
                m[kk]=0.9*m[kk]+0.1*g[kk]; v[kk]=0.999*v[kk]+0.001*(g[kk]**2)
                mh=m[kk]/(1-0.9**t); vh=v[kk]/(1-0.999**t)
                P[kk]=P[kk]-lr*mh/(np.sqrt(vh)+1e-8)
    # eval on test (held labels masked in their MSA too)
    msa_te=mask_heldout(msa_org, msa, held_org_set)
    logit,clogit,_=forward(P, foc[test_idx], msa_te[test_idx], mask[test_idx])
    return sigmoid(logit), sigmoid(clogit)

if __name__=="__main__":
    import sys
    if "--gradcheck" in sys.argv:
        P=init(); bi=np.arange(8)
        msa_tr=msa.copy()
        def loss(P):
            lo,cl,_=forward(P,foc[bi],msa_tr[bi],mask[bi])
            p=sigmoid(lo); yb=y[bi]
            return -np.mean(yb*np.log(p+1e-9)+(1-yb)*np.log(1-p+1e-9))
        lo,cl,cache=forward(P,foc[bi],msa_tr[bi],mask[bi])
        p=sigmoid(lo); dlogit=(p-y[bi])/len(bi)
        g=backward(P,cache,dlogit,np.zeros_like(cl))
        for kk in ["Wq","Wk","Wv","W1","w2"]:
            num=np.zeros_like(P[kk]); eps=1e-4
            flat=P[kk].ravel(); ng=num.ravel()
            for j in range(min(8,flat.size)):
                old=flat[j]; flat[j]=old+eps; lp=loss(P); flat[j]=old-eps; lm=loss(P); flat[j]=old
                ng[j]=(lp-lm)/(2*eps)
            print(f"  {kk}: analytic {g[kk].ravel()[:3]} numeric {ng[:3]}")
        sys.exit(0)

    # leave-one-CLADE-out for major clades
    MAJOR=["pseudomonas","ralstonia","shewanella","dickeya","burkholderia"]
    results={}; all_p=np.full(N,np.nan); all_c=np.full(N,np.nan)
    for cl in MAJOR:
        test_idx=np.where(clade==cl)[0]
        if len(test_idx)<100: continue
        train_idx=np.where(clade!=cl)[0]
        held_orgs=set(int(o) for o in np.unique(meta_org[test_idx]))
        p,c=train_eval(train_idx,test_idx,held_orgs)
        all_p[test_idx]=p; all_c[test_idx]=c
        yt=y[test_idx]
        results[cl]=dict(n=int(len(test_idx)), base_ess=round(float(yt.mean()),3),
            auc=round(auc(p,yt),3), auprc=round(auprc(p,yt),3),
            ne_cov_p90=round(cov_at_p(1-p,1-yt,0.9),3),
            ess_cov_p90=round(cov_at_p(p,yt,0.9),3),
            ess_cov_p70=round(cov_at_p(p,yt,0.7),3))
        print(f"{cl:<14} n={len(test_idx):<5} base={yt.mean():.3f} AUC={results[cl]['auc']:.3f} "
              f"AUPRC={results[cl]['auprc']:.3f} | nonEss cov@P90={results[cl]['ne_cov_p90']:.2f} "
              f"Ess cov@P70={results[cl]['ess_cov_p70']:.2f}")

    # pooled across held clades
    ev=~np.isnan(all_p); pe=all_p[ev]; ye=y[ev]; ce=all_c[ev]
    pooled=dict(n=int(ev.sum()), auc=round(auc(pe,ye),3), auprc=round(auprc(pe,ye),3),
        ne_cov_p90=round(cov_at_p(1-pe,1-ye,0.9),3),
        ess_cov_p70=round(cov_at_p(pe,ye,0.7),3),
        ess_cov_p90=round(cov_at_p(pe,ye,0.9),3))
    print(f"\nPOOLED (leave-one-clade-out, {ev.sum()} genes): AUC {pooled['auc']} "
          f"AUPRC {pooled['auprc']} | nonEss cov@P90 {pooled['ne_cov_p90']} "
          f"Ess cov@P70 {pooled['ess_cov_p70']}")

    # confidence calibration: does brightness predict correctness?
    pred=(pe>0.5).astype(int); corr=(pred==ye).astype(int)
    print("\nbrightness calibration (confidence bin -> actual accuracy):")
    calib=[]
    for lo,hi in [(0,.5),(.5,.7),(.7,.85),(.85,1.01)]:
        b=(ce>=lo)&(ce<hi)
        if b.sum()>50:
            calib.append(dict(bin=f"{lo}-{hi}", n=int(b.sum()), acc=round(float(corr[b].mean()),3)))
            print(f"  conf {lo:.2f}-{hi:.2f}: n={int(b.sum()):<6} accuracy={corr[b].mean():.3f}")

    json.dump(dict(per_clade=results, pooled=pooled, calibration=calib),
              open(OUT/"af_model_results.json","w"), indent=2)

    # figure: AlphaFold-style brightness
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(17,5.5))
    # A: per-clade AUC
    cls=list(results); ax[0].bar(range(len(cls)),[results[c]["auc"] for c in cls],color="#2980B9")
    ax[0].set_xticks(range(len(cls))); ax[0].set_xticklabels(cls,rotation=30,ha="right",fontsize=8)
    ax[0].axhline(0.5,color="gray",ls="--",lw=0.8); ax[0].set_ylim(0,1); ax[0].set_ylabel("AUC (leave-one-clade-out)")
    ax[0].set_title("A. Cross-clade AUC (AlphaFold-MSA model)")
    for i,c in enumerate(cls): ax[0].text(i,results[c]["auc"],f"{results[c]['auc']:.2f}",ha="center",va="bottom",fontsize=8)
    # B: brightness calibration
    bb=[c["bin"] for c in calib]; ba=[c["acc"] for c in calib]
    ax[1].plot(range(len(bb)),ba,"o-",color="#C0392B",lw=2)
    ax[1].set_xticks(range(len(bb))); ax[1].set_xticklabels(bb,fontsize=8)
    ax[1].set_ylim(0,1.02); ax[1].set_ylabel("actual accuracy"); ax[1].set_xlabel("predicted confidence (brightness) bin")
    ax[1].set_title("B. Brightness calibration\n(does the pLDDT-style head predict correctness?)")
    ax[1].grid(alpha=0.3)
    # C: scatter prob vs brightness colored by correctness
    sub=rng.choice(len(pe),min(4000,len(pe)),replace=False)
    sc=ax[2].scatter(pe[sub],ce[sub],c=corr[sub],cmap="RdYlGn",s=6,alpha=0.5,vmin=0,vmax=1)
    ax[2].set_xlabel("p(essential)"); ax[2].set_ylabel("brightness (confidence)")
    ax[2].set_title("C. Prediction vs brightness\n(green=correct, red=wrong)")
    fig.suptitle("AlphaFold-style essentiality model: attention over the ortholog-MSA + confidence head",
                 fontweight="bold",fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT/"af_model.png",dpi=140,bbox_inches="tight")
    print("\nwrote af_model_results.json + af_model.png")
