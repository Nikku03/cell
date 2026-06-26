"""STAGE 2 (+ clean Stage 1 redo): leak-free, streaming, head-to-head.

Fixes vs Stage 1:
  - LEAK: eval rows no longer enter training (strict split).
  - OOM: stream features by gathering G[gi]/C[ci] per batch from per-org matrices.

Two models on IDENTICAL splits:
  MLP   : concat(gene_enc, cond_bag_enc) -> head           (Stage 1 baseline)
  XATTN : gene query cross-attends over the condition's COMPOUND SET -> head
          (Stage 2; this is the gene<->nutrient interaction mechanism)

Splits (all leak-free):
  (1) within-org HELD-OUT CONDITIONS  : 80/20 condition split; train+eval same org
  (2) within-org HELD-OUT GENES       : 80/20 gene split
  (3) leave-one-ORG-out               : train all-but-one, eval the held org

GATE for Stage 2: on held-out CONDITIONS, (XATTN R - gene_mean R) must exceed
(MLP R - gene_mean R) by a clear margin (target +0.10 absolute over gene-mean).
"""
import json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs")
OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0); device="cpu"
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

# ---- load per-org; derive per-condition compound sets from C[:, :200] ----
ORG={}
for k,fb in enumerate(PILOT):
    d=np.load(INP/f"{fb}.npz", allow_pickle=True)
    C=d["C"]
    comp_sets=[]   # per condition: list of token ids (0..199 compounds, 200 aerobic, 201 stress)
    for r in C:
        ids=list(np.nonzero(r[:200])[0])
        if r[200]>0.5: ids.append(200)
        if r[201]>0.5: ids.append(201)
        comp_sets.append(np.array(ids if ids else [202], dtype=np.int64))  # 202 = empty/pad-real
    ORG[fb]=dict(G=d["G"].astype(np.float32), C=C.astype(np.float32),
                 gi=d["gi"].astype(np.int64), ci=d["ci"].astype(np.int64),
                 fit=d["fit"].astype(np.float32), t=d["t"].astype(np.float32),
                 comp=comp_sets, oid=k)
G_DIM=ORG[PILOT[0]]["G"].shape[1]; C_DIM=ORG[PILOT[0]]["C"].shape[1]
N_TOK=203+1  # 0..202 + pad(203)
PAD=203
MAXC=64
print(f"gene_dim={G_DIM} cond_dim={C_DIM} compound_vocab={N_TOK}")

def pad_comp(idx_list):
    out=np.full((len(idx_list),MAXC),PAD,dtype=np.int64)
    for i,a in enumerate(idx_list):
        a=a[:MAXC]; out[i,:len(a)]=a
    return out

# precompute per-org padded compound matrix once (n_cond, MAXC) -> vectorized gather
for fb in PILOT:
    ORG[fb]["comp_pad"]=pad_comp(ORG[fb]["comp"])

# ---- models ----
class MLP(nn.Module):
    def __init__(s,g,c,D=64,h=128):
        super().__init__()
        s.ge=nn.Sequential(nn.Linear(g,128),nn.ReLU(),nn.Linear(128,D))
        s.ce=nn.Sequential(nn.Linear(c,64),nn.ReLU(),nn.Linear(64,D))
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,gv,cv,comp,mask):
        return s.h(torch.cat([s.ge(gv),s.ce(cv)],-1)).squeeze(-1)

class XAttn(nn.Module):
    def __init__(s,g,D=64,h=128):
        super().__init__()
        s.ge=nn.Sequential(nn.Linear(g,128),nn.ReLU(),nn.Linear(128,D))
        s.emb=nn.Embedding(N_TOK+1,D,padding_idx=PAD)  # +1 safety
        s.q=nn.Linear(D,D); s.k=nn.Linear(D,D); s.v=nn.Linear(D,D)
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
        s.D=D
    def forward(s,gv,cv,comp,mask):
        g=s.ge(gv)                       # B,D
        E=s.emb(comp)                    # B,L,D
        q=s.q(g).unsqueeze(1)            # B,1,D
        k=s.k(E); v=s.v(E)               # B,L,D
        sc=(q*k).sum(-1)/ (s.D**0.5)     # B,L
        sc=sc.masked_fill(~mask, -1e9)
        w=torch.softmax(sc,-1).unsqueeze(-1)  # B,L,1
        ctx=(w*v).sum(1)                 # B,D
        return s.h(torch.cat([g,ctx],-1)).squeeze(-1)

def pearson(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-12))
def mse(a,b): return float(((np.asarray(a)-np.asarray(b))**2).mean())

def make_batches(rows, bs, shuffle=True):
    idx=np.arange(len(rows[0]))
    if shuffle: np.random.shuffle(idx)
    for i in range(0,len(idx),bs):
        yield idx[i*0+i:i+bs] if False else idx[i:i+bs]

def gather(org_rows, sel):
    """org_rows: dict with arrays already restricted to a split; sel: indices."""
    oi=org_rows["oid"][sel]; gi=org_rows["gi"][sel]; ci=org_rows["ci"][sel]
    # features gathered per source org
    gv=np.empty((len(sel),G_DIM),np.float32); cv=np.empty((len(sel),C_DIM),np.float32)
    comp=np.full((len(sel),MAXC),PAD,np.int64)
    for k,fb in enumerate(PILOT):
        m=oi==k
        if not m.any(): continue
        d=ORG[fb]
        gv[m]=d["G"][gi[m]]; cv[m]=d["C"][ci[m]]
        comp[m]=d["comp_pad"][ci[m]]      # vectorized -- no python loop
    return gv,cv,comp

def train_model(Model, train, eval_, epochs, bs=4096, lr=1e-3, sample=None):
    net=(Model(G_DIM,C_DIM) if Model is MLP else Model(G_DIM)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    lf=nn.HuberLoss(delta=1.0,reduction='none')
    Ntr=len(train["fit"])
    for ep in range(epochs):
        order=np.random.permutation(Ntr)
        if sample and Ntr>sample: order=order[:sample]
        net.train(); tot=0; nb=0
        for i in range(0,len(order),bs):
            sel=order[i:i+bs]
            gv,cv,comp=gather(train,sel)
            gv=torch.from_numpy(gv); cv=torch.from_numpy(cv); comp=torch.from_numpy(comp)
            mask=comp!=PAD
            y=torch.from_numpy(train["fit"][sel]); w=torch.from_numpy(np.clip(np.abs(train["t"][sel]),0,10))
            pred=net(gv,cv,comp,mask)
            ls=(lf(pred,y)*w).mean()
            opt.zero_grad(); ls.backward(); opt.step(); tot+=ls.item(); nb+=1
        # eval
        net.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(eval_["fit"]),bs):
                sel=np.arange(i,min(i+bs,len(eval_["fit"])))
                gv,cv,comp=gather(eval_,sel)
                gv=torch.from_numpy(gv); cv=torch.from_numpy(cv); comp=torch.from_numpy(comp)
                preds.append(net(gv,cv,comp,comp!=PAD).numpy())
        preds=np.concatenate(preds)
        r=pearson(preds,eval_["fit"]); m=mse(preds,eval_["fit"])
    return r,m,preds

def subset(org_or_orgs, row_mask=None):
    """build a streaming-friendly split dict over one or more orgs."""
    oid=[]; gi=[]; ci=[]; fit=[]; t=[]
    orgs=[org_or_orgs] if isinstance(org_or_orgs,str) else org_or_orgs
    for fb in orgs:
        d=ORG[fb]; n=len(d["fit"])
        m=row_mask[fb] if (row_mask and fb in row_mask) else np.ones(n,bool)
        oid.append(np.full(m.sum(),d["oid"])); gi.append(d["gi"][m]); ci.append(d["ci"][m])
        fit.append(d["fit"][m]); t.append(d["t"][m])
    return dict(oid=np.concatenate(oid),gi=np.concatenate(gi),ci=np.concatenate(ci),
                fit=np.concatenate(fit),t=np.concatenate(t))

results=[]
# ============ (1) within-org HELD-OUT CONDITIONS ============
print(f"\n{'='*64}\n(1) within-org HELD-OUT CONDITIONS  (leak-free)\n{'='*64}")
for eo in ["Putida","Keio","Btheta"]:
    d=ORG[eo]; nC=d["C"].shape[0]
    hold=np.zeros(nC,bool); hold[np.random.choice(nC,int(nC*0.2),replace=False)]=True
    tr_mask=~hold[d["ci"]]; ev_mask=hold[d["ci"]]
    train=subset(eo,{eo:tr_mask}); eval_=subset(eo,{eo:ev_mask})
    # gene-mean baseline from train
    gmean=np.zeros(d["G"].shape[0]); cnt=np.zeros(d["G"].shape[0])
    for gix,f in zip(train["gi"],train["fit"]): gmean[gix]+=f; cnt[gix]+=1
    gmean/=np.maximum(cnt,1)
    base=gmean[eval_["gi"]]; r_base=pearson(base,eval_["fit"])
    r_mlp,_,_=train_model(MLP,train,eval_,epochs=4)
    r_xat,_,_=train_model(XAttn,train,eval_,epochs=4)
    print(f"  {eo:<8} n_eval={len(eval_['fit']):>7}  gene-mean R={r_base:.3f}  MLP R={r_mlp:.3f}  XATTN R={r_xat:.3f}  "
          f"| XATTN-genemean={r_xat-r_base:+.3f}  MLP-genemean={r_mlp-r_base:+.3f}", flush=True)
    results.append(dict(test="held_cond",org=eo,gene_mean=r_base,MLP=r_mlp,XATTN=r_xat))

# ============ (2) within-org HELD-OUT GENES ============
print(f"\n{'='*64}\n(2) within-org HELD-OUT GENES  (leak-free)\n{'='*64}")
for eo in ["Putida","Keio"]:
    d=ORG[eo]; nG=d["G"].shape[0]
    hold=np.zeros(nG,bool); hold[np.random.choice(nG,int(nG*0.2),replace=False)]=True
    tr_mask=~hold[d["gi"]]; ev_mask=hold[d["gi"]]
    train=subset(eo,{eo:tr_mask}); eval_=subset(eo,{eo:ev_mask})
    cmean=np.zeros(d["C"].shape[0]); cnt=np.zeros(d["C"].shape[0])
    for cix,f in zip(train["ci"],train["fit"]): cmean[cix]+=f; cnt[cix]+=1
    cmean/=np.maximum(cnt,1)
    base=cmean[eval_["ci"]]; r_base=pearson(base,eval_["fit"])
    r_mlp,_,_=train_model(MLP,train,eval_,epochs=4)
    r_xat,_,_=train_model(XAttn,train,eval_,epochs=4)
    print(f"  {eo:<8} n_eval={len(eval_['fit']):>7}  cond-mean R={r_base:.3f}  MLP R={r_mlp:.3f}  XATTN R={r_xat:.3f}", flush=True)
    results.append(dict(test="held_gene",org=eo,cond_mean=r_base,MLP=r_mlp,XATTN=r_xat))

# ============ (3) leave-one-ORG-out (streamed, no OOM) ============
print(f"\n{'='*64}\n(3) leave-one-ORG-out  (streamed)\n{'='*64}")
for eo in ["Keio","Caulo"]:
    tr_orgs=[o for o in PILOT if o!=eo]
    train=subset(tr_orgs); eval_=subset(eo)
    r_mlp,_,_=train_model(MLP,train,eval_,epochs=2,sample=1_500_000)
    r_xat,_,_=train_model(XAttn,train,eval_,epochs=2,sample=1_500_000)
    print(f"  hold {eo:<8} n_eval={len(eval_['fit']):>7}  MLP R={r_mlp:.3f}  XATTN R={r_xat:.3f}  (vs zero MSE baseline)", flush=True)
    results.append(dict(test="leave_org",org=eo,MLP=r_mlp,XATTN=r_xat))

json.dump(results, open(OUT/"stage2_xattn.json","w"), indent=2)
print(f"\nwrote {OUT}/stage2_xattn.json")
