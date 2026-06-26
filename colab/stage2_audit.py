"""STAGE 2 AUDIT: the CORRECT conditional gate.

Aggregate R on held-out conditions is Bayes-dominated by gene-mean (most genes
are condition-flat). The real question is whether the model predicts the
CONDITION-SPECIFIC RESIDUAL: fit - gene_mean.

Metrics on held-out conditions:
  R_resid   = pearson(pred - gene_mean, fit - gene_mean)   <- conditional signal
  AUC_def   = detect strong defects (residual_true < -1) by predicted residual
  also report aggregate R for reference.

If R_resid ~ 0 for both MLP and XATTN, the conditional signal is not learnable
from (gene features, media composition) here -> framing problem, not metric.
If R_resid > 0 meaningfully, the signal exists and Stage 3 (better gene encoder
+ genome context) is worth building.

Fast: Btheta (542 conditions, richest) + Keio. 3 epochs.
"""
import json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
ORGS=["Btheta","Keio","Putida"]

def load(fb):
    d=np.load(INP/f"{fb}.npz",allow_pickle=True); C=d["C"]
    comp=[]
    for r in C:
        ids=list(np.nonzero(r[:200])[0])
        if r[200]>0.5: ids.append(200)
        if r[201]>0.5: ids.append(201)
        comp.append(np.array(ids if ids else [202],dtype=np.int64))
    MAXC=64; PAD=203
    cp=np.full((len(comp),MAXC),PAD,np.int64)
    for i,a in enumerate(comp): a=a[:MAXC]; cp[i,:len(a)]=a
    return dict(G=d["G"].astype(np.float32),C=C.astype(np.float32),gi=d["gi"].astype(np.int64),
                ci=d["ci"].astype(np.int64),fit=d["fit"].astype(np.float32),t=d["t"].astype(np.float32),comp_pad=cp)
PAD=203; N_TOK=204; G_DIM=load_dim=None

class MLP(nn.Module):
    def __init__(s,g,c,D=64,h=128):
        super().__init__(); s.ge=nn.Sequential(nn.Linear(g,128),nn.ReLU(),nn.Linear(128,D))
        s.ce=nn.Sequential(nn.Linear(c,64),nn.ReLU(),nn.Linear(64,D))
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,gv,cv,comp,mask): return s.h(torch.cat([s.ge(gv),s.ce(cv)],-1)).squeeze(-1)
class XAttn(nn.Module):
    def __init__(s,g,c,D=64,h=128):
        super().__init__(); s.ge=nn.Sequential(nn.Linear(g,128),nn.ReLU(),nn.Linear(128,D))
        s.emb=nn.Embedding(N_TOK+1,D,padding_idx=PAD); s.q=nn.Linear(D,D); s.k=nn.Linear(D,D); s.v=nn.Linear(D,D)
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1)); s.D=D
    def forward(s,gv,cv,comp,mask):
        g=s.ge(gv); E=s.emb(comp); q=s.q(g).unsqueeze(1); k=s.k(E); v=s.v(E)
        sc=(q*k).sum(-1)/(s.D**0.5); sc=sc.masked_fill(~mask,-1e9); w=torch.softmax(sc,-1).unsqueeze(-1)
        return s.h(torch.cat([g,(w*v).sum(1)],-1)).squeeze(-1)

def pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-12))
def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1; pos=p.sum()
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

def run(fb, Model, gdim, cdim, epochs=3, bs=8192):
    d=load(fb); nC=d["C"].shape[0]
    hold=np.zeros(nC,bool); hold[np.random.choice(nC,int(nC*0.2),replace=False)]=True
    tr=~hold[d["ci"]]; ev=hold[d["ci"]]
    gmean=np.zeros(d["G"].shape[0]); cnt=np.zeros(d["G"].shape[0])
    for gix,f in zip(d["gi"][tr],d["fit"][tr]): gmean[gix]+=f; cnt[gix]+=1
    gmean/=np.maximum(cnt,1)
    net=Model(gdim,cdim); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.HuberLoss(delta=1.0,reduction='none')
    tri=np.nonzero(tr)[0]
    for ep in range(epochs):
        np.random.shuffle(tri); net.train()
        for i in range(0,len(tri),bs):
            sel=tri[i:i+bs]
            gv=torch.from_numpy(d["G"][d["gi"][sel]]); cv=torch.from_numpy(d["C"][d["ci"][sel]])
            comp=torch.from_numpy(d["comp_pad"][d["ci"][sel]]); mask=comp!=PAD
            y=torch.from_numpy(d["fit"][sel]); w=torch.from_numpy(np.clip(np.abs(d["t"][sel]),0,10))
            ls=(lf(net(gv,cv,comp,mask),y)*w).mean(); opt.zero_grad(); ls.backward(); opt.step()
    net.eval(); evi=np.nonzero(ev)[0]; preds=[]
    with torch.no_grad():
        for i in range(0,len(evi),bs):
            sel=evi[i:i+bs]
            gv=torch.from_numpy(d["G"][d["gi"][sel]]); cv=torch.from_numpy(d["C"][d["ci"][sel]])
            comp=torch.from_numpy(d["comp_pad"][d["ci"][sel]])
            preds.append(net(gv,cv,comp,comp!=PAD).numpy())
    preds=np.concatenate(preds)
    yt=d["fit"][evi]; gm=gmean[d["gi"][evi]]
    resid_true=yt-gm; resid_pred=preds-gm
    R_agg=pear(preds,yt); R_gm=pear(gm,yt); R_resid=pear(resid_pred,resid_true)
    # defect detection: residual_true < -1 is a condition-specific strong defect
    ydef=(resid_true<-1).astype(int)
    auc_def=auc(-resid_pred, ydef)   # more-negative predicted residual -> more likely defect
    frac_def=ydef.mean()
    return dict(org=fb,model=Model.__name__,R_agg=round(R_agg,3),R_genemean=round(R_gm,3),
                R_resid=round(R_resid,3),defect_AUC=round(auc_def,3),frac_strong_defect=round(float(frac_def),4),n_eval=len(evi))

d0=load("Keio"); G_DIM=d0["G"].shape[1]; C_DIM=d0["C"].shape[1]
res=[]
print(f"{'org':<8}{'model':<7}{'R_agg':>7}{'R_genemean':>11}{'R_resid':>9}{'defect_AUC':>11}{'%defect':>9}")
for fb in ORGS:
    for M in (MLP,XAttn):
        r=run(fb,M,G_DIM,C_DIM); res.append(r)
        print(f"  {fb:<6}{r['model']:<7}{r['R_agg']:>7.3f}{r['R_genemean']:>11.3f}{r['R_resid']:>9.3f}{r['defect_AUC']:>11.3f}{r['frac_strong_defect']*100:>8.1f}%",flush=True)

mlp_resid=np.mean([r['R_resid'] for r in res if r['model']=='MLP'])
xat_resid=np.mean([r['R_resid'] for r in res if r['model']=='XAttn'])
mlp_auc=np.mean([r['defect_AUC'] for r in res if r['model']=='MLP'])
xat_auc=np.mean([r['defect_AUC'] for r in res if r['model']=='XAttn'])
print(f"\nMEAN residual-R: MLP={mlp_resid:.3f}  XATTN={xat_resid:.3f}")
print(f"MEAN defect-AUC: MLP={mlp_auc:.3f}  XATTN={xat_auc:.3f}")
print("\nVERDICT:")
if max(mlp_resid,xat_resid)<0.05 and max(mlp_auc,xat_auc)<0.6:
    print("  conditional signal ~ABSENT from (gene feats, media comp). Framing/feature problem.")
else:
    print("  conditional signal PRESENT -> Stage 3 (ESM + genome context) worth building.")
json.dump(res, open(OUT/"stage2_audit.json","w"), indent=2)
print("\nwrote outputs/orphan/stage2_audit.json")
