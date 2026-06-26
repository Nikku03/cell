"""STAGE 3 GATE: does ESM gene encoder lift the conditional signal?

Sharpened metrics on held-out conditions:
  R_dc    = double-centered residual = fit - gene_mean - cond_mean + global_mean
            (pure gene x condition interaction; removes condition-global harshness)
  defect_AUC = detect strong gene-specific defects (dc_resid < -1)
Plus held-out-GENE R (ESM vs classical) -- the sequence->fitness generalization.

Head-to-head: MLP-classical  vs  MLP-ESM  vs  XATTN-ESM.
GATE: ESM must lift held-gene R (from ~0.1) AND defect_AUC > 0.6.
"""
import json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
ORGS=["Btheta","Keio","Putida"]
PAD=203; N_TOK=204

def load(fb):
    d=np.load(INP/f"{fb}.npz",allow_pickle=True); C=d["C"]
    comp=[]
    for r in C:
        ids=list(np.nonzero(r[:200])[0])
        if r[200]>0.5: ids.append(200)
        if r[201]>0.5: ids.append(201)
        comp.append(np.array(ids if ids else [202],dtype=np.int64))
    cp=np.full((len(comp),64),PAD,np.int64)
    for i,a in enumerate(comp): a=a[:64]; cp[i,:len(a)]=a
    esm=np.load(INP/f"{fb}_esm.npz",allow_pickle=True)["G_esm"].astype(np.float32)
    return dict(G=d["G"].astype(np.float32), Gesm=esm, C=C.astype(np.float32),
                gi=d["gi"].astype(np.int64), ci=d["ci"].astype(np.int64),
                fit=d["fit"].astype(np.float32), t=d["t"].astype(np.float32), comp_pad=cp)

class MLP(nn.Module):
    def __init__(s,g,c,D=64,h=128):
        super().__init__(); s.ge=nn.Sequential(nn.Linear(g,256),nn.ReLU(),nn.Linear(256,D))
        s.ce=nn.Sequential(nn.Linear(c,64),nn.ReLU(),nn.Linear(64,D))
        s.h=nn.Sequential(nn.Linear(2*D,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,gv,cv,comp,mask): return s.h(torch.cat([s.ge(gv),s.ce(cv)],-1)).squeeze(-1)
class XAttn(nn.Module):
    def __init__(s,g,c,D=64,h=128):
        super().__init__(); s.ge=nn.Sequential(nn.Linear(g,256),nn.ReLU(),nn.Linear(256,D))
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

def train(d, gkey, Model, mask_tr, mask_ev, epochs=3, bs=8192):
    Gf=d[gkey]; gdim=Gf.shape[1]; net=Model(gdim,d["C"].shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.HuberLoss(delta=1.0,reduction='none')
    tri=np.nonzero(mask_tr)[0]
    for ep in range(epochs):
        np.random.shuffle(tri); net.train()
        for i in range(0,len(tri),bs):
            sel=tri[i:i+bs]
            gv=torch.from_numpy(Gf[d["gi"][sel]]); cv=torch.from_numpy(d["C"][d["ci"][sel]])
            comp=torch.from_numpy(d["comp_pad"][d["ci"][sel]]); mask=comp!=PAD
            y=torch.from_numpy(d["fit"][sel]); w=torch.from_numpy(np.clip(np.abs(d["t"][sel]),0,10))
            ls=(lf(net(gv,cv,comp,mask),y)*w).mean(); opt.zero_grad(); ls.backward(); opt.step()
    net.eval(); evi=np.nonzero(mask_ev)[0]; preds=[]
    with torch.no_grad():
        for i in range(0,len(evi),bs):
            sel=evi[i:i+bs]
            gv=torch.from_numpy(Gf[d["gi"][sel]]); cv=torch.from_numpy(d["C"][d["ci"][sel]])
            comp=torch.from_numpy(d["comp_pad"][d["ci"][sel]])
            preds.append(net(gv,cv,comp,comp!=PAD).numpy())
    return np.concatenate(preds), evi

res=[]
print(f"{'org':<8}{'encoder':<14}{'R_agg':>7}{'R_dc':>7}{'defect_AUC':>11}{'heldgene_R':>11}")
for fb in ORGS:
    d=load(fb); nC=d["C"].shape[0]; nG=d["G"].shape[0]
    # held-condition split
    hold=np.zeros(nC,bool); hold[np.random.choice(nC,int(nC*0.2),replace=False)]=True
    tr=~hold[d["ci"]]; ev=hold[d["ci"]]
    gm=np.zeros(nG); gc=np.zeros(nG); cm=np.zeros(nC); cc=np.zeros(nC)
    for gix,cix,f in zip(d["gi"][tr],d["ci"][tr],d["fit"][tr]):
        gm[gix]+=f; gc[gix]+=1; cm[cix]+=f; cc[cix]+=1
    gm/=np.maximum(gc,1); cm/=np.maximum(cc,1); glob=d["fit"][tr].mean()
    # held-gene split (for sequence->fitness)
    hg=np.zeros(nG,bool); hg[np.random.choice(nG,int(nG*0.2),replace=False)]=True
    trg=~hg[d["gi"]]; evg=hg[d["gi"]]
    cmg=np.zeros(nC); ccg=np.zeros(nC)
    for cix,f in zip(d["ci"][trg],d["fit"][trg]): cmg[cix]+=f; ccg[cix]+=1
    cmg/=np.maximum(ccg,1)
    for gkey,enc,Model in [("G","classical-MLP",MLP),("Gesm","ESM-MLP",MLP),("Gesm","ESM-XATTN",XAttn)]:
        preds,evi=train(d,gkey,Model,tr,ev)
        yt=d["fit"][evi]; gmv=gm[d["gi"][evi]]; cmv=cm[d["ci"][evi]]
        R_agg=pear(preds,yt)
        dc_true=yt-gmv-cmv+glob; dc_pred=preds-gmv-cmv+glob
        R_dc=pear(dc_pred,dc_true)
        ydef=(dc_true<-1).astype(int); aucd=auc(-dc_pred,ydef)
        # held-gene
        pg,evgi=train(d,gkey,Model,trg,evg)
        ytg=d["fit"][evgi]; base=cmg[d["ci"][evgi]]
        Rhg=pear(pg-base, ytg-base)  # residual over cond-mean (pure gene effect)
        res.append(dict(org=fb,encoder=enc,R_agg=round(R_agg,3),R_dc=round(R_dc,3),
                        defect_AUC=round(aucd,3),heldgene_R=round(Rhg,3)))
        print(f"  {fb:<6}{enc:<14}{R_agg:>7.3f}{R_dc:>7.3f}{aucd:>11.3f}{Rhg:>11.3f}",flush=True)

# summary by encoder
print("\nMEAN by encoder:")
for enc in ["classical-MLP","ESM-MLP","ESM-XATTN"]:
    rs=[r for r in res if r["encoder"]==enc]
    print(f"  {enc:<14} R_dc={np.mean([r['R_dc'] for r in rs]):.3f}  defect_AUC={np.mean([r['defect_AUC'] for r in rs]):.3f}  heldgene_R={np.mean([r['heldgene_R'] for r in rs]):.3f}")
json.dump(res, open(OUT/"stage3_gate.json","w"), indent=2)
print("\nwrote outputs/orphan/stage3_gate.json")
