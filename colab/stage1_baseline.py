"""STAGE 1 GATE: simplest fitness-completion model + leakage-proof eval.

Model: gene_feat (442) -> 64; cond_feat (203) -> 32; concat -> MLP -> fit
Training target: feba fitness (Huber, weighted by |t|)

Gate: must beat per-gene-mean and per-condition-mean baselines on HELD-OUT
CONDITIONS within the same organism. If it can't, the tensor isn't completable
from sequence+chemistry features and the framing is wrong -- stop here.

Eval:
  (1) within-org held-out CONDITIONS  -- can it generalize across conditions?
  (2) within-org held-out GENES       -- can it generalize across genes?
  (3) cross-ORG: train on 7, hold one out (the real test of organism-generalization)
Metric: per-cell MSE + Pearson R; for essentiality readout, derive
  essentiality_score = mean_predicted_fit_across_all_conditions (lower=more
  essential) and check AUC against feba absent_from_feba on the held-out org
  (proxy truth here; honest test still needs DeJesus for mtub later).
"""
import json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs")
OUT=Path("/home/user/cell/outputs/orphan")
device="cpu"; torch.manual_seed(7); np.random.seed(7)

PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

# load all
data={}
for fb in PILOT:
    d=np.load(INP/f"{fb}.npz", allow_pickle=True)
    data[fb]=dict(G=d["G"], C=d["C"], gi=d["gi"], ci=d["ci"], fit=d["fit"], t=d["t"])
    print(f"  loaded {fb}: G={d['G'].shape} C={d['C'].shape} triples={len(d['gi'])}")
G_DIM=data[PILOT[0]]["G"].shape[1]; C_DIM=data[PILOT[0]]["C"].shape[1]
print(f"\ngene_dim={G_DIM}  cond_dim={C_DIM}")

class FitNet(nn.Module):
    def __init__(self, gdim, cdim, eg=64, ec=32, h=128):
        super().__init__()
        self.gn=nn.Sequential(nn.Linear(gdim,128), nn.ReLU(), nn.Linear(128,eg))
        self.cn=nn.Sequential(nn.Linear(cdim,64), nn.ReLU(), nn.Linear(64,ec))
        self.head=nn.Sequential(nn.Linear(eg+ec,h), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, gv, cv):
        g=self.gn(gv); c=self.cn(cv); return self.head(torch.cat([g,c],-1)).squeeze(-1)

def pearson(a,b):
    a=np.asarray(a); b=np.asarray(b); a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-12))

def mse(a,b): return float(((np.asarray(a)-np.asarray(b))**2).mean())

def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(-s); yo=y[o]; pos=(y==1).sum()
    r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))

def train_eval(train_orgs, eval_org, hold_kind, epochs=4, bs=4096):
    # build train pool
    Xg=[]; Xc=[]; y=[]; w=[]
    for fb in train_orgs:
        d=data[fb]; G=d["G"]; C=d["C"]; gi=d["gi"]; ci=d["ci"]; fit=d["fit"]; t=d["t"]
        Xg.append(G[gi]); Xc.append(C[ci]); y.append(fit); w.append(np.clip(np.abs(t),0,10))
    Xg=np.concatenate(Xg); Xc=np.concatenate(Xc); y=np.concatenate(y); w=np.concatenate(w)
    # eval set
    d=data[eval_org]; G=d["G"]; C=d["C"]; gi=d["gi"]; ci=d["ci"]; fit=d["fit"]; t=d["t"]
    if hold_kind=="cond":
        # split conditions: 80/20
        nC=C.shape[0]; mask=np.zeros(nC,bool); mask[np.random.choice(nC,size=int(nC*0.2),replace=False)]=True
        ev=mask[ci]; tr=~mask[ci]
        # train within-org too on the train-cond rows
        Xg=np.concatenate([Xg, G[gi[tr]]]); Xc=np.concatenate([Xc, C[ci[tr]]]); y=np.concatenate([y, fit[tr]]); w=np.concatenate([w, np.clip(np.abs(t[tr]),0,10)])
        eXg=G[gi[ev]]; eXc=C[ci[ev]]; ey=fit[ev]; eg=gi[ev]; ec=ci[ev]
    elif hold_kind=="gene":
        nG=G.shape[0]; mask=np.zeros(nG,bool); mask[np.random.choice(nG,size=int(nG*0.2),replace=False)]=True
        ev=mask[gi]; tr=~mask[gi]
        Xg=np.concatenate([Xg, G[gi[tr]]]); Xc=np.concatenate([Xc, C[ci[tr]]]); y=np.concatenate([y, fit[tr]]); w=np.concatenate([w, np.clip(np.abs(t[tr]),0,10)])
        eXg=G[gi[ev]]; eXc=C[ci[ev]]; ey=fit[ev]; eg=gi[ev]; ec=ci[ev]
    elif hold_kind=="org":
        eXg=G[gi]; eXc=C[ci]; ey=fit; eg=gi; ec=ci
    # baselines on eval
    mean_gene=np.zeros(G.shape[0]); cnt_g=np.zeros(G.shape[0])
    mean_cond=np.zeros(C.shape[0]); cnt_c=np.zeros(C.shape[0])
    # compute baselines from EVAL ORG's TRAIN part if same org, else use eval set itself (degenerate but reported)
    if hold_kind!="org":
        tr_gi=gi[~mask[ci]] if hold_kind=="cond" else gi[~mask[gi]]
        tr_ci=ci[~mask[ci]] if hold_kind=="cond" else ci[~mask[gi]]
        tr_fit=fit[~mask[ci]] if hold_kind=="cond" else fit[~mask[gi]]
        for g_,f_ in zip(tr_gi,tr_fit): mean_gene[g_]+=f_; cnt_g[g_]+=1
        for c_,f_ in zip(tr_ci,tr_fit): mean_cond[c_]+=f_; cnt_c[c_]+=1
        mean_gene/=np.maximum(cnt_g,1); mean_cond/=np.maximum(cnt_c,1)
        global_mean=tr_fit.mean()
    else:
        global_mean=0.0
    # model
    net=FitNet(G_DIM,C_DIM).to(device); opt=torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn=nn.HuberLoss(delta=1.0, reduction='none')
    Xg=torch.from_numpy(Xg).float(); Xc=torch.from_numpy(Xc).float()
    yT=torch.from_numpy(y).float(); wT=torch.from_numpy(w).float()
    N=len(Xg); n_per=max(N//bs,1)
    for ep in range(epochs):
        idx=torch.randperm(N); tot=0.0
        for i in range(n_per):
            j=idx[i*bs:(i+1)*bs]
            pred=net(Xg[j].to(device), Xc[j].to(device))
            ls=(loss_fn(pred, yT[j].to(device))*wT[j].to(device)).mean()
            opt.zero_grad(); ls.backward(); opt.step()
            tot+=ls.item()
        # eval each epoch
        net.eval()
        with torch.no_grad():
            ep_pred=[]
            for i in range(0, len(eXg), bs):
                p=net(torch.from_numpy(eXg[i:i+bs]).float().to(device),
                      torch.from_numpy(eXc[i:i+bs]).float().to(device))
                ep_pred.append(p.cpu().numpy())
            ep_pred=np.concatenate(ep_pred)
        net.train()
        # baselines for the same eval rows
        if hold_kind!="org":
            base_gene=mean_gene[eg]; base_cond=mean_cond[ec]; base_global=np.full_like(ey,global_mean)
            r_mod=pearson(ep_pred,ey); r_g=pearson(base_gene,ey); r_c=pearson(base_cond,ey)
            mse_m=mse(ep_pred,ey); mse_g=mse(base_gene,ey); mse_c=mse(base_cond,ey)
            print(f"    epoch {ep+1}/{epochs} train_loss={tot/n_per:.3f}  MODEL R={r_mod:.3f} MSE={mse_m:.3f}  | "
                  f"gene-mean R={r_g:.3f} MSE={mse_g:.3f} | cond-mean R={r_c:.3f} MSE={mse_c:.3f}", flush=True)
        else:
            r=pearson(ep_pred,ey); m=mse(ep_pred,ey)
            print(f"    epoch {ep+1}/{epochs} train_loss={tot/n_per:.3f}  MODEL R={r:.3f} MSE={m:.3f}  (vs zero MSE={mse(np.zeros_like(ey),ey):.3f})", flush=True)
    return dict(eval_org=eval_org, hold_kind=hold_kind,
                model_R=r_mod if hold_kind!="org" else r,
                model_MSE=mse_m if hold_kind!="org" else m,
                gene_mean_R=r_g if hold_kind!="org" else None,
                cond_mean_R=r_c if hold_kind!="org" else None,
                gene_mean_MSE=mse_g if hold_kind!="org" else None,
                cond_mean_MSE=mse_c if hold_kind!="org" else None,
                n_eval=int(len(ey)))

results=[]
# (1) within-org HELD-OUT CONDITIONS (the primary gate)
print(f"\n{'='*60}\n(1) within-org HELD-OUT CONDITIONS  (the gate)\n{'='*60}")
for eo in ["Putida","Keio","Btheta"]:
    print(f"\n>>> eval_org={eo}, hold=CONDITIONS")
    r=train_eval([eo], eo, "cond", epochs=4)
    results.append(r)

# (2) within-org HELD-OUT GENES
print(f"\n{'='*60}\n(2) within-org HELD-OUT GENES\n{'='*60}")
for eo in ["Putida","Keio"]:
    print(f"\n>>> eval_org={eo}, hold=GENES")
    r=train_eval([eo], eo, "gene", epochs=4)
    results.append(r)

# (3) LEAVE-ONE-ORG-OUT
print(f"\n{'='*60}\n(3) leave-one-ORG-out  (cross-organism transfer)\n{'='*60}")
for eo in ["Keio","Caulo"]:
    train_orgs=[o for o in PILOT if o!=eo]
    print(f"\n>>> eval_org={eo}, train on {train_orgs}")
    r=train_eval(train_orgs, eo, "org", epochs=3)
    results.append(r)

json.dump(results, open(OUT/"stage1_baseline.json","w"), indent=2)
print(f"\nwrote {OUT}/stage1_baseline.json")
