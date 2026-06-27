"""Clean head-to-head for CROSS-ORG gene/protein essentiality on mtub (DeJesus):
  ESM-protein model  vs  conservation (feba-ortholog projection)  vs  combined.
Same eval set. Answers: does the protein (ESM) signal ADD over conservation?
"""
import csv, json, sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

def feba_label(fb,g_keys):
    seen=set(); minf={}
    for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId=?",(fb,)):
        if fit is None: continue
        seen.add(lid)
        if lid not in minf or fit<minf[lid]: minf[lid]=fit
    return np.array([1 if ((k not in seen) or (minf.get(k,0.0)<-3.0)) else 0 for k in g_keys])

# train pool
X=[];Y=[]
for fb in PILOT:
    d=np.load(INP/f"{fb}_esm.npz",allow_pickle=True); gk=[str(k) for k in d["g_keys"]]
    X.append(d["G_esm"].astype(np.float32)); Y.append(feba_label(fb,gk))
X=np.concatenate(X); Y=np.concatenate(Y)

class Net(nn.Module):
    def __init__(s,d,h=256): super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def auc(p,y):
    p=np.asarray(p,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=y==1; pos=m.sum()
    return float((r[m].sum()-pos*(pos-1)/2)/(pos*(~m).sum()))
mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
p1=Y.mean(); lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/p1))
Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(Y).float()
for ep in range(30):
    idx=torch.randperm(len(Xt)); net.train()
    for i in range(0,len(idx),2048):
        j=idx[i:i+2048]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()

# mtub eval
m=np.load(INP/"MycoTube_esm.npz",allow_pickle=True); mk=[str(k) for k in m["g_keys"]]; ME=m["G_esm"].astype(np.float32)
dej={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv"))
     if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}
net.eval()
with torch.no_grad(): esm_pred_all=torch.sigmoid(net(torch.from_numpy(((ME-mu)/sd)).float())).numpy()
esm_pred={mk[i]:esm_pred_all[i] for i in range(len(mk))}

# conservation via feba Ortholog: project panel LABELS onto mtub genes
panel_label={}  # (org, locus)->ess, panel = all feba orgs except MycoTube
# use our feba_label proxy per pilot AND broader: use beril truth labels for ALL non-mtub orgs
lab=defaultdict(dict)
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    if r["organism"]=="mtub": continue
    fb=r["organism"].replace("beril_","")
    lab[fb][r["locus_tag"]]=int(r["essential"])
orth=defaultdict(list)
for o2,l2,l1 in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1='MycoTube'"):
    orth[l1].append((o2,l2))
cons={}
for g in mk:
    vals=[lab[o][l] for o,l in orth.get(g,[]) if o in lab and l in lab[o]]
    if vals: cons[g]=np.mean(vals)

# eval set = mtub genes with DeJesus truth AND a conservation value AND esm pred
common=[g for g in mk if g in dej and g in cons and g in esm_pred]
y=np.array([dej[g] for g in common])
e=np.array([esm_pred[g] for g in common]); c=np.array([cons[g] for g in common])
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1)
comb=z(e)+z(c)
print(f"eval set (mtub, DeJesus & conservation & ESM): n={len(common)} ess={int(y.sum())}")
print(f"  ESM-protein        AUC = {auc(e,y):.3f}")
print(f"  conservation       AUC = {auc(c,y):.3f}")
print(f"  ESM + conservation AUC = {auc(comb,y):.3f}")
json.dump(dict(n=len(common),ess=int(y.sum()),esm_auc=round(auc(e,y),3),
               cons_auc=round(auc(c,y),3),combined_auc=round(auc(comb,y),3)),
          open(OUT/"protein_ess_compare.json","w"),indent=2)
print("\nwrote outputs/orphan/protein_ess_compare.json")
