"""PROTEIN-LEVEL essentiality: does an ESM protein model predict a NEW organism's
essential GENES? (the gene/protein question, not chemical-response)

Train ESM(protein) -> P(essential) on the 8 pilot orgs.
  essential label (pilots) from feba: gene is essential-proxy if ABSENT from the
  fitness table (no surviving mutants) OR min-fit < -3 across conditions.
Test:
  (a) within-org held genes (mean over pilots): ESM -> own feba-essential
  (b) CROSS-ORG to mtub on INDEPENDENT DeJesus truth (clean, no feba labels)
  (c) baseline: conservation (leak-clean OG essential rate) on mtub DeJesus
Compare. The protein-level signal should TRANSFER (proteins are homologous),
unlike the chemical-conditional signal that collapsed in Stage 5.
"""
import csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
INP=Path("/home/user/cell/data/external_data/transformer_inputs"); OUT=Path("/home/user/cell/outputs/orphan")
torch.manual_seed(0); np.random.seed(0)
PILOT=["Putida","Keio","Koxy","Caulo","Btheta","MR1","PV4","Dda3937"]

# pilot ESM + feba-essential proxy label
def feba_essential_label(fb, g_keys):
    # need seen + minfit; recompute from the per-org npz triples? easier: from db
    import sqlite3
    con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
    seen=set(); minf={}
    for lid,fit in cur.execute("SELECT locusId,fit FROM GeneFitness WHERE orgId=?",(fb,)):
        if fit is None: continue
        seen.add(lid)
        if lid not in minf or fit<minf[lid]: minf[lid]=fit
    y=[]
    for k in g_keys:
        ess = (k not in seen) or (minf.get(k,0.0) < -3.0)
        y.append(1 if ess else 0)
    return np.array(y)

X=[]; Y=[]
for fb in PILOT:
    d=np.load(INP/f"{fb}_esm.npz",allow_pickle=True)
    g_keys=[str(k) for k in d["g_keys"]]; E=d["G_esm"].astype(np.float32)
    y=feba_essential_label(fb,g_keys)
    X.append(E); Y.append(y)
    print(f"  {fb:<8} {len(g_keys)} genes, feba-essential {y.sum()} ({y.mean():.2%})")
X=np.concatenate(X); Y=np.concatenate(Y)
print(f"train pool: {len(X)} genes, {Y.sum()} essential ({Y.mean():.2%})")

class Net(nn.Module):
    def __init__(s,d,h=256):
        super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def auc(p,y):
    p=np.asarray(p,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=y==1; pos=m.sum()
    return float((r[m].sum()-pos*(pos-1)/2)/(pos*(~m).sum()))
def train(X,y,Xv,epochs=30,bs=2048):
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1
    Xn=(X-mu)/sd; Xvn=(Xv-mu)/sd
    net=Net(X.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    p1=y.mean(); pw=torch.tensor((1-p1)/max(p1,1e-6))
    lf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.from_numpy(Xn).float(); yt=torch.from_numpy(y).float()
    for ep in range(epochs):
        idx=torch.randperm(len(Xt)); net.train()
        for i in range(0,len(idx),bs):
            j=idx[i:i+bs]; ls=lf(net(Xt[j]),yt[j]); opt.zero_grad(); ls.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pv=torch.sigmoid(net(torch.from_numpy(Xvn).float())).numpy()
    return pv

# (a) within-org held genes: 5-fold on the pooled pilots
print("\n(a) within-pilot 5-fold (ESM -> feba-essential)")
idx=np.arange(len(X)); np.random.shuffle(idx); oof=np.zeros(len(X))
for f in np.array_split(idx,5):
    tr=np.setdiff1d(idx,f); oof[f]=train(X[tr],Y[tr],X[f],epochs=20)
print(f"  within-pilot AUC = {auc(oof,Y):.3f}")

# (b) cross-org to mtub on DeJesus truth
m=np.load(INP/"MycoTube_esm.npz",allow_pickle=True)
mk=[str(k) for k in m["g_keys"]]; ME=m["G_esm"].astype(np.float32)
dej={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv"))
     if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}
keep=[i for i,k in enumerate(mk) if k in dej]
Xm=ME[keep]; ym=np.array([dej[mk[i]] for i in keep])
pm=train(X,Y,Xm,epochs=30)
print(f"\n(b) CROSS-ORG mtub on DeJesus: n={len(ym)} ess={ym.sum()}  ESM-model AUC = {auc(pm,ym):.3f}")

# (c) conservation baseline on mtub DeJesus
ALL=list(csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
held=clade_of.get("mtub","mycobacterium")
n=defaultdict(int); e=defaultdict(int)
for r in ALL:
    if clade_of.get(r["organism"])==held: continue
    og=og_of.get((r["organism"],r["locus_tag"]))
    if og: n[og]+=1; e[og]+=int(r["essential"])
rate={og:e[og]/n[og] for og in n if n[og]>=5}
cons=[]; yc=[]
for i in keep:
    og=og_of.get(("mtub",mk[i]))
    if og is None: continue
    cons.append(rate.get(og,0.0)); yc.append(dej[mk[i]])
print(f"(c) conservation baseline on mtub DeJesus: n={len(yc)}  AUC = {auc(cons,yc):.3f}")

json.dump(dict(within_pilot_auc=round(auc(oof,Y),3),
               crossorg_mtub_esm_auc=round(auc(pm,ym),3),
               conservation_mtub_auc=round(auc(cons,yc),3),
               n_mtub=len(ym), ess_mtub=int(ym.sum())),
          open(OUT/"protein_essentiality.json","w"),indent=2)
print("\nwrote outputs/orphan/protein_essentiality.json")
