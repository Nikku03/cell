"""Cross-organism ESM transfer: train essentiality on E. coli ESM -> predict
M. tuberculosis (mtub) essentiality (DeJesus truth). Compare ESM-8M vs ESM-150M
(does a bigger sequence model help where sequence actually matters?). + conservation.
"""
import csv, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
OUT=Path("outputs/orphan"); torch.manual_seed(0); np.random.seed(0)
# truths
b_of={r["locus_tag"]:r["b_number"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
ecoli_ess={}
for r in csv.DictReader(open("data/drive_import/labels/labels.csv")):
    if r["organism"]=="beril_Keio":
        b=b_of.get(r["locus_tag"]);
        if b: ecoli_ess[b]=int(r["essential"])
dej={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("data/drive_import/labels/labels.csv"))
     if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}
def auc(p,y):
    p=np.asarray(p);y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(p);r=np.empty(len(p));r[o]=np.arange(len(p));m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
class Net(nn.Module):
    def __init__(s,d,h=128): super().__init__(); s.f=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x): return s.f(x).squeeze(-1)
def train_predict(Xtr,ytr,Xte,ep=40):
    mu=Xtr.mean(0);sd=Xtr.std(0);sd[sd<1e-9]=1;Xn=(Xtr-mu)/sd;Xen=(Xte-mu)/sd
    net=Net(Xtr.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    p1=ytr.mean();lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-p1)/max(p1,1e-6)))
    Xt=torch.from_numpy(Xn).float();yt=torch.from_numpy(ytr).float()
    for e in range(ep):
        idx=torch.randperm(len(Xt))
        for i in range(0,len(idx),256):
            j=idx[i:i+256];ls=lf(net(Xt[j]),yt[j]);opt.zero_grad();ls.backward();opt.step()
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(torch.from_numpy(Xen).float())).numpy()

def test(ecoli_npz, mtub_npz, tag):
    de=np.load(OUT/ecoli_npz,allow_pickle=True); eb=[str(x) for x in de["b"]]; EE=de["E"].astype(np.float32)
    ei={b:i for i,b in enumerate(eb)}
    dm=np.load(OUT/mtub_npz,allow_pickle=True)
    mk=[str(x) for x in (dm["g_keys"] if "g_keys" in dm else dm["b"])]; ME=(dm["G_esm"] if "G_esm" in dm else dm["E"]).astype(np.float32)
    # train set: E. coli genes with truth
    tr_b=[b for b in ecoli_ess if b in ei]
    Xtr=EE[[ei[b] for b in tr_b]]; ytr=np.array([ecoli_ess[b] for b in tr_b])
    # test: mtub genes with DeJesus
    keep=[i for i,k in enumerate(mk) if k in dej]
    Xte=ME[keep]; yte=np.array([dej[mk[i]] for i in keep])
    if len(keep)<10: print(f"  {tag}: mtub match {len(keep)} (id mismatch?)"); return None
    pm=train_predict(Xtr,ytr,Xte)
    a=auc(pm,yte)
    print(f"  {tag:<14} train E.coli {len(tr_b)} -> mtub {len(keep)} (ess {yte.sum()})  cross-org AUC = {a:.3f}")
    return a

print("=== cross-organism ESM essentiality transfer (E. coli -> mtub DeJesus) ===")
res={}
try: res["esm8M"]=test("ecoli_esm.npz","MycoTube_esm.npz","ESM-8M")
except Exception as e: print("  8M failed:",e)
try: res["esm150M"]=test("ecoli_esm150.npz","mtub_esm150.npz","ESM-150M")
except Exception as e: print("  150M failed:",e)
import json; json.dump({k:(round(v,3) if v else None) for k,v in res.items()},open(OUT/"cross_org_esm.json","w"),indent=2)
print("\nwrote outputs/orphan/cross_org_esm.json")
