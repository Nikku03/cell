"""SOLVE intrinsic promoter strength from sequence, on the Urtecho 2019 MPRA
(10,898 sigma70 variants, measured expression). This is the clean target (fixed
condition) that our in-vivo PRECISE test (0.13) could not be.
  A: element model (UP/-35/spacer/-10/background one-hot) -> ridge on log-expr
  B: raw sequence one-hot (150 bp) -> ridge
  C: sequence -> small MLP
Held-out R^2. Confirms strength is computable from sequence.
"""
import csv, math
from pathlib import Path
import numpy as np
P=Path("data/external_data/promoter_mpra/rlp5Min_SplitVariants.txt")
rng=np.random.default_rng(0)
hdr=None; seqs=[]; el=[]; y=[]
B={"A":0,"C":1,"G":2,"T":3}
with open(P) as f:
    hdr=f.readline().split()
    ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        p=line.split()
        if len(p)<=ix["RNA_exp_average"]: continue
        try: e=float(p[ix["RNA_exp_average"]])
        except: continue
        if e<=0 or not np.isfinite(e): continue
        try: nb=int(p[ix["num_barcodes_integrated"]])
        except: nb=0
        if nb<5: continue
        seqs.append(p[ix["variant"]].upper())
        el.append((p[ix["UP_element"]],p[ix["Minus35"]],p[ix["Spacer"]],p[ix["Minus10"]],p[ix["Background"]]))
        y.append(math.log10(e))
y=np.array(y); n=len(y)
print(f"variants: {n}   log10(expr) range [{y.min():.2f}, {y.max():.2f}]")

# ---- Model A: element one-hot ----
cats=[{} for _ in range(5)]
for row in el:
    for k,v in enumerate(row): cats[k].setdefault(v,len(cats[k]))
def onehot_el(row):
    v=[]
    for k in range(5):
        z=np.zeros(len(cats[k])); z[cats[k][row[k]]]=1; v.append(z)
    return np.concatenate(v)
XA=np.array([onehot_el(r) for r in el])
# ---- Model B: sequence one-hot ----
L=min(len(s) for s in seqs);
def onehot_seq(s):
    z=np.zeros((L,4))
    for i in range(L):
        b=B.get(s[i],-1)
        if b>=0: z[i,b]=1
    return z.ravel()
XB=np.array([onehot_seq(s) for s in seqs])
print(f"feature dims: elements={XA.shape[1]}  sequence={XB.shape[1]} (L={L})")

def ridge(Xtr,ytr,Xte,l2=1.0):
    mu=Xtr.mean(0); Xc=Xtr-mu; yc=ytr-ytr.mean()
    A=Xc.T@Xc+l2*np.eye(Xc.shape[1]); w=np.linalg.solve(A,Xc.T@yc)
    return (Xte-mu)@w+ytr.mean()
def r2(pred,true):
    ss=((true-pred)**2).sum(); st=((true-true.mean())**2).sum(); return 1-ss/st
idx=rng.permutation(n); cut=int(0.8*n); tr,te=idx[:cut],idx[cut:]
pA=ridge(XA[tr],y[tr],XA[te]); pB=ridge(XB[tr],y[tr],XB[te],l2=2.0)
print(f"\n(A) element log-linear model:   held-out R^2 = {r2(pA,y[te]):.3f}")
print(f"(B) sequence one-hot ridge:     held-out R^2 = {r2(pB,y[te]):.3f}")

# ---- Model C: sequence MLP ----
try:
    import torch, torch.nn as nn
    torch.manual_seed(0)
    Xt=torch.tensor(XB,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    mu=Xt[tr].mean(0); sd=Xt[tr].std(0)+1e-6
    Xn=(Xt-mu)/sd
    net=nn.Sequential(nn.Linear(XB.shape[1],256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4); lf=nn.MSELoss()
    tri=torch.tensor(tr);
    for ep in range(60):
        net.train(); perm=tri[torch.randperm(len(tri))]
        for i in range(0,len(perm),256):
            j=perm[i:i+256]; opt.zero_grad(); l=lf(net(Xn[j]).squeeze(-1),yt[j]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pC=net(Xn[torch.tensor(te)]).squeeze(-1).numpy()
    print(f"(C) sequence MLP:               held-out R^2 = {r2(pC,y[te]):.3f}")
except Exception as ex:
    print("(C) MLP skipped:",ex)

import json
json.dump(dict(n=n,r2_element=round(float(r2(pA,y[te])),3),r2_seq_ridge=round(float(r2(pB,y[te])),3)),
          open("outputs/orphan/promoter_strength_solve.json","w"),indent=2)
print("\nwrote outputs/orphan/promoter_strength_solve.json")
