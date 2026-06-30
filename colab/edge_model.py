"""Learned regulatory-EDGE model: fuse co-expression + co-fitness + adjacency +
operon-proximity + expression level into a supervised TF->target predictor.
Leave-TF-out CV. Beat the co-expression-alone baseline (0.626)?
"""
import csv, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
rng=np.random.default_rng(0)
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
# expression
bs=[];rows=[]
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: bs.append(line[0]); rows.append([float(x) for x in line[1:]])
X=np.array(rows); bidx={b:i for i,b in enumerate(bs)}
Z=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-9); ns=X.shape[1]
exprmean={b:float(X[bidx[b]].mean()) for b in bs}
# names / coords
name2b={}; coord={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit():
        b=name2b.get(p[1].lower())
        if b: coord[b]=((int(p[3])+int(p[4]))//2, 1 if p[5]=="forward" else -1)
# cofit (Keio b-space)
kb={l:sn for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
cofit=defaultdict(dict)
for l,h,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    if l in kb and h in kb: cofit[kb[l]][kb[h]]=abs(cf)
# edges
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())

def feats(tfb,gb):
    co=float((Z[bidx[tfb]]@Z[bidx[gb]])/ns)
    cf=cofit.get(tfb,{}).get(gb,0.0)
    d=99; ss=0
    if tfb in coord and gb in coord:
        d=abs(coord[tfb][0]-coord[gb][0])/1e6; ss=1 if (d<0.005 and coord[tfb][1]==coord[gb][1]) else 0
    return [abs(co),co,cf,1/(1+d*50),ss,exprmean.get(gb,0)]
FN=["coexp_abs","coexp_signed","cofit","adjacency","operon_near","gene_expr"]

allb=[b for b in bs]
TFs=[tf for tf in tf_t if name2b.get(tf) in bidx and len([t for t in tf_t[tf] if name2b.get(t) in bidx])>=10]
data=[]  # (tf, gb, label, feats)
for tf in TFs:
    tfb=name2b[tf]; tg=set(name2b.get(t) for t in tf_t[tf]) & set(allb); tg.discard(tfb)
    negs=list(rng.choice([b for b in allb if b not in tg and b!=tfb],size=min(len(tg)*5,300),replace=False))
    for gb in list(tg)+negs:
        data.append((tf,gb,1 if gb in tg else 0,feats(tfb,gb)))
print(f"TFs {len(TFs)}, pairs {len(data)}")
def logit(X,y,l2=1.0,it=400,lr=0.2):
    X=np.asarray(X,float); mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
    Xn=np.hstack([Xn,np.ones((len(Xn),1))]); w=np.zeros(Xn.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xn@w)); g=Xn.T@(p-y)/len(y)+l2*np.r_[w[:-1],0]/len(y); w-=lr*g
    return (mu,sd,w)
def pred(m,X):
    mu,sd,w=m; X=np.asarray(X,float); Xn=(X-mu)/sd; Xn=np.hstack([Xn,np.ones((len(Xn),1))]); return 1/(1+np.exp(-Xn@w))
def auc(p,y):
    p=np.asarray(p);y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return None
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
# leave-TF-out
def run(cols):
    idx=[FN.index(c) for c in cols]; res=[]
    for hold in TFs:
        tr=[(d[3],d[2]) for d in data if d[0]!=hold]; te=[(d[3],d[2]) for d in data if d[0]==hold]
        if sum(l for _,l in te)<3: continue
        m=logit([[f[i] for i in idx] for f,_ in tr],np.array([l for _,l in tr]))
        a=auc(pred(m,[[f[i] for i in idx] for f,_ in te]),np.array([l for _,l in te]))
        if a is not None: res.append(a)
    return round(float(np.mean(res)),3),len(res)
import numpy as np
print("\n=== leave-TF-out edge AUC ===")
for name,cols in [("coexp only",["coexp_abs"]),("cofit only",["cofit"]),
                  ("coexp+cofit",["coexp_abs","cofit"]),
                  ("ALL features",FN)]:
    a,n=run(cols); print(f"  {name:<16} AUC={a}  (n={n} TFs)")
import json
a_all,_=run(FN); a_co,_=run(["coexp_abs"])
json.dump(dict(coexp=a_co,all_features=a_all),open("outputs/orphan/edge_model.json","w"),indent=2)
print("\nwrote outputs/orphan/edge_model.json")
