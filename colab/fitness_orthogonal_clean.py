"""THE clean decisive test: on mtub (independent DeJesus 2017 truth), does feba
fitness add coverage ORTHOGONAL to cross-organism conservation?

Why this is clean (zero circularity):
  - TRUTH  = DeJesus 2017 MTBseq essentiality (independent of feba).
  - CONSERVATION = for each MycoTube gene, use feba.db's Ortholog table to find
    its orthologs in the OTHER panel organisms, and average those organisms'
    essentiality labels. mtub's own DeJesus labels NEVER enter this.
  - FITNESS = MycoTube fitness features (absent + min_fit, |t|>4).
Both predictors are independent of the DeJesus truth.

Compare (5-fold CV OOF):
  A = conservation only
  B = conservation + fitness
essCov/neCov @ P>=0.90, AUC, and the orthogonality breakdown (does fitness catch
essentials conservation misses?).
"""
import sqlite3, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan"); rng=np.random.default_rng(0)

def auc(s,y):
    s=np.array(s); y=np.array(y)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1; pos=p.sum()
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))
def cov_at_p(s,y,t=0.90,ne=False,mc=10):
    s=np.array(s); y=np.array(y); o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1); cls=(yo==0) if ne else (yo==1)
    tp=np.cumsum(cls); prec=tp/n; pos=(y==(0 if ne else 1)).sum(); v=(prec>=t)&(n>=mc)
    if not v.any(): return (0.0,0.0,0)
    k=np.max(np.where(v)[0]); return (float(prec[k]),float(tp[k]/max(pos,1)),int(n[k]))
class Sc:
    def fit(s,X):
        s.mu=X.mean(0); s.sd=X.std(0); s.sd=np.where(s.sd<1e-9,1.0,s.sd); return s
    def tf(s,X): return (X-s.mu)/s.sd
def logit(X,y,it=600,C=1.0):
    n,d=X.shape; Xb=np.c_[X,np.ones(n)]; w=np.zeros(d+1)
    p1=y.mean(); cw=np.where(y==1,0.5/max(p1,1e-6),0.5/max(1-p1,1e-6)); lr=0.1
    for i in range(it):
        z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z)); w-=lr*(Xb.T@((p-y)*cw)/n+(w/C)*np.r_[np.ones(d),0])
        if i%200==199: lr*=0.5
    return w
def predf(X,w): return 1/(1+np.exp(-np.clip(np.c_[X,np.ones(len(X))]@w,-30,30)))
def cv(X,y):
    idx=np.arange(len(X)); rng.shuffle(idx); folds=np.array_split(idx,5); oof=np.zeros(len(X))
    for f in folds:
        tr=np.setdiff1d(idx,f); s=Sc().fit(X[tr]); w=logit(s.tf(X[tr]),y[tr]); oof[f]=predf(s.tf(X[f]),w)
    return oof

# 1. DeJesus truth
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv"))
       if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}
print(f"mtub DeJesus truth: {len(truth)} ({sum(truth.values())} ess)")

# 2. panel essentiality keyed by feba orgId + locus (from OTHER organisms' labels)
fb_orgs={r[0] for r in cur.execute("SELECT orgId FROM Organism")}
panel=defaultdict(dict)
for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")):
    o=r["organism"]
    if o=="mtub": continue   # never use mtub's own labels
    fb=o.replace("beril_","")
    if fb in fb_orgs:
        panel[fb][r["locus_tag"]]=int(r["essential"])
print(f"panel orgs usable for conservation: {len(panel)}")

# 3. conservation per MycoTube gene = mean essentiality of its orthologs in panel
cons={}; n_orth={}
orth=defaultdict(list)
for o2,l2,l1 in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1='MycoTube'"):
    orth[l1].append((o2,l2))
for g, lst in orth.items():
    vals=[panel[o2][l2] for (o2,l2) in lst if o2 in panel and l2 in panel[o2]]
    if vals: cons[g]=np.mean(vals); n_orth[g]=len(vals)

# 4. fitness features (MycoTube)
any_seen=set(); sig_min={}
for lid,fit,t in cur.execute("SELECT locusId,fit,t FROM GeneFitness WHERE orgId='MycoTube'"):
    if fit is None: continue
    any_seen.add(lid)
    if t is not None and abs(t)>4 and (lid not in sig_min or fit<sig_min[lid]): sig_min[lid]=fit

# 5. assemble: need conservation present (fitness can be absent-coded)
rows=[]
for lt,tv in truth.items():
    if lt not in cons: continue       # require a conservation value
    absent = 1.0 if lt not in any_seen else 0.0
    mf = -sig_min[lt] if lt in sig_min else 0.0
    rows.append((lt,tv,cons[lt],absent,mf))
print(f"mtub genes with conservation+truth: {len(rows)} ({sum(r[1] for r in rows)} ess)")

y=np.array([r[1] for r in rows])
consF=np.array([[r[2]] for r in rows])
fitF=np.array([[r[3],r[4]] for r in rows])
both=np.array([[r[2],r[3],r[4]] for r in rows])

predC=cv(consF,y); predF=cv(fitF,y); predB=cv(both,y)
print(f"\n=== AUC (independent DeJesus truth, 5-fold OOF) ===")
print(f"  conservation only       : {auc(predC,y):.3f}")
print(f"  fitness only            : {auc(predF,y):.3f}")
print(f"  conservation + fitness  : {auc(predB,y):.3f}")

for ne,lab in [(False,"essential"),(True,"non-essential")]:
    pC,cC,nC=cov_at_p(predC,y,0.90,ne=ne); pB,cB,nB=cov_at_p(predB,y,0.90,ne=ne)
    print(f"\n  {lab} cov @ P>=0.90:")
    print(f"    conservation       : cov={cC:.3f} (P={pC:.3f}, n={nC})")
    print(f"    conservation+fitness: cov={cB:.3f} (P={pB:.3f}, n={nB})   delta={(cB-cC)*100:+.1f}pp")

# orthogonality: essentials conservation MISSES (bottom-half conservation), does fitness rank them?
order=np.argsort(predC); missed=order[:len(order)//2]
a_missed=auc(predF[missed], y[missed])
print(f"\n  ORTHOGONALITY: among genes conservation ranks LOW (bottom half), "
      f"{int(y[missed].sum())} are truly essential.")
print(f"    fitness-only AUC on the conservation-missed subset: {a_missed:.3f}")
print(f"    (>0.5 means fitness carries essentiality info conservation lacks)")

s=Sc().fit(both); w=logit(s.tf(both),y)
print(f"\n  weights: conservation={w[0]:+.2f}  absent={w[1]:+.2f}  min_fit={w[2]:+.2f}")

eC=cov_at_p(predC,y,0.90); eB=cov_at_p(predB,y,0.90)
json.dump(dict(n=len(rows),ess=int(y.sum()),
  auc_cons=auc(predC,y),auc_fit=auc(predF,y),auc_both=auc(predB,y),
  essCov_cons=eC[1],essCov_both=eB[1],essCov_delta_pp=(eB[1]-eC[1])*100,
  fitness_auc_on_cons_missed=a_missed,
  weights=dict(conservation=float(w[0]),absent=float(w[1]),min_fit=float(w[2]))),
  open(OUT/"fitness_orthogonal_clean.json","w"),indent=2)
print("\nwrote outputs/orphan/fitness_orthogonal_clean.json")
