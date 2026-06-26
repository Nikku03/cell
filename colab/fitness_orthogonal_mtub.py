"""DECISIVE TEST: does feba.db fitness add coverage ORTHOGONAL to W1+W2?

mtub is the clean case: truth = DeJesus 2017 (INDEPENDENT of feba), and we have
all of W1 (transformer), W2 (leak-clean conservation), and feba MycoTube fitness.

Tests (5-fold CV, out-of-fold predictions):
  A = W1+W2
  B = W1+W2+fitness
Compare essCov & neCov @ P>=0.90 and AUC. Then the orthogonality breakdown:
  - among essentials W1+W2 MISS (low W1+W2 score but truth=1), does fitness
    rank them up?
  - feature weights (is fitness pulling weight or redundant?)

Fitness features (3, from feba MycoTube, no truth leak):
  absent  : gene absent from GeneFitness (no mutants recovered)
  min_fit : -min(fit) over |t|>4 entries (severity of worst significant defect)
  n_severe: count of |t|>4 conditions with fit < -2
"""
import sqlite3, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db")
cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan")
rng=np.random.default_rng(0)

def auc(s,y):
    s=np.array(s); y=np.array(y)
    if (y==1).sum()==0 or (y==0).sum()==0: return float("nan")
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1; pos=p.sum()
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))
def cov_at_p(s,y,target=0.90,ne=False,mc=10):
    s=np.array(s); y=np.array(y); o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1); cls=(yo==0) if ne else (yo==1)
    tp=np.cumsum(cls); prec=tp/n; pos=(y==(0 if ne else 1)).sum()
    v=(prec>=target)&(n>=mc)
    if not v.any(): return (0.0,0.0,0)
    k=np.max(np.where(v)[0]); return (float(prec[k]),float(tp[k]/max(pos,1)),int(n[k]))
class Sc:
    def fit(s,X): s.mu=X.mean(0);s.sd=X.std(0);s.sd[s.sd<1e-9]=1;return s
    def tf(s,X): return (X-s.mu)/s.sd
def logit(X,y,it=600,C=1.0):
    n,d=X.shape; Xb=np.c_[X,np.ones(n)]; w=np.zeros(d+1)
    p1=y.mean(); cw=np.where(y==1,0.5/max(p1,1e-6),0.5/max(1-p1,1e-6)); lr=0.1
    for i in range(it):
        z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z))
        w-=lr*(Xb.T@((p-y)*cw)/n+(w/C)*np.r_[np.ones(d),0])
        if i%200==199: lr*=0.5
    return w
def pred(X,w): return 1/(1+np.exp(-np.clip(np.c_[X,np.ones(len(X))]@w,-30,30)))
def cv(X,y):
    idx=np.arange(len(X)); rng.shuffle(idx); folds=np.array_split(idx,5); oof=np.zeros(len(X))
    for f in folds:
        tr=np.setdiff1d(idx,f); s=Sc().fit(X[tr]); w=logit(s.tf(X[tr]),y[tr]); oof[f]=pred(s.tf(X[f]),w)
    return oof

# ---- mtub data ----
truth={r["locus_tag"]:int(r["essential"]) for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv"))
       if r["organism"]=="mtub" and r["source"]=="DeJesus 2017 (MTBseq)"}
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/clade_splits.csv"))}
held=clade_of.get("mtub","mycobacterium")
og_of={(r["organism"],r["locus_tag"]):r["og_id"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
ALL=list(csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")))
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1={str(_lt[i]):float(_p[i]) for i in range(len(_p)) if str(_cl[i])==held and not np.isnan(_p[i])}
def cons_rate(h):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==h: continue
        og=og_of.get((r["organism"],r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og:e[og]/n[og] for og in n if n[og]>=5}
rate=cons_rate(held)

# fitness features from MycoTube
any_seen=set(); sig_min={}; n_sev=defaultdict(int)
for lid,fit,t in cur.execute("SELECT locusId,fit,t FROM GeneFitness WHERE orgId='MycoTube'"):
    if fit is None: continue
    any_seen.add(lid)
    if t is not None and abs(t)>4:
        if lid not in sig_min or fit<sig_min[lid]: sig_min[lid]=fit
        if fit<-2: n_sev[lid]+=1

# assemble rows with W1+W2+truth all present
rows=[]
for lt,tv in truth.items():
    w1v=w1.get(lt); og=og_of.get(("mtub",lt)); w2v=rate.get(og) if og else None
    if w1v is None or w2v is None: continue
    absent = 1.0 if lt not in any_seen else 0.0
    mf = -sig_min[lt] if lt in sig_min else 0.0
    ns = n_sev.get(lt,0)
    rows.append((lt,tv,w1v,w2v,absent,mf,ns))
print(f"mtub genes with W1+W2+truth: {len(rows)} ({sum(r[1] for r in rows)} essential)")

y=np.array([r[1] for r in rows])
base=np.array([[r[2],r[3]] for r in rows])           # W1,W2
full=np.array([[r[2],r[3],r[4],r[5],r[6]] for r in rows])  # +absent,min_fit,n_severe
fit_only=np.array([[r[4],r[5],r[6]] for r in rows])

predA=cv(base, y)
predB=cv(full, y)
predF=cv(fit_only, y)

print(f"\n=== AUC (5-fold OOF, independent DeJesus truth) ===")
print(f"  W1+W2          : {auc(predA,y):.3f}")
print(f"  W1+W2+fitness  : {auc(predB,y):.3f}")
print(f"  fitness-only   : {auc(predF,y):.3f}")

for ne,lab in [(False,"essential"),(True,"non-essential")]:
    pA,cA,nA=cov_at_p(predA,y,0.90,ne=ne); pB,cB,nB=cov_at_p(predB,y,0.90,ne=ne)
    print(f"\n  {lab} coverage @ P>=0.90:")
    print(f"    W1+W2         : cov={cA:.3f} (P={pA:.3f}, n={nA})")
    print(f"    W1+W2+fitness : cov={cB:.3f} (P={pB:.3f}, n={nB})   delta={(cB-cA)*100:+.1f}pp")

# orthogonality: essentials W1+W2 misses (bottom-half W1+W2 score), does fitness rank up?
order=np.argsort(predA)  # ascending = W1+W2 thinks non-essential
missed = order[:len(order)//2]   # bottom half by W1+W2
y_missed=y[missed]
predF_missed=predF[missed]
a_missed=auc(predF_missed, y_missed)
print(f"\n  ORTHOGONALITY: among genes W1+W2 ranks LOW (bottom half), "
      f"{int(y_missed.sum())} are truly essential.")
print(f"    fitness-only AUC on this W1+W2-missed subset: {a_missed:.3f}")

# feature weights on full standardized fit
s=Sc().fit(full); w=logit(s.tf(full),y)
names=["W1","W2","absent","min_fit","n_severe"]
print(f"\n  feature weights (standardized): " + "  ".join(f"{n}={wt:+.2f}" for n,wt in zip(names,w[:-1])))

eA=cov_at_p(predA,y,0.90); eB=cov_at_p(predB,y,0.90)
nAx=cov_at_p(predA,y,0.90,ne=True); nBx=cov_at_p(predB,y,0.90,ne=True)
json.dump(dict(n=len(rows), ess=int(y.sum()),
   auc_w1w2=auc(predA,y), auc_w1w2_fit=auc(predB,y), auc_fit_only=auc(predF,y),
   essCov_w1w2=eA[1], essCov_w1w2_fit=eB[1], essCov_delta_pp=(eB[1]-eA[1])*100,
   neCov_w1w2=nAx[1], neCov_w1w2_fit=nBx[1], neCov_delta_pp=(nBx[1]-nAx[1])*100,
   fitness_auc_on_w1w2_missed=a_missed,
   feature_weights=dict(zip(names,[float(x) for x in w[:-1]]))),
   open(OUT/"fitness_orthogonal_mtub.json","w"), indent=2)
print("\nwrote outputs/orphan/fitness_orthogonal_mtub.json")
