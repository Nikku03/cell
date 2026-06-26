"""Ceiling test: does the REAL TF network (RegulonDB) improve essentiality
prediction when wired alongside W1 + W2? E. coli only (real network, no proxy).

Features per Keio gene:
  W1  : transformer prediction
  W2  : leak-clean conservation rate
  regulatory STRUCTURE (no truth leak):
    is_tf, log(out_degree), log(in_degree), n_activators, n_repressors,
    is_autoregulated, regulated_by_global_hub, is_sigma_target
Models compared via 5-fold CV (out-of-fold predictions):
  A = logistic(W1,W2)
  B = logistic(W1,W2 + regulatory features)
Metrics: essCov & neCov @ P>=0.90, AUC. Plus regulatory-only AUC.
"""
import csv, re, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(0)

# ---- numpy logistic (CV) ----
class Scaler:
    def fit(self,X): self.mu=X.mean(0); self.sd=X.std(0); self.sd[self.sd<1e-9]=1; return self
    def tf(self,X): return (X-self.mu)/self.sd
def fit_logit(X,y,iters=500,C=1.0):
    n,d=X.shape; Xb=np.c_[X,np.ones(n)]; w=np.zeros(d+1)
    p1=y.mean(); cw=np.where(y==1,0.5/max(p1,1e-6),0.5/max(1-p1,1e-6)); lr=0.1
    for it in range(iters):
        z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z))
        w-=lr*(Xb.T@((p-y)*cw)/n+(w/C)*np.r_[np.ones(d),0])
        if it%150==149: lr*=0.5
    return w
def pred_logit(X,w):
    Xb=np.c_[X,np.ones(len(X))]; return 1/(1+np.exp(-np.clip(Xb@w,-30,30)))

# ---- E. coli data ----
ALL=list(csv.DictReader(open("data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"]
       for r in csv.DictReader(open("data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
truth={r["locus_tag"]:int(r["essential"]) for r in ALL if r["organism"]=="beril_Keio"}
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True)
_cl,_lt,_p=P["clade"],P["lt"],P["p"]
kclade=clade_of.get("beril_Keio","escherichia")
w1={str(_lt[i]):float(_p[i]) for i in range(len(_p)) if str(_cl[i])==kclade and not np.isnan(_p[i])}
def cons_rate(held):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==held: continue
        og=og_of.get((r["organism"],r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og:e[og]/n[og] for og in n if n[og]>=5}
rate=cons_rate(kclade)

# ---- RegulonDB network -> per-gene structural features ----
sym2b={}
for l in open(REG/"GeneProductSet.txt"):
    if l.strip() and not l.startswith("#"):
        p=l.rstrip("\n").split("\t")
        if len(p)>=3 and p[2].startswith("b") and p[1].strip(): sym2b[p[1].strip().lower()]=p[2]
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("data/drive_import/labels/keio_to_bnumber.csv"))}
def name_to_keio(nm):
    b=sym2b.get(nm.lower()); return b_to_keio.get(b) if b else None

out_deg=Counter(); in_deg=Counter(); n_act=Counter(); n_rep=Counter()
autoreg=set(); reg_by=defaultdict(set); is_tf=set()
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf,tgt,eff=p[0].lower(),p[1].lower(),p[2].lower()
    out_deg[tf]+=1; in_deg[tgt]+=1; reg_by[tgt].add(tf); is_tf.add(tf)
    if eff in ("activator","dual"): n_act[tgt]+=1
    if eff in ("repressor","dual"): n_rep[tgt]+=1
    if tf==tgt: autoreg.add(tf)
GLOBAL=set([t for t,_ in out_deg.most_common(15)])
sigma_targets=set()
for l in open(REG/"network_sigma_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    sigma_targets.add(l.split("\t")[1].lower())

# ---- assemble per-gene rows (Keio genes with W1+W2+truth) ----
# map keio_locus <- gene name via reverse of name_to_keio
keio_to_name={}
for nm in sym2b:
    kl=name_to_keio(nm)
    if kl: keio_to_name[kl]=nm

rows=[]
for lt,tv in truth.items():
    w1v=w1.get(lt); og=og_of.get(("beril_Keio",lt)); w2v=rate.get(og) if og else None
    if w1v is None or w2v is None: continue
    nm=keio_to_name.get(lt)   # gene name for regulatory lookup (may be None)
    od=out_deg.get(nm,0) if nm else 0
    idg=in_deg.get(nm,0) if nm else 0
    rows.append(dict(lt=lt, y=tv, w1=w1v, w2=w2v,
        is_tf=int(nm in is_tf) if nm else 0,
        log_out=np.log1p(od), log_in=np.log1p(idg),
        n_act=(n_act.get(nm,0) if nm else 0), n_rep=(n_rep.get(nm,0) if nm else 0),
        autoreg=int(nm in autoreg) if nm else 0,
        reg_by_global=int(bool(reg_by.get(nm,set()) & GLOBAL)) if nm else 0,
        sigma=int(nm in sigma_targets) if nm else 0))
print(f"E. coli genes with W1+W2+truth: {len(rows)}")
mapped=sum(1 for r in rows if r["is_tf"] or r["log_in"]>0)
print(f"  with a regulatory annotation (in RegulonDB): {sum(1 for r in rows if keio_to_name.get(r['lt']) in reg_by or keio_to_name.get(r['lt']) in is_tf)}")

y=np.array([r["y"] for r in rows])
base_feats=np.array([[r["w1"],r["w2"]] for r in rows])
reg_feats=np.array([[r["is_tf"],r["log_out"],r["log_in"],r["n_act"],r["n_rep"],
                     r["autoreg"],r["reg_by_global"],r["sigma"]] for r in rows])
full_feats=np.concatenate([base_feats,reg_feats],axis=1)

def auc(s,yy):
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); pos=yy==1
    if pos.sum()==0 or (~pos).sum()==0: return 0.5
    return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*(~pos).sum()))
def cov_at_p(s,yy,target=0.90,ne=False,min_calls=10):
    o=np.argsort(s) if ne else np.argsort(-s); yo=yy[o]; n=np.arange(1,len(yo)+1)
    cls=(yo==0) if ne else (yo==1); tp=np.cumsum(cls); prec=tp/n
    pos=(yy==(0 if ne else 1)).sum(); valid=(prec>=target)&(n>=min_calls)
    if not valid.any(): return (0.0,0.0,0)
    k=np.max(np.where(valid)[0]); return (float(prec[k]),float(tp[k]/max(pos,1)),int(n[k]))

# 5-fold CV out-of-fold predictions
def cv_predict(X):
    idx=np.arange(len(X)); rng.shuffle(idx); folds=np.array_split(idx,5)
    oof=np.zeros(len(X))
    for f in folds:
        tr=np.setdiff1d(idx,f)
        sc=Scaler().fit(X[tr]); w=fit_logit(sc.tf(X[tr]),y[tr])
        oof[f]=pred_logit(sc.tf(X[f]),w)
    return oof

predA=cv_predict(base_feats)     # W1+W2
predB=cv_predict(full_feats)     # W1+W2+regulatory

# regulatory-only AUC (do reg features alone predict essentiality?)
predR=cv_predict(reg_feats)

print(f"\n=== AUC (5-fold out-of-fold) ===")
print(f"  W1+W2            : {auc(predA,y):.3f}")
print(f"  W1+W2+regulatory : {auc(predB,y):.3f}")
print(f"  regulatory-only  : {auc(predR,y):.3f}")

for ne,label in [(False,"essential"),(True,"non-essential")]:
    pA,cA,nA=cov_at_p(predA,y,0.90,ne=ne)
    pB,cB,nB=cov_at_p(predB,y,0.90,ne=ne)
    print(f"\n  {label} coverage @ P>=0.90:")
    print(f"    W1+W2            : cov={cA:.3f} (P={pA:.3f}, n={nA})")
    print(f"    W1+W2+regulatory : cov={cB:.3f} (P={pB:.3f}, n={nB})   delta={ (cB-cA)*100:+.1f}pp")

# feature importance of regulatory block (coef on full standardized fit)
scF=Scaler().fit(full_feats); wF=fit_logit(scF.tf(full_feats),y)
names=["w1","w2","is_tf","log_out","log_in","n_act","n_rep","autoreg","reg_by_global","sigma"]
imp=sorted(zip(names,wF[:-1]),key=lambda kv:-abs(kv[1]))
print(f"\n  feature weights (standardized):")
for nme,wt in imp: print(f"    {nme:<16}{wt:+.3f}")

eA=cov_at_p(predA,y,0.90,ne=False); eB=cov_at_p(predB,y,0.90,ne=False)
nA_=cov_at_p(predA,y,0.90,ne=True); nB_=cov_at_p(predB,y,0.90,ne=True)
json.dump(dict(n_genes=len(rows),
   auc_w1w2=auc(predA,y), auc_w1w2_reg=auc(predB,y), auc_reg_only=auc(predR,y),
   essCov_w1w2=eA[1], essCov_w1w2_reg=eB[1], essCov_delta_pp=(eB[1]-eA[1])*100,
   neCov_w1w2=nA_[1], neCov_w1w2_reg=nB_[1], neCov_delta_pp=(nB_[1]-nA_[1])*100,
   feature_weights=dict(imp)), open(OUT/"tf_into_wheel2.json","w"), indent=2)
print("\nwrote outputs/orphan/tf_into_wheel2.json")
