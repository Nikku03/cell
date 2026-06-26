"""Wire feba fitness as Wheel 4 across the panel + TF features into Wheel 2 (Keio).

For each org with W1 + W2 + feba-fitness + truth:
  A = W1+W2            (current 2-wheel)
  B = W1+W2+W4         (+ fitness: absent_from_feba, min_fit_sig)
Compare essCov & neCov @ P>=0.90 (5-fold CV OOF).
For Keio additionally:
  C = W1+W2+W4+TF      (+ RegulonDB structural features)

CAVEAT: panel truth is 'via essential_families' (feba-derived) -> panel Wheel-4
lift is an UPPER BOUND (circular). mtub independent result is the honest anchor.
"""
import sqlite3, csv, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
con=sqlite3.connect("/home/user/cell/data/external_data/feba/feba.db"); cur=con.cursor()
OUT=Path("/home/user/cell/outputs/orphan"); rng=np.random.default_rng(0)

def auc(s,y):
    s=np.asarray(s,float); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return float('nan')
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1; pos=p.sum()
    return float((r[p].sum()-pos*(pos-1)/2)/(pos*(~p).sum()))
def cov(s,y,t=0.90,ne=False,mc=10):
    s=np.asarray(s,float); y=np.asarray(y); o=np.argsort(s) if ne else np.argsort(-s)
    yo=y[o]; n=np.arange(1,len(yo)+1); cls=(yo==0) if ne else (yo==1)
    tp=np.cumsum(cls); prec=tp/n; pos=(y==(0 if ne else 1)).sum(); v=(prec>=t)&(n>=mc)
    if not v.any(): return 0.0
    return float(tp[np.max(np.where(v)[0])]/max(pos,1))
class Sc:
    def fit(s,X): s.mu=X.mean(0); s.sd=np.where(X.std(0)<1e-9,1.0,X.std(0)); return s
    def tf(s,X): return (X-s.mu)/s.sd
def logit(X,y,it=400,C=1.0):
    n,d=X.shape; Xb=np.c_[X,np.ones(n)]; w=np.zeros(d+1)
    p1=y.mean(); cw=np.where(y==1,0.5/max(p1,1e-6),0.5/max(1-p1,1e-6)); lr=0.1
    for i in range(it):
        z=np.clip(Xb@w,-30,30); p=1/(1+np.exp(-z)); w-=lr*(Xb.T@((p-y)*cw)/n+(w/C)*np.r_[np.ones(d),0])
        if i%150==149: lr*=0.5
    return w
def pf(X,w): return 1/(1+np.exp(-np.clip(np.c_[X,np.ones(len(X))]@w,-30,30)))
def cvp(X,y):
    idx=np.arange(len(X)); rng.shuffle(idx); fol=np.array_split(idx,5); oof=np.zeros(len(X))
    for f in fol:
        tr=np.setdiff1d(idx,f); s=Sc().fit(X[tr]); oof[f]=pf(s.tf(X[f]),logit(s.tf(X[tr]),y[tr]))
    return oof

# shared
ALL=list(csv.DictReader(open("/home/user/cell/data/drive_import/labels/labels.csv")))
clade_of={r["organism"]:r["clade"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/clade_splits.csv"))}
og_of={(r["organism"],r["locus_tag"]):r["og_id"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/orthology_features.csv")) if r.get("og_id")}
truth=defaultdict(dict)
for r in ALL: truth[r["organism"]][r["locus_tag"]]=int(r["essential"])
P=np.load(OUT/"af_torch_preds_aug.npz",allow_pickle=True); _cl,_lt,_p=P["clade"],P["lt"],P["p"]
w1all=defaultdict(dict)
for i in range(len(_p)):
    if not np.isnan(_p[i]): w1all[str(_cl[i])][str(_lt[i])]=float(_p[i])
def cons_rate(h):
    n=defaultdict(int); e=defaultdict(int)
    for r in ALL:
        if clade_of.get(r["organism"])==h: continue
        og=og_of.get((r["organism"],r["locus_tag"]))
        if og: n[og]+=1; e[og]+=int(r["essential"])
    return {og:e[og]/n[og] for og in n if n[og]>=5}

fb_orgs={r[0] for r in cur.execute("SELECT orgId FROM Organism")}
bridge=[(o,o.replace("beril_","")) for o in sorted(truth) if o.replace("beril_","") in fb_orgs and o!="mtub"]
print(f"panel orgs (W1+W2+fitness candidates): {len(bridge)}", flush=True)

# fitness features per org
def fit_feats(fb):
    seen=set(); sm={}
    for lid,fit,t in cur.execute("SELECT locusId,fit,t FROM GeneFitness WHERE orgId=?", (fb,)):
        if fit is None: continue
        seen.add(lid)
        if t is not None and abs(t)>4 and (lid not in sm or fit<sm[lid]): sm[lid]=fit
    return seen,sm

# Keio TF features
sym2b={}
for l in open("/home/user/cell/data/external_data/regulon/GeneProductSet.txt"):
    if l.strip() and not l.startswith("#"):
        p=l.rstrip("\n").split("\t")
        if len(p)>=3 and p[2].startswith("b") and p[1].strip(): sym2b[p[1].strip().lower()]=p[2]
b_to_keio={r["b_number"]:r["locus_tag"] for r in csv.DictReader(open("/home/user/cell/data/drive_import/labels/keio_to_bnumber.csv"))}
keio_b_to_name={}
for nm,b in sym2b.items():
    kl=b_to_keio.get(b)
    if kl: keio_b_to_name[kl]=nm
out_deg=Counter(); in_deg=Counter(); n_act=Counter(); n_rep=Counter(); autoreg=set(); reg_by=defaultdict(set); is_tf=set()
for l in open("/home/user/cell/data/external_data/regulon/network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf,tgt,eff=p[0].lower(),p[1].lower(),p[2].lower()
    out_deg[tf]+=1; in_deg[tgt]+=1; reg_by[tgt].add(tf); is_tf.add(tf)
    if eff in("activator","dual"): n_act[tgt]+=1
    if eff in("repressor","dual"): n_rep[tgt]+=1
    if tf==tgt: autoreg.add(tf)
GLOBAL=set([t for t,_ in out_deg.most_common(15)])
sigma_t=set(l.split("\t")[1].lower() for l in open("/home/user/cell/data/external_data/regulon/network_sigma_gene.txt") if l.strip() and not l.startswith("#"))

print(f"\n{'organism':<24}{'n':>6}{'ess':>5}{'essCov_A':>10}{'essCov_B':>10}{'dEss':>7}{'neCov_A':>9}{'neCov_B':>9}{'dNe':>7}", flush=True)
rows=[]; t0=__import__("time").time()
for our,fb in bridge:
    held=clade_of.get(our,"?"); rate=cons_rate(held); W1=w1all.get(held,{})
    seen,sm=fit_feats(fb)
    rec=[]
    for lt,tv in truth[our].items():
        w1v=W1.get(lt); og=og_of.get((our,lt)); w2v=rate.get(og) if og else None
        if w1v is None or w2v is None: continue
        absent=1.0 if lt not in seen else 0.0
        mf=-sm[lt] if lt in sm else 0.0
        rec.append((tv,w1v,w2v,absent,mf))
    if len(rec)<200: continue
    y=np.array([r[0] for r in rec]);
    if y.sum()<10 or (len(y)-y.sum())<10: continue
    A=np.array([[r[1],r[2]] for r in rec]); B=np.array([[r[1],r[2],r[3],r[4]] for r in rec])
    pA=cvp(A,y); pB=cvp(B,y)
    eA,eB=cov(pA,y),cov(pB,y); nA,nB=cov(pA,y,ne=True),cov(pB,y,ne=True)
    rows.append(dict(org=our,n=len(y),ess=int(y.sum()),essCov_A=eA,essCov_B=eB,
                     dEss=eB-eA,neCov_A=nA,neCov_B=nB,dNe=nB-nA))
    print(f"  {our:<22}{len(y):>6}{int(y.sum()):>5}{eA:>10.3f}{eB:>10.3f}{eB-eA:>+7.3f}{nA:>9.3f}{nB:>9.3f}{nB-nA:>+7.3f}", flush=True)

import numpy as np
mEA=np.mean([r["essCov_A"] for r in rows]); mEB=np.mean([r["essCov_B"] for r in rows])
mNA=np.mean([r["neCov_A"] for r in rows]); mNB=np.mean([r["neCov_B"] for r in rows])
print(f"\n=== PANEL AGGREGATE ({len(rows)} orgs) -- CIRCULAR (truth feba-derived), UPPER BOUND ===")
print(f"  essCov@P0.9: W1+W2 {mEA:.3f} -> +W4 {mEB:.3f}  ({(mEB-mEA)*100:+.1f}pp)")
print(f"  neCov@P0.9 : W1+W2 {mNA:.3f} -> +W4 {mNB:.3f}  ({(mNB-mNA)*100:+.1f}pp)")

# Keio: add TF features
print(f"\n=== Keio: TF features into Wheel 2 (real RegulonDB network) ===")
our="beril_Keio"; held=clade_of["beril_Keio"]; rate=cons_rate(held); W1=w1all.get(held,{})
seen,sm=fit_feats("Keio")
rec=[]
for lt,tv in truth[our].items():
    w1v=W1.get(lt); og=og_of.get((our,lt)); w2v=rate.get(og) if og else None
    if w1v is None or w2v is None: continue
    absent=1.0 if lt not in seen else 0.0; mf=-sm[lt] if lt in sm else 0.0
    nm=keio_b_to_name.get(lt)
    tf_feats=[int(nm in is_tf) if nm else 0, np.log1p(out_deg.get(nm,0) if nm else 0),
              np.log1p(in_deg.get(nm,0) if nm else 0), int(nm in autoreg) if nm else 0,
              int(bool(reg_by.get(nm,set())&GLOBAL)) if nm else 0, int(nm in sigma_t) if nm else 0]
    rec.append((tv,w1v,w2v,absent,mf,tf_feats))
y=np.array([r[0] for r in rec])
A=np.array([[r[1],r[2]] for r in rec]); B=np.array([[r[1],r[2],r[3],r[4]] for r in rec])
C=np.array([[r[1],r[2],r[3],r[4]]+r[5] for r in rec])
pA,pB,pC=cvp(A,y),cvp(B,y),cvp(C,y)
print(f"  essCov@P0.9: W1+W2 {cov(pA,y):.3f}  +W4 {cov(pB,y):.3f}  +W4+TF {cov(pC,y):.3f}")
print(f"  neCov@P0.9 : W1+W2 {cov(pA,y,ne=True):.3f}  +W4 {cov(pB,y,ne=True):.3f}  +W4+TF {cov(pC,y,ne=True):.3f}")
print(f"  AUC        : W1+W2 {auc(pA,y):.3f}  +W4 {auc(pB,y):.3f}  +W4+TF {auc(pC,y):.3f}")

json.dump(dict(panel=rows, panel_essCov_A=float(mEA),panel_essCov_B=float(mEB),
   panel_neCov_A=float(mNA),panel_neCov_B=float(mNB),
   keio=dict(essCov_w1w2=cov(pA,y),essCov_w4=cov(pB,y),essCov_w4_tf=cov(pC,y),
             neCov_w1w2=cov(pA,y,ne=True),neCov_w4=cov(pB,y,ne=True),neCov_w4_tf=cov(pC,y,ne=True),
             auc_w1w2=auc(pA,y),auc_w4=auc(pB,y),auc_w4_tf=auc(pC,y)),
   caveat="panel truth feba-derived -> panel W4 lift is upper bound; mtub independent is the honest anchor"),
   open(OUT/"wheel4_panel.json","w"),indent=2)
print(f"\nwrote outputs/orphan/wheel4_panel.json ({__import__('time').time()-t0:.0f}s)")
