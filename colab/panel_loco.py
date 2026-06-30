"""Panel-wide leave-one-clade-out essentiality: the universality proof.
Across 59 bacteria, predict essentiality from leak-free conservation (OG essential
rate computed EXCLUDING the same clade). Report per-organism AUC and the headline
coverage: what fraction of the genome can we call essential/non-essential at
precision >= 0.90, universally.
"""
import csv
from collections import defaultdict
import numpy as np
L="data/drive_import/labels/"
ess={};
for r in csv.DictReader(open(L+"labels.csv")):
    ess[(r["organism"],r["locus_tag"])]=int(r["essential"])
score={}
for r in csv.DictReader(open(L+"orthology_features.csv")):
    k=(r["organism"],r["locus_tag"]); v=r.get("family_frac_essential_leakfree","")
    if v not in ("","NA",None):
        try: score[k]=float(v)
        except: pass
# pooled arrays
pairs=[(k,ess[k],score[k]) for k in ess if k in score]
y=np.array([p[1] for p in pairs]); s=np.array([p[2] for p in pairs])
print(f"genes with truth + leak-free conservation score: {len(pairs)}  essential rate {y.mean():.2%}")
def auc(sc,yy):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(len(sc)); m=yy==1
    if m.sum()==0 or m.sum()==len(yy): return None
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
print(f"pooled AUC (leak-free conservation -> essential): {auc(s,y):.3f}")
# per-organism AUC
byorg=defaultdict(list)
for (o,l),yy,sc in pairs: byorg[o].append((yy,sc))
aucs=[]
for o,v in byorg.items():
    yy=np.array([a for a,b in v]); sc=np.array([b for a,b in v])
    a=auc(sc,yy)
    if a is not None: aucs.append((o,a,len(v)))
print(f"mean per-organism AUC: {np.mean([a for _,a,_ in aucs]):.3f}  (n={len(aucs)} orgs)")

# ---- coverage at precision >= 0.90 (the headline number) ----
def cov_at_p(s,y,target=0.90):
    order=np.argsort(-s)  # high score first = essential calls
    # ESSENTIAL coverage: largest top-set with precision>=target
    ys=y[order]; cum=np.cumsum(ys); n=np.arange(1,len(ys)+1); prec=cum/n
    ess_n=0
    for i in range(len(prec)-1,-1,-1):
        if prec[i]>=target: ess_n=i+1; break
    essCov=ess_n/len(y)
    # NON-ESS coverage: lowest scores, precision of calling non-ess
    order2=np.argsort(s); yl=y[order2]; cum2=np.cumsum(1-yl); n2=np.arange(1,len(yl)+1); prec2=cum2/n2
    ne_n=0
    for i in range(len(prec2)-1,-1,-1):
        if prec2[i]>=target: ne_n=i+1; break
    neCov=ne_n/len(y)
    return essCov,neCov
essCov,neCov=cov_at_p(s,y,0.90)
print(f"\n=== universal coverage at precision >= 0.90 (leave-clade-out) ===")
print(f"  essential-callable : {100*essCov:.0f}%  (high-conservation genes, precision>=0.90)")
print(f"  non-ess-callable   : {100*neCov:.0f}%")
print(f"  TOTAL genome callable at >=90% precision: {100*(essCov+neCov):.0f}%")
print(f"  remaining (ambiguous/conditional/orphan): {100*(1-essCov-neCov):.0f}%")
import json
json.dump(dict(n=len(pairs),pooled_auc=round(auc(s,y),3),
               mean_org_auc=round(float(np.mean([a for _,a,_ in aucs])),3),
               essCov=round(essCov,3),neCov=round(neCov,3),total_cov=round(essCov+neCov,3)),
          open("outputs/orphan/panel_loco.json","w"),indent=2)
print("\nwrote outputs/orphan/panel_loco.json")
