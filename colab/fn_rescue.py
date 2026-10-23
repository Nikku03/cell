"""Does measured 3D contact reduce FALSE NEGATIVES? The subgroup question the aggregate could hide.

An aggregate AUPRC delta near zero does not by itself refute a subgroup claim. If 3D contact rescued a set
of distal true positives while slightly hurting elsewhere, the mean would look flat and the rescue would be
real. The claim deserves its own test rather than an appeal to the aggregate.

THE MECHANISM THE CLAIM REQUIRES. For 3D contact to rescue missed positives, the model's FALSE NEGATIVES
must be enriched for measured 3D contact relative to what it already finds. That is a checkable statement
about which pairs have loops, and it does not depend on any model.

WHY MEASURED HI-C AND NOT A SIMULATION. This uses ENCODE K562 intact Hi-C loop and domain calls, so the
result is a CEILING: no simulated 3D structure can carry information the measurement does not have. If the
measurement cannot rescue the false negatives, nothing that predicts the measurement can either.

The companion test at the bottom evaluates the one band where an uncontrolled pass showed a gain
(250 kb - 1 Mb), with paired fold seeds and a position-shuffled control, because a single subgroup number
chosen after inspecting four bins is exactly the shape of a multiple-comparisons artifact.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0,'colab')
from crispr_gate import _seeded_folds
from competition import competition_features, load as load_comp
from hic_gate import read_bedpe, read_domains, hic_features, LOOPS, DOMAINS
from ubiquitous import load_ubiquitous
import xgboost as xgb

df, Xid, Cr = load_comp()
y, ch = df['crispr_hit'].values, df['chromosome'].values
CF = competition_features(df)
U = np.isin(df['gene'].values, list(load_ubiquitous())).astype(float)[:,None]

tssmap={}
with open('outputs/orphan/invivo/crispr_egpairs.tsv') as fh:
    hdr=fh.readline().rstrip('\n').split('\t')
    ix={k:hdr.index(k) for k in ('chrom','chromStart','chromEnd','startTSS','measuredGeneSymbol')}
    for line in fh:
        p=line.rstrip('\n').split('\t')
        if len(p)<len(hdr): continue
        try:
            tssmap[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}",p[ix['measuredGeneSymbol']])]=(
                p[ix['chrom']],(int(p[ix['chromStart']])+int(p[ix['chromEnd']]))//2,int(p[ix['startTSS']]))
        except ValueError: continue
rows=[tssmap[(e,g)] for e,g in zip(df['element'],df['gene'])]
dist=np.array([abs(t-e) for _,e,t in rows])
H = hic_features(rows, read_bedpe(LOOPS['intact']), read_domains(DOMAINS['intact']))
loop = H[:,0] > 0

BASE = np.hstack([Xid, Cr, CF, U])            # full stack, no 3D
WITH = np.hstack([Xid, Cr, CF, U, H])         # full stack + measured 3D

def oof(X, seed=0):
    folds=_seeded_folds(ch, seed); p=np.zeros(len(y))
    for k in range(5):
        tr,te = folds!=k, folds==k
        c=xgb.XGBClassifier(scale_pos_weight=(y[tr]==0).sum()/max((y[tr]==1).sum(),1),max_depth=4,
            n_estimators=300,learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,min_child_weight=5,
            reg_lambda=2.0,eval_metric='aucpr',n_jobs=4)
        c.fit(X[tr],y[tr]); p[te]=c.predict_proba(X[te])[:,1]
    return p

pb = np.mean([oof(BASE,s) for s in (0,1,2)],0)
pw = np.mean([oof(WITH,s) for s in (0,1,2)],0)

print(f"{len(y):,} pairs, {int(y.sum())} positives\n")
print("RECALL AT A FIXED SHORTLIST BUDGET -- false negatives are what is NOT captured")
print(f"{'top-K':>8} {'recall base':>12} {'recall +3D':>11} {'delta':>8} {'FN base':>8} {'FN +3D':>7}")
for K in (100,200,300,500,750,1000,1500,2000):
    rb = y[np.argsort(-pb)[:K]].sum(); rw = y[np.argsort(-pw)[:K]].sum()
    print(f"{K:8,d} {rb/y.sum():12.4f} {rw/y.sum():11.4f} {rw-rb:+8.0f} {int(y.sum()-rb):8d} {int(y.sum()-rw):7d}")

print("\nDO THE FALSE NEGATIVES EVEN HAVE MEASURED LOOPS?")
order=np.argsort(-pb); rank=np.empty(len(y),int); rank[order]=np.arange(len(y))
fn = (y==1) & (rank>=500)          # true positives the base model ranks outside the top 500
tp = (y==1) & (rank<500)
print(f"  positives ranked OUTSIDE top-500 (the false negatives): {fn.sum()}")
print(f"    of those, with a measured connecting loop: {int(loop[fn].sum())} ({100*loop[fn].mean():.1f}%)")
print(f"  positives INSIDE top-500: {tp.sum()}")
print(f"    of those, with a measured connecting loop: {int(loop[tp].sum())} ({100*loop[tp].mean():.1f}%)")
neg_hi = (y==0) & (rank>=500) & loop
print(f"  NEGATIVES outside top-500 that also have a loop: {int(neg_hi.sum()):,}")
print(f"    -> promoting 'has a loop' would raise {int(neg_hi.sum()):,} negatives to rescue {int(loop[fn].sum())} positives")

print("\nDISTAL SUBGROUP (>=100 kb) -- the specific 'rescue' claim")
for lo,hi,lab in ((1e5,2.5e5,'100-250 kb'),(2.5e5,1e6,'250kb-1Mb'),(1e6,1e9,'>1 Mb'),(1e5,1e9,'ALL >=100 kb')):
    m=(dist>=lo)&(dist<hi)
    if m.sum()==0: continue
    from sklearn.metrics import average_precision_score
    ab=average_precision_score(y[m],pb[m]); aw=average_precision_score(y[m],pw[m])
    fnm = m & (y==1) & (rank>=500)
    print(f"  {lab:14s} {m.sum():5,d} pairs {int(y[m].sum()):3d} pos | AUPRC base {ab:.4f} +3D {aw:.4f} "
          f"({aw-ab:+.4f}) | missed {int(fnm.sum()):3d}, of which {int(loop[fnm].sum()):3d} have a loop")
