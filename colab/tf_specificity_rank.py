"""TF->operator, the specificity-ordered way (user's strategy), on E. coli/RegulonDB.

For each TF: learn its operator (PWM), then scan all gene promoters. Define a
"position to attach to" = any promoter scoring >= the weakest of the TF's own
known sites. N_sites = how many such positions exist genome-wide.
  - few sites  -> specific operator (high info) -> should predict targets well
  - many sites -> degenerate operator            -> the wall
Rank TFs by N_sites ASCENDING; report target-prediction precision/recall at each
level (leak-free: PWM learned on train-half targets, scored on held-out half).
Universal/housekeeping genes: largely sigma70-constitutive (no specific TF) -> the
no-specific-regulator floor; global regulators (CRP/FNR) are the high-N_sites end.
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(4)
B={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([B.get(c,-1) for c in s],dtype=np.int8)
def rc(a): comp=np.array([3,2,1,0,-1]); return comp[a][::-1]
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
gc=(G.count("G")+G.count("C"))/GLEN; bg=np.array([(1-gc)/2,gc/2,gc/2,(1-gc)/2])
genes={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
W=20
def ups(name,pad=200):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
# precompute all promoter encodings
U={g:ups(g) for g in genes}
allg=[g for g in genes if U[g] is not None and len(U[g])>=W and (U[g]>=0).all()]
print(f"genes with usable promoter: {len(allg)}")
def em(seqs,W=20,n_iter=22,restarts=3):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
    if len(seqs)<5: return None
    best=None; bll=-1e18
    for _ in range(restarts):
        s0=seqs[rng.integers(len(seqs))]; st=rng.integers(0,len(s0)-W+1)
        c=np.ones((W,4))*0.5
        for j in range(W): c[j,s0[st+j]]+=1
        pwm=c/c.sum(1,keepdims=True)
        for it in range(n_iter):
            new=np.ones((W,4))*0.25; lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
            for s in seqs:
                ap=[]
                for arr in (s,rc(s)):
                    n=len(arr)-W+1
                    if n<=0: continue
                    sc=np.zeros(n)
                    for j in range(W): sc+=lo[j,arr[j:j+n]]
                    ap.append((arr,sc))
                if not ap: continue
                scs=np.concatenate([x for _,x in ap]); mx=scs.max(); w=np.exp(scs-mx); w/=w.sum(); off=0
                for arr,sc in ap:
                    n=len(sc); ww=w[off:off+n]; off+=n
                    for j in range(W): np.add.at(new[j],arr[j:j+n],ww)
            pwm=new/new.sum(1,keepdims=True)
        lo=np.log(pwm+1e-9)-np.log(bg+1e-9); ll=0
        for s in seqs:
            bv=-1e18
            for arr in (s,rc(s)):
                n=len(arr)-W+1
                if n<=0: continue
                sc=np.zeros(n)
                for j in range(W): sc+=lo[j,arr[j:j+n]]
                bv=max(bv,sc.max())
            ll+=bv
        if ll>bll: bll=ll; best=pwm
    return best
def score(arr,lo):
    bv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
def ic(p): return float(np.sum(p*np.log2((p+1e-9)/0.25)))

TFs=[t for t in tf_t if len([x for x in tf_t[t] if x in genes and U[x] is not None])>=12]
rows=[]
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes and U[x] is not None]
    rng.shuffle(ts); h=len(ts)//2; train,test=ts[:h],ts[h:]
    if len(train)<5 or len(test)<3: continue
    pwm=em([U[x] for x in train])
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    sc={g:score(U[g],lo) for g in allg}
    thr=min(sc[x] for x in train if x in sc)          # operator must match its own known sites
    predicted=set(g for g in allg if sc[g]>=thr)
    nsite=len(predicted)
    alltgt=set(ts)
    rec=len([x for x in test if x in predicted])/len(test)        # held-out recall
    prec=len(predicted & alltgt)/max(nsite,1)                      # precision (all known targets)
    rows.append(dict(tf=t,n_targets=len(ts),n_sites=nsite,ic=round(ic(pwm),1),
                     recall=round(rec,3),precision=round(prec,3)))
rows.sort(key=lambda r:r["n_sites"])   # ascending specificity (fewest first)
print(f"\nTFs ranked by N_sites ascending (most specific operator first):")
print(f"{'rank':>4}{'TF':<9}{'n_tgt':>6}{'N_sites':>8}{'IC':>6}{'recall':>8}{'prec':>7}")
for i,r in enumerate(rows):
    print(f"{i+1:>4} {r['tf']:<8}{r['n_targets']:>6}{r['n_sites']:>8}{r['ic']:>6.1f}{r['recall']:>8.3f}{r['precision']:>7.3f}")

# accuracy vs specificity buckets
def bucket(rs,name):
    if not rs: return
    print(f"  {name:<22} TFs={len(rs):>2}  mean N_sites={np.mean([r['n_sites'] for r in rs]):>6.0f}  recall={np.mean([r['recall'] for r in rs]):.3f}  prec={np.mean([r['precision'] for r in rs]):.3f}")
print("\n=== prediction quality vs operator specificity ===")
bucket([r for r in rows if r['n_sites']<=100],"very specific (<=100)")
bucket([r for r in rows if 100<r['n_sites']<=400],"specific (100-400)")
bucket([r for r in rows if 400<r['n_sites']<=1000],"moderate (400-1000)")
bucket([r for r in rows if r['n_sites']>1000],"degenerate (>1000)")
json.dump(dict(n_tf=len(rows),per_tf=rows),open(OUT/"tf_specificity_rank.json","w"),indent=2)
print(f"\nwrote {OUT}/tf_specificity_rank.json")
