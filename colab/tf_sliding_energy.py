"""Test the literal 'TF slides on DNA' energy equation:
   E(x) = E_groove(fixed) + sum_j  hbond(base_j)      [A-T=2, G-C=3 H-bonds]
   dwell tau(x) ~ exp(-E/kT)   ('slowing effect')
vs the learned POSITION-SPECIFIC energy matrix (= PWM). Which part discriminates?
On E. coli / RegulonDB: AUC of target-upstream vs non-target-upstream.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); rng=np.random.default_rng(4)
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
U={g:ups(g) for g in genes}
allg=[g for g in genes if U[g] is not None and len(U[g])>=W and (U[g]>=0).all()]
HB=np.array([2,3,3,2])  # A,C,G,T -> H-bonds (A-T=2, G-C=3); E_groove cancels in ranking
GROOVE=5.0
def hbond_score(arr):  # MODEL A: fixed groove + sum of H-bonds over best W window (max over strands)
    best=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        e=HB[a]; csum=np.convolve(e,np.ones(W,int),'valid')
        best=max(best,(GROOVE+csum).max())
    return float(best)
def em(seqs,n_iter=22,restarts=3):
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
def pwm_score(arr,lo):  # MODEL B: position-specific energy (= PWM)
    bv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
def auc(pos,neg):
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

print(f"{'TF':<8}{'modelA_hbond':>14}{'modelB_PWM':>12}{'dwell=expE':>12}")
rows=[]
for tf in ["gntr","trpr","lexa","purr","crp","fnr","arca"]:
    ts=[x for x in tf_t[tf] if x in genes and U[x] is not None]
    if len(ts)<10: continue
    rng.shuffle(ts); h=len(ts)//2; tr,te=ts[:h],ts[h:]
    pwm=em([U[x] for x in tr]); lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    negs=list(rng.choice([g for g in allg if g not in set(ts)],size=200,replace=False))
    # Model A
    aA=auc([hbond_score(U[x]) for x in te],[hbond_score(U[n]) for n in negs])
    # Model B
    posB=[pwm_score(U[x],lo) for x in te]; negB=[pwm_score(U[n],lo) for n in negs]
    aB=auc(posB,negB)
    # dwell = exp(E/kT) on Model B energies -> monotonic, identical AUC
    aDw=auc([np.exp(v/2) for v in posB],[np.exp(v/2) for v in negB])
    rows.append((tf,aA,aB,aDw))
    print(f"  {tf:<6}{aA:>14.3f}{aB:>12.3f}{aDw:>12.3f}")
import numpy as np
print(f"\nMEAN   modelA(H-bond/groove)={np.mean([r[1] for r in rows]):.3f}   "
      f"modelB(PWM)={np.mean([r[2] for r in rows]):.3f}   dwell=expE={np.mean([r[3] for r in rows]):.3f}")
print("\nNote: dwell AUC == PWM AUC exactly (exp is monotonic -> same ranking).")
