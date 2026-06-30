"""Can we COMPUTE a global TF's combinatorial targets via the coincidence (AND-gate)?
For (global G, specific co-regulator S) pairs sharing targets, predict the SHARED
targets with: G-motif alone | S-motif alone | coincidence z(G)+z(S). Leak-free
(PWMs learned excluding the shared eval set). If coincidence >> G alone, the
global TF's functional sites are computable by anchoring on the specific partner.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); rng=np.random.default_rng(0)
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
U={g:ups(g) for g in genes}; allg=[g for g in genes if U[g] is not None and len(U[g])>=W and (U[g]>=0).all()]
def em(seqs,n_iter=20,restarts=2):
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
        s=np.zeros(n)
        for j in range(W): s+=lo[j,a[j:j+n]]
        bv=max(bv,s.max())
    return float(bv)
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if len(pos)<3 or len(neg)<3: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
def zc(d):
    v=np.array(list(d.values()),float); mu=v.mean(); sd=v.std() or 1
    return {k:(x-mu)/sd for k,x in d.items()}

size={t:len([x for x in tf_t[t] if x in genes]) for t in tf_t}
globals_=[t for t in tf_t if size.get(t,0)>=80]
specifics=[t for t in tf_t if 12<=size.get(t,0)<=60]
rows=[]
for Gtf in globals_:
    TG=set(x for x in tf_t[Gtf] if x in genes and U[x] is not None)
    for Stf in specifics:
        if Stf==Gtf: continue
        TS=set(x for x in tf_t[Stf] if x in genes and U[x] is not None)
        shared=TG&TS
        if len(shared)<6: continue
        trG=[x for x in TG-shared]; trS=[x for x in TS-shared]
        if len(trG)<10 or len(trS)<8: continue
        pG=em([U[x] for x in trG]); pS=em([U[x] for x in trS])
        if pG is None or pS is None: continue
        loG=np.log(pG+1e-9)-np.log(bg+1e-9); loS=np.log(pS+1e-9)-np.log(bg+1e-9)
        pos=list(shared); negs=list(rng.choice([g for g in allg if g not in TG and g not in TS],size=200,replace=False))
        cand=pos+negs
        sg={g:score(U[g],loG) for g in cand}; ss={g:score(U[g],loS) for g in cand}
        zg=zc(sg); zs=zc(ss)
        aG=auc([sg[g] for g in pos],[sg[g] for g in negs])
        aS=auc([ss[g] for g in pos],[ss[g] for g in negs])
        aC=auc([zg[g]+zs[g] for g in pos],[zg[g]+zs[g] for g in negs])
        if None in (aG,aS,aC): continue
        rows.append((Gtf,Stf,len(shared),aG,aS,aC))
print(f"(global,specific) pairs tested: {len(rows)}\n")
print(f"{'global':<8}{'specific':<9}{'shared':>7}{'G_alone':>9}{'S_alone':>9}{'coincid':>9}")
for r in sorted(rows,key=lambda x:-x[5])[:18]:
    print(f"  {r[0]:<7}{r[1]:<8}{r[2]:>7}{r[3]:>9.3f}{r[4]:>9.3f}{r[5]:>9.3f}")
import numpy as np
mG=np.mean([r[3] for r in rows]); mS=np.mean([r[4] for r in rows]); mC=np.mean([r[5] for r in rows])
print(f"\nMEAN  global-alone={mG:.3f}   specific-alone={mS:.3f}   COINCIDENCE={mC:.3f}")
print(f"coincidence beats global-alone in {sum(1 for r in rows if r[5]>r[3])}/{len(rows)} pairs")
print(f"coincidence beats specific-alone in {sum(1 for r in rows if r[5]>r[4])}/{len(rows)} pairs")
import json
json.dump(dict(n_pairs=len(rows),mean_global=round(float(mG),3),mean_specific=round(float(mS),3),
               mean_coincidence=round(float(mC),3)),open("outputs/orphan/coincidence_anchor.json","w"),indent=2)
print("\nwrote outputs/orphan/coincidence_anchor.json")
