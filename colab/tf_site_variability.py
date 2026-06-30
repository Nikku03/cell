"""Are the 1000+ sites of a degenerate TF the SAME sequence? Extract the actual
matched site sequences for degenerate (CRP,FNR) vs specific (gntR,trpR) TFs and
measure their variability.
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon")
rng=np.random.default_rng(4)
B={"A":0,"C":1,"G":2,"T":3}; INV="ACGT"
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
def em(seqs,n_iter=25,restarts=3):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
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
def best_hit(arr,lo):
    bv=-1e18; bs=None
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        k=int(sc.argmax())
        if sc[k]>bv: bv=sc[k]; bs=a[k:k+W]
    return bv,bs
def ic(p): return float(np.sum(p*np.log2((p+1e-9)/0.25)))
def consensus(p): return "".join(INV[i] for i in p.argmax(1))

for tf in ["gntr","trpr","crp","fnr"]:
    ts=[x for x in tf_t[tf] if x in genes and U[x] is not None]
    pwm=em([U[x] for x in ts])
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    thr=min(best_hit(U[x],lo)[0] for x in ts)
    sites=[]
    for g in allg:
        v,s=best_hit(U[g],lo)
        if v>=thr and s is not None: sites.append("".join(INV[b] for b in s))
    cons=consensus(pwm)
    n=len(sites); distinct=len(set(sites))
    exact=sum(1 for s in sites if s==cons)
    # mean per-position identity to consensus
    arr=np.array([[B[c] for c in s] for s in sites])
    consarr=np.array([B[c] for c in cons])
    perpos_match=(arr==consarr).mean(0)            # fraction matching consensus at each position
    mean_id=float((arr==consarr).mean())
    print(f"\n=== {tf.upper()}  (info {ic(pwm):.1f} bits, {n} sites) ===")
    print(f"  consensus:            {cons}")
    print(f"  distinct sequences:   {distinct}/{n}  ({100*distinct/n:.0f}% unique)")
    print(f"  exactly == consensus: {exact}/{n}  ({100*exact/max(n,1):.1f}%)")
    print(f"  mean identity to consensus: {100*mean_id:.0f}% per base")
    print(f"  per-position conservation (frac matching consensus):")
    print(f"    {' '.join(f'{x:.1f}' for x in perpos_match)}")
    print(f"  example matched sites:")
    for s in sites[:6]: print(f"    {s}")
