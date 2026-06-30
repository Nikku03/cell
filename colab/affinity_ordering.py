"""The genome-derivable slice of the binding pathway: relative site AFFINITY
(step 3, = PWM energy) — computable without [TF] (step 4). Test whether affinity
RANK within a TF's regulon predicts regulatory COUPLING strength (proxy: |corr|
of target expression with the TF across PRECISE conditions). If yes, the pathway
ordering computes from genome even though absolute occupancy does not.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
rng=np.random.default_rng(0)
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
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
# PRECISE expression matrix (b -> z-scored vector for correlation)
bs=[]; rows=[]
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: bs.append(line[0]); rows.append([float(x) for x in line[1:]])
X=np.array(rows); bidx={b:i for i,b in enumerate(bs)}
Z=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-9); ns=X.shape[1]
W=20
def ups(name,pad=200):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
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
def aff(arr,lo):
    bv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
def spear(a,b):
    if len(a)<5: return None
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return float(np.corrcoef(ra,rb)[0,1])

rows_out=[]
for tf,tg in tf_t.items():
    tb=name2b.get(tf)
    if tb not in bidx: continue
    tgts=[t for t in tg if t in genes and name2b.get(t) in bidx and ups(t) is not None]
    if len(tgts)<8: continue
    pwm=em([ups(t) for t in tgts])
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    affs=[aff(ups(t),lo) for t in tgts]
    # coupling proxy: |corr(target expr, TF expr)| across PRECISE
    tfv=Z[bidx[tb]]
    coup=[abs(float((Z[bidx[name2b[t]]]@tfv)/ns)) for t in tgts]
    s=spear(affs,coup)
    if s is not None: rows_out.append((tf,len(tgts),s))
sp=[r[2] for r in rows_out]
print(f"TFs tested: {len(rows_out)}")
print(f"mean Spearman(site affinity rank, regulatory coupling) within regulon = {np.mean(sp):+.3f}")
print(f"median = {np.median(sp):+.3f}   ; fraction positive = {np.mean([x>0 for x in sp]):.2f}")
pos=sorted(rows_out,key=lambda r:-r[2])[:8]
print("\nstrongest (affinity predicts coupling within regulon):")
for tf,n,s in pos: print(f"  {tf:<8} n={n:>3}  rho={s:+.3f}")
