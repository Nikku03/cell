"""Remove coding DNA, search operators only in the intergenic fraction.

Genome-wide PWM scan per TF, then split hits into CODING vs INTERGENIC. Measure:
  - genome composition (coding vs intergenic %)
  - where TRUE sites fall (coding vs intergenic)
  - decoys removed by dropping coding (% of hits that were in CDS)
  - precision of site->target: whole-genome vs intergenic-only
Leak-free: PWM from train-half targets; "true operator region" = 200bp upstream
of a real RegulonDB target.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(4)
B={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([B.get(c,-1) for c in s],dtype=np.int8)
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
W=20; PAD=200
# coding mask
coding=np.zeros(GLEN,bool)
for s,e,st in genes.values(): coding[max(0,s-1):e]=True
cod_frac=coding.mean()
print(f"genome {GLEN} bp  coding={100*cod_frac:.1f}%  intergenic={100*(1-cod_frac):.1f}%")
# true-operator-region mask builder for a TF's targets (200bp upstream, strand-aware)
def true_region(tgts):
    mask=np.zeros(GLEN,bool)
    for t in tgts:
        if t not in genes: continue
        s,e,st=genes[t]
        if st=="forward": mask[max(0,s-PAD-1):s-1]=True
        else: mask[e:min(GLEN,e+PAD)]=True
    return mask
def ups(name):
    if name not in genes: return None
    s,e,st=genes[name]
    a=Genc[max(0,s-PAD-1):s-1] if st=="forward" else Genc[e:min(GLEN,e+PAD)]
    if st=="reverse": a=np.array([3,2,1,0,-1])[a][::-1]
    return a
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
                for arr in (s,np.array([3,2,1,0,-1])[s][::-1]):
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
            for arr in (s,np.array([3,2,1,0,-1])[s][::-1]):
                n=len(arr)-W+1
                if n<=0: continue
                sc=np.zeros(n)
                for j in range(W): sc+=lo[j,arr[j:j+n]]
                bv=max(bv,sc.max())
            ll+=bv
        if ll>bll: bll=ll; best=pwm
    return best
def genome_scan(lo):
    n=GLEN-W+1; Gv=Genc[:n+W-1]
    sc=np.zeros(n);
    for j in range(W): sc+=lo[j,Genc[j:j+n]]
    rclo=lo[::-1][:,[3,2,1,0]]; scr=np.zeros(n)
    for j in range(W): scr+=rclo[j,Genc[j:j+n]]
    return np.maximum(sc,scr)   # best-strand score per start position
def besthit_up(arr,lo):
    bv=-1e18
    for a in (arr,np.array([3,2,1,0,-1])[arr][::-1]):
        n=len(a)-W+1
        if n<=0: continue
        s=np.zeros(n)
        for j in range(W): s+=lo[j,a[j:j+n]]
        bv=max(bv,s.max())
    return bv

TFs=[t for t in tf_t if len([x for x in tf_t[t] if x in genes])>=12]
rows=[]
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes and ups(x) is not None]
    rng.shuffle(ts); h=len(ts)//2; train,test=ts[:h],ts[h:]
    if len(train)<5 or len(test)<3: continue
    pwm=em([ups(x) for x in train])
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    thr=min(besthit_up(ups(x),lo) for x in train)
    sc=genome_scan(lo)
    hit=np.where(sc>=thr)[0]                         # genome positions of operator matches
    if len(hit)==0: continue
    hit_cod=coding[hit]                              # is each hit in a CDS?
    n_tot=len(hit); n_cod=int(hit_cod.sum()); n_int=n_tot-n_cod
    treg=true_region(ts)                             # all true target promoters
    in_true=treg[hit]
    prec_all=in_true.mean()
    inter=~hit_cod
    prec_int=in_true[inter].mean() if inter.any() else 0.0
    rows.append(dict(tf=t,n_targets=len(ts),hits=n_tot,
                     pct_hits_coding=round(100*n_cod/n_tot,1),
                     prec_genome=round(float(prec_all),4),
                     prec_intergenic=round(float(prec_int),4)))
import numpy as np
def mean(k): return round(float(np.mean([r[k] for r in rows])),4)
print(f"\nTFs: {len(rows)}")
print(f"mean % of operator HITS that fall in coding DNA (= decoys removed): {mean('pct_hits_coding')}%")
print(f"mean precision (site -> true target region):")
print(f"   whole-genome scan : {mean('prec_genome')}")
print(f"   intergenic only   : {mean('prec_intergenic')}")
print(f"   precision lift from dropping coding: x{mean('prec_intergenic')/max(mean('prec_genome'),1e-9):.2f}")
rows.sort(key=lambda r:r["hits"])
print(f"\n{'TF':<9}{'tgts':>5}{'hits':>7}{'%coding':>8}{'prec_gen':>9}{'prec_int':>9}")
for r in rows[:12]+rows[-6:]:
    print(f"  {r['tf']:<7}{r['n_targets']:>5}{r['hits']:>7}{r['pct_hits_coding']:>8.1f}{r['prec_genome']:>9.3f}{r['prec_intergenic']:>9.3f}")
import json
json.dump(dict(coding_pct=round(100*cod_frac,1),intergenic_pct=round(100*(1-cod_frac),1),
               n_tf=len(rows),mean_pct_hits_coding=mean('pct_hits_coding'),
               mean_prec_genome=mean('prec_genome'),mean_prec_intergenic=mean('prec_intergenic'),
               per_tf=rows),open(OUT/"intergenic_operator.json","w"),indent=2)
print(f"\nwrote {OUT}/intergenic_operator.json")
