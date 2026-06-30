"""Can cooperativity + position + occupancy find the operator where the motif alone
can't? Leak-free, leave-TF-out, E. coli / RegulonDB.

Per (TF, promoter) features:
  SEQ  : best PWM hit (affinity)                                  [baseline]
  COOP : 2nd-best PWM hit (homotypic clustering) + sigma70 -35/-10
         promoter score near the site (co-occurrence)            [cooperativity]
  POS  : distance of best hit to gene start + helical phase       [position]
  OCC  : measured TF expression level (PRECISE) as a [TF] proxy   [occupancy]
Ablate. NOTE: OCC is a per-TF constant across that TF's candidate genes, so it
cannot discriminate WHICH gene within a TF -- we test that explicitly.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
rng=np.random.default_rng(6)
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
PAD=200; W=20
def ups(name,pad=PAD):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
U={g:ups(g) for g in genes}
allg=[g for g in genes if U[g] is not None and len(U[g])>=W and (U[g]>=0).all()]
# TF expression level (PRECISE mean log-TPM) -> [TF] proxy
expr={}
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: expr[line[0]]=float(np.mean([float(x) for x in line[1:]]))
def tf_expr(tf):
    b=name2b.get(tf); return expr.get(b,np.nan) if b else np.nan
# sigma70 -35 TTGACA / -10 TATAAT scoring (consensus match, spacer 15-19)
M35=enc("TTGACA"); M10=enc("TATAAT")
def matchrun(arr,mot):
    n=len(arr)-len(mot)+1
    if n<=0: return np.zeros(0)
    return np.array([int((arr[i:i+len(mot)]==mot).sum()) for i in range(n)])
def sigma_score(arr):
    best=0
    for a in (arr,rc(arr)):
        s35=matchrun(a,M35); s10=matchrun(a,M10)
        for i in range(len(s35)):
            for sp in range(15,20):
                j=i+6+sp
                if j<len(s10): best=max(best,s35[i]+s10[j])
    return best
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
def hits(arr,lo):
    """return best, 2nd-best (non-overlapping), and argmax position (from end)."""
    allsc=[]
    for strand,a in enumerate((arr,rc(arr))):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        for k in range(n): allsc.append((sc[k],strand,k))
    if not allsc: return None
    allsc.sort(key=lambda x:-x[0])
    best=allsc[0]; second=None
    for s in allsc[1:]:
        if abs(s[2]-best[2])>=W: second=s; break
    L=len(arr)
    return dict(best=best[0], second=(second[0] if second else best[0]-3),
                dist=(L-best[2])/PAD, phase=2*np.pi*((L-best[2])%10.5)/10.5)

# precompute sigma per promoter (independent of TF)
SIG={g:sigma_score(U[g]) for g in allg}
TFs=[t for t in tf_t if len([x for x in tf_t[t] if x in genes and U[x] is not None])>=12 and not np.isnan(tf_expr(t))]
rows=[]
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes and U[x] is not None]
    rng.shuffle(ts); h=len(ts)//2; train,test=ts[:h],ts[h:]
    if len(train)<5 or len(test)<3: continue
    pwm=em([U[x] for x in train])
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    negs=list(rng.choice([g for g in allg if g not in set(ts)],size=min(len(test)*5,200),replace=False))
    te_expr=tf_expr(t)
    for g,lab in [(x,1) for x in test]+[(n,0) for n in negs]:
        hh=hits(U[g],lo)
        if hh is None: continue
        rows.append((t,lab,dict(seq=hh["best"],second=hh["second"],sigma=SIG[g],
                                dist=hh["dist"],sin=np.sin(hh["phase"]),cos=np.cos(hh["phase"]),expr=te_expr)))
print(f"rows={len(rows)} TFs={len(set(r[0] for r in rows))}")
def vec(f,dims):
    v=[]
    if "SEQ" in dims: v+=[f["seq"]]
    if "COOP" in dims: v+=[f["second"],f["sigma"]]
    if "POS" in dims: v+=[f["dist"],f["sin"],f["cos"]]
    if "OCC" in dims: v+=[f["expr"]]
    return v
def logit(X,y,l2=1.0,it=300,lr=0.1):
    X=np.asarray(X,float); mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
    Xn=np.hstack([Xn,np.ones((len(Xn),1))]); w=np.zeros(Xn.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xn@w)); g=Xn.T@(p-y)/len(y)+l2*np.r_[w[:-1],0]/len(y); w-=lr*g
    return (mu,sd,w)
def pred(m,X):
    mu,sd,w=m; X=np.asarray(X,float); Xn=(X-mu)/sd; Xn=np.hstack([Xn,np.ones((len(Xn),1))]); return 1/(1+np.exp(-Xn@w))
def auc(p,y):
    p=np.asarray(p); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return None
    r=np.empty(len(p)); o=np.argsort(p); r[o]=np.arange(len(p)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
ABL=[("SEQ",["SEQ"]),("SEQ+COOP",["SEQ","COOP"]),("SEQ+POS",["SEQ","POS"]),
     ("SEQ+OCC",["SEQ","OCC"]),("SEQ+COOP+POS",["SEQ","COOP","POS"]),
     ("ALL",["SEQ","COOP","POS","OCC"])]
tfl=sorted(set(r[0] for r in rows)); res={n:[] for n,_ in ABL}
for name,dims in ABL:
    for hold in tfl:
        tr=[(r[1],r[2]) for r in rows if r[0]!=hold]; te=[(r[1],r[2]) for r in rows if r[0]==hold]
        if not te or sum(l for l,_ in te)<3 or len(te)-sum(l for l,_ in te)<3: continue
        m=logit([vec(f,dims) for _,f in tr],np.array([l for l,_ in tr]))
        a=auc(pred(m,[vec(f,dims) for _,f in te]),np.array([l for l,_ in te]))
        if a is not None: res[name].append(a)
import json
print("\n=== leave-TF-out mean AUC ===")
out={}
for n,_ in ABL:
    out[n]=round(float(np.mean(res[n])),3); print(f"  {n:<14} {out[n]:.3f}  (n={len(res[n])})")
json.dump(out,open(Path("outputs/orphan")/"coop_occ_pos.json","w"),indent=2)
print("\nwrote outputs/orphan/coop_occ_pos.json")
