"""Edge model v2: fuse the OPERATOR (sequence) signal into the functional edge model.
Per (TF,gene): coexp + cofit + adjacency + operon + expr + OPERATOR-AFFINITY (TF's
learned PWM scored on the gene promoter). Leak-free (PWM learned on a target subset;
eval positives held out). Leave-TF-out. Does sequence+functional beat functional alone?
"""
import csv, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); COE=Path("data/external_data/coexpression")
rng=np.random.default_rng(0)
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
B={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([B.get(c,-1) for c in s],dtype=np.int8)
def rc(a): comp=np.array([3,2,1,0,-1]); return comp[a][::-1]
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
gc=(G.count("G")+G.count("C"))/GLEN; bg=np.array([(1-gc)/2,gc/2,gc/2,(1-gc)/2])
bs=[];rows=[]
with open(COE/"log_tpm.csv") as f:
    r=csv.reader(f); next(r)
    for line in r: bs.append(line[0]); rows.append([float(x) for x in line[1:]])
Xe=np.array(rows); bidx={b:i for i,b in enumerate(bs)}
Z=(Xe-Xe.mean(1,keepdims=True))/(Xe.std(1,keepdims=True)+1e-9); ns=Xe.shape[1]; exprmean={b:float(Xe[bidx[b]].mean()) for b in bs}
name2b={}; coord={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
gpos={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit():
        b=name2b.get(p[1].lower())
        if b: gpos[b]=(int(p[3]),int(p[4]),p[5]); coord[b]=((int(p[3])+int(p[4]))//2,1 if p[5]=="forward" else -1)
kb={l:sn for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
cofit=defaultdict(dict)
for l,h,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    if l in kb and h in kb: cofit[kb[l]][kb[h]]=abs(cf)
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
W=20
def ups(b,pad=200):
    if b not in gpos: return None
    s,e,st=gpos[b]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
def em(seqs,n_iter=18,restarts=2):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
    if len(seqs)<5: return None
    best=None;bll=-1e18
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
    if arr is None or len(arr)<W or not (arr>=0).all(): return 0.0
    bv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
allb=list(bs)
TFs=[tf for tf in tf_t if name2b.get(tf) in bidx and len([t for t in tf_t[tf] if name2b.get(t) in bidx and t in name2b])>=12]
data=[]
for tf in TFs:
    tfb=name2b[tf]; tg=[name2b[t] for t in tf_t[tf] if name2b.get(t) in bidx]; tg=[b for b in tg if b in gpos and b!=tfb]
    if len(tg)<12: continue
    rng.shuffle(tg); h=len(tg)//2; train,test=tg[:h],tg[h:]
    pwm=em([ups(b) for b in train]); lo=np.log(pwm+1e-9)-np.log(bg+1e-9) if pwm is not None else None
    tgset=set(tg)
    negs=list(rng.choice([b for b in allb if b not in tgset and b!=tfb and b in gpos],size=min(len(test)*5,200),replace=False))
    for gb in test+negs:
        co=float((Z[bidx[tfb]]@Z[bidx[gb]])/ns); cf=cofit.get(tfb,{}).get(gb,0.0)
        d=99;ss=0
        if tfb in coord and gb in coord: d=abs(coord[tfb][0]-coord[gb][0])/1e6; ss=1 if(d<0.005 and coord[tfb][1]==coord[gb][1]) else 0
        a=aff(ups(gb),lo) if lo is not None else 0.0
        data.append((tf,1 if gb in tgset else 0,[abs(co),cf,1/(1+d*50),ss,exprmean.get(gb,0),a]))
FN=["coexp","cofit","adjacency","operon","expr","operator_aff"]
print(f"TFs {len(set(d[0] for d in data))}, pairs {len(data)}")
def logit(X,y,l2=1.0,it=400,lr=0.2):
    X=np.asarray(X,float);mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1;Xn=(X-mu)/sd;Xn=np.hstack([Xn,np.ones((len(Xn),1))]);w=np.zeros(Xn.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-Xn@w));g=Xn.T@(p-y)/len(y)+l2*np.r_[w[:-1],0]/len(y);w-=lr*g
    return (mu,sd,w)
def pred(m,X):
    mu,sd,w=m;X=np.asarray(X,float);Xn=(X-mu)/sd;Xn=np.hstack([Xn,np.ones((len(Xn),1))]);return 1/(1+np.exp(-Xn@w))
def auc(p,y):
    p=np.asarray(p);y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return None
    o=np.argsort(p);r=np.empty(len(p));r[o]=np.arange(len(p));m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
TFlist=sorted(set(d[0] for d in data))
def run(cols):
    idx=[FN.index(c) for c in cols];res=[]
    for hold in TFlist:
        tr=[([d[2][i] for i in idx],d[1]) for d in data if d[0]!=hold]; te=[([d[2][i] for i in idx],d[1]) for d in data if d[0]==hold]
        if sum(l for _,l in te)<3: continue
        m=logit([x for x,_ in tr],np.array([l for _,l in tr]));a=auc(pred(m,[x for x,_ in te]),np.array([l for _,l in te]))
        if a is not None: res.append(a)
    return round(float(np.mean(res)),3)
print("\n=== leave-TF-out edge AUC ===")
print(f"  functional (coexp+cofit+adj+operon+expr) : {run(['coexp','cofit','adjacency','operon','expr'])}")
print(f"  + OPERATOR affinity (sequence)           : {run(FN)}")
import json; json.dump(dict(functional=run(['coexp','cofit','adjacency','operon','expr']),with_operator=run(FN)),open("outputs/orphan/edge_model_v2.json","w"),indent=2)
print("\nwrote outputs/orphan/edge_model_v2.json")
