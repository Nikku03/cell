"""TRY the 4-dimensional biophysical TF->operator framework, faithfully, and
measure what each dimension ADDS over the plain motif. Leave-TF-out CV.

Per E. coli TF (>=12 RegulonDB targets, Pfam family via feba): learn operator
PWM, then for each candidate promoter (target vs non-target) build features:
  D2 Direct readout : best PWM log-odds hit                       (sequence)
  D1 Architecture   : dyad/palindrome symmetry of the hit window  (family geometry)
  D3 Shape field    : DNA shape from dinucleotide params (twist, roll, MGW-proxy,
                      electrostatic/AT proxy) over the hit window  (indirect readout)
  D4 Macro-context  : distance of hit to gene start + helical PHASE (sin/cos 10.5bp)
  Vtf Family        : one-hot family class                        (protein side)
Train logistic regression with LEAVE-TF-OUT CV; ablate dimensions; report mean
per-held-TF AUC. Question: do D1/D3 (sequence-derived) add anything over D2, or
is the only real lift from D4 (position) + family?
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(5)
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
BB={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([BB.get(c,-1) for c in s],dtype=np.int8)
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
PAD=200
def ups(name,pad=PAD):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
loc2dom=defaultdict(list)
for l,dn in cur.execute("SELECT locusId,domainName FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"): loc2dom[l].append(dn or "")
FAMP=[("HTH_AraC","AraC"),("LysR_substrate","LysR"),("HTH_1","LysR"),("LacI","LacI"),
      ("Trans_reg_C","OmpR"),("GerE","NarL"),("Crp","Crp"),("HTH_Crp","Crp"),
      ("GntR","GntR"),("FCD","GntR"),("HTH_18","MarR"),("MarR","MarR"),("TetR","TetR"),
      ("Sigma54","bEBP"),("Lrp","Lrp"),("AsnC","Lrp"),("IclR","IclR"),("DeoR","DeoR")]
def fam_of(tf):
    b=name2b.get(tf); kl=b2kl.get(b) if b else None
    if not kl: return "other"
    ds=" ".join(loc2dom.get(kl,[]))
    for k,f in FAMP:
        if k.lower() in ds.lower(): return f
    return "other"
# dinucleotide structural params (approx consensus, Olson-style). idx = 4*b1+b2
TW=np.zeros(16); RO=np.zeros(16)
tw={"AA":35.6,"AT":31.5,"TA":36.0,"AC":34.4,"GT":34.4,"AG":27.7,"CT":27.7,"CA":34.5,"TG":34.5,
    "CC":33.7,"GG":33.7,"CG":29.8,"GA":36.9,"TC":36.9,"GC":40.0,"TT":35.6}
ro={"AA":0.7,"AT":1.1,"TA":3.3,"AC":0.6,"GT":0.6,"AG":3.1,"CT":3.1,"CA":3.3,"TG":3.3,
    "CC":3.6,"GG":3.6,"CG":6.0,"GA":1.6,"TC":1.6,"GC":1.2,"TT":0.7}
inv="ACGT"
for i in range(4):
    for j in range(4):
        d=inv[i]+inv[j]; TW[4*i+j]=tw[d]; RO[4*i+j]=ro[d]
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
                scs=np.concatenate([x for _,x in ap]); m=scs.max(); w=np.exp(scs-m); w/=w.sum(); off=0
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
W=20
def feats(arr,lo):
    """return feature dict for a promoter window given a TF's PWM logodds."""
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    bestv=-1e18; bestpos=0; bestseq=None
    L=len(arr)
    for strand,a in enumerate((arr,rc(arr))):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        k=int(sc.argmax())
        if sc[k]>bestv: bestv=sc[k]; bestpos=k; bestseq=a[k:k+W]
    # D1 architecture: palindrome symmetry of hit window
    onehot=np.zeros((W,4)); onehot[np.arange(W),bestseq]=1
    rcoh=onehot[::-1][:,[3,2,1,0]]
    arch=float(np.corrcoef(onehot.ravel(),rcoh.ravel())[0,1])
    # D3 shape over hit window (dinucleotide)
    di=4*bestseq[:-1]+bestseq[1:]
    tw_w=TW[di]; ro_w=RO[di]
    at=float(np.mean((bestseq==0)|(bestseq==3)))           # AT fraction -> electrostatic/MGW proxy
    # longest A/T run (A-tract -> narrow minor groove)
    run=mx=0
    for b in bestseq:
        if b in (0,3): run+=1; mx=max(mx,run)
        else: run=0
    shape=[tw_w.mean(),tw_w.std(),ro_w.mean(),ro_w.max(),at,mx/W]
    # D4 macro-context: distance of hit to gene start (window end = -1) + helical phase
    dist=(L-bestpos)/PAD
    phase=2*np.pi*((L-bestpos)%10.5)/10.5
    return dict(pwm=bestv,arch=arch,shape=shape,dist=dist,sin=np.sin(phase),cos=np.cos(phase))

# build dataset
TFs=[t for t in tf_t if len(tf_t[t])>=12 and t in genes]
fams=sorted(set(fam_of(t) for t in TFs))
fidx={f:i for i,f in enumerate(fams)}
rows=[]  # (tf, fam, label, featdict)
negpool=list(rng.choice(list(genes),size=500,replace=False))
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes]
    rng.shuffle(ts); h=len(ts)//2
    train,test=ts[:h],ts[h:]              # PWM from TRAIN; positives = held-out TEST (no leak)
    if len(train)<5 or len(test)<3: continue
    pwm=em([ups(x) for x in train],W=W)
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9); fa=fam_of(t)
    negs=[n for n in negpool if n not in set(ts)][:min(len(test)*4,120)]
    for x,lab in [(x,1) for x in test]+[(n,0) for n in negs]:
        f=feats(ups(x),lo)
        if f: rows.append((t,fa,lab,f))
print(f"rows={len(rows)} TFs={len(set(r[0] for r in rows))} families={len(fams)}")

def vec(f,fa,dims):
    v=[]
    if "D2" in dims: v+=[f["pwm"]]
    if "D1" in dims: v+=[f["arch"]]
    if "D3" in dims: v+=f["shape"]
    if "D4" in dims: v+=[f["dist"],f["sin"],f["cos"]]
    if "FAM" in dims:
        oh=[0.0]*len(fams); oh[fidx[fa]]=1.0; v+=oh
    return v
def logit_train(X,y,l2=1.0,iters=300,lr=0.1):
    X=np.asarray(X,float); mu=X.mean(0); sd=X.std(0); sd[sd<1e-9]=1; Xn=(X-mu)/sd
    Xn=np.hstack([Xn,np.ones((len(Xn),1))]); w=np.zeros(Xn.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-Xn@w)); g=Xn.T@(p-y)/len(y)+l2*np.r_[w[:-1],0]/len(y); w-=lr*g
    return (mu,sd,w)
def logit_pred(m,X):
    mu,sd,w=m; X=np.asarray(X,float); Xn=(X-mu)/sd; Xn=np.hstack([Xn,np.ones((len(Xn),1))])
    return 1/(1+np.exp(-Xn@w))
def auc(p,y):
    p=np.asarray(p); y=np.asarray(y)
    if y.sum()==0 or y.sum()==len(y): return None
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(len(p)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

ABL=[("D2",["D2"]),("D2+D1",["D2","D1"]),("D2+D3",["D2","D3"]),("D2+D4",["D2","D4"]),
     ("D2+FAM",["D2","FAM"]),("ALL",["D2","D1","D3","D4","FAM"])]
tflist=sorted(set(r[0] for r in rows))
res={name:[] for name,_ in ABL}
for name,dims in ABL:
    for hold in tflist:   # leave-TF-out
        tr=[(r[1],r[2],r[3]) for r in rows if r[0]!=hold]
        te=[(r[1],r[2],r[3]) for r in rows if r[0]==hold]
        if not te or sum(l for _,l,_ in te)<3 or (len(te)-sum(l for _,l,_ in te))<3: continue
        Xtr=[vec(f,fa,dims) for fa,l,f in tr]; ytr=np.array([l for fa,l,f in tr])
        Xte=[vec(f,fa,dims) for fa,l,f in te]; yte=np.array([l for fa,l,f in te])
        m=logit_train(Xtr,ytr); a=auc(logit_pred(m,Xte),yte)
        if a is not None: res[name].append(a)
print("\n=== leave-TF-out mean AUC by dimension set ===")
for name,_ in ABL:
    print(f"  {name:<10} {np.mean(res[name]):.3f}   (n={len(res[name])} TFs)")
json.dump({name:round(float(np.mean(v)),3) for name,v in res.items()},
          open(OUT/"integrated_binding.json","w"),indent=2)
print(f"\nwrote {OUT}/integrated_binding.json")
