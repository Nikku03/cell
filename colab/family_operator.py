"""Does TF protein FAMILY predict the OPERATOR? Decisive transfer test.

If family carries operator information, then a TF's family-MATE motif should
find that TF's targets better than a random DIFFERENT-family motif.

Per E. coli TF (>=12 RegulonDB targets, Pfam family via feba):
  - learn its operator PWM (EM on its targets' promoters)
For every ordered pair (A,B): score B's targets vs non-targets using A's PWM
(fully out-of-sample for B). Split by SAME-family vs DIFFERENT-family.
Compare: within-family transfer AUC  vs  between-family transfer AUC  vs  self.
Also: operator architecture (palindrome strength, info content) by family.
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(3)
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
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
def ups(name,pad=200):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])
# family by DNA-binding Pfam domain (priority order)
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
loc2dom=defaultdict(list)
for l,dn in cur.execute("SELECT locusId,domainName FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"): loc2dom[l].append(dn or "")
FAMP=[("HTH_AraC","AraC"),("LysR_substrate","LysR"),("HTH_1","LysR"),("LacI","LacI"),
      ("Trans_reg_C","OmpR"),("GerE","NarL/LuxR"),("Crp","Crp/FNR"),("HTH_Crp","Crp/FNR"),
      ("GntR","GntR"),("FCD","GntR"),("HTH_18","MarR"),("MarR","MarR"),("TetR","TetR"),
      ("Sigma54","bEBP"),("Fis","Fis"),("IclR","IclR"),("DeoR","DeoR"),("Lrp","Lrp"),
      ("AsnC","Lrp"),("MerR","MerR"),("ArsR","ArsR"),("Rrf2","Rrf2")]
def fam_of(tf):
    b=name2b.get(tf); kl=b2kl.get(b) if b else None
    if not kl: return None
    ds=" ".join(loc2dom.get(kl,[]))
    for k,f in FAMP:
        if k.lower() in ds.lower(): return f
    return None
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
def lo_of(pwm): return np.log(pwm+1e-9)-np.log(bg+1e-9)
def score(arr,lo,W=20):
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    bv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if len(pos)<3 or len(neg)<3: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))
def palindrome(pwm):
    # correlation between PWM and its reverse-complement (column-reverse + base-complement)
    rcp=pwm[::-1][:, [3,2,1,0]]
    return float(np.corrcoef(pwm.ravel(), rcp.ravel())[0,1])
def ic(p): return float(np.sum(p*np.log2((p+1e-9)/0.25)))

alln=list(genes)
W=20
TFs=[t for t in tf_t if len(tf_t[t])>=12 and t in genes and fam_of(t)]
# learn PWM per TF on ALL its targets; cache upstream encodings of targets + a shared neg pool
pwm={}; fam={}; tgt_ups={}; tgt_set={}
negpool=list(rng.choice(alln,size=400,replace=False))
neg_ups=[ups(n) for n in negpool]
self_auc={}
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes]
    p=em([ups(x) for x in ts],W=W)
    if p is None: continue
    pwm[t]=p; fam[t]=fam_of(t); tgt_set[t]=set(ts)
    tgt_ups[t]=[ups(x) for x in ts]
    # self: train on half
    rng.shuffle(ts); h=len(ts)//2
    ph=em([ups(x) for x in ts[:h]],W=W)
    if ph is not None:
        lo=lo_of(ph); negs=[u for u in neg_ups if u is not None][:120]
        self_auc[t]=auc([score(ups(x),lo,W) for x in ts[h:]],[score(u,lo,W) for u in negs])
TFs=[t for t in TFs if t in pwm]
print(f"TFs with PWM+family: {len(TFs)}")
from collections import Counter
print("family sizes:",Counter(fam[t] for t in TFs))

# pairwise transfer: A's PWM -> B's targets vs negs
within=[]; between=[]; permat={}
for A in TFs:
    loA=lo_of(pwm[A])
    for Bt in TFs:
        if A==Bt: continue
        # B's positives must exclude any gene also targeted by A? keep simple: B targets vs neg pool (excl B & A targets)
        pos=[score(u,loA,W) for u in tgt_ups[Bt]]
        negs=[score(u,loA,W) for n,u in zip(negpool,neg_ups) if n not in tgt_set[Bt] and n not in tgt_set[A]][:120]
        a=auc(pos,negs)
        if a is None: continue
        (within if fam[A]==fam[Bt] else between).append(a)
        permat[(A,Bt)]=a
def mean(v): return round(float(np.mean(v)),3) if v else None
print(f"\n=== family-operator transfer ===")
print(f"  WITHIN-family transfer AUC  mean = {mean(within)}  (n={len(within)} pairs)")
print(f"  BETWEEN-family transfer AUC mean = {mean(between)} (n={len(between)} pairs)")
print(f"  self (own held-out) AUC     mean = {mean([v for v in self_auc.values() if v])}")
# per-family within-transfer
byfam=defaultdict(list)
for (A,Bt),a in permat.items():
    if fam[A]==fam[Bt]: byfam[fam[A]].append(a)
print("\n  within-family transfer by family:")
for f,v in sorted(byfam.items(),key=lambda x:-(np.mean(x[1]) if x[1] else 0)):
    if len(v)>=2: print(f"    {f:<10} n_pairs={len(v):>3}  AUC={np.mean(v):.3f}")
# architecture by family
print("\n  operator architecture by family (palindrome corr, info content):")
fam_arch=defaultdict(lambda:[[],[]])
for t in TFs: fam_arch[fam[t]][0].append(palindrome(pwm[t])); fam_arch[fam[t]][1].append(ic(pwm[t]))
for f,(pal,ics) in sorted(fam_arch.items(),key=lambda x:-np.mean(x[1][0])):
    if len(pal)>=2: print(f"    {f:<10} n={len(pal):>2}  palindrome={np.mean(pal):+.2f}  IC={np.mean(ics):.1f}")
json.dump(dict(n_tf=len(TFs),within_mean=mean(within),between_mean=mean(between),
               self_mean=mean([v for v in self_auc.values() if v]),
               within_by_family={f:round(float(np.mean(v)),3) for f,v in byfam.items() if len(v)>=2},
               palindrome_by_family={f:round(float(np.mean(p)),3) for f,(p,i) in fam_arch.items() if len(p)>=2}),
          open(OUT/"family_operator.json","w"),indent=2)
print(f"\nwrote {OUT}/family_operator.json")
