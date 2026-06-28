"""Find operators WITHOUT an operator database: the co-fitness bootstrap.

Mode B of "use TF binding info to look for operators":
  for a TF with no known motif, take the genes it CO-REGULATES (feba co-fitness),
  discover the DNA motif over-represented in THEIR promoters (EM) -> that motif
  IS the operator -> scan all promoters with it -> recover the regulon.

Validate on E. coli (RegulonDB truth):
  - discover motif from the TF's COFIT partners' promoters (blind to RegulonDB)
  - test: does it separate HELD-OUT RegulonDB targets from non-targets? (AUC)
  - upper bound: motif discovered from RegulonDB targets directly (supervised)
  - floor: cofit membership alone as the predictor
Question: can functional data substitute for an operator DB to find operators?
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(7)
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
name2b={};
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
b2name={b:n for n,b in name2b.items()}
tf_targets=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf_targets[p[0].lower()].add(p[1].lower())
def ups(name,pad=200):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])

# feba Keio: b<->locus, and cofit partners per locus
b2kl={}; kl2b={}
for lid,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'"):
    b2kl[sn]=lid; kl2b[lid]=sn
cofit=defaultdict(list)
for lid,hit,cf in cur.execute("SELECT locusId,hitId,cofit FROM Cofit WHERE orgId='Keio'"):
    cofit[lid].append((hit,abs(cf)))
def cofit_partner_names(tf_name,topK=40):
    b=name2b.get(tf_name); kl=b2kl.get(b) if b else None
    if kl is None: return []
    parts=sorted(cofit.get(kl,[]),key=lambda x:-x[1])[:topK]
    out=[]
    for hit,_ in parts:
        bb=kl2b.get(hit); nm=b2name.get(bb) if bb else None
        if nm and nm in genes: out.append(nm)
    return out

# ---- EM OOPS + scoring (W=20) ----
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
        lo=np.log(pwm+1e-9)-np.log(bg+1e-9); ll=0.0
        for s in seqs:
            best_s=-1e18
            for arr in (s,rc(s)):
                n=len(arr)-W+1
                if n<=0: continue
                sc=np.zeros(n)
                for j in range(W): sc+=lo[j,arr[j:j+n]]
                best_s=max(best_s,sc.max())
            ll+=best_s
        if ll>bll: bll=ll; best=pwm
    return best
def score(arr,lo,W=20):
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    bestv=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bestv=max(bestv,sc.max())
    return float(bestv)
def ic(p): return float(np.sum(p*np.log2((p+1e-9)/0.25)))
def cons(p): return "".join("ACGT"[i] for i in p.argmax(1))
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if len(pos)<3 or len(neg)<3: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

allnames=list(genes)
W=20
rows=[]
cand=[(tf,[t for t in tg if t in genes]) for tf,tg in tf_targets.items()]
cand=[(tf,ts) for tf,ts in cand if len(ts)>=12 and tf in name2b and name2b[tf] in b2kl]
cand.sort(key=lambda x:-len(x[1]))
print(f"TFs (>=12 coord targets, in feba w/ cofit): testing up to 30\n")
for tf,tgts in cand[:30]:
    partners=cofit_partner_names(tf,topK=40)
    if len(partners)<8: continue
    tset=set(tgts)
    # motif discovered from COFIT partners (blind to RegulonDB)
    pwm_c=em([ups(p) for p in partners],W=W)
    # supervised upper bound: motif from a random half of RegulonDB targets
    tl=list(tgts); rng.shuffle(tl); tr,te=tl[:len(tl)//2],tl[len(tl)//2:]
    pwm_s=em([ups(t) for t in tr],W=W)
    if pwm_c is None or pwm_s is None: continue
    lo_c=np.log(pwm_c+1e-9)-np.log(bg+1e-9); lo_s=np.log(pwm_s+1e-9)-np.log(bg+1e-9)
    # eval on HELD-OUT targets (not used by supervised) and NOT among cofit partners
    pset=set(partners)
    held=[t for t in te if t not in pset]
    negs=list(rng.choice([n for n in allnames if n not in tset and n not in pset],
                         size=min(len(held)*5,300),replace=False))
    if len(held)<3: continue
    a_c=auc([score(ups(t),lo_c,W) for t in held],[score(ups(n),lo_c,W) for n in negs])
    a_s=auc([score(ups(t),lo_s,W) for t in held],[score(ups(n),lo_s,W) for n in negs])
    rows.append(dict(tf=tf,n_tgt=len(tgts),n_part=len(partners),n_held=len(held),
                     auc_cofit_operator=a_c,auc_supervised=a_s,
                     motif_cofit=cons(pwm_c),ic_cofit=round(ic(pwm_c),1)))
    print(f"  {tf:<8} tgt={len(tgts):>3} part={len(partners):>2} held={len(held):>2}  "
          f"cofit-op AUC={a_c if a_c else 0:.3f}  supervised={a_s if a_s else 0:.3f}  {cons(pwm_c)}")

def mean(k):
    v=[r[k] for r in rows if r[k] is not None]; return round(float(np.mean(v)),3) if v else None
won=sum(1 for r in rows if r['auc_cofit_operator'] and r['auc_cofit_operator']>=0.6)
print(f"\n=== {len(rows)} TFs ===")
print(f"  cofit-discovered operator  mean AUC = {mean('auc_cofit_operator')}")
print(f"  supervised (RegulonDB)     mean AUC = {mean('auc_supervised')}")
print(f"  cofit-operator reaches AUC>=0.60 for {won}/{len(rows)} TFs")
json.dump(dict(n=len(rows),mean_cofit_operator=mean('auc_cofit_operator'),
               mean_supervised=mean('auc_supervised'),n_strong=won,per_tf=rows),
          open(OUT/"cofit_operator.json","w"),indent=2)
print(f"\nwrote {OUT}/cofit_operator.json")
