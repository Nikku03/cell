"""Footprinting done right: comparator DISTANCE matters. v1 used 57 genomes incl.
distant phyla -> signal washed out (PWM 0.531 -> comb 0.534). Here:
  NEAR  = Enterobacterales relatives only (Klebsiella, Dickeya spp) -> site still
          conserved, regulatory network not yet rewired
  FAR   = distant orgs (Pseudomonas/Burkholderia/Ralstonia) -> negative control
Footprint score = MAX PWM hit across the comparator's orthologous upstreams
(conservation = the site recurs in a relative). Compare PWM vs NEAR vs FAR vs
PWM+NEAR, on RegulonDB held-out targets. Does near-relative conservation help?
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(11)
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
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv"))
        if r.get("gene_name") and r.get("gene_id")}
tf_targets=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf_targets[p[0].lower()].add(p[1].lower())
def ups_ec(name,pad=250):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rc(Genc[e:min(GLEN,e+pad)])

NEAR=["Koxy","Dda3937","Ddia6719","DdiaME23"]
FAR=["pseudo5_N2C3_1","WCS417","Putida","Burk376","RalstoniaGMI1000","Xantho"]
b2kl={sn:lid for lid,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
coord={}; scaf={}
for org in NEAR+FAR:
    for lid,sc,b,e,st in cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(org,)):
        coord[(org,lid)]=(sc,b,e,st)
    for sc,s in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(org,)):
        scaf[(org,sc)]=s.upper()
orth=defaultdict(lambda:defaultdict(list)); grp={**{o:"NEAR" for o in NEAR},**{o:"FAR" for o in FAR}}
for o2,l2,l1 in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1='Keio'"):
    if o2 in grp: orth[l1][grp[o2]].append((o2,l2))
def ups_feba(org,lid,pad=250):
    c=coord.get((org,lid))
    if not c: return None
    sc,b,e,st=c; s=scaf.get((org,sc))
    if not s: return None
    sub=s[max(0,b-pad-1):b-1] if st=='+' else s[e:min(len(s),e+pad)]
    a=enc(sub); return rc(a) if st=='-' else a
def ortho_ups(name,grpkey):
    b=name2b.get(name); kl=b2kl.get(b) if b else None
    if kl is None: return []
    out=[]
    for org,l2 in orth.get(kl,{}).get(grpkey,[]):
        u=ups_feba(org,l2)
        if u is not None and len(u)>=20 and (u>=0).all(): out.append(u)
    return out

def em(seqs,W=20,n_iter=25,restarts=3):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
    if len(seqs)<4: return None
    best=None; bll=-1e18
    for _ in range(restarts):
        s0=seqs[rng.integers(len(seqs))]; st=rng.integers(0,len(s0)-W+1)
        c=np.ones((W,4))*0.5
        for j in range(W): c[j,s0[st+j]]+=1
        pwm=c/c.sum(1,keepdims=True)
        for it in range(n_iter):
            new=np.ones((W,4))*0.25; ll=0.0; lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
            for s in seqs:
                ap=[]
                for arr in (s,rc(s)):
                    n=len(arr)-W+1
                    if n<=0: continue
                    sc=np.zeros(n)
                    for j in range(W): sc+=lo[j,arr[j:j+n]]
                    ap.append((arr,sc))
                if not ap: continue
                scs=np.concatenate([x for _,x in ap]); m=scs.max(); w=np.exp(scs-m); w/=w.sum()
                ll+=m+np.log(np.exp(scs-m).sum()); off=0
                for arr,sc in ap:
                    n=len(sc); ww=w[off:off+n]; off+=n
                    for j in range(W): np.add.at(new[j],arr[j:j+n],ww)
            pwm=new/new.sum(1,keepdims=True)
        if ll>bll: bll=ll; best=pwm
    return best
def score(arr,lo,W=20):
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    best=-1e18
    for a in (arr,rc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        best=max(best,sc.max())
    return float(best)
def cons(p): return "".join("ACGT"[i] for i in p.argmax(1))
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if len(pos)<3 or len(neg)<3: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

near_c={}; far_c={}
def gN(n):
    if n not in near_c: near_c[n]=ortho_ups(n,"NEAR")
    return near_c[n]
def gF(n):
    if n not in far_c: far_c[n]=ortho_ups(n,"FAR")
    return far_c[n]
def has_near(n): return len(gN(n))>=1
alln=[n for n in genes if has_near(n)]
print(f"genes with a NEAR ortholog: {len(alln)}  (NEAR={NEAR})")
W=20
testable=[(tf,[t for t in tg if t in genes and has_near(t)]) for tf,tg in tf_targets.items()]
testable=[(tf,ts) for tf,ts in testable if len(ts)>=12]; testable.sort(key=lambda x:-len(x[1]))
print(f"testable TFs: {len(testable)}\n")
def zc(d):
    a=np.array(list(d.values()),float); mu=a.mean(); sd=a.std() or 1
    return {k:(v-mu)/sd for k,v in d.items()}
def fscore(ous,lo):
    sc=[score(u,lo,W) for u in ous]; sc=[x for x in sc if x is not None]
    return max(sc) if sc else None

rows=[]
for tf,tgts in testable[:35]:
    tgts=list(tgts); rng.shuffle(tgts); half=len(tgts)//2; train,test=tgts[:half],tgts[half:]
    pwm=em([ups_ec(t) for t in train],W=W)
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    negs=list(rng.choice([n for n in alln if n not in tf_targets[tf]],size=min(len(test)*4,160),replace=False))
    cand=test+negs; yt={n:(1 if n in set(test) else 0) for n in cand}
    sp={n:score(ups_ec(n),lo,W) for n in cand}
    sn={n:fscore(gN(n),lo) for n in cand}
    sf={n:fscore(gF(n),lo) for n in cand}
    keep=[n for n in cand if sp[n] is not None and sn[n] is not None]
    pos=[n for n in keep if yt[n]]; neg=[n for n in keep if not yt[n]]
    if len(pos)<3 or len(neg)<3: continue
    zp=zc({n:sp[n] for n in keep}); zn=zc({n:sn[n] for n in keep})
    a_pwm=auc([sp[n] for n in pos],[sp[n] for n in neg])
    a_near=auc([sn[n] for n in pos],[sn[n] for n in neg])
    keepf=[n for n in keep if sf[n] is not None]; posf=[n for n in keepf if yt[n]]; negf=[n for n in keepf if not yt[n]]
    a_far=auc([sf[n] for n in posf],[sf[n] for n in negf]) if posf and negf else None
    a_comb=auc([zp[n]+zn[n] for n in pos],[zp[n]+zn[n] for n in neg])
    a_max=auc([max(zp[n],zn[n]) for n in pos],[max(zp[n],zn[n]) for n in neg])
    rows.append(dict(tf=tf,n=len(tgts),npos=len(pos),pwm=a_pwm,near=a_near,far=a_far,comb=a_comb,maxz=a_max,motif=cons(pwm)))
    fp=f"{a_far:.3f}" if a_far is not None else "  -  "
    print(f"  {tf:<8} n={len(tgts):>3} pwm={a_pwm:.3f} near={a_near:.3f} far={fp} comb={a_comb:.3f} max={a_max:.3f}")
def mean(k):
    v=[r[k] for r in rows if r[k] is not None]; return round(float(np.mean(v)),3)
print(f"\n=== {len(rows)} TFs ===")
print(f"  PWM (E.coli only)        = {mean('pwm')}")
print(f"  NEAR footprint (Entero)  = {mean('near')}")
print(f"  FAR footprint (control)  = {mean('far')}")
print(f"  PWM + NEAR (z-sum)       = {mean('comb')}")
print(f"  PWM + NEAR (z-max)       = {mean('maxz')}")
won=sum(1 for r in rows if r['comb'] and r['comb']>r['pwm'])
print(f"  combined beats PWM in {won}/{len(rows)} TFs")
json.dump(dict(n_tf=len(rows),near=NEAR,far=FAR,mean_pwm=mean('pwm'),mean_near=mean('near'),
               mean_far=mean('far'),mean_comb=mean('comb'),mean_maxz=mean('maxz'),
               combined_beats_pwm=won,per_tf=rows),open(OUT/"tf_footprint_v2.json","w"),indent=2)
print(f"\nwrote {OUT}/tf_footprint_v2.json")
