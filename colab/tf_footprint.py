"""Solve TF->genome by ADDING bits the single motif lacks: phylogenetic
footprinting. A real binding site is conserved in the upstream of the
ORTHOLOGOUS gene across related genomes; a spurious PWM match is not.

Baseline (pwm_denovo_test): learned PWM on E. coli upstreams = AUC 0.555 (the
Wunderlich-Mirny wall). Here, same learned PWM, but each candidate is also
scored by how strongly the motif recurs in the upstreams of that gene's
orthologs across the feba panel (62 genomes). Compare:
   PWM(E.coli only)  vs  FOOTPRINT(ortholog conservation)  vs  COMBINED
validated on RegulonDB held-out targets. Does conservation break the wall?
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
def revcomp_enc(a):
    comp=np.array([3,2,1,0,-1]); return comp[a][::-1]

# ---- E. coli genome ----
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
gc=(G.count("G")+G.count("C"))/GLEN; bg=np.array([(1-gc)/2,gc/2,gc/2,(1-gc)/2])
print(f"E.coli genome {GLEN}bp GC={gc:.3f}")

# ---- genes (coords) + name->b ----
genes={}
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
name2b={}
for r in csv.DictReader(open(REG/"TRN.csv")):
    if r.get("gene_name") and r.get("gene_id"): name2b[r["gene_name"].lower()]=r["gene_id"]
tf_targets=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf_targets[p[0].lower()].add(p[1].lower())

def upstream_ec(name,pad=250):
    if name not in genes: return None
    s,e,strand=genes[name]
    if strand=="forward": return Genc[max(0,s-pad-1):s-1]
    return revcomp_enc(Genc[e:min(GLEN,e+pad)])

# ---- feba: b -> Keio locus; ortholog upstreams across related orgs ----
b2kl={}
for lid,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'"):
    b2kl[sn]=lid
# footprint organisms: related panel with good ortholog coverage
FOOT=[o for o,c in cur.execute("SELECT orgId2,COUNT(*) c FROM Ortholog WHERE orgId1='Keio' GROUP BY orgId2 HAVING c>=400 ORDER BY c DESC")]
print(f"footprint orgs ({len(FOOT)}): {FOOT}")
# preload coords + scaffolds for FOOT orgs
coord={}; scaf={}
for org in FOOT:
    for lid,sc,b,e,st in cur.execute("SELECT locusId,scaffoldId,begin,end,strand FROM Gene WHERE orgId=? AND type=1",(org,)):
        coord[(org,lid)]=(sc,b,e,st)
    for sc,s in cur.execute("SELECT scaffoldId,sequence FROM ScaffoldSeq WHERE orgId=?",(org,)):
        scaf[(org,sc)]=s.upper()
print(f"  loaded {len(coord)} ortholog gene coords, {len(scaf)} scaffolds")
# Keio locus -> [(org,locus)] in FOOT
ortho=defaultdict(list); FOOTset=set(FOOT)
for o2,l2,l1 in cur.execute("SELECT orgId2,locusId2,locusId1 FROM Ortholog WHERE orgId1='Keio'"):
    if o2 in FOOTset: ortho[l1].append((o2,l2))

def upstream_feba(org,lid,pad=250):
    c=coord.get((org,lid));
    if not c: return None
    sc,b,e,st=c; s=scaf.get((org,sc))
    if not s: return None
    if st=='+': sub=s[max(0,b-pad-1):b-1]
    else: sub=s[e:min(len(s),e+pad)]
    a=enc(sub)
    return revcomp_enc(a) if st=='-' else a

def ortho_ups(name):
    b=name2b.get(name); kl=b2kl.get(b) if b else None
    if kl is None: return []
    out=[]
    for org,l2 in ortho.get(kl,[]):
        u=upstream_feba(org,l2)
        if u is not None and len(u)>=20 and (u>=0).all(): out.append(u)
    return out

# ---- EM OOPS PWM (same as pwm_denovo) ----
def em_motif(seqs,W=20,n_iter=25,restarts=3):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
    if len(seqs)<4: return None
    best=None; bll=-1e18
    for _ in range(restarts):
        s0=seqs[rng.integers(len(seqs))]; st=rng.integers(0,len(s0)-W+1)
        counts=np.ones((W,4))*0.5
        for j in range(W): counts[j,s0[st+j]]+=1
        pwm=counts/counts.sum(1,keepdims=True)
        for it in range(n_iter):
            new=np.ones((W,4))*0.25; ll=0.0
            lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
            for s in seqs:
                allpos=[]
                for arr in (s,revcomp_enc(s)):
                    n=len(arr)-W+1
                    if n<=0: continue
                    sc=np.zeros(n)
                    for j in range(W): sc+=lo[j,arr[j:j+n]]
                    allpos.append((arr,sc))
                if not allpos: continue
                scs=np.concatenate([sc for _,sc in allpos]); m=scs.max()
                w=np.exp(scs-m); w/=w.sum(); ll+=m+np.log(np.exp(scs-m).sum()); off=0
                for arr,sc in allpos:
                    n=len(sc); ww=w[off:off+n]; off+=n
                    for j in range(W): np.add.at(new[j],arr[j:j+n],ww)
            pwm=new/new.sum(1,keepdims=True)
        if ll>bll: bll=ll; best=pwm
    return best
def score(arr,lo,W=20):
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    best=-1e18
    for a in (arr,revcomp_enc(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        best=max(best,sc.max())
    return float(best)
def consensus(p): return "".join("ACGT"[i] for i in p.argmax(1))
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if not pos or not neg: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

# candidates must have coords AND >=1 ortholog (footprint defined) for fairness
def has_foot(name): return len(ortho_ups(name))>=1
all_names=[n for n in genes if has_foot(n)]
print(f"genes with coords+ortholog (footprint-eligible): {len(all_names)}")
foot_cache={}
def get_ortho(name):
    if name not in foot_cache: foot_cache[name]=ortho_ups(name)
    return foot_cache[name]

W=20
testable=[(tf,[t for t in tg if t in genes and has_foot(t)]) for tf,tg in tf_targets.items()]
testable=[(tf,ts) for tf,ts in testable if len(ts)>=12]
testable.sort(key=lambda x:-len(x[1]))
print(f"testable TFs (>=12 footprint-eligible targets): {len(testable)}\n")

def zc(d):
    a=np.array(list(d.values()),float); mu=a.mean(); sd=a.std() or 1
    return {k:(v-mu)/sd for k,v in d.items()}

rows=[]
for tf,tgts in testable[:35]:
    tgts=list(tgts); rng.shuffle(tgts); half=len(tgts)//2
    train,test=tgts[:half],tgts[half:]
    pwm=em_motif([upstream_ec(t) for t in train],W=W)
    if pwm is None: continue
    lo=np.log(pwm+1e-9)-np.log(bg+1e-9)
    negs=list(rng.choice([n for n in all_names if n not in tf_targets[tf]],
                         size=min(len(test)*4,160),replace=False))
    cand=test+negs; ytrue={n:(1 if n in set(test) else 0) for n in cand}
    s_pwm={}; s_foot={}
    for n in cand:
        s_pwm[n]=score(upstream_ec(n),lo,W)
        ous=get_ortho(n); sc=[score(u,lo,W) for u in ous]; sc=[x for x in sc if x is not None]
        s_foot[n]=float(np.mean(sc)) if sc else None
    # keep only cands with both defined
    keep=[n for n in cand if s_pwm[n] is not None and s_foot[n] is not None]
    if sum(ytrue[n] for n in keep)<3 or len(keep)-sum(ytrue[n] for n in keep)<3: continue
    zp=zc({n:s_pwm[n] for n in keep}); zf=zc({n:s_foot[n] for n in keep})
    pos=[n for n in keep if ytrue[n]]; neg=[n for n in keep if not ytrue[n]]
    a_pwm=auc([s_pwm[n] for n in pos],[s_pwm[n] for n in neg])
    a_foot=auc([s_foot[n] for n in pos],[s_foot[n] for n in neg])
    a_comb=auc([zp[n]+zf[n] for n in pos],[zp[n]+zf[n] for n in neg])
    rows.append(dict(tf=tf,n=len(tgts),npos=len(pos),nneg=len(neg),
                     auc_pwm=a_pwm,auc_foot=a_foot,auc_comb=a_comb,motif=consensus(pwm)))
    print(f"  {tf:<8} n={len(tgts):>3} pwm={a_pwm:.3f} foot={a_foot:.3f} comb={a_comb:.3f}  {consensus(pwm)}")

def mean(k):
    v=[r[k] for r in rows if r[k] is not None]; return round(float(np.mean(v)),3)
print(f"\n=== {len(rows)} TFs (footprint via {len(FOOT)} orthologous genomes) ===")
print(f"  PWM (E.coli only)   mean AUC = {mean('auc_pwm')}   (matches wall ~0.555)")
print(f"  FOOTPRINT (ortholog) mean AUC = {mean('auc_foot')}")
print(f"  COMBINED            mean AUC = {mean('auc_comb')}")
won=sum(1 for r in rows if r['auc_comb'] and r['auc_pwm'] and r['auc_comb']>r['auc_pwm'])
print(f"  combined beats PWM in {won}/{len(rows)} TFs")
json.dump(dict(n_tf=len(rows),foot_orgs=FOOT,
               mean_auc_pwm=mean('auc_pwm'),mean_auc_foot=mean('auc_foot'),mean_auc_comb=mean('auc_comb'),
               combined_beats_pwm=won,per_tf=rows),open(OUT/"tf_footprint.json","w"),indent=2)
print(f"\nwrote {OUT}/tf_footprint.json")
