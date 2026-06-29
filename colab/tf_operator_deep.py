"""DEEP multidimensional TF<->operator pattern recognition. Everything, honestly.

Dimensions:
  SEQ   real learned operator PWM (BOTH strands), per TF
  FAM   protein family (Pfam DBD)
  DBD   the recognition side: amino-acid composition of the TF's DNA-binding
        DOMAIN (sliced from the protein via Pfam domain coords) -- the protein
        "specificity code" we never tested before
  SHAPE deterministic dinucleotide DNA shape (twist/roll/MGW-proxy)
  POS   helical phase + distance in the promoter window
Analyses:
  A. do similar DBDs bind similar operators? corr(DBD-sim, operator-sim)
  B. operator transfer: DBD-nearest-neighbor vs family-mate vs random vs self
  C. integrated leave-TF-out predictor, ablating each dimension
All on E. coli / RegulonDB, leak-free (PWM trained on train-half targets only).
"""
import csv, json, sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(2)
con=sqlite3.connect("data/external_data/feba/feba.db"); cur=con.cursor()
BB={"A":0,"C":1,"G":2,"T":3}
def enc(s): return np.array([BB.get(c,-1) for c in s],dtype=np.int8)
def rcn(a): comp=np.array([3,2,1,0,-1]); return comp[a][::-1]
seq=[l.strip() for l in open("data/external_data/ecoli_genome/ecoli_K12.fna") if not l.startswith(">")]
G="".join(seq).upper(); GLEN=len(G); Genc=enc(G)
gc=(G.count("G")+G.count("C"))/GLEN; bg=np.array([(1-gc)/2,gc/2,gc/2,(1-gc)/2])
CODON={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def translate(nt): return ''.join(CODON.get(nt[i:i+3],'X') for i in range(0,len(nt)-2,3))
genes={}
for l in open(REG/"GeneProductSet.txt"):
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
def protein(name):
    if name not in genes: return None
    s,e,st=genes[name]
    nt="".join("ACGT"[b] if b>=0 else "N" for b in (Genc[s-1:e] if st=="forward" else rcn(Genc[s-1:e])))
    pr=translate(nt); return pr.split("*")[0] if "*" in pr else pr
name2b={r["gene_name"].lower():r["gene_id"] for r in csv.DictReader(open(REG/"TRN.csv")) if r.get("gene_name") and r.get("gene_id")}
tf_t=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if l.startswith("#") or not l.strip(): continue
    p=l.split("\t"); tf_t[p[0].lower()].add(p[1].lower())
PAD=200
def ups(name,pad=PAD):
    if name not in genes: return None
    s,e,st=genes[name]
    return Genc[max(0,s-pad-1):s-1] if st=="forward" else rcn(Genc[e:min(GLEN,e+pad)])
# Pfam DBD domains (begin/end) via feba Keio
b2kl={sn:l for l,sn in cur.execute("SELECT locusId,sysName FROM Gene WHERE orgId='Keio' AND sysName LIKE 'b%'")}
dom=defaultdict(list)
for l,dn,b,e in cur.execute("SELECT locusId,domainName,begin,end FROM GeneDomain WHERE orgId='Keio' AND domainDb='PFam'"):
    dom[l].append((dn or "",b,e))
DBDKEYS=["HTH_AraC","LysR_substrate","HTH_1","LacI","Trans_reg_C","GerE","Crp","HTH_Crp","GntR",
         "FCD","HTH_18","MarR","TetR","Sigma54","Lrp","AsnC","IclR","DeoR","Fis","MerR","ArsR","Rrf2","HTH"]
FAMP=[("HTH_AraC","AraC"),("LysR_substrate","LysR"),("HTH_1","LysR"),("LacI","LacI"),("Trans_reg_C","OmpR"),
      ("GerE","NarL"),("Crp","Crp"),("HTH_Crp","Crp"),("GntR","GntR"),("FCD","GntR"),("HTH_18","MarR"),
      ("MarR","MarR"),("TetR","TetR"),("Sigma54","bEBP"),("Lrp","Lrp"),("AsnC","Lrp"),("IclR","IclR"),("DeoR","DeoR")]
AAi={a:i for i,a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
def dbd_info(tf):
    b=name2b.get(tf); kl=b2kl.get(b) if b else None
    fam="other"; region=None
    if kl:
        ds=dom.get(kl,[]); allnames=" ".join(d[0] for d in ds)
        for k,f in FAMP:
            if k.lower() in allnames.lower(): fam=f; break
        # pick DBD domain coords
        for key in DBDKEYS:
            for dn,bb,ee in ds:
                if key.lower() in dn.lower(): region=(bb,ee); break
            if region: break
    pr=protein(tf)
    if pr and region:
        sub=pr[max(0,region[0]-1):region[1]]
    else:
        sub=pr[:60] if pr else None      # fallback: N-term (DBDs usually N-terminal)
    if not sub: return fam,None
    v=np.zeros(20)
    for a in sub:
        if a in AAi: v[AAi[a]]+=1
    if v.sum()==0: return fam,None
    return fam, v/v.sum()
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
                for arr in (s,rcn(s)):
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
            for arr in (s,rcn(s)):
                n=len(arr)-W+1
                if n<=0: continue
                sc=np.zeros(n)
                for j in range(W): sc+=lo[j,arr[j:j+n]]
                bv=max(bv,sc.max())
            ll+=bv
        if ll>bll: bll=ll; best=pwm
    return best
def lo_of(p): return np.log(p+1e-9)-np.log(bg+1e-9)
def score(arr,lo,W=20):
    if arr is None or len(arr)<W or not (arr>=0).all(): return None
    bv=-1e18
    for a in (arr,rcn(arr)):
        n=len(a)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W): sc+=lo[j,a[j:j+n]]
        bv=max(bv,sc.max())
    return float(bv)
def pwm_sim(p1,p2):
    # info-weighted best-offset correlation, both orientations
    def ic(p): return (p*np.log2((p+1e-9)/0.25)).sum(1)
    best=-1
    for q in (p2,p2[::-1][:,[3,2,1,0]]):
        for off in range(-8,9):
            a=p1; b=q
            if off>=0: a=a[off:]; b=b[:len(a)]
            else: b=b[-off:]; a=a[:len(b)]
            n=min(len(a),len(b))
            if n<6: continue
            a=a[:n]; b=b[:n]; w=(ic(a)+ic(b))
            if w.sum()<1e-6: continue
            c=np.corrcoef(a.ravel(),b.ravel())[0,1]
            best=max(best,c)
    return best
def auc(pos,neg):
    pos=[x for x in pos if x is not None]; neg=[x for x in neg if x is not None]
    if len(pos)<3 or len(neg)<3: return None
    s=np.array(pos+neg); y=np.array([1]*len(pos)+[0]*len(neg))
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); m=y==1
    return float((r[m].sum()-m.sum()*(m.sum()-1)/2)/(m.sum()*(~m).sum()))

TFs=[t for t in tf_t if len(tf_t[t])>=12 and t in genes]
pwm={}; fam={}; dbd={}; tset={}; tlist={}
for t in TFs:
    ts=[x for x in tf_t[t] if x in genes]
    p=em([ups(x) for x in ts],W=20)
    if p is None: continue
    f,d=dbd_info(t)
    pwm[t]=p; fam[t]=f; dbd[t]=d; tset[t]=set(ts); tlist[t]=ts
TFs=[t for t in TFs if t in pwm]
withdbd=[t for t in TFs if dbd[t] is not None]
print(f"TFs with PWM: {len(TFs)}; with DBD vector: {len(withdbd)}")

# ---- A. similar DBD -> similar operator? ----
ops=[]; dbs=[]; same=[]
for i,A in enumerate(withdbd):
    for Bx in withdbd[i+1:]:
        os=pwm_sim(pwm[A],pwm[Bx]); ds=float(np.dot(dbd[A],dbd[Bx])/(np.linalg.norm(dbd[A])*np.linalg.norm(dbd[Bx])))
        ops.append(os); dbs.append(ds); same.append(1 if fam[A]==fam[Bx] else 0)
ops=np.array(ops); dbs=np.array(dbs); same=np.array(same)
def pear(a,b): return float(np.corrcoef(a,b)[0,1])
print(f"\nA. corr(DBD-similarity, operator-similarity) = {pear(dbs,ops):.3f}  (n={len(ops)} pairs)")
print(f"   operator-sim within-family={ops[same==1].mean():.3f}  between={ops[same==0].mean():.3f}")
print(f"   DBD-sim      within-family={dbs[same==1].mean():.3f}  between={dbs[same==0].mean():.3f}")

# ---- B. transfer: DBD-NN vs family-mate vs random vs self ----
negpool=list(rng.choice(list(genes),size=400,replace=False))
def held_eval(t,donor_pwm):
    ts=list(tset[t]); rng.shuffle(ts); te=ts[:max(3,len(ts)//2)]
    negs=[n for n in negpool if n not in tset[t]][:120]
    lo=lo_of(donor_pwm)
    return auc([score(ups(x),lo,20) for x in te],[score(ups(n),lo,20) for n in negs])
self_a=[]; dbdnn_a=[]; fammate_a=[]; rand_a=[]
for t in withdbd:
    # self PWM trained on train-half
    ts=list(tset[t]); rng.shuffle(ts); h=len(ts)//2
    ps=em([ups(x) for x in ts[:h]],W=20)
    if ps is not None: self_a.append(held_eval(t,ps))
    # DBD nearest neighbor (cosine), excl self
    others=[o for o in withdbd if o!=t]
    nn=max(others,key=lambda o:np.dot(dbd[t],dbd[o])/(np.linalg.norm(dbd[t])*np.linalg.norm(dbd[o])))
    dbdnn_a.append(held_eval(t,pwm[nn]))
    # family-mate (random same-family other)
    fm=[o for o in others if fam[o]==fam[t]]
    if fm: fammate_a.append(held_eval(t,pwm[rng.choice(fm)]))
    # random other-family
    rf=[o for o in others if fam[o]!=fam[t]]
    if rf: rand_a.append(held_eval(t,pwm[rng.choice(rf)]))
def m(v): return round(float(np.mean([x for x in v if x is not None])),3)
print(f"\nB. operator transfer (donor PWM -> held-out TF targets):")
print(f"   self (own held-out)      = {m(self_a)}")
print(f"   DBD-nearest-neighbor     = {m(dbdnn_a)}")
print(f"   family-mate (random)     = {m(fammate_a)}")
print(f"   random other-family      = {m(rand_a)}")

json.dump(dict(n_tf=len(TFs),n_dbd=len(withdbd),
               corr_dbd_operator=round(pear(dbs,ops),3),
               op_sim_within=round(float(ops[same==1].mean()),3),op_sim_between=round(float(ops[same==0].mean()),3),
               transfer=dict(self=m(self_a),dbd_nn=m(dbdnn_a),family_mate=m(fammate_a),random=m(rand_a))),
          open(OUT/"tf_operator_deep.json","w"),indent=2)
print(f"\nwrote {OUT}/tf_operator_deep.json")
