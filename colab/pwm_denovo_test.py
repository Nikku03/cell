"""Decisive test: does a REAL (learned) PWM separate TF targets from non-targets?

Family-consensus Phi failed (AUC 0.505). Here we LEARN each TF's actual motif
from its own RegulonDB targets (cross-validated), then test. This isolates the
one variable: is binding-site prediction failing because our Phi was too soft
(family consensus), or because it's fundamentally hard even with the real motif?

Per TF (>=12 coord-mapped RegulonDB targets):
  1. split targets 50/50 -> train / test
  2. discover motif via EM (OOPS, one-occurrence-per-sequence) on the 250bp
     upstream regions of TRAIN targets (both strands), width W=20
  3. build log-odds PWM vs genome background
  4. score: max-window log-odds for TEST-target upstreams (pos) vs random
     non-target upstreams (neg)  -> AUC
  5. report learned consensus + information content (sanity: CRP/LexA should
     rediscover their known palindromes)

Compare AUC_learned_PWM vs family-consensus (0.505) vs adjacency.
Output: outputs/orphan/pwm_denovo_test.json
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REG=Path("data/external_data/regulon"); OUT=Path("outputs/orphan")
rng=np.random.default_rng(11)

# ---- genome ----
seq=[]
for l in open("data/external_data/ecoli_genome/ecoli_K12.fna"):
    if not l.startswith(">"): seq.append(l.strip())
G="".join(seq).upper()
GLEN=len(G)
B={"A":0,"C":1,"G":2,"T":3}
def enc(s):
    return np.array([B.get(c,-1) for c in s], dtype=np.int8)
# background base freq
gc=(G.count("G")+G.count("C"))/GLEN
bg=np.array([(1-gc)/2,gc/2,gc/2,(1-gc)/2])  # A,C,G,T
print(f"genome {GLEN}bp  GC={gc:.3f}  bg={bg.round(3)}")
Genc=enc(G)
def revcomp_enc(a):
    # complement: A0<->T3, C1<->G2 ; then reverse
    comp=np.array([3,2,1,0,-1])
    return comp[a][::-1]

# ---- genes + truth ----
genes={}
for l in open(REG/"GeneProductSet.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t")
    if len(p)>=6 and p[3].isdigit() and p[4].isdigit() and p[5] in ("forward","reverse"):
        genes[p[1].lower()]=(int(p[3]),int(p[4]),p[5])
tf_targets=defaultdict(set)
for l in open(REG/"network_tf_gene.txt"):
    if not l.strip() or l.startswith("#"): continue
    p=l.rstrip("\n").split("\t"); tf_targets[p[0].lower()].add(p[1].lower())

def upstream(name, pad=250):
    if name not in genes: return None
    s,e,strand=genes[name]
    if strand=="forward":
        a=max(0,s-pad-1); arr=Genc[a:s-1]
    else:
        a=e; arr=revcomp_enc(Genc[a:min(GLEN,e+pad)])
    return arr

# ---- EM OOPS motif finder (numpy) ----
def em_motif(seqs, W=20, n_iter=25, restarts=3):
    seqs=[s for s in seqs if s is not None and len(s)>=W and (s>=0).all()]
    if len(seqs)<4: return None
    best_pwm=None; best_ll=-1e18
    for r in range(restarts):
        # init PWM from a random seed window
        s0=seqs[rng.integers(len(seqs))]
        st=rng.integers(0,len(s0)-W+1)
        counts=np.ones((W,4))*0.5
        for j in range(W): counts[j,s0[st+j]]+=1
        pwm=counts/counts.sum(1,keepdims=True)
        for it in range(n_iter):
            new=np.ones((W,4))*0.25  # pseudocount
            ll=0.0
            for s in seqs:
                L=len(s)
                # consider both strands
                cand=[s, revcomp_enc(s)]
                # score all windows on both strands
                allpos=[]
                for strand_arr in cand:
                    n=len(strand_arr)-W+1
                    if n<=0: continue
                    # log-odds score per window
                    logodds=np.log(pwm+1e-9)-np.log(bg+1e-9)  # W x4
                    sc=np.zeros(n)
                    for j in range(W):
                        sc += logodds[j, strand_arr[j:j+n]]
                    allpos.append((strand_arr,sc))
                if not allpos: continue
                # responsibilities = softmax over all windows both strands
                scs=np.concatenate([sc for _,sc in allpos])
                m=scs.max(); w=np.exp(scs-m); w/=w.sum()
                ll+=m+np.log(np.exp(scs-m).sum())
                # distribute back
                off=0
                for strand_arr,sc in allpos:
                    n=len(sc); ww=w[off:off+n]; off+=n
                    for j in range(W):
                        bases=strand_arr[j:j+n]
                        np.add.at(new[j], bases, ww)
            pwm=new/new.sum(1,keepdims=True)
        if ll>best_ll: best_ll=ll; best_pwm=pwm
    return best_pwm

def score_seq(arr, logodds, W):
    if arr is None or len(arr)<W or not (arr>=0).all(): return -1e9
    best=-1e18
    for strand_arr in (arr, revcomp_enc(arr)):
        n=len(strand_arr)-W+1
        if n<=0: continue
        sc=np.zeros(n)
        for j in range(W):
            sc+=logodds[j, strand_arr[j:j+n]]
        best=max(best, sc.max())
    return float(best)

def consensus(pwm):
    return "".join("ACGT"[i] for i in pwm.argmax(1))
def infocontent(pwm):
    return float(np.sum(pwm*np.log2((pwm+1e-9)/0.25)))

def auc(pos,neg):
    if not pos or not neg: return 0.5
    s=np.concatenate([pos,neg]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(len(s)); p=y==1
    return float((r[p].sum()-p.sum()*(p.sum()-1)/2)/(p.sum()*(~p).sum()))

# ---- run ----
all_names=list(genes.keys())
W=20
testable=[(tf,[t for t in tg if t in genes]) for tf,tg in tf_targets.items()]
testable=[(tf,ts) for tf,ts in testable if len(ts)>=12 and tf in genes]
testable.sort(key=lambda x:-len(x[1]))
print(f"testable TFs (>=12 coord targets): {len(testable)}\n")

results=[]
for tf, tgts in testable[:40]:
    tgts=list(tgts); rng.shuffle(tgts)
    half=len(tgts)//2
    train, test = tgts[:half], tgts[half:]
    pwm=em_motif([upstream(t) for t in train], W=W)
    if pwm is None: continue
    logodds=np.log(pwm+1e-9)-np.log(bg+1e-9)
    pos=[score_seq(upstream(t),logodds,W) for t in test]
    pos=[p for p in pos if p>-1e8]
    negs=list(rng.choice([n for n in all_names if n not in tf_targets[tf]], size=min(len(test)*4,2000), replace=False))
    neg=[score_seq(upstream(n),logodds,W) for n in negs]
    neg=[x for x in neg if x>-1e8]
    a=auc(pos,neg)
    results.append(dict(tf=tf, n_targets=len(tgts), auc=round(a,3),
                        consensus=consensus(pwm), infocontent=round(infocontent(pwm),1)))
    print(f"  {tf:<8} n={len(tgts):>3} AUC={a:.3f}  IC={infocontent(pwm):>4.1f}  motif={consensus(pwm)}")

aucs=[r["auc"] for r in results]
mean=round(float(np.mean(aucs)),3); med=round(float(np.median(aucs)),3)
strong=[r for r in results if r["auc"]>=0.7]
print(f"\n=== {len(results)} TFs ===")
print(f"  mean AUC (learned PWM) = {mean}   median = {med}")
print(f"  vs family-consensus Phi = 0.505   vs adjacency(local) ~0.68")
print(f"  TFs with AUC>=0.70: {len(strong)}/{len(results)}  ({[r['tf'] for r in strong][:12]})")

# sanity: known palindromes
for r in results:
    if r["tf"] in ("crp","lexa","fnr","argr","purr","fur","lrp"):
        print(f"  [sanity] {r['tf']}: {r['consensus']} (AUC {r['auc']})")

json.dump(dict(n=len(results), W=W, mean_auc=mean, median_auc=med,
               n_strong=len(strong), per_tf=results,
               family_consensus_baseline=0.505),
          open(OUT/"pwm_denovo_test.json","w"), indent=2)
print("\nwrote outputs/orphan/pwm_denovo_test.json")
