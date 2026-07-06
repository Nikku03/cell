"""REVERSE INFERENCE (phenotype -> cause) — the discovery engine.

Input: what the disease DOES to the cell, as a molecular signature (gene -> signed dysregulation score,
+up / -down in diseased vs healthy). Output: a ranked list of CANDIDATE CAUSAL genes/pathways -- the source
of the lesion = the target -- discovered from the phenotype alone, WITHOUT being told the target or mutation.

It inverts the model's forward map by composing independent evidence 'like algebra' -- each lens is a term,
and the cause is where they agree:

  1. MASTER-REGULATOR   : which regulator's regulon (signed targets) is coherently dysregulated? (VIPER-like)
                          -- catches drivers whose TARGETS move even if the driver's own mRNA doesn't.
  2. NETWORK DIFFUSION  : heat-diffuse the |signature| over the DENSE multi-lens network (PPI+reg+co-ess) ->
                          the causal NEIGHBOURHOOD. Catches an upstream cause not itself in the signature
                          (a mutation changes ACTIVITY, not necessarily its own expression).
  3. CASCADE MATCH      : forward-simulate each candidate's perturbation (signed cascade) and correlate with
                          the observed signature -> which single perturbation best REPRODUCES the phenotype.
  4. PRIOR (light)      : LoF-constraint x essentiality as a weak tie-breaker (a real driver tends to matter),
                          never strong enough to override the signature-driven terms.

ANTI-CHEAT: uses only biology/network + the input signature. NEVER the disease-gene label being predicted.
-> callable reverse_infer(signature) ; CLI runs a self-check.
"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def load_model(path=None):
    D=json.load(open(path or (OUT/"cell_complete.json")))
    G=D["genes"]; N=len(G); nm=[g["name"] for g in G]; ni={n:i for i,n in enumerate(nm)}
    regulon=defaultdict(list); targets_of=defaultdict(list)
    for e in D.get("reg",[]):
        if len(e)<2: continue
        s=e[0]; t=e[1]; sg=e[2] if len(e)>2 and e[2] in (1,-1) else 1
        if 0<=s<N and 0<=t<N: regulon[s].append((t,sg)); targets_of[t].append((s,sg))
    # dense multi-lens adjacency (PPI + reg + co-essentiality), symmetric-normalized
    import scipy.sparse as sp
    ew=defaultdict(float)
    def add(a,b,w=1.0):
        if a!=b and 0<=a<N and 0<=b<N: ew[(min(a,b),max(a,b))]+=w
    for e in D.get("ppi",[]):
        if len(e)>=2: add(e[0],e[1])
    for e in D.get("reg",[]):
        if len(e)>=2: add(e[0],e[1])
    for k,v in (D.get("codep") or {}).items():
        for p in v: add(int(k), p[0] if isinstance(p,(list,tuple)) else p)
    r=[];c=[];w=[]
    for (a,b),x in ew.items(): r+=[a,b]; c+=[b,a]; w+=[x,x]
    A=sp.coo_matrix((w,(r,c)),shape=(N,N)).tocsr()+sp.identity(N)
    d=np.asarray(A.sum(1)).ravel(); di=1.0/np.sqrt(np.maximum(d,1e-9))
    A=sp.diags(di)@A@sp.diags(di)
    codep={}                                                 # co-essentiality (INDEPENDENT of the reg network)
    for k,v in (D.get("codep") or {}).items():
        codep[int(k)]=[(p[0],float(p[1])) for p in v if isinstance(p,(list,tuple)) and len(p)>=2]
    prior=np.zeros(N)
    for i,g in enumerate(G):
        lo=g.get("loeuf",-1); pc=(1.0 if (isinstance(lo,(int,float)) and 0<=lo<0.35) else 0.0)
        prior[i]=pc+float(g.get("ess",0))
    return dict(D=D,G=G,N=N,nm=nm,ni=ni,regulon=regulon,targets_of=targets_of,codep=codep,A=A.tocsr(),prior=prior)

def codep_coherence(cands, s, M):
    """INDEPENDENT lens (DepMap co-essentiality, not the reg network): is the candidate co-essential with the
    strongly-dysregulated genes? A true causal gene is functionally coupled to what it dysregulates."""
    codep=M.get("codep",{}); out={}
    dys=set(np.argsort(-np.abs(s))[:250].tolist())
    for g in cands:
        sc=sum(r for j,r in codep.get(g,[]) if j in dys)
        if sc>0: out[g]=sc
    return out

def build_signature_library(lfc, cols, pert_of_row, name_set, rows=None):
    """LINCS reference library: per perturbed gene, the MEAN of its signatures. `rows` restricts to a subset
    (for held-out validation: build from library-half rows, test on the other half). cols = landmark symbols."""
    from collections import defaultdict
    idx=defaultdict(list)
    it = rows if rows is not None else range(len(pert_of_row))
    for i in it:
        p=str(pert_of_row[i])
        if p in name_set: idx[p].append(i)
    lib={g: lfc[r].mean(0) for g,r in idx.items() if r}
    return lib, list(cols)

def build_signature_index(lfc, cols, pert_of_row, name_set, rows=None, rank=True):
    """keep INDIVIDUAL library signatures (not per-gene averages) + their gene labels, for k-NN retrieval.
    Rank-transforms each signature (robust). Returns (matrix[n_sig,978] normalized, gene_labels, cols)."""
    it = rows if rows is not None else range(len(pert_of_row))
    idx=[i for i in it if str(pert_of_row[i]) in name_set]
    X=lfc[idx].astype("float32"); labels=[str(pert_of_row[i]) for i in idx]
    if rank:
        from scipy.stats import rankdata
        X=np.vstack([rankdata(r) for r in X]).astype("float32")
    X=X-X.mean(1,keepdims=True); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
    return X, labels, list(cols)

def knn_signature_match(query_sig, index, M, k=40, rank=True):
    """match the query to its NEAREST individual library signatures and VOTE by gene (sharper than centroid)."""
    X,labels,cols=index
    q=np.array([float(query_sig.get(g,0.0)) for g in cols],dtype="float32")
    if np.linalg.norm(q)==0: return {}
    if rank:
        from scipy.stats import rankdata; q=rankdata(q).astype("float32")
    q=q-q.mean(); q=q/(np.linalg.norm(q)+1e-9)
    sims=X@q                                                 # cosine to every library signature (BLAS)
    top=np.argpartition(-sims, min(k,len(sims)-1))[:k]
    out={}
    for j in top:
        i=M["ni"].get(labels[j])
        if i is not None and sims[j]>0: out[i]=out.get(i,0.0)+float(sims[j])
    return out

def signature_match(query_sig, lib, lib_cols, M, metric="rank"):
    """Match the query signature to each library gene's signature. metric='cosine' (magnitude) or 'rank'
    (Spearman-like: rank-transform both, robust to LINCS magnitude noise -- CMap-style). The disease phenotype
    most RESEMBLES the perturbation that caused it -> gene_idx: similarity."""
    if not lib: return {}
    q=np.array([float(query_sig.get(g,0.0)) for g in lib_cols])
    if np.linalg.norm(q)==0: return {}
    if metric=="rank":
        from scipy.stats import rankdata
        q=rankdata(q); q=q-q.mean()
    nq=np.linalg.norm(q)
    out={}
    for g,vec in lib.items():
        i=M["ni"].get(g)
        if i is None: continue
        v=vec
        if metric=="rank":
            from scipy.stats import rankdata as _rd
            v=_rd(vec); v=v-v.mean()
        nv=np.linalg.norm(v)
        if nv>0 and nq>0: out[i]=float(np.dot(v,q)/(nv*nq))
    return out

def _sig_array(sig, M):
    s=np.zeros(M["N"])
    for k,v in sig.items():
        i=M["ni"].get(k)
        if i is not None:
            try: s[i]=float(v)
            except: pass
    return s

def master_regulators(s, M, min_targets=8):
    out={}
    for R,tgts in M["regulon"].items():
        vals=[sg*s[t] for t,sg in tgts if s[t]!=0]
        if len(vals)>=min_targets:
            out[R]=sum(vals)/math.sqrt(len(vals))            # normalized enrichment; |.| = coherent dysregulation
    return out

def diffuse(s, M, alpha=0.6, iters=12):
    h=np.abs(s).copy(); s0=h.copy()
    if s0.sum()>0: s0/=s0.sum()
    h=s0.copy()
    A=M["A"]
    for _ in range(iters): h=alpha*(A@h)+(1-alpha)*s0
    return h

def cascade_match(cands, s, M):
    """forward signed 2-hop cascade of each candidate; corr with the observed signature."""
    reg=M["regulon"]; out={}
    sig_idx=np.where(s!=0)[0]; sset=set(sig_idx.tolist())
    for g in cands:
        eff=defaultdict(float)
        for t,sg in reg.get(g,[]):
            eff[t]+=sg
            for u,sg2 in reg.get(t,[])[:25]: eff[u]+=0.4*sg*sg2
        if not eff: continue
        common=[t for t in eff if t in sset]
        if len(common)<5: continue
        a=np.array([eff[t] for t in common]); b=np.array([s[t] for t in common])
        if a.std()>0 and b.std()>0: out[g]=abs(float(np.corrcoef(a,b)[0,1]))
    return out

def _z(d):
    if not d: return {}
    v=np.array(list(d.values())); mu,sd=v.mean(),v.std()+1e-9
    return {k:(x-mu)/sd for k,x in d.items()}

def reverse_infer(signature, M=None, top=25, library=None, lib_cols=None, index=None, sig_metric="rank",
                  w_mr=1.0, w_diff=0.6, w_casc=1.0, w_codep=0.8, w_sig=2.5, w_prior=0.3):
    """signature: {gene: signed_score}. Signature-matching lens: k-NN over `index` (sharpest) if given, else
    centroid `library` (CMap). -> ranked candidate causal genes with per-lens breakdown."""
    M=M or load_model(); s=_sig_array(signature,M)
    mr_signed=master_regulators(s,M)                          # compute ONCE (signed = direction)
    mr={g:abs(v) for g,v in mr_signed.items()}
    h=diffuse(s,M); diff={i:h[i] for i in np.argsort(-h)[:300]}
    if index is not None: sigm=knn_signature_match(signature, index, M)         # k-NN retrieval (sharpest)
    elif library: sigm=signature_match(signature, library, lib_cols, M, sig_metric)  # centroid CMap
    else: sigm={}
    cands=set(mr)|set(diff)|set(sigm)
    casc=cascade_match(cands, s, M)
    cdp=codep_coherence(cands, s, M)                          # independent (co-essentiality) lens
    zmr,zdiff,zcasc,zcdp,zsig=_z(mr),_z(diff),_z(casc),_z(cdp),_z(sigm)
    pr=M["prior"]; score={}
    for g in cands:
        score[g]=(w_mr*zmr.get(g,0)+w_diff*zdiff.get(g,0)+w_casc*zcasc.get(g,0)+w_codep*zcdp.get(g,0)
                  +w_sig*zsig.get(g,0)+w_prior*pr[g])
    ranked=sorted(score, key=lambda g:-score[g])[:top]
    G=M["G"]
    res=[dict(gene=G[g]["name"], score=round(score[g],3),
             master_reg=round(mr.get(g,0),2), diffusion=round(diff.get(g,0),4),
             cascade_match=round(casc.get(g,0),3),
             direction=("over-active" if mr_signed.get(g,0)>0 else "under-active/loss"),
             loeuf=G[g].get("loeuf",-1), essential=int(G[g].get("ess",0))) for g in ranked]
    return res

def main():
    import sys
    M=load_model()
    # self-check: perturb a known regulator through the forward cascade, then reverse-infer it
    G=M["G"]; reg=M["regulon"]
    seed=next((i for i in reg if len(reg[i])>=20 and G[i]["name"]=="TP53"), next(iter(reg)))
    sig={}
    for t,sg in reg[seed]:
        sig[G[t]["name"]]=sg*(2.0)
        for u,sg2 in reg.get(t,[])[:10]: sig[G[u]["name"]]=sig.get(G[u]["name"],0)+0.5*sg*sg2
    res=reverse_infer(sig, M, top=15)
    print(f"self-check: seeded a signature from {G[seed]['name']}'s regulon; reverse-inference top candidates:")
    for i,r in enumerate(res[:15]):
        mark=" <== TRUE SOURCE" if r["gene"]==G[seed]["name"] else ""
        print(f"  {i+1:2d}. {r['gene']:12} score {r['score']:+.2f} (MR {r['master_reg']}, diff {r['diffusion']}, casc {r['cascade_match']}){mark}")
    rank=next((i+1 for i,r in enumerate(res) if r["gene"]==G[seed]["name"]), None)
    print(f"  -> {G[seed]['name']} recovered at rank {rank}")

if __name__=="__main__":
    main()
