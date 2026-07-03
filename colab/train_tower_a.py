"""TRANSFORMER Tower A — prototype + honest link-prediction test (sandbox, no GPU).

Tests the core claim: is the multi-omic KG structurally predictive, and do node features help
generalize? Builds a graph embedding (truncated SVD of the normalized adjacency = the linear analog of
a GNN's message passing), fuses node features, trains a logistic edge classifier, and evaluates
held-out edges (leakage-free: the embedding is built on the training graph only) against baselines.

The real Tower A is a torch heterogeneous GNN (train_tower_a_gnn.py, Colab); this validates the concept
and gives an honest AUC before investing GPU. -> prints AUCs.
"""
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
OUT=Path("outputs/orphan"); rng=np.random.default_rng(0)

def load():
    pairs={}
    for l in open(OUT/"kg_edges.tsv"):
        a,b,t,w,s=l.rstrip("\n").split("\t"); a,b=int(a),int(b); w=float(w)
        k=(a,b) if a<b else (b,a)
        pairs[k]=max(pairs.get(k,0),abs(w) or 1.0)
    z=np.load(OUT/"kg_nodes.npz"); F=z["features"]
    return list(pairs.items()), F

def embed(train_pairs, N, k=64):
    r=[];c=[];d=[]
    for (a,b),w in train_pairs:
        r+=[a,b]; c+=[b,a]; d+=[w,w]
    A=csr_matrix((d,(r,c)),shape=(N,N))
    deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
    Dinv=1/np.sqrt(deg)
    An=A.multiply(Dinv[:,None]).multiply(Dinv[None,:])          # symmetric-normalized adjacency
    U,S,_=svds(An.astype(np.float32), k=k)
    return U*np.sqrt(S)

def neg_pairs(pos_set, N, n):
    out=set()
    while len(out)<n:
        a,b=int(rng.integers(N)),int(rng.integers(N))
        if a==b: continue
        k=(a,b) if a<b else (b,a)
        if k in pos_set or k in out: continue
        out.add(k)
    return list(out)

def edgefeat(E, pairs):
    a=np.array([p[0] for p in pairs]); b=np.array([p[1] for p in pairs])
    return np.concatenate([E[a]*E[b], np.abs(E[a]-E[b])], axis=1)

def evaluate(name, Xtr,ytr,Xte,yte):
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=1000,C=1.0).fit(sc.transform(Xtr),ytr)
    auc=roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:,1])
    print(f"  {name:34} AUC={auc:.3f}")
    return auc

def main():
    pairs,F=load(); N=F.shape[0]
    pos=[p for p,_ in pairs]; pos_set=set(pos)
    rng.shuffle(pos)
    cut=int(len(pos)*0.85); tr_pos,te_pos=pos[:cut],pos[cut:]
    print(f"Tower A test: {N} nodes, {len(pos)} undirected edges | train {len(tr_pos)} / test {len(te_pos)}")
    # embedding on TRAIN graph only (no leakage)
    E=embed([((a,b),1.0) for a,b in tr_pos], N)
    # standardize features
    Fs=(F-F.mean(0))/(F.std(0)+1e-6)
    tr_neg=neg_pairs(pos_set,N,len(tr_pos)); te_neg=neg_pairs(pos_set,N,len(te_pos))
    ytr=np.r_[np.ones(len(tr_pos)),np.zeros(len(tr_neg))]
    yte=np.r_[np.ones(len(te_pos)),np.zeros(len(te_neg))]
    # (1) structural embedding only
    Xtr=edgefeat(E,tr_pos+tr_neg); Xte=edgefeat(E,te_pos+te_neg)
    a1=evaluate("KG embedding (structure)",Xtr,ytr,Xte,yte)
    # (2) structural + node features (the generalization lever)
    EF=np.concatenate([E,Fs],axis=1)
    Xtr2=edgefeat(EF,tr_pos+tr_neg); Xte2=edgefeat(EF,te_pos+te_neg)
    a2=evaluate("KG embedding + node features",Xtr2,ytr,Xte2,yte)
    # (3) honest baseline: Adamic-Adar + common-neighbours from the train graph (no learning)
    from collections import defaultdict
    adj=defaultdict(set)
    for a,b in tr_pos: adj[a].add(b); adj[b].add(a)
    deg={n:len(s) for n,s in adj.items()}
    def struct(p):
        a,b=p; common=adj[a]&adj[b]
        aa=sum(1/np.log(deg[x]+1e-9) for x in common if deg[x]>1)
        return [aa, len(common), np.log1p(len(adj[a]))*np.log1p(len(adj[b]))]   # AA, #common, deg-product
    Str_tr=np.array([struct(p) for p in tr_pos+tr_neg]); Str_te=np.array([struct(p) for p in te_pos+te_neg])
    a3=roc_auc_score(yte,Str_te[:,0]); print(f"  {'Adamic-Adar baseline':34} AUC={a3:.3f}")
    # (4) HYBRID: learned embedding + node features + structural heuristics (the improvement)
    Xtr4=np.concatenate([edgefeat(EF,tr_pos+tr_neg),Str_tr],axis=1)
    Xte4=np.concatenate([edgefeat(EF,te_pos+te_neg),Str_te],axis=1)
    a4=evaluate("HYBRID (embedding+features+struct)",Xtr4,ytr,Xte4,yte)
    print(f"\nverdict: KG is {'PREDICTIVE' if a2>0.7 else 'weak'} (embedding+feat AUC {a2:.3f}); "
          f"features add +{a2-a1:.3f} over structure; AA heuristic {a3:.3f}; "
          f"HYBRID {a4:.3f} {'BEATS' if a4>max(a2,a3)+0.005 else 'ties'} both -> "
          f"{'the learned model + structure together win' if a4>max(a2,a3)+0.005 else 'heuristics already capture the signal'}.")

if __name__=="__main__":
    main()
