"""MODEL 4 — perturbation-response predictor.

Perturb-seq (Replogle 2022) measured the transcriptional KO signature for ~2,000 genes. This
model learns  static gene features -> perturbation-outcome space  (PCA of the measured
signatures), VALIDATES on held-out perturbed genes (neighbor recall@10 vs random — the honest
number), then PREDICTS the outcome position for the ~14,000 genes that were never perturbed ->
each one's predicted functional neighbors + predicted affected pathways.

Novelty is honest by construction: the held-out recall says how much real structure the model
learned. If it is ~random we say so (as we did for Geneformer). Reuses perturbseq_gwps_bulk.h5ad
(fetched by the notebook); skips gracefully if absent so the pipeline still completes on machines
without it. -> model4_predictions.json
"""
import json, numpy as np
from collections import defaultdict, Counter
from pathlib import Path
OUT=Path("outputs/orphan"); H=Path("data/external_data/human")

# ---------------------------------------------------------------- features
def load_features():
    """Per-gene static feature matrix from the assembled cell model. Returns (names, X, meta)."""
    D=json.load(open(OUT/"cell_complete.json"))
    G=D["genes"]; N=len(G)
    idx={g["name"]:i for i,g in enumerate(G)}
    # network-derived degrees
    rout=np.zeros(N); rin=np.zeros(N)
    for e in D.get("reg",[]): rout[e[0]]+=1; rin[e[1]]+=1
    codep=D.get("codep",{}); g2c=D.get("gene2cplx",{})
    comps=sorted({g["comp"] for g in G}); procs=sorted({g["proc"] for g in G})
    ci={c:k for k,c in enumerate(comps)}; pi={p:k for k,p in enumerate(procs)}
    med_loeuf=np.median([g["loeuf"] for g in G if g["loeuf"]>=0] or [1.0])
    rows=[]
    for i,g in enumerate(G):
        loeuf=g["loeuf"] if g["loeuf"]>=0 else med_loeuf
        ess=1.0 if g["ess"]==1 else 0.0
        num=[loeuf, g["tf"], np.log1p(g["ppi"]), g["ndis"], g["npath"], g["cpg"],
             np.log1p(g["enh"]), g.get("dep_frac",0) or 0, np.log1p(g["pubs"]), ess,
             np.log1p(len(codep.get(str(i),codep.get(i,[])))),
             1.0 if (str(i) in g2c or i in g2c) else 0.0,
             np.log1p(rout[i]), np.log1p(rin[i])]
        oh=[0.0]*(len(comps)+len(procs))
        oh[ci[g["comp"]]]=1.0; oh[len(comps)+pi[g["proc"]]]=1.0
        rows.append(num+oh)
    X=np.array(rows,dtype=float)
    meta=dict(idx=idx, path=[ (g.get("path") or g.get("proc") or "other") for g in G ],
              dark=[g["dark"] for g in G], names=[g["name"] for g in G])
    return meta["names"], X, meta

# ---------------------------------------------------------------- targets
def load_targets_perturbseq(names_idx, k=50):
    """Measured KO signatures -> PCA(k). Returns (gene_names_measured, Y) aligned to features,
    or (None,None) if the h5ad is absent."""
    f=H/"perturbseq_gwps_bulk.h5ad"
    if not f.exists(): return None,None
    import anndata as ad, scipy.sparse as sp
    A=ad.read_h5ad(f); obs=A.obs
    gcol=next((c for c in ["gene","gene_symbol","target_gene","perturbation","perturbed_gene"] if c in obs.columns),None)
    perts=(obs[gcol] if gcol else obs.index).astype(str).values
    Xm=A.X; Xm=np.asarray(Xm.todense()) if sp.issparse(Xm) else np.asarray(Xm)
    bad=np.array([any(t in p.lower() for t in ["non-targeting","control","ntc","scramble","safe"]) or p in ("","nan") for p in perts])
    Xm=Xm[~bad]; perts=perts[~bad]
    uniq=defaultdict(list)
    for i,g in enumerate(perts): uniq[g].append(i)
    genes=[g for g in uniq if g in names_idx]              # only genes we also have features for
    M=np.vstack([Xm[uniq[g]].mean(0) for g in genes])
    M=np.nan_to_num(M); M=(M-M.mean(1,keepdims=True))/(M.std(1,keepdims=True)+1e-6)
    # PCA to k dims (SVD on centered signatures)
    Mc=M-M.mean(0,keepdims=True)
    U,S,Vt=np.linalg.svd(Mc,full_matrices=False)
    Y=U[:,:k]*S[:k]
    return genes, Y

# ---------------------------------------------------------------- train / validate
def _topk_neighbors(E, k=10):
    """cosine top-k neighbor indices for every row of E."""
    En=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
    S=En@En.T; np.fill_diagonal(S,-9)
    return np.argsort(-S,axis=1)[:,:k]

def train_validate(Xtr, Ytr, folds=5, seed=0):
    """Out-of-fold predictions + honest neighbor-recall@10 vs random."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import KFold
    n=len(Xtr); oof=np.zeros_like(Ytr)
    kf=KFold(folds,shuffle=True,random_state=seed)
    r2s=[]
    for tr,te in kf.split(Xtr):
        sc=StandardScaler().fit(Xtr[tr])
        m=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-3,max_iter=400,random_state=seed)
        m.fit(sc.transform(Xtr[tr]),Ytr[tr])
        pred=m.predict(sc.transform(Xtr[te])); oof[te]=pred
        ss_res=((Ytr[te]-pred)**2).sum(); ss_tot=((Ytr[te]-Ytr[te].mean(0))**2).sum()
        r2s.append(1-ss_res/(ss_tot+1e-9))
    true_nb=_topk_neighbors(Ytr,10); pred_nb=_topk_neighbors(oof,10)
    recall=np.mean([len(set(true_nb[i])&set(pred_nb[i]))/10 for i in range(n)])
    rand=10/max(1,n-1)
    return dict(recall_at10=float(recall), random_at10=float(rand),
                lift=float(recall/rand), pc_r2=float(np.mean(r2s)), n_train=n)

def fit_full(Xtr, Ytr, seed=0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    sc=StandardScaler().fit(Xtr)
    m=MLPRegressor(hidden_layer_sizes=(256,128),alpha=1e-3,max_iter=500,random_state=seed)
    m.fit(sc.transform(Xtr),Ytr)
    return sc,m

# ---------------------------------------------------------------- predict all
def predict_all(names, X, meta, measured_names, Y, val, k=10):
    """Predict outcome position for every gene; report neighbors + affected pathway for the
    genes that were NEVER perturbed (the novel predictions)."""
    idx=meta["idx"]; mi=[idx[g] for g in measured_names]
    sc,m=fit_full(X[mi],Y)
    P=m.predict(sc.transform(X))                       # predicted embedding for ALL genes
    Pn=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9)
    measured_set=set(measured_names)
    path=meta["path"]; conf_scale=min(1.0,val["lift"]/3)  # calibrate wording to measured lift
    out={}
    # neighbor search over the whole predicted space
    for i,g in enumerate(names):
        if g in measured_set: continue                 # only genes with no measurement
        sims=Pn@Pn[i]; sims[i]=-9
        nb=np.argsort(-sims)[:k]
        nb=[j for j in nb if sims[j]>0.2][:k]
        if not nb: continue
        votes=Counter(); ev=[]
        for j in nb:
            p=path[j]
            if p and p!="other": votes[p]+=1; ev.append(names[j])
        pred=votes.most_common(1)[0][0] if votes else "unclear"
        cnt=votes.most_common(1)[0][1] if votes else 0
        conf="high" if (cnt>=4 and val["lift"]>=1.5) else ("medium" if cnt>=3 else "low")
        out[g]=dict(pred=pred, neighbors=[names[j] for j in nb][:8], ev=ev[:5], conf=conf)
    return out

def main():
    names,X,meta=load_features()
    measured,Y=load_targets_perturbseq(meta["idx"])
    if measured is None:
        print("Model 4: perturbseq h5ad absent -> skipping (runs on Colab). No model4_predictions.json written.")
        return
    # keep only measured genes present in features, aligned
    mi=[meta["idx"][g] for g in measured]
    Xtr=X[mi]
    val=train_validate(Xtr,Y)
    print(f"Model 4 validation (held-out perturbed genes): neighbor recall@10={val['recall_at10']:.3f} "
          f"vs random {val['random_at10']:.4f}  =>  {val['lift']:.1f}x   | PC R2={val['pc_r2']:.3f} | trained on {val['n_train']} genes")
    if val["lift"]<1.3:
        print("  ** honest caveat: lift < 1.3x — the model barely beats random; predictions are weak, treat as noise-adjacent.")
    preds=predict_all(names,X,meta,measured,Y,val)
    payload=dict(_meta=val, predictions=preds)
    json.dump(payload,open(OUT/"model4_predictions.json","w"))
    print(f"Model 4: predicted perturbation response for {len(preds)} never-perturbed genes -> model4_predictions.json")

if __name__=="__main__":
    main()
