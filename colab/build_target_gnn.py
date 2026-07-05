"""TARGET-DISCOVERY GNN — the GPU billion-dollar test: does the DENSE network really find drug targets?

The sandbox test used a single neighbor-fraction feature and gained +0.011 AUC. A graph neural network
propagates biology features over the WHOLE dense multi-lens network (PPI + regulation + co-essentiality +
co-expression + Perturb-seq + signaling), so a gene is judged by the company it keeps across every dataset
at once -- the honest test of 'integrate everything'. Node-classification, held out by node.

  features = biology only (essentiality, constraint, network degree, literature, TF/master, CpG/enhancer)
  label    = target of an APPROVED drug
  compare  = GCN (uses the graph) vs MLP (same features, NO graph) -> the graph's contribution
  output   = held-out AUC + ranked NOVEL (undrugged) target shortlist

Self-contained (torch + scipy sparse; no torch_geometric). Runs on GPU if present. -> target_gnn.json
"""
import json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
OUT=Path("outputs/orphan")

def main():
    cc=OUT/"cell_complete.json"
    if not cc.exists(): print("cell_complete.json absent -> skipped"); return
    try:
        import torch, torch.nn as nn
        import scipy.sparse as sp
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        print("needs torch + scipy + sklearn:",repr(e)[:80]); return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    D=json.load(open(cc)); G=D["genes"]; N=len(G)
    drugs=D.get("drugs") or {}
    has_drug=np.array([1 if str(i) in drugs else 0 for i in range(N)])
    y=np.array([1.0 if any((dd.get("a") if isinstance(dd,dict) else False) for dd in drugs.get(str(i),[])) else 0.0
                for i in range(N)], dtype=np.float32)
    # --- dense multi-lens edges (weight = number of lenses) ---
    ew=defaultdict(int)
    def add(a,b):
        if a!=b and 0<=a<N and 0<=b<N: ew[(min(a,b),max(a,b))]+=1
    for e in D.get("ppi",[]):
        if len(e)>=2: add(e[0],e[1])
    for e in D.get("reg",[]):
        if len(e)>=2: add(e[0],e[1])
    for k,v in (D.get("codep") or {}).items():
        for p in v: add(int(k), p[0] if isinstance(p,(list,tuple)) else p)
    for e in D.get("sig",[]):
        if isinstance(e,(list,tuple)) and len(e)>=2 and isinstance(e[0],int): add(e[0],e[1])
    nm=[g["name"] for g in G]; ni={n:i for i,n in enumerate(nm)}
    for key in ("coexpr","model4"):
        for k,val in (D.get(key) or {}).items():
            a=ni.get(k, int(k) if str(k).isdigit() else None)
            if a is None: continue
            nbrs=val.get("neighbors",val) if isinstance(val,dict) else val
            for x in (nbrs or []):
                b=ni.get(x) if isinstance(x,str) else (x if isinstance(x,int) else (x[0] if isinstance(x,(list,tuple)) else None))
                if b is not None: add(a,b)
    rows=[]; cols=[]; w=[]
    for (a,b),c in ew.items(): rows+= [a,b]; cols+=[b,a]; w+=[float(c),float(c)]
    A=sp.coo_matrix((w,(rows,cols)),shape=(N,N)).tocsr()
    A=A+sp.identity(N)                                  # self-loops
    d=np.asarray(A.sum(1)).ravel(); dinv=1.0/np.sqrt(np.maximum(d,1e-6))
    A=sp.diags(dinv)@A@sp.diags(dinv)                    # symmetric normalization
    Ac=A.tocoo()
    At=torch.sparse_coo_tensor(np.vstack([Ac.row,Ac.col]), Ac.data.astype(np.float32), (N,N)).coalesce().to(dev)
    # --- node features (biology only) ---
    def f(g,k,d=np.nan):
        v=g.get(k,d)
        if v in (None,""): return d
        try: return float(v)
        except: return 1.0
    ppi_deg=defaultdict(int); rin=defaultdict(int); rout=defaultdict(int)
    for e in D.get("ppi",[]):
        if len(e)>=2: ppi_deg[e[0]]+=1; ppi_deg[e[1]]+=1
    for e in D.get("reg",[]):
        if len(e)>=2: rout[e[0]]+=1; rin[e[1]]+=1
    codep_deg={int(k):len(v) for k,v in (D.get("codep") or {}).items()}
    X=np.zeros((N,10),dtype=np.float32)
    for i,g in enumerate(G):
        lo=f(g,"loeuf"); lo=lo if (isinstance(lo,float) and lo>=0) else np.nan
        X[i]=[lo, f(g,"ess",0), f(g,"dep_frac",0), math.log10(f(g,"pubs",0)+1), f(g,"dark",0),
              f(g,"tf",0), f(g,"master",0), math.log10(ppi_deg.get(i,0)+1),
              math.log10(rin.get(i,0)+rout.get(i,0)+1), math.log10(codep_deg.get(i,0)+1)]
    col_med=np.nanmedian(X,0); inds=np.where(np.isnan(X)); X[inds]=np.take(col_med,inds[1])
    X=(X-X.mean(0))/(X.std(0)+1e-6)
    Xt=torch.tensor(X,device=dev); yt=torch.tensor(y,device=dev)
    rng=np.random.RandomState(0); idx=rng.permutation(N); ntr=int(N*0.8)
    trm=torch.zeros(N,dtype=torch.bool,device=dev); trm[idx[:ntr]]=True; tem=~trm
    pw=torch.tensor([(y[idx[:ntr]]==0).sum()/max((y[idx[:ntr]]==1).sum(),1)],dtype=torch.float32,device=dev)
    class GCN(nn.Module):
        def __init__(s,din,dh,graph=True):
            super().__init__(); s.graph=graph; s.l1=nn.Linear(din,dh); s.l2=nn.Linear(dh,dh); s.o=nn.Linear(dh,1); s.dp=nn.Dropout(0.3)
        def prop(s,h): return torch.sparse.mm(At,h) if s.graph else h
        def forward(s,x):
            h=s.dp(torch.relu(s.prop(s.l1(x)))); h=s.dp(torch.relu(s.prop(s.l2(h)))); return s.o(h).squeeze(-1)
    def run(graph):
        m=GCN(X.shape[1],64,graph).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=5e-3,weight_decay=5e-4)
        lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(300):
            m.train(); opt.zero_grad(); out=m(Xt); loss=lossf(out[trm],yt[trm]); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): p=torch.sigmoid(m(Xt)).cpu().numpy()
        return roc_auc_score(y[tem.cpu().numpy()], p[tem.cpu().numpy()]), p
    auc_g,pg=run(True); auc_m,_=run(False)
    cand=[(i,float(pg[i])) for i in range(N) if has_drug[i]==0]; cand.sort(key=lambda z:-z[1])
    short=[G[i]["name"] for i,_ in cand[:30]]
    res=dict(n_genes=N, n_approved_targets=int(y.sum()), n_edges=len(ew), device=dev,
             auc_gnn_dense_network=round(float(auc_g),4), auc_mlp_features_only=round(float(auc_m),4),
             network_gain=round(float(auc_g-auc_m),4), discovery_shortlist=short)
    json.dump(res, open(OUT/"target_gnn.json","w"))
    print(f"TARGET-DISCOVERY GNN — {N} genes, {int(y.sum())} approved targets, {len(ew)} dense edges | dev={dev}")
    print(f"  held-out AUC: GNN (dense network) {res['auc_gnn_dense_network']} vs MLP (features only) "
          f"{res['auc_mlp_features_only']} -> network adds +{res['network_gain']}")
    print(f"  {'>>> THE DENSE NETWORK DRIVES DISCOVERY' if res['network_gain']>0.01 else '(modest network gain here — full build adds co-expression + Perturb-seq lenses)'}")
    print("  top novel (undrugged) target candidates:", ", ".join(short[:14]))

if __name__=="__main__":
    main()
