"""TRANSFORMER Tower A (production) — heterogeneous GraphSAGE link-prediction on the KG. Colab/GPU.

The real gene tower: a 2-layer GraphSAGE over the KG (kg_edges.tsv + kg_nodes.npz) with node-feature
init, trained self-supervised on link prediction (the GEARS lesson). Produces gene embeddings g_i for
ALL genes — the Tower-A output the perturbation head (Tower C) consumes. The sandbox prototype
(train_tower_a.py, SVD) validated the concept at AUC 0.94-0.95; this is the trainable, generalizing
version. NOT sandbox-tested (needs torch/GPU) — standard GraphSAGE pattern.
-> tower_a_embeddings.npz
"""
import numpy as np
from pathlib import Path
OUT=Path("outputs/orphan")

def main():
    import torch, torch.nn as nn, torch.nn.functional as Fn
    dev="cuda" if torch.cuda.is_available() else "cpu"
    z=np.load(OUT/"kg_nodes.npz"); F=torch.tensor(z["features"],dtype=torch.float32); N=F.shape[0]
    # symmetric edge index (all types pooled; a typed R-GCN is the next upgrade)
    E=[]
    for l in open(OUT/"kg_edges.tsv"):
        a,b,t,w,s=l.rstrip("\n").split("\t"); E.append((int(a),int(b))); E.append((int(b),int(a)))
    ei=torch.tensor(E,dtype=torch.long).t().contiguous()
    # normalized sparse adjacency for mean-aggregation message passing
    deg=torch.zeros(N).index_add_(0,ei[0],torch.ones(ei.shape[1]))
    norm=1.0/deg.clamp(min=1)
    A=torch.sparse_coo_tensor(ei, norm[ei[0]], (N,N)).coalesce()

    class SAGE(nn.Module):
        def __init__(s,din,d=128):
            super().__init__(); s.l1=nn.Linear(din*2,d); s.l2=nn.Linear(d*2,d)
        def forward(s,x,A):
            h=Fn.relu(s.l1(torch.cat([x,torch.sparse.mm(A,x)],1)))
            h=s.l2(torch.cat([h,torch.sparse.mm(A,h)],1))
            return Fn.normalize(h,dim=1)
    # link-prediction split
    pos=ei[:, ei[0]<ei[1]].t()
    perm=torch.randperm(pos.shape[0]); cut=int(0.9*pos.shape[0])
    tr,te=pos[perm[:cut]],pos[perm[cut:]]
    A=A.to(dev); F=F.to(dev); model=SAGE(F.shape[1]).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    def score(h,e): return (h[e[:,0]]*h[e[:,1]]).sum(1)
    for ep in range(1,201):
        model.train(); opt.zero_grad(); h=model(F,A)
        neg=torch.randint(0,N,(tr.shape[0],2),device=dev)
        loss=Fn.binary_cross_entropy_with_logits(
            torch.cat([score(h,tr.to(dev)),score(h,neg)]),
            torch.cat([torch.ones(tr.shape[0]),torch.zeros(tr.shape[0])]).to(dev))
        loss.backward(); opt.step()
        if ep%50==0:
            model.eval()
            with torch.no_grad():
                h=model(F,A); negt=torch.randint(0,N,(te.shape[0],2),device=dev)
                from sklearn.metrics import roc_auc_score
                s=torch.cat([score(h,te.to(dev)),score(h,negt)]).cpu().numpy()
                y=np.r_[np.ones(te.shape[0]),np.zeros(te.shape[0])]
                print(f"  epoch {ep}: loss {loss.item():.3f} | test link-AUC {roc_auc_score(y,s):.3f}")
    model.eval()
    with torch.no_grad(): emb=model(F,A).cpu().numpy()
    np.savez(OUT/"tower_a_embeddings.npz", emb=emb, names=z["names"])
    print(f"Tower A: gene embeddings {emb.shape} -> tower_a_embeddings.npz (feeds Tower C perturbation head)")

if __name__=="__main__":
    main()
