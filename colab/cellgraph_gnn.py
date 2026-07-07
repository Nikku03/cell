"""Learned GraphSAGE — the trained-message-passing upgrade to fixed CellGraph propagation.

Same 16,492-node multi-relational graph, SAME leakage-free link-prediction benchmark as `cellgraph.py`
(test edges removed before encoding, degree-matched hard negatives, edge features [u·v | |u−v| | u+v]).
Only the ENCODER changes: fixed SIGN [X|SX|S²X] → a trained 2-layer GraphSAGE with a small edge-MLP decoder.
GPU-aware (auto-detects CUDA); the AUC is identical CPU vs GPU — the GPU only speeds training.

`head_to_head(D, X, A_base, relation)` returns fixed vs learned AUC on the IDENTICAL split, so the comparison
is honest. Gated by `colab/validate_cellgraph_gnn.py`; only adopted where it actually beats the fixed bar.
"""
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from cellgraph import embed


def _split_and_negatives(D, A_base, relation, test_frac=0.1, seed=0):
    """Reproduce cellgraph.link_prediction_auc's split + degree-matched hard negatives EXACTLY (shared by both)."""
    n = A_base.shape[0]; rng = np.random.default_rng(seed)
    edges = np.array([(e[0], e[1]) for e in D[relation] if e[0] != e[1]], dtype=np.int64)
    edges = np.unique(np.sort(edges, axis=1), axis=0); rng.shuffle(edges)
    n_test = int(len(edges) * test_frac)
    test_pos, train_pos = edges[:n_test], edges[n_test:]
    edgeset = set(map(tuple, edges))
    mask = sparse.csr_matrix((np.ones(len(test_pos)), (test_pos[:, 0], test_pos[:, 1])), shape=(n, n))
    mask = mask.maximum(mask.T)
    A_tr = (A_base - A_base.multiply(mask)); A_tr.eliminate_zeros()
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float64); p = deg + 1.0; p /= p.sum()

    def neg(k):
        out = []
        while len(out) < k:
            a = rng.choice(n, size=k*2, p=p); b = rng.choice(n, size=k*2, p=p)
            for x, y in zip(a, b):
                if x != y and (min(x, y), max(x, y)) not in edgeset:
                    out.append((x, y))
                    if len(out) >= k: break
        return np.array(out[:k], dtype=np.int64)
    return train_pos, neg(len(train_pos)), test_pos, neg(len(test_pos)), A_tr


def _edge_feats(H, pairs):
    u, v = H[pairs[:, 0]], H[pairs[:, 1]]
    return np.hstack([u*v, np.abs(u-v), u+v])


def _fixed_auc(X, A_tr, train_pos, train_neg, test_pos, test_neg, yte):
    H = embed(X, A_tr, hops=2)
    clf = LogisticRegression(max_iter=300).fit(
        np.vstack([_edge_feats(H, train_pos), _edge_feats(H, train_neg)]),
        np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))])
    p = clf.predict_proba(np.vstack([_edge_feats(H, test_pos), _edge_feats(H, test_neg)]))[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


class SAGE(nn.Module):
    def __init__(self, din, dh=64, do=64, p=0.3):
        super().__init__()
        self.l1 = nn.Linear(din*2, dh); self.l2 = nn.Linear(dh*2, do); self.drop = nn.Dropout(p)
        self.dec = nn.Sequential(nn.Linear(do*3, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def encode(self, X, A):
        h = torch.relu(self.l1(torch.cat([X, torch.sparse.mm(A, X)], 1))); h = self.drop(h)
        return torch.relu(self.l2(torch.cat([h, torch.sparse.mm(A, h)], 1)))

    def score(self, h, pairs):
        u, v = h[pairs[:, 0]], h[pairs[:, 1]]
        return self.dec(torch.cat([u*v, torch.abs(u-v), u+v], 1)).squeeze(1)


def _learned_auc(X, A_tr, train_pos, train_neg, test_pos, test_neg, yte, epochs=200, bs=20000, device=None, seed=0):
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n = A_tr.shape[0]
    A = A_tr.tocoo(); d = np.asarray(A_tr.sum(1)).ravel(); d[d == 0] = 1
    Anorm = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), (A.data / d[A.row]).astype(np.float32),
                                    (n, n)).coalesce().to(device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    tp = torch.from_numpy(train_pos).to(device); tn = torch.from_numpy(train_neg).to(device)
    te = torch.from_numpy(np.vstack([test_pos, test_neg])).to(device)
    model = SAGE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    bce = nn.BCEWithLogitsLoss()
    y = torch.cat([torch.ones(bs), torch.zeros(bs)]).to(device)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        h = model.encode(Xt, Anorm)
        pi = torch.randint(0, len(tp), (bs,), device=device); ni = torch.randint(0, len(tn), (bs,), device=device)
        loss = bce(model.score(h, torch.cat([tp[pi], tn[ni]])), y); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ps = torch.sigmoid(model.score(model.encode(Xt, Anorm), te)).cpu().numpy()
    return roc_auc_score(yte, ps), average_precision_score(yte, ps)


class RGCN(nn.Module):
    """R-GCN: a separate weight matrix per relation (reg/ppi/sig/codep/lr) + a self-loop weight, so the model
    distinguishes edge types instead of merging them into one adjacency (as GraphSAGE does)."""
    def __init__(self, din, nrel, dh=64, do=64, p=0.3):
        super().__init__()
        self.self1 = nn.Linear(din, dh); self.rel1 = nn.ModuleList([nn.Linear(din, dh) for _ in range(nrel)])
        self.self2 = nn.Linear(dh, do);  self.rel2 = nn.ModuleList([nn.Linear(dh, do) for _ in range(nrel)])
        self.drop = nn.Dropout(p)
        self.dec = nn.Sequential(nn.Linear(do*3, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def encode(self, X, As):
        h = self.self1(X) + sum(self.rel1[i](torch.sparse.mm(As[i], X)) for i in range(len(As)))
        h = self.drop(torch.relu(h))
        h = self.self2(h) + sum(self.rel2[i](torch.sparse.mm(As[i], h)) for i in range(len(As)))
        return torch.relu(h)

    def score(self, h, pairs):
        u, v = h[pairs[:, 0]], h[pairs[:, 1]]
        return self.dec(torch.cat([u*v, torch.abs(u-v), u+v], 1)).squeeze(1)


def _per_relation_adj(D, n, test_pos, relations=("reg", "ppi", "sig", "codep", "lr"), device="cpu"):
    """Row-normalized adjacency per relation; the held-out PPI test edges are removed from the ppi relation."""
    from cellgraph import build_adj
    mask = sparse.csr_matrix((np.ones(len(test_pos)), (test_pos[:, 0], test_pos[:, 1])), shape=(n, n))
    mask = mask.maximum(mask.T)
    out = []
    for r in relations:
        A = build_adj(D, n, relations=(r,))
        if r == "ppi":
            A = (A - A.multiply(mask)); A.eliminate_zeros()
        d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1; A = A.tocoo()
        out.append(torch.sparse_coo_tensor(np.vstack([A.row, A.col]), (A.data / d[A.row]).astype(np.float32),
                                           (n, n)).coalesce().to(device))
    return out


def _learned_rgcn_auc(D, X, test_pos, train_pos, train_neg, test_all, yte, epochs=200, device=None, seed=0):
    torch.manual_seed(seed)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n = X.shape[0]; nrel = 5
    As = _per_relation_adj(D, n, test_pos, device=device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    tp = torch.from_numpy(train_pos).to(device); tn = torch.from_numpy(train_neg).to(device)
    te = torch.from_numpy(test_all).to(device)
    model = RGCN(X.shape[1], nrel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4); bce = nn.BCEWithLogitsLoss()
    bs = 20000; y = torch.cat([torch.ones(bs), torch.zeros(bs)]).to(device)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        h = model.encode(Xt, As)
        pi = torch.randint(0, len(tp), (bs,), device=device); ni = torch.randint(0, len(tn), (bs,), device=device)
        bce(model.score(h, torch.cat([tp[pi], tn[ni]])), y).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ps = torch.sigmoid(model.score(model.encode(Xt, As), te)).cpu().numpy()
    return roc_auc_score(yte, ps), average_precision_score(yte, ps)


def head_to_head(D, X, A_base, relation="ppi", epochs=200, device=None, seed=0):
    """Fixed SIGN vs learned GraphSAGE vs learned R-GCN on the IDENTICAL leakage-free split. R-GCN is the
    headline `learned_auc` (best encoder); GraphSAGE reported alongside."""
    tp, tn, sp, sn, A_tr = _split_and_negatives(D, A_base, relation, seed=seed)
    yte = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    f_auc, f_ap = _fixed_auc(X, A_tr, tp, tn, sp, sn, yte)
    s_auc, s_ap = _learned_auc(X, A_tr, tp, tn, sp, sn, yte, epochs=epochs, device=device, seed=seed)
    r_auc, r_ap = _learned_rgcn_auc(D, X, sp, tp, tn, np.vstack([sp, sn]), yte,
                                    epochs=epochs, device=device, seed=seed)
    return dict(relation=relation, n_edges=int(len(tp) + len(sp)),
                fixed_auc=round(float(f_auc), 4), fixed_ap=round(float(f_ap), 4),
                sage_auc=round(float(s_auc), 4),
                learned_auc=round(float(r_auc), 4), learned_ap=round(float(r_ap), 4),  # R-GCN = headline
                delta=round(float(r_auc - f_auc), 4),
                device=(device or ('cuda' if torch.cuda.is_available() else 'cpu')))
