"""CellGraph — a learned representation of the whole-cell knowledge graph.

The cell model is a multi-relational graph: 16,492 gene/protein NODES with features, connected by typed
EDGES (reg = signed directed regulation, ppi = physical interaction, sig = signed signaling, codep =
co-dependency, lr = ligand-receptor, complex co-membership, metabolic). This module learns a vector
embedding per gene from that graph + node features, and exposes it for the mechanistic queries the project
needs (link prediction / "what can this protein bind", perturbation propagation, function prediction, ...).

Architecture (CPU-friendly, no GPU / no torch needed for a graph this size): SIGN / SGC-style propagation —
precompute multi-hop smoothed features H = [X | S X | S^2 X] over the symmetric-normalized adjacency S,
then train light task heads (logistic regression) on them. This is a genuine graph neural network with
fixed (non-learned) propagation; for 16k nodes it is fast, strong, and fully reproducible.

Round 1: features + adjacency + embeddings + held-out link-prediction AUC (the foundation + first metric).
"""
import json
import numpy as np
from pathlib import Path
from scipy import sparse

OUT = Path("outputs/orphan")
CC = OUT / "cell_complete.json"

NUM_FIELDS = ["loeuf", "tf", "ppi", "ndis", "npath", "cpg", "enh", "dep_frac", "dark", "pubs", "ess"]


def load_model():
    D = json.load(open(CC))
    G = D["genes"]
    name = [g["name"] for g in G]
    idx = {n: i for i, n in enumerate(name)}
    return D, G, name, idx


def node_features(G):
    """numeric (standardized) + one-hot(compartment) + one-hot(process) -> dense feature matrix X."""
    n = len(G)
    num = np.zeros((n, len(NUM_FIELDS)), dtype=np.float32)
    for i, g in enumerate(G):
        for j, f in enumerate(NUM_FIELDS):
            v = g.get(f)
            num[i, j] = float(v) if isinstance(v, (int, float)) else 0.0
    # log1p the heavy-tailed counts
    for j, f in enumerate(NUM_FIELDS):
        if f in ("ppi", "npath", "ndis", "enh", "pubs"):
            num[:, j] = np.log1p(np.clip(num[:, j], 0, None))
    num = (num - num.mean(0)) / (num.std(0) + 1e-6)
    comps = sorted({g.get("comp", "") for g in G})
    procs = sorted({g.get("proc", "") for g in G})
    ci = {c: k for k, c in enumerate(comps)}; pi = {p: k for k, p in enumerate(procs)}
    oc = np.zeros((n, len(comps)), np.float32); op = np.zeros((n, len(procs)), np.float32)
    for i, g in enumerate(G):
        oc[i, ci[g.get("comp", "")]] = 1.0; op[i, pi[g.get("proc", "")]] = 1.0
    X = np.hstack([num, oc, op]).astype(np.float32)
    return X


def build_adj(D, n, relations=("reg", "ppi", "sig", "codep", "lr")):
    """symmetric weighted adjacency over the chosen relations (undirected, for representation learning)."""
    rows, cols, vals = [], [], []
    def add(a, b, w=1.0):
        rows.append(a); cols.append(b); vals.append(w)
    for e in D.get("reg", []):
        if "reg" in relations: add(e[0], e[1], 1.0)
    for e in D.get("ppi", []):
        if "ppi" in relations: add(e[0], e[1], 1.0)
    for e in D.get("sig", []):
        if "sig" in relations: add(e[0], e[1], 1.0)
    if "codep" in relations:
        for k, lst in (D.get("codep", {}) or {}).items():
            a = int(k)
            for it in lst:
                b = it[0] if isinstance(it, (list, tuple)) else it
                if isinstance(b, int): add(a, b, 1.0)
    for e in D.get("lr", []):
        if "lr" in relations: add(e[0], e[1], 1.0)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    A = A.maximum(A.T)                                    # symmetrize
    A.setdiag(0); A.eliminate_zeros()
    return A


def normalize_adj(A):
    """S = D^-1/2 (A + I) D^-1/2  (GCN-style)."""
    n = A.shape[0]
    A = A + sparse.identity(n, format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    Dm = sparse.diags(dinv)
    return Dm @ A @ Dm


def embed(X, A, hops=2):
    """SIGN embedding: concat features + multi-hop smoothed features. H = [X | SX | S^2X | ...]."""
    S = normalize_adj(A)
    feats = [X]
    cur = X
    for _ in range(hops):
        cur = S @ cur
        feats.append(cur)
    H = np.hstack(feats).astype(np.float32)
    # L2-normalize rows so dot-products behave
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return H


# ---------------- Round 1 task: link prediction ----------------
def link_prediction_auc(D, H, relation="ppi", test_frac=0.1, seed=0):
    """Hold out `test_frac` of `relation` edges; train logistic regression on Hadamard(H[u],H[v]) to
    distinguish true edges from random non-edges. Report ROC-AUC and average precision on the held-out set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    edges = np.array([(e[0], e[1]) for e in D[relation] if e[0] != e[1]], dtype=np.int64)
    edges = np.unique(np.sort(edges, axis=1), axis=0)     # dedupe undirected
    rng.shuffle(edges)
    n_test = int(len(edges) * test_frac)
    test_pos, train_pos = edges[:n_test], edges[n_test:]
    edgeset = set(map(tuple, edges))
    def neg_sample(k):
        out = []
        while len(out) < k:
            a = rng.integers(0, n, size=k); b = rng.integers(0, n, size=k)
            for x, y in zip(a, b):
                if x != y and (min(x, y), max(x, y)) not in edgeset:
                    out.append((x, y))
                    if len(out) >= k: break
        return np.array(out[:k], dtype=np.int64)
    train_neg = neg_sample(len(train_pos)); test_neg = neg_sample(len(test_pos))
    def feats(pairs): return H[pairs[:, 0]] * H[pairs[:, 1]]        # Hadamard product
    Xtr = np.vstack([feats(train_pos), feats(train_neg)])
    ytr = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    Xte = np.vstack([feats(test_pos), feats(test_neg)])
    yte = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return dict(relation=relation, n_edges=len(edges), auc=round(float(roc_auc_score(yte, p)), 4),
                ap=round(float(average_precision_score(yte, p)), 4), dim=H.shape[1])


def main():
    D, G, name, idx = load_model()
    n = len(G)
    print(f"CellGraph — {n} nodes")
    X = node_features(G)
    print(f"  node features: {X.shape[1]} dims")
    A = build_adj(D, n)
    print(f"  adjacency: {A.nnz//2} undirected edges over reg+ppi+sig+codep+lr")
    H = embed(X, A, hops=2)
    print(f"  embedding: {H.shape[1]} dims (SIGN, 2 hops)")
    res = {}
    for rel in ["ppi", "reg", "sig"]:
        r = link_prediction_auc(D, H, relation=rel)
        res[rel] = r
        print(f"  link-prediction[{rel}]: AUC {r['auc']}  AP {r['ap']}  ({r['n_edges']} edges)")
    json.dump(dict(n_nodes=n, feat_dim=int(X.shape[1]), emb_dim=int(H.shape[1]),
                   link_prediction=res), open(OUT / "cellgraph_r1.json", "w"), indent=2)
    print("-> outputs/orphan/cellgraph_r1.json")
    return res


if __name__ == "__main__":
    main()
