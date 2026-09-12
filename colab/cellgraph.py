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
import os
import numpy as np
from pathlib import Path
from scipy import sparse

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
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
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    return H


def structural_embedding(A, dim=64, hops=3, seed=0):
    """LEAKAGE-FREE embedding from pure TOPOLOGY: propagate random Gaussian features through the graph.
    Contains no node labels, so predicting compartment/process/essentiality from it honestly tests whether
    the cell's WIRING encodes function."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, dim)).astype(np.float32)
    return embed(R, A, hops=hops)


# ---------------- Round 1/2 task: link prediction (hard negatives + NO test-edge leakage) ----------------
def link_prediction_auc(D, X, A_base, relation="ppi", test_frac=0.1, seed=0, hard_negatives=True):
    """Honest link prediction: hold out test edges of `relation`, REMOVE them from the graph, rebuild the
    embedding, then predict them. Degree-matched hard negatives. This is leakage-free (the model never sees
    the test edges during representation learning)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(seed)
    n = A_base.shape[0]
    edges = np.array([(e[0], e[1]) for e in D[relation] if e[0] != e[1]], dtype=np.int64)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rng.shuffle(edges)
    n_test = int(len(edges) * test_frac)
    test_pos, train_pos = edges[:n_test], edges[n_test:]
    edgeset = set(map(tuple, edges))
    # rebuild adjacency with the TEST edges removed, then embed -> no leakage
    mask = sparse.csr_matrix((np.ones(len(test_pos)), (test_pos[:, 0], test_pos[:, 1])), shape=(n, n))
    mask = mask.maximum(mask.T)
    A_tr = (A_base - A_base.multiply(mask)); A_tr.eliminate_zeros()
    H = embed(X, A_tr, hops=2)
    deg = np.bincount(edges.ravel(), minlength=n).astype(np.float64)
    p_deg = (deg + 1.0); p_deg /= p_deg.sum()             # degree-proportional sampling distribution
    def neg_sample(k):
        out = []
        while len(out) < k:
            if hard_negatives:
                a = rng.choice(n, size=k * 2, p=p_deg); b = rng.choice(n, size=k * 2, p=p_deg)
            else:
                a = rng.integers(0, n, size=k * 2); b = rng.integers(0, n, size=k * 2)
            for x, y in zip(a, b):
                if x != y and (min(x, y), max(x, y)) not in edgeset:
                    out.append((x, y))
                    if len(out) >= k: break
        return np.array(out[:k], dtype=np.int64)
    train_neg = neg_sample(len(train_pos)); test_neg = neg_sample(len(test_pos))
    def feats(pairs):                                     # richer edge features (R10): Hadamard | |diff| | sum
        u, v = H[pairs[:, 0]], H[pairs[:, 1]]
        return np.hstack([u * v, np.abs(u - v), u + v])
    Xtr = np.vstack([feats(train_pos), feats(train_neg)])
    ytr = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    Xte = np.vstack([feats(test_pos), feats(test_neg)])
    yte = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return dict(relation=relation, n_edges=len(edges),
                auc=round(float(roc_auc_score(yte, p)), 4),
                ap=round(float(average_precision_score(yte, p)), 4),
                negatives="degree-matched" if hard_negatives else "random", dim=H.shape[1])


# ---------------- Round 2 task: node function prediction (does the embedding encode biology?) ----------------
def node_function_eval(H, G, seed=0):
    """Predict biological node labels from a STRUCTURE-ONLY embedding (pass H = structural_embedding(A), which
    contains no node labels). 80/20 split vs a majority/prior baseline. Tests whether the cell's WIRING alone
    encodes compartment, process, essentiality and TF identity -- leakage-free."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    n = len(G); rng = np.random.default_rng(seed)
    out = {}
    # multiclass: compartment, process
    for field in ("comp", "proc"):
        labs = [g.get(field, "") for g in G]
        classes = sorted(set(labs)); ci = {c: k for k, c in enumerate(classes)}
        y = np.array([ci[l] for l in labs])
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(H[tr], y[tr])   # multinomial is the default
        acc = accuracy_score(y[te], clf.predict(H[te]))
        base = max(np.bincount(y).max() / n, 1e-9)
        out[field] = dict(accuracy=round(float(acc), 3), majority_baseline=round(float(base), 3),
                          n_classes=len(classes))
    # binary: essentiality, tf
    for field in ("ess", "tf"):
        y = np.array([1 if g.get(field) else 0 for g in G])
        if y.sum() < 20 or (n - y.sum()) < 20:
            continue
        tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced").fit(H[tr], y[tr])
        p = clf.predict_proba(H[te])[:, 1]
        out[field] = dict(auc=round(float(roc_auc_score(y[te], p)), 3), positive_rate=round(float(y.mean()), 3))
    return out


# ---------------- Round 3: perturbation -> downstream effect predictor ----------------
def directed_signed(D, n):
    """directed signed out-adjacency from reg + sig (unknown sign -> +1). For perturbation propagation."""
    rows, cols, vals = [], [], []
    for e in D.get("reg", []) + D.get("sig", []):
        s = 1.0 if (len(e) < 3 or e[2] >= 0) else -1.0
        rows.append(e[0]); cols.append(e[1]); vals.append(s)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def perturb_downstream(seed_idx, W, alpha=0.5, hops=3):
    """Signed propagation of a knockout from `seed_idx` through the directed signed network W (out-adjacency).
    Returns a vector of net signed downstream effect on every gene. |effect| ranks the affected genes;
    sign gives up/down. This answers: remove protein X -> what changes, and which way."""
    n = W.shape[0]
    v = np.zeros(n, np.float32)
    cur = np.zeros(n, np.float32)
    for s in np.atleast_1d(seed_idx):
        cur[s] = -1.0                                     # knockout = remove the gene's activity
    Wt = W.T.tocsr()                                      # to propagate along out-edges via matrix-vector
    for _ in range(hops):
        cur = alpha * (W.T @ cur)                         # effect flows source->target
        v = v + cur
    return v


def validate_perturbation(D, n, W, seed=0, max_sources=400):
    """Does knockout propagation predict DepMap CO-DEPENDENCY (an independent signal)? For each source gene
    with >=3 codep partners, rank all genes by |downstream effect| and score recovery of its codep partners
    (ROC-AUC) vs a degree-matched random baseline."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    codep = D.get("codep", {}) or {}
    srcs = []
    for k, lst in codep.items():
        partners = [it[0] if isinstance(it, (list, tuple)) else it for it in lst]
        partners = [p for p in partners if isinstance(p, int) and p != int(k)]
        if len(partners) >= 3:
            srcs.append((int(k), set(partners)))
    rng.shuffle(srcs); srcs = srcs[:max_sources]
    aucs, rand = [], []
    for s, partners in srcs:
        eff = np.abs(perturb_downstream(s, W))
        eff[s] = 0.0
        y = np.zeros(n, np.int8);
        for p in partners:
            if p < n: y[p] = 1
        if y.sum() < 3 or eff.max() == 0:
            continue
        aucs.append(roc_auc_score(y, eff))
        rand.append(roc_auc_score(y, rng.random(n)))
    return dict(n_sources=len(aucs), auc=round(float(np.mean(aucs)), 4) if aucs else None,
                random_auc=round(float(np.mean(rand)), 4) if rand else None)


def validate_perturbation_direction(D, n, seed=0, test_frac=0.15):
    """Honest directional test: hold out 15% of KNOWN-SIGN reg/sig edges, rebuild the network WITHOUT them,
    propagate each source's knockout, and check whether the predicted downstream SIGN on the held-out target
    matches biology (KO of an activator -> target DOWN; KO of a repressor -> target UP). Accuracy vs 0.5."""
    rng = np.random.default_rng(seed)
    signed = [(e[0], e[1], (1 if e[2] > 0 else -1)) for e in D.get("reg", []) + D.get("sig", [])
              if len(e) > 2 and e[2] != 0]
    signed = list({(a, b, s) for a, b, s in signed})
    rng.shuffle(signed)
    n_test = int(len(signed) * test_frac)
    test = signed[:n_test]; train = signed[n_test:]
    rows = [a for a, b, s in train]; cols = [b for a, b, s in train]; vals = [float(s) for a, b, s in train]
    # add the unsigned edges (sign 0 -> +1) so the network stays connected
    for e in D.get("reg", []) + D.get("sig", []):
        if len(e) < 3 or e[2] == 0:
            rows.append(e[0]); cols.append(e[1]); vals.append(1.0)
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    by_src = {}
    for a, b, s in test:
        by_src.setdefault(a, []).append((b, s))
    correct = tot = 0
    for src, tgts in by_src.items():
        eff = perturb_downstream(src, W)
        for b, s in tgts:
            pred = eff[b]
            if pred == 0:
                continue
            if np.sign(pred) == -s:                        # KO of activator(+) -> down(-); repressor(-) -> up(+)
                correct += 1
            tot += 1
    return dict(n_test_edges=tot, sign_accuracy=round(correct / tot, 4) if tot else None, baseline=0.5)


# ---------------- Round 4: drug polypharmacology — "what else can this drug hit" ----------------
def drug_polypharmacology_eval(D, H, seed=0, min_t=3, max_t=40):
    """For each drug with several protein targets, hold out ONE target and predict it by ranking all genes by
    embedding similarity to the drug's OTHER targets. Tests whether the cell embedding captures shared drug
    targeting (off-target / polypharmacology prediction). ROC-AUC of recovering the held-out target."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    drug2genes = {}
    for gi, lst in (D.get("drugs", {}) or {}).items():
        for d in lst:
            nm = d.get("d")
            if nm and nm != "Null":
                drug2genes.setdefault(nm, set()).add(int(gi))
    aucs, rnd = [], []
    for drug, genes in drug2genes.items():
        genes = list(genes)
        if not (min_t <= len(genes) <= max_t):
            continue
        held = genes[rng.integers(len(genes))]
        rest = [g for g in genes if g != held]
        centroid = H[rest].mean(0)
        sim = H @ centroid                                # cosine-ish (H is L2-normalized)
        for g in rest:
            sim[g] = -np.inf                              # exclude the known targets from ranking
        y = np.zeros(n, np.int8); y[held] = 1
        mask = np.ones(n, bool); mask[rest] = False
        aucs.append(roc_auc_score(y[mask], sim[mask]))
        rnd.append(0.5)
    return dict(n_drugs=len(aucs), auc=round(float(np.mean(aucs)), 4) if aucs else None,
                random=0.5, note="held-out drug-target recovery from embedding similarity")


# ---------------- Round 5: unified queryable model ----------------
class CellGraph:
    """One object that answers the mechanistic questions, built on the validated capabilities:
      .bind_partners(gene)   — what proteins can this bind? (link prediction, R1)
      .knockout_effect(gene) — remove this protein -> what changes downstream & which way (R3, 81% direction)
      .drug_off_targets(drug)— give a drug -> what else can it hit (polypharmacology, R4)
    """
    def __init__(self):
        self.D, self.G, self.name, self.idx = load_model()
        self.n = len(self.G)
        X = node_features(self.G)
        self.A = build_adj(self.D, self.n)
        self.H = embed(X, self.A, hops=2)                 # feature+structure embedding (binding, drugs)
        self.W = directed_signed(self.D, self.n)          # directed signed (perturbation)
        self.indeg = np.sqrt(np.asarray(np.abs(self.W).sum(0)).ravel() + 1.0)  # for specificity weighting
        self._partners = None

    def _known_partners(self):
        if self._partners is None:
            p = {i: set() for i in range(self.n)}
            for e in self.D["ppi"]:
                p[e[0]].add(e[1]); p[e[1]].add(e[0])
            self._partners = p
        return self._partners

    def bind_partners(self, gene, k=10):
        if gene not in self.idx: return []
        i = self.idx[gene]; sim = self.H @ self.H[i]
        known = self._known_partners()[i]
        order = np.argsort(-sim)
        out = [self.name[j] for j in order if j != i and j not in known][:k]
        return out

    def knockout_effect(self, gene, k=12):
        """Remove protein -> ranked downstream genes (up/down). Specificity-weighted (÷√in-degree) so hubs
        don't dominate. Effectors with no transcriptional out-edges fall back to their physical (PPI) partners
        — they act locally, not by transcription."""
        if gene not in self.idx: return []
        i = self.idx[gene]
        eff = perturb_downstream(i, self.W, hops=1)         # DIRECT downstream — multi-hop floods hubs
        if np.abs(eff).sum() == 0:                          # no transcriptional downstream (e.g. secreted effector)
            partners = sorted(self._known_partners()[i])
            return [(self.name[j], "local", None) for j in partners[:k]] or \
                   [("(no downstream — acts post-transcriptionally)", "", None)]
        order = np.argsort(-np.abs(eff))
        return [(self.name[j], "down" if eff[j] < 0 else "up", round(float(eff[j]), 2))
                for j in order if j != i][:k]

    def drug_off_targets(self, drug, k=10):
        genes = [int(gi) for gi, lst in self.D["drugs"].items()
                 for d in lst if d.get("d") == drug]
        if not genes: return []
        centroid = self.H[genes].mean(0); sim = self.H @ centroid
        gs = set(genes); order = np.argsort(-sim)
        return [self.name[j] for j in order if j not in gs][:k]


def demo():
    cg = CellGraph()
    print("\n" + "=" * 74 + "\nCellGraph — live query demo (the model answering the questions)\n" + "=" * 74)
    print("Q: what can TP53 bind? ->", cg.bind_partners("TP53", 8))
    print("Q: remove SREBF2 -> downstream? ->", cg.knockout_effect("SREBF2", 8))
    print("Q: remove PCSK9 -> downstream? ->", cg.knockout_effect("PCSK9", 6))
    print("Q: give Imatinib -> what else can it hit? ->", cg.drug_off_targets("Imatinib", 8))


def main():
    D, G, name, idx = load_model()
    n = len(G)
    print(f"CellGraph — {n} nodes")
    X = node_features(G)
    print(f"  node features: {X.shape[1]} dims")
    A = build_adj(D, n)
    print(f"  adjacency: {A.nnz//2} undirected edges over reg+ppi+sig+codep+lr")
    res = {}
    print("  [R1/R2] link prediction (hard negatives + test-edge removed = leakage-free):")
    for rel in ["ppi", "reg", "sig"]:
        r = link_prediction_auc(D, X, A, relation=rel, hard_negatives=True)
        res[rel] = r
        print(f"     {rel:4} AUC {r['auc']}  AP {r['ap']}  ({r['n_edges']} edges, {r['negatives']} neg)")
    print("  [R2] node-function from STRUCTURE ONLY (leakage-free: does wiring predict function?):")
    Hs = structural_embedding(A, dim=64, hops=3)
    fn = node_function_eval(Hs, G)
    for k, v in fn.items():
        if "accuracy" in v:
            print(f"     {k:5} acc {v['accuracy']} (majority {v['majority_baseline']}, {v['n_classes']} classes)")
        else:
            print(f"     {k:5} AUC {v['auc']} (positive rate {v['positive_rate']})")
    print("  [R4] drug polypharmacology — recover a held-out drug target from the embedding:")
    Hf = embed(X, A, hops=2)
    drug = drug_polypharmacology_eval(D, Hf)
    print(f"     off-target/polypharmacology AUC {drug['auc']} (random {drug['random']}, {drug['n_drugs']} drugs)")
    print("  [R3] perturbation -> downstream effect:")
    W = directed_signed(D, n)
    pdir = validate_perturbation_direction(D, n)
    print(f"     downstream DIRECTION (held-out signed edges): sign-accuracy {pdir['sign_accuracy']} "
          f"(baseline {pdir['baseline']}, {pdir['n_test_edges']} held-out targets)")
    pert = validate_perturbation(D, n, W)
    print(f"     (co-dependency recovery — different signal — AUC {pert['auc']} vs random {pert['random_auc']})")
    json.dump(dict(n_nodes=n, feat_dim=int(X.shape[1]), struct_emb_dim=int(Hs.shape[1]),
                   link_prediction=res, node_function=fn,
                   perturbation=dict(direction=pdir, codependency=pert), drug_polypharmacology=drug),
              open(OUT / "cellgraph_metrics.json", "w"), indent=2)
    print("-> outputs/orphan/cellgraph_metrics.json")
    return dict(link_prediction=res, node_function=fn)


if __name__ == "__main__":
    main()
