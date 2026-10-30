"""THE UNIFIED CELL GRAPH: stoichiometry and interaction in one typed object.

WHY THIS EXISTS. This repository holds two kinds of structure that have never been in the same
object. One is CHEMICAL: 12,931 Human-GEM reactions over 8,461 metabolites, properly chained --
73.0% of metabolites are both produced and consumed, there are zero orphans, and 96.2% of reactions
sit in one connected component once currency hubs are removed. The other is RELATIONAL: 191,447 PPI
edges, 17,432 signalling edges, 55,716 curated regulatory edges, 2,039 complexes. The first knows
what turns into what; the second knows what touches what. Neither knows the other.

They join at exactly one place: 2,568 of the model's 16,492 genes carry a Human-GEM reaction. That
join is narrow -- 15.6% of genes -- and it is the only bridge that exists, so any fused
representation lives or dies on it.

WHAT THIS BUILDS. A typed graph with three node kinds and six edge kinds:

    gene        16,492      the model's genes
    reaction    12,931      Human-GEM reactions
    metabolite   8,461      Human-GEM metabolites

    gene -> reaction        catalyses, from the GEM gene-protein-reaction rules
    reaction -> metabolite  consumes (negative stoichiometry) or produces (positive)
    gene -- gene            ppi, signalling, regulation, co-complex

Edges are returned as a sparse matrix plus a type vector, so a caller can weight or drop any
channel. Nothing here embeds or learns; that is the caller's job and the point of keeping it
separate is that the same intermediate can be handed to a linear probe, a GNN or anything else and
the comparison stays honest.

WHAT IT DELIBERATELY DOES NOT DO. It does not collapse a protein and its phospho-form into
different nodes -- they remain one node, which is exactly the limitation measured earlier: 9,809
signalling edges are topologically chained but chemically inert for that reason. It does not
compartment-resolve genes. Both are known holes and are recorded rather than papered over, because
a fused graph that hides them would make the fusion look better than it is.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loop_replication as LR  # noqa: E402

EDGE_TYPES = ("catalyses", "consumes", "produces", "ppi", "signal", "regulate", "complex")
CURATED = 55716

_CACHE = {}


def build(include_gem=True, gem_model=None):
    """Return the typed graph. Cached, because loading Human-GEM takes about half a minute."""
    key = ("g", bool(include_gem))
    if key in _CACHE:
        return _CACHE[key]

    C = json.load(open(LR.CELL))
    names = [g["name"] for g in C["genes"]]
    gidx = {n: i for i, n in enumerate(names)}
    nG = len(names)

    rows, cols, etype = [], [], []

    def add(a, b, t):
        rows.append(a)
        cols.append(b)
        etype.append(EDGE_TYPES.index(t))

    # ---- gene-gene channels ------------------------------------------------------------------
    for a, b in C["ppi"]:
        if a < nG and b < nG:
            add(a, b, "ppi")
    for e in C["sig"]:
        if e[0] < nG and e[1] < nG:
            add(e[0], e[1], "signal")
    for e in C["reg"][:CURATED]:
        if e[0] < nG and e[1] < nG:
            add(e[0], e[1], "regulate")
    comps = C["complexes"]
    for _, mem in (comps.items() if isinstance(comps, dict) else comps):
        m = [int(x) for x in mem if int(x) < nG]
        for i, a in enumerate(m):
            for b in m[i + 1:]:
                add(a, b, "complex")

    rxn_names, met_names = [], []
    if include_gem:
        import cell_sim as CS
        m = gem_model if gem_model is not None else CS.load_model()
        sym = CS.ensembl_to_symbol()
        rxn_names = [r.id for r in m.reactions]
        met_names = [x.id for x in m.metabolites]
        ridx = {r: nG + i for i, r in enumerate(rxn_names)}
        midx = {x: nG + len(rxn_names) + i for i, x in enumerate(met_names)}
        for r in m.reactions:
            ri = ridx[r.id]
            for g in r.genes:
                s = sym.get(g.id, g.id)
                if s in gidx:
                    add(gidx[s], ri, "catalyses")
            for met, coef in r.metabolites.items():
                add(ri, midx[met.id], "consumes" if coef < 0 else "produces")

    n = nG + len(rxn_names) + len(met_names)
    rows = np.array(rows, np.int64)
    cols = np.array(cols, np.int64)
    etype = np.array(etype, np.int8)
    # symmetric, unweighted; the type vector is kept so a caller can re-weight per channel
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    kind = np.zeros(n, np.int8)
    kind[nG:nG + len(rxn_names)] = 1
    kind[nG + len(rxn_names):] = 2

    out = {"A": A, "kind": kind, "n_gene": nG, "n_rxn": len(rxn_names),
           "n_met": len(met_names), "gene_names": names, "gene_index": gidx,
           "rxn_names": rxn_names, "met_names": met_names,
           "edge_rows": rows, "edge_cols": cols, "edge_type": etype}
    _CACHE[key] = out
    return out


def channel_adjacency(G, types):
    """Adjacency restricted to a subset of edge channels, symmetrised the same way build() does."""
    keep = np.isin(G["edge_type"], [EDGE_TYPES.index(t) for t in types])
    r, c = G["edge_rows"][keep], G["edge_cols"][keep]
    n = G["A"].shape[0]
    A = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def spectral_embed(A, k=64, seed=0):
    """Normalised-Laplacian eigenvectors. Deterministic, no training, so it is the honest floor
    that any learned model has to beat before the learning can be said to have added anything."""
    from scipy.sparse.linalg import eigsh
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    Dm = sp.diags(1.0 / np.sqrt(d))
    L = sp.eye(A.shape[0], format="csr") - Dm @ A @ Dm
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=A.shape[0])
    vals, vecs = eigsh(L, k=k + 1, sigma=-1e-3, which="LM", v0=v0)
    order = np.argsort(vals)
    return vecs[:, order[1:k + 1]].astype(np.float32)


def summary(G):
    A = G["A"]
    deg = np.asarray(A.sum(1)).ravel()
    cnt = collections.Counter(G["edge_type"].tolist())
    return {"nodes": int(A.shape[0]), "genes": G["n_gene"], "reactions": G["n_rxn"],
            "metabolites": G["n_met"], "undirected_edges": int(A.nnz // 2),
            "by_channel": {EDGE_TYPES[k]: int(v) for k, v in sorted(cnt.items())},
            "isolated_nodes": int((deg == 0).sum()),
            "median_degree_gene": float(np.median(deg[:G["n_gene"]])),
            "genes_bridging_to_gem": int(((G["edge_type"] == EDGE_TYPES.index("catalyses")).sum()))}
