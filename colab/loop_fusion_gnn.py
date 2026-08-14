"""LOOP 107-GNN -- DOES FUSING CHEMISTRY WITH INTERACTION HELP, IF THE MODEL IS A GNN?

THE QUESTION, AND WHY A GNN IS THE RIGHT WAY TO ASK IT. cell_graph.py put stoichiometry and
interaction into one typed object for the first time: 16,492 genes, 12,931 Human-GEM reactions and
8,461 metabolites, joined by 333,003 undirected edges over seven named channels. A spectral
embedding of that object cannot tell a catalyses edge from a PPI edge -- it sees one adjacency
matrix and one Laplacian, so "fusion" for it means nothing more than adding edges. A relational GNN
CAN tell them apart, because it carries a separate weight matrix per edge type, and it can therefore
learn that a metabolic neighbour and a binding partner mean different things about whether knocking
a gene out kills the cell. If fusion is ever going to pay, this is the model class where it should.

So the informative output here is not the accuracy. It is the per-channel weight and ablation table:
which of the seven channels the trained model actually leaned on. A fusion that "works" while
putting all its weight on PPI has not fused anything.

THE SHARED PROTOCOL. Target: DepMap dep_frac from outputs/orphan/cell_complete.json, scored as
regression (Spearman) and as a binary label at dep_frac >= 0.5 (AUC). Split: 5-fold CV with
folds from np.random.default_rng(11100).integers(0, 5, size=n_genes), so every arm and every one
of the three model classes being run against this protocol sees IDENTICAL folds. Arms: A1
reaction-only (catalyses/consumes/produces), A2 graph-only (ppi/signal/regulate/complex), A3
combined (all seven), B1 FAME = log1p(pubs) alone, B2 DEGREE = combined-graph degree alone.

THE GATES, PREDECLARED VERBATIM BEFORE ANY NUMBER WAS COMPUTED:

  E1 the intermediate is used as built and its bridge is reported (23,225 catalyses edges,
     only 2,568 of 16,492 genes carry a GEM reaction)
  E2 GATE: the COMBINED arm must beat BOTH A1 and A2 on held-out Spearman, by more than the
     across-fold sd. This is the synergy gate and it is the whole point.
  E3 GATE: the combined arm must beat the FAME baseline B1. Publication count has beaten the
     biology in loops 87, 87b, 94, 96, 97 and 99 in this project. It gets first refusal.
  E4 GATE: the combined arm must beat the DEGREE baseline B2. A graph embedding that only
     recovers degree has learned nothing about structure.
  E5 GATE: a degree-preserving edge rewiring of the combined graph must destroy the effect.
     Use colab/gate_guard.py: GG.null_can_move(real_feature, null_feature) to CONFIRM the null
     changes the statistic's input BEFORE reading its verdict, then GG.survival(real, nulls).
     In loop 94 a "degree-preserving rewiring" preserved in-degree EXACTLY and reported 100%
     survival from a null that was arithmetically unable to move. Do not repeat that.
  E6 report whether your learned model beats CG.spectral_embed + ridge regression, the untrained
     floor. If it does not, say so plainly.

OPERATIONALISATION, ALSO FIXED BEFORE ANY NUMBER (so no threshold can move later):

  E2 passes iff  mean(A3) - mean(A1) > max(sd(A3), sd(A1))  AND the same against A2, where the
     mean and sd are over the five held-out folds and the statistic is Spearman. The paired
     per-fold difference is reported alongside as supporting detail but does NOT set the verdict.
  E3, E4 pass iff mean held-out Spearman of A3 is strictly greater than the baseline's. The
     sd-sized margin is reported next to it but the gate is the strict inequality, as written.
  E5 passes iff (a) GG.null_can_move on the neighbourhoods the GNN actually consumes returns
     capable=True, AND (b) GG.survival(real, nulls) returns defined=True with fraction < 1 --
     i.e. the rewired graph really is a different input, and the real result really does stand
     above it. If the null is inert, E5 FAILS regardless of what the numbers say.
  E6 is reported, not gated. Beating the untrained floor is stated plainly either way.

TWO DEVIATIONS, DECLARED HERE RATHER THAN BURIED:

  (i)  THE TARGET COUNT. The protocol says dep_frac "present and finite ... that is ~6,111 genes".
       Read literally, present-and-finite is 15,913 genes, of which 9,802 sit at exactly 0.0.
       6,111 is exactly the count of genes whose dep_frac is present, finite AND non-zero, which
       is what a plain `if g.get('dep_frac')` truth test selects. This module uses the 6,111 so
       that its folds are identical to the other two model classes', and reports the 15,913
       reading here so the choice is visible rather than silent. Dropping the 9,802 zeros removes
       the easiest negatives and makes both statistics HARDER, not easier.
  (ii) THE COMPUTE. CPU only, 4 threads. Reductions from a textbook setup, all reported in the
       output: hidden width 32, 2 message-passing layers (the protocol allows 2-3), 140 Adam
       epochs with epoch selection on an inner 15% validation split of the training folds only,
       and 3 rewiring nulls rather than 20. Every arm and every null gets the identical budget.
  (iii) THE FLOOR'S ARITHMETIC. CG.spectral_embed uses shift-invert eigsh, which needs a sparse LU
       of L + 1e-3 I; on this 37,884-node graph that factorisation did not finish in ten minutes
       of CPU and would have consumed the whole budget by itself. The same eigenvectors are the
       algebraically largest of S = D^-1/2 A D^-1/2, since L = I - S, and that route is
       matvec-only. E6 uses it, at the protocol's k=64 and seed 0, and the equivalence is not
       asserted but CHECKED against CG.spectral_embed itself on a connected subgraph where the
       shift-invert does finish. The check and its result are in the output.

-> outputs/loop_fusion_gnn.json
"""
import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_replication as LR  # noqa: E402
import gate_guard as GG  # noqa: E402
import cell_graph as CG  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(4)
_HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CELL_OUT", str(_HERE.parent / "outputs")))

RXN_CH = ("catalyses", "consumes", "produces")
GRAPH_CH = ("ppi", "signal", "regulate", "complex")
ALL_CH = RXN_CH + GRAPH_CH

FOLD_SEED = 11100          # fixed by the shared protocol
N_FOLD = 5
HID = 32                   # reduced from 64 for the CPU budget
LAYERS = 2                 # protocol allows 2-3; 2 is enough for gene -> reaction -> gene
RPROJ = 24                 # width of the fixed random projection input
EPOCHS = 140
EVAL_EVERY = 10
LR_ADAM = 0.02
WD = 1e-4
INNER_VAL = 0.15
MODEL_SEED = 0
N_NULL = 3                 # reduced from 20 for the CPU budget
NULL_SEED = 970001
SWAPS_PER_EDGE = 10
SPECTRAL_K = 64
RIDGE_ALPHA = 1.0

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 20 or np.std(a[f]) == 0 or np.std(b[f]) == 0:
        return float("nan")
    return float(spearmanr(a[f], b[f]).statistic)


def auc(ybin, score):
    from scipy.stats import rankdata
    ybin = np.asarray(ybin).astype(bool)
    score = np.asarray(score, float)
    f = np.isfinite(score)
    ybin, score = ybin[f], score[f]
    n1, n0 = int(ybin.sum()), int((~ybin).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[ybin].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))


# --------------------------------------------------------------------------------------------
# graph plumbing
# --------------------------------------------------------------------------------------------
def adj_from_edges(rows, cols, n):
    """Exactly cell_graph.build()'s symmetrise-and-binarise, so a rewired graph is built the same
    way the real one is and no difference can come from the construction."""
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def row_normalise(A):
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    return (sp.diags(1.0 / d) @ A).tocoo()


def to_torch(Acoo):
    i = torch.tensor(np.vstack([Acoo.row, Acoo.col]), dtype=torch.long)
    v = torch.tensor(Acoo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, Acoo.shape).coalesce().to_sparse_csr()


def channel_mats(edges, n, chans):
    """Mean-aggregation (row-normalised) adjacency, one matrix per edge CHANNEL. Kept separate is
    the whole point: the model gets a distinct weight matrix per channel."""
    out = []
    for c in chans:
        r, cc = edges[c]
        out.append(to_torch(row_normalise(adj_from_edges(r, cc, n))))
    return out


def split_edges(G):
    """Per-channel deduplicated undirected edge lists."""
    ed = {}
    for c in ALL_CH:
        keep = G["edge_type"] == CG.EDGE_TYPES.index(c)
        r, cc = G["edge_rows"][keep], G["edge_cols"][keep]
        lo, hi = np.minimum(r, cc), np.maximum(r, cc)
        key = lo.astype(np.int64) * (G["A"].shape[0] + 1) + hi
        _, u = np.unique(key, return_index=True)
        ed[c] = (lo[u].copy(), hi[u].copy())
    return ed


def rewire_channel(r, c, rng, swaps_per_edge=None):
    """Double-edge swap. (a,b),(x,y) -> (a,y),(x,b): every node keeps exactly its endpoint count,
    so the degree sequence is preserved EXACTLY, and the orientation is never flipped so the
    bipartite channels (gene->reaction, reaction->metabolite) stay bipartite."""
    swaps_per_edge = SWAPS_PER_EDGE if swaps_per_edge is None else swaps_per_edge
    m = len(r)
    rl, cl = r.tolist(), c.tolist()
    S = set()
    for a, b in zip(rl, cl):
        S.add((a, b) if a < b else (b, a))
    n_try = int(swaps_per_edge * m)
    ii = rng.integers(0, m, n_try)
    jj = rng.integers(0, m, n_try)
    done = 0
    for k in range(n_try):
        i, j = int(ii[k]), int(jj[k])
        if i == j:
            continue
        a, b = rl[i], cl[i]
        x, y = rl[j], cl[j]
        if a == y or x == b:
            continue
        k1 = (a, y) if a < y else (y, a)
        k2 = (x, b) if x < b else (b, x)
        if k1 == k2 or k1 in S or k2 in S:
            continue
        S.discard((a, b) if a < b else (b, a))
        S.discard((x, y) if x < y else (y, x))
        S.add(k1)
        S.add(k2)
        rl[i], cl[i] = a, y
        rl[j], cl[j] = x, b
        done += 1
    return np.asarray(rl, np.int64), np.asarray(cl, np.int64), done


def rewire_all(edges, rng):
    return {c: rewire_channel(edges[c][0], edges[c][1], rng)[:2] for c in ALL_CH}


def neighbourhood_fingerprint(A):
    """The statistic's INPUT, as the GNN sees it: each node's neighbour set. Loop 94's null moved
    nothing because it was checked on a quantity it could not change; this is checked on the
    quantity the message passing actually reads."""
    A = A.tocsr()
    ip, ix = A.indptr, A.indices
    return np.array([hash(np.sort(ix[ip[i]:ip[i + 1]]).tobytes()) for i in range(A.shape[0])],
                    dtype=np.int64)


def combined_adj(edges, n):
    r = np.concatenate([edges[c][0] for c in ALL_CH])
    c = np.concatenate([edges[c][1] for c in ALL_CH])
    return adj_from_edges(r, c, n)


def spectral_floor(A, k=SPECTRAL_K, seed=0):
    """CG.spectral_embed's object, reached by Lanczos instead of shift-invert -- see deviation
    (iii). L = I - S with S = D^-1/2 A D^-1/2, so the k smallest eigenvectors of L are the k
    largest of S, and eigsh(S, which='LA') needs only matvecs. Same k, same seed, same v0, same
    'drop the first' convention as CG.spectral_embed."""
    from scipy.sparse.linalg import eigsh
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = (Dm @ A @ Dm).tocsr()
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=A.shape[0])
    vals, vecs = eigsh(S, k=k + 1, which="LA", v0=v0, tol=1e-6)
    lam = 1.0 - vals
    order = np.argsort(lam)
    return vecs[:, order[1:k + 1]].astype(np.float32), lam[order[:k + 1]]


def spectral_equivalence_check(G, emit=print, n_sub=900, k=12):
    """Run BOTH routines on a subgraph small enough for the shift-invert to finish, and compare.

    Two traps, both hit on the way here and both worth stating. The subgraph must be CONNECTED,
    or the bottom of the spectrum is a degenerate null space whose basis is arbitrary. And it must
    not be a STAR: the first attempt took a BFS ball around the highest-degree node, which in this
    graph is 1,199 leaves on one hub, whose entire non-trivial spectrum is the eigenvalue 1 with
    multiplicity 1,198 -- both routines returned identical eigenvalues and completely different
    (equally correct) bases. So the induced subgraph on the highest-degree nodes is used instead,
    and the eigenVALUES, which are basis-independent, are compared as well as the subspace.
    """
    A = G["A"].tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    idx = np.sort(np.argsort(-deg)[:n_sub])
    Asub = A[idx][:, idx].tocsr()
    lab = sp.csgraph.connected_components(Asub, directed=False)
    big = np.argmax(np.bincount(lab[1]))
    keep = np.where(lab[1] == big)[0]
    Asub = Asub[keep][:, keep].tocsr()
    E_ref = CG.spectral_embed(Asub, k=k, seed=0)
    E_new, lam = spectral_floor(Asub, k=k, seed=0)
    lam_ref = np.sort(np.linalg.eigvalsh(
        (sp.eye(Asub.shape[0]) - sp.diags(1 / np.sqrt(np.maximum(
            np.asarray(Asub.sum(1)).ravel(), 1))) @ Asub @ sp.diags(
            1 / np.sqrt(np.maximum(np.asarray(Asub.sum(1)).ravel(), 1)))).toarray()))[:k + 1]
    dlam = float(np.max(np.abs(np.sort(lam) - lam_ref)))
    q1, _ = np.linalg.qr(E_ref)
    q2, _ = np.linalg.qr(E_new)
    sv = np.linalg.svd(q1.T @ q2, compute_uv=False)
    degenerate = bool(np.min(np.diff(np.sort(lam))) < 1e-6)
    ok = bool(dlam < 1e-4 and (sv.min() > 0.99 or degenerate))
    emit(f"     EQUIVALENCE CHECK: induced subgraph on the {n_sub} highest-degree nodes, largest "
         f"component n={Asub.shape[0]}, k={k}")
    emit(f"       eigenvalues vs a DENSE reference eigvalsh: max abs diff {dlam:.2e}; "
         f"spectrum degenerate at the bottom: {degenerate}")
    emit(f"       principal angles between CG.spectral_embed's subspace and this one: "
         f"cos min {sv.min():.6f}, mean {sv.mean():.6f}")
    emit(f"       -> the two routines compute the same object: {ok}")
    return {"n_sub": int(Asub.shape[0]), "k": k, "max_abs_eigenvalue_diff": dlam,
            "degenerate": degenerate, "cos_min": float(sv.min()), "cos_mean": float(sv.mean()),
            "identical": ok}


# --------------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------------
class RGNN(nn.Module):
    """H <- act( sum_t Anorm_t H W_t + H W_self ), one W_t per edge TYPE.

    The per-type blocks are stored as one (T*d_in, d_out) matrix so the whole sum is a single
    GEMM -- mathematically identical to T separate products, several times faster on 4 CPU
    threads, and the per-type block is still individually readable (and individually zeroable,
    which is how the ablation table below is produced)."""

    def __init__(self, in_dim, hid, n_type, n_layer, seed=MODEL_SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.T, self.n_layer = n_type, n_layer
        dims = [in_dim] + [hid] * n_layer
        self.msg = nn.ModuleList([nn.Linear(n_type * dims[i], dims[i + 1], bias=False)
                                  for i in range(n_layer)])
        self.slf = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(n_layer)])
        self.head = nn.Linear(dims[-1], 1)
        self.dims = dims

    def forward(self, X, mats, drop_type=None):
        H = X
        for li in range(self.n_layer):
            M = torch.cat([torch.sparse.mm(A, H) for A in mats], 1)
            W = self.msg[li].weight                              # (d_out, T*d_in)
            if drop_type is not None:
                d_in = self.dims[li]
                W = W.clone()
                W[:, drop_type * d_in:(drop_type + 1) * d_in] = 0.0
            H = torch.relu(M @ W.t() + self.slf[li](H))
        return self.head(H).squeeze(1)

    def type_norms(self):
        out = []
        for li in range(self.n_layer):
            d_in = self.dims[li]
            W = self.msg[li].weight.detach()
            out.append([float(W[:, t * d_in:(t + 1) * d_in].norm()) for t in range(self.T)])
        return out


def train_gnn(X, mats, gene_pos, y, train_mask, seed, epochs=None):
    """Adam on the training folds only. Epoch selection uses an inner 15% split of the TRAINING
    genes -- the held-out fold is never read during training or model selection. The returned
    model is RESTORED to the selected epoch, so the weight norms and the channel ablation below
    describe the same model whose predictions were scored."""
    epochs = EPOCHS if epochs is None else epochs
    rng = np.random.default_rng(1000 + seed)
    tr_idx = np.where(train_mask)[0]
    rng.shuffle(tr_idx)
    n_val = max(30, int(INNER_VAL * len(tr_idx)))
    val_idx, fit_idx = tr_idx[:n_val], tr_idx[n_val:]

    mu, sd = float(y[fit_idx].mean()), float(y[fit_idx].std()) or 1.0
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    gp = torch.tensor(gene_pos, dtype=torch.long)
    fit_t = torch.tensor(fit_idx, dtype=torch.long)

    model = RGNN(X.shape[1], HID, len(mats), LAYERS, seed=MODEL_SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR_ADAM, weight_decay=WD)
    best = (-2.0, None, -1, None)
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred_all = model(X, mats)
        p = pred_all[gp]
        loss = ((p[fit_t] - yt[fit_t]) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % EVAL_EVERY == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                p = model(X, mats)[gp].numpy()
            rv = spearman(p[val_idx], y[val_idx])
            if np.isfinite(rv) and rv > best[0]:
                best = (rv, p.copy(), ep, copy.deepcopy(model.state_dict()))
    if best[1] is None:
        model.eval()
        with torch.no_grad():
            best = (float("nan"), model(X, mats)[gp].numpy(), epochs, None)
    if best[3] is not None:
        model.load_state_dict(best[3])
    model.eval()
    return model, best[1], best[0], best[2]


def ridge_fit_predict(Xtr, ytr, Xte, alpha=RIDGE_ALPHA):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean()
    G = A.T @ A + alpha * np.eye(A.shape[1])
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    return B @ w + ym


def cv_feature(F, y, ybin, fold):
    """Same folds, same held-out statistics, for any fixed feature matrix (baselines + floor)."""
    rs, au = [], []
    for f in range(N_FOLD):
        te = fold == f
        tr = ~te
        p = ridge_fit_predict(F[tr], y[tr], F[te])
        rs.append(spearman(p, y[te]))
        au.append(auc(ybin[te], p))
    return rs, au


def cv_gnn(X, mats, gene_pos, y, ybin, fold, tag, quiet=False):
    rs, au, models, preds = [], [], [], []
    for f in range(N_FOLD):
        t0 = time.time()
        te = fold == f
        model, p, vr, bep = train_gnn(X, mats, gene_pos, y, ~te, seed=f)
        rs.append(spearman(p[te], y[te]))
        au.append(auc(ybin[te], p[te]))
        models.append(model)
        preds.append(p)
        if not quiet:
            say(f"       {tag} fold {f}: rho {rs[-1]:+.4f}  auc {au[-1]:.4f}  "
                f"(inner-val rho {vr:+.4f} @ epoch {bep}, {time.time() - t0:.0f}s)")
    return rs, au, models, preds


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 107-GNN -- fuse chemistry with interaction, and ask a relational GNN if it helped")
    say("=" * 100)
    say()

    # -------------------------------------------------------------------- E1
    say("E1 THE INTERMEDIATE IS USED AS BUILT, AND ITS BRIDGE IS REPORTED")
    G = CG.build()
    S = CG.summary(G)
    n = G["A"].shape[0]
    say(f"     {S['nodes']:,} nodes = {S['genes']:,} genes + {S['reactions']:,} reactions + "
        f"{S['metabolites']:,} metabolites")
    say(f"     {S['undirected_edges']:,} undirected edges over {len(S['by_channel'])} channels: " +
        ", ".join(f"{k} {v:,}" for k, v in S["by_channel"].items()))
    cat = G["edge_type"] == CG.EDGE_TYPES.index("catalyses")
    n_cat = int(cat.sum())
    n_bridge = int(len(set(G["edge_rows"][cat].tolist())))
    say(f"     THE BRIDGE: {n_cat:,} catalyses edges, carried by {n_bridge:,} of {S['genes']:,} "
        f"genes ({n_bridge / S['genes']:.1%})")
    gc.collect()
    gc.freeze()          # the cobra model leaves millions of live objects behind; without this
    #                      every gen-2 sweep walks all of them and a 0.05 s epoch becomes 1 s
    e1 = (n_cat == 23225 and n_bridge == 2568)
    say(f"     matches the declared bridge (23,225 / 2,568): {e1}")
    say(f"     {S['isolated_nodes']:,} nodes are isolated in the combined graph")
    say()

    # -------------------------------------------------------------------- target + folds
    C = json.load(open(LR.CELL))
    genes = C["genes"]
    names = [g["name"] for g in genes]
    assert names[:5] == G["gene_names"][:5] and len(names) == G["n_gene"]
    dep_raw = np.array([np.nan if g.get("dep_frac") is None else float(g["dep_frac"])
                        for g in genes])
    finite = np.isfinite(dep_raw)
    sel = finite & (dep_raw != 0.0)
    gene_pos = np.where(sel)[0]                     # node index == gene index in the graph
    y = dep_raw[gene_pos].astype(np.float64)
    ybin = (y >= 0.5).astype(int)
    ng = len(gene_pos)
    say("THE TARGET AND THE SPLIT")
    say(f"     dep_frac present and finite: {int(finite.sum()):,} genes, of which "
        f"{int((dep_raw == 0).sum()):,} are exactly 0.0")
    say(f"     used: {ng:,} genes (present, finite and non-zero) -- this is the protocol's "
        f"'~6,111', see the docstring deviation (i)")
    say(f"     dep_frac  median {np.median(y):.3f}  mean {y.mean():.3f}   "
        f"positives at >= 0.5: {ybin.sum():,} ({ybin.mean():.1%})")
    rng = np.random.default_rng(FOLD_SEED)
    fold = rng.integers(0, N_FOLD, size=ng)
    say(f"     folds from default_rng({FOLD_SEED}).integers(0,5,size={ng}): sizes " +
        ", ".join(str(int((fold == f).sum())) for f in range(N_FOLD)))
    n_cat_target = int(np.isin(gene_pos, np.unique(G["edge_rows"][cat])).sum())
    say(f"     of the {ng:,} scored genes, {n_cat_target:,} ({n_cat_target / ng:.1%}) carry a GEM "
        f"reaction -- the reaction-only arm is blind to the other {ng - n_cat_target:,}")
    say()

    # -------------------------------------------------------------------- inputs
    edges = split_edges(G)
    rs_seed = np.random.default_rng(MODEL_SEED)
    kind = G["kind"]
    onehot = np.zeros((n, 3), np.float32)
    onehot[np.arange(n), kind] = 1.0
    proj = (rs_seed.normal(size=(n, RPROJ)) / np.sqrt(RPROJ)).astype(np.float32)
    X = torch.tensor(np.hstack([onehot, proj]).astype(np.float32), dtype=torch.float32)
    say("THE MODEL")
    say(f"     inputs: one-hot node kind (3) + a fixed random projection of node identity "
        f"({RPROJ}), seed {MODEL_SEED}")
    say(f"     {LAYERS} layers of mean-aggregation message passing, hidden {HID}, one weight "
        f"matrix per edge TYPE plus a self matrix")
    say(f"     Adam lr {LR_ADAM}, weight decay {WD}, {EPOCHS} epochs, epoch chosen on an inner "
        f"{INNER_VAL:.0%} split of the TRAINING folds")
    say(f"     NOTE, because it bears on E4: mean-aggregated random features have norm ~ "
        f"1/sqrt(degree), so a GNN CAN read degree off them. That is exactly why E4 exists, and")
    say(f"     the correlation between this model's predictions and degree is reported below.")
    say()

    arms = {}
    say("A1 / A2 / A3 -- THE THREE GRAPHS, SAME MODEL, SAME FOLDS")
    for tag, chans in (("A1 reaction-only", RXN_CH), ("A2 graph-only", GRAPH_CH),
                       ("A3 combined", ALL_CH)):
        mats = channel_mats(edges, n, chans)
        rs, au, models, preds = cv_gnn(X, mats, gene_pos, y, ybin, fold, tag)
        arms[tag] = {"channels": list(chans), "rho": rs, "auc": au}
        if tag == "A3 combined":
            comb_models, comb_preds, comb_mats = models, preds, mats
        say(f"     {tag:<18} rho {ms(rs)[0]:+.4f} +/- {ms(rs)[1]:.4f}    "
            f"auc {ms(au)[0]:.4f} +/- {ms(au)[1]:.4f}")
    say()

    # -------------------------------------------------------------------- baselines
    say("B1 / B2 -- THE TWO BASELINES, SAME FOLDS")
    pubs = np.array([float(g.get("pubs") or 0.0) for g in genes])[gene_pos]
    deg_all = np.asarray(G["A"].sum(1)).ravel()
    deg = deg_all[gene_pos]
    RM.check_features({"log1p_pubs": np.log1p(pubs), "degree": deg},
                      ["log1p_pubs", "degree"], emit=say)
    for tag, F in (("B1 FAME log1p(pubs)", np.log1p(pubs)[:, None]),
                   ("B2 DEGREE combined", deg[:, None].astype(float))):
        rs, au = cv_feature(F, y, ybin, fold)
        arms[tag] = {"channels": [], "rho": rs, "auc": au}
        say(f"     {tag:<22} rho {ms(rs)[0]:+.4f} +/- {ms(rs)[1]:.4f}    "
            f"auc {ms(au)[0]:.4f} +/- {ms(au)[1]:.4f}")
    say(f"     raw (untrained) Spearman of the features themselves: log1p(pubs) "
        f"{spearman(np.log1p(pubs), y):+.4f}, degree {spearman(deg, y):+.4f}")
    say()

    # -------------------------------------------------------------------- which channel
    say("WHICH CHANNEL DID THE MODEL ACTUALLY USE -- the most informative output here")
    tn = np.mean([m.type_norms() for m in comb_models], axis=0)     # (layers, T)
    contrib = np.zeros(len(ALL_CH))
    with torch.no_grad():
        m0 = comb_models[0]
        H = X
        for li in range(m0.n_layer):
            Ms = [torch.sparse.mm(A, H) for A in comb_mats]
            d_in = m0.dims[li]
            W = m0.msg[li].weight.detach()
            for t in range(len(ALL_CH)):
                Ct = Ms[t] @ W[:, t * d_in:(t + 1) * d_in].t()
                contrib[t] += float(Ct[torch.tensor(gene_pos)].norm(dim=1).mean())
            H = torch.relu(torch.cat(Ms, 1) @ W.t() + m0.slf[li](H))
    contrib = contrib / contrib.sum()
    abl, loo = {}, {}
    base_rho = ms(arms["A3 combined"]["rho"])[0]
    for t, c in enumerate(ALL_CH):
        drs = []
        for f in range(N_FOLD):
            te = fold == f
            with torch.no_grad():
                p = comb_models[f](X, comb_mats, drop_type=t)[torch.tensor(gene_pos)].numpy()
            drs.append(spearman(p[te], y[te]))
        abl[c] = ms(drs)[0]
        rest = [x for x in ALL_CH if x != c]
        rs_l, _, _, _ = cv_gnn(X, channel_mats(edges, n, rest), gene_pos, y, ybin, fold,
                               f"-{c}", quiet=True)
        loo[c] = ms(rs_l)
    say(f"     {'channel':<12}{'|W_t| L1':>10}{'|W_t| L2':>10}{'msg share':>11}"
        f"{'zeroed':>9}{'drop':>8}{'RETRAINED without it':>22}{'drop':>8}")
    chan_table = {}
    for t, c in enumerate(ALL_CH):
        say(f"     {c:<12}{tn[0][t]:>10.3f}{tn[1][t]:>10.3f}{contrib[t]:>10.1%}"
            f"{abl[c]:>9.4f}{base_rho - abl[c]:>+8.4f}"
            f"{loo[c][0]:>16.4f} +/- {loo[c][1]:.3f}{base_rho - loo[c][0]:>+8.4f}")
        chan_table[c] = {"w_norm_layer": [float(tn[li][t]) for li in range(LAYERS)],
                         "message_share": float(contrib[t]), "rho_zeroed": float(abl[c]),
                         "drop_zeroed": float(base_rho - abl[c]),
                         "rho_retrained_without": float(loo[c][0]),
                         "sd_retrained_without": float(loo[c][1]),
                         "drop_retrained": float(base_rho - loo[c][0])}
    say(f"     'zeroed' sets W_t to zero in the ALREADY-TRAINED model: it perturbs a jointly")
    say(f"     trained representation, so every channel shows a drop and the column ranks rather")
    say(f"     than measures. 'RETRAINED without it' refits the whole model on the other six")
    say(f"     channels and is the column to read.")
    top = max(chan_table, key=lambda k: chan_table[k]["drop_retrained"])
    rxn_drop = sum(chan_table[c]["drop_retrained"] for c in RXN_CH)
    gph_drop = sum(chan_table[c]["drop_retrained"] for c in GRAPH_CH)
    say(f"     the single channel the model leans on hardest (retrained): {top}")
    say(f"     summed retrained drop -- reaction channels {rxn_drop:+.4f}, "
        f"graph channels {gph_drop:+.4f}")
    say()

    # degree confound diagnostic
    dcorr = [spearman(comb_preds[f][fold == f], deg[fold == f]) for f in range(N_FOLD)]
    say(f"     DEGREE CONFOUND: held-out Spearman(GNN prediction, degree) "
        f"{ms(dcorr)[0]:+.4f} +/- {ms(dcorr)[1]:.4f}")
    say()

    # -------------------------------------------------------------------- E2/E3/E4
    say("E2 GATE -- COMBINED MUST BEAT BOTH A1 AND A2 BY MORE THAN THE ACROSS-FOLD SD")
    m3, s3 = ms(arms["A3 combined"]["rho"])
    m1, s1 = ms(arms["A1 reaction-only"]["rho"])
    m2, s2 = ms(arms["A2 graph-only"]["rho"])
    r3 = np.asarray(arms["A3 combined"]["rho"])
    p1, p2 = r3 - np.asarray(arms["A1 reaction-only"]["rho"]), \
        r3 - np.asarray(arms["A2 graph-only"]["rho"])
    e2a = (m3 - m1) > max(s3, s1)
    e2b = (m3 - m2) > max(s3, s2)
    e2 = bool(e2a and e2b)
    say(f"     A3 {m3:+.4f} +/- {s3:.4f}   A1 {m1:+.4f} +/- {s1:.4f}   A2 {m2:+.4f} +/- {s2:.4f}")
    say(f"     A3 - A1 = {m3 - m1:+.4f} vs threshold {max(s3, s1):.4f}  -> {'PASS' if e2a else 'FAIL'}")
    say(f"     A3 - A2 = {m3 - m2:+.4f} vs threshold {max(s3, s2):.4f}  -> {'PASS' if e2b else 'FAIL'}")
    say(f"     paired per-fold (support only, not the verdict): A3-A1 {ms(p1)[0]:+.4f} +/- "
        f"{ms(p1)[1]:.4f}, A3-A2 {ms(p2)[0]:+.4f} +/- {ms(p2)[1]:.4f}")
    say(f"     E2 {'PASS' if e2 else 'FAIL'}")
    say()

    say("E3 GATE -- COMBINED MUST BEAT THE FAME BASELINE")
    mb1, sb1 = ms(arms["B1 FAME log1p(pubs)"]["rho"])
    e3 = bool(m3 > mb1)
    say(f"     A3 {m3:+.4f} +/- {s3:.4f}   B1 fame {mb1:+.4f} +/- {sb1:.4f}   "
        f"margin {m3 - mb1:+.4f} (sd-sized margin would be {max(s3, sb1):.4f})")
    say(f"     AUC: A3 {ms(arms['A3 combined']['auc'])[0]:.4f}   "
        f"B1 {ms(arms['B1 FAME log1p(pubs)']['auc'])[0]:.4f}")
    say(f"     E3 {'PASS' if e3 else 'FAIL'}")
    say()

    say("E4 GATE -- COMBINED MUST BEAT THE DEGREE BASELINE")
    mb2, sb2 = ms(arms["B2 DEGREE combined"]["rho"])
    e4 = bool(m3 > mb2)
    say(f"     A3 {m3:+.4f} +/- {s3:.4f}   B2 degree {mb2:+.4f} +/- {sb2:.4f}   "
        f"margin {m3 - mb2:+.4f} (sd-sized margin would be {max(s3, sb2):.4f})")
    say(f"     AUC: A3 {ms(arms['A3 combined']['auc'])[0]:.4f}   "
        f"B2 {ms(arms['B2 DEGREE combined']['auc'])[0]:.4f}")
    say(f"     E4 {'PASS' if e4 else 'FAIL'}")
    say()

    # -------------------------------------------------------------------- E5
    say("E5 GATE -- A DEGREE-PRESERVING REWIRING MUST DESTROY THE EFFECT")
    A_real = combined_adj(edges, n)
    fp_real = neighbourhood_fingerprint(A_real)
    deg_real = np.asarray(A_real.sum(1)).ravel()
    null_rho, caps = [], []
    for b in range(N_NULL):
        tb = time.time()
        rr = np.random.default_rng(NULL_SEED + b)
        ed_n = rewire_all(edges, rr)
        A_n = combined_adj(ed_n, n)
        fp_n = neighbourhood_fingerprint(A_n)
        deg_n = np.asarray(A_n.sum(1)).ravel()
        cap = GG.null_can_move(fp_real, fp_n)
        deg_move = GG.null_can_move(deg_real, deg_n)
        per_type_ok = all(
            np.array_equal(np.sort(np.bincount(np.concatenate(edges[c]), minlength=n)),
                           np.sort(np.bincount(np.concatenate(ed_n[c]), minlength=n)))
            for c in ALL_CH)
        exact_deg = all(np.array_equal(np.bincount(np.concatenate(edges[c]), minlength=n),
                                       np.bincount(np.concatenate(ed_n[c]), minlength=n))
                        for c in ALL_CH)
        caps.append({"neighbourhoods_changed": cap["changed"], "capable": bool(cap["capable"]),
                     "degree_entries_changed": deg_move["changed"],
                     "per_type_degree_identical": bool(exact_deg),
                     "per_type_degree_sequence_identical": bool(per_type_ok)})
        say(f"     null {b}: CAPABILITY CHECK on the neighbourhoods the GNN reads -- "
            f"{cap['changed']:.1%} of nodes changed, capable={cap['capable']}")
        say(f"             per-channel degree preserved EXACTLY: {exact_deg}; combined-degree "
            f"entries that changed at all: {deg_move['changed']:.2%}")
        mats_n = channel_mats(ed_n, n, ALL_CH)
        rs_n = []
        for f in range(N_FOLD):
            te = fold == f
            _, p, _, _ = train_gnn(X, mats_n, gene_pos, y, ~te, seed=f)
            rs_n.append(spearman(p[te], y[te]))
        null_rho.append(ms(rs_n)[0])
        say(f"             rewired combined arm: rho {ms(rs_n)[0]:+.4f} +/- {ms(rs_n)[1]:.4f}  "
            f"[{time.time() - tb:.0f}s]")
    say(f"     NOTE, and this is the loop-94 lesson: this rewiring preserves every per-channel")
    say(f"     degree exactly, so the DEGREE baseline B2 is arithmetically INVARIANT under it and")
    say(f"     the null could never move B2. It is checked against the neighbourhoods instead,")
    say(f"     which is the input the message passing actually consumes.")
    e5_cap = all(c["capable"] for c in caps)
    surv = GG.survival(m3, null_rho)
    GG.report("combined-arm held-out rho under degree-preserving rewiring", surv, emit=say)
    e5 = bool(e5_cap and surv.get("defined") and surv.get("fraction") != GG.UNDEFINED
              and isinstance(surv.get("fraction"), float) and surv["fraction"] < 1.0)
    say(f"     E5 {'PASS' if e5 else 'FAIL'} -- null capable: {e5_cap}, "
        f"survival defined: {bool(surv.get('defined'))}")
    say()

    # -------------------------------------------------------------------- E6
    say("E6 THE UNTRAINED FLOOR -- CG.spectral_embed + ridge")
    eqv = spectral_equivalence_check(G, emit=say)
    floor = {}
    for tag, chans in (("A1 reaction-only", RXN_CH), ("A2 graph-only", GRAPH_CH),
                       ("A3 combined", ALL_CH)):
        tsp = time.time()
        Ach = CG.channel_adjacency(G, list(chans))
        E, lam = spectral_floor(Ach, k=SPECTRAL_K, seed=0)
        say(f"       {tag}: Laplacian eigenvalues {lam[0]:.5f} .. {lam[-1]:.5f}")
        rs, au = cv_feature(E[gene_pos].astype(np.float64), y, ybin, fold)
        floor[tag] = {"rho": rs, "auc": au}
        say(f"     floor {tag:<18} rho {ms(rs)[0]:+.4f} +/- {ms(rs)[1]:.4f}   "
            f"auc {ms(au)[0]:.4f} +/- {ms(au)[1]:.4f}   [{time.time() - tsp:.0f}s, k={SPECTRAL_K}]")
    mf3 = ms(floor["A3 combined"]["rho"])[0]
    e6 = bool(m3 > mf3)
    say(f"     learned GNN {m3:+.4f} vs untrained spectral floor {mf3:+.4f} on the SAME combined "
        f"graph, margin {m3 - mf3:+.4f}")
    if e6:
        say(f"     the learning adds something over the floor.")
    else:
        say(f"     PLAINLY: the trained GNN does NOT beat the untrained spectral floor. Whatever")
        say(f"     the arms show, the training is not what produced it.")
    say()

    gates = {"E1 intermediate used as built, bridge reported": bool(e1),
             "E2 combined beats A1 and A2 by more than the fold sd": e2,
             "E3 combined beats the FAME baseline": e3,
             "E4 combined beats the DEGREE baseline": e4,
             "E5 degree-preserving rewiring destroys the effect": e5,
             "E6 learned model beats the untrained spectral floor": e6}
    say("GATES")
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")
    say()

    man = RM.manifest(
        inputs=[str(LR.CELL), str(_HERE / "cell_graph.py"), str(_HERE.parent / "HumanGEM.xml")],
        available=int(finite.sum()), used=ng, selection="filtered", seed=FOLD_SEED,
        controls=["A1 reaction-only and A2 graph-only arms on the identical folds",
                  "B1 publication-count baseline",
                  "B2 combined-degree baseline",
                  "per-channel ablation of the trained combined model",
                  "degree-preserving double-edge-swap rewiring, checked for capability first",
                  "CG.spectral_embed + ridge as the untrained floor",
                  "held-out correlation between prediction and degree, reported"],
        note="relational GNN with one weight matrix per edge type over cell_graph.build(); folds "
             "fixed by the shared protocol so three model classes see identical splits")
    RM.report(man, emit=say)

    res = {"test": "loop_fusion_gnn", "model_class": "GNN", "manifest": man, "gates": gates,
           "protocol": {"fold_seed": FOLD_SEED, "n_fold": N_FOLD, "n_genes": ng,
                        "n_dep_frac_finite": int(finite.sum()),
                        "n_dep_frac_zero": int((dep_raw == 0).sum()),
                        "positives_at_0.5": int(ybin.sum())},
           "graph": S, "bridge": {"catalyses_edges": n_cat, "genes_with_reaction": n_bridge,
                                  "scored_genes_with_reaction": n_cat_target},
           "hyper": {"hidden": HID, "layers": LAYERS, "rproj": RPROJ, "epochs": EPOCHS,
                     "lr": LR_ADAM, "weight_decay": WD, "inner_val": INNER_VAL,
                     "n_null": N_NULL, "swaps_per_edge": SWAPS_PER_EDGE, "spectral_k": SPECTRAL_K,
                     "reductions": "hidden 32 not 64; 2 layers not 3; 140 epochs; 3 rewiring "
                                   "nulls not 20 -- CPU-only budget, identical for every arm"},
           "arms": {k: {"channels": v["channels"],
                        "rho": v["rho"], "rho_mean": ms(v["rho"])[0], "rho_sd": ms(v["rho"])[1],
                        "auc": v["auc"], "auc_mean": ms(v["auc"])[0], "auc_sd": ms(v["auc"])[1]}
                    for k, v in arms.items()},
           "channel_table": chan_table,
           "channel_summary": {"hardest_channel": top, "reaction_drop_sum": float(rxn_drop),
                               "graph_drop_sum": float(gph_drop)},
           "degree_confound": {"rho_pred_vs_degree_mean": ms(dcorr)[0],
                               "rho_pred_vs_degree_sd": ms(dcorr)[1]},
           "e5": {"capability": caps, "null_rho": null_rho, "survival": surv},
           "e6_floor": {k: {"rho": v["rho"], "rho_mean": ms(v["rho"])[0],
                            "auc_mean": ms(v["auc"])[0]} for k, v in floor.items()},
           "e6_equivalence_check": eqv,
           "seconds": time.time() - t0, "log": log}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "loop_fusion_gnn.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_fusion_gnn.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
