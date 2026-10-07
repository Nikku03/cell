"""perturb_prioritizer — turn INTERVENTIONAL (Perturb-seq) data into an experiment prioritizer, and measure it
HONESTLY on top-K: given an observed cell-state signature, rank candidate genes by "perturbing this gene causes
this state" and check whether the true driver lands in the top 10.

Why interventional: every correlational feature in this project (topology, coexpression, domains, AF structure)
sits at chance on genuine discovery. Perturb-seq measures the CAUSAL effect of knocking each gene down, so the
signal is interventional, not correlational.

The honest, NON-circular benchmark = CROSS-CELL-LINE identity recovery. A perturbation's signature is measured in
cell line A (RPE1); the reference effect library is a DIFFERENT cell line (K562). Rank the K562 library by match to
the RPE1 signature; record the rank of the TRUE perturbed gene. Top-K recall = fraction of genes whose truth lands
in the top K. Within-line recovery would be circular (a signature trivially matches its own row); cross-line forces
the causal fingerprint to generalise across context — the real "can we identify the driver" question.

Data: Replogle et al. 2022 processed pseudobulk (figshare 20029387): K562_essential + rpe1 normalized_bulk h5ad,
each ~2,000 gene knockdowns x ~8,500 genes (normalised delta). Read directly with h5py (no anndata needed).
-> outputs/orphan/perturb_prioritizer.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

OUT = "outputs/orphan"
SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def load_pseudobulk(path):
    """-> (X [n_pert x n_gene] float32, pert_genes [n_pert], gene_syms [n_gene]). Perturbation target = 2nd field
    of the obs index label (e.g. '10023_ZC3H18_P1P2_ENSG..' -> 'ZC3H18'); gene symbols decoded from var categories."""
    import h5py
    f = h5py.File(path, "r")
    X = f["X"][:]
    idxkey = f["obs"].attrs.get("_index", "gene_transcript")
    labels = [x.decode() if isinstance(x, bytes) else x for x in f["obs"][idxkey][:]]
    pert_genes = [l.split("_")[1] if len(l.split("_")) > 1 else l for l in labels]
    gn = f["var"]["gene_name"]
    if gn.dtype.kind in ("i", "u"):                       # categorical codes -> decode via __categories
        cats = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["__categories"]["gene_name"][:]]
        gene_syms = [cats[c] for c in gn[:]]
    else:
        gene_syms = [x.decode() if isinstance(x, bytes) else x for x in gn[:]]
    f.close()
    return X.astype(np.float32), pert_genes, gene_syms


def _dedup_rows(X, pert_genes):
    """average replicate perturbations of the same gene -> one row per gene."""
    from collections import defaultdict
    rows = defaultdict(list)
    for i, g in enumerate(pert_genes):
        rows[g].append(i)
    genes = sorted(rows)
    M = np.vstack([X[rows[g]].mean(0) for g in genes])
    return M, genes


def _rank_recall(ranks, ks=(1, 5, 10, 50, 100)):
    ranks = np.asarray(ranks)
    return {f"top_{k}": round(float((ranks <= k).mean()), 3) for k in ks} | {
        "median_rank": int(np.median(ranks)), "n": len(ranks)}


def cross_line_topk(lib_path, qry_path, method="cosine", min_var_genes=2000, restrict=None):
    """rank the LIBRARY (lib cell line) perturbations by match to each QUERY (other cell line) signature; record the
    rank of the TRUE gene. method: 'cosine' (raw), 'zcosine' (per-gene standardised across the library), 'spearman'."""
    Xl, pl, gl = load_pseudobulk(lib_path)
    Xq, pq, gq = load_pseudobulk(qry_path)
    Ml, lib_genes = _dedup_rows(Xl, pl)
    Mq, qry_genes = _dedup_rows(Xq, pq)
    # align on shared measured genes
    shared = [g for g in gl if g in set(gq)]
    gi_l = {g: i for i, g in enumerate(gl)}; gi_q = {g: i for i, g in enumerate(gq)}
    cl = [gi_l[g] for g in shared]; cq = [gi_q[g] for g in shared]
    Ml = Ml[:, cl]; Mq = Mq[:, cq]
    # optional restriction to the most informative (high-variance across the library) genes
    if min_var_genes and min_var_genes < Ml.shape[1]:
        v = Ml.var(0)
        keep = np.argsort(-v)[:min_var_genes]
        Ml = Ml[:, keep]; Mq = Mq[:, keep]
    if method == "zcosine":                               # standardise each gene across the library (whiten scale)
        mu = Ml.mean(0); sd = Ml.std(0) + 1e-8
        Ml = (Ml - mu) / sd; Mq = (Mq - mu) / sd
    if method == "spearman":                              # rank-transform each signature (robust to cross-line scale)
        Ml = np.argsort(np.argsort(Ml, 1), 1).astype(np.float32)
        Mq = np.argsort(np.argsort(Mq, 1), 1).astype(np.float32)
    # unit-normalise rows -> cosine via matmul
    Ln = Ml / (np.linalg.norm(Ml, axis=1, keepdims=True) + 1e-8)
    Qn = Mq / (np.linalg.norm(Mq, axis=1, keepdims=True) + 1e-8)
    lib_idx = {g: i for i, g in enumerate(lib_genes)}
    both = [g for g in qry_genes if g in lib_idx]         # genes perturbed in BOTH cell lines (the testable set)
    if restrict:
        both = [g for g in both if g in restrict]
    ranks = []
    for g in both:
        qi = qry_genes.index(g)
        sims = Ln @ Qn[qi]                                # similarity to every library perturbation
        order = np.argsort(-sims)
        truth = lib_idx[g]
        rank = int(np.where(order == truth)[0][0]) + 1
        ranks.append(rank)
    res = _rank_recall(ranks)
    res.update(method=method, n_library=len(lib_genes), n_shared_genes=len(shared),
               genes_kept=int(Ml.shape[1]), tested_genes=len(both), random_top10=round(10 / len(lib_genes), 4))
    return res, ranks, both


def run():
    lib = f"{SCRATCH}/k562.h5ad"; qry = f"{SCRATCH}/rpe1.h5ad"
    out = {"benchmark": "cross-cell-line perturbation identity recovery (RPE1 signature -> rank K562 library)",
           "data": "Replogle 2022 pseudobulk (K562_essential + rpe1)"}
    for m in ("cosine", "zcosine", "spearman"):
        res, ranks, _ = cross_line_topk(lib, qry, method=m)
        out[m] = res
        print(f"[{m:8}] top1={res['top_1']}  top5={res['top_5']}  top10={res['top_10']}  "
              f"top100={res['top_100']}  median_rank={res['median_rank']}  (of {res['n_library']}; "
              f"random top10={res['random_top10']})")
    json.dump(out, open(f"{OUT}/perturb_prioritizer.json", "w"), indent=2)
    print(f"-> {OUT}/perturb_prioritizer.json")
    return out


if __name__ == "__main__":
    run()
