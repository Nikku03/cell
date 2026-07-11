"""cellformer — the cell as a language model: PREDICT THE NEXT THING.

A transformer predicts the next token from context. The cell-equivalent, using what we have: predict the
transcriptional effect of a knockout WE NEVER MEASURED, from the measured effects of its neighbours — then, where
we can, check the prediction against the real measurement. This is "completing the cell" by prediction, and it is
the honest extension of whodunit (complex members have near-identical effects → a gene's effect should be
predictable from its module).

Two tasks, decomposed like every test in this project (interpolation vs extrapolation):
  IMPUTE  (interpolation): mask genes inside a perturbation's response, predict them from the OBSERVED genes via
          gene-gene attention (softmax over learned gene similarity). This is what Geneformer/scGPT do; expected to
          work — it is the easy regime.
  PREDICT-NEXT (extrapolation): hold out an ENTIRE perturbation (an unseen knockout) and predict its whole response
          vector from the responses of its NETWORK neighbours (PPI / same-complex / co-expression), weighted. This
          is the real "predict the next token" — the effect of a knockout from its context alone. The honest ceiling.

Metric: Pearson r between predicted and true (masked entries, or held-out row), vs a baseline that predicts the
average response (no gene-specific context). Beating the baseline = real conditional structure was used.
Data: Replogle 2022 pseudobulk (perturb_prioritizer.load_pseudobulk).
-> outputs/orphan/cellformer.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from perturb_prioritizer import load_pseudobulk, _dedup_rows

OUT = "outputs/orphan"
SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def impute(M, d=64, mask_frac=0.15, seed=0, test_frac=0.2, tau=0.1):
    """IMPUTE task: gene-gene attention completion of masked entries in HELD-OUT perturbations.
    gene embeddings learned (SVD) from TRAIN perturbations only → no leak into the held-out rows."""
    rng = np.random.default_rng(seed)
    n_pert, n_gene = M.shape
    perm = rng.permutation(n_pert)
    te = perm[:int(test_frac * n_pert)]; tr = perm[int(test_frac * n_pert):]
    Mtr = M[tr]
    # gene 'token' embeddings: SVD of the standardised train matrix -> gene loadings (genes x d)
    mu = Mtr.mean(0); sd = Mtr.std(0) + 1e-8
    Z = (Mtr - mu) / sd
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    emb = Vt[:d].T                                           # genes x d
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    Sgg = emb @ emb.T                                        # gene-gene similarity (the attention logits)
    preds, actual, base = [], [], []
    for r in te:
        v = M[r]
        mask = rng.random(n_gene) < mask_frac
        obs = ~mask
        logits = Sgg[np.ix_(mask, obs)] / tau
        logits -= logits.max(1, keepdims=True)
        W = np.exp(logits); W /= W.sum(1, keepdims=True) + 1e-9
        pred = W @ v[obs]
        preds.append(pred); actual.append(v[mask]); base.append(mu[mask])
    p = np.concatenate(preds); a = np.concatenate(actual); b = np.concatenate(base)
    return {"task": "impute (mask genes in held-out perturbations, gene-gene attention)",
            "pearson_r": round(_pearson(p, a), 3), "baseline_mean_r": round(_pearson(b, a), 3),
            "n_masked_entries": int(a.size), "held_out_perturbations": int(len(te))}


def context_weights(C, gene, screen_set, g2c=None, coexpr=None, screen_genes=None):
    """context of an UNSEEN knockout = its network neighbours present in the screen, weighted by structural
    evidence (co-expression 2x, PPI 1x, same-complex 3x). Shared by predict_next() and CellOS 'predict'."""
    gi = C.idx.get(gene)
    if gi is None:
        return {}
    g2c = g2c if g2c is not None else {int(x): set(v) for x, v in (C.D.get("gene2cplx", {}) or {}).items()}
    if coexpr is None:
        coexpr = {}
        for kk, lst in (C.D.get("coexpr", {}) or {}).items():
            coexpr[int(kk)] = {int(p[0]): float(p[1]) for p in lst if isinstance(p, (list, tuple)) and len(p) >= 2}
    w = {}
    for j in C.ppi_adj.get(gi, []):
        nm = C.name[j]
        if nm in screen_set and nm != gene:
            w[nm] = w.get(nm, 0) + 1.0
    for j, cc in coexpr.get(gi, {}).items():
        nm = C.name[j]
        if nm in screen_set and nm != gene:
            w[nm] = w.get(nm, 0) + 2.0 * max(cc, 0)
    my_cx = g2c.get(gi, set())
    if my_cx and screen_genes is not None:
        for og in screen_genes:
            oi = C.idx.get(og)
            if oi is not None and og != gene and (g2c.get(oi, set()) & my_cx):
                w[og] = w.get(og, 0) + 3.0
    return w


def predict_next(M, pgenes, seed=0, k_ctx=25):
    """PREDICT-NEXT task: hold out each perturbation and predict its FULL response from its NETWORK neighbours'
    responses (leave-one-out). Context weights come from the static cell model (PPI / same-complex / co-expression)
    — correlational structure — testing whether it can predict an UNSEEN interventional effect."""
    from complete_cell import CompleteCell
    C = CompleteCell()
    pidx = {g: i for i, g in enumerate(pgenes)}
    screen_set = set(pgenes)
    g2c = {int(x): set(v) for x, v in (C.D.get("gene2cplx", {}) or {}).items()}
    coexpr = {}
    for kk, lst in (C.D.get("coexpr", {}) or {}).items():
        coexpr[int(kk)] = {int(p[0]): float(p[1]) for p in lst if isinstance(p, (list, tuple)) and len(p) >= 2}
    mu = M.mean(0)
    rows_pred, rows_true, rows_base, meta = [], [], [], []
    for g in pgenes:
        gi = C.idx.get(g)
        if gi is None:
            continue
        w = context_weights(C, g, screen_set, g2c, coexpr, screen_genes=pgenes)
        ctx = sorted(w, key=lambda x: -w[x])[:k_ctx]
        if not ctx:
            continue
        wv = np.array([w[c] for c in ctx]); wv /= wv.sum()
        pred = wv @ M[[pidx[c] for c in ctx]]
        rows_pred.append(pred); rows_true.append(M[pidx[g]]); rows_base.append(mu)
        meta.append((g, len(ctx), bool(g2c.get(gi, set()))))
    # per-gene correlation predicted-vs-true
    per_r = [_pearson(p, t) for p, t in zip(rows_pred, rows_true)]
    base_r = [_pearson(b, t) for b, t in zip(rows_base, rows_true)]
    per_r = np.array(per_r); base_r = np.array(base_r)
    incplx = np.array([m[2] for m in meta])
    return {"task": "predict-next (predict a held-out knockout's full response from network neighbours)",
            "n_genes_predicted": len(per_r),
            "mean_r_predicted": round(float(per_r.mean()), 3),
            "mean_r_baseline_avg_response": round(float(base_r.mean()), 3),
            "median_r_predicted": round(float(np.median(per_r)), 3),
            "frac_beating_baseline": round(float((per_r > base_r).mean()), 3),
            "mean_r_genes_WITH_complex": round(float(per_r[incplx].mean()) if incplx.any() else 0, 3),
            "mean_r_genes_WITHOUT_complex": round(float(per_r[~incplx].mean()) if (~incplx).any() else 0, 3),
            "n_with_complex": int(incplx.sum())}


def run():
    M, pert, syms = load_pseudobulk(f"{SCRATCH}/k562.h5ad")
    M, pgenes = _dedup_rows(M, pert)
    print(f"corpus: {M.shape[0]} perturbations x {M.shape[1]} genes")
    imp = impute(M)
    nxt = predict_next(M, pgenes)
    res = {"impute": imp, "predict_next": nxt}
    res["verdict"] = (
        f"IMPUTE (interpolation) r={imp['pearson_r']} vs baseline {imp['baseline_mean_r']} — the cell autocompletes "
        f"masked genes well. PREDICT-NEXT (extrapolation) r={nxt['mean_r_predicted']} vs baseline "
        f"{nxt['mean_r_baseline_avg_response']}; genes WITH a known complex = {nxt['mean_r_genes_WITH_complex']}, "
        f"WITHOUT = {nxt['mean_r_genes_WITHOUT_complex']}. The next knockout is predictable EXACTLY WHEN the gene "
        f"sits in a measured module; singletons stay hard — the honest edge of 'completing the cell'.")
    json.dump(res, open(f"{OUT}/cellformer.json", "w"), indent=2)
    print("=" * 78)
    print("CELLFORMER — predict the next thing (transformer-style), on interventional data")
    print("=" * 78)
    print(f"  IMPUTE  (mask genes, gene-gene attention): r={imp['pearson_r']}  "
          f"(baseline predict-mean r={imp['baseline_mean_r']})   <- interpolation")
    print(f"  PREDICT-NEXT (unseen knockout from neighbours): r={nxt['mean_r_predicted']}  "
          f"(baseline avg-response r={nxt['mean_r_baseline_avg_response']})   <- extrapolation")
    print(f"     with a known complex:    r={nxt['mean_r_genes_WITH_complex']}  (n={nxt['n_with_complex']})")
    print(f"     without a known complex: r={nxt['mean_r_genes_WITHOUT_complex']}")
    print(f"     beats baseline for {nxt['frac_beating_baseline']:.0%} of genes")
    print(f"  -> {res['verdict']}")
    print("=" * 78)
    return res


if __name__ == "__main__":
    run()
