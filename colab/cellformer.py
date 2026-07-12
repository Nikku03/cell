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
    # gene 'token' embeddings: truncated SVD of the standardised train matrix -> gene loadings (genes x d)
    mu = Mtr.mean(0); sd = Mtr.std(0); sd[sd < 1e-8] = 1.0
    Z = np.nan_to_num((Mtr - mu) / sd)
    from sklearn.decomposition import TruncatedSVD
    emb = TruncatedSVD(n_components=d, random_state=seed).fit(Z).components_.T   # genes x d
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


def build_context_index(C, screen_set):
    """precompute the maps context_weights needs, ONCE (so complex-partner lookup is O(members), not O(genome^2)).
    Returns (g2c, coexpr, cx2members) where cx2members[cx_id] = set of SCREEN gene names in that complex.
    Merges hu.MAP 2.0 complexes (outputs/orphan/humap_complexes.json) if present — the coverage-boosting overlay."""
    # NOTE: hu.MAP 2.0 physical complexes were TESTED as a coverage booster and REJECTED — they predict knockout
    # effects at only r~0.09 (vs curated r=0.23), even at confidence 5, because physical co-purification != a
    # functional unit whose members share a knockout effect. Counting them inflated 'trustworthy' 55%->67% with
    # r~0.06 genes. So the predict context uses CURATED complexes only (see epistasis/humap note in UPDATES).
    g2c = {int(x): set(v) for x, v in (C.D.get("gene2cplx", {}) or {}).items()}
    coexpr = {}
    for kk, lst in (C.D.get("coexpr", {}) or {}).items():
        coexpr[int(kk)] = {int(p[0]): float(p[1]) for p in lst if isinstance(p, (list, tuple)) and len(p) >= 2}
    cx2members = {}
    for gi, cxs in g2c.items():
        nm = C.name[gi] if 0 <= gi < len(C.name) else None
        if nm in screen_set:
            for c in cxs:
                cx2members.setdefault(c, set()).add(nm)
    return g2c, coexpr, cx2members


def context_weights(C, gene, screen_set, g2c=None, coexpr=None, cx2members=None, screen_genes=None):
    """context of an UNSEEN knockout = its network neighbours present in the screen, weighted by structural
    evidence (co-expression 2x, PPI 1x, same-complex 3x). Shared by predict_next() and CellOS 'predict'."""
    gi = C.idx.get(gene)
    if gi is None:
        return {}
    if g2c is None or coexpr is None:
        g2c, coexpr, cx2members = build_context_index(C, screen_set)
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
    if my_cx and cx2members is not None:
        for c in my_cx:
            for nm in cx2members.get(c, ()):
                if nm != gene:
                    w[nm] = w.get(nm, 0) + 3.0
    return w


def predict_next(M, pgenes, seed=0, k_ctx=25):
    """PREDICT-NEXT task: hold out each perturbation and predict its FULL response from its NETWORK neighbours'
    responses (leave-one-out). Context weights come from the static cell model (PPI / same-complex / co-expression)
    — correlational structure — testing whether it can predict an UNSEEN interventional effect."""
    from complete_cell import CompleteCell
    C = CompleteCell()
    pidx = {g: i for i, g in enumerate(pgenes)}
    screen_set = set(pgenes)
    g2c, coexpr, cx2members = build_context_index(C, screen_set)
    mu = M.mean(0)
    rows_pred, rows_true, rows_base, meta = [], [], [], []
    for g in pgenes:
        gi = C.idx.get(g)
        if gi is None:
            continue
        w = context_weights(C, g, screen_set, g2c, coexpr, cx2members)
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


def coverage(pgenes, C=None):
    """genome COMPLETENESS: of all genes, how many can CellOS answer for = MEASURED (in the screen -> strace/
    whodunit) + PREDICTABLE (not measured, but has a complex partner or >=3 network neighbours in the screen ->
    predict). The 'complete cell as software' number."""
    if C is None:
        from complete_cell import CompleteCell
        C = CompleteCell()
    screen = set(pgenes)
    g2c, coexpr, cx2m = build_context_index(C, screen)
    n = len(C.name)
    measured = predict_good = predict_weak = dark = 0
    for gi, nm in enumerate(C.name):
        if nm in screen:
            measured += 1; continue
        has_cx = any(cx2m.get(c) for c in g2c.get(gi, set()))       # complex partner in screen -> r~0.23
        w = context_weights(C, nm, screen, g2c, coexpr, cx2m)
        if has_cx:
            predict_good += 1
        elif len(w) >= 3:
            predict_weak += 1                                       # neighbours but no module -> r~0.05
        else:
            dark += 1
    # coverage_frac = ANSWERS-for (measured+any context); well_covered = trustworthy (measured + complex-predictable)
    return {"genome": n, "measured_debugger": measured,
            "predictable_complex_r0.23": predict_good, "predictable_weak_r0.05": predict_weak,
            "dark_no_context": dark,
            "answers_for_frac": round((measured + predict_good + predict_weak) / n, 3),
            "WELL_covered_frac": round((measured + predict_good) / n, 3)}


def run(path=None):
    path = path or next((p for p in [f"{SCRATCH}/gwps.h5ad", f"{SCRATCH}/k562.h5ad"] if os.path.exists(p)),
                        f"{SCRATCH}/k562.h5ad")
    M, pert, syms = load_pseudobulk(path)
    M, pgenes = _dedup_rows(M, pert)
    M = np.nan_to_num(M.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)   # gwps has NaN AND inf entries
    M = np.clip(M, -20.0, 20.0)                             # guard against overflow in mean/std / SVD
    print(f"corpus: {os.path.basename(path)} -> {M.shape[0]} perturbations x {M.shape[1]} genes", flush=True)
    imp = impute(M); print(f"  impute done: r={imp['pearson_r']}", flush=True)
    nxt = predict_next(M, pgenes); print(f"  predict-next done: r={nxt['mean_r_predicted']}", flush=True)
    cov = coverage(pgenes); print(f"  coverage done: trustworthy {cov['WELL_covered_frac']:.0%}", flush=True)
    res = {"corpus": os.path.basename(path), "impute": imp, "predict_next": nxt, "completeness": cov}
    res["verdict"] = (
        f"IMPUTE r={imp['pearson_r']} vs {imp['baseline_mean_r']}; PREDICT-NEXT r={nxt['mean_r_predicted']} "
        f"(complex {nxt['mean_r_genes_WITH_complex']} / singleton {nxt['mean_r_genes_WITHOUT_complex']}). "
        f"COVERAGE vs ACCURACY: CellOS ANSWERS for {cov['answers_for_frac']:.0%} of the genome, but only "
        f"{cov['WELL_covered_frac']:.0%} is TRUSTWORTHY ({cov['measured_debugger']:,} measured + "
        f"{cov['predictable_complex_r0.23']:,} complex-predictable); {cov['predictable_weak_r0.05']:,} are weak "
        f"(r~0.05) and {cov['dark_no_context']:,} dark. 'Complete' in coverage, honest about accuracy.")
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
    print(f"  COMPLETENESS (coverage != accuracy):")
    print(f"     {cov['measured_debugger']:,} MEASURED (debugger, high quality)")
    print(f"     {cov['predictable_complex_r0.23']:,} PREDICTABLE via a complex (r~0.23)")
    print(f"     {cov['predictable_weak_r0.05']:,} weak context only (r~0.05)  |  {cov['dark_no_context']:,} dark")
    print(f"     -> ANSWERS for {cov['answers_for_frac']:.0%};  TRUSTWORTHY for {cov['WELL_covered_frac']:.0%}")
    print("=" * 78)
    return res


if __name__ == "__main__":
    run()
