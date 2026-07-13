"""pathway_decode — can the software DECODE pathway membership from measured data, instead of reading a curated
label? Genes in the same pathway are broken together, so they should have similar functional fingerprints. This
tests exactly that: predict a gene's pathway from the DATA (DepMap co-dependency across 1,150 cell lines + the
Perturb-seq signature + PPI + complex co-membership), validate blind on genes whose pathway we DO know, then use it
to assign pathways to the currently-UNANNOTATED genes.

"Give it all the help": the neighbor score fuses every functional signal we have —
    score(g,h) = codependency_cosine  + 0.5·perturbseq_cosine  + 0.3·[shares a PPI]  + 0.5·[shares a complex]
k-NN label transfer: a held-out gene inherits the pathways its nearest functional neighbors carry. Honest metric:
top-1 / top-3 accuracy (predicted pathway is a REAL pathway of the gene) vs a most-common-pathway baseline, with an
ablation so each signal's contribution is visible. Then: how many unannotated genes get a confident pathway call.

-> outputs/orphan/pathway_decode.json  (+ a sample of newly-assigned pathways). Honest null reported if it fails.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def main(sample=1500, k=15, seed=0):
    from complete_cell import CompleteCell
    import cellos
    C = CompleteCell()
    z = np.load(f"{OUT}/depmap_vecs.npz", allow_pickle=True)
    syms = [str(s) for s in z["syms"]]
    Z = np.nan_to_num(z["Z"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    idx = {s: i for i, s in enumerate(syms)}
    norms = np.linalg.norm(Z, axis=1)
    valid = norms > 1e-6
    Zn = Z.copy(); Zn[valid] /= norms[valid][:, None]                # L2-normalized rows → dot = cosine

    def paths(name):
        i = C.idx.get(name)
        if i is None:
            return []
        p = C.genes[i].get("path") or []
        return [str(x) for x in ([p] if isinstance(p, str) else p)]

    labels = {s: set(paths(s)) for s in syms if valid[idx[s]] and paths(s)}
    labeled = [s for s in syms if s in labels]
    lab_idx = np.array([idx[s] for s in labeled])
    print(f"pathway decoding: {len(labeled):,} labeled+valid genes, {len(syms):,} with a co-dependency vector")

    # deep booster: Perturb-seq signatures (normalized)
    dbg = cellos.CellKernel(quiet=True)._load_debugger()
    P = {}
    if dbg:
        M = dbg["M"].astype(np.float32); pn = dbg["norms"];
        P = {g: M[dbg["pidx"][g]] / (pn[dbg["pidx"][g]] + 1e-9) for g in dbg["pgenes"]}

    ppi = {C.name[i]: {C.name[j] for j in C.ppi_adj.get(i, [])} for i in range(len(C.genes))}
    g2c = C.D.get("gene2cplx", {}) or {}
    cplx = {}
    for gi, cs in g2c.items():
        try:
            nm = C.name[int(gi)]
        except (ValueError, IndexError):
            continue
        cplx[nm] = set(cs)

    rng = np.random.RandomState(seed)
    test = list(rng.choice(labeled, min(sample, len(labeled)), replace=False))
    test_idx = np.array([idx[g] for g in test])
    # precompute cosine of every test gene to every labeled gene, once
    S = Zn[lab_idx] @ Zn[test_idx].T                                 # (n_labeled, n_test)

    def predict(col, g, mode):
        sims = S[:, col]
        top = np.argpartition(-sims, min(60, len(sims) - 1))[:60]
        top = top[np.argsort(-sims[top])]
        votes, cnt = {}, 0
        for j in top:
            h = labeled[j]
            if h == g or sims[j] <= 0:
                continue
            w = float(sims[j])
            if mode >= 1 and g in P and h in P:
                w += 0.5 * float(np.dot(P[g], P[h]))
            if mode >= 2:
                if h in ppi.get(g, ()):
                    w += 0.3
                if cplx.get(g) and cplx.get(h) and (cplx[g] & cplx[h]):
                    w += 0.5
            for pw in labels[h]:
                votes[pw] = votes.get(pw, 0.0) + w
            cnt += 1
            if cnt >= k:
                break
        return [p for p, _ in sorted(votes.items(), key=lambda x: -x[1])]

    print(f"\n  {'signal':<32}{'top-1':>8}{'top-3':>8}")
    print("  " + "-" * 48)
    res = {}
    for mode, name in [(0, "co-dependency only"), (1, "+ Perturb-seq signature"), (2, "+ PPI + complex (all help)")]:
        t1 = t3 = 0
        for col, g in enumerate(test):
            pred = predict(col, g, mode)
            tru = labels[g]
            t1 += int(bool(pred) and pred[0] in tru)
            t3 += int(any(p in tru for p in pred[:3]))
        res[name] = {"top1": t1 / len(test), "top3": t3 / len(test)}
        print(f"  {name:<32}{t1/len(test):>7.1%}{t3/len(test):>8.1%}")

    pc = Counter(p for s in labeled for p in labels[s])
    top_pw, top_n = pc.most_common(1)[0]
    base = sum(1 for g in test if top_pw in labels[g]) / len(test)
    print("  " + "-" * 48)
    print(f"  {'baseline (predict most-common)':<32}{base:>7.1%}")
    best = res["+ PPI + complex (all help)"]["top1"]
    print(f"\n  → the software decodes a gene's pathway from data at {best:.0%} top-1 "
          f"({best/max(base,1e-9):.0f}× the {base:.0%} baseline)")

    # ---- fill the unannotated genes ----
    unann = [s for s in syms if valid[idx[s]] and s not in labels]
    fill_sample = list(rng.choice(unann, min(2000, len(unann)), replace=False))
    fi = np.array([idx[g] for g in fill_sample])
    Sf = Zn[lab_idx] @ Zn[fi].T
    assigned, examples = 0, []
    for col, g in enumerate(fill_sample):
        sims = Sf[:, col]
        top = np.argpartition(-sims, 60)[:60]; top = top[np.argsort(-sims[top])]
        votes = {}
        for j in top[:k]:
            h = labeled[j]
            if sims[j] <= 0:
                continue
            w = float(sims[j])
            if g in P and h in P:
                w += 0.5 * float(np.dot(P[g], P[h]))
            for pw in labels[h]:
                votes[pw] = votes.get(pw, 0.0) + w
        if votes:
            ranked = sorted(votes.items(), key=lambda x: -x[1])
            margin = ranked[0][1] / (sum(v for _, v in ranked) + 1e-9)
            if margin > 0.25 and ranked[0][1] > 2.0:                # confident call
                assigned += 1
                if len(examples) < 12:
                    examples.append({"gene": g, "pathway": ranked[0][0][:60], "confidence": round(margin, 2)})
    frac = assigned / len(fill_sample)
    est_total = int(frac * len(unann))
    print(f"\n  UNANNOTATED genes: confident pathway call for {frac:.0%} of a {len(fill_sample)}-gene sample "
          f"→ ~{est_total:,} of {len(unann):,} fillable genes")
    for e in examples[:8]:
        print(f"    {e['gene']:<10} → {e['pathway']}  (conf {e['confidence']})")

    R = {"n_labeled": len(labeled), "ablation": res, "baseline_top1": base, "k": k, "sample": len(test),
         "unannotated_fillable": len(unann), "confident_fill_frac": frac, "confident_fill_est": est_total,
         "examples": examples}
    json.dump(R, open(f"{OUT}/pathway_decode.json", "w"), indent=1)
    return R


if __name__ == "__main__":
    main()
