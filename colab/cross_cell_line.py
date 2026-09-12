"""cross_cell_line — the honest test behind "get other cell lines". We have two: K562 (erythroleukemia) and RPE1
(retinal epithelium). RPE1 adds 0 NEW genes (its screen is essential-scoped, a subset of K562). So the value of a
second cell line for the SHARED genes is not coverage — it's the answer to: does a knockout's transcriptional
response TRANSFER across cell types, or is it context-specific?

  * if responses TRANSFER (high cross-line correlation) → the K562 measurements generalize; the 55% we have is
    portable to other contexts, and other lines mainly matter for the genes K562 can't express.
  * if responses are CONTEXT-SPECIFIC (low correlation) → coverage is even narrower than 55% (it's 55% *in K562*),
    and other cell lines are essential not just for new genes but to re-measure the ones we have.

Method: for each knockout measured in BOTH lines, correlate its K562 response vector with its RPE1 response vector
over the shared response genes. Compare to a shuffled baseline (a KO's K562 response vs a RANDOM RPE1 knockout) to
prove any transfer is real, not a global artifact. -> outputs/orphan/cross_cell_line.json
"""
import os, sys, json
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")
SCRATCH = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"


def load_screen(path):
    """return dict gene->response vector (mean over that KO's cells) and the response-gene name list."""
    f = h5py.File(path, "r")
    X = f["X"][:]
    idxkey = f["obs"].attrs.get("_index", "_index")
    labels = [x.decode() if isinstance(x, bytes) else x for x in f["obs"][idxkey][:]]
    pert = [str(l).split("_")[1] if len(str(l).split("_")) > 1 else str(l) for l in labels]
    gn = f["var"]["gene_name"]
    if gn.dtype.kind in ("i", "u"):
        cats = [x.decode() if isinstance(x, bytes) else x for x in f["var"]["__categories"]["gene_name"][:]]
        syms = [cats[c] for c in gn[:]]
    else:
        syms = [x.decode() if isinstance(x, bytes) else x for x in gn[:]]
    f.close()
    from collections import defaultdict
    rows = defaultdict(list)
    for k, g in enumerate(pert):
        rows[g].append(k)
    resp = {g: np.clip(np.nan_to_num(X[idx].mean(0), nan=0, posinf=0, neginf=0), -20, 20)
            for g, idx in rows.items()}
    return resp, syms


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def main():
    k_resp, k_syms = load_screen(f"{SCRATCH}/k562.h5ad")
    r_resp, r_syms = load_screen(f"{SCRATCH}/rpe1.h5ad")

    # shared response-gene space (align columns)
    kidx = {s: i for i, s in enumerate(k_syms)}
    ridx = {s: i for i, s in enumerate(r_syms)}
    shared_genes = [s for s in k_syms if s in ridx]
    kcol = np.array([kidx[s] for s in shared_genes]); rcol = np.array([ridx[s] for s in shared_genes])

    shared_ko = sorted(set(k_resp) & set(r_resp))
    print(f"cross-cell-line transfer — K562 vs RPE1")
    print(f"  knockouts measured in BOTH lines: {len(shared_ko)}   shared response genes: {len(shared_genes)}\n")

    rng = np.random.RandomState(0)
    real, shuf = [], []
    for g in shared_ko:
        kv = k_resp[g][kcol]; rv = r_resp[g][rcol]
        real.append(pearson(kv, rv))
        gr = shared_ko[rng.randint(len(shared_ko))]                 # a random OTHER knockout in RPE1
        shuf.append(pearson(kv, r_resp[gr][rcol]))
    real, shuf = np.array(real), np.array(shuf)
    frac_transfer = float((real > 0.2).mean())
    R = {"n_shared_ko": len(shared_ko), "n_shared_genes": len(shared_genes),
         "mean_cross_line_r": float(real.mean()), "median_cross_line_r": float(np.median(real)),
         "mean_shuffled_r": float(shuf.mean()), "frac_ko_with_r>0.2": frac_transfer,
         "new_genes_from_rpe1": 0}

    print(f"  a knockout's response, K562 → RPE1 (same gene):   mean r = {real.mean():+.3f}  (median {np.median(real):+.3f})")
    print(f"  shuffled baseline (K562 gene vs RANDOM RPE1 gene): mean r = {shuf.mean():+.3f}")
    print(f"  fraction of knockouts with cross-line r > 0.2:     {frac_transfer:.0%}")

    if real.mean() > shuf.mean() + 0.1 and real.mean() > 0.25:
        verdict = (f"RESPONSES TRANSFER: a knockout's effect in K562 predicts its effect in RPE1 (r={real.mean():.2f} "
                   f"vs shuffled {shuf.mean():.2f}). The measured signatures GENERALIZE across cell types, so the 55% "
                   f"we have is portable — other cell lines are needed mainly for the genes K562 can't EXPRESS, not "
                   f"to re-measure shared ones. RPE1 adds 0 new genes (essential-scoped); the lever is a GENOME-WIDE "
                   f"screen in a complementary cell type (now emerging: primary T-cell / myeloid Perturb-seq, 2025).")
    elif real.mean() > shuf.mean() + 0.05:
        verdict = (f"PARTIAL TRANSFER: responses are correlated across lines (r={real.mean():.2f} vs {shuf.mean():.2f}) "
                   f"but not strongly — knockout effects are partly context-specific, so other cell lines matter both "
                   f"for new genes AND to re-measure shared ones in a new context.")
    else:
        verdict = (f"CONTEXT-SPECIFIC: knockout responses do NOT transfer across cell lines (r={real.mean():.2f} ≈ "
                   f"shuffled {shuf.mean():.2f}). Coverage is 55% *in K562 only*; every cell type needs its own screen.")
    R["verdict"] = verdict
    print("\n" + "=" * 76 + f"\nVERDICT: {verdict}\n" + "=" * 76)
    json.dump(R, open(f"{OUT}/cross_cell_line.json", "w"), indent=1)
    return R


if __name__ == "__main__":
    main()
