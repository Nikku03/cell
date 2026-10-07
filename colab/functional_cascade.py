"""FUNCTIONAL CASCADE: knockout -> TF targets -> what those proteins DO -> genes that do the same -> repeat.

A DIFFERENT HYPOTHESIS FROM THE ROUTING TEST, and worth separating from it. route_map.py propagated along
graph EDGES and found the real network indistinguishable from a degree-preserving rewiring beyond one hop.
This propagates along shared FUNCTION, which is not an edge at all: two genes with no interaction can be in
the same complex, the same pathway, or carry the same GO process. If a knockout perturbs a job rather than a
partner, edge-following would miss it entirely and the previous negative would be the wrong conclusion.

THE CHAIN, as specified:

    KO gene  --TF edges-->  its regulatory targets
             --GO / pathway / complex-->  genes doing the same job
             --repeat-->  second-order spread

TWO CONFOUNDS DECIDE WHETHER THIS MEANS ANYTHING, and both get an explicit control.

  1 ANNOTATION SIZE. "Protein binding" holds thousands of genes; sharing it says nothing. Every functional
    edge is therefore weighted 1/|term| -- inverse term size -- so a 5-gene complex counts far more than a
    5,000-gene GO bucket. Without this the cascade simply returns the largest annotation categories.

  2 BIBLIOMETRIC BIAS, which is the serious one. Well-studied genes carry more GO terms AND are more often
    reported as movers. A cascade over real annotation would preferentially reach well-annotated genes and
    look predictive for a reason that is about publication history, not biology. The control is a
    DEGREE-PRESERVING SHUFFLE of the gene-to-term assignment: every gene keeps its exact number of terms and
    every term keeps its exact number of genes, but which gene has which term is randomised. If the real
    annotation scores no better than that shuffle, the cascade is measuring how well studied a gene is.

That control is the analogue of the rewiring in route_map.py, and it is the arm that decides this test.
"""
import collections
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAU, TIDE_FRAC, MIN_SPEC, NPICK = 1.0, 0.05, 5, 50
RNG = np.random.default_rng(0)
R = {}


def hdr(s):
    print("\n" + "=" * 104)
    print(s)
    print("=" * 104)


def bci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    return tuple(float(v) for v in np.percentile(
        r.choice(x, size=(n, len(x)), replace=True).mean(1), [2.5, 97.5]))


# --------------------------------------------------------------- annotation -> sparse incidence
def incidence(groups, gi, n, min_sz=2, max_sz=2000):
    """gene x term matrix, entries 1/|term| so a small specific category outweighs a huge generic one."""
    rows, cols, vals = [], [], []
    t = 0
    sizes = []
    for members in groups:
        m = sorted({gi[x] for x in members if x in gi})
        if not (min_sz <= len(m) <= max_sz):
            continue
        w = 1.0 / len(m)
        for g in m:
            rows.append(g)
            cols.append(t)
            vals.append(w)
        sizes.append(len(m))
        t += 1
    if not t:
        return sparse.csr_matrix((n, 0)), []
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, t)), sizes


def shuffle_incidence(A, rng):
    """Randomise WHICH gene holds which term, preserving both marginals exactly.

    Shuffling the row indices within the sparse structure keeps every term's member count and every gene's
    term count intact -- so the shuffled annotation has the identical size distribution, and any advantage the
    real annotation shows cannot be a size or study-depth artifact.
    """
    A = A.tocoo()
    order = np.arange(A.shape[0])
    rng.shuffle(order)
    return sparse.csr_matrix((A.data, (order[A.row], A.col)), shape=A.shape)


def main():
    import cellformer_af as CF
    d = CF.load_all()
    genes, kos, gi = d["genes"], d["kos"], d["gi"]
    spec, tide, nspec = d["spec"], d["tide"], d["nspec"]
    n = len(genes)
    tr = np.array([int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 for k in kos])
    test = [q for q in range(len(kos)) if not tr[q] and nspec[q] >= MIN_SPEC and kos[q] in gi]
    freq = spec[tr].mean(0)
    print(f"\nheld-out scorable with KO gene in the readout: {len(test):,}")

    C = json.load(gzip.open(Path(__file__).resolve().parent / "data" / "cell_complete.json.gz", "rt"))
    names = [g["name"] for g in C["genes"]]

    # ---- functional groupings ----
    go = C.get("go") or {}
    byterm = collections.defaultdict(list)
    for g, dd in go.items():
        if not isinstance(dd, dict):
            continue
        for br in ("P", "F", "C"):
            for t in (dd.get(br) or []):
                byterm[(br, t)].append(g)
    go_groups = list(byterm.values())
    pw_groups = [[names[i] for i in idx if 0 <= i < len(names)]
                 for idx in (C.get("pathways") or {}).values() if isinstance(idx, list)]
    cx = collections.defaultdict(list)
    for gidx, cl in (C.get("gene2cplx") or {}).items():
        try:
            nm = names[int(gidx)]
        except (ValueError, IndexError):
            continue
        for c in (cl if isinstance(cl, list) else []):
            cx[c].append(nm)
    cx_groups = list(cx.values())

    A_go, s_go = incidence(go_groups, gi, n)
    A_pw, s_pw = incidence(pw_groups, gi, n)
    A_cx, s_cx = incidence(cx_groups, gi, n)
    print(f"GO terms {A_go.shape[1]:,} (median size {np.median(s_go) if s_go else 0:.0f}), "
          f"pathways {A_pw.shape[1]:,}, complexes {A_cx.shape[1]:,}")

    # ---- TF network, DIRECTED: rows = TF, cols = target ----
    rows, cols = [], []

    def add_tf(tf, tgt):
        if tf in gi and tgt in gi and tf != tgt:
            rows.append(gi[tf])
            cols.append(gi[tgt])

    for f in ("trrust_regulon.json", "chip_reg_edges.json"):
        p = OUT / f
        if p.exists():
            for tf, tg in (json.load(open(p)).get("tf_targets") or {}).items():
                for t in (tg if isinstance(tg, (list, tuple)) else []):
                    add_tf(tf, t if isinstance(t, str) else (t[0] if isinstance(t, (list, tuple)) else ""))
    p = OUT / "tf_core.json"
    if p.exists():
        j = json.load(open(p))
        nm = j.get("name") or {}
        for e in (j.get("edges") or []):
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a = nm.get(str(e[0]), e[0]) if not isinstance(e[0], str) else e[0]
                b = nm.get(str(e[1]), e[1]) if not isinstance(e[1], str) else e[1]
                if isinstance(a, str) and isinstance(b, str):
                    add_tf(a, b)
    TF = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)) if rows else \
        sparse.csr_matrix((n, n))
    TF.data[:] = 1.0
    is_tf = np.asarray(TF.sum(1)).ravel() > 0
    print(f"TF network: {TF.nnz:,} directed edges, {int(is_tf.sum()):,} genes act as regulators")
    ko_is_tf = sum(1 for q in test if is_tf[gi[kos[q]]])
    print(f"  held-out knockouts whose KO gene IS a regulator: {ko_is_tf:,}/{len(test):,} "
          f"({100*ko_is_tf/len(test):.1f}%)   <- upper bound on what a TF-first chain can ever explain")

    def fn_step(x, Ago, Apw, Acx):
        """One functional hop: spread onto genes sharing a term, weighted by 1/|term|."""
        return (Ago @ (Ago.T @ x)) + (Apw @ (Apw.T @ x)) + (Acx @ (Acx.T @ x))

    def run(tag, mode, Ago, Apw, Acx):
        recs = []
        for q in test:
            k = gi[kos[q]]
            T = set(int(x) for x in np.where(spec[q])[0] if not tide[x])
            if not T:
                continue
            x = np.zeros(n)
            x[k] = 1.0
            if mode == "tf1":                       # KO -> its regulatory targets
                s = TF.T @ x
            elif mode == "fn1":                     # KO -> genes sharing its function
                s = fn_step(x, Ago, Apw, Acx)
            elif mode == "tf_fn":                   # KO -> targets -> their function
                t1 = TF.T @ x
                s = t1 + fn_step(t1, Ago, Apw, Acx)
            elif mode == "chain2":                  # full: TF + function, two rounds
                t1 = (TF.T @ x) + fn_step(x, Ago, Apw, Acx)
                t2 = (TF.T @ t1) + fn_step(t1, Ago, Apw, Acx)
                s = t1 + 0.5 * t2
            s = np.asarray(s).ravel().astype(float)
            s[k] = -1e9
            s[tide] = -1e9
            recs.append(len(set(int(i) for i in np.argsort(-s)[:NPICK]) & T) / len(T))
        recs = np.array(recs)
        lo, hi = bci(recs)
        print(f"    {tag:44s} recall {recs.mean():.4f}  95% CI [{lo:.4f},{hi:.4f}]")
        return {"tag": tag, "recall": float(recs.mean()), "ci": [lo, hi], "n": int(len(recs))}, recs

    hdr("FUNCTIONAL CASCADE vs the frequency baseline")
    fr = np.array([len(set(int(i) for i in np.argsort(-np.where(tide, -1, freq))[:NPICK])
                       & set(int(x) for x in np.where(spec[q])[0] if not tide[x]))
                   / max(len(set(int(x) for x in np.where(spec[q])[0] if not tide[x])), 1) for q in test])
    flo, fhi = bci(fr)
    print(f"    {'frequency baseline':44s} recall {fr.mean():.4f}  95% CI [{flo:.4f},{fhi:.4f}]")

    res = []
    keep = {}
    for tag, mode in (("TF targets only (1 step)", "tf1"),
                      ("function of the KO gene only", "fn1"),
                      ("KO -> TF targets -> their function", "tf_fn"),
                      ("full chain, 2 rounds (TF + function)", "chain2")):
        r, v = run(tag, mode, A_go, A_pw, A_cx)
        res.append(r)
        keep[mode] = v

    print("\n  DEGREE-PRESERVING ANNOTATION SHUFFLE -- every gene keeps its term count, every term its size:")
    rs = np.random.default_rng(11)
    Sgo, Spw, Scx = (shuffle_incidence(A_go, rs), shuffle_incidence(A_pw, rs), shuffle_incidence(A_cx, rs))
    shuf = []
    for tag, mode in (("function only [SHUFFLED]", "fn1"),
                      ("KO -> TF -> function [SHUFFLED]", "tf_fn"),
                      ("full chain, 2 rounds [SHUFFLED]", "chain2")):
        r, v = run(tag, mode, Sgo, Spw, Scx)
        shuf.append(r)
        # paired: same perturbations, real annotation minus shuffled
        real = keep[mode]
        m = min(len(real), len(v))
        dd = real[:m] - v[:m]
        lo, hi = bci(dd)
        print(f"      real MINUS shuffled: {dd.mean():+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"  {'-> real annotation helps' if lo > 0 else '-> NOT distinguishable from shuffled'}")
        r["delta_vs_real"] = float(dd.mean())
        r["delta_ci"] = [lo, hi]

    R["frequency"] = float(fr.mean())
    R["arms"] = res
    R["shuffled"] = shuf
    R["ko_is_tf_frac"] = float(ko_is_tf / len(test))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "functional_cascade.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'functional_cascade.json'}")


if __name__ == "__main__":
    main()
