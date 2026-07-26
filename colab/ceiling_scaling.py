"""WHAT IS THE CEILING, AND IS IT METHOD OR DATA? A neighbour-pool learning curve.

THE MOTIVATING OBSERVATION. Across this whole session, method changes did essentially nothing:

    fixing the PPI graph (34,440 real edges recovered)   ensemble  0.4720 -> 0.4709
    six different Markov state spaces                    all indistinguishable from the node walk
    deleting the 56% of proteins K562 does not express   0.4809 -> 0.4792
    a kitchen sink of 55 features vs 17                  0.4629 -> 0.4629
    Monte Carlo, three separate times                    no gain, three different failure modes

ONE CHANGE DID SOMETHING LARGE, and it was not a method. When the neighbour POOL was widened from the 378 scorable
knockouts to all 1,274 non-test knockouts, recall@50 went 0.3342 -> 0.4768. A 3.4x larger pool bought +0.14 --
roughly fourteen times the largest effect any modelling idea produced. That is a strong hint that the binding
constraint is how many knockouts are available to average, not how cleverly they are chosen.

AND THE DATA TO TEST IT IS ALREADY ON DISK. scratchpad/gwps.h5ad holds 11,258 K562 perturbations; every module in
this repo has been working from a 1,400-knockout subset. Before doing the substantial work of rebuilding the
pipeline on 11,258, the cheap question is: what does the curve look like, and does it flatten before it gets there?

WHAT THIS MEASURES. Hold the method completely fixed (the incumbent 2-step chain + learned similarity neighbour
transfer, N_NEI=10, recall@50, tide-removed specific movers) and vary ONLY the size of the pool from which the 10
neighbours may be drawn. Sub-sample the pool at 100, 200, 400, 700, 1000 and the full ~1,274, five splits each,
and fit the curve.

HOW THE EXTRAPOLATION IS DONE, AND WHY IT IS LABELLED A GUESS. Nearest-neighbour methods typically improve as a
power law in pool size with a saturating offset, so the fit is recall = c - a * n^(-b). That form is imposed, not
discovered, and extrapolating 9x beyond the largest measured point is exactly the kind of move this repo has been
sceptical of all session. The extrapolated number is therefore reported as a PREDICTION TO BE FALSIFIED by actually
running on gwps.h5ad, not as a result. What the curve CAN establish without extrapolation is the local slope: if
recall is still climbing steeply at n=1,274, more knockouts help; if it has visibly flattened, they will not, and
the ceiling is the method.
"""
import collections
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit

OUT = Path("outputs/orphan")
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
TAU, K, MIN_SPEC, TIDE_FRAC, N_NEI, NSPLIT = 1.0, 50, 5, 0.05, 10, 5
POOLS = [100, 200, 400, 700, 1000, 1274]
NREP = 3          # independent pool subsamples per (split, size)


def main():
    KO = pickle.load(open(f"{SP}/nlz_K562.pkl", "rb"))
    kos = sorted(KO); ki = {k: i for i, k in enumerate(kos)}
    genes = sorted({g for v in KO.values() for g in v}); gi = {g: i for i, g in enumerate(genes)}
    nK, nG = len(kos), len(genes)
    M = np.zeros((nK, nG), np.float32)
    for k, v in KO.items():
        for g, z in v.items():
            M[ki[k], gi[g]] = z
    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    nontide_idx = np.where(~tide)[0]
    spec_n = np.array([int(((A[i] >= TAU) & (~tide)).sum()) for i in range(nK)])

    D = json.load(open(OUT / "cell_complete.json"))
    names = [g["name"] for g in D["genes"]]
    nidx = {n: i for i, n in enumerate(names)}; N = len(names)
    pe = set()
    for e in D.get("ppi", []):
        try:
            a, b = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= a < N and 0 <= b < N and a != b:
            pe.add((min(a, b), max(a, b)))
    pe = sorted(pe)
    pa = np.array([a for a, b in pe]); pb = np.array([b for a, b in pe])
    memb = collections.defaultdict(list)
    for g, cs in (D.get("gene2cplx", {}) or {}).items():
        if g.isdigit() and int(g) < N:
            for c in cs:
                memb[c].append(int(g))
    ca, cb = [], []
    for c, mem in memb.items():
        mm = sorted(set(mem))
        if 2 <= len(mm) <= 60:
            for x in mm:
                for y in mm:
                    if x != y:
                        ca.append(x); cb.append(y)
    ca = np.array(ca); cb = np.array(cb)
    cw = 0.1                      # the optimum measured in complex_weight.py
    ra = np.r_[pa, pb, ca]; rb = np.r_[pb, pa, cb]
    w = np.r_[np.ones(2 * len(pa), np.float32), np.full(len(ca), cw, np.float32)]
    G = sparse.coo_matrix((w, (ra, rb)), shape=(N, N)).tocsr()
    mg = np.asarray(abs(G).sum(1)).ravel(); iv = np.zeros(N, np.float32)
    nz = mg > 0; iv[nz] = 1.0 / mg[nz]
    PT = (sparse.diags(iv) @ G).T.tocsr()
    konode = np.array([nidx.get(k, -1) for k in kos]); have = konode >= 0
    idx = np.where(have)[0]
    L = np.zeros((N, len(idx)), np.float32); L[konode[idx], np.arange(len(idx))] = 1.0
    cur = L; tot = np.zeros_like(L)
    for s in (1, 2):
        cur = PT @ cur; tot = tot + (0.5 ** s) * cur
    S = np.zeros((nK, nK), np.float32)
    S[np.ix_(idx, idx)] = tot[konode[idx]].T
    print(f"chain precomputed over {nK} knockouts", flush=True)

    def rec(sc, truth):
        o = nontide_idx[np.argsort(-sc[nontide_idx])][:K]
        return len(set(o.tolist()) & truth) / max(len(truth), 1)

    def prof(ix):
        ix = np.asarray(list(ix), int)
        return np.abs(M[ix]).mean(0) if len(ix) else np.zeros(nG, np.float32)

    scor = np.array([i for i in range(nK) if spec_n[i] >= MIN_SPEC])
    curve = collections.defaultdict(list)
    for sp in range(NSPLIT):
        srng = np.random.RandomState(100 + sp)
        perm = srng.permutation(scor); nt = len(scor) // 3
        test = np.array(sorted(perm[:nt].tolist())); tset = set(test.tolist())
        full_pool = np.array([i for i in range(nK) if i not in tset])
        SPEC = {int(x): set(int(t) for t in np.where(A[x] >= TAU)[0] if ~tide[t]) for x in test}
        for psz in POOLS:
            if psz > len(full_pool):
                continue
            for rep in range(NREP):
                r = np.random.RandomState(5000 + sp * 100 + rep)
                pool = full_pool if psz >= len(full_pool) else r.choice(full_pool, psz, replace=False)
                vals = []
                for xi in test:
                    p = pool[pool != xi]
                    v = S[xi].astype(np.float64).copy()
                    msk = np.full(nK, -np.inf); msk[p] = 0.0
                    v = v + msk; v[xi] = -np.inf
                    vals.append(rec(prof(np.argsort(-v)[:N_NEI]), SPEC[int(xi)]))
                curve[psz].append(float(np.mean(vals)))
                if psz >= len(full_pool):
                    break
        print(f"  split {sp} done", flush=True)

    xs = np.array(sorted(curve)); ys = np.array([np.mean(curve[p]) for p in xs])
    ses = np.array([np.std(curve[p]) / np.sqrt(len(curve[p])) for p in xs])
    print(f"\n  NEIGHBOUR-POOL LEARNING CURVE (method fixed; only pool size varies)")
    print(f"    {'pool':>6s} {'recall@50':>10s} {'+/-':>7s}")
    for x, y, e in zip(xs, ys, ses):
        print(f"    {x:6d} {y:10.4f} {e:7.4f}")

    def f(n, c, a, b):
        return c - a * np.power(n, -b)

    try:
        p0, _ = curve_fit(f, xs, ys, p0=[0.6, 1.0, 0.4], maxfev=20000)
        c_, a_, b_ = p0
        pred = {int(n): float(f(n, *p0)) for n in (2000, 5000, 11258, 20000)}
        fitok = True
    except Exception:
        c_ = a_ = b_ = float("nan"); pred = {}; fitok = False

    # local slope: how much does recall gain from the LAST doubling actually measured?
    last2 = ys[-1] - ys[-2]
    ratio = xs[-1] / xs[-2]
    print(f"\n  local slope: {xs[-2]} -> {xs[-1]} ({ratio:.2f}x) gives {last2:+.4f}")
    if fitok:
        print(f"  fit recall = {c_:.4f} - {a_:.4f} * n^-{b_:.4f}   (asymptote {c_:.4f})")
        print(f"  EXTRAPOLATION, to be falsified rather than believed:")
        for n, v in pred.items():
            print(f"    n={n:6d}  predicted recall@50 {v:.4f}")

    still_climbing = last2 > 2 * ses[-1]
    verdict = (
        f"IS THE CEILING METHOD OR DATA? Every modelling idea tried this session moved recall@50 by less than 0.01: "
        f"repairing 34,440 PPI edges, six Markov state spaces, cell-specific graph pruning, a 55-feature kitchen "
        f"sink, and three Monte Carlos. One change moved it by 0.14, and it was widening the neighbour pool from 378 "
        f"to 1,274. This module holds the method completely fixed and varies ONLY the pool size. "
        f"MEASURED CURVE: " + ", ".join(f"n={x} {y:.4f}" for x, y in zip(xs, ys)) + ". "
        + (f"IT IS STILL CLIMBING at the largest pool we have: the last step, {xs[-2]} -> {xs[-1]} knockouts, is "
           f"worth {last2:+.4f} against a standard error of {ses[-1]:.4f}. The curve has not flattened, so the "
           f"binding constraint at 1,400 knockouts is HOW MANY NEIGHBOURS ARE AVAILABLE TO AVERAGE, not how they "
           f"are chosen. " if still_climbing else
           f"IT HAS FLATTENED: the last step, {xs[-2]} -> {xs[-1]} knockouts, is worth only {last2:+.4f} against a "
           f"standard error of {ses[-1]:.4f}. More knockouts of the same kind will not break the ceiling, and the "
           f"limit is the method or the representation rather than the sample size. ")
        + (f"Fitting recall = c - a*n^-b gives an asymptote of {c_:.4f} and predicts {pred.get(11258, float('nan')):.4f} "
           f"at the 11,258 perturbations sitting unused in scratchpad/gwps.h5ad. THAT NUMBER IS A PREDICTION TO BE "
           f"FALSIFIED, NOT A RESULT: the functional form is imposed rather than discovered, and it extrapolates "
           f"roughly 9x beyond the largest measured pool, which is exactly the move this repo has spent the session "
           f"being sceptical of. The honest use of it is as a go/no-go for rebuilding the pipeline on the "
           f"genome-wide dataset. " if fitok else "")
        + f"WHAT THIS CANNOT SAY. The sub-sampled pools are drawn from the same 1,400 knockouts, so this measures "
        f"the effect of pool SIZE holding the KIND of knockout fixed; the genome-wide set is not a random 8x "
        f"enlargement of this one but a different, less curated selection, and its extra knockouts may be weaker. "
        f"The ORACLE for this architecture is about 0.61, so no amount of pool growth takes neighbour transfer past "
        f"that without a different readout. Five splits, one cell line, {NREP} subsamples per point.")
    print(f"\nVERDICT: {verdict}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"curve": {int(x): {"mean": float(y), "se": float(e)} for x, y, e in zip(xs, ys, ses)},
               "fit": {"c": float(c_), "a": float(a_), "b": float(b_)} if fitok else None,
               "extrapolation": pred, "still_climbing": bool(still_climbing), "verdict": verdict},
              open(OUT / "ceiling_scaling.json", "w"), indent=2)
    print(f"\n  -> {OUT/'ceiling_scaling.json'}")


if __name__ == "__main__":
    main()
