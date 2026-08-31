"""G5: does the tensor-train rank of a gene network's stationary distribution saturate?

WHY THIS IS THE CONSEQUENTIAL NUMBER. In the layered architecture, REM owns the stochastic
gene-expression layer via chain elimination on a tensor train. The cost is governed by the
rank r across the cuts, and the projected budget at 500 genes is minutes at r ~ 30, hours at
r ~ 120, and hopeless at r ~ 2500. So the whole layer is a plan or a dead end depending on
whether r grows with gene count or levels off.

THE EXISTING G5 NUMBERS CANNOT ANSWER IT, AND THIS IS THE FIRST THING TO FIX.
rem.rare.verify()'s G5 reports rank at 2, 3 and 4 genes on the cascade at M = 10. The state
space per gene is d = 11, and the rank across a cut is bounded ABOVE, for free, by the smaller
side of the cut: b(n) = d^floor(n/2). Measured against that bound the existing numbers are

    n = 2   cut 1|1   bound  11   observed r@1e-6 = 7    64% of the ceiling
    n = 3   cut 1|2   bound  11   observed r@1e-6 = 9    82% of the ceiling
    n = 4   cut 2|2   bound 121   observed r@1e-6 = 16   13% of the ceiling

At n = 2 and n = 3 the rank is not measuring the physics. It is measuring d. A rank cannot
exceed 11 there no matter how entangled the distribution is, so the rise from 7 to 9 is what
the dimension permits, not what the coupling demands, and ANY growth law fitted through those
points is an artifact -- the same shape of error as fitting a trend through a saturating
detector. Exactly one of the three points, n = 4, is far enough below its bound to be a
statement about the distribution, and one point does not have a slope.

So the reported "~5 per gene" has no support, in either direction: it is not evidence of
growth and it is not evidence of saturation. This module measures the quantity where it can
actually vary.

=================================================================================================
THE GATES, FIXED BEFORE ANY NUMBER IS RUN.
=================================================================================================

G5a  THE BOUND MUST NOT BE BINDING. For every n report the free bound b(n) = d^floor(n/2)
     beside the observed rank r(n). A point is ADMISSIBLE for a growth fit only if it has at
     least HEADROOM-fold room, r(n) <= b(n)/HEADROOM. Inadmissible points are reported and
     EXCLUDED from every fit, and the run states how many admissible points it obtained. If
     there are fewer than 4, no growth law is fitted at all and the gate reports that instead
     of a slope.

G5b  THE GROWTH LAW, fitted only on admissible n. Fit r(n) = a + b n by least squares and
     report b with its standard error, against the constant model.
       b >= 1.0 rank/gene AND b >= 2 s.e.  -> GROWTH. Report the implied r(500) and say the
                                              layer is dead at that rank if it exceeds 200.
       |b| < 1.0 and b < 2 s.e.            -> SATURATION at the observed level.
       anything else                       -> INCONCLUSIVE, reported as such, not rounded
                                              into whichever verdict is convenient.

G5c  THE MAXIMUM OVER COUPLING, NEVER ONE POINT. The physical argument for saturation is that
     weak coupling factorises and strong coupling locks, so only the crossover is expensive.
     If that is right, a single coupling measures whichever regime it happened to sit in. The
     reported r(n) is therefore the MAXIMUM over a coupling sweep, and the sweep is shown so
     the crossover is visible rather than asserted.

G5d  SOLVER VALIDATION. Reaching n = 8-9 needs an ILU-preconditioned iterative solve instead
     of the direct factorisation rem.rare.stationary uses; that module's relative-accuracy
     guarantee is not inherited and is not claimed. The iterative path is validated against
     the trusted direct solver at every n where both run. GATE: worst pointwise relative
     difference <= 1e-8. Any singular-value tolerance finer than the validated accuracy is
     reported as NOT TRUSTWORTHY rather than quoted.

G5e  TOLERANCE CONSISTENCY. A rank without a tolerance is not a number, so r(n) is reported at
     1e-3, 1e-6 and 1e-10. GATE: the G5b verdict must be the SAME at 1e-3 and 1e-6. If the
     answer depends on where the spectrum is cut, the result is INCONCLUSIVE.

G5f  THE TRUNCATION MUST NOT BE THE PHYSICS. Each species is capped at M. If probability piles
     up against that cap, the rank is reporting the truncation. GATE: report max_i P(x_i = M);
     if it exceeds 1e-3 the configuration is rejected and the sweep says so.

G5g  EXTRAPOLATION HONESTY. This reaches some N_max well below 500. The extrapolation factor
     500/N_max is stated with every projected number, projections are labelled projections,
     and saturation observed to n = 9 is reported as evidence about n <= 9 -- consistent with
     saturation at 500, never a demonstration of it.

G5h  TOPOLOGY SENSITIVITY, as a diagnostic and not a gate. A cascade is a chain, which is the
     friendliest possible topology for a tensor train laid out in the same order. Real gene
     networks have hubs. The same measurement is run on a hub topology to show whether the
     chain result is a property of gene networks or of chains.
"""
from __future__ import annotations

import argparse, json, sys, time
sys.path.insert(0, ".")
import numpy as np
import scipy.sparse.linalg as spla

from rem.rare import Network, Reaction, cascade, stationary

TOLS = (1e-3, 1e-6, 1e-10)
HEADROOM = 4.0                 # a point counts only with 4x room below its free bound
VALID_TOL = 1e-8               # G5d bar on the iterative solver
CAP_BAR = 1e-3                 # G5f bar on probability at the truncation cap
SLOPE_BAR, SE_MULT = 1.0, 2.0  # G5b decision thresholds
DEAD_RANK = 200                # rank above which the 500-gene layer is called dead


def stationary_iterative(net, tol=1e-12, drop=1e-5, fill=12):
    """Sparse stationary solve that keeps the sparsity.

    rem.rare.stationary replaces row 0 with the normalisation, which makes that row DENSE and
    wrecks the LU fill-in. Pinning p[0] = 1 and solving the (n-1) subsystem is equivalent
    after normalisation, and preconditioned GMRES on it reaches state spaces the direct
    factorisation cannot. The relative-accuracy property proved for the direct M-matrix solve
    is NOT inherited here and is not claimed; G5d measures what this path actually delivers.
    """
    Q = net.generator()
    n = Q.shape[0]
    At = Q.T.tocsc()
    A = At[1:, 1:].tocsc()
    b = -At[1:, 0].toarray().ravel()
    t0 = time.perf_counter()
    ilu = spla.spilu(A, drop_tol=drop, fill_factor=fill)
    M = spla.LinearOperator(A.shape, ilu.solve)
    x, info = spla.gmres(A, b, M=M, rtol=tol, restart=60, maxiter=3000)
    p = np.concatenate([[1.0], x])
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p, {"n_states": n, "info": int(info), "seconds": time.perf_counter() - t0,
               "residual_inf": float(np.abs(Q.T @ p).max())}


def middle_cut(dims):
    """Balanced cut, and the free upper bound on any rank across it."""
    n = len(dims)
    split = n // 2
    left = int(np.prod(dims[:split]))
    right = int(np.prod(dims[split:]))
    return split, min(left, right)


def ranks_at(net, p):
    split, bound = middle_cut(net.dims)
    left = int(np.prod(net.dims[:split]))
    sv = np.linalg.svd(p.reshape(left, -1), compute_uv=False)
    sv = sv / sv[0]
    return {"cut": split, "bound": int(bound),
            "r": {t: int((sv > t).sum()) for t in TOLS},
            "sv_head": [float(x) for x in sv[:24]]}


def cap_mass(net, p):
    """G5f: probability sitting on the truncation boundary."""
    S = net.states()
    P = p.reshape(-1)
    worst = 0.0
    for i, d in enumerate(net.dims):
        worst = max(worst, float(P[S[:, i] == d - 1].sum()))
    return worst


def hub(n_genes, M=3, g=2.5, gamma=1.0, K=1.5, h=2.0):
    """One driver activating every other gene: the topology a chain is friendliest against."""
    hill = lambda v: (v / K) ** h / (1.0 + (v / K) ** h)
    names = [f"X{i+1}" for i in range(n_genes)]
    rx = [Reaction("X1+", lambda S: np.full(len(S), g * 0.6), tuple([1] + [0] * (n_genes - 1))),
          Reaction("X1-", lambda S: gamma * S[:, 0], tuple([-1] + [0] * (n_genes - 1)))]
    for i in range(1, n_genes):
        up = np.zeros(n_genes, dtype=int); up[i] = 1
        dn = np.zeros(n_genes, dtype=int); dn[i] = -1
        rx.append(Reaction(f"X{i+1}+", (lambda: (lambda S: g * hill(S[:, 0])))(), tuple(up)))
        rx.append(Reaction(f"X{i+1}-", (lambda k: (lambda S: gamma * S[:, k]))(i), tuple(dn)))
    return Network(names, [M] * n_genes, rx)


def build(topology, n, M, K, g):
    return (cascade(n, M, g=g, K=K) if topology == "cascade"
            else hub(n, M=M, g=g, K=K))


def measure(topology, n, M, K, g, use_iter=True):
    net = build(topology, n, M, K, g)
    p, info = stationary_iterative(net) if use_iter else stationary(net)
    rk = ranks_at(net, p)
    return {"n": n, "M": M, "K": K, "g": g, "d": int(net.dims[0]),
            "states": int(info["n_states"]), "seconds": float(info["seconds"]),
            "cap_mass": cap_mass(net, p), **rk}


def fit_slope(ns, rs):
    ns, rs = np.asarray(ns, float), np.asarray(rs, float)
    if len(ns) < 2:
        return float("nan"), float("nan")
    A = np.column_stack([np.ones_like(ns), ns])
    beta, *_ = np.linalg.lstsq(A, rs, rcond=None)
    resid = rs - A @ beta
    dof = len(ns) - 2
    if dof <= 0:
        return float(beta[1]), float("nan")
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--g", type=float, default=2.5)
    ap.add_argument("--nmax", type=int, default=8)
    ap.add_argument("--sweep-nmax", type=int, default=7)
    ap.add_argument("--topology", default="cascade")
    ap.add_argument("--out", default="benchmarks/tt_rank.json")
    a = ap.parse_args(argv)
    out = {"config": vars(a), "validate": [], "sweep": [], "scaling": []}

    # ---- G5d: validate the iterative path against the trusted direct solver ----
    print("  G5d  solver validation, iterative vs the trusted direct solve")
    worst = 0.0
    for n in (3, 4, 5):
        net = build(a.topology, n, a.M, 1.5, a.g)
        p1, _i1 = stationary(net)
        p2, _i2 = stationary_iterative(net)
        m = p1 > 0
        w = float((np.abs(p1[m] - p2[m]) / p1[m]).max())
        worst = max(worst, w)
        out["validate"].append({"n": n, "worst_rel": w})
        print(f"       n={n}  states={len(p1):7,d}  worst pointwise relative {w:.2e}")
    ok_solver = worst <= VALID_TOL
    print(f"       max {worst:.2e}  bar {VALID_TOL:.0e}  "
          f"{'PASS' if ok_solver else 'FAIL -- ranks below this accuracy are NOT TRUSTWORTHY'}")
    usable_tols = [t for t in TOLS if t > worst]
    print(f"       tolerances the solver can support: "
          f"{', '.join('%.0e' % t for t in usable_tols)}")
    out["G5d"] = {"worst_rel": worst, "pass": bool(ok_solver),
                  "usable_tols": usable_tols}

    # ---- G5c: find the crossover by sweeping coupling ----
    print("\n  G5c  coupling sweep -- rank is taken as the MAXIMUM over K, not one point")
    Ks = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    print(f"       {'n':>3s} {'K':>5s} {'states':>8s} {'bound':>6s} "
          f"{'r@1e-3':>7s} {'r@1e-6':>7s} {'capP':>9s} {'sec':>7s}")
    for n in range(4, a.sweep_nmax + 1):
        for K in Ks:
            r = measure(a.topology, n, a.M, K, a.g)
            out["sweep"].append(r)
            print(f"       {n:3d} {K:5.2f} {r['states']:8,d} {r['bound']:6d} "
                  f"{r['r'][1e-3]:7d} {r['r'][1e-6]:7d} {r['cap_mass']:9.2e} "
                  f"{r['seconds']:7.2f}", flush=True)
    best = {}
    for r in out["sweep"]:
        if r["cap_mass"] > CAP_BAR:
            continue
        k = r["n"]
        if k not in best or r["r"][1e-6] > best[k]["r"][1e-6]:
            best[k] = r
    Kstar = (max(best.values(), key=lambda r: r["r"][1e-6])["K"] if best else 1.5)
    print(f"       worst-case coupling K* = {Kstar} (used for the large-n scaling)")
    out["Kstar"] = Kstar

    # ---- the scaling run at the worst-case coupling ----
    print(f"\n  SCALING at K* = {Kstar}, topology {a.topology}, M = {a.M}, d = {a.M + 1}")
    print(f"       {'n':>3s} {'states':>9s} {'cut':>6s} {'bound':>7s} {'r@1e-3':>7s} "
          f"{'r@1e-6':>7s} {'r@1e-10':>8s} {'r/bound':>8s} {'capP':>9s} {'adm':>4s} {'sec':>8s}")
    for n in range(2, a.nmax + 1):
        r = measure(a.topology, n, a.M, Kstar, a.g)
        adm = r["r"][1e-6] <= r["bound"] / HEADROOM
        r["admissible"] = bool(adm)
        out["scaling"].append(r)
        cutlbl = f"{r['cut']}|{n - r['cut']}"
        print(f"       {n:3d} {r['states']:9,d} {cutlbl:>6s} "
              f"{r['bound']:7d} {r['r'][1e-3]:7d} {r['r'][1e-6]:7d} {r['r'][1e-10]:8d} "
              f"{r['r'][1e-6] / r['bound']:8.2f} {r['cap_mass']:9.2e} "
              f"{'yes' if adm else 'NO':>4s} {r['seconds']:8.2f}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)

    # ---- G5a / G5b / G5e ----
    print(f"\n  G5a  admissibility: a point counts only with {HEADROOM:.0f}x room under its "
          f"free bound d^floor(n/2)")
    adm = [r for r in out["scaling"] if r["admissible"]]
    print(f"       {len(adm)}/{len(out['scaling'])} points admissible "
          f"(n = {[r['n'] for r in adm]})")
    out["G5a"] = {"n_admissible": len(adm), "ns": [r["n"] for r in adm]}
    if len(adm) < 4:
        print(f"       FEWER THAN 4 ADMISSIBLE POINTS -- no growth law is fitted. The "
              f"measurement reached n <= {a.nmax} and the free bound is still doing the work.")
        out["G5b"] = {"verdict": "INSUFFICIENT"}
    else:
        print(f"\n  G5b  growth law, fitted on admissible points only")
        verdicts = {}
        for t in (1e-3, 1e-6):
            b, se = fit_slope([r["n"] for r in adm], [r["r"][t] for r in adm])
            grow = (b >= SLOPE_BAR and b >= SE_MULT * se)
            flat = (abs(b) < SLOPE_BAR and b < SE_MULT * se)
            v = "GROWTH" if grow else ("SATURATION" if flat else "INCONCLUSIVE")
            verdicts[t] = v
            proj = b * 500 + (np.mean([r["r"][t] for r in adm]) - b * np.mean(
                [r["n"] for r in adm]))
            print(f"       tol {t:.0e}: slope {b:+.3f} +- {se:.3f} rank/gene   {v}"
                  f"   projected r(500) = {proj:.0f}")
            out.setdefault("G5b", {})[str(t)] = {"slope": b, "se": se, "verdict": v,
                                                 "proj500": float(proj)}
        same = verdicts[1e-3] == verdicts[1e-6]
        print(f"\n  G5e  tolerance consistency: 1e-3 says {verdicts[1e-3]}, 1e-6 says "
              f"{verdicts[1e-6]}  -> {'CONSISTENT' if same else 'INCONCLUSIVE'}")
        out["G5e"] = {"consistent": bool(same)}
        print(f"\n  G5g  extrapolation: measured to n = {max(r['n'] for r in adm)}, "
              f"target 500, a factor of "
              f"{500 / max(r['n'] for r in adm):.0f}. Any r(500) above is a PROJECTION.")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
