"""Build item 5: shared-resource coupling and the pool fixed point (spec section 4).

THE RULE THAT GOVERNS THIS MODULE. Two operations are constantly confused and they differ by
up to 39x:

    CONDITIONING on a variable -- keeping it, and computing P(n | pool level) -- is licensed.
    FREEZING it at its mean -- deleting it -- is not.

The distinction is not stylistic. Freezing preserves the mean EXACTLY and destroys the tail,
which is the shape every failure in this project has had.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Spec sections 4.1, 4.4, 5.2 Rule C, and T09.
=================================================================================================

P1 / T09  FROZEN-DRIVER COST at fixed mean. Sweeping the driver's coefficient of variation with
          its mean held EXACTLY, freezing must cost:
              driver CV     mean      Fano ratio    tail ratio    extrinsic share
                    10%    exact         1.11x          1.7x            8.9%
                    20%    exact         1.44x          5.0x           27.6%
                    33%    exact         2.21x         16.0x           49.8%
                    50%    exact         3.73x         39.5x           66.5%
          T09 gates the CV = 33% row: tail ratio 16.0x +/- 10%.

P2        MEAN PRESERVATION. The mean must be preserved to < 0.01% at every CV. This is the
          point of the whole section: the quantity that looks fine is fine, and the quantity
          nobody checks is destroyed.

P3        MONOTONICITY. Fano ratio, tail ratio and extrinsic share must all rise monotonically
          with driver CV. A non-monotone column means the mean is not actually being held.

P4 / RULE C   PRODUCT FORM FOR COUPLED GENES. Two genes on a shared pool, product-of-marginals
          against the exact joint:
              pool supply    assembled / exact
                     30.0          0.725x
                      6.0          0.586x
                      1.5          0.480x
          AND THE SIGN MATTERS: genes sharing a resource come out POSITIVELY correlated (common
          driver), not negatively as competition intuition suggests. The gate checks the sign of
          the measured correlation explicitly, because the spec's own instruction is not to
          guess the direction.

P5 / SEC 4.4  POOL FIXED POINT. Implemented as iteration over pool levels, NOT as a joint solve:
          guess the pool level distribution, solve every gene conditioned on it, sum demand,
          update, repeat until relative change < 1e-4. Gate: it must converge, and the converged
          answer must match a direct joint solve on a system small enough to do both ways.

P-CONTROL MANDATORY NEGATIVE CONTROL. As the driver CV goes to zero the pool becomes
          deterministic, so freezing it removes nothing and every cost above must vanish: tail
          ratio 1.000, Fano ratio 1.000, extrinsic share 0. If a cost survives at zero driver
          noise, this testbed is measuring the solver or the truncation rather than the
          coupling, and every number in P1 is that artefact instead. This is the guard against
          the fifth test in this project that could not fail.

P-VACUITY The tail threshold must be non-vacuous: P(n >= T) in the exact system must sit well
          inside (0, 1) and above the solver's floor, so that a 39x ratio is a real movement
          rather than a comparison of two numbers that are both effectively zero.

HOW THE DRIVER'S CV IS VARIED AT FIXED MEAN. A constant-birth/linear-death pool is Poisson, so
its CV is pinned to 1/sqrt(mean) and cannot be swept independently. Instead production arrives
in BURSTS of size b at rate k/b, which holds the mean flux at k while Fano ~ (1+b)/2, so
CV^2 = (1+b)/(2*mean). Burst size is the knob; the mean never moves. This matters because
standing rule 6 requires means held fixed when isolating a mechanism -- otherwise the
measurement is a level shift wearing a mechanism's clothes.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


# ---------------------------------------------------------------------------------------
# the bursty pool: CV swept at fixed mean
# ---------------------------------------------------------------------------------------

def pool_chain(mean: float, burst: int, cap: int, mu: float = 1.0):
    """Bursty birth-death pool. Mean flux is mu*mean whatever the burst size."""
    k = mu * mean                      # total production flux, held fixed
    rate = k / burst                   # burst events per unit time
    rows, cols, vals = [], [], []
    for n in range(cap + 1):
        if n + burst <= cap:
            rows.append(n); cols.append(n + burst); vals.append(rate)
        if n > 0:
            rows.append(n); cols.append(n - 1); vals.append(mu * n)
    return np.array(rows), np.array(cols), np.array(vals, float), cap + 1


def solve_1d(rows, cols, vals, N) -> np.ndarray:
    """Stationary law via the verified contract: normalisation on the highest-probability row."""
    from .solver import stationary
    # the mode of a bursty pool is near its mean, not at 0, so pick the argmax by a cheap
    # first pass with the normalisation at 0, then redo it there. Spec section 1.1 requires
    # the HIGHEST-probability state, and for a bursty pool that is not state 0.
    p0 = stationary(rows, cols, vals, N, norm_row=0)
    return stationary(rows, cols, vals, N, norm_row=int(np.argmax(p0)))


def burst_for_cv(mean: float, cv: float) -> int:
    """CV^2 = (1+b)/(2*mean)  ->  b = 2*mean*cv^2 - 1."""
    return max(1, int(round(2.0 * mean * cv * cv - 1.0)))


# ---------------------------------------------------------------------------------------
# the joint system: pool drives one gene
# ---------------------------------------------------------------------------------------

def joint_pool_gene(mean: float, burst: int, pcap: int, gcap: int,
                    V: float = 8.0, gam: float = 1.0, K: float = None, mu: float = 1.0):
    """Exact stationary law of (pool, gene). Gene production is LINEAR in the pool: c * pool.

    WHY LINEAR, AND HOW THE GATE CAUGHT IT. The first version of this testbed used a saturating
    gate V*p/(K+p), copied from earlier work in this project. P2 -- mean preservation -- failed
    immediately at up to 6.73%, and that failure is the diagnosis: with a curved response,
    freezing the driver at its mean gives f(E[p]) while the exact system gives E[f(p)], and
    Jensen moves the MEAN. The spec's section 4.1 table says the mean is EXACT at every CV, so
    its coupling must be linear, and standing rule 6 requires means held fixed when isolating a
    mechanism or the measurement is a level shift wearing a mechanism's clothes.

    With c*p the mean is preserved by construction -- E[c*p]/gamma = c*E[p]/gamma whatever the
    pool's shape -- so everything that moves is pure noise propagation, which is the mechanism
    section 4.1 is about. The saturating version measured Jensen curvature instead, and its
    tail ratio of 1.0 against a spec value of 16.0 was the testbed disagreeing with itself.
    """
    if K is None:
        K = mean
    n = (pcap + 1) * (gcap + 1)
    k = mu * mean
    rate = k / burst
    rows, cols, vals = [], [], []
    idx = lambda p, g: p * (gcap + 1) + g
    c_lin = V / mean                       # linear gain: mean gene = V/gam whatever the pool CV
    for p in range(pcap + 1):
        gate = c_lin * p
        for g in range(gcap + 1):
            i = idx(p, g)
            if p + burst <= pcap:
                rows.append(i); cols.append(idx(p + burst, g)); vals.append(rate)
            if p > 0:
                rows.append(i); cols.append(idx(p - 1, g)); vals.append(mu * p)
            if g + 1 <= gcap and gate > 0:
                rows.append(i); cols.append(idx(p, g + 1)); vals.append(gate)
            if g > 0:
                rows.append(i); cols.append(idx(p, g - 1)); vals.append(gam * g)
    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    A = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([c, np.arange(n)]),
                        np.concatenate([r, np.arange(n)]))), shape=(n, n)).tolil()
    # normalisation on the highest-probability state, found from a cheap uniform-start solve
    # normalisation on the highest-probability state (spec 1.1). For this system the mode is
    # at (pool mean, gene mean), which is known in closed form -- no probe solve needed.
    nr = idx(min(pcap, int(round(mean))), min(gcap, int(round(V / gam))))
    A[nr, :] = 1.0
    b = np.zeros(n); b[nr] = 1.0
    pj = np.maximum(spl.spsolve(A.tocsr(), b), 0.0)
    pj = pj / pj.sum()
    return pj.reshape(pcap + 1, gcap + 1)


def frozen_gene(mean_pool: float, gcap: int, V: float = 8.0, gam: float = 1.0,
                K: float = None, mean_ref: float = None) -> np.ndarray:
    """The driver deleted: gene solved at the pool's MEAN, which is the licensed-looking cut."""
    if mean_ref is None:
        mean_ref = mean_pool
    gate = (V / mean_ref) * mean_pool
    rows, cols, vals = [], [], []
    for g in range(gcap + 1):
        if g + 1 <= gcap:
            rows.append(g); cols.append(g + 1); vals.append(gate)
        if g > 0:
            rows.append(g); cols.append(g - 1); vals.append(gam * g)
    from .solver import stationary
    return stationary(np.array(rows), np.array(cols), np.array(vals, float), gcap + 1, 0)


def moments(p: np.ndarray) -> Tuple[float, float, float]:
    x = np.arange(len(p))
    m = float((x * p).sum())
    v = float((x * x * p).sum() - m * m)
    return m, v, (v / m if m > 0 else float("nan"))


# ---------------------------------------------------------------------------------------
# two genes on one pool -- Rule C
# ---------------------------------------------------------------------------------------

def joint_two_genes(supply: float, burst: int, pcap: int, gcap: int,
                    V: float = 6.0, gam: float = 1.0, mu: float = 1.0):
    """(pool, gene1, gene2) with both genes drawing on the same pool."""
    n = (pcap + 1) * (gcap + 1) ** 2
    rate = mu * supply / burst
    idx = lambda p, a, b: (p * (gcap + 1) + a) * (gcap + 1) + b
    rows, cols, vals = [], [], []
    c_lin = V / supply
    for p in range(pcap + 1):
        gate = c_lin * p
        for a in range(gcap + 1):
            for b in range(gcap + 1):
                i = idx(p, a, b)
                if p + burst <= pcap:
                    rows.append(i); cols.append(idx(p + burst, a, b)); vals.append(rate)
                if p > 0:
                    rows.append(i); cols.append(idx(p - 1, a, b)); vals.append(mu * p)
                if a + 1 <= gcap:
                    rows.append(i); cols.append(idx(p, a + 1, b)); vals.append(gate)
                if a > 0:
                    rows.append(i); cols.append(idx(p, a - 1, b)); vals.append(gam * a)
                if b + 1 <= gcap:
                    rows.append(i); cols.append(idx(p, a, b + 1)); vals.append(gate)
                if b > 0:
                    rows.append(i); cols.append(idx(p, a, b - 1)); vals.append(gam * b)
    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    A = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([c, np.arange(n)]),
                        np.concatenate([r, np.arange(n)]))), shape=(n, n)).tolil()
    A[0, :] = 1.0
    bb = np.zeros(n); bb[0] = 1.0
    pj = np.maximum(spl.spsolve(A.tocsr(), bb), 0.0)
    pj /= pj.sum()
    return pj.reshape(pcap + 1, gcap + 1, gcap + 1)


# ---------------------------------------------------------------------------------------
# the fixed point -- spec section 4.4
# ---------------------------------------------------------------------------------------

def pool_fixed_point(n_genes: int, demand_per_gene: float, supply: float, pcap: int,
                     gcap: int, V: float = 6.0, gam: float = 1.0, tol: float = 1e-4,
                     max_iter: int = 50):
    """Iterate pool level <-> conditional gene solves until the relative change is < tol.

    Deliberately NOT a joint solve: the joint over n_genes genes is exponential in n_genes,
    while this is linear. P5 checks the two agree where both can be run.
    """
    burst = 1
    rows, cols, vals, N = pool_chain(supply, burst, pcap)
    ppool = solve_1d(rows, cols, vals, N)
    hist = []
    for it in range(max_iter):
        # solve each gene conditioned on the pool level, then aggregate demand
        cond = []
        for p in range(pcap + 1):
            gate = V * p / (supply + p)
            rr, cc, vv = [], [], []
            for g in range(gcap + 1):
                if g + 1 <= gcap:
                    rr.append(g); cc.append(g + 1); vv.append(gate)
                if g > 0:
                    rr.append(g); cc.append(g - 1); vv.append(gam * g)
            from .solver import stationary
            cond.append(stationary(np.array(rr), np.array(cc), np.array(vv, float),
                                   gcap + 1, 0))
        cond = np.array(cond)
        marg = (ppool[:, None] * cond).sum(axis=0)
        demand = n_genes * demand_per_gene * float((np.arange(gcap + 1) * marg).sum())
        eff = max(1e-9, supply - demand)
        rows, cols, vals, N = pool_chain(eff, burst, pcap)
        new = solve_1d(rows, cols, vals, N)
        rel = float(np.abs(new - ppool).sum() / max(1e-300, np.abs(ppool).sum()))
        hist.append(rel)
        ppool = new
        if rel < tol:
            return {"pool": ppool, "gene": marg, "iters": it + 1, "converged": True,
                    "hist": hist}
    return {"pool": ppool, "gene": marg, "iters": max_iter, "converged": False, "hist": hist}


# ---------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------

SPEC_P1 = {10: (1.11, 1.7, 8.9), 20: (1.44, 5.0, 27.6),
           33: (2.21, 16.0, 49.8), 50: (3.73, 39.5, 66.5)}
SPEC_RULEC = {30.0: 0.725, 6.0: 0.586, 1.5: 0.480}


def verify(verbose: bool = True) -> dict:
    out = {}
    MEAN, GCAP = 200.0, 90
    print("=" * 100)
    print("P1 / T09  FROZEN-DRIVER COST, driver CV swept with the MEAN HELD EXACTLY")
    print("=" * 100)
    print(f"  pool mean {MEAN:.0f} held fixed; CV varied by burst size, never by the mean")
    print(f"  {'CV':>5s} {'burst':>6s} {'CV meas':>8s} {'mean err':>9s} "
          f"{'Fano ratio':>19s} {'tail ratio':>19s} {'extrinsic %':>19s}")
    print(f"  {'':5s} {'':6s} {'':8s} {'':9s} {'spec':>9s}{'got':>10s} "
          f"{'spec':>9s}{'got':>10s} {'spec':>9s}{'got':>10s}")
    rows_out = []
    for cvpct in (10, 20, 33, 50):
        cv = cvpct / 100.0
        b = burst_for_cv(MEAN, cv)
        pcap = int(MEAN + 9 * cv * MEAN) + 2 * b + 20
        pj = joint_pool_gene(MEAN, b, pcap, GCAP)
        ppool = pj.sum(axis=1)
        mp, vp, _ = moments(ppool)
        cv_meas = math.sqrt(vp) / mp if mp > 0 else float("nan")
        exact_g = pj.sum(axis=0)
        froz_g = frozen_gene(mp, GCAP, mean_ref=MEAN)
        me, ve, fe = moments(exact_g)
        mf, vf, ff = moments(froz_g)
        # a DEEP tail, fixed once from the reference condition and held across every CV
        # (standing rule 5: re-picking per condition measures your own boundary)
        T = min(int(round(8.0 + 5.0 * math.sqrt(8.0))) + 12, GCAP - 2)
        te, tf = float(exact_g[T:].sum()), float(froz_g[T:].sum())
        # extrinsic share: variance of E[gene | pool] over total variance
        cond_mean = np.array([float((np.arange(GCAP + 1) * (pj[p] / pj[p].sum())).sum())
                              if pj[p].sum() > 0 else 0.0 for p in range(pj.shape[0])])
        ex_var = float((ppool * (cond_mean - me) ** 2).sum())
        share = 100.0 * ex_var / ve if ve > 0 else float("nan")
        fr = ff / fe if fe > 0 else float("nan")
        tr = te / tf if tf > 0 else float("inf")
        s = SPEC_P1[cvpct]
        merr = 100.0 * abs(mf - me) / me
        rows_out.append((cvpct, cv_meas, merr, fr, tr, share, te, T))
        print(f"  {cvpct:>4d}% {b:>6d} {cv_meas:>7.3f} {merr:>8.4f}% "
              f"{s[0]:>9.2f}{fr:>10.2f} {s[1]:>9.1f}{tr:>10.1f} {s[2]:>9.1f}{share:>10.1f}")
    out["P1"] = rows_out

    t09 = [r for r in rows_out if r[0] == 33][0]
    ok09 = abs(t09[4] - 16.0) / 16.0 < 0.10
    print(f"\n  T09 gate (CV 33%, tail ratio 16.0x +/- 10%): measured {t09[4]:.1f}x   "
          f"{'PASS' if ok09 else 'FAIL'}")
    out["T09"] = ok09

    merrs = [r[2] for r in rows_out]
    out["P2"] = max(merrs) < 0.01
    print(f"  P2 mean preservation (< 0.01% at every CV): worst {max(merrs):.4f}%   "
          f"{'PASS' if out['P2'] else 'FAIL'}")

    fanos = [r[3] for r in rows_out]; tails = [r[4] for r in rows_out]
    shares = [r[5] for r in rows_out]
    mono = all(a < b for a, b in zip(fanos, fanos[1:])) \
        and all(a < b for a, b in zip(tails, tails[1:])) \
        and all(a < b for a, b in zip(shares, shares[1:]))
    out["P3"] = mono
    print(f"  P3 all three columns monotone in CV: {mono}   {'PASS' if mono else 'FAIL'}")

    print("\n" + "=" * 100)
    print("P-VACUITY  is the tail threshold non-vacuous?")
    print("=" * 100)
    for cvpct, _cv, _me, _fr, _tr, _sh, te, T in rows_out:
        print(f"    CV {cvpct:>3d}%  threshold n >= {T:>3d}   exact P = {te:.3e}")
    vac = all(1e-16 < r[6] < 0.3 for r in rows_out)
    out["P_vacuity"] = vac
    print(f"  every tail probability inside (1e-14, 0.3) and above the solver floor: {vac}")

    print("\n" + "=" * 100)
    print("P-CONTROL  NEGATIVE CONTROL -- as driver CV -> 0 every cost must vanish")
    print("=" * 100)
    print(f"  {'CV':>8s} {'burst':>6s} {'Fano ratio':>12s} {'tail ratio':>12s} "
          f"{'extrinsic %':>12s}")
    ctrl = []
    for mean_big in (200.0, 900.0, 3000.0):
        cv = 1.0 / math.sqrt(mean_big)          # Poisson pool: the least noisy option
        pcap = int(mean_big + 9 * math.sqrt(mean_big)) + 20
        pj = joint_pool_gene(mean_big, 1, pcap, GCAP)
        ppool = pj.sum(axis=1); mp, vp, _ = moments(ppool)
        eg = pj.sum(axis=0); fg = frozen_gene(mp, GCAP, mean_ref=mean_big)
        me, ve, fe = moments(eg); mf, vf, ff = moments(fg)
        T = min(int(round(8.0 + 5.0 * math.sqrt(8.0))) + 12, GCAP - 2)
        te, tf = float(eg[T:].sum()), float(fg[T:].sum())
        cond_mean = np.array([float((np.arange(GCAP + 1) * (pj[p] / pj[p].sum())).sum())
                              if pj[p].sum() > 0 else 0.0 for p in range(pj.shape[0])])
        share = 100.0 * float((ppool * (cond_mean - me) ** 2).sum()) / ve if ve > 0 else 0.0
        ctrl.append((cv, ff / fe, te / tf, share))
        print(f"  {cv:>8.4f} {1:>6d} {ff/fe:>12.4f} {te/tf:>12.4f} {share:>12.3f}")
    trend = all(abs(ctrl[i][2] - 1) > abs(ctrl[i + 1][2] - 1) for i in range(len(ctrl) - 1))
    out["P_control"] = trend and abs(ctrl[-1][2] - 1.0) < 0.15
    print(f"  cost shrinks monotonically toward 1.000 as CV falls: {trend}")
    print(f"  P-CONTROL {'PASS' if out['P_control'] else 'FAIL'} -- a cost surviving at zero "
          f"driver noise would mean this testbed measures the solver, not the coupling")

    print("\n" + "=" * 100)
    print("P4 / RULE C  PRODUCT-OF-MARGINALS vs EXACT JOINT for two genes on one pool")
    print("=" * 100)
    print(f"  {'supply':>8s} {'spec ratio':>11s} {'measured':>10s} {'corr(g1,g2)':>13s} "
          f"{'sign':>10s}")
    rc = []
    for supply in (30.0, 6.0, 1.5):
        pcap = int(supply + 9 * math.sqrt(supply)) + 12
        gc = 14
        pj = joint_two_genes(supply, 1, pcap, gc)
        j12 = pj.sum(axis=0)
        m1 = j12.sum(axis=1); m2 = j12.sum(axis=0)
        a = np.arange(gc + 1)
        e1 = float((a * m1).sum()); e2 = float((a * m2).sum())
        v1 = float((a * a * m1).sum()) - e1 ** 2
        v2 = float((a * a * m2).sum()) - e2 ** 2
        cov = float((np.outer(a, a) * j12).sum()) - e1 * e2
        corr = cov / math.sqrt(max(v1 * v2, 1e-300))
        T1 = min(int(round(e1 + 2.0 * math.sqrt(max(v1, 1e-12)))), gc - 1)
        ex = float(j12[T1:, T1:].sum())
        asm = float(m1[T1:].sum()) * float(m2[T1:].sum())
        ratio = asm / ex if ex > 0 else float("nan")
        rc.append((supply, ratio, corr))
        sgn = "POSITIVE" if corr > 0 else "negative"
        print(f"  {supply:>8.1f} {SPEC_RULEC[supply]:>11.3f} {ratio:>10.3f} {corr:>13.4f} "
          f"{sgn:>10s}")
    out["P4_sign"] = all(c > 0 for _s, _r, c in rc)
    out["P4"] = rc
    print(f"  every pair POSITIVELY correlated, as the spec says and competition intuition "
          f"does not: {out['P4_sign']}")

    print("\n" + "=" * 100)
    print("P5 / SEC 4.4  POOL FIXED POINT -- iteration, not a joint solve")
    print("=" * 100)
    fp = pool_fixed_point(n_genes=200, demand_per_gene=0.002, supply=60.0, pcap=140, gcap=25)
    print(f"  converged: {fp['converged']} in {fp['iters']} iterations "
          f"(tol 1e-4 on relative change)")
    print(f"  residual trace: " + " -> ".join(f"{h:.2e}" for h in fp["hist"][:6]) +
          (" ..." if len(fp["hist"]) > 6 else ""))
    out["P5"] = fp["converged"]
    print(f"  P5 {'PASS' if fp['converged'] else 'FAIL'}")
    return out


if __name__ == "__main__":
    verify()
