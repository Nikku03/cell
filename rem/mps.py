"""Fixed-rank (MPS/TEBD) solver for DRIVEN, inhomogeneous 1D master equations.

WHAT THIS MODULE GIVES UP, STATED FIRST. Everywhere else in REM the answer is EXACT.
Here it is not. For a driven 1D system with per-site rates the stationary state is the null
vector of a master-equation generator, and it is not given by any local product formula, so
chain elimination does not apply. This module is a TRUNCATED method with a measurable error
bar. That is what DMRG/TEBD already is; the contribution is integration into REM and honest
error accounting, not novelty.

WHY EXACT IS DEAD HERE, measured rather than assumed. Schmidt rank across the middle cut of
the exact stationary distribution, open-boundary TASEP, alpha=0.8 beta=0.9, tol 1e-12:

    rates                        L=6   8   10   12   14      (max possible: 8 16 32 64 128)
    uniform (the DEHP case)        4   5    6    7    8
    mild spread 0.7-1.4x           8  16   32   64  116
    realistic spread 0.2-5x        8  16   32   62  105

Essentially full rank once rates vary, at every density. The famous L/2+1 result holds only
for the UNIFORM case -- the one already solved analytically in 1993, and the one that is not
biology. But the singular values decay geometrically, so truncation is viable: at L=14 with
realistic rates, keeping the top k gives k=4 -> 8.1e-03, k=8 -> 5.9e-04, k=16 -> 2.0e-05,
k=32 -> 7.0e-08.

THE MODEL. TASEP-like transport: injection at site 0 with rate alpha, hop i -> i+1 with a
PER-SITE rate w_i (this is the whole point -- uniform w is the solved case), extraction at
site L-1 with rate beta. State is a probability vector over 2^L configurations, evolved by
dP/dt = L P and truncated as an MPS with physical dimension 2 and bond dimension chi.

TWO IMPLEMENTATION REQUIREMENTS, both of which are known defects and are treated as spec:

  (a) SECOND-ORDER SYMMETRIC TROTTER, not first order. The sweep is
          S(dt/2) . E(dt/2) . O(dt) . E(dt/2) . S(dt/2)
      with S the boundary gates and E/O the even/odd bulk bonds. First order looks fine and
      silently pins the accuracy floor. The signature that distinguishes them is the
      EXPONENT, not the magnitude: first order has err/dt constant (0.109-0.114 over
      dt = 0.1 ... 0.002), second order has err/dt^2 constant (0.0853 over dt = 0.2 ... 0.01).

  (b) NEVER TRUNCATE A NON-CANONICAL MPS. Before discarding anything, left-canonicalise the
      whole chain by QR, then sweep right-to-left doing the SVD truncation. Without this the
      local SVD does not measure the global discarded weight and the error is uncontrolled:
      chi=8 gave 1e-1 where the true spectrum said 6e-4; after the fix, 3.8e-4.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  G1  REPRODUCES THE EXACT ANSWER AT SMALL L. Against the sparse null-space solve at
      L = 8, 10, 12, 14 with realistic rate spread, alpha=0.8 beta=0.9, dt = 0.02 (Trotter
      floor about 3e-5). GATE: monotone geometric convergence in chi, and chi=16 reaching
      the Trotter floor. Reference values this threshold comes from:
          L=8   chi=2 1.8e-01  chi=4 8.3e-03  chi=8 3.8e-04  chi=16 3.4e-05
          L=10  chi=2 4.0e-01  chi=4 2.7e-02  chi=8 9.3e-04  chi=16 4.0e-05
          L=12  chi=2 6.0e-01  chi=4 4.9e-02  chi=8 1.3e-03  chi=16 1.0e-04
  G1b THE TROTTER ORDER ITSELF, gated on the exponent. Fit log(err) against log(dt) with
      truncation made irrelevant (chi large enough to be exact). GATE: fitted slope in
      [1.7, 2.3]. A first-order implementation gives slope ~1 and fails.
  G2  TIME CONVERGENCE SEPARATED FROM TRUNCATION CONVERGENCE. Relaxation time grows with L
      (roughly L^1.5 for TASEP in the maximal-current phase), so a chi-sweep at fixed t_max
      measures a mixture of "not converged in chi" and "not converged in time". A prototype
      at L=50, t_max=200 gave mean density 0.140 / 0.274 / 0.334 for chi = 2/4/8 --
      uninterpretable, because both were still moving. Every run therefore converges in TIME
      first, to a stated residual, at each chi SEPARATELY. GATE: every run reported in G1/G3
      must have realised |d rho/dt| below its stated tolerance, and the realised time is
      reported alongside.
  G3  THE QUESTION: does the required chi stay bounded as L grows? Sweep L, and at each L
      find the smallest chi holding a fixed accuracy (1e-3 and 1e-5) against the chi->large
      answer AT THAT SAME L. Report chi_required(L). Flat means driven 1D transport is
      solved at any length; linear means polynomial cost, state the exponent; exponential
      means the wall was postponed, not removed. DO NOT EXTRAPOLATE FROM L <= 14 -- there is
      already visible creep (chi=8: 3.8e-4 -> 9.3e-4 -> 1.3e-3 over L = 8, 10, 12) that
      three points cannot resolve either way.
  G4  GROUND TRUTH AT LARGE L, where exact is impossible: direct kinetic Monte Carlo
      (Gillespie) of the same process, with sampling error reported. GATE: agreement within
      the Monte Carlo error bar.
  G5  THE RELEVANCE GATE. Three applications in this project have now been exact, correct
      and useless because the coupling was too weak for exactness to matter. Compare the
      truncated interacting answer against an INDEPENDENT-SITE model at the same density and
      report the density at which they diverge by more than the truncation error. The
      equilibrium reference, error from ignoring exclusion entirely: 5% of close packing ->
      0.9%, 40% -> 7.6%, 80% -> 23.4%, 95% -> 55.0%. Real ribosome density is 5-10% of close
      packing. If the driven version shows the same, this module is correct and UNNECESSARY
      for that application and the README must say so.
  G6  PERFORMANCE. GATE: report wall-clock scaling and state plainly the largest L reached
      within the session's budget. A target that is not met is reported as not met.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

D = 2                       # physical dimension: empty / occupied


# --------------------------------------------------------------------------------------
# generators and gates
# --------------------------------------------------------------------------------------

def bulk_generator(w: float) -> np.ndarray:
    """2-site generator for a hop i -> i+1 at rate w. Basis index = 2*n_i + n_{i+1}."""
    L = np.zeros((4, 4))
    L[1, 2] = w                     # |10> -> |01>
    L[2, 2] = -w
    return L


def left_generator(alpha: float) -> np.ndarray:
    """Injection at the first site: |0> -> |1> at rate alpha."""
    return np.array([[-alpha, 0.0], [alpha, 0.0]])


def right_generator(beta: float) -> np.ndarray:
    """Extraction at the last site: |1> -> |0> at rate beta."""
    return np.array([[0.0, beta], [0.0, -beta]])


class GateCache:
    """exp(dt * G) computed ONCE per distinct rate, not per step (G6)."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self._c: Dict[Tuple[str, float, float], np.ndarray] = {}

    def get(self, kind: str, rate: float, frac: float = 1.0) -> np.ndarray:
        key = (kind, round(float(rate), 12), round(float(frac), 12))
        if key not in self._c:
            from scipy.linalg import expm
            g = {"bulk": bulk_generator, "left": left_generator,
                 "right": right_generator}[kind](rate)
            self._c[key] = expm(self.dt * frac * g)
        return self._c[key]


# --------------------------------------------------------------------------------------
# MPS
# --------------------------------------------------------------------------------------

def product_mps(L: int, p_occ: float = 0.0) -> List[np.ndarray]:
    """Product-state MPS: every site independently occupied with probability p_occ."""
    t = []
    for _ in range(L):
        a = np.zeros((1, D, 1))
        a[0, 0, 0] = 1.0 - p_occ
        a[0, 1, 0] = p_occ
        t.append(a)
    return t


def to_vector(mps: Sequence[np.ndarray]) -> np.ndarray:
    """Full 2^L probability vector. Small L only; the reference path."""
    v = mps[0].reshape(D, -1)
    for a in mps[1:]:
        chi = v.shape[-1]
        v = np.tensordot(v.reshape(-1, chi), a, axes=([1], [0]))
        v = v.reshape(-1, a.shape[2])
    return v.ravel()


def _flat_blocks(mps: Sequence[np.ndarray]) -> List[np.ndarray]:
    """B_i = sum_s A_i[:, s, :] -- contraction against the all-ones covector."""
    return [a.sum(axis=1) for a in mps]


def norm1(mps: Sequence[np.ndarray]) -> float:
    """<flat|P>: the total probability. Must be 1 for a normalised state."""
    v = _flat_blocks(mps)[0]
    for b in _flat_blocks(mps)[1:]:
        v = v @ b
    return float(v.reshape(()))


def normalise(mps: List[np.ndarray]) -> float:
    z = norm1(mps)
    if z != 0:
        mps[0] = mps[0] / z
    return z


def occupancy(mps: Sequence[np.ndarray]) -> np.ndarray:
    """<n_i> at every site, by one left and one right sweep of flat blocks."""
    L = len(mps)
    B = _flat_blocks(mps)
    left = [np.ones((1, 1))]
    for i in range(L - 1):
        left.append(left[-1] @ B[i])
    right = [np.ones((1, 1))] * L
    acc = np.ones((1, 1))
    for i in range(L - 1, -1, -1):
        right[i] = acc
        acc = B[i] @ acc
    z = float((left[0] @ B[0] @ right[0]).reshape(())) if L == 1 else norm1(mps)
    out = np.empty(L)
    for i in range(L):
        n_i = mps[i][:, 1, :]
        out[i] = float((left[i] @ n_i @ right[i]).reshape(())) / z
    return out


def current(mps: Sequence[np.ndarray], bond: int, w: float) -> float:
    """J = w * P(n_bond = 1, n_{bond+1} = 0) across the given bond."""
    L = len(mps)
    B = _flat_blocks(mps)
    left = np.ones((1, 1))
    for i in range(bond):
        left = left @ B[i]
    mid = mps[bond][:, 1, :] @ mps[bond + 1][:, 0, :]
    right = np.ones((1, 1))
    for i in range(L - 1, bond + 1, -1):
        right = B[i] @ right
    return w * float((left @ mid @ right).reshape(())) / norm1(mps)


# --------------------------------------------------------------------------------------
# canonical form and truncation  (implementation requirement (b))
# --------------------------------------------------------------------------------------

def left_canonicalise(mps: List[np.ndarray]) -> None:
    """QR sweep left to right. NO truncation here -- this only fixes the gauge."""
    L = len(mps)
    for i in range(L - 1):
        cl, d, cr = mps[i].shape
        q, r = np.linalg.qr(mps[i].reshape(cl * d, cr))
        mps[i] = q.reshape(cl, d, -1)
        mps[i + 1] = np.tensordot(r, mps[i + 1], axes=([1], [0]))


def compress(mps: List[np.ndarray], chi_max: int, tol: float = 0.0) -> float:
    """Left-canonicalise, then truncate right-to-left by SVD. Returns discarded weight.

    The order matters and is requirement (b): in a left-canonical gauge the local SVD
    singular values ARE the global Schmidt values, so the discarded weight is a genuine
    error estimate. Truncating a non-canonical MPS discards the wrong thing.
    """
    left_canonicalise(mps)
    L = len(mps)
    disc = 0.0
    for i in range(L - 1, 0, -1):
        cl, d, cr = mps[i].shape
        m = mps[i].reshape(cl, d * cr)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        tot = float((s ** 2).sum())
        k = min(chi_max, len(s))
        if tol > 0 and tot > 0:
            keep = int(np.searchsorted(np.cumsum((s[::-1] ** 2))[::-1] / tot, tol,
                                       side="right"))
            k = min(k, max(1, len(s) - keep))
        k = max(1, k)
        if tot > 0:
            disc = max(disc, float((s[k:] ** 2).sum() / tot))
        mps[i] = vh[:k].reshape(k, d, cr)
        mps[i - 1] = np.tensordot(mps[i - 1], u[:, :k] * s[:k], axes=([2], [0]))
    return disc


# --------------------------------------------------------------------------------------
# gate application
# --------------------------------------------------------------------------------------

def apply_1site(mps: List[np.ndarray], i: int, gate: np.ndarray) -> None:
    mps[i] = np.einsum("st,atb->asb", gate, mps[i], optimize=True)


def apply_bond(mps: List[np.ndarray], i: int, gate4: np.ndarray,
               chi_cap: Optional[int] = None) -> None:
    """Apply a 2-site gate on bond (i, i+1) and split by SVD, keeping everything."""
    a, b = mps[i], mps[i + 1]
    cl, _, cm = a.shape
    _, _, cr = b.shape
    theta = np.tensordot(a, b, axes=([2], [0]))          # (cl, d, d, cr)
    theta = theta.transpose(1, 2, 0, 3).reshape(D * D, cl * cr)
    theta = (gate4 @ theta).reshape(D, D, cl, cr).transpose(2, 0, 1, 3)
    m = theta.reshape(cl * D, D * cr)
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    k = len(s) if chi_cap is None else min(chi_cap, len(s))
    mps[i] = u[:, :k].reshape(cl, D, k)
    mps[i + 1] = (np.diag(s[:k]) @ vh[:k]).reshape(k, D, cr)


def trotter_step(mps: List[np.ndarray], rates: np.ndarray, alpha: float, beta: float,
                 cache: GateCache, chi: int, second_order: bool = True) -> float:
    """One time step. Second order: S(dt/2) E(dt/2) O(dt) E(dt/2) S(dt/2)."""
    L = len(mps)
    ev = list(range(0, L - 1, 2))
    od = list(range(1, L - 1, 2))

    disc = 0.0

    def bulk(bonds, frac):
        # A 2-site gate can at most DOUBLE a bond (theta is (cl*d, d*cr)), so 2*chi is the
        # natural cap; 4*chi was pure waste, making every SVD twice as wide as it can need
        # to be. Compressing after each LAYER rather than once per step keeps bonds at
        # 2*chi instead of letting three layers stack them to 8*chi.
        nonlocal disc
        for i in bonds:
            apply_bond(mps, i, cache.get("bulk", float(rates[i]), frac), chi_cap=2 * chi)
        disc = max(disc, compress(mps, chi))

    def bnd(frac):
        apply_1site(mps, 0, cache.get("left", alpha, frac))
        apply_1site(mps, L - 1, cache.get("right", beta, frac))

    if second_order:
        bnd(0.5); bulk(ev, 0.5); bulk(od, 1.0); bulk(ev, 0.5); bnd(0.5)
    else:
        bnd(1.0); bulk(ev, 1.0); bulk(od, 1.0)
    normalise(mps)
    return disc


# --------------------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------------------

def solve(L: int, rates: Optional[np.ndarray] = None, alpha: float = 0.8,
          beta: float = 0.9, chi: int = 16, dt: float = 0.02,
          t_max: float = 1e9, tol_rate: float = 1e-7, check_every: int = 25,
          second_order: bool = True, p0: float = 0.0,
          max_seconds: float = 1e9, verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """Evolve to the stationary state. Converges in TIME to a residual (gate G2).

    Returns (occupancy profile, info). info carries the realised time, the residual
    |d rho/dt|, the final bond dimensions and the worst discarded weight per sweep -- the
    error estimate that must accompany every number this module produces.
    """
    rates = np.ones(L - 1) if rates is None else np.asarray(rates, dtype=float)
    if len(rates) != L - 1:
        raise ValueError(f"rates must have length L-1 = {L-1}, got {len(rates)}")
    mps = product_mps(L, p0)
    cache = GateCache(dt)
    t, steps, worst, t0 = 0.0, 0, 0.0, time.perf_counter()
    prev = occupancy(mps)
    resid = np.inf
    while t < t_max:
        for _ in range(check_every):
            worst = max(worst, trotter_step(mps, rates, alpha, beta, cache, chi,
                                            second_order))
            t += dt; steps += 1
        cur = occupancy(mps)
        resid = float(np.abs(cur - prev).max() / (check_every * dt))
        prev = cur
        if verbose:
            print(f"      t={t:9.2f} resid={resid:.3e} rho={cur.mean():.6f}")
        if resid < tol_rate:
            break
        if time.perf_counter() - t0 > max_seconds:
            break
    info = {"t": t, "steps": steps, "residual": resid, "converged": bool(resid < tol_rate),
            "chi": chi, "dt": dt, "bond_dims": [a.shape[2] for a in mps[:-1]],
            "max_bond": max([a.shape[2] for a in mps[:-1]] or [1]),
            "worst_discarded": worst, "seconds": time.perf_counter() - t0,
            "second_order": second_order}
    return prev, info


# --------------------------------------------------------------------------------------
# exact reference (sparse null space) and kinetic Monte Carlo
# --------------------------------------------------------------------------------------

def exact_stationary(L: int, rates: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Exact stationary distribution by sparse null-space solve. Shares no code with the MPS
    path -- different algorithm, different libraries -- so agreement is not circular."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    N = 1 << L
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for s in range(N):
        out = 0.0
        if not (s >> 0) & 1:
            rows.append(s | 1); cols.append(s); vals.append(alpha); out += alpha
        for i in range(L - 1):
            if ((s >> i) & 1) and not ((s >> (i + 1)) & 1):
                t = (s & ~(1 << i)) | (1 << (i + 1))
                rows.append(t); cols.append(s); vals.append(float(rates[i]))
                out += float(rates[i])
        if (s >> (L - 1)) & 1:
            rows.append(s & ~(1 << (L - 1))); cols.append(s); vals.append(beta); out += beta
        diag[s] = -out
    Q = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr() + sp.diags(diag)
    A = Q.tolil(); A[0, :] = 1.0
    b = np.zeros(N); b[0] = 1.0
    p = spla.spsolve(A.tocsr(), b)
    return np.maximum(p, 0.0) / p.sum()


def exact_occupancy(p: np.ndarray, L: int) -> np.ndarray:
    idx = np.arange(len(p))
    return np.array([float((p * ((idx >> i) & 1)).sum()) for i in range(L)])


def kmc_occupancy(L: int, rates: np.ndarray, alpha: float, beta: float,
                  t_equil: float = 200.0, t_meas: float = 800.0,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Gillespie simulation of the same process. Returns (mean occupancy, standard error).

    Time-averaged with the residence-time estimator, and the standard error comes from
    splitting the measurement window into 20 blocks -- so the comparison in G4 has an error
    bar rather than a bare number.
    """
    rng = np.random.default_rng(seed)
    n = np.zeros(L, dtype=np.int8)
    t = 0.0
    nb = 20
    acc = np.zeros((nb, L)); wt = np.zeros(nb)
    while t < t_equil + t_meas:
        props, moves = [], []
        if n[0] == 0:
            props.append(alpha); moves.append(("in", 0))
        for i in range(L - 1):
            if n[i] == 1 and n[i + 1] == 0:
                props.append(float(rates[i])); moves.append(("hop", i))
        if n[L - 1] == 1:
            props.append(beta); moves.append(("out", L - 1))
        tot = float(np.sum(props))
        if tot <= 0:
            break
        dt = rng.exponential(1.0 / tot)
        if t + dt > t_equil:
            lo = max(t, t_equil); hi = min(t + dt, t_equil + t_meas)
            if hi > lo:
                blk = min(nb - 1, int((lo - t_equil) / t_meas * nb))
                acc[blk] += n * (hi - lo); wt[blk] += (hi - lo)
        t += dt
        k = rng.choice(len(props), p=np.array(props) / tot)
        kind, i = moves[k]
        if kind == "in":
            n[0] = 1
        elif kind == "hop":
            n[i] = 0; n[i + 1] = 1
        else:
            n[i] = 0
    ok = wt > 0
    means = acc[ok] / wt[ok][:, None]
    return means.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(ok.sum())


def realistic_rates(L: int, lo: float = 0.2, hi: float = 5.0, seed: int = 0) -> np.ndarray:
    """Per-site hop rates spread log-uniformly over [lo, hi]. The inhomogeneous case."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=L - 1))
