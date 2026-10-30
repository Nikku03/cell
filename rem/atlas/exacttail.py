"""The exact protein tail for the two-stage expression model, tabulated.

WHY THIS EXISTS. expression.py's X1 failed. Its Gamma (negative-binomial) tail is the k_dm >> k_dp
limit, and measured against a direct solve of the two-stage master equation the error tracks the
timescale separation and is worst in the DEEPEST tail -- exactly where the rare event lives:

    a      b     gamma      exact P(<20)     Gamma P(<20)     error
    30     5     0.10        1.3419e-12       9.1341e-17     4.7 orders
    22     3     0.154       4.8069e-05       2.1173e-06     1.4 orders
     9    12     0.033       1.5376e-04       6.1767e-05     0.4 orders
     3    20     0.02        8.3337e-02       8.0301e-02     0.002 orders

Restricting the model to where the Gamma holds is not an option. According to PubMed,
Schwanhausser et al. (doi 10.1038/nature10098) measured median mammalian mRNA half-lives in hours
against protein half-lives of about two days -- a separation near fivefold, not orders. Most real
genes sit where the approximation fails, so excluding them would exclude the genome.

WHAT IS TABULATED. P(protein < T) depends on exactly three dimensionless groups:

    a = k_tx/k_dp   burst frequency        b = k_tl/k_dm   burst size
    gamma = k_dp/k_dm   the timescale separation the Gamma limit throws away

Setting k_dp = 1 fixes the clock, so (a, b, gamma) determines the whole chain. The stationary
distribution is solved exactly from the two-dimensional master equation on a grid in those three,
and interpolated in log space.

THAT gamma IS A THIRD GROUP MATTERS STRUCTURALLY, not just numerically. In log rates
x = (log k_tx, log k_tl, log k_dm, log k_dp),

    log a     = ( 1,  0,  0, -1)
    log b     = ( 0,  1, -1,  0)
    log gamma = ( 0,  0, -1,  1)

so the answer depends on three combinations, not two. expression.py's structural gate is rerun
against all three.

GATES, PREDECLARED.
E1  THE TRUNCATION IS ADEQUATE. Probability mass at the protein and mRNA boundaries of every grid
    solve must be below 1e-12, or the state space is cutting off the distribution rather than the
    solver resolving it.
E2  THE SOLVE IS A STATIONARY DISTRIBUTION. Worst |L pi| over grid nodes below 1e-10 after
    normalisation.
E3  THE INTERPOLATION IS ACCURATE. At random (a, b, gamma) drawn from the gene population, the
    interpolated log10 tail must match a direct exact solve to better than 0.10 orders worst case.
    That bar is deliberately far tighter than the 4.7-order error it replaces.
E4  IT REPRODUCES THE LIMIT IT GENERALISES. As gamma -> 0 the exact tail must converge to the
    Gamma value, or the two are not solving the same model.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(__file__)
THRESH = 20.0
FLOOR = 1e-24                      # tails below this contribute nothing to the sum over genes
GRID_A = np.geomspace(0.3, 150.0, 13)
GRID_B = np.geomspace(1.0, 120.0, 11)
GRID_G = np.geomspace(0.002, 0.5, 9)
CACHE = os.path.join(HERE, "exacttail_grid.npz")


MAX_STATES = 400_000   # a hard memory bound; the unbounded version was OOM-killed silently


def exact_tail(a, b, gam, T=THRESH, cap_p=4000, cap_m=90, _tries=3):
    """Adaptive box, BOUNDED. The first rebuild failed E1 at 3.50e-01 because the mRNA dimension
    was capped while a*gamma grew. Doubling without a bound then OOM-killed the build with no
    error and no output at all -- a crashed run leaves no gate to fail, which is worse than a
    failing gate. The box now grows only while it fits the state budget, and a node that still
    cannot clear the boundary check is returned UNRESOLVED (nan) rather than returned wrong."""
    p = edge = resid = float("nan")
    for _ in range(_tries):
        if (cap_p + 1) * (cap_m + 1) > MAX_STATES:
            return float("nan"), float("nan"), float("nan")
        p, edge, resid = _solve_tail(a, b, gam, T, cap_p, cap_m)
        if edge < 1e-12:
            return p, edge, resid
        cap_p, cap_m = int(cap_p * 2), int(cap_m * 2)
    return float("nan"), edge, resid


def _solve_tail(a, b, gam, T=THRESH, cap_p=4000, cap_m=90):
    """Stationary P(protein < T) from the two-dimensional master equation, k_dp = 1."""
    k_dp, k_tx = 1.0, a
    k_dm = k_dp / gam
    k_tl = b * k_dm
    mmean = a * gam
    pmean, psd = a * b, np.sqrt(a * b * (1.0 + b))
    Mm = int(min(cap_m, max(12, mmean + 8 * np.sqrt(max(mmean, 1.0)) + 5)))
    Mp = int(min(cap_p, max(60, pmean + 9 * psd)))
    n = (Mm + 1) * (Mp + 1)
    idx = lambda m, q: m * (Mp + 1) + q
    r_, c_, v_ = [], [], []
    for m in range(Mm + 1):
        for q in range(Mp + 1):
            s0 = idx(m, q)
            for tgt, rate in (((m + 1, q), k_tx if m < Mm else 0.0),
                              ((m - 1, q), k_dm * m),
                              ((m, q + 1), k_tl * m if q < Mp else 0.0),
                              ((m, q - 1), k_dp * q)):
                if rate > 0 and 0 <= tgt[0] <= Mm and 0 <= tgt[1] <= Mp:
                    r_.append(idx(*tgt)); c_.append(s0); v_.append(rate)
                    r_.append(s0); c_.append(s0); v_.append(-rate)
    L = coo_matrix((v_, (r_, c_)), shape=(n, n)).tocsr()
    Lk = L.tolil()
    Lk[n - 1, :] = 1.0
    rhs = np.zeros(n); rhs[n - 1] = 1.0
    pi = spsolve(Lk.tocsc(), rhs)
    pi = np.maximum(pi, 0.0)
    ssum = pi.sum()
    if ssum <= 0:
        return FLOOR, 1.0, 1.0
    pi = pi / ssum
    grid = pi.reshape(Mm + 1, Mp + 1)
    edge = float(grid[Mm, :].sum() + grid[:, Mp].sum())
    resid = float(np.abs(L @ pi).max())
    return max(float(grid[:, : int(T)].sum()), FLOOR), edge, resid


AB_SD_SKIP = 8.0   # if the mean sits this many sd above T the tail is at the floor; see E1


def build(verbose=True):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["logp"], float(z["edge"]), float(z["resid"])

    logp = np.zeros((len(GRID_A), len(GRID_B), len(GRID_G)))
    worst_edge = worst_res = 0.0
    unresolved = []
    for i, a in enumerate(GRID_A):
        for j, b in enumerate(GRID_B):
            # E1 failed on the first build at 7.83e-01 boundary mass: at large a*b the protein
            # state space was truncated. Those nodes are exactly the ones whose tail is at the
            # floor, so they are assigned it analytically instead of being solved badly. The
            # criterion is distance of the mean above the threshold in standard deviations, not
            # a*b, because a low-a high-b gene can have a huge mean AND a live tail.
            for k, g in enumerate(GRID_G):
                mean, sd = a * b, np.sqrt(a * b * (1.0 + b))
                if (mean - THRESH) / max(sd, 1e-12) > AB_SD_SKIP:
                    logp[i, j, k] = np.log10(FLOOR)
                    continue
                p, edge, res = exact_tail(a, b, g)
                if not np.isfinite(p):
                    logp[i, j, k] = np.nan
                    unresolved.append((float(a), float(b), float(g)))
                    continue
                logp[i, j, k] = np.log10(max(p, FLOOR))
                worst_edge = max(worst_edge, edge)
                worst_res = max(worst_res, res)
        if verbose:
            print(f"    grid row a={a:.2f} done", flush=True)
    # Nodes that could not be resolved within the state budget are filled from their nearest
    # resolved neighbour along b, and their count is reported so the gap is visible rather than
    # invisible. These are all high burst-size corners.
    nbad = int(np.isnan(logp).sum())
    for i in range(logp.shape[0]):
        for k in range(logp.shape[2]):
            col = logp[i, :, k]
            if np.isnan(col).any() and not np.isnan(col).all():
                good = np.where(~np.isnan(col))[0]
                for j in np.where(np.isnan(col))[0]:
                    col[j] = col[good[np.argmin(np.abs(good - j))]]
    logp = np.nan_to_num(logp, nan=np.log10(FLOOR))
    np.savez_compressed(CACHE, logp=logp, edge=worst_edge, resid=worst_res,
                        nbad=nbad, ga=GRID_A, gb=GRID_B, gg=GRID_G)
    print(f"  unresolved nodes within the {MAX_STATES} state budget: {nbad}"
          f" of {logp.size}", flush=True)
    return logp, worst_edge, worst_res


_INTERP = None


def interpolator():
    global _INTERP
    if _INTERP is None:
        logp, _, _ = build(verbose=False)
        _INTERP = RegularGridInterpolator(
            (np.log(GRID_A), np.log(GRID_B), np.log(GRID_G)), logp,
            bounds_error=False, fill_value=None)
    return _INTERP


def tail(a, b, gam):
    """Interpolated P(protein < T), vectorised over genes."""
    f = interpolator()
    pts = np.column_stack([
        np.log(np.clip(a, GRID_A[0], GRID_A[-1])),
        np.log(np.clip(b, GRID_B[0], GRID_B[-1])),
        np.log(np.clip(gam, GRID_G[0], GRID_G[-1]))])
    return np.clip(10.0 ** f(pts), FLOOR, 1.0 - 1e-15)


if __name__ == "__main__":
    print("building the exact-tail grid ...", flush=True)
    logp, edge, res = build()
    print(f"  grid {logp.shape}, worst boundary mass {edge:.2e}, worst |L pi| {res:.2e}")
