"""Closed-form partition functions for ring and circulant structure.

A ring is the one topology where elimination's O(n) sweep is not needed at all. Two
distinct closed forms, and they are for different models -- conflating them is easy:

  DISCRETE RING, homogeneous transfer matrix T (d x d) on every edge:
      Z = tr(T^n) = sum_i lambda_i^n           lambda = eigenvalues of T
  cost O(d^3 + d log n), INDEPENDENT of n. A million sites is as cheap as ten.

  GAUSSIAN CIRCULANT, translation-invariant quadratic form with first row c:
      eigenvalues = FFT(c),  log Z = (n/2) log(2 pi) - (1/2) sum_k log(eig_k)
  cost O(n log n) via one FFT, because a circulant matrix is diagonal in the Fourier basis.

Both are verified against brute force / dense linear algebra by verify().
"""
from __future__ import annotations

import itertools
import numpy as np


def ring_logZ_transfer(logT: np.ndarray, n: int) -> float:
    """log Z for a homogeneous ring of n sites with log-transfer-matrix logT.

    Z = tr(T^n). Computed from eigenvalues in log space so n = 10^6 does not overflow.
    T may be non-symmetric; eigenvalues can be complex, and the imaginary parts cancel."""
    T = np.exp(np.asarray(logT, dtype=float))
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("logT must be square")
    ev = np.linalg.eigvals(T)
    a = np.max(np.abs(ev))
    if a == 0:
        return -np.inf
    z = np.sum((ev / a) ** n)                 # scaled so no term overflows
    val = np.real(z)
    if val <= 0:
        return -np.inf
    return float(n * np.log(a) + np.log(val))


def ring_logZ_bruteforce(logT: np.ndarray, n: int) -> float:
    d = logT.shape[0]
    tot = []
    for cfg in itertools.product(range(d), repeat=n):
        s = 0.0
        for i in range(n):
            s += logT[cfg[i], cfg[(i + 1) % n]]
        tot.append(s)
    tot = np.asarray(tot)
    m = tot.max()
    return float(m + np.log(np.exp(tot - m).sum()))


def circulant_eigenvalues(first_row: np.ndarray) -> np.ndarray:
    """Eigenvalues of the circulant matrix whose first row is `first_row`, via one FFT."""
    return np.fft.fft(np.asarray(first_row, dtype=float))


def circulant_gaussian_logZ(first_row: np.ndarray) -> float:
    """log Z of exp(-1/2 x^T C x) for circulant C, from FFT eigenvalues."""
    ev = np.real(circulant_eigenvalues(first_row))
    if np.any(ev <= 0):
        raise ValueError("circulant matrix is not positive definite; min eig "
                         f"{ev.min():.3e}")
    n = len(first_row)
    return float(0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum(np.log(ev)))


def circulant_gaussian_logZ_dense(first_row: np.ndarray) -> float:
    """Same quantity by dense linear algebra. O(n^3); verification only."""
    n = len(first_row)
    C = np.stack([np.roll(np.asarray(first_row, dtype=float), i) for i in range(n)])
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("dense circulant not positive definite")
    return float(0.5 * n * np.log(2 * np.pi) - 0.5 * logdet)


def verify(verbose: bool = True) -> dict:
    rng = np.random.default_rng(0)
    e_ring = 0.0
    for _ in range(12):
        d = int(rng.integers(2, 4))
        n = int(rng.integers(3, 8))
        logT = rng.normal(size=(d, d))
        e_ring = max(e_ring, abs(ring_logZ_transfer(logT, n)
                                 - ring_logZ_bruteforce(logT, n)))
    e_circ = 0.0
    for _ in range(12):
        n = int(rng.integers(4, 40))
        c = rng.normal(size=n) * 0.1
        c[0] = abs(c).sum() + 1.0                  # diagonally dominant -> positive definite
        c = (c + np.roll(c[::-1], 1)) / 2          # symmetric circulant
        e_circ = max(e_circ, abs(circulant_gaussian_logZ(c)
                                 - circulant_gaussian_logZ_dense(c)))
    if verbose:
        print(f"  rem.circulant.verify")
        print(f"    max |ring transfer-matrix logZ - brute force|  {e_ring:.3e}")
        print(f"    max |circulant FFT logZ - dense slogdet|       {e_circ:.3e}")
    return {"max_err_ring": e_ring, "max_err_circulant": e_circ}
