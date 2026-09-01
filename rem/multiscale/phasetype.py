"""Deliverable 7 (spec section 8): processive machines are not memoryless.

A ribosome takes about 25 s to make a protein, and it takes that long NEARLY
DETERMINISTICALLY -- it is a fixed-length walk down a transcript, not a coin flipped every
instant. The CME's exponential waiting time gives the same MEAN with a completely different
shape: an exponential's standard deviation equals its mean, while an Erlang of k phases has
standard deviation mean/sqrt(k). Replacing the walk by a coin flip therefore injects
variance that is not there, and injected variance lands almost entirely in the tail.

The fix is not to explode the machine into hundreds of states. It is a phase-type block:
k internal phases each advancing at rate k/tau, emitting on completion and resetting. The
mean holding time is tau for every k, so the first moment is untouched by construction, and
k is the only knob.

=================================================================================================
GATES, PREDECLARED.
=================================================================================================

G8a  Same mean occupancy: the mean protein number must be 6.000 at EVERY step count,
     identical to four decimals. This is a structural identity -- production is 1/tau per
     completion whatever the phase count -- so a discrepancy here is an implementation bug,
     not a modelling difference, and it is the first thing checked for that reason.

G8b  Fano factor must differ by about 19% between exponential (k=1) and the deterministic
     limit.

G8c  A rare-event probability must differ by about 35.5x. Same mean, same model, one
     modelling choice about waiting-time shape.

G8d  Convergence: 3 Erlang phases must capture most of the correction and 8 nearly all, so
     that a small block suffices and 300 states are never needed. Quantified here as the
     fraction of the k=1 -> k=inf change in log10 P(rare) already achieved at k = 3 and k = 8.

G8e  NEGATIVE CONTROL (standing rule 2). With the protein made non-rare -- the question asked
     where the mass is instead of in the tail -- the k-dependence must nearly vanish. If a
     quantity in the bulk moves as much as the tail quantity, the testbed is measuring
     something other than waiting-time shape.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def erlang_machine(k: int, tau: float, gamma: float, Pmax: int):
    """Stationary law of (phase, protein) for a k-phase machine with mean cycle tau.

    Phase advances at rate k/tau. Completing the last phase emits one protein and returns to
    phase 0. Protein degrades at gamma*P. k = 1 is the ordinary exponential CME.
    """
    rate = k / tau
    n = k * (Pmax + 1)

    def idx(ph, p):
        return ph * (Pmax + 1) + p

    rows, cols, vals = [], [], []
    diag = np.zeros(n)

    def add(i, j, r):
        if r <= 0:
            return
        rows.append(i); cols.append(j); vals.append(r); diag[i] -= r

    for ph in range(k):
        for p in range(Pmax + 1):
            i = idx(ph, p)
            if ph < k - 1:
                add(i, idx(ph + 1, p), rate)               # advance a phase
            else:
                if p + 1 <= Pmax:
                    add(i, idx(0, p + 1), rate)            # emit and reset
                else:
                    add(i, idx(0, p), rate)                # emit into a full buffer
            if p > 0:
                add(i, idx(ph, p - 1), gamma * p)

    Q = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() + sp.diags(diag)
    A = Q.T.tolil()
    A[0, :] = 1.0
    b = np.zeros(n); b[0] = 1.0
    pr = spla.spsolve(A.tocsr(), b)
    pr = np.maximum(pr, 0.0)
    pr /= pr.sum()
    pp = pr.reshape(k, Pmax + 1).sum(axis=0)
    return pp


def moments(pp: np.ndarray):
    x = np.arange(len(pp))
    m = float((x * pp).sum())
    v = float((x * x * pp).sum() - m * m)
    return m, v, (v / m if m > 0 else float("nan"))


def verify(verbose: bool = True) -> dict:
    TAU, GAMMA, PMAX = 1.0, 1.0 / 6.0, 60      # mean protein = (1/tau)/gamma = 6
    THRESH = 30
    ks = [1, 2, 3, 5, 8, 16, 32, 64]
    print("=" * 92)
    print("G8  EXPONENTIAL vs ERLANG at identical mean occupancy")
    print("=" * 92)
    print(f"  {'k':>4s} {'mean':>10s} {'Fano':>9s} {'P(P>=30)':>12s} "
          f"{'ratio vs k=1':>13s}")
    res = {}
    for k in ks:
        pp = erlang_machine(k, TAU, GAMMA, PMAX)
        m, v, fano = moments(pp)
        P = float(pp[THRESH:].sum())
        res[k] = (m, fano, P)
        r = res[1][2] / P if P > 0 else float("nan")
        print(f"  {k:>4d} {m:>10.4f} {fano:>9.4f} {P:>12.4e} {r:>13.2f}")

    means = [res[k][0] for k in ks]
    g8a = max(means) - min(means) < 1e-4
    print(f"\n  G8a  mean identical to 4 dp across all k: spread "
          f"{max(means)-min(means):.2e}   {'PASS' if g8a else 'FAIL'}")

    f1, fL = res[1][1], res[ks[-1]][1]
    d_fano = 100.0 * (f1 - fL) / f1
    g8b = abs(d_fano - 19.0) < 6.0
    print(f"  G8b  Fano k=1 {f1:.4f} -> k={ks[-1]} {fL:.4f}  = {d_fano:.1f}% different "
          f"(expected ~19%)   {'PASS' if g8b else 'FAIL'}")

    ratio = res[1][2] / res[ks[-1]][2]
    g8c = 0.5 < ratio / 35.5 < 2.0
    print(f"  G8c  P(rare) ratio k=1 vs k={ks[-1]}: {ratio:.1f}x (expected ~35.5x)   "
          f"{'PASS' if g8c else 'FAIL'}")

    l1, lL = np.log10(res[1][2]), np.log10(res[ks[-1]][2])
    frac = lambda k: (np.log10(res[k][2]) - l1) / (lL - l1)
    f3, f8 = frac(3), frac(8)
    g8d = f3 > 0.5 and f8 > 0.8
    print(f"  G8d  fraction of the full log10 correction captured: k=3 {100*f3:.0f}%, "
          f"k=8 {100*f8:.0f}%   {'PASS' if g8d else 'FAIL'}")
    print(f"       (the spec's claim: 3 phases captures most, 8 nearly all -- so a small "
          f"block suffices)")

    print("\n" + "=" * 92)
    print("G8e  NEGATIVE CONTROL -- ask the question where the mass is, not in the tail")
    print("=" * 92)
    bulk = []
    for k in (1, 64):
        pp = erlang_machine(k, TAU, GAMMA, PMAX)
        bulk.append(float(pp[:7].sum()))
    db = 100.0 * abs(bulk[0] - bulk[1]) / bulk[0]
    dt = 100.0 * abs(res[1][2] - res[64][2]) / res[1][2]
    g8e = db < 0.05 * dt
    print(f"  P(protein <= 6): k=1 {bulk[0]:.6f}  k=64 {bulk[1]:.6f}   {db:.2f}% apart")
    print(f"  P(protein >= 30): {dt:.2f}% apart")
    print(f"  bulk moves {dt/db:.0f}x less than the tail   G8e "
          f"{'PASS' if g8e else 'FAIL'}")
    print("  (standing rule 5: report error at the tail, never in L2 -- this is why)")
    return {"res": res, "G8a": g8a, "G8b": g8b, "G8c": g8c, "G8d": g8d, "G8e": g8e}


if __name__ == "__main__":
    verify()
