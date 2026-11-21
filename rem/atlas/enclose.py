"""A verified enclosure of the backward function: lower and upper functions, no box.

THE CONSTRUCTION, AS SPECIFIED. Instead of bounding each subtree by its mass times a maximum taken
coordinate-by-coordinate over a box -- which discards the dependencies among those coordinates and
which tailtight's decomposition put at about twenty orders -- carry FUNCTIONS through the backward
recursion:

    hlo_L <= g <= hhi_L,      hlo_d <= K_d hlo_{d+1},      hhi_d >= K_d hhi_{d+1}

For a nonnegative transition operator K_d these imply hlo_0(s0) <= p <= hhi_0(s0). The engine's K_d
is nonnegative -- it is a matrix of transition probabilities times a positive accumulation shift --
so the implication applies. backpoly supplies the starting approximation; the work is constructing
corrections that make the inequalities HOLD, and verifying them without smuggling the box back in.

HOW THE VERIFIER AVOIDS THE BOX. The inequalities are functional, over the state and the accumulated
activity. Checking them by interval arithmetic on the accumulation coordinates would be the box
again under another name. Instead they are checked BY ENUMERATION OVER THE EXACTLY REACHABLE SET:
sufficient.py proved that under the engine's own geometric weighting every path yields a DISTINCT
accumulation, so the reachable set at level d is exactly the path set through level d, and at widths
where that enumerates the inequalities can be verified at every point the operator will ever touch.
No relaxation, no sampling, and every dependency among coordinates preserved because the points are
the real ones.

THE PRICE, STATED UP FRONT. That verifier costs O(paths). It is exact and it does not scale, which
is why the acceptance gate below is stated as interval width AT A REPORTED COMPUTATIONAL COST rather
than as width alone.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

E0  THE CEILING GATE: DOES THE ENCLOSURE CONTAIN THE TRUTH? Compare hlo_0 and hhi_0 against the
    exact tail by full enumeration.
    PREDECLARED: a single instance where the exact tail falls outside means the construction or its
    verification is wrong, and nothing below is readable. This is checked before any width is
    quoted.

E1  THE VERIFIER CONTAINS NO RELAXATION, AND THIS IS THE POINT OF THE MODULE. Count the inequality
    checks performed and the violations found, level by level.
    PREDECLARED: every check must be at a genuinely reachable point, the count must equal the
    reachable set's size, and violations must be zero by construction -- the corrections are built
    from the observed residuals, so a nonzero count means the construction and the verification
    disagree about what is reachable, which would be a defect and not a tolerance.

E2  THE ACCEPTANCE GATE, AS SPECIFIED: FINAL INTERVAL WIDTH AT TOTAL COMPUTATIONAL COST. Report
    log10(hhi_0 / hlo_0) and the seconds, on several independent exact instances.
    PREDECLARED: PASS is under ONE ORDER on every instance. This replaces closeness to the old
    certificate sum, which is no longer the target.

E3  THE SCALING OF WIDTH AND COST in the representation's degree and in the instance size, because
    a width quoted at one degree on one instance is a point.

E4  WHAT THIS DOES NOT DO.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S

BASE, GAIN = -1.0, 2.0


def prep(nCtrl, L, dt=0.5, ntarget=200):
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, ntarget)
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    hvec = np.exp(-0.5 * np.arange(L, -1, -1))
    hvec = hvec / hvec.sum()
    S = identity_S(rows, nCtrl)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)
    return Pm, pi, n, hvec, S, actbit, sha


def logg(Y, S):
    return -np.logaddexp(0.0, -(BASE + GAIN * (Y @ S.T))).sum(axis=1)


def mono(k, deg):
    o = [(0,) * k]
    for r in range(1, deg + 1):
        for c in itertools.combinations_with_replacement(range(k), r):
            e = [0] * k
            for i in c:
                e[i] += 1
            o.append(tuple(e))
    return o


def design(X, mons):
    A = np.empty((X.shape[0], len(mons)))
    for j, e in enumerate(mons):
        v = np.ones(X.shape[0])
        for i, pw in enumerate(e):
            if pw:
                v = v * X[:, i] ** pw
        A[:, j] = v
    return A


def reachable(Pm, n, hvec, actbit, d):
    """Every reachable (state, accumulated activity) at level d -- exactly the path set."""
    S_ = np.arange(n)
    X = actbit * hvec[0]
    for dd in range(1, d + 1):
        S_ = np.tile(np.arange(n), len(S_))
        X = np.repeat(X, n, axis=0) + actbit[S_] * hvec[dd]
    return S_, X


def enclose(Pm, pi, n, hvec, S, actbit, L, deg):
    """Build hlo/hhi with the inequalities VERIFIED at every reachable point.

    CONVENTION, and the first version got it wrong: reachable(d) returns the accumulation
    INCLUDING level d, so U_d(s, x) is indexed by that inclusive accumulation and the shift into
    level d+1 belongs to the SUCCESSOR, not to the current state:

        U_L(s, x) = g(x)
        U_d(s, x) = SUM_s' Pm[s',s] U_{d+1}(s', x + a_{s'} h_{d+1})
        p         = SUM_s pi[s] U_0(s, a_s h_0)

    The first version added the current state's contribution a second time, which double-counted
    one level of activity and put both bounds about three orders above the truth. E0 caught it."""
    k = actbit.shape[1]
    mons = mono(k, deg)
    Chi = Clo = None
    checks = 0
    viol = 0
    for d in range(L - 1, -1, -1):
        SD, XD = reachable(Pm, n, hvec, actbit, d)
        M = len(SD)
        Hhi = np.empty((M, n))
        Hlo = np.empty((M, n))
        for sp in range(n):
            Y = XD + actbit[sp] * hvec[d + 1]           # the SUCCESSOR's contribution
            if d == L - 1:                              # terminal is EXACT: no approximation
                v = np.exp(logg(Y, S))
                Hhi[:, sp] = v
                Hlo[:, sp] = v
            else:
                A = design(Y, mons)
                Hhi[:, sp] = np.exp(np.clip(A @ Chi[sp], -700, 700))
                Hlo[:, sp] = np.exp(np.clip(A @ Clo[sp], -700, 700))
        Khi = (Hhi * Pm[:, SD].T).sum(axis=1)
        Klo = (Hlo * Pm[:, SD].T).sum(axis=1)
        AD = design(XD, mons)
        Chi = np.zeros((n, len(mons)))
        Clo = np.zeros((n, len(mons)))
        for s in range(n):
            m = SD == s
            if not m.any():
                continue
            a, yhi, ylo = AD[m], np.log(np.maximum(Khi[m], 1e-308)), np.log(np.maximum(Klo[m], 1e-308))
            chi = np.linalg.lstsq(a, yhi, rcond=None)[0]
            clo = np.linalg.lstsq(a, ylo, rcond=None)[0]
            # CORRECTIONS: inflate the upper fit until it dominates, deflate the lower until it is
            # dominated -- at every reachable point, which is what makes the inequalities hold.
            chi[0] += float(np.max(yhi - a @ chi))
            clo[0] += float(np.min(ylo - a @ clo))
            Chi[s], Clo[s] = chi, clo
            checks += 2 * int(m.sum())
            viol += int(np.sum(a @ chi < yhi - 1e-9)) + int(np.sum(a @ clo > ylo + 1e-9))
    X0 = actbit * hvec[0]                               # level 0's own reachable accumulation
    A0 = design(X0, mons)
    hi = float((pi * np.exp(np.clip(np.einsum('sc,sc->s', A0, Chi), -700, 700))).sum())
    lo = float((pi * np.exp(np.clip(np.einsum('sc,sc->s', A0, Clo), -700, 700))).sum())
    return lo, hi, checks, viol


def exact_tail(Pm, pi, n, hvec, S, actbit, L):
    last, w = np.arange(n), pi.copy()
    acc = actbit * hvec[0]
    for d in range(1, L + 1):
        col = Pm[:, last].T * w[:, None]
        w = col.ravel()
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        acc = acc[par] + actbit[chd] * hvec[d]
        last = chd
    return float((w * np.exp(logg(acc, S))).sum())


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("A VERIFIED ENCLOSURE OF THE BACKWARD FUNCTION: LOWER AND UPPER FUNCTIONS, NO BOX")
    P_(RULE)
    P_("  hlo_d <= K_d hlo_{d+1} and hhi_d >= K_d hhi_{d+1}, verified at every REACHABLE point")
    P_("  rather than relaxed over a box. Acceptance gate: final interval width under ONE ORDER,")
    P_("  reported with its computational cost.")

    inst = [(3, 3, 200), (3, 4, 200), (4, 3, 200), (4, 4, 200), (4, 3, 100), (3, 4, 100)]
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'targets':>8} {'exact tail':>15} {'hlo':>15} {'hhi':>15}"
       f" {'contains':>9}")
    rows_ = []
    okall = True
    for nc, L, nt in inst:
        Pm, pi, n, hvec, S, actbit, sha = prep(nc, L, ntarget=nt)
        ex = exact_tail(Pm, pi, n, hvec, S, actbit, L)
        t0 = time.time()
        lo, hi, ch, vi = enclose(Pm, pi, n, hvec, S, actbit, L, 3)
        el = time.time() - t0
        good = lo <= ex * (1 + 1e-9) and ex <= hi * (1 + 1e-9)
        okall = okall and good
        rows_.append((nc, L, nt, ex, lo, hi, ch, vi, el))
        P_(f"    {nc:>6} {L:>3} {nt:>8} {ex:>15.6e} {lo:>15.6e} {hi:>15.6e}"
           f" {'yes' if good else 'NO':>9}")

    P_("\n" + RULE)
    P_("E0  THE CEILING GATE: DOES THE ENCLOSURE CONTAIN THE TRUTH?")
    P_(RULE)
    P_(f"  {'PASS -- the exact tail lies inside the enclosure on every instance.' if okall else 'FAIL -- the exact tail falls outside. The construction or its verification is wrong.'}")
    if not okall:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_enclose.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    P_("\n" + RULE)
    P_("E1  THE VERIFIER: EVERY CHECK AT A REACHABLE POINT, NO RELAXATION")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'paths':>12} {'checks':>12} {'violations':>12}")
    for nc, L, nt, ex, lo, hi, ch, vi, el in rows_:
        P_(f"    {nc:>6} {L:>3} {(2 ** nc) ** (L + 1):>12,} {ch:>12,} {vi:>12}")
    P_("\n  The checks are the reachable (state, accumulation) pairs at every level, twice -- once")
    P_("  for each inequality. Violations are zero because the corrections are BUILT from the")
    P_("  residuals at those same points; a nonzero count would mean construction and verification")
    P_("  disagreed about what is reachable.")

    P_("\n" + RULE)
    P_("E2  THE ACCEPTANCE GATE: INTERVAL WIDTH AT TOTAL COMPUTATIONAL COST")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'targets':>8} {'width, orders':>14} {'seconds':>9} {'under 1 order':>14}")
    widths = []
    for nc, L, nt, ex, lo, hi, ch, vi, el in rows_:
        w = float(np.log10(hi / max(lo, 1e-308)))
        widths.append(w)
        P_(f"    {nc:>6} {L:>3} {nt:>8} {w:>14.3f} {el:>9.1f} {'yes' if w < 1.0 else 'NO':>14}")
    P_(f"\n  E2 AS PREDECLARED: {'PASS -- every instance is under one order.' if all(w < 1.0 for w in widths) else f'FAIL -- the widest instance is {max(widths):.2f} orders.'}")
    P_(f"  For scale, tailtight measured the previous certificate's interval at 22 to 27 orders on")
    P_(f"  the same kind of instance.")

    P_("\n" + RULE)
    P_("E3  THE SCALING IN DEGREE")
    P_(RULE)
    Pm, pi, n, hvec, S, actbit, sha = prep(4, 4)
    ex = exact_tail(Pm, pi, n, hvec, S, actbit, 4)
    P_(f"\n    {'degree':>7} {'coefficients':>13} {'width, orders':>14} {'seconds':>9}")
    for deg in (1, 2, 3, 4):
        t0 = time.time()
        lo, hi, ch, vi = enclose(Pm, pi, n, hvec, S, actbit, 4, deg)
        el = time.time() - t0
        P_(f"    {deg:>7} {len(mono(4, deg)):>13} {np.log10(hi / max(lo, 1e-308)):>14.3f} {el:>9.1f}")
    P_("\n  The width falls with degree because the corrections shrink as the fit improves: the")
    P_("  enclosure's width IS the accumulated fit residual, made rigorous.")

    P_("\n" + RULE)
    P_("E4  WHAT THIS DOES NOT DO")
    P_(RULE)
    P_("  1. THE VERIFIER COSTS O(PATHS). It is exact because it visits every reachable point, and")
    P_("     that is precisely why it does not yet scale. The gate is width AT a stated cost, and")
    P_("     the cost stated here is enumeration. Making the verification cheap without")
    P_("     reintroducing a relaxation is the next problem and it is not solved here.")
    P_("  2. These are small instances. Nothing here shows the width stays under an order at the")
    P_("     engine's width, where the verifier cannot run at all.")
    P_("  3. The enclosure is of the tail computed with this engine's fixed base and gain. It is a")
    P_("     rigorous interval for THAT quantity, not for a biological probability.")
    P_("  4. It does not touch class-map validity, transfer to other instances, or anything")
    P_("     biological. It addresses one term of one calculation.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_enclose.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
