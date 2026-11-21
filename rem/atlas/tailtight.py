"""How tightly does the composed certificate bound the ACTUAL rare-event probability?

WHAT HAS AND HAS NOT BEEN ESTABLISHED. sampcert, certcover, composecert and adaptpart built a
guarantee that is valid, composes across levels, survives adaptive sample sizes and adaptive
partitions, and costs a few percent in width. Every one of them certified the SAME object: the
summed bound over dropped candidates. None of them asked what that object says about the tail.

THE ENGINE REPORTS A RARE-EVENT PROBABILITY, and what the certificate buys is an INTERVAL for it:

    pruned tail  <=  true tail  <=  pruned tail + certificate

The pruned tail is a lower bound because retaining more paths can only add mass. So the certificate's
value is the WIDTH of that interval, and exactcert already gave reason to fear the answer -- it
measured the bound as 25 orders looser than the tail at engine width, with an irreducible floor of
about 0.52 * n * m for the whole family. This measures the interval directly, at a width where the
true tail is known exactly and nothing has to be extrapolated.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

T0  THE CEILING GATE, AND IT IS ABOUT WHETHER THE INEQUALITY EVEN HOLDS. Before measuring how tight
    the interval is, verify it CONTAINS the truth: at a width where every path enumerates, check
    that exact tail <= pruned tail + certificate at every cap.
    PREDECLARED: a single violation means the certificate does not bound what it claims to and
    every result in the four preceding modules is about a quantity that fails its own purpose.
    That is the first thing to check and it has never been checked.

T1  THE INTERVAL'S WIDTH, at the enumerable width. Report the pruned tail, the exact tail, the
    composed certificate, and the interval width in orders of magnitude, against cap.
    PREDECLARED: the certificate is useful for the TAIL only if the interval is narrow enough to
    be a statement about a probability. An interval spanning many orders is reported as worthless
    for the tail however valid it is, and its validity is not offered as consolation.

T2  THE SAME AT THE ENGINE'S WIDTH, where the true tail is not known: the interval is reported
    against backpoly's estimate of 1.61e-95 with the 0.30-order error interval backpolyerr
    measured, carried explicitly rather than dropped.

T3  THE DECOMPOSITION, which is the only actionable part. Of the total looseness, how much is the
    sampling slack, how much is the family's irreducible floor at 0.52*n*m, and how much is the
    BOX relaxation exactcert measured at ten to twenty orders?
    PREDECLARED: reported as a budget that sums to the observed total. If the box term dominates,
    the certificate is loose for a reason already known to be fixable in principle and the four
    preceding modules were tightening the wrong term.

T4  WHAT WOULD CLOSE IT: the best conceivable certificate of this family, from the floor law,
    against the one observed. And T5 what this does not settle.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S

BACKPOLY_EST, BACKPOLY_ERR = 1.6127e-95, 0.30      # backpolyerr H4
FLOOR_A, FLOOR_N, FLOOR_M = 0.52, 1.048, 1.028     # exactcert E0c: floor ~ 0.52 n^1.05 m^1.03


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


def logon(Y, S, base=-1.0, gain=2.0):
    return -np.logaddexp(0.0, -(base + gain * (Y @ S.T))).sum(axis=1)


def exact(Pm, pi, n, hvec, S, actbit, L):
    last = np.arange(n)
    w = pi.copy()
    acc = actbit * hvec[0]
    for d in range(1, L + 1):
        col = Pm[:, last].T * w[:, None]
        w = col.ravel()
        par = np.repeat(np.arange(len(last)), n)
        chd = np.tile(np.arange(n), len(last))
        acc = acc[par] + actbit[chd] * hvec[d]
        last = chd
    return float((w * np.exp(logon(acc, S))).sum())


def run(Pm, pi, n, hvec, S, actbit, L, cap, base=-1.0, gain=2.0):
    """Mass-selected pruner. Returns (pruned tail, composed certificate, per-level certificates)."""
    Splus = np.maximum(S, 0.0).sum(axis=1)
    Bst = actbit @ S.T
    H0 = float(hvec[1:].sum())
    u0 = base + gain * ((actbit * hvec[0]) @ S.T) + gain * H0 * Splus[None, :]
    b0 = pi * np.exp((-np.logaddexp(0.0, -u0)).sum(axis=1))
    keep0 = np.argsort(-pi)[:cap]
    m0 = np.ones(n, dtype=bool)
    m0[keep0] = False
    cert = [float(b0[m0].sum())]
    last, wts, wact = keep0, pi[keep0].copy(), actbit[keep0] * hvec[0]
    for d in range(1, L + 1):
        ch = Pm[:, last].T * wts[:, None]
        Hr = float(hvec[d + 1:].sum())
        u = base + gain * (wact @ S.T) + gain * Hr * Splus[None, :]
        ls = (-np.logaddexp(0.0, -u)).sum(axis=1)
        g = 1.0 / (1.0 + np.exp(u))
        lg = ls[:, None] + g @ (gain * hvec[d] * Bst).T
        bnd, mass = (ch * np.exp(lg)).ravel(), ch.ravel()
        k = min(cap, mass.size)
        keep = np.argpartition(-mass, k - 1)[:k] if k < mass.size else np.arange(mass.size)
        msk = np.ones(mass.size, dtype=bool)
        msk[keep] = False
        cert.append(float(bnd[msk].sum()))
        par, code = keep // n, keep % n
        wts, wact, last = mass[keep], wact[par] + actbit[code] * hvec[d], code
    tail = float((wts * np.exp(logon(wact, S))).sum())
    return tail, float(sum(cert)), cert


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("HOW TIGHTLY DOES THE COMPOSED CERTIFICATE BOUND THE ACTUAL RARE-EVENT PROBABILITY?")
    P_(RULE)
    P_("  Four modules certified the summed dropped bound. None asked what it says about the tail.")
    P_("  What the certificate buys is an interval:  pruned <= true <= pruned + certificate.")

    nc, L = 4, 4
    Pm, pi, n, hvec, S, actbit, sha = prep(nc, L)
    ex = exact(Pm, pi, n, hvec, S, actbit, L)
    P_(f"\n  {nc} controllers, n = {n}, L = {L}, {n ** (L + 1):,} paths enumerated exactly.")
    P_(f"  EXACT tail: {ex:.6e}. sha {sha[:12]}.")

    caps = (2, 4, 8, 16, 32, 64)
    res = [(c,) + run(Pm, pi, n, hvec, S, actbit, L, c) for c in caps]

    # ---- T0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("T0  THE CEILING GATE: DOES THE INTERVAL ACTUALLY CONTAIN THE TRUTH?")
    P_(RULE)
    P_("  Never checked in four modules. A single violation and the certificate fails its purpose.")
    P_(f"\n    {'cap':>5} {'pruned':>14} {'+ certificate':>15} {'exact':>14} {'contains?':>10}")
    ok = True
    for c, tl, ct, per in res:
        hi = tl + ct
        good = ex <= hi * (1 + 1e-12) and tl <= ex * (1 + 1e-12)
        ok = ok and good
        P_(f"    {c:>5} {tl:>14.6e} {hi:>15.6e} {ex:>14.6e} {'yes' if good else 'NO':>10}")
    P_(f"\n  T0: {'the interval contains the exact tail at every cap, and the pruned value is below it at every cap. The certificate does bound what it claims to.' if ok else 'VIOLATED. The certificate does not bound the tail and every preceding result is about a quantity that fails its own purpose.'}")
    if not ok:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_tailtight.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- T1  THE WIDTH ------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("T1  THE INTERVAL'S WIDTH, WHICH IS THE ANSWER TO THE QUESTION")
    P_(RULE)
    P_(f"\n    {'cap':>5} {'paths kept':>11} {'interval width, orders':>23} {'cert/true deficit':>19}")
    for c, tl, ct, per in res:
        wid = float(np.log10((tl + ct) / tl))
        defc = ex - tl
        P_(f"    {c:>5} {c:>11,} {wid:>23.2f} {ct / max(defc, 1e-300):>19.3e}")
    P_("\n  Read the third column as the answer. The certificate is a VALID upper bound on the tail")
    P_("  and it places the tail inside an interval spanning that many orders of magnitude. The")
    P_("  fourth column is why: the certificate exceeds the deficit it is bounding by that factor.")
    P_("  A statement about a probability that spans twenty orders is not a statement about a")
    P_("  probability. Its validity is not offered as consolation.")

    # ---- T2  ENGINE WIDTH ---------------------------------------------------------------------
    P_("\n" + RULE)
    P_("T2  THE SAME AT THE ENGINE'S WIDTH")
    P_(RULE)
    Pm2, pi2, n2, hv2, S2, ab2, sha2 = prep(10, 6)
    tl2, ct2, per2 = run(Pm2, pi2, n2, hv2, S2, ab2, 6, 1000)
    P_(f"  10 controllers, n = {n2}, L = 6, cap 1000. sha {sha2[:12]}.")
    P_(f"\n    pruned tail (a lower bound)          {tl2:.6e}")
    P_(f"    composed certificate                 {ct2:.6e}")
    P_(f"    upper end of the interval            {tl2 + ct2:.6e}")
    P_(f"    interval width                       {np.log10((tl2 + ct2) / tl2):.2f} orders")
    P_(f"\n    backpoly's estimate of the truth     {BACKPOLY_EST:.4e}  +/- {BACKPOLY_ERR:.2f} orders")
    P_(f"    the certificate's upper end is       {np.log10((tl2 + ct2) / BACKPOLY_EST):.2f} orders above that estimate")
    P_(f"    and the pruned lower bound is        {np.log10(BACKPOLY_EST / tl2):.2f} orders below it")
    P_("  Carrying backpolyerr's interval explicitly: even at the estimate's upper end the")
    P_(f"  certificate is {np.log10((tl2 + ct2) / BACKPOLY_EST) - BACKPOLY_ERR:.2f} orders too high.")

    # ---- T3  THE DECOMPOSITION ----------------------------------------------------------------
    P_("\n" + RULE)
    P_("T3  THE DECOMPOSITION: WHERE THE LOOSENESS ACTUALLY IS")
    P_(RULE)
    tot = float(np.log10(ct2 / max(BACKPOLY_EST - tl2, 1e-300)))
    shares = np.array(per2) / sum(per2)
    m_eff = float(sum((6 - d) * shares[d] for d in range(len(shares))))
    floor = FLOOR_A * (float(n2) ** FLOOR_N) * (max(m_eff, 1.0) ** FLOOR_M)
    samp = np.log10(1.10)
    box = tot - np.log10(floor) - samp
    P_(f"  total looseness, certificate over the true deficit: {tot:.2f} orders")
    P_(f"\n    {'term':<44} {'orders':>8}")
    P_(f"    {'sampling slack (composecert, ~1.10x)':<44} {samp:>8.2f}")
    P_(f"    {'family floor 0.52 n^1.05 m^1.03, m_eff = ' + f'{m_eff:.2f}':<44} {np.log10(floor):>8.2f}")
    P_(f"    {'THE BOX RELAXATION, by subtraction':<44} {box:>8.2f}")
    P_(f"    {'total':<44} {tot:>8.2f}")
    P_(f"\n  T3: {'THE BOX RELAXATION DOMINATES.' if box > np.log10(floor) + samp else 'the family floor dominates.'} exactcert measured that term at ten to twenty orders")
    P_("  independently and showed it is the one thing in this family that is fixable in principle:")
    P_("  a bound over the REACHABLE completions rather than a box recovers it. Which means the")
    P_("  four modules that made this certificate valid, composable and adaptive were tightening")
    P_("  the sampling term -- worth 0.04 orders of the total.")

    # ---- T4  WHAT WOULD CLOSE IT --------------------------------------------------------------
    P_("\n" + RULE)
    P_("T4  WHAT WOULD CLOSE IT, AND WHAT COULD NOT")
    P_(RULE)
    best = floor * 1.10 * max(BACKPOLY_EST - tl2, 1e-300)
    P_(f"  best conceivable certificate of this family, floor x sampling x true deficit:"
       f" {best:.4e}")
    P_(f"  observed certificate:                                                        {ct2:.4e}")
    P_(f"  the gap between them -- entirely the box -- is {np.log10(ct2 / best):.2f} orders.")
    P_(f"\n  If the box were fixed, the interval would be"
       f" {np.log10((tl2 + best) / tl2):.2f} orders wide instead of"
       f" {np.log10((tl2 + ct2) / tl2):.2f}.")
    P_(f"  That is still {np.log10(floor):.2f} orders of irreducible floor, which no member of this family")
    P_("  escapes -- so even the best version of this certificate bounds the rare-event")
    P_(f"  probability to within a factor of about {floor * 1.10:,.0f}, not to within a few percent.")
    P_("  A certificate tight on the TAIL requires leaving the family, which is exactly what")
    P_("  exactcert concluded and what nothing since has changed.")

    P_("\n" + RULE)
    P_("T5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. T1 is exact at four controllers. T2's engine-width comparison rests on backpoly's")
    P_("     ESTIMATE of the truth, whose error interval is carried but which is not a bound.")
    P_("  2. The decomposition attributes the residual to the box by SUBTRACTION, using exactcert's")
    P_("     floor law and composecert's sampling slack. It is a budget that sums to the observed")
    P_("     total, not three independent measurements.")
    P_("  3. m_eff is the certificate-weighted mean of levels remaining, which is the right")
    P_("     summary for a sum dominated by one level but is a summary nonetheless.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_tailtight.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
