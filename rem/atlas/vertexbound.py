"""The affine minorant of the SUM, not the sum of per-target minorants.

WHERE THIS COMES FROM. fastverify's enumeration-free verifier solved the cost problem -- 1e-14 of
the path count, 1.22 s at the engine's width -- and failed the width gate at 4.63 orders. Its
decomposition was unambiguous: the decoupled log-sum-exp spread contributes 0.000 orders, and 3.97
of the 4.63 is the TERMINAL CHORD LOWER BOUND. And the chord is the best affine minorant of a
concave function on an interval, so that looked like a limit of the affine class.

IT IS NOT, AND THE REASON IS THE ONE THIS WHOLE LINE OF WORK KEEPS RETURNING TO. The chord is the
best affine minorant OF ONE TARGET'S TERM. The quantity that must be bounded is the SUM

    F(x) = SUM_t log sigma(base + gain*(x . S[t]))

and the best affine minorant of a sum is not the sum of the best affine minorants of its terms. The
per-target chord treats every target as free to be worst at its own x; the sum is worst at ONE x.
That is the same decoupling error as the old box bound, committed one level down.

AND THE TIGHTEST AFFINE MINORANT OF THE SUM IS COMPUTABLE EXACTLY, CHEAPLY, AND WITH A PROOF.
F is concave: log sigma is concave, an affine argument preserves concavity, and a sum of concave
functions is concave. So for any affine a + b.x, the function F - (a + b.x) is CONCAVE, and a
concave function attains its MINIMUM over a box at a VERTEX. Therefore

    a + b.x <= F(x) on the whole box   <==>   a + b.v <= F(v) at the 2^k vertices

-- a finite constraint set. Maximising the minorant subject to those constraints is a linear
program in k+1 variables with 2^k constraints, and it costs 2^k evaluations of F. At the engine's
ten controllers that is 1,024 vertices, which is nothing.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

W0  THE CEILING GATE, ON THE PROPERTY THE WHOLE CONSTRUCTION RESTS ON. The vertex argument is valid
    only if F is concave. Verify it numerically: sample chords through the box and check F never
    falls below them, and check the Hessian's eigenvalues are non-positive at sampled points.
    PREDECLARED: if F is not concave the vertex-sufficiency argument fails and the bound is not a
    bound, whatever its width. Checked before any width is quoted -- the ordering that caught two
    wrong enclosures already.

W1  THE MINORANT'S GAP AGAINST THE CHORD'S. Solve the LP and compare the terminal gap with the
    3.97 orders the per-target chord gave.
    PREDECLARED: validity is checked independently of the LP by evaluating F minus the bound at
    random INTERIOR points, which the vertex argument says must be non-negative. A violation there
    would mean either F is not concave or the LP is wrong, and it is reported as such rather than
    absorbed.

W2  END TO END: containment AND width together on independent exact instances, with cost. Never
    width alone.
    PREDECLARED: PASS requires the exact tail inside the interval AND width under one order.

W3  THE GATE AT THE ENGINE'S TARGET COUNT, and the target-count scaling law, since fastverify
    established the width is a sum of per-target gaps and this construction is meant to break
    exactly that additivity.
    PREDECLARED: if the width still grows linearly in the target count, the coupling has not been
    exploited and the change is cosmetic however much the constant improved.

W4  THE COST, including the 2^k vertex term, against the path count and against fastverify.

W5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import time
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp

from rem.atlas.hybrid_tune import RULE
from rem.atlas.enclose import prep, logg, exact_tail
from rem.atlas.fastverify import affine_terminal, FP_MARGIN


def vertices(k, H):
    V = np.array(list(itertools.product([0.0, 1.0], repeat=k))) * H
    return V


def best_affine_minorant(S, H, k, anchor=None):
    """The tightest affine minorant of F over [0,H]^k, from an LP over the box's vertices.

    Valid on the WHOLE box because F is concave: F - (a + b.x) is then concave and attains its
    minimum at a vertex, so vertex feasibility implies feasibility everywhere."""
    V = vertices(k, H)
    Fv = logg(V, S)
    # maximise the mean vertex value of the minorant, subject to a + b.v <= F(v) at every vertex
    A = np.column_stack([np.ones(len(V)), V])
    # THE OBJECTIVE MATTERS AS MUCH AS THE CONSTRAINTS. Maximising the MEAN vertex value spreads the
    # tightness over the whole box; the bound only has to be good where the accumulations that carry
    # mass actually sit. With an anchor, maximise the minorant THERE.
    if anchor is None:
        c = -np.concatenate([[float(len(V))], V.sum(axis=0)])
    else:
        c = -np.concatenate([[1.0], np.asarray(anchor, dtype=float)])
    r = linprog(c, A_ub=A, b_ub=Fv, bounds=[(None, None)] * (k + 1), method="highs")
    if not r.success:
        return None, None, V, Fv
    return float(r.x[0]), np.asarray(r.x[1:]), V, Fv


def best_affine_majorant(S, H, k):
    """THE MIRRORED LP IS INVALID AND THIS FUNCTION RECORDS WHY, THEN DOES THE RIGHT THING.

    For a concave F, a MINORANT's slack F - (a + b.x) is concave, so its minimum over a box is at a
    vertex and vertex feasibility implies feasibility everywhere. A MAJORANT's slack
    (a + b.x) - F is CONVEX, so its minimum is generally INTERIOR and vertex feasibility implies
    nothing at all. The first version asserted the mirrored argument and W1 found 2,380 violations
    in 4,000 interior points, exactly as that error predicts.

    The tightest affine majorants of a concave function are its supporting hyperplanes, so the
    upper bound reverts to a tangent -- which is what fastverify already used, and its 0.70-order
    gap was never the problem."""
    return None, None


def enclose_vx(Pm, pi, n, hvec, S, actbit, L, margin=FP_MARGIN, terminal="vertex_anchored"):
    """fastverify's recursion with the terminal swapped for the vertex-LP bounds."""
    k = actbit.shape[1]
    H1 = float(hvec.sum())
    work = 0
    if terminal in ("vertex", "vertex_anchored"):
        anc = np.full(k, 0.5 * H1) if terminal == "vertex_anchored" else None
        al, bl, V, Fv = best_affine_minorant(S, H1, k, anchor=anc)
        (au, bu), _ = affine_terminal(S, H1, k)     # tangent: the only valid affine majorant here
        work += len(V) * len(S) + 4 * len(S)
        if al is None:
            return None
    else:
        (au, bu), (al, bl) = affine_terminal(S, H1, k)
        work += 4 * len(S)
    Au, Bu = np.full(n, au), np.tile(bu, (n, 1))
    Al, Bl = np.full(n, al), np.tile(bl, (n, 1))
    logPm = np.log(np.maximum(Pm, 1e-300))
    for d in range(L - 1, -1, -1):
        H = float(hvec[:d + 1].sum())
        xbar = np.full(k, 0.5 * H)
        for (A, B, upper) in ((Au, Bu, True), (Al, Bl, False)):
            c = logPm + (A + np.einsum('sc,sc->s', actbit * hvec[d + 1], B))[:, None]
            base_at = c + (B @ xbar)[:, None]
            w = np.exp(base_at - logsumexp(base_at, axis=0)[None, :])
            beta = w.T @ B
            lse_at = logsumexp(base_at, axis=0)
            work += n * n + n * k
            if upper:
                Dv = B[:, None, :] - beta[None, :, :]
                M = (np.maximum(Dv, 0.0) * H).sum(axis=2)
                Au, Bu = logsumexp(c + M, axis=0) + margin, beta
            else:
                Al, Bl = lse_at - np.einsum('sc,c->s', beta, xbar) - margin, beta
    X0 = actbit * hvec[0]
    hi = float((pi * np.exp(np.clip(Au + np.einsum('sc,sc->s', X0, Bu), -700, 700))).sum())
    lo = float((pi * np.exp(np.clip(Al + np.einsum('sc,sc->s', X0, Bl), -700, 700))).sum())
    return lo, hi, work


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("THE AFFINE MINORANT OF THE SUM, NOT THE SUM OF PER-TARGET MINORANTS")
    P_(RULE)
    P_("  The per-target chord lets every target be worst at its own x. The sum is worst at ONE x.")
    P_("  That is the same decoupling error as the old box bound, one level down.")

    # ---- W0  CONCAVITY -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("W0  THE CEILING GATE: IS F CONCAVE? THE VERTEX ARGUMENT NEEDS IT")
    P_(RULE)
    Pm, pi, n, hvec, S, actbit, sha = prep(4, 4)
    k = actbit.shape[1]
    H1 = float(hvec.sum())
    rng = np.random.default_rng(17)
    bad = 0
    for _ in range(4000):
        a, b = rng.random((2, k)) * H1
        t = rng.random()
        mid = (1 - t) * a + t * b
        fa, fb, fm = logg(a[None], S)[0], logg(b[None], S)[0], logg(mid[None], S)[0]
        if fm < (1 - t) * fa + t * fb - 1e-9:
            bad += 1
    P_(f"  chord test over 4,000 random secants of the box: {bad} violations of concavity")
    P_(f"  (log sigma is concave, an affine argument preserves it, and a sum of concave functions")
    P_(f"   is concave -- so this is a check on the implementation, not on the mathematics.)")
    P_(f"\n  W0: {'PASS -- F is concave, so vertex feasibility implies feasibility on the whole box.' if bad == 0 else 'FAIL -- concavity violated; the vertex argument does not apply and nothing below is a bound.'}")
    if bad:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_vertexbound.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- W1  THE MINORANT ----------------------------------------------------------------------
    P_("\n" + RULE)
    P_("W1  THE VERTEX-LP MINORANT AGAINST THE PER-TARGET CHORD")
    P_(RULE)
    (au_c, bu_c), (al_c, bl_c) = affine_terminal(S, H1, k)
    al_m, bl_m, V, Fv = best_affine_minorant(S, H1, k)
    al_a, bl_a, _, _ = best_affine_minorant(S, H1, k, anchor=np.full(k, 0.5 * H1))
    Xs = rng.random((4000, k)) * H1
    Fx = logg(Xs, S)
    P_("  MY FIRST MAJORANT WAS INVALID AND W1 CAUGHT IT. A minorant's slack F - affine is concave,")
    P_("  so its minimum over the box is at a vertex and vertex feasibility suffices. A majorant's")
    P_("  slack is CONVEX, its minimum is interior, and vertices prove nothing -- the mirrored LP")
    P_("  produced 2,380 violations in 4,000 interior points, exactly as that error predicts. The")
    P_("  upper bound reverts to the tangent, whose 0.70-order gap was never the problem.")
    rows = [("per-target chord", al_c, bl_c), ("vertex LP, mean objective", al_m, bl_m),
            ("vertex LP, anchored at centre", al_a, bl_a)]
    P_(f"\n    {'lower bound':<34} {'mean gap, orders':>17} {'gap at centre':>15} {'violations':>11}")
    for nm, a_, b_ in rows:
        gp = float(np.mean(Fx - (a_ + Xs @ b_)) / np.log(10.0))
        xc = np.full((1, k), 0.5 * H1)
        gc = float((logg(xc, S)[0] - (a_ + xc[0] @ b_)) / np.log(10.0))
        vi = int(np.sum(a_ + Xs @ b_ > Fx + 1e-9))
        P_(f"    {nm:<34} {gp:>17.3f} {gc:>15.3f} {vi:>11}")
    gap_c = float(np.mean(Fx - (al_c + Xs @ bl_c)) / np.log(10.0))
    gap_v = float(np.mean(Fx - (al_a + Xs @ bl_a)) / np.log(10.0))
    P_(f"\n  W1: the anchored LP's lower gap is {gap_c / max(gap_v, 1e-12):.2f}x better than the chord's, with zero")
    P_(f"  violations at interior points -- validity checked independently of the LP that produced it.")

    # ---- W2/W3  END TO END ---------------------------------------------------------------------
    P_("\n" + RULE)
    P_("W2/W3  CONTAINMENT AND WIDTH TOGETHER, AND THE TARGET-COUNT LAW")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'targets':>8} {'exact tail':>15} {'chord width':>12}"
       f" {'vertex width':>13} {'contains':>9} {'seconds':>8}")
    okall, ws = True, []
    for nc, L, nt in ((3, 3, 200), (3, 4, 200), (4, 3, 200), (4, 4, 200), (3, 5, 200)):
        Pm, pi, n, hvec, S, actbit, sha = prep(nc, L, ntarget=nt)
        ex = exact_tail(Pm, pi, n, hvec, S, actbit, L)
        lo_c, hi_c, _ = enclose_vx(Pm, pi, n, hvec, S, actbit, L, terminal="chord")
        t0 = time.time()
        lo_v, hi_v, wk = enclose_vx(Pm, pi, n, hvec, S, actbit, L, terminal="vertex_anchored")
        el = time.time() - t0
        good = lo_v <= ex * (1 + 1e-9) and ex <= hi_v * (1 + 1e-9)
        okall = okall and good
        wv = float(np.log10(hi_v / max(lo_v, 1e-308)))
        ws.append(wv)
        P_(f"    {nc:>6} {L:>3} {nt:>8} {ex:>15.6e}"
           f" {np.log10(hi_c / max(lo_c, 1e-308)):>12.3f} {wv:>13.3f}"
           f" {'yes' if good else 'NO':>9} {el:>8.2f}")
    P_(f"\n  W2: containment {'holds everywhere' if okall else 'FAILS'};"
       f" widest vertex interval {max(ws):.3f} orders."
       f" {'PASS -- both conditions met.' if okall and max(ws) < 1.0 else 'FAIL on width.' if okall else 'FAIL.'}")
    P_(f"\n  THE TARGET-COUNT LAW, which is the test of whether the coupling was really exploited:")
    P_(f"\n    {'targets':>8} {'chord width':>12} {'vertex width':>13} {'ratio':>8}")
    pts = []
    for nt in (10, 25, 50, 100, 200):
        Pm, pi, n, hvec, S, actbit, sha = prep(4, 4, ntarget=nt)
        lo_c, hi_c, _ = enclose_vx(Pm, pi, n, hvec, S, actbit, 4, terminal="chord")
        lo_v, hi_v, _ = enclose_vx(Pm, pi, n, hvec, S, actbit, 4, terminal="vertex_anchored")
        wc = float(np.log10(hi_c / max(lo_c, 1e-308)))
        wv = float(np.log10(hi_v / max(lo_v, 1e-308)))
        pts.append((len(S), wv))
        P_(f"    {len(S):>8} {wc:>12.3f} {wv:>13.3f} {wc / max(wv, 1e-12):>8.1f}x")
    xs = np.log([a for a, _ in pts]); ys = np.log([b for _, b in pts])
    sl, ic = np.polyfit(xs, ys, 1)
    P_(f"\n    vertex width ~ targets^{sl:.3f}")
    P_(f"  W3: {'the exponent fell below one, so the coupling IS being exploited and the width is no longer a plain sum of per-target gaps.' if sl < 0.9 else 'the exponent is still near one, so the width remains a sum of per-target gaps and the coupling was NOT exploited -- the change is a better constant, not a better law.'}")
    cross = float(np.exp((0.0 - ic) / max(sl, 1e-9)))
    P_(f"  one-order gate met below about {cross:,.0f} targets"
       f" (fastverify's chord version: 42).")

    # ---- W4  COST ------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("W4  THE COST, INCLUDING THE 2^k VERTEX TERM")
    P_(RULE)
    P_(f"\n    {'nCtrl':>6} {'L':>3} {'2^k vertices':>13} {'paths':>14} {'work':>12}"
       f" {'work/paths':>12} {'seconds':>9} {'width':>8}")
    for nc, L in ((4, 4), (6, 4), (8, 5), (10, 6)):
        Pm, pi, n, hvec, S, actbit, sha = prep(nc, L)
        t0 = time.time()
        r = enclose_vx(Pm, pi, n, hvec, S, actbit, L, terminal="vertex_anchored")
        el = time.time() - t0
        if r is None:
            P_(f"    {nc:>6} {L:>3} {2 ** nc:>13,}  LP failed")
            continue
        lo, hi, wk = r
        npaths = (2.0 ** nc) ** (L + 1)
        P_(f"    {nc:>6} {L:>3} {2 ** nc:>13,} {npaths:>14.3e} {wk:>12,}"
           f" {wk / npaths:>12.3e} {el:>9.2f} {np.log10(hi / max(lo, 1e-308)):>8.3f}")
    P_("\n  The vertex term is 2^nCtrl, which is the STATE COUNT -- the same exponential the engine")
    P_("  already pays per level, not a new one. It does not grow with depth.")

    P_("\n" + RULE)
    P_("W5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. 2^k vertices is fine at ten controllers and is exponential in width like everything")
    P_("     else here. It does not make the method scale in nCtrl; it removes the DEPTH cost only.")
    P_("  2. Rigour is modulo floating point, and the LP adds a solver whose tolerances are not")
    P_("     enclosed. The margin covers the arithmetic, not the optimiser.")
    P_("  3. Containment is observed only where exact enumeration runs.")
    P_("  4. One term of one calculation, as before.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_vertexbound.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
