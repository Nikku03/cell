"""The error meter, built forward-backward: does it scale like the error, or like the bug?

THE PROPOSAL UNDER TEST. To decide whether two coupled parts may be separated, do not ask whether
they are weakly correlated. Ask how much the FINAL ANSWER moves. Estimate that cheaply as

    R(t)  =  | a(t)^T  dL  p(t) |          Dhat  =  integral_0^T R(t) dt

  p(t)  forward:  where the probability actually is at time t
  a(t)  backward: how much each state at time t contributes to the final rare event
  dL            : the generator change caused by the proposed simplification

WHY THIS SHOULD FIX THE BUG, structurally rather than hopefully. a^T dL p is LINEAR in dL and
therefore linear in the coupling lambda. A mutual information is QUADRATIC in lambda -- that
asymmetry is exactly what rem/atlas/RESULTS_grouping_law.txt measured as tail_err = c*sqrt(MI),
c -> 20.23. A first-order adjoint estimator cannot inherit that defect by construction.

THE ONE PLACE THE PROPOSAL OVERREACHES, and it is the pass criterion. It asks for Dhat >= Dtrue,
a one-sided bound. A first-order estimator cannot deliver one: the truth is Dhat + O(lambda^2) and
the remainder has no fixed sign. At small lambda this is immaterial; the exposure is the
INTERMEDIATE regime, which is where merges are actually decided. So the bound property is not
assumed here. It is measured, and the lambda at which it first fails is reported.

A CHOICE IN THE PROPOSAL THAT IS BETTER THAN IT LOOKS. The signed integral of a^T dL p IS the
first-order change in Y exactly. Placing the absolute value INSIDE the time integral discards
cancellation across time, so int|R| >= |int R|. That is deliberate conservatism with respect to
the first-order term, and both forms are reported below so the size of that choice is visible.

THE SYSTEM. A drives B:
    A:  birth kA,              death muA*A
    B:  birth kB + lambda*A,   death muB*B
The proposed simplification is the honest one -- MEAN-FIELD DECOUPLING: replace lambda*A by
lambda*<A>, which preserves B's mean exactly and destroys only the correlation. So dL carries
lambda*(A - <A>) and nothing else, and any measured error is attributable to correlation alone.
A is autonomous, so <A> is exact and independent of lambda.

Y is a rare event: P(B >= THRESH) at time T_END, started from empty.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

E1  EXACTNESS. Both generators must have columns summing to zero to < 1e-12, and B's mean under
    the split model must equal B's mean under the full model to < 1e-9 -- that is what makes the
    comparison attributable to correlation rather than to a shift in level.

E2  THE SCALING, AND IT IS THE WHOLE TEST. Sweep lambda over decades and fit exponents.
    Predeclared: true error ~ lambda^1, adjoint estimate ~ lambda^1, mutual information
    ~ lambda^2. Bar: the true-error and adjoint exponents within 0.15 of 1.0, and MI within 0.15
    of 2.0. If the adjoint estimator comes out quadratic, the bug has been recreated and the
    proposal fails here.

E3  THE BOUND, MEASURED NOT ASSUMED. Report the fraction of swept lambda at which
    Dhat >= |Dtrue|, for both the signed and the absolute-value forms, and report the largest
    lambda at which the bound holds. A first-order estimator is NOT expected to bound globally;
    the deliverable is where it stops.

E4  ACCURACY. Relative error |Dhat - Dtrue| / |Dtrue| across the sweep, reported at each lambda
    rather than as a single worst case, because the useful statement is where it degrades.

E5  ZERO-COUPLING CONTROL. At lambda = 0 the two models are identical, so Dtrue, Dhat and MI must
    all be exactly 0 to < 1e-12. A residual here means the machinery is measuring discretisation.

E6  AGAINST THE BULK BASELINE. The adjoint estimate must predict the true error better than
    c*sqrt(MI) with c calibrated in RESULTS_grouping_law.txt. If the bulk baseline wins there is
    no case for the adjoint construction.

E7  NON-VACUITY. Y must lie inside (1e-10, 0.2) at every lambda, and the true error must span at
    least two decades across the sweep, or E2's exponent fit has nothing to resolve.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm

RULE = "=" * 97

NA, NB = 14, 24
KA, MUA = 3.0, 1.0
KB, MUB = 0.6, 1.0
THRESH, T_END, NT = 12, 6.0, 401
C_BULK = 20.23


def generator(lam, mean_field_A=None):
    """(A,B) generator. mean_field_A set => B sees lambda*<A> instead of lambda*A (the split)."""
    n = (NA + 1) * (NB + 1)
    idx = lambda a, b: a * (NB + 1) + b
    L = np.zeros((n, n))

    def add(i, j, r):
        if r > 0:
            L[j, i] += r      # column-stochastic convention: dp/dt = L p
            L[i, i] -= r

    for a in range(NA + 1):
        for b in range(NB + 1):
            i = idx(a, b)
            if a + 1 <= NA:
                add(i, idx(a + 1, b), KA)
            if a > 0:
                add(i, idx(a - 1, b), MUA * a)
            drive = lam * (mean_field_A if mean_field_A is not None else a)
            if b + 1 <= NB:
                add(i, idx(a, b + 1), KB + drive)
            if b > 0:
                add(i, idx(a, b - 1), MUB * b)
    return L


def observable():
    f = np.zeros((NA + 1) * (NB + 1))
    for a in range(NA + 1):
        for b in range(THRESH, NB + 1):
            f[a * (NB + 1) + b] = 1.0
    return f


def mean_A_exact():
    """A is autonomous birth-death: exact stationary mean, independent of lambda."""
    p = np.zeros(NA + 1); p[0] = 1.0
    for a in range(1, NA + 1):
        p[a] = p[a - 1] * KA / (MUA * a)
    p /= p.sum()
    return float(np.arange(NA + 1) @ p)


def solve(L, p0, f, nt=NT, T=T_END):
    """Y, plus the forward and backward trajectories on a uniform grid."""
    dt = T / (nt - 1)
    step = expm(L * dt)
    stepT = expm(L.T * dt)
    fwd = np.zeros((nt, L.shape[0])); fwd[0] = p0
    for k in range(1, nt):
        fwd[k] = step @ fwd[k - 1]
    bwd = np.zeros((nt, L.shape[0])); bwd[-1] = f
    for k in range(nt - 2, -1, -1):
        bwd[k] = stepT @ bwd[k + 1]
    return float(f @ fwd[-1]), fwd, bwd


def mutual_information(L, p0, f):
    """MI between A and B in the joint law at T_END -- the bulk criterion, for comparison."""
    _, fwd, _ = solve(L, p0, f)
    P = fwd[-1].reshape(NA + 1, NB + 1)
    P = P / P.sum()
    PA = P.sum(axis=1); PB = P.sum(axis=0)
    outer = np.outer(PA, PB)
    m = (P > 0) & (outer > 0)
    return float(np.sum(P[m] * np.log(P[m] / outer[m])))


LAMBDAS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)


def run(lam, mA, f, p0):
    L_full = generator(lam)
    L_split = generator(lam, mean_field_A=mA)
    Y_full, _, _ = solve(L_full, p0, f)
    Y_split, fwd_s, bwd_s = solve(L_split, p0, f)
    d_true = Y_full - Y_split

    # first-order adjoint estimate, expanded about the CHEAP (split) model -- the direction the
    # engine would actually use, since the split model is the one it holds.
    dL = L_full - L_split
    t = np.linspace(0.0, T_END, NT)
    R = np.array([bwd_s[k] @ (dL @ fwd_s[k]) for k in range(NT)])
    d_signed = float(np.trapezoid(R, t))
    d_abs = float(np.trapezoid(np.abs(R), t))
    # a strictly conservative variant: absolute values BEFORE contracting
    R_el = np.array([np.abs(bwd_s[k]) @ np.abs(dL @ fwd_s[k]) for k in range(NT)])
    d_el = float(np.trapezoid(R_el, t))

    mi = mutual_information(L_full, p0, f)
    # B mean under both models, to confirm the split preserves the level
    Pf = solve(L_full, p0, f)[1][-1].reshape(NA + 1, NB + 1)
    Ps = fwd_s[-1].reshape(NA + 1, NB + 1)
    bax = np.arange(NB + 1, dtype=float)
    return dict(lam=lam, Y_full=Y_full, Y_split=Y_split, d_true=d_true,
                d_signed=d_signed, d_abs=d_abs, d_el=d_el, mi=mi,
                mB_full=float(Pf.sum(axis=0) @ bax), mB_split=float(Ps.sum(axis=0) @ bax))


def report():
    out = []; P = out.append
    mA = mean_A_exact()
    f = observable()
    p0 = np.zeros((NA + 1) * (NB + 1)); p0[0] = 1.0
    P(RULE)
    P("A FORWARD-BACKWARD ERROR METER: does it scale like the error, or like the bug?")
    P(RULE)
    P(f"  A drives B. Split = mean-field decoupling, lambda*A -> lambda*<A> with <A> = {mA:.6f}")
    P(f"  exact, so B's mean is preserved and only the correlation is destroyed.")
    P(f"  Y = P(B >= {THRESH}) at t = {T_END}, from an empty start. Estimator expanded about the")
    P("  SPLIT model, which is the one the engine actually holds.")
    P("")

    rows = [run(l, mA, f, p0) for l in LAMBDAS]
    zero = run(0.0, mA, f, p0)

    P(RULE)
    P("E1 / E7  PRECONDITIONS")
    P(RULE)
    L0 = generator(0.1)
    P(f"  generator columns sum to zero: max |sum| = {np.abs(L0.sum(axis=0)).max():.3e}   "
      f"{'PASS' if np.abs(L0.sum(axis=0)).max() < 1e-12 else 'FAIL'}")
    wm = max(abs(r["mB_full"] - r["mB_split"]) / r["mB_full"] for r in rows)
    P(f"  B mean preserved by the split: worst relative gap {wm:.3e}   "
      f"{'PASS' if wm < 1e-9 else 'FAIL'} (bar 1e-9)")
    ys = [r["Y_full"] for r in rows]
    P(f"  Y range {min(ys):.4e} to {max(ys):.4e}   "
      f"{'PASS' if all(1e-10 < y < 0.2 for y in ys) else 'FAIL'} (bar inside 1e-10..0.2)")
    dts = [abs(r["d_true"]) for r in rows]
    P(f"  true error spans {min(dts):.3e} to {max(dts):.3e} = {max(dts)/min(dts):.0f}x   "
      f"{'PASS' if max(dts)/min(dts) > 100 else 'FAIL'} (bar 100x)")
    P("")

    P(RULE)
    P("THE SWEEP")
    P(RULE)
    P(f"  {'lambda':>8s} {'Y_full':>11s} {'true err':>12s} {'adjoint':>12s} {'|.| inside':>12s}"
      f" {'elementwise':>12s} {'MI':>11s}")
    for r in rows:
        P(f"  {r['lam']:8.4f} {r['Y_full']:11.4e} {r['d_true']:+12.4e} {r['d_signed']:+12.4e}"
          f" {r['d_abs']:12.4e} {r['d_el']:12.4e} {r['mi']:11.4e}")
    P("")

    P(RULE)
    P("E2  THE SCALING -- the whole test")
    P(RULE)
    lam = np.array([r["lam"] for r in rows])
    def expo(v):
        v = np.abs(np.array(v)); m = v > 0
        return np.polyfit(np.log(lam[m]), np.log(v[m]), 1)[0]
    e_true = expo([r["d_true"] for r in rows])
    e_adj = expo([r["d_signed"] for r in rows])
    e_abs = expo([r["d_abs"] for r in rows])
    e_mi = expo([r["mi"] for r in rows])
    P(f"  true error      ~ lambda^{e_true:.4f}   "
      f"{'PASS' if abs(e_true-1) < 0.15 else 'FAIL'} (predeclared 1.0 +/- 0.15)")
    P(f"  adjoint (signed)~ lambda^{e_adj:.4f}   "
      f"{'PASS' if abs(e_adj-1) < 0.15 else 'FAIL'} (predeclared 1.0 +/- 0.15)")
    P(f"  adjoint (abs)   ~ lambda^{e_abs:.4f}")
    P(f"  mutual info     ~ lambda^{e_mi:.4f}   "
      f"{'PASS' if abs(e_mi-2) < 0.15 else 'FAIL'} (predeclared 2.0 +/- 0.15)")
    P("")
    if abs(e_adj - 1) < 0.15 and abs(e_mi - 2) < 0.15:
        P("  THE ESTIMATOR IS FIRST ORDER WHERE THE BULK CRITERION IS SECOND ORDER. That is the")
        P("  defect grouping.py measured, and this construction does not have it.")
    P("")

    P(RULE)
    P("E3  THE BOUND -- measured, not assumed")
    P(RULE)
    P(f"  {'lambda':>8s} {'|true|':>12s} {'signed':>12s} {'bounds?':>9s} {'|.| inside':>12s}"
      f" {'bounds?':>9s} {'elementwise':>12s} {'bounds?':>9s}")
    nb_s = nb_a = nb_e = 0
    for r in rows:
        t_ = abs(r["d_true"])
        bs, ba, be = abs(r["d_signed"]) >= t_, r["d_abs"] >= t_, r["d_el"] >= t_
        nb_s += bs; nb_a += ba; nb_e += be
        P(f"  {r['lam']:8.4f} {t_:12.4e} {abs(r['d_signed']):12.4e} {'yes' if bs else 'NO':>9s}"
          f" {r['d_abs']:12.4e} {'yes' if ba else 'NO':>9s} {r['d_el']:12.4e}"
          f" {'yes' if be else 'NO':>9s}")
    n = len(rows)
    P(f"  bounds in: signed {nb_s}/{n}, |.| inside {nb_a}/{n}, elementwise {nb_e}/{n}")
    P("  A first-order estimator was NOT predicted to bound globally. What is deliverable is")
    P("  which form bounds and over what range, and that is the table above.")
    P("")

    P(RULE)
    P("E4 / E6  ACCURACY, AND AGAINST THE BULK BASELINE")
    P(RULE)
    P(f"  {'lambda':>8s} {'adjoint rel err':>16s} {'bulk c*sqrt(MI) rel err':>25s}")
    ra, rb = [], []
    for r in rows:
        t_ = abs(r["d_true"])
        a_ = abs(abs(r["d_signed"]) - t_) / t_
        b_ = abs(C_BULK * np.sqrt(max(r["mi"], 0.0)) - t_) / t_
        ra.append(a_); rb.append(b_)
        P(f"  {r['lam']:8.4f} {a_:16.4f} {b_:25.4f}")
    P(f"  worst adjoint {max(ra):.4f}   worst bulk {max(rb):.3e}")
    P(f"  E6 {'PASS -- adjoint beats bulk' if max(ra) < max(rb) else 'FAIL -- bulk wins'}")
    P("")

    P(RULE)
    P("E5  ZERO-COUPLING CONTROL")
    P(RULE)
    w = max(abs(zero["d_true"]), abs(zero["d_signed"]), zero["d_abs"], abs(zero["mi"]))
    P(f"  at lambda = 0: true {zero['d_true']:.3e}, adjoint {zero['d_signed']:.3e}, "
      f"|.| {zero['d_abs']:.3e}, MI {zero['mi']:.3e}")
    P(f"  worst {w:.3e}   {'PASS' if w < 1e-12 else 'FAIL'} (bar 1e-12)")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
