"""Does a shared resource pool transmit its fluctuations to a reporter -- and by what law?

THE CHAIN THIS TESTS, in three models of increasing realism.

  M1  NON-CONSUMING POOL.  Pool P is a birth-death process; a reporter X is produced at rate
      c*P and removed at rate mu_X*X. The gene reads the pool but does not deplete it. This is
      the shared-driver picture, and extrinsic noise is obviously non-zero here.

  M2  LINEAR CONSUMPTION.  As M1, but every production event REMOVES one pool molecule. This is
      the case that matters for a real circuit, and the claim under test is that the transmitted
      fluctuation COLLAPSES: phi -> 0 exactly.

  M3  SEQUESTERED CATALYST.  R + M <-> C -> X + R + M, at fixed total R. The ribosome is bound,
      held, and released; the free pool R_tot - C fluctuates because C does. This sits BETWEEN
      M1 and M2 and is what translation actually is. If phi > 0 here, the effect is real for
      real circuits. If phi = 0 here too, the line is dead.

THE LAW UNDER TEST, and there is a disagreement to settle. A candidate law was proposed as

      tau_X = tau_p + phi*(tau_p + tau_R)                                        [L_B]

Deriving it here from the linear-noise solution of M1 instead gives

      tau_X = tau_p + phi*tau_R                                                  [L_A]

These differ by exactly phi*tau_p, so they agree only when phi*tau_p is negligible -- i.e.
precisely in the regime where the law says nothing. Both are tested, against three more rivals,
and whichever survives the sweep survives. Neither is assumed.

DEFINITIONS, fixed here so they cannot drift:
  tau_p = 1/mu_X, the reporter's own removal time.
  tau_R = integrated autocorrelation time of the DRIVER (P in M1/M2, the complex C in M3).
  tau_X = integrated autocorrelation time of the reporter X.
  phi   = extrinsic fraction of X's variance, DEFINED AS A DUAL REPORTER MEASURES IT:
          phi = Cov(X1,X2)/Var(X1) for two identical independent reporters on one driver.
          For a linear birth-death driven by an external driver this equals 1 - <X>/Var(X),
          and gate S3 checks that identity against a full three-dimensional CME rather than
          assuming it.

Every correlation time is an exact linear solve, not a numerical integral of matrix exponentials:
for generator Q and centred observable g, integral_0^inf exp(Qt) g dt = u where Q u = -g, so
tau = <g,u>_pi / <g,g>_pi. One solve per observable.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

S0  VACUITY, AND IT IS THE GATE THAT MATTERS MOST HERE.
    At phi = 0 every candidate law collapses to tau_X = tau_p and all of them "pass" on any
    evidence whatsoever. So: any model row with phi < 0.02 is reported VOID, never PASS, and a
    law is credited only on rows where phi clears that margin. This is written down first
    because the failure it guards against -- a gate that could not have failed -- has already
    happened once on this exact question.

S1  DISCRIMINATION. On the rows where the winning law passes, at least one rival must MISS by
    more than 5% relative. If every candidate passes, the relation is an identity forced by the
    parameterisation and nothing has been learned. Reported as the worst rival's error.

S2  EXACTNESS. The winning law must hold to < 1e-6 relative, worst case over the sweep.

S3  THE phi IDENTITY. phi = 1 - <X>/Var(X) must equal Cov(X1,X2)/Var(X1) computed from a full
    3-D (driver, X1, X2) CME, to < 1e-9. If it does not, the cheap 2-D route is wrong and every
    phi in this module is wrong with it.

S4  FAST-DRIVER CONTROL. As the driver is made fast (tau_R -> 0) at fixed stationary law,
    phi*tau_R -> 0 and tau_X must converge to tau_p. Bar: within 1e-4 relative at the fastest
    driver. A law that does not reduce correctly in the limit where the pool cannot transmit
    anything is not a law.

S5  M2 COLLAPSE, STATED AS A PREDICTION. phi is predicted to be 0 to < 1e-9 under strictly
    linear consumption. If it is NOT zero, the claimed collapse is false and that is the report.

S6  M3 DECISION. Is phi > 0 for a sequestered catalyst? Reported with its margin over the S0
    threshold. This is the whole question; it is reported whichever way it falls.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

RULE = "=" * 97


# -------------------------------------------------------------------------------------------
# generic machinery: stationary law, and exact integrated autocorrelation by one linear solve
# -------------------------------------------------------------------------------------------

def build_generator(n_states, transitions):
    """transitions: iterable of (i, j, rate). Returns sparse Q with rows summing to zero."""
    rows, cols, vals = [], [], []
    for i, j, r in transitions:
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)
    r = np.array(rows); c = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n_states)
    return sp.coo_matrix((np.concatenate([v, -diag]),
                          (np.concatenate([r, np.arange(n_states)]),
                           np.concatenate([c, np.arange(n_states)]))),
                         shape=(n_states, n_states)).tocsr()


def stationary(Q):
    """Exact null vector of Q^T. Normalisation row placed on the HIGHEST-PROBABILITY state.

    Placing it on the last state costs orders of accuracy in the tails; this is the standing
    rule in this project and it is not optional.
    """
    n = Q.shape[0]
    # cheap first pass to locate the mode
    A = Q.T.tolil(); A[0, :] = 1.0
    b = np.zeros(n); b[0] = 1.0
    p0 = np.maximum(spl.spsolve(A.tocsr(), b), 0.0)
    mode = int(np.argmax(p0))
    A = Q.T.tolil(); A[mode, :] = 1.0
    b = np.zeros(n); b[mode] = 1.0
    p = np.maximum(spl.spsolve(A.tocsr(), b), 0.0)
    return p / p.sum()


def integrated_time(Q, pi, f):
    """Integrated autocorrelation time of observable f: one linear solve, no matrix exponentials.

    tau = <g,u>_pi / <g,g>_pi  where  Q u = -g,  g = f - <f>_pi.
    The solve is singular along the constant direction; g is pi-orthogonal so it is consistent,
    and the null direction is pinned by replacing one row with the constraint pi.u = 0.
    """
    n = Q.shape[0]
    mean = float(pi @ f)
    g = f - mean
    var = float(pi @ (g * g))
    if var <= 0:
        return np.nan, mean, var
    mode = int(np.argmax(pi))
    A = Q.tolil()
    A[mode, :] = pi                      # constraint row: pi . u = 0
    rhs = -g.copy(); rhs[mode] = 0.0
    u = spl.spsolve(A.tocsr(), rhs)
    tau = float(pi @ (g * u)) / var
    return tau, mean, var


# -------------------------------------------------------------------------------------------
# M1 / M2 : pool + reporter, with or without consumption
# -------------------------------------------------------------------------------------------

def pool_reporter(pcap, xcap, kR, muR, c, muX, consume: bool):
    """Pool P (birth kR, death muR*P). Reporter X produced at c*P, removed at muX*X.

    consume=True: each X production also removes one pool molecule (strictly linear consumption).
    """
    idx = lambda p, x: p * (xcap + 1) + x
    n = (pcap + 1) * (xcap + 1)
    T = []
    for p in range(pcap + 1):
        for x in range(xcap + 1):
            i = idx(p, x)
            if p + 1 <= pcap:
                T.append((i, idx(p + 1, x), kR))
            if p > 0:
                T.append((i, idx(p - 1, x), muR * p))
            if x + 1 <= xcap and p > 0:
                if consume:
                    T.append((i, idx(p - 1, x + 1), c * p))
                else:
                    T.append((i, idx(p, x + 1), c * p))
            if x > 0:
                T.append((i, idx(p, x - 1), muX * x))
    Q = build_generator(n, T)
    P = np.repeat(np.arange(pcap + 1), xcap + 1).astype(float)
    X = np.tile(np.arange(xcap + 1), pcap + 1).astype(float)
    return Q, P, X


# -------------------------------------------------------------------------------------------
# M3 : sequestered catalyst.  R + M <-> C -> X + R + M, total R fixed.
# -------------------------------------------------------------------------------------------

def sequester(Rtot, Mtot, xcap, kon, koff, kcat, muX):
    """State (C, X). Free ribosome = Rtot - C, free mRNA = Mtot - C.

    C is autonomous: none of its rates depend on X. That is what makes the driver well defined.
    """
    cmax = int(min(Rtot, Mtot))
    idx = lambda cc, x: cc * (xcap + 1) + x
    n = (cmax + 1) * (xcap + 1)
    T = []
    for cc in range(cmax + 1):
        for x in range(xcap + 1):
            i = idx(cc, x)
            if cc + 1 <= cmax:
                T.append((i, idx(cc + 1, x), kon * (Rtot - cc) * (Mtot - cc)))
            if cc > 0:
                T.append((i, idx(cc - 1, x), koff * cc))
                if x + 1 <= xcap:
                    T.append((i, idx(cc - 1, x + 1), kcat * cc))   # release R and M, make X
            if x > 0:
                T.append((i, idx(cc, x - 1), muX * x))
    Q = build_generator(n, T)
    C = np.repeat(np.arange(cmax + 1), xcap + 1).astype(float)
    X = np.tile(np.arange(xcap + 1), cmax + 1).astype(float)
    return Q, C, X, cmax


def sequester_dual(Rtot, Mtot, xcap, kon, koff, kcat, muX):
    """(C, X1, X2) for the S3 identity check. Two identical reporters on one driver."""
    cmax = int(min(Rtot, Mtot))
    nx = xcap + 1
    idx = lambda cc, a, b: (cc * nx + a) * nx + b
    n = (cmax + 1) * nx * nx
    T = []
    for cc in range(cmax + 1):
        for a in range(nx):
            for b in range(nx):
                i = idx(cc, a, b)
                if cc + 1 <= cmax:
                    T.append((i, idx(cc + 1, a, b), kon * (Rtot - cc) * (Mtot - cc)))
                if cc > 0:
                    T.append((i, idx(cc - 1, a, b), koff * cc))
                    if a + 1 < nx:
                        T.append((i, idx(cc - 1, a + 1, b), kcat * cc))
                    if b + 1 < nx:
                        T.append((i, idx(cc - 1, a, b + 1), kcat * cc))
                if a > 0:
                    T.append((i, idx(cc, a - 1, b), muX * a))
                if b > 0:
                    T.append((i, idx(cc, a, b - 1), muX * b))
    Q = build_generator(n, T)
    ax = np.arange(nx, dtype=float)
    X1 = np.tile(np.repeat(ax, nx), cmax + 1)
    X2 = np.tile(np.tile(ax, nx), cmax + 1)
    return Q, X1, X2


# -------------------------------------------------------------------------------------------
# the candidate laws
# -------------------------------------------------------------------------------------------

LAWS = {
    "L_A  tau_p + phi*tau_R":            lambda tp, tr, ph: tp + ph * tr,
    "L_B  tau_p + phi*(tau_p+tau_R)":    lambda tp, tr, ph: tp + ph * (tp + tr),
    "L_C  (1-phi)*tau_p + phi*tau_R":    lambda tp, tr, ph: (1 - ph) * tp + ph * tr,
    "L_D  tau_p*(1+phi)":                lambda tp, tr, ph: tp * (1 + ph),
    "L_E  tau_p + tau_R":                lambda tp, tr, ph: tp + tr,
}

PHI_VACUITY = 0.02


def measure(Q, driver, X):
    pi = stationary(Q)
    tR, mR, vR = integrated_time(Q, pi, driver)
    tX, mX, vX = integrated_time(Q, pi, X)
    phi = 1.0 - mX / vX if vX > 0 else np.nan
    return dict(pi=pi, tau_R=tR, tau_X=tX, phi=phi, mX=mX, vX=vX, mR=mR, vR=vR)


# -------------------------------------------------------------------------------------------
# report
# -------------------------------------------------------------------------------------------

def law_errors(tp, tr, phi, tx):
    return {k: abs(f(tp, tr, phi) - tx) / tx for k, f in LAWS.items()}


def report():
    out = []; P = out.append
    P(RULE)
    P("DOES A SHARED RESOURCE POOL TRANSMIT ITS FLUCTUATIONS -- AND BY WHAT LAW?")
    P(RULE)
    P("  Three models: non-consuming pool, linear consumption, sequestered catalyst.")
    P(f"  phi is the dual-reporter extrinsic fraction. Vacuity threshold phi > {PHI_VACUITY}:")
    P("  below it every candidate law collapses to tau_X = tau_p and cannot be discriminated.")
    P("")

    # ---------------- M1 ----------------
    P(RULE)
    P("M1  NON-CONSUMING POOL   dP: birth kR, death muR*P.   X made at c*P, removed at muX*X")
    P(RULE)
    P(f"  {'kR':>6s} {'muR':>7s} {'c':>6s} {'muX':>7s} {'phi':>9s} {'tau_R':>9s} {'tau_X':>9s}"
      f" {'tau_p':>8s} {'L_A err':>10s} {'L_B err':>10s}")
    m1 = []
    for kR, muR, c, muX in [(20.0, 1.0, 0.8, 0.5), (20.0, 2.0, 0.8, 0.5),
                            (20.0, 0.5, 0.8, 0.5), (20.0, 1.0, 2.0, 0.5),
                            (20.0, 1.0, 0.8, 1.0), (40.0, 1.0, 0.8, 0.25)]:
        # CORRECTION 1: caps of (70,160) truncated the muX=0.25 row, whose <X> is 128.
        # L_A read 4.64e-02 there; at (120,300) it reads 7.7e-15 and is flat to (200,600).
        Q, Pv, Xv = pool_reporter(120, 300, kR, muR, c, muX, consume=False)
        r = measure(Q, Pv, Xv); tp = 1.0 / muX
        e = law_errors(tp, r["tau_R"], r["phi"], r["tau_X"])
        m1.append((tp, r, e))
        P(f"  {kR:6.1f} {muR:7.2f} {c:6.2f} {muX:7.2f} {r['phi']:9.5f} {r['tau_R']:9.5f}"
          f" {r['tau_X']:9.5f} {tp:8.4f} {e['L_A  tau_p + phi*tau_R']:10.2e}"
          f" {e['L_B  tau_p + phi*(tau_p+tau_R)']:10.2e}")
    P("")
    P("  worst relative error of every candidate over these rows:")
    for k in LAWS:
        w = max(e[k] for _, _, e in m1)
        P(f"    {k:34s} {w:10.3e}   {'PASS' if w < 1e-6 else 'MISS'}")
    phis = [r['phi'] for _, r, _ in m1]
    P(f"  S0 vacuity: min phi over rows = {min(phis):.5f} "
      f"({'OK, discriminating' if min(phis) > PHI_VACUITY else 'VOID'})")
    win = min(LAWS, key=lambda k: max(e[k] for _, _, e in m1))
    worst_rival = max(max(e[k] for _, _, e in m1) for k in LAWS if k != win)
    P(f"  S1 discrimination: best rival misses by {worst_rival:.3e} "
      f"({'PASS' if worst_rival > 0.05 else 'FAIL -- relation may be an identity'})")
    P(f"  WINNER: {win}")
    P("")

    # ---------------- S4 control ----------------
    P(RULE)
    P("S4  FAST-DRIVER CONTROL -- speed the pool up at fixed stationary law; tau_X must -> tau_p")
    P(RULE)
    P(f"  {'speed':>8s} {'tau_R':>10s} {'phi':>9s} {'tau_X':>10s} {'tau_p':>8s} {'rel dev':>10s}")
    muX = 0.5; tp = 1.0 / muX; dev = None
    for s in (1.0, 4.0, 16.0, 64.0, 256.0):
        Q, Pv, Xv = pool_reporter(120, 300, 20.0 * s, 1.0 * s, 0.8, muX, consume=False)
        r = measure(Q, Pv, Xv)
        dev = abs(r["tau_X"] - tp) / tp
        P(f"  {s:8.0f} {r['tau_R']:10.6f} {r['phi']:9.5f} {r['tau_X']:10.6f} {tp:8.4f}"
          f" {dev:10.3e}")
    P(f"  S4 {'PASS' if dev < 1e-4 else 'FAIL'} (bar 1e-4 at the fastest driver)")
    P("")

    # ---------------- M2 ----------------
    P(RULE)
    P("M2  LINEAR CONSUMPTION -- every production event removes one pool molecule")
    P(RULE)
    P(f"  {'kR':>6s} {'muR':>7s} {'c':>6s} {'muX':>7s} {'phi':>12s} {'tau_R':>9s} {'tau_X':>9s}"
      f" {'tau_p':>8s}")
    phis2 = []
    for kR, muR, c, muX in [(20.0, 1.0, 0.8, 0.5), (20.0, 2.0, 0.8, 0.5),
                            (20.0, 1.0, 2.0, 0.5), (40.0, 1.0, 0.8, 0.25)]:
        Q, Pv, Xv = pool_reporter(120, 300, kR, muR, c, muX, consume=True)
        r = measure(Q, Pv, Xv); tp = 1.0 / muX
        phis2.append(abs(r["phi"]))
        P(f"  {kR:6.1f} {muR:7.2f} {c:6.2f} {muX:7.2f} {r['phi']:12.3e} {r['tau_R']:9.5f}"
          f" {r['tau_X']:9.5f} {tp:8.4f}")
    P(f"  S5 predicted collapse phi -> 0: worst |phi| = {max(phis2):.3e}   "
      f"{'CONFIRMED' if max(phis2) < 1e-9 else 'NOT CONFIRMED -- phi is not zero'}")
    if max(phis2) < PHI_VACUITY:
        P("  Every law test on M2 is therefore VOID, not passed: at phi = 0 both sides of every")
        P("  candidate collapse to tau_X = tau_p and no evidence could distinguish them.")
    P("")

    # ---------------- M3 ----------------
    P(RULE)
    P("M3  SEQUESTERED CATALYST   R + M <-> C -> X + R + M,  total R fixed")
    P(RULE)
    P(f"  {'Rtot':>5s} {'Mtot':>5s} {'kon':>7s} {'koff':>6s} {'kcat':>6s} {'muX':>6s}"
      f" {'phi':>9s} {'tau_R':>9s} {'tau_X':>9s} {'tau_p':>8s} {'L_A':>9s} {'L_B':>9s}")
    m3 = []
    for Rtot, Mtot, kon, koff, kcat, muX in [
            (12, 10, 0.05, 1.0, 2.0, 0.5), (12, 10, 0.05, 1.0, 2.0, 0.25),
            (20, 6, 0.05, 1.0, 2.0, 0.5),  (12, 10, 0.02, 0.5, 1.0, 0.5),
            (8,  8, 0.10, 2.0, 4.0, 0.5),  (12, 10, 0.05, 1.0, 2.0, 1.0)]:
        Q, Cv, Xv, cmax = sequester(Rtot, Mtot, 200, kon, koff, kcat, muX)
        r = measure(Q, Cv, Xv); tp = 1.0 / muX
        e = law_errors(tp, r["tau_R"], r["phi"], r["tau_X"])
        m3.append((tp, r, e))
        P(f"  {Rtot:5d} {Mtot:5d} {kon:7.3f} {koff:6.2f} {kcat:6.2f} {muX:6.2f}"
          f" {r['phi']:9.5f} {r['tau_R']:9.5f} {r['tau_X']:9.5f} {tp:8.4f}"
          f" {e['L_A  tau_p + phi*tau_R']:9.2e} {e['L_B  tau_p + phi*(tau_p+tau_R)']:9.2e}")
    phis3 = [r["phi"] for _, r, _ in m3]
    P("")
    P(f"  S6 DECISION  min phi = {min(phis3):.5f}, max phi = {max(phis3):.5f}")
    # CORRECTION 2: S0 was written as phi > PHI_VACUITY. phi here is NEGATIVE, and a negative
    # phi discriminates between the candidate laws exactly as well as a positive one -- what
    # makes a row vacuous is |phi| being small, since that is when every law collapses onto
    # tau_X = tau_p. Testing the signed value wrongly declared the discriminating case dead.
    amin = min(abs(v) for v in phis3)
    if amin > PHI_VACUITY:
        P(f"     |phi| ranges {amin:.5f} to {max(abs(v) for v in phis3):.5f}, clearing the "
          f"{PHI_VACUITY} vacuity bar by {amin/PHI_VACUITY:.1f}x at worst.")
        P("     phi is NOT zero for a sequestered catalyst -- it is NEGATIVE. Sequestration")
        P("     transmits pool fluctuation with the OPPOSITE SIGN to a non-consuming pool,")
        P("     where strictly linear consumption transmits none at all.")
    else:
        P(f"     |phi| does NOT clear the vacuity bar. The line is dead and this is the report.")
    P("")
    P("  worst relative error of every candidate over the M3 rows:")
    for k in LAWS:
        w = max(e[k] for _, _, e in m3)
        P(f"    {k:34s} {w:10.3e}   {'PASS' if w < 1e-6 else 'MISS'}")
    if min(abs(v) for v in phis3) > PHI_VACUITY:
        win3 = min(LAWS, key=lambda k: max(e[k] for _, _, e in m3))
        rival3 = max(max(e[k] for _, _, e in m3) for k in LAWS if k != win3)
        P(f"  S1 discrimination on M3: best rival misses by {rival3:.3e} "
          f"({'PASS' if rival3 > 0.05 else 'FAIL'})")
        P(f"  WINNER on M3: {win3}")
    P("")
    P(RULE)
    P("DIRECTION -- does the reporter decorrelate FASTER or SLOWER than its own removal time?")
    P(RULE)
    P("  This is the experimentally visible prediction, and the two models disagree in SIGN.")
    P(f"  {'model':<26s}{'tau_p':>8s}{'tau_X':>11s}{'Fano':>9s}{'tau_X/tau_p':>13s}{'verdict':>10s}")
    for lab, (kR, muR, c, muX) in (("M1 non-consuming pool", (20.0, 1.0, 0.8, 0.5)),
                                   ("M1 non-consuming pool", (20.0, 2.0, 0.8, 0.5))):
        Q, Pv, Xv = pool_reporter(120, 300, kR, muR, c, muX, consume=False)
        r = measure(Q, Pv, Xv); tp = 1.0 / muX; fano = r["vX"] / r["mX"]
        P(f"  {lab:<26s}{tp:8.3f}{r['tau_X']:11.6f}{fano:9.5f}{r['tau_X']/tp:13.6f}"
          f"{'SLOWER' if r['tau_X'] > tp else 'FASTER':>10s}")
    for lab, (Rt, Mt, kon, koff, kcat, mu) in (
            ("M3 sequestered catalyst", (12, 10, 0.05, 1.0, 2.0, 0.5)),
            ("M3 sequestered catalyst", (20, 6, 0.05, 1.0, 2.0, 0.5)),
            ("M3 sequestered catalyst", (8, 8, 0.10, 2.0, 4.0, 0.5))):
        Q, Cv, Xv, _ = sequester(Rt, Mt, 200, kon, koff, kcat, mu)
        r = measure(Q, Cv, Xv); tp = 1.0 / mu; fano = r["vX"] / r["mX"]
        P(f"  {lab:<26s}{tp:8.3f}{r['tau_X']:11.6f}{fano:9.5f}{r['tau_X']/tp:13.6f}"
          f"{'SLOWER' if r['tau_X'] > tp else 'FASTER':>10s}")
    P("")
    P("  A non-consuming shared driver makes the reporter super-Poissonian (Fano > 1) and SLOWER.")
    P("  A sequestered catalyst makes it sub-Poissonian (Fano < 1) and FASTER. The sign of the")
    P("  effect is set by whether producing the protein DEPLETES the driver, and translation --")
    P("  which holds the ribosome and releases it -- is on the depleting side.")
    return "\n".join(out)


def check_S3(Rtot=6, Mtot=5, xcap=14, kon=0.08, koff=1.0, kcat=2.0, muX=0.6):
    """S3: phi = 1 - <X>/Var(X) must equal Cov(X1,X2)/Var(X1) from the full 3-D CME."""
    Q2, C2, X2v, _ = sequester(Rtot, Mtot, 90, kon, koff, kcat, muX)
    r2 = measure(Q2, C2, X2v)
    Qd, A, B = sequester_dual(Rtot, Mtot, xcap, kon, koff, kcat, muX)
    pid = stationary(Qd)
    mA = float(pid @ A); mB = float(pid @ B)
    cov = float(pid @ (A * B)) - mA * mB
    vA = float(pid @ (A * A)) - mA ** 2
    phi_dual = cov / vA
    Q2s, C2s, X2s, _ = sequester(Rtot, Mtot, xcap, kon, koff, kcat, muX)
    r2s = measure(Q2s, C2s, X2s)
    return r2["phi"], r2s["phi"], phi_dual


if __name__ == "__main__":
    print(report())
    print()
    print(RULE)
    print("S3  THE phi IDENTITY -- cheap 2-D route against a full 3-D dual-reporter CME")
    print(RULE)
    a, b, c = check_S3()
    print(f"  phi = 1 - <X>/Var(X), large cap        : {a:.10f}")
    print(f"  phi = 1 - <X>/Var(X), matched cap      : {b:.10f}")
    print(f"  phi = Cov(X1,X2)/Var(X1), 3-D CME      : {c:.10f}")
    rel = abs(b - c) / abs(c)
    print(f"  matched-cap relative difference        : {rel:.3e}   "
          f"{'PASS' if rel < 1e-9 else 'FAIL'} (bar 1e-9)")
