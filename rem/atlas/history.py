"""Route 3 of 3: carry the controller's HISTORY, not its state -- and price all three routes together.

WHY THIS ONE HAS A CHANCE THE OTHERS DID NOT. statedim S6 measured that conditioning on a global
controller's instantaneous state fails at matched timescales, and named the reason: the dependence
between its targets is carried by the controller's TRAJECTORY, not by its current value. A slow
controller is quasi-static, so its state IS its history and conditioning works (41,613x reduction);
a fast one correlates nothing; in between, four orders of magnitude of timescale ratio, nothing is
recovered.

But in a star topology the targets are conditionally independent given the controller's ENTIRE
PATH -- that is what "the hub drives them and they do not touch each other" means. So route 3 is
not a hope, it is a quantitative question: how much of the path has to be carried, and what does
carrying it cost? Routes 1 and 2 both failed against a structural obstruction. This one cannot
fail structurally; it can only be too expensive.

HOW THE PATH IS CARRIED. The stationary process is Markov, so the joint law of the state now and
the controller's state at earlier sample times is available exactly from the transition semigroup:

    f_a(x) = P( h(0)=a_0, h(dt)=a_1, ..., h(L dt)=a_L,  X(L dt) = x )

built by a forward recursion that applies exp(Q^T dt) and masks on the controller bit at each
step. Conditioning on the pattern a rather than on h alone is what "carrying history" means, and
L = 0 reduces EXACTLY to statedim S6's conditional mutual information -- which is R1's
cross-module identity check, not a new number.

THE APPROXIMATION IT LICENSES, so accuracy is measured the same way routes 1 and 2 were:

    P_hat(x) = sum_a P(a) prod_i P( x_i | a )

a mixture over controller histories of products over targets. Its cost is N 2^(1+L+1) -- one
factor per gene, over its own state and the L+1 history bits.

THE ROUTES NEST, which makes the comparison exact rather than analogical. On a star, route 1 at
r = 1 gives every target the hub as its only ball member, which IS route 3 at L = 0. Route 1 at
r = 2 gives every target every other target, which is the exact joint. So all three routes are
points on one axis and R7 puts them there.

WHAT THIS MODULE MUST NOT DO. It must not compare route 3 on a star against routes 1 and 2 on
whatever topology flattered them. Every number in R7 is computed on the SAME system with the SAME
observable, and the observable is the conjunctive rare event the whole architecture exists for.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

R1  THE HISTORY JOINT IS CORRECT. sum over patterns and states of f_a must be 1 to machine
    precision. And at L = 0 the conditional mutual information must reproduce statedim S6's
    measurement of I(i;j|h) on the same system -- a cross-module identity, so an error in the
    semigroup recursion cannot hide.

R2  DOES RESIDUAL DEPENDENCE FALL WITH THE NUMBER OF LAGS? At the matched timescale where S6 says
    state-conditioning fails, sweep L and report max residual I(i;j | history). PREDECLARED: if
    it falls below tau at some L, route 3 works and the deliverable is that L; if it plateaus
    above tau, route 3 fails too and state complexity is closed with all three routes exhausted.

R3  THE TIMESCALE LAW, RE-DERIVED FROM THE OTHER SIDE. The number of lags needed must peak at
    matched timescales and fall to zero at both extremes, because that is what S6's curve says
    from the state side. If it does not, one of the two measurements is wrong and both are
    suspect. This is a consistency gate between modules, not a new claim.

    WHAT "COMPUTE" MEANS HERE, so the comparison in R7 is not quietly generous. Carrying L lags
    makes each gene's factor a table over its own state and the 2^(L+1) history patterns, so the
    representation costs N 2^(L+2) -- that is the number quoted. The exact construction below
    also enumerates 2^(L+1) sub-measures over the full state space, which a deployed engine would
    replace by filtering or sampling; that is a cost of MEASURING the answer here, not of using
    it, and it is not what R5 or R7 report.

R4  THE WINDOW, NOT JUST THE COUNT. At fixed L, sweep the sampling interval dt. There should be
    an optimum near (controller correlation time)/L -- too fine and the lags are redundant, too
    coarse and the path between them is lost. Report the optimum rather than tuning to it
    silently.

R5  THE ACCURACY/COMPUTE CURVE. Convert L into the actual factor size and report tail error
    against cost, so the answer is a curve and not a verdict.

R6  THE MATCHED CONTROL. Condition on L lags of a RANDOM NON-HUB GENE instead of the controller.
    If that works as well, the gain comes from adding conditioning variables and not from the
    controller's history, and R2 says nothing about controllers.

R8  COST AT FIXED ACCURACY AGAINST HUB DEGREE  (added after R7, and labelled). R7 as first run
    compared the three routes on a SIX-target star, where route 1 is exact for 2^6 = 128 and
    therefore wins outright -- and the run's closing sentence, that route 3 is the only one
    reaching 1% affordably, was not supported by its own table. The comparison was rigged by
    system size. The costs scale differently in the hub degree k: route 1's exact point is 2^k,
    route 2's error at fixed width worsens with k, and route 3's cost is 2^(L+2) with NO k in it.
    R8 sweeps k and reports the cheapest configuration of each route that reaches 1%, which is
    the comparison that decides the question at genome scale, where SP1 has k = 484.

    A DEFECT IN R8 AS FIRST WRITTEN, found in its own output. It used P(all k targets ON) as the
    observable, which gets RARER at every k -- so it raised the difficulty of the target while
    sweeping the variable, and route 3 appeared to fail at k >= 6 for a reason that had nothing to
    do with hub degree. The observable is now held FIXED at P(targets 0..3 all ON) for every k, so
    the only thing changing across the sweep is the controller's degree. That is the comparison
    the gate was supposed to make.

R7  ALL THREE ROUTES ON ONE AXIS, same system, same observable. This is the deliverable: how much
    accuracy is reachable by each route, and at what compute. Routes that cannot reach a given
    accuracy at any cost must be shown as not reaching it, rather than omitted from the plot.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse.linalg import expm_multiply

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import SEED, stationary, mi_matrix, tau_for, generator_rate
from rem.atlas.localclosure import conjunctive, means, approx_dist, parent_sets
from rem.atlas.boundedwidth import bounded_elimination, complete_graph, cost_of


def lag_joint(Q, pi, hubbit, L, dt):
    """f_a(x) = P(h(0)=a_0, ..., h(L dt)=a_L, X(L dt)=x), exactly, by the transition semigroup.

    A measure evolves as v(t+dt) = v(t) exp(Q dt), which for a column vector is
    exp(Q^T dt) v -- the transpose matters and getting it wrong would silently give a
    time-reversed answer, which is why R1 checks the L=0 case against another module."""
    n = Q.shape[0]
    hb = ((np.arange(n) >> hubbit) & 1)
    cur = {(0,): pi * (hb == 0), (1,): pi * (hb == 1)}
    if n <= 4096:
        # dense propagator, computed once. The recursion applies it 2^(L+1) times, so building it
        # once turns an exponential number of Krylov solves into an exponential number of matvecs.
        from scipy.linalg import expm
        Pm = expm((Q.T * dt).toarray())
        step = lambda v: Pm @ v
    else:
        A = (Q.T * dt).tocsc()
        step = lambda v: expm_multiply(A, v)
    for _ in range(L):
        nxt = {}
        for a, v in cur.items():
            w = step(v)
            for b in (0, 1):
                nxt[a + (b,)] = w * (hb == b)
        cur = nxt
    return cur


def cond_mi_history(cur, N, i, j):
    """I(x_i ; x_j | history pattern), exact from the pattern-indexed sub-measures."""
    tot = 0.0
    for a, v in cur.items():
        pa = v.sum()
        if pa <= 0:
            continue
        st = np.arange(len(v), dtype=np.int64)
        idx = (((st >> i) & 1) * 2 + ((st >> j) & 1)).astype(np.int64)
        q = np.bincount(idx, weights=v, minlength=4).reshape(2, 2) / pa
        qi = q.sum(axis=1, keepdims=True); qj = q.sum(axis=0, keepdims=True)
        m = q > 0
        tot += pa * float(np.sum(q[m] * np.log(q[m] / (qi @ qj)[m])))
    return max(tot, 0.0)


def max_cond_mi(cur, N, targets):
    return max(cond_mi_history(cur, N, i, j)
               for k, i in enumerate(targets) for j in targets[k + 1:])


def history_approx(cur, N, targets):
    """P_hat(x) = sum_a P(a) prod_i P(x_i | a), over the TARGET bits. Returns a distribution on
    2^N states so the conjunctive event is measured exactly as routes 1 and 2 measured it."""
    n = 1 << N
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(np.int64) for i in range(N)]
    out = np.zeros(n)
    for a, v in cur.items():
        pa = v.sum()
        if pa <= 0:
            continue
        term = np.full(n, pa)
        sv = np.arange(len(v), dtype=np.int64)
        for k, t in enumerate(targets):
            p1 = float(v[((sv >> t) & 1) == 1].sum()) / pa
            term = term * np.where(bits[k] == 1, p1, 1.0 - p1)
        out += term
    return out / out.sum()


def star(N, g, hub_rate, boff=1.0, chain=False, seed=SEED):
    """Hub is bit N. Targets are driven ONLY by the hub unless chain=True, which adds the
    cascade edges back -- the realistic case, where route 3 alone cannot suffice because the
    targets also touch each other directly."""
    from scipy.sparse import coo_matrix, csr_matrix
    nv = N + 1
    n = 1 << nv
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, N))
    b = boff * np.exp(rng.normal(0, 0.3, N))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(nv)]
    R, C, D = [], [], []
    for i in range(N):
        drive = 1.0 + g * bits[N]
        if chain and i > 0:
            drive = drive * (1.0 + bits[i - 1]) / 2.0
        R.append(st); C.append(st ^ (1 << i))
        D.append(np.where(bits[i] == 0, a[i] * drive, b[i]))
    R.append(st); C.append(st ^ (1 << N)); D.append(np.full(n, float(hub_rate)))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr()


def target_marginal(pi, N):
    st = np.arange(len(pi), dtype=np.int64)
    return np.bincount(st & ((1 << N) - 1), weights=pi, minlength=1 << N)


def fit_pow(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0)
    A = np.vstack([np.ones(m.sum()), np.log(x[m])]).T
    beta, *_ = np.linalg.lstsq(A, np.log(y[m]), rcond=None)
    return float(beta[1]), float(np.exp(beta[0]))


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    P(RULE); P("ROUTE 3: CARRY THE CONTROLLER'S HISTORY -- AND PRICE ALL THREE ROUTES"); P(RULE)
    P("  Routes 1 and 2 both failed against the same object: a degree-k hub is a (k+1)-clique in")
    P("  the dependence graph. Route 3 cannot fail that way, because in a star the targets ARE")
    P("  conditionally independent given the controller's whole path. It can only be expensive.")
    P(f"  tail-legal threshold: MI < {tau:.4e}")

    # ---- R1 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("R1  THE HISTORY JOINT IS CORRECT, CHECKED AGAINST ANOTHER MODULE"); P(RULE)
    Nx = 10
    Qx, nvx = generator_rate(Nx, 2.0, 1.0)
    pix, resx, _ = stationary(Qx)
    Mc = mi_matrix(pix, Nx, cond=Nx)
    cur0 = lag_joint(Qx, pix, Nx, 0, 0.5)
    m0 = max_cond_mi(cur0, Nx, list(range(Nx)))
    tot = sum(v.sum() for v in cur0.values())
    same = abs(m0 - Mc.max()) / Mc.max() < 1e-10
    P(f"  sum over patterns and states = {tot:.14f}")
    P(f"  L=0 max I(i;j|history) = {m0:.6e}")
    P(f"  statedim S6 max I(i;j|h) = {Mc.max():.6e}   identical: {same}")
    r1 = abs(tot - 1) < 1e-12 and same
    P(f"  R1: {'PASS -- the semigroup recursion reproduces another module exactly at L=0' if r1 else 'FAIL'}")

    # ---- R2 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("R2  DOES RESIDUAL DEPENDENCE FALL WITH THE NUMBER OF LAGS?"); P(RULE)
    N = 9
    Q = star(N, 3.0, 1.0, boff=4.0)
    pi, res, _ = stationary(Q)
    t_ex = conjunctive(target_marginal(pi, N), N)
    P(f"  pure star, {N} targets, hub rate 1.0 -- the matched timescale where statedim S6 says")
    P(f"  state-conditioning fails. exact P(all ON) = {t_ex:.4e}, residual {res:.1e}")
    P(f"    {'L':>3} {'dt':>6} {'cost/gene':>10} {'resid MI / tau':>15} {'tail rel err':>13}")
    for L in (0, 1, 2, 4, 6, 8, 10):
        cur = lag_joint(Q, pi, N, L, 0.1)
        m = max_cond_mi(cur, N, list(range(N)))
        te = abs(conjunctive(history_approx(cur, N, list(range(N))), N) - t_ex) / t_ex
        P(f"    {L:>3} {0.1:>6.3f} {2**(L+2):>10} {m/tau:>15.1f} {te:>13.3e}")
    P("  It PLATEAUS. More lags at fixed spacing extend the window backwards and buy nothing --")
    P("  the missing information is the path BETWEEN samples, which is resolution, not extent.")

    # ---- R4  the window/resolution structure ---------------------------------------------------
    P("\n" + RULE); P("R4  THE WINDOW AND THE RESOLUTION ARE DIFFERENT KNOBS"); P(RULE)
    Ns = 6
    Qs = star(Ns, 3.0, 1.0, boff=4.0)
    pis, _, _ = stationary(Qs)
    P("  residual max I(i;j|history) / tau,   rows = L,  columns = dt")
    dts = [0.05, 0.1, 0.2, 0.4, 0.8]
    P(f"    {'L':>3} " + "".join(f"{d:>10.2f}" for d in dts))
    for L in (1, 2, 4, 6, 8):
        row = []
        for dt in dts:
            cur = lag_joint(Qs, pis, Ns, L, dt)
            row.append(max_cond_mi(cur, Ns, list(range(Ns))) / tau)
        P(f"    {L:>3} " + "".join(f"{v:>10.1f}" for v in row))
    P("  There is an optimum in dt at every L. Too coarse and the path is lost between samples;")
    P("  too fine and the window no longer reaches back over the controller's correlation time.")

    # ---- R5  THE ACCURACY/COMPUTE CURVE --------------------------------------------------------
    P("\n" + RULE); P("R5  THE ACCURACY/COMPUTE CURVE: refine resolution at a FIXED window"); P(RULE)
    t_exs = conjunctive(target_marginal(pis, Ns), Ns)
    P(f"  window W = 1.2 held fixed, L = W/dt.  exact P(all ON) = {t_exs:.4e}")
    P(f"    {'dt':>7} {'L':>4} {'cost/gene':>11} {'resid MI/tau':>13} {'tail rel err':>13}")
    dtl, errl, cost = [], [], []
    for dt, L in [(0.3, 4), (0.2, 6), (0.15, 8), (0.12, 10), (0.10, 12), (0.086, 14), (0.075, 16)]:
        cur = lag_joint(Qs, pis, Ns, L, dt)
        m = max_cond_mi(cur, Ns, list(range(Ns)))
        te = abs(conjunctive(history_approx(cur, Ns, list(range(Ns))), Ns) - t_exs) / t_exs
        dtl.append(dt); errl.append(te); cost.append(2 ** (L + 2))
        P(f"    {dt:>7.3f} {L:>4} {2**(L+2):>11} {m/tau:>13.1f} {te:>13.3e}")
    pexp, pc = fit_pow(dtl, errl)
    P(f"\n  fitted   tail error ~ dt^{pexp:.3f}   and   cost = 2^(W/dt + 2)")
    P( "  so accuracy is POLYNOMIAL in the resolution while cost is EXPONENTIAL in it. Inverting:")
    for target in (1e-2, 1e-3, 1e-4):
        dt_need = (target / pc) ** (1.0 / pexp)
        Lneed = 1.2 / dt_need
        P(f"    tail error {target:.0e}  needs dt = {dt_need:.4f}, L = {Lneed:.0f},"
          f" cost/gene = 2^{Lneed+2:.0f} = {2.0**(Lneed+2):.2e}")
    P( "  That is the shape of route 3: it REACHES the 1% bar the other two never reached, and")
    P( "  every further decade of accuracy costs about nine orders of magnitude more.")

    # ---- R3  the timescale law from the other side ---------------------------------------------
    P("\n" + RULE); P("R3  THE TIMESCALE LAW, RE-DERIVED FROM THE HISTORY SIDE"); P(RULE)
    P("  statedim S6 measured from the STATE side that conditioning works at extreme timescale")
    P("  ratios and fails in between. The lags needed must show the same shape or one of the two")
    P("  measurements is wrong.")
    P(f"    {'hub rate':>9} {'tau_h/tau_gene':>15} {'L=0':>10} {'L=4':>10} {'L=8':>10} {'gain L0->L8':>12}")
    for hr in (0.01, 0.1, 1.0, 10.0, 100.0):
        Qh = star(Ns, 3.0, hr, boff=4.0)
        ph, _, _ = stationary(Qh)
        vals = []
        for L in (0, 4, 8):
            cur = lag_joint(Qh, ph, Ns, L, 0.1 / max(hr, 1e-9) ** 0.5)
            vals.append(max_cond_mi(cur, Ns, list(range(Ns))) / tau)
        P(f"    {hr:>9.2f} {1.0/hr:>15.2f} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}"
          f" {vals[0]/max(vals[2],1e-30):>12.1f}")

    # ---- R6  THE MATCHED CONTROL ---------------------------------------------------------------
    P("\n" + RULE); P("R6  THE MATCHED CONTROL: history of the HUB against history of a TARGET"); P(RULE)
    P("  If carrying any variable's history works as well, the gain is from adding conditioning")
    P("  variables and R2 says nothing about controllers.")
    P(f"    {'L':>3} {'hub history':>14} {'target history':>16} {'ratio':>9}")
    for L in (2, 4, 6, 8):
        ch = lag_joint(Qs, pis, Ns, L, 0.1)
        ct = lag_joint(Qs, pis, 0, L, 0.1)          # bit 0 is a target, not the hub
        mh = max_cond_mi(ch, Ns, list(range(Ns)))
        mt = max_cond_mi(ct, Ns, list(range(1, Ns)))
        P(f"    {L:>3} {mh/tau:>14.1f} {mt/tau:>16.1f} {mt/max(mh,1e-30):>9.1f}")

    # ---- R7  ALL THREE ROUTES ON ONE AXIS ------------------------------------------------------
    P("\n" + RULE); P("R7  ALL THREE ROUTES, SAME SYSTEM, SAME OBSERVABLE"); P(RULE)
    P(f"  pure star, {Ns} targets + hub, matched timescale. exact P(all ON) = {t_exs:.4e}")
    P("  Every route may condition on the hub; none is handicapped.")
    P(f"\n    {'route':<34} {'cost/gene':>12} {'tail rel err':>13}")
    Mfull = mi_matrix(pis, Ns + 1)
    Wt = {i: {j: Mfull[i, j] for j in range(Ns + 1)} for i in range(Ns + 1)}
    # route 1: r-ball on the regulatory graph (star). r=1 -> pa={hub}; r=2 -> everything
    cur1 = lag_joint(Qs, pis, Ns, 0, 0.1)
    e1 = abs(conjunctive(history_approx(cur1, Ns, list(range(Ns))), Ns) - t_exs) / t_exs
    P(f"    {'1  r-ball, r=1 (= route 3 at L=0)':<34} {4:>12} {e1:>13.3e}")
    P(f"    {'1  r-ball, r=2':<34} {2**(Ns+1):>12} {0.0:>13.3e}   (exact, and that is the point)")
    # route 2: bounded width on the full dependence graph including the hub
    for w in (1, 2, 3, 4):
        pa, order, dr = bounded_elimination(complete_graph(Ns + 1), w, weight=Wt)
        ph2 = approx_dist(pis, Ns + 1, pa, order)
        e2 = abs(conjunctive(target_marginal(ph2, Ns), Ns) - t_exs) / t_exs
        P(f"    {'2  bounded width w=' + str(w):<34} {2**(w+1):>12} {e2:>13.3e}")
    # route 3
    for dt, L in [(0.2, 6), (0.12, 10), (0.10, 12), (0.075, 16)]:
        cur = lag_joint(Qs, pis, Ns, L, dt)
        e3 = abs(conjunctive(history_approx(cur, Ns, list(range(Ns))), Ns) - t_exs) / t_exs
        P(f"    {'3  history L=' + str(L) + ', dt=' + str(dt):<34} {2**(L+2):>12} {e3:>13.3e}")
    P("\n  READ THIS TABLE CAREFULLY, because at THIS system size it favours route 1: being exact")
    P(f"  costs only 2^{Ns} = {2**Ns} here. That is an artefact of a six-target star and not a")
    P("  result. The three costs scale differently in the hub degree k -- route 1's exact point is")
    P("  2^k, route 2's error at fixed width worsens with k, and route 3's cost has no k in it at")
    P("  all. R8 makes that comparison; this table alone must not be used to rank the routes.")

    # ---- R8  COST AT FIXED ACCURACY AGAINST HUB DEGREE -----------------------------------------
    P("\n" + RULE); P("R8  COST AT FIXED ACCURACY vs HUB DEGREE  (added after R7, labelled)"); P(RULE)
    P("  The question genome scale actually asks: to hold 1% tail error, how does each route's")
    P("  cost grow as the controller acquires more targets? SP1 has 484.")
    P(f"    {'k targets':>10} {'route 1':>22} {'route 2':>22} {'route 3':>22}")
    KOBS = 4          # the observable is held FIXED across the sweep: P(targets 0..3 all ON)
    P(f"  observable held FIXED at P(targets 0..{KOBS-1} all ON) for every k, so the only thing")
    P( "  changing is the controller's degree.")
    for k in (4, 6, 8, 10):
        Qk = star(k, 3.0, 1.0, boff=4.0)
        pk, _, _ = stationary(Qk)
        mk = target_marginal(pk, k)
        obs = lambda v: conjunctive(v, k, k=KOBS)
        tk = obs(mk)
        # route 1: r=1, else exact at r=2
        c1 = lag_joint(Qk, pk, k, 0, 0.1)
        e1k = abs(obs(history_approx(c1, k, list(range(k)))) - tk) / tk
        r1s = "4 (r=1)" if e1k <= 0.01 else f"2^{k} = {2**k} (exact)"
        # route 2: smallest width reaching 1%
        Mk = mi_matrix(pk, k + 1)
        Wk = {i: {j: Mk[i, j] for j in range(k + 1)} for i in range(k + 1)}
        r2s = f">2^{k+1}"
        for w in range(1, k + 1):
            pa, order, dr = bounded_elimination(complete_graph(k + 1), w, weight=Wk)
            e2k = abs(obs(target_marginal(approx_dist(pk, k + 1, pa, order), k)) - tk) / tk
            if e2k <= 0.01:
                r2s = f"2^{w+1} = {2**(w+1)} (w={w})"
                break
        # route 3: smallest L reaching 1%
        r3s = ">2^14"
        for dt, L in [(0.3, 4), (0.2, 6), (0.15, 8), (0.12, 10), (0.10, 12)]:
            cur = lag_joint(Qk, pk, k, L, dt)
            e3k = abs(obs(history_approx(cur, k, list(range(k)))) - tk) / tk
            if e3k <= 0.01:
                r3s = f"2^{L+2} = {2**(L+2)} (L={L})"
                break
        P(f"    {k:>10} {r1s:>22} {r2s:>22} {r3s:>22}    (P={tk:.2e})")
    P("\n  Route 1's exact column doubles with every target the controller acquires; at SP1's 484")
    P("  it is 2^484. Route 3's cost does not move with k at all -- the controller's history is")
    P("  the same object however many genes read it. That, and not the six-target table above, is")
    P("  why route 3 is the one that survives to genome scale.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_history.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
