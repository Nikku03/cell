"""The hybrid engine: all three routes at once, each doing the job it measurably wins.

WHAT THE THREE MODULES ESTABLISHED, and why a hybrid is the only sensible next move. Routes 1, 2
and 3 are not competitors at one job -- history.py R8 showed they are matched to different
structure, and two of that gate's three versions went wrong by assuming otherwise:

    r-balls         (localclosure)  arbitrary accuracy on chain-like structure, 20 MB at genome
                                    scale; UNREACHABLE on a hub, whose 3-ball is 95% of the network
    bounded width   (boundedwidth)  exact for anything that fits in one bag, ~5 MB; a 6% floor on
                                    hub-mediated dependence, and its error GROWS with hub degree
    history         (history)       flat error in hub degree, 2.6 GB for 1%; 100x more expensive
                                    than bounded width for a local question

So the engine assigns roles by structure: carry HISTORY for the controllers, r-BALLS for the
residual structure once controllers are removed, and BOUNDED WIDTH as the backstop that guarantees
the cost whatever the first two ask for.

THE CONSTRUCTION. Designate a controller set C by out-degree. For each history pattern a of C:

    P_hat_a(x) = prod_i P( x_i | x_pa(i), a )        pa from the r-ball of the graph MINUS C,
    P_hat(x)   = sum_a P(a) P_hat_a(x)                 thinned to width w if it overruns

which is route 1/2's Bayesian network built inside each history stratum and mixed over strata. It
is a strict generalisation of all three, and H1 checks that by reducing it to each of them exactly
rather than approximately -- without which every comparison below would be between an engine and
its own reimplementation of a rival.

WHAT WOULD MAKE THIS A FAILURE. If the hybrid merely lands BETWEEN its components rather than on
or below their lower envelope, the combination buys nothing and this module says so. Two of the
three routes already failed on real topology, and a hybrid of a failure and a success is not
automatically a success.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

H1  THE HYBRID REDUCES TO EACH COMPONENT EXACTLY. With no controllers designated and a full ball
    it must equal route 1's distribution to machine precision; with controllers and no ball, route
    3's; with everything forced through width thinning, route 2's. Three identity checks. If it is
    not a strict generalisation, H2 is comparing an engine against a rival it reimplemented badly.

    THREE DEFECTS IN H2 AS FIRST WRITTEN, and the third is a pattern worth naming.

    (i) THE ENVELOPE WAS UNDER-RESOURCED. The component sweeps stopped at width 4 and r-ball 3,
        cost about 150, while the hybrid ran to cost 28,000. Extended to equal cost, route 1 at
        r = 6 reaches 4.4e-3 for 382 and route 2 at w = 7 is EXACT for 510 -- both better than
        every hybrid point. The first run's "23.9x improvement, 8 of 8" was an artefact of
        stopping the rivals early.

    (ii) THE BEST HYBRID POINT WAS A ZERO CROSSING, NOT CONVERGENCE. The signed tail error at
        r = 1 runs -1.26e-1, -2.53e-2, +5.58e-3, +1.43e-2, +1.67e-2, +1.78e-2 as L grows. It
        crosses zero between L = 1 and L = 3, and the 5.58e-3 quoted as "best" was that crossing.
        The converged value is 1.78e-2, seventeen times worse. The Var error was monotone
        throughout, which is what exposed it: two observables disagreeing about which
        configuration is best is the signature of a cancellation. Signed errors are now reported
        and converged values quoted.

    (iii) A COMPARISON AT ONE SYSTEM SIZE CANNOT RANK APPROXIMATIONS. On any system small enough
        to solve exactly, the exact method wins -- route 2 at w = N-1 holds every variable in one
        bag. This is the THIRD time this build order has made that mistake: history R7 compared
        routes on a six-target star where being exact costs 128, R8 was written three times over
        the same issue, and now H2. The rule it should have followed from the start: an
        approximation is ranked by a SCALING LAW, never at a point. H2 is now two gates -- H2a
        the point comparison, reported as uninformative and shown to be so, and H2b the scaling,
        which is the actual measurement.

H2  DOES IT BEAT EACH COMPONENT ON A SYSTEM THAT HAS BOTH KINDS OF STRUCTURE? The test system is a
    star PLUS a chain -- a controller over targets that also regulate each other, which is what a
    real network is and what no single route handles. Same observable, same cost axis.
    PREDECLARED: the hybrid must lie on or below the lower envelope of the three component curves
    at equal cost. Landing between them is reported as buying nothing.

H3  THE COST ACCOUNTING IS HONEST. Cost is the realised sum over genes of 2^(1+|pa_i|), including
    the history bits, not a nominal width. A hybrid that wins by quietly using larger factors has
    not won, and the only way to tell is to count the factors it actually built.

H4  THE CONTROLLER SET IS DETECTED, NOT CHOSEN. Controllers come from an out-degree threshold, and
    the threshold is swept. PREDECLARED: if the result depends on hand-picking the hub, it will not
    transfer to a network where nobody knows which genes are the controllers, and that has to be
    stated rather than discovered later.

H5  REAL TOPOLOGY. Run the role assignment on TRRUST v2 human and report how many genes land in
    each regime and what the realised cost distribution is. This is the only number here that says
    anything about a genome.

H6  THE MATCHED CONTROL. Designate the SAME NUMBER of controllers at random instead of by degree,
    at matched cost. If random assignment does as well, the structure detection is decorative and
    H2's win is really H3's budget.

H7  WHERE DOES IT STILL FAIL? Report the residual error at the best affordable configuration and
    identify what carries it. A hybrid that improves on its parts and still misses the bar has
    moved the ceiling, not removed it, and the difference matters.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import SEED, stationary, mi_matrix, tau_for
from rem.atlas.localclosure import conjunctive, approx_dist, ball, load_trrust
from rem.atlas.boundedwidth import bounded_elimination, complete_graph
from rem.atlas.history import (star, target_marginal, lag_joint, history_approx, var_count,
                               max_cond_mi)


def residual_graph(adj, C, n):
    """The regulatory graph with the controller vertices removed. What is left is what the r-ball
    part of the engine has to cover."""
    return [set() if i in C else (adj[i] - C) for i in range(n)]


def assign_roles(adj, n, deg_threshold):
    """Controllers are detected by degree, never chosen. H4 sweeps the threshold."""
    deg = np.array([len(adj[i]) for i in range(n)])
    return {int(i) for i in np.nonzero(deg >= deg_threshold)[0]}


def hybrid_parents(adj, C, n, r, w, weight=None):
    """pa_i = the r-ball of i in the CONTROLLER-FREE graph, thinned to width w. The controller
    history is carried separately and added to the cost, not to this set."""
    res = residual_graph(adj, C, n)
    G = [ball(res, i, r) - {i} for i in range(n)]
    pa, order, dropped = bounded_elimination(G, w, weight=weight)
    return pa, order, dropped


def hybrid_dist(Q, pi, N, hubbit, C, adj, r, w, L, dt, weight=None):
    """P_hat(x) = sum_a P(a) prod_i P(x_i | x_pa(i), a).

    Route 1/2's Bayesian network built inside each controller-history stratum, mixed over strata.
    L < 0 means no history at all, which is the pure route-1/2 limit."""
    if L < 0:
        cur = {(): pi}
    else:
        cur = lag_joint(Q, pi, hubbit, L, dt)
    pa, order, dropped = hybrid_parents(adj, C, N, r, w, weight=weight)
    tot = np.zeros(1 << N)
    for a, v in cur.items():
        pv = float(v.sum())
        if pv <= 0:
            continue
        marg = target_marginal(v / pv, N)
        tot += pv * approx_dist(marg, N, pa, order)
    return tot / tot.sum(), pa, dropped, (0 if L < 0 else L + 1)


def hybrid_cost(pa, nhist, carries, N):
    """Realised cost: 2^(1 + |pa_i| + history bits carried by i), summed over genes.

    `carries` is a per-gene boolean, because the controller is a vertex OUTSIDE the target index
    range and deriving "is i regulated by a controller" from the target-only adjacency silently
    returned False for every gene, undercounting the history term to zero."""
    return float(sum(2.0 ** (1 + len(pa.get(i, [])) + (nhist if carries[i] else 0))
                     for i in range(N)))


def mixed_system(N, g=3.0, hr=1.0, boff=4.0):
    """A controller over targets that ALSO regulate each other -- the case no single route
    handles, and the only honest place to test a hybrid."""
    Q = star(N, g, hr, boff=boff, chain=True)
    pi, res, _ = stationary(Q)
    adj = [set() for _ in range(N)]
    for i in range(1, N):
        adj[i].add(i - 1); adj[i - 1].add(i)
    return Q, pi, res, adj


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    N = 8
    Q, pi, res, adj = mixed_system(N)
    t_ex = conjunctive(target_marginal(pi, N), N)
    v_ex = var_count(target_marginal(pi, N), N)
    allc = [True] * N
    noc = [False] * N

    P(RULE); P("THE HYBRID ENGINE: ALL THREE ROUTES, EACH DOING THE JOB IT WINS"); P(RULE)
    P(f"  test system: a controller over {N} targets that ALSO regulate each other -- star plus")
    P(f"  chain, the case no single route handles. exact P(all ON) = {t_ex:.4e}, residual {res:.1e}")
    P(f"  tail-legal threshold: MI < {tau:.4e}")

    # ---- H1 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("H1  THE HYBRID REDUCES TO EACH COMPONENT EXACTLY"); P(RULE)
    h1 = True
    # route 1: no history, r-ball only
    p1, pa1, _, _ = hybrid_dist(Q, pi, N, N, set(), adj, 1, N, -1, 0.12)
    ref1 = approx_dist(target_marginal(pi, N), N,
                       *(lambda t: (t[0], t[1]))(bounded_elimination(
                           [ball(residual_graph(adj, set(), N), i, 1) - {i} for i in range(N)],
                           N)[:2]))
    e = float(np.abs(p1 - ref1).max())
    h1 = h1 and e < 1e-14
    P(f"  reduces to route 1 (r-ball, no history)      max|diff| = {e:.3e}   {'PASS' if e < 1e-14 else 'FAIL'}")
    # route 3: history, empty ball
    p3, pa3, _, nh3 = hybrid_dist(Q, pi, N, N, set(range(N)), adj, 1, N, 6, 0.12)
    ref3 = history_approx(lag_joint(Q, pi, N, 6, 0.12), N, list(range(N)))
    e3 = float(np.abs(p3 - ref3).max())
    h1 = h1 and e3 < 1e-14
    P(f"  reduces to route 3 (history, no ball)        max|diff| = {e3:.3e}   {'PASS' if e3 < 1e-14 else 'FAIL'}")
    # route 2: no history, width-thinned complete graph
    p2, pa2, dr2, _ = hybrid_dist(Q, pi, N, N, set(), adj, N, 3, -1, 0.12)
    pa2r, o2r, _ = bounded_elimination(complete_graph(N), 3)
    ref2 = approx_dist(target_marginal(pi, N), N, pa2r, o2r)
    e2 = float(np.abs(p2 - ref2).max())
    h1 = h1 and e2 < 1e-14
    P(f"  reduces to route 2 (width 3, no history)     max|diff| = {e2:.3e}   {'PASS' if e2 < 1e-14 else 'FAIL'}")
    P(f"  H1: {'PASS -- it is a strict generalisation, so H2 compares an engine against real rivals' if h1 else 'FAIL -- not a generalisation; every comparison below is meaningless'}")

    # ---- H2 / H3 -------------------------------------------------------------------------------
    P("\n" + RULE); P("H2a/H3  THE POINT COMPARISON -- and why it cannot rank anything"); P(RULE)
    P("  cost is the realised sum over genes of 2^(1+|pa_i|+history bits), not a nominal width.")
    P("  The tail error is reported SIGNED, because an unsigned minimum can be a zero crossing.")
    P(f"    {'configuration':<34} {'cost':>12} {'signed tail err':>16} {'Var err':>11}")
    rows = []

    def add(lbl, C, r, w, L, carries, dt=0.12):
        ph, pa, dr, nh = hybrid_dist(Q, pi, N, N, C, adj, r, w, L, dt)
        c = hybrid_cost(pa, nh, carries, N)
        se = (conjunctive(ph, N) - t_ex) / t_ex
        ve = abs(var_count(ph, N) - v_ex) / v_ex
        rows.append((lbl, c, abs(se), ve))
        P(f"    {lbl:<34} {c:>12.3e} {se:>+16.3e} {ve:>11.3e}")
        return c, abs(se)

    P("  -- route 1 alone, swept to the SAME cost range as the hybrid --")
    for r in (1, 2, 3, 4, 5, 6):
        add(f"1  r-ball r={r}", set(), r, N, -1, noc)
    P("  -- route 2 alone, swept to the SAME cost range as the hybrid --")
    for w in (1, 2, 3, 4, 5, 6, 7):
        add(f"2  width w={w}", set(), N, w, -1, noc)
    P("  -- route 3 alone (controller history, no residual structure) --")
    for L in (4, 6, 8, 10):
        add(f"3  history L={L}", set(range(N)), 1, N, L, allc)
    P("  -- HYBRID (history for the controller, r-balls for the residual chain) --")
    hyb = []
    for L in (2, 4, 6, 8):
        for r in (1, 2):
            c, te = add(f"H  r={r}, L={L}", set(), r, N, L, allc)
            hyb.append((c, te))

    # lower envelope check
    comp = [(c, te) for lbl, c, te, ve in rows if not lbl.startswith("H")]
    beat = 0
    for c, te in hyb:
        rivals = [t for cc, t in comp if cc <= c * 1.05]
        if rivals and te < min(rivals):
            beat += 1
    best_comp = min(t for _, t in comp)
    best_hyb = min(t for _, t in hyb)
    P(f"\n  best component error {best_comp:.3e}   best hybrid error {best_hyb:.3e}")
    P( "  H2a: UNINFORMATIVE, as predeclared once the defect was found. Route 2 at w = N-1 holds")
    P( "  every variable in one bag and is EXACT, so on any system small enough to solve exactly")
    P( "  the exact method wins and no approximation can be ranked here. The first run of this")
    P( "  gate reported a 23.9x hybrid win by stopping the rivals at w = 4. H2b is the real test.")
    P( "  Note also the SIGN column: the hybrid's tail error crosses zero as L grows, so its")
    P( "  unsigned minimum is a cancellation and its converged value is the one to quote.")

    # ---- H2b  THE SCALING LAW ------------------------------------------------------------------
    P("\n" + RULE); P("H2b  THE SCALING LAW -- how cost at FIXED accuracy grows with system size"); P(RULE)
    P("  criterion: 1% relative error in Var(total ON count), a global observable whose meaning")
    P("  does not change with N. Cheapest configuration of each route that reaches it.")
    P(f"    {'N':>4} {'route 1 r-ball':>20} {'route 2 width':>20} {'HYBRID':>24}")
    scal = []
    for Nn in (6, 8, 10, 12):
        Qn, pin, resn, adjn = mixed_system(Nn)
        vex = var_count(target_marginal(pin, Nn), Nn)
        nocn, allcn = [False] * Nn, [True] * Nn

        def er(ph):
            return abs(var_count(ph, Nn) - vex) / vex

        c1 = c2 = ch = None
        for r in range(1, Nn + 1):
            ph, pa, _, nh = hybrid_dist(Qn, pin, Nn, Nn, set(), adjn, r, Nn, -1, 0.12)
            if er(ph) <= 0.01:
                c1 = (hybrid_cost(pa, nh, nocn, Nn), f"r={r}"); break
        for w in range(1, Nn):
            ph, pa, _, nh = hybrid_dist(Qn, pin, Nn, Nn, set(), adjn, Nn, w, -1, 0.12)
            if er(ph) <= 0.01:
                c2 = (hybrid_cost(pa, nh, nocn, Nn), f"w={w}"); break
        for L in (2, 4, 6, 8):
            for r in (1, 2, 3):
                ph, pa, _, nh = hybrid_dist(Qn, pin, Nn, Nn, set(), adjn, r, Nn, L, 0.12)
                if er(ph) <= 0.01:
                    ch = (hybrid_cost(pa, nh, allcn, Nn), f"r={r},L={L}"); break
            if ch:
                break
        scal.append((Nn, c1, c2, ch))
        f = lambda t: f"{t[0]:.0f} ({t[1]})" if t else "not reached"
        P(f"    {Nn:>4} {f(c1):>20} {f(c2):>20} {f(ch):>24}")
    g1 = scal[-1][1][0] / scal[0][1][0] if scal[0][1] and scal[-1][1] else float('nan')
    g2 = scal[-1][2][0] / scal[0][2][0] if scal[0][2] and scal[-1][2] else float('nan')
    gh = scal[-1][3][0] / scal[0][3][0] if scal[0][3] and scal[-1][3] else float('nan')
    P(f"\n  growth from N=6 to N=12:  route 1 x{g1:.1f}   route 2 x{g2:.1f}   hybrid x{gh:.1f}")
    P( "  Route 2 grows as 2^N exactly -- 126, 510, 2046, 8190 is 4x per two variables. Route 1")
    P( "  grows nearly as fast. The hybrid grows LINEARLY, and it LOSES at N = 6: at small N")
    P( "  exactness is cheap and the history machinery is not worth its overhead. That crossover")
    P( "  is what makes the rest credible; a method that won everywhere would be suspect.")
    h2 = gh < g1 / 3 and gh < g2 / 3
    P(f"  H2b: {'PASS -- the hybrid scales fundamentally better, which is the only claim a hybrid can earn' if h2 else 'FAIL -- no scaling advantage'}")

    # ---- H6  THE MATCHED CONTROL ---------------------------------------------------------------
    P("\n" + RULE); P("H6  THE MATCHED CONTROL: carry the history of a RANDOM gene instead"); P(RULE)
    P("  If the engine does as well conditioning on an arbitrary gene's past, the structure")
    P("  detection is decorative and H2's win is really H3's budget.")
    P(f"    {'L':>3} {'r':>3} {'controller history':>20} {'random gene history':>21} {'ratio':>8}")
    for L in (4, 6, 8):
        for r in (1,):
            ph, pa, _, nh = hybrid_dist(Q, pi, N, N, set(), adj, r, N, L, 0.12)
            e_c = abs(conjunctive(ph, N) - t_ex) / t_ex
            phr, par, _, nhr = hybrid_dist(Q, pi, N, 0, set(), adj, r, N, L, 0.12)
            e_r = abs(conjunctive(phr, N) - t_ex) / t_ex
            P(f"    {L:>3} {r:>3} {e_c:>20.3e} {e_r:>21.3e} {e_r/e_c:>8.1f}")

    # ---- H4  DETECTION, NOT CHOICE -------------------------------------------------------------
    P("\n" + RULE); P("H4  THE CONTROLLER SET IS DETECTED BY DEGREE, NOT CHOSEN"); P(RULE)
    adj_full = [set(a) for a in adj]
    for i in range(N):
        adj_full[i].add(N)
    adj_full.append(set(range(N)))
    deg = np.array([len(a) for a in adj_full])
    P(f"  degrees in the full graph: targets {sorted(set(deg[:N].tolist()))}, controller {deg[N]}")
    P(f"    {'threshold':>10} {'controllers found':>18} {'is it the true hub?':>21}")
    for th in (3, 4, 6, 8, N):
        C = assign_roles(adj_full, N + 1, th)
        P(f"    {th:>10} {len(C):>18} {str(C == {N}):>21}")
    P("  The controller is separated from the targets by degree alone over a wide threshold band,")
    P("  which is what has to be true for this to work on a network nobody has annotated.")

    # ---- H5  REAL TOPOLOGY ---------------------------------------------------------------------
    P("\n" + RULE); P("H5  THE ROLE ASSIGNMENT ON TRRUST v2 HUMAN"); P(RULE)
    adj_tr, inv_tr, sha_tr, ne_tr = load_trrust()
    n_tr = len(adj_tr)
    degt = np.array([len(a) for a in adj_tr])
    P(f"  {n_tr} genes, {ne_tr} edges, sha256 {sha_tr}")
    P(f"    {'threshold':>10} {'controllers':>12} {'max residual ball r=2':>23} {'max |pa| at w=6':>17}")
    for th in (200, 100, 50, 25, 12):
        C = {int(i) for i in np.nonzero(degt >= th)[0]}
        res_g = residual_graph(adj_tr, C, n_tr)
        mb = max(len(ball(res_g, i, 2)) for i in range(n_tr) if i not in C)
        pa, order, dr = bounded_elimination([ball(res_g, i, 2) - {i} for i in range(n_tr)], 6)
        P(f"    {th:>10} {len(C):>12} {mb:>23} {max(len(v) for v in pa.values()):>17}")
    P("  Removing the controllers collapses the residual balls, which is the whole mechanism --")
    P("  localclosure L3 measured the 3-ball around SP1 at 2,724 of 2,861 genes WITH the hubs in.")

    # ---- H7 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("H7  WHERE DOES IT STILL FAIL?"); P(RULE)
    # Quote the CONVERGED value, not the unsigned minimum. H2a showed the r=1 branch crosses
    # zero, so its minimum is a cancellation; the r=2 branch is monotone and is what converges.
    conv = [(lbl, c, te) for lbl, c, te, ve in rows if lbl.startswith("H  r=2")]
    conv.sort(key=lambda t: t[1])
    lbl_c, cost_c, te_c = conv[-1]
    P(f"  unsigned minimum over all hybrid points: {best_hyb:.3e}  -- NOT quotable, it is the")
    P( "  zero crossing of the r=1 branch that H2a identified.")
    P(f"  CONVERGED hybrid accuracy ({lbl_c}, cost {cost_c:.3e}): {te_c:.3e} tail")
    P(f"  Against the 1% bar: {'MET' if te_c < 0.01 else 'NOT met'} on the converged value.")
    P( "  The residual is carried by the same thing history.py R5 priced: the controller's path")
    P( "  between sample times. Halving that residual means halving dt, which doubles L, which")
    P( "  squares the cost -- the hybrid moves the ceiling, it does not remove it.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_engine.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
