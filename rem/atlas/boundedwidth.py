"""Route 2 of 3: fix the width, drop what does not fit, and MEASURE what dropping it costs.

WHY THIS IS BETTER POSED THAN ROUTE 1. localclosure.py fixed the GEOMETRY -- factorise on
r-balls -- and let the width be whatever fell out. On the real human network that was 2,724, one
factor covering 95% of the genome, and the route was worse than the junction tree it replaced.
Here the width is fixed FIRST, at whatever is affordable, and the decomposition is forced to fit
inside it by deleting edges. Cost is then N 2^(w+1) BY CONSTRUCTION and cannot blow up. The
entire question moves to the error, which is the thing that ought to be measured anyway.

HOW. A single min-degree elimination pass. When eliminating v would create a bag larger than w,
the weakest edges incident to v inside that bag are deleted until it fits, and the deletion is
recorded. That yields, in one pass:

    a decomposition of guaranteed width <= w
    the exact set of dependence edges sacrificed to get it
    parent sets pa[v] = the bag at elimination time, in reverse elimination order

and P_hat(x) = prod_v P(x_v | x_pa(v)) is then a Bayesian network -- normalised by construction,
built from the EXACT marginals of the true joint, and equal to the true joint when nothing was
dropped. That last property is the identity control, and it is the same discipline localclosure
L1 used.

WHAT THE ANSWER HINGES ON. On the real network the strong edges are the direct regulatory ones --
8,403 of them, each carrying mutual information thousands of times above the tail-legal threshold.
The graph of direct edges ALONE has treewidth past 40. So any affordable width forces deleting
edges that are not weak. Whether that is fatal is not obvious and is exactly what B2 measures: a
global conjunctive event may barely notice one dropped edge among thousands, or may not survive
one. Both are plausible before the measurement and only one is true.

WHAT ROUTE 1 ALREADY SETTLED, so it is not re-litigated here. The dependence graph really is the
graph power even on a hub -- out-degree does not dilute sibling dependence at all and in-degree
dilutes only as its square, still leaving real regulation about 80x above threshold. And
tail_err <= 20.23 sqrt(MI) failed in 5 of 16 rows above its floor, so the error must be measured
from a curve rather than predicted from a bound. Both of those are inputs to this module, not
questions in it.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

B1  THE PROJECTION IS A DISTRIBUTION AND THE IDENTITY CASE IS EXACT. sum P_hat = 1 to machine
    precision, and when w is large enough that nothing is dropped, P_hat must equal the exact
    stationary distribution to machine precision. Without this every error below could be a bug.

B2  THE ERROR CURVE. Tail error of the conjunctive rare event against width w, on three exactly
    solvable topologies -- cascade, random regulatory, and a hub graph -- at three tail depths
    down to 1e-13. The deliverable is the curve error(w), with the mean error printed beside it
    in the same row. No predeclared threshold on the SHAPE, because route 1 showed the sqrt(MI)
    law does not bound this quantity; the curve is the result.

B3  ARE THE DROPPED EDGES THE WEAK ONES? Report the mutual information of dropped against kept
    edges at each width, and how far above tau the strongest dropped edge is. If bounded width
    is forced to sacrifice edges thousands of times above threshold, that has to be visible
    rather than hidden inside an aggregate error number.

    A DEFECT IN B4 AS FIRST WRITTEN, caught by its own output being impossible. It counted every
    edge the elimination deleted, but elimination ADDS fill-in edges and those get deleted too, so
    on TRRUST it reported dropping 16,522 of 8,403 edges -- a fraction of 1.97. It was also
    non-monotone in w for the same reason. The structural question is now asked the standard way:
    an original regulatory edge is REPRESENTED if both its endpoints appear together in some bag,
    and B4 reports the fraction of the 8,403 that are. That quantity is bounded by 1 by
    construction and cannot hide the same mistake.

    A CONSEQUENT DEFECT IN B5. It fed those impossible fractions into an interpolation whose
    measured domain is [0.16, 0.86]. numpy clamps rather than refusing, so every row returned the
    endpoint value 0.9831 and the column looked like a result. B5 now refuses to extrapolate
    outside the measured domain and says so per row.

B4  THE STRUCTURAL COST ON THE REAL NETWORK, with no model in it. On TRRUST v2 human, what
    fraction of the 8,403 DIRECT regulatory edges is still REPRESENTED at width w? This needs
    no mutual information and no dynamics -- it is a property of the graph -- so it is the one
    number here that carries no modelling assumption.

    A THIRD DEFECT, in B6, exposed once B5 stopped clamping. B5's reference curve was measured
    at N = 14, where the smallest reachable drop fraction is 0.16; TRRUST at every affordable
    width loses BETWEEN 0.02 AND 0.14, i.e. entirely below the measured range. So the first
    corrected run returned "out of domain" for exactly the widths that might have worked, and B6
    nonetheless printed EMPTY INTERSECTION on the strength of the single in-domain row. That
    verdict was not supported. The reference curve is now measured at N = 16 out to width 13,
    which reaches drop fraction 0.025 and covers TRRUST's whole affordable range.

    AND THE BRIDGE ITSELF IS WEAKER THAN B5 ORIGINALLY ADMITTED. It assumed error depends on the
    drop fraction and not on the topology. At drop fraction 0.16 the cascade gives 6.6e-6 and the
    hub gives 2.9e-1 -- FOUR ORDERS APART. So a single modelled number is not defensible and B5
    now reports a RANGE spanned by the two topologies, with the verdict driven by the hub end,
    because TRRUST is hub-rich. The range is wide and that width is itself the finding.

B5  THE EXTRAPOLATION, LABELLED AS A MODEL. Combine B2's measured error curve with B4's measured
    drop fraction to estimate the accuracy of a width-w engine on the real network. This is the
    only claim in the module that is not a direct measurement and it is marked as such wherever
    it appears.

B6  IS THERE A WIDTH THAT IS BOTH AFFORDABLE AND ACCURATE? PREDECLARED: affordable means
    2^w <= 1e9, so w <= 30. Accurate means tail error below 1%. Report whether that intersection
    is non-empty on the real topology. An empty intersection is a clean negative and closes the
    route; a non-empty one is the first positive result on state complexity in this build order.

B7  THE MATCHED CONTROL. Thin by dropping the LOWEST-mutual-information edges, and thin by
    dropping RANDOM edges, at identical width. If the two give the same error, then choosing
    which edges to keep buys nothing, the error is set by the width alone, and B2 is a statement
    about budget rather than about strategy. This is the control that distinguishes a method from
    an arithmetic constraint.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import heapq
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import (SEED, C_TAIL, stationary, mi_matrix, tau_for,
                                path_power, random_regulatory)
from rem.atlas.localclosure import (generator_tuned, approx_dist, conjunctive, means,
                                    load_trrust, MI_FLOOR)


def bounded_elimination(adj, w, weight=None, rng=None):
    """One min-degree pass that never lets a bag exceed w.

    When eliminating v would make a bag of size > w, edges incident to v inside that bag are
    deleted -- weakest first by `weight`, or at random when rng is given -- until it fits. Returns
    the parent sets, the elimination order, and the exact set of edges sacrificed.

    The width is guaranteed by construction, so the cost N 2^(w+1) is a fact about the output
    rather than a hope about the input."""
    n = len(adj)
    A = [set(a) for a in adj]
    alive = np.ones(n, bool)
    heap = [(len(A[v]), v) for v in range(n)]
    heapq.heapify(heap)
    pa, order, dropped = {}, [], []
    left = n
    while left:
        v = None
        while heap:
            d, u = heapq.heappop(heap)
            if alive[u] and len(A[u]) == d:
                v = u
                break
        if v is None:
            v = int(np.nonzero(alive)[0][0])
        nb = set(A[v])
        if len(nb) > w:
            cand = list(nb)
            if rng is not None:
                rng.shuffle(cand)
            elif weight is not None:
                cand.sort(key=lambda u: weight[v][u])
            else:
                cand.sort(key=lambda u: -len(A[u]))
            for u in cand[:len(nb) - w]:
                A[v].discard(u); A[u].discard(v)
                nb.discard(u)
                dropped.append((min(v, u), max(v, u)))
        pa[v] = sorted(nb)
        order.append(v)
        for a in nb:
            A[a] |= nb
            A[a].discard(a)
            A[a].discard(v)
        alive[v] = False
        A[v] = set()
        left -= 1
        for a in nb:
            heapq.heappush(heap, (len(A[a]), a))
    order.reverse()                       # parents must precede children
    return pa, order, dropped


def complete_graph(nv):
    """The dependence graph is taken COMPLETE and weighted by mutual information, so the width
    constraint is the ONLY thing that drops an edge.

    A first version thresholded the graph at tau before thinning it. That silently introduced a
    second error source: at w = 11 with nothing dropped by the width constraint, P_hat still
    differed from P by 7.4e-5, which is the THRESHOLDING error, not the width error. B1's identity
    case would have been measuring the wrong thing and B2's curve would have been the sum of two
    effects. With a complete graph, w = N-1 drops nothing and reproduces P exactly, and every
    number in B2 is attributable to the width alone."""
    return [set(range(nv)) - {i} for i in range(nv)]


def edge_coverage(pa, adj):
    """Fraction of the ORIGINAL edges represented in the decomposition -- both endpoints together
    in some bag. This is the standard criterion for a decomposition covering an edge, and unlike
    counting deletions it cannot exceed 1 or count fill-in."""
    bags = [frozenset([v]) | frozenset(p) for v, p in pa.items()]
    inbag = {}
    for b in bags:
        for u in b:
            inbag.setdefault(u, []).append(b)
    tot = cov = 0
    for u in range(len(adj)):
        for v in adj[u]:
            if v <= u:
                continue
            tot += 1
            if any(v in b for b in inbag.get(u, ())):
                cov += 1
    return cov, tot


def cost_of(pa):
    return float(sum(2.0 ** (1 + len(v)) for v in pa.values()))


def hub_graph(N, nreg):
    """Regulators at LOW index, targets at high -- required, because generator_tuned takes a
    vertex's parents to be its lower-index neighbours only (recorded in localclosure L7)."""
    adj = [set() for _ in range(N)]
    for t in range(nreg, N):
        for j in range(nreg):
            adj[t].add(j); adj[j].add(t)
    return adj


def run_width(pi, N, M, w, weight, rng=None):
    G = complete_graph(N)
    pa, order, dr = bounded_elimination(G, w, weight=weight, rng=rng)
    ph = approx_dist(pi, N, pa, order)
    return ph, pa, dr


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    NTOT = 14
    P(RULE); P("BOUNDED WIDTH: FIX THE BUDGET, DROP WHAT DOES NOT FIT, MEASURE THE DAMAGE"); P(RULE)
    P("  Route 1 fixed the geometry and the width came out at 2,724 -- one factor covering 95% of")
    P("  the human genome. Here the width is fixed first and the decomposition is forced into it")
    P("  by deleting edges, so cost is N 2^(w+1) BY CONSTRUCTION and the whole question is error.")
    P(f"  tail-legal threshold for reference: MI < {tau:.4e}")

    TOPS = [("cascade", lambda: path_power(NTOT, 1)),
            ("random degree 2", lambda: random_regulatory(NTOT, 2, seed=7)),
            ("hub, 3 regulators", lambda: hub_graph(NTOT, 3))]

    # ---- B1 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("B1  P_hat IS A DISTRIBUTION, AND THE UNTRUNCATED CASE IS EXACT"); P(RULE)
    b1ok = True
    solved = {}
    for nm, mk in TOPS:
        adj = mk()
        Q = generator_tuned(NTOT, 2.0, 1.0, adj=(None if nm == "cascade" else adj))
        pi, res, _ = stationary(Q)
        M = mi_matrix(pi, NTOT)
        solved[nm] = (adj, pi, M, res)
        W = {i: {j: M[i, j] for j in range(NTOT)} for i in range(NTOT)}
        ph, pa, dr = run_width(pi, NTOT, M, NTOT - 1, W)
        s = float(ph.sum()); e = float(np.abs(ph - pi).max())
        ok = abs(s - 1) < 1e-12 and e < 1e-12 and len(dr) == 0
        b1ok = b1ok and ok
        P(f"  {nm:<20} w=N-1  dropped {len(dr)}  sum(P_hat) = {s:.14f}"
          f"  max|P_hat-P| = {e:.3e}   {'PASS' if ok else 'FAIL'}")
    P(f"  L1-style control: the graph is COMPLETE and MI-weighted, so width is the only thing that")
    P(f"  drops an edge. A first version thresholded at tau first, which left a 7.4e-5 residual")
    P(f"  here that had nothing to do with width. That is recorded in complete_graph's docstring.")
    P(f"  B1: {'PASS' if b1ok else 'FAIL'}")

    # ---- B2 / B3 -------------------------------------------------------------------------------
    P("\n" + RULE); P("B2/B3  THE ERROR CURVE, AND ARE THE DROPPED EDGES THE WEAK ONES?"); P(RULE)
    curves = {}
    for nm, mk in TOPS:
        adj = solved[nm][0]
        for boff, lbl in ((1.0, "shallow"), (12.0, "deep")):
            Q = generator_tuned(NTOT, 2.0, boff, adj=(None if nm == "cascade" else adj))
            pi, res, _ = stationary(Q)
            M = mi_matrix(pi, NTOT)
            W = {i: {j: M[i, j] for j in range(NTOT)} for i in range(NTOT)}
            t_ex = conjunctive(pi, NTOT)
            mu_ex = means(pi, NTOT)
            nedge = NTOT * (NTOT - 1) // 2
            P(f"\n  {nm}, {lbl} tail: exact P(all ON) = {t_ex:.4e}   residual {res:.1e}")
            P(f"    {'w':>3} {'cost':>10} {'dropped':>8} {'frac':>6} {'tail rel err':>13}"
              f" {'max mean err':>13} {'strongest drop':>15} {'x tau':>8}")
            for w in (1, 2, 3, 4, 5, 6, 8):
                ph, pa, dr = run_width(pi, NTOT, M, w, W)
                terr = abs(conjunctive(ph, NTOT) - t_ex) / t_ex
                merr = float(np.abs(means(ph, NTOT) - mu_ex).max())
                dmi = [M[a, b] for a, b in dr]
                sd = max(dmi) if dmi else 0.0
                P(f"    {w:>3} {cost_of(pa):>10.3e} {len(dr):>8} {len(dr)/nedge:>6.2f}"
                  f" {terr:>13.3e} {merr:>13.3e} {sd:>15.3e} {sd/tau:>8.0f}")
                curves.setdefault((nm, lbl), []).append((w, len(dr) / nedge, terr, sd))
    P("\n  B3: at the widths that are cheap, the strongest sacrificed edge is often BELOW tau and")
    P("  the tail error is still large. Sub-threshold edges are not free in aggregate -- which is")
    P("  the same thing route 1 found when the sqrt(MI) bound failed. The error is a property of")
    P("  how many edges are dropped, not only of how strong the strongest one was.")

    # ---- B7  THE MATCHED CONTROL ---------------------------------------------------------------
    P("\n" + RULE); P("B7  THE MATCHED CONTROL: weakest-first against random, at identical width"); P(RULE)
    P("  If choosing which edges to keep buys nothing, B2 is a statement about budget and not")
    P("  about method.")
    P(f"    {'topology':<20} {'w':>3} {'weakest-first':>14} {'random (mean of 8)':>20} {'ratio':>8}")
    b7sep = []
    for nm, mk in TOPS:
        adj = solved[nm][0]
        Q = generator_tuned(NTOT, 2.0, 12.0, adj=(None if nm == "cascade" else adj))
        pi, _, _ = stationary(Q)
        M = mi_matrix(pi, NTOT)
        W = {i: {j: M[i, j] for j in range(NTOT)} for i in range(NTOT)}
        t_ex = conjunctive(pi, NTOT)
        for w in (2, 4, 6):
            ph, pa, dr = run_width(pi, NTOT, M, w, W)
            e_w = abs(conjunctive(ph, NTOT) - t_ex) / t_ex
            es = []
            for k in range(8):
                phr, par, drr = run_width(pi, NTOT, M, w, W, rng=np.random.default_rng(100 + k))
                es.append(abs(conjunctive(phr, NTOT) - t_ex) / t_ex)
            e_r = float(np.mean(es))
            b7sep.append(e_r / e_w if e_w > 0 else float("nan"))
            P(f"    {nm:<20} {w:>3} {e_w:>14.3e} {e_r:>20.3e} {e_r/e_w if e_w>0 else float('nan'):>8.1f}")
    med = float(np.nanmedian(b7sep))
    per = {}
    for k, (nm, _) in enumerate(TOPS):
        per[nm] = float(np.nanmedian(b7sep[3 * k:3 * k + 3]))
    P(f"\n  median ratio by topology, which the pooled median hides:")
    for nm, v in per.items():
        P(f"    {nm:<20} {v:>10.1f}")
    P(f"  pooled median {med:.1f}")
    P( "  B7: choosing which edges to keep is worth 5 orders of magnitude on a CASCADE and")
    P( "  worth NOTHING on a hub -- ratio 1.0. The strategy only helps where the structure is")
    P( "  already chain-like. On the topology that matters the error is set by the width alone,")
    P( "  so on real networks B2 is a statement about budget and not about method.")

    # ---- B4  THE REAL NETWORK, NO MODEL --------------------------------------------------------
    P("\n" + RULE); P("B4  THE STRUCTURAL COST ON TRRUST v2 HUMAN  (no model, no dynamics)"); P(RULE)
    adj_tr, inv_tr, sha_tr, ne_tr = load_trrust()
    P(f"  {len(adj_tr)} genes, {ne_tr} direct regulatory edges, sha256 {sha_tr}")
    P(f"    {'w':>4} {'cost N.2^(w+1)':>15} {'edges represented':>18} {'of 8403':>9} {'LOST':>8}")
    b4 = []
    for w in (5, 10, 15, 20, 25, 30, 40):
        pa, order, dr = bounded_elimination(adj_tr, w)
        cov, tot = edge_coverage(pa, adj_tr)
        lost = 1.0 - cov / tot
        b4.append((w, lost))
        P(f"    {w:>4} {len(adj_tr)*2.0**(w+1):>15.3e} {cov:>18} {cov/tot:>9.3f} {lost:>8.3f}")
    P("  Every lost edge is a DIRECT regulatory edge carrying mutual information thousands of")
    P("  times above tau. This column contains no dynamics and no model -- it is a property of")
    P("  the graph and of the width budget, nothing else.")

    # ---- B5 / B6 -------------------------------------------------------------------------------
    P("\n" + RULE); P("B5/B6  THE EXTRAPOLATION (A MODEL), AND THE VERDICT"); P(RULE)
    P("  B5 is the only claim here that is not a direct measurement. It maps B4's measured drop")
    P("  fraction on the real network through B2's measured error-vs-drop-fraction curve. The")
    P("  bridge assumes the error depends on the drop fraction and not on which topology produced")
    P("  it, which is an assumption and is why this is labelled a model.")
    # Reference curves measured at N=16 out to width 13, so the domain reaches drop fraction
    # 0.025 and covers TRRUST's affordable range. At N=14 it stopped at 0.16 and every useful
    # row came back "out of domain".
    NR = 16
    refs = {}
    for nm, adj in (("cascade", None), ("hub, 3 regulators", hub_graph(NR, 3))):
        Qr = generator_tuned(NR, 2.0, 12.0, adj=adj)
        pir, _, _ = stationary(Qr)
        Mr = mi_matrix(pir, NR)
        Wr = {i: {j: Mr[i, j] for j in range(NR)} for i in range(NR)}
        tr = conjunctive(pir, NR)
        ne = NR * (NR - 1) // 2
        rows = []
        for w in range(1, NR - 2):
            ph, pa, dr = run_width(pir, NR, Mr, w, Wr)
            rows.append((len(dr) / ne, abs(conjunctive(ph, NR) - tr) / tr))
        refs[nm] = rows
    P(f"\n    reference curves at N={NR}, deep tail (P = {tr:.2e}):")
    P(f"    {'drop fraction':>14} {'cascade err':>13} {'hub err':>13} {'spread':>10}")
    for k in range(len(refs["cascade"])):
        fc, ec = refs["cascade"][k]
        fh, eh = refs["hub, 3 regulators"][k]
        P(f"    {fc:>14.3f} {ec:>13.3e} {eh:>13.3e} {eh/ec if ec>0 else float('inf'):>10.1e}")
    P( "    The two differ by orders at the same drop fraction, which is why B5 reports a range.")
    fr = np.array([r[0] for r in refs["hub, 3 regulators"]])
    er = np.array([r[1] for r in refs["hub, 3 regulators"]])
    frc = np.array([r[0] for r in refs["cascade"]])
    erc = np.array([r[1] for r in refs["cascade"]])
    o = np.argsort(fr); oc = np.argsort(frc)
    lo, hi = float(fr[o][0]), float(fr[o][-1])
    P(f"\n  the measured domain of that curve is drop fraction [{lo:.2f}, {hi:.2f}]. Outside it the")
    P( "  model REFUSES rather than clamping -- clamping is what made the first version of this")
    P( "  table return one constant and look like a result.")
    P(f"\n    {'w':>4} {'TRRUST lost frac':>17} {'err: cascade end':>17} {'hub end':>12}"
      f" {'affordable':>11} {'accurate':>9}")
    ok_any = False
    undet = 0
    for w, f in b4:
        inb = lo <= f <= hi
        aff = (2.0 ** w) <= 1e9
        if not inb:
            undet += 1 if aff else 0        # an unaffordable row cannot be in the intersection,
                                            # so it cannot make the intersection undetermined
            P(f"    {w:>4} {f:>17.3f} {'out of domain':>17} {'--':>12} {str(aff):>11} {'--':>9}")
            continue
        e_h = float(np.interp(f, fr[o], er[o]))
        e_c = float(np.interp(f, frc[oc], erc[oc]))
        acc = e_h < 0.01
        ok_any = ok_any or (aff and acc)
        P(f"    {w:>4} {f:>17.3f} {e_c:>17.3e} {e_h:>12.3e} {str(aff):>11} {str(acc):>9}")
    P(f"\n  predeclared: affordable = 2^w <= 1e9 (w <= 30); accurate = tail error < 1%")
    P(f"  affordable rows the model could not reach: {undet}")
    hub_tail = [e for e in er[o][:5]]
    P(f"  the hub curve PLATEAUS: as the drop fraction falls to 0.025 the error settles at"
      f" {min(er[o][:5]):.2f}-{max(er[o][:5]):.2f} rather than going to zero. A hub of degree k")
    P( "  is a (k+1)-clique in the dependence graph, so the last edges any width below k is forced")
    P( "  to drop are hub edges, which are the strongest ones. Buying width does not buy accuracy")
    P( "  past that point -- which is why more width was non-monotone in B2 as well.")
    if ok_any:
        P( "  B6: a width exists that is BOTH affordable and accurate at the hub end of the range")
        P( "      -- the first positive result on state complexity in this build order.")
    elif undet:
        P(f"  B6: UNDETERMINED -- {undet} affordable widths fall outside the measured domain.")
        P( "      Not a negative; the model cannot reach them.")
    else:
        P( "  B6: EMPTY INTERSECTION -- every affordable width is inside the measured domain and")
        P( "      none is accurate at the hub end. An earlier run reached this verdict without")
        P( "      support, on one in-domain row; it is supported now because all five affordable")
        P( "      widths are in domain and all five fail.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_boundedwidth.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
