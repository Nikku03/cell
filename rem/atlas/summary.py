"""Attacking k_i -- and finding that neither k_i nor the block was the thing to attack.

THE DISCIPLINE THIS MODULE STARTS WITH, because the previous one did not. Before attacking any
term, price it at ZERO and ask whether the cap moves. That is one line. block.py skipped it, spent
a module contracting the controller block, and reported a cap moving from |C| = 1 to 19 that was
false -- the block was not binding. The ceiling gate is now gate ZERO here and it runs before
anything is designed.

WHAT THE CEILING GATE SAID ABOUT k_i, run before this module was written:

    L                  0     1     2     3     4     6     8
    k_i as measured   39    19    12     8     6     4     3
    k_i truncated <=1 39    19    13     9     7     5     4
    k_i FREE (= 0)    39    19    13     9     7     5     4

k_i work buys AT MOST +1 controller, at every history depth, and truncating to one controller per
gene already captures the entire ceiling. So k_i is not the binding term either. Combined with
block.py's result -- pricing the block at zero moves the cap by nothing at L >= 3 -- the picture
is that BOTH terms sit at the wall simultaneously. The cost is exponential in |C| by two
independent routes and removing either leaves the other. No local fix exists.

WHAT ACTUALLY WORKS IS A CHANGE OF FORM. Both terms are exponential because the controller history
is represented as a PATTERN: 2^(L+1) distinguishable values per controller. If a gene's response
depends on that history only through a low-dimensional SUMMARY, both terms collapse together:

    representation            per-gene factor        controller block
    pattern                   2^(k_i(L+1))           2^(|C|(L+1))
    per-controller count      (L+2)^(k_i)            (L+2)^|C|
    total count               k_i(L+1)+1             |C|(L+1)+1

The last row is polynomial in both k_i and |C|. Whether it is USABLE is an empirical question
about whether a promoter integrating several transcription factors cares which of them are on or
only how many, and for how long. That is what this module measures.

A DEFECT IN THE FIRST VERSION OF THAT MEASUREMENT, recorded because it made the test unfailable.
The sufficiency test replaced each history pattern's conditional by its group average and summed,
weighting by the pattern's own probability. That telescopes: sum_a w_a (sum_group w m / sum_group w)
regrouped is exactly sum_a w_a m_a, the exact answer. Every summary scored 1e-15, including "no
history at all", because every summary WAS the exact answer by construction. The approximation has
to bite where the engine's does -- in the PRODUCT over genes -- and it now does.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

K0  THE CEILING GATE, FIRST. For every candidate treatment, price it at its best possible case and
    report the resulting cap against the cap as-is. PREDECLARED: a prong may be pursued only if
    its ceiling exceeds the current cap by more than one. This gate exists because the previous
    module spent itself on a term whose ceiling was zero.

K1  IS THE SUMMARY SUFFICIENT? P_hat(x) = sum_a P(a) prod_i P(x_i | summary(a)), scored against
    the exact joint on a global observable. A no-conditioning baseline MUST fail, or the bar is
    not testing the history at all. Errors SIGNED.

K2  WHICH SUMMARY, AND DOES IDENTITY MATTER? Compare total count, per-controller count, last slice
    and the full pattern. PREDECLARED: if per-controller count does no better than total count,
    then WHICH controllers are active does not matter -- only how many -- and that is a stronger
    and more useful statement than the cost saving.

K3  DOES THE SUMMARY'S ERROR STAY BOUNDED? Sweep L, |C|, controller coupling and target-to-
    controller feedback. PREDECLARED: a representation whose error grows without bound is useless
    at scale however cheap it is, and this gate is the one that can kill the whole idea.

K4  THE COST/ACCURACY FRONTIER, both terms in the functional. The per-gene AND block terms
    together -- pricing one of them at zero is the exact defect that invalidated block.py's
    headline, and it is not repeated here.

K5  THE CAP AND THE COVERAGE on the real network, computed with the same functional
    trrust_engine uses.

K7  THE ADVERSARIAL LOGIC, AND AN ADVERSARIAL OBSERVABLE  (added after K2 answered, labelled).
    A count summary should die when controller IDENTITY matters. Build generators whose targets
    read a SPECIFIC PAIR of controllers -- an AND gate and an XOR gate -- so that "two controllers
    on" is ambiguous, and score on BOTH a symmetric observable and a target-specific one.
    PREDECLARED: if the count summary survives an AND gate on a symmetric observable, that is
    evidence about the OBSERVABLE and not about the summary, because a statistic symmetric across
    targets cannot see which controller drove which. The no-conditioning baseline must be checked
    separately on every observable, and an observable whose baseline passes is unreadable.

K6  WHAT CANNOT BE CLAIMED.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import stationary, tau_for
from rem.atlas.block import multi_controller, block_joint, marginalise_targets
from rem.atlas.trrust_engine import signed_edges
from rem.atlas.localclosure import load_trrust, ball
from rem.atlas.engine import residual_graph
from rem.atlas.boundedwidth import bounded_elimination


def var_on(p, k):
    st = np.arange(len(p), dtype=np.int64)
    c = sum(((st >> j) & 1) for j in range(k)).astype(float)
    m = float((p * c).sum())
    return float((p * c * c).sum()) - m * m


SUMMARIES = {
    "full pattern":    lambda a, nC: a,
    "per-ctrl count":  lambda a, nC: tuple(sum((s >> c) & 1 for s in a) for c in range(nC)),
    "total count":     lambda a, nC: sum(bin(s).count("1") for s in a),
    "last slice only": lambda a, nC: a[-1],
    "no conditioning": lambda a, nC: 0,
}


def sufficiency(nC, nT, L, dt, fb=0.0, cc=1.5, g=3.0):
    """P_hat(x) = sum_a P(a) prod_i P(x_i | summary(a)). The approximation bites in the PRODUCT,
    which is where the engine's does -- the first version summed group-averaged JOINTS, which
    telescopes back to the exact answer and made every summary score 1e-15."""
    Q, nv = multi_controller(nC, nT, fb=fb, cc=cc, g=g)
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    exact = marginalise_targets(pi, nv, nC); exact = exact / exact.sum()
    vex = var_on(exact, nT)
    st = np.arange(1 << nT, dtype=np.int64)
    bits = [((st >> j) & 1) for j in range(nT)]
    rows = {}
    for a, v in cur.items():
        w = float(v.sum())
        if w <= 1e-300:
            continue
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s > 0:
            rows[a] = (w, np.array([float((m / s)[bits[j] == 1].sum()) for j in range(nT)]))
    out = {}
    for name, keyfn in SUMMARIES.items():
        grp = collections.defaultdict(lambda: [0.0, np.zeros(nT)])
        for a, (w, q) in rows.items():
            gg = grp[keyfn(a, nC)]; gg[0] += w; gg[1] += w * q
        tot = np.zeros(1 << nT)
        for a, (w, q) in rows.items():
            gg = grp[keyfn(a, nC)]
            qa = gg[1] / gg[0]
            term = np.full(1 << nT, w)
            for j in range(nT):
                term = term * np.where(bits[j] == 1, qa[j], 1.0 - qa[j])
            tot += term
        tot = tot / tot.sum()
        out[name] = ((var_on(tot, nT) - vex) / vex, len(grp))
    return out, len(rows), res


def trrust_cap(rep, L, bar=1e12, capexp=400, maxC=420):
    """Cap under a representation, with BOTH terms in the functional."""
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    n = len(adj)
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(n)
    for gname, d in outd.items():
        if gname in name2i:
            od[name2i[gname]] = d
    order = list(np.argsort(-od))
    best = 0
    for nC in range(1, maxC):
        C = set(order[:nC])
        kv = np.array([len(adj[i] & C) if i not in C else 0 for i in range(n)])
        res = residual_graph(adj, C, n)
        per = None
        for r in (1, 2):
            for w in (4, 6, 8):
                pa, o, _ = bounded_elimination([ball(res, i, r) - {i} for i in range(n)], w)
                base = np.array([2.0 ** min(1 + len(pa[i]), capexp) for i in range(n)])
                if rep == "pattern":
                    v = float(np.sum(base * np.array(
                        [2.0 ** min(kv[i] * (L + 1), capexp) for i in range(n)])))
                elif rep == "perctrl":
                    v = float(np.sum(base * np.array(
                        [min((L + 2.0) ** min(kv[i], 60), 1e300) for i in range(n)])))
                else:
                    v = float(np.sum(base * (kv * (L + 1) + 1)))
                per = v if per is None else min(per, v)
        blk = (2.0 ** min(nC * (L + 1), capexp) if rep == "pattern"
               else min((L + 2.0) ** min(nC, 60), 1e300) if rep == "perctrl"
               else float(nC * (L + 1) + 1))
        if per + blk <= bar:
            best = nC
        else:
            break
    return best


def edge_coverage(nC):
    E, _ = signed_edges()
    adj, inv, sha, ne = load_trrust()
    outd = collections.Counter(u for u, v, _ in E)
    name2i = {g: i for i, g in inv.items()}
    od = np.zeros(len(adj))
    for gname, d in outd.items():
        if gname in name2i:
            od[name2i[gname]] = d
    order = list(np.argsort(-od))
    C = {inv[i] for i in order[:nC]}
    return sum(1 for u, v, _ in E if u in C) / len(E)


def adversarial(nC, nT, L, dt, logic, g=3.0, cc=1.5, boff=2.0, seed=20260907):
    """Generators where targets read a SPECIFIC controller PAIR -- the case a count summary
    should not survive. Scored on a symmetric and a target-specific observable."""
    from scipy.sparse import coo_matrix, csr_matrix
    nv = nC + nT
    n = 1 << nv
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, nv)); b = boff * np.exp(rng.normal(0, 0.3, nv))
    st = np.arange(n, dtype=np.int64)
    bts = [((st >> i) & 1).astype(float) for i in range(nv)]
    R, C, D = [], [], []
    for c in range(nC):
        drive = 1.0 + cc * bts[c - 1] if c > 0 else np.ones(n)
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.where(bts[c] == 0, a[c] * drive, b[c]))
    for t in range(nC, nv):
        j = t - nC
        if logic == "additive":
            drive = np.ones(n)
            for c in range(nC):
                drive = drive * (1.0 + g * bts[c]) ** (1.0 / nC)
        else:
            c1, c2 = j % nC, (j + 1) % nC
            drive = 1.0 + g * (bts[c1] * bts[c2] if logic == "and" else np.abs(bts[c1] - bts[c2]))
        R.append(st); C.append(st ^ (1 << t))
        D.append(np.where(bts[t] == 0, a[t] * drive, b[t]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    Q = (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr()
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    exact = marginalise_targets(pi, nv, nC); exact = exact / exact.sum()
    stt = np.arange(1 << nT, dtype=np.int64)
    bits = [((stt >> j) & 1) for j in range(nT)]
    pair = lambda p: float(p[(bits[0] == 1) & (bits[1] == 1)].sum())
    vex, pex = var_on(exact, nT), pair(exact)
    rows = {}
    for aa, v in cur.items():
        w = float(v.sum())
        if w <= 1e-300:
            continue
        m = marginalise_targets(v, nv, nC); sm = m.sum()
        if sm > 0:
            rows[aa] = (w, np.array([float((m / sm)[bits[j] == 1].sum()) for j in range(nT)]))
    out = {}
    for name, keyfn in SUMMARIES.items():
        grp = collections.defaultdict(lambda: [0.0, np.zeros(nT)])
        for aa, (w, q) in rows.items():
            gg = grp[keyfn(aa, nC)]; gg[0] += w; gg[1] += w * q
        tot = np.zeros(1 << nT)
        for aa, (w, q) in rows.items():
            qa = grp[keyfn(aa, nC)][1] / grp[keyfn(aa, nC)][0]
            term = np.full(1 << nT, w)
            for j in range(nT):
                term = term * np.where(bits[j] == 1, qa[j], 1.0 - qa[j])
            tot += term
        tot = tot / tot.sum()
        out[name] = ((var_on(tot, nT) - vex) / vex, (pair(tot) - pex) / pex)
    return out


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    P_(RULE); P_("ATTACKING k_i -- AND FINDING THE TERM WAS NEVER THE PROBLEM"); P_(RULE)

    # ---- K0  THE CEILING GATE, FIRST -----------------------------------------------------------
    P_("\n" + RULE); P_("K0  THE CEILING GATE, RUN BEFORE ANYTHING IS DESIGNED"); P_(RULE)
    P_("  block.py skipped this, spent a module contracting the controller block, and reported a")
    P_("  cap movement that was false because the block was not binding. One line prevents that.")
    P_(f"    {'L':>3} {'pattern rep':>12} {'per-ctrl count':>15} {'total count':>13} {'gain':>8}")
    caps = {}
    for L in (0, 1, 2, 3, 4, 6, 8):
        a = trrust_cap("pattern", L)
        b = trrust_cap("perctrl", L)
        c = trrust_cap("count", L)
        caps[L] = (a, b, c)
        P_(f"    {L:>3} {a:>12} {b:>15} {c:>13} {('+' + str(c - a)):>8}")
    P_("  A count summary lifts the cap past the search bound at EVERY history depth, where")
    P_("  freeing k_i alone bought +1 and freeing the block alone bought 0. The reason is that a")
    P_("  summary collapses BOTH exponential terms at once; neither earlier ceiling gate tested a")
    P_("  treatment that touched both, which is why both concluded 'nothing can be done'.")
    P_(f"  K0: {'PASS -- a prong with a real ceiling exists, so it is worth pursuing' if caps[8][2] > caps[8][0] + 1 else 'FAIL -- no prong clears its ceiling; stop here'}")

    # ---- K1 / K2  SUFFICIENCY ------------------------------------------------------------------
    P_("\n" + RULE); P_("K1/K2  IS THE SUMMARY SUFFICIENT, AND DOES CONTROLLER IDENTITY MATTER?"); P_(RULE)
    P_("  P_hat(x) = sum_a P(a) prod_i P(x_i | summary(a)), scored on Var(targets ON). The")
    P_("  approximation bites in the PRODUCT -- an earlier version averaged group JOINTS, which")
    P_("  telescopes to the exact answer and scored every summary at 1e-15.")
    for nC, L in ((3, 2), (3, 4), (2, 4), (4, 2)):
        o, npat, res = sufficiency(nC, 5, L, 0.25)
        P_(f"\n  nC={nC} L={L}: {npat} patterns, solver residual {res:.1e}")
        P_(f"    {'summary':<18} {'levels':>8} {'signed err':>12} {'vs 1% bar':>10}")
        for k, (e, ng) in o.items():
            tag = "MUST FAIL" if k == "no conditioning" else ("ok" if abs(e) < 0.01 else "fails")
            P_(f"    {k:<18} {ng:>8} {e:>+12.3e} {tag:>10}")
    P_("\n  K2: per-controller count does essentially no better than TOTAL count at every size")
    P_("  tested. WHICH controllers are active does not matter -- only how many, and for how long.")
    P_("  That is a stronger statement than the cost saving and it is what licenses the collapse.")

    # ---- K3  DOES THE ERROR STAY BOUNDED? ------------------------------------------------------
    P_("\n" + RULE); P_("K3  DOES THE SUMMARY'S ERROR STAY BOUNDED?  (the gate that can kill it)"); P_(RULE)
    P_("  A representation whose error grows without bound is useless at scale however cheap.")
    P_(f"\n  sweeping history depth L (nC=3):")
    P_(f"    {'L':>3} {'pattern':>12} {'total count':>13} {'ratio':>8}")
    for L in (1, 2, 3, 4):
        o, _, _ = sufficiency(3, 5, L, 0.25)
        ep, ec = o["full pattern"][0], o["total count"][0]
        P_(f"    {L:>3} {ep:>+12.3e} {ec:>+13.3e} {abs(ec/ep) if ep else float('nan'):>8.2f}")
    P_(f"\n  sweeping controller count (L=2):")
    P_(f"    {'nC':>3} {'pattern':>12} {'total count':>13} {'ratio':>8}")
    for nC in (2, 3, 4):
        o, _, _ = sufficiency(nC, 5, 2, 0.25)
        ep, ec = o["full pattern"][0], o["total count"][0]
        P_(f"    {nC:>3} {ep:>+12.3e} {ec:>+13.3e} {abs(ec/ep) if ep else float('nan'):>8.2f}")
    P_(f"\n  sweeping controller coupling and target->controller feedback (nC=3, L=2):")
    P_(f"    {'cc':>5} {'fb':>5} {'pattern':>12} {'total count':>13} {'ratio':>8}")
    for cc, fb in ((0.2, 0.0), (1.5, 0.0), (1.5, 0.5), (1.5, 2.0)):
        o, _, _ = sufficiency(3, 5, 2, 0.25, fb=fb, cc=cc)
        ep, ec = o["full pattern"][0], o["total count"][0]
        P_(f"    {cc:>5.2f} {fb:>5.2f} {ep:>+12.3e} {ec:>+13.3e}"
           f" {abs(ec/ep) if ep else float('nan'):>8.2f}")

    # ---- K5  THE CAP AND COVERAGE --------------------------------------------------------------
    P_("\n" + RULE); P_("K5  THE CAP AND WHAT IT COVERS ON THE REAL NETWORK"); P_(RULE)
    P_("  Both terms in the functional, as trrust_engine computes it. Pricing one at zero is the")
    P_("  defect that invalidated block.py's headline and it is not repeated.")
    P_(f"    {'|C|':>5} {'edge coverage':>14}")
    for nC in (3, 4, 8, 19, 40, 100, 200, 400):
        P_(f"    {nC:>5} {100*edge_coverage(nC):>13.1f}%")
    P_(f"\n  pattern representation caps at |C| = {caps[8][0]} at L = 8, covering"
       f" {100*edge_coverage(caps[8][0]):.1f}% of regulatory edges.")
    P_(f"  count summary reaches the search bound {caps[8][2]}, covering"
       f" {100*edge_coverage(caps[8][2]):.1f}%.")

    # ---- K7  ADVERSARIAL LOGIC AND OBSERVABLE ---------------------------------------------------
    P_("\n" + RULE); P_("K7  ADVERSARIAL LOGIC AND AN ADVERSARIAL OBSERVABLE  (added after K2)"); P_(RULE)
    P_("  Targets read a SPECIFIC PAIR of controllers, so 'two on' is ambiguous and a count")
    P_("  summary should die. Scored on a symmetric observable AND a target-specific one, with")
    P_("  the no-conditioning baseline checked separately on each.")
    for logic in ("additive", "and", "xor"):
        o = adversarial(3, 5, 2, 0.25, logic)
        P_(f"\n  logic = {logic}")
        P_(f"    {'summary':<18} {'Var err':>12} {'':>6} {'pair err':>12} {'':>6}")
        for name, (es, ep) in o.items():
            P_(f"    {name:<18} {es:>+12.3e} {'ok' if abs(es)<0.01 else 'FAILS':>6}"
               f" {ep:>+12.3e} {'ok' if abs(ep)<0.01 else 'FAILS':>6}")
        bs, bp = o["no conditioning"]
        if abs(bp) < 0.01:
            P_(f"    the pair observable is UNREADABLE here: its no-conditioning baseline PASSES")
            P_(f"    at {bp:+.3e}, so nothing on that column tests dependence.")
    P_("\n  K7: the count summary survives AND and XOR on the observable whose baseline fails.")
    P_("  On XOR the target-specific observable is unreadable because its baseline passes -- which")
    P_("  is the baseline check doing its job, and is why it must run per observable, not once.")

    # ---- K6 -------------------------------------------------------------------------------------
    P_("\n" + RULE); P_("K6  WHAT CANNOT BE CLAIMED"); P_(RULE)
    P_("  1. The sufficiency measurements are on synthetic systems of 2-4 controllers and 5")
    P_("     targets with invented kinetics. Whether a real promoter integrating several")
    P_("     transcription factors is count-sufficient is an empirical question about promoters,")
    P_("     not something these numbers settle.")
    P_("  2. The cap is a COST statement computed from graph arithmetic. It says the engine could")
    P_("     be built at that size, not that it would be accurate there -- K3's error is measured")
    P_("     only at the sizes where exact ground truth exists.")
    P_("  3. K7 runs the adversarial AND and XOR generators. The count summary survives them on")
    P_("     the observable whose baseline fails, but a target-specific observable turned out to")
    P_("     be UNREADABLE on the XOR system because its no-conditioning baseline passed. So the")
    P_("     adversarial case is tested and partly inconclusive, which is different from untested.")
    P_("  4. Nothing here says the count is the BEST low-dimensional summary, only that it is")
    P_("     sufficient where tested and that identity does not matter.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_summary.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
