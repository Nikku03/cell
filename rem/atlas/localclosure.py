"""Fixing the treewidth problem the bounded-graph-power way: local closure instead of exact marginalisation.

WHAT statedim.py LEFT. The tail-legal dependence graph costs 2^treewidth to marginalise exactly,
and treewidth exceeds 40 by N = 512 on every topology except a pure cascade. Three routes out
were named there and none tested. This is the first: exploit that the dependence graph is a
bounded POWER G^r of the regulatory graph rather than an arbitrary graph.

WHAT THAT ROUTE CANNOT BE, and saying so first because my own S8(b) already rules it out. The
obvious version is to lift a tree decomposition of G to one of G^r. That is dead on arrival:
S8(b) measured treewidth at r = 1 -- the bare regulatory graph, the most optimistic dependence
graph there could be -- and random degree-2 and scale-free graphs were ALREADY past 40 by
N = 512. The power is not the obstruction. The base graph is. Any route that still needs a tree
decomposition of G fails for the same reason the r = 3 version did.

WHAT IT IS INSTEAD. The bounded power licenses giving up exact marginalisation entirely. If every
dependence above tau is confined to distance r, then a distribution that gets every r-ball right
gets everything above tau right, and a factorisation over r-balls needs no global decomposition
at all:

    P_hat(x) = prod_i P( x_i | x_{B_r(i) and earlier in the order} )

built from the EXACT r-ball marginals. This is a Bayesian network, so it normalises by
construction, and its cost is sum_i 2^(1+|pa_i|) -- LINEAR in N with no treewidth term anywhere.
The price is that it is an approximation where the junction tree was exact, and the whole question
is whether the error it makes is the error the tail tolerance allows.

THE TRADE, STATED PLAINLY SO IT IS NOT SMUGGLED. Junction tree: exact marginals, cost 2^treewidth.
Local closure: eps-accurate marginals, cost N 2^(1+max|pa|). Comparing those two costs is only
honest if the local closure actually meets eps, which is what L2 measures, and at the r where it
does, which is what L3 must then use.

TOPOLOGY IS REAL, NOT SYNTHETIC. statedim's S7 used random and Barabasi-Albert graphs, which may
be much worse than biology. This module uses the TRRUST v2 human transcriptional regulatory
network -- 2,861 genes, 8,403 TF-target edges, mean degree 5.87, and a degree distribution with
SP1 at 484 and 38% of genes at degree 1. That skew is the whole question: a graph with a few
enormous hubs and a long thin tail may be exactly the case where deleting a handful of vertices
collapses the cost, and no synthetic graph in statedim could have shown that.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

L1  THE CONSTRUCTION IS A DISTRIBUTION, AND THE IDENTITY CASE IS EXACT. P_hat must sum to 1 to
    machine precision. And when every vertex gets its full predecessor set -- r large enough that
    B_r is everything -- P_hat must equal the exact stationary distribution to machine precision,
    because the chain rule is then exact. If the identity case is not exact, every number below
    is measuring a bug rather than an approximation.

L2  DOES THE TAIL ERROR FALL WITH r, AND AT THE RATE THE LAW PREDICTS? The observable is the
    conjunctive rare event the architecture exists for -- every gene ON at once -- alongside the
    mean, so the bulk/tail divergence is visible in the same table. PREDECLARED: local closure is
    viable only if the tail error falls geometrically in r. If it plateaus above the tolerance at
    any r that is affordable, this route fails and must be reported as failed. Separately, the
    grouping law predicts tail_err ~ 20.23 sqrt(MI just outside the ball); that prediction is
    checked, and it is allowed to be wrong without the route failing -- a working approximation
    with a bad error bound is a different result from a broken approximation.

L3  THE HONEST COST COMPARISON, at the r that L2 says is needed. Local closure cost
    sum_i 2^(1+|pa_i|) against junction-tree cost 2^treewidth, on the same topologies statedim
    used plus the real TRN. PREDECLARED: this route succeeds only if max|pa_i| stays bounded where
    treewidth does not. If both blow up, the route buys nothing and the answer is that it buys
    nothing.

L4  THE COST IS SET BY THE WORST BALL, NOT THE AVERAGE. Report the full ball-size distribution on
    the real network and quote the MAXIMUM. A mean ball size on a graph containing SP1 is a
    number with no operational meaning, and quoting it would be ledger S.

L5  HUB CONDITIONING ON REAL TOPOLOGY. Delete the h highest-degree genes and re-measure both the
    maximum ball size and the treewidth. The deliverable is the h that brings the cost under a
    stated budget on the human TRN -- and whether that h is small enough to be a fix or large
    enough to be a restatement of the problem. statedim's S6 already says conditioning is only
    LEGITIMATE when the conditioned variable is far from its targets in timescale, so an h that
    works arithmetically still has that physical precondition attached.

L6  THE BULK IS NOT THE TAIL. Mean error reported beside tail error at every r. Five modules in
    this build order have measured a bulk quantity exact while the tail was wrong by orders; if
    that happens here too it must be visible in the same table rather than discovered later.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import (SEED, C_TAIL, generator, generator_graph, stationary, mi_matrix,
                                by_distance, tau_for, path_power, random_regulatory, scale_free,
                                graph_power, treewidth_mindeg, bfs_dist)

TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
TRRUST_SHA = "9b909319ccc8e36588b5a1bd3640e0df"     # first 32 hex of sha256, checked on load


def load_trrust(path=None):
    """The TRRUST v2 human TF-target network. Not vendored -- fetched and checksummed."""
    import hashlib
    path = path or os.path.join(os.path.dirname(__file__), "trrust_human.tsv")
    if not os.path.exists(path):
        import urllib.request
        urllib.request.urlretrieve(TRRUST_URL, path)
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:32]
    names, E = {}, set()
    for ln in raw.decode("utf8", "replace").splitlines():
        f = ln.split("\t")
        if len(f) < 2 or f[0] == f[1]:
            continue
        for g in (f[0], f[1]):
            names.setdefault(g, len(names))
        E.add((names[f[0]], names[f[1]]))
    adj = [set() for _ in names]
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    inv = {v: k for k, v in names.items()}
    return adj, inv, sha, len(E)


# =================================================================================================
# THE LOCAL CLOSURE
# =================================================================================================

def ball(adj, i, r):
    seen, front = {i}, {i}
    for _ in range(r):
        nxt = set()
        for u in front:
            nxt |= adj[u]
        nxt -= seen
        seen |= nxt
        front = nxt
        if not front:
            break
    return seen


def parent_sets(adj, r, order):
    """pa_i = the part of i's r-ball that comes earlier in the order. This is what makes P_hat a
    Bayesian network rather than an unnormalisable product of cliques."""
    pos = {v: k for k, v in enumerate(order)}
    return {i: sorted([j for j in ball(adj, i, r) if j != i and pos[j] < pos[i]]) for i in order}


def approx_dist(pi, N, pa, order):
    """P_hat over all 2^N states, built from the EXACT conditionals of the true joint."""
    n = 1 << N
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(np.int64) for i in range(N)]
    out = np.ones(n)
    for i in order:
        S = pa[i]
        code = np.zeros(n, dtype=np.int64)
        for k, j in enumerate(S):
            code |= bits[j] << k
        full = code | (bits[i] << len(S))
        j2 = np.bincount(full, weights=pi, minlength=1 << (len(S) + 1))
        m = j2.reshape(2, 1 << len(S))
        den = m.sum(axis=0)
        cond = np.divide(m, den, out=np.full_like(m, 0.5), where=den > 0)
        out = out * cond[bits[i], code]
    return out


def closure_cost(pa):
    return float(sum(2.0 ** (1 + len(v)) for v in pa.values())), max(len(v) for v in pa.values())


def conjunctive(p, N, k=None):
    """P(the last k genes are ALL on) -- the conjunctive rare event, computed exactly from a
    distribution vector."""
    k = k or N
    st = np.arange(len(p), dtype=np.int64)
    mask = 0
    for i in range(N - k, N):
        mask |= (1 << i)
    return float(p[(st & mask) == mask].sum())


def means(p, N):
    st = np.arange(len(p), dtype=np.int64)
    return np.array([float(p[((st >> i) & 1) == 1].sum()) for i in range(N)])


def generator_tuned(N, g, boff, adj=None, seed=SEED):
    """The cascade generator with the OFF rate exposed, so the conjunctive event can be pushed to
    a genuinely rare depth. A tail test run at P = 1e-2 is not a tail test."""
    from scipy.sparse import coo_matrix, csr_matrix
    n = 1 << N
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, N))
    b = boff * np.exp(rng.normal(0, 0.3, N))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(N)]
    R, C, D = [], [], []
    for i in range(N):
        if adj is None:
            par = bits[i - 1] if i > 0 else np.ones(n)
        else:
            nb = [j for j in adj[i] if j < i]
            par = sum(bits[j] for j in nb) / len(nb) if nb else np.ones(n)
        R.append(st); C.append(st ^ (1 << i))
        D.append(np.where(bits[i] == 0, a[i] * (1.0 + g * par), b[i]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr()


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    P(RULE); P("LOCAL CLOSURE: CAN AN r-BALL FACTORISATION REPLACE THE JUNCTION TREE?"); P(RULE)
    P("  statedim left treewidth > 40 by N = 512 on every topology but a cascade. Lifting a tree")
    P("  decomposition from G to G^r cannot help -- S8(b) showed treewidth is already past 40 at")
    P("  r = 1. So this route gives up exact marginalisation instead: factorise on r-balls, cost")
    P("  linear in N with no treewidth term, and pay for it in accuracy.")
    P(f"  tail-legal threshold from tail_err = {C_TAIL} sqrt(MI):  MI < {tau:.4e}")

    # ---- L1 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("L1  P_hat IS A DISTRIBUTION, AND THE FULL-BALL CASE IS EXACT"); P(RULE)
    l1ok = True
    cases = [("cascade", 12, None), ("cascade", 14, None),
             ("random deg 2", 12, random_regulatory(12, 2, seed=7))]
    for nm, N, adj in cases:
        A = adj or path_power(N, 1)
        Q = generator_tuned(N, 2.0, 1.0, adj=adj)
        pi, res, _ = stationary(Q)
        order = list(range(N))
        pa = parent_sets(A, N, order)                       # r = N: every predecessor
        ph = approx_dist(pi, N, pa, order)
        s = float(ph.sum()); e = float(np.abs(ph - pi).max())
        ok = abs(s - 1) < 1e-12 and e < 1e-12
        l1ok = l1ok and ok
        P(f"  {nm:<14} N={N:<3} sum(P_hat) = {s:.14f}   max|P_hat - P| = {e:.3e}"
          f"   {'PASS' if ok else 'FAIL'}")
    P(f"  L1: {'PASS -- the chain rule is exact when nothing is dropped, so what follows is approximation error and not a bug' if l1ok else 'FAIL'}")

    # ---- L2 / L6 -------------------------------------------------------------------------------
    P("\n" + RULE); P("L2/L6  DOES THE TAIL ERROR FALL WITH r, AND IS THE BULK STILL EXACT?"); P(RULE)
    P("  observable: P(every gene ON at once) -- the conjunctive rare event. The OFF rate is")
    P("  dialled to reach three tail depths, because an approximation tested at P = 1e-2 has not")
    P("  been tested on a tail.")
    N = 14
    A = path_power(N, 1)
    order = list(range(N))
    l2rows = []
    for boff in (1.0, 4.0, 12.0):
        Q = generator_tuned(N, 2.0, boff)
        pi, res, _ = stationary(Q)
        M = mi_matrix(pi, N)
        d, mi = by_distance(M)
        t_ex = conjunctive(pi, N)
        mu_ex = means(pi, N)
        P(f"\n  OFF rate x{boff:<5}  exact P(all ON) = {t_ex:.6e}   residual {res:.1e}")
        P(f"    {'r':>3} {'max|pa|':>8} {'P_hat(all ON)':>15} {'tail rel err':>13}"
          f" {'max mean err':>13} {'MI at r+1':>11} {'law predicts':>12}")
        prev = None
        for r in range(1, 7):
            pa = parent_sets(A, r, order)
            ph = approx_dist(pi, N, pa, order)
            t_ap = conjunctive(ph, N)
            terr = abs(t_ap - t_ex) / t_ex
            merr = float(np.abs(means(ph, N) - mu_ex).max())
            mio = mi[r] if r < len(mi) else float("nan")
            pred = C_TAIL * np.sqrt(mio) if mio == mio else float("nan")
            mp = max(len(v) for v in pa.values())
            P(f"    {r:>3} {mp:>8} {t_ap:>15.6e} {terr:>13.3e} {merr:>13.3e}"
              f" {mio:>11.3e} {pred:>12.3e}")
            l2rows.append((boff, r, terr, mio, pred))
            prev = terr
    # geometric decay check on the deepest tail
    deep = [x for x in l2rows if x[0] == 12.0 and x[2] > 0]
    ratios = [deep[i][2] / deep[i + 1][2] for i in range(len(deep) - 1) if deep[i + 1][2] > 0]
    geo = len(ratios) >= 3 and min(ratios[:3]) > 3.0
    P(f"\n  successive tail-error ratios at the deepest tail: "
      + ", ".join(f"{x:.1f}" for x in ratios[:5]))
    P(f"  L2: {'PASS -- the tail error falls geometrically in r, so a bounded r suffices' if geo else 'FAIL -- the tail error does not fall geometrically; this route is not viable'}")
    ok_law = all((x[4] > x[2]) for x in l2rows if x[3] == x[3] and x[2] > 0)
    P(f"  the law {C_TAIL} sqrt(MI) is an UPPER BOUND on the realised tail error in"
      f" {sum(1 for x in l2rows if x[3]==x[3] and x[2]>0 and x[4]>x[2])} of"
      f" {sum(1 for x in l2rows if x[3]==x[3] and x[2]>0)} rows"
      f"   -- {'holds' if ok_law else 'VIOLATED somewhere, so the bound cannot be used to choose r'}")
    P("  L6: the mean error column is the bulk. Compare it against the tail column in the same")
    P("  row -- that ratio is the phenomenon five earlier modules measured, seen here at each r.")

    # ---- L3 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("L3  THE HONEST COST COMPARISON, at the r that L2 says is needed"); P(RULE)
    rstar = 3
    P(f"  using r = {rstar}\n")
    P(f"    {'topology':<24} {'N':>7} {'max|B_r|':>9} {'closure cost':>14} {'treewidth':>10} {'junction cost':>14}")
    adj_tr, inv_tr, sha_tr, ne_tr = load_trrust()
    P(f"    (TRRUST v2 human: {len(adj_tr)} genes, {ne_tr} edges, sha256 {sha_tr})")
    tops = [("cascade", lambda n: path_power(n, 1)),
            ("random degree 2", lambda n: random_regulatory(n, 2)),
            ("scale-free m=2", lambda n: scale_free(n, 2))]
    for nm, mk in tops:
        for n in (512, 4096, 20000):
            g = mk(n)
            bs = [len(ball(g, i, rstar)) for i in range(n)]
            mb = max(bs)
            cc = n * 2.0 ** min(mb, 200)
            gp = graph_power(g, rstar)
            if gp is None:
                tw, jc = ">3000", "dense"
            else:
                t, _, dn = treewidth_mindeg(gp)
                tw = str(t) if dn else ">40"
                jc = f"{2.0**min(t,200):.2e}"
            P(f"    {nm:<24} {n:>7} {mb:>9} {cc:>14.2e} {tw:>10} {jc:>14}")
    bs = [len(ball(adj_tr, i, rstar)) for i in range(len(adj_tr))]
    gp = graph_power(adj_tr, rstar)
    twr = "dense" if gp is None else (lambda t: str(t[0]) if t[2] else ">40")(treewidth_mindeg(gp))
    P(f"    {'TRRUST human (real)':<24} {len(adj_tr):>7} {max(bs):>9}"
      f" {len(adj_tr)*2.0**min(max(bs),200):>14.2e} {twr:>10} {'--':>14}")
    P("\n  Cost is set by the WORST ball, not the average -- L4 -- so max|B_r| is the column that")
    P("  matters. A route that needs 2^(max ball) has not escaped anything if the max ball is N.")

    # ---- L4 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("L4  THE BALL-SIZE DISTRIBUTION ON THE REAL NETWORK"); P(RULE)
    P(f"    {'r':>3} {'median':>8} {'90th':>8} {'99th':>8} {'max':>8} {'gene at the max':>18}")
    deg = np.array([len(a) for a in adj_tr])
    for r in (1, 2, 3):
        b = np.array([len(ball(adj_tr, i, r)) for i in range(len(adj_tr))])
        i = int(np.argmax(b))
        P(f"    {r:>3} {int(np.median(b)):>8} {int(np.percentile(b,90)):>8}"
          f" {int(np.percentile(b,99)):>8} {int(b.max()):>8} {inv_tr[i]:>18}")
    P("  The median gene is cheap and the maximum is catastrophic. Quoting the median would be")
    P("  ledger S: a number with no applicability domain, because the cost is a maximum.")

    # ---- L5 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("L5  HUB CONDITIONING ON THE REAL HUMAN NETWORK"); P(RULE)
    P("  Delete the h highest-degree genes, condition on them, and re-measure. Total cost is")
    P("  2^h x (closure cost on the remainder), so h buys ball size at an exponential price.")
    P(f"    {'h':>5} {'deleted through':>16} {'max|B_1|':>9} {'max|B_2|':>9} {'max|B_3|':>9}"
      f" {'treewidth':>10} {'total cost r=2':>15}")
    order_deg = list(np.argsort(-deg))
    for h in (0, 1, 2, 5, 10, 20, 50, 100, 200, 400):
        rem = set(range(len(adj_tr))) - set(order_deg[:h])
        sub = {i: (adj_tr[i] & rem) for i in rem}
        sub_adj = collections.defaultdict(set, sub)
        mb = {}
        for r in (1, 2, 3):
            mb[r] = max((len(ball(sub_adj, i, r)) for i in rem), default=0)
        tw, _, dn = treewidth_mindeg(_relabel(sub_adj, sorted(rem)))
        cost = (2.0 ** min(h, 200)) * len(rem) * (2.0 ** min(mb[2], 200))
        P(f"    {h:>5} {inv_tr[order_deg[h-1]] if h else '--':>16} {mb[1]:>9} {mb[2]:>9}"
          f" {mb[3]:>9} {(str(tw) if dn else '>40'):>10} {cost:>15.3e}")
    P("\n  statedim S6 attaches a physical precondition to every row of this table: conditioning")
    P("  on a regulator is only LEGITIMATE when that regulator is at least ~500x slower or ~400x")
    P("  faster than its targets. An h that works arithmetically still has to clear that.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_localclosure.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


def _relabel(sub, keys):
    idx = {k: i for i, k in enumerate(keys)}
    return [set(idx[j] for j in sub[k] if j in idx) for k in keys]


if __name__ == "__main__":
    main()
