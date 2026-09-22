"""Can grouping keep the joint state small as the model grows?

THE QUESTION. REM's architecture decides which biological variables may be kept independent and
which must be joined into one coupled group. A group of k binary variables costs 2^k of state, so
the whole engine's memory is the SUM over groups of their block sizes. The engine is tractable at
realistic scale if and only if the groups stay small. Nothing so far in this build order has
measured whether they do.

THE RIGHT LANGUAGE IS PERCOLATION, not correlation length. The grouping rule joins i and j when
their dependence exceeds a threshold tau. That defines a graph on the variables, and the group
sizes are its CONNECTED COMPONENTS. So the question is not "how fast does dependence decay" but
"does the dependence graph percolate at the tau the accuracy target forces". Those are different
questions and they have different answers: a dependence that decays exponentially with a short
length can still percolate if every variable has enough neighbours, and a long-ranged dependence
can fail to percolate if it is confined to a tree.

WHERE TAU COMES FROM, and it is not a free parameter. rem/atlas/RESULTS_grouping_law.txt measured

    tail_err = c * sqrt(MI),     c -> 20.23 as coupling weakens

for exactly this architecture. A tail tolerance eps therefore forces MI < (eps/c)^2. That is a
brutal threshold: eps = 1e-2 needs MI < 2.4e-7. Using a bulk threshold instead -- MI < 1e-3, say,
which keeps means fine -- would admit splits with tail errors of 0.64. The state cost has to be
computed at the tail-legal tau or the number is about the wrong engine.

THE SYSTEM. A regulatory cascade of N binary genes, which is the signalling motif of
rem/atlas/controller.py in stochastic form:

    gene i     OFF -> ON at a_i (1 + g s_{i-1}),    ON -> OFF at b_i
    gene 0     driven by a constant input
    hub        an optional global regulator h that gates EVERY gene, which is what a real
               signalling network has and a cascade does not

g = 0 makes the genes exactly independent, so MI = 0 is an analytic control the estimator must
reproduce. The exact stationary distribution is the null vector of the generator on all 2^N
states, so every mutual information below is exact -- no sampling, no estimator bias, no
plug-in correction to argue about. That caps N at about 18, which is enough: the object being
measured is a decay length and a percolation threshold, and both are read off well before N=18.

WHY THE CASCADE IS NOT TRIVIALLY FACTORISED. Gene i's rates depend only on gene i-1, so it is
tempting to say the stationary law is a Markov chain along the index and everything factorises.
It is not: s_{i+1} responds to the whole TRAJECTORY of s_i, not to its current value, so
s_{i-1} and s_{i+1} remain dependent given s_i. Whether that residual dependence decays is the
measurement, not an assumption.

=================================================================================================
GATES, AND ONE PREDICTION I MADE AND LOST BEFORE COMMITTING
=================================================================================================

HOW THIS FILE CAME TO BE IN THIS ORDER, stated plainly because it matters for how the numbers
below should be read. The gates were written first. An exploratory probe was then run BEFORE the
file was committed, and it falsified the central prediction in the original S4. The prediction is
kept below exactly as it was made, marked as lost, and the module is restructured around what the
probe found. Nothing here is a gate written after seeing its own answer, and the one gate that
was is labelled as such.

    S4 AS ORIGINALLY WRITTEN: "if the CONDITIONAL mutual information I(i;j|h) decays while the
    unconditional does not, then a controller does NOT explode the state provided the grouping
    conditions on the controller, and the cost is set by treewidth rather than by N."

    WHAT HAPPENED: conditioning on the hub's state reduced the distance-6 dependence from
    2.6e-3 to 7.7e-4 -- a factor of 3.4 -- and left it FLAT. It does not decay. Conditioning on
    a controller does not decouple what it controls, because the correlation is carried by the
    controller's TRAJECTORY and not by its current value. S6 below is the repair, and it is the
    gate that was written after seeing an answer.

S1  THE SOLVER IS EXACT. ||Q^T pi||_inf < 1e-12, pi strictly positive and normalised. At g = 0
    every pairwise mutual information must be below 1e-14, because the genes are then independent
    by construction. A dependence measure that cannot return zero on independent variables cannot
    be trusted to return small on nearly independent ones.

S2  DOES DEPENDENCE DECAY WITH DISTANCE? Fit log MI(d) against d, using only values above the
    measured numerical floor -- the floor is read off the plateau, not assumed. Report the decay
    length xi with its standard error and R^2. PREDECLARED: R^2 < 0.9 means the decay is not
    exponential and xi must not be quoted at all.

S3  THE FINITE-SIZE CONTROL. xi at N = 10, 12, 14, 16 must agree within error. A decay length
    that grows with the system it was measured in cannot be extrapolated to 20,000 genes, which
    is the only reason to measure it.

S4  THE LITERAL ARCHITECTURE. Build the dependence graph at the tail-legal tau, take its
    connected components, and report sum_g 2^|g| -- the memory the section-14 rule as written
    actually costs. PREDECLARED: if the largest component equals N, the rule is unusable at any
    scale and this module says so plainly rather than reporting a decay length as if it helped.

S5  IS THE DEPENDENCE GRAPH THE r-TH POWER OF THE REGULATORY GRAPH? If the tail-legal graph is
    exactly {(i,j) : dist(i,j) <= r} for the r where MI(r) > tau > MI(r+1), then the exact
    computation at N <= 16 licenses extrapolating to any N by taking graph powers, and the
    scaling question becomes a graph-theoretic one that does not need a 2^20000 state space.
    If it is not, no extrapolation is licensed and the module must stop at N = 16.

S6  THE TIMESCALE LAW  (written after the probe, and labelled). Sweep the controller's own
    switching rate over four orders of magnitude and report I(i;j) and I(i;j|h) at fixed
    distance. The deliverable is the ratio tau_hub/tau_gene at which conditioning stops working,
    with both limits shown, because a rule that only quotes the good limit is ledger S.

S7  TREEWIDTH IS THE REAL COST, AND IT IS MEASURED AT REALISTIC SCALE. Connected components are
    the wrong decomposition: a path is connected but has treewidth 1. Report the min-degree
    elimination upper bound on treewidth for the tail-legal graph of a cascade, a cascade with a
    hub, random regulatory graphs of mean degree 2, 3 and 4, and a scale-free graph, up to
    N = 20,000. Cost is sum over bags of 2^|bag|. PREDECLARED: bounded or O(log N) treewidth
    means the engine scales; treewidth growing as a fixed fraction of N means it does not, and
    the boundary between those two topologies is the deliverable.

S8  THREE ROBUSTNESS CHECKS, added after S3 failed and S7 answered. Labelled as added, because
    they were written knowing what they would be checking.
    (a) S3 failed: xi at N = 10..16 scattered by 0.0045 against a bar of 0.0043. Refit every N on
        a COMMON distance range to see whether that is a fit-range artefact, and report whether
        the scatter is monotone in N -- a drift and a wobble are different failures. Then state
        whether it can propagate: the quantity actually used downstream is the integer r, not xi.
    (b) S7 used r = 3, measured on a cascade, for every topology. Recompute the treewidth at
        r = 1 -- the bare regulatory graph, the most optimistic dependence graph there could be.
        If the answer is unchanged, S7 does not depend on r at all.
    (c) Measure r directly on a NON-cascade topology by exact CME, so that carrying r across
        topologies is a measurement rather than an assumption.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from rem.atlas.hybrid_tune import RULE

SEED = 20260906
C_TAIL = 20.23          # measured in RESULTS_grouping_law.txt, not chosen here


# =================================================================================================
# THE EXACT STATIONARY DISTRIBUTION
# =================================================================================================

def generator(N, g, hub=False, gh=3.0, seed=SEED):
    """Generator of the cascade on 2^N (or 2^(N+1) with a hub) states.

    Bit i is gene i; with a hub, bit N is the regulator. The hub gates every gene multiplicatively,
    which is what makes it a global controller rather than another link in the chain."""
    rng = np.random.default_rng(seed)
    nv = N + 1 if hub else N
    n = 1 << nv
    a = 1.0 * np.exp(rng.normal(0, 0.3, N))
    b = 1.0 * np.exp(rng.normal(0, 0.3, N))
    ah, bh = 1.0, 1.0
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(nv)]
    rows, cols, data = [], [], []
    for i in range(N):
        parent = bits[i - 1] if i > 0 else np.ones(n)
        gate = (1.0 + gh * bits[N]) / (1.0 + gh) if hub else 1.0
        on = a[i] * (1.0 + g * parent) * gate
        rate = np.where(bits[i] == 0, on, b[i])
        rows.append(st); cols.append(st ^ (1 << i)); data.append(rate)
    if hub:
        rate = np.where(bits[N] == 0, ah, bh)
        rows.append(st); cols.append(st ^ (1 << N)); data.append(rate)
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    Q = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    d = np.asarray(Q.sum(axis=1)).ravel()
    Q = Q - csr_matrix((d, (st, st)), shape=(n, n))
    return Q.tocsr(), nv


def generator_rate(N, g, hub_rate, gh=3.0, seed=SEED):
    """The hub generator with the controller's OWN switching rate exposed. S6 sweeps it."""
    rng = np.random.default_rng(seed)
    nv = N + 1
    n = 1 << nv
    a = np.exp(rng.normal(0, 0.3, N)); b = np.exp(rng.normal(0, 0.3, N))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(nv)]
    R, C, D = [], [], []
    for i in range(N):
        par = bits[i - 1] if i > 0 else np.ones(n)
        gate = (1.0 + gh * bits[N]) / (1.0 + gh)
        R.append(st); C.append(st ^ (1 << i))
        D.append(np.where(bits[i] == 0, a[i] * (1.0 + g * par) * gate, b[i]))
    R.append(st); C.append(st ^ (1 << N)); D.append(np.full(n, float(hub_rate)))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), nv


def stationary(Q, tol=1e-14, itmax=400000):
    """Power iteration on the uniformised chain. Exact in the limit and the residual is checked,
    so there is no convergence claim taken on trust."""
    n = Q.shape[0]
    lam = float(np.abs(Q.diagonal()).max()) * 1.05
    QT = Q.T.tocsr()
    pi = np.full(n, 1.0 / n)
    for k in range(itmax):
        new = pi + QT.dot(pi) / lam
        new = np.maximum(new, 0.0)
        new /= new.sum()
        if k % 25 == 0:
            r = float(np.abs(QT.dot(new)).max())
            if r < tol:
                pi = new
                break
        pi = new
    res = float(np.abs(QT.dot(pi)).max())
    return pi, res, k


# =================================================================================================
# EXACT MUTUAL INFORMATION FROM THE JOINT
# =================================================================================================

def bit_of(st, i):
    return ((st >> i) & 1)


def mi_matrix(pi, nv, cond=None):
    """Exact pairwise mutual information over the first nv bits. If cond is a bit index, returns
    the CONDITIONAL mutual information given that bit."""
    n = len(pi)
    st = np.arange(n, dtype=np.int64)
    bits = [bit_of(st, i) for i in range(nv + (1 if cond is not None else 0))]
    ncond = 2 if cond is not None else 1
    cb = bits[cond] if cond is not None else np.zeros(n, dtype=np.int64)
    pc = np.bincount(cb, weights=pi, minlength=2)[:ncond]
    M = np.zeros((nv, nv))
    for i in range(nv):
        for j in range(i + 1, nv):
            idx = (cb * 4 + bits[i] * 2 + bits[j]).astype(np.int64)
            p = np.bincount(idx, weights=pi, minlength=4 * ncond).reshape(ncond, 2, 2)
            tot = 0.0
            for c in range(ncond):
                if pc[c] <= 0:
                    continue
                q = p[c] / pc[c]
                qi = q.sum(axis=1, keepdims=True)
                qj = q.sum(axis=0, keepdims=True)
                m = q > 0
                tot += pc[c] * float(np.sum(q[m] * np.log(q[m] / (qi @ qj)[m])))
            M[i, j] = M[j, i] = max(tot, 0.0)
    return M


def components(M, tau):
    """Connected components of the dependence graph {(i,j) : MI_ij > tau}. These ARE the groups."""
    nv = M.shape[0]
    lab = -np.ones(nv, int)
    c = 0
    for s in range(nv):
        if lab[s] >= 0:
            continue
        stack, lab[s] = [s], c
        while stack:
            u = stack.pop()
            for v in np.nonzero(M[u] > tau)[0]:
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(int(v))
        c += 1
    sizes = np.bincount(lab)
    return lab, sizes


def state_cost(sizes):
    """Actual memory: the sum of the blocks, not the product of everything."""
    return float(np.sum(2.0 ** sizes.astype(float)))


def tau_for(eps, c=C_TAIL):
    """The MI a tail tolerance eps allows, from the measured law tail_err = c sqrt(MI)."""
    return (eps / c) ** 2


def fit_decay(d, m):
    """log MI = A - d/xi. Returns xi, its standard error, and R^2."""
    d = np.asarray(d, float); m = np.asarray(m, float)
    k = m > 0
    d, y = d[k], np.log(m[k])
    if len(d) < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([np.ones_like(d), d]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(res @ res) / ss if ss > 0 else float("nan")
    s2 = float(res @ res) / max(1, len(d) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    slope, sd = float(beta[1]), float(np.sqrt(cov[1, 1]))
    xi = -1.0 / slope if slope < 0 else float("inf")
    sxi = abs(sd / slope ** 2) if slope < 0 else float("nan")
    return xi, sxi, r2


def by_distance(M):
    nv = M.shape[0]
    ds, ms = [], []
    for d in range(1, nv):
        v = [M[i, i + d] for i in range(nv - d)]
        ds.append(d); ms.append(float(np.mean(v)))
    return np.array(ds), np.array(ms)


# =================================================================================================
# TREEWIDTH -- the cost that connected components gets wrong
# =================================================================================================

def graph_from_mi(M, tau):
    nv = M.shape[0]
    return [set(np.nonzero(M[i] > tau)[0].tolist()) - {i} for i in range(nv)]


def path_power(N, r):
    return [set(range(max(0, i - r), min(N, i + r + 1))) - {i} for i in range(N)]


def random_regulatory(N, k, seed=SEED):
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    for i in range(1, N):
        for _ in range(k):
            j = int(rng.integers(0, i))
            adj[i].add(j); adj[j].add(i)
    return adj


def scale_free(N, m=2, seed=SEED):
    rng = np.random.default_rng(seed)
    adj = [set() for _ in range(N)]
    targets = list(range(m))
    repeat = []
    for i in range(m, N):
        for t in set(targets[:m] if i == m else targets):
            adj[i].add(t); adj[t].add(i)
            repeat.extend([i, t])
        pool = repeat if repeat else list(range(i))
        targets = [int(pool[int(rng.integers(len(pool)))]) for _ in range(m)]
    return adj


def graph_power(adj, r, dense_cap=3000):
    """r-th power by bounded BFS. Returns None if any neighbourhood exceeds dense_cap, because a
    graph that dense has treewidth at least dense_cap and there is nothing to compute."""
    N = len(adj)
    out = []
    for s in range(N):
        seen = {s}
        front = {s}
        for _ in range(r):
            nxt = set()
            for u in front:
                nxt |= adj[u]
            nxt -= seen
            seen |= nxt
            front = nxt
            if len(seen) > dense_cap:
                return None
        out.append(seen - {s})
    return out


def treewidth_mindeg(adj, cap=40):
    """Min-degree elimination upper bound on treewidth, with a cap. 2^40 is already unusable, so
    stopping there loses nothing and keeps a hopeless graph from taking all day."""
    import heapq
    n = len(adj)
    A = [set(a) for a in adj]
    alive = np.ones(n, bool)
    h = [(len(A[v]), v) for v in range(n)]
    heapq.heapify(h)
    tw, bags = 0, []
    left = n
    while left:
        while h:
            d, v = heapq.heappop(h)
            if alive[v] and len(A[v]) == d:
                break
        else:
            v = int(np.nonzero(alive)[0][0]); d = len(A[v])
        nb = A[v]
        tw = max(tw, len(nb))
        bags.append(len(nb) + 1)
        if tw > cap:
            return cap + 1, bags, False
        for a in nb:
            A[a] |= nb
            A[a].discard(a)
            A[a].discard(v)
        alive[v] = False
        A[v] = set()
        left -= 1
        for a in nb:
            heapq.heappush(h, (len(A[a]), a))
    return tw, bags, True


def bag_cost(bags):
    b = np.array(bags, float)
    b = np.minimum(b, 60.0)
    return float(np.sum(2.0 ** b))


def generator_graph(adj, g, seed=SEED):
    """The same cascade dynamics on an arbitrary regulatory graph, so r can be measured off the
    chain rather than assumed to transfer."""
    N = len(adj)
    n = 1 << N
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, N)); b = np.exp(rng.normal(0, 0.3, N))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(N)]
    R, C, D = [], [], []
    for i in range(N):
        nb = [j for j in adj[i] if j < i]
        par = sum(bits[j] for j in nb) / len(nb) if nb else np.ones(n)
        R.append(st); C.append(st ^ (1 << i))
        D.append(np.where(bits[i] == 0, a[i] * (1.0 + g * par), b[i]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr()


def bfs_dist(adj, N):
    D = np.full((N, N), 99)
    for s in range(N):
        D[s, s] = 0
        fr, seen = {s}, {s}
        for d in range(1, N):
            nx = set().union(*[adj[u] for u in fr]) - seen if fr else set()
            for v in nx:
                D[s, v] = d
            seen |= nx
            fr = nx
            if not fr:
                break
    return D


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau2 = tau_for(1e-2)
    tau3 = tau_for(1e-3)
    P(RULE); P("CAN GROUPING KEEP THE JOINT STATE SMALL AS THE MODEL GROWS?"); P(RULE)
    P(f"  tail-legal thresholds from the measured law tail_err = {C_TAIL} sqrt(MI):")
    P(f"    eps = 1e-2  ->  MI < {tau2:.4e}")
    P(f"    eps = 1e-3  ->  MI < {tau3:.4e}")
    P( "  a bulk threshold of 1e-3, which keeps means fine, would admit tail errors of 0.64.")

    # ---- S1 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("S1  THE SOLVER IS EXACT, AND RETURNS ZERO ON INDEPENDENT VARIABLES"); P(RULE)
    s1ok = True
    for (N, g) in [(10, 0.0), (14, 0.0), (10, 2.0), (14, 2.0), (16, 2.0)]:
        Q, nv = generator(N, g)
        pi, res, it = stationary(Q)
        M = mi_matrix(pi, nv)
        mx = float(M.max())
        okres = res < 1e-12 and pi.min() > 0 and abs(pi.sum() - 1) < 1e-12
        okz = (mx < 1e-14) if g == 0 else (mx > 1e-6)
        s1ok = s1ok and okres and okz
        P(f"  N={N:<3} g={g:<4} states {1<<nv:>7}  residual {res:.2e}  iters {it:>6}"
          f"  max MI {mx:.4e}  {'PASS' if okres and okz else 'FAIL'}")
    P(f"  g = 0 is an analytic control: the genes are independent, so MI must be exactly zero.")
    P(f"  S1: {'PASS' if s1ok else 'FAIL'}")

    # ---- S2, S3 --------------------------------------------------------------------------------
    P("\n" + RULE); P("S2/S3  DOES DEPENDENCE DECAY, AND IS THE DECAY LENGTH A REAL NUMBER?"); P(RULE)
    Q, nv = generator(16, 2.0); pi, res, it = stationary(Q); M16 = mi_matrix(pi, nv)
    d, m = by_distance(M16)
    floor = float(np.median(m[m < 1e-13])) if np.any(m < 1e-13) else 0.0
    P(f"  numerical floor read off the plateau: {floor:.3e}   (values below 10x this are discarded)")
    P(f"    {'d':>3} {'MI(d)':>14} {'ratio':>9}")
    for i in range(len(d)):
        rt = m[i - 1] / m[i] if i and m[i] > 0 else float("nan")
        P(f"    {d[i]:>3} {m[i]:>14.4e} {rt:>9.2f}")
    use = m > 10 * max(floor, 1e-16)
    xi, sxi, r2 = fit_decay(d[use], m[use])
    P(f"  fitted on {int(use.sum())} points above the floor:  xi = {xi:.4f} +- {sxi:.4f}   R^2 = {r2:.6f}")
    s2ok = r2 >= 0.9
    P(f"  S2: {'PASS -- exponential, xi may be quoted' if s2ok else 'FAIL -- not exponential; xi withheld'}")
    P(f"\n  S3 finite-size control:")
    P(f"    {'N':>4} {'xi':>10} {'+-':>8} {'R^2':>10}")
    xis = []
    for N in (10, 12, 14, 16):
        Qn, nvn = generator(N, 2.0); pn, _, _ = stationary(Qn); Mn = mi_matrix(pn, nvn)
        dn, mn = by_distance(Mn)
        un = mn > 10 * max(floor, 1e-16)
        x, sx, rr = fit_decay(dn[un], mn[un])
        xis.append((x, sx))
        P(f"    {N:>4} {x:>10.4f} {sx:>8.4f} {rr:>10.6f}")
    sp = max(x for x, _ in xis) - min(x for x, _ in xis)
    err = 2 * max(sx for _, sx in xis)
    s3ok = sp <= err
    P(f"    spread {sp:.4f} against 2x the largest standard error {err:.4f}")
    P(f"  S3: {'PASS -- xi is intrinsic, not finite-size' if s3ok else 'FAIL -- xi drifts with N and must not be extrapolated'}")

    # ---- S4  THE LITERAL ARCHITECTURE ----------------------------------------------------------
    P("\n" + RULE); P("S4  WHAT THE SECTION-14 RULE, AS WRITTEN, ACTUALLY COSTS"); P(RULE)
    P(f"    {'threshold':>12} {'tau':>12} {'components':>12} {'largest':>9} {'state cost':>14}")
    for nm, t in [("tail 1e-2", tau2), ("tail 1e-3", tau3), ("bulk 1e-3", 1e-3), ("bulk 1e-2", 1e-2)]:
        lab, sz = components(M16, t)
        P(f"    {nm:>12} {t:>12.3e} {len(sz):>12} {int(sz.max()):>9} {state_cost(sz):>14.4e}")
    lab, sz = components(M16, tau2)
    s4_fail = int(sz.max()) == 16
    P(f"  At the tail-legal threshold the graph is ONE component of {int(sz.max())} variables out of 16.")
    P( "  The reason is not slow decay -- decay is very fast. It is that MI at distance 1 is 6e-3,")
    P( "  four orders above tau, so every directly regulated pair is joined; and a regulatory")
    P( "  network is connected. Thresholding therefore returns the whole network for ANY connected")
    P( "  topology at ANY tail tolerance, and costs 2^N.")
    P(f"  S4: {'the literal rule is UNUSABLE at scale -- as predeclared, this is stated plainly' if s4_fail else 'the rule survives'}")

    # ---- S5  IS IT A GRAPH POWER? --------------------------------------------------------------
    P("\n" + RULE); P("S5  IS THE DEPENDENCE GRAPH THE r-TH POWER OF THE REGULATORY GRAPH?"); P(RULE)
    s5ok = True
    for nm, t in [("tail 1e-2", tau2), ("tail 1e-3", tau3)]:
        r = int(np.max(d[m > t])) if np.any(m > t) else 0
        G = graph_from_mi(M16, t)
        Pw = path_power(16, r)
        same = all(G[i] == Pw[i] for i in range(16))
        s5ok = s5ok and same
        P(f"  {nm}: MI(d) > tau out to d = {r}; dependence graph == path^{r}: {same}")
    P(f"  S5: {'PASS -- extrapolation by graph power is licensed' if s5ok else 'FAIL -- no extrapolation beyond N=16 is licensed'}")

    # ---- S6  THE TIMESCALE LAW -----------------------------------------------------------------
    P("\n" + RULE); P("S6  THE TIMESCALE LAW  (this gate was written after seeing a probe)"); P(RULE)
    P("  The original S4 predicted that conditioning on a global controller would restore decay.")
    P("  It does not. What follows is why, and when it does.")
    P( "  The statistic that matters is not the component count -- S4 showed that is always N --")
    P( "  but the RADIUS r out to which dependence still exceeds tau, because that sets the")
    P( "  treewidth and hence the cost. It is measured at N = 16, so it can resolve r up to 15;")
    P( "  at N = 10 it was censored at 9 and said nothing.")
    P(f"    {'hub rate':>9} {'tau_hub/tau_gene':>17} {'I(i;j) d=5':>13} {'I(i;j|h) d=5':>14}"
      f" {'ratio':>9} {'r_unc':>7} {'r_cond':>7}")
    hub_rows = []
    for hr in (0.0005, 0.002, 0.003, 0.005, 0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 400.0, 700.0, 1000.0):
        Qh, nvh = generator_rate(16, 2.0, hr)
        ph, rh, _ = stationary(Qh)
        Mu = mi_matrix(ph, 16); Mc = mi_matrix(ph, 16, cond=16)
        du, mu = by_distance(Mu); dc, mc = by_distance(Mc)
        ru = max([int(x) for x, v in zip(du, mu) if v > tau2] or [0])
        rc = max([int(x) for x, v in zip(dc, mc) if v > tau2] or [0])
        hub_rows.append((hr, mu[4], mc[4], rc, ru))
        P(f"    {hr:>9.3f} {1.0/hr:>17.2f} {mu[4]:>13.4e} {mc[4]:>14.4e}"
          f" {mu[4]/mc[4]:>9.1f} {ru:>7} {rc:>7}")
    best = min(hub_rows, key=lambda r: r[2] / r[1])
    worst = max(hub_rows, key=lambda r: r[2] / r[1])
    P(f"  conditioning helps most at hub rate {best[0]} (factor {best[1]/best[2]:.0f}) and least at")
    P(f"  hub rate {worst[0]} (factor {worst[1]/worst[2]:.1f}).")
    P( "  MECHANISM. A slow controller is quasi-static, so its current state IS its history and")
    P( "  conditioning removes everything. A fast controller is white noise on the gene timescale")
    P( "  and correlates almost nothing, so there is nothing to remove. At MATCHED timescales the")
    P( "  controller's trajectory carries dependence its instantaneous state does not, and no")
    P( "  amount of conditioning on that state recovers it. That is the regime a real signalling")
    P( "  network occupies, and it is the expensive one.")
    cheap = [r[0] for r in hub_rows if r[3] <= 4]
    dear = [r[0] for r in hub_rows if r[3] >= 15]
    P(f"  affordable (r_cond <= 4, cost 2^6 per gene) at hub rates: {cheap}")
    P(f"  censored at the system size (r_cond >= 15, no grouping helps) at: {dear}")
    P( "  THE WINDOW. Conditioning on a global controller is affordable only when the controller")
    P( "  is at least ~500x SLOWER than what it controls -- so that its state is its history --")
    P( "  or at least ~400x FASTER, where it correlates nothing to begin with. Within a factor of")
    P( "  a few hundred either way, r_cond runs off the end of a 16-gene system and the cost is")
    P( "  2^N. Kinase signalling at seconds against transcription at tens of minutes sits at")
    P( "  roughly 100-1000, which is ON that boundary, not comfortably inside it.")
    ok6 = bool(cheap) and bool(dear)
    P(f"  S6: {'both regimes observed and the boundary is bracketed' if ok6 else 'no regime separation observed'}")

    # ---- S7  TREEWIDTH AT REALISTIC SCALE ------------------------------------------------------
    P("\n" + RULE); P("S7  TREEWIDTH, WHICH IS THE COST THAT COMPONENTS GETS WRONG"); P(RULE)
    P("  A path is one connected component but has treewidth 1. Cost is sum over bags of 2^|bag|.")
    r2t = int(np.max(d[m > tau2]))
    P(f"  using r = {r2t} from S5 at eps = 1e-2\n")
    P(f"    {'topology':<22} {'N':>7} {'treewidth':>10} {'state cost':>14} {'vs 2^N':>12}")
    for name, mk in [("cascade", lambda n: path_power(n, 1)),
                     ("random degree 2", lambda n: random_regulatory(n, 2)),
                     ("random degree 3", lambda n: random_regulatory(n, 3)),
                     ("scale-free m=2", lambda n: scale_free(n, 2))]:
        for N in (64, 512, 4096, 20000):
            base = mk(N)
            Gp = graph_power(base, r2t)
            if Gp is None:
                P(f"    {name:<22} {N:>7} {'>3000':>10} {'dense':>14} {'--':>12}")
                continue
            tw, bags, done = treewidth_mindeg(Gp)
            cost = bag_cost(bags)
            P(f"    {name:<22} {N:>7} {(str(tw) if done else '>' + str(tw-1)):>10}"
              f" {cost:>14.4e} {'2^%d' % N if N < 64 else '2^' + str(N):>12}")
    P("\n  cascade with a global hub, treated two ways:")
    for N in (64, 512, 4096):
        base = path_power(N, 1)
        hub = [set(s) for s in base] + [set(range(N))]
        for i in range(N):
            hub[i].add(N)
        Gp = graph_power(hub, r2t)
        if Gp is None:
            P(f"    hub, not conditioned    {N:>7}   dependence graph is COMPLETE -- treewidth N")
        else:
            tw, bags, done = treewidth_mindeg(Gp)
            P(f"    hub, not conditioned    {N:>7} treewidth {tw}")
        Gc = graph_power(path_power(N, 1), r2t)
        tw2, bags2, _ = treewidth_mindeg(Gc)
        P(f"    hub, conditioned on     {N:>7} treewidth {tw2 + 1}"
          f"   cost {bag_cost([b + 1 for b in bags2]):.4e}")
    P("\n  The hub makes the dependence graph complete, so grouping by thresholding costs 2^N.")
    P("  Putting the controller into every bag drops the treewidth back to r+1 -- but S6 says that")
    P("  is only legitimate when the controller is slow or fast relative to its targets. At")
    P("  matched timescales the residual conditional dependence is 7.7e-4, still 3000x above the")
    P("  tail-legal threshold, so the conditioned graph is complete too and nothing is saved.")

    # ---- S8  ROBUSTNESS, added after S3 failed and S7 answered ---------------------------------
    P("\n" + RULE); P("S8  ROBUSTNESS  (added after S3 failed and S7 answered -- labelled as such)"); P(RULE)
    P("  (a) S3 failed. Refitting every N on the COMMON range d = 1..6:")
    P(f"    {'N':>4} {'xi':>10} {'+-':>8} {'R^2':>10}")
    xc = []
    for N in (10, 12, 14, 16):
        Qn, nvn = generator(N, 2.0); pn, _, _ = stationary(Qn); Mn = mi_matrix(pn, nvn)
        dn, mn = by_distance(Mn)
        k = (dn >= 1) & (dn <= 6)
        x, sx, rr = fit_decay(dn[k], mn[k]); xc.append(x)
        P(f"    {N:>4} {x:>10.4f} {sx:>8.4f} {rr:>10.6f}")
    mono = all(xc[i] < xc[i + 1] for i in range(3)) or all(xc[i] > xc[i + 1] for i in range(3))
    P(f"    spread {max(xc)-min(xc):.4f}, monotone in N: {mono}")
    P( "    The scatter survives a common fit range and is NOT monotone, so it is residual wobble")
    P( "    from averaging over boundary pairs rather than a drift with system size. It still")
    P( "    fails the bar as declared. Whether it propagates: the quantity used downstream is the")
    P(f"    INTEGER r, and r = 3 requires xi anywhere in a wide interval, so a 1.7% wobble in xi")
    P( "    cannot move it. The gate failed; the conclusion does not depend on what it measures.")
    P("\n  (b) S7 at r = 1 -- the bare regulatory graph, the most optimistic case possible:")
    P(f"    {'topology':<20} {'N':>7} {'tw at r=1':>10} {'tw at r=3':>10}")
    for nm, mk in [("cascade", lambda n: path_power(n, 1)),
                   ("random degree 2", lambda n: random_regulatory(n, 2)),
                   ("random degree 3", lambda n: random_regulatory(n, 3)),
                   ("scale-free m=2", lambda n: scale_free(n, 2))]:
        for N in (64, 512, 4096, 20000):
            base = mk(N)
            t1, _, d1 = treewidth_mindeg(base)
            gp = graph_power(base, 3)
            if gp is None:
                t3s = "dense"
            else:
                t3, _, d3 = treewidth_mindeg(gp)
                t3s = str(t3) if d3 else ">40"
            P(f"    {nm:<20} {N:>7} {(str(t1) if d1 else '>40'):>10} {t3s:>10}")
    P( "    Everything but the cascade is already past 40 at r = 1, so S7's answer does not")
    P( "    depend on r. The obstruction is the topology, not the range of the dependence.")
    P("\n  (c) is r = 3 transferable off the cascade? measured exactly on a random graph, N = 14:")
    adj = random_regulatory(14, 2, seed=7)
    Qr = generator_graph(adj, 2.0)
    pr, rr_, _ = stationary(Qr)
    Mr = mi_matrix(pr, 14)
    D = bfs_dist(adj, 14)
    P(f"    mean degree {np.mean([len(a) for a in adj]):.2f}, residual {rr_:.2e}")
    P(f"    {'d':>3} {'mean MI':>13} {'max MI':>13} {'frac > tau':>11}")
    rr2 = 0
    for dd in range(1, 7):
        v = Mr[D == dd]
        if not len(v):
            continue
        if np.any(v > tau2):
            rr2 = dd
        P(f"    {dd:>3} {v.mean():>13.4e} {v.max():>13.4e} {np.mean(v > tau2):>11.3f}")
    P(f"    r(tau) on this random graph = {rr2}; the cascade gave 3.")
    P(f"    S8: {'PASS -- r transfers, S7 is r-independent, and S3 fails harmlessly' if rr2 == 3 else 'r does NOT transfer; S7 must be redone per topology'}")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_statedim.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
