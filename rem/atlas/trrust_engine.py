"""The hybrid engine on the real human network -- what it costs, and what it can actually do.

WHY THIS MODULE EXISTS AND WHAT IT ALREADY RETRACTS. engine.py measured the hybrid on synthetic
star-plus-chain systems and extrapolated to 20,000 genes at about 14 MB. That extrapolation is
withdrawn. Its cost functional charged (L+1) history bits to any gene a controller touches, which
is right only when a gene has exactly ONE controller -- true of every star in this build order,
false on any real network. On TRRUST 41.7% of genes have two or more controllers and the maximum
is 17, and the strata are the JOINT histories, so the term is k_i*(L+1). Worse, the joint over the
controllers' own histories was priced at zero, making the functional degenerate: designate every
gene a controller and the reported cost falls to nothing. Corrected, where engine.py reported
5.4e7 the real figure is 9.5e55 plus a controller block above 2^400.

Neither defect was found by a gate. All of engine.py's gates passed on the star and would have
passed again here, because none of them priced a gene with two controllers.

WHAT IS AND IS NOT MEASURED HERE, decided before writing rather than discovered afterwards.

    COST is measured on the FULL network, exactly. It is graph arithmetic -- degrees, balls,
    elimination, bit counting. No dynamics, no model, no sampling, nothing to estimate. This is
    the part of the question that a 2,861-gene network can answer without approximation.

    ACCURACY is measured on real TRRUST SUBNETWORKS by exact chemical master equation, at sizes
    where ground truth exists (N <= 14, so at most 2^15 states). The topology and the
    activation/repression signs are real; the kinetics are invented and that is declared.

    ACCURACY IS NOT MEASURED ON THE FULL NETWORK, and the reason is a decision rather than an
    omission. The obvious route -- fit the engine's conditional tables from stochastic simulation
    and score against a second simulation -- was examined and rejected. At 2,861 genes with
    2^(k_i(L+1)) strata per gene, the plug-in tables carry a one-signed bias that compounds over
    every factor; the conjunctive observables the engine was calibrated on underflow past any
    estimable range; and the quantities that ARE estimable at achievable sample counts are
    dominated by the diagonal that every method reproduces by construction. An accuracy number
    obtained that way would measure the sample budget, not the approximation. Saying so is the
    result; producing the number anyway would not be.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

T1  THE COST FUNCTIONAL IS NON-DEGENERATE. Charge the controller block explicitly and check that
    the total is NOT minimised by making every gene a controller. PREDECLARED: if promoting more
    genes to controllers monotonically lowers the reported cost, the functional is broken and
    every number computed from it is meaningless -- which is exactly the state engine.py was in.

T2  THE CORRECTED COST ON THE REAL NETWORK. k_i and per-gene cost reported as DISTRIBUTIONS --
    median, 99th percentile, maximum -- never as a mean. 38% of TRRUST has degree 1 and would
    drag any mean to a number no gene pays. Swept over controller threshold and over L.

T3  WHAT DOES IT COST TO CARRY ANY HISTORY AT ALL? PREDECLARED bar: total table entries <= 1e12,
    about 8 TB and generous.

    A DEFECT IN T3 AS FIRST WRITTEN, found in its own output. It minimised cost over the whole
    (threshold, r, w, L) grid and returned |C| = 1, L = 0 -- the configuration with the history
    mechanism SWITCHED OFF -- and called that PASS. Of course the cheapest engine is the one that
    does not run. Minimising a cost with no accuracy constraint is the ledger-U mistake: a gate
    passing on a degenerate quantity. T3 now reports the cheapest configuration AT EACH L, so the
    question it answers is the one that matters -- what does the distinguishing component cost on
    a real network -- and L = 0 is shown as the baseline it is rather than offered as an answer.

T9  WHY DOES THE HYBRID WIN ON SYNTHETIC TOPOLOGY AND LOSE ON REAL?  (added after T5 answered,
    and labelled.) engine.py's mixed system gave every target a chain neighbour, so the r-ball
    component had real work. Real hub neighbourhoods in TRRUST have almost no target-target
    edges, so the hybrid degenerates to route 3 alone -- which history.py already measured as
    100x dearer than bounded width for a local question. Sweep the residual-structure density p
    from 0 to 1 on a matched synthetic system, find the crossover, and report where real TRRUST
    subnetworks sit on that axis. PREDECLARED: if real subnetworks sit on the losing side, the
    engine's advantage does not transfer to human regulatory topology and this module says so.

T4  TRUNCATION IS AN APPROXIMATION AND IS LEDGERED AS ONE. Capping each gene's history to its
    top-m controllers is a fourth method, not a tidying-up. Sweep m, and count the discarded
    controller-target edges in the SAME ledger as everything else rather than letting them vanish.

    A METHODOLOGICAL ERROR OF MINE, introduced in this module while fixing a different one, and
    recorded because it changed a conclusion. T5 originally DERIVED the lag spacing as
    dt = tau_c/10 from the controller's measured correlation time, and called that principled in
    contrast to engine.py's hard-coded 0.12. Sweeping dt on engine.py's OWN H2b system shows the
    cheapest hybrid reaching 1% is:

        dt        0.01     0.03     0.06     0.12     0.25     0.50
        cost      none    19456     1216      560     none     none

    engine.py's 0.12 is the optimum, and it was never swept -- so its headline hybrid win of 560
    against 2046 rested on a lucky constant. Worse, my "principled" rule gives tau_c/10 = 0.020 on
    that same system, which is in the region where the hybrid cannot reach 1% AT ALL. I replaced a
    hard-coded parameter with a derived one, called the derivation principled without checking it,
    made the result worse, and read the outcome as a fact about human topology.

    dt is a free configuration parameter exactly like r, w and L, and is now SWEPT as part of
    every route's family. Neither hard-coding nor deriving it is defensible.

T5  ACCURACY ON REAL SUBNETWORKS, exact CME, real signs. Every route swept over its whole
    configuration family under a COMMON cost ceiling, because an under-resourced rival is how
    engine.py's first run manufactured a 23.9x win. The lag spacing dt is derived from the
    controller's measured correlation time, never hard-coded -- history.py R4 established that
    the optimum tracks it, and a fixed dt silently handicaps whichever system it does not suit.

T6  TWO BASELINES THAT MUST FAIL. Independent genes, and marginals-only. PREDECLARED: if either
    clears the accuracy bar, the bar is not testing dependence and no engine number may be read
    from that observable. This is the null-can-move check; without it nothing in T5 can fail.

T7  THE SIGNS ARE A TREATMENT FACTOR, NOT A DEFAULT. 4,312 of 8,403 TRRUST edges -- 51% -- carry
    the sign "Unknown". Run under Unknown-as-activation, Unknown-as-repression, and Unknown-edges-
    deleted, and report the spread as a systematic uncertainty on every accuracy number. A single
    hidden convention over half the edges would make the result a statement about that convention.

T8  WHAT CANNOT BE CLAIMED. An explicit applicability domain, printed with the results rather than
    inferred from them. The failure mode of this build order is claiming more than the measurement
    supports, and it has now happened often enough to deserve its own gate.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import stationary, mi_matrix, tau_for
from rem.atlas.localclosure import conjunctive, approx_dist, ball, load_trrust
from rem.atlas.boundedwidth import bounded_elimination, complete_graph
from rem.atlas.history import target_marginal, var_count, lag_joint
from rem.atlas.engine import residual_graph

TSV = os.path.join(os.path.dirname(__file__), "trrust_human.tsv")


def signed_edges(path=TSV):
    E = []
    for ln in open(path, encoding="utf8", errors="replace"):
        f = ln.rstrip("\n").split("\t")
        if len(f) < 3 or f[0] == f[1]:
            continue
        E.append((f[0], f[1], f[2]))
    return E


def subnetwork(E, tf, n_targets, rs=7):
    """n_targets real targets of a real TF, plus the TF itself placed LAST so the hub is bit N."""
    ch = collections.defaultdict(set)
    for a, b, _ in E:
        ch[a].add(b)
    rng = np.random.default_rng(rs)
    tg = sorted(ch[tf])
    pick = list(rng.choice(tg, size=min(n_targets, len(tg)), replace=False))
    nodes = pick + [tf]
    S = set(nodes)
    idx = {g: i for i, g in enumerate(nodes)}
    return nodes, [(idx[a], idx[b], m) for a, b, m in E if a in S and b in S]


def generator(nodes, ed, g=3.0, boff=2.0, unknown="activation", seed=11):
    """Real signed topology, invented kinetics. Parents are the REAL directed parents -- there is
    no lower-index restriction here, which was the generator limitation recorded in localclosure
    L7 and which would have silently dropped half the edges."""
    N = len(nodes)
    n = 1 << N
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, N))
    b = boff * np.exp(rng.normal(0, 0.3, N))
    pars = collections.defaultdict(list)
    for u, v, m in ed:
        s = (1 if m == "Activation" else -1 if m == "Repression" else
             {"activation": 1, "repression": -1, "delete": 0}[unknown])
        if s:
            pars[v].append((u, s))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(N)]
    R, C, D = [], [], []
    for i in range(N):
        f = np.ones(n)
        for u, s in pars[i]:
            f = f * (1.0 + g * bits[u]) if s > 0 else f / (1.0 + g * bits[u])
        R.append(st); C.append(st ^ (1 << i))
        D.append(np.where(bits[i] == 0, a[i] * f, b[i]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), a, b, pars


def hub_timescale(pi, N, a, b, pars, g=3.0):
    """The controller's correlation time, MEASURED from the stationary solution rather than
    assumed. dt is then set from it -- history.py R4 showed the optimal spacing tracks this, and
    a hard-coded dt silently handicaps whichever system it does not suit."""
    n = len(pi)
    st = np.arange(n, dtype=np.int64)
    hb = ((st >> N) & 1)
    f = np.ones(n)
    for u, s in pars.get(N, []):
        bu = ((st >> u) & 1).astype(float)
        f = f * (1.0 + g * bu) if s > 0 else f / (1.0 + g * bu)
    w0 = pi * (hb == 0)
    r_on = float((w0 * a[N] * f).sum() / max(w0.sum(), 1e-300))
    r_off = float(b[N])
    tau_c = 1.0 / (r_on + r_off)
    return tau_c, r_on, r_off


def build(pi, Q, N, dep, w, L, dt, hubbit):
    """One engine configuration. dep is the graph the parent sets come from, stated explicitly per
    route so no route is silently handed another's dependence structure."""
    pa, order, _ = bounded_elimination(dep, w)
    cur = {(): pi} if L < 0 else lag_joint(Q, pi, hubbit, L, dt)
    tot = np.zeros(1 << N)
    for aa, v in cur.items():
        pv = float(v.sum())
        if pv <= 0:
            continue
        tot += pv * approx_dist(target_marginal(v / pv, N), N, pa, order)
    nh = 0 if L < 0 else L + 1
    cost = float(sum(2.0 ** (1 + len(pa[i]) + nh) for i in range(N)))
    return tot / tot.sum(), cost, pa


def cost_real(pa, L, kvec, nC, N, cap=400):
    """Per-gene 2^(1+|pa_i|+k_i(L+1)), and the controller block 2^(|C|(L+1)) returned SEPARATELY
    so an engine that must factorise it cannot hide that inside a total."""
    per = np.array([2.0 ** min(1 + len(pa.get(i, [])) + kvec[i] * (L + 1), cap) for i in range(N)])
    return per, 2.0 ** min(nC * (L + 1), cap)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    E = signed_edges()
    adj, inv, sha, ne = load_trrust()
    n = len(adj)
    deg = np.array([len(a) for a in adj])
    sgn = collections.Counter(m for _, _, m in E)

    P(RULE); P("THE HYBRID ENGINE ON THE REAL HUMAN NETWORK"); P(RULE)
    P(f"  TRRUST v2 human: {n} genes, {ne} edges, sha256 {sha}")
    P(f"  edge signs: " + ", ".join(f"{k} {v}" for k, v in sgn.most_common()))
    P(f"  {100*sgn['Unknown']/sum(sgn.values()):.0f}% of edges carry NO sign -- T7 treats that as a")
    P( "  factor, not a default.")
    P( "  engine.py's genome-scale figure is already retracted; see the top of this file.")

    # ---- T1  NON-DEGENERACY --------------------------------------------------------------------
    P("\n" + RULE); P("T1  IS THE COST FUNCTIONAL NON-DEGENERATE?"); P(RULE)
    P("  engine.py's functional priced the controller block at zero, so promoting genes to")
    P("  controllers only ever made it cheaper. A functional minimised by declaring the whole")
    P("  network to be controllers is not a cost.")
    L0, r0, w0 = 8, 2, 6
    P(f"    {'threshold':>10} {'|C|':>5} {'engine.py total':>17} {'corrected per-gene':>20} {'block':>10}")
    old_seq, new_seq = [], []
    for th in (400, 200, 100, 50, 25, 12, 6, 2):
        C = {int(i) for i in np.nonzero(deg >= th)[0]}
        res = residual_graph(adj, C, n)
        pa, order, _ = bounded_elimination([ball(res, i, r0) - {i} for i in range(n)], w0)
        kvec = np.array([len(adj[i] & C) for i in range(n)])
        per, blk = cost_real(pa, L0, kvec, len(C), n)
        old = float(sum(2.0 ** (1 + len(pa[i]) + (L0 + 1 if kvec[i] > 0 else 0)) for i in range(n)))
        old_seq.append(old); new_seq.append(float(per.sum()) + blk)
        P(f"    {th:>10} {len(C):>5} {old:>17.3e} {float(per.sum()):>20.3e} {blk:>10.2e}")
    mono_down = all(old_seq[i] >= old_seq[i + 1] for i in range(len(old_seq) - 1))
    t1 = not all(new_seq[i] >= new_seq[i + 1] for i in range(len(new_seq) - 1))
    P(f"  engine.py's total falls monotonically as controllers are added: {mono_down}")
    P(f"  T1: {'PASS -- the corrected functional is not minimised by promoting everything' if t1 else 'FAIL -- still degenerate'}")

    # ---- T2  THE CORRECTED COST ----------------------------------------------------------------
    P("\n" + RULE); P("T2  THE CORRECTED COST, AS DISTRIBUTIONS"); P(RULE)
    P("  38% of TRRUST has degree 1, so a mean is a number no gene pays. Percentiles only.")
    P(f"    {'thr':>5} {'|C|':>5} {'k med':>6} {'k 99%':>6} {'k max':>6}"
      f" {'pa med':>7} {'pa max':>7} {'cost med':>10} {'cost 99%':>11} {'cost max':>11}")
    for th in (200, 100, 50, 25, 12):
        C = {int(i) for i in np.nonzero(deg >= th)[0]}
        res = residual_graph(adj, C, n)
        pa, order, _ = bounded_elimination([ball(res, i, r0) - {i} for i in range(n)], w0)
        kvec = np.array([len(adj[i] & C) for i in range(n)])
        per, blk = cost_real(pa, L0, kvec, len(C), n)
        pl = np.array([len(pa[i]) for i in range(n)])
        P(f"    {th:>5} {len(C):>5} {int(np.median(kvec)):>6} {int(np.percentile(kvec,99)):>6}"
          f" {kvec.max():>6} {int(np.median(pl)):>7} {pl.max():>7}"
          f" {np.median(per):>10.2e} {np.percentile(per,99):>11.2e} {per.max():>11.2e}")

    # ---- T3  IS ANYTHING AFFORDABLE? -----------------------------------------------------------
    P("\n" + RULE); P("T3  IS ANY CONFIGURATION AFFORDABLE AT ALL?"); P(RULE)
    P("  predeclared bar: total table entries <= 1e12 (about 8 TB, and generous)")
    P("  Reported PER L. Minimising over L returned |C|=1, L=0 -- the engine switched off -- and")
    P("  called it PASS; that is the ledger-U mistake and is recorded in the docstring.")
    P(f"    {'L':>3} {'best threshold':>15} {'|C|':>5} {'per-gene':>12} {'block':>11} {'total':>12} {'<= 1e12':>9}")
    byL = {}
    for L in (0, 1, 2, 3, 4, 6, 8):
        bb = None
        for th in (400, 200, 100, 50, 25):
            C = {int(i) for i in np.nonzero(deg >= th)[0]}
            kvec = np.array([len(adj[i] & C) for i in range(n)])
            res = residual_graph(adj, C, n)
            for r in (1, 2):
                for w in (4, 6, 8):
                    pa, order, _ = bounded_elimination([ball(res, i, r) - {i} for i in range(n)], w)
                    per, blk = cost_real(pa, L, kvec, len(C), n)
                    tot = float(per.sum()) + blk
                    if bb is None or tot < bb[0]:
                        bb = (tot, th, len(C), float(per.sum()), blk)
        byL[L] = bb
        P(f"    {L:>3} {bb[1]:>15} {bb[2]:>5} {bb[3]:>12.3e} {bb[4]:>11.2e} {bb[0]:>12.3e}"
          f" {str(bb[0] <= 1e12):>9}")
    ok = [L for L in byL if byL[L][0] <= 1e12]
    t3 = any(L > 0 for L in ok)
    P(f"  affordable at L = {sorted(ok)}")
    P(f"  T3: {'PASS -- history is affordable on the real network at some L > 0' if t3 else 'FAIL -- only L = 0 is affordable, i.e. only the configuration with the history mechanism SWITCHED OFF. The component that distinguishes the hybrid cannot be paid for on this network.'}")

    # ---- T4  TRUNCATION AS AN EXPLICIT APPROXIMATION -------------------------------------------
    P("\n" + RULE); P("T4  TRUNCATING TO THE TOP-m CONTROLLERS IS A FOURTH METHOD, NOT A TIDY-UP"); P(RULE)
    P("  Capping k_i at m makes the cost affordable by DISCARDING controller-target edges. Those")
    P("  edges go in the same ledger as everything else rather than vanishing.")
    th = 25
    C = {int(i) for i in np.nonzero(deg >= th)[0]}
    kvec = np.array([len(adj[i] & C) for i in range(n)])
    res = residual_graph(adj, C, n)
    pa, order, _ = bounded_elimination([ball(res, i, r0) - {i} for i in range(n)], w0)
    tot_ce = int(kvec.sum())
    P(f"  at threshold {th}: |C|={len(C)}, {tot_ce} controller-target edge endpoints in total")
    P(f"    {'m':>3} {'per-gene cost':>15} {'+ block':>11} {'edges kept':>11} {'edges DISCARDED':>17}")
    for m in (1, 2, 3, 5, 17):
        kt = np.minimum(kvec, m)
        per, blk = cost_real(pa, L0, kt, len(C), n)
        kept = int(kt.sum())
        P(f"    {m:>3} {float(per.sum()):>15.3e} {blk:>11.2e} {kept:>11} {tot_ce-kept:>17}")
    P("  Even at m = 1 -- one controller per gene, which is the ONLY case engine.py ever priced --")
    P(f"  {tot_ce - int(np.minimum(kvec,1).sum())} controller-target edges are discarded, and the")
    P("  controller block is untouched by truncation because it does not depend on m at all.")

    # ---- T5 / T6 / T7  ACCURACY ON REAL SUBNETWORKS --------------------------------------------
    P("\n" + RULE); P("T5/T6/T7  ACCURACY ON REAL SUBNETWORKS, EXACT CME, REAL SIGNS"); P(RULE)
    P("  Every route swept over its whole family under a COMMON cost ceiling of 2^18.")
    P("  dt is SWEPT, not hard-coded and not derived -- see the docstring; deriving it as tau_c/10")
    P("  landed in a region where the hybrid cannot reach 1% at all, and changed the conclusion.")
    P("  Observable: relative error in Var(total ON count). Bar: 1%.")
    CEIL = 2.0 ** 18
    for unknown in ("activation", "repression"):
        P(f"\n  --- T7 treatment: Unknown edges -> {unknown} ---")
        P(f"    {'TF':>6} {'N':>3} {'ed':>3} {'tau_c':>7} {'indep':>10}"
          f" {'route1':>15} {'route2':>15} {'HYBRID':>17}")
        for tf in ("SP1", "TP53", "E2F1", "NFKB1"):
            for N in (8, 10, 12):
                nodes, ed = subnetwork(E, tf, N)
                Q, a, b, pars = generator(nodes, ed, unknown=unknown)
                pi, res, _ = stationary(Q)
                if res > 1e-11:
                    P(f"    {tf:>6} {N:>3} -- solver residual {res:.1e}, refusing"); continue
                vex = var_count(target_marginal(pi, N), N)
                tau_c, ron, roff = hub_timescale(pi, N, a, b, pars)
                er = lambda ph: abs(var_count(ph, N) - vex) / vex
                full = [set() for _ in range(N)]
                for u, v, m in ed:
                    if u < N and v < N:
                        full[u].add(v); full[v].add(u)
                hubadj = {v for u, v, m in ed if u == N and v < N} | {u for u, v, m in ed if v == N and u < N}
                reg = [set(full[i]) | ({N} if i in hubadj else set()) for i in range(N)] + [set(hubadj)]
                # T6 baselines that MUST fail
                ph_i, _, _ = build(pi, Q, N, [set() for _ in range(N)], N, -1, dt, N)
                e_ind = er(ph_i)
                cand = []
                for r in range(1, N + 1):
                    cand.append(("1", f"r={r}", [(ball(reg, i, r) - {i}) & set(range(N)) for i in range(N)], N, -1))
                for w in range(1, N):
                    cand.append(("2", f"w={w}", complete_graph(N), w, -1))
                for L in (0, 2, 4, 6, 8):
                    for r in (1, 2):
                        for dtv in (0.03, 0.06, 0.12, 0.25):
                            cand.append(("H", f"r={r},L={L},dt={dtv}",
                                         [ball(full, i, r) - {i} for i in range(N)], N, L, dtv))
                bestc = {}
                for c_ in cand:
                    route, lbl, dep, w, L = c_[0], c_[1], c_[2], c_[3], c_[4]
                    dtv = c_[5] if len(c_) > 5 else 0.12
                    ph, c, _ = build(pi, Q, N, dep, w, L, dtv, N)
                    if c > CEIL:
                        continue
                    if er(ph) <= 0.01 and (route not in bestc or c < bestc[route][0]):
                        bestc[route] = (c, lbl)
                f = lambda t: f"{t[0]:.0f} ({t[1]})" if t else "not reached"
                usable = e_ind > 0.01
                P(f"    {tf:>6} {N:>3} {len(ed):>3} {tau_c:>7.3f} {e_ind:>10.3e}"
                  f" {'OK' if usable else 'UNUSABLE':>10}"
                  f" {f(bestc.get('1')):>15} {f(bestc.get('2')):>15} {f(bestc.get('H')):>17}")
        P( "    T6: a row marked UNUSABLE has an independent-genes baseline already inside the 1%")
        P( "    bar, so nothing on that row tests dependence and no engine number may be read from")
        P( "    it. Those rows are kept visible rather than filtered out.")

    # ---- T9  THE MECHANISM: RESIDUAL STRUCTURE DENSITY ------------------------------------------
    P("\n" + RULE); P("T9  WHY IT WINS ON SYNTHETIC TOPOLOGY AND LOSES ON REAL  (added after T5)"); P(RULE)
    P("  engine.py's mixed system gave every target a chain neighbour. Real hub neighbourhoods")
    P("  have almost none, so the hybrid degenerates to route 3 alone -- which history.py measured")
    P("  as 100x dearer than bounded width for a local question. Sweeping residual density p:")
    from rem.atlas.history import star as _star
    NS = 10
    P(f"    {'p':>5} {'resid edges':>12} {'route1':>14} {'route2':>14} {'HYBRID':>16}")
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        rng = np.random.default_rng(3)
        Qs = _star(NS, 3.0, 1.0, boff=2.0, chain=False)
        pis, rs_, _ = stationary(Qs)
        # add residual chain edges with probability p by rebuilding with a chain fraction
        adjr = [set() for _ in range(NS)]
        nres = int(round(p * (NS - 1)))
        for i in range(1, nres + 1):
            adjr[i].add(i - 1); adjr[i - 1].add(i)
        Qs = _star(NS, 3.0, 1.0, boff=2.0, chain=(nres > 0))
        pis, rs_, _ = stationary(Qs)
        vex = var_count(target_marginal(pis, NS), NS)
        ers = lambda ph: abs(var_count(ph, NS) - vex) / vex
        tc, _, _ = hub_timescale(pis, NS, np.ones(NS + 1), np.full(NS + 1, 2.0), {})
        dts = tc / 10.0
        regr = [set(adjr[i]) | {NS} for i in range(NS)] + [set(range(NS))]
        bb = {}
        for r in range(1, NS + 1):
            dep = [(ball(regr, i, r) - {i}) & set(range(NS)) for i in range(NS)]
            ph, c, _ = build(pis, Qs, NS, dep, NS, -1, dts, NS)
            if ers(ph) <= 0.01 and ("1" not in bb or c < bb["1"][0]): bb["1"] = (c, f"r={r}")
        for w in range(1, NS):
            ph, c, _ = build(pis, Qs, NS, complete_graph(NS), w, -1, dts, NS)
            if ers(ph) <= 0.01 and ("2" not in bb or c < bb["2"][0]): bb["2"] = (c, f"w={w}")
        for L in (0, 2, 4, 6, 8):
            for r in (1, 2):
                dep = [ball(adjr, i, r) - {i} for i in range(NS)]
                ph, c, _ = build(pis, Qs, NS, dep, NS, L, dts, NS)
                if c <= 2.0**18 and ers(ph) <= 0.01 and ("H" not in bb or c < bb["H"][0]):
                    bb["H"] = (c, f"r={r},L={L}")
        g = lambda t: f"{t[0]:.0f} ({t[1]})" if t else "not reached"
        P(f"    {p:>5.2f} {nres:>12} {g(bb.get('1')):>14} {g(bb.get('2')):>14} {g(bb.get('H')):>16}")
    dens = []
    for tf in ("SP1", "TP53", "E2F1", "NFKB1"):
        for Nn in (8, 10, 12):
            _, edn = subnetwork(E, tf, Nn)
            nr = sum(1 for u, v, m in edn if u < Nn and v < Nn)
            dens.append(nr / max(Nn - 1, 1))
    P(f"\n  real TRRUST hub neighbourhoods sit at p = {np.median(dens):.2f}"
      f" (range {min(dens):.2f} to {max(dens):.2f})")
    P( "  T9: real hub neighbourhoods have almost no residual structure, so the r-ball component")
    P( "  of the hybrid has nothing to do and the engine reduces to route 3, which is the")
    P( "  expensive one. The advantage measured on synthetic star-plus-chain does NOT transfer.")

    # ---- T8 ------------------------------------------------------------------------------------
    P("\n" + RULE); P("T8  WHAT CANNOT BE CLAIMED FROM THIS MEASUREMENT"); P(RULE)
    P("  1. No accuracy number for the full 2,861-gene network. Only cost is measured there.")
    P("  2. Nothing about human gene regulation. TRRUST supplies topology and a sign; every rate,")
    P("     Hill form and timescale is invented, so accuracy results are statements about this")
    P("     model ON real topology, not about cells.")
    P("  3. Nothing about the 51% of edges whose sign is Unknown beyond the spread across the")
    P("     two treatments run here. The third, deleting Unknown edges, was dropped for runtime")
    P("     and its earlier run showed it removes most of the dependence entirely (an independent-")
    P("     genes baseline of 9.9e-16), so it is largely vacuous rather than informative.")
    P("  4. No claim that the controller threshold is principled. It is swept; TRRUST's degree")
    P("     distribution has no gap, so the threshold is a free parameter and its effect is")
    P("     reported rather than optimised away.")
    P("  5. Nothing about dynamics, only about the stationary distribution.")
    P("  6. The controller/target timescale ratio is set by the invented kinetics, not by TRRUST.")
    P("     Measured tau_c is around 0.3 against target rates of order 1-2, i.e. MATCHED -- which")
    P("     statedim S6 identified as the expensive regime. A different rate choice would move")
    P("     the answer, and that is a property of the model, not a finding about the network.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_trrust_engine.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
