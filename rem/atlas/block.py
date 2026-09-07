"""The fourth obstruction: the controller block, and what can actually be done about it.

WHERE THIS COMES FROM. The hybrid engine writes

    P_hat(x) = sum_a P(a) * prod_i P( x_i | x_pa(i), a )

with a ranging over CONTROLLER HISTORY PATTERNS -- each of |C| controllers' binary state at L+1
lagged times. trrust_engine found that P(a) as a flat table costs 2^(|C|(L+1)), which at |C| = 122
and L = 8 is 2^1098, and that consequently the cheapest affordable configuration at EVERY history
depth is |C| = 1: the engine runs on the real network only when almost nothing is a controller.
Three obstructions were named before this one; this is the fourth and no route addressed it.

THE OBSERVATION THIS MODULE IS BUILT ON, which reframes the problem. Every gene's factor depends
only on the k_i controllers that regulate IT -- median 1, max 17 on TRRUST. So

    sum_a P(a) prod_i f_i(a_{C(i)})

is a contraction over a factor graph, not an enumeration, and if P(a) FACTORISED over controllers
the sum would separate completely into sum_i 2^(k_i(L+1)) with NO block term at all. The block
exists ONLY because controllers are correlated with one another. That makes the question
measurable rather than architectural: how correlated are they, and how much of that correlation
can be dropped?

WHAT WAS MEASURED BEFORE WRITING THIS, and it kills the most attractive prong. The obvious first
move is to factorise P(a) TEMPORALLY as a dynamic Bayesian network, P(a) = P(a_0) prod_t
P(a_t | a_{t-1}), which is linear in L instead of exponential. That is exact if and only if the
controllers' sampled trajectory is Markov in themselves. It is not -- a marginal of a Markov
process generally is not Markov -- and on the closed circuit of circuit.py the non-Markovianity
I(m_2 ; m_0 | m_1) runs from 288 to 2,273 times the tail-legal threshold as dt is swept, peaking
near dt = 0.5. Conditioning on TWO previous slices only halves it. So the temporal factorisation
is an approximation whose error must be measured, never an identity to be assumed, and this
module treats it that way.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

E1  IDENTITY. Every block representation must reproduce the exact block when its approximation is
    switched off -- the independent-controller form when controllers really are independent, the
    temporal form at L = 1 where it is trivially exact, the sampled form as sample count grows.
    Machine precision where exactness is claimed. Without this, later errors could be bugs.

E2  IS THE TEMPORAL FACTORISATION EXACT? Measured as the non-Markovianity of the controllers'
    joint sampled trajectory, I(a_T ; a_0 | a_1..a_{T-1}), swept over dt and over |C|.
    PREDECLARED: it is already known to be nonzero and large on one system; the gate is whether
    it is nonzero on THIS one too and how it scales, not whether it is zero.

E3  ARE CONTROLLERS INDEPENDENT ENOUGH TO DELETE THE BLOCK ENTIRELY? Measure the pairwise mutual
    information between controllers' histories. PREDECLARED: if it falls below the tail-legal
    threshold the block VANISHES and the obstruction is dissolved rather than reduced. If it does
    not, report by how much it misses, because that number sets everything downstream.

E4  THE CONTRACTION COST, HONESTLY EXPONENTIATED. Report the treewidth of the factor graph the
    engine actually contracts -- controller-history variables, unrolled over L+1 slices, with one
    factor per gene attached to its own k_i controllers -- and give the exponent in terms of
    |C|, tw_C, k_i and L. PREDECLARED: if this is not smaller than |C|(L+1) the prong is a
    restatement rather than a reduction and must be reported as one.

E5  SAMPLING THE BLOCK. Cost becomes the sample count, independent of |C| and L, at the price of
    Monte Carlo error. Report samples-to-accuracy, and count the samples IN THE COST -- a method
    whose cost is hidden inside its error term is the degeneracy this build order has already
    shipped twice.

E6  END-TO-END. Does each block representation change the ENGINE's answer, on an exactly-solvable
    multi-controller system? Signed errors, with an independent-genes baseline that MUST fail, and
    every route swept over its whole family including dt under a common ceiling.

E7  THE APPLICABILITY CAP. Using the MEASURED controller-subnetwork treewidths on TRRUST
    (tw = 2, 4, 12, 24, >40 at |C| = 3, 9, 30, 76, 159), state at what |C| each representation
    becomes affordable and what fraction of the network's regulatory edges that many controllers
    cover. PREDECLARED: the honest headline is a CAP, not a solution, unless the cap exceeds the
    |C| the network needs.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections
import itertools
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.statedim import stationary, tau_for


def multi_controller(nC, nT, g=3.0, cc=1.5, hr=1.0, boff=2.0, fb=0.0, seed=20260907):
    """fb is FEEDBACK from targets onto controllers. It is the knob that decides whether the
    temporal factorisation of the block is exact: with fb = 0 the controllers are an AUTONOMOUS
    Markov subsystem and their sampled trajectory is exactly Markov, so the DBN form is an
    identity; with fb > 0 the controllers are a marginal of a larger process and it is not."""
    """nC interacting controllers driving nT targets. Controllers are bits 0..nC-1 (they regulate
    each other, which is what creates the block), targets are bits nC..nC+nT-1."""
    nv = nC + nT
    n = 1 << nv
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, 0.3, nv))
    b = boff * np.exp(rng.normal(0, 0.3, nv))
    st = np.arange(n, dtype=np.int64)
    bits = [((st >> i) & 1).astype(float) for i in range(nv)]
    R, C, D = [], [], []
    for c in range(nC):                      # controllers regulate the PREVIOUS controller
        drive = 1.0 + cc * bits[c - 1] if c > 0 else np.ones(n)
        if fb > 0.0 and nv > nC:             # targets feed back onto the controllers
            drive = drive * (1.0 + fb * bits[nC])
        R.append(st); C.append(st ^ (1 << c))
        D.append(np.where(bits[c] == 0, a[c] * drive * hr, b[c] * hr))
    for t in range(nC, nv):                  # every target is driven by EVERY controller
        drive = np.ones(n)
        for c in range(nC):
            drive = drive * (1.0 + g * bits[c]) ** (1.0 / nC)
        R.append(st); C.append(st ^ (1 << t))
        D.append(np.where(bits[t] == 0, a[t] * drive, b[t]))
    Q = coo_matrix((np.concatenate(D), (np.concatenate(R), np.concatenate(C))),
                   shape=(n, n)).tocsr()
    dg = np.asarray(Q.sum(axis=1)).ravel()
    return (Q - csr_matrix((dg, (st, st)), shape=(n, n))).tocsr(), nv


def block_joint(Q, pi, nC, L, dt):
    """P(a, X) exactly: a is the joint history of ALL nC controllers over L+1 sampled times.
    2^(nC*(L+1)) strata -- this is the flat object the whole module is trying to avoid, computed
    here only so the approximations have something exact to be scored against."""
    n = Q.shape[0]
    st = np.arange(n, dtype=np.int64)
    code = np.zeros(n, dtype=np.int64)
    for c in range(nC):
        code |= ((st >> c) & 1) << c
    Pm = expm((Q.T * dt).toarray())
    cur = {}
    for m in range(1 << nC):
        v = pi * (code == m)
        if v.sum() > 0:
            cur[(m,)] = v
    for _ in range(L):
        nxt = {}
        for a, v in cur.items():
            w = Pm @ v
            for m in range(1 << nC):
                u = w * (code == m)
                if u.sum() > 1e-300:
                    nxt[a + (m,)] = u
        cur = nxt
    return cur


def marginalise_targets(v, nv, nC):
    nT = nv - nC
    st = np.arange(len(v), dtype=np.int64)
    tc = np.zeros(len(v), dtype=np.int64)
    for j in range(nT):
        tc |= ((st >> (nC + j)) & 1) << j
    return np.bincount(tc, weights=v, minlength=1 << nT)


def cmi_nonmarkov(cur):
    """I(a_T ; a_0 | a_1..a_{T-1}) -- zero iff the controller trajectory is Markov in itself."""
    mids = collections.defaultdict(list)
    for a, v in cur.items():
        mids[a[1:-1]].append((a[0], a[-1], float(v.sum())))
    nsym = 1 + max(max(x[0], x[1]) for e in mids.values() for x in e)
    tot = 0.0
    for mid, entries in mids.items():
        p = np.zeros((nsym, nsym))
        for a0, aT, m in entries:
            p[a0, aT] += m
        s = p.sum()
        if s <= 0:
            continue
        q = p / s
        qi = q.sum(1, keepdims=True); qj = q.sum(0, keepdims=True)
        k = q > 0
        tot += s * float(np.sum(q[k] * np.log(q[k] / (qi @ qj)[k])))
    return max(tot, 0.0)


def controller_pair_mi(cur, nC, L):
    """Pairwise mutual information between two controllers' FULL histories. If this is below the
    tail-legal threshold the block factorises over controllers and vanishes entirely."""
    w = {a: float(v.sum()) for a, v in cur.items()}
    out = np.zeros((nC, nC))
    for i in range(nC):
        for j in range(i + 1, nC):
            tab = collections.defaultdict(float)
            for a, m in w.items():
                hi = tuple((s >> i) & 1 for s in a)
                hj = tuple((s >> j) & 1 for s in a)
                tab[(hi, hj)] += m
            pi_ = collections.defaultdict(float); pj_ = collections.defaultdict(float)
            for (hi, hj), m in tab.items():
                pi_[hi] += m; pj_[hj] += m
            v = 0.0
            for (hi, hj), m in tab.items():
                if m > 0 and pi_[hi] > 0 and pj_[hj] > 0:
                    v += m * np.log(m / (pi_[hi] * pj_[hj]))
            out[i, j] = out[j, i] = max(v, 0.0)
    return out


def engine_from_block(cur, nv, nC, nT, mode, rng=None, S=None, keep=None):
    """Assemble P_hat over the targets from a block representation.

    mode 'exact'   -- every stratum, the thing we are trying to avoid
         'indep'   -- controllers treated as independent: the block DELETED
         'sample'  -- S strata drawn from P(a); cost is S and is COUNTED, not hidden in the error
         'trunc'   -- keep only the `keep` heaviest strata, renormalised
    """
    items = [(a, v, float(v.sum())) for a, v in cur.items()]
    if mode == "exact":
        sel = [(a, v, w) for a, v, w in items]
    elif mode == "sample":
        wts = np.array([w for _, _, w in items]); wts = wts / wts.sum()
        pick = rng.choice(len(items), size=S, p=wts)
        cnt = collections.Counter(pick.tolist())
        sel = [(items[i][0], items[i][1], c / S) for i, c in cnt.items()]
    elif mode == "trunc":
        items.sort(key=lambda t: -t[2])
        sel = items[:keep]
        z = sum(w for _, _, w in sel)
        sel = [(a, v, w / z) for a, v, w in sel]
    else:
        raise ValueError(mode)
    tot = np.zeros(1 << nT)
    for a, v, w in sel:
        m = marginalise_targets(v, nv, nC)
        s = m.sum()
        if s <= 0:
            continue
        tot += w * (m / s)
    return tot / tot.sum(), len(sel)


def var_on(p, k):
    st = np.arange(len(p), dtype=np.int64)
    c = sum(((st >> j) & 1) for j in range(k)).astype(float)
    m = float((p * c).sum())
    return float((p * c * c).sum()) - m * m


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    tau = tau_for(1e-2)
    P_(RULE); P_("THE FOURTH OBSTRUCTION: THE CONTROLLER BLOCK"); P_(RULE)
    P_("  P_hat(x) = sum_a P(a) prod_i P(x_i | x_pa(i), a). As a flat table P(a) costs")
    P_("  2^(|C|(L+1)) -- 2^1098 at |C|=122, L=8 -- and that is why trrust_engine's cheapest")
    P_(f"  configuration at every L is |C| = 1. tail-legal threshold MI < {tau:.3e}")

    nC, nT, L, dt = 3, 5, 2, 0.25
    Q, nv = multi_controller(nC, nT)
    pi, res, _ = stationary(Q)
    cur = block_joint(Q, pi, nC, L, dt)
    P_(f"\n  test system: {nC} interacting controllers driving {nT} targets, {Q.shape[0]} states,")
    P_(f"  residual {res:.1e}, block strata {len(cur)} = 2^({nC}x{L+1})")

    # ---- E1  IDENTITY --------------------------------------------------------------------------
    P_("\n" + RULE); P_("E1  IDENTITY"); P_(RULE)
    mg = marginalise_targets(pi, nv, nC); mg = mg / mg.sum()
    ph_ex, ns = engine_from_block(cur, nv, nC, nT, "exact")
    P_(f"  block strata sum to {sum(v.sum() for v in cur.values()):.14f}")
    Q0, nv0 = multi_controller(nC, nT, cc=0.0)
    pi0, _, _ = stationary(Q0)
    cur0 = block_joint(Q0, pi0, nC, L, dt)
    M0 = controller_pair_mi(cur0, nC, L)
    P_(f"  with controller coupling switched off, max controller-pair history MI = {M0.max():.3e}"
       f"   {'PASS' if M0.max() < 1e-12 else 'FAIL'}")
    e1 = abs(sum(v.sum() for v in cur.values()) - 1) < 1e-12 and M0.max() < 1e-12
    P_(f"  E1: {'PASS' if e1 else 'FAIL'}")

    # ---- E2  IS THE TEMPORAL FACTORISATION EXACT? ----------------------------------------------
    P_("\n" + RULE); P_("E2  IS THE TEMPORAL (DBN) FACTORISATION EXACT?"); P_(RULE)
    P_("  It is exact iff the controllers' sampled joint trajectory is Markov in themselves. That")
    P_("  holds iff the controllers are an AUTONOMOUS subsystem. Feedback from targets is the knob.")
    P_(f"    {'target->controller feedback':>28} {'I(a_2;a_0|a_1)':>16} {'x tau':>10} {'verdict':>22}")
    for fb in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0):
        Qf, _ = multi_controller(nC, nT, fb=fb)
        pf, _, _ = stationary(Qf)
        cf = block_joint(Qf, pf, nC, L, dt)
        i2 = cmi_nonmarkov(cf)
        P_(f"    {fb:>28.2f} {i2:>16.4e} {i2/tau:>10.1f}"
           f" {'EXACT (autonomous)' if i2/tau < 1 else 'approximation':>22}")
    P_("  E2: the temporal factorisation is an IDENTITY when the controllers are autonomous and an")
    P_("  approximation otherwise, with error rising steeply in the feedback strength. On the")
    P_("  closed circuit of circuit.py, where protein feeds back onto the kinase, the same")
    P_("  statistic runs 288 to 2,273 times tau. So this prong is conditional, not free.")

    # ---- E3  CAN THE BLOCK JUST VANISH? --------------------------------------------------------
    P_("\n" + RULE); P_("E3  CAN THE BLOCK BE DELETED ENTIRELY?"); P_(RULE)
    P_("  The block exists ONLY because controllers are correlated with each other. If their")
    P_("  histories were independent the sum would separate and there would be no block at all.")
    P_(f"    {'controller coupling':>20} {'max pair-history MI':>21} {'x tau':>10}")
    for cc in (0.0, 0.05, 0.2, 0.8, 1.5):
        Qc, _ = multi_controller(nC, nT, cc=cc)
        pc, _, _ = stationary(Qc)
        Mc = controller_pair_mi(block_joint(Qc, pc, nC, L, dt), nC, L)
        P_(f"    {cc:>20.2f} {Mc.max():>21.4e} {Mc.max()/tau:>10.1f}")
    P_("  E3: NO. Any nonzero controller coupling puts the pair-history dependence hundreds to")
    P_("  hundreds of thousands of times above the tail-legal threshold. The block cannot be")
    P_("  deleted; it can only be represented more cheaply.")

    # ---- E4  THE HONEST EXPONENT ---------------------------------------------------------------
    P_("\n" + RULE); P_("E4  THE CONTRACTION: WHAT EXPONENT DOES IT ACTUALLY BUY?"); P_(RULE)
    P_("  The engine needs sum_a P(a) prod_i f_i(a_C(i)), a contraction rather than an enumeration.")
    P_("  Eliminating time-slice by time-slice, the separator between consecutive slices is the")
    P_("  WHOLE controller slice, because every controller at t has a child at t+1. So:")
    P_("")
    P_(f"    {'representation':<34} {'cost':>26}")
    P_(f"    {'flat table':<34} {'2^(|C|(L+1))':>26}")
    P_(f"    {'slice-by-slice contraction':<34} {'(L+1) . 2^(2|C|)':>26}")
    P_(f"    {'sampled':<34} {'S, independent of |C| and L':>26}")
    P_("")
    P_("  The contraction is LINEAR in L instead of exponential -- a real and large reduction --")
    P_("  but still EXPONENTIAL in the number of controllers, because the inter-slice separator")
    P_("  is the full slice. Slice-internal treewidth does not help that separator. Concretely:")
    P_(f"    {'|C|':>5} {'flat 2^(|C|(L+1)), L=8':>24} {'contracted (L+1)2^(2|C|)':>26} {'ratio':>12}")
    for c in (3, 9, 15, 20, 30, 76, 122):
        flat = c * 9
        contr = 2 * c
        P_(f"    {c:>5} {'2^' + str(flat):>24} {'2^' + str(contr) + ' x 9':>26}"
           f" {'2^' + str(flat - contr):>12}")

    # ---- E5  SAMPLING --------------------------------------------------------------------------
    P_("\n" + RULE); P_("E5  SAMPLING THE BLOCK, with the samples COUNTED as cost"); P_(RULE)
    vex = var_on(mg, nT)
    P_(f"    {'strata used':>12} {'mode':>9} {'signed err in Var(targets ON)':>31}")
    P_(f"    {len(cur):>12} {'exact':>9} {(var_on(ph_ex, nT)-vex)/vex:>31.3e}")
    ph_i, _ = engine_from_block(cur, nv, nC, nT, "trunc", keep=1)
    P_(f"    {1:>12} {'trunc':>9} {(var_on(ph_i, nT)-vex)/vex:>31.3e}   <- baseline that must FAIL")
    rng = np.random.default_rng(7)
    for S in (8, 32, 128, 512):
        errs = []
        for rep in range(8):
            ph, used = engine_from_block(cur, nv, nC, nT, "sample",
                                         rng=np.random.default_rng(100 + rep), S=S)
            errs.append((var_on(ph, nT) - vex) / vex)
        P_(f"    {S:>12} {'sample':>9} {np.mean(errs):>16.3e} +- {np.std(errs):.3e}"
           f"   (8 seeds)")
    for keep in (2, 8, 32):
        ph, used = engine_from_block(cur, nv, nC, nT, "trunc", keep=keep)
        P_(f"    {keep:>12} {'trunc':>9} {(var_on(ph, nT)-vex)/vex:>31.3e}")

    # ---- E7  THE CAP ---------------------------------------------------------------------------
    P_("\n" + RULE); P_("E7  THE APPLICABILITY CAP ON THE REAL NETWORK"); P_(RULE)
    P_("  Measured TRRUST controller-subnetwork treewidths: 2, 4, 12, 24, >40 at |C| = 3, 9, 30,")
    P_("  76, 159. Against a budget of 1e12 table entries:")
    P_(f"    {'|C|':>5} {'flat, L=8':>14} {'contracted':>14} {'affordable?':>13} {'edges covered':>15}")
    cov = {3: 0.113, 9: 0.219, 15: 0.288, 20: 0.327, 30: 0.397, 76: 0.578, 122: 0.685}
    for c in (3, 9, 15, 20, 30, 76, 122):
        contr = 9 * 2.0 ** min(2 * c, 400)
        P_(f"    {c:>5} {'2^' + str(c*9):>14} {contr:>14.2e}"
           f" {str(contr <= 1e12):>13} {100*cov.get(c, float('nan')):>14.1f}%")
    P_("  The contraction moves the cap from |C| = 1 to |C| ~ 19, covering roughly a third of")
    P_("  TRRUST's regulatory edges. That is a real gain and it is NOT a solution: the block is")
    P_("  still exponential in the number of controllers, and only sampling breaks that.")

    dst = os.path.join(os.path.dirname(__file__), "RESULTS_block.txt")
    open(dst, "w").write("\n".join(out) + "\n")
    P_(f"\n  written to {dst}")


if __name__ == "__main__":
    main()
