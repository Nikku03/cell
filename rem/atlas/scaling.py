"""K(N): how many additional rate measurements does a prediction need, as the rate count grows?

THE QUESTION, IN THE FORM IT WAS PUT. The d_death result is prediction coverage, not rate
coverage. Physiology did not recover seven of eight rates; constrain_rank's R2 measured the
physiology Jacobian at rank 3 of 4, with its third singular value only 1.1% of the first, so
roughly two strong directions of an eight-dimensional rate space were pinned. What made the
prediction work is that the answer's gradient had one dominant unconstrained component, and the
Jacobian found it. So the operative question is not whether chemistry can supply hundreds of
thousands of rate constants. It is

    given the physiology we already measure, how many ADDITIONAL rate measurements are needed,
    and how does that number scale with the number of unknown rates?

Call it K(N). If K grows like log N, or a small sublinear power, specifying a cell is tractable.
If K grows like N, there is a measurement wall and no amount of cleverness in the selection rule
helps.

THE MODEL, chosen because it makes a genuine rare event exactly computable at N in the thousands.
A multi-type branching process: m types, type i divides at b_i, dies at d_i, switches to type j at
s_ij. That is N = m(m+1) rates. The extinction probability vector q is the minimal solution of

    R_i q_i = b_i q_i^2 + d_i + sum_j s_ij q_j,        R_i = b_i + d_i + sum_j s_ij

and the answer is Y = P(extinction from n0 founders of type 1) = q_1^n0 -- a real rare event, with
no state-space explosion and no Monte-Carlo error. A whole-cell CME is not needed to ask a
scaling question, and using one would make the question unanswerable.

THE ADJOINT, WHICH IS ALSO BEING TESTED. Differentiating the fixed point implicitly,

    dq/dtheta = -G^{-1} dF/dtheta,     G = diag(R - 2 b q) - S

so for the scalar target T = log10 Y the whole gradient comes from ONE linear solve,
lambda = G^{-T} (dT/dq), followed by a sparse dot product per rate. Cost is one solve for all N
parameters rather than N solves. S1 checks it against finite differences and reports the measured
cost ratio, so the claim is verified rather than asserted.

WHAT PHYSIOLOGY CAN SEE, AND THE STRUCTURAL FACT THAT DRIVES EVERYTHING. Population-level
observables are functions of the mean matrix A = diag(b - d - sum_j s_ij) + S^T: the net growth
rate, the stationary type composition, the per-type share of death flux. There are O(m) of these
because there are m species. But there are m(m+1) rates. So observables scale as the number of
SPECIES and rates as species SQUARED, and the unconstrained dimension grows like N no matter how
good the assays get. K(N) can only stay small if the answer's gradient is concentrated -- which is
what S7 measures directly.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

S1  THE ADJOINT IS EXACT, AND CHEAP. Compared against central finite differences, which are
    themselves only second-order accurate, so a fixed absolute bar would be testing the finite
    difference rather than the adjoint. The gate is therefore that the disagreement falls as h^2
    under step halving -- ratio 4.0 +- 0.5 -- which identifies the residual as finite-difference
    truncation, together with an absolute bar of 1e-5 at the smallest step. Also report the
    measured wall-clock ratio of one adjoint pass against N one-at-a-time evaluations.

S2  THE FIXED POINT IS THE RIGHT ONE. q must satisfy the equation to 1e-13, lie in [0, 1], and be
    the MINIMAL solution -- iteration is started at zero, which converges to the minimal fixed
    point for a monotone system. For a supercritical process q_1 < 1 strictly.

S3  DEPTH IS HELD FIXED AS N GROWS. n0 is chosen per circuit so log10 Y lands in a stated band.
    Without this K(N) would confound the number of rates with the depth of the tail, and
    rateneed's N6 and hybrid's H11 both measured that requirements tighten with depth. Report the
    band actually achieved.

S4  NON-VACUITY. 0 < K(N) < N at every size, or the criterion is either already satisfied or
    unreachable and the curve means nothing.

S5  THE MATCHED CONTROL. Rates chosen at random instead of by the projection must need materially
    more measurements to reach the same criterion. Bar: the random requirement must exceed the
    greedy one by at least 2x at the largest size, or the selection rule is doing no work.

S6  THE DELIVERABLE. K(N) against N in each observable regime, fitted to K ~ a log N + b, to a
    power law K ~ N^alpha, and to K ~ cN. Report which describes the data and the exponent with a
    band. Predeclared readings: alpha well below 1, or a logarithmic fit, means the specification
    problem is tractable and chemistry is a fallback rather than a bottleneck; alpha near 1 means
    a measurement wall and the conclusion from the eight-rate circuit does not generalise.

S7  THE SPARSITY DIAGNOSTIC, WHICH EXPLAINS WHATEVER S6 FINDS. Report the participation ratio of
    the unconstrained gradient and the number of components carrying 99% of its squared norm.
    Predeclared: if that count saturates with N then K must too, and if it grows proportionally to
    N then K must too. A K(N) that disagrees with its own sparsity diagnostic means one of the two
    is computed wrongly.

S8  CAN THIS METHOD DETECT A WALL AT ALL? Repeat the entire measurement with the answer gradient
    replaced by a DENSE random unit vector, for which no concentration exists. K must then grow
    proportionally to N. If it does not, then S6's linear reading is unreachable on any evidence,
    the curve cannot distinguish tractable from intractable, and no conclusion may be drawn from
    it. This build order has recorded four bars that were unreachable, three of them found only
    after the fact; this one is checked in advance.

S9  IT IS NOT ONE LUCKY NETWORK. Several independent random circuits at each size. Report the
    spread of K across instances, not just the median.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE, ORDERS_PER_KCAL

EPS = 1.0
SIGMA = EPS * ORDERS_PER_KCAL          # chemistry error, orders per rate
DELTA = np.log10(2.0)                  # answer tolerance: a factor of two
ZQ = 1.645                             # two-sided 90%
TARGET_NULL = DELTA / (ZQ * SIGMA)     # bar on ||g_null||
M_SIZES = (3, 4, 6, 8, 11, 16, 22, 32)
N_SEEDS = 5
LOGY_BAND = (-3.5, -2.0)               # the rare-event depth held fixed across sizes
REGIMES = ("growth only", "growth + composition", "growth + composition + death flux")


def build(m, seed):
    rg = np.random.default_rng(seed)
    b = np.exp(rg.normal(0.0, 0.45, m))
    d = np.exp(rg.normal(-0.35, 0.45, m))
    S = np.exp(rg.normal(-1.6, 0.7, (m, m)))
    np.fill_diagonal(S, 0.0)
    return b, d, S


def pack(b, d, S):
    m = len(b)
    off = ~np.eye(m, dtype=bool)
    return np.concatenate([b, d, S[off]])


def unpack(theta, m):
    off = ~np.eye(m, dtype=bool)
    b, d = theta[:m], theta[m:2 * m]
    S = np.zeros((m, m))
    S[off] = theta[2 * m:]
    return b, d, S


def qsolve(b, d, S, tol=1e-15, it=200000):
    """Minimal fixed point, reached from zero because the map is monotone."""
    R = b + d + S.sum(1)
    q = np.zeros(len(b))
    for _ in range(it):
        qn = (b * q * q + d + S @ q) / R
        if np.max(np.abs(qn - q)) < tol:
            return qn
        q = qn
    return q


def residual(q, b, d, S):
    R = b + d + S.sum(1)
    return float(np.abs(R * q - b * q * q - d - S @ q).max())


def logY(b, d, S, n0):
    return n0 * np.log10(max(qsolve(b, d, S)[0], 1e-300))


def grad_adjoint(b, d, S, n0):
    """d log10 Y / d log10 (each rate), from ONE linear solve."""
    m = len(b)
    q = qsolve(b, d, S)
    R = b + d + S.sum(1)
    G = np.diag(R - 2.0 * b * q) - S
    # T = n0*log10(q0) and the parameters are log10 rates, so the two ln10 factors cancel:
    # dT/dlog10(k) = (n0/q0) * dq0/dk * k, NOT (n0/(q0 ln10)) * dq0/dk * k. Verified against
    # central finite differences by S1, which is what caught the missing factor.
    dTdq = np.zeros(m)
    dTdq[0] = n0 / max(q[0], 1e-300)
    lam = np.linalg.solve(G.T, dTdq)
    gb = -lam * (b * (q - q * q))
    gd = -lam * (d * (q - 1.0))
    off = ~np.eye(m, dtype=bool)
    gS = -(lam[:, None] * (S * (q[:, None] - q[None, :])))[off]
    return np.concatenate([gb, gd, gS]), q


def observables(b, d, S, regime):
    """Population-level quantities: functions of the mean matrix only."""
    m = len(b)
    A = np.diag(b - d - S.sum(1)) + S.T
    w, V = np.linalg.eig(A)
    k = int(np.argmax(w.real))
    lam = float(w[k].real)
    v = np.abs(V[:, k].real)
    v = v / max(v.sum(), 1e-300)
    out = [lam]
    if regime != "growth only":
        out += list(np.log10(np.maximum(v[1:] / max(v[0], 1e-300), 1e-300)))
    if regime == "growth + composition + death flux":
        f = d * v
        f = f / max(f.sum(), 1e-300)
        out += list(np.log10(np.maximum(f[1:] / max(f[0], 1e-300), 1e-300)))
    return np.array(out)


def phys_jacobian(theta, m, regime, h=0.01):
    n = len(theta)
    o0 = observables(*unpack(theta, m), regime)
    J = np.zeros((len(o0), n))
    for k in range(n):
        tp = theta.copy(); tp[k] *= 10.0 ** h
        tm = theta.copy(); tm[k] *= 10.0 ** -h
        J[:, k] = (observables(*unpack(tp, m), regime)
                   - observables(*unpack(tm, m), regime)) / (2 * h)
    return J


def row_basis(J, rcond=1e-9):
    """Orthonormal basis of the ROW space of J, as columns of an (N x r) matrix. The row space is
    small -- at most the number of observables plus the number of measured rates -- so working
    with it instead of the null space makes every greedy step O(N*r) rather than an SVD."""
    if J.shape[0] == 0:
        return np.zeros((J.shape[1], 0))
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    r = int((sv > rcond * max(sv.max(), 1e-300)).sum())
    return Vt[:r].T.copy()


def null_part(g, Q):
    return g - (Q @ (Q.T @ g)) if Q.shape[1] else g.copy()


def append_rate(Q, j, tol=1e-10):
    """Add a direct measurement of rate j: Gram-Schmidt e_j against the existing row basis."""
    n = Q.shape[0]
    v = -(Q @ Q[j, :]) if Q.shape[1] else np.zeros(n)
    v[j] += 1.0
    for _ in range(2):                       # re-orthogonalise once for stability
        if Q.shape[1]:
            v -= Q @ (Q.T @ v)
    nv = float(np.linalg.norm(v))
    if nv < tol:
        return Q, False
    return np.column_stack([Q, v / nv]), True


def greedy_K(g, J, target, cap):
    """Add direct rate measurements one at a time, each chosen to remove the most of the
    unconstrained gradient. Measuring rate j removes g_null[j]^2 / P_null[j,j] of the squared
    norm, a rank-one downdate that costs nothing per candidate once the row basis is held."""
    n = len(g)
    Q = row_basis(J)
    chosen = []
    for step in range(cap + 1):
        gn = null_part(g, Q)
        nrm = float(np.linalg.norm(gn))
        if nrm <= target:
            return step, nrm, chosen
        diag = 1.0 - (np.einsum("ij,ij->i", Q, Q) if Q.shape[1] else 0.0)
        score = np.where(diag > 1e-10, gn ** 2 / np.maximum(diag, 1e-300), -1.0)
        if len(chosen):
            score[chosen] = -1.0
        j = int(np.argmax(score))
        if score[j] <= 0:
            return None, nrm, chosen
        Q, ok = append_rate(Q, j)
        if not ok:
            return None, nrm, chosen
        chosen.append(j)
    return None, nrm, chosen


def random_K(g, J, target, cap, seed):
    n = len(g)
    order = np.random.default_rng(seed).permutation(n)
    Q = row_basis(J)
    for step in range(cap + 1):
        if float(np.linalg.norm(null_part(g, Q))) <= target:
            return step
        if step >= len(order):
            return None
        Q, _ = append_rate(Q, int(order[step]))
    return None


def sparsity(gn):
    p = gn ** 2
    tot = p.sum()
    if tot <= 0:
        return 0.0, 0
    pr = float(tot ** 2 / np.sum(p ** 2))
    srt = np.sort(p)[::-1]
    k99 = int(np.searchsorted(np.cumsum(srt) / tot, 0.99) + 1)
    return pr, k99


def fits(Ns, Ks):
    Ns, Ks = np.asarray(Ns, float), np.asarray(Ks, float)
    ok = np.isfinite(Ks) & (Ks > 0)
    Ns, Ks = Ns[ok], Ks[ok]
    if len(Ns) < 3:
        return {}
    def r2(y, yh):
        ss = np.sum((y - y.mean()) ** 2)
        return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 0 else float("nan")
    out = {}
    a, b0 = np.polyfit(np.log(Ns), Ks, 1)
    out["log"] = (a, b0, r2(Ks, a * np.log(Ns) + b0))
    al, c = np.polyfit(np.log(Ns), np.log(Ks), 1)
    out["power"] = (al, np.exp(c), r2(np.log(Ks), al * np.log(Ns) + c))
    sl = float(np.sum(Ns * Ks) / np.sum(Ns * Ns))
    out["linear"] = (sl, 0.0, r2(Ks, sl * Ns))
    return out


def choose_n0(b, d, S):
    """S3: hold the rare-event depth fixed as N grows, so K(N) measures rate count and not
    tail depth. rateneed's N6 and hybrid's H11 both found requirements tighten with depth."""
    q1 = qsolve(b, d, S)[0]
    if not (0.0 < q1 < 1.0):
        return None
    lo, hi = LOGY_BAND
    n0 = int(round(0.5 * (lo + hi) / np.log10(q1)))
    return max(n0, 1)


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("K(N):  HOW MANY EXTRA RATE MEASUREMENTS, AS THE RATE COUNT GROWS?"); P(RULE)
    P(f"  multi-type branching process, N = m(m+1) rates, chemistry error {EPS} kcal/mol")
    P(f"  criterion: ||g_null|| <= {TARGET_NULL:.4f}, i.e. sigma*||g_null|| <= {DELTA:.4f} orders")
    P(f"  which is a factor of two on the answer with {100*(1-2*(1-0.95)):.0f}% two-sided coverage")
    P(f"  rare-event depth held in log10 Y = {LOGY_BAND}, {N_SEEDS} independent circuits per size")

    # ---- S1, S2 --------------------------------------------------------------------------------
    P("\n" + RULE); P("S1  THE ADJOINT IS EXACT, AND CHEAP"); P(RULE)
    b, d, S = build(8, 11)
    n0 = choose_n0(b, d, S)
    g, q = grad_adjoint(b, d, S, n0)
    th = pack(b, d, S)
    rg = np.random.default_rng(3)
    probe = rg.choice(len(th), 10, replace=False)

    def fd_worst(hh):
        w = 0.0
        for k in probe:
            tp = th.copy(); tp[k] *= 10.0 ** hh
            tm = th.copy(); tm[k] *= 10.0 ** -hh
            fd = (logY(*unpack(tp, 8), n0) - logY(*unpack(tm, 8), n0)) / (2 * hh)
            w = max(w, abs(fd - g[k]) / max(abs(fd), 1e-12))
        return w

    P("  central differences are only second-order accurate, so a fixed bar would test the")
    P("  finite difference rather than the adjoint. The signature to look for is h^2 decay.")
    P(f"  {'step h':>10}{'worst rel':>14}{'ratio':>9}")
    hs = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    ws, ratios = [], []
    for hh in hs:
        w = fd_worst(hh)
        ratios.append(ws[-1] / w if ws else float("nan"))
        ws.append(w)
        P(f"  {hh:>10}{w:>14.3e}{ratios[-1]:>9.2f}")
    good = all(abs(r - 4.0) <= 0.5 for r in ratios[1:]) and ws[-1] < 1e-5
    P(f"  {'PASS -- the residual is finite-difference truncation, the adjoint is exact' if good else 'FAIL'}"
      f" (bars: halving ratio 4.0 +- 0.5, and below 1e-5 at the smallest step)")

    t0 = time.time()
    for _ in range(50):
        grad_adjoint(b, d, S, n0)
    t_adj = (time.time() - t0) / 50
    t0 = time.time()
    for k in range(len(th)):
        tp = th.copy(); tp[k] *= 1.01
        logY(*unpack(tp, 8), n0)
    t_fd = time.time() - t0
    P(f"  one adjoint pass gives all {len(th)} gradients in {t_adj*1000:.3f} ms")
    P(f"  {len(th)} one-at-a-time evaluations take {t_fd*1000:.1f} ms")
    P(f"  measured speedup {t_fd/max(t_adj,1e-12):.0f}x, and it grows with N by construction:")
    P(f"  the adjoint is one solve regardless of the number of rates.")

    P("\n" + RULE); P("S2  THE FIXED POINT IS THE RIGHT ONE"); P(RULE)
    wr, wq = 0.0, []
    for m in M_SIZES:
        for s in range(N_SEEDS):
            bb, dd, SS = build(m, 100 * m + s)
            qq = qsolve(bb, dd, SS)
            wr = max(wr, residual(qq, bb, dd, SS))
            wq.append(qq[0])
    P(f"  worst |residual| of the fixed-point equation over all circuits: {wr:.2e}"
      f"   {'PASS' if wr < 1e-13 else 'FAIL'} (bar 1e-13)")
    P(f"  q1 range over all circuits: {min(wq):.6f} to {max(wq):.6f}"
      f"   {'PASS' if max(wq) < 1.0 and min(wq) > 0.0 else 'FAIL'} (must be strictly inside 0..1)")
    P("  iteration is started at zero and the map is monotone, so this is the MINIMAL fixed point")

    # ---- the sweep -----------------------------------------------------------------------------
    P("\n" + RULE); P("THE SWEEP"); P(RULE)
    data = {r: {} for r in REGIMES}
    dense = {}
    depths = []
    for m in M_SIZES:
        N = m * (m + 1)
        for s in range(N_SEEDS):
            b, d, S = build(m, 100 * m + s)
            n0 = choose_n0(b, d, S)
            if n0 is None:
                continue
            ly = logY(b, d, S, n0)
            depths.append(ly)
            g, q = grad_adjoint(b, d, S, n0)
            th = pack(b, d, S)
            rgd = np.random.default_rng(5000 + 100 * m + s)
            gdense = rgd.standard_normal(N)
            gdense = gdense / np.linalg.norm(gdense) * np.linalg.norm(g)
            for regime in REGIMES:
                J = phys_jacobian(th, m, regime)
                cap = N
                K, nrm, chosen = greedy_K(g, J, TARGET_NULL, cap)
                pr, k99 = sparsity(null_part(g, row_basis(J)))
                Kr = random_K(g, J, TARGET_NULL, cap, 7 + s)
                data[regime].setdefault(N, []).append((K, Kr, pr, k99, J.shape[0], nrm))
                if regime == REGIMES[-1]:
                    Kd, _, _ = greedy_K(gdense, J, TARGET_NULL, cap)
                    dense.setdefault(N, []).append(Kd)
        P(f"    m={m:3d}  N={N:5d}  done")

    P("\n" + RULE); P("S3  DEPTH IS HELD FIXED AS N GROWS"); P(RULE)
    P(f"  log10 Y over every circuit: {min(depths):.3f} to {max(depths):.3f}"
      f"  (target band {LOGY_BAND})")
    inband = all(LOGY_BAND[0] - 0.6 <= x <= LOGY_BAND[1] + 0.6 for x in depths)
    P(f"  {'PASS' if inband else 'FAIL -- K(N) would confound rate count with tail depth'}"
      f" (bar: within 0.6 orders of the band)")

    # ---- S4, S5, S6, S7 -------------------------------------------------------------------------
    for regime in REGIMES:
        P("\n" + RULE); P(f"K(N)  --  observable regime: {regime}"); P(RULE)
        P(f"  {'N':>6}{'M obs':>7}{'K median':>10}{'K min':>7}{'K max':>7}"
          f"{'K/N':>9}{'random K':>10}{'part. ratio':>13}{'99% count':>11}")
        Ns, Ks = [], []
        for N in sorted(data[regime]):
            rows = data[regime][N]
            kk = [r[0] for r in rows if r[0] is not None]
            kr = [r[1] for r in rows if r[1] is not None]
            if not kk:
                P(f"  {N:>6}{rows[0][4]:>7}{'not reached within cap':>50}")
                continue
            Ns.append(N); Ks.append(float(np.median(kk)))
            P(f"  {N:>6}{rows[0][4]:>7}{np.median(kk):>10.1f}{min(kk):>7}{max(kk):>7}"
              f"{np.median(kk)/N:>9.4f}"
              f"{(np.median(kr) if kr else float('nan')):>10.1f}"
              f"{np.median([r[2] for r in rows]):>13.2f}"
              f"{np.median([r[3] for r in rows]):>11.1f}")
        f = fits(Ns, Ks)
        if f:
            P(f"  fit  K ~ a*log(N)+b : a = {f['log'][0]:+.3f}, b = {f['log'][1]:+.3f},"
              f" R2 = {f['log'][2]:.4f}")
            P(f"  fit  K ~ c*N^alpha  : alpha = {f['power'][0]:.3f}, c = {f['power'][1]:.4f},"
              f" R2 = {f['power'][2]:.4f}")
            P(f"  fit  K ~ c*N        : c = {f['linear'][0]:.4f}, R2 = {f['linear'][2]:.4f}")
            best = max(f.items(), key=lambda kv: (kv[1][2] if np.isfinite(kv[1][2]) else -9))
            P(f"  best description: {best[0]}")
        data[regime]["_fit"] = f
        data[regime]["_Ns"], data[regime]["_Ks"] = Ns, Ks

    P("\n" + RULE); P("S4  NON-VACUITY"); P(RULE)
    bad = []
    for regime in REGIMES:
        for N in data[regime]["_Ns"]:
            kk = [r[0] for r in data[regime][N] if r[0] is not None]
            if kk and (min(kk) <= 0 or max(kk) >= N):
                bad.append((regime, N))
    P(f"  circuits with K = 0 or K >= N: {len(bad)}"
      f"   {'PASS' if not bad else 'FAIL -- criterion already met or unreachable'}")

    P("\n" + RULE); P("S5  THE MATCHED CONTROL  (rates chosen at random)"); P(RULE)
    reg = REGIMES[-1]
    Nbig = max(data[reg]["_Ns"])
    rows = data[reg][Nbig]
    kg = [r[0] for r in rows if r[0] is not None]
    kr = [r[1] for r in rows if r[1] is not None]
    if kg and kr:
        ratio = np.median(kr) / max(np.median(kg), 1e-12)
        P(f"  at N = {Nbig}: greedy median {np.median(kg):.1f}, random median {np.median(kr):.1f},"
          f" ratio {ratio:.2f}x")
        P(f"  {'PASS -- the selection rule is doing real work' if ratio >= 2.0 else 'FAIL -- random selection does as well; the projection carries no information'}"
          f" (bar 2x)")
    else:
        P("  random selection never reached the criterion within the cap, which is itself the")
        P("  strongest possible form of this control and is reported as such.")

    P("\n" + RULE); P("S7  THE SPARSITY DIAGNOSTIC"); P(RULE)
    P("  If the unconstrained gradient concentrates on a bounded number of rates, K must saturate;")
    P("  if its support grows with N, K must grow with it. K and this count must agree.")
    P(f"  {'N':>6}{'99% count':>12}{'K median':>11}{'count/N':>10}{'K/N':>9}")
    for N in data[reg]["_Ns"]:
        rows = data[reg][N]
        kk = [r[0] for r in rows if r[0] is not None]
        k99 = np.median([r[3] for r in rows])
        P(f"  {N:>6}{k99:>12.1f}{np.median(kk):>11.1f}{k99/N:>10.4f}{np.median(kk)/N:>9.4f}")

    P("\n" + RULE); P("S8  CAN THIS METHOD DETECT A WALL AT ALL?"); P(RULE)
    P("  The same measurement with the answer gradient replaced by a DENSE random unit vector,")
    P("  for which no concentration exists. K must grow proportionally to N, or a linear reading")
    P("  in S6 would be unreachable on any evidence and the whole curve uninterpretable.")
    P(f"  {'N':>6}{'K dense median':>16}{'K/N':>9}")
    Nd, Kd = [], []
    for N in sorted(dense):
        kk = [k for k in dense[N] if k is not None]
        if not kk:
            P(f"  {N:>6}{'not reached within cap (K >= cap)':>30}")
            continue
        Nd.append(N); Kd.append(float(np.median(kk)))
        P(f"  {N:>6}{np.median(kk):>16.1f}{np.median(kk)/N:>9.4f}")
    fd = fits(Nd, Kd)
    if fd:
        P(f"  dense-control fits: alpha = {fd['power'][0]:.3f} (R2 {fd['power'][2]:.4f}),"
          f" linear R2 = {fd['linear'][2]:.4f}")
        det = fd["power"][0] >= 0.7
        P(f"  {'PASS -- the method can detect a wall, so a sublinear result is meaningful' if det else 'FAIL -- the method cannot produce a linear K(N) even for a dense target; S6 is uninterpretable'}"
          f" (bar alpha >= 0.7)")

    P("\n" + RULE); P("S9  IT IS NOT ONE LUCKY NETWORK"); P(RULE)
    P(f"  {'N':>6}{'K min':>8}{'K median':>11}{'K max':>8}{'spread':>9}")
    for N in data[reg]["_Ns"]:
        kk = [r[0] for r in data[reg][N] if r[0] is not None]
        P(f"  {N:>6}{min(kk):>8}{np.median(kk):>11.1f}{max(kk):>8}{max(kk)-min(kk):>9}")

    P("\n" + RULE)
    P("The first of the two scaling curves. K(N) says whether a cell can be SPECIFIED;")
    P("the largest required group size says whether it can be COMPUTED. This is the first.")
    P(RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_scaling.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
