"""Does K(N) survive a SPARSE network, where rates grow like species rather than species squared?

WHAT THIS TESTS, AND WHY IT WAS FLAGGED BEFORE IT WAS RUN. scaling.py measured K(N) flat to
logarithmic across 12 to 1056 rates, and its post-run reading recorded a caveat that the result
might not survive: switching rates there were drawn for all m(m-1) ordered pairs, so the network
gets DENSER as it grows and per-rate sensitivity dilutes by construction. The diagnostic added
after that run measured the dilution directly -- ||g|| ~ N^-0.304 -- so larger circuits started
12.9 times above the criterion at N = 12 and only 4.4 times above it at N = 1056. Part of the flat
K was a closer starting point, not a better selection rule.

Real networks are sparse. This module builds the sparse counterpart and runs the dense family
beside it on identical criteria, so the comparison is matched rather than remembered.

THE STRUCTURAL DIFFERENCE, WHICH IS THE WHOLE POINT.

    dense   N = m(m+1) ~ m^2   observables ~ m   =>  M/N ~ 1/sqrt(N)  ->  0
    sparse  N = m(2+c) ~ m     observables ~ m   =>  M/N ~ 2/(2+c)    ->  constant

In the dense family the fraction of rate-space that physiology can see vanishes as the network
grows. In the sparse family it does not. If K(N) stays bounded in the dense case only because
sensitivity dilutes, the sparse case should expose it: there, the unconstrained dimension grows
like N and a dense answer-gradient would need K ~ (1 - 2/(2+c)) N measurements.

THE CONSTRUCTION. Each type switches to exactly c others: one edge along a fixed cycle, which
guarantees the network is strongly connected, plus c-1 drawn at random. Without the cycle a random
c-out digraph need not be irreducible, and a reducible network is several smaller extinction
problems rather than one large one -- an easier question wearing the same name. P1 checks it.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

P1  THE CIRCUIT IS SPARSE AND IRREDUCIBLE. Exactly c outgoing switches per type, N = m(2+c)
    rates, and the switch graph strongly connected at every size. A reducible network decouples
    into independent sub-problems and would answer an easier question under the same name.

P2  THE ADJOINT IS EXACT ON THIS FAMILY TOO. Same test as scaling.py's S1: the finite-difference
    disagreement must fall as h^2 under halving, ratio 4.0 +- 0.5, and be below 1e-5 at the
    smallest step. The adjoint was derived for the dense family and is being reused; that it
    remains correct is a check, not an assumption.

P3  DEPTH HELD FIXED. Same band as scaling.py, so K(N) here measures rate count rather than tail
    depth and is comparable with the dense curve.

P4  THE STRUCTURAL RATIO IS WHAT IT CLAIMS. Report M/N against N for both families. Predeclared:
    dense must fall towards zero and sparse must approach 2/(2+c). If sparse does not stay
    constant, the two families are not the contrast this module is built on.

P5  THE DILUTION DIAGNOSTIC, PREDECLARED THIS TIME. ||g|| and the residual before any measurement,
    against N, fitted to a power law, for both families. This was added after the fact last time
    and it changed the reading, so it is a gate here. Predeclared: if the sparse family shows no
    dilution -- exponent near zero -- while the dense family shows N^-0.30, then the dense result
    was substantially a dilution artefact and the sparse curve is the one to believe.

P6  THE DELIVERABLE. K(N) for the sparse family, fitted to logarithmic, power-law and linear
    forms, with the dense curve computed in the same run for a matched comparison. Predeclared
    readings: an exponent well below 1 means the specification problem is tractable for sparse
    networks and the earlier conclusion generalises; an exponent near 1 means the earlier result
    was a property of dense coupling and does not.

P7  THE DETECTION CONTROL. The same measurement on the sparse family with the gradient replaced
    by a dense random vector must give K growing proportionally to N. Without it a sublinear
    sparse result is unfalsifiable, exactly as it would have been for the dense family.

P8  THE MATCHED CONTROL. Rates chosen at random rather than by projection must need materially
    more. Bar: 2x at the largest size.

P9  IT IS NOT ONE CONNECTIVITY. Repeat at c = 2, 3 and 5. If the conclusion depends on c, that
    dependence is the result and must be reported rather than a single c quoted.

P10 IT IS NOT ONE NETWORK. Several independent instances per size; report the spread of K.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.scaling import (
    SIGMA, TARGET_NULL, LOGY_BAND, REGIMES,
    build as build_dense, qsolve, residual, logY, grad_adjoint, observables,
    row_basis, null_part, greedy_K, random_K, sparsity, fits, choose_n0,
)

C_VALUES = (2, 3, 5)
C_MAIN = 3
M_SPARSE = (4, 6, 10, 16, 25, 40, 64, 100, 160)
M_DENSE = (3, 4, 6, 8, 11, 16, 22, 32)
N_SEEDS = 5
REGIME = REGIMES[-1]


def build_sparse(m, c, seed):
    """c outgoing switches per type: one along a fixed cycle so the graph is strongly connected,
    plus c-1 drawn at random from the remaining types."""
    rg = np.random.default_rng(seed)
    b = np.exp(rg.normal(0.0, 0.45, m))
    d = np.exp(rg.normal(-0.35, 0.45, m))
    S = np.zeros((m, m))
    targets = []
    for i in range(m):
        outs = [(i + 1) % m]
        pool = [j for j in range(m) if j != i and j not in outs]
        rg.shuffle(pool)
        outs += pool[:max(c - 1, 0)]
        targets.append(sorted(outs))
        for j in outs:
            S[i, j] = np.exp(rg.normal(-1.6, 0.7))
    return b, d, S, targets


def pack_sparse(b, d, S, targets):
    vals = [S[i, j] for i in range(len(b)) for j in targets[i]]
    return np.concatenate([b, d, np.array(vals)])


def unpack_sparse(theta, m, targets):
    b, d = theta[:m], theta[m:2 * m]
    S = np.zeros((m, m))
    k = 2 * m
    for i in range(m):
        for j in targets[i]:
            S[i, j] = theta[k]; k += 1
    return b, d, S


def grad_sparse(b, d, S, targets, n0):
    """Same adjoint, restricted to the rates that exist. One linear solve for all of them."""
    m = len(b)
    q = qsolve(b, d, S)
    R = b + d + S.sum(1)
    G = np.diag(R - 2.0 * b * q) - S
    dTdq = np.zeros(m)
    dTdq[0] = n0 / max(q[0], 1e-300)
    lam = np.linalg.solve(G.T, dTdq)
    gb = -lam * (b * (q - q * q))
    gd = -lam * (d * (q - 1.0))
    gs = [-lam[i] * (S[i, j] * (q[i] - q[j])) for i in range(m) for j in targets[i]]
    return np.concatenate([gb, gd, np.array(gs)]), q


def phys_jac_sparse(theta, m, targets, regime, h=0.01):
    o0 = observables(*unpack_sparse(theta, m, targets), regime)
    J = np.zeros((len(o0), len(theta)))
    for k in range(len(theta)):
        tp = theta.copy(); tp[k] *= 10.0 ** h
        tm = theta.copy(); tm[k] *= 10.0 ** -h
        J[:, k] = (observables(*unpack_sparse(tp, m, targets), regime)
                   - observables(*unpack_sparse(tm, m, targets), regime)) / (2 * h)
    return J


def strongly_connected(targets):
    m = len(targets)
    def reach(adj):
        seen = {0}; stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        return len(seen) == m
    rev = [[] for _ in range(m)]
    for i, outs in enumerate(targets):
        for j in outs:
            rev[j].append(i)
    return reach(targets) and reach(rev)


M_BIG = 160          # one larger size, fewer seeds, because the Jacobian costs O(m^4)
SEEDS_BIG = 3
M_CSWEEP = (6, 16, 40, 100)
SEEDS_CSWEEP = 3


def sparse_point(m, c, seed, want_dense_grad=False):
    b, d, S, tg = build_sparse(m, c, seed)
    if not strongly_connected(tg):
        return None
    n0 = choose_n0(b, d, S)
    if n0 is None:
        return None
    th = pack_sparse(b, d, S, tg)
    g, q = grad_sparse(b, d, S, tg, n0)
    J = phys_jac_sparse(th, m, tg, REGIME)
    N = len(th)
    K, nrm, _ = greedy_K(g, J, TARGET_NULL, N)
    Kr = random_K(g, J, TARGET_NULL, N, 17 + seed)
    Q = row_basis(J)
    gn = null_part(g, Q)
    pr, k99 = sparsity(gn)
    out = dict(N=N, M=J.shape[0], K=K, Kr=Kr, gnorm=float(np.linalg.norm(g)),
               gnull=float(np.linalg.norm(gn)), k99=k99, pr=pr,
               logY=logY(b, d, S, n0), res=residual(q, b, d, S), scc=True)
    if want_dense_grad:
        rgd = np.random.default_rng(9000 + seed)
        gd = rgd.standard_normal(N)
        gd = gd / np.linalg.norm(gd) * np.linalg.norm(g)
        out["Kd"] = greedy_K(gd, J, TARGET_NULL, N)[0]
    return out


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("DOES K(N) SURVIVE A SPARSE NETWORK?"); P(RULE)
    P(f"  sparse: c outgoing switches per type, N = m(2+c); dense: N = m(m+1), for comparison")
    P(f"  same criterion as scaling.py: ||g_null|| <= {TARGET_NULL:.4f}, depth band {LOGY_BAND}")
    P(f"  observable regime: {REGIME}")

    # ---- P2 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("P2  THE ADJOINT IS EXACT ON THIS FAMILY TOO"); P(RULE)
    b, d, S, tg = build_sparse(16, C_MAIN, 5)
    n0 = choose_n0(b, d, S)
    g, _ = grad_sparse(b, d, S, tg, n0)
    th = pack_sparse(b, d, S, tg)
    probe = np.random.default_rng(2).choice(len(th), 10, replace=False)
    P(f"  {'step h':>10}{'worst rel':>14}{'ratio':>9}")
    ws, ratios = [], []
    for hh in (0.02, 0.01, 0.005, 0.0025, 0.00125):
        w = 0.0
        for k in probe:
            tp = th.copy(); tp[k] *= 10.0 ** hh
            tm = th.copy(); tm[k] *= 10.0 ** -hh
            fd = (logY(*unpack_sparse(tp, 16, tg), n0)
                  - logY(*unpack_sparse(tm, 16, tg), n0)) / (2 * hh)
            w = max(w, abs(fd - g[k]) / max(abs(fd), 1e-12))
        ratios.append(ws[-1] / w if ws else float("nan"))
        ws.append(w)
        P(f"  {hh:>10}{w:>14.3e}{ratios[-1]:>9.2f}")
    ok2 = all(abs(r - 4.0) <= 0.5 for r in ratios[1:]) and ws[-1] < 1e-5
    P(f"  {'PASS' if ok2 else 'FAIL'} (halving ratio 4.0 +- 0.5, below 1e-5 at the smallest step)")

    # ---- the sparse sweep -------------------------------------------------------------------------
    P("\n" + RULE); P(f"THE SPARSE SWEEP  (c = {C_MAIN})"); P(RULE)
    rows = {}
    for m in M_SPARSE + (M_BIG,):
        ns = SEEDS_BIG if m == M_BIG else N_SEEDS
        got = [sparse_point(m, C_MAIN, 100 * m + s, want_dense_grad=True) for s in range(ns)]
        got = [r for r in got if r]
        if got:
            rows[got[0]["N"]] = got
        P(f"    m={m:4d}  N={got[0]['N'] if got else '?':>5}  {len(got)} circuits")

    # ---- P1 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("P1  THE CIRCUIT IS SPARSE AND IRREDUCIBLE"); P(RULE)
    allscc = True
    for m in M_SPARSE + (M_BIG,):
        for s in range(2):
            _, _, Ss, tgs = build_sparse(m, C_MAIN, 100 * m + s)
            allscc &= strongly_connected(tgs)
            allscc &= all(len(t) == C_MAIN for t in tgs)
            allscc &= int((Ss > 0).sum()) == C_MAIN * m
    P(f"  every type has exactly {C_MAIN} outgoing switches, switch matrix has exactly c*m nonzeros,")
    P(f"  and the switch graph is strongly connected at every size: {allscc}"
      f"   {'PASS' if allscc else 'FAIL -- a reducible network is several easier problems'}")

    # ---- P3 -------------------------------------------------------------------------------------
    dep = [r["logY"] for v in rows.values() for r in v]
    res = max(r["res"] for v in rows.values() for r in v)
    P("\n" + RULE); P("P3  DEPTH HELD FIXED, AND THE FIXED POINT IS THE RIGHT ONE"); P(RULE)
    P(f"  log10 Y over every sparse circuit: {min(dep):.3f} to {max(dep):.3f} (band {LOGY_BAND})")
    inb = all(LOGY_BAND[0] - 0.6 <= x <= LOGY_BAND[1] + 0.6 for x in dep)
    P(f"  {'PASS' if inb else 'FAIL'} (within 0.6 orders of the band)")
    P(f"  worst fixed-point residual {res:.2e}   {'PASS' if res < 1e-13 else 'FAIL'} (bar 1e-13)")

    # ---- dense family, same run, for a matched comparison ---------------------------------------
    from rem.atlas.scaling import pack as pack_dense, phys_jacobian as phys_jac_dense
    drows = {}
    for m in M_DENSE:
        acc = []
        for s in range(N_SEEDS):
            bb, dd, SS = build_dense(m, 100 * m + s)
            nn0 = choose_n0(bb, dd, SS)
            if nn0 is None:
                continue
            gg, _ = grad_adjoint(bb, dd, SS, nn0)
            tt = pack_dense(bb, dd, SS)
            JJ = phys_jac_dense(tt, m, REGIME)
            NN = len(tt)
            KK, _, _ = greedy_K(gg, JJ, TARGET_NULL, NN)
            gnn = null_part(gg, row_basis(JJ))
            acc.append(dict(N=NN, M=JJ.shape[0], K=KK, gnorm=float(np.linalg.norm(gg)),
                            gnull=float(np.linalg.norm(gnn))))
        if acc:
            drows[acc[0]["N"]] = acc

    # ---- P4 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("P4  THE STRUCTURAL RATIO IS WHAT IT CLAIMS"); P(RULE)
    P(f"  {'sparse N':>10}{'M':>6}{'M/N':>9}      {'dense N':>9}{'M':>6}{'M/N':>9}")
    sN, dN = sorted(rows), sorted(drows)
    for i in range(max(len(sN), len(dN))):
        a = f"{sN[i]:>10}{rows[sN[i]][0]['M']:>6}{rows[sN[i]][0]['M']/sN[i]:>9.4f}" if i < len(sN) else " " * 25
        bb = f"{dN[i]:>9}{drows[dN[i]][0]['M']:>6}{drows[dN[i]][0]['M']/dN[i]:>9.4f}" if i < len(dN) else ""
        P(f"  {a}      {bb}")
    sr = [rows[n][0]["M"] / n for n in sN]
    dr = [drows[n][0]["M"] / n for n in dN]
    okp4 = (max(sr) - min(sr) < 0.05) and (dr[-1] < 0.5 * dr[0])
    P(f"  sparse M/N spans {min(sr):.4f} to {max(sr):.4f}, expected 2/(2+c) = {2/(2+C_MAIN):.4f}")
    P(f"  dense M/N falls {dr[0]:.4f} to {dr[-1]:.4f}")
    P(f"  {'PASS -- the two families are the contrast this module is built on' if okp4 else 'FAIL'}")

    # ---- P5 -------------------------------------------------------------------------------------
    P("\n" + RULE); P("P5  THE DILUTION DIAGNOSTIC, PREDECLARED THIS TIME"); P(RULE)
    P(f"  {'family':>8}{'N':>7}{'||g||':>10}{'||g_null||':>12}{'ratio to target':>17}")
    for lbl, R in (("sparse", rows), ("dense", drows)):
        for n in sorted(R):
            gm = np.median([r["gnorm"] for r in R[n]])
            gn = np.median([r["gnull"] for r in R[n]])
            P(f"  {lbl:>8}{n:>7}{gm:>10.3f}{gn:>12.3f}{gn/TARGET_NULL:>17.2f}")
    exps = {}
    for lbl, R in (("sparse", rows), ("dense", drows)):
        ns = np.array(sorted(R), float)
        gm = np.array([np.median([r["gnorm"] for r in R[n]]) for n in sorted(R)])
        gn = np.array([np.median([r["gnull"] for r in R[n]]) for n in sorted(R)])
        exps[lbl] = (np.polyfit(np.log(ns), np.log(gm), 1)[0],
                     np.polyfit(np.log(ns), np.log(gn), 1)[0])
        P(f"  {lbl}: ||g|| ~ N^{exps[lbl][0]:+.3f},  ||g_null|| ~ N^{exps[lbl][1]:+.3f}")
    P("  PREDECLARED READING: if sparse shows no dilution while dense shows about N^-0.30, the")
    P("  dense result was substantially a dilution artefact and the sparse curve is the one to trust.")
    if abs(exps["sparse"][0]) < 0.5 * abs(exps["dense"][0]):
        P("  -> sparse dilutes materially LESS than dense. The dense curve was flattered by dilution.")
    else:
        P("  -> both families dilute comparably. Dilution does not separate them.")

    # ---- P6, P8, P10 ------------------------------------------------------------------------------
    P("\n" + RULE); P(f"P6  THE DELIVERABLE  --  K(N) for the sparse family (c = {C_MAIN})"); P(RULE)
    P(f"  {'N':>6}{'K median':>10}{'K min':>7}{'K max':>7}{'K/N':>9}{'random K':>10}"
      f"{'99% count':>11}")
    Ns, Ks = [], []
    for n in sN:
        kk = [r["K"] for r in rows[n] if r["K"] is not None]
        kr = [r["Kr"] for r in rows[n] if r["Kr"] is not None]
        if not kk:
            continue
        Ns.append(n); Ks.append(float(np.median(kk)))
        P(f"  {n:>6}{np.median(kk):>10.1f}{min(kk):>7}{max(kk):>7}{np.median(kk)/n:>9.4f}"
          f"{(np.median(kr) if kr else float('nan')):>10.1f}"
          f"{np.median([r['k99'] for r in rows[n]]):>11.1f}")
    fs = fits(Ns, Ks)
    P(f"  fit  K ~ a*log(N)+b : a = {fs['log'][0]:+.3f}, R2 = {fs['log'][2]:.4f}")
    P(f"  fit  K ~ c*N^alpha  : alpha = {fs['power'][0]:.3f}, R2 = {fs['power'][2]:.4f}")
    P(f"  fit  K ~ c*N        : c = {fs['linear'][0]:.4f}, R2 = {fs['linear'][2]:.4f}")
    P(f"  best description: {max(fs.items(), key=lambda kv: kv[1][2])[0]}")

    dKs = [float(np.median([r['K'] for r in drows[n] if r['K'] is not None])) for n in dN]
    fdd = fits(dN, dKs)
    P(f"\n  dense family in the same run, for comparison:")
    P(f"  {'N':>6}{'K median':>10}{'K/N':>9}")
    for n, k in zip(dN, dKs):
        P(f"  {n:>6}{k:>10.1f}{k/n:>9.4f}")
    P(f"  dense: alpha = {fdd['power'][0]:.3f} (R2 {fdd['power'][2]:.4f}),"
      f" log R2 = {fdd['log'][2]:.4f}")

    P("\n" + RULE); P("P8  THE MATCHED CONTROL"); P(RULE)
    nbig = Ns[-1]
    kg = np.median([r["K"] for r in rows[nbig] if r["K"] is not None])
    kr = [r["Kr"] for r in rows[nbig] if r["Kr"] is not None]
    if kr:
        ratio = np.median(kr) / max(kg, 1e-12)
        P(f"  at N = {nbig}: greedy {kg:.1f}, random {np.median(kr):.1f}, ratio {ratio:.2f}x"
          f"   {'PASS' if ratio >= 2.0 else 'FAIL'} (bar 2x)")
    else:
        P(f"  at N = {nbig}: random selection never reached the criterion; greedy needed {kg:.1f}")

    P("\n" + RULE); P("P7  THE DETECTION CONTROL ON THIS FAMILY"); P(RULE)
    P(f"  {'N':>6}{'K dense-gradient':>18}{'K/N':>9}")
    Nd, Kd = [], []
    for n in sN:
        kk = [r["Kd"] for r in rows[n] if r.get("Kd") is not None]
        if not kk:
            continue
        Nd.append(n); Kd.append(float(np.median(kk)))
        P(f"  {n:>6}{np.median(kk):>18.1f}{np.median(kk)/n:>9.4f}")
    fdg = fits(Nd, Kd)
    P(f"  alpha = {fdg['power'][0]:.3f} (R2 {fdg['power'][2]:.4f}), linear R2 = {fdg['linear'][2]:.4f}")
    P(f"  {'PASS -- a sublinear sparse result is falsifiable' if fdg['power'][0] >= 0.7 else 'FAIL -- the method cannot produce a linear K(N) here; P6 is uninterpretable'}"
      f" (bar alpha >= 0.7)")

    P("\n" + RULE); P("P9  IT IS NOT ONE CONNECTIVITY"); P(RULE)
    P(f"  {'c':>4}{'M/N':>8}" + "".join(f"{'N='+str(m*(2+c)):>12}" for m, c in zip(M_CSWEEP, [C_MAIN]*4)))
    for c in C_VALUES:
        cells, ns, ks = [], [], []
        for m in M_CSWEEP:
            got = [sparse_point(m, c, 700 * m + s) for s in range(SEEDS_CSWEEP)]
            got = [r for r in got if r and r["K"] is not None]
            if not got:
                cells.append("--"); continue
            ns.append(got[0]["N"]); ks.append(float(np.median([r["K"] for r in got])))
            cells.append(f"N={got[0]['N']} K={np.median([r['K'] for r in got]):.0f}")
        fc = fits(ns, ks)
        al = fc["power"][0] if fc else float("nan")
        P(f"  {c:>4}{2/(2+c):>8.3f}" + "".join(f"{x:>12}" for x in cells) + f"   alpha = {al:.3f}")

    P("\n" + RULE); P("P10  IT IS NOT ONE NETWORK"); P(RULE)
    P(f"  {'N':>6}{'K min':>8}{'K median':>11}{'K max':>8}{'spread':>9}")
    for n in sN:
        kk = [r["K"] for r in rows[n] if r["K"] is not None]
        if kk:
            P(f"  {n:>6}{min(kk):>8}{np.median(kk):>11.1f}{max(kk):>8}{max(kk)-min(kk):>9}")

    P("\n" + RULE)
    open(os.path.join(os.path.dirname(__file__), "RESULTS_sparse.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
