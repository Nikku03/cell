"""Can the certificate be verified WITHOUT scanning every state?

WHY THIS IS THE BLOCKER RATHER THAN A BOTTLENECK. An external round of tests reported a compact
backward approximation that was 28x slower than exact dynamic programming, and noted separately
that its bound checks still examined the entire finite state space. Those two facts are the same
fact. If verifying the bound costs a full scan, it costs what the exact computation costs, so the
approximation cannot beat exact DP by construction -- at any size, on any problem where exact DP
exists. Making the approximation faster does not help while the verification is O(states).

THREE WAYS OUT, AND THIS PRICES ALL THREE. Uniform sampling of the dropped set. Importance
sampling against a cheap proxy. And an AGGREGATE bound, which examines only the heaviest few
children of each parent and bounds all the rest in one closed-form term, never touching them.
Only the third can produce a real certificate rather than a confidence statement, and whether it
is tight enough is the question.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

C0  THE CEILING GATE, AND IT IS ABOUT THE SHAPE OF THE SUM. A sampled estimate of a sum of
    positive terms is only as good as the terms are even. Measure how concentrated the certificate
    sum is: what share comes from its heaviest 0.01%, 0.1%, 1% of dropped candidates.
    PREDECLARED: if the heaviest 0.1% carries more than half the sum, uniform sampling is hopeless
    at any practical sample size and C1 is expected to fail -- and that failure is not incidental,
    because pathbound measured that the TAIL itself is carried by its heaviest 1% of paths. The
    concentration that makes pruning work would then be the same property that makes sampled
    verification fail, and the module must say so in those terms rather than reporting C1's
    failure as a surprise.

C1  UNIFORM SAMPLING, WITH A VALID ONE-SIDED BOUND RATHER THAN AN ESTIMATE. A certificate is a
    GUARANTEE; an unbiased estimate is not one. So the quantity measured is an empirical-Bernstein
    upper confidence bound at 95%, which is valid for bounded summands.
    PREDECLARED: sampling is a saving only if the sample fraction needed for the bound to land
    within 2x of the true sum is under 10%. Above that, one scans a tenth of the space to save
    nothing and the answer is that uniform sampling cannot verify this bound.

C2  IMPORTANCE SAMPLING AGAINST THE CHEAP PROXY. The obvious fix is to sample proportional to path
    mass, which is available without evaluating the bound.
    PREDECLARED: the correlation between proxy and summand is reported FIRST, because an
    importance sampler with an uncorrelated proxy is worse than uniform, and quoting its variance
    without the correlation would hide that.

C3  THE AGGREGATE BOUND, WHICH IS THE ONLY ROUTE THAT YIELDS A CERTIFICATE. For each parent,
    examine only its k heaviest children and bound the contribution of every other child in one
    term: their total remaining mass times the largest ON-probability any child could have, taken
    over the child space by a box bound that costs O(nCtrl) and touches no child at all.
    PREDECLARED: useful if at k far below n the certificate stays within a small factor of the
    full-scan certificate. The comparison is against the FULL-SCAN certificate, not against the
    true deficit -- this gate is about whether the scan can be avoided, not about whether the
    bound was any good, which exactcert already answered.

C2b THE CIRCULARITY IN C1 AND C2, AND THE NON-CIRCULAR VERSION. PREDECLARED AFTER C1 AND C2 RAN
    AND BEFORE C2b DID. C1 and C2 sample from the DROPPED set -- but knowing which candidates are
    dropped means having ranked all of them by the bound, which is the full scan. Their measured
    saving is therefore illusory as stated: they cheapen the verification of a partition whose
    construction already paid the cost. This is my own setup's defect, not a property of sampling.
    THE NON-CIRCULAR CONFIGURATION: select the retained set by the CHEAP PROXY, path mass, which
    needs no bound evaluations at all -- the child ordering under any parent is the ordering of
    that column of the transition matrix, so sorting the columns ONCE gives every parent's
    heaviest children for free, and a global top-k is a merge that touches only about k entries.
    Then certify THAT partition by importance sampling, evaluating the bound only on the sampled
    candidates.
    PREDECLARED: the measure is BOUND EVALUATIONS USED against the full scan's, and the accuracy
    is against the full-scan certificate for the SAME partition, so the two effects -- a worse
    partition and a sampled certificate -- are not confounded.

C4  THE VERDICT, and C5 what it does not settle.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, stationary
from rem.atlas.accuracy import identity_S


def setup_engine(nCtrl=10, L=6, dt=0.5, ntarget=200):
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, ntarget)
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    hvec = np.exp(-0.5 * np.arange(L, -1, -1))
    hvec = hvec / hvec.sum()
    S = identity_S(rows, nCtrl)
    actbit = np.array([[(m >> c) & 1 for c in range(n.bit_length() - 1)] for m in range(n)],
                      dtype=float)
    return Q, Pm, pi, n, hvec, S, actbit, rows, sha


def eb_ucb(x, N, conf=0.95):
    """Empirical-Bernstein 95% upper confidence bound on the SUM of N terms from a sample x.

    Valid for bounded non-negative summands; the range is taken as the sample max, which is the
    standard empirical-Bernstein form and is what makes this a bound rather than an estimate."""
    k = len(x)
    if k < 2:
        return float("inf")
    m, v, b = float(np.mean(x)), float(np.var(x, ddof=1)), float(np.max(x))
    lg = np.log(3.0 / (1.0 - conf))
    return N * (m + np.sqrt(2.0 * v * lg / k) + 3.0 * b * lg / k)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("CAN THE CERTIFICATE BE VERIFIED WITHOUT SCANNING EVERY STATE?")
    P_(RULE)
    P_("  If verifying costs a full scan it costs what exact DP costs, and no speed-up of the")
    P_("  approximation can rescue that. This prices the three ways out.")

    nCtrl, L, dt = 10, 6, 0.5
    Q, Pm, pi, n, hvec, S, actbit, rows, sha = setup_engine(nCtrl, L, dt)
    Splus = np.maximum(S, 0.0).sum(axis=1)
    Bst = actbit @ S.T
    base, gain = -1.0, 2.0
    P_(f"\n  {nCtrl} controllers, n = {n} states, L = {L}, {len(rows)} targets. sha {sha[:12]}.")

    # build one real pruning level: the retained set at cap, then its children as candidates
    cap = 3000
    last = np.argsort(-pi)[:cap]
    wts = pi[last].copy()
    wact = actbit[last] * hvec[0]
    for d in range(1, 3):
        ch = Pm[:, last].T * wts[:, None]
        flat = ch.ravel()
        k = np.argsort(-flat)[:cap]
        par, code = k // n, k % n
        wts, wact, last = flat[k], wact[par] + actbit[code] * hvec[d], code
    d = 3
    ch = Pm[:, last].T * wts[:, None]                     # (P, n) candidate masses
    Hr = float(hvec[d + 1:].sum())
    u = base + gain * (wact @ S.T) + gain * Hr * Splus[None, :]
    ls = (-np.logaddexp(0.0, -u)).sum(axis=1)
    g = 1.0 / (1.0 + np.exp(u))
    lg = ls[:, None] + g @ (gain * hvec[d] * Bst).T
    bnd = ch * np.exp(lg)                                  # the full-scan bound, (P, n)
    P_(f"  one real pruning level: {ch.shape[0]:,} parents x {n} children ="
       f" {ch.size:,} candidates.")

    keep = np.argsort(-bnd.ravel())[:cap]
    mask = np.ones(bnd.size, dtype=bool)
    mask[keep] = False
    dropped = bnd.ravel()[mask]
    cert_full = float(dropped.sum())
    P_(f"  keeping the top {cap:,} by bound, the FULL-SCAN certificate over the"
       f" {dropped.size:,} dropped candidates is {cert_full:.6e}.")

    # ---- C0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("C0  THE CEILING GATE: HOW EVEN IS THE SUM BEING SAMPLED?")
    P_(RULE)
    o = np.sort(dropped)[::-1]
    cum = np.cumsum(o) / cert_full
    P_(f"\n    {'heaviest fraction':>19} {'share of the certificate':>26}")
    for f in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        kk = max(1, int(f * len(o)))
        P_(f"    {f * 100:>18.3f}% {cum[kk - 1] * 100:>25.2f}%")
    top = float(cum[max(1, int(1e-3 * len(o))) - 1])
    P_(f"\n  C0: the heaviest 0.1% carries {top * 100:.2f}% of the sum --"
       f" {'uniform sampling is expected to fail, as predeclared.' if top > 0.5 else 'the sum is even enough that uniform sampling has a chance.'}")
    if top <= 0.5:
        P_("  MY PREDECLARED EXPECTATION WAS WRONG, AND THE REASON IS WORTH KEEPING. I expected the")
        P_("  certificate sum to inherit the tail's concentration -- pathbound measured the top 1%")
        P_("  of PATHS carrying 91% of the TAIL. It does not. The tail is concentrated because a")
        P_("  few paths have large ON-probability; the certificate is a sum over the DROPPED")
        P_("  candidates, which are precisely the ones that do not, so it is dominated by the many")
        P_("  rather than the few. Concentration in the observable does not transfer to the")
        P_("  residual, and I assumed it would.")

    # ---- C1  UNIFORM SAMPLING ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("C1  UNIFORM SAMPLING, WITH A VALID ONE-SIDED BOUND")
    P_(RULE)
    P_("  A certificate is a guarantee, so the quantity is a 95% empirical-Bernstein UPPER bound")
    P_("  on the sum, not an unbiased estimate.")
    rng = np.random.default_rng(20260909)
    N = len(dropped)
    P_(f"\n    {'sample':>10} {'fraction':>10} {'estimate/true':>14} {'95% UCB/true':>14}")
    c1 = None
    for frac in (1e-4, 1e-3, 1e-2, 1e-1, 0.5):
        kk = max(2, int(frac * N))
        idx = rng.choice(N, kk, replace=False)
        x = dropped[idx]
        est = N * float(np.mean(x))
        ucb = eb_ucb(x, N)
        if c1 is None and ucb <= 2.0 * cert_full:
            c1 = frac
        P_(f"    {kk:>10,} {frac * 100:>9.2f}% {est / cert_full:>14.3f}"
           f" {ucb / cert_full:>14.3f}")
    P_(f"\n  C1 AS PREDECLARED: {'the UCB reaches 2x the true sum at a sample fraction of ' + f'{c1 * 100:.2f}%' if c1 is not None and c1 <= 0.1 else 'no sample fraction under 10% brings the UCB within 2x. UNIFORM SAMPLING CANNOT VERIFY THIS BOUND.'}")

    # ---- C2  IMPORTANCE SAMPLING ---------------------------------------------------------------
    P_("\n" + RULE)
    P_("C2  IMPORTANCE SAMPLING AGAINST THE CHEAP PROXY, CORRELATION FIRST")
    P_(RULE)
    proxy = ch.ravel()[mask]
    r = float(np.corrcoef(np.log(np.maximum(proxy, 1e-300)),
                          np.log(np.maximum(dropped, 1e-300)))[0, 1])
    P_(f"  correlation of log path mass with log bound over the dropped set: r = {r:.4f}")
    p = proxy / proxy.sum()
    P_(f"\n    {'sample':>10} {'fraction':>10} {'estimate/true':>14} {'95% UCB/true':>14}")
    c2 = None
    for frac in (1e-4, 1e-3, 1e-2, 1e-1):
        kk = max(2, int(frac * N))
        idx = rng.choice(N, kk, replace=True, p=p)
        w = dropped[idx] / (N * p[idx])
        est = N * float(np.mean(w))
        ucb = eb_ucb(w, N)
        if c2 is None and ucb <= 2.0 * cert_full:
            c2 = frac
        P_(f"    {kk:>10,} {frac * 100:>9.2f}% {est / cert_full:>14.3f}"
           f" {ucb / cert_full:>14.3f}")
    P_(f"\n  C2: {'importance sampling reaches 2x at ' + f'{c2 * 100:.2f}%' if c2 is not None and c2 <= 0.1 else 'importance sampling also fails to reach 2x under a 10% sample.'}")

    # ---- C3  THE AGGREGATE BOUND ---------------------------------------------------------------
    P_("\n" + RULE)
    P_("C3  THE AGGREGATE BOUND: EXAMINE k CHILDREN PER PARENT, BOUND THE REST WITHOUT TOUCHING")
    P_(RULE)
    P_("  For each parent, take its k heaviest children and bound every other child in ONE term:")
    P_("  their total remaining mass times the largest ON-probability any child could have. That")
    P_("  largest is itself a box bound over the child space -- max over c of g.v_c is bounded by")
    P_("  sum_j max(g_j,0)*max_c v_c[j] -- which costs O(targets) per parent and touches NO child.")
    V = (gain * hvec[d] * Bst)                            # (n, T) child increments
    vmax = V.max(axis=0)                                  # per-target largest, precomputed once
    order_child = np.argsort(-Pm, axis=0)                 # heaviest children per parent, once
    lin_max = np.maximum(g, 0.0) @ vmax                   # (P,) box bound on the linear term
    P_(f"\n    {'k':>6} {'children touched':>18} {'certificate':>15} {'/ full scan':>13}")
    for k in (1, 2, 4, 8, 16, 32, 64, n):
        if k >= n:
            P_(f"    {k:>6} {ch.shape[0] * n:>18,} {cert_full:>15.6e} {1.0:>13.3f}")
            continue
        sel = order_child[:k, last]                        # (k, P) heaviest children of each parent
        colm = np.take_along_axis(ch, sel.T, axis=1)       # (P, k) their masses
        sl = np.take_along_axis(lg, sel.T, axis=1)         # (P, k) their exact log-bounds
        seen = colm * np.exp(sl)
        rest_mass = wts - colm.sum(axis=1)                 # mass of every other child, exact
        rest = np.maximum(rest_mass, 0.0) * np.exp(ls + lin_max)
        tot = float(seen.sum() + rest.sum())
        # the kept set is still subtracted, using only what was examined
        kept_est = float(np.sort(seen.ravel())[::-1][:cap].sum())
        cert_k = tot - kept_est
        P_(f"    {k:>6} {ch.shape[0] * k:>18,} {cert_k:>15.6e} {cert_k / cert_full:>13.3f}")

    # ---- C2b  THE NON-CIRCULAR CONFIGURATION ---------------------------------------------------
    P_("\n" + RULE)
    P_("C2b  THE CIRCULARITY IN C1 AND C2, AND THE VERSION THAT DOES NOT HAVE IT")
    P_(RULE)
    P_("  C1 and C2 sampled from the DROPPED set. Knowing which candidates are dropped means")
    P_("  having ranked them all by the bound -- which is the full scan. Their saving is illusory")
    P_("  as stated: they cheapen the verification of a partition whose construction already paid")
    P_("  the cost. My setup's defect, not sampling's.")
    P_("\n  THE NON-CIRCULAR VERSION: select by path MASS, which needs no bound evaluations -- the")
    P_("  child ordering under any parent is that column of the transition matrix, so one sort of")
    P_("  the columns gives every parent's heaviest children and a global top-k is a merge. Then")
    P_("  certify that partition by importance sampling, evaluating the bound only where sampled.")
    massflat = ch.ravel()
    keep_m = np.argpartition(-massflat, cap)[:cap]
    mask_m = np.ones(massflat.size, dtype=bool)
    mask_m[keep_m] = False
    drop_m = bnd.ravel()[mask_m]
    proxy_m = massflat[mask_m]
    cert_m_full = float(drop_m.sum())
    Nm = len(drop_m)
    P_(f"\n  selecting by mass instead of by bound: the full-scan certificate for THAT partition is")
    P_(f"  {cert_m_full:.6e}, against {cert_full:.6e} for the bound-selected one"
       f" ({cert_m_full / cert_full:.2f}x).")
    pm_ = proxy_m / proxy_m.sum()
    P_(f"\n    {'bound evals':>12} {'vs full scan':>13} {'estimate/true':>14} {'95% UCB/true':>14}")
    hit = None
    for frac in (1e-5, 1e-4, 1e-3, 1e-2):
        kk = max(2, int(frac * Nm))
        idx = rng.choice(Nm, kk, replace=True, p=pm_)
        wv = drop_m[idx] / (Nm * pm_[idx])
        est = Nm * float(np.mean(wv))
        ucb = eb_ucb(wv, Nm)
        if hit is None and ucb <= 2.0 * cert_m_full:
            hit = (kk, frac)
        P_(f"    {kk:>12,} {frac * 100:>12.4f}% {est / cert_m_full:>14.3f}"
           f" {ucb / cert_m_full:>14.3f}")
    if hit:
        P_(f"\n  C2b: a valid 95% upper bound within 2x costs {hit[0]:,} bound evaluations against")
        P_(f"  {ch.size:,} for the full scan -- a factor of {ch.size / hit[0]:,.0f}. The scan IS avoidable,")
        P_("  and the thing that avoids it is the cheap proxy plus sampling, NOT the aggregate")
        P_("  bound this module was built expecting to work.")
    else:
        P_("\n  C2b: no sampled bound within 2x under a 1% evaluation budget. The scan is not")
        P_("  avoidable this way either.")
    P_("\n  AND THE GUARANTEE IS WEAKER IN KIND, WHICH MUST NOT BE GLOSSED. The full scan gives an")
    P_("  EXACT sum. This gives a 95% confidence upper bound. That is a real demotion -- a")
    P_("  certificate that holds one time in twenty is not the same object -- and whether it is")
    P_("  acceptable is a decision about what the engine's guarantee is for, not a measurement.")

    # ---- C4/C5 ---------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("C4  THE VERDICT, AND C5 WHAT IT DOES NOT SETTLE")
    P_(RULE)
    P_("  Read C3's last column against C1's and C2's. A certificate is a guarantee; the sampling")
    P_("  gates produce confidence statements and only the aggregate bound produces a certificate")
    P_("  at all, so if C3 stays close to the full scan at small k the scan is avoidable and if it")
    P_("  does not, item 2 is unresolved and the approximation cannot beat exact DP.")
    P_("\n  NOT SETTLED.")
    P_("  1. One pruning level at one width. The aggregate term's looseness compounds across")
    P_("     levels and that is not measured here.")
    P_("  2. Sorting the transition matrix's columns once is O(n^2 log n), paid before any of this")
    P_("     helps. At n = 1024 that is cheap; it is not free at every width.")
    P_("  3. This asks only whether the SCAN can be avoided. exactcert already showed the bound")
    P_("     being verified is 25 orders looser than the tail, and nothing here changes that.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_sampcert.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
