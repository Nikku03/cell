"""How reliably does sampcert's bound actually hold?

WHAT WAS CLAIMED AND WHAT WAS MEASURED. sampcert reported that a 95% upper bound on the dropped
contribution costs 306 bound evaluations against 3,072,000 for a full scan -- a factor of 10,039 --
for the mass-selected partition. That figure came from ONE draw at each sample size. One draw says
nothing about coverage: a bound that holds once is not a bound that holds 95% of the time, and the
whole value of the claim is the "95%".

AND THERE IS A SPECIFIC REASON TO DOUBT IT. The empirical-Bernstein form used takes the RANGE of
the summand to be the SAMPLE maximum. That is standard practice and it is not valid: the true
maximum can exceed anything a small sample has seen, and when it does the bound is too small in
exactly the cases where it needs to be large. With importance sampling the same worry sharpens --
a candidate with a large bound and a small mass gets a huge weight and appears rarely, which is the
classic way an importance sampler reports a confident wrong answer.

BUT A VALID RANGE EXISTS HERE AND COSTS NOTHING, which is why this module can do more than
complain. Every summand is a mass times an ON-probability, and an ON-probability is at most one, so
bound(p,c) <= mass(p,c). Under importance sampling proportional to mass the weight is therefore
bounded by the DROPPED SET'S TOTAL MASS divided by its size -- a constant known before any bound is
evaluated. A bound built on that range is valid rather than heuristic, and K3 measures what it
costs.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

K0  THE CEILING GATE, ON VALIDITY RATHER THAN ON TIGHTNESS. Measure how often the SAMPLE maximum
    understates the TRUE maximum of the summand, at each sample size used.
    PREDECLARED: if the sample max understates the true max in most draws at the sample sizes
    sampcert quoted, then the reported "95%" was never established and the headline factor of
    10,039 is a claim about a bound that was not valid. That must be said before any coverage
    number is reported, because it is a defect in the instrument and not in the answer.

K1  EMPIRICAL COVERAGE, which is the question actually asked. Repeat the whole sampling procedure
    many times independently and count how often the upper bound really does exceed the true sum.
    PREDECLARED: coverage at or above 95% means the guarantee holds as claimed. Below it, the
    guarantee fails and the honest statement is the measured coverage, not the nominal one.

K2  THE IMPORTANCE WEIGHTS' TAIL, which is the mechanism if K1 fails. Report the largest weight
    against the mean weight, and what share of the total sum sits in candidates whose mass rank is
    far from their bound rank -- those are the ones importance sampling by mass will miss.

K3  THE VALID BOUND, BUILT ON THE KNOWN RANGE. Rerun coverage using the analytic range instead of
    the sample maximum.
    PREDECLARED: this bound is valid by construction, so its coverage must be at or above 95% --
    if it is not, the implementation is wrong rather than the theory. The number that matters is
    then the COST: how many bound evaluations the valid version needs to reach the same tightness
    the heuristic one claimed at 306.

K4  THE VERDICT AND WHAT IT DOES NOT SETTLE.
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

CONF = 0.95
REPS = 400


def build(nCtrl=10, L=6, dt=0.5, cap=3000):
    """The same real pruning level sampcert used, rebuilt here so the two are comparable."""
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, 200)
    pi, res, _ = stationary(Q)
    n = Q.shape[0]
    Pm = expm((Q.T * dt).toarray())
    hvec = np.exp(-0.5 * np.arange(L, -1, -1))
    hvec = hvec / hvec.sum()
    S = identity_S(rows, nCtrl)
    actbit = np.array([[(m >> c) & 1 for c in range(nCtrl)] for m in range(n)], dtype=float)
    Splus = np.maximum(S, 0.0).sum(axis=1)
    Bst = actbit @ S.T
    base, gain = -1.0, 2.0
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
    ch = Pm[:, last].T * wts[:, None]
    Hr = float(hvec[d + 1:].sum())
    u = base + gain * (wact @ S.T) + gain * Hr * Splus[None, :]
    ls = (-np.logaddexp(0.0, -u)).sum(axis=1)
    g = 1.0 / (1.0 + np.exp(u))
    lg = ls[:, None] + g @ (gain * hvec[d] * Bst).T
    bnd = ch * np.exp(lg)
    massflat = ch.ravel()
    keep = np.argpartition(-massflat, cap)[:cap]          # MASS-selected, as sampcert's C2b
    mask = np.ones(massflat.size, dtype=bool)
    mask[keep] = False
    return bnd.ravel()[mask], massflat[mask], sha, ch.size


def ucb(x, N, b, conf=CONF):
    """Empirical-Bernstein upper bound on the sum, with `b` the range of the summand."""
    k = len(x)
    if k < 2:
        return float("inf")
    m, v = float(np.mean(x)), float(np.var(x, ddof=1))
    lg = np.log(3.0 / (1.0 - conf))
    return N * (m + np.sqrt(2.0 * v * lg / k) + 3.0 * b * lg / k)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("HOW RELIABLY DOES sampcert's BOUND HOLD? COVERAGE, NOT ONE DRAW")
    P_(RULE)
    f, gmass, sha, ncand = build()
    N = len(f)
    truth = float(f.sum())
    p = gmass / gmass.sum()
    wmax_true = float(np.max(f / (N * p)))
    P_(f"\n  the mass-selected partition: {N:,} dropped candidates of {ncand:,},"
       f" true sum {truth:.6e}. sha {sha[:12]}.")
    P_(f"  analytic range on the importance weight, total dropped mass / N ="
       f" {gmass.sum() / N:.6e}")
    P_(f"  the largest weight that actually occurs:                        {wmax_true:.6e}")
    rng = np.random.default_rng(31337)
    sizes = (30, 306, 3069, 30690)

    # ---- K0  VALIDITY --------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("K0  THE CEILING GATE: IS THE RANGE THE HEURISTIC USES EVEN AN UPPER BOUND?")
    P_(RULE)
    P_(f"\n    {'sample':>8} {'draws where sample max < true max':>36} {'median shortfall':>18}")
    bad0 = None
    for k in sizes:
        under, ratios = 0, []
        for _ in range(REPS):
            idx = rng.choice(N, k, replace=True, p=p)
            w = f[idx] / (N * p[idx])
            if w.max() < wmax_true:
                under += 1
                ratios.append(wmax_true / max(w.max(), 1e-300))
        frac = under / REPS
        if bad0 is None and frac <= 0.5:
            bad0 = k
        P_(f"    {k:>8,} {frac * 100:>35.1f}% "
           f"{(np.median(ratios) if ratios else 1.0):>17.1f}x")
    P_(f"\n  K0: {'the sample maximum understates the true maximum in most draws at the sizes sampcert quoted. The reported 95% was NOT established.' if bad0 is None or bad0 > 306 else 'the sample maximum is usually adequate at the quoted sizes.'}")

    # ---- K1  COVERAGE --------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("K1  EMPIRICAL COVERAGE OF THE BOUND AS sampcert COMPUTED IT")
    P_(RULE)
    P_(f"  {REPS} independent repetitions per sample size. Nominal coverage {CONF * 100:.0f}%.")
    P_(f"\n    {'sample':>8} {'coverage':>10} {'median UCB/true':>17} {'worst UCB/true':>16}")
    cov_heur = {}
    for k in sizes:
        hits, rat = 0, []
        for _ in range(REPS):
            idx = rng.choice(N, k, replace=True, p=p)
            w = f[idx] / (N * p[idx])
            u = ucb(w, N, float(np.max(w)))
            rat.append(u / truth)
            hits += int(u >= truth)
        cov_heur[k] = hits / REPS
        P_(f"    {k:>8,} {hits / REPS * 100:>9.1f}% {np.median(rat):>17.3f}"
           f" {min(rat):>16.3f}")
    P_(f"\n  K1: coverage at the headline size of 306 is {cov_heur[306] * 100:.1f}% against a nominal 95%."
       f" {'The guarantee HOLDS.' if cov_heur[306] >= 0.95 else 'THE GUARANTEE FAILS at the size sampcert quoted.'}")

    # ---- K2  THE WEIGHT TAIL -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("K2  THE MECHANISM: THE IMPORTANCE WEIGHTS' TAIL")
    P_(RULE)
    wall = f / (N * p)
    P_(f"  weight mean {wall.mean():.4e}, max {wall.max():.4e},"
       f" max/mean {wall.max() / wall.mean():,.0f}x")
    rb = np.argsort(np.argsort(-f))
    rm = np.argsort(np.argsort(-gmass))
    disl = np.abs(rb - rm)
    heavy = np.argsort(-f)[:max(1, N // 1000)]
    P_(f"  rank correlation of bound against mass: {np.corrcoef(rb, rm)[0, 1]:.4f}")
    P_(f"  the heaviest 0.1% by BOUND sit at a median mass-rank displacement of"
       f" {np.median(disl[heavy]):,.0f} places")
    P_(f"  and carry {f[heavy].sum() / truth * 100:.2f}% of the sum.")

    # ---- K3  THE VALID BOUND -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("K3  THE BOUND BUILT ON THE RANGE THAT IS ACTUALLY KNOWN")
    P_(RULE)
    P_("  Every summand is a mass times an ON-probability at most one, so bound <= mass, and under")
    P_("  sampling proportional to mass the weight is at most (total dropped mass)/N -- known")
    P_("  before any bound is evaluated. This bound is valid by construction.")
    bknown = float(gmass.sum() / N)
    P_(f"\n    {'sample':>8} {'coverage':>10} {'median UCB/true':>17}")
    cov_valid = {}
    for k in sizes:
        hits, rat = 0, []
        for _ in range(REPS):
            idx = rng.choice(N, k, replace=True, p=p)
            w = f[idx] / (N * p[idx])
            u = ucb(w, N, bknown)
            rat.append(u / truth)
            hits += int(u >= truth)
        cov_valid[k] = (hits / REPS, float(np.median(rat)))
        P_(f"    {k:>8,} {hits / REPS * 100:>9.1f}% {np.median(rat):>17.3f}")
    need = [k for k in sizes if cov_valid[k][0] >= 0.95 and cov_valid[k][1] <= 2.0]
    P_(f"\n  K3: {'the valid bound reaches 95% coverage within 2x at ' + f'{min(need):,}' + ' evaluations' if need else 'the valid bound does not reach 2x tightness at any size tested'}"
       f" against {ncand:,} for the full scan"
       f"{f' -- a factor of {ncand / min(need):,.0f}.' if need else '.'}")

    # ---- K4 ------------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("K4  THE VERDICT, AND WHAT IT DOES NOT SETTLE")
    P_(RULE)
    P_(f"  heuristic range, coverage at 306: {cov_heur[306] * 100:.1f}%")
    P_(f"  known range,      coverage at 306: {cov_valid[306][0] * 100:.1f}%"
       f"  at {cov_valid[306][1]:.2f}x the true sum")
    P_("\n  NOT SETTLED.")
    P_("  1. One pruning level, one width, one partition. Coverage is a property of the summand")
    P_("     distribution and that changes with level.")
    P_("  2. Coverage is measured against the TRUE sum for this partition, which required the full")
    P_("     scan to know. In use there is nothing to check against -- that is what a guarantee is")
    P_("     for -- so this validates the procedure, not any particular run of it.")
    P_("  3. A 95% bound is still not a certificate in the sense exactcert used. It is a")
    P_("     confidence statement, and the engine's guarantee is demoted by adopting it.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_certcover.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
