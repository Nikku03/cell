"""Composing the guarantee across levels and adaptive decisions.

THE GAP. certcover validated ONE level's sampled bound at one partition. The engine prunes at
every level of the window and its certificate is the SUM of the level-wise dropped bounds -- which
is the right composition, since a subtree bound already covers that prefix's whole contribution to
the final tail. But giving each level a separate 95% bound does NOT give the sum 95% coverage. At
L = 6 there are seven calls, and a union bound over seven nominal-95% statements guarantees only
65%. Adaptive decisions make it worse: if the engine uses the certificates to choose where to
spend, the partitions stop being fixed and a fixed-time bound is not valid under a stopping rule
that looked at the data.

TWO CONSTRUCTIONS ARE ON THE TABLE and this prices both.
    ALLOCATE A TOTAL FAILURE BUDGET. Give call j a level delta_j with sum delta_j = delta. Valid
    by the union bound, for a fixed, data-independent schedule of calls. The cost is that each
    call's bound widens.
    A VALID SEQUENTIAL CONSTRUCTION. A confidence sequence is valid at ALL sample sizes at once and
    therefore under optional stopping and adaptive continuation. It costs more at any single time
    and buys the right to look at the data before deciding what to do next.

AND ONE THING HAS TO BE MEASURED BEFORE EITHER, because certcover already suggests it. Its
per-level bounds were not 95% accurate -- they covered 400 times out of 400. A composition of
seven bounds that each hold with probability near one may hold with probability near one, and the
union bound's 65% would then be a worst case that does not describe this engine. That is testable
and it is the ceiling gate.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

Z0  THE CEILING GATE: WHAT IS THE END-TO-END COVERAGE OF THE NAIVE COMPOSITION? Run the whole
    multi-level procedure many times with each level certified at nominal 95%, and measure how
    often the SUMMED bound covers the true summed deficit.
    PREDECLARED: if naive composition already covers at or above 95%, the union bound's 65% is a
    worst case that does not bind here, and the honest report is that the problem is real in
    principle and absent in practice at this scale -- which must be said in those words rather
    than by quietly declaring the gap closed. If it covers below 95%, the gap is real and Z1 and
    Z3 are the fixes.

Z1  THE TOTAL FAILURE BUDGET, which is valid whenever the schedule of calls is fixed in advance.
    Each of the K levels gets delta/K. Measure coverage and the sample cost of reaching the same
    tightness the naive version reached.
    PREDECLARED: coverage must be at or above the target by construction; if it is not, the
    implementation is wrong rather than the theory. The number that matters is the COST of
    validity, in samples, against the naive version.

Z2  ADAPTIVITY, WHICH IS WHERE THE FIXED-TIME CONSTRUCTION SHOULD BREAK. Make the procedure
    genuinely adaptive: spend the remaining sample budget where the running certificates are
    largest, so the number of samples at each level depends on data already seen.
    PREDECLARED: a fixed-time bound applied at a data-dependent sample size is not valid, so its
    coverage should fall. If it does not fall, the reason must be identified rather than taken as
    permission -- an invalid construction that happens to hold is what certcover already found
    once and it was relabelled, not accepted.

Z3  THE SEQUENTIAL CONSTRUCTION, valid at every sample size simultaneously by a union over dyadic
    epochs, hence valid under optional stopping.
    PREDECLARED: coverage at or above target under the SAME adaptive rule that breaks Z2, and the
    cost reported as the sample size needed to match the naive version's tightness.

Z4  THE RECOMMENDATION, with the numbers attached, and Z5 what it does not settle.
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

DELTA = 0.05
REPS = 400


def levels(nCtrl=10, L=6, dt=0.5, cap=1000):
    """Run the real multi-level pruner with MASS selection, and return, for every level, the
    dropped candidates' exact bounds and masses plus the cheap valid ON-probability ceiling."""
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
    lv = []
    for d in range(1, L + 1):
        ch = Pm[:, last].T * wts[:, None]
        Hr = float(hvec[d + 1:].sum())
        u = base + gain * (wact @ S.T) + gain * Hr * Splus[None, :]
        ls = (-np.logaddexp(0.0, -u)).sum(axis=1)
        g = 1.0 / (1.0 + np.exp(u))
        V = gain * hvec[d] * Bst
        lg = ls[:, None] + g @ V.T
        bnd = (ch * np.exp(lg)).ravel()
        mass = ch.ravel()
        keep = np.argpartition(-mass, cap)[:cap]
        mask = np.ones(mass.size, dtype=bool)
        mask[keep] = False
        ceiling = float(np.exp(np.max(ls + np.maximum(g, 0.0) @ V.max(axis=0))))
        lv.append({"f": bnd[mask], "g": mass[mask], "true": float(bnd[mask].sum()),
                   "ceil": ceiling, "ncand": mass.size})
        par, code = keep // n, keep % n
        wts, wact, last = mass[keep], wact[par] + actbit[code] * hvec[d], code
    return lv, sha


def ucb_fixed(x, N, b, delta):
    """Empirical-Bernstein upper bound on the sum, valid at a FIXED sample size."""
    k = len(x)
    if k < 2:
        return float("inf")
    m, v = float(np.mean(x)), float(np.var(x, ddof=1))
    lg = np.log(3.0 / delta)
    return N * (m + np.sqrt(2.0 * v * lg / k) + 3.0 * b * lg / k)


def ucb_seq(x, N, b, delta):
    """Anytime-valid: a union over dyadic epochs, so it holds at EVERY sample size at once.

    Sample size k sits in epoch j = floor(log2 k); spending delta_j = delta * 6/(pi^2 (j+1)^2),
    which sums to delta over all epochs, makes the whole sequence valid -- hence valid under a
    stopping rule that looked at the data."""
    k = len(x)
    if k < 2:
        return float("inf")
    j = int(np.floor(np.log2(k)))
    dj = delta * 6.0 / (np.pi ** 2 * (j + 1) ** 2)
    return ucb_fixed(x, N, b, dj)


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("COMPOSING THE GUARANTEE ACROSS LEVELS AND ADAPTIVE DECISIONS")
    P_(RULE)
    lv, sha = levels()
    K = len(lv)
    total_true = sum(l["true"] for l in lv)
    P_(f"\n  {K} pruning levels, mass-selected, 10 controllers, n = 1024. sha {sha[:12]}.")
    P_(f"\n    {'level':>6} {'dropped':>12} {'true cert':>15} {'share':>8}")
    for i, l in enumerate(lv, 1):
        P_(f"    {i:>6} {len(l['f']):>12,} {l['true']:>15.6e} {l['true'] / total_true * 100:>7.2f}%")
    P_(f"\n  TRUE TOTAL DEFICIT BOUND, summed over levels: {total_true:.6e}")
    P_("  This is the object a composed guarantee has to cover. It is a sum of level-wise subtree")
    P_("  bounds, which is the correct composition -- a subtree bound already covers that prefix's")
    P_("  whole contribution to the final tail, so dropping it costs at most that much.")

    rng = np.random.default_rng(4242)
    ps = [l["g"] / l["g"].sum() for l in lv]
    bs = [float(l["g"].sum() / len(l["g"]) * l["ceil"]) for l in lv]

    def draw(i, k):
        idx = rng.choice(len(lv[i]["f"]), k, replace=True, p=ps[i])
        return lv[i]["f"][idx] / (len(lv[i]["f"]) * ps[i][idx])

    def run(kper, mode, delta=DELTA):
        """One end-to-end pass. mode: 'naive' | 'bonf' | 'seq'."""
        tot = 0.0
        for i in range(K):
            w = draw(i, kper)
            if mode == "naive":
                tot += ucb_fixed(w, len(lv[i]["f"]), bs[i], delta)
            elif mode == "bonf":
                tot += ucb_fixed(w, len(lv[i]["f"]), bs[i], delta / K)
            else:
                tot += ucb_seq(w, len(lv[i]["f"]), bs[i], delta / K)
        return tot

    # ---- Z0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("Z0  THE CEILING GATE: END-TO-END COVERAGE OF THE NAIVE COMPOSITION")
    P_(RULE)
    P_(f"  Each of the {K} levels certified at nominal 95%. A union bound guarantees only"
       f" {100 * (1 - K * DELTA):.0f}%.")
    P_(f"\n    {'samples/level':>14} {'coverage':>10} {'median total/true':>19}")
    z0 = {}
    for kper in (100, 300, 1000, 3000):
        hits, rat = 0, []
        for _ in range(REPS):
            t = run(kper, "naive")
            rat.append(t / total_true)
            hits += int(t >= total_true)
        z0[kper] = (hits / REPS, float(np.median(rat)))
        P_(f"    {kper:>14,} {hits / REPS * 100:>9.1f}% {np.median(rat):>19.3f}")
    worst = min(v[0] for v in z0.values())
    P_(f"\n  Z0: worst end-to-end coverage {worst * 100:.1f}% against a nominal 95% and a union-bound")
    P_(f"  guarantee of {100 * (1 - K * DELTA):.0f}%. {'THE NAIVE COMPOSITION COVERS. The union bound is a worst case that does not bind here -- and the reason is that each level s bound is far more conservative than its nominal, which certcover measured as 100% coverage where 95% was claimed. The gap is REAL IN PRINCIPLE AND ABSENT IN PRACTICE AT THIS SCALE, which is not the same as closed.' if worst >= 0.95 else 'THE NAIVE COMPOSITION FAILS its nominal. The gap is real and Z1 and Z3 are the fixes.'}")

    # ---- Z1  THE FAILURE BUDGET ----------------------------------------------------------------
    P_("\n" + RULE)
    P_("Z1  ALLOCATING A TOTAL FAILURE BUDGET: delta/K PER LEVEL")
    P_(RULE)
    P_(f"\n    {'samples/level':>14} {'coverage':>10} {'median total/true':>19} {'vs naive':>10}")
    for kper in (100, 300, 1000, 3000):
        hits, rat = 0, []
        for _ in range(REPS):
            t = run(kper, "bonf")
            rat.append(t / total_true)
            hits += int(t >= total_true)
        P_(f"    {kper:>14,} {hits / REPS * 100:>9.1f}% {np.median(rat):>19.3f}"
           f" {np.median(rat) / z0[kper][1]:>9.3f}x")
    P_("\n  Z1: valid by the union bound whenever the schedule of calls is fixed in advance. The")
    P_("  last column is the price of that validity at equal sample size.")

    # ---- Z2  ADAPTIVITY ------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("Z2  ADAPTIVITY, WHERE THE FIXED-TIME CONSTRUCTION SHOULD BREAK")
    P_(RULE)
    P_("  A genuinely adaptive rule: take a pilot sample everywhere, then spend the remaining")
    P_("  budget on whichever levels look largest. The sample size at each level now depends on")
    P_("  data already seen, and a fixed-time bound applied at a data-dependent size is not valid.")

    def run_adaptive(pilot, extra, mode, delta=DELTA):
        pilots = [draw(i, pilot) for i in range(K)]
        score = np.array([np.mean(w) * len(lv[i]["f"]) for i, w in enumerate(pilots)])
        share = score / score.sum()
        tot = 0.0
        for i in range(K):
            k = pilot + int(extra * share[i])
            w = np.concatenate([pilots[i], draw(i, max(k - pilot, 1))])
            if mode == "bonf":
                tot += ucb_fixed(w, len(lv[i]["f"]), bs[i], delta / K)
            else:
                tot += ucb_seq(w, len(lv[i]["f"]), bs[i], delta / K)
        return tot

    P_(f"\n    {'construction':<16} {'pilot+extra':>14} {'coverage':>10} {'median total/true':>19}")
    z2 = {}
    for mode in ("bonf", "seq"):
        for pilot, extra in ((100, 2000), (300, 6000)):
            hits, rat = 0, []
            for _ in range(REPS):
                t = run_adaptive(pilot, extra, mode)
                rat.append(t / total_true)
                hits += int(t >= total_true)
            z2[(mode, pilot)] = (hits / REPS, float(np.median(rat)))
            nm = "fixed-time" if mode == "bonf" else "sequential"
            P_(f"    {nm:<16} {f'{pilot}+{extra}':>14} {hits / REPS * 100:>9.1f}%"
               f" {np.median(rat):>19.3f}")
    fb = min(v[0] for kk, v in z2.items() if kk[0] == "bonf")
    sq = min(v[0] for kk, v in z2.items() if kk[0] == "seq")
    P_(f"\n  Z2 AS PREDECLARED: the fixed-time construction covers {fb * 100:.1f}% under the adaptive rule;")
    P_(f"  the sequential one covers {sq * 100:.1f}%.")
    if fb >= 0.95:
        P_("  THE FIXED-TIME BOUND DID NOT BREAK, AND THAT IS NOT PERMISSION. It is invalid under")
        P_("  a data-dependent sample size regardless of what it does here; the reason it survives")
        P_("  is the same slack certcover found, not a property of the construction. Reporting it")
        P_("  as adequate would repeat exactly the error that made sampcert's 10,039x need")
        P_("  relabelling.")

    # ---- Z3/Z4/Z5 -------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("Z3  THE SEQUENTIAL CONSTRUCTION'S PRICE, AND Z4 THE RECOMMENDATION")
    P_(RULE)
    P_(f"\n    {'construction':<34} {'valid under':<26} {'median total/true':>19}")
    P_(f"    {'per-level 95%, summed':<34} {'nothing -- 65% guaranteed':<26} {z0[1000][1]:>19.3f}")
    P_(f"    {'failure budget delta/K':<34} {'a fixed schedule of calls':<26} "
       f"{z2[('bonf', 300)][1]:>19.3f}")
    P_(f"    {'sequential, delta/K per level':<34} {'optional stopping, adaptive':<26} "
       f"{z2[('seq', 300)][1]:>19.3f}")
    P_("\n  Z4 THE RECOMMENDATION. The engine's certificate is a SUM over levels, so the failure")
    P_("  budget composes additively and delta/K per level is the right default whenever the")
    P_("  schedule of calls is fixed. The moment the engine uses its own certificates to decide")
    P_("  where to spend -- which is the whole point of an adaptive pruner -- that construction")
    P_("  stops being valid and the sequential one is required. Its price is the last column.")
    P_("\n" + RULE)
    P_("Z5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. The partitions here are MASS-selected and therefore fixed given the chain. A pruner")
    P_("     that selected using the sampled certificates would make the partitions themselves")
    P_("     random, and nothing here covers that -- the sequential bound handles adaptive SAMPLE")
    P_("     SIZES, not adaptive PARTITIONS.")
    P_("  2. Coverage is measured against the true summed bound, which needed a full scan per")
    P_("     level to know. This validates the composition, not any run of it.")
    P_("  3. The object composed is the certificate, not the tail. exactcert's finding stands:")
    P_("     the bound being composed is far looser than the quantity it bounds.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_composecert.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
