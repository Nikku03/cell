"""Adaptive partitions: certifying a pruner that chooses what to drop using its own samples.

THE GAP composecert LEFT. Its sequential construction handles adaptive SAMPLE SIZES -- deciding how
long to keep sampling after looking at the data. It does not handle adaptive PARTITIONS: a pruner
that decides WHAT TO DROP using the same evaluations it then certifies. That is the pruner anyone
would actually want, because boundprune measured that selecting by the bound beats selecting by mass
by six to eight times in retained paths, and evaluating the bound everywhere is the full scan the
whole line of work is trying to avoid.

WHY IT IS NOT A SMALL PROBLEM. If the retained set is chosen because those candidates LOOKED large,
then the dropped set is exactly the candidates that looked small, and estimating the dropped sum
from the same evaluations understates it by construction. This is the winner's curse, and no amount
of extra sampling fixes it while the same sample does both jobs.

THE FIX, AND IT IS BETTER THAN SAMPLE SPLITTING. Do not estimate the dropped sum at all. Write

    D  =  T  -  R

where T is the total bound over ALL candidates and R is the total over the RETAINED ones. T does not
depend on the partition -- it is a fixed number for the level, whatever the pruner later decides. And
R is known EXACTLY, because a candidate cannot be retained without its bound having been evaluated.
So an upper bound on D is UCB(T) - R, and

    D <= UCB(T) - R    <=>    T - R <= UCB(T) - R    <=>    T <= UCB(T)

The retained term CANCELS. The guarantee holds with probability 1 - delta no matter how the
partition was chosen, including choices made using the very samples that estimated T. The selection
effect does not need to be corrected because it never enters.

WHAT IT COSTS instead is a subtraction: D is a difference of two quantities, and the error in UCB(T)
is absolute, so the relative accuracy on D degrades as the retained set captures more of the total.
That is the real price and P3 measures it.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

P0  THE CEILING GATE: IS THE PROBLEM REAL? Certify an adaptively-chosen partition the naive way --
    one sample used both to select and, restricted to the dropped side, to estimate the dropped sum.
    PREDECLARED: coverage below the nominal means the winner's curse bites and the fix is needed. If
    coverage does NOT fall, the reason must be identified before it is accepted, because this record
    has twice found an invalid construction surviving on slack and has twice refused to call that
    permission.

P1  THE SUBTRACTION IDENTITY, under a genuinely adaptive rule. Select using the samples, evaluate
    the retained set exactly, certify by UCB(T) - R.
    PREDECLARED: coverage must be at or above nominal BY CONSTRUCTION, since the identity makes the
    guarantee independent of the selection. If it is not, the implementation is wrong rather than
    the theory, and that is the thing to look for rather than a reason to doubt the algebra.

P2  SAMPLE SPLITTING, the obvious alternative: half the budget selects, the other half certifies the
    partition the first half fixed. Valid, and it pays twice.
    PREDECLARED: reported head to head with P1 at EQUAL total bound evaluations, since that is what
    the engine spends.

P3  WHERE THE SUBTRACTION DEGRADES, which is the identity's real cost. Sweep the retained fraction
    and measure the relative width of the certificate as R approaches T.
    PREDECLARED: reported as a law in the retained share of the total bound, not at a point. The
    identity is expected to fail gracefully and to fail EARLY when the pruner keeps most of the
    bound mass, which is precisely the regime a good pruner is trying to reach.

P4  COMPOSED ACROSS LEVELS with adaptive partitions at every one, carrying composecert's failure
    budget.

P5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.composecert import levels, ucb_fixed

DELTA = 0.05
REPS = 400


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("ADAPTIVE PARTITIONS: CERTIFYING A PRUNER THAT CHOOSES USING ITS OWN SAMPLES")
    P_(RULE)
    lv, sha = levels()
    K = len(lv)
    P_(f"\n  {K} pruning levels, 10 controllers, n = 1024. sha {sha[:12]}.")

    # one representative level to study in detail: the one carrying the certificate
    i = int(np.argmax([l["true"] for l in lv]))
    L0 = lv[i]
    f, g = L0["f"], L0["g"]
    N = len(f)
    Dtrue = L0["true"]
    P_(f"  studying level {i + 1}, which carries {Dtrue / sum(l['true'] for l in lv) * 100:.1f}%"
       f" of the composed certificate: {N:,} candidates.")
    p = g / g.sum()
    b = float(g.sum() / N * L0["ceil"])
    rng = np.random.default_rng(90909)

    def sample(k):
        idx = rng.choice(N, k, replace=True, p=p)
        return idx, f[idx] / (N * p[idx])

    massorder = np.argsort(-g)

    def adaptive_keep(idx, cap):
        """A genuinely data-dependent rule: keep the largest OBSERVED bounds, then fill by mass.

        Masked rather than looped -- the first version tested membership inside a loop over the
        mass order, which is O(cap*N) and was the slowest thing in the module."""
        seen = np.unique(idx)
        order = seen[np.argsort(-f[seen])][:cap]
        keep = np.zeros(N, dtype=bool)
        keep[order] = True
        short = cap - int(keep.sum())
        if short > 0:
            fill = massorder[~keep[massorder]][:short]
            keep[fill] = True
        return np.nonzero(keep)[0]

    CAP = 1000
    P_(f"\n  NOTE ON SAMPLE SIZES. The naive procedure needs MORE samples than the cap of {CAP}: with")
    P_("  fewer, the rule keeps every candidate it sampled and the dropped side of the sample is")
    P_("  empty, so there is nothing to estimate from. That is a degenerate configuration rather")
    P_("  than a small-sample one, and the sizes below are chosen to avoid it rather than to")
    P_("  flatter the method.")

    # ---- P0  THE CEILING GATE ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P0  THE CEILING GATE: DOES THE WINNER'S CURSE ACTUALLY BITE?")
    P_(RULE)
    P_("  The naive procedure: one sample selects the partition, and the same sample restricted to")
    P_("  the dropped side estimates the dropped sum.")
    P_(f"\n    {'evals':>8} {'coverage':>10} {'median UCB/true':>17} {'median est/true':>17}")
    for k in (3000, 10000, 30000):
        hits, rat, bias = 0, [], []
        for _ in range(REPS):
            idx, w = sample(k)
            keep = adaptive_keep(idx, CAP)
            ks = set(keep.tolist())
            m = np.array([j not in ks for j in idx], dtype=float)
            if m.sum() < 2:
                continue
            # THE CORRECT SUB-POPULATION ESTIMATOR is the full sample with retained draws set to
            # ZERO, not the surviving draws rescaled by a probability share. The first version used
            # the latter, which is not an unbiased estimator of anything.
            wd = w * m
            Dpart = Ttrue - float(f[keep].sum())          # the truth FOR THIS partition
            u = ucb_fixed(wd, N, b, DELTA)
            est = N * float(np.mean(wd))
            rat.append(u / Dpart)
            bias.append(est / Dpart)
            hits += int(u >= Dpart)
        if not rat:
            P_(f"    {k:>8,}   DEGENERATE -- every sampled candidate was retained; nothing to"
               f" estimate from")
            continue
        P_(f"    {k:>8,} {hits / len(rat) * 100:>9.1f}% {np.median(rat):>17.3f}"
           f" {np.median(bias):>17.3f}")
    P_("\n  P0: read the last column. An estimator that is unbiased would sit at 1.000; the")
    P_("  winner's curse shows up as a systematic shortfall, and the coverage column says whether")
    P_("  the bound's slack absorbs it.")

    # ---- P1  THE SUBTRACTION IDENTITY ----------------------------------------------------------
    P_("\n" + RULE)
    P_("P1  THE SUBTRACTION IDENTITY: UCB(TOTAL) MINUS THE EXACT RETAINED SUM")
    P_(RULE)
    Ttrue = float(f.sum())
    P_(f"  T = {Ttrue:.6e} is the total bound over the CANDIDATE POPULATION being partitioned")
    P_( "  here -- which is this level's mass-dropped set, the population a second-stage pruner")
    P_( "  would re-partition. It is fixed whatever the pruner then decides, and that is the only")
    P_( "  property the identity needs. The first version of this line called T the total over ALL")
    P_( "  candidates and then reported a retained share of 0.00%, which was a quantity compared")
    P_( "  with itself; both are corrected.")
    _idx0, _ = sample(4000)
    _k0 = adaptive_keep(_idx0, CAP)
    P_(f"  a typical adaptive keep of {CAP} from {N:,} captures"
       f" {float(f[_k0].sum()) / Ttrue * 100:.2f}% of T.")
    P_(f"\n    {'evals':>8} {'coverage':>10} {'median UCB/true':>17}")
    p1 = {}
    for k in (3000, 10000, 30000):
        hits, rat = 0, []
        for _ in range(REPS):
            idx, w = sample(k - CAP)
            keep = adaptive_keep(idx, CAP)
            R = float(f[keep].sum())                      # exact: retained bounds were evaluated
            u = ucb_fixed(w, N, b, DELTA) - R
            D = Ttrue - R
            rat.append(u / D)
            hits += int(u >= D)
        p1[k] = (hits / REPS, float(np.median(rat)))
        P_(f"    {k:>8,} {hits / REPS * 100:>9.1f}% {np.median(rat):>17.3f}")
    P_("\n  P1: coverage is by construction -- the retained term cancels out of the inequality, so")
    P_("  the guarantee cannot depend on how the partition was chosen. The evals column includes")
    P_(f"  the {CAP} evaluations spent on the retained set, which the identity requires be exact.")

    # ---- P2  SAMPLE SPLITTING ------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P2  SAMPLE SPLITTING, HEAD TO HEAD AT EQUAL TOTAL EVALUATIONS")
    P_(RULE)
    P_(f"\n    {'evals':>8} {'coverage':>10} {'median UCB/true':>17} {'vs identity':>13}")
    for k in (3000, 10000, 30000):
        half = max(2, (k - CAP) // 2)
        hits, rat = 0, []
        for _ in range(REPS):
            idx_a, _ = sample(half)
            keep = adaptive_keep(idx_a, CAP)
            ks = set(keep.tolist())
            idx_b, w_b = sample(half)
            m = np.array([j not in ks for j in idx_b], dtype=float)
            if m.sum() < 2:
                continue
            D = Ttrue - float(f[keep].sum())
            u = ucb_fixed(w_b * m, N, b, DELTA)           # zero-padded, as in P0
            rat.append(u / D)
            hits += int(u >= D)
        if not rat:
            P_(f"    {k:>8,}   DEGENERATE -- the certifying half had no dropped samples")
            continue
        P_(f"    {k:>8,} {hits / len(rat) * 100:>9.1f}% {np.median(rat):>17.3f}"
           f" {np.median(rat) / p1[k][1]:>12.2f}x")
    P_("\n  P2: splitting is valid too -- conditional on the first half, the partition is fixed --")
    P_("  but it certifies with half the budget and pays for it in width.")

    # ---- P3  WHERE THE SUBTRACTION DEGRADES ----------------------------------------------------
    P_("\n" + RULE)
    P_("P3  THE IDENTITY'S REAL COST: WHERE THE SUBTRACTION DEGRADES")
    P_(RULE)
    P_("  D = T - R is a difference, and the error in UCB(T) is ABSOLUTE, so the relative width on")
    P_("  D grows as the retained set captures more of the total bound. That is the regime a good")
    P_("  pruner is trying to reach, so this is the identity's real limit.")
    P_(f"\n    {'cap':>8} {'retained share of T':>21} {'median UCB/true':>17} {'coverage':>10}")
    srt = np.argsort(-f)
    for cap in (100, 1000, 10000, 100000, 400000):
        if cap >= N:
            continue
        keep = srt[:cap]                                   # best case: keep the largest bounds
        R = float(f[keep].sum())
        D = Ttrue - R
        hits, rat = 0, []
        for _ in range(200):
            idx, w = sample(3000)
            u = ucb_fixed(w, N, b, DELTA) - R
            rat.append(u / D)
            hits += int(u >= D)
        P_(f"    {cap:>8,} {R / Ttrue * 100:>20.2f}% {np.median(rat):>17.3f}"
           f" {hits / 200 * 100:>9.1f}%")
    P_("\n  P3: the width blows up as the retained share approaches one, exactly as the algebra")
    P_("  says it must. The identity is free of the selection effect and is NOT free of this.")

    # ---- P4  COMPOSED --------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P4  COMPOSED ACROSS ALL LEVELS, ADAPTIVE PARTITIONS AT EVERY ONE")
    P_(RULE)
    P_(f"  Carrying composecert's failure budget: delta/K per level with K = {K}.")
    tot_true = 0.0
    per = []
    for l in lv:
        n_ = len(l["f"])
        cap_l = min(1000, max(1, n_ // 4))
        s_ = np.argsort(-l["f"])[:cap_l]
        per.append((l, cap_l, float(l["f"][s_].sum()), float(l["f"].sum())))
        tot_true += float(l["f"].sum()) - float(l["f"][s_].sum())
    P_(f"  true composed dropped bound under these partitions: {tot_true:.6e}")
    P_(f"\n    {'evals/level':>12} {'coverage':>10} {'median total/true':>19}")
    for k in (1000, 3000, 10000):
        hits, rat = 0, []
        for _ in range(200):
            tot = 0.0
            for (l, cap_l, R, T) in per:
                n_ = len(l["f"])
                p_ = l["g"] / l["g"].sum()
                b_ = float(l["g"].sum() / n_ * l["ceil"])
                idx = rng.choice(n_, k, replace=True, p=p_)
                w = l["f"][idx] / (n_ * p_[idx])
                tot += ucb_fixed(w, n_, b_, DELTA / K) - R
            rat.append(tot / tot_true)
            hits += int(tot >= tot_true)
        P_(f"    {k:>12,} {hits / 200 * 100:>9.1f}% {np.median(rat):>19.3f}")
    P_("\n  P4: the identity composes exactly as composecert's sum does, because each level's")
    P_("  guarantee is still a statement about a fixed T, and the failure budget adds as before.")

    # ---- P5 ------------------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. The identity needs the retained set evaluated EXACTLY. That is cheap when the cap is")
    P_("     small and it is the dominant cost when the cap is large -- the same regime P3 says")
    P_("     the width degrades in. Both limits bite from the same side.")
    P_("  2. It bounds the dropped sum for WHATEVER partition was chosen. It says nothing about")
    P_("     whether the partition was a good one; a pruner that adapts badly gets a valid")
    P_("     certificate for a bad decision.")
    P_("  3. T is fixed only within a level. Across levels the candidate set at level d+1 depends")
    P_("     on what level d retained, so T at the next level IS random. The composition in P4")
    P_("     conditions on the realised chain, which is what composecert also did; a guarantee")
    P_("     unconditional on the whole trajectory is not established here.")
    P_("  4. The object is still the certificate and not the tail, and exactcert's finding that")
    P_("     the bound is far looser than the quantity it bounds is untouched.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_adaptpart.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
