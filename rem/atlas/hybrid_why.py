"""Why the greedy rule failed H6, and whether anything cheap fixes it.

WHAT FAILED. hybrid.py's H6 compared the greedy-by-|S| subset of each size against the best
subset of that size, on identical draws. It failed: worst shortfall 0.1400 at 2 kcal/mol,
factor-of-2 tolerance, m = 6, against a paired standard error of 0.0142 -- about ten sigma, not
noise. A second real failure sits at 1 kcal/mol, same tolerance and same m (0.0400 against 0.0080).
Both are at LARGE m and LARGE chemistry error, and both are on the tight tolerance.

THE SUSPICION, WHICH THIS MODULE EXISTS TO TEST RATHER THAN ASSERT. |S| is a local derivative,
measured with a step of 0.02 orders. At 2 kcal/mol the draws have sigma = 1.466 orders, so a rate
routinely lands three or four orders from its true value. A rate whose derivative is tiny at the
base point can still wreck the answer once it is moved that far -- b_off, the waking rate, is
0.03 per hour at the base point and a plus-three-order draw makes it 30 per hour, which empties
the dormant compartment during every drug-free window. Nothing about the derivative at 0.03
predicts that.

If the suspicion is right, the fix is not to abandon ranking. It is to rank by the damage a rate
actually does AT THE ERROR LEVEL IN USE, which costs one one-dimensional sweep per rate and no
joint enumeration at all.

=================================================================================================
GATES, PREDECLARED BEFORE THIS RUN
=================================================================================================

W1  IT REPRODUCES. Re-running the same seed must recover hybrid.py's greedy hit fractions at the
    failing cells to within 1e-12. If the failure does not reproduce it was a harness artefact and
    everything below is void.

W2  WHAT THE BEST SUBSET KNOWS THAT GREEDY DOES NOT. Report, at each failing cell, which rates the
    best subset of that size measures and which greedy measures, and hence which rate greedy left
    to chemistry that it should not have. Reported, not gated -- this is the diagnosis.

W3  IS THE LOCAL DERIVATIVE THE CULPRIT? Rank the rates instead by REALISED damage at the error
    level in use: perturb one rate at a time by the actual draws and take the standard deviation
    of log10(Y_hat/Y_true). Predeclared readings: if this ranking differs from the |S| ranking
    exactly at the rate identified in W2, the local-derivative explanation is confirmed; if the
    two rankings agree, the explanation is REFUTED and the failure needs another cause.

W4  DOES THE FIX ACTUALLY CLOSE THE GAP? Recompute the whole greedy curve under the realised-
    damage ranking, on the same draws and against the same exhaustive best. Predeclared: worst
    shortfall within 0.03 at every m means the fix works and H6 passes under the repaired rule;
    anything larger is reported as the residual, and the honest claim becomes that no cheap
    ranking recovers the optimum.

W5  IS THE OPTIMUM EVEN REACHABLE BY ANY RANKING? A ranking produces a NESTED sequence of subsets.
    Check whether the exhaustive best subsets are themselves nested, i.e. whether best(m) is
    contained in best(m+1) at every m. If they are not, then no ranking-based rule of any kind can
    be optimal at every size, and W4's bar is unreachable in principle rather than in practice.
    This is checked BEFORE W4's result is interpreted, precisely so that an unreachable bar is
    identified as unreachable rather than recorded as a failure -- the fourth time this build
    order has had to make that distinction.

W6  THE COST OF THE FIX. Model evaluations needed by the realised-damage ranking versus by the
    exhaustive enumeration it would replace. A fix that costs what it saves is not a fix.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np

from rem.atlas.hybrid_tune import (
    RULE, OFF_RATES, ON_RATES, NAMES, CANDIDATE, ORDERS_PER_KCAL,
    state_index, eradication, sensitivity,
)
from rem.atlas.hybrid import (
    K, G0, CYCLES, T_ON, T_OFF, N_TRIALS, TOLS, TARGET, SEED,
    N_RATES, N_ON, N_SUB, run_trial, subset_sizes, greedy_masks,
)

EPS_LEVELS = (1.0, 2.0)          # the two levels where H6 failed
N_PROBE = 400                    # draws for the one-dimensional realised-damage sweep


def members(idx):
    i, j = divmod(int(idx), 2 ** N_ON)
    got = [nm for k, nm in enumerate(OFF_RATES) if (i >> k) & 1]
    got += [nm for k, nm in enumerate(ON_RATES) if (j >> k) & 1]
    return set(got)


def mask_of(chosen):
    i = sum(1 << k for k, nm in enumerate(OFF_RATES) if nm in chosen)
    j = sum(1 << k for k, nm in enumerate(ON_RATES) if nm in chosen)
    return i * (2 ** N_ON) + j


def order_masks(order):
    return [mask_of(set(order[:m])) for m in range(N_RATES + 1)]


def main():
    out = []

    def P(s=""):
        print(s, flush=True)
        out.append(s)

    P(RULE); P("WHY GREEDY FAILED H6, AND WHETHER ANYTHING CHEAP FIXES IT"); P(RULE)

    S, IX = state_index(K)
    y_true = eradication(CANDIDATE, K=K, g0=G0, cycles=CYCLES)
    kw = dict(K=K, g0=G0, cycles=CYCLES)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, **kw) for nm in NAMES}
    order_S = sorted(NAMES, key=lambda n: -abs(Sd[n]))
    gm_S = order_masks(order_S)
    sizes = subset_sizes()
    ly = np.log10(y_true)

    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N_TRIALS, N_RATES))       # identical to hybrid.py

    P(f"  Y_true = {y_true:.6e};  |S| ranking: {order_S}")

    # ---- rerun the two failing error levels, same draws --------------------------------------
    Yl = {}
    for eps in EPS_LEVELS:
        sigma = eps * ORDERS_PER_KCAL
        Y = np.empty((N_TRIALS, N_SUB))
        t0 = time.time()
        for t in range(N_TRIALS):
            Y[t] = run_trial({nm: Z[t, k] for k, nm in enumerate(NAMES)}, sigma, K, S, IX, G0)
            if t and t % 200 == 0:
                P(f"    ... eps={eps}: {t}/{N_TRIALS}, {time.time()-t0:.0f}s")
        Yl[eps] = np.log10(np.maximum(Y, 1e-300))

    # ---- W1 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("W1  IT REPRODUCES"); P(RULE)
    P("  hybrid.py recorded, tolerance x2: greedy 0.9600 at eps=1 m=6, 0.8600 at eps=2 m=6.")
    tol2 = dict(TOLS)["x2"]
    repro = {}
    for eps in EPS_LEVELS:
        h = (np.abs(Yl[eps] - ly) <= tol2)
        repro[eps] = float(h[:, gm_S[6]].mean())
        P(f"  recomputed at eps={eps}, m=6: {repro[eps]:.4f}")
    ok = abs(repro[1.0] - 0.9600) < 1e-12 and abs(repro[2.0] - 0.8600) < 1e-12
    P(f"  {'PASS -- the failure is real and reproducible' if ok else 'FAIL -- does not reproduce'}")

    # ---- W5 first: is the bar even reachable? -------------------------------------------------
    P("\n" + RULE); P("W5  IS THE OPTIMUM REACHABLE BY ANY RANKING AT ALL?"); P(RULE)
    P("  A ranking gives a NESTED chain of subsets. If the exhaustive best subsets are not")
    P("  themselves nested, no ranking rule can match them at every size, and W4's bar is")
    P("  unreachable in principle. Checked before W4 is read.")
    nested_all = True
    for eps in EPS_LEVELS:
        for lab, tol in TOLS:
            h = (np.abs(Yl[eps] - ly) <= tol)
            best = []
            for m in range(N_RATES + 1):
                cols = np.where(sizes == m)[0]
                best.append(int(cols[int(np.argmax(h[:, cols].mean(axis=0)))]))
            breaks = [m for m in range(N_RATES)
                      if not members(best[m]).issubset(members(best[m + 1]))]
            nested_all &= not breaks
            P(f"  eps={eps} {lab}: best-subset chain nested? "
              f"{'yes' if not breaks else 'NO, breaks at m = ' + str(breaks)}")
    P(f"  {'Every chain is nested, so a perfect ranking could in principle exist.' if nested_all else 'At least one chain is NOT nested: no ranking rule can be optimal at every size.'}")

    # ---- W2 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("W2  WHAT THE BEST SUBSET KNOWS THAT GREEDY DOES NOT"); P(RULE)
    culprits = []
    for eps in EPS_LEVELS:
        for lab, tol in TOLS:
            h = (np.abs(Yl[eps] - ly) <= tol)
            for m in range(N_RATES + 1):
                cols = np.where(sizes == m)[0]
                fr = h[:, cols].mean(axis=0)
                b = int(cols[int(np.argmax(fr))])
                short = float(fr.max() - h[:, gm_S[m]].mean())
                d = h[:, b].astype(float) - h[:, gm_S[m]].astype(float)
                se = float(d.std(ddof=1) / np.sqrt(N_TRIALS)) if d.std() > 0 else 0.0
                if short > max(0.03, 3 * se):
                    gset, bset = members(gm_S[m]), members(b)
                    miss = sorted(bset - gset)
                    extra = sorted(gset - bset)
                    culprits.extend(miss)
                    P(f"  eps={eps} {lab} m={m}: shortfall {short:.4f} ({short/max(se,1e-12):.1f} se)")
                    P(f"      best measures instead: {miss}   greedy wasted its budget on: {extra}")
    P(f"  rates greedy under-rates: {sorted(set(culprits))}")
    for nm in sorted(set(culprits)):
        P(f"      {nm}: |S| = {abs(Sd[nm]):.4f}, rank {order_S.index(nm)+1} of {N_RATES},"
          f" base value {CANDIDATE[nm]}")

    # ---- W3 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("W3  IS THE LOCAL DERIVATIVE THE CULPRIT?"); P(RULE)
    P("  Realised damage: perturb ONE rate by the actual draws and take sd of log10(Y_hat/Y_true).")
    dmg = {}
    for eps in EPS_LEVELS:
        sigma = eps * ORDERS_PER_KCAL
        zp = np.random.default_rng(SEED + 1).standard_normal((N_PROBE, N_RATES))
        d = {}
        for k, nm in enumerate(NAMES):
            vals = []
            for t in range(N_PROBE):
                r = dict(CANDIDATE); r[nm] = CANDIDATE[nm] * 10.0 ** (sigma * zp[t, k])
                vals.append(np.log10(max(eradication(r, t_on=T_ON, t_off=T_OFF, **kw), 1e-300)) - ly)
            d[nm] = float(np.std(vals, ddof=1))
        dmg[eps] = d
        ordD = sorted(NAMES, key=lambda n: -d[n])
        P(f"\n  eps = {eps} kcal/mol")
        P(f"  {'rate':>9}{'|S| local':>12}{'|S| rank':>10}{'realised sd':>14}{'damage rank':>13}")
        for nm in ordD:
            P(f"  {nm:>9}{abs(Sd[nm]):>12.4f}{order_S.index(nm)+1:>10}"
              f"{d[nm]:>14.4f}{ordD.index(nm)+1:>13}")
        P(f"  |S| order      : {order_S}")
        P(f"  damage order   : {ordD}")
        moved = [nm for nm in NAMES if order_S.index(nm) != ordD.index(nm)]
        P(f"  rates whose rank MOVED: {moved}")
        hit = sorted(set(culprits)) and all(nm in moved for nm in set(culprits))
        P(f"  every rate W2 blamed also moved rank: {bool(hit)}"
          f"   -> local-derivative explanation {'CONFIRMED' if hit else 'REFUTED, needs another cause'}")

    # ---- W4 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("W4  DOES THE FIX CLOSE THE GAP?"); P(RULE)
    worst_S, worst_D = 0.0, 0.0
    mstar = {}
    for eps in EPS_LEVELS:
        ordD = sorted(NAMES, key=lambda n: -dmg[eps][n])
        gm_D = order_masks(ordD)
        for lab, tol in TOLS:
            h = (np.abs(Yl[eps] - ly) <= tol)
            gS = np.array([h[:, gm_S[m]].mean() for m in range(N_RATES + 1)])
            gD = np.array([h[:, gm_D[m]].mean() for m in range(N_RATES + 1)])
            bv = np.array([h[:, np.where(sizes == m)[0]].mean(axis=0).max()
                           for m in range(N_RATES + 1)])
            worst_S = max(worst_S, float((bv - gS).max()))
            worst_D = max(worst_D, float((bv - gD).max()))
            msS = next((m for m in range(N_RATES + 1) if gS[m] >= TARGET), None)
            msD = next((m for m in range(N_RATES + 1) if gD[m] >= TARGET), None)
            mstar[(eps, lab)] = (msS, msD)
            P(f"\n  eps = {eps} kcal/mol, tolerance {lab}")
            P(f"  {'m':>3}{'greedy |S|':>12}{'greedy damage':>15}{'best':>9}"
              f"{'shortfall |S|':>15}{'shortfall damage':>18}")
            for m in range(N_RATES + 1):
                P(f"  {m:>3}{gS[m]:>12.4f}{gD[m]:>15.4f}{bv[m]:>9.4f}"
                  f"{bv[m]-gS[m]:>15.4f}{bv[m]-gD[m]:>18.4f}")
            P(f"  m* by |S| = {msS}, m* by realised damage = {msD}")
    P(f"\n  worst shortfall under |S| ranking            : {worst_S:.4f}"
      f"   {'within' if worst_S <= 0.03 else 'BEYOND'} the 0.03 band")
    P(f"  worst shortfall under realised-damage ranking: {worst_D:.4f}"
      f"   {'within' if worst_D <= 0.03 else 'BEYOND'} the 0.03 band")
    P(f"  {'FIX CONFIRMED' if worst_D <= 0.03 else 'FIX INCOMPLETE -- residual reported above'}")

    # ---- W6 -----------------------------------------------------------------------------------
    P("\n" + RULE); P("W6  THE COST OF THE FIX"); P(RULE)
    cheap = N_RATES * N_PROBE
    dear = N_SUB * N_TRIALS
    P(f"  realised-damage ranking : {N_RATES} rates x {N_PROBE} draws = {cheap} evaluations")
    P(f"  exhaustive enumeration  : {N_SUB} subsets x {N_TRIALS} draws = {dear} evaluations")
    P(f"  ratio {dear/cheap:.0f}x, and the cheap one is linear in the number of rates while the")
    P(f"  dear one is exponential -- 2^n against n. At 20 rates the enumeration is 1e6 subsets.")

    open(os.path.join(os.path.dirname(__file__), "RESULTS_hybrid_why.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
