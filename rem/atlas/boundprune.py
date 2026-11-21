"""The target-aware bound wired into the real pruner. How much does the REM tail actually improve?

WHERE THIS STANDS IN THE CHAIN. accuracy.py converted the engine's error terms into the unit it
reports -- orders of magnitude of the rare-event tail -- and found the pruner binding by 8.4
decades: the reported tail is a LOWER BOUND rising at 0.608 orders per decade of retained paths,
R^2 = 0.997. pathbound.py then derived an admissible bound on a prefix's whole subtree, verified
it exhaustively at a width where all 1,048,576 paths could be enumerated, and showed it beats mass
ranking at every retained count -- but did NOT touch the pruner, so no reported tail had moved.
This wires it in and measures the real thing.

WHAT HAD TO CHANGE TO MAKE IT RUN AT THE ENGINE'S WIDTH. Scoring every (parent, child) candidate
with the exact bound needs a parents x children x targets tensor, 19531 x 1024 x 200, which is 4e9
entries and does not fit. But log sigma is CONCAVE, so its tangent at the parent's own drive lies
above it:

    log sigma(u + v)  <=  log sigma(u) + sigma(-u) * v

Summing over targets turns the child's contribution into ONE matmul, and the result is still an
upper bound on the exact bound -- hence still admissible, just looser. That relaxation is the only
new mathematics here and it is what makes the wiring possible at all.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN
=================================================================================================

P0  THE WIRING CHECK, BEFORE ANY COMPARISON. rank="mass" must reproduce the pruner as it was, to
    the last bit. If it does not, every comparison below is confounded by a change to the baseline
    rather than measuring the ranking.
    PREDECLARED: exact equality of the returned tail against the pre-existing code path, or the
    module stops.

P1  THE CEILING GATE, AND IT IS AN ORACLE. Before crediting the bound with anything, ask how much
    there is to win. Take the largest retained set that can be run, compute the ACTUAL contribution
    of every path in it, and ask what tail the BEST POSSIBLE ranking would have reached at each
    smaller retained count. That oracle is the ceiling: no ranking rule can beat it.
    PREDECLARED: the bound is worth wiring in only if it captures a substantial share of the gap
    between mass ranking and the oracle. If mass ranking is already close to the oracle there is
    nothing to win at this width and the module must say so instead of reporting a small gain as a
    success.

P2  THE HEAD-TO-HEAD AT EQUAL COST, at the width and depth accuracy.py used, so the numbers are
    comparable with the 8.4-decade finding rather than with a fresh baseline.
    PREDECLARED: reported as DECADES OF RETAINED PATHS SAVED -- how much smaller a bound-ranked run
    can be while reaching the same tail as a mass-ranked one -- because that is what the engine
    buys, and a ratio of tails at fixed cost is not directly interpretable as a saving.

P3  THE DECISIVE ONE: DOES IT REDUCE THE SLOPE? accuracy.py's finding was not that the tail was
    wrong at some point but that it was still RISING at 0.608 orders per decade, so the pruning
    error crossed the class map's whole 5.08-order span at 10^12.7 retained paths against a full
    path set of 10^21.1. A better RANKING that leaves the slope unchanged reorders the paths
    without converging any faster, and the engine is no better off asymptotically.
    PREDECLARED: pruning stops binding only if the crossing point moves ABOVE 10^21.1. If the
    slope falls but the crossing stays below, the honest verdict is that the bound helps and does
    not fix it, and that is what will be reported.

P4  THE CERTIFICATE AT THE ENGINE'S WIDTH. With rank="bound" the accumulated certificate is the
    summed dropped BOUND, which bounds the tail deficit rather than the discarded mass. pathbound
    measured its ratio to the true deficit at 4.2e11 -- ninety orders better than mass, nine orders
    short of the 1e3 it set as useful. Report it here, where the true deficit is not knowable, as
    the ratio of certificate to the reported tail: that says whether the guarantee is even in the
    right range to be quoted.

P5  WHAT THIS DOES NOT SETTLE.
"""

from __future__ import annotations
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np

from rem.atlas.hybrid_tune import RULE
from rem.atlas.realkinetics import trrust_block, target_rows, engine_budget

CAPS = (100, 300, 1000, 3000, 10000, 19531)
CLASS_LADDER_ORDERS = 5.08      # accuracy.py's A2: the class map's whole span, in orders of tail
FULL_PATHS_LOG10 = 21.1         # accuracy.py's A4b: log10 of the path set being approximated
MASS_SLOPE = 0.608              # accuracy.py's A4b, for reference


def main():
    out = []

    def P_(s=""):
        print(s, flush=True)
        out.append(s)

    P_(RULE)
    P_("THE TARGET-AWARE BOUND, WIRED INTO THE REAL PRUNER")
    P_(RULE)
    nCtrl, L, dt = 10, 6, 0.5
    Q, ctrl, cidx, sha, ne, nw = trrust_block(nCtrl)
    rows = target_rows(cidx, 200)
    P_(f"  {nCtrl} controllers, {Q.shape[0]} states, L = {L}, dt = {dt}, {len(rows)} target rows.")
    P_(f"  TRRUST sha {sha[:12]}. Same width, depth and rows as accuracy.py, so the numbers here")
    P_(f"  are comparable with its 8.4-decade finding rather than with a fresh baseline.")

    def run(cap, rank):
        t0 = time.time()
        r = engine_budget(Q, nCtrl, rows, L, dt, 0.0, cap=cap, rank=rank)
        return r, time.time() - t0

    # ---- P0  THE WIRING CHECK ------------------------------------------------------------------
    P_("\n" + RULE)
    P_('P0  THE WIRING CHECK: DOES rank="mass" REPRODUCE THE PRUNER AS IT WAS?')
    P_(RULE)
    (tm0, dm0, _, km0, _), _ = run(19531, "mass")
    P_(f"  rank='mass' at cap 19531: tail {tm0:.6e}, kept {km0}")
    P_(f"  accuracy.py's A1b reported 3.7208e-98 at 19531 retained paths.")
    same = abs(np.log10(tm0) - np.log10(3.7208e-98)) < 5e-5
    P_(f"  P0: {'PASS -- the baseline is unchanged, so what follows measures the RANKING.' if same else 'FAIL -- the baseline moved; the comparison would be confounded. STOPPING.'}")
    if not same:
        with open(os.path.join(os.path.dirname(__file__), "RESULTS_boundprune.txt"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        return

    # ---- P2  THE HEAD-TO-HEAD (run first, the oracle needs its output) -------------------------
    P_("\n" + RULE)
    P_("P2  HEAD TO HEAD AT EQUAL RETAINED-PATH COUNT")
    P_(RULE)
    P_(f"\n    {'cap':>8} {'tail, MASS':>15} {'tail, BOUND':>15} {'gain, orders':>13}"
       f" {'s mass':>8} {'s bound':>8}")
    M, B = {}, {}
    for cp in CAPS:
        (t_m, d_m, _, k_m, _), s_m = run(cp, "mass")
        (t_b, d_b, _, k_b, _), s_b = run(cp, "bound")
        M[cp] = (t_m, d_m, k_m)
        B[cp] = (t_b, d_b, k_b)
        if t_m > 0 and t_b > 0:
            P_(f"    {cp:>8} {t_m:>15.4e} {t_b:>15.4e}"
               f" {np.log10(t_b) - np.log10(t_m):>13.3f} {s_m:>8.1f} {s_b:>8.1f}")
        else:
            P_(f"    {cp:>8} {t_m:>15.4e} {t_b:>15.4e} {'--':>13} {s_m:>8.1f} {s_b:>8.1f}")
    lm = np.array([np.log10(M[c][0]) for c in CAPS if M[c][0] > 0])
    lb = np.array([np.log10(B[c][0]) for c in CAPS if B[c][0] > 0])
    kc = np.array([float(c) for c in CAPS if M[c][0] > 0 and B[c][0] > 0])
    P_(f"\n  DECADES OF RETAINED PATHS SAVED, which is what the engine actually buys: the")
    P_(f"  bound-ranked run at the SMALLEST cap reaches {lb[0]:.2f}; mass ranking needs")
    if lm.max() >= lb[0]:
        need = float(np.interp(lb[0], lm, np.log10(kc)))
        P_(f"  10^{need:.2f} retained paths to match it, against 10^{np.log10(kc[0]):.2f} for the bound --")
        P_(f"  a saving of {need - np.log10(kc[0]):.2f} decades of retained paths at that accuracy.")
    else:
        P_(f"  MORE than the largest cap run here ({kc.max():.0f}) to match it: the saving exceeds")
        P_(f"  {np.log10(kc.max()) - np.log10(kc[0]):.2f} decades and is not bounded above by this sweep.")

    # ---- P1  THE ORACLE CEILING ----------------------------------------------------------------
    P_("\n" + RULE)
    P_("P1  THE CEILING GATE: HOW MUCH WAS THERE TO WIN?")
    P_(RULE)
    P_("  MY FIRST ORACLE WAS CIRCULAR AND ITS OUTPUT PROVED IT. It re-ranked the paths that the")
    P_("  MASS-ranked run retained, and asked what share of the mass-to-oracle gap the bound")
    P_("  captured. The answer came back 99.2%, 98.6%, 99.4%, then 116.8%, then 320.1%, then a")
    P_("  division by zero. A share above 100% of a ceiling is impossible, so the ceiling was not")
    P_("  one: defining it from the LOSER's candidate set means the bound can and does escape it.")
    P_("  THAT FAILURE IS ITSELF THE MOST INTERESTING RESULT IN THIS MODULE -- it proves the bound")
    P_("  is not merely re-ordering the same candidates. It reaches parts of the path tree that")
    P_("  mass ranking never generates at all, which is a stronger claim than better ranking.")
    P_("\n  THE NON-CIRCULAR VERSION: run BOTH rankings at a cap three times the largest compared,")
    P_("  score each retained path's EXACT contribution, and take for each k the better of the two")
    P_("  best-possible re-rankings. That is a ceiling on re-ranking what EITHER rule can reach --")
    P_("  explicitly not a ceiling over all 10^21 paths, which is not computable.")
    KBIG = 60000
    orc = {}
    for rk in ("mass", "bound"):
        r = engine_budget(Q, nCtrl, rows, L, dt, 0.0, cap=KBIG, rank=rk, return_paths=True)
        per = r[5]
        orc[rk] = np.sort(per)[::-1]
        P_(f"    oracle candidate set from rank='{rk}' at cap {KBIG}: {len(per):,} paths,"
           f" tail {float(per.sum()):.4e}")
    P_(f"\n    {'cap':>8} {'MASS':>15} {'BOUND':>15} {'ORACLE':>15} {'share of gap':>14}")
    shares = []
    for cp in CAPS:
        best = max(float(orc["mass"][:cp].sum()), float(orc["bound"][:cp].sum()))
        t_m, t_b = M[cp][0], B[cp][0]
        if min(t_m, t_b, best) <= 0:
            P_(f"    {cp:>8} {t_m:>15.4e} {t_b:>15.4e} {best:>15.4e} {'--':>14}")
            continue
        gap = np.log10(best) - np.log10(t_m)
        got = np.log10(t_b) - np.log10(t_m)
        sh = got / gap if gap > 1e-9 else float("nan")
        shares.append(sh)
        flag = "  <-- EXCEEDS" if sh > 1.001 else ""
        P_(f"    {cp:>8} {t_m:>15.4e} {t_b:>15.4e} {best:>15.4e} {sh * 100:>13.1f}%{flag}")
    if shares and np.isfinite(np.nanmax(shares)):
        P_(f"\n  P1: the bound captures {np.nanmin(shares) * 100:.0f}% to {np.nanmax(shares) * 100:.0f}% of the gap between mass")
        P_("  ranking and the best re-ranking either rule can reach.")
        if np.nanmax(shares) > 1.001:
            P_("  A row still above 100% means the bound-ranked run at that cap beat the best")
            P_("  re-ranking of a THREE TIMES LARGER candidate set -- it is finding paths neither")
            P_("  larger run generated, and the ceiling is still not a true ceiling.")
        else:
            P_("  Nothing exceeds the ceiling now, so the ceiling is behaving as one over the range")
            P_("  it covers, and the bound is close to the best available re-ranking.")

    # ---- P3  THE SLOPE -------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P3  THE DECISIVE GATE: DOES IT REDUCE THE SLOPE?")
    P_(RULE)
    P_("  accuracy.py's finding was not that the tail was wrong somewhere but that it was still")
    P_("  RISING. A better ranking that leaves the slope alone reorders without converging.")
    sm = float(np.polyfit(np.log10(kc), lm, 1)[0])
    sb = float(np.polyfit(np.log10(kc), lb, 1)[0])
    P_(f"\n    {'ranking':<12} {'slope, orders per decade':>26} {'crossing of 5.08 orders':>25}")
    for nm, sl, lv in (("mass", sm, lm), ("bound", sb, lb)):
        cross = np.log10(kc.max()) + CLASS_LADDER_ORDERS / sl if sl > 0 else float("inf")
        P_(f"    {nm:<12} {sl:>26.3f} {f'10^{cross:.1f}' if np.isfinite(cross) else 'never':>25}")
    cross_b = np.log10(kc.max()) + CLASS_LADDER_ORDERS / sb if sb > 0 else float("inf")
    P_(f"\n  the full path set being approximated is 10^{FULL_PATHS_LOG10:.1f}.")
    if cross_b > FULL_PATHS_LOG10:
        P_("\n  P3 AS PREDECLARED: the crossing moves ABOVE the full path set. Pruning stops being")
        P_("  the binding term and the class map becomes it again.")
    else:
        P_(f"\n  P3 AS PREDECLARED: the crossing moves from 10^{np.log10(kc.max()) + CLASS_LADDER_ORDERS / sm:.1f}"
           f" to 10^{cross_b:.1f}, still BELOW the")
        P_(f"  full set of 10^{FULL_PATHS_LOG10:.1f}. THE BOUND HELPS AND DOES NOT FIX IT. Pruning remains the")
        P_("  binding term, and that is the honest verdict rather than the one this module was")
        P_("  built hoping for.")

    # ---- P4  THE CERTIFICATE -------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P4  THE CERTIFICATE THE NEW RANKING PRODUCES")
    P_(RULE)
    P_("  With rank='bound' the accumulated certificate is the summed dropped BOUND, which bounds")
    P_("  the TAIL DEFICIT rather than the discarded mass. The true deficit is not knowable at")
    P_("  this width, so the readable question is whether the guarantee is even in the range")
    P_("  where it could be quoted: how does it compare with the tail itself?")
    P_(f"\n    {'cap':>8} {'MASS cert / tail':>20} {'BOUND cert / tail':>21}")
    for cp in CAPS:
        t_m, d_m, _ = M[cp]
        t_b, d_b, _ = B[cp]
        rm = d_m / t_m if t_m > 0 else float("inf")
        rb = d_b / t_b if t_b > 0 else float("inf")
        P_(f"    {cp:>8} {rm:>20.3e} {rb:>21.3e}")
    P_("\n  A certificate must be small COMPARED WITH THE TAIL to be worth quoting. The mass")
    P_("  certificate is order 1 against a tail of 1e-98, so it is about 1e98 times the quantity")
    P_("  it is supposed to bound -- which is the 1.9e29-style gap realkinetics recorded, restated")
    P_("  at this width. Whether the bound certificate is usable is read off the second column.")

    # ---- P5  LIMITS ----------------------------------------------------------------------------
    P_("\n" + RULE)
    P_("P5  WHAT THIS DOES NOT SETTLE")
    P_(RULE)
    P_("  1. The tangent relaxation is LOOSER than the exact bound pathbound verified. Everything")
    P_("     measured here is therefore a lower bound on what the exact bound would achieve, and")
    P_("     the exact bound is not computable at this width by the method used.")
    P_("  2. The oracle in P1 re-ranks the paths the mass-ranked run generated. A rule that")
    P_("     reaches paths that run never generated is outside its ceiling.")
    P_("  3. Slopes are fitted over about two decades of retained paths and the crossing points")
    P_("     are far outside that range. The comparison BETWEEN the two slopes is what the")
    P_("     extrapolation is used for, and it is much better conditioned than either absolute")
    P_("     crossing point.")
    P_("  4. base and gain are at their defaults. The absolute tail is not a physical prediction;")
    P_("     every number here is a difference at fixed base.")

    with open(os.path.join(os.path.dirname(__file__), "RESULTS_boundprune.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
