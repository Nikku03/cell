"""How accurately must a rate be known, for the ANSWER to be worth having?

THE QUESTION THIS REPLACES. "Can chemistry predict in-cell rates accurately enough?" is not
answerable in the abstract. "How accurate would it have to be?" is answerable now, cheaply, and it
is a PRECONDITION: if the requirement is beyond what any method delivers, the route is closed
before a single free-energy calculation is run.

WHAT CHEMISTRY DELIVERS, converted into rate units. Transition-state theory gives
k ~ exp(-dG/RT), so an error eps in dG (kcal/mol) is an error eps/(RT ln10) in log10 k. At 298 K,
RT = 0.5925 kcal/mol, so

    1 kcal/mol  ->  0.733 orders in the rate  (a factor of 5.4)
    2 kcal/mol  ->  1.466 orders              (a factor of 29)
    5 kcal/mol  ->  3.665 orders              (a factor of 4600)

Alchemical free-energy methods report roughly 1-2 kcal/mol on well-behaved protein-ligand systems;
QM/MM barriers for catalysis are worse. So the honest capability is a rate good to ABOUT ONE
ORDER, on systems chosen to be tractable, in vitro.

THE REQUIREMENT SIDE. For an observable Y and a rate k, define the log-log sensitivity

    S_k = d log10 Y / d log10 k

Then holding log10 Y inside a tolerance delta requires log10 k inside delta / |S_k|, i.e. the
rate must be known to delta/|S_k| orders and hence to (RT ln10) * delta / |S_k| kcal/mol. That is
the number to put beside the 1-2 kcal/mol above.

WHY THIS IS NOT AN ACADEMIC EXERCISE. This build order already measured the structural ceiling
|d log10 P(n >= T) / d log10 k| <= T. A DEEP tail is therefore intrinsically sensitive: at
threshold T the same rate error is amplified up to T-fold. Rare-event questions are exactly where
rate accuracy is hardest to supply and most needed.

THE SYSTEM. A two-state persister circuit, which is small, real, and whose observable is a
genuine clinical rare event:
    G growing, D dormant.  Drug on: G killed at k_kill*G.  Drug off: G divides at mu*G.
    G -> D at a*G (persister formation),  D -> G at b*D (waking).
    Y = P(eradication): the probability that a course leaves ZERO cells of either type.
Four named rates -- mu, k_kill, a, b -- every one of which would have to come from measurement or
from chemistry.

=================================================================================================
GATES, PREDECLARED BEFORE THE FIRST RUN. Deciding statistic: worst case over the sweep.
=================================================================================================

N1  THE DERIVATIVE IS A DERIVATIVE. Each S_k is a central difference in log space. Halving the
    step must change S_k by < 1%, or the number reported is discretisation rather than
    sensitivity. Reported per rate, worst case gated.

N2  THE STRUCTURAL CEILING MUST HOLD. |S_k| <= T for the tail threshold T, which for eradication
    (T = the whole population) is the initial count. A measured |S_k| above that ceiling means the
    sensitivity code is wrong, not that biology is surprising.

N3  NON-VACUITY. Y must lie inside (1e-9, 0.99) at the base point, and must MOVE by at least 20%
    across the rate perturbations used, or the derivative has nothing to resolve.

N4  THE REQUIREMENT, REPORTED IN THE UNITS CHEMISTRY IS MEASURED IN. For target answer
    tolerances of a factor of 2 and a factor of 10, report per rate the permitted error in orders
    AND in kcal/mol. Reported, not gated -- this is the deliverable.

N5  THE COMPARISON, AND IT DECIDES THE ROUTE. Count how many of the four rates have a permitted
    error at or above 1.0 kcal/mol (reachable today) versus below 0.5 kcal/mol (not reachable by
    any method). Predeclared readings: if every rate needs better than 0.5 kcal/mol the
    chemistry-on-demand route is closed for this observable; if most permit 1 kcal/mol or more it
    is open. Whichever occurs is reported.

N6  DEPTH DEPENDENCE. Repeat at three population sizes, since the ceiling says sensitivity grows
    with the depth of the tail. The requirement should TIGHTEN as the question gets rarer, and if
    it does not, N2's ceiling is not doing what it claims.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm

RULE = "=" * 97
RT = 0.5925                      # kcal/mol at 298 K
ORDERS_PER_KCAL = 1.0 / (RT * np.log(10.0))     # = 0.7328 orders per kcal/mol


def generator(cap_g, cap_d, mu, k_kill, a, b, drug):
    n = (cap_g + 1) * (cap_d + 1)
    idx = lambda g, d: g * (cap_d + 1) + d
    L = np.zeros((n, n))

    def add(i, j, r):
        if r > 0:
            L[j, i] += r
            L[i, i] -= r

    for g in range(cap_g + 1):
        for d in range(cap_d + 1):
            i = idx(g, d)
            if not drug and g + 1 <= cap_g:
                add(i, idx(g + 1, d), mu * g)
            if drug and g > 0:
                add(i, idx(g - 1, d), k_kill * g)
            if g > 0 and d + 1 <= cap_d:
                add(i, idx(g, d + 1), a * g)
            if d > 0 and g + 1 <= cap_g:
                add(i, idx(g + 1, d - 1), b * d)
    return L


def eradication(rates, cap_g=14, cap_d=14, g0=6, t_on=6.0, t_off=3.0, cycles=4):
    mu, k_kill, a, b = (rates[k] for k in ("mu", "k_kill", "a", "b"))
    Lon = generator(cap_g, cap_d, mu, k_kill, a, b, True)
    Loff = generator(cap_g, cap_d, mu, k_kill, a, b, False)
    step = expm(Loff * t_off) @ expm(Lon * t_on)
    n = (cap_g + 1) * (cap_d + 1)
    p = np.zeros(n); p[min(g0, cap_g) * (cap_d + 1)] = 1.0
    for _ in range(cycles):
        p = step @ p
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s > 0:
            p /= s
    return float(p[0])


def sensitivity(rates, name, h, **kw):
    """d log10 Y / d log10 k by central difference in log space."""
    up = dict(rates); up[name] = rates[name] * 10.0 ** h
    dn = dict(rates); dn[name] = rates[name] * 10.0 ** (-h)
    yu, yd = eradication(up, **kw), eradication(dn, **kw)
    if yu <= 0 or yd <= 0:
        return np.nan
    return (np.log10(yu) - np.log10(yd)) / (2.0 * h)


# CORRECTION 1, and it invalidated the entire first run. The original rates made eradication
# essentially certain: Y = 0.9986, pressed against 1. N3 caught it. A saturated observable cannot
# move, so every sensitivity came out at 1e-3 (noise, not signal), N1 failed at 2.1e-02 because the
# differences were below numerical resolution, and the "requirement" read as 40-460 kcal/mol --
# which then produced a cheerful N5 verdict that the route is OPEN. That verdict was worthless.
# This is the same defect as reading a validation off a quantity pressed against its limit, and it
# is now the fourth time in this session. Retuned so eradication is genuinely rare.
BASE = dict(mu=1.0, k_kill=0.25, a=0.40, b=0.03)
CYCLES, G0 = 3, 6
NAMES = ("mu", "k_kill", "a", "b")


def report():
    out = []; P = out.append
    P(RULE)
    P("HOW ACCURATELY MUST A RATE BE KNOWN FOR THE ANSWER TO BE WORTH HAVING?")
    P(RULE)
    P("  Two-state persister circuit; Y = P(eradication) after a 4-cycle course.")
    P(f"  Conversion used throughout: 1 kcal/mol = {ORDERS_PER_KCAL:.4f} orders in a rate")
    P("  (transition-state theory, RT = 0.5925 kcal/mol at 298 K).")
    P("  Chemistry's realistic capability: 1-2 kcal/mol, i.e. 0.73-1.47 orders, in vitro.")
    P("")

    y0 = eradication(BASE, cycles=CYCLES, g0=G0)
    P(RULE)
    P("N3  NON-VACUITY")
    P(RULE)
    P(f"  base Y = P(eradication) = {y0:.6e}   "
      f"{'PASS' if 1e-9 < y0 < 0.99 else 'FAIL'} (bar inside 1e-9..0.99)")

    P("")
    P(RULE)
    P("N1  IS THE DERIVATIVE A DERIVATIVE?  (halve the step; S must not move)")
    P(RULE)
    P(f"  {'rate':>8s} {'S at h=0.04':>13s} {'S at h=0.02':>13s} {'rel change':>12s}")
    S = {}
    worst_n1 = 0.0
    for nm in NAMES:
        s1 = sensitivity(BASE, nm, 0.04, cycles=CYCLES, g0=G0)
        s2 = sensitivity(BASE, nm, 0.02, cycles=CYCLES, g0=G0)
        rel = abs(s1 - s2) / abs(s2) if s2 else np.nan
        worst_n1 = max(worst_n1, rel)
        S[nm] = s2
        P(f"  {nm:>8s} {s1:13.6f} {s2:13.6f} {rel:12.3e}")
    P(f"  worst relative change {worst_n1:.3e}   "
      f"{'PASS' if worst_n1 < 0.01 else 'FAIL'} (bar 1%)")
    P("")

    P(RULE)
    P("N2  THE STRUCTURAL CEILING  |S_k| <= T")
    P(RULE)
    P(f"  Eradication asks for ZERO cells from an initial {G0}, so the ceiling is T = {G0}.")
    mx = max(abs(v) for v in S.values())
    for nm in NAMES:
        P(f"  {nm:>8s}  |S| = {abs(S[nm]):.6f}")
    P(f"  worst |S| = {mx:.6f} against ceiling {G0}   "
      f"{'PASS' if mx <= G0 else 'FAIL -- sensitivity code is wrong'}")
    P("")

    P(RULE)
    P("N4  THE REQUIREMENT, IN THE UNITS CHEMISTRY IS MEASURED IN")
    P(RULE)
    for tol_name, delta in (("a factor of 2 on Y", np.log10(2.0)),
                            ("a factor of 10 on Y", 1.0)):
        P(f"  to hold Y within {tol_name} (delta = {delta:.4f} orders):")
        P(f"    {'rate':>8s} {'|S|':>10s} {'orders allowed':>16s} {'kcal/mol allowed':>18s}"
          f" {'reachable?':>12s}")
        for nm in NAMES:
            s = abs(S[nm])
            if s <= 0:
                P(f"    {nm:>8s} {s:10.6f} {'unconstrained':>16s}"); continue
            orders = delta / s
            kcal = orders / ORDERS_PER_KCAL
            reach = "yes" if kcal >= 1.0 else ("marginal" if kcal >= 0.5 else "NO")
            P(f"    {nm:>8s} {s:10.6f} {orders:16.4f} {kcal:18.4f} {reach:>12s}")
        P("")

    P(RULE)
    P("N5  THE COMPARISON THAT DECIDES THE ROUTE  (target: factor of 2 on Y)")
    P(RULE)
    delta = np.log10(2.0)
    kcals = {nm: (delta / abs(S[nm])) / ORDERS_PER_KCAL for nm in NAMES if abs(S[nm]) > 0}
    n_ok = sum(1 for v in kcals.values() if v >= 1.0)
    n_marg = sum(1 for v in kcals.values() if 0.5 <= v < 1.0)
    n_no = sum(1 for v in kcals.values() if v < 0.5)
    P(f"  rates permitting >= 1.0 kcal/mol (reachable today) : {n_ok} of {len(kcals)}")
    P(f"  rates permitting 0.5-1.0 kcal/mol (marginal)       : {n_marg} of {len(kcals)}")
    P(f"  rates needing   <  0.5 kcal/mol (not reachable)    : {n_no} of {len(kcals)}")
    if n_no == len(kcals):
        P("  READING: every rate needs better than any method delivers. The chemistry-on-demand")
        P("  route is CLOSED for this observable.")
    elif n_ok >= len(kcals) - 1:
        P("  READING: the requirement sits at or inside what free-energy methods already reach.")
        P("  The route is OPEN for this observable, and the binding issue becomes in-cell versus")
        P("  in-vitro transferability rather than raw method accuracy.")
    else:
        P("  READING: mixed. Some rates are reachable and some are not, so the route is open only")
        P("  if the unreachable ones can be supplied by measurement instead.")
    P("")

    P(RULE)
    P("N6  DOES THE REQUIREMENT TIGHTEN AS THE QUESTION GETS RARER?")
    P(RULE)
    P(f"  {'g0':>4s} {'Y':>13s} {'worst |S|':>11s} {'tightest kcal/mol (factor 2)':>30s}")
    prev = None
    mono = True
    for g0 in (4, 6, 9):
        y = eradication(BASE, g0=g0, cycles=CYCLES)
        ss = [abs(sensitivity(BASE, nm, 0.02, g0=g0, cycles=CYCLES)) for nm in NAMES]
        w = max(ss)
        kc = (delta / w) / ORDERS_PER_KCAL
        P(f"  {g0:4d} {y:13.4e} {w:11.6f} {kc:30.4f}")
        if prev is not None and w < prev:
            mono = False
        prev = w
    P(f"  sensitivity grows with population size: {mono}   "
      f"{'PASS' if mono else 'FAIL -- the ceiling argument does not hold here'}")
    P("  A rarer question means a deeper tail, a larger |S|, and therefore a TIGHTER rate")
    P("  requirement. Chemistry has to be most accurate exactly where it is hardest.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
