"""Can the machinery find the knob WITHOUT being told which knob it is?

THE SETUP. A model predicts a rare-event rate; a measurement disagrees by nine and a half orders
of magnitude; and a human (me) then looked at the parameter table, noticed that the burst
frequency was measured at 37 C while the switching rate was measured at 30 C, and declared the
discrepancy a conditions mismatch. That is a story told after seeing the answer. This module asks
whether the same conclusion is reachable MECHANICALLY, from the provenance table alone, with no
hint about which parameter is the culprit.

THE METHOD. Minimal-perturbation attribution. Each parameter gets a prior width IN DEX derived
from its provenance, not from taste:

    a MEASURED parameter that was retrieved      -> width from its own quoted error bar
    a MEASURED parameter that was NOT retrieved  -> width from the bracket used in its place,
                                                    treated as uniform, plus HARD BOUNDS
    a NOT STATED quantity (assay bias, missing
    route)                                       -> NO WIDTH AT ALL

The last line is the whole discipline, carried over from the factor budget: a gap in the record
is not an explanation. A knob with no provenance-derived width would otherwise reconcile anything
at zero cost and win every time. So such knobs are reported as SUFFICIENT BUT UNCOSTABLE and are
structurally forbidden from being ranked or selected.

For every costed knob the module solves EXACTLY (not by linearisation) for the multiplicative
change that reconciles model with measurement, rejects it if it leaves the knob's hard bounds,
and reports the cost in units of that knob's own prior width. The winner is the cheapest.

=================================================================================================
THE GATES, FIXED HERE BEFORE ANY NUMBER IS RUN.
=================================================================================================

K0  VACUITY GUARD. The question "which knob" is meaningless if only one knob can reconcile. The
    run must first confirm that AT LEAST TWO distinct single-knob perturbations each reach the
    measurement. If not, the attribution is tautological and every gate below returns VOID.
    (Ledger defect P, in the form of a test that can only return one answer.)

K1  BLIND ATTRIBUTION. Run the method on the provenance table AS IT ACTUALLY EXISTS in
    rem/lysogen.py -- two axes, origin and retrieval, no conditions field. PASS if the cheapest
    costed explanation is the burst frequency or the doubling time, at a cost the method itself
    would call plausible (<= 2 prior widths). This is the gate that answers the question asked.

K2  DEGENERACY HONESTY. If a second costed explanation lies within 1 prior width of the winner,
    the method must return AMBIGUOUS rather than a winner. A ranking that hides a near-tie is
    worse than no ranking.

K3  PLANT-AND-RECOVER CONTROL, without which K1 means nothing.
    K3a  perturb a costed knob by a known amount inside its prior, regenerate the observable,
         and require the method to name that knob and recover its size to within 0.05 dex at a
         cost inside 2 widths.
    K3b  perturb the system by an UNCOSTABLE route instead (an additive rate the model does not
         contain), regenerate the observable, and require the method NOT to cheaply misattribute
         it: the cheapest costed explanation must come back with cost > 2 widths. A method that
         confidently blames a measured parameter for a missing mechanism is worse than useless.

K4  AMPLIFICATION STRUCTURE. The sensitivity d log S / d log k_on must equal -N, the burst count
    itself. If it does, the method has independently rediscovered the crowding error-bar result
    (rare_error ~ exp[(N - <X>) * Delta]) as a derivative rather than as an assumption, and the
    nine orders are a sensitivity of -40, not a mystery.

K5  THE THIRD AXIS. If K1 fails, the run must report WHAT WOULD HAVE TO BE TRUE for it to pass --
    i.e. the prior width the culprit parameter would need -- and name that as a specific missing
    entry in the record rather than as a conclusion. A required-but-unmeasured width is a gap,
    and by the same rule as the uncostable knobs it may not be spent.

=================================================================================================
WHAT IS DELIBERATELY NOT GIVEN TO THE METHOD.
=================================================================================================

The method is not told that k_on was measured at 37 C and applied at 30 C. It is not told the
answer, the direction, or that a conditions mismatch exists as a category. It receives the model,
the measurement, and the two-axis provenance table, and nothing else. That is the point: if the
conclusion only appears once a human has already seen it, the pipeline did not find it.
"""
from __future__ import annotations

import math

UNCOSTABLE = None       # a knob with no provenance-derived width; may never be ranked


class Knob:
    """One adjustable quantity, its provenance-derived prior width, and its hard bounds.

    `sigma_dex is None` marks a NOT STATED quantity. Such a knob can still be reported as
    sufficient, and is structurally barred from being selected -- `cost()` returns inf rather
    than a small number, so it can never win by being unconstrained.
    """

    def __init__(self, name, nominal, sigma_dex, bounds, provenance):
        self.name, self.nominal = name, float(nominal)
        self.sigma_dex = sigma_dex
        self.lo, self.hi = bounds
        self.provenance = provenance

    @property
    def costable(self):
        return self.sigma_dex is not None

    def cost(self, value):
        """Cost of moving to `value`, in prior widths. inf outside bounds or without a prior."""
        if value <= 0 or not (self.lo <= value <= self.hi):
            return float("inf")
        if not self.costable:
            return float("inf")
        return abs(math.log10(value / self.nominal)) / self.sigma_dex

    def dex(self, value):
        return math.log10(value / self.nominal)


def uniform_sigma_dex(lo, hi):
    """Prior width for a bracket used in place of an unretrieved value: uniform on log scale."""
    return (math.log10(hi) - math.log10(lo)) / math.sqrt(12.0)


def solve_knob(model, theta, knob, target, n_iter=200):
    """Exact 1-D solve: the value of `knob` that makes model(theta) equal `target`.

    Bisection in log space over the knob's hard bounds. Returns None when no value inside the
    bounds reaches the target -- which is a real answer (the knob is INSUFFICIENT), not a failure.
    """
    def f(v):
        t = dict(theta)
        t[knob.name] = v
        return math.log10(model(t)) - math.log10(target)

    a, b = knob.lo, knob.hi
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)) or fa * fb > 0:
        return None
    for _ in range(n_iter):
        m = math.sqrt(a * b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return math.sqrt(a * b)


def sensitivity(model, theta, name, eps=1e-4):
    """d log10(observable) / d log10(parameter), by central difference in log space."""
    v = theta[name]
    up, dn = dict(theta), dict(theta)
    up[name] = v * (1.0 + eps)
    dn[name] = v * (1.0 - eps)
    return (math.log10(model(up)) - math.log10(model(dn))) / (
        math.log10(up[name]) - math.log10(dn[name]))


def attribute(model, theta, knobs, target, ambiguity_widths=1.0, plausible_widths=2.0):
    """Rank single-knob reconciliations by cost. Uncostable knobs are listed, never ranked."""
    costed, uncostable, insufficient = [], [], []
    for k in knobs:
        v = solve_knob(model, theta, k, target)
        if v is None:
            insufficient.append({"knob": k.name, "reason": "no value inside hard bounds reaches "
                                                           "the measurement",
                                 "bounds": (k.lo, k.hi), "provenance": k.provenance})
            continue
        rec = {"knob": k.name, "value": v, "dex": k.dex(v), "cost": k.cost(v),
               "provenance": k.provenance,
               "sensitivity": sensitivity(model, theta, k.name)}
        (costed if k.costable else uncostable).append(rec)

    costed.sort(key=lambda r: r["cost"])
    n_sufficient = len(costed) + len(uncostable)
    verdict = {"n_sufficient": n_sufficient, "vacuous": n_sufficient < 2}
    if costed:
        best = costed[0]
        near = [r for r in costed[1:] if r["cost"] - best["cost"] <= ambiguity_widths]
        verdict["best"] = best
        verdict["ambiguous"] = bool(near)
        verdict["ties"] = [r["knob"] for r in near]
        verdict["plausible"] = best["cost"] <= plausible_widths
    else:
        verdict["best"] = None
        verdict["ambiguous"] = False
        verdict["ties"] = []
        verdict["plausible"] = False
    verdict["costed"] = costed
    verdict["uncostable"] = uncostable
    verdict["insufficient"] = insufficient
    return verdict


def required_width(best, plausible_widths=2.0):
    """K5: the prior width the winning knob would NEED for its move to count as plausible."""
    if best is None:
        return None
    return abs(best["dex"]) / plausible_widths


# =================================================================================================
# THE LYSOGEN INSTANCE
# =================================================================================================

LN2 = math.log(2.0)


def burst_model(theta):
    """S = f_assay * exp(-k_on * mu * tau / ln2) + k_extra. The last two default to no-op."""
    n = theta["k_on"] * theta.get("mu", 1.0) * theta["tau"] / LN2
    return theta.get("f_assay", 1.0) * math.exp(-n) + theta.get("k_extra", 0.0)


S_OBS = 9.0e-9              # Zong Table 1, lambda-IG831 wild-type cI, RecA- host JL5902
K_ON_NOMINAL = 1.4          # /min, MEASURED 1.4 +/- 0.2, retrieved
K_ON_SIGMA_DEX = math.log10(1.6 / 1.4)
TAU_LO, TAU_HI = 20.0, 60.0
TAU_NOMINAL = TAU_LO        # the model's BEST case: any other choice makes the required move
                            # larger, so starting here is conservative toward the model


def lysogen_knobs():
    """The provenance table as it ACTUALLY EXISTS -- two axes, no conditions field."""
    return [
        Knob("k_on", K_ON_NOMINAL, K_ON_SIGMA_DEX, (0.01, 10.0),
             "MEASURED, retrieved: 1.4 +/- 0.2 /min (smFISH)"),
        Knob("tau", TAU_NOMINAL, uniform_sigma_dex(TAU_LO, TAU_HI), (TAU_LO, TAU_HI),
             "MEASURED, UNRETRIEVED: replaced by a 20-60 min bracket"),
        Knob("f_assay", 1.0, UNCOSTABLE, (1e-12, 1e12),
             "NOT STATED: no source quantifies an assay bias"),
        Knob("k_extra", 1e-30, UNCOSTABLE, (1e-30, 1.0),
             "NOT STATED: RecA-independent routes exist (Rozanov) but no baseline magnitude"),
    ]


def theta0():
    return {"k_on": K_ON_NOMINAL, "tau": TAU_NOMINAL, "mu": 1.0,
            "f_assay": 1.0, "k_extra": 1e-30}


def _line(r, width=2.0):
    tag = "PLAUSIBLE" if r["cost"] <= width else "IMPLAUSIBLE"
    return (f"    {r['knob']:<10s} -> {r['value']:>10.4g}  ({r['dex']:+.3f} dex)   "
            f"cost {r['cost']:>6.2f} widths  {tag}\n"
            f"               sensitivity dlogS/dlog{r['knob']} = {r['sensitivity']:+.1f}\n"
            f"               {r['provenance']}")


def verify():
    th, kn = theta0(), lysogen_knobs()
    pred = burst_model(th)
    res = attribute(burst_model, th, kn, S_OBS)

    print("=" * 96)
    print("BLIND ATTRIBUTION -- the method is not told which parameter is suspect")
    print("=" * 96)
    print(f"  model prediction at nominal: {pred:.3e}")
    print(f"  measurement:                 {S_OBS:.3e}")
    print(f"  discrepancy: {math.log10(S_OBS / pred):.2f} orders")

    print("\n  K0  VACUITY GUARD")
    print(f"      distinct single-knob reconciliations found: {res['n_sufficient']}")
    if res["vacuous"]:
        print("      VOID -- only one knob can reconcile; the question is tautological.")
        return res
    print("      PASS -- more than one knob can reach the measurement, so the ranking is a "
          "choice.")

    print("\n  COSTED EXPLANATIONS (ranked)")
    for r in res["costed"]:
        print(_line(r))
    print("\n  SUFFICIENT BUT UNCOSTABLE (listed, structurally barred from ranking)")
    for r in res["uncostable"]:
        print(f"    {r['knob']:<10s} -> {r['value']:>10.4g}  ({r['dex']:+.3f} dex)   "
              f"cost UNDEFINED\n               {r['provenance']}")
    if res["insufficient"]:
        print("\n  INSUFFICIENT (cannot reach the measurement inside its own bounds)")
        for r in res["insufficient"]:
            print(f"    {r['knob']:<10s} bounds {r['bounds']}  -- {r['reason']}\n"
                  f"               {r['provenance']}")

    best = res["best"]
    print("\n  K1  BLIND ATTRIBUTION")
    if best is None:
        print("      FAIL -- no costed explanation exists at all.")
    else:
        named = best["knob"] in ("k_on", "tau")
        print(f"      cheapest costed explanation: {best['knob']} at "
              f"{best['cost']:.2f} prior widths")
        print(f"      names the burst frequency or doubling time: {named}")
        print(f"      cost inside 2 widths (the method's own plausibility bar): "
              f"{res['plausible']}")
        print(f"      K1 {'PASS' if (named and res['plausible']) else 'FAIL'}")

    print("\n  K2  DEGENERACY HONESTY")
    print(f"      near-ties within 1 width: {res['ties'] or 'none'}  -> "
          f"{'AMBIGUOUS' if res['ambiguous'] else 'unambiguous'}")

    print("\n  K4  AMPLIFICATION STRUCTURE")
    n_nom = th["k_on"] * th["tau"] / LN2
    s_k = sensitivity(burst_model, th, "k_on")
    print(f"      N at nominal = {n_nom:.2f};  dlogS/dlog k_on = {s_k:.2f}")
    ok = abs(s_k + n_nom) < 0.05 * n_nom
    print(f"      sensitivity equals -N: {ok}  -> K4 {'PASS' if ok else 'FAIL'}")
    print("      the nine orders are a derivative of -40, not a mystery: this is")
    print("      rem/crowding_errorbar.py's exp[(N-<X>)*Delta] recovered as a slope.")

    print("\n  K5  WHAT WOULD HAVE TO BE TRUE")
    if best is not None and not res["plausible"]:
        need = required_width(best)
        print(f"      for {best['knob']} to be a plausible culprit its prior width would have to")
        print(f"      be >= {need:.3f} dex (a factor of {10 ** need:.2f}), against the "
              f"{best['cost'] and abs(best['dex']) / best['cost']:.3f} dex it actually carries.")
        print("      NO SOURCE STATES SUCH A WIDTH. By the same rule as the uncostable knobs,")
        print("      a required-but-unmeasured width is a gap in the record and may not be")
        print("      spent as an explanation.")
    return res


def plant_and_recover(true_knob, factor, seed_theta=None):
    """K3a: perturb a costed knob, regenerate the observable, ask the method to find it."""
    th = seed_theta or theta0()
    truth = dict(th)
    truth[true_knob] = th[true_knob] * factor
    target = burst_model(truth)
    res = attribute(burst_model, th, lysogen_knobs(), target)
    return res, target, math.log10(factor)


def plant_uncostable(rate, seed_theta=None):
    """K3b: perturb by a route the model does not contain, and check for misattribution."""
    th = seed_theta or theta0()
    truth = dict(th)
    truth["k_extra"] = rate
    target = burst_model(truth)
    res = attribute(burst_model, th, lysogen_knobs(), target)
    return res, target


def verify_controls():
    print("\n" + "=" * 96)
    print("K3  PLANT-AND-RECOVER CONTROLS")
    print("=" * 96)
    res, target, true_dex = plant_and_recover("k_on", 0.85)
    best = res["best"]
    got = best["dex"] if best else float("nan")
    ok_a = (best is not None and best["knob"] == "k_on"
            and abs(got - true_dex) < 0.05 and best["cost"] <= 2.0)
    print(f"  K3a planted k_on x0.85 ({true_dex:+.3f} dex) -> observable {target:.3e}")
    print(f"      method names '{best['knob'] if best else None}' at {got:+.3f} dex, "
          f"cost {best['cost'] if best else float('nan'):.2f} widths")
    print(f"      K3a {'PASS' if ok_a else 'FAIL'}")

    res_b, target_b = plant_uncostable(3.0e-9)
    best_b = res_b["best"]
    ok_b = best_b is None or best_b["cost"] > 2.0
    print(f"\n  K3b planted an additive route at 3.0e-9 (a mechanism the model lacks) "
          f"-> observable {target_b:.3e}")
    if best_b:
        print(f"      cheapest costed explanation: {best_b['knob']} at "
              f"{best_b['cost']:.2f} widths ({best_b['dex']:+.3f} dex)")
    print(f"      method refuses to cheaply blame a measured parameter: {ok_b}")
    print(f"      K3b {'PASS' if ok_b else 'FAIL'}")
    return ok_a, ok_b


if __name__ == "__main__":
    verify()
    verify_controls()
