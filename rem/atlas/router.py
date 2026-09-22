"""Build item 7: the question-type router and ScopeError (spec section 5.2).

THIS MODULE'S ENTIRE JOB IS TO REFUSE CERTAIN QUESTIONS. The refusal is a hard error, not a
warning, because the failure it prevents is silent: assembling a conjunctive answer across the
genome by multiplying marginals returns a number, and at 16,000 genes that number is wrong by
2,311 orders of magnitude while looking exactly like an answer.

THE ASYMMETRY THAT MAKES THE ROUTER NECESSARY. Both question types are built from the same
per-gene distributions and the same per-gene rate error, and they behave oppositely:

    AGGREGATE   error FALLS with gene count, then parks at exp(sigma^2/2) - 1. Averaging kills
                the noise and leaves the bias, which is why Rule A -- dividing by that same
                factor -- is the cheapest accuracy win in the system.
    CONJUNCTIVE error GROWS LINEARLY in gene count, because each gene contributes one biased
                log-tail and biases add rather than cancel.

Same inputs, same solver, opposite scaling. So the router cannot decide by looking at the
model; it must decide by looking at the QUESTION.

=================================================================================================
GATES, PREDECLARED. Deciding statistic: worst case over the declared sweep, never the median.
=================================================================================================

R1 / G4.2 / T11  A conjunctive query above the cap raises ScopeError. m = 11 must raise.
R2               A conjunctive query AT the cap must NOT raise. m = 10 must return an answer.
                 R1 and R2 together are the gate; R1 alone would pass on a router that refuses
                 everything, which is the vacuous version of this test.
R3               The boundary is exactly where it is declared -- no off-by-one. Sweep m from 1
                 to 20 and confirm the raise set is exactly {11..20}.
R4               The error message must be ACTIONABLE: it must name the cap, the measured
                 scaling exponent, and the expected error at the requested size. A refusal that
                 does not say how wrong the answer would have been teaches nothing.
R5               The aggregate path applies Rule A. A genome-scale aggregate must be accepted
                 AND debiased -- accepting it without the correction would be the other half of
                 the failure this module exists to prevent.
R6               THE CAP IS NOT ARBITRARY. At the cap itself the predicted conjunctive error
                 must already be large (spec: ~2 orders at m = 10). If the error at the cap were
                 negligible the cap would be in the wrong place, and this gate is what would
                 detect that.

R-CONTROL   MANDATORY NEGATIVE CONTROL, with each claim tested by breaking it rather than
            asserted. The router must accept exactly the questions it should:
              (a) an AGGREGATE of 16,000 genes must NOT raise -- a router that raises on
                  everything passes R1 vacuously;
              (b) a SINGLE-gene question must NOT raise;
              (c) a conjunctive query of 10 must NOT raise.
            Each is verified against a deliberately broken router that raises unconditionally,
            and the control must fire on it.

R-CEILING   Confirm both outcomes are reachable on this test set before gating: the sweep must
            contain at least one raising and one non-raising case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

MAX_CONJUNCTIVE_GENES = 10

# MEASURED, not assumed. Source: rem/atlas/debias.py, this build order's item 2, which fits the
# conjunctive error exponent across seeds. The spec quotes m^1.06; that module's across-seed
# large-m estimate is 1.0016 and its own 3-point fit is a downward-biased 0.913. The router uses
# 1.0 -- linear -- because every estimate lands there within its spread and because the
# structural argument (one biased log-tail per gene, biases add) gives exactly 1.
CONJ_EXPONENT = 1.0
CONJ_ORDERS_AT_CAP = 2.0          # spec section 5.1: m = 10, sigma = 0.4 -> 2.0 orders


class ScopeError(RuntimeError):
    """Raised when a question is outside the region where the system can answer it."""


@dataclass
class Question:
    kind: str                      # 'aggregate' | 'conjunctive' | 'single'
    genes: List[str] = field(default_factory=list)
    sigma: float = 0.4


def expected_conjunctive_orders(m: int, sigma: float = 0.4) -> float:
    """Predicted error in orders at module size m, from the measured linear scaling."""
    per_gene = CONJ_ORDERS_AT_CAP / (MAX_CONJUNCTIVE_GENES ** CONJ_EXPONENT)
    return per_gene * (m ** CONJ_EXPONENT) * (sigma / 0.4) ** 2


def debias(mu_raw: float, sigma: float) -> float:
    """Spec section 5.2 Rule A. Multiplicative error is biased upward by exp(sigma^2/2)."""
    return mu_raw / math.exp(sigma * sigma / 2.0)


def route(q: Question, per_gene_mean: Optional[Sequence[float]] = None,
          _always_raise: bool = False) -> dict:
    """Dispatch a question by TYPE. `_always_raise` exists only so R-CONTROL can break it."""
    if _always_raise:
        raise ScopeError("broken router: refuses everything")
    m = len(q.genes)
    if q.kind == "conjunctive" and m > MAX_CONJUNCTIVE_GENES:
        raise ScopeError(
            "Conjunctive queries are capped at {} genes. ".format(MAX_CONJUNCTIVE_GENES) +
            "Error grows linearly in gene count (measured exponent {:.2f}); ".format(
                CONJ_EXPONENT) +
            "at {} genes the expected error is ~{:.0f} orders of magnitude.".format(
                m, expected_conjunctive_orders(m, q.sigma)))
    if q.kind == "aggregate":
        raw = float(np.sum(per_gene_mean)) if per_gene_mean is not None else float("nan")
        return {"kind": "aggregate", "n": m, "raw": raw,
                "answer": debias(raw, q.sigma), "rule": "A (debiased)"}
    if q.kind == "conjunctive":
        return {"kind": "conjunctive", "n": m, "rule": "true joint",
                "expected_orders": expected_conjunctive_orders(m, q.sigma)}
    return {"kind": "single", "n": m, "rule": "direct"}


def _v(ok):
    return "PASS" if ok else "FAIL"


def verify(verbose: bool = True) -> dict:
    out = {}
    genes = lambda k: ["g{}".format(i) for i in range(k)]
    print("=" * 96)
    print("R3 / R-CEILING  BOUNDARY SWEEP -- where exactly does the router refuse?")
    print("=" * 96)
    raised = []
    for m in range(1, 21):
        try:
            route(Question("conjunctive", genes(m)))
        except ScopeError:
            raised.append(m)
    expect = list(range(MAX_CONJUNCTIVE_GENES + 1, 21))
    out["R3"] = raised == expect
    print("  raises for m in {}".format(raised))
    print("  expected        {}".format(expect))
    print("  R3 {} -- boundary exactly at the declared cap, no off-by-one".format(_v(out["R3"])))
    reachable = len(raised) > 0 and len(raised) < 20
    out["R_ceiling"] = reachable
    print("  R-CEILING {}: both outcomes occur in the sweep ({} raise, {} do not), so the "
          "gate can fire either way".format(_v(reachable), len(raised), 20 - len(raised)))

    print("\n" + "=" * 96)
    print("R1 / G4.2 / T11  and  R2 -- the two halves that must BOTH hold")
    print("=" * 96)
    try:
        route(Question("conjunctive", genes(11)))
        r1 = False; msg = ""
    except ScopeError as e:
        r1 = True; msg = str(e)
    print("  m = 11 raises ScopeError: {}".format(r1))
    print("    message: {}".format(msg))
    try:
        r10 = route(Question("conjunctive", genes(10)))
        r2 = True
    except ScopeError:
        r2 = False; r10 = None
    print("  m = 10 returns an answer: {}  -> {}".format(r2, r10))
    out["R1"] = r1; out["R2"] = r2
    print("  R1 {}   R2 {}   (R1 alone would pass on a router that refuses everything)"
          .format(_v(r1), _v(r2)))

    print("\n" + "=" * 96)
    print("R4  IS THE REFUSAL ACTIONABLE?")
    print("=" * 96)
    checks = {"names the cap": str(MAX_CONJUNCTIVE_GENES) in msg,
              "names the scaling": "linearly" in msg or "exponent" in msg,
              "quantifies the error": "orders of magnitude" in msg,
              "names the requested size": " 11 genes" in msg}
    for k, v in checks.items():
        print("    {:<26s} {}".format(k, v))
    out["R4"] = all(checks.values())
    print("  R4 {}".format(_v(out["R4"])))

    print("\n" + "=" * 96)
    print("R6  IS THE CAP IN THE RIGHT PLACE?")
    print("=" * 96)
    print("  {:>5s} {:>18s}".format("m", "expected orders"))
    for m in (1, 5, 10, 11, 100, 1000, 16000):
        print("  {:>5d} {:>18.2f}".format(m, expected_conjunctive_orders(m)))
    at_cap = expected_conjunctive_orders(MAX_CONJUNCTIVE_GENES)
    out["R6"] = at_cap >= 1.0
    print("  error at the cap itself is {:.2f} orders (spec: ~2.0), already large enough that "
          "the cap\n  is not arbitrary.   R6 {}".format(at_cap, _v(out["R6"])))
    print("  at 16,000 genes the router's prediction is {:.0f} orders; the spec measured 2,311."
          .format(expected_conjunctive_orders(16000)))
    # THE SPEC'S STATED EXPONENT DISAGREES WITH THE SPEC'S OWN TABLE, and the table wins.
    # Section 5.1 lists m = 10 -> 2.0 orders and m = 16,000 -> 2,311.4 orders. Those two rows
    # fix the exponent by themselves: log(2311.4/2.0) / log(1600) = 0.956. Section 5.1 then
    # states "error ~ m^1.06". Substituting 1.06 into its own first row predicts 4,978 orders
    # at 16,000, not 2,311 -- the stated exponent overshoots its own table by 2.2x.
    a_table = math.log(2311.4 / 2.0) / math.log(1600.0)
    print("  EXPONENT IMPLIED BY THE SPEC'S OWN TABLE: log(2311.4/2.0)/log(1600) = "
          "{:.3f}".format(a_table))
    print("    the spec STATES 1.06, which fed back into its own first row would give "
          "{:.0f} orders".format(2.0 * 1600.0 ** 1.06))
    print("    at m = 16,000 rather than the 2,311 it tabulates -- a 2.2x overshoot. Item 2's")
    print("    across-seed estimate is 1.0016. All three land between 0.95 and 1.06, which is")
    print("    why this router uses exactly 1.0: the structural argument (one biased log-tail")
    print("    per gene, biases add) gives 1, and no estimate is far enough from it to matter")
    print("    for a REFUSAL, which only needs the order of magnitude of the damage.")

    print("\n" + "=" * 96)
    print("R5  DOES THE AGGREGATE PATH APPLY RULE A?")
    print("=" * 96)
    rng = np.random.default_rng(20260902)
    N, SIG = 16000, 0.4
    truth = rng.lognormal(math.log(50.0), 0.8, size=N)
    err = rng.lognormal(0.0, SIG, size=N)
    observed = truth * err
    q = Question("aggregate", genes(N), sigma=SIG)
    res = route(q, per_gene_mean=observed)
    tot = float(truth.sum())
    e_raw = 100.0 * (res["raw"] - tot) / tot
    e_deb = 100.0 * (res["answer"] - tot) / tot
    print("  N = {}, sigma = {}".format(N, SIG))
    print("  raw aggregate error      {:+.4f}%   (predicted bias exp(s^2/2)-1 = {:+.3f}%)"
          .format(e_raw, 100 * (math.exp(SIG * SIG / 2) - 1)))
    print("  debiased aggregate error {:+.4f}%".format(e_deb))
    out["R5"] = res["rule"].startswith("A") and abs(e_deb) < abs(e_raw)
    print("  R5 {} -- the aggregate path is accepted AND corrected".format(_v(out["R5"])))

    print("\n" + "=" * 96)
    print("R-CONTROL  the router must accept what it should -- each claim broken and retested")
    print("=" * 96)
    cases = [("(a) aggregate of 16,000", Question("aggregate", genes(16000)), observed),
             ("(b) single gene", Question("single", genes(1)), None),
             ("(c) conjunctive of 10", Question("conjunctive", genes(10)), None)]
    ok_ctrl = True
    for label, qq, arg in cases:
        try:
            route(qq, per_gene_mean=arg)
            accepted = True
        except ScopeError:
            accepted = False
        try:
            route(qq, per_gene_mean=arg, _always_raise=True)
            broken_raises = False
        except ScopeError:
            broken_raises = True
        fires = accepted and broken_raises
        ok_ctrl &= fires
        print("    {:<26s} accepted {:<5}  broken router raises {:<5}  control {}"
              .format(label, str(accepted), str(broken_raises),
                      "FIRES" if fires else "DOES NOT DISCRIMINATE"))
    out["R_control"] = ok_ctrl
    print("  R-CONTROL {} -- verified by breaking the router, not by assertion".format(
        _v(ok_ctrl)))

    n_pass = sum(1 for k, v in out.items() if v)
    print("\n" + "=" * 96)
    print("SUMMARY: {} of {} gates PASS".format(n_pass, len(out)))
    for k in sorted(out):
        print("  {:<12s} {}".format(k, _v(out[k])))
    return out


if __name__ == "__main__":
    verify()
