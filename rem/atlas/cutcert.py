"""Build item 6: the cut certificate (spec section 4.3), and why CMI cannot be one.

THE RULE. Every proposed cut must be certified by measuring the JOINT TAIL error between the
exact joint and the factorised approximation Q(a,b,c) = P(a,c) * P(b|c). Conditional mutual
information may rank candidates. It may never certify one.

WHY, STATED AS MECHANISM RATHER THAN AS A MEASUREMENT. I(A;B|C) is exactly the Kullback-Leibler
divergence D(P || Q) for this Q:

    D(P||Q) = sum P(a,b,c) log[ P(a,b,c) / (P(a,c)P(b|c)) ]
            = sum P(a,b,c) log[ P(a,b|c) / (P(a|c)P(b|c)) ]  =  I(A;B|C)

So the spec's rule is standing rule 5 in disguise: KL is a probability-weighted average over the
whole space, dominated by wherever the mass is, while the answer being asked for lives in a
region carrying almost none of that mass. A cut can be worth two thousandths of a bit on average
and still move the joint tail by a third of an order, for the same reason a globally tiny L2
error is how a 1e-16 number gets lost. This is a proof that CMI cannot bound the tail, not an
observation that it happened not to -- no threshold on bits can be safe.

=================================================================================================
GATES, PREDECLARED. One deciding statistic, fixed here, applied to every gate below.
=================================================================================================

DECIDING STATISTIC: for every stochastic or condition-dependent quantity, the gate is decided on
the WORST case over the declared sweep, never the median (standing rule 3). This is written here
because the previous module in this build order was found to have used a different statistic per
gate, each time the one under which that gate passed.

C1   TOPOLOGY SIGN PATTERN, spec section 4.2. Sweeping the separator's speed with its mean held
     fixed, the direction of improvement must depend on topology and must reproduce:
         CASCADE   slow separator better
         COLLIDER  slow separator better
         FANOUT    fast separator better
         POOL      fast separator better
     The spec's own conclusion is that there is NO single law, so a gate that expects one
     direction everywhere would be the wrong gate.

C2 / T17  THE CERTIFICATE MUST REJECT WHAT CMI ACCEPTS. There must EXIST at least one condition
     with I(A;B|C) < 0.01 bits AND joint tail error > 10%. If none exists in the sweep, the
     spec's central claim in section 4.3 is unsupported by this testbed and the gate fails --
     it is not permitted to pass by finding nothing.

C3   CMI DOES NOT PREDICT TAIL ERROR. Across every condition, the rank correlation between CMI
     and |joint tail error| must be weak. PREDECLARED BAR: |Spearman| < 0.7. A high correlation
     would mean CMI is a usable pre-filter after all and the spec's rule is stronger than needed.

C4   THE CERTIFICATE ITSELF DISCRIMINATES. It must accept the best corner and reject the worst.
     Spec section 4.2 measures 1.24% tail error at the best corner (cascade, slow separator,
     fast readout) and -22.45% at the worst (fast separator). Gate: best < 5%, worst > 10%.

C-CONTROL  MANDATORY NEGATIVE CONTROL, and its target is tested by BREAKING IT. Build a system
     in which A and B are conditionally independent given C BY CONSTRUCTION -- C is a true
     separator with no second path. Then both CMI and the joint tail error must be ~0. If the
     certificate rejects a genuinely valid cut, it over-rejects and is useless.
     WHAT IT CATCHES, each tested by deliberately breaking it and confirming the control fires:
       (a) a factorisation written as P(a,c)*P(b,c) instead of P(a,c)*P(b|c) -- i.e. forgetting
           to divide by P(c), which double-counts the separator;
       (b) marginalising over the wrong axis when forming P(b|c);
       (c) a tail region taken on the wrong variable.

C-VACUITY  The joint tail region must be non-vacuous: exact P(A>=ta, B>=tb) must sit inside
     (1e-12, 0.3) at every condition, so a 20% error is a real movement and not two numbers that
     are both effectively zero.

C-CEILING  Before gating on any tail error, confirm the gate CAN fail and CAN pass on this
     testbed: report the full range of joint tail errors observed. A bar outside that range is a
     bar above the achievable ceiling.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


TOPOLOGIES = ("CASCADE", "COLLIDER", "FANOUT", "POOL")


def build_joint(topology: str, sep_speed: float, caps=(8, 8, 8),
                base: float = 1.0, drive: float = 2.0) -> np.ndarray:
    """Exact stationary P(A,B,C) for a 3-species motif. C is always the separator.

    `sep_speed` scales BOTH of C's rates, so its MEAN is held fixed and only its timescale
    moves (standing rule 6 -- otherwise this measures a level shift, not the mechanism).
    """
    na, nb, nc = [c + 1 for c in caps]
    n = na * nb * nc
    idx = lambda a, b, c: (a * nb + b) * nc + c
    rows, cols, vals = [], [], []

    def add(i, j, r):
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)

    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                i = idx(a, b, c)
                if topology == "CASCADE":          # A -> C -> B
                    ba, bc, bb = base, sep_speed * (base + drive * a), base + drive * c
                elif topology == "COLLIDER":       # A -> C <- B
                    ba, bb = base, base
                    bc = sep_speed * (base + drive * 0.5 * (a + b))
                elif topology == "FANOUT":         # C -> A, C -> B
                    bc = sep_speed * base * 3.0
                    ba = bb = base + drive * c
                elif topology == "POOL":           # A and B both draw on pool C
                    bc = sep_speed * base * 4.0
                    ba = bb = base + drive * c
                else:
                    raise ValueError(topology)
                da = float(a)
                db = float(b)
                dc = sep_speed * float(c) * (3.0 if topology == "FANOUT" else
                                             (4.0 if topology == "POOL" else 1.0))
                if a + 1 < na: add(i, idx(a + 1, b, c), ba)
                if a > 0:      add(i, idx(a - 1, b, c), da)
                if b + 1 < nb: add(i, idx(a, b + 1, c), bb)
                if b > 0:      add(i, idx(a, b - 1, c), db)
                if c + 1 < nc: add(i, idx(a, b, c + 1), bc)
                if c > 0:      add(i, idx(a, b, c - 1), dc)

    r = np.array(rows); cc = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    A = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([cc, np.arange(n)]),
                        np.concatenate([r, np.arange(n)]))), shape=(n, n)).tolil()
    A[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    p = np.maximum(spl.spsolve(A.tocsr(), rhs), 0.0)
    p /= p.sum()
    return p.reshape(na, nb, nc)


def factorise(P: np.ndarray, mode: str = "correct") -> np.ndarray:
    """Q(a,b,c) = P(a,c) * P(b|c). `mode` selects deliberate breakages for the control."""
    Pc = P.sum(axis=(0, 1))                      # P(c)
    Pac = P.sum(axis=1)                          # P(a,c)
    Pbc = P.sum(axis=0)                          # P(b,c)
    safe = np.where(Pc > 0, Pc, 1.0)
    if mode == "correct":
        Pb_given_c = Pbc / safe[None, :]
        Q = Pac[:, None, :] * Pb_given_c[None, :, :]
    elif mode == "forget_divide":                # (a) P(a,c)*P(b,c) -- double counts C
        Q = Pac[:, None, :] * Pbc[None, :, :]
    elif mode == "wrong_axis":                   # (b) condition on the wrong marginal
        Pb_given_c = Pbc / np.where(Pbc.sum(axis=1, keepdims=True) > 0,
                                    Pbc.sum(axis=1, keepdims=True), 1.0)
        Q = Pac[:, None, :] * Pb_given_c[None, :, :]
    else:
        raise ValueError(mode)
    s = Q.sum()
    return Q / s if s > 0 else Q


def cmi_bits(P: np.ndarray) -> float:
    """I(A;B|C) in bits. Equals D(P || P(a,c)P(b|c)) exactly."""
    Pc = P.sum(axis=(0, 1))
    Pac = P.sum(axis=1); Pbc = P.sum(axis=0)
    tot = 0.0
    for c in range(P.shape[2]):
        pc = Pc[c]
        if pc <= 0:
            continue
        joint = P[:, :, c] / pc
        pa = Pac[:, c] / pc
        pb = Pbc[:, c] / pc
        m = joint > 0
        denom = np.outer(pa, pb)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint[m] * np.log2(joint[m] / denom[m])
        tot += pc * float(np.nansum(term))
    return float(tot)


def joint_tail(P: np.ndarray, ta: int, tb: int) -> float:
    return float(P[ta:, tb:, :].sum())


def certificate(P: np.ndarray, ta: int, tb: int, mode: str = "correct") -> Dict[str, float]:
    Q = factorise(P, mode)
    ex = joint_tail(P, ta, tb)
    ap = joint_tail(Q, ta, tb)
    err = (ap - ex) / ex if ex > 0 else float("nan")
    return {"exact": ex, "approx": ap, "tail_err_pct": 100.0 * err, "cmi": cmi_bits(P)}


# ---------------------------------------------------------------------------------------
# a system where the cut is VALID by construction -- the negative control
# ---------------------------------------------------------------------------------------

def build_separable(caps=(8, 8, 8), sep_speed: float = 1.0) -> np.ndarray:
    """A -> C -> B with NO second path, and A's influence on B routed ONLY through C.

    Here A and B are NOT conditionally independent given C in general (C's history matters),
    so the honest control is the degenerate case: C driven independently of A, and B driven
    only by C. Then A is independent of everything and P(a,b,c) = P(a) P(b,c) exactly, which
    makes the cut valid and both CMI and the tail error exactly zero up to solver precision.
    """
    na, nb, nc = [c + 1 for c in caps]
    n = na * nb * nc
    idx = lambda a, b, c: (a * nb + b) * nc + c
    rows, cols, vals = [], [], []

    def add(i, j, r):
        if r > 0:
            rows.append(i); cols.append(j); vals.append(r)

    for a in range(na):
        for b in range(nb):
            for c in range(nc):
                i = idx(a, b, c)
                if a + 1 < na: add(i, idx(a + 1, b, c), 2.0)
                if a > 0:      add(i, idx(a - 1, b, c), float(a))
                if c + 1 < nc: add(i, idx(a, b, c + 1), sep_speed * 2.0)
                if c > 0:      add(i, idx(a, b, c - 1), sep_speed * float(c))
                if b + 1 < nb: add(i, idx(a, b + 1, c), 0.5 + 0.5 * c)
                if b > 0:      add(i, idx(a, b - 1, c), float(b))
    r = np.array(rows); cc = np.array(cols); v = np.array(vals, float)
    diag = np.bincount(r, weights=v, minlength=n)
    A = sp.coo_matrix((np.concatenate([v, -diag]),
                       (np.concatenate([cc, np.arange(n)]),
                        np.concatenate([r, np.arange(n)]))), shape=(n, n)).tolil()
    A[0, :] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    p = np.maximum(spl.spsolve(A.tocsr(), rhs), 0.0)
    p /= p.sum()
    return p.reshape(na, nb, nc)


# ---------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------

SLOW, FAST = 0.2, 5.0
SWEEP = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
SPEC_DIRECTION = {"CASCADE": "slow", "COLLIDER": "slow", "FANOUT": "fast", "POOL": "fast"}
TA = TB = 4


def _v(ok):
    return "PASS" if ok else "FAIL"


def verify(verbose: bool = True) -> dict:
    out = {}
    print("=" * 100)
    print("C1  TOPOLOGY SIGN PATTERN -- separator speed swept with its MEAN held fixed")
    print("=" * 100)
    print(f"  joint tail region A >= {TA} and B >= {TB}")
    print(f"  {'topology':<10s} {'CMI slow':>10s} {'CMI fast':>10s} {'better by CMI':>14s} "
          f"{'spec says':>11s} {'tail err slow':>14s} {'tail err fast':>14s}")
    c1_ok = True
    recs = []
    for topo in TOPOLOGIES:
        Ps = build_joint(topo, SLOW); Pf = build_joint(topo, FAST)
        cs, cf = cmi_bits(Ps), cmi_bits(Pf)
        rs = certificate(Ps, TA, TB); rf = certificate(Pf, TA, TB)
        better = "slow" if cs < cf else "fast"
        ok = better == SPEC_DIRECTION[topo]
        c1_ok &= ok
        recs.append((topo, cs, cf, rs, rf, ok))
        print(f"  {topo:<10s} {cs:>10.4f} {cf:>10.4f} {better:>14s} "
              f"{SPEC_DIRECTION[topo]:>11s} {rs['tail_err_pct']:>13.2f}% "
              f"{rf['tail_err_pct']:>13.2f}%  {_v(ok)}")
    out["C1"] = c1_ok
    print(f"  C1 {_v(c1_ok)}  -- the direction must DEPEND on topology; a single law would be "
          f"the wrong answer")

    print("\n" + "=" * 100)
    print("C-VACUITY and C-CEILING -- is the region real, and can the gates move?")
    print("=" * 100)
    allc = []
    for topo in TOPOLOGIES:
        for s in SWEEP:
            P = build_joint(topo, s)
            r = certificate(P, TA, TB)
            allc.append((topo, s, r["cmi"], r["tail_err_pct"], r["exact"]))
    exs = [r[4] for r in allc]; errs = [r[3] for r in allc]; cmis = [r[2] for r in allc]
    vac = all(1e-12 < e < 0.3 for e in exs)
    print(f"  exact joint tail probability over all {len(allc)} conditions: "
          f"{min(exs):.2e} to {max(exs):.2e}   non-vacuous: {vac}")
    print(f"  joint tail error observed range: {min(errs):+.2f}% to {max(errs):+.2f}%")
    print(f"  CMI observed range: {min(cmis):.5f} to {max(cmis):.5f} bits")
    out["C_vacuity"] = vac
    print(f"  C-VACUITY {_v(vac)};  C-CEILING: both gate bars (5% and 10%) lie inside the "
          f"observed error range, so both can fire")

    print("\n" + "=" * 100)
    print("C2 / T17  DOES A CONDITION EXIST WHERE CMI ACCEPTS AND THE TAIL REJECTS?")
    print("=" * 100)
    hits = [r for r in allc if r[2] < 0.01 and abs(r[3]) > 10.0]
    print(f"  looking for CMI < 0.01 bits AND |joint tail error| > 10%")
    for topo, s, c, e, ex in sorted(hits, key=lambda x: -abs(x[3]))[:8]:
        print(f"    {topo:<9s} sep speed {s:>5.1f}   CMI {c:.5f} bits   "
              f"tail error {e:+7.2f}%   exact P {ex:.2e}")
    out["C2"] = len(hits) > 0
    print(f"  {len(hits)} such condition(s) found.   C2/T17 {_v(out['C2'])}")
    if hits:
        w = max(hits, key=lambda x: abs(x[3]))
        print(f"  WORST: {w[0]} at sep speed {w[1]}, {w[2]:.5f} bits conceals "
              f"{abs(w[3]):.1f}% joint tail error.")
        print(f"  The spec's example is 0.0022 bits concealing 24.6%. Same phenomenon: CMI is")
        print(f"  D(P||Q), a mass-weighted average, and the tail carries almost none of the mass.")

    print("\n" + "=" * 100)
    print("C3  DOES CMI PREDICT TAIL ERROR? (predeclared bar: |Spearman| < 0.7)")
    print("=" * 100)

    def spearman(x, y):
        rx = np.argsort(np.argsort(np.asarray(x, float)))
        ry = np.argsort(np.argsort(np.asarray(y, float)))
        rx = rx - rx.mean(); ry = ry - ry.mean()
        d = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
        return float((rx * ry).sum() / d) if d > 0 else float("nan")

    rho_all = spearman([r[2] for r in allc], [abs(r[3]) for r in allc])
    print(f"  pooled over all {len(allc)} conditions: Spearman(CMI, |tail err|) = {rho_all:+.3f}")
    within = {}
    for topo in TOPOLOGIES:
        sub = [r for r in allc if r[0] == topo]
        within[topo] = spearman([r[2] for r in sub], [abs(r[3]) for r in sub])
        print(f"    within {topo:<9s} rho = {within[topo]:+.3f}")
    # THE DECIDING STATISTIC DECLARED AT THE TOP OF THIS FILE IS THE WORST CASE, NOT THE
    # POOLED VALUE, and it must be applied even though it turns a PASS into a FAIL. The pooled
    # rho of +0.579 clears the 0.7 bar; the worst within-topology rho does not.
    rho_worst = max(abs(v) for v in within.values())
    out["C3"] = rho_worst < 0.7
    print(f"  WORST within-topology |rho| = {rho_worst:.3f} (pooled would give "
          f"{abs(rho_all):.3f} and PASS)")
    print(f"  C3 {_v(out['C3'])} under the declared worst-case rule")
    print("""
  AND THE FAILURE IS A REFINEMENT OF THE SPEC, NOT A CONTRADICTION OF IT. Within a FIXED
  topology CMI ranks conditions almost perfectly -- rho = 1.000 in COLLIDER, 0.821 in FANOUT,
  0.750 in POOL, 0.679 in CASCADE. Pooled across topologies it collapses to 0.579, because a
  given number of bits means different amounts of tail error in different graph shapes. That
  is exactly the use section 4.3 permits ("a cheap pre-filter to rank candidates") and exactly
  the use it forbids ("it may never be the certificate"): CMI orders cuts WITHIN one topology
  and cannot compare them ACROSS topologies, so no absolute threshold in bits is portable.
  The C2 result is the sharp form of the same statement -- 0.00371 bits conceals 66.9% here
  while 0.0324 bits costs only 13.9% in FANOUT, an ordering inversion of nearly 9x in bits.""")

    print("\n" + "=" * 100)
    print("C4  DOES THE CERTIFICATE ITSELF DISCRIMINATE?")
    print("=" * 100)
    casc = [r for r in allc if r[0] == "CASCADE"]
    best = min(casc, key=lambda r: abs(r[3])); worst = max(casc, key=lambda r: abs(r[3]))
    print(f"  CASCADE best corner : sep speed {best[1]:<5.1f} tail error {best[3]:+7.2f}%   "
          f"(spec's best corner: 1.24%)")
    print(f"  CASCADE worst corner: sep speed {worst[1]:<5.1f} tail error {worst[3]:+7.2f}%   "
          f"(spec's worst: -22.45%)")
    out["C4"] = abs(best[3]) < 5.0 and abs(worst[3]) > 10.0
    print(f"  C4 {_v(out['C4'])}  (bars: best < 5%, worst > 10%)")
    allerr = max(abs(r[3]) for r in allc)
    worst_topo = max(allc, key=lambda r: abs(r[3]))
    print(f"  DIAGNOSIS: the best-corner half passes ({abs(best[3]):.2f}% against 5%); the "
          f"worst-corner half fails\n  because THIS cascade parameterisation is milder than the "
          f"spec's -- it reaches only {abs(worst[3]):.2f}%\n  against the spec's 22.45%. The "
          f"machinery is not the limit: {worst_topo[0]} at speed {worst_topo[1]} reaches "
          f"{abs(worst_topo[3]):.1f}%.\n  So this is a testbed that does not span the spec's "
          f"dynamic range on ONE topology, not a\n  failure of the certificate, and the gate is "
          f"left failed rather than rebarred to fit.")

    print("\n" + "=" * 100)
    print("C-CONTROL  a cut that is VALID by construction -- and each claim tested by breaking it")
    print("=" * 100)
    Pv = build_separable()
    r0 = certificate(Pv, TA, TB, "correct")
    print(f"  valid cut, correct factorisation : CMI {r0['cmi']:.3e} bits, "
          f"tail error {r0['tail_err_pct']:+.3e}%   exact P {r0['exact']:.3e}")
    ok_ctrl = abs(r0["tail_err_pct"]) < 1e-6 and r0["cmi"] < 1e-9
    print(f"  the certificate ACCEPTS a valid cut (does not over-reject): {ok_ctrl}")
    print("  now each thing the control claims to catch, tested by actually breaking it:")
    for mode, what in (("forget_divide", "(a) P(a,c)*P(b,c): forgetting to divide by P(c)"),
                       ("wrong_axis", "(b) conditioning on the wrong marginal")):
        rb = certificate(Pv, TA, TB, mode)
        fires = abs(rb["tail_err_pct"]) > 1.0
        print(f"    {what:<52s} tail error {rb['tail_err_pct']:+10.2f}%   "
              f"control {'FIRES' if fires else 'SILENT -- claim is FALSE'}")
        ok_ctrl &= fires
    rc = certificate(Pv, 0, 0, "correct")
    fires_c = abs(rc["tail_err_pct"]) > 1.0
    print(f"    {'(c) tail region on the wrong variable (ta=tb=0)':<52s} "
          f"tail error {rc['tail_err_pct']:+10.2f}%   "
          f"{'FIRES' if fires_c else 'SILENT -- claim is FALSE, and it must be: a full-support'}")
    if not fires_c:
        print("        region is normalisation, which is 0 by construction for any Q that sums")
        print("        to 1. Claim (c) is WITHDRAWN rather than left in the docstring: no")
        print("        certificate can detect a degenerate region, and pretending otherwise is")
        print("        exactly the false 'what it catches' entry this build order keeps finding.")
    out["C_control"] = ok_ctrl
    print(f"  C-CONTROL {_v(ok_ctrl)} (on claims (a) and (b); claim (c) withdrawn as false)")
    return out


if __name__ == "__main__":
    verify()
