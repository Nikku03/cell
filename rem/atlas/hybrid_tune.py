"""Choosing the base point for the hybrid test, BEFORE its gates are written.

This script produces no headline claim. It exists because hybrid.py hard-codes a circuit, a
state-space cap, an initial population and a base rate vector, and those are modelling choices
that must be visible rather than asserted. Everything hybrid.py imports from here was chosen by
running this and reading the output, which is saved beside it as RESULTS_hybrid_tune.txt.

WHAT IS BEING CHOSEN, AND AGAINST WHAT CRITERION.

1. THE CIRCUIT. The hide-the-rate test used a four-rate persister circuit. Four rates cannot
   answer "how many rates do we need to measure" -- the answer would be 0,1,2,3 or 4 and the
   curve has no shape. This is the same persister biology resolved into EIGHT distinct physical
   processes, each of which is a separate barrier that a chemistry calculation would have to
   supply separately:
       mu       G divides                     (drug absent)
       k_kill   G killed                      (drug present)
       a_off    G -> D, spontaneous switching (drug absent)
       a_on     G -> D, stress-induced        (drug present)
       b_off    D -> G, waking                (drug absent)
       b_on     D -> G, waking under drug     (drug present)
       d_death  D dies spontaneously          (drug absent)
       kd_kill  D killed -- tolerant, not immune  (drug present)

2. THE CAP. The chain is truncated at cap_g x cap_d. Chemistry draws move rates by up to an
   order, so a cap that binds under perturbation would let truncation, not biology, set the
   answer. The criterion is that Y agree with a much larger cap to better than 1e-3 relative,
   BOTH at the base point and under 10x single-rate perturbations.

3. g0 AND cycles. The criterion is that Y sit well inside (1e-9, 0.99) so the observable is not
   saturated -- the defect that has now cost four separate reruns in this build order.

4. N_TRIALS AND THE COST. Reported so the run length of hybrid.py is a prediction, not a hope.
   The dominant cost is 2^8 = 256 subsets x N_TRIALS evaluations.

5. THE SENSITIVITIES. Reported here with SIGNS, because the hide-the-rate correction showed the
   signed sum is what governs correlated error, and because the greedy ordering hybrid.py tests
   is defined by their magnitudes.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy.linalg import expm

RULE = "=" * 97
NAMES = ("mu", "k_kill", "a_off", "a_on", "b_off", "b_on", "d_death", "kd_kill")


def generator(cap_g, cap_d, r, drug):
    n = (cap_g + 1) * (cap_d + 1)
    idx = lambda g, d: g * (cap_d + 1) + d
    L = np.zeros((n, n))

    def add(i, j, rate):
        if rate > 0:
            L[j, i] += rate
            L[i, i] -= rate

    for g in range(cap_g + 1):
        for d in range(cap_d + 1):
            i = idx(g, d)
            if not drug:
                if g + 1 <= cap_g:
                    add(i, idx(g + 1, d), r["mu"] * g)
                if g > 0 and d + 1 <= cap_d:
                    add(i, idx(g, d + 1), r["a_off"] * g)
                if d > 0 and g + 1 <= cap_g:
                    add(i, idx(g + 1, d - 1), r["b_off"] * d)
                if d > 0:
                    add(i, idx(g, d - 1), r["d_death"] * d)
            else:
                if g > 0:
                    add(i, idx(g - 1, d), r["k_kill"] * g)
                if g > 0 and d + 1 <= cap_d:
                    add(i, idx(g, d + 1), r["a_on"] * g)
                if d > 0 and g + 1 <= cap_g:
                    add(i, idx(g + 1, d - 1), r["b_on"] * d)
                if d > 0:
                    add(i, idx(g, d - 1), r["kd_kill"] * d)
    return L


def eradication(r, cap_g=10, cap_d=10, g0=4, t_on=6.0, t_off=3.0, cycles=3):
    """P(zero cells of either type) after `cycles` on/off courses."""
    step = expm(generator(cap_g, cap_d, r, False) * t_off) @ expm(generator(cap_g, cap_d, r, True) * t_on)
    n = (cap_g + 1) * (cap_d + 1)
    p = np.zeros(n)
    p[min(g0, cap_g) * (cap_d + 1)] = 1.0
    for _ in range(cycles):
        p = np.maximum(step @ p, 0.0)
        s = p.sum()
        if s > 0:
            p /= s
    return float(p[0])


def sensitivity(base, name, h=0.02, **kw):
    up = dict(base); up[name] = base[name] * 10.0 ** h
    dn = dict(base); dn[name] = base[name] * 10.0 ** -h
    return (np.log10(eradication(up, **kw)) - np.log10(eradication(dn, **kw))) / (2 * h)


CANDIDATE = dict(mu=1.0, k_kill=0.25, a_off=0.40, a_on=0.60,
                 b_off=0.03, b_on=0.008, d_death=0.02, kd_kill=0.010)


def main():
    out = []
    P = lambda s="": (print(s), out.append(s))

    P(RULE); P("CHOOSING THE BASE POINT FOR THE HYBRID TEST"); P(RULE)

    P("\n1  CAP CONVERGENCE AT THE BASE POINT")
    P(f"  {'cap':>6}{'states':>9}{'Y':>16}{'ms/eval':>10}")
    ys = {}
    for c in (8, 10, 12, 14):
        t = time.time()
        for _ in range(20):
            y = eradication(CANDIDATE, c, c, g0=4)
        ms = (time.time() - t) / 20 * 1000
        ys[c] = y
        P(f"  {c:>6}{(c+1)**2:>9}{y:>16.8e}{ms:>10.2f}")
    ref = ys[14]
    P(f"  cap 10 vs cap 14: relative {abs(ys[10]-ref)/ref:.2e}")

    P("\n2  CAP CONVERGENCE UNDER 10x SINGLE-RATE PERTURBATION  (the case that matters)")
    P(f"  {'perturbation':>18}{'cap 10':>16}{'cap 14':>16}{'relative':>12}")
    worst = 0.0
    for nm in NAMES:
        for mult in (10.0, 0.1):
            r = dict(CANDIDATE); r[nm] = CANDIDATE[nm] * mult
            y10 = eradication(r, 10, 10, g0=4)
            y14 = eradication(r, 14, 14, g0=4)
            rel = abs(y10 - y14) / max(y14, 1e-300)
            worst = max(worst, rel)
            if rel > 1e-5:
                P(f"  {nm+' x'+str(mult):>18}{y10:>16.6e}{y14:>16.6e}{rel:>12.2e}")
    P(f"  worst relative change over all 16 perturbations: {worst:.2e}")
    P(f"  CHOICE: cap_g = cap_d = 10 {'(criterion 1e-3 met)' if worst < 1e-3 else '(CRITERION FAILED)'}")

    P("\n3  g0 AND cycles -- Y must not be saturated")
    P(f"  {'g0':>4}{'cycles':>8}{'Y':>16}{'headroom up (orders)':>24}")
    for g0 in (3, 4, 5, 6):
        for cy in (3, 4):
            y = eradication(CANDIDATE, 10, 10, g0=g0, cycles=cy)
            P(f"  {g0:>4}{cy:>8}{y:>16.6e}{np.log10(1.0/y):>24.3f}")
    P("  CHOICE: g0 = 4, cycles = 3 for the main run; g0 = 6 for the depth check (gate Y10).")

    P("\n4  SENSITIVITIES AT THE CHOSEN BASE POINT, WITH SIGNS")
    S = {nm: sensitivity(CANDIDATE, nm, 0.02, cap_g=10, cap_d=10, g0=4) for nm in NAMES}
    S2 = {nm: sensitivity(CANDIDATE, nm, 0.01, cap_g=10, cap_d=10, g0=4) for nm in NAMES}
    P(f"  {'rate':>9}{'S (h=0.02)':>14}{'S (h=0.01)':>14}{'rel change':>13}")
    wr = 0.0
    for nm in NAMES:
        rel = abs(S[nm] - S2[nm]) / max(abs(S2[nm]), 1e-12)
        wr = max(wr, rel)
        P(f"  {nm:>9}{S[nm]:>+14.4f}{S2[nm]:>+14.4f}{rel:>13.2e}")
    P(f"  worst step-halving change {wr:.2e}  (a derivative, not a difference, if < 1%)")
    v = np.array([S[nm] for nm in NAMES])
    P(f"  signed sum       {v.sum():+.4f}")
    P(f"  quadrature norm  {np.sqrt((v**2).sum()):.4f}")
    P(f"  correlated/independent ratio for this circuit: {abs(v.sum())/np.sqrt((v**2).sum()):.4f}")
    P("  greedy order by |S|: " + str([nm for nm in sorted(NAMES, key=lambda n: -abs(S[n]))]))

    P("\n5  RESIDUAL SPREAD AFTER MEASURING THE TOP m, FROM THE LINEAR FORMULA")
    P("  sd_residual = sigma * sqrt(sum of S^2 over the UNMEASURED rates), sigma = eps * 0.7328")
    order = sorted(NAMES, key=lambda n: -abs(S[n]))
    P(f"  {'m':>3}{'measured last':>15}{'resid |S|':>12}{'sd @1kcal':>12}")
    for m in range(len(NAMES) + 1):
        rest = [S[n] for n in order[m:]]
        q = float(np.sqrt(sum(x * x for x in rest)))
        P(f"  {m:>3}{(order[m-1] if m else '--'):>15}{q:>12.4f}{q*0.73284:>12.4f}")

    P("\n6  COST FORECAST FOR hybrid.py")
    t = time.time()
    for _ in range(50):
        eradication(CANDIDATE, 10, 10, g0=4)
    per = (time.time() - t) / 50
    n_tr = 500
    evals = 2 ** 8 * n_tr + 2 * 9 * n_tr + 9 * n_tr
    P(f"  measured {per*1000:.2f} ms per evaluation at cap 10, one BLAS thread")
    P(f"  planned {evals} evaluations at N_TRIALS = {n_tr}  ->  {evals*per/60:.1f} minutes")
    P("  CHOICE: N_TRIALS = 500, common random numbers shared across all subsets.")

    open(os.path.join(os.path.dirname(__file__), "RESULTS_hybrid_tune.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
