"""Choosing the base point for the hybrid test, BEFORE its gates are written.

This script produces no headline claim. It exists because hybrid.py hard-codes a circuit, a
state space, an initial population and a base rate vector, and those are modelling choices that
must be visible rather than asserted. Everything hybrid.py imports from here was chosen by running
this and reading the output, saved beside it as RESULTS_hybrid_tune.txt.

=================================================================================================
CORRECTION. THE FIRST VERSION OF THIS SCRIPT FAILED ITS OWN CRITERION, AND THE MODEL WAS WRONG.
=================================================================================================
The first version (commit before this one) used an UNBOUNDED birth process, G -> G+1 at mu*g with
no carrying capacity, truncated at a state cap, with the distribution renormalised after each
cycle to hide the leakage. Its criterion 2 -- that the cap must not change Y by more than 1e-3
relative under perturbation -- FAILED at 2.36e-02, and the failure got worse the harder it was
pushed. Measured against a much larger cap, on the ACTUAL chemistry draws rather than the tidy 10x
probes:

    cap 10 vs cap 14   worst |d log10 Y| = 0.776 orders (eps=1), 3.97 orders (eps=2)
    cap 14 vs cap 20   worst |d log10 Y| = 1.267 orders (eps=1), 2.93 orders (eps=2)
    and 2 to 3 percent of factor-of-2 verdicts FLIPPED between caps.

The tempting move was to relax the criterion, on the argument that a 2% error in Y is small next
to a spread measured in orders. That argument is wrong and I nearly made it: the deliverable is a
CLASSIFICATION (is this trial within a factor of 2 of the truth?), and the flip test above shows
the classification itself was moving. Raising the cap does not fix it either -- an unbounded birth
process has no converged truncation, so every cap is doing some of the work.

The defect was in the MODEL, not the bar. Real populations do not grow without limit. Adding a
carrying capacity K, so the birth rate is mu*g*(1 - (g+d)/K), closes the state space EXACTLY: the
only transition that raises g+d is birth, birth is zero at g+d = K, so no probability can leave
{g+d <= K}. There is then no truncation error to bound and no renormalisation to hide it, and the
generator's column sums and the propagated mass are checkable to machine precision (section 1).

K is now a MODELLING parameter -- a carrying capacity, which the model previously lacked and
should always have had -- not a numerical cap. The state space is the triangle {g+d <= K}, which
is also half the size of the square grid it replaces.

=================================================================================================
WHAT IS BEING CHOSEN, AND AGAINST WHAT CRITERION
=================================================================================================

1. THE CIRCUIT. The hide-the-rate test used a four-rate persister circuit. Four rates cannot
   answer "how many rates do we need to measure" -- the answer would be 0 to 4 and the curve has
   no shape. This is the same persister biology resolved into EIGHT distinct physical processes,
   each a separate barrier a chemistry calculation would have to supply on its own:

       drug ABSENT                            drug PRESENT
       mu       G divides (logistic)          k_kill   G killed
       a_off    G -> D, spontaneous           a_on     G -> D, stress-induced
       b_off    D -> G, waking                b_on     D -> G, waking under drug
       d_death  D dies spontaneously          kd_kill  D killed -- tolerant, not immune

   Note the split is 4/4 by phase, and no rate appears in both phases. That is not cosmetic: it
   means the drug-on and drug-off propagators depend on disjoint halves of the rate vector, so the
   256 subsets of an 8-rate lattice need only 16+16 = 32 matrix exponentials per trial instead of
   512. hybrid.py exploits this, which is what makes an exhaustive subset enumeration affordable.

2. EXACTNESS OF THE STATE SPACE. Criterion: generator column sums and propagated mass at machine
   precision, with no renormalisation anywhere. This replaces the cap-convergence criterion that
   the unbounded model failed.

3. K, g0 AND cycles. Criterion: Y well inside (1e-9, 0.99), with enough upward headroom that the
   factor-of-10 band is not pressed against the probability ceiling. Saturation has now cost this
   build order four separate reruns, so the headroom is reported explicitly.

4. THE SENSITIVITIES, with SIGNS -- the hide-the-rate correction showed the signed sum governs
   correlated error, and the greedy ordering hybrid.py tests is defined by their magnitudes.

5. THE LINEAR PREDICTION OF THE ANSWER, written down here so hybrid.py's Monte-Carlo can refute
   it rather than be tuned to it.

6. N_TRIALS AND THE COST, so the run length of hybrid.py is a prediction and not a hope.
"""

from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import numpy as np
from scipy.linalg import expm

RULE = "=" * 97

# Rate names, grouped by the phase whose generator they enter. The grouping is load-bearing.
OFF_RATES = ("mu", "a_off", "b_off", "d_death")
ON_RATES = ("k_kill", "a_on", "b_on", "kd_kill")
NAMES = OFF_RATES + ON_RATES

CANDIDATE = dict(mu=1.0, k_kill=0.25, a_off=0.40, a_on=0.60,
                 b_off=0.03, b_on=0.008, d_death=0.02, kd_kill=0.010)

RT = 0.5925                                      # kcal/mol at 298 K
ORDERS_PER_KCAL = 1.0 / (RT * np.log(10.0))      # 0.7328 orders per kcal/mol


def state_index(K):
    """The triangle {g + d <= K}. Birth is the only transition that raises g+d, and it is zero
    at g+d = K, so this set is closed -- there is nothing to truncate."""
    S = [(g, d) for n in range(K + 1) for g in range(n + 1) for d in (n - g,)]
    return S, {s: i for i, s in enumerate(S)}


def generator(K, IX, r, drug):
    n = len(IX)
    L = np.zeros((n, n))

    def add(i, j, rate):
        if rate > 0:
            L[j, i] += rate
            L[i, i] -= rate

    for (g, d), i in IX.items():
        room = max(0.0, 1.0 - (g + d) / K)
        if not drug:
            if g > 0 and g + d < K:
                add(i, IX[(g + 1, d)], r["mu"] * g * room)
            if g > 0:
                add(i, IX[(g - 1, d + 1)], r["a_off"] * g)
            if d > 0:
                add(i, IX[(g + 1, d - 1)], r["b_off"] * d)
            if d > 0:
                add(i, IX[(g, d - 1)], r["d_death"] * d)
        else:
            if g > 0:
                add(i, IX[(g - 1, d)], r["k_kill"] * g)
            if g > 0:
                add(i, IX[(g - 1, d + 1)], r["a_on"] * g)
            if d > 0:
                add(i, IX[(g + 1, d - 1)], r["b_on"] * d)
            if d > 0:
                add(i, IX[(g, d - 1)], r["kd_kill"] * d)
    return L


def eradication(r, K=20, g0=6, t_on=6.0, t_off=3.0, cycles=3):
    """P(zero cells of either type) after `cycles` on/off courses. No renormalisation: the state
    space is closed, so any mass loss would be a bug and is gated as such."""
    S, IX = state_index(K)
    step = expm(generator(K, IX, r, False) * t_off) @ expm(generator(K, IX, r, True) * t_on)
    p = np.zeros(len(S))
    p[IX[(min(g0, K), 0)]] = 1.0
    for _ in range(cycles):
        p = np.maximum(step @ p, 0.0)
    return float(p[IX[(0, 0)]])


def sensitivity(base, name, h=0.02, **kw):
    up = dict(base); up[name] = base[name] * 10.0 ** h
    dn = dict(base); dn[name] = base[name] * 10.0 ** -h
    return (np.log10(eradication(up, **kw)) - np.log10(eradication(dn, **kw))) / (2 * h)


def main():
    out = []
    P = lambda s="": (print(s), out.append(s))

    P(RULE); P("CHOOSING THE BASE POINT FOR THE HYBRID TEST"); P(RULE)
    P("  Superseding an unbounded-birth model that failed its own cap criterion; see the")
    P("  CORRECTION block in this file's docstring for the numbers that condemned it.")

    K, G0, CYCLES = 20, 6, 3
    S, IX = state_index(K)

    P("\n2  EXACTNESS OF THE CLOSED STATE SPACE  (replaces the failed cap-convergence criterion)")
    worst_col = 0.0
    for drug in (False, True):
        L = generator(K, IX, CANDIDATE, drug)
        worst_col = max(worst_col, float(np.abs(L.sum(axis=0)).max()))
    P(f"  states in the triangle g+d <= {K}: {len(S)}   (a {K+1}x{K+1} grid would be {(K+1)**2})")
    P(f"  worst |column sum| of the generator: {worst_col:.2e}   (exact conservation if ~1e-15)")
    step = expm(generator(K, IX, CANDIDATE, False) * 3.0) @ expm(generator(K, IX, CANDIDATE, True) * 6.0)
    p = np.zeros(len(S)); p[IX[(G0, 0)]] = 1.0
    masses = []
    for _ in range(CYCLES):
        p = step @ p
        masses.append(p.sum())
    P("  propagated mass with NO renormalisation: " + ", ".join(f"{m:.15f}" for m in masses))
    P(f"  worst deviation from 1: {max(abs(m-1.0) for m in masses):.2e}")

    P("\n3  K, g0 AND cycles -- Y must not be saturated, and needs upward headroom")
    P(f"  {'K':>4}{'states':>8}{'g0':>4}{'Y':>15}{'headroom up':>14}{'ms/eval':>10}")
    for Kc in (14, 20, 26):
        for g0 in (4, 6, 8):
            t = time.time()
            y = eradication(CANDIDATE, Kc, g0, cycles=CYCLES)
            ms = (time.time() - t) * 1000
            P(f"  {Kc:>4}{(Kc+1)*(Kc+2)//2:>8}{g0:>4}{y:>15.6e}{np.log10(1.0/y):>14.3f}{ms:>10.2f}")
    P(f"  CHOICE: K = {K}, g0 = {G0}, cycles = {CYCLES} for the main run; g0 = 8 for the depth gate.")
    P("  K is a carrying capacity, a property of the model. Y still depends on it -- more room to")
    P("  regrow means harder eradication -- and that dependence is biology, not truncation.")

    P("\n4  SENSITIVITIES AT THE CHOSEN BASE POINT, WITH SIGNS")
    kw = dict(K=K, g0=G0, cycles=CYCLES)
    Sd = {nm: sensitivity(CANDIDATE, nm, 0.02, **kw) for nm in NAMES}
    Sh = {nm: sensitivity(CANDIDATE, nm, 0.01, **kw) for nm in NAMES}
    P(f"  base Y = {eradication(CANDIDATE, **kw):.6e}")
    P(f"  {'rate':>9}{'phase':>7}{'S (h=.02)':>13}{'S (h=.01)':>13}{'rel change':>13}")
    wr = 0.0
    for nm in NAMES:
        rel = abs(Sd[nm] - Sh[nm]) / max(abs(Sh[nm]), 1e-12)
        wr = max(wr, rel)
        P(f"  {nm:>9}{('off' if nm in OFF_RATES else 'on'):>7}{Sd[nm]:>+13.4f}{Sh[nm]:>+13.4f}{rel:>13.2e}")
    P(f"  worst step-halving change {wr:.2e}   (a derivative, not a difference, if < 1%)")
    v = np.array([Sd[nm] for nm in NAMES])
    P(f"  signed sum {v.sum():+.4f}   quadrature norm {np.sqrt((v**2).sum()):.4f}"
      f"   correlated/independent ratio {abs(v.sum())/np.sqrt((v**2).sum()):.4f}")
    order = sorted(NAMES, key=lambda n: -abs(Sd[n]))
    P("  greedy order by |S|: " + str(order))
    P("  near-ties, where greedy has the least to stand on:")
    for i in range(len(order) - 1):
        r1, r2 = abs(Sd[order[i]]), abs(Sd[order[i + 1]])
        if r2 > 0 and r1 / r2 < 1.15:
            P(f"    {order[i]} ({r1:.4f}) vs {order[i+1]} ({r2:.4f}) -- ratio {r1/r2:.3f}")

    P("\n5  THE LINEAR PREDICTION, WRITTEN DOWN BEFORE THE MONTE-CARLO CAN SEE IT")
    P("  Measure the top m by |S| exactly, let chemistry supply the rest at sigma = eps*0.7328")
    P("  orders. Residual spread sd(m) = sigma * sqrt(sum of S^2 over the UNMEASURED rates).")
    P("  A two-sided 90% band needs 1.645*sd(m) <= delta, delta = 0.3010 (x2) or 1.0000 (x10).")
    P(f"  {'m':>3}{'measured last':>15}{'resid |S|':>12}"
      + "".join(f"{'sd@'+e:>9}" for e in ("0.5", "1.0", "2.0"))
      + f"{'m* x2':>8}{'m* x10':>8}")
    pred = {}
    for eps in (0.5, 1.0, 2.0):
        for tol, lab in ((np.log10(2.0), "x2"), (1.0, "x10")):
            pred[(eps, lab)] = next(
                (m for m in range(len(NAMES) + 1)
                 if 1.645 * eps * ORDERS_PER_KCAL * np.sqrt(sum(Sd[n] ** 2 for n in order[m:])) <= tol),
                None)
    for m in range(len(NAMES) + 1):
        q = float(np.sqrt(sum(Sd[n] ** 2 for n in order[m:])))
        row = f"  {m:>3}{(order[m-1] if m else '--'):>15}{q:>12.4f}"
        for eps in (0.5, 1.0, 2.0):
            row += f"{q*eps*ORDERS_PER_KCAL:>9.4f}"
        P(row)
    P("  PREDICTED SMALLEST m REACHING 90% COVERAGE:")
    for eps in (0.5, 1.0, 2.0):
        P(f"    eps = {eps} kcal/mol :  within x2 needs m = {pred[(eps,'x2')]},"
          f"   within x10 needs m = {pred[(eps,'x10')]}   (of {len(NAMES)})")

    P("\n6  COST FORECAST FOR hybrid.py")
    L = generator(K, IX, CANDIDATE, True) * 6.0
    t = time.time()
    for _ in range(20):
        expm(L)
    per_expm = (time.time() - t) / 20
    n_tr, n_eps = 600, 3
    P(f"  one matrix exponential at n = {len(S)}: {per_expm*1000:.2f} ms, one BLAS thread")
    P(f"  the 4/4 phase split means {2**4} + {2**4} = 32 exponentials per trial cover all"
      f" {2**len(NAMES)} subsets")
    P(f"  naive cost would be {2**len(NAMES)} subsets x 2 exponentials = {2**len(NAMES)*2} per trial,"
      f" a {2**len(NAMES)*2/32:.0f}x saving")
    P(f"  full enumeration at {n_eps} error levels, N_TRIALS = {n_tr}:"
      f" {n_eps*n_tr*32*per_expm/60:.1f} minutes of exponentials")
    P(f"  binomial standard error at p = 0.9, n = {n_tr}: {np.sqrt(0.9*0.1/n_tr):.4f}")
    P(f"  CHOICE: N_TRIALS = {n_tr}, common random numbers shared across all subsets, so that")
    P("  greedy and its competitors are compared on identical draws and the paired standard")
    P("  error of a subset-vs-subset difference is far below the unpaired one.")

    open(os.path.join(os.path.dirname(__file__), "RESULTS_hybrid_tune.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
