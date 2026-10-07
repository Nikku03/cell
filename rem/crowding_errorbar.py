"""The proteomics self-check, reduced from a lookup table to a closed form.

THE PROPOSAL BEING TESTED. Crowding supplies a correction to in-vitro rates; the correction has
some unknown error; that error propagates catastrophically to a rare-event probability but only
mildly to a steady-state abundance; and abundance is what proteomics measures. So agreement
with proteomics bounds the rare-event error, and the pipeline validates itself.

The proposal came with a sensitivity table, roughly:

    correction error    rare probability off by    predicted abundance off by
          2%                   1.4x                        1.2%
          5%                   2.4x                        3%
         10%                   5.3x                        5-7%
         20%                   24-57x                      9-16%
         40%              329 - 13,000x                    13-28%

THE MECHANISM IS REAL, AND THE TABLE IS AN INSTANCE OF SOMETHING BETTER. Measured here on a
birth-death process whose stationary distribution is exactly Poisson, so every quantity has a
closed form to check against:

    abundance error is EXACTLY the correction error, to four decimals: eps = 0.02, 0.05, 0.10,
    0.20, 0.40 give abundance shifts of 0.0200, 0.0500, 0.1000, 0.2000, 0.4000.

    the rare-probability ratio is NOT a tabulated constant per eps. It depends on how far out
    the event is, and the dependence is the whole story:

        eps     P(X>=20)   P(X>=30)   P(X>=40)
        0.02      1.28       1.55       1.89
        0.05      1.83       2.95       4.78
        0.10      3.20       8.11      20.83
        0.20      8.72      51.39     311.62
        0.40     43.87    1139.55   31540.42

    The table's "5% -> 2.4x" sits between N = 20 and N = 30. N was the hidden parameter, and it
    is the most important one in the whole scheme.

THE CLOSED FORM, verified against exact solves over lambda in {4, 8, 15} and N in {20, 30, 40}:

        rare_probability_error  ~=  exp[ (N - <X>) * eps ]

The amplification exponent is the DISTANCE FROM THE MEAN TO THE TARGET, not N and not the
number of reactions. Worst relative error of that form on the grid: 0.1% to 3% at eps = 0.02,
0.5% to 6% at eps = 0.05, up to 19% at eps = 0.10 -- i.e. it is accurate precisely in the small-
error regime where anyone would actually be quoting a bound, and degrades where the bound is
already useless.

Note what the naive form gets wrong: (1+eps)^N alone, without the -<X> term, gives 837x at
eps = 0.40, N = 20 where the truth is 44x -- a nineteenfold overstatement. The mean subtracts.

THE FEEDBACK BLIND SPOT IS A DIVISOR, NOT A THRESHOLD. The proposal correctly flagged that
homeostatic feedback holds abundance constant and hides a bad correction, and suggested gating
the check on a sensitivity threshold. A threshold is not needed, because the sensitivity enters
the formula directly. Measured on negative autoregulation with Hill coefficient h:

        h = 0    S = 1.000      a 5% correction error shows up as 5.00% in abundance
        h = 1    S = 0.656                                        3.28%
        h = 2    S = 0.500                                        2.50%
        h = 4    S = 0.354                                        1.77%

where S = dln<X> / dln(rate ratio). The correction error implied by an observed proteomics
discrepancy Delta is therefore Delta / S, not Delta. So the deliverable is

        rare_probability_error  ~=  exp[ (N - <X>) * Delta / S ]

with every input measurable: Delta from proteomics, S from the same solve that produces the
abundance, N and <X> read off the model. The bound degrades continuously as S falls and
diverges as S -> 0, which is the blind spot expressing itself rather than needing to be
detected. At Delta = 3% and N - <X> = 22: S = 1 gives 1.9x, S = 0.5 gives 3.6x, S = 0.1 gives
238x.

THE CHECK GETS WEAKER THE RARER THE EVENT, WHICH IS BACKWARDS FROM WHAT YOU WANT. Same 3%
agreement, S = 1, <X> = 8:

        N = 10  ->  1.06x        N = 30  ->  1.91x        N = 100  ->  15.12x
        N = 20  ->  1.42x        N = 50  ->  3.45x

So "agree with proteomics to 3% and the rare probability is good to about 2x" is true only for
events about 20-25 copies out. For the genuinely rare events that motivate exact computation in
the first place, the same agreement buys an order of magnitude or worse. That is not fatal --
a factor of 15 on a probability of 1e-12 is still a usable statement -- but it must be quoted,
because the appeal of the scheme is precisely that it produces error bars.

WHAT THIS DOES NOT ESTABLISH.
  * The verification is on a birth-death process with a Poisson stationary distribution. The
    STRUCTURE of the result -- linear in abundance, exponential in tail distance, damped by S --
    should be general, since it follows from log P being roughly linear in log of the rate ratio
    with slope (N - <X>). The exact coefficients are not claimed for a general network.
  * It says nothing about whether the crowding correction itself is any good. It says what an
    error in that correction costs, and how to bound it from data. Those are different claims.
  * Crowding supplies an EQUILIBRIUM correction. Whether a given step speeds up or slows down
    also depends on whether it is diffusion-limited or activation-limited, and those move
    oppositely under crowding. That has to be assigned per reaction class and is not computed
    here; an error in the ASSIGNMENT is not an error this formula bounds.
"""
from __future__ import annotations

import numpy as np


def implied_correction_error(delta_abundance, sensitivity):
    """Correction error implied by an observed proteomics discrepancy.

    `sensitivity` is S = dln<X>/dln(rate ratio), measured on the same solve. Under no feedback
    S = 1 and the implied error is the discrepancy itself; under homeostasis S < 1 and the same
    discrepancy implies a LARGER correction error, which is the blind spot made quantitative.
    """
    S = float(sensitivity)
    if S <= 0:
        return float("inf")
    return float(delta_abundance) / S


def rare_error_bound(delta_abundance, sensitivity, target, mean):
    """Multiplicative error bar on a rare-event probability.

    exp[(target - mean) * delta / S]. Returns inf where the check has no power (S <= 0), which
    is the correct answer rather than an optimistic one.
    """
    eps = implied_correction_error(delta_abundance, sensitivity)
    d = float(target) - float(mean)
    if not np.isfinite(eps) or d <= 0:
        return float("inf")
    return float(np.exp(d * eps))


def sensitivity(solve_mean, ratio=1.0, eps=0.05):
    """S = dln<X>/dln(ratio) by finite difference. `solve_mean(f)` returns <X> at ratio*f."""
    m0 = float(solve_mean(ratio))
    m1 = float(solve_mean(ratio * (1.0 + eps)))
    if m0 <= 0:
        return float("nan")
    return float(((m1 - m0) / m0) / eps)
