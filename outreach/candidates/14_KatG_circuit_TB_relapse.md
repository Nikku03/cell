# The circuit isoniazid attacks — and why its mean cannot predict relapse

**An addendum to document 01** (Maiello, Fortune, Flynn, Lin — macaque TB relapse), answering a
sharper question: *can we find the problem by looking at the circuit the drug attacks?*

---

## Why isoniazid is the drug where this works

**INH is a prodrug.** It does nothing until the bacterium's own catalase-peroxidase, KatG,
activates it. The killing rate on a given cell is set by that cell's KatG level — **the bacterium
supplies the weapon used against it.**

That makes the drug's target a small stochastic gene-expression circuit rather than a
pharmacokinetic quantity. And it is already measured at single-cell resolution.

According to PubMed, Wakamoto Y, Dhar N, Chait R, Schneider K, Signorino-Gelo F, Leibler S,
McKinney JD, "Dynamic persistence of antibiotic-stressed mycobacteria", *Science* 339:91–95 (2013),
PMID 23288538, [DOI](https://doi.org/10.1126/science.1229858):

> "Single cells expressed catalase-peroxidase (KatG), which activates INH, in stochastic pulses
> that were negatively correlated with cell survival."

> "*Mycobacterium smegmatis* persists by dividing in the presence of the drug isoniazid (INH)…
> this apparent stability was actually a dynamic state of balanced division and death."

> "KatG pulsing and death were correlated between sibling cells."

Independently, according to PubMed, Srinivas *et al.*, *mSystems* 5:e01127-20 (2020),
[DOI](https://doi.org/10.1128/mSystems.01127-20), report that in their translationally dormant
subpopulation "MSMEG_3729 …, which encodes a catalase that converts INH into its active form was
downregulated by ~60-fold."

**Two labs, two methods, the same circuit.**

## The model, deliberately the smallest that can carry the question

KatG copy number produced in geometric bursts, degraded linearly, with killing as an absorbing exit
at a rate **proportional to KatG**. There is **no dormant state and no persister compartment
anywhere in the generator.**

One knob decides everything: how fast KatG fluctuates relative to killing. Scaling the expression
generator holds its stationary distribution *exactly* fixed while changing only the correlation
time. Fast → every cell sees the mean and mean-field is exact. Slow → each cell keeps its level and
survival is the Laplace transform of the KatG distribution.

**Wakamoto et al. report pulsing correlated between sibling cells — heritable, therefore slow.**

Only one quantity is fitted: κ·E[n], set so a mean-field model reproduces the ~5% INH tolerance at
12 h that Srinivas *et al.* report. Nothing else.

## 1. Both limits check out first

| s (fluctuation speed) | exact killing rate | ratio to mean-field |
|---|---|---|
| 0.01 | 0.018517 | 0.074 |
| 1 | 0.221941 | 0.889 |
| 100 | 0.249333 | 0.9988 |
| 1000 | 0.249613 | 0.99987 |

Fast limit converges on mean-field to **1.3 × 10⁻⁴**; frozen limit converges on the quenched
Laplace transform to **2.4 × 10⁻⁴**. Both PASS before anything else is allowed to mean anything.

## 2. Biphasic killing appears from expression spread alone

| s | early slope | late slope | late shallower by |
|---|---|---|---|
| 0.001 | 0.096658 | 0.015089 | 6.41× |
| 0.01 | 0.096718 | 0.018737 | 5.16× |
| 0.1 | 0.097293 | 0.048813 | 1.99× |
| 1 | 0.101301 | 0.096388 | 1.05× |

**6.41× separation at slow fluctuation, in a generator containing no dormant state, no switching,
and no second compartment.** The classic persister signature falls out of KatG heterogeneity by
itself — which is exactly what Wakamoto et al. concluded from watching cells *divide* under drug
rather than sit dormant.

**Recorded honestly:** the gate as I wrote it set a 5× bar without fixing the fluctuation speed,
and **FAILED at the declared setting (1.05×)**. A bar without an operating point is not a test.
That is a defect in my gate, and the sweep above is the repair, labelled as such.

## 3. The decisive result — and it indicts the measurement, not the model

Hold the **mean KatG exactly fixed** and change only its spread:

| burst b | E[n] (held) | Fano | mean-field rate | exact rate | ratio |
|---|---|---|---|---|---|
| 1 | 12.000000 | 1.00 | 0.249644 | 0.244557 | 0.980 |
| 4 | 12.000000 | 4.00 | 0.249644 | 0.230466 | 0.923 |
| 16 | 12.000000 | 16.00 | 0.249644 | 0.187300 | 0.750 |
| 32 | 12.000000 | 32.00 | 0.249644 | 0.149872 | 0.600 |

The mean-field rate is invariant to **2.9 × 10⁻¹⁵** — it depends on the mean alone. The true
killing rate moves **1.63×**.

> **Every bulk assay of KatG returns an identical number down that entire table while the truth
> changes. The information is in the spread, and a mean cannot carry it.**

This is a property of the *measurement*, not of the model — which is the same structural statement
as "sterilization cannot be predicted by PET CT", now made about the drug's own target rather than
about inflammation.

## 4. What it costs over the actual 8-week course

A mean-field model calibrated to 5% tolerance at 12 h, extrapolated to the macaque treatment
duration:

| burst b | exact log₁₀ S at 8 weeks | gap vs mean-field (orders) |
|---|---|---|
| 1 | −142.7 | 3.0 |
| 4 | −134.5 | 11.2 |
| 16 | −109.3 | 36.4 |
| 32 | −87.5 | 58.2 |

**Non-vacuity is the finding, not a caveat.** Both numbers are far below anything anyone should
quote as a probability. What is quotable is the **gap: 3.0 to 58.2 orders, set by burstiness
alone** — and always in the direction of declaring sterilisation certain when it is not.

## 5. A falsifiable prediction on data that already exists

Under frozen heterogeneity, survivors are enriched in low-KatG cells by a computable amount:

| t (h) | exact survivor mean KatG | closed form E[n]/(1+κbt) |
|---|---|---|
| 0 | 12.000000 | 12.000000 |
| 3 | 8.655899 | 8.730660 |
| 12 | 4.444185 | 4.804101 |

> **The mean KatG of the surviving population must FALL during exposure**, by roughly this amount.

Wakamoto et al. imaged KatG in single cells *through* INH exposure. The trajectory needed to
confirm or refute this is already collected. (The closed form tracks the exact answer to ~7% by
12 h; the drift is that a negative binomial is not exactly gamma and the sweep is not exactly
frozen. The prediction is the *direction and rough magnitude*, not the third digit.)

## 6. Control

Remove the heterogeneity — burst size 1 and fast fluctuation — and killing becomes exactly
single-exponential (|late/early − 1| = 0.0000) with mean-field exact to 1 × 10⁻⁴. **Every gap above
is attributable to the spread, not to the solver.**

---

## What is NOT claimed

- The copy-number scale is arbitrary and absorbed into κ. **Only κ·E[n] is calibrated, to one
  published number.**
- *M. smegmatis* is not *M. tuberculosis*, and a macaque granuloma is not a microfluidic chamber.
  **The circuit is the same; the rates are not measured here.**
- Killing is taken as linear in KatG. That is the mechanism of prodrug activation, not a fitted
  form — but it is an assumption and it is load-bearing.
- **Rifampin was given alongside INH in the macaque study.** This speaks only to the INH arm;
  rifampin does not act through a self-supplied activator.
- Three of my own bugs were caught by the module's own assertions during this run (a geometric
  burst mean off by one, a cap that truncated the widest row, and discarded burst flux). All three
  are recorded in the results file rather than tidied away.

## The question, as a question

*Isoniazid is activated by the bacterium's own KatG, and Wakamoto et al. showed that KatG is
expressed in heritable stochastic pulses anticorrelated with survival. If that is so, the surviving
fraction is set by the low tail of the KatG distribution — and we find that holding mean KatG
exactly fixed while changing only its spread moves the true killing rate by 1.63×, and the 8-week
survival by up to 58 orders, with every bulk assay reading identically throughout. Would the spread
of KatG across bacilli within a lesion be measurable in your barcoded necropsy material — and is
that a more promising relapse predictor than any bulk quantity?*

---
*Computation: `rem/atlas/katg.py`, output `rem/atlas/RESULTS_katg.txt`. Gates predeclared and
committed before running (`0550b98`); results and corrections after (`3f4bf1a`).*
