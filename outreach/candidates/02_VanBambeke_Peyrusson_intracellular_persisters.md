# Intracellular *S. aureus* persisters — Peyrusson & Van Bambeke

**Their paper.** According to PubMed, Peyrusson F, Nguyen TK, Najdovski T, Van Bambeke F, "Host
Cell Oxidative Stress Induces Dormant *Staphylococcus aureus* Persisters", *Microbiology Spectrum*
10:e0231321 (2022), PMID 35196815, [DOI](https://doi.org/10.1128/spectrum.02313-21).

**Why this paper and not another.** It publishes something rare: a **fully specified biphasic kill
curve** — two slopes and a breakpoint, in numbers, in the text.

> "The kill rate is estimated as a 0.2- or 0.3-log decrease in propidium iodide-negative events per
> hour over the first 3 h of incubation in J774 and human macrophages, respectively, and a
> 0.02-log decrease per hour for longer incubations up to 48 h in both cell types."

That is enough to ask a precise question: **what do those two slopes actually pin down?**

---

## 1. The answer: not the thing you need

For the two-state model (growing cells killed at rate *k*, switching to dormant at rate *a*,
waking at rate *b*), the two measured slopes are **two equations in three unknowns**. The solution
set is a curve, not a point — and every point on it reproduces their published curve *equally
exactly*.

**J774 macrophages** (0.2 then 0.02 log₁₀/h):

| a (/h) | k (/h) | b (/h) | early | late | persister fraction | log₁₀ plateau |
|---|---|---|---|---|---|---|
| 0.0001 | 0.46063 | 0.039950 | 0.200000 | 0.020000 | 0.002497 | −1.4588 |
| 0.0030 | 0.46381 | 0.046169 | 0.200000 | 0.020000 | 0.061014 | −1.3767 |
| 0.0300 | 0.49442 | 0.049127 | 0.200000 | 0.020000 | 0.379136 | −1.0531 |
| 0.3000 | 0.91220 | 0.062002 | 0.200000 | 0.020000 | 0.828724 | −0.5729 |
| 1.0000 | 2.55886 | 0.064379 | 0.200000 | 0.020000 | 0.939515 | −0.5400 |

Slope drift across the entire feasible curve: **9.0 × 10⁻¹⁵**. Every row is an exact fit.

The persister fraction spans **376×** (J774) and **391×** (human macrophages) along a curve their
time-kill data cannot resolve. **That fraction is the quantity a dosing schedule acts on.** Two
slopes look like a complete characterisation and are not.

## 2. What their curve *does* pin, and it is new

The feasibility boundary is a real constraint. Above a certain switching rate, **no** (k, b)
reproduces their own two slopes:

- **J774 macrophages: a ≤ 1.9892 /h**
- **Human macrophages: a ≤ 0.8293 /h**

Bisected to 10⁻⁴ relative — not read off a grid. (Read off the grid these would have been 1 and
0.3 /h, out by factors of 2 and 2.8. That grid-limiting defect is recorded in the output.)

This bound costs no new experiment. It follows from numbers already in their paper.

## 3. The one measurement that collapses the family

The **slow-phase intercept** — the late line extrapolated back to t = 0, i.e. the persister plateau
height — varies along the curve:

- human macrophages: **1.385 orders** across the family → PASS against the predeclared 1-order bar
- J774 macrophages: **0.919 orders** → **FAIL** against the same bar

Worst case decides, so the honest verdict is **FAIL**: one intercept *narrows* the family, it does
not close it. Two intercepts at different host-ROS levels — which their experimental design
already produces — should.

This number is in their Fig 1B. It is an intercept, not a new experiment.

## 4. Control

Setting a = 0 (no persisters) makes killing single-exponential: early and late slopes agree to
3.9 × 10⁻¹³. The biphasic shape is the mechanism, not the driver. ✓

---

## A correction, recorded

The first implementation of this case compared their **0–3 h average** slope against an
**instantaneous** slope at t = 1 h, and fixed *k* in closed form as if the switching rate were
negligible. The predeclared 1% gate **FAILED at 2.98 × 10⁻¹**. The bar was right; the instrument
was wrong. Reimplemented against the observable they actually report — same bar, now 9 × 10⁻¹⁵.
The failed run is kept in the output rather than deleted.

## What is NOT claimed

- This does not fit their data. It shows what their data **cannot** fit.
- The two-state model is the simplest one consistent with a biphasic curve. A deeper dormancy
  continuum — which their own paper argues for — would widen the family further, not narrow it.

## UNRETRIEVED — the data request

| What | Why |
|---|---|
| The slow-phase intercept (persister plateau height) from Fig 1B, per cell type | Collapses a 376× family; §3 |
| Single-cell resuscitation lag distribution (Fig 1G, measured, plotted) | b is 1/mean lag — this over-determines the fit and turns the degeneracy into a consistency check |

The second is the interesting one. They **measured single-cell lag times** for exactly these
persisters. That distribution independently determines *b*, which fixes *a* and *k*, which closes
the whole family — and then the same machinery that reproduced Fridman & Balaban's lag-matching
optimum from first principles can be pointed at oxacillin dosing intervals.

## The question, as a question

*If the slow-phase intercept and your measured single-cell lag distribution were fed in together,
they would over-determine the persister formation and waking rates — could that be used as a
consistency check, and then to ask which oxacillin exposure interval a dormancy-depth-heterogeneous
population is least able to survive?*

---
*Computation: `rem/atlas/candidates.py` case B. Full output: `rem/atlas/RESULTS_candidates.txt`.*
