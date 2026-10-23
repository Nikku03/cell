# Lag-time optimisation — Fridman & Balaban

**Their paper.** According to PubMed, Fridman O, Goldberg A, Ronin I, Shoresh N, Balaban NQ,
"Optimization of lag time underlies antibiotic tolerance in evolved bacterial populations",
*Nature* 513:418–421 (2014), PMID 25043002, [DOI](https://doi.org/10.1038/nature13469).

> "the lag time of bacteria before regrowth was optimized to match the duration of the
> antibiotic-exposure interval"

**This is the validation anchor, not an offer.** Before claiming anything, the engine had to
reproduce a result it was not told about.

---

## 1. What was reproduced

Sweeping mean lag time against a fixed drug-exposure duration and taking the exact eradication
probability from the chemical master equation, the lag that maximises survival came out **equal to
the exposure duration**, at **4 of 4** exposure durations tested — as a computed optimum, from
first principles, with no fitting to their result.

A mandatory ablation was run: with the dormant state removed, the matching **vanishes** and killing
becomes single-exponential. The testbed measures the mechanism, not the driver.

## 2. What is new — the curvature, not the peak

Their result gives the location of the optimum. REM gives its **shape**, which predicts how tightly
replicate evolved populations should cluster:

| T_on (h) | peak lag | fold-width of the peak | max abs(selection) | curvature |
|---|---|---|---|---|
| 1.5 | 1.48 | 15.0× (≥ grid) | 0.067 | −0.050 |
| 2.5 | 2.32 | 15.0× (≥ grid) | 0.200 | −0.174 |
| 3.5 | 2.91 | 15.0× (≥ grid) | 0.841 | −0.371 |
| 5.0 | 4.57 | 6.8× | 2.800 | −0.704 |

**Three of the four widths are grid-limited**, not measured — the e-fold region reaches both ends
of the lag grid, so those rows are **lower bounds** and the true peaks are wider still. That
weakens the magnitude and is stated here rather than buried.

**The robust statement is the selection strength**, which is not grid-limited: it rises **42×**
from 0.067 at a 1.5 h exposure to 2.800 at a 5.0 h exposure.

## 3. The testable prediction

> Across replicate evolved populations at a given exposure duration, evolved lag should **scatter
> widely at short exposures and cluster tightly at long ones.**

This is a trend across conditions rather than a single number, which is harder to match by chance —
and **their replicate populations already carry it.** No new experiment.

---

## What is NOT claimed

- The peak locations are a reproduction of their result, not a new one.
- Three of four fold-widths are lower bounds, not measurements.
- The absolute selection strengths depend on model rate constants; the **42× trend across exposure
  durations** is the claim that survives.

## The question, as a question

*Does the scatter in evolved lag across your replicate populations narrow as the exposure interval
lengthens — and if so, does it narrow by roughly the factor the selection curvature predicts?*

---
*Computation: `rem/atlas/persistence.py`, output `rem/atlas/RESULTS_selection.txt`.*
