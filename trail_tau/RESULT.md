# RESULT — state-relaxation time `tau` from Roux/Chaves single-cell TRAIL data

## 1. Headline

**`tau` is not recoverable from this data, because the deposited dataset contains exactly one
static state measurement per cell and no time axis along which state could relax** — no lineage
identifiers, no pre-treatment frames, and no repeated measurement of the state variable. The
deposited archive holds 1,324 cells, each contributing a single fitted `(pC8(0), FLIP(0))` pair,
plus 24 example FRET trajectories that report the death cascade rather than the cell state.

This is not a power problem. No reanalysis of these files can produce `tau`.

## 2. Gate table

| Gate | Status | Deciding value |
|---|---|---|
| **Phase 0 (a)** lineage IDs | **ABSENT** | 0 identifier variables among 39; all 5 integer-typed variables are `Tend_*` death times |
| **Phase 0 (b)** pre-treatment series | **ABSENT** | `t_obs = linspace(5,600,120)` min; first frame is +5 min **after** dosing |
| **Phase 0 (c)** repeated state measurement | **ABSENT** | state = fitted initial conditions; `d(pC8)/dt`, `d(FLIP)/dt` have binding terms only, no synthesis or decay |
| **Phase 0 (d)** per-cell fitted parameters | **PRESENT** | 1,324 cells x `(pC8(0), FLIP(0))`; 24 cells x full 5-parameter vector |
| **Phase 0 (e)** fate labels / death times | **PRESENT / partial** | fate for all 1,324 via R vs S file split; death times for 24 cells only |
| **R0** reproduce a published quantity | **PASS** | median Pearson r = **0.9965**, min **0.9789**, across all 24 cells; 0 excluded |
| **R0b** tolerant fraction per dose | **PASS (consistency)** | 0.4587 / 0.2754 / 0.1115 at 25/50/100 ng, from counting deposited fits |
| **R0c** their simulation vs their measurement | **PASS** | simulated tolerant fraction 0.460 / 0.262 / 0.101 vs measured 0.4587 / 0.2754 / 0.1115 |
| **T1–T6** (`tau` estimation) | **NOT RUN** | gated off by the predeclared decision rule; no route exists |
| **Phase 3** kill function | **NOT RUN** | requires `tau` to be carried into Phase 4; running it alone would invite the invented-constant failure the task forbids |
| **Phase 4 / S1–S6** schedule `ratio` | **NOT RUN** | requires `tau` |
| **P1** design estimator unbiased | **PASS** | median 3.0080 vs true 3.0 at n=1600 (rel dev 0.0027) |
| **P2** design CI width scales as n^-1/2 | **PASS on 3rd attempt** | robust slope **-0.5575**; first two attempts FAILED at -2.46 and -0.98 (see §7) |
| **P3** required n invariant to true `tau` | **PASS** | CI factor 1.699 / 1.625 / 1.686 at true `tau` = 1 / 3 / 10 h; spread 4.4% (bar 15%) |
| **P4** noise biases `rho_0` not `tau` | **PASS** | median `tau` = 2.987 / 3.032 / 3.003 / 2.985 at noise 0 / 10 / 25 / 50% of state sd |

R0 is a real test, not decoration: I re-implemented the paper's ODE independently in Python and
integrated it with their deposited per-cell parameters. Mis-ordering the 5-vector, mis-scaling
time by `K1`, or using the wrong `TRAIL0` collapses the correlation. It could have failed; it did
not. That is what makes the negative verdict credible — I can read their data correctly, and the
thing I am reporting missing is genuinely missing.

## 3. `tau` by each route

| Route | Requires | Available | Result |
|---|---|---|---|
| A — lineage correlation decay | mother/daughter or sister links | **No** | not attempted |
| B — pre-treatment autocorrelation | frames before TRAIL | **No** | not attempted |
| C — repeated state measurement | same cell's state at ≥2 times | **No** | not attempted |

No route was available, so the question of whether routes agree does not arise.

**Route C deserves a stronger statement than "absent from this archive".** In this assay the
state `(pC8(0), FLIP(0))` is *inferred by fitting the death-cascade model to a cell's FRET
response after TRAIL*. Measuring a cell's state therefore requires dosing it — which either kills
it or converts it into a survivor. **The measurement is destructive.** Route C is not merely
missing from the deposit; it is unavailable from this assay in principle, and no amount of
re-imaging with these reporters would supply it.

## 4. Kill-function fits

**Not run.** Phase 3 exists to feed Phase 4, and Phase 4 requires `tau`. Fitting `s(x)` in
isolation and reporting it would produce a number with no destination, and the task's own record
shows an unrecorded kill-function choice previously moved a headline answer by a factor of 2.6.
The honest action is to leave it unfitted and say so.

For the record, the data to do it later **is** present: 1,324 cells with fate labels and a
2-D state, and the paper's own dose-dependent hyperplane defines the projection `x`.

## 5. `ratio` versus gap

**Not computed.** `ratio(g)` is a function of `tau`. With `tau` unmeasured, every value of
`ratio` would be a function of an invented constant. Reporting one — even with caveats — is
precisely the failure mode listed in §9 of the task.

What can be said without `tau`, from structure alone: `ratio > 1` requires the state to be
correlated across the dosing interval. As `g/tau -> infinity`, `ratio -> 1` exactly. So the
qualitative claim "close-spaced doses waste drug" is true for *some* spacing; the whole content
is the number `tau`, and it is missing.

## 6. Practical statement

**Not supported.** No dose spacing can be recommended, in hours or otherwise.

## 7. Everything I got wrong

Four errors, three of them mine and caught by my own checks.

**(i) FRET monotonicity — wrong test, stated as a surprise.** I predicted the experimental FRET
traces would be monotone non-decreasing, reasoning from the model where `dFRET/dt = K_FRET*C8`
with `C8 >= 0`. The check returned **not monotone**, most negative single-frame step −0.00345.
Following rule 6 I suspected the test first, and the test was wrong: monotonicity is a property
of the *model's* FRET variable, whereas the deposited trace is a *noisy measurement* of it. 175
negative steps, median −0.00087, against a dynamic range of ~0.81 — i.e. noise at ~0.4% of range.
The conclusion survived (FRET is a cumulative death reporter, not a stationary state variable)
but my stated reason for it was wrong and is corrected here.

**(ii) Gate P2, first attempt — wrong statistic.** I measured CI width as `(CI_factor - 1)`.
`tau` is a scale parameter, so its interval width belongs on the log scale. The mis-stated
statistic produced a log-log slope of **-2.46** against a `-0.5` bar: a spectacular FAIL that was
entirely an artefact of my own summary. The bar was right; the statistic was wrong.

**(iii) Gate P2, second attempt — right statistic, contaminated by heavy tails.** Using
`log(CI_factor)` over all n still FAILED at **-0.98**, because at n=25 the 95% interval spans a
factor of 1.8x10^14 — a handful of runaway maximum-likelihood fits. That is small-sample heavy
tail behaviour before asymptotic normality sets in, not a broken estimator. The third attempt
used `log(P75/P25)`, a robust spread that drops **no rows**, and PASSED at **-0.5575**. As an
independent check, the fragile statistic restricted to n >= 200 gives **-0.5593** — the two agree,
which is what convinced me the estimator really is root-n rather than that I had found a third
statistic that flattered it.

**(iv) The binned estimator was limited by my binning, not by the data.** My first Route A
estimator binned pairs by separation and required >= 8 pairs per bin; at n=25 fewer than 3 bins
survived and it returned `nan`. That is a property of a choice I made. Replacing it with an
unbinned maximum-likelihood fit changed the required sample size for a factor-2 CI from **~536 to
~236 pairs** — a factor of 2.3 purely from estimator choice. Both are reported in §8 rather than
the more flattering one.

**A prediction that was not tested.** `PREDECLARE.md` recorded the expectation that `tau`, if
measurable, would be of order the cell doubling time. Nothing here tests it. It is recorded as an
untested expectation, not as a result.

## 8. What would be needed, and how much

Two designs. Both give `tau`; the first uses the existing assay, the second is cleaner.

### Design 1 — sister pairs (works with the destructive assay they already have)
Track cells through division. At a chosen separation `t` after a division, dose **both** sisters
and fit each one's state from its FRET response. Each cell is measured once, destructively, but
the *pair* yields `rho(t)`. Vary `t` across pairs, then fit `rho(t) = rho_0 * exp(-t/tau)`.

Sample size, computed by simulating the estimator (`scripts/07_design_power_v2.py`,
`scripts/08_p2_robust.py`), with separations spread uniformly on `[0, 2*tau]` and measurement
noise at 10% of the state's standard deviation:

| target 95% CI on `tau` | sister pairs needed (ML) | (binned estimator) |
|---|---|---|
| within a factor of 5 | **~60** | ~144 |
| within a factor of 2 | **~236** | ~536 |
| within a factor of 1.5 | **~598** | ~1,435 |

**This can be quoted without knowing `tau`** because gate P3 confirms the requirement is
invariant to the true value once the design is expressed in units of `tau` (CI factor 1.699 /
1.625 / 1.686 for true `tau` = 1 / 3 / 10 h; 4.4% spread). Practically: pick the widest sensible
window, e.g. separations spanning 0–24 h, and the design is valid for any `tau` in roughly 1–12 h.

Gate P4 shows measurement noise **attenuates `rho_0` but leaves `tau` unbiased** (median `tau`
2.985–3.032 across noise from 0 to 50% of state sd), because the inference error is independent
between sisters and does not depend on `t`. So imperfect state inference costs precision, not
accuracy — which is what makes this design workable despite a noisy destructive readout.

The state scale used in the simulation is the **measured** one: sd of log10 `FLIP0` = 0.2141 and
of log10 `pC80` = 0.2075, across all 1,324 deposited cells. Noise was swept, not assumed.

### Design 2 — a live, non-destructive state reporter
Endogenous tags on CASP8 and CFLAR (the two state variables) imaged **without** drug. Then a
single cell's state can be followed directly and `tau` read from the autocorrelation of its
trajectory. This removes the destructive-measurement problem entirely and also supplies Route B.
Requirement: traces spanning >= 3`tau` at a frame interval << `tau`, with the finite-trace
autocorrelation bias corrected by simulating an Ornstein–Uhlenbeck process at the same sampling
and length. The 5-minute frame interval already used is ample for any `tau` above ~30 min.

## 9. Limits and modelling choices

Every choice made, and how it would move the answer:

1. **The design calculation assumes exponential relaxation.** If state relaxation is multi-
   exponential or has a non-relaxing component (`rho(t) = A*exp(-t/tau) + C`), the required
   sample sizes in §8 are underestimates, and `C` would need its own test. This is why the task
   specifies fitting both forms; neither could be fitted here.
2. **The design calculation assumes sisters start perfectly correlated** (`rho_0` free but state
   shared at division). If division itself partitions state asymmetrically, `rho_0 < 1` and
   precision falls. The ML fit estimates `rho_0`, so this costs sample size rather than
   introducing bias.
3. **Gaussian state on a log scale.** The measured `FLIP0`/`pC80` distributions are
   approximately lognormal (sd of log10 ~0.21). Heavier tails would degrade the Pearson-based
   binned estimator more than the ML one.
4. **Noise treated as independent between sisters.** If the inference error is *correlated*
   between sisters — plausible if they are fitted in the same imaging field with shared
   background — it would inflate `rho` at all `t` and bias `tau` **upward**. This is the one
   assumption in §8 that could produce a biased answer rather than a wider interval, and it
   should be checked with a same-cell technical replicate.
5. **`x` was never defined**, because Phase 3 was not run. Had it run, `x` would be the
   projection onto the normal of the paper's dose-dependent hyperplane, and that choice would
   need recording.
6. **Excluded from analysis:** 41 `__MACOSX` resource-fork stubs and 2 `.DS_Store` files, which
   contain no data. **No cell was dropped from any reported quantity** — R0 compared 24 of 24
   cells, and the counts in §2 use all 1,324.

## 10. Files

```
INVENTORY.md              Phase 0, with (a)-(e) answered and evidence for each
INVENTORY_raw.txt         per-variable dump of all 39 .mat files
PREDECLARE.md             gates and the decision rule, committed before any fit
RESULT.md                 this file
scripts/00_acquire.sh     download + SHA-256
scripts/01_unpack.sh      unpack
scripts/02_inventory.py   Phase 0 inventory
scripts/03_R0_reproduce.py    Gate R0 -- re-implements their ODE, reproduces Figure 2
scripts/04_phase0_structural.py   structural checks 1-4
scripts/05_phase0_followup.py     corrected CHECK 1 framing; settles CHECK 3
scripts/06_design_power.py        design power, first version (P2 FAILED -- kept)
scripts/07_design_power_v2.py     design power, ML estimator (P2 FAILED again -- kept)
scripts/08_p2_robust.py           P2 third attempt, PASSED
scripts/09_figure.py              figures
data/raw/                 unpacked archive, unmodified
figures/fig1_state_distributions.png   what the data contains: one static state per cell
figures/fig2_required_sample_size.png  Route A design: pairs needed vs CI width
```

The two figures the task requested — correlation decay with fit, and `ratio` vs gap — are
**absent by necessity**. Both are plots of `tau`. Drawing either would require inventing it.
