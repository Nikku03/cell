# Candidates — what REM computed, per group

**Status of the evidence.** No group's raw data was in hand. Every number in these documents is
either (a) a figure a named paper states in words, quoted with its source sentence, or (b) a
variable swept over a stated range. Nothing is reconstructed, interpolated, or assumed. Where a
needed input could not be retrieved it is marked **UNRETRIEVED** and appears as a data request,
never as an explanation.

All articles were retrieved from **PubMed / PubMed Central**. DOIs are linked at each use.

Computation: `rem/atlas/candidates.py`, output `rem/atlas/RESULTS_candidates.txt`. Gates were
predeclared and committed before the first run (commit `9a042a6`); results and corrections
committed after (`25b8817`).

## Tier 1 — real published inputs, gates run

| No | Group | What REM computed | Headline |
|---|---|---|---|
| 01 | Maiello, Fortune, Flynn, Lin — macaque TB relapse | Conjunctive sterilisation across lesions; identifiability of lesion count vs dissemination rate | The spread hypothesis is false as posed; the covariance is the whole question; lesion count is structurally unidentifiable from relapse counts |
| 02 | Peyrusson, Van Bambeke — intracellular S. aureus persisters | What their two published kill slopes pin down | Persister fraction spans 376× along a curve their data cannot resolve; formation rate bounded at 1.99 /h for free |
| 03 | Srinivas, Baliga — PerSort | Purity of a gate on a 1% subpopulation | 93% efficiency → 11.8% purity at 1% prevalence; effects attenuated 8.5×, so measured effects are floors |
| 04 | Fridman, Balaban — lag-time optimisation | Curvature around the optimum they found | Lag matching reproduced 4/4 from first principles; selection strength rises 42× |
| 12 | Tian (ASU) — resource competition | Correlation between two genes on one shared pool | Positive correlation (0.65 at tight supply) with no growth-feedback term in the model; composing by multiplying marginals off by 5.4× |
| 13 | Marr, Theis (Helmholtz Munich) — residence times | Does memorylessness survive a multi-step gating reaction | It survives and gets stronger; but the rate moves 2.12 orders at identical mean flux |

## Tier 2 — offer stated, inputs not retrieved

| No | Group | Why no computation |
|---|---|---|
| 05 | Roux & Chaves — single-cell signalling dynamics | Nearest neighbour methodologically; novelty margin is thin and is stated as such |
| 06 | Sorger & Spencer — fractional killing | No published rate constants retrieved |
| 07 | McFadden & Hingley-Wilson — TB persistence | No published rate constants retrieved |
| 08 | Wright — antibiotic adjuvants / resistome | Offer is combinatorial, not tail-shaped |
| 09 | Burrows — Pseudomonas screening | Spatial; outside what has been validated |
| 10 | Maxwell & Davidson — gyrase, phage | Self-amplifying agent; outside what has been validated |
| 11 | York University | Affiliation could not be verified against the described work. Do not send. |

## Synthetic biology — why it is the best structural fit

The parts list is complete by construction: you built the circuit. That removes the one unbounded
error class in this whole project. Everywhere else a missing mechanism can cost orders and cannot
be ruled out — measured at 78.51 orders in `gapdetect` gate GD1, mean-invisible. In a circuit you
designed, it cannot happen. Small circuits, characterised parts, and an endpoint that is itself a
distribution: every structural obstacle drops away at once.

## The three things that are true of every document here

1. **Every claim ships with a band or a bar.** A number without one is not reported.
2. **Failed gates are printed, not deleted.** Three of my own defects were caught and corrected
   during this run and all three appear in the output.
3. **The offer is a calculation, not a result.** In each case REM computes something the group
   cannot observe directly, from data they already hold, and says what it would take to be wrong.
