# What this cell map can answer, and over what horizon

Written by loop 198 from the loop outputs on disk. Every verdict below was checked against the
JSON file that recorded it rather than transcribed.

## Horizons

STATIC      a property with no time in it.
ONE-STEP    predicts a change across one measured interval.
TRAJECTORY  runs forward unaided over multiple intervals.

**Maximum demonstrated horizon: STATIC.**

## The step-forward test

Fitted on intervals inside an early window of the A549 dexamethasone course and scored on intervals
reaching into later timepoints the model never saw. The target is the CHANGE in expression, so
persistence -- predict no change -- is the baseline, and it is the baseline most trajectory models
avoid because smooth biology makes it strong.

    persistence (predict no change)   held-out R2 -0.0295
    training-set mean change          held-out R2 -0.0303
    accessibility-informed ridge      held-out R2 -0.0520

The informed model does NOT beat persistence. Predicting that nothing changes is at least as good as everything this project knows about dynamics, so the map does not step state forward.

## The dynamic rules that were tested

Five candidate rules; one survives and it is unreplicated.

| rule | verdict |
|---|---|
| promoter accessibility leads transcription | holds -- +48 min over 1,310 genes, p 6.4e-58, holds in all three magnitude terciles |
| feedback sign orders response time | refuted -- negative two-cycle 77 min vs positive 70 min, p 0.47 |
| promoter occupancy carries timing | refuted -- rho -0.160 pooled but holds in 1 of 3 magnitude terciles |
| enzymes sharing chemistry are co-timed | refuted -- z -0.8 against 1,000 graph-fixed permutations, stable across hub thresholds |
| curated pathway members are co-timed | refuted -- z +1.1 |
| the accessibility clock replicates outside A549 | refuted -- four timepoints cannot resolve it; the A549 lead reverses when downsampled |

## How this must be quoted

Nine descriptive layers with one unreplicated clock is not a running cell. Every capability above is
single-system: K562 for enhancers, A549 for timing, mixed cell lines for the network. Every
trajectory is bulk, so a fast response in a fifth of the cells and a slow one in all of them give
the same curve. See NOTES_accessibility_clock_status.md for the clock's own qualification.
