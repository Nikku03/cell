# SIREN + HNN data-scaling smoke test

This directory contains an empirical follow-up to the April 2026
"Dark Manifold salvage" tests. It is **not** production code, **not**
a test of the Dark Manifold concept, and **not** integrated with the
Syn3A simulator.

## Scope

Two of the three components in the original salvage tests failed:

* SIREN spatial field — failed at 500 training points (test MSE
  0.039 vs plain MLP 0.008).
* Hamiltonian Neural Network — passed on harmonic oscillator;
  failed 14× on damped oscillator.

This smoke test re-runs both at substantially more training data to
distinguish **data-sparsity failures** (would resolve with more data)
from **structural failures** (would not).

## What this test does NOT do

* It does **not** test the full Dark Manifold concept (4D spacetime
  field with dark matter coupling, quantum fluctuations as sampling,
  superposition collapse, cognitive scaffold). That architecture was
  never built and remains untested.
* It does **not** establish viability of either technique for cell
  biology. Cell-biology training data is fundamentally sparse
  (typically 100s–1000s of measured points). Even if SIREN wins at
  500k synthetic points, that does not establish viability for
  cell-biology applications.
* It does **not** modify production simulator code. No changes to
  `cell_sim/`, `cell_sim_rust/`, or fact files. The v15, v16, R1,
  R2a, R2b state is unchanged.

## Layout

```
experiments/dark_manifold_salvage_followup/
├── README.md                         (this file)
├── siren_data_scaling_test.py        SIREN @ 500/5k/50k/500k points
├── hnn_data_scaling_test.py          HNN @ 100/1k/10k/100k trajectory points
├── results/
│   ├── siren_results.csv
│   ├── hnn_results.csv
│   └── plots/
│       ├── siren_data_scaling.png
│       └── hnn_data_scaling.png
└── FINDINGS.md                       interpretation
```

## Reproducibility

Both tests are deterministic at `seed=42`. Run from repo root:

```
python experiments/dark_manifold_salvage_followup/siren_data_scaling_test.py
python experiments/dark_manifold_salvage_followup/hnn_data_scaling_test.py
```

Each test caps wall time per data condition to keep CPU runs
tractable. The cap is documented in the test source and in the
`results/*.csv` `wall_s` column.

## Hardware envelope

These tests were run on a 4-core CPU with 15 GB RAM, no GPU. Both
training scripts respect a per-condition wall-time cap (default
5 minutes); conditions that would exceed the cap are reduced and the
reduction is documented in the CSV output.
