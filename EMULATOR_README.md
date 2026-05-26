# Differentiable Syn3A cell emulator (LGNN + physics constraints)

A neural emulator of the JCVI-Syn3A whole-cell simulator (Thornburg et al.,
*Cell* 2026, [DOI](https://doi.org/10.1016/j.cell.2026.02.009)). Trained on
50 parquet trajectories from the upstream's 4D Whole-Cell Model (4DWCM),
runs ~100–1000× faster, and is **differentiable end-to-end** — a capability
the upstream simulator structurally cannot provide.

This is the LGNN emulator track of the project. The earlier `cell_sim/`
event-driven Gillespie simulator is described in the top-level `README.md`.

## Quick start

The model trains and evaluates entirely on Colab. Open the notebook:

> https://colab.research.google.com/github/Nikku03/cell/blob/claude/bio-inspired-neural-network-6dFAZ/colab_cell_emulator.ipynb

then **Runtime → Run all** (`~15 min` on a Blackwell GPU). The setup cell will:

1. Mount Drive
2. Pull SBML, kinetics, initial concentrations, complex formation, gibbs.csv, LargeSubunit from the upstream repo (or Drive cache)
3. Index the 50 `counts_and_fluxes*.parquet` trajectories under `MyDrive/`
4. Train the model, evaluate, save checkpoint to Drive

Outputs land in `MyDrive/cell_emulator_v13.pt` (model) and
`MyDrive/cell_traj_51_v13.npy` (generated 51st trajectory).

## What's in this project

```
colab_cell_emulator.py        ← single-file project, ~3,700 lines
colab_cell_emulator.ipynb     ← auto-generated from .py for Colab
EQUATIONS_TODO.md             ← the build list and module status
REFERENCE_SIMULATOR_TOUR.md   ← tour of the upstream paper (Thornburg 2026)
test_conservation_violations.py  ← post-hoc test: do we refuse simulator's physics violations?
```

## Architecture

```
state(t-7..t) ─► TemporalContext (2-layer transformer, ~178k params)
                  └─► context vector c added to LGNN hidden as broadcast bias
                      (v15.0 hybrid — "two-cortex" model)

state(t) ─┬─► LGNN (3-layer CfC graph, hidden=64, ~1.17M params)
          │      └─► 36,712 graph edges from 7 sources (SBML co-occurrence,
          │            central-dogma per gene, enzyme→flux, subunit→complex,
          │            protein↔metabolite, 50S assembly, self-loops)
          │
          ├─► MetabolismCore (frozen) — bi-bi rate law for ~115 SBML reactions
          │     with measured k_cat / K_m; ΔG° < −10 kJ/mol → irreversible-forward
          │
          ├─► VolumeCore — dynamic V_L from membrane-lipid sum
          │
          ├─► CentralDogmaCore — first-order tx/tl/decay per gene,
          │     k_tx_g, k_tl_g calibrated per-gene from initial counts
          │     (Thornburg 2026 promoter-strength formulation)
          │
          ├─► StochasticHead — per-species log σ for NLL loss
          │
          ├─► KnockoutAugmentation — random species-zero during training,
          │     persistent through K-step rollout
          │
          ├─► Hypothesis aux loss — Tier-2 monotone + pair constraints
          │     discovered by 6 lens algorithms
          │
          └─► ATP ledger — soft penalty when net ATP < calibrated maintenance floor
              ΔG° sign clamp — irreversible-forward for strongly exergonic reactions
              σ-anchor + trajectory-variance loss — anti-mode-collapse
```

Each box is opt-in via a `USE_*` flag at the top of the file. The "match the
paper" calibration uses tunable knobs (`CD_TRANSLATION_SCALE`, `SAMPLE_NOISE_SCALE`,
`LAMBDA_*`, `ATP_MAINTENANCE_RATE` — calibrated from data, not literature) all
visible in the train log.

## Current results (v15.0)

> Update this section with numbers from the latest Colab run.

| Metric | Persistence baseline | Our model | Upstream actual | Paper target |
|---|---|---|---|---|
| 1-step R² | 0.781 | – | – | – |
| Rollout R² (mean) | – | – | – | – |
| Top-12 KO precision | – | – | – | – |
| Doubling time (min) | – | – | – | 105 |
| Ribosome fold | – | – | – | 1.76 |
| ori:ter ratio | – | – | – | 1.28 |
| Protein fold (median) | – | – | – | 1.40 |

**Wall-clock per cell cycle**:
- Upstream 4DWCM: 4–6 days × 2 GPUs (~250 GPU-hours)
- This emulator: ~15 min training + inference in seconds

## The build list

`EQUATIONS_TODO.md` is the architectural build list. Each module was added
incrementally with a standalone test, an integration check, and a Colab
validation run. The current state:

| # | Module | Status |
|---|---|---|
| 1 | MetabolismCore (bi-bi) | wired |
| 2 | VolumeCore (dynamic V_L) | wired |
| 3 | CentralDogmaCore (per-gene tx/tl/decay) | wired, per-gene calibrated |
| 4 | AssemblyCore (mass-action) | wired but disabled (low coverage on data) |
| 5–6 | Transport, tRNAChargingCore | subsumed by MetabolismCore |
| 7 | KnockoutAugmentation | wired |
| 8 | ReplicationCore (DnaA + replisome) | deferred (state-machine complexity) |

Plus the v14 physics-list:

| # | Item | Status |
|---|---|---|
| #2 | ΔG° sign clamp | wired (day 1) |
| #7 | Ribosome pool cap | wired (day 2, then refactored to linear ribo scaling in v14.3) |
| #3 | ATP energy ledger | wired (days 3–4, data-calibrated rate in v14.2) |
| #8-lite | σ-calibration anchor + trajectory-variance loss + sampled rollouts | wired (day 5, strengthened in v14.7) |

## What this emulator does that the upstream can't

- **Differentiable**: gradients flow through every step. Enables:
  - Parameter sensitivity in one backward pass (vs 10³ finite-difference runs)
  - Inverse design (given desired phenotype, solve for parameters)
  - Gradient-based drug-target ranking
  - Identifiability analysis
- **Fast**: full cell-cycle prediction in seconds, vs days for the simulator
- **Smooth interpolation**: query state at any time, run backward, do bifurcation analysis on the learned dynamics

## What this emulator gives up

- **Stochastic discreteness**: Gaussian σ approximates Poisson noise. For
  rare-event statistics or low-copy-number bursts, this is a structural limit.
- **Trajectory R² ≤ 1**: by construction it can't exceed its own training
  target. Practical ceiling is ~0.85, current is ~0.5–0.6 depending on config.
- **No spatial structure**: the upstream's 10 nm RDME lattice is collapsed
  to mean field.
- **No chromosome dynamics**: DNA replication, SMC, topoisomerases are
  handled implicitly by the LGNN without mechanism.

## How "better than simulator" is framed

Three legitimate senses (see `REFERENCE_SIMULATOR_TOUR.md` for full
discussion):

1. **Refuse universal-law violations.** Where the simulator's numerical
   coupling produces conservation-violating transients, our constrained
   model can in principle produce the physically admissible trajectory
   instead. Test infrastructure exists (`test_conservation_violations.py`);
   results so far are mixed.
2. **Tail statistics from massive sampling.** We can run thousands of
   rollouts in the time the simulator does one. Limited by σ approximation.
3. **Data fusion.** Fine-tune on real experimental measurements (Breuer
   essentiality, growth-rate distributions, fluorescence morphologies) on
   top of the simulator-derived base. Future work; the cleanest path to a
   defensible scientific claim.

## Repo / commit conventions

All work happens on branch `claude/bio-inspired-neural-network-6dFAZ`.
Commit messages follow `vMAJ.MIN: short description` and explain what
changed, why, and what the expected effect is.

Standalone tests live in `/tmp/test_*.py` (developer-local; not committed
because they pre-populate input files the user runs separately).

## License & attribution

Built on top of:
- Thornburg et al., *Cell* 2026 — 4DWCM model, kinetic parameters
- Thornburg et al., *Cell* 2022 — well-stirred predecessor
- Breuer et al., *eLife* 2019 — essentiality dataset, kinetic params
- Hasani et al., 2022 — Liquid Time-constant networks (CfC formulation)
- Raissi et al., 2019 — PINN concept

The `colab_cell_emulator.py` file is original work; everything it consumes
(SBML, xlsx files, parquet trajectories, gene tables) is from the public
Luthey-Schulten Lab repositories and dependent literature.
