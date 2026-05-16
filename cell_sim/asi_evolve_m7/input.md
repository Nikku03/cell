# Task: Improve the M7 cell-state dynamics surrogate

## Background

**M7** is a Graph Neural Network with ~3.4M parameters that predicts
the next state of a simulated bacterial cell (Syn3A, a minimal-genome
mycoplasma with 493 genes). It replaces the Luthey-Schulten whole-cell
simulator for the **knockout screening use case**: each KO takes ~3
ms on M7 vs ~6 hours on the simulator (~7 million× speedup).

The model uses:
- A 3-layer CfC-Attention GNN encoder over a species-reaction graph
  (8572 nodes, ~150k edges)
- A PINN output head that enforces Δx = S·v stoichiometric conservation
  for the 169 SBML-mapped reactions
- A residual head that handles the other 8400 non-mapped species rows
- 8 feature groups (role, spatial, proteomics, kinetic_priors,
  medium, regulatory, ribosome_subunit, thermodynamics)

## Current best metrics (M7.7)

| Metric | Value | Notes |
|--------|-------|-------|
| val_singlestep_mse | 0.0589 | one-step prediction MSE on val replicate |
| val_rollout_200_mse | 0.0826 | 200-step trajectory MSE |
| val_mse_count | 0.0716 | **STOCHASTIC NOISE FLOOR — cannot be improved deterministically** |
| AUROC vs Breuer 2019 essentiality | 0.6416 | gene-essentiality screening accuracy |
| **composite** (what we minimise) | **0.0837** | = val_singlestep + 0.3 × val_rollout_200 |

## Your goal

**Modify `initial_program.py` to produce a fine-tuned M7 with composite < 0.0837.**

Each candidate is run for at most 10 minutes wall-clock. The
evaluator parses the JSON metrics block printed between the
`===EVAL_METRICS===` markers.

## What you can change

- **Hyperparameters** at the top (LR, BIAS_STRENGTH, weights, batch
  size, steps).
- **target_category filter logic** — currently selects which row-types
  to focus weakness sampling on. You can add new categories or
  change the prefix lists.
- **The fine-tune pipeline itself** — wrap the existing
  `fine_tune_on_weaknesses` call, replace it entirely, add new loss
  terms, change the weakness-profile criteria, add dropout, change
  the optimizer, etc.
- **Weakness criteria** — by default uses worst-per-timestep MSE; you
  could instead use worst-per-species, or per-pathway, or
  uncertainty-weighted.

## Domain-specific facts (use these to guide mutations)

1. **`val_count = 0.0716` is an irreducible stochastic noise floor.**
   All M7.3, M7.6, M7.7 variants hit this same number despite very
   different feature configurations. Targeting `val_mse_count` is
   futile until the model has a stochastic head (M8). **Focus on
   reducing `val_rollout_200_mse`** — that has room to move.

2. **Long-horizon error compounds.** The model's gradient signal
   beyond ~10 timesteps is weak. Mutations that strengthen the
   relationship between far-horizon targets and current loss tend
   to help more than uniform fine-tuning.

3. **High LR is unstable.** Learning rates above ~5e-4 destabilize the
   encoder; the safe range is 3e-5 to 3e-4.

4. **Biased sampling has a sweet spot.** bias_strength ≈ 0.7 helps;
   above 0.85 the model overfits weak regions and loses uniform
   performance.

5. **F_\* (flux) rows are already very low MSE (~0.0010).** Most error
   is in the count rows. But the flux head's behaviour is constrained
   by PINN; targeting fluxes more aggressively probably hurts.

6. **The simulator-init artifact**: timesteps t < 10 contain huge
   non-physical residuals (the simulator's startup protocol). The
   pipeline already skips these (`T_SKIP_INITIAL=10`).

## Constraints

- Must complete within 15 minutes wall-clock per candidate
- Must use the existing M7.7 starting checkpoint
- Cannot require new external datasets or APIs
- Total fine-tune steps must stay ≤ 5000
- Must print metrics in the exact `===EVAL_METRICS===` / `===END_EVAL_METRICS===` format

## Output format you must produce

```
===EVAL_METRICS===
{"val_singlestep_mse": 0.0xxx, "val_rollout_200_mse": 0.0xxx,
 "val_mse_count": 0.0xxx, "val_mse_flux": 0.0xxx,
 "val_mse_cum": 0.0xxx, "composite": 0.0xxx}
===END_EVAL_METRICS===
```

(The `compute_full_val_metrics` function in
`cell_sim.lgnn.self_improve.iterate` returns this dict ready to print.)
