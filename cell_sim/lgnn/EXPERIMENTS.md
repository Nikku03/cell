# LGNN experiments log

One row per training run. Numbers are honest: bad results get logged
just like good ones. If a config is missing a number, the run didn't
finish or didn't measure that metric — say so explicitly rather than
leaving the cell blank.

Metric definitions
------------------

- **val_mse**: per-element MSE on Δsigned-log1p of the held-out
  replicate(s). Averaged over all 8572 rows × 7200 timesteps.
- **val_r2**: per-species R² on the same hold-out, averaged over all
  rows that had nonzero target variance. Negative values mean the
  model is worse than the per-row mean predictor.
- **median_r2_top100**: per-species R² over the 100 rows with the
  highest *state* variance on the val replicate. This filters out the
  thousands of constant-near-zero rows that drag the mean R² up
  artificially.
- **rollout_drift_at_7200s**: max |signed-log1p(predicted) -
  signed-log1p(truth)| over the 8572 rows after rolling the model
  forward 7200 steps from x_0. <1 is excellent; >100 is divergent.
- **knockout_mcc_breuer**: Matthews correlation between the model's
  knockout-impact ranking and the Breuer 2019 essentiality calls.
  Computed only after week 5.

Runs
----

| id  | date       | model        | cfg                                                                     | train_mse | val_mse | val_r2 | median_r2_top100 | rollout_drift | notes |
|-----|------------|--------------|-------------------------------------------------------------------------|-----------|---------|--------|------------------|---------------|-------|
| M0  | 2026-05-07 | MLP_baseline | hidden=1024, n_blocks=2, lr=3e-4, bs=256, epochs=2, buffered_shuffle=F  | 0.0308    | 0.0303  | -0.62  | **-0.502**       | ~10³          | floor. checkpoint: count_dynamics_v0.pt. |
| M0a | 2026-05-07 | MLP_baseline | as M0 but buffered_shuffle=T, buffer_size=3                             | 0.0307    | 0.0312  | -0.64  | -0.785           | ~10³          | regression. 3 replicates is not enough buffer to be IID-ish — gradient sees ~3 correlated runs at a time, slightly worse than the natural per-replicate-block traversal. Default flipped back to False. Worth retrying with buffer_size=10+ if A100 RAM allows. |

Planned slots (delete once filled, don't pre-fill numbers)
----------------------------------------------------------

| id | model                  | what's new vs prior | notes |
|----|------------------------|---------------------|-------|
| M1 | graph_dynamics         | SBML graph + hetero-edge MLPs + heteroscedastic NLL head + zero-init residual delta + multi-step (k=10) loss | week 2. expectation: median R² >0, rollout drift <100. |
| M2 | liquid_gnn             | replace GRU-style update with CfC cell. same edges, same loss. | week 3. expectation: rollout drift <10. |
| M3 | sparse_liquid_gnn      | + K=128 pattern bottleneck (Sinkhorn + top_k=5 + dropout) | week 4. expectation: similar metrics to M2; primary value is interpretability. |

Microscopes (week 5)
--------------------

- pattern_atlas.csv  — 128 patterns × top-genes/reactions, hand labels
- knockout_sweep.csv — per-species Δ-trajectory norm vs Breuer ranking
- anomaly_map.csv    — top-20 worst-prediction windows on held-out trajectories

Things to call out if they happen
---------------------------------

* If M1 ≥ M0: graph isn't pulling its weight. Check edge-degree
  distribution and edge-type ablation before adding the LNN.
* If M2 learned-τ values are biologically nonsensical (e.g. mRNA τ in
  hours): the LNN isn't learning what we hoped. Investigate before
  layering on M3.
* If knockout_mcc_breuer < 0.2: the surrogate didn't capture
  essentiality structure. Document and stop adding architecture; pivot
  to anomaly-map-driven analysis.
