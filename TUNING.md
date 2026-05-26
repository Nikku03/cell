# Tuning knobs for `colab_cell_emulator.py`

The file has ~30 config flags at the top.  This doc covers the ones that
actually matter for matching the upstream's biological validation targets
(Thornburg 2026), grouped by what they affect.  If you're picking up the
project, this is what to tweak when a specific metric is off.

## Capacity flags — what's running

| Flag | Default | What it does | When to flip |
|------|---|---|---|
| `USE_METABOLISM_CORE` | True | bi-bi rate law on ~115 SBML reactions with measured k_cat/K_m | leave on |
| `USE_VOLUME_CORE` | True | dynamic cell volume from lipid sum | leave on |
| `USE_CENTRAL_DOGMA` | True | per-gene tx/tl/decay with calibrated rates | turn off if rollout R² collapses; the LGNN takes over |
| `USE_ASSEMBLY_CORE` | False | complex-formation mass-action | leave off — only 2/24 wired on real data; needs calibration treatment like CD got |
| `USE_KO_AUGMENTATION` | True | random species-zero during training | leave on if you want KO-essentiality predictions to mean something |
| `USE_ATP_LEDGER` | True | soft penalty on ATP rate < maintenance floor | rarely matters; calibration auto-tunes the floor |
| `USE_SIGMA_ANCHOR` | True | anchor predicted log σ to empirical std | leave on — without it the stochastic head collapses |
| `USE_PINN_HEAD` | True | mass-balance for SBML species (auto-disabled when MetabolismCore is on) | informational |
| `USE_STOCHASTIC_HEAD` | True | per-species log σ + NLL loss | leave on |
| `USE_TORCH_COMPILE` | True | torch.compile for ~1.5× speedup | turn off if it triggers a recompile loop |
| `USE_TEMPORAL_CONTEXT` | True | v15.0: small transformer attends over past 8 states, adds context vector to LGNN hidden | turn off to revert to v14.9 pure-LGNN behaviour |

## Loss-term weights — λ knobs

| Knob | Default | What it pushes | If too high | If too low |
|------|---|---|---|---|
| `LAMBDA_1STEP` | 1.0 | NLL weight on the first rollout step | over-emphasises 1-step accuracy, hurts long rollouts | over-emphasises long rollouts at expense of next-step |
| `LAMBDA_HYP` | 0.01 | Tier-2 hypothesis aux loss (mono + pair) | model gets pulled toward maybe-wrong hypotheses | hypotheses ignored, missing signal |
| `LAMBDA_ATP` | 0.01 | ATP-deficit penalty | model over-produces ATP to avoid penalty | ATP balance unconstrained, drifts |
| `LAMBDA_SIGMA_ANCHOR` | 0.2 (v14.7, was 0.05) | log σ → empirical std | NLL gets fought, σ matches data even when wrong | mode collapse (model produces zero variance) |
| `LAMBDA_TRAJ_VAR` | 0.05 (v14.7) | pred.std(batch) → target σ | training unstable, predictions diverge | mode collapse persists |

## Rate scaling — calibration knobs

| Knob | Default | What it scales | Symptom of "too high" | Symptom of "too low" |
|------|---|---|---|---|
| `CD_TRANSCRIPTION_SCALE` | 1.0 | every k_tx_per_gene | mRNA over-produces | mRNA stays at initial, never doubles |
| `CD_TRANSLATION_SCALE` | 0.7 (v14.6, was 1.0) | every k_tl_per_gene | protein fold-change > upstream (~2×) | protein fold-change < upstream |
| `SAMPLE_NOISE_SCALE` | 0.5 (v14.7) | training-time noise injection scale | rollout R² tanks, training unstable | mode collapse, predicted std too low |
| `ATP_MAINTENANCE_RATE` | 4e5 → calibrated at runtime from data | ATP-deficit floor | model over-produces ATP | rarely fires (data-calibrated value usually lower) |
| `GIBBS_DG_THRESHOLD_KJ` | 10.0 | ΔG° magnitude below which reaction is left reversible | too many reactions clamped (forces wrong direction) | too few clamped (no useful constraint) |

## Physical constants — change carefully

| Constant | Default | Where it came from |
|------|---|---|
| `SYN3A_VOLUME_L` | 3.35e-17 | Thornburg 2026: sphere of 200 nm radius |
| `NA_AVOGADRO` | 6.02214076e23 | physics |
| `T_HALF_MRNA_S` | 120.0 | Thornburg 2026 (median ~2 min) |
| `T_HALF_PROTEIN_S` | 36000.0 | literature estimate (~10 h); paper notes this is an unmeasured assumption |

## Training-curriculum knobs

| Knob | Default | What it does | When to change |
|------|---|---|---|
| `TIME_STRIDE` | 30 | seconds per model step | drop to 10 for finer dynamics (but 3× slower training); raise to 60 to match earlier v9 results |
| `K_MAX` | 120 | training rollout max steps | matches half cell-cycle at stride=30; increase for terminal-state losses (future work) |
| `STEPS` | 3000 | training optimization steps (v15.0 bumped 1500→3000 for hybrid arch) | fewer for quick experiments, more for convergence |
| `BATCH` | 32 | training batch size | bump on bigger GPUs |
| `LGNN_HIDDEN` | 64 | hidden dim (v14.9 bumped 32→64; v15.0 keeps 64) | 128 doubles param count again |
| `T_CTX_WINDOW` | 8 | v15.0: past states attended over by transformer | larger → more global trajectory context, but more compute/memory |
| `T_CTX_HIDDEN` | 32 | v15.0: transformer per-token embed dim | larger → more transformer capacity, ~5k extra params per +8 dim |
| `T_CTX_LAYERS` | 2 | v15.0: transformer encoder depth | 1 → cheaper but less expressive; 3+ → diminishing returns vs cost |
| `KO_AUG_PROB` | 0.3 | fraction of batch elements with knockout perturbation | raise to 0.5 for more KO emphasis |
| `K_M_TOTAL_MRNA` | 100 | (legacy — superseded by v14.3 linear ribo scaling) | don't change |
| `K_PER_RIBO` | 1.5e-3 | (legacy — superseded by v14.3 per-gene calibration) | don't change |

## Diagnostic — what each line in the train log means

```
[metabcore] wired N/356 reactions; ...                ← MetabolismCore coverage
[metabcore] ΔG° clamp: M reactions forced irreversible ← how many reactions had |ΔG°| > 10 kJ/mol
[metabcore] ATP ledger: P producing + C consuming     ← stoichiometric ATP roles
[volumecore] tracking L lipid species, t=0 mean ...   ← VolumeCore initialization
[cdcore] N genes wired (G+R+P all present); ...       ← CentralDogmaCore coverage
[cdcore] per-gene calibration: A direct, B proxy, C floor ← where rate constants came from
[cdcore]   scales: k_tx × X.X, k_tl × X.X             ← global scaling knobs
[sigma_anchor] empirical log σ: median ...            ← what σ-anchor is pulling toward
[atp_calibrate] data-derived ATP rate: ...            ← per-run ATP floor calibration

step N  K=K  1-step X.XXX  rollout X.XXX  hyp[...]  atp=...  log σ=...
   ↑ training step
       ↑ rollout horizon used at this step
            ↑ MSE/NLL on the first predicted step (1-step)
                          ↑ MSE/NLL averaged over the K-step rollout
                                          ↑ hypothesis aux loss EMA per kind
                                                       ↑ ATP rate EMA (target: above floor)
                                                                ↑ log σ EMA (target: empirical log σ)
```

## Eval block

```
persistence 1-step R²       ← baseline: "predict next = current"
model 1-step R² (test)      ← our model on held-out trajs
model full rollout R²       ← roll forward 240 steps, mean R² across species
median R² on top-200 vars   ← the HONEST metric — variance-weighted; baseline R² is misleadingly high on flat species
σ calibration               ← log10(predicted std / empirical std); 0.0 = perfectly calibrated
paper validation block      ← doubling time, ori:ter, ribosome fold, protein fold — model vs upstream vs paper
[1] worst-predicted species ← diagnostic: which species the model gets least right
[2] element balance drift   ← C, H, N, O, P, etc. — should drop as MetabolismCore covers more
[3] SBML coverage           ← reaches the limit of what's in the kinetic data file
KNOCKOUT SWEEP              ← 490 genes tested, MCC vs Breuer 2019, top-12 predictions
```

## Common iteration loop

After a Colab run:

1. **Did rollout R² move?** Check `model full rollout R²` against the previous run's number.
2. **Are the metrics in the paper validation block close to the upstream column?** That's the actual target.
3. **Is σ calibration near 0?** If not, mode collapse — bump `LAMBDA_SIGMA_ANCHOR` or `LAMBDA_TRAJ_VAR`.
4. **Is KO MCC random or above 0.1?** Above 0.1 means physics constraints are propagating to essentiality.
5. **Top-12 KO precision** — count the ✓ marks; ≥ 8/12 is good.

Tweak ONE knob per iteration so attribution is clean.
