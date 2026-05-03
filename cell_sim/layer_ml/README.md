# cell_sim.layer_ml — LNN for cross-organism essentiality

A liquid-neural-net stack for predicting gene essentiality from
multimodal genomic features, with operon-graph message passing,
hyperbolic memory, and Elastic Weight Consolidation for continual
learning across organisms.

Architecture borrowed from
[Nikku03/enzyme_Software/liquid_nn_v2](https://github.com/Nikku03/enzyme_Software)
(chemistry LNN with `ContextAwareTauPredictor` + `EdgeAwareMessagePassing`)
and tuned for genes:

* tau bounds for gene-network timescales
* edge features represent operon membership (gap_bp, strand match)
* hyperbolic memory bank (Poincaré ball) accumulates representations
  across continual-learning steps so the model can "remember pathway X
  from organism A" when scoring organism B
* EWC penalty preserves Phase 1 weights during Phase 2 fine-tuning

## Module layout

| file | what it has |
|---|---|
| `liquid_lnn.py` | `ContextAwareTau` + `EdgeAwareMessagePassing` + `LiquidStep` (one ODE-style integration step) |
| `hyperbolic_memory.py` | `HyperbolicMemoryBank` — K-slot Poincaré-ball memory with circular FIFO writes + top-k hyperbolic-distance retrieval |
| `multimodal_encoder.py` | `MultimodalEncoder` with 4 branches (scalar / regulatory / kinetic / sequence-kmer), per-modality presence flags, missing-data masking |
| `continual_learning.py` | `EWCRegularizer` — Fisher-info-weighted quadratic penalty on weights moving away from previous-task values |
| `essentiality_lnn.py` | `EssentialityLNN` — top-level model composing all above plus essentiality + regulator-proxy heads |
| `data_loader.py` | `load_organism_batches()` — pulls v25 features + v27 regulatory + v29 conservation + GenBank sequence + operon edges into per-organism multimodal batches |
| `checkpoints/` | training-output `.pt` files (gitignored) |

## Self-tests

Every module has a `__main__` self-test that runs the component on dummy
inputs:

```
python -m cell_sim.layer_ml.liquid_lnn
python -m cell_sim.layer_ml.hyperbolic_memory
python -m cell_sim.layer_ml.multimodal_encoder
python -m cell_sim.layer_ml.continual_learning
python -m cell_sim.layer_ml.essentiality_lnn
python -m cell_sim.layer_ml.data_loader
```

Each self-test:
* asserts shape / dtype correctness
* exercises edge cases (empty inputs, missing modalities, isolated nodes)
* runs a forward + backward pass to verify gradient flow
* prints "ALL PASS" on success

## Training

```
python scripts/train_essentiality_lnn.py
```

Three phases:

1. **Phase 1** — joint training on Salmonella + Syn3A (the rich-annotation
   organisms). Joint multitask objective: class-weighted essentiality BCE
   + auxiliary regulator-proxy BCE.
2. **Phase 2** — continual learning across the other 5 essentiality
   organisms (Caulobacter, S. aureus, M. tb, A. baylyi, M. pneumoniae).
   For each, EWC penalty preserves Phase 1 weights while the model adapts.
   Hyperbolic memory accumulates representations.
3. **Phase 3** — per-organism evaluation. Reports MCC + confusion matrix
   for each organism and "forgetting" (Phase 1 vs Phase 3 MCC delta on
   the original training organisms).

## Inputs

Per gene (assembled by `data_loader.load_organism_batches`):

| modality | dim | source |
|---|---|---|
| scalar | 20 | v25 features (length, GC, position, kw_*, etc.) |
| regulatory | 50 | v27 PWM scores (RBS / -10 / -35 / 14 TF binding sites). Mask=1.0 for Salmonella (RegulonDB-curated), 0.5 for others (PWM-applied with caveat). |
| kinetic | 4 | placeholder for SBML kinetic params (Km, kcat, Vmax, has_kinetics). All-zeros currently; populated when SBML→gene mapping is wired in. |
| sequence k-mer | 320 | 64 codon 3-mers + 256 4-mers, normalized counts |
| operon edges | (2, E) | same-strand consecutive genes within 100bp. `edge_attr` = log(gap_bp+1) + strand_match flag. |

## Continual-learning rationale

Naive transfer would: train on Salmonella+Syn3A, fine-tune on Caulobacter,
fine-tune on M.tb, fine-tune on A.baylyi → catastrophic forgetting.
The model that handles M.tb at the end has lost the Salmonella-specific
pathway knowledge encoded during Phase 1.

EWC fixes this by computing the Fisher Information Matrix on the
Salmonella+Syn3A loss after Phase 1, snapshotting the model weights, and
adding a quadratic penalty during Phase 2 that pulls weights back toward
their Phase 1 values, weighted by their importance to the original task.

The hyperbolic memory bank is complementary: it stores hidden states
across all phases and lets the model retrieve "this organism's gene X
looks like a gene from Phase 1's pathway Y" when scoring new organisms.

## Honest expectations

The cross-organism essentiality ceiling on this dataset is **MCC ≈ 0.24**
(established by v25-v29 facts). The LNN architecture won't break that
ceiling — the bottleneck is feature representation, not model capacity.

What this LNN DOES bring vs gradient boosting:

* multi-modal gating (the model explicitly knows when a feature is
  unavailable for a given organism, vs zero-imputation)
* operon-graph context (genes inherit signal from their operon neighbors)
* continual learning protocol (handles arrival of new organisms without
  retraining from scratch)
* regulator-proxy localization (auxiliary head predicts where regulators
  should be in unseen organisms — usable for biological discovery)

These are architectural wins the user explicitly asked for. The MCC may
or may not exceed v29's 0.244; the architecture is the deliverable.

## Self-test output (representative)

```
liquid_lnn self-test:
  ContextAwareTau output: shape=(20,), min=0.923, max=1.213
  EdgeAwareMessagePassing output: shape=(20, 32), mean_norm=0.763
  LiquidStep output: shape=(20, 32), delta_norm=1.515
  empty-edges edge case: passes (zero message)
  with-memory step: changes output as expected
  ALL PASS

hyperbolic_memory self-test:
  empty bank retrieve: returns zeros (correct)
  stored 20 embeddings; n_populated = 20
  retrieved norms (first 5): [0.69, 0.69, 0.56, 0.81, 0.81]
  far-query retrieve: norms = [0.43, 0.45, 0.43, 0.45, 0.36]
  after 6 stores of 20: n_populated = 64 (cap 64)
  d(x,x) = [0.0045, 0.0045, 0.0045] (numerical floor ~ sqrt(2*eps))
  d(x,y) = [1.21, 1.52, 1.47] (>0)
  reset: bank cleared
  ALL PASS

essentiality_lnn self-test:
  total params: 108,199  hidden=64  liquid_steps=3  memory_size=128
  forward: essentiality shape = (20,), regulator_proxy shape = (20, 3)
  backward: ok (loss = 0.7008)
  memory n_populated after one train forward: 20
  forward on org B (N=15): essentiality shape (15,)
  reset_memory: ok
  no-edges (isolated nodes): forward ok
  degraded input (only scalar+kmer): forward ok
  ALL PASS
```
