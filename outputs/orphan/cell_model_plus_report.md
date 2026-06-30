# 'Better than a simulator': three layers fused into one perturbation engine

One perturbation propagates through ALL layers:
  ESSENTIAL CORE (informational machinery) + FBA METABOLIC + condition-gated
  REGULATORY->METABOLIC cascade (KO an ACTIVE TF -> sole-activated targets forced
  off -> blocked in FBA).

## Validated improvement over the FBA-only simulator (full genome, Keio truth)
| predictor | precision | recall | accuracy |
|---|---|---|---|
| FBA only | 0.453 | 0.163 | 0.821 |
| FBA + essential core | 0.543 | 0.264 | 0.834 |
Essential-core overlay catches non-metabolic essentials (ribosome/RNAP/replication
/tRNA-synthetase) -> both precision AND recall up.

## Condition-dependent perturbations (correct)
- KO rpsL / dnaA -> LETHAL via essential core (FBA-only missed)
- KO crp on glucose -> VIABLE (CRP inactive with glucose; real crp mutants grow)
- KO crp on arabinose -> LETHAL (CRP active -> araBAD forced off -> can't use arabinose)
- KO sdhA on succinate -> LETHAL (conditional)

## The two upgrades, both real
1. essential-core overlay -> recall 0.16->0.26, precision 0.45->0.54 (validated)
2. condition-gated regulatory->metabolic coupling -> TF KO propagates to a metabolic
   outcome, correctly gated by condition (CRP matters on arabinose, not glucose)

## Honest remaining gaps
- recall still 0.264: keyword essential-core is conservative; plugging the full W1
  (ESM+conservation, 0.768) as the essentiality predictor catches more (envelope/
  membrane/other essentials).
- regulatory cascade is condition-gated only for regulators with effector logic
  (CRP/FNR); others don't propagate (conservative). Broader coverage needs the
  activity model for more regulators (the multiplier work supplies the conserved set).
- crp-on-arabinose forces off ~225 genes (over-broad) though the lethal verdict is correct.

Files: colab/cell_model_plus.py, outputs/orphan/cell_model_plus.json.
