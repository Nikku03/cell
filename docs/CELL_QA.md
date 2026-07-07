# CellQA — the whole-cell question-answering layer

**The goal made concrete.** The stated goal is: *"map the whole cell, miss nothing, answer anything"* — track
mutations, pathway effects, knockout downstream, phenotype→cause, drug design, drug interactions. CellQA is the
single interface that answers those question types by routing to the validated engines, and — the part that
makes it a *product* — **tags every answer as fact vs prediction, with a confidence and provenance, and
abstains when it genuinely can't.**

`colab/cell_qa.py` (`CellQA`, `demo()`, `coverage()`).

## The design principle it enforces

*Measured data = fact (use it). Prediction = only for unknowns. Benchmarks = the trust certificate.*
- **Measured** answers come back with `tier="measured"`, `confidence=1.0` (known PPI edge, regulatory edge,
  measured kcat, known drug target).
- **Predicted** answers fill the unknowns with the validated predictor's **calibrated confidence** (link score,
  ΔΔG reliability, kcat tier).
- Below a confidence floor, CellQA **abstains** instead of guessing.

So a cell map with holes doesn't fail silently — every gap is either filled with a confidence-tagged prediction
or explicitly abstained.

## Coverage — the six question types (each with its trust certificate)

| question | engine | validated accuracy | answer tier |
|---|---|---|---|
| `what_binds(X)` | CellGraph link (R-GCN/hybrid) | PPI link AUC **0.89** | fact + prediction |
| `knockout(X)` → downstream | CellGraph perturbation | direction acc **0.81** | prediction |
| `mutation_effect(X, mut)` | ΔΔG predictor | S669 r=**0.41** (DDGun-tier) | prediction (low per-call) |
| `drug_interactions(drug)` | CellGraph polypharmacology | drug AUC **0.80** | fact + prediction |
| `regulates(X)` | regulatory network | curated | fact |
| `kcat(enzyme)` | tiered kinetics (CatPred) | **3.3×**, at the noise floor | fact or prediction |

The right column is the scorecard: each answer-type is deployed on the unknowns only because its benchmark
certifies it works. AUC/fold-error are the *trust layer*, not the product.

## Live examples (real output)

```
Q: what does TP53 bind?
   measured : ANXA2(1.0), APEX1(1.0), ARID1A(1.0) …        <- fact, from PPI database
   predicted: CTNNB1(0.96), SMAD4(0.95), AR(0.95) …        <- the TP53 tumor-suppressor network, confidence-ranked
Q: remove SREBF2 -> downstream?
   predicted: HMGCR(down,1.0), LDLR(down,1.0), ABCG5(up,1.0) …   <- textbook cholesterol regulation
Q: SOD1 A4V mutation effect?
   ddg_kcal_mol: -1.25  confidence: 0.156                  <- honestly LOW confidence (DDGun-tier, per-call noisy)
Q: unknown gene?
   ABSTAIN — NOTAGENE not in model
```

Note the SOD1 A4V answer: the model returns a per-call ΔΔG but flags **confidence 0.156** — it doesn't
overclaim on a prediction it knows is noisy. That honest self-rating is the point.

## Honest scope

- CellQA is an **integration layer over already-validated engines** — it adds no new accuracy, it adds the
  fact/prediction/confidence/provenance **contract** and a single entry point.
- The weakest answer-type is `mutation_effect` (DDGun-tier ΔΔG) — flagged with low confidence and a caveat;
  strengthening it (ESM-2/ThermoMPNN) is the top upgrade (see `docs/FUTURE_IDEAS.md`).
- Still to add: `disease_cause` / `disease_target` (the multilayer pipeline exists in
  `disease_target_pipeline.py`; wiring it behind CellQA is the next step), and cell-type conditioning via
  `emask`.

## Why this is the goal, not a benchmark

The product isn't an AUC — it's a system that **answers any question about the cell and tells you how much to
trust each answer.** CellQA is that surface; the scorecard is what lets it say "fact," "predicted (0.89)," or
"I don't know" honestly. Filling the unknowns with confidence — that's "map the whole cell, miss nothing."
