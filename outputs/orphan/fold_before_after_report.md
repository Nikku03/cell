# Before / after: wild-type vs mutant structure (literal ESMFold)

Folded the wild-type and mutant sequences with ESMFold, Kabsch-superimposed the Cα atoms,
and measured the real structural difference. Two iconic small disease proteins — one with a
SURFACE mutation, one with a BURIED mutation. Code: `colab/fold_before_after.py`.

## Result

| protein | mutation | location | global Cα-RMSD | local RMSD (±6) | ΔpLDDT@site |
|---|---|---|---|---|---|
| HBB | E6V (sickle-cell) | surface | **0.012 Å** | 0.022 Å | 0 |
| SOD1 | A4V (ALS) | buried core | **0.263 Å** | 0.382 Å | 0 |

## Reading it — this quantifies the whole structure story

- **Surface disease mutation (sickle) → the fold is literally identical** (0.012 Å = numerical
  zero). Sickle-cell does **not** misfold hemoglobin. The disease comes from swapping a surface
  charge for a greasy residue → a sticky patch → polymerization. *Same structure, harmful new
  surface property.* The before/after picture confirms exactly what the earlier analysis argued.
- **Buried disease mutation (ALS) → a real local perturbation** (0.26 Å global, 0.38 Å local) —
  **~20× larger** than the surface case. Jamming a bigger residue into the packed core repacks
  the neighbourhood. The global fold is still the same protein, but the core is measurably
  disturbed.

**So a point mutation almost never redraws the fold (both are sub-Ångström), but *where* it
sits sets the magnitude: a buried change perturbs local structure ~17× more than a surface
change.** This is the direct structural confirmation of why **burial was the single strongest
pathogenicity feature (0.84 AUC)** in the classifier.

## Honest caveats
- **ESMFold is a static single-structure predictor.** It captures the *folded snapshot*, not
  stability, dynamics, or aggregation propensity. So these RMSDs are **lower bounds** on the
  real impact: SOD1 A4V destabilizes and misfolds far more in the cell than a 0.26 Å static
  snapshot conveys, and sickle's aggregation is invisible to a single-chain fold. The *ranking*
  (buried ≫ surface) is the robust, meaningful signal.
- pLDDT (model confidence) barely moved — the model is equally sure of both folds; the
  difference is geometric, not confidence.
- Two-protein illustration, not a benchmark.

## Where it fits
This closes the structure/mutation arc: population variant → same fold (surface); disease
surface variant → same fold + new property (sickle); disease buried variant → local core
perturbation (ALS); disease active-site variant → function broken directly. *Where* on the
structure a mutation lands — not whether it changes the fold — is what determines its effect.
