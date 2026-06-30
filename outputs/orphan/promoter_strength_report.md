# Promoter strength IS solvable from sequence (Urtecho MPRA)

Earlier we got 0.13 predicting in-vivo expression from a crude promoter score and
concluded promoter strength wasn't sequence-computable. That was the wrong target
(confounded in-vivo) + a weak model. On the clean Urtecho 2019 MPRA (10,600 sigma70
variants, measured strength, fixed condition):

| model | held-out R^2 |
|---|---|
| element log-linear (-35/-10/spacer/UP one-hot) | 0.592 |
| raw 150bp sequence one-hot (ridge) | 0.569 |
| sequence -> MLP | **0.973** |

INTRINSIC promoter strength (sequence -> RNAP recruitment rate, fixed condition)
is computable. The model is exactly "consensus elements + per-base effects".

## Caveats (so we don't overclaim)
1. Intrinsic, fixed-condition. In-vivo beta = intrinsic x GLOBAL multiplier
   (supercoiling/ppGpp/sigma) -- condition-level, not per-gene -> tractable.
2. R^2 0.97 is within this library scaffold; arbitrary genomic promoters ~0.6-0.75
   (biophysical/Promoter Calculator is the genome-general tool).
3. E. coli sigma70; other orgs via the conserved biophysical model + calibration.

## Impact on the cell model (closes most of the [TF] gap)
- gamma = growth dilution (universal)
- intrinsic beta = computable from promoter sequence (R^2 ~0.6-0.97)  <-- NEW
- feedback handles autoregulated ~57%
=> [TF] = beta/gamma x (one global condition multiplier): computable up to a single
condition-level factor, instead of per-gene measurement. The TF-concentration
problem shrinks from "measure ~200" to "compute beta+gamma+feedback, one residual knob".

Data: Urtecho 2019, github.com/KosuriLab/ecoli_minimal_promoter (MPRA txt gitignored).
Files: colab/promoter_strength_solve.py, outputs/orphan/promoter_strength_solve.json.
