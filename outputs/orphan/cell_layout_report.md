# Yes: E. coli K-12 MG1655, laid out

The cell we can lay out most completely (every data layer exists). Integrated
blueprint across all layers we built; per-gene table in cell_layout.csv.

## The cell, laid out (4,689 protein-coding genes)
| layer | count | how it was laid out |
|---|---|---|
| all genes (parts) | 4689 | genome -> genes -> products -> families |
| essential core | 412 | Keio truth; predictable by ESM+conservation (0.768 cross-org) |
| metabolic (iJO1366) | 1217 (168 essential) | auto-FBA + conditional essentiality across media |
| transcription factors | 193 | DBD families |
|   - global TFs | 8 | activity computable (effectors); targets need data/anchor |
|   - specific TFs | 49 | operators + targets computable from sequence |
| autoregulated TFs | 120 (93 neg) | [TF] concentration computable via feedback setpoint |

## Worked examples through the layers
- rpoB/rpoC (RNAP) -> essential core, universal machinery
- crp -> global TF, regulon 531, neg-autoreg ([TF] computable; targets from data)
- trpR/gntR -> specific TF, regulon 12, sharp operator (targets from sequence)
- sdhA, lacZ -> metabolic (conditional-essential on succinate / lactose)
- fnr -> essential global TF (O2 sensor); dnaA -> essential specific TF (replication)

## What "laid out" means here
Each gene placed into the blueprint with its layer(s), essentiality, metabolic
role, regulatory role + class, and [TF]-computability tag. Every element labeled
computed / transferred / measured.

## Other cells we can lay out
- E. coli / B. subtilis: full layout (all layers, this).
- ~60 feba organisms: parts + essential core (ESM+conservation) + auto-metabolism
  + co-fitness regulatory edges -> mid-confidence layout.
- ANY bacterial genome: parts + essential core + auto-FBA metabolism + operons +
  specific-TF operators + conserved-global-regulator activity -> universal
  blueprint, lower confidence (no measured regulatory edges).

Files: colab/cell_layout.py, outputs/orphan/cell_layout.{csv,json,png}.
