# cell_v5: measured knockdowns in four cell types

Written by `rem/atlas/cellbank5.py build`. Read with `cellbank4.Bank(path)`; the format is the same as `cell_v4`.

## What changed from cell_v4

`cell_v4` is kept unchanged. This version holds screens A, B and C in the same order, plus two new screens.

| Screen | Cell type | Records | Source |
|---|---|---|---|
| A | K562, genome-wide (day 8) | 5,052 | Replogle et al. 2022 |
| B | K562, essential genes (day 6) | 1,550 | Replogle et al. 2022 |
| C | RPE1, essential genes | 1,953 | Replogle et al. 2022 |
| **D** | **HepG2, liver, essential genes** | **1,631** | Nadig et al. 2025, *Nature Genetics*; figshare 29498366 |
| **E** | **Jurkat, T cell, essential genes** | **1,563** | Nadig et al. 2025, *Nature Genetics*; figshare 29498366 |

That is 11,749 records in 3.7 MB. All sources are CC BY 4.0.

## How the new screens are processed

Screens D and E start from Nadig's log2 fold-change tables.

1. **Phenotype filter.** A knockdown is stored only if at least 20 measured genes reach p < 0.001, not counting its own gene. About 8 would pass by chance. This filter stands in for the energy-test filter used for the Replogle screens.
2. **Same processing as cell_v4.** Remove the screen's own mean response over TRAIN genes. Map onto screen A's genes by Ensembl id. Project onto the shared 128 components. Store as int8 codes plus one scale.
3. **Field differences.**
   - `energy_p` is empty for D and E.
   - `records_meta.json["n_sig"]` holds each stored knockdown's count of genes at p < 0.001.

## Data check (H2)

On TRAIN knockdowns measured in both screens, each new screen agrees with K562-essential:

| Screen | Knockdowns compared | Median r | Shuffled pairs |
|---|---|---|---|
| HepG2 | 654 | 0.46 | −0.01 |
| Jurkat | 609 | 0.57 | −0.01 |

## Rules for users

These are the same as cell_v4. Any evaluation must hide:
- all screen-A TEST records;
- for each query gene, that gene's own records in every screen.
