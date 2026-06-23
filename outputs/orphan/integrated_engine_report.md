# Two-wheel integrated engine — E. coli report

**Setup.** Model trained leave-`escherichia`-out (never saw any Keio label).
Both wheels run on the 3566 Keio genes; results validated against truth.

## How the genome is partitioned

| verdict | count | what it means |
|---|---|---|
| CONFIDENT_ESSENTIAL | 333 | Wheel 1 calls at P>=0.90 |
| CELL_SUPPORTED | 313 | Wheel 1 abstained; Wheel 2 lifted by role/context |
| EXPERIMENTAL_TARGET | 559 | both wheels uncertain — lab queue |
| CONFIDENT_NON_ESSENTIAL | 2361 | Wheel 1 calls at P>=0.90 |

## What the cell adds to Wheel 1

| | Wheel 1 only | Wheel 1 + Wheel 2 (combined) |
|---|---|---|
| genes called essential | 333 | 646 |
| true essentials recovered | 300 | 414 |
| precision | 0.901 | 0.641 |
| recall | 0.51 | 0.704 |
| F1 | 0.651 | 0.671 |

## Cell-level audit: gaps

Subsystems below the viable minimum:

- **synthetase**: 4/12 (need 8 more)

## Top-50 experimental queue hit rate

**30%** of the top-50 ranked soup genes are real essentials.
(soup baseline is ~15%; random would be that.)
