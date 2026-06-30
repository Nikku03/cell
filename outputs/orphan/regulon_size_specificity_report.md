# Regulon size vs operator specificity: why multi-target TFs can't be found from sequence

A TF that "attaches to more than one gene" is the norm (a regulon). The decisive
finding: how MANY genes a TF binds is near-deterministically tied to how
findable its operator is.

## Measured (E. coli, 63 TFs)
| relationship | correlation |
|---|---|
| log(#targets) vs operator info content | **-0.94** |
| log(#targets) vs #genome matches | +0.88 |
| log(#targets) vs precision-from-sequence | -0.54 |

| group | mean #targets | operator info | genome sites | precision |
|---|---|---|---|---|
| few targets (<=20) | 14 | 13.5 bits | 81 | 0.29 |
| many targets (>=80) | 199 | 9.2 bits | 4006 | 0.05 |

## Why - information theory forces it
A motif of I bits occurs ~once per 2^I bp by chance:
  - 14-bit operator -> ~1 / 16,000 bp -> ~280 genomic sites
  - 7-bit operator  -> ~1 / 128 bp    -> ~36,000 sites
To bind 500 genes a TF's operator MUST recur near 500 promoters, so it MUST be
degenerate. A sharp operator simply cannot occur in hundreds of places. The loose
motif IS the mechanism that lets a global regulator reach many targets.

## Consequence
- Multi-target / global regulators (CRP, FNR, IHF, H-NS, Fis) -> degenerate
  operators by necessity -> unpredictable from sequence (and they control the
  most genes). Must use the functional edge (co-expression + co-fitness).
- Few-target / local regulators (gntR, trpR, torR) -> sharp operators -> findable
  from sequence.
You cannot have both a specific operator and a large regulon - they are two
readouts of the same quantity (motif information content).
