# Gene/protein essentiality transfers cross-organism (ESM adds over conservation)

Reframe (correct): essentiality is a gene/protein property. The chemical axis was
the instrument, not the target. Proteins are homologous -> the protein-level
signal should transfer across organisms (unlike the chemical-conditional layer,
which collapsed in Stage 5).

## Cross-org test: train on 8 pilots, predict mtub essential GENES, INDEPENDENT
## DeJesus 2017 truth (no feba labels). Same 1038-gene eval set.

| signal | AUC |
|---|---|
| conservation (panel labels via feba orthologs) | 0.725 |
| ESM-2 protein model (trained on pilot feba-essential) | 0.733 |
| ESM + conservation | 0.768 |

(ESM over ALL mtub genes incl. orphans = 0.667; on the conserved core = 0.733.)

## Findings
1. The protein-level essentiality signal TRANSFERS cross-organism (ESM 0.73,
   conservation 0.73) -- both well above chance, on independent truth.
2. ESM and conservation are ORTHOGONAL: combined 0.768 (+0.04 over either).
   ESM (what the protein DOES, from sequence) adds signal beyond conservation
   (how often the family is essential).
3. This is the FIRST clean case in the project where a signal BEATS the
   conservation baseline instead of collapsing onto it -- because at the
   gene/protein level, sequence-function is genuinely orthogonal to
   conservation-rate.

## Contrast (the two layers, two verdicts)
| question | transfers to new organism? | best AUC |
|---|---|---|
| gene/protein essentiality | YES | ESM+conservation 0.77 |
| conditional/chemical essentiality | NO (Stage 5 collapse) | ~chance |

## Takeaway
For predicting a NEW organism's essential GENES, combine ESM (protein function)
+ conservation -> 0.77, the best cross-org essentiality achieved here. The
conditional/chemical layer remains measurement-bound. The user's reframe was
right: center on genes/proteins, and the signal travels.
