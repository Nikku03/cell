# Does TF protein family predict the operator? — Yes, but family-specifically.

Decisive transfer test on E. coli (RegulonDB, 44 TFs, Pfam families via feba):
learn each TF's operator PWM, then score every TF's targets with every *other*
TF's motif (out-of-sample), split by same-family vs different-family.

## Headline

| transfer | mean AUC |
|---|---|
| within-family (family-mate's motif → TF targets) | **0.544** |
| between-family (random other-family motif) | 0.520 |
| self (TF's own motif, held-out) | 0.541 |

- **within-family ≈ self (0.544 vs 0.541):** a family-mate's motif recovers a TF's
  targets as well as its *own* motif — family carries operator information.
- **within > between (0.544 vs 0.520):** the relation is real, but the *magnitude*
  is small, and the absolute level (~0.54) is the same Wunderlich-Mirny wall —
  family lets you *borrow* a motif, it does not make the motif more informative.

## The relation is family-specific (this is the real biology)

| family | within-family AUC | reading |
|---|---|---|
| **AraC** (MarA/SoxS/Rob) | **0.595** | paralogs share the marbox/soxbox → family ≈ operator |
| bEBP (σ54 activators) | 0.575 | conserved enhancer architecture |
| OmpR / LacI / NarL | 0.53–0.54 | modest |
| **LysR** (OxyR/CysB) | **0.502 (chance)** | share the fold + T-N₁₁-A *architecture*, not the sequence |

This is exactly the principle, now measured: **family conserves the operator
GEOMETRY; the specific sequence is conserved only when paralogs also kept the
same recognition-helix residues** (AraC/XylS stress regulators did; LysR
paralogs diverged on purpose to regulate different genes).

## What this means for "solving" TF → genome

- **Where it works (AraC/XylS-type, σ54-bEBP):** the family motif *is* the
  operator. For these regulators you can predict binding from family alone, with
  no per-TF data. That's a genuine, usable win — name the family, get the motif.
- **Where it doesn't (LysR and most others):** family gives the architecture
  (palindrome/spacing/symmetry) but not the bases. You still need the TF's own
  sites or close orthologs.
- **The ceiling is unchanged.** Even perfect family-transfer tops out at ~0.54–0.60
  because the operator itself is a low-information (~10-bit) signal. Family
  transfer escapes the *data-scarcity* problem (no sites for this TF), not the
  *information* problem (the site is intrinsically degenerate).

## Bottom line
Family → operator is **real, modest, and family-dependent**. It buys you a motif
for a TF you have no data on — strongly for AraC/XylS and σ54 regulators, not at
all for LysR. It does not break the information wall; it just shares the same
(weak) motif across relatives. The honest "solve": use family-borrowed motifs
for the families that conserve specificity, ortholog/footprint transfer for the
rest, and treat the exact site as measured where precision matters.
