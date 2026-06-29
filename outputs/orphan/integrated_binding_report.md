# The 4-dimensional biophysical TF->operator framework — built and measured

Faithfully implemented the proposed multidimensional cross-correlation:
D1 structural architecture (dyad/palindrome), D2 direct readout (PWM),
D3 indirect readout (DNA shape: twist/roll/MGW/electrostatic from dinucleotide
params), D4 macro-context (distance-to-gene + helical phase), plus protein-side
family class. Logistic fusion, **leave-TF-out** cross-validation, E. coli/RegulonDB.

## A leak first (the cautionary tale)
Initial run scored **AUC 0.874** — a leak: the PWM was trained on the same
targets later scored as positives (in-sample). This is exactly why such
pipelines *look* like they work. Fixed with a train-motif / test-on-held-out
split.

## Honest result

| dimension set | leave-TF-out AUC | Δ vs PWM |
|---|---|---|
| D2 sequence PWM | 0.542 | — |
| D2 + D1 architecture | 0.548 | +0.006 |
| D2 + D3 shape field | 0.541 | −0.001 |
| D2 + D4 position/phasing | 0.542 | +0.000 |
| D2 + family | 0.542 | +0.000 |
| **ALL four dimensions** | **0.546** | +0.004 |

The full framework ties the plain motif at the ~0.54 wall.

## Why (confirmed empirically, not asserted)
- **D1, D2, D3 are all deterministic functions of the operator sequence.** Shape,
  symmetry, and the PWM are transformations of the same ~10 bits. A
  cross-correlation of derived channels cannot exceed the information in the
  variable they are derived from. D3 adds nothing (−0.001); our prior
  `binding_field_test` got the same null (0.505) with proper Rohs pentamer
  shapes — so it is structural, not a bad shape table.
- **D4 (position) does not discriminate** because both targets and non-targets
  are already inside the promoter window; the positional prior's value is in
  restricting to the window, which both classes share.
- **Family** is real but small and family-specific (AraC/XylS), and does not
  generalize in leave-TF-out.

## Caveat
The protein side (V_TF) was proxied by **family class**, not the actual
**recognition-helix residue code** (D2 covariance). That residue->base code is
the one untested lever with potentially *independent* (protein-side) information,
but family-transfer already showed it pays off only where specificity is
conserved (AraC/XylS), and no accurate general protein->DNA code exists.

## Bottom line
The framework is a correct and beautiful representation of the binding physics
that adds ~0 predictive bits, because all DNA-side dimensions re-encode the same
sequence. The physics says the binding signal is genuinely degenerate (~10 bits),
not that we were representing it poorly. The levers that carry *independent*
information remain: measured sites (ChIP/DAP-seq), many-genome footprinting,
recognition-helix residues (for specificity-conserving families), and the
regulatory EDGE from co-fitness. Representation was never the bottleneck;
information is.
