# `dock_diagnose.json` — the numbers are real, the READING in its `log` is wrong

This file is kept byte-identical to what the script printed. Its measurements stand. Its conclusion does not,
and the way it failed is worth more than the run itself.

## What it concluded

    Native clash is 0.000x interlock -- the true pose is NOT being penalised, so the grid is fine.
    ... So the score is WEAK, not broken, and no tuning of the clash penalty rescues it.

## Why that is wrong

The script branched on one question: *is the native pose being charged a clash penalty?* Clash was 0, so it
took the "grid is fine" branch. But the number printed immediately beside it was:

    1ACB: native interlock     8   |  max interlock anywhere   989  |  true 5A atom contacts 509
    1AK4: native interlock    -0   |  max interlock anywhere  1873  |  true 5A atom contacts   0

The native interface was not being *outranked*. It was scoring **nothing**. And 1AK4 reported zero atom pairs
within 5 Å across what is supposed to be a crystal interface, which is impossible.

Two bugs, both in `dock_fft.py` as it stood, both invisible to this script's single predeclared branch:

1. **The shells never touched.** Atom *centres* rasterised and dilated one cell. At 1.4 Å spacing a van der
   Waals contact is 3.5 Å ≈ 2.5 cells, so one cell of dilation per side leaves a 0.5-cell gap. Rasterising to
   real vdW radii with the receptor shell grown outward took native interlock from **8 to 177**, clash still 0.
2. **Wrong chain pair.** `chains[0], chains[1]` is alphabetical, not biological. 1AK4 A/B has zero contacts
   (real pair B/C); 1BRS and 1B2S picked up crystal-packing contacts of 153 and 140 instead of the real 568
   and 574. Four of six complexes were docked on a non-interface.

Corrected, the native translation at the native rotation ranks in the **top 1–4 of all 884,736 placements in
all six complexes** — the opposite of "the score cannot rank".

## The transferable lesson

A gate that only tests the failure mode you thought of will confidently clear the one you did not. This script
predeclared a branch for "native is being penalised" and none for "native is scoring zero", so it read a broken
representation as a weak score and would have written that into the record as a finding about docking.

`dock_fft.py` now carries a hard sanity gate on the other side of the same question: a complex whose native pose
does not score well above an arbitrary placement is reported as REPRESENTATION FAIL and excluded, rather than
silently dragging the hit rate down.

Superseded by the rewritten `dock_fft.py` (commit `f57547b`) and its regenerated `dock_fft.json`.
