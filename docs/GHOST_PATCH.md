# Ghost-gene patch — de-ghosting the dark proteome into the cell

A **ghost gene** is a node the model carries but never fills in: all **4,993** dark-proteome genes sat in
**zero** of the 60 pathways, most with an empty function field. Before any whole-cell ML runs over the model,
those holes are patched — so the ML completes a *complete* cell, not a Swiss cheese one.

`colab/patch_ghosts.py` (`build_patch`, `apply_patch`), gated by `colab/validate_ghost_patch.py`
(scorecard axis `ghost_patch`). Applied automatically when `CellQA` loads, so the model is complete in the system.

## Two patches, strictly separated by evidence tier

| patch | tier | source | coverage |
|---|---|---|---|
| **function** | **fact** | UniProt curated function distilled from the literature (evidence-tagged experimental / by-similarity) | **3,933 / 4,993** ghosts de-ghosted |
| **pathway membership** | **prediction** | network guilt-by-association: assign a ghost to a pathway when its known partners are **enriched** there | **520** ghosts, kept in a separate `pathways_predicted` (never merged with curated members) |

The function patch is the authoritative de-ghosting. The pathway patch is a prediction — and it is only trusted
because it is **cross-validated by an independent signal**.

## The pathway patch is validated by agreement with function

A ghost's network-predicted pathway should match the theme of its *curated* function (two independent methods —
network vs literature). Measured against a shuffled baseline:

- **agreement 0.46 vs shuffled 0.24 → 1.94× lift.** The predicted pathway genuinely aligns with the independent
  function, ~2× above random. (The theme-matcher is deliberately crude and *under*-counts real agreement — e.g.
  chromatin-remodeling ACTR6 → "HATs acetylate histones" reads as a miss — so 1.94× is a conservative floor.)

Enrichment normalization was essential: without it, the huge "Generic Transcription Pathway" over-attracted
unrelated ghosts (hub bias). Requiring the ghost's partners to be **≥2× enriched** in a pathway (not just
present) fixed it — the confident calls now read right: ABI3 → RAC1 GTPase cycle (10×), ACTR6 → histone
acetylation (12×), AKAP8 → mRNA polyadenylation (7×), AKIRIN2 → Neddylation (7×).

## Why an overlay, not a 36 MB rewrite

The patch is a compact 1.4 MB overlay (`ghost_patch.json`); `apply_patch(D)` completes a loaded model in memory.
This keeps the 36 MB model file out of every commit and — more importantly — keeps **curated and predicted
strictly distinguishable** (`func`/`func_evidence` fields + a separate `pathways_predicted` map), honoring the
measured-vs-predicted discipline the rest of the model runs on.

## What this unlocks (next, not yet)

With the ghosts given a function and a pathway home, the cell is complete enough for the whole-cell ML pass —
holding all layers together to find missing/wrong links and drive completion. That runs *after* this patch, so
it operates on a filled-in cell rather than propagating the holes.
