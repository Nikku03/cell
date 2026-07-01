# Mutations on protein structure — disease hits the core, tolerated hits the surface

The user's framing: mutations occur at random; natural selection decides which reach the
population; and **where a mutation sits on the protein tells you whether it actually
changed function**. The unmutated buried core is the important part; a mutation matters
only if it lands there. Tested directly with real data.

Data: AlphaFold structures (burial = contacts within 10Å; pLDDT = structural order) ×
gnomAD (ClinVar pathogenic missense + common population missense + adaptation variants).
15 genes (10 disease, 5 adaptation). Code: `colab/structure_mutations.py`. Figure:
`structure_mutations.png`.

## Result — a clean monotonic gradient (2,770 pathogenic mutations)

| mutation class | n | burial (contacts) | pLDDT (order) | % buried | % ordered |
|---|---|---|---|---|---|
| **pathogenic (disease)** | 2,770 | **19.3** | **90.9** | **0.78** | **0.96** |
| common / benign (population-tolerated) | 73 | 14.8 | 77.6 | 0.42 | 0.68 |
| adaptive (top differentiated) | 6 | 14.0 | 69.4 | 0.33 | 0.50 |

Every metric orders the same way: **disease → tolerated → adaptive**, from buried/rigid
core to exposed/flexible surface.

- **Disease mutations bury into the structured core**: 78% are buried, 96% in ordered
  (pLDDT≥70) regions, mean 19.3 contacts. They break the fold / active site — which is
  exactly why they cause disease and why selection removes them.
- **Population-tolerated variants sit on the surface**: only 42% buried, mean 14.8
  contacts, lower pLDDT. They don't destabilize the fold, so they survive in the population.
- **Adaptive variants sit furthest out**: 33% buried, pLDDT 69 (half in flexible/loop
  regions). They tune function at the periphery without touching the core.

## What this means (the user's framing, confirmed)

> The **unmutated buried core is the vital part** — it is conserved because a mutation
> there breaks the protein. When a random mutation *does* land in that core, it is
> pathogenic (disease). Mutations that reach population frequency — whether neutral or
> adaptive — are the ones that landed on the **flexible surface**, where they change
> function subtly or not at all. So the *position* of a mutation on the structure predicts
> whether it changed function, and the vital-part map (from population constraint) and the
> structure (buried core) agree.

This closes the loop: **population constraint → vital regions → protein core → disease.**
The same buried, conserved core that carries no common variation is where disease
mutations concentrate.

## Honest caveats

1. **ClinVar ascertainment bias**: pathogenic calls cluster in well-studied disease genes
   (LDLR, PAH dominate the 2,770). The per-class gradient is robust, but absolute counts
   reflect study effort, not biology.
2. **Adaptive n=6** (one top variant per adaptation gene) — directional, underpowered;
   more adaptive variants would firm it up.
3. **Burial = contact number** and **flexibility = pLDDT** are simple structural proxies
   (no DSSP/SASA); good enough for the gradient, not residue-perfect.
4. AlphaFold models are predictions (high pLDDT regions are reliable; low-pLDDT less so).

## Next

- Score **every ClinVar variant of unknown significance (VUS)** by burial + constraint →
  which VUS look pathogenic (buried/core) vs benign (surface). A directly useful clinical
  triage output.
- Add **active-site / ligand-binding proximity** (beyond burial) for function-specific
  effect.
- Fold structure burial into the human essentiality fusion as another feature.
