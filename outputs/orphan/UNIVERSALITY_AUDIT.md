# Universality audit — what works on ANY bacterium from genome alone

Design contract: the **deployed** model may use only inputs derivable for an
arbitrary bacterium (its genome + a fixed reference panel for conservation).
Anything needing organism-specific *measured* data (a curated metabolic model,
RegulonDB, an expression compendium, fitness screens) is **validation/ceiling
scaffolding, not part of the universal pipeline.**

Legend: 🟢 universal-from-genome · 🟡 universal but degraded/partial · 🔴 organism-specific (not universal)

## Component-by-component

| component | input it needs | universal? | note / fix |
|---|---|---|---|
| gene calling | genome | 🟢 | Prodigal etc. |
| protein features (ESM) | protein sequence | 🟢 | ESM runs on any sequence |
| conservation / orthology | gene + reference panel | 🟢 | universal if panel is fixed & broad |
| **W1 protein essentiality (ESM+conservation)** | above | 🟢 | **validated cross-org: mtub/DeJesus 0.768.** The cleanest universal win. |
| metabolic model | curated model (iJO1366) | 🔴 | **swap for auto-reconstruction** (CarveMe/ModelSEED/gapseq) → then 🟡 (~80–90%) |
| **W2 metabolic necessity / conditional essentiality** | auto-model + media | 🟡 | universal once model is auto-built; FBA + media is organism-agnostic |
| TF identification | DBD Pfam families (sequence) | 🟢 | HMM scan, any genome |
| operon inference | gene order/strand/intergenic dist (genome) | 🟢 | **universal co-regulation signal we haven't fully exploited** |
| specific-TF operators (few-target) | sequence + intergenic scan | 🟡 | universal but only the sharp/few-target TFs (precision 0.5–1.0) |
| family→operator (AraC/XylS, σ54) | family + sequence | 🟡 | universal, but only the specificity-conserving families |
| **regulatory edges — global regulators** | RegulonDB / expression / co-fitness | 🔴 | **not obtainable from genome.** Hard universal limit. |
| co-expression / co-fitness edges | measured compendia | 🔴 | E. coli / ~60 feba orgs only |
| network motif detection | an edge graph | 🟡 | universal only as far as the edge graph is (operon + specific-TF level) |
| Gillespie dynamics engine | a network + params | 🟢 engine / 🔴 rates | engine universal; quantitative rates are illustrative either way |
| closed-loop effector→TF logic (CRP) | hand-coded E. coli logic | 🟡 | the *principle* is universal; global regulators (CRP/FNR/ArcA) are conserved so partly transfers; per-organism specifics are not universal |

## The universal core (deployable on any genome)
1. genome → genes → **ESM + conservation → essentiality** (W1) 🟢 — the proven cross-organism predictor.
2. genome → **auto-reconstructed metabolic model → FBA necessity + conditional essentiality across media** (W2) 🟡.
3. genome → **operons** (gene-order) + **TF identification** (DBD families) → an **operon-level + specific-TF regulatory skeleton** 🟡.
4. genome → **family→operator for AraC/XylS & σ54** 🟡 (the families where sequence works).
5. motifs + dynamics on whatever edge graph the above yields.

## The honest hard limit (state plainly)
You **cannot** get the **global-regulator regulatory edges** or **quantitative
kinetics** (rates, exact sites, concentrations) from genome alone for an arbitrary
bacterium — these require measurement that exists for only a handful of species.
This is not a tooling gap; it's the information boundary we proved repeatedly
(operator wall ~0.55, kcat non-conserved, regulon-size↔specificity r=−0.94). So
the universal deliverable is: **essentiality (protein + metabolic) + conditional
(medium-driven) essentiality + operon/specific-TF regulation** — *not* a
fully-quantitative global regulatory dynamics model.

## What must change to honor "universal, not E. coli"
- **Demote** RegulonDB-, PRECISE-, iJO1366-, and CRP-logic-based pieces to the
  *validation* layer (they proved the mechanisms; they are not the product).
- **Promote / build** the genome-only path: auto-reconstruction for W2, operon
  inference, sequence-only specific-TF/family operators — and validate W1 by
  **leave-one-organism-out across the whole panel**, not on one organism.
- Report every future result with its universality tag (🟢/🟡/🔴).

Bottom line: W1 is universal and proven. W2 is universal once we auto-reconstruct
instead of using curated models. The regulatory layer is universal only at the
operon + specific-TF level; global regulation and quantitative dynamics are
fundamentally organism-specific (measured), and we should stop presenting the
E. coli versions as the universal model.
