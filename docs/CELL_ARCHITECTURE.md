# The cell map — architecture, and how three models complete each other

## What this is (and isn't)

This is a **map-level model of a human cell**: every localized protein placed in its real
compartment, wired by measured regulatory and physical networks, annotated with importance,
function, disease, and trafficking route. You can knock a protein out and watch the cascade,
switch cell type, trace a metabolic pathway, or infect it with HIV and see what gets hijacked.

It is **not** a kinetic physics simulator. There are no ODEs, no concentrations over time, no
diffusion. It answers *structural / logical* questions — "what is connected to what, what breaks
if this is removed, where does this protein go, what is HIV's weak point" — not *dynamical* ones
("what is the steady-state flux through glycolysis at 5 mM glucose"). That was the deliberate goal
from the start: **lay out the cell, don't run its kinetics.**

## The backbone: our integrated model (identity & importance)

The spine of the cell is `integrated_cell_human.csv` (30,093 genes × 14 layers), built by us:

| layer | source | what it gives the cell |
|---|---|---|
| essentiality | gnomAD LOEUF constraint fused with conservation/features | is this protein load-bearing? (AUC 0.86 genome-wide, incl. the 83% non-metabolic genes FBA misses) |
| LOEUF | gnomAD | how badly does the population tolerate loss-of-function? (drives mutation lethality) |
| TF / regulon | CollecTRI + DoRothEA + TRRUST (326k edges) | who regulates whom |
| PPI degree | STRING physical ≥700 | who binds whom (hub-ness) |
| pathways | Reactome | what process a protein belongs to |
| disease | Open Targets + HPO + ClinVar | clinical weight |
| compartment | UniProt subcellular location | **where the protein physically is** |

This backbone is what the app draws. It is the one model that measurably works for the core
question ("what matters, and how is it wired"). Everything else *completes* it where it is blind.

## The three models, and why no single one is enough

The honest finding across this project: **no one model answers everything; each is strong exactly
where the others are blind.** The right design is not "pick the best model," it's "give each model
the job it is actually good at."

### 1. Our integrated model → **structure & importance** (the "what matters / how is it wired" axis)
- Answers: which proteins are essential, how strongly constrained, who regulates/binds whom,
  what compartment, what disease.
- Blind spot: it is **cell-type agnostic**. The same table describes a neuron and a hepatocyte.
  It has no notion of "this gene is switched *off* in this cell."

### 2. Atlas / scGPT-style cell-type layer → **which genes are ON, per cell type** (the "context" axis)
- From Tabula Sapiens (CELLxGENE census), per-cell-type mean expression → active gene sets and the
  master TFs that define each lineage (validated: hepatocyte→HNF4A, cardiac→GATA4/TBX5, NK→EOMES,
  macrophage→SPI1/CEBPB, endothelial→ERG/FLI1). scGPT / a cell-type GRN model plays the same role:
  it conditions the *static* wiring on a *cell state*.
- This is what makes "same genome → many cells" real in the app's cell-type selector: the backbone
  supplies the full wiring; this layer selects which sub-network is live.
- Blind spot: it says what's expressed, not what's *important* or what a perturbation *does*.

### 3. Geneformer → **in-silico perturbation direction** (the "what if I break it" axis)
- Geneformer's genuine strength is *in-silico perturbation*: delete a gene from a cell's rank-encoding
  and read how the rest of the transcriptome shifts. That is a **direction of effect**, which is
  exactly what a knockout cascade needs.
- Where it **failed** (and we proved it, so we don't misuse it): its *static* gene embeddings are
  useless for **essentiality** — 0.529 AUC, indistinguishable from random, vs 0.86 for our
  constraint backbone. We tested it on our strong suit and it lost; that told us *not* to use it
  for importance, and to reserve it for the perturbation-direction job it's built for.

### How they compose
```
                 our integrated model  ──►  identity, importance, compartment, wiring   (the map)
   atlas / scGPT cell-type layer       ──►  which sub-network is live in THIS cell type  (the context)
        Geneformer perturbation        ──►  direction/spread of a knockout's effect       (the dynamics-lite)
                                            ─────────────────────────────────────────────
                                            = a cell you can place, contextualize, and perturb
```
Each covers the other's blind spot: importance without context (1) + context without importance (2)
+ effect-direction that neither encodes (3). The app's KO cascade currently propagates over the
*measured* reg+PPI graph (deterministic, auditable); Geneformer's role is to **enrich the direction
and reach** of that cascade where measured edges are sparse — an additive signal, never the sole one.

## What is measured vs predicted (so nothing is oversold)
- **Measured / curated** (drawn as-is): compartments (UniProt), reg edges (CollecTRI/DoRothEA/TRRUST),
  PPI (STRING physical), pathways (Reactome), HIV host interactions (NCBI HIV-1 DB), disease (Open
  Targets/ClinVar), cell-type master TFs (Tabula Sapiens).
- **Predicted / modeled**: essentiality where no CRISPR truth exists (our fusion model), mutation
  lethality (from LOEUF), the KO cascade (graph propagation, optionally enriched by Geneformer).
- **Illustrative**: the 31 metabolic reactions are a curated canonical set (glycolysis→TCA→OxPhos→
  PPP→urea→nucleotide), not a genome-scale stoichiometric model.

## Honest limitations
- Map, not kinetics: no time-courses, no concentrations, no flux balance here.
- KO cascade is 2-hop over the measured graph — a reachability estimate, not a dynamical simulation.
- "Cell-inviable" is a heuristic (essential gene, or LoF of a strongly-constrained gene), not a
  growth assay.
- Dark genes (5006) are placed by compartment + network even without annotation; that placement is a
  hypothesis (guilt-by-association), explicitly flagged.

## Files
- `outputs/orphan/integrated_cell_human.csv` — the 14-layer backbone (committed).
- `colab/build_cell_complete.py` — assembles `cell_complete.json` (proteins, networks, HIV, reactions, dark, cell types).
- `colab/build_cell_app_complete.py` — renders the self-contained interactive `cell_complete.html`.
- `colab/serve_cell.py` — serves it at `http://localhost:8000/cell`.
- `colab/build_complete_cell.ipynb` — the notebook that runs the three models in their roles and builds the cell end-to-end.
