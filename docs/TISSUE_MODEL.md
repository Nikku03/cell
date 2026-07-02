# The Tissue Model — cell–cell communication (separate from the cell model)

A **distinct** model from the single-cell map: a tissue is *many cell types talking to each other*.
Where the cell model resolves one cell's inside, the tissue model resolves **who signals whom** across
the cells that make up an organ. Kept separate on purpose (own data, own files, own app) so the two
don't mix.

## What it is
- **8 tissues** (Liver, Heart, Lung, Brain, Kidney, Intestine, Skin, Immune) — curated from real HPA cell types.
- **~5 cell types each**, with identity markers.
- **Cell–cell communication**: for every ordered pair of cell types, the ligand→receptor channels where
  the **ligand is specifically produced by the sender** and the **receptor is expressed on the receiver**.
- **Perturbation**: knock out a gene → the channels that use it break → you see which cell–cell links
  weaken **and which downstream genes lose input inside the receiving cells** (receptor → SIGNOR → TF → targets).

## Data (self-contained — no census/GPU needed)
| source | role | scale |
|---|---|---|
| **HPA single-cell RNA** (`rna_single_cell_type.tsv`) | which genes each cell type expresses | 20,151 genes × 154 cell types |
| **Omnipath ligand–receptor** (`ligrecextra`) | the ligand→receptor pairs (signed) | 6,555 pairs |

## How it's built (`build_tissue_model.py`)
1. HPA → per cell type, expressed genes + a **specificity** score (enrichment vs the mean across all 154 types).
2. A ligand counts as "produced by" a cell type only if it's **expressed AND specific** (spec ≥ 2) — this
   is what removes ubiquitous housekeeping genes (RPS27A, B2M) that would otherwise look like signals.
3. For each tissue, each (sender→receiver) pair: channels = LR pairs with a specific ligand in the sender
   and the receptor expressed in the receiver, ranked by combined specificity.

## Validation (recovers textbook biology)
- **Immune checkpoint:** T-cell → macrophage returns **CTLA4→CD80/CD86**, CD28→CD86 — the canonical costimulation/checkpoint axis.
- **Synapse:** astrocyte/neuron returns **NRXN1→NLGN2** — the neurexin–neuroligin synaptic adhesion axis.
- **Fibrosis:** hepatic stellate → endothelial returns **COL1A1/COL1A2→CD93**, and stellate markers are all collagen/ECM (COL1A1, COL3A1, BGN) — the hallmark of liver fibrosis. Knocking out **COL1A1** breaks 29 channels across 5 links.
- **Endothelial signaling:** **GDF2 (BMP9)→ACVRL1 (ALK1)** recovered.

## Honest limits
- **Potential communication, not measured flux.** A channel means "sender *can* make the ligand, receiver *can* receive it" — not that the signal is firing at a given moment.
- **No spatial architecture** — cell types are shown in a ring, not their real tissue geometry (no spatial-transcriptomics layer yet).
- **Expression is HPA reference** (healthy), and specificity thresholds are tunable — a few real axes (e.g. VEGFA from AT2) fall below threshold.
- **Tissue composition is curated** (the major cell types per organ), not exhaustive.
- Downstream *inside* the receiver IS now wired (receptor → SIGNOR signaling → CollecTRI TFs → target genes), reusing the cell model's shared network data — so a perturbation in one cell propagates through the broken signal into the next cell's genes. Still **topological, not kinetic** (which genes respond, not by how much).

## Files
- `colab/build_tissue_model.py` → `tissue_model.json`
- `colab/build_tissue_explorer.py` → `tissue_explorer.html` (the app)
