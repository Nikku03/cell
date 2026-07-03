# Tissue model — plan (mirrors the cell-model phasing)

## Current state (checked)
`build_tissue_model.py` builds `tissue_model.json`: 9 tissues, cell types from **HPA single-cell
expression**, wired by Omnipath ligand-receptor cell-cell communication (specificity-filtered), with
CellChat pathway families, CellPhoneDB paracrine/juxtacrine mode, DGIdb druggable channels,
SIGNOR+CollecTRI downstream, 12 curated endocrine axes, and (now) NicheNet ligand→target. Explorer:
`build_tissue_explorer.py` (BODY view + TISSUE view).

**The gap (the user's vision):** the tissue model uses raw HPA expression per cell type — it does **not**
instantiate the rich single-cell model (`cell_complete.json`) per cell type. "Make cell models and
integrate them to make tissue" = load N cell-type instances of the full cell model and wire them.

## The bridge: cell model → per-cell-type instances → tissue
`build_tissue_from_cells.py` (new): takes `cell_complete.json` + `celltype_expression.csv` (Model 2's
per-cell-type expression) and produces, per cell type, a **cell instance** = the full cell model masked
to the genes that cell type expresses (its active network, active TFs, active reactions). Then wires the
instances by cell-cell communication where the ligand is *specifically* expressed in the sender and the
receptor in the receiver — the same specificity logic, but now over the full modelled network, so a
received signal can be propagated through that cell type's *actual* regulatory/PPI graph to its targets.

## Phases (same discipline as the cell model)
- **T1 — Per-cell-type instances.** Mask `cell_complete` by each cell type's expressed genes →
  `cell_instances.json` {celltype: active gene set + active subnetwork}. *Sandbox-testable with synthetic
  celltype expression.*
- **T2 — Cell-cell communication graph.** Reuse Omnipath LR + specificity; edge = ligand specific in
  sender ∧ receptor present in receiver. (Already largely in `build_tissue_model`.)
- **T3 — Intracellular propagation.** For each received signal, propagate receptor → SIGNOR/NicheNet →
  TF → targets **within the receiver's active network** → predicted transcriptional consequence in the
  receiver. This is the payoff of using the full model per cell type (not just HPA expression).
- **T4 — Tissue-level convergence / emergent properties.** Apply the convergence idea across cell types:
  signaling loops (A→B→A), feedback, and multicellular modules that no single cell shows.
- **T5 — Explorer.** Extend `build_tissue_explorer` to render the instance networks + propagated signals.

## What it needs to run (Colab)
`cell_complete.json` (built) + `celltype_expression.csv` (Model 2, Colab) + HPA + Omnipath (fetched).
The bridge processor is sandbox-unit-tested on synthetic celltype expression; the real run is Colab.

## Honest note
Tissue adds **cell-cell** and **cross-tissue** structure — still steady-state, still no kinetics. The
value is the same as the cell model: convergence + propagation over independent lenses, now multicellular.
