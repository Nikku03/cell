# The virtual cell — full 360

A complete walkthrough of every part of the cell model: its **data source**, **how it's integrated**,
and **what it does**. The system is a 3-layer stack:

- **Substrate** — measured facts from public databases (what connects to what).
- **Models** — three orthogonal learned signals: importance (M1), context (M2), dynamics (M3).
- **Reasoning** — derived layers that fuse the above: measured essentiality, co-essentiality,
  synthetic lethality, ensemble confidence, novelty flags.

Everything is baked at build time (`build_cell_complete.py` → `cell_complete.json`) and rendered as a
self-contained interactive page (`build_cell_app_complete.py` → `cell_complete.html`).

---

## PART A — The three models

### M1 · our integrated essentiality model  (axis: **importance**)
- **Data:** the 14-layer `integrated_cell_human.csv` (gnomAD LOEUF constraint, TF/regulon, PPI degree,
  pathways, CpG/enhancers, disease) + Hart CEGv2/NEGv1 truth. Trained with DepMap as truth.
- **Integration:** a gradient-boosted classifier (CV AUC 0.974) predicts essentiality for genes with no
  CRISPR label → `predicted_essentiality.csv`; the builder fills unlabeled genes (`ess_src="model1"`,
  with probability).
- **Does:** gives every gene a load-bearing score even where nothing was measured; drives KO lethality.

### M2 · atlas / scGPT cell-type layer  (axis: **context**)
- **Data:** Tabula Sapiens via CELLxGENE census (streamed + subsampled at build time).
- **Integration:** per-cell-type mean expression → `celltype_expression.csv`; the builder compacts it to
  per-gene **abundance** (0–15) + a **cell-type expression bitmask**, and derives **master TFs** by
  specificity → `celltype_masters.json`.
- **Does:** abundance (node size), which genes are active per cell type, **cell-type-specific wiring**
  (network pruned to expressed genes), and the **differentiation** engine. *(Fallback = 8 curated cell
  types until the GPU run populates ~40 real ones.)*

### M3 · Geneformer  (axis: **dynamics**)
- **Data:** Geneformer pretrained weights (HuggingFace) + tokenized atlas cells.
- **Integration:** gene embeddings → cosine neighbors for target regulators → `gf_perturb.json`;
  optional true `InSilicoPerturber` delete.
- **Does:** enriches the knockout cascade with predicted downstream genes (shown purple); powers
  **dark-gene function** by embedding proximity. *(Populated by the GPU run.)*

---

## PART B — Substrate data layers (measured, all integrated)

Each entry: **Source → Integration → Does.**

| layer | source | integration | does |
|---|---|---|---|
| **Localization** | UniProt subcellular | `gene_compartment.json`, 13 buckets | places every protein in its compartment (the spatial cell) |
| **Regulation (signed)** | CollecTRI/DoRothEA/TRRUST | `reg` edges `[a,b,sign]` (45,499) | GRN; activates ▲ / represses ▼; drives KO + overexpression cascades |
| **PPI** | STRING physical ≥700 | `ppi` edges (141,532) | who binds whom; cascade propagation; hub detection |
| **Signaling** | SIGNOR | `sig` edges `[a,b,sign]` (17,432) | directed signal flow "signals to ↑/↓ / signaled by" |
| **Metabolism** | Human-GEM (12,931 rxns) | ENSG→symbol; per-enzyme reaction list (`generxn`) | 2,549 enzymes show real substrate→product chemistry |
| **Pathways** | Reactome | `path` per gene + 60-pathway selector | functioning modules; pathway highlight |
| **Complexes** | EBI Complex Portal | UniProt→symbol; `complexes` + `gene2cplx` (2,039) | named machines (e.g. TP53–MDM2 complex) |
| **Drugs** | DGIdb | `drugs` per gene (4,275 genes) | drug → target → action (EGFR → gefitinib…) |
| **PTMs** | UniProt `ft_mod_res` | `ptm` per gene (8,425 genes) | modification sites + types (TP53: 30 sites) |
| **Ligand–receptor** | CellPhoneDB | `lr` pairs (948) | cell–cell signaling partners |
| **Cell cycle** | Reactome | `cellcycle` phase tag (153 genes) | G1/S/G2/M phase per gene |
| **Genome locus** | NCBI gene_info | `chrom`, `tss` per gene | chromosomal position |
| **Regulatory elements** | UCSC CpG + enhancer counts | `cpg`, `enh` per gene | CpG-island promoter, enhancer load |
| **Disease** | Open Targets / HPO / ClinVar | `ndis`, `otdis` per gene | disease associations + druggability |
| **Structure** | AlphaFold / ESMFold | `struct` (13 curated) + `fold` | mutation site, WT-vs-mutant fold RMSD |
| **Virus (HIV)** | NCBI HIV-1 DB | `hiv` map + weak points (19 proteins) | infection: hijacked host machinery + drug-target weak points |
| **3D genome** | GM12878 HiCCUPS loops | TSS→anchor mapping (`loops3d`, 767 genes) | enhancer-promoter chromatin loops per gene |

---

## PART C — Reasoning / derived layers (fuse models + data)

### Measured essentiality (DepMap)
- **Source:** CRISPRGeneEffect.csv — 1,100 cancer cell lines × 18,443 genes (Chronos).
- **Integration:** `compute_depmap_essentiality.py` → per-gene mean effect + fraction of lines dependent
  (`depmap_essentiality.csv`); builder uses it as **top-priority truth** (overrides Hart + M1).
- **Does:** 15,913 genes now have **measured** essentiality; each shows "dependent in X% of cancer lines".

### Co-essentiality + synthetic lethality (DepMap)
- **Source:** the same CRISPR matrix.
- **Integration:** `compute_depmap_codep.py` correlates every gene's knockout profile across the 1,100
  lines → `depmap_codep.json` (co-essential partners) + `depmap_sl.json` (pairs where both are
  individually dispensable yet co-dependent).
- **Does:** per-gene co-essential partners (TSC1 → TSC2/DEPDC5/NPRL2 = mTOR/GATOR module);
  **double-KO** shows the measured DepMap r and flags synthetic-lethal candidates.

### Ensemble confidence + novelty flags
- **Source:** agreement across DepMap (measured) + LOEUF (constraint) + M1 (model).
- **Integration:** per gene, a `conf` = "high" (all agree) or "split" (disagree); disagreements set a
  `flag`.
- **Does:** 15,006 genes carry a confidence badge; **1,172 novelty candidates** flagged — e.g.
  *cancer-dependency* (essential in cancer, LoF-tolerant in population) or *germline-constrained yet
  cancer-dispensable*. This is the discovery shortlist.

### Dark genes
- **Source:** genes with no pathway/disease annotation (5,006).
- **Does:** the function frontier, placed by compartment + network; function predicted by M3 embeddings.

---

## PART D — Engines (what you can *do*)

| engine | how it works | output |
|---|---|---|
| **Remove / Mutate** | BFS 2-hop over reg+PPI (cell-type-gated) + M3 enrichment; lethality from DepMap/constraint | damaged cell: affected proteins, disrupted processes/pathways, viable or inviable |
| **Overexpress / activate** | propagate **signed** regulation 2 hops | up (green) / down (red) targets, processes shifted |
| **Double knockout** | union two cascades + shared targets + DepMap co-dependency | synthetic-lethal flag with measured r |
| **Differentiation** | genes ON/OFF between two cell-type expression states + driver TFs | reprogramming plan (e.g. hepatocyte→cardiac) |
| **Infect: HIV** | NCBI host-interaction map + dependency factors | hijacked machinery + weak points (drug targets) |
| **Functioning pipeline** | genome → input (regulators) → expressed → function (reaction/TF/complex) → output | per-gene mechanistic trace, all links clickable |
| **Trafficking journey** | compartment-derived birth→destination with machinery per step | gene→mRNA→ribosome→ER→Golgi→location |

## Interactive modes
Explore · Processes · Metabolism · Remove/Mutate · Overexpress · Dark genes · Infect:HIV ·
cell-type selector · pathway selector · search.

---

## PART E — By the numbers (current build)
- 16,492 localized proteins · 45,499 signed reg + 141,532 PPI + 17,432 signaling edges
- 2,549 metabolic enzymes (12,931 Human-GEM reactions) · 2,039 complexes · 60-pathway selector
- 4,275 drug-targeted genes · 8,425 PTM genes · 948 ligand-receptor pairs · 153 cell-cycle genes
- 15,913 DepMap-measured essential · 531 model-filled · 13,451 co-essential · 1,256 SL pairs
- 767 genes with 3D loops · 15,006 confidence-scored · **1,172 novelty candidates** · 5,006 dark genes
- HIV: 19 viral proteins → ~3,500 host, 747 weak points

## PART F — How to run
1. **Download data:** `download_all_data.ipynb` → `MyDrive/virtual_cell_data/`.
2. **Build (full):** `build_complete_cell.ipynb` on an A100 — runs M1/M2/M3 + DepMap reductions +
   all layers → `cell_complete.html` (renders inline).
3. **Serve locally:** `python colab/serve_cell.py` → `http://localhost:8000/cell`, or just open the HTML.

## PART G — Honest boundaries
- **Map, not simulator:** no kinetics/flux/time-courses (no clean bulk kcat source; BRENDA gated).
- **Bias:** DepMap = cancer lines; Tabula Sapiens = healthy adult — context claims inherit these.
- **Predictions are hypotheses:** novel calls (dark-gene function, disagreement flags, SL pairs) are
  ranked leads requiring experimental validation, not proven results.
- **Correlation ≠ causation** for co-essentiality/co-expression — the confidence layer + adversarial
  checks are the guardrails.
- Still open (no clean data): true kinetics, lipids/ions as entities, protein turnover, tissue/spatial,
  full methylation tracks; GEO conditions pending integration.
