# A Map-Level Virtual Human Cell
### Integrating 20 measured datasets and 3 machine-learning models into an interactive, perturbable, confidence-scored cell

---

## 1. One-line pitch
A single interactive cell where **every human protein** carries its importance, context, wiring, and
druggability — and where you can **knock it out, mutate it, overexpress it, infect it, or differentiate
it into another cell type** and watch the consequences propagate, each answer backed by a **source and a
confidence score**.

---

## 2. Executive summary
Biology's knowledge is scattered across dozens of databases and a handful of foundation models, none of
which talk to each other. This project fuses **20 measured datasets** and **3 ML models** into one
coherent, queryable cell:

- **16,492 localized proteins**, wired by 45,499 signed regulatory + 141,532 physical + 17,432 signaling
  edges, carrying 12,931 metabolic reactions, 2,039 complexes, drugs, PTMs, disease, and structure.
- **Measured essentiality** for 15,913 genes (DepMap, 1,100 cancer lines) — not predicted.
- **Ensemble confidence** on every prediction, surfacing **1,172 novelty candidates** where measurement,
  constraint, and model disagree.
- A perturbation/differentiation engine that is **cell-type-specific** and **validated against known
  biology** (recovers mTOR co-essentiality, EGFR drugs, TP53 targets, HNF4A→liver identity).

The deliverable is a **self-contained interactive HTML cell** + reproducible notebooks. It is a *map*,
not a kinetic simulator — it answers "what is connected, what breaks, what's important, in what context,
and what to test," not "what is the flux at time t."

---

## 3. The problem
- **Fragmentation:** essentiality is in DepMap, regulation in CollecTRI, metabolism in Human-GEM, drugs in
  DGIdb, structure in AlphaFold — a researcher must query each separately and mentally integrate.
- **Foundation models are siloed and mis-applied:** Geneformer/scGPT are powerful but often benchmarked on
  the wrong task; their static embeddings score *random* on essentiality, yet are excellent at perturbation.
- **No confidence:** most in-silico predictions arrive with no honest measure of trust.
- **Context blindness:** most "networks" are the same for a neuron and a hepatocyte.

This project addresses all four: integration, right-tool-for-right-job models, ensemble confidence, and
cell-type context.

---

## 4. Design principles
1. **Map, not simulator.** Structural/logical resolution (what connects to what, what breaks) — kinetics
   are explicitly out of scope (no clean bulk rate data exists; see §12).
2. **Substrate × Behavior.** Databases supply the *substrate* (facts); models supply the *behavior*
   (importance, context, dynamics). A complete cell needs both.
3. **Three orthogonal models, each doing only what it's best at** — proven by benchmark, not assumed.
4. **Every label carries provenance and confidence.**

---

## 5. Architecture & pipeline stages

```
STAGE 0  DOWNLOAD        20 datasets -> MyDrive/virtual_cell_data/   (download_all_data.ipynb)
STAGE 1  BUILD SUBSTRATE parse DBs -> localized proteins + all measured layers
STAGE 2  MODEL INFERENCE M1 (essentiality) · M2 (atlas expression) · M3 (Geneformer)   [GPU]
STAGE 3  REASONING/FUSE  DepMap essentiality + co-essentiality + SL + ensemble confidence + novelty
STAGE 4  RENDER/INTERACT self-contained cell_complete.html + engines
```
Build code: `build_cell_complete.py` (assembles `cell_complete.json`) → `build_cell_app_complete.py`
(renders HTML). Orchestrated by `build_complete_cell.ipynb`.

---

## 6. DATA APPENDIX — every dataset: source, stage, use

Legend for **Stage**: 0 download · 1 substrate build · 2 model input · 3 reasoning · 4 render.

| # | dataset | source (URL host) | license | size | stage | processing | powers (what it does) |
|---|---|---|---|---|---|---|---|
| 1 | UniProt subcellular localization | rest.uniprot.org | CC BY 4.0 | 4 MB | 1 | parse to 13 compartment buckets | places each protein in its organelle (the spatial cell) |
| 2 | CollecTRI / DoRothEA / TRRUST | omnipathdb / raw repo | CC BY | 26 MB | 1 | signed edges (activate/repress) | gene-regulatory network; KO & overexpression cascades |
| 3 | STRING physical links v12 | stringdb-downloads.org | CC BY 4.0 | 27 MB | 1 | ENSP→symbol, score ≥700 | protein–protein binding; cascade propagation |
| 4 | SIGNOR | signor.uniroma2.it | CC BY-SA | 20 MB | 1 | directed signed signaling | receptor→kinase→TF signal flow |
| 5 | Human-GEM | github SysBioChalmers | CC BY 4.0 | 1.8 MB | 1 | ENSG→symbol; reaction per enzyme | genome-scale metabolism (2,549 enzymes) |
| 6 | Reactome (Ensembl2Reactome) | reactome.org | CC0 | 51 MB | 1 | pathway per gene + cell-cycle phase | pathway modules; cell-cycle tagging |
| 7 | EBI Complex Portal (9606) | ftp.ebi.ac.uk | CC0 | 5 MB | 1 | UniProt→symbol; members | named complexes (ribosome, TP53–MDM2…) |
| 8 | DGIdb interactions | dgidb.org | open (MIT-ish) | 12 MB | 1 | gene→drug→action | drug targeting per gene |
| 9 | UniProt PTM features | rest.uniprot.org | CC BY 4.0 | 5.4 MB | 1 | count + type of MOD_RES | post-translational modifications |
| 10 | CellPhoneDB | github ventolab | MIT | 0.5 MB | 1 | UniProt→symbol pairs | ligand–receptor / cell–cell |
| 11 | NCBI gene_info | ftp.ncbi.nlm.nih.gov | public domain | 4.9 MB | 1 | entrez↔symbol, locus | genome position; HIV host mapping |
| 12 | UCSC CpG islands | hgdownload.soe.ucsc.edu | free | 0.7 MB | 1 | promoter CpG flag | regulatory-element annotation |
| 13 | HPO / Open Targets / ClinVar | HPO, OT GraphQL | CC BY | 20 MB | 1 | disease + druggability | disease associations |
| 14 | gnomAD constraint (LOEUF) | gnomAD (baked in CSV) | free | — | 1,3 | LOEUF per gene | germline importance; confidence input |
| 15 | AlphaFold / ESMFold | AlphaFold DB / API | CC BY 4.0 | on-demand | 1 | WT vs mutant fold RMSD | mutation structural impact |
| 16 | NCBI HIV-1 interaction DB | ftp.ncbi.nlm.nih.gov | public domain | 4.8 MB | 1 | viral→host map | HIV infection engine |
| 17 | **DepMap CRISPRGeneEffect 24Q2** | figshare | CC BY 4.0 | 382 MB | 2,3 | Chronos matrix reductions | **measured essentiality + co-essentiality + SL** |
| 18 | HiCCUPS 3D loops (GSE63525) | ftp.ncbi (GEO) | public | 0.6 MB | 1 | TSS→anchor mapping | enhancer-promoter chromatin loops |
| 19 | Tabula Sapiens (CELLxGENE) | census S3 | CC BY 4.0 | streamed | 2 | per-cell-type mean expression | abundance, cell-type wiring, differentiation |
| 20 | Geneformer weights | HuggingFace | Apache-2.0 | ~GB | 2 | gene embeddings / perturbation | dark-gene function; cascade enrichment |
| — | RNAcentral (ncRNA) | ftp.ebi.ac.uk | CC0 | 160 MB | (pending) | id-mapping | ncRNA inventory (needs targets) |
| — | ARCHS4 / GEO | maayanlab | CC BY | 57.6 GB | (pending) | per-condition expression | disease/condition states |
| — | Hart CEGv2 / NEGv1 | repo | free | tiny | 2 | truth labels | Model-1 training set |

---

## 7. THE THREE MODELS — functioning & stats

### M1 — Integrated essentiality model  ·  axis: *importance*
- **What:** a gradient-boosted classifier over 14 biological features (LOEUF, TF/regulon, PPI degree,
  pathways, CpG/enhancers, disease).
- **Data / stage:** trained at Stage 2 on features + DepMap/Hart truth.
- **Stat:** **cross-validated AUC 0.974** (genome-wide constraint alone: 0.86).
- **Role:** fills essentiality for genes with no CRISPR label (now only 531 gaps, since DepMap covers the rest).

### M2 — Atlas / cell-type layer  ·  axis: *context*
- **What:** per-cell-type mean expression from Tabula Sapiens; master TFs by specificity.
- **Data / stage:** Stage 2, streamed + subsampled from the CELLxGENE census.
- **Stat / validation:** recovers canonical master TFs — HNF4A→hepatocyte, GATA4/TBX5→cardiac,
  SPI1/CEBPB→macrophage, EOMES→NK, ERG/FLI1→endothelial.
- **Role:** abundance (node size), which genes are active per cell type, **cell-type-specific network
  pruning**, and the **differentiation engine** (ON/OFF genes + driver TFs between states).

### M3 — Geneformer  ·  axis: *dynamics*
- **What:** pretrained rank-encoding transformer; gene embeddings + in-silico perturbation.
- **Data / stage:** Stage 2, weights from HuggingFace + tokenized atlas cells.
- **Stat (honest):** static embeddings score **AUC 0.53 (random) on essentiality** — which is *why we do
  not use it for importance*. Its genuine strength is perturbation direction and functional proximity.
- **Role:** enriches the KO cascade with predicted downstream genes; predicts **dark-gene function** by
  embedding neighbors.

### DepMap — measured ground truth (not a model)
- 1,100 cancer cell lines × 18,443 genes of CRISPR Chronos scores.
- **15,913 genes measured**, 1,591 common-essential; used as top-priority truth + co-essentiality + SL.

**Design lesson (benchmarked, not assumed):** graph nets *lost* to a plain MLP (GCN 0.947 / GAT 0.943 vs
MLP 0.973); Geneformer embeddings are random for essentiality; FBA covers only ~17% of human essentiality.
So each model is assigned the one job it measurably wins at.

---

## 8. THE REASONING / FUSION LAYER
| derived signal | method | result |
|---|---|---|
| Measured essentiality | fraction of DepMap lines dependent (<−0.5) | 15,913 genes, measured |
| Co-essentiality | correlate every gene's KO profile across 1,100 lines | 13,451 genes → functional partners |
| Synthetic lethality | co-dependent **and** both individually dispensable | 1,256 candidate pairs |
| **Ensemble confidence** | agreement across DepMap + LOEUF + M1 | 15,006 scored |
| **Novelty flags** | measurement-vs-constraint disagreement | **1,172 candidates** (e.g. cancer-dependency) |

This layer is where the models *complete each other*: agreement → confidence, disagreement → discovery,
embeddings → dark-gene function, co-dependency → combinations.

---

## 9. THE CELL MODEL — what each part does
- **Spatial scaffold:** proteins drawn in their real compartments (nucleus, cytoplasm, ER, Golgi,
  mitochondria, membrane, …), node size = abundance, ring = disease, color = process.
- **Per-gene identity:** every protein carries — compartment, process, genome locus, essentiality
  (measured/predicted + % cancer lines), LOEUF, confidence badge, novelty flag, abundance, PTMs,
  complex membership, signaling partners, drugs, disease, ligand/receptor role, co-essential partners,
  3D loops, and its metabolic reactions.
- **Networks:** signed regulation, physical binding, directed signaling, metabolism — all traversable.
- **Functioning pipeline (per gene):** genome → input (regulators) → expressed → function (its reaction,
  or the genes it switches, or its complex) → output. Every link clickable.

---

## 10. THE ENGINES — what you can *do*
| engine | mechanism | output |
|---|---|---|
| **Remove / Mutate** | 2-hop cascade over reg+PPI (cell-type-gated) + M3 enrichment; DepMap/constraint lethality | damaged cell: affected proteins, disrupted processes/pathways, viable/inviable |
| **Overexpress** | propagate signed regulation | up/down targets, shifted processes |
| **Double knockout** | union cascades + shared targets + DepMap co-dependency r | synthetic-lethal flag (measured) |
| **Differentiation** | ON/OFF genes between two cell-type states + driver TFs | reprogramming plan |
| **Infect: HIV** | host-interaction map + dependency factors | hijacked machinery + weak points |
| **Trace / pipeline / pathway** | graph traversal | full mechanistic path, module views |

---

## 11. END PRODUCT
1. **`cell_complete.html`** — a self-contained ~10 MB interactive cell (open in any browser or serve at
   `localhost:8000/cell`). No backend, no dependencies.
2. **Reproducible notebooks** — `download_all_data.ipynb` (data → Drive), `build_complete_cell.ipynb`
   (full GPU build with all models), `serve_cell.py` (localhost).
3. **A validation report** (from the build) proving it recovers known biology.
4. **A ranked novelty table** — dark-gene functions, cancer-specific dependencies, novel SL pairs — the
   experiment shortlist.

---

## 12. QUESTIONS THE MODEL CAN ANSWER

**Importance & essentiality**
- "Is gene X essential — measured, and in what fraction of cancers?"
- "Which genes are essential in cancer but tolerated in the germline?" (novelty flags)

**Perturbation & therapy**
- "What breaks if I knock out X in a hepatocyte vs a neuron?"
- "What happens if I overexpress this TF?"
- "What is synthetic-lethal with gene X?" (measured co-dependency)
- "Which damaged nodes in this cascade are druggable, and by what drug?"

**Variants & disease**
- "This mutation — where does it sit, does it break the fold, what disease?"
- "This non-coding variant — which gene does its enhancer loop to?" (3D loops)
- "What rewires in disease Y?" (with GEO conditions, pending)

**Identity & differentiation**
- "What master TFs define this cell type?"
- "To convert cell A → B, which genes turn on/off and which TFs drive it?"

**Function discovery**
- "This unannotated (dark) gene — what's its likely function, and on what evidence?"
- "What proteins act together as a module?" (co-essentiality / complexes)

**Infection**
- "What does HIV hijack, and what are its host-dependency weak points (drug targets)?"

**Trust**
- "How confident is this prediction?" — every answer carries model-agreement confidence + sources.

---

## 13. VALIDATION — how we know it works
The build emits a report showing recovery of known biology:
- Essentiality set ≈ DepMap common-essential (POLR2A/RPL* essential; A2M dispensable).
- Co-essentiality recovers real modules — TSC1↔TSC2 (r=0.91), the SAGA complex, mTOR/GATOR.
- Cell-type masters recover textbook identities (HNF4A→liver, etc.).
- Drugs: EGFR→gefitinib/osimertinib/cetuximab; BRAF→vemurafenib.
- Perturbation: POLR2A KO → inviable; tissue-restricted genes → tissue-specific lethality.

## 14. NOVELTY — what new it produces
Novel, evidence-backed **hypotheses** (not proven results), concentrated in three places:
1. **Dark-gene function** (~5,000 genes) via embedding + co-expression + compartment + network.
2. **Model-disagreement dependencies** (1,172 flags) — context/cancer-specific biology the simple rules miss.
3. **Untested synthetic-lethal / regulatory combinations** from co-dependency and 3D loops.
Ranked by ensemble confidence → a validation shortlist of hundreds of leads.

## 15. HONEST LIMITATIONS
- **Not a kinetic simulator** — no rates, concentrations, flux, or time-courses (no clean bulk kcat;
  BRENDA gated).
- **Bias:** DepMap = cancer lines; Tabula Sapiens = healthy adult — context claims inherit these.
- **Predictions are hypotheses**, requiring experimental validation.
- **Correlation ≠ causation** for co-essentiality/co-expression — the confidence layer is the guardrail.
- **Pending:** GEO conditions, RNAcentral targets, full methylation tracks, lipids/ions, tissue/spatial.

## 16. ROADMAP
- **Now:** full A100 build → real ~40 cell types (M2) + Geneformer enrichment (M3) + validation report.
- **Next:** GEO/condition integration (disease states); real curated SL from DepMap; miRNA targets for RNAcentral.
- **Later:** a grounded **LLM agent** over the cell + models as tools — natural-language, multi-hop,
  cited answers with confidence. Kinetics only if an open bulk source appears.

## 17. WHO IT'S FOR
- **Pharma / biotech:** target triage, synthetic-lethal combinations, druggability in context, variant interpretation.
- **Researchers:** hypothesis generation for dark genes and context-specific dependencies.
- **Educators / communicators:** an explorable, mechanistic human cell.

The value proposition, stated plainly: **it turns 20 disconnected datasets and 3 models into one
interactive cell that answers importance/context/wiring/perturbation/drug/variant questions across cell
types — each with a source and a confidence — and hands you a ranked shortlist of testable, non-obvious
leads.** That is exactly what an integrative map should do, and it is honest about being a map, not a
simulator.
