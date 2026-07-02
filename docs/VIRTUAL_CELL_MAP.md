# Virtual cell — requirements map

Everything needed to build a complete virtual (human) cell, what supplies it, its status, and
how to achieve what's left. The cell has **two halves**:

- **Behavior** (expression, regulation, importance, state, response) → produced by the **3 models**.
- **Substrate** (physical structure, chemistry, genome) → comes from **external databases**.

A complete cell = substrate × behavior. The 3 models cover their entire half; everything still
open is a database or a different-class-of-model (kinetics) job.

**Legend:** ✅ built · 🟡 partial · ❌ not yet · **·run** = built & wired, needs the Colab notebook run to populate with real data.
**Source:** `M1` our integrated model · `M2` atlas/scGPT cell-type layer · `M3` Geneformer · `DB` external database.

---

## A. Physical scaffold — where things are
| component | status | source | how to achieve |
|---|---|---|---|
| Compartments / organelles + membranes | ✅ | DB (UniProt subcellular) | done |
| Spatial position per molecule | ✅ | built | deterministic layout by compartment |
| Cytoskeleton as structure | 🟡 | DB | region exists; needs filament model |
| Membrane as a real bilayer w/ embedded transporters | ❌ | DB + drawing | render lipid bilayer; place SLC/channel proteins in it (data already in compartments) |

## B. Molecular inventory — what's inside
| component | status | source | how to achieve |
|---|---|---|---|
| Genes + proteins | ✅ | M1 + DB | done (16,492 localized) |
| Abundance / copy-number | ✅·run | **M2** | atlas expression → per-gene abundance (run notebook) |
| Metabolites (in reactions) | ✅ | **DB (Human-GEM)** | done — 2,549 enzymes carry real reactions w/ metabolites |
| Metabolites as discrete nodes | 🟡 | DB (Human-GEM) | promote reaction species to clickable nodes (app redesign) |
| Macromolecular complexes as named machines | 🟡 | DB (CORUM) | fetch CORUM complexes → group subunits (ribosome, proteasome…) |
| Dark-gene function (guilt-by-association) | 🟡·run | **M3** | Geneformer embeddings → nearest functional neighbors (run notebook) |
| RNAs / lipids / ions as first-class entities | ❌ | DB (RNAcentral, SwissLipids) | add entity types + their interactions |

## C. Genome / epigenetics
| component | status | source | how to achieve |
|---|---|---|---|
| Chromosome + TSS locus | ✅ | DB | done (shown per gene) |
| Regulatory elements (CpG promoter, enhancer count) | ✅ | DB | done (shown per gene) |
| Methylation / chromatin state per gene | 🟡 | DB (ENCODE/Roadmap) | group-level stats exist; wire per-gene ATAC/methyl tracks |
| 3D genome / chromatin contacts (TADs) | ❌ | DB (Hi-C, 4DN) | overlay contact domains on loci |

## D. Central-dogma pipeline — gene → function
| component | status | source | how to achieve |
|---|---|---|---|
| Trafficking journey + functioning pipeline (with machinery) | ✅ | built | done |
| Which genes actually transcribed/translated in this state | ✅·run | **M2** | expression gates the pipeline (run notebook) |
| PTMs, protein turnover / half-life | ❌ | DB (PhosphoSitePlus, degradation) | add PTM sites + half-life per protein |

## E. Functional networks
| component | status | source | how to achieve |
|---|---|---|---|
| Signed gene-regulatory network (activate/repress) | ✅ | DB (CollecTRI/DoRothEA) | done |
| Protein–protein interactions | ✅ | DB (STRING physical) | done |
| Cell-type-specific wiring (neuron ≠ hepatocyte) | ✅·run | **M2 / scGPT** | prune network to expressed genes per type (run notebook) |
| Metabolic network (genome-scale) | ✅ | **DB (Human-GEM)** | done — 12,931 reactions, 2,549 enzymes |
| Signaling chains (receptor→kinase→TF) | 🟡 | DB (SIGNOR/Reactome reactions) | add directed signaling reactions |

## F. Processes running
| component | status | source | how to achieve |
|---|---|---|---|
| Every protein tagged by process | ✅ | built | done |
| Which processes on/off per cell type | 🟡·run | **M2** | aggregate expression by process (run notebook) |
| Uptake / secretion as membrane flows | 🟡 | DB | animate transporters moving species across the membrane |
| Cell cycle as a running program with phases | ❌ | DB (Cyclebase) | add phase-specific gene sets + a cycle view |

## G. State & identity — the models' home turf
| component | status | source | how to achieve |
|---|---|---|---|
| Cell type via master TFs | ✅·run | **M2** | atlas-derived masters (run notebook) |
| Cell-type-specific network | ✅·run | **M2** | done (see E) |
| Differentiation / reprogramming engine | ✅·run | **M2 + M3** | ON/OFF genes + driver TFs between states (run notebook) |
| Conditions (O₂/nutrients/stress) + per-state pathway activity | ❌ | DB (GEO/perturbation) | add condition profiles; recolor by pathway activity |

## H. Dynamics
| component | status | source | how to achieve |
|---|---|---|---|
| Discrete before→after state shift | 🟡 | **M3** | differentiation + perturbation give state transitions |
| True kinetics (concentrations/rates/flux/time) | ❌ | DB (BRENDA/SABIO-RK) + ODE | kcat/Km per enzyme → build an ODE/FBA layer |

## I. Perturbation & prediction — the interactive payoff
| component | status | source | how to achieve |
|---|---|---|---|
| Gene knockout → cascade → viable/inviable (cell-type-gated) | ✅ | M1 + graph | done |
| Overexpression / activation | ✅ | built (signed net) | done |
| Mutation → structure / fold / disease | ✅ | DB (UniProt/ClinVar/AlphaFold) | done (13 curated + genome-wide LOEUF) |
| Viral infection (HIV) → hijack + weak points | ✅ | DB (NCBI HIV) | done |
| Geneformer-enriched cascade | 🟡·run | **M3** | in-silico downstream genes (run notebook) |
| Drug → target → effect | 🟡 | DB (DrugBank/ChEMBL) | map drugs to targets, then reuse the perturbation engine |
| Double-KO / synthetic lethality (human) | ❌ | DB (SynLethDB / DepMap co-dep) | add SL pairs; two-gene cascade |

## J. Traceability & confidence
| component | status | source | how to achieve |
|---|---|---|---|
| Gene trace / functioning pipeline / pathway view | ✅ | built | done |
| Provenance (source of every label) | ✅ | built | done (per-gene sources footer) |
| Confidence from ensemble agreement | 🟡 | **M1+M2+M3** | essentiality confidence done; extend to M1/M2/M3 agreement per prediction |

## K. Multicellular context
| component | status | source | how to achieve |
|---|---|---|---|
| Cell–cell signaling / tissue neighborhood | ❌ | DB (CellPhoneDB) + spatial atlas | add ligand-receptor edges between cells |

---

## Summary map

**Covered by the 3 models** (the behavior half — run the notebook to populate):
- `M1` essentiality/importance, disease priority, KO-lethality prediction.
- `M2` abundance, which-genes-active-per-cell-type, cell-type-specific wiring, master TFs.
- `M2+M3` differentiation / reprogramming engine.
- `M3` in-silico perturbation enrichment, dark-gene function, overexpression direction.
- `M1+M2+M3` ensemble confidence (partial).

**Covered by external datasets** (the substrate half — already integrated):
- UniProt (compartments, localization), CollecTRI/DoRothEA (signed GRN), STRING (PPI),
  Reactome (pathways), **Human-GEM (genome-scale metabolism)**, gnomAD (constraint),
  Open Targets/ClinVar (disease), NCBI (HIV), AlphaFold/ESMFold (structure).

**Still open, and the dataset/method to get there:**
| gap | get it from |
|---|---|
| Membrane bilayer w/ transporters | rendering (data present) |
| Metabolites as discrete nodes | promote Human-GEM species (app work) |
| Named complexes | CORUM |
| RNAs / lipids / ions | RNAcentral, SwissLipids |
| Per-gene methylation/chromatin | ENCODE / Roadmap |
| 3D genome | Hi-C / 4DN |
| PTMs / turnover | PhosphoSitePlus, dbPTM |
| Signaling chains | SIGNOR / Reactome reactions |
| Cell cycle program | Cyclebase |
| Conditions / environment | GEO / perturbation atlases |
| True kinetics | BRENDA / SABIO-RK + ODE/FBA |
| Drug → effect | DrugBank / ChEMBL |
| Synthetic lethality (human) | SynLethDB / DepMap co-dependencies |
| Tissue / cell–cell | CellPhoneDB + spatial atlas |

**Bottom line:** the 3 models' half is complete; the substrate half is largely integrated
(UniProt, networks, Reactome, Human-GEM, structure, disease, HIV). What remains red is each a
specific *external database* or a *kinetic model* — the table above says exactly which.
