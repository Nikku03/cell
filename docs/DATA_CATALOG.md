# Data catalog — every dataset the models use

Two models, kept separate. **CELL** = single-cell map (`cell_complete.json`). **TISSUE** = multi-cell-type
communication map (`tissue_model.json`). A dataset's "used by" column says which.

## 1. Ground truth — essentiality (training labels)
| dataset | source | contains | powers |
|---|---|---|---|
| DepMap CRISPRGeneEffect | depmap.org (figshare) | gene knockout fitness effect across ~1,100 cancer cell lines | CELL: Model 1 truth label; co-essentiality; synthetic-lethal pairs; Model 4 features |
| CEGv2 / NEGv1 | Hart et al. | curated core-essential & non-essential reference gene sets | CELL: essentiality labels / sanity baseline |

## 2. Perturbation response
| dataset | source | contains | powers |
|---|---|---|---|
| Perturb-seq (genome-wide) | Replogle 2022 (figshare) | transcriptional signature of knocking out each of ~2,000 genes (K562) | CELL: **Model 4** (perturbation predictor); dark-gene function via measured functional neighbors |

## 3. Protein–protein interactions (physical)
| dataset | source | contains | powers |
|---|---|---|---|
| STRING (physical) | string-db.org | physical PPIs, confidence-scored | CELL: PPI network (perturbation propagation) |
| BioPlex 3.0 | bioplex.hms.harvard.edu | AP-MS measured interactome (293T) | CELL: adds measured interactions, shrinks isolated-gene fraction |
| OpenCell | opencell.czbiohub.org | proximity-labeling interactome | CELL: adds measured interactions |
| Complex Portal | EBI IntAct | named, curated protein complexes | CELL: complex membership; "complex partners lost" on knockout |

## 4. Regulatory & signaling networks (directed)
| dataset | source | contains | powers |
|---|---|---|---|
| CollecTRI / TRRUST / DoRothEA | Saez-Rodriguez lab | signed TF → target-gene regulation | CELL: regulatory cascade; TF/master identification |
| SIGNOR | signor.uniroma2.it | causal signaling interactions (activation/inhibition) | CELL: signaling propagation; TISSUE: receptor→downstream response |

## 5. Localization & genome regulation
| dataset | source | contains | powers |
|---|---|---|---|
| gene_compartment (UniProt/HPA) | UniProt + HPA | subcellular compartment per protein | CELL: compartment placement; Model 4 features |
| gnomAD constraint | gnomad.broadinstitute.org | LOEUF — intolerance to loss-of-function mutation | CELL: safety window / essentiality feature |
| ENCODE cCREs | ENCODE/SCREEN | candidate enhancers | CELL: "how heavily regulated" feature |
| UCSC refGene + cpgIslandExt | UCSC | gene models, CpG-island promoters | CELL: promoter type, genomic coordinates |
| HiCCUPS loops (GSE63525) | Rao 2014 (GEO) | 3D chromatin loops (GM12878) | CELL: 3D genome layer |

## 6. Metabolism
| dataset | source | contains | powers |
|---|---|---|---|
| Human-GEM | SysBio Chalmers | genome-scale metabolic model (~13k reactions) | CELL: reactions catalyzed; metabolic routes / alternative pathways |

## 7. Expression / cell types
| dataset | source | contains | powers |
|---|---|---|---|
| CELLxGENE census (Tabula Sapiens) | chanzuckerberg | per-cell-type single-cell expression (human atlas) | CELL: **Model 2** data-driven master TFs & cell-type active networks |
| Geneformer (pretrained) | ctheodoris/Geneformer (HF) | foundation-model gene embeddings | CELL: **Model 3** — *validated as ~random for target recovery; NOT fed into the cascade* |
| HPA single-cell | proteinatlas.org | expression across 154 cell types | TISSUE: which cell type expresses each ligand/receptor |
| CCLE expression | DepMap/Broad | cancer cell-line expression | CELL: biomarkers of drug sensitivity |

## 8. Pathways, drugs, disease
| dataset | source | contains | powers |
|---|---|---|---|
| Reactome | reactome.org | curated pathway membership | CELL: pathway layer; guilt-by-association votes |
| DGIdb | dgidb.org | drug–gene interactions | CELL: druggability; TISSUE: druggable communication channels |
| Open Targets | opentargets.org | target–disease associations | CELL: disease expansion / target intelligence |
| HPO genes_to_phenotype | Human Phenotype Ontology | gene → disease/phenotype links | CELL: disease-link count; dark-gene definition |
| CCLE mutations | DepMap/Broad | damaging mutations per cell line | CELL: mutation biomarkers |

## 9. PTMs, structure, literature
| dataset | source | contains | powers |
|---|---|---|---|
| UniProt (PTM + acc→symbol) | uniprot.org | post-translational modification sites | CELL: PTM layer; ID mapping |
| AlphaFold-derived (structure/fold) | precomputed | structure/fold-effect of mutations | CELL: mutation-impact + fold layers |
| gene2pubmed | NCBI | publication count per gene | CELL: literature coverage → Target-Intelligence white-space |

## 10. Cell–cell communication (tissue model)
| dataset | source | contains | powers |
|---|---|---|---|
| Omnipath ligand–receptor | omnipathdb.org | ligand→receptor pairs (signed) | TISSUE: cell-cell communication edges |
| CellChat (via Omnipath) | Jin 2021 | signaling pathway families | TISSUE: pathway-family coloring of communication |
| CellPhoneDB | ventolab | secreted vs transmembrane annotation | TISSUE: paracrine / juxtacrine / autocrine mode |

## 11. Virus / infection
| dataset | source | contains | powers |
|---|---|---|---|
| NCBI GeneRIF hiv_interactions | NCBI | HIV protein → human host-gene interactions | CELL: HIV hijack map & host-dependency weak points |

## ID mapping (plumbing, not a layer)
gene_info, genes.tsv, string_aliases — Entrez/Ensembl/symbol cross-mapping so every source lands on the same gene.

---
**Honest note on provenance:** every dataset above is public. The model's value is not the raw data
(anyone can download it) but the integration + the derived, validated inferences on top (co-essentiality,
SL, dark-gene function, Model 4). See `docs/NOVELTY_STRESS_TEST.md`.
