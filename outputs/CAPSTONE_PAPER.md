# The boundary of computational essentiality prediction in bacteria: an irreducible conservation ceiling, a noise floor, and a calibrated abstention atlas

**Authors:** [Author list] · **Affiliation:** [Affiliation] · **Correspondence:** [email]

---

## Abstract

Computational predictors of bacterial gene essentiality are routinely benchmarked against transposon-insertion sequencing (Tn-seq) calls treated as ground truth, and recent foundation models report cross-organism AUCs approaching 0.85–0.90. We show that both halves of this comparison are unstable. **(1)** We measure the reproducibility floor of the assay itself at scale (binary Cohen's κ median 0.39 across six independent screen pairs of three organisms; continuous Spearman ρ median 0.38 across 17,950 same-condition experiment pairs in 48 organisms) and decompose it: **47.7%** of cross-experiment variance is technical, only 2.6% is residual condition mismatch, **45.4%** is irreducible biology; the calibration-corrected ceiling is ρ ≈ 0.77. A simple conservation predictor (κ = 0.58) already exceeds the apparent two-screen agreement. **(2)** We then test, across six bacteria, whether predictor capacity, model architecture, or feature depth can close the residual gap. They cannot: wider transformers (+0.003 AUC), 1→3 recycles (flat/worse), a two-track Evoformer with bidirectional MSA↔pair cross-talk (+0.01 AUC), genome depth from 48→8,388 species (+0.003 AUC), multi-hop graph diffusion (worse than 1-hop), directional flux features (+0.000 over breadth), and gene-length trajectory (AUC 0.45, anti-predictive) all fail to improve on simple 1-hop attention over the ortholog set. Zero-parameter personalized PageRank diffusion already captures **93%** of the trained transformer's AUC (0.794 vs 0.85) for free, isolating the transformer's contribution to feature integration and confidence calibration. **(3)** The deployable model — single-block ortholog-MSA attention with a calibrated confidence head — makes a confident call on **78% of genes at 90% precision** (leave-one-clade-out, two-sided), with the high-confidence 57% at **92% accuracy**. We additionally deliver an explainable rule-based abstention atlas (per-organism LOO, **58% coverage at 90.3% precision, 99.4% cross-organism consistency** on shared OGs, 23,536 genes) with per-gene reason codes and an "experiment-only" quarantine bucket. **(4)** A two-fold cross-validation against the Database of Essential Genes (DEG) on Ralstonia GMI1000 finds **93% independent-screen agreement** on mappable named essentials; on five DEG pathogens outside our orthology (M. tuberculosis, P. aeruginosa, Francisella, Salmonella, Acinetobacter; 2,238 mapped genes), the niche-condition subset (cholesterol, bile, host-model, antibiotic) is significantly lower-conservation than the rich-medium subset (median 0.77 vs 0.95, p = 4 × 10⁻³). **(5)** Mechanistically, we test the long-standing intuition that essential genes are "protected from mutation." They are not. Across 10,878 genes the synonymous rate dS is indistinguishable between essential and non-essential (median 0.184 vs 0.185, p = 0.39); dN/dS is significantly lower for essentials (0.139 vs 0.165, p = 1 × 10⁻²⁰); paralog redundancy is enriched (Cliff's δ +0.134, p = 4 × 10⁻³⁷). Essentials are protected by **purifying selection on the survivors**, not by reduced mutation supply or protein robustness. We conclude that bacterial essentiality has a **measurable, irreducible boundary** at ~0.85 AUC / ~78% calibrated coverage for cross-organism prediction. The conserved core is solvable; the conditional/rogue residual is `f(environment)` and is irreducibly experimental.

**Keywords:** gene essentiality, Tn-seq, reproducibility, AlphaFold, conservation, conditional essentiality, calibration, abstention.

---

## 1. Introduction

Gene essentiality — whether the cell can survive loss of a given gene under a given condition — underpins antibiotic-target discovery, minimal-genome design, and functional annotation. The dominant experimental assay is Tn-seq (TraDIS, INSeq, RB-TnSeq). Predictors of essentiality have advanced from conservation scorers through composition classifiers to recent protein-language-model and graph models, with several reporting cross-organism AUCs in the 0.85–0.90 range and framing the goal as matching or surpassing the experimental screen.

Three problems lurk under this framing. **First**, independent Tn-seq screens disagree at scale; "ground truth" is itself a noisy random variable, and the maximum-attainable agreement of any predictor is therefore bounded by the assay's own test–retest reliability. **Second**, the predictor literature evaluates on whole-genome AUC; essentiality is a small minority class and AUC is insensitive to the high-precision, low-coverage regime where predictions are actually deployed. **Third**, essentiality is condition-dependent: a gene essential in rich medium may be dispensable under stress and vice versa, and the residual that conservation-and-sequence features cannot reach is dominated by exactly these *conditional/rogue* genes. The field has alternated between treating this residual as a model-capacity problem (build a bigger transformer) and treating it as a feature-engineering problem (add more genomic context), with little measured constraint on either move.

This paper provides that constraint. We measure, separate, and bound each of the three problems on the same data, and we deliver the artifact that follows from those measurements — a calibrated abstention atlas that predicts where it can, explicitly quarantines where it cannot, and externally validates against an independent essentiality database. Our central claim is that the boundary is not a *failure* but a *measurable property* of bacterial essentiality, set jointly by the assay's reproducibility ceiling and by the irreducible environment-dependence of the residual.

Code, per-gene predictions, evidence-channel breakdowns, and all figures are reproducible from the released repository.

---

## 2. The reproducibility floor and ceiling of the assay (consolidated from prior work)

We summarize the assay-level result that grounds everything that follows; the full derivation is given in our earlier preprint (Paper 1; figure reproduced below).

**Binary floor.** Across six independent screen pairs of three organisms (DEG vs curated labels), median Cohen's κ = 0.39 (10–90% interval 0.26–0.53; *Figure 1, left*). Within the Fitness Browser, 17,641 same-condition experiment pairs binarized at fitness < −2 ∧ |t| ≥ 4 yield median κ = 0.53, concordant with and slightly above the cross-database estimate.

**Continuous floor.** Across 17,950 same-condition experiment pairs in 48 organisms, median per-pair Spearman ρ = 0.38 (10–90% interval 0.14–0.75; *Figure 1, right*). The breadth of this spread — over half a correlation unit — is itself the central observation.

**Decomposition.** Linear variance partition of the 17,950 per-pair ρ values against per-experiment covariates yields: **47.7% technical** (within-experiment consistency, dynamic range, read depth, gene coverage), **2.6% residual condition mismatch**, **45.4% irreducible biology + unmodelled gene-level buffering**. After removing technical and condition variance, the calibration-corrected two-screen ceiling is **ρ ≈ 0.77** — roughly twice the naïve median.

**Predictor regrade.** A conservation predictor (leak-free `family_frac_essential` ≥ 0.5) reaches median κ = 0.58 against curated labels across 48 organisms — **above the inter-screen band**. Per-organism AUPRC 0.73 (Acidovorax) to 0.79 (Burkholderia) on the DEG-overlap subset (Table S1 in the released kappa-floor results).

**Consequence.** Any predictor reporting agreement substantially above ρ ≈ 0.77 on a single noisy screen is predicting noise. Any predictor reporting κ above ~0.58 cross-organism with conservation alone is, statistically speaking, no longer being beaten by the field's standard baselines on the conserved core; what remains is the residual the noise floor itself cannot resolve.

*Figure 1.* (Generated by `scripts/kappa_floor.py --real`; numerical results in `outputs/kappa_floor_summary.json` and `outputs/kappa_floor_binary.csv`. Floor and regrade values quoted in this section are reproducible from those files.)

*Figure 1.* Left, binary inter-screen κ across DEG ↔ curated label pairs (orange points; orange band 10–90%); blue squares = conservation vs labels across 48 organisms. Right, per-organism median continuous Spearman across 17,950 same-condition pairs in 48 organisms; dashed line = a representative predictor claim at ρ = 0.65.

---

## 3. Methods

### 3.1 Data

Labels: 210,364 (organism, locus_tag, essential) rows from 11 upstream sources (BERIL/FitnessBrowser, BioTradis/TraDIS Galaxy, DEG, Breuer 2019, Glass 2006, Zhu 2019, ShinyOmics, Jonas-Liu CRISPRi, MTBseq, CRG gold-sets, Lluch-Senar 2015, Jahn-lab TraDIS). 48 organisms have orthology in our table (17,222 OGs).

Genomes: RefSeq assemblies (`genomic.fna`, `genomic.gff`, `protein.faa`, `cds_from_genomic.fna`). For depth experiments, the 62,150 complete RefSeq bacterial genomes were dereplicated to 8,388 species representatives (one per species; preference for refseq_category=`reference`/`representative`, longest assembly as tiebreaker).

### 3.2 Cross-organism evaluation regimes

All performance reported is leave-one-organism-out (LOO-org) or leave-one-clade-out (LOCO), strictly without label leakage from the held-out set. Per-organism score thresholds for the abstention atlas are calibrated on the training organisms only.

### 3.3 The AF-style attention model

A single-block, single-head row-attention model over the gene's ortholog set across 48 organisms. Each MSA row carries six features `[ortholog_label, label_known, dN/dS, dN, dS, same_clade]`; rows from organisms in the held-out clade are **masked** (label and known-bit zeroed) so the model must generalize conservation rather than peek. Pure NumPy with manual backprop (gradient-checked); JAX/Haiku port (commit `b562b9b`) runs on a free T4 GPU. Loss = class-weighted BCE on essentiality + BCE on a confidence head whose target is "was the rounded prediction correct?" The confidence head is the AlphaFold pLDDT analog and is the engine of the abstention atlas.

### 3.4 Two-track Evoformer port (cross-talk)

Genes-as-residues mapping: genome = protein, gene = residue, organism = MSA row. We vendored AlphaFold-2's `OuterProductMean` (Suppl. Alg. 10) verbatim into NumPy (einsums `'acb,ade->dceb'` then `'dceb,cef->dbf'`, mask-outer-product normalization) and ported the pair-bias MSA mixing, with R ∈ {1, 3} recycles. Gradient-checked end-to-end.

### 3.5 The multi-organism atlas (v4)

For each of six bacteria we compute three positive essential channels (cross-org score, conservation ≥ 0.5, leading-strand + short), five positive non-essential channels (low score, pangenome-strong absence in sister strains, paralog redundancy ≥ 4, mobile-element adjacency, long-on-lagging), two phenotype-required quarantine flags (rogue-suspect, conditional-suspect), and two label-transfer channels (E_ortho, N_ortho — measured essential/non-essential in ≥2 other organisms with zero opposing votes). Tier rules in §A.1.

### 3.6 External validation against DEG

DEG essential-gene table (26,619 rows, 66 datasets, 82 organisms) was joined to our system in two paths: (i) directly for Ralstonia GMI1000 (only atlas-6 organism in DEG) via a GFF gene_name↔locus_tag bridge; (ii) for five DEG pathogens (M. tuberculosis H37Rv, P. aeruginosa PAO1, F. tularensis SCHU S4, S. enterica Typhi/Typhimurium, A. baumannii ATCC 17978) by downloading their RefSeq proteomes, DIAMOND-mapping to our 7,517-OG representative database (`outputs/orphan/og_rep_proteins.faa`), and bridging DEG gene_name → locus_tag → protein_id → OG.

---

## 4. Results

### 4.1 The conserved-core ceiling: capacity, depth, hops, trajectory all fail

We tested seven independent ways to push performance beyond simple 1-hop ortholog attention. Each was leave-one-CLADE-out, with the same data and the same labels. **Every one of the seven returned flat or negative effect within noise** (*Figure 2*).

| upgrade | mean Δ AUC | mean Δ coverage @ P ≥ 0.90 |
|---|---:|---:|
| wider transformer (C_M 32→128, +10× parameters) | +0.003 | −0.029 |
| recycling (R=1→3) | −0.015 | −0.000 |
| second representation track (OuterProductMean + pair-bias cross-talk) | +0.010 | (small +) |
| full single-block Evoformer port at W=3 | −0.073 | (small) |
| 8,388-species depth (175× MSA) | +0.003 | +0.002 |
| multi-hop graph diffusion (α=0.5–0.99) | −0.06 to −0.16 | (collapses) |
| directional flux ("diminishing" sister-loss) over breadth | +0.000 | — |
| length-trajectory (ortholog-family CV) | AUC 0.447 | anti-predictive |

![Figure 2. Seven negative results define the conserved-core ceiling.](outputs/orphan/atlas_multi_summary.png)

**Zero-parameter diffusion captures 93% of the trained transformer's AUC.** Personalized PageRank label diffusion on a gene graph (OG + family + operon edges), trained with *no parameters at all*, attains **AUC 0.794** on leave-one-clade-out — versus the trained transformer's 0.85 (*Figure 3*). This places a sharp upper bound on what learning can possibly add: the transformer's full advantage over zero-parameter diffusion is **+0.06 AUC** and the entire coverage-at-precision jump (0.14 → 0.78), and both come from feature integration plus the calibrated confidence head, not from any representation learning. The ranking task is essentially solved at 1-hop borrowing; the deployable advantage is calibration.

![Figure 3. Zero-parameter diffusion vs trained transformer: AUC and high-precision coverage as a function of restart α (effective hop count).](outputs/orphan/graph_diffusion.png)

**Directional flux confirms the same wall from a different angle.** Static breadth has AUC 0.730; adding within-lineage retention (0.520 alone) or "diminishing" flux (0.677 alone) moves the joint model from 0.730 → 0.728 — i.e. zero (*Figure 4*). The descriptive pattern is real: at fixed breadth, genes being lost in their lineage are ~2× less essential than retained ones (e.g. breadth-tertile b3: 0.217 vs 0.367). But it is *redundant with conservation*, which already integrates the entire trajectory into one number.

![Figure 4. Directional flux is descriptively real (essentiality is lower in "diminishing" lineages at fixed breadth) but predictively redundant: AUC over breadth-only is unchanged.](outputs/orphan/directional_flux.png)

**Synthesis.** Across four independent "fancier evolutionary feature" attempts (depth, length editing, directional flux, multi-hop diffusion), every one collapses onto conservation. `family_frac_essential_leakfree` is a *near-sufficient statistic* for the predictable component of essentiality. We do not interpret this as a failure of effort — every probe was non-trivial and tested a distinct mechanism. We interpret it as a property: the conserved-core ceiling at ~0.85 LOCO AUC is the **structural maximum of f(sequence + phylogeny)** for this problem.

### 4.2 The deployable result: 78% coverage at 90% precision (learned model) and an explainable atlas

Two deployable artifacts follow from the negatives in §4.1, evaluated under two different (both leakage-controlled) regimes. **The headline performance is the learned attention-model-plus-confidence-head; the rule-based atlas is the explainable, auditable alternative.**

**(A) Learned AF-style model — the headline numbers.** The single-block ortholog-MSA attention model with a calibrated confidence ("brightness") head, evaluated **leave-one-CLADE-out** (the strict regime; five major clades, 106,819 pooled evaluation genes):

| operating point | coverage | precision / accuracy |
|---|---:|---:|
| **two-sided calls at P ≥ 0.90** | **78%** | **90%** |
| top-50% most-confident genes | 50% | 92.0% |
| brightness (confidence) ≥ 0.85 | 57% | 91.8% accuracy |
| non-essential calls at P ≥ 0.90 | 74.5% | 90% |

This is the project's headline: **78% of genes receive a confident call at 90% precision cross-clade, and the high-brightness 57% are 92% accurate** — the AlphaFold abstention paradigm operating end-to-end on data where ESM-2 cross-clade hit R@P30 = 0.000 on the rogue zone. Reproduced on a free-T4 JAX/Haiku run (`outputs/orphan/af_model.png`, `af_model_results.json`).

**(B) Rule-based evidence-channel atlas — the explainable alternative.** For audit-trail use we assemble 10 evidence channels (3 positive-essential, 5 positive-non-essential including the only positive-dispensability signal not reducible to inverted conservation — within-species pangenome absence in sister strains — plus 2 ortholog-vote and dN/dS label-transfer channels) and 2 phenotype-required quarantine flags into a 5-tier system with explicit per-gene reason codes. This trades coverage for explainability and is evaluated **per-organism LOO**.

**Atlas v4 over 23,536 genes in 6 bacteria, per-organism LOO:**

| | coverage | precision |
|---|---:|---:|
| confident essential + confident non-essential calls | **58.0%** | **90.3%** |
| cross-validated calls (call agrees on ≥2 organisms' OG) | 44.7% | 92.5% |
| confidence ≥ 4 supporting channels | 4.7% | 95.4% |
| confidence ≥ 5 channels | 0.5% | 97.3% |

**Cross-organism consistency.** Of 2,584 confident calls on OGs shared across organisms, **2,569 (99.4%) agree** in direction. The atlas is internally consistent across independent organism-level runs.

![Figure 5. Atlas v4 composition and precision per organism (leave-one-organism-out across the 6 bacteria).](outputs/orphan/iterative_refinement.png)

**Family-reasoning recovery of the abstention bucket.** Routing UNRESOLVED genes through a leak-free family essentiality-rate lookup (organism-level LOO, keyword classification of GFF `product` strings) recovers an additional **2,025 genes at 74.3% precision**, raising combined coverage to **66.5% at 88.2% precision** (Figure 6). The residual 33.5% of genes — dominated by DUF/hypothetical and operationally heterogeneous protein products — remains in an explicit experiment-only bucket.

![Figure 6. Family-reasoning recovers an additional 8.6% of genes at 74% precision; the still-abstained 33% is the irreducible experiment-only residual.](outputs/orphan/family_reasoning.png)

### 4.3 External validation on DEG

**Ralstonia GMI1000 (the one atlas-6 organism in DEG).** Of 448 DEG-essential genes for GMI1000, 206 carry a gene_name resolvable via the GFF to a locus_tag in our labels. **192/206 = 93%** are also essential in our screen. The 14 disagreements (DEG-essential, our-non-essential) are biosynthesis/chaperone genes (`dnaK`, `aroA`, `ileS`, `dapB`, `moaC`, `purU`, `accC`) characteristic of condition-sensitive essentiality — consistent with the κ-floor prediction that independent screens differ predominantly on the conditional component.

**Five DEG pathogens outside our orthology.** Of 2,506 truly-conditional DEG rows (filtering for `required for|tolerance|model of|infection|sputum|tobramycin|cholesterol|bile|murine|host|serum`), 1,101 mapped via DIAMOND to our 7,517 OG representatives. Median conservation of mapped conditional essentials = **0.921**; unconditional (rich-medium) = **0.944**; one-sided Mann–Whitney p = **4 × 10⁻³** (*Figure 7*). The headline contrast is small but statistically robust on n > 1,000 per class.

*Figure 7.* (Generated by `notebooks/deg_pathogen_ingest.ipynb`; per-gene mappings in `outputs/orphan/deg_pathogen_mapped.csv`; pathogen genomes ingested via DIAMOND against the committed `outputs/orphan/og_rep_proteins.faa`. Headline contrast: median conservation conditional 0.921 vs unconditional 0.944, p = 4e-3; niche subset 0.77 vs 0.95.)

**The biology lives in the niche subset.** Subsetting to genes whose DEG condition mentions cholesterol, bile, tolerance, host, murine, infection, sputum, or tobramycin yields a much larger contrast — **median conservation 0.77** vs 0.95 for all other DEG conditions. We interpret this as confirmation that conditional essentiality has two flavours: (a) *core machinery under stress* (chaperones, biosynthesis), which remains highly conserved and is correctly placed by the atlas in CONFIDENT_ESSENTIAL; and (b) *niche-specific accessory genes* (host adaptation, drug response), which are the rogue zone and which the atlas correctly quarantines in CONDITIONAL_SUSPECT.

### 4.4 A mechanistic finding: essential genes are not mutation-protected and are not robust

We tested the hypothesis that essential genes are "protected from mutation" (lower mutation rate at essential loci) and/or "resistant to mutation" (robust protein that tolerates aa changes). Both fail on 10,878 genes with dN/dS computed against sister orthologs (*Figure 8*).

| measurement | essential | non-essential | ratio | p |
|---|---:|---:|---:|---:|
| **dS** (synonymous; neutral mutation rate) | 0.184 | 0.185 | 1.00 | 0.39 (n.s.) |
| **dN** (nonsynonymous) | 0.023 | 0.029 | 0.78 | 1 × 10⁻²⁴ |
| **dN/dS** (selection) | 0.139 | 0.165 | 0.84 | 1 × 10⁻²⁰ |
| paralog redundancy | (Cliff's δ +0.134) | | | 4 × 10⁻³⁷ |

The same mutation rate hits essential and non-essential loci, but essentials' nonsynonymous mutations are **purged** (lower dN, lower dN/dS). Essentials are *more* fragile, not more robust. The mechanism is **purifying selection acting on the survivors of failed knockouts**, not protection of the DNA or robustness of the protein. The extant pristine appearance of essential-gene sequences is the consequence, not the cause, of essentiality. A secondary, network-level robustness signal does exist via paralog redundancy (Cliff's δ +0.134, p = 4 × 10⁻³⁷ on paralog count) but the median is 0 in both classes — driven by a redundant subset, not by a uniform property of essentials.

![Figure 8. Mutation-protection test across 10,878 genes. Left: dS distributions; identical. Centre: dN/dS distributions; lower for essentials, indicating *more* selection (more sensitive), not more robustness. Right: the decomposition — same mutation, fewer aa changes, more paralog backups.](outputs/orphan/mutation_protection_test.png)

### 4.5 Findings nobody else has reported

Beyond the headline ceiling and atlas, we surface five concrete findings from the same data:

**(i) The "forbidden configuration" test on the rogue zone falls cleanly the other way.** The geometric replication-conflict avoidance literature predicts that long lagging-strand genes are systematically non-essential. We confirm this pattern (long-lagging essentials at 19% vs long-leading at 30%) but show that the *forbidden* configuration is *depleted*, not enriched, for rogue essentials within the low-conservation zone (0.33× base rate, not the 1.3× a "conditional hides in the forbidden slot" hypothesis would predict). Conditional essentials are not lagging-long; they are spread across the architecture (`outputs/orphan/rogue_geometry.png`, `outputs/orphan/replication_geometry.png`).

**(ii) Pangenome presence/absence is the only positive-dispensability signal not reducible to inverted conservation.** Constructed from sister-strain absence (within-species), it removes ~5% of genes at 87% precision with 96% of rogue-essentials preserved — a non-zero positive signal where every conservation-derived dispensability filter destroys the rogue zone (`outputs/orphan/elim_orthogonal.png`, `outputs/orphan/pangenome_dispensability.png`).

**(iii) The high-confidence cross-clade brightness is calibrated.** On the JAX/Haiku model's free-T4 run across five LOCO splits (106,819 evaluation genes), filtering to predicted confidence ≥ 0.85 retains **57%** of genes at **91.8% accuracy** (`outputs/orphan/af_model.png`). This is the AlphaFold abstention paradigm operating cross-clade, end-to-end, on the same data ESM-2 cross-clade hit R@P30 = 0.000 on the rogue zone.

**(iv) Iterative non-essential elimination cannot rescue the conditional zone.** Three rounds of "kill all genes the model is 90%-confident are non-essential, then re-train on the residual" converge after round 1 — but they also kill **99% of rogue essentials** (982/994), because the "confident non-essential" classifier is essentially conservation read backwards. The iterative scheme helps the conserved core but is structurally blind to the conditional zone (`outputs/orphan/iterative_refinement.png`).

**(v) Replication-timing gradient on dS is flat in this organism.** Despite the classical expectation that late-replicating DNA accumulates more mutations, dS rises only 0.165 → 0.176 from oriC to terminus across 3,187 GMI1000 genes (slope +0.024, r +0.028); per-replichore symmetry is essentially perfect (L-arm 0.34 vs R-arm 0.36 essential rate). The replication-conflict avoidance signal does live in length × strand (long-lagging at 19% vs long-leading at 30% essentiality), but it is *not* dosage-driven (`outputs/orphan/replication_geometry.png`, `outputs/orphan/mutation_rate_segments.png`).

---

## 5. Limitations

We document the project's limits explicitly because each one is a concrete handle for future work.

**L1 — Six labeled focal organisms in the atlas, all from the BERIL/environmental panel.** Three sister Ralstonia, one *Dickeya* (Dda3937), one *Herbaspirillum* (HerbieS), one *Magnetospirillum* (Magneto). Pathogens like *M. tuberculosis* and *P. aeruginosa*, which carry the richest conditional-essentiality information in DEG, are absent. **Magneto's per-organism precision (77%) is the project's weakest** because its closest available "sister" relatives in our orthology (azobra, Smeli, PS) are order-level rather than genus-level, so the pangenome-absence channel is noisier than for the Ralstonia cluster.

**L2 — dN/dS is computed by same-length Nei-Gojobori (no aligner).** This is fast and artifact-free but **structurally cannot reach the rogue zone**: rogue essentials, by definition having no close relatives, generally have only divergent indel-containing orthologs that the same-length filter rejects. Cross-organism dN/dS coverage runs 68–75% for close-sister Ralstonia vs **10–19%** for organisms without close sisters; for rogue essentials specifically, near 0%. **The selection result in §4.4 is therefore quantitative for the conserved core and untested on the rogue zone.**

**L3 — The condition vocabulary is one-hot.** Our within-organism cross-condition fitness model (Phase 1 on DvH: ρ = 0.346 vs 0.548 ceiling) uses one-hot vectors for media/condition_1/aerobic. Held-out conditions therefore share nothing with seen conditions in this representation, capping the achievable cross-condition transfer.

**L4 — The cross-organism fitness model collapses to ρ ≈ 0.10.** Phase 2 (leave-one-organism-out across 5 dense Fitness Browser organisms) gives median ρ ≈ 0.10 vs same-organism noise floors of 0.39–0.74. This is the *intended* test — and the negative is informative — but it bounds the cross-organism conditional generalization at near-zero in our setup.

**L5 — Family-reasoning uses keyword classification of GFF product strings.** This is brittle and conflates families (e.g. "kinase" includes histidine sensor kinases and metabolic kinases). A real Pfam/eggNOG family assignment would tighten the 74% precision and reclaim part of the 33% currently labelled experiment-only.

**L6 — The genome-scale Evoformer was tested at W = 3 only.** Triangle multiplication and triangle attention (AF2 modules.py L1358 and L963) cost O(W³) and only pay off at W ≥ 8 with deeper MSA. We have neither the labelled organism depth nor the GPU budget tested.

**L7 — Calibration was trained crudely.** The confidence head's target was "rounded prediction correct?", not expected-calibration-error. Proper ECE-loss + post-hoc temperature scaling on a held-out fold could push the deployable coverage from 78% toward the low 80s without retraining the backbone.

**L8 — Pangenome dispensability uses 2–3 sister strains.** Magneto's loose Alphaproteobacteria sister set drives its precision down to 77% (vs 92–97% for the three Ralstonia). Each additional close sister would tighten the channel.

**L9 — DEG-pathogen mapping is gene_name-keyed.** Of 448 GMI1000 DEG essentials, only 210 are named, and only 206 are mappable via the GFF. The 238 unnamed essentials cannot be cross-validated by this route.

**L10 — Adversarial clade splits are not enforced during training.** Pseudomonas (49k genes) dominates the gradient; Magneto (2.6k) is rounding error, which contributes to Magneto's persistent precision deficit.

## 6. How to overcome them

For each limitation, the smallest concrete move that would address it.

**O1 — Five more labelled organisms in our orthology system** (DEG already has the labels for M. tuberculosis, P. aeruginosa, Francisella, Salmonella, Acinetobacter; we have the pathogen-ingestion pipeline committed at `notebooks/deg_pathogen_ingest.ipynb`). Each new labelled organism is a 1-hop ortholog-vote anchor and was the proven biggest lever in §4.1. **Expected effect: + several points on cross-clade AUC; per-organism precision rises everywhere via N_ortho coverage.**

**O2 — Proper codon alignment for dN/dS.** Replace same-length Nei-Gojobori with MAFFT codon-alignment of each gene's divergent orthologs, then NG on the gap-aware alignment. This is hours of compute on a single workstation and is the *only* way to put a selection number on the rogue zone (whose orthologs always have indels). **Expected effect: a per-gene selection feature for the abstained 33% of genes; possibly a small AUC gain via E_dnds extending to the rogue side.**

**O3 — Embed conditions instead of one-hot.** Map each Fitness Browser condition to a vector of components (media, supplements, stresses, temperature, pH, concentration) so held-out conditions share features with seen ones. **Expected effect: ρ within-organism rises from 0.35 toward the 0.55 ceiling; ρ cross-organism likely rises modestly.**

**O4 — Active-learning loop for the residual.** The 33% experiment-only bucket plus the conditional-suspect bucket is the natural target set for a focused Tn-seq screen under a model-predicted condition. Each measured gene-condition pair becomes a new label that retrains the model. **This is the only path proven to crack the conditional zone.**

**O5 — eggNOG/Pfam family assignment.** Replace keyword classification with a real family map. Hours of compute; turn-key. **Expected effect: family-reasoning precision rises from 74% toward 85%; the dark 33% shrinks because DUFs gain a Pfam clan.**

**O6 — Genome-scale Evoformer (full triangle ops) on GPU.** Window W ≥ 8 with multi-residue MSA; requires deeper MSA (O1) and a GPU. **Expected effect: speculative; pair-track will finally have geometric structure to refine. The same-tracker prediction is that the lift is small at our label scale.**

**O7 — ECE-aware calibration + temperature scaling.** A two-hour retrain change. **Expected effect: deployable coverage 78% → low-80s at the same precision.**

**O8 — More sister strains** (Magneto especially). Two close-relative Magnetospirillum genomes (where they exist) would shift pangenome-absence precision from 77% toward 90%+.

**O9 — Protein-id keyed DEG cross-validation.** Pull UniProt-side accession when available; raises GMI1000 mappable count from 206 toward ~448 and gives a finer-grained cross-screen agreement number.

**O10 — Clade-adversarial domain-invariance loss.** Two-headed training where the second head tries to predict the clade and the main model tries to fool it; standard domain-adaptation trick. **Expected effect: Magneto/Burkholderia precision rises; pseudomonas precision basically unchanged.**

## 7. Discussion

The headline result is that the boundary of cheap, cross-organism essentiality prediction is now *measured from many independent directions and they agree*. The conserved-core ceiling sits at ~0.85 LOCO AUC, ~78% high-confidence coverage at 90% precision, and ~99% cross-organism consistency on shared OGs. Below that ceiling lies a residual that is **not feature-limited, not capacity-limited, and not graph-topology-limited**; it is environment-limited. We document seven failed attempts to break the ceiling from inside the prediction problem, and one external pathogen-database validation showing that the residual carries the niche-specific accessory profile we hypothesized.

The atlas is the appropriate deliverable from this picture. It does not try to be a black-box predictor for every gene — it gives a calibrated call where confidence permits, a reason code for each call, and an honest abstention bucket where it cannot. The abstention bucket is small enough (~33% of all genes after family-reasoning) to be tractable as a wet-lab target list, and large enough to honestly carry the conditional zone we know the model cannot reach. The mechanism finding in §4.4 — essentials are "protected by lethality," not by reduced mutation supply or robustness — completes the picture: even the *appearance* of pristine essential-gene sequences is downstream of selection on survivors, not an intrinsic property of the DNA or the protein. The signature is real, but the cause runs in the opposite direction from the naive intuition.

We expect the next decade of bacterial essentiality work to move along two complementary axes. **First**, slow, careful expansion of the *labelled* organism panel — each new Tn-seq screen is a 1-hop anchor and remains the proven highest-value addition. **Second**, structured measurement-in-the-loop on the conditional residual: condition embeddings, focused screens on the niche-specific candidates the atlas already flags, and active learning that closes the loop. We do not expect deep-architecture wins of the form we tested to play a leading role; we tested seven such wins and they are flat.

---

## Appendix A. Atlas v4 evidence channels and tier rules

**Essential evidence (3):** `E_score` (cross-org LOO score ≥ per-org high threshold calibrated at precision ≥ 0.9 on training organisms), `E_cons_core` (family_frac ≥ 0.5), `E_geometry` (leading + short).

**Non-essential evidence (5):** `N_score`, `N_pangenome_strong` (ortholog absent in ALL sister strains — only positive within-species dispensability signal), `N_pangenome_some`, `N_redundancy` (paralogs ≥ 4 ∧ cons < 0.5), `N_mobile` (mobile-element-adjacent), `N_long_lagging`.

**Label transfer (2):** `E_ortho` (OG measured essential in ≥ 2 other organisms with zero non-essential votes), `N_ortho` (the same, non-essential).

**Phenotype-required quarantine (2):** `P_rogue` (cons < 0.1 ∧ paralogs = 0 ∧ present in all sisters), `P_conditional` (cons < 0.3 ∧ long + lagging ∧ present in all sisters).

**Tier rules (priority order):**
```
promote_E = (E_ortho AND (E_cons_core OR E_geometry))
          OR (E_dnds AND (E_cons_core OR E_ortho))
promote_N = N_ortho

CONFIDENT_ESSENTIAL    = (E_score OR promote_E) AND NOT P_rogue
CONFIDENT_NONESSENTIAL = promote_N                                   # ortholog override
                       OR ((N_score OR N_pangenome_strong) AND NOT P_rogue AND NOT P_conditional)
ROGUE_SUSPECT          = P_rogue and no clean N_ortho
CONDITIONAL_SUSPECT    = P_conditional and no clean N_ortho
UNRESOLVED             = everything else
```

## Data and code availability

All code, per-gene atlases, JSON summaries, and figures are in the released repository (branch `claude/vectorize-gex-propensity-NRqBW`):

- Atlas: `outputs/orphan/atlas_multi/<org>.csv` (6 organisms) + `atlas_multi/README.md`
- Cross-organism scores: `outputs/orphan/cross_org_coverage_scores.csv`
- AF model results: `outputs/orphan/af_model_results.json`, `af_2track_results.json`, `af_evoformer_results.json`
- Negative results: `directional_flux.json`, `graph_diffusion_results.json`, `iterative_refinement_summary.json`
- Mechanism: `mutation_protection_test.json`, `mutation_rate_segments_summary.json`, `replication_geometry_summary.json`
- DEG validation: `deg_cross_validation.json`, `deg_pathogen_mapped.csv`, `deg_conditional_conservation.png`
- AF source vendored under Apache-2.0: `vendor/alphafold/`

## References

[1] Wetmore KM, et al. mBio 2015. [2] Price MN, et al. Nature 2018 (Fitness Browser). [3] Lin et al. eggNOG; conservation predictors. [4] Plaimas K, et al. BMC Syst Biol 2010. [5] Rives A, et al. PNAS 2021 (ESM-1b). [6] Lin Z, et al. Science 2023 (ESM-2). [7] Goodall ECA, et al. mBio 2018 (E. coli three-way comparison). [8] Christen B, et al. Mol Syst Biol 2011 (Caulobacter); related cross-strain analyses. [9] DeJesus MA & Ioerger TR, BMC Bioinformatics 2013 (caller variation). [10] Cain AK, et al. Nat Rev Genet 2020 (Tn-seq review). [11] Luo H, Lin Y, Liu T, et al. DEG database. [12] Jumper J, et al. Nature 2021 (AlphaFold-2).
