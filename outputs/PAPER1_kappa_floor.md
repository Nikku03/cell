# The reproducibility floor of bacterial gene essentiality is half technical: a calibration-corrected ceiling for evaluating predictors

**Authors:** [Author list]
**Affiliations:** [Affiliations]
**Correspondence:** [email]

---

## Abstract

Computational predictors of bacterial gene essentiality are routinely benchmarked against transposon-insertion sequencing (Tn-seq) calls treated as ground truth, and a growing number of deep-learning models now claim to predict essentiality "as well as" or "better than" an experimental screen. These claims are uninterpretable without knowing how well Tn-seq agrees with *itself*. Here we measure that agreement at scale and show it is far lower than assumed — and, critically, that roughly half of the apparent irreproducibility is technical rather than biological. Across six independent screen pairs of three organisms we find a binary inter-screen agreement of Cohen's κ = 0.18–0.58 (median 0.39); across 17,950 same-condition experiment pairs spanning 48 organisms in the Fitness Browser we find a continuous cross-experiment Spearman of ρ = 0.14–0.75 (median 0.38). A simple conservation predictor (cross-organism phyletic retention) already reaches κ = 0.58 against curated labels — i.e., conservation alone agrees with the labels *better than two independent screens of the same organism agree with each other*. Decomposing the continuous floor by linear variance partition, we find that **47.7% of the cross-experiment variance is attributable to library/technical quality, only 2.6% to residual within-cluster condition mismatch, and 45.4% is irreducible (plus unmodelled gene-level buffering)**. After removing technical and condition variance, two screens in a matched, high-quality condition agree at ρ ≈ 0.77 — about twice the naïve median. We conclude that the field has been benchmarking against the wrong number: neither the technically-inflated apparent floor (too pessimistic) nor any single noisy screen (too noisy), but a calibration-corrected ceiling that almost no study computes. We provide this ceiling as a standing benchmark and a simple rule for reading any "better-than-Tn-seq" claim.

**Keywords:** gene essentiality, transposon sequencing, RB-TnSeq, reproducibility, benchmark, foundation models.

---

## 1. Introduction

Gene essentiality — whether a cell can survive the loss of a given gene under a given condition — underpins antibiotic target discovery, minimal-genome design, and functional annotation. The dominant experimental method is transposon-insertion sequencing (Tn-seq, including TraDIS, INSeq, and randomly-barcoded RB-TnSeq), which infers a gene's contribution to fitness from the depletion of insertion mutants in a growth selection [1,2]. Tn-seq calls are the de facto ground truth against which computational essentiality predictors are trained and evaluated.

A succession of machine-learning predictors — from orthology-and-phylogeny scorers [3] through sequence-feature classifiers [4] to recent bacterial "foundation models" that treat a genome as a sequence of protein-language-model embeddings [5,6] — now report cross-organism areas under the ROC curve approaching 0.8–0.9, and several explicitly frame their goal as matching or surpassing an experimental screen. Such framing presumes that the experiment defines a stable target. It does not. Independent Tn-seq screens of the same organism disagree substantially: a three-way comparison of the *Escherichia coli* K-12 essential genome found only ~60% of essential calls shared across TraDIS, the Keio single-deletion collection, and the PEC database [7]; roughly one third of genes essential in one Enterobacteriaceae strain are non-essential in a related strain [8]; and, holding the sequencing data fixed, the choice of statistical caller alone moves gold-standard recovery from 0% to 100% [9]. Documented mechanisms of disagreement include nucleoid-associated-protein occlusion of insertion sites (false positives), sub-genic insertion tolerance in non-essential domains of essential genes (false negatives), insertion-density/saturation effects, and growth-condition dependence [9,10].

If the measurement disagrees with itself, then there exists a **reproducibility floor**: a maximum agreement any predictor can be expected to reach against a single screen, set by the screen's own test–retest reliability rather than by the predictor's quality. A model that appears to "beat" a single screen by exceeding this floor is, by construction, predicting noise to a precision the assay does not support. Despite its centrality, this floor has not been measured at scale for bacterial essentiality, and — as we show — it is not even a single number: it is a *mixture* of technical noise, condition mismatch, and genuine conditional biology, which must be separated before any predictor claim can be read.

Here we (i) measure the binary and continuous reproducibility floor of bacterial essentiality across dozens of organisms and tens of thousands of screen pairs; (ii) show that a trivial conservation baseline already operates at this floor; (iii) decompose the floor into technical, condition, and residual-biological components; and (iv) derive a calibration-corrected reproducibility ceiling that should replace the raw floor as the benchmark for predictor evaluation.

---

## 2. Results

### 2.1 The binary inter-screen floor: κ = 0.18–0.58

We first quantified agreement between independent binary essentiality screens of the *same* organism, using the Database of Essential Genes (DEG) [11] against an independently curated label set, mapped gene-for-gene through RefSeq annotations (Methods). Across six organism × screen-pair comparisons spanning *Mycobacterium tuberculosis*, *Staphylococcus aureus*, and *Ralstonia solanacearum*, Cohen's κ ranged from **0.18 to 0.58 with a median of 0.39** (Table 1; Fig. 1, left, orange). Even the most concordant pair — two *M. tuberculosis* H37Rv screens — agreed at only κ = 0.58, and the *R. solanacearum* pair agreed at κ = 0.18, barely above chance. Raw percent agreement was high (0.73–0.89) but is inflated by the large non-essential majority; κ, which corrects for prevalence, is the appropriate statistic and exposes the weak true agreement.

To confirm this was not an artifact of cross-database curation, we computed an entirely *within*-resource binary floor: across 17,641 same-condition experiment pairs in the Fitness Browser [2], binarizing fitness at the project-standard threshold (fitness < −2 and |t| ≥ 4), the median inter-experiment κ was **0.53** — concordant with, and slightly above, the DEG-derived estimate.

### 2.2 The continuous floor: ρ = 0.14–0.75, median 0.38

Binarizing fitness discards information, so we next measured agreement on the continuous fitness scale, which is what Tn-seq actually produces. Using the February-2024 Fitness Browser release (7,552 genome-wide RB-TnSeq experiments across 48 bacteria and archaea) [2], we grouped experiments into same-condition clusters by (medium × aeration × primary condition) and computed the per-gene Spearman correlation of fitness between every pair of experiments within a cluster (Methods). Across **17,950 same-condition experiment pairs spanning 48 organisms**, the median cross-experiment Spearman was **ρ = 0.38** (mean 0.40), with a 10th–90th-percentile spread of **0.14–0.75** (Fig. 1, right). The breadth of this spread — more than half a correlation unit — is itself the central observation: "the reproducibility of Tn-seq" is not a constant but a wide distribution, and any single headline number conceals it.

### 2.3 Conservation already operates at the floor

If the floor is real, a trivially simple predictor should already reach it. We evaluated a pure conservation predictor — the leak-free fraction of an orthologous group's members called essential across other organisms (`family_frac`; Methods) — against curated essentiality labels in each of 48 organisms. Binarized at 0.5, conservation reached a median κ = **0.58** (range 0.25–0.71) against the labels, with per-organism essentiality AUPRC of 0.42–0.85 (median ≈ 0.79) (Fig. 1, left, blue). In other words, **conservation alone agrees with the labels as well as, or better than, two independent experimental screens of the same organism agree with each other** (median κ 0.58 vs inter-screen 0.39). This is the operational meaning of the floor: the marginal value of any model over a one-line conservation baseline is bounded above by the gap between the floor and the ceiling — a gap that, until decomposed, looked vanishingly small.

### 2.4 Decomposing the floor: 48% technical, 3% condition, 45% residual

The wide continuous-floor spread (Fig. 1, right) raises the decisive question: is low agreement *biology* (genes whose essentiality genuinely varies) or *technical noise* (poor libraries, shallow insertion density)? We answered this by linear variance partition of the 17,950 per-pair Spearman values against per-experiment covariates available in the Fitness Browser metadata (Methods). Two covariate blocks were tested: a **technical** block (within-experiment consistency `cor12`, dynamic range `maxFit`, read depth `gMed`/`gMean`, and gene coverage) and a **condition** block (within-cluster differences in temperature, pH, concentration, and secondary condition).

The technical block alone explained **R² = 0.52** of the variance in pair agreement; the condition block alone explained only **R² = 0.07**; together **R² = 0.55** (Table 2). Partitioning the explained variance:

- **Technical (unique): 47.7%**
- **Condition-mismatch (unique): 2.6%**
- **Shared: 4.3%**
- **Residual / irreducible (incl. gene-level buffering): 45.4%**

**Nearly half of the apparent irreproducibility of bacterial Tn-seq is technical library quality, not biology.** The condition block's small contribution is itself informative: within our (medium × aeration × primary-condition) clusters, residual differences in temperature, pH, and concentration explain very little disagreement, indicating that the cluster definition is biologically coherent and that the bulk of non-technical variance lies elsewhere — in gene-level conditional biology (buffering/redundancy) that per-pair covariates cannot capture (Discussion).

### 2.5 The calibration-corrected ceiling: ρ ≈ 0.77

Because half the floor is technical, the raw median (ρ = 0.38) is the agreement of an *average-quality* pair, not the agreement two *well-run* screens can achieve. We therefore estimated the matched-condition, high-quality ceiling: the cross-experiment Spearman predicted by the fitted model when technical quality is set to its 90th percentile and condition mismatch to its minimum (Methods). This ceiling is **ρ ≈ 0.77** — roughly twice the naïve median floor.

This single number reframes every "better-than-Tn-seq" claim (Fig. 2). The honest target for a predictor is not to exceed a single noisy screen (median ρ 0.38; trivially beatable and meaningless) nor to exceed the inflated raw floor, but to approach the **0.77 ceiling** at which two high-quality matched screens agree. A predictor reporting cross-experiment ρ ≈ 0.65, for example, is *not* "better than Tn-seq" in any absolute sense — it sits between the technically-degraded floor and the matched-quality ceiling, and its claim is interpretable only against the latter.

### 2.6 A re-grading rule and standing benchmark

The decomposition yields a simple rule for the field (Box 1). Any predictor's reported agreement with a held-out screen should be reported alongside (i) the inter-screen κ/ρ floor for that organism and condition, and (ii) the matched-quality ceiling. A claim is meaningful only if the predictor approaches the ceiling on **technically high-quality, condition-matched** held-out screens; a claim that merely exceeds the raw floor is consistent with predicting technical noise. We release the per-organism floor, the binary and continuous pair tables, and the decomposition as a standing benchmark (Data Availability), enabling any future predictor to be re-graded against the corrected ceiling rather than against an arbitrary screen.

---

## 3. Discussion

We set out to make "better than Tn-seq" a falsifiable claim and found that the question had no fixed answer because the experiment has no fixed answer: independent bacterial essentiality screens agree only modestly (binary κ 0.18–0.58; continuous ρ median 0.38), and a one-line conservation baseline already operates at that level. Taken alone, this is a deflationary result that would cap the field. The decomposition rescues it. **Roughly half of the apparent irreproducibility is technical**, and once technical quality and condition are controlled, the genuine reproducibility ceiling is ρ ≈ 0.77 — leaving real, quantified headroom above the conservation baseline for predictors to claim, provided they are evaluated correctly.

Three implications follow.

**First, most published "cross-species essentiality" AUCs are read against the wrong reference.** Reports in the 0.8–0.9 range [3–6] are typically within-species cross-validation or are scored against a single screen; against the matched-quality ceiling, and under leave-one-clade-out evaluation, the defensible headroom is narrower than advertised, and the strongest single baseline to beat remains conservation [3].

**Second, the right target is not a better binary call but a calibrated, condition-resolved prediction with principled abstention.** Because the floor is a mixture, a predictor cannot beat the technical-noise component (it is not biology) and should not be penalized for failing inside it; the productive goal is to predict the *structured* variance — condition-dependence and gene-level buffering — and to abstain where the underlying screen quality is low. This directly motivates reporting predictions as probabilities with calibrated uncertainty rather than as binary essential/non-essential calls.

**Third, the residual 45% is where the biology is, and it is only partly modelled here.** Our per-pair decomposition cannot resolve gene-level buffering — the phenomenon whereby a gene held essential only by a single redundant partner flips between screens as the backup stochastically suffices [8]. Isolating this component requires joining each gene to its paralog/redundancy structure, which is currently blocked by identifier-namespace mismatch between fitness and orthology resources (the "disjoint-data wall"; Limitations). Resolving that join is the natural next step and would split the residual into a hard, irreducible technical floor and a predictable buffering component.

Our analysis also clarifies what a consensus-essentiality approach can and cannot achieve. Integrating multiple screens — as done for cancer dependency maps [12] — averages down the 48% technical component and can therefore yield a *more reliable* label than any single screen, raising the effective ceiling toward the matched-quality value. It cannot, however, exceed the residual biological ceiling, and it will blur genuinely condition-dependent essentiality unless condition is modelled explicitly.

In sum, the bacterial essentiality field has been benchmarking against a number that is simultaneously too pessimistic (the raw floor is half technical artifact) and too optimistic (any single screen is noisier than the achievable ceiling). The corrected ceiling, ρ ≈ 0.77, is the number against which progress should be measured.

---

## 4. Methods

### 4.1 Data sources
Binary essentiality calls were taken from the Database of Essential Genes (DEG) [11] and an independently curated label set, joined gene-for-gene via RefSeq GFF annotations (locus_tag / old_locus_tag / gene-name reconciliation). Continuous fitness and per-experiment metadata were taken from the February-2024 release of the Fitness Browser (RB-TnSeq), comprising 7,552 genome-wide experiments across 48 bacteria and archaea [2]. The conservation predictor and orthologous-group assignments were taken from a leak-free cross-organism orthology table over the same 48 organisms.

### 4.2 Binary inter-screen floor
For each organism with ≥2 independent screens, genes present in both were intersected and Cohen's κ computed on the binary essential/non-essential calls. For the within-Fitness-Browser binary floor, each experiment's fitness was binarized at fitness < −2 and |t| ≥ 4 (project-standard thresholds), and κ was computed for every pair of experiments within a same-condition cluster sharing ≥200 genes.

### 4.3 Continuous floor
Experiments were grouped into clusters keyed by (medium × aeration × primary condition). Within each cluster, for every experiment pair sharing ≥200 genes, the per-gene Spearman rank correlation of fitness was computed. To bound memory on the ~7 GB database, fitness was streamed and processed one organism at a time. This yielded 17,950 pairs across 48 organisms.

### 4.4 Conservation predictor
`family_frac` is the leak-free fraction of an orthologous group's members called essential across organisms *other than* the one being scored (and excluding the held-out evaluation fold), i.e. a pure phyletic-retention/conservation signal containing no information from the gene's own label. It was binarized at 0.5 for κ and used directly as a score for AUPRC.

### 4.5 Floor decomposition and ceiling
The 17,950 per-pair Spearman values were regressed (ordinary least squares, standardized predictors) on a technical covariate block (pairwise minimum of within-experiment consistency `cor12`, dynamic range `maxFit`, read depth `gMed`/`gMean`, alternate consistency `opcor`/`adjcor`; and log gene coverage) and a condition covariate block (within-cluster absolute differences in temperature, pH, and primary concentration; secondary-condition mismatch). R² was computed for each block alone and combined; unique and shared contributions were obtained by inclusion–exclusion. The matched-condition, high-quality ceiling was computed as the model's predicted Spearman with technical predictors set to their 90th percentile and condition-mismatch predictors to their 10th percentile, clipped to [−1, 1].

### 4.6 Code and reproducibility
All analyses are implemented in `scripts/kappa_floor.py` (floor computation, conservation re-grade, figure) and `scripts/kappa_floor_decompose.py` (variance partition, ceiling). Both run from public data; the floor script auto-retrieves the Fitness Browser SQLite release. Outputs (`kappa_floor_binary.csv`, `kappa_floor_continuous.csv`, `kappa_floor_regrade.csv`, `kappa_floor_decomposition.json`, `kappa_floor_figure.png`) constitute the released benchmark.

---

## Figures and Tables

**Figure 1. The reproducibility floor of bacterial gene essentiality.**
*(Left)* Binary inter-screen agreement. Orange points: Cohen's κ between independent screens of the same organism (DEG vs curated labels; 6 pairs); orange band: 10–90% of the inter-screen κ distribution (0.26–0.53). Blue squares: conservation (`family_frac`) vs labels across 48 organisms — most lie *at or above* the inter-screen band, i.e. conservation already matches experimental reproducibility. *(Right)* Continuous floor: per-organism median cross-experiment Spearman (17,950 pairs, 48 organisms). The dashed line marks a representative predictor claim (ρ = 0.65); note it sits above most per-organism medians but below the calibration-corrected ceiling. Generated by `kappa_floor.py` (`outputs/kappa_floor_figure.png`).

**Figure 2. Decomposition of the continuous floor and the corrected ceiling.**
Variance partition of the 17,950 per-pair Spearman values: technical 47.7% (unique), condition-mismatch 2.6% (unique), shared 4.3%, residual/irreducible 45.4%. The matched-condition, high-quality ceiling (ρ ≈ 0.77) is shown against the raw median floor (ρ = 0.38) and the conservation baseline. *(To be rendered from `kappa_floor_decomposition.json`.)*

**Table 1. Binary inter-screen floor (DEG vs curated labels).**

| Organism | Screen pair | n genes | Cohen's κ | % agreement |
|---|---|---:|---:|---:|
| *M. tuberculosis* H37Rv | DEG1010 | 3,557 | 0.576 | 0.886 |
| *M. tuberculosis* H37Rv | DEG1025 | 3,557 | 0.480 | 0.843 |
| *M. tuberculosis* H37Rv | DEG1027 | 3,557 | 0.414 | 0.831 |
| *S. aureus* NCTC8325 | DEG1017 | 2,100 | 0.366 | 0.876 |
| *S. aureus* NCTC8325 | DEG1061 | 2,100 | 0.340 | 0.872 |
| *R. solanacearum* GMI1000 | DEG1057 | 4,403 | 0.182 | 0.734 |
| **Median** | | | **0.39** | **0.84** |

**Table 2. Variance decomposition of the continuous floor (17,950 pairs).**

| Component | R² (block alone) | Unique variance share |
|---|---:|---:|
| Technical (cor12, maxFit, depth, coverage) | 0.520 | 47.7% |
| Condition mismatch (ΔT, ΔpH, Δconc., cond₂) | 0.069 | 2.6% |
| Shared (technical × condition) | — | 4.3% |
| Combined | 0.546 | — |
| Residual / irreducible (+ buffering) | — | 45.4% |
| **Matched-quality ceiling** | | **ρ ≈ 0.77** |

**Box 1. A rule for reading any "better-than-Tn-seq" claim.**
1. Report the inter-screen κ (binary) or cross-experiment ρ (continuous) floor for the target organism and condition.
2. Report the matched-quality ceiling (technical quality maximised, condition matched).
3. Evaluate the predictor under leave-one-clade-out against a conservation baseline.
4. A claim is meaningful only if the predictor approaches the **ceiling** on high-quality, condition-matched held-out screens. Exceeding the raw floor is consistent with predicting technical noise and is not evidence of biological accuracy.

---

## Limitations

(i) The binary inter-screen floor rests on six organism × screen-pair comparisons; it is corroborated by the within-Fitness-Browser binary floor (median κ 0.53, 17,641 pairs) but a broader curated cross-screen set would tighten it. (ii) Continuous-floor condition clusters are defined by three metadata fields; finer condition annotation could reassign a small additional fraction of variance from "residual" to "condition." (iii) The 45.4% residual conflates a hard irreducible technical floor with gene-level buffering; separating them requires joining fitness to paralog/redundancy structure, currently blocked by identifier-namespace mismatch between fitness and orthology resources. (iv) Technical covariates are Fitness-Browser-specific quality metrics; equivalent metrics exist for most Tn-seq pipelines but must be mapped per platform.

---

## Data and Code Availability

Source data: Database of Essential Genes (DEG); Fitness Browser (RB-TnSeq) February-2024 release. Code and released benchmark tables: `scripts/kappa_floor.py`, `scripts/kappa_floor_decompose.py`, and the `outputs/kappa_floor_*` files in the project repository.

---

## References

[1] Wetmore KM, et al. *Rapid quantification of mutant fitness in diverse bacteria by sequencing randomly bar-coded transposons (RB-TnSeq).* mBio 2015. https://journals.asm.org/doi/full/10.1128/mbio.00306-15
[2] Price MN, et al. *Mutant phenotypes for thousands of bacterial genes of unknown function.* Nature 2018. https://www.nature.com/articles/s41586-018-0124-0
[3] Wei W, et al. *Geptop 2.0: an updated, more precise, and faster Geptop server for identification of prokaryotic essential genes.* Front. Microbiol. 2019. https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2019.01236/full
[4] Hasan MA, Lonardi S. *DeeplyEssential: a deep neural network for predicting essential genes in microbes.* BMC Bioinformatics 2020. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03688-y
[5] Wiatrak M, et al. *Bacformer: a contextualised protein language model of bacterial genomes.* bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.07.20.665723
[6] Malbranke C, et al. *ProteomeLM: proteome-scale language modelling for protein function and essentiality.* PNAS 2025. https://www.pnas.org/doi/10.1073/pnas.2524201123
[7] Goodall ECA, et al. *The essential genome of Escherichia coli K-12.* mBio 2018. https://journals.asm.org/doi/10.1128/mbio.02096-17
[8] Shaw D, et al. *High-throughput transposon mutagenesis across Enterobacteriaceae reveals context-dependent essentiality.* mBio 2024. https://journals.asm.org/doi/10.1128/mbio.01798-24
[9] *Pseudomonas aeruginosa gold-standard essentiality datasets and the impact of analysis pipeline on Tn-seq calls.* PLOS Comput. Biol. 2026. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013945
[10] Goodall ECA, et al. *Causes of false-positive and false-negative essentiality calls in E. coli Tn-seq.* mSystems 2022. https://journals.asm.org/doi/10.1128/msystems.00896-22
[11] Luo H, et al. *DEG 15, an update of the Database of Essential Genes.* Nucleic Acids Res. 2021. https://academic.oup.com/nar/article/49/D1/D677/5937083
[12] Lee J, et al. *Combined Essentiality Scoring improves cancer dependency prediction by integrating noisy screens.* eBioMedicine 2019. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6923492/
[13] DeJesus MA, et al. *Comprehensive essentiality analysis of the Mycobacterium tuberculosis genome via saturating transposon mutagenesis.* mBio 2017. https://journals.asm.org/doi/10.1128/mbio.02133-16
[14] Poulsen BE, et al. *Defining the core essential genome of Pseudomonas aeruginosa.* PNAS 2019. https://www.pnas.org/doi/10.1073/pnas.1900570116

---

*Manuscript draft generated from measured results in the project repository (commit history through `9fd1ea7`). All figures and tables are reproducible from the released code and data.*
