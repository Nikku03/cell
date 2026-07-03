# Integration homework brief (research workflow output)

# Bioinformatics Integration — Engineering Brief

Actionable compilation across five topics. Each gives code-ready facts, method, and hard caveats. Decisions consolidated at the end.

---

## 1. ARCHS4 (human gene HDF5 → co-expression network)

**Facts for code**
- File: v2.x `human_gene` HDF5 (~59 GB). Layout stable across point releases.
- Exact HDF5 paths (verified vs `archs4py` reader source, [github.com/MaayanLab/archs4py](https://github.com/MaayanLab/archs4py)):
  - Matrix: `data/expression` — **INTEGER** (`np.uint32`), **raw Kallisto pseudocounts, rounded**. Not normalized, not log.
  - Genes: `meta/genes/symbol` (v2 field is `symbol`, NOT `gene_symbol`), `meta/genes/ensembl_gene`, `meta/genes/biotype`.
  - Samples: `meta/samples/geo_accession` (GSM, column labels), `meta/samples/series_id` (GSE = batch/study key), `meta/samples/singlecellprobability` (float64), `meta/samples/characteristics_ch1`, `source_name_ch1`, `title` (free text — **no clean `tissue` field**, regex-mine it).
- **Orientation: v2.x is genes × samples** (`gene_axis=0`). Legacy v1 was transposed — detect programmatically by matching axis length to `len(symbol)`, never hard-code.
- I/O: chunked+gzip. **Read by full sample columns** `X[:, i]`; a gene row spans all chunks and is slow. Gather chosen sample columns into RAM once, then transpose.
- Source of truth on contents: [ARCHS4 download page](https://maayanlab.cloud/archs4/download.html) ("Kallisto pseudocounts rounded to integer"); [Lachmann et al., Nat Commun 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5893633/).

**Method**
- Stratified subsample: drop single-cell (`singlecellprobability >= 0.5`), **cap samples per `series_id`** (and per mined tissue) so no study/tissue dominates — flat random draw inflates blood/brain/cancer-line correlations.
- Filter to expressed genes (`archs4py.utils.filter_genes`-style: count > ~5–20 in ≥2–20% of subsample) → ~15–25k genes from raw ~67k.
- Normalize per sample: **log2(CPM+1)** (min viable) or UQ+log ([Johnson & Krishnan, PLOS One 2022](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263344) found UQ best); then **rank-transform each gene across samples → Pearson-on-ranks (= Spearman)** ([Kumari et al., Sci Rep 2018](https://www.nature.com/articles/s41598-018-29077-3)).
- **CLR / per-gene z-standardize** the correlation matrix → threshold to sparse edges (ARCHS4-style edges, suppresses hub artifacts).

**Caveats**
- Matrix is raw counts — skipping normalization gives library-depth-driven garbage.
- ~10k samples is the plateau point (Lachmann 2018: gains marginal beyond 10k); a dense genes×10k float32 matrix + gene×gene corr fits in RAM.
- Cross-study batch cannot be ComBat'd across all GEO; mitigate via stratified sampling + rank standardization + optional PC-sphering, or **aggregate per-context networks** ([Ballouz et al. 2015](https://academic.oup.com/bioinformatics/article/31/13/2123/196230), GeneFriends).

---

## 2. Co-expression validation (functional-relationship lens)

**Facts for code**
- Standard validation = **guilt-by-association / neighbor-voting AUROC** against held-out curated gene sets (GO, KEGG, Reactome, CORUM complexes).
- Use **EGAD** (fast neighbor-voting AUROC with degree-null controls) — do not roll your own ([Ballouz et al. 2017, btw695](https://doi.org/10.1093/bioinformatics/btw695)). Overview: [van Dam et al. 2018](https://doi.org/10.1093/bib/bbw139).
- Realistic target: **aggregate GO macro-AUROC ~0.70–0.75** is a solid, defensible RNA-seq result ([Ballouz, Verleyen & Gillis 2015](https://doi.org/10.1093/bioinformatics/btv118)). **Consistently ≥0.80 → run a leakage/bias check.**

**Method**
- Report three things, not one mean: (a) macro-AUROC, (b) **per-term AUROC distribution** + which pathways actually recover, (c) **degree/multifunctionality-null AUROC** — gain over the null is the real signal ([Gillis & Pavlidis 2011](https://doi.org/10.1371/journal.pone.0017258); [2012](https://doi.org/10.1371/journal.pcbi.1002444)).
- **Replicate on an independent dataset** (GTEx or held-out ARCHS4 partition): recompute AUROC + top-edge overlap. Replicability > any single AUROC ([Farahbod & Pavlidis 2019](https://doi.org/10.1093/bioinformatics/bty538)).
- Bias audits: confirm hub degree ≠ mean expression (mean-matched null); confirm AUROC survives PC/latent-factor correction ([Parsana et al. 2019](https://doi.org/10.1186/s13059-019-1700-9)).
- Keep the three lenses (co-expression, PPI, CRISPR co-essentiality) as **separate rank-standardized edge lists**, validate each with the same EGAD framework, then combine by rank-sum.

**Caveats**
- Aggregate AUROC is driven by outlier terms — "GBA is the exception, not the rule." Don't sell a single mean.
- Node degree alone can produce high AUROC with zero edge information — always report the degree-null.
- Reactome recovers worse than KEGG (large, over-subdivided pathways dilute signal) ([Ballouz et al. 2019, Sci Rep](https://www.nature.com/articles/s41598-019-50885-8)).
- Up to **75% of apparent tissue-specific differential co-expression is just mean-expression change** ([Farahbod & Pavlidis 2019](https://doi.org/10.1093/bioinformatics/bty538)) — hence Spearman + filtering + mean-matched nulls.
- Don't treat scale-free topology as biological validity (functionally uninformative, Gillis & Pavlidis 2012).

---

## 3. ncRNA → target regulatory databases

**Facts for code (recommended two-DB backbone)**

*miRNA → mRNA — miRTarBase 2025* ([NAR 53(D1):D147](https://academic.oup.com/nar/article/53/D1/D147/7907368))
- Portal: `https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/`. Format: **.xlsx**. Files: `hsa_MTI.xlsx` (human), `miRTarBase_MTI.xlsx` (all species).
- Columns: `miRTarBase ID | miRNA | Species (miRNA) | Target Gene | Target Gene (Entrez Gene ID) | Species (Target Gene) | Experiments | Support Type | References (PMID)`.
- `Support Type` tiers edges: strong/functional (reporter assay, Western blot, qRT-PCR) vs weak/high-throughput (CLIP-seq). Filter to strong for gold-standard repression.
- HGNC map: **Entrez Gene ID → `hgnc_complete_set.txt` (`entrez_id`)**, clean 1:1. Prefer Entrez over symbol. License: **CC BY-NC**.

*lncRNA → target — LncTarD 2.0* ([NAR 51(D1):D199](https://academic.oup.com/nar/article/51/D1/D199/6793804))
- Portal: `https://lnctard.bio-database.com/`. Format: **flat tab/CSV**, ~8,360 rows, human-only.
- Carries regulatory **mechanism + direction + disease + PMID**; targets span mRNA / miRNA / TF. Mechanisms: ceRNA/sponge (3,719), transcriptional (595), epigenetic (436), protein-interaction (706), etc.
- IDs: symbol + alias + Entrez + Ensembl → join on Entrez/Ensembl to HGNC; resolve symbol-only rows via `alias_symbol`/`prev_symbol`. License: free non-commercial (CC BY-NC).

*Optional supplements*
- **ENCORI/starBase** (`https://rnasysu.com/encori/`) — CLIP-seq **binding** evidence, per-query tab-delimited **API** (no bulk dump; pass `all`, loop endpoints, control with `clipExpNum`). Returns symbol + Ensembl (hg38). No explicit open license — verify before redistribution.
- **NPInter v5** (`http://bigdata.ibp.ac.cn/npinter5/`) — only if you need lncRNA-protein / broad ncRNA-RNA. CC BY (most permissive) but only 8,586 of 2.6M entries are literature-validated; hardest mapping (NONCODE/miRBase/Ensembl/UniProt → HGNC).

**Method**
- One mapping backbone: download HGNC `hgnc_complete_set.txt` ([genenames.org](https://www.genenames.org/download/statistics-and-files/)) once; key every source's `entrez_id`/`ensembl_gene_id`/`uniprot_ids`/`alias_symbol`/`prev_symbol` to `hgnc_id`.
- Keep miRNA nodes on **miRBase v22.1** accessions (RNAcentral already cross-refs miRBase + NONCODE).
- Filter every source to `Homo sapiens` (miRNA prefix `hsa-`).

**Caveats**
- miRTarBase/LncTarD/NPInter download servers (China-hosted) returned **HTTP 503 through the proxy** this session — exact byte-sizes unverified live; formats/columns are from papers + documented distribution. Budget a retry/mirror.
- Don't mix binding evidence (ENCORI, CLIP) with functional repression (miRTarBase strong tier) as one confidence level — tag edge evidence type.

---

## 4. GTEx v8 trans-eQTL

**Facts for code**
- **Primary trans file** (verified HTTP 200, `adult-gtex` bucket):
  `https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-trans-qtl/GTEx_Analysis_v8_trans_eGenes_fdr05.txt`
- Format: **plain TSV, uncompressed (~22 KB), 162 data rows + header.** FDR < 0.05, one row per significant tissue × distant-gene trans-eQTL (gene-centric, one top variant per trans-eGene per tissue — NOT all-pairs).
- Columns: `tissue_id | gene_id | gene_name | gene_chr | biotype | gene_mappability | variant_id | tissue_af | slope | slope_se | pval_nominal | fdr`.
  - `variant_id` = `chr_pos_ref_alt_b38` (GRCh38); `gene_id` = versioned GENCODE/Ensembl; `tissue_id` e.g. `Nerve_Tibial`. Gives the requested (variant, gene, tissue) triple directly.
- Companion trans-sQTL: same folder, `GTEx_Analysis_v8_trans_sGenes_fdr05.txt`.
- **cis** (much richer): `.../single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar` (per tissue: `<TISSUE>.v8.egenes.txt.gz`, `<TISSUE>.v8.signif_variant_gene_pairs.txt.gz` with `variant_id, gene_id, tss_distance, maf, pval_nominal, slope, slope_se, pval_beta`), plus `README_eQTL_v8.txt`. Full cis all-associations (~462 GB) in requester-pays `gs://gtex-resources`.

**Method**
- Fetch the single 22 KB trans file directly; parse as TSV; project `(variant_id, gene_id, tissue_id)`. Use `gene_chr` to confirm cross-chromosome pairs.
- For cis, download the `.tar`, iterate per-tissue `signif_variant_gene_pairs`.

**Caveats**
- **Legacy path `gtex_analysis_v8/single_tissue_qtl_data/...` now returns 404** — use the `adult-gtex/bulk-qtl/v8/` path.
- Format asymmetry: trans is gene-centric (top variant per eGene); cis significant-pairs is variant×gene layout — don't assume identical schemas.
- Only 162 trans rows genome-wide — trans signal is sparse; don't expect an exhaustive network.
- Sources: [GTEx QTL downloads](https://www.gtexportal.org/home/downloads/adult-gtex/qtl); [GTEx v8, Science 2020](https://www.science.org/doi/10.1126/science.aaz1776).

---

## 5. Transformer (KG-conditioned perturbation prediction)

**Facts for code (the five FMs, on tokenization/objective)**
- **Geneformer** ([Nature 2023](https://www.nature.com/articles/s41586-023-06139-9)): rank-value encoding (genes ranked by expr÷corpus-mean, top ~2,048; **no expression magnitudes**), BERT encoder-only, masked-gene-prediction. In-silico perturbation = cosine shift of embedding — qualitative only, no dose/fold-change.
- **scGPT** ([Nat Methods 2024](https://www.nature.com/articles/s41592-024-02201-0)): per gene = learned gene-ID emb + binned-expression emb + condition token; generative masked pretraining; dedicated perturbation fine-tune mode but requires per-dataset fine-tuning, preprocessing-sensitive.
- **scFoundation** ([Nat Methods 2024](https://www.nature.com/articles/s41592-024-02305-7)): continuous scalar expr embedding + 2 read-depth (RDA) tokens, full ~19,264 genes, xTrimoGene asymmetric encoder-decoder, read-depth-aware masked modeling.
- **CellPLM**: cells-as-tokens, transformer attends across cells (uses spatial transcriptomics), GMM latent prior; smallest corpus (~11.4M).
- **UCE** ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)): **gene token = ESM2 protein embedding** → zero-shot to unseen genes/species. Embedding model only, no perturbation head. **Borrow the ESM2 gene-init trick.**

**The uncomfortable reality (design around this)**
- Deep FMs **do not beat trivial baselines** for unseen perturbations: mean-of-training for single-gene, additive model for two-gene ([Ahlmann-Eltze, Huber & Anders, Nat Methods 2025, DOI 10.1038/s41592-025-02772-6](https://www.nature.com/articles/s41592-025-02772-6)). PCA matches/beats FMs ([Csendes et al., arXiv:2410.13956](https://arxiv.org/pdf/2410.13956)). FM attention captures co-expression, not causal signal (arXiv:2602.17532).
- What helps: **structured KG prior over genes** + **predict the change (Δ), not absolute state**. Precedents: **GEARS** (GNN over GO+co-expression KG → perturbation embeddings, extrapolates to never-perturbed genes; [Nat Biotech 2024](https://www.biorxiv.org/content/10.1101/2022.07.12.499735v2.full)); **STATE** (Arc, set-level bidirectional transformer over cell populations); **CPA** (additive latent perturbation operator, [DOI 10.15252/msb.202211517](https://doi.org/10.15252/msb.202211517)).

**Recommended architecture — two-tower + transition operator**
- **Tower A (gene/KG encoder):** node init = ESM2 emb + KG features (compartment, process, CRISPR essentiality, PTMs); heterogeneous/relational GNN (R-GCN or HGT) over typed edges (TF→target directed, PPI, same-complex, co-essentiality, co-expression, shared-pathway, drug→target). Pretrain via link prediction + node-feature reconstruction. This is the source of unseen-gene generalization.
- **Tower B (expression):** per-gene token = Tower-A `g_i` + binned-expression emb + condition emb; transformer → cell embedding; masked-expression pretraining (keep scFoundation read-depth variant if depths vary).
- **Tower C (transition operator):** perturbation emb `p = MLP(pool_i g_i, direction, dose)` (unseen target still yields `p` from KG neighborhood); set-level bidirectional transformer over control cells + `p` → **predict per-cell Δ (post−pre)**; decoder head **weight-tied to `g_i`** so any gene with an embedding gets an output.
- **Training:** Stage 1 KG pretrain → Stage 2 masked-expression pretrain → Stage 3 fine-tune B+C on Perturb-seq, **holding out whole genes/contexts** to force and measure KG extrapolation.
- **Losses:** Δ-reconstruction (Poisson/NB or MSE-on-log-Δ) **weighted toward DE genes**; distributional MMD/Sinkhorn between predicted vs true perturbed populations; auxiliary contrastive alignment between `z_c` and KG pathway-activity embedding.

**Caveats**
- Global MSE is minimized by the control mean — always evaluate against mean-of-training, additive, and PCA baselines; split by unseen-single / unseen-combination / unseen-cell-context; metrics = DE-direction accuracy, Pearson/Spearman of Δ on top-DE genes, energy distance. Adopt Virtual Cell Challenge protocol for comparability.
- Set realistic expectations: wins are most attainable for in-distribution genes, ranking, and cell-context transfer; exact magnitudes for genuinely novel perturbations remain hard. The KG (co-essentiality + TF→target + complex/pathway edges) is the primary lever.

---

## Decisions

1. **ARCHS4 subsample size: 10,000 samples** — the published accuracy plateau (Lachmann 2018); keeps dense genes×10k float32 + gene×gene correlation in RAM. Draw stratified (single-cell removed, capped per `series_id`).
2. **Correlation type: Spearman** (rank-transform each gene across samples, then Pearson-on-ranks) — neutralizes the highly-expressed-gene bias, outliers, and per-study scale differences that dominate ARCHS4; then CLR / per-gene z-standardize before thresholding.
3. **Co-expression validation: EGAD neighbor-voting AUROC** vs GO/KEGG/Reactome/CORUM, reported as macro + per-term distribution + **degree-null**, target **0.70–0.75**, with independent-dataset replication as the primary bar (≥0.80 triggers a leakage check).
4. **ncRNA DBs: miRTarBase 2025 (miRNA→mRNA, `hsa_MTI.xlsx`, filter Support Type = strong) + LncTarD 2.0 (lncRNA→target).** HGNC `hgnc_complete_set.txt` as the single mapping backbone; miRNA nodes on miRBase v22.1. ENCORI as optional lower-confidence binding layer; NPInter only if lncRNA-protein breadth is needed.
5. **GTEx: pull `GTEx_Analysis_v8_trans_eGenes_fdr05.txt` from the `adult-gtex/bulk-qtl/v8/single-tissue-trans-qtl/` path** (not the 404 legacy path); project `(variant_id, gene_id, tissue_id)`.
6. **Transformer architecture: custom two-tower KG-conditioned model** — heterogeneous-GNN gene tower (ESM2 + multi-omic KG init, GEARS-generalized) + scGPT-style expression tower reusing gene embeddings + STATE-style set-level transition operator predicting **Δ**, weight-tied output head for unseen readouts. Not an off-the-shelf FM — always benchmarked against mean/additive/PCA baselines.
