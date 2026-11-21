# memory_bank/datasets/

Catalogues of datasets that other work used: what each dataset is, where it lives, how big it is, and which
`memory_bank/sources/` record to cite. **No data files are stored here** -- only type and location, so the
reasoning engine knows what exists and can decide what is worth fetching.

| File | What it catalogues |
| --- | --- |
| `alphagenome2.json` | Every dataset AlphaGenome 2 (Google DeepMind, 2026) was trained or evaluated on: 28 entries (bulk RNA-seq, ATAC/DNase, ChIP-seq, CAGE, PRO-cap, Hi-C/Micro-C, WGBS and snm3C methylation, 36 single-cell studies, a 107-million-cell CELLxGENE snapshot; GTEx, TenK10K, IBDverse, Parkinson's and vaccine QTL/single-cell evaluations, CRISPRi enhancer screens, TraitGym, Open Targets + gnomAD) and the model's own released predictions. Source: `sources/gayoso_2026_alphagenome2.json`. |

Each entry: `role` (training / evaluation / model output), `data_types`, `species`, `scale` (as the paper states it;
chart-read counts are marked approximate), `where` (repository, accession, URL), `source` (id(s) in `sources/`),
`paper_section`, and `relevance_to_this_project` where there is one (e.g. K562 data). Licences live on the source
records; "not checked" means exactly that.

## Holdings: the data itself (`alphagenome2_holdings.json`)

The datasets in `alphagenome2.json` were then pulled into the memory bank as **compressed per-cell-type records**
(`rem/atlas/ag2bank.py`; local cache `rem/atlas/_cache/ag2bank`, git-ignored, ~3.8 GB; rebuild with the commands
listed in the holdings file). Raw, these sources run to many terabytes.

| Layer | What it holds |
| --- | --- |
| `cellxgene_human` / `_mouse` | every gene x every (tissue, cell type) in CZ CELLxGENE: 60,810 genes x 6,790 pairs (64 tissues, 883 cell types, ~87M cells) / 49,413 x 2,044 (32 tissues, 570 cell types, ~18M cells); mean expression in expressing cells + share expressing |
| `encode_rna_human` / `_mouse` | 1,517 / 452 ENCODE RNA-seq experiments (296 / 80 biosamples), TPM per gene |
| `gtex_v8_tissue_median` | 54 human tissues, median TPM per gene |
| `fantom5_cage_human` / `_mouse` | 1,829 / 1,073 CAGE samples, TPM per gene |
| `ccre_human` / `_mouse` | ENCODE Registry v4: which of 2.35M / 0.93M regulatory elements are active (promoter, enhancer, CTCF, ...) in each of 1,678 / 393 biosamples |
| `methylation_wgbs` | 253 sorted-cell WGBS samples (Loyfer 2023), promoter methylation for 39,323 genes |
| `genetics` | GTEx v11 SuSiE eQTL/sQTL/apaQTL; IBDverse sc-eQTL credible sets (72 cell types); Open Targets 26.03 credible sets (slimmed) + studies; TraitGym |
| `enhancer_gene_crispri` | ENCODE-rE2G CRISPRi enhancer-gene benchmarks (K562; 5 held-out cell lines) |

Not held, with the reason for each (contact maps, single-nucleus methylation, raw single-cell matrices, TenK10K,
Parkinson's cohort, COVID atlas, astrocyte screen, AlphaGenome 2's own predictions behind a sign-in): see the holdings file.
