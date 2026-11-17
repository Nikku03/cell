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
