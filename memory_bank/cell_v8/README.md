# cell_v8: 11 more screens, local only

Written by `rem/atlas/cellbank8.py pull` and `build`. The records are kept in `rem/atlas/_cache/cell_v8/`, which git ignores. This file and the build statistics in `outputs/cellbank8/` are the only parts committed.

## Why the records are not in the repository

The new screens come from a uniformly reprocessed collection: figshare 34002315, "Uniformly processed single-cell CRISPRi screens", processed with crispyx v0.1.4. That collection is licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike). Records derived from it would carry the same terms, so they stay on the machine that built them.

To rebuild them on another machine, run `cellbank8.py pull` and then `build`. `remotezip.py` reads only the effect matrix out of each archive with HTTP range requests, about 8.5 GB in total instead of about 60 GB.

## Contents

Everything in `cell_v6`, plus these 11 screens:

| Screen | Cell type | Perturbation set | Records kept |
|---|---|---|---|
| RPE1 (reprocessed) | RPE1 | essential | 1,884 |
| HepG2 (reprocessed) | HepG2 | essential | 1,859 |
| Jurkat (reprocessed) | Jurkat | essential | 1,956 |
| HCT116 | HCT116 | genome-wide | 4,693 |
| HEK293T | HEK293T | genome-wide | 4,740 |
| iPSC-N (Nourreddine) | iPSC | genome-wide | 4,281 |
| iPSC-Fsf (Feng) | iPSC | genome-wide sub-screen | 1,392 |
| iPSC-Fsnf (Feng) | iPSC | genome-wide sub-screen | 1,173 |
| iPSC-Fts (Feng) | iPSC | targeted | 358 |
| hESC-P (Pan) | H1 hESC | growth / pluripotency | 2,214 |
| hESC-V (Virtual Cell Challenge) | H1 hESC | effect strength | 183 |

Only knockdowns of genes with screen-A features are kept. That gives 37,234 records in 7.0 MB.

Test genes with a record in another cell type: **1,007 before, 2,512 now**, out of 2,525.

## How a record is built

1. Take the effect (perturbed minus matched control, log1p scale).
2. Divide by the gene's median control standard deviation, floored at the 10th percentile, to reach roughly the Replogle z scale.
3. Remove the screen's mean response over TRAIN genes.
4. Map onto screen A's genes by symbol (HPA synonyms allowed).
5. Project onto the bank's 128 components and store as int8 codes plus one scale.

`records_meta.json["strength"]` holds each record's norm divided by its screen's median norm.

## Sources

The original screens:
- Replogle et al. 2022 (*Cell*)
- Nadig et al. 2025 (*Nature Genetics*)
- Huang et al. 2025 (X-Atlas/Orion, *bioRxiv*)
- Nourreddine et al. 2026 (*Nature Biotechnology*)
- Feng et al. 2026 (*Cell Genomics*)
- Pan et al. 2026 (*bioRxiv*)
- Roohani et al. 2025 (*Cell*; Virtual Cell Challenge)

Cite them and the reprocessing deposit when using these records.
