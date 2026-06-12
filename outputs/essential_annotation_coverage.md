# Functional annotation coverage of essentials — local data

Honest audit of how much of our essentiality data carries biological
function annotation that we can use to categorize the genes.

## Headline numbers

Of **52,940 essential gene labels** across 59 organisms in `labels.csv`:

| | n | % |
|---|---|---|
| has any `gene_name` string | 3,021 | 5.7% |
| of those, real functional annotation (canonical symbol like `rpoB` or descriptive like "ribosomal protein S5") | 1,230 | **2.3%** |
| has og_id + has functional name | **0** | **0%** |

## The disjoint-data problem (this is the real finding)

Our 59 organisms split into two non-overlapping subsets:

**The 11 DEG-benchmark organisms** have canonical gene names but NO og_id
in our local orthology table:

| organism | n essentials | % with real annotation |
|---|---|---|
| ecoli_BW25113_tradis | 139 | **100%** |
| aeromonas | 394 | 77% |
| syn3a | 383 | 70% |
| spne19F | 196 | 55% |
| spneT4 | 197 | 55% |
| mgen | 382 | 54% |

**The 48 beril_ Fitness Browser organisms** have og_id but NO gene names
(accession-style locus_tags like `RS_RS12300`, `SAOUHSC_01668`):

| organism | n essentials | % with real annotation |
|---|---|---|
| any beril_ org | ~1,000 each | **0%** (except SynE 11%) |

The exception is beril_SynE (cyanobacterium, 11%) — Synechococcus' locus
tags happen to embed canonical symbols.

## Why this matters

In our local data, the **5,658 essential OGs** that come out of the
rarefaction / bucket analysis are ENTIRELY in the beril_ subset. None of
them have a functionally-annotated member in our local labels.csv — not
even via cross-organism orthology, because the og_id system doesn't
bridge to the DEG-named subset.

**So when we ask "what category does each essential OG fall into?" using
local data alone, the answer is structurally limited to flag-based buckets
(regulator/transporter/signaling/metabolic) — 16% coverage at best. The
remaining 84% of essential OGs are "unannotated by local flags."**

This is why the proper categorization required `feba.db` on Drive
(`BestHitKEGG` → `KEGGMember` → KO → `KgroupDesc`). That run resolved
**24% of essential OGs** by KEGG categorization — the right number for
"how many essential OGs do we have biological function for?"

## The full reality check

| layer | coverage of 5,658 essential OGs with function |
|---|---|
| local flag features (regulator/transporter/signaling/metabolic) | 16% |
| KEGG KO via feba.db (BestHitKEGG → KEGGMember) | **69%** of essentials map to a KO |
| KEGG KO with our keyword categorization | 24% of OGs |
| Pfam domain (feba.db GeneDomain) | ~55% of genes |
| FBA / metabolic model presence | 6% of OGs |
| **REAL "no function annotation anywhere"** | **~30% of essential OGs** |

So the honest answer: **about 70% of our essential OGs have some kind of
functional annotation available somewhere in the broader dataset, but only
about 24-40% can be cleanly categorized into a named functional bucket
without manual curation. The remaining ~30% are "no KEGG/Pfam hit" — these
are usually hypothetical / uncharacterized proteins, which is itself
biologically meaningful: hypothetical proteins essential in only 1-2
organisms are likely the lineage-specific / accessory essentialome.**

## Implication for the bucket-method idea

The clean version of the bucket method works for the **24-40% of
essentials with KEGG/Pfam categorization**. The other ~30% would need:
- per-organism manual curation (expensive)
- or domain-only categorization (less specific)
- or accept the "uncharacterized hypothetical" bucket as a real category

This is a real bound on how complete a function-bucket model can be on
this dataset — not a limitation of our analysis, but a property of the
underlying biology and annotation state.
