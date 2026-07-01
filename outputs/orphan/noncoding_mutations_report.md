# Accounting for non-coding disease mutations

Categorized every ClinVar **pathogenic** variant across 22 disease genes by the ELEMENT it
hits: coding/ORF vs splice vs regulatory-non-coding (promoter/UTR/intron). Code:
`colab/noncoding_mutations.py`.

## Result (21,738 pathogenic variants)

| element | count | fraction |
|---|---|---|
| coding / ORF (missense, nonsense, frameshift) | 18,656 | **85.8%** |
| splice site | 2,737 | **12.6%** |
| regulatory / non-coding (promoter, UTR, intron) | 323 | **1.5%** |

Genes with the most regulatory/non-coding pathogenic variants: BRCA1 (54), RB1 (45),
MLH1 (39), ATM (26), PAH (25), CFTR (23).

## The honest reading — this is ascertainment, not biology

**The 1.5% regulatory figure massively undercounts real regulatory disease**, for a clear
reason: clinical sequencing has been **exome-based** — it looks at the ORF and splice sites
and barely sequences promoters/enhancers. And non-coding variants are hard to interpret, so
they stay classified "uncertain," not "pathogenic." So ClinVar is coding-biased *by design*.

The famous regulatory disease mutations confirm the gap: **TERT promoter** (cancer — but
somatic, so absent from germline ClinVar → we see 0 for TERT), **F9 promoter** (hemophilia B
Leyden), **HBB promoter** (β-thalassemia), disease **enhancer** mutations — these are real
and textbook, yet barely reach the germline pathogenic set. Our scan sees only the tip.

**Splice (12.6%) is the dominant "beyond-simple-coding" mechanism** — splice-site mutations
don't change the protein sequence directly but wreck the mRNA, and they are well-captured by
exome sequencing, so they show up. Together, ~14% of cataloged disease is already non-(simple-)coding.

## Connection to the regulatory network

Genes carrying pathogenic regulatory variants (BRCA1, RB1, MLH1, ATM, PAH, CFTR) have their
**regulation** disease-linked — a promoter/UTR mutation disrupts a TF binding site, so the
**measured TF→gene edges** into those genes (from the merged CollecTRI/DoRothEA network) are
candidate disease mechanisms. This is the element-level bridge between the mutation and the
regulatory layer.

## What this means for completing the picture

To *properly* account for non-coding regulatory disease you cannot use exome ClinVar — you
need **whole-genome** disease cohorts + **functional/enhancer annotation** (ENCODE cCREs,
enhancer–gene links) + the epigenetic layer. The thinness of the regulatory signal here is
itself the argument for building that layer: it's the least-explored, highest-upside frontier.

## Honest caveats
- ClinVar germline, coding-biased ascertainment (stated above) — not a real disease-mechanism
  frequency.
- Consequence categories from gnomAD's `major_consequence`; "intron_variant" (306 of the 323)
  often means deep-intronic splice/cryptic, not classic enhancer.
- 22-gene panel, well-studied genes.
