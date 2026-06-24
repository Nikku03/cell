# Metabolic gap-fill rescue of the essentiality soup

**The idea (yours):** the cell is a set of pathways; some have a reaction *hole*
that must be filled for the cell to function. Take the SOUP (genes neither the
transformer nor cross-organism conservation can confidently call), read each
gene's molecular function, and ask: does its reaction fill a non-redundant hole
in the metabolic network? If yes, it's essential by **necessity**, regardless of
conservation.

**Made rigorous:** this is flux-balance gap analysis. We ran it for real on the
genome-scale E. coli model **iJO1366** (2,583 reactions, 1,805 metabolites,
1,366 b-number genes, bundled with cobrapy), joined to the Keio essentiality
screen via our b-number bridge. No proxy, no synthetic data, ~30s runtime.

## What we found

**Network holes:** 878 / 2,583 reactions are blocked (can't carry flux);
127 dead-end metabolites. The gaps are real and abundant.

**Gap-fill demo (literal):** delete porphobilinogen synthase (`b0369`, heme
biosynthesis) → growth **0.982 → 0.000**. A confirmed, non-redundant hole — the
cell cannot make heme any other way, so the gene is essential by necessity.

**The rescue — and why medium matters.** The soup ∩ model = 559 genes
(base rate 9.3% essential). Running gene-deletion FBA:

| medium | rescued | true essential | precision | lift |
|---|---|---|---|---|
| M9 minimal | 92 | 16 | 0.174 | +8pp |
| **Rich (matches Keio screen)** | **29** | **12** | **0.414** | **+32pp** |

The minimal-medium model over-calls amino-acid biosynthesis as essential
(Keio supplies those amino acids), dragging precision down. Supplying amino
acids + nucleosides + vitamins (mimicking Keio's rich medium) lifts rescue
precision from 0.17 to **0.41 — a 4.4× enrichment over the 0.09 soup base rate.**

**The 12 rescued essentials are textbook by-necessity genes** — exactly your
intuition:
- Cofactor biosynthesis (7): porphobilinogen synthase, uroporphyrinogen
  decarboxylase (heme), GTP cyclohydrolase I + riboflavin synthase +
  3,4-dihydroxy-2-butanone-4-P synthase (riboflavin), UbiX prenyltransferase
  (ubiquinone), NAD synthetase, NaMN adenylyltransferase
- Cell envelope (1): glutamate racemase (peptidoglycan D-Glu)
- Lysine/DAP (2): diaminopimelate epimerase, dihydrodipicolinate synthase
  (DAP → peptidoglycan cross-link, not salvageable from medium)
- Lipid (1): lysophospholipid acyltransferase

These are essential even on rich medium because you *cannot* buy heme,
riboflavin, peptidoglycan precursors, or membrane lipids from the broth — they
must be synthesized. The model caught exactly the holes that stay holes.

## The honest limitation: orthogonality is small

Of the 12 rescued essentials, **only 1 was ranked non-essential by BOTH the
transformer and conservation** (b3041, dihydroxybutanone-phosphate synthase:
conservation 0.20, model p 0.31). The other 11 already had conservation ≥ 0.59
— the existing wheels were *almost* calling them; FBA confirmed rather than
discovered. For **E. coli specifically**, conservation already covers nearly all
metabolic essentials, so the orthogonal yield is thin.

Where this pays off more is the **Putida** case (different organism, poorer
conservation match): the precomputed FBA there flagged 17 essentials
conservation missed — a whole pyrimidine-biosynthesis pathway + ferredoxin-NADP
reductase. The orthogonal value of metabolic necessity scales with how *poorly*
conservation transfers to the organism.

## Bottom line

Your idea is correct, it's real, and it works: metabolic gap analysis rescues
soup genes by biochemical necessity at **4.4× the random rate**, and the
rescued genes are interpretable pathway-completion enzymes. The catch is two-
fold: (1) you must match the FBA medium to the screen condition, or precision
collapses; (2) for well-studied organisms conservation already covers most of
what FBA would find — the orthogonal lift is largest exactly where the other
wheels are weakest. It is a genuine third signal, not a redundant one, but its
size depends on the organism.
