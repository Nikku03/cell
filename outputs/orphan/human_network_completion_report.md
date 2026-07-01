# Completing the TF network from disease data

Principle: a disease mutation is a natural perturbation experiment. Genes that cause the
**same disease/phenotype** as a transcription factor are candidate members of that TF's
regulatory module. Use ClinVar/HPO disease data to grow the hand-curated TRRUST network and
to annotate TFs with the diseases they cause. Code: `colab/human_network_completion.py`.

## Inputs
- TRRUST: 795 TFs, 8,427 curated TF→target edges.
- HPO gene→phenotype→disease: 5,268 disease genes, 9,155 diseases (332k links). 361 TFs
  have disease annotations.

## Validation — is "same disease" a real edge signal?

| test | known TRRUST edges | random pairs | ratio |
|---|---|---|---|
| TF & target share a disease | 4.0% | 0.07% | **53×** |
| phenotype Jaccard (mean) | 0.046 | 0.028 | 1.6× |
| edge-recovery AUC (Jaccard) | — | — | 0.530 |

**Honest read:** *sharing a disease* is a **strong but sparse** signal — known regulatory
edges are **53× more likely** to have TF+target cause the same disease than random, but only
4% of edges qualify (most gene pairs aren't both disease-annotated). The *continuous*
phenotype similarity is weak (AUC 0.53) — HPO terms are too broad/sparse for fine edge
prediction. So co-disease adds high-confidence *specific* candidates, not blanket coverage.

## Result — the network grows ~24% with candidate edges

- **2,059 candidate new TF→gene edges** (co-disease + phenotype Jaccard ≥0.20), growing the
  network 8,427 → 10,486 (+24%).
- The candidates form **real disease modules**, e.g.:
  - **Cardiac (dilated cardiomyopathy, ORPHA:154)**: HAND2 ↔ ANKRD1 ↔ FHL2 ↔ TMPO ↔ TAF1A —
    HAND2 is a cardiac TF, ANKRD1/CARP a cardiac stress gene; a plausible cardiac regulatory
    module.
  - **Metabolic (OMIM:601665)**: NR0B2(SHP) → GHRL, AGRP — a metabolic/appetite module.

## TF → disease layer (completed)
Every disease-associated TF is now annotated with the diseases it causes: TP53 (20), PAX6
(15, aniridia/foveal), GATA4 (10, heart defects), SOX9 (9, campomelic dysplasia), TBX5
(Holt-Oram), RUNX1 (leukemia). This is the TF-node disease layer.

## Honest caveats (important)

1. **Co-disease = same module/pathway, which *includes* but isn't *only* regulation.** Two
   genes causing the same monogenic disease may be genetic heterogeneity (independent causes
   in one pathway), not one regulating the other. So candidates are **functional-link
   hypotheses**, not proven regulatory edges, and **direction is not determined**.
2. **Sparse coverage** — only helps where both genes are disease-annotated (disease-gene
   ascertainment bias toward well-studied genes).
3. **Gene-level, not element-level.** The user's distinction (TF-coding vs promoter vs ORF
   mutation) needs *variant-level* annotation. ClinVar does carry non-coding regulatory
   pathogenic variants (5′UTR/upstream/regulatory_region) — flagging those genes as
   "regulation-disease-linked" is the natural element-level extension (not done here).
4. **Epigenetics not yet integrated.** This pass used disease/phenotype only. Adding an
   epigenetic layer needs ENCODE cCREs / Roadmap histone marks / enhancer–gene links
   (GeneHancer, ABC model) + disease methylation (e.g., TCGA) — a larger fetch; documented
   as the next layer, not claimed here.

## Bottom line
Disease data genuinely extends the curated regulatory network — a validated (53× enriched)
signal yields ~2,000 candidate module edges (+24%) and a full TF→disease annotation — but
these are **hypotheses (module co-membership), not confirmed directed regulatory edges**, and
the epigenetic + element-level layers remain to be added. A real completion needs
integrating measured TF-binding (ENCODE/GTRD) + epigenomics on top of this disease scaffold.
