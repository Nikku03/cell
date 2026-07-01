# Physical wiring + pathways + metabolic layer (anchored by disease)

Added the missing physical/signaling and pathway layers: STRING physical protein-protein
interactions + Reactome pathways, validated and anchored by disease (mutation → disease →
pathway). Code: `colab/physical_pathway_layer.py`.

## Physical wiring (STRING physical, high-confidence ≥700)

Essential proteins are the **hubs** of the physical interaction network:

| gene group | mean physical interactions |
|---|---|
| **essential (CEG)** | **48.4** |
| disease genes | 16.7 |
| TFs | 17.1 |
| all | 16.1 |
| non-essential | 5.1 |

Essential proteins have ~10× more interactions than dispensable ones — the classic
**centrality-lethality** rule (hubs are essential), now confirmed on our set. This is an
*orthogonal* essentiality signal (topology of the physical network), independent of
constraint/conservation/structure.

**Validation — physical interaction ↔ shared disease: 54×.** Physically-interacting
disease proteins share a disease 54× more often than random pairs — the same strength as
the TF co-disease signal (53×). So the physical wiring is real disease-module structure,
anchored by mutation data exactly as requested.

## Pathway layer (Reactome, 2,311 pathways, 11,272 genes)

**Mutation → disease → pathway resolves correctly** for every test gene:
PAH → phenylketonuria / phenylalanine metabolism · G6PD → pentose phosphate pathway ·
MLH1 → mismatch repair · LDLR → LDL clearance · RET → RET signaling · GATA4 → cardiogenesis ·
HBB → heme. The disease pins the gene to its pathway — the exact "we know what mutated,
what disease, what pathway" chain.

**505 pathway-level disease modules** (≥3 genes in a pathway sharing a disease): Complex I
biogenesis (mitochondrial disease), rRNA processing (ribosomopathies), NMD/translation.
These are pathways where genetic heterogeneity converges — multiple genes, one disease.

## Metabolic layer

Reactome metabolic genes (587) ∪ Human-GEM (2,848) = **3,085 metabolic genes**. Only
**13/684 essential genes are metabolic** — re-confirming (a 4th time) that human
essentiality is overwhelmingly *non*-metabolic (translation/splicing/cell-cycle), unlike
bacteria. The metabolic layer is real but is not where human essentiality lives.

## Where this leaves the cell layout

We now have, for the human cell: parts + essentiality + structure + regulatory wiring +
epigenetics + **physical/signaling interactions** + **pathways** + **metabolism** +
mutation→disease. The physical + pathway + metabolic layers were the main structural gaps
from the synthesis. What remains for a runnable kinetics-free map is chiefly **cell-type
specificity** (which subset is active per cell type — single-cell atlases) and **integration**
into one perturbable object.

## Honest caveats
- STRING "physical" ≥700 is high-confidence but includes some inferred interactions (not all
  are experimentally-validated direct binding).
- PPI and pathways are **cell-type-agnostic aggregates** — the same superposition limitation
  as the TF network; a cell-type-specific active interactome needs proteomics/expression per
  tissue.
- Reactome coverage is curation-limited; disease→pathway depends on annotation completeness.
- Human metabolic layer remains weak for *essentiality* (13/684) — expected, not a defect.
