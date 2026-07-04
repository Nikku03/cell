# Condition-awareness + more dark-gene identification

Two linked expansions: new lenses to identify dark genes, and a condition layer so the cell model can
answer "what happens under condition X".

## Dark-gene identification — two NEW independent lenses
Guilt-by-association (expression/genetic/interaction) fails for dark genes with no signal in the measured
contexts. Two orthogonal lenses reach them:

- **Structure (`compute_structure.py`)** — Foldseek structural search over **AlphaFold** structures: a dark
  gene's *fold* matches a characterized protein → transfer its function. Works even with zero
  expression/interaction data. Verified: a dark gene matching AURKA/PLK1/CDK1 folds → "Cell Cycle Kinase"
  (high conf); a weak-only hit → no call. Runs on Colab (foldseek + the AlphaFold human proteome).
- **Domains (`compute_domains.py`)** — Pfam/InterPro: a dark gene's domain, shared with characterized
  proteins, transfers their majority function (domain-based guilt-by-association). UniProt-sourced.

Both merge into `darkfn` in `build_cell_complete`: they **fill dark genes with no neighbor signal** and
**promote confidence to "high" when they agree** with the expression/genetic call. So a dark gene can now
be reached by *fold* or *domain* when expression says nothing.

**Honest ceiling:** a residue stays dark until an experiment is done — structure/domain help when the
protein resembles something known, not when it is genuinely novel.

## Condition-awareness (`compute_conditions.py`)
The model was steady-state / context-averaged. Now each condition is wired to its molecular **sensor** and
the response is pulled **live from the model's regulatory network** — so "under heat, HSF1 activates its
regulon; here are the responders." Includes the physical conditions you asked for:

| condition | sensor(s) | response genes (live) |
|---|---|---|
| heat / high temperature | HSF1 | 492 |
| cold / low temperature | CIRBP, RBM3 | (few — not TF regulons) |
| **acidosis / low pH** | ATF4, GPR68, LDHA | 424 |
| **high pressure / mechanical** | YAP1, WWTR1 | 13 |
| osmotic stress | NFAT5 | 51 |
| hypoxia / low O2 | HIF1A, EPAS1 | 1,586 |
| oxidative stress | NFE2L2 | 598 |
| nutrient starvation | ATF4, FOXO3, TFEB | 600 |
| ER stress | XBP1, ATF6, ATF4 | 860 |
| DNA damage | TP53 | 2,899 |
| inflammation / immune | NFKB1, STAT1 | 9,523 |
| heavy metal / xenobiotic | MTF1, AHR | 607 |

Each condition also lists its **dark responders** — dark genes in that condition's response set, giving them
a **condition-specific functional hint** (a dark gene responding to heat is likely a chaperone/stress gene).
**This is the link between the two problems:** condition-specific views wake up dark genes that the averaged
model can't place.

## The data-driven upgrade (next, Colab)
The ontology above is grounded in known sensors. The data-driven version — **per-condition co-expression
from ARCHS4** (subset ~1M samples by disease/treatment metadata) and drug response from **LINCS**, disease-vs-
control from **GEO** — would let the model learn *new* condition responses (and rewired networks) rather than
only the curated ones. `compute_coexpr.py` already reads ARCHS4; the condition-stratified version is the
documented next step.

## Honest limit
Condition-awareness here predicts *which genes/modules respond* (transcriptional/network level), **not** the
quantitative, time-resolved dynamics — that needs kinetics (`KINETICS_ASSESSMENT.md`), which the data can't
supply.
