# More mutation/disease data — Open Targets expansion

Open Targets integrates GWAS + ClinVar + somatic (cancer) + literature + drug evidence into
one evidence-scored gene→disease resource. Pulled it for 48 disease/master TFs to expand the
disease layer far beyond ClinVar+HPO. Code: `colab/ot_disease_expansion.py`.

## Result — a huge, correct expansion

- **48 TFs → 94,977 gene-disease associations** (HPO gave ~hundreds). Each TF's *top* disease
  is the textbook one — a full validation:

| TF | top disease | TF | top disease |
|---|---|---|---|
| TP53 | Li-Fraumeni | MECP2 | Rett syndrome |
| PAX6 | aniridia | FOXP3 | IPEX |
| PAX3 | Waardenburg | FOXP2 | apraxia of speech |
| SOX9 | campomelic dysplasia | CRX | Leber congenital amaurosis |
| TBX5 | Holt-Oram | NRL | retinitis pigmentosa |
| HNF1A/HNF4A | MODY | WT1 | Denys-Drash |
| GATA4/NKX2-5 | septal defect | TBX1 | 22q11.2 deletion |
| RUNX1 | thrombocytopenia/AML | GLI3 | Greig cephalopolysyndactyly |
| CREBBP/EP300 | Rubinstein-Taybi | TCF4 | Pitt-Hopkins |
| ATRX | alpha-thalassemia/ID | SMAD4 | juvenile polyposis |

## What the evidence types reveal (about mutations)

Aggregate evidence composition (dominant → rare):
`genetic_association 242 · genetic_literature 136 · literature 91 · animal_model 78 ·
somatic_mutation 35 · affected_pathway 31`.

- **These TF diseases are mutation-driven** — genetic association is by far the biggest
  evidence type. TF disease = germline mutation breaking the TF.
- **Cancer/somatic-driven TFs** stand out on somatic_mutation evidence: **TP53, HIF1A, MYC,
  CEBPA, SMAD4, EP300, CTCF, IKZF1** — all bona fide cancer drivers.

## The honest, important negative: 0 of 48 TFs are drug targets

None of these master TFs has approved-drug (`known_drug`) evidence. This is **correct
biology**: transcription factors are the classic **"undruggable" class** — flat surfaces,
no binding pocket, act in the nucleus. You treat TF-driven disease by targeting their
**pathways, partners, or upstream signals**, not the TF itself. (Contrast: metabolic enzymes,
our bacterial targets, are highly druggable.) This is a real constraint for the therapeutic
angle, surfaced directly by the data.

## Where this fits

- The **TF→disease node layer** is now evidence-scored and ~100× richer than HPO/ClinVar.
- Combined with the measured wiring (CollecTRI/DoRothEA) and the coding/non-coding mutation
  breakdown, each disease TF now has: its mutations (coding, DNA-binding-enriched) → its
  measured target regulon → its evidence-scored diseases → (and the sobering fact) no direct
  drug.

## Honest caveats
- Open Targets `n_diseases` counts include weak/literature-only associations; the *top-scored*
  ones are the reliable signal (used above).
- 48 disease/master TFs, not genome-wide (per-gene API calls).
- "undruggable" = no *approved* drug; TF-targeting modalities (degraders, PPI inhibitors) are
  an active frontier, just not yet in the approved-drug evidence.
