# How the datasets interconnect to produce novel discoveries

Not "what each dataset contains" (see `DATA_CATALOG.md`) — this is how *combining* them yields
discoveries **no single dataset holds**, and why those discoveries are more than interpolation.

## The core principle

Two facts drive every engine below:

1. **Novelty lives where independent assays agree but annotation/literature is silent.** If a gene
   pair is linked by physical binding *and* genetic co-dependency *and* transcriptional response —
   three assays that can't share an artifact — the link is almost certainly real. If no pathway,
   complex, or paper connects them, it's *novel*. Convergence is simultaneously the discovery **and**
   its own first validation. (This is exactly the KIDINS220–XPR1 pattern — see `NOVELTY_STRESS_TEST.md`.)
2. **Measured → unmeasured extrapolation.** Train where we have ground truth, predict where we don't,
   rank by confidence, validate on held-out data. (Model 1 for essentiality, Model 4 for perturbation.)

## We have 7 *independent* lenses on "are two genes functionally related?"

| lens | evidence type | datasets |
|---|---|---|
| physical | do the proteins touch? | STRING, BioPlex, OpenCell, HuRI, Complex Portal |
| genetic | do cells need them together? | DepMap co-essentiality / synthetic-lethal |
| transcriptional | does perturbing one move the other's program? | Perturb-seq, LINCS, **GEO co-expression** |
| regulatory | does one control the other? | CollecTRI, ReMap (binding), GTEx (genetic) |
| metabolic | do they act on shared metabolites? | Human-GEM |
| spatial | are they in the same compartment? | localization |
| annotated | are they in the same pathway? | Reactome, GO |

Because these are **orthogonal** (a Y2H artifact can't also be a CRISPR artifact), agreement across
≥3 with **disagreement from the "annotated" lens** is the signature of a novel, real relationship.

## Discovery engines (each = a specific dataset combination → a novel, testable output)

### 1. Convergent co-dependency → novel synthetic-lethal / drug-combination pairs
**Inputs:** DepMap co-essentiality + PPI (all four) + Perturb-seq/LINCS + Reactome (as the *filter*).
**Mechanism:** find gene pairs strong in ≥2 measured lenses, in *different* pathways, with *no* direct
PPI → candidate parallel-pathway buffering (classic SL). **Novel output:** ranked SL pairs absent from
literature → drug-combination hypotheses. **Validation:** held-out DepMap + a targeted CRISPR double-KO.

### 2. Dark-gene function by convergence → annotate the unknown 30%
**Inputs:** dark genes + Perturb-seq + LINCS + co-essentiality + HuRI/BioPlex + ReMap (its regulators)
+ compartment. **Mechanism:** for each unannotated gene, take the majority pathway of its neighbors
across *every* lens; keep only where lenses **agree**. **Novel output:** confident function calls for
genes with zero annotation, ranked by cross-lens agreement (stronger than the current single-lens
`darkfn`). **Validation:** does a KO (Perturb-seq/GEO) move the predicted pathway?

### 3. Causal regulome → upgrade "binding" to "regulation"
**Inputs:** ReMap (TF binds gene) + Perturb-seq/LINCS (TF-KO actually changes gene) + GTEx (genetic).
**Mechanism:** ReMap binding is a *candidate*; keep an edge only if perturbing the TF measurably moves
the target. **Novel output:** the first *causally-supported* human regulatory network — including
TF→target edges in **no curated database** (binding + response but never annotated). **Validation:**
built-in (the perturbation *is* the causal test).

### 4. Context-conditioned targets → precision-oncology hypotheses
**Inputs:** DepMap essentiality + CCLE expression/mutation (biomarkers) + HPA/Tabula cell-type
expression + **GEO** disease-vs-normal. **Mechanism:** a gene essential only in *some* lines → learn
the expression/mutation context that predicts dependence → map which tissues/patients have that
context. **Novel output:** "target X matters *when* context C" + its biomarker → who to treat.
**Validation:** GEO disease cohorts where context C holds should show the dependency's signature.

### 5. Metabolic bypass vulnerabilities → metabolic drug combinations
**Inputs:** Human-GEM + DepMap co-essentiality + CCLE expression. **Mechanism:** co-essential enzyme
pairs catalyzing *different* reactions = redundant metabolic routes; find contexts (CCLE/GEO) where one
route is already off. **Novel output:** enzyme pairs that are synthetic-lethal only in a metabolic
context → conditional metabolic targets. **Validation:** flux modeling + double-KO.

### 6. Cross-tissue disease mechanism → multi-organ hypotheses (tissue model)
**Inputs:** tissue ligand-receptor + endocrine axes + NicheNet/SIGNOR downstream + disease links +
**GEO** multi-tissue expression. **Mechanism:** signal from tissue A → receptor in tissue B → downstream
TFs → if those targets are disease genes, that's a cross-organ mechanism. **Novel output:** e.g. how a
secreted liver factor drives a cardiac program. **Validation:** GEO datasets of tissue B under tissue-A
perturbation should show the predicted downstream shift.

### 7. Virus-host safety-window targets
**Inputs:** HIV-host interactions + PPI + DepMap essentiality + Perturb-seq. **Mechanism:** host factors
HIV depends on that are *buffered* in the host (dispensable) = antiviral targets with a therapeutic
window. **Novel output:** host-directed antiviral candidates. **Validation:** infectivity assay on KO.

## Where the GEO data plugs in

Folder `expression_geo` → most likely bulk/array/scRNA **expression**. Its value depends on content:

- **If condition/tissue expression compendium** → build a **co-expression network** = an 8th, fully
  independent functional lens to triangulate with PPI/co-essentiality (Engines 1, 2). Also the primary
  **external validation substrate**: our model predicts "KO of X moves pathway Y"; a GEO cohort where X
  is lost/low should show pathway Y shifted. External validation is what turns a prediction into a claim.
- **If any series is a perturbation** (shRNA/CRISPR/drug/OE) → feeds **Model 4** directly (more
  perturbation coverage) *and* validates it (does the GEO-measured KO signature match Model 4's
  prediction for that gene?).
- **If disease-vs-control** → the substrate for Engine 4 (context) and Engine 6 (cross-tissue).

*(Exact wiring depends on the actual GSE accessions — pending Drive access / the accession list.)*

## Honest ranking

| engine | novelty potential | tractable now? | needs |
|---|---|---|---|
| 3 Causal regulome | **high** — first causal human regulome | yes | ReMap + Perturb-seq/LINCS (have) |
| 2 Dark-gene convergence | high — annotates 30% unknown | yes | already partly built (`darkfn`) |
| 1 Convergent SL | high — drug combos | yes | shortlist scan (offered) |
| 4 Context targets | high — precision medicine | partly | GEO/CCLE context labels |
| 6 Cross-tissue | medium-high | partly | NicheNet (next session) + GEO |
| 5 Metabolic bypass | medium | yes | Human-GEM + DepMap (have) |
| 7 Virus safety-window | medium | yes | have |

**The unifying build:** a *convergence engine* that scores every gene pair across all 7–8 lenses and
surfaces the high-agreement / zero-annotation cases. Every engine above is a view on that one score.
That is the single most novelty-dense thing we can build from what we already have — and GEO adds the
8th lens plus the external validator that makes the outputs credible.
