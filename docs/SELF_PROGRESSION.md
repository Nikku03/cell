# Self-progression — inferring data we don't directly have

The point of chaining lenses: the pipeline *produces* data present in **no single input**. Measured
live on the sandbox model (PPI + co-essentiality only; Colab adds co-expression + Perturb-seq lenses,
so these numbers grow):

## What the combinations already produced (not in any input dataset)
| new data | count | how |
|---|---|---|
| novel functional links (multi-lens agreement, unannotated) | **1,204** | convergence engine (P2) |
| ...involving a **dark** protein (a new interaction for an uncharacterized gene) | **144** | convergence ∩ dark set |
| dark genes assigned a predicted function | **4,368 / 5,006** | guilt-by-association across lenses |
| regulatory edges after union/causal upgrade | **278,387** (from ~45k curated) | DoRothEA∪TRRUST∪CollecTRI + ReMap/GTEx + causal |

These are genuine inferences — "gene A functionally interacts with dark gene B" and "dark gene B does
X" are outputs, not lookups.

## The self-progression loop (how to keep inferring)
Each new lens raises the confidence and count of the others, in a loop:
1. **Add a lens** (co-expression, Perturb-seq, causal) → more pairs cross the ≥2-lens bar → more novel links.
2. **Novel links → dark-gene function** (a dark gene's convergent neighbors vote its function).
3. **Predicted function → predicted interactions** (a gene placed in a pathway inherits candidate partners).
4. **Predicted interactions become new edges** → feed back into step 1 as a (weaker, flagged) lens.
   *Guardrail:* inferred edges are tagged and **never counted as an independent lens for themselves**
   (else convergence would confirm its own guesses — the circularity we avoided in the KIDINS220 work).

## Data we don't have but CAN infer (chains to build next)
| target we lack | inference chain | confidence |
|---|---|---|
| missing PPI edges | convergence (co-ess + co-expr + shared-complex) minus known PPI → predicted physical edges | medium |
| dark-protein interactions | convergence restricted to dark genes (already 144) + co-expression neighbors | medium |
| dark-protein pathway/reaction | convergent-neighbor pathway vote (Product D) → assign pathway → inherit reactions | medium |
| edge **direction** (who regulates whom) | causal regulome (binding × response) + GTEx trans-eQTL direction | medium-high where Perturb-seq covers the TF |
| context-specific networks | ARCHS4 co-expression per tissue subset (P1 metadata) vs global | medium |
| perturbation response of unperturbed genes | Model 4 (weak, 9.2×/neg-R²) → the transformer is the real attempt | low now, TBD |

## The hard boundary — what NO combination of these datasets can infer
**Kinetics / rate constants / absolute dynamics.** Every lens here is *structural or steady-state*
(who interacts, who depends on whom, who co-varies). None encodes reaction *rates*, binding *affinities*,
or time-resolved *concentrations*. You cannot derive a rate constant from co-expression or co-essentiality
— it is not in the information. This needs **new measurement** (enzyme assays, kinetics, time-course/
live-imaging), consistent with the project goal of "everything traced **except kinetics**." Public
kinetics (BRENDA/SABIO-RK) is sparse and mostly non-human; it does not close this.

## Verdict
The map is **self-progressing on structure/function** (links, functions, directions, context) — each lens
compounds the others — and **saturated on dynamics** without new experiments. The convergence loop is the
engine; kinetics is the wall.
