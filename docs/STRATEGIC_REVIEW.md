# Strategic Review — the whole battlefield, and every soldier

*Written after run 4, as the model crossed into quantitative territory (flux, concentration, metabolites,
variants, space). Eagle-eye first, then the zoom-ins, then the multidimensional plays — where data for X
combined with Y and a little external data yields Z, in places we have almost no data for Z directly.*

## 1. The battlefield, tiered by real strength (not by effort spent)

**Strong (dense, measurement-anchored, honest):**
- Core graph — 16.5k genes, 612k reg edges (ReMap now resolves), 191k PPI.
- Causal regulome — 811 signed binding×response edges (finally alive).
- Protein concentration — 16k PaxDb-measured → nM. This is the spine now: everything quantitative hangs off it.
- Convergence + reasoning — 8.6k novel links, 16k derived facts, with a calibrated confidence tier that
  genuinely separates good from bad (high-conf 40% vs low-conf 7%).
- KG link-prediction — AUC 0.95.

**Real but thin:**
- Flux — solves genome-wide, absolute once the medium is upper-bounded (fixed). Objective (biomass-max) is
  the soft spot for human cells.
- Metabolites — 4,142 first-class, but only ~52 concentration-anchored (thermo NET is glycolysis-only).
- Kinetics — 391 measured, rest imputed at ~9.6× (near the experimental noise floor — not fixable by code).
- Attractors — 12 basins, now with real reprogramming signatures. Boolean-crude (no graded activity).
- Space, Variants — new, commercial-safe, queryable, but shallow (compartment-resolution, gene-level).
- dFBA — culture-scale trajectories, honest but qualitative until the medium is fully mapped.

**Broken / missing (the gaps that matter):**
- **Structure (AlphaFold) — NOT LOADING (404).** This is the single highest-leverage hole (see Play F).
- ncRNA = 0 (LncTarD SSL fail), CORUM failed (SSL) — cheap data losses.
- **Validation — still zero wet-lab.** Everything is held-out/computational.

**The one-sentence diagnosis:** *the model is now dense in parts + wiring + static quantities, and thin in
dynamics + structure + validation.* Strategy follows from that: stop adding parts; start (a) fixing the
few cheap data holes that multiply across layers, (b) combining the layers we have into things no single
layer gives, and (c) closing one validation loop.

## 2. Zoom-in: each soldier's specific weakness and its fix

| Layer | Specific weakness | Cheapest real fix | Ripple effect |
|---|---|---|---|
| Structure | AlphaFold URL 404 | working AF proteome mirror / EBI bulk | **multiplies** variants+kinetics+dark (Play F) |
| ncRNA | LncTarD SSL fail | mirror or miRTarBase-only | miRNA/lncRNA regulation lens returns |
| Complexes | CORUM SSL fail | mirror | more complex-based reasoning |
| Metabolite conc | thermo NET = glycolysis only (~52) | eQuilibrator ΔG° for all rxns | conc for thousands of metabolites |
| Flux | biomass-max objective is arbitrary for human | data-driven objective (fit to measured exchange) | more trustworthy flux everywhere downstream |
| Kinetics | 9.6× imputation, near noise floor | can't beat it — route around via in-vivo kcat | in-vivo kcat (built) sidesteps it |
| Attractors | Boolean threshold, no graded state | weighted activity from expression | smoother, quantitative basins |
| Concentration | absolute conversion uses crude cell vol/mass | per-cell-type volume/mass | tightens flux, in-vivo kcat, occupancy |
| Model 4 | pure-expression can't predict perturbation | structural limit — keep honest | — |

## 3. The multidimensional plays — X + Y (+ external) → Z, where Z is data-poor

This is the point. Each layer alone is ordinary; the value is in the combinations, because a virtual cell's
whole thesis is that *integration substitutes for missing direct measurement.*

**Play A — in-vivo kcat = flux ÷ abundance. [BUILT tonight]**
Human in-vivo turnover numbers barely exist (you normally need paired fluxomics+proteomics). We have both →
generate kcat_app for every flux-carrying enzyme + a saturation ratio vs in-vitro. Zero new data. Also a
triple self-consistency check on flux+concentration+kinetics.

**Play B — variant → mechanism → biomarker (validation-shaped).**
`ClinVar pathogenic variant in enzyme E` × `E's reaction + flux role` × `the metabolite E produces` →
predict *which metabolite accumulates* when the variant breaks E. Then cross-check against the known
disease biomarker (HMDB/OMIM metabolite-disease). This converts a pathogenicity *score* into a *mechanism*
**and a retrospective validation** — does the predicted accumulating metabolite match the documented one for
that inborn error of metabolism? We already emit `metabolic_variant_nodes`; this is the next hop.
Layers: variants × flux × metabolites × external(HMDB). **This is the cheapest path to our first validation.**

**Play C — mislocalization disease = variant × space × flux.**
A variant that disrupts a targeting signal moves an enzyme to the wrong compartment → its reaction can't run
there → compartment-resolved flux breaks. We now have HPA multi-localization + compartment-transport topology
+ flux. Z = mislocalization-driven disease, which has very little systematic mechanistic data.

**Play D — condition-specific whole-cell quantitative state.**
Chain one condition through every quantitative layer: `ARCHS4 condition multiplier` → condition-specific
concentration → condition-specific flux (dFBA with condition medium) → condition-specific TF occupancy. Z =
the cell's full quantitative state under hypoxia/heat/etc. — which nobody has, because it requires exactly
this stack. Gated on #1 multipliers (in progress).

**Play E — dark-gene function via the constraint lens.**
LOEUF/essentiality are an *independent* signal we barely use for the 5,006 dark genes. `dark × highly
constrained × essential × convergence-neighbours-in-pathway-P` = a strong functional bet. And `dark ×
constrained × essential × no ClinVar × few pubs` = the highest-value unknown genes in the genome (we already
emit a first cut as `predicted_vulnerable_understudied`). External add-on: cross with a phenotype DB (IMPC
mouse knockouts) to see if the mouse ortholog's KO phenotype hints at the human function.

**Play F — structure as the universal multiplier. [BLOCKED on the 404]**
One dataset (AlphaFold) strengthens three layers at once: (1) map ClinVar/variants onto 3D → which mutations
hit active sites/interfaces (variant *mechanism*, not score); (2) active-site geometry → kcat sanity; (3)
Foldseek → dark-gene function by structural homology. Fixing the AlphaFold source is the highest ROI single
task on the board, precisely because it is a multiplier, not an addition.

**Play G — the self-consistency web (redundancy → error-detection).**
flux, [E], kcat, occupancy now over-determine each other (`flux ≈ kcat·[E]·saturation`; `occupancy =
[TF]/([TF]+Kd)`). Where the identities disagree, we have either a bad imputed value or a genuinely regulated
enzyme — either way, information. in-vivo kcat is the first instance; generalise it into a consistency audit
that flags the worst-fitting nodes for refinement.

## 4. How the fixes ripple (dependency-aware sequencing)

- **Flux→absolute (done)** unlocked: in-vivo kcat, metabolic-variant nodes, dFBA realism. *Already paying off.*
- **AlphaFold fix** unlocks: variant-3D, kinetics-active-site, dark-Foldseek. *3 layers, 1 task.*
- **#1 multipliers** unlock: condition-specific concentration → Play D (the whole condition stack).
- **One validation (Play B)** changes the value story from "plausible" to "calibrated" — worth more than any
  new layer.

## 5. Priority order (attack weakness, leverage strength)

1. **Fix the cheap data holes** — AlphaFold mirror, CORUM mirror, ncRNA mirror. Small effort, Play F unblocks.
2. **Build Play B** (variant→metabolite→biomarker) — our first *retrospective validation*, using data we have.
3. **Build Play D** once #1 fires — condition-specific quantitative state, a genuinely novel artifact.
4. **Generalise Play G** — a consistency audit that makes the whole quantitative stack self-correcting.
5. Everything else (graded attractors, per-cell-type volumes, eQuilibrator ΔG°) is refinement, not frontier.

**The Sun Tzu line:** we've spent the campaign taking territory (layers). The war is won not by taking more,
but by connecting what we hold so the whole moves as one — and by proving, once, that a prediction survives
contact with data we didn't train on. Integration and validation, not addition.
