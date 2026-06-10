# Phase 1 surprise — bacitracin sensitization via envZ–ompR and pspB in three Shewanella isolates

Built from `outputs/atlas_phase1_A_antibiotic.csv` (top 100 antibiotic-potentiation
candidates ranked by cross-organism consistency × magnitude).

## The observation

Three genes from two distinct envelope-stress circuits, each knocked out in
multiple *Shewanella* isolates, all show strong negative fitness specifically
under bacitracin and (essentially) nowhere else in the cleaned conditional
atlas.

| gene | conditions hit (whole atlas) | bacitracin orgs | median fit | worst fit |
|------|------------------------------|-----------------|------------|-----------|
| envZ | 3 — Bacitracin × 2, Cu × 1   | PV4, SB2B       | −4.65      | −5.20     |
| ompR | 2 — Bacitracin × 2           | PV4, SB2B       | −4.60      | −5.17     |
| pspB | 2 — Bacitracin × 2           | MR1, SB2B       | −4.46      | −5.80     |

Six of seven total atlas hits across these three genes are bacitracin — and
the seventh (envZ × Cu²⁺ in Keio) is a known E. coli envZ stress.

Three independent *Shewanella* isolates are implicated:
- **MR1** — *S. oneidensis* MR-1 (pspB)
- **PV4** — *S. loihica* PV-4 (envZ, ompR)
- **SB2B** — *S. amazonensis* SB2B (envZ, ompR, pspB)

E. coli Keio shows no envZ/ompR/pspB sensitization under bacitracin in the atlas.

## Why this is a "surprise" and not a textbook hit

1. **Bacitracin is a Gram-positive antibiotic.** It binds undecaprenyl-pyrophosphate
   and blocks lipid-II recycling; Gram-negatives are intrinsically resistant via
   outer-membrane exclusion. So a Gram-negative strain becoming bacitracin-sensitive
   on knockout of an envelope regulator is the *interesting* phenotype, not the
   default.

2. **The hit is condition-specific, not "sick cells die."** envZ/ompR/pspB don't
   appear under any other antibiotic in the cleaned atlas — only bacitracin (and
   the one Cu²⁺ Keio hit). Broad envelope-stress vulnerability would produce hits
   across many cell-wall drugs (vanco, β-lactams, fosfomycin, …).

3. **Three independent Shewanella isolates, two distinct stress circuits**
   (envZ–ompR two-component system + Psp response) converge on the same drug
   condition. That kind of mechanistic convergence across genetically distinct
   genomes is what makes the signal believable, not a one-organism artifact.

4. **The conservation prior cannot see this.** A binary essentiality predictor
   trained on sequence + family conservation would say envZ/ompR/pspB are
   "sometimes essential" in baseline growth (family_frac_essential_leakfree
   = 0.29 / 0.74 / 0.60 on the labeled subset) — but provides zero information
   about *which condition* makes them lethal. The conditional atlas is the only
   feature class that exposes the bacitracin specificity.

## Mechanistic hypothesis

EnvZ–OmpR is the canonical Gram-negative two-component system that gates
outer-membrane porin expression in response to envelope cues. PspB anchors
the phage-shock protein response, which is engaged by perturbations to
inner-membrane proton-motive force and envelope integrity.

Hypothesis (testable, single sentence): in *Shewanella*, the EnvZ/OmpR-controlled
porin profile + the Psp response together comprise the envelope program that
keeps bacitracin excluded from its undecaprenyl-pyrophosphate target. Knocking
out any one element (a) admits more drug across the outer membrane (porin
rewiring on ΔompR / ΔenvZ), or (b) removes the inner-envelope buffering that
contains the resulting cell-wall recycling failure (ΔpspB). The cross-isolate
reproducibility says this circuit, not E. coli's, is the right model
organism for envelope-mediated bacitracin tolerance in Gram-negatives.

This is consistent with the recent literature trajectory but goes beyond it:
- 2024 *J. Bacteriol.* (Mitchell et al.) — novel pleiotropic envZ/ompR alleles
  in E. coli alter envelope-stress phenotypes and *vancomycin / rifampin*
  sensitivity. Bacitracin not tested; Shewanella not tested.
- 2011 *PLOS ONE* (Wang et al.) — characterized EnvZ/OmpR in S. oneidensis,
  showed osmotic stress and motility roles. Antibiotics not tested.

So: the regulatory machinery is published, the drug isn't paired with it
anywhere we can find, and our cross-isolate Shewanella signal points exactly
into that gap.

## Druggable framing

- Bacitracin is FDA-approved (topical, intranasal). Combo testing is
  uncomplicated regulatorily.
- EnvZ-family histidine-kinase inhibitors exist (walrycin B and analogs,
  gallotannin derivatives). A WT vs. inhibitor-pretreated Shewanella +
  bacitracin MIC drop is the minimal first wet-lab test.
- Shewanella is an emerging opportunistic pathogen — bacteremia, hepato-biliary,
  soft-tissue. Most clinical isolates retain intrinsic bacitracin resistance
  via OM exclusion. Re-sensitization via envelope-regulator inhibition would
  be a non-obvious adjuvant strategy.

## How we'd be wrong

The honest counter-cases the wet bench would falsify:
- Bacitracin lots from Fitness Browser may contain Zn²⁺ as a co-factor / impurity;
  the envZ × Cu Keio hit raises a "is this a metal stress proxy?" question.
  If isogenic ΔompR + Zn-free bacitracin shows no phenotype, the signal is
  metal-stress, not bacitracin-specific.
- Shewanella isolates share ecology (anaerobic respiration, Fe / Mn redox); the
  three-isolate consistency could be a *Shewanella*-clade artifact rather than
  a Gram-negative-wide adjuvant target. Validation in a second Gram-negative
  clade (Pseudomonas, Vibrio) is the natural next test.

## Why this is the phase 1 deliverable

Of all the cross-organism candidates in the antibiotic bucket, this is the
only cluster that simultaneously satisfies the PHASE1_BRIEF bar:

1. mechanistically explainable (envelope regulator → OM exclusion of a
   Gram-positive drug)
2. not obvious from sequence/conservation (textbook envZ–ompR role is
   osmoregulation, not bacitracin tolerance)
3. literature-checkable and apparently un-paired with bacitracin in
   Shewanella
4. drug-pair framing with an FDA-approved antibiotic

The uvrABCD + recFOR / cisplatin cluster in the same bucket (top of the
shortlist, 4–2 orgs) is the positive-control side of the same scan — it's
textbook NER/HR DNA-damage repair and the cross-organism reproducibility
shows the atlas method is sound. It is not the surprise.

## Phase 2 hook

Phase 2 (two-tower `fit(gene, condition)`) should be evaluated, among other
holdouts, on its ability to recover this cluster when (a) bacitracin is held
out as a condition and (b) Shewanella isolates are held out as organisms.
That is the bar — predicting cross-organism, cross-condition adjuvant pairs
the model has never seen.
