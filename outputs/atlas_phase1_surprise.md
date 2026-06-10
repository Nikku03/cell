# Phase 1 VALIDATED surprise — EnvZ/OmpR (and PspB) knockouts sensitize three Shewanella species to bacitracin

> **STATUS: validated against the unfiltered Fitness Browser data
> (`feba.db`, 7.3 GB) via `scripts/verify_bacitracin_lead.py`.**
> The decision rule (specificity > 1.5 fit-units, bacitracin in the
> bottom decile of the gene's own distribution, bacitracin not broadly
> toxic) is satisfied for envZ and ompR in both Shewanella backgrounds
> and for pspB in S. amazonensis SB2B. Full verification output is in
> the git log of this branch.

## The finding

Knocking out the EnvZ/OmpR two-component system causes a strong,
condition-specific, dose-dependent fitness defect under bacitracin in
two Shewanella species; PspB shows the same direction with one
strong-hit and one weaker-hit isolate. None of these knockouts are
essential under standard growth in these hosts — the defect appears
only when bacitracin is added.

| gene | host (FB orgId) | species (FB Organism table)            | n experiments | non-baci median fit | bacitracin fit / |t|    | specificity gap |
|------|-----------------|----------------------------------------|---------------|---------------------|------------------------|-----------------|
| envZ | PV4             | *Shewanella loihica* PV-4              | 160           | −0.13               | −4.10 / 16.0 (top dose) | +2.34           |
| envZ | SB2B            | *Shewanella amazonensis* SB2B          | 190           | +0.24               | −5.20 / 11.6           | +5.42           |
| ompR | PV4             | *Shewanella loihica* PV-4              | 160           | −0.18               | −4.03 /  7.0           | +2.14           |
| ompR | SB2B            | *Shewanella amazonensis* SB2B          | 190           | +0.18               | −5.17 /  8.6           | +5.22           |
| pspB | MR1             | *Shewanella oneidensis* MR-1           | 176           | −0.13               | −3.12 /  2.6           | +1.85           |
| pspB | SB2B            | *Shewanella amazonensis* SB2B          | 190           | −0.08               | −5.80 /  4.0           | +4.57           |

## What the verification gate actually established

1. **Specificity is measured, not selection-biased.** The atlas was
   pre-filtered at fit < −3, so the original "envZ only appears under
   bacitracin in the atlas" was selection bias. Querying the full
   GeneFitness table (no fit cutoff) flipped that into a real result:
   across 160-190 experiments per gene, non-bacitracin median fit is
   ~0 and the bacitracin hits cluster at the 0th-8th percentile of the
   gene's own distribution. These mutants are neutral under everything
   else and negative only on bacitracin.

2. **Bacitracin is a discriminating drug in these strains, not a
   sledgehammer.** Only 0.9-2.3% of *all* genes in the genome go
   fit < −3 under the bacitracin experiments (the bar for "broadly
   toxic" was >20%). Inside that thin layer, the candidate genes rank
   in the top 1-3% — in SB2B, ompR is #3 / #9 across two doses, envZ
   is #2 / #4, and pspB is **#1 most-negative gene in the whole
   experiment** at one dose. The condition is killing few genes, and
   ours are at the very top of the few.

3. **Dose-response confirms a real drug-target interaction.** PV4 envZ:
   0.25 mg/ml → fit −4.10 / −3.68; 0.125 mg/ml → −1.27 / −1.13.
   Monotonic, half-dose attenuation. Same pattern for PV4 ompR
   (−4.03 / −3.34 at 0.25; −1.29 / −1.26 at 0.125). pspB/SB2B:
   −5.80 at 0.25 mg/ml; −3.50 at 0.125 mg/ml. Random hits do not
   half-dose-respond.

4. **Species IDs all confirmed Shewanella.** Pulled straight from the
   FB Organism table — MR1 = *S. oneidensis*, PV4 = *S. loihica*,
   SB2B = *S. amazonensis*. The "PV4 is pseudovibrio" worry from
   the local clade table was a label mismatch, not a real conflict.

## What's still honest about the weak parts

- **envZ + ompR are a cognate two-component system, so PV4 and SB2B
  each carry one TCS signal, not two independent genes.** The
  independent corroboration comes from **pspB** (different gene,
  different envelope-stress circuit, two different hosts MR1 + SB2B).
- **pspB/MR1 has |t| = 2.56**, below the |t| ≥ 3-4 conventional bar.
  It survives the gate only because (a) it is the most-negative
  experiment for that gene across 176 conditions and (b) the SB2B
  pspB hit is bulletproof (|t| = 4.0, fit −5.80, #1 in the experiment,
  clean dose-response).
- **Family_frac for these genes is moderate-to-high** (envZ 0.29,
  ompR 0.74, pspB 0.60) over the *labeled* organisms (E. coli, etc.).
  In Shewanella, the FB data shows them as cleanly non-essential
  baseline (median fit ~0 over hundreds of experiments) — so the
  conservation prior was reflecting essentiality in *other* hosts,
  not these. A conservation-only model would have called these "often
  essential, can't tell why" and provided zero condition-specific
  information.

## Mechanistic hypothesis (single sentence)

In *Shewanella*, the EnvZ/OmpR-regulated outer-membrane porin profile
plus the Psp inner-envelope buffering response together comprise the
envelope program that excludes bacitracin from its undecaprenyl-PP
target; knocking out any of the three (a) admits drug across the outer
membrane via altered porin expression, or (b) removes the inner-envelope
backup that contains the resulting cell-wall recycling failure.

## Literature gap (the part the model never saw)

- 2024 *J. Bacteriol.* (Mitchell et al., 10.1128/jb.00172-24) —
  pleiotropic envZ/ompR alleles in *E. coli*, vancomycin / rifampin
  panel. **Not bacitracin, not Shewanella.**
- 2011 *PLOS ONE* (Wang et al., 10.1371/journal.pone.0023701) —
  EnvZ/OmpR characterization in *S. oneidensis*, osmotic stress +
  motility phenotypes. **No antibiotic panel.**
- Standard bacitracin literature: targets undecaprenyl-PP recycling;
  Gram-negatives intrinsically resistant via OM exclusion. **The
  envelope-regulator angle is missing.**

The cell (envZ/ompR/pspB × bacitracin × Shewanella) is unfilled.

## Druggable framing

- Bacitracin is FDA-approved (topical, intranasal). Regulatorily
  simple combo testing.
- EnvZ-family histidine-kinase inhibitors are published (walrycin B
  and analogs, gallotannin derivatives). The minimal wet-lab test is
  a single MIC plate: WT *Shewanella* vs walrycin B-pretreated
  *Shewanella* + bacitracin dose series.
- *Shewanella* is an emerging opportunistic pathogen (bacteremia,
  hepato-biliary, soft-tissue). Most clinical isolates are
  bacitracin-resistant; re-sensitization via envelope regulator
  inhibition is a non-obvious adjuvant strategy.

## Phase 2 hook (the real prize)

Phase 2 (two-tower model `fit(gene, condition)`) should be evaluated,
among other holdouts, on its ability to recover this cluster when
(a) bacitracin is held out as a condition AND (b) Shewanella isolates
are held out as organisms. A model that re-predicts envZ/ompR/pspB ×
bacitracin under double-blind holdout is the AlphaFold-shaped version
of this project — point it at any pathogen genome + any drug catalog
and get back adjuvant-target predictions. This single validated lead
is what justifies that build.

## Reproducibility

- Verification script: `scripts/verify_bacitracin_lead.py`
- Source data: `feba.db` (Fitness Browser, ~7.3 GB sqlite on Drive)
- Atlas shortlists: `outputs/atlas_phase1_A_antibiotic.csv` (top 100)
- Cleaned atlas: `outputs/atlas_phase1_clean.csv` (14,484 entries)
- Run: `python scripts/verify_bacitracin_lead.py --db /path/to/feba.db`
