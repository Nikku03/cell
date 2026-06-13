# Bacitracin lead — EnvZ/OmpR (and PspB) sensitize Shewanella to bacitracin
## A PI-to-postdoc handoff document

**Status:** Validated lead, ready for a single MIC-plate wet-lab experiment.
**Validation source:** Unfiltered Fitness Browser data (`feba.db`), audited
against the original conditional vulnerability atlas (`outputs/atlas_phase1_clean.csv`).
**Reproducibility script:** `scripts/verify_bacitracin_lead.py`.

---

## The single-sentence finding

Knocking out the EnvZ/OmpR two-component system causes a strong,
condition-specific, dose-dependent fitness defect under bacitracin in two
*Shewanella* species (*S. loihica* PV-4 and *S. amazonensis* SB2B); PspB
(phage-shock response) shows the same direction with one strong (SB2B) and
one weaker (MR-1) signal — and this pairing of envelope regulators with a
Gram-positive antibiotic has not, to our reading, been paired in
*Shewanella* in the published literature.

---

## The measurement evidence (raw, unfiltered, validated)

Six (gene, organism, bacitracin) rows from `feba.db` after the unfiltered
verification:

| gene | host (FB orgId) | species | n exps | non-baci median fit | bacitracin fit / \|t\| | specificity gap |
|------|------|------|------|------|------|------|
| envZ | PV4  | *S. loihica* PV-4         | 160 | −0.13 | **−4.10 / 16.0** (top dose) | **+2.34** |
| envZ | SB2B | *S. amazonensis* SB2B     | 190 | +0.24 | **−5.20 / 11.6**            | **+5.42** |
| ompR | PV4  | *S. loihica* PV-4         | 160 | −0.18 | **−4.03 /  7.0**            | **+2.14** |
| ompR | SB2B | *S. amazonensis* SB2B     | 190 | +0.18 | **−5.17 /  8.6**            | **+5.22** |
| pspB | MR1  | *S. oneidensis* MR-1      | 176 | −0.13 | **−3.12 /  2.6** (sub-bar)  | +1.85 |
| pspB | SB2B | *S. amazonensis* SB2B     | 190 | −0.08 | **−5.80 /  4.0**            | **+4.57** |

Read: across 160-190 conditions per gene, the gene is essentially neutral
(median fit ≈ 0). Bacitracin alone drives the fitness defect.

### Why this is a lead and not a measurement artifact

Four independent properties hold simultaneously (recorded by
`verify_bacitracin_lead.py` and re-tested unfiltered):

1. **Condition specificity.** Across 160–190 experiments per gene, non-baci
   median fit is ≈ 0; bacitracin hits cluster at the 0th–8th percentile of
   the gene's own distribution.

2. **Bacitracin is not broadly toxic in these strains.** Only 0.9–2.3% of
   *all* genes have fit < −3 under the bacitracin experiments. ompR is
   #3/#9 most-negative on the entire chromosome (SB2B), envZ is #2/#4, and
   pspB is **#1 most-negative gene in the whole experiment** at one dose.

3. **Dose response.** PV4 envZ at 0.25 mg/mL → −4.10/−3.68; at 0.125 mg/mL →
   −1.27/−1.13. Same monotonic pattern for ompR and pspB/SB2B. Random hits
   do not half-dose-respond.

4. **Species IDs confirmed in the FB Organism table.** MR-1 = *S. oneidensis*,
   PV-4 = *S. loihica*, SB2B = *S. amazonensis*.

---

## Mechanistic hypothesis (testable, one sentence)

> In *Shewanella*, the **EnvZ/OmpR-controlled outer-membrane porin profile**
> plus the **Psp inner-envelope buffering response** comprise the envelope
> program that *excludes* bacitracin from its undecaprenyl-PP target;
> knocking out any one element (a) admits more drug across the outer
> membrane (porin rewiring via ΔompR / ΔenvZ), or (b) removes the
> inner-envelope buffering that contains the resulting cell-wall recycling
> failure (ΔpspB).

This is a Gram-negative envelope-permeability story applied to a
Gram-positive-targeting antibiotic. That framing is what makes it
unusual — and testable.

## Literature gap

- **2024 *J. Bacteriol.*** (Mitchell et al., 10.1128/jb.00172-24) —
  pleiotropic envZ/ompR alleles in *E. coli*, vancomycin / rifampin panel.
  **Not bacitracin, not Shewanella.**
- **2011 *PLOS ONE*** (Wang et al., 10.1371/journal.pone.0023701) —
  EnvZ/OmpR characterization in *S. oneidensis*, osmotic stress + motility
  phenotypes. **No antibiotic panel.**
- Standard bacitracin literature: targets undecaprenyl-PP recycling;
  Gram-negatives intrinsically resistant via OM exclusion. **The
  envelope-regulator angle is missing.**

So the cell (envZ/ompR/pspB × bacitracin × *Shewanella*) is unfilled in
our reading.

---

## The phase 2 predictor context (honest framing)

When *Shewanella* is held out as an organism, the trained Phase 2
predictor ranks the envZ/ompR cluster in the **top 4-8% of all
(gene × bacitracin) pairs** for SB2B (with the kernel feature). Concretely:

| gene/org   | pctile of all candidates | calibrated probability |
|------------|---|---|
| envZ / SB2B | top ~5.9% | ~0.69 |
| ompR / SB2B | top ~2.6% | ~0.85 |
| pspB / SB2B | top ~35%  | ~0.13 |

**Note this is propagation, not de-novo discovery.** With PV4 + MR1 in
training, the model leverages that the (envZ-OG × bacitracin) pair is a
hit in those Shewanella relatives. A *leave-one-clade-out* test (hold out
ALL Shewanella) is the honest de-novo ceiling and is not yet run.

This honesty is exactly the cascade's design point: the lead is
**measurement-backed** (Phase 1 atlas + unfiltered verification), and the
**Phase 2 ranking is consistent**, but the lead does not rely on Phase 2
for its primary evidence.

---

## The wet-lab experiment to do (one MIC plate)

**Question:** Does pharmacologically inhibiting envZ kinase activity
re-sensitize WT *Shewanella* to bacitracin (i.e., does the genetic
phenotype phenocopy with a small molecule)?

**Strains:** *S. oneidensis* MR-1 and *S. amazonensis* SB2B
(WT + ΔenvZ knockout for comparison if available).

**Drug pair:**
- bacitracin (FDA-approved; standard MIC concentrations 1–256 μg/mL).
- walrycin B (published EnvZ-family histidine-kinase inhibitor; 1–25 μM)
  or a gallotannin derivative.

**Plate layout:** standard 96-well checkerboard MIC plate:
- rows = bacitracin dose (2-fold dilutions across 8 columns).
- columns = walrycin B dose (2-fold dilutions across 12 rows).
- read OD₆₀₀ at 12 h, 24 h.

**Read-out:**
- **MIC drop of bacitracin in the presence of subinhibitory walrycin B**
  → mechanism confirmed (envZ inhibition phenocopies ΔenvZ + bacitracin).
- If MIC is unchanged → either the inhibitor doesn't engage envZ in
  *Shewanella*, or the genetic phenotype isn't pharmacologically tractable;
  honest negative.

**Expected magnitude (from the genetic data):** bacitracin MIC should drop
≥ 4-fold in the presence of walrycin B, paralleling the −5 fitness defect
under bacitracin in ΔenvZ.

---

## Counter-cases the wet bench would falsify

The earlier reverification flagged these honestly; record them so the
postdoc tests them:

1. **Zn²⁺ co-factor / impurity?** The envZ × Cu²⁺ Keio hit in the atlas
   raises a "is bacitracin's signal really a metal-stress proxy?" question.
   Use Zn-free bacitracin if MIC drops disappear, the signal is
   metal-stress not bacitracin-specific.

2. ***Shewanella*-clade artifact?** All three isolates share marine /
   anaerobic ecology. The de-novo signal could be a *Shewanella* trait
   rather than a Gram-negative-wide adjuvant target. Validate in a
   second Gram-negative clade (*Pseudomonas* or *Vibrio*) before claiming
   broader.

3. **Walrycin B specificity.** Walrycin B inhibits multiple histidine
   kinases. If MIC drops with walrycin B but not with envZ-specific
   genetic knockout, the signal is pleiotropic kinase inhibition, not
   envZ specifically.

---

## If the experiment succeeds

**Direct value:** a Gram-negative-pathogen adjuvant story: combine
bacitracin (FDA-approved, off-patent, cheap) with an envZ-class inhibitor
to re-sensitize a clinically important opportunistic pathogen (Shewanella
is an emerging cause of bacteremia, hepato-biliary infection, and
soft-tissue infection) to a topical antibiotic it currently shrugs off.

**Indirect value:** validates the project's broader claim — a
conditional-vulnerability atlas + cross-organism propagation can surface
mechanistically novel, druggable, publishable leads. One MIC plate
converts the entire predictor effort into a real biological finding.

---

## How to reproduce the lead computationally

```bash
# clone repo
git clone -b claude/vectorize-gex-propensity-NRqBW https://github.com/Nikku03/cell.git

# the validated phase-1 surprise document
cat outputs/atlas_phase1_surprise.md

# the unfiltered re-verification script
python scripts/verify_bacitracin_lead.py --db /path/to/feba.db
# -> prints per-experiment fit, |t|, specificity gap, condition background
```

---

## Reproducibility receipts

- `outputs/atlas_phase1_surprise.md` — the original validated surprise doc
- `outputs/atlas_phase1_clean.csv` — cleaned conditional-vulnerability atlas
- `outputs/atlas_phase1_A_antibiotic.csv` — top-100 antibiotic shortlist
- `scripts/verify_bacitracin_lead.py` — the unfiltered-data verification
- `feba.db` (Fitness Browser) — primary source data
