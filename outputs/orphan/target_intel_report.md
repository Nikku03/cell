# Target-intelligence layer — the "scientific completeness" pass

Turns the cell-map engine from a target-*hypothesis* generator into a **de-risked,
prioritized antibacterial target list**. Seven dimensions, on real local data where
possible, each labeled real vs proxy. Output: ranked `target_intel.csv` +
`target_intel.json`. Code: `colab/target_intel.py`.

## The seven dimensions

| # | dimension | how | data | status |
|---|---|---|---|---|
| 1 | **host selectivity** | BLASTP every E. coli protein vs the human proteome; flag human homologs | UniProt human (20,431 prot) + blastp | 🟢 real |
| 2 | **pan-strain essentiality** | for each gene, fraction of feba organisms where the ortholog has a strong fitness defect | feba Ortholog × GeneFitness (~60 orgs) | 🟢 real |
| 3 | **ESKAPE coverage** | essential-reaction sets for real pathogen models; intersect by BiGG id → broad-spectrum | iCN718 *A. baumannii*, iYL1228 *K. pneumoniae*, STM *Salmonella*, iEK1008 *M. tuberculosis* | 🟢 real |
| 4 | **in-host conditions** | FBA essentiality under iron-limited / anaerobic / carbon-limited vs rich | iJO1366 | 🟢 real (coarse) |
| 5 | **escape difficulty** | breadth of essentiality across conditions (broad = harder to escape by environment) | iJO1366 | 🟡 partial |
| 6 | **druggability** | target-class precedent prior from metabolic/core/TF flags + subsystem | flags + iJO1366 | 🟠 proxy (labeled) |
| 7 | **calibrated confidence** | monotone reliability-binning of P(essential) vs Keio truth | learned model + Keio | 🟢 real |

Composite **TargetScore** gates on calibrated essentiality × host-selectivity, and
rewards pan-strain conservation, druggability, and escape difficulty.

## Credibility benchmark — does it recover KNOWN antibiotic targets?

**29 of ~35 validated antibacterial target genes recovered; 25 in the top 10%, 27 in
the top 25%** (of 3,411 genes). The ranking concentrates real drug targets at the top
*without being told what they are*:

```
lpxA 2 · lpxD 3 · folP 13 · glmU 14 · murC 21 · murE 22 · murF 23 · waaA 29
ftsI 38 · lpxC 47 · accD 81 · efp 86 · folC 95 · folA 133 · fabI 135 · fabB 137
```

## Top targets (all bacteria-specific essential enzymes, zero human homolog)

```
 1 coaD  0.86  CoA biosynthesis
 2 lpxA  0.86  lipid A / LPS          (LpxC-class precedent)
 3 lpxD  0.86  lipid A / LPS
 4 murJ  0.86  peptidoglycan flippase (hot target)
 5 dapF  0.81  DAP/lysine -> cell wall
 6 murI  0.80  peptidoglycan
 7 dapB  0.77  DAP/lysine
 8 fabZ  0.77  FAS-II fatty acid
 9 ispH  0.75  MEP isoprenoid         (no human MEP pathway)
10 kdsA  0.71  KDO / LPS
11 fabH  0.67  FAS-II
12 asd   0.64  aspartate semialdehyde
13 folP  0.64  folate                 (sulfonamides)
14 glmU  0.64  UDP-GlcNAc / cell wall
15 hemA  0.64  heme
```

## Cross-pathogen (ESKAPE) broad-spectrum layer

**114 reactions are essential in *all* pathogen models that contain them** (≥2 of
*A. baumannii, K. pneumoniae, Salmonella, M. tuberculosis*) — genuine broad-spectrum
targets: chorismate synthase (CHORS), peptidoglycan (PGAMT, UAAGDS, UDCPDP),
FAS-II (MCOATA), dTDP-rhamnose (TDPGDH). Because BiGG reaction ids are shared, this
intersection is real cross-organism, not a name match.

## Honest limitations (stated plainly)

1. **Absence-based selectivity misses structurally-selective targets.** Fluoroquinolones
   (gyrase, rank 4%), rifampin (RNAP/rpoB, 6%), aminoglycosides (ribosome/rpsL, 13%),
   and tRNA-synthetases (metG, 51%) hit *conserved* enzymes selectively via structural
   differences a homology filter can't see — so they are correctly down-ranked by
   "no human counterpart," but that is a philosophy, not the whole truth. Production
   needs a structure/pocket-level selectivity model, not just BLAST.
2. **In-host conditions are coarse.** FBA medium changes gave only 4 host-conditional
   targets; iron limitation in a stoichiometric model is near-binary (sufficient or
   growth-blocking). Real host-niche essentiality needs transcriptomic/ionome-informed
   conditions, not medium edits alone.
3. **Druggability is a class prior, not pocket detection.** "Metabolic enzyme = 0.9" is
   precedent, not a binding-site assessment. Real druggability needs AlphaFold + pocket
   detection (fpocket) + chemical-matter precedent.
4. **Escape difficulty = condition breadth**, not full resistance modeling (target
   mutability, efflux, horizontal resistance genes). Synthetic-lethal pairing to close
   bypass routes is the next module.
5. Everything inherits the essentiality model's ceiling (calibrated, but ~0.84 AUC).

## What this delivers for the industrial question

This is the concrete answer to "what would it need for pharma": the target list is now
**selectivity-filtered, pan-strain-validated, cross-ESKAPE, condition-aware, druggability-
scored, and calibrated** — and it **recovers known antibiotic targets** as a credibility
check. The remaining gaps (structure-based selectivity, real druggability, resistance
modeling, prospective validation) are named and mostly buildable; the target→compound
chemistry leap remains the hard wall (the earlier toxicity gate returned a definitive
null on database coverage).
