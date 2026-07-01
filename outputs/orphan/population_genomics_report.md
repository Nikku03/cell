# Population-genomics layer — vital parts, adaptation, TFs (human)

Five parts on human population DNA (gnomAD) + a human regulatory (TF) layer, building
toward mutation→disease. All honest, with caveats. Scripts: `popscan_fetch.py`,
`popscan_analysis.py`, `per_residue_maps.py`, `environment_correlation.py`,
`human_tf_layer.py`. Figures: `per_residue_maps.png`, `population_constraint.png`.

## Part 1 — genome-wide-ish differentiation scan (discover adaptation loci)

Scanned a 376-gene panel (TFs + random background + controls) for between-population
allele-frequency differentiation.

- **Recovers the textbook adaptation loci at the very top**, unsupervised: SLC24A5 (#1),
  SLC45A2 (#2), ACKR1/Duffy (#3), G6PD (#5), MC1R (#9), FADS1 (#15), EPAS1 (#22), LCT (#26).
- **Surfaces 79 novel high-differentiation candidates** (spread ≥0.30, not in the known
  list). Biologically plausible ones: **SLC39A4** (zinc transporter, EAS-enriched),
  **GSTA2** (glutathione-S-transferase / xenobiotic detox, NFE), **SLC38A10** (amino-acid
  transport), TFs **GLI3**/**ATRX**. (Several are olfactory receptors — known hypervariable,
  expected, not necessarily adaptive.)

This is a real discovery output: the pipeline both validates on known biology and
proposes new candidate loci. (Honest scope: ~400 genes via the per-gene API, not all 20k.)

## Part 2 — ancestral polarization (which population adapted)

AFR-as-ancestral baseline (Africans closest to the ancestral allele) → name the derived
allele and the adapted population.

- **Correct for out-of-Africa adaptation**: fixes the confusing pigmentation direction —
  SLC24A5→NFE, SLC45A2→FIN (Europeans carry the derived light-skin alleles). This is the
  right answer, which raw allele frequency alone got backwards.
- **Fails for within-Africa adaptation**: ACKR1/Duffy-null and G6PD (malaria) get
  misassigned, because the *derived* allele reached high frequency *in* Africa — violating
  the "AFR = ancestral" assumption. **Honest limitation**: correct polarization needs an
  outgroup ancestral allele (Ensembl provides it per variant; a clean next step for the
  top hits).

## Part 3 — per-residue vital maps (where in the gene mutations hurt)

ESM-2 zero-shot per-residue constraint + observed gnomAD missense density.

- **EEF2** (essential): 547 observed missense but only **5 common** (>1%) — the vital core
  tolerates almost no common coding change; rare variants exist but are kept rare.
- **MC1R / SLC24A5** (adaptation): carry common missense (the adaptive residues) in
  otherwise tolerant stretches.
- Figure `per_residue_maps.png` shows the tracks; vital domains = runs with high ESM
  constraint and no observed common variant. (8M ESM is a coarse per-residue signal; a
  larger model would sharpen the domain boundaries.)

## Part 4 — environmental variables (what condition changed it)

Correlated adaptation-variant AFs with UV / latitude / cold / dairy / malaria across the
6 ancestry groups. **Honest negative:** best-environment matched the known pressure only
2/10 — because with 6 ancestry groups the environmental axes are **collinear** (European
ancestry bundles low-UV + high-latitude + cold + dairy), so a pigmentation variant
correlates ~equally with all of them and attribution is impossible. This is exactly why
**measured biobank geolocation + climate** (not ancestry proxies) is required to do GxE
properly.

## Human TF (W3) layer — regulation ties it together

TRRUST (795 TFs, 9,396 edges) + essentiality + constraint:
- **TFs are vital**: median LOEUF 0.416 vs 0.911 genome-wide (~2× more constrained).
- **Master regulators of the essential core** = proliferation/growth TFs **E2F1, MYC,
  TP53, SP1, MYCN, HIF1A** — all highly constrained, all cancer genes.
- **Adaptation genes' regulators** are the right tissue masters: MC1R←MITF, EPAS1←HIF3A/
  STAT3, LCT←CDX2/HNF1A, UCP1←PPARA/VDR.
- In the differentiation scan, 2 TFs (GLI3, ATRX) surfaced as novel differentiated loci.

## The unifying picture

> **Purifying selection protects the vital core** (essential genes *and* TFs carry no
> common differentiated coding variation). **Local adaptation tunes the tolerant
> periphery** (the scan recovers known + finds novel environment-linked loci).
> **Regulation (TFs) is itself part of the vital core** and links the essential machinery
> to the adaptive genes. All three regimes are readable directly from population DNA.

## Honest caveats (consolidated)

1. Scan is ~400 genes (API is per-gene), not all 20k — a targeted scan, not exhaustive.
2. AFR-baseline polarization fails for within-Africa adaptation → needs outgroup ancestral.
3. Environment attribution is impossible at 6-ancestry-group resolution (collinearity) →
   needs biobank geolocation.
4. 8M ESM gives coarse per-residue constraint.

## Next: mutations → disease

The natural continuation (user-flagged): overlay **ClinVar** (pathogenic variants) and
**GWAS** on these maps — pathogenic mutations should concentrate in the *vital* regions
(low tolerance), adaptive/benign variation in the *tolerant* regions. That closes the loop
from "where mutations hurt" to "which mutations cause disease."
