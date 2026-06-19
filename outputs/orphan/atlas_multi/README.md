# Essentiality-Prediction Atlas (multi-organism)

## 1. What this is

A 6-organism, 23,536-gene atlas of bacterial gene-essentiality predictions. Each gene is assigned to one of five tiers (confident essential, confident non-essential, rogue suspect, conditional suspect, unresolved) based on the agreement of up to ten independent evidence channels. The point of the atlas is not to predict every gene — published predictors top out around AUC 0.81 cross-organism but only reach ~28% coverage at precision >=0.70, and they collapse on the "rogue zone" of essential-but-not-conserved genes (about 14% of all essentials, where every $0 sequence feature including ESM-2 hits zero R@P30 cross-clade). The atlas instead predicts where we can, explicitly abstains where we can't, and attaches an audit trail of evidence channels to every call.

## 2. Headline numbers

**Atlas total: 23,536 genes across 6 organisms. 58% confident coverage at 90.3% precision.** (v4)
(Published cross-organism predictors reach only ~28% coverage at precision >=0.70. The jump
from an earlier 34%/86% came from an ortholog-transfer vote channel: a gene whose
orthologous group is measured non-essential in >=2 OTHER organisms with zero essential
votes is called at ~98% precision, even when its own genomic features are ambiguous or
structurally rogue-suspicious. This clean measured vote OVERRIDES the structural quarantine.
v4 then added a dN/dS selection channel: strong purifying selection (dN/dS <= 0.10 vs sister
orthologs) paired with conservation or an ortholog vote promotes a gene to confident
essential. This grew the essential tier by ~580 calls, concentrated in the Ralstonia strains
that have close sisters with many same-length orthologs.)

Per-organism:

| organism            |    n | confident_ess | confident_noness | rogue_suspect | conditional | unresolved | coverage | precision |
|---------------------|-----:|--------------:|-----------------:|--------------:|------------:|-----------:|---------:|----------:|
| RalstoniaGMI1000    | 4403 |           422 |             2501 |           457 |         130 |        893 |      66% |       88% |
| RalstoniaBSBF1503   | 4431 |           454 |             2385 |           457 |         134 |       1001 |      64% |       91% |
| RalstoniaPSI07      | 4298 |           644 |             1923 |           483 |         123 |       1125 |      60% |       92% |
| Dda3937             | 3926 |           218 |             1655 |           358 |         115 |       1580 |      48% |       97% |
| HerbieS             | 3847 |           302 |             1436 |           228 |         110 |       1771 |      45% |       94% |
| Magneto             | 2631 |           191 |             1505 |            99 |          69 |        767 |      64% |       78% |

**Cross-organism consistency.** Of 2,584 confident calls on genes that share an orthologous group across organisms, 2,569 agree (99.4%). The atlas does not just have surface-level coverage — independently called organisms converge on the same gene-level verdict. 10,517 genes (45%) carry a cross-validated confident call at 92.5% precision.

**Precision stratifies by per-gene confidence (number of independent channels firing in the same direction):**

| channels firing | precision |
|----------------:|----------:|
|               1 |     83.1% |
|               2 |     91.0% |
|               3 |     91.2% |
|               4 |     95.4% |
|               5 |     97.3% |

A user who wants ~95%-precision calls can filter `confidence >= 4`; a user who wants maximum recall can accept `confidence >= 1`.

**Coverage cost (honest).** The ortholog-vote override reclassifies most rogue/conditional
suspects as confident non-essential at 98% precision, but the ~2% it gets wrong are true
conditional essentials (essential here, non-essential in the reference panel). Rogue-zone
protection therefore drops from ~57% to ~46%: roughly half the true conditional essentials
are now absorbed into non-essential calls. This is the deliberate coverage/safety trade.
For a conservative atlas that keeps the full rogue quarantine, ignore the `N_ortho` override
(rebuild with `promote_N` gated behind `not quarantine`).

## 3. How to read a row

Each CSV has one row per gene with these columns:

`locus_tag, product, tier, predicted_call, confidence, evidence, measured_essential, ess_score, noness_score, conservation, n_paralogs, gene_len_aa, leading, mobile_dist_kb, n_sisters, max_sisters, cross_validated`

`evidence` is a pipe-separated list of channel codes (see Section 4). `confidence` is the count of channels firing in the predicted direction. `cross_validated` is True when the cross-organism leave-one-out score agrees with the call.

**Example A — confident essential (RalstoniaGMI1000, RS_RS25220, "50S ribosomal protein L34")**

```
tier=CONFIDENT_ESSENTIAL  predicted_call=essential  confidence=3
evidence=E_score|E_cons_core|E_geometry
ess_score=0.97  noness_score=0.02  conservation=1.0  gene_len_aa=45  leading=1
```

All three essential channels fire: the cross-organism LOO score is above the high-precision threshold (E_score), the gene is universally conserved (E_cons_core), and it is a short gene on the leading strand (E_geometry, the canonical essential geometric signature). No phenotype-quarantine flag fires. This is the kind of call the atlas exists to make confidently.

**Example B — confident non-essential (RalstoniaGMI1000, RS_RS00025, "low MW protein tyrosine phosphatase family protein")**

```
tier=CONFIDENT_NONESSENTIAL  predicted_call=non-essential  confidence=2
evidence=E_geometry|N_score|N_pangenome_some
noness_score=0.76  ess_score=0.25  conservation=0.0  n_sisters=2  max_sisters=3
```

The cross-organism score is on the non-essential side of the threshold (N_score), and the ortholog is absent in at least one sister strain (N_pangenome_some) — meaning at least one closely related genome has discarded this gene, which is positive within-species dispensability evidence rather than mere low-conservation noise. `E_geometry` also fires (leading-strand short-ish gene), but because the dominant signal points non-essential and no phenotype quarantine flag fires, the row resolves to CONFIDENT_NONESSENTIAL.

**Example C — rogue suspect (RalstoniaGMI1000, RS_RS00060, "patatin-like phospholipase family protein")**

```
tier=ROGUE_SUSPECT  predicted_call=unknown  confidence=0
evidence=E_geometry|N_score|P_rogue
conservation=0.0  n_paralogs=0  n_sisters=3  max_sisters=3
```

This gene matches the structural rogue-essential profile (P_rogue): nearly no conservation outside the genus, no paralogs, but present in every sister strain — i.e. retained where it matters but invisible to deep-homology features. The atlas explicitly refuses to predict it. The N_score and E_geometry channels conflict, but P_rogue overrides both and routes the gene to quarantine. Calls in this tier should be settled by experiment, not by sequence features.

## 4. Evidence channels reference

Ten binary channels, in three groups. The five non-essential channels are deliberate **inversions** of the protection-panel findings about essential genes — flipping each essential-gene marker into a positive marker for dispensability.

### Essential (3)

| code            | fires when                                                                                                                                                          |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `E_score`       | Cross-organism LOO essential-prediction score >= the per-organism high threshold, calibrated at precision >= 0.9 on held-out training organisms.                    |
| `E_cons_core`   | Conservation >= 0.5 across the reference panel (gene family deeply retained).                                                                                       |
| `E_geometry`    | Leading-strand AND short gene — the essential geometric signature (avoids head-on replication-transcription collisions, low cost to express).                        |
| `E_ortho`       | Ortholog-transfer vote: the gene's OG is measured ESSENTIAL in >=2 OTHER organisms with zero non-essential votes. Independent of this gene's own features.            |
| `E_dnds`        | Strong purifying selection: dN/dS <= 0.10 vs the closest sister ortholog (synonymous rate dS > 0.01). Promotes to essential only when paired with E_cons_core or E_ortho. Sparse for distant-sister organisms (Dda3937/HerbieS/Magneto). |

### Non-essential (6, inverted from protection-panel research)

| code                  | fires when                                                                                                                                                              | inverted from                                                                 |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `N_score`             | Low cross-organism essential score (below the symmetric non-essential threshold).                                                                                       | mirror of `E_score`                                                           |
| `N_pangenome_strong`  | Ortholog ABSENT in ALL sister strains for this organism.                                                                                                                | the only channel that is positive within-species dispensability, not a mirror |
| `N_pangenome_some`    | Ortholog absent in at least one sister strain.                                                                                                                          | weaker form of `N_pangenome_strong`                                           |
| `N_redundancy`        | `n_paralogs >= 4` AND conservation < 0.5 — redundancy-buffered, loss of one copy compensable.                                                                            | "essentials have few paralogs"                                                |
| `N_mobile`            | Within 5 kb of a mobile element (HGT / accessory belt).                                                                                                                 | "essentials avoid mobile elements"                                            |
| `N_long_lagging`      | Long gene on lagging strand AND conservation < 0.3.                                                                                                                     | "essentials avoid head-on collisions, especially when long"                   |
| `N_ortho`             | Ortholog-transfer vote: the gene's OG is measured NON-ESSENTIAL in >=2 OTHER organisms with zero essential votes. ~98% precise; OVERRIDES the rogue/conditional quarantine. | mirror of `E_ortho` — the single biggest coverage lever                    |

### Phenotype-required quarantine (2)

These do not predict — they **veto** confident calls and route the gene to a suspect tier.

| code             | fires when                                                                                                                              | meaning                                                                |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `P_rogue`        | conservation < 0.1 AND `n_paralogs == 0` AND present in ALL sisters.                                                                    | Structural rogue-essential profile — invisible to homology features, but retained where it counts. Sequence features cannot resolve this. |
| `P_conditional` | conservation < 0.3 AND long-on-lagging-strand AND present in ALL sisters.                                                                | Niche / virulence / conditional-essential candidate.                   |

## 5. Tier rules

Applied in this order; the first match wins (phenotype vetoes have priority over both essential and non-essential resolutions for the tiers below):

```
promote_E = (E_ortho AND (E_cons_core OR E_geometry))     # validated >=0.89 precision
            OR (E_dnds AND (E_cons_core OR E_ortho))       # v4: dN/dS, ~0.85 precision
promote_N = N_ortho                                       # validated ~0.98 precision

CONFIDENT_ESSENTIAL     = (E_score OR promote_E) AND NOT P_rogue
CONFIDENT_NONESSENTIAL  = promote_N                                          # overrides quarantine
                          OR ((N_score OR N_pangenome_strong) AND NOT P_rogue AND NOT P_conditional)
ROGUE_SUSPECT           = P_rogue        (and no clean N_ortho vote)
CONDITIONAL_SUSPECT     = P_conditional  (and no clean N_ortho vote)
UNRESOLVED              = everything else
```

The two `*_ortho` channels are what lifted coverage from 34% to 55% at higher precision. A
clean measured non-essential ortholog vote (`promote_N`) is trusted even over a structural
rogue/conditional quarantine, because it is right 98% of the time; the residual 2% error is
the conditional-essential population (see the coverage-cost note in section 2).

`predicted_call` is `essential` / `non-essential` / `unknown` according to the tier. `confidence` counts the channels firing in the predicted direction (so 0 for the two SUSPECT tiers and for UNRESOLVED rows where channels disagree).

## 6. Sister-strain mapping

The pangenome channels (`N_pangenome_strong`, `N_pangenome_some`, and the "present in all sisters" conditions for `P_rogue` / `P_conditional`) are computed against a per-organism panel of close relatives:

| organism             | sister strains                            | relationship                                |
|----------------------|-------------------------------------------|---------------------------------------------|
| RalstoniaGMI1000     | RalstoniaBSBF1503, RalstoniaPSI07         | same genus (each Ralstonia is the others')  |
| RalstoniaBSBF1503    | RalstoniaGMI1000, RalstoniaPSI07          | same genus                                  |
| RalstoniaPSI07       | RalstoniaGMI1000, RalstoniaBSBF1503       | same genus                                  |
| Dda3937              | Ddia6719, DdiaME23                        | same genus (Dickeya dadantii)               |
| HerbieS              | Cup4G11, BFirm, Burk376                   | Burkholderiales order                       |
| Magneto              | azobra, Smeli, PS                         | Alphaproteobacteria order                   |

`n_sisters` in the CSV is how many sisters have an orthologous-group hit for that gene; `max_sisters` is how many sister genomes are in the panel.

## 7. Caveats

- **Domain barrier.** Every reference and sister genome in the atlas is bacterial. The conservation scores, ortholog groups, and protection-panel inversions are bacterial-context calibrations. The atlas is not expected to work on archaea or eukaryotes and should not be applied to them without recalibration.
- **Magneto loose-sister set.** The Magneto pangenome channel is noisier than the others. Its sisters (azobra, Smeli, PS) sit at the Alphaproteobacteria-order level rather than the genus level, sharing only ~1,300 orthologous groups with Magneto's 2,631 genes. This drives Magneto's lower precision (77%) and inflated coverage (58%) versus the Ralstonia / Dda3937 / HerbieS tiers — many ortholog-absence calls reflect divergence, not within-species dispensability. Filter Magneto on `confidence >= 3` if you need a stricter precision floor.
- **Rogue zone is quarantined, not predicted.** ROGUE_SUSPECT and CONDITIONAL_SUSPECT rows are not a prediction failure to apologize for; they are the atlas refusing to call genes where no $0 sequence feature has positive predictive power across clades. About 14% of true essentials live in this zone. Treat these tiers as "experiment required" rather than as a weaker version of the confident tiers.
- **Cross-validation flag.** `cross_validated == False` means the leave-one-out cross-organism prediction disagreed with the tier; this is informative for the SUSPECT tiers (which by design ignore the cross-organism score) but should be treated as a warning sign for any CONFIDENT row that carries it.

## Files

- `beril_RalstoniaGMI1000.csv`, `beril_RalstoniaBSBF1503.csv`, `beril_RalstoniaPSI07.csv`, `beril_Dda3937.csv`, `beril_HerbieS.csv`, `beril_Magneto.csv` — one row per gene, schema as in Section 3.
- `../atlas_multi_summary.json` — per-organism tier counts, coverage, and precision in machine-readable form.
- `../atlas_multi_summary.png` — six-panel visual summary (tier composition + precision-by-confidence) for the whole atlas.
