# Essentiality-Prediction Atlas (multi-organism)

## 1. What this is

A 6-organism, 23,536-gene atlas of bacterial gene-essentiality predictions. Each gene is assigned to one of five tiers (confident essential, confident non-essential, rogue suspect, conditional suspect, unresolved) based on the agreement of up to ten independent evidence channels. The point of the atlas is not to predict every gene — published predictors top out around AUC 0.81 cross-organism but only reach ~28% coverage at precision >=0.70, and they collapse on the "rogue zone" of essential-but-not-conserved genes (about 14% of all essentials, where every $0 sequence feature including ESM-2 hits zero R@P30 cross-clade). The atlas instead predicts where we can, explicitly abstains where we can't, and attaches an audit trail of evidence channels to every call.

## 2. Headline numbers

**Atlas total: 23,536 genes across 6 organisms. 34% confident coverage at 86.0% precision.**
(Single-organism, self-calibrated baselines reach 38% at 89% but do not generalize cross-clade.)

Per-organism:

| organism            |    n | confident_ess | confident_noness | rogue_suspect | conditional | unresolved | coverage | precision |
|---------------------|-----:|--------------:|-----------------:|--------------:|------------:|-----------:|---------:|----------:|
| RalstoniaGMI1000    | 4403 |            57 |             1495 |          1433 |         168 |       1250 |      35% |       82% |
| RalstoniaBSBF1503   | 4431 |            83 |             1383 |          1438 |         164 |       1363 |      33% |       87% |
| RalstoniaPSI07      | 4298 |           522 |              927 |          1462 |         145 |       1242 |      34% |       88% |
| Dda3937             | 3926 |           146 |              642 |          1327 |         150 |       1661 |      20% |       94% |
| HerbieS             | 3847 |           223 |             1047 |           499 |         142 |       1936 |      33% |       93% |
| Magneto             | 2631 |            91 |             1427 |           177 |          75 |        861 |      58% |       77% |

**Cross-organism consistency.** Of 1,312 confident calls on genes that share an orthologous group across organisms, 1,300 agree (99.1%). The atlas does not just have surface-level coverage — independently called organisms converge on the same gene-level verdict.

**Precision stratifies by per-gene confidence (number of independent channels firing in the same direction):**

| channels firing | precision |
|----------------:|----------:|
|               1 |     85.0% |
|               2 |     85.5% |
|               3 |     88.6% |
|               4 |     95.4% |

A user who wants 95%-precision calls can filter `confidence >= 4`; a user who wants maximum recall can accept `confidence >= 1`.

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

### Non-essential (5, inverted from protection-panel research)

| code                  | fires when                                                                                                                                                              | inverted from                                                                 |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `N_score`             | Low cross-organism essential score (below the symmetric non-essential threshold).                                                                                       | mirror of `E_score`                                                           |
| `N_pangenome_strong`  | Ortholog ABSENT in ALL sister strains for this organism.                                                                                                                | the only channel that is positive within-species dispensability, not a mirror |
| `N_pangenome_some`    | Ortholog absent in at least one sister strain.                                                                                                                          | weaker form of `N_pangenome_strong`                                           |
| `N_redundancy`        | `n_paralogs >= 4` AND conservation < 0.5 — redundancy-buffered, loss of one copy compensable.                                                                            | "essentials have few paralogs"                                                |
| `N_mobile`            | Within 5 kb of a mobile element (HGT / accessory belt).                                                                                                                 | "essentials avoid mobile elements"                                            |
| `N_long_lagging`      | Long gene on lagging strand AND conservation < 0.3.                                                                                                                     | "essentials avoid head-on collisions, especially when long"                   |

### Phenotype-required quarantine (2)

These do not predict — they **veto** confident calls and route the gene to a suspect tier.

| code             | fires when                                                                                                                              | meaning                                                                |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `P_rogue`        | conservation < 0.1 AND `n_paralogs == 0` AND present in ALL sisters.                                                                    | Structural rogue-essential profile — invisible to homology features, but retained where it counts. Sequence features cannot resolve this. |
| `P_conditional` | conservation < 0.3 AND long-on-lagging-strand AND present in ALL sisters.                                                                | Niche / virulence / conditional-essential candidate.                   |

## 5. Tier rules

Applied in this order; the first match wins (phenotype vetoes have priority over both essential and non-essential resolutions for the tiers below):

```
CONFIDENT_ESSENTIAL     = E_score AND NOT P_rogue
CONFIDENT_NONESSENTIAL  = (N_score OR N_pangenome_strong) AND NOT P_rogue AND NOT P_conditional
ROGUE_SUSPECT           = P_rogue
CONDITIONAL_SUSPECT     = P_conditional
UNRESOLVED              = everything else
```

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
