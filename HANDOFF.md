# HANDOFF — Bacterial Gene Essentiality Prediction

## Repo + branch + data
- **GitHub:** https://github.com/Nikku03/cell.git  (canonical owner = `Nikku03`)
- **Branch:** `claude/vectorize-gex-propensity-NRqBW`
- **Drive:** `/content/drive/MyDrive/cell_count_dynamics/multiorg/`
  - `labels.csv` (210,363 rows, 59 organisms, 52,940 essential)
  - `orthology_features.csv` (179,237 rows, 17,222 OGs)
  - `genome_cache/<accession>/` x91: `genomic.gff`, `protein.faa`, `genome.fna`
  - `deg/` (DEG independent essentiality, 66 datasets, 26,619 essentials)
  - `esm2_embeddings*/` (only if the embedding pass was run)
- In-repo copy for sandbox work: `data/drive_import/` (labels, orthology, deg,
  genome_cache GFF+faa+10 fna).

## WHERE WE LANDED (the measured truth — don't re-derive)
- Best honest model: **calibrated leak-free LOO-organism MCC ~0.63** (matches
  across XGBoost stacks). AUC ~0.86.
- **The gains came from honest evaluation + calibration, NOT features.** Six
  engineered feature classes (cooccur, regulators, FBA, synteny, BLO ontology,
  codon) each added ~0 — all absorbed by `family_frac` (conservation prior,
  AUC 0.94 alone, 56% of model importance).
- **The ONE feature that beat absorption:** "standard-condition / rich-medium
  biochemistry" (does the medium provide this gene's product?) — +0.017 AUC
  overall, +0.030 conditional-zone MCC. It works because it's BIOCHEMISTRY,
  not phylogeny (orthogonal to family_frac). See `outputs/environment_standard_condition.md`.
- **AlphaFold-paradigm result:** 95%+ accuracy on the most-confident ~39% of
  genes (consistent-label subset), flag the rest for experiment. Risk-coverage,
  not a single number. See `outputs/best_attempt_95.md`.

## THE CEILING — quantified, not guessed (`outputs/plan_assessment.md`)
Variance decomposition of essentiality FLIPS (50,574 genes in conditional/
flippable families):
- organism identity (env proxy): **7.5%**
- gene/OG identity (already used by family_frac): **13.6%**
- residual = **organism x gene interaction = environment-dependence: ~79%**

The 79% is provably NOT in the genome+identity data. No architecture recovers
it. BUT: ~18% of the organism term IS inferable from genome CONTENT (size,
paralog redundancy) — niche is partly written in gene content, and that
generalizes to novel organisms (unlike one-hot identity).

## REFUTED IDEAS (don't rebuild — tested, leak-free)
- **Two-tier rogue specialist** (remove family_frac, train on rogue zone):
  REFUTED. High ROC-AUC was a mirage; lost at every usable precision because
  residual family_frac still discriminates in the rogue zone. The intrinsic
  features DO help, but FUSED in one model, not as a specialist.
  See `outputs/rogue_specialist_results.md`.
- **GNN over chromosomal adjacency:** ~0 (duplicates the operon feature).
- **k-mer structural-isozyme proxy:** weak (func_redundancy/product-isozyme was
  the strongest intrinsic). The ESM-embedding version is UNTESTED — see below.

## OPEN / UNTESTED HIGH-VALUE MOVES (ranked)
1. **Fitness Browser condition matrix** (fit.genomics.lbl.gov, ~2-3 GB,
   ~7000 gene x condition fitness scores). THIS IS THE 79% RESIDUAL — the only
   data that contains the conditional-essentiality prize. Reframes target to
   `essential(gene, condition)`. Build a **bilinear/two-tower** model (gene
   tower x condition tower) — the one architecture where deep learning beats
   XGBoost on this problem.
2. **ESM-2 structural-isozyme test** (1 hr A100): replace the k-mer within-
   genome uniqueness proxy with ESM-2 cosine to nearest within-genome protein.
   Decisive yes/no on whether fold-aware backup detection lifts the rogue zone.
   Frozen ESM as feature extractor + XGBoost (NOT fine-tuned: 33K labels would
   overfit 650M params). Skeleton: `scripts/train_neural_essentiality.py`,
   `scripts/best_attempt_95.py`.
3. **Phylogenetic-profile-as-MSA (the AlphaFold-transpose):** pull THOUSANDS of
   genomes (not 59), build gene x genome presence/absence tensor, train a deep
   reader of higher-order co-occurrence. The crude scalar co-occurrence was
   absorbed; the full matrix is a different, undrawn well. Nature already ran
   the conditional-essentiality experiment via gene gain/loss across niches.
4. **One wet-validatable surprise:** mine the confident-39% predictions for a
   biologically UNEXPECTED rogue essential, cross-check vs literature not used
   in training. One confirmed surprise > 0.05 MCC. Orthogonal — needs no new data.

## THE GENIUS REFRAME (the direction, if going for groundbreaking)
Stop predicting a per-gene binary score. Build a **conditional-vulnerability map**:
per organism, per environment, where is the cell killable and by what mechanism.
- target = mechanism (backup-paralog / env-rescue / network-position), not just prob
- environment = explicit input (from Fitness Browser, OR partly inferred from
  genome content per move #1's 18%)
- the deliverable is a therapeutic-target atlas + one validated surprise, NOT
  a leaderboard number. Accuracy becomes a footnote.
Honest constraint: requires the condition data (it's the 79%); the genome alone
reconstructs at most ~21% of the conditional signal.

## LEAK RULES (non-negotiable — every result above obeys these)
- `family_frac` and OG-entropy: recompute per held-out organism, EXCLUDING that
  organism's labels. (helper pattern: `family_frac_excl(org)` in the scripts)
- intrinsic/sequence/structure/env features: from the gene's OWN genome, label-free.
- NO neighbour-essentiality-LABEL features (leaks test labels within an organism).
- Evaluation = leave-one-ORGANISM-out (deployment-realistic), then isotonic
  calibration, then risk-coverage. Report rogue-zone / conditional-zone metrics
  SEPARATELY from the overall headline.

## KEY SCRIPTS
- `scripts/best_attempt_95.py` — strongest leak-free stack + risk-coverage (the AlphaFold paradigm)
- `scripts/test_environmental_conditions.py` — the rich-medium biochemistry feature (the one that worked)
- `scripts/rogue_specialist_experiment.py` — the refuted two-tier test (decisive metrics shown)
- `scripts/syn3a_full_pipeline.py` — sequence->structure->function->network on unclear essentials
- `scripts/build_per_clade_evaluation.py` — the LOO-organism + leak-free family_frac harness
- `scripts/train_neural_essentiality.py` — ESM+GNN+XGBoost hybrid (Colab; scope-bug noted: train on ALL 175K rows with ESM as NaN-optional cols, NOT just the 10 genome orgs)
