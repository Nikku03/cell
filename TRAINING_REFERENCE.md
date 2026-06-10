# TRAINING REFERENCE — datasets, tier list, feature set

Companion to HANDOFF.md. This is the "what to train on" reference.

═══════════════════════════════════════════════════════════════════════
## PART 1 — DATASET TIER LIST (by measured value)
═══════════════════════════════════════════════════════════════════════

### TIER 0 — core, already have (the substrate)
| dataset | what | location |
|---|---|---|
| labels.csv | 210,363 genes, 59 orgs, 52,940 essential (BINARY) | Drive multiorg/ |
| orthology_features.csv | 179,237 rows, 17,222 OGs, family_frac per fold | Drive multiorg/ |
| genome_cache/ x91 | GFF (positions/products) + protein.faa (ESM input) + genome.fna | Drive multiorg/ |

### TIER 1 — the unlock (downloading) + have-but-underused
| dataset | what | why it matters |
|---|---|---|
| **Fitness Browser (Feb 2024)** | gene x condition fitness matrix, ~50 orgs x ~50-100 conditions | **IS the 79% residual.** Turns target into essential(gene,condition). GeneFitness + Experiment tables. |
| FB reannotations | better functional annotation + protein seqs | upgrades the env-biochemistry feature (the one that worked) |
| DEG (have) | 26,619 essentials, 66 datasets, 82 orgs, **COG codes** | independent labels (external test) + COG functional categories |

### TIER 2 — would help, not the bottleneck
| dataset | use | honest expectation |
|---|---|---|
| AlphaFold structures (329K proteins) | real fold for structural-isozyme | +0.02-0.04 rogue zone; big fetch |
| KEGG / MetaCyc pathways | pathway co-essentiality | small unless paired w/ condition data |
| thousands more genomes | phylogenetic-profile-as-MSA | the AlphaFold-transpose; undrawn well |

### TIER 3 — marginal
syn3a/mgen/mpne minimal cells (have; great for function-inference demos, small
for training), lifestyle/niche metadata, STRING/BioGRID PPI.

═══════════════════════════════════════════════════════════════════════
## PART 2 — FEATURE SET (by validated status + leak notes)
═══════════════════════════════════════════════════════════════════════

### A. VALIDATED — KEEP (the working stack)
| feature | signal | leak handling |
|---|---|---|
| **family_frac** | AUC 0.94 alone; the backbone | recompute per-fold EXCL test org |
| OG-essentiality entropy | confidence signal (pLDDT analog) | recompute per-fold EXCL test org |
| family_n_organisms | conservation breadth | from orthology (cross-org, fold-safe) |
| n_paralogs_in_genome | genome redundancy | label-free |
| is_orphan, family_size_total | family structure | label-free |
| **env_substitutable** | **+0.017 AUC, +0.030 cond-MCC — ONLY feature that beat absorption** | label-free (from product annotation) |
| func_redundancy (product isozyme) | strongest intrinsic, d=0.24 | label-free (within-genome) |
| org-content (genome size, paralog density) | ~18% of org flip-bias, GENERALIZES | label-free (per-organism) |

### B. WEAK BUT CHEAP — include
operon, same_strand_prev/next, intergenic_prev/next, biophysics
(length, gravy, aromaticity, charged_frac, frac_K, frac_R). All label-free.

### C. ABSORBED — DROP (tested, ~0 over family_frac)
cooccurrence (SCALAR version), regulator flags, FBA flags, synteny,
BLO lifecycle ontology, codon/CAI. Each added ~0 MCC — family_frac ate them.
(NOTE: the scalar co-occurrence was absorbed; the full phylo-profile MATRIX
across thousands of genomes is a DIFFERENT, untested thing — see HANDOFF #3.)

### D. UNTESTED — TRY (ranked)
1. **ESM-2 embedding** (frozen, feature-extractor): the real structural signal.
   The k-mer proxy was WEAK — ESM sees fold-level homology k-mers miss.
2. **ESM structural-isozyme**: cosine to nearest within-genome protein in ESM
   space = "is there a fold-level backup here." The real version of the idea
   that the k-mer proxy couldn't validate. Targets rogue essentials.
3. **phylogenetic-profile matrix** (deep reader, not scalar): the AlphaFold-transpose.

### E. NEW WITH FITNESS BROWSER
- condition vector: media, carbon source, stress, expGroup (one/multi-hot or learned embed)
- fitness-derived CONTINUOUS target (re-derive essentiality from fitness < lethal threshold)
- architecture: TWO-TOWER (gene tower x condition tower) -> fitness; the one
  place deep learning beats XGBoost (matrix completion over gene x condition).

═══════════════════════════════════════════════════════════════════════
## PART 3 — THE ASSEMBLY (how it all fits)
═══════════════════════════════════════════════════════════════════════

WITHOUT Fitness Browser (current ceiling, MCC ~0.63, 95% on confident 39%):
  XGBoost on [A + B feature groups], leak-free LOO-organism, isotonic calibration,
  risk-coverage reporting. Optionally + ESM features (group D 1-2) as NaN-optional
  columns trained on ALL 175K rows (NOT just the 10 genome orgs -- scope bug in
  train_neural_essentiality.py).

WITH Fitness Browser (the leap):
  1. Verify FB locusId joins our labels.locus_tag (BERIL derived from FB -> should match)
  2. Re-derive per-(gene,condition) essentiality from GeneFitness
  3. Two-tower model: gene_repr(ESM+tabular A/B) x condition_repr(Experiment meta)
  4. Target = fitness (continuous) AND/OR conditional-essential (binary per condition)
  5. Deliverable = conditional-vulnerability MAP (per org, per stress, where killable
     + mechanism), not a single MCC. + one wet-validatable surprise.

LEAK RULES (every result must obey):
  - family_frac / OG-entropy: per-fold, exclude held-out organism.
  - all sequence/structure/env/content features: from own genome, label-free.
  - NO neighbour-essentiality-LABEL features.
  - eval = leave-one-ORGANISM-out -> calibrate -> risk-coverage.
  - report conditional/rogue-zone metrics SEPARATELY from overall headline.
