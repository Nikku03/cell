# Human genes → orthologs → function

Every human protein-coding gene, the species that carry an ortholog of it, and
what those orthologs say the gene does.

This is the same annotation-transfer logic Session 22 used to corroborate the
`JCVISYN3A_0876 × 0878` synthetic-lethal pair — a Syn3A gene of uncertain
function was pinned down through its *M. mycoides* SC and *M. pneumoniae*
relatives — run at human-genome scale instead of on one gene pair.

Results live in `outputs/human_orthologs/`; `REPORT.md` there is the generated
summary with all tables and caveats.

## Headline numbers

| quantity | value |
|:---|---:|
| human protein-coding genes | 20,622 rows / 20,596 distinct genes |
| genes with ≥1 ortholog | 19,266 (93.4%) |
| species searched | 899 |
| human↔other ortholog pairs | 11,329,528 |
| median ortholog species per gene | 619 |
| genes reaching budding yeast | 4,577 (22.2%) |
| genes reaching fly / worm | 10,750 / 9,671 |
| genes with no ortholog anywhere | 1,356 (6.6%) |
| genes not fully characterized in human | 8,540 |
| ↳ with transferable experimental evidence from an ortholog | 4,010 (47.0%) |
| ↳ independent functional hypotheses (paralogs collapsed) | 3,456 |
| ↳ dark genes rescued by a fly/worm/yeast ortholog | 235 |

Note the gene count: the request said ~16,000 human genes, but NCBI currently
annotates **20,622 protein-coding records** for *H. sapiens*. Nothing was
subsampled to 16,000 — the full set is covered. The ~16k figure is close to the
number of human genes that have a fly ortholog (10,750), a worm ortholog
(9,671), or a mouse ortholog (16,853), so it may have come from a mouse-ortholog
count.

## Where the data comes from

| source | file | what it supplies |
|:---|:---|:---|
| NCBI Gene | `gene_orthologs.gz` (123 MB) | ortholog calls, 895 vertebrate species vs human |
| Alliance / DIOPT | `ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz` (15 MB) | fly, worm, yeast, *X. laevis* orthologs with algorithm-consensus counts |
| NCBI Gene | `All_Data.gene_info.gz` (1.5 GB) | symbols, descriptions, FlyBase/WormBase/SGD/Xenbase cross-references |
| NCBI Gene | `gene2go.gz` (1.3 GB) | GO annotations **with evidence codes** |
| UniProt / Swiss-Prot | REST stream, 20,431 reviewed human entries | curated human FUNCTION text, protein existence level |
| NCBI Taxonomy | `taxdump` | lineage for each ortholog species → clade → conservation depth |

The two ortholog sources are complementary rather than redundant: NCBI's
pipeline is vertebrate-only (fly, worm and yeast return zero hits against
human), while the Alliance file covers exactly those invertebrate and fungal
models. Combining them is what makes the yeast-to-primate depth axis possible.

## Method

1. **Gene universe** — protein-coding entries from `Homo_sapiens.gene_info`.
2. **Orthologs** — one streaming group-by over the 11.3 M human rows of
   `gene_orthologs`, plus the Alliance stringent set joined on HGNC ID for
   human and on FlyBase/WormBase/SGD/Xenbase accessions for the other side.
3. **Conservation depth** — each ortholog species is walked up the NCBI
   taxonomy to its nearest matching clade, and the gene is stamped with the
   *deepest* clade it still reaches. Ranks follow divergence time from human,
   so lungfish (~415 Mya) sits nearer than ray-finned fish (~430 Mya) even
   though both read as "fish".
4. **What we already know** — UniProt FUNCTION text plus the count of GO terms
   carrying experimental evidence codes (`EXP`, `IDA`, `IPI`, `IMP`, `IGI`,
   `IEP`, `HTP`, `HDA`, `HMP`, `HGI`, `HEP`). Computational and
   already-transferred-by-homology codes (`IEA`, `ISS`, `ISO`, `IBA`) are
   deliberately excluded so that ortholog transfer is not scored against
   evidence that was itself transferred from an ortholog.
5. **Function transfer** — for every gene, collect the experimental GO terms
   attached to its orthologs, subtract the ones the human gene already carries
   experimentally, and keep the remainder as transferable evidence, tagged with
   the source species and the number of independent species supporting each
   term.

Three annotation tiers: `characterized` (FUNCTION text **and** ≥3 experimental
GO terms), `sparse` (some curated function, thinner than that), and
`uncharacterized` (no FUNCTION text and no experimental GO term at all). Only
the third tier is genuinely dark; see the caveats in `REPORT.md`.

## Outputs

| file | rows | contents |
|:---|---:|:---|
| `human_gene_ortholog_function.tsv.gz` | 20,622 | the main table — one row per human gene, with panel orthologs, conservation depth, clade counts, human function, and ortholog-inferred function |
| `human_ortholog_pairs_panel.tsv.gz` | 284,571 | long form: one row per human gene × panel species ortholog |
| `inferred_function_dark_genes.tsv` | 4,010 | genes not fully characterized in human that have transferable ortholog evidence |
| `highlight_ancient_dark_genes.tsv` | 235 | the payload: no human annotation at all, but an experimentally-studied fly/worm/yeast ortholog |
| `species_coverage.tsv` | 895 | per-species ortholog counts with clade labels |
| `summary.json`, `REPORT.md` | — | statistics and the generated narrative report |

The 20-species reference panel spans chimpanzee → macaque → mouse/rat →
dog/cow/pig → opossum → platypus → chicken → anole → *Xenopus* → zebrafish →
fugu → elephant shark → lamprey → fly → worm → yeast.

## What the result actually shows

**Most of the human genome is old.** 61% of protein-coding genes have an
ortholog in fly, worm or yeast; 22% reach budding yeast, meaning the gene
predates animals entirely. Only 1.3% are primate-restricted.

**Ortholog transfer recovers function for about half the under-annotated
genes.** Of 8,540 genes that are not fully characterized in human, 4,010 have
at least one experimental GO term on an ortholog that the human gene does not
carry itself — 3,456 of them independent once paralog families are collapsed.

**The 235 genes in `highlight_ancient_dark_genes.tsv` are the interesting
ones**: no curated human function whatsoever, yet an ortholog in fly, worm or
yeast has been characterized experimentally. Examples that check out against
the literature — `PPP4R3C` → yeast `PSY2` (PP4 regulatory subunit, DNA damage
response), `RNF113B` → yeast `CWC24` (spliceosomal RING protein), `XKR6` → worm
`ced-8` (phospholipid scramblase in apoptotic corpse clearance), `TSNARE1` →
yeast `PEP12` (endosomal SNARE), `IGBP1C` → yeast `TAP42` (PP2A regulator).

**Deep conservation and being well-studied are not the same thing.** Genes like
`LYAR`, `NDC1`, `PDE12` and `CZIB` are present in ~890 of 899 species — as
conserved as anything in the genome — and several still sit in the `sparse`
tier.

## From evidence to a statement you can actually use

The table above proves a gene has orthologs and lists the GO terms attached to
them. A GO term list is evidence, not an answer — "process:
deadenylation-dependent decapping" is not a sentence you can put in a report. A
third layer fetches the curated UniProt FUNCTION *prose* of the orthologs
themselves and assigns every human gene the best statement available, tagged
with where it came from: `outputs/human_orthologs/FUNCTION_STATEMENTS.md`.

| evidence tier | genes | meaning |
|:---|---:|:---|
| `human_curated` | 16,779 | the human gene already has curated function text |
| `ortholog_curated` | 455 | statement taken from a model-organism ortholog's record |
| `ortholog_experimental` | 489 | no ortholog prose, but ortholog GO with experimental evidence |
| `none` | 2,899 | no usable evidence anywhere |

**3,843 human genes have no curated function text of their own; orthologs
supply a statement for 1,004 of them.** 2,839 still have nothing.

The revealing number is *which* species does the rescuing. Of the 455 prose
transfers, only **38** come from mouse or rat — **408 come from fly, worm or
yeast**. That is not a coverage artifact: mouse curation largely mirrors human
curation, so a gene nobody has characterized in human is usually
uncharacterized in mouse too. The genes that get rescued are rescued by
classical invertebrate and yeast genetics, which is exactly the literature a
human-only search never surfaces.

Two things are recorded per row because they decide how far to trust the
sentence:

- `transfer_confidence` — `high_mammal` (use nearly verbatim) down to
  `low_fungal` (molecular/complex-level only). ADAMTS16's statement comes from
  worm and talks about cuticle collagen and body size; the metalloprotease/ECM
  core transfers, the cuticle does not.
- `source_shared_with_n_human_genes` — 168 of the 455 transfers are shared with
  at least one human paralog (ACTL7B, ACTL8, ACTRT2 and ACTRT3 all inherit the
  same fly Act53D sentence). Identical statements on paralogs are one piece of
  evidence, not four; the real denominator is 361 distinct ortholog sources.

## Going deeper than the databases: bacteria and archaea

Both ortholog sources stop at fly, worm and yeast, so neither can say whether a
gene reaches *prokaryotes*. For the 235 dark genes that question is answered by
direct sequence search instead — see `outputs/human_orthologs/DEEP_HOMOLOGY.md`.

234 of the 235 (MYH16 has no protein product in human — it is a pseudogene)
were searched against 283,294 proteins from 26 species that appear in neither
ortholog source: 8 bacteria, 4 archaea, plus plants, algae, *Dictyostelium*,
*Plasmodium*, *Trypanosoma*, *Giardia*, fission yeast, choanoflagellate,
sponge, placozoan, cnidarian, sea urchin, amphioxus and *Ciona*.

| result | value |
|:---|---:|
| genes with ≥1 homolog in the panel | 231 of 234 |
| genes with no hit anywhere | 3 (DMRTC1, DMRTC1B, TEX13D) |
| **genes reaching bacteria or archaea** | **94** |
| ↳ with at least one reciprocal best hit | 7 |
| genes that only the profile search found | 38 of 94 |

Two passes, because single-sequence search is nearly blind at this distance:
`phmmer` for the conservative baseline, then a profile HMM built from each
gene's *eukaryotic* homologs and searched against the prokaryotic proteomes
only. That is jackhmmer's mechanism with the expensive search restricted to 30k
sequences instead of 283k — measured at 4.25 hours for jackhmmer versus about 7
minutes this way. 38 of the 94 prokaryote-reaching genes are invisible to the
direct search and only appear once the profile exists.

The seven reciprocal-best-hit genes are the safest calls: `ACYP1` →
*Thermococcus* acylphosphatase (E=2e-19), `PPM1N` → *B. subtilis* PrpC
(E=9e-11), `AMMECR1L` → *M. jannaschii* MJ0810 (E=3e-22), `MCTS2` →
*M. jannaschii* MJ1432 (E=3e-31), `SDR42E1` → mycobacterial 3β-HSD (E=2e-74),
`METTL25B`, `DIP2C`.

The more interesting pattern is in the other 87. Their bacterial homology is
often overwhelming (down to E=1e-206) but the reverse search lands on a
*different* human gene — ABCF2-H2BK1 → ABCF1, ATAD3C → VCP, ATP13A5 → ATP2C1,
MGAM2 → MGAM. These dark genes are young duplicates and readthrough products of
ancient families: the lineage is prokaryotic, but the human representative of
it is the parent gene, not the dark one. A failed reciprocal test here is
evidence about which paralog represents the family, not evidence against deep
ancestry.

Worth noting for this repo specifically: 19 of the dark genes have a homolog in
*Mycoplasma mycoides* SC PG1 and 18 in *M. genitalium* — the organism family
JCVI-Syn3A was built from.

## Reproducing

```bash
scripts/fetch_ortholog_data.sh        # ~3 GB into data_cache/human_orthologs/, ~4 min
python3 scripts/human_gene_orthologs.py   # ~6 min, single-threaded, peak RSS ~1 GB
python3 scripts/human_ortholog_report.py  # instant, writes REPORT.md

python3 scripts/deep_homology_dark_genes.py --resolve-panel   # proteome IDs
python3 scripts/deep_homology_dark_genes.py --resolve-queries # dark-gene sequences
scripts/fetch_deep_homology_data.sh       # ~180 MB of proteomes
python3 scripts/deep_homology_dark_genes.py --cpus 4   # ~55 min (25 min phmmer,
                                          # 7 min profiles, 20 min reciprocal)
python3 scripts/deep_homology_report.py   # writes DEEP_HOMOLOGY.md

scripts/fetch_ortholog_function_data.sh   # ~25 MB, curated function text
python3 scripts/ortholog_function_statements.py --fetch-deep-annotations
python3 scripts/ortholog_function_statements.py   # writes FUNCTION_STATEMENTS.md
```

The forward search caches to `data_cache/deep_homology/forward_hits.json.gz`;
`--reuse-forward` re-runs only the reciprocal stage from it.

The cache directory is gitignored. Runtime is dominated by the two big NCBI
dumps (60 M and 120 M rows), which are streamed and filtered rather than loaded.

## Relationship to the rest of this repo

The Syn3A work uses knowledge-based detectors — protein-complex membership and
gene annotation — that carry more of the v15 MCC than the trajectory detectors
do (`figures/` detector contributions). Those priors are ortholog-derived
annotations at heart. This table is the same evidence type at eukaryotic scale,
and the 3,456 independent functional hypotheses in it are the kind of input a
comparable human-cell essentiality prior would need. It is a standalone data
product for now: no Syn3A metric depends on it, and nothing in the Breuer
benchmark changes because of it.
