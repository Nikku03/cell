# Layer 0 — Genome — REPORT

Layer 0 is **complete** per the phase definitions in the brief section 5.2. This report documents what the layer does, what it validates against, its known limitations, and how subsequent layers should consume it.

## What the layer does

Provides the typed, memory-bank-backed reference-data accessor `Genome`. Loads 496 genes of JCVI-Syn3A from a CSV pointed to by a fact, cross-checks every derived value against the facts that claim it, and exposes a small, stable API (`genome.length_bp`, `genome["JCVISYN3A_0001"]`, `genome.cds_genes()`, `genome.knocked_out([...])`, etc.).

Public API lives in `cell_sim/layer0_genome/genome.py`. No downstream layer should read the GenBank file or the gene-table CSV directly.

## What it validates against

All 12 tests in `cell_sim/tests/test_layer0_genome.py` pass:

| Test | Claim validated | Fact |
|---|---|---|
| `test_chromosome_length_matches_fact` | 543,379 bp | `syn3a_chromosome_length` |
| `test_gene_count_matches_fact` | 496 total genes | `syn3a_gene_count` |
| `test_oric_matches_fact` | oriC at position 1 | `syn3a_oric_position` |
| `test_feature_type_breakdown` | 458 CDS / 29 tRNA / 6 rRNA / 2 ncRNA / 1 tmRNA | `syn3a_gene_count.value.breakdown` |
| `test_dnaA_is_first_gene` | JCVISYN3A_0001 is dnaA, starts at 1 | implicit oriC anchor |
| `test_coordinates_within_chromosome` | no gene extends past 543379 | `syn3a_gene_table` |
| `test_locus_tag_format` | all locus tags match `JCVISYN3A_\d{4}` | `syn3a_gene_table` |
| `test_knockout_removes_one_gene` | knockout is a view, not mutation | contract |
| `test_knockout_unknown_tag_raises` | unknown tag fails loud | contract |
| `test_sequence_load_when_requested` | dnaA CDS starts with ATG | CP016816.2 |
| `test_rpoD_exists` | JCVISYN3A_0407 is the sigma factor | brief §3 |
| `test_load_smoke` | accession / organism / topology | `genbank_cp016816` |

If any of these tests fails in the future, the memory bank and the code have drifted. Fix one or the other before landing the change.

## Validation target from the brief (Layer 0-3)

The brief's Layer 0-3 target is "reproduce measured steady-state protein counts (Thornburg 2022) within 2× for 90% of genes". That is a Layer 3 test (protein counts), not a Layer 0 test. Layer 0's contribution is to make sure downstream layers know which 458 genes are protein-coding and where their CDS are on the chromosome. We do not validate protein counts here.

## Known limitations

1. **Sequence loading is optional.** `Genome.load()` does not load the 543 kbp sequence by default because most downstream layers don't need it. Pass `include_sequence=True` when you need it (e.g. for codon frequencies in Layer 1).
2. **Circular topology is declared but not exploited.** The `topology` attribute is `"circular"`. Callers that need wrap-around indexing (e.g. a transcription-unit spanning the origin) must implement it themselves. `Gene.start_1based` and `.end` follow GenBank coordinates — no wrap.
3. **No pseudogene flagging.** CP016816.2 annotates a handful of features that may be pseudogenes or split genes; we surface them as `feature_type == "CDS"` if that's what the GenBank record says. Refer to the full GenBank qualifiers for detail.
4. **oriC is a point estimate.** `oric_position = 1` is the conventional placement coincident with dnaA; the functional DnaA-box footprint extent is not tracked. Layer 5 (division) will have to decide whether that matters for replication timing.
5. **Breuer 2019 essentiality labels are not in Layer 0.** They belong to Layer 6's ground truth. We register Breuer 2019 as a source now; the specific supplementary file gets pinned when Layer 6 Phase A starts.

## How Layers 1-6 should use it

```python
from cell_sim.layer0_genome.genome import Genome

genome = Genome.load()

# Layer 1 (transcription): iterate CDS + sigma-factor promoter search
for g in genome.cds_genes():
    ...                       # g.start_1based, g.end, g.strand, g.product

# Layer 6 (essentiality): knockout sweep
for tag in essential_candidates:
    ko_genome = genome.knocked_out([tag])
    # feed ko_genome into the existing cell_sim engine...

# Layer 1 codon analysis (needs sequence):
g = Genome.load(include_sequence=True)
```

The existing `cell_sim/layer0_genome/parser.py` and `syn3a_real.py` remain as-is. They are the heavier, GenBank-plus-Luthey-Schulten-Excel path that Layer 3's metabolic setup already uses. Layers 1-6 should prefer the thin `Genome` API for anything that doesn't specifically need that extra data.

## Handoff to Layer 1

Layer 1 (transcription machinery) should start with Phase A against:
- `genbank_cp016816` — already registered; contains promoters and UTRs if annotated.
- `thornburg_2022_cell` — already registered; has transcription rate constants.
- A new source to be added for Shine-Dalgarno / promoter databases if needed.

The Layer 0 `Genome` gives Layer 1 everything it needs on the sequence side; Layer 1's work is measuring, not parsing.
