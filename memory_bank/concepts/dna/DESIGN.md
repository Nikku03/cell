# Layer 0 — Genome — DESIGN

_Scope: the thin Layer 0 API that every other layer reads from. This is not the simulator engine; it is the reference-data accessor that ensures every derived quantity has a cited source._

## State tracked by Layer 0

A `Genome` object exposes:

| Attribute | Type | Source fact |
|---|---|---|
| `accession` | `str` (e.g. `"CP016816.2"`) | `genbank_cp016816` |
| `organism` | `str` (`"JCVI-Syn3A"`) | `genbank_cp016816` |
| `length_bp` | `int` (543379) | `syn3a_chromosome_length` |
| `topology` | `Literal["circular"]` | `syn3a_chromosome_length` |
| `oric_position` | `int` (1) | `syn3a_oric_position` |
| `genes` | `tuple[Gene, ...]` (496 entries) | `syn3a_gene_table` |
| `sequence` | `str` (optional; only if loaded from `.gb`) | `genbank_cp016816` |

Each `Gene` is a frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Gene:
    locus_tag: str            # e.g. "JCVISYN3A_0407"
    gene_name: str            # e.g. "rpoD" (may be "")
    feature_type: str         # "CDS" | "tRNA" | "rRNA" | "ncRNA" | "tmRNA"
    start_1based: int         # 1-based inclusive (GenBank)
    end: int
    strand: str               # "+" or "-"
    length_bp: int
    product: str              # e.g. "RNA polymerase sigma factor"
    protein_id: str           # e.g. "AVX54569.1", "" for RNA genes
```

## Interface that Layers 1-6 will use

```python
from cell_sim.layer0_genome.genome import Genome, Gene

genome = Genome.load()           # memory-bank path, default. No args.
assert genome.length_bp == 543_379
assert genome.oric_position == 1
assert len(genome) == 496
gene = genome["JCVISYN3A_0407"]  # raises KeyError if missing
assert gene.gene_name == "rpoD"
for g in genome.cds_genes():     # 458 CDS, in order of start_1based
    ...
```

Layers never read `.gb` or CSV paths directly. They get a typed object.

## Why this interface

- **Every value is cited.** The loader asserts, at load time, that the CSV's row count matches `facts/structural/syn3a_gene_count.json`'s `value.number`. If a future session silently edits the CSV without updating the fact, the loader fails loud.
- **Decouples downstream layers from file formats.** If we later add a more detailed per-gene record (operon, promoter strength, half-life), we extend `Gene` without rewriting every caller.
- **Preserves existing code.** `cell_sim/layer0_genome/parser.py` and `syn3a_real.py` keep working. The new `genome.py` is a parallel, thinner API. Downstream layers can migrate to it incrementally.

## Data path (concrete)

```
memory_bank/facts/structural/syn3a_gene_table.json
          |
          | "value.data_file": "memory_bank/data/syn3a_gene_table.csv"
          v
memory_bank/data/syn3a_gene_table.csv  (496 rows, 9 columns, 1-based coords)
          |
          | loaded by cell_sim/layer0_genome/genome.py Genome.load()
          v
      Genome object with 496 Gene entries
```

The sequence (`genome.sequence`) is optional — loading it requires the 1.18 MB GenBank file. `Genome.load()` does not load the sequence by default because most downstream layers don't need it. Use `Genome.load(include_sequence=True)` when you do (e.g. Layer 1 needs codon frequencies; Layer 6 doesn't).

## What Layer 0 is NOT

- Not the simulator state. The simulator's runtime state (protein counts, metabolite pools, RNG) lives in Layer 2's `CellState`.
- Not a mutable model. `Genome` is frozen reference data. Knockouts do not mutate the `Genome` — they filter it. `knocked_out(locus_tags)` returns a view, not a new file.
- Not where we put promoter / operon / regulatory information. Those go in `facts/regulatory/` as Layer 1 work begins.

## Test plan (Phase D)

1. **Load smoke test.** `Genome.load()` must succeed without args if the CSV is present; raise a clear error otherwise (with the re-staging command from `memory_bank/data/STAGING.md`).
2. **Cross-check against facts.** The loader internally asserts `len(self.genes) == <fact value>`, `self.length_bp == <fact value>`, `self.oric_position == <fact value>`. Unit test also asserts this at the boundary.
3. **Lookup consistency.** `genome["JCVISYN3A_0001"].gene_name == "dnaA"`; `genome["JCVISYN3A_0407"]` exists.
4. **Feature-type breakdown.** Counter of feature types must match the fact's `breakdown` dict (458/29/6/2/1).
5. **Coordinate sanity.** No gene's `start_1based` < 1. No gene's `end` > `length_bp`.
6. **Optional sequence mode.** `Genome.load(include_sequence=True)` only runs if the `.gb` file is staged; test skips with a clear reason otherwise.

## Out of scope for Layer 0

- Codon frequency tables (Layer 1 concern).
- Promoter / operon annotations (Layer 1 concern; requires new sources).
- Protein mass tables (Layer 3 concern).
- Methylation / chromatin (brief section 3 notes Syn3A has minimal/none).
