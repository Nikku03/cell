# memory_bank/data/

_Data files that are citable reference material for the simulator._

Files here follow the same rule as facts: every CSV/JSON/TSV must be pointed at by a fact file in `memory_bank/facts/` (most typically `structural/`) that records its provenance. The fact file is the citation; the data file is the content.

## Files present

| File | Pointer fact | Provenance |
|------|--------------|------------|
| `syn3a_gene_table.csv` | `facts/structural/syn3a_gene_table.json` | Parsed from `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb` (CP016816.2) in Session 3. |

## Luthey-Schulten input data (staged locally, NOT in this directory)

The following binary / large files are staged at `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` (gitignored) rather than under `memory_bank/data/` because the existing `cell_sim/layer0_genome/syn3a_real.py` loader hard-codes that path. Keeping them at the legacy path avoids changes to working code. Each is sourced from https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation (master branch, retrieved 2026-04-21).

| File | Bytes | SHA-256 |
|------|-------|---------|
| `syn3A.gb` | 1,180,612 | `f1d1d19b3624c8391deff73ab60627478df89cd55dfafdfb67d0df14a7680fdf` |
| `kinetic_params.xlsx` | 59,406 | _to be recorded when first ingested_ |
| `initial_concentrations.xlsx` | 164,362 | _to be recorded when first ingested_ |
| `complex_formation.xlsx` | 12,940 | _to be recorded when first ingested_ |
| `Syn3A_updated.xml` | 364,329 | _to be recorded when first ingested_ |

To re-stage after a fresh checkout:

```bash
mkdir -p cell_sim/data/Minimal_Cell_ComplexFormation/input_data
cd cell_sim/data/Minimal_Cell_ComplexFormation/input_data
BASE=https://raw.githubusercontent.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation/master/input_data
for f in syn3A.gb kinetic_params.xlsx initial_concentrations.xlsx complex_formation.xlsx Syn3A_updated.xml ; do
  curl -sS -o "$f" "$BASE/$f"
done
```

The upstream repo had no `LICENSE` file at the root as of 2026-04-21. We stage the data locally and do **not** redistribute it via this repo (the `data/` rule in `.gitignore` keeps the input_data directory out of git).

## Policy

- Every data file added to `memory_bank/data/` must be pointed at by a fact file within the same commit.
- Binary data files (`.gb`, `.xlsx`, `.xml`) live under `cell_sim/data/` and are gitignored. Parsed / derived text data (`.csv`, `.tsv`) can live under `memory_bank/data/` and is tracked.
- File SHA-256 is required for any external data source that we can't fully regenerate on demand.
