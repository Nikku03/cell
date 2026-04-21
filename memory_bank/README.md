# memory_bank/

Structured biological knowledge for the Syn3A simulator. Plain files + SQLite + (later) a FAISS semantic index. No neural nets, no graph DB, no GPU.

## Directory layout

```
memory_bank/
  facts/           one JSON per atomic claim
    parameters/      k_cats, Kms, counts, rates, concentrations
    structural/      complex stoichiometry, gene coordinates
    regulatory/      promoter strengths, allosteric regulation
    uncertainty/     contested or open-question facts
  concepts/        structured notes, hierarchical (dna/, transcription/, ...)
  sources/         one JSON per paper / database / model file
  index/           auto-generated: facts.sqlite, semantic.faiss, concept_graph.json
  .invariants/
    check.py         run after every write - returns 0 if consistent, 1 otherwise
    ranges.json      physical-range limits for parameter values
```

## Fact file format

Every fact file MUST have all of these fields:

```json
{
  "id": "unique_snake_case_identifier",
  "claim": "human-readable statement",
  "value": {
    "parameter": "k_cat_per_s",
    "number": 123.4,
    "units": "1/s"
  },
  "source": "id_of_a_file_in_sources",
  "source_detail": "table 2, row X, page Y",
  "context": {"entity": "...", "organism": "...", "temperature_C": 37, "pH": 7.4},
  "confidence": "measured | inferred | estimated | assumed",
  "caveats": ["things that could be wrong"],
  "dependencies": ["other fact IDs this depends on"],
  "last_verified": "YYYY-MM-DD",
  "used_by": ["path/to/code.py:symbol"]
}
```

Rules:

- **No fact without a source.** Every fact cites a file in `sources/`.
- **No fact without a confidence level.** Pick one of the four.
- **No fact outside physical range.** `ranges.json` caps each parameter type.
- **No contradictions.** Two facts claiming different numbers for the same (parameter, entity, context) are rejected.
- **`used_by` must resolve.** If you claim code uses this fact, the code path must exist.
- **Stale (>90 days) facts are flagged.** Re-verify and bump `last_verified`.

## Source file format

```json
{
  "id": "author_year_journal",
  "citation": "full citation string",
  "doi": "10.xxxx/yyyy",
  "type": "peer_reviewed_paper | database | sbml_model | genbank_record | preprint",
  "organism": "JCVI-Syn3A",
  "authority": "primary | secondary",
  "url": "https://..."
}
```

## Running the checker

From the repo root:

```
python memory_bank/.invariants/check.py
```

Prints the counts and any warnings/errors. Exits 0 on `OK`, 1 on `FAIL`. The SQLite index is only rebuilt when the check passes, so `memory_bank/index/facts.sqlite` never contains a broken state.

## What this is NOT

- Not a vector store that holds embeddings of everything (the FAISS index, when we add it, is a supplement to the JSON files, not a replacement).
- Not a graph database with Cypher / SPARQL queries.
- Not anything that requires a GPU.

Keep it boring. Boring scales.
