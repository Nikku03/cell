# NEXT_SESSION — queued work for the next Claude session

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight (every session)

1. `python memory_bank/.invariants/check.py` — must print `OK` before doing any work. If it prints `FAIL`, stop and resolve errors.
2. Read `PROJECT_STATUS.md`.
3. Read this file.

## Queued work (in order)

The current layer is **Layer 0 — Genome**, currently at **Phase A — Literature survey**. Stay on Phase A until all three items below are done. Do not start Phase B (design) until the user approves the Phase A outputs.

### 1. Inventory existing `cell_sim/` and `cell_sim_rust/` components

Before reading any new papers, enumerate what already exists so we reuse rather than rewrite. Produce `memory_bank/concepts/dna/EXISTING_CODE_INVENTORY.md` listing, for each file in `cell_sim/layer0_genome/`, `cell_sim/layer1_atomic/`, `cell_sim/layer2_field/`, `cell_sim/layer3_reactions/`, and `cell_sim_rust/src/`:
- one-line summary of what it does,
- what it exposes (classes, functions),
- whether it is usable as-is for the Layer 0 genome representation, needs light adaptation, or should be replaced (and why).

Keep each entry <= 3 lines. No speculative rewrites yet.

### 2. Canonical Syn3A genome sources

Identify and register (in `memory_bank/sources/`) the three canonical sources for Syn3A's genome:
- NCBI GenBank CP016816 (Syn3A sequence + annotations),
- Hutchison 2016, Science (original Syn3A design + gene list),
- Breuer 2019, eLife (essentiality labels - our validation ground truth).

For each: create a source JSON with `id`, `citation`, `doi`, `type`, `organism`, `authority`, and `url`. Do NOT fabricate DOIs or URLs - leave the field empty with a `caveats` note if not known, and ask the user.

### 3. Minimum viable fact set for Layer 0

Extract and register the following facts. Each needs a proper fact file under `memory_bank/facts/`:
- total gene count of Syn3A (from Hutchison 2016 / Breuer 2019),
- total chromosome length in bp (from CP016816),
- list of 452 coding genes with gene locus tag, name, start/end, strand - this probably belongs as a single `structural/syn3a_gene_table.json` fact pointing to a data file under `memory_bank/data/` rather than 452 separate fact files,
- number of essential vs. non-essential genes per Breuer 2019,
- location of origin of replication.

At the end of Phase A, write `memory_bank/concepts/dna/PHASE_A_SUMMARY.md` (short: what we know, what is uncertain, what we still need). Then stop and ask the user to approve moving to Phase B.

## Explicitly out of scope for next session

- Do not design or implement any Layer 0 code. Phase A is literature only.
- Do not touch layers 1-6.
- Do not modify the existing `cell_sim/` code - only catalogue it.

## Decision points that need the user

- Whether to keep the existing `cell_sim/layer0_genome/` representation or build a new one backed by the memory bank.
- Whether the 452-gene table lives as (a) a single large fact file, (b) a CSV alongside a pointer fact, or (c) many small fact files. Brief section 4 implies (b) is fine.
- Preferred Breuer 2019 essentiality label file - we need confirmation of exact path or format.
