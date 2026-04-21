# NEXT_SESSION — queued work for the next Claude session

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight (every session)

1. `python memory_bank/.invariants/check.py` — must print `OK` before doing any work.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/dna/PHASE_A_SUMMARY.md` (the Phase A work product).
4. Read this file.

## Status

Layer 0 Phase A is **complete**. Phase B (design) is **blocked on user input.** Do not start Phase B work until the four open questions in `PHASE_A_SUMMARY.md` have been answered.

## Queued work, once the user has answered

### If the user says "proceed with Phase B" without staging data

Then Phase B's deliverable is a `memory_bank/concepts/dna/DESIGN.md` for Layer 0 that specifies:

- **State**: what `Genome` state the simulator tracks (chromosome sequence string, gene list with locus tags and coordinates, oriC position, topology flag).
- **Data sources**: explicit memory-bank-backed pointers, not hard-coded paths.
- **Interface**: what Layers 1-6 read from Layer 0 (and what they do not).
- **How the existing `cell_sim/layer0_genome/parser.py` + `syn3a_real.py` get adapted**, not rewritten, to sit behind the new interface.
- **Test plan**: assertions that tie parsed values back to recorded facts (e.g. "`parse()['JCVISYN3A_0407']['product']` must match the `rpoD` sigma-factor annotation recorded in the memory bank").

Then STOP and get the design approved before Phase C (implementation).

### If the user stages the Luthey-Schulten data

Then also:

- Resolve `syn3a_gene_count_dispute` by parsing CP016816 and recording the authoritative count as a `facts/structural/syn3a_gene_count.json` (with the uncertainty fact updated to point at the resolution).
- Promote `syn3a_chromosome_length_pending` to a measured `facts/structural/syn3a_chromosome_length.json`.
- Register the `syn3a_gene_table.csv` (or equivalent) as a data file plus a pointer fact.
- Record the six patched transporter k_cats from `cell_sim/layer3_reactions/nutrient_uptake.py` as `facts/parameters/*.json` with `confidence: estimated` and appropriate caveats.

### Independent of the user's answer

- Commit a small follow-up that drops the unused `torch` import from `cell_sim/layer3_reactions/network.py` (flagged as a warning in `EXISTING_CODE_INVENTORY.md`). This is a one-line cleanup and does not require approval.
- Write a `memory_bank/data/README.md` that documents whatever data-staging convention the user approves.

## Open questions for the user (copied from PHASE_A_SUMMARY.md)

1. **Data staging location** for the Luthey-Schulten files — (a) keep in `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/`, (b) move under `memory_bank/data/`, or (c) external path via env-var / symlink? **Recommendation: (a)** to minimise changes to working code.
2. **Gene-table representation** — one fact per gene, one CSV + one pointer fact, or one giant fact file? **Recommendation: CSV + pointer fact.**
3. **Breuer 2019 essentiality label file** — which supplementary file / format should we treat as the Layer 6 ground truth?
4. **Prototype files at repo root** — leave as read-only history? **Recommendation: yes.**

## Explicitly out of scope

- No Layer 0 implementation code yet.
- No Layer 1-6 work.
- No modifications to `cell_sim/` other than the single `torch`-import cleanup noted above.
- No pulling large data from external sources (NCBI, Luthey-Schulten repo) without user sign-off per brief section 5.3.
