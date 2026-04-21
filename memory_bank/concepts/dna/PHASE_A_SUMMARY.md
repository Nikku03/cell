# Layer 0 — Phase A Summary

_Phase A is the literature/inventory survey that precedes design. Per the brief section 5.2, no Layer 0 code is written until the user approves this summary and we move to Phase B._

## What we now know

1. **Existing code is a real starting point.** The prior `cell_sim/` and `cell_sim_rust/` have working components that cover more of Layers 0-4 than the project brief's framing implied:
   - A working event-driven engine (`cell_sim/layer2_field/fast_dynamics.py`) — the one the brief says to reuse.
   - A real-Syn3A loader (`cell_sim/layer0_genome/syn3a_real.py`) that already knows how to parse a GenBank file and join it with Luthey-Schulten proteomics/kinetics/complex data.
   - A 10-file Layer 3 reaction suite (reversible MM kinetics, SBML parser, nutrient uptake, novel substrate bridge) ready to be cited and reused.
   - Existing knockout tests (`cell_sim/tests/test_knockouts.py`) — the closest existing analogue to the Layer 6 essentiality analysis.
   - Cross-implementation equivalence tests (fast vs reference simulator, rust vs python) that should be kept green.
   - **No file needs to be thrown away.** Full per-file inventory and verdicts: `EXISTING_CODE_INVENTORY.md`.

2. **Canonical sources are registered.** Three in `memory_bank/sources/`:
   - `genbank_cp016816` — the Syn3A reference genome sequence + feature table.
   - `hutchison_2016_science` — Syn3A design + synthesis + transposon essentiality classification.
   - `breuer_2019_elife` — iMB155 metabolic model + FBA essentiality labels (our Layer 6 validation ground truth, and the MCC 0.59 benchmark).
   - (`thornburg_2022_cell` was pre-registered in scaffolding.)

3. **Gene count is disputed.** Recorded as an uncertainty fact (`facts/uncertainty/syn3a_gene_count_dispute.json`):
   - 452 — Breuer 2019's protein-coding genes covered by the metabolic model.
   - 493 — the brief's citation of "total annotated".
   - 496 — quoted inside the existing code's docstring.
   - 473 — Syn3.0 precursor count (pre-19-gene-restoration).
   - Resolution requires parsing CP016816 directly with a fixed `gene`/`CDS`/`feature` definition.

4. **Chromosome length is approximate.** ~543 kbp per the code's docstring, recorded as `facts/uncertainty/syn3a_chromosome_length_pending.json`. Authoritative number pending CP016816 staging.

## What we still don't have

1. **The GenBank record itself is not on disk.** `cell_sim/layer0_genome/syn3a_real.py` expects it at `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb`; that path is empty. Until we stage CP016816, we cannot:
   - Resolve the gene count.
   - Produce the 452-gene (or 493-gene) table.
   - Locate the origin of replication from the feature table.
2. **The Luthey-Schulten kinetic / proteomic / complex Excel files** are also not on disk. The current code imports them but they are not shipped (license/size). This blocks most of Layers 2-4's parameter registration.
3. **Breuer 2019's essentiality labels** are the Layer 6 ground truth but we have not pinned the specific supplementary file / format they live in. Asking the user to confirm is in `NEXT_SESSION.md`.
4. **Origin of replication position** is not yet recorded.
5. **The six patched transporter k_cats** in `nutrient_uptake.py` have no citation trail. They are usable but must be converted into facts with `confidence: estimated` and a caveat before they can be cited by downstream layers.

## Open questions for the user (required before Phase B)

1. **Data staging.** Where do we put the Luthey-Schulten input files? Options:
   - (a) `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/` — matches what `syn3a_real.py` currently expects. Existing code works unchanged.
   - (b) `memory_bank/data/` — keeps biological data colocated with the memory bank. Requires an indirection in the existing loader.
   - (c) Symlink or env-var pointer to an external path that the user will prepare. Keeps the repo clean.
   - **Recommendation:** (a) for now, because it minimises changes to working code. Revisit if licensing or size becomes a problem.
2. **Gene-table representation.** Options as in `NEXT_SESSION.md`:
   - (a) one fact file per gene (452 files) — enforces memory-bank discipline but heavy.
   - (b) one CSV under `memory_bank/data/` + a single pointer fact (`structural/syn3a_gene_table.json`) — the brief's hint.
   - (c) one giant fact file.
   - **Recommendation:** (b). The invariant checker already runs in <1s on JSON files; the 452-row gene table is structured reference data that belongs in a CSV citable as a single atomic fact.
3. **Breuer 2019 essentiality label file.** Which supplementary file / format should we treat as the ground truth? The user may have a preferred copy; otherwise we pull from the eLife supplementary material.
4. **Should we keep the prototype_*.py files at the repo root?** They are unrelated to the essentiality goal. Recommendation: leave them in place (history) but exclude from all downstream Layer 0+ work. Parked as a footnote; no action needed.

## Status

- Phase A work product: this file, `EXISTING_CODE_INVENTORY.md`, 3 sources, 2 uncertainty facts.
- Invariant checker: `OK` (3 facts, 4 sources, 0 warnings).
- **Ready for Phase B (design) once the user answers the four questions above.**
