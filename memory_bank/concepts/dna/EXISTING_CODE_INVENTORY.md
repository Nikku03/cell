# Existing Code Inventory — pre-brief code in /home/user/cell/

_Generated in Session 2, Layer 0 Phase A. Purpose: decide what to reuse vs rewrite before writing Layer 0 code. Per the brief section 8, we reuse working components — we do not rewrite. Verdicts below._

Verdict codes: `keep-asis` | `adapt` | `replace` | `skip`.

## cell_sim/layer0_genome/

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `parser.py` | Generic FASTA/ORF parser; builds a `CellSpec` data structure with proteins, metabolites, reactions (heuristic function classification). | `CellSpec`, `Protein`, `Metabolite`, `Reaction`, `build_cell_spec`, `find_orfs`, `classify_protein` | **keep-asis** — reasonable Layer 0 primitive. |
| `syn3a_real.py` | Loads real Syn3A: parses a GenBank file, loads proteomics / kinetics / complexes from Luthey-Schulten Excel inputs, populates `CellSpec` with protein counts and k_cats. | `build_real_syn3a_cellspec`, `parse_genbank`, `load_metabolites`, `load_complexes`, `ComplexDef` | **adapt** — hardcoded `DATA_ROOT = cell_sim/data/Minimal_Cell_ComplexFormation/input_data/`, which is not on disk. Needs a data-root indirection that points at wherever we stage the Luthey-Schulten files, and a memory-bank-backed way to cite each loaded value. |

## cell_sim/layer1_atomic/

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `engine.py` | k_cat estimation for novel substrates; pluggable `SimilarityBackend` (RDKit) and optional `MACEBackend` (torch + MACE-OFF for bond-dissociation-energy physics). | `AtomicEngine`, `SimilarityBackend`, `MACEBackend`, `KcatEstimate`, `EnzymeProfile` | **adapt** — CPU path is fine; gate the MACE import harder so a missing torch install doesn't print noisy warnings during normal runs. |

## cell_sim/layer2_field/  (the event-driven engine the brief says to reuse)

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `dynamics.py` | Reference event-driven simulator state: `CellState`, `ProteinInstance`, `Complex`, `MetabolitePool`, `TransitionRule`. Python closure-based `EventSimulator` (slow but correct, the source of truth). | `CellState`, `ProteinInstance`, `Complex`, `TransitionRule`, `EventSimulator` | **keep-asis** — reference implementation. |
| `fast_dynamics.py` | Phase 1 vectorized Gillespie: pre-flattened numpy arrays per rule, Michaelis-Menten compiled to vector form. Claimed bit-identical to `EventSimulator`. | `FastEventSimulator` | **keep-asis** — this is the engine the brief says to reuse. Do not rewrite. |
| `next_reaction_dynamics.py` | Gibson-Bruck next-reaction method: priority queue with dependency graph, recomputes only affected propensities. | `NextReactionSimulator` | **keep-asis** — statistically equivalent Phase 2 alternative; not currently critical. |
| `real_syn3a_rules.py` | Builds transition rules from the real Syn3A `CellSpec`: enzyme catalysis with measured k_cats, complex formation, initial protein instantiation scaled from proteomics. | `populate_real_syn3a` | **adapt** — currently sources parameters directly from Luthey-Schulten files with no citation; as we bring parameters into the memory bank, this module should read from bank-backed data. |
| `rust_dynamics.py` | Drop-in wrapper around `FastEventSimulator` that delegates propensity compute to a Rust extension (`cell_sim_rust`). Requires a maturin build. | `RustBackedFastEventSimulator` | **skip (for now)** — premature optimization; revisit only if profiling shows the propensity loop is the bottleneck for a whole-cell run. |

## cell_sim/layer3_reactions/

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `coupled.py` | Metabolite pool utilities: mM↔count conversion, pool initialization, infinite-reservoir species (water, H+, gases). | `initialize_metabolites`, `mM_to_count`, `count_to_mM`, `get_species_count`, `update_species_count` | **keep-asis**. |
| `gene_expression.py` | Priority 2 rules: transcription (precomputed duration), translation at 12 aa/s, mRNA/protein degradation. NTP and amino acid pools lumped; RNAP/ribosomes not tracked as instances. | `build_gene_expression_rules` | **adapt** — transcription/translation are exactly Layers 1 and 2 of the brief. Revisit when we get there — the lumped pools will likely need to be disaggregated. |
| `kinetics.py` | Loader for per-reaction k_cat + Km + reversibility from `kinetic_params.xlsx`. | `ReactionKinetics`, `load_kinetics` | **keep-asis** — but again, add memory-bank citation trail. |
| `metabolite_smiles.py` | Hand-curated BiGG → SMILES mapping for the Syn3A metabolome. Excludes protein-bound species (thioredoxin, ACP). | `BIGG_TO_SMILES` | **keep-asis**. |
| `network.py` | Alternative mass-action ODE integrator (4th-order RK with adaptive stepping). Imports `torch` for "differentiable dynamics" that isn't exercised in the main path. | `ReactionRates`, `default_rates_for_spec` | **adapt** — at minimum, drop the unused `torch` import; at best, evaluate whether this path is needed at all (Layer 4 will use the event-driven engine, not ODEs). |
| `nutrient_uptake.py` | Patches 6 missing transporter k_cats (GLCpts, GLYCt, FAt, O2t, CHOLt, TAGt) that Luthey-Schulten's kinetics file does not cover. Without this, the cell cannot replenish consumed carbon/lipids. | `add_transport_kcats` | **keep-asis but audit** — each of those 6 patched k_cats must become a memory-bank fact with a source and explicit confidence (most will be `estimated`). |
| `novel_substrates.py` | Bridge Layer 1 → Layer 2: register a novel substrate, query the atomic engine for k_cat, spawn catalysis rules. | `add_novel_substrate`, `NovelSubstrateInfo` | **keep-asis**. |
| `reversible.py` | Forward + reverse Michaelis-Menten catalysis with proper Km saturation; transport reactions with a buffered medium. Replaces older naive rules. | `build_reversible_catalysis_rules`, `build_transport_rules`, `initialize_medium` | **keep-asis**. |
| `sbml_parser.py` | Minimal SBML-FBC parser for the iMB155 / `Syn3A_updated.xml` model: species, reactions, reversibility, gene associations. No flux bounds/objectives. | `SBMLModel`, `SBMLSpecies`, `SBMLReaction` | **keep-asis**. |

## cell_sim/routing/

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `controller.py` | Rule-based router that maps biological questions to a layer invocation plan (which layers, duration, wall-time budget). Self-labeled as a 24-hour prototype. | `Router`, `Question`, `SimulationPlan`, `LayerInvocation` | **skip (for now)** — not on the essentiality critical path. Parked in FUTURE_WORK. |

## cell_sim/tests/

| File | What it does | Verdict |
|------|---|---|
| `test_fast_equivalence.py` | `FastEventSimulator` vs `EventSimulator` bit-identity. | **keep-asis** — must stay green. |
| `test_next_reaction.py` | `NextReactionSimulator` statistical equivalence vs Phase 1. | **keep-asis**. |
| `test_rust_equivalence.py` | Rust extension vs pure-Python `FastEventSimulator`. | **skip** — only runs if Rust extension is built. |
| `test_knockouts.py` | Gene knockout tests: remove a gene, assert viability / growth / metabolite behaviour changes. | **keep-asis** — this is the closest existing analogue to Layer 6 essentiality. |
| `test_end_to_end.py` | Integration: Layer 0 → 1 → 2 → 3 pipeline smoke test. | **keep-asis**. |
| `demo_priority3.py`, `render_priority_*.py`, `render_real_syn3a.py`, `render_whole_cell.py`, `render_movie.py` | Demonstration / rendering scripts. | **keep-asis** — pedagogical; not on the critical path. |
| `benchmark_priority3.py`, `compare_uptake.py`, `viz_demo.py` | Profiling and comparison utilities. | **adapt** — re-profile on target hardware before trusting numbers. |

## cell_sim_rust/src/

| File | What it does | Key exports | Verdict |
|------|---|---|---|
| `lib.rs` | PyO3 extension: stateless `compute_propensities(...)` function + stateful `SimCore` PyClass for cached propensities. Uses exact `AVOGADRO = 6.022e23` and banker's rounding for FP-exact replication of Python. | `compute_propensities`, `SimCore` | **skip (for now)** — premature; only relevant once we know the Python propensity loop is the bottleneck for whole-cell essentiality sweeps. |

## cell_sim/data/ and cell_sim/docs/

- `cell_sim/data/priority3_benchmark.csv` — benchmark timing data from prior work.
- `cell_sim/data/priority3_benchmark_scatter.png` — benchmark scatter plot.
- **Missing on disk:** `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/syn3A.gb` and the Luthey-Schulten Excel files that `syn3a_real.py` expects. Staging these is a Phase B task (or we declare a new data dependency and ask the user).
- `cell_sim/docs/DESIGN.md` — prior architecture overview. Read separately as context before writing Layer 0's own `DESIGN.md`.
- `cell_sim/docs/BUILD_LOG.md` — prior build notes / excerpted git log.

## Global warnings

1. **Stray `torch` dependency in `cell_sim/layer3_reactions/network.py`.** The main data path does not use it. Either remove it or isolate it. Brief section 6.1 forbids unexplained dependencies.
2. **Luthey-Schulten input data not in the tree.** Everything in `cell_sim/layer0_genome/syn3a_real.py`, `layer2_field/real_syn3a_rules.py`, and `layer3_reactions/kinetics.py` depends on files under `cell_sim/data/Minimal_Cell_ComplexFormation/input_data/`. None of those files are present. A Phase B decision: where do we stage them, and under what license.
3. **Six transporter k_cats are patched from thin air** in `nutrient_uptake.py`. These must be converted into proper `facts/parameters/*.json` with explicit `confidence: estimated` and a caveat, before we can cite them in Layer 4.
4. **Prototype files at the repo root (`prototype_p0.py` ... `prototype_p15_rule_discovery.py`) are not part of the simulator.** They are exploratory and unrelated to the essentiality goal. We treat them as read-only history.

## Summary verdicts (counts)

- `keep-asis`: 15 files
- `adapt`: 6 files
- `skip`: 4 files
- `replace`: 0 files

**No file needs to be thrown away. The existing cell_sim is a real starting point, not a sunk cost.** Our job in Layers 0-6 is to put a memory-bank-backed citation trail under it and fill in the gaps the current code patches over.
