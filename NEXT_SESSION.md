# NEXT_SESSION — queued work

_Read this file FIRST, immediately after running the invariant checker._

## Pre-flight

1. `python memory_bank/.invariants/check.py` — must print `OK`.
2. Read `PROJECT_STATUS.md`.
3. Read `memory_bank/concepts/dna/REPORT.md` and `memory_bank/concepts/essentiality/REPORT.md` for what's done.
4. Read this file.

## Current state (summary)

- Layer 0 is **done**. `cell_sim.layer0_genome.genome.Genome` is the canonical API; 12 validation tests pass.
- Layer 6 skeleton is **done** and tested against a MockSimulator. 13 tests pass. The detection logic, the MCC harness, and the Breuer 2019 label pipeline all work.
- Layer 6 is **blocked** on wiring the real `FastEventSimulator` backend into `KnockoutHarness`. That is the biggest single unlock.
- Layers 1-5 have partial implementations in existing `cell_sim/` code but lack memory-bank citations.

## Highest-priority queue (pick from the top)

### 1. Wire the real simulator into Layer 6 ("Layer 6 Phase C completion")

Goal: replace `MockSimulator` with a `RealSimulator` that wraps:

- `cell_sim.layer0_genome.syn3a_real.build_real_syn3a_cellspec` — loader.
- `cell_sim.layer3_reactions.sbml_parser.parse_sbml` + `kinetics.load_all_kinetics` — reaction setup.
- `cell_sim.layer2_field.real_syn3a_rules.populate_real_syn3a` — rule builder.
- `cell_sim.layer2_field.fast_dynamics.FastEventSimulator` — runtime.
- `cell_sim.layer3_reactions.nutrient_uptake.build_missing_transport_rules` — the gap-filling transporters.

Required adapters:
- **Sample grid**: the event-driven sim is continuous-time; `Trajectory` expects regular samples. Add a sampling adapter in `cell_sim/layer6_essentiality/real_simulator.py` that snapshots pool counts every `sample_dt_s` bio-time.
- **Knockout semantics**: remove rules whose primary gene product is in the knockout set. The existing `test_knockouts.py` (at repo path `cell_sim/tests/test_knockouts.py`) already demonstrates the pattern — lift it.
- **Pool key schema**: `FailureDetector` expects keys `ATP`, `CHARGED_TRNA_FRACTION`, `NTP_TOTAL`, and the ones in `ESSENTIAL_METABOLITES`. Aggregate them from the sim's species counts in the adapter.

Acceptance test: `test_real_simulator_small_sweep` that runs 4 genes (2 Breuer-essential, 2 non-essential — pgi / ptsG / ftsZ / 0305 as in the old knockout test) and asserts the essential ones come back `essential=True`, non-essential `essential=False`. Runtime budget: <15 min for 4 genes at 25% scale.

### 2. Genome-wide sweep + MCC measurement

Once (1) passes: run `run_sweep` over all 458 CDS at 25% scale (or 50% if compute allows). Write `predictions.csv`. Compute MCC. Record as `memory_bank/facts/measured/mcc_against_breuer_v1.json`. This is the first quantitative check of the brief's 0.59 target.

### 3. Six patched-transporter-k_cat facts

`cell_sim/layer3_reactions/nutrient_uptake.py` adds k_cats for GLCpts, GLYCt, FAt, O2t, CHOLt, TAGt without citation. Create `memory_bank/facts/parameters/transporter_kcat_<gene>.json` for each, marked `confidence: estimated`, with the existing code file in `used_by`. No new biology work — just the paper trail.

## Lower-priority queue

### 4. Torch-import cleanup
Drop the unused `torch` import from `cell_sim/layer3_reactions/network.py`. One-line diff; no approval required.

### 5. Layer 5 (biomass + division) Phase A
Register `thornburg_2022_cell` as the source of the biomass composition vector; draft a fact for the 55% protein / 20% RNA / 3% DNA / 10% lipid / 12% other split. Without Layer 5, the sweep's "one doubling time = 7200 s" cutoff is the only proxy for viability.

### 6. Hutchison 2016 secondary label set
Layer 6 currently benchmarks against Breuer 2019. Hutchison 2016 transposon labels are a secondary ground-truth; adding them lets us report MCC against both and flag genes where the two sources disagree.

## Deferred (not this session)

- Rust hot paths (`cell_sim_rust/`).
- Neural-net anything (brief section 2 forbids it without justification).
- Multi-gene knockouts, synthetic lethality.

## Git state

Unpushed commits on `claude/syn3a-whole-cell-simulator-REjHC`:
- `900b79d` scaffolding
- `d265fa3` gitignore patches/
- `6916aca` Layer 0 Phase A (inventory + sources)
- `ba58794` GenBank parse + structural facts
- `0ad3190` Layer 0 complete (Genome API + tests + docs)
- `ccdd53a` Layer 6 + Layers 1-5 triage + Breuer labels

Push is still blocked on GitHub auth. Patch backups live in `patches/*.patch` if the sandbox state ever needs rebuilding.

## Open questions still worth the user's attention

- Are the autonomous defaults in `memory_bank/concepts/dna/DECISIONS.md` acceptable? If you want to move data elsewhere or restructure the gene table, say so now before more code depends on these paths.
- Confirm the Luthey-Schulten repo licensing is permissive enough for our local staging. If not, we need an alternative staging path (user's own clone) before the sweep can run reproducibly on another machine.
