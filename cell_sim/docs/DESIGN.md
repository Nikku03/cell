# Design

A multi-layer event-driven simulator for JCVI-Syn3A, built on real kinetic
data from the Luthey-Schulten Lab's published whole-cell model.

## Core architectural decision: events, not fields

An earlier iteration of this project (late March 2026) tried a 4D
spacetime neural-field simulator. We abandoned it because field dynamics
don't match how the biology actually is measured — everything in the
published data is rate constants on discrete molecular events, not
continuum fields. The event-driven design matches the data directly and
lets us inherit 10 years of Luthey-Schulten kinetic parameter work
without fighting a mismatched abstraction.

Event-driven also solves the "how much atomic detail" question
elegantly: catalysis events can carry an optional atomic-scale callback
(Layer 1) to recompute k_cat for novel substrates, without demanding
that the whole cell run at atomic resolution.

## Four layers

### Layer 0 — Genome / proteome ingest

Input: `syn3A.gb` (GenBank CP016816.2), `initial_concentrations.xlsx`,
`complex_formation.xlsx`, `kinetic_params.xlsx`.

Output: a `CellSpec` dataclass with 458 proteins (real locus tags, gene
names, products, EC numbers, translations), 140 metabolites with
concentrations, 24 known complex definitions with subunit stoichiometry,
and 160 reactions with k_cat values.

Function-class heuristic splits proteins into enzyme / ribosomal /
transport / rna_processing / regulatory / structural_division /
unknown / other — not perfect (90 proteins of unknown function exist in
Syn3A itself) but good enough for visualization.

Single file: `layer0_genome/syn3a_real.py`. ~230 lines. Uses Biopython
for GenBank parsing and pandas for xlsx.

### Layer 1 — Atomic / ML physics (stub)

Intended as the MACE-OFF foundation model wrapper. When wired in,
Layer 3 rules that need a k_cat for a substrate not in the database
would call `layer1_atomic.engine.estimate_kcat(substrate, enzyme)` and
get back a value from MACE. Currently stubbed — the wrapper is in
place but the model isn't loaded.

This is deferred because no one has time-sensitive need for it and
because real MACE-OFF-from-scratch k_cat prediction is itself an open
research problem.

### Layer 2 — Event-driven stochastic simulator

`layer2_field/dynamics.py` — Gillespie algorithm with per-molecule
identity tracking. Each `ProteinInstance` has a unique ID, conformation
state (`unfolded`, `native`, various bound states), and lineage tracking.

The simulator's main loop is standard Gillespie:
1. Query every `TransitionRule` for its current propensity via `can_fire(state)`
2. Sample time to next event from exponential distribution
3. Choose which rule fires with probability proportional to propensity
4. Apply the rule's `apply(state, candidates, rng)` mutation
5. Log the event with full context (time, participants, description)

Performance note: propensity computation scales with rule count × average
candidate count per rule. We cap candidate tokens per rule at 100 to
bound step cost. This is an approximation to true mass-action Gillespie
but preserves aggregate rates.

Single file: `layer2_field/dynamics.py`. `layer2_field/real_syn3a_rules.py`
builds the standard rule set (folding, catalysis, complex assembly) from
the Syn3A data.

### Layer 3 — Reaction network coupling

Four files, each adds a layer of biological realism:

- `sbml_parser.py` — reads `Syn3A_updated.xml`. 308 species, 356
  reactions with full stoichiometry and gene→reaction associations.

- `kinetics.py` — reads the full `kinetic_params.xlsx`. For each of
  160 reactions: forward k_cat, reverse k_cat (0 if irreversible), and
  K_m for every substrate and product. Also loads the 56-species
  medium composition.

- `coupled.py` — metabolite utilities. Manages counts, mM↔molecule
  conversion, and the infinite-reservoir list (water, H+,
  extracellular nutrients). Rules from both `reversible.py` and
  `gene_expression.py` call into it to read and mutate metabolite
  state during event firing. An earlier Priority 1 simulator
  (forward-only stoichiometric coupling) lived here but was superseded
  by `reversible.py`; the file is now pure utilities.

- `reversible.py` — the main simulator. Builds forward + reverse rules
  per reaction using Michaelis-Menten saturation factors. Medium species
  registered as buffered reservoirs. This is the version where the
  simulator runs stably at biological steady state for seconds of
  simulated time.

- `gene_expression.py` — adds transcription, translation, mRNA
  degradation, protein degradation events using the Gene Expression
  sheet rates (85 nt/s transcription, 12 aa/s translation, 88 nt/s
  degradation).

## How the layers compose

No routing controller. The event simulator is the control flow — rules
from every layer compete in the same Gillespie loop. A gene expression
event fires at the same priority level as a catalysis event, with
propensity from its own k_bind × substrate availability.

This is simpler than the "routing" idea from the original design doc,
and it's correct: there's no biological hierarchy that says metabolism
runs faster than transcription. They all happen concurrently at rates
set by physics.

## What we deliberately didn't build

- **Spatial layout.** Every molecule is in a single well-mixed
  compartment. Real cells have nucleoid, membrane, cytoplasm with
  distinct diffusion. Luthey-Schulten's 4D Whole-Cell Model uses RDME
  (reaction-diffusion master equation) with Lattice Microbes; we
  didn't. Visualization uses fake 2D positions for watchability, not
  real spatial dynamics.

- **Allosteric regulation and feedback.** PYK isn't inhibited by pyruvate,
  PFK isn't activated by ATP, etc. The data exists in the literature
  but isn't in the kinetic_params.xlsx we inherited.

- **Ribosome biogenesis.** The SSU and LSU assembly pathways exist in
  `SSU_assembly_raw.json` and the LSU Assembly sheets but we haven't
  wired them in. We treat ribosomes as a fixed pool.

- **DNA replication.** Rates exist (100 bp/s elongation, binding
  affinities). Not wired in.

- **Substrate inhibition and cooperativity.** No Hill coefficients,
  no multi-substrate kinetics beyond the product rule in
  `mm_saturation_factor`.

## Honest assessment of what this is and isn't

This is a working architecture demonstration that shows the event-driven
abstraction can reproduce published whole-cell kinetics using the real
data. It is not a scientific instrument — the Luthey-Schulten model
with Lattice Microbes is the scientific instrument.

The value of this codebase is (a) it's 322 KB of Python, readable in
one sitting, and (b) it's architected to allow atomic-scale callbacks
into MACE-OFF, which is genuinely novel.

## File layout

```
cell_sim/
├── layer0_genome/
│   ├── parser.py             # generic (synthetic) genome parser
│   └── syn3a_real.py         # real Syn3A loader
├── layer1_atomic/
│   └── engine.py             # MACE-OFF stub
├── layer2_field/
│   ├── dynamics.py           # Gillespie simulator
│   └── real_syn3a_rules.py   # standard rule builders
├── layer3_reactions/
│   ├── sbml_parser.py        # Syn3A_updated.xml reader
│   ├── kinetics.py           # kcat_fwd, kcat_rev, Km extractor
│   ├── coupled.py            # metabolite utilities (counts, mM↔N, reservoirs)
│   ├── reversible.py         # main simulator: reversible MM + medium uptake
│   └── gene_expression.py    # Priority 2 (central dogma)
├── routing/controller.py     # skeleton only; not used in main path
├── tests/                    # 5 render scripts, one per priority level
└── docs/
    ├── DESIGN.md             # this file
    └── BUILD_LOG.md          # chronological build history
```
