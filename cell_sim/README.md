# cell_sim — Multi-scale JCVI-Syn3A simulator

A multi-layer event-driven simulator for the genetically minimal bacterium
JCVI-Syn3A, built on top of published kinetic data from the
[Luthey-Schulten Lab's whole-cell model](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation).

This is an exploratory research codebase, not a production tool. It was
built to test whether a simpler, event-driven architecture can reproduce
the core behaviors of the full CME-ODE hybrid while keeping the code
small enough to read in one sitting.

## What it does

Simulates a cell at the resolution of individual molecules and discrete
reaction events, with every turnover properly coupled to substrate/product
pools. Produces inline-renderable MP4s showing the simulation over time.

Four layers:

- **Layer 0** — Genome / proteome parser. Reads `syn3A.gb` (GenBank
  CP016816.2), the proteomics xlsx, and complex definitions; produces a
  fully annotated `CellSpec` with 458 real proteins.

- **Layer 2** — Event-driven stochastic simulator (Gillespie). Tracks
  each protein molecule individually. Fires transition rules (folding,
  catalysis, binding, etc.) with per-event propensity.

- **Layer 3** — Reaction network coupling. Full reactant/product
  stoichiometry parsed from `Syn3A_updated.xml` (SBML-FBC), 308 species
  and 356 reactions. Reversible Michaelis-Menten kinetics with real
  `k_cat` and `K_m` values from `kinetic_params.xlsx`. Medium uptake
  from buffered extracellular reservoirs. Transcription, translation,
  and mRNA degradation events.

- **Layer 1** — Atomic / ML physics (stub). Intended for MACE-OFF
  foundation model to estimate `k_cat` for substrates not in the
  kinetic database. Not wired in yet.

## Status

Incremental build log, in rough chronological order:

- [x] Layer 0: parse GenBank, xlsx, SBML → 458 real Syn3A proteins with
      annotations, real initial counts (455 proteins, 158,828 total
      molecules at t=0)
- [x] Layer 2: event-driven Gillespie simulator with per-molecule identity
- [x] Layer 3 stoichiometry: every catalysis event properly consumes
      substrates and produces products via SBML-parsed reactions
- [x] Layer 3 reversibility: forward + reverse rules per reaction with
      proper Michaelis-Menten saturation (k_cat_fwd, k_cat_rev, K_m for
      every substrate/product)
- [x] Medium uptake: 58 transport reactions with 56 buffered extracellular
      species
- [x] Gene expression: transcription, translation, mRNA degradation,
      protein degradation events with real rate constants (85 nt/s
      transcription, 12 aa/s translation, 88 nt/s mRNA degradation)
- [ ] PTS glucose uptake (special rate law, not standard MM)
- [ ] Allosteric regulation and feedback inhibition
- [ ] Ribosome biogenesis from subunits (rates are in `SSU_assembly_raw.json`
      and the LSU Assembly sheets)
- [ ] DNA replication (rates exist for initiation and elongation)
- [ ] Atomic-layer MACE-OFF for novel substrate `k_cat` estimation

## Honest caveats

Read these before interpreting any output:

1. **Scale factor.** Default simulations run at `scale_factor=0.02`
   meaning each real protein count is multiplied by 0.02. This gives
   ~2,700 molecules instead of ~160,000 for tractable per-event
   simulation in-container on CPU. On GPU / full scale, remove the scale.

2. **Propensity approximation.** The Gillespie simulator caps candidate
   token count at `MAX_TOKENS=100` per rule to keep the step cost
   bounded. For highly-abundant substrates (>100k molecules), this
   caps the effective propensity; aggregate rates are correct on average
   but individual timing loses some stochastic correlation structure.

3. **Lumped NTP and AA pools in gene expression.** Transcription is
   modeled as consuming from the ATP/GTP/CTP/UTP pool evenly. Translation
   uses a lumped aa pool via a few amino acids (alanine, glycine, serine,
   leucine) as proxies. The full Luthey-Schulten model tracks each of the
   20 amino acids and 4 NTPs individually.

4. **Translation / transcription as single "completion" events.** A whole
   mRNA is transcribed in one fired event, timed by `gene_length / kcat`.
   Real RNAP moves elongation-step by elongation-step. At steady state
   the result is equivalent; at sub-elongation-time scales it's coarser.

5. **No substrate inhibition or allosteric regulation.** Pyruvate does
   not inhibit PYK, ATP does not activate PFK, etc. Feedback loops are
   missing, so some metabolite pools drift toward zero (e.g., PEP) when
   the real cell would regulate them.

6. **Forward-only reactions still exist.** A handful of reactions have
   `kcat_reverse = 0` in the kinetic data, meaning they cannot
   equilibrate. This is correct for truly irreversible steps (PYK, PFK)
   but can occasionally cause drift on longer timescales.

## Running it

The project assumes the Luthey-Schulten lab's data files are cloned into
`data/Minimal_Cell_ComplexFormation/`. The Colab notebook does this
automatically. For local:

```bash
cd data
git clone https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation.git
cd ..
pip install -r requirements.txt
python tests/render_priority_15.py
```

### Tests / render scripts in order of increasing realism

Each produces an MP4. Pick one based on what you want to see:

- `tests/render_movie.py` — generic toy demo (3 fake proteins, 4 rules)
- `tests/render_real_syn3a.py` — real Syn3A identities, enzymes turn
  over but metabolites don't couple
- `tests/render_coupled.py` — Priority 1: stoichiometric coupling
- `tests/render_priority_15.py` — Priority 1.5: reversibility +
  medium uptake
- `tests/render_priority_2.py` — Priority 2: full central dogma

### Resource expectations

In-container CPU-only benchmarks at `scale_factor=0.02` (2,726 molecules):

| Script | Sim time | Wall time | Events | Events/s |
|-------|---------:|----------:|-------:|---------:|
| render_real_syn3a.py | 0.5 s | ~15 s | 41 k | 2,700 |
| render_coupled.py    | 0.2 s | ~140 s | 194 k | 1,400 |
| render_priority_15.py| 1.0 s | ~100 s | 83 k  | 850 |
| render_priority_2.py | 1.5 s | ~400 s+ | 150 k | 380 |

Expect 5-10× speedup on a decent GPU (or with Numba / Cython — not yet
done). The Luthey-Schulten model on their Delta supercomputer reports
~6 hours for 2 hours of simulated biological time; we're at around
~2.8 minutes of simulated time per hour of wall time on this
unoptimized CPU baseline.

## Directory layout

```
cell_sim/
├── layer0_genome/     # GenBank, proteome, SBML, complexes
│   ├── parser.py
│   └── syn3a_real.py
├── layer1_atomic/     # MACE-OFF wrapper (stub)
│   └── engine.py
├── layer2_field/      # Event-driven simulator
│   ├── dynamics.py
│   └── real_syn3a_rules.py
├── layer3_reactions/  # Metabolic + GEX coupling
│   ├── sbml_parser.py
│   ├── kinetics.py
│   ├── coupled.py
│   ├── reversible.py
│   └── gene_expression.py
├── routing/
│   └── controller.py
├── tests/             # Demos + render scripts
├── docs/
│   ├── DESIGN.md
│   └── BUILD_LOG.md
└── cell_sim_colab.ipynb
```

## Data attribution

All kinetic parameters, proteomics counts, SBML model, complex
definitions, and the annotated genome come from the Luthey-Schulten
Lab's published `Minimal_Cell_ComplexFormation` repository, based on:

- Breuer et al., *eLife* 2019 — essential metabolism model
- Thornburg et al., *Cell* 2022 — whole-cell kinetic model with
  complex formation
- Glass et al. (JCVI), 2016 — JCVI-Syn3A genome (GenBank CP016816.2)
- Zhou et al., *J. Phys. Chem. B* 2025 — augmented WCM with complex
  assembly (current reference: https://doi.org/10.1021/acs.jpcb.5c04532)

This codebase uses those data to test a simpler simulator architecture.
It is not a replacement for the Luthey-Schulten model and should not be
cited as such.

## License

Code: MIT.
Data (when cloned): whatever license the Luthey-Schulten repo uses.
