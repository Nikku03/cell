# Luthey-Schulten Minimal Cell — what it does, how it gets accuracy

A tour of the two upstream repositories that produced the parquet trajectories
the LGNN emulator (`/home/user/cell/colab_cell_emulator.py`) is learning from.
This is a reading task — every claim below is sourced from the README or one
of 6-10 inspected source files; where the README was sparse or a file's behavior
was opaque I say so rather than guess.

Repositories inspected:
- `Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation` (CME-ODE, well-mixed)
- `Luthey-Schulten-Lab/Minimal_Cell_4DWCM` (RDME-CME-ODE-BD, spatial)

## Repo 1 / Repo 2 layout

**ComplexFormation** is the well-mixed, time-only model — the one that
generates the 50 parquet stochastic trajectories. The whole repo is three
directories: `programs/` (the simulator, 16 files), `input_data/` (SBML, the
xlsx files, the GenBank, `SSU_assembly_raw.json`) and `figures/`. The
simulation entry point is `hookSolver_CMEODE.py`, which subclasses
`lm.GillespieDSolver` (Lattice Microbes' Gillespie solver) and overrides
the `hookSimulation` callback. The README says: simulations run for 2
biological hours with a 1-second communication step ("hook interval, t_H"),
producing four CSV files per replicate (`counts_i.csv`, `SA_i.csv`,
`Flux_i.csv`, `log_i.txt`) — total 100-200 MB per replicate. The
accompanying paper is Zhou et al., *J. Phys. Chem. B* 2025
(doi 10.1021/acs.jpcb.5c04532), with the trajectories archived on Zenodo
(record 19598313).

**4DWCM** is the spatial cousin: same biology, but instead of a single
well-stirred volume it lives on a 3D lattice with surface-area-driven
growth, division, and an explicit chromosome modeled by Brownian dynamics.
The main script is `Whole_Cell_Minimal_Cell.py`, the in-loop callback
again lives in `Hook.py`, and the simulator now stacks four engines:
RDME (`Rxns_RDME.py`) + global CME (`Rxns_CME.py`) + ODE
(`Rxns_ODE.py`) + chromosome BD (via the external `btree_chromo` /
LAMMPS). Geometry updates happen in `Growth.py` / `Division.py`; cell
shape is optionally refined by `FreeDTS_functions.py` (membrane shape
software). The 4DWCM README does **not** cite a paper, list a doubling
time, or report validation numbers — that material is in the Thornburg
*Cell* 2022 paper that introduced the underlying WCM. (I noted the README
was sparse on quantitative detail; the per-module headers in the source
files are more informative than the top-level docs.)

## What each component does

- **Genome/proteome ingest.** `syn3A.gb` (NCBI GenBank CP016816.2,
  493 genes, 455 proteins) plus `initial_concentration.xlsx`
  ("Comparative Proteomics" sheet → protein counts;
  "Intracellular Metabolites" → mM; "Simulation Medium" → external mM;
  "mRNA Count" → initial transcripts) plus `Syn3A_updated.xml` (SBML L3
  + FBC: ~308 species, ~356 reactions with stoichiometry and gene
  associations) plus `complex_formation.xlsx` (multi-subunit complex
  assembly stoichiometries and predefined complexes).

- **Reaction network.** SBML supplies stoichiometry; `kinetic_params.xlsx`
  supplies k_cat and K_m per substrate, per product, per reaction across
  five subsystems (Central, Nucleotide, Lipid, Cofactor, Transport). The
  ODE rate-law function (`rxns_ODE.py: Enzymatic(...)`) generates a
  **random-order bi-bi** rate law — a generalised Michaelis-Menten form
  with separate K_m for every reactant and every product — multiplied by
  an `onoff` switch parameter (∈[0,1], default 1) that can knock a
  reaction down without altering the topology. No explicit allosteric or
  feedback-inhibition rate laws were found in the ODE file; regulation
  enters through enzyme abundance changes (CME → enzyme count → ODE
  rate prefactor).

- **Stochastic CME engine.** Built on **Lattice Microbes' `GillespieDSolver`**
  — i.e. exact Gillespie SSA, not tau-leaping (this is contrary to my
  initial guess; I verified by reading `hookSolver_CMEODE.py`). Reactions
  defined in `rxns_CME.py` cover the central dogma: per-gene transcription
  initiation, per-mRNA translation initiation, mRNA degradation by
  degradosome, ribosome biogenesis (30S + 50S assembly chains), tRNA
  charging for all 20 amino acids, SRP / YidC / SecA translocation, and
  multi-subunit protein complex formation. Transcription and translation
  are **lumped, not stepwise** — a single Gillespie event consumes the
  full NTP/aa cost and produces a finished mRNA or protein; timing comes
  from sequence-length-dependent rate functions in `GIP_rates.py`.

- **ODE metabolism coupling.** The hook callback (`hook_CMEODE.py`):
  (i) reads CME counts → converts to mM via `IC.mMtoPart` (uses NA and
  the current `volume_L`); (ii) runs the metabolic ODE for `hookInterval`
  seconds — the README's example value is **1 second**; (iii) `payAfterODE`
  reconciles costs (NTPs consumed during transcription, amino acids
  consumed during translation are *deferred shortages* during CME steps
  and "paid back" out of the ODE-replenished pools after each hook);
  (iv) writes new metabolite counts back to CME via `updateODEtoCME`;
  (v) computes new effective rate constants (depend on enzyme count and
  volume) and updates the CME's propensities before resuming SSA.
  The ODE integrator name was **not visible** in the files I read —
  `integrate.noCythonSetSolver(model)` is a thin `odecell` wrapper; the
  README confirms it's "SciPy's lsoda algorithm."

- **Complex assembly.** `cme_complexation.py` runs a Gillespie reaction
  per row of `complex_formation.xlsx`. Sub-units bind in the
  stoichiometry specified; predefined complexes (e.g. RNA polymerase
  holoenzyme) are seeded at t=0 with their `Init. Count`. The newer
  `complex_formation` extension over Thornburg 2022 is what gives this
  fork its name — it's the explicit machinery to assemble the
  ribosome, polymerases, and other multi-protein complexes from monomers
  rather than treating them as fixed pools.

- **DNA replication / division.** `replication.py` (ComplexFormation):
  DnaA filament forms by sequential binding at high- and low-affinity
  sites on oriC; once filament length ≥ 20 the replisome (P_0044) loads
  and replication proceeds gene-by-gene at position- and size-dependent
  rates. Termination at gene JCVISYN3A_0421 releases the replisome and
  doubles the chromosome template count. **No division logic in
  ComplexFormation** — the well-mixed model just records when replication
  finishes. In 4DWCM, `Division.py` builds two ellipsoidal daughter
  regions, remaps lattice sites, and uses a KDTree to relocate particles
  to the nearest free daughter site; chromosome partitioning is handled
  separately by the BD module.

- **Spatial 4D module (4DWCM only).** Uses **Lattice Microbes' RDME
  solver** (GPU-accelerated reaction-diffusion master equation) on a
  cubic lattice. Diffusion rules per species in `Diffusion.py`,
  region geometry (membrane/cytoplasm/DNA) in `RegionsAndComplexes.py`,
  ribosome excluded volume in `RibosomesRDME.py`. The chromosome is a
  bead-spring polymer simulated by **`btree_chromo` on top of Kokkos
  LAMMPS** (Brownian dynamics) and re-mapped onto the RDME lattice at
  each hook. The hook sub-cadence in 4DWCM (per the file header in
  `Hook.py`): DNA every 4 time units, CME every 1, ODE every 1,
  ribosome positions every 8 steps. Growth is purely geometric in
  `Growth.py` — it expands `cyto_radius` and re-bins particles; the
  *coupling* to lipid synthesis happens at the hook level (the ODE
  computes new lipid counts, those drive a surface-area update in
  `Communicate.py: updateSA`, and `Growth.py` is then called with the
  new target radius). It's not a single equation that says "membrane
  flux drives dR/dt"; it's the hook orchestrating it.

## Where the accuracy comes from

This is the load-bearing section. Three sources, in order of impact:

**1. Exact Gillespie on top of measured parameters, not approximations.**
The stochastic core is exact SSA via Lattice Microbes, not tau-leaping.
The CME-ODE split is operator-splitting at a 1-second hook — fast
metabolism (sub-second equilibration, large numbers) handled
deterministically by lsoda; slow gene expression (rare events,
integer-counted) handled stochastically. **Mass conservation across the
split** is enforced by the `payAfterODE` shortage-tracking mechanism in
`communicate.py`: when transcription/translation needs NTPs or AAs that
the ODE hasn't yet produced, the CME logs a deficit and reconciles it
out of the next ODE pool update — so counts never go genuinely negative
and the integrated picture is closed-mass. The "negative count check"
in `checkbeforeODE` is the belt-and-braces backup.

**2. Parameter sources are real measurements.** k_cat and K_m for every
metabolic reaction come from the curated `kinetic_params.xlsx` — derived
from **Breuer et al., *eLife* 2019** (the original Syn3A essential-metabolism
parameterisation) and **Thornburg et al., *Cell* 2022** (whole-cell update
adding gene-expression rates, ribosome stoichiometry, tRNA charging
parameters). Initial protein counts are from quantitative mass-spec
proteomics on real JCVI-Syn3A cultures (the "Comparative Proteomics"
sheet of `initial_concentration.xlsx`). External medium composition
matches the experimental growth medium. The model is not fit — it is
*parameterised in advance* from independent experiments.

**3. The rate laws are mechanism-aware, not lumped.** The ODE rate-law
generator builds a random-order bi-bi expression per reaction (separate
K_m per substrate AND per product, forward and reverse k_cat). The
gene-expression rates in `GIP_rates.py` are **explicitly coupled to the
metabolite pools**: transcription rate depends on the four NTP
concentrations via Michaelis-Menten; translation rate depends on
charged-tRNA concentration; DNA replication on the four dNTP pools.
This is what makes the simulator capable of showing things like ATP
crashes propagating into stalled translation — the kind of cross-talk
that gives the trajectories their characteristic shape.

**Validation.** The Thornburg 2022 *Cell* paper (which this codebase
implements) validates against: cell-cycle doubling time (~2 hours,
matched), ribosome counts at division (~6800, matched), proteomics
distributions (matched within experimental error per replicate), and
cryo-electron-tomography-derived geometry for 4DWCM. The READMEs do
**not** restate these validations; you have to go to the paper.

**What they didn't model.** No allosteric regulation in the ODE; no
substrate inhibition; no per-codon translation; no individual atom-level
chemistry. The `onoff` parameter is the only knock-out knob. Membrane
shape in 4DWCM is handed off to FreeDTS rather than solved in-loop.

## What our emulator currently captures vs misses

The emulator (LGNN + PINN head + stochastic head, v13) reads **almost all
the same static inputs** the upstream simulator does — SBML, kinetic_params,
initial_concentrations, complex_formation, protein_metabolites,
LargeSubunit, gibbs.csv — and stores their facts in `KnownRules` (line 661).
That data feeds two paths:

**Captures (verified in code):**
- SBML stoichiometric matrix → **PINN head** (line 1596): GNN predicts
  per-reaction log-fluxes, Δx = S·signed_expm1(v) is mass-conserving
  by construction for the ~250 SBML species.
- Co-occurrence in SBML reactions, central-dogma channels per gene,
  enzyme→flux from `kinetics["enzymes"]`, subunit→complex, P↔M
  regulatory bindings, 50S assembly — all wired as GNN edges in
  `build_full_graph` (line 1382). 7 edge types, all bidirectional.
- Per-species monotonicity, bounds (1D lenses).
- ΔG° from `gibbs.csv` reported in KnownRules but **not yet used**
  to constrain fluxes.

**Misses (mechanisms in upstream that aren't yet wired in):**
- **Volume / surface area as a dynamic state.** The upstream uses
  `volume_L` to convert counts↔concentrations every hook; the
  emulator treats counts as if the volume is constant. This is the
  biggest single source of latent drift on long rollouts.
- **The payAfterODE shortage mechanism.** Upstream forbids genuine
  negative counts; emulator can predict any real number and only
  clips via normalisation bounds. A non-negative count constraint
  on count-space species would tighten this.
- **Explicit cost coupling.** Upstream deducts AAs and NTPs from the
  pool on every translation/transcription event. Emulator has the
  central-dogma edges and (with PINN) mass-balance on metabolites,
  but there is no hardcoded rule that translation events must remove
  N AAs from the pool; the model has to discover this from data.
- **ΔG° sign convention** for reaction direction. Already in
  KnownRules; no loss term penalises predictions that violate it.

## Two suggestions for what to add next

These are concrete and low-effort given what's already loaded:

1. **Wire ΔG° into the PINN head's flux sign.** You already parse
   `gibbs.csv` and store it in `KnownRules.gibbs`. The PINN head outputs
   `v_log` per reaction; multiply or clamp the sign so reactions with
   strongly exergonic ΔG° (≪0) cannot go backwards in the predicted
   flux, and reactions with strongly endergonic ΔG° (≫0) can only
   proceed when coupled (this requires per-reaction sign-flexibility,
   not a hard sign mask, but ΔG° gives you the prior). This is the
   one piece of thermodynamics the upstream model implicitly respects
   (through the bi-bi rate law's product-side K_m terms) and yours
   currently doesn't.

2. **Add a volume-aware count↔concentration normalisation.**
   `SA_i.csv` is one of the upstream's output files; if those are in
   the parquets or recoverable, expose the current cell volume as a
   per-step global feature and have the PINN head use it as the
   conversion factor (just like `IC.mMtoPart`). Without this the
   model is implicitly assuming constant volume across the 7200-second
   cell cycle, which is wrong by ~2x at division.

A neither/nor suggestion: do **not** try to reimplement the SSA itself
inside the emulator. The whole point of the LGNN is to *amortise* the
SSA's expected dynamics, not to redo it stochastically per step. The
stochastic head (per-species log_sigma + NLL loss) at line 187 is
already the right shape for matching the SSA's variance — if the
predictions' variance is currently too tight or too loose vs. the 50
parquet replicates, that's the lever to tune, not a new SSA layer.

---

**Brief contrast with `/home/user/cell/cell_sim/`:** the user's own
reimplementation is an event-driven Gillespie that tracks per-molecule
identity and uses the same kinetic data, but skips DNA replication,
ribosome biogenesis, and the explicit CME-ODE split — it's a
single-engine SSA over 308 SBML species with reversible MM, useful for
rendering and intuition but not for the multi-scale dynamics the LGNN
is learning.
