# Build Log

Chronological record of what was built, with honest notes on what works,
what's approximated, and what's still missing.

## Phase 1 — Scaffold (early April 2026)

Built the generic four-layer scaffold with synthetic test data.
Toy proteins (`kinase_A`, `substrate_B`, `monomer_C`), 4 rules
(folding, phosphorylation, dimerization, tetramerization),
3-metabolite reaction network. First MP4 rendered showing
~160 synthetic molecules undergoing assembly over 2 simulated seconds.

This phase validated that the event-driven architecture was worth
pursuing, but didn't touch real biology.

## Phase 2 — Pivot to real Syn3A data (mid-April 2026)

Cloned the Luthey-Schulten Lab's
`Minimal_Cell_ComplexFormation` repository, which contains the data
files supporting their Cell 2022 and JPC-B 2025 publications:

- `syn3A.gb` — real annotated genome (CP016816.2, 543 kbp, 458 CDS)
- `initial_concentrations.xlsx` — experimental protein counts (455 proteins)
  and 140 real metabolites with concentrations
- `kinetic_params.xlsx` — 160 metabolic reactions with k_cat and K_m
- `complex_formation.xlsx` — 24 named complexes with gene compositions
- `Syn3A_updated.xml` — SBML-FBC metabolic model, 308 species, 356 reactions
- `SSU_assembly_raw.json` — ribosome small subunit assembly pathway

Built `layer0_genome/syn3a_real.py` to parse all of this into a
`CellSpec`. First real-Syn3A MP4 rendered: 2,726 molecules (at
`scale_factor=0.02`), real locus tags in the event log, enzymes turning
over at their real k_cat values (PYK at 2136/s, PGI at 804/s, etc.).

41,238 catalysis events in 500 ms of simulated cell life. Five real
complexes assembled: SMC at 36.3 ms, Degradosome at 103.4 ms, ECF at
245 ms, aaABC at 324.7 ms, 5fthfECF at 344 ms.

**Caveat flagged at this point**: enzymes turn over but their
catalysis doesn't actually do anything — no substrate consumption, no
product accumulation. This is why Priority 1 exists.

## Phase 3 — Priority 1: stoichiometric coupling

Built `layer3_reactions/sbml_parser.py` (SBML-FBC reader) and
`layer3_reactions/coupled.py`.

Every catalysis event now decrements substrate counts and increments
product counts by the correct stoichiometry from the SBML file. Medium
species (water, H+, extracellular) treated as infinite reservoirs to
avoid integer-overflow churn.

Over 200 ms of simulated cell life with 153 coupled reactions:
- 193,969 events
- ATP: 73,738 → 53,907 (Δ = −0.98 mM)
- ADP: 4,396 → 26,361 (Δ = +1.09 mM, mass-balances ATP)
- Lactate: 2,019 → 9,826 (fermentation output visible)
- PEP: 825 → 0 (depleted — caveat, PEP regulation not modeled)

Works, but reactions are forward-only. Many metabolites drift to zero
because reversibility isn't implemented.

## Phase 4 — Priority 1.5: reversibility + medium uptake

Extended `kinetics.py` to extract both `kcat_forward` and `kcat_reverse`
plus K_m for every substrate and product (160 reactions, 87 marked
reversible, 73 irreversible).

Built `layer3_reactions/reversible.py`:
- For each reaction, generates forward rule at rate `k_cat_fwd`
- If reversible, generates reverse rule at rate `k_cat_rev`
- Both use Michaelis-Menten saturation: `Π c_i/(c_i + K_m_i)` across substrates
- All 56 medium species registered as buffered reservoirs

Result: 232 rules (145 fwd + 87 rev). System enters quasi-steady-state
quickly. Over 1 full simulated second:
- 83,049 events (vs 193,969 in Priority 1 — saturation slows things appropriately)
- ATP: 73,738 → 53,200 (Δ = −1.02 mM)
- PGI: 4,869 fwd vs 4,002 rev → net +867 (very close to equilibrium)
- ATPase: 464 fwd vs 2,062 rev → net −1,598 (running in hydrolysis direction)
- TPI: 2,349 fwd vs 169 rev → near but not at equilibrium

Transport working: 5,525 uptake events over 1 second, including Pi
(499), Mg²⁺ (179), K⁺ (147), isoleucine (166), lysine (143),
leucine (141), serine (102), glutamine (97), threonine (96).

**Caveat flagged here**: PEP still runs to zero because PYK has
k_cat 2136/s forward vs ENO producing PEP more slowly. Real cell
regulates this via substrate inhibition of PYK — not modeled.

## Phase 5 — Priority 2: central dogma

Built `layer3_reactions/gene_expression.py`:
- `make_transcription_rule`: gene → mRNA, rate = 1/elongation_time
  where elongation_time = gene_length_nt / 85 nt/s
- `make_translation_rule`: mRNA → protein, rate = 1/elongation_time
  where elongation_time = protein_length_aa / 12 aa/s
- `make_mrna_degradation_rule`: mRNA → NMPs, rate = 88 nt/s /
  gene_length_nt (first-order decay, Poisson-distributed lifetimes)
- `make_protein_degradation_rule`: protein → amino acids, rate from
  half-life (default 25 min)

Each event also updates metabolite pools:
- Transcription consumes from ATP/GTP/CTP/UTP evenly, produces Pi
- Translation consumes from amino acid pool (via alanine/glycine/serine/leucine
  as proxies for the 20), consumes GTP → GDP for peptide bond formation
- mRNA degradation recycles Pi
- Protein degradation recycles amino acids

Tested at 300 ms simulated with 20 highly-expressed genes: 14 gene
expression events, including actual protein synthesis
(e.g., `JCVISYN3A_0131 (fbaA) synthesized (297aa, cost 594 GTP)`).

**Performance note**: with 317 total rules (232 metabolic + 60 gene
expression + folding + assembly), simulator drops from ~850 events/sec
(Priority 1.5) to ~380 events/sec. In-container CPU 1.5-second
simulations take ~400s wall — too slow to fit in a 5-minute tool-use
window. On GPU or with Numba this becomes tractable.

## What's next

- Priority 2 extensions: wire in individual NTP tracking, 20 separate
  amino acid pools, proper elongation-step events
- Priority 3: MACE-OFF wrapper for novel-substrate k_cat estimation
- PTS glucose uptake (non-standard rate law)
- Allosteric regulation for key enzymes (PYK by pyruvate, PFK by ATP)
- Spatial compartments (cytoplasm / membrane / nucleoid) even without full RDME
- Numba-accelerated Gillespie for ~10× speedup on CPU

## Performance numbers (in-container CPU baseline)

All measured at `scale_factor=0.02` (2,726 molecules):

| Phase | Simulated time | Wall time | Events | Events/sec |
|------|---------------:|----------:|-------:|-----------:|
| Real Syn3A (Phase 2) | 0.5 s | 15 s | 41 k | 2,700 |
| Priority 1 | 0.2 s | 140 s | 194 k | 1,400 |
| Priority 1.5 | 1.0 s | 100 s | 83 k | 850 |
| Priority 2 | 1.5 s (ran into timeout; est.) | ~400 s+ | ~150 k | ~380 |

For reference, the Luthey-Schulten whole-cell model on the Delta
supercomputer reports ~6 hours of wall time for 2 hours of biological
time. We're at roughly 2.8 minutes of simulated biological time per
hour of wall time on this CPU baseline — about three orders of
magnitude slower per simulated second, which is expected for
unoptimized Python running at ~1% of real-scale molecules.

## Phase 6 — Priority 1 deprecated (late April 2026)

Priority 1 was removed from the active codebase. Priority 1.5 turned
out to be a strict superset: same catalysis events, same stoichiometry,
but with reversibility and medium uptake added. Priority 1 on its own
was misleading as a deliverable because substrate pools drained to zero
in ~200 ms when the full simulator would hold steady state.

What was removed:
- `tests/render_coupled.py` — the Priority 1 renderer
- `build_coupled_catalysis_rules()` and `make_coupled_catalysis_rule()`
  in `layer3_reactions/coupled.py` — the naive forward-only rule builders
- The Priority 1 section of `cell_sim_colab.ipynb`
- The Priority 1 row of the benchmark table and references in README

What was kept:
- `layer3_reactions/coupled.py` — now pure metabolite utilities
  (count/mM conversion, `initialize_metabolites`, `get_species_count`,
  `update_species_count`, `INFINITE_SPECIES`). These are shared
  infrastructure used by both `reversible.py` and `gene_expression.py`,
  so the file stays; only the Priority 1 rule builders were stripped.
  The file went from 401 lines to 131.
- Phase 3 above — the historical record. Priority 1 existed as a real
  build step and it's worth keeping that accurate in the log. The
  current state is "superseded by Priority 1.5", not "never existed".

The user-visible change for people running the notebook: three cells
fewer and faster end-to-end run, since Priority 1.5 already does
everything Priority 1 did and more.
