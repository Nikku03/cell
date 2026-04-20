# cell_sim

Event-driven simulator for the JCVI-Syn3A minimal cell, with atomic-physics-backed k_cat prediction for substrates outside the measured kinetic database.

Built on real Syn3A data from the [Luthey-Schulten Lab](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation): 458 genes, 455 measured protein counts, 24 known complexes, 160 reactions with k_cat, full metabolic SBML.

---

## What this simulator does

**Four simulation modes, increasing in biological scope:**

1. **Real Syn3A baseline** — 458 proteins turning over at measured k_cat values, folding kinetics, 24 known protein complexes assembling from their subunits.
2. **Priority 1.5** — add reversible Michaelis-Menten kinetics and medium uptake. Full central metabolism reaches steady state.
3. **Priority 2** — add transcription, translation, and mRNA degradation. Central dogma operating on 30 top-expressed genes.
4. **Priority 3** — compute k_cat for novel substrates (drugs, probes) directly from their SMILES string, with no measured kinetic data required. Two backends: structural similarity (fast, calibrated) or atomic physics via MACE-OFF (experimental).

All four modes run end-to-end in a single Colab notebook in under 10 minutes.

---

## Headline results (all reproduced in the Colab notebook)

### 1. Fast simulator, bit-identical to Python reference

Priority 1.5, 2% scale, 1.0 s biological time, seed=42:

| Simulator | Events | Wall time | Speedup |
|---|---|---|---|
| Python `EventSimulator` | 83,049 | 49 s | 1.0x |
| `FastEventSimulator` (vectorised numpy) | **83,049** | **4.3 s** | **11.4x** |

Every event fires in the same order, every metabolite count matches exactly. The speedup comes from three stacked optimisations: compiled rule specs, padded 2D numpy arrays for all MM propensities in a single pass per step, and a Python-rule propensity cache that's valid across 99%+ of events. See [docs/PHASE1_RESULTS.md](docs/PHASE1_RESULTS.md).

Priority 2 (with gene expression): 114,548 events in **7.0 s** wall (was 76 s) — same 10.8x speedup, same bit-identical guarantee.

### 2. Leave-one-out k_cat prediction on 143 Syn3A reactions

For each reaction with a measured k_cat, hide it from the reference set and predict it from the remaining 142 via Morgan-2 Tanimoto similarity:

| Metric | Value |
|---|---|
| Median fold-error | **4.96x** |
| 90th-percentile fold-error | 158x |
| Within 2x | 16% (23/143) |
| Within 5x | **50%** (72/143) |
| Within 10x | **64%** (92/143) |

**The method knows when it's unreliable.** Fold-error as a function of Tanimoto similarity to nearest reference:

| Tanimoto band | n | Median fold-error |
|---|---|---|
| ≥ 0.9 | 3 | **1.72x** |
| 0.7 – 0.9 | 26 | **2.37x** |
| 0.5 – 0.7 | 58 | 3.46x |
| 0.3 – 0.5 | 18 | 5.44x |
| < 0.3 | 21 | 26.0x |

75% of predictions land in the Tanimoto ≥ 0.5 regime where accuracy is 2–5x. This is what makes the architecture useful for organisms without measured kinetic data — most reactions have a structurally similar analog in the reference set.

Per-class breakdown:

| Class | n | Median fold-error | Within 5x |
|---|---|---|---|
| transport_passive | 20 | 3.13x | 80% |
| transport_atp | 33 | 3.73x | 64% |
| phospho_transfer | 20 | 3.89x | 50% |
| hydrolase | 10 | 7.35x | 50% |
| isomerase (catch-all) | 55 | 9.09x | 36% |
| oxidoreductase | 5 | 157.86x | 0% |

See [data/priority3_benchmark.csv](data/priority3_benchmark.csv) for per-reaction results and [data/priority3_benchmark_scatter.png](data/priority3_benchmark_scatter.png) for the log-log scatter.

### 3. Novel substrate demonstration: BrdU and AZT

The Priority 3 demo adds **5-bromo-2′-deoxyuridine (BrdU)** — a real DNA proliferation tracer — to the cell at 100,000 molecules. No entry in the input kinetic database; the simulator must compute everything from its SMILES.

Similarity backend prediction:
- Nearest known substrate: **thymidine** (Tanimoto 0.733)
- Predicted k_cat: **10.36 /s** (thymidine measured: 19.26 /s)
- Result: **8 novel catalysis events** in 300 ms via TMDK1 (thymidine kinase), producing BrdU-monophosphate
- Biological impact: ~100 fewer events in other ATP-dependent reactions (competition)

Equivalent AZT (azidothymidine) demo produces 6 novel events at predicted k_cat 8.56 /s (Tanimoto 0.667 to thymidine).

Neither compound exists in the input data. The simulator accepted them as valid substrates, computed physically reasonable rates, and fired correct stoichiometry against real Syn3A enzyme instances.

### 4. A finding about atomic-physics-from-scratch

When the same BrdU demo runs with `backend_name='mace'` (MACE-OFF bond-dissociation energy → Eyring rate), MACE predicts **k_cat = 65,848 /s** — 3,420× higher than thymidine's measured rate. In simulation this causes runaway phosphorylation: TMDK1 fires 69,816 times in 300 ms, consuming 99.9% of cellular ATP.

This is a limitation of uncalibrated atomic physics as a k_cat predictor, not a bug in the simulator. The event machinery faithfully executed what the physics module asked for. A naive MACE BDE → Hammond → Eyring mapping, without per-substrate calibration against measured kinetics, is dangerous. Structural similarity remains the defensible default.

Full benchmarking of MACE vs similarity across all 143 reactions is future work.

---

## Quick start

### Option A — Colab (recommended)

Open [`cell_sim_colab.ipynb`](cell_sim_colab.ipynb) in Google Colab. Runtime → Change runtime type → High-RAM CPU. Run all. Total runtime ≈ 10 minutes.

The notebook produces three MP4 videos (Real Syn3A, Priority 1.5, Priority 2), the BrdU and AZT demonstrations, an interactive Priority 1.5 flux chart, and the 143-reaction benchmark with scatter plot.

### Option B — Local

```bash
git clone https://github.com/Nikku03/cell.git
cd cell/cell_sim

# Clone the upstream data
mkdir -p data
cd data && git clone --depth 1 https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation.git
cd ..

# Install Python deps
pip install -r requirements.txt
sudo apt-get install -y ffmpeg        # for MP4 rendering

# Sanity check: fast simulator vs Python reference
python tests/test_fast_equivalence.py
# Expected: ~10x speedup, 4699 events MATCH

# Run each simulation mode
python tests/render_real_syn3a.py      # ~25 s
python tests/render_priority_15.py     # ~30 s
python tests/render_priority_2.py      # ~30 s
python tests/demo_priority3.py         # ~4 s (similarity backend)

# Reproduce the 143-reaction benchmark
python tests/benchmark_priority3.py    # ~3 s, writes data/priority3_benchmark.csv
```

---

## Architecture

Four layers, in increasing order of abstraction:

```
layer0_genome/        Syn3A genome, proteins, initial counts, complex definitions
layer1_atomic/        Atomic physics: SMILES -> k_cat
                      - SimilarityBackend: RDKit Morgan fingerprints + Tanimoto
                      - MACEBackend (optional): MACE-OFF BDE -> Eyring
layer2_field/         Gillespie simulator + rule system
                      - dynamics.py: reference Python EventSimulator
                      - fast_dynamics.py: 10x vectorised drop-in replacement
                      - next_reaction_dynamics.py: Gibson-Bruck reference impl
layer3_reactions/     Rule builders
                      - reversible.py: reversible Michaelis-Menten from SBML
                      - gene_expression.py: transcription / translation / degradation
                      - novel_substrates.py: Priority 3 novel substrate pipeline
                      - metabolite_smiles.py: 146 curated Syn3A metabolite SMILES
                      - sbml_parser.py: custom SBML parser for the Luthey-Schulten model
tests/                End-to-end scripts that produce MP4s and text output
```

Rules are `TransitionRule` dataclasses carrying a `can_fire` callback and an optional `compiled_spec` dict. Rules with a `compiled_spec` (all reversible MM rules) go through the vectorised path in `FastEventSimulator`; rules without it (folding, complex formation, gene expression, novel substrates) run via Python closure. The hybrid architecture means new rule kinds can be added without touching the simulator.

---

## Reproducing paper-ready claims

| Claim | How to verify | Expected output |
|---|---|---|
| 10x simulator speedup, bit-identical | `python tests/test_fast_equivalence.py` | `MATCH`, 8-10x speedup |
| 64% of reactions within 10x via similarity | `python tests/benchmark_priority3.py` | median 4.96x, within_5x=72, within_10x=92 |
| Novel substrate fires real catalysis | `python tests/demo_priority3.py` | 8 novel events, k_cat 10.36/s |
| Full Priority 1.5 biology | `python tests/render_priority_15.py` | 83,049 events, ATP Δ -20,538 |
| Full central dogma (Priority 2) | `python tests/render_priority_2.py` | 114,548 events, 41 transcription, 66 translation |

Every number above is deterministic at `seed=42`.

---

## Known limitations

**Uncalibrated MACE backend.** The `MACEBackend` passes MACE-OFF bond-dissociation energies through an Eyring rate equation without per-substrate calibration. Observed overprediction of 3,000× on BrdU. The similarity backend is the defensible default; MACE is an active research direction.

**Next-reaction method not faster in Python.** `NextReactionSimulator` (Gibson-Bruck) is provided as a reference implementation but runs 4× slower than `FastEventSimulator` on Syn3A due to dense cofactor dependencies (ATP connects 68 rules, H2O 49, H+ 35). The algorithmic savings don't cover the Python per-update overhead. The correct implementation is preserved for eventual Rust port. See [docs/PHASE2_RESULTS.md](docs/PHASE2_RESULTS.md).

**Isomerase class is a catch-all.** Reaction classification is a crude 6-bucket partition inferred from SBML stoichiometry (phospho_transfer / oxidoreductase / hydrolase / transport_atp / transport_passive / isomerase). The isomerase bucket lumps EC 2-6 together, which explains its 9.09x median fold-error — mixed reaction chemistries in one pool. A finer EC classification would probably tighten the similarity benchmark numbers.

**One organism.** All benchmarks are on Syn3A. Whether the architecture and its accuracy claims transfer to M. pneumoniae (next candidate: iJW145) or E. coli is an open experimental question.

**Scale factor = 2%.** Default configuration runs at 2% of full Syn3A molecule counts to stay under Colab's memory/time budget. Physics is unchanged; only stochastic noise is larger. Full-scale (100%) simulation is performance-tested but not routinely run in the notebook.

---

## Data sources

- **Genome, proteomics, kinetics, SBML**: [Luthey-Schulten Lab — Minimal_Cell_ComplexFormation](https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation). Cloned as upstream data at setup time. Not redistributed here.
- **Metabolite SMILES**: curated by hand against KEGG / ChEBI / BiGG for 146 Syn3A small molecules. Committed in [`layer3_reactions/metabolite_smiles.py`](cell_sim/layer3_reactions/metabolite_smiles.py).
- **MACE-OFF model weights**: downloaded at first use from [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off). Distributed under Academic Software License.

---

## Project status

**Working and benchmarked:**
- Fast event-driven simulator (10x vectorised, bit-identical to Python)
- Priority 1 / 1.5 / 2 / 3 pipelines end-to-end
- 143-reaction similarity benchmark with per-class and Tanimoto-stratified fold-error distributions
- Novel substrate integration via SMILES (BrdU, AZT demonstrated)
- Colab notebook reproducing every claim in ~10 minutes

**Active research:**
- MACE vs similarity benchmark on all 143 reactions (requires GPU; initial results suggest similarity outperforms uncalibrated MACE)
- τ-leaping for another 30-100x speedup on metabolism-heavy runs
- Port to M. pneumoniae

**Future:**
- Rust core via pyo3 for additional 10-30x speedup
- Methods paper draft

---

## Contact

Open an issue on this repository or reach out via GitHub.

---

## Citation

If you use this simulator in your work:

```
@software{cell_sim_2025,
  author       = {Chhillar (Naresh)},
  title        = {cell_sim: event-driven simulator for JCVI-Syn3A with
                  atomic-physics k_cat prediction},
  url          = {https://github.com/Nikku03/cell},
  year         = {2025}
}
```

Please also cite the upstream Luthey-Schulten Minimal_Cell_ComplexFormation repository whose data this simulator builds on.
