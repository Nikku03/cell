# Dark Manifold Virtual Cell

Prototype series exploring physics-shaped neural architectures for
cell simulation. Two parallel tracks: a **cell-simulator substrate**
(P0–P10) and a **neural-architecture investigation** (P11–P14).

Both tracks are now at honest intermediate milestones. This README
summarizes what was built, what works, and what remains open.

---

## The central bet

Standard biochemical simulation is expensive (ODE integration with
hundreds of species per cell) or inaccurate (coarse stoichiometric
models). The bet behind this project: **a neural network whose
architecture reflects physics — locality, conservation, unitarity,
field-theoretic structure — learns biology from much less data than a
generic network**.

Two complementary things need to be true for the bet to pay off:
1. Physics-shaped networks can in fact be trained on biochemical data
   and produce accurate dynamics (cell-simulator track).
2. The physics-shaped inductive biases are genuinely load-bearing —
   they beat matched-capacity unconstrained baselines (architecture
   track).

---

## Track 1 — Cell-simulator substrate (P0 → P10)

**What's built:** a compartment-aware spatial simulator of *Mycoplasma
Syn3A* with real biochemistry, real kinetics, open boundaries,
homeostasis, and learned rate prediction plugged into the spatial
simulator.

| Prototype | What it validates | Status |
|---|---|---|
| P0  | Atom conservation via subspace projection | 15 orders of magnitude enforcement |
| P1  | Stamp+flavor embeddings, balanced reactions | 5/5 tests pass, exactly zero drift |
| P2  | Load real Syn3A SBML (304 species, 244 reactions) | 83/244 balanced as written (implicit proton convention) |
| P2b | Auto-rebalance with H⁺ and H₂O | 242/242 reactions balanced after rebalancing |
| P3b | Compartment-aware stamp subspace | Transport visibly moves mass with global conservation intact |
| P4  | Liebermeister convenience kinetics, well-mixed | Central carbon stable; passes stiff-reaction tests |
| P4b | Kinetics coupled to spatial Ψ field | 3/4 pass, stiffness mitigated by rate cap |
| P5  | Boundary fluxes via medium buffering | Self-sustaining cell achieved |
| P6  | Cytoplasmic cofactor buffering for physiological state | ATP converges toward 3.65 mM; central metabolism stable |
| P7  | First learned rate predictor (glycolysis, MLP) | Baseline training loop works; trajectory error ~5% |
| P7b | Tuned learned predictor | Trajectory error 0.40%, rate error 3.93% |
| P8  | Permutation-invariant network (cross-reaction generalization) | Scenario 1: 47% zero-shot on held-out reactions. Scenario 2: full Syn3A scale not yet converged. |
| P9  | LSODA stiff integration | ATP reaches 91% of setpoint; ~50× faster than explicit Euler |
| P10 | Learned rates in spatial simulator (hybrid) | 0.85% trajectory match to pure-hand-coded baseline |

**Overall track completeness: ~62%** against "universal cell simulator
from small data." The pieces not yet built: full Syn3A learned-rate
convergence, multi-organism training, atomic-resolution query
mechanism.

---

## Track 2 — Neural-architecture investigation (P11 → P15)

**What's investigated:** does each architectural primitive proposed
by the Dark Manifold design individually outperform a matched-parameter
baseline on a target PDE where the constraint is supposedly relevant?

| Prototype | Architectural constraint | Target PDE | Result |
|---|---|---|---|
| P11 | Local field evolution (3×3 conv + periodic padding) | Fisher-KPP | **PASS** — 0.87% rollout error |
| P12 | Explicit gauge/connection factorization | Gauged complex Ginzburg-Landau | **NULL** — baseline absorbed gauge implicitly; learned A collapsed to near-constant |
| P13 | Structural unitary evolution | 2D Schrödinger | **PASS, strong** — 3× better accuracy than baseline; renormalized baseline doesn't close the gap |
| P14 | Retrieval-augmented memory bank | Parameter-inference Ginzburg-Landau | **NULL** — attention collapsed to uniform; task wasn't memory-hungry |
| P15 | Mixture-of-K rules / rule discovery | Multi-regime reaction-diffusion | **NULL** — mode collapse; rule-region agreement at chance level |

**Detailed findings:** [`DARK_MANIFOLD_FINDINGS.md`](DARK_MANIFOLD_FINDINGS.md)

**Emergent principle (three-null pattern):** **monolithic structural
invariants** (P11 every-layer conv, P13 every-step unitarity) are load-
bearing. **Modular factorizations** (P12 gauge channels, P14 memory
retrieval, P15 K-rule mixture) consistently get absorbed by the
baseline's implicit capacity and never receive gradient signal to do
work. Three consecutive null results on modular mechanisms is no longer
coincidence — it's a coherent architectural lesson.

**Overall track completeness: ~35%** against the original Dark Manifold
architecture. Two primitives validated; three filed as null results
with a shared diagnosis. The track has reached a natural stopping point.

---

## Getting started

### Setup

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/Luthey-Schulten-Lab/Minimal_Cell.git \
  data/Minimal_Cell
```

Required: numpy, python-libsbml, scipy, torch. Nothing exotic.

### Running

Cell-simulator track (order matters; some prototypes import earlier ones):

```bash
python prototype_p0.py             # fast, ~2s
python prototype_p3b_stamps.py     # ~10s
python prototype_p6_physiological.py   # ~2min; ATP convergence demo
python prototype_p10_learned_spatial.py  # ~3min; learned rates in spatial sim
```

Architecture track (standalone, any order):

```bash
python prototype_p11_neural_pde.py # ~1min; neural PDE baseline
python prototype_p13_unitary.py    # ~1min; unitary evolution
python prototype_p12_gauge.py      # null result, reproducible
python prototype_p14_memory.py     # null result, reproducible
```

Or use [`DMVC_Colab.ipynb`](DMVC_Colab.ipynb) for a clean top-to-bottom
Colab run with explanations between cells.

---

## Files

```
prototype_p0.py ... prototype_p10_learned_spatial.py   Cell-simulator track
prototype_p11_neural_pde.py ... prototype_p14_memory.py   Architecture track
NOTES_P12_null.md, NOTES_P14_null.md   Contemporaneous notes on null results
DARK_MANIFOLD_FINDINGS.md   Consolidated architecture-track findings
DMVC_Colab.ipynb    End-to-end Colab notebook for the cell simulator
requirements.txt, .gitignore, README.md
```

Total: 18 Python files, ~7 000 lines of code. Runs on a laptop CPU.

---

## Honest caveats

Read before concluding anything from this repo:

- The cell-simulator track uses an explicit-Euler rate cap for stiff
  kinetics in P4b–P10. P9 demonstrates LSODA is faster and more accurate
  but hasn't been retrofit everywhere. Not production-quality.
- About 10 polymerization reactions with large stoichiometric
  coefficients are excluded from the kinetic loop. Convenience
  kinetics does not handle them regardless of integrator choice.
- Cytoplasmic cofactor buffering in P6 is a modeling shortcut
  representing homeostatic control systems we don't explicitly model.
- The learned rate predictors in P7, P7b, P8, P10 are trained on
  well-mixed synthetic data. Training on real experimental
  trajectories would be a different project.
- All architecture-track tests use small grids (16×16 to 32×32) and
  modest networks (~20k parameters). Scaling behavior not
  characterized.
- The P12 and P14 null results are specific to their experimental
  setups. Different setups (smaller baselines, non-Markovian tasks,
  richer gauge variation) could yield different results. Each null
  flags a place where more careful experimental design could yield
  a real answer.

---

## Status

**Cell-simulator track:** ready for continued development toward
multi-organism training and full-Syn3A learned kinetics.

**Architecture track:** at a natural pause point. Two clear positives,
two diagnosed nulls, one emergent principle. The honest next step
would be either:
- continue with P15 (rule discovery) carefully designed to require
  the rule-discovery mechanism,
- OR redo P12/P14 with experimental designs that provably stress
  their respective mechanisms,
- OR return to the cell-simulator track and apply the validated
  architectural primitives (P11-style local evolution, P13-style
  unitarity) inside it.

No path is forced. Each represents a different research prioritization.
