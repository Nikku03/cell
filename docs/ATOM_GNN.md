# AtomGNN — atom-scale message-passing for reactive MD

A small E(3)-equivariant graph neural network that runs on top of the atomistic molecular dynamics engine. Three prediction heads on the same backbone:

1. **Will this atom react in the next time horizon?** (binary classifier)
2. **What kind of reaction?** (15-way multiclass)
3. **What force is acting on it right now?** (3D vector regression, equivariant)

Trained on snapshots from `cell_sim/atom_engine/integrator.py` runs. The trained force head can drive a `NeuralForceField` integrator step (`cell_sim/atom_engine/neural_force_field.py`) in place of the analytic force field — a learned-physics surrogate.

---

## 1. Architecture in one diagram

```
                           per-atom features (12 dim)
                           [element one-hot 8 + valence + speed + n_bonds + |momentum|]
                                          │
                              encoder (Linear → ReLU → Linear)
                                          │
                                          ▼
                                        h ∈ ℝ^hidden
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
             message round 1                                      │
             ┌──────────┐                                         │
             │ msg(h_i, │  edge features:                         │
             │  h_j, e) │  [r, is_single, is_multi, is_bonded]    │
             │          │  (r = pairwise distance)                │
             └────┬─────┘                                         │
                  │ index_add to dst                              │
                  ▼                                               │
             update(h, agg) → h + Δh                              │
                  │                                               │
             message round 2                                      │
                  ▼                                               │
             ┌────┴────┬─────────────┬────────────────┐
             ▼         ▼             ▼                ▼
         binary    reaction      force          force
        head (1)   head (15)     head (3)       head (equivariant)
       "react?"   "what?"      "vector"        "scalar × r̂_ij sum"
```

~4,000 parameters at the default size (hidden=32, n_rounds=2). Tiny — fits anywhere.

`cell_sim/atom_engine/ml_model.py:80` defines `AtomGNN`.

---

## 2. Inputs and outputs

### Per-atom node features (12 dimensions)

`cell_sim/atom_engine/ml_dataset.py:47` (`extract_node_features`):

| Index | Feature |
|---|---|
| 0–7 | one-hot of element (H, C, N, O, P, S, COARSE_HEAD, COARSE_TAIL — defined in `_ELEMENTS_FOR_ML`) |
| 8 | normalised valence remaining (= bonds available / default_valence) |
| 9 | atom speed in nm/ps |
| 10 | n_bonds / 4 |
| 11 | |momentum| |

### Per-edge features (4 dimensions)

`cell_sim/atom_engine/ml_dataset.py` (`extract_all_edges`):

| Index | Feature |
|---|---|
| 0 | pairwise distance r in nm |
| 1 | is_single bond flag |
| 2 | is_multi bond flag |
| 3 | is_bonded (1 if any bond, 0 if proximity-only edge) |

### Heads

| Head | Output shape | Loss | Use |
|---|---|---|---|
| `forward` | (N,) logits | BCE | "will atom i react within horizon?" |
| `predict_reaction_type` | (N, 15) logits | class-balanced CE | reaction-class prediction |
| `predict_forces` | (N, 3) | MSE on per-element-normalised forces | non-equivariant baseline |
| `predict_forces_equivariant` | (N, 3) | MSE on per-element-normalised forces | **the equivariant force head** — see §4 |

Reaction classes are listed in `ml_dataset.py:178` (`REACTION_CLASSES`):
```
other, none, H2, O2, N2, H2O, HO, NH, CH, CH2, CH3, CH4, CO, CO2, NH3
```
Class 0 = "other" (product not in the enumerated list); class 1 = "no event".

---

## 3. Equivariance: the right inductive bias

The plain `predict_forces` head is just an MLP on node embeddings. It can produce force vectors but the network has to learn rotational symmetry from scratch — and given limited data, it usually doesn't, well.

`predict_forces_equivariant` solves this by construction:

```
force on atom i = Σ_j  scalar_ij · unit_vector(r_j − r_i)
```

Where `scalar_ij` is computed from invariant quantities (node embeddings, distance basis-expanded by RBF, bond flags). The unit vector rotates with the geometry. **Result**: rotate the input atoms, the predicted force vector rotates by exactly the same matrix.

This is the same symmetry property NequIP, MACE, and Allegro use. We verify it numerically — see the smoke test in `cell_sim/tests/test_cell_gnn.py:[3]` (the `CellGNN` descendant); error is ~1e-8 in fp32.

---

## 4. Training entry points

`cell_sim/atom_engine/ml_model.py`:

| Function | Trains | Loss |
|---|---|---|
| `train_atom_gnn` (line 267) | binary `forward` head | BCE with `pos_weight=20` (class imbalance: ~95% non-reactive) |
| `train_reaction_classifier` (line 325) | 15-class `reaction_head` | class-balanced CE |
| `train_force_surrogate` (line 403) | non-equivariant `force_head` | per-element normalised MSE |
| `train_force_surrogate_equivariant` (line 501) | equivariant `edge_scalar` → force | per-element normalised MSE |

Default `TrainConfig` (line 227):
```python
epochs = 20
batch_size = 16
lr = 3e-3
hidden = 32
n_rounds = 2
device = "cpu"   # tiny model, CPU is fine
```

---

## 5. The actual training run — concrete numbers

The headline AtomGNN result on this repo lives in `cell_sim/atom_engine/level_up_equivariant_sample.json`. Here's exactly what was trained, on what, for how long, and how well.

### Training dataset

Generated by running the analytic MD engine (`cell_sim/atom_engine/integrator.py`) on synthetic atom soups, then snapshotting:

| Quantity | Value |
|---|---|
| Number of trajectories | **80** |
| MD steps per trajectory | **4,000** |
| Integration timestep | 1 fs |
| **Total simulated time** | **80 × 4,000 × 1 fs = 320 ps of atom-scale MD** |
| Train snapshots extracted | **2,560** |
| Validation snapshots | **640** |
| Atoms per snapshot | small soups, typically 50-100 atoms |
| Soup composition | mixed H/C/N/O/P/S, randomly placed in a confined sphere |

These trajectories are *not* of a cell — they're tiny atom soups designed to produce reactive bond events that the GNN can learn from.

### Training time + epochs (per head)

All 4 heads were trained sequentially on the same 2,560 training snapshots. CPU only — the model is so small (~4K params) that GPU adds no value.

| Head | Epochs | Training metric @ end | Validation metric @ end |
|---|---|---|---|
| **Move 1 — binary `react?`** | 20 | train BCE 1.245 | val AUC **0.833** (baseline KNN was 0.754) |
| **Move 2 — reaction class** | 25 | train CE 0.859 | val top-1 **0.473** / top-3 **0.845** |
| **Move 3 — equivariant force** | 40 | train MSE 1268 (per-element-normalised) | val R² **0.772** (started at 0.421) |
| **Move 4 — cell-essentiality on top of pattern features** | — | train acc 0.742 | val acc **0.770**, val MCC **0.532** |

**Wall-clock training time:** all four heads together completed overnight on a single CPU. Individual training calls (20-40 epochs each) take **5-15 minutes** per head depending on the loss and dataset.

### MD-surrogate benchmark

After training, the GNN's force head was wrapped as a `NeuralForceField` and used to drive an MD integrator for 200 steps, head-to-head against the analytic force field on the same initial conditions. From the same JSON:

| Metric | Native (analytic FF) | Surrogate (GNN) |
|---|---|---|
| Wall time for 200 steps | 0.057 s | 0.327 s |
| Steps / second | 3,538 | 612 |
| Final temperature | 1,792 K | 1,910 K |
| Force MSE vs native | — | 1.25 (per-element-normalised units) |
| Force relative error | — | 19.2% |

**Speedup: 0.17× (i.e. the GNN is ~6× *slower* than the analytic FF at this scale).** Honest read: at 50-100 atoms, the analytic force field is already extremely cheap (3,538 steps/s in Rust); the GNN forward pass dominates. The surrogate's value is methodological (proves learned forces can drive a stable integrator, gives differentiable forces for downstream learning) — not raw speed at this size. It would only pay off at much larger N or where analytic forces aren't available.

### Cell-essentiality task on top — Move 4

A separate downstream experiment (`outputs/atom_engine_full_panel_results.json`) takes the same atom-engine machinery and asks: *can chemistry-pair-level "essentiality" (which element pairs are essential for survival) be detected by running atom soups and measuring bond-event rates?*

Setup:
- 17 element-pair "genes" (HH, CH, HN, HO, CC, CO, etc.)
- Wild-type sim: 8 ps of MD at 3000 K with all pairs allowed
- Knockout: same sim with one element pair forbidden
- Detector watches for which pairs cause >10% deviation from wild-type bond-event counts

| Detector | MCC | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|
| ShortWindow | 0.618 | 0.765 | 6 | 4 | 7 | 0 |
| **PerRule** | **1.000** | **1.000** | 6 | 0 | 11 | 0 |
| Ensemble (OR-high-conf) | 0.696 | 0.824 | 6 | 3 | 8 | 0 |

**Wall-clock for the full panel:** 42 seconds / 17 pairs = **2.4 seconds per gene**. The PerRule detector got perfect (MCC = 1.0) on the chemistry-essentiality task. That's an artefact of the toy setup — real syn3A essentiality from this stack only gets MCC 0.190 (`outputs/atom_engine_on_syn3a_results.json`) because mapping syn3A genes to element-pair verdicts via metabolite formulas is structurally weak (most metabolites contain CHNO, so essential vs non-essential genes look alike under that lens).

### Honest summary of what was demonstrated

1. The GNN learns reactive-bond prediction (AUC 0.83) cleanly — beats baseline.
2. Reaction-class prediction is harder (top-1 47%) but top-3 stays high (84%).
3. Force regression converges to R² 0.77 — the model captures most of the per-element force structure.
4. The GNN-driven MD integrator runs stably but slower than analytic FF at this scale.
5. Chemistry-pair essentiality detection works perfectly on the toy panel (PerRule MCC 1.0); applying that same machinery to real syn3A genes via a metabolite bridge gives only MCC 0.19 — the bridge is the problem, not the GNN.

This is **not a production result for cell biology**. It's a methodological demonstration that learned MD with E(3)-equivariant message passing works end-to-end, with the trained model + integrator + downstream task all wired together.

---

## 6. NeuralForceField — the trained force head as MD surrogate

`cell_sim/atom_engine/neural_force_field.py`

A drop-in replacement for `force_field.compute_forces`. Takes atoms + bonds + cfg + positions, returns (N, 3) forces predicted by the trained AtomGNN. Used like this:

```python
from cell_sim.atom_engine import integrator as integ_mod
from cell_sim.atom_engine.neural_force_field import (
    NeuralForceField, make_surrogate_step,
)

nff = NeuralForceField(model=trained_gnn, mode="equivariant",
                       max_force_kj_per_nm=2.0e4)
with make_surrogate_step(nff):
    for _ in range(100):
        forces = integ_mod.step(state, ff_cfg, int_cfg, forces)
```

The context manager monkey-patches `integrator.compute_forces` to the GNN-driven surrogate for the caller's scope. Caps per-atom force magnitude at `max_force_kj_per_nm` (default 2e4) so spurious extrapolations can't blow up the integrator.

**This is not faster than the analytic force field for small systems**. Its main use is methodological — demonstrating that learned forces can drive a stable MD trajectory and are differentiable for downstream learning.

---

## 7. Per-element force normalisation

`train_force_surrogate_equivariant` (line 501) computes per-element mean and standard-deviation tensors `force_elem_mean` and `force_elem_std` over the training set, attaches them to the model. At inference, `NeuralForceField` un-normalises by element type:

```python
elem_idx = argmax(node_features[:, :N_ELEM_FEATURES])  # which element?
m = model.force_elem_mean[elem_idx]
s = model.force_elem_std[elem_idx]
forces_real = pred_normalised * s[:, None] + m[:, None]
```

Lighter elements (H) have very different force scales than heavy ones (S, P). Per-element normalisation lets the network predict everything in unit-scale and compose with the right magnitudes at decode time.

---

## 8. What this is good for / not good for

**Good for**:
- Demonstrating learned-physics works end-to-end at atom scale
- Predicting reaction events (binary + class) from snapshot states
- Studying which reactions a learned model thinks are next, comparing to ground truth
- A small, well-tested template for cell-scale generalisations (`CellGNN` is the cell-scale descendant — see `docs/CELL_GNN.md` if it exists; otherwise `cell_sim/atom_engine/cell_gnn.py`)

**Not good for**:
- Speeding up MD over the analytic force field. With ~100 atoms, the GNN forward pass costs more than `force_field.compute_forces` directly.
- Long-time stability without the per-atom force cap. Untrained or out-of-distribution states can produce arbitrary force magnitudes.
- Predicting reactions outside the 15 enumerated classes — anything else lumps into `"other"`.

---

## 9. Files at a glance

| File | What it holds |
|---|---|
| `cell_sim/atom_engine/ml_model.py` | `AtomGNN` model + 4 training entry points |
| `cell_sim/atom_engine/ml_dataset.py` | `Snapshot` dataclass + node/edge feature extraction + `REACTION_CLASSES` |
| `cell_sim/atom_engine/neural_force_field.py` | `NeuralForceField` — drop-in surrogate for `force_field.compute_forces` |
| `cell_sim/atom_engine/force_field.py` | the analytic force field this replaces (LJ + harmonic + angle + dihedral + Coulomb) |
| `cell_sim/atom_engine/integrator.py` | the MD integrator that consumes either |
| `cell_sim/atom_engine/atom_unit.py`, `element.py`, `atom_soup.py` | atom data types and initial-state builders |

---

## 10. Relationship to CellGNN

`AtomGNN` operates at atomic scale — 8 elements, ps timestep, ~100 atoms. `CellGNN` (`cell_sim/atom_engine/cell_gnn.py`) is its scaled-up descendant for cell-scale particle dynamics (12+ species, ns timestep, 10⁴-10⁶ particles). The architecture pattern (encoder → message-passing rounds → equivariant force head) is identical. CellGNN adds heterogeneous edge types (SPATIAL, COVALENT, COMPLEX, REACTIVE, ENZYME, MEMBRANE, REGULATORY) to encode the richer biology of cell-scale interactions.

If you understand AtomGNN, CellGNN reads as "AtomGNN with a richer edge vocabulary for biology."
