# LGNN Architectural Critiques + M8 Implementation Roadmap

**Status:** Frozen — to be implemented when user signals "go."

**Branch:** `claude/vectorize-gex-propensity-NRqBW`

**Saved on:** end of a session reviewing M5 failure mode and 9 architectural critiques.

---

## Context (what we know up to this point)

Through M0 → M5 we built a count-dynamics surrogate GNN over Syn3A's 8572 species + 45376 edges. M3 hit val_singlestep=0.0350. M5 added 41 static identity features and broke training (val plateau at 0.0590, 11× weight ratio static-vs-dynamic), confirming that name-based features cause shortcut learning.

Knockout-divergence Breuer MCC ceiling: ~0.13 (LGNN), v15 symbolic detector hits 0.530 strict.

The 9 critiques below identify the architectural reasons the LGNN underperforms physics-grounded surrogates.

---

## The 9 critiques (verdicts + engineering cost)

| # | Flaw | File:Line | Verdict | Cost (days) | Subsumed |
|---|------|-----------|---------|------------|----------|
| 1 | CfC at fixed Δt=1 (not truly continuous) | `gnn_v2.py:167` | RIGHT (modulo data is 1s-sampled) | 3-4 | — |
| 2 | Rigid scaffold — 7 hardcoded EdgeKinds + immutable static features | `species_graph.py:51`, `gnn_v4.py:73` | RIGHT (modulo simulator's fixed reaction set) | 5-7 | — |
| 3 | Predicts absolute count not delta | `train_m3.py:209-232` | **WRONG** — we already predict delta via residual `x_pred = x_pred + dx` | 0 | — |
| 4 | Variance σ fed as input (cheat code; deterministic solver + stochastic input) | `train_m3.py` `compute_variance_channel` | RIGHT — needs Neural SDE (predict drift μ + diffusion σ) | 4-5 | — |
| 5 | Softmax destroys mass conservation (per-dst normalization) | `gnn_v2.py:147` `_segment_softmax` | RIGHT — replace with additive (PNA or softplus-bounded) | 1-2 | — |
| 6 | `EdgeKind.FLUX_COUPLING` used for both Species→Flux and Flux→Species | `species_graph.py:289-315` (author already flagged as TODO) | RIGHT — split into `RATE_LAW` + `MASS_BALANCE` | 0.5 | — |
| 7 | Explicit Forward Euler integration `x_pred = x_pred + dx` (unstable for stiff systems) | `train_m3.py` rollout loop | RIGHT — same fix as Flaw 1 (adaptive solver) | 0 | #1 |
| 8 | Stoichiometry fed as MLP "feature" — PINN should hardwire `Δx = S·v` | `species_graph.py:270`, `gnn_v2.py` `msg_mlp` | RIGHT + **HIGHEST IMPACT** — stoichiometric matrix is on disk at `Model/Reaction/StoichiometricMatrix` shape `(5489, 3556)` int32 in every `MinCell_*.lm` | 3-5 | — |
| 9 | "Ghost" central dogma edges bypass resource pools (G→RP without ATP coupling) | `species_graph.py:351-358` `_CENTRAL_DOGMA_PATTERNS` | RIGHT — need bipartite reaction-node topology | 5-7 | — |

**Summary**: 8 of 9 valid (Flaw 3 was incorrect about the code). Total if all implemented: ~3 weeks engineering.

---

## The user's consolidated 5-step roadmap (PRIORITIZED — implement in this order)

### Step 1 — Thermodynamic target shift (`train_m3.py`)

Predict Δ explicitly instead of residual update. Mathematically equivalent but better gradient flow.

```python
# CURRENT (residual form, in train_m3.py rollout loop)
for s in range(k_cur):
    dx = model(x_pred, x_var=v_cur)
    x_pred = x_pred + dx
    target = x_w[:, s + 1, :]
    step_loss = _multi_task_mse(x_pred, target, ...)

# UPGRADE (explicit delta target)
for s in range(k_cur):
    dx = model(x_pred, x_var=v_cur)
    x_pred = x_pred + dx                                    # state still tracked for next step
    delta_target = x_w[:, s + 1, :] - x_w[:, s, :]
    step_loss = _multi_task_mse(dx, delta_target, ...)      # compare dx, not x_pred
```

Why: removes "memorize baseline" gradient pressure. Network learns rates of change, not absolute values.

### Step 2 — Mass conservation patch (`gnn_v2.py`)

Replace softmax with additive pooling. Allows chemical pressures to accumulate.

```python
# CURRENT (gnn_v2.py:147)
alpha = _segment_softmax(logit, dst, N)
weighted = msg * alpha.unsqueeze(-1)

# UPGRADE (softplus — positive, unbounded sum)
alpha = torch.nn.functional.softplus(logit)
weighted = msg * alpha.unsqueeze(-1)
# Self-loop mask still applied via masked_fill before activation
# Self-loop edges: set logit to -inf -> softplus(-inf) -> 0 -> OK
```

**Caveat:** softplus is unbounded; might need `LayerNorm` on aggregated output to prevent activation explosion on high-degree nodes. Consider PNA as alternative if softplus is unstable.

### Step 3 — Decouple thermodynamic arrow (`species_graph.py`)

Split `FLUX_COUPLING` into `RATE_LAW` and `MASS_BALANCE`.

```python
# species_graph.py
class EdgeKind(IntEnum):
    SBML            = 0
    SELF_LOOP       = 1
    TRANSCRIPTION   = 2
    TRANSLATION     = 3
    TRANSLOCATION   = 4
    DEGRADATION     = 5
    FLUX_COUPLING   = 6   # DEPRECATED, kept for back-compat
    RATE_LAW        = 7   # species → flux (concentration drives rate)
    MASS_BALANCE    = 8   # flux → species (rate × stoichiometry alters concentration)

# In the graph construction loop around line 296-315:
for rxn_id, stoich in reactions.items():
    for flux_name in (f'F_{rxn_id}', f'F_{rxn_id}_end'):
        flux_idx = name_to_idx.get(flux_name)
        if flux_idx is None: continue
        for sid in stoich:
            spec_idx = sbml_to_row_idx.get(sid)
            if spec_idx is None or spec_idx == flux_idx: continue
            # flux → species: MASS_BALANCE (with stoich coefficient as attr)
            src_list.append(flux_idx); dst_list.append(spec_idx)
            attrs.append([float(stoich[sid]), 0.0, 0.0, 0.0, 0.0])
            kinds.append(int(EdgeKind.MASS_BALANCE))
            # species → flux: RATE_LAW
            src_list.append(spec_idx); dst_list.append(flux_idx)
            attrs.append([float(stoich[sid]), 0.0, 0.0, 0.0, 0.0])
            kinds.append(int(EdgeKind.RATE_LAW))
```

Also update `N_EDGE_KINDS` in `gnn_v1_axis2.py` from 7 to 9.

### Step 4 — n_nodes purge (`gnn_v2.py` / `gnn_v4.py`)

Replace per-node `cfc_A`, `cfc_B` parameters with projections from static features. Makes model organism-agnostic.

```python
# DELETE from _CfCAttentionGNNLayer.__init__:
self.cfc_A = nn.Parameter(torch.empty(n_nodes, hidden))
self.cfc_B = nn.Parameter(torch.empty(n_nodes, hidden))

# ADD to CellGNNv4.__init__ (or new CellGNNv8):
self.A_proj = nn.Linear(self.n_static_features, hidden)
self.B_proj = nn.Linear(self.n_static_features, hidden)

# In forward(), after computing channels with static features:
A_b = self.A_proj(stat).unsqueeze(0)   # (1, N, hidden) — broadcastable
B_b = self.B_proj(stat).unsqueeze(0)
# Pass A_b, B_b into each layer instead of layer.cfc_A / cfc_B
```

**Critical context from M5 lesson:** the static features must be NORMALIZED before projection (z-score per column) to avoid the 11× magnitude shortcut we measured. The M5 input projection showed static channels totaling 30.7 norm vs dynamic 2.8 norm. Without normalization, this step will reproduce M5's failure mode.

### Step 5 — True adaptive ODE integration (`train_m3.py` + new `gnn_v8.py`)

Wrap forward pass in `torchdiffeq.odeint`. Replaces explicit Euler with adaptive Runge-Kutta-45.

```python
from torchdiffeq import odeint

class DerivativeFunc(nn.Module):
    """Wraps the GNN so it returns dx/dt at a given (t, x)."""
    def __init__(self, gnn_core):
        super().__init__()
        self.gnn = gnn_core

    def forward(self, t, x):
        # t is a scalar; x is (B, N) state
        return self.gnn(x)   # the GNN now outputs dx/dt directly

# In train_m8.py rollout:
deriv = DerivativeFunc(model)
t_eval = torch.arange(k_cur + 1, device=device, dtype=torch.float32)
x_traj = odeint(deriv, x, t_eval, method='dopri5', rtol=1e-3, atol=1e-4,
                 adjoint_params=tuple(model.parameters()))
# x_traj shape: (k_cur+1, B, N), then compute loss against x_w
```

**Caveat:** Use `odeint_adjoint` for memory-efficient backprop through many integration steps. Cost: 5-10× slower than current training.

---

## The M8 architecture spec (post-roadmap)

Combining all 5 steps gives **CellGNNv8 / train_m8.py**:

- **Output**: rate vector `v` (3556 reactions) + per-rate noise scale `σ` (for future SDE upgrade)
- **Forward**: `odeint(dopri5, rtol=1e-3, atol=1e-4)`
- **Mass balance**: hardwired `Δx = S @ v` using `(5489, 3556)` int32 stoichiometric matrix from `MinCell_*.lm`
- **Graph**: SBML + central-dogma reaction edges, with `RATE_LAW` and `MASS_BALANCE` directional split. `EdgeKind.FLUX_COUPLING` deprecated.
- **Aggregation**: softplus-based additive (or PNA if unstable)
- **CfC biases**: `A_proj`, `B_proj` from normalized static features (not per-node)
- **No identity features** (M5 lesson)
- **Loss**: `MSE(v_pred, F_observed) + MSE(x_next_pred, x_next) + conservation_penalty`

Files to create/modify:
- `cell_sim/lgnn/models/gnn_v8.py` — new model class
- `cell_sim/lgnn/data/species_graph.py` — add `RATE_LAW`, `MASS_BALANCE` edge kinds
- `cell_sim/lgnn/data/stoichiometric_matrix.py` — new module to load `S` from `.lm`
- `cell_sim/lgnn/training/train_m8.py` — new training script with `odeint` rollout
- `cell_sim/lgnn/data/static_node_features.py` — add z-score normalization option

Deferred for M9:
- Flaw 2 (dynamic k-NN edges)
- Flaw 9 (bipartite reaction-node refactor)
- Flaw 4 part 2 (Neural SDE — drift+diffusion + `torchsde.sdeint`)

---

## How to resume

When user says "implement the critiques" or similar, the next agent should:
1. Read this file first
2. Verify all 9 critiques against current code (commits may have changed line numbers)
3. Execute steps 1-5 in the prioritized order above
4. Train M8 on RTX 6000 (~3-5 hr expected with adaptive solver)
5. Compare M8 val_singlestep + Breuer MCC vs M3 baseline (0.0350, 0.126)

The user explicitly wants the 5 steps in this exact order. Step 1 (delta target) is cheapest and informative; Step 2-3 are mass-conservation foundations; Step 4 enables organism-agnostic generalization; Step 5 is the heaviest lift (adaptive ODE).

**Baselines to beat:**
- M3 val_singlestep: 0.0350
- M3 knockout MCC strict: 0.126 at top-270
- v15 detector MCC strict: 0.530 (separate methodology, hard target)
