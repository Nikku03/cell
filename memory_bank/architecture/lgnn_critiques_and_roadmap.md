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

---

## Addendum: 3 PINN-prerequisite critiques (post-roadmap)

Three more critiques flagged as prerequisites before implementing the
M-PINN architecture. My honest verdicts:

### Critique 10: Log-space PINN collision

`Δx = S · v` is a LINEAR operation. The LGNN operates in signed-log1p
space (counts span 6 orders of magnitude, linear gradients would
explode). Applying the stoichiometric equation directly in log space is
mathematically wrong: `log(x) + S · log(v) ≠ log(x + S · v)`.

**Verdict: agree.** The fix is an explicit exponential bridge inside
the PINN output head:
```
v_linear      = signed_expm1(v_log)
x_linear      = signed_expm1(x_log)
dx_linear     = S @ v_linear                          # exact, hardwired
x_next_linear = x_linear + dx_linear
x_next_log    = signed_log1p(x_next_linear)
```

**Gradient analysis** (why this is stable): `∂x_next_log/∂v_log =
(1/x_next_linear) · S · v_linear ≈ S` because `x_next_linear ~
v_linear` in magnitude for non-tiny species. The exp and log
derivatives cancel by design.

**Status: IMPLEMENTED** as `cell_sim/lgnn/models/pinn_head.py`
(class `PINNHead`, helpers `signed_log1p`, `signed_expm1`,
`build_flux_indices`). Stoichiometric matrix loader at
`cell_sim/lgnn/data/stoichiometric_matrix.py` (loads from
`Model/Reaction/StoichiometricMatrix` in any `MinCell_*.lm` and maps
the 5489 spatial species onto the LGNN's 8572 row order, zero-padding
the 3083 LGNN-only species like RP_*, PM_*, DM_*).

Unit-tested: roundtrip accuracy 6e-5 fp32, mass conservation
violation 1.5e-7 for stoichiometry-zero reactions, gradients finite
and bounded even at log-magnitude 8 (count ~3000).

### Critique 11: Multi-task gradient collision (PINN double-dipping)

Once the PINN enforces `Δx = S · v`, the count target is
deterministically bound to the flux target. Supervising both
double-dips the gradient on the same physical mechanism; if the
training data has Gillespie stochasticity, the count loss can
actively fight the flux loss.

**Verdict: agree with nuance.** "Remove count loss entirely" is too
strong — a small count regularization weight helps prevent the model
from drifting into negative concentrations (which the bridge's expm1
allows). The right setting for M-PINN training:
```
weight_flux       = 1.0    # primary supervision on v
weight_cumulative = 0.5    # secondary (cumulative counters as observed
                            # rate counters complementary to F_*)
weight_count      = 0.05   # regularization only - prevents negative
                            # predicted counts via x_next_log MSE
```

This contrasts with M3's `weight_flux=40, weight_count=1,
weight_cumulative=5`, which was tuned for the non-PINN regime where
count is the primary signal.

**Status: documented for use in `train_m_pinn.py` when built.** Not
implemented yet because the PINN training script doesn't exist; this
config goes in there.

### Critique 12: Bipartite reaction-node mandate (pure topology agnosticism)

Without static features (post-M5 lesson), the only thing teaching the
model what each species "is" is the graph topology. A direct G → RP
edge teaches transcription as a two-body topological correlation,
bypassing ATP/RNAP/nucleotide-pool dependencies. The fix is bipartite:
gene → [TranscriptionReaction] → transcript, with ATP/GTP/etc.
connected to the reaction node.

**Verdict: agree in principle, disagree on implementing now.** The
M-PINN's stoichiometric matrix ALREADY encodes which species are
substrates and products for each reaction including resources — that's
literally what S contains. Mass balance correctness is enforced by
`Δx = S · v` at the output regardless of the GNN's message-passing
graph topology. The bipartite refactor would improve message-passing
efficiency (the model would learn faster) but not correctness. Worth
doing eventually; not a blocker.

**Status: documented; not implemented.** If after training M-PINN we
find the GNN encoder struggles with resource dependencies, we'll come
back to this.

---

## Updated tally (12 critiques total)

| # | Source | Verdict | Implemented? |
|---|--------|---------|--------------|
| 1 | CfC → Neural ODE | RIGHT (modulo data res.) | No - defer to M-PINN |
| 2 | Rigid scaffold | RIGHT (modulo simulator) | No - defer |
| 3 | Absolute vs delta | **WRONG** (already delta via residual) | N/A |
| 4 | Variance cheat | RIGHT | No - separate experiment |
| 5 | Softmax mass conservation | DISAGREE for hidden state | No - rejected |
| 6 | RATE_LAW / MASS_BALANCE | RIGHT | **YES** (M6) |
| 7 | Forward Euler | RIGHT (subsumed by #1) | No - subsumed |
| 8 | PINN with hardwired S | RIGHT + highest impact | Scaffolding done, training script pending |
| 9 | Bipartite reaction nodes | RIGHT (deferred) | No - deferred |
| 10 | Log-space PINN bridge | RIGHT | **YES** (pinn_head.py) |
| 11 | PINN loss double-dipping | RIGHT with nuance | Documented for M-PINN config |
| 12 | Pure-topology bipartite mandate | DISAGREE on doing now | No - deferred |

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
