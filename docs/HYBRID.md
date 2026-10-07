# Hybrid — fixed and ML, side by side, interacting

The design principle: **use each method for what it's good at, keep both running, and let them feed each
other.** Fixed methods and learned models are not competitors to pick between — they're complementary lenses,
and the *interaction* between them beats either alone.

## What each is good at

| | Fixed methods | ML methods |
|---|---|---|
| **strength** | interpretable, deterministic, zero training, recover *known* biology, causal tracing | generalize to the unseen, quantitative output, fill gaps, **always learning** |
| **examples** | signed propagation, Boolean attractor dynamics, causal cascade, ec-flux (LP) | learned GNN (link/perturbation), ΔΔG, CatPred, structure→function |
| **weakness** | can't predict what's not in the graph; qualitative | noisier per-call; needs data; less interpretable |
| **use it for** | "what does the known network say?", mechanism, recovery tests | "what's the most likely answer for something not in it?", magnitude |

## The interaction (both directions)

```
   FIXED  ── embedding / priors / structure ──▶  ML   (aids ML accuracy)
     ▲                                            │
     └────── predicted edges / quantities ────────┘   (enriches the fixed graph)
```

### Fixed → ML — *validated*
The fixed method's global multi-hop structure is fed to the learned model as **input features**. Concretely,
the fixed SIGN embedding `[X | SX | S²X]` (leakage-free) is concatenated to the node features and given to the
R-GCN — so the 2-hop GNN also sees the long-range structure it can't reach on its own. Measured on the PPI
link benchmark (`cellgraph_gnn.fixed_input_features`):

| encoder | PPI link AUC |
|---|---|
| fixed SIGN propagation | 0.826 |
| learned GraphSAGE | 0.875 |
| learned R-GCN | 0.886 |
| **hybrid — R-GCN over [features \| fixed embedding]** | **0.893** |

**Feeding the fixed output into the ML lifts it +0.006 over R-GCN and +0.067 over fixed.** Small but real and
above the noise band — the fixed method genuinely aids the ML's accuracy, exactly as intended. Adopted as the
headline encoder (`learned_auc` in `cellgraph_gnn_validation.json`).

### ML → fixed — the return path
The ML's high-confidence **predicted edges** (link prediction AUC 0.89) can be written back into the graph the
fixed methods propagate over — filling gaps the curated network is missing — and the ML's **quantitative
outputs** (ΔΔG, ec-flux %) supply magnitudes the fixed qualitative methods lack. The fixed method stays the
interpretable spine; the ML extends its reach.

## Enriching the shared substrate — ΔΔG and other info into the fixed method

Both engines read the same node features, so enriching those features helps *both*. Planned additions
(◐ needs a per-node structural pass over ~16k AlphaFold structures):
- **mutational fragility** — mean predicted ΔΔG over an in-silico saturation scan per protein (how
  destabilizing the average mutation is → a proxy for how buried/rigid the fold is).
- **structural descriptors** — mean pLDDT, disordered fraction, radius of gyration (AlphaFold).
- **kinetic tier** — CatPred kcat/Km confidence per enzyme.
These make the fixed causal/topological methods stability- and structure-aware, and give the ML richer input —
the same enrichment serves both sides of the loop.

## Why this is the right shape

- Neither method is discarded; the **scorecard keeps both honest** (fixed axes + learned axes, same bar).
- The ML is *always learning* — it improves as data grows — while the fixed spine stays interpretable and
  stable, so answers keep their provenance.
- The interaction is a genuine win, not just an aesthetic: fixed→ML is measured (+0.006), and ML→fixed
  (predicted edges + quantities) extends what the fixed methods can answer.

See `docs/CELLGRAPH_GNN.md` (encoders), `docs/PRODUCT_ARCHITECTURE.md` (full stack), `docs/FUTURE_IDEAS.md`
(the structural-enrichment build).
