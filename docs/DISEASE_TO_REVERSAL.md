# Disease → Reversal — attractor reprogramming as target control

An integration layer that turns the conversation's pathway-reversal idea into a runnable, validated
algorithm. Instead of asking the (weak, ~0.15 recall@10) signature-retriever to *name* the target, we:

1. **Localize** the disrupted regulatory sub-state from the phenotype (seed the network from the disease
   signature and settle it to its attractor).
2. Treat the signed TF→TF network as a **threshold-Boolean dynamical system**; the disease and healthy
   cell states are its **attractors**.
3. **Search** for the minimal set of interventions (TF activations/repressions) that reprogram the disease
   basin back to the healthy basin — "reach the target or a generic corrector."

This is the discrete-dynamics ("Boolean attractor reprogramming") route, the same family as the Cho-lab
cancer-reversion work. It is **topological** (which TFs to flip, in what order), not kinetic (by how much /
how fast) — kinetics is the model's acknowledged wall.

## Files
- `colab/disease_to_reversal.py` — self-contained algorithm (Boolean dynamics, `transition_setdiff`
  baseline, `control_drivers` target-control search, `disease_to_reversal` top-level) + `load_cell_model()`
  adapter that plugs into the NRxBW branch's `cell_complete.json` on Colab.
- `colab/validate_disease_to_reversal.py` — dependency-free validation on GRNs with **known** ground-truth
  drivers → `outputs/orphan/disease_to_reversal_validation.json`.

## What's new vs the existing attractor layer
`compute_attractors.py` reports a transition as *every TF that differs between two basins*, ranked by
out-degree (`transition_setdiff` here reproduces it). That over-reports: it can't tell a causal driver from
a downstream passenger, can't give a **minimal** set, doesn't **order** the interventions, and never
**verifies** that applying them actually reaches the healthy state. `control_drivers` does all four — it
simulates the network's response to each candidate intervention and keeps only those that measurably move
the cell toward health, stopping when the healthy attractor is reached.

## Validation (self-contained, real numbers)

Modular GRNs where the ground truth is known by construction: `M` master TFs each strictly gate a
downstream module; a "disease" flips a subset `D` of masters; flipping exactly `D` cures it and `D` is
minimal. Crosstalk + noise edges pollute the naive baseline while the true drivers stay = `D`.

**[1] Driver recovery — 60 random GRNs**
| metric | control (this module) | set-diff baseline |
|---|---:|---:|
| recall of true drivers | **1.00 ± 0.00** | 1.00 |
| precision (drivers only) | **0.94 ± 0.15** | 0.21 |
| reached healthy state | **98.3%** | — |
| exact driver set (recall=precision=1) | **85%** | — |

Mean barrier (naive # differing TFs) = 9.35 vs mean true drivers = 1.9 — the naive metric overcounts the
causal set ~5×; control finds the real ~2.

**[2] Textbook Boolean motifs** — attractor recovery correct (toggle → {A},{B}; 3-gene lateral inhibition
→ 3 single-winner states; linear cascade propagates the master). Confirms the dynamics are right.

**[3] Single-gene "mutation" disease (canonical case, 50 instances)** — the exact driver is recovered and
the healthy state reached in **96%**, in a **single intervention 96%** of the time.

**[4] Noise robustness** — recall stays ≈1.0 as crosstalk/noise edges are added; precision degrades
gracefully (1.00 → 0.67 at 64 spurious edges) and the cure still reaches health ≥90%.

## Honest scope
- This validates the **algorithm's correctness** on networks with known ground truth. It is **not** a
  biological-accuracy claim on the 16k-gene model — that requires `cell_complete.json` (Colab) and a real
  reprogramming benchmark (e.g. does the transition map recover a known reprogramming factor set held out
  from the seeds). `load_cell_model()` is the hook; the number is pending, exactly like the LINCS test.
- Synthetic GRNs have clean master→module structure; real regulatory networks are messier, so real-world
  precision will be lower.
- Topological, not kinetic. "Reverse the state" is a wet-lab-testable hypothesis, **not** a validated cure
  — a signature/state reversal can in principle be corrective or merely cytotoxic; only driver-classified,
  bench-validated hits should be trusted.
