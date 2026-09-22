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

## Real-model validation (the human TF network)

`colab/validate_reversal_realmodel.py` runs the same search on the **real signed TF→TF regulatory core**
of the 16,492-gene human model — **1,146 TFs / 9,906 signed edges** (CollecTRI/DoRothEA-derived), extracted
from the provided `cell_explorer.html` → `outputs/orphan/tf_core.json`. Ground truth = **published
reprogramming recipes** (experimentally-validated master-TF sets that convert one cell type to another).

- **[A] Attractor landscape (honest limitation):** unconstrained Boolean dynamics are *activation-dominated*
  — 120 random seeds give 120 different near-fully-ON states (median 956/1,146 active). The network does
  **not** self-organize into a few clean cell-type attractors from random starts.
- **[B] Lineage coherence (positive):** clamping each published recipe from an all-OFF baseline yields a
  **compact attractor** (58–80 active TFs); iPSC separates from soma (Jaccard 0.52); and **no foreign-lineage
  master ever turns on** in a lineage attractor (lineage-exclusive). So the *anchored* attractor framing is
  biologically coherent — which is how it is used (you know the disease state and the target state).
- **[C] Reprogramming recovery (the headline):** over 12 cross-lineage transitions, the control search —
  choosing among all differing TFs with **no privilege for the recipe** — recovers the known factors at
  **recall 0.97 vs a random-TF baseline of 0.18 = 6.2× enrichment**, reaching the target attractor in 8/12.
  It returns the textbook recipes verbatim: soma→iPSC → **OCT4/SOX2/KLF4/MYC** (Yamanaka); →cardiac →
  **GATA4/MEF2C/TBX5**; B-cell→macrophage → **CEBPA/SPI1** (the Xie/Graf conversion); →hepatocyte →
  **HNF4A/FOXA1/FOXA2**; myeloid→B-cell → **PAX5/EBF1/TCF3**.

  | metric | value |
  |---|---:|
  | recipe-factor recall | **0.97** |
  | random baseline | 0.18 |
  | **enrichment** | **6.2×** |
  | reached target attractor | 8/12 (0.67) |
  | mean path length | 4.8 |

  Honest caveats: the target attractor is *anchored* by clamping B's recipe, so "clamp B's factors reaches
  B" is partly built in — the non-trivial content is that the search picks the recipe factors over random
  differing TFs (6.2×) and finds them minimal. The failures are reported, not hidden: iPSC→soma exits fail
  to fully settle (pluripotency is a deep basin), and iPSC→neuron recovers ASCL2 (a paralog) instead of
  ASCL1. Topological, not kinetic; a recovered factor is a model-consistent hypothesis, not proof.

## Interactome-robustness (Robust axis)

The human interactome is incomplete (Menche 2015) and this TF network is **83% activating / 17% repressing**
(45% of nodes have zero repressive input — curated regulatory DBs under-annotate repression). A prediction
that only holds on the exact edge set is fragile. `bootstrap_stability()` resamples the network (drop 15% of
edges, K resamples), re-runs the reversal, and scores each driver by selection frequency; `flag_by_stability()`
splits robust from fragile.

**Result (`cardiac→iPSC`, drop 15% ×6):** the recipe factors POU5F1/SOX2/KLF4/MYC survive at **stability 1.0**;
the passenger MEF2C at **0.33** and is flagged fragile. The bootstrap cleanly separates real drivers from
passengers (recipe 1.0 vs passenger 0.33), so every recovered driver now carries an honest stability score
instead of a bare point prediction.

**A tested negative (kept for honesty):** we also tried fixing the activation-dominated Boolean dynamics with
absolute/degree thresholds. A positive threshold reduces the random-seed ON-collapse only marginally
(71%→63%) and **regresses reprogramming recall (1.00→0.72)** — because any rule that suppresses the activation
runaway also suppresses the legitimate master→target activation reversal depends on. The activation-dominance
is a **data-completeness property**, not a dynamics-rule bug, so no threshold change was committed.

## Honest scope
- The synthetic test validates the **algorithm's correctness**; the real-model test shows it **recovers
  experimentally-validated reprogramming recipes** on the human TF network (6.2× over random), and the
  bootstrap adds a **robustness/confidence flag** per driver. For a disease application the analogous claim
  is: given a disease attractor and a healthy attractor, it proposes the driver TFs to flip (with a stability
  score) — and where ground truth exists (reprogramming), that mechanism recovers the known answer. The real
  number is no longer pending.
- Synthetic GRNs have clean master→module structure; real regulatory networks are messier, so real-world
  precision will be lower.
- Topological, not kinetic. "Reverse the state" is a wet-lab-testable hypothesis, **not** a validated cure
  — a signature/state reversal can in principle be corrective or merely cytotoxic; only driver-classified,
  bench-validated hits should be trusted.
