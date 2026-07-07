# Self-consistency / anomaly engine — the model checking itself

A pile of predictions and measurements is a lookup table. What makes it a *model* is **self-consistency**: the
whole constrains the parts, so a wrong part sticks out. This engine takes all layers together and finds the bits
that don't fit — "something's odd here" — as **ranked, confidence-tagged flags**, under one strict hierarchy
that avoids the trap of a mediocre model rewriting real biology.

`colab/self_consistency.py` (`SelfConsistency.scan()`), gated by `colab/validate_self_consistency.py`.

## The strict hierarchy (the anti-trap rule)

```
hard constraints  >  measured facts  >  predictions
```

- Every finding is a **flag for review**, ranked by tier then confidence. **Nothing is auto-applied.**
- A prediction **never** overwrites a measurement — it only flags it.
- The **only** thing allowed to challenge a measurement is a hard-constraint (physics) violation, and even then
  it's flagged for a human/experiment, not silently changed.

## The four detectors (most to least trustworthy)

| tier | detects | example found |
|---|---|---|
| **T1 hard_constraints** | provable oddness — physics/thermo violated | **17 enzymes with kcat/Km past the diffusion limit** (GPI, ACHE have kcat = exactly 1e6 — a placeholder, not a measurement) |
| **T2 cross_layer** | same-axis views of one entity conflict (model compartment vs curated UniProt location) | **0 hard conflicts** — the model is 98.7% consistent with UniProt on localization (a *good* result); 58 soft disagreements flagged low-confidence |
| **T3 learned** | link model on the KNOWN graph: known edge that shouldn't exist (candidate DB error) or missing edge that should (**completion**) | completion **CPSF6—SNRPA** (169 shared partners, both splicing) |
| **T4 pathway_gap** | metabolic dead-ends — "a step is missing here" | metabolites produced-but-never-consumed → a missing reaction/transporter to check |

## Validation — the completion certificate

The riskiest claim is T3 completion ("this edge is missing"), so it is validated **leakage-free**: hold out 20%
of known PPI edges, rebuild partner sets without them, and check the held-out real edges separate from
**degree-matched** non-edges.

- **Triadic-closure (shared-partner) completion AUC = 0.774** — real missing edges are recovered well above chance.
- Honest recall limit: only **46%** of held-out real edges have the ≥3-shared-partner evidence the engine
  requires, so it proposes completions where triadic evidence exists and stays silent otherwise.
- We tested combining the learned embedding with triadic closure — it only **diluted** the signal (0.77 → 0.68),
  so the embedding was dropped. The engine ranks completions by shared-partner triadic closure alone.

## The iteration that got here (build → test → improve → retest)

Testing exposed two real failure modes, both fixed:
1. **Tier 2 flagged multi-localized proteins as odd** (a secreted protein with a minor nuclear isoform read as
   "nuclear vs secretory"). Fixed by comparing the full location *set* and flagging only mutually-exclusive
   groups → false conflicts went to zero.
2. **Tier 3 completions were paralog artifacts** (olfactory-receptor pairs that look similar but don't interact).
   Fixed by requiring triadic-closure evidence (shared partners) and dropping same-family pairs → the
   embedding-similarity artifacts vanished and validation rose to AUC 0.77.

## What it delivers for the goal

The cell can now say *"this part doesn't fit the rest — check it"* and even *"the pathway is missing a step
here"* — as a hypothesis with a confidence and a provenance, adjudicated by facts, never a self-certain oracle.
It lets the model improve **itself** instead of only answering.
