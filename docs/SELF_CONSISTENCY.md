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

## Fill-and-verify — propose a fix, re-run the mechanism, keep only what resolves the oddness

Flagging isn't enough — the engine now proposes a **fix**, applies it, **re-runs the mechanism**, and reports
only the fixes that verifiably resolve the problem, under the same anti-trap hierarchy. `fill_and_verify()`,
gated by `colab/validate_fill_verify.py` (scorecard axis `fill_verify`).

| flag | proposed fix | verification (re-run the mechanism) | anti-trap |
|---|---|---|---|
| **bad kcat** (past diffusion limit) | kcat bounded to the diffusion limit | check kcat/Km is now legal (+ enzyme still carries its flux) | **17/17 verified**; the 6 that touch a **measured** kcat (e.g. GPI) are **ESCALATED for human sign-off**, never auto-applied |
| **missing edge** | add as a predicted PPI | require shared **complex** membership or ≥20 shared partners | **25/25 verified**; never touches measured data |
| **pathway gap** | add reaction(s), **iteratively** | re-run FBA each round: does the dead-end reaction now carry flux? | **retries with failure data** — a single sink fails on a multi-reaction gap, so the next round opens the metabolite that *was still blocking* and retries; **3/3 tested gaps resolve on round 2** |

### Retry with what-failed knowledge (and a check for false self-rejection)

A first fix failing doesn't end it. If the single-sink fix leaves the reaction blocked, the loop reads the
**failure data** — *which* metabolite is still a dead-end — opens that too, and retries, up to a depth cap. So a
2-reaction gap that the naive fix can't touch resolves on the second round:

```
(2R)-pristanoyl-CoA:  round 1 open 1 reaction → flux 0 (fails)
                      round 2 open 2 reactions → flux 1000 ✓ resolved
```

It never repeats a dead attempt, and if it can't resolve within the cap it says so honestly ("part of a large
disconnected module — not a point fix").

**Did the verifier ever reject a *correct* fix (a simulation bug, not a real gap)?** Checked directly: the
round-1 rejections were **genuine** 2-reaction gaps — tracing the blocked reaction showed a second dead-end
substrate, and only opening it too restores flux. So the single-sink rejection was right, and the retry finds the
real minimal fix. No false rejections (kcat and edges had zero rejections to begin with).

A verified fix is "apply-pending-review"; a fix that corrects a **measured** value is "ESCALATE"; an unverifiable
fix stays a flag. Nothing is ever silently changed, and nothing is falsely reported as fixed *or* falsely rejected.

## In the system

Exposed as **`CellQA.audit()`** — the model checking itself is now a first-class capability alongside the
question-answering ones: it returns the ranked flags + the verified fixes, each with its provenance and its
place in the hierarchy.

## What it delivers for the goal

The cell can now say *"this part doesn't fit the rest — here's the fix, and I re-ran the mechanism to confirm it
works"* — or, honestly, *"I tried a fix and it didn't resolve it, so this stays flagged."* A hypothesis with a
confidence, a provenance, and a **verification**, adjudicated by facts, never a self-certain oracle. It lets the
model improve **itself** instead of only answering.
