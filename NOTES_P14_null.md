# P14 — Null result notes

Date: 2026-04-19
Status: null result, filed

## Claim tested
Adding a retrieval-augmented memory bank (encoder → attention over past
states → retrieved context fused with local features) to a neural PDE
should improve prediction accuracy on a task where memory of past states
carries information not present in the current state.

## Setup
* Task: 2D complex Ginzburg-Landau with a per-trajectory reaction rate r
  drawn from [0.3, 1.5]. Different trajectories have different r.
* Model A Markovian: current state → next state. No memory.
* Model B Windowed(K=4): last K states concatenated as channels → next state.
* Model C Retrieval(M=6): encoder + attention over last M states; retrieved
  context fused with local conv features, output → next state.

## Result
Final-step rollout error, averaged across validation trajectories:

| Model      | Params | Rollout error |
|------------|--------|---------------|
| Markovian  | 19,170 | 18.55%        |
| Windowed   | 20,898 | 16.01%        |
| Retrieval  | 32,162 | 15.46%        |

All three models are within 20% of each other. Windowed improves over
Markov by 14% relative, just under the 15% threshold for T2. Retrieval
beats Windowed by only 3%, likely attributable to the 50% extra parameter
budget rather than the retrieval mechanism per se.

## Diagnosis
The retrieval attention **collapsed to uniform**:
- Mean entropy: 1.789 / uniform 1.792  → ratio 0.999
- Mean max weight: 0.177 / uniform 0.167 → peakiness 0.18

The mechanism is mathematically present in the forward pass but never
got a gradient signal that favored peaked attention over uniform. That's
the signature of a no-op.

Why:
1. The task wasn't memory-hungry enough. 2D Ginzburg-Landau is diffusive;
   r-variability has only subtle effect on short-horizon predictions.
2. The Markovian baseline's 7×7 receptive field on a 16×16 grid covers
   ~20% of the domain, which is enough to infer much of the relevant
   dynamics directly.
3. When all memory entries are roughly equally informative, uniform
   attention is optimal. The mechanism worked; there was nothing to
   select.

## What this does and doesn't mean
What it means:
* For typical Markov-ish neural PDE tasks, explicit memory-augmentation
  is architecturally redundant. The network's receptive field already
  covers the relevant history via the locality of dynamics.
* A retrieval-augmented memory bank is not automatically load-bearing.
  It becomes load-bearing only on tasks where memory is genuinely
  required — non-Markovian dynamics, few-shot adaptation, or information
  that cannot be inferred locally.

What it does NOT mean:
* "Memory banks are useless." They're load-bearing in NLP, in meta-learning,
  and in any setting with genuine non-local information requirements.
* "The Dark Manifold memory concept is wrong." If the memory is tied to
  later pieces (rule discovery, cross-trajectory inference, zero-shot
  organism transfer), it could become essential then.

## Pattern accumulating across P11–P14

| Prototype | Architectural constraint     | Result | Generalization         |
|-----------|------------------------------|--------|------------------------|
| P11       | Local field evolution (conv) | PASS   | Foundation; always load-bearing. |
| P12       | Gauge connection (factored)  | NULL   | Redundant when receptive field already covers it. |
| P13       | Unitary evolution (global)   | PASS   | Load-bearing when task has global constraint. |
| P14       | Memory bank (non-local)      | NULL   | Redundant when task is Markovian. |

**The emergent rule:** architectural constraints help *if and only if*
they encode something the baseline cannot easily learn implicitly.
Global/non-local constraints (unitarity, conservation laws) are
load-bearing. Local factorizations (gauge, memory on Markov tasks) are
redundant.

## Action taken
Accept as null result. P14 did not invalidate the memory-bank concept;
it showed that on physics PDE tasks the mechanism sits idle. A genuine
test of memory-augmentation for the Dark Manifold would require a task
with non-Markovian structure or cross-trajectory information requirements.

## Implementation lesson
Same as P12 but sharper: **the task must genuinely require the
architectural mechanism, or the mechanism will be silently unused.**
Next time, before building, construct a test that would provably fail
without the mechanism. If baseline can solve the task, variant cannot
be shown to be necessary.
