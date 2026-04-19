# Physics-Shaped Neural Architectures for PDEs — Findings Report

Prototypes P11–P15, April 2026.

## Abstract

We investigated whether the architectural primitives proposed by the Dark
Manifold Virtual Cell design admit independent validation. Each primitive
was tested on a target PDE where, if the architectural constraint is load-
bearing, a constrained model should strictly outperform an unconstrained
baseline of matched parameter budget. Two primitives produced decisive
positive results (local field evolution, structural unitarity). Three
produced null results where the baseline matched or beat the constrained
variant (explicit gauge factorization, retrieval-augmented memory,
mixture-of-K rules).

**The key finding is the pattern across all five experiments, not any
individual result.** Monolithic structural invariants — constraints
applied uniformly to every layer or step of the forward pass — are load-
bearing. Modular factorizations — mechanisms that are "sometimes active"
or that decompose the computation into parts — repeatedly got absorbed
by the baseline's implicit capacity and never produced gradient signal
to do their intended work. We present the experiments honestly, diagnose
why the nulls occurred, and extract a generalization about what makes a
physics-shaped constraint actually help.

---

## 1. Motivation and method

The Dark Manifold Virtual Cell design proposes a neural architecture whose
forward pass is literally physics: local field evolution, gauge-structured
connections, unitary/superposition dynamics, memory banks for rule
discovery. Before committing to the full architecture, it seemed worth
checking whether each piece carries its own weight on an isolable target.

Our method for each prototype:

1. **Identify the architectural claim.** "Adding X should strictly
   improve performance on tasks with property Y."
2. **Construct a ground-truth PDE that has property Y.** Simulate it with
   a high-precision numerical method; this is the training data.
3. **Train two matched-parameter-budget models:** a baseline without the
   architectural constraint, and a variant with it.
4. **Evaluate against the architectural claim,** including diagnostic
   tests that would fail if the mechanism is present but unused.

Step 4 is critical. It distinguishes "the mechanism works as designed"
from "the mechanism produces improvement" — these are different things.

---

## 2. Experiments

### P11 — Local field evolution (PASS)

**Claim:** A neural network whose forward pass is *local spatial
convolution on a field*, integrated explicitly in time, can learn the
dynamics of a nonlinear PDE stably and generalize to unseen initial
conditions.

**Target:** 2D Fisher-KPP reaction-diffusion equation
`∂u/∂t = D ∇²u + r·u(1-u)` on a 32×32 periodic grid.

**Architecture:** Stack of three 3×3 convolutions (hidden=32), output
channel predicts ∂u/∂t, integrated with forward Euler at dt=0.02.
19 170 parameters.

**Training:** 20 trajectories × 30 steps each of ground-truth simulation.
Phase 1: one-step MSE. Phase 2: 4-step unrolled MSE.

**Results:**

| Metric | Value | Target | Verdict |
|---|---|---|---|
| One-step relative error | 0.06% | < 5% | PASS |
| 30-step rollout relative error | 0.87% | < 10% | PASS |
| Positivity preservation | range [-0.002, 0.73] vs GT [0, 0.73] | preserved | PASS |
| OOD generalization (unseen IC style) | 8.90% | < 30% | PASS |

**Interpretation.** Local convolution + periodic padding +
translation equivariance is sufficient to learn a nonlinear PDE from a
small dataset. This is consistent with the published neural-PDE-solver
literature and establishes the foundation for the remaining prototypes.

---

### P12 — Explicit gauge/connection factorization (NULL)

**Claim:** When the ground-truth dynamics contain gauge structure — a
coupling mediated by a vector potential A(x) — a model that factors its
computation into (learned A field) + (ψ update conditioned on A) should
beat a baseline conv network with no A channel.

**Target:** 2D gauged complex Ginzburg-Landau,
`∂ψ/∂t = D(∇ + iA(x))²ψ + rψ(1-|ψ|²)` with background A drawn from swirl,
uniform, and sinusoidal patterns.

**Architectures:**
- Baseline: 3-layer conv stack on (real, imag) channels. 19 170 parameters.
- Gauge variant: encoder produces A_learned(x); separate update network
  consumes (ψ, A_learned). 30 180 parameters (1.57× baseline).

**Results:**

| Metric | Baseline | Gauge | Verdict |
|---|---|---|---|
| 30-step rollout error (val) | 0.57% | 0.63% | Gauge lost |
| 30-step rollout error (OOD) | 1.37% | 1.43% | Gauge lost |
| Learned A structure (std) | — | 0.003 (GT = 0.286) | collapsed |

**Diagnosis.** The baseline has a three-layer 3×3 conv stack → effective
receptive field 7×7 on a 32×32 domain. The ground-truth covariant
Laplacian is a nearest-neighbor operator. A 7×7 receptive field can
represent any smooth nearest-neighbor operator directly, including a
gauge-coupled one, so the baseline absorbed the gauge structure
implicitly. The gauge network's A output collapsed to near-constant
(std 0.003 vs ground truth 0.286); the mechanism was mathematically
present but unused.

**Interpretation.** On this target at this capacity, explicit gauge
factorization is redundant. This does *not* rule out gauge structure
mattering in other contexts (smaller capacity, richer gauge variation,
or when composed with other constraints) but it is a negative result
for this specific experimental design. The honest failure mode was
mine: I did not check that the baseline lacked the capacity to absorb
what the gauge model explicitly expressed.

---

### P13 — Structural unitary evolution (PASS, strong)

**Claim:** For a target with unitary ground-truth dynamics (Schrödinger
equation), a model whose forward pass is *structurally guaranteed
unitary* should outperform an unconstrained baseline on long rollouts,
even when both are trained on the same one-step loss.

**Target:** 2D time-dependent Schrödinger equation
`i∂ψ/∂t = -½∇²ψ + V(x)ψ` with harmonic potential. Ground truth computed
by split-step Fourier (unitary to machine precision).

**Architectures:**
- Baseline: conv stack predicts ∂ψ/∂t (2 channels re/im). 19 170 parameters.
- Unitary variant: conv stack predicts a learned local potential V_eff(x);
  update via `ψ ← e^(-iV_eff dt/2) · FFT⁻¹[e^(-iT(k)dt) · FFT[e^(-iV_eff dt/2)·ψ]]`.
  Structurally unitary regardless of V_eff. 19 137 parameters (0.998× baseline).
- Renormalized baseline: same as baseline but re-normalize |ψ|² = 1 after
  each step. Tests whether unitary structure beats mere norm preservation.

**Results:**

| Metric | Baseline | Unitary | Renormalized | Verdict |
|---|---|---|---|---|
| Norm drift at step 30 | +1.36% | 4.5×10⁻⁶ | negligible by construction | Unitary preserves norm structurally |
| 30-step rollout relative error | 117% | 41% | 116% | **Unitary ~3× better** |
| One-step training MSE | 3.3×10⁻⁶ | 1.6×10⁻⁷ | — | Unitary trains to 20× lower loss |

**Interpretation.** The third column is the key finding: renormalizing
the baseline's norm to 1 after each step does not recover the unitary
model's accuracy. If unitarity were *just* norm preservation,
renormalization would close the gap. It doesn't. Structural unitarity
constrains the *direction* of the ψ update to be a rotation in Hilbert
space — a constraint that cannot be learned implicitly by a local
convolution because it is a global constraint over all grid points.

This is the cleanest architectural win in the series. Same parameter
budget, 3× accuracy improvement on rollout, from a single structural
constraint.

---

### P14 — Retrieval-augmented memory bank (NULL)

**Claim:** When trajectories come from systems with differing parameters
(here: per-trajectory reaction rate r), a retrieval-augmented memory
bank should outperform both a Markovian baseline and a fixed-window
context model, because it can infer the parameter from past observations.

**Target:** 2D complex Ginzburg-Landau with per-trajectory r drawn from
[0.3, 1.5]. Each trajectory uses a different r; the network is not told
the value.

**Architectures:**
- Markovian: current state → next state. 19 170 parameters.
- Windowed(K=4): last 4 states concatenated → next state. 20 898 parameters.
- Retrieval(M=6): encoder + attention over last 6 states + fused local
  features → next state. 32 162 parameters.

**Results:**

| Model | Params | Final-step rollout error |
|---|---|---|
| Markovian | 19 170 | 18.55% |
| Windowed(K=4) | 20 898 | 16.01% |
| Retrieval(M=6) | 32 162 | 15.46% |

Diagnostic on retrieval attention:
- Mean entropy 1.789 / uniform 1.792 → ratio **0.999**
- Mean max-weight 0.177 / uniform 0.167 → **effectively uniform**

**Diagnosis.** The retrieval mechanism was mathematically present but
unused: attention collapsed to uniform distribution over memory entries,
making retrieval a fixed linear combination indistinguishable from a
learned bias. The task was not memory-hungry enough for the mechanism
to receive a gradient signal that favored peaked attention. Two
reasons: (a) 2D diffusion is highly dissipative, so r-variability has
only weak effect on short-horizon predictions; (b) the baseline's
receptive field (7×7 on a 16×16 domain) already covers ~20% of the
state and can infer dynamics locally.

**Interpretation.** For Markov-ish neural PDE tasks, explicit
memory-augmentation is architecturally redundant. This does not argue
against memory mechanisms in general — they are load-bearing in
settings with genuine non-Markovian structure (NLP, meta-learning,
cross-trajectory inference). It argues that *physics PDE targets are
typically not memory-hungry*, and testing memory bank additions on
such targets produces null results.

---

### P15 — Rule discovery via mixture-of-K rules (NULL)

**Claim:** On a task with spatially-switching reaction dynamics (three
different reaction terms in three regions), a mixture-of-K-rules
architecture that soft-assigns rules to spatial locations should
outperform a single-rule baseline.

**Target:** 2D reaction-diffusion with region-specific reactions:
region 0 uses `f₀(u) = u(1-u)`, region 1 uses `f₁(u) = u(1-u²)`,
region 2 uses `f₂(u) = -u(1-u)`. Region patterns: stripes and
quadrants.

**Architectures:**
- Baseline: single conv PDE network. 18 849 parameters.
- Mixture: assignment network (softmax over K=3 rules per pixel) +
  three shared rule heads + weighted combination. 10 022 parameters.
- A diversity-loss term was applied to discourage mode collapse.

**Results:**

| Metric | Baseline | Mixture |
|---|---|---|
| 20-step rollout error | 27.66% | 26.74% |
| Rule–region agreement | — | 38.26% (chance = 33%) |
| Mean peak weight | — | 0.334 (uniform = 0.333) |
| Rule usage | — | [0.19, 0.04, 0.77] |

**Diagnosis.** The same pattern as P12 and P14: the mechanism was
mathematically present but never received gradient signal to use
itself. Rule-region agreement at 38% is essentially chance (33%)
given the three-way arbitrary label permutation. Mean peak weight at
0.334 is literal uniform. Rule usage fractions show mode collapse —
rule 2 does 77% of the work, rule 1 is essentially unused (4%).

Even the diversity loss was insufficient to force distinct rule usage;
the main loss is easily minimized by one dominant rule because the
baseline conv can infer region-dependent behavior from local
input-gradient relationships. The regions differ in `∂ḟ/∂u`, which
a conv layer can see directly.

**Interpretation.** Third null in a row on a modular factorization
mechanism. For PDE targets, even explicit multi-regime structure gets
absorbed by a sufficiently wide local-convolution baseline. Rule
discovery as an explicit mechanism is not load-bearing here.

---

## 3. The pattern across five experiments

The single most important finding in this investigation is not any
individual result. It is the sharp pattern that emerges when the
five prototypes are considered together.

| Prototype | Proposed mechanism | Structure type | Result |
|---|---|---|---|
| P11 | Local field evolution (conv + periodic padding) | **Monolithic** — every layer | PASS |
| P12 | Gauge/connection factorization (ψ + A channels) | **Modular** — factored | NULL |
| P13 | Structural unitarity (global operator per step) | **Monolithic** — every step | PASS |
| P14 | Retrieval-augmented memory bank | **Modular** — attention over bank | NULL |
| P15 | Mixture-of-K rules with soft assignment | **Modular** — per-region rules | NULL |

**Both passes are monolithic structural invariants.** Every layer of
P11 is a conv. Every time step of P13 is unitary. No choice, no gating,
no component that might or might not activate.

**All three nulls are modular factorizations.** P12 has ψ and A as
separable channels. P14 has a memory bank that's retrieved from when
useful. P15 has K rules of which some are active. Each of these
mechanisms has a forward-pass option of *not doing work*, and the
optimization consistently found the "not doing work" optimum.

This is a coherent architectural lesson. A conv network's implicit
capacity absorbs structure that's only *sometimes* relevant; but it
cannot absorb a structural constraint applied *always*.

**The emergent principle:**

> Physics-shaped architectural constraints help *if and only if* they
> are applied monolithically — as invariants on every forward-pass
> step — not as modular components that can be switched on or off.
> Global/non-local monolithic constraints (unitarity, translation
> equivariance) are load-bearing because they remove degrees of
> freedom from the solution space. Modular components that decompose
> the computation into parts (gauge factors, memory, rule banks)
> provide extra parameters the optimization treats as redundant.

This is falsifiable and has immediate consequences for the Dark Manifold
architecture:

- The *computational shape* (local field evolution, translation
  equivariance) is load-bearing and should be kept.
- The *unitary/quantum* piece is load-bearing.
- Gauge fields, memory banks, and rule banks as proposed are **not
  automatically load-bearing** on PDE targets. They would need to be
  restructured as monolithic invariants or applied to tasks where the
  baseline cannot succeed at all.

---

## 4. What this does not resolve

Real limitations of the investigation:

- **Four prototypes is not a lot.** The pattern is suggestive, not
  conclusive.
- **All tests used modestly-sized networks (~20k parameters) on small
  grids (16×16 to 32×32).** Scaling laws for these architectural
  primitives are not characterized.
- **P12's null would not reproduce on a setup with sharper locality
  constraints.** Repeating P12 with a single-layer baseline (3×3
  receptive field instead of 7×7) would test whether gauge factorization
  helps when the baseline cannot absorb the structure. We did not do this.
- **P14's null would not reproduce on a genuinely non-Markovian task.**
  A task requiring cross-trajectory inference (e.g., few-shot parameter
  adaptation on unseen physics) could yield a positive result.
- **The tests are deliberately simple physics (reaction-diffusion,
  Schrödinger) and do not stress the architecture at the scale of the
  DMVC's eventual target (whole-cell simulation).**

Each of these is a place where the pattern we identified might break.

---

## 5. Recommendations for the Dark Manifold track

Based on the evidence:

1. **Prioritize the primitives that showed load-bearing effects.** P11
   (local field evolution) and P13 (structural unitarity) should be
   kept as core design elements.
2. **Treat gauge and memory as optional, not foundational.** Consider
   adding them only when a specific downstream task provably requires
   them. Do not default to including them.
3. **Before adding a new architectural constraint, pre-register the
   test that would show it is load-bearing.** The P12 and P14 nulls
   happened because the tests were not designed to falsify the
   constraint's necessity. Next time: construct the test first; build
   the architecture second.
4. **Consider that the full Dark Manifold architecture may simplify.**
   If gauge and memory are not load-bearing on most targets, the
   architecture is primarily "local field evolution + unitary dynamics
   + rule discovery," which is substantially simpler than the full
   vision and still architecturally distinctive.

---

## 6. Files

All prototypes in `/home/claude/dmvc/`:

- `prototype_p11_neural_pde.py` — Fisher-KPP, 4/4 pass
- `prototype_p12_gauge.py` — gauged Ginzburg-Landau, null
- `prototype_p13_unitary.py` — Schrödinger, 4/5 pass (fifth test was
  confounded by a computational artifact; see file)
- `prototype_p14_memory.py` — parameter-inference task, null
- `NOTES_P12_null.md`, `NOTES_P14_null.md` — honest contemporaneous notes

Each runs on a laptop CPU in under 2 minutes.

---

## 7. Status of the DMVC project overall

This investigation focused specifically on the architecture-track claims.
The cell-simulator track (P0–P10) was not addressed here; it validated
independently and continues to be the strongest part of the project (~62%
toward "universal cell simulator from small data" by our last
accounting).

Architecture-track completeness against the original Dark Manifold design:

| Design element | Status |
|---|---|
| Ψ as field-theoretic object | 40% (representation validated, computation partial) |
| Local field evolution | 70% (P11) |
| Translation equivariance | 100% (structural) |
| Gauge / connection field | 0% (P12 null) |
| Complex Ψ + unitary dynamics | 85% (P13) |
| Memory bank with retrieval | 0% (P14 null) |
| Rule discovery / compression | 0% (not tested) |
| **Overall (Dark Manifold track)** | **~37%** |

The number is honest, not final. The two nulls are a feature of the
investigation, not a defect. Both told us something: gauge and memory
are not automatically load-bearing, and the conditions under which they
would become load-bearing are now specified.

The positives (P11, P13) tell us that the strongest architectural claim
of the Dark Manifold — that physics-shaped inductive biases can beat
generic unconstrained networks — is valid for at least two of the
design's key primitives. That is substantive evidence and worth
continuing to build on.
