# `rem.mechanism` and `rem.reach` — claim status

Date: 2026-08-31
Both modules were previously prototyped in a scratch sandbox on a single model each and were
**not in the repo**. This note records, for each claim, whether the implementation supports it,
supports a narrower version, or retracts it.

## Claim: "Mechanism — the committor and the reactive pathway, showing that a switch flips by
## depleting one gene to near zero rather than passing through a balance point."

**SUPPORTED WITH A NARROWER STATEMENT.** `rem/mechanism.py`, `tests/test_mechanism.py`.

Machinery verified first:

| gate | result |
|---|---|
| V1 committor vs gambler's-ruin closed form | max err **9.4e-16**, harmonic residual 8.9e-16, boundary exact |
| V1a test region spans 0.1 < q < 0.9 | 18 states inside, range 0.014–0.913 |
| V2 independent Monte Carlo (jump chain) | worst **2.11 s.e.** against a 3 s.e. bar, span OK |
| V3 symmetry q(x,y)+q(y,x)=1 | **1.8e-15**; diagonal pinned at 0.5 to 7.8e-16 |
| V4 flux conservation (1D, reversible) | A–B imbalance **7.8e-14**, divergence 3.7e-13 |

**V1a earned its place on the first run.** The initial rate choice (death = 0.30x) put a stable
point mid-region and drove the committor to 0.66–0.997 across the whole transition band. V2
"passed" at worst 1.17 s.e. while testing nothing, and V1a failed it correctly. Rates were
changed to a mild constant drift so q sweeps 0.08–0.83 over the tested states.

What survives of the claim, swept over the K the prototype never specified (K = 4 … 20):

* **Holds at every K**: the committor on x = y is far from 0.5, and the tipping point lies
  below the diagonal. So **x − y is not the reaction coordinate** and flips are asymmetric.
* **Holds at K ≈ 4 only**: the dominant reactive flux running along a wall. At K = 4 the top
  edge is `(13,0) → (12,0)`, matching the prototype's reported edge — which pins its unstated K
  at about 4 — and by K = 6 the dominant path runs at 2–7 copies instead.
* **The quoted numbers do not reproduce at any K tried**: 0.736 / 0.707 / 0.696 on the diagonal
  against 0.93–1.00 here; y*(12) = 7 against 0–2.

A boundary artefact was suspected and **tested rather than assumed**: with A = {a≥hi, b≤lo},
any lo ≥ 1 absorbs the wall into A where it cannot carry flux. Measured at lo = 0,1,2,3,5 the
top edge sits at min-coordinate 3,4,5,3,5 — so the wall is free at lo = 0 and still carries no
dominant flux. The artefact hypothesis was **wrong**, and the retraction is not a boundary
effect.

**Quotable form:** *an asymmetric toggle does not flip through a balance point; the committor is
asymmetric about x = y and the tipping point lies below the diagonal, robustly across K.*
Not quotable: "it runs down one wall to near zero", which is a K ≈ 4 statement.

## Claim: "Reach — a mutation's influence dies out within one or two genes, unless a global
## regulator is involved, and then it's a uniform background shift."

**SUPPORTED**, with one verification route void. `rem/reach.py`, `tests/test_reach.py`.

| gate | result |
|---|---|
| R1 linearity | **ε = 0.10 FAILS** (0.0225 > 0.01). Largest linear ε = **0.02** (0.0046) |
| R4 direct re-solve vs finite difference | gap 0.014 at ε = 0.02; all numbers are re-solves |
| R3 local topologies | chain/ring **2/12** move, radius **1**, geometric decay |
| R3 hub | **12/12** move, far-field spread/mean **0.000** |
| R3 chain+global | **12/12** move, far-field spread/mean **0.012** (bar 0.35) |
| R2 independent decay route | **VOID** — see below |

**R1 is a precondition, not a report.** The first version failed linearity at ε = 0.10 and then
printed a decay table anyway — ledger defect C, a downstream measurement blind to a failed
precondition. The perturbation size is now *chosen* by R1, and if nothing passes, no decay
number is produced at all.

The prototype's qualitative claims reproduce: local wiring gives radius 1 with a clean
geometric decay, and a global regulator moves everything. **The load-bearing half — that the
far field is a near-uniform offset rather than gene-specific — reproduces strongly**
(spread/mean 0.000 for a pure hub, 0.012 for chain+global, against a 0.35 bar). That is what
licenses a single aggregate standing in for the far field.

**R2 is VOID for a physical reason, not a numerical one.** The spec asked for the
transfer-matrix correlation length ξ = 1/ln(λ₀/λ₁). A CME stationary state is a null vector,
not a transfer-matrix product — established earlier in this project when chain elimination
failed on driven 1D transport — so that route does not exist. The substitute (equilibrium
connected-correlation length) equals the response length only under fluctuation–dissipation,
which holds **only in equilibrium**; this chain is driven. Measured ξ_pert = 1.46 against
ξ_corr = 0.65, and the disagreement is evidence against neither. **The decay length therefore
rests on one route, not two, and is weaker for it exactly as the spec anticipated.**

## What neither module does

Neither touches the mutation → rate map, which remains missing-item #1 and is the docking
problem. `rem.reach` answers "if this rate moves, who else moves" — not "what does this
mutation do to this rate".
