# Dynamic regulatory layer: TF occupancy → production rate → concentration (Gillespie)

The work-around for the TF→site wall: don't predict the binding site. Take the
EDGE (measured RegulonDB / inferred from co-expression), put a thermodynamic
occupancy→rate law on it, and simulate concentrations stochastically (Gillespie
SSA) under changing scenarios. This is the conditional/dynamic engine the cell
model needs.

## What's real vs illustrative (honest)
- **Topology + sign** (which TF→which gene, activator/repressor): **measured**
  (RegulonDB). We do not predict the site.
- **Occupancy → transcription rate**: thermodynamic/Hill function of the
  regulator's **active concentration** — concentration is the deciding variable.
- **Scenario knob**: each TF's active fraction φ(t) (effector present → repressor
  off, etc.).
- **Rate/conc parameters**: E. coli literature-typical defaults (mRNA t½≈5 min,
  protein dilution ≈40 min, Hill n=2). Dynamics are **qualitatively** correct;
  absolute molecule counts and exact fold-ranges are illustrative, not predicted.

## Demo A — SOS response on the real LexA regulon
7 genes (recA, sulA, uvrA, dinB, ruvA, umuD, lexA), all LexA repressor targets.
DNA damage (t=120) inactivates LexA → derepression; repair (t=260) restores it.

| gene | basal | induced | fold |
|---|---|---|---|
| dinB | 516 | 3728 | 7.2× |
| umuD | 557 | 2957 | 5.3× |
| recA | 704 | 3296 | 4.7× |
| sulA | 680 | 3086 | 4.5× |
| lexA | 730 | 2994 | 4.1× |
| uvrA | 969 | 3700 | 3.8× |
| ruvA | 864 | 3234 | 3.7× |

Clean conditional ON→OFF switch with intrinsic noise: genes sit repressed, jump
on damage, hold while LexA is inactive, decay back after repair. (Real SOS fold
ranges are wider; tightening leak/K reproduces that — it's a parameter, not a
mechanism, issue.)

## Demo B — coherent feed-forward loop (AND): response time
X→Y, X→Z, Y→Z with AND logic at Z. On the X signal, Z rises only after Y
accumulates → the sign-sensitive **delay** (Alon). Demonstrates the "response
time" behavior that network motifs impose — the engine reproduces it.

## Why this is the right work-around for the cell model
- It needs only the **edge** (which we can get: measured or co-expression-inferred,
  AUC 0.63) — never the unpredictable binding site.
- It produces exactly what conditional essentiality needs: **is a gene ON/OFF and
  at what concentration in this scenario, over time**, stochastically.
- It is the Gillespie/ODE foundation: plug in any TF subnetwork + scenario knobs,
  get concentration trajectories. Swap φ(t) for effector dynamics to chain
  metabolite → TF activity → expression → phenotype.

Files: colab/grn_gillespie.py, outputs/orphan/grn_gillespie.{png,json}.
