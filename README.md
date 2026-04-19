# Dark Manifold Virtual Cell (DMVC) — Prototype Series

Architecture prototypes for a physics-shaped neural network for minimal-cell
simulation. Target organism: JCVI-Syn3A.

## Why this exists

Standard biochemical simulation is expensive (ODE with hundreds of species)
or inaccurate (coarse stoichiometric models). The DMVC approach is a neural
network whose **inductive biases reflect physics**: field-theoretic
representation, locality, gauge-like couplings, symmetry-broken phases,
geometry-responds-to-energy. The bet: a network shaped like physics learns
biology from orders of magnitude less data than a generic transformer.

These prototypes validate the physics-shaped architecture *before* we start
training anything. Each prototype isolates one invariant and proves it with
a structural test — not statistically, but to machine precision.

## What's here

| File | What it proves | Status |
|---|---|---|
| `prototype_p0.py` | atom conservation via subspace projection | 15 orders of magnitude enforcement |
| `prototype_p1.py` | stoichiometric reactions, stamp+flavor embeddings | 5/5 tests pass |
| `prototype_p2_syn3a.py` | load real Syn3A SBML (304 mets, 244 rxns) | 83/244 balanced as-written |
| `prototype_p2b_rebalance.py` | H⁺/H₂O rebalance closes implicit balance | 242/242 balanced |
| `prototype_p3_compartments.py` | compartment-aware v1 (bug exposed) | revealed transport invisibility |
| `prototype_p3b_stamps.py` | compartment-aware stamps fix it | 4/4 proper passes |
| `prototype_p4_kinetics.py` | load + run Luthey-Schulten kinetics (well-mixed) | central metabolism stable |
| `prototype_p4b_kinetics_coupled.py` | kinetics on P3b spatial architecture | 3/4 pass, stiff-reaction cap documented |
| `prototype_p5_boundary.py` | extracellular medium buffering, open system | 3/3 pass |
| `prototype_p6_physiological.py` | cytoplasmic cofactor homeostasis + real glucose | 3/4 pass, ATP converges slowly |

Total: ~4600 lines of code across 10 prototypes.

## What's validated (so far)

- Representation: Ψ field with compartment-aware stamp/flavor embeddings
- Conservation: atoms + charge, structural, machine-precision, survives real
  Syn3A biochemistry (242 reactions) and real kinetic rate laws
- Compartments: cytoplasm / membrane / extracellular with proper separation
- Transport: moves mass across boundaries with global conservation intact
- Real kinetics: Michaelis-Menten / convenience rate laws from the
  Luthey-Schulten parameter database drive Ψ evolution
- Open-system dynamics: medium buffering → indefinite simulation
- Physiological tuning: NAD/NADH in correct regime, central carbon stable

## What's not yet done

- Learned Lagrangian (no neural network has been trained yet — we built the
  substrate that one will sit on)
- Stiff integrator (using a rate-cap workaround for Syn3A's fast reactions;
  production would use scipy LSODA or CVODE)
- Polymerization reactions (11 excluded; need specialized rate laws)
- Diffusion (spatial transport within compartments)
- Multi-organism generalization
- Atomic-resolution query mechanism

## How to run

```bash
pip install -r requirements.txt

# Clone the Luthey-Schulten Minimal_Cell repo next to prototypes (used for SBML + kinetics)
git clone --depth 1 https://github.com/Luthey-Schulten-Lab/Minimal_Cell.git data/Minimal_Cell

# Run any prototype
python prototype_p0.py
python prototype_p3b_stamps.py
python prototype_p6_physiological.py
```

Note: `SBML_PATH` is hard-coded near the top of `prototype_p2_syn3a.py` at
`/home/claude/dmvc/data/Minimal_Cell/CME_ODE/model_data/iMB155_NoH2O.xml`.
Update this to match your local path.

## Project lineage

This DMVC prototype series descends from the V1–V24 cell-simulation work at
`Nikku03/enzyme_Software`. Those earlier versions explored neural architectures
(Mamba, Perceiver IO, neural ODEs, graph reasoning) on top of generic cell
models. The prototypes in this repo step back from model complexity to
**validate the physical scaffolding** — making sure atoms conserve, reactions
run correctly, and the representation supports what the network will eventually
learn.

## Honest caveats

Everything in this repo is a **validation prototype**, not a production
simulator. The rate cap, hand-tuned buffering, excluded polymerization
reactions, and fixed enzyme concentrations are all workarounds chosen so
we could test one architectural invariant at a time. A production system
replaces each of these with the real physics or the real learned function.

See individual prototype docstrings for the specific honest limitations of
that piece.
