# ADRN package — what this is, and what it is not

Packaged on request, to be evaluated for applied science. **Read the scope section before benchmarking it**, because
the stated target — "whole protein interaction level physics, fast and efficient, maybe billions of atoms" — is not
what this code does, and evaluating it for that would waste your time.

---

## SCOPE: this is not a physics engine

**ADRN is a statistical predictor of transcriptional response to gene knockout in one cell line (K562).**

The chain is: gene → annotation/measurement feature vector → ridge regression → NMF programme mixture → ranked
gene list. There are no atoms, no force field, no integrator, no coordinates. Scaling it to "billions of atoms" is
not a matter of optimisation; the object has no atomistic representation to scale.

For calibration on what "efficient" means in this codebase, the one component that *is* physics — `physics/extrude.py`,
a coarse-grained Langevin polymer with 64 replicas × 3000 steps over ~200–1200 beads (**beads, not atoms**, at
2 kb/bead) — measured at:

| platform | per pair |
|---|---:|
| 4 CPU cores | 41 s |
| A100 / RTX PRO 6000 | 278 ms |

That is ~150× and it is coarse-grained polymer dynamics, roughly 10^2–10^3 degrees of freedom. Billion-atom MD is
6–7 orders of magnitude beyond it and belongs to GROMACS / OpenMM / NAMD / Anton, not here.

**Also relevant before you invest in it: `physics/extrude.py`'s scientific gate FAILED.** At 2,984 pairs on an A100
the degree-matched shuffled control scored 0.6806 against the real simulation's 0.6779 (net −0.0027 ± 0.0052), and
⟨d⁻³⟩ matched ⟨d⟩ to four decimals (+0.0000416) — meaning the shape sensitivity that justified the simulation does
not exist at this scale. See `results/extrude_gate_a100_3000.json`. Six independent contact attempts all landed at
or below zero.

### What in here is closest to protein-interaction physics

`physics/flex_physics.py` (ΔΔG of binding: rotamer sampling, clash relief, electrostatics/induction),
`physics/nexus.py`, `physics/molecular_engine.py`. These operate at single-complex scale — tens to thousands of
residues, one complex at a time. They are node-level scorers, not simulation engines, and none was built for
throughput.

---

## What the ADRN code actually achieves (measured, sealed holdout)

Two 200-knockout cohorts, predictions committed before answers were opened, `NPRED = 20`.

| | cohort 1 | cohort 2 |
|---|---:|---:|
| chan2a (annotation channels) | 0.2327 | 0.2893 |
| + DepMap co-dependency | 0.2540 | 0.3127 |
| frequency baseline | 0.1615 | 0.1990 |
| permuted control | 0.1210 | — |
| twin ceiling (the cap that binds) | 0.5360 | 0.6210 |
| same-knockout re-measurement | 0.6330 | 0.7100 |

So: a 20-gene list with roughly 4.7–6.3 correct, against ~3.2–4.0 free from the frequency prior, at ~45% of the
achievable ceiling. Mechanism decoding reaches 68–73% of its split-half ceiling.

**Measured negatives, so you do not re-run them:** drug-response transfer fails (ties its own shuffled control);
cross-cell-type pooling costs −0.037; end-to-end learning loses by 0.08–0.13; nine encoders sit within 0.008 of
each other; ESM-2 sequence embeddings add nothing over annotation; data scaling is +0.024/doubling and bending.

---

## Layout

    code/       38 files. adrn_*.py are the experiments; adrn_ko_conjunctions.py and adrn_ko_channels2.py are
                the shared substrate (build_channels, ridge_fit/apply, the 696 annotation channels).
                norman_*.py cover combinatorial pairs; esm_gene_embed.py builds sequence embeddings.
    harness/    robustness.py — the sweeper described below.
    physics/    the coarse-grained and single-complex physics nodes, for the scope discussion above.
    results/    40 JSON artifacts. Every headline number in this README traces to one of them.

## The part most likely to be reusable

`harness/robustness.py` is domain-agnostic and is arguably the most transferable thing here. It exists because a
PPI block was reported at +0.0140/+0.0152 after surviving three separate attacks on its control (degree preserved
exactly, 98.2% of edges rewired, seeds stable, leakage ratio 1.05×) — and then turned out to live in one of six
(edge-set × rank) cells: the configuration that happened to be run first. `SVD_K = 128` was an undeliberated
default carrying the entire finding.

Control checks ask *"is the null valid?"*. None of them asks *"is this robust to defaults I chose without
thinking?"* `Sweeper` enforces the second structurally: it refuses an axis with fewer than two values, refuses a
config outside its declared grid, and raises on a verdict from a single cell. The ladder — ROBUST / DIRECTIONAL /
FRAGILE / NULL / HARMFUL — is fixed in the module before any numbers arrive.

## Reproducing

Scripts expect to run from the repo root (`outputs/orphan` is a relative path) and read
`nlz_K562_gwps.npz`, `CRISPRGeneEffect.csv` and `cell_complete.json`, which are data files not included here for
size. `results/` is therefore the auditable record; the code is included so the method can be inspected and reused.

## Known caveat in the harness

`Sweeper` counts `config × cohort` cells as independent observations, but two cohorts within a config share a
fitted model and are correlated. Resolved-cell counts are therefore somewhat optimistic; sign-consistency across
cells is the more trustworthy signal.
