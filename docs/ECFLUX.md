# Enzyme-constrained flux — the quantitative rung

The model could already answer *which* enzymes matter and *which direction* things move. This layer adds
**how much**: it turns "losing enzyme X reduces flux" into "…reduces flux/growth by N%". It's the coupling
that was missing between kinetics/stability and metabolic phenotype — the quantitative rung of the
mutation→phenotype chain.

Built on the curated **Human-GEM** (12,931 reactions, grows under FBA). `colab/ecflux.py`, gated by
`colab/validate_ecflux.py` as the 13th recovery-scorecard axis.

## What it does

1. **`flux_control_curve`** — give an enzyme a finite capacity `C = flux_WT / σ` (σ = saturation, so the WT
   enzyme carries `1/σ` excess), then dial its activity 100%→0% and re-solve for biomass. The response curve
   *is* the quantitative enzyme→flux map.
2. **`essentiality`** — FBA single-gene deletion: which enzymes the cell cannot lose.
3. **`mutation_to_flux`** — wires in the ΔΔG node: a destabilizing ΔΔG shrinks the folded/active fraction via
   the two-state folding equilibrium `f = 1/(1+e^(−ΔG_unfold/RT))`, which sets the capacity multiplier, which
   moves the enzyme along its flux-control curve. So: **mutation → ΔΔG → active fraction → Δflux**.

## Validation

**A) Essentiality recovery (non-circular).** FBA gene deletion vs our measured `ess` labels (n≈2,570):

| metric | value |
|---|---|
| precision | **0.82** |
| precision-lift over base rate | **8.3×** |
| recall | 0.27 |

High precision, modest recall — the textbook metabolic-model profile: when FBA calls a gene essential it's
almost always right, but it misses essentials that a rich medium rescues or that act non-metabolically.

**B) Metabolic dominance (the quantitative claim).** Flux-control curve averaged over 150 flux-carrying
enzymes:

| enzyme activity | biomass retained |
|---|---|
| 100% → 50% | **100%** (fully buffered) |
| 35% | ~92% |
| 15% | ~82% |
| 5% | ~77% |
| 0% (full KO) | ~74% |

Two results here: partial loss is **buffered** (the σ excess-capacity assumption), and — the *emergent*,
assumption-free part — even at **complete knockout** only ~**1%** of single enzymes collapse growth; the
network reroutes around the rest. This reproduces **Kacser–Burns metabolic control theory** and the molecular
basis of **recessivity**: a heterozygote with ~50% enzyme is silent, disease appears only near-complete loss.

**C) ΔΔG→flux demo** (mutation of a flux-sensitive enzyme, via `mutation_to_flux`):

| mutation ΔΔG (kcal/mol) | active fraction | biomass retained |
|---|---|---|
| 0–6 | ≈1.0 | **100%** (silent / benign) |
| 8 | 0.16 | **32%** |
| 10 | 0.01 | **1%** (collapse / pathogenic) |

The threshold behavior is the point: mild-to-moderate destabilizers are absorbed (why most missense variants
are benign), and only a *severe* destabilizer (ΔΔG ≳ 8) crosses into flux collapse — the carrier-vs-affected
boundary, emergent from folding thermodynamics + the flux-control curve.

**D) Measured per-enzyme capacity (uses data we already have).** The σ above is a *blanket* assumption — every
enzyme given the same 2× excess. But we **measure** proteome abundance (`ppm`, 16,015 genes) and estimate kcat
(2,549 enzymes), so `Vmax = kcat·[E]` gives the capacity per enzyme from data. `capacities_from_ppm` uses this
for the *distribution* — which enzymes are buffered vs dose-sensitive — while keeping the single global scale
anchored to σ (the median enzyme still carries 1/σ excess). On central-carbon metabolism:

| metric | value |
|---|---|
| central-carbon flux-carrying enzymes | 120 |
| with measured ppm + kcat | **108 (90%)** |
| measured Vmax spread (log₁₀ std) | **2.6 → ≈400× range** |

The measured Vmax spans ~400×, so the buffered-vs-dose-sensitive split is now **data-backed, not assumed
uniform**. `flux_control_curve(..., capacities=caps)` and `mutation_to_flux(..., capacity=c)` take these
measured caps directly; without them they fall back to the blanket σ. Gated as the 14th scorecard axis
(`ecflux_measured_capacity`, `colab/validate_ecflux_ppm.py`). This is the "use the measured data we were
ignoring" fix — it removes the uniform-σ assumption for exactly the enzymes where we have real abundance.

## Honest limits

- **Absolute %s still carry a scale assumption.** The measured-capacity path (D) makes the *relative*
  per-enzyme headroom data-backed, but the single global scale is still anchored to σ (measured `ppm` is a
  proteome fraction, not copies/cell in flux units). The *shape* of the dominance law, the essentiality
  recovery, and now the *heterogeneity* of capacity are the solid parts; an absolute per-enzyme % in a
  specific condition still needs matched absolute proteomics + condition-specific flux.
- **Simplified ecModel** — per-reaction capacity caps, not the full shared-proteome-pool GECKO (no global
  protein budget). Enough for the dominance/knockdown response; a proteome-pool constraint is the next refinement.
- **The ΔΔG→flux link is a demonstrated mechanism, not yet an end-to-end validated number** — validating it
  quantitatively needs measured mutation→flux data. The honest end-to-end test is running it on metabolic IEM
  mutations where the biomarker end is already validated.

## What it now answers

- "Knock out enzyme X — can the cell still grow / make Y?" → yes/no, validated.
- "Halve enzyme X's activity — how much does flux/growth change?" → a number (buffered vs sensitive).
- "Is this destabilizing mutation likely silent or pathogenic?" → where it lands on the flux-control curve.
