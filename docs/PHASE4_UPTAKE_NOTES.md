# Phase 4 — Nutrient uptake patch

## The diagnosis

The 100%-scale whole-cell run showed catastrophic ATP→AMP conversion
(ATP 73,738 → 39,738, AMP 2,019 → 37,705) and PPi depletion over 1 s
of biological time. Root cause: the Luthey-Schulten kinetic database
has measured k_cats for only 58 of the 189 transport reactions in the
Syn3A SBML network.

Critical missing transporters:
- `GLCpts` — PTS-dependent glucose import (the cell's primary carbon source)
- `GLYCt` — glycerol uptake
- `FAt` — fatty acid uptake
- `O2t` — oxygen diffusion
- `CHOLt` / `TAGt` — lipid precursors

No corresponding SBML reactions for free nucleobase uptake at all
(adenine, guanine, uracil, cytidine).

Result: the cell was sitting in nutrient-rich medium but had no mouth
for glucose, and no way to replenish adenine for the purine salvage
loop. Internal G6P was being slowly depleted while internal adenine
pools were unable to close the ADPT → AMP flux.

## The patch

`layer3_reactions/nutrient_uptake.py` builds 15 new transport rules:

| rule | k_cat /s | rationale |
|---|---|---|
| GLCpts | 500 | PTS glucose import; Postma 1993 ~500/s/complex |
| GLYCt (fwd+rev) | 200 | facilitated glycerol diffusion |
| O2t (fwd+rev) | 500 | passive O2 diffusion |
| FAt (fwd+rev) | 50 | fatty acid flippase |
| CHOLt (fwd+rev) | 50 | cholesterol uptake |
| TAGt (fwd+rev) | 50 | triacylglycerol uptake |
| ADEt_syn | 20 | synthetic adenine uptake (not in SBML) |
| GUAt_syn | 20 | synthetic guanine uptake |
| URAt_syn | 20 | synthetic uracil uptake |
| CYTDt_syn | 10 | synthetic cytidine uptake |

Each rule is mapped to a real Syn3A gene locus where possible
(PtsG = JCVISYN3A_0779 for GLCpts; FakB = JCVISYN3A_0616 for FAt) and
a placeholder membrane-protein locus (JCVISYN3A_0034, the highest-
abundance uncharacterized efflux ABC transporter, 1795 copies) for
reactions with no annotated carrier.

**All k_cat values are literature-informed order-of-magnitude
estimates, not measurements.** They should be replaced with real
values as they become available.

## What the patch fixes (verified at 10% scale)

Side-by-side 300 ms runs, scale=0.1, seed=42:

| species | baseline Δ | patched Δ | effect |
|---|---|---|---|
| adenine | -68 | **+226** | salvage closed |
| guanine | -146 | **+179** | salvage closed |
| uracil | +96 | **+363** | salvage closed |
| glycerol | -19,217 | -18,993 | partial rescue |
| ATP/ADP/AMP | similar | similar | small change at this scale |
| PPi | -3,295 | -3,196 | unchanged |

At 10% scale the ATP catastrophe didn't occur in either run — it's
scale-dependent. **The decisive test is at 100% scale**, which can only
be done on Colab (30+ min wall time).

## What the 100% Colab run will test

The prediction: with the patch, at 100% scale over 1 s bio time,

1. **ATP should stabilize** (not drop from 73k to 40k)
2. **AMP should not accumulate** (not rise from 2k to 38k)
3. **Internal glucose pools** should be replenished by GLCpts firing
4. **Nucleobase pools** should stay within 2× of initial

If the prediction holds: the decay was a "no-food" problem and the
architecture handles 100%-scale biology correctly once fed.

If the prediction fails: there's a deeper issue (substrate-saturation
regime change, enzyme-count clamp artifact, or something else) that
the uptake patch can't fix alone. That would be its own publishable
finding — "full-scale whole-cell simulation exposes scale-dependent
artifacts in standard Gillespie approximations."

## Files added

- `cell_sim/layer3_reactions/nutrient_uptake.py` — the patch module
- `cell_sim/tests/compare_uptake.py` — side-by-side comparison harness
- `cell_sim/tests/render_whole_cell.py` — updated with `WC_WITH_UPTAKE=1`

## Notebook integration

The existing `cell_sim_colab.ipynb` Section 11 is already wired to:
- Run Section 11.2 with `WC_WITH_UPTAKE=1` (fed cell)
- Run Section 11.2b with `WC_WITH_UPTAKE=0` (starving cell comparison)
- Plot both on the same axes in Section 11.3 (ATP, AMP, G6P, PPi panels)
- Show ATP/ADP ratio with physiological range shaded
- Event breakdown in Section 11.4 with uptake rules highlighted in green

## How to apply

```bash
tar -xzf phase4_uptake.tar.gz
cp phase4_uptake/cell_sim/layer3_reactions/nutrient_uptake.py \
   ~/path/to/cell/cell_sim/layer3_reactions/
cp phase4_uptake/cell_sim/tests/compare_uptake.py \
   ~/path/to/cell/cell_sim/tests/
cp phase4_uptake/cell_sim/tests/render_whole_cell.py \
   ~/path/to/cell/cell_sim/tests/

cd ~/path/to/cell
git add cell_sim/layer3_reactions/nutrient_uptake.py \
        cell_sim/tests/compare_uptake.py \
        cell_sim/tests/render_whole_cell.py
git commit -m "Nutrient-uptake patch + whole-cell runner + comparison harness"
git push
```

Then rerun Section 11 of the Colab notebook. **Run the fed cell first
(Section 11.2), then the starving comparison (Section 11.2b).** Each
is ~20 min wall at 100% scale with Rust, ~30 min without.

## Fast path for verification before the big run

If you want to confirm the patch works before committing to a 60-min
Colab run, use the comparison harness at 25% scale:

```bash
cd cell_sim
WC_SCALE=0.25 WC_T_END=1.0 python tests/compare_uptake.py
```

Wall time: ~5 min for both runs combined. Output: `data/whole_cell_compare/
summary.txt` and `comparison.png`.
