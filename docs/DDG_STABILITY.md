# ΔΔG stability predictor — the mutation→phenotype keystone

Predicts the folding-stability change **ΔΔG (kcal/mol)** of a single missense mutation. It is the first built
node of the mechanistic chain that explains *why* a mutation changes function, not just *that* it does:

```
Mutation ─┬─→ mutant sequence ─→ CatPred/RealKcat ─→ Δ(kcat/Km) ─┐
          └─→ structure ─→ ΔΔG ─→ Δ(active fraction) ─→ Δ[E] ─────┴─→ ec-flux ─→ pathway ─→ phenotype
```

ΔΔG sits on the lower fork: a destabilizing mutation lowers the folded/**active** fraction of enzyme, i.e. the
usable `[E]` — which is also the missing input for computing in-cell kcat from flux (`kcat = flux / [E]`). So
this one node serves both the mutation chain and the enzyme-concentration estimate.

`colab/ddg_predictor.py` (`DDGPredictor`), gated by `colab/validate_ddg.py` as the 12th recovery-scorecard axis.

## What it is

CPU-only, **no torch / no MSA / no protein language model**. Two feature groups → gradient-boosted regressor:
- **19 biophysical features** — amino-acid property deltas (Kyte-Doolittle hydrophobicity, volume, charge,
  polarity, flexibility, MW), BLOSUM62 substitution score, Gly/Pro and hydrophobic flags, pH/T.
- **1 structural feature** — **atomic contact number** (heavy atoms within 10 Å of the mutation-site Cβ),
  a numpy-only proxy for burial / inverse solvent accessibility — the dominant stability determinant.

Trained on **S2648** with reverse-mutation augmentation (each A→B also trains B→A with −ΔΔG) for thermodynamic
consistency. Convention: **ΔΔG > 0 = destabilizing**.

## Blind validation (S669, ≤30% identity to training)

| metric | value |
|---|---|
| **Pearson r** | **0.405** |
| RMSE | 1.59 kcal/mol |
| anti-symmetry bias `⟨ΔΔG_fwd + ΔΔG_rev⟩` | **+0.001** (ideal 0) |
| anti-symmetry corr(fwd, −rev) | **0.993** (ideal 1) |

Head-to-head on the *same* S669 mutations (published predictions embedded in the benchmark, |r|):

| predictor | \|r\| | |
|---|---|---|
| ACDC-NN | 0.46 | deep, structure |
| DDGun3D | 0.43 | structure |
| **this (from scratch)** | **0.405** | numpy+sklearn |
| DDGun | 0.40 | ✓ matched |
| ThermoNet | 0.39 | ✓ beaten |
| mCSM | 0.36 | ✓ beaten |
| FoldX | 0.21 | ✓ beaten |

It **matches DDGun and beats ThermoNet/mCSM/FoldX** with a fraction of the machinery, and is below the deep
structure-based methods (ACDC-NN, DDGun3D). Its near-perfect anti-symmetry is notable — many older predictors
fail here, calling both a mutation *and its reverse* destabilizing.

## Honest limits

- **DDGun-tier, not ThermoMPNN-tier.** At r≈0.4, a single small-magnitude call is noisy (the SOD1 A4V demo
  even got the sign wrong). Use it as a **ranked mechanistic hypothesis with confidence**, not a precise
  calculator — consistent with the model's abstain-when-unsure philosophy. It's most reliable for
  large-effect buried mutations, weakest for surface/marginal ones.
- **Folding stability ≠ binding/catalysis.** ΔΔG tells you if the protein is destabilized (→ less active
  enzyme). It does *not* itself give the kinetic change — that comes from feeding the **mutant sequence** to
  CatPred/RealKcat (a separate fork), which is why the chain forks rather than chaining ΔΔG→binding→kcat.
- **Errors multiply down the chain**; the ΔΔG node must carry its confidence forward.

## Use on our proteins

`DDGPredictor.alphafold_pdb(uniprot)` fetches the AlphaFold structure (version resolved via the EBI API) and
`predict_from_structure(pdb, chain, pos, wt, mut)` returns ΔΔG — so it runs directly on any human protein, not
just the benchmark PDBs.

## Next in the chain

1. **Mutant kinetics** — feed mutant sequences to CatPred/RealKcat for Δ(kcat/Km) (the upper fork).
2. **ec-flux** — enzyme-constrained FBA (GECKO-style) on Human-GEM so Δ[E]/Δkcat propagate to Δflux
   (Human-GEM FBA already shown to run; see kinetics notes).
3. **Chain recovery test** — run the full fork on metabolic IEM mutations where the pathway→phenotype end is
   already validated (biomarker recovery), checking the chain reconstructs a *known* mechanism.
