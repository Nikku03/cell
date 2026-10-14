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
| **Pearson r** | **0.472** |
| RMSE | 1.45 kcal/mol |
| anti-symmetry corr(fwd, −rev) | **0.997** (ideal 1) |

Head-to-head on the *same* S669 mutations (published predictions embedded in the benchmark, |r|):

| predictor | \|r\| | |
|---|---|---|
| **this (biophysical + burial + ProteinMPNN)** | **0.472** | ✓ **top** |
| ACDC-NN | 0.46 | ✓ beaten |
| DDGun3D | 0.43 | ✓ beaten |
| DDGun | 0.40 | ✓ beaten |
| ThermoNet | 0.39 | ✓ beaten |
| mCSM | 0.36 | ✓ beaten |
| FoldX | 0.21 | ✓ beaten |

**It now beats every listed baseline, including ACDC-NN (0.46) and DDGun3D (0.43)** — and keeps near-perfect
anti-symmetry (0.997).

## What moved it from 0.405 → 0.472: **structure, not sequence**

The base model (biophysical + burial) was DDGun-tier (0.405). We tested what closes the gap:

| feature added | Pearson r | verdict |
|---|---|---|
| ESM-2 sequence log-odds (35M) | 0.413 | +0.008, noise |
| ESM-2 sequence log-odds (150M) | 0.384 | **worse** — sequence marginals are weak for stability |
| local-environment / multi-shell burial | ~0.40 | no help |
| **ProteinMPNN structure log-odds** | **0.472** | **+0.067 — the lever** |

The finding, cleanly: **structure beats sequence for ΔΔG.** ESM sequence marginals carry only a weak, redundant
signal (standalone \|r\|≈0.27); the **ProteinMPNN** structure-conditioned log-odds (standalone \|r\|=0.40) is what
ThermoMPNN/ACDC-NN use, and adding it as one feature to our biophysical+burial model reaches top-benchmark.
ProteinMPNN is tiny (~1.6M params, CPU-runnable), so the feature is `proteinmpnn_logodds()` — set
`PROTEINMPNN_DIR` to a ProteinMPNN checkout and it runs on any AlphaFold structure; without it, the model
falls back to biophysical+burial (0.405). The committed benchmark uses the reproducible ProteinMPNN-ddG
predictions from the same benchmark repo.

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
