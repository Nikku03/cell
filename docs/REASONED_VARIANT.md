# Reasoned variant predictor — reliable, and honest about its ceiling

You asked for two things: **be reliable (not a coin-flip)** and **know *why* it predicts**. This replaces the
stability-only mutation chain (which was a coin-flip, AUC 0.52, because it saw only destabilization) with a
predictor that is far more reliable *and* explains itself — and, critically, refuses to be falsely certain
exactly where the whole field is blind.

`colab/reasoned_variant.py` (`ReasonedVariant.predict`), gated by `colab/validate_reasoned_variant.py`; wired
into `CellQA.mutation_effect`.

## The design

| rung | role | source |
|---|---|---|
| **AlphaMissense** | the reliable **call** — function-aware, catches loss- and change-of-function, not just unfolding | MyVariant.info (dbNSFP) |
| **ΔΔG** | mechanism: *is it destabilizing?* | our ΔΔG predictor + AlphaFold |
| **functional-site distance** | mechanism: *is it at a catalytic/binding residue?* | UniProt curated features |
| **gain-of-function guard** | catches the blind spot the ML shares — overrides a false-benign call | charge→hydrophobic + fold intact |

Every call carries a **reasoning** object: which mechanism(s) fired, the proof (numbers + the specific residue),
a confidence, and — where the ML is known to fail — an explicit **"possible ML blind spot"** instead of a
confident answer.

## What we verified

**1. Big reliability gain.** Swapping the stability-only chain (AUC **0.52**, a coin-flip) for AlphaMissense
lifts discrimination to **0.68 on our hard metabolic-enzyme ClinVar set** (+0.16), and AlphaMissense's published
accuracy on standard ClinVar is **~0.9**. Either way, a real predictor, not a coin-flip.

**2. The honest ceiling — sickle cell defeats even the SOTA.** You named sickle cell, and it turns out to be the
perfect proof that "being sure" is a frontier limit, not an engineering gap. Verified independently via **rs334**:

> AlphaMissense scores sickle cell (HBB Glu6Val) at **0.23 → "benign"**.

Because HbS pathology is **deoxy-polymerization** — a gain-of-function, aggregation mechanism at a *non-conserved
surface residue* that leaves the monomer's fold and function intact. **No per-residue predictor** (AlphaMissense,
ESM, ΔΔG) can see it. This is documented, and our testing rediscovered it cleanly.

**3. The right response — refuse false certainty.** On the sickle-cell pattern (charge→hydrophobic, fold intact)
the predictor does **not** report "benign." It overrides to **"uncertain — possible ML blind spot"** and asks for
a functional assay:

```
HBB E6V  →  AlphaMissense 0.23 (benign)  →  call: "uncertain — possible ML blind spot"
           why: charge→hydrophobic with the fold intact is the sickle-cell-like pattern per-residue
                predictors systematically miss; do not trust the benign call — needs a functional assay
PAH R408W →  AlphaMissense 0.92          →  call: "likely damaging"  (correct, no false GOF noise)
```

## The honest bottom line

- **Reliable? Much more so than before** — from a coin-flip to a real, function-aware predictor (0.68 on our
  hard set, ~0.9 published).
- **Sure? No — and no one is.** The exact disease you picked breaks the world's best model. So the predictor's
  job is not to pretend certainty; it is to give the reliable call, the mechanistic *why*, a calibrated
  confidence, and an explicit **blind-spot flag** when the mechanism is one the ML can't see.

## Next lever

The remaining gap to full reliability is the gain-of-function/aggregation class. The honest fixes: an
aggregation-propensity model (e.g. TANGO/Zyggregator-style) as a fourth rung, and — for the cell-level
consequence — the self-verification step that runs the variant through ec-flux before trusting a "damaging" call.
