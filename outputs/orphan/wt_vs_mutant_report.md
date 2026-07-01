# Normal vs mutated protein — does a population variant change the structure?

Take a wild-type protein and a **non-disease population variant**, compare, and report:
same fold or changed? Method: the AlphaFold wild-type structure (where the residue sits:
surface/core, burial, pLDDT) + the amino-acid change (volume/hydrophobicity/charge) + ESM's
variant-effect score. Disease variants included for contrast. Code: `colab/wt_vs_mutant.py`.

*(A single point substitution does not re-fold the backbone; we compare the WT structure
context + the side-chain change. "Fold unchanged" for benign substitutions is the
established result — we did not run a separate mutant structure prediction.)*

## Result

| gene | variant | kind | location | ΔVolume | ΔCharge | ESM | verdict |
|---|---|---|---|---|---|---|---|
| SLC24A5 | T111A | population (light skin) | core* | −28 | 0 | **+2.8** | **fold unchanged** (mutant even favored) |
| SLC45A2 | L374F | population (pigment) | core* | +23 | 0 | −0.2 | **fold unchanged** |
| MC1R | R163Q | population (E.Asian) | surface | −30 | −1 | −1.9 | fold same; **new surface charge** |
| ACKR1 | G44D | population (Duffy/malaria) | surface | +51 | −1 | −0.4 | fold same; **new surface charge** |
| ALDH2 | E504K | population (alcohol) | surface | +30 | +2 | +0.4 | fold same; new surface charge (affects assembly) |
| MTHFR | A222V | population (common) | core | +51 | 0 | −1.1 | local core perturbation (mild) |
| **HBB** | **E6V** | **DISEASE (sickle)** | surface | +2 | +1 | −2.0 | **fold SAME; harmful new surface patch** |
| **PAH** | **R408W** | **DISEASE (PKU)** | core | +54 | −1 | −1.8 | **buried core destabilized (fold broken)** |
*membrane-embedded position — "core" by contact count means lipid-facing, not solvent-buried.*

## The answer — mostly the same shape; the difference is local

**For population (non-disease) variants, the protein's overall structure — its fold — is
essentially unchanged.** The backbone is the same; a single side chain is swapped, almost
always on the surface (or, for membrane proteins, in the membrane-facing region). ESM even
*favors* the pigmentation variant (SLC24A5 T111A, +2.8) — the position happily tolerates it.

What actually changes is a **local property of one residue**:
- a **surface charge** (MC1R, ACKR1/Duffy, ALDH2) — which can retune a surface function
  (a receptor the malaria parasite grips, an enzyme's assembly, hair/skin pigment) *without
  touching the fold*;
- occasionally a **mild core repacking** (MTHFR A222V) that slightly destabilizes but doesn't
  break the protein.

So: **normal vs population-variant protein = same fold, one different surface side chain.**
The protein is still "the same protein"; evolution tuned a dial, it didn't rebuild the machine.

## The instructive contrast with disease

Disease doesn't require a changed fold either — it depends on *where* and *what*:
- **Sickle-cell (HBB E6V)**: the mutant hemoglobin **folds perfectly normally**. The disease
  comes from swapping a surface glutamate (−) for a hydrophobic valine, creating a **sticky
  patch** so deoxygenated molecules polymerize into fibers that deform the red cell. Same
  fold, catastrophic new surface property.
- **PKU (PAH R408W)**: a large residue jammed into the **buried core** destabilizes the fold,
  so the enzyme misfolds and is degraded. Here the fold *is* broken.

## The rule this illustrates

> A mutation almost never changes the overall protein shape. Whether it matters depends on
> **location and property change**: a conservative surface swap tunes function (adaptation)
> or does nothing (benign); a new sticky/charged surface patch can cause disease *without*
> changing the fold (sickle); a bulky change in the buried core breaks the fold (PKU); a hit
> to an active/binding site breaks function directly.

This is exactly why the classifier fuses **location (surface/core/functional site) + the
substitution severity + constraint** — because "is the structure changed" is the wrong
question; "where did it land and what did it change" is the right one.

## Honest caveats

1. We compared the WT structure + substitution, not two independently folded structures; a
   literal ESMFold WT-vs-mutant RMSD would confirm the "fold unchanged" call numerically
   (expected ≈0.2–0.4 Å for these) — heavy on CPU, not run here.
2. Contact-number "burial" is ambiguous for membrane proteins (lipid vs solvent).
3. ESM-8M is a small model; scores are directional, not calibrated effect sizes.
