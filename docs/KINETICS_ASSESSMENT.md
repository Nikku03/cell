# Kinetics data availability — can the "kinetics wall" be closed?

Honest assessment of how much quantitative rate/kinetics data exists for human, to decide whether a
kinetics-aware cell model is achievable. (Research: BRENDA/SABIO-RK/BioNumbers stats + ML-kcat papers.)

## The numbers (human)
| source | content | human coverage | license |
|---|---|---|---|
| **BRENDA** (2026.1) | 186k Km, 94k kcat, 49k Ki (all organisms) | human is the #1 organism (~11% of kcat) but only **~1,000–1,500 distinct human enzymes** have any measured kcat | CC BY 4.0 (commercial OK) |
| **SABIO-RK** | ~71k curated reaction rate *laws* (SBML) | >1/3 mammalian; **low thousands** of human reactions | academic-only |
| **BioNumbers** | ~13k quantitative numbers | cross-organism, not systematic human | free |
| **SKEMPI/PDBbind** | PPI binding affinities | **<1%** of the human interactome | mixed |

**Denominator:** Recon3D = 3,288 metabolic genes / 13,543 reactions. Human enzyme complement ~5,000.

## Coverage verdict
- **Km:** ~40–60% of core metabolic enzymes have *at least one* measured value.
- **kcat:** ~20–40% have *any* measured value; complete in-cell parameter *sets* exist for **<10%**.
- **Mechanistic rate laws** (needed for true ODE dynamics): only a **few percent** of the 13,543 reactions.
- **Signaling / PPI kinetics:** **<1%.**
- Even in *E. coli* (best-studied), in-vitro kcat exists for only ~10–12% of reactions. Human is worse.

## ML-predicted kinetics (the coverage fix)
DLKcat, TurNuP, CatPred, CataPro predict kcat/Km for every enzyme — **~100% coverage** — but all train on
the same small BRENDA core: held-out **R²≈0.6 best case, ≈0.3 for novel sequences → routine 2–5× per-parameter
error**, and catastrophic collapse on enzyme families absent from training (e.g. DLKcat Spearman −0.09 on
175 adenylate kinases). Coverage solved; **accuracy is the wall**, and dynamic models compound per-reaction
error multiplicatively.

## Bottom line for us
- **Genome-scale dynamic (ODE) human cell from measured data: NOT possible** — data is hopelessly sparse.
- **Curated-subset kinetic models** (glycolysis, TCA, erythrocyte metabolism): already exist, buildable from
  the literature — a *pathway* at a time, not the cell.
- **Enzyme-constrained stoichiometric models** (GECKO / ecHumanGEM): buildable genome-scale *today* — they
  need only one (imputable) kcat per reaction, not full rate laws. This is the realistic "kinetics-aware"
  layer: it improves flux/proteome-allocation predictions and *ranks* enzyme efficiencies, but it is **not**
  a true dynamic simulator.
- **Would partial help?** Yes, for *ranking and constraints* (ecModel-style), not for trustworthy dynamics.

## Recommendation
Do **not** promise dynamic simulation. The viable, honest path is a **hybrid enzyme-constrained layer**:
Human-GEM (we have it) + ML-predicted kcat (CatPred, with per-prediction uncertainty) + the ~10% measured
BRENDA anchors + fit to omics/flux. That gives *bounded, uncertainty-flagged* rate priors — enough to say
"this enzyme is likely rate-limiting" but never "the concentration of X at t=5min is Y". The kinetics wall is
**approachable (ranking/constraints), not closeable (true dynamics)** with today's data. Matches the project
goal of "everything except kinetics" — and defines exactly how far *toward* kinetics we can honestly go.
