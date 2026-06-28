# "Active part + kinetics" — does it do just fine?

**Question.** You concluded: *we only need active part and kinetics, that should do
just fine.* I operationalized both and tested them on the only task that has ever
transferred cross-organism for us: predict a NEW organism's essential genes
(train on 8 feba pilots → test on **mtub / DeJesus 2017**, independent truth).

**How each was built**
- **Active part** = Pfam-domain composition (multi-hot, vocab of the 400 most
  common domains, built from training orgs only). The domain is the functional /
  active module and is cross-organism by construction.
- **Kinetics** = sequence-implied **dynamics** descriptors (length, Vihinen
  flexibility, hydropathy, net charge, aromatic/flexible/rigid residue fractions).
  These are honest fold-coupled *dynamics* proxies — **not kcat**. Real turnover is
  not conserved (Bar-Even 2011: median kcat ~14/s spanning 6–8 orders), so there is
  no transferable kcat signal to use, and I did not fabricate one.

## Result (cross-org mtub DeJesus AUC, n=1038, 32% essential)

| Representation | AUC |
|---|---|
| active (Pfam) alone | 0.626 |
| kinetics alone | 0.636 |
| **active + kinetics** | **0.698** |
| conservation | 0.725 |
| ESM | 0.727 |
| active + kinetics + cons | 0.745 |
| ESM + active | 0.737 |
| ESM + active + kinetics | 0.741 |
| **ESM + cons** (prior best) | **0.768** |
| ESM + active + kinetics + cons | **0.773** |

## Verdict — honest

1. **Active + kinetics alone does NOT do just fine.** At **0.698** it lands *below*
   conservation (0.725) and below ESM (0.727) — i.e. below either single signal we
   already had. As a standalone representation it is the weakest option here.

2. **They are mostly already inside ESM.** Adding active+kinetics on top of ESM
   moves 0.727 → 0.741 (+0.014). ESM, being a protein language model, already
   encodes fold → active-site constellation → fold-coupled dynamics implicitly. The
   hand-crafted descriptors are therefore largely *redundant* with it, not additive.
   This is the same lesson as the activator fusion: when two signals see the same
   thing, stacking them barely helps.

3. **The full stack is the new best, but only barely.** ESM+active+kinetics+cons =
   **0.773** vs the prior ESM+cons **0.768** — a **+0.005** gain that is within
   noise on 1038 genes. Real, but not a breakthrough.

4. **Why kinetics couldn't deliver more.** The conserved part of "kinetics" is the
   *dynamics* (normal-mode flexibility), which is fold-determined and thus already
   captured by ESM. The part you might have hoped carried extra information — actual
   catalytic rate / flux — is the part that is **not conserved across organisms**, so
   it cannot transfer by construction. There is no large untapped kinetics signal at
   the gene-essentiality level.

## Bottom line
The winning representation is still **ESM (sequence→fold→function) + conservation**.
"Active part + kinetics" is a correct *intuition about what's invariant* — and ESM
is exactly the thing that already learned those invariants from data. Encoding them
by hand adds at most a rounding-error improvement (0.768 → 0.773) and underperforms
badly on its own. Keep ESM+conservation as the protein-essentiality model; the
explicit active-site/dynamics features are not worth carrying as separate inputs.
