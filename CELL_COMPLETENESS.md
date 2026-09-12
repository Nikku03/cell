# Is the cell model complete? — an honest scorecard

**Verdict: the model is a *complete, validated static map + first-order mechanism* — and an *incomplete dynamical
simulator*. The gap is precisely a measurement gap, not a modeling failure.** Every claim below is tied to a measured number.

---

## ✅ COMPLETE and validated — the parts list, the wiring, and the first-order mechanism

| Layer | What it answers | Status / measured |
|---|---|---|
| **Parts list** | which genes/proteins exist, where they localize, what complexes/pathways they're in | 16,509 genes; localization ~100%; Reactome + decoded pathways; complexes/gene2cplx — **substantially complete** |
| **Wiring (map)** | who physically touches / regulates / depends on whom | PPI 191k, regulatory 612k, causal 60k, synthetic-lethal — **one connected multi-layer graph**; complementary families validated (physical ≠ functional, 46–214× above chance) |
| **In-vivo TF binding** | where a TF actually binds in a cell | gating (open→active→3D) vs ChIP: kills ~97% of sequence false-positives |
| **Binding → regulation** | does that binding actually regulate a gene | productive (Pol II+eRNA) + 3D-linked = **~6× vs CRISPR**; literature regulon vs binding **2.3× (GATA1, p=5e-4)** |
| **Element→gene predictor** | ML on the whole regulation problem | **AUPRC ~0.61** (TF-identity GBM), leakage-controlled, SOTA-competitive |
| **Protein mutation → activity** | how a variant changes a protein | NEXUS dual-sensor **~0.5 Pearson / 0.77 hotspot AUC** (near-field structure) |
| **Direction & context** | activate vs repress; when a pathway is on | sign ~**73%**; condition inference top-1 **67%** |
| **Cross-layer chain** | a TF mutation → its direct regulon | `mutreg`: NEXUS activity × curated direct regulon (first-order) |

This half is real. The model is a genuinely good **encyclopedia + local mechanism engine**: it answers *who, where, what
directly, and which direction* — and it's been validated against measured ground truth (ChIP, CRISPR, SKEMPI, Perturb-seq).

---

## 🟡 PARTIAL — quantitative sub-systems that work only on their domain

| Sub-system | Works where | Doesn't |
|---|---|---|
| **Metabolic flux (FBA / ecFlux)** | the metabolic network (mutation → flux, enzyme-constrained) | non-metabolic genes |
| **kcat / enzyme kinetics** | enzymes with measured kcat (validated) | the rest of the proteome |
| **GRN dynamics** | converges, non-trivial, robust as a *system* | far-field propagation doesn't compose (below) |

---

## ⛔ THE WALL — the dynamical whole-cell cascade (NOT complete)

This is the part that would make it a true *simulator*, and it is **not** solved. Measured, repeatedly:

- **Genome-wide knockout cascade** (which genes move when you perturb one): best fused model **recall@50 ≈ 18%**, graph
  propagation alone **≈ 3% (chance)**, forward field-sim **AUC ≈ 0.50**. The far field does not compose.
- **Per-site residence time** (what sets regulation quantitatively): **uncomputable** from sequence (affinity→productivity
  r≈0); measured occupancy predicts (r≈0.3) but there's **no genome-wide dwell-time assay**.
- **Deep sequence model for regulation**: even the full genome-wide-pretrained CNN (**0.479**) loses to the GBM (**0.608**) —
  you can't out-predict measured ChIP, and 569 CRISPR positives are the binding constraint.

**Why it's a wall (the honest root cause):** every failure traced to a *missing measurement*, not a missing model —
per-site in-vivo residence, deeper perturbation data (only GATA1 is well-powered), context-specific function, and more
labelled regulation. More compute / bigger nets did not move any of these.

---

## Bottom line

- **As a map + first-order mechanism model: essentially complete and validated.** You can ask it who a TF/protein directly
  controls, where it binds, whether that binding regulates, which direction, in what context, and how a mutation dents the
  protein — and get answers backed by measured ground truth.
- **As a whole-cell dynamical simulator: not complete, and not close.** It cannot predict the genome-wide response to an
  arbitrary perturbation, because the cascade doesn't compose and the quantities that would let it (per-site residence,
  context-specific function) are **unmeasured**, not un-modeled.

The most honest framing: **we built a very good cell *encyclopedia* with a working *local-mechanism* engine bolted on — not a
cell *simulator*.** And the session's real achievement is that the boundary between the two is now *measured and named*, so
we know exactly which experiments (not which algorithms) would move it.
