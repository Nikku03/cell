# Cross-organism essentiality & the cell model — master synthesis

What the whole project established, what works, what generalizes, and the
honest ceilings. One principle runs through everything:

> **Identity / topology is knowable (often from sequence). Quantity / parameters
> (rates, concentrations, exact binding sites) are not computable from sequence —
> they must be measured or read from functional data. Essentiality is a topology
> question; conditional behavior is a quantity question.**

---

## The wheels (gene/protein essentiality)

| wheel | what it answers | how well | generalizes? |
|---|---|---|---|
| **W1 sequence/ESM** | is this protein essential? | ESM 0.733, **ESM+conservation 0.768** on mtub/DeJesus (cross-org, clean) | ✅ — first signal to beat conservation; transfers because fold→function is conserved |
| **W2 FBA (metabolic)** | is the gene metabolically necessary (this medium)? | precision ~0.45 vs Keio; recovers conditional essentials | ✅ via auto-reconstruction (~80–90%) |
| **W3 assembly/regulatory** | layout + regulation | see below | ⚠️ E. coli-complete, degrades elsewhere |
| **W4 feba fitness** | graded conditional fitness | orthogonal to conservation (verified, +0.095 AUC on mtub) | ✅ for the ~60 feba orgs |

**Universal walls proven:** (1) "transfer = conservation" — any orthology-transferred
signal collapses onto conservation; (2) kcat / reaction rate is **not** conserved
(6–8 orders), so kinetics can't be predicted from family — confirmed by the
enzyme-constrained Wheel-2 test (kinetics added **0** to essentiality).

**Active-site + kinetics descriptors:** tested; add ~nothing over ESM (ESM already
encodes the conserved fold/active-site manifold; kcat doesn't transfer).

---

## The TF / regulatory layer — the deep dive

**The binding SITE is at the information floor; the regulatory EDGE is recoverable.**

| question | result |
|---|---|
| predict operator from sequence (PWM) | AUC ~0.55 (Wunderlich–Mirny wall) |
| + DNA shape / architecture / 4-D tensor | +0.00 — all are functions of the same ~10-bit sequence |
| + cooperativity (homotypic + σ box) | +0.02 (only real sequence lever) |
| + position / occupancy | +0.00 (position: both classes in promoters; occupancy: a per-TF constant) |
| remove coding DNA (search 11% intergenic) | ~2× precision (0.07→0.14); specific TFs → 0.5–1.0 |
| family → operator | real but family-specific: AraC/σ54 ≈ operator; LysR = chance |
| **regulon size ↔ operator specificity** | **r = −0.94** — binding many genes *requires* a degenerate operator |
| **EDGE from co-expression + co-fitness** | **AUC 0.626** (clears the wall), 0.8–1.0 for many TFs |

**The decision rule that falls out:** specific-operator / few-target TFs → predict
from sequence (sharp, intergenic-restricted); global / many-target TFs (CRP, FNR,
IHF…) → operator unpredictable *by necessity* → use the functional edge.

**Network motifs** (RegulonDB, vs degree-preserving null): reproduces Milo/Alon —
negative autoregulation Z=+93, coherent FFL Z=+12, bi-fan Z=+23. Detect motifs in
the *edge graph*, don't predict them from sequence.

**How the cell actually finds operators** (mechanism): it doesn't pinpoint one
site — it slides (facilitated diffusion), binds thousands of degenerate sites
transiently, and *function* emerges from occupancy (concentration) × cooperativity
× position. The deciding bits live in the cell's state, not the DNA — which is
exactly why sequence prediction caps where it does.

---

## The dynamic layer — closing the loop

- **Gillespie GRN engine**: TF occupancy → Hill production rate → mRNA/protein
  concentrations, stochastic, under scenario knobs. Reproduces the real SOS/LexA
  switch (DNA damage → derepression → repair) and the coherent-FFL response delay.
- **Closed loop** (environment → cAMP/CRP → expression → FBA → essentiality):
  on arabinose, CRP-active grows (0.81), CRP-inactive dies (0.00) though
  metabolically capable — **regulation changes the answer pure FBA gets wrong**;
  diauxie reproduced. Conditional essentiality gated by **necessity AND expression**.
- **Genome-wide conditional essentiality** (7 media, iJO1366): 207 core-essential,
  **26 conditional** (11%), textbook-correct (sdh/succinate, mal/maltose,
  atp/acetate). Adding the conditional set raises recall vs Keio (0.43→0.49).

---

## What generalizes vs what is E. coli-specific

| layer | engine | data availability off E. coli |
|---|---|---|
| protein essentiality (W1) | general | ✅ ESM+conservation transfers |
| metabolic (W2/FBA) | general | ✅ auto-reconstruct (~80–90%) |
| regulatory edges | general | ⚠️ co-fitness (~60 feba orgs, weak) / co-expression (few) / measured (E. coli, B. subtilis) |
| motifs / dynamics / closed loop | general | ⚠️ only as good as the edge graph |

The metabolic/topology half generalizes; the regulatory/quantitative half is
data-limited — the same boundary, top to bottom.

---

## BDH (Dragon Hatchling) — the architecture question
BDH is, mechanically, a positive/sparse/signed/scale-free **graph-dynamics SSM** —
the same class biological regulatory networks occupy (strong *structural*
convergence). Its one biologically-wrong part is **Hebbian fast-weight plasticity**
(cells use attractor memory on fixed wiring + effector-gated TF activity, not
co-firing edge updates). Proposal: adopt BDH's trainable graph-SSM form as the
successor to the Gillespie regulatory layer, but **gate edges with effector
signals (our φ(t)) instead of Hebbian σ**, and use Hill not ReLU. Details in
`BDH_cell_analysis.md`.

---

## Honest bottom line
A cell factors into a **knowable backbone** (universal core ~4%, TFs ~4%,
homology-reconstructable metabolism ~30%) and a **hard periphery** (~60%,
lineage-specific / poorly characterized). We can predict the *essentiality
topology* and *conditional logic* well where data exists; we cannot compute the
*quantitative parameters* (rates, sites, concentrations) from sequence — those are
measured. The working system is **hybrid by necessity**: sequence + conservation
for proteins, FBA for metabolism, functional data for regulation, and a dynamic
loop that couples them for conditional essentiality.

---

## UPDATE — current position: genome -> annotated cell blueprint

Later work substantially upgraded the regulatory/quantitative layer from
"measured-only" to "mostly computable". Current cell-layout status:

| layer | status | how |
|---|---|---|
| parts (genes->products->families) | computable | gene-calling, ESM, Pfam |
| essential core | computable | ESM+conservation 0.768 + auto-FBA necessity |
| metabolism + conditional essentiality | computable | auto-reconstruct + FBA + thermo directionality |
| promoter strength beta | computable (intrinsic) | Urtecho MPRA model R^2 0.6-0.97 |
| TF concentration [TF] | mostly computable | beta/gamma + feedback(57%) + inheritance + multiplier |
| global condition multiplier | verified (57%) | master-regulator activities from FBA-growth + conserved effectors |
| specific-TF targets | computable | sequence affinity ordering 0.5-1.0 |
| operons | computable | gene order |
| global-TF activity | computable | conserved effector logic |
| global-TF specific targets | data / anchored | coincidence on specific partner (H-NS+RcsB 0.84); else measured |
| network motifs | computable | edge graph (Milo/Alon reproduced) |
| dynamics -> conditional essentiality | computable | Gillespie + closed loop |

DELIVERABLE: a universal, confidence-tagged cell BLUEPRINT from genome+condition,
every element labeled computed/transferred/measured.

NOT delivered: a complete universal quantitative whole-cell simulator (needs
per-organism measured params; doesn't robustly exist even for one organism).

Residual walls (narrow, bounded): global-TF target identity from sequence; fine
~43% of condition multiplier (supercoiling/org-specific -> calibration); absolute
kinetic/occupancy scales (one constant each); ~30-40% dark genome.

Key chain now closed & verified end-to-end:
  in-vivo beta = intrinsic beta(promoter seq) x SUM master-program activities(effectors)
  [TF] = beta/gamma, with feedback setpoints + inherited bistable states
  specific-TF occupancy -> targets (affinity ordering) -> conditional essentiality (FBA-coupled)
