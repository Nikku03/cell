# CellOS — Approved Products

The **curated registry of final, validated, working capabilities** — the ones that passed honest validation and are
wired into the running cell. It is deliberately small.

The full research log — every experiment, including the dead ends and negative results — lives in `UPDATES.md`. This
file holds only what **shipped**. A capability is listed here only if all three are true:

1. **Validated** — on real held-out or measured data, non-circular;
2. **Wired** — usable as a CellOS syscall (`python3 colab/cellos.py`);
3. **Working** — verified end-to-end.

Every entry states what it does, how to run it, the **measured** numbers, and the honest boundary of what it does *not*
do. If a number isn't here, it isn't claimed.

---

## 1 · NEXUS — mutation → (LOF / GOF) → fold-break / bind-break → cell-dependency&nbsp;&nbsp;·&nbsp;&nbsp;✅ APPROVED

**What it is.** A dual-sensor mutation-effect node, wired live into the cell. Given a protein mutation it (a) calls the
**direction** — loss vs gain of function; (b) decomposes the structural effect into **folding** vs **binding** via two
*orthogonal* sensors; and (c) reports whether the cell **depends** on the protein.

**Run it.**
```
nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]]
  nexus BRAF P15056                    # direction + cell-dependency (structure-free)
  nexus TP53 P04637 175 R H            # a known LOF hotspot
  nexus GHR  P10912 304 W A 1A22 B     # full structural dual-sensor (fold ΔΔG + measured bind ΔΔG)
```
(`nexus` alone = the validation report; `nexus run` = re-run the validation.)

**Validated components** (each measured, non-circular):

| Component | What it establishes | Measured result |
|---|---|---|
| Dual-sensor orthogonality | you need **both** fold + bind sensors | Pearson(ΔΔG_fold, ΔΔG_bind) = **0.15**; a stability-only node misses **94%** of interface breakers (SKEMPI, n=808) |
| Sensor fusion | the 2nd sensor catches what the 1st is blind to | intrinsic-only AUC **0.56** → +extrinsic **0.75** (complex-held-out) |
| Regulatory sign (the GOF lever) | LOF vs GOF *direction* | **86%** precision when it fires, **5.4×** enrichment (oncogene vs tumor-suppressor) |
| dMaSIF surface sensor | interface recognition (the trainable part) | held-out interface AUC **0.90** (260 cplx) → **0.947** (2,198 cplx, 549 held-out) |
| Live cell integration | end-to-end query | verified: GHR W304A → fold **+0.64** (folds) / bind **+4.73** (~2158× weaker); BRAF/ABL1 → GOF; TP53 → LOF |

**What it answers now (verified live):**
- *Is a mutation activating or inactivating?* — BRAF/ABL1 → **GOF-capable** (breakable brake); TP53 R175H → **LOF-only**; KRAS G12D → **GOF**.
- *Does it break folding or binding?* — GHR W304A → **fold intact (+0.64), binding broken (+4.73)** → LOF on the binding axis, which a fold-only sensor is blind to.
- *Does the cell depend on it?* — the kernel's calibrated essentiality (e.g. ABL1 P≈54%, GHR P≈4%), reported as **context**.

**Honest boundaries — what it does NOT do:**
- **activity → whole-cell phenotype is far-field / buffered** — reported as *context*, **not** a validated phenotype predictor (the soft-AND occupancy saturates at high WT affinity, so the ΔΔG is the informative readout, not the occupancy number).
- The **extrinsic axis uses the measured SKEMPI bind ΔΔG** when we have it; a *calibrated predictor* of binding ΔΔG for arbitrary mutations is only the physics node at **r≈0.52**, not exposed as a scalar — so on an unmeasured mutation the binding sensor abstains.
- **Fusing** the dMaSIF surface embedding *into* the ΔΔG node does **not** help (it slightly hurts: node-only hotspot AUC 0.744 → 0.717) — because dMaSIF embeds *wild-type* geometry while a mutation's effect is a mutant-minus-WT delta. So the surface model is a **pipeline** stage (find/score the interface), not a fusion feature.
- The regulatory **sign is recall-limited** (~10% of GOF genes have the brake annotated) — an information gap, not a compute gap.
- **Neomorphic GOF** (a brand-new interface) remains **out of reach** — needs docking + experiment.

**Files.** `colab/nexus_cell.py` (the `nexus GENE …` syscall) · `colab/nexus.py` (dual-sensor node) · `colab/regsign.py`
(GOF lever) · `colab/dmasif.py` + `colab/nexus_train.py` + `colab/nexus_colab.ipynb` (surface sensor, train at scale) ·
`colab/flex_physics.py` + `ddg_predictor` (the ΔΔG sensors) · `colab/fusion_test.py` (the pipeline-vs-fusion check).

---

*Add the next approved product below, same shape: what it is → how to run → measured results → honest boundaries → files.*
