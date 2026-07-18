# TODO — tomorrow's build list

Everything we said we'd add, scoped honestly. Each item: **what / why / how (concrete method + data) / honest ceiling**.
Ordered by leverage. The through-line rule stays the same as the whole project: build it, measure it against ground
truth, and name the wall when we hit one — don't fake a cascade.

> **Status update (done today):** C1 ranked/calibrated `propagate` — **DONE** (2.32× AUPRC, committed). A1 splicing — **DONE**
> (real SpliceAI in torch, `splice` syscall, validated on HBB). The first-principles mechanism for every remaining layer
> (capping, poly-A/APA, mRNA-decay, epigenetics) is now derived and PMID-grounded in **`FIRST_PRINCIPLES.md`** — the "how"
> sections below are summarized; that doc is the build spec.

---

## A. Post-transcriptional layers (your explicit list: splicing, capping, tail)

The regulation stack currently stops at *transcription* (a TF sets a gene's rate). A real gene product goes
**pre-mRNA → spliced → capped → cleaved+polyadenylated → exported → translated → degraded**. Four missing stages:

### A1. Splicing — variant → isoform  ✅ DONE
- **Built:** the real pretrained **SpliceAI** re-hosted in torch (`spliceai_torch.py`, no TensorFlow) + wired into NEXUS
  (`splice_nexus.py`, `splice` syscall): variant → SpliceAI delta → activity (1−delta) → feeds `propagate` as LOF.
  Validated on HBB (exon-2 junctions predicted exactly; donor GT disruption → delta 0.999 → activity 0.001; mid-exon → ~0).
- **Ceiling (confirmed):** strong on *canonical* splice-disrupting variants; weak on deep-intronic/tissue-specific; predicts
  splice-site *usage* change, not the exact isoform ratio (needs RNA-seq junctions). See `FIRST_PRINCIPLES.md`.

### A2. 5′ capping — m7G cap state
- **Why:** capping gates stability + translation initiation. Uncapped mRNA is degraded; cap methylation (CMTR1) tunes
  translation. Currently ignored.
- **How:** this is mostly a *constitutive* machinery layer (RNGTT/RNMT/CMTR1), not a per-gene variable — so the honest
  build is (a) a machinery-integrity check (is the capping complex intact? → NEXUS on RNGTT/RNMT), and (b) flag transcripts
  whose 5′UTR structure / TOP motifs make them cap-/eIF4E-sensitive (5′UTR feature scan). Not a big predictive win alone;
  it's a completeness stage feeding A4 (mRNA fate).
- **Ceiling:** capping is near-binary and near-universal; the informative signal is narrow (cap-methylation mutants,
  5′TOP translational control). Build it as a *state flag*, not a predictor. Low expected lift, included for completeness.

### A3. Poly-A tail + alternative polyadenylation (APA)
- **Why:** the 3′ cleavage/polyadenylation choice sets (i) which 3′UTR the transcript carries (→ miRNA/RBP sites →
  stability) and (ii) tail length → stability/translation. APA is a real, measurable regulatory switch (proliferative
  cells shorten 3′UTRs). This is the highest-value of the three RNA-end layers.
- **How:** (a) 3′UTR isoform scan — polyadenylation-site strength (canonical AAUAAA + downstream GU/U elements, a PWM/
  HMM over the 3′ end); (b) couple the chosen 3′UTR to stability via its miRNA-seed + AU-rich-element content (we already
  have miRNA/ncRNA fields in the cell image). A variant that weakens a poly-A signal → readthrough / 3′UTR lengthening →
  altered stability.
- **Ceiling:** poly-A *site* strength from sequence is tractable (well-studied motifs). The *quantitative* tail-length →
  half-life map needs measured 3′-seq / TAIL-seq in K562, which we don't have — so direction (which site, longer/shorter
  UTR) is reliable, magnitude (Δhalf-life) is not. Same shape as the transcription-rate caveat.

### A4. mRNA fate throughline (stability + translation) — the layer that *uses* A1–A3
- **Why:** splicing/cap/tail only matter because they change **mRNA half-life and translation rate**. Right now
  `propagate`'s L1 outputs a *transcription-rate* change; there's no step from mRNA level → **protein** level.
- **How:** a per-transcript stability/translation score = f(3′UTR ARE/miRNA sites [A3], 5′UTR structure/uORFs/TOP [A2],
  NMD trigger from splicing [A1], codon-optimality). Then protein-change = Δtranscription (L1) × translation-efficiency ×
  (1/degradation). We already have `ppm` (protein abundance) + `abund` to calibrate the steady-state.
- **Ceiling:** turns the rate model from "mRNA rate" into "protein level" *mechanistically*, but the coefficients
  (miRNA repression strength, codon→rate) are again population-averaged — direction over magnitude.

---

## B. Epigenetics layer (your explicit list: epigenetics)

- **Why:** we have *hooks* — `invivo_gate` uses DNase/H3K27ac/ABC as a static chromatin gate — but no **epigenetic state
  layer**: DNA methylation, the histone-mark landscape, and how a perturbation *moves* them. A TF knockout that closes an
  enhancer should propagate to loss of H3K27ac → silencing; we don't model that.
- **How, in tractable pieces:**
  1. **Methylation state** — load K562 WGBS (ENCODE) as a per-promoter/enhancer feature; a methylated promoter is a hard
     gate on transcription (extends the `_promoter_rate` proxy with a methylation veto).
  2. **Histone-mark state vector** — per-cCRE {H3K27ac, H3K4me1, H3K4me3, H3K27me3, H3K9me3} from ENCODE K562 (several
     tracks already staged). Classify each element's chromatin state (active/poised/bivalent/repressed) — a proper
     ChromHMM-style state instead of the single H3K27ac gate.
  3. **Writer/eraser perturbation coupling** — mutating a writer/eraser/reader (EZH2, DNMT3A, KDM6A, BRD4…) should shift
     the state of *its* targets. This is the epigenetic analogue of `propagate`, and it's the genuinely novel piece.
- **Ceiling (honest, and this is the same wall):** the *static* chromatin state is measured and reliable (that's what
  `invivo` already uses). The **dynamical** part — "knock out EZH2, which H3K27me3 domains actually decompact and which
  genes de-repress" — is the same far-field cascade that doesn't compose. Expect: static state ✓, first-order writer→direct-
  target ✓, genome-wide chromatin cascade ✗. Build the first two, name the third.

---

## C. Precision & dynamics (today's thread)

### C1. Ranked + calibrated `propagate` (the precision fix) — ✅ DONE (committed)
- **What:** replace the flat blast-radius set (5,183 genes @ ~chance precision) with a single **ranked, calibrated**
  propagation score: `composite = 10·regulon·(0.5+promoter_rate) + RWR_nearfield`, RWR over the weighted multi-layer
  graph (reg 1.0 / PPI 0.5 / complex 1.0 / pathway 0.0).
- **Measured today:** composite AUPRC **2.3×** base, **P@10 ≈ 0.10** (~12× base), top decile **2.4×** enriched;
  beats degree (0.99× = chance), distance, random, and RWR-alone (1.7×); label-shuffle **p=0.015**; robust across |z|
  thresholds. Being independently reverified before commit.
- **Honest bound:** it *ranks* the blast radius so the top is usable; it does **not** make the far field precise. Proven
  why: GATA1's real movers (myeloid de-repression) are **0.0×** enriched for the SPI1/CEBPA curated relay targets — the
  edges connecting a TF to its real secondary program are unmeasured.

### C2. A stronger propagation method (if warranted)
- Candidates to evaluate vs the 2.3× RWR baseline: **heat-kernel diffusion**, **GLIDE** (local+global proximity),
  **network denoising/enhancement** before propagation, and learned perturbation models (**GEARS**, **scGPT**) as an
  upper-bound reference. *(The literature-lens verdict from today's verification workflow will name the specific one worth
  building — fold its recommendation in here tomorrow.)*
- **Prior:** the limiting factor measured today is **missing edges**, not the algorithm — so expect a modest bump at best
  from a better propagator, and a real bump only from better *edges* (C3).

### C3. Close the residual wall where it's actually closable
- The per-TF→target **rate coefficient** and the TF→secondary-program **edges** are the unmeasured quantities. The honest
  levers: (a) mine the directed `reg` graph (612k edges) for the specific de-repression relays that Perturb-seq confirms
  (learn the coefficient from the one well-powered perturbation); (b) treat this as strictly bounded until more perturbation
  data exists (see D).

---

## D. Measurement gaps (what would actually move the wall — not code, but the honest roadmap)

These are why the model is a *complete map + first-order engine* but *not a simulator*. Listed so we don't keep
re-deriving the wall:
- **More well-powered perturbations.** Only GATA1 is well-powered in K562 (confirmed genome-wide: ~2 TFs with ≥15 movers).
  The single biggest lever on `propagate`/cascade validation is more deep, per-perturbation Perturb-seq.
- **Per-site in-vivo residence time (SMT).** The `kinetics` wall: sequence→residence is uncomputable; no genome-wide
  dwell-time assay exists.
- **3′-seq / TAIL-seq / WGBS in K562** for A3/B — turns "direction" into "magnitude" for tail-length and methylation.

---

### Suggested order for tomorrow
1. **C1** — finish + commit the ranked/calibrated `propagate` (verdict permitting). *(fast, today's work)*
2. **A1 splicing** — highest-value post-transcriptional layer, cleanest method (SpliceAI). *(new NEXUS mode)*
3. **A3 poly-A/APA + A4 mRNA-fate** — the stability throughline that makes A1–A3 matter (mRNA→protein). 
4. **B epigenetics** — state vector + writer/eraser first-order coupling (static ✓, cascade ✗).
5. **A2 capping** — completeness state flag (low lift, quick).
6. **C2** — evaluate one stronger propagator only if the edges (C3) justify it.
