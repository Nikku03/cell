# First principles of the missing layers — worker, recognition, force

The remaining layers (splicing, 5′ capping, poly-A/APA, mRNA decay, epigenetics) were derived from first principles the same
way NEXUS was built for protein mutations: **model the WORKER enzyme reading its substrate, and the physical force that makes
the reaction favorable.** For each machinery below: *who does it, what signal they read, why the reaction proceeds, what is
computable from sequence vs what needs measurement.* Every load-bearing claim is PMID-grounded.

Splicing is already built (`spliceai_torch.py` / `splice_nexus.py`, the real SpliceAI in torch). The other four are specced here.

---

## The one pattern that repeats

Every one of these layers is the same shape, and it's the shape NEXUS already uses:

> **A worker enzyme reads a recognition signal on its substrate and moves a group from a metabolic cofactor onto a defined
> position. The cofactor is the energy; the recognition pocket is the specificity.**

And the honest split repeats too, matching the project's measured wall:

| Tier | What | Predictable? |
|---|---|---|
| The **menu / site / direction** | which sites exist, their intrinsic strength, the sign of a variant's effect | **sequence-computable** ✓ |
| The **ratio / magnitude / rate** | which site the cell picks *today*, Δhalf-life in hours, per-locus flip probability | **needs measurement** (cell-state: factor abundances, RBP milieu, mark landscape) |
| The **genome-wide dynamical cascade** | which of thousands of loci actually flip and which genes move, quantitatively | **the wall** — doesn't compose |

"Direction is more reliable than magnitude" — the same honesty line the transcription-rate and propagate layers already carry.

---

## 1. 5′ Capping — *provenance-gated, not sequence-regulated*

**Worker & chemistry.** Three enzymes, four activities, fixed order: **RNGTT** (bifunctional 5′-triphosphatase +
guanylyltransferase; one gene, two domains) forms the 5′–5′ GpppN bridge via a covalent lysyl–GMP intermediate releasing PPᵢ;
**RNMT+RAM** N7-methylates the cap guanine (SAM→SAH) = cap0; **CMTR1/CMTR2** 2′-O-methylate nucleotides 1/2 = cap1/cap2. Cofactors GTP + SAM. (Aregger & Cowling 2013 [10.1042/BJ20130378]; Trotman 2018 [10.1074/jbc.RA118.004973]; Shuman & Schwer 1995.)

**Recognition — the worker reads the polymerase, not the RNA.** The capping enzyme docks the **Ser5-phosphorylated Pol II CTD**
(CDK7/TFIIH-deposited, peaks at gene 5′ ends), which both recruits and *allosterically activates* it (Ho & Shuman 1999
[10.1016/s1097-2765(00)80468-2]; Bage 2021 [10.1093/nar/gkab130]). The substrate is a nascent 5′-triphosphate, but the true
discriminator is **"is this a Ser5P Pol II product"** — Pol I/III transcripts also start pppN yet are never capped because they
lack a CTD. The ~22–29 nt emergence gate is pure molecular **geometry** — the distance from the Pol II RNA-exit to each enzyme's
active site (cryo-EM, Garg 2023 [10.1016/j.molcel.2023.06.002]).

**Force.** Covalent-nucleotidyltransferase chemistry pulled irreversible by PPᵢ hydrolysis (same energetics as DNA ligation).
The 5′–5′ linkage is a "molecular hard hat": no free 5′ end for exonucleases + a docking pad for CBC→eIF4E.

**CellOS model — two layers.** (A) **cap status = a near-constitutive Boolean**: `is_PolII_product (Ser5P ChIP at TSS) AND
machinery_intact (NEXUS over RNGTT ∧ RNMT ∧ RAM) AND length≥~30nt`. Do *not* try to predict cap presence from sequence — an
ordinary mRNA is always capped if the machinery is intact. (B) a **sequence-computable modulation layer** that tunes what the
cap *does*: 5′TOP score (mTOR/LARP1 translation control), recapping propensity (3′UTR ARE/miRNA density), cap-dependence.
The testable first-principles prediction is the **hub-vs-modifier asymmetry**: **RNGTT/RNMT LoF = globally lethal** (no cap →
no eIF4E, all mRNAs collapse), **CMTR1/CMTR2 LoF = viable modifier** (cap0 still eIF4E-competent; loses the innate-immune
self-mark → IFIT1/RIG-I flag; Daffis 2010 [10.1038/nature09489]).

**Ceiling.** Deterministic: cap0/cap1 presence given provenance+machinery, the geometric gate, the hub/modifier essentiality
calls. Measurement-only: per-transcript cap0:cap1:cap2 fractions, recapping-pool membership, the RAM/RNMT methylation setpoint.

---

## 2. Poly-A tail & APA — *the site is computable, the choice is cell-state*

**Worker & chemistry.** A megadalton machine: **mPSF** (CPSF160 + **WDR33 + CPSF30 read AAUAAA** — not CPSF160 as long thought;
Chan 2014 [10.1101/gad.250993.114]; cryo-EM Sun 2018 [10.1073/pnas.1718723115]) + **CPSF73 endonuclease** cuts; **CstF64 reads
the GU/U downstream element**; **CFIm25/NUDT21 reads UGUA** (bivalent clamp; Yang 2010 [10.1073/pnas.1000848107]); **PAP**
polymerizes A's template-independently; **PABPN1** sets tail length. Steps: recognize → RBBP6/CFIIm-gated cleavage at a CA →
processive A addition (PPᵢ pull) → length control.

**Recognition — cis-elements at fixed positions.** AAUAAA hexamer ~−21 nt; CA cut site; U/GU DSE ~+14–30; UGUA USE ~−40–100.
**APA site choice** is set by (i) **element strength** (hexamer consensus, DSE richness, UGUA), (ii) **factor abundance** — the
master knobs: **↑CFIm25 → distal/long UTR; ↓CFIm25 → proximal/short** (Masamha 2014 [10.1038/nature13261], glioblastoma);
**↑CstF64 → proximal**; (iii) **Pol II elongation rate** (slow → proximal, "first come first served"; Moreira 2011).

**Force.** AAUAAA is read at **~3 nM affinity** by CPSF30 zinc fingers + WDR33, with a steep K_d penalty for any deviation
(Hamilton 2019 [10.1261/rna.070870.119]) — the thermodynamic reason the hexamer dominates specificity. Cooperative multivalent
assembly converts "a few motifs" into "a committed site"; the CPSF73 cut is gated so it never fires at the wrong place.

**CellOS model.** (a) **Site-strength score** from the four windows (hexamer PWM with position weight + DSE + UGUA + CA) — a
solved problem; ship the transparent PWM/logistic scorer and expose **APARENT** (Bogard 2019 [10.1016/j.cell.2019.04.046]) as a
drop-in CNN (compatible with the existing `seq_model.py` one-hot encoder). (b) **APA predictor**:
`Ψ_prox = σ(β·(S_p−S_d) − β_CFIm·[CFIm25] + β_CstF·[CstF64] − β_elong·PolII_speed)`. (c) **Coupling to decay**: the chosen 3′UTR
fixes its miRNA-seed + ARE load → feeds the stability layer (§3). A variant weakening a poly-A signal → readthrough / UTR
lengthening → more repressive elements → lower stability.

**Ceiling.** Site strength + variant *direction* (stronger/weaker site, longer/shorter UTR): reliable. Cell-type APA *ratios*
(set by CFIm25/CstF64/PAP abundances) and the tail-length→half-life map: measurement (3′-seq/TAIL-seq) — and tail length is
**not** even a clean monotonic half-life predictor (highly expressed genes carry *short* tails; Lima 2017 [10.1038/nsmb.3499]).

---

## 3. mRNA decay / half-life — *many signals collapse onto one rate-limiting node*

**Worker & chemistry.** An assembly line, rate-limited at step 1: **deadenylation** (PAN2-PAN3 then **CCR4-NOT**: CNOT7/8 +
CNOT6/6L on the CNOT1 scaffold) → **decapping** (DCP2/DCP1, DDX6, LSM1-7) → **exonucleolysis** (5′→3′ **XRN1**; 3′→5′ exosome
DIS3/EXOSC10). All specialized pathways — **NMD** (UPF1/SMG), **ARE-mediated** (TTP/ZFP36, KHSRP destabilize; HuR/ELAVL1
stabilizes), **miRNA** (RISC→CCR4-NOT), **m⁶A** (YTHDF2; Wang 2014 [10.1038/nature12730]) — converge on the *same* CCR4-NOT →
decapping → XRN1 pipe. So you model *recruitment pressure onto CCR4-NOT*, not each enzyme.

**Recognition — what sets the rate.** Feature importance in mammals (Saluki meta-analysis, Agarwal & Kelley 2022
[10.1186/s13059-022-02811-x]): **(1) ORF/exon-junction architecture** (dominant), **(2) codon optimality** (CDS — the
codon-stability coefficient; ribosome dwell on non-optimal codons recruits CCR4-NOT via DDX6; Presnyak 2015 [10.1016/j.cell.2015.02.029];
Wu 2019 [10.7554/eLife.45396]), **(3) 3′UTR AU-rich elements** (strongest at UTR termini), **(4) m⁶A** (just 3′ of the stop),
**(5) miRNA seeds** (light per-site, additive). **NMD** (PTC >50–55 nt upstream of the last exon-junction) is the most
deterministic single signal.

**Force.** Deadenylation is the committed, rate-limiting step (losing the poly(A)–PABPC1 clamp ends translational protection),
so it's the ideal integration node — small changes in recruitment rate → large half-life differences. The ribosome is the
sensor (dwell→DDX6→CCR4-NOT); ARE-BP occupancy (TTP vs HuR) is a thermodynamic switch; exonucleases are an irreversible ratchet.

**CellOS model — a log-additive `k_deg`** (interpretable, NEXUS-perturbable):
`ln(k_deg) = β0 + β_cod·CodOpt + β_are·ARE + β_mir·miR + β_m6a·m6A + β_len·ln(L3utr) + NMD·Δnmd`, then
`t½ = ln2/k_deg` and **`[mRNA]_ss = k_txn/k_deg`** — this is the layer that turns a `propagate` transcription-rate change into a
steady-state *level*. Every feature is computable from CDS/3′UTR/miRNA fields. NEXUS coupling: **synonymous → CodOpt lever;
nonsense/frameshift → NMD flip (often the dominant LoF effect); 3′UTR SNV → ARE/miRNA/m⁶A motif change.**

**Ceiling.** Sequence-only mammalian half-life tops out at **r≈0.77 / R²≈0.5–0.6** against *denoised consensus* data (raw data
R²≈0.2–0.39; Saluki). So: reliable **ranking** and t½ within a factor of ~2–3; **absolute half-life in a given cell state needs
measurement** (SLAM-seq/BRIC-seq) — the remainder is cell-state (TTP/HuR levels, expressed miRNAs, m⁶A landscape, signaling).

---

## 4. Epigenetics — *read-write feedback makes bistable domains, and that's the wall*

**Workers & chemistry.** Each mark = a group moved from a metabolic cofactor onto a defined residue. **DNA methylation**:
DNMT3A/3B (de novo, SAM→5mC at CpG; DNMT3L boosts SAM binding ~20×, Kareta 2006 [10.1074/jbc.M603140200]), DNMT1 (maintenance
via UHRF1 reading hemimethylated CpG + H3K9me2/3, Liu 2013 [10.1038/ncomms2562]), TET1/2/3 erasers (Fe/2-OG oxidation, Ito 2011
[10.1126/science.1210597]). **Acetylation** (active, dynamic): p300/CBP (acetyl-CoA→H3K27ac, neutralizes lysine charge →
loosens DNA), HDAC/SIRT erasers, BRD4 reader. **Methylation** (state, no charge — pure recognition epitope): PRC2/EZH2
(SAM→H3K27me3 repressive), MLL/KMT2 (H3K4me3/1), SUV39H/SETDB1 (H3K9me3), KDM erasers.

**Recognition — sequence vs trans, and the feedback that matters most.** Sequence-templated part: **CpG islands** (CellOS
already has this as the per-gene `cpg` flag, 10,257/16,492 genes) → CFP1 reads unmethylated CpG → H3K4me3; the same islands are
default PRC2 nucleation sites → bivalency. Trans part (needs measurement): PRC2/lncRNA/TF recruitment; recruitment is even
*separable* from activity (Lee 2018 [10.1016/j.molcel.2018.03.020]). **The load-bearing mechanism is read-write feedback**: EED
reads H3K27me3 and allosterically activates EZH2 to write more (Ueda 2016 [10.1073/pnas.1600070113]); HP1 reads H3K9me3 →
recruits more SUV39H1. This autocatalysis is what makes marks self-propagating **domains**, not gradients.

**Force & why it's bistable.** Per-event: SAM/acetyl-CoA group transfer, exergonic; the SAM/SAH ratio is the shared
thermodynamic set-point for all methylation. The *interesting* physics is collective: autocatalysis (allosteric activation by
the existing mark) + eraser turnover + **cooperativity + occasional long-range stimulation** gives a system with two stable
fixed points — the canonical result that read-write feedback yields **bistable domains that survive 50% replication dilution,
but only if modification is cooperative** (Dodd, Micheelsen, Sneppen & Thon 2007 [10.1016/j.cell.2007.02.053]). Consequence:
loci are ~ON/OFF, heritable, **hysteretic** — the perturbation to *flip* a domain ≫ the perturbation to *hold* it.

**CellOS model — three pieces on the existing hooks.** (a) **Static ChromHMM-style state vector** per promoter/enhancer from
ENCODE K562 marks {H3K27ac, H3K4me1/3, H3K27me3, H3K9me3, DNase, WGBS} via the existing `crispr_gate._max_signal` reader →
{active/enhancer/poised/bivalent/Polycomb/heterochromatin/quiescent}. (b) **Methylation gate** on transcription rate, applied
*only* where `cpg==1` (dense-island methylation silences; scattered methylation doesn't): `rate_eff = _promoter_rate ×
meth_gate × state_gate`. (c) **`mutate_writer`** — the epigenetic analogue of `nexus_regulate.mutate_tf`: NEXUS ΔΔG → writer
activity `a`; targets = loci carrying that writer's mark (EZH2→H3K27me3/bivalent loci, DNMT→methylated islands, KDM6→gain);
`Δmark = (a−1)·current_mark`; flipped promoters re-enter `propagate` as new L0 seeds.

**Ceiling — exactly the project's measured wall.** Static state = a *labeling* of real ENCODE data (**reliable**). Writer→direct
target = first-order, direction-only (enriched at the direct layer, magnitudes unmeasured — same as the per-TF→target
coefficient). The genome-wide decompaction cascade (**"knock out EZH2, which domains flip, which genes move"**) **does not
compose** — and the reason is first-principles, not a data gap: **bistability + hysteresis** (most domains sit far from
threshold, so losing one writer often does nothing — surviving marks re-propagate), **enzyme redundancy** (EZH1/EZH2,
DNMT3A/3B), and **methylation-vs-H3K27me3 context choice** (in real tumors most transcriptional change on writer perturbation is
DNA-methylation-*independent*; Court 2019 [10.1101/gr.249219.119]). Same shape as the TF knockout-cascade wall.

---

## How this maps onto CellOS (build order)

| Layer | Status | Entry into the stack | Honest ceiling |
|---|---|---|---|
| **Splicing** | **BUILT** (`splice`) | variant → SpliceAI delta → NEXUS activity (LOF) → propagate | canonical sites ✓; deep-intronic/isoform-ratio ✗ |
| **mRNA decay** | spec | log-additive `k_deg` → `[mRNA]_ss = k_txn/k_deg` (closes rate→level) | rank/±2–3× ✓ (R²~0.5); absolute t½ ✗ |
| **Poly-A / APA** | spec | site-strength + APA Ψ → chosen 3′UTR → decay layer | site/direction ✓; cell-type ratio ✗ |
| **Capping** | spec | provenance+machinery Boolean gate; hub/modifier essentiality | presence deterministic ✓; methylation state ✗ |
| **Epigenetics** | spec | ChromHMM state + meth-gate on rate; `mutate_writer` → propagate L0 | static state ✓; genome-wide cascade ✗ (bistability) |

The recurring lesson, now derived from the molecular mechanism in five independent places: **the site, the strength, and the
direction of a perturbation are written in the sequence and are computable; the ratio, the magnitude, and the genome-wide
dynamical outcome are set by cell-state factor abundances and self-reinforcing feedback, and require measurement.** That is not
a limitation of effort — it is where the biology stops being a function of sequence.
