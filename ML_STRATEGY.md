# ML Training Strategy — grounded in what we measured

**The one principle that makes this good:** across this project we did not *guess* which biological questions are
learnable — we **measured** it. So the strategy is not "throw a big model at everything." It is: **train exactly where we
proved signal + labels exist, and refuse the tasks we proved are walls.** Every target below is annotated with the measured
number that justifies (or forbids) training it.

---

## 1. What we measured — the learnability map

| Task | Measured result (this project) | Verdict |
|---|---|---|
| In-vivo TF **binding** from sequence+chromatin | gating→ChIP: kills ~97% of sequence FPs; motif+chromatin strongly predictive | **LEARNABLE (easy)** |
| **Productive** element (Pol II/eRNA) | measured occupancy→productivity r≈0.3; productivity marks discriminate | **LEARNABLE (moderate)** |
| **Functional regulation** (element→gene, CRISPR) | productive + 3D-linked → 31% Significant, **5.7× base rate** | **LEARNABLE (the flagship)** |
| **Direct TF→target** edges | literature-powered: binding→curated regulon **2.3× (p=5e-4)** | **LEARNABLE (well-characterised TFs)** |
| **Direction** (activate/repress) | signed edges ~**73%**; curated (SIGNOR) 2.7× ≫ raw ChIP 1.1× | **LEARNABLE (with curated labels)** |
| Variant → **protein stability/binding** (ΔΔG) | NEXUS bind ΔΔG ≈ 0.5 Pearson / 0.77 hotspot AUC | **LEARNABLE (second track)** |
| Enzyme kcat / flux | ecModel + kcat calibration validated | **LEARNABLE (narrow)** |
| **Sequence → residence time (k_off)** | affinity→productivity **r≈0**; Kd≠k_off; no genome-wide labels | **WALL — do not train** |
| **Genome-wide knockout cascade** (which genes move) | best fused model recall@50 **18%**, graph-alone **3%**, propagation ≈ chance | **WALL — triage only, never claim** |
| Per-TF Perturb-seq response as a dense target | only **GATA1** well-powered of ~30 TFs | **WALL (power) — do not train per-TF** |

**Reading:** the learnable frontier is the **direct, first-order effect of a perturbation** — where a TF binds, whether that
binding is productive, which gene it regulates, in which direction, and how a variant changes a protein. The **indirect
cascade** and **kinetic rate constants** are walls — not modeling failures but missing measurements/power. A good model
lives entirely on the left column and is honest about the right.

---

## 2. Data inventory (real, on disk)

**A. Genomic / regulatory (K562-anchored — the strongest substrate)**
- Sequence: `chr22.fa` (one chrom; genome-wide needs full hg38 — *acquire*), `tf_motifs.json` (743 PWMs), `composite_motifs.json` (479 dimer motifs).
- Chromatin/functional tracks (ENCODE K562, genome-wide peaks): DNase (open), H3K27ac (active), POLR2A + POLR2AphosphoS5 (Pol II), PRO-cap bidirectional (eRNA), `abc_all.bedpe.gz` (276k 3D enhancer→gene links).
- Binding: ChIP-seq — 12 TFs (chr22) + 7 TFs genome-wide (`chip_gw/`); ENCODE has ~300 more K562 TFs to *acquire*.
- **Regulation ground truth:** `crispr_egpairs.tsv` — **10,412 CRISPR element→gene pairs** with `Significant` labels (the flagship's answer key).
- Perturbation: `k562.h5ad` (2,285 essential-gene Perturb-seq, deep), `gwps.h5ad` (11,258 × 8,248 genome-wide, shallow), `norman.h5ad`, `hct116.h5ad` (different cell — transfer test).

**B. Regulation labels / networks**
- `trrust_regulon.json` (795 TFs / 9,396 signed curated edges), `causal_edges.json` (SIGNOR, 60k signed causal), `regsign.json`, `chip_reg_edges.json`, `tf_core.json`.
- `cell_complete.json` (36 MB: genes, 191k PPI, 612k reg edges, complexes, `emask` 200 cell types), `reactome_pathways.json` (2,792 pathways).

**C. Structural / protein (second track)**
- `enzyme_records.json` (2.6 MB), `davidi_kcat.json`, `saturation_kapp.json` (measured kcat/kapp).
- `domains.json`, `humap_complexes.json`, `interface_pairs.json`, `interface_hotspots.json`, `string_degree.json`, `localization.json`.
- ΔΔG stack: `nexus.json`, `flex_physics.json`, `surface_apbs.json`, `dmasif.json`.

**D. Fitness / dependency**
- DepMap `dependency.M` (K562 gene-effect), `fba_essentiality.json`, `context_dependency.json` (2 MB).

---

## 3. The flagship: a multi-task genomic regulation model for K562

**Why this target.** It is (a) the highest-value learnable task, (b) the *direct-regulon* core of the "TF knockout"
question, (c) backed by real labels (CRISPR) and dense features (the whole productivity+3D stack we built), and (d) a task
the field proves is real (ENCODE-rE2G reaches AUPRC ≈ 0.6–0.7 on this exact CRISPR benchmark). We aim to **match or beat that
SOTA** — a defensible, non-overclaiming bar.

**Task.** Given a candidate `(element, gene)` pair in K562, predict `P(element regulates gene)` (CRISPR `Significant`).

**Labels.** The 10,412 CRISPR element→gene pairs (`ValidConnection=TRUE`), positives = Significant. ~5–12% base rate — a
class-imbalanced, small-label problem. This drives every architecture choice below.

**Features (per pair), all already on disk:**
- *Element (Gate 1–4):* DNase signal, H3K27ac, POLR2A, POLR2AphosphoS5, PRO-cap eRNA; max PWM score for each of the 743
  motifs (or a learned sequence embedding); ChIP occupancy of the TFs we have.
- *3D / distance (Gate 3):* ABC contact score, genomic distance, same-TAD, whether ABC links this exact pair.
- *Gene:* promoter accessibility/activity, expression level, is-TF, essentiality.
- *Pair:* number of other elements competing for the gene; number of other genes competing for the element.

**The key trick — multi-task to rescue the sparse label.** 10k labels is too few for a large model alone. So train a
**shared encoder with dense auxiliary heads** that regularize it:
- Head A — **binding**: predict ChIP for N TFs from sequence (labels are dense: genome × TFs). Teaches motif grammar.
- Head B — **productivity**: predict Pol II / eRNA presence (dense). Teaches "active element."
- Head C — **accessibility**: predict DNase (dense).
- Head D — **regulation (FLAGSHIP)**: CRISPR `Significant` (sparse). Rides on the shared representation.
- Head E — **direction**: TRRUST/SIGNOR sign for TF→target edges (activate/repress).

The dense heads (A–C) give the encoder millions of training positions so the sparse head (D) generalizes. This is the
principled way to make ~10k labels trainable, and it is exactly matched to our data (dense tracks + sparse CRISPR).

**Model tiers (honest, matched to label volume):**
1. **Tier-1 — gradient-boosted trees** on the engineered per-pair features. On ~10k biological labels this usually *wins*.
   This is the first model and the fallback we ship if deep nets don't beat it. Cheap, fast, interpretable.
2. **Tier-2 — sequence CNN** (BPNet/Enformer-lite dilated CNN) encoding the element's DNA, fused with assay features,
   multi-task heads A–E. Only adopted if it beats Tier-1 on held-out chromosomes.

**Evaluation — leakage-proof, non-negotiable (we saw leakage risks this project):**
- **Split by chromosome** — train on a chromosome set, test on held-out chromosomes; also report held-out-**by-gene**.
- Metric: **AUPRC** (imbalanced) + recall@precision. Baselines to beat: distance-only, ABC-only, and published ENCODE-rE2G.
- Report calibration; report per-TF-family performance (we know enhancer-acting TFs like TAL1/RUNX1 behave differently).

**Honest expected accuracy (grounded in measured components):**
- Binding heads (A): **AUC ≈ 0.85–0.95** (dense, easy — the field routinely hits this).
- Productivity head (B): **moderate** (occupancy→productivity r≈0.3 sets the ceiling).
- **Flagship regulation head (D): AUPRC ≈ 0.6–0.7** — matching/approaching ENCODE-rE2G SOTA. This is genuinely good for
  element→gene functional linking, and it is the real number to quote for "the model."
- Direction (E): **≈ 73–80%** on edges with a curated sign.

---

## 4. Second track: variant → protein effect (the mutation side)

Distinct model family, also learnable, serving the *protein knockout / mutation* half:
- **Target:** variant → ΔΔG (stability), Δbinding at a known interface, or Δkcat/flux.
- **Labels/features:** `enzyme_records`/`davidi_kcat`/`saturation_kapp` (kcat), `interface_hotspots`/`interface_pairs`
  (binding), the NEXUS/flex/APBS/dMaSIF structural stack; ESM embeddings.
- **Expected:** ΔΔG-bind ≈ 0.5 Pearson / 0.77 hotspot AUC (measured); a learned head over these features should hold or
  modestly improve. **NOT** absolute affinity, **NOT** de-novo complex formation (both unreliable — established).

Both tracks share one honest frame: **they predict the direct, first-order effect of a perturbation (a variant, or a TF/
element knockout). Neither predicts the downstream cellular cascade — that is the wall, explicitly out of scope.**

---

## 5. What we will NOT train (the discipline that makes it credible)

- **The genome-wide knockout cascade** (which genes move on a KO). Measured ceiling ~18% recall@50, ~3% from network
  structure, propagation ≈ chance. We ship at most a *base-rate triage* model, clearly labeled as triage, never as
  cascade prediction.
- **Sequence → residence time / any rate constant.** Affinity→productivity r≈0; no genome-wide k_off labels exist.
- **Per-TF Perturb-seq response** as a dense supervised target — only GATA1 is well-powered; training it would learn noise.

Refusing these is not timidity; it is what stops the model from producing confident garbage on the 80% of the biology that
is a measurement gap rather than a learning gap.

---

## 6. Execution plan (phased, with go/no-go gates)

1. **Data assembly.** Fetch full hg38 (genome-wide, not just chr22) + ~50–100 more ENCODE K562 TF ChIP tracks. Build the
   unified per-pair feature matrix anchored on the CRISPR pairs. *Gate: feature matrix reproduces the `regulate` 5.7× signal.*
2. **Tier-1 GBM.** Train the gradient-boosted regulation head with chromosome-held-out CV. *Gate: beats distance-only and
   ABC-only baselines on held-out AUPRC.* If it doesn't, stop — the features don't carry it and no deep net will fix that.
3. **Tier-2 multi-task encoder.** Add the sequence CNN + dense auxiliary heads. *Gate: beats Tier-1 on the SAME held-out
   split.* Adopt only if it wins.
4. **Direction + TF-edge head.** Add sign prediction on curated edges.
5. **Second track (structural).** Train the variant→effect head on the kcat/interface/ΔΔG stack.
6. **Honest report.** Per-head accuracy vs baselines vs SOTA, held-out, with the walls stated. Wire the winning model as a
   syscall (`predict_regulation ELEMENT GENE` / extend `bindreg`).

---

## 7. Bottom line

**The deliverable is a functional-regulation model whose flagship number is AUPRC ≈ 0.6–0.7 on CRISPR-validated element→gene
links (SOTA-competitive), plus ≈0.85–0.95 binding heads and ≈73–80% direction — an assembled *direct-regulon* predictor for
K562.** It is "good" precisely because it is scoped to the measured-learnable frontier and refuses the measured walls, so
every number it reports is real and defensible. It sharpens *who a TF/element directly controls and how* — it does not, and
will not claim to, predict the whole-cell cascade.
