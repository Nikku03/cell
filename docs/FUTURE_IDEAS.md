# Future ideas & parking lot

Things we **designed toward but left unbuilt, unfinished, or deliberately deferred** — so we can return and
know *what it was, where it fits, what was planned, the data source, and why it's parked*. This is a living
doc: **when a new idea is deferred, add an entry; when one is built, move it to its docs and note the commit.**

Entry template: **what it is → where it fits → plan → sources → status/blocker → effort/value.**

Legend: 🟢 cheap/next · 🟡 moderate · 🔴 heavy build · ⛔ blocked on data/license.

---

## A. Mutation→phenotype chain — remaining forks

The chain is: `Mutation ─┬─ mutant seq → kcat/Km ─┐  └─ structure → ΔΔG → active fraction [E] ─┴─ ec-flux → pathway → phenotype`.
Built so far: **ΔΔG node** (`docs/DDG_STABILITY.md`, commit 13b7eaa) and **ec-flux rung** (`docs/ECFLUX.md`, commit d7e0944). Remaining:

### A1. Mutant kinetics (the upper fork) — Δ(kcat/Km) from the mutant sequence 🟡
- **What:** predict how a mutation changes catalytic rate/affinity, not just stability. Feeds the kcat side of the fork (complements ΔΔG on the [E] side).
- **Where:** between `colab/ddg_predictor.py` and `colab/ecflux.py`; output is a per-reaction kcat multiplier that joins the active-fraction multiplier in `ecflux.mutation_to_flux`.
- **Plan:** feed WT vs mutant sequence to CatPred → Δ(kcat/Km); validate on measured mutant-kcat pairs (does the model even respond to a single residue change?).
- **Sources:** CatPred (Nat Commun 2025, PMC11871309); **RealKcat** (bioRxiv 2025, PMC11844551) — purpose-built for enzyme *variants*, likely the right tool.
- **Status/blocker:** sequence kcat models are known to be **insensitive to single-point mutations** (why RealKcat exists). Expect an honest test first — CatPred may return ~0 Δ. Needs running the model itself (torch + weights) → Colab, not sandbox.
- **Value:** completes the fork; without it, only the stability arm (→[E]) drives the flux effect.

### A2. End-to-end chain validation on metabolic IEM mutations 🟡 — BUILT, honest negative (blocked on ΔΔG strength)
- **What:** run the whole fork (mutation → ΔΔG → [E] → ec-flux → biomarker) on real IEM mutations; reconstruct the *known* mechanism.
- **Where:** `colab/probe_iem_chain.py` (AlphaFold fetch → ΔΔG → folded fraction → ec-flux), residue-verified against the AlphaFold sequence (the numbering check correctly caught a BCKDHA mismatch).
- **RESULT (honest):** the chain **runs end-to-end** but does **not** recover the IEM mechanism yet — 0/6 documented pathogenic mutations (PAH R408W, MTHFR A222V, GALT Q188R, ALDOB A150P, G6PD S188F, SOD1 A5V) were called destabilizing. **Two diagnosed causes:** (1) the DDGun-tier ΔΔG predictor (r=0.40) regresses to the mean and **under-calls the high-ΔΔG tail** where disease mutations live (they came out ~neutral/−); (2) the folding step at `dG_unfold_wt=7` is **too lenient** (a correct ΔΔG=4 → 99% active). The chain **logic is sound** — a correct ΔΔG (+3.5) with realistic marginal stability (dG_unfold≈4–5) → 50% active → collapse — so the block is **predictor accuracy, not wiring**.
- **Unblock:** a stronger ΔΔG node (**ESM-2 / ThermoMPNN**, GPU-tier — see §A1/B3 and the CellGraph_GNN precedent that learning beats fixed) + per-protein marginal-stability estimate for the folding step. This re-orders the plan: the chain isn't "wire it up," it needs the better ΔΔG first.
- **Sources:** ClinVar/literature (variants), AlphaFold (EBI API), `ecflux` + `iem_mechanism_validation.json`.

---

## B. Kinetics — making flux quantitative & absolute

### B1. Absolute in-cell effective kcat (v/[E]) 🔴⛔
- **What:** real per-second in-cell kcat = FBA flux ÷ absolute enzyme concentration — the number a cell model actually needs (in-vitro kcat overestimates it ~5×).
- **Where:** upgrades `colab/ecflux.py` capacity from the σ-assumption to real capacities; feeds the kinetics layer.
- **Plan:** curated cell-type medium on Human-GEM (close the rich default) → physiological flux; + **absolute proteomics** (copies/cell) → v/[E] with real units; validate v/[E] ≤ in-vitro capacity.
- **Sources:** **PaxDb** human absolute abundance (we only have ordinal 0–15 now); Davidi 2016 (in-vivo kcat, E. coli-derived, noisy proxy); Human-GEM.
- **Status/blocker:** **ordinal abundance is the wall** — need PaxDb copies/cell. Also medium curation (the RECON1/Human-GEM free-energy-loop + rich-default issues we hit).
- **Value:** the single thing that makes flux *absolute* rather than relative.

### B2. Full GECKO ecModel (shared proteome pool) 🟡
- **What:** replace our per-reaction capacity caps with a global protein budget (Σ enzyme mass ≤ total proteome), the real GECKO/sMOMENT constraint.
- **Where:** refinement of `colab/ecflux.py`.
- **Plan:** add enzyme pseudo-metabolites + a proteome-pool exchange; validate it predicts overflow metabolism / Warburg-type proteome-limited switches.
- **Sources:** GECKO (Sánchez 2017; Chen 2021), Human-GEM.
- **Status:** simplified per-reaction caps built; pool constraint deferred.
- **Value:** captures proteome-allocation trade-offs the per-reaction caps miss.

### B3. Structure-based kcat to shrink CatPred's tail 🟡
- **What:** a structure-based kcat predictor for the promiscuous-enzyme tail where CatPred is >10× off (CYP450s, dehydrogenases). We *have* AlphaFold structures — an unused asset.
- **Where:** kinetics layer, alongside CatPred.
- **Plan:** run **KcatNet** on our enzymes; test whether it fixes the specific tail outliers CatPred misses.
- **Sources:** KcatNet (Genome Biology 2026, geometric DL, structure-based); AlphaFold (EBI).
- **Status/blocker:** deferred; median CatPred is already at the ~4× measurement noise floor, so this is a tail-only gain. Heavy (torch + structures).
- **Value:** only the >10× tail; not the median.

---

## C. Temporal-trajectory prediction — a whole new track (NOT started)

Predict a gene's expression **over time** after a perturbation: `mRNA(t) = ∫(synthesis − k·mRNA)dt`. None of this
exists in the repo yet (verified — no `RNADecayCafe`, no EGF test, no half-life relaxation model). Captured
here so the design + the dataset scouting isn't lost. The insight: **the ceiling is set by information, not zero.**

**The error ladder (each rung = an ingredient + a dataset):**

| Rung | Ingredient | Best commercial-safe source | Est. error | Cost |
|---|---|---|---|---|
| 0 | none (assume instant) | — | ~0.55 | — |
| 1 | relaxation shape (half-lives) | RNADecayCafe / Mathieson | ~0.31 | 🟢 build the base first |
| 2 | onset from network cascade depth | **OmniPath** (source-filtered) / **Reactome (CC0)** + **Tullai 2007 IEG tiers** | ~0.20 | 🟡 no experiment |
| 3 | **measured** transcription rate over time | TT-seq human 0–15min (GSE85201) / SLAM-seq (GSE111463) | ~0.10 | 🔴 BigWig→per-gene quantification |
| 4 | generalize across perturbations | **LINCS L1000** (GSE92742/70138, CC-BY via GEO) | ~0.10 | 🟡 only 2 timepoints (6h/24h) |
| — | floor | assay replicate noise (~20–30% CV) | ~0.05–0.08 | irreducible |

- **Two ceilings:** *predictive* (unseen perturbation, onset from topology) ~**0.20**; *descriptive* (measured transcription input) ~**0.05–0.10** but you've stopped predicting and started interpolating a measured signal.
- **Cheapest real gain = Rung 2** (network onset, no experiment). **Biggest leap = Rung 3** (measured input makes the equation exact) but costs a genomic-track-quantification build, not a download.
- **License notes from the scouting:** SIGNOR = CC-BY-**NC** → blocked commercially; KEGG = paid; **Reactome = CC0** (airtight, but sign must be derived); OmniPath `license=commercial` filter currently broken (still returns SIGNOR) → filter sources ourselves; LINCS via GEO FTP = cite-only OK; clue.io API = non-commercial (avoid); MCF10A MDD dense EGF time course = behind free Synapse login (not anon-fetchable).
- **Status:** ⛔ **premise check first** — Rung 1 (half-life relaxation + an EGF test with a *measured* fraction-error) must be built before any "0.20" is real. Do not quote ladder numbers as results until Rung 0/1 exists.

---

## D. Smaller deferred items

- **Wire `kinetics_refined.json` into the cell model** as reaction annotations (kcat_per_s / km_uM / tier per metabolic reaction) 🟢 — offered, not done.
- **Confidence propagation through the mutation chain** 🟡 — each node has error; the chain multiplies them. Carry per-node uncertainty so the end output abstains when a link is weak (consistent with the model's calibration philosophy).
- **In-vivo effective-kcat estimator from abundance + CellGraph** 🟡 — a learned estimator distinct from FBA v/[E].

---

## Deliberately decided AGAINST (don't re-try without new data)

- **Recalibrating CatPred / cell-context kcat correction** — an artifact of partly-synthetic labels; hurts on real measurements. See `docs/KINETICS_CALIBRATION.md` (guardrail axis locks this out).
- **Cell context for in-vitro kcat** — carries ~no information (13.99× vs CatPred 3.3× on real measurements). In-vitro kcat is active-site chemistry; network position is redundant.
