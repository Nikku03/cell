# Predicting gene essentiality in a minimal bacterium

I'm **Naresh Chhillar**, a third-year biomedical sciences student at York University.

Over the past several months I built an event-driven simulator of *Mycoplasma mycoides* JCVI-Syn3A — the smallest viable bacterium ever assembled in a lab — and used it to predict which of its 455 genes are essential for cell survival. Predictions are scored against the published wet-lab knockout study by Breuer et al. (2019, *eLife*). Headline result: **MCC 0.5372** on all 455 genes, comparable to the FBA benchmark in Breuer 2019 (MCC 0.59).

I'm applying for wet-lab internships because the predictions this project generates need bench experiments to mean anything; this repo is the work that led me to that conclusion.

## What I built

A three-layer pipeline that simulates a single bacterial cell stochastically (Gillespie chemistry on a 356-reaction metabolic network), runs an in-silico knockout for every gene, and decides whether each knockout would kill the cell. The decision uses a "vote of seven" detector framework — three detectors look at simulator outputs (metabolite pool changes, reaction firing patterns, parallel-pathway redundancy) and four use biological knowledge sources (protein complex membership, gene annotations). All 455 knockout simulations run in about 50 minutes on a 4-core laptop.

## What I found

- **Matthews Correlation Coefficient = 0.537** against Breuer 2019's experimental labels on all 455 Syn3A genes. This is comparable to but does not exceed Breuer's own FBA-based benchmark (MCC 0.59).
- **287 true positives, 3 false positives, 69 true negatives, 96 false negatives** — precision is essentially perfect (0.990); the gap is recall (0.749).
- **The same 3 false-positive genes have persisted from version 12 onwards** (tracked in `memory_bank/facts/measured/mcc_against_breuer_v15_round2_priors.json`). They're a wet-lab target in this work — see "What this taught me" below.
- **Multi-seed reproducibility verified** — the same simulator at three different RNG seeds produces bit-identical confusion matrices (zero variance).
- **Reproducibly-detected synthetic-lethal pair: `JCVISYN3A_0876 × JCVISYN3A_0878`.** I extended the harness for pairwise knockouts and ran two pilot screens (~445 distinct pairs total). The same pair was flagged by three independent selection criteria with confidence 1.0, against zero false positives in 121 random-pair negative controls. The pair is corroborated by an 8-layer external verification stack including UniProt-curated orthologs in *M. mycoides* SC (Q6MS69 + Q6MS71, Pfam family PF13520 "AA_permease_2") and a strong NCBI BLAST hit to *M. pneumoniae* MPN_308 (E=2e-14). Population-level paralog-vs-random enrichment is NOT statistically significant (Fisher one-sided p=0.23) — n=24 tight paralogs is too small. The pair itself is a wet-lab-testable double-knockout hypothesis.
- **Three documented negative results** are recorded as measured facts — see `RESULTS.md` for full diagnoses.

## What this project does that others don't

- **Cross-language and cross-environment integration.** I built the simulator in Python with a Rust hot path (PyO3) for the Gillespie inner loop, integrated three pretrained protein models (ESM-2 for 1280-dim sequence embeddings, ESMFold for structure prediction, MACE-OFF for substrate bond-energy estimation) as feature extractors, and split execution between local CPU (the simulator) and Colab GPU (the embeddings). The pipeline runs end-to-end. The Rust hot path is 1.86× faster than pure Python on the production sweep — useful but not dramatic; the ML features didn't improve essentiality MCC on Syn3A's 455 genes.

- **Reproducibility infrastructure.** Every measurement in this repo is backed by a JSON fact file under `memory_bank/facts/measured/` that pins the result to a reproducible script, a commit, and a SHA256-validated data cache where applicable. An invariant checker (`memory_bank/.invariants/check.py`) validates the full fact graph in under a second; it currently passes with 47 measured + 14 structural facts. I built this primarily so I could resume work cleanly across sessions; whether it would help another researcher pick the project up is untested.

- **Documented negative results.** Three independent extension attempts — Tier-1 ML feature stacking, longer biological-time simulation, and a toxicity-prediction extension — were each evaluated rigorously and halted with diagnostic detail when the data said stop, rather than dropped silently or quietly buried. Each result is informative about what doesn't work; none of the three turned into a positive contribution.

- **Methodological combination on one evaluation framework.** Stochastic whole-cell simulation, a multi-detector ensemble (3 trajectory-based + 4 knowledge-based detectors), and pretrained-ML feature stacking are all evaluated against the same Breuer 2019 panel using the same MCC measurement. The integration is in one configuration that I haven't seen elsewhere — but the data also shows that on Syn3A's 455-gene set the components are not strongly complementary; the trajectory and ML pieces underperform the knowledge-based priors.

- **Position on the speed-fidelity tradeoff.** The full 458-gene sweep takes about 50 minutes on a 4-core CPU. That sits between Breuer 2019's seconds-scale FBA (steady-state only, no temporal information) and the Thornburg 2026 4D whole-cell model (much richer biology, GPU-days of compute). I haven't yet used this niche for an application that the other tools couldn't address — having the speed available isn't the same as having found the right question for it.

- **Scope discipline across long-running development.** Twenty-two sessions over several months, with a measurable deliverable per session (an MCC number, a feature parquet, a falsification, a notebook). When the data said a direction was over — Tier-1 ML in Session 17, toxicity prediction in Session 22 — I halted and documented the negative result in the same commit. This is process, not science; I list it because the artifact reflects it and reviewers may notice.

## How it compares to existing work

| Approach | Reference | Speed | MCC on Breuer panel |
|---|---|---|---:|
| FBA + curated gene-protein-reaction map | Breuer 2019 | seconds (steady-state) | 0.59 |
| Well-stirred Gillespie | Thornburg 2022 | hours per cell cycle | not measured on this panel |
| 4D whole-cell model | Thornburg 2026 | days per cell cycle | not measured on this panel |
| **This work** (event-driven Gillespie + 7-detector vote) | — | **6.6 s/gene; ~50 min full panel** | **0.537** |

The honest framing: this is a **fast** approach that reaches **comparable but not better** accuracy than the existing FBA benchmark. The speed comes from the Rust hot path (1.86× faster than pure Python, measured) and 4-worker parallelism; the missing 0.05 MCC reflects detector-design choices rather than simulator fidelity.

## Visual summary

Three figures sit under `figures/`:

- **MCC progression** — how the score evolved across detector versions v0 → v15
- **Detector contributions** — isolated MCC per detector; shows that knowledge-based priors carry the v15 result
- **Confusion-matrix grid** — visualises that false positives stayed flat at 3 from v12 onwards while true positives grew

The data is checked in as CSVs and the matplotlib generation scripts are also committed; running `python figures/scripts/plot_mcc_progression.py` produces the PNG/PDF locally. See `figures/README.md` for details.

## What this taught me, and why I'm applying for wet-lab positions

This project sits at the boundary between computational prediction and experimental biology, and what I learned most clearly is that **the predictions need wet-lab tests to mean anything**. Four concrete experiments from this work that would be testable at the bench:

1. **The reproducible synthetic-lethal pair `JCVISYN3A_0876 × JCVISYN3A_0878`.** Across 445 distinct pair attempts in two pilot screens, the same pair was flagged synthetic-lethal by three independent selection criteria, against zero false positives in 121 random-pair controls. Each gene alone is dispensable (0876 Nonessential, 0878 Quasiessential per Breuer 2019); under joint knockout the simulator silences all 18 amino-acid transport reactions. The two genes have UniProt-curated orthologs in *M. mycoides* SC (Q6MS69 + Q6MS71, both "Amino acid permease", both Pfam PF13520 "AA_permease_2"), and 0878 has a strong BLAST hit to *M. pneumoniae* MPN_308 (E=2e-14). The wet-lab test: double-knockout in JCVI-Syn3A, growth at 36 °C in defined medium with full amino-acid supplementation, three biological replicates. Predicted phenotype: 0876-only and 0878-only viable (mild defect for 0878), but 0876 + 0878 fails to grow because amino-acid uptake collapses. ~1 week of bench work.

2. **The 3 stubborn false positives.** Three Syn3A genes are persistently called essential by every detector version since v12, but Breuer 2019 labels them nonessential. Either the simulator has a bug for those genes, or Breuer's labels sit at the boundary of what their growth assay could resolve. A single-gene knockout repeat in the JCVI-Syn3A line — checking growth rate at 36 °C across three independent biological replicates — would tell which it is.

3. **The 96 false negatives, especially the "Uncharacterized" pool.** 84 of the 96 misses are genes annotated only as "Uncharacterized" / "putative" / "hypothetical" — the keyword-prior detectors can't help by construction. A targeted reverse-genetics approach (CRISPRi knockdown of the candidate set in *M. genitalium* as a more tractable proxy organism) would identify which of these are genuinely essential and let the simulator's knowledge base be expanded.

4. **The trxA / hupA / recR specific failures.** These 3 genes are functionally characterised (thioredoxin, DNA-binding HU protein, recombination-repair RecR) but the simulator doesn't capture their essentiality — likely because the failure manifests on biological timescales longer than the 0.5-second simulated window. A growth-curve experiment at varying simulation times, ideally with single-cell tracking, would tell us how long after knockout these failures actually appear.

Each of these is the kind of computational-prediction-meets-bench-experiment work I want to do during a wet-lab internship. The simulator generates testable hypotheses; the wet lab is where they get answered. I'd rather contribute to one of those experiments than write a fourth detector version.

## Repo structure

| Path | What's there |
|---|---|
| `cell_sim/` | The simulator (~10,400 lines): Layer 0 genome → Layer 2 Gillespie → Layer 3 reactions → Layer 6 detectors |
| `cell_sim/layer6_essentiality/composed_detector.py` | The v15 essentiality detector — current best |
| `cell_sim/layer6_essentiality/harness.py` | Knockout harness with both single and pairwise (synth-lethality) modes |
| `cell_sim/features/cache/` | Tier-1 ML features (ESM-2, ESMFold, MACE-OFF) for downstream work |
| `scripts/run_sweep_parallel.py` | Driver for the full 455-gene knockout sweep |
| `scripts/synthlet_pilot_v2_pairs.py` + `scripts/run_synthlet_pilot.py` | Synth-lethality screening pipeline |
| `memory_bank/facts/measured/` | 52 fact JSONs, one per measurement; every claim traces here |
| `memory_bank/.invariants/check.py` | Validates the fact graph; runs in <1 s |
| `figures/` | Plot-ready CSV data + matplotlib scripts |
| `RESULTS.md` | Longer scientific summary |
| `PROJECT_STATUS.md` | Current state + session log |

## How to reproduce the headline number

```bash
git clone https://github.com/Nikku03/cell.git
cd cell
pip install -r cell_sim/requirements.txt

# Stage the upstream Luthey-Schulten input data (gitignored, fetched once).
python -c "
import urllib.request, os
os.makedirs('cell_sim/data/Minimal_Cell_ComplexFormation/input_data', exist_ok=True)
for f in ('syn3A.gb', 'kinetic_params.xlsx', 'initial_concentrations.xlsx',
          'complex_formation.xlsx', 'Syn3A_updated.xml'):
    urllib.request.urlretrieve(
        f'https://raw.githubusercontent.com/Luthey-Schulten-Lab/'
        f'Minimal_Cell_ComplexFormation/master/input_data/{f}',
        f'cell_sim/data/Minimal_Cell_ComplexFormation/input_data/{f}',
    )
"

# Run the v15 sweep on all 455 Breuer-labelled genes (~50 min on 4 workers).
python scripts/run_sweep_parallel.py \
    --all --workers 4 --use-rust \
    --scale 0.05 --t-end-s 0.5 --detector composed \
    --seed 42 --threshold 0.1 --enable-imb155-patches
```

Expected output: predictions CSV + metrics JSON under `outputs/`, with confusion `tp=287 fp=3 tn=69 fn=96` and MCC 0.5372.

## Citations & credits

This project depends on open work from several research groups:

- **Luthey-Schulten Lab (UIUC)** — `Minimal_Cell_ComplexFormation` repo provides the curated SBML + kinetic parameters + protein complex data that underpin Layers 1-3.
- **Breuer et al. 2019, *eLife*** — the experimental gene essentiality labels that everything in this work is scored against.
- **Hutchison et al. 2016, *Science*** — the JCVI-Syn3.0 minimal cell paper itself.
- **Meta AI** — `facebook/esm2_t33_650M_UR50D` for ESM-2 protein embeddings; `facebook/esmfold_v1` for structure prediction.
- **Cambridge / Oxford MACE-OFF developers** — the foundation model used for substrate-bond-energy estimation.

**Anthropic.** I used Claude Code (Anthropic's coding assistant) as a development tool throughout this project. The scientific decisions are mine: when to halt directions that weren't working (Tier-1 ML stacking in Session 17, toxicity prediction at Gate B in Session 22, the v1 synthetic-lethality pilot's NARROW_SCOPE call after the eligibility issue surfaced); which experiments mattered (re-running the synth-lethality pilot in v2 with an eligibility filter applied at selection time; running 8 cheap external verification layers on the 0876 × 0878 pair instead of committing to the 19-hour full screen; shipping at MCC 0.537 rather than chasing Breuer's 0.59 brief by piling on more Syn3A-specific priors); and how to interpret the results (the SBML reframing of the 0876 × 0878 synthetic-lethal finding from "emergent simulator discovery" to "the simulator faithfully executing a curator-supplied OR rule, with the biological credibility coming from the SBML curation itself"). The AI helped me write code faster, debug stuck pipelines, draft documentation, and cross-check my reasoning against the existing fact graph. What one undergraduate can build with this kind of assistance is larger than it was five years ago, and I tried to use that productively while keeping the scientific judgement human.

## Status & limitations

- **Headline MCC 0.537** does not exceed Breuer 2019's FBA at 0.59. The remaining gap is documented in `RESULTS.md`.
- **Single-organism validation only.** The detector stack (especially the annotation-keyword priors and iMB155 metabolic patches) is heavily Syn3A-tuned and would need re-mining for any other bacterium.
- **Layer 5 (biomass + division) is not implemented.** This is whole-cell scope creep that the project deliberately deferred.
- **The Tier-1 ML feature cache exists but the supervised XGBoost detector built on it underperforms v15** — see `RESULTS.md` for the falsification trail.
- **Toxicity prediction was attempted (Sessions 21-22) and halted** when the metabolic SBML turned out not to encode any canonical antibiotic targets. The negative finding is recorded as a measured fact.

## Contact

**Naresh Chhillar**
Third-year B.Sc., Biomedical Sciences
York University, Toronto, ON, Canada
Email: _<your-email-here>_
