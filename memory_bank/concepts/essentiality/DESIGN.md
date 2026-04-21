# Layer 6 — Essentiality Analysis — DESIGN

Layer 6 is the **scientific answer-generator** for the project. Every other layer exists to make this one's output trustworthy.

## What Layer 6 produces

For each gene `g` in the Syn3A genome, Layer 6 decides:

1. **`essential` (bool)**: does knocking out `g` cause the cell to fail to replicate within one doubling time?
2. **`time_to_failure` (float, seconds)**: if essential, when does the first irrecoverable failure occur? `None` if non-essential.
3. **`failure_mode` (enum)**: which pathway gave up first? One of:
   - `atp_depletion` — ATP falls below viable threshold.
   - `essential_metabolite_depletion` — a named metabolite pool (e.g. G6P, pyruvate, dNTP) crashes.
   - `translation_stall` — amino-acyl tRNA pool crashes / ribosome demand exceeds supply.
   - `transcription_stall` — NTP pool crashes or mRNA count collapses.
   - `membrane_integrity` — lipid synthesis blocked.
   - `dna_replication_blocked` — dnaA, dnaN, dnaE, etc. knocked out.
   - `none` — the knockout ran to completion without triggering any failure signal (non-essential).
4. **`confidence` (float in [0, 1])**: how strong the signal is — monotone in the fractional deviation from WT of the tripped metric.

The predictor output is a single CSV with 458 rows (one per CDS) and these columns plus `locus_tag` and `gene_name`.

## How Layer 6 decides

Layer 6 wraps the event-driven simulator (Layer 2 `FastEventSimulator`) with two adapters:

### Adapter A — `KnockoutHarness`

Given a `Genome`, a list of locus tags to knock out, and simulator hyper-parameters (scale, t_end, seed), produce a fully-populated simulator state with:

- All transition rules whose primary gene product is in the knockout set **removed** (not just their propensity zeroed, but dropped from the rule vector, so the event queue is smaller).
- All protein instances whose `locus_tag` is in the knockout set **removed**.
- Initial mRNA counts zeroed for knocked-out genes (transcription of them cannot restart; there is no gene to transcribe).

### Adapter B — `FailureDetector`

Monitors a running simulation for the seven failure signatures above. Key design choices:

- **Evaluation is periodic, not event-by-event.** Every `delta_t` (default 10 s bio-time) we sample the pools, compute ratios-to-WT, and decide whether any signature has triggered.
- **Ratio-to-WT is the canonical signal.** We need a WT (no knockout) baseline trajectory. A single WT run per `(scale, seed, t_end)` tuple is cached in `cache/wt_baseline_<hash>.pkl` and reused across knockouts.
- **A signature trips when the ratio crosses a threshold at least twice in a row** (to reject transient noise). Thresholds:
  - ATP: KO/WT < 0.5 at two consecutive sample points.
  - Essential metabolites (listed below): KO/WT < 0.2 at two consecutive sample points.
  - Translation stall: mean charged-tRNA / total-tRNA < 0.3.
  - Transcription stall: aggregate NTP / WT < 0.3.
- **`time_to_failure` is the time of the *first* trigger**, not the second-confirm time.
- **`confidence` is `1 - min_ratio_at_trigger`** clamped to [0, 1] — stronger depletions give higher confidence.

Monitored metabolites (named in `initial_concentrations.xlsx > Intracellular Metabolites`): ATP, ADP, G6P, F6P, PYR, dATP, dGTP, dCTP, dTTP, CTP, GTP, UTP, NADH, NAD.

### The sweep

```python
from cell_sim.layer6_essentiality import run_sweep

predictions = run_sweep(
    scale=0.5,
    t_end_s=7200,        # brief target: one doubling time
    sample_dt_s=60,
    seed=42,
    genes=None,          # None = all 458 CDS
    use_wt_cache=True,
)
predictions.to_csv("predictions.csv")
```

The sweep is *embarrassingly parallel* — each knockout is an independent simulation. On a single CPU core it's ~(458 × 20 min) at 50% scale = 150 hours. The sweep therefore should run on a cluster in practice; the harness exposes a `run_one(gene)` function that scripts can parallelise externally.

### The MCC evaluator

`evaluate_mcc(predictions_csv, labels_csv) -> Metrics` loads Breuer 2019's labels and computes:

- Matthews correlation coefficient (binary: `{Essential, Quasiessential}` vs `{Nonessential}` — matches the brief's 0.59 target).
- Confusion matrix.
- Optional: MCC with Quasiessential dropped; MCC with Quasiessential as negative.

## Interface surface

```python
from cell_sim.layer6_essentiality.harness import KnockoutHarness, FailureDetector, FailureMode
from cell_sim.layer6_essentiality.sweep import run_sweep
from cell_sim.layer6_essentiality.metrics import evaluate_mcc, Metrics
from cell_sim.layer6_essentiality.labels import load_breuer2019_labels
```

All of Layer 6 is new code; it uses Layers 0-5 but does not modify them.

## What Layer 6 is NOT

- Not a replacement for the simulator. It is a harness around it.
- Not trying to predict essentiality without running the simulator. No machine-learned shortcut. If the sim crashes, we re-run; we don't impute.
- Not handling multi-gene knockouts in the first version (brief's goal is single-gene essentiality only).
- Not trying to match Thornburg 2022's exact protocol — we use the simulator state the existing cell_sim/ gives us, which is not identical to Thornburg's.

## Session-3 delivery vs. deferred

**Delivered in Session 3:**
- This design doc.
- `cell_sim/layer6_essentiality/labels.py` — loader for Breuer 2019 labels (reads the CSV, exposes a binary labeller).
- `cell_sim/layer6_essentiality/metrics.py` — MCC and confusion matrix (pure Python, no sklearn dependency).
- `cell_sim/layer6_essentiality/harness.py` — `KnockoutHarness` and `FailureDetector` skeleton: the API is concrete, the simulator integration is deliberately minimal (a `_run_mock` that exercises the failure-detection logic without booting the full Layer 3-4 metabolic stack). This unblocks the metrics work and lets us test the pipeline end-to-end on mock data.
- `cell_sim/tests/test_layer6_essentiality.py` — unit tests for labels + metrics + harness logic (no full-simulator dependency).

**Deferred to future sessions (NEXT_SESSION.md):**
- Wiring `KnockoutHarness._run_real` to the real `FastEventSimulator + populate_real_syn3a` path (needs pandas/scipy installed and ~20 min per gene; out of this session's budget).
- Running the 458-gene sweep.
- Comparing the resulting MCC to 0.59.
- The six transporter-k_cat fact files noted in the Layers-1-5 triage.
