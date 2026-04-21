# notebooks/

Colab notebooks that orchestrate compute-heavy work outside the sandbox.

## `colab_bc_sweep.ipynb`

Runs the **B + C sweep blocks** queued in `NEXT_SESSION.md`:

- **B** — multi-seed replicates (5 seeds × 5 detector configs × balanced n=40) to produce mean ± std error bars on the v1–v6 MCC history. Gene panel is fixed (`--panel-seed 42`); only the simulator RNG varies.
- **C** — single sweep at `scale=0.5, t_end=1.0 s` (the one corner Session 5 timed out on in the sandbox). Recorded as `mcc_against_breuer_v8`.

Writes:
- 26 `outputs/metrics_parallel_*.json` files (one per sweep).
- `memory_bank/facts/measured/mcc_replicates_summary.json`.
- `memory_bank/facts/measured/mcc_against_breuer_v8.json`.

### Why not just use the sandbox

The sandbox is 4 vCPUs; Colab A100 instances have 12 vCPUs. Block B = 25 sweeps × ~2 min each = 50 min in the sandbox vs ~20 min on A100. Block C never finished in the sandbox at all (timed out at 9 min on `scale=0.5`).

### GPU is not used

`FastEventSimulator` is stochastic event-driven Gillespie — single-threaded CPU per simulation. The reason to pick a GPU Colab instance is the higher vCPU count that comes with it (L4 → 8, A100 → 12, RTX 6000 Pro → 24). Standard Colab runtimes have only 2 vCPUs — worse than the sandbox.

Recommended: **L4 instance** — cheapest useful option, 8 vCPUs, ~55 min total.

### Running

1. Upload this notebook to Colab (or: Open in Colab → GitHub → paste `Nikku03/cell` URL, select `claude/syn3a-whole-cell-simulator-REjHC` branch, pick `notebooks/colab_bc_sweep.ipynb`).
2. Pick Runtime → Change runtime type → L4 GPU (for the vCPUs, not the GPU).
3. Run cells top to bottom. Each cell's purpose is in its Markdown header.
4. At the end, either push via PAT (final cell) or download the JSONs from the file browser.

### What this notebook does NOT do

- Does not reach MCC > 0.59. Sessions 4–9 diagnosed that ceiling as a simulator-biology gap (pathway redundancy, translation dynamics). More compute won't fix it.
- Does not run the full 458-CDS sweep. The v7 n=20 result made that 3-hour compute unjustified.
- Does not touch simulator internals. That's the Session 11+ plan.

### What this notebook DOES deliver

- Error bars on every existing measurement → makes the history-table comparisons statistically honest.
- One new data point (v8, scale=0.5) → closes the last unmeasured corner of the config grid.

Per the brief: this is measurement hygiene, not a pivot.
