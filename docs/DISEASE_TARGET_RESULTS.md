# Disease → Target pipeline — results

A 3-layer, **blind, out-of-distribution** disease→target selector. The disease is absent from the model's
curated disease layers (`otdis`/`biomarkers`); the drug target is **never named** — it must be *selected*
by simulation. Reproduce on Colab with `colab/disease_target.ipynb`; validate with
`colab/validate_disease_target.py` (the 8th recovery-scorecard axis).

## The pipeline

| Layer | What it does |
|-------|--------------|
| **1 · Causal** | Extract the disease's **apex→readout signal-flow subgraph** (nodes on directed paths from the known driver to the pathogenic readout) and rank pathway candidates by net signed influence on the readout. |
| **2 · Perturb → wild-type** | Degree-normalized **signed influence propagation**. For each candidate, *disable* and *activate* it, re-propagate, and measure how much the pathogenic readout collapses toward wild-type. The required **direction** is read off automatically. |
| **3 · Druggable** | Fetch protein **family/structure** (UniProt — the model itself has almost no structure data) and decide whether the required direction is achievable by that family's **modality**. |

## Result — 2/2 OOD diseases recover the real approved target

### Psoriasis (IL-23/IL-17 axis)
Final call: **IL23A + IL23R** — rescue **1.00**, direction **disable**, druggable
(secreted cytokine → neutralizing antibody / receptor → antagonist).

| gene | rescue | dir | class | verdict |
|------|-------:|-----|-------|---------|
| **IL23A** | 1.00 | disable | secreted cytokine | ✅ guselkumab / risankizumab / ustekinumab |
| **IL23R** | 1.00 | disable | cell-surface receptor | ✅ oral IL-23R antagonists (trials) |
| STAT4 | 0.60 | disable | transcription factor | driver, no ligand pocket → demoted |
| STAT3 | 0.20 | disable | transcription factor | driver, undruggable → demoted |
| JAK2 | 0.00 | disable | kinase | missed (IL23R→STAT4 network shortcut) |

### Atopic dermatitis (Type-2, IL-4/IL-13)
Final call: **IL4R** — rescue **1.00**, direction **disable** — the **dupilumab** target.
Crucially IL4R ≠ the apex: IL-4 alone scored **0.00** (IL-13 routes around it), so the pipeline found the
**convergent receptor** as the bottleneck — exactly why anti-IL4Rα beats blocking IL-4 alone.

| gene | rescue | note |
|------|-------:|------|
| **IL4R** | 1.00 | ✅ dupilumab (anti-IL4Rα) |
| IL13 | 0.60 | ✅ tralokinumab / lebrikizumab |
| JAK1 / STAT6 | 0.40 | partial (JAK inhibitors real; STAT6 a TF) |
| IL4 | 0.00 | routed around by IL-13 → not the lever |

**Recovery scorecard: 8/8** (the 7 prior axes still pass; `disease_target_recovery` added, gated on
recovering a known target in every OOD disease above a random-label baseline).

## Honest scope (what it is / is not)

- ✅ **Mechanism → intervention.** *Given the driver*, it finds the druggable bottleneck + direction + modality, matching approved drugs.
- ❌ **Not autonomous driver discovery.** In an open driver competition (IL-23 vs TNF/IL-6/IL-1β/IFN-γ) IL-23 does **not** uniquely win — STAT3/IL-6 tie higher. The clinical fact "IL-23 blockade > IL-6 blockade in psoriasis" is not in the network topology.
- **Re-discovery**, not novel targets. **Cytokine-cascade diseases** only (metabolic/structural diseases lack the transcriptional readout — e.g. gout's key genes are dynamical dead-ends).
- **Topological, not kinetic.** JAK2 missed in psoriasis (network shortcut). n = 2 diseases.
