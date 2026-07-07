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

## Cause inference (reverse: phenotype → driver) — `colab/cause_inference.py`

Can it find the *cause* from the phenotype alone (driver never supplied)? Signed VIPER (up **and** down
genes + regulon mode-of-action) + an independent genetic prior (GWAS risk genes). Benchmarked vs the naive
up-only enrichment:

| method | precision@5 | precision@10 | top-5 |
|--------|:-----------:|:------------:|-------|
| A — naive up-only z | 0.20 | 0.20 | DDIT3, STAT6, CEBPE, STAT3, CEBPB |
| B — +down/mode-of-action | — | — | (kills the wrong-direction confound) |
| **C — +genetic prior** | **0.80** | **0.90** | STAT3, REL, CRP, KLF4, RUNX3 |

**What genuinely improved**
- **Mode-of-action kills wrong-*direction* confounds.** STAT6 (Th2 driver — its targets are *down* in psoriasis) drops **#2 → #42**. Signed VIPER penalizes it instead of merely not rewarding it.
- **The integrated shortlist is far cleaner:** top-5 precision **0.2 → 0.8**; the effector-arm confounds (DDIT3/CEBPE) fall out of the top-10.

**The honest boundary (why this is *not* a scorecard axis)**
- On **non-circular** drivers (mechanism TFs *not* in the genetic prior — RORC, NF-κB…), the combined method is **worse**, not better (RORC 25→57, NFKB1 14→101). The network *alone* does not out-rank the naive baseline.
- The clean top-5 is **genetics-driven**. This is **evidence integration (network + human genetics)**, exactly how Open Targets works — **not autonomous network cause-discovery.**
- Two network-only ideas for isolating upstream cause from downstream effector both failed: enrichment rewards small effector regulons; multi-hop "upstream-ness" saturates (the dense graph reaches ~100% of the signature in 3 hops from almost any node).

**Bottom line:** cause-finding works as *multi-evidence integration* — genetics names the causal genes, the network + mode-of-action confirm the TF drivers and strip wrong-direction confounds. Ranking the single true culprit #1 from the *network alone* remains open.
