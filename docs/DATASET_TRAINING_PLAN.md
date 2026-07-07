# Dataset training plan — what to train on, and *how*

The "how" matters more than the "what", so this leads with the method. The one-line thesis learned by running
the cross-validation harness (`colab/crossval_measured.py`): **re-aggregating data the model already summarises
changes almost nothing; the gain comes from using a dataset as a *different axis of evidence* — and only where
that dataset can actually see the thing you're asking about.**

## The four integration modes (the "how")

Every dataset enters through exactly one of these. Pick by what the dataset *is*, not by convenience.

| mode | when | what it does | anti-trap rule |
|---|---|---|---|
| **new fact** | the dataset is a *measurement* of a thing the model predicts (a real drug→target, a measured kcat) | insert as `tier=measured, confidence=1.0`; it *replaces* the prediction | facts win over predictions; never the reverse |
| **validator** | the dataset is an *independent* measurement of the same relationship (co-dependency for a predicted edge) | cross-check each prediction; corroborated → confidence up, contradicted → down | a prediction is only *upgraded/flagged*, the raw data is never edited |
| **feature** | the dataset is a *per-node/edge signal* (expression, dependency profile) | add as an input feature to the learned GNN / a node embedding dimension | keep it a feature; don't let it leak the label |
| **training data** | the dataset gives *new labelled edges/effects* of a type the model already learns | fine-tune the learned GNN / retrain the predictor on the union | leakage-controlled split (held-out **genes**, not edges) |

**The matching rule (the key lesson).** A dataset can only validate/inform predictions it can *see*. Proven the
hard way: `codep` (CRISPR co-dependency) corroborates *known* PPI edges at **23× over random** — it is a valid
validator — yet it corroborated **0** of the audit's predicted edges, because those are in **pan-essential
housekeeping complexes** (spliceosome, ribosome, SRP) whose essentiality doesn't *vary* across cell lines, so
co-dependency is structurally blind to them. Not wrong predictions — wrong validator. **Match the dataset to the
prediction type.**

## The datasets — ranked by *new* signal, with the recipe for each

### 1. Tahoe-100M (drug × cell-line single-cell perturbation) — highest new signal
- **Adds:** measured **drug → transcriptome response** — *interventional*, the axis the static map lacks.
- **Mode:** **validator + new fact** for the weakest layers. Our `drug_interactions` / `disease_target` are
  association-based; Tahoe says what *actually* changed when the drug was given.
- **How:** derive a per-drug response signature (mean Δexpression vs control per cell line); (a) a predicted
  off-target is corroborated if knocking/inhibiting it shifts the same genes; (b) a predicted disease target is
  upgraded if its perturbation *reverses* the disease signature. Plug the signature graph into
  `crossval_measured.py` as `measured`.
- **Blind spot / caveat:** ~50 cancer lines — strong for cancer/proliferation, weak for tissue-specific disease.
- **Size:** ~100M cells → **the morning job**. Derive signatures once (drug × gene matrix, a few hundred MB),
  then only the derived matrix is needed; never load the raw cells into the model.

### 2. DepMap (CRISPR gene-effect + CCLE expression/mutation, ~1150 lines) — strong, downloadable
- **Adds:** **context-variable dependency** — the full, dense co-dependency the model only stores a top-k
  summary of (`codep`, avg 5.7 partners/gene). The stored summary is too thin to validate much; the raw
  correlation matrix is dense enough to.
- **Mode:** **validator** (for context-variable predicted edges) + **feature** (each gene's dependency profile
  as a GNN feature).
- **How:** download `CRISPRGeneEffect.csv`, compute gene–gene Pearson across lines, threshold; swap into
  `crossval_measured.py` as `measured`. Corroborated context-variable completions get a confidence bump.
- **Blind spot:** pan-essential genes (no variance → no correlation) — same as `codep`.
- **Size:** ~200 MB raw → commit only the derived correlation edges for the audit's candidate genes.

### 3. Disease GEO series (per-disease case/control expression) — grounds disease→target
- **Adds:** the **measured dysregulated state** of a disease. Today `disease_target` runs on textbook pathways;
  this lets it run on the real signature.
- **Mode:** **new fact** (the disease readout) → feed as the `readout` to `disease_target_pipeline`, replacing
  the hand-listed effectors; **validator** for the rescuers (does perturbing the target reverse the signature?).
- **How:** NCBI eutils `db=gds` → pick a 2★+ series → GEO2R-style limma case-vs-control → top DE genes as the
  readout. (GEO is reachable here: 187 psoriasis series found.)
- **Blind spot:** batch effects; bulk masks cell-type — pair with the emask cell-type gate.

### 4. Cell-line panels already summarised (Replogle Perturb-seq, coexpr, emask) — mostly done
- **Status:** already aggregated into the model (co-essentiality `darkfn`, `coexpr`, `emask`, measured-cause
  finder). **Re-aggregating adds little** — this is the "sums them up, won't change much" case you named. Only
  revisit for a *specific* disease-matched screen the current aggregates miss (e.g. an immune Perturb-seq for
  psoriasis, which the cancer-line screens don't cover).

## What's built now (so the morning is plug-in, not build)
- **`colab/crossval_measured.py`** — the dataset-agnostic validator harness. It already runs against the
  in-model cell-line signal (`codep`) and produced the matching-rule lesson above. To add any external dataset:
  derive its edge/similarity graph and pass it as `measured`. Scorecard axis `crossval_measured`.
- **`self_consistency.whole_cell_audit`** — the predictions to be cross-validated (150 completions + flags).
- The **anti-trap hierarchy** is enforced everywhere: external measured data enters as facts/validators; it can
  upgrade, downgrade, or flag a prediction, but it never gets overwritten by one.

## The morning checklist (in priority order)
1. **DepMap** `CRISPRGeneEffect.csv` → dense co-dependency → `crossval_measured` (feature + validator). Cheapest
   real external win; storage-light after deriving correlations.
2. **A disease GEO series** → real readout into `disease_target_pipeline` → validate the rescuer reverses it.
3. **Tahoe** signatures (the big one) → ground the drug/off-target/disease-target layer in real interventions.

Each is a *plug-in* to an existing harness, confidence-tagged, facts-over-predictions — not a re-training of the
whole model, and not a re-aggregation of what's already there.
