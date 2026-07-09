# Cancer-cell model — updates to do

Working roadmap for the whole-cell cancer map (`colab/full_cell_map.py`, `cancer_cell_map.py`,
`context_dependency.py`, notebook `colab/cancer_cell.ipynb`). Each item states the **problem**, **why it matters**
(which cancers it unblocks), the **concrete fix** (data source + where it plugs in), and **effort**.

Anti-trap rule carried throughout: measured > predicted; never overwrite a measured fact, flag/compare instead.
Honesty rule: a layer that can't reach an answer must SAY so (orphan flag), not fabricate one.

---

## Where the model stands now (baseline for this list)

Reliable today (full stack, tested): **oncogene-driven cancers of every flavour** — point mutation (BRAF, KRAS,
EGFR), allosteric GOF (JAK2 V617F via role, not proximity), amplification (ERBB2 + ERBB3 co-dep), metabolic
oncogene (IDH1). Correct co-dependencies surface (KRAS→RAF1, ERBB2→ERBB3, BRAF→MEK/MAP2K1).

Not yet reachable (named, structural reasons): **tumour-suppressor / synthetic-lethal targets** — VHL→HIF2A
(degradation edge missing), BRCA→PARP/POLQ and MMR→WRN (loss-context attribution incomplete), and
context-dependent GOF/LOF (JAK1 miscall). These are the P1 items below.

8 layers wired: A molecular (role+ΔΔG+domain+selectivity) · B pathway · C regulatory (stub) · D complexes ·
E metabolic (stub) · F dependency (DepMap co-ess+SL+selective) · G CellGraph perturbation · H tissue baseline.
Plus the standing **context-dependency attribution** layer on `CompleteCell.gene(g)`.

---

## P1 — unblocks a named failure class

### U1. Degradation / E3-ubiquitin-ligase → substrate edge layer
- **Problem:** VHL loss does NOT light up HIF2A/EPAS1. The VHL→HIF relation is protein DEGRADATION
  (VHL ubiquitinates HIF for destruction); our propagation only carries signalling/transcriptional edges, so a
  whole edge *type* is missing. The complex layer sees the VHL E3 complex is broken but nothing propagates
  "E3 lost → substrate stabilised → up".
- **Unblocks:** 786-O renal (VHL→HIF2A, the belzutifan axis); MDM2→TP53; SKP2/FBXW7→substrates; any
  ubiquitin-driven cancer. Broadly, ~10–15% of the tumour-suppressor class.
- **Fix:** add a signed **degradation layer** `degradation.json` = E3 → substrate, sign −1 (E3 represses substrate
  abundance). Source: UbiBrowser 2.0 (E3–substrate), Reactome "degradation"/"ubiquitination" pathways, or
  the SIGNOR "post-translational" degradation edges we may already carry but filter out. Rule in propagation:
  an E3 knocked out (LOF, clamp −1) → its substrates get **+1** (released/stabilised). Plug into
  `cancer_cell_map._signed_out` as an extra edge set, tagged so the sign semantics are correct
  (E3 loss → substrate UP, opposite of a normal activating edge).
- **Effort:** medium. Data fetch + edge fold + a sign-convention test (VHL-KO should raise EPAS1).

### U2. Loss-context attribution — complete the orphans (Colab)
- **Problem:** WRN (#92), PRMT5, POLQ, USP1, PKMYT1 are strong selective dependencies but **orphan** — their
  lesion is a *loss* (MMR, MTAP-del, HRD) invisible in the gene-effect matrix, so co-essentiality can't attribute
  them. Local DepMap matrix stripped per-line ModelIDs/MSI labels.
- **Unblocks:** MSI-H → WRN, HRD/BRCA → POLQ/PARP, MTAP-del → PRMT5, CCNE1-amp → PKMYT1. The entire
  synthetic-lethal trial-target class.
- **Fix:** the hook exists — `context_dependency.resolve_orphans_colab(gene_effect_csv, model_csv, genes)`.
  In Colab: download `CRISPRGeneEffect.csv` (ModelID-indexed) + `Model.csv` (MSIStatus, OncotreeLineage) +
  `OmicsSomaticMutations.csv` (per-line driver flags) + copy-number (for MTAP/CDKN2A deletion). Regress each
  orphan's gene-effect on the lesion features; fold `lesion_context` back into `context_dependency.json` and
  clear `orphan`. Notebook cell 7 already scaffolds the download + regression; needs a real run + the
  copy-number feature added for MTAP/CCNE1.
- **Effort:** low–medium (mostly a Colab run; add CNV features). **This is the single highest-value item.**

### U3. Context-dependent GOF/LOF caller (replace the static role list for the call)
- **Problem:** JAK1 is called GOF from the role list, but in MSI-H it's an immune-escape **loss**. A static
  oncogene/suppressor list can't be context-aware; ΔΔG can flag destabilising LOF but a stability-neutral
  activating mutation stays ambiguous (can't call GOF).
- **Unblocks:** MSI-H (JAK1/B2M immune-escape losses), any dual-role gene (NOTCH1, MEN1, GATA3, RUNX1,
  TP53-GOF hotspots).
- **Fix:** per-mutation call from three signals combined: (1) **ΔΔG** (destabilising → LOF, already wired),
  (2) an **activating-hotspot / effect annotation** table — OncoKB / CIViC / COSMIC "Mutation Somatic Status +
  functional effect" giving GOF/LOF per residue, (3) **truncation** (frameshift/nonsense → LOF). Precedence:
  measured effect annotation > ΔΔG > role list. Emit a confidence + the evidence used. Plug into
  `cancer_cell_map.classify`.
- **Effort:** medium. OncoKB needs a (free academic) token; CIViC is open. Start with CIViC.

---

## P2 — completes the stack (the layers still stubbed or GPU-only)

### U4. Wire the learned GNN + perturbation→wildtype into layer G (GPU/Colab)
- **Problem:** layer G uses the fixed numpy CellGraph. The trained R-GCN (`make_gnn_notebook`) and the
  Replogle-validated perturbation→wildtype predictor are not fed into the map; they need a GPU.
- **Fix:** in Colab, train/load the R-GCN embeddings, expose a `predict_expression_response(clamps)` that returns
  the ML-predicted Δexpression per gene, and use it as layer G instead of `perturb_downstream`. Compare against
  the signed-propagation map (agreement = corroboration).
- **Effort:** high (GPU training + interface). Notebook cell 8 documents the entry point.

### U5. Live metabolic layer (FBA / PROM), not a membership stub
- **Problem:** layer E only lists which mutated genes are metabolic; it never re-solves flux. For IDH1/2, FH,
  SDH, metabolic rewiring is the phenotype.
- **Fix:** call `rprom.py` / `ecflux` inside the map: apply the mutation as a flux constraint (enzyme LOF → bound
  its reactions toward 0; GOF metabolic → allow the neomorphic reaction, e.g. IDH1 R132H → 2-HG production) and
  re-solve biomass FBA. Report the changed fluxes/subsystems. Human-GEM downloads in Colab (~40 MB).
- **Effort:** medium. Heavy per-run (FBA solve); cache the WT solution.

### U6. Real regulatory propagation (layer C)
- **Problem:** layer C only counts a mutated gene's TF targets; it doesn't propagate the 612k signed reg edges,
  so transcriptional reprogramming isn't in the map.
- **Fix:** include `reg_out` (signed, row-normalised) in the propagation network alongside causal+sig; report the
  top up/down regulated **transcriptional programmes** separately from signalling. Watch flooding (U9).
- **Effort:** low–medium.

### U7. Restore the concentration / structure / translation data layers
- **Problem:** `structure.json`, `concentration.json`, `translation.json` are Colab-only (not downloaded locally),
  so pLDDT, absolute copy number, and translation efficiency are dark on a laptop.
- **Fix:** the data-hunt builders exist (`new_data.py`); ensure the notebook restores them from Drive and the map
  reads them (e.g. abundance-weight the perturbation: a low-copy enzyme's loss matters less).
- **Effort:** low (mostly wiring + Drive restore).

---

## P3 — accuracy & robustness

### U8. Proper tissue → cell-type mapping (emask), fix the keyword matcher
- **Problem:** the emask matcher is substring keywords; 'epitheli' matched **kidney** for an **ovarian** tumour.
- **Fix:** replace `emask_layer` keyword match with a curated tumour-type → Cell-Ontology term map (or match on
  the OncotreeLineage the attribution layer already uses). Fall back to keyword only if unmapped.
- **Effort:** low.

### U9. Normalise the CellGraph perturbation (it floods)
- **Problem:** `perturb_downstream` marks ~15k genes affected (no row-normalisation) — only the top ranks are
  meaningful; the count is noise.
- **Fix:** row-normalise `directed_signed` W by out-degree (same fix that de-flooded the pathway map), or report
  only the top-k by |effect| and drop the raw count.
- **Effort:** low.

### U10. ΔΔG: variant granularity + strengthen the model
- **Problem:** ΔΔG needs the exact substitution (V600**E**); panels carry only position, so ΔΔG runs only for
  genes with a hand `META` entry. Model also under-calls some destabilisers (VHL R167W read neutral).
- **Fix:** (a) pull the substitution from the variant input (require `wt/mut` in the genome, or annotate via a
  VCF/MAF). (b) strengthen ΔΔG with the ESM-2 feature (task started as "Gap 1") when GPU is available; keep the
  CPU model as fallback.
- **Effort:** low (a) / medium (b).

### U11. Fusion input type
- **Problem:** BCR-ABL, EML4-ALK, ROS1, RET fusions have no representation — we crudely encode the kinase at a
  residue. No true fusion detection.
- **Fix:** add a fusion input `("BCR::ABL1", "fusion")` that clamps the kinase partner to constitutive GOF (+1)
  and, if the partner provides dimerisation, notes it. Mark clearly it's a fusion, not a point mutation.
- **Effort:** low–medium.

### U12. Mutational-burden / immunotherapy readout
- **Problem:** MSI-H's established target is immunotherapy (hypermutation → neoantigens → checkpoint). This is a
  mutational-rate/immune mechanism, absent from every signalling layer.
- **Fix:** from MMR/POLE-loss (or a supplied mutation count) estimate **TMB / MSI status** → flag
  "checkpoint-inhibitor candidate" + annotate the checkpoint axis (PD-1/PD-L1/CTLA4). This is a rules readout,
  not propagation — keep it clearly separate and honest (it's a biomarker call, not a network prediction).
- **Effort:** low.

---

## P4 — integration & presentation

### U13. Surface the attributed context inside `full_cell_map` output
- **Problem:** the context-dependency layer lives on `.gene(g)` but the map's dependency section still prints a
  raw selective-dep list; it doesn't say "WRN → MSI" / "MAP2K1 → BRAF-context" inline.
- **Fix:** in `full_cell_map.full_map`, for each selective dependency shown, pull `C.gene(g)['context_dependency']`
  and print the attribution (or "orphan — needs lesion table").
- **Effort:** low.

### U14. Full 8-layer HTML visualisation
- **Problem:** `cancer_cell_html` renders only the pathway sub-map; the molecular/ΔΔG/dependency/tissue layers
  aren't in the plot.
- **Fix:** extend the card: mutation chips with ΔΔG + GOF/LOF, a dependency-target panel (with attribution), the
  tissue baseline, and the complex-disruption list. One card = the whole cell.
- **Effort:** low–medium.

---

## Suggested order

1. **U2** (loss-context attribution, Colab) — highest value, lowest effort, unblocks the whole SL trial-target class.
2. **U1** (degradation edges) — unblocks VHL→HIF2A and the ubiquitin class.
3. **U3** (context GOF/LOF) — fixes the JAK1-type miscalls that survive all 8 layers.
4. **U13 + U8 + U9** — cheap correctness/integration wins.
5. **U5, U6, U4** — fill the stubbed/GPU layers (metabolic, regulatory, learned GNN).
6. **U10–U12, U7, U14** — accuracy, fusion, immuno, data, visualisation.
