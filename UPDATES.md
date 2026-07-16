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

---

## ✅ DONE THIS SESSION (overnight build) — discovery engine + roadmap patches

### The five discovery-engine pieces (recall → discovery)

The roadmap below makes the model a complete, honest RECALL engine. Discovery is a different axis — built and
MEASURED this session:

1. **Blind holdout harness** (`discovery_validation.py`) — the epistemic foundation. Hold out true edges, rebuild
   the graph without them, score recovery split by hardness. EASY (triadic-recoverable) = interpolation; HARD
   (zero shared train-neighbours) = the discovery metric. **Baseline result: ppi AUC 0.795 overall but HARD-edge
   AUC 0.516 (chance), discovery_lift 0.016 — the model INTERPOLATES, it does not discover.** This is the
   yardstick every future change must move.
2. **Learned variant-effect engine** (`molecular_engine.py`) — UniProt seq → ESM-2 zero-shot LLR + AlphaFold ΔΔG
   + truncation. GENERALISES to unseen variants. Tested: BRAF V600E / KRAS G12D / **JAK2 V617F → GOF-candidate**
   (the case proximity could never call); frameshift → LOF. Runs on CPU.
3. **Perturbation→response** — the fixed CellGraph lens is wired (layer G) + normalised (U9). The LEARNED torch
   GNN / Replogle forward-test remain GPU/Colab (Piece 3 partial — see U4).
4. **Discovery from measured data** (`discovery_engine.py`) — propose co-essential non-edges from DepMap, validate
   against a held-out oracle. **22 proposals validated (precision 0.099 vs random ~0), top novel candidate
   ATIC–CAD (both nucleotide-biosynthesis enzymes) — a real, modest discovery signal in the high-confidence
   tail.** Bulk co-essentiality still weak (consistent with piece 1).
5. **Test oracle / closed loop** — held-out edges as the oracle in `discovery_engine.py`; a proposal is accepted
   only if it validates against data the proposer never saw.

**Honest verdict:** the model is quantifiably a recall engine (HARD-edge AUC ≈ 0.5). The variant engine adds a
generalising component; the measured-data tail shows a real but small discovery signal. Discovery is now
*measurable*, which is the prerequisite for improving it.

### Where AlphaFold / Foldseek / UniProt / ESM come in (all four, concretely)

These are the sequence→structure→function substrate — the extrapolation engine that turns a gene ID + mutation
into a generalising call instead of a lookup. Each does a job the others can't:

| Tool | Role | Wired in |
|---|---|---|
| **UniProt** | canonical SEQUENCE + per-residue functional sites (active/binding). The input everything needs. | `molecular_engine.uniprot_seq`, `blind_target` sites, `gene_universe` accessions |
| **ESM-2** | protein LANGUAGE model. Zero-shot variant effect = masked-marginal logP(mut)−logP(wt). GENERALISES to unseen variants — the "is it functional?" extrapolator. | `molecular_engine.esm_scores` (runs on CPU) |
| **AlphaFold** | 3D STRUCTURE → burial/contact → ΔΔG stability (destabilising = LOF) + interface/pocket. Direction evidence ESM can't give. | `ddg_predictor` (fetches structures on demand), `molecular_engine._ddg` |
| **Foldseek** | STRUCTURAL-homology search → transfer function to a query by FOLD when sequence homology is absent (dark proteome) + find structural analogues of a mutated site. | `foldseek_function.py` (built); to be wired into the variant call for dark genes |

Together they are the honest path from retrodiction toward discovery on the sequence axis: ESM extrapolates
effect, AlphaFold gives structural direction, UniProt grounds both, Foldseek covers the un-annotated fold space.

### Roadmap items patched this session

- **P0 ✅** — `gene_universe.py`: 84.1% coverage quantified (3,004 missing, 2,071 isolated nodes); priority
  disease genes (globins, TYK2, PIK3CA, SMN1/2, PAH, GBA, F8/9, …) ADDED as real nodes with independent STRING
  edges, folded into CompleteCell (16,492 → 16,509). Unblocks Bucket A (TYK2) and the PIK3CA cancer gap.
  *Remaining: add the other ~2,990 in Colab; add signed regulatory edges (BCL11A⊣HBG) for the sickle-cell test.*
- **U1 ✅** — `degradation.py`: E3→substrate degradation edge layer (sign −1). VHL-KO now raises HIF1A/EPAS1 (the
  belzutifan axis). Seed of 22 E3s / 55 edges; extend from UbiBrowser in Colab.
- **U3 ✅** — `classify()` now calls the variant-effect engine per mutation. **MSI-H driver flipped JAK1→BRAF**
  (fixed the context miscall from the mutation itself). Needs the real variant identity per gene (VCF/MAF) to
  generalise beyond the hand `variant_meta`.
- **U8 ✅** — tissue matcher priority-ordered (ovarian no longer matches kidney).
- **U9 ✅** — CellGraph perturbation reports strongly-affected, not raw fan-out.
- **U13 ✅** — full_cell_map surfaces the context attribution inline per selective-dep (addiction / lesion /
  ORPHAN), so the data states its own gaps.

### Still open (need GPU/Colab or larger data)

- **U2** — loss-context attribution (WRN→MSI): the `resolve_orphans_colab` hook + notebook cell exist; needs the
  DepMap per-line lesion table (Colab).
- **U4** — learned torch GNN / perturbation→wildtype (GPU).
- **U5/U6/U7** — live FBA, real regulatory propagation, concentration/structure restore.
- **U10** — ΔΔG under-calls destabilisation (TP53 R175H / VHL R167W read neutral) → strengthen with ESM features
  or AlphaMissense; and pull variant identity from a VCF/MAF so U3 runs on real genomes.
- **U11/U12/U14** — fusions, TMB/immuno readout, honest-by-design HTML (prototype done).

---

## P0 — FOUNDATIONAL: complete the gene universe (the model is only ~83% of the genome)

- **Problem:** the model holds **16,492 of ~19,800 human protein-coding genes — ~17% (≈3,300) are absent**, and
  the gap is *idiosyncratic*, not a clean tissue-specificity filter. Category probe (evidence):
  - **Entire hemoglobin locus missing** — HBB, HBA1/2, HBG1/2, HBD, HBE1 (0/7). No hemoglobinopathy (sickle cell,
    β-thalassaemia) can be modelled; the BCL11A→fetal-Hb test was un-runnable for this reason.
  - **Major signalling / cancer genes missing** — TYK2 (autoimmune, deucravacitinib), PIK3CA (one of the most
    mutated genes in cancer — this is what broke the MCF7 case).
  - **Major monogenic-disease genes missing** — PAH (phenylketonuria), SMN1/SMN2 (spinal muscular atrophy),
    PKLR (pyruvate-kinase deficiency), LPA (Lp(a)).
  - Present by contrast: ion channels, GPCRs, keratins, HLA, and every well-studied cancer driver
    (TP53/BRAF/KRAS/EGFR/MYC) — which is exactly why the cancer demo looked strong.
- **What sort of genes we don't have (and the likely cause):** the index appears to come from a dataset
  INTERSECTION (genes surviving DepMap ∩ expression ∩ network QC), which silently drops:
  1. **Paralogs / near-identical duplicates** removed by dedup — SMN1/SMN2, some globins.
  2. **Symbol / ID-resolution failures** — PIK3CA absent while PIK3CB is present points to an Ensembl↔HGNC
     mismatch, not a real biological exclusion.
  3. **Genes absent from a specific source matrix** — globins are low-variance / excluded from some CRISPR
     libraries, so an intersection filter drops them.
  4. **Tissue-restricted genes with sparse interactome annotation** — erythroid, germline, some secreted factors.
  So the missing set is not "the unimportant genes" — it is a scattered ~3,300 that INCLUDES disease-causal genes.
- **Why it's needed (thorough):**
  1. **The stated goal is "map the whole cell, miss nothing."** 83% is not that — ~1 in 6 genes cannot be queried,
     perturbed, or mutated at all.
  2. **Disease coverage is holed exactly where it matters.** The missing set contains disease-CAUSAL genes
     (globins, SMN1, PAH, TYK2, PIK3CA). A cell model without a disease's causal gene cannot model that disease —
     Bucket A proved it: the first two non-cancer tests both hit missing-gene walls immediately.
  3. **The cancer results are flattered by selection bias.** Cancer drivers are the most heavily annotated genes
     in the genome, so they are all present with rich edges; measured capability drops the moment you leave that
     corner of the genome. Coverage must be uniform before cross-disease claims are honest.
  4. **No perturbation without the node.** You cannot knock out or mutate a gene that is not in the graph.
  5. **Edge completeness, not just nodes.** Even present genes miss key edges — BCL11A is in the model but its
     BCL11A⊣HBG repression (the sickle-cell mechanism) is absent.
- **Fix:**
  1. Reconcile the index against the full HGNC/GENCODE protein-coding set (~19,800); enumerate the ~3,300 missing.
  2. Add them as nodes; pull edges from INDEPENDENT sources (STRING, BioGRID, SIGNOR, Reactome, GTEx
     co-expression) — never curated to an answer, so downstream prospective tests stay fair.
  3. Fix paralog dedup (keep SMN1 AND SMN2) and symbol/ID resolution (restore PIK3CA, audit all HGNC aliases).
  4. Tag every gene fully-wired / sparse / isolated-node, so the model reports its OWN coverage per query instead
     of silently returning a partial answer.
  5. Prioritise by disease relevance: OMIM/ClinVar disease genes first (globins, SMN, PAH, TYK2, PIK3CA, …).
- **Effort:** medium–high (reconciliation + multi-source edge fetch + dedup/ID fixes). **Foundational — unblocks
  Bucket A and essentially every non-cancer disease; do before broad cross-disease claims.**

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

### U14. Honest-by-design (self-diagnosing) HTML visualisation
- **Problem:** `cancer_cell_html` is a *results view* — it renders answers with UNIFORM confidence and hides the
  gaps. The MSI-H HTML labelled WRN (the actual trial target) a plain "passenger" and showed JAK1 (a miscall) as
  a confident driver, with no ΔΔG, no selective-dep rank, no orphan flag, no missing-gene markers. A viewer
  cannot read the roadmap off it — the most important items (U2 attribution, P0 missing genes) are invisible or
  actively contradicted.
- **Fix:** make the cell a *diagnostic view*. Every call carries **confidence + evidence** (ΔΔG, selective-dep
  rank, orphan flag, which layer produced it); a panel **derived from the cell's own annotations** lists the
  roadmap items it exposes (passenger-but-top-selective-dep-and-orphan → U2; GOF-by-role ΔΔG-can't-confirm → U3;
  driver gene absent from index → P0). Prototype built: `outputs/orphan/msih_selfdiagnosing.html` — it read U2
  (WRN #92 orphan), U3 (GOF unconfirmed), and P0 (PIK3CA/TYK2 missing) straight off the cell. Generalise it to
  the full 8-layer card and every tumour.
- **Why it matters:** an honest model surfaces its own uncertainty. If the cell can't tell you where it's weak,
  a user will trust a wrong call (JAK1) as much as a right one (BRAF). This is the honesty rule applied to the UI.
- **Effort:** low–medium.

---

## Suggested order

0. **P0 gene-universe completion** — foundational; without it every non-cancer disease hits a missing-gene wall.
   Can run in parallel with the P1 items (it's data plumbing, they're modelling).
1. **U2** (loss-context attribution, Colab) — highest value, lowest effort, unblocks the whole SL trial-target class.
2. **U1** (degradation edges) — unblocks VHL→HIF2A and the ubiquitin class.
3. **U3** (context GOF/LOF) — fixes the JAK1-type miscalls that survive all 8 layers.
4. **U13 + U8 + U9** — cheap correctness/integration wins.
5. **U5, U6, U4** — fill the stubbed/GPU layers (metabolic, regulatory, learned GNN).
6. **U10–U12, U7, U14** — accuracy, fusion, immuno, data, visualisation.

---

## Connected reasoning chain (`colab/reasoning_chain.py`) — the integration layer

The full-test scorecard measured the *departments in isolation* (ESM alone, interface alone, propagation alone),
and each scores mediocre-to-chance on discovery. But that is not how the pieces are meant to run. The value is
the **coupling**: each step's output conditions the next, so a strong step can carry a weak one and localisation
changes *what* is propagated. `reason(gene,pos,wt,mut,...)` wires them into one chain:

  1. **molecular** (ESM + ΔΔG) + **recurrence** — the ONLY "does it matter?" signals.
  2. **localisation** (experimental complex via `interface_analysis`) — a **mechanism** signal (WHICH interaction
     breaks), conditional on step 1; it sets the propagation MODE but **never** decides significance.
  3. **propagation** — localisation decides the injection: an interface break releases **only that substrate**
     through the signed network (the gene's other arms untouched); a whole-product loss injects −1 on the gene
     node (all edges); a GOF injects +1.
  4/5. phenotype → target.

**Blind test (`colab/reasoning_chain_test.py`) — VHL first, then a 6‑protein panel.** The first design let interface
localisation *rescue* a weak ESM call (treat "at an interface" as evidence the variant matters).
- **VHL alone looked null.** Like‑for‑like (pathogenic vs benign, both in the folded domain), interface membership
  was identical: pathogenic 34% vs benign 33% (Fisher p=1.0); ESM in‑domain AUC 0.50. An early apparent positive
  (87% vs 76%) was a confound — in‑domain pathogenic vs out‑of‑domain benign (VHL's disordered N‑term).
- **A 6‑protein panel corrected the over‑generalisation.** VHL, HRAS, TP53, BRCA1, MSH2, RB1 (numbering‑verified;
  SMAD4 auto‑excluded at 3% match), pooled with **Cochran‑Mantel‑Haenszel stratified by protein** (436 path / 262
  benign):
  - **DIRECT** (all in‑domain): pathogenic **ENRICHED at interface, OR 2.38, p=0.0018 — SIGNIFICANT.** Interface
    *is* a real pathogenicity signal; VHL (all‑interface adaptor, OR~1.2) was an outlier, not the rule.
  - **RESCUE** (the increment among the variants ESM *misses*): OR 1.85, **p=0.14 — not significant** (n=64). The
    interface signal is largely **redundant with ESM**; the part ESM misses is not reliably recovered by interface.

**Design decision (committed):** significance = **ESM + ΔΔG + recurrence only**; interface localisation is a
**mechanism** signal plus at most a *weak conditional lean*, never an independent rescue — because the only thing
that would justify a rescue (signal *beyond* ESM) is not significant (p=0.14). A variant ESM/ΔΔG miss but at a known
interface → **low‑confidence conditional hypothesis** ("IF pathogenic, mechanism = X interface loss"), not a call.
This is well‑calibrated: OR 1.8 / p 0.14 is exactly "weak lean, not a confident call."

**Demonstration (same gene, two mutations, two honest chains):**
- **VHL R167W** — ESM catches it (fs 1.74), not at any partner interface → confident **whole‑product LOF**
  (loses O₂‑dependent proline hydroxylation / RHOBTB3). `mode=whole-product`, conf 0.8.
- **VHL Y115H** — ESM/ΔΔG neutral; sits 3.2 Å from HIF‑1α in 1LM8 → **significance‑uncertain**: cannot confirm it
  matters, but IF pathogenic the mechanism is HIF‑interface loss. `mode=significance-uncertain`, conf 0.3.

Surviving validated value: **mechanism‑conditioned propagation** + honest abstention. The chain is primarily a
mechanism reasoner; its does‑it‑matter power is ESM/ΔΔG plus a *weak* interface lean (real overall, p=0.002, but
ESM‑redundant so it earns only a conditional hypothesis, not a call).

---

## Session-derived guardrails (added from tested findings)

Two honesty guardrails built from measurements made this session — both fence off regions we *proved* are
unreliable, rather than adding new predictions:

- **`colab/kinetic_confidence.py`** (fixed model, kinetic layer). Tested: in-vitro kcat overestimates in-vivo by
  ~1 order of magnitude (Davidi 2016); our `davidi_*` in-vivo field behaves like a **cross-species proxy**
  (measured-kcat vs it: log-log r=0.22, ~30× offset); enzymes run at ~60% capacity and the kcat→flux mismatch
  spans ~8.6 orders of magnitude. `annotate()` attaches a **log10 σ / fold-uncertainty** (median **8.7×**) and a
  provenance caveat to every kcat entry (cross-species proxy flagged on 581), so no kcat-derived rate masquerades
  as precise.
- **`colab/ml_guardrail.py`** (ML model, signal_combiner). Tested: the learned edge-scorer **hallucinates on
  hubs** (TP53 → mitochondrial-ribosomal "partners"; true partners ranked below random). `ml_reliability()` flags
  any gene above the **p90 PPI degree** (threshold 64) as a hub where the combiner is untrusted → defer to
  curated + structural lenses. Correctly flags TP53 (219), MRPS25 (117), EGFR (144); passes non-hubs (BRAF, VHL).

Not added (tested, unreliable): **ESM off-target similarity** — auROC 0.85 but ranked imatinib's *primary* target
ABL1 11th and missed LCK, because ESM captures kinase-family similarity, not the DFG-out binding-pocket that
decides drug binding. It is a coarse family filter, not a predictor, so it is deliberately excluded.

---

## ChIP-seq in ML training + joined enzyme records (session build)

**ChIP-seq → measured regulatory edges** (`colab/chip_edges.py`, `outputs/orphan/chip_reg_edges.json`). Pulled
ENCODE TF ChIP-seq for **12 HepG2 liver TFs**, assigned peaks to genes (±5kb TSS) → **68,233 measured TF→target
edges**. Result of the experiment ("where does it get us"):
- **Coverage win:** **97% of the ChIP edges are NEW** (vs ~57k curated) — 12 TFs nearly double the regulatory map
  with *measured* edges. Scaling to all ~1,600 TFs would add vastly more.
- **But no discovery:** blind-holdout gives overall CN-AUC **0.824** — yet **96% of held-out edges are EASY**
  (hub-dense: TFs bind thousands of genes, so interpolation is trivial). On the genuine HARD/discovery set (4%),
  **AUC 0.15 (below chance)**. ChIP-seq makes interpolation denser; it does **not** enable discovery of novel
  regulation.
- **Honest flags on the artifact:** measured **BINDING, not validated regulation**; **HepG2-specific**; TSS-
  proximity assignment **over-calls** targets. Provided as a flagged binding prior, not universal signed edges.

**Joined enzyme records** (`colab/enzyme_record.py`, `outputs/orphan/enzyme_records.json`). One record per enzyme
(2,549) joining: uniprot, PPI partners (93%), reaction metabolites (100%), **in-vitro kcat (100%, but only 16%
MEASURED)**, **in-vivo kcat (23%, flagged cross-species proxy)**, the **in-vitro/in-vivo ratio + honest ~8.7×
fold-uncertainty** (via `kinetic_confidence`), Km (58%), pathway membership (91%), and in-cell operating rate
(11%). Honest gaps stated inline per record: true *pathway flux* is not measured (only per-enzyme in-cell rate for
~270); most kcat is predicted; the in-vivo value is a proxy.

## Does a structural feature lift DISCOVERY? (domain compatibility, confound-controlled)

Question tested: *"can we add PPI structural features to the existing combiner to make it stronger?"* — specifically,
does a structure-derived feature move the **HARD / discovery** AUC (endpoints share ZERO train neighbours; topology
useless) off its 0.50 chance line? Feasible, leak-CONTROLLABLE structural signal here = **domain–domain interaction
compatibility** (intrinsic domain architecture, orthogonal to the graph). ESM interface-complementarity needs a
predicted 3D complex per pair (no GPU here) and leaks if only done for solved co-complexes, so domains are the honest
proxy.

**Test** (`colab/struct_discovery_test.py`, `outputs/orphan/struct_discovery_test.json`) — the established discovery
holdout, with two confound controls the older `domain_ppi.py` lacked: **matched negatives** (same anchor, decoy
matched on #domains + degree) and an **n_domains-only baseline**. Enrichment learned on TRAIN positives only.

| HARD / zero-shared held-out edges | AUC (mean of 3 seeds) |
|---|---|
| topology + coessentiality (the current discovery score) | **0.50** (chance) |
| + domain compatibility, vs **random** negatives | 0.62 |
| + domain compatibility, vs **matched** decoys (#domains + degree) | **0.61** |
| n_domains alone, vs matched decoys | 0.50 (controlled) |

**Finding:** domain compatibility is a **real but WEAK** discovery signal — it survives the confound controls
(0.61 vs matched, stable across seeds), so it is *not* protein-complexity/hubness. It corrects the older
`domain_ppi.py` headline (0.856 on triadic-blind): that number was inflated by **random negatives** (which lack
complementary domains and are trivially separable). Honest discovery lift from structure = **+0.10 AUC over
topology's 0.50**, not the +0.35 the random-negative test implied. Residual paralog leakage means even 0.61 is a
soft ceiling.

**Integration** (`colab/signal_combiner.py`): `domain_compat` added as a live combiner feature. Two correctness
points baked in:
- **Out-of-fold target encoding.** Domain enrichment is target-encoding; fitting it on train positives and scoring
  those same rows inflated them and the GBM overfit → test AUC craters (measured: 0.796 → 0.72, Brier 0.17 → 0.37).
  Fixed with OOF within train (each row scored by enrichment that never saw its label) + all-train-fit for the test
  rows. After the fix: **0.796 → 0.801 aggregate (no regression), Brier 0.1725 → 0.1713**, and `domain_compat` is
  kept by add→measure→keep.
- **Deployment vs validation leak boundary.** Validation enrichment = train-positives only (honest AUC); deployed
  enrichment = ALL known edges (correct — scoring a novel pair should use all known biology). Consumer
  (`phase2_loop`) verified: rebuilt `features_list` matches the saved model, known interactors still score high.

**Honest bottom line:** yes, a structural feature genuinely strengthens the model — but on the DISCOVERY regime
(chance → 0.61), not the aggregate (which is easy-edge-dominated, so it barely moves, +0.005). The value is
concentrated exactly where the mission is (novel edges with no network shortcut), and it was added without leakage
or regression.

## Does a TRUE interface feature (AF-Multimer) beat domains on discovery? (fetched, no GPU) — NO

Follow-up to the domain result (0.61 on discovery): would a real structural **interface** feature push it toward
0.70? Rather than generate, we **fetched** the Predictomes human AF-Multimer screen (Schmid & Walter, Mol Cell 2025)
— per-pair scores for **1.6M pairs** (28 MB), of which **1.28M map to our genes**. Feature = `num_unique_contacts`
(raw interface contacts from the predicted complex; purely structural, so — unlike their SPOC score — not circular
against our DB-sourced edges). `colab/interface_discovery_test.py`.

**The confound that decides it: AlphaFold was trained on the PDB.** A pair whose complex is already solved
(`in_pdb=1`) is effectively memorised. Splitting on that:

| Clean labels: gold PPI vs zero-evidence non-edge | contacts AUC (ALL) | contacts AUC (**HARD, 0-shared**) |
|---|---|---|
| positives **in_pdb=1** (AF memorised the co-structure) | 0.82 | **0.82** |
| positives **in_pdb=0** (real, AF had to predict) | 0.53 | **0.50** |

Median contacts: memorised pos **65**, genuine pos **0**, non-edges **0**. So AF-Multimer's headline separation is
**PDB recall, not discovery**. On genuine novel interactions it produces the same (near-zero) interface as
non-interactors — **AUC 0.50 on the discovery split**. Domains (**0.61**) actually *beat* AF-Multimer contacts here.

**Why this is the answer, honestly:** it matches the known limitation — AF-Multimer's PPI signal comes from
paired-MSA co-evolution + PDB precedent, both thin for novel human pairs. The 28 MB fetch settled it without a
minute of GPU.

**One untested variant → Colab GPU notebook** (`colab/make_interface_af_notebook.py` →
`colab/interface_af_multimer.ipynb`, pair set `colab/export_interface_pairs.py` → `interface_pairs.json`).
Predictomes ran AF at *high throughput* (reduced MSA, raw contacts). The notebook re-tests the only thing that
could differ: a **careful full-MSA run reading ipTM + pDockQ**, with **templates DISABLED** (so AF cannot retrieve
the co-structure — prediction, not recall), on 80 in_pdb=0 real PPIs + 80 degree-matched zero-evidence non-edges.
Expectation stated in the notebook: near chance. If ipTM ≈ 0.5–0.6, structure does not beat domains for discovery
and the branch closes honestly; ESM interface-conservation is expected to behave the same (same co-evolution
dependence), so it was not pursued separately.

**Net:** the cheap, honest discovery feature remains **domain compatibility (0.61)**, already integrated. A
fetched, PDB-trained interface model does **not** add discovery signal; the GPU notebook is provided to confirm the
last variant, not because a win is expected.

## 2026-07-11 — Today: what actually helps (three directions), with the evidence that pointed there

Tested whether structure/interface features can beat the discovery ceiling, then pivoted to interventional data.
Measured today:
- **Domain compatibility** — real but WEAK discovery signal: 0.50 (topology) → **0.61** AUC on the HARD/zero-shared
  split, confound-controlled (matched negatives), stable over 3 seeds. Integrated into the combiner (out-of-fold, no
  regression).
- **AF-Multimer interface** (fetched Predictomes screen, 1.6M pairs, no GPU) — **chance (0.50)** on genuine
  discovery; the eye-catching 0.82 was **PDB memorisation** (in_pdb=1 pairs AF was trained on). Median interface
  contacts for real-but-unsolved interactions = 0, same as non-interactors. GPU notebook built to confirm the one
  untested variant (careful full-MSA ipTM, templates off); expectation = near chance.
- **Interventional prioritizer** (Replogle Perturb-seq, cross-cell-line identity recovery — the honest non-circular
  test) — the existing cosine cause-finder lands the true gene in the **top-10 only ~21%** of the time. That is
  **44× better than random** (0.005), so interventional data genuinely carries signal correlational data lacked —
  but it is not yet "good." (Paused mid-improvement: decomposing self-knockdown vs trans-network signal.)

**Conclusion — the three directions that actually move the needle** (a better *predictor* does not; discovery from
correlational data is now measured at chance across topology / ML / domains / AF structure):
1. **Interventional data** — the only input with genuine causal (not correlational) signal. Perturb-seq / CRISPR
   screens as directed causal edges. Being built now (perturb_prioritizer.py).
2. **Repurpose oracle → experiment-prioritizer** over the calibrated uncertainty we already have: rank the few
   experiments that most reduce map uncertainty — put the answer in the top-10, not top-100, and name the
   disambiguating experiment.
3. **Deepen where we actually win** — mechanism annotation of KNOWN biology (the reasoning chain reproducing
   TP53/VHL/HRAS residue-by-residue) + the honesty/calibration layer (guardrails, abstention, multi-lens conflict).

**Proven not to help, so stop paying for it:** more correlational features, bigger ML models, better structure
prediction — all saturate at interpolation, chance on discovery.

## CellOS — the cell as an operating system (colab/cellos.py)

Took "biology is software" literally (per the DNA-malware article: DNA is untrusted input; you cannot READ the
source and know behaviour — you must RUN and POKE it). Built a bootable kernel over the real ~16.5k-gene model:
genes=processes, essentiality=scheduler priority, TFs=the scheduler, compartment=memory segment, causal edges=
control flow, CRISPR knockout=`kill -9`, Perturb-seq=the debugger, synthetic-lethal=deadlock pairs, mutation=code
patch, our confound audits=the security layer. Every syscall is backed by real data; causal ones ([C]) use
INTERVENTIONAL data, correlational ones are flagged [~].

Syscalls (all runnable, `python3 colab/cellos.py --demo`): `top` (kernel threads = most essential), `man`
(process docs), `strace GENE` (SIGKILL → MEASURED downstream effect from Perturb-seq), `whodunit GENE` (recover the
implicated module from a knockout's fingerprint — SF3B1 → the whole spliceosome), `diagnose up/down` (root-cause:
whose knockout reverses a state), `deadlock GENE` (SL partners — FANCI → the Fanconi pathway), `patch GENE:MUT`
(static-lint a mutation), `lint "CLAIM"` (the security layer: TRUSTED if curated, UNTRUSTED if a novel prediction
in the 0.5-AUC regime, flags hub-leak). Honest throughout: `whodunit` excludes the trivial self-match and cites the
measured ~21% cross-context top-10 ceiling; a sign bug (ranking by reversal instead of +cosine match put the true
cause LAST) was caught and fixed. This reframes the whole project as one coherent system: the debugger (interventional
data) is the crown jewel, the linter (honesty layer) is the security boundary, everything correlational is flagged.

## CellFormer — "predict the next thing" (transformer-style completion on interventional data)

A transformer completes a sequence by predicting the next token from context. The cell-equivalent, built on the
Replogle Perturb-seq corpus (2,058 knockouts × 8,563 genes): predict the transcriptional effect of a knockout we
NEVER measured, from the measured effects of its network neighbours (`colab/cellformer.py`). Two regimes, decomposed
like everything else:
- **IMPUTE (interpolation):** mask genes inside a held-out perturbation, predict from the observed genes via
  gene-gene attention (softmax over SVD gene embeddings) → **r = 0.52** vs 0.24 predict-mean baseline. The cell
  autocompletes masked genes well — exactly what Geneformer/scGPT do.
- **PREDICT-NEXT (extrapolation — the real "next token"):** hold out an ENTIRE knockout, predict its full response
  from its network neighbours' responses → **r = 0.34** overall, but **r = 0.43 for genes in a known complex**
  (n=1,110) vs **0.19 for singletons**; beats baseline for 65% of genes. So an unseen knockout is predictable
  EXACTLY WHEN the gene sits in a measured module; singletons stay hard. Honest, sharp boundary.

Wired into CellOS as `predict GENE` [C~] (causal source, predicted value). `predict PSMB5` → predicted UP =
HSPA1A / UBC / SQSTM1 (the textbook proteasome-inhibition proteostasis-stress response), with a live self-check:
predicted-vs-real **r = +0.86**. Tested (cellos_test.py): the self-check and the stress-response recovery both
assert-pass. This is the transformer paradigm made real on the cell — and honest about where completion stops
working (no module → no prediction).

## Genome-wide scale-up + the "complete cell" completeness number

Scaled CellFormer/CellOS from the 2,058-gene essential screen to the **genome-wide Replogle screen (9,867
knockouts × 8,248 genes, 375 MB)** and added a COMPLETENESS metric — the fraction of the ~16,509-gene genome CellOS
can answer for. The honest, tiered result (coverage ≠ accuracy):

| tier | genes | how |
|---|---|---|
| **MEASURED** (debugger, high quality) | 8,693 (53%) | directly in the genome-wide screen |
| **PREDICTABLE via a complex** (r≈0.23) | 426 (3%) | cellformer predict-next from complex partners |
| weak context only (r≈0.05) | 5,121 (31%) | has neighbours but no module |
| dark (no context) | 2,269 (14%) | nothing to predict from |

**→ CellOS ANSWERS for 86% of the genome, but only 55% is TRUSTWORTHY** (measured + complex-predictable). Stated
that way on purpose: coverage is not accuracy.

Measured coverage↔precision tradeoff: the genome-wide screen (many weak non-essential knockouts) is NOISIER than
the essential screen. On genome-wide, IMPUTE r=0.39 (vs 0.52 essential), PREDICT-NEXT r=0.10 overall / 0.23 for
complex genes (vs 0.43 essential), whodunit coherence 4% but still **17.5× random**, `predict PSMB5` self-check
r=0.30 (vs 0.86 essential). So CellOS DEFAULTS the interactive debugger to the high-precision essential screen
(syscalls stay sharp — SF3B1→spliceosome 8/8, PSMB5 r=0.86); mount the genome-wide screen for breadth
(`PERTURB_H5AD=…/gwps.h5ad`). Fixed a real bug found here: `_load_debugger` didn't sanitise the gwps NaN/inf
entries, so every genome-wide syscall computed with NaN (predict r=nan) until cleaned.

Net "complete cell as software": ~half the genome is directly measured cause-and-effect, another few % confidently
predictable, ~a third gesturable-but-weak, ~1/7 dark — and CellOS says which tier every answer is in.

## Inspiration from The Matrix: "there is no spoon" + "free the cell"

The Matrix's real idea maps onto this project: a cell is a model of reality you can only wield once you see its code
AND can bend it. We had "see the code" (CellOS); added "bend it":
- **`simulate G1 G2 …`** [C~] — "there is no spoon": edit the cell (knock out one or more genes) and propagate to
  the resulting state by combining measured effects. A single KO is MEASURED (exact); a combination is the additive
  sum — flagged, because the error IS the genetic interaction (epistasis), which single-perturbation data can't
  measure. `simulate SF3B1 PSMB5` → amplified proteostasis stress (HSPA1A +5.3, UBC, SQSTM1), the honest sum of a
  splicing + proteasome knockout.
- **`cure up=… down=…`** [C] — "free the cell": given a corrupted/disease state, greedily search the knockout that
  best reverses it, then a complementary second — a combination-therapy search. For a MYC/CCND1-driven proliferative
  state it prescribes co-knockout of **CSNK2B + NAE1** (CK2 and the NEDD8 E1 — the latter is the target of the drug
  pevonedistat); reversal values are honestly small (the essential screen is an imperfect context match), but the
  2nd target adds measurable gain.

Both tested (cellos_test): single-KO simulate is exact (== strace), combos are additive-and-flagged, cure returns a
combination prescription. Same discipline as everything else — the reality-bending is real for measured single
perturbations and an honestly-labelled approximation for combinations.

## Epistasis: does the additive combo model hold? (Norman 2019 real doubles)

Tested `simulate`/`cure`'s additive assumption against MEASURED double perturbations — Norman 2019 CRISPRa
genetic-interaction map (K562: ~100 singles + 131 doubles), pseudobulked from 111k cells (`colab/epistasis.py`).
For each double A+B, compared additive (delta_A + delta_B) to the real delta_AB.

- **Additive predicts the real double at r = 0.884** — a strong FIRST-ORDER model (it gets the direction right).
- It adds only **+0.05 over the best single** (r 0.83 → 0.88): the double is dominated by the stronger single.
- **But mean epistasis residual = 54% of the double's magnitude**, and **47/131 pairs (36%) are strongly
  non-additive** (>50% residual). So additive captures the PATTERN but misses the MAGNITUDE/synergy — and synergy
  is exactly what combination therapy exploits.
- Top synergy pairs are real biology: **DUSP9+MAPK1** (phosphatase + its ERK kinase), **CEBPE+KLF1** (two
  differentiation TFs). Synergy concentrates around driver hubs (MAPK1, CEBPE, OSR2 recur) — but is **NOT
  predictable from our static PPI/pathway annotations** (most synergistic pairs score `related=False`): it needs the
  measured double.

**Consequence, wired in honestly:** `simulate`/`cure` now flag every combination — "additive gets direction right
(r=0.88) but misses ~54% magnitude; 36% of pairs strongly synergistic; trust the pattern, not the scale; confirm
synergy with a measured double." So the reality-bending is a real first-order screen, honestly bounded: exact for
single perturbations, directional-but-not-quantitative for combinations, and synergy remains a measure-it problem.

## `stat` — whole-cell completeness dashboard

Added a `stat`/`df` syscall to CellOS: a df/htop-style dashboard of how complete the cell model is, layer by layer
(all counts real, from the model). Snapshot (16,509 genes):
- ANNOTATION: localization/role/domains/LOEUF 100%, GO 99%, pathway membership 63%, PTM 51%, dark (no known
  function) 30%.
- NETWORK: 86% have a PPI partner; edges = PPI 191k, regulatory 612k, signaling 17k, causal(directed) 60k; SL
  pairs 1,256; ligand-receptor 948; co-expression 99%, co-dependency 81%.
- MODULES: 2,792 Reactome pathways (cover 64% of genes), 2,039 complexes (3,257 genes), 2,549 metabolic enzymes.
- QUANTITATIVE/PHARMA: protein abundance 97%, cell-type expression 45%, 4,275 drugs.
- INTERVENTIONAL (the debugger): 2,058 measured on the default essential screen (12%); genome-wide → 8,693
  measured + 426 well-predictable + 5,121 weak + 2,269 dark = ANSWERS for 86%, TRUSTWORTHY for 55%.

So "the complete cell" has near-total *descriptive* coverage (what each gene is, its pathways/complexes/edges) but
the *causal/interventional* layer — the part that actually predicts what happens when you poke it — is the honest
frontier: 55% trustworthy, and that's the number that matters.

## Trying to raise the 55%: hu.MAP complexes REJECTED (validated, not counted)

To lift the trustworthy-coverage number (55%), fetched two levers: richer complexes (hu.MAP 2.0 — 6,333 physical
complexes, 8,671 genes) and combined interventional screens (gwps+RPE1+essential).

- **Interventional union barely moved** measured (8,693 → 8,696): the screens are all K562/RPE1 and overlap; gwps is
  already genome-wide, so that lever is tapped.
- **hu.MAP raised the COUNT 55% → 67%** — and then the honest re-validation killed it. Leave-one-out predicting a
  measured gene's knockout effect from its complex partners: **curated complexes r=0.233, but hu.MAP r≈0.09 at EVERY
  confidence tier** (even conf 5 = 0.087). The ~4,000 hu.MAP-added "complex" genes predict at r≈0.06 — the weak tier.
  Reason: hu.MAP is *physical co-purification*; two proteins that stick together don't share a knockout effect
  unless they're a *functional* unit — which curated complexes already capture. So the 67% was relabeling weak
  genes as trustworthy. **Rejected**; coverage stays the honest **55%** (`outputs/orphan/humap_validation.json`).

Lesson (same as the whole project): adding annotation data raised the metric but not the capability, caught by
re-validating (predict r) instead of counting membership. The real lever for 55% isn't more complex annotation —
it's more INTERVENTIONAL measurement, and the readily-available Perturb-seq screens are already folded in.

## Can the fixed model's data raise the 55%? Tested every edge type — NO (only complexes predict)

Reasonable idea: the fixed cell model has 90-95% descriptive coverage (PPI 191k, regulatory 612k, causal 60k,
signaling, pathways, co-expression 99%, co-dependency 81%) — use it as prediction context to raise trustworthy
coverage. Tested each edge type by leave-one-out (predict a measured gene's knockout effect from its neighbours of
that type; r vs real):

| relationship | genes | predict r |
|---|---|---|
| **complex (curated)** | 2,261 | **0.233** |
| co-dependency | 7,149 | 0.094 |
| PPI | 8,045 | 0.091 |
| pathway (Reactome) | 5,893 | 0.064 |
| co-expression | 8,524 | 0.063 |
| regulatory (612k edges) | 8,640 | 0.034 |
| causal (directed) | 4,602 | 0.032 |
| signaling | 3,185 | 0.030 |

**None predict knockout effects.** The layers that cover ~half the genome (regulatory, coexpr, PPI, pathway) sit at
r=0.03-0.09 — the weak-tier floor. Only tightly-curated functional complexes predict (0.23), and they're already
used. The reason is the project's through-line: **descriptive relationship ≠ causal effect similarity.** Two genes
can be co-expressed, PPI-linked, in the same pathway, even regulate each other, and still have completely different
knockout consequences. So the fixed model's 90-95% is DESCRIPTIVE completeness; it does not translate into causal
PREDICTION. Trustworthy coverage stays **55%** — raising it needs interventional measurement, not more of the map
we already have. (`outputs/orphan/edge_predict_validation.json`)

## The software analogy, tested directly: the cell is ROBUST, software is BRITTLE

Sharp challenge: in software, delete a component and everything connected to it breaks — so the dependency graph
predicts the blast radius. Does the cell work that way? This is a DIFFERENT test than "do connected genes have
similar effects" (that's interchangeability) — it's PROPAGATION: knock out A, do A's connections move? Tested on
gwps: is a knockout's annotated targets/partners enriched among the genes that actually change (Mann-Whitney AUC)?

| knock out A → do A's connections move? | AUC | in top-10% vs random |
|---|---|---|
| regulatory → targets | **0.499** | 10% vs 10% |
| PPI partners | 0.516 | 12% vs 10% |

**The analogy breaks.** A TF's targets change no more than random genes (0.50); physical partners barely (0.52). The
reason is the deepest point of the whole "biology as software" thread: **the cell is ROBUST by evolutionary design**
— redundancy, feedback, buffering absorb single-gene perturbations, so connected components usually do NOT fall
apart. Software is brittle (no evolved redundancy), so a deleted function reliably breaks its callers. The ONE place
the cell is software-like is a tightly-coupled functional COMPLEX (no redundancy: pull one subunit and the machine
fails) — and complexes are exactly the only relationship that predicts knockout effects (r=0.23). Everywhere else,
robustness severs the wiring-diagram → effect link.

This is the unifying answer to the entire session: you can't predict a knockout's blast radius from the descriptive
graph because the cell evolved specifically to survive having its components deleted. That is *why* it must be
MEASURED (the debugger), not read (the map). (`outputs/orphan/propagation_test.json`)

## The paradigm shift that works: prioritise survival+reproduction under physics (FBA), AUC 0.70

The whole session showed BOTTOM-UP fails on robustness: reading the wiring predicts a knockout's blast radius at
chance (propagation AUC 0.50; edge-predict r 0.03-0.09). The fix is TOP-DOWN — don't read the wiring, give the model
the objective real cells have (stay alive + REPRODUCE) under the constraints they obey (mass/energy conservation +
enzyme capacity = physics limits) and ask "can it still hit the objective after I break this part?" This is
constraint-based modelling / FBA, and it was already in the repo (`ecflux.py`, Human-GEM, cobra).

Ran objective-driven FBA (Human-GEM: maximise biomass s.t. mass balance + capacity) single-gene deletion over 2,848
metabolic genes, vs measured DepMap essentiality:

| predict knockout SURVIVAL | score |
|---|---|
| wiring propagation (bottom-up) | AUC **0.50** (chance) |
| any descriptive edge | r 0.03-0.09 |
| **objective-driven FBA (top-down)** | **AUC 0.70** (dep≥0.9), corr 0.49 |

And it reproduces ROBUSTNESS for free: of 2,848 genes FBA calls only 88 lethal — most knockouts survive because the
model REROUTES flux around them, exactly like the real cell, while still nailing the true bottlenecks. That is the
user's insight, measured: robustness stops being a mystery and becomes a prediction, because it's a *consequence* of
optimising survival under constraints.

Wired into CellOS as `viability GENE` [FBA]: RAE1 -> LETHAL (0% growth, unreroutable bottleneck); SLC22A1 -> SURVIVES
(100%, flux reroutes); TP53 -> honestly out of scope (no biomass-flux objective for signalling/TFs). The top-down
complement to strace's bottom-up. Honest scope: metabolism only (~2,800 genes) — the part of the cell governed by a
conservation law; there's no clean physical objective for signalling/regulation yet. (`fba_essentiality.json`)

## The fitness objective hierarchy: survive → grow → compete → reproduce (what's captured, what's the frontier)

Extending the objective past "maximise biomass". Mapping the biological lifecycle onto what the model actually
optimises, honestly:
- **Survive** = a feasible flux state that meets maintenance ATP (don't die). Captured (FBA feasibility).
- **Grow** = maximise biomass. Captured — objective-driven FBA predicts knockout survival AUC 0.70.
- **Compete** = optimise for RATE over YIELD (grow fast even if wasteful) + don't waste (parsimony/pFBA = min flux
  among max-growth). PARTIALLY captured: the growth-max solution on Human-GEM is fermentative — it secretes lactate
  and skips respiration even with O2 available (the **Warburg effect**, the cancer signature), i.e. the model
  already picks the compete/grow-fast strategy over the efficient one. HONEST CAVEAT: simple glucose/O2 bound
  sweeps did NOT produce a clean rate-yield transition (biomass is capped by other media bounds); a proper
  rate-yield frontier needs the enzyme-constrained ecModel (proteome allocation), not shown here.
- **Reproduce (as fitness)** = long-term reproductive success across time, fluctuating environments, and against
  actual competitors. This is the FRONTIER — beyond steady-state FBA. It needs dynamic FBA (time), community FBA
  (competitors), or evolutionary/agent-based dynamics, and it answers DIFFERENT questions (population heterogeneity,
  resistance evolution, bet-hedging) than the per-gene survival prediction we validated.

Key point: each objective term predicts a different SLICE. survive+grow -> knockout lethality (AUC 0.70).
compete (rate>yield) -> cancer's Warburg metabolism. reproduce/evolve -> drug-resistance evolution & heterogeneity.
Adding evolutionary fitness would NOT necessarily improve per-gene survival prediction; it opens a different layer.

## Tried more Perturb-seq screens: +19 genes, 55% ceiling is data-generation-limited

To raise the 55% by measurement, combined every readily-available Perturb-seq screen (surveyed the full scPerturb
catalog, 50 datasets):

| screen | perturbs (model genes) | new |
|---|---|---|
| K562 genome-wide (gwps) | 8,693 | — |
| + RPE1 | 2,155 | +2 |
| + K562 essential | 1,854 | +1 |
| + Norman 2019 | 94 | +16 |
| **combined** | **8,712 (53%)** | trustworthy **55%** (unchanged) |

**+19 genes total** — the number doesn't move. Why: all the large screens target the same K562-expressed /
essential-gene space gwps already covers. The only catalog entry that could add many new genes — Joung 2023 TF
atlas (~1,700 TFs) — is OVEREXPRESSION (wrong modality for the knockdown debugger) and 5.8GB (throttled download,
abandoned). The real barrier: the ~7,800 unmeasured genes are mostly NOT EXPRESSED in K562, so no K562 screen can
reach them, and no public GENOME-SCALE KNOCKOUT Perturb-seq exists in a complementary cell type. So the 55% ceiling
is **data-generation-limited** (needs new wet-lab screens in other cell types), not fetch-limited — confirmed
empirically, not assumed. (`outputs/orphan/perturb_screens_combined.json`)

## Extract individual gene knowledge from PubMed (litmine + `lit` syscall)

The screens can't reach the ~7,800 genes K562 doesn't express — but 2,540 of the model's DARK genes still have ≥5
PubMed papers. That focused single-gene literature is the one source of causal/functional knowledge for them.
`colab/litmine.py` mines it, GROUNDED in real abstracts with DOIs (NCBI E-utilities: search biased to
function/knockout/regulates/mechanism, fetch abstracts). Demo batch of 8 dark genes (FNDC5, NPPB, POSTN, BMAL1, …)
→ `outputs/orphan/litmine.json`.

Example — FNDC5, which the model calls DARK, is richly known to the field (per PubMed): cleaved/secreted as the
myokine irisin, induced by exercise via PGC-1α; drives browning of white adipose tissue; induces hippocampal BDNF
→ cognitive benefit, therapeutic potential in Alzheimer's (Wrann 2013 doi:10.1016/j.cmet.2013.09.008; Islam 2021
doi:10.1038/s42255-021-00438-z; Boström 2012 doi:10.1038/nature10777).

Wired into CellOS as `lit GENE` [LIT] — the "read the source papers" syscall (grounded titles + DOIs, flags DARK
genes). HONEST scope, stated in the output: this is a DESCRIPTIVE/QUALITATIVE layer (function + citations) that
fills the model's annotation gaps from literature; it does NOT raise the quantitative causal-prediction number
(that needs measured perturbation signatures). The fetch is deterministic/reproducible; turning abstracts into
structured facts is an LLM step done downstream, auditable against the stored source papers. Tests green.

## Scale the dark-proteome mine (litmine `--dark`, full run)

Scaled the miner from 8 demo genes to the ENTIRE literature-rich dark proteome: all 4,765 model-dark genes that
carry ≥5 prior PubMed papers, mined end-to-end (resumable, saves every 25, one retry on transient failure).
Result (`outputs/orphan/litmine.json`, 36 MB): **4,670 / 4,765 genes (98%) retrieved ≥1 causal/functional paper**,
95 returned none, 0 fetch errors; **25,934 papers total (5.4/gene), 25,682 with DOIs** — every fact auditable.
Same honest scope: closes the descriptive/annotation gap for genes no screen can reach; does NOT move the 55%
causal number.

## The cell as RUNNING software — `boot` syscall + the honest dynamical null (grn.py)

Everything before this was either a photograph (measured snapshot) or a destination without a journey (FBA solves
the steady operating point directly). `colab/grn.py` adds the one piece with a **clock**: a continuous-state
recurrent map over the 7,480 genes that have regulators (54k signed SIGNOR/CollecTRI edges), row-normalized (hub
control) with a negative threshold that fixes activation-domination (the edge set is 4.5:1 activating, so without a
threshold everything trivially turns on). It **boots from an initial state and flows to an attractor** — genuinely
*executes* the genome forward. Wired into CellOS as `boot [GENE]` (aliases `exec`/`run`).

Blind validation (`colab/grn_validate.py`; only the ON-fraction operating point was calibrated — nothing tuned to
the answer):

| test | result | verdict |
|---|---|---|
| T1 convergence | 40/40 random starts settle in ~30 ticks | a real, stable dynamical system |
| T2 non-triviality | attractor 28% ON (not all-on/all-off) | the threshold beat activation-domination |
| T3 robustness | 68% of single knockouts barely move the attractor | **matches measured biology** (cells are robust) |
| T4 essentiality (blind) | fragility→essential **AUC 0.47** vs out-degree baseline 0.47; partial ρ 0 | **NULL** |

The null is **gain-robust** (`grn_diag.py`, AUC 0.474/0.470/0.464 across β=4/8/15 — always ≈ the hub baseline), so
it is not a mushy-dynamics artifact. And it is not a surprise: it is the **dynamical confirmation** of a result we
already measured statically (`edge_predict_validation.json`: regulatory edges don't predict knockouts; only
physical complexes do, r=0.23). **You cannot get more causal signal out of running a network than its edges
contain.**

Honest verdict, stated in the `boot` output itself: the cell genuinely runs, converges, and is robust — a real
"software executing" — but running it adds **no** causal/predictive power over chance/connectivity. Committed as a
validated demonstration (you can watch a cell-state emerge, and watch it re-settle after a knockout — TP53
displaces the attractor 2.9 and flips 133 genes, a leaf gene barely moves it) plus a documented null. The layers
that DO predict knockouts stay the physical/measured ones: `viability` (FBA, AUC 0.70) and `strace` (the debugger).

## Reprogram the running cell with transcription factors — `induce` syscall (the forward win)

The essentiality null was the BACKWARD question ("remove a gene → what breaks"), and running the network can't
answer it. TFs are built for the FORWARD question, so we asked that instead: force a master TF ON in the resting
cell, run the program forward, and does the cell flow to that TF's real lineage — SPECIFICALLY? (`grn_reprogram.py`)

9 master TFs with textbook marker programs (erythroid/myeloid/muscle/pluripotency/B-cell/hepatocyte/p53/Treg/HSC),
programs defined independently of the edge list, forced TF excluded, averaged over 5 resting states:

| metric | result |
|---|---|
| reprogramming hit@1 (own program most-induced) | **8 / 9** (only SPI1 loses, to PAX5 — both lympho-myeloid) |
| matched-vs-mismatched induction AUC | **0.988** |
| median rank of own program (of 9) | **1** |

Forcing GATA1 on lights up erythroid (SLC25A37, RHAG, HEMGN, ALAS2, KLF1) and almost nothing else; MYOD1→muscle
(CKMT2, MUSTN1, CHRNG); POU5F1→pluripotency (UTF1, ZSCAN10, ESRG, EED); FOXP3→Treg (IKZF2, PTPN22). Wired into
CellOS as `induce TF [TF..]` (alias `reprogram`).

HONEST scope (disclosed in the output): this is **not** emergent discovery. The textbook markers are mostly DIRECT
edge-targets of the TF (e.g. GATA1 6/7, HNF4A 6/6), and the indirect-only markers are too few (8 genes, AUC 0.67)
to claim the dynamics compute new downstream biology. What it DOES show, cleanly: the curated regulatory wiring
encodes **correct, lineage-specific** master-TF→program logic, and the running dynamics faithfully express it when
you flip a master switch. It is master-regulator readback played forward.

**The dichotomy (the real finding).** Same running cell, two directions:
- **backward** (knock a gene out → what breaks): NULL — the cell is robust, fragility isn't in the wiring (AUC 0.47).
- **forward** (flip a master switch → what turns on): REAL — the wiring is right and the cell reprograms to the
  correct lineage, specifically (AUC 0.99).

Which is exactly what you'd expect from how the regulatory genome was charted: master-regulator→target maps are its
best-known part; global knockout-fragility is not written in the edges at all. `induce` (forward, works) and
`viability`/`strace` (backward, measured) are the two honest ways to ask; `boot`+backward-fragility is the one that
politely returns "chance." Tests green.

## Whole-software audit — one command, five sections, 13/13 (cellos_full_audit.py)

Before calling it done, a single end-to-end pass over the live product (not "does a file exist" — "does it boot,
recover known biology, reproduce its headline numbers, and is every claim backed by a committed artifact"):

| section | result |
|---|---|
| 1 import audit (each module isolated in a subprocess) | runtime-critical **13/13**; historical **99/101** (2 run-at-import, 0 real breakage) |
| 2 CellOS integration (every syscall + correctness + coherence) | ALL PASS; whodunit coherence 25% vs 1% random (24×) |
| 3 running-cell layer | boot converges (32 ticks, 28% ON); reprogram **8/9, AUC 0.99**; essentiality **null 0.47** (null-by-design = pass) |
| 4 claims ↔ artifacts | FBA AUC 0.68, reprogram 8/9 / 0.988, essentiality null gain-robust, litmine 4,670 genes, epistasis r 0.884 — all backed |
| 5 artifact integrity | 60 present artifacts all parse as valid JSON |

**13/13 checks pass** — the whole software boots, recovers known biology, and every headline number a syscall
prints is traceable to the committed evidence that earned it. `python3 colab/cellos_full_audit.py` reproduces it.

## Combine top-down + bottom-up so they help each other — `assess` syscall (the coverage synergy)

The critique of a physics-first CellOS vision landed on: top-down simulation and bottom-up data hit the same wall
from opposite sides, blind in DIFFERENT places. That difference is exploitable. Tested it (`cellos_synthesis.py`):

- **TOP-DOWN = FBA** (physics): predicts essentiality for metabolic genes even when silent in the assay. Blind to
  signalling/TFs. AUC **0.81 on the 2,250 genes only it reaches**.
- **BOTTOM-UP = measured Perturb-seq shock** (a gene whose knockout shocks the transcriptome tends to matter):
  covers genes expressed in the screen, blind to the ~7,800 K562 doesn't express. AUC **0.72 on its screen-only genes**.
- They overlap on only **295 of 4,082** covered genes — almost disjoint.

Routing each gene to its specialist modality (calibrated to a real P(essential) out-of-fold, then combined):
**coverage 2,526 → 4,082 genes (+62%) at effective per-gene accuracy 0.77, above the best single modality (0.71).**
Broader AND slightly sharper — the first thing in the project to move the coverage/accuracy frontier, because it
is not more of the same data, it is two modalities reaching different blind spots.

Three honesty guardrails, all in the code/verdict:
- **naive pooling FAILS** (AUC 0.62): the two populations have very different base rates (metabolic 9% essential vs
  the K562-essential screen 72%), so averaging raw ranks miscalibrates. HOW you combine is the whole game.
- the **pooled calibrated AUC (0.93) is base-rate-inflated** — it partly separates genes by which screen they came
  from, not per-gene skill. The honest number is the within-population **effective accuracy 0.77**, not 0.93.
- overlap agreement is weak (r=0.20), so this is **coverage complementarity, not mutual confirmation** — stated plainly.
- (also fixed a tie-handling bug in the AUC: FBA growth ratios have many ties; switched to average-rank Mann-Whitney,
  which made the numbers deterministic.)

Wired into CellOS as `assess GENE`: gathers every independent layer (measured DepMap, top-down FBA, bottom-up
Perturb-seq shock) for one gene, shows each with its evidence grade, and gives a combined call whose CONFIDENCE
rises when independent layers agree (RAE1: all three agree → HIGH; SF3B1: measured-essential but modest shock →
honestly flagged MIXED). This is the physics↔data synthesis made per-gene: they cover for each other where one is
blind, and corroborate where both see. Tests green.

## Top-down simulation that doesn't drift — data checkpoints on the rails (`cellsim` syscall)

The critique landed on: pure top-down simulation drifts (the free-running GRN lands in arbitrary attractors), so
run it but ASSIMILATE measured data at checkpoints — the way weather models stay tethered to observations.
Tested it as a reconstruction task (`cellsim.py`): hold out a knockout's measured response, reveal a fraction K of
its genes as CHECKPOINTS, reconstruct the rest. Two predictors, same checkpoints, same held-out genes:

| checkpoints | MECH sim (regulatory graph) | MECH fair (only genes w/ a regulator) | DATA fill (measured co-response) | baseline |
|---|---|---|---|---|
| 0% | 0.00 | 0.00 | 0.25 | 0.25 |
| 10% | 0.01 | 0.01 | **0.42** | 0.25 |
| 25% | 0.01 | 0.01 | **0.48** | 0.25 |
| 50% | 0.01 | 0.01 | **0.50** | 0.25 |

**The idea works — but the honest split is sharp.** Data checkpoints genuinely rescue the reconstruction: reveal
10% of a knockout's response and the rest comes back at r=0.42, rising to 0.50 at 50% (baseline 0.25). But the
mechanistic regulatory SIMULATION contributes ~nothing (r=0.01), even scored only on the 78% of genes that HAVE a
regulator in the subgraph (so it's not a sparsity artifact) — consistent with the earlier finding that regulatory
edges don't predict knockout effects. The simulation is a scaffold; the measured data checkpoints are the signal.
The sim stays on the rails only because the data holds it there — assimilation, not physics.

Wired as `cellsim GENE`: reveals 20% of a knockout's measured response as checkpoints and reconstructs the rest
(SF3B1 r=0.71, RPL13 r=0.93 — ribosomal responses are highly coherent). Honest label in the output: the data
anchors carry it, the mechanistic sim adds ~nothing. This is the honest form of "top-down simulation with data
helpers" — genuinely useful for reconstructing a partially-measured cell, and clear about WHERE the power comes
from. Tests green.

## Move beyond 55% — honestly — and make the results experienceable (coverage + readout)

"Keep moving beyond 55%" — done without faking the number. `colab/coverage.py` counts EVERY trustworthy answer
the software gives, per gene, across all validated capabilities (not just the one deep axis):

| axis | reach | grade |
|---|---|---|
| essentiality (measured DepMap) | **96.4%** | measured |
| RESPONSE prediction ← the 55% | **55.2%** | measured/modeled — data-limited ceiling, unchanged |
| viability by physics (FBA) | 17.2% | modeled |
| reprogrammable (TF, forward 0.99) | 7.4% | modeled |
| documented (grounded literature) | 98.4% | LIT |
| **≥1 trustworthy answer (union)** | **99.4%** | — |
| truly dark (nothing at all) | **0.6% (91 genes)** | — |

The honest reframe: the 55% is the DEEPEST axis (predict the full knockout response) and it stays data-limited —
but it was never the ceiling of the whole system. Counting every validated capability, the software gives at least
one trustworthy answer for 99.4% of the genome; only 91 genes are truly dark. Wired as `coverage` syscall.

Also shipped an experienceable readout (`viz_cellos_readout.html`, published as an Artifact): an instrument-panel
dashboard of the coverage map, the capability scorecard, the forward/backward dichotomy, and the honest grades —
every number traceable to committed evidence. `stat` still shows the deep-axis 55%; `coverage` shows the whole
reach. Tests + audit green.

## "Get other cell lines" — the honest test (cross_cell_line.py) + why 55% is per-cell-line

Read Pillai/Hochberg/Thornton, "Simple mechanisms for the evolution of protein complexity" (Protein Science 2022):
complex features sit at the genetic EDGE, reachable by many short paths → chance joins selection. Relevant to us as
a theory companion to the discovery-null and robustness results (degeneracy = many-to-one = unpredictable from one
trajectory), but it's an evolution paper, not cell-line data.

The actionable point — the 55% ceiling is single-cell-line (K562). Tested "get another cell line" with the second
screen already on disk (RPE1):
- **RPE1 adds 0 NEW genes.** Its 2,394 perturbed genes are a complete SUBSET of K562's 9,871 — because Replogle's
  RPE1 screen is essential-gene-scoped, and essential genes are housekeeping (expressed in every line). A second
  cell line's ESSENTIAL screen re-measures the same core; only a GENOME-WIDE screen in that line reaches its unique genes.
- **Responses barely transfer across lines.** For the 2,056 knockouts measured in BOTH lines, the same KO's response
  correlates only r=0.19 across K562↔RPE1 (vs 0.06 shuffled — real but weak; 40% clear r>0.2). So a knockout does
  substantially DIFFERENT things in different cell types.

Implication: **"55%" is really "55% in K562."** Other cell lines matter not only for the ~7,800 genes K562 can't
express, but because knockout effects are largely context-specific — the cell is a different machine per context.
The lever is a GENOME-WIDE knockout Perturb-seq in a complementary cell type. That data did NOT exist when the
ceiling was set, but is now emerging: genome-scale Perturb-seq in primary CD4+ T cells (Zhu 2025), pooled CRISPR-KO
in primary myeloid cells (Jung 2025), genome-wide CRISPRi (Bradu 2026). Fetching + integrating one is the real
next step — a heavy multi-GB pull best run in the Colab data pipeline (built for it), not this ephemeral container.
(`outputs/orphan/cross_cell_line.json`)

## The getter for other cell lines (fetch_celltype_screen.py)

Built the mechanism to actually GET a complementary cell line and MEASURE the payoff, parameterized by a
processed-h5ad URL so the heavy pull runs in the Colab pipeline (this container has a small disk allowance):
`--probe` HEAD-checks size vs free disk and refuses a download that won't fit (no bricking the session); `--fetch`
streams it with the guard; `--integrate` parses it into the debugger format and counts the NEW measured genes it
adds against K562 + the model universe. The coverage lift is COUNTED from the real perturbed-gene set, not asserted
— self-checked by integrating RPE1, which correctly reports 0 new genes (8,696 → 8,696), matching cross_cell_line.
When a genome-wide screen in a complementary cell type is provided (primary T-cell / myeloid, 2025), one command
integrates it and reports the honest lift. This is how "get other cell lines" becomes real coverage, not a promise.

## Got Zhu 2025's data (the gene list) and broke 55% — before/after

Fetched the Marson/Zhu 2025 CD4+ T-cell genome-scale Perturb-seq target library from the analysis repo
(emdann/GWT_perturbseq_analysis_2025, `metadata/sgRNA_library_curated.csv`) and integrated the perturbed-gene set.
The raw data is un-fetchable here (every S3 object is 16–170 GB; pseudobulk alone 44.5 GB vs 21 GB free) — but the
COVERAGE question ("how much can we reach") only needs which genes were perturbed, not their responses.

**BEFORE (K562 only) → AFTER (+ CD4+ T-cell screen):**

| axis | before | after | gain |
|---|---|---|---|
| measured response coverage | 8,696 (52.7%) | **11,461 (69.4%)** | **+2,765 genes** |
| trustworthy "55%" axis | 9,122 (55.3%) | **~11,887 (72.0%)** | **+16.7 pts** |

Zhu targeted 12,783 genes (11,039 in our model); 8,274 were already K562-measured; **2,765 are genuinely NEW —
T-cell-expressed genes K562 never reached** (A2M, ABCA1/2/3, ABCB1, immune/transporter genes, …). This is the
FIRST real, non-faked movement of the ceiling, achieved exactly as predicted: new measured data in a complementary
cell type reaching the genes K562 silences. `fetch_celltype_screen.py --genelist` reproduces it from the committed
`zhu_targeted_genes.txt`.

Honest caveats (in `zhu_coverage.json`): (1) this is the TARGETED set — an upper bound; the effectively-knocked-down
set (pseudobulk `keep_effective_guides`) is slightly smaller. (2) COVERAGE is computed; the actual response
PREDICTIONS on the new genes need the 44.5 GB matrix (Colab step, not run here). (3) These are T-cell-CONTEXT
measurements — and given cross-line transfer is only r=0.19, they extend coverage IN the T-cell context: the honest
frame is not "72% of the cell" but "we can now reach ~72% of the genome's knockout responses across K562 + T-cell
contexts." The cell stays contextual; we just measured one more context.

## Stack more cell lines — the honest diminishing-returns curve (coverage_stack.py)

Extended "get more cell lines" to every published Perturb-seq we could get a gene list for WITHOUT the huge
matrices: K562 (on disk) + CD4+ T cell (Zhu 2025 committed list) + ~8 scPerturb screens across NEW lineages
(melanoma FrangiehIzar2021, THP-1 PapalexiSatija2021, iNeuron/iPSC TianKampmann2019/2021, DatlingerBock2017),
pulling each dataset's perturbed genes straight from its `pairwise_edist` table header (scPerturb repo, no download).

Cumulative measured-response coverage:

| stacking | genes | % genome |
|---|---|---|
| K562 alone | 8,696 | 52.7% |
| + CD4+ T cell (genome-wide) | 11,461 | 69.4%  (**+2,765**) |
| + 8 targeted scPerturb screens (5 new lineages) | 11,504 | 69.7%  (**+43 total**) |

**The honest lesson, quantified:** the two GENOME-WIDE screens contribute +2,765 genes; all 8 TARGETED screens
combined add just 43 (melanoma +16, TianKampmann CRISPRa +21, CRISPRi +6, the rest 0 — they re-hit the shared
essential/regulatory core). A genome-wide screen in a new lineage is worth ~65× a targeted one. So "get more cell
lines" pays off only for GENOME-WIDE screens in cell types expressing new genes — chasing the many small targeted
screens is nearly tapped out. Frontier reach today ≈ 70% (across mixed K562/T-cell/… contexts; and with cross-line
transfer only r=0.19, this is coverage-in-context, not one universal map). (`coverage_stack.json`)

## Can the software DECODE pathways from data? (pathway_decode.py + `pathway` syscall)

Pathways are only 63% labeled and 17% mechanized — so we asked whether the software can recover a gene's pathway
from MEASURED data (genes in the same pathway are broken together → similar functional fingerprints), and fill the
unannotated genes. k-NN label transfer on DepMap co-dependency (1,150 cell lines), given all the help — fused with
Perturb-seq signatures + PPI + complex co-membership:

| signal | top-1 | top-3 |
|---|---|---|
| co-dependency only | 17.4% | 24.5% |
| + Perturb-seq signature | 17.7% | 24.6% |
| + PPI + complex (all help) | **21.3%** | **26.1%** |
| baseline (most-common pathway) | 2.8% | — |

**Verdict: partial — real signal, honest ceiling.** The software decodes a gene's exact Reactome pathway at 21%
top-1 = **8× chance** — the measured data genuinely encodes pathway structure. But exact fine-grained assignment is
noisy (1,200+ sub-pathways), so it's a strong SHORTLIST/neighbourhood engine, not an auto-annotator: the functional
NEIGHBOURS are consistently right (PSMB5→PSMB6/7 proteasome; RPL13→ribosomal; unannotated NEPRO→POP4/RPP40/RPP30 =
the RNase P/MRP complex, a genuinely useful new call) while the aggregated label drifts to a related pathway. The
"help" ablation is honest too: PPI+complex add +4 pts; Perturb-seq barely moves it (co-dependency already captures
it). Confident auto-fill of unannotated genes is limited (~2%, ~123 of 8,246) — it suggests, the annotation confirms.

Wired as `pathway GENE` (alias `decode`): decodes any gene's pathway from co-dependency neighbours, tags [matches
annotation] or [NEW] — turning the 37% unlabeled genes from blank into a ranked, data-grounded shortlist. Tests green.

## Reason across the data — the layer that matters more than the data (reason.py + `reason` syscall)

Having the data isn't the point; reasoning with it to a conclusion you can TRUST is. Any single layer is imperfect
(Perturb-seq shock ~0.64, FBA 0.64, LOEUF 0.57, complex 0.75, hub 0.50). Reasoning = converging INDEPENDENT lines,
and knowing when to trust the result. Validated on essentiality (ground truth = DepMap dep_frac), with every
evidence line independent of it (no circularity):

**(a) Reasoning beats any single datum.** Weighing the five independent lines by reliability (learned, out-of-fold)
→ **AUC 0.80**, above the best single line (complex, 0.75). Naive equal-weight averaging only ties it (0.757) —
because a noisy line dilutes a strong one — so the win comes from *weighing* evidence, which is what reasoning is.

**(b) The confidence is CALIBRATED — the important part.** P(truly essential) rises cleanly with agreement:

| independent lines agreeing | P(essential) |
|---|---|
| 0 | 10% |
| 1 | 42% |
| 2 | 76% |
| 3 | 88% |

So the software doesn't just answer — it knows *how much to trust its own answer*, and that trust tracks reality.
Wired as `reason GENE` (alias `explain`): shows the chain of independent evidence, the conclusion, and a calibrated
confidence (RPL13/PSMB5 → ESSENTIAL, P≈88%; SLC22A1 → NON-ESSENTIAL, P≈10%), and flags UNCERTAIN when lines split.
This is the layer above the data: not one measurement, but the convergence of independent ones. Tests + audit green.

## Test the biochemical limit — can we decode enzyme kinetics? (biochem_limit.py)

The executable layer (metabolism → FBA 0.68) runs only because the missing quantity — kcat (catalytic turnover) —
was imputed. So the real limit test: can the software DECODE kcat from the parts it has (domains, substrates,
partners, Km, abundance), for the enzymes it's NOT measured for? Validated ONLY against the 420 experimentally-
MEASURED kcats (the other ~2,100 records are themselves model-predicted — using them as truth would be circular),
cross-validated:

| features | Spearman | fold-error |
|---|---|---|
| domain family only | 0.226 | 46× |
| + substrate/partner counts | 0.197 | 50× |
| + Km + abundance (all parts) | **0.347** | **40×** |
| baseline (predict mean) | 0.00 | 65× |

**Verdict: PARTIAL — real signal, honest ceiling.** Enzyme FAMILY (domains) sets the ballpark and Km sharpens it,
cutting the error from 65-fold to 40-fold — but coarse parts decode kcat only weakly (Spearman 0.35), below the
deep protein-LM + substrate-chemistry state of the art (~0.5–0.7, which is what CatPred did to fill the other
records). So the biochemical LIMIT is the kinetics: the quantity you'd need to RUN the non-metabolic 83% of the
proteome the way we run metabolism is only partially recoverable from the parts we hold — it needs measurement, or
active-site/sequence-level modelling these features don't carry. Not exposed as a syscall (0.35 is too weak to
present as a reliable capability — reported as a limit, not a feature). (`biochem_limit.json`)

## The right way to use a wrong kcat — the running cell FLAGS it (kcat_flag.py + `check` syscall)

Reframe (correct, and how the field actually does it): don't PREDICT kcat (weak, 0.35) — put it into the running
cell and if it's wrong, the software can't run and flags it. For metabolism that's exact physics: an enzyme's kcat
(max turnover) must be ≥ the rate it actually operates at in the cell (incell_rate = measured flux ÷ measured
abundance). A kcat below that can't carry the flux → the model can't reach measured growth → IMPOSSIBLE.

Tested on 269 enzymes with a cell-imposed operating rate:

| kcat put in | flagged as impossible |
|---|---|
| correct value | 6% (false-flag) |
| 10× too slow | **93%** |
| 100× too slow | **99%** |
| too fast (×10–100) | 2–4% (not flagged — one-sided) |

Real measured kcats pass at **100%** — the cell runs with them. **Verdict: the complete-cell model validates its
own parameters.** You can't *predict* kcat well, but the flux the cell must carry *constrains* it, so a bad value is
caught 99% of the time with only 6% false-flags. Honest caveat: one-sided — it flags impossibly-SLOW kcats; catching
impossibly-FAST ones needs the two-sided ecFBA-at-measured-growth bound.

Wired as `check ENZYME KCAT`: put in a turnover number and the running cell returns CONSISTENT / TIGHT / FLAGGED —
IMPOSSIBLE (SLC2A8 at 100× too slow → flagged; at 100× too fast → passes, honestly noted as the blind side). This is
the positive flip of the kcat limit: prediction is weak, but the cell is a 99%-accurate parameter LINTER. Tests +
audit green.

## Verify the kcat flag against MEASURED values (kcat_verify.py)

Grounded the `check` constraint in measurement, not just derived numbers. On the enzymes with an experimentally-
MEASURED in-vitro kcat AND a flux-derived operating rate (n=46):
- **100% obey the constraint** the check relies on (measured kcat ≥ operating rate), **0 violations**.
- enzymes run at a **median 40% of their measured max** (p10–p90: 8%–93%) — which MATCHES the known in-vivo
  saturation (~50%, Davidi & Milo 2016). Independent grounding, not circular.

Honest caveats: (1) the overlap of measured-kcat AND operating-rate is small (n=46) — verified where we can, and
no measured kcat contradicts it. (2) Bonus finding — on the ~530 DERIVED/predicted kcats, median in-vivo/in-vitro
= **1.57** (apparent rate exceeds the predicted kcat for most), i.e. the check FLAGS many predicted values as too
slow. Since the measured kcats pass cleanly, this points to the predicted (CatPred/EC-prior) kcats running low —
exactly the linter doing its job: measured values pass, questionable predicted ones get flagged. (`kcat_verify.json`)

## Correction: the flux-derived rate PREDICTS measured kcat within experimental noise (kapp)

Follow-up to the flag: does the operating rate merely BOUND kcat, or PREDICT it? Tested against measured kcats —
and the flag's one-sidedness does NOT stop prediction:
- **Spearman(operating rate, measured kcat) = 0.93**; predicting kcat = rate/0.40 gives median **1.9× fold-error**,
  vs the **8.7× experimental noise floor** on kcat measurements → WITHIN noise. For near-saturated enzymes, 1.3×.
- This is exactly the published **in-vivo kcat / kapp** method (Davidi & Milo 2016; Heckmann 2018): back-calculate
  catalytic rates from measured flux ÷ measured proteomics, recovering kcat within measurement noise.

So for enzymes that carry flux, the running cell effectively MEASURES kcat within experimental noise — correcting an
earlier over-cautious framing that it could only lower-bound it. Honest boundary: NOT universal — only enzymes with
a measurable operating rate (269 of 2,549 in this condition) and not far below saturation; idle enzymes and the
low-saturation tail (26%) stay under-determined (loose bound only). Non-circular: the flux is kcat-independent
stoichiometric FBA ÷ measured abundance. (`kcat_verify.json` → kapp_prediction)

## How much flux data / the flag base vs actual kcat (fba_flux_coverage + flag_base_vs_kcat)

Ran the real Human-GEM FBA (12,931 reactions, growth 124.9). Flux data is SPARSE per state: only 1,149 reactions
(9%) carry flux at the WT optimum; 736 distinct enzyme genes carry flux (26% of 2,848 metabolic genes); ~269 of
those also have measured abundance → an operating rate; 46 also have a measured kcat. So the "running cell measures
/ flags kcat" instrument reaches ~736 enzymes PER CONDITION — an enzyme idle in the current state gives no flux,
hence nothing to flag. That sparsity is the ceiling: you can only read the kcat of a reaction the cell is using.

Flag base = operating rate = FBA flux ÷ measured abundance; the software flags kcat < base as impossible. Against
the 46 measured kcats: the base sits BELOW the actual kcat 100% of the time (never false-flags a real value), with
headroom (actual/base) median 2.5x (p10 1.1x, p90 12.5x). So the flag line is drawn at ~40% of the true kcat, and
the real value sits a median 2.5x above it — tight for saturated enzymes (CAD 1.1x, CS 1.5x), loose for under-used
ones (LDHA 11x, ARG1 16x). The headroom spread IS the saturation, which is why the bound never crosses a real value
yet only PREDICTS kcat tightly where the enzyme runs near capacity. (`fba_flux_coverage.json`, `flag_base_vs_kcat.json`)

## ⚠ CORRECTION — the kcat flag/prediction claims above were CIRCULAR; honest result vs Drive in-vivo kapp

Prompted to check the user's Drive set of ~500+ in-vivo kcat values, I found `davidi_kcat.json` (592 enzymes; kapp
= max over 13 NCI-60 conditions of |v|/[E] — the genuine Davidi & Milo 2016 in-vivo-kcat method, computed from FBA
flux ÷ abundance with **no kcat input**). Testing my own recent claims against it **overturned them**:

- The "operating rate" behind the flag and the "0.93 Spearman / 1.9× prediction" was `enzyme_records.incell_rate_per_s`,
  which is **literally `kcat_invitro × sigma`** (`incell_rates.py:161` → `enzyme_record.py:70`). Correlating kcat×sigma
  against kcat is not a prediction, and "kcat ≥ kcat×sigma" is trivially true. The 0.93, the 99% flag, the "100% obey,
  2.5× headroom" — all **circular artifacts of sigma**, not physics. (Tell: headroom 2.5× = 1/0.40 = 1/median-sigma.)
- Tested honestly on the **148** enzymes that have BOTH an experimentally-measured kcat AND an independent flux kapp
  (`davidi_kcat.json`, measured/EC-measured tiers only): the flux kapp sits ≤ measured kcat only **70%** of the time
  (→ **30% false-flag** on correct kcats), a 100×-too-slow kcat is caught only **65%**, and kapp does **NOT** predict
  the kcat value: **Spearman +0.08**, median **92× fold-error** (≫ the 8.7× experimental noise floor). The file's own
  validation block agrees (median fold-error 95.9×, 21% within 2×).
- **Why** the same method works in *E. coli* (Davidi 2016, r≈0.6) but fails here: there flux is **measured** (¹³C-MFA)
  with **absolute** proteomics; our human reconstruction uses **FBA-predicted** flux and a **single reference**
  proteome — too noisy to flag or recover kcat per enzyme.

**What survives, honestly:** flux-derived kapp is a genuine but noisy *lower-bound* signal — useful in aggregate,
not as a per-enzyme verdict. The `check` syscall now uses the non-circular flux kapp and reports the flag as a WEAK,
one-sided **hint to re-check**, not a proof of impossibility. `kcat_flag.json`/`kcat_verify.json` are stamped
`_SUPERSEDED`; the audit now asserts the honest weak/null numbers. (`kcat_invivo_validate.json`)

## Reasoning WITH a mutation — fuse variant-damage × cell-dependency (`mutate` syscall)

Ran the reasoning engine on specific mutations by fusing the two validated reasoners: the VARIANT layer
(`reasoned_variant`: AlphaMissense call + ΔΔG/functional-site mechanism + a sickle-cell gain-of-function override)
with the CELL layer (`reason`, AUC 0.80, calibrated: does the cell depend on this protein?). The join NAMES the
regime, because cell-fitness essentiality and disease pathogenicity are **different axes**. Illustrative panel
(not a new benchmark — the underlying engines were validated separately), each vs clinical truth:

- **TP53 R175H** → DAMAGING (AM 0.98, at a binding site) × ESSENTIAL → *LOF in a load-bearing gene, both layers
  agree, HIGH* → pathogenic ✅
- **TP53 P72R** (same gene) → TOLERATED (AM 0.09) → *no cell consequence* → benign ✅ — the variant layer
  discriminates WITHIN one gene, which the cell layer alone cannot.
- **HBB E7V (sickle)** → AlphaMissense says benign (0.23) but the **GOF override fires** → *"do not trust the
  benign call, needs a functional assay", LOW* ✅ — refuses the false-benign that per-residue predictors give.
- **MLH1 K618T** → DAMAGING (0.83) but the cell-fitness lens can't resolve it (mismatch-repair loss isn't
  cell-lethal) → honest MEDIUM-LOW, flagging that essentiality is the *wrong ruler* for a repair/suppressor gene.

The honest payload: the fusion surfaces the **essentiality ≠ pathogenicity divergence** (tumor-suppressors and
GOF are pathogenic on a different axis than cell fitness) instead of hiding it. `mutate GENE UNIPROT POS WT MUT`.
(`reason_mutation.py`, `reason_mutation_demo.json`)

## The reasoning PRINCIPLE generalized to any part of the cell (reasoner_core) — and where it pays off

The `reason` engine was built for genes, but its principle is substrate-agnostic: gather independent,
differently-blind evidence lines → weight each by measured reliability (out-of-fold, no circularity) → fuse →
calibrate confidence by how many lines agree. I lifted that machinery out of the gene-specific working into
`reasoner_core.reason_over(lines, truth)`. It reproduces the validated GENE result **exactly** (fused AUC 0.799,
best single 0.75, calibration 10%→42%→76%→88%) — proof the generalization is faithful, not asserted.

Then I pointed the SAME core at a different part of the cell — metabolic **reactions** — with an independent
ground truth (FBA single-reaction deletion: delete the reaction, does growth collapse?). Evidence lines swapped to
what's observable about a reaction: carries-flux, single-gene (no isozyme), its gene is DepMap-essential
(measured), metabolite-bottleneck (topology). n=4,000 reactions, 29 FBA-essential (~1%). Honest result:

- **With the `flux` line:** fused AUC 0.99 — but `flux` alone is 0.98 because a reaction carrying no flux at the
  optimum can't be essential, so `flux` is **near-tautological** with the FBA-deletion truth. That's one line, not
  convergence.
- **Independent lines only (drop flux):** the genuinely independent evidence (measured essentiality + topology)
  reasons to only **AUC 0.66**, with flat calibration (0→1→1→0%) → "rank it, don't quote its confidence."

**The honest lesson: the principle transfers mechanically, but reasoning only ADDS value where the substrate has
several independent *informative* lines that converge.** Genes have that (5 lines, no single dominates → 0.80 with
clean calibration). Reaction-essentiality is a physics property that FBA already answers, and measured/topological
evidence recovers it only weakly — so there, reasoning is redundant with the physics, not additive.

Also fixed a latent honesty bug in the core surfaced by this test: a flat near-zero calibration curve was passing
the monotonicity check via its ±0.02 tolerance. The verdict now requires real dynamic range (ΔP≥15 points) before
claiming "calibrated" — the gene case still passes (ΔP=78), the weak reaction case correctly does not.
(`reasoner_core.py`, `reason_reactions.py`, `reason_reactions.json`)

## Textbook-style reasoning for a property with NO measured layer — a protein's LEVEL in a pathway

We know a protein's pathway membership but often not its LEVEL (step 1 vs step 9), and no dataset gives this
per-protein. So we go textbook: collect little bits from INDEPENDENT sources and reason them together, exactly
like the gene case. Two genuinely independent modalities, neither reading the answer:
- **KEGG topology** (curated stoichiometry): BFS depth of each enzyme's product compound from the pathway entry
  compound, on the KEGG reaction graph — the order EMERGES from substrate→product edges.
- **PubMed literature** (the textbook/citation gradient): per enzyme, co-mention with the pathway ENTRY metabolite
  vs its EXIT metabolite (early enzymes co-cite the substrate, late ones the product) via NCBI eutils counts.

Ground truth = textbook step order (uncontroversial), used only to score. Result (`pathway_position.json`):

| pathway | KEGG topology | literature | fused | note |
|---|---|---|---|---|
| glycolysis (10) | **+0.99** | +0.87 | **+0.99** | both independent sources recover the order → cross-verified |
| TCA cycle (8) | +0.32 | **+0.93** | +0.90 | cyclic: KEGG topology breaks; literature is the robust generalist |

**Honest read — same shape as the gene/reaction findings.** It works, but the value is **coverage + independent
verification, not a fusion ranking-lift**: fusion does not out-rank the best single source (mean 0.95 vs 0.96).
What it buys is (1) **cross-verification** — two totally different sources (a curated reaction graph and citation
counts) independently arrive at the same order, so you can trust it; and (2) **coverage** — where one source is
blind, the other fills it (PKM is missing from the KEGG topology parse but literature places it last; the whole
cyclic TCA defeats topology but literature still gets 0.93). Each source has a blind spot; reasoning across them
buys trustworthy breadth. This is the "no-layer, go-textbook" case: sum up small independent clues, and where they
agree, trust the answer. (`pathway_position.py`)

## Tier-2: pathway LEVEL for the ~10k signaling/regulatory membership genes (SIGNOR directed graph + feedback-honest)

Tier-1 (2,549 metabolic enzymes) has clean substrate-chain levels. Tier-2 is the ~10,489 genes with pathway
MEMBERSHIP but no substrate chain (mostly signaling). Their "level" is an upstream→downstream tier in the DIRECTED
CAUSAL GRAPH (SIGNOR) — the stoichiometry source replaced by regulatory wiring: a gene's tier = its topological
position in the pathway's directed subgraph.

The catch metabolism didn't have: **signaling pathways have FEEDBACK LOOPS**, so a linear level is ill-defined for
the looped part (ERK→EGFR feedback; NF-κB induces its own regulators). Longest-path and trophic-level both blew up
on MAPK/TLR. The correct handling is **strongly-connected-component condensation**: collapse each feedback loop to
one node, tier the resulting DAG, and FLAG every gene in a multi-gene loop as "feedback module — no internal level."

Validated on textbook cascades — the flag is an honest **abstention signal**:
- apoptosis **+0.94**, Wnt **+0.85** (feed-forward → clean tier); MAPK/ERK −0.36, TLR-NFkB −0.62 (feedback → flagged)
- pooled: **feed-forward genes Spearman +0.55**, **feedback-loop genes −0.16** → the flag separates trust from noise

**Whole-cell census (10,489 membership genes):**
- **47% (4,960)** get a trustworthy feed-forward tier
- **3% (329)** collapse into feedback modules — the honest answer is "a module," not a rank
- **50% (5,200)** have no directed context within their pathway → abstain

So the answer to "label all pathway-membership genes with a level": metabolism is a clean pipeline (tier-1 done),
but signaling is a control system with loops — **level is only partially definable (~half), and the software says
so** (feed-forward tier / feedback module / no-context) instead of fabricating a number. Same principle, same
honesty: reason from the wiring where it's ordered, abstain where the biology loops. (`pathway_tier.py`)

## The software completes the pathway-level labeling itself (`cell_levels` + `level` syscall)

Handed the software the new machinery (tier-1 metabolic position + tier-2 signaling tier) and let it label the
whole cell, completing what it honestly can and abstaining where it can't. `cell_levels.py` runs both labelers at
scale into one per-gene table, served by the `level GENE` syscall. Whole-cell result (`cell_levels.json`):

- **metabolic step level:** 18 (the validated glycolysis/TCA enzymes; scaling tier-1 to all KEGG maps is the
  remaining network pass)
- **signaling upstream→downstream tier:** 4,787 (feed-forward, trustworthy)
- **context-dependent (flagged):** 160 — a hub whose level DIFFERS across pathways gets no single number
- **feedback module (flagged):** 329 — mutually-regulating, no linear level
- **no orderable context (abstain):** 5,196
- → **a trustworthy level for ~46% of the 10.5k membership genes; the rest it flags or abstains on.**

A real honesty fix surfaced here: the first pass picked each gene's level from its *smallest* pathway, which gave
a pathway-LOCAL position — so CASP3 (a downstream executioner caspase) came out "upstream" because it's at the top
of the micro-pathway "CASP5-mediated substrate cleavage." A gene's level is **pathway-relative**. Fixed to average
across all of a gene's pathways and flag high-variance genes as context-dependent: now `level CASP3` → "context-
dependent (varies across 18 pathways)", `level MAPK1` → "signaling midstream ±0.25", `level PKM` → "metabolic step
10/10 glycolysis", `level RELA` → "feedback module — no linear level". The software says WHERE it can place a gene,
and where the concept of a single level breaks down. (`cell_levels.py`, `level` syscall)

## Scaled tier-1 to all KEGG metabolic maps — the difference

Scaled the metabolic step-level from the 2 validated pathways to ALL 92 human KEGG metabolic maps
(`metabolic_levels.py`): auto-detect each map's entry compounds (no hardcoded roots), SCC-condense to survive
metabolic cycles, longest-path rank → per-enzyme step. Validation holds: glycolysis Spearman **0.90** (TCA −0.74,
cyclic → flagged, honest). Then re-ran the whole-cell labeling.

**The difference (`cell_levels.json`), metabolic step level: 18 → 746:**

| | before scaling | after |
|---|---|---|
| metabolic step (KEGG chains) | 18 | **577** |
| metabolic cycle (flagged) | — | 169 |
| signaling tier | 4,787 | 4,513 |
| **trustworthy level total** | ~4,800 | **5,259** |

Two honest calibrations that fell out of doing it at scale:
- **It's ~750 metabolic enzymes, not ~2,500.** Only enzymes sitting in an *ordered* KEGG reaction chain get a
  placeable step; transporters, isozymes, and endpoint enzymes don't. Earlier "~2,500" was the count of metabolic
  *genes*; the count with a real step level is ~750. (Of 3,773 raw KEGG symbols, only 772 are actual cell genes —
  the rest were KEGG aliases/outdated names, dropped.)
- **Metabolism is a small slice of the labelable genome.** Signaling tier-2 (4,513) still dominates; scaling tier-1
  added ~460 net trustworthy levels. The cell's "levels" live mostly in the regulatory network, not the metabolic
  pipeline.

Now `level FASN` → metabolic-step downstream (fatty-acid synthesis), `level CS` → upstream (TCA entry), `level PFKL`
→ midstream (glycolysis). The `level` syscall serves all 5,259 completed levels; the rest stays honestly flagged or
abstained. (`metabolic_levels.py`, `cell_levels.json`)

## The software's own bill of materials — what it still needs (`needs` syscall)

Because we know the complete-cell end state, the gap is computable. `needs.py` reads every layer's committed
coverage and tags each gap by what would close it — DATA (a findable measurement), METHOD (need an algorithm), or
HARD (biology has no single answer → must stay flagged, not faked). Current end-product state: 16,418/16,509 genes
have ≥1 trustworthy answer (91 truly dark); 5,259 pathway levels placed; reason AUC 0.80; response prediction 55%.

**Gap manifest (`needs.json`, served by `needs`):** 6 of 9 layers are DATA-limited, 2 METHOD, 1 HARD.
- **DATA (a data hunt / claude-science run closes these):** 91 dark genes (bounded list); per-cell-line
  essentiality (more CRISPR screens); response prediction 55%→ (more Perturb-seq — the biggest single axis);
  4,781 no-context pathway genes (more directed edges + Reactome ordering); reliable in-vivo kcat (measured human
  13C-flux + proteomics); condition-specific metabolic flux (13C-MFA).
- **METHOD:** gain-of-function / moonlighting variant effect (need a function-change model, not just ΔΔG);
  knockout-outcome-from-wiring (needs per-edge kinetics — mostly unmeasured).
- **HARD (must stay abstained):** 502 feedback-module + context-dependent-hub pathway genes have no single linear
  level; the GRN robustness that makes topology→essentiality genuinely underdetermined. Completing these would be
  fabrication — the honest end state is the flag.

The headline: **most of what's missing is DATA-limited, not a modeling wall** — which is exactly where new data
moves the needle. The incompleteness is honest and itemized, not hand-waved. (`needs.py`, `needs` syscall)

## "The feedback-module level is HARD" — tested the pushback, it was WRONG (it's predictable)

Flagged the ~322 feedback-module pathway genes as a HARD limit ("no linear level, must stay flagged"). Challenged
to test whether they can still be PREDICTED — and, like the kcat episode, the pessimism didn't survive the test.

- **Topology can't do it** (confirmed): SCC condensation, longest-path, AND the proper trophic-level linear solve
  all fail on feedback cascades (MAPK −0.39, TLR −0.50, PI3K −0.29). Not a bug — the feedback edges are REAL (ERK
  phosphorylates upstream EGFR/SOS), so the wiring genuinely says ERK is both up- and downstream.
- **An independent source CAN** (`feedback_order.py`): the literature co-mention gradient (upstream ligand/receptor
  vs downstream nuclear endpoint) recovers the canonical forward order — MAPK **+0.86**, TLR **+0.86**, PI3K
  **+0.82** (mean +0.85). EGFR 0.01 → … → ELK1 0.65 (the transcription-factor endpoint). Exactly how literature
  rescued the cyclic TCA cycle when metabolic topology failed.

So the feedback level is NOT hard — it reclassifies from HARD to DATA/METHOD (get the literature source at scale
with per-pathway markers). Updated `needs.json` accordingly: the HARD column collapsed to **0 genuinely-hard
layers** (7/10 DATA, 3 METHOD). The honest residue: the one thing not achievable is deriving outcomes by
*simulating* the topology — but the outcomes themselves (essentiality 0.80, level +0.85) are predictable from
data/literature. Prediction almost always has a source; the question is which one, not whether a wall exists.
(`feedback_order.py`, updated `needs.py`)

## Ingested the claude-science reference data — a canonical ID backbone (id_map)

The data run produced reference/infrastructure tables (Ensembl BioMart GRCh38.p14 + Human-GEM, 2026-07-13), pulled
from Drive: genes (86,411, Ensembl↔HGNC), gene_xrefs (308,270, Ensembl↔UniProt/RefSeq/Entrez), model_genes (2,848
Human-GEM genes with uniprot/entrez), reactions (12,931 with subsystem/EC/KEGG/GPR). `id_map.py` folds them into one
per-gene join backbone (`id_map.parquet`):

- **16,509 cell genes → 100% Ensembl, 99% UniProt, 99% Entrez** — the mismatch problem that lost metabolic enzymes
  earlier (KEGG symbols → cell genes) is now solved by a real ID bridge; any future dataset joins by any identifier.
- **2,417 metabolic genes, 2,014 with a Human-GEM subsystem** (147 subsystems) — a clean metabolic pathway
  membership straight from the model (PKM→Glycolysis, CS→TCA, FASN→Fatty-acid biosynthesis), better than KEGG
  scraping and a candidate to strengthen the pathway-level layer.

Honest bug caught in build: the first pass used `zip(df['hgnc_symbol'].dropna(), df['ensembl_gene_id'])`, which
MISALIGNS (dropna shifts one column, not the other) — TP53 mapped to the wrong UniProt (Q8IZL8 not P04637). Fixed
by dropping rows together; verified on known genes (TP53→P04637, MAPK1→P28482) and coverage jumped 82%→99% UniProt.

Honest scope: these are REFERENCE data (IDs, model tables, gene universe), NOT the measurements that close the deep
gaps (response prediction, kcat, essentiality — still "yet to come"). The one genuinely-new LAYER in the run,
`proteins.parquet` (20,431 proteins with subcellular localization + TM + domains), is 23 MB and exceeds the Drive
connector's 10 MB download cap — needs a slimmer export (uniprot_id + localization + TM columns, or split) to
ingest as a localization axis. (`id_map.py`, `id_map.parquet`)

## New LAYER from the science run: subcellular localization — and it genuinely lifts reasoning

The `proteins.parquet` upload (20,431 reviewed Swiss-Prot proteins) carried the one thing that was a genuinely new
LAYER, not just plumbing: **subcellular localization**. `localization.py` parses UniProt compartment keywords,
joins to the cell through the ID backbone, and — before trusting it — TESTS whether it adds signal:

- **compartment for 100% of genes** (16,451), sanity-checked (TP53→Nucleus, SDHA→Mitochondrion, ALB→Secreted)
- **it predicts essentiality** (non-circular: UniProt localization vs DepMap): Nucleus **19%** vs 9% base,
  Mitochondrion 12%, Secreted **0.5%**, Membrane 3.7% — biologically right (core in, periphery out)
- **it LIFTS the reasoning engine 0.799 → 0.858** (the localization line alone scores 0.73, near the previous best
  single line 0.75); calibration now 1%→67% over 5 agreement buckets

So this is the first science-run input that actually moved a metric. Folded it in as a 6th evidence line in the
`reason` engine (regenerated `reason.json` → AUC 0.86) and added a `loc GENE` syscall. Audit calibration check
updated to require real dynamic range (ΔP>40pt) rather than a fixed top bucket. The ID backbone from the previous
step is what made the join clean — reference data first, then the layer that uses it. (`localization.py`, `loc` syscall)

## Other reasoning engines — reasoner_core is a factory, not one engine (disease engine added)

reasoner_core reasons about gene essentiality (0.86); the same machinery works for ANY property with independent
evidence lines + a ground truth. Built a SECOND engine to show it: **disease-gene reasoning** (truth: ndis>0, 4,933
disease genes). It is a genuinely different axis — disease and essentiality overlap only 11% vs 9% base (nearly
orthogonal). Lines: constraint, complex, hub, pleiotropy, localization; literature EXCLUDED (circular with disease
databases).

Honest result: **fused AUC 0.65** (beats best single 0.62, pleiotropy), calibrated P(disease) 22%→62% with
agreement. It WORKS but is MODEST — far below essentiality's 0.86. The lesson: the reasoning PRINCIPLE generalizes
to any axis, but engine QUALITY depends on whether strong independent lines exist for that axis. Essentiality has
them (DepMap, Perturb-seq, FBA, localization → 0.86); disease-association is a harder axis where structural lines
carry less (→ 0.65, useful as a calibrated triage prior, not a strong classifier). Added the `disease GENE` syscall.

The current engine roster on the one core: essentiality (0.86), disease (0.65), pathway-position (literature 0.85–
0.99), feedback-order (literature 0.85), mutation (variant×cell fusion), assess (physics+data router). Same
principle, honestly different strengths. (`disease_reason.py`, `disease` syscall)

## Checked the network/annotation batch (complexes, pathways) — honest null; STRING PPI pending

Fifth science-run batch: complexes/complex_members (Complex Portal EBI, 2,498 complexes), pathways/
pathway_participants (Reactome release 97, 2,928 pathways, 178k participations), string_ppi. Checked whether they
IMPROVE anything before integrating:

- **Complex Portal**: high-quality curated, but REDUNDANT with our existing complex data — swapping it into the
  reasoning engine's complex line moves nothing (line AUC 0.751→0.748, fused reason 0.858→0.857, Δ −0.000). Adds
  only 145 genes over the current 3,257. No integration — it doesn't improve the model.
- **Reactome-97 pathways/participants**: a coverage REFRESH (10,176 participants vs current 10,489, +28 new genes),
  plus a pathway parent-hierarchy we didn't have (enables future pathway drill-down, but closes no current gap).
- Kept both as current-version, provenanced reference tables in science_data/; no claimed metric improvement.

- **string_ppi.parquet (11.3 MB)**: the one that could genuinely matter — STRING confidence-scored PPIs could
  expand the network/link layer beyond our current PPI. It's over the connector's 10 MB download cap; needs a
  direct upload (or a confidence-filtered slice, e.g. combined_score ≥ 700) to test whether it lifts link
  prediction or the complex/hub lines. Honest so far: 4 of 5 checked, redundant; the potentially-useful 1 is pending.

## STRING PPI (uploaded) — a real network layer that lifts reasoning 0.86→0.89

The uploaded string_ppi.parquet (STRING v12.0, 929k edges, per-channel scores) is the second science-run input that
moves a metric. Honest channel choice was the whole game:
- **physical channels only** (experimental + database): degree → essentiality AUC **0.795**. TEXTMINING excluded
  because it's literature-derived → CIRCULAR with essentiality (studied genes get more edges); textmining degree
  scores only 0.668, confirming the trap.
- Added the physical-STRING-hub as a 7th reason line: **reason 0.858 → 0.887** (the old causal-hub line was
  near-chance 0.50; STRING physical connectivity is a far better centrality-lethality signal). Calibration 1%→67%.

`string_ppi.py` emits a compact per-gene layer (degree + top partners) — the 929k-edge parquet stays out of git.
Added `ppi GENE` (top physical partners; `ppi TP53` → TP53BP2/ATM/CREBBP/USP7 …). This makes **two** science-run
wins (localization +0.06, STRING +0.03) and **one** honest null (complexes/pathways). The reason engine has grown
from 5 lines (0.80) to 7 (0.89) purely by ingesting validated data — each line tested for lift before wiring, each
non-circular. (`string_ppi.py`, `ppi` syscall)

## Don't get attached to one metric — a multi-metric data-impact evaluator (data_impact.py)

Correcting my own habit: I evaluated every new science layer against ONE number (the essentiality-reasoning AUC).
That's the same myopia as the kcat episode. Built `data_impact.py` to score each new layer across ALL engines, and
the honest matrix shows why it matters:

| layer | essentiality reason | disease reason | pathway-tier (STRING propagation) |
|---|---|---|---|
| **localization** | **0.80→0.86 (+0.059)** | 0.646→0.656 (+0.010) | — |
| **STRING PPI** | **0.80→0.85 (+0.048)** | 0.646→0.645 (−0.001) | ρ=0.22, reaches 2,469 genes — WEAK, not wired |

**A layer that wins one engine is not a win for the cell.** Localization and STRING lift essentiality strongly but
the disease axis barely — because they're essentiality-correlated signals (hubs, compartments), and disease is a
near-orthogonal axis. STRING's undirected network can propagate a pathway tier to 2,469 no-context genes but only at
Spearman 0.22 (undirected smears upstream/downstream), so it's honestly too weak to wire as a trustworthy tier.

The fix is process: every future dataset now gets scored across all engines, not one — so the full impact (and the
non-impacts) are visible before anything is claimed. (`data_impact.py`, `data_impact.json`)

## Is the software connected like a cell? Tested by knockout — CONNECTED and ROBUST (and it hides damage the same way)

The user's question: a cell is connected — knock out one line and you see the consequence unless a backup pathway
buffers it. Is our software the same? Tested it the way you'd test a cell (`connectivity.py`): hide a component,
watch what propagates.

- **CONNECTED (changes propagate):** removing the localization line drops essentiality reasoning 0.887→0.851
  (Δ −0.035); removing the STRING line drops it →0.858 (Δ −0.029). The layers are genuinely wired into the answer,
  not decorative — a change has a direct downstream consequence.
- **ROBUST (backup pathways):** knocking out localization.json / string_degree.json / reason.json does NOT crash
  the software — the dependent syscall runs DEGRADED via a fallback (skips the missing line, falls back to the
  uncalibrated agreement fraction). Exactly like a cell buffering a gene knockout.
- **The honest catch — same as the cell:** that robustness HIDES damage. A corrupted layer silently lowers accuracy
  without erroring; you only catch it by RE-RUNNING the audit (which regenerates and re-checks the number). This is
  precisely why the cell's own knockout→essentiality is null (robustness masks the perturbation) — our software
  inherited the same trait.
- **Honest limit:** connectivity is real at the reasoning layer + the ID backbone, but it is NOT a live signal —
  most modules are batch scripts writing artifacts, wired together through the reason engine, not a continuously-
  propagating network. So it's a connected, robust DAG — not a running organism. (`connectivity.py`)

## Replicate biology as software — a RUNNING runtime (biosim): near-field real, whole-cell response null

Made the connected-but-static DAG actually RUN. `biosim.py`: perturb a node → the consequence PROPAGATES through
the physical (STRING) network by random-walk-with-restart, buffered by robustness, with a checkpoint that fires when
the blast radius reaches an essential hub. Then tested — honestly — whether the running network reproduces MEASURED
biology (Perturb-seq M: 2,058 KOs × 8,563 genes).

It splits exactly the way everything in this project has:
- **NEAR-FIELD — real:** the propagation recovers the functional MODULE. `perturb PSMB5` → the proteasome
  (PSMB2/PSMA4/PSMB7…), `perturb RPL13` → the ribosome, `perturb SF3B1` → the spliceosome; the checkpoint fires on
  the essential hubs it reaches. That's genuine (the whodunit complex-coherence, now as a live run).
- **FAR-FIELD — honest null:** it does NOT reproduce the measured transcriptional RESPONSE. Propagated stress vs
  Perturb-seq response is Spearman **+0.02** (baseline +0.01, lift +0.01) — barely above chance, because the response
  is downstream/regulatory, not carried on the physical graph. Same structure-vs-dynamics split as the GRN
  essentiality null (0.47) and mechanistic sim (r=0.01).

So "biology as running software" is REAL as **propagation + robustness + checkpoint over the near-field** — a live
`perturb GENE` syscall that shows WHO is in the blast radius — and a NULL for the whole-cell dynamic response. The
running network tells you who's in the module, not how the whole cell answers. That is the honest ceiling of a
topology-driven runtime; the far-field needs the measured data, not the graph. (`biosim.py`, `perturb` syscall)

## Batch: tfbs_datasets + metabolite_properties (provenance-corrected) — descriptive, no gene-engine lift

Provenance labels corrected by the data run (UniBind release-year not exposed → stripped; ChEBI backend labelled
"ebi.ac.uk/chebi 2025-07"); values verified genuine. Tested for engine lift before claiming, per the standing rule:

- **tfbs_datasets** (3,478 UniBind ChIP datasets, 268 TFs): a dataset REGISTRY, not TF→target edges. #ChIP-datasets
  per gene → essentiality AUC **0.495**, disease **0.507** (both chance); reason lift **+0.000**. Can't be wired as
  regulatory edges (no peak→target assignments); kept as descriptive TF annotation.
- **metabolite_properties** (4,165 Human-GEM metabolites: ChEBI/KEGG/PubChem IDs + SMILES/InChI + mass/charge/logP):
  genuine metabolite-side ID+structure infrastructure — the metabolite analog of the gene id_map — but metabolite-
  level chemistry doesn't feed the gene-level engines. Kept as a reference metabolite layer.

Running tally of the science batches: 2 engine wins (localization +0.06, STRING +0.03), and now 3 honest
descriptive/null batches (complexes/pathways redundant; tfbs a registry; metabolite_properties descriptive). The
discipline holds — data that moves a metric gets wired, data that doesn't gets flagged and kept only as reference.

## Batch: intact_interactions — a real curated PPI network, but REDUNDANT with STRING (no engine lift)

IntAct (EBI, 68,499 human curated interactions, each PubMed-cited, 7,187 genes mapped). Tested as a network layer:
- IntAct degree → essentiality AUC **0.578** (vs STRING physical **0.795**) — weaker, because IntAct degree tracks
  how much a protein has been STUDIED (bait/prey in pulldowns) = literature bias, not true connectivity.
- On top of the current 7 lines (which already include STRING): 0.887 → 0.886 (**−0.000**) — no lift.
- As a REPLACEMENT for STRING: 0.858 — strictly worse.

STRING v12.0 already ingests IntAct plus more channels, so IntAct is subsumed. Kept as reference (its per-edge
PubMed citations could enrich a literature-backed `ppi` view, but that's descriptive, not an engine gain).

The pattern across the science run is now clear and honest: the FIRST high-quality instance of a data TYPE wins
(STRING = the PPI win, localization = the compartment win), and SUBSEQUENT instances of the same type are redundant
(IntAct = another PPI; complexes = another complex source). Diminishing returns per data type — new *axes* move the
needle, new *copies of an existing axis* don't. Tally: 2 engine wins, 5 descriptive/redundant batches.

## discover — aiming the validated reasoning machinery at the UNKNOWNS (not essentiality)

The user's directive: essentiality is already measured — stop re-scoring it and point the engine at things WITHOUT a
known answer. Four discovery engines were built (`discover.py` → `discover.json`, `discover` syscall). Each was graded
honestly; only the ones that validate are presented as capabilities, the rest are flagged.

**1. SELECTIVE dependencies — the one genuine WIN.** A selective (druggable) dependency has a BIMODAL raw gene-effect:
a tail of strong dependency in a small subset of lines against a bulk near zero (KRAS in KRAS-mutant lines, FOXA1 in
lineage-addicted lines). The metric is the LEFT-SKEW of the reconstructed RAW DepMap gene-effect (Z·sd+mu — the per-gene
z-scoring in `depmap_vecs.npz` had DESTROYED this signal, which is why an earlier frac-in-window attempt gave a 1.01×
null), floored to genes with ≥3 strongly-dependent lines so a single CRISPR outlier can't score. Validation (the known
set was NOT used to build the metric): **20/22 known selective oncology targets land in the top-500 of 18,443 genes —
33.5× enrichment.** Surfaces novel selective targets: SHLD3 (Shieldin, synthetic-lethal with BRCA), CD24, FGFR2, MAF,
NEDD4L, FZD5.

**2. DRUGGABLE triage — a useful derivative (partial).** Selective dependency AND surface-accessible (membrane/secreted,
from the localization layer): 9 genes, including real tractable targets — TNFSF10 (TRAIL), CD24, FGFR2, FZD5, IL2RG.
Honest gap: no Pharos/DGIdb tractability data loaded yet (dgidb_interactions.tsv is available but un-ingested), so this
is "selective + on the cell surface," not a full druggability call.

**3. NOVEL DISEASE candidates — kept, but flagged WEAK.** Genes not yet disease-linked that score in the top bucket of
the disease reasoner's prior: 36 genes at P≈0.62 (ARNT, ATF2/4, CDK1, CLOCK, HIF1A, FOXO3…). Honest caveats: the disease
axis is modest (AUC 0.65) and the calibration is COARSE — only 6 buckets — so this is a candidate SET, not a
high-resolution ranking.

**4. Dark-gene FUNCTION — an honest NULL.** Co-dependency + STRING label transfer. It VALIDATES on well-connected
annotated genes (their neighbours are pathway-coherent — tight complexes) but that number does NOT transfer to the
truly dark proteome: every dark gene has co-dependency neighbours, but they SCATTER across ~50 pathways (median vote
margin **0.02**), so the "prediction" is noise. Worse, the naive confidence is INVERTED — conf=1.0 came from a handful
of weak STRING partners all pointing one way (least evidence, not most). The only dark calls that survive a real
vote-margin threshold (≥0.15) are genes sitting in one tight coherent module — and they're all correct (NOL10, NOM1,
PPAN, MAK16, GTPBP4 → rRNA processing). So function-by-label-transfer is a narrow coherent-module gap-filler, NOT a
general dark-gene function predictor. This is the same structure-vs-dynamics / well-connected-vs-dark split seen across
the GRN (0.47), cellsim (r0.01) and biosim (+0.02) nulls.

Net: pointing the machinery at the unknowns yields ONE validated new discovery engine (selective dependencies, 33.5×
enrichment) plus a tractable-target derivative, with the disease axis honestly weak and dark-gene function honestly
null. The discipline is unchanged — validate before claiming, flag the nulls, and don't fake a number the data won't
support. Served by the `discover` syscall (`discover selective` for the ranked list).

## causal_reach — "remove any piece and calculate its effect": how far does it actually reach? (an honest limit)

The user's frame: the cell map is a CENSUS (who exists, who lives where, whose family is whose), but cause-and-effect —
remove any piece and compute the consequence — needs the *behavioural/surveillance* layer. This test quantifies exactly
how far the census alone carries toward that goal. For measured knockouts (Perturb-seq = ground-truth "removed this
piece, recorded the neighbourhood"), predict the effect from STRUCTURE ALONE (STRING random-walk propagation, the gene's
own measurement held out) and score predicted-vs-measured in shells at network distance 1/2/3 from the removed piece.

- **Whole-field Spearman +0.020** (all reached nodes) — reproduces biosim's positive number exactly, confirming the
  pipeline. But this is only the DISTANCE ENVELOPE: "closer to the removed piece = more affected, on average" — trivially
  true, and not "the effect."
- **Within-shell (envelope removed): NULL at every distance.** 1-hop lift −0.022, 2-hop −0.050, 3-hop −0.101 vs a
  random-source baseline. The larger 2-hop (23-gene) and 3-hop (55-gene) shells are null too, so it is not a
  small-sample artifact. Structure propagation cannot rank WHICH pieces inside a neighbourhood actually fire.
- **Structure-only cause screen** (the "eliminate + select candidates" direction, run on the network): near-chance —
  true cause in the top 10% only 14% of the time, median fractional rank 0.565 (chance 0.50).

**Conclusion (the sharpest form of the structure-vs-dynamics split):** the MAP is done and it is powerful for what a map
is (who exists, who neighbours whom, which module sits where — biosim's blast radius is real for tight complexes, but
coarse). It CANNOT compute a removal's effect. Genuine remove-and-calculate lives entirely in the INTERVENTION DATA: for
the ~2,000 pieces measured (Perturb-seq) the effect is known, and the only engines that work on cause-and-effect
(strace = direct observation; whodunit/diagnose = the measured co-dependency fingerprint) stand on that measurement, NOT
on the census. So the bottleneck to "remove ANY piece and calculate its effect" is SURVEILLANCE — measured interventions
across more pieces and conditions — not more census. More census (localization, PPI, complexes) refines the map; only
more interventions extend the causal reach. (causal_reach.py → causal_reach.json)

## causal_patch — can the on-disk intervention data extend causal reach? (honest null, forecloses the shortcut)

Follow-up to causal_reach ("structure can't compute a removal's effect"). Before asking for new data, test the tempting
shortcut: we hold a SECOND genome-wide intervention dataset on disk — DepMap CRISPR (18,443 genes × 1,150 knockout
contexts). Does DepMap CO-ESSENTIALITY (shared knockout-fitness profile) predict the MEASURED Perturb-seq transcriptional
response, which would let us extend "remove X → here's the effect" from 2,058 pieces to ~18,443 with data already here?

Head-to-head, predicting the measured Perturb-seq response over 7,990 shared genes (n=400 knockouts in both datasets):
- **DepMap co-essentiality: Spearman +0.009** (median +0.008)
- STRING structure (the causal_reach null): +0.015
- random gene: −0.001

**NULL.** DepMap co-essentiality is no better than the structure null and barely above chance. Co-essentiality (which
genes you ALSO need when you need X — a FITNESS-coupling axis) and the transcriptional RESPONSE to removing X (direct
targets + stress/compensation) are genuinely different axes. The genome-wide DepMap data is a functional-coupling proxy
(good for naming the affected MODULE, which `pathway`/`discover` already use) but NOT a response predictor. So there is
no cheap on-disk substitute — the causal reach can only be extended by ACTUAL measured interventions in more contexts.
This pins the requirement exactly: the multi-cell-line Perturb-seq (Replogle-Nadig: hepg2/jurkat/k562/rpe1) is the data
that would extend it, and it must be ingested, not inferred. (causal_patch.py → causal_patch.json)

## causal_expand — patching in the genome-wide screen: 2,058 → 9,871 measured removals (breadth up, precision flagged)

The payoff of "patch the available data." causal_reach + causal_patch proved the reach can only grow with MEASURED
interventions, not inference. The genome-wide Replogle K562 Perturb-seq (gwps: 11,258 guides → 9,867 gene knockdowns)
was already on disk. The debugger's loader was refactored to MERGE it with the essential screen instead of choosing one:
essential rows win on overlap (high precision preserved), gwps supplies the other ~7,800 genes as a measured extension,
on a common response-gene column space. Cosine matching (already used by every causal syscall) makes the merge drop-in.

Honest validation (biology recovery = a knockout's nearest co-perturbation twin is a real STRING physical partner):
- **Reach: 2,058 → 9,871 knockouts (+7,813 genes now measured, not inferred).** Every causal syscall (strace / whodunit
  / diagnose / predict) reaches the bigger set.
- **Essential precision PRESERVED: 16.5% (merge off) → 16.1% (merge on)** — adding 7,813 noisier genes and searching 4.8×
  more candidates does NOT pollute the high-precision answers (SF3B1 → SF3B3/SF3B2/SNRPA1, its spliceosome, unchanged).
- **New gwps-only genes: 1.2% vs 0.0% random** — genuinely above chance but WEAK, exactly the coverage/precision tradeoff
  the genome-wide screen carries (per-gene r~0.3 vs the essential screen's ~0.86).

So the honest result is BREADTH, not precision: the causal engine now gives a MEASURED (if weak) answer for 9,871 pieces
instead of nothing for 7,813 of them — and every low-precision answer is FLAGGED in the output (`_prec_note`:
"[LOW-PRECISION: measured only in the genome-wide screen (r~0.3) — a weak lead; verify]"). A weak measured answer beats
the inference null (causal_reach/causal_patch gave ~0), as long as the confidence is labelled. The merge is default-ON
when gwps.h5ad is staged (PERTURB_MERGE=0 to disable); it falls back to the essential screen when the file is absent.
Still pending (user uploading): the RPE1 second cell type, for the cross-cell-type causal-consistency test in
causal_expand.py (does removing X do the same thing when the context changes?). (causal_expand.py → causal_expand.json)

## investigate — predicted investigative dossiers for under-recorded genes (the profile-a-person-from-partial-records engine)

The user's method, built: for a gene we lack a full record for, assemble a case file the way you'd profile a person —
textbook RECORD first, then FAMILY (co-dependent + interacting relatives → guilt-by-association), DESTINATION
(compartment), PATH (pathway), TIMING (pathway tier, relative to a reference), INTERACTORS (STRING), SURVEILLANCE
(measured removal-effect if in the 9,871-gene debugger, else PREDICTED from measured neighbours), and the REQUIRED
SUPPORT its job implies (reasoned, not looked up: a secreted protein needs ER translocation + signal-peptide cleavage +
glycosylation + vesicle traffic; a mito protein needs TOM/TIM import; an enzyme needs substrate + cofactor + product
disposal). Closest relatives to the measured set FIRST — that is where prediction is reliable.

**Honest validation (held-out: hide a gene's own record, predict from family, measure recovery):**
- **JOB (proc, 12 classes): ~55% top-1** vs 27% majority baseline (~2.0×), across the closest 100/200/500.
- **DESTINATION (compartment, 12 classes): ~54% top-1** vs 22% baseline (~2.4×).
- Accuracy is stable across the closest 500 (they all have a near-perfect co-dependency twin) and degrades beyond;
  confidence is reported per line and per band, honestly (many single-gene confidences are ~0.33 even when the top
  call is right). PATH (fine-grained Reactome) is the weakest line (co-dependency decode ~21%, as in discover).

**THE UNDERWORLD (the standout):** dark genes whose strongest co-dependency partners form a COHERENT CHARACTERIZED
module — the hidden crew. Ranked by module coherence × job-consensus, it correctly places genuinely dark genes inside
real machines: **EMC7** → the ER Membrane protein Complex (crew EMC3/EMC1/EMC2), **MTG2** → mito-ribosome assembly
(crew MARS2/MRPL20/HSD17B10), **NCLN** → the ER Nicalin–NOMO complex (crew CCDC47/TMEM147). These are real assignments
recoverable at the same ~55% job / ~54% place reliability, and they surface the "someone must be doing this job and
it's nobody on file" cases the user described.

Served by the `investigate GENE` syscall (full dossier) and `investigate underworld` (hidden-crew list). This is an
UPGRADE — additive; the essentiality/discover/causal engines are unchanged (regression-checked). Honest scope: coarse
labels (job/place) predict well (~2× baseline); fine labels (exact pathway) and the required-support "likely role"
(often the generic first service, e.g. ER translocation for membrane genes) are weaker reasoned leads, flagged as such.
(investigate.py → investigate.json; bands 100/200/500)

## investigate — SHARPENED: the specific MACHINE, not the generic service

The required-support "likely role" was too generic (every membrane gene got "ER translocation"). Sharpened it to name
the SPECIFIC machine a gene belongs to, from a curated library of 44 named cellular machines (EMC, OST, SEC61, COPII/I,
mito & cytosolic ribosomes, TOM/TIM, proteasome 20S/19S, spliceosome, exosome, TRiC, V-ATPase, respiratory complexes
I/III/IV, ATP synthase, nuclear pore, MCM, ORC, cohesin/condensin, APC/C, TFIID, Mediator, Integrator, BAF, MICOS,
NDC80, TREX, GPI, SRP, GET/TRC, ALG, ERAD, MMR, Fanconi, mTOR/GATOR, TRAPP, retromer). A gene's machine = the one whose
canonical anchor genes it most co-depends with (co-essentiality recovers complexes strongly), always leave-one-out.

**Validation: 59% top-1 across 44 machines (chance 2%, n=297) — ~26× over chance** (held-out known members matched back
to their own machine; strict top-1, so sibling near-misses like 20S↔19S proteasome count as wrong — the "right
neighbourhood" rate is higher). Matches carry a confidence tier (confident ≥0.40 / probable ≥0.25 / tentative) plus the
raw score and margin, so weak calls are visible not hidden: EMC7 → EMC complex [confident 0.587, margin 0.415], MTG2 &
PTCD1 → mito ribosome [confident], while NCLN → EMC [tentative 0.29] and a borderline DHX16 → proteasome [tentative
0.22] are flagged as uncertain. When no library machine clears threshold, the fallback now names the gene's OWN module
by its crew ("uncharacterized membrane module (co-crew: SMIM32, SMIM31, VSIG10L2)") — a real coherent cluster of small
membrane microproteins — instead of a generic service. Surfaced in the dossier (MACHINE line) and `investigate
underworld`. Additive; regression-checked (reason/discover/strace unchanged).

## investigate — FULL surveillance dossier for the unmeasured genes (built, not just noted; honestly graded)

Re-scoped to the user's spec: (1) target = the closest genes we have NO measured surveillance about (7,303 genes, not
in the merged debugger), ranked by proximity to the measured set; (2) for each, BUILD the surveillance data itself, not
a note; (3) full protein-level profile — product, where it goes, when, HOW it acts, WHY it exists, function, PPI.

New dossier lines: PRODUCT (protein type + compartment), HOW (operates as part of {machine}, physically via {PPI}),
WHY (serves {pathway}; knockout drives down {blast radius}), and the SURVEILLANCE is now the actual predicted
removal-effect (blast radius: which genes go up/down), built via the VALIDATED cellformer context predictor
(same-complex 3× / co-expression 2× / PPI 1×), with honest abstention for singletons.

**Honestly graded — the two halves are very different:**
- **WHAT / WHERE / MACHINE — solid.** proc (job) 60% top-1 vs 27% baseline (2.2×), comp (destination) 40–46% (1.8–2.1×),
  machine 59% (26× chance). PPI + required-support are direct/reasoned.
- **The SURVEILLANCE VECTOR — weak, and flagged as such.** Built removal-effect vs the real measured response (Pearson,
  self held out): **0.14 for genes with a complex, 0.06 for singletons, 0.12 overall** (random 0.01). Buildable for
  ~77% (the ones with a complex/≥6-gene context); singletons are refused, not faked. My first attempt used a flat
  co-dependency average and stamped a false "expected r≈0.43" — corrected: the honest number is ~0.14, a DIRECTIONAL
  lead (complex members share responses), NOT a precise response vector. cellformer's own committed ceiling is r~0.23 on
  the clean essential screen; ~0.14 here reflects the noisier merged (genome-wide) debugger.

So the engine reliably answers what a gene IS / where it goes / what machine it's in (2×–26× baseline), and gives a weak
directional guess at its removal-effect — with the fidelity printed on every prediction so nothing is oversold. The full
100 dossiers are saved as the deliverable (investigate.json bands[100].dossiers). Additive; regression-checked.

## investigate — REFOCUSED on connections + raw surveillance, dropped cause-and-effect (per spec)

The user narrowed it: they need the SURVEILLANCE raw data and the CONNECTIONS a gene has with others — NOT the predicted
cause-and-effect (the removal-response vector, which validated weak at ~0.14). This is exactly the observation-vs-
intervention cut, and it lands on the reliable side: connections are DIRECTLY MEASURED, not predicted.

The dossier now leads with the observed profile (product / where / when / how / why / function / machine) and a
CONNECTIONS block of real edges:
- **co-dependency** — top co-essential partners across 1,150 DepMap lines (weight = correlation), INCLUDING dark
  partners (flagged *), because the map needs them;
- **physical PPI** — STRING partners with confidence scores;
- **complex co-members** — the curated machine it belongs to (EMC7 → EMC1/2/3/4/6/8);
- **pathway** — its route.

The predicted removal-response VECTOR (cause-and-effect) was removed from the dossier; `SURVEILLANCE raw` now just states
whether the real Perturb-seq is on file (measured genes → "use strace") or not (unmeasured). Coverage: CONNECTIONS mapped
for 100/100, 200/200, 500/500 of the closest unmeasured genes — median 10 co-dependency + 8 physical-PPI edges each, all
real measured associations. Attribute recovery unchanged (proc 60% / comp 40–46% / machine 59%). EMC7's dossier now
shows its whole complex as edges (co-dep EMC3..MMGT1; PPI all EMC subunits at 999) — the connection map, not a guess at
what its removal would do. Additive; regression-checked (reason/discover/underworld unchanged).

## anomaly — patch connections + measured perturbation into one map, scan for disagreement (1 of 3 types real)

Assembled the map: the 9,561 genes that have BOTH a measured Perturb-seq knockout fingerprint AND a co-dependency
vector, plus their physical-PPI edges. Scanned for where the structural connections and the measured perturbation
DISAGREE, three ways — then verified each against known biology (the honest step):

- **COMPARTMENT-OUTLIER — REAL (92 genes).** A gene whose whole co-dependency module sits in a different compartment
  than its label. Verified catches: FASTKD5 labeled nucleus, module mitochondrial 10/10 — FASTKD5 IS a mito
  RNA-processing protein (mislabel); ACTR6 labeled cytoskeleton but nuclear (an INO80 remodeller subunit); ELP5
  nuclear→cytoplasm (Elongator is cytoplasmic); DIS3/EXOSC10 cytoplasm→nucleus (nuclear exosome). The module CORRECTS
  the annotation — a genuine annotation-error / incompleteness detector.
- **DECOUPLED — CONFOUND, not trustworthy.** Even after a magnitude control (both genes strong responders), the top
  hits are obligate mito/metabolic complex-mates (PET117~COA6 COX assembly, NDUFB3/NDUFA10~NDUFS1 complex I) whose K562
  knockdown responses are WEAK — weak signal reads as "decoupled," not real biology. TSC1~TSC2 (an obligate complex)
  appeared before the control and was correctly filtered out.
- **HIDDEN-LINK — CONFOUND, not trustworthy.** After excluding pan-essential genes, the survivors are spurious
  correlations between weak responders (AGRN~AQP10, MYB~PTGR2) plus shared erythroid-lineage signal (K562 = CML).

Root cause and honest conclusion: the measured Perturb-seq response in K562 is dominated by response MAGNITUDE and the
proliferation/stress axis, so "structure vs measured response" disagreement mostly measures assay SIGNAL QUALITY, not
biology — the same structure(recoverable)-vs-dynamics(noisy) split the whole project keeps hitting. The trustworthy
anomaly is the cross-ANNOTATION one (compartment vs co-dependency module), which found real mislabels; the
perturbation-based anomalies are not reliable on this screen. (anomaly.py → anomaly.json)

## cell_map — the complete-cell database rendered as an interactive NETWORK MAP

Turned the complete-cell database into a spatial map of the cell and its networks (cell_map.py -> cell_map.json ->
viz/cell_network_map.html, published as an artifact). Nodes = the connected functional core (3,200 genes: every
essential gene, TF, complex member, and network hub, of 16,509), placed by SUBCELLULAR COMPARTMENT in a concentric
cell layout (nucleus + cytoplasm interior, organelles as clusters, membrane on the rim) via a compartment-anchored
force layout computed in Python (blow-up-guarded, aspect-preserving). Edges = the two real measured networks: physical
PPI (STRING, 22,000 shown of 191,944) and co-dependency (DepMap). Colour = compartment (12-hue palette, validated 3:1
on the near-black ground); size = PPI degree; essential/TF/dark are flag-highlightable. Interactive: pan/zoom, hover a
node (compartment + role tags + degree), toggle PPI vs co-dependency, highlight essential/TF/dark, search a gene.
Screenshot-verified through Chromium (fixed a canvas-sizing bug and a force-layout blow-up along the way); all
interactions error-free. It is the honest map — the functional core and its measured wiring, laid out by where each
gene lives — not a fabricated whole-cell picture.

## network — the complete-cell database wired into ONE interconnected + interdependent graph (in the software)

Took the same complete-cell image that built the cell HTML and wired every gene into one queryable multi-layer
network object inside CellOS (network.py -> CellNetwork; the `network` syscall). The database's edges split cleanly
into two families the object keeps distinct:

- **INTERCONNECTED (physical — who TOUCHES whom, undirected):** ppi (STRING, 191,944), complex co-membership (2,039
  complexes), ligand→receptor. 208,735 edges.
- **INTERDEPENDENT (functional — who NEEDS / CONTROLS whom):** causal (SIGNOR/CollecTRI signed direction, 60,103),
  regulatory (TF→target signed, 612,133), signaling (17,432), co-dependency (DepMap co-essentiality), co-expression,
  synthetic-lethal. 922,265 edges.

Every gene is a node wired across all layers; the object answers `wire(gene)` (its full place in both networks),
`link(a,b)` (every relationship + direction between two genes — e.g. MDM2↔TP53 returns the full feedback loop: PPI +
complex + causal both ways + regulatory + signaling), `depends_on`/`controls` (upstream vs downstream in the
functional graph), `path(a,b,family)` (a physical vs a functional route), and `hubs(family)`.

**Honest validation.** The point of keeping BOTH families is that they are *complementary yet coherent*, and both are
measured:
- **Complementary, not duplicates.** Only ~6% of a functional edge (co-dependency/causal/…) is also a physical edge —
  who-NEEDS-whom is mostly NOT who-TOUCHES-whom, so the interdependent layer carries information the physical layer
  doesn't (regulatory: 99% non-physical; co-expression 98%; co-dependency 94%; causal 92%).
- **Coherent, not noise.** That same overlap is 46–214× the 0.14% chance rate — every functional layer recovers real
  physical complexes far above random (synthetic-lethal 214×, signaling 139×, causal 56×, co-dependency 46×). And
  physical partners share a curated pathway above chance: PPI 1.6×, complex 3.6×, causal 2.4×.
- **Connected.** One giant component per family (interconnected 99.7%, interdependent 100%).

**Honest caveats (flagged in the verdict and the syscall).** (1) Co-dependency and co-expression partners do NOT map
onto curated *literature* pathways (~0.9× chance) — they track physical modules instead; that's expected for
co-essentiality, but it means those two layers should not be read as pathway annotations. (2) Every edge is an
observed/curated relationship, not a computed prediction. (3) Consistent with dependency.py and causal_reach.py, the
wired edges do NOT let you PROPAGATE an unmeasured knockout's far-field cascade — the structure is real and traversable,
but long-range dynamics don't compose. Use the wired edge; don't trust a propagated one. (network.py -> network.json;
`network GENE | network link A B | network path A B [fam] | network hubs [fam]`)

## knockout / dependency — the MEASURED Perturb-seq surveillance as a directed knockout-effect engine

Separate from the curated `network` above, this turns the measured Perturb-seq screen (the debugger M, ~9,871
knockouts × 8,202 measured genes) into a directed dependency network and a knockout-effect syscall (dependency.py ->
Dependency; the `knockout` syscall). `knockout GENE` returns the REAL measured blast radius of removing a gene — which
genes go down (need it) and up (released by it) — for the ~7,651 genes that are both knocked out and measured; e.g.
removing SF3B1 (splicing factor) releases the snoRNA-host lncRNAs GAS5/SNHG1/ZFAS1 (biologically sensible). `knockout
impact` ranks genes by blast-radius size — a measured load-bearing score (top: spliceosome/proteasome/ribosome/RNA-pol
machinery — PSMD14, RBM22, POLR3A, RPL7A, SNRPD2, EIF4A3).

**Honest validation (the direct effect is real; propagation is not).** The wired knockout edge is real measured data.
But the dependencies do NOT chain: only ~0.5% of a knockout's strong responders are the gene's own STRING partners (vs
~0.1% chance) — most of the effect is distal/regulatory, not direct binding — and propagating one step through the
graph to predict the 2nd-order effect is Spearman ~0.009 (~chance ~0.019). So the software can SHOW a measured
knockout's effect and rank genes by measured impact, but it cannot COMPUTE an unmeasured knockout's cascade by chaining
dependencies — the same structure(measured)-vs-dynamics(non-composing) split found in causal_reach and confirmed here.
Use the measured edge; don't trust the propagated one. (dependency.py -> dependency.json; `knockout GENE | knockout
impact`. Merged essential+genome-wide screen via `_load_debugger`; genome-wide-only genes are flagged lower-precision.)

## protein / degrade — knock out the PROTEIN, not the gene (structural disassembly vs transcriptional response)

Asked to knock out a protein instead of a gene, and the two really are different operations — the software now does
both (protein_knockout.py -> ProteinKnockout; the `protein`/`degrade` syscall). A GENE knockout (CRISPR, the existing
`knockout`/`strace`) deletes the coding sequence and you measure the TRANSCRIPTIONAL response (the Perturb-seq blast
radius). Degrading a PROTEIN (a PROTAC / molecular glue — the real way to remove a protein instead of a gene) removes
the physical molecule, and its immediate effect is STRUCTURAL: a protein is a shared component wired into several
molecular machines, so removing it pulls it from EVERY complex it is a subunit of at once. `degrade SMARCA4` collapses
all four SWI/SNF chromatin-remodeling variants; `degrade PSMA1` all four proteasome assemblies; `degrade SF3B1` the
spliceosomal complexes — each leaving the co-subunits still expressed but their assembly incomplete. It also lists the
physical interactions severed, and falls back to PPI partners for proteins that aren't curated complex subunits (PCNA
-> the RFC clamp loader + CDKN1A).

**Honest validation — is "degrading a protein disables its machine" grounded?** Two tests, and they say different
things on purpose:
- **STRUCTURAL (strong).** 46% of complex proteins have a complex-mate among their top co-dependencies, vs 0.04%
  chance (~1000x) — machine parts are co-essential, so removing one cripples the assembly's function and the DepMap
  fitness data proves it. This is the real validation.
- **MEASURED / transcriptional (deliberately weak).** Across 705 knocked-out proteins the complex-mates move a median
  of only 1.07x the average gene in the protein's OWN measured knockout, and only 18% of machines show a strong (>1.5x)
  transcriptional bounce (the proteasome, PSMA1 3.1x, and spliceosome are the feedback-wired exceptions). That is not a
  failure — a protein knockout acts POST-transcriptionally, so mRNA-seq is largely blind to it. The blindness is the
  point: the structural `degrade` sees exactly the machine-level damage the measured `knockout` cannot, so the two are
  complementary, not redundant.

**Honest limits.** One-protein-per-gene: the cell image carries no isoform/PTM/domain-resolved network, so there is no
isoform-specific knockout (I checked — the gene records don't hold that, so claiming it would be fabrication). And it
names the machines that BREAK from curated complex membership; it does not compute the far-field transcriptional
cascade — that stays the measured `knockout`, and cascades still don't compose. (protein_knockout.py ->
protein_knockout.json; `protein GENE` / `degrade GENE`.)

## influence / cascade — perturbation as a "what-affects-what" network, with each effect's route reconstructed

Per the spec: use the perturbation data for ONE thing — a directed network of what a gene's removal AFFECTS in other
gene-products (the measured ENDS; we know the endpoints, not whether it was transcriptional or downstream). Then
reconstruct the ROUTE of each effect from the structure we DO know, WITHOUT ever predicting an unmeasured effect
(influence.py -> InfluenceNetwork; the `cascade`/`trace` syscall). Every strong measured effect KO(X)->P is labelled:
- **DIRECT** — X physically/causally contacts P (STRING PPI / curated causal edge / same complex): stage 1, the entry.
- **MEASURED-MEDIATED** — a specific, non-hub, MEASURED stepping-stone M: X strongly moves M AND M strongly &
  specifically moves P, sign-consistent (X→M→P). BOTH hops are real knockout data — this is exactly "keep the earlier
  knockout data in context": the intermediate is another gene we actually knocked out, so the step is measured, not
  guessed.
- **UNRESOLVED** — neither; the distal/regulatory/diffuse far-field that does not compose.

`cascade X` stages X's whole blast radius; `cascade X P` reconstructs the single route (e.g. `cascade SF3B1 RBM22` ->
DIRECT physical PPI; GATA1's cascade enters through its curated causal targets FCER1G/BST2/HLA-E then routes on through
measured stepping-stones).

**Honest validation (measured on this screen, 4,981 effects).** DIRECT contact **1.2%** (6.5x chance) — real but
sparse; MEASURED-MEDIATED **14.3%** (placebo where X does NOT move P: **0.3%**, so **43x** — the stepping-stones are
non-random); UNRESOLVED **84.4%**. So **~16% of the cascade is reconstructable** and the rest is honestly flagged
unrouted. The influence network itself is 100% real data.

**The confound that had to be controlled, and a correction to the plan.** Naively, mediators are dominated by pan-hub
genes (knockout moves everything → M→P holds trivially; the raw search picked 97th-percentile-blast hubs). Excluding
pan-hubs and requiring M→P to be SPECIFIC to M is essential. And the probe showed that using PPI/pathway structure as a
mediator *filter* does NOT help (it just re-selects hubs; structural-required and random-order search scored
identically) — so structure is used only to name the DIRECT entry point, and the MEASURED stepping-stones carry the
cascade. That is the honest version of "track at what stage it happened, then follow the cascade down": it follows for
the ~16% the data can actually support, and says so for the 84% it cannot. (influence.py -> influence.json; `cascade
GENE` / `cascade GENE TARGET`.)

## pertseq — the four canonical Perturb-seq applications, attempted on our data and honestly graded

Asked whether the perturbation data can do the four textbook Perturb-seq applications (GRN mapping / unknown-gene
function / disease convergence / immune logic). Built each, VALIDATED each, and graded honestly (pertseq_apps.py ->
PertSeqApps; the `pertseq` syscall). The structural constraint that drives the answers: our data is K562 (leukemia), a
single unstimulated bulk-averaged condition.

- **[1] GRN wiring — DOES NOT work on this data.** Inferred 4,132 TF→TF influence edges from the 617 TFs that were
  knocked out & measured, but they recover curated SIGNED direct regulation (SIGNOR/CollecTRI) at only **1.6x chance**
  with **48% sign agreement (= chance)**, and the 73,878 feed-forward loops are only **52% sign-coherent (= chance)**.
  So this is a CO-RESPONSE network (genes that move together), NOT the causal wiring diagram / real FFLs. Why: a single
  bulk steady-state condition mixes direct+indirect (the 84%-unresolved result from `influence`), so TF→target
  influence is dominated by indirect effects. Real GRN/FFL mapping needs time-resolved or nascent-RNA data to isolate
  direct targets. The idea is sound; the data property is the blocker — reported as an honest negative, not dressed up.
- **[2] Unknown-gene function by transcriptomic fingerprint — WORKS.** A gene whose KO response matches a known gene's
  KO response shares its role. Held-out validation: fingerprint-kNN recovers a known gene's pathway **84%** of the time
  (chance 21%, **4.1x**); yields **128 confident dark-gene calls** (TIMM23B → mito import/cristae via HSPA9/DNAJC19;
  GPN3 → RNA-pol assembly via RPAP2). Honest: it is a validated SIMILARITY hint, not proof — a shared growth-arrest
  fingerprint can mimic a shared pathway, so it can mislead on the truly dark tail. `pertseq function GENE` does the
  live lookup. (This beats the earlier co-dependency `discover function`, which was a null on dark genes.)
- **[3] Disease convergence — method validated, data thin + wrong context.** The convergence test (do a gene set's KO
  fingerprints cluster tighter than random?) is validated HARD on pathway positive controls (spliceosome z=156,
  proteasome z=71). But only **1 disease** has enough knocked-out risk genes in the thin OpenTargets annotation, and
  K562 is the wrong cell context for the neuro/immune polygenic diseases the application targets. Method real; can't be
  meaningfully applied here.
- **[4] Immune brakes-vs-gas — NOT feasible.** Defining brakes vs gas needs a ± stimulus axis (KO, then stimulate, read
  the CHANGE in the response) and immune cells with a response program. Our data has neither (one unstimulated leukemia
  condition). Only basal regulators of an immune-gene panel can be listed — a different and much weaker claim — and it
  is labelled as such, not as brakes-vs-gas.

Honest scorecard: one works (function), one is a validated hint awaiting a cleaner apply (disease), one is a method
waiting for the right data type (GRN needs time resolution), one is out of scope (immune needs stimulus + cell type).
(pertseq_apps.py -> pertseq_apps.json; `pertseq` / `pertseq function GENE`.)

## fetch_xaira / xcell — staged Xaira X-Atlas/Orion (a SECOND cell type), lifting knockout coverage 53% -> 95%

Got the Xaira X-Atlas/Orion genome-wide Perturb-seq atlas (Xaira-Therapeutics/X-Atlas-Orion on HuggingFace): 8M
HCT116 + HEK293T cells, all 18,903 protein-coding targets, ~16k UMIs/cell, median ~140 cells/perturbation, CRISPRi.
It ships as ~330 per-cell parquet batches (~115GB total) — far too big to hold — so fetch_xaira.py STREAMS it: download
one batch -> fold into a running sum(log1p(CP10k)) & cell-count per gene_target -> delete the parquet -> checkpoint ->
next. Peak disk stays ~one batch; the output is a compact control-relative [perturbation x gene] signature h5ad
({SCRATCH}/hct116.h5ad) in the exact format cellos._read_pert already mounts. Folded all 109 HCT116 batches (median 150
cells/pert, 17,768 perturbations x 16,380 genes).

**Coverage it adds** (the answer to "how much coverage does it give us"):
- **Knockouts of cell-model genes: 8,696 (K562 alone) -> 15,749 UNION = 95.4%** of the 16,509-gene cell now perturbed
  in >=1 cell type (up from 53%); **+7,053 genes K562 never knocked out**.
- **8,288 knockouts are now in BOTH cell types** — cross-cell-validatable.
- **Genes measured (readout): 6,977 -> 16,383** (2.3x — Xaira's deep sequencing reads ~2x more genes per knockout).
- A genuine **second cell context** (colorectal HCT116; HEK293T also available) — the gap behind the immune/disease/
  cross-cell-type limits flagged earlier.

**Validated cross-cell-type consistency.** On strongly-moved genes, a knockout's HCT116 signature matches its K562
signature at **Pearson 0.611** (shuffled 0.064, n=1,128 paired perturbations) — the same knockout does the same thing
in leukemia and colorectal cells, so agreement is meaningful (global all-gene correlation is low, ~0.07, because most
perturbations barely move most genes — the strong-effect number is the honest one).

**New capability: `xcell GENE` (crosscell.py).** For a knockout measured in both screens, split its responders into
cell-type-ROBUST (moved in both, same direction — a property of the gene, the trustworthy core of a cascade) vs
cell-type-specific. Validated by biology: PSMB5 (proteasome) is robust (r0.55, 11 conserved proteostasis responders
UBC/FTL/HSPA1B); GATA1 (master hematopoietic TF) has 0 robust responders and 1,867 K562-only — correctly flagged
cell-type-specific, because GATA1 is functional in erythroleukemia K562 but inert in colorectal HCT116. This is exactly
the cross-cell confirmation the single-K562 `cascade`/`influence` engines lacked.

NOTE: the pipeline (fetch_xaira.py, crosscell.py) is committed and reproducible; the staged data (hct116.h5ad, 1.1GB)
lives in the ephemeral scratchpad and is regenerated by `python3 colab/fetch_xaira.py`, not committed.
(fetch_xaira.py -> {SCRATCH}/hct116.h5ad + outputs/orphan/xaira_coverage.json; crosscell.py -> `xcell GENE`.)

## fieldsim — the four-layer "protein-field" whole-cell engine, built end-to-end and tested honestly

Asked to build a proposed continuous-field whole-cell predictive engine (spherical-harmonic field decomposition +
mutation->ΔΔG->γ pipeline + SDF organelle boundaries + Hill source terms) as one closed loop. Built all four layers
with REAL math and transparent lightweight stand-ins for the heavy external services (ESMFold/AlphaFold/FoldX/APBS/
Enformer), so it runs end-to-end here, then tested it against measured data and against its own physics (fieldsim.py ->
the `fieldsim` syscall):

- **Layer 1 (SH compute):** real spherical-harmonic projection on radial shells; the overlap integral ∫Φ_AΦ_B d³r
  computed as a coefficient contraction matches the direct numeric integral to **0.1%** — the O(L) overlap claim is
  correct FOR CO-CENTRED fields.
- **Layer 2 (mutation->params):** mutation -> ΔΔG (proxy) -> decay rate γ. Shows the specified γ=γ_wt·exp(ΔΔG/RT)
  **diverges** (2.1e7 at ΔΔG=10 kcal/mol — an unphysical rate); replaced with a bounded folded-fraction mapping.
- **Layer 3 (SDF PDE):** a reaction-diffusion-decay PDE on an SDF cell (nucleus/mito/membrane) with harmonic-mean face
  conductance so D->0 barriers seal properly. A mito-emitted field stays **100% confined** without an export motif vs
  **28%** with one — the spatial gate works.
- **Layer 4 (Hill source):** TF field density at a gene locus gates enzyme production; TF knockout collapses the
  downstream enzyme field **100%**. The closed loop runs (a destabilising enzyme missense raises γ and drops output
  36%; a benign one 0%).

**Honest verdict.** It assembles, runs, and is internally consistent — a genuinely useful *mechanistic demonstrator*.
But the tests quantify why it is NOT a validated predictor, confirming the two problems flagged when reviewing the
proposal: (1) **the coordinate-frame seam is real** — a co-centred compact source is 1 coefficient, but the coefficient
count climbs to ~121 (L=10) once the source sits 0.34 off-centre, and moving the expansion centre at all is an O(L³)
FMM translation that mixes l-channels, so the cheap-overlap advantage holds for a STATIC co-centred descriptor and
erodes for a moving/diffusing field (the honest design diffuses a scalar and keeps c_lm co-centred, not the SH object);
(2) the exp(ΔΔG/RT) γ-mapping is unphysical. And most important: the external models are stand-ins, and per every
measured result in this repo (transitivity ~0.009, causal reach ~chance) the whole-cell mutation->macro-phenotype map
does not compose from these local fields. Use it to study the mechanism and generate hypotheses; do not trust its
quantitative phenotype without checking each edge against measured data. (fieldsim.py -> fieldsim.json; `fieldsim` /
`fieldsim run`.)

## fieldsim vs measured — the honest knockout test: the field engine's prediction is at chance

Wired the test the internal-consistency checks couldn't give: run REAL gene knockouts through the fieldsim engine's
forward model (its predictive core = regulatory Hill+decay propagation over the curated network, Layers 2+4; the
spatial/SH Layers 1+3 spread fields but carry no signal about WHICH genes change) and score the predicted downstream
signature against the MEASURED Perturb-seq knockout effect, with proper baselines (fieldsim_test.py -> the
`fieldsim validate` syscall). Tested on 52 knockouts that actually do something (>=25 measured movers each, >=10
regulatory targets).

**Result — it does not beat chance:**
- predicting which genes move: **AUC ~0.50** (chance 0.5)
- **static baseline** (just list the knockout's curated targets, NO dynamics): **AUC ~0.50** — identical, so the
  Hill+decay+diffusion machinery adds nothing over the network topology it was handed
- **recall ~3%**: fieldsim only reaches genes on a curated regulatory path from the knockout, so it touches ~3% of the
  genes that actually moved while lighting up a ~500-gene neighborhood that mostly did NOT move (huge false reach)
- direction on the few movers it reaches is **below a coin-flip** (pooled sign agreement ~0.16) — the propagated
  curated regulatory signs actively disagree with the measured response (consistent with the earlier pertseq GRN
  finding that curated regulation recovers the measured knockout at ~chance)

**Honest process note:** first pass had a broken metric (exactly-0.0 sign agreement) and a backwards coverage label; I
inspected individual predictions (they were non-degenerate — MYC's few reached movers were 4/4 on direction), found I
was testing on knockouts with ~no measured effect, restricted to real-effect knockouts, fixed the metrics to
recall/precision + pooled sign agreement, and corrected a verdict that had called a 0.16 sign-agreement "better than
chance" (it is worse). The corrected numbers are the ones above.

**Conclusion.** fieldsim is a correct, internally-consistent MECHANISTIC DEMONSTRATOR whose forward prediction of a
real knockout does NOT beat chance or a trivial network lookup — the same far-field-does-not-compose wall (transitivity
~0.009) every engine in this repo hit. The measured `knockout`/`cascade` engines — which report what was observed and
honestly flag the ~84% they can't route — remain the trustworthy tools. (fieldsim_test.py -> fieldsim_test.json;
`fieldsim validate`.)

## latent bridge — testing the M-L-M "foundation-model Phase 2" premise before building it

A proposed fix for the transitivity wall: keep the near-field physics, but bridge the far-field with a pre-trained
single-cell foundation model (scGPT/Geneformer) that "knows the statistical topology" — claim: it fixes recall and
sign, and the moat is feeding it the physics of NOVEL mutations. Tested the load-bearing assumption directly, with a
faithful Geneformer-style latent (gene-context embedding from the co-expression graph via truncated SVD — exactly the
signal Geneformer learns — + k-NN response transfer), held out, vs the baseline that has beaten this whole model class
in fair benchmarks: **predict the mean perturbation response** (latent_bridge_test.py -> the `latent` syscall).

**Result (200 held-out knockouts with real effects):**
- predict which genes move: **LATENT AUC 0.711 vs MEAN 0.693** — a gain of just **+0.018**
- correlation on moved genes: **LATENT 0.518 vs MEAN 0.537** — the mean actually WINS
- **stratified by novelty (the decisive test of the claimed moat):** for NOVEL perturbations (farthest from any
  training gene) the latent is **worse** than the mean (gain **−0.017**); it only helps when a close twin is already in
  training (+0.04).

**Conclusion.** The latent bridge is ~the generic mean response (most of a knockout's signature is a shared stress/
proliferation/cell-cycle program, which is why "predict the mean" is so hard to beat — and why scGPT/Geneformer/GEARS
repeatedly fail to beat it in fair benchmarks, Ahlmann-Eltze et al. 2024). Phase 2 does NOT cross the transitivity
wall — it relocates it into latent space, where it reappears as "this perturbation is unlike anything in training." A
foundation model is worth wiring for INTERPOLATION (a perturbation observed in another context — which our cross-cell
`xcell` already does at r~0.6), NOT for the physics-guided novel-missense whole-cell prediction that is the pitch. And
for the stated metabolic targets (CYP3A4 etc.) there is a second problem: these are transcriptomic models, but
metabolic phenotype is largely post-transcriptional (a protein knockout's effect is ~invisible to mRNA, median 1.07x —
see protein_knockout), so an scRNA foundation model is doubly removed from metabolic flux. If a learned model is used,
it MUST be reported against the mean baseline and the novel stratum or it will look far better than it is.
(latent_bridge_test.py -> latent_bridge_test.json; `latent` / `latent run`.)

## pathway_struct / pstruct — step-by-step structural dossier of a known pathway (glycolysis), real data

Ran a structural analysis of the ordered steps of glycolysis (pathway_struct.py -> the `pstruct` syscall), assembling
per step from REAL sources: gene(s)/isoenzymes and reaction order (substrate->product, before/after); protein
FAMILY/fold and residue-level ACTIVE-SITE + ligand-binding residues (UniProt REST — e.g. GAPDH catalytic Cys152
nucleophile with NAD+/G3P binding residues, TPI1 electrophile@96 + proton-acceptor@166, PGAM1 phosphohistidine@11);
oligomeric state + allosteric regulation (UniProt — PFK's ATP/citrate/F2,6BP valve, PKM2's FBP activation); PPI
partners (our cell DB); essentiality/disease (DB); and an AlphaFold-model structural analysis.

**Induced fit, computed and cross-checked (not asserted).** For each enzyme, split the fold into two lobes and measure
whether the catalytic residues sit in the inter-lobe cleft — requiring BOTH a split active site AND genuinely
separated lobes (lobe gap and gap/Rg). Result recovers the textbook DOMAIN-CLOSURE enzymes: HK1 (66Å lobe gap) and
PGK1 rank top — their active sites in the cleft that closes on substrate — joined by the multidomain PKM and GPI.
Cross-checked against known biology with the failures disclosed: the gap criterion correctly demotes PGAM1 (a compact
single domain the raw cleft-span had false-positived by an arbitrary bisection); and NO domain-split metric can see
LOOP-closure induced fit, so TPI1 (real induced fit via catalytic loop-6) scores flat — a true, stated false negative.

This is the NEAR-FIELD structural layer the whole session found trustworthy: gene, reaction order, family, active-site
residues, interactions and allostery per step are real and recoverable. HONEST LIMITS: the cleft-span is a
single-structure PROXY for induced fit (a true apo-vs-holo domain-closure measurement needs two experimental
conformers, which AlphaFold's single model doesn't provide); active sites/families are curated UniProt facts, not
predictions; and none of this predicts the far-field cellular consequence of perturbing a step — that stays
measurement's job. Sources: UniProt REST + AlphaFold DB (v6 via API) + cell DB, cached in scratchpad; biopython for
structure parsing. (pathway_struct.py -> pathway_struct.json; `pstruct`.)

## enzyme_patterns / enzyme — research: what governs metabolic enzyme essentiality (a self-correcting finding)

Turned the structural work into actual research: a multidimensional analysis over 2,511 metabolic enzymes asking what
makes one ESSENTIAL (DepMap dep_frac>0.5, 9% of them), with a held-out model, a label-shuffled null, and effect sizes
(enzyme_patterns.py -> the `enzyme` syscall). Dimensions: reaction-redundancy, sequence-family paralogs, LOEUF
constraint, PPI degree, co-dependency degree, complex membership, # reactions, # pathways, disease count, master.

**Finding — two independent axes (held-out AUC 0.862 vs null 0.51):**
1. **CENTRALITY (dominant).** Essential enzymes are embedded and irreplaceable-by-position: in a protein complex
   (effect +0.50, the single strongest), spanning many pathways (+0.48), PPI hubs (+0.42) — and, notably, SPECIALISTS
   (they catalyse FEWER reactions, -0.31, not more). Evolutionary constraint (LOEUF -0.35) is a correlated axis.
2. **PARALOG BUFFERING (real, orthogonal).** Essential enzymes lack a same-family isoenzyme backup — median 0 vs 2
   sequence-family paralogs (effect -0.30, p=2e-14), and this adds +0.021 AUC on top of centrality alone. An enzyme is
   indispensable when it sits at a central junction AND has no twin to cover for it.

**The process is the point — a live self-correction.** I went in expecting the paralog-buffering law. My first
redundancy proxy — genes sharing a REACTION — not only failed but *reversed the sign* (essential enzymes appeared to
have MORE "backups"), because that proxy counts obligate complex co-members as isoenzymes. Rather than trust the
confident wrong answer, I fetched UniProt sequence FAMILIES for all 2,549 enzymes and measured true paralogs — and
buffering came back clean and correct. It's the session's core discipline in miniature: a confounded proxy gives a
confident wrong answer until a better measurement corrects it. HONEST LIMITS: UniProt family granularity varies (broad
superfamilies dilute paralog counts); centrality's sub-features are correlated (one latent axis); LOEUF and dep_frac
are both importance measures; and this explains WHICH enzymes are essential, not the phenotype of removing one — the
near-field structure/network layer that IS recoverable, unlike the far-field dynamics. (enzyme_patterns.py ->
enzyme_patterns.json; `enzyme` / `enzyme run`.)

## interface_hotspots / interface — the pattern INSIDE PPIs at the amino-acid level (for the single-nucleotide-mutation goal)

The real ask: for the mutation problem, find which residues at a PPI interface actually CARRY the interaction, so a
single amino-acid change there predictably alters binding. Built on SKEMPI 2.0 — 4,956 MEASURED single-point interface
mutations across 345 real complex structures, each with the change in binding affinity (ΔΔG) and, for ~1,400, the on/off
rates (interface_hotspots.py -> the `interface` syscall). Four findings, all from measured data:

- **Position carries the energy (the hotspot / O-ring law, reproduced).** By SKEMPI's interface classification, buried
  CORE residues have mean |ΔΔG| **1.92** kcal/mol (**38% hotspots** >2), support 1.48, rim 0.84, surface **0.29 (0%
  hotspots)**. Binding energy is concentrated in a few buried residues; rim/surface mutations mostly don't matter.
- **Identity matters.** The strongest hotspot residues are the **aromatic/charged** ones (Y, R, K, F, D, W, L, E);
  small/flexible ones (S, V, Q, A, N) are weak. Alanine scan: 516/2,878 X→Ala mutations are hotspots (ΔΔG>2).
- **Barrier vs stability (the energy-barrier / induced-fit axis).** With on/off rates, a mutation weakens binding
  **~68% by destabilising the bound state** (faster off-rate) and **~32% by raising the association barrier** (slower
  on-rate) — so the kinetics separate an encounter-barrier effect from a stability/induced-fit effect.
- **Predict ΔΔG.** From residue change + interface depth ALONE, complex-held-out **Pearson 0.36** (Spearman 0.40) —
  honest for simple features (full structural predictors like mCSM-PPI2/BeAtMuSiC reach ~0.5-0.6 on blind splits by
  adding real interface burial/contacts, so there's headroom by fetching the complex geometry). Top features:
  interface depth, Δvolume, Δhydropathy.

So for the mutation goal we CAN flag which interface positions are load-bearing (core, aromatic/charged, high predicted
ΔΔG) and whether a variant there will act via the barrier or the stability — a genuine near-field structural
capability, the recoverable side of the ledger. HONEST LIMITS: r~0.36 means we TRIAGE/RANK interface variants, we don't
nail every ΔΔG; SKEMPI is biased to well-studied complexes; and it needs a solved/predicted COMPLEX structure to place
the residue at an interface (no complex -> no call). (interface_hotspots.py -> interface_hotspots.json; `interface`.)

## flex_physics / flex — a physics ΔΔG-binding node with clash relief, tested on measured SKEMPI, patched with the research

Asked to build the "Local Backbone Flex" fix for the rigid-physics steric-clash trap (a bulky mutation makes two atoms
overlap and the Lennard-Jones r^-12 term explodes to +thousands kcal/mol), test it on all mutation data, then combine
with the interface research (flex_physics.py -> the `flex` syscall). Implemented a real reduced force field (LJ in the
AMBER rmin form + harmonic restraints as the bonded 'springs' + gradient-descent minimiser + a soft-core variant) and
tested it against measured SKEMPI ΔΔG on 140 fetched complex structures.

**Findings (measured):**
- **The bug is real, reproduced.** A rigid LJ evaluation of a bulky (->Trp) interface mutation reads **1e6-1e10
  kcal/mol** on real structures — the r^-12 wall, exactly as described.
- **The proposed fix is necessary but NOT sufficient — an honest correction.** Local backbone minimisation relieves
  MODERATE clashes (e.g. 1JCK B20->Trp 1334 -> ~180 kcal/mol with ~2Å physical backbone motion) but CANNOT rescue a
  deeply BURIED bulky clash by moving atoms alone (a fixed-rotamer Trp jammed in an Ile pocket stays explosive),
  because full relief also needs ROTAMER sampling, which position-only gradient descent doesn't do.
- **The soft-core (my variant) is the robust practical fix.** Capping the repulsion returns physical values for **92%**
  of mutations vs **67%** for rigid, and it's the best-behaved clash score (it's what Rosetta effectively does by
  ramping fa_rep) — no minimiser required.
- **The biggest win was patching physics into the research.** The WT sidechain's buried cross-interface CONTACT count
  alone tracks measured alanine-scan ΔΔG at **r 0.47**, and adding physics (vdW + contacts) to the interface_hotspots
  features lifts complex-held-out ΔΔG prediction from **r 0.37 to 0.48 (+0.12)** — real structural burial from the 3D
  coordinates beats the coarse core/rim position labels.

**Honest limits (this pass).** A REDUCED force field (vdW + soft-core + harmonic restraints, uniform epsilon, single
fixed-rotamer mutant, no rotamer packing / electrostatics / solvation), so its absolute ΔΔG is a TRIAGE signal, not
FoldX/Rosetta accuracy; and it speaks only to the near-field INTERFACE effect of a variant, not the downstream cellular
consequence. Net: the steric-clash fix is real, the soft-core is its practical form, and feeding real interface geometry
into the empirical predictor helped most. (flex_physics.py -> flex_physics.json; `flex` / `flex run`.)

### Chemistry added: sp3 rotamer sampling + electrostatics + induction

Then asked to add the chemistry: sp3 sidechains do not rotate freely — steric hindrance around sp3-hybridised carbon
pins the χ1/χ2 dihedrals into discrete wells at **+60° (gauche+), 180° (trans), −60° (gauche−)** (the Dunbrack
rotamer-library idea), plus "induction and forces". Implemented (a) **rotamer sampling** — enumerate the χ1 (about the
CA–CB bond) and χ2 (about CB–CG) wells, rebuild each pose by rotating the sidechain atoms to set that dihedral, score it
against the partner with the soft-core LJ, and keep the lowest-clash rotamer; (b) **electrostatics** — Coulomb with a
distance-dependent dielectric ε(r)=r on resname-aware united-atom partial charges; (c) **induction** — the
charge-induced-dipole (Debye) term U_ind = −½ Σ αᵢ|Eᵢ|²/K, the leading polarisation "force". Measured each against
SKEMPI — a **two-sided, honest** result:

- **Rotamer sampling is the right PHYSICS for the buried clash minimisation couldn't fix.** Letting a bulky mutant jump
  to its lowest-clash sp3 well turns impossible energies physical — **1TM1 I58→Trp goes from ~9×10¹⁰ to −2.4 kcal/mol**,
  1JCK B23→Trp from ~5×10⁸ to −1.8 — and cuts the bulky-mutation explosion rate to **4%** (vs 33% rigid single-rotamer,
  8% soft-core). This is exactly the escape-by-rotating mechanism, and it's what you need for a usable *absolute* energy.
- **But — the honest catch, measured not assumed — rotamer sampling does NOT improve bulky-mutation RANKING.** As a
  standalone rank predictor of measured ΔΔG on the larger-residue set, the *relieved* best-rotamer score correlates
  **worse (Spearman +0.13)** than the *raw, un-relieved* rigid clash magnitude (**+0.35**). The reason is real: the raw
  magnitude encodes "how much steric burden this bigger residue imposes", which tracks destabilisation; once you relieve
  the clash to the best well, that variance is discarded. So the correct use is BOTH — rotamer-relieved energy as the
  physical absolute value, and the raw steric burden kept as a separate ranking feature.
- **Electrostatics + induction add real, held-out accuracy on the polar/charged mutations.** The lost cross-interface
  Coulomb term alone correlates r −0.16 with measured ΔΔG(X→Ala) and the Debye induction term r −0.35; stacking both on
  top of the steric features lifts complex-held-out r from **0.485 to 0.521** (+0.037), and the full physics+research
  stack reaches **r 0.52** (+0.15 over the research features alone).
- **On the Rosetta scoreboard (flex_vs_rosetta.py) this moved the node from Pearson 0.47 → 0.528** on the same hard
  complex-held-out split (Spearman 0.547, RMSE 1.18, AUC 0.806, n=2602, ~2 ms/mut) — from ~FoldX up to between Rosetta
  cartesian_ddg (~0.50) and mCSM-PPI2 blind (~0.58), and now **~84% of flex_ddG's** correlation (~0.63) at ~10⁴–10⁶×
  less compute. The remaining gap is the expensive part we still don't do: full partner-sidechain repacking,
  Dunbrack-probability-weighted rotamers (we select by lowest steric energy, not library likelihood), and explicit
  solvation/desolvation.

Net: rotamer sampling fixed the buried-clash PHYSICS (physical energies, no explosions) but not the ranking; the raw
clash magnitude still ranks better there. Electrostatics + induction gave the genuine held-out accuracy gain
(0.47→0.53). Both conclusions are measured against real SKEMPI, not assumed. (flex_physics.py, flex_vs_rosetta.py;
`flex` / `flex run` / `flex rosetta`.)

### Implicit desolvation — an honest NEGATIVE result (added, tested, left out)

Then asked to add an implicit **desolvation** term (the highest-value *cheap* physics we were missing: burying a carbon
is favourable via the hydrophobic effect, but burying an unsatisfied polar/charged atom pays a desolvation penalty).
Implemented an EEF1/Eisenberg-style occlusion term — the partner atoms Gaussian-occlude solvent from each interface
sidechain atom, times an Eisenberg atomic solvation parameter — which is **O(n)**, so no speed cost. Then measured it,
and the honest answer is that **it did not help on our validation:**

- On its own it's a weak signal (r **−0.18** with measured ΔΔG(X→Ala)) that is **largely redundant** with the
  buried-contact count (r 0.47) — both mostly encode hydrophobic burial, and alanine hotspots are hydrophobic/aromatic
  anyway, so the polar/charged distinction desolvation uniquely adds is a minority of the cases.
- Folded into the predictor it moved the alanine held-out r only **+0.01** (0.52→0.53, within noise), and on the larger
  2,602-mutation flex_vs_rosetta split it slightly **HURT** (Pearson **0.528 → 0.515**) — a partly-redundant weak
  feature just adds variance the RF overfits on the training folds.

So per the "only keep it if it genuinely improves the model" rule, the desolvation term is **implemented and available**
(`sidechain_vdw(...)["desolv"]`, `_desolv`/`_asp`) but is **NOT folded into the headline feature set**; the honest best
stays at Pearson 0.528 from sterics + electrostatics + induction. A proper SASA / Poisson-Boltzmann solvation model (what
FoldX/Rosetta actually use) might extract more than an O(n) occlusion proxy — but on this alanine-heavy validation, the
cheap version doesn't earn its place, and I'm reporting that rather than dressing up a +0.01. (flex_physics.py [2].)

## ppi_screen / screen — the ΔΔG node as a PPI partner-screening pass (loss-localization; honest NO on novel gain)

After establishing (in the AlphaFold discussion) that our ΔΔG node answers "does this mutation still do the PPI" better
than AlphaFold does for a point mutation — and that AlphaFold is mutation-blind and can't be fed our data — the natural
build was a **partner-screening pass**: take a mutation, score it against a panel of candidate partners, flag **loss** of
a known PPI at the right partner and **spare** the ones it doesn't touch (ppi_screen.py → the `screen` syscall). The
honest result is a **decomposition**: screening-for-loss is two separable jobs, and the node does the first for free but
needs its full ΔΔG model for the second.

**[1] LOCALISATION — "which partner's interface is this residue on" — near-exact, but that's structure-reading.**
In the 91 cached complexes with ≥3 protein chains, scoring each interface residue against the panel of all other chains,
the node picks the true contacting partner **top-1 98%** of 2,975 interface residues (random baseline 25%). Reported
honestly as a **sanity check, not the achievement** — localisation is pure geometry / near-field structure (the
recoverable regime), so ~exact is expected. I deliberately did **not** headline the trivially-perfect
true-vs-decoy-partner AUC (1.0), because the score and the ground truth are both just proximity.

**[2] MAGNITUDE — "how much does THIS substitution change binding" — the real problem, anchored to measured ΔΔG.**
The key honest finding: the localisation/burial score **alone does NOT predict magnitude** — across mixed substitutions,
on-target burial vs measured |ΔΔG| is **r −0.01 (~0)**, because burial ignores *what you mutate to*. Fix the substitution
(X→Ala) and the on-target signal reappears (**r 0.51**). And burial cleanly separates the **functional** interface from
the rest: SKEMPI **core** residues (burial 1.5, measured |ΔΔG| **1.97**) vs **surface** (burial 0.1, |ΔΔG| **0.27**). So
the usable screen is **LOCALISATION (structure, exact) × MAGNITUDE (the full ΔΔG model, Pearson ~0.52)** — flag "this
mutation sits on partner P's interface AND is predicted to cost a lot" ⇒ predicted **loss of P**, and ~no effect on
partners the residue never contacts.

**[3] GAIN — the honest limit.** Affinity-increasing mutations are real (**6%** of SKEMPI have ΔΔG < −0.5) and the full
model can score them **at an existing interface**. But a **novel emergent PPI** — a mutation building a brand-new
interface with a partner it never bound — is **not** something this pipeline can claim: it would require docking the
mutant onto every candidate partner (the unreliable step) plus experimental confirmation. I state this plainly rather
than fabricate a gain result.

**Net:** a good **loss / specificity** screen — "which known partner does this mutation break, and which does it leave
alone" — which works because it's near-field structure; and an **honest NO** on novel-partner discovery, which is the
dynamics-side problem the whole project keeps finding does not compose. (ppi_screen.py → ppi_screen.json; `screen` /
`screen run`.)

## surface_fingerprint / surface — the REAL trained MaSIF-style model (after a mock was exposed)

A "SurfaceDLFramework" deep-learning model was proposed — geometric surface fingerprinting to screen the proteome for
"neo-morphic interaction candidates." Running and dissecting it showed it was a **mock**: the "trained Mesh-CNN/PointNet
weights" were `np.random.normal(seed=42)` (trained on nothing), the "proteome database" was `np.random.uniform` noise,
the protein IDs were fabricated by arithmetic (`ENSG00000{100000+idx//20}`, and they change with the random seed), the
"99% match confidence" was an artifact of a rank-6 bottleneck (garbage 6-value patches scored 98–99% too), and the "59×
speedup" compared two unrelated operations with an O(N²)-as-linear extrapolation (off by 200×). It also claimed to solve
the exact **novel-PPI** task that `ppi_screen` had just concluded is not solvable without docking + experiment.

So I built the **honest, real version** (surface_fingerprint.py → the `surface` syscall): REAL surface patches from REAL
atomic coordinates + REAL residue chemistry of REAL interface-labelled complexes (**58,121 patches, 139 complexes,
13,921 true contacting patch-pairs**), a REAL contrastive encoder **trained by gradient descent** (torch), **validated on
complexes held out by GroupKFold** (no complex in both train and test). The objective is the real MaSIF one: embed
surface patches so that patches that actually **contact** across an interface land close (surface complementarity).

**Two-sided honest result:**
- **What works — pairwise discrimination.** On held-out complexes the trained encoder separates true interface
  patch-pairs from non-contacting pairs at **AUC 0.66**, versus **0.50** for the untrained raw-feature-cosine baseline —
  so learning complementarity adds **+0.16** of real, generalising signal, and (unlike the mock) **training earns it**.
- **What mostly doesn't — exact retrieval.** Given a patch, finding its *actual* partner patch among hard decoys (other
  surface patches on the same partner protein) is only **weakly above chance** — top-1 **2.2%** vs 1.5%, top-5 **11.2%**
  vs 7.7% (a consistent ~1.4× lift in every fold, not solved). Pinpointing the one right patch among dozens of
  near-interface patches needs finer spatial resolution than residue-level features carry.

**Honest limits driving the ceiling:** residue-level features (geometry-from-atoms + chemistry-from-identity), **no MSMS
molecular-surface mesh and no APBS electrostatics** — coarser than real MaSIF, which uses true surfaces + Poisson-Boltzmann
fields and reports stronger discrimination *and* usable retrieval. Trained on the 139 real complexes we have, **not
20,000 proteins**: surface-complementarity training needs **complexes** for interface labels (monomers have none), so the
20k-monomer "proteome DB" is a downstream fetch that would **not** change this validated accuracy. And this is near-field
surface **structure**, **not** validated discovery of a novel PPI.

**The point:** the truth (0.66 AUC, at-chance exact retrieval) is *less* impressive than the mock's 99%-confidence
fiction — and that gap is the lesson. The approach is real and the signal is real; MaSIF-grade performance needs the
molecular-surface + electrostatics tooling this environment lacks. (surface_fingerprint.py → surface_fingerprint.json;
`surface` / `surface run`.)

### Desolvation, retested the RIGHT way — real SASA + the satisfaction correction (honest: correct but redundant here)

The crude occlusion desolvation above hurt the model (0.528→0.515). Asked to try it properly — the Lazaridis-Karplus
(EEF1) idea with real **SASA** and the **hydrophobic-illusion / satisfaction correction** — I built desolv_eef1.py (the
`desolv` syscall): **real Shrake-Rupley SASA** (Bio.PDB), bound-in-complex vs unbound-isolated-chain, for the burial
area; hydrophobic (C/S) burial rewarded; and the key fix — a buried **polar/charged** atom is taxed the desolvation cost
**only when UNSATISFIED** (no H-bond/salt partner N/O within 3.5 Å across the interface); a satisfied one gets the tax
waived.

**Honest result on the complex-held-out alanine task (n=1156):**
- **The physics validates.** Hydrophobic-buried area tracks measured ΔΔG at **r +0.38**, and the satisfaction correction
  genuinely separates **satisfied**-polar burial (r +0.25) from **unsatisfied**-polar burial (r −0.05) — two distinct
  signals the crude blanket-penalty term collapsed into one.
- **It fixes the crude term's flaw** — it no longer *hurts*.
- **But it does not improve held-out accuracy**: baseline (research + sterics + elec/induction) **0.522 → +EEF1-desolv
  0.522 (+0.000)**. The hydrophobic-burial signal is already captured by the vdW + buried-contact features, and the
  alanine scan (which *removes* a sidechain) is a weak place for a desolvation term to add orthogonal value.

**Where it *would* matter:** the satisfaction penalty's designed strength is scoring a mutation that **introduces** a
buried *unsatisfied* polar/charged group — which needs the mutant's SASA (rotamer-placed), a bigger build and the honest
next step. So this is a **correct, more physical, no-longer-harmful** desolvation term that I'm **not** folding into the
headline (0.528 stays) — same discipline: keep only what measurably helps, and report the rest straight. (desolv_eef1.py
→ desolv_eef1.json; `desolv` / `desolv run`.)

### Adding real MSMS-style surface + APBS electrostatics to the DL model — the physics HELPS (+0.068)

The coarse surface encoder above topped out at AUC 0.66 and I named the reason: no molecular surface, no electrostatics.
Asked to add MSMS surfaces + APBS, I installed the real tools — **APBS** (apt) + **pdb2pqr** + **scikit-image** (pip) —
and built surface_apbs.py (the `surface apbs` syscall):
- **Electrostatics (real APBS):** pdb2pqr assigns AMBER charges + protonates → APBS solves the **Poisson-Boltzmann**
  equation → the electrostatic-potential grid (kT/e), sampled at each patch. This is the exact APBS pipeline requested.
- **Molecular surface:** a **marching-cubes** surface over an atomic Gaussian-density grid (the dMaSIF-style substitute
  for the MSMS *binary*, which isn't installable here) → real surface curvature/planarity at each patch.

**Controlled head-to-head** — identical patch framework, identical complex-held-out split, identical contrastive
encoder as the coarse model; the *only* change is enriching each patch's features with the APBS potential +
molecular-surface geometry, so any AUC change is attributable to the real physics:

| Features | Held-out discrimination AUC (60 complexes) |
|---|---|
| coarse residue (geometry-from-atoms + chemistry-from-identity) | 0.645 |
| **+ real APBS electrostatics + marching-cubes surface** | **0.713 (+0.068)** |

**So the real physics genuinely helps** — the APBS potential + true surface shape carry interface-complementarity signal
beyond residue-identity chemistry, closing part of the gap to real MaSIF, exactly as predicted. This is a *positive*
result (unlike the desolvation retest, which was redundant) — worth keeping.

**Honest limits:** coarse APBS grid (dime 65) for speed; a Gaussian marching-cubes surface, not an exact MSMS
solvent-excluded surface; a **60-complex subset** (APBS costs ~5s + a few MB each — stated, not hidden), re-running the
coarse baseline on the *same* subset for a like-for-like number; and residue-centre patches, not per-surface-point
geodesic patches. The tooling is now **real** (APBS PB potential + marching-cubes surface); the remaining gap to
MaSIF-grade is the fine surface-point patch representation — the honest next step. (surface_apbs.py → surface_apbs.json;
`surface apbs`.)

### Scaling to all the data — fetched the full SKEMPI structure set (345 complexes) and re-ran

Asked to expand to more proteins, I fetched every complex SKEMPI references — **205 missing PDBs, 0 failures → 345
total** — and re-ran both models on the full set. Two honest findings:

- **Coarse model: more data does NOT move the ceiling.** Trained on 344 complexes (134,745 patches, 2.3× the data),
  held-out discrimination AUC is **0.653** — statistically identical to the 0.66 on 139. The coarse residue features are
  *feature-limited*, not data-limited.
- **APBS + surface model: the gain GREW with data.** On **333 complexes** (12 failed pdb2pqr/APBS, reported), coarse
  **0.652 → +APBS electrostatics + marching-cubes surface 0.760 (+0.109)** — a *bigger* lift than the +0.068 on 60. So
  the richer physics-based features both start higher and benefit more from scale, while the coarse model plateaus.

| Model | 60–139 complexes | 333–344 complexes (all data) |
|---|---|---|
| coarse residue features | 0.645–0.66 | 0.653 |
| + real APBS electrostatics + surface | 0.713 (+0.068) | **0.760 (+0.109)** |

The lesson is clean and honest: **the bottleneck was feature quality, not dataset size** — real electrostatics + surface
geometry is what scales, and it scales *better* with more data. (surface_apbs.py → surface_apbs.json; `surface apbs`.)

## dmasif / dmasif — Geodesic Surface Learning: the fine surface-point model closes the gap

surface_apbs flagged that its ceiling was **residue-centre** patches. So I built the real dMaSIF approach (dmasif.py →
the `dmasif` syscall): the primitives are **surface POINTS** (from a marching-cubes molecular surface with oriented
normals), each carrying **real APBS Poisson-Boltzmann potential** sampled at the point + local curvature + nearest-residue
chemistry; the model is a **learnable quasi-geodesic convolution** — for each point, gather its K nearest surface
neighbours, express them in the point's **local tangent frame** (two tangent axes + normal), MLP over
[local-coords, distance, neighbour-features], Gaussian-weighted by a **learned range**, stacked twice → a per-point
embedding. Trained contrastively (true interface point-pairs close, non-contacting far), held out by complex.

**Result on 70 complexes — a real, large gain:**
- **Discrimination AUC 0.846** — vs residue-patch+APBS **0.76**, vs coarse residue **0.65**.
- **Retrieval — the test the residue model essentially *failed* (~1.4× chance) — is now solved-enough to be useful:**
  top-1 **7.5%** (chance 0.4%), top-5 **23.2%** (chance 1.8%) = **~13× chance**. The finer surface-point resolution +
  geodesic convolution genuinely **pinpoints the actual contacting point**, which residue centres could not.

**The full honest arc, each step a real measured gain from the right representation/physics:**

| Model | Discrimination AUC | Retrieval (top-5 vs chance) |
|---|---|---|
| the mock (random weights on random data) | — | fake "99% confidence" |
| coarse residue features | 0.66 | ~1.4× (fails) |
| + real APBS electrostatics + surface (residue patches) | 0.76 | ~1.4× |
| **geodesic surface points + learned conv (dMaSIF)** | **0.85** | **~13×** |

**Honest limits:** subsampled surface points (~1500/complex), a compact 2-layer geodesic net on CPU (not the full
multi-scale dMaSIF at native density on GPU), a marching-cubes Gaussian surface (not exact MSMS), a 70-complex subset.
The representation and physics are real, and the comparison to the residue-patch model is reported straight. (dmasif.py →
dmasif.json; `dmasif` / `dmasif run`.)

### Complementary layers, tested honestly: fusion of two EXTRINSIC layers is redundant → NEXUS needs intrinsic × extrinsic

Asked whether dMaSIF can *replace* or *augment* the ΔΔG node, I built the leakage-free fusion test (complementary.py):
feed dMaSIF's geodesic **surface embedding at the mutation site** as extra features into the ΔΔG-binding node, with
**dMaSIF trained only on TRAIN complexes** and ΔΔG scored on **held-out TEST complexes** (3 splits, node-only baseline on
the same splits). Result on **82 complexes / 2,456 alanine mutations**: node-only Pearson **0.542 → node+dMaSIF 0.476
(Δ −0.066)** — fusing them **hurts**.

The reason is structural, not a bug: **`flex_physics` is a *binding* (extrinsic) predictor and dMaSIF is *also* extrinsic
surface complementarity**, so the 32-d embedding re-expresses interface geometry the node already encodes (buried
vdW/contacts) and adds variance the RF overfits. Two extrinsic layers are **redundant, not complementary**. The honest
architectures are (a) a **pipeline** — dMaSIF *localises* which interface/partner, the ΔΔG node scores the *magnitude* —
or (b) the **NEXUS intrinsic × extrinsic** pairing: a *folding-stability* sensor AND a *binding* sensor, which catch two
orthogonal failure modes and genuinely stack. Fusion-of-two-extrinsics does not. (complementary.py → complementary.json.)

## nexus / nexus — the dual-sensor enzyme-health node (intrinsic × extrinsic), with the three refinements

The NEXUS design — two orthogonal ways a protein machine dies, each with its own sensor, combined into a graded activity
that drives the metabolic model — is right, and the complementary.py result proved *why*: fusing two **extrinsic** layers
was redundant, so the real pairing is **intrinsic × extrinsic**. Built it (nexus.py → the `nexus` syscall) from parts
already on disk (ddg_predictor stability node, flex_physics binding, ecflux FBA), with the three refinements:

- **#1 Soft AND, not Boolean:** `activity = folded_fraction(ΔΔG_fold) · bound_fraction(ΔΔG_bind)` — a product of two
  two-state equilibrium fractions in [0,1], so partial damage → partial activity.
- **#2 Extrinsic = loss only:** `bound_fraction` never rewards a mutation; no neomorphic-gain claim.
- **#3 Validate, don't assume** — and it holds on **measured, non-circular** data.

**The "you need both" thesis, measured (808 SKEMPI interface mutations):**
- The two failure modes are **orthogonal**: Pearson(predicted ΔΔG_fold, measured ΔΔG_bind) = **0.15**. A stability-only
  node **misses 94%** of the strong interface breakers — it literally cannot see them.
- **Fusion value (complex-held-out CV):** classifying a real interface failure from the intrinsic sensor **alone is
  near-chance (AUC 0.56)**; adding the extrinsic sensor (flex_physics interface features) lifts it to **0.75 (+0.19)**.
  The second sensor is what catches the failure the first is blind to. This is non-circular (the extrinsic sensor is a
  *prediction*, cross-validated — not the measured label).

**Metabolic consequence (HumanGEM, an essential reaction):** the activity→flux mapping **works** — catastrophic damage
→ activity 0.02 → biomass **0.04** (it propagates). But **moderate** damage is **buffered**: the two-state fractions
saturate (a single mutation rarely fully unfolds a protein) and metabolic excess capacity absorbs moderate loss. That's
real physics, not a bug.

**Honest boundary:** the two **sensors** live in near-field **structure** — the recoverable regime, which is why they
work and why the orthogonality is measurable. The last link — activity → FBA → whole-cell **phenotype** — is the
**far-field dynamics** step this whole project keeps finding does *not* compose. So the metabolic step is a **wiring
demonstration**, not a validated phenotype predictor; validating it needs measured IEM/essentiality outcomes. The
dual-sensor node is real and the orthogonality that makes it necessary is measured; the structural half is trustworthy,
the phenotype half is the honest open edge. (nexus.py → nexus.json; `nexus` / `nexus run`.)

## regsign / regsign — the regulatory-sign annotation: the GAIN-of-function lever (high-precision, annotation-limited)

The structural sensors detect an interface **break** but not its **direction**. The missing piece for gain-of-function
is the interface's **sign**: break an *activating* interface → loss of function; break an *inhibitory / autoinhibitory
brake* → **gain** of function (the brake is gone, activity up). Built it (regsign.py → the `regsign` syscall) from real
UniProt **Activity-regulation** text (autoinhibition / intramolecular-repression / inactive-conformation cues) and tested
it against real GOF/LOF ground truth (UniProt **proto-oncogene** vs **tumor-suppressor** keywords).

**Result, read the fair way** (a rare, high-confidence flag — precision/recall, not AUC):
- **When the sign fires** (a brake is annotated), the gene is GOF with **86% precision** (19 oncogenes vs 3 tumor
  suppressors), a **5.4× enrichment** — and the hits are textbook autoinhibited oncogenes: **ABL1** ("stabilized in the
  inactive form"), **KIT/PDGFRA/FGFR2** ("inactive conformation in the absence of ligand"), **BRAF** ("maintained in an
  inactive state via an intramolecular interaction").
- **Its limit is recall:** only ~**10%** of GOF genes have the brake annotated in this one source (which is why the AUC
  is only 0.54 — dragged down by the un-annotated 0s, *not* by wrong calls).

So the regulatory sign is the **correct and necessary** lever for GOF, it carries **real directional signal** (high
precision where it fires), and its bottleneck is **annotation coverage** — an **information gap, not a compute gap**,
exactly as predicted. That's why running the structural sensors on all 20k proteins couldn't have solved GOF: scale adds
coverage of *breaks*, but the *sign* is separate information.

**Wired into NEXUS** via `directed_activity(inhibitory=True)`: a mutation breaking a brake-bearing interface pushes
activity **up** (GOF) instead of down, still **gated by folding** (an unfolded protein can't gain function either — a
broken brake on a misfolded protein correctly stays LOF). **Still out of reach:** *neomorphic* GOF (a brand-new
interface) — needs docking + experiment. **Raising recall** means broadening the sign sources (SIGNOR, structural
autoinhibition annotations) — more information, not more compute. (regsign.py → regsign.json; `regsign` / `regsign run`.)

## nexus_train / nexus_colab.ipynb — train NEXUS at scale on human PPIs (Colab), sandbox-verified

To take NEXUS from a set of validated pieces to something trainable at scale, built a portable driver (nexus_train.py)
and a Colab notebook (nexus_colab.ipynb). The **trainable component is the dMaSIF geodesic surface model** (the extrinsic
sensor — the one we measured keeps improving with more complexes); the intrinsic stability node and the regulatory-sign
GOF layer are fixed/annotation.

**How many human PPIs / where to get them** (documented in the notebook):

| Layer | Count | Source |
|---|---|---|
| Binary PPIs (high-quality) | ~53,000 | HuRI |
| All curated interactions | ~0.6–1M | BioGRID, IntAct, STRING (physical) |
| **PPIs with a 3D complex structure** ← the sensor needs this | **a few thousand** | PDB, Interactome3D, SKEMPI |
| Monomer structures (intrinsic sensor) | ~all 20,000 | AlphaFold DB |

The binding sensor's bottleneck is **complex structures** (thousands, not 20k) — to scale further you predict complexes
with **AlphaFold-Multimer** and add them.

**The driver** patches the module paths to a portable cache (so it runs anywhere, not just the author's sandbox),
fetches complexes from RCSB, builds surface clouds (marching-cubes + APBS), trains the geodesic net **held out by
complex**, saves the weights, and runs the full sensor stack (intrinsic ΔΔG_fold + extrinsic ΔΔG_bind → LOF/GOF activity)
on real mutations as a working-check.

**Sandbox-verified before shipping** — ran it on 5 complexes: fetch → surface+APBS → trained dMaSIF (held-out
interface-discrimination AUC **0.664**) → sensor stack ran end-to-end → **WORKING-CHECK PASS**, all in ~70s on CPU. The
notebook adds a GPU scale-up cell (downloads SKEMPI, trains on all ~345 measured complexes) and accepts any user PDB list
(incl. AlphaFold-Multimer complexes). (nexus_train.py, nexus_colab.ipynb → nexus_train.json / nexus_dmasif.pt.)

### Auto-fetcher: scale past SKEMPI's 345 to the ~34,000 PDB complexes

345 (SKEMPI) is only the *labelled* set. The dMaSIF surface sensor trains on interface **geometry** — self-supervised on
cross-chain contacts, **no ΔΔG labels** — so it can consume every protein-protein complex in the PDB. Added
`fetch_complexes.py`: it queries the **RCSB Search API** for heteromeric protein complexes (resolution ≤3Å, with an
atom-count cap so ribosomes/viruses don't choke APBS, optional `human_only`), and downloads them. The PDB has
**~34,000** such complexes — 100× SKEMPI.

`nexus_train.fetch_and_train(n, cache, epochs, device, human_only=)` does it in one call: auto-fetch n complexes → build
surfaces → train dMaSIF (held out by complex) → run the sensor stack. **Sandbox-verified** with `fetch:8`: RCSB search →
downloaded 8 → 7 usable surfaces → trained dMaSIF (held-out AUC **0.76**) → sensor stack ran → **WORKING-CHECK PASS**.
(The sensor demo falls back to a fixed SKEMPI complex when the auto-fetched ones have no measured mutations — the sensor
stack is independent of which complexes the surface model trained on.)

The Colab notebook now features the auto-fetcher as the scale path (`fetch_and_train(300, human_only=True)`), keeps
SKEMPI as the labelled validation set, and still accepts any user PDB list (incl. AlphaFold-Multimer complexes for PPIs
without solved structures). (fetch_complexes.py, nexus_train.py, nexus_colab.ipynb.)

## fusion_test — does the STRONG (0.947) dMaSIF embedding lift the ΔΔG node? Re-run of "complementary layers"

The earlier `complementary.py` found the dMaSIF surface embedding did NOT improve the physics ΔΔG node — but it used a
WEAK per-split net (trained on ~60 complexes). After the surface net reached **0.947** held-out interface AUC on 2,198
complexes (Drive run), the fair question is whether a *strong* embedding changes that verdict. Built `fusion_test.py`:
loads a FIXED pretrained dMaSIF net, pools its geodesic per-point embedding at the mutation site (8Å), concatenates it
onto the physics node features, and compares node-only vs node+embedding under **GroupKFold-by-complex** (every mutation
of a complex in one fold; node-only baseline on the SAME folds). The net is a self-supervised geometry feature extractor
that never saw ΔΔG labels, so using its embedding is pretrain→probe, not leakage; `probe_pdbs` forces the ΔΔG-test
complexes to be disjoint from the net's training set for a strict check.

**Sandbox result** (net trained on 30 SKEMPI complexes → held-out AUC **0.789**; probed on **12 disjoint held-out
complexes / 243 alanine mutations** the net never saw):
- ΔΔG magnitude (Pearson): node-only **0.346** → node+dMaSIF **0.321** (**Δ −0.025**)
- binding-hotspot ΔΔG>1 (ROC-AUC): node-only **0.744** → node+dMaSIF **0.717** (**Δ −0.027**)

So even a strong surface embedding does **not** improve mutation-effect prediction — it slightly *hurts*. This confirms
the structural prediction and is not a data-quantity problem: **dMaSIF embeds WILD-TYPE surface geometry, and a point
mutation's effect is a mutant-minus-WT DELTA the WT embedding cannot express, however good the embedding is.** A better
WT surface sensor is still a WT surface sensor. (Note the node-only hotspot AUC **0.744** matches the ~0.75 NEXUS binding
number; the surface embedding adds nothing on top.)

**Honest verdict:** the architecture stays a **PIPELINE, not a fusion** — dMaSIF finds/scores the interface and the
right partner (its 0.947 strength is real and useful *there*); the physics node scores the specific substitution. The
0.947 upgrade sharpens interface *recognition*, not mutation-*effect* prediction — the two are orthogonal by construction.
The definitive check on the exact 2,198-complex / 0.947 net is the one-cell Colab runner (`fusion_test.run` pointed at the
Drive weights); theory says it will give the same "no lift" result, since the limitation is *what* the embedding encodes,
not how well. (fusion_test.py → fusion_test.json.)

## nexus_cell / `nexus GENE ...` — NEXUS wired INTO the running cell (live mutation → LOF/GOF → cell dependency)

The nexus.py node was validated in isolation (dual-sensor orthogonality; regsign direction). This makes it a LIVE
CellOS query — `nexus GENE [UNIPROT [POS WT MUT [PDB CHAIN]]]` — that reasons a real mutation through four
already-validated layers and hands the result to the cell's own essentiality reasoner, so the cell can now answer a
question it could not before: **given a mutation, is it LOSS or GAIN of function, does it break FOLDING or BINDING, and
does the cell DEPEND on the protein?**

- **DIRECTION** (regsign, structure-free): fetches the gene's UniProt *Activity regulation* text live → a breakable
  inhibitory brake ⇒ **GOF-capable**, else **LOF-only**. Works on any human gene, no structure. Verified live:
  BRAF/ABL1 ("maintained in an inactive state" / "stabilized in the inactive form") → GOF-capable; TP53 → LOF-only
  (no brake — a tumor suppressor); KRAS G12D → GOF (fires on the "inactive form bound to GDP" cue — the right
  direction, though that cue is nucleotide-state, not classic autoinhibition: an honest heuristic edge, flagged).
- **INTRINSIC** (fold ΔΔG): ddg_predictor on the structure — does it still fold?
- **EXTRINSIC** (bind ΔΔG): the **measured** SKEMPI value when the mutation is one we have (honestly labelled
  `[measured (SKEMPI)]`, not a prediction) — abstains otherwise (a calibrated binding-ΔΔG predictor for arbitrary
  mutations is the physics node at r~0.52, not exposed as a scalar — the honest current limit).
- **ACTIVITY**: nexus.directed_activity(fold, bind, inhibitory=sign) — soft-AND for LOF, brake-release for GOF.
- **CELL**: the kernel's own reason() essentiality (calibrated P) as the dependency CONTEXT.

**Full-pipeline demonstration** (the dual-sensor thesis, live on one mutation): **GHR B/W304A** — the classic
hGH–receptor hot spot (1A22) — comes back **fold ΔΔG +0.64 (still folds) but bind ΔΔG +4.73 = ~2158× weaker binding**:
a loss-of-function that lives entirely on the **BINDING axis while FOLDING is intact** — exactly the orthogonality the
two-sensor node exists for (a fold-only sensor is blind to it). Cell: NON-ESSENTIAL (P≈4%), so the loss is tolerated —
reported as CONTEXT.

**Honest boundary** (unchanged): the structural sensors are near-field (solid); the cell-dependency is CONTEXT, not a
validated phenotype (activity→phenotype is far-field/buffered, and the soft-AND occupancy SATURATES at high WT affinity
— the ΔΔG is the informative readout, not the occupancy number). Where no structure/partner is given the structural
sensors abstain and the query still returns the structure-free regulatory direction + cell dependency. Wired as the
live `nexus GENE ...` syscall; the bare `nexus` report and `nexus run` are unchanged. (nexus_cell.py → nexus_cell.json.)

## bind_ddg — the extrinsic magnitude sensor becomes a PREDICTOR (all 20 AAs); and ESM is the wrong axis for binding

The extrinsic (binding) axis used to be a measured-value lookup — on any mutation not in SKEMPI it abstained. Built a
trained predictor so it works on **any** interface mutation on a complex, from **structure-derivable features only**:
property deltas + the WT sidechain's interface contribution (LJ/contacts/Coulomb/induction) + the MUTANT sidechain's
**best precomputed-χ-well rotamer** clash + a **charge-superposition** field. RandomForest, held-out **by complex**.

- **Accuracy (46-complex sandbox set, held-out by complex): all-AA r 0.46, alanine 0.51, non-alanine 0.30, hotspot
  (ΔΔG>1) AUC 0.79.** Wired into `nexus_cell` as the extrinsic sensor: measured SKEMPI value when available, else the
  predictor (labelled `predicted (bind_ddg, r~0.46)`). (bind_ddg.py → bind_ddg_model.pkl / bind_ddg.json.)

**The honest experiments behind the feature choice (measured, held-out by complex):**
- **All-AA de-biasing lowers the number, doesn't raise it.** Alanine-only r 0.55 → all-AA 0.49 → **non-alanine alone
  0.22**. Alanine scanning is the *easy* subtractive case; the alanine bias was flattering us for real disease variants.
- **Best precomputed rotamer helps the hard cases; the Boltzmann ensemble does NOT.** Building the mutant sidechain and
  placing it in the best χ-well lifts non-ala 0.22 → 0.26; integrating the binding energy over the whole rotamer
  ensemble (log-sum-exp free energy + Boltzmann mean + flexibility count) is **no better than the single best well**
  (0.244 vs 0.246) — a clean negative. Charge superposition adds a little more (→ ~0.30). On the **alanine** set these
  mutant features are correctly **inert** (0.551 → 0.543) — a passing negative control (alanine has no rotamer/charge).
- **ESM does NOT help binding — and it's illuminating.** ESM-2 WT-marginal log-likelihood-ratio, fused with physics:
  all-AA 0.460 → 0.468 (nothing); **ESM alone r=0.09** on ΔΔG-binding (0.02 on non-ala). Two reasons, both the
  dual-sensor thesis restated: (1) ESM reads a **single chain** — it is *partner-blind* by construction, so it cannot
  see a binding interaction; (2) evolutionary tolerance is the **fold/fitness axis**, which NEXUS measured is orthogonal
  to binding (Pearson 0.15). So ESM belongs on the **intrinsic fold sensor** (pathogenicity/stability), not the binding
  sensor. This corrected an earlier over-claim that ESM was the magnitude lever — it is the lever for the *other* axis.

Net: binding magnitude is near its honest ceiling with our tools (~0.46 all-AA / ~0.30 non-ala); the remaining untested
lever is a *partner-aware* structural net fine-tuned end-to-end on ΔΔG (right axis), not ESM. (esm_channel/rotamer_ensemble
experiments in scratch; the committed product is bind_ddg.)

**The last lever, tested and dead (bind_ddg_e2e.py):** the one untested option for binding magnitude was a *partner-aware*
geodesic surface encoder fine-tuned **end-to-end** on ΔΔG (the surface is at least on the right axis — structural,
sees the partner — where ESM was on the wrong one). Three readouts, held-out by complex, 42 complexes / 2,269 muts:
**phys all-AA 0.42, surf-alone 0.10, both 0.33.** Decisive negative — the surface encoder alone barely predicts binding,
and adding it to physics **hurts** (0.42 → 0.33, *worse* than the frozen fusion_test's −0.03). The reason is not small
data, it's **mutation-blindness**: the surface cloud is the *wild-type* surface — **identical** for W304A vs W304G — so
the encoder can only see the *position* (redundant with physics burial), never the *substitution*. More data can't add
information the input doesn't contain. So the surface is confirmed (a second, stronger time) to belong on **Part-1
recognition, not Part-2 magnitude**, and **physics is the honest ceiling** for binding magnitude (~0.46). `bind_ddg_colab.ipynb`
runs the full-SKEMPI (~345) physics retrain (the production model) + this e2e test at scale on GPU. (bind_ddg_e2e.py → bind_ddg_e2e.json.)

**Full-SKEMPI result — and a correction to my own conclusion (Colab, held-out by complex):** trained bind_ddg on the
**full 277-complex / 4,417-mutation SKEMPI set** (vs the 42-complex sandbox subset). Result: **all-AA r 0.51,
non-alanine 0.51, alanine 0.51, hotspot AUC 0.77.** Two findings: (1) the number rose 0.46 → **0.51**, solidly in
Rosetta flex_ddG territory (~0.6), a real held-out-by-complex number on a large set; (2) **the "non-alanine is
intrinsically harder" conclusion was WRONG.** On 42 complexes non-ala was 0.30 and I attributed it to difficulty
(subtractive+additive substitutions, "alanine bias flatters us"). The full set proves it was **data-starvation**: 42
complexes held only ~358 non-ala examples; the full set has 1,686, and non-ala jumps to **0.51, dead even with
alanine.** The physics+rotamer+charge features predict general substitutions *as well as* alanine given enough data to
fit them — the substitution-class gap vanishes at scale. The through-line holds: **the ceiling was DATA, not the model
or the features** — scaling 42 → 277 complexes did what ESM, end-to-end nets, rotamer ensembles, and charge
superposition all could not. (Production model on Colab; the committed .pkl is the 42-complex fallback.)

**Multi-mutant augmentation — a free scope-add (measured):** SKEMPI 2.0's 7,085 entries are ~4,900 single-point +
~1,973 MULTI-point (2+ substitutions, excluded from the single predictor as a different/epistatic task) + ~200 without
clean Kd. Tested whether adding the multis helps, representing a multi-mutant by SUMMING its per-site features + an
`n_sites` feature (63-complex sandbox, held-out by complex): (A) single-only→single 0.454; (B) single+multi→single
**0.450** — adding multis is NEUTRAL to the single prediction NEXUS uses (as expected, different distribution); (C)
single+multi→multi **Pearson 0.38 / hotspot AUC 0.76** — we CAN predict multi-point mutations, nearly as well as single
on the hotspot call. So it's not a win for the single sensor but a **free scope expansion** (engineered / combinatorial
variants) at zero cost to single. Shipped as a JOINT model: bind_ddg now trains on single+multi with the n_sites feature
and exposes `Predictor.predict` (single) + `Predictor.predict_multi(pdb, sites)`; the single number is unchanged and
`r_multi` is reported separately. (bind_ddg.py — features_agg / _load_all / predict_multi.)

## graph_label / `kg` — the LABELED multi-relational cell graph (ppi + pathway + literature)

The CellGraph adjacency already carried physical/functional edges (ppi, reg, sig, codep, lr) but was missing two
relations: **pathway co-membership** and **literature co-mention**. Added both and labelled every edge by its source
relation, so the graph says *how* two genes relate, not just *that* they do:
- **ppi** — physical interaction (191,447 edges over 14,230 genes)
- **path** — same Reactome pathway (208,881 edges over 7,282 genes; capped ≤50 genes/pathway so giant generic
  pathways like *Metabolism* don't hairball)
- **lit** — co-mentioned in a paper (litmine shared PMID; 4,395 edges over 2,677 genes)

Built graph: **16,492 nodes, 15,103 with ≥1 labeled edge, 383,246 unique labeled pairs.** The three relations are
**complementary** — cross-type overlap ppi&path 0.11, ppi&lit 0.03, path&lit 0.00 — i.e. each edge type says a
*different* thing about a gene pair (physical ≠ same-pathway ≠ co-mentioned), not redundant. Wired as the `kg` syscall:
`kg` = whole-graph label stats, `kg GENE` = a gene's typed neighbourhood.

**HONEST SCOPE (unchanged from the edge-predict finding):** this is a **descriptive near-field knowledge graph**. These
edge types predict a gene's knockout effect at ~floor (r 0.03–0.09; pathway 0.064; only curated complexes 0.23), so the
labeled graph is the right substrate for **representation / link-prediction / querying / the near-field GNN**, and it is
honest about **not** composing to phenotype. (graph_label.py → graph_label.json; `kg` syscall.)

## spatial_ladder / `ladder` — pathways laid out by compartment (a measured NO on the directional ladder)

The idea (the user's): lay a pathway out as a **trajectory through compartments** — a signalling pathway that starts at
the membrane and ends in the nucleus would tell you which reactions happen in which compartment, *in order*. We have both
ingredients: an ORDER (cell_levels: signalling tier upstream→downstream) and a COMPARTMENT for essentially every protein
(localization, ~100% coverage). So: does the order track the compartment depth?

**Reality-checked first, and the answer is NO.** Across 4,155 signalling-tier genes with a compartment, the regulatory
tier does **not** track membrane→nucleus depth:
- nuclear-annotated vs membrane-annotated signalling genes have **near-identical tier means** (0.68 vs 0.72, difference
  **−0.039** — if the ladder were real this would be large and positive);
- the tell: there are **more TFs among UPSTREAM genes (24%) than downstream (8%)** — the *opposite* of a
  membrane→nucleus flow ending at nuclear TFs.

The cell_levels "tier" is a **regulatory-graph order, not a spatial one**, and every large pathway spans all four
compartments anyway. So the directional ladder is a just-so story and is **not** claimed.

**What IS delivered (honest):** a per-pathway compartment **LAYOUT** read directly from localization. A subtlety that
mattered: the localization source lists **every** compartment a protein is seen in, **not in primacy order** (HRAS is
tagged `Nucleus, Golgi, Cell membrane, Cytoplasm` all at once; taking the first entry would file the membrane GTPase
HRAS — and EGFR — under "nucleus"). So each protein is counted in **every** layer it's annotated in (multi-compartment
membership), not one arbitrary pick — giving a compartment **PROFILE / occupancy view** plus per-layer gene lists,
depth-ordered outside→in for readability only. Example: *Signaling by EGFR* → membrane 38 · cytoplasm 36 · nucleus 11 ·
extracellular 7.

Wired as the `ladder` syscall: `ladder PATHWAY` (fuzzy name) shows one pathway's compartment layers, `ladder GENE` shows
a gene's compartments + the pathways it's in, `ladder` shows the reality-check. **HONEST SCOPE:** curated/observed
annotations joined together (near-field descriptive) — an occupancy profile, **not** a directional route, **not** a
reaction-localization claim, and **not** a phenotype predictor. (spatial_ladder.py → spatial_ladder.json; `ladder`
syscall.)

## lit_place — placing orphan / dark genes from their literature (extends `ladder GENE`)

Follow-on to the ladder: **the ladder needs a pathway, and 6,003 genes aren't in one** — but 4,708 of those dark/orphan
genes have literature abstracts (litmine). Can we place *them*? Two halves, with very different confidence:

- **Compartment — solid.** An orphan gene's compartment is read *directly* from localization (~100% coverage, dark
  genes included). No prediction. This alone slots the gene into the compartment layers.
- **Functional context — a validated but noisy hypothesis.** Mine the gene's abstracts for co-mentioned *known* genes
  (case-sensitive symbol regex + an English-word stoplist so SET/MET/CAD aren't false hits), then transfer those known
  genes' Reactome pathways by an **IDF-weighted vote** (IDF down-weights promiscuous hubs like TP53 that co-occur with
  everything, so they don't dominate).

**Validated leave-one-out** (the load-bearing assumption "genes co-mentioned in one abstract share a pathway"): build a
known–known co-mention graph from the abstracts, hide a known gene's pathway, predict it from its co-mentioned
neighbours. Result: **pathway top-1 ~0.13 / top-5 ~0.21 vs random-from-candidate-pool ~0.055 = ~3.5–4× chance**
(IDF-weighted beats raw-count and binary). So literature co-mention is a **real but noisy** pathway signal — a
*hypothesis generator* (the right pathway lands in the top-5 about 1 in 5 times), **not** a confident annotation and
**not** a phenotype predictor. (The compartment half is trustworthy precisely because it's *read*, not predicted.)

**The hypotheses are genuinely right when checked:**
- `FNDC5` (irisin) → co-mentioned UCP1 / PGC-1α / BDNF → *brown & beige adipocyte differentiation / mitochondrial
  uncoupling* — correct thermogenesis biology.
- `STRA8`, `DAZL` (germ-cell genes) → *Meiotic recombination* — STRA8 is literally the meiosis gatekeeper.
- `TNFRSF19` (TROY), `SP5` → *Signaling by WNT / β-catenin* — both are genuine Wnt-pathway genes.

Wired into `ladder GENE`: an orphan gene now returns its **known compartment + literature co-mentions + a ranked
candidate function**, so the compartment-ladder covers genes with no pathway annotation at all. **HONEST SCOPE:** curated
regex mining (not full NER), one source capped at ≤6 papers/gene, descriptive near-field — a starting hypothesis for an
uncharacterised gene, to be confirmed experimentally. (lit_place.py → lit_place.json; `ladder GENE`.)

## connect — triangulate literature + PPI + spatial into one mechanistic hypothesis (`connect` syscall)

The user's next step: don't stop at literature + compartment — *connect*. If a paper puts an orphan gene in pathway X,
find its physical (PPI) partners, see where they come from and what reactions they run, verify the locations against each
other, and ask whether some partner could *carry* the gene between compartments to where the reaction happens.

The honest core is **cross-source agreement as the confidence signal**, and it validates decisively (held-out on known
genes, n=5,409, pathway top-1 precision):

| signal | top-1 precision | fires |
|---|---|---|
| literature alone | 0.125 | 100% |
| PPI partners alone | 0.538 | 98% |
| **corroborated (lit AND ppi agree)** | **0.706** | 18% |

So a physical partner's pathway is already a *much* stronger vote than literature (0.54 vs 0.13 — PPI is measured, and
interacting proteins share function), and when the literature co-mention **and** a physical partner independently point
to the *same* pathway, the call jumps to **71% precise — 5.6× literature alone**, firing for ~18% of genes. That
agreement is the thing that turns a noisy guess into a confident lead.

**The spatial cross-check is grounded too:** PPI partners share ≥1 compartment **0.927 vs 0.507 for random pairs = 1.83×**
(interacting proteins co-localize, as they must). So verifying an orphan's location against its partners is sound — and
the ~7% of partners that *don't* co-localize are exactly where a transport step is needed.

**Transport reasoning is a flagged heuristic** (not validated): when the gene and its inferred reaction sit in different
compartments, `connect` flags a partner annotated `transport/uptake` or `trafficking/secretion` whose compartments bridge
the gap — a candidate carrier to test, clearly labelled as a lead.

**The cards are biologically coherent when checked:**
- `WDHD1` (AND-1) → **corroborated** *DNA strand elongation / unwinding* with its **MCM2-7 helicase** partners; gap =
  cytoplasm, bridged by KPNA1 (importin-α imports replication factors) — textbook replication-fork biology.
- `CSE1L` (CAS) → nuclear-transport context via KPNA1 / NUP50; flagged carrier **KPNA1** — CSE1L *is* the exportin that
  recycles importin-α. Correct machinery.
- `UNC13C` → *Neurotransmitter Release Cycle* via UNC13B (Munc13 vesicle priming).
- `STRA8` → honestly **no corroboration** (its partners don't match the meiosis literature), so it falls back to
  lit-only — no forced high-confidence call.

Confidence **tiers** keep it honest: measured PPI > corroborated > lit-hypothesis > transport-heuristic. Wired as the
`connect GENE` syscall. **HONEST SCOPE:** descriptive near-field triangulation — a strong *starting mechanism*
(corroborated ~71% precision) to test experimentally, **not** a proven annotation and **not** a phenotype predictor.
(connect.py → connect.json; `connect` syscall.)

## pathway_graph — five pathways as one compartment-layered crosstalk graph (viz)

A visual answer to "make a graph of 5 pathways and how they connect, top-to-down and interconnected." Built entirely
from the model's real data (`viz/pathway_crosstalk.html`, generator `colab/pathway_graph.py`): the connective proteins of
five Reactome pathways — **EGFR receptor, MAPK activation, MAPK nuclear targets, PI3K/AKT, WNT** — laid out as columns,
positioned **top-to-down by compartment** (localization: extracellular → membrane → cytoplasm → nucleus) and
**interconnected by measured PPI** (cell_complete edges). 41 proteins shown, **14 shared** (in two pathways at once — the
literal hand-off nodes), 113 within-pathway and 21 across-pathway edges.

The story the data tells is the textbook cascade: EGFR at the membrane → the **RAS relay** (HRAS/KRAS/NRAS/GRB2, drawn as
shared nodes bridging EGFR and the MAPK arm) → splits into the MAPK and PI3K arms → converges on **ERK (MAPK1/MAPK3)** →
nuclear MAPK targets; WNT crosstalks near the nucleus through the **PP2A phosphatase (PPP2*)**. The shared nodes are
exactly the right biology — they weren't hand-picked, they fell out of pathway co-membership.

**HONEST SCOPE (same as the ladder):** vertical position is the **observed compartment**, not a proven flow; the edges
are **measured but undirected** physical interactions — real crosstalk and shared machinery, **not** a proven directional
signal hand-off. The downward arrow marks the **textbook** cascade direction for these specific pathways, not something
derived from the edges. A descriptive near-field map of who-touches-whom-and-where, not a phenotype predictor.
(colab/pathway_graph.py → viz/pathway_crosstalk.html.)

**Nucleus-rooted view (`viz/pathway_radial.html`):** same five pathways, but rooted at the genome — proteins branch
*outward* from a central nucleus core through cytoplasm → membrane → extracellular, pathways as angular sectors, with
faint spokes tracing each protein back to its gene. This is the **biogenesis** direction (every protein is made from a
gene in the nucleus — the one origin we're *certain* of), the complement and opposite of the inward signalling layout.
Same honest scope: rings = observed compartment, PPI edges measured but undirected, spokes mark the central-dogma origin
not 41 distinct measured edges.

## Testing the nucleus-rooted / compartment-direction idea on `connect` — an honest negative

The radial reframe raised a real question: if compartment/direction is meaningful, does adding it to the `connect`
predictor sharpen the pathway calls (the 0.538 ppi-only / 0.706 corroborated numbers)? Tested four compartment-aware
weightings of the PPI-partner vote, held-out on the same 5,409 known genes:

| PPI-vote weighting | ppi-only top-1 | corroborated top-1 |
|---|---|---|
| baseline (compartment-blind) | 0.539 | 0.706 |
| co-localizing partner ×1 / else ×0.35 | 0.540 | 0.698 |
| only co-localizing partners vote | 0.543 (fires 91%) | 0.702 |
| directional ring-distance weight | 0.542 | 0.696 |

**It does not help — the numbers are flat (±0.004), and corroboration nudges slightly *down*.** The reason is concrete
and measured: **84% of a gene's physical partners already share its compartment.** Interacting proteins co-localize (we'd
already measured 93% for all PPI; 84% here on the held-out set), so the compartment layer is *redundant* with PPI for
prediction — weighting by it just re-weights an already-co-localized set, and the 16% cross-compartment partners it would
down-weight are a mix of noise and real transient/transport edges, so dropping them doesn't help.

This is the same lesson as the dMaSIF+physics redundancy: **two layers that encode the same thing don't compose.** The
compartment/nucleus-rooted layer earns its keep in the *visualization* and in the *transport cross-check* (the ~16% that
DON'T co-localize — exactly where a carrier is needed), but **not** as a prediction booster. `connect` stays as-is; the
honest answer to "does direction change the 71%?" is **no, and here's why.**

## orphan_network — wiring the unknowns into the pathway network (viz)

Ties the whole arc together: take the orphan / dark genes that `connect` places (no pathway annotation) and draw them
*into* the 5-pathway crosstalk network (`viz/orphan_network.html`, generator `colab/orphan_network.py`). Known pathway
proteins are circles (only the anchors the orphans touch are shown); orphan genes are **diamonds**, wired in by their
**measured PPI links** to those anchors. Confidence is colour-coded from `connect`'s tiers: **gold filled diamonds + gold
links = corroborated** (literature and PPI agree on the pathway, ~71% precision), **hollow diamonds + dashed links =
PPI-only** leads. Same compartment layout (columns = pathways, rows = compartments, top-to-down). ~15 orphans wired to
~38 known anchors.

103 orphans link into these 5 pathways; the corroborated ones check out: **ERRFI1** (MIG6, an EGFR feedback inhibitor) →
EGFR; **CSNK1A1L** (a CK1 kinase) → WNT via AXIN1/CTNNB1; **CREB5** (a CREB-family TF) → MAPK nuclear targets; **DISC1** →
WNT via GSK3B. The caption is generated from the actual shown genes so it always matches the graph.

**HONEST SCOPE:** the **edges are measured** (PPI); the **placement is a hypothesis** — gold is the ~71%-precision
corroborated tier, hollow is a PPI-only lead. Vertical = observed compartment. A descriptive near-field map that slots
each unknown next to the machinery it physically touches — a hypothesis to test, not a proven annotation or a phenotype
predictor. (colab/orphan_network.py → viz/orphan_network.html.)

## How accurate are the orphan pathway placements? (honest accounting)

Asked "as a pathway, how accurate are they?" for the orphan_network placements. The honest answer has three parts:

1. **Directly unmeasurable on the orphans.** All 103 orphans that wire into the 5 pathways have `npath=0` — they are in
   *no* Reactome pathway at all (capped or uncapped). By construction there is no internal ground truth for them, so we
   cannot compute a direct accuracy on the actual orphans.

2. **Regime-matched proxy (the graph's real task = "which of these 5 pathway families?").** Leave-one-out on KNOWN genes
   in the *same* operating regime (a gene wired into these 5 pathways by ≥3 PPI links), family-level:
   - **PPI-only assignment: 96%** correct family (132/138)
   - **Corroborated (lit+PPI agree): 85%** correct family (77/91, fires 66%)
   This is much higher than `connect`'s headline 54%/71% because that number is *exact* pathway top-1 among ~2,000
   Reactome pathways, whereas the graph only claims one of **5 broad families** — a far easier call.

3. **Manual spot-check of the 12 shown corroborated orphans, and it's lower than the proxy** (as expected for dark
   genes): ~5–6 clearly correct (ERRFI1→EGFR = MIG6 feedback inhibitor; CSNK1A1L→WNT via LRP5/6/CTNNB1; DISC1→WNT via
   GSK3B; ATF7→MAPK-nuclear with ATF2/JUN/FOS; CHMP1B→EGFR receptor sorting via HGS/STAM), ~3 plausible, and **~3–4 weak
   and hub-driven** — the failure mode is generic-hub links (PLEKHB2/TRIM23→EGFR *via ubiquitin* UBB/UBC/RPS27A;
   SNX27→MAPK *via* fibrinogen). Roughly ~60–70% by eye, so the 85% proxy is somewhat **optimistic** for dark genes.

**Bottom line:** the graph gets the broad pathway *family* right most of the time (~85% corroborated on the matched
proxy, likely ~65% on true dark genes), but "exact mechanism" is weaker and the confident tier still lets through
hub-driven false positives (ubiquitin especially). A strong lead-generator at family resolution, not a proven annotation
— exactly the honest scope the tool claims.

## cell_pathway_map — the whole cell as interconnected pathways (viz)

Scaled the pathway-crosstalk idea from 5 hand-picked pathways to the **whole cell** (`viz/cell_pathway_map.html`,
generator `colab/cell_pathway_map.py`). The honest resolution: the **25 top-level Reactome systems** (Signal Transduction,
Metabolism, Gene expression, Cell Cycle, Immune, DNA repair/replication, …), covering 9,682 annotated genes. A circular
meta-network — node size = gene count, chord thickness = **how much two systems interconnect** (fraction of proteins they
share, Jaccard ≥ 0.08, corroborated by cross-system PPI). 42 interconnections above threshold.

Not all 2,792 Reactome pathways: those are mostly nested sub-pathways sharing ~all members (parent-child redundancy), so
a raw all-pathway graph is a hairball of trivial overlaps. The ~25 roots partition the annotated cell into its major
systems, and the overlaps *between* roots are the real cross-system interconnection.

The map passes the sanity check — its strongest links are textbook: **DNA Replication–DNA Repair–Cell Cycle** (genome
maintenance), **Signal Transduction–Development** (551 shared proteins), **Gene expression–Development**, **Vesicle
transport–Protein modification** (the secretory pathway), **Gene expression/Responses-to-stimuli–Immune**.

**HONEST SCOPE:** interconnection = **shared machinery + physical contact** — descriptive crosstalk, **not** causal signal
flow and **not** directional. The biggest hubs (Signal transduction, Metabolism, Gene expression) connect broadly partly
because they're huge and full of multifunctional proteins, and Reactome's annotation choices shape the overlaps. Dark /
orphan genes aren't here (no pathway — that's the orphan-network view). A map of who-shares-and-touches-whom across the
cell, not a wiring diagram of control, and not a phenotype predictor. (colab/cell_pathway_map.py → viz/cell_pathway_map.html.)

## impact — mutate/remove anything, see how far the effect is knowable (`impact` syscall)

Answering "now that we have NEXUS + the whole-cell network, can we mutate/remove anything and see the whole-cell effect?"
— the honest answer is *partly*, and `impact` is the integration that makes the boundary explicit instead of faking a
cascade. `impact GENE [UNIPROT POS WT MUT [PDB CHAIN]]` chains only the validated layers and abstains at the wall:

- **[1] Near-field** (validated ~0.75): NEXUS — does the mutation break the protein (fold/bind ΔΔG → LOF/GOF via
  regsign)? Or, for a plain removal, full loss of function.
- **[2] Direct cell effect** — the **measured** Perturb-seq blast radius (real data, for the ~9,871 knocked-out genes).
  If the gene has no measurement it says so and **refuses to predict** (network edges predict a knockout at r~0.03–0.09 =
  floor).
- **[3] Context** (descriptive) — which top-level cell systems the gene sits in (from cell_pathway_map) + essentiality.
- **[4] The far-field wall** — explicit **abstention**: propagating past the direct measured set to a whole-cell
  phenotype is measured *not* to compose (knockout transitivity ~0.009; forward-model AUC ~0.50 = chance). The pathway
  map shows who-shares-and-touches, not a wiring diagram that carries the effect forward.

Demonstrated live: `impact POLR2A` (removal) → measured collapse of ribosomal/nuclear transcription (RPLP0/RPS12/NPM1
down, mito genes up) — correct biology, real data; `impact TP53 P04637 175 R H` → near-field LOF-only (tumor suppressor,
no brake) + weak measured K562 effect; `impact FNDC5` (dark) → honestly *unobserved*, no prediction. So: we can see the
protein-level break (solid) and the direct measured response (real, subset), and we **stop** at the whole-cell cascade —
by design, because that's where the data says it stops. Not a whole-cell simulator; an honest "here's how far we can see."
