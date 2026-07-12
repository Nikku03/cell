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
