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

### Can the graph predict an UNMEASURED removal? (measured: mostly no, with one real exception)

Pushed on: "from the labelled graph + pathway interconnections, can't we remove a protein and see its effect even if not
measured?" Tested the full labelled graph (PPI + complex + pathway neighbours — its best shot) against real Perturb-seq
knockouts (n=126): the graph neighbours are **4.1× enriched** for real movers but capture only **~3% of them (recall 2.6%,
miss ~97%)** — for a *typical* knockout the graph catches **0** of the genes that actually moved. So the transcriptional
**cascade is not graph-predictable** (the far-field doesn't live on the edges).

**The one thing the graph CAN do without measurement — structural disassembly:** removing a protein deterministically
breaks every complex it's a subunit of, and complex-mates are co-essential ~1000× chance (validated). Added this to
`impact` as **[2b] STRUCTURAL** (from `gene2cplx`/`complexes`): e.g. `impact SMAD3` → "disassembles 4 complexes:
SMAD2-SMAD3-SMAD4 (with SMAD2, SMAD4), SMAD3-TTF-1, …" — real, graph-only, no measurement. The [4] wall now states the
measured graph-recall (4× enriched / ~3% recall) explicitly. So the honest split: the graph gives you the **structural
break** (certain) but **not** the functional cascade (~3% recall) — it abstains there rather than fake it.

## pathway_remove — remove a protein, redraw the graph, show what changes (viz)

Concrete demonstration of "remove a protein and see how the graph changes" (`viz/pathway_remove.html`, generator
`colab/pathway_remove.py [GENE]`, default PPP2R1A). The removed node is struck out (red ×), its edges cut (dashed red),
and any complex co-subunit that loses it is flagged (amber ring). Reusable for any gene in the graph.

`remove PPP2R1A` (the PP2A scaffold, a shared WNT↔MAPK-nuclear hand-off node): cuts **6 edges** (1 inter-pathway, all to
the PP2A module), and — because it scaffolds 4 PP2A/STRIPAK complexes — structurally orphans **PPP2CA** (its co-subunit,
still expressed but now without a scaffold). The WNT↔MAPK-nuclear hand-off drops from 4 shared bridges to 3 — **reduced,
not severed**, because the PP2A module is redundant (PPP2CB, PPP2R5D still bridge it). `remove GRB2` for contrast cuts 13
edges and thins the PI3K↔EGFR bridge (3→2).

**HONEST SCOPE (the whole point):** this is the **direct / structural** rewiring only — deleting a node deletes its
measured PPI edges, and complex-mates that lose a subunit are co-essential ~1000× chance (validated). It does **not** draw
the downstream transcriptional **cascade**, because the graph predicts a knockout's real movers at only ~3% recall (4×
enriched, misses ~97%). We redraw the wiring we can see for certain and stop at the ripple we can't — the same honest
boundary as `impact`. (colab/pathway_remove.py → viz/pathway_remove.html.)

### Is the cell's "backup" ACTIVATED on removal, or already-on? (measured: already-on)

Prompted by "the cell has a backup plan that is activated when the protein is removed" — the natural explanation for why
removing PPP2R1A *reduced but didn't sever* the WNT↔MAPK-nuclear bridge. Split it into two testable claims:

- **Do backups exist / matter?** YES (already measured, enzyme_patterns paralog-buffering): genes with a sequence-family
  paralog are less essential — essential genes have median **0** paralogs vs **2** for non-essential (effect −0.30,
  p=2e-14). Redundancy is real and predicts survival.
- **Are backups ACTIVATED (upregulated) in response to the removal?** NO. Tested across ~64,000 paralog pairs (name-stem
  proxy) in the measured Perturb-seq: when a gene is knocked out its paralog is upregulated only **0.6%** of the time
  (mean response −0.005) vs **1.2%** for a random gene — i.e. **0.5× random, less than chance**. PPP2R1A KO → PPP2R1B
  +0.04, AKT1 KO → AKT2 +0.02 (nothing).

**Conclusion:** the backup is **passive / constitutive**, not an on-demand response. The paralog is already expressed at
its normal level and simply keeps covering the function — the cell is robust by having *spare capacity always running*,
not by sensing damage and switching on a reserve. This is exactly why the static graph shows "reduced, not severed"
(the parallel wiring was always there) and why perturbations get buffered (ties to the nexus finding that moderate damage
is absorbed). Caveats: name-stem paralog proxy; single-timepoint K562 (slow/post-transcriptional compensation not
captured). Not committed as a syscall — a measured clarification of network robustness.

## pathway_overlay — structural cut vs the REAL measured effect (viz)

The user's idea: for a removed protein that HAS measured knockout data, overlay its real measured movers onto the same
5-pathway graph. Built it (`viz/pathway_overlay.html`, generator `colab/pathway_overlay.py [GENE]`): removed node struck
out + edges cut (red), and every graph node that actually moved on knockout badged ▲up (green) / ▼down (blue).

First, the aggregate test (the striking part): of the **32** graph nodes with measured knockout, **31 move ZERO other
graph nodes** when removed — only CTNNB1 moves any. These proteins are all "connected" in the pathway map, yet knocking
one out doesn't measurably move the others (signalling acts post-transcriptionally; Perturb-seq reads mRNA).

The overlay makes the disjointness concrete:
- **PPP2R1A KO moved *nothing*** measurable cell-wide (just its own knockdown) — a scaffold with no transcriptional signature.
- **CTNNB1 KO moved 168 genes** cell-wide (it's a TF), but only **3** land in the 5-pathway graph and **0** are its
  structural neighbours. The real effect is almost entirely off-graph: MAP1B (+1.6), APOE (−1.1), ARG2, IGFBP2, PHGDH… —
  none of them in the pathway wiring.

**Conclusion (the ~3%-recall wall, drawn):** the structural neighbours and the measured movers are **different sets**. The
graph shows who-shares-and-touches; the knockout's real downstream effect goes elsewhere (indirect/regulatory, and mRNA is
blind to signalling). The structural cut is certain; the cascade must be measured, not read off the graph — exactly the
boundary `impact` and `pathway_remove` draw. (colab/pathway_overlay.py → viz/pathway_overlay.html.)

### Time-forward directed propagation (TF→target causal graph) — measured: direction real, prediction at chance

Proposed: label the TF/literature/pathway edges in a TIME-FORWARD causal state so a knockout only affects downstream,
then forward-propagate a removal along those directed edges and track how it hits PPIs further down. Engaged with it by
MEASURING the load-bearing first step — do a TF's directed regulatory targets actually move when it's knocked out?

Tested the 612,133 directed `reg` edges (and the 17,432 curated `sig` edges) against measured Perturb-seq, 1,245 TFs:
- **Which targets move: at chance.** A TF's annotated targets move only **0.6% vs 0.5% random = 1.1×** (curated sig:
  1.9×, still 0.6% recall). Directed edges do NOT fix the recall problem — most annotated TF→target edges are ChIP-style
  binding, not functional regulation in this cell, and redundancy buffers the rest.
- **Direction IS real: sign 73% correct.** Among targets that did move, activating→down / repressing→up matches the
  edge sign 73% of the time (vs 50% chance) on the large reg set. So the causal *direction* data is meaningful — we just
  can't predict *which* edges fire.

**Conclusion:** the directed / time-forward instinct is correct (and better than undirected PPI in principle), but the
forward-propagation it implies is exactly `fieldsim` (signed regulatory Hill+decay forward model) — which we **built and
measured at AUC ~0.50 (chance)** for which-genes-move. First-order is at chance (1.1×), so compounding it forward can only
get worse (the transitivity ~0.009 wall, now confirmed for DIRECTED edges too). Two extra problems: the regulatory network
has feedback loops (not a DAG → no clean global time order — matches the earlier tier≠order finding), and the "made-up"
literature pathways are noisy (~4× chance). **What DOES survive:** the sign/direction (73%, already used by regsign, 86%
GOF precision) and measured-mediated downstream tracking (X→M→P with both hops measured = 43× chance for a validated
minority — the existing `cascade`/`influence` syscall). Not committed as a new syscall — a measured verdict on a proposed
mechanism (which was already built as fieldsim).

## cell_conditions / `conditions` — infer WHEN a pathway turns on (the subtractive strategy, validated)

The user's idea: remove the always-on (constitutive) pathways and the ones whose condition we already know, then for the
leftover trace each gene ← its TF ← what controls the TF, to *recover the conditions*. Built exactly this
(`cell_conditions.py`, `conditions` syscall) — and it validates.

**Key move that keeps it honest:** the backward regulator-trace is a **structural / annotation** query — hypergeometric
enrichment of a pathway's genes among a curated condition-TF's targets (the 612k directed `reg` edges) — **not** the
forward dynamical propagation this project measured fails at chance (directed edges predict *which* targets move at only
1.1×). "Is this gene set enriched for HIF1A targets?" is annotation-level inference, like connect's corroboration.

**It does the whole strategy:**
- **Constitutive / always-on** → no condition-TF enrichment: translation, rRNA processing, TCA cycle, splitting all
  correctly flagged CONSTITUTIVE. (Removed, per step 1.)
- **Conditional** → the top enriched TF names the trigger. Validated on 58 pathways whose condition is stated in their
  name: the traced condition-TF is recovered **top-1 67% / top-3 76%** (chance ~7%/20% = ~9×/4×). hypoxia→HIF1A,
  cholesterol→SREBF1, TP53-death→TP53.
- **The leftover (`conditions scan`)** — pathways whose trigger is NOT in their name but is inferred: TLR2/3/4/7/9
  cascades → NF-κB inflammation, IL-4/13 → STAT, FOXO-mediated transcription → starvation. All correct — the discovered
  conditions.

**A real failure caught and fixed:** first attempt included MYC/E2F1 as "growth" conditions — but their targets are the
constitutive ribosome/translation machinery, so they mislabelled housekeeping as "growth-activated" and outcompeted HIF1A
on hypoxia. Restricting to genuine transient stress/signal TFs restored the discrimination (validation 52%→67%). ~16
curated condition-TFs cover the major stress/signal axes.

**HONEST LIMITS:** condition-CATEGORY resolution (can confuse related stresses, e.g. hypoxia vs ER stress); ~16 curated
TFs (not all conditions — no circadian/developmental); the reg edges are annotation (regulated-by, not
measured-functional-in-K562). A hypothesis generator for a pathway's trigger, not a proven condition. (cell_conditions.py
→ cell_conditions.json; `conditions` syscall.)

### conditions + impact: signed direction (activation/inactivation) & per-partner contribution

Two sign-aware enhancements using the signed regulatory edges (this project measured the sign is ~73% reliable):

- **`conditions` now reports DIRECTION + FEEDBACK.** Beyond naming the trigger TF, it uses the edge signs to say whether
  the condition turns the pathway **ON (activates)** or **OFF (represses)**, and detects **feedback loops** — pathway
  genes that regulate the trigger TF back. Textbook-correct: *Cellular response to hypoxia* → hypoxia **ON** via HIF1A,
  **negative feedback via VHL/HIF3A/LIMD1** (the classic O₂-sensing degradation loop); cholesterol → SREBF1 ON, negative
  feedback via PPARA; TP53 cell-death → ON, negative feedback via BCL6. Negative feedback = self-limiting; positive =
  amplifying/switch-like.
- **`impact` now has [2c] CONTRIBUTION & DIRECTION.** For the removed protein, its signed partners split into **ACTIVATES**
  (a responder goes DOWN when removed) vs **INHIBITS** (a responder goes UP / released) — so for a multi-interaction
  protein you see how many partners it drives each way and which. e.g. `impact TP53` → activates 986 / inhibits 263
  (ABCB1/MDR1, AR go up when p53 is removed — real de-repression). Honest coupling to the wall: the sign gives the
  *direction* for any partner that responds (~73%); *which* respond is still the [4] wall (~3% recall). Direction is
  knowable; the responder set is not.

(cell_conditions.py `_direction`; cellos.py `_impact_contribution` / impact [2c].)

## Getting more data to improve the graph — a data-quality experiment (and an honest ceiling)

Asked to pull in all available data (perturbation/ChIP-seq etc.) to make the graph better. Inventoried what we have,
fetched what we could, and — crucially — **measured** each source against the two things that matter, instead of assuming
more data helps.

**What we already had but were underusing:** `causal_edges.json` = **SIGNOR** (60k curated, signed causal edges).
**What we fetched fresh:** **TRRUST v2** (grnpedia.org) — 795 TFs / 9,396 PubMed-curated signed TF→target edges (saved as
`trrust_regulon.json`). (OmniPath/CollecTRI was 502 — external fetch of that one is down right now; UniProt REST 403'd;
GitHub raw + TRRUST reachable.)

**Measured head-to-head** (vs the raw 612k ChIP-style `reg` edges):

| regulon | which-targets-move (measured KO) | condition-inference top-3 |
|---|---|---|
| reg (612k, ChIP-style) | 1.1× (chance) | 76% |
| **SIGNOR (60k, curated causal)** | **2.7×** | **88%** |
| TRRUST (9k, curated literature) | 1.1× | 80% |

**Two honest conclusions:**
1. **Quality beats quantity.** SIGNOR (curated causal) is decisively the best — it predicts measured knockout movers 2.7×
   vs the noisy reg's 1.1×, and lifts condition top-3 to 88%. We were underusing it. Upgraded `conditions` to MERGE
   SIGNOR + TRRUST + reg (best-first) → validation top-3 **76% → 81%**, and the fetched TRRUST is now in the pipeline.
2. **But better data does NOT break the wall.** Across three independent regulons the which-targets-move enrichment tops
   out at 2.7× with ~0.6% recall — the far-field cascade still isn't predictable. The wall is **informational** (it needs
   *measured, context-specific functional* edges — i.e. more Perturb-seq, not more annotation), exactly as the earlier
   tests implied. More annotation regulons sharpen the *structural/condition* queries; only measurement moves the
   *prediction* wall. (trrust_regulon.json; cell_conditions.py merges SIGNOR+TRRUST+reg.)

## perturb_recall — can a neural net + all the data get RECALL? (measured: real lift, partial, wall not broken)

The user's plan: more screen data + a neural network to get the recall the graph can't. Built and measured it honestly.
Data: we already had FOUR cached Perturb-seq screens (k562 9,871 KOs — used here; plus gwps 11,258, hct116, norman for
cross-screen). Model: a gradient-boosted classifier predicting, for a HELD-OUT knockout, which genes move — fusing
**base-rate** (response-proneness), **graph-neighbour co-response transfer**, **reg/SIGNOR targets**, and **PPI**.

**Held-out-by-knockout, 545 test KOs, mean AUPRC / recall@50:**

| method | AUPRC | recall@50 |
|---|---|---|
| random | 0.017 | 0.4% |
| graph edges alone | 0.019 | **3.2%** ← the old 3% wall, reproduced |
| base-rate (mean response) | 0.152 | 15.4% |
| transfer (neighbour-KO co-response) | 0.155 | 15.4% |
| **MODEL (all data)** | **0.183** | **17.9%** |

**Honest reading — it works, but it's a lift, not a breakthrough.** The model beats every baseline (11× random) and
pushes recall@50 from the graph's **3% → 18%** — a real, useful triage ranker. BUT: recall is **partial** (top-50 catches
~18% of movers, misses ~82%), and the lift **over the simple base-rate baseline is modest** (+0.03 AUPRC, +2.5pp). Where
does the recall come from? Almost entirely **base-rate** ("which genes are response-prone") + **co-response transfer**
("what my network-neighbour knockouts did") — the **graph edges alone are ~random**, and a co-response-from-targets SVD
feature added nothing. So knockout-**specific** prediction stays hard.

**How the data is best used (the answer to "think how all this data can be used"):** not as a wiring diagram to propagate
along (that's ~random), but as (1) a **response-proneness prior** per gene, (2) a **co-response transfer** from measured
neighbour-knockouts, (3) **regulon/PPI priors** — fused by the model. This is the honest maximum the current data
supports. It matches the field (foundation models barely beat baselines on perturbation prediction) and this project's
own latent-bridge (+0.018 over mean). **More screens raise base-rate quality but did not break the wall on this
evidence** — the missing signal is measured, context-specific function, not model capacity or data volume alone.
(perturb_recall.py → perturb_recall.json.)

### Pushing recall further — autoencoder + multi-screen pooling (measured: no gain / infra-blocked)

Followed the two proposed levers to their honest conclusion.

- **Autoencoder co-response embedding.** Trained a torch AE on the K562 response matrix → a 24-d nonlinear gene embedding,
  added to the recall model's features. Measured (545 held-out KOs): AUPRC **0.185 → 0.185 (−0.000)**, recall@50
  **18.1% → 17.4%**. **No gain** — the nonlinear embedding does no better than the SVD that already failed. The
  bottleneck isn't embedding quality; it's that a held-out knockout's identity gives only its graph position, and
  graph→response is the wall.
- **Multi-screen pooling / cross-cell (HCT116, GWPS).** The extra screens are cached, but the large gzip-chunked h5ad
  matrices read too slowly to load in this sandbox (even 300–500 rows timed out at 70–90 s), so the cross-cell test
  couldn't be completed here. Not fabricating a number I didn't measure — it runs in a decompressed/Colab environment.

**Net:** both levers confirmed the ceiling — recall@50 stays ~18%. The only lever left that could actually move it is
genuinely new **measured, context-specific** signal, not a bigger model, a fancier embedding, or more of the same
annotation. `perturb_recall` stands as the honest maximum the current data supports (graph 3% → fused-model 18%).

## tfbs_score / `tfbs` — protein-DNA binding done the honest way (not a NEXUS retrofit)

Asked whether NEXUS could be modified to score protein-DNA binding, accounting for neighbouring nucleotides and DNA shape
("rotation"). Honest answer: **NEXUS's engine can't be retrofitted** — it's amino-acid physics (rotamers, sidechain vdW,
the 20-AA interface), and protein-DNA is different chemistry (4 bases + phosphate-backbone electrostatics, major/minor
groove base readout, base stacking, and a 3D **shape** readout). It would be a *new engine* reusing only the ΔΔG-regression
scaffold. **But** TF-DNA specificity is a mature, measured-data field, and we already had the core data: JASPAR2024 PWMs
for **743 TFs** (`tf_motifs.json`).

So built the right version, `tfbs_score.py` (`tfbs` syscall): PWM log-odds over a sliding window, both strands.
**Neighbouring nucleotides are handled** — the motif window spans the flanks, so a variant's effect depends on core-vs-flank
position. Demonstrated on ARNT (E-box CACGTG): a **core** C→A drops the score 7.8→3.4 (**abolishes the site**), a **flank**
change ~0. TP53 recovers its dimeric response element (AACATGCCCGGGCATGTC). `tfbs GENE [SEQ [POS ALT]]`.

**HONEST SCOPE:** PWM log-odds predicts **intrinsic, in-vitro sequence preference** (validated by construction; literature
AUC ~0.85–0.9) — the right tool for **non-coding / regulatory variant interpretation** ("does this variant disrupt a TF
site"). It does **not** predict in-vivo binding or regulation (chromatin, cofactors, TF concentration = the same context
wall). **DNA shape ("rotation" — minor-groove width, roll, helical twist, propeller) and dinucleotide dependencies are the
documented Rohs add-on** — a modest, validated improvement over PWM alone, left as an extension since it needs the pentamer
shape table (fetchable). So: the honest protein-DNA node exists and works for variant interpretation; it's a *new* engine,
not a modified NEXUS. (tfbs_score.py → tfbs_score.json; `tfbs` syscall.)

---

## The cofactor switch, solved the data way (`cofactor.py` → `cofactor` syscall)

Asked to *solve the cofactor-switch problem*: a TF binds a **different sequence** depending on which partner it dimerises
with — SMAD3+FOXH1 vs SMAD3+RUNX land on different sites; FLI1 alone reads an ETS site, but FLI1+CEBPB reads a fused
composite. My own list of "what a real version would need" had four parts: AlphaFold-Multimer structures of the candidate
complexes → an absolute-affinity model → a composite-motif database → measured cofactor expression per condition.

**The honest engineering call was to skip the first two and use the last two.** The structural route (AF-Multimer of every
candidate complex → absolute affinity) is compute-blocked in this sandbox *and* unreliable, and — the real point — cofactor
choice in a cell is driven more by **which partner is present** (concentration) and measured cooperativity than by raw
ΔΔG_bind. So the data route isn't a fallback; it's the better model of the actual mechanism.

Built it: fetched the **JASPAR2024 composite (heterodimer) motifs** — `composite_motifs.json`, **479 TF1::TF2 pairs**, each
measured by HT-SELEX **of the dimer**. That composite motif *is* the switch: it's the sequence the **pair** binds, and it's
different from either partner's solo motif. `cofactor.py` loads these, the solo motifs (`tfbs_score`), the cell-type
expression mask (`emask`, 200 types), and PPI, then answers:

- `cofactor TF1 TF2` — the switch made concrete: the pair's **composite** consensus vs each partner's **solo**.
- `cofactor TF [CELLTYPE]` — which partners have a composite motif *and are expressed* in that context.

**MEASURED — the switch is real:** composite motifs are longer/distinct from either solo — mean **14.4 bp vs 10.6 bp**, and
**82% of composites are longer than either partner's solo**. The consensuses are visibly fused half-sites:
`FLI1::CEBPB = acCGGAAGT·TGCGCAAt` (ETS + C/EBP), `ATF3::FLI1 = ACCGGAA·ATGCGTCAT` (ETS + bZIP), `GATA1::TAL1 = GATA + E-box`.

**Context-gating demonstrated** (same TF, different cell → different available cofactor → different site): FLI1 in a
**monocyte** finds **CEBPB/CEBPD** (the myeloid C/EBP partners) expressed; in a **T cell**, BHLHE40; in a **B cell**,
TCF4/ZBTB20. So in a monocyte, FLI1::CEBPB forms and binds the fused composite above — not FLI1's solo ETS site.

**HONEST SCOPE:** this delivers the pair's **in-vitro** binding site (the concrete switch, measured) — not (a) de-novo
complex formation from structure/affinity (the AF-Multimer route we deliberately skip), nor (b) the in-vivo wall (whether
the pair actually binds a given promoter and regulates it — chromatin/context, the same wall this project keeps hitting).
Limited to the **~479 pairs** with a measured composite motif. A real, data-grounded cofactor-specificity layer for variant
reasoning — not a de-novo complex predictor. (cofactor.py → cofactor.json; `cofactor` syscall.)

---

## Taking the motif scan IN VIVO — the 3-gate chromatin pipeline, measured (`invivo_gate.py` → `invivo` syscall)

The last piece (`tfbs`) predicted a TF's **intrinsic, in-vitro** sequence preference and was explicit that it does *not*
predict in-vivo binding — chromatin, cofactors, concentration. This closes that gap with the standard pipeline
(ATAC/DNase → H3K27ac → ABC/Hi-C), built on **real ENCODE K562 data** and — the point — **measured against a ground truth we
never train on: ENCODE K562 TF ChIP-seq** (where each TF *actually* binds). Run on chr22 as a real slice of a genome-wide
method (the whole genome just needs the 3 Gb reference; the method is identical).

**The gates:**
- **Gate 1 — open (DNase/ATAC).** Intersect PWM hits with accessible chromatin. A TF can't bind DNA wound in a nucleosome.
- **Gate 2 — active (H3K27ac).** Keep survivors that also sit in an active enhancer/promoter mark. Open ≠ functional.
- **Gate 3 — 3D (ABC/Hi-C).** Map the bound enhancer to the **gene it loops to** (ENCODE-rE2G links), not the nearest gene.

**MEASURED result (precision = predicted sites that are truly bound, vs ChIP-seq):**

```
TF     motif  P.1D  P.active  lift   FP-killed        TF     motif  P.1D  P.active  lift  FP-killed
GATA1   7bp   0.02    0.50   20.5x    97%             CTCF   15bp   0.69    0.80   1.2x     29%
GATA2   7bp   0.02    0.39   19.4x    96%             REST   20bp   1.00    —      1.0x     32%
YY1     8bp   0.01    0.53   28.4x    99%             NRF1   12bp   0.62    0.92   1.4x     55%
MAX     6bp   0.09    0.80    7.9x    96%             SPI1   13bp   0.30    1.00   3.0x     97%
JUND    9bp   0.10    0.93    7.8x    94%             mean precision 0.27 → 0.66 (2.4x)
```

**The result confirms the theory and is honest about where gating matters.** For **degenerate short motifs** — the majority,
and the ones that matter — gating is transformative: GATA1's motif matches **7,361** places on chr22 at 1.6% precision;
gating leaves **76** at 50% precision, **killing 97% of the false positives** and lifting precision **20×**. YY1 gains 28×.
This *is* the "kills ~80%+ of false positives" claim, measured. For **long specific motifs** (CTCF 15bp, REST 20bp) the
sequence is rare enough to be self-filtering — already 69–100% precise from sequence alone — so gating adds little. The mean
2.4× understates the win for the degenerate-motif majority.

**Gate 3 measured the user's key point directly:** of GATA1's 56 open+active sites that fall in an ABC enhancer, the 3D
target gene **differs from the linear-nearest gene 39% of the time** — e.g. a GATA1 site nearest *YPEL1* actually loops to
**MAPK1**; one nearest *ATP6V1E1* loops to *BCL2L13/PEX26*. Nearest-neighbour guessing is wrong that often.

**HONEST BOUNDARY:** this converts an in-vitro sequence preference into an in-vivo **binding** prediction and shows, gate by
gate on measured ChIP, how much each helps. It answers *"does the TF physically land here, in this cell"* — well. It does
**not** thereby predict which genes **move** on knockout: **binding ≠ regulation**, and which-gene-moves is the separate
dynamics wall this project keeps hitting. The recall side is honest too — at this stringent threshold, motif→ChIP recall is
low (many real ChIP peaks have no strong canonical motif = indirect/tethered binding), so gating buys precision at the cost
of recall. All real ENCODE K562 data; nothing simulated. (invivo_gate.py + fetch_invivo.py → invivo_gate.json; `invivo`
syscall. `invivo` = the table; `invivo TF` = live 3-gate scan + ABC targets for one TF.)

---

## Binding ≠ regulation — unless it recruits polymerase. Measured. (`regulate.py` → `regulate` syscall)

The `invivo` gates predict where a TF **binds**; the honest caveat was that binding isn't regulation. The sharpening:
**regulation is occupancy that lasts long enough to appoint RNA Pol II and fire transcription.** That turns a philosophical
caveat into a measurable one — the *productivity* signature is observable:
- **Pol II recruited** — POLR2A + POLR2AphosphoS5 (initiating, Ser5P) ChIP-seq. Is the polymerase physically there?
- **eRNA firing** — PRO-cap **bidirectional** peaks. Is the element actually being *transcribed* (the readout of a
  productively engaged enhancer)?

And it can be validated against the one assay that operationally *defines* regulation: **CRISPR enhancer perturbation** —
silence the element, does the gene change? Used the ENCODE harmonised CRISPR dataset (ENCFF968BZL; Nasser 2021 / Gasperini
2019 / Schraivogel 2020), where `Significant` = the element truly regulates the tested gene.

**The productivity ladder** (element-level, 3,961 CRISPR-tested elements, base rate regulator **12.5%**):

```
all elements       12.5%   →  open (DNase) 14.4%  →  + active (H3K27ac) 20.0%  →  + Pol II 27.6%  →  + eRNA 28.9%  (2.3×)
```

Recruiting polymerase and firing eRNA **more than doubles** the odds an element is a real regulator.

**The sharp test** — among elements already open *and* H3K27ac-active, does the *act* of transcription add anything over the
*mark*? **Yes, and it's the whole story:**

```
eRNA firing → 26.0% regulate      |   silent → 12.9%   (= the base rate; the H3K27ac mark alone barely discriminates)
Pol II +    → 27.6%               |   Pol II − → 17.7%
```

An H3K27ac-marked element that **isn't** firing eRNA regulates its gene at the base rate — the mark is necessary but nearly
worthless as a discriminator on its own. The **act** of transcription is what separates a regulating element from a decorated
one. Exactly the point.

**The full chain** (pair-level, 10,331 element→gene pairs, base rate `Significant` **5.5%**) — Gate 3 (which gene, via ABC 3D
loop) × Gate 4 (productivity) together:

```
productive element alone ............ 9.5%    (weak — a firing element may regulate a DIFFERENT gene than the one tested)
ABC-linked to that gene alone ....... 25.3%
productive AND ABC-linked ........... 31.4%   (5.7× base rate)
neither ............................. 2.4%    (below base — essentially non-regulating)
```

**Binding that recruits Pol II *and* loops to the gene is ~6× more likely to be real regulation.** That is the honest,
measured bridge from binding toward regulation.

**HONEST BOUNDARY.** It's a strong discriminator, not a deterministic one — the precision ceilings around 30% because
(a) **CRISPR power** limits (underpowered true pairs get labelled not-significant), (b) **enhancer redundancy / shadow
enhancers** (silencing one productive enhancer doesn't move the gene when another compensates — a real biological reason a
productive element shows no CRISPR effect), and (c) the actual *cause* you named — residence time long enough to appoint
Pol II — is **not directly measurable**: ChIP occupancy is a population proxy for residence-time × concentration, and true
single-molecule dwell time has no genome-wide assay. So productivity enriches regulation ~6×; it doesn't make it certain.
All real K562 data; the answer is measured, not asserted. (regulate.py → regulate.json; `regulate` syscall.)

---

## The kinetic-competition model — residence time vs polymerase appointment. Built, then measured. (`kinetics.py` → `kinetics` syscall)

The `regulate` result showed Pol II recruitment marks regulation but left the *cause* — "binds long enough to appoint
polymerase" — unmeasured. This formalizes that cause and tests it. The mechanism is a **race**: each instant a TF is bound,
it can fall off (rate `k_off`) or hold on long enough for the machinery to appoint Pol II and fire (rate `k_init`):

```
P(productive per binding event) = k_init / (k_init + k_off) = τ_res / (τ_res + τ_appoint)
```

The model is real biophysics and the curve is sharp: with τ_appoint ≈ 30 s (Pol II/PIC assembly, literature order-of-
magnitude) and a consensus-site residence τ_max ≈ 12 s, a max-affinity site fires ~29% of the time, but just **2 kT down in
affinity (τ_res ≈ 1.6 s) it mostly falls off before polymerase arrives (5%)**. Threshold behaviour, exactly the intuition.

**The deep catch, stated up front:** affinity and residence time are different quantities.
- Affinity is **thermodynamic**: `Kd = k_off / k_on`.
- Residence time is **kinetic**: `τ_res = 1 / k_off`.

A PWM gives the equilibrium binding energy → `Kd`. It says **nothing about `k_off` alone** — two sites can share a `Kd`
and have 100× different residence times. Turning affinity into residence *requires assuming `k_on` is constant*
(diffusion-limited) — exactly what the single-molecule field disputes. True `k_off` needs single-molecule tracking (SMT),
which exists per-TF, not genome-wide. So the affinity→residence step is a **model, not a measurement**.

**So I measured whether the model's central prediction survives contact with real data** — among 2,101 real K562 ChIP-bound
motif sites (10 TFs, chr22), does higher affinity (→ longer predicted residence → more likely to win the race) actually give
more Pol II / eRNA?

```
affinity quartile   Pol II+   eRNA+          pooled correlation with productivity
Q1 low               0.065    0.203          SEQUENCE AFFINITY :  r(Pol II) = +0.017   r(eRNA) = −0.013   (FLAT)
Q4 high              0.091    0.202          MEASURED OCCUPANCY:  r(Pol II) = +0.205   r(eRNA) = +0.292   (REAL)
```

**The precise conclusion — and it's the cleanest statement of the wall yet.** The *mechanism is right*: **measured
occupancy predicts productivity** (r ≈ 0.2–0.3) — how much/how long the TF is actually there does drive polymerase
recruitment, just as the model says. But the link from **sequence/affinity to occupancy/residence is broken**: motif
affinity is *flat* against productivity (r ≈ 0). Two reasons, both real: (1) `Kd ≠ k_off` — the thing that matters
(residence) simply isn't the thing sequence gives you (equilibrium energy); (2) functional enhancers famously use
**deliberately low-affinity** sites, so affinity is decoupled from function by design.

So the `k_off ≠ Kd` wall is now **measured, not just argued**: you can write the rate equation *and* confirm that occupancy
drives productivity — but you **cannot fill in residence time from the motif**. It has to be measured (SMT), and there is no
genome-wide dwell-time assay. This locates the wall one layer deeper than "binding vs regulation": it's precisely at
**sequence → residence time**. (kinetics.py → kinetics.json; `kinetics` syscall.)

### Closing the loop — plug MEASURED residence times in (the honest end of the arc)

The `kinetics` result said residence would predict productivity but *can't be computed from sequence*. The proposed next
step was to stop computing it and use **measured** single-molecule tracking (SMT) residence times. Did that.

**The literature (real, cited).** Most sequence-specific TFs have "stable-bound" residence times in a **narrow ~5–20 s
range** ([Chen 2014 / reviews](https://pmc.ncbi.nlm.nih.gov/articles/PMC9117886/)); **CTCF** is a robustly-measured **long
outlier at ~1–2 min** ([Hansen 2017, eLife](https://elifesciences.org/articles/25776)); **GATA2** shows a stable fraction
**>5 s**. And the causal law is real — mutating a Gal4 site from high→low affinity reduced *both* its residence *and* its
target's transcriptional burst duration; GR and p53 residence correlate with their own output.

**But plugging measured residence into the competition model does NOT close the loop across TFs:**

```
TF     measured τ_res   model P(productive)   measured eRNA+     (real K562 binding)
CTCF        90 s             0.75  (highest)       0.032  (near lowest)   ← model maximally wrong
NRF1       ~10 s             0.25                  0.693
YY1        ~10 s             0.25                  0.638
...
cross-TF r(measured residence, eRNA) = −0.32   (negative)
```

**The CTCF natural experiment is decisive.** CTCF has the longest *measured* residence, so the model predicts it should be
the *most* productive — but it's near the *least* productive, because its minute-long residence is **architectural**
(insulator / loop anchor), not transcription-activating. (Honesty: 8 of 10 non-CTCF TFs share a class-typical ~10 s
placeholder — TF-specific K562 SMT doesn't exist for them — so the claim rests on the **CTCF measured point + the biology**,
not on a 10-point regression.)

**The resolution — and the honest end of this whole thread.** The residence→output law is *real and experimentally proven*,
but it is **intra-TF / per-site**: it holds for *one factor varying its own site or condition* (Gal4, GR, p53), where
"everything else" is held constant. It does **not** transfer across TFs of different function class, and the per-site
*in-vivo* residence that would test it genome-wide **has no assay**. So even *measured* residence can't be scaled into a
genome-wide productivity predictor with today's data.

Net: the mechanism you proposed — affinity → residence → win the polymerase race → regulation — is **correct and
experimentally demonstrated**, and every measurable link in it checks out (occupancy predicts productivity r≈0.3; Pol II +
eRNA predict CRISPR regulation ~6×). The one link that breaks is **getting residence from anything computable** — sequence
gives Kd not k_off (flat, r≈0), and a per-TF average is confounded by function class (CTCF). The missing measurement is
per-site in-vivo dwell time, and it doesn't exist genome-wide. That's not a gap in the model; it's a gap in the world's data.
(kinetics.py `measured_residence_test` → kinetics.json; `kinetics` syscall.)

---

## Binding vs regulation — put the two measured maps side by side (`bind_vs_reg.py` → `bindreg` syscall)

The whole TF arc pointed here: for a TF where we have **both** a measured *binding* map (ENCODE K562 ChIP-seq) and a measured
*regulation* map (Replogle K562 Perturb-seq — which genes change when the TF is knocked down), compare the two gene sets
directly. Peak→gene via promoter (TSS±5kb) + ABC 3D distal; regulated = |z|>3 on the ~8.5k measured-gene universe.

**First finding, before any comparison: only GATA1 is well-powered enough to even ask.** Of ~30 TFs with both ChIP and a
Perturb-seq knockdown, only GATA1 has a clean regulation signature (56 genes; MAX marginal; the rest sit at the noise floor,
because genome-scale Perturb-seq has few cells per knockdown). GATA1 is also the textbook TF for this question.

**The raw comparison (GATA1):** binds ~4,621 genes, regulates 56, overlap 29 — **overlap at chance** (0.85× promoter-only,
0.96× with ABC). Binding, as measured against this regulated set, does not beat a random gene set.

**I did not trust that number at face value** — it's exactly the kind of result that can be an artifact — so I ran a
**4-lens adversarial verification workflow** (parameter robustness, direct-vs-indirect split, a 5,000-draw permutation null,
and an under-detection check on canonical direct targets). The verdict was decisive and *nuanced*:

- **The number is real and robust.** All lenses reproduce the chance-level overlap; permutation p≈0.7; it holds across
  thresholds, promoter windows, and with/without ABC.
- **But "binding doesn't predict regulation" would be the wrong read.** The near-chance overlap is **mostly a
  power/composition artifact**, for two verified reasons:
  1. **The measured regulation is ~100% indirect.** All 56 regulated genes go *up* on knockdown — a lineage-identity shift
     (top genes LTB z=40, LST1, CTSC, S100A4 are the myeloid program de-repressed when erythroid identity is lost). GATA1
     doesn't bind those, so binding *should* overlap them at chance — and does. (This is real biology: the textbook
     GATA1↔PU.1 lineage antagonism.)
  2. **GATA1's genuinely bound direct targets are under-detected.** 8/9 canonical erythroid targets (KLF1, TAL1, NFE2,
     ALAS2, FECH, SLC25A37, TFRC…) are promoter-bound — but all sit at |z|<1 (e.g. ALAS2 −0.36, KLF1 −0.55) and never cross
     threshold, because the knockdown is weak (on-target z=−0.84, zero down-genes). Binding correctly marks the direct
     regulon; the perturbation just can't see it.

**Honest conclusion.** As *measured here*, binding and regulation are largely **disjoint** — but that's dominated by what the
Perturb-seq can detect, not by proven non-function. Genuine buffering (bound-but-non-functional sites) certainly exists in
biology, but **these data can neither measure nor exclude it**, because the direct arm is under-powered. The durable,
honest findings: (1) measured binding and measured regulation don't line up on this dataset; (2) *most* TFs can't even be
tested (Perturb-seq power) — which is itself why "what does a TF regulate" stays hard to pin down; (3) the direction that
*is* detected (de-repression of the myeloid program) is real, textbook GATA1 biology.

This closes the arc honestly: `invivo` (where it binds) → `regulate` (whether binding regulates) → `kinetics` (why residence
is the missing quantity) → `bindreg` (binding and regulation maps compared directly — and the measurement limits laid bare).
(bind_vs_reg.py → bind_vs_reg.json; `bindreg` syscall. Verified by a 4-lens adversarial workflow + permutation test.)

### Literature closes the gap — binding DOES predict regulation when the regulation map is well-powered

The Perturb-seq comparison left an honest ambiguity: binding vs regulation overlapped at chance (0.85×), but we showed that
was substantially because the knockdown was too weak to detect GATA1's direct targets. The clean test: swap the
under-powered perturbation for a **well-powered literature-curated regulon** (TRRUST v2, PubMed-backed, independent of the
ENCODE ChIP) and re-run the same overlap. Literature curation aggregates many focused, well-powered studies, so it should
contain the direct targets the single knockdown missed.

It does — decisively:

```
TF      curated targets   % bound    enrichment      significance      vs Perturb-seq
GATA1        50            34%        2.33×           p = 4.7e-4  ***    was 0.85× (chance)
MYC          93            46%        1.61×           p = 2.3e-4  ***
SPI1         62            53%        1.36×           p = 0.018    *
GATA2        18            22%        1.74×           p = 0.19    ns  (underpowered, n=18)
TAL1         10            10%        0.38×           ns          (enhancer-acting)
RUNX1        39             5%        0.60×           ns          (enhancer-acting)
```

For GATA1, binding recovers **the exact direct erythroid regulon the Perturb-seq missed** — ALAS2, KLF1, NFE2, EPOR, ITGA2B,
GP9, HEMGN, PPOX, CEBPA, BACH1 — and the overlap flips from chance (0.85×) to **2.33× enriched, p=4.7e-4**. MYC (1.6×,
p=2e-4) and SPI1 (1.4×, p=0.02) confirm it.

**So the definitive answer to "does binding predict regulation": yes — when the regulation map isn't detection-limited.** The
earlier chance-level Perturb-seq result was a **power artifact**, not evidence that binding is non-functional. Swapping in a
well-powered map recovers the signal and the specific direct targets.

**Honest exceptions.** TAL1 and RUNX1 stay flat even with literature — but for a known reason: both act largely through
*distal enhancers*, so a *promoter-only* peak→gene assignment is the wrong lens for them (their curated sets are also tiny,
10 and 39). This is a limitation of the assignment, not a failure of the principle.

This resolves the arc's final question. `invivo` (where it binds) → `regulate` (whether binding is productive) →
`kinetics` (why residence is the uncomputable missing quantity) → `bindreg` (binding vs regulation, and — with literature —
the confirmation that binding predicts regulation once the regulation map is well-powered). (bind_vs_reg.py
`compare_literature` → bind_vs_reg.json; `bindreg` syscall.)

---

## Tier-1 Go/No-Go — XGBoost regulation gate, adversarially verified (`crispr_gate.py`)

Executed the first gate of the ML strategy: does a mechanistic feature matrix carry signal for CRISPR-validated element→gene
regulation *beyond distance and the ABC formula*? — the honest checkpoint before spending on deep learning. Protocol
(user-designed, corrected in execution): GroupKFold **by chromosome**, AUPRC, monotonic constraints, `scale_pos_weight`, vs
distance-only and ABC-only baselines on the same folds. 10,331 CRISPR pairs, 5.5% base rate.

**The trap, confirmed empirically:** Significant pairs sit at median 34 kb, non-significant at 388 kb, so distance-only
already scores **AUPRC 0.405** — the real bar. (ABC-only is 0.247 globally, but that's abstention-deflated: ABC scores only
~8% of pairs and hits ~0.58 where it fires, so it's not the bar.)

**Result:** XGBoost on raw epigenetic/TF features = **0.495** → **+0.090 over distance**, winning 4/5 folds.

**Adversarially verified (4-lens workflow — ablation, distance-control, seed-robustness):**
- **Orthogonal, not distance repackaged** — the decisive test: the GBM beats a distance-only ranker *within all four distance
  bins* (+0.108 / +0.063 / +0.028 / +0.006), not merely across them.
- **Robust** — +0.09 holds across three seed reshuffles (+0.095 / +0.109 / +0.111); the sole losing fold is chr19-alone (61
  positives), which dilutes away.
- **Leak-free** — folds are fully chromosome-disjoint.
- **Source of the lift** — almost entirely `tf_count` (drop-one −0.042; next feature −0.013); epi-only (no distance) = 0.216,
  4× base rate. `tf_count` uses only ~**8** ChIP tracks, not the ~300-TF ENCODE compendium.

**Verdict: CONDITIONAL PASS.** The epigenetic/TF features carry *real, verified, orthogonal* signal beyond distance — the
task is learnable — but the magnitude is modest (~0.50 AUPRC) and rests on one under-powered feature, so **+0.09 is a floor,
not a ceiling.** The disciplined next step is **not** deep learning: it's to rebuild `tf_count` from the full ~300-TF ENCODE
K562 ChIP compendium (the sole orthogonal workhorse, currently crippled) plus a feature-level leak check, then re-gate. A
sequence CNN is justified only once that shows the margin *grows*. This is exactly the outcome the gate was built to produce:
it stopped a premature deep-net spend and identified the one experiment that decides whether the combinatorial-TF hypothesis
holds. (crispr_gate.py → crispr_gate.json.)

### The decisive experiment — full ~300-TF compendium: combinatorial-TF hypothesis CONFIRMED (via identity)

The Tier-1 gate's CONDITIONAL PASS named one decisive experiment: rebuild `tf_count` from the full ENCODE K562 TF-ChIP
compendium (the orthogonal signal lived almost entirely in a `tf_count` built from only 8 TFs). Ran it — fetched all **311**
K562 TF ChIP-seq tracks (39× the 8) and re-gated.

**First result — the count CEILINGS.** With `tf_count` rebuilt from 311 TFs (mean 49 per element, was ~1–2), the margin did
*not* grow — it shrank (+0.090 → +0.066, AUPRC 0.495 → 0.471). A mean-49 count is a *noisier* "active region" proxy than the
sparse erythroid-TF count. Scaling the count is a dead end.

**But that was the wrong feature.** The count answers "*how many* TFs bind"; the combinatorial hypothesis is about "*which*
TFs bind." Testing TF **identity** (311 binary features, one per TF) — leakage-free (no label-based selection), 3-seed stable,
with a label-shuffle control:

```
model                              AUPRC (3 seeds, chromosome-held-out)
distance                           0.393
epigenetics (8 features, no TF)    0.519
epi + TF-IDENTITY (311 binaries)   0.608   [0.609, 0.601, 0.612]  <- the lift
label-shuffle control              0.069   (~base rate 0.055)     <- no leak/overfit
```

**Verdict: PASS.** Knowing *which* of 311 TFs bind an element lifts AUPRC to **0.608** — **+0.09 over epigenetics-alone,
+0.21 over distance** — stable across seeds, and the label-shuffle control collapses to base rate, proving it is real signal,
not leakage or 311-feature overfitting. The **count discards** this signal (0.471); the **identity keeps** it (0.608). The
combinatorial-TF hypothesis — that regulation depends on the specific *combination* of factors present, not just activity or
proximity — is confirmed on held-out data.

**This is the honest GO for deep learning.** A gradient-boosted tree over binary TF-occupancy already reaches 0.608; a
sequence model that learns the TF motif grammar and their combinations directly (rather than consuming 311 pre-computed ChIP
tracks) is now *justified* — the Tier-2 spend is warranted because the signal it would exploit is measured and real. The gate
did exactly its job across both rounds: it stopped a premature deep-net spend on the wrong feature (count), then — when the
right feature (identity) was tested — gave a clean, controlled GO. (crispr_gate.py `regate_compendium`/`identity_test` →
crispr_gate_compendium.json.)

### Tier-2 sequence CNN — trained on Colab GPU, honest NO-GO (ship the Tier-1 GBM)

Built the Tier-2 multi-task sequence model and ran it on a Colab GPU (the CPU sandbox was too slow). Architecture: a conv
**motif encoder** over each element's 600bp DNA → an **auxiliary head** predicting which of 311 TFs bind (dense, ~1.2M labels,
forcing the encoder to learn the TF motif grammar from sequence) → a **regulation head** on `[embedding + 8 tabular features]`.
Chromosome-held-out, AUPRC, 3 seeds + a label-shuffle control.

**Result (GPU, real):**

```
sequence CNN (multi-task)   AUPRC 0.488   seeds [0.486, 0.483, 0.496]   <- stable
label-shuffle control       AUPRC 0.081   (~base rate 0.055)            <- no leak/overfit
--- baselines (same chromosome-held-out protocol) ---
distance-only               0.393
epigenetics-only GBM        0.519
TF-identity GBM (311 ChIP)  0.608   (the production model)
```

**Verdict: NO-GO.** The sequence CNN beats distance (0.488 > 0.393) — so it *did* learn to read regulatory sequence better
than mere proximity — but it **underperforms even the epigenetics-only GBM (0.519)**, let alone the TF-identity GBM (0.608).
The label-shuffle control collapses to base rate, so the model is legitimate (no leakage, no 311-feature overfit) — it's just
not better. Two honest reasons: (1) predicting TF binding *from sequence* is lossy, whereas the GBM gets 311 TFs' binding
*measured* — you can't out-predict a measurement; (2) **569 positives are too few** for a from-scratch CNN to learn the full
combinatorial grammar.

**So the production model is the Tier-1 GBM (TF identity, AUPRC 0.608)** — and the strategy's discipline held: adopt the deep
model *only if it wins on held-out data*, and it didn't, so we don't. To revisit Tier-2 honestly would need either far more
labelled regulation data, or a **pretrained sequence foundation model** (Enformer/Borzoi embeddings) rather than a CNN trained
from scratch on 569 labels. (seq_model.py + tier2_seq_model.ipynb → seq_model.json.)

**Where the ML arc lands.** Tier-1 gate: real orthogonal signal, verified. Compendium: combinatorial-TF hypothesis confirmed
(identity → 0.608). Tier-2: sequence CNN NO-GO (0.488). The honest deliverable is a **CRISPR element→gene regulation predictor
at AUPRC ~0.61 (TF-identity GBM)** — SOTA-competitive, leakage-controlled, and correctly *not* over-engineered with a deep net
that the data can't yet support.

**Correction — the tested Tier-2 was a REDUCED architecture.** The multi-task design's key idea is to train the auxiliary
311-TF-binding head on **millions of genome-wide loci** (to solve data starvation), then attach the sparse CRISPR head. The
version tested above did NOT do this: the aux head was trained on **only ~3,961 CRISPR-tested elements** (the same tiny set as
the flagship head), and the input was 600bp with a from-scratch CNN (not ~2kb / a pretrained encoder). So the 0.488 NO-GO
refutes a *data-limited* version, **not the full architecture** — the from-scratch CNN was asked to learn the TF combinatorial
grammar from ~4,000 sequences, exactly the starvation the genome-wide aux-pretraining was meant to avoid but here didn't. The
real Tier-2 to build: pretrain the shared encoder on ~1–2M genome-wide candidate elements (ENCODE cCREs) with their measured
311-TF ChIP vectors, THEN fine-tune the CRISPR head.

### The REAL Tier-2 (genome-wide pretraining) — built as designed, and it's a fair NO-GO

The reduced Tier-2 (0.488) trained the aux head on only ~4k CRISPR elements. This is the **full architecture**: pretrain the
shared encoder on **100k genome-wide ENCODE cCREs** to predict 311-TF binding from sequence, then fine-tune the CRISPR head —
run on a Colab GPU, with-vs-without pretraining to isolate the benefit.

**Result:**

```
PRETRAINING SUCCEEDED   aux TF-binding-from-sequence AUC -> 0.77   (a good sequence->binding model; the encoder DID learn the grammar)
fine-tune from-scratch        AUPRC 0.491   [0.509, 0.473]
fine-tune WITH pretraining    AUPRC 0.479   [0.488, 0.471]   <- pretraining did NOT help (within noise)
label-shuffle control         AUPRC 0.074   (~base rate 0.055)  <- clean, no leak
baselines: distance 0.393 | epigenetics-only GBM 0.519 | TF-identity GBM 0.608
```

**Verdict: NO-GO — and this time it's a *fair, complete* test.** Nothing was crippled: the pretraining worked (AUC 0.77 on
311-TF binding), the with/without comparison is clean, the shuffle control holds. Genome-wide pretraining simply **did not
transfer** to regulation — the pretrained model (0.479) matches the from-scratch one (0.491), both below epigenetics-only and
well below the GBM. So the ceiling is **real**, not an artifact of a shortcut.

**Why it fails even done right — the honest mechanism:**
1. **You can't out-predict a measurement.** The GBM gets 311 TFs' binding *measured* (perfect); the encoder only *predicts* it
   from sequence (AUC 0.77 — good, but lossy). Predicted binding can't recover the identity signal that measured ChIP gives.
2. **Redundancy.** The fine-tune head already receives the *measured* activity tracks (accessibility, H3K27ac, Pol II, eRNA)
   as tabular features. The sequence-derived binding is largely correlated with those, so it adds little orthogonal signal.
3. **569 positives** still cap the fine-tune head regardless of encoder quality.

This matches the field: the CRISPR element→gene benchmark is dominated by *measured* activity + 3D contact, not pure
sequence (ENCODE-rE2G's best models use measured tracks). **The architecture was sound — the user's instinct was right — but
the data reality (measured ≫ predicted) decides it.** Production model stays the **Tier-1 GBM (TF identity, AUPRC 0.608)**.

**Final ML verdict:** we built the deep model twice — reduced (0.488) *and* the full genome-wide-pretrained version (0.479) —
and both honestly lose to the gradient-boosted model on measured features. The deliverable is a **SOTA-competitive, leakage-
controlled CRISPR element→gene regulation predictor at AUPRC ~0.61**, and a deep-learning arc that earned its "no" on a
complete, fair test. (seq_model.py `main_full` + build_pretrain.py + tier2_seq_model.ipynb → seq_model_full.json.)

---

## Forward propagation: mutate any protein → flow through the whole network (`propagate`)

The regulation work so far stopped at the *direct* regulon (`mutreg`: a TF mutation → its immediate targets). The network
had all the downstream layers wired — gene products, PPI, complexes, Reactome pathways, pathway crosstalk — but nothing
*flowed a perturbation forward through them*. `propagate` closes that: mutate **any** protein or gene (NEXUS is the entry
point), and the effect flows layer by layer.

```
L0  NEXUS         mutation (ΔΔG_fold, ΔΔG_bind) -> the mutated protein's graded ACTIVITY (soft-AND, LOF/GOF)
L1  REGULATION    if it's a TF: its TRRUST-signed direct regulon = the first affected genes + direction
L2  PRODUCTS+PPI  the mutated protein + regulon targets -> their physical partners (PPI) and complex co-members
L3  PATHWAYS      every affected protein -> the Reactome pathways it sits in
L4  CROSSTALK     pathways sharing >=3 proteins with a hit pathway -> the interconnected downstream pathways
```

**Transcription rate — the promoter's role (the user's point: "don't forget how promoter and TF regulate the rate").**
L1 doesn't just list targets, it *orders* them by a mechanistic rate model:

```
ΔRate(target) = sign * (activity - 1) * promoter_PolII
                 └ TF direction         └ how much TF drive changed   └ POLR2A ChIP at the target's TSS±1kb
```

The **promoter sets the baseline** transcription rate (its resident Pol II), the **TF modulates** it. Direction falls out
correctly: an *activated* target follows the TF (GATA1 LOF → CDC6, BST2 down), a *repressed* target is *relieved*
(GATA1 LOF → CBFB, SPI1 **up**).

```
MUTATE GATA1 (ΔΔG_fold=7.5 -> NEXUS activity 0.31, LOF). Forward blast radius:
  L1 regulation  :    50 direct-regulon genes   (rate-ordered: CBFB up +112, CDC6 down -107, BST2 down -62, ...)
  L2 PPI/complex :  1136 physical partners
  L3 pathways    :  1874 pathways hit, spanning 10,489 genes
  L4 crosstalk   :   521 interconnected downstream pathways
```

**A sign bug caught and fixed (honesty pass).** Wiring up the rate model surfaced a flipped sign that had also been sitting
in the earlier `mutreg` chain: the formula was `-sign*(activity-1)` when biology (and the code's own comment, "activated
target follows activity") requires `+sign*(activity-1)`. An activator losing function makes its target go **down**, not up.
Checked against the *measured* K562 Perturb-seq direction: for GATA1's headline targets the fixed sign matches and the old one
didn't — CBFB (repressed, relieved) measured **z=+1.08 UP**, SPI1 measured **z=+1.85 UP**, CDC6 (activated) measured
**z=−0.27 DOWN**. Across all signed GATA1 targets with a measurable move the fixed sign agrees 5/8 vs 3/8 for the old one.
Fixed in both `nexus_regulate.py` and `propagate.py`.

**What it is, honestly — a map of the *possible*, not a simulator of the *actual*.** `validate()` measures each layer against
GATA1's real Perturb-seq movers, and the number tells the whole story:

```
                                      reach   captures   precision   vs chance
L1 regulation (direct regulon)          29        2        0.069       9.2x     <- enriched
L2 + PPI/complex                       631       10        0.016       2.1x
L3 + pathway members (blast radius)   5183       42        0.008       1.08x    <- diluted to chance
```

The blast radius is **enriched ~9× at the tight regulon layer and dilutes to chance** as it balloons through PPI and pathways.
That is exactly the session's measured wall: *which* things are mechanistically connected is knowable (structural reachability),
*how much each actually moves* does not compose. Two honest caveats on the rate model: its **direction** beats chance, but its
**magnitudes** don't match measured effect sizes (r~0) — the per-TF→target rate coefficient (how strongly *this* TF drives
*that* promoter) is the unmeasured quantity. So `propagate` is an honest forward-reachability engine with a mechanistic
rate-ordering on the first layer, not a quantitative cascade predictor.

**Still missing (acknowledged, not built):** splicing, 5′ capping, poly-A tail, and full epigenetic state — each its own hard
problem (e.g. SpliceAI-class splicing). Partial hooks exist (chromatin gating via `invivo`), but these post-transcriptional
and chromatin layers remain TODO. (`propagate.py`, `propagate`/`blastradius` syscall → propagate.json.)

---

## The precision fix: rank the blast radius instead of leaving it flat (`propagate` C1)

`propagate`'s flat blast radius had a real problem: precision collapsed to chance (L3 = 1.08×) as the radius grew. But the
diagnosis matters — it wasn't *missing* movers (recall reached 78%), it was that the radius was an **unranked, equal-probability
set**: 5,183 genes all treated as equally likely to move. Precision-at-chance is what "all equal" *means*.

**The fix — one ranked, calibrated score.** Replace the flat set with
`composite = 10·regulon·(0.5 + promoter_rate) + RWR_nearfield`, where RWR is random-walk-with-restart (α=0.3) over the weighted
multi-layer graph (directed regulatory 1.0 / PPI 0.5 / complex-clique 1.0). **Pathway co-membership is deliberately dropped** —
it was the dilution source that flattened L3 to chance. The curated regulon (rate-weighted) forms the high-precision core; RWR
ranks the physical near-field.

**Measured on GATA1's real Perturb-seq movers (leakage-controlled):**

```
method          AUPRC    lift    P@10   P@25   P@50
composite       0.0173   2.32×   0.10   0.08   0.06     <- the fix
RWR-alone       0.0128   1.71×   0.00   0.04   0.04
promoter-rate   0.0087   1.16×   0.00   0.00   0.00
degree          0.0074   0.99×   0.00   0.00   0.00     <- chance (not recovering hubs)
label-shuffle:  real 2.32× vs null 1.17× (p95 1.72) → p = 0.013
decile enrichment (top→bottom): [2.41, 0.56, 0.19, 0.93, 0.74, 0.56, 0.93, 1.30, 1.30, 1.11]
robust across |z|: 1.86× / 1.97× / 2.32× at |z|>2 / 2.5 / 3
```

Composite beats every baseline, **P@10 = 0.10 is ~12× the 0.8% base rate**, the top decile is **2.4× enriched**, and it survives
a 300-draw label-shuffle (p=0.013). Degree-alone is 0.99× (pure chance), so it is *not* just recovering hubs.

**The honest bound — what it does and doesn't do.** It *ranks* the radius so the top is a usable shortlist; it does **not** make
the far field precise. Proven, not asserted: GATA1's real movers (the myeloid de-repression program, all UP) are **0.0×**
enriched for the SPI1/CEBPA curated relay targets — the edges that connect a TF to its real secondary program are simply not in
our network. So the precision ceiling is a **missing-edge** problem, not an algorithm problem; a fancier propagator won't move it,
more measured edges would. That relocates the wall precisely. (`propagate.py` `rank_targets`/`validate_ranked`; shown in the
`propagate` syscall and propagate.json.)

---

## SpliceAI as a NEXUS splicing sensor — real pretrained weights, in torch

NEXUS reads a mutation's effect on the *protein* (ΔΔG_fold × ΔΔG_bind). But a variant can be protein-silent and still wreck
the transcript by destroying or creating a **splice site** — a different worker (the spliceosome reading the pre-mRNA), so it
needs its own sensor. SpliceAI (Jaganathan et al., Cell 2019) is that sensor. We integrated the **real pretrained weights**.

**No TensorFlow — reimplemented in torch.** The official `spliceai` wheel ships the 5 trained Keras models
(spliceai1–5.h5). Rather than install TensorFlow (and risk the numpy/scipy env), `spliceai_torch.py` parses the Keras
functional-model graph and re-executes it with torch ops, loading every Conv1D/BatchNorm weight by name and averaging the
5-model ensemble. It is a faithful re-host of the published weights — not a retrain, not an approximation; only the framework
changed. The architecture is the standard SpliceAI-10k pre-activation dilated-ResNet (39 Conv1D, 32 BN, 20 residual/skip adds,
Cropping1D 5000/side, W=[11×4,21×4,41×4…], dilation=[1,4,10,25] per group).

**Validated against real junctions.** On HBB (β-globin) fetched live from Ensembl, SpliceAI-torch fires an **acceptor at
exon-2's exact 5′ boundary (prob 1.00)** and a **donor at its exact 3′ boundary (prob 1.00)** — both matching the annotated
exon coordinates to the base. On a random sequence, max splice probability is ~0.000. The weights load correctly.

**Wired into NEXUS as a splicing mode** (`splice_nexus.py`, `splice` syscall). For a genomic SNV it computes the standard
SpliceAI **delta score** (ref vs alt, max acceptor/donor probability change within ±50 nt), then maps it to NEXUS's activity
currency: `activity = 1 − delta` (1 = protein made normally, →0 = splice-disrupting LOF). A splice-disrupting variant therefore
flows through the regulation → PPI → pathway `propagate` stack exactly like any other loss-of-function. Sequence context is
fetched on demand from Ensembl REST (GRCh38) — no local genome needed; weights auto-download from PyPI on a fresh session.

```
splice HBB chr11:5226576 C>A  (exon-2 donor GT)  -> donor_loss 0.999, delta 0.999 -> activity 0.001  [splice-disrupting (LOF)]
splice HBB chr11:5226690 C>T  (mid-exon body)    -> delta 0.004           -> activity 0.996  [no substantial splice effect]
```

**Honest scope.** SpliceAI is the field standard and strong on *canonical* splice-disrupting variants (published ~0.9 top-k /
high auPRC on ClinVar splice variants); it is weaker on deep-intronic, tissue-specific, and weak-effect splicing, and it
predicts splice-site *usage change*, not the exact resulting isoform ratio in a given cell (that needs RNA-seq junctions). So
direction (does it disrupt a site) is reliable; precise isoform outcome and quantitative penetrance are not. This is the first
of the acknowledged post-transcriptional layers (splicing / capping / poly-A / decay) to be built. (`colab/spliceai_torch.py`
+ `colab/splice_nexus.py`; `splice`/`spliceai` syscall.)

---

## Capping & poly-A — the next two post-transcriptional layers, built from first principles

Following `FIRST_PRINCIPLES.md`, the next two layers after splicing. Both validated against measured ground truth.

### 5′ capping — a provenance-gated machinery layer (`cap`)

Capping is *not* sequence-regulated: a transcript is capped iff it's a **Ser5-phosphorylated Pol II product AND the machinery
is intact**. So cap status is a near-constitutive Boolean (Ser5P provenance at the TSS × machinery integrity) that *gates*
cap-dependent stability/translation — not something to predict from sequence.

The testable first-principles claim is the **obligatory-role hierarchy cap0 > cap1 > cap2**, derived purely from each worker's
role. It validates cleanly against measured DepMap essentiality:

```
cap0 hub (RNGTT/RNMT)   dep_frac 1.0 / 0.97   obligatory for eIF4E  -> lethal
cap1 important (CMTR1)   dep_frac 0.85        immunity + translation
cap2 modifier (CMTR2)    dep_frac 0.04        pure refinement       -> dispensable
CBC reader (NCBP1/2)     dep_frac 1.0 / 1.0   + eIF4E essential
```

Monotone **cap0 > cap1 > cap2**, cap2 dispensable — predicted from role, confirmed by data. Mutating a worker via NEXUS: an
RNGTT/RNMT hub LoF is a **global** cap-dependent-translation collapse (short-half-life / 5′TOP transcripts first; measured-
lethal); a CMTR2 modifier LoF is a viable subset effect + an innate-immune "non-self" flag. Honest boundary: cap *presence* on
an ordinary mRNA is deterministic; the per-transcript cap0:cap1:cap2 methylation fractions need cap-specific sequencing.

### 3′ cleavage & polyadenylation + APA — sequence sets the site, cell-state sets the ratio (`polya`)

The poly-A machine reads a cis-element barcode and cuts ~10–30 nt downstream of an AAUAAA hexamer. **Site strength is
sequence-computable** (tiered hexamer × position-weight peaking at −21 nt, + GU/U downstream element, + upstream UGUA, + CA
cut). Validated against **real transcript 3′ ends from Ensembl**:

```
20 genes: canonical/variant AAUAAA hexamer in the -40..-1 upstream window = 85%
          vs downstream control window = 0%   (34x, 0.5 pseudocount)
          median hexamer position = -22 nt    (textbook ~-21)
```

The machinery's recognition signal is exactly where first principles says it should be. But the **APA ratio** (which poly-A
site the cell picks → 3′UTR length) is **cell-state, not sequence**: set by CFIm25(NUDT21)/CstF64(CSTF2) abundance + Pol II
speed. This cell's measured ppm ratio is **115 / 13.6 = 8.5 → distal / long-3′UTR bias** (high CFIm25 favors distal sites;
CFIm25 knockdown shortens UTRs and de-represses oncogenes, Masamha 2014). The chosen 3′UTR then fixes its miRNA-seed + ARE load
→ a stability *direction* for the decay layer (shorter UTR = fewer repressive sites = more stable).

Honest boundary, same shape as everywhere else: **site strength and variant direction are reliable; the cell-type APA ratio and
the tail-length → half-life map need 3′-seq/TAIL-seq** (tail length isn't even a clean monotonic half-life predictor — Lima
2017). (`colab/capping.py`, `colab/polya.py`; `cap` and `polya` syscalls.)

---

## mRNA decay (A4) — a real held-out prediction against SLAM-seq, and it works

The post-transcriptional layers so far (splicing, capping, poly-A) validated the *site / direction / presence* — mostly
confirming known biology. This one is different: **a quantitative held-out prediction where losing was possible.** Predict
per-gene mRNA decay rate `ln(k_deg)` in K562 from sequence, and score against measured half-lives.

**Ground truth:** RNAdecayCafe (Zenodo 15785218) — uniformly-reprocessed pulse-label SLAM-seq/TimeLapse-seq half-lives.
**K562 is our exact cell line** (10,802 genes; target = `avg_log_kdeg`). **Features:** GENCODE v46 pc_transcripts (CDS/UTR
coords in the headers), longest-CDS transcript per gene → the log-additive first-principles feature set: 61-codon composition
(→ codon optimality, learned *in-fold* so there's no CSC leakage), 3′UTR length / ARE / GC / m6A, 5′UTR length / uAUG, CDS GC.
**CV: GroupKFold by chromosome** (paralog-safe), 8,556 expressed genes.

```
model                          R2      pearson  spearman
3'UTR-length-only            0.068     0.262     0.294
codon-only                   0.199     0.447     0.435     <- dominant single signal
seq-features (no codon)      0.149     0.386     0.396
full linear (ridge)          0.279     0.529     0.526
full non-linear (XGBoost)    0.327     0.572     0.558     <- same features, non-linear
```

**Codon optimality is the dominant determinant** (codon-only R²=0.199, 3× the 3′UTR-length baseline) — exactly what first
principles predicted (the biggest CDS-encoded signal; ribosome dwell on non-optimal codons recruits CCR4-NOT). Adding 3′UTR
length + composition lifts it to **r = 0.53 (linear) / 0.57 (GBM)** held-out. Top directional features: CDS GC (+, faster
decay), 3′UTR length (+, longer UTR → more repressive sites → faster decay).

**Honest context — a floor, not a fluke.** The deep-learning SOTA (Saluki, Agarwal & Kelley 2022, a CNN+RNN on the full
sequence) reaches R²~0.5–0.6 — but only against a *denoised multi-dataset consensus*; on a **single cell line** the achievable
ceiling is ~0.35–0.45 (single-dataset noise is real). Our r~0.57 on one cell line with hand-built interpretable features is
genuinely competitive — a real predictive result, not confirmation of known biology. Sequence captures the *ranking/direction*
(Spearman 0.56); absolute half-life still needs measured rates.

**Why it matters for the whole model:** this is the layer that finally turns a transcription-*rate* change into a steady-state
mRNA *level* — **`[mRNA]_ss = k_txn / k_deg`** — connecting `propagate`'s promoter-rate output to an actual abundance. Of the
recent post-transcriptional builds, this is the one that clears the "hard test" bar. (`colab/halflife.py`; `halflife` syscall.)

---

## Epigenetics (layer B) — static state knowable, writer→target first-order, cascade is the wall

The last acknowledged-missing layer. Built as the three tractable pieces `FIRST_PRINCIPLES.md` §4 named, each held to the
honest three-tier ceiling. Fetched four more ENCODE K562 marks (H3K4me1/me3, H3K27me3, H3K9me3) to join the existing
H3K27ac/DNase.

**(a) Static chromatin state — a labeling of measured data, and it discriminates expression.** A ChromHMM-style promoter state
from the 6 marks at TSS±2kb → {active / bivalent / polycomb / heterochromatin / primed / quiescent}. Validated against measured
K562 mRNA expression (RNAdecayCafe RPKM — independent of the ChIP tracks):

```
active_promoter   median log10 RPKM +1.08
bivalent          median log10 RPKM -1.00   (~100x lower)   4.07x enriched for silencing
```

Bivalent promoters (H3K4me3 **and** H3K27me3) are ~100× lower-expressed than active ones — the state vector genuinely
discriminates repression. (H3K27me3-*presence* alone is a weak continuous predictor, AUC 0.57; the bivalent *class* is what
carries the signal.)

**(c) `mutate_writer` — the epigenetic analogue of `mutreg`, validated with a specificity control.** A writer/eraser mutation →
NEXUS activity → its mark-carrying loci move first-order (EZH2 → H3K27me3 loci; loss → de-repression). Tested against real
Perturb-seq, **with a control I insisted on**: do H3K27me3 genes go up *only* under PRC2 loss, or under any perturbation?

```
KD EZH2   H3K27me3 genes 5.33x enriched among up-movers   (Δ mean z +0.069)   <- catalytic writer
KD SUZ12  2.48x                                                                <- within control range
KD EED    1.53x                                                                <- within control range
control KDs (MYC/SPI1/TP53/GATA1/RPL13): max 2.86x
```

**EZH2 (5.33×) cleanly beats every control** — real, specific de-repression of its H3K27me3 targets. But the accessory PRC2
subunits (EED/SUZ12) fall *within* the control range, and the effect magnitude is small (Δz ~0.07). That's reported honestly: it
is exactly the predicted first-order direction, enriched but weak — **PRC2 redundancy + the bistability wall**, not a clean
quantitative cascade.

**The honest three-tier ceiling** (the same shape as the project's knockout-cascade wall, now derived from the chromatin
mechanism): static state is **knowable** (a measured labeling); writer→direct-target is **first-order, direction-only**
(enriched but weak); the **genome-wide decompaction cascade does not compose** — because read-write feedback makes marks
bistable and hysteretic (losing one writer often does nothing; surviving marks re-propagate) and writers are redundant
(EZH1/EZH2, DNMT3A/3B). A real static-state + first-order-writer engine, stopped honestly at the wall. (`colab/epigenetics.py`;
`epi` syscall.)

---

## Protein abundance — closing the lifecycle to protein, and testing the knockout-importance claim

Following the chain [protein] = k_translation · [mRNA] / k_protein_decay: does modeling **translation (production) + protein
degradation** beat the mRNA-level-alone baseline? Target = measured protein `ppm`; base = measured K562 mRNA (RNAdecayCafe RPKM);
translation features (codon composition/optimality, 5′UTR len/uORF/GC, CDS len) + degradation features (protein length, N-end-rule
class, PEST/disorder/hydrophobic fractions from the translated CDS). Chromosome-held-out, 7,515 genes.

```
model               R2      pearson
mRNA-only          0.352    0.594
+ translation      0.501    0.708     <- +0.11: production dominates
+ degradation      0.372    0.610     <- +0.02: protein decay barely helps
full (XGBoost)     0.538    0.734
```

**The production half of the chain works (r 0.59 → 0.73); the protein-decay half barely adds (+0.02).** That's the honest, and
biologically correct, asymmetry — Schwanhäusser 2011: translation is the bigger determinant of protein level, and protein
half-life is degron/context-specific, not sequence-predictable (unlike mRNA half-life). So the abundance lifecycle now closes:
**transcription rate → mRNA (r~0.57) → protein (r~0.73)**, with absolute concentration coming from measured `ppm` where needed.

**Step 5 — does abundance predict a knockout's importance?** Tested directly: protein abundance vs essentiality (dep_frac)
**Spearman +0.35** (essential proteins are ~40× more abundant, median log 1.03 vs −0.59), **on par with network degree (+0.37)**.
So abundance *is* a modest, real predictor of a knockout's **seed** importance — better than "weak." But it ranks the seed, not
the genome-wide response; "how much effect to the whole cell" (the cascade) stays the measured wall. (`colab/protein_abundance.py`;
`pabund` syscall.)

---

## Layers — the graph made manageable, organized by readout

The 889k→1.25M-edge graph was one tangled union. This reorganizes it into **12 named layers in 6 readout tiers**, using the
principle the last few turns *measured*: the cell isn't one network — it's several typed networks, each predicting its **own**
readout, and collapsing them loses that. So layers are grouped by *what an edge means and what it predicts*, not lumped:

```
PHYSICAL     ppi / complex / ligand_receptor        -> binding & complex assembly     [complementary, 46-214x coherent]
REGULATORY   regulatory / signaling / causal        -> transcription (Perturb-seq)    [propagate L1 = 9.2x; sign 73%]
REACTION     reaction_metabolic (substrate->product)-> metabolic flux                 [ecFlux mutation->flux VALIDATED]
DEPENDENCY   codependency / synthetic_lethal / coexpr-> fitness / essentiality         [centrality+paralog AUC 0.86]
SPATIAL      chromatin_loops                        -> enhancer->gene 3D contact      [invivo ABC; nearest wrong 39%]
MEMBERSHIP   reactome_pathway                       -> shared-process grouping        [condition top-1 67%]
```

`layers` prints this tiered manifest; `layers GENE` shows one gene across the tiers (e.g. GATA1: heavy in REGULATORY as a TF,
sparse in PHYSICAL). It's a registry/manifest + per-tier query on top of `network.py`, so you can pull a single layer or a whole
readout tier instead of the union — and, crucially, apply each layer to its *right* target.

The honest record from the reaction-network test is baked into the manifest: a Reactome 15k-reaction co-membership superset
**did not beat abstract PPI for the transcriptional readout** once size-controlled (Wilcoxon p=0.32) — a readout mismatch
(reaction predicts flux, not transcription). Recording it here is the point: the layering exists so each typed network is used
against the readout it actually predicts, not collapsed into one graph and pointed at the wrong target. (`colab/layers.py`;
`layers` syscall.)

---

## The whole model, end-to-end — how much we get

The culminating integration: chain every validated steady-state layer into one pipeline and measure the total against real
ground truth. **genome → regulation (promoter Pol II) → mRNA level → protein level**, scored at each hand-off and end-to-end
vs measured K562 mRNA (RPKM) and protein (ppm), chromosome-held-out, 7,515 genes.

```
STAGE 1  genome/regulation → mRNA:   promoter Pol II only  r 0.29  →  + decay model  r 0.60
STAGE 2  mRNA → protein (oracle):    measured mRNA + sequence         r 0.71   ← ceiling
END-TO-END  genome/promoter → PROTEIN (no measured mRNA anywhere):
   promoter only   r 0.15   →   + decay   r 0.61   →   FULL CHAIN   r 0.63  (R² 0.40)
```

**The steady-state chain composes.** Chaining promoter → decay → translation → protein-degradation retains **r = 0.63 /
R² = 0.40** end-to-end from genome to protein — close to the oracle ceiling of **0.71** you'd get with a perfect (measured)
mRNA level. The gap is the single lossy hand-off: **promoter → mRNA** (one Pol II track is a crude transcription-rate proxy).
Each layer earns its place — the decay model is the big jump (r 0.29 → 0.60 for mRNA), because `[mRNA] = k_txn / k_deg` genuinely
needs *both* the transcription rate and the decay rate.

This is the honest answer to "how much do we get": **the assembled steady-state cell predicts protein abundance from genome +
regulation at r ≈ 0.63 held-out** — a real, composed, quantitative whole-model number, not a demo.

And the honest boundary, unchanged: this is the **steady-state abundance arm**, the part that composes. The **mutation arm**
(NEXUS mutation → network propagation → genome-wide cascade) is separate — near-field validated (regulon 9.2×) and far-field the
measured wall. Assembling the layers didn't move that wall; it produced a genuine end-to-end *abundance* predictor alongside it.
(`colab/wholecell.py`; `wholecell` syscall.)

---

## The fair shot at the third arm — and an honest correction

I've repeatedly called the far-field cascade "~chance." That was loose. Here is the fair, exhaustive test: predict the
genome-wide knockout response (which genes move, vs Perturb-seq) by **combining every layer** into one supervised model —
regulon, RWR over the multi-layer graph, PPI/complex/coexpression, the reaction network, signaling, plus a generic
gene-responsiveness prior — tested on **held-out knockouts** (GroupKFold by knockout, so we predict cascades of genes never
seen perturbed in training).

```
88 held-out knockouts, 23,040 (G,J) pairs, base mover rate 6.2%
model                          AUPRC    lift
J-responsiveness prior only    0.559    8.95×   ← "which genes are generically responsive"
G-specific layers only         0.066    1.06×   ← regulon+RWR+network+reaction, NO prior = ~CHANCE
FULL (everything combined)     0.567    9.08×
full − prior = +0.008          top feature: J_movefreq (0.78); every mechanistic feature ≈ 0.02–0.04
```

**The correction:** the cascade is **not** ~chance — it is **~9× predictable** (AUPRC 0.57). But the fair test reveals *why*,
and it's not what the mechanistic story hoped: the predictability is almost entirely a **generic responsiveness prior** — some
genes (stress/identity programs) move under *many* different knockouts, and simply knowing which genes are generically
responsive predicts the cascade at AUPRC 0.56. The **G-specific mechanistic layers alone are ~chance (1.06×)** for unseen
knockouts, and combining *everything* adds only **+0.008** over the prior.

So the honest, precise reframing of the wall: **the knockout response is predictable, but by gene-intrinsic responsiveness, not
by knockout-specific mechanism.** You can guess *that* a responsive gene will move; you cannot compose *which* of a given
knockout's specific downstream targets move — the mechanistic cascade — for a knockout you haven't measured. GATA1's 9.2×
direct-regulon enrichment was the well-powered *exception*, not a composable rule; pooled across held-out knockouts, mechanism
washes out.

*Caveat (honest):* the responsiveness prior is computed cross-fold (leave-self-out but across all knockouts), so its absolute
AUPRC is mildly optimistic; the leakage-free number is the **G-specific-only 1.06×**, which is the clean measurement — combined
mechanistic layers predict an unseen knockout's cascade at chance. That is the wall, taken at with everything and measured.
(`colab/cascade_all.py`; `cascadeall` syscall.)

### Precision@10 — the concrete "top 10, how many right?"

Ranking the full gene set for each held-out knockout and taking the top 10 predicted movers:

```
                              of top 10, how many actually move
random baseline               ~0/10
G-specific mechanism only     0.0/10   ← the mechanistic layers alone: ZERO
responsiveness prior only     1.3/10
FULL model (everything)       1.8/10   (~18%)
```

**~2 of 10** — but the killer stat: across **88 different knockouts, the top-10 lists use only 57 distinct genes, and the single
most common gene appears in 79/88 of them.** The model predicts **nearly the same 10 genes no matter what you knock out.** So the
~2 hits are generic "usual-suspect" responders, not the knockout's specific cascade — and the knockout-specific mechanism scores
**0/10**. This is the concrete face of the wall: useful as a static "these genes tend to respond" list, useless as "knock out X →
these particular genes."

---

## Engineering roadmap, argued and measured: activity-inference, supervised regulon, and the FBA metabolic bridge

A four-point roadmap proposed to break the far-field wall. Argued each within the project's measured findings, built the three
whose data existed, and measured them honestly. All three are recorded as **precisely-diagnosed negatives that locate the wall**
— not solved. (`colab/viper.py`, `colab/regulon_learn.py`, `colab/metabridge.py`; `viper` and `metabridge` syscalls.)

### Point 4 — VIPER/DoRothEA activity inference (the clear winner conceptually)

Read a regulator's **activity** from its target **footprint**, not its own mRNA (aREA on the canonical CollecTRI signed
regulon). Fixes "activity ≠ abundance" in principle. Validated on Perturb-seq: knocking out TF X should leave X's activity —
computed from its **targets**, zero info from X's own mRNA — among the most-inactivated regulons.

```
recovery AUC (true TF more-inactivated)   0.564   vs shuffled control 0.495   (0.5 = chance)
true TF in top-10 most-inactivated        6.6%
```

**Honest negative, precisely diagnosed** (the method is faithful, not buggy): (1) 97% of pseudobulk TF-KO footprints are
near-empty (<5 movers) — nothing to read; (2) where a footprint exists the **pan-tissue** regulon *mis-signs* the K562 response
(GATA1 sign-agreement **0.33 < chance** — wrong direction); (3) |NES| tracks regulon size (r=0.64). The recurring wall in the
**inference** direction: the generic regulon doesn't carry K562-specific wiring. It *does* work where signal exists — `viper
GATA1` returns MYC/E2F inactivated (cell-cycle arrest) + SPI1 activated (the GATA1–SPI1 antagonism), biologically correct.

### Point 1 — Perturb-seq-supervised K562 regulon (fix the recall collapse)

For the 8 erythroid TFs with genome-wide K562 ChIP: candidates = ChIP-bound promoters, label = moves in that TF's Perturb-seq
KO, model learns the functional subset (binding ≠ function). Honesty gate = **leave-one-TF-out vs a responsiveness prior**.

**Untestable / honest negative:** only **23 functional positives** across 8 TFs (mostly GATA1=19); **6 of 8 TF knockouts have
zero** promoter-bound movers → <4 evaluable held-out TFs, so the AUPRC (mechanism 0.032 vs prior 0.006) is noise. Two real
causes: these TFs act through **distal enhancers** (promoter-binding is the wrong candidate set), and the footprints are too
**sparse** (same emptiness that sank VIPER). No regulon exported, no generalization claimed.

### Point 5 — the FBA metabolic bridge (replace the RWR walk with mass-balance)

The strongest idea, and it matches what `REACTION_CHAIN.md` reached independently: the one place a bypass model is exactly
computable is metabolism. Pipeline: knockout → enzyme capacity (Human-GEM) → **FBA reroute** → metabolite shift →
metabolite-sensing TF (SREBP/LXR/PPAR/FXR/RAR/HIF, curated) → near-field regulon = predicted second wave — two accurate 1-step
hops bridged by a metabolic solve instead of a 4-step guess.

**Correct architecture; it locates the wall, does not break it — three quantified buffering layers:**

```
sole-catalyst genes (no isozyme)       625 / 2848  (22%)   ← the other 78% lose ZERO flux on KO
sole-catalyst biosynthetic enzymes     mostly lose nothing too — open-exchange model IMPORTS the product
                                        (only a few, e.g. DHODH, actually bite)
metabolic KOs in Perturb-seq           1441; only 6 move ≥10 genes  ← closure has no ground truth to validate against
```

The bypass/redundancy that defeats far-field prediction is **baked into metabolism itself** (isozymes + import). The
metabolite→sensor→regulon closure is concrete (shown on *stated* metabolite changes — cholesterol↓→SREBF2, FA↑→PPAR,
succinate↑→HIF — decoupled from the non-unique pfba flux step) but its transcriptional readout is absent in the data. Also
honest: steady-state FBA gives fluxes, not concentration transients ("substrate spikes" need dFBA/kinetics we only partly have).

**Bottom line for the roadmap:** the two deferred points were the honest calls (Point 3 ODE needs time-course data we don't
have; Point 2 chromatin-imputation needs multi-cell-type paired data and is partly circular). The three built points all hit the
**same two-headed wall from different directions — pan-tissue/generic wiring that doesn't transfer to K562, and Perturb-seq
footprints too sparse to supervise or validate against.** That convergence, measured three independent ways, is the finding.

---

## "Replace the Random Walk (Stage 4) with a Continuous Equivariant Tensor Field" — built and measured

Directly testing the proposal: signal shouldn't diffuse as an equal scalar down every edge — carry a directional/tensor field
instead. So each gene is a point in a manufactured continuous geometry (randomized-SVD spectral embedding of the reg+PPI graph —
there is **no physical 3D substrate**: `loops3d` is 767 anchors, no Hi-C, so E(3)-equivariance is literally undefined and the
geometry is *learned*). The knockout seeds a scalar RWR field **and** an E(d)-equivariant vector field
`v_i = Σ_j K_ij h_j (x_j − x_i)`; the rotation-invariant readouts `‖v‖` and alignment go into the **same xgboost, GroupKFold by
knockout** as `cascade_all`. Parameter-free operator on purpose (a trained net can silently relearn the prior — and a torch EGNN
was tried first but sparse-autograd **segfaulted in-sandbox**, so the fixed operator is the robust equivalent).
(`colab/tensorfield_cascade.py`.)

```
model (held-out, 88 knockouts, base 6.2%)          AUPRC    lift
responsiveness prior only                          0.5542   8.87×   (control)
RWR scalar mechanism  (the thing being replaced)   0.0637   1.02×   ← the wall
TENSOR FIELD mechanism (‖v‖ + alignment)           0.0699   1.12×   ← the replacement
RWR + tensor field (mechanism)                     0.0841   1.35×
FULL (+ prior)                                     0.5588   8.94×
```

**Honest negative.** The continuous equivariant tensor field lands at the **same chance-level wall as the scalar RWR** (1.12× vs
1.02×). The directional/rotation-equivariant information carries only a *sliver* (combined 1.35×), a rounding error next to the
responsiveness prior (8.87×), and adding it to the full model moves nothing (0.559 vs 0.55). This is the **decisive** evidence
that Stage 4's far-field wall is **informational, not representational**: the missing quantity is the per-edge *transmission
coefficient*, which is not in the graph geometry at all — so no propagator (scalar, vector, or tensor; learned or fixed;
equivariant or not) can manufacture it. Consistent with the earlier learned-GNN head-to-heads (R-GCN / GraphSAGE never beat fixed
propagation for the far-field). Caveat kept explicit: with no physical 3D, the equivariance is over a *manufactured* embedding, so
this fairly tests richer geometric/tensor propagation — not physical E(3) symmetry, which would need Hi-C coordinates we don't have.

---

## What would break the far-field wall? — the learning-curve diagnosis

The decisive test of *why* held-out G-specific mechanism sits at chance: grow the number of training knockouts against a **fixed
held-out set** and watch whether mechanism recall rises (data-volume problem) or stays flat (data-*kind* problem).
(`colab/wall_diagnosis.py`.)

```
#train KOs   MECHANISM lift   PRIOR lift
    20            1.03×          3.09×
    40            1.01×          4.31×
    80            1.03×          6.12×
   160            1.00×          7.81×
   178            0.96×          7.79×
```

**Mechanism is flat at chance while the prior climbs 3×→7.8×.** More single-knockout steady-state Perturb-seq does **not** break
the wall — it only sharpens the generic responsiveness prior. The per-edge transmission coefficient is not being under-sampled;
it is **not recoverable from single-KO pseudobulk endpoints at any scale** (direct vs indirect are confounded; buffering is
unobserved). What would break it is a *different kind* of measurement that observes the coefficient directly:

1. **Combinatorial double-KO Perturb-seq** (Norman 2019 GI, K562, same context) — observes epistasis/buffering directly: if KO(A)
   is silent but KO(A)+KO(B) is not, you've measured the bypass. This is the most on-point, and it's in-context.
2. **Time-course / 4sU nascent-RNA** after perturbation — separates direct targets (early) from cascade (late), de-confounding
   the steady-state mixture that makes single-KO endpoints unpredictable.
3. **Dense multi-context Perturb-seq** (Replogle–Nadig multi-cell-line) — learns conserved transmission vs context-specific
   responsiveness. (Caveat: this session's RPE1 check already showed weak cross-line transfer, r≈0.19 — so multi-context helps
   least of the three.)

Honest ceiling note: even with these, "predict which specific genes move for an arbitrary knockout" is partly irreducible —
biological buffering and stochasticity mean the ~9× responsiveness prior may be near the achievable far-field ceiling. The
realistic win is *measurable G-specific signal above the 1× mechanism floor*, which combinatorial and time-course data are where
it demonstrably exists.

---

## Getting the right data: combinatorial double-perturbation (Norman 2019 K562) — the wall reproduces one level up

The learning curve proved single-KO data can't break the wall, so I fetched the *kind* the diagnosis pointed to:
**Norman & Weissman 2019** combinatorial CRISPRa Perturb-seq (K562 — our own context), 699 MB from scPerturb/Zenodo, 111k
cells, 105 singles + 131 doubles. The test measures the buffering/interaction signal single-KO endpoints structurally cannot
see: **epistasis = double response − additive(sum of singles)** — and whether it's predictable for *held-out pairs*.
(`colab/norman_epistasis.py`.)

```
additive (sum of singles) predicts the double:     r = 0.858
epistasis variance fraction (non-additive):        46%
held-out epistasis prediction (GroupKFold by pair):
   saturation-only  (from additive magnitude)      r = 0.45
   + the two singles' per-gene effects             r = 0.459
   + relational pair features (PPI/reg/pathway)    r = 0.45
   => PAIR-SPECIFIC gain over generic saturation   −0.001
```

**The wall reproduces exactly, one level up.** Epistasis is large (46%) and predictable (r=0.45) — but the control shows it's
**~entirely the generic saturation nonlinearity** (a gene strongly moved by both singles sub-adds in the double). Pair identity —
*which* two genes, and whether they're connected/co-pathway in the graph — adds essentially **nothing** on held-out pairs
(+0.009 from the singles, −0.001 from relational features). This is the **same generic-predictable / specific-unpredictable
split** as single-KO: there, the generic *responsiveness* prior predicted and the *G-specific* mechanism was chance; here, the
generic *saturation* nonlinearity predicts and the *pair-specific* interaction is chance.

**So what would break the wall — the honest, now-measured answer:** the missing quantity is a *specific* coefficient (which
gene→which gene, which pair→which interaction), and across every probe — single-KO scale, equivariant tensor fields, metabolic
mass-balance, and now combinatorial data in-context — the **specific** signal fails to generalize while the **generic magnitude**
signal is easy. The far-field wall is not a data-volume problem and not (only) a data-*kind* problem within transcriptomics; it is
that gene-identity-specific transmission generalizes poorly, period. **Honest caveats:** CRISPRa (activation) doubles, only 131
pairs, a coarse 3-feature relational set; the GEARS/GI literature reports *modest* above-chance pair-level GI prediction with more
pairs and richer features — so this bounds the claim rather than closing it. The genuinely different lever left untested here is
**time-resolved** data (4sU/nascent-RNA), which de-confounds direct from indirect rather than adding more endpoints — that remains
the one measurement type this project has never been able to source.

---

## The NEXUS Thermodynamic Mass-Action Sub-Network Simulator — computing the transmission coefficient from first principles

The one proposal that attacks the *actual* missing quantity (the per-edge transmission coefficient) with real physics instead of
a fancier propagator: treat each PPI edge as a coupled binding equilibrium `Kd = [A][B]/[AB] = exp(ΔG/RT)`, let a mutation's
NEXUS ΔΔG shift one interface's Kd (`Kd_mut = Kd_wt·exp(ΔΔG/RT)`), and re-solve the whole system so the perturbation propagates
*quantitatively* downstream. (`colab/nexus_massaction.py`.)

**Tested four ways — the physics is correct:**
```
[1] SOLVER       2-body [AB] = 0.5 (matches analytic exactly); mass conservation residual 0.0
[2] FUNNELING    weakening A:B (+4 kcal) reroutes A from B onto Z (0.06→0.36) — mass action routes by affinity, kills the hairball
[3] TRANSMISSION downstream ABCD collapses 1.0→0.43 as ΔΔG→4; coefficient d ln[Y]/d ΔΔG = −0.20 /kcal
[5] MAPK demo    active ERK-complex fraction 1.0→0.29 as RAF:MEK ΔΔG→3
```

**The payoff — run on real data, the same physics reproduces the wall and names its cause:**
```
[4] SKEMPI (6,798 real measured ΔΔG): mass action SELF-BUFFERS. Median destabilizing mutation loses
      87% of complex near Kd  →  but only 18% when [conc] = 100·Kd (saturated).
[5] REACH: at cellular saturation the cascade collapses to an effective reach of ~2–3 hops
      (fractional terminal loss 54% → 3% → ~0 by node 5–8) — MIRRORING the measured regulon decay 9.2×→2.1×→1.08×.
```

So the far-field decay we measured all session is **re-derived from first principles**: it is not topological dilution (an
obligate chain near Kd reaches deep, 63% at node 8) — it is **per-step saturation-buffering**, because cellular complexes sit at
[conc] ≫ Kd and are saturated, so mutations barely move them. This is the physical mechanism behind every "buffering" result in
this project (bypass in metabolism, epistasis-saturation in Norman, RWR dilution).

**The honest verdict (arguing with the idea):** the engine is *correct* and genuinely computes the transmission coefficient —
but only where it's parameterizable, and that's the catch the user pre-named: (a) we have WT Kd + absolute concentrations for
curated pathways and SKEMPI, not for all 191k PPI edges, so it's a **sub-network** engine, not whole-cell — it cannot itself
break the *genome-wide* wall; (b) binding ≠ catalysis (no kcat for kinase steps); (c) it closes to the transcriptional far-field
only through a *terminal TF's* near-field regulon — the same two-hop structure as the metabolic bridge. **Net: the right physics.
It doesn't remove the wall; it explains it, and it delivers a validated mutation→known-pathway-output quantifier for the circuits
we can parameterize** (variant/drug effect on MAPK, apoptosis, etc.) — which is a genuinely useful, honest deliverable.

---

## Found and tested the last lever: time-resolved perturbation data (RENGE / GSE213069)

The one lever this project had never sourced. Fetched **RENGE (Ishikawa 2023, GSE213069)** — hiPSC, **23 pluripotency-TF CRISPR
knockouts, genome-wide scRNA-seq at 4 time points (day2–day5)** with CRISPR guide capture (~375 MB, GEO). Knocking out
OCT4/SOX2/NANOG etc. drives a differentiation cascade over days, so in principle early = direct targets, late = cascade.
(`colab/renge_timecourse.py`.)

```
[A] direct-target (regulon) enrichment among movers, by day:   2.0× → 2.4× → 1.9× → 30× (day5)
[B] held-out (GroupKFold by KO) MECHANISM lift, by day:        1.18 → 1.24 → 1.21 → 1.20   (FLAT)
    held-out prior lift, by day:                               5.3  → 8.2  → 8.5  → 7.8
```

**The wall held, even with time.** Held-out mechanism lift is **flat at ~1.2×** at every time point — predicting which genes move
for a *held-out* knockout does not improve with time-resolution here; the responsiveness prior still does all the work. The KO's
own direct regulon *is* strongly enriched among its movers (up to 30× by day5), but that's **descriptive** — it requires knowing
the TF and having its curated regulon (driven by the annotated core), not blind held-out prediction.

**Two honest confounds bound this result (it's not a clean disproof of the lever):**
1. **Wrong timescale.** RENGE uses CRISPR-KO — protein depletes over *days*, so even day2 is post-establishment. This is not the
   *minutes-to-hours* window a **degron (dTAG) or 4sU** metabolic-labeling experiment gives, where direct and indirect cleanly
   separate. The RENGE window samples the differentiation cascade, not the primary response.
2. **Too few TFs.** 23 knockouts is underpowered for held-out generalization, and most lack curated regulons to transfer.

**Verdict on the lever:** bounded, not disproven. Time-resolution is the right idea and the direct-target enrichment confirms the
biology, but the definitive test — a **fast-degron / 4sU, many-TF, dense (hours) time-course in K562** — does not exist as clean
public data. That is the honest frontier: not a modeling gap, a measurement that hasn't been made at the scale and timescale the
question needs.

---

## The mRNA-dynamics ODE with MEASURED K562 rates — the response-time half, validated

We don't have a K562 *perturbation* time-course, but we do have K562 4sU labeling-time snapshots of the *resting* cell
(RNAdecayCafe SLAM-seq), which give per-gene synthesis and decay **rates**. Those parameterize the dynamics ODE with real
numbers instead of guesses. (`colab/kinetics_ode.py`.)

```
d[mRNA]/dt = k_syn − k_deg·[mRNA]      steady state [mRNA]_ss = k_syn/k_deg
response time: the transition is governed ENTIRELY by k_deg → t_half-response = ln2/k_deg = the mRNA half-life
```
The mRNA half-life **is** the response time — a gene can only change as fast as its old copies decay.

```
8,867 K562 genes; ODE solver exact (Euler vs analytic err 1e-6)
half-life (= response half-time):  median 3.0 h   (q10 1.0 h → q90 8.4 h)
t90 response time (ln10/k_deg):    median 10 h     (fast 3.2 h → slow 28 h)

VALIDATION 1 — immediate-early genes respond fast:  13 IEGs median 0.47 h vs genome 3.0 h
               = 6.4× faster, Mann-Whitney p = 4×10⁻⁹   ✓ (built to respond fast, as τ=1/k_deg predicts)
VALIDATION 2 — TFs turn over faster than non-TFs:   686 TFs median 1.9 h vs 3.1 h, p ≈ 0   ✓

APPLICATION — time-resolved near-field (GATA1 KO): its regulon (9.2×, predictable) now has a WHEN —
              NFE2 (t90 2.6 h), RUNX1/WT1 (3.4 h), GFI1B (3.8 h) move fast; others slow.
```

**What this adds:** the **time axis** the steady-state models never had, from *measured* rates and *biologically validated*
(IEGs 6.4× faster). For the near-field regulon — the part we *can* predict — we can now say not just *which* direct targets move
but *when*. **Honest limit, unchanged:** this is the DYNAMICS of each gene given a synthesis change; it does **not** supply the
TRANSMISSION (which genes' k_syn a knockout actually changes = the far-field wall). It's resting-cell kinetics, not a perturbation
time-course — so it upgrades the near-field into a time-resolved readout, and cannot by itself break the far field.

---

## Does "how fast" (half-life), combined with everything, make a difference? — tested

The intuition: maybe combining the measured response speed (half-life) with the rest moves the cascade. Tested in the exact
cascade_all held-out protocol (`colab/kinetics_prior.py`), 20,351 (G,J) pairs, 237 knockouts.

```
Q1  does half-life EXPLAIN the responsiveness prior?   Spearman(half-life, movefreq) = 0.007  → NO, independent axes
Q2  held-out AUPRC lift (chance = 1.0×):
      mechanism only (reg/ppi)                 0.99×   (transmission = chance, as always)
      half-life / k_deg only (measured)        4.04×   ← its own real, non-leaky generic predictor
      movefreq prior only (leaky)              8.30×
      mechanism + kinetics                     4.32×   ≈ kinetics-alone → mechanism adds ~nothing
      kinetics replaces movefreq (no leak)     5.92×
      full (everything)                        8.64×   (+0.34× over movefreq)
```

**A real but bounded difference — exactly where theory says it should land.** Three findings:
1. Half-life does **not** explain the responsiveness prior (correlation ~0). "Twitchy" and "short-lived" are *independent* axes —
   that part of the intuition is wrong.
2. Half-life **is** its own measured, **non-leaky** generic predictor of movers (4.04× held-out) — weaker than the leaky
   movefreq prior (8.3×) but leak-free, so it's an honest prior for a genuinely-new knockout where movefreq can't be computed,
   and it adds a small real gain stacked on top (full 8.64× vs 8.3×).
3. It adds **no transmission**: mechanism+kinetics (4.32×) ≈ kinetics-alone (4.04×); the mechanism (which genes X *hits*) stays
   chance. Half-life is a property of the *target gene J*, not of the *G→J edge* — the wrong side of the equation for the wall.

So combining "how fast" with everything makes a genuine difference on the **generic axis** (a cleaner prior + the near-field
time-stamp from kinetics_ode) but does **not** break the far-field wall. The instinct was partly right — it adds real signal —
it just lands on responsiveness, not transmission.

---

## Boosting the interconnections — weights + conditions + rates on the edges (controlled honestly)

The directive: enrich the network's interconnections with edge WEIGHTS, CONDITIONS (when an edge fires), and RATES (how fast) —
the exact missing ingredients of the transmission coefficient. Built (`colab/cond_network.py`): weighted edges (reg-sign,
coexpression strength, co-dependency strength, signaling), every edge gated on BOTH endpoints being expressed in K562 (removing
edges that can't fire), and each target annotated with its measured mRNA half-life (response rate). Weighted-RWR propagation vs
the plain binary graph, held-out cascade, 25k pairs, 237 knockouts.

```
plain binary graph (reg + RWR)                          1.01×
enriched conditional-dynamic (all features)             3.64×   ← looks like a boost
  → G-SPECIFIC relational only (cond-RWR/coexpr/codep)   1.12×   ← the real transmission test
  → GENERIC per-gene only (half-life/expressed)          3.62×   ← what actually drives it
responsiveness prior only                               8.65×
```

**The boost is generic, not transmission — and the decomposition control is what proved it.** Enriching the edges appears to lift
the mechanism 1.0×→3.6×, but almost all of that is the generic per-gene features (half-life + expressed = 3.62× on their own,
consistent with `kinetics_prior`). The purely G-specific relational part — the conditioned, weighted edges plus coexpression and
co-dependency — gives only **1.12×**, essentially the plain graph. So weighting/conditioning/rating the interconnections sharpens
edge quality and re-imports the generic half-life prior, but it does **not** manufacture the per-edge *transmission coefficient*.

This is the same lesson every enrichment has taught, now controlled directly on the specific "rates + conditions" idea: **the
far-field wall is a measurement gap (the unmeasured per-edge transmission coefficient), not a graph-richness gap.** No amount of
weighting, conditioning, or rate-annotating the edges we have supplies a number that was never measured — you can only *read* the
transmission coefficient from a perturbation experiment, not compute it from a richer static graph. The honest control (splitting
G-specific from generic) is what keeps this from looking like a win it isn't.

---

## Verifying the labelled network against knockout ground truth — which edges are real?

Instead of predicting the far cascade, verify the network we labelled: for each labelled edge type A→B, when we knock out A
(K562 Perturb-seq), does B actually move? Enrichment over base = whether that edge *type* carries real functional signal.
(`colab/pathway_ko_verify.py`, 237 knockouts.)

```
edge type                         n tested    enrichment
regulatory (curated TRRUST)            99        6.0×    ← VERIFIED real (transcriptional, near-field)
coexpression                        2,631        1.44×   ← weak but real
PPI (physical)                     12,553        1.29×   ← weak but real
co-dependency (fitness)             1,241        1.15×
regulatory (bulk inferred, 612k)   10,376        0.97×   ← NOISE (chance)
signaling                             245        0.0×
complex co-membership               3,686        0.0×    ← NOT transcriptional
pathway co-membership                 483        0.0×    ← NOT transcriptional
pathway cases (KO inside a pathway → members co-move):  0 qualify
```

**A clean QC of the network, per label type — this is genuinely useful:**
- **Trustworthy:** the *curated* regulatory edges (TRRUST TF→target, 6×). Knock out a TF, its curated targets move. This is the
  transcriptionally-real sub-graph.
- **Weak but real:** coexpression (1.44×) and PPI (1.29×) — co-expressed / physically-bound partners co-move a little.
- **Noise:** the **bulk inferred** regulatory network — 10,376 tested edges at **0.97× = chance**. Most of the 612k "reg" labels
  do **not** predict KO effects and should be treated as low-confidence. (Only the curated subset is real.)
- **Not transcriptional:** complex/pathway co-membership and signaling (~0×). Knocking out one member of a complex or pathway
  does **not** move the others' mRNA — these labels mark *protein/functional modules*, not transcriptional co-response. No pathway
  showed member co-movement above threshold.

**How a KO affects the network structure (aim 1):** the perturbation propagates through **directed regulatory** edges (TF→target),
not through physical/membership structure — a complex member's knockout doesn't transcriptionally ripple to its complex-mates. So
the "how does a KO affect the network" answer is: through the curated regulon, and there it's real (6×); everywhere else the
labelled structure describes who-groups-with-whom, not who-moves-when-you-knock-out. This both **verifies** the network (curated
regulon = real; bulk-inferred reg = noise) and shows which layer actually carries KO transmission.

## Doing it step by step — one validated hop, re-lay-out the graph, next hop (where does the chain die?)

The idea: stop predicting the far field all at once. Take **one** validated near-field hop — the curated regulon (~6×) — then
**re-lay-out the graph**: which of the predicted movers are themselves TFs ("carriers")? Seed the *next* hop from only those,
and iterate. Score at **every** hop so you can watch exactly where a chained-one-step cascade holds and where it dies. Built
(`colab/iterative_cascade.py`): discrete, **sign-aware** (compose activation/repression signs down the chain), **TF-gated** (only
carriers seed the next step). Pooled over the 24 K562 TF knockouts that have a curated TRRUST regulon (17 reach scoreable steps),
scored against Perturb-seq.

```
step   new pred  correct  precision   ENRICH  sign acc  cum recall
step1     3          2       2.1%      6.63×     50%       0.2%     ← the validated regulon (small N)
step2    26          3       0.4%      2.83×     67%       0.7%
step3   164         21       0.6%      3.78×     52%      15.8%
step4   370         27       0.5%      3.12×     44%      31.8%     ← sign now worse than a coin flip
```

**It does not die the naive way I expected (a clean monotonic decay). Two things happen at once, and together they *are* the wall:**

- **(A) Precision collapses after hop 1 and never recovers.** Step 1 (the direct curated regulon) is 2.1% correct at 6.63×
  enrichment — real, but tiny in absolute terms (2 pooled hits). Steps 2–4 sit at the ~0.5% floor. Only the first hop is
  meaningfully better than a coin toss about *which* genes move.
- **(B) Enrichment does *not* decay — it plateaus at ~3×. But that plateau is the responsiveness prior re-entering by breadth.**
  By step 3–4 the TF-gated fan-out has ballooned to a median of 164→370 predictions and simply *reaches the generically-responsive
  hub genes* — the ones that move under almost any knockout. The tells that it's generic, not transmission: precision is pinned at
  the floor, and **sign accuracy drifts to chance/below** (50→67→52→**44%** — worse than a coin flip by step 4), so the composed
  up/down direction is no longer preserved.
- **(C) Cumulative recall reaches 32% by step 4 — but only by carpet-bombing.** You must emit ~370 predictions to recover that
  fraction. Recall is available *only* at <1% precision — the wall stated as a trade-off.

**The GATA1 layout makes it concrete:** step 1 → 56 predicted (14 are TFs, carry forward); step 2 → 204 (35 carriers); step 3 →
806 (146 carriers). The prediction set balloons via carriers while the confident signal is already gone.

**Bottom line:** doing it step-by-step and re-laying-out the graph each hop is **more honest and interpretable than diffuse RWR** —
you can point to exactly which TF carries what and watch confidence evaporate hop by hop — and it confirms the wall from a new
angle. The **only** genuinely confident hop is step 1, the validated near-field regulon. Every subsequent hop is either
precise-but-negligible or broad-but-generic; the chain never compounds into a correct far field, because each discrete step carries
**sign + topology but not per-edge magnitude** (the transmission coefficient). After one hop the confident signal is gone and only
the generic responsiveness prior — reached by breadth — remains. (Even step 1's sign is at chance here on small N, matching the
earlier finding that the curated regulon predicts *which* genes move ~6× far better than the *direction* they move.)

---

## Rebuilding the graph as a typed causal reaction network (CellOS 2.0, step 1) — SIGNOR + ComplexPortal + Reactome

The directive: stop drawing flat `A — B` lines and rebuild every interaction as a **typed reaction** — `source --(effect / mechanism /
residue)--> target` — where the effect is signed *and* typed as activity-change (signaling state), quantity-change (transcription),
or complex-formation. Fetched and parsed three curated sources: **SIGNOR 2.0** (43k signed causal relations with mechanism, residue,
and confidence), **ComplexPortal** (2,498 named macromolecular *products*, each with GO function), and **Reactome** (58 MB
interactions). Built 24,210 typed protein→protein causal edges mapped to our gene universe, then verified against K562 Perturb-seq.

```
COVERAGE — the typed causal core:
  SIGNOR protein→protein causal edges   24,210   {quantity 5,765 · activity 7,521 · complex 5,394 · other 5,530}
  ComplexPortal named products           2,498   (all with GO function)
  vs current graph: 612,133 bulk-reg (0.97× NOISE) + 191,447 flat PPI; trustworthy core was 9,396 TRRUST edges

EXAMPLE TYPED REACTIONS (signed / mechanism / residue):
  PTPN1 --−/dephosphorylation@Tyr1190--> INSR
  PRKCA --+/phosphorylation@Thr159--> SRF

VERIFICATION vs Perturb-seq:
  T3 signaling A→TF → TF's regulon      7.62×   (n=1,250)   ← the interesting positive
     shuffled-KO generic control        1.91×   (sd 1.0, 30 shuffles)   → bridge ~4× above generic, several sd out
     direct TF→its regulon (baseline)   6.32×
  T2 generic causal hop-2 reach         1.79×   ← does not compound
  T1 per-class direct edge verify       underpowered (n=69/136 intersect the KO set) — 0.0× is small-N noise, not a result
```

**Two clean results and one honest limitation:**

- **Coverage win (real, needs no wall to fall).** The signed/typed/residue-resolved causal edges plus named products-with-function
  *replace* the 612k bulk-inferred reg edges (measured at 0.97× = chance) and upgrade flat PPI lines into **directional reactions
  carrying a residue**. A mutation now lands on a specific residue of a specific reaction — which feeds NEXUS directly. The
  trustworthy, mutation-addressable fraction of the graph goes up.
- **The signaling→terminal-TF→regulon bridge — the positive that beat my prediction.** I expected composing a signaling edge with a
  regulon to be generic-dominated. It wasn't: **7.62× vs a 30-shuffle generic control of 1.91× (±1.0)** — ~4× above the generic
  floor, several sd out, and this is *after* excluding any target A already regulates directly. So it's a true 2-step
  signaling→transcription bridge, and it's a genuine **capability extension**: you can now predict the transcriptional footprint of
  knocking out a *signaling protein* (a kinase/phosphatase has no regulon of its own) by routing through its terminal TF — something
  the flat regulon alone could not do. Bounded, though: near-field (2 hops, terminal-TF-gated), **descriptive** (needs A's curated
  annotation), and it does not compound (generic hop-2 reach was 1.79×).
- **Honest limitation.** The direct per-class edge verification is underpowered — only 69/136 SIGNOR transcriptional/activity edges
  have a source that is also CRISPR-knocked-out here with a measured target. Those 0.0× readings are small-N noise, *not* evidence the
  classes fail.

**Bottom line:** the rebuild is worth doing and is now started — it swaps 612k noise edges for a signed, typed, residue-level causal
core (mutation-addressable, a real signaling layer, an explicit universal-biochemistry / cell-specific-mask split), and it *extends*
the trustworthy near-field one signaling hop upstream. As predicted, it **explains** the far-field wall rather than breaking it: the
causal edges carry sign and mechanism, not the per-edge magnitude, so far-field mRNA prediction past the terminal TF remains a
measurement gap. (`reaction_graph.py` → reaction_graph.json.)

---

## CellOS 2.0 runtime, built both ways — boolean vs magnitude termination, gates measured (the blueprint, tested)

Took the typed causal graph and implemented the blueprint's runtime loop as an honest experiment: apply the cell-specific mask
(spatial compartment + abundance/K562-expressed gates) to get an **Operational Interactome**, propagate step-wise from each
knockout, read out mRNA at terminal TFs (respecting the readout mismatch — only mRNA-terminal predictions are scored), and compare
**four conditions** — {ungated, gated} × {boolean termination, magnitude termination} — against a shuffled-KO generic control.

```
condition            FULL SET (n_pred med/KO prec ENRICH)     TOP-10 ranked (prec ENRICH)
ungated_boolean      10,373   265   0.004   3.38×             0.003    2.54×
ungated_magnitude     3,640    99   0.006   4.19×             0.016   13.61×
gated_boolean         9,837   265   0.004   3.13×             0.003    2.67×
gated_magnitude       3,408    68   0.006   4.21×             0.016   13.61×
generic control (shuffled-KO):   full set 3.37× (sd 0.7)   |   TOP-10 3.86× (sd 2.94)
```

**Q1 — does gating help?** Barely, here. Spatial + abundance prune only **8.2%** of state edges and move enrichment 4.19×→4.21×.
Gating removes biochemically-impossible edges (real, and it raises precision of the *possible* set) but it does **not** manufacture
magnitude — exactly the `cond_network` lesson (abundance-gated G-specific signal was 1.12×). The gates would matter more for
*cross-cell-type transfer*, which this test doesn't exercise. **An agent handed the blueprint would over-credit the gates.**

**Q2 — whose termination law wins? (the real result)** **Magnitude beats boolean, decisively, at the actionable end.** The
blueprint's boolean law ("reach = full strength, die only at a closed gate") over-propagates into a 265-gene/KO unranked set at
~3.1× — a cleaner hairball. The magnitude law (attenuate by SIGNOR confidence × per-hop factor, threshold, and **rank**) predicts
68 genes/KO and its **ranked top-10 hits 13.61× vs boolean's unranked top-10 at 2.67×**. Mass-action attenuation concentrates the
signal; boolean smears it. **Ship the magnitude termination law, not the boolean one.**

**Q3 — does it break the far field?** No — and the control says exactly where the line is. The **full-set** enrichment (4.21×) is
barely above the generic full-set floor (3.37×): the broad prediction is mostly the responsiveness prior, as always. But the
**ranked top-10** (13.61×) is **clearly specific** — ~3 sd above the top-10 generic control (3.86× ± 2.94). So the confident top
predictions are a *real, specific* signal: the terminal-TF near-field, reached through gated signaling hops. It does not compound
into a far field (only 31 KOs even route to a terminal TF within 3 hops — the scope is signaling-protein knockouts, not
genome-wide).

**The verdict for CellOS 2.0:** build it — bipartite typed graph + cell-mask gates + **magnitude (mass-action) termination**. That
yields a cleaner, mutation-addressable Operational Interactome whose confident predictions for signaling-protein knockouts are
genuinely specific (a capability the flat regulon lacked). But architecture does not recover far-field predictability: the broad
response stays generic, and the per-edge magnitude past the terminal TF remains the measurement gap. The blueprint's data model is
right; its boolean termination and its implicit "gating makes the far field predictable" promise are the two things to correct
before handing it to a coding agent. (`reaction_sim.py` → reaction_sim.json.)

---

## The full stack, run end to end: 10 predictions per knockout, how many are true

The concrete forward test — no enrichment multiples, just "emit 10 predictions, reveal the measured answer, count the hits."
The full-stack ranker uses mechanism first (curated regulon tier → the CellOS-2.0 signaling→terminal-TF→regulon bridge tier)
and backs off to the leave-one-out responsiveness prior (held-out for the target knockout), takes the top 10, and scores against
K562 Perturb-seq. Run over all 237 knockouts, no cherry-picking.

```
KNOCKOUT   moves   full-stack top-10                                              hits
GATA1       68     PRG2 ALAS2 GYPB GP1BB HEMGN [DPYSL2✓] LTBP4 [BST2✓] TAL1 EPOR   2/10
RIPOR1      78     [PRG2✓] [FTH1✓] MT-ATP6 TPT1 MT-CO3 [MALAT1✓] MT-CO2 ALAS2 ...   3/10
EIF2S1      14     APOE TRIB3 HSPA5 SIRT2 DDIT4 SIRT1 ATG5 VEGFA NDC80 ATF3         0/10  ← textbook ISR targets, right biology, data-blind

AGGREGATE (237 KOs)      avg/10   median   ≥1 correct   best
  full stack (mech+prior)  1.00      1         56%        7/10
  prior only (generic)     1.07      1         60%        7/10
  → mechanism adds        −0.06 hits/10
```

**The answer: about 1 in 10.** Best-powered knockouts reach 3–7; the median knockout gets exactly 1; many get 0.

**The mechanism layer does not improve the absolute count.** The entire CellOS-2.0 causal rebuild — the typed reactions, the curated
regulon, the signaling bridge — adds **−0.06 hits/10** over the dumb generic prior. Its members are genuinely enriched (6–13×) but
they are *less* likely to actually move than the top handful of usual-suspect genes, so tiering them into the top-10 slightly
displaces better bets. The one prediction that lands is a **generic** mover (FTH1, MALAT1, PRG2 — genes that move under many
knockouts), not knockout-specific biology.

**And "0/10" sometimes means the biology was right and the data was blind.** EIF2S1 (eIF2α) knockout predicted ATF3, DDIT4, TRIB3,
HSPA5 — the canonical integrated-stress-response targets, exactly correct mechanistically — and scored 0, because the sparse
pseudobulk didn't register those 14 movers as those genes. So this honest count, if anything, *understates* mechanistic
correctness; it is the true score against measured ground truth.

**Why the ceiling is here:** precision@10 is pinned by (a) the far-field wall — specific transmission is unpredictable held-out,
so only the generic prior works — and (b) data sparsity — the median knockout moves only 6 genes out of ~8,000 at pseudobulk, so
even a perfect oracle is capped. Denser per-knockout signal (single-cell, or a 4sU/degron time-course), not more graph, is the only
lever that moves this number. (`predict10.py` → predict10.json.)

---

## Is the 1/10 ceiling the model or the data? — re-run against better-resolved ground truth

The forward test said ~1/10. To find out whether that's the *model* or the *sparse pseudobulk* (median knockout moves only 6 genes),
I re-ran the identical full-stack predictor against better-resolved ground truth. We don't have raw single-cell K562 (our Perturb-seq
is already Replogle's per-perturbation pseudobulk z-scores), so the honest proxy is the **deeper** k562.h5ad essential-gene screen
(many more cells per perturbation, wider z dynamic range) vs the shallow genome-wide gwps, plus a mover-threshold sweep.

```
dataset                          thr   #KO   movers/KO   FULL/10   PRIOR/10   mech adds
genome-wide (shallow) baseline   2.0   237      6          1.00      1.07       −0.06
genome-wide (shallow)            1.5   507      6          1.17      1.21       −0.04
genome-wide (shallow)            1.0  1125     10          1.77      1.81       −0.04
essential (DEEP) resolved        2.0   232      6          1.33      1.34       −0.01
essential (DEEP) resolved        1.5   448      8          1.56      1.57       −0.01
essential (DEEP) resolved        1.0   824     16          2.04      2.05       −0.01
```

**Yes — the ceiling was partly a data-sparsity artifact.** At matched threshold, deeper data alone lifts precision@10 from **1.0 →
1.33/10 (+33%)**; adding a richer mover set takes it to **2.04/10** (median 16 movers). With only ~6 measured movers a top-10
*physically* cannot score high; give it more real movers and the absolute count roughly doubles. So the honest headline isn't
"~1/10", it's "**~1/10 on shallow data, ~2/10 on the best-resolved data we have.**"

**But two things keep this from being a break of the wall:**

1. **Mechanism still adds ~0 at every single setting** (−0.01 to −0.06 hits/10). The entire lift comes from the generic
   responsiveness prior having *more true movers to hit* — not from any new knockout-specific transmission. Better data makes the
   *generic* predictor look better; it does nothing for the *specific* one.
2. **Part of the lift is a looser threshold** admitting easier (noisier) targets, not purely more real signal.

**Net:** the ~1/10 number was depressed by pseudobulk sparsity, and with denser measurement the true number is ~2/10 — but it's
still the generic prior doing all the work, and the far-field specific-transmission wall is exactly where it was. The lever that
would raise the *specific* count — per-edge magnitude from a degron/4sU time-course — remains the missing measurement, not something
resolution alone can supply. (`predict10_deep.py` → predict10_deep.json.)

---

## Adding a biology-expert reasoning layer on top — does reasoning beat the mechanical predictor?

The idea: put a reasoning layer over the full stack — give a biology expert (the LLM) all the context *except* the answer, and let
it rerank the predictions. Strict held-out protocol: a script dumps each knockout's identity, function, network context (regulon,
signaling partners) and the full-stack candidate pool, but **hides which genes actually moved**; the reasoner commits a top-10 per
knockout **in writing before any reveal**; a second script reveals and scores. Panel = 6 well-powered K562-deep knockouts —
GATA1 (erythroid TF) plus five essential core-machinery genes (POT1, SUPT6H, CHMP3, RPL7A, AQR).

```
knockout   movers   REASONED   full-stack   prior    reasoned correct
GATA1        295      0/10        2/10       1/10     — (bet erythroid ALAS2/GYPB/HEMGN/GP1BB — none moved)
POT1         201      2/10        2/10       2/10     SNHG1, EEF1A1
SUPT6H       168      4/10        5/10       5/10     (bet histone HIST1H1C — cost a slot)
CHMP3        126      3/10        1/10       1/10     FTH1, MALAT1, FTL  (generic reorder luck)
RPL7A        124      6/10        7/10       7/10     RPS9, RPS12, ... (bet RP genes — net −1)
AQR          117      4/10        3/10       3/10     SNHG1, GAS5, MT-CO3, TPT1 (generic reorder luck)

AVERAGE:   reasoned 3.17/10   full-stack 3.33/10   prior 3.17/10
           reasoning adds −0.17 vs full-stack, +0.00 vs prior
```

**The reasoning layer did not help — and its confident biology bets backfired.** Reasoned (3.17) ties the *dumb generic prior*
(3.17) and lands slightly below the mechanical full-stack (3.33). Every place the reasoner deviated toward specific, biologically-
correct mechanism, it lost ground:

- **GATA1 → 0/10.** I bet the canonical erythroid effector program (ALAS2, GYPB, HEMGN, GP1BB). With **295 genes moving**, essentially
  none were the erythroid targets — the movers are generic. The textbook answer scored zero.
- **SUPT6H → 4 vs 5.** SPT6's known specialty is histone-gene control; betting HIST1H1C cost a slot.
- **RPL7A → 6 vs 7.** The ribosomal-stress bet (RP genes) half-worked (RPS9, RPS12 moved) but was net negative.
- Where reasoning "won" (CHMP3 +2, AQR +1) it was **luck reordering generic prior genes** (MALAT1, FTL, TPT1), not insight.

**This is the EIF2S1 lesson, generalized and now measured head-to-head:** the transcriptomic readout is dominated by generic
stress-responsive genes, so betting on the biologically-correct *specific* program strictly hurts. **The far-field wall is a
measurement gap, not a reasoning gap** — a domain-expert reasoner handed everything except the answer cannot beat the prior, because
what's missing is the *measurement* of specific transmission, not its interpretation. Honest caveats: n=6 (the −0.17 is within
noise), single-shot (iterative feedback would leak the test set), and the reasoner reranked a mostly-generic candidate pool — a
tool-augmented reasoner might differ. But the direction is unambiguous and consistent with every prior result in this thread.
(`reason_layer.py` → reason_layer.json.)

---

## The recommended lever, tested: does cross-cell-line conservation add specific signal? (K562 + HCT116)

The plan to actually *improve* the specific prediction: use a second cell line (HCT116, colon) alongside K562 (erythroleukemia).
A gene that moves under knockout X in *both* lineages is a stronger candidate real target than one that moves in one noisy line.
1,694 knockouts are measured in both; movers defined rank-based (top-100 by |z| per line, since the two matrices are on very
different scales).

```
Q1 CONSERVATION   median Jaccard(K562, HCT116 top-100 movers) = 0.025   (~6 of 100 reproduce)
Q2 GENERIC        responsiveness-prior:  conserved 167.7   vs   cell-specific 91.1   (conserved ~2× more generic)
Q3 MECHANISM      mechanism-fraction:    conserved 0.0146  vs   cell-specific 0.0068 (~2.1× — but off a ~1% base)
precision@10      vs K562-only 2.73 (mech −0.07)   |   vs CONSERVED target 0.98 (mech −0.01)
```

**The lever did not pan out — it confirms the wall from a second lineage rather than breaking it.**

- **Q1 — responses are overwhelmingly cell-specific.** Only ~6 of a knockout's top-100 movers reproduce across K562 and HCT116
  (Jaccard 0.025). A knockout's transcriptional response barely transfers between lineages. This also **falsifies the CellOS-2.0
  "cell-agnostic simulator" premise** — you cannot read one line's response off another.
- **Q2 — the conserved core is the usual suspects.** The handful of genes that *do* conserve are ~2× more generically responsive
  (prior 168 vs 91) — conservation preferentially re-selects genes that move under many knockouts in any line.
- **Q3 — conservation does concentrate mechanism ~2×, but off a negligible base.** 1.46% vs 0.68%: even in the conserved core,
  **98.5% is still non-mechanism.** And it doesn't translate — precision@10 against the conserved target is *lower* (0.98, because
  the target is only ~6 genes) and mechanism still adds ~0.

**Conclusion:** a second cell line confirms that knockout responses are lineage-specific and the conserved signal is dominated by
generic movers. The 2× mechanism concentration is real but far too small and non-actionable to build a specific predictor on. The
genuinely useful takeaway is a **negative that reframes the goal**: a "universal" cross-line predictor is the wrong target — the
honest path is a **high-precision, abstaining near-field tool per cell line**, not a broad cross-line one. The far-field wall now
stands from two independent cell types. (`conserved_signal.py` → conserved_signal.json.)

---

## Can the Capacity/Saturation model work? — predict response MAGNITUDE from node capacity (bypassing k_f)

The proposal: don't try to predict *which* far-field genes move (conceded — the far-field is a generic stress program). Instead
compute a **capacity/saturation** state for each node from the DNA→protein lifecycle + FBA, and predict the **magnitude** of a
knockout's response from its **capacity deficit** — a saturated chokepoint causes a big biomass hit → big stress response; a
buffered node is absorbed. This bypasses the missing per-edge $k_f$ entirely: it predicts response *size* from a node property, not
response *identity* from an edge. Tested with `dep_frac` (DepMap dependency = an empirical saturation index that folds in
buffering+redundancy) plus abundance, centrality, and half-life, against response magnitude (n_movers, total stress).

```
Per-feature Spearman with response magnitude (n_movers):
  ess         +0.189      essential → bigger response          ✓ chokepoint
  dep_frac    +0.113      more dependent → bigger response      ✓ empirical saturation
  log_ppm     −0.100      MORE abundant → SMALLER response      ✓ buffering (excess capacity absorbs)
  log_hl      −0.164 ·  is_tf +0.083 ·  ppi_deg +0.081

Held-out (5-fold) Spearman(predicted, actual) response magnitude:
  capacity model (no technical)          +0.224
  n_measured ALONE (technical confound)  −0.038   ← so the signal is NOT technical
```

**It works — for what it actually claims, which is the right claim.** Response magnitude is predictable held-out at **Spearman
0.224** from capacity features alone (~5% of variance — modest, but real and it beats the technical confound cleanly). And the
*directions* all confirm the model:

- **Essential / chokepoint nodes drive bigger responses** (ess +0.19, dep_frac +0.11) — low-tolerance nodes whose deficit isn't buffered.
- **The non-obvious confirmation:** **high abundance → *smaller* response** (log_ppm −0.10). More protein = more excess capacity =
  more buffered = smaller perturbation. That's exactly Step 2 of the model (the Saturation Index), and it's a directional
  prediction we didn't put in by hand.

**What this settles:** the architecture does **not** bypass the which-genes wall — nothing can without the missing measurement, and
the model itself concedes the far-field is generic. But it does two valuable things the wall-chasing never did: (1) it **explains**
why the far-field is generic — a global biomass/budget breach trips the *same* stress program regardless of which gene you hit, which
is precisely why the response is gene-nonspecific and matches everything we measured; and (2) it converts the wall from a failure
into a **correctly-scoped capability** — predict *how disruptive* a perturbation is (response size, variant impact) from node
capacity, even though not *which* genes. **Magnitude yes, identity no.** The metabolic Saturation Index (FBA flux / kcat·abundance)
is the finer-grained version for the ~1,500 metabolic enzymes; `dep_frac` is the empirical proxy that generalizes to all knockouts.
(`capacity_trigger.py` → capacity_trigger.json.)

---

## Predict the tide, not the ripples — the far-field is a few programs, and the "generic prior" IS the answer

The reframe that resolves the whole far-field arc: we spent the thread trying to trace a single causal wire (KO A → B → C → D → gene E)
and hitting 1.06× chance, and measuring success as *per-gene identity* (~1/10). The capacity model showed why that wire is
physically untraceable (buffering dissolves the specific signal) — and pointed at the right question: the far-field is not thousands
of bespoke wires, it's a **small set of transcriptomic programs** (stress modules / panic buttons) tripped when a capacity deficit
breaches a global budget. Tested on 1,851 K562-deep knockouts × 8,563 genes with 15 NMF modules.

```
C1 MODULARITY   15 programs explain 56% of response variance; dominant program covers 55% of knockouts
C2 THE TIDE     recall of a knockout's real movers by the generic top-M:
                  top-50 → 34.5%   top-100 → 48.4%   top-200 → 63.3%
C3 FLAVOUR      dominant module predictable held-out 64.0% vs 54.5% baseline (+9.5 pts)
                module-genes-alone recall 29.6% < tide 48.4%  → additive layer, recovers 25.9% of the specific residual
```

- **C1 — the far-field is modular.** 15 programs explain 56% of the variance; one dominant program covers 55% of knockouts. Cells
  fire shared modules, not bespoke responses — exactly the model.
- **C2 — the "generic prior" is not a baseline to beat; it *is* the far-field.** The top-200 most-responsive genes recover **63% of a
  typical knockout's actual movers.** So **precision@10 measured the wrong unit** all along: at the *program* level the far-field is
  largely predictable (high recall). We were penalizing the model for missing per-gene identity it was never going to get. This is
  the key correction of the whole arc.
- **C3 — the flavour is modestly predictable.** Which module a knockout fires is callable held-out at 64% vs 54.5% (+9.5 pts) from
  capacity/function — so SREBP-vs-heme-vs-UPR is partly separable beyond generic stress. But honestly: the predicted module's genes
  *alone* recall *less* (29.6%) than the broad tide (48.4%), so the module is an **additive flavour layer** (recovering 26% of the
  specific residual the tide misses), not a replacement.

**The honest, correct scope of CellOS's far-field — no longer a wall, but a boundary:**

| Layer | What it predicts | Measured |
|---|---|---|
| **Magnitude** | how big the response is | capacity/saturation, Spearman **0.22** |
| **The tide (recall)** | which wave of genes shifts | generic program, **63%** recall @ top-200 |
| **Flavour** | which panic button fires | module, **+9.5 pts** / +26% of the tail |
| **The ripple** ⛔ | the specific distal gene identity | still walled (needs the measurement) |

We stopped trying to predict the ripples and started predicting the tide — and the tide is **63% recoverable**. The only thing that
stays walled is the individual unannotated distal gene, which was always chaotic. That's not a failure; it's the correct physical
boundary between what's predictable (the program, set by which global budget breaks) and what isn't (the specific molecule).
(`farfield_modules.py` → farfield_modules.json.)

---

## The whole stack + LLM reasoning, re-run at the module level — does reasoning finally help?

The reasoning layer failed at the per-gene level (correct biology backfired). The tide/module reframe said the predictable unit is
the *program*, so we re-ran it there: strict held-out, the reasoner sees each knockout's identity/function/capacity plus the 15 NMF
program fingerprints and commits *which module(s) each knockout fires* before any reveal — scored against the mechanical XGBoost
module-classifier and the plain generic tide, recall at a 100-gene budget. Panel = the 8 biggest responders (GATA1 + essential
ribosome/spliceosome machinery).

```
ko         movers  trueM  mechM  reasonM   recall: tide  mech  reason
GATA1        722    14     13     [10]           0.09   0.06   0.06
RPL7A        466     9      9     [1]            0.12   0.12   0.09
AQR          392     4      4     [5]            0.08   0.11   0.07
SNRPD2       358     4      4     [5,1]          0.11   0.13   0.08
...
AVG RECALL @ 100:  tide 0.095   mechanical 0.094   REASONED 0.075
MODULE-SELECTION accuracy:  reasoned 0/8 (0%)   mechanical 62.5%
```

**Reasoning does not help even at the module level — it slightly hurts — and *why* is the real finding.** My biologically-"obvious"
calls were **wrong on all 8**: I picked heme (M10) for GATA1, ribosome (M01) for RPL7A, ISR (M05) for the spliceosome knockouts — and
every one missed, because **a strong knockout's *dominant* module is not its functional pathway, it's a generic one.** GATA1's
dominant response is an inflammatory/generic program, not the heme module; RPL7A's is generic, not the ribosome module; the
spliceosome knockouts' is generic, not the ISR.

**The mechanical classifier wins (62.5%) precisely because it learns "big essential knockout → generic panic button" — which is
exactly what the capacity model predicts.** The largest deficits breach the global budget and trip the *same* generic stress program
regardless of which gene was hit. So reasoning about specific biology *fights the capacity logic* — and loses at every level, gene
**and** module. This is the EIF2S1 lesson, now proven one level up.

**Honest caveat:** this panel is the top-8 by response size — the strongest, most generic-dominated knockouts. A weaker, more
specific perturbation (a single lipid enzyme nudging SREBP) might let a reasoned specific module help; that's the one place left to
look. **Bottom line for the whole stack:** the recoverable far-field is the **tide** (the generic dominant program, recovered by the
generic prior) plus a **capacity-driven mechanical module classifier**; an LLM biology reasoner adds nothing there. Reasoning earns
its keep in the **near-field** (mechanism, mutations, ΔΔG); the far-field belongs to **capacity + the tide.** (`reason_modules.py`
→ reason_modules.json.)

---

## The full stack with reasoning OUT — the honest measured ceiling of CellOS's far-field

With the LLM reasoner removed (it helped only the near-field), here is the purely mechanical far-field predictor, scored over all
824 knockouts at the resolution each layer actually works:

```
resolution     what it answers              full stack (reasoning out)
MAGNITUDE      how big the response is       Spearman 0.224           (capacity / saturation)
TIDE (recall)  which wave of genes moves     48% @ top-100,  63% @ top-200   (generic program)
FLAVOUR        which panic-button program    64% module accuracy
  + on tide    tide + mechanical module      48% recall  (beats tide-only in only 24% of KOs → ~a wash)
IDENTITY       the one specific distal gene  2.04/10 precision@10      (WALLED)
```

**This is the whole story, honestly, in one place.** The far-field is not one number — it's predictable at three resolutions and
unpredictable at a fourth:

- **Magnitude** — *how disruptive* a knockout is — Spearman **0.224** from node capacity. Real, useful for variant-effect.
- **The tide** — *which wave of genes shifts* — the generic program recovers **48%** of a knockout's real movers at a 100-gene
  budget, **63%** at 200. **This is the far-field, and it is predictable** — precision@10's "~2/10" was measuring the wrong unit
  (per-gene identity) all along.
- **Flavour** — *which panic button* — the mechanical capacity/function classifier calls the dominant module **64%** of the time.
  But at a 100-gene budget, layering it on the tide is **a wash** (48% vs 48%, wins in only 24% of KOs) — the dominant program the
  tide already carries is usually the answer.
- **Identity** — *the one specific distal gene* — stays **walled** at ~2.04/10, and no measurement we have changes that.

**Bottom line:** reasoning removed, CellOS predicts the far-field as **magnitude (capacity) + tide (generic program, ~half to
two-thirds of the movers) + a modest flavour call.** Only the chaotic per-gene ripple is unpredictable. That's the honest, measured
ceiling — and it's a genuine predictor at the right unit, not the 1-in-10 the wrong metric implied. (`full_stack.py` →
full_stack.json.)

---

## Trying the Google Maps algorithm — routing (Dijkstra) instead of flooding (diffusion)

Diffusion (Random Walk / PageRank) already hit the wall — it floods the graph and re-finds the generic hubs. So we tried the *other*
Google algorithm: **Dijkstra shortest weighted path** — find the single best *route* from the knockout to each gene, not flood.
Given its best shot: edges weighted by verified trustworthiness (curated regulon / SIGNOR = fast roads, noisy bulk reg = slow roads),
cost = 1/log(trust). 25k (knockout, gene) pairs, held-out by knockout.

```
Does a gene move, by shortest-path distance from the knockout? (base rate 7.7%)
  dist~1   10.1%   ← direct regulon, modestly enriched
  dist~2    6.4%   ← already back to chance
  dist~3+   ~8%    ← flat around base

Held-out AUPRC lift (chance = 1.0x):
  shortest-path closeness (Google Maps)          1.19×
  generic closeness (from a RANDOM knockout)     1.16×   ← the control
  responsiveness prior                           8.69×
  shortest-path + prior                          8.77×   (+0.08)
```

**Google Maps recovers the near-field, then hits the exact same wall.** Distance-1 neighbours (the direct regulon) move a bit more
than chance, but by distance 2 you're already back to the base rate — there's no "closer = more likely to move." And the decisive
control settles it: shortest-path closeness (**1.19×**) is essentially identical to closeness *from a random knockout* (**1.16×**) —
so the "signal" is just **centrality wearing a costume**. Being close to *this* knockout is no better than being close to *any*
knockout. It adds nothing over the generic tide (8.69× → 8.77×).

**Why the algorithm that conquered city navigation fails here:** Google Maps works because every road has a *measured* travel time
and a trip is a single *physical* route. In the cell, the edge "travel times" are the transmission coefficients that were never
measured — so "shortest path" collapses to plain topological distance, which is just centrality. And the far-field isn't a *route*
anyway: it's a **flood** — the generic stress tide triggered by a global budget breach — which no path-finder addresses. **The
far-field is weather, not traffic.** The right tools remain capacity (how big the flood) and the tide (which flood), not routing.
(`gmaps.py` → gmaps.json.)

---

## Trying the weather-forecast algorithm — accept the chaos, deliver a calibrated forecast

Google Maps failed because the far-field is weather, not traffic. So we ran the *weather* playbook. Its genius isn't one algorithm —
it's a stance: you can't predict the microstate (which molecule, which gene — chaos), so you forecast a **calibrated probability**
per outcome, attach **confidence**, and accept a **predictability horizon**. We tested whether CellOS can deliver that *form*.

```
1 CALIBRATION (ECE = 0.006 — near perfect)
   forecast says   →  observed move-rate
      0.94                 0.94
      0.75                 0.78
      0.45                 0.48
      0.02                 0.02

2 CONFIDENCE — commit only the surest genes:
   top-2%  → 93% move   (12× base)
   top-5%  → 81%        (10.6×)
   top-10% → 56%        (7.3×)

3 HORIZON — near-field lift 4.6× · far-field 9.1×  (a base-rate effect, see below)
```

**This is the first idea that landed cleanly — because it stopped trying to break the wall and found the right *shape* for the
answer.**

- **Calibration (ECE 0.006).** When the forecast says a gene has a 94% chance of moving, 94% of them move; 2% means 2%. **The
  probabilities are honest** — the single most important property of a forecast, and the one thing every point-accuracy metric we ran
  was blind to.
- **Confidence → an abstaining forecast that works.** Commit only the genes we're surest of and we're right **93% of the time**
  (top-2%). The "1 in 10" that haunted this whole thread was the *wrong framing* — it forced a blanket top-10 for *every* knockout,
  including where there's no signal. The honest forecast says "here are the few I'm sure of," and it's right 9 times in 10.
- **Horizon (honest caveat).** The lift numbers look backwards (far 9.1× > near 4.6×), but that's a **base-rate artifact**, not the
  far field being more predictable: near-field genes (regulon/PPI neighbours) simply *move a lot* (high base, low lift-room), while
  the far-field forecast is pure **climatology** — "which genes usually move," the tide — which scores high lift against a sparse
  base, exactly like "it usually rains in Seattle" beats a dry baseline. Both are honestly forecast; neither names the specific gene.

**The bottom line — and the right final form for the whole project:** weather forecasting doesn't break the wall, and doesn't try
to. It gives the correct *output form*: an honest **knockout weather report** — a calibrated move-probability for every gene, a
high-confidence subset it will commit to, a magnitude (how big the storm), a tide (which program), and an explicit "beyond here it's
climatology" for the chaotic far field. **Not a GPS route — a weather report.** (`weather.py` → weather.json.)

---

## Is the forecast near or far field, and do we still need magnitude? — the capstone decomposition

Two questions about the weather forecast, answered with a decisive cross-knockout control (the forecast's confident 93% could be
real far-field skill, or just the near-field + climatology re-confirmed — this settles which).

```
Q1  Where does the forecast's confident-correct accuracy come from?
      near-field (direct neighbour)        1%
      climatology (usual-suspect gene)    63%
      "specific far-field" (neither)      36%      ← looks like far-field skill...

    forecast lift by stratum:
      near-field pairs                    5.08×    ← the one real KO-specific signal
      climatology pairs                   1.11×
      "specific-far" pairs                8.52×    ← ...but this is climatology in disguise

    CROSS-KNOCKOUT CONTROL (decisive):
      similarity of predictions between two RANDOM knockouts = 0.779
      → the forecast issues nearly the SAME "which genes move" list for ANY knockout

Q2  Does the forecast already contain the magnitude?
      summed-forecast vs actual response size   Spearman −0.063   (noise)
      standalone capacity model                 Spearman ~0.11–0.22
```

**Q1 — it's near-field + climatology. No genuine specific far-field skill.** The tempting misread was the 8.52× lift on
"specific-far" genes — but the cross-knockout control kills it: the forecast predicts almost the *same gene ranking regardless of
which gene you knock out* (0.779 similarity between random knockout pairs). It has exactly one knockout-specific feature — the
near-field neighbour flag (a real 5.08× there) — and for every far gene it ranks by *generic responsiveness*, which is a property of
the gene, not the knockout. So the 8.52× is the responsiveness gradient sorted finer: **finer-grained climatology, not
transmission.** The specific far field stays walled.

**Q2 — no, the forecast does not give the magnitude.** Summing the calibrated probabilities predicts response size at **−0.06**
(noise) — worse than the weak capacity model (~0.11–0.22), because the sum is dominated by universe size and the constant
climatology, not by how big the response is. So the magnitude stays a separate, still-weak job for the capacity model. (This is the
honest answer to the earlier "we never improved magnitude" — we don't get it for free from the forecast either.)

**Bottom line:** the weather forecast's value is exactly two honest things — **calibration** (when it says 70%, it's 70%) and a
**confident near-field + tide subset** — and it issues essentially the same list for every knockout. It does **not** secretly crack
the specific far field, and it does **not** secretly contain the magnitude. Knowing precisely what it is (and isn't) is the point.
(`forecast_decompose.py` → forecast_decompose.json.)

---

## Solving the magnitude weak spot — how big will the response be?

Magnitude ("how many genes move") was the honest weak spot — the capacity model was Spearman ~0.22 and tail-blind. Three levers:
a cleaner target (total displacement vs thresholded count), richer features (centrality/connectivity/reach, not just essentiality),
and log-space regression. Built and measured on deep k562 (1,851 knockouts, held-out 5-fold).

```
                                        Spearman(predicted, actual)
old features   → n_movers                     0.521   ← matched baseline ON THIS data
old features   → total displacement           0.487
RICH features  → n_movers                      0.558   ← best (+0.037 from features)
RICH features  → total displacement           0.498
top drivers: dep_frac, centrality, ppi_deg, reach, regulon_size
```

**It improved — modestly and honestly — and the raw numbers are easy to over-read, so here's the precise version:**

- **The matched, apples-to-apples gain from my features is +0.037** (0.521 → 0.558). Real, held-out, driven by the connectivity
  features I'd never added — centrality, 2-hop reach, regulon size, PPI/coexpr/co-dependency degree — i.e. *how central / how much
  the gene reaches*. That's the genuine improvement.
- **The "0.22 → 0.56" is mostly a dataset artifact, not my model.** The old 0.22 came from gwps, *pre-filtered to knockouts that
  already respond* — the hard "how much among responders" question. On the full spectrum here (including the many non-responders),
  even the *old* features reach 0.52, because essentiality cleanly separates "big response" from "barely responds." Honesty demands
  I not claim a 2.5× leap I didn't make.
- **Two of my three fixes failed, and I'm reporting them:** total displacement did *worse* than the plain count (0.50 vs 0.56), so
  that hypothesis was wrong; and coverage (`n_measured`) is constant on this dense matrix, so the technical-coverage worry never
  applied.

**Net:** magnitude is moderately predictable (~0.56) on the full knockout spectrum — it tracks how essential and connected the
knocked-out gene is — and my connectivity features add a real, modest **+0.037** on top. This is probably near the ceiling for
predicting response size from static gene properties: the heavy tail (a GATA1-scale surprise from a *non-essential* master regulator)
stays unpredictable, because essentiality literally can't see it. The weak spot is genuinely improved, honestly bounded — not solved
to precision, but no longer the 0.22 embarrassment either. (`magnitude.py` → magnitude.json.)

---

## Coverage — and a correction: the "9/10" was an evaluation artifact

The practical question: a researcher brings a knockout; for how many can the model deliver a useful high-confidence result? Measuring
it exposed — and forced me to retract — the "9 out of 10" I'd been citing.

```
A  MUTATION effect (does a variant break/destabilise the protein)   ~proteome-wide (~16.5k genes), r≈0.63
B  SPECIFIC direct targets of a knockout                            only 24/237 (10%) — curated TFs (+ signalling)
C  high-confidence forecast, P≥0.5, on the FULL universe            2.9% precision, 0% of KOs reach a 9/10 bar
D  REAL DEPLOYMENT (score every gene, top-K per KO):
     top-1  15%     top-5  11%     top-10  9%
```

**The "93% / 9-out-of-10" does not survive real deployment — it was a subsampling artifact.** That number (`weather.py`, top-2%) was
measured on a **negatively-subsampled evaluation set**: to balance training, ~12 of every 13 non-moving genes were thrown out, so
movers were ~13× over-represented (base 7.7% vs a true ~1%). On that mover-rich set the top predictions look 93% precise. But when
you **deploy** — point it at a knockout and score *every* gene, as a researcher actually would — the probabilities are calibrated to
the wrong base rate and collapse: **top guess right 15%, top-10 ~9%, and no knockout reaches a genuine 9/10.** The honest deployment
number was `predict10`'s ~1–2 in 10 all along; I let the shinier 93% overshadow it and should have flagged that it came from an
easier evaluation. Correcting it here, front and center.

**So the honest coverage for a researcher:**
- **"Will this mutation break/destabilise the protein?"** → reliable, nearly every gene (r≈0.63). The model's genuine strength.
- **"Which genes does this knockout *directly* control?"** → only ~**10%** of knockouts (the annotated TFs) get a specific list.
- **"Give me a confident downstream 9/10 list on demand"** → **no.** Every knockout gets a calibrated forecast + a magnitude
  estimate, but its top picks are right ~1–2 in 10 on real deployment, and the confident ones are generic climatology genes, not
  protein-specific. There is no on-demand 9/10.

The model is a strong, high-coverage *mutation/structure* engine and an honest-but-generic *downstream forecaster* — not a
per-request specific-knockout oracle. (`coverage_delivery.py` → coverage_delivery.json.)

---

## Full-stack run on 60 held-out knockouts — the honest deployment picture (and a correction to my correction)

Ran the whole current stack on 60 held-out knockouts at the real deployment protocol (train on the other ~1,790, score *every*
gene — no subsampling). It refined *both* earlier overstatements.

```
per-knockout (strongest shown):
  knockout  actual  pred_mag  top10 hits
  EIF2S3     201      22          9
  RPL27      129      25         10
  RPL27A      84      32          9
  RBM25       82       4         10
  CTR9        80      30          9
  ...
  KCTD10       5       0          1
  CHMP1A       7       1          1

AGGREGATE (60 KOs):
  MAGNITUDE (how big):     Spearman = 0.459
  IDENTITY (which genes):  average 2.02/10 — but it SCALES with strength:
       STRONG KOs (≥50 movers, n=7):  9.0/10
       WEAK   KOs (<15,        n=48):  0.81/10
  NEAR-FIELD: 4 panel KOs are TFs; regulon top-10 = 0/10 (too small to conclude)
```

**The key finding corrects my own over-correction from the prior turn.** I'd said the "9/10" was a subsampling artifact that
doesn't survive deployment. That was too harsh. The truth: **top-10 deployment precision scales with how strong the knockout is.**
Strong knockouts (that move ≥50 genes) genuinely get **9/10** in their top-10 on real full-universe deployment; weak knockouts get
**~1/10**; the average is ~2/10. What the subsampling did was inflate the *average* to look uniformly 9/10. So both extremes I'd
stated were wrong — it's neither a flat 9/10 nor a flat 1/10; it's **9/10 for strong knockouts, 1/10 for weak.**

The essential caveat holds in every case: even the 9/10 for strong knockouts is on the **generic usual-suspect genes** — a strong
knockout disrupts so much of the transcriptome that predicting "the genes that usually move" simply lands. It's not protein-specific
biology; it's the model correctly betting that a badly-stressed cell runs its generic program.

**So what you actually get, at scale, honestly:** the response *size* (Spearman ~0.46 — usable order-of-magnitude), and the *which
genes* at a precision that runs from ~9/10 for a strong knockout down to ~1/10 for a weak one — always the generic stress genes,
never the protein-unique fingerprint. That is the complete, corrected deployment picture. (`fullstack_run.py` → fullstack_run.json.)

---

## Strong vs weak knockouts — what actually makes the difference (`strong_vs_weak.py`)

Follow-up to the full-stack run: if the top-10 precision is ~9/10 for strong knockouts and ~1/10 for weak ones, **what
separates them?** Split all deep-k562 knockouts into **strong (≥50 movers, n=189)** vs **weak (<15 movers, n=1412)** and compare
their gene properties head-to-head (medians, Mann-Whitney), plus a function/process enrichment.

```
property             STRONG       WEAK  strong/weak
dep_frac               0.99       0.77         1.3x *
essential               1.0        1.0         1.0x *   (binary flag ~1 for both)
ppi_degree             65.0       13.0         5.0x *
coexpr_degree          12.0       12.0         1.0x     (FLAT — not a separator)
codep_degree            7.0        7.0         1.0x     (FLAT — not a separator)
centrality           8244.0     2461.0         3.3x *
abundance_ppm          11.8       5.44         2.2x *
n_pathways              7.0        2.0         3.5x *

FUNCTION (fraction of KOs by process):
transcription   strong 42%  weak 20%   (2.1x)
translation     strong 31%  weak 13%   (2.4x)
other           strong 18%  weak 31%   (0.56x)

example STRONG: GATA1, POT1, RPL7A, SUPT6H, RBM22, AQR, SNRPD2, POLR2K, MED9, INTS2, MED30
example WEAK:   ZNF100, ZNF133, ZNF207, ZNF24, ZNF253, ZNF284, ZNF407, ZNF468, ZMAT2, ...
```

Strong knockouts separate from weak on **three measured axes** — and, correcting a guess I'd made before, **not** on a fourth:

1. **Essentiality, by degree.** dep_frac 0.99 vs 0.77. The *binary* essential flag is ~1 for **both** groups (nearly every
   profiled knockout is essential-ish — that's why it has movers at all); the separation is in the *degree* of dependence, not
   essential-yes/no.
2. **Physical connectivity.** PPI degree 65 vs 13 (5×, p≈3e-30), network centrality 3.3×, pathway membership 7 vs 2 (3.5×) —
   strong knockouts sit at physical/pathway hubs. **Correction:** I'd expected coexpression/codependency degree to separate them
   too; they're **flat** (12 vs 12, p=0.19; 7 vs 7). So it's specifically *physical-interaction + pathway* centrality that
   separates strong from weak, not transcriptional co-regulation breadth.
3. **Function.** Strong knockouts are heavily enriched for the **core gene-expression machinery** — transcription 42% vs 20%,
   translation 31% vs 13%, and "other/peripheral" is depleted (18% vs 31%). Concretely: strong = GATA1, RPL7A (ribosome),
   SUPT6H/POLR2K/MED9/MED30/INTS2 (Pol II / Mediator / Integrator), RBM22/AQR/SNRPD2 (spliceosome). Weak = **zinc-finger paralog
   families** (ZNF100/133/207/253/468, ZMAT2…) — redundant, specialised, low-connectivity. Strong knockouts are also themselves
   more highly **expressed** (11.8 vs 5.4 ppm), consistent with being abundant core machinery.

**Mechanism — the capacity model, made concrete.** Knock out an essential, physically-central, core-machinery gene and you punch
a large **capacity deficit** into a fundamental process (transcription, translation, splicing); the cell mounts a large *generic*
stress response — so many genes move that predicting the usual-suspect movers lands 9/10. Knock out a peripheral/redundant gene
(a zinc finger with paralogs to cover for it) and the loss is **buffered** — few genes move, so there's little to predict (~1/10).

So "strong vs weak" is essentially "**essential + PPI/pathway-central + core-machinery**" vs "**dispensable + peripheral +
redundant**." That is exactly why the response **size** is predictable (Spearman ~0.5 from these same features) while the specific
mover **identity** is not — the size is set by the node's importance in the network; the identity, by chaos. (`strong_vs_weak.py`
→ strong_vs_weak.json.)

---

## Can we crack the weak knockouts? Testing "Compensatory Shunting" (`compensation.py`)

The proposal: to raise the global top-10 average from ~2/10 to 6–7/10 you can't lean on the generic tide — you have to predict the
1–5 specific genes that move when a redundant/peripheral node is buffered. Mechanism 1 of the proposed *Structural Compensation
Engine*: when you knock down a redundant gene, the cell shifts load to its **closest structural paralog**, which is **upregulated**
and is one of those specific movers. This is directly falsifiable, so I tested it rather than debating it.

**Method honesty (a detour worth recording).** I first used the on-hand ESM-2 matrix (`esm_normal.parquet`) as the proposed
"continuous structural field" and took cosine-nearest-neighbors. A **positive control killed that route**: the matrix is
per-dimension *standardized* (per-dim mean ≈ 0, std ≈ 1), which destroys cosine geometry — known paralogs scored at chance
(RPL7/RPL7A cos **0.043**, ACTB/ACTG1 **0.024**, vs a random 99th-percentile of 0.063). Any negative from it would be an artifact,
so I discarded it and switched to a **curated Ensembl paralog map** (10,627 genes; correct where present: GATA1 → GATA2/3/4/5).

**Result — a clean negative.** Of the 123 deep-k562 knockouts that have ≥1 *measured* paralog:

```
                     paralog is a mover   paralog upregulation
                     vs base  -> lift     above KO's own drift
ALL (n=123)          0.000 / 0.004  0.0x        -0.001
WEAK (n=79)          0.000 / 0.001  0.0x        +0.017
STRONG (n=21)        0.000 / 0.017  0.0x        +0.001
```

**Literally zero** paralogs crossed the mover threshold, and paralog upregulation above the knockout's own drift is ≈0. Even with
real curated paralogs, knocking a gene down does **not** light up its paralog in this data. Three reasons, and they matter:

1. **Assay modality.** K562 Perturb-seq is CRISPRi knock*down* (dCas9-KRAB transcriptional repression). The classic
   paralog/genetic-compensation pathway (transcriptional adaptation) is triggered by *mutant-mRNA degradation*, which CRISPRi does
   not produce — so the mechanism the engine models is largely **absent from the measurement**.
2. **Steady-state timing.** It's a single snapshot ~days post-perturbation. A transient paralog that fired at hour 4 and returned
   to baseline **cannot be in the data** — exactly the "photograph of the calm lake" point. This is the one mechanism from the
   proposal I fully agree diagnoses the ceiling — but it's a **data** limit, not an architecture we can out-engineer here.
3. **Coverage ceiling.** Only ~123 of ~1,123 knockouts (**~11%**) even *have* a measured paralog. So paralog compensation — even
   if it worked perfectly — could touch ~11% of knockouts and **structurally cannot** lift the global average to 6–7/10. The weak
   set is dominated by genes without a clean paralog (and it's *not* mostly zinc fingers — those were 1% of the weak set, just the
   alphabetical tail of an earlier example list; the weak set is a broad mix of "other"/transcription/translation/transport genes).

**And the metric itself is capped.** Precision@10 ≤ n_movers/10, so a weak knockout like KCTD10 with 5 true movers can *never*
exceed 5/10 on a fixed top-10 list, regardless of the model — the "raise the average" framing is partly a metric-design problem,
not only a modeling one.

**Bottom line.** The Structural Compensation Engine is biologically reasonable, but it has **no signal to learn or be validated
against in this data**, and too little reach to move the average even in principle. The weak-knockout ceiling is a
**data/measurement limit** (knockdown modality + steady-state timing) plus a **metric cap** — not something a bigger model on this
dataset can fix. The honest path to it is a better *metric* (recall of the true movers, k=n_movers) and new *data* (time-resolved
and/or true-knockout Perturb-seq), not a heavier architecture. (`compensation.py` → compensation.json.)

---

## Do the other cell lines help? Cross-dataset reproducibility (`reproduce.py`)

We downloaded perturb-seq from more than one cell line — genome-wide K562 (`gwps`) and HCT116 colon. The highest-value use isn't
"more training rows"; it's the one test K562-alone can't run: **when the same gene is knocked out in two independent experiments,
do the same genes move?** That decides whether the far-field wall is real signal our features miss, or irreproducible noise.

*Metric honesty:* the |z|>1 "mover %" and its fold-over-chance are threshold/scale-sensitive across datasets (HCT116's z is
compressed — almost nothing clears |z|>1, so its chance rate ≈ 0 and the fold blows up). The fair, rank-based metrics are the
**z-profile Spearman** and the **top-20 mover overlap**, so those lead.

```
                              n     profile ρ    top-20 movers recur (/20)
K562 deep vs gwps  ALL     1208       0.323              8.0
   (same cell line) STRONG  197       0.53               9.4
                    WEAK    734       0.251              7.2
K562 vs HCT116     ALL     1063       0.132              2.1
   (cross line)     STRONG  125       0.281              2.0
                    WEAK    714       0.075              2.1
```

This is a genuine two-sided refinement — and it **corrects the pure-noise framing** of the wall:

1. **Within a cell line, the specific movers are reproducible — including the far field, including weak knockouts.** A weak
   knockout's response replicates at Spearman 0.25 with **7 of its 20 biggest movers recurring** in an independent experiment,
   far above chance. So the far-field is *not* pure noise, and the 1.06× graph-prediction "wall" is largely a **feature/model gap**
   — our PPI/regulon graph doesn't encode the paths that generate the reproducible response — not a noise floor. That's more
   hopeful than "microstate chaos." The sobering rider: the obvious features to close that gap already failed (paralogs →
   `compensation.py`; graph propagation → earlier).
2. **Much of the reproducibility is the generic tide.** The mover *sets* overlap a lot because the same usual-suspect stress genes
   recur, yet the graded Spearman is only ~0.32 — identities and magnitudes are only weakly pinned even where the set overlaps.
   Same climatology result, from a new angle.
3. **The response is strongly cell-type-specific.** K562 → HCT116 roughly **halves** reproducibility (ρ 0.32 → 0.13; top-20 8 → 2),
   and for weak knockouts cross-line it hits the floor (ρ **0.075**). Those specific movers are largely context-specific.

**So, answering "why not use the other datasets" — we should, for what they actually add:**

- They **prove** the response (even the weak-KO far-field) is reproducible real biology within a cell line, relocating the wall
  from *"noise"* to *"missing features"* — a harder but more honest and more attackable target.
- They **quantify** cell-type specificity (~half the signal is K562-specific) — which is exactly the conditioning signal a
  cross-cell-line model needs, and a caution that a K562-trained predictor transfers only weakly (ρ 0.13) to another lineage.
- They make the **near-field + generic tide** more trustworthy to fold in.

What they do **not** do is hand us a reproducible weak-KO far-field that transfers *across* cell lines — that's near the floor.
Multi-dataset data reframes the wall and enables cell-type conditioning; it doesn't by itself breach the far field.
(`reproduce.py` → reproduce.json.)

---

## The combinatorial testbed: Norman 2019 genetic interactions (`norman_gi.py`)

The one dataset that can actually *show* compensation is combinatorial: Norman 2019 (K562, 106 single + 131 double CRISPRa,
all doubles have both singles + 11,855 controls). If two genes buffer each other, the double should deviate from the sum of the
singles and reveal movers that neither single shows. Pseudobulked by streaming the 2.9 GB sparse matrix.

- **Predictability (Q1):** the simple additive model (double ≈ single_A + single_B) predicts the real double at **Spearman 0.40**
  and recovers **37%** of the double's top movers. So combinatorial responses are *largely, not fully* predictable from singles —
  a real capability number for combo screens we didn't have before.
- **Compensation/emergence (Q2):** only **28%** of a double's movers are *emergent* (move in the double but neither single), and
  **15%** of doubles are strongly non-additive (additive ρ<0.3). The strongest genetic-interaction pairs are exactly the
  functionally-redundant partners — **KIF18B_KIF2C** (two kinesins), PLK4_STIL (centriole), BCL2L11_BAK1 (apoptosis) — so the
  biology validates.

**Verdict:** buffering / genetic interaction is **real and measurable in combinatorial data** — unlike the steady-state
single-KO knockdown data, where paralog compensation was at chance (`compensation.py`). But it's the **minority** of the signal,
concentrated in specific redundant pairs, not a large hidden reservoir that would lift weak *single*-KO prediction to 6–7/10.
It's a third independent line pointing the same way: compensation shows up only when you actually remove both copies. Practically:
an additive-from-singles model is a decent combo predictor (ρ 0.40), and the non-additive 15% is exactly where a dedicated
interaction model — trained on Norman GIs — earns its keep. (`norman_gi.py` → norman_gi.json.)

---

## A third cell line, downloaded and tested: K562 vs RPE1 (`reproduce_rpe1.py`)

The CELLxGENE Census route was checked first and **does not carry guide-labeled perturbation screens** — I searched all 2,203
catalog datasets and every "perturbation" match was a cell *atlas*, not a screen. The correct source is **scPerturb (Zenodo)**,
which works via direct HTTPS through the proxy. Downloaded **Replogle-Weissman 2022 RPE1** (non-cancer retinal epithelium, 2,394
CRISPRi knockouts × 247,914 cells; streamed the gzip-dense matrix in cell-blocks and pseudobulked). RPE1 is the *same lab, same
CRISPRi protocol* as our deep K562 — so K562-vs-RPE1 is the cleanest cross-cell-line test possible (protocol held fixed).

```
K562 vs RPE1 (same lab/protocol)   1126 shared KOs   profile ρ   top-20/20
   ALL                                                  0.132        1.6
   STRONG                                               0.24         1.3
   WEAK                                                 0.089        1.8
   (compare: K562 vs HCT116 cross-protocol = 0.132;  K562 vs gwps same-line = 0.32)
```

**The key control result:** K562-vs-RPE1 reproducibility is **0.132 — identical to the cross-protocol K562-vs-HCT116 (0.132)**.
Removing the protocol confound did **not** raise cross-line reproducibility at all. That proves the ~half loss of signal across
cell lines is **real cell-type biology, not a batch/protocol artifact.** Strong knockouts (conserved essential machinery) transfer
best (ρ 0.24); weak knockouts sit near the floor (ρ 0.089) — their specific movers are genuinely cell-type-specific.

So the third cell line confirms and sharpens the reproducibility verdict: within-line ρ≈0.32, cross-line ρ≈0.13 *regardless of how
carefully protocols are matched*. Multi-cell-line data is essential **conditioning** signal — a K562-trained far-field predictor
transfers only weakly to another lineage — but there is no cell-type-invariant weak-KO far-field to recover.
(`reproduce_rpe1.py` → reproduce_rpe1.json.)

---

## Sci-Plex Saturation Index: dose titration (`sciplex_saturation.py`)

Downloaded Sci-Plex 3 (Srivatsan-Trapnell 2020) from scPerturb — 188 drugs × 4 doses (10/100/1000/10000 nM) in K562/24h, the
graded "capacity choke" the user wanted for the Saturation Index. Streamed the 800k-cell matrix, restricted to the ~8.4k genes
shared with our K562 set. **Honest split result:**

- **Q1 confirmed — magnitude scales with dose.** Median Spearman(log dose, n_movers) = **0.40**; **71%** of drugs show the response
  growing with dose. So response *size* is a graded, dose-tunable quantity — exactly the capacity/saturation prediction, and it
  extends the K562-knockout magnitude finding (~0.5) to graded chemical perturbation.
- **Q2 not supported — no dose-invariant program.** I expected the same program to simply scale up with dose. It doesn't: low-vs-high
  dose profile Spearman **0.007**, and even between the two *highest* (both-responding) doses only **0.022**. Either the program
  genuinely reorganizes with dose, or (more likely) per-drug pseudobulk profiles are too noisy at these doses for a stable-program
  rank-correlation to survive — only a few drugs (HDAC inhibitors Panobinostat/Dacinostat/Quisinostat/Givinostat; CDK: Flavopiridol)
  drive large responses. **I will not claim program stability.**

Net: Sci-Plex confirms **magnitude is dose-tunable** (the predictable axis) but does **not** here demonstrate a fixed
dose-invariant identity program. "Dose predicts how big"; "which genes" stays the hard part — consistent with the whole
investigation. (`sciplex_saturation.py` → sciplex_saturation.json.)

---

## The "missing wire" test: is the tide routed by open chromatin? (`missing_wire.py`)

Hypothesis (user): since PPI/regulon fail to predict the reproducible tide, maybe it's routed by the *physical genome* — genes
become "usual suspects" because their chromatin is already open. Tested directly with **real K562 ATAC-seq** (ENCODE IDR peaks,
promoter signal), predicting each gene's tide score (mover frequency across ~2,000 KOs), with the crucial **expression control**
(accessible ≈ expressed) and a genomic-neighborhood TAD proxy.

```
held-out Spearman, predicting tide score (7,211 genes)
   baseline EXPRESSION ....... 0.150
   PPI / regulon control ..... 0.108
   ACCESSIBILITY (ATAC+enh) .. 0.068     <- weakest group
   neighborhood (TAD proxy) .. 0.050
   expression + PPI .......... 0.203
   + ATAC .................... 0.221     (delta +0.018)
   FULL ...................... 0.219
   corr(ATAC promoter, tide) = -0.095    top importance: PPI degree 0.31
```

**Honest negative on the promoter-ATAC version.** Accessibility is the *weakest* predictor, doesn't beat PPI or expression, and
adds only +0.018. Most telling: ATAC promoter signal **anti-correlates** with the tide (−0.095) — the widest-open promoters are
housekeeping genes that are *stable and don't move*. The tide movers are inducible/poised stress genes, not the constitutively
open ones. So "open doors get bound" is the wrong picture for the tide.

**But two caveats keep the broader idea alive, honestly:** (1) *no* static per-gene feature explains the tide well — the full model
is only 0.22, so the "usual suspects" are only weakly a fixed per-gene property; (2) I tested promoter ATAC + a crude neighborhood
proxy — I did **not** test real Hi-C 3D TAD co-localization or the metabolic-messenger wire (Acetyl-CoA/ATP → HAT → global
chromatin). Those are the genuinely untested parts and the honest next fetch. Net: the specific "open chromatin routes the tide"
claim is falsified for promoter accessibility; the tide stays weakly-and-best explained by network connectivity + expression; the
3D-Hi-C and metabolite versions remain open. (`missing_wire.py` → missing_wire.json.)

---

## The combination that works: analogy / transfer (`analogy.py`)

The reproducibility result reframed the far-field wall: the specific movers are **reproducible signal our graph can't read**
(ρ 0.25 within-line), not noise. So instead of routing through the PPI graph, predict a held-out knockout's movers by **analogy** —
borrow from its most **functionally-similar other knockouts** (k-NN / collaborative filtering). Similarity is built *only* from
response-independent annotation (shared protein complexes, pathway, process, coexpression + codependency partners, PPI
neighborhood) — never the held-out KO's own response. Deployment on 1,160 K562 knockouts, K=15 neighbors.

```
                              TOP-10 precision    specific-mover recall@50
                              (full universe)     (movers OUTSIDE tide top-100)
   ANALOGY (functional nbrs)      0.307                 0.288
   RANDOM neighbours (control)    0.103                 0.044
   PRIOR (generic tide)           0.168                 0.000  (can't, by construction)
   cross-KO prediction overlap:   0.232   (old forecast was 0.779 → genuinely KO-specific)
```

**This is the first real partial breach of the far-field wall — and it survives the control.** The decisive test is recovering
the *specific* movers that lie outside the global tide, which the climatology prior cannot get. Functional-neighbor transfer
recovers **28.8%** of them, vs **4.4%** for random neighbors (a **6.5× gain from functional similarity**, not from averaging strong
profiles) and **0%** for the prior. Its predictions are genuinely KO-specific (cross-KO overlap 0.23 vs the forecast's 0.78).

The biology is clean: knockouts of **complex/pathway partners share each other's specific movers**, so borrowing across the
functional neighborhood reads a slice of the reproducible far-field the protein graph missed. The right lens was analogy, not the
graph — the payoff of the "signal is real, graph is the wrong lens" reframe.

**Honest caveats:** absolute specific-recall is ~29% (the wall is bent, not broken); it's a functional-*neighborhood* signal, not
per-gene mechanism; and the similarity includes coexpression/codependency (baseline data, not KO-response, so not circular, but
response-adjacent — an annotation-only complex+pathway+PPI variant would be even cleaner). Still, it's a genuine, buildable gain,
worth wiring in as a forecast component. (`analogy.py` → analogy.json.)

---

## Working the last three data sources (user-requested, one by one)

### 3D chromatin — real Hi-C TADs (`missing_wire_3d.py`)
Follow-up to `missing_wire.py`, replacing the crude ±1Mb linear proxy with **real K562 in-situ Hi-C TADs** (ENCODE ENCFF271SAF,
5,703 Arrowhead contact domains). Does sharing a physical 3D TAD neighborhood with other high-tide genes make a gene a "usual
suspect"?

```
held-out Spearman predicting tide score (4,337 genes, 1,923 TADs, median 1 gene/TAD)
   real TAD 3D-neighborhood ... 0.016   (corr 0.038 ≈ 0)
   linear ±1Mb proxy .......... 0.039
   baseline expression ........ 0.194
   PPI/regulon control ........ 0.091
   + TAD3D over expr+PPI ...... +0.017 ;  over linear proxy +0.018
```
**Weak/negative.** Real 3D TAD co-localization is barely better than 1D proximity and far short of a dominant wire. With the
promoter-ATAC negative, the physical-genome hypothesis for the tide is only weakly supported — the "usual suspects" are best
explained by baseline expression + network connectivity, not chromatin geography. Caveat: median 1 measured gene/TAD limits power;
shared stress *enhancers* within a TAD, or per-cell Perturb-ATAC, remain untested (Perturb-ATAC isn't cleanly available as an h5ad).

### Primary CD4⁺ T cells — cross-cell-type transfer (`cd4_transfer.py`)
Downloaded Shifrut-Marson 2018 (primary human CD4⁺ T-cell CRISPR KO, 52k cells, 20 immune-gene KOs) as the tractable stand-in for
the 22M-cell CD4 screen. **Zero knockouts overlap with K562** (the CD4 screen targets T-cell signaling genes), so a same-KO test
is impossible; tested generic transfer.

- **Tide conservation: ρ = −0.065** (0/50 usual-suspect overlap). The generic "usual suspect" program does **not** transfer to a
  primary immune cell — unlike between cancer lines (K562↔RPE1↔HCT116, which shared it). Honest reason: both the cell type *and* the
  perturbation biology (T-cell activation vs essential-gene stress) differ.
- **Magnitude transfer: ρ 0.29 (essentiality), 0.40 (PPI degree).** The size-vs-importance intuition *does* carry across cell types.

Net: the model's generic tide is a **cancer-cell-stress program, not a universal one**; only the magnitude/importance axis is
portable. Strongly reinforces the cross-cell-line finding.

### Time-resolved — see next section (Sci-Plex 24h vs 72h)
No sub-24h genetic perturb-seq exists as a clean h5ad (the ideal transient-window data would need GEO MTX assembly); the Aissa
scPerturb file turned out to be single-timepoint. The best available real time contrast is Sci-Plex 24h vs 72h.

---

### Time-resolved — Sci-Plex 24h vs 72h reshape (`sciplex_time.py`)
No sub-24h *genetic* perturb-seq exists as a clean h5ad (the Aissa scPerturb file was single-timepoint; the ideal data needs GEO
MTX assembly). The best available real time contrast is Sci-Plex 24h vs 72h — and only **A549** has both timepoints (K562/MCF7 are
24h-only). 47 A549 drugs with both:

```
profile Spearman(24h, 72h) ...... 0.062     (near zero — profiles barely correlate)
mover-set Jaccard(24h, 72h) ..... 0.128
transient (24h-only movers) ..... 77%
adaptive  (72h-only movers) ..... 77%
```
**Time demonstrably matters** — the response reshapes massively between 24h and 72h; a single steady-state snapshot misses most of
the time-specific structure. That's direct (proxy) support for the transient hypothesis behind the compensation wall. **Honest
caveats:** (1) for weak drugs the mover sets are near-noise, so the 77% is likely inflated by pseudobulk noise — but the near-zero
profile correlation and the biology (24h acute vs 72h survivor state) support a genuine large reshape for responding drugs; (2)
this is 24h-vs-72h *chemical* (A549), not the 2–12h *genetic* window where paralog compensation is expected. So it shows
time-resolution matters, but it's a proxy — the exact sub-24h genetic perturb-seq remains the real highest-value unmet fetch (GEO).

**Across all three requested sources:** time-resolution matters (Sci-Plex), but the ideal dataset to break the
transient-compensation wall isn't cleanly fetchable; 3D-TAD chromatin and primary-CD4 transfer were both honest negatives. The one
thing that *did* move a wall this stretch was the analogy/transfer model, not a new data source.

---

## Cell-type-specific network: does masking to K562-active genes help? (`fullstack_celltype.py`)

Your idea: run the same 60-KO deployment but activate the network **only for genes expressed in K562**, not all ~16.5k. Every
network feature (PPI degree, regulon, coexpr/codep degree, centrality, near-field) and the candidate universe restricted to
K562-active genes (baseline `clean_mean` > 0.05 in the K562 data itself; 8,561 of 16,492 network genes). Same panel + seed, run
both ways.

```
metric                 BASELINE(all)   MASKED(active)   delta
magnitude Spearman         0.459           0.426        -0.033
top-10 (all KOs)           1.98            2.03         +0.05
top-10 STRONG KOs          9.0             9.29         +0.29   (~7 KOs → within noise)
top-10 WEAK KOs            0.77            0.79         +0.02
```

**~Neutral on native K562 deployment** — every delta is within xgboost run-to-run noise (~±0.1 on top-10). (An earlier
too-aggressive threshold, `clean_mean`>0.2, actually *hurt* because it dropped ~48% of real movers from the universe; even at
0.05, 37% of real movers are lowly-expressed and become unreachable — masking by expression inherently loses some movers.)

**Two honest reasons it's a wash here:** (1) the deployment universe was *already* cell-type-specific — it scores each knockout's
*measured* (= K562-expressed) genes, so inactive genes were never candidates; masking mostly changes network-*degree* features the
magnitude model already handles. (2) The forecast's dominant signal is the generic expression/abundance prior, not fine graph
topology.

**The strategic point:** cell-type activation is the *correct*, principled thing to do (and it can't hurt much) — but its real
payoff is for **transfer to a different cell line** (mask to *that* line's active subgraph), which is exactly where the cross-line
result (K562→RPE1/HCT116, ρ 0.13) says cell-type specificity actually bites. On K562's own data the readout was already K562's
expressed transcriptome, so masking is redundant. (`fullstack_celltype.py` → fullstack_celltype.json.)

---

## Closing the loop: does expression → function reconstruct cell identity? (`celltype_coherence.py`)

Your integrative idea: reverse the per-cell-type mRNA to the genes it expresses, then check those genes' **functions** against the
cell's own function — and pair with protein and 3D. Built it on the `emask` (200 atlas cell types) + GO Biological Process.

- **Q1 — marker coherence (fixed a display bug first).** Each cell type's expressed genes recover its *specific* biology:
  hepatocyte → xenobiotic/fatty-acid/steroid metabolism + bile acid + coagulation; CD4 T cell → TCR signaling + adaptive immunity;
  neuron → synaptic transmission + axon guidance; monocyte → antigen processing + MHC; cardiac muscle → sarcomere + force of heart
  contraction. (My first pass ranked terms vs the global background and showed housekeeping "translation" for *every* cell type — I
  caught it and switched to cross-cell-type *specificity* ranking. Q2 below was always correct since it uses the full profile.)
- **Q2 — reverse identification (decisive).** Cell identity is reconstructable from function *alone*: each cell type's nearest
  neighbor by expressed-gene-function profile shares its lineage **78%** of the time vs **11%** chance (**lift 7.08×**), across 997
  GO-BP terms. Monocyte→monocyte, CD8-memory→CD4-memory (both T), cDC→DC, etc. **The mRNA layer and the gene-function layer are
  mutually consistent — the loop closes.**

**Honest limits on the other two pairings you asked for:**
- **Protein** — we have only a *generic* proteome (one `ppm` per gene, not per cell type). Active genes are 2.88× background
  abundance, but a true per-cell-type mRNA↔protein comparison is **not possible** without cell-type proteomes (a real data gap;
  HPA/Tabula proteomes would fill it).
- **3D chromatin** — only K562 Hi-C exists, and `missing_wire_3d` already showed TAD co-localization is a weak wire (0.016).
  Per-cell-type 3D for the 200 atlas types isn't in our data.

**Bottom line:** where the model *is* cell-type-resolved (mRNA + gene-function) it's internally coherent and reconstructs cell
identity (~7× lift); it is *not* cell-type-resolved for protein or 3D chromatin — those are the honest missing modalities to
acquire. (`celltype_coherence.py` → celltype_coherence.json.)

---

## Correction: how well can we actually "guess the exact cell"? (`celltype_id.py`)

The earlier coherence test's "78–80%" was **family** matching (same lineage) and was **not** held-out — so it overstated
"guessing the cell." Proper held-out identification (split genes in half; use one half to identify a profile built from the other;
out of all 200 cell types, chance 0.5%):

```
                              gene function   protein type
exact cell = #1 match .......    26%             18%
exact cell in top-3 .........    48%             38%
exact cell in top-5 .........    56%             50%
right family (#1) ...........    68%             44%
median rank of correct cell .   3 of 200        4 of 200
```

**Honest reading:** from just the functions/types of a cell's expressed genes we almost always get the right **family** (~68%),
and we pick the **exact** cell as #1 about **26%** of the time — far above the 0.5% chance, with the correct cell usually near the
very top (median rank 3 of 200, top-5 ~56%). Where exact-top-1 loses points is distinguishing near-identical sub-types (memory vs
effector CD8 T cells, monocyte subsets) that share most genes. So the corrected claim is "right family almost always, right exact
cell often and near-top otherwise" — **not** a literal ~80% exact ID. (`celltype_id.py` → celltype_id.json.)

---

## Combining all the feature views to identify the cell (`celltype_id_combined.py`)

The right version of the question: fuse gene FUNCTION + protein TYPE + LOCATION + PATHWAY into one profile, held-out exactly as
before (out of all 200 cell types, chance 0.5%).

```
feature (alone)        exact top-1   top-5   family   med.rank
FUNCTION (GO-BP)          26%         56%      68%        3
TYPE (GO-MF)              18%         50%      44%        4
LOCATION (GO-CC)         24%         54%      60%        4
PATHWAY (Reactome)       22%         57%      57%        3
COMBINED (all four)      38%         67%      71%        1
raw mRNA-content (Jaccard, family) ......... 72%  (not held-out; reported for context)
```

**Combining helps and the views are complementary:** exact top-1 rises from 26% (best single) to **38%** (+12 pts), top-5 to
**67%**, and the correct cell's **median rank drops to 1 of 200** — it's usually the single best pick. Right family ~71%. The
remaining misses are near-twin sub-types (memory vs effector CD8 T, monocyte flavors) that share most of their biology, which no
combination fully separates from expression annotation alone. (`celltype_id_combined.py` → celltype_id_combined.json.)

---

## Do we need a model per cell type? Fine-tune vs transfer across 3 lines (`fullstack_multicell.py`)

Tested knockouts on three cell lines with their own genome-scale perturb-seq — K562 (leukemia), RPE1 (retinal epithelium),
HCT116 (colon) — comparing a model **fine-tuned on each line** against **transferring the K562 model unchanged**. Held-out
deployment, scale-free rank-based movers (top ~1.5% by |z| per KO, so HCT116's compressed z-scale is comparable).

```
line      own fine-tuned top10    K562 transferred    gain from fine-tuning
K562            5.05                   (self)              —
RPE1            2.22                    0.95              +1.27  (>2x)
HCT116          6.98                    3.35              +3.63  (~2x)
```

**Yes — you clearly need a model per cell type.** Applying the K562 model unchanged to another line roughly **halves** deployment
(RPE1 0.95 vs its own 2.22; HCT116 3.35 vs its own 6.98). The reason is exactly the cross-line finding (ρ 0.13): the K562 model's
**tide prior** — which genes usually move in K562 — is the *wrong* prior for a different cell type, so transfer lands few of the
other line's top movers. Each line's own perturb-seq re-fits that prior to its own usual-suspect program and roughly **doubles**
deployment.

**Honest caveats:** (1) HCT116 needed the rank-based mover definition to be scorable at all (fixed |z|>1 gave ~0 movers). (2) Only
the forecast's **tide prior** is fine-tuned here — magnitude and near-field/regulon features are still generic, so a *fully*
cell-type-specific model would gain even more from cell-type essentiality + cell-type regulons (which we largely lack). (3) RPE1's
absolute number is lower (essential-gene screen, noisier pseudobulk), but the fine-tune ≫ transfer gap holds on every line.
(`fullstack_multicell.py` → fullstack_multicell.json.)

---

## Can it scale to all cell types? Data efficiency of fine-tuning (`celltype_scaling.py`)

The determinant of scaling: how much perturbation data does a new cell type need? Varied the number of training knockouts and
measured held-out top-10 deployment (tide prior learned from only those training KOs).

```
train KOs      K562 top-10     HCT116 top-10
   50            4.8             6.35
  100            5.4             7.08
  200            6.4             7.47
  400            6.4             8.73
  800            7.1             8.53
```

**Encouraging: the curve rises fast and flattens.** ~90% of full performance is reached by **~200–400 knockouts**, and even
**50–100 gives a usable model**. So a new cell type does NOT need a genome-wide screen to be covered — a few hundred well-chosen
knockouts fine-tune its tide prior to most of the achievable accuracy. That makes covering **many/most cell types realistic** as
perturb-seq atlases accumulate (the field already runs genome-wide screens on a few lines and smaller screens on many).

Honest limits carried forward: (1) each cell type still needs *some* of its own perturbation data — no free zero-shot transfer;
(2) the gain is on the deployable top-10 / generic tide, not the specific far field (still walled); (3) tissue/organ is a *different*
problem — a tissue is a *mixture* of cell types with cell–cell signaling and spatial structure, so it needs per-cell-type models
*plus* an intercellular layer we have neither built nor measured. (`celltype_scaling.py` → celltype_scaling.json.)

---

## Fine-tune WITHOUT a perturbation screen? Building the prior from annotation alone (`annotation_prior.py`)

`celltype_scaling` says a new cell type needs *some* of its own knockouts; the availability audit says most of the 200 atlas
cell types have **none**. So the question: can we build a cell type's tide **prior** — which genes usually move in the far field —
from **annotation alone** (gene function/proc, protein abundance `ppm`, TF in-degree, essentiality `dep_frac`, PPI degree), with no
perturbation data for that cell type? The cell-type-specific ingredient would then come only from *which genes the cell expresses*
(obtainable from ordinary scRNA-seq, not a screen). Validated on the three lines that *do* have ground-truth movers: rank each
cell's expressed genes by a score, count true movers in the held-out **top-10**.

```
                       K562 hit@10   HCT116 hit@10   RPE1 hit@10
real_tide (ceiling)      3.73          3.00           0.53        ← perturbation-learned prior
annot_apriori            1.28          0.72           0.13        ← annotation only, fixed weights
annot_xline (learned)    1.50          0.15           0.20        ← annotation→mover map learned on OTHER lines
depfrac_only             0.27          0.45           0.08
random                   0.17          0.12           0.15
```
*(Reproducible: candidate lists sorted + `PYTHONHASHSEED=0`; earlier unseeded runs drifted on the tie-heavy / near-floor scorers.)*

**Annotation is a weak cold-start, not a substitute for a screen.** Measured as *fraction of the real-tide ceiling recovered
above the random floor*, the fixed a-priori score recaptures only **~31% (K562) / ~21% (HCT116)** — roughly a fifth-to-a-third —
and effectively **0 on RPE1** (uninformative there: RPE1's own `real_tide` sits near the random floor, noisy pseudobulk, so nothing
could score). It *does* beat essentiality-alone and random on the clean lines, and ranks genes like the true mover-frequency at
**Spearman 0.14 / 0.35 / ~0** (K562 / HCT116 / RPE1). So the tide *is* partly written in gene annotation — essential,
highly-expressed, transcription/translation machinery move most — just a **minority** of it.

Two honest negatives fell out. (1) The **learned** cross-line map — training "what makes a frequent mover" on the *other* two
lines and transferring it — **does not transfer**: 37% recovery on K562 but it *collapses to ~1% (the random floor)* on HCT116,
because the annotation→tide mapping is itself cell-type-specific. The fixed a-priori score is the more trustworthy of the two. (2) Annotation gives a **generic** per-gene propensity; the only cell-type-specific ingredient available without a screen is
the expressed-gene gate, which is broad and weakly discriminating — so an annotation prior can only ever behave like a
generic/transferred tide, never the fine-tuned one. That is exactly the part that made K562→other transfer lose half its accuracy
(`fullstack_multicell`).

**Bottom line:** "fine-tune without perturbation data" buys a real but small cold-start (~21–31% of the deployable top-10 on clean
lines) for the **conserved, essential** part of the tide — genuinely better than random or essentiality-alone, and usable for the
~185 zero-screen atlas cell types — but it recovers only a minority of the signal and **cannot supply the cell-type-specific
reweighting**. That last part still needs some of the cell's own perturbations. Annotation narrows the cold-start gap; it does not
close it. (`annotation_prior.py` → annotation_prior.json.)

---

## Can we "reverse the cancer" to get normal-cell models? (`cancer_reversal.py`)

Almost all our perturbation data comes from **cancer lines**, but we want *normal*-cell models. The idea: take a cancer line's
knockout tide and **reverse the cancer** using knowledge of its driver lesion and *how* it acts — sign-aware, per the biology:
**subtract** an activating oncogene's program (ON in cancer, OFF in normal), **add back** a lost tumor-suppressor's program (OFF in
cancer, ON in normal). Implemented as: K562 (BCR-ABL + MYC-amp → subtract; **TP53-null** → add p53 targets), HCT116 (KRAS/PIK3CA/
β-catenin/MYC → subtract; **TP53-wt** → no add), RPE1 (near-normal hTERT line → the normal-ish anchor). Oncogene program =
GO cell-cycle/proliferation + measured driver-KO movers + MYC targets (~1,100 genes/line, only **14–16.5%** of the tide mass).

```
pair            raw ρ   corrected ρ   random-zero ctrl   surviving-genes ρ
K562~HCT116      0.18       0.35            0.10               0.18
K562~RPE1        0.245      0.43            0.12               0.243
HCT116~RPE1     -0.003      0.22           -0.01               0.012
```

**I corrected myself twice here** (this is the honest-iteration discipline working). First I guessed the effect would be *small*;
the run showed every correlation jumping **+0.19**. Then I guessed **tie-inflation artifact** (zeroing ~1,100 shared genes forces a
concordant (0,0) block). Both wrong:

- **Control 1 (random-block):** zeroing a *random* equal-size block does **not** reproduce the gain — it *lowers* correlation
  (~−0.07). So the effect is **specific** to the oncogene/proliferation program, not generic zeroing.
- **Control 2 (decisive, artifact-free):** agreement among the **surviving (non-oncogene) genes is unchanged** (Δ ≈ +0.006). So
  removing the cancer program yields **no cleaner universal signal** in the genes that remain — the +0.19 comes almost entirely from
  replacing the high-magnitude, cross-line-**discordant** proliferation block with zeros, which mechanically lifts the full-vector
  Spearman.

**What's actually true:** the proliferation/cell-cycle program is a real, high-magnitude, **cell-line-specific (discordant)** slice
of the tide — the part where cancer lines *differ*; the shared agreement lives in the essential-machinery remainder. **Why it still
cannot deliver normal-cell data:** (1) **no ground truth** — there is no normal, lineage-matched perturbation screen to validate
against; RPE1 is our only near-normal anchor and is itself a *proliferating* immortalized line (carries the same cell-cycle tide) of
a *different* lineage, so the "toward RPE1" gain is the same shared-zero mechanism, not de-cancered resemblance; (2) driver rewiring
is **non-linear** — normal ≠ cancer minus a fixed oncogene term; (3) the tide is dominated by cancer-**agnostic** essential
machinery, leaving little cancer-specific signal to remove.

**Bottom line:** the sign-aware framing (subtract activated oncogenes, restore lost suppressors) is mechanistically sound and does
isolate a real cell-line-specific proliferation component — a defensible covariate correction — but it neither cleans up the
surviving tide **nor**, absent a normal ground-truth screen, can be shown to recover a normal-cell response. Same lesson as the
annotation prior: knowledge can nudge and *decompose* a prior, but it cannot **manufacture** the cell-type-specific ground truth —
the normal cell still has to be measured. (`cancer_reversal.py` → cancer_reversal.json.)

---

## A 500–800-gene screen costs $200–400k — so what's the cheapest route? (`cheap_screen.py`)

The economic wall: a 500–800-gene KO Perturb-seq screen runs ~$200–400k (~$400–500/knockout). `celltype_scaling` already showed
~50–100 knockouts recover most of the deployable accuracy. Can we go *further* by choosing **which** genes to knock out smartly —
a-priori, from annotation, before running anything — rather than at random? On K562/HCT116 we fixed a held-out panel and built the
training set of size *n* by four strategies, learning the tide prior from only those *n* knockouts:

```
K562  top-10        n=20  n=30  n=50  n=100  n=200  n=400        HCT116       n=20  n=30  n=50  n=100 n=200 n=400
random              4.10  4.57  5.09  6.23   6.37   7.10         random       5.01  5.36  6.32  7.42  7.85  8.33
ppi_hub (degree)    2.62  3.57  4.15  5.28   5.88   7.37         ppi_hub      4.80  4.93  6.35  6.95  7.88  8.42
essential (dep)     3.93  4.45  5.62  6.10   7.45   7.07         essential    4.98  5.92  6.57  6.80  7.98  8.13
diverse (pathway)   3.92  4.28  4.73  5.70   6.27   7.02         diverse      4.77  4.82  6.80  7.93  8.28  8.63
```

**Smart guide selection is essentially a wash — and I corrected my own auto-verdict.** A "90%-of-best" threshold first made it look
like smart selection *halved* the screen, but that was a single noisy spike. The robust metric — mean top-10 delta vs random, pooled
over both lines and all sizes — says: **ppi_hub −0.46** (pure network-hub selection actually *hurts*, badly at small n), **essential
+0.10** (negligible), **diverse −0.05** (helps HCT116 +0.16, hurts K562 −0.26). No a-priori strategy reliably beats random on both
lines; the wins are scattered within run-to-run noise, and the one clear signal is that picking pure hubs is *worse*. So **which**
genes you knock out barely matters — a modest *random* targeted panel works about as well as any clever guide list (mild hedge:
prefer pathway-diverse/essential over pure hubs).

**The real cost lever is screen *size*, and it's strong.** At ~$400–500/KO, the ~50–100-KO plateau (K562 50→5.1 / 100→6.2 vs
400→7.1; HCT116 50→6.3 / 100→7.4 vs 400→8.3 — 70–90% of full accuracy) cuts cost to **~$20–40k (~10×)**, and even ~20–30 random KOs
give a usable model (K562 4.1–4.6, HCT116 5.0–5.4 top-10).

**Bottom line:** the cheapest *reliable* route to a per-cell-type model is a **small random (or mildly pathway-diverse) targeted
screen of a few tens of knockouts (~$10–40k)** — ~10× cheaper than 500–800 genes. Being clever about which genes buys little; the
money is saved by running *fewer*, not smarter. Honest limits carry over: it still needs *some* of the cell type's own perturbations
(cheaper, not free — the annotation-only prior recovers just ~21–31% of the ceiling), and the gain is on the deployable tide, not the
still-walled specific far field. (`cheap_screen.py` → cheap_screen.json.)

---

## The other cost lever: a cheaper *readout* (L1000/TAP-seq landmark panel) (`cheap_readout.py`)

An exhaustive research sweep (13 agents) flagged the second cost lever: not fewer knockouts, but measuring **fewer genes** —
L1000/TAP-seq read a ~1000-gene targeted panel instead of the whole transcriptome (~10–50× cheaper sequencing). Does that preserve
the deployable tide? On K562/HCT116 I restricted the measured universe to a panel of size *P* — `tide_hub` (the *P* most-frequently-
moving genes from training, the "landmark" analogue) vs `random` — and re-ran the fine-tuned forecast.

**I caught my own metric trap.** The `tide_hub` panel made held-out top-10 look like **95–158% of full** — *higher than whole-
transcriptome*. That's a **circular artifact**: restricting the candidate pool to pre-vetted frequent movers (and grading against
panel-restricted movers) makes "predict a mover" trivially easy and just re-reports the generic tide. The honest, unambiguous metric
is **coverage** — what fraction of a knockout's *true* whole-transcriptome movers the panel even contains:

```
panel size      tide_hub coverage    random coverage
   300              29%                  3%
   500              37%                  5%
  1000              52%                  9%       (5.5× more efficient than random)
  2000              70%                 20%
```

**The tide is concentrated, so a smart landmark panel is far more efficient than random** — a 1000-gene hub panel covers ~52% of
true movers vs ~9% for random. **But even 1000 genes miss ~half of each knockout's movers, and 2000 still miss ~30%** — and the
missed part is the diffuse, knockout-specific far field.

**Net for cost:** a cheap targeted readout (~10–50× cheaper sequencing; TAP-seq keeps single cells + pooled guides) is a genuine
lever for the deployable **generic tide** — ~half the response with ~5% of the genes — and stacks with the small-screen (~50–200 KO)
lever. But (1) it discards ~30–55% of the response, all knockout-specific, so it *cannot* recover the far field; (2) the apparent
top-10 "parity" is a metric artifact, not real parity; (3) sequencing is only part of cost — library synthesis, delivery, and labor
are fixed; (4) it presumes a fixed landmark panel transfers across cell types (true for the generic tide, not cell-type-specific
movers). **Bottom line:** cheaper readout + fewer knockouts together bring a per-cell-type *tide* model to the low tens of $k, but
the cheap readout buys the generic tide (~half the response), not the specific movers — and neither lever removes the need for *some*
of the target cell's own perturbations. (`cheap_readout.py` → cheap_readout.json.)

---

## Fine-tuning cancer cell models from *free* perturbation data (`scperturb_finetune.py`)

If a $20–40k screen is out of budget, the honest $0 route is to use screens **other labs already paid for** and released free on
scPerturb/Zenodo. How many distinct cell types can we actually fine-tune from that free corpus? A general loader (per-cell CP10k
log-normalization, pseudobulk by gene, guide→gene mapping, OOM-safe in-place CSR/CSC) + control-relative variance-shrunk z feeds the
*same* fine-tune + held-out top-10 as our K562/HCT116 pipeline.

**The normalization mattered enormously — and it's a clean worked example of the honesty loop.** A first *crude* pseudobulk
(sum-CPM + cross-KO z) gave low scores, and I flagged them as a likely **processing floor, not biology**. Fixing the normalization
then *confirmed it* — every raw-count line roughly **tripled**:

```
cell type (source)     KOs   cells/KO   crude → IMPROVED   ≥5/10?
Melanoma (Frangieh)    218    ~1000      3.00 → 7.85         ✅
RPE1 (Replogle)       2151    ~115       2.22 → 7.72         ✅
HCT116 (pre-z-scored) ~1300   deep       —    → 6.98         ✅
HepG2 (Nadig 2024)    2151    ~68        1.43 → 5.47         ✅
Jurkat (Nadig 2024)   2151    ~122       2.50 → 5.42         ✅
K562 (pre-z-scored)   ~1300   deep       —    → 5.05         ✅
THP-1 (Papalexi)        23    ~900       0.57 → 2.86         (23 KOs, below floor)
```

**Six distinct cell types now score ≥5/10 held-out top-10 — all from free data, zero wet-lab cost:** Melanoma 7.85, RPE1 7.72,
HCT116 6.98, HepG2 5.47, Jurkat 5.42, K562 5.05. (K562/HCT116 come from pre-z-scored matrices; the rest were fine-tuned here from
raw counts.) The earlier low numbers really were the crude pipeline — RPE1 went 2.22 → 7.72 under the same fine-tune, just better
normalization.

**Honest caveats:** (1) this is the deployable **tide** (which genes tend to move) — a cleaner tide is more *predictable*, so much of
the gain is measuring the generic stress program well, **not** cracking the knockout-specific far field (still walled); (2) THP-1's
2.86 rides only 23 distinct KOs (below the fine-tuning floor); (3) the normalizer is proper but still simple — a scran/edgeR-style
pipeline could refine further; (4) all cancer/immortalized lines plus RPE1 (near-normal) — normal tissue stays uncovered.

**Bottom line:** with correct normalization the free perturb-seq corpus yields **six fine-tuned cell-type models at ≥5/10** at $0 — a
real multi-cell-type asset, more addable as screens accumulate (Datlinger, Dixit, McFarland MIX-seq, …). The hard ceiling is
unchanged: the deployable generic tide, not the cell-type-specific far field. (`scperturb_finetune.py` → scperturb_finetune.json.)

---

## Can we mine novel hypotheses for researchers? Turning the "wall" into a discovery list (`novel_links.py`)

We can't *predict* the knockout-specific far field, but it's **measured**. So: for each knockout X, take its strongest *specific*
movers Y (rank-based, non-tide) and keep only those with **no known relationship** to X in any layer (PPI, TRRUST regulon, complex,
Reactome pathway, GO-BP process, co-expression, co-dependency) — measured strong effects with no annotated mechanism. The intended
deliverable: the ones **reproducible in both K562 and HCT116**, with a **permutation chance control**.

**First pass (2 lines) was at chance; the 6-line version works.** With only K562+HCT116, of 15,923 unexplained effects just 9
reproduced vs ~4 by chance (2.2×) and those were weak — no credible list. So I re-ran across **all six fine-tunable lines**
(K562/HCT116 from z-matrices; Melanoma/RPE1/HepG2/Jurkat from the improved pseudobulk), requiring reproducibility in **≥3 of 6 lines**
with the same sign, plus a permutation chance control.

**Result: 332 effects reproduce in ≥3 of 6 lines vs ~0 expected by chance** — permuting knockout labels leaves essentially *zero*
3-line agreements, so the reproducible set is *massively* above chance (real, non-random). Two honest qualifiers: most are weak
(median |z| 0.68; only 10 reach |z|≥2), and the strong ones still include known stress leaking as annotation gaps (HSPA5→HYOU1/CRELD2
= textbook UPR; PSMA3/PSMB5→HSPA1B = proteasome→HSP70).

**The credible, possibly-novel signal is *complex-convergence* — a moved gene hit reproducibly by many distinct subunits of one
machine** (a whole complex agreeing is real, not noise):

```
moved gene   ← # distinct reproducible KOs      interpretation
PPP1R10      ← 9  (EXOSC2/3/4/5/8/9, DIS3, MTREX)   RNA exosome → PPP1R10 mRNA as a surveillance substrate
RASSF1       ← 7  (UPF1, UPF2, SMG7, KPNB1, …)      NMD factors → RASSF1 as an NMD substrate
MTHFD2L      ← 5  (ZFC3H1, ZC3H3, EXOSC…)           PAXT/nuclear exosome → MTHFD2L substrate
CRELD2/HYOU1 ← 4  (HSPA5, DERL2, CALR, …)           known UPR program (validates the method)
HSPA1B       ← 3  (PSMA3, PSMB5, …)                 known proteasome→HSP70 (validates the method)
```

The RNA-decay-machinery → specific-mRNA-up hypotheses (**exosome→PPP1R10, NMD→RASSF1, PAXT→MTHFD2L**) are internally consistent and
directly testable (measure the mRNA's stability on knockout). That the method *also* re-derives the known UPR and proteotoxic
programs is a validation that the convergence signal is real.

**Deliverable for researchers:** prioritize the complex-convergent targets (especially **exosome→PPP1R10**), then the strong
reproducible pairs after a STRING/OmniPath check to drop known stress. Full ranked list: `novel_links_candidates.csv`. Honest
framing: this is hypothesis *generation* on measured multi-line data, not validated discovery — most raw hits are weak or
known-stress, **convergence is the filter that makes it credible**, and a complete interactome would sharpen it further. But unlike
the 2-line version, **this is the version worth a researcher's time, led by the convergent targets.** (`novel_links.py` →
novel_links.json + novel_links_candidates.csv.)

---

## Validating the top lead: exosome→PPP1R10 vs literature + our own stack (`exosome_pnuts_stack.py`)

I took the strongest discovery — **RNA exosome knockout → PPP1R10/PNUTS mRNA up**, reproducible in ≥3 of 6 lines — and cross-checked
it against the literature and ran it back through our own model.

**Literature: novel, but mechanistically predicted.** PubMed has **zero** co-mentions of PPP1R10 and "exosome" (despite 69 PPP1R10
papers), so the direct link is undocumented. But it fits a *known paradigm*: PNUTS/PPP1R10 is itself a core premature-termination
factor (PP1-PNUTS is *required* for the Restrictor complex ZC3H4/WDR82,
[Cell Reports 2025](https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00335-3)); the RNA exosome degrades prematurely-
terminated protein-coding transcripts ([Davidson 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6403362/)); and termination-
factor genes autoregulate via exosome-sensitive premature termination (NRD1/HRP1/PCF11,
[PMC7471841](https://pmc.ncbi.nlm.nih.gov/articles/PMC7471841/)). So "exosome KO → PNUTS mRNA up" is a termination factor
autoregulating its own message via the exosome — applied to a gene where it has never been shown.

**Our own stack confirms it's a genuinely missing edge, not something the model knew:**
- **Forecast** — our predictor emits the *same generic tide* (FTH1, MT-ATP6, MALAT1…) for every exosome KO and *never* predicts
  PPP1R10 (which moves in only 7/237 KOs — specific, not tide). The model could not have surfaced this by forward prediction; it had
  to be **mined from the measured far field** (the wall in action).
- **Propagate** — *no* exosome subunit reaches PPP1R10 through any known edge (PPI/complex/pathway/regulon) → a **missing edge** in
  our knowledge graph, matching the zero PubMed co-mentions. Meanwhile PPP1R10's *own* blast radius is coherent and validates the
  graph: PPP1CC/CA/CB (the PP1 catalytic subunits PNUTS regulates), **WDR82** (the Restrictor/termination component PNUTS activates),
  SSRP1/TOP1 (transcription), CPSF6/SNRP* (RNA 3′-processing/splicing).
- **NEXUS / dependency** — all 8 exosome subunits *and* PPP1R10 are core essential dependencies (dep_frac 0.90–1.0), so this
  autoregulatory loop sits on essential machinery. NEXUS's structural ΔΔG sensors don't score an expression change (this is
  regulatory, not a coding mutation), so NEXUS contributes the essentiality context, not a stability number.

**Bottom line:** our model neither predicted nor can explain exosome→PPP1R10 — which is exactly the point. It's a real, measured,
mechanistically-plausible but **un-annotated** link — a missing edge — and a concrete, testable hypothesis to hand a researcher:
*does the exosome directly degrade a prematurely-terminated PPP1R10 transcript (autoregulation of PNUTS via exosome-sensitive
premature termination)?* (`exosome_pnuts_stack.py` → exosome_pnuts_stack.json.)

---

## Making the discovery self-contained: the software proposes the connection with a confidence score (`connection_proposer.py`)

**The honest critique that prompted this.** On exosome→PPP1R10 the division of labour was uncomfortable: the *model* couldn't
predict it, the *graph* couldn't connect it, so it was **I (the reasoner)** who eyeballed the mined table, noticed a whole complex
converging on one gene, and called it a hypothesis. That is "Opus did it," not "the software did it." The fix is to move the
*judgement that makes a convergence credible* out of the reasoner and into deterministic code that emits a **ranked proposal with a
confidence number**, run by anyone, agent-independent.

**What the tool does (no LLM in the loop for the score).** From the six lines' measured z-profiles it (1) finds reproducible,
knockout-specific gene→gene effects; (2) for each moved target, takes the set of knockouts that reproducibly move it and asks *does
that knockout set form a real annotated machine?* — a hypergeometric enrichment against CORUM complexes + Reactome pathways + GO
cellular-components (capped at 120 members so a loose compartment like "nucleus" can't score); (3) gates that complex-coherence by
cross-line reproducibility. The confidence is `coherence · (0.4 + 0.6·reproducibility)`, both bounded to [0,1].

**Result — the tool emits the hypothesis on its own, ranked #1:**

```
target     confidence  machine (annotated)                       converging KOs / in-complex
PPP1R10    0.82        nuclear exosome (RNase complex)           11 / 10   EXOSC2/3/4/5/6/7/8/9,DIS3,MTREX  ← NOVEL, #1
HYOU1      0.79        oligosaccharyltransferase / ER (UPR)      12 / 4    HSPA5,DDOST,RPN1,DAD1,…          (known UPR)
LETMD1     0.78        exon-exon junction complex (NMD)          6  / 4    UPF1,UPF2,RBM8A,MAGOH,SMG5/7     (known NMD)
HSPA1B     0.76        proteasome complex                        7  / 6    PSMC1,PSMA3,PSMB5/6,…            (known → HSP70)
MTHFD2L    0.75        nuclear RNA decay (PAXT/exosome)          17 / 12   EXOSC…,ZFC3H1                    (novel-ish)
SCO2       0.74        m6A methyltransferase complex             4  / 4    METTL3,METTL14,RBM15,ZC3H13      (novel)
SNRNP40    0.73        Ino80 chromatin-remodeller                5  / 4    INO80,ACTR8,INO80B,NFRKB        (novel)
```

`PPP1R10 ← nuclear exosome, confidence 0.82` (coherence p≈2×10⁻³¹, 10 of 11 converging knockouts inside the named complex) is now
produced by **code, not by me**. The full ranked novel list (80 rows, `connection_proposer_candidates.csv`) is a coherent set of
RNA-metabolism machine→target hypotheses.

**Calibration — and its honest weakness.** Confidence separates convergences that map to a real annotated complex from random gene
sets at **AUC 1.0** — but that is *partly definitional*, because coherence is literally in the score, so I flag it rather than sell
it. **The decisive validation is the positive control:** the highest-confidence proposals are dominated by **known machine→response
programs the method was never told** — proteasome→HSP70 (HSPA1B), UPR (HYOU1/DNAJB11 ← OST/ER machinery), NMD (LETMD1 ← exon-junction
complex), splicing. It re-derives textbook biology at the top on its own, so a high score genuinely tracks real coherent machinery —
and the novel proposals (exosome→PPP1R10 0.82, m6A→SCO2, Ino80→SNRNP40) sit in the **same range** as those knowns.

**Honest bounds — what the number is and isn't.** The confidence is `P(this reproducible convergent effect looks like a real
coherent-machine effect)`, **not** `P(true direct mechanism/edge)`: high means "very unlikely a fluke, and it matches a real
machine," not "proven edge." The tool **cannot** supply the mechanism or the novelty-vs-literature judgement — that still needs a
reasoner. It only scores machine→target *convergences*; a lone pairwise effect with no complex behind it earns no coherence and
stays low, so genuinely novel one-off edges are invisible to it by construction.

**What actually changed (the point of the critique).** The ranking and the number are now deterministic code from data. "Opus
eyeballed a table" becomes "the software proposes X ← machine, confidence Y." The **honest division of labour**: the software
scores and ranks the convergences; a reasoner still owns choosing *which* high-confidence proposal to chase and reasoning out *why*
(the mechanism, the literature novelty, the experiment). (`connection_proposer.py` → connection_proposer.json +
connection_proposer_candidates.csv.)

### Per-line: run it on the other 5 cell lines (`connection_proposer_perline.py`)

The pooled proposer above blends all six lines. Running it **separately on each of the other five** fine-tunable lines (HCT116,
Melanoma, RPE1, HepG2, Jurkat) asks a different question: *what novel machine→target connection does each cell type propose on its
own?* — cell-type-specific hypotheses, still deterministic code, no LLM in the score.

**Honest method difference, stated up front.** A single line has **no cross-line reproducibility gate** — the pooled tool's main
noise filter. So per-line the only credibility gate is *within-line complex-convergence*: a target must be hit, same sign, by ≥3
distinct knockouts that together form an annotated machine (CORUM/Reactome/GO, ≤120 members), plus non-tide + effect-size filters.
That is **genuinely weaker, noisier evidence** than the pooled run. Three consequences I surface rather than hide:

1. **Confidence saturates at 1.00** in the big lines (RPE1/HepG2/Jurkat, ~2151 KOs) — any large coherent machine maxes both the
   coherence and subunit-convergence terms, so the per-line *number* cannot rank the top tier. **Rank by cross-line concordance, not
   the per-line confidence.**
2. **The top of each per-line list is dominated by large *known* response programs** the method correctly re-derives — V-ATPase /
   lysosome → sterol/SREBP (SQLE, HMGCS1, MSMO1, DHCR7…) in RPE1/HepG2, integrated-stress/eIF2 & proteasome→HSP70, IFN-γ→ISG
   (STAT1, GBP2, IDO1…) in melanoma. These *validate* the method but are **not novel**; they read "novel" only because the downstream
   target isn't itself a pathway *member*.
3. Per-line calibration (confidence separates real machine-convergences from a random-gene null) holds — **AUC 0.99+ in every line**.

**The trustworthy deliverable is the cross-line concordant set** — 24 novel targets proposed *independently* by ≥2 of the 5 lines
(two cell types agreeing *without* pooling is real, not a single-line artifact). It cleanly re-surfaces the RNA-decay-machinery
cluster and separates it from the per-line program noise:

```
target      lines  machine                    per-line confidence
PPP1R10       2     nuclear exosome (RNase)    HepG2:1.00, Jurkat:1.00     ← pooled tool's #1, now confirmed unaided
MTHFD2L       3     nuclear RNA decay          RPE1:1.00, HepG2:1.00, Jurkat:0.65
SCO2          3     m6A methyltransferase      HepG2:0.80, RPE1:0.62, Jurkat:0.62
SAR1A         2     m6A methyltransferase      HepG2:0.56, Jurkat:0.56
PMF1          2     INTAC complex              RPE1:0.65, Jurkat:0.65
SNRNP40       2     Ino80 complex              Jurkat:0.80, HepG2:0.68
LETMD1 / BAG1 2     nonsense-mediated decay    Jurkat / RPE1 / HepG2
VMP1 / BANP   2     nuclear RNA decay          HepG2 / RPE1 / Jurkat
HYOU1         2     OST / ER (UPR)             RPE1:0.90, Jurkat:0.80   (re-derived known)
HSPA1B/MLLT11 2     proteasome complex         RPE1 / HepG2 / Jurkat    (re-derived known)
```

**Key result.** The pooled tool's #1, **exosome→PPP1R10**, is now **independently proposed by HepG2 *and* Jurkat on their own**
(confidence 1.00 each) — so it was **not** an artifact of the cross-line pooling; two cell types propose it unaided. That is the
strongest cross-line confirmation of the exosome→PPP1R10 hypothesis so far. Bottom line: five cell-type-specific ranked hypothesis
lists (`connection_proposer_perline_candidates.csv`, 204 rows), and a 24-target cross-line-concordant set — led by exosome→PPP1R10 —
as the part worth a researcher's time. (`connection_proposer_perline.py` → connection_proposer_perline.json +
connection_proposer_perline_candidates.csv.)
