# Integration plan — wiring the unused data into *new derived data*

Goal: not "add more raw layers" but **produce new derived data** by chaining datasets, and score
where independent lenses converge. Raw data is commodity; the derived convergence layer is the asset.

Guiding rule (from `DISCOVERY_ENGINES.md`): a functional link supported by ≥3 *independent* assays
but absent from annotation is a novel, self-validating discovery. So every new layer must either
(a) add an **independent lens**, or (b) produce a **testable prediction**.

---
## Part 1 — What each unused/underused dataset provides (analysis)

| dataset | in Drive | modality | direct content | NEW data it can yield | priority |
|---|---|---|---|---|---|
| **ARCHS4** (`archs4_human_gene.h5`) | 59 GB | expression | ~1M+ uniformly-processed human RNA-seq samples, gene×sample | **co-expression network** (8th independent lens); context-specific networks; expression modules | **P1 — highest** |
| **ReMap** (`remap.bed.gz`) | 1.4 GB | TF ChIP-seq | genome-wide TF binding peaks | TF→target binding edges → (×Perturb-seq) **causal regulome** | P2 |
| **GTEx trans-eQTL** | (needs URL) | genetics | variant→distant-gene associations | cross-gene genetic-regulation edges | P3 |
| **NicheNet** (`.rds`) | 250 MB | prior model | ligand→target regulatory potential | tissue-model downstream wiring | P3 (tissue) |
| **DoRothEA / TRRUST** | small | regulation | curated TF→target | union with CollecTRI → regulatory coverage | P4 (cheap) |
| **RNAcentral** | 160 MB | ncRNA | non-coding RNA registry + genome mapping | ncRNA parts-list; needs a target DB (miRTarBase) to become interactions | P5 (stretch) |

Homework flags up front: ReMap/GTEx are **binding/association, not causal** (unsigned candidates);
ARCHS4 co-expression is **confounded** (batch, tissue composition) and must be rank-based + robustly
normalized; RNAcentral alone is a registry, not interactions.

---
## Part 2 — The new derived-data products (chains → new data)

Each product = inputs → method → **new data** → validation.

### Product A — Co-expression network (from ARCHS4)  ⟵ the missing 8th lens
- **Inputs:** ARCHS4 gene×sample matrix.
- **Method:** subsample ~100k samples → per-sample CPM + log1p → per-gene rank (Spearman) → gene×gene
  correlation → keep top-50 partners per gene.
- **New data:** `coexpr_neighbors.json` — an independent functional-partner layer.
- **Validation:** does it recover Reactome pathways (GBA-style, target ≥ our 23%)? report the number.

### Product B — Convergence functional-link score  ⟵ THE meta-product
- **Inputs:** co-expression (A) + PPI (STRING/BioPlex/OpenCell/HuRI) + co-essentiality (DepMap) +
  Perturb-seq neighbors + shared complex/pathway.
- **Method:** for every gene pair, count how many *independent* lenses support it; weight by lens
  reliability; flag pairs with high agreement **and** no annotation/literature.
- **New data:** `convergence.json` — ranked novel functional links (the KIDINS220-XPR1 pattern, at scale).
- **Validation:** known complexes score high (positive control); PubMed co-occurrence check on the novel top-N.

### Product C — Causal regulome  ⟵ upgrade binding → regulation
- **Inputs:** ReMap (TF binds gene) ∩ Perturb-seq/LINCS (TF-KO actually moves gene) [∩ GTEx].
- **Method:** keep a TF→target edge only when binding **and** measured response agree.
- **New data:** `causal_reg.json` — causally-supported edges, incl. ones in no curated DB.
- **Validation:** built-in (the perturbation is the causal test); recovery of CollecTRI as positive control.

### Product D — Dark-gene function, convergence-upgraded
- **Inputs:** convergence score (B) restricted to the 5,006 dark genes.
- **Method:** assign function by the majority pathway of a dark gene's convergent (multi-lens) neighbors.
- **New data:** stronger `darkfn` — replaces today's single-lens (co-essentiality-only) calls.
- **Validation:** hold out annotated genes, predict their pathway from convergent neighbors, report recall.

### Product E — Context-specific networks (ARCHS4 subsets)
- **Inputs:** ARCHS4 samples split by tissue/disease metadata.
- **Method:** co-expression within each context → compare to global → context-rewired links.
- **New data:** `context_networks.json` — which links are condition-dependent (feeds precision-target engine).
- **Validation:** tissue-specific known pathways enrich in the matching context.

### Product F — ncRNA regulatory layer (stretch)
- **Inputs:** RNAcentral (registry) + an external target DB (miRTarBase / lncRNA targets).
- **New data:** ncRNA→gene edges. **Blocked** until a target DB is added — RNAcentral alone won't do it.

---
## Part 3 — Chains that make *new* data (explicit recipes)

1. **ARCHS4 co-expr × PPI × co-essentiality × Perturb-seq → convergent novel links** (Product B). The
   core engine: 4 orthogonal assays agreeing where no paper connects the genes.
2. **ReMap binding × Perturb-seq response → causal edges** (Product C). Neither alone is causal; the
   intersection is.
3. **Co-expr modules × dark genes → function** (Product D). A dark gene sitting inside a co-expressed
   module inherits a testable functional hypothesis.
4. **ARCHS4 context × DepMap essentiality → context-specific dependency.** Which tissue/disease expresses
   an essential gene's dependency context → who a target would work in.
5. **Co-expr × ReMap → active regulation.** TF binds a gene *and* they co-express → the edge is likely on.
6. **DoRothEA ∪ TRRUST ∪ CollecTRI ∪ causal-regulome → the fullest regulatory graph** we can assemble.

---
## Part 4 — Homework / technical analysis

- **ARCHS4 (the hard one):** 59 GB h5 — cannot load whole. Verify layout with h5py keys
  (`/data/expression`, `/meta/genes`, `/meta/samples`); expression is raw counts → normalize (CPM+log1p)
  before correlation. Subsample ~100k samples (20k genes × 100k × float32 ≈ 8 GB → high-RAM runtime);
  rank-transform (Spearman) to beat batch effects; corr matrix 20k×20k ≈ 1.6 GB → keep top-50/gene only.
  The `.h5.part` (1.3 GB) is a stale partial download → delete.
- **ReMap:** processor already written (`compute_remap.py`); just needs `refGene.txt.gz` present and to
  actually run (last time it was skipped by the fast-assemble path). Confirm the peak→TSS window.
- **GTEx:** find a stable `trans_qtl_pairs` URL (still unresolved).
- **Symbol harmonization:** the AARS→AARS1 / ENSG issues we already hit — every new source must pass
  through the same symbol/ENSG resolver used for Perturb-seq.
- **Convergence scoring:** dedupe pairs, decide lens weights, and cap per-gene degree so hubs don't dominate.
- **Compute budget:** ARCHS4 co-expression is the only heavy step; everything downstream is cheap.

---
## Part 5 — Sequenced plan

- **Phase 1 — ARCHS4 co-expression (Product A).** Biggest unused asset, adds the independent lens, and
  *doesn't depend on Model 4* (which came back weak). Ship `coexpr_neighbors.json` + the pathway-recovery number.
- **Phase 2 — Convergence engine (Product B + D).** Once co-expression exists, score every pair across all
  lenses; regenerate dark-gene function from convergence; run the PubMed novelty check on the top-N.
- **Phase 3 — Causal regulome (Product C).** Run ReMap, cross with Perturb-seq, emit causal edges.
- **Phase 4 — Context networks (E), GTEx, NicheNet (tissue), DoRothEA/TRRUST union.** Incremental.
- **Phase 5 — ncRNA (F).** Only after adding a target DB.

---
## Part 6 — Honest caveats

- **Convergence de-risks, it does not validate.** High multi-lens agreement raises confidence; only
  wet-lab or held-out data confirms. This produces *credible hypotheses*, not proven facts.
- **Co-expression is the most confounded lens** — batch/tissue structure inflates correlations; rank +
  context-splitting mitigates but never removes it. Weight it below physical/genetic evidence.
- **Binding ≠ regulation, eQTL ≠ causation** — ReMap/GTEx edges stay labeled "candidate."
- **This is a data + validation story, not a Model-4 story.** After Model 4's weak result, the value now
  lives in convergence across measured lenses, not in a learned regressor.
