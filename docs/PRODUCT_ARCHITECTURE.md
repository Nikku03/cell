# Full cell-model AI — product architecture

What a final-product-grade whole-cell AI would be made of: every layer, the data source behind it, the model
that consumes it, and — honestly — whether we've **built it** (✓, with a scorecard axis), **partly built** it
(◐), have it **planned** (○), or need **external data/compute** we don't yet have (⚑).

The shape is a stack. Each layer feeds the one above; the **validation spine** and **confidence/provenance**
run vertically through all of them.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  QUERY LAYER   natural-language / API → router → answer       │
  │                + confidence + provenance + abstention         │
  ├─────────────────────────────────────────────────────────────┤
  │  REASONING     metabolic FBA · regulatory/signaling dynamics  │
  │  ENGINES       · mutation→phenotype chain · causal inference  │
  ├─────────────────────────────────────────────────────────────┤
  │  REPRESENTATION  protein-LM · structure nets · learned GNN ·  │
  │  (the AI models) single-cell foundation models · kinetics     │
  ├─────────────────────────────────────────────────────────────┤
  │  DATA FOUNDATION   genome · structure · networks · expression │
  │                    · perturbation · kinetics · variants · clin│
  ├─────────────────────────────────────────────────────────────┤
  │  INFRASTRUCTURE   fetch/cache · ID-mapping · CPU+GPU · seeds   │
  └─────────────────────────────────────────────────────────────┘
        VALIDATION SPINE (recovery scorecard) threads all layers
```

---

## 1 · Data foundation — the sources

| Domain | Source(s) | Feeds | Status |
|---|---|---|---|
| Genome / genes / proteins | Ensembl/GENCODE, **UniProt** (seq, domains, PTM), HGNC | everything; ID backbone | ✓ (symbols) |
| ID mapping | **mygene.info**, Ensembl BioMart | ENSG↔symbol↔UniProt glue | ✓ |
| Protein structure | **AlphaFold DB** (EBI API, per-UniProt), PDB | ΔΔG, structure-kcat, interfaces | ✓ fetch |
| On-demand folding | **ESMFold / AlphaFold3** | mutant + uncovered structures | ⚑ GPU |
| PPI | STRING, BioGRID, IntAct | the graph (ppi edges) | ✓ |
| Signaling | **Reactome (CC0)**, OmniPath; SIGNOR (CC-BY-NC⚠), KEGG (paid⚠) | graph (sig), causal cascade | ◐ |
| Gene-regulatory | DoRothEA/**CollecTRI**, ENCODE, hTFtarget | graph (reg), TF programs | ✓ |
| Ligand–receptor | **CellPhoneDB**, NicheNet, CellChat | tissue comms | ✓ |
| Single-cell expression | **CELLxGENE census**, Human Cell Atlas, Tabula Sapiens | emask (cell-type identity), node features | ✓ |
| Tissue bulk | GTEx | context priors | ○ |
| **Perturbation (interventional)** | **Replogle Perturb-seq**, **DepMap**, LINCS L1000, Tahoe-100M | cause-finder, essentiality, response models | ✓ (Replogle+DepMap) |
| Metabolic network | **Human-GEM**, Recon3D | FBA / ecModel | ✓ |
| Measured kinetics | BRENDA, SABIO-RK | kcat/Km ground truth | ✓ (via CatPred set) |
| Absolute proteomics | **PaxDb**, ProteomicsDB (copies/cell) | real capacities for in-cell kcat | ⚑ (have ordinal only) |
| In-vivo effective kcat | Davidi 2016 | kinetics prior | ◐ (E.coli-mapped) |
| Stability (ΔΔG) | **S2648/S669**, Tsuboyama mega-scale | ΔΔG training | ✓ |
| Variants / labels | **ClinVar**, gnomAD (AF), ProTherm/ThermoMutDB | variant effect, mutation module | ○ |
| Clinical ground truth | OMIM, Orphanet, **IEM biomarkers**, Open Targets | disease→target, IEM validation | ✓ (IEM) |

---

## 2 · Representation layer — the AI models

The learned functions that turn raw data into predictions. This is where GPUs earn their keep.

| Model | Maps | Status |
|---|---|---|
| **Protein language model** (ESM-2/3) | sequence → embedding + zero-shot variant effect | ⚑ GPU, planned node-feature source |
| **CatPred / RealKcat** | (substrate SMILES + sequence) → kcat/Km, incl. mutants | ✓ WT set / ○ mutant fork |
| **Structure predictor** (AlphaFold/ESMFold) | sequence → 3D structure | ✓ fetch / ⚑ fold mutants |
| **ΔΔG predictor** | structure+mutation → stability change | ✓ (DDGun-tier, S669 0.41) |
| **KcatNet** | structure → kcat (fixes CatPred's tail) | ○ GPU |
| **Learned GNN** (GraphSAGE→R-GCN/GAT) | cell graph → link/function/perturbation | ✓ GraphSAGE (beats fixed +0.05 AUC) |
| **Single-cell foundation model** (Geneformer/scGPT/scFoundation) | expression → cell-state embedding, node features | ○ GPU |
| **Perturbation-response model** | (perturbation, cell state) → Δexpression | ◐ (direction via CellGraph/Replogle) |

---

## 3 · Reasoning / simulation engines

Where the representations get composed into answers about the cell as a *system*.

- **Metabolic — enzyme-constrained FBA (ecModel/GECKO)** on Human-GEM: `v ≤ kcat·[E]`. Answers viability,
  flux magnitude, quantitative knockdown. ✓ (simplified per-reaction caps; ○ full proteome-pool GECKO).
- **Regulatory / signaling — logical attractor dynamics + learned GNN propagation**: perturbation → downstream
  direction/state. ✓ (Boolean reversal + CellGraph perturb-direction).
- **The mutation→phenotype chain** (the crown): `mutation ─┬─ seq→kcat ─┐ └─ struct→ΔΔG→[E] ─┴─ ec-flux →
  pathway → phenotype`. ✓ ΔΔG + ec-flux + folding link; ○ mutant-kcat fork; ○ end-to-end IEM validation.
- **Causal inference**: disease→target (3-layer causal→perturb-to-wildtype→druggability) + measured "alibi"
  test on real knockouts. ✓
- **Temporal trajectory** (mRNA(t) after perturbation): the error-ladder track. ○ not built (see FUTURE_IDEAS §C).

---

## 4 · Variant / mutation module (what "mutation generation" means)

Three distinct jobs, all feeding the mutation→phenotype chain:

1. **Ingest real variants** — ClinVar / gnomAD / a patient VCF → map to protein position → run the chain →
   pathogenic-vs-benign call with mechanism. ○ (chain built; ingestion/mapping is the wrapper).
2. **In-silico saturation mutagenesis** — enumerate all 20×L single mutants of a protein → predict ΔΔG /
   Δkcat / Δflux for each → the full **mutational landscape** (which residues are fragile, which are silent).
   ◐ (ΔΔG runs per-mutation now; batch + fold-mutant is the scale step).
3. **Generative design** — design a mutation to *achieve* a target (stabilize an enzyme, tune a rate).
   ○ (inverse of the chain; needs a generator + the chain as an oracle).

**AlphaFold fetching** sits under all three: `DDGPredictor.alphafold_pdb(uniprot)` resolves the current model
version via the EBI API and caches the PDB. For a mutant, the honest current approach uses the **WT** structure
+ a contact-number proxy at the site; the product upgrade is **folding the mutant sequence** (ESMFold, GPU).

---

## 5 · Validation spine — the actual product differentiator

The **recovery scorecard** (14 axes today): every capability is gated against known biology at a fixed bar,
and any change that breaks an axis fails the gate. This is what makes it a *product* rather than a demo —
**calibrated recovery + honest abstention**. A final product extends it with:
- per-answer **confidence** propagated through the chain (each node's uncertainty multiplies),
- **provenance** (which source/model produced each step),
- **abstention** when a link is weak — say "I don't know" instead of a confident wrong number.

Current axes (all PASS): algorithm-correctness · reprogramming · reversal-robustness · lens-calibration ·
IEM-mechanism · celltype-identity · tissue-communication · disease-target · measured-cause ·
cellgraph-capabilities · learned-GNN-beats-fixed · kcat-calibration-honesty · ΔΔG-stability · ec-flux.

---

## 6 · Infrastructure

- **Fetch layer:** AlphaFold EBI API, UniProt, GEO/FTP (perturbation), Drive/cloud (the assembled model),
  mygene (IDs). Everything cached (structures, embeddings, benchmark data).
- **Compute tiers:** **CPU** for numpy propagation / sklearn / LP solvers (cheap, reproducible, most of the
  validated core); **GPU** for torch GNN, protein-LMs, structure folding, structure-kcat.
- **Reproducibility:** seeded stochastic components, versioned validation JSONs, the scorecard as a regression
  gate. Deterministic engines reproduce to the digit; learned ones within a stated noise band.

---

## 7 · The honest status map

- **Built + validated (14 scorecard axes):** the causal/topological core, the learned GNN, ΔΔG, ec-flux,
  kcat-calibration honesty. This is a real, reproducible whole-cell *reasoning* engine.
- **One build away:** mutant-kcat fork, end-to-end IEM chain test, absolute in-cell kcat (needs PaxDb).
- **GPU-tier next:** ESM-2 node features, structure-based kcat (KcatNet), mutant folding, R-GCN/attention GNN,
  single-cell foundation models.
- **New track:** temporal trajectory prediction (the error ladder).
- **Deliberately not done** (honest negatives, don't re-try): cell-context kcat correction, CatPred
  recalibration — see `docs/KINETICS_CALIBRATION.md`.

The through-line of the whole product: **every layer predicts, every prediction carries confidence, and the
scorecard keeps all of it honest.** See `docs/FUTURE_IDEAS.md` for the sequenced build plan.
