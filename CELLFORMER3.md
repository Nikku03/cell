# Cellformer v3 — the complete architecture, every dataset it needs, and the order things get tested

This file exists because the previous two attempts were built the right way round and trained on the wrong
object. `CELLFORMER.md` specified an AlphaFold-shaped model; `CELLFORMER2.md` corrected its metric. Neither
noticed that **every transformer in the arc trained against a target that is 96.5% censored**.

That is the finding this document starts from, and it reframes everything downstream.

---

## 0. What was actually wrong, measured

The `nlz_*` benches store a median of **250 genes per knockout** out of 7,223.

| | K562 bench |
|---|---:|
| matrix cells | 10,112,200 |
| values actually recorded | **349,999 (3.5%)** |
| the rest | zeros that mean **NOT RECORDED**, consumed as real zeros |

`eval_harness` writes recorded values into a zeros matrix, takes `A = |M|`, and thresholds `mover = A >= 1.0`.
So the retrieval target — cosine between tide-removed response profiles — was computed between vectors that
are 96.5% structural zeros. Two knockouts resemble each other partly because both are mostly zeros. **The
0.607 "oracle" is a ceiling of the truncation, not of the biology.**

`colab/dense_response.py` rebuilds the same experiment from the cell-level deposits:

| | benches | dense |
|---|---:|---:|
| density | 3.5% | **100.0%** |
| genes jointly observed in all 4 lines, per perturbation | median **3** | median **6,550** |
| usable cross-cell observations | ~9,700 | **11,547,208** |

A factor of **1,186**. Nothing about the architecture changes this; the target does.

**Consequence for every published number in the arc.** Internal comparisons on the same target still stand —
random-partner, wrong-knockout and shuffled-sign controls are like-for-like. What does *not* stand: absolute
recalls, the oracle, and the conclusion that the gap to it is representation-limited. It is limited by the
bench format.

---

## 1. Architecture — parts, structures, components

Six blocks. Each is separately ablatable and separately testable, and nothing enters the stack before it has
cleared its own test.

### Block A — Entity layer (typed tokens)

| component | content | status |
|---|---|---|
| target token | the perturbed gene | **built, confirmed** |
| typed neighbour tokens | co-dependency, complex, PPI, signed perturbational | **built, confirmed** |
| relation type embedding | 5 types | built |
| relation sign + evidence strength | sign, \|z\| as confidence | **built, four times not detected** |
| perturbation strength | cells, UMI, knockdown efficacy | built |
| baseline state per token | that cell's expression of that gene | built, **ignored by the model** |

### Block B — Encoder

Set-transformer over the token set. Multi-head attention, relation-specific additive attention bias,
mean-pool. **Measured: the bias is not detected (+0.0034), and message-passing on real wiring is
indistinguishable from shuffled wiring of the same density (+0.005 ± 0.008).** A knockout is well described
as a *bag* of its typed partners; topology within the bag is latent in the node features.

### Block C — Context conditioning

Where the model is told which cell it is in. Four modes, all measured:
cell-line token; global baseline vector; per-token gene context; context-gated neighbours.
**Measured: all four are ignored.** The counterfactual swap moves recall by −0.0007 to +0.0004.

### Block D — Objective

The reason C is ignored. With the same perturbation present in every training cell and a conserved response
dominating the loss, `Ŷ[c,p,g] ≈ shared response of p` is optimal and the cell can be dropped.

    Y[c,p,g] = μ[p,g] + δ[c,p,g]

Shared branch estimates μ, is **frozen**, and a residual branch is trained on δ alone against a
training-cell-only μ̂. Staged rather than joint — with four contexts there is not enough data to police a
stop-gradient inside one loss.

### Block E — Heads

| head | supervised? | measured |
|---|---|---|
| specific-mover ranking | yes | 0.4301 |
| residual effect (tide + learned) | yes | 0.3716, above the 0.2910 floor |
| response probability | yes | AUC 0.912 vs per-gene marginal 0.882 → **+0.030** |
| direction | yes | AUC 0.975 vs per-gene marginal 0.949 → **+0.026** |
| uncertainty | yes | ρ +0.603 vs total-movement baseline +0.426 → **+0.177** |

**Every head must be reported against its own marginal.** Direction at 0.975 looks like the strongest result
in the project; a marginal using no knockout information reaches 0.949 of it.

### Block F — Explicit non-learned tide

Per-gene training-set mean added *before* the loss, so predicting the tide earns zero. **This works** — it
prevented the collapse `neural_ko` documented, where a regression net landed below the tide floor.

---

## 2. Datasets — needed, available, curated

### Available raw (on disk)

| dataset | size | what it is | used? |
|---|---:|---|---|
| `gwps.h5ad` | 0.37 GB | K562 Perturb-seq, dense pseudobulk 11,258 × 8,248 | partly |
| `rpe1.h5ad` | 1.24 GB | RPE1, 247,914 cells | now |
| `nadig_jurkat.h5ad` | 1.29 GB | Jurkat, 262,956 cells | now |
| `nadig_hepg2.h5ad` | 0.85 GB | HepG2, 145,473 cells | now |
| `hct116.h5ad` | 1.17 GB | HCT116 | no |
| `frangieh.h5ad` | 1.46 GB | melanoma, 3 conditions | no |
| `sciplex.h5ad` | 2.53 GB | chemical perturbation | no |
| `shifrut.h5ad` | 0.87 GB | primary T cells | no |
| `papalexi.h5ad` | 0.15 GB | ECCITE-seq | no |
| `CRISPRGeneEffect.csv` | 0.42 GB | DepMap, 1,150 lines | yes |
| `cCREs.bed`, `hg38.2bit` | 0.90 GB | SCREEN elements, genome | no |
| `HumanGEM.xml`, `RECON1.xml` | 0.06 GB | metabolic models | no |
| `signor.tsv`, `collectri.tsv` | 0.02 GB | signed regulation | yes |

### Curated by this project (556 artifacts; the load-bearing ones)

| artifact | content | verdict |
|---|---|---|
| `dense_response.npz` | **4 lines × 1,397 perturbations × 5,770 genes, 100% dense** | **unblocked, 1,186× more data** |
| `baseline_state.json.gz` | per-line baseline RNA, 6,550 shared genes | 4 distinct states, r 0.805–0.884 |
| `entity_registry` | 172,539 typed entities, declared-rate joins | backbone |
| `causal_reg` | 537k signed perturbational edges | **+15.0 sign transfer, audit-confirmed** |
| `bound_causal` | 392 TFs, DIRECT/INDIRECT tiers | directness is a real criterion |
| `reliable_edges` | power-weighted edge selection | **+0.0174 after 39% leakage removed** |
| `measurement_power` | reliability ceiling | **R² 0.477 — the real ceiling** |
| `depmap_codep` | co-dependency on viability | viability yes, transcription no |
| `k562_protein` | cell-matched protein | mRNA explains 15.9% |
| `contradictions` | 18 entries + 8 method errors | what must not be re-derived |
| `environment` | 126 dataset × variable slots | 42% recorded |

### Needed but absent

ATAC/chromatin for these four lines; protein for RPE1/HepG2/Jurkat; any timepoint other than one;
**more independent cell contexts** — the binding limit is 4.

---

## 3. Verify, then re-verify

Before any training, three checks that have each already caught a real defect:

1. **Density and censoring** — a zero must be distinguishable from unrecorded. Caught: 96.5% censoring.
2. **Provenance on the axis** — no processing difference may lie along the axis measured. Caught: K562 is
   the only line not re-derived; cross-line variance correlation makes it inspectable.
3. **Feature canaries** — a feature that is silently constant passes every control. Caught: essentiality
   read from a per-gene z-scored matrix, giving 0 genes essential in >50% of lines.

Re-verification is not re-reading the code. It is: recompute the number by a second route and require the
two to agree, or state the disagreement.

---

## 4. Test order — one at a time, then the stack

Each stage declares its threshold **before** the numbers, and reports its minimum detectable increment.

| # | test | question | gate to pass |
|---|---|---|---|
| 1 | dense retrieval baseline | does the arc's one confirmed win survive an untruncated target? | real-vs-random partners still ≫ 0 |
| 2 | dense ceiling | what is the oracle on dense data? | report; 0.607 is void |
| 3 | context, dense | is context still ignored at 1,186× the data? | swap gap > 0 |
| 4 | contrast objective | does forcing the residual make context readable? | all 7 predeclared criteria |
| 5 | heads vs marginals | does any head beat its own per-gene marginal? | delta > MDE |
| 6 | cross-cell CASP | train on 3 lines, predict the 4th from baseline state only | beat that line's own tide |
| 7 | full stack | everything that passed, together | beat the best single component |

**Nothing joins the stack that failed its own gate.** On current evidence that means the sign channel and the
relation bias are out unless the dense target revives them.

---

## 5. What this cannot become

Four independent cell contexts. Genes and perturbations buy precision *inside* them; they do not create new
cell states. This can show whether context is readable and can fit a four-context correction. **It cannot
establish a universal cell-state encoder**, and a positive result is a reason to acquire more cell lines,
not a claim of biological generality.
