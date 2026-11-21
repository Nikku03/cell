# Improvement analysis — the whole model, critically

A critical audit of the ML model, the fixed methods, and the data, to raise AUC, add features, and reach
novel results. Findings are grouped by: (0) integrity checks, (1) validated quick wins, (2) real bugs,
(3) the big theme — underused data, (4) highest-leverage upgrades, (5) novel results, (6) scorecard soft spots.

## 0 · Integrity checks (verify before improving)

- **Link-prediction degree leakage — checked, negligible.** The `ppi` node-degree feature is computed on the
  full graph (incl. held-out test edges). Tested: leakage-free degree gives AUC 0.8271 vs 0.8261 current
  (within noise); zeroing the feature → 0.8229. The **degree-matched hard negatives neutralize it** (negatives
  share the positives' degree profile, so degree can't separate them). Reported AUCs are honest. ✓

## 1 · Validated quick wins (measured this pass, CPU)

The learned link-AUC ladder, all on the same leakage-free PPI benchmark:

| step | PPI link AUC |
|---|---|
| fixed SIGN propagation | 0.826 |
| learned GraphSAGE | 0.870 |
| learned R-GCN (per-relation) | 0.887 |
| **+ unused-data node features** (abund, ptm, complex-count, coexpr degree/strength, drugs, ppm, GO-count) | 0.889 (+0.003) |
| **+ fixed→ML embedding injection** (hybrid) | **0.898 (+0.011)** |

Both stack. The unused data and the fixed→ML injection are free, real gains → **adopt** (fold the 8 extra
features into `node_features`).

## 2 · Real bugs / correctness issues (found in audit)

- **`cellgraph.py` — advertised hub-suppression not applied.** `knockout_effect`'s docstring claims
  "÷√in-degree" specificity weighting; `self.indeg` is computed but never used in the ranking (`:334`, `:353-366`).
  Fix: apply it, or correct the doc.
- **`cellgraph.py` — hub flooding forces `hops=1`.** `perturb_downstream` (`:209-222`) has no per-step degree
  normalization, so multi-hop floods hubs and indirect effects are discarded. `disease_target_pipeline.py:88-93`
  already does the degree-normalized walk — move that into `perturb_downstream`.
- **`disease_to_reversal.py` — no Boolean cycle detection** (`:62-78`). Synchronous threshold-Boolean updates
  often land in 2-cycles; the returned "fixed point" is then phase-dependent, silently corrupting the
  disease/healthy attractor. Add oscillation detection or async updates.
- **Systematic "unknown sign → +1" activation bias** across all propagation (`cellgraph.py:203-205`,
  `disease_target_pipeline.py:54-56`, `disease_to_reversal.py:38-50`, `cause_inference.py:52-57`). The reg
  network is ~83% activating and under-annotated for repression, so propagation over-activates — hurts the
  0.809 perturbation-direction and every causal ranking. Fix: abstain on unknown sign, or infer it.
- **`cellgraph.py` — dead code** `Wt = W.T.tocsr()` (`:218`) unused; `W.T` recomputed each iteration.

## 3 · The big theme — the model ignores most of its own data

The GNN uses **5 of ~10 relations** and **11 scalar fields**; a large fraction of the assembled data feeds
zero predictions:

| data (in cell_complete.json) | size | used? | plug-in |
|---|---|---|---|
| `coexpr` | 16,374 × ~12 weighted (~200k edges) | ✗ | graph edges (weighted) |
| `complexes` / `gene2cplx` | 2,039 / 3,257 | ✗ | co-membership edges + node feature |
| `sl` (synthetic lethal) | 1,256 scored pairs | ✗ | edges + a new SL-prediction task |
| `nichenet` | 1,195 ligand→target | ✗ | ligand→target edges; ligand KO signature |
| `ppm` (abundance, copies) | 16,015 | ✗ | **ec-flux capacity `Vmax = kcat·[E]`** |
| `abund`, `ptm`, `cellcycle`, `go` | 16k / 8k / 153 / 16k | ✗ | node features / fine function labels |
| `emask` (200-cell-type mask) | 7,496 | localization only | **cell-type-specific graphs** |
| graph edge weights / signs | present | discarded (`w=1.0`) | weighted, signed propagation |
| `gf_perturb`, `ncrna` | 0 | empty | dead placeholders |

Two standouts: **`ppm` sits unused while `ecflux.py` assumes a blanket `sigma=0.5`** for capacity — the
"no absolute proteomics" wall we documented is *already in the file*; and **`build_adj` throws away all edge
weights/signs** (`w=1.0`), so the graph is far poorer than the data.

## 4 · Highest-leverage upgrades (ranked)

1. **Enrich the graph with unused edges** (coexpr + complex co-membership + sl as new R-GCN relations). ~200k
   weighted edges + 2k complexes + 1.2k SL pairs currently discarded. Directly lifts link/struct-fn/perturbation. CPU.
2. **Weighted + signed graph** — use coexpr correlations, sl scores, reg signs instead of `w=1.0`. CPU.
3. **`ppm`/`abund` → ec-flux capacity** — replace `sigma=0.5` with data (`Vmax = kcat·[E]`); makes the
   dominance %s data-backed and closes the documented wall. CPU (FBA).
4. **Fold the unused node features in** (validated +0.003) — abund, ppm, ptm, complex-count, cellcycle. CPU.
5. **ESM-2 protein-LM node features** (GPU) — the single biggest lever: real sequence biology the graph lacks.
   Plausibly lifts every head (link, function, perturbation) and unblocks the ΔΔG node.
6. **Strengthen ΔΔG** (thinnest scorecard margin, 0.405 vs 0.38) — add DSSP secondary structure / neighbor
   composition (CPU) or a ProteinMPNN log-odds feature (GPU). Also unblocks the IEM chain.
7. **Fix the sign bias + hub normalization** (§2) — improves perturb-direction and causal rankings. CPU.

## 5 · Novel results (existing pieces, newly combined)

- **Cell-type-specific link prediction** — mask the graph to a cell type via `emask` (200-type bitmask), build
  a per-type embedding → lineage-resolved interaction prediction. Genuinely new; CPU (data present).
- **Synthetic-lethal prediction** — held-out link prediction on `sl` (1,256 scored pairs); clinically relevant
  (PARP-style), a new scorecard axis. CPU.
- **Dark-proteome function** — `structural_embedding` (leakage-free) → predict `proc`/`go` for the ~5,000
  `dark` genes; validate against `darkfn` (independently built from Perturb-seq). CPU.
- **Complex-member completion** — predict missing `complexes` members using `gene2cplx` as held-out truth. CPU.

## 6 · Scorecard soft spots (where to harden)

- **`ddg_stability`** — thinnest margin (+0.025), ties DDGun, below DDGun3D/ACDC-NN. Most headroom → §4.6.
- **`disease_target_recovery`** — statistically flimsiest PASS: **n=2 diseases**, hand-curated targets, weak
  null. Expand to more diseases with independent ground truth.
- **`ecflux` essentiality recall 0.27** — a real weakness invisible to the gate (bar only checks precision-lift).
  Report/raise recall (medium constraints), or state the limit in the axis.
- **`measured_cause`** — the strongest (interventional) witness isn't wired into `detective_cause`
  (`measured_cause.integrate_as_witness` never called; `gf_perturb` empty).

## The path

Demonstrated already: **0.826 → 0.898** (learned + unused data + fixed→ML). The unused edges (§4.1–2) plus
ESM-2 (§4.5) are the realistic route to **~0.92+**, and §5 is where genuinely *new* biology (cell-type-specific
interactions, SL pairs, dark-gene function) comes from. Adopt order: §1 (free, now) → §2 bugs → §4.1–4 (data
we own) → §4.5–6 (GPU) → §5 (novel).
