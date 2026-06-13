# Tier 2 — The covariation model (research bet)

The AlphaFold-pairwise analog for essentiality: stop collapsing the
cross-organism essentiality table to `family_frac` (a scalar); feed the full
(organism × ortholog group) presence + essentiality matrix and learn a
**gene-gene coupling head** that captures the synthetic-lethal /
context-dependent residual conservation can't predict.

**Status:** Stage A (data) and Stage B (infrastructure) **complete and
sandbox-tested**. Stage C (training) is Colab+GPU work.

---

## What this targets

The measured ceiling of the conservation-collapsed approach:
- family_frac alone hits AUC 0.87 on strict essentiality
- Phase 2 hits recall@P30 0.193 on the de-novo / novel slice
- The 43% singleton tail is unreachable by any cross-organism collapsed feature

The covariation model targets the slice **where conservation can't help but
gene-gene structure can** — synthetic-lethal pairs, NOGD redundancy,
context-dependent essentiality. Forecast: +10-20% recall on the conditional
residual specifically, ~20-30% probability of working (the review's number).

---

## The four stages

| stage | what | runs | status |
|---|---|---|---|
| **A** | Build the matrices + leak-free schema + permutation control + coupling-recovery target | sandbox | **DONE, 7/7 tests PASS** |
| **B** | Infrastructure for the model + the two kill-gates (label permutation, coupling recovery) | sandbox | **DONE, 5/5 tests PASS** |
| **C** | Train minimal coupling model; run both kill-gates; gate decision | Colab+GPU | next |
| **D** | If gates pass: scale. If not: clean kill, document as null result | Colab+GPU | gated by C |

---

## Stage A — `scripts/tier2_stage_a_build_matrices.py`

Outputs (all in `outputs/tier2/`):
- `presence.parquet` — (organism, og_id, present=1) long form
- `essentiality.parquet` — (organism, og_id, ess) long form, ess in {0,1}
- `clade_holdout.json` — 34 clade → list of orgs (the leak-free splits)
- `synthetic_lethal_pairs.csv` — known SL pairs (empty in sandbox due to the
  disjoint-data wall; recoverable via feba.db on Colab)
- `operon_adjacent_pairs.csv` — **13,693 OG-pair coupling targets** with
  cross-organism support, the backup recovery test
- `matrix_stats.json`, `per_org_coverage.csv`, `per_og_coverage.csv` — diagnostics

Measured:
- **48 organisms × 17,222 OGs**
- **132,957 essentiality calls** (31,976 essential / 100,981 non-essential)
- **16% matrix density**
- 6,218 OGs labeled in ≥5 orgs, 3,496 in ≥10 orgs

Tests in-script (7/7 PASS):
1. Matrix shape (48+ orgs, thousands of OGs)
2. Per-organism label density (every org ≥100 labels)
3. Essentiality values valid (0/1 only)
4. Permutation preserves per-org marginals
5. Clade map is a valid partition
6. Presence/essentiality keys consistent
7. At least one coupling-recovery set non-empty (operon pairs: 13,693)

---

## Stage B — `scripts/tier2_stage_b_coupling_model.py`

Infrastructure + the two hard kill-gates. Smoke-tested on real Stage A data:

**T1 — clade holdout discipline.** Held-out clade (e.g. Shewanella) fully
absent from training; test set fully contained in held-out clade. **PASS**.

**T2 — family_frac baseline** on Shewanella held out:
- AUPRC **0.712**, R@P30 **0.958** on 8,315 test rows
- **This is the bar Stage C must beat by ≥10% relative.**

**T3 — Gate 1 (label permutation) infrastructure.**
- Permute labels within each organism, recompute family_frac, score
- Result: permuted/real AUPRC ratio = **0.264** → signal collapses cleanly
- Gate 1 ready: the real model must show even sharper collapse

**T4 — Gate 2 (coupling recovery) infrastructure.**
- Naive coupling proxy = per-OG essentiality correlation across organisms
- Tested on **200 operon-adjacent pairs** vs 1,000 random pairs
- Result: **operon pairs co-vary 5.40× more than random**
- The biological signal is in the data; the model just needs to learn it

**T5 — PyTorch availability** — absent in sandbox (expected); Stage C is Colab.

---

## Stage C plan (next; Colab + GPU)

**Minimal architecture (target <200K params):**
1. Per-OG learned embedding (d=64).
2. Organism representation = mean of present-OG embeddings + a per-org
   bias for label-rate.
3. Pairwise coupling head: for each (org, og), score depends on:
   - organism_repr · og_embedding (the marginal — what family_frac captures)
   - a learned coupling matrix `C[og, og']` summed over OGs essential in
     phylogenetically similar organisms (the genuine new signal)

**Training discipline:**
- Leave-one-clade-out splits (Stage A's `clade_holdout.json`).
- Two hard gates **both required** before any scaling:
  - **Gate 1:** train on label-permuted essentiality; held-out R@P30 must
    drop ≥80% relative.
  - **Gate 2:** trained coupling matrix C[a,b] must score operon-adjacent
    pairs higher than random by ≥2× (the naive proxy already gets 5.4×;
    the model should preserve or improve on this).
- **Performance gate:** held-out R@P30 must beat family_frac by ≥10%
  relative on the same held-out test set.

**Kill-fast policy:**
- Each gate runs on a small prototype (3 clades, 200K params, 10 epochs).
- If any gate fails on the prototype: kill cleanly, write up as null result,
  return to Tier 1.
- No scaling permitted until all three gates pass on the prototype.

---

## What success looks like

| measurement | gate | success |
|---|---|---|
| held-out R@P30 vs family_frac | performance | ≥10% relative lift |
| permuted-labels R@P30 | Gate 1 | drops ≥80% relative |
| coupling matrix vs operon pairs | Gate 2 | ≥2× random-pair lift |
| **all three pass** | **proceed to Stage D scaling** | research direction validated |
| any fail | clean kill | write null result; return to Tier 1 |

---

## What this is NOT

- Not a serial mechanistic chain.
- Not a single-classifier-from-features.
- Not a replacement for Phase 2 — Phase 2 predicts conditional vulnerability;
  this predicts strict essentiality from gene-content + coupling structure.
- Not a guarantee — the review puts P(meaningful gain) at ~20-30%. The
  kill-gates exist exactly to prevent a months-long time sink on a model
  that's secretly fitting phylogeny.

---

## Sandbox tests passed (today)

```
TIER 2 STAGE A
ALL TESTS PASS (7 checks)

TIER 2 STAGE B
T1 clade-holdout: PASS
T2 family_frac baseline: AUPRC 0.712, R@P30 0.958 (the bar)
T3 permutation control: ratio 0.264 (signal collapses as expected)
T4 coupling recovery infrastructure: operon pairs co-vary 5.40x random
T5 PyTorch in sandbox: absent (expected; Stage C is Colab)

=== STAGE B SMOKE PASS ===
```

The data carries the signal we need. The infrastructure to train and
validate the model is built. Stage C scaffolding (PyTorch model + training
loop + gate runners) is the next step, Colab-side.
