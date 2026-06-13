# Tier 1 — Consolidate
## What's built, what's tested, what to run where

Status: **all six steps built; five tested end-to-end in sandbox; LOCO retraining + OGEE/STRING data fetching require Colab.**

---

## The six steps

| step | what it does | sandbox test | real run requires |
|------|--------------|:------------:|--------|
| **1.1a** | LOCO eval — hold each drug out, retrain Phase 2, score | **smoke PASS** (leak audit + metrics validated on mock) | Colab + Drive (feba.db + Phase 2 frame) |
| **1.1b** | Per-organism ranked GOLD/SILVER/ABSTAIN product output with plain-English "why" | **smoke PASS** (tiering + WHY field validated on mock) | Drive (Phase 2 per-fold predictions) |
| **1.1c** | Calibration (isotonic) of Phase 2 probabilities | uses existing `calibrate_and_threshold.py` (tested previously) | Drive predictions |
| **1.2** | OGEE external validation, methodology-filtered | **smoke PASS** (self-screen filter + agreement metrics validated) | Colab: download OGEE flat file |
| **1.3** | STRING degree kill-gate (review predicted DROP — cheap check first) | **smoke PASS** (DROP vs KEEP scenarios both classified correctly) | Colab: download STRING flat files |
| **1.4** | Bacitracin lead PI-handoff document | written, all referenced artifacts exist | none — sandbox final |

---

## Run everything in one command

**Sandbox (validates all logic before Colab):**
```bash
python scripts/tier1_master.py --smoke
```

**Colab (the real measurements):**
```bash
python scripts/tier1_master.py --real
```

Each step prints its own per-step status; the master prints a final pass/fail matrix.

---

## What each step's success looks like

### 1.1a LOCO — the headline measurement
- Holds out: cisplatin, bacitracin, nalidixic acid, fusidic acid, D-cycloserine, gentamicin.
- Positives: `fit < -3 AND |t| >= 3` (project standard; rejects looser thresholds).
- Reports: per-drug **recall@P30 + Precision@top-20 + AUPRC**, distribution + pooled macro.
- **Hard leak audit**: raises on any cross-contamination between train and the held-out compound.
- **Honest answer it produces:** does the kernel transfer to *unseen drugs*, or only within seen drug classes? If pooled recall@P30 < 0.2 → market only within known classes.

### 1.1b Per-organism product
- For each organism: ranked TSV with columns `tier, pred, locusId, compound, fit, t, strong_hit, og_cpd_hit_rate, og_cpd_n, why`.
- **GOLD** = `pred >= 0.80 AND atlas-corroborated (og_cpd_hit_rate >= 0.30 AND og_cpd_n >= 3)`.
- **SILVER** = `pred >= 0.60`.
- **ABSTAIN** = mid-range (0.10 < pred < 0.60).
- **NEGATIVE** = `pred <= 0.10` (confident no).
- WHY field: one-sentence mechanistic hypothesis a wet-lab can read.

### 1.1c Calibration
- Reuses `scripts/calibrate_and_threshold.py` (already validated).
- **Hardened**: calibration validation set must fully exclude the test organism (no peeking).

### 1.2 OGEE external validation
- Filters OGEE to entries from *different methodology* than our training data (excludes Keio / BERIL / RB-Tn-seq).
- Honest weakness flag: if <20% of OGEE survives the self-screen filter, marks as "weak / mostly self-referential" rather than over-claiming.
- **Honest answer:** does our ~0.78 strict gold-tier precision hold against independent curation, or revise down to ~0.65?

### 1.3 STRING degree kill-gate
- Computes degree from STRING links (cheap: pandas groupby).
- **Error-overlap test vs family_frac**: ratio > 2.0 → DROP, ratio > 1.5 → WEAK, ratio < 1.5 → KEEP.
- **Review prediction:** ratio is likely > 2.0 (conserved essentials are also hubs); the cheap test saves the work of computing betweenness.
- If DROP: betweenness is **not** computed; the strict cascade keeps 3 streams.

### 1.4 Bacitracin lead writeup
- Self-contained PI handoff: finding, evidence, mechanism, the single MIC-plate experiment, the counter-cases to falsify.

---

## Sandbox test results (already verified)

```
TIER 1 SUMMARY
======================================================================
  [PASS]  Tier 1.1a   LOCO evaluation
  [PASS]  Tier 1.1b   per-organism product
  [PASS]  Tier 1.2    OGEE external validation
  [PASS]  Tier 1.3    STRING degree kill-gate
  [PASS]  Tier 1.4    bacitracin writeup
  [PASS]  Tier 1.1c   calibration available
```

---

## When you get to Colab — order to run

1. **1.1a LOCO first** (the missing measurement; longest run; ~6 retraining passes of Phase 2).
2. **1.1b in parallel** (uses existing Phase 2 predictions; no retraining; ~minutes).
3. **1.1c calibration** (also uses existing predictions; ~minutes).
4. **1.2 OGEE** (1 Colab cell to download; runs in seconds).
5. **1.3 STRING degree** (download STRING for the 5 most-essential orgs; runs in seconds).
6. **1.4 ship** (no compute).

**Buffer week** after 1.1a to absorb the result — if the LOCO number says "kernel doesn't generalize to unseen drugs," the feature set needs rethinking before the rest ships.

---

## What Tier 1 produces (the artifact)

1. The **honest unseen-drug number** (the LOCO macro + per-drug distribution).
2. A **per-organism ranked target list** (GOLD/SILVER/ABSTAIN) with mechanism notes.
3. **Externally validated** strict cascade precision (against filtered OGEE).
4. A **decision** on STRING (keep or drop, with the numbers).
5. A **PI-grade writeup** of the bacitracin lead, ready for wet-lab handoff.

That's the consolidation. Shippable.
