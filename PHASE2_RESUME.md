# PHASE 2 RESUME — pick up here next session

Last updated mid-bake-off. Everything below is committed to git or saved on
Drive. Nothing is lost on disconnect.

## One-line status
Phase 2 frame is BUILT (Drive), fast bake-off RAN once (XGBoost beats additive
on 3/4 folds), litmus path bug FIXED but the corrected litmus has NOT been
read yet. Next action: run the litmus re-read (command below) and read the
bacitracin cluster ranks.

## State of the world

### Git (branch `claude/vectorize-gex-propensity-NRqBW`, all pushed)
- `PHASE2_DESIGN.md`      — design + the 4 measured feasibility numbers
- `PHASE2_RESUME.md`      — this file
- `scripts/phase2_feasibility.py`   — Q1-Q4 gating numbers (DONE, ran)
- `scripts/build_phase2_frame.py`   — sharded, resumable frame builder (DONE, ran)
- `scripts/train_phase2_bakeoff.py` — additive vs XGBoost bake-off (latest commit ae8d110+)
- `scripts/verify_bacitracin_lead.py` — phase 1 litmus verifier (DONE)
- `outputs/atlas_phase1_surprise.md`  — phase 1 VALIDATED finding

### Drive (`/content/drive/MyDrive/cell_count_dynamics/multiorg/`)
- `fitness_browser/feba.db`       — 7.3 GB source (also copy to /content/feba.db each session)
- `phase2/frame/<org>.parquet`    — 48 downsampled shards (~86 MB) DONE
- `phase2/agg/<org>_{gene,cpd}.parquet` — 96 full-data suff-stat files DONE
- `phase2/_build_manifest.json`   — 27.4M rows, 231,515 pos, base rate 0.0084
- `phase2/eval/loo-org/preds_*.parquet` — fast-bakeoff predictions (4 orgs x 2 archs)
- `phase2/models/loo-org/xgb__*.json`   — saved XGBoost models

## Measured numbers so far (the honest scoreboard)

Feasibility (against feba.db):
- Noise ceiling: continuous-fit tail corr 0.34 (DEAD target) vs strong-hit
  reproducibility 0.62-0.78 (LIVE target -> we predict strong_hit binary)
- Interaction = 0.64 of variance; additive misses bacitracin litmus by 3.24 fit
- LOCO feasible: 97% compounds have CAS, 153 in >=3 orgs
- Split by COMPOUND not experiment (1815/2425 cells have replicates)

Fast bake-off (loo-org-fast, 4 held-out orgs; AUPRC on 10:1 DOWNSAMPLED test,
so AUPRC ~10x optimistic — recall@P is honest/base-rate-invariant):
| org            | additive AUPRC | xgb AUPRC | xgb R@P30 | xgb R@P50 |
|----------------|----------------|-----------|-----------|-----------|
| SB2B (litmus)  | (cached)       | 0.718     | 0.882     | 0.741     |
| Keio           | 0.649          | 0.752     | 0.922     | 0.785     |
| Phaeo          | 0.590          | 0.587     | 0.784     | 0.678     |
| pseudo5_N2C3_1 | 0.739          | 0.802     | 0.902     | 0.819     |

Verdict so far: XGBoost > additive on 3/4 (Phaeo, distant marine clade, is the
tie). Trees capture interaction. Litmus result still PENDING.

## NEXT ACTION (exact command)

```python
import subprocess, shutil
from pathlib import Path
REPO = Path("/content/cell")
PHASE2 = Path("/content/drive/MyDrive/cell_count_dynamics/multiorg/phase2")

# 0. clone/refresh repo + copy db if runtime is fresh
import os
if not REPO.exists():
    subprocess.run(["git","clone","-b","claude/vectorize-gex-propensity-NRqBW",
                    "https://github.com/Nikku03/cell.git", str(REPO)], check=True)
else:
    subprocess.run(["git","-C",str(REPO),"pull","origin",
                    "claude/vectorize-gex-propensity-NRqBW"], check=True)
if not Path("/content/feba.db").exists():
    subprocess.run(["cp", str(PHASE2.parent/"fitness_browser"/"feba.db"),
                    "/content/feba.db"], check=True)

# 1. migrate any old-path predictions (one-time, safe to re-run)
old = PHASE2/"eval"/"loo-org-fast"; new = PHASE2/"eval"/"loo-org"
new.mkdir(parents=True, exist_ok=True)
for p in (old.glob("preds_*.parquet") if old.exists() else []):
    t = new/p.name
    if not t.exists(): shutil.move(str(p), str(t))

# 2. re-run fast bakeoff (recomputes metrics from cache in seconds; runs litmus)
proc = subprocess.run(["python","-u", str(REPO/"scripts"/"train_phase2_bakeoff.py"),
                       "--mode","loo-org-fast"])
print("RC:", proc.returncode)
```

Read the `=== bacitracin litmus ===` block:
- cluster ranks: envZ/SB2B, ompR/SB2B, pspB/SB2B percentile. Top 1% (rank
  <~1100 of 113k) = pass; top 0.1% = strong pass.
- top-10 bacitracin predictions per org: sanity that the model picks the
  right KIND of gene under bacitracin.

## After the litmus, the remaining phase 2 plan
1. If litmus passes: run `--mode loo-org-full` (43 folds; the publishable
   LOO-organism headline). ~1-2 hr on L4, resumable per fold.
2. Add LOCO mode (leave-one-compound-out incl. bacitracin) — needs a small
   code addition (loco fold loop already sketched in design; harness has the
   compound-split plumbing via compute_additive_effects held_out_cpds).
3. Add chem features from CAS (PubChem -> MoA / fingerprint) to lift LOCO.
4. Calibrate + abstention (AlphaFold paradigm) on the winning model.
5. Decide FiLM only if XGBoost plateaus below the 0.62-0.78 ceiling.

## Gotchas learned (don't re-hit these)
- Sandbox/Colab repo resets to old commit on reconnect -> always `git pull`/reset first.
- feba.db stores blanks as '' not NULL; numeric cols need pd.to_numeric(coerce).
- `aerobic` is the STRING 'aerobic'/'anaerobic' -> map to 1.0/0.0.
- Concat-all-30M-rows OOMs -> per-organism sharding is mandatory.
- AUPRC here is on downsampled negs (10:1) -> ~10x optimistic; quote recall@P.
- PAT push: set-url with token, push, scrub back to clean URL.
