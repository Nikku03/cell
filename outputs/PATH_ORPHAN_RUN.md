# PATH-ORPHAN — detecting conditional & orphan essentials via dN/dS + structure

**Question this run answers:** for the genes we *cannot* currently detect —
rogue essentials (essential but conservation < 0.1) — do evolutionary selection
(dN/dS) and structural homology (AFDB + Foldseek), plus cofitness, separate the
true essentials from the non-essentials *after* conservation is already in the
model? If yes at ≥10% rogue-zone recall lift, orphans are tractable for $0. If
no, orphans are genuinely opaque to anything but interventional data.

## Organism — `beril_RalstoniaGMI1000` (decided, not Keio)

Keio was the first pick and is **wrong for the sandbox**: its label namespace
(numeric ids like `16662`) joins the cached genome at **0%** — it would have
produced an empty experiment on Colab (the disjoint-data wall, live). GMI1000 is
the pick instead, all sandbox-verified by `scripts/orphan_bridge.py`:

| requirement | GMI1000 | Keio |
|---|---|---|
| label→protein-sequence join | **99%** (4,394/4,445) | **0%** |
| independent cross-grade screen | **DEG1057** (R. solanacearum) | DEG1018/19 (namespace mismatch) |
| Paper-1 tie-in | **the κ=0.18 worst-reproducibility org** | reference org |
| rogue essentials isolated | **140** (essential & cons<0.1) | — |

Only gap: the RefSeq GFF carries no UniProtKB xref, so AFDB needs one
`WP_→UniProt` idmapping call (`uniprot_request_*.txt`, Phase A, network).

## Caching contract (makes runtime-switching safe)

Every step writes a parquet/file under `outputs/orphan/` (Colab: also copy to
`/content/drive/MyDrive/path_orphan/`) and is **skip-if-exists** (`--force` to
rebuild). Switching Colab runtime wipes `/content` but Drive persists, so
finished steps survive a CPU↔GPU switch and a full re-run is instant.

## Phase split — GPU is on for ONE step only

| phase | runtime | steps | GPU |
|---|---|---|---|
| **A** | CPU (free) | 0 bridge ✓, 1 CES label, 2 baseline, 3 dN/dS, 4 Foldseek-CPU, 5 cofit, 7 DEG | $0 |
| **B** | A100/L4 (only this) | ESM-2 embeddings (orphan-tail feature; deep orphans where dN/dS can't reach) | ~20 min A100 / ~8 min Blackwell |
| **C** | CPU (free) | 6 model + rogue-zone gate, 8 permutation null | $0 |

Total paid GPU for the whole experiment ≈ **15–30 min** (A100). Core experiment
is 0-GPU; ESM is the only GPU step and is optional-but-recommended.

## Steps

| # | script | phase | status | output |
|---|---|---|---|---|
| 0 | `orphan_bridge.py` | A | **DONE — smoke+real PASS, 99% join** | `bridge_*.parquet`, `proteins_*.faa`, `uniprot_request_*.txt` |
| 1 | `orphan_ces_label.py` (TODO) | A | reuse `ces_consensus.py` logic, keyed to org | `ces_label_*.parquet` (y_ess + total_weight) |
| 2 | `orphan_baseline.py` (TODO) | A | family_frac leak-free xgb — **the bar** | rogue-zone R@P30 baseline |
| 3 | `orphan_dnds.py` (TODO) | A | per-OG MAFFT+FastTree+PAL2NAL → HyPhy aBSREL; OGs with ≥3 orthologs | `dnds_*.parquet` |
| 4 | `orphan_foldseek.py` (TODO) | A | AFDB pull (via uniprot_request) → Foldseek vs AFDB-cluster reps; **name-grade hits** (STRICT=characterized vs LOOSE=orphan↔orphan) | `foldhit_*.parquet` |
| 5 | `orphan_cofit.py` (TODO) | A | feba.db `Cofit`: median essentiality of top-10 cofit partners | `cofit_*.parquet` |
| B | `orphan_esm.py` (TODO) | **B** | ESM-2 650M on `proteins_*.faa` (reuse `build_esm2_embeddings.py`) | `esm_*.parquet` |
| 6 | `orphan_model.py` (TODO) | C | xgb: family_frac + dN/dS + fold_hit_named + cofit + ESM, leak-free | rogue-zone metrics |
| 7 | `orphan_deg_grade.py` (TODO) | A/C | grade rogue-zone wins vs **DEG1057** (independent) | precision vs non-RB-TnSeq screen |
| 8 | `orphan_null.py` (TODO) | C | permute y_ess (fixed features), refit, 100×; real lift ≫ permuted | null distribution |

## Kill-gate (the only thing that matters)

Each new feature must lift the **rogue zone (cons<0.1)**, not overall AUPRC —
six prior features were absorbed by family_frac when judged on overall metrics.

| comparison (rogue-zone R@P=0.3) | gate |
|---|---|
| family_frac only | the bar (step 2) |
| + dN/dS | report Δ |
| + Foldseek named-hit | report Δ |
| + cofit + ESM | report Δ |
| **all** | **≥ +10% relative over the bar → orphans tractable; else clean kill** |

Plus: (a) univariate sanity — dN/dS≪1 must enrich for essentials before
modeling; (b) name-graded fold hits — only characterized-fold hits count as
rescue; (c) DEG1057 cross-grade — rogue wins must hold against an independent
screen, not just the RB-TnSeq we trained on; (d) permutation null — real lift
must exceed label-shuffled lift.

## Honest limits (decided up front)

- **dN/dS floor:** needs ≥3 orthologs; true singletons (`family_n_orgs≤2`) get
  no estimate — they rely on structure/ESM alone.
- **Foldseek:** pLDDT≥70 mean only (bad predictions fake remote homology);
  hits to other DUFs/hypotheticals are not rescue.
- **Genuinely de-novo orphans** (no relatives, no fold hit) remain opaque to all
  evolutionary signal — only a screen decides. Evolution covers most, not all.

## Exact commands

```bash
# ---- Phase A (CPU; free; resumable) ----
python scripts/orphan_bridge.py --real          # DONE (cached)
python scripts/orphan_ces_label.py --real       # TODO
python scripts/orphan_baseline.py --real
python scripts/orphan_dnds.py --real            # MAFFT/FastTree/PAL2NAL/HyPhy
python scripts/orphan_foldseek.py --real        # needs AFDB-cluster DB (~25 GB, once)
python scripts/orphan_cofit.py --real           # needs feba.db
python scripts/orphan_deg_grade.py --real
# copy outputs/orphan/* to Drive

# ---- switch runtime to A100 ----
python scripts/orphan_esm.py --real             # ONLY this on GPU (~20 min)
# copy esm_*.parquet to Drive; switch runtime back to CPU

# ---- Phase C (CPU; free) ----
python scripts/orphan_model.py --real           # rogue-zone gate
python scripts/orphan_null.py --real            # permutation null
```

## Status

Step 0 built, tested (smoke + real), gate-passed at 99% join. The 140 rogue
essentials are isolated and the protein FASTA + UniProt request are emitted.
Steps 1–8 are scoped above and are the next build. Nothing here needs a GPU
except the one ESM pass.
