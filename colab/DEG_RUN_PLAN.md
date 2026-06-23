# DEG + protein-family run — plan & status

Goal: bring the DEG cells (mtub, syn3a, mgen, S. aureus, S. pneumoniae) into the
two-wheel system, and add real protein-family info (ESM-2) for Stage-5 assembly.

## State (what's prepped vs blocked)

| piece | status | where |
|---|---|---|
| DEG essential labels (mtub 760, syn3a 383, mgen 382, S.aur 415, S.pne 197) | ✅ in `labels.csv` | — |
| DEG clade assignments | ✅ in `clade_splits.csv` + `labels_aug/clade_splits.csv` | — |
| OG reference DB (`data/gtdb/og_reps.faa`) | ✅ present | — |
| merge glue (`merge_deg_ogs.py`) | ✅ written + self-tested (synthetic) | sandbox |
| cache builder works w/ DEG | ✅ verified: 210,363 genes, all 5 cells in cache | sandbox |
| consumer parameterised (`AF_PREDS/AF_CACHE/AF_LABELS_DIR/ASSEMBLY_ORGS`) | ✅ + regression-tested | sandbox |
| one-click notebook (`run_deg.ipynb`) | ✅ written | Colab |
| **DEG proteomes (NCBI fetch)** | ❌ 403-blocked here | **Colab** |
| **mmseqs binary** | ❌ 403-blocked here | **Colab** |
| **DEG OG assignments** | ❌ needs proteomes + mmseqs | **Colab** |
| **transformer retrain (scores DEG)** | ❌ needs GPU | **Colab** |
| **ESM-2 protein families** | ❌ needs GPU | **Colab** |

The single missing **data** dependency is DEG OG assignments; the two GPU jobs
are the retrain and the ESM pass. Everything else is done and tested.

## Run order

### On Colab (open `colab/run_deg.ipynb`, GPU runtime, Run All)
1. clone + install mmseqs + transformers
2. `fetch_proteomes.py --mode deg --include_all`   (NCBI)
3. `assign_ogs.py` -> `colab/work/og_assignments.csv`
4. `merge_deg_ogs.py --assignments colab/work/og_assignments.csv`
5. `build_cache.py --labels_dir data/drive_import/labels_aug --out af_msa_cache_aug2.npz --include_orphans`
6. `af_torch2.py --cache af_msa_cache_aug2.npz --all_clades --tag _aug2`   **(GPU)**
7-8. `embed_proteins_esm.py` -> `esm_embeddings.npz`   **(GPU)**
9. bundle 4 artifacts -> download zip or `git push`

### Bring back to the sandbox
Drop into `outputs/orphan/`: `af_msa_cache_aug2.npz`, `af_torch_preds_aug2.npz`,
`esm_embeddings.npy`
Drop into `colab/work/`: `og_assignments.csv`, `deg_protein_index.json`
Then re-run `merge_deg_ogs.py --assignments colab/work/og_assignments.csv` in the
sandbox to regenerate `labels_aug/orthology_features.csv` from the REAL OGs.

### Back in the sandbox (one command)
```bash
AF_PREDS=af_torch_preds_aug2.npz \
AF_CACHE=af_msa_cache_aug2.npz \
AF_LABELS_DIR=data/drive_import/labels_aug \
ASSEMBLY_ORGS=mtub,syn3a,mgen,saur_NCTC8325_biotradis,spneT4 \
python3 colab/multi_org_assembly.py
```
(then `integrated_engine.py` / `assemble_cell.py` for per-organism reports.)

## Verify-before-trust (60-second checks after Colab)
- locus-tag join: DEG predictions should match `labels.csv` tags
  (`Rv####`, `JCVISYN3A_####`, `MG_###`, `SAOUHSC_#####`, `SP_####`).
- OG assignment rate per DEG org (expect 50-80%; low rate = strain mismatch).
- DEG ess-rate in predictions should track the label ess-rate
  (mtub ~0.21, syn3a ~0.84).

## Known risks
- Strain mismatch on `saur_NCTC8325_biotradis` (NCBI NCTC 8325 release) and
  `spneT4` (TIGR4) — the historic failure mode. Check join rate before training.
- `af_torch2.py` flag names (`--all_clades`, `--tag`) — confirm against its
  argparse; fall back to `af_torch.py --big` if needed.
- `embed_proteins_esm.py` expects sequences keyed a certain way; the family
  step (kNN gap-slot matching) is finished sandbox-side once embeddings land.
