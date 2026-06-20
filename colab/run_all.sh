#!/usr/bin/env bash
# Self-contained end-to-end run for a fresh Colab A100:
#   mmseqs -> DEG fetch -> OG assign -> integrate -> build cache ->
#   train two-track model (af_torch2) -> GBM rescue -> summary.
# Resumable: skips the (slow) DEG build if the augmented cache already exists.
# Env: SEEDS (default 3), NCBI_API_KEY (optional, speeds the DEG fetch).
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# --- mmseqs (static binary, no conda) ---
if ! command -v mmseqs >/dev/null 2>&1 && [ ! -x /tmp/mmseqs/bin/mmseqs ]; then
  echo "== installing mmseqs2 =="
  wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tar.gz
  tar -xzf /tmp/mm.tar.gz -C /tmp
fi
export PATH="/tmp/mmseqs/bin:$PATH"
mmseqs version

AUG=outputs/orphan/af_msa_cache_aug.npz
if [ ! -f "$AUG" ]; then
  echo "== DEG fetch -> assign -> integrate -> build cache =="
  python colab/fetch_proteomes.py --mode deg
  python colab/assign_ogs.py --proteomes colab/work/proteomes \
         --reps data/gtdb/og_reps.faa --out colab/work/og_assignments.csv \
         --min_pident 30 --min_qcov 0.5
  python colab/integrate_deg.py --min_match 0.30 --min_pident 30 --min_assigned 500 \
         --min_ess_rate 0.05 --max_ess_rate 0.50
  python colab/build_cache.py --labels_dir data/drive_import/labels_aug --out "$AUG"
else
  echo "== augmented cache present, skipping DEG build =="
fi

echo "== train two-track model (af_torch2) =="
python colab/af_torch2.py --cache "$AUG" --labels_dir data/drive_import/labels_aug \
       --tag _v2 --all_clades --seeds "${SEEDS:-3}"

echo "== GBM rescue on top of v2 =="
python colab/smart_cooccur.py --cache "$AUG" --preds outputs/orphan/af_torch_preds_v2.npz \
       --labels_dir data/drive_import/labels_aug --tag _v2 --model gbm

echo "== summary =="
python - <<'PY'
import json
r = json.load(open("outputs/orphan/af_torch2_results_v2.json"))
s = json.load(open("outputs/orphan/smart_cooccur_results_v2.json"))["with_dnds"]["smart_combined"]
print("\n==== v2 two-track (DEG-augmented, all clades) ====")
print("pooled       :", r["pooled"])
print("smart+GBM    :", s)
print("\nper-clade (AUC | neCov@P90):")
for cl, d in sorted(r["per_clade"].items()):
    print(f"  {cl:<22} {d['auc']:.3f} | {d['ne_cov_p90']:.2f}  (n={d['n']})")
PY
echo "== done =="
