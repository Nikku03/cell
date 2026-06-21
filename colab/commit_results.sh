#!/usr/bin/env bash
# Persist the run artifacts so results are reproducible from the repo (the
# consultant's #1 point). Stages the small/medium evidence files; the caller
# pushes. Big rebuildable inputs (af_msa_cache_aug.npz, labels_aug/*.csv) are
# intentionally NOT committed -- run_all.sh regenerates them.
set -e
cd "$(dirname "$0")/.."

git add -f outputs/orphan/*.json 2>/dev/null || true
git add -f outputs/orphan/*.png  2>/dev/null || true
git add -f outputs/orphan/af_torch_preds*.npz 2>/dev/null || true   # for offline rescue/diagnostics
# small artifact documenting how DEG orgs were assigned to clades
[ -f data/drive_import/labels_aug/clade_splits.csv ] && \
    git add -f data/drive_import/labels_aug/clade_splits.csv || true

git -c user.email="${GIT_EMAIL:-colab@local}" -c user.name="${GIT_NAME:-colab}" \
    commit -m "Colab run artifacts: result JSONs + figures + preds + DEG clade map" \
    || { echo "nothing to commit"; exit 0; }
echo "committed. Now push:  git push origin claude/vectorize-gex-propensity-NRqBW"
