#!/usr/bin/env bash
# Phase 0 acquisition. Every reported hash comes from this script.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dl data/raw
PDF='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41540-026-00782-4/MediaObjects/41540_2026_782_MOESM1_ESM.pdf'
ZIP='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41540-026-00782-4/MediaObjects/41540_2026_782_MOESM2_ESM.zip'
curl -sSL --retry 4 --retry-delay 2 -o data/dl/MOESM1_ESM.pdf "$PDF"
curl -sSL --retry 4 --retry-delay 2 -o data/dl/MOESM2_ESM.zip "$ZIP"
echo "--- SHA-256 ---"
sha256sum data/dl/MOESM1_ESM.pdf data/dl/MOESM2_ESM.zip
echo "--- sizes ---"
ls -l data/dl/
echo "--- file types ---"
file data/dl/MOESM1_ESM.pdf data/dl/MOESM2_ESM.zip
