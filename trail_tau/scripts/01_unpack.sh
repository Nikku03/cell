#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
rm -rf data/raw && mkdir -p data/raw
unzip -q -o data/dl/MOESM2_ESM.zip -d data/raw
