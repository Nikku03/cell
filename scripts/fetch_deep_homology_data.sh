#!/usr/bin/env bash
# Stage the proteomes used by scripts/deep_homology_dark_genes.py.
#
# 26 reference proteomes spanning bacteria, archaea, plants, protists and the
# basal metazoans/invertebrates that NCBI's vertebrate-only ortholog set never
# sees, plus the human reference proteome for the reciprocal search.
# ~180 MB into data_cache/deep_homology/ (gitignored). Safe to re-run.
set -uo pipefail

CACHE="${1:-$(dirname "$0")/../data_cache/deep_homology}"
mkdir -p "$CACHE/proteomes"
PANEL="$CACHE/panel.json"

if [ ! -s "$PANEL" ]; then
  echo "missing $PANEL — run scripts/deep_homology_dark_genes.py --resolve-panel first" >&2
  exit 1
fi

get () {  # upid dest
  [ -s "$2" ] && { echo "SKIP $(basename "$2")"; return 0; }
  for attempt in 1 2 3 4; do
    if curl -sS --max-time 900 --compressed -G "https://rest.uniprot.org/uniprotkb/stream" \
        --data-urlencode "query=(proteome:$1)" --data-urlencode "format=fasta" -o "$2.part" \
        && [ -s "$2.part" ]; then
      mv "$2.part" "$2"
      echo "OK   $(basename "$2") ($(grep -c '^>' "$2") sequences)"
      return 0
    fi
    echo "RETRY($attempt) $1"; sleep $((2 ** attempt))
  done
  echo "FAIL $1"; return 1
}

python3 -c "import json,sys;[print(p['upid'], p['label'].replace(' ','_')) for p in json.load(open('$PANEL'))]" \
| while read -r upid label; do
  get "$upid" "$CACHE/proteomes/${label}.fasta"
done

# Human proteome: the reciprocal-search target that decides whether a hit is a
# genuine ortholog or just the nearest member of a big family. Reviewed entries
# only -- one canonical protein per gene. The unfiltered proteome download is
# 147k records once isoforms and TrEMBL are included, which makes the
# reciprocal search seven times slower without making it any more correct.
if [ ! -s "$CACHE/human_reference_reviewed.fasta" ]; then
  curl -sS --max-time 900 --compressed -G "https://rest.uniprot.org/uniprotkb/stream" \
    --data-urlencode "query=(proteome:UP000005640) AND (reviewed:true)" \
    --data-urlencode "format=fasta" -o "$CACHE/human_reference_reviewed.fasta" \
    && echo "OK   human_reference_reviewed.fasta ($(grep -c '^>' "$CACHE/human_reference_reviewed.fasta") sequences)"
else
  echo "SKIP human_reference_reviewed.fasta"
fi

echo "--- staged ---"
du -sh "$CACHE"
