#!/usr/bin/env bash
# Stage the curated function text used by scripts/ortholog_function_statements.py.
#
# Reviewed (Swiss-Prot) entries for the seven model organisms whose orthologs
# the upstream table names, keyed by NCBI GeneID so they join straight onto
# human_ortholog_pairs_panel.tsv.gz. Reviewed only: cc_function is a curator-
# written field and TrEMBL entries almost never carry it.
#
# ~25 MB into data_cache/ortholog_function/ (gitignored). Safe to re-run.
#
# The second half of the data -- UniProt records for the bacterial/archaeal/
# protist hits from the deep-homology search -- is fetched by
#   python3 scripts/ortholog_function_statements.py --fetch-deep-annotations
# because the accession list comes from that search's output.
set -uo pipefail

CACHE="${1:-$(dirname "$0")/../data_cache/ortholog_function}"
mkdir -p "$CACHE"
cd "$CACHE" || exit 1

FIELDS="accession,xref_geneid,protein_name,cc_function,go_id,ec,cc_catalytic_activity,cc_subunit"

fetch () {  # name taxid
  local f="uniprot_$1.tsv"
  [ -s "$f" ] && { echo "SKIP $f"; return 0; }
  for attempt in 1 2 3; do
    if curl -sS --max-time 600 --compressed -G "https://rest.uniprot.org/uniprotkb/stream" \
        --data-urlencode "query=(organism_id:$2) AND (reviewed:true)" \
        --data-urlencode "fields=$FIELDS" --data-urlencode "format=tsv" -o "$f.part" \
        && [ -s "$f.part" ]; then
      mv "$f.part" "$f"
      echo "OK   $f ($(($(wc -l < "$f") - 1)) entries)"
      return 0
    fi
    echo "RETRY($attempt) $1"; sleep $((2 ** attempt))
  done
  echo "FAIL $1"; return 1
}

fetch mouse              10090
fetch rat                10116
fetch zebrafish          7955
fetch xenopus_tropicalis 8364
fetch fly                7227
fetch worm               6239
fetch yeast              559292

echo "--- staged ---"
du -sh "$CACHE"
