#!/usr/bin/env bash
# Stage the bulk ortholog/annotation dumps used by scripts/human_gene_orthologs.py.
#
# ~3 GB into data_cache/human_orthologs/ (gitignored). Takes ~4 min on a
# fast link. Safe to re-run: existing complete files are skipped.
set -uo pipefail

CACHE="${1:-$(dirname "$0")/../data_cache/human_orthologs}"
mkdir -p "$CACHE"
cd "$CACHE" || exit 1
NCBI=https://ftp.ncbi.nlm.nih.gov

dl () {  # url dest
  [ -s "$2" ] && { echo "SKIP $2 (present)"; return 0; }
  for attempt in 1 2 3 4; do
    if curl -sS -L --fail --max-time 1800 -o "$2.part" "$1"; then
      mv "$2.part" "$2"; echo "OK   $2 ($(stat -c%s "$2") bytes)"; return 0
    fi
    echo "RETRY($attempt) $2"; sleep $((2 ** attempt))
  done
  echo "FAIL $2"; return 1
}

# Orthologs: NCBI's ortholog pipeline (vertebrates only, 895 species vs human).
dl "$NCBI/gene/DATA/gene_orthologs.gz"                          gene_orthologs.gz
# Orthologs: Alliance/DIOPT stringent set -- adds fly, worm, yeast, X. laevis.
dl "https://fms.alliancegenome.org/download/ORTHOLOGY-ALLIANCE_COMBINED.tsv.gz" \
                                                                alliance_orthology.tsv.gz
# Gene symbols/descriptions.
dl "$NCBI/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz" Homo_sapiens.gene_info.gz
dl "$NCBI/gene/DATA/GENE_INFO/All_Data.gene_info.gz"             All_Data.gene_info.gz
# GO annotations with evidence codes (the functional payload).
dl "$NCBI/gene/DATA/gene2go.gz"                                  gene2go.gz

# Human function text from UniProt/Swiss-Prot (reviewed entries only).
if [ ! -s uniprot_human.tsv ]; then
  curl -sS --max-time 900 --compressed -G "https://rest.uniprot.org/uniprotkb/stream" \
    --data-urlencode 'query=(organism_id:9606) AND (reviewed:true)' \
    --data-urlencode 'fields=accession,gene_primary,protein_name,cc_function,go_id,annotation_score,protein_existence,xref_geneid,cc_caution' \
    --data-urlencode 'format=tsv' -o uniprot_human.tsv \
    && echo "OK   uniprot_human.tsv ($(wc -l < uniprot_human.tsv) lines)"
else
  echo "SKIP uniprot_human.tsv (present)"
fi

# Taxonomy, for assigning each ortholog species to a clade.
if [ ! -s nodes.dmp ] || [ ! -s names.dmp ]; then
  dl "$NCBI/pub/taxonomy/taxdump.tar.gz" taxdump.tar.gz \
    && tar -xzf taxdump.tar.gz nodes.dmp names.dmp && rm -f taxdump.tar.gz \
    && echo "OK   nodes.dmp names.dmp"
else
  echo "SKIP taxdump (present)"
fi

echo "--- cache contents ---"
ls -lh "$CACHE"
