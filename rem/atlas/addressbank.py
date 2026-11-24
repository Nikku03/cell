"""ADDRESSBANK -- protein address labels, RNA zip codes and RNA versions (isoforms), added to the memory bank (cellbrain).

Asked by the user: "can we get data for which RNA gets a tag that tells the transporter where the protein must be
delivered?" -- then "yes add them and run the tests", with partner finding by compatibility then destination, and
RNA maturation (introns, exons, the proteins they make, when, in what condition, what governs them).
Sources (each in its own raw folder; read as data, never run):
  UniProt (reviewed human; rest.uniprot.org; CC BY 4.0): signal peptides, transit peptides, transmembrane segments,
     motifs (nuclear localization, ER retention, peroxisome targeting), lipid anchors, subcellular location text,
     Pfam domains, alternative sequences of isoforms, Ensembl transcript <-> isoform links.
  MitoCarta 3.0 (Rath et al. 2021 NAR; Broad Institute): 1,136 mitochondrial proteins. Local only (terms unclear).
  RNALocate v3 (rnalocate.org): experimentally supported mRNA subcellular locations. Local only (terms unclear).
  HPA transcript RNA by tissue (proteinatlas.org; CC BY-SA 3.0): expression of each transcript (isoform) per tissue.
  ENCODE RBP knockdown RNA-seq (encodeproject.org; ENCODE data, free use): transcript quantifications for knockdowns
     of RNA-binding proteins that are splicing regulators, and their controls (pulled in a later step, gene list fixed
     here: Reactome "mRNA Splicing" members that ENCODE knocked down in K562).
Layers: ADDR_UNIPROT, ADDR_MITOCARTA (local), RNA_LOCATE (local), ISOFORM_HPA_TISSUE.

Run: python3 -I rem/atlas/addressbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("xcellbank", os.path.join(HERE, "xcellbank.py"))
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
cbb, Fact = xb.cbb, xb.Fact

RAW = os.path.join(HERE, "_cache", "address")
UNIPROT_FIELDS = ["accession", "gene_primary", "xref_ensembl", "ft_signal", "ft_transit", "ft_transmem", "ft_intramem", "ft_motif", "ft_lipid",
                  "cc_subcellular_location", "xref_pfam", "ft_var_seq", "length"]
SRC = {
    "uniprot": ("uniprot", [("uniprot_human_reviewed.tsv.gz", "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&format=tsv&compressed=true&fields="
                             + ",".join(UNIPROT_FIELDS))]),
    "mitocarta": ("mitocarta", [("Human.MitoCarta3.0.xls", "https://personal.broadinstitute.org/scalvo/MitoCarta3.0/Human.MitoCarta3.0.xls")]),
    "rnalocate": ("rnalocate", [("mRNA_subcellular_localization.zip", "http://www.rnalocate.org/static/download/mRNA%20subcellular%20localization%20information.zip")]),
    "hpa_tx": ("hpa", [("transcript_rna_tissue.tsv.zip", "https://www.proteinatlas.org/download/tsv/transcript_rna_tissue.tsv.zip")]),
}
cbb.SRC["uniprot"] = ("UniProt Knowledgebase, reviewed human entries (rest.uniprot.org)", "https://www.uniprot.org", "CC BY 4.0")
cbb.SRC["mitocarta"] = ("Rath et al. 2021 NAR: MitoCarta3.0", "https://www.broadinstitute.org/mitocarta", "unclear (local only)")
cbb.SRC["rnalocate"] = ("RNALocate v3: RNA subcellular localization database", "http://www.rnalocate.org", "unclear (local only)")
for k in ("mitocarta", "rnalocate"):
    cbb.LOCAL_ONLY.add(k)


def raw(key, name):
    return os.path.join(RAW, key, "raw", name)


def pull():
    rep = {}
    for key, (_, files) in SRC.items():
        for name, url in files:
            rep[f"{key}/{name}"] = xb.fetch(url, raw(key, name)); print(key, name, rep[f"{key}/{name}"], flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1)


if __name__ == "__main__":
    {"pull": pull}[sys.argv[1]]()
