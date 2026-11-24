"""ADDRESSBANK -- protein address labels, RNA zip codes and RNA versions (isoforms), added to the memory bank (cellbrain).

Asked by the user: "can we get data for which RNA gets a tag that tells the transporter where the protein must be
delivered?" -- then "yes add them and run the tests", with partner finding by compatibility then destination, and
RNA maturation (introns, exons, the proteins they make, when, in what condition, what governs them).
Sources (each in its own raw folder; read as data, never run):
  UniProt (reviewed human; rest.uniprot.org; CC BY 4.0): signal peptides, transit peptides, transmembrane segments,
     motifs (nuclear localization, ER retention, peroxisome targeting), lipid anchors, subcellular location text,
     Pfam domains, alternative sequences of isoforms, Ensembl transcript <-> isoform links.
  MitoCarta 3.0 (Rath et al. 2021 NAR; Broad Institute): 1,136 mitochondrial proteins. Local only (terms unclear).
  RNA location: RNALocate's download links return 404 and APEX-seq (GEO; paper not open access) is unreachable here,
     so ENCODE fractionation RNA-seq is used (Lecuyer lab, CeFra-seq design; encodeproject.org): polyA RNA from the
     cytosolic, insoluble-cytoplasmic, membrane and nuclear fractions of K562 and HepG2, 2 replicates each (GENCODE V29
     gene quantifications, md5 from ENCODE). RNA in the membrane fraction is RNA translated at the ER.
  HPA transcript RNA by tissue (proteinatlas.org; CC BY-SA 3.0): expression of each transcript (isoform) per tissue.
  ENCODE RBP knockdown RNA-seq (encodeproject.org; ENCODE data, free use): transcript quantifications for knockdowns
     of RNA-binding proteins that are splicing regulators, and their controls (pulled in a later step, gene list fixed
     here: Reactome "mRNA Splicing" members that ENCODE knocked down in K562).
Layers: ADDR_UNIPROT, ADDR_MITOCARTA (local), RNA_FRACTIONS_ENCODE, ISOFORM_HPA_TISSUE.

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
    "hpa_tx": ("hpa", [("transcript_rna_tissue.tsv.zip", "https://www.proteinatlas.org/download/tsv/transcript_rna_tissue.tsv.zip")]),
}
cbb.SRC["uniprot"] = ("UniProt Knowledgebase, reviewed human entries (rest.uniprot.org)", "https://www.uniprot.org", "CC BY 4.0")
cbb.SRC["mitocarta"] = ("Rath et al. 2021 NAR: MitoCarta3.0", "https://www.broadinstitute.org/mitocarta", "unclear (local only)")
cbb.SRC["encode_frac"] = ("ENCODE fractionation RNA-seq (Lecuyer lab): K562 and HepG2 cytosolic, insoluble cytoplasmic, membrane, nuclear fractions",
                          "https://www.encodeproject.org", "ENCODE data, free use")
FRACTIONS = {("K562", "cytosolic"): "ENCSR384ZXD", ("K562", "insoluble"): "ENCSR594NJP", ("K562", "membrane"): "ENCSR596ACL", ("K562", "nuclear"): "ENCSR530NHO",
             ("HepG2", "cytosolic"): "ENCSR931WGT", ("HepG2", "insoluble"): "ENCSR019MXZ", ("HepG2", "membrane"): "ENCSR541TIG", ("HepG2", "nuclear"): "ENCSR058OSL"}
for k in ("mitocarta",):
    cbb.LOCAL_ONLY.add(k)


def raw(key, name):
    return os.path.join(RAW, key, "raw", name)


def encode_files(acc, annotation="V29"):
    """gene quantification files (GRCh38) per biological replicate for one ENCODE experiment: [(rep, accession, md5, href)]."""
    d = json.loads(xb.urllib.request.urlopen(xb.urllib.request.Request(f"https://www.encodeproject.org/experiments/{acc}/?format=json",
                                                                       headers={"User-Agent": "cell-atlas-research/1.0"}), timeout=120).read())
    out = []
    for f in d["files"]:
        if f.get("output_type") == "gene quantifications" and f.get("assembly") == "GRCh38" and f.get("genome_annotation") == annotation and f.get("status") == "released":
            out.append((f["biological_replicates"][0], f["accession"], f["md5sum"], "https://www.encodeproject.org" + f["href"]))
    return sorted(out)


def pull():
    rep = {}
    for (cell, frac), acc in FRACTIONS.items():
        for r, fa, md5, url in encode_files(acc):
            name = f"{cell}_{frac}_rep{r}_{fa}.tsv"
            rep[f"encode/{name}"] = xb.fetch(url, raw("encode_frac", name), md5); print("encode", name, rep[f"encode/{name}"], flush=True)
    for key, (_, files) in SRC.items():
        for name, url in files:
            rep[f"{key}/{name}"] = xb.fetch(url, raw(key, name)); print(key, name, rep[f"{key}/{name}"], flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1)


if __name__ == "__main__":
    {"pull": pull}[sys.argv[1]]()
