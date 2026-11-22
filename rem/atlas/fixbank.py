"""FIXBANK -- public data that fixes the cell-map limits, added to the memory bank (cellbrain).

Asked by the user: "first add public available info, then the yes things, then complete the list" (after the list of
cell-map limits: RNA not protein; only the busiest genes; rooms and partners from few cells; no timing; no on/off
switches; a quarter of proteins unknown; cannot tell if a machine is built).
Sources (each in its own raw folder, md5-checked where the host publishes one; read as data, never run):
  ProCan-DepMapSanger (Goncalves et al. 2022 Cancer Cell; figshare 19345397; CC BY 4.0): protein intensities, 949
     cancer lines x 8,498 proteins (averaged replicates) + sample mapping.            -> limit 1 (protein), 7 (machines)
  CCLE proteomics (Nusinow et al. 2020 Cell; gygi.hms.harvard.edu): TMT protein quantities, 375 lines. Terms unclear:
     LOCAL ONLY.                                                                      -> limit 1, 7 (independent replicate)
  HPA normal tissue immunohistochemistry (proteinatlas.org, CC BY-SA 3.0): protein staining level per gene, tissue and
     cell type.                                                                       -> limit 1 (protein in body cell types)
  HPA single cell type RNA (CC BY-SA 3.0): pseudobulk nTPM per cell type (deeper).    -> limit 2
  BioPlex 3.0 (Huttlin et al. 2021 Cell): AP-MS interactions measured in HEK293T and in HCT116. Terms unclear: LOCAL
     ONLY.                                                                            -> limit 3 (partners in two cell types)
  PROGENy pathway footprints (Schubert et al. 2018; via OmniPath, academic): LOCAL ONLY.  -> limit 5 (on/off)
  (Timing, limit 4, uses the HPA cell-cycle-dependence columns already banked in HPA_PROTEINATLAS.)
Layers: PROT_PROCAN, PROT_CCLE (local), PROT_HPA_IHC, EXPR_HPA_SCT, PPI_BIOPLEX (local), PATH_PROGENY (local),
TIMING_HPA_CCD.

Run: python3 -I rem/atlas/fixbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import json
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("xcellbank", os.path.join(HERE, "xcellbank.py"))
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
cbb, Fact = xb.cbb, xb.Fact

RAW = os.path.join(HERE, "_cache", "fix")
SRC = {
    "procan": ("procan2022", [("ProCan-DepMapSanger_protein_matrix_8498_averaged.txt", "https://ndownloader.figshare.com/files/34411172", "940812e0362385ee5219166053201062"),
                              ("ProCan-DepMapSanger_mapping_file_averaged.txt", "https://ndownloader.figshare.com/files/34411133", "21407e29046d77f42b4f39298ffee7f2"),
                              ("README.md", "https://ndownloader.figshare.com/files/36155439", "a5beb4ccdbdda90c8fb1266ae5869f55")]),
    "ccle_prot": ("ccle_prot2020", [("protein_quant_current_normalized.csv.gz", "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz", None)]),
    "hpa_ihc": ("hpa", [("normal_ihc_data.tsv.zip", "https://www.proteinatlas.org/download/tsv/normal_ihc_data.tsv.zip", None)]),
    "hpa_sct": ("hpa", [("rna_single_cell_type.tsv.zip", "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip", None)]),
    "bioplex": ("bioplex", [("BioPlex_293T_Network_10K_Dec_2019.tsv", "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv", None),
                            ("BioPlex_HCT116_Network_5.5K_Dec_2019.tsv", "https://bioplex.hms.harvard.edu/data/BioPlex_HCT116_Network_5.5K_Dec_2019.tsv", None)]),
    "progeny": ("progeny", [("progeny_omnipath.tsv", "https://omnipathdb.org/annotations?resources=PROGENy&license=academic", None)]),
}
cbb.SRC["procan2022"] = ("Goncalves et al. 2022 Cancer Cell: ProCan-DepMapSanger pan-cancer proteomic map of 949 cell lines (figshare 19345397)",
                         "https://doi.org/10.6084/m9.figshare.19345397", "CC BY 4.0")
cbb.SRC["ccle_prot2020"] = ("Nusinow et al. 2020 Cell: CCLE quantitative proteomics, 375 lines", "https://gygi.hms.harvard.edu/publications/ccle.html", "unclear (local only)")
cbb.SRC["bioplex"] = ("Huttlin et al. 2021 Cell: BioPlex 3.0 interactomes of HEK293T and HCT116", "https://bioplex.hms.harvard.edu", "unclear (local only)")
cbb.SRC["progeny"] = ("Schubert et al. 2018 Nat Commun: PROGENy pathway footprints (via OmniPath)", "https://omnipathdb.org", "academic (local only)")
cbb.SRC.setdefault("depmap_expr", ("DepMap 24Q4 Public (figshare 27993248)", "https://doi.org/10.25452/figshare.plus.27993248", "CC BY 4.0"))
for k in ("ccle_prot2020", "bioplex", "progeny"):
    cbb.LOCAL_ONLY.add(k)


def raw(key, name):
    return os.path.join(RAW, key, "raw", name)


def pull():
    rep = {}
    for key, (_, files) in SRC.items():
        for name, url, md5 in files:
            rep[f"{key}/{name}"] = xb.fetch(url, raw(key, name), md5); print(key, name, rep[f"{key}/{name}"], flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1)


if __name__ == "__main__":
    {"pull": pull}[sys.argv[1]]()
