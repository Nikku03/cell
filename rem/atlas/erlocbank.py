"""ERLOCBANK -- where each RNA is inside a human cell, measured right at the ER membrane and the mitochondrial surface
(APEX-seq), added to the memory bank (cellbrain), to re-test "the RNA travels with its protein" (G12, erloctest.py).

Asked by the user: "can we get data for which rna will get modified tail which tells the transporter protein to where
that protein needs to be delivered?"; G9-A2/A3 failed because ENCODE's "membrane" fraction is not an ER read-out; the
user then said "do #2" (find ER-specific data and re-test).
Source: Fazal et al. 2019 Cell, "Atlas of subcellular RNA localization revealed by APEX-seq" (Ting and Chang labs;
GEO GSE116008; HEK293T). APEX2 parked at nine places labels the RNA within ~nanometres: ERM (ER membrane, cytosolic
face), KDEL (ER lumen), OMM (outer mitochondrial membrane), MITO (matrix), NES (cytosol), NLS (nucleus), NIK
(nucleolus), LMA (nuclear lamina), NUCP (nuclear pore); 2 labelled + 2 unlabelled libraries each; DESeq2 log2 fold
change labelled vs unlabelled = enrichment at that place. Drug series at OMM and in the cytosol: puromycin (releases
growing protein chains from ribosomes), cycloheximide (freezes ribosomes on the RNA), CCCP (removes mitochondrial
membrane potential). Files: the authors' processed tables (counts and DESeq2), read as data. GEO terms are set by the
submitters, so kept local; only summaries are committed.
Layer: APEX_SEQ_HEK293T (npz): genes (Ensembl) x enrichment / p / library counts per place and drug.

Run: python3 -I rem/atlas/erlocbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import csv
import gzip
import hashlib
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

RAW = os.path.join(HERE, "_cache", "erloc", "apexseq", "raw")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116008/suppl/"
FILES = ["GSE116008_APEX_seq_deseq2_all_data_polyA_hek293t.txt.gz", "GSE116008_APEX_seq_counts_all_data_polyA_hek293t.txt.gz",
         "GSE116008_APEX_seq_deseq2_omm_nes_drug_polyA_hek293t.txt.gz", "GSE116008_APEX_seq_counts_omm_nes_drug_polyA_hek293t.txt.gz",
         "GSE116008_Fractionation_counts_polyA_hek293t.txt.gz"]
PLACES = ["ERM", "KDEL", "OMM", "MITO", "NES", "NLS", "NIK", "LMA", "NUCP"]
cbb.SRC["apexseq"] = ("Fazal et al. 2019 Cell: APEX-seq atlas of subcellular RNA localization, HEK293T (GEO GSE116008)",
                      "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116008", "GEO, terms set by the submitters (local only)")
cbb.LOCAL_ONLY.add("apexseq")


def pull():
    rep = {}
    for f in FILES:
        rep[f] = xb.fetch(GEO + f, os.path.join(RAW, f)); rep[f] = dict(bytes=rep[f][0], how=rep[f][1], sha256=hashlib.sha256(open(os.path.join(RAW, f), "rb").read()).hexdigest())
        print(f, rep[f], flush=True)
    json.dump(rep, open(os.path.join(os.path.dirname(RAW), "pull.json"), "w"), indent=1)


def _table(name):
    rows = [r for r in csv.reader(gzip.open(os.path.join(RAW, name), "rt"), delimiter="\t") if r and r[0]]
    return rows[0], rows[1:]


def _num(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def build():
    cols, data = [], {}
    h, rows = _table(FILES[0])                                                   # enrichment and p per place
    for r in rows:
        data.setdefault(r[0].split(".")[0], {}).update({f"{p}_lfc": _num(r[h.index(f"{p}.18C")]) for p in PLACES})
        data[r[0].split(".")[0]].update({f"{p}_p": _num(r[h.index(f"{p}.18C.P")]) for p in PLACES})
    h, rows = _table(FILES[1])                                                   # library counts: total per place (T1+T2+C1+C2)
    for r in rows:
        d = data.setdefault(r[0].split(".")[0], {})
        for p in PLACES:
            d[f"{p}_count"] = sum(_num(r[h.index(f"{p}-{k}")]) for k in ("T1", "T2", "C1", "C2"))
    h, rows = _table(FILES[2])                                                   # drugs at OMM and in the cytosol
    drug_cols = {"OMM_basal.18C": "OMM_basal_lfc", "NES_basal.18C": "NES_basal_lfc", "OMM.puro.18C": "OMM_puro_lfc", "NES.puro.18C": "NES_puro_lfc",
                 "OMM.cyc.18C": "OMM_cyc_lfc", "NES.cyc.18C": "NES_cyc_lfc", "OMM.cccp.18C": "OMM_cccp_lfc", "NES.cccp.18C": "NES_cccp_lfc"}
    for r in rows:
        d = data.setdefault(r[0].split(".")[0], {})
        for c, k in drug_cols.items():
            d[k] = _num(r[h.index(c)])
    h, rows = _table(FILES[3])
    for r in rows:
        d = data.setdefault(r[0].split(".")[0], {})
        for dr in ("puro", "cyc", "cccp"):
            d[f"OMM_{dr}_count"] = sum(_num(r[h.index(f"OMM-{dr}-{k}")]) for k in ("T1", "T2", "C1", "C2"))
    genes = sorted(g for g in data if g.startswith("ENSG"))
    cols = sorted({k for g in genes for k in data[g]})
    X = np.array([[data[g].get(c, np.nan) for c in cols] for g in genes], np.float32)
    np.savez_compressed(os.path.join(cbb.LAY, "apex_seq_hek293t.npz"), X=X, rows=np.array(genes), cols=np.array(cols))
    info = dict(genes=len(genes), columns=cols, with_erm=int(np.isfinite(X[:, cols.index("ERM_lfc")]).sum()),
                with_omm_puro=int(np.isfinite(X[:, cols.index("OMM_puro_lfc")]).sum()))
    json.dump(info, open(os.path.join(os.path.dirname(RAW), "build_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


def bank():
    info = json.load(open(os.path.join(os.path.dirname(RAW), "build_info.json")))
    raw_f = cbb.layer_fact("APEX_SEQ_RAW", 1, "apexseq", RAW, "downloaded files", "-", "as published")
    lay = cbb.layer_fact("APEX_SEQ_HEK293T", 1, "apexseq", os.path.join(cbb.LAY, "apex_seq_hek293t.npz"), "genes (Ensembl)",
                         "enrichment (log2 labelled / unlabelled), p, library counts per place; OMM / cytosol under puromycin, cycloheximide, CCCP",
                         "log2 fold change (DESeq2, the authors')", inputs=(raw_f.ref,), tier="derived")
    n = cbb.bank().add_many([raw_f, lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={"APEX_SEQ_RAW": raw_f.ref, "APEX_SEQ_HEK293T": lay.ref}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "erloc_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:1200])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
