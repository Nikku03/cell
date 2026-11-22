"""PARALOGBANK -- substitute proteins (paralogs) and cancer-cell-line expression, added to the memory bank (cellbrain).

Asked by the user: "but what about substitute proteins?" A substitute is a paralog: a sister gene from the same
family that can do the same job. Testing whether substitutes cover for a gene needs (1) who the sisters are and (2)
how much of each sister a cell makes, in many cells where we also know how much the cell needs the gene.
1. Ensembl Compara 116 homologies (protein_default, human file; Ensembl data, no restrictions): human-human paralog
   pairs with % identity both ways (ftp.ensembl.org/pub/current/tsv/ensembl-compara/homologies/homo_sapiens/).
2. DepMap 24Q4 Public (figshare 27993248, CC BY 4.0): OmicsExpressionProteinCodingGenesTPMLogp1 (RNA per cell line,
   log2(TPM+1)) and Model.csv (names), the same release as the CRISPR gene-effect layer already banked (DEP_DEPMAP).
Downloads are untrusted: each in its own raw folder, md5-checked; read as data, never run.
Layers: PARALOGS_ENSEMBL (pairs, identities, type), EXPR_DEPMAP (genes x cell lines, log2 TPM+1, float16).

Run: python3 -I rem/atlas/paralogbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import gzip
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("xcellbank", os.path.join(HERE, "xcellbank.py"))
xb = importlib.util.module_from_spec(_s); _s.loader.exec_module(xb)
cbb, Fact = xb.cbb, xb.Fact

RAW = os.path.join(HERE, "_cache", "paralog")
ENS_RAW, DM_RAW = os.path.join(RAW, "ensembl", "raw"), os.path.join(RAW, "depmap", "raw")
ENS_URL = "https://ftp.ensembl.org/pub/current/tsv/ensembl-compara/homologies/homo_sapiens/Compara.116.protein_default.homologies.tsv.gz"
ENS_MD5 = "59857f48bbbdf6812999d58d7a24ccc4"
DM = {"OmicsExpressionProteinCodingGenesTPMLogp1.csv": (51065489, "71794802b750ce77c422dad0720a40af"), "Model.csv": (51065297, "675210d17675f3517b0ce39a3c274f16")}
cbb.SRC["ensembl_compara"] = ("Ensembl Compara 116: human paralogues (protein_default homologies)", "https://www.ensembl.org/info/genome/compara/", "no restrictions (Ensembl)")
cbb.SRC["depmap_expr"] = ("DepMap 24Q4 Public: OmicsExpressionProteinCodingGenesTPMLogp1 and Model.csv (figshare 27993248)", "https://doi.org/10.25452/figshare.plus.27993248", "CC BY 4.0")


def pull():
    rep = {"ensembl": xb.fetch(ENS_URL, os.path.join(ENS_RAW, "Compara.116.protein_default.homologies.tsv.gz"), ENS_MD5)}
    for name, (fid, md5) in DM.items():
        rep[name] = xb.fetch(xb.FS.format(fid), os.path.join(DM_RAW, name), md5)
    print(json.dumps(rep, indent=1))


def build():
    import pandas as pd
    info = {}
    keep = []
    with gzip.open(os.path.join(ENS_RAW, "Compara.116.protein_default.homologies.tsv.gz"), "rt") as f:
        head = f.readline().rstrip("\n").split("\t"); ix = {c: i for i, c in enumerate(head)}
        for line in f:
            x = line.rstrip("\n").split("\t")
            if x[ix["homology_species"]] == "homo_sapiens" and x[ix["species"]] == "homo_sapiens" and "paralog" in x[ix["homology_type"]]:
                a, b = x[ix["gene_stable_id"]], x[ix["homology_gene_stable_id"]]
                if a != b:
                    keep.append((a, b, float(x[ix["identity"]]), float(x[ix["homology_identity"]]), x[ix["homology_type"]]))
    seen, rows = set(), []
    for a, b, ia, ib, t in keep:                     # one row per unordered pair (the file lists both directions)
        k = (a, b) if a < b else (b, a)
        if k not in seen:
            seen.add(k); rows.append((a, b, ia, ib, t))
    np.savez(os.path.join(cbb.LAY, "paralogs_ensembl.npz"), a=np.array([r[0] for r in rows]), b=np.array([r[1] for r in rows]),
             id_a=np.array([r[2] for r in rows], np.float32), id_b=np.array([r[3] for r in rows], np.float32), type=np.array([r[4] for r in rows]))
    info["paralogs"] = dict(pairs=len(rows), genes=len({r[0] for r in rows} | {r[1] for r in rows}),
                            min_identity_ge_30=int(sum(min(r[2], r[3]) >= 30 for r in rows)), types=sorted({r[4] for r in rows}))
    d = pd.read_csv(os.path.join(DM_RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"), index_col=0)
    H = cbb.hgnc(); cols, gi = [], []
    for j, c in enumerate(d.columns):
        m_ = re.match(r"(.+) \((\d+)\)", c)
        e = (H["ent"].get(m_.group(2)) or cbb.to_ens(m_.group(1))) if m_ else cbb.to_ens(c)
        if e and e not in cols:
            cols.append(e); gi.append(j)
    ids = [str(x) for x in d.index]
    if not ids[0].startswith("ACH-"):              # some releases put metadata columns first
        mid = [c for c in d.columns if "ModelID" in str(c)]
        ids = [str(x) for x in d[mid[0]]] if mid else ids
    X = d.values[:, gi].T.astype(np.float16)
    np.savez(os.path.join(cbb.LAY, "expr_depmap.npz"), X=X, rows=np.array(cols), cols=np.array(ids))
    m = pd.read_csv(os.path.join(DM_RAW, "Model.csv"), dtype=str)
    info["depmap_expr"] = dict(genes=len(cols), cell_lines=len(ids), first_ids=ids[:3], model_rows=len(m))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


def bank():
    lf, facts = {}, []

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    add(cbb.layer_fact("ENSEMBL_COMPARA_RAW", 1, "ensembl_compara", ENS_RAW, "homologies TSV (human file)", "-", "Ensembl Compara 116", extra={"md5": ENS_MD5}), "ENSEMBL_COMPARA_RAW")
    add(cbb.layer_fact("DEPMAP_EXPR_RAW", 1, "depmap_expr", DM_RAW, "2 CSV", "-", "DepMap 24Q4", extra={"md5": {k: v[1] for k, v in DM.items()}}), "DEPMAP_EXPR_RAW")
    add(cbb.layer_fact("PARALOGS_ENSEMBL", 1, "ensembl_compara", os.path.join(cbb.LAY, "paralogs_ensembl.npz"), f"{info['paralogs']['pairs']:,} human paralog pairs",
                       "a, b (Ensembl), id_a, id_b (% identity), type", "percent identity", inputs=(lf["ENSEMBL_COMPARA_RAW"],), tier="derived"), "PARALOGS_ENSEMBL")
    add(cbb.layer_fact("EXPR_DEPMAP", 1, "depmap_expr", os.path.join(cbb.LAY, "expr_depmap.npz"), f"{info['depmap_expr']['genes']:,} genes (Ensembl)",
                       f"{info['depmap_expr']['cell_lines']:,} cell lines (ModelID)", "log2(TPM+1), float16", inputs=(lf["DEPMAP_EXPR_RAW"],), tier="derived"), "EXPR_DEPMAP")
    n = cbb.bank().add_many(facts)
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers=lf, build=info, licences={"ensembl_compara": "no restrictions", "depmap_expr": "CC BY 4.0"})
    json.dump(rep, open(R("outputs", "cellbrain", "paralog_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
