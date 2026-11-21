"""XCELLBANK -- the same gene knockdowns measured in four cell types, gene by gene, added to the memory bank (cellbrain).

Asked by the user: "how influence of one gene on other changes with cell type, and which gene influences which and at
what level". That needs the SAME knockdowns read out in several cell types, target gene by target gene. The bank had
that only for K562 (RPE1, HepG2 and Jurkat were stored as 128-number summaries). This adds:

1. Nadig et al. 2025 Nat Genet (TRADE; figshare 29498366; CC BY 4.0): differential expression of every measured gene
   after every knockdown, from one pipeline (DESeq2 on the Perturb-seq data vs non-targeting cells), for four
   essential-gene CRISPRi screens with the same ~2,400-gene library -- K562 and RPE1 (Replogle 2022, reanalysed),
   HepG2 and Jurkat (new in Nadig 2025) -- and the K562 genome-wide screen (~10,000 knockdowns; an independent
   experiment in the same cell type, used as the replicate). Per file: Log2FoldChange, lfcSE, p (rows: measured genes,
   Ensembl; columns: knocked-down genes, symbols).
2. CollecTRI (Mueller-Dott et al. 2023 NAR; via OmniPath, omnipathdb.org): literature-curated signed transcription
   factor -> target gene interactions. Mixed source licences (some academic-only): kept local, never committed.
Downloads are untrusted: each lands in its own raw folder and is checked against the md5 figshare publishes; this code
reads them as data, it never runs them.
Layers: XCELL_<SCREEN> (lfc, se, -log10 p as float16; measured genes x knockdowns), REG_COLLECTRI (local only).

Run: python3 -I rem/atlas/xcellbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("cellbrain_bank", os.path.join(HERE, "cellbrain_bank.py"))
cbb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cbb)
from brain.bank import Fact  # noqa: E402

RAW = os.path.join(HERE, "_cache", "xcell")
NAD_RAW, CT_RAW = os.path.join(RAW, "nadig2025", "raw"), os.path.join(RAW, "collectri", "raw")
FS = "https://ndownloader.figshare.com/files/{}"
# screen -> {kind: (figshare file id, md5 published by figshare)}
FILES = {
    "K562ESS": {"Log2FoldChange": (56049056, "72b8deccb64c0f62f7becbd630d02382"), "lfcSE": (56049050, "86a97086ad077bb22b059a2dda30af9e"),
                "p": (56049053, "921a2f0f52ff21c1787b7eaed1c8ad0f")},
    "RPE1ESS": {"Log2FoldChange": (56049083, "83215c46746e34b2319bb316fe51c6db"), "lfcSE": (56049080, "db3eae7014fd8716f379f0c760e52fd6"),
                "p": (56049086, "60f74e506a3132dd84514efab95a4d48")},
    "HEPG2ESS": {"Log2FoldChange": (56049065, "aa56f1ecf3d48d1121895d1eaa0796ee"), "lfcSE": (56049059, "1fcb1515acda970b992d5a6021479d06"),
                 "p": (56049068, "58d7312c06b6438e850570ca41d41ac2")},
    "JURKATESS": {"Log2FoldChange": (56049071, "844e78051b1721bc27dcb6b6f9f84136"), "lfcSE": (56049062, "ed500e67fccba1feca4a0d7d61a9a8af"),
                  "p": (56049074, "8f8b9400c7e000cada8178a7b3e13d87")},
    "K562GW": {"Log2FoldChange": (56075999, "70f1419a906719fb32442b3bf1449083"), "lfcSE": (56075993, "697eb5fe6988523b506e844da31c8e42"),
               "p": (56075996, "43af5b667130a96c1117f1e3c0d0a786")},
}
CELL = {"K562ESS": "K562", "RPE1ESS": "RPE1", "HEPG2ESS": "HepG2", "JURKATESS": "Jurkat", "K562GW": "K562"}
DESIGN = {"K562ESS": "essential-gene library (Replogle 2022 data, reanalysed)", "RPE1ESS": "essential-gene library (Replogle 2022 data, reanalysed)",
          "HEPG2ESS": "essential-gene library (Nadig 2025)", "JURKATESS": "essential-gene library (Nadig 2025)",
          "K562GW": "genome-wide library (Replogle 2022 data, reanalysed); independent experiment"}
COLLECTRI_URL = "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=sources,references,curation_effort&license=academic"

cbb.SRC["nadig2025_de"] = ("Nadig et al. 2025 Nat Genet: transcriptome-wide differential expression for K562, RPE1, HepG2, Jurkat essential and "
                           "K562 genome-wide CRISPRi Perturb-seq (TRADE; figshare 29498366)", "https://doi.org/10.6084/m9.figshare.29498366", "CC BY 4.0")
cbb.SRC["collectri"] = ("Mueller-Dott et al. 2023 NAR: CollecTRI signed TF-target regulons (via OmniPath)", "https://omnipathdb.org", "mixed (academic)")
cbb.LOCAL_ONLY.add("collectri")


def fetch(url, path, md5=None, tries=6):
    """Stream to disk (no whole-file reads), check md5; resumable by re-running (a finished file with the right md5 is kept)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and md5 and _md5(path) == md5:
        return os.path.getsize(path), "kept"
    for k in range(tries):
        try:
            h = hashlib.md5()
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cell-atlas-research/1.0"}), timeout=900) as r, \
                    open(path + ".part", "wb") as fh:
                for b in iter(lambda: r.read(1 << 22), b""):
                    fh.write(b); h.update(b)
            if md5 and h.hexdigest() != md5:
                raise IOError(f"md5 mismatch {path}: {h.hexdigest()} != {md5}")
            os.replace(path + ".part", path)
            return os.path.getsize(path), "fetched"
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def raw_path(screen, kind):
    return os.path.join(NAD_RAW, f"{screen}_{kind}.csv.gz")


def pull():
    rep = {}
    for s, d in FILES.items():
        for kind, (fid, md5) in d.items():
            rep[f"{s}_{kind}"] = fetch(FS.format(fid), raw_path(s, kind), md5)
            print(s, kind, rep[f"{s}_{kind}"], flush=True)
    rep["collectri"] = fetch(COLLECTRI_URL, os.path.join(CT_RAW, "collectri.tsv"))
    print(json.dumps(rep, indent=1))


def _read(path):
    import pandas as pd
    df = pd.read_csv(path, index_col=0, compression="gzip")
    return [str(x) for x in df.index], [str(x) for x in df.columns], df.to_numpy(dtype=np.float32)


def build():
    os.makedirs(cbb.LAY, exist_ok=True); info = {}
    for s in FILES:
        rows, cols, lfc = _read(raw_path(s, "Log2FoldChange"))
        r2, c2, se = _read(raw_path(s, "lfcSE")); r3, c3, p = _read(raw_path(s, "p"))
        assert r2 == rows and r3 == rows and c2 == cols and c3 == cols, s
        nlp = -np.log10(np.clip(p, 1e-300, 1.0)); nlp[~np.isfinite(p)] = np.nan
        ens = np.array([cbb.to_ens(c) or "" for c in cols])
        rows_e = np.array([r.split(".")[0] for r in rows])
        # data check (not an outcome): are p values the Wald p of lfc / lfcSE (unshrunk), or independent of a shrunk lfc?
        m = np.isfinite(lfc) & np.isfinite(se) & (se > 0) & np.isfinite(p) & (p > 1e-300)
        idx = np.flatnonzero(m.ravel())[:: max(1, m.sum() // 200000)]
        from scipy.stats import norm, spearmanr
        zw = np.abs(lfc.ravel()[idx] / se.ravel()[idx]); zp = norm.isf(p.ravel()[idx] / 2)
        info[s] = dict(cell=CELL[s], design=DESIGN[s], measured_genes=len(rows), knockdowns=len(cols), mapped_knockdowns=int((ens != "").sum()),
                       nan_lfc=int((~np.isfinite(lfc)).sum()), wald_check=dict(spearman=float(spearmanr(zw, zp)[0]),
                       median_ratio=float(np.median(zw / np.maximum(zp, 1e-9))), n=int(len(idx))),
                       duplicate_knockdown_symbols=int(len(cols) - len(set(cols))))
        np.savez(os.path.join(cbb.LAY, f"xcell_{s.lower()}.npz"), lfc=lfc.astype(np.float16), se=se.astype(np.float16), nlp=nlp.astype(np.float16),
                 rows=rows_e, cols=np.array(cols), cols_ens=ens)
        print(s, json.dumps(info[s]), flush=True)
        del lfc, se, p, nlp
    import pandas as pd
    ct = pd.read_csv(os.path.join(CT_RAW, "collectri.tsv"), sep="\t", dtype=str)
    a = np.array([cbb.to_ens(x) or "" for x in ct.source_genesymbol]); b = np.array([cbb.to_ens(x) or "" for x in ct.target_genesymbol])
    sign = np.where(ct.consensus_stimulation == "True", 1, np.where(ct.consensus_inhibition == "True", -1, 0)).astype(np.int8)
    keep = (a != "") & (b != "")
    np.savez(os.path.join(cbb.LAY, "reg_collectri.npz"), tf=a[keep], target=b[keep], sign=sign[keep],
             tf_symbol=np.array(ct.source_genesymbol)[keep].astype(str), target_symbol=np.array(ct.target_genesymbol)[keep].astype(str))
    info["collectri"] = dict(rows=int(len(ct)), mapped=int(keep.sum()), tfs=int(len(set(a[keep]))), targets=int(len(set(b[keep]))),
                             activating=int((sign[keep] == 1).sum()), repressing=int((sign[keep] == -1).sum()))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1)
    print(json.dumps(info["collectri"]))


def bank():
    lf, facts = {}, []

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    add(cbb.layer_fact("NADIG2025_DE_RAW", 1, "nadig2025_de", NAD_RAW, "15 csv.gz (5 screens x Log2FoldChange, lfcSE, p)", "-",
                       "DESeq2 per knockdown vs non-targeting cells", extra={"md5": {f"{s}_{k}": v[1] for s, d in FILES.items() for k, v in d.items()}}),
        "NADIG2025_DE_RAW")
    add(cbb.layer_fact("COLLECTRI_RAW", 1, "collectri", os.path.join(CT_RAW, "collectri.tsv"), "TF-target interactions", "OmniPath columns",
                       "signed curated interactions"), "COLLECTRI_RAW")
    for s in FILES:
        i = info[s]
        add(cbb.layer_fact(f"XCELL_{s}", 1, "nadig2025_de", os.path.join(cbb.LAY, f"xcell_{s.lower()}.npz"),
                           f"{i['measured_genes']:,} measured genes (Ensembl)", f"{i['knockdowns']:,} knockdowns (symbol; cols_ens = Ensembl)",
                           "lfc = log2 fold change, se = lfcSE, nlp = -log10 p (float16)", inputs=(lf["NADIG2025_DE_RAW"],), tier="derived",
                           extra={"cell": CELL[s], "design": DESIGN[s], "wald_check": i["wald_check"]}), f"XCELL_{s}")
    add(cbb.layer_fact("REG_COLLECTRI", 1, "collectri", os.path.join(cbb.LAY, "reg_collectri.npz"), "TF-target pairs (Ensembl)", "sign (+1 / -1 / 0)",
                       "curated", inputs=(lf["COLLECTRI_RAW"],), tier="derived"), "REG_COLLECTRI")
    t = cbb.now(); src = cbb.source("nadig2025_de")
    for s in FILES:
        i = info[s]
        facts.append(Fact(f"SCREEN-XCELL_{s}", "screen", {"cell": CELL[s], "design": DESIGN[s], "measured_genes": i["measured_genes"],
                                                          "knockdowns": i["knockdowns"], "layer": f"XCELL_{s}"}, t[:10], t, "reported", src,
                          inputs=(lf[f"XCELL_{s}"],)))
    n = cbb.bank().add_many(facts)
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers=lf, build=info,
               licences={"nadig2025_de": "CC BY 4.0", "collectri": "mixed, academic (local only)"})
    json.dump(rep, open(R("outputs", "cellbrain", "xcell_report.json"), "w"), indent=1, default=int)
    print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
