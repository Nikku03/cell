"""LAGTAGBANK -- gene activity before and after Gata4 switches on in mouse embryonic stem cells (Takei et al. 2026 Cell
Systems, LagTag; GEO GSE305653), added to the memory bank (cellbrain) to test whether our regulator map predicts the
next state (G15, lagtagtest.py).

Asked by the user: "do the small test with their mouse data".
Sources (each file read as data):
  GEO GSE305653 per-sample 50-bp bigWigs (the authors' CPM-normalised tracks, mm10): RNA Pol II Ser2p CUT&Tag in
    mESC-Gata4 cells, pluripotent vs 24 h after doxycycline-induced Gata4 + LIF withdrawal (endoderm), parental line
    (GSM9181445-50) and the engineered recording clone (GSM9181451-56), 3 replicates each; multi-LagTag condition 1
    (GSM9181467-69): recorded m6A (before) and endpoint Ser2p (after) in the same cells. No licence stated -> local only.
  GENCODE vM25 basic annotation (mm10; EMBL-EBI; free use): protein-coding genes.
  MGI HOM_MouseHumanSequence.rpt (Mouse Genome Informatics; CC BY 4.0): mouse-human homology classes; one-to-one only.
Signal per gene = mean bigWig value over the gene body to 3 kb past its end (strand-aware), where Ser2p Pol II sits.
Layer: LAGTAG_GATA4 (npz): mouse genes x samples, with the human ortholog (Ensembl) of each gene.

Run: python3 -I rem/atlas/lagtagbank.py pull | build | bank      (committed before it ran)
"""

from __future__ import annotations
import collections
import gzip
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

RAW = os.path.join(HERE, "_cache", "lagtag")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9181nnn/{gsm}/suppl/{f}"
SAMPLES = {   # gsm: (file stem, group)
    "GSM9181445": ("GSM9181445_mESC_Gata4_CUTTag_Ser2p_c5_rep1", "after"), "GSM9181446": ("GSM9181446_mESC_Gata4_CUTTag_Ser2p_c5_rep2", "after"),
    "GSM9181447": ("GSM9181447_mESC_Gata4_CUTTag_Ser2p_c5_rep3", "after"), "GSM9181448": ("GSM9181448_mESC_Gata4_CUTTag_Ser2p_c6_rep1", "before"),
    "GSM9181449": ("GSM9181449_mESC_Gata4_CUTTag_Ser2p_c6_rep2", "before"), "GSM9181450": ("GSM9181450_mESC_Gata4_CUTTag_Ser2p_c6_rep3", "before"),
    "GSM9181451": ("GSM9181451_mESC_Gata4_clone2_1_CUTTag_Ser2p_c5_rep1", "after"), "GSM9181452": ("GSM9181452_mESC_Gata4_clone2_1_CUTTag_Ser2p_c5_rep2", "after"),
    "GSM9181453": ("GSM9181453_mESC_Gata4_clone2_1_CUTTag_Ser2p_c5_rep3", "after"), "GSM9181454": ("GSM9181454_mESC_Gata4_clone2_1_CUTTag_Ser2p_c6_rep1", "before"),
    "GSM9181455": ("GSM9181455_mESC_Gata4_clone2_1_CUTTag_Ser2p_c6_rep2", "before"), "GSM9181456": ("GSM9181456_mESC_Gata4_clone2_1_CUTTag_Ser2p_c6_rep3", "before"),
}
LAGTAG = {"GSM9181467": "rep1", "GSM9181468": "rep2", "GSM9181469": "rep3"}     # condition 1: recorded before (m6A) + endpoint after (Ser2p)
GENCODE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gtf.gz"
MGI = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
cbb.SRC["lagtag2026"] = ("Takei et al. 2026 Cell Systems: LagTag chromatin recording, mESC-Gata4 (GEO GSE305653)", "https://doi.org/10.1016/j.cels.2026.101717",
                         "no licence stated (local only)")
cbb.SRC["gencode_mouse"] = ("GENCODE vM25 basic annotation, mouse (mm10)", "https://www.gencodegenes.org/mouse/", "free use")
cbb.SRC["mgi_homology"] = ("Mouse Genome Informatics: mouse-human homology classes (HOM_MouseHumanSequence.rpt)", "https://www.informatics.jax.org", "CC BY 4.0")
cbb.LOCAL_ONLY.add("lagtag2026")


def files():
    out = {f"{v[0]}.50bp.bw": (g, v[1]) for g, v in SAMPLES.items()}
    for g, rep in LAGTAG.items():
        for kind, grp in (("rb_m6A_Ser2p", "lagtag_recorded_before"), ("ms_Ser2p", "lagtag_endpoint_after")):
            out[f"{g}_mESC_Gata4_clone2_1_multi_LagTag_1_{kind}_{rep}.50bp.bw"] = (g, grp)
    return out


def pull():
    rep = {}
    for f, (g, grp) in files().items():
        rep[f] = xb.fetch(GEO.format(gsm=g, f=f), os.path.join(RAW, "geo", "raw", f)); print(f, rep[f], flush=True)
    for name, url, key in (("gencode.vM25.basic.annotation.gtf.gz", GENCODE, "gencode"), ("HOM_MouseHumanSequence.rpt", MGI, "mgi")):
        rep[name] = xb.fetch(url, os.path.join(RAW, key, "raw", name)); print(name, rep[name], flush=True)
    json.dump(rep, open(os.path.join(RAW, "pull.json"), "w"), indent=1, default=str)


def genes():
    out = []
    for line in gzip.open(os.path.join(RAW, "gencode", "raw", "gencode.vM25.basic.annotation.gtf.gz"), "rt"):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[2] != "gene" or 'gene_type "protein_coding"' not in f[8]:
            continue
        a = dict(x.strip().split(" ", 1) for x in f[8].strip(";").split(";") if x.strip())
        out.append((a["gene_id"].strip('"').split(".")[0], a["gene_name"].strip('"'), f[0], int(f[3]) - 1, int(f[4]), f[6]))
    return out


def orthologs():
    """one-to-one mouse -> human (symbols) from MGI homology classes."""
    cls = collections.defaultdict(lambda: {"mouse": [], "human": []})
    rows = open(os.path.join(RAW, "mgi", "raw", "HOM_MouseHumanSequence.rpt"), encoding="utf-8", errors="replace").read().splitlines()
    head = rows[0].split("\t"); ik, io, isym = head.index("DB Class Key"), head.index("Common Organism Name"), head.index("Symbol")
    for r in rows[1:]:
        f = r.split("\t")
        if len(f) <= isym:
            continue
        org = "mouse" if f[io].startswith("mouse") else "human" if f[io].startswith("human") else None
        if org:
            cls[f[ik]][org].append(f[isym])
    return {v["mouse"][0]: v["human"][0] for v in cls.values() if len(v["mouse"]) == 1 and len(v["human"]) == 1}


def build():
    import pyBigWig
    G = genes(); o2o = orthologs(); H = cbb.hgnc()
    hum = [H["sym"].get(o2o.get(g[1], ""), "") or "" for g in G]
    cols, groups = [], []; X = np.zeros((len(G), len(files())), np.float32)
    for j, (f, (gsm, grp)) in enumerate(sorted(files().items())):
        bw = pyBigWig.open(os.path.join(RAW, "geo", "raw", f)); chroms = bw.chroms()
        for i, (gid, sym, ch, s, e, strand) in enumerate(G):
            if ch not in chroms:
                continue
            a, b = (s, min(e + 3000, chroms[ch])) if strand == "+" else (max(0, s - 3000), e)
            v = bw.stats(ch, a, b, type="mean", exact=True)[0]; X[i, j] = 0.0 if v is None else v
        bw.close(); cols.append(f.replace(".50bp.bw", "")); groups.append(grp); print("read", f, flush=True)
    np.savez_compressed(os.path.join(cbb.LAY, "lagtag_gata4.npz"), X=X, gene=np.array([g[0] for g in G]), symbol=np.array([g[1] for g in G]),
                        human=np.array(hum), length=np.array([g[4] - g[3] for g in G]), cols=np.array(cols), group=np.array(groups))
    info = dict(genes=len(G), with_human_ortholog=int(sum(1 for h in hum if h)), samples=collections.Counter(groups))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info))


def bank():
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    raws = [cbb.layer_fact(f"LAGTAG_RAW_{k.upper()}", 1, src, os.path.join(RAW, k, "raw"), "downloaded files", "-", "as published")
            for k, src in (("geo", "lagtag2026"), ("gencode", "gencode_mouse"), ("mgi", "mgi_homology"))]
    lay = cbb.layer_fact("LAGTAG_GATA4", 1, "lagtag2026", os.path.join(cbb.LAY, "lagtag_gata4.npz"), "mouse protein-coding genes (GENCODE vM25) + human ortholog",
                         "RNA Pol II Ser2p samples: before / after Gata4 (CUT&Tag), LagTag recorded before / endpoint after", "mean CPM over gene body + 3 kb",
                         inputs=tuple(f.ref for f in raws), tier="derived")
    n = cbb.bank().add_many(raws + [lay])
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={f.entity: f.ref for f in raws + [lay]}, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "lagtag_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:800])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
