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


def uniprot_map():
    import pandas as pd
    h = pd.read_csv(os.path.join(HERE, "hgnc_complete_set.txt"), sep="\t", dtype=str, usecols=["uniprot_ids", "ensembl_gene_id"]).dropna()
    out = {}
    for u, e in zip(h.uniprot_ids, h.ensembl_gene_id):
        for x in u.split("|"):
            out.setdefault(x, e)
    return out


def models():
    import pandas as pd
    m = pd.read_csv(os.path.join(HERE, "_cache", "paralog", "depmap", "raw", "Model.csv"), dtype=str)
    return dict(zip(m.SangerModelID.dropna(), m.ModelID[m.SangerModelID.notna()])), dict(zip(m.CCLEName.dropna(), m.ModelID[m.CCLEName.notna()]))


def _collapse(genes, X):
    """genes may repeat (isoforms / several accessions): keep the row with most values per gene."""
    best = {}
    for i, g in enumerate(genes):
        if g and (g not in best or np.isfinite(X[i]).sum() > np.isfinite(X[best[g]]).sum()):
            best[g] = i
    keys = sorted(best); return np.array(keys), X[[best[k] for k in keys]]


def build():
    import pandas as pd
    os.makedirs(cbb.LAY, exist_ok=True); info = {}; U = uniprot_map(); sidm, ccle = models(); H = cbb.hgnc()
    # ProCan
    d = pd.read_csv(raw("procan", "ProCan-DepMapSanger_protein_matrix_8498_averaged.txt"), sep="\t", index_col=0)
    lines = [str(x).split(";")[0] for x in d.index]; ids = np.array([sidm.get(x, x) for x in lines])
    genes = [U.get(str(c).split(";")[0], "") for c in d.columns]
    g, X = _collapse(genes, d.values.T.astype(np.float32))
    np.savez_compressed(os.path.join(cbb.LAY, "prot_procan.npz"), X=X.astype(np.float16), rows=g, cols=ids, sidm=np.array(lines))
    info["procan"] = dict(lines=len(ids), lines_with_modelid=int(sum(i.startswith("ACH-") for i in ids)), proteins=len(g), finite_share=float(np.isfinite(X).mean()))
    # CCLE proteomics (local only)
    d = pd.read_csv(raw("ccle_prot", "protein_quant_current_normalized.csv.gz"))
    meta = ["Protein_Id", "Gene_Symbol", "Description", "Group_ID", "Uniprot", "Uniprot_Acc"]
    samp = [c for c in d.columns if c not in meta and not c.endswith("_Peptides")]
    cl = {}
    for c in samp:
        cl.setdefault(c.rsplit("_TenPx", 1)[0], []).append(c)
    names = sorted(cl); Xs = np.column_stack([d[cl[n]].astype(float).mean(1).values for n in names]).astype(np.float32)
    genes = [U.get(str(a), "") or (cbb.to_ens(s) or "") for a, s in zip(d.Uniprot_Acc, d.Gene_Symbol)]
    g, X = _collapse(genes, Xs)
    ids = np.array([ccle.get(n, n) for n in names])
    np.savez_compressed(os.path.join(cbb.LAY, "prot_ccle.npz"), X=X.astype(np.float16), rows=g, cols=ids, ccle_name=np.array(names))
    info["ccle_prot"] = dict(lines=len(ids), lines_with_modelid=int(sum(i.startswith("ACH-") for i in ids)), proteins=len(g), finite_share=float(np.isfinite(X).mean()))
    # HPA tissue immunohistochemistry
    z = zipfile.ZipFile(raw("hpa_ihc", "normal_ihc_data.tsv.zip")); d = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype=str)
    lev = {"Not detected": 0, "Low": 1, "Medium": 2, "High": 3}; rel = {"Enhanced": 3, "Supported": 2, "Approved": 1, "Uncertain": 0}
    d = d[d.Level.isin(lev)]
    np.savez_compressed(os.path.join(cbb.LAY, "prot_hpa_ihc.npz"), gene=d.Gene.to_numpy(dtype=str), tissue=d.Tissue.to_numpy(dtype=str),
                        cell_type=d["Cell type"].to_numpy(dtype=str), level=d.Level.map(lev).values.astype(np.uint8),
                        reliability=d.Reliability.map(rel).fillna(0).values.astype(np.uint8))
    info["hpa_ihc"] = dict(rows=len(d), genes=int(d.Gene.nunique()), tissue_cell_types=int((d.Tissue + "|" + d["Cell type"]).nunique()),
                           detected_share=float((d.Level != "Not detected").mean()))
    # HPA single cell type RNA (pseudobulk; deeper)
    z = zipfile.ZipFile(raw("hpa_sct", "rna_single_cell_type.tsv.zip")); d = pd.read_csv(z.open(z.namelist()[0]), sep="\t")
    piv = d.pivot_table(index="Gene", columns="Cell type", values="nCPM", aggfunc="mean")
    np.savez_compressed(os.path.join(cbb.LAY, "expr_hpa_sct.npz"), X=piv.values.astype(np.float32), rows=np.array(piv.index).astype(str), cols=np.array(piv.columns).astype(str))
    info["hpa_sct"] = dict(genes=int(piv.shape[0]), cell_types=int(piv.shape[1]), median_genes_ge1_per_type=float(np.median((piv.values >= 1).sum(0))))
    # BioPlex (local only)
    rows = []
    for line, f in (("HEK293T", "BioPlex_293T_Network_10K_Dec_2019.tsv"), ("HCT116", "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv")):
        b = pd.read_csv(raw("bioplex", f), sep="\t", dtype=str)
        for x, y, p in zip(b.GeneA, b.GeneB, b.pInt):
            ea, eb = H["ent"].get(str(x)), H["ent"].get(str(y))
            if ea and eb and ea != eb:
                rows.append((ea, eb, line, float(p)))
    np.savez_compressed(os.path.join(cbb.LAY, "ppi_bioplex.npz"), a=np.array([r[0] for r in rows]), b=np.array([r[1] for r in rows]),
                        line=np.array([r[2] for r in rows]), pint=np.array([r[3] for r in rows], np.float32))
    info["bioplex"] = {ln: sum(r[2] == ln for r in rows) for ln in ("HEK293T", "HCT116")}
    # PROGENy (local only): one record per gene x pathway
    d = pd.read_csv(raw("progeny", "progeny_omnipath.tsv"), sep="\t", dtype=str)
    w = d.pivot_table(index=["record_id", "genesymbol"], columns="label", values="value", aggfunc="first").reset_index()
    w["ens"] = [cbb.to_ens(x) or "" for x in w.genesymbol]; w = w[w.ens != ""]
    np.savez_compressed(os.path.join(cbb.LAY, "path_progeny.npz"), gene=w.ens.to_numpy(dtype=str), pathway=w.pathway.to_numpy(dtype=str),
                        weight=w.weight.astype(float).values.astype(np.float32), p=w.p_value.astype(float).values.astype(np.float32))
    info["progeny"] = dict(rows=len(w), pathways=sorted(w.pathway.unique().tolist()), genes=int(w.ens.nunique()))
    # timing: HPA cell-cycle dependence
    z = zipfile.ZipFile(os.path.join(HERE, "_cache", "cellbank4", "data", "proteinatlas.tsv.zip"))
    d = pd.read_csv(z.open(z.namelist()[0]), sep="\t", usecols=["Ensembl", "Gene", "CCD Protein", "CCD Transcript"], dtype=str)
    ccd = {e: {"protein": p, "transcript": t} for e, p, t in zip(d.Ensembl, d["CCD Protein"], d["CCD Transcript"]) if isinstance(e, str) and (isinstance(p, str) or isinstance(t, str))}
    json.dump(ccd, open(os.path.join(cbb.LAY, "timing_hpa_ccd.json"), "w"))
    info["timing"] = dict(genes=len(ccd), ccd_protein_yes=sum(v["protein"] == "Yes" for v in ccd.values()), ccd_transcript_yes=sum(v["transcript"] == "Yes" for v in ccd.values()))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


def bank():
    lf, facts = {}, []

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    for key, (src, _) in SRC.items():
        add(cbb.layer_fact(f"FIX_RAW_{key.upper()}", 1, src, os.path.join(RAW, key, "raw"), "downloaded files", "-", "as published"), f"FIX_RAW_{key.upper()}")
    spec = [("PROT_PROCAN", "procan2022", "prot_procan.npz", "procan", "proteins (Ensembl)", "cell lines (ModelID; sidm)", "log2 protein intensity (DIA-MS)"),
            ("PROT_CCLE", "ccle_prot2020", "prot_ccle.npz", "ccle_prot", "proteins (Ensembl)", "cell lines (ModelID; ccle_name)", "relative TMT quantity (log2)"),
            ("PROT_HPA_IHC", "hpa", "prot_hpa_ihc.npz", "hpa_ihc", "gene x tissue x cell type rows", "level 0-3, reliability 0-3", "immunohistochemistry"),
            ("EXPR_HPA_SCT", "hpa", "expr_hpa_sct.npz", "hpa_sct", "genes (Ensembl)", "cell types (HPA names)", "nCPM (pseudobulk)"),
            ("PPI_BIOPLEX", "bioplex", "ppi_bioplex.npz", "bioplex", "interaction pairs (Ensembl)", "line HEK293T / HCT116, pInt", "AP-MS"),
            ("PATH_PROGENY", "progeny", "path_progeny.npz", "progeny", "gene x pathway rows", "weight, p", "footprint weights"),
            ("TIMING_HPA_CCD", "hpa", "timing_hpa_ccd.json", "hpa_ihc", "genes", "CCD protein / transcript (Yes/No)", "cell-cycle dependence")]
    for name, src, fn, rk, rows, cols, units in spec:
        inp = (lf[f"FIX_RAW_{rk.upper()}"],) if name != "TIMING_HPA_CCD" else (cbb.bank().facts("LAYER-HPA_PROTEINATLAS", "layer")[-1].ref,)
        add(cbb.layer_fact(name, 1, src, os.path.join(cbb.LAY, fn), rows, cols, units, inputs=inp, tier="derived"), name)
    n = cbb.bank().add_many(facts)
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers=lf, build=info,
               licences={k: cbb.SRC[v[0]][2] for k, v in SRC.items()})
    json.dump(rep, open(R("outputs", "cellbrain", "fix_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, indent=1, default=int)[:3000])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
