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
import collections
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


TAGS = {"Nuclear localization signal": "NLS", "Bipartite nuclear localization signal": "NLS", "Nuclear localization signal (NLS)": "NLS",
        "Nuclear export signal": "NES", "Prevents secretion from ER": "ER_RETENTION", "Microbody targeting signal": "PTS1"}


def _ranges(txt, key):
    import re
    return [(int(a), int(b), n) for a, b, n in re.findall(key + r' (\d+)\.\.(\d+);(?: /note="([^"]*)")?', str(txt))] if isinstance(txt, str) else []


def build():
    import re
    import pandas as pd
    info = {}; H = cbb.hgnc()
    hg = pd.read_csv(os.path.join(HERE, "hgnc_complete_set.txt"), sep="\t", dtype=str, usecols=["uniprot_ids", "ensembl_gene_id"]).dropna()
    u2e = {}
    for u, e in zip(hg.uniprot_ids, hg.ensembl_gene_id):
        for x in u.split("|"):
            u2e.setdefault(x, e)
    d = pd.read_csv(raw("uniprot", "uniprot_human_reviewed.tsv.gz"), sep="\t", dtype=str)
    prot, iso_rows = {}, []
    for r in d.itertuples(index=False):
        acc, gene = r[0], r[1]; ens = u2e.get(acc) or (cbb.to_ens(gene) if isinstance(gene, str) else None)
        if not ens:
            continue
        sig = _ranges(r[3], "SIGNAL"); tra = _ranges(r[4], "TRANSIT"); tm = _ranges(r[5], "TRANSMEM"); mot = _ranges(r[7], "MOTIF")
        lip = str(r[8]) if isinstance(r[8], str) else ""
        tags = sorted({TAGS[n] for _, _, n in mot if n in TAGS})
        v = dict(uniprot=acc, signal=[a for a in sig][:1], transit=[[a, b, n] for a, b, n in tra][:1], tm=len(tm), intramem=int(isinstance(r[6], str)),
                 motifs=tags, motif_pos=[[a, b, TAGS[n]] for a, b, n in mot if n in TAGS], gpi=("GPI-anchor" in lip), lipid=bool(lip),
                 location=str(r[9])[:400] if isinstance(r[9], str) else "", pfam=[x for x in str(r[10]).split(";") if x.startswith("PF")] if isinstance(r[10], str) else [],
                 length=int(r[12]) if str(r[12]).isdigit() else None)
        tx = re.findall(r"(ENST\d+)(?:\.\d+)?(?: \[([A-Z0-9]+-\d+)\])?", str(r[2])) if isinstance(r[2], str) else []
        v["transcripts"] = [[t, i or (acc + "-1")] for t, i in tx]
        # isoforms whose alternative sequence touches the address tag (signal / transit peptide, or a localization motif)
        tag_spans = [(a, b, "signal") for a, b, _ in sig] + [(a, b, "transit") for a, b, _ in tra] + [(a, b, t) for a, b, t in v["motif_pos"]]
        hits = collections.defaultdict(set)
        for a, b, note in _ranges(r[11], "VAR_SEQ"):
            isos = re.findall(r"isoform ([^,)]+)", note or "")
            for ta, tb, kind in tag_spans:
                if a <= tb and b >= ta:
                    for iso in isos:
                        hits[iso.strip()].add(kind)
        v["isoforms_changing_tag"] = {k: sorted(x) for k, x in hits.items()}
        v["has_var_seq"] = isinstance(r[11], str)
        prot.setdefault(ens, v)
    json.dump(prot, open(os.path.join(cbb.LAY, "addr_uniprot.json"), "w"))
    info["uniprot"] = dict(proteins=len(prot), signal=sum(bool(v["signal"]) for v in prot.values()), transit_mito=sum(bool(v["transit"]) and "Mitochondrion" in v["transit"][0][2] for v in prot.values()),
                           tm=sum(v["tm"] > 0 for v in prot.values()), nls=sum("NLS" in v["motifs"] for v in prot.values()), er_retention=sum("ER_RETENTION" in v["motifs"] for v in prot.values()),
                           pts1=sum("PTS1" in v["motifs"] for v in prot.values()), gpi=sum(v["gpi"] for v in prot.values()),
                           genes_with_isoforms_changing_tag=sum(bool(v["isoforms_changing_tag"]) for v in prot.values()))
    # MitoCarta
    m = pd.read_excel(raw("mitocarta", "Human.MitoCarta3.0.xls"), sheet_name="A Human MitoCarta3.0")
    mc = {}
    for e, sub, pth in zip(m["EnsemblGeneID_mapping_version_20200130"], m["MitoCarta3.0_SubMitoLocalization"], m["MitoCarta3.0_MitoPathways"]):
        for x in str(e).split("|"):
            if x.startswith("ENSG"):
                mc[x] = dict(sub=str(sub), pathways=str(pth)[:200])
    json.dump(mc, open(os.path.join(cbb.LAY, "addr_mitocarta.json"), "w")); info["mitocarta"] = dict(genes=len(mc))
    # ENCODE fractions: genes x (cell, fraction, rep) TPM
    files = sorted(f for f in os.listdir(os.path.join(RAW, "encode_frac", "raw")) if f.endswith(".tsv"))
    cols, mats, genes = [], [], None
    for f in files:
        q = pd.read_csv(raw("encode_frac", f), sep="\t", usecols=["gene_id", "TPM"]); q = q[q.gene_id.str.startswith("ENSG")]
        q["g"] = q.gene_id.str.split(".").str[0]; q = q.groupby("g").TPM.sum()
        genes = q.index if genes is None else genes.intersection(q.index); mats.append(q); cols.append(f.rsplit("_", 1)[0])
    X = np.column_stack([q.reindex(genes).values for q in mats]).astype(np.float32)
    np.savez_compressed(os.path.join(cbb.LAY, "rna_fractions_encode.npz"), X=X, rows=np.array(genes, dtype=str), cols=np.array(cols, dtype=str))
    info["encode_fractions"] = dict(genes=int(len(genes)), samples=cols)
    # HPA transcripts by tissue: mean nTPM per tissue
    import zipfile
    z = zipfile.ZipFile(raw("hpa_tx", "transcript_rna_tissue.tsv.zip")); t = pd.read_csv(z.open(z.namelist()[0]), sep="\t")
    ncols = [c for c in t.columns if c.startswith("nTPM.")]; tissues = sorted({c.split(".")[1] for c in ncols})
    T = np.column_stack([t[[c for c in ncols if c.split(".")[1] == ts]].mean(1).values for ts in tissues]).astype(np.float32)
    np.savez_compressed(os.path.join(cbb.LAY, "isoform_hpa_tissue.npz"), X=T, gene=t.ensgid.to_numpy(dtype=str), transcript=t.enstid.to_numpy(dtype=str), cols=np.array(tissues, dtype=str))
    info["hpa_transcripts"] = dict(transcripts=int(len(t)), genes=int(t.ensgid.nunique()), tissues=len(tissues))
    json.dump(info, open(os.path.join(RAW, "build_info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


def bank():
    lf, facts = {}, []

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    info = json.load(open(os.path.join(RAW, "build_info.json")))
    for key, src in (("uniprot", "uniprot"), ("mitocarta", "mitocarta"), ("hpa_tx", "hpa"), ("encode_frac", "encode_frac")):
        add(cbb.layer_fact(f"ADDR_RAW_{key.upper()}", 1, src, os.path.join(RAW, key, "raw"), "downloaded files", "-", "as published"), f"ADDR_RAW_{key.upper()}")
    for name, src, fn, rk, rows, cols in (("ADDR_UNIPROT", "uniprot", "addr_uniprot.json", "uniprot", "genes (Ensembl) -> address tags, domains, isoforms", "-"),
                                          ("ADDR_MITOCARTA", "mitocarta", "addr_mitocarta.json", "mitocarta", "genes", "sub-mitochondrial location, pathways"),
                                          ("RNA_FRACTIONS_ENCODE", "encode_frac", "rna_fractions_encode.npz", "encode_frac", "genes", "K562/HepG2 fraction x replicate TPM"),
                                          ("ISOFORM_HPA_TISSUE", "hpa", "isoform_hpa_tissue.npz", "hpa_tx", "transcripts (gene, transcript)", "tissues, mean nTPM")):
        add(cbb.layer_fact(name, 1, src, os.path.join(cbb.LAY, fn), rows, cols, "see addressbank.py", inputs=(lf[f"ADDR_RAW_{rk.upper()}"],), tier="derived"), name)
    t = cbb.now(); P = json.load(open(os.path.join(cbb.LAY, "addr_uniprot.json"))); src = cbb.source("uniprot")
    for g, v in P.items():
        lab = dict(signal_peptide=bool(v["signal"]), mito_transit=bool(v["transit"]), transmembrane=v["tm"], motifs=v["motifs"], gpi=v["gpi"],
                   isoforms_changing_tag=v["isoforms_changing_tag"], uniprot=v["uniprot"])
        facts.append(Fact(g, "address/label", lab, t[:10], t, "reported", src, inputs=(lf["ADDR_UNIPROT"],)))
    n = cbb.bank().add_many(facts)
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers=lf, build=info)
    json.dump(rep, open(R("outputs", "cellbrain", "address_report.json"), "w"), indent=1, default=int); print(json.dumps(rep, default=int)[:1500])


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
