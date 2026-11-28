"""CELLBRAIN BANK -- both data groups in the brain's memory bank, so the brain reaches every dataset through cited facts.

The bank is the brain's own (Nikku03/money, brain/bank.py: append-only, bitemporal, every fact with its source and
tier, derived facts cite their inputs), imported from a read-only checkout of the brain (env BRAIN_REPO, default
../nikku03/money-neuro beside this repository; its commit is recorded in every source). It lives in
rem/atlas/_cache/cellbrain/bank (git-ignored: some inputs are CC BY-NC-SA or of unclear licence).

WHAT IT HOLDS
  LAYER-<NAME>   one fact per dataset ('layer'): the file or folder, its sha256, shape, what rows and columns are,
                 units, the transform applied, its group, source and licence. The numbers stay in the file (millions of
                 values); the fact is the brain's handle on them, and `layer()` refuses a file whose sha changed.
                 Pointer layers are files as pulled; derived layers are built here (rows = Ensembl gene ids) and cite
                 the layers they came from.
  <ENSG...>      every GENCODE 46 protein-coding / lncRNA gene and every gene a screen or DepMap measures: identity
                 (symbol, biotype, chromosome, TSS, HGNC), memberships (protein complexes, Reactome pathways, HPA
                 subcellular locations, drugs acting on it with stage) and which layers hold it.
  SCREEN-<X>     each knockdown / drug screen: cell type, design, records, source, licence.
  BIO-<TERM>     each ENCODE biosample term: classification, ENCODE slims, RNA-seq experiments, cCRE annotations.
  CT-<CL id> / TISSUE-<UBERON id>   each CELLxGENE cell type and tissue: name, cells, pairs.
  COMPLEX-<id>, PATHWAY-<id>, DRUG-<CHEMBL id>   members; drug name, stage, targets, indications.
GROUP 1 (what knocking a gene down, or a drug, does): K562 genome-wide and essential Perturb-seq, RPE1 (Replogle
  2022), HepG2, Jurkat (Nadig 2025), the reprocessed CRISPRi collection (HCT116, HEK293T, iPSC x4, hESC x2), sci-Plex3
  drugs, all as 128-component records on K562's genes; K562 Perturb-seq in full gene space; LINCS shRNA / CRISPR
  knockdowns; DepMap gene effects; HPA protein atlas, cell-line RNA, locations, interactions; Complex Portal; Reactome;
  Open Targets drugs; Geneformer embeddings, ARCHS4 expression, Tahoe drug responses.
GROUP 2 (what genes and DNA do in every cell type; the AlphaGenome 2 data): CELLxGENE aggregates (human, mouse),
  ENCODE RNA-seq (human, mouse), GTEx v8, FANTOM5 CAGE (human, mouse), ENCODE Registry v4 cCREs (human, mouse; DNase
  and mark-only kinds), promoter open-chromatin per gene x biosample (derived), WGBS promoter methylation, GTEx v11
  QTLs, IBDverse sc-eQTLs, Open Targets credible sets and studies, TraitGym, ENCODE-rE2G benchmarks, the use-3 map.

Run: python3 -P rem/atlas/cellbrain_bank.py build | check | card ENTITY
"""

from __future__ import annotations
import collections
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
BRAIN = os.environ.get("BRAIN_REPO") or R("..", "nikku03", "money-neuro")
sys.path.insert(0, BRAIN)
from brain.bank import Bank, Fact  # noqa: E402  (the brain's own bank)

HOME = os.path.join(HERE, "_cache", "cellbrain")
BANKD, LAY = os.path.join(HOME, "bank"), os.path.join(HOME, "layers")
C = os.path.join(HERE, "_cache")
AG = os.path.join(C, "ag2bank")
CB4 = os.path.join(C, "cellbank4", "data")
OUTD = R("outputs", "cellbrain")


def brain_commit():
    import subprocess
    r = subprocess.run(["git", "-C", BRAIN, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


SRC = {   # key: name, url, licence (local_only: kept out of git, results from it too)
    "replogle2022": ("Replogle et al. 2022 Cell: genome-wide Perturb-seq (K562 GWPS, K562 essential, RPE1)", "https://doi.org/10.1016/j.cell.2022.05.013", "CC BY 4.0"),
    "nadig2025": ("Nadig et al. 2025 Nat Genet: Perturb-seq in HepG2 and Jurkat", "https://doi.org/10.1038/s41588-025-02169-3", "CC BY 4.0"),
    "crispyx": ("Uniformly reprocessed CRISPRi Perturb-seq collection (figshare 34002315; crispyx)", "https://figshare.com/articles/34002315", "CC BY-NC-SA 4.0"),
    "sciplex": ("Srivatsan et al. 2020 Science: sci-Plex3 (scPerturb, Zenodo 13350497)", "https://zenodo.org/records/13350497", "CC BY 4.0"),
    "lincs": ("LINCS L1000 2020 (CLUE): shRNA and CRISPR knockdown signatures", "https://clue.io/data", "CC BY 4.0"),
    "depmap": ("DepMap CRISPRGeneEffect (Chronos)", "https://depmap.org/portal/data_page", "CC BY 4.0"),
    "hpa": ("Human Protein Atlas: proteinatlas.tsv, rna_celline, subcellular_location, interaction_consensus", "https://www.proteinatlas.org/about/download", "CC BY-SA 3.0"),
    "complexportal": ("EMBL-EBI Complex Portal, human complexes (2026-01-14)", "https://www.ebi.ac.uk/complexportal", "CC0"),
    "reactome": ("Reactome pathways (ReactomePathways.gmt)", "https://reactome.org/download-data", "CC BY 4.0"),
    "opentargets_drugs": ("Open Targets Platform 26.09: drugs, mechanisms, indications (memory_bank/cell_v6)", "https://platform.opentargets.org/downloads", "CC0"),
    "geneformer": ("Geneformer gene embeddings (Theodoris et al. 2023)", "https://huggingface.co/ctheodoris/Geneformer", "Apache-2.0"),
    "archs4": ("ARCHS4 human gene expression v2 (6,244 samples read)", "https://archs4.org", "see source"),
    "tahoe": ("Tahoe-100M differential expression (tahoe-de-rhaister)", "https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister", "see source"),
    "hgnc": ("HGNC complete set", "https://www.genenames.org/download", "CC0"),
    "gencode46": ("GENCODE release 46 basic annotation (TSS)", "https://www.gencodegenes.org/human/release_46.html", "open"),
    "cellxgene": ("CZ CELLxGENE Discover gene-expression aggregates (WMG)", "https://cellxgene.cziscience.com/gene-expression", "CC BY 4.0"),
    "encode": ("ENCODE portal: RNA-seq gene quantifications; Registry v4 cCREs", "https://www.encodeproject.org", "ENCODE data use: free"),
    "gtex": ("GTEx v8 tissue medians; GTEx v11 SuSiE QTLs", "https://gtexportal.org/home/downloads", "GTEx open access"),
    "fantom5": ("FANTOM5 reprocessed CAGE gene TPM", "https://fantom.gsc.riken.jp/5/", "CC BY 4.0"),
    "loyfer2023": ("Loyfer et al. 2023 Nature: WGBS methylation atlas (GEO GSE186458)", "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458", "GEO public"),
    "ibdverse": ("IBDverse sc-eQTL credible sets (eQTL Catalogue r8 beta)", "https://www.ebi.ac.uk/eqtl/", "CC BY 4.0"),
    "opentargets_genetics": ("Open Targets Platform 26.03: credible sets and studies", "https://platform.opentargets.org/downloads", "CC0"),
    "traitgym": ("TraitGym (Benegas et al. 2025)", "https://huggingface.co/datasets/songlab/TraitGym", "see source"),
    "encode_re2g": ("ENCODE-rE2G CRISPRi enhancer-gene benchmarks (EngreitzLab/CRISPR_comparison)", "https://github.com/EngreitzLab/CRISPR_comparison", "MIT"),
    "e2gmap": ("e2gmap.py use-3 switch-to-gene map (this repository)", "rem/atlas/e2gmap.py", "derived"),
}
LOCAL_ONLY = {"crispyx", "tahoe", "archs4", "traitgym"}
SCREENS = {  # record screen key -> layer name, cell type, design, source
    "A": ("KO_K562_GW", "K562", "genome-wide (TRAIN/TEST split)", "replogle2022"), "B": ("KO_K562_ESS", "K562", "essential", "replogle2022"),
    "C": ("KO_RPE1", "RPE1", "essential", "replogle2022"), "D": ("KO_HEPG2", "HepG2", "essential", "nadig2025"),
    "E": ("KO_JURKAT", "Jurkat", "essential", "nadig2025"), "S:RPE1": ("KO_RPE1_X", "RPE1", "essential (reprocessed)", "crispyx"),
    "S:HepG2": ("KO_HEPG2_X", "HepG2", "essential (reprocessed)", "crispyx"), "S:Jurkat": ("KO_JURKAT_X", "Jurkat", "essential (reprocessed)", "crispyx"),
    "S:HCT116": ("KO_HCT116", "HCT116", "genome-wide", "crispyx"), "S:HEK293T": ("KO_HEK293T", "HEK293T", "genome-wide", "crispyx"),
    "S:iPSC-N": ("KO_IPSC_N", "iPSC", "genome-wide", "crispyx"), "S:iPSC-Fsf": ("KO_IPSC_FSF", "iPSC", "genome-wide sub-screen", "crispyx"),
    "S:iPSC-Fsnf": ("KO_IPSC_FSNF", "iPSC", "genome-wide sub-screen", "crispyx"), "S:iPSC-Fts": ("KO_IPSC_FTS", "iPSC", "targeted", "crispyx"),
    "S:hESC-P": ("KO_HESC_P", "H1 hESC", "growth / pluripotency", "crispyx"), "S:hESC-V": ("KO_HESC_V", "H1 hESC", "effect strength", "crispyx"),
    "F": ("DRUG_K562_SCIPLEX", "K562", "188 compounds x 4 doses", "sciplex"),
}
P = lambda *p: os.path.join(*p)
POINTERS = [  # name, group, source, path, rows, cols, units / content
    ("CX_HUMAN", 2, "cellxgene", P(AG, "cellxgene", "cellxgene_human.npz"), "60,810 genes (Ensembl)", "6,790 tissue x cell type pairs (pairs.json)", "CSR: me = mean log-norm expression among expressing cells (uint8 x 0.04), pc = share expressing (uint8/255)"),
    ("CX_HUMAN_PAIRS", 2, "cellxgene", P(AG, "cellxgene", "cellxgene_human_pairs.json"), "pairs", "-", "tissue / cell type ids and names, cells, labelled"),
    ("CX_MOUSE", 2, "cellxgene", P(AG, "cellxgene", "cellxgene_mouse.npz"), "49,413 genes (Ensembl mouse)", "2,044 pairs", "as CX_HUMAN"),
    ("CX_MOUSE_PAIRS", 2, "cellxgene", P(AG, "cellxgene", "cellxgene_mouse_pairs.json"), "pairs", "-", "as CX_HUMAN_PAIRS"),
    ("RNA_ENCODE_HUMAN", 2, "encode", P(AG, "bulk", "encode_rna_human.npz"), "1,517 experiments", "58,735 genes (Ensembl)", "log1p TPM, float16"),
    ("RNA_ENCODE_HUMAN_META", 2, "encode", P(AG, "bulk", "encode_rna_human_meta.json"), "experiments", "-", "biosample, term id, assay"),
    ("RNA_ENCODE_MOUSE", 2, "encode", P(AG, "bulk", "encode_rna_mouse.npz"), "452 experiments", "genes (Ensembl mouse)", "log1p TPM"),
    ("RNA_ENCODE_MOUSE_META", 2, "encode", P(AG, "bulk", "encode_rna_mouse_meta.json"), "experiments", "-", "biosample"),
    ("RNA_GTEX_V8", 2, "gtex", P(AG, "bulk", "gtex_v8_tissue_median.npz"), "54 tissues", "56,200 genes (Ensembl)", "log1p median TPM"),
    ("CAGE_FANTOM5_HUMAN", 2, "fantom5", P(AG, "bulk", "fantom5_cage_human.npz"), "1,829 samples", "23,274 genes (symbols)", "log1p TPM"),
    ("CAGE_FANTOM5_HUMAN_META", 2, "fantom5", P(AG, "bulk", "fantom5_cage_human_meta.json"), "samples", "-", "description, library"),
    ("CAGE_FANTOM5_MOUSE", 2, "fantom5", P(AG, "bulk", "fantom5_cage_mouse.npz"), "1,073 samples", "genes (symbols)", "log1p TPM"),
    ("METH_PROMOTER", 2, "loyfer2023", P(AG, "methylation", "loyfer2023_promoter_beta.npz"), "253 samples", "39,323 genes (Ensembl)", "promoter beta x 250 (uint8; 255 = NA)"),
    ("CCRE_HUMAN", 2, "encode", P(AG, "ccre", "human"), "2,348,854 registry elements", "1,678 annotations (1,325 DNase, 353 mark-only)", "class codes 0-8 DNase-based, 9-15 marks"),
    ("CCRE_MOUSE", 2, "encode", P(AG, "ccre", "mouse"), "926,843 elements", "393 annotations", "as CCRE_HUMAN"),
    ("QTL_GTEX_V11", 2, "gtex", P(AG, "genetics", "gtex_v11"), "eQTL, sQTL, apaQTL SuSiE tars", "50 tissues", "credible sets with PIP"),
    ("SCEQTL_IBDVERSE", 2, "ibdverse", P(AG, "genetics", "ibdverse"), "72 cell-type datasets", "-", "credible sets (parquet)"),
    ("GWAS_OPENTARGETS", 2, "opentargets_genetics", P(AG, "genetics", "opentargets_26.03"), "credible sets (200 parts) + study table", "-", "variant PIPs per credible set"),
    ("TRAITGYM", 2, "traitgym", P(AG, "genetics", "traitgym"), "variants", "-", "causal / control variants"),
    ("E2G_BENCHMARKS", 2, "encode_re2g", P(AG, "enhancer_gene"), "CRISPRi enhancer-gene pairs", "-", "Regulated labels"),
    ("E2G_MAP", 2, "e2gmap", P(AG, "e2g_map"), "10 cell lines", "-", "element, gene, score"),
    ("TSS_GENCODE46", 2, "gencode46", P(AG, "raw", "gencode46_tss.json"), "39,323 genes", "-", "Ensembl, symbol, biotype, chromosome, TSS"),
    ("KO_RECORDS_V8", 1, "replogle2022", P(C, "cell_v8", "records.npz"), "37,234 records", "128 components (+ basis 8,248 K562 genes x 128)", "int8 codes x scale"),
    ("KO_RECORDS_V8_META", 1, "replogle2022", P(C, "cell_v8", "records_meta.json"), "records", "-", "gene, screen, split, strength, drug"),
    ("GWPS_K562_H5AD", 1, "replogle2022", P(C, "perturbseq", "K562_gwps_normalized_bulk_01.h5ad"), "11,258 knockdowns", "8,248 genes", "normalised pseudo-bulk"),
    ("ESS_K562_H5AD", 1, "replogle2022", P(CB4, "ess.h5ad"), "2,285 knockdowns", "8,563 genes", "normalised pseudo-bulk"),
    ("DEPMAP_CSV", 1, "depmap", P(CB4, "CRISPRGeneEffect.csv"), "cell lines", "genes", "Chronos gene effect"),
    ("HPA_PROTEINATLAS", 1, "hpa", P(CB4, "proteinatlas.tsv.zip"), "genes", "annotations", "classes, specificity"),
    ("HPA_RNA_CELLLINE", 1, "hpa", P(CB4, "rna_celline.tsv.zip"), "gene x cell line", "-", "nTPM"),
    ("HPA_LOCATION", 1, "hpa", P(CB4, "subcellular_location.tsv.zip"), "genes", "-", "subcellular locations"),
    ("HPA_INTERACTIONS", 1, "hpa", P(CB4, "interaction_consensus.tsv.zip"), "gene pairs", "-", "consensus interactions"),
    ("REACTOME_GMT", 1, "reactome", P(CB4, "ReactomePathways.gmt.zip"), "pathways", "genes (symbols)", "membership"),
    ("COMPLEX_PORTAL", 1, "complexportal", R("outputs", "orphan", "science_data", "complex_members.parquet"), "complex members", "-", "Ensembl members"),
    ("OT_DRUGS", 1, "opentargets_drugs", R("memory_bank", "cell_v6", "drugs.json"), "13,128 drugs", "-", "name, stage, mechanisms, indications"),
    ("OT_TARGETS", 1, "opentargets_drugs", R("memory_bank", "cell_v6", "targets.json"), "17,095 genes", "-", "drugs per target, tractability, safety"),
    ("SCIPLEX_COMPOUNDS", 1, "sciplex", R("memory_bank", "cell_v6", "sciplex_compounds.json"), "188 compounds", "-", "targets, top matching knockdowns"),
    ("LINCS_SH_SRC", 1, "lincs", P(C, "cellbank9", "src_LINCS-sh.npz"), "1,668 genes", "8,248 K562 genes", "signature"),
    ("LINCS_XPR_SRC", 1, "lincs", P(C, "cellbank9", "src_LINCS-xpr.npz"), "genes", "8,248 K562 genes", "signature"),
    ("GENEFORMER_SRC", 1, "geneformer", P(C, "cellbank9", "nb_Geneformer.npz"), "4,967 genes", "768", "embedding"),
    ("ARCHS4_SRC", 1, "archs4", P(C, "cellbank9", "nb_ARCHS4.npz"), "4,985 genes", "6,244 samples", "expression"),
    ("TAHOE_SRC", 1, "tahoe", P(C, "cellbank9", "nb_Tahoe.npz"), "159 genes", "65,218 conditions", "differential expression"),
    ("HGNC", 1, "hgnc", P(HERE, "hgnc_complete_set.txt"), "genes", "-", "symbols, aliases, Ensembl, Entrez"),
]


# ------------------------------------------------------------------------------------------------ helpers
def sha_path(p):
    h = hashlib.sha256(); n = 0
    files = [p] if os.path.isfile(p) else sorted(os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs)
    for f in files:
        if os.path.isdir(p):
            h.update(os.path.relpath(f, p).encode() + b"\0")
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b); n += len(b)
    return h.hexdigest(), n, len(files)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source(key):
    n, u, l = SRC[key]
    return {"name": n, "url": u, "licence": l, "local_only": key in LOCAL_ONLY, "brain_commit": brain_commit()}


_HG = None


def hgnc():
    """symbol / alias / previous symbol / Entrez -> Ensembl, and Ensembl -> symbol (HGNC approved)."""
    global _HG
    if _HG is None:
        import pandas as pd
        h = pd.read_csv(P(HERE, "hgnc_complete_set.txt"), sep="\t", dtype=str, usecols=["hgnc_id", "symbol", "alias_symbol", "prev_symbol", "entrez_id", "ensembl_gene_id", "locus_group"])
        h = h[h.ensembl_gene_id.notna()]
        sym, ali, ent, e2s = {}, {}, {}, {}
        for r in h.itertuples():
            sym[r.symbol] = r.ensembl_gene_id; e2s[r.ensembl_gene_id] = r.symbol
            if isinstance(r.entrez_id, str):
                ent[r.entrez_id] = r.ensembl_gene_id
            for col in (r.alias_symbol, r.prev_symbol):
                if isinstance(col, str):
                    for a in col.split("|"):
                        ali.setdefault(a, r.ensembl_gene_id)
        _HG = dict(sym=sym, ali=ali, ent=ent, e2s=e2s, hgnc={r.ensembl_gene_id: r.hgnc_id for r in h.itertuples()})
    return _HG


def to_ens(s):
    H = hgnc(); s = str(s)
    if s.startswith("ENSG"):
        return s.split(".")[0]
    return H["sym"].get(s) or H["ali"].get(s)


def save_layer(name, **arrays):
    os.makedirs(LAY, exist_ok=True); p = P(LAY, name.lower() + ".npz")
    arrays = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else v) for k, v in arrays.items()}   # no pickled arrays
    np.savez_compressed(p, **arrays); return p


# ------------------------------------------------------------------------------------------------ derived layers
def build_ko():
    """Every screen's records as 128-d vectors (code x scale) on K562's genes, rows = Ensembl; plus the basis."""
    z = np.load(P(C, "cell_v8", "records.npz")); m = json.load(open(P(C, "cell_v8", "records_meta.json")))
    codes, scale = z["codes"].astype(np.float32), z["scale"].astype(np.float32)
    genes = json.load(open(P(C, "cell_v8", "genes.json")))
    out = {}
    by = collections.defaultdict(list)
    for i, s in enumerate(m["screen"]):
        by[s].append(i)
    for s, ix in by.items():
        name = SCREENS[s][0]
        if s == "F":
            rows = [f"{m['drug'][str(i)]['compound']}|{m['drug'][str(i)]['dose_nM']:g}nM" for i in ix]
        else:
            rows = [(m["ens"][i] or to_ens(m["gene"][i]) or f"SYMBOL:{m['gene'][i]}") for i in ix]
        st = np.array([float(m["strength"].get(str(i), np.nan)) for i in ix], np.float32)
        sp = np.array([m["split"][i] for i in ix])
        p = save_layer(name, X=(codes[ix] * scale[ix, None]).astype(np.float16), rows=np.array(rows), split=sp, strength=st,
                       symbols=np.array([m["gene"][i] for i in ix]))
        out[name] = (p, s, len(ix))
    order = sorted(genes, key=lambda g: genes[g]["row"])
    p = save_layer("KO_BASIS", X=z["basis"], rows=np.array([genes[g]["ens"] or to_ens(g) or f"SYMBOL:{g}" for g in order]), symbols=np.array(order))
    out["KO_BASIS"] = (p, "basis", len(order))
    return out


def _h5_matrix(path, name):
    import h5py
    with h5py.File(path, "r") as f:
        X = f["X"][:].astype(np.float16)
        obs = [x.decode() for x in f["obs"][f["obs"].attrs["_index"]][:]]
        var = [x.decode() for x in f["var"][f["var"].attrs["_index"]][:]]
    rows = [o.split("_")[-1] if o.split("_")[-1].startswith("ENSG") else (to_ens(o.split("_")[1]) or o) for o in obs]
    return save_layer(name, X=X, rows=np.array(rows), cols=np.array(var), labels=np.array(obs))


def build_lincs_etc():
    out = {}
    basis_genes = np.load(P(LAY, "ko_basis.npz"))["rows"]
    for name, f, key in (("LINCS_SH", "src_LINCS-sh.npz", "M"), ("LINCS_XPR", "src_LINCS-xpr.npz", "M"),
                         ("EMB_GENEFORMER", "nb_Geneformer.npz", "V"), ("EXPR_ARCHS4", "nb_ARCHS4.npz", "V"), ("PERT_TAHOE", "nb_Tahoe.npz", "V")):
        z = np.load(P(C, "cellbank9", f))
        rows = np.array([to_ens(g) or f"SYMBOL:{g}" for g in z["genes"]])
        kw = dict(X=z[key].astype(np.float16), rows=rows, symbols=z["genes"])
        if key == "M":
            kw["cols"] = basis_genes
        out[name] = (save_layer(name, **kw), f, len(rows))
    return out


def build_depmap():
    import pandas as pd
    d = pd.read_csv(P(CB4, "CRISPRGeneEffect.csv"), index_col=0)
    H = hgnc(); rows, keep = [], []
    for j, c in enumerate(d.columns):
        m_ = re.match(r"(.+) \((\d+)\)", c)
        e = (H["ent"].get(m_.group(2)) or to_ens(m_.group(1))) if m_ else to_ens(c)
        if e:
            rows.append(e); keep.append(j)
    X = d.values[:, keep].T.astype(np.float16)
    return save_layer("DEP_DEPMAP", X=X, rows=np.array(rows), cols=np.array(d.index.astype(str)))


def build_hpa():
    import pandas as pd
    with zipfile.ZipFile(P(CB4, "rna_celline.tsv.zip")) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), sep="\t", usecols=["Gene", "Cell line", "nTPM"])
    piv = d.pivot_table(index="Gene", columns="Cell line", values="nTPM", aggfunc="mean")
    p1 = save_layer("EXPR_HPA_CELLLINES", X=np.log1p(piv.values).astype(np.float16), rows=np.array(piv.index), cols=np.array(piv.columns.astype(str)))
    with zipfile.ZipFile(P(CB4, "interaction_consensus.tsv.zip")) as z:
        e = pd.read_csv(z.open(z.namelist()[0]), sep="\t")
    p2 = save_layer("PPI_HPA", a=e.iloc[:, 0].astype(str).values, b=e.iloc[:, 1].astype(str).values, datasets=e.iloc[:, 2].astype(str).values)
    return p1, p2


def build_sets():
    """Gene sets as JSON layers: {set id: {name, members (Ensembl)}}."""
    import pandas as pd
    H = hgnc(); out = {}
    cm = pd.read_parquet(R("outputs", "orphan", "science_data", "complex_members.parquet"))
    comp = collections.defaultdict(set)
    for c, e in zip(cm.complex_id, cm.ensembl_gene_id):
        if isinstance(e, str) and e.startswith("ENSG"):
            comp[c].add(e)
    out["SETS_COMPLEX"] = {c: {"name": c, "members": sorted(m)} for c, m in comp.items() if len(m) >= 2}
    pw = {}
    with zipfile.ZipFile(P(CB4, "ReactomePathways.gmt.zip")) as z:
        for line in z.read(z.namelist()[0]).decode().splitlines():
            f = line.split("\t"); mem = sorted({to_ens(g) for g in f[2:]} - {None})
            if 5 <= len(f) - 2 <= 500 and len(mem) >= 2:
                pw[f[1]] = {"name": f[0], "members": mem}
    out["SETS_PATHWAY"] = pw
    with zipfile.ZipFile(P(CB4, "subcellular_location.tsv.zip")) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype=str)
    loc = collections.defaultdict(set)
    for g, a, b in zip(d["Gene"], d["Main location"], d["Additional location"]):
        for col in (a, b):
            if isinstance(col, str):
                for x in col.split(";"):
                    loc[x.strip()].add(g)
    out["SETS_LOCATION"] = {k: {"name": k, "members": sorted(v)} for k, v in loc.items() if len(v) >= 2}
    tg = json.load(open(R("memory_bank", "cell_v6", "targets.json"))); dr = json.load(open(R("memory_bank", "cell_v6", "drugs.json")))
    by = collections.defaultdict(set)
    for g, v in tg.items():
        for d_ in v.get("drugs", []):
            by[d_["chembl"]].add(g)
    out["SETS_DRUGTARGET"] = {c: {"name": (dr.get(c) or {}).get("name") or c, "members": sorted(m)} for c, m in by.items() if m}
    paths = {}
    for k, v in out.items():
        os.makedirs(LAY, exist_ok=True); p = P(LAY, k.lower() + ".json"); json.dump(v, open(p, "w")); paths[k] = (p, len(v))
    return paths


def build_promoters():
    """Promoter open (any DNase-based element of class PLS..CA-only within 1 kb of the TSS) per gene x annotation."""
    sys.path.insert(0, HERE)
    import importlib.util
    s = importlib.util.spec_from_file_location("e2gmap", P(HERE, "e2gmap.py")); e2 = importlib.util.module_from_spec(s); s.loader.exec_module(e2)
    reg = e2.Registry(); tss = json.load(open(P(AG, "raw", "gencode46_tss.json")))
    win = [reg.overlap(ch, pos - 1000, pos + 1000) for gid, name, gt, ch, pos in tss]
    owner = np.full(len(reg.chrom), -1, np.int64); gi, ei = [], []
    for i, w in enumerate(win):
        gi.extend([i] * len(w)); ei.extend(w.tolist())
    gi, ei = np.array(gi), np.array(ei)
    ann = [b for b in reg.meta if b.get("kind") == "dnase"]
    O = np.zeros((len(tss), len(ann)), bool)
    for j, b in enumerate(ann):
        sh = np.load(P(e2.CC, "shards", b["annotation"] + ".npz")); op = np.zeros(len(reg.chrom), bool); op[sh["idx"][sh["cls"] <= 6]] = True
        hit = op[ei]; O[np.unique(gi[hit]), j] = True
    del owner
    return save_layer("PROMOTER_OPEN_HUMAN", bits=np.packbits(O, axis=1), n_cols=np.array(len(ann)), rows=np.array([t[0] for t in tss]),
                      cols=np.array([b["annotation"] for b in ann]), col_biosample=np.array([b["biosample"] for b in ann]))


# ------------------------------------------------------------------------------------------------ facts
def layer_fact(name, group, src, path, rows, cols, units, inputs=(), tier="reported", extra=None):
    sha, n, nf = sha_path(path)
    v = {"path": os.path.relpath(path, R()), "sha256": sha, "bytes": n, "files": nf, "rows": rows, "cols": cols, "units": units,
         "group": group, "licence": SRC[src][2], "local_only": src in LOCAL_ONLY, **(extra or {})}
    t = now()
    return Fact(f"LAYER-{name}", "layer", v, t[:10], t, tier, source(src), inputs=tuple(inputs))


def entity_facts(lf):
    """Genes, screens, biosamples, cell types, tissues, complexes, pathways, drugs."""
    t = now(); F = []
    H = hgnc(); tss = json.load(open(P(AG, "raw", "gencode46_tss.json")))
    sets = {k: json.load(open(P(LAY, k.lower() + ".json"))) for k in ("SETS_COMPLEX", "SETS_PATHWAY", "SETS_LOCATION", "SETS_DRUGTARGET")}
    mem = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, S in sets.items():
        for sid, d in S.items():
            for g in d["members"]:
                e = g if g.startswith("ENSG") else to_ens(g)
                if e:
                    mem[e][k].append(sid if k != "SETS_DRUGTARGET" else d["name"])
    presence = collections.defaultdict(list)
    for name, ref in lf.items():
        p = P(LAY, name.lower() + ".npz")
        if name.startswith(("KO_", "DEP_", "EXPR_", "LINCS", "EMB_", "PERT_", "PROMOTER")) and os.path.exists(p):
            for e in set(np.load(p)["rows"].tolist()):
                presence[e].append(name)
    for nm, fn in (("CX_HUMAN", lambda: np.load(P(AG, "cellxgene", "cellxgene_human.npz"))["genes"]),
                   ("RNA_ENCODE_HUMAN", lambda: np.load(P(AG, "bulk", "encode_rna_human.npz"))["cols"]),
                   ("RNA_GTEX_V8", lambda: np.load(P(AG, "bulk", "gtex_v8_tissue_median.npz"))["cols"]),
                   ("METH_PROMOTER", lambda: np.load(P(AG, "methylation", "loyfer2023_promoter_beta.npz"))["genes"])):
        for e in set(str(x).split(".")[0] for x in fn()):
            presence[e].append(nm)
    genes = {g[0]: dict(symbol=g[1], biotype=g[2], chrom=g[3], tss=g[4]) for g in tss}
    measured = {e for e, ls in presence.items() if any(n.startswith(("KO_", "DEP_", "EXPR_HPA", "LINCS")) for n in ls)}
    for e in measured:                                      # genes a screen, DepMap or HPA measures, outside the TSS list
        if e.startswith("ENSG") and e not in genes:
            genes[e] = dict(symbol=H["e2s"].get(e), biotype=None, chrom=None, tss=None)
    gs = source("gencode46")
    for e, d in genes.items():
        if e not in presence and e not in mem:
            continue                                        # a GENCODE gene no dataset measures: no entity
        d["hgnc"] = H["hgnc"].get(e)
        F.append(Fact(e, "identity", d, t[:10], t, "reported", gs))
        m_ = mem.get(e, {})
        for k, attr in (("SETS_COMPLEX", "member/complex"), ("SETS_PATHWAY", "member/pathway"), ("SETS_LOCATION", "location"), ("SETS_DRUGTARGET", "drugs")):
            if m_.get(k):
                F.append(Fact(e, attr, sorted(set(m_[k])), t[:10], t, "derived", {"name": "cellbrain_bank:sets"}, inputs=(lf[k],)))
        if presence.get(e):
            F.append(Fact(e, "data/layers", sorted(set(presence[e])), t[:10], t, "derived", {"name": "cellbrain_bank:presence"},
                          inputs=tuple(lf[n] for n in sorted(set(presence[e])) if n in lf)))
    meta8 = json.load(open(P(C, "cell_v8", "records_meta.json"))); cnt = collections.Counter(meta8["screen"])
    for s, (name, cell, design, src) in SCREENS.items():
        F.append(Fact(f"SCREEN-{name}", "screen", {"record_key": s, "cell": cell, "design": design, "records": cnt[s]}, t[:10], t, "reported",
                      source(src), inputs=(lf[name],) if name in lf else ()))
    ccre = json.load(open(P(AG, "ccre", "human", "meta.json")))["biosamples"]
    rnam = json.load(open(P(AG, "bulk", "encode_rna_human_meta.json")))["meta"]
    slp = R("outputs", "trait2cell", "biosample_slims.json"); sl = json.load(open(slp))["terms"] if os.path.exists(slp) else {}
    bio = collections.defaultdict(lambda: dict(term_id=None, classification=None, rna_experiments=0, dnase_annotations=0, mark_annotations=0))
    for b in ccre:
        d = bio[b["biosample"]]; d["term_id"] = b["term_id"]; d["classification"] = b["classification"]
        d["dnase_annotations" if b.get("kind") == "dnase" else "mark_annotations"] += 1
    for b, tid, cl in zip(rnam["biosample"], rnam["term_id"], rnam["classification"]):
        d = bio[b]; d["rna_experiments"] += 1; d["term_id"] = d["term_id"] or tid; d["classification"] = d["classification"] or cl
    es = source("encode")
    for name, d in bio.items():
        d = dict(d, name=name, slims=sl.get(name))
        F.append(Fact(f"BIO-{name}", "biosample", d, t[:10], t, "reported", es, inputs=(lf["CCRE_HUMAN"], lf["RNA_ENCODE_HUMAN_META"])))
    pairs = json.load(open(P(AG, "cellxgene", "cellxgene_human_pairs.json")))["pairs"]
    ct, ti = collections.defaultdict(lambda: dict(name=None, cells=0, tissues=0)), collections.defaultdict(lambda: dict(name=None, cells=0, cell_types=0))
    for p_ in pairs:
        c = ct[p_["cell_type"]]; c["name"] = p_["cell_type_name"]; c["tissues"] += 1; c["cells"] += int(p_["cells"])
        s_ = ti[p_["tissue"]]; s_["name"] = p_["tissue_name"]; s_["cell_types"] += 1
        if p_["cell_type"] == "CL:0000000":
            s_["cells"] = int(p_["cells"])
    cs = source("cellxgene")
    for k, d in ct.items():
        F.append(Fact(f"CT-{k}", "cell_type", d, t[:10], t, "reported", cs, inputs=(lf["CX_HUMAN_PAIRS"],)))
    for k, d in ti.items():
        F.append(Fact(f"TISSUE-{k}", "tissue", d, t[:10], t, "reported", cs, inputs=(lf["CX_HUMAN_PAIRS"],)))
    for k, attr, src in (("SETS_COMPLEX", "complex", "complexportal"), ("SETS_PATHWAY", "pathway", "reactome")):
        for sid, d in sets[k].items():
            F.append(Fact(f"{attr.upper()}-{sid}", attr, {"name": d["name"], "members": d["members"], "n": len(d["members"])}, t[:10], t,
                          "reported", source(src), inputs=(lf[k],)))
    dr = json.load(open(R("memory_bank", "cell_v6", "drugs.json"))); tg = sets["SETS_DRUGTARGET"]
    ds = source("opentargets_drugs")
    for c, d in dr.items():
        F.append(Fact(f"DRUG-{c}", "drug", {"name": d.get("name"), "type": d.get("type"), "max_stage": d.get("max_stage"),
                                             "targets": (tg.get(c) or {}).get("members", []), "indications": d.get("indications", [])},
                      t[:10], t, "reported", ds, inputs=(lf["OT_DRUGS"],)))
    return F


def unique_shards(facts):
    """Two entity names that map to the same shard file get a short hash suffix (the bank keys shards by a safe name)."""
    seen = {}
    for f in facts:
        k = re.sub(r"[^A-Z0-9._-]", "_", f.entity.upper()); seen.setdefault(k, set()).add(f.entity)
    clash = {e for v in seen.values() if len(v) > 1 for e in v}
    if not clash:
        return facts
    out = []
    for f in facts:
        if f.entity in clash:
            d = f.to_dict(); d["entity"] = f.entity + "~" + hashlib.sha256(f.entity.encode()).hexdigest()[:4]; d.pop("id"); d.pop("retrieved_at")
            f = Fact.from_dict(d)
        out.append(f)
    print(f"  {len(clash)} entity names shared a shard name; suffixed", flush=True)
    return out


def build():
    t0 = time.time(); os.makedirs(BANKD, exist_ok=True); os.makedirs(OUTD, exist_ok=True)
    facts, lf = [], {}

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    for name, g, src, path, rows, cols, units in POINTERS:
        if not os.path.exists(path):
            print(f"  missing, not in the bank: {name} ({path})", flush=True); continue
        add(layer_fact(name, g, src, path, rows, cols, units), name)
    print(f"pointer layers: {len(lf)} ({time.time() - t0:.0f}s)", flush=True)
    ko = build_ko()
    for name, (p, s, n) in ko.items():
        src = SCREENS[s][3] if s in SCREENS else "replogle2022"
        add(layer_fact(name, 1, src, p, f"{n} records (Ensembl rows; drug|dose for sci-Plex)", "128 components" if name != "KO_BASIS" else "128",
                       "response in K562 Perturb-seq components (code x scale)", inputs=(lf["KO_RECORDS_V8"],), tier="derived",
                       extra={"screen": s}), name)
    for name, path, src in (("KO_K562_GENESPACE", P(C, "perturbseq", "K562_gwps_normalized_bulk_01.h5ad"), "GWPS_K562_H5AD"),
                            ("KO_K562ESS_GENESPACE", P(CB4, "ess.h5ad"), "ESS_K562_H5AD")):
        p = P(LAY, name.lower() + ".npz")
        if not os.path.exists(p):
            p = _h5_matrix(path, name)
        add(layer_fact(name, 1, "replogle2022", p, "knockdowns (Ensembl)", "measured genes (Ensembl)", "normalised pseudo-bulk expression change",
                       inputs=(lf[src],), tier="derived"), name)
    for name, (p, f, n) in build_lincs_etc().items():
        src = {"LINCS_SH": "lincs", "LINCS_XPR": "lincs", "EMB_GENEFORMER": "geneformer", "EXPR_ARCHS4": "archs4", "PERT_TAHOE": "tahoe"}[name]
        add(layer_fact(name, 1, src, p, f"{n} genes (Ensembl)", "features", "as pulled by cellbank9", tier="derived",
                       inputs=(lf[{"lincs": "LINCS_SH_SRC" if "SH" in name else "LINCS_XPR_SRC", "geneformer": "GENEFORMER_SRC", "archs4": "ARCHS4_SRC", "tahoe": "TAHOE_SRC"}[src]],)), name)
    p = P(LAY, "dep_depmap.npz"); p = p if os.path.exists(p) else build_depmap()
    add(layer_fact("DEP_DEPMAP", 1, "depmap", p, "genes (Ensembl, via Entrez)", "cell lines (ModelID)", "Chronos gene effect", inputs=(lf["DEPMAP_CSV"], lf["HGNC"]), tier="derived"), "DEP_DEPMAP")
    if not os.path.exists(P(LAY, "ppi_hpa.npz")):
        build_hpa()
    add(layer_fact("EXPR_HPA_CELLLINES", 1, "hpa", P(LAY, "expr_hpa_celllines.npz"), "genes (Ensembl)", "cell lines", "log1p nTPM", inputs=(lf["HPA_RNA_CELLLINE"],), tier="derived"), "EXPR_HPA_CELLLINES")
    add(layer_fact("PPI_HPA", 1, "hpa", P(LAY, "ppi_hpa.npz"), "interaction edges (Ensembl a, b)", "-", "consensus datasets", inputs=(lf["HPA_INTERACTIONS"],), tier="derived"), "PPI_HPA")
    for k, (p, n) in build_sets().items():
        src = {"SETS_COMPLEX": "complexportal", "SETS_PATHWAY": "reactome", "SETS_LOCATION": "hpa", "SETS_DRUGTARGET": "opentargets_drugs"}[k]
        inp = {"SETS_COMPLEX": "COMPLEX_PORTAL", "SETS_PATHWAY": "REACTOME_GMT", "SETS_LOCATION": "HPA_LOCATION", "SETS_DRUGTARGET": "OT_TARGETS"}[k]
        add(layer_fact(k, 1, src, p, f"{n} sets", "members (Ensembl)", "membership", inputs=(lf[inp], lf["HGNC"]), tier="derived"), k)
    p = P(LAY, "promoter_open_human.npz"); p = p if os.path.exists(p) else build_promoters()
    add(layer_fact("PROMOTER_OPEN_HUMAN", 2, "encode", p, "39,323 GENCODE 46 genes (Ensembl)", "1,325 DNase-based cCRE annotations",
                   "promoter open: a PLS..CA-only element within 1 kb of the TSS (packbits)", inputs=(lf["CCRE_HUMAN"], lf["TSS_GENCODE46"]), tier="derived"), "PROMOTER_OPEN_HUMAN")
    print(f"derived layers built ({time.time() - t0:.0f}s)", flush=True)
    ents = entity_facts(lf); print(f"entity facts: {len(ents):,} ({time.time() - t0:.0f}s)", flush=True)
    ents = unique_shards(ents); bank = Bank(BANKD); n = bank.add_many(facts + ents)
    rep = dict(built=time.strftime("%Y-%m-%d"), brain_commit=brain_commit(), bank_dir=os.path.relpath(BANKD, R()), facts_written=n,
               facts_total=len(facts) + len(ents), layers={f.entity[6:]: {k: f.value[k] for k in ("path", "sha256", "bytes", "group", "licence", "local_only", "rows", "cols")}
                                                           for f in facts},
               entities=dict(collections.Counter(f.entity.split("-")[0] if not f.entity.startswith("ENSG") else "GENE" for f in ents)),
               seconds=round(time.time() - t0))
    json.dump(rep, open(P(OUTD, "bank_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=1))


def bank():
    return Bank(BANKD)


def layer(name, check=True, _seen={}):
    """The brain's handle on a dataset: the layer fact; refuses a file whose sha changed since it was banked."""
    fs = bank().facts(f"LAYER-{name}", "layer")
    if not fs:
        raise KeyError(f"LAYER-{name} not in the bank")
    f = fs[-1]; path = R(f.value["path"])
    if check and name not in _seen:
        if sha_path(path)[0] != f.value["sha256"]:
            raise RuntimeError(f"LAYER-{name}: {path} changed since it was banked")
        _seen[name] = True
    return f, path


def check():
    b = bank(); bad = 0; ents = b.entities()
    print(f"{len(ents):,} entities")
    lay = [e for e in ents if e.startswith("LAYER-")]
    for e in lay:
        try:
            layer(e[6:])
        except Exception as ex:
            bad += 1; print("  ", ex)
    rng = np.random.default_rng(0)
    for e in rng.choice(ents, size=min(500, len(ents)), replace=False):
        v = b.verify(e)
        if v:
            bad += 1; print("  ", e, v[:2])
    print(f"{len(lay)} layers sha-checked, 500 entities verified: {'OK' if not bad else f'{bad} problems'}")
    return bad


def card(ent):
    b = bank()
    for f in b.facts(ent):
        v = json.dumps(f.value)
        print(f"  {f.attr:28s} {v[:160]}{'...' if len(v) > 160 else ''}  [{f.tier}; {f.source.get('name', '')[:50]}; {f.ref}]")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "check":
        sys.exit(1 if check() else 0)
    elif cmd == "card":
        card(sys.argv[2])
