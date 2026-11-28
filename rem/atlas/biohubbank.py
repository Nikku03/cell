"""BIOHUBBANK -- two Biohub datasets added to the memory bank (cellbrain bank), as the user asked after reading biohub.org.

1. OpenCell (Cho et al. 2022 Science; Chan Zuckerberg Biohub; https://opencell.sf.czbiohub.org/download).
   1,310 endogenously tagged HEK293T lines. Processed datasets (CSV, with READMEs): library metadata; protein
   abundance (RNA TPM, protein nM and copies per cell, whole HEK293T proteome); protein-protein interactions from
   1,260 IP-MS pull-downs (p value, enrichment, interaction and abundance stoichiometry); localization annotations
   (graded 3 prominent / 2 clear / 1 weak). Licence: CC BY-SA 4.0 (AWS Open Data registry entry for czb-opencell).
2. Psoriasis CRISPR screen (Zhao, Shih et al. 2026 Nat Commun, "AI-guided CRISPR screening reveals therapeutic targets
   in psoriasis"; PMC13473102; supplementary files from Europe PMC). Genome-wide CRISPR knockout in primary human
   keratinocytes (two donors, ~77,000 guides, ~19,000 genes), sorted on IL17RA: MAGeCK FDR and LFC for knockouts
   enriched in IL17RA-low and IL17RA-high cells (Supplementary Data 1, 18,785 rows); VirtualCRISPR top 10; proteomics
   of ALOX5 and OXTR knockout keratinocytes (Data 2-3); mouse imiquimod skin proteomics under anti-IL17RA, zileuton,
   cligosiban (Data 4-6). Raw data: PRIDE PXD071833, BioProject PRJNA1451949 (not pulled). Licence: CC BY-NC-ND
   (article and supplements): kept local, never committed, and nothing derived from it is published.
Downloads are untrusted: each source lands in its own raw folder; this code reads them, it never runs them.
Layers go to rem/atlas/_cache/cellbrain/layers and LAYER facts, gene facts and a SCREEN entity to the cellbrain bank.

Run: python3 -P rem/atlas/biohubbank.py pull | build | bank      (each step committed before it ran)
"""

from __future__ import annotations
import collections
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.normpath(os.path.join(HERE, "..", "..", *p))
sys.path.insert(0, HERE)
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("cellbrain_bank", os.path.join(HERE, "cellbrain_bank.py"))
cbb = importlib.util.module_from_spec(_s); _s.loader.exec_module(cbb)
from brain.bank import Fact  # noqa: E402

RAW = os.path.join(HERE, "_cache", "biohub")
OC_RAW, PS_RAW = os.path.join(RAW, "opencell", "raw"), os.path.join(RAW, "psoriasis", "raw")
OC_URL = "https://opencell.sf.czbiohub.org/data/datasets/{}"
OC_FILES = ["opencell-library-metadata", "opencell-protein-abundance", "opencell-protein-interactions", "opencell-localization-annotations"]
PMC = "PMC13473102"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/{}/{}"
SUPP = {"screen": "41467_2026_75249_MOESM3_ESM.xlsx", "alox5_ko": "41467_2026_75249_MOESM4_ESM.xlsx", "oxtr_ko": "41467_2026_75249_MOESM5_ESM.xlsx",
        "mouse_anti_il17ra": "41467_2026_75249_MOESM6_ESM.xlsx", "mouse_zileuton": "41467_2026_75249_MOESM7_ESM.xlsx",
        "mouse_cligosiban": "41467_2026_75249_MOESM8_ESM.xlsx"}
cbb.SRC["opencell"] = ("Cho et al. 2022 Science: OpenCell, endogenous tagging of 1,310 human proteins in HEK293T (Chan Zuckerberg Biohub)",
                       "https://opencell.sf.czbiohub.org/download", "CC BY-SA 4.0")
cbb.SRC["psoriasis2026"] = ("Zhao, Shih et al. 2026 Nat Commun: AI-guided CRISPR screening reveals therapeutic targets in psoriasis (PMC13473102)",
                            "https://doi.org/10.1038/s41467-026-75249-5", "CC BY-NC-ND 4.0")
cbb.LOCAL_ONLY.add("psoriasis2026")


def _get(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    b = cbb_get(url); open(path + ".part", "wb").write(b); os.replace(path + ".part", path)
    return len(b)


def cbb_get(url):
    import importlib.util as iu
    s = iu.spec_from_file_location("ag2bank", os.path.join(HERE, "ag2bank.py")); ag = iu.module_from_spec(s); s.loader.exec_module(ag)
    return ag._get(url, timeout=900)


def pull():
    rep = {}
    for f in OC_FILES:
        for name in (f + ".csv", f + "-readme.csv"):
            rep[name] = _get(OC_URL.format(name), os.path.join(OC_RAW, name))
    rep["supp.zip"] = _get(EPMC.format(PMC, "supplementaryFiles"), os.path.join(PS_RAW, "supp.zip"))
    rep["fulltext.xml"] = _get(EPMC.format(PMC, "fullTextXML"), os.path.join(PS_RAW, "fulltext.xml"))
    print(json.dumps(rep, indent=1))


def _sheet(name, idx=0):
    import openpyxl
    z = zipfile.ZipFile(os.path.join(PS_RAW, "supp.zip"))
    wb = openpyxl.load_workbook(io.BytesIO(z.read(name)), read_only=True, data_only=True)
    return [r for r in wb.worksheets[idx].iter_rows(values_only=True)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def build():
    import pandas as pd
    out = {}
    # ---- OpenCell
    loc = pd.read_csv(os.path.join(OC_RAW, "opencell-localization-annotations.csv"), dtype=str)
    sets = collections.defaultdict(lambda: {"grade3": [], "grade2": [], "grade1": []})
    for r in loc.itertuples():
        e = r.ensg_id if isinstance(r.ensg_id, str) and r.ensg_id.startswith("ENSG") else cbb.to_ens(r.target_name)
        if not e:
            continue
        for g, col in (("grade3", r.annotations_grade_3), ("grade2", r.annotations_grade_2), ("grade1", r.annotations_grade_1)):
            if isinstance(col, str):
                for c in col.split(";"):
                    if c.strip():
                        sets[c.strip()][g].append(e)
    S = {k: {"name": k, "members": sorted(set(v["grade3"] + v["grade2"])), "weak": sorted(set(v["grade1"])), "grade3": sorted(set(v["grade3"]))}
         for k, v in sets.items()}
    p = os.path.join(cbb.LAY, "sets_opencell_location.json"); json.dump(S, open(p, "w")); out["SETS_OPENCELL_LOCATION"] = (p, len(S))
    ppi = pd.read_csv(os.path.join(OC_RAW, "opencell-protein-interactions.csv"))
    a = ppi.target_ensg_id.astype(str).values; b = ppi.interactor_ensg_id.astype(str).values
    out["PPI_OPENCELL"] = (cbb.save_layer("PPI_OPENCELL", a=a, b=b, a_symbol=ppi.target_gene_name.astype(str).values, b_symbol=ppi.interactor_gene_name.astype(str).values,
                                          neglog10_p=ppi.pval.astype(np.float32).values, enrichment=ppi.enrichment.astype(np.float32).values,
                                          interaction_stoichiometry=ppi.interaction_stoichiometry.astype(np.float32).values,
                                          abundance_stoichiometry=ppi.abundance_stoichiometry.astype(np.float32).values), len(ppi))
    ab = pd.read_csv(os.path.join(OC_RAW, "opencell-protein-abundance.csv"))
    out["EXPR_OPENCELL_HEK293T"] = (cbb.save_layer("EXPR_OPENCELL_HEK293T", rows=ab.ensg_id.astype(str).values, symbols=ab.gene_name.astype(str).values,
                                                   X=np.c_[ab.hek_rna_tpm.astype(float), ab.hek_protein_conc_nm.astype(float), ab.hek_protein_copy_number.astype(float)].astype(np.float32),
                                                   cols=np.array(["rna_tpm", "protein_nM", "protein_copies_per_cell"])), len(ab))
    # ---- psoriasis screen (local only)
    rows = _sheet(SUPP["screen"]); head = rows[0]; body = [r for r in rows[1:] if r and r[0]]
    sym = [str(r[0]) for r in body]; ens = [cbb.to_ens(s) or f"SYMBOL:{s}" for s in sym]
    X = np.array([[_f(r[1]), _f(r[2]), _f(r[3]), _f(r[4])] for r in body], np.float32)
    top = [str(r[0]) for r in _sheet(SUPP["screen"], 1)[1:] if r and r[0]]
    out["SCREEN_PSORIASIS_IL17RA"] = (cbb.save_layer("SCREEN_PSORIASIS_IL17RA", rows=np.array(ens), symbols=np.array(sym), X=X,
                                                     cols=np.array([str(h) for h in head[1:5]]), virtualcrispr_top10=np.array(top)), len(body))
    for key in ("alox5_ko", "oxtr_ko", "mouse_anti_il17ra", "mouse_zileuton", "mouse_cligosiban"):
        rr = _sheet(SUPP[key]); h = [str(x) for x in rr[0]]; body = [r for r in rr[1:] if r and r[0]]
        gi, li, pi = h.index("Genes"), h.index("logFC"), h.index("p.mod")
        g = [str(r[gi]) for r in body]
        e = [cbb.to_ens(x.split(";")[0]) or f"SYMBOL:{x}" for x in g] if not key.startswith("mouse") else [f"MOUSE:{x}" for x in g]
        name = "PROT_PSORIASIS_" + key.upper()
        out[name] = (cbb.save_layer(name, rows=np.array(e), symbols=np.array(g), logfc=np.array([_f(r[li]) for r in body], np.float32),
                                    p_mod=np.array([_f(r[pi]) for r in body], np.float32)), len(body))
    rep = {k: {"rows": v[1], "file": os.path.relpath(v[0], R())} for k, v in out.items()}
    json.dump(rep, open(os.path.join(RAW, "build.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


def bank():
    lf, facts = {}, []

    def add(f, name):
        facts.append(f); lf[name] = f.ref
    for f in OC_FILES:
        add(cbb.layer_fact(f.upper().replace("-", "_"), 1, "opencell", os.path.join(OC_RAW, f + ".csv"), "see README", "see README",
                           open(os.path.join(OC_RAW, f + "-readme.csv")).read()[:600]), f.upper().replace("-", "_"))
    add(cbb.layer_fact("PSORIASIS_SUPPLEMENT", 1, "psoriasis2026", os.path.join(PS_RAW, "supp.zip"), "Supplementary Data 1-9, Source Data 1-2", "-",
                       "xlsx tables as published"), "PSORIASIS_SUPPLEMENT")
    add(cbb.layer_fact("SETS_OPENCELL_LOCATION", 1, "opencell", os.path.join(cbb.LAY, "sets_opencell_location.json"), "compartments", "members (Ensembl; grades 3+2), weak (grade 1)",
                       "OpenCell localization annotations", inputs=(lf["OPENCELL_LOCALIZATION_ANNOTATIONS"],), tier="derived"), "SETS_OPENCELL_LOCATION")
    add(cbb.layer_fact("PPI_OPENCELL", 1, "opencell", os.path.join(cbb.LAY, "ppi_opencell.npz"), "bait-prey pairs (Ensembl)", "-log10 p, enrichment, stoichiometries",
                       "OpenCell IP-MS interactions", inputs=(lf["OPENCELL_PROTEIN_INTERACTIONS"],), tier="derived"), "PPI_OPENCELL")
    add(cbb.layer_fact("EXPR_OPENCELL_HEK293T", 1, "opencell", os.path.join(cbb.LAY, "expr_opencell_hek293t.npz"), "genes (Ensembl)", "rna_tpm, protein_nM, copies per cell",
                       "HEK293T RNA and absolute protein abundance", inputs=(lf["OPENCELL_PROTEIN_ABUNDANCE"],), tier="derived"), "EXPR_OPENCELL_HEK293T")
    add(cbb.layer_fact("SCREEN_PSORIASIS_IL17RA", 1, "psoriasis2026", os.path.join(cbb.LAY, "screen_psoriasis_il17ra.npz"), "18,785 genes (Ensembl where mapped)",
                       "MAGeCK FDR and LFC, IL17RA-low and IL17RA-high", "genome-wide CRISPR knockout, primary human keratinocytes",
                       inputs=(lf["PSORIASIS_SUPPLEMENT"],), tier="derived"), "SCREEN_PSORIASIS_IL17RA")
    for key in ("ALOX5_KO", "OXTR_KO", "MOUSE_ANTI_IL17RA", "MOUSE_ZILEUTON", "MOUSE_CLIGOSIBAN"):
        n = "PROT_PSORIASIS_" + key
        add(cbb.layer_fact(n, 1, "psoriasis2026", os.path.join(cbb.LAY, n.lower() + ".npz"), "proteins", "logFC, moderated p", "proteomics vs control",
                           inputs=(lf["PSORIASIS_SUPPLEMENT"],), tier="derived"), n)
    t = cbb.now(); oc = cbb.source("opencell"); ps = cbb.source("psoriasis2026")
    S = json.load(open(os.path.join(cbb.LAY, "sets_opencell_location.json"))); per = collections.defaultdict(lambda: collections.defaultdict(list))
    for comp, d in S.items():
        for g in d["grade3"]:
            per[g]["prominent"].append(comp)
        for g in set(d["members"]) - set(d["grade3"]):
            per[g]["clear"].append(comp)
        for g in d["weak"]:
            per[g]["weak"].append(comp)
    for g, d in per.items():
        facts.append(Fact(g, "opencell/localization", {k: sorted(v) for k, v in d.items()}, t[:10], t, "derived", oc, inputs=(lf["SETS_OPENCELL_LOCATION"],)))
    z = np.load(os.path.join(cbb.LAY, "ppi_opencell.npz")); part = collections.defaultdict(dict)
    for a, b, sb, p, en in zip(z["a"], z["b"], z["b_symbol"], z["neglog10_p"], z["enrichment"]):
        for x, y, ys in ((str(a), str(b), str(sb)), (str(b), str(a), None)):
            if x.startswith("ENSG") and y != x and (y not in part[x] or p > part[x][y][0]):
                part[x][y] = (float(p), float(en))
    for g, d in part.items():
        top = sorted(d.items(), key=lambda kv: -kv[1][0])[:40]
        facts.append(Fact(g, "opencell/interactors", [[k, round(v[0], 2), round(v[1], 2)] for k, v in top], t[:10], t, "derived", oc,
                          inputs=(lf["PPI_OPENCELL"],), context={"n_partners": len(d), "fields": "partner Ensembl, -log10 p, enrichment"}))
    z = np.load(os.path.join(cbb.LAY, "expr_opencell_hek293t.npz"))
    for g, x in zip(z["rows"], z["X"]):
        if str(g).startswith("ENSG") and np.isfinite(x).any():
            facts.append(Fact(str(g), "opencell/abundance_hek293t", {c: (None if not np.isfinite(v) else round(float(v), 3)) for c, v in zip(z["cols"], x)},
                              t[:10], t, "derived", oc, inputs=(lf["EXPR_OPENCELL_HEK293T"],)))
    z = np.load(os.path.join(cbb.LAY, "screen_psoriasis_il17ra.npz"))
    for g, s, x in zip(z["rows"], z["symbols"], z["X"]):
        if str(g).startswith("ENSG"):
            facts.append(Fact(str(g), "screen/psoriasis_il17ra", {"fdr_il17ra_low": float(x[0]), "lfc_low": float(x[1]), "fdr_il17ra_high": float(x[2]),
                                                                  "lfc_high": float(x[3]), "symbol": str(s)}, t[:10], t, "derived", ps,
                              inputs=(lf["SCREEN_PSORIASIS_IL17RA"],)))
    lo = z["X"][:, 0]; order = np.argsort(lo)
    facts.append(Fact("SCREEN-PSORIASIS_IL17RA", "screen", {
        "cell": "primary human keratinocytes (two donors)", "design": "genome-wide CRISPR knockout, ~77,000 guides, ~19,000 genes, sorted on IL17RA",
        "genes": int(len(lo)), "fdr_low_below_0.05": int((lo < 0.05).sum()), "fdr_high_below_0.05": int((z["X"][:, 2] < 0.05).sum()),
        "validated_hits": ["ALOX5 (zileuton)", "OXTR (cligosiban)"], "virtualcrispr_top10": [str(x) for x in z["virtualcrispr_top10"]],
        "raw_data": ["PRIDE PXD071833", "BioProject PRJNA1451949"]}, t[:10], t, "reported", ps, inputs=(lf["SCREEN_PSORIASIS_IL17RA"],)))
    n = cbb.bank().add_many(facts)
    rep = dict(added=time.strftime("%Y-%m-%d"), facts_written=n, layers={k: v for k, v in lf.items()},
               gene_facts=collections.Counter(f.attr for f in facts if f.entity.startswith("ENSG")),
               licences={"opencell": "CC BY-SA 4.0", "psoriasis2026": "CC BY-NC-ND 4.0 (local only)"})
    json.dump(rep, open(R("outputs", "cellbrain", "biohub_report.json"), "w"), indent=1, default=int)
    print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    {"pull": pull, "build": build, "bank": bank}[sys.argv[1]]()
