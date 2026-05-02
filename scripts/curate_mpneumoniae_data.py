"""Phase D2 conversion — assemble M. pneumoniae cell_sim input data
from the curated published sources pushed in commit 4519f00.

Output layout mirrors Syn3A:

  cell_sim/data/Mycoplasma_pneumoniae/input_data/
    M_pneumoniae_iJW145.xml         # symlink/copy of the SBML (already there)
    kinetic_params.xlsx             # reactions + gene mapping; Km/kcat blank
    initial_concentrations.xlsx     # protein copies/cell from Maier 2011
                                    # metabolite section LEFT EMPTY (Yus 2009 missing)
    complex_formation.xlsx          # ribosome populated from Maier S7
                                    # generic complexes LEFT EMPTY (Kuhner missing)
    DATA_GAPS.md                    # what's missing and why
    README.md                       # provenance + caveat list

Source files (under memory_bank/data/multiorg_essentiality/raw/mpneumoniae/):
  wodke_2013_iJW145/MODEL1301290000_iJW145.xml          — SBML L2 (306 rxns)
  wodke_2013_iJW145/44320_2013_BFMSB20136_MOESM2_ESM.xls
    sheet 'table S1' — model_ID, reaction_ID, equation, gene_number,
                       enzyme, EC, pathway, reversibility
  maier_2011_proteomics/44320_2011_BFMSB201138_MOESM3_ESM.zip
    Table S2 Protein abundances all conditions.xlsx — 414 proteins x 15 cols
    Table S7 ribosomal proteins.xlsx                 — 46 r-proteins
    Table S4 Translation efficiency...               — 218 turnover rates

What this script does NOT do
----------------------------

* Does NOT fabricate Km / kcat values. Wodke 2013 is constraint-based
  (FBA), not kinetic; reaction-level Km/kcat for M. pneumoniae are
  scattered across BRENDA + papers and not in the pushed source bundle.
  The kinetic_params.xlsx columns 'Value' / 'Units' are left blank with
  a single NEEDS_CURATION row at the top of each pathway sheet.
* Does NOT fabricate metabolite initial concentrations. Yus 2009 is
  Cloudflare-blocked from the acquisition path. The 'Intracellular
  Metabolites' sheet is left empty.
* Does NOT populate non-ribosomal complex_formation entries. Kuhner
  2009 (the canonical organisation-of-cell-by-mass-spec paper) is
  Cloudflare-blocked. Maier 2011 Table S7 covers ribosomal proteins
  only.
* Does NOT wire any of this into cell_sim's runtime. The simulator's
  organism-config refactor is a separate Phase D3 session.

Reproducibility
---------------

Deterministic given the source bundle (commit 4519f00). Re-running
overwrites the output files. Each output cell carries provenance in
the cell_sim/data/Mycoplasma_pneumoniae/input_data/README.md.
"""
from __future__ import annotations

import io
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "memory_bank/data/multiorg_essentiality/raw/mpneumoniae"
OUT = REPO / "cell_sim/data/Mycoplasma_pneumoniae/input_data"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Parse Wodke 2013 reaction-gene-EC-pathway map
# ---------------------------------------------------------------------
WODKE_S1 = RAW / "wodke_2013_iJW145/44320_2013_BFMSB20136_MOESM2_ESM.xls"
print(f"[1] Wodke 2013 MOESM2 Table S1 -> reaction skeleton")
df_s1 = pd.read_excel(WODKE_S1, sheet_name="table S1", header=1)
df_s1.columns = [c.strip() for c in df_s1.columns]
df_s1 = df_s1.rename(columns={
    "model_ID": "model_id",
    "reaction_ID (Yus2009)": "yus2009_id",
    "gene number": "gene_locus_tags",
    "enzyme (short)": "enzyme_short",
    "EC number": "ec_number",
    "reversi-bility": "reversibility",
})
print(f"    {len(df_s1)} reactions, {df_s1.gene_locus_tags.notna().sum()} with gene tags")

# Group by pathway -> separate sheet per pathway (matches Syn3A schema)
pathways = df_s1["pathway"].fillna("Unassigned").unique()
print(f"    pathways: {sorted(pathways)}")


# ---------------------------------------------------------------------
# 2. Parse Maier 2011 protein abundances + ribosomes
# ---------------------------------------------------------------------
MAIER_ZIP = RAW / "maier_2011_proteomics/44320_2011_BFMSB201138_MOESM3_ESM.zip"
maier_tables = {}
with zipfile.ZipFile(MAIER_ZIP) as z:
    for fn in z.namelist():
        if fn.endswith((".xls", ".xlsx")):
            with z.open(fn) as fh:
                data = fh.read()
            try:
                # Pick first sheet
                xl = pd.ExcelFile(io.BytesIO(data))
                df = pd.read_excel(io.BytesIO(data), sheet_name=xl.sheet_names[0])
                # Normalize key by short table label
                if "Table S1 " in fn:
                    maier_tables["S1_quantified"] = df
                elif "Table S2 " in fn:
                    maier_tables["S2_abundances"] = df
                elif "Table S3 " in fn:
                    maier_tables["S3_stress"] = df
                elif "Table S4 " in fn:
                    maier_tables["S4_turnover"] = df
                elif "Table S5 " in fn:
                    maier_tables["S5_leptospira"] = df
                elif "Table S6 " in fn:
                    maier_tables["S6_peptides"] = df
                elif "Table S7 " in fn:
                    maier_tables["S7_ribosome"] = df
                elif "Table S8 " in fn:
                    maier_tables["S8_mrna"] = df
            except Exception as e:
                print(f"    WARN {fn}: {e}")
print(f"[2] Maier 2011 tables loaded: {sorted(maier_tables)}")


# ---------------------------------------------------------------------
# 3. Build kinetic_params.xlsx (one sheet per pathway)
# ---------------------------------------------------------------------
print(f"[3] writing kinetic_params.xlsx (Km/kcat NOT populated)")
kinetic_path = OUT / "kinetic_params.xlsx"
with pd.ExcelWriter(kinetic_path, engine="openpyxl") as writer:
    for pw in sorted(pathways):
        sub = df_s1[df_s1["pathway"].fillna("Unassigned") == pw].copy()
        # Match Syn3A schema: 'Reaction Name', 'Subsystem', 'Parameter Type',
        # 'Related Species', 'Value', 'Units'
        rows = []
        for _, r in sub.iterrows():
            rows.append({
                "Reaction Name": r["enzyme_short"] or r["model_id"],
                "Subsystem": pw,
                "Parameter Type": "kcat (per s, NEEDS_CURATION)",
                "Related Species": r["gene_locus_tags"],
                "Value": None,
                "Units": "1/s",
                # Extra columns preserved from Wodke for traceability
                "wodke_model_id": r["model_id"],
                "yus2009_id": r["yus2009_id"],
                "ec_number": r["ec_number"],
                "reversibility": r["reversibility"],
                "equation": r["equation"],
            })
        out_df = pd.DataFrame(rows)
        # Excel sheet names: max 31 chars, no [ ] : / \ ? *
        if isinstance(pw, str):
            sheet_name = pw[:31].replace("/", "-").replace("\\", "-")
            sheet_name = sheet_name.replace(":", "-").replace("?", "")
            sheet_name = sheet_name.replace("*", "").replace("[", "(").replace("]", ")")
        else:
            sheet_name = "Unassigned"
        out_df.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"    -> {kinetic_path}")


# ---------------------------------------------------------------------
# 4. Build initial_concentrations.xlsx
# ---------------------------------------------------------------------
print(f"[4] writing initial_concentrations.xlsx")
init_path = OUT / "initial_concentrations.xlsx"

# 4a. Comparative Proteomics (from Maier 2011 Table S2, control condition)
maier_s2 = maier_tables.get("S2_abundances", pd.DataFrame()).copy()
maier_s1 = maier_tables.get("S1_quantified", pd.DataFrame()).copy()
proteomics_rows = []
if not maier_s2.empty:
    for _, r in maier_s2.iterrows():
        mpn_id = str(r.get("MPN ID", "")).strip()
        if not mpn_id or not mpn_id.startswith("MPN"):
            continue
        copies = r.get("Control (copies/cell)")
        proteomics_rows.append({
            "Locus Tag": mpn_id,
            "Gene Name": "",   # not in Maier S2; would need GenBank join
            "Gene Product": "",
            "Protein Length": "",
            "Exp. Ptn Cnt": copies,
            "Essentiality": "",
            "Primary Function": "",
            "Localization": "",
        })
prot_df = pd.DataFrame(proteomics_rows)
print(f"    proteomics rows: {len(prot_df)}")

with pd.ExcelWriter(init_path, engine="openpyxl") as writer:
    prot_df.to_excel(writer, sheet_name="Comparative Proteomics", index=False)
    pd.DataFrame(columns=["Metabolite name", "Met ID", "KEGG ID", "InChI key", "Conc (mM)"]).to_excel(
        writer, sheet_name="Simulation Medium", index=False
    )
    pd.DataFrame(columns=["Metabolite name", "Met ID", "KEGG ID", "InChI key", "Init Conc (mM)"]).to_excel(
        writer, sheet_name="Intracellular Metabolites", index=False
    )
    pd.DataFrame(columns=["LocusTag", "free", "ribosome", "degradosome", "total"]).to_excel(
        writer, sheet_name="mRNA Count", index=False
    )
print(f"    -> {init_path}")


# ---------------------------------------------------------------------
# 5. Build complex_formation.xlsx (ribosome only)
# ---------------------------------------------------------------------
print(f"[5] writing complex_formation.xlsx (ribosome only; non-ribosomal complexes EMPTY)")
cmplx_path = OUT / "complex_formation.xlsx"
maier_s7 = maier_tables.get("S7_ribosome", pd.DataFrame())

complexes_rows = []
ribo_rows = []
if not maier_s7.empty:
    print(f"    Maier S7 ribosomal proteins: {len(maier_s7)}")
    members = []
    stoichs = []
    init_count = []
    for _, r in maier_s7.iterrows():
        mpn = str(r.get("MPN ID", "")).strip()
        if not mpn:
            continue
        members.append(mpn)
        stoichs.append(1)
        # Use direct quantified column if available else indirect
        for col in r.index:
            if "direct quantified" in str(col).lower():
                init_count.append(r[col])
                break
    if members:
        # 'n/d' (not detected) appears in some direct-quantified rows;
        # coerce to numeric and drop non-numeric.
        init_numeric = pd.to_numeric(pd.Series(init_count), errors="coerce").dropna()
        median_init = int(init_numeric.median()) if len(init_numeric) else None
        complexes_rows.append({
            "Name": "Ribosome",
            "Genes Products": ",".join(members),
            "Stoichiometries": ",".join(str(s) for s in stoichs),
            "Init. Count": median_init,
            "Assembly Pathway": "Maier 2011 Table S7",
            "PDB Structures": "",
            "How Cal Count": "median of direct-quantified copies/cell across r-proteins (Maier 2011 'n/d' rows excluded)",
        })

with pd.ExcelWriter(cmplx_path, engine="openpyxl") as writer:
    pd.DataFrame(complexes_rows, columns=[
        "Name", "Genes Products", "Stoichiometries", "Init. Count",
        "Assembly Pathway", "PDB Structures", "How Cal Count",
    ]).to_excel(writer, sheet_name="Complexes", index=False)
    pd.DataFrame(columns=["Name", "Gene Product", "Domain"]).to_excel(
        writer, sheet_name="ABC Transporter", index=False
    )
    pd.DataFrame(columns=["locusNum", "No. of TMDs"]).to_excel(
        writer, sheet_name="Sole YidC IMP", index=False
    )
    pd.DataFrame(columns=[
        "locusNum", "Gene Name", "Gene Product",
        "How Recruited to Membrane",
        "Peripheral MPs but not diffuse to membrane automatically",
    ]).to_excel(writer, sheet_name="Peri MP in Cyto", index=False)
    pd.DataFrame(columns=["Name", "Init. Count"]).to_excel(
        writer, sheet_name="Predefined Complexes", index=False
    )
print(f"    -> {cmplx_path}")


# ---------------------------------------------------------------------
# 6. Copy SBML to convenient location
# ---------------------------------------------------------------------
print(f"[6] copying iJW145 SBML to input_data/")
sbml_src = RAW / "wodke_2013_iJW145/MODEL1301290000_iJW145.xml"
sbml_dst = OUT / "M_pneumoniae_iJW145.xml"
shutil.copyfile(sbml_src, sbml_dst)
print(f"    -> {sbml_dst}")


print()
print("DONE. Phase D2 conversion produced schema-compatible files.")
print("Next: write DATA_GAPS.md + README.md.")
