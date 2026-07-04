"""PREFLIGHT — verify every data source is in place before the (long) assemble, and say exactly which
layer each missing file would silently disable. Informational only (never fails the run).

Three classes:
  CRITICAL   the final model can't build without it.
  LAYER      an optional layer skips gracefully if missing (the model still builds).
  RUNTIME    fetched over the network during the run (checked for reachability, not presence).
"""
import os, urllib.request
from pathlib import Path
H=Path("data/external_data/human")
DRIVE=[Path("/content/drive/MyDrive/virtual_cell_data"),
       Path("/content/drive/MyDrive/virtual_cell_data/human_raw"),
       Path("/content/drive/MyDrive/virtual_cell_data/expression_geo")]

def present(names, patterns=()):
    for n in names:
        if (H/n).exists() and (H/n).stat().st_size>500: return f"{n}"
        for d in DRIVE:
            if (d/n).exists(): return f"{n} (Drive)"
    for pat in patterns:
        for d in [H]+DRIVE:
            if d.exists() and any(d.glob(pat)): return f"{pat} (glob)"
    return None

# (label, filenames, which layer/consumer, class)
CRITICAL=[
 ("gene_info",        ["gene_info.gz"],                     "gene id backbone (all layers)"),
 ("STRING aliases",   ["string_aliases.txt.gz"],            "ENSP->symbol mapping (build, abundance, flux)"),
 ("Human-GEM",        ["human_gem.txt"],                    "metabolic network -> flux / metabolites / dFBA"),
 ("CollecTRI",        ["collectri.tsv"],                    "regulatory network (reg edges, attractors)"),
 ("DepMap essentiality",["CRISPRGeneEffect.csv"],           "essentiality + co-dependency"),
]
LAYER=[
 ("ARCHS4/GEO h5",    ["archs4_human_gene.h5","human_gene_v2.h5"], "co-expression + context nets + #1 condition multipliers", ("archs4*.h5","*human*gene*.h5")),
 ("PaxDb abundance",  ["paxdb_human.txt","9606-WHOLE_ORGANISM-integrated.txt"], "protein concentration -> flux Vmax", ("*paxdb*.txt","*WHOLE_ORGANISM*integrated*")),
 ("Perturb-seq",      ["perturbseq_gwps_bulk.h5ad"],        "Model-4 + causal regulome", ("perturbseq_*.h5ad",)),
 ("ReMap",            ["remap.bed","remap.bed.gz"],         "TF binding -> causal regulome", ()),
 ("refGene",          ["refGene.txt.gz"],                   "TSS for ReMap / GTEx", ()),
 ("cell-type expr",   ["celltype_expression.csv"],          "concentration #2/#4 + tissue", ()),
 ("NicheNet",         ["nichenet_ligand_target_matrix.rds"],"ligand->target signaling", ()),
 ("CCLE expr/mut",    ["CCLE_expression.csv"],              "biomarkers", ()),
 ("CORUM",            ["corum.txt","corum.zip"],            "protein complexes (optional; ComplexPortal covers)", ()),
 ("AlphaFold",        ["af_human.tar"],                     "structure lens (optional)", ("*.pdb","AF-*",)),
]
RUNTIME=[
 ("FluxProfilingREGP exchange fluxes","https://raw.githubusercontent.com/Radboud-YuLab/FluxProfilingREGP/main/data/exchange_fluxes/output/exch_fluxes_MCF10A_NCI60.csv","absolute exchange -> flux_medium (absolute flux)"),
 ("DLKcat kcat","https://raw.githubusercontent.com/SysBioChalmers/DLKcat/master/DeeplearningApproach/Data/database/Kcat_combination_0918.json","measured kcat anchors -> kinetics"),
 ("gene2go","https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz","GO molecular function -> de-darking"),
]

def reachable(url):
    try:
        req=urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r: return r.status in (200,301,302)
    except Exception:
        try:
            with urllib.request.urlopen(url, timeout=15) as r: r.read(1); return True
        except Exception: return False

def main():
    print("="*70+"\nPREFLIGHT — data source verification\n"+"="*70)
    miss_crit=0
    print("\n[CRITICAL — model won't build without these]")
    for label,names,role in CRITICAL:
        p=present(names); print(f"  {'OK ' if p else 'MISSING':7} {label:22} {p or '-- '+role}")
        if not p: miss_crit+=1
    print("\n[LAYERS — skip gracefully if missing; model still builds]")
    miss_layer=[]
    for label,names,role,pats in LAYER:
        p=present(names,pats); print(f"  {'OK ' if p else 'skip':7} {label:22} {p or '('+role+')'}")
        if not p: miss_layer.append(label)
    print("\n[RUNTIME fetches — reachability check]")
    for label,url,role in RUNTIME:
        ok=reachable(url); print(f"  {'OK ' if ok else 'UNREACH':7} {label:34} {'' if ok else '('+role+')'}")
    print("\n"+"-"*70)
    if miss_crit: print(f"!! {miss_crit} CRITICAL file(s) missing — assemble will fail. Restore them before running.")
    else: print("All CRITICAL sources present. Model will build.")
    if miss_layer: print(f"Layers that will skip (optional): {', '.join(miss_layer)}")
    print("-"*70)

if __name__=="__main__":
    main()
