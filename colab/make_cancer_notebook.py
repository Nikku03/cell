"""Build colab/cancer_cell.ipynb — the COMPLETE cancer cell in one Colab run.

Takes a tumour genome and plots the whole perturbed cell across EVERY layer we built, ML included:
  molecular (role + ΔΔG structural LOF + domain + DepMap selectivity) · pathway (signed propagation) ·
  regulatory · complexes · metabolic · dependency (DepMap co-essentiality + SL + selective) ·
  independent CellGraph perturbation lens · tissue baseline (emask) · conclusion + HTML plot.

Colab is where the heavy data lives (DepMap matrix, AlphaFold structures on demand, Human-GEM for FBA), so the
layers that are dark in a laptop session light up here. Each layer degrades gracefully if its data is absent.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"
REPO = "https://github.com/Nikku03/cell.git"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}


cells = [
    md("# The complete cancer cell — genome in, whole perturbed cell out",
       "",
       "Give it a tumour's mutations; it plots the **entire perturbed cell** across every layer, then concludes.",
       "The target is just one thing read off the map — the point is the whole altered-cell state.",
       "",
       "| Layer | What it contributes |",
       "|---|---|",
       "| **A molecular** | call each mutation GOF/LOF by role **+ ΔΔG structural stability (ML)** + domain + DepMap selectivity |",
       "| **B pathway** | propagate through the signed causal+signalling net → hyperactive / lost programme |",
       "| **C regulatory** | TF programmes hit |",
       "| **D complexes** | protein complexes losing a subunit |",
       "| **E metabolic** | metabolic genes / FBA flux touched |",
       "| **F dependency (ML)** | DepMap co-essentiality + synthetic-lethal partners + genome-wide selective dependencies |",
       "| **G CellGraph** | independent perturbation lens (agreement = corroboration) |",
       "| **H tissue** | the emask cell-type baseline the generic map lacked |",
       "",
       "> Honest scope carried in the output: topological (direction, not kinetics); GOF/LOF from role+ΔΔG (a "
       "stability-neutral mutation stays ambiguous); a WRN-type context dependency appears only in the "
       "selective-dependency scan, not attributable to the tumour without per-line genotype data."),

    md("## 1. Setup — clone the branch + deps"),
    code(f"!git clone --depth 1 -b {BR} {REPO} 2>/dev/null || (cd cell && git pull)",
         "%cd cell",
         "!pip -q install numpy scipy scikit-learn pandas biopython cobra 2>/dev/null  # biopython: BLOSUM/PDB; cobra: FBA",
         "import sys, os; sys.path.insert(0, 'colab')",
         "os.makedirs('outputs/orphan', exist_ok=True)",
         "print('ready')"),

    md("## 2. Mount Drive — restore the cell + ML models",
       "Restores `cell_complete.json` (36 MB core), the trained **signal_combiner**, the **ΔΔG model**, and the "
       "**DepMap** matrix. Every file is optional — a missing one just dims that layer; the run still completes."),
    code("from google.colab import drive; drive.mount('/content/drive')",
         "import glob, shutil, os",
         "D = '/content/drive/MyDrive'",
         "def grab(name, *pats):",
         "    dst = f'outputs/orphan/{name}'",
         "    if os.path.exists(dst): return True",
         "    for p in pats:",
         "        g = sorted(glob.glob(p, recursive=True), key=lambda x:-os.path.getsize(x))",
         "        if g: shutil.copy(g[0], dst); print('restored', name, 'from', g[0]); return True",
         "    print('  (missing', name, '- its layer will be dimmed)'); return False",
         "grab('cell_complete.json', f'{D}/**/cell_complete.json')",
         "grab('signal_combiner.pkl', f'{D}/**/signal_combiner.pkl')",
         "grab('ddg_model.pkl', f'{D}/**/ddg_model.pkl')",
         "grab('depmap_vecs.npz', f'{D}/**/depmap_vecs.npz')",
         "grab('reactome_pathways.json', f'{D}/**/reactome_pathways.json')",
         "# also restore the rest of the artifact set if present",
         "import persist; persist.restore_from_drive(D)",
         "print('restore done')"),

    md("## 3. Load the cell + confirm which layers are live"),
    code("from complete_cell import CompleteCell",
         "import full_cell_map as fcm",
         "C = CompleteCell()",
         "dm = fcm._depmap()",
         "print('genes:', len(C.name), '| signed causal edges:', sum(len(v) for v in C.causal_out.values()))",
         "print('DepMap matrix:', 'loaded' if dm else 'MISSING', '| combiner:', os.path.exists('outputs/orphan/signal_combiner.pkl'),",
         "      '| ddG model:', os.path.exists('outputs/orphan/ddg_model.pkl'))"),

    md("## 4. Define the tumour genome(s)",
       "Real documented alterations (driver + passengers). Edit freely — add your own tumour. `META` supplies the "
       "residue substitution + UniProt so the **ΔΔG structural layer** can fetch the AlphaFold structure and score "
       "each mutation."),
    code("PANELS = {",
         "  'A375 melanoma': [('BRAF',600),('CDKN2A',58),('TTN',20000)],",
         "  'MSI-H colorectal': [('MLH1',300),('MSH6',1088),('PMS2',600),('TGFBR2',128),('ACVR2A',400),",
         "                       ('RNF43',117),('BAX',41),('BRAF',600),('B2M',50),('JAK1',860),('WRN',577)],",
         "}",
         "# residue + UniProt for ΔΔG (hotspot alleles); genes without an entry just skip the ΔΔG layer",
         "META = fcm.META",
         "TISSUE = fcm.TISSUE",
         "print('tumours:', list(PANELS))"),

    md("## 5. Plot the complete cell — every layer, ML included",
       "This runs all 8 layers per tumour. The ΔΔG layer fetches AlphaFold structures on demand (first run is "
       "slower). The conclusion integrates driver + dependency targets, with the honest gaps stated inline."),
    code("seldep = fcm.selective_dependencies(dm) if dm else []",
         "report = {}",
         "for name, panel in PANELS.items():",
         "    r = fcm.full_map(C, panel, dm, seldep, panel_meta=META, tissue_kw=TISSUE.get(name))",
         "    report[name] = r",
         "    fcm._print(name, r)",
         "import json; json.dump(report, open('outputs/orphan/full_cell_map.json','w'), indent=2)"),

    md("## 6. Render the visual map"),
    code("import cancer_cell_html as h, cancer_cell_map as ccm",
         "# reuse the card renderer on the pathway sub-map of each tumour",
         "sub = {k: {'mutation_calls': v['A_molecular'],",
         "           'hyperactive_pathways': v['B_pathway']['up'], 'lost_pathways': v['B_pathway']['lost'],",
         "           'n_genes_perturbed': v['B_pathway']['n_genes_perturbed'],",
         "           'driver': v['conclusion']['driver'], 'target_note': ''} for k,v in report.items()}",
         "open('outputs/orphan/complete_cancer_cell.html','w').write(h.render(sub))",
         "from IPython.display import HTML, display",
         "display(HTML(open('outputs/orphan/complete_cancer_cell.html').read()))"),

    md("## 7. (optional, GPU) the learned layers not runnable on a laptop",
       "The **learned torch GNN / R-GCN** and the **perturbation→wildtype** predictor need a GPU. Run "
       "`colab/make_gnn_notebook.py`'s cells here to train them and feed their embeddings back into layer G. "
       "Likewise the **metabolic FBA** layer (Human-GEM) and **concentration/structure** data-hunt layers activate "
       "when their downloads are present — see `colab/rprom.py` and `colab/new_data.py`. These are the pieces a "
       "laptop session cannot reach; here they complete the stack."),
    code("# example: run the regulatory->metabolic FBA coupling (PROM) as the live metabolic layer",
         "# import rprom; rprom.run(max_tfs=60)   # uncomment: downloads Human-GEM (~40 MB) + solves biomass FBA",
         "print('optional heavy layers available: rprom (FBA), make_gnn_notebook (learned GNN)')"),

    md("## Save any new artifacts back to Drive"),
    code("import persist; persist.save_to_drive(D)"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 0}

out = Path("colab/cancer_cell.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
