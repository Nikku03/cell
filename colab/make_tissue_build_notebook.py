"""Build colab/tissue_build.ipynb — a clean tissue-layer build (the body's cell-cell + cross-organ wiring).

Produces tissue_model.json + tissue_explorer.html: for each organ, the cell types and who-signals-whom
(ligand->receptor), the druggable channels, the downstream response inside receiver cells, and the
cross-tissue endocrine (hormone) axes. Independent of the cell layer, CPU-only, ~minutes.

Improvements over the old build_tissue_model.ipynb: restores the Drive data that powers the richer layers
(dgidb / signor / collectri / nichenet) and SAVES the outputs to Drive so they can be handed off.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-NRqBW"

def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = []

cells.append(md(
    "# Build the tissue layer — the body's cell-cell + cross-organ wiring",
    "",
    "For each organ: the cell types, **who signals whom** (ligand→receptor), which channels are druggable,",
    "the **downstream response inside receiver cells**, and the cross-tissue **endocrine (hormone) axes**.",
    "Independent of the cell layer — **CPU is fine, ~a few minutes**. No GPU needed.",
))

cells.append(md("## 1 · Setup — clone repo"))
cells.append(code(
    "import os, sys, urllib.request",
    f"BR = '{BR}'",
    "if not os.path.exists('colab/build_tissue_model.py'):",
    "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
    "    if os.path.isdir('cell') and os.path.exists('cell/colab/build_tissue_model.py'): os.chdir('cell')",
    "assert os.path.exists('colab/build_tissue_model.py'), 'repo not cloned correctly'",
    "H='data/external_data/human'; OUT='outputs/orphan'",
    "os.makedirs(H, exist_ok=True); os.makedirs(OUT, exist_ok=True)",
))

cells.append(md(
    "## 2 · Restore the Drive data that powers the richer layers",
    "HPA + Omnipath are downloaded fresh below; these Drive files add druggable channels + downstream response."))
cells.append(code(
    "import shutil",
    "try:\n    from google.colab import drive; drive.mount('/content/drive')",
    "except Exception as e: print('Drive mount skipped:', e)",
    "DV='/content/drive/MyDrive/virtual_cell_data'; CM='/content/drive/MyDrive/cell_model'",
    "def _get(src, dst):",
    "    if os.path.isfile(src) and not (os.path.exists(dst) and os.path.getsize(dst)>1000):",
    "        shutil.copy2(src, dst); print('restored', os.path.basename(dst))",
    "for fn in ['dgidb.tsv','signor.tsv','collectri.tsv','cellphonedb.csv','cpdb_gene.csv','cpdb_protein.csv']:",
    "    _get(f'{DV}/human_raw/{fn}', f'{H}/{fn}')",
    "# NicheNet ligand->target (a computed output from the full build) enables downstream-inside-receiver wiring",
    "for src in [f'{CM}/nichenet_ligand_targets.json', f'{DV}/cell_build/nichenet_ligand_targets.json']:",
    "    _get(src, f'{OUT}/nichenet_ligand_targets.json')",
))

cells.append(md(
    "## 3 · Download the two core sources (HPA single-cell + Omnipath ligand-receptor)"))
cells.append(code(
    "op=urllib.request.build_opener(); op.addheaders=[('User-Agent','Mozilla/5.0')]; urllib.request.install_opener(op)",
    "SRC={'hpa_sc.tsv.zip':'https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip',",
    "     'omnipath_ligrec.tsv':'https://omnipathdb.org/interactions?datasets=ligrecextra&types=post_translational"
    "&genesymbols=yes&organisms=9606&fields=sources,references&format=tsv'}",
    "for fn,url in SRC.items():",
    "    p=f'{H}/{fn}'",
    "    if os.path.exists(p) and os.path.getsize(p)>10000: print('have',fn); continue",
    "    print('downloading',fn,'...'); urllib.request.urlretrieve(url,p); print('  ',os.path.getsize(p)//1024,'KB')",
    "# optional: CellChat pathway families (for colouring channels by pathway) — skip if it fails",
    "try:",
    "    cc='https://omnipathdb.org/annotations?resources=CellChatDB&genesymbols=yes&format=tsv'",
    "    if not os.path.exists(f'{H}/omnipath_cellchat.tsv'): urllib.request.urlretrieve(cc, f'{H}/omnipath_cellchat.tsv')",
    "except Exception as e: print('cellchat optional -> skip:', e)",
))

cells.append(md("## 4 · Build the tissue model + explorer"))
cells.append(code(
    "import subprocess",
    "subprocess.run([sys.executable,'colab/build_tissue_model.py'], check=True)",
    "subprocess.run([sys.executable,'colab/build_tissue_explorer.py'], check=True)",
))

cells.append(md("## 5 · Verify + save to Drive + download"))
cells.append(code(
    "import json",
    "T=json.load(open(f'{OUT}/tissue_model.json'))",
    "tis=T.get('tissues',{})",
    "nch=sum(len(v.get('links',v)) if isinstance(v,dict) else 0 for v in tis.values()) if isinstance(tis,dict) else 0",
    "print('tissues:', list(tis.keys()) if isinstance(tis,dict) else tis)",
    "print('ligand-receptor pairs:', T.get('n_lr'), '| downstream receptors:', len(T.get('downstream',{})),",
    "      '| druggable LR genes:', len(T.get('drugs',{})), '| endocrine axes:', len(T.get('endocrine',[])),",
    "      '| nichenet ligands:', len(T.get('nichenet',{})))",
    "assert isinstance(tis,dict) and len(tis)>0, 'no tissues built — check HPA/Omnipath downloads'",
    "CM='/content/drive/MyDrive/cell_model'; os.makedirs(CM, exist_ok=True)",
    "for f in ['tissue_model.json','tissue_explorer.html']:",
    "    if os.path.exists(f'{OUT}/{f}'): shutil.copy(f'{OUT}/{f}', f'{CM}/{f}'); print('saved ->', f'{CM}/{f}')",
    "try:",
    "    from google.colab import files; files.download(f'{OUT}/tissue_model.json')",
    "except Exception as e: print('download skipped:', e)",
    "print('Send me tissue_model.json for the cell<->tissue coupling.')",
))

cells.append(md(
    "## What you built",
    "Pick a tissue · click a cell type (who it signals) · click an arrow (ligand→receptor channels) ·",
    "knock out a gene (e.g. COL1A1, CTLA4) to see which cell-cell signals break — in `tissue_explorer.html`.",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "tissue_build.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
