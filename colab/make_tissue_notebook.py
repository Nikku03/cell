"""Generate build_tissue_model.ipynb — downloads the tissue-model data (HPA single-cell + Omnipath
ligand-receptor) and builds the tissue explorer. Separate from the cell model. Run: python colab/make_tissue_notebook.py"""
import json
from pathlib import Path
def md(*t): return {"cell_type":"markdown","metadata":{},"source":[l if l.endswith("\n") else l+"\n" for l in t]}
def code(*t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[l if l.endswith("\n") else l+"\n" for l in t]}
BR="claude/vectorize-gex-propensity-NRqBW"
cells=[
 md("# Build the Tissue Model","Multi-cell tissue with **cell-cell communication** (ligand&rarr;receptor). Separate from the cell model.",
    "Data: HPA single-cell expression (154 cell types) + Omnipath ligand-receptor (6,555 pairs). CPU is fine."),
 code("import os,sys,urllib.request",
    "if not os.path.exists('colab/build_tissue_model.py'):",
    f"    !git clone -q --branch {BR} https://github.com/nikku03/cell.git",
    "    os.chdir('cell')",
    "H='data/external_data/human'; os.makedirs(H,exist_ok=True); os.makedirs('outputs/orphan',exist_ok=True)"),
 code("op=urllib.request.build_opener();op.addheaders=[('User-Agent','Mozilla/5.0')];urllib.request.install_opener(op)",
    "SRC={'hpa_sc.tsv.zip':'https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip',",
    " 'omnipath_ligrec.tsv':'https://omnipathdb.org/interactions?datasets=ligrecextra&types=post_translational&genesymbols=yes&organisms=9606&fields=sources,references&format=tsv'}",
    "for fn,url in SRC.items():",
    "    p=f'{H}/{fn}'",
    "    if os.path.exists(p) and os.path.getsize(p)>10000: print('have',fn); continue",
    "    print('downloading',fn,'...'); urllib.request.urlretrieve(url,p)"),
 code("import subprocess as _sp",
    "_sp.run([sys.executable,'colab/build_tissue_model.py'],check=True)",
    "_sp.run([sys.executable,'colab/build_tissue_explorer.py'],check=True)",
    "print('built tissue_explorer.html')"),
 code("from IPython.display import HTML,display",
    "html=open('outputs/orphan/tissue_explorer.html').read().replace('\"','&#34;')",
    "display(HTML(f'<iframe srcdoc=\"{html}\" width=100% height=760 style=border:0></iframe>'))"),
 md("## What you can do","Pick a tissue &middot; click a cell type (who it signals) &middot; click a communication arrow (ligand&rarr;receptor channels) &middot; knock out a gene (e.g. COL1A1, CTLA4) to see which cell-cell signals break."),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
Path("colab/build_tissue_model.ipynb").write_text(json.dumps(nb,indent=1))
print("wrote colab/build_tissue_model.ipynb")
