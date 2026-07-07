"""Build colab/cellgraph.ipynb — run the learned CellGraph model + live queries on Colab.

CPU-only (the graph is 16k nodes). Restores the deeper cell_complete.json from Drive, trains the SIGN/SGC
embeddings, prints the four validated capability metrics, and runs the live query demo.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = [
    md("# CellGraph — a learned model of the whole cell",
       "",
       "A graph neural network (SIGN/SGC) over the 16,492-node multi-relational cell knowledge graph.",
       "Answers: what binds what · remove protein X → downstream · drug → off-targets · does wiring encode",
       "function. CPU-only, a few minutes. See `docs/CELLGRAPH.md`."),
    md("## 1 · Clone + install"),
    code("import os, sys",
         f"BR='{BR}'",
         "if not os.path.exists('colab/cellgraph.py'):",
         "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
         "    if os.path.isdir('cell'): os.chdir('cell')",
         "os.system('pip -q install numpy scipy scikit-learn')",
         "sys.path.insert(0,'colab'); os.makedirs('outputs/orphan', exist_ok=True)"),
    md("## 2 · Restore the deeper model from Drive"),
    code("from google.colab import drive; drive.mount('/content/drive')",
         "import glob, gzip, shutil, json",
         "dst='outputs/orphan/cell_complete.json'",
         "if not os.path.exists(dst):",
         "    c=sorted(glob.glob('/content/drive/MyDrive/cell_model/**/cell_complete*.json*',recursive=True),",
         "             key=lambda p: os.path.getsize(p), reverse=True)",
         "    src=c[0]; print('using', src)",
         "    (shutil.copyfileobj(gzip.open(src,'rb'),open(dst,'wb')) if src.endswith('.gz') else shutil.copy(src,dst))",
         "D=json.load(open(dst)); print('cell types', len(D['ctnames']), '| emask', len(D['emask']))"),
    md("## 3 · Train embeddings + the four capability metrics"),
    code("import subprocess",
         "print(subprocess.run([sys.executable,'colab/cellgraph.py'],capture_output=True,text=True).stdout)",
         "print(subprocess.run([sys.executable,'colab/validate_cellgraph.py'],capture_output=True,text=True).stdout)"),
    md("## 4 · Live queries — the model answering mechanistic questions"),
    code("from cellgraph import CellGraph",
         "cg = CellGraph()",
         "print('TP53 can bind      ->', cg.bind_partners('TP53', 10))",
         "print('remove SREBF2      ->', cg.knockout_effect('SREBF2', 10))",
         "print('remove TP53        ->', cg.knockout_effect('TP53', 10))",
         "print('Imatinib off-target->', cg.drug_off_targets('Imatinib', 10))",
         "print('what binds EGFR    ->', cg.bind_partners('EGFR', 10))"),
    md("## Scale path (next rounds)",
       "1. **Learned GNN (torch GraphSAGE/R-GCN)** to beat the fixed-propagation link AUC.",
       "2. **Supervised perturbation on the real Replogle screen** — calibrate magnitude, not just direction.",
       "3. **Fold in Geneformer / scGPT / Tahoe embeddings** (on Drive) as extra node features."),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "cellgraph.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
