"""Build colab/gnn.ipynb — train the learned GraphSAGE on GPU and run the honest head-to-head vs fixed CellGraph.

This is the GPU-tier run: same graph, same leakage-free benchmark, trained message-passing instead of fixed
propagation. On a Colab GPU runtime it trains in seconds; the AUC is identical to CPU (GPU only speeds it).
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = [
    md("# Learned GraphSAGE — the GPU-tier upgrade to fixed CellGraph",
       "",
       "Trains a 2-layer GraphSAGE on the 16,492-node cell graph and runs the **honest head-to-head** vs the",
       "fixed SIGN propagation, on the *same* leakage-free link-prediction benchmark (test edges removed,",
       "degree-matched hard negatives). Set **Runtime → GPU**. The AUC is identical CPU vs GPU — the GPU only",
       "makes training seconds instead of minutes. See `docs/CELLGRAPH_GNN.md`."),
    md("## 1 · Clone + install (torch)"),
    code("import os, sys",
         f"BR='{BR}'",
         "if not os.path.exists('colab/cellgraph.py'):",
         "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
         "    if os.path.isdir('cell'): os.chdir('cell')",
         "os.system('pip -q install numpy scipy scikit-learn torch')",
         "sys.path.insert(0,'colab'); os.makedirs('outputs/orphan', exist_ok=True)",
         "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu (set Runtime->GPU)')"),
    md("## 2 · Restore the deeper model from Drive"),
    code("from google.colab import drive; drive.mount('/content/drive')",
         "import glob, gzip, shutil, json",
         "dst='outputs/orphan/cell_complete.json'",
         "if not os.path.exists(dst):",
         "    c=sorted(glob.glob('/content/drive/MyDrive/cell_model/**/cell_complete*.json*',recursive=True),",
         "             key=lambda p: os.path.getsize(p), reverse=True)",
         "    src=c[0]; print('using', src)",
         "    (shutil.copyfileobj(gzip.open(src,'rb'),open(dst,'wb')) if src.endswith('.gz') else shutil.copy(src,dst))"),
    md("## 3 · Head-to-head — learned GraphSAGE vs fixed propagation (300 epochs on GPU)"),
    code("import subprocess",
         "print(subprocess.run([sys.executable,'colab/validate_cellgraph_gnn.py','300'],",
         "      capture_output=True,text=True,env={**os.environ,'PYTHONPATH':'colab'}).stdout)"),
    md("## 4 · Train a full model + inspect learned embeddings"),
    code("from cellgraph import load_model, node_features, build_adj",
         "import cellgraph_gnn as gnn",
         "D,G,name,idx = load_model(); X = node_features(G); A = build_adj(D, len(G))",
         "r = gnn.head_to_head(D, X, A, relation='ppi', epochs=300)",
         "print('PPI link prediction  fixed', r['fixed_auc'], ' -> learned', r['learned_auc'],",
         "      ' (Δ', r['delta'], ',', r['device'], ')')"),
    md("## Notes",
       "- Adopt the learned encoder only where it beats the fixed bar (the validator checks this).",
       "- Next GPU-tier items in `docs/FUTURE_IDEAS.md`: ESM-2 node features, structure-based kcat (KcatNet),",
       "  mutant kinetics (CatPred/RealKcat)."),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "gnn.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
