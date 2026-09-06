"""Build colab/cellgraph.ipynb — the model's mechanistic-capabilities run on Colab.

Runs the learned CellGraph model (SIGN/SGC GNN) AND the newer mechanistic nodes — the ΔΔG stability
predictor and the enzyme-constrained flux (ecModel) layer — then the full 13-axis recovery scorecard.
CPU-only. Restores the deeper cell_complete.json from Drive; benchmark data / Human-GEM are fetched at runtime.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = [
    md("# Cell model — mechanistic capabilities (Colab)",
       "",
       "One run, three engines, then the scorecard:",
       "1. **CellGraph** — a graph neural network (SIGN/SGC) over the 16,492-node cell knowledge graph:",
       "   what binds what · remove protein X → downstream · drug → off-targets · does wiring encode function.",
       "2. **ΔΔG predictor** — mutation → folding-stability change (structure+sequence), blind-tested on S669.",
       "3. **Enzyme-constrained flux** — quantitative mutation→flux on Human-GEM (essentiality + dominance).",
       "",
       "CPU-only. See `docs/CELLGRAPH.md`, `docs/DDG_STABILITY.md`, `docs/ECFLUX.md`."),
    md("## 1 · Clone + install"),
    code("import os, sys",
         f"BR='{BR}'",
         "if not os.path.exists('colab/cellgraph.py'):",
         "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
         "    if os.path.isdir('cell'): os.chdir('cell')",
         "os.system('pip -q install numpy scipy scikit-learn cobra mygene biopython')",
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
    md("## 3 · CellGraph — train embeddings + four capability metrics"),
    code("import subprocess",
         "print(subprocess.run([sys.executable,'colab/cellgraph.py'],capture_output=True,text=True).stdout)",
         "print(subprocess.run([sys.executable,'colab/validate_cellgraph.py'],capture_output=True,text=True).stdout)"),
    md("## 4 · CellGraph — live queries"),
    code("from cellgraph import CellGraph",
         "cg = CellGraph()",
         "print('TP53 can bind      ->', cg.bind_partners('TP53', 10))",
         "print('remove SREBF2      ->', cg.knockout_effect('SREBF2', 10))",
         "print('Imatinib off-target->', cg.drug_off_targets('Imatinib', 10))"),
    md("## 5 · ΔΔG stability predictor — mutation → destabilization",
       "Trains on S2648, blind-tests on S669 (fetches the benchmark + PDBs), then predicts on an AlphaFold structure."),
    code("print(subprocess.run([sys.executable,'colab/validate_ddg.py'],capture_output=True,text=True,cwd='.',",
         "      env={**os.environ,'PYTHONPATH':'colab'}).stdout)",
         "from ddg_predictor import DDGPredictor",
         "P = DDGPredictor('outputs/orphan/ddg_model.pkl')",
         "pdb = P.alphafold_pdb('P00441')            # SOD1",
         "for pos,wt,mut in [(5,'A','V'),(94,'G','A'),(94,'A','G')]:",
         "    d,ok = P.predict_from_structure(pdb,'A',pos,wt,mut)",
         "    print(f'SOD1 {wt}{pos}{mut}  predicted ΔΔG = {d:+.2f} kcal/mol  (positive = destabilizing)')"),
    md("## 6 · Enzyme-constrained flux — quantitative mutation → flux",
       "Live demo: the flux-control curve (metabolic dominance) + a mutation's ΔΔG→flux threshold on Human-GEM.",
       "The full essentiality validation (`validate_ecflux.py`, single-gene deletion, ~8 min) writes the scorecard axis."),
    code("import ecflux, numpy as np",
         "from cobra.flux_analysis import pfba",
         "m = ecflux.load_humangem()                 # fetches Human-GEM (~43 MB) first time",
         "bio = ecflux.biomass_id(m); wt = pfba(m)",
         "enz = [r.id for r in m.reactions if abs(wt.fluxes[r.id])>1e-4 and r.genes and r.subsystem",
         "       and 'Transport' not in (r.subsystem or '') and r.id!=bio][:40]",
         "curves, acts, _ = ecflux.flux_control_curve(m, enz)",
         "print('flux-control curve (mean biomass retained) — reproduces metabolic dominance:')",
         "for a,v in zip(acts, curves.mean(0)): print(f'  enzyme activity {int(a*100):3d}% -> biomass {100*v:5.1f}%')",
         "sens = enz[int(np.argmin(curves[:, acts.index(0.15)]))]",
         "print(f'\\nΔΔG → flux on a flux-sensitive enzyme ({sens}) — the carrier/affected threshold:')",
         "for ddg in [0,2,4,6,8,10]:",
         "    r = ecflux.mutation_to_flux(m, sens, ddg)",
         "    print(f'  ΔΔG={ddg:4.1f}  active_fraction={r[\"active_fraction\"]:.2f}  biomass={int(r[\"biomass_retained\"]*100)}%')",
         "# optional full validation (writes ecflux_validation.json for the scorecard):",
         "# print(subprocess.run([sys.executable,'colab/validate_ecflux.py'],capture_output=True,text=True,",
         "#       env={**os.environ,'PYTHONPATH':'colab'}).stdout)"),
    md("## 7 · Recovery scorecard — all 13 axes",
       "Regression gate: reads the committed + freshly-regenerated validation JSONs and asserts every claim still clears its bar."),
    code("print(subprocess.run([sys.executable,'colab/recovery_scorecard.py'],capture_output=True,text=True).stdout)"),
    md("## Scale path (next rounds)",
       "See `docs/FUTURE_IDEAS.md` for the full parking lot. Highlights:",
       "1. **Learned GNN (torch GraphSAGE/R-GCN)** to beat CellGraph's fixed-propagation link AUC.",
       "2. **Mutant kinetics** (CatPred/RealKcat on the mutant sequence) — the kcat fork of the mutation chain.",
       "3. **Absolute in-cell kcat** (PaxDb proteomics + curated medium) → real per-second flux capacities."),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "cellgraph.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
