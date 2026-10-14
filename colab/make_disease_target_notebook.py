"""Build colab/disease_target.ipynb — reproduce the blind, OOD disease->target pipeline on Colab.

Clones the zp09w8 branch (where the pipeline lives), restores cell_complete.json from Drive, then runs the
3 layers + the validation panel + the recovery scorecard, and saves the result JSONs back to Drive.
CPU-only, ~minutes. Layer 3 fetches protein family/structure from UniProt (needs internet, no token).
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = []

cells.append(md(
    "# Blind disease → target pipeline (out-of-distribution)",
    "",
    "The disease is **absent from the model's curated disease layers**; the drug target is **never named** —",
    "it must be *selected* by simulation:",
    "1. **Causal** — build the apex→readout signal-flow sub-graph, rank pathway candidates.",
    "2. **Perturb → wild-type** — disable/activate each candidate, measure readout collapse, read off direction.",
    "3. **Druggable** — UniProt family/structure → is the required direction achievable by that modality?",
    "",
    "Validated on psoriasis (IL-23/IL-17) and atopic dermatitis (Type-2). **CPU is fine, ~minutes.**",
))

cells.append(md("## 1 · Clone the branch"))
cells.append(code(
    "import os, sys",
    f"BR = '{BR}'",
    "if not os.path.exists('colab/disease_target_pipeline.py'):",
    "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
    "    if os.path.isdir('cell') and os.path.exists('cell/colab/disease_target_pipeline.py'): os.chdir('cell')",
    "assert os.path.exists('colab/disease_target_pipeline.py'), 'repo not cloned correctly'",
    "os.makedirs('outputs/orphan', exist_ok=True)",
    "sys.path.insert(0, 'colab')",
))

cells.append(md(
    "## 2 · Restore the model (`cell_complete.json`) from Drive",
    "The pipeline reads `outputs/orphan/cell_complete.json` (the assembled 16k-gene model). It is produced by",
    "the cell build and saved to Drive; here we copy/decompress it into place."))
cells.append(code(
    "import shutil, gzip, glob",
    "try:\n    from google.colab import drive; drive.mount('/content/drive')",
    "except Exception as e: print('Drive mount skipped:', e)",
    "dst = 'outputs/orphan/cell_complete.json'",
    "if not os.path.exists(dst):",
    "    cands = []",
    "    for base in ['/content/drive/MyDrive/cell_model', '/content/drive/MyDrive/virtual_cell_data']:",
    "        cands += glob.glob(f'{base}/**/cell_complete*.json*', recursive=True)",
    "    cands = sorted(cands, key=lambda p: os.path.getsize(p), reverse=True)",
    "    print('found on Drive:', cands[:3])",
    "    if cands:",
    "        src = cands[0]",
    "        if src.endswith('.gz'):",
    "            with gzip.open(src, 'rb') as f, open(dst, 'wb') as o: shutil.copyfileobj(f, o)",
    "        else:",
    "            shutil.copy(src, dst)",
    "        print('restored ->', dst, os.path.getsize(dst)//1024//1024, 'MB')",
    "    else:",
    "        print('!! cell_complete.json not found on Drive — run the cell build first, or upload it.')",
    "assert os.path.exists(dst), 'model missing'",
))

cells.append(md("## 3 · Run the pipeline — Layers 1-2 (psoriasis worked example)"))
cells.append(code(
    "import subprocess",
    "print(subprocess.run([sys.executable,'colab/disease_target_pipeline.py'],",
    "                     capture_output=True, text=True).stdout)",
))

cells.append(md("## 4 · Layer 3 — structural druggability (UniProt family/structure)"))
cells.append(code(
    "print(subprocess.run([sys.executable,'colab/druggability_layer3.py'],",
    "                     capture_output=True, text=True).stdout)",
))

cells.append(md("## 5 · Validation panel (OOD diseases) + recovery scorecard"))
cells.append(code(
    "print(subprocess.run([sys.executable,'colab/validate_disease_target.py'],",
    "                     capture_output=True, text=True).stdout)",
    "print(subprocess.run([sys.executable,'colab/recovery_scorecard.py'],",
    "                     capture_output=True, text=True).stdout)",
))

cells.append(md("## 6 · Record — show the target calls + save results to Drive"))
cells.append(code(
    "import json",
    "L3 = json.load(open('outputs/orphan/psoriasis_target_layer3.json'))",
    "V  = json.load(open('outputs/orphan/disease_target_validation.json'))",
    "SC = json.load(open('outputs/orphan/recovery_scorecard.json'))",
    "print('PSORIASIS final targets:', L3['final_targets'])",
    "print('Validation:', V['summary'])",
    "for d in V['diseases']:",
    "    if d.get('ok'): print('  ', d['disease'], '-> hits', d['hits'], '| top', d['top_rescuers'][:3])",
    "print('SCORECARD:', SC['n_pass'], '/', SC['n_total'], 'pass')",
    "CM='/content/drive/MyDrive/cell_model'; os.makedirs(CM, exist_ok=True)",
    "for f in ['disease_target_validation.json','psoriasis_target_layer12.json',",
    "          'psoriasis_target_layer3.json','recovery_scorecard.json']:",
    "    p=f'outputs/orphan/{f}'",
    "    if os.path.exists(p): shutil.copy(p, f'{CM}/{f}'); print('saved ->', f'{CM}/{f}')",
))

cells.append(md(
    "## What you reproduced",
    "- **Psoriasis** → selects **IL23A + IL23R** (disable; cytokine/receptor → antibody) = guselkumab/risankizumab.",
    "- **Atopic dermatitis** → selects **IL4R** (the dupilumab target), not the apex ligand.",
    "- **Recovery scorecard 8/8** — the disease→target axis is gated above a random-label baseline.",
    "",
    "**Honest scope:** mechanism→intervention (given the driver), *not* autonomous driver discovery;",
    "re-discovery on cytokine-cascade diseases; topological, not kinetic. See `docs/DISEASE_TARGET_RESULTS.md`.",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "disease_target.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
