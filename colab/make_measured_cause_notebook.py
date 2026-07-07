"""Build colab/measured_cause.ipynb — run the MEASURED causal cause-finder on the real Replogle data.

Uses the Replogle-Nadig Perturb-seq deltas on Drive (perturbation_signatures/replogle_nadig/cell_eval/
all_delta.parquet, ~6.5 GB) — the measured effect of knocking down ~2,000 genes. For a disease signature,
ranks knockdowns by how much their MEASURED effect reverses the phenotype (the interventional alibi test).
Memory-safe: reads ONLY the signature's columns from the parquet (pyarrow column projection), not the whole
6.5 GB. Jurkat is the T-cell line (closest to immune disease). No GPU; a normal runtime is fine.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = []

cells.append(md(
    "# Measured causal cause-finder — the alibi test with REAL knockout data",
    "",
    "Every network-only cause-finder hit the same wall: a static correlation graph can't separate the",
    "*driver* from its *downstream effects*. This uses **interventional** data instead — the Replogle-Nadig",
    "Perturb-seq screen (~2,000 gene knockdowns × 4 cell lines, measured transcriptome-wide effects).",
    "",
    "**Measured alibi test:** for a disease signature, rank knockdowns by how much their *measured* effect",
    "**reverses** the phenotype. The suspect whose removal undoes the crime is the causal driver. No network.",
))

cells.append(md("## 1 · Clone branch + import the method"))
cells.append(code(
    "import os, sys, json, glob",
    f"BR = '{BR}'",
    "if not os.path.exists('colab/measured_cause.py'):",
    "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
    "    if os.path.isdir('cell') and os.path.exists('cell/colab/measured_cause.py'): os.chdir('cell')",
    "sys.path.insert(0, 'colab')",
    "from measured_cause import measured_alibi, signature_vector, integrate_as_witness",
    "os.makedirs('outputs/orphan', exist_ok=True)",
))

cells.append(md(
    "## 2 · Locate the Replogle Perturb-seq deltas on Drive",
    "`perturbation_signatures/replogle_nadig/cell_eval/all_delta.parquet` + `definition/feature_names.json`."))
cells.append(code(
    "from google.colab import drive; drive.mount('/content/drive')",
    "ROOT='/content/drive/MyDrive/virtual_cell_data/perturbation_signatures/replogle_nadig'",
    "DELTA=f'{ROOT}/cell_eval/all_delta.parquet'",
    "FEAT =f'{ROOT}/definition/feature_names.json'",
    "assert os.path.exists(DELTA), f'missing {DELTA}'",
    "feat = json.load(open(FEAT))            # index -> gene symbol (list or dict)",
    "feat = feat if isinstance(feat, list) else [feat[str(i)] for i in range(len(feat))]",
    "sym2idx = {g: i for i, g in enumerate(feat)}",
    "print('features:', len(feat), '| delta parquet:', round(os.path.getsize(DELTA)/1e9,2),'GB')",
))

cells.append(md(
    "## 3 · Memory-safe load — read ONLY the columns we need (not the whole 6.5 GB)",
    "For a target signature + validation genes, project just those columns via pyarrow, filter a cell line,",
    "average sequencing batches → one measured effect vector per knockdown."))
cells.append(code(
    "import pyarrow.parquet as pq, pyarrow.compute as pc, pandas as pd, numpy as np",
    "pf = pq.ParquetFile(DELTA)",
    "allcols = [c for c in pf.schema.names]",
    "meta = [c for c in allcols if c in ('cell_line','gem_group','gene')]",
    "# columns are named by feature index (str) or by symbol; build symbol->column",
    "def col_for(symbol):",
    "    if symbol in allcols: return symbol",
    "    i = sym2idx.get(symbol)",
    "    return str(i) if (i is not None and str(i) in allcols) else None",
    "",
    "def effect_vectors(genes_needed, cell_line='jurkat'):",
    "    cols = meta + [c for c in {col_for(g) for g in genes_needed} if c]",
    "    df = pf.read(columns=cols).to_pandas()",
    "    if 'cell_line' in df and cell_line: df = df[df['cell_line']==cell_line]",
    "    gcols = [c for c in df.columns if c not in ('cell_line','gem_group','gene')]",
    "    agg = df.groupby('gene')[gcols].mean()",
    "    inv = {}",
    "    for g in genes_needed:",
    "        c = col_for(g)",
    "        if c in agg.columns: inv[c]=g",
    "    ev = {}",
    "    for tgt, row in agg.iterrows():",
    "        ev[tgt] = {inv[c]: float(v) for c,v in row.items() if c in inv and v==v}",
    "    return ev  # {knockdown_target: {measured_gene: delta}}",
    "print('cell lines available:', pf.read(columns=['cell_line']).to_pandas()['cell_line'].unique() if 'cell_line' in meta else 'n/a')",
))

cells.append(md(
    "## 4 · SANITY on real data — does a known regulator's knockdown reverse its own module?",
    "Before the disease, confirm the measured alibi test recovers a KNOWN driver in these cell lines: form a",
    "signature from a TF's targets and check the TF's own knockdown tops the reversal ranking."))
cells.append(code(
    "# STAT1 drives interferon-stimulated genes; knocking STAT1 down should reverse an ISG-up signature.",
    "ISG_UP=['STAT1','IRF1','GBP1','GBP2','IRF9','PSMB9','TAP1','B2M','STAT2','IRF7','UBE2L6','SP100']",
    "sig_val = signature_vector(ISG_UP, [])",
    "cands = ISG_UP + ['STAT1','JAK1','JAK2','IRF1','MYC','TP53','ACTB','GAPDH']",
    "ev = effect_vectors(set(ISG_UP)|set(cands), cell_line='jurkat')",
    "ranked = measured_alibi(sig_val, {t:ev[t] for t in ev if t in cands})",
    "print('measured reversal of the ISG signature (want STAT1/JAK/IRF1 on top):')",
    "for r in ranked[:8]: print(f\"  {r['target']:8} reversal={r['reversal']:+.3f} drives={r['drives_disease']}\")",
))

cells.append(md(
    "## 5 · The disease — measured causal drivers of psoriasis",
    "Honest caveat: Replogle is in cancer cell lines (Jurkat is the T-cell one), NOT diseased Th17 tissue,",
    "so inducible Th17 genes may be weakly expressed. This is measured causal evidence, tissue-mismatched."))
cells.append(code(
    "UP=['DEFB4A','S100A7','S100A8','S100A9','PI3','LCN2','CXCL1','CXCL8','CCL20','IL17A','IL17F','IL22',",
    "    'CXCL9','CXCL10','IL1B','CCL2','OASL','RSAD2','STAT1']",
    "DOWN=['FLG','KRT1','KRT10','KRT2','GATA3','IL4','IL13','CCL17','CCL22','WIF1']",
    "PSOR_DRIVERS=['STAT3','RORC','RORA','RELA','NFKB1','JAK2','JAK1','STAT4','IRF4','BATF','AHR','TYK2',",
    "              'STAT1','IL23A','IL12B','TNF','IL6','MYC','CEBPB']",
    "sig = signature_vector(UP, DOWN)",
    "need = set(UP)|set(DOWN)|set(PSOR_DRIVERS)",
    "for cl in ['jurkat','k562','hepg2','rpe1']:",
    "    ev = effect_vectors(need, cell_line=cl)",
    "    rk = measured_alibi(sig, {t:ev[t] for t in ev if t in PSOR_DRIVERS})",
    "    top = [f\"{r['target']}({r['reversal']:+.2f})\" for r in rk[:6]]",
    "    print(f'{cl:7} top measured reversers: {top}')",
    "# keep the T-cell line result as the main call",
    "ev = effect_vectors(need, cell_line='jurkat'); rk = measured_alibi(sig, {t:ev[t] for t in ev if t in PSOR_DRIVERS})",
    "json.dump(dict(disease='psoriasis', cell_line='jurkat', ranking=rk),",
    "          open('outputs/orphan/measured_cause_psoriasis.json','w'), indent=2)",
    "print('\\nmeasured causal driver ranking (jurkat):')",
    "for r in rk[:10]: print(f\"  {r['target']:8} reversal={r['reversal']:+.3f} drives_disease={r['drives_disease']}\")",
))

cells.append(md(
    "## 6 · Save + hand back",
    "The measured ranking is the 5th, strongest witness for the detective — interventional, not correlational."))
cells.append(code(
    "import shutil",
    "CM='/content/drive/MyDrive/cell_model'; os.makedirs(CM, exist_ok=True)",
    "shutil.copy('outputs/orphan/measured_cause_psoriasis.json', f'{CM}/measured_cause_psoriasis.json')",
    "print('saved -> ', f'{CM}/measured_cause_psoriasis.json')",
    "print('Send me measured_cause_psoriasis.json to fold the measured witness into the detective.')",
))

cells.append(md(
    "## What this is",
    "The first **measured causal** cause-finder in the model: it ranks disease drivers by the *real observed*",
    "effect of knocking each gene down, not by network propagation. Correctness is unit-tested in",
    "`measured_cause.py`; here it runs on the real Replogle screen. Limitation: cancer-cell-line context, not",
    "the diseased tissue — the honest next data is a disease-tissue Perturb-seq (or the Tahoe drug screen).",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "measured_cause.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
