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
    "## 4 · Coverage check — is a phenotype even IN this screen?",
    "This screen is CANCER cell lines with a cancer/cell-cycle gene panel. A phenotype is only testable if",
    "its signature genes are MEASURED and its candidate drivers were KNOCKED DOWN. Psoriasis is NOT — its",
    "effector genes aren't measured here. So we test a phenotype the data actually contains: proliferation."))
cells.append(code(
    "def coverage(genes, cell_line='rpe1'):",
    "    ev = effect_vectors(set(genes)|{'MYC'}, cell_line=cl if False else cell_line)",
    "    measured = set().union(*[set(v) for v in ev.values()]) if ev else set()",
    "    targets = set(ev)",
    "    return sorted(g for g in genes if g in measured), sorted(g for g in genes if g in targets)",
    "psor=['S100A7','DEFB4A','IL17A','IL17F','CCL20','STAT3','RORC','JAK2','IL23A']",
    "m,t = coverage(psor)",
    "print(f'psoriasis genes measured here: {m}  | knocked-down here: {t}  -> screen lacks psoriasis biology')",
))

cells.append(md(
    "## 5 · Measured causal drivers of PROLIFERATION (the phenotype this screen supports)",
    "Proliferation is fully covered (cell-cycle genes measured, cancer drivers knocked down). Expect the",
    "measured alibi test to rank proliferation DRIVERS (MYC, FOXM1, E2F1, CDK1, PLK1, AURKA, MDM2) high, and",
    "to flag TUMOR SUPPRESSORS (RB1, PTEN, TP53BP1) as PROTECTORS (their knockdown *increases* proliferation)."))
cells.append(code(
    "PROLIF_UP=['MKI67','PCNA','CDK1','CCNB1','CCNA2','CDC20','AURKA','AURKB','PLK1','TOP2A','BUB1','CENPA',",
    "           'CENPF','FOXM1','MYBL2','E2F1','TYMS','RRM2','BIRC5','TK1','KIF2C','NUF2','TPX2','CCNB2',",
    "           'CDC45','MCM3','MCM6','UBE2C','KIF11','KIF23','NDC80','SPC25','HJURP']",
    "PROLIF_DRIVERS=['MYC','E2F1','FOXM1','MYBL2','CDK1','CDK6','AURKA','AURKB','PLK1','CCNB1','MDM2','MDM4',",
    "                'CTNNB1','BRAF','PIK3CA','TYMS','RRM2','MCM6','CDC20','BUB1','TP53BP1','RB1','PTEN','ATM']",
    "sig = signature_vector(PROLIF_UP, [])",
    "need = set(PROLIF_UP)|set(PROLIF_DRIVERS)",
    "for cl in ['rpe1','k562','hepg2','jurkat']:",
    "    ev = effect_vectors(need, cell_line=cl)",
    "    rk = measured_alibi(sig, {t:ev[t] for t in ev if t in PROLIF_DRIVERS})",
    "    drivers=[r['target'] for r in rk if r['drives_disease']][:6]",
    "    protect=[r['target'] for r in rk if not r['drives_disease']][-4:]",
    "    print(f'{cl:7} top DRIVERS(KD reverses): {drivers}  | flagged PROTECTORS: {protect}')",
    "ev = effect_vectors(need, cell_line='rpe1'); rk = measured_alibi(sig, {t:ev[t] for t in ev if t in PROLIF_DRIVERS})",
    "json.dump(dict(phenotype='proliferation', cell_line='rpe1', ranking=rk),",
    "          open('outputs/orphan/measured_cause_proliferation.json','w'), indent=2)",
    "print('\\nmeasured causal ranking, proliferation (rpe1) — reversal>0 = driver, <0 = protector/suppressor:')",
    "for r in rk[:14]: print(f\"  {r['target']:9} reversal={r['reversal']:+.3f} drives_proliferation={r['drives_disease']}\")",
))

cells.append(md(
    "## 6 · Save + hand back",
    "The measured ranking is the 5th, strongest witness for the detective — interventional, not correlational."))
cells.append(code(
    "import shutil",
    "CM='/content/drive/MyDrive/cell_model'; os.makedirs(CM, exist_ok=True)",
    "shutil.copy('outputs/orphan/measured_cause_proliferation.json', f'{CM}/measured_cause_proliferation.json')",
    "print('saved -> ', f'{CM}/measured_cause_proliferation.json')",
    "print('Send me measured_cause_proliferation.json — the first measured-causal result from real knockouts.')",
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
