"""Build colab/master.ipynb — the whole current cell system in one Drive-aware Colab run.

Mounts Google Drive, restores the 36 MB cell, wires the STRONG dense features (STRING physical, Geneformer
embeddings) and the DepMap matrix straight off Drive, then runs the current pipeline end to end:
  scorecard -> CompleteCell (Phase 1) -> self-healing LOOP (Phase 2) -> signal-combiner (trained WITH the Drive
  features) -> loop again with the stronger combiner -> DepMap co-essentiality (Phase 3).
Core cells are dependency-light; the heavy features auto-activate only when the Drive files are present.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-zp09w8"
REPO = "https://github.com/Nikku03/cell.git"


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}


cells = [
    md("# The cell model — full system, one Colab run (Drive-aware)",
       "",
       "Everything current, in order, with your Google Drive mounted so the **strong** features load off disk:",
       "1. **Recovery scorecard** — capability axes gated vs known biology.",
       "2. **CompleteCell (Phase 1)** — the full-fidelity, per-gene-queryable cell the ML consumes.",
       "3. **Self-healing LOOP (Phase 2)** — analyse → detect every field → fix/verify (failure-guided, 3 "
       "outcomes) → repeat until convergence. Locked ledger + regression check.",
       "4. **Signal-combiner** — one calibrated P(edge) from many signals; trains WITH the Drive dense features "
       "(STRING physical, Geneformer embeddings) when present.",
       "5. **Loop again** with the stronger combiner.",
       "6. **DepMap co-essentiality (Phase 3)**.",
       "",
       "> Anti-trap throughout: physics > measured > predicted; measured facts are never overwritten."),

    md("## 1. Setup — clone the branch + deps"),
    code(f"!git clone --depth 1 -b {BR} {REPO} 2>/dev/null || (cd cell && git pull)",
         "%cd cell",
         "!pip -q install numpy scipy scikit-learn pandas pyarrow mygene 2>/dev/null   # pyarrow: Tahoe parquet; mygene: ENSG->symbol",
         "import sys, os; sys.path.insert(0, 'colab')",
         "os.makedirs('outputs/orphan', exist_ok=True)",
         "print('ready')"),

    md("## 2. Mount Drive — restore the cell + wire the STRONG features off disk  **(the key cell)**",
       "Restores the 36 MB `cell_complete.json`, copies the DepMap matrix local, and points the combiner's "
       "dense-feature env vars at your Drive files. Paths are tried in a few known locations; adjust if yours "
       "differ. Every heavy feature is **optional** — a missing file just turns that feature off, the run still "
       "completes."),
    code("from google.colab import drive; drive.mount('/content/drive')",
         "import glob, gzip, shutil, os",
         "D = '/content/drive/MyDrive'",
         "def first(*pats):",
         "    for p in pats:",
         "        g = sorted(glob.glob(p, recursive=True), key=lambda x: -os.path.getsize(x)) if os.path.sep in p else []",
         "        if g: return g[0]",
         "    return None",
         "",
         "# --- 2a. cell_complete.json (git-ignored 36 MB core data) ---",
         "dst = 'outputs/orphan/cell_complete.json'",
         "if not os.path.exists(dst):",
         "    src = first(f'{D}/cell_model/**/cell_complete*.json*', f'{D}/**/cell_complete*.json*')",
         "    assert src, 'cell_complete.json(.gz) not found under MyDrive/cell_model/'",
         "    print('restoring', src)",
         "    (shutil.copyfileobj(gzip.open(src,'rb'), open(dst,'wb')) if src.endswith('.gz') else shutil.copy(src, dst))",
         "import json; print('cell:', len(json.load(open(dst))['genes']), 'genes')",
         "",
         "# --- 2b. DepMap gene-effect matrix (co-essentiality) -> copy local for speed ---",
         "os.makedirs('depmap', exist_ok=True)",
         "if not os.path.exists('depmap/CRISPRGeneEffect.csv'):",
         "    ce = first(f'{D}/depmap_data/**/CRISPRGeneEffect.csv', f'{D}/**/CRISPRGeneEffect.csv')",
         "    if ce: print('copying DepMap', ce); shutil.copy(ce, 'depmap/CRISPRGeneEffect.csv')",
         "    else:  print('DepMap not on Drive — Phase 3 cell can download it from figshare instead')",
         "os.environ['DEPMAP_DIR'] = 'depmap'",
         "",
         "# --- 2c. STRING physical + Geneformer embeddings -> the combiner auto-uses them ---",
         "sl = first(f'{D}/virtual_cell_data/networks/string_physical*.gz', f'{D}/**/string_physical*.gz')",
         "sa = first(f'{D}/virtual_cell_data/networks/string_aliases*.gz', f'{D}/**/string_aliases*.gz')",
         "gf = first(f'{D}/cell_model/geneformer_gene_emb.npz', f'{D}/**/geneformer_gene_emb.npz')",
         "ex = first(f'{D}/depmap_data/**/OmicsExpression*.csv', f'{D}/cell_model/celltype_expression.csv',",
         "           f'{D}/**/celltype_expression.csv')   # dense co-expression matrix",
         "# Tahoe-100M: use the FILTERED cell_eval table (wide pseudobulk), NOT pdex (4.1B rows) or the per-cell emb",
         "tah = next(iter(glob.glob(f'{D}/**/tahoe_de', recursive=True)), None)",
         "if sl: os.environ['STRING_LINKS'] = sl",
         "if sa: os.environ['STRING_ALIASES'] = sa",
         "if gf: os.environ['GENEFORMER_NPZ'] = gf",
         "if ex: os.environ['EXPR_MATRIX'] = ex",
         "if tah: os.environ['TAHOE_DE_DIR'] = tah",
         "print('STRING:', bool(sl), '| aliases:', bool(sa), '| Geneformer:', bool(gf),",
         "      '| expr-matrix:', bool(ex), '| Tahoe-DE:', bool(tah),",
         "      '| DepMap:', os.path.exists('depmap/CRISPRGeneEffect.csv'))",
         "",
         "# --- 2d. restore any saved trained artifacts (so a reconnect doesn't start from scratch) ---",
         "import persist; persist.restore_from_drive(D)"),

    md("## 3. Recovery scorecard"),
    code("!python colab/recovery_scorecard.py"),

    md("## 4. CompleteCell (Phase 1) — the full-fidelity ML entry point",
       "Every layer reachable per gene at full resolution; `.apply_ledger()` folds in the loop's verified fixes."),
    code("from complete_cell import CompleteCell",
         "cell = CompleteCell()",
         "print('layers:', len(cell.layers()['coverage']), '| genes:', len(cell.genes))",
         "r = cell.gene('TP53'); print('TP53 -> ppi', len(r['ppi_partners']), '| regulates', len(r['regulates']),",
         "      '| complexes', r['complexes'][:2])"),

    md("## 5. Self-healing LOOP (Phase 2) — analyse → detect → fix/verify, until convergence",
       "First pass (before the combiner is trained it uses the single-lens corroboration). Reports the three "
       "outcomes per field, the locked ledger, and the regression check."),
    code("!python colab/phase2_loop.py"),

    md("## 6. Signal-combiner — train ONE calibrated P(edge) from every signal  *(uses Drive features)*",
       "If cell 2c found STRING / Geneformer, they enter as extra feature columns automatically (watch the "
       "printed feature list + the structural-vs-independent AUC — the dense features are what lift the "
       "independent AUC)."),
    code("!python colab/signal_combiner.py"),

    md("## 7. Loop again — now with the trained, stronger combiner",
       "The combiner is picked up as a calibrated lens (with the independence guard). Compare the locked "
       "`completion` count to cell 5."),
    code("!python colab/phase2_loop.py"),

    md("## 8. DepMap co-essentiality (Phase 3) — train the edge model + corroborate the additions",
       "Uses the Drive DepMap matrix from cell 2b (or downloads from figshare if absent)."),
    code("import os",
         "if not os.path.exists('depmap/CRISPRGeneEffect.csv'):",
         "    import urllib.request, json",
         "    j = json.loads(urllib.request.urlopen('https://api.figshare.com/v2/articles/25880521').read())",
         "    url = next(f['download_url'] for f in j['files'] if f['name']=='CRISPRGeneEffect.csv')",
         "    print('downloading DepMap (419 MB)…'); urllib.request.urlretrieve(url, 'depmap/CRISPRGeneEffect.csv')",
         "!DEPMAP_DIR=depmap python colab/phase3_depmap.py"),

    md("## 9. SAVE the trained artifacts to Drive  **(so a disconnect doesn't reset anything)**",
       "Colab wipes the VM on disconnect. This copies the trained combiner, the healed-cell ledger, and every "
       "result JSON to `MyDrive/cell_model/artifacts/`. Cell 2d restores them next session — reconnect = instant, "
       "and expensive derived features (Tahoe/FEBA later) are cached under `caches/`."),
    code("import persist; persist.save_to_drive(D)"),

    md("## 10. Extras — reasoned variant, whole-cell kcat, disease  *(optional)*"),
    code("!python colab/whole_cell_kcat.py            # test all predicted kcats vs physics + in-vivo floor",
         "!python colab/disease_data.py               # blind disease->target vs real Open Targets genes",
         "from reasoned_variant import ReasonedVariant",
         "rv = ReasonedVariant()",
         "r = rv.predict('HBB','P68871',7,'E','V')   # sickle cell — the gain-of-function blind-spot demo",
         "print('sickle:', r['call'], '| blind_spot:', r.get('ml_blind_spot'))"),
]

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path("colab/master.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "-", len(cells), "cells")
