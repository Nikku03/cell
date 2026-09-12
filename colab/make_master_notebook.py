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
         "# --- 2c. restore saved artifacts + derived caches NOW, BEFORE the Tahoe/expr cache checks below — else a",
         "#         reconnect re-downloads ~900 MB of Tahoe parquet even though tahoe_vecs.npz is already on Drive ---",
         "import persist; persist.restore_from_drive(D)",
         "",
         "# --- 2d. STRING physical + Geneformer embeddings -> the combiner auto-uses them ---",
         "sl = first(f'{D}/virtual_cell_data/networks/string_physical*.gz', f'{D}/**/string_physical*.gz')",
         "sa = first(f'{D}/virtual_cell_data/networks/string_aliases*.gz', f'{D}/**/string_aliases*.gz')",
         "gf = first(f'{D}/cell_model/geneformer_gene_emb.npz', f'{D}/**/geneformer_gene_emb.npz')",
         "# dense co-expression matrix — MUST be a BULK expression file (cell-lines x genes, ~0.4 GB). Never the",
         "# largest OmicsExpression*.csv: an all-transcripts / per-profile giant parses to >100 GB and OOMs. Prefer",
         "# the canonical protein-coding TPM matrix, then celltype_expression, then the SMALLEST OmicsExpression<2GB.",
         "def pick_expr():",
         "    for p in (f'{D}/depmap_data/**/OmicsExpressionProteinCodingGenesTPMLogp1.csv',",
         "              f'{D}/**/OmicsExpressionProteinCodingGenesTPMLogp1.csv',",
         "              f'{D}/cell_model/celltype_expression.csv', f'{D}/**/celltype_expression.csv'):",
         "        g = glob.glob(p, recursive=True)",
         "        if g: return g[0]",
         "    cand = [x for x in glob.glob(f'{D}/depmap_data/**/OmicsExpression*.csv', recursive=True)",
         "            if os.path.getsize(x) < 2e9]      # bulk matrices are small; a huge one is the wrong file",
         "    return min(cand, key=os.path.getsize) if cand else None",
         "ex = pick_expr()",
         "# Tahoe-100M was MEASURED DEAD for edge prediction (AUC 0.50 on ppi/reg/sig), so SKIP it: no download, no",
         "# load, no RAM — and it drops out of the retrained models. To re-measure it later, clear SKIP_FEATURES.",
         "os.environ['SKIP_FEATURES'] = 'tahoe_corr'          # add 'embedding_cos' here too — Geneformer is also dead",
         "skip = {s.strip() for s in os.environ['SKIP_FEATURES'].split(',') if s.strip()}",
         "import fetch_tahoe",
         "if 'tahoe_corr' in skip:",
         "    tah = False; print('tahoe: SKIPPED (measured dead 0.50 x3 — no download/load)')",
         "elif os.path.exists('outputs/orphan/tahoe_vecs.npz'):",
         "    tah = True; print('tahoe: using restored derived cache (no download)')",
         "else:",
         "    tah = fetch_tahoe.fetch()                                  # 14 wide plates from HuggingFace",
         "    if not tah: tah = next(iter(glob.glob(f'{D}/**/tahoe_de', recursive=True)), None)   # Drive fallback",
         "    if isinstance(tah, str): os.environ['TAHOE_DE_DIR'] = tah",
         "esm = first(f'{D}/esm_embeddings.parquet', f'{D}/**/esm_embeddings.parquet')",
         "# ONLY a GMT (pathway<TAB>id<TAB>gene-SYMBOLS) parses cleanly. A Drive Ensembl2Reactome .txt (ENSG ids,",
         "# 'parsed 0 pathways') is skipped on purpose; we fetch the canonical GMT instead.",
         "rx  = first(f'{D}/**/reactome*.gmt', f'{D}/**/*eactome*.gmt')",
         "if not rx:                                                 # not a GMT on Drive -> fetch canonical GMT (~0.3 MB)",
         "    try:",
         "        import urllib.request, zipfile, io",
         "        raw = urllib.request.urlopen('https://reactome.org/download/current/ReactomePathways.gmt.zip', timeout=120).read()",
         "        zipfile.ZipFile(io.BytesIO(raw)).extract('ReactomePathways.gmt', 'outputs/orphan')",
         "        rx = 'outputs/orphan/ReactomePathways.gmt'; print('reactome: downloaded canonical GMT (2,868 pathways)')",
         "    except Exception as e:",
         "        print('reactome auto-download failed:', str(e)[:70])",
         "sig = first(f'{D}/**/signor*.tsv', f'{D}/**/*SIGNOR*.tsv', f'{D}/**/signor*.csv', f'{D}/**/*signor*.txt')",
         "col = first(f'{D}/**/collectri*.tsv', f'{D}/**/*CollecTRI*.tsv', f'{D}/**/collectri*.csv',",
         "            f'{D}/**/*collectri*.txt', f'{D}/**/*ollec[Tt][Rr][Ii]*.tsv')",
         "def _omnipath(url, dst):                                   # causal edges, signed, mapped to gene symbols",
         "    try:",
         "        import urllib.request; urllib.request.urlretrieve(url, dst); return dst",
         "    except Exception as e:",
         "        print('  omnipath download failed:', str(e)[:60]); return None",
         "OP = 'https://omnipathdb.org/interactions?genesymbols=yes&fields=sources&organisms=9606&format=tsv&'",
         "if not sig: sig = _omnipath(OP + 'resources=SIGNOR', 'outputs/orphan/signor.tsv')",
         "if not col: col = _omnipath(OP + 'datasets=collectri', 'outputs/orphan/collectri.tsv')",
         "if sl: os.environ['STRING_LINKS'] = sl",
         "if sa: os.environ['STRING_ALIASES'] = sa",
         "if gf: os.environ['GENEFORMER_NPZ'] = gf",
         "if ex: os.environ['EXPR_MATRIX'] = ex",
         "if esm: os.environ['ESM_PARQUET'] = esm",
         "if rx: os.environ['REACTOME_TXT'] = rx",
         "if sig: os.environ['SIGNOR_TSV'] = sig",
         "if col: os.environ['COLLECTRI_TSV'] = col",
         "# heavy-tier data-hunt sources: point env vars at Drive files if present (new_data.py folds them in 7a-2)",
         "def _envfile(ev, *pats):",
         "    for p in pats:",
         "        g = sorted(glob.glob(p, recursive=True))",
         "        if g: os.environ[ev] = g[0]; return True",
         "    return False",
         "_hv = {",
         "  'INTERPRO_TSV': [f'{D}/**/protein2ipr*', f'{D}/**/*interpro*.tsv*'],",
         "  'HMDB_XML':     [f'{D}/**/hmdb_metabolites.xml'],",
         "  'CONC_TSV':     [f'{D}/**/*copies*cell*', f'{D}/**/*proteomic*ruler*', f'{D}/**/*absolute*abundance*', f'{D}/**/*Wang*tissue*'],",
         "  'TE_TSV':       [f'{D}/**/*translation*efficien*', f'{D}/**/*ribobase*', f'{D}/**/*TE*.tsv'],",
         "  'HALFLIFE_TSV': [f'{D}/**/*half*life*', f'{D}/**/*turnover*', f'{D}/**/*Mathieson*'],",
         "  'RE2G_TSV':     [f'{D}/**/*rE2G*', f'{D}/**/*E2G*link*', f'{D}/**/*enhancer*gene*'],",
         "  'CHIP_TSV':     [f'{D}/**/*chip*target*', f'{D}/**/*hTFtarget*', f'{D}/**/*ChIP-Atlas*'],",
         "  'BIOPLEX_TSV':  [f'{D}/**/*BioPlex*'],",
         "  'HURI_TSV':     [f'{D}/**/*HuRI*', f'{D}/**/HI-union*'],",
         "  'NEGATOME_TSV': [f'{D}/**/*negatome*'],",
         "}",
         "for ev, pats in _hv.items(): _envfile(ev, *pats)",
         "# AlphaFold: find the ~5 GB human tar (or a CIF dir) in Drive; else autofetch the EBI tar in structure()",
         "af = next(iter("
         "glob.glob(f'{D}/**/UP000005640*9606*.tar', recursive=True) + "
         "glob.glob(f'{D}/**/*9606*HUMAN*.tar', recursive=True) + "
         "glob.glob(f'{D}/**/*alphafold*human*.tar', recursive=True) + "
         "glob.glob(f'{D}/**/*alphafold*.tar', recursive=True) + "
         "glob.glob(f'{D}/**/alphafold*/', recursive=True) + "
         "glob.glob(f'{D}/**/AF_cifs/', recursive=True)), None)",
         "if af:",
         "    os.environ['AF_DIR'] = af.rstrip('/'); print('AlphaFold source found in Drive:', af)",
         "else:",
         "    os.environ['AF_AUTOFETCH'] = '1'; print('no AlphaFold file in Drive -> will autofetch the 5 GB EBI tar')",
         "print('heavy sources set:', [k for k in _hv if os.environ.get(k)] + "
         "(['AF_DIR'] if os.environ.get('AF_DIR') else ['AF_AUTOFETCH']))",
         "print('STRING:', bool(sl), '| aliases:', bool(sa), '| Geneformer:', bool(gf),",
         "      '| expr-matrix:', bool(ex), '| Tahoe:', bool(tah), '| ESM:', bool(esm),",
         "      '| Reactome:', bool(rx), '| SIGNOR:', bool(sig), '| CollecTRI:', bool(col),",
         "      '| DepMap:', os.path.exists('depmap/CRISPRGeneEffect.csv'))"),

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

    md("## 6. Signal-combiner — WHOLE-CELL: one calibrated model PER relation  *(uses Drive features)*",
       "The combiner isn't PPI-only — it predicts ANY relation by swapping the labels. This trains three: **ppi** "
       "(physical binding), **reg** (TF→target regulation), **sig** (signaling). Each keeps the features that "
       "actually predict IT (add→measure→keep, per relation) — so watch where the dense datasets land: STRING "
       "tends to win PPI, while co-essentiality / co-expression / **Tahoe drug-response** should matter more for "
       "**reg** (co-regulation), the relation they're actually about. All Drive features from cell 2c apply."),
    code("# train once, reuse — but RETRAIN when the available feature set CHANGED (e.g. ESM now loads, or a new",
         "# dataset appeared) so add->measure->keep actually measures the newcomer; otherwise skip (restored). One",
         "# process => the big features (DepMap/expr/ESM/STRING) load ONCE, not per-relation. CPU-bound; GPU idle.",
         "import os, json",
         "def _env_feats():",
         "    f = {'shared_partners','jaccard','same_complex','coexpression','codependency','coessentiality'}",
         "    if os.environ.get('STRING_LINKS'): f.add('string_score')",
         "    if os.environ.get('GENEFORMER_NPZ'): f.add('embedding_cos')",
         "    if os.environ.get('EXPR_MATRIX') or os.path.exists('outputs/orphan/expr_vecs.npz'): f.add('expr_corr')",
         "    if os.environ.get('TAHOE_DE_DIR') or os.path.exists('outputs/orphan/tahoe_vecs.npz'): f.add('tahoe_corr')",
         "    if os.environ.get('ESM_PARQUET'): f.add('esm_cos')",
         "    return f - {s.strip() for s in os.environ.get('SKIP_FEATURES','').split(',') if s.strip()}",
         "import sklearn",
         "def _saved(rel):",
         "    vj = 'outputs/orphan/signal_combiner_validation.json' if rel=='ppi' else f'outputs/orphan/signal_combiner_{rel}_validation.json'",
         "    return json.load(open(vj)) if os.path.exists(vj) else None",
         "want = _env_feats()",
         "need = []",
         "for rel in ('ppi','reg','sig'):",
         "    pkl = 'outputs/orphan/signal_combiner.pkl' if rel=='ppi' else f'outputs/orphan/signal_combiner_{rel}.pkl'",
         "    sv = _saved(rel)",
         "    feats = set(sv.get('features_all', [])) if sv else None",
         "    stale_sklearn = sv is not None and sv.get('sklearn_version') != sklearn.__version__   # avoid pickle-version drift",
         "    if os.environ.get('FORCE_RETRAIN') or not os.path.exists(pkl) or feats is None or (want - feats) or stale_sklearn:",
         "        need.append(rel)",
         "if need:",
         "    newish = sorted(want - set((_saved('ppi') or {}).get('features_all', [])))",
         "    print('training:', need, '| available:', sorted(want), '| NEW vs saved ppi:', newish or 'none')",
         "    os.system('python colab/signal_combiner.py ' + ' '.join(need))",
         "else:",
         "    print('all combiners current with the available features -> skipping retrain (FORCE_RETRAIN=1 to redo)')"),

    md("## 7. Loop again — now with the trained, stronger combiner",
       "The combiner is picked up as a calibrated lens (with the independence guard). Compare the locked "
       "`completion` count to cell 5."),
    code("!python colab/phase2_loop.py"),

    md("## 7a. Field fixes from Drive — full Reactome pathways + SIGNOR/CollecTRI causal edges",
       "Not combiner features — direct FIELD fixes: full Reactome lifts the pathway-coverage gap (was 23%), and "
       "SIGNOR + CollecTRI add directed/signed edges (the causal-direction the model lacked). Each prints the "
       "detected format + what it adds; overlays are written, nothing measured is overwritten."),
    code("!python colab/extra_data.py"),

    md("## 7a-2. New layers from the data hunt — TF motifs + extra complexes",
       "Genuinely-NEW layers (verified not already held): JASPAR binding motifs per TF (the mechanism beneath the "
       "reg edges) and CORUM/hu.MAP complexes beyond our 2,039. Downloads that the sandbox blocked run fine here on "
       "Colab's open internet; overlays are folded into the cell. GO was skipped — we already hold it for ~all genes."),
    code("!python colab/new_data.py"),

    md("## 7a-3. In-cell rates from kcat — cell-type capacity + substrate saturation",
       "Two in-cell quantities derived from the in-vitro kcat, neither needing measured flux: (A) cell-type "
       "**capacity** Vmax=kcat·ppm gated by the 200-cell-type expression mask; (B) the **saturation operating "
       "rate** v=kcat·[S]/([S]+Km) using each enzyme's Human-GEM primary substrate against measured mammalian "
       "intracellular concentrations. Validated: the saturation sigma distribution centres near Davidi's measured "
       "in-vivo 0.5. Uses cobra + Human-GEM (already downloaded by the ecflux cell)."),
    code("!python colab/incell_rates.py"),

    md("## 7a-4. New inter-layer connections — wire the dead-end layers in",
       "Connections between data layers that were not made before, each built only from data we already hold: "
       "**metabolite ↔ enzyme** (parsed from the Human-GEM reactions in generxn — opens metabolite→enzyme→pathway) "
       "and **domains → PPI** (self-supervised domain-pair interaction feature; validated +0.077 AUC over the "
       "structural features, and the disorder→PPI check runs here too once MobiDB disorder is present)."),
    code("!python colab/metabolite_enzyme.py\n!python colab/domain_ppi.py"),

    md("## 7a-5. Simulatable step — couple regulation to metabolism (PROM)",
       "The biggest *buildable* piece of the simulatable-cell gap: the regulatory network and the metabolic model "
       "were separate — a TF knockout did nothing to flux. This couples them (PROM): the 200-cell-type expression "
       "mask constrains each TF's metabolic targets, and biomass FBA is re-solved, so **TF knockout -> predicted "
       "growth**. Validated: **3.94x enrichment** for known metabolic-master TFs (PPARA, HNF4A, NFE2L2...) among the "
       "top knockouts. Honest limit: impact saturates (binary mask) -> qualitative ranker, not a graded growth "
       "predictor. Compute-heavy (200 FBA solves); reuses the ecflux Human-GEM download."),
    code("!python colab/rprom.py"),

    md("## 7b. WHOLE-CELL summary — every layer, not just PPI",
       "Genome · all three networks (ppi/reg/sig) with their per-relation combiner AUCs · complexes/SL/LR · "
       "kinetics & metabolism · pathways/PTMs · single-cell expression & context · disease/drugs · the ML "
       "self-healing outcomes across every field · the capability scorecard."),
    code("!python colab/cell_stats.py"),

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
