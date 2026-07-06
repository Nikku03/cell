"""Build colab/phase1_rebuild.ipynb — a FAST, focused Phase-1 rebuild that populates `emask`.

Phase 1 goal: produce a cell_complete.json / cell_explorer.html whose `emask` (cell-type expression) is
POPULATED, so the context-specific-network and cell<->tissue improvements can be tested. Design:
  - clone the repo for the tested build code;
  - restore ALL previously-computed intermediates from Drive (so we SKIP the 60-step recompute);
  - FORCE a fresh CELLxGENE census run (this is what fills emask) with a valid census version;
  - skip the 59 GB ARCHS4 stage (WITH_COEXPR=0 — not needed for emask);
  - re-assemble + verify emask is non-empty;
  - save the export to Drive.
Reuses the exact restore/census/masters cells from make_cell_notebook.py (only change: force fresh census
+ census_version='stable'). Everything else comes from Drive; nothing large is copied through the FUSE mount.
"""
import json
from pathlib import Path

BR = "claude/vectorize-gex-propensity-NRqBW"

def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(*L): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in L]}

cells = []

cells.append(md(
    "# Phase 1 — fast rebuild to populate `emask`",
    "",
    "Produces a complete export whose **cell-type expression (`emask`) is populated**, so the context-network",
    "and cell↔tissue improvements can be tested. It **restores everything else from Drive** (no 60-step",
    "recompute), **forces a fresh CELLxGENE census** (the step that fills `emask`), and **skips the 59 GB",
    "ARCHS4 stage** (not needed here). Set **Runtime → GPU** for the census + assemble.",
    "",
    "The two fixes vs the old build: (1) force a fresh census even if a stale `celltype_expression.csv` is on",
    "Drive; (2) use `census_version='stable'` (the pinned date had expired).",
))

cells.append(md("## 1 · Setup — clone repo + installs"))
cells.append(code(
    "import os, sys",
    "IN_COLAB = 'google.colab' in sys.modules",
    f"BR = '{BR}'",
    "if not os.path.exists('colab/build_cell_complete.py'):",
    "    os.system(f'git clone -q --branch {BR} https://github.com/nikku03/cell.git')",
    "    if os.path.isdir('cell') and os.path.exists('cell/colab/build_cell_complete.py'): os.chdir('cell')",
    "assert os.path.exists('colab/build_cell_complete.py'), 'repo not cloned correctly'",
    "os.system('pip -q install cellxgene-census pandas numpy scipy anndata scikit-learn huggingface_hub')",
    "import json, numpy as np, pandas as pd",
    "OUT='outputs/orphan'; os.makedirs(OUT, exist_ok=True)",
))

cells.append(md(
    "## 2 · Restore from Drive (substrate + ALL precomputed outputs) and FORCE a fresh census",
    "Restores the raw substrate and every previously-computed layer from Drive so the 60-step pipeline is",
    "skipped — then deletes `celltype_expression.csv`/`celltype_masters.json` so the census actually re-runs."))
cells.append(code(
    "import shutil, glob as _glob, tarfile",
    "H='data/external_data/human'; os.makedirs(H, exist_ok=True)",
    "if IN_COLAB:",
    "    try:\n        from google.colab import drive; drive.mount('/content/drive')",
    "    except Exception as e: print('Drive mount skipped:', e)",
    "def _restore(srcdir, dst, only=None):",
    "    if not os.path.isdir(srcdir): return 0",
    "    os.makedirs(dst, exist_ok=True); n=0",
    "    for fn in os.listdir(srcdir):",
    "        s=os.path.join(srcdir,fn); t=os.path.join(dst,fn)",
    "        if not os.path.isfile(s): continue",
    "        if only is not None and fn not in only: continue",
    "        if os.path.exists(t) and (os.path.getsize(t)==os.path.getsize(s) or os.path.getsize(t)>10000): continue",
    "        shutil.copy2(s,t); n+=1",
    "    return n",
    "DV='/content/drive/MyDrive/virtual_cell_data'; CM='/content/drive/MyDrive/cell_model'",
    "n_raw=_restore(f'{DV}/human_raw', H)                # raw substrate (DepMap, Perturb-seq, STRING, ...)",
    "n_out=_restore(CM, OUT) + _restore(f'{DV}/cell_build', OUT)   # ALL precomputed layers -> skip recompute",
    "n_cat=_restore(f'{DV}/catpred_kinetics', OUT)",
    "if os.path.exists(f'{OUT}/analysis_outputs.tgz'):",
    "    try:\n        tarfile.open(f'{OUT}/analysis_outputs.tgz').extractall(OUT); print('extracted analysis_outputs.tgz')",
    "    except Exception as e: print('tgz extract skipped:', e)",
    "# FORCE fresh census: remove any restored cell-type files so _SKIP_CENSUS is False (fixes the empty emask)",
    "for f in ['celltype_expression.csv','celltype_masters.json']:",
    "    p=f'{OUT}/{f}'",
    "    if os.path.exists(p): os.remove(p); print('removed stale', f, '-> census will run fresh')",
    "print(f'restored: {n_raw} substrate + {n_out} precomputed + {n_cat} CatPred file(s)')",
    "assert os.path.exists(f'{OUT}/integrated_cell_human.csv'), 'integrated_cell_human.csv missing from Drive (cell_build/) — needed as the backbone'",
))

cells.append(md("## 3 · Load the gene backbone (needed by the census cell)"))
cells.append(code(
    "bb=pd.read_csv(f'{OUT}/integrated_cell_human.csv')",
    "print('backbone:', bb.shape, '| TFs:', int((bb.is_tf==1).sum()))",
))

cells.append(md(
    "## 4 · Model 2 — CELLxGENE census (fills `emask`)  ·  census_version='stable'",
    "Per-cell-type mean CP10k expression → `celltype_expression.csv`. Needs internet (and GPU/RAM for speed)."))
cells.append(code(
    "import cellxgene_census, scipy.sparse as sp, traceback",
    "TFset=set(bb.loc[bb.is_tf==1,'gene'])",
    "adata=None; cell_means=None",
    "TISSUES=['blood','liver','heart','lung','brain','kidney']",
    "N_TYPES=40; PER_TYPE=60; MIN_CELLS=25",
    "try:",
    "    with cellxgene_census.open_soma(census_version='stable') as census:",
    "        VF=\"is_primary_data==True and tissue_general in \"+repr(TISSUES)",
    "        obs=pd.DataFrame(cellxgene_census.get_obs(census,'Homo sapiens',value_filter=VF,",
    "                                     column_names=['soma_joinid','cell_type']))",
    "        vc=obs['cell_type'].value_counts(); keep=vc[vc>=MIN_CELLS].head(N_TYPES).index",
    "        obs=obs[obs['cell_type'].isin(keep)]",
    "        ids=(obs.groupby('cell_type',observed=True,group_keys=False)",
    "                .apply(lambda d:d.sample(min(len(d),PER_TYPE),random_state=0)))",
    "        print('fetching',len(ids),'cells x ALL genes across',ids['cell_type'].nunique(),'cell types...')",
    "        adata=cellxgene_census.get_anndata(census,'Homo sapiens',",
    "               obs_coords=sorted(ids['soma_joinid'].tolist()), obs_column_names=['cell_type'])",
    "    print('got',adata.shape[0],'cells x',adata.shape[1],'genes')",
    "    X=adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)",
    "    lib=np.asarray(X.sum(1)).ravel(); lib[lib==0]=1",
    "    Xn=X.multiply(1e4/lib[:,None]).tocsr()",
    "    cts=adata.obs['cell_type'].astype(str).values; uc=pd.unique(cts); ci={c:i for i,c in enumerate(uc)}",
    "    rows=np.array([ci[c] for c in cts])",
    "    Ind=sp.csr_matrix((np.ones(len(rows)),(rows,np.arange(len(rows)))),shape=(len(uc),len(rows)))",
    "    counts=np.asarray(Ind.sum(1)).ravel()",
    "    means=np.log1p(np.asarray((Ind@Xn).todense())/counts[:,None])",
    "    genes=adata.var['feature_name'].astype(str).values",
    "    cell_means=pd.DataFrame(means,index=uc,columns=genes)",
    "    cell_means=cell_means.loc[:,~cell_means.columns.duplicated()]",
    "    cell_means.to_csv(f'{OUT}/celltype_expression.csv')",
    "    print('cell-type expression:',cell_means.shape,'-> celltype_expression.csv')",
    "except Exception as _e:",
    "    traceback.print_exc(); print('CENSUS FAILED — emask will stay empty. Check internet / census version.')",
))
cells.append(code(
    "masters={}",
    "if cell_means is not None:",
    "    glob=cell_means.mean(axis=0)+1e-6; spec=cell_means.div(glob,axis=1)",
    "    for ct in cell_means.index:",
    "        tfs=[g for g in spec.columns if g in TFset]",
    "        top=spec.loc[ct,tfs].sort_values(ascending=False).head(4)",
    "        top=[g for g in top.index if cell_means.loc[ct,g]>0.5]",
    "        if top: masters[str(ct)]=top",
    "    json.dump(dict(list(masters.items())[:40]),open(f'{OUT}/celltype_masters.json','w'),indent=1)",
    "    print('celltype_masters.json:',len(masters),'cell types')",
))

cells.append(md(
    "## 5 · Assemble (WITH_COEXPR=0 — skip the 59 GB ARCHS4 stage)",
    "Reads the restored substrate + precomputed layers + the fresh `celltype_expression.csv` → the export."))
cells.append(code(
    "import subprocess",
    "os.environ['WITH_COEXPR']='0'   # no ARCHS4 stage; emask does NOT need it",
    "for step in ['colab/build_cell_complete.py','colab/build_cell_app_complete.py','colab/build_cell_explorer.py',",
    "             'colab/compute_metabolic_graph.py','colab/build_metabolic_routes.py']:",
    "    print('===',step,'==='); r=subprocess.run([sys.executable,step]); ",
    "    if r.returncode!=0: print('  WARN non-zero exit from',step)",
))

cells.append(md("## 6 · Verify `emask` is populated (fail loudly if not)"))
cells.append(code(
    "import gzip",
    "p=f'{OUT}/cell_complete.json'",
    "D=json.load(gzip.open(p+'.gz')) if os.path.exists(p+'.gz') else json.load(open(p))",
    "ne=len(D.get('emask',{})); na=len(D.get('abund',{})); nct=len(D.get('ctnames',[]))",
    "print(f'emask genes: {ne} | abund genes: {na} | cell types: {nct}')",
    "assert ne>0, 'emask STILL empty — census did not produce celltype_expression.csv (check cell 4 output)'",
    "print('OK — emask is populated. Export is ready.')",
))

cells.append(md("## 7 · Save the export to Drive + download"))
cells.append(code(
    "PROJ='/content/drive/MyDrive/cell_model'; os.makedirs(PROJ, exist_ok=True)",
    "for f in ['cell_complete.json.gz','cell_complete.json','cell_explorer.html','cell_complete.html','cell_model_summary.json']:",
    "    s=f'{OUT}/{f}'",
    "    if os.path.exists(s): shutil.copy(s, f'{PROJ}/{f}'); print('saved ->', f'{PROJ}/{f}')",
    "try:",
    "    from google.colab import files",
    "    _dl = [f for f in ['cell_complete.json.gz','cell_complete.json'] if os.path.exists(f'{OUT}/{f}')][:1]",
    "    for f in _dl + (['cell_explorer.html'] if os.path.exists(f'{OUT}/cell_explorer.html') else []):",
    "        files.download(f'{OUT}/{f}')",
    "except Exception as e: print('download skipped:', e)",
    "print('Send me cell_complete.json.gz (or cell_explorer.html) to run the Phase-1 tests.')",
))

cells.append(md(
    "## 8 · (optional, Phase-2 prep) direct-download HF drug signatures to fast SSD",
    "Not needed for Phase 1. Public (no token). Downloads to /content SSD (fast, bypasses Drive). tahoe_de is",
    "also already on your Drive — use this only if you want it on fast local disk for Phase-2 processing."))
cells.append(code(
    "GET_TAHOE_DE=False   # flip to True when you start Phase 2",
    "if GET_TAHOE_DE:",
    "    from huggingface_hub import snapshot_download",
    "    d='/content/tahoe_de'; os.makedirs(d, exist_ok=True)",
    "    snapshot_download('tahoebio/tahoe-de-rhaister', repo_type='dataset', local_dir=d,",
    "        allow_patterns=['*cell_eval*','*control_expression*','*cell_centroids*','*.md','*.json'])",
    "    print('drug signatures on SSD ->', d)",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "phase1_rebuild.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
