"""Build colab/fetch_external_data.ipynb — a Colab notebook that fetches the remaining external datasets
(the ones NOT already on Drive) and saves them into MyDrive/virtual_cell_data/<dataset>/.

Skips ARCHS4 and lincs_train.npz (already on Drive). Every source below was reachability-checked on
2026-07-06; access method + size + license noted per dataset. Run: `python colab/make_fetch_notebook.py`.
"""
import json
from pathlib import Path

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": [l + "\n" for l in lines]}

cells = []

cells.append(md(
    "# Fetch external datasets → Drive",
    "",
    "Populates `MyDrive/virtual_cell_data/<dataset>/` with the datasets that unblock the Better/Diverse",
    "improvements. **Skips ARCHS4 and `lincs_train.npz`** (already on Drive).",
    "",
    "Each dataset cell is independent, **skip-if-exists**, and prints size + license. Sources were",
    "reachability-checked 2026-07-06. Big/gated sources (Tahoe-100M, spatial) note their access method.",
    "",
    "| dataset | unblocks | size | access | license |",
    "|---|---|---|---|---|",
    "| Perturb-seq (scPerturb) | trained perturbation model, causal edges | ~1–10 GB | Zenodo direct | CC-BY |",
    "| Cell-type expression (CELLxGENE census) | context-specific networks, cell↔tissue (emask/abund) | ~2 GB | `cellxgene-census` API | CC-BY |",
    "| Dev + disease atlases (census) + TCGA (Xena) | beyond cancer/healthy, disease attractors | ~1–20 GB | census API + Xena direct | CC-BY / open |",
    "| Spatial (10x Xenium / Vizgen MERFISH) | real tissue geometry | ~1–30 GB/sample | direct S3 | CC-BY |",
    "| Cross-species conservation (UCSC phyloP) | conservation prior for dark genes | ~9 GB bigWig → tiny TSV | direct | open |",
    "| Tahoe-100M (Arc) | large drug-perturbation model | ~100s GB (fetch a subset) | HuggingFace (token) | CC-BY |",
    "| AlphaFold (per-accession) | structure layer (fixes the 404) | ~KB/protein | AFDB API | CC-BY 4.0 |",
))

cells.append(md("## Setup — mount Drive + a robust skip-if-exists fetcher"))
cells.append(code(
    "from google.colab import drive; drive.mount('/content/drive')",
    "import os, sys, subprocess, urllib.request, shutil, json, hashlib, time",
    "from pathlib import Path",
    "DATA = Path('/content/drive/MyDrive/virtual_cell_data'); DATA.mkdir(parents=True, exist_ok=True)",
    "def have(p): p=Path(p); return p.exists() and p.stat().st_size>0",
    "def sh(mb): return f'{mb/1024:.1f} GB' if mb>=1024 else f'{mb:.0f} MB'",
    "def fetch(url, dest, headers=None):",
    "    dest=Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)",
    "    if have(dest): print('  skip (exists):', dest, sh(dest.stat().st_size/1e6)); return dest",
    "    print('  downloading', url, '->', dest)",
    "    req=urllib.request.Request(url, headers=headers or {'User-Agent':'Mozilla/5.0'})",
    "    with urllib.request.urlopen(req) as r, open(dest,'wb') as f: shutil.copyfileobj(r, f, 1<<20)",
    "    print('  done', sh(dest.stat().st_size/1e6)); return dest",
    "def pipi(*pkgs): subprocess.run([sys.executable,'-m','pip','install','-q',*pkgs], check=False)",
))

cells.append(md("## 0 · Sanity — confirm the two you already have are on Drive"))
cells.append(code(
    "for p in ['expression_geo/archs4_human_gene.h5', 'lincs_train.npz',",
    "          '../cell_count_dynamics/multiorg']:",
    "    q=DATA/p; print(('OK  ' if q.exists() else 'MISSING '), q)",
    "# lincs_train.npz may live under outputs/orphan on Drive — adjust the path if so.",
))

cells.append(md(
    "## 1 · Perturb-seq (scPerturb) — for the trained perturbation model + causal edges",
    "Standardized h5ad (Norman 2019 combinatorial, Replogle 2022 genome-scale). Zenodo record 13350497.",
    "Wire into the build with `PERTURBSEQ_NORMAN_URL` / `PERTURBSEQ_RPE1_URL` = the local paths."))
cells.append(code(
    "import urllib.request, json",
    "rec=json.load(urllib.request.urlopen('https://zenodo.org/api/records/13350497'))",
    "want=('Norman','Replogle')  # pick the key screens; drop this filter to grab all",
    "dest=DATA/'perturbseq'",
    "for f in rec['files']:",
    "    nm=f['key']",
    "    if nm.endswith('.h5ad') and any(w in nm for w in want):",
    "        print(nm, sh(f['size']/1e6)); fetch(f['links']['self'], dest/nm)",
    "print('PERTURBSEQ dir:', dest)",
))

cells.append(md(
    "## 2 · Cell-type expression (CELLxGENE census / Tabula Sapiens)",
    "**This populates the currently-empty `emask`/`abund`** → unblocks context-specific networks and",
    "cell↔tissue coupling. Per-cell-type mean expression over the human atlas."))
cells.append(code(
    "pipi('cellxgene-census')",
    "import cellxgene_census, numpy as np, pandas as pd",
    "dest=DATA/'celltype_expression'; dest.mkdir(parents=True, exist_ok=True)",
    "out=dest/'tabula_sapiens_celltype_mean.parquet'",
    "if have(out): print('skip', out)",
    "else:",
    "    with cellxgene_census.open_soma(census_version='stable') as census:",
    "        adata = cellxgene_census.get_anndata(census, organism='Homo sapiens',",
    "            obs_value_filter=\"dataset_id=='<TABULA_SAPIENS_DATASET_ID>'\",  # or filter by tissue",
    "            column_names={'obs':['cell_type','tissue']})",
    "    df = (adata.to_df().groupby(adata.obs['cell_type'].values).mean())",
    "    df.to_parquet(out); print('wrote', out, df.shape)",
    "# Tip: to avoid loading everything, iterate tissues and mean per cell_type, then concat.",
))

cells.append(md(
    "## 3 · Disease-state expression — TCGA via UCSC Xena (open, direct)",
    "Bulk tumor-vs-normal expression → disease signatures as first-class attractors for the reversal engine."))
cells.append(code(
    "dest=DATA/'atlases/tcga'; dest.mkdir(parents=True, exist_ok=True)",
    "cohorts=['TCGA-BRCA','TCGA-LUAD','TCGA-COAD']  # add more as needed",
    "for c in cohorts:",
    "    url=f'https://gdc-hub.s3.us-east-1.amazonaws.com/download/{c}.star_tpm.tsv.gz'",
    "    fetch(url, dest/f'{c}.star_tpm.tsv.gz')",
    "# Developmental / disease single-cell atlases: query CELLxGENE census by development_stage / disease",
    "# (same API as cell 2, changing obs_value_filter).",
))

cells.append(md(
    "## 4 · Spatial transcriptomics — real tissue geometry (measured co-localization)",
    "Concrete public samples; add more sample URLs from 10x/Vizgen portals. These are per-sample large."))
cells.append(code(
    "dest=DATA/'spatial'; dest.mkdir(parents=True, exist_ok=True)",
    "# 10x Xenium human breast (public sample) — outs bundle",
    "fetch('https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/'",
    "      'Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip', dest/'xenium_human_breast_rep1.zip')",
    "# Vizgen MERFISH public data is on S3 (vizgen.com/data-release-program) — add the S3 object URLs there.",
    "print('spatial dir:', dest, '— unzip and read cell_feature_matrix + cells.parquet')",
))

cells.append(md(
    "## 5 · Cross-species conservation — UCSC phyloP100way → per-gene score (tiny, committable)",
    "Downloads the bigWig once, summarizes mean phyloP over each gene's exons → a small TSV you can even",
    "check into the repo. Orthogonal to the expression layers; a strong prior for dark genes."))
cells.append(code(
    "pipi('pyBigWig')",
    "import pyBigWig",
    "dest=DATA/'conservation'; dest.mkdir(parents=True, exist_ok=True)",
    "bw_path=dest/'hg38.phyloP100way.bw'",
    "fetch('https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw', bw_path)",
    "# needs a gene->exon BED (from refGene, already downloaded in the main pipeline). Pseudocode:",
    "# bw=pyBigWig.open(str(bw_path)); score={g: mean(bw.stats(chrom,s,e)) over the gene's exons}",
    "# pd.Series(score).to_csv(dest/'phyloP_pergene.tsv', sep='\\t')  # small -> commit-able",
    "print('conservation bigWig at', bw_path, '(~9 GB); summarize to phyloP_pergene.tsv)')",
))

cells.append(md(
    "## 6 · Tahoe-100M (Arc Institute) — large drug-perturbation corpus (subset)",
    "**Gated:** needs a HuggingFace token (`huggingface-cli login`). Full set is 100s of GB — fetch one plate."))
cells.append(code(
    "pipi('huggingface_hub')",
    "from huggingface_hub import snapshot_download",
    "dest=DATA/'tahoe100m'; dest.mkdir(parents=True, exist_ok=True)",
    "# set HF_TOKEN in the environment first. allow_patterns limits to a single shard/plate to stay small.",
    "try:",
    "    snapshot_download(repo_id='arcinstitute/Tahoe-100M', repo_type='dataset',",
    "        local_dir=str(dest), allow_patterns=['*plate1*','README*'])",
    "except Exception as e: print('Tahoe needs HF auth / correct shard pattern:', e)",
))

cells.append(md(
    "## 7 · AlphaFold structures — fixes the 404 (per-accession AFDB API, targeted)",
    "The EBI proteome tar path 404s; fetch only the accessions the model uses, via the stable AFDB API."))
cells.append(code(
    "dest=DATA/'alphafold'; dest.mkdir(parents=True, exist_ok=True)",
    "# accs = list of UniProt accessions from cell_complete.json (D['acc']); example:",
    "accs=['P04637','P01116','Q9Y6K9']  # TP53, KRAS, NEMO — replace with the model's acc list",
    "for a in accs:",
    "    url=f'https://alphafold.ebi.ac.uk/files/AF-{a}-F1-model_v4.pdb'",
    "    try: fetch(url, dest/f'AF-{a}-F1-model_v4.pdb')",
    "    except Exception as e: print('  no model for', a, e)",
    "# Bulk alternative: gs://public-datasets-deepmind-alphafold-v4/ via gsutil (whole proteomes).",
))

cells.append(md(
    "## Done — where things landed + how to wire them in",
    "```",
    "MyDrive/virtual_cell_data/",
    "  perturbseq/            -> PERTURBSEQ_NORMAN_URL / PERTURBSEQ_RPE1_URL (Model 4, causal edges)",
    "  celltype_expression/   -> populates emask/abund (context networks, cell<->tissue)",
    "  atlases/tcga/          -> disease-state attractors for reversal",
    "  spatial/               -> tissue-model geometry",
    "  conservation/          -> phyloP_pergene.tsv (dark-gene prior)",
    "  tahoe100m/             -> large drug-perturbation model",
    "  alphafold/             -> structure/fold layer",
    "```",
    "Then re-run `build_complete_cell.ipynb` with the new env vars set. Each addition should be gated by",
    "the recovery scorecard (`colab/recovery_scorecard.py`) — commit only what keeps it at all-PASS.",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "fetch_external_data.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
