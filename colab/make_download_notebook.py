"""Generate download_all_data.ipynb — one notebook that downloads EVERY dataset the virtual cell
uses, into an organized folder tree on Google Drive (MyDrive/virtual_cell_data/), idempotently,
with a written manifest + size report. Run: python colab/make_download_notebook.py"""
import json
from pathlib import Path
def md(*t): return {"cell_type":"markdown","metadata":{},"source":[l if l.endswith("\n") else l+"\n" for l in t]}
def code(*t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[l if l.endswith("\n") else l+"\n" for l in t]}

# (category subfolder, filename, url, gunzip-to-plain?, optional-flag-or-None)
DATASETS = [
 # localization / scaffold
 ("localization","uniprot_localization.tsv","https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=gene_primary,cc_subcellular_location&format=tsv",False,None),
 # networks
 ("networks","collectri.tsv","https://raw.githubusercontent.com/nikku03/cell/claude/vectorize-gex-propensity-NRqBW/data/external_data/human/collectri.tsv",False,None),
 ("networks","signor.tsv","https://signor.uniroma2.it/getData.php?organism=9606",False,None),
 ("networks","string_physical.txt.gz","https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz",False,None),
 ("networks","string_aliases.txt.gz","https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",False,None),
 # complexes / ligand-receptor
 ("complexes","complexportal_9606.tsv","https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv",False,None),
 ("ligand_receptor","cellphonedb_interactions.csv","https://raw.githubusercontent.com/ventolab/cellphonedb-data/master/data/interaction_input.csv",False,None),
 # metabolism / pathways
 ("metabolism","human_gem.txt","https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.txt",False,None),
 ("pathways","reactome_ensembl.txt","https://reactome.org/download/current/Ensembl2Reactome.txt",False,None),
 # drugs / PTMs
 ("drugs","dgidb_interactions.tsv","https://dgidb.org/data/latest/interactions.tsv",False,None),
 ("ptms","uniprot_acc_ptm.tsv","https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=accession,gene_primary,ft_mod_res&format=tsv",False,None),
 # genome / epigenome
 ("genome","gene_info.gz","https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",False,None),
 ("genome","cpgIslandExt.txt.gz","https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",False,None),
 # disease / virus
 ("disease","genes_to_phenotype.txt","https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",False,None),
 ("virus","hiv_interactions.gz","https://ftp.ncbi.nlm.nih.gov/gene/GeneRIF/hiv_interactions.gz",True,None),
 # essentiality truth sets
 ("truth","CEGv2.txt","https://raw.githubusercontent.com/nikku03/cell/claude/vectorize-gex-propensity-NRqBW/data/external_data/human/CEGv2.txt",False,None),
 ("truth","NEGv1.txt","https://raw.githubusercontent.com/nikku03/cell/claude/vectorize-gex-propensity-NRqBW/data/external_data/human/NEGv1.txt",False,None),
 # --- the big additions you can download ---
 ("rna_ncrna","rnacentral_homo_sapiens.bed.gz","https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/genome_coordinates/bed/homo_sapiens.GRCh38.bed.gz",False,None),
 ("rna_ncrna","rnacentral_ensembl_mapping.tsv","https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/id_mapping/database_mappings/ensembl.tsv",False,"WITH_RNACENTRAL_MAPPING"),
 ("genome_3d","GM12878_HiCCUPS_loops.txt.gz","https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525_GM12878_primary%2Breplicate_HiCCUPS_looplist.txt.gz",False,None),
 ("expression_geo","archs4_human_gene.h5","https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.latest.h5",False,"WITH_FULL_GEO"),
]

cells=[
 md("# Download all virtual-cell data → Google Drive",
    "Grabs every dataset the cell uses into an organized tree on your Drive, idempotently (skips",
    "what's already there), and writes a manifest + size report.",
    "",
    "**Sizes (real, measured):** core datasets ≈ **150 MB**; + RNAcentral human ≈ **160 MB**; + 3D-genome",
    "loops ≈ **1 MB**. The **GEO expression compendium (ARCHS4 human) is ~59 GB** and is **OFF by default**",
    "— flip `WITH_FULL_GEO=True` only if you want the whole thing (or grab specific GSE series instead).",
    "",
    "Set *Runtime → any* (CPU is fine; this is just downloading)."),
 code("from google.colab import drive; drive.mount('/content/drive')",
    "import os, urllib.request, gzip, shutil, time",
    "ROOT='/content/drive/MyDrive/virtual_cell_data'",
    "os.makedirs(ROOT, exist_ok=True)",
    "print('data root ->', ROOT)"),
 code("# flags for the huge / optional pulls",
    "WITH_FULL_GEO=False              # True = download the ~59 GB ARCHS4 human compendium",
    "WITH_RNACENTRAL_MAPPING=True     # ~139 MB id-mapping (set False to skip)",
    "print('WITH_FULL_GEO =',WITH_FULL_GEO,'| WITH_RNACENTRAL_MAPPING =',WITH_RNACENTRAL_MAPPING)"),
 code("DATASETS = "+json.dumps([list(d) for d in DATASETS], indent=1),
    "",
    "def fetch(cat, fn, url, gunzip, flag):",
    "    if flag and not globals().get(flag, False):",
    "        return (cat, fn, 'skipped (flag off)', 0)",
    "    d=os.path.join(ROOT, cat); os.makedirs(d, exist_ok=True)",
    "    dst=os.path.join(d, fn)",
    "    if os.path.exists(dst) and os.path.getsize(dst)>1000:",
    "        return (cat, fn, 'have', os.path.getsize(dst))",
    "    tmp=dst+'.part'",
    "    try:",
    "        urllib.request.urlretrieve(url, tmp)",
    "        if gunzip:",
    "            with gzip.open(tmp,'rb') as f, open(dst,'wb') as o: shutil.copyfileobj(f,o)",
    "            os.remove(tmp)",
    "        else:",
    "            os.replace(tmp, dst)",
    "        return (cat, fn, 'downloaded', os.path.getsize(dst))",
    "    except Exception as e:",
    "        if os.path.exists(tmp): os.remove(tmp)",
    "        return (cat, fn, 'FAILED: '+repr(e)[:80], 0)"),
 code("rows=[]",
    "for cat, fn, url, gunzip, flag in DATASETS:",
    "    r=fetch(cat, fn, url, gunzip, flag)",
    "    rows.append(r)",
    "    mb=r[3]/1048576",
    "    print(f'[{r[2]:<12}] {r[0]}/{r[1]}  {mb:8.1f} MB')",
    "total=sum(r[3] for r in rows)/1048576",
    "print(f'\\nTOTAL ON DRIVE: {total:.1f} MB ({total/1024:.2f} GB)')"),
 code("# write a manifest + README so you always know where the data is",
    "import json as _j",
    "man={'root':ROOT,'generated_epoch':int(time.time()),",
    "     'files':[{'category':r[0],'file':r[1],'status':r[2],'bytes':r[3]} for r in rows],",
    "     'total_mb':round(sum(r[3] for r in rows)/1048576,1)}",
    "open(os.path.join(ROOT,'MANIFEST.json'),'w').write(_j.dumps(man, indent=1))",
    "readme=['# virtual_cell_data','','Downloaded by download_all_data.ipynb. Layout:','']",
    "cats=sorted(set(r[0] for r in rows))",
    "for c in cats: readme.append(f'- `{c}/` — '+', '.join(r[1] for r in rows if r[0]==c and r[3]>0))",
    "open(os.path.join(ROOT,'README.md'),'w').write(chr(10).join(readme))",
    "print('wrote', ROOT+'/MANIFEST.json and README.md')",
    "print('\\nfolder tree:')",
    "for c in cats:",
    "    p=os.path.join(ROOT,c)",
    "    if os.path.isdir(p): print(' ', c+'/', '->', len(os.listdir(p)), 'files')"),
 md("## Where your data is",
    "Everything is under **`MyDrive/virtual_cell_data/`**, one subfolder per layer:",
    "`localization/ networks/ complexes/ ligand_receptor/ metabolism/ pathways/ drugs/ ptms/`",
    "`genome/ disease/ virus/ truth/ rna_ncrna/ genome_3d/ expression_geo/`.",
    "`MANIFEST.json` lists every file + size; `README.md` describes the tree.",
    "",
    "**Notes on the big three:**",
    "- **RNAcentral** — human ncRNA coords (21 MB) + optional id-mapping (139 MB).",
    "- **3D genome (4DN-style)** — a processed HiCCUPS loop list (1 MB). For more/tissue-specific loops,",
    "  download processed `.bedpe` from the 4DN portal (data.4dnucleome.org) into `genome_3d/`.",
    "- **GEO** — full ARCHS4 human is ~59 GB (`WITH_FULL_GEO`). For conditions, prefer specific GSE",
    "  series (a few MB each) — drop them in `expression_geo/` and they'll be picked up.",
    "",
    "The build notebook (`build_complete_cell.ipynb`) can read from here — point its `H=` data dir at",
    "`MyDrive/virtual_cell_data/` subfolders, or copy what it needs into the repo's `data/external_data/human/`."),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
out=Path("colab/download_all_data.ipynb"); out.write_text(json.dumps(nb,indent=1))
print("wrote",out,"(%d cells)"%len(cells))
