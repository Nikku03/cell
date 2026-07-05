"""FETCH + assemble LINCS L1000 into a training table for the dynamics transformer.

LINCS L1000 (GSE70138 phase-2 default, 5 GB / GSE92742 phase-1 optional, 20 GB) = ~hundreds of thousands of
perturbation signatures (compounds, shRNA KD, cDNA OE, ligands) x 12,328 genes at 6 h / 24 h, commercial-safe
via GEO FTP. We keep the 978 measured LANDMARK genes and build:
  lincs_train.npz : lfc (n_sig x 978 float32), pert (idx), cell (idx), dose (float), time_h (float),
                    gene_symbols[978], pert_names[], cell_names[]
Metadata parsed with pandas; the GCTX matrix with cmapPy (falls back to h5py). Env:
  LINCS_GSE (GSE70138|GSE92742), LINCS_MAX_SIGS (subsample cap), LINCS_PERT_TYPES (csv).
"""
import os, gzip, urllib.request
from pathlib import Path
import numpy as np
H=Path("data/external_data/human"); OUT=Path("outputs/orphan")
GSE=os.environ.get("LINCS_GSE","GSE70138")
MAX_SIGS=int(os.environ.get("LINCS_MAX_SIGS","120000"))
PERT_TYPES=set(os.environ.get("LINCS_PERT_TYPES","trt_cp,trt_sh,trt_oe,trt_lig").split(","))
FILES={
 "GSE70138":dict(gctx="GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz",
                 sig="GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
                 gene="GSE70138_Broad_LINCS_gene_info_2017-03-06.txt.gz", nnn="GSE70nnn"),
 "GSE92742":dict(gctx="GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
                 sig="GSE92742_Broad_LINCS_sig_info.txt.gz",
                 gene="GSE92742_Broad_LINCS_gene_info.txt.gz", nnn="GSE92nnn"),
}

def _url(fn): return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{FILES[GSE]['nnn']}/{GSE}/suppl/{fn}"
def _dl(fn, big=False):
    f=H/fn
    if f.exists() and f.stat().st_size>(1e6 if big else 1000): return f
    print(f"  downloading {fn}{' (large)' if big else ''} ...")
    urllib.request.urlretrieve(_url(fn), f); return f

def _read_tsv_gz(f):
    import pandas as pd
    return pd.read_csv(f, sep="\t", compression="gzip", dtype=str, low_memory=False)

def main():
    OUT.mkdir(parents=True, exist_ok=True); H.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    meta=FILES[GSE]
    gi=_read_tsv_gz(_dl(meta["gene"]))
    lm=gi[gi["pr_is_lm"]=="1"] if "pr_is_lm" in gi else gi
    gene_ids=lm["pr_gene_id"].tolist(); gene_syms=lm["pr_gene_symbol"].tolist()
    print(f"  landmark genes: {len(gene_ids)}")
    si=_read_tsv_gz(_dl(meta["sig"]))
    si=si[si["pert_type"].isin(PERT_TYPES)].copy()
    def _num(s, default):
        try: return float(str(s).split()[0])
        except: return default
    si["time_h"]=si["pert_itime"].map(lambda s:_num(s,6.0))
    si["dose"]=si["pert_idose"].map(lambda s:_num(s,-1.0))
    if len(si)>MAX_SIGS: si=si.sample(MAX_SIGS, random_state=0)
    sig_ids=si["sig_id"].tolist()
    print(f"  signatures kept: {len(sig_ids)} (types {sorted(PERT_TYPES)}); loading GCTX matrix ...")
    gctx=_dl(meta["gctx"], big=True)
    # GCTX is gzipped on GEO; cmapPy needs the raw .gctx (HDF5). decompress once.
    raw=str(gctx)[:-3] if str(gctx).endswith(".gz") else str(gctx)
    if not os.path.exists(raw):
        import shutil
        with gzip.open(gctx,"rb") as i, open(raw,"wb") as o: shutil.copyfileobj(i,o)
    try:
        import warnings; warnings.simplefilter("ignore")     # silence cmapPy's deprecated-pandas FutureWarnings
        from cmapPy.pandasGEXpress.parse import parse
        g=parse(raw, cid=sig_ids, rid=[str(x) for x in gene_ids])
        df=g.data_df                                        # genes x sigs
        lfc=df.T.reindex(sig_ids).values.astype("float32")  # sigs x genes, ordered
        gene_order=[s for s in df.index]                    # gene ids as read
        sym_map=dict(zip(gene_ids, gene_syms))
        gene_syms=[sym_map.get(int(x), sym_map.get(x, str(x))) for x in gene_order]
    except Exception as e:
        print("  cmapPy parse failed, trying h5py:",repr(e)[:100])
        import h5py
        with h5py.File(raw,"r") as h:
            M=h["/0/DATA/0/matrix"]; rid=[x.decode() if isinstance(x,bytes) else str(x) for x in h["/0/META/ROW/id"][:]]
            cid=[x.decode() if isinstance(x,bytes) else str(x) for x in h["/0/META/COL/id"][:]]
            genes_on_rows = len(set(rid)&set(str(x) for x in gene_ids)) > len(set(cid)&set(str(x) for x in gene_ids))
            gset=set(str(x) for x in gene_ids); sset=set(sig_ids)
            if genes_on_rows:
                gidx=[i for i,r in enumerate(rid) if r in gset]; sidx=[i for i,c in enumerate(cid) if c in sset]
                lfc=M[gidx,:][:,sidx].T.astype("float32"); gene_order=[rid[i] for i in gidx]; sig_ids=[cid[i] for i in sidx]
            else:
                gidx=[i for i,c in enumerate(cid) if c in gset]; sidx=[i for i,r in enumerate(rid) if r in sset]
                lfc=M[sidx,:][:,gidx].astype("float32"); gene_order=[cid[i] for i in gidx]; sig_ids=[rid[i] for i in sidx]
        sym_map=dict(zip([str(x) for x in gene_ids], gene_syms)); gene_syms=[sym_map.get(x,x) for x in gene_order]
    si=si.set_index("sig_id").reindex(sig_ids)
    pert_names=si["pert_iname"].fillna("?").tolist(); cell_names=si["cell_id"].fillna("?").tolist()
    up={p:i for i,p in enumerate(sorted(set(pert_names)))}; uc={c:i for i,c in enumerate(sorted(set(cell_names)))}
    np.savez_compressed(OUT/"lincs_train.npz",
        lfc=lfc, pert=np.array([up[p] for p in pert_names],dtype=np.int32),
        cell=np.array([uc[c] for c in cell_names],dtype=np.int32),
        dose=si["dose"].values.astype("float32"), time_h=si["time_h"].values.astype("float32"),
        gene_symbols=np.array(gene_syms), pert_names=np.array(sorted(set(pert_names))),
        cell_names=np.array(sorted(set(cell_names))), pert_of_row=np.array(pert_names))
    print(f"LINCS training table: {lfc.shape[0]} signatures x {lfc.shape[1]} landmark genes | "
          f"{len(up)} perturbations, {len(uc)} cell lines | times {sorted(set(si['time_h']))[:6]} "
          f"-> lincs_train.npz")

if __name__=="__main__":
    main()
