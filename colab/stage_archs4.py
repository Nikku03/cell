"""Stage the ARCHS4 expression h5 (== the GEO RNA-seq compendium, uniformly reprocessed; ~59 GB) from
Google Drive onto LOCAL SSD before any consumer reads it.

Why: h5py does many small random seeks, and over the Google Drive FUSE mount each seek is very high
latency -- reading the matrix in place is so slow it effectively hangs. A one-time sequential copy to
local disk fixes the access pattern. We copy to data/external_data/human/archs4_human_gene.h5, which is
the FIRST path every ARCHS4 finder checks (compute_coexpr / compute_context_networks /
fit_condition_multipliers), so they all transparently switch to the fast local copy -- no env needed.

Falls back to leaving the file on Drive (read in place) if local SSD can't hold it. -> local h5 on SSD
"""
import os, shutil
from pathlib import Path
H=Path("data/external_data/human"); H.mkdir(parents=True, exist_ok=True)
LOCAL=H/"archs4_human_gene.h5"
DIRS=["/content/drive/MyDrive/virtual_cell_data/expression_geo",
      "/content/drive/MyDrive/virtual_cell_data/human_raw", str(H)]

def find_src():
    """largest ARCHS4/GEO .h5 across the Drive folders (and H, in case it is already local)."""
    cands=[]
    for d in DIRS:
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            low=fn.lower()
            if low.endswith(".h5") and ("archs4" in low or ("human" in low and "gene" in low)):
                p=os.path.join(d,fn)
                try:
                    if os.path.getsize(p)>1e7: cands.append(p)
                except OSError: pass
    return max(cands, key=os.path.getsize) if cands else None

def valid_h5(p):
    """True if p opens as HDF5 with the datasets the readers need (rejects truncated/partial files)."""
    try:
        import h5py
        with h5py.File(p,"r") as h:
            has_expr = ("data/expression" in h) or ("expression" in h)
            has_genes = any(k in h for k in ("meta/genes/symbol","meta/genes/genes","meta/genes/gene_symbol"))
            return bool(has_expr and has_genes)
    except Exception as e:
        print(f"  stage_archs4: '{os.path.basename(p)}' is NOT a valid/complete HDF5 ({repr(e)[:60]})"); return False

def main():
    # clean any stale LOCAL partial from a previously-killed copy (never touches the Drive .part)
    stale=str(LOCAL)+".part"
    if os.path.exists(stale):
        try: os.remove(stale); print(f"stage_archs4: removed stale local partial {stale}")
        except OSError: pass
    src=find_src()                                    # find_src already ignores anything not ending .h5 (so .part is never picked)
    if src is None:
        print("stage_archs4: no ARCHS4/GEO .h5 found (expression_geo/ or human_raw/) -> coexpr / context / #1 will skip"); return
    print(f"stage_archs4: found ARCHS4 -> {src} ({os.path.getsize(src)/1e9:.1f} GB)")
    if not valid_h5(src):
        print("stage_archs4: the .h5 is unreadable/incomplete -> coexpr / context / #1 will skip. "
              "Re-upload a complete archs4_human_gene.h5 (the .part is an unfinished download; delete it)."); return
    # DEFAULT: read in place from Drive (reliable; worked in prior runs). Copy to SSD only if opted in.
    if os.environ.get("STAGE_ARCHS4","0")!="1":
        print("stage_archs4: reading ARCHS4 IN PLACE from Drive (validated). "
              "Set STAGE_ARCHS4=1 to copy it to local SSD for faster reads (needs ~64 GB free; risks a long copy)."); return
    if os.path.abspath(src)==os.path.abspath(LOCAL):
        print(f"stage_archs4: already local ({os.path.getsize(LOCAL)/1e9:.1f} GB)"); return
    if LOCAL.exists() and valid_h5(LOCAL) and abs(LOCAL.stat().st_size-os.path.getsize(src))<1e6:
        print(f"stage_archs4: already staged ({LOCAL.stat().st_size/1e9:.1f} GB) -> reusing"); return
    need=os.path.getsize(src); free=shutil.disk_usage(str(H)).free
    if free < need*1.08:
        print(f"stage_archs4: STAGE_ARCHS4=1 but only {free/1e9:.1f} GB free < {need/1e9:.1f} GB needed "
              "-> reading in place from Drive instead."); return
    print(f"stage_archs4: copying {need/1e9:.1f} GB -> local SSD (one-time) ...")
    shutil.copyfile(src, stale); os.replace(stale, LOCAL)
    print(f"stage_archs4: staged -> {LOCAL} ({LOCAL.stat().st_size/1e9:.1f} GB).")

if __name__=="__main__":
    main()
