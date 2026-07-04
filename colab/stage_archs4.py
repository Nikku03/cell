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

def main():
    src=find_src()
    if src is None:
        print("stage_archs4: no ARCHS4/GEO h5 found on Drive (expression_geo/ or human_raw/) "
              "-> co-expression / context-networks / condition-multiplier(#1) steps will skip"); return
    if os.path.abspath(src)==os.path.abspath(LOCAL):
        print(f"stage_archs4: ARCHS4/GEO already on local SSD ({os.path.getsize(LOCAL)/1e9:.1f} GB)"); return
    if LOCAL.exists() and abs(LOCAL.stat().st_size-os.path.getsize(src))<1e6:
        print(f"stage_archs4: already staged ({LOCAL.stat().st_size/1e9:.1f} GB) on local SSD -> reusing"); return
    need=os.path.getsize(src); free=shutil.disk_usage(str(H)).free
    print(f"stage_archs4: source '{os.path.basename(src)}' = {need/1e9:.1f} GB on Drive; "
          f"local SSD free = {free/1e9:.1f} GB")
    if free < need*1.08:
        print("stage_archs4: NOT enough local SSD to stage -> leaving on Drive (h5py reads in place, slow). "
              "Use a larger-disk runtime (or free space) to enable fast staging."); return
    print("stage_archs4: copying ARCHS4/GEO -> local SSD (one-time sequential copy; turns hours of random "
          "FUSE seeks into fast local reads) ...")
    tmp=str(LOCAL)+".part"
    shutil.copyfile(src, tmp); os.replace(tmp, LOCAL)
    print(f"stage_archs4: staged -> {LOCAL} ({LOCAL.stat().st_size/1e9:.1f} GB). "
          f"co-expression / context-networks / #1 now read from local SSD.")

if __name__=="__main__":
    main()
