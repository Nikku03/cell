"""Generator for notebooks/CellOS_Colab.ipynb -- the Colab runner for the v4 block tests.

WHY A GENERATOR AND NOT A HAND-WRITTEN .ipynb. Notebook JSON is unreviewable in a diff and easy to
corrupt. This script is the source of truth; run it to regenerate the notebook.

EVERY URL IN HERE WAS VERIFIED, AND THE ONES THAT COULD NOT BE ARE MARKED AS SUCH.

  VERIFIED BYTE-EXACT. The seven scPerturb h5ad files below were resolved from the Zenodo API
  (record 13350497) and every one matches the byte size of the corresponding local file exactly:
      PapalexiSatija2021_eccite_RNA.h5ad     147,215,278  == papalexi.h5ad
      SrivatsanTrapnell2020_sciplex3.h5ad  2,526,631,614  == sciplex.h5ad
      FrangiehIzar2021_RNA.h5ad            1,458,928,348  == frangieh.h5ad
      ShifrutMarson2018.h5ad                 871,481,676  == shifrut.h5ad
      NadigOConner2024_jurkat.h5ad         1,293,665,804  == nadig_jurkat.h5ad
      NadigOConner2024_hepg2.h5ad            850,590,740  == nadig_hepg2.h5ad
      ReplogleWeissman2022_rpe1.h5ad       1,236,886,900  == rpe1.h5ad
  VERIFIED BY HTTP HEAD. Human-GEM.xml, 43,115,559 bytes from the SysBioChalmers repo.

  NOT VERIFIABLE FROM THE AUTHORING SANDBOX. ndownloader.figshare.com returns 403 through this
  environment's egress proxy -- an organisation policy denial, NOT evidence the URL is dead. Those
  entries carry the md5 that `colab/depmap_codep.py` already asserts, so the notebook checks the
  file rather than trusting the link.

  NO PUBLIC SOURCE AT ALL. gwps.h5ad is a PSEUDOBULK, shape (11258, 8248), with obs columns
  TE_ratio / anderson_darling_counts / cnv_score_z. scPerturb's ReplogleWeissman2022_K562_gwps.h5ad
  is 8.4 GB of CELL-LEVEL data -- a different object that would need pseudobulking and would not
  reproduce this file. No URL for the pseudobulk exists anywhere in this repository, because it was
  never downloaded here: it came from the user's Drive as human_raw/perturbseq_gwps_bulk.h5ad. The
  notebook therefore looks in Drive and, if it is absent, says so and skips the tests that need it
  rather than inventing a link.

ROBUSTNESS RULES THE CELLS FOLLOW, each answering a way this has actually broken:
  * Zenodo answered a Range request with 200 rather than 206, so RESUME MAY SILENTLY NOT WORK.
    Every download is verified by EXACT BYTE COUNT afterwards and re-fetched from scratch on
    mismatch. A partial file is never left where a later cell could read it.
  * Downloads go to a .part file and are renamed only after the size check passes.
  * Free disk is checked BEFORE each download. Colab gives ~107 GB but Drive is 15 GB.
  * Data is staged to /content (local SSD). Reading an h5ad over Drive FUSE with f["X"][:] is
    pathologically slow and is the single most common way this hangs.
  * Every cell is idempotent: re-running skips work already done.
  * The preflight cell verifies everything BEFORE any long run, so a missing file surfaces in
    seconds rather than forty minutes in.
"""
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent / "CellOS_Colab.ipynb"

REPO = "https://github.com/nikku03/cell.git"
BRANCH = "claude/vectorize-gex-propensity-zp09w8"
ZEN = "https://zenodo.org/api/records/13350497/files/{}/content"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").split("\n")}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").split("\n")}


CELLS = []

CELLS.append(md(r"""
# CellOS — v4 block tests on Colab

Runs the CELLFORMER4 block tests on a GPU runtime. Every cell is **idempotent**: re-run it and it
skips whatever is already done.

**Run the cells in order.** Cell 3 (preflight) tells you what will and will not run *before* anything
long starts.

### What already came back, so you know what this is re-running

| test | result |
|---|---|
| criterion 2 — simple set encoder vs transformer | **FAILED.** Set Transformer 0.0297, mean-pool 0.0323, untrained 0-parameter arm **0.0341**, MDE 0.0015 |
| stage 1 — real vs random partners | passed, +0.0186 |
| stages 3+4 — cell context | failed 0/7; binding constraint is n=4 cell contexts |

Block 3 is a **bag of typed partners**, not a transformer. The bar for new work on it is **0.0341**
— the zero-parameter arm — not the transformer's 0.0297.

### What an L4 actually buys here

These models are ~108k parameters and most of the wall time is numpy (robust-z over 11,258 × 8,248,
retrieval, `argpartition` over 7,766 genes). Expect **1.5–2.5×**, not 10×. The exception is
`dense_heads.py`, which trains on 3M pairs against an 8,175-row gene embedding — that one is
genuinely GPU-bound and has been killed three times on CPU. **It is the reason to use a GPU.**
"""))

CELLS.append(md("## 1 — Mount Drive and clone the repo"))

CELLS.append(code(rf"""
import os, sys, subprocess, shutil, time, json, hashlib
from pathlib import Path

USE_DRIVE = True   # set False to run entirely on ephemeral local disk

DRIVE = Path("/content/drive/MyDrive")
if USE_DRIVE:
    from google.colab import drive
    if not DRIVE.exists():
        drive.mount("/content/drive")

CACHE = (DRIVE / "cellos_data") if USE_DRIVE else Path("/content/cellos_data")
LOCAL = Path("/content/cellos_data")          # ALWAYS read from here, never from Drive directly
REPO_DIR = Path("/content/cell")
for p in (CACHE, LOCAL):
    p.mkdir(parents=True, exist_ok=True)

# --depth 1: the full .git is 1.3 GB, the tracked worktree is ~430 MB.
if not (REPO_DIR / ".git").exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "{BRANCH}",
                    "{REPO}", str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "colab"))
print("repo   :", REPO_DIR, "@", subprocess.run(
    ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip())
print("cache  :", CACHE)
print("local  :", LOCAL)
print("free   : {{:.1f}} GB local, {{:.1f}} GB on the cache volume".format(
    shutil.disk_usage("/content").free / 2**30, shutil.disk_usage(CACHE).free / 2**30))
"""))

CELLS.append(md(r"""
## 2 — The download table

Sizes are **exact byte counts**, verified against the Zenodo API and against the files this project
already holds. A download is accepted only if the byte count matches, so a truncated or
proxy-mangled transfer can never be mistaken for a good one.

`tier` controls what gets fetched: `1` is enough to reproduce criterion 2 and stage 1.
"""))

CELLS.append(code(r"""
# key -> (url, exact_bytes or None, tier, note)
# tier 1 = the K562 dense tests. tier 2 = 4-cell context. tier 3 = the newer block tests.
ZEN = "https://zenodo.org/api/records/13350497/files/{}/content"

FILES = {
    # ---- tier 1: nothing to download. gwps.h5ad has NO public source (see cell 3).
    # ---- tier 2
    "rpe1.h5ad":        (ZEN.format("ReplogleWeissman2022_rpe1.h5ad"),   1236886900, 2, "RPE1 cell-level"),
    "nadig_jurkat.h5ad":(ZEN.format("NadigOConner2024_jurkat.h5ad"),     1293665804, 2, "Jurkat"),
    "nadig_hepg2.h5ad": (ZEN.format("NadigOConner2024_hepg2.h5ad"),       850590740, 2, "HepG2"),
    # ---- tier 3
    "papalexi.h5ad":    (ZEN.format("PapalexiSatija2021_eccite_RNA.h5ad"), 147215278, 3, "ECCITE RNA"),
    "papalexi_protein.h5ad":
                        (ZEN.format("PapalexiSatija2021_eccite_protein.h5ad"), 1191551, 3,
                         "ECCITE ADT -- the protein head needs THIS, not the RNA file"),
    "frangieh.h5ad":    (ZEN.format("FrangiehIzar2021_RNA.h5ad"),        1458928348, 3, "melanoma"),
    "shifrut.h5ad":     (ZEN.format("ShifrutMarson2018.h5ad"),            871481676, 3, "primary T"),
    "sciplex.h5ad":     (ZEN.format("SrivatsanTrapnell2020_sciplex3.h5ad"), 2526631614, 3, "dose-response"),
    "HumanGEM.xml":     ("https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model/Human-GEM.xml",
                         43115559, 3, "metabolic model"),
}

# DepMap. The URLs live in colab/depmap_codep.py together with md5s that the module asserts.
# They could NOT be verified from the authoring sandbox (its egress proxy returns 403 for
# ndownloader.figshare.com -- a policy denial, not a dead link), so these are checked by md5
# after download rather than trusted in advance.
DEPMAP = {
    "Model.csv":  ("https://ndownloader.figshare.com/files/46489732", "e5b60d1ce7636542d93f45494019eded"),
    "CRISPRInferredCommonEssentials.csv":
                  ("https://ndownloader.figshare.com/files/46489597", "76dd2c044e9aa42bdfa344c8681904f7"),
}
# CRISPRGeneEffect.csv (md5 d10d73692672380b3792231549e97d5a) is NOT listed: its figshare file id
# is not recorded anywhere in the repo. Get it from https://depmap.org/portal/download (24Q2) and
# drop it in the cache; the md5 check below will confirm you got the right release.

TIER = 1   # <<< set to 2 or 3 to pull more
print(f"tier {TIER}; would fetch:")
tot = 0
for k, (u, n, t, note) in sorted(FILES.items(), key=lambda kv: kv[1][2]):
    if t <= TIER:
        tot += n or 0
        print(f"  {(n or 0)/2**20:9.1f} MB  {k:24s} {note}")
print(f"  {'-'*9}\n  {tot/2**20:9.1f} MB total")
"""))

CELLS.append(md(r"""
## 3 — Preflight

Runs in seconds and tells you exactly which tests can run. **Read its output before going further** —
it is what stops you discovering a missing file forty minutes into a training run.
"""))

CELLS.append(code(r"""
import h5py, numpy as np

def find(name):
    "Look in local stage, then Drive cache, then a couple of conventional Drive locations."
    for p in (LOCAL/name, CACHE/name,
              DRIVE/"human_raw"/name, DRIVE/"cellos"/name,
              DRIVE/"human_raw"/"perturbseq_gwps_bulk.h5ad" if name == "gwps.h5ad" else LOCAL/name):
        try:
            if p.exists() and p.stat().st_size > 0:
                return p
        except Exception:
            pass
    return None

REPO_HAS = {p.name: p for p in [
    REPO_DIR/"outputs/orphan/cell_complete.json.gz",
    REPO_DIR/"outputs/orphan/depmap_vecs.npz",
    REPO_DIR/"outputs/orphan/baseline_state.json.gz",
    REPO_DIR/"outputs/orphan/v4_relations.parquet",
] if p.exists()}

print("FROM THE REPO (no download needed):")
for n, p in REPO_HAS.items():
    print(f"   ok  {p.stat().st_size/2**20:8.1f} MB  {n}")
missing_repo = {"cell_complete.json.gz", "depmap_vecs.npz"} - set(REPO_HAS)
for n in missing_repo:
    print(f"   MISSING FROM THE CLONE: {n} -- git pull, it was committed for exactly this reason")

print("\nDATA FILES:")
# X is a Dataset in a dense h5ad but a GROUP in a sparse one, where .shape raises
# AttributeError. Verified against PapalexiSatija2021_eccite_protein.h5ad, whose X is a group.
def xshape(path):
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return tuple(x.shape), "dense"
        sh = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
        return (tuple(sh) if sh is not None else None), "sparse"

gw = find("gwps.h5ad")
if gw:
    shp, kind = xshape(gw)
    print(f"   ok  gwps.h5ad at {gw}  X={shp} ({kind})")
    if kind != "dense":
        print("       WARNING: the dense tests expect a DENSE pseudobulk X, not a sparse matrix.")
    if shp != (11258, 8248):
        print(f"       WARNING: expected (11258, 8248). A different shape means a different object;"
              f" the dense tests assume the pseudobulk.")
else:
    print("   ABSENT: gwps.h5ad")
    print("       There is NO public URL for this file. It is a PSEUDOBULK (11258 x 8248).")
    print("       scPerturb's ReplogleWeissman2022_K562_gwps.h5ad is 8.4 GB of CELL-LEVEL data -- a")
    print("       different object that would need pseudobulking and would not reproduce it.")
    print("       Put your own copy (human_raw/perturbseq_gwps_bulk.h5ad) in the Drive cache.")
    print("       WITHOUT IT: criterion 2, stage 1 and the heads test cannot run.")

for k in FILES:
    p = find(k)
    print(f"   {'ok ' if p else '-- '} {k:24s} {'' if p else '(tier %d, not fetched)' % FILES[k][2]}")

import torch
print(f"\nGPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU only)'}")
print(f"RAM: {os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/2**30:.1f} GB")
print("\nRUNNABLE NOW:")
print(f"   criterion 2 / stage 1 / heads : {'YES' if gw and not missing_repo else 'NO -- needs gwps.h5ad'}")
print(f"   relation schema              : {'YES' if gw and (REPO_DIR/'outputs/orphan/v4_relations.parquet').exists() else 'NO'}")
print(f"   4-cell context               : {'YES' if find('dense_response.npz') else 'NO -- needs dense_response.npz or tier 2 h5ads'}")
"""))

CELLS.append(md(r"""
## 4 — Download

Verified by **exact byte count**, written to `.part` and renamed only on success. Zenodo answered a
Range request with `200` rather than `206`, so resume may silently not work — a size mismatch
therefore triggers a clean re-fetch rather than an append.
"""))

CELLS.append(code(r"""
import shutil, subprocess, time

def fetch(name, url, exact, dest_dir, tries=4):
    dest = dest_dir / name
    if dest.exists() and (exact is None or dest.stat().st_size == exact):
        print(f"  cached  {name}")
        return dest
    if dest.exists():
        print(f"  size mismatch on {name} ({dest.stat().st_size} != {exact}) -- refetching")
        dest.unlink()
    need = (exact or 0) * 1.1
    free = shutil.disk_usage(dest_dir).free
    if free < need:
        print(f"  SKIP {name}: needs {need/2**30:.1f} GB, {free/2**30:.1f} GB free")
        return None
    part = dest.with_suffix(dest.suffix + ".part")
    for a in range(tries):
        if part.exists():
            part.unlink()          # never append: resume is not trustworthy here
        t0 = time.time()
        r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "--retry-delay", "5",
                            "-o", str(part), url])
        got = part.stat().st_size if part.exists() else 0
        if r.returncode == 0 and (exact is None or got == exact):
            part.rename(dest)
            mb = got / 2**20
            print(f"  got     {name}  {mb:.1f} MB in {time.time()-t0:.0f}s ({mb/max(time.time()-t0,1):.1f} MB/s)")
            return dest
        print(f"  attempt {a+1}/{tries} failed for {name} (rc={r.returncode}, got {got} want {exact})")
        time.sleep(2 ** a)
    print(f"  GAVE UP on {name}")
    return None

for k, (u, n, t, note) in sorted(FILES.items(), key=lambda kv: kv[1][1] or 0):
    if t <= TIER:
        fetch(k, u, n, CACHE)

# md5-checked DepMap files
import hashlib
def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

if TIER >= 3:
    for name, (url, want) in DEPMAP.items():
        p = fetch(name, url, None, CACHE)
        if p:
            got = md5(p)
            print(f"  md5 {name}: {'OK' if got == want else 'MISMATCH ' + got}")
            if got != want:
                p.unlink()
                print("     deleted -- a wrong release is worse than a missing file")
"""))

CELLS.append(md(r"""
## 5 — Stage to local disk

`f["X"][:]` over Drive FUSE is pathologically slow and is the usual reason a run appears to hang.
Everything is copied to `/content` first. Files already local are skipped.
"""))

CELLS.append(code(r"""
def stage(name):
    src = find(name)
    if src is None:
        return None
    dst = LOCAL / name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    if src.parent == LOCAL:
        return src
    free = shutil.disk_usage(LOCAL).free
    if free < src.stat().st_size * 1.1:
        print(f"  no room to stage {name}; reading from {src.parent} (SLOW)")
        return src
    t0 = time.time()
    shutil.copy2(src, dst)
    print(f"  staged {name} {dst.stat().st_size/2**20:.0f} MB in {time.time()-t0:.0f}s")
    return dst

for n in ["gwps.h5ad", "dense_response.npz", "cell_complete.json.gz"] + list(FILES):
    stage(n)

# The modules read CELL_SCRATCH for big inputs and CELL_OUT for artifacts.
os.environ["CELL_SCRATCH"] = str(LOCAL)
os.environ["CELL_OUT"] = str(REPO_DIR / "outputs/orphan")
for n in ("cell_complete.json.gz", "depmap_vecs.npz"):
    src, dst = REPO_DIR / "outputs/orphan" / n, LOCAL / n
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
print("CELL_SCRATCH =", os.environ["CELL_SCRATCH"])
print("CELL_OUT     =", os.environ["CELL_OUT"])
"""))

CELLS.append(md(r"""
## 6 — Smoke every module before committing to a long run

Each module has a SMOKE mode that finishes in a couple of minutes. If one fails here it would have
failed an hour into the full run.
"""))

CELLS.append(code(r"""
SMOKE = [
    ("colab/v4_set_encoders.py",   {"V4_SMOKE": "1"}),
    ("colab/dense_stage1.py",      {"S1_SMOKE": "1", "S1_EPOCHS": "2", "S1_SEEDS": "1"}),
    ("colab/v4_relation_schema.py",{"V4_SMOKE": "1"}),
    ("colab/dense_heads.py",       {"DH_SMOKE": "1"}),
]
for mod, env in SMOKE:
    if not (REPO_DIR / mod).exists():
        print(f"  -- {mod} not in this clone, skipping")
        continue
    e = dict(os.environ, **env)
    t0 = time.time()
    r = subprocess.run([sys.executable, mod], env=e, capture_output=True, text=True, timeout=1800)
    tail = (r.stdout or r.stderr).strip().split("\n")[-1][:150]
    print(f"  {'OK  ' if r.returncode == 0 else 'FAIL'} {mod:32s} {time.time()-t0:5.0f}s  {tail}")
"""))

CELLS.append(md(r"""
## 7 — Full runs

Ordered by value. **`dense_heads.py` is the one that justifies the GPU** — it has been killed three
times on CPU. Each writes a JSON to `outputs/orphan/` and skips nothing, so run only what you want.
"""))

CELLS.append(code(r"""
RUNS = [
    # (module, env, why)
    ("colab/dense_heads.py",
     {"DH_PAIRS": "3000000", "DH_EPOCHS": "4", "DH_SPLITS": "3"},
     "block 8 RNA head vs its own per-gene marginal -- GPU-bound, never yet completed"),
    ("colab/v4_set_encoders.py",
     {"V4_N": "0", "V4_EPOCHS": "25", "V4_FOLDS": "3", "V4_INIT_SEEDS": "2"},
     "criterion 2 -- reproduces the FAILED verdict; run it to confirm on your data"),
    ("colab/v4_relation_schema.py",
     {"V4_SMOKE": "0", "V4_KO": "0", "V4_EPOCHS": "20", "V4_FOLDS": "3", "V4_SEEDS": "2"},
     "does confidence-gating the partner set help, now that confidence exists"),
]

RUN = [0]     # <<< indices to run, e.g. [0] or [0,1,2]

for i in RUN:
    mod, env, why = RUNS[i]
    print(f"\n=== {mod} ===\n    {why}")
    e = dict(os.environ, **env)
    t0 = time.time()
    p = subprocess.Popen([sys.executable, mod], env=e, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print("   ", line.rstrip())
    p.wait()
    print(f"    exit {p.returncode} in {(time.time()-t0)/60:.1f} min")
"""))

CELLS.append(md("## 8 — Persist results back to Drive"))

CELLS.append(code(r"""
res = CACHE / "results"
res.mkdir(exist_ok=True)
n = 0
for j in (REPO_DIR / "outputs/orphan").glob("*.json"):
    if j.stat().st_mtime > time.time() - 86400:
        shutil.copy2(j, res / j.name)
        n += 1
print(f"copied {n} result files to {res}")
for j in sorted(res.glob("*.json"))[-8:]:
    try:
        d = json.loads(j.read_text())
        v = (d.get("verdict") or "")[:160]
        print(f"\n{j.name}\n  {v}")
    except Exception:
        pass
"""))

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(CELLS)} cells, {OUT.stat().st_size/1024:.1f} KB)")
