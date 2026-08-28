"""
TAHOE-100M PSEUDOBULK STREAMER -- resumable, nothing large ever touches disk.

Loop 265 measured cross-line transfer rising +0.00094 per added cell line with no sign of
flattening, and extrapolated ~61 lines to close the gap to a line's own operator. LINCS is
exhausted: 20 cell lines exist for shRNA and 12 are usable. Tahoe-100M has 50, one platform,
one study, CC0.

WHAT THE PROBE ESTABLISHED, before writing any of this:
  - the pseudobulk table is 4,089,820,780 rows over 1,026 parquet shards, 88.9 GB
  - EACH SHARD HOLDS EXACTLY ONE CELL LINE, so shards can be sampled per line
  - row groups are 1,000 rows; one (drug, dose) condition spans ~62.7 of them
  - so a shard is ~63 conditions, and 1,026 shards / 50 lines is ~20 shards per line
  - reading 5 of 16 columns costs 0.5s per 126 row groups -> ~17s per whole shard

Nothing is downloaded. pyarrow issues HTTP range requests for the column chunks it needs,
so only the five columns below cross the wire and only for the row groups requested.

RESUMABLE BY DESIGN. The container running this session rebooted once already and killed a
77-minute job with no traceback. Each shard writes its own .npz and is skipped if present,
so an interrupted run resumes at the shard it died on rather than from the start.
"""
import sys, json, time, collections
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fsspec

SCR = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
OUTD = SCR / "tahoe"
BASE = ("https://huggingface.co/api/datasets/tahoebio/Tahoe-100M/parquet/"
        "pseudobulk_differential_expression/train")
COLS = ["gene_name", "log2FoldChange", "drug", "concentration", "plate"]
SHARDS_PER_LINE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NSHARD = 1026


def manifest():
    """Which cell line is in which shard. Reads row group 0's cell-line column only."""
    mf = OUTD / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    fs = fsspec.filesystem("http")
    m = {}
    for sh in range(NSHARD):
        for attempt in range(4):
            try:
                f = pq.ParquetFile(fs.open(f"{BASE}/{sh}.parquet"))
                t = f.read_row_groups([0], columns=["Cell_ID_DepMap"])
                m[str(sh)] = t.column("Cell_ID_DepMap")[0].as_py()
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  shard {sh}: FAILED {type(e).__name__}", flush=True)
                    m[str(sh)] = None
                time.sleep(2 ** attempt)
        if sh % 100 == 0:
            print(f"  manifest {sh}/{NSHARD}", flush=True)
    mf.write_text(json.dumps(m))
    return m


def extract(sh, lmset):
    """One shard -> (conditions x landmark genes) matrix of log2 fold changes."""
    out = OUTD / f"shard_{sh:04d}.npz"
    if out.exists():
        return "skip"
    fs = fsspec.filesystem("http")
    f = pq.ParquetFile(fs.open(f"{BASE}/{sh}.parquet"))
    t = f.read(columns=COLS)
    gn = np.asarray(t.column("gene_name"))
    lfc = np.asarray(t.column("log2FoldChange"), dtype=np.float32)
    dr = np.asarray(t.column("drug")); co = np.asarray(t.column("concentration"))
    pl = np.asarray(t.column("plate"))
    keep = np.array([g in lmset for g in gn])
    gn, lfc, dr, co, pl = gn[keep], lfc[keep], dr[keep], co[keep], pl[keep]
    conds = collections.defaultdict(dict)
    for g, v, d, c, p in zip(gn, lfc, dr, co, pl):
        conds[(str(d), float(c), str(p))][str(g)] = v
    genes = sorted(lmset)
    gpos = {g: i for i, g in enumerate(genes)}
    keys = sorted(conds)
    M = np.full((len(keys), len(genes)), np.nan, np.float32)
    for i, k in enumerate(keys):
        for g, v in conds[k].items():
            M[i, gpos[g]] = v
    np.savez_compressed(out, M=M, drug=np.array([k[0] for k in keys]),
                        conc=np.array([k[1] for k in keys], np.float32),
                        plate=np.array([k[2] for k in keys]), genes=np.array(genes))
    return f"{len(keys)} conditions"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))
    import lincs_harness as H
    D = H.load()
    lmset = set(str(x) for x in D["lmsym"] if x != "?")
    print(f"landmark genes to keep: {len(lmset)}", flush=True)

    print("building shard -> cell line manifest ...", flush=True)
    m = manifest()
    byline = collections.defaultdict(list)
    for sh, cl in m.items():
        if cl: byline[cl].append(int(sh))
    print(f"{len(byline)} cell lines across {sum(len(v) for v in byline.values())} shards",
          flush=True)

    todo = []
    for cl in sorted(byline):
        todo += sorted(byline[cl])[:SHARDS_PER_LINE]
    print(f"extracting {len(todo)} shards ({SHARDS_PER_LINE} per line) ...", flush=True)
    t0 = time.time()
    for i, sh in enumerate(todo):
        for attempt in range(4):
            try:
                r = extract(sh, lmset)
                break
            except Exception as e:
                r = f"ERR {type(e).__name__}"
                time.sleep(2 ** attempt)
        if i % 10 == 0 or r.startswith("ERR"):
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] shard {sh} {m[str(sh)]} -> {r}   "
                  f"{el:.0f}s elapsed, {el/(i+1)*(len(todo)-i-1)/60:.0f} min left", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
