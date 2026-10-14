"""Fetch the full ENCODE K562 TF-ChIP compendium and build element x TF binding for CRISPR elements."""
import urllib.request, json, gzip, os, bisect, time
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
OUT = "outputs/orphan/invivo"
refs = json.load(open(f"{SP}/k562_tf_chip.json"))

# --- unique CRISPR elements ---
rows = open(f"{OUT}/crispr_egpairs.tsv").read().splitlines()
h = rows[0].split("\t"); ci = {c: i for i, c in enumerate(h)}
elems = {}                                   # key -> (chrom, s, e)
for line in rows[1:]:
    p = line.split("\t")
    if p[ci["ValidConnection"]] != "TRUE":
        continue
    key = f'{p[ci["chrom"]]}:{p[ci["chromStart"]]}-{p[ci["chromEnd"]]}'
    elems[key] = (p[ci["chrom"]], int(p[ci["chromStart"]]), int(p[ci["chromEnd"]]))
keys = list(elems)
kidx = {k: i for i, k in enumerate(keys)}
# per-chrom sorted element intervals
bychr = {}
for k, (c, s, e) in elems.items():
    bychr.setdefault(c, []).append((s, e, kidx[k]))
for c in bychr:
    bychr[c].sort()
print(f"unique CRISPR elements: {len(keys)}", flush=True)

tf_hits = [set() for _ in keys]              # per element: set of TF indices that bind it
tf_list = sorted(refs)
tf_ok = []

def overlap_elems(chrom, ps, pe):
    arr = bychr.get(chrom)
    if not arr:
        return
    starts = [x[0] for x in arr]
    j = bisect.bisect_right(starts, pe)
    i = j - 1
    while i >= 0 and arr[i][0] >= ps - 10000:
        s, e, eid = arr[i]
        if e >= ps and s <= pe:
            yield eid
        i -= 1

for ti, tf in enumerate(tf_list):
    href = refs[tf]["href"]; url = "https://www.encodeproject.org" + href
    tmp = f"{SP}/_tf.bed.gz"
    got = False
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, tmp); got = True; break
        except Exception:
            time.sleep(1.0)
    if not got:
        continue
    try:
        with gzip.open(tmp, "rt") as fh:
            for line in fh:
                p = line.split("\t")
                if len(p) < 3:
                    continue
                for eid in overlap_elems(p[0], int(p[1]), int(p[2])):
                    tf_hits[eid].add(ti)
        tf_ok.append(tf)
    except Exception:
        pass
    if (ti + 1) % 25 == 0:
        print(f"{ti+1}/{len(tf_list)} TFs processed ({len(tf_ok)} ok)", flush=True)
os.path.exists(f"{SP}/_tf.bed.gz") and os.remove(f"{SP}/_tf.bed.gz")

# save: per-element tf_count + which TFs (indices) -> for compendium feature + leak check
out = {"tf_list": tf_list, "n_tf_ok": len(tf_ok),
       "element_tf_count": {keys[i]: len(tf_hits[i]) for i in range(len(keys))},
       "element_tfs": {keys[i]: sorted(tf_hits[i]) for i in range(len(keys))}}
json.dump(out, open(f"{OUT}/compendium_tf.json", "w"))
import statistics
counts = [len(s) for s in tf_hits]
print(f"DONE: {len(tf_ok)} TFs intersected. tf_count per element: mean {statistics.mean(counts):.1f}, "
      f"median {statistics.median(counts)}, max {max(counts)} -> {OUT}/compendium_tf.json", flush=True)
