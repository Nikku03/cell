"""Build the genome-wide auxiliary-pretraining dataset: ~100k ENCODE cCREs x 311-TF ChIP binding + their 600bp sequences.
This is the dense data that solves the CRISPR label-starvation problem for the shared encoder."""
import json, gzip, urllib.request, time, bisect, random, os
import numpy as np
import twobitreader
SP = "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"
OUT = "outputs/orphan/invivo/pretrain"
os.makedirs(OUT, exist_ok=True)
W = 300  # 600bp window
N = 100000

# --- 1. subsample cCREs (deterministic) ---
lines = open(f"{SP}/cCREs.bed").read().splitlines()
random.seed(0); random.shuffle(lines)
cres = []
for l in lines:
    p = l.split("\t")
    if p[0] in ("chrM", "chrY"):
        continue
    cres.append((p[0], int(p[1]), int(p[2])))
    if len(cres) >= N:
        break
print(f"cCREs sampled: {len(cres)}", flush=True)

# --- 2. 311-TF binding labels (intersect ChIP peaks with cCREs) ---
bychr = {}
for i, (c, s, e) in enumerate(cres):
    bychr.setdefault(c, []).append((s, e, i))
for c in bychr:
    bychr[c].sort()
starts = {c: [x[0] for x in v] for c, v in bychr.items()}

def overlap_ids(chrom, ps, pe):
    v = bychr.get(chrom)
    if not v:
        return
    j = bisect.bisect_right(starts[chrom], pe); i = j - 1
    while i >= 0 and v[i][0] >= ps - 5000:
        s, e, idx = v[i]
        if e >= ps and s <= pe:
            yield idx
        i -= 1

refs = json.load(open(f"{SP}/k562_tf_chip.json")); tfl = sorted(refs)
labels = np.zeros((len(cres), len(tfl)), dtype=np.uint8)
for ti, tf in enumerate(tfl):
    url = "https://www.encodeproject.org" + refs[tf]["href"]
    tmp = f"{SP}/_ptf.bed.gz"; ok = False
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, tmp); ok = True; break
        except Exception:
            time.sleep(1.0)
    if not ok:
        continue
    try:
        with gzip.open(tmp, "rt") as fh:
            for line in fh:
                p = line.split("\t")
                if len(p) < 3:
                    continue
                for idx in overlap_ids(p[0], int(p[1]), int(p[2])):
                    labels[idx, ti] = 1
    except Exception:
        pass
    if (ti + 1) % 50 == 0:
        print(f"  ChIP {ti+1}/{len(tfl)} intersected", flush=True)
os.path.exists(f"{SP}/_ptf.bed.gz") and os.remove(f"{SP}/_ptf.bed.gz")
print(f"labels: mean TFs/cCRE {labels.sum(1).mean():.1f}, cCREs with >=1 TF {int((labels.sum(1)>0).sum())}", flush=True)

# --- 3. extract 600bp sequences from hg38.2bit (wait for the genome download to finish) ---
while not os.path.exists(f"{SP}/genome.flag"):
    print("  waiting for genome download...", flush=True); time.sleep(15)
gen = twobitreader.TwoBitFile(f"{SP}/hg38.2bit")
order = sorted(range(len(cres)), key=lambda i: cres[i][0])   # group by chrom for cache efficiency
seqs = [None] * len(cres)
cur = None; chromseq = None
for cnt, i in enumerate(order):
    c, s, e = cres[i]; mid = (s + e) // 2
    if c != cur:
        cur = c; chromseq = str(gen[c]).upper()
    a, b = mid - W, mid + W
    seq = chromseq[max(a, 0):b]
    if len(seq) < 2 * W:
        seq = ("N" * (2 * W - len(seq))) + seq if a < 0 else seq + "N" * (2 * W - len(seq))
    seqs[i] = seq
    if (cnt + 1) % 20000 == 0:
        print(f"  seq {cnt+1}/{len(cres)} extracted", flush=True)

# --- 4. save (compact) ---
with gzip.open(f"{OUT}/pretrain_seqs.txt.gz", "wt") as fh:
    fh.write("\n".join(seqs))
np.savez_compressed(f"{OUT}/pretrain_labels.npz", labels=np.packbits(labels, axis=1), n_tf=len(tfl))
json.dump({"tf_list": tfl, "n": len(cres), "window": 2 * W,
           "coords": [f"{c}:{s}-{e}" for c, s, e in cres]}, open(f"{OUT}/pretrain_meta.json", "w"))
print(f"DONE: {len(cres)} cCREs x {len(tfl)} TFs -> {OUT}/ (seqs+labels+meta)", flush=True)
