"""The chromatin marks the benchmark does not carry, fetched and reduced to the loci that need them.

WHAT IS ALREADY IN HAND AND WAS NOT BEING USED. The EP CRISPR benchmark ships per-element signal for
five marks and this project had used two of them. H3K27ac and DHS went into loop 185 and were worth
+0.0693 within-gene R@1 in 5 of 5 seeds. H3K27me3, H3K4me1, CTCF and the categorical
`elementChromatinCategory` were sitting in the same file untouched -- H3K27me3 in particular is the
repressive mark, so it is the one column that should push in the OPPOSITE direction from every
activity feature in the stack, and a model that has never seen it has no way to say "open, marked,
and silenced".

WHAT HAS TO BE FETCHED.
    H3K4me3   the benchmark carries H3K4me1 but not H3K4me3, so the me1-over-me3 ratio -- the
              standard discriminator between a distal enhancer and a promoter -- cannot be formed
              from the file alone. ENCODE's K562 replicated-peak call is 1 MB.
    5mC       CpG methylation from ENCODE K562 WGBS. Methylation at a CpG inside a motif suppresses
              binding for many factors, and it is the one epigenetic axis in this arc that is
              measured per BASE rather than per region, so it can be attributed to the motif's own
              cytosines rather than to the element as a whole. The released bed is 589 MB and is
              streamed, filtered to the 4,482 elements and 2,205 promoters as it arrives, and
              deleted -- keeping it would cost a fifth of the free space on this machine for a
              table that fits in a few megabytes.

COORDINATES. Both are GRCh38, and so are the benchmark's element and promoter coordinates, so this
module stays out of the hg19 lift chain the sequence features use.

Nothing here is scored. This fetches and reduces; whether any of it predicts anything is the loop's
question and is gated there.
"""
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
EPI = SP / "epigenome"
API = "https://www.encodeproject.org"
CACHE = EPI / "k562_marks.npz"
PROMOTER_PAD = 1000


def api(path, tries=4, timeout=240):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def pick(output_type, fmt, target=None, report=print):
    """Smallest released K562 GRCh38 file of the given kind. Smallest, not newest: it is a
    deterministic rule, and for a peak call the size differences are between replicates rather
    than between qualities."""
    p = ("/search/?type=File&assembly=GRCh38&biosample_ontology.term_name=K562&status=released"
         f"&file_format={fmt}&output_type={urllib.parse.quote(output_type)}"
         "&limit=all&frame=object&format=json")
    if target:
        p += f"&target.label={target}"
    g = api(p).get("@graph", [])
    if not g:
        return None
    g.sort(key=lambda z: int(z.get("file_size", 0)))
    x = g[0]
    report(f"    {target or output_type}: {x['accession']}, "
           f"{int(x.get('file_size', 0))/1e6:.0f} MB, {len(g)} candidates")
    return x


def intervals(keys):
    """chrom -> (starts, ends, row index), sorted, for streaming overlap."""
    d = defaultdict(list)
    for i, k in enumerate(keys):
        c, rest = k.split(":")
        a, b = rest.split("-")
        d[c].append((int(a), int(b), i))
    return {c: (np.array([x[0] for x in sorted(v)], np.int64),
                np.array([x[1] for x in sorted(v)], np.int64),
                np.array([x[2] for x in sorted(v)], np.int64)) for c, v in d.items()}


def peak_overlap(url, keys, report=print):
    """Fraction of each interval covered by a peak, and the max signal over it."""
    EPI.mkdir(parents=True, exist_ok=True)
    p = EPI / url.rsplit("/", 1)[-1]
    if not p.exists():
        with urllib.request.urlopen(url, timeout=900) as r:
            p.write_bytes(r.read())
    iv = intervals(keys)
    cov = np.zeros(len(keys))
    sig = np.zeros(len(keys))
    op = gzip.open(p, "rt") if p.read_bytes()[:2] == b"\x1f\x8b" else open(p, "rt")
    n = 0
    for line in op:
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        c = f[0]
        e = iv.get(c)
        if e is None:
            continue
        try:
            s, t = int(f[1]), int(f[2])
            v = float(f[6]) if len(f) > 6 else 1.0
        except ValueError:
            continue
        st, en, idx = e
        j = int(np.searchsorted(st, t))
        for k in range(max(0, j - 64), j):
            if en[k] > s:
                i = int(idx[k])
                cov[i] += max(0, min(en[k], t) - max(st[k], s))
                sig[i] = max(sig[i], v)
        n += 1
    width = np.array([int(k.split("-")[1]) - int(k.split(":")[1].split("-")[0]) for k in keys],
                     dtype=float)
    report(f"      {n:,} peaks read; {(cov > 0).mean():.1%} of intervals overlap one")
    return np.clip(cov / np.maximum(width, 1), 0, 1), sig


def methylation(url, keys, report=print):
    """Mean methylation percentage and CpG count per interval, streamed from a 589 MB bed without
    ever holding it: the file arrives line by line, every CpG outside the intervals is dropped
    immediately, and nothing is written to disk."""
    iv = intervals(keys)
    tot = np.zeros(len(keys))
    cnt = np.zeros(len(keys))
    t0 = time.time()
    n = 0
    req = urllib.request.Request(url, headers={"User-Agent": "cellos"})
    with urllib.request.urlopen(req, timeout=1800) as raw:
        stream = gzip.GzipFile(fileobj=raw) if url.endswith(".gz") else raw
        for bline in stream:
            line = bline.decode("ascii", "replace")
            f = line.rstrip("\n").split("\t")
            if len(f) < 11:
                continue
            c = f[0]
            e = iv.get(c)
            if e is None:
                continue
            try:
                s = int(f[1])
                pct = float(f[10])
            except ValueError:
                continue
            st, en, idx = e
            j = int(np.searchsorted(st, s + 1))
            for k in range(max(0, j - 64), j):
                if en[k] > s:
                    i = int(idx[k])
                    tot[i] += pct
                    cnt[i] += 1
            n += 1
            if n % 5_000_000 == 0:
                report(f"      {n:,} CpGs streamed, {int(cnt.sum()):,} kept "
                       f"[{time.time()-t0:.0f}s]")
    with np.errstate(invalid="ignore"):
        mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    report(f"      {n:,} CpGs streamed; {int(cnt.sum()):,} inside the intervals; "
           f"{(cnt > 0).mean():.1%} of intervals have at least one")
    return mean, cnt


def build(el_key, gn_key, report=print, force=False):
    keys_e = [str(k) for k in el_key]
    keys_p = []
    for k in gn_key:
        c, p, _ = str(k).split(":")
        keys_p.append(f"{c}:{max(0, int(p)-PROMOTER_PAD)}-{int(p)+PROMOTER_PAD}")
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        if list(z["elkey"]) == keys_e and list(z["prkey"]) == keys_p:
            report(f"    epigenome from cache: {CACHE.name}")
            return {k: z[k] for k in z.files}
    out = {}
    report("    H3K4me3 replicated peaks")
    x = pick("replicated peaks", "bed", "H3K4me3", report)
    if x:
        out["el_h3k4me3_cov"], out["el_h3k4me3_sig"] = peak_overlap(API + x["href"], keys_e, report)
        out["pr_h3k4me3_cov"], out["pr_h3k4me3_sig"] = peak_overlap(API + x["href"], keys_p, report)
    report("    CpG methylation (streamed, nothing written to disk)")
    x = pick("methylation state at CpG", "bed", None, report)
    if x:
        out["el_5mc"], out["el_ncpg"] = methylation(API + x["href"], keys_e, report)
        out["pr_5mc"], out["pr_ncpg"] = methylation(API + x["href"], keys_p, report)
    out["elkey"] = np.array(keys_e, dtype=object)
    out["prkey"] = np.array(keys_p, dtype=object)
    EPI.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    report(f"    -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    from enh import scan as SC
    print("=" * 100)
    print("K562 CHROMATIN MARKS THE BENCHMARK DOES NOT CARRY: H3K4me3 and CpG methylation")
    print("=" * 100)
    S = SC.load(print)
    build(S["el_key"], S["gn_key"], print)
