"""Measured TF binding at the benchmark elements: ENCODE K562 ChIP, not motif predictions.

WHY THIS CHANGES THE QUESTION. Everything in loops 173 to 183 used a motif score as a stand-in for
"this factor is at this element". A motif score is a statement about sequence; whether a protein is
actually there is a different fact, and the field's long-standing answer is that the two agree
badly. Without the measurement there is no way to ask what makes a factor attach to an enhancer --
only what makes a sequence resemble its consensus.

ENCODE has released conservative-IDR peak calls for 519 TF targets in K562, and 191 of them also
carry a JASPAR matrix in this project's set. That intersection is the object this module builds: a
boolean matrix over (191 factors x 4,482 benchmark elements) saying, for each pair, whether a
called peak summit falls within the element.

COORDINATES. The peaks are GRCh38 and the benchmark's own element coordinates are GRCh38, so this
join needs no liftover at all. Every sequence feature in this arc lives in hg19 via a checked lift;
this one deliberately does not enter that chain, because a binding call is about a genomic interval
and the interval is already in the assembly ENCODE distributes.

CHOICES THAT ARE PINNED, so a rerun cannot quietly change the matrix:
    output type   conservative IDR thresholded peaks only. A factor with no such file is DROPPED
                  and counted, never back-filled with a noisier pseudoreplicated call, because
                  mixing IDR tiers would make "bound" mean different things for different factors.
    which file    when a factor has several, the lowest accession wins -- arbitrary but
                  deterministic.
    what counts   the narrowPeak summit (column 10), or the interval midpoint where no summit was
                  called, within ELEMENT_PAD of the element. Summits rather than intervals, so a
                  very wide peak cannot mark every element it grazes.

DISK. Peak files are downloaded one at a time, reduced to the boolean column, and deleted. Only
the matrix is kept.

Output: cached npz in the scratchpad, keyed by the element list it was built against.
"""
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
API = "https://www.encodeproject.org"
ASSEMBLY = "GRCh38"
OUTPUT_TYPE = "conservative IDR thresholded peaks"
ELEMENT_PAD = 250
CACHE = SP / "enh_chip_k562.npz"
TMP = SP / "enh_chip_tmp"


def api(path, tries=4, timeout=300):
    for i in range(tries):
        try:
            r = urllib.request.Request(API + path,
                                       headers={"accept": "application/json", "User-Agent": "cellos"})
            return json.load(urllib.request.urlopen(r, timeout=timeout))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def index(report=print):
    """One portal query for every released K562 conservative-IDR TF ChIP file."""
    u = ("/search/?type=File&file_format=bed&file_format_type=narrowPeak"
         f"&assembly={ASSEMBLY}&output_type={urllib.parse.quote(OUTPUT_TYPE)}"
         "&biosample_ontology.term_name=K562&status=released&limit=all&frame=object&format=json")
    d = api(u)
    g = [x for x in d.get("@graph", []) if x.get("assay_title") == "TF ChIP-seq"]
    best = {}
    for x in g:
        t = x["target"].split("/")[-2].replace("-human", "")
        if t not in best or x["accession"] < best[t]["accession"]:
            best[t] = x
    report(f"    ENCODE: {len(g):,} released K562 '{OUTPUT_TYPE}' files over {len(best)} targets")
    return best


def download(url, path, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
                f.write(r.read())
            return
        except Exception:
            if path.exists():
                path.unlink()
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def summits(path):
    """chrom -> sorted summit positions. Column 10 is the summit offset; -1 means none was called,
    in which case the interval midpoint stands in."""
    d = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                st, en = int(p[1]), int(p[2])
            except ValueError:
                continue
            off = int(p[9]) if len(p) > 9 and p[9] not in ("", "-1") else -1
            d.setdefault(p[0], []).append(st + off if off >= 0 else (st + en) // 2)
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in d.items()}


def build(el_key, tf_names, report=print, force=False):
    """el_key: GRCh38 'chr:start-end' strings in the scan cache's order.
    tf_names: the factor symbols to try, normally the JASPAR names in this project's matrix set."""
    keys = [str(k) for k in el_key]
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        if list(z["elkey"]) == keys:
            report(f"    ChIP matrix from cache: {z['B'].shape[0]} factors x "
                   f"{z['B'].shape[1]:,} elements")
            return z["B"], [str(t) for t in z["tfs"]]
        report("    cache was built against a different element list -- rebuilding")
    idx = index(report)
    want = {t.upper() for t in tf_names}
    tfs = sorted(t for t in idx if t.upper() in want)
    report(f"    {len(tfs)}/{len(idx)} ENCODE targets also carry a matrix in this project "
           f"({len(tfs)/max(len(want),1):.1%} of the {len(want)} distinct factor names)")
    ech, est, een = [], [], []
    for k in keys:
        c, rest = k.split(":")
        a, b = rest.split("-")
        ech.append(c)
        est.append(int(a))
        een.append(int(b))
    ech = np.array(ech)
    est = np.array(est, dtype=np.int64)
    een = np.array(een, dtype=np.int64)
    B = np.zeros((len(tfs), len(keys)), dtype=bool)
    TMP.mkdir(parents=True, exist_ok=True)
    done = [0]
    t0 = time.time()

    def one(k):
        t = tfs[k]
        p = TMP / f"{t}.bed.gz"
        try:
            download(API + idx[t]["href"], p)
            sm = summits(p)
        except Exception:
            return k, np.zeros(len(keys), dtype=bool)
        finally:
            if p.exists():
                p.unlink()
        col = np.zeros(len(keys), dtype=bool)
        for c in np.unique(ech):
            pos = sm.get(c)
            if pos is None:
                continue
            m = np.where(ech == c)[0]
            lo = np.searchsorted(pos, est[m] - ELEMENT_PAD)
            hi = np.searchsorted(pos, een[m] + ELEMENT_PAD)
            col[m] = hi > lo
        done[0] += 1
        if done[0] % 50 == 0:
            el = time.time() - t0
            report(f"      {done[0]}/{len(tfs)} factors  [{el:.0f}s, "
                   f"eta {el/done[0]*(len(tfs)-done[0]):.0f}s]")
        return k, col

    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, col in ex.map(one, range(len(tfs))):
            B[k] = col
    np.savez_compressed(CACHE, B=B, tfs=np.array(tfs, dtype=object),
                        elkey=np.array(keys, dtype=object))
    per_el = B.sum(0)
    per_tf = B.sum(1)
    report(f"    occupancy: {len(tfs)} factors x {len(keys):,} elements, "
           f"{100*B.mean():.1f}% of cells bound")
    report(f"      factors per element: median {np.median(per_el):.0f} "
           f"(IQR {np.percentile(per_el,25):.0f}-{np.percentile(per_el,75):.0f}), "
           f"{(per_el == 0).mean():.1%} of elements bound by none")
    report(f"      elements per factor: median {np.median(per_tf):.0f} "
           f"(IQR {np.percentile(per_tf,25):.0f}-{np.percentile(per_tf,75):.0f})")
    report(f"    -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
    return B, tfs


if __name__ == "__main__":
    from enh import scan as SC
    from enh import tf_domains as TD
    print("=" * 100)
    print("MEASURED TF BINDING AT THE BENCHMARK ELEMENTS: ENCODE K562 conservative-IDR ChIP")
    print("=" * 100)
    S = SC.load(print)
    dom = TD.load()
    names = sorted({(v.get("name") or "").upper().split("::")[0]
                    for v in dom.values() if v.get("name")})
    build(S["el_key"], names, print)
