"""Real three-dimensional contact for the K562 enhancer-gene task: Hi-C, loops and contact domains.

WHY THIS IS THE RIGHT INPUT AND SEQUENCE WAS NOT. Loops 173 to 180 established that DNA sequence
answers "is this an enhancer" well -- AUC 0.8506 against distance-matched genome -- and answers
"which gene does it act on" barely at all: strip the six gene-varying columns from the stage-two
stack and within-gene R@1 falls from 0.6050 to 0.3427, a lift of 1.23x over a 0.0404 base rate.
That is not a modelling failure. Which promoter an enhancer serves is decided by whether the two
are brought into physical proximity, and that is a property of the folded chromosome, not of the
element's own bases. No sequence feature can read it. This module fetches the measurement that can.

WHAT IS FETCHED, AND WHY EACH PIECE.

  CONTACT FREQUENCY. Rao et al. (Cell 2014, GSE63525) K562 combined Hi-C, KR-normalised, at 5 kb.
  Streamed region by region with straw rather than downloaded: the full matrix is tens of
  gigabytes and only a 4 Mb strip around each promoter is needed. Critically this map is hg19,
  which is the assembly every sequence feature in this arc was computed in, so no second liftover
  enters the chain.

  OBSERVED OVER EXPECTED. Raw contact is dominated by distance -- two loci 10 kb apart touch far
  more than two 1 Mb apart no matter what else is true -- and distance is already the strongest
  feature this task has. So the expected count at each separation is estimated from the data
  itself, pooling every promoter strip on a chromosome and taking the median count per distance
  bin, and the ratio is carried alongside the raw number. If Hi-C adds anything over distance it
  has to be visible in this column; if only the raw column moves, Hi-C is re-encoding distance and
  the loop should say so.

  LOOPS. HiCCUPS calls on the same experiment: pairs of anchors between which the map shows a
  focal enrichment above local background. A called loop connecting an element's anchor to a
  promoter's anchor is the most direct statement the data can make that these two specific loci
  touch each other, as opposed to being in a generally contact-rich neighbourhood.

  CONTACT DOMAINS. Arrowhead calls on the same experiment. Enhancer-promoter regulation is largely
  confined within domains, so whether the pair sits inside one domain, and how many domain
  boundaries lie between them, is the classic structural constraint on which gene an element can
  reach.

THE ONE THING THAT MAKES THESE FEATURES HONEST. Contact and distance are almost the same variable.
Every gate in the loop that uses this module has to be an increment OVER distance, and the
observed-over-expected column exists so the increment can be attributed rather than assumed.

Nothing here is scored. This module fetches, caches and assembles columns; whether any of it
predicts anything is the loop's question.
"""
import gzip
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
HIC = SP / "hic"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl"
HIC_URL = f"{GEO}/GSE63525_K562_combined.hic"
LOOPS = "GSE63525_K562_HiCCUPS_looplist.txt.gz"
DOMAINS = "GSE63525_K562_Arrowhead_domainlist.txt.gz"
RES = 5000
HALF_WINDOW = 2_000_000          # the plan's +/- 2 Mb, and the widest distance the benchmark holds
NORM = "KR"
STRIPS = HIC / f"k562_strips_{RES}.npz"


def fetch(name, report=print):
    p = HIC / name
    if not p.exists():
        HIC.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{GEO}/{name}", timeout=600) as r:
            p.write_bytes(r.read())
        report(f"    fetched {name} ({p.stat().st_size:,} bytes)")
    return p


def load_bedpe(name, report=print):
    """HiCCUPS / Arrowhead output. Chromosome names carry no `chr` prefix in these files."""
    p = fetch(name, report)
    rows = []
    with gzip.open(p, "rt") as f:
        hdr = f.readline()
        for line in f:
            g = line.rstrip("\n").split("\t")
            if len(g) < 6:
                continue
            try:
                rows.append(("chr" + g[0].lstrip("chr"), int(g[1]), int(g[2]),
                             "chr" + g[3].lstrip("chr"), int(g[4]), int(g[5])))
            except ValueError:
                continue
    report(f"    {name}: {len(rows):,} records")
    return rows


def strips(tss, report=print, force=False):
    """tss: list of (chrom, position). Returns a list of (bins, counts) arrays, one per entry --
    the KR-normalised contact profile of that promoter's own bin against everything within
    HALF_WINDOW. Streamed from the remote .hic; cached because it is 2,205 network round trips."""
    if STRIPS.exists() and not force:
        z = np.load(STRIPS, allow_pickle=True)
        if int(z["n"]) == len(tss):
            report(f"    contact strips from cache: {STRIPS.name} "
                   f"({STRIPS.stat().st_size/1e6:.1f} MB)")
            return [(z[f"b{i}"], z[f"c{i}"]) for i in range(len(tss))]
        report(f"    cache holds {int(z['n'])} strips but {len(tss)} were asked for -- refetching")
    import hicstraw
    t0 = time.time()
    h = hicstraw.HiCFile(HIC_URL)
    report(f"    {HIC_URL.rsplit('/', 1)[-1]}: genome {h.getGenomeID()}, "
           f"resolutions {sorted(h.getResolutions())[:4]}...")
    by_chrom = defaultdict(list)
    for i, (c, p) in enumerate(tss):
        by_chrom[c].append(i)
    out = [(np.zeros(0, np.int64), np.zeros(0, np.float32))] * len(tss)
    done = 0
    for c in sorted(by_chrom):
        name = c.replace("chr", "")
        try:
            mz = h.getMatrixZoomData(name, name, "observed", NORM, "BP", RES)
        except Exception as e:
            report(f"    {c}: no matrix ({type(e).__name__}) -- {len(by_chrom[c])} strips empty")
            continue
        for i in by_chrom[c]:
            p = tss[i][1]
            b = (p // RES) * RES
            lo, hi = max(0, p - HALF_WINDOW), p + HALF_WINDOW
            try:
                recs = mz.getRecords(b, b + RES, lo, hi)
            except Exception:
                recs = []
            if not recs:
                try:
                    recs = mz.getRecords(lo, hi, b, b + RES)
                except Exception:
                    recs = []
            bs, cs = [], []
            for r in recs:
                other = r.binY if abs(r.binX - b) < RES else r.binX
                v = r.counts
                if v == v:                      # KR leaves NaN in unmappable bins
                    bs.append(other)
                    cs.append(v)
            out[i] = (np.asarray(bs, np.int64), np.asarray(cs, np.float32))
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                report(f"      strip {done}/{len(tss)}  [{el:.0f}s, "
                       f"eta {el/done*(len(tss)-done):.0f}s]")
    HIC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STRIPS, n=len(tss),
                        **{f"b{i}": out[i][0] for i in range(len(tss))},
                        **{f"c{i}": out[i][1] for i in range(len(tss))})
    nz = sum(1 for b, c in out if len(b))
    report(f"    {nz}/{len(tss)} promoters have a contact profile  "
           f"-> {STRIPS.name} [{time.time()-t0:.0f}s]")
    return out


def expected_by_distance(tss, prof, report=print):
    """Median KR count per distance bin, pooled over every promoter on the same chromosome. This is
    the denominator that turns a contact count into a statement about contact BEYOND distance."""
    acc = defaultdict(list)
    for (c, p), (b, v) in zip(tss, prof):
        if not len(b):
            continue
        d = np.abs(b - (p // RES) * RES) // RES
        for k, x in zip(d, v):
            acc[(c, int(k))].append(float(x))
    exp = {k: float(np.median(v)) for k, v in acc.items() if v}
    report(f"    expected-by-distance curve over {len(exp):,} (chromosome, separation) cells")
    return exp


def interval_index(rows, which):
    """chrom -> sorted (start, end) array, for the anchor side `which` (0 or 1)."""
    d = defaultdict(list)
    for r in rows:
        if which == 0:
            d[r[0]].append((r[1], r[2]))
        else:
            d[r[3]].append((r[4], r[5]))
    return {c: np.array(sorted(v), dtype=np.int64) for c, v in d.items()}


def overlaps(idx, chrom, a, b):
    """Indices of intervals on `chrom` overlapping [a, b)."""
    arr = idx.get(chrom)
    if arr is None or not len(arr):
        return np.zeros(0, np.int64)
    j = int(np.searchsorted(arr[:, 0], b))
    lo = max(0, j - 64)
    sl = arr[lo:j]
    if not len(sl):
        return np.zeros(0, np.int64)
    m = sl[:, 1] > a
    return np.arange(lo, j)[m]


def tss_hg19(report=print):
    """The benchmark promoters in hg19, in the same order as the scan cache's gn_key, so every
    contact profile lines up with the gene index every other loop already uses."""
    from enh import genome as GEN
    from enh import scan as SC
    S = SC.load(lambda *_: None)
    lo = GEN.LiftOver()
    out, miss = [], 0
    for k in S["gn_key"]:
        c, p, _ = str(k).split(":")
        q = lo.lift(c, int(p))
        if q is None:
            miss += 1
            out.append((c, 0))
        else:
            out.append((c, q))
    report(f"    {len(out):,} promoters, {miss} failed the hg38->hg19 lift")
    return out


if __name__ == "__main__":
    print("=" * 100)
    print("K562 3D CONTACT: Rao 2014 Hi-C strips, HiCCUPS loops, Arrowhead contact domains")
    print("=" * 100)
    t = tss_hg19()
    pr = strips(t)
    exp = expected_by_distance(t, pr)
    lp = load_bedpe(LOOPS)
    dm = load_bedpe(DOMAINS)
    nz = sum(1 for b, c in pr if len(b))
    tot = sum(len(b) for b, c in pr)
    print(f"  coverage: {nz}/{len(pr)} promoters with a profile, {tot:,} contact cells total, "
          f"{tot/max(nz,1):.0f} per promoter")
    print(f"  loops {len(lp):,}; contact domains {len(dm):,}")
    json.dump({"promoters": len(pr), "with_profile": nz, "cells": tot,
               "loops": len(lp), "domains": len(dm), "resolution": RES,
               "half_window": HALF_WINDOW, "norm": NORM,
               "source": "Rao et al. Cell 2014, GSE63525, K562 combined, hg19"},
              open(HIC / "contact_manifest.json", "w"), indent=1)
    print(f"  -> {HIC / 'contact_manifest.json'}")
