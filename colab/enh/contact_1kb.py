"""Contact at 1 kb, because loop 181 measured exactly what 5 kb costs.

WHY THIS EXISTS AND WHAT IT IS EXPECTED TO FIX. Loop 181 put a real K562 Hi-C map on the stage-two
task and it failed almost every gate: the observed-over-expected column added nothing once the
distance decay was divided out, and twenty draws giving each gene a stranger's contact profile
scored HIGHER than the real one. The diagnosis was not that contact is irrelevant, it was
resolution, and it was measured rather than asserted:

    bin size   candidates sharing a bin with another   bins holding both a positive
               candidate of the SAME gene              and a negative for that gene
     5,000 bp            33.7%                                    40
     1,000 bp             2.6%                                     2
       500 bp             0.4%                                     0

A third of the decisions this benchmark asks for are, at 5 kb, between candidates the map cannot
tell apart. At 1 kb that falls to one in forty. So the same experiment is worth repeating at 1 kb,
and it is worth repeating with a prediction attached: if resolution was the binding constraint, the
observed-over-expected column should now carry something and the stranger-swap should now lose.
If it still fails, the honest conclusion changes -- it is not the map's resolution, and contact as
this benchmark can measure it does not decide which gene an element serves.

WHAT IS USED. ENCODE released K562 contact matrices on GRCh38 with bins down to 1 kb. GRCh38 is the
benchmark's NATIVE coordinate system, so unlike loop 181's hg19 map this one needs no liftover at
any point -- the elements, the promoters and the contacts are all in one assembly.

THREE THINGS THIS FILE HANDLES THAT THE hg19 VERSION DID NOT.
    CHROMOSOME NAMING. These files name chromosomes `chr1`, where the GEO file used `1`. Getting
    this wrong returns an empty strip rather than an error, which is the worst kind of failure, so
    the name is read from the file's own chromosome table rather than assumed.
    NORMALISATION. The released files carry no KR, SCALE or VC vectors at fine resolutions, so raw
    observed counts are used. That is not a compromise here: every quantity this project derives
    from contact is either a within-window ratio or an observed-over-expected against a decay
    estimated from the same data, and both are invariant to a per-bin scaling that does not exist
    anyway.
    THE STREAM DROPS. Long-lived HTTP range sessions against these files fail with SSL and curl
    errors partway through, exactly as the GEO file did at strip 1,250 of 2,205. Every strip is
    retried on a fresh handle and progress is checkpointed, so a drop costs one strip and not a run.
"""
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
ENCODE = "https://www.encodeproject.org"
# chosen by probe: K562, GRCh38, opens remotely, carries a 1 kb zoom level, and is small enough
# that the block index loads in seconds rather than minutes
ACCESSION = "ENCFF777GXN"
RES = 1000
HALF_WINDOW = 2_000_000
NORM = "NONE"
STRIPS = HIC / f"k562_encode_strips_{RES}.npz"
PARTIAL = HIC / f"k562_encode_strips_{RES}_partial.npz"


def file_url(acc=ACCESSION, report=print):
    r = urllib.request.Request(f"{ENCODE}/files/{acc}/?frame=object&format=json",
                               headers={"accept": "application/json", "User-Agent": "cellos"})
    d = json.load(urllib.request.urlopen(r, timeout=180))
    report(f"    {acc}: {int(d.get('file_size', 0))/1e9:.1f} GB, {d.get('output_type')}, "
           f"assembly {d.get('assembly')}")
    return ENCODE + d["href"]


def _checkpoint(out, got):
    idx = sorted(got)
    HIC.mkdir(parents=True, exist_ok=True)
    tmp = PARTIAL.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, idx=np.array(idx, np.int64),
                        **{f"b{i}": out[i][0] for i in idx},
                        **{f"c{i}": out[i][1] for i in idx})
    tmp.replace(PARTIAL)


def strips(tss, report=print, force=False):
    """tss: list of (chrom, position) in GRCh38, in the scan cache's gene order."""
    if STRIPS.exists() and not force:
        z = np.load(STRIPS, allow_pickle=True)
        if int(z["n"]) == len(tss):
            report(f"    1 kb strips from cache: {STRIPS.name} "
                   f"({STRIPS.stat().st_size/1e6:.1f} MB)")
            return [(z[f"b{i}"], z[f"c{i}"]) for i in range(len(tss))]
    import hicstraw
    u = file_url(report=report)
    t0 = time.time()
    h = hicstraw.HiCFile(u)
    have = {c.name for c in h.getChromosomes()}
    rs = sorted(h.getResolutions())
    report(f"    genome {h.getGenomeID()}, resolutions {rs[:8]}, "
           f"chromosome names look like {sorted(have)[:3]}")
    if RES not in rs:
        raise SystemExit(f"{ACCESSION} has no {RES} bp zoom level: {rs}")
    out = [(np.zeros(0, np.int64), np.zeros(0, np.float32))] * len(tss)
    got = set()
    if PARTIAL.exists():
        z = np.load(PARTIAL, allow_pickle=True)
        for i in z["idx"].tolist():
            out[int(i)] = (z[f"b{i}"], z[f"c{i}"])
            got.add(int(i))
        report(f"    resuming: {len(got):,} strips already on disk")
    by_chrom = defaultdict(list)
    for i, (c, p) in enumerate(tss):
        by_chrom[c].append(i)
    done = len(got)
    for c in sorted(by_chrom):
        name = c if c in have else c.replace("chr", "")
        if name not in have:
            report(f"    {c}: not in the file's chromosome table -- {len(by_chrom[c])} strips empty")
            continue
        try:
            mz = h.getMatrixZoomData(name, name, "observed", NORM, "BP", RES)
        except Exception as e:
            report(f"    {c}: no matrix ({type(e).__name__}) -- {len(by_chrom[c])} strips empty")
            continue
        for i in by_chrom[c]:
            if i in got:
                continue
            p = tss[i][1]
            b = (p // RES) * RES
            lo, hi = max(0, p - HALF_WINDOW), p + HALF_WINDOW
            recs = []
            for attempt in range(4):
                try:
                    recs = mz.getRecords(b, b + RES, lo, hi)
                    if not recs:
                        recs = mz.getRecords(lo, hi, b, b + RES)
                    break
                except Exception as e:
                    if attempt == 3:
                        report(f"      {c}:{p} gave up ({type(e).__name__})")
                        recs = []
                        break
                    time.sleep(2 ** attempt)
                    try:
                        h = hicstraw.HiCFile(u)
                        mz = h.getMatrixZoomData(name, name, "observed", NORM, "BP", RES)
                    except Exception:
                        pass
            bs, cs = [], []
            for r in recs:
                other = r.binY if abs(r.binX - b) < RES else r.binX
                v = r.counts
                if v == v:
                    bs.append(other)
                    cs.append(v)
            out[i] = (np.asarray(bs, np.int64), np.asarray(cs, np.float32))
            got.add(i)
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                report(f"      strip {done}/{len(tss)}  [{el:.0f}s, "
                       f"eta {el/max(done-0,1)*(len(tss)-done):.0f}s]")
            if done % 250 == 0:
                _checkpoint(out, got)
    HIC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STRIPS, n=len(tss),
                        **{f"b{i}": out[i][0] for i in range(len(tss))},
                        **{f"c{i}": out[i][1] for i in range(len(tss))})
    nz = sum(1 for b, c in out if len(b))
    cells = sum(len(b) for b, c in out)
    report(f"    {nz}/{len(tss)} promoters have a 1 kb profile, {cells:,} contact cells "
           f"({cells/max(nz,1):.0f} per promoter of {2*HALF_WINDOW//RES} possible bins) "
           f"[{time.time()-t0:.0f}s]")
    return out


def tss_grch38(report=print):
    """The benchmark promoters in GRCh38 -- the assembly the coordinates arrived in, so no lift."""
    from enh import scan as SC
    S = SC.load(lambda *_: None)
    out = []
    for k in S["gn_key"]:
        c, p, _ = str(k).split(":")
        out.append((c, int(p)))
    report(f"    {len(out):,} promoters, GRCh38 native, no liftover in this chain")
    return out


if __name__ == "__main__":
    print("=" * 100)
    print(f"K562 CONTACT AT {RES} bp: ENCODE {ACCESSION}, GRCh38, raw observed counts")
    print("=" * 100)
    t = tss_grch38()
    pr = strips(t)
    d = np.array([len(b) for b, c in pr])
    print(f"  cells per promoter: median {np.median(d):.0f}, "
          f"IQR {np.percentile(d,25):.0f}-{np.percentile(d,75):.0f}")
    json.dump({"accession": ACCESSION, "resolution": RES, "norm": NORM,
               "half_window": HALF_WINDOW, "promoters": len(pr),
               "with_profile": int((d > 0).sum()), "cells": int(d.sum())},
              open(HIC / "contact_1kb_manifest.json", "w"), indent=1)
    print(f"  -> {HIC / 'contact_1kb_manifest.json'}")
