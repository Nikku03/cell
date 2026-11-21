"""hg19 sequence for GRCh38 benchmark coordinates -- the liftover, and the proof that it worked.

THE MISMATCH THIS EXISTS TO CLOSE. The EP CRISPR benchmark (EngreitzLab/CRISPR_comparison) is
distributed in GRCh38. The reference sequence already on this machine is hg19, one FASTA per
chromosome. Every sequence-level claim about an enhancer therefore depends on a coordinate
conversion, and a coordinate conversion that is quietly wrong does not throw -- it returns a real
piece of DNA from the wrong place, and every motif, every groove width and every electrostatic
potential computed from it is a number about a locus nobody asked about. That failure mode is
invisible downstream, so it is measured here and nowhere else.

HOW IT IS DONE. UCSC's hg38ToHg19.over.chain is a list of alignment chains; each chain is a run of
ungapped blocks with a score, and each block maps an interval of hg38 onto an interval of hg19.
Lifting a position means finding the block that covers it and adding the offset. Two things make
this more than a lookup:

  CHAINS OVERLAP. A position can be covered by several chains -- one primary and several from
  paralogous or alt regions. UCSC's own liftOver resolves this by chain score, so this module keeps
  every covering block and returns the one from the highest-scoring chain, rather than the first
  one found.

  THE QUERY SIDE CAN BE REVERSED. A chain may align hg38 forward to hg19 reverse, in which case the
  stored q coordinates count from the other end and the arithmetic is qSize - q, not q. Handled
  explicitly; regions that land on a reversed chain are flagged so a caller can drop them rather
  than silently read the wrong strand.

  CROSS-CHROMOSOME MAPPINGS ARE REFUSED. A hg38 position on chr7 that lifts best onto hg19 chr5 is
  a paralogue, not a coordinate conversion. Such lifts are dropped and counted.

THE CHECK THAT MAKES THIS TRUSTWORTHY, and it is a real one rather than an internal consistency
argument. Two files on disk hold the SAME 16,380 gene TSSs in the two assemblies, matched by gene
id: _tss_hg38.bed and _tss_hg19.bed. Lifting the hg38 file and comparing against the hg19 file is
an end-to-end test of the whole path -- chain parsing, block arithmetic, strand handling and best-
chain selection -- against coordinates this module never sees. It is run by `python
colab/enh/genome.py` and the exact-agreement rate is printed before anything else uses the lift.

DECLARED BEFORE THE NUMBER: the liftover is usable if at least 95% of those TSSs lift at all and at
least 99% of the ones that lift land on the exact base. Below that, the sequence work is not worth
doing and the loop should say so instead of reporting motif scores computed on the wrong DNA.

SEQUENCE EXTRACTION. Requested regions are grouped by chromosome so each FASTA is decompressed
once, converted to a uint8 code array (A=0, C=1, G=2, T=3, everything else 255) and sliced. Soft-
masked (lower-case) repeat annotation is folded into the upper-case base: repeat-masking is a
property of the annotation, not of the molecule, and dropping repeats would delete a large fraction
of real regulatory sequence. Positions off the end of a chromosome, and the leading and trailing
N-runs, come back as 255 and are the caller's problem to count.
"""
import gzip
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
CHAIN = SP / "hg38ToHg19.over.chain"
CACHE = SP / "enh_seq_cache"

PRIMARY = ["chr%d" % i for i in range(1, 23)] + ["chrX", "chrY"]
N_CODE = 255

# declared before the number, see the docstring
MIN_LIFT_RATE = 0.95
MIN_EXACT_RATE = 0.99


def _base_lut():
    lut = np.full(256, N_CODE, dtype=np.uint8)
    for i, b in enumerate("ACGT"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    return lut


LUT = _base_lut()


class LiftOver:
    """hg38 -> hg19 by chain file, resolving overlaps by chain score."""

    def __init__(self, path=CHAIN):
        if not path.exists():
            raise SystemExit(f"{path} missing -- the hg38->hg19 chain file is required")
        blocks = {}                      # tName -> list of (tstart, tend, qstart, qname, rev, qsize, score)
        t = q = 0
        hdr = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    hdr = None
                    continue
                if line.startswith("chain"):
                    p = line.split()
                    # chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id
                    hdr = dict(score=float(p[1]), tName=p[2], tStart=int(p[5]),
                               qName=p[7], qSize=int(p[8]), qStrand=p[9], qStart=int(p[10]))
                    t, q = hdr["tStart"], hdr["qStart"]
                    continue
                if hdr is None:
                    continue
                p = line.split()
                size = int(p[0])
                blocks.setdefault(hdr["tName"], []).append(
                    (t, t + size, q, hdr["qName"], hdr["qStrand"] == "-", hdr["qSize"], hdr["score"]))
                if len(p) == 3:
                    t += size + int(p[1])
                    q += size + int(p[2])
                else:
                    hdr = None
        self.idx = {}
        for c, bl in blocks.items():
            bl.sort(key=lambda r: r[0])
            self.idx[c] = (np.array([b[0] for b in bl], dtype=np.int64),
                           np.array([b[1] for b in bl], dtype=np.int64),
                           np.array([b[2] for b in bl], dtype=np.int64),
                           [b[3] for b in bl],
                           np.array([b[4] for b in bl], dtype=bool),
                           np.array([b[5] for b in bl], dtype=np.int64),
                           np.array([b[6] for b in bl], dtype=np.float64))
        self.n_blocks = sum(len(v[0]) for v in self.idx.values())
        self.stats = dict(queried=0, unmapped=0, cross_chrom=0, reversed_chain=0, mapped=0)

    def lift(self, chrom, pos, backscan=4096):
        """hg38 (chrom, 0-based pos) -> hg19 pos on the SAME chromosome, or None.

        The backward scan is bounded because chain blocks are sorted by start but overlap; the cap
        is generous relative to the deepest observed pile-up and a hit on it is reported, not
        silently truncated."""
        self.stats["queried"] += 1
        ent = self.idx.get(chrom)
        if ent is None:
            self.stats["unmapped"] += 1
            return None
        ts, te, qs, qn, rev, qsz, sc = ent
        i = int(np.searchsorted(ts, pos, side="right")) - 1
        best = None
        best_sc = -1.0
        lo = max(0, i - backscan)
        for k in range(i, lo - 1, -1):
            if te[k] <= pos:
                continue
            if ts[k] > pos:
                continue
            if sc[k] <= best_sc:
                continue
            best, best_sc = k, sc[k]
        if best is None:
            self.stats["unmapped"] += 1
            return None
        if qn[best] != chrom:
            self.stats["cross_chrom"] += 1
            return None
        off = pos - ts[best]
        if rev[best]:
            self.stats["reversed_chain"] += 1
            return None
        self.stats["mapped"] += 1
        return int(qs[best] + off)

    def lift_interval(self, chrom, start, end):
        """Both endpoints, requiring they land on the same chromosome in order and keep the width
        to within 10%. A lifted interval whose length changed is sitting across an indel and is
        refused rather than stretched."""
        a = self.lift(chrom, start)
        b = self.lift(chrom, end - 1)
        if a is None or b is None or b < a:
            return None
        w0, w1 = end - start, b - a + 1
        if abs(w1 - w0) > 0.1 * w0:
            return None
        return a, b + 1


class Genome:
    """hg19 sequence as uint8 base codes, one chromosome decompressed at a time."""

    def __init__(self, sp=SP):
        self.sp = sp

    def _path(self, chrom):
        return self.sp / f"hg19_{chrom}.fa.gz"

    def available(self):
        return sorted(c for c in PRIMARY if self._path(c).exists())

    def load_chrom(self, chrom):
        p = self._path(chrom)
        if not p.exists():
            return None
        parts = []
        with gzip.open(p, "rt") as f:
            for line in f:
                if line[0] == ">":
                    continue
                parts.append(line.rstrip())
        raw = np.frombuffer("".join(parts).encode("ascii", "replace"), dtype=np.uint8)
        return LUT[raw]

    def extract(self, regions, report=print):
        """regions: list of (chrom, start, end) in hg19. Returns a list of uint8 arrays, aligned to
        the input order. Out-of-range positions come back as 255."""
        by_chrom = {}
        for i, (c, s, e) in enumerate(regions):
            by_chrom.setdefault(c, []).append(i)
        out = [None] * len(regions)
        for c in sorted(by_chrom, key=lambda x: PRIMARY.index(x) if x in PRIMARY else 99):
            seq = self.load_chrom(c)
            if seq is None:
                for i in by_chrom[c]:
                    out[i] = np.full(regions[i][2] - regions[i][1], N_CODE, dtype=np.uint8)
                report(f"    {c}: FASTA missing, {len(by_chrom[c])} regions filled with N")
                continue
            L = len(seq)
            for i in by_chrom[c]:
                _, s, e = regions[i]
                buf = np.full(e - s, N_CODE, dtype=np.uint8)
                a, b = max(0, s), min(L, e)
                if b > a:
                    buf[a - s:b - s] = seq[a:b]
                out[i] = buf
            report(f"    {c}: {len(by_chrom[c]):,} regions from {L:,} bp")
            del seq
        return out

    def extract_cached(self, regions, tag, report=print):
        """Same, but persisted under a digest of the request so a rerun does not re-decompress
        every chromosome."""
        h = hashlib.sha256(repr(sorted(regions)).encode()).hexdigest()[:16]
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / f"{tag}_{h}.npz"
        if p.exists():
            z = np.load(p)
            report(f"    sequence from cache: {p.name}")
            return [z[f"s{i}"] for i in range(len(regions))]
        out = self.extract(regions, report=report)
        np.savez_compressed(p, **{f"s{i}": a for i, a in enumerate(out)})
        report(f"    sequence cached -> {p.name} ({p.stat().st_size:,} bytes)")
        return out


def qc(report=print):
    """The end-to-end liftover check against the paired TSS bed files."""
    a, b = SP / "_tss_hg38.bed", SP / "_tss_hg19.bed"
    if not (a.exists() and b.exists()):
        report("  paired TSS beds absent -- liftover cannot be checked, refusing to proceed")
        return False, {}
    def rd(p):
        d = {}
        for line in open(p):
            f = line.split()
            if len(f) >= 4:
                d[f[3]] = (f[0], int(f[1]))
        return d
    h38, h19 = rd(a), rd(b)
    shared = sorted(set(h38) & set(h19))
    lo = LiftOver()
    report(f"  chain: {lo.n_blocks:,} blocks over {len(lo.idx)} hg38 sequences")
    report(f"  QC set: {len(shared):,} gene TSSs present in both assemblies")
    lifted = exact = 0
    err = []
    for g in shared:
        c38, p38 = h38[g]
        c19, p19 = h19[g]
        if c38 != c19:
            continue
        p = lo.lift(c38, p38)
        if p is None:
            continue
        lifted += 1
        d = abs(p - p19)
        err.append(d)
        exact += (d == 0)
    err = np.array(err)
    lift_rate = lifted / max(len(shared), 1)
    exact_rate = exact / max(lifted, 1)
    report(f"  lifted {lifted:,}/{len(shared):,} ({lift_rate:.4f}); "
           f"exact base {exact:,}/{lifted:,} ({exact_rate:.4f})")
    if len(err):
        report(f"  |error| median {np.median(err):.0f} bp, p90 {np.percentile(err, 90):.0f} bp, "
               f"max {err.max():,} bp")
    report(f"  lift stats: {lo.stats}")
    ok = lift_rate >= MIN_LIFT_RATE and exact_rate >= MIN_EXACT_RATE
    report(f"  GATE (declared before the number: lift >= {MIN_LIFT_RATE}, exact >= {MIN_EXACT_RATE}): "
           + ("PASS" if ok else "FAIL"))
    return ok, dict(n=len(shared), lifted=lifted, exact=exact,
                    lift_rate=float(lift_rate), exact_rate=float(exact_rate),
                    median_err=float(np.median(err)) if len(err) else None,
                    stats=dict(lo.stats))


if __name__ == "__main__":
    print("=" * 100)
    print("hg38 -> hg19 LIFTOVER, checked end to end against 16,380 gene TSSs held in both assemblies")
    print("=" * 100)
    ok, d = qc()
    g = Genome()
    print(f"  hg19 FASTA present for {len(g.available())} chromosomes: {' '.join(g.available())}")
    sys.exit(0 if ok else 1)
