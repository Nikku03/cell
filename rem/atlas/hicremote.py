"""HICREMOTE -- read parts of a Juicer .hic file (version 7-8) over HTTPS with byte-range requests, so a 5-25 GB contact
map never has to be stored: only the header, the index and the blocks of one chromosome at one resolution are fetched.

Used by k3dbank.py (G16, K562 time course). Written from the published .hic format (Durand et al. 2016 Cell Systems;
the straw reader's layout): header -> master index (chr1_chr2 -> matrix position) -> matrix -> zoom levels -> blocks
(zlib) -> records (bin x, bin y, count). Returns RAW counts (no normalisation) for one intra-chromosomal matrix.

Use:  h = HicRemote(url); h.chroms; h.resolutions; i, j, v = h.intra(chrom, binsize)
"""

from __future__ import annotations
import struct
import time
import urllib.request
import zlib

import numpy as np


class HicRemote:
    def __init__(self, url, chunk=1 << 26):
        self.url, self.chunk = url, chunk
        b = self._get(0, 1 << 16)
        if b[:4] != b"HIC\0":
            raise ValueError("not a .hic file")
        p = [4]
        ver = self._i32(b, p); master = self._i64(b, p); self._s(b, p)
        if ver < 7 or ver > 8:
            raise ValueError(f".hic version {ver} not supported")
        for _ in range(self._i32(b, p)):           # attributes (can be long: statistics, graphs)
            while True:
                try:
                    self._s(b, p); self._s(b, p); break
                except ValueError:
                    b = self._get(0, len(b) * 4)
        n = self._i32(b, p); self.chroms = {}
        for k in range(n):
            name = self._s(b, p); ln = self._i32(b, p); self.chroms[name] = (k, ln)
        self.resolutions = [self._i32(b, p) for _ in range(self._i32(b, p))]
        self.version = ver
        f = self._get(master, 1 << 20); q = [0]
        self._i32(f, q); self.index = {}
        for _ in range(self._i32(f, q)):
            key = self._s(f, q); pos = self._i64(f, q); size = self._i32(f, q); self.index[key] = (pos, size)

    # -- byte helpers
    def _get(self, start, n, tries=6):
        for k in range(tries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + n - 1}", "User-Agent": "cellbrain/1.0"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    if r.status != 206:
                        raise IOError(f"server did not honour the byte range (status {r.status})")
                    return r.read()
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(5 * 2 ** k)

    @staticmethod
    def _i32(b, p):
        v = struct.unpack_from("<i", b, p[0])[0]; p[0] += 4; return v

    @staticmethod
    def _i64(b, p):
        v = struct.unpack_from("<q", b, p[0])[0]; p[0] += 8; return v

    @staticmethod
    def _f32(b, p):
        v = struct.unpack_from("<f", b, p[0])[0]; p[0] += 4; return v

    @staticmethod
    def _s(b, p):
        e = b.index(b"\0", p[0]); v = b[p[0]:e].decode("utf-8", "replace"); p[0] = e + 1; return v

    # -- matrix
    def blocks(self, chrom, binsize):
        k = self.chroms[chrom][0]; pos, size = self.index[f"{k}_{k}"]
        b = self._get(pos, size); p = [0]
        self._i32(b, p); self._i32(b, p)
        for _ in range(self._i32(b, p)):
            unit = self._s(b, p); self._i32(b, p)
            for _ in range(4):
                self._f32(b, p)
            bs = self._i32(b, p); self._i32(b, p); self._i32(b, p); nb = self._i32(b, p)
            bl = []
            for _ in range(nb):
                bn = self._i32(b, p); fp = self._i64(b, p); sz = self._i32(b, p); bl.append((fp, sz))
            if unit == "BP" and bs == binsize:
                return sorted(bl)
        raise KeyError(f"{chrom} has no {binsize} bp level")

    def intra(self, chrom, binsize):
        """raw counts of chrom x chrom at binsize: arrays (bin_i, bin_j, count), i <= j, one entry per pixel."""
        bl = [x for x in self.blocks(chrom, binsize) if x[1] > 0]
        I, J, V = [], [], []
        g = 0
        while g < len(bl):                       # fetch runs of consecutive blocks in one request (up to self.chunk bytes)
            h = g; start = bl[g][0]
            while h + 1 < len(bl) and bl[h + 1][0] + bl[h + 1][1] - start <= self.chunk and bl[h + 1][0] <= bl[h][0] + bl[h][1] + 4096:
                h += 1
            buf = self._get(start, bl[h][0] + bl[h][1] - start)
            for fp, sz in bl[g:h + 1]:
                i, j, v = self._records(zlib.decompress(buf[fp - start:fp - start + sz]))
                I.append(i); J.append(j); V.append(v)
            g = h + 1
        i, j, v = (np.concatenate(x) if x else np.zeros(0) for x in (I, J, V))
        a, c = np.minimum(i, j).astype(np.int32), np.maximum(i, j).astype(np.int32)
        return a, c, v.astype(np.float64)

    def _records(self, d):
        p = [0]; n = self._i32(d, p)
        xo = self._i32(d, p); yo = self._i32(d, p)
        short = d[p[0]] == 0; p[0] += 1
        typ = d[p[0]]; p[0] += 1
        if typ == 1:
            X, Y, V = [], [], []
            rows = struct.unpack_from("<h", d, p[0])[0]; p[0] += 2
            ct = np.dtype([("x", "<i2"), ("v", "<i2" if short else "<f4")])
            for _ in range(rows):
                y, cols = struct.unpack_from("<hh", d, p[0]); p[0] += 4
                a = np.frombuffer(d, ct, cols, p[0]); p[0] += cols * ct.itemsize
                X.append(a["x"].astype(np.int64) + xo); Y.append(np.full(cols, yo + y, np.int64)); V.append(a["v"].astype(np.float64))
            x, y, v = (np.concatenate(z) if z else np.zeros(0) for z in (X, Y, V))
        elif typ == 2:
            npts = self._i32(d, p); w = struct.unpack_from("<h", d, p[0])[0]; p[0] += 2
            v = np.frombuffer(d, "<i2" if short else "<f4", npts, p[0]).astype(np.float64)
            k = np.arange(npts); x = xo + k % w; y = yo + k // w
            ok = (v != -32768) if short else ~np.isnan(v)
            x, y, v = x[ok], y[ok], v[ok]
        else:
            raise ValueError(f"unknown block type {typ}")
        if len(v) != n:
            raise ValueError(f"block says {n} records, read {len(v)}")
        return x, y, v
