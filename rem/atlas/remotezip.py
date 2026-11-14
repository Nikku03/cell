"""Read one file inside a remote, uncompressed (store-mode) zip archive with HTTP range requests.

The reprocessed CRISPRi screens used by cellbank8 come as 0.2-20 GB store-mode zips holding gzip-chunked HDF5
(.h5ad) files. Only the effect matrix X is needed, about a tenth of each file, so instead of downloading the
archive, h5py reads the member through a file-like object that fetches only the byte ranges it asks for
(4 MB blocks, cached). The download link is a figshare redirect to a time-limited S3 URL, refreshed on 403.
"""

from __future__ import annotations
import collections
import io
import struct
import time
import urllib.error
import urllib.request

BLOCK = 4 << 20


class RangeFile(io.RawIOBase):
    def __init__(self, link, size=None, cache_blocks=64):
        self.link, self.url, self.pos = link, None, 0
        self.cache = collections.OrderedDict(); self.cache_blocks = cache_blocks
        self.fetched = 0
        self.size = size if size is not None else self._size()

    def _resolve(self):
        req = urllib.request.Request(self.link, method="HEAD", headers={"User-Agent": "cell-remotezip"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.url = r.geturl()

    def _get(self, a, b):
        for k in range(6):
            try:
                if self.url is None:
                    self._resolve()
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "cell-remotezip"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != b - a:
                    raise IOError(f"short read {len(data)} of {b - a}")
                self.fetched += len(data)
                return data
            except (urllib.error.HTTPError, urllib.error.URLError, IOError, TimeoutError) as ex:
                if isinstance(ex, urllib.error.HTTPError) and ex.code in (403, 400):
                    self.url = None                       # the signed URL expired: ask figshare again
                time.sleep(2 ** k)
        raise IOError(f"range {a}-{b} failed")

    def _size(self):
        self._resolve()
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0", "User-Agent": "cell-remotezip"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(r.headers["Content-Range"].split("/")[-1])

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        a = i * BLOCK; b = min(a + BLOCK, self.size)
        d = self._get(a, b); self.cache[i] = d
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return d

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 2 * BLOCK:                                 # large read: fetch directly, do not pollute the cache
            d = self._get(self.pos, self.pos + n); self.pos += n; return d
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, BLOCK)
            blk = self._block(i)[o:o + n]
            out += blk; self.pos += len(blk); n -= len(blk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


class MemberFile(io.RawIOBase):
    """A window [start, start + size) of a RangeFile, presented as its own file."""

    def __init__(self, base, start, size):
        self.base, self.start, self.size, self.pos = base, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        self.base.seek(self.start + self.pos); d = self.base.read(n); self.pos += len(d)
        return d

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def open_member(link, name_suffix):
    """Find a stored (uncompressed) member of a remote zip by name suffix; return (MemberFile, RangeFile)."""
    import zipfile
    base = RangeFile(link)
    z = zipfile.ZipFile(base)
    info = next(i for i in z.infolist() if i.filename.endswith(name_suffix))
    assert info.compress_type == zipfile.ZIP_STORED, "member is compressed"
    base.seek(info.header_offset); h = base.read(30)
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", h)
    assert sig == 0x04034B50
    n_name, n_extra = struct.unpack("<HH", h[26:30])
    start = info.header_offset + 30 + n_name + n_extra
    return MemberFile(base, start, info.file_size), base
