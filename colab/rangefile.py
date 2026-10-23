import io, os, requests, threading
class RangeFile(io.RawIOBase):
    """Read-only file-like over an HTTP resource that redirects to a short-lived presigned URL."""
    def __init__(self, url, size=None, block=4 << 20):
        self.url, self.block = url, block
        self.pos, self._cache, self._lock = 0, {}, threading.Lock()
        self.nbytes = 0
        self.size = size if size is not None else self._probe()
    def _probe(self):
        r = requests.get(self.url, headers={"Range": "bytes=0-0"}, timeout=120,
                         allow_redirects=True, stream=True)
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        r.close()
        return int(cr.split("/")[-1])
    def _get(self, idx):
        with self._lock:
            if idx in self._cache:
                return self._cache[idx]
        lo = idx * self.block
        hi = min(lo + self.block, self.size) - 1
        for attempt in range(5):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={lo}-{hi}"},
                                 timeout=180, allow_redirects=True)
                if r.status_code in (200, 206):
                    b = r.content
                    self.nbytes += len(b)
                    with self._lock:
                        if len(self._cache) > 400:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[idx] = b
                    return b
            except Exception:
                pass
        raise IOError(f"range {lo}-{hi} failed")
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx = self.pos // self.block
            off = self.pos - idx * self.block
            b = self._get(idx)[off: off + n]
            if not b: break
            out += b; self.pos += len(b); n -= len(b)
        return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
