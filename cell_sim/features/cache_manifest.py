"""On-disk manifest of cached feature tables.

Every parquet file in ``cell_sim/features/cache/`` should have a
corresponding entry in ``manifest.json`` recording:

  * ``sha256``     — content hash of the parquet file (hex string)
  * ``version``    — semver-ish version tag supplied by the producer
  * ``rows``       — number of rows in the parquet (populated at
                     registration time; purely informational)
  * ``created_at`` — UTC ISO-8601 timestamp when the entry was added

The manifest is the authority on *what the cache ought to contain*;
on every ``FeatureRegistry.load`` call we recompute the SHA256 of
the parquet and compare against the manifest. A mismatch means
either:

  - the parquet was tampered with since it was cached, or
  - the manifest was not updated after the parquet was regenerated

Either way, the load raises ``ValueError`` with a clear message so
a downstream sweep never silently feeds a detector stale features.

This module has zero heavy dependencies: ``hashlib``, ``json``,
``dataclasses``, and ``pathlib`` from the stdlib, plus
``pyarrow`` only indirectly via the producer (the manifest itself
doesn't need to parse parquet — it only hashes raw bytes).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 of ``path`` streaming in 1 MiB chunks so we
    don't OOM on multi-GB parquet files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 with second precision. Avoids timezone
    ambiguity; never depends on the caller's locale."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CachedFeatureManifest:
    """In-memory representation of ``manifest.json``.

    The on-disk schema is::

        {
            "sources": {
                "<name>": {
                    "sha256": "<hex>",
                    "version": "<semver>",
                    "rows": <int>,
                    "created_at": "<iso8601 utc>"
                },
                ...
            }
        }

    An empty manifest has ``sources == {}``.
    """

    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- serialisation ----

    def save(self, path: Path) -> None:
        """Atomically write the manifest to ``path``.

        Writes to a sibling ``.tmp`` file and renames so concurrent
        readers can't observe a half-written JSON.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"sources": self.sources}, indent=2,
                                   sort_keys=True) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CachedFeatureManifest":
        """Read ``path``. If the file doesn't exist return an empty
        manifest (so ``FeatureRegistry`` can be constructed before any
        source has been cached)."""
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "sources" not in raw:
            raise ValueError(
                f"manifest at {path} is malformed: missing top-level "
                f"'sources' key"
            )
        if not isinstance(raw["sources"], dict):
            raise ValueError(
                f"manifest at {path} has a non-dict 'sources' value: "
                f"{type(raw['sources']).__name__}"
            )
        return cls(sources=dict(raw["sources"]))

    # ---- registration ----

    def add(self, name: str, parquet_path: Path, version: str) -> None:
        """Compute the SHA-256 of ``parquet_path`` and record a
        manifest entry under ``name``.

        ``rows`` is populated opportunistically — if pyarrow can open
        the parquet we record the row count; if not we default to -1
        (the row count is informational only, never used for
        validation). Raises ``FileNotFoundError`` if the parquet
        doesn't exist.
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"cannot register feature source {name!r}: "
                f"parquet {parquet_path} does not exist"
            )
        sha = _sha256_of_file(parquet_path)
        rows = -1
        try:
            import pyarrow.parquet as _pq
            meta = _pq.read_metadata(parquet_path)
            rows = int(meta.num_rows)
        except Exception:
            # pyarrow unavailable or parquet unreadable — don't fail
            # the registration. The SHA check is the one that matters.
            pass
        self.sources[name] = {
            "sha256": sha,
            "version": version,
            "rows": rows,
            "created_at": _utc_now_iso(),
        }

    # ---- verification ----

    def verify(self, name: str, parquet_path: Path) -> bool:
        """Recompute the parquet's SHA-256 and compare against the
        recorded entry. Returns True only when the manifest contains
        an entry for ``name`` AND the hash matches.

        Returns False (never raises) when:
          * the file doesn't exist
          * the manifest has no entry for ``name``
          * the recorded SHA doesn't match the current file
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            return False
        entry = self.sources.get(name)
        if entry is None:
            return False
        stored = entry.get("sha256")
        if not stored:
            return False
        return _sha256_of_file(parquet_path) == stored

    def remove(self, name: str) -> None:
        """Drop an entry. No-op if ``name`` isn't present — the
        cache directory may be gc'd before we notice."""
        self.sources.pop(name, None)


__all__ = ["CachedFeatureManifest"]
