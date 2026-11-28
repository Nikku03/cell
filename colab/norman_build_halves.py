"""Stream GSE133344 (Norman 2019) into pseudobulk profiles, SPLIT INTO TWO DISJOINT HALVES per program.

WHY HALVES, and why this supersedes norman_build.py for the composition test.  That script accumulates one
profile per guide identity, which is enough to ask "does A+B look like A plus B" but NOT enough to know whether
a disagreement is real. Two samples of the SAME program disagree too, because a top-20 ranking off a few hundred
cells is noisy. Today's depth test measured exactly how much: top-20 self-agreement runs 0.156 at 10 cells and
0.374 at 60. Without a matched floor, any composition result here would be uninterpretable -- which is the
mistake that produced the retracted "time reshapes 87% of the response" claim from sci-Plex.

So every program is accumulated TWICE, into two disjoint halves of its cells. That buys:

    the floor      half0 vs half1 of the SAME double -> how well the measurement agrees with itself
    the full       half0 + half1 -> identical to what norman_build.py would have produced, so nothing is lost

Cells are assigned to halves by a deterministic hash of the barcode, so the split is reproducible across runs and
does not depend on row order in the stream.

The matrix is 1.13 GB gzipped and 362M nonzero entries. It is never stored: it is streamed from GEO and
accumulated straight into (group x half x gene) sums.

SCOPE, stated here because it is easy to forget downstream: Norman 2019 is CRISPRa -- ACTIVATION, not knockout.
Every other number in this project is built on CRISPRi knockdown (gwps) or CRISPR KO. This dataset tests whether
PERTURBATION EFFECTS compose. Carrying that conclusion to knockouts is an inference, not a measurement.
"""
import csv
import gzip
import hashlib
import io
import subprocess
from pathlib import Path

import numpy as np

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/"
       "GSE133344_filtered_matrix.mtx.gz")


def half_of(barcode):
    """Deterministic, order-independent 0/1 assignment. md5 rather than hash() -- Python's hash is salted per
    process and would give a different split on every run."""
    return hashlib.md5(barcode.encode()).digest()[0] & 1


def main():
    genes = [l.split("\t")[1].strip() for l in gzip.open(SP / "filtered_genes.tsv.gz", "rt")]
    barcodes = [l.strip() for l in gzip.open(SP / "filtered_barcodes.tsv.gz", "rt")]
    bidx = {b: i for i, b in enumerate(barcodes)}
    print(f"genes {len(genes):,} barcodes {len(barcodes):,}", flush=True)

    ident = {}
    with gzip.open(SP / "filtered_cell_identities.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            b = row["cell_barcode"]
            if b in bidx:
                ident[bidx[b]] = row["guide_identity"]
    groups = sorted(set(ident.values()))
    gid = {g: i for i, g in enumerate(groups)}

    col_group = np.full(len(barcodes), -1, np.int32)
    col_half = np.zeros(len(barcodes), np.int8)
    for c, g in ident.items():
        col_group[c] = gid[g]
        col_half[c] = half_of(barcodes[c])
    assigned = int((col_group >= 0).sum())
    print(f"{len(groups)} guide identities; {assigned:,} cells assigned; "
          f"half0 {int((col_half[col_group >= 0] == 0).sum()):,} / "
          f"half1 {int((col_half[col_group >= 0] == 1).sum()):,}", flush=True)

    acc = np.zeros((len(groups), 2, len(genes)), np.float64)
    ncell = np.zeros((len(groups), 2), np.int64)
    for c in np.where(col_group >= 0)[0]:
        ncell[col_group[c], col_half[c]] += 1

    proc = subprocess.Popen(["curl", "-sS", "-m", "6000", URL], stdout=subprocess.PIPE)
    stream = gzip.GzipFile(fileobj=proc.stdout)
    head, n, transposed = 0, 0, False
    for raw in io.TextIOWrapper(stream, encoding="ascii"):
        if raw[0] == "%":
            continue
        if head == 0:
            head = 1
            nr, nc, nnz = (int(x) for x in raw.split())
            print(f"matrix {nr:,} x {nc:,}, {nnz:,} nonzero", flush=True)
            transposed = (nr == len(barcodes))
            continue
        a, b, v = raw.split()
        r = int(a) - 1
        c = int(b) - 1
        if transposed:
            r, c = c, r
        g = col_group[c]
        if g >= 0:
            acc[g, col_half[c], r] += float(v)
        n += 1
        if n % 20_000_000 == 0:
            print(f"  {n:,} entries", flush=True)
    proc.stdout.close()
    rc = proc.wait()
    print(f"done, {n:,} entries (curl exit {rc})", flush=True)
    assert n >= nnz * 0.99, f"stream truncated: {n:,} of {nnz:,} entries -- do not use this output"

    # CP10k then log1p, per group x half. The full profile is recovered by summing the halves BEFORE
    # normalising, so it is identical to a single-pass build rather than an average of two normalised halves.
    def norm(a):
        tot = a.sum(-1, keepdims=True)
        tot[tot == 0] = 1
        return np.log1p(a / tot * 1e4).astype(np.float32)

    prof_half = norm(acc)
    prof_full = norm(acc.sum(1))
    np.savez_compressed(SP.parent / "norman_pseudobulk_halves.npz",
                        groups=np.array(groups), genes=np.array(genes),
                        profile_half=prof_half, profile=prof_full,
                        ncell_half=ncell, ncell=ncell.sum(1))
    print(f"-> {SP.parent / 'norman_pseudobulk_halves.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
