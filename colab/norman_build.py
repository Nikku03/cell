"""Stream GSE133344 (Norman 2019) into pseudobulk profiles for singles, pairs and controls.

The matrix is 1.13 GB gzipped and would be many times that expanded, so it is never stored or
materialised: it is streamed from GEO and accumulated straight into a (group x gene) sum.
"""
import csv, gzip, io, subprocess, sys, collections
import numpy as np
from pathlib import Path

SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/norman")
URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/"
       "GSE133344_filtered_matrix.mtx.gz")

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
for c, g in ident.items():
    col_group[c] = gid[g]
print(f"{len(groups)} guide identities; {int((col_group >= 0).sum()):,} cells assigned", flush=True)

acc = np.zeros((len(groups), len(genes)), np.float64)
ncell = np.bincount(col_group[col_group >= 0], minlength=len(groups))

proc = subprocess.Popen(["curl", "-sS", "-m", "3000", URL], stdout=subprocess.PIPE)
stream = gzip.GzipFile(fileobj=proc.stdout)
head = 0
n = 0
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
    r = int(a) - 1; c = int(b) - 1
    if transposed:
        r, c = c, r
    g = col_group[c]
    if g >= 0:
        acc[g, r] += float(v)
    n += 1
    if n % 20_000_000 == 0:
        print(f"  {n:,} entries", flush=True)
proc.stdout.close(); proc.wait()
print(f"done, {n:,} entries", flush=True)

# CP10k then log1p, per group
tot = acc.sum(1, keepdims=True)
tot[tot == 0] = 1
prof = np.log1p(acc / tot * 1e4).astype(np.float32)
np.savez_compressed(SP / "norman_pseudobulk.npz",
                    groups=np.array(groups), genes=np.array(genes),
                    profile=prof, ncell=ncell)
print(f"-> {SP/'norman_pseudobulk.npz'}", flush=True)
