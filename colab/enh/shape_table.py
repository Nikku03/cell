"""The DNA shape and minor-groove electrostatics table, per pentamer.

WHY THIS FILE EXISTS AT ALL. The plan for the enhancer loop needs three quantities the sequence
does not give you directly: how wide the minor groove is at a site, how the base pairs are twisted
there, and how negative the electrostatic potential is inside that groove. A position weight matrix
carries none of them -- it is a per-base log-odds and cannot express that AAAAA and AAAAT present
different grooves to the same protein. Those quantities come from the Rohs-lab pentamer query
table: Monte Carlo simulations of every 5-mer in its own sequence context, tabulated once and
looked up thereafter (Zhou et al., NAR 2013, the DNAshape method; Chiu et al., NAR 2017, for the
minor-groove electrostatic potential column).

WHERE IT IS TAKEN FROM, AND WHY FROM THERE. The table is not distributed as a data file. It is a
987-entry array of strings compiled into DNAshapeR's C++ source (src/TableCompiler.cpp), which is
why the usual route to it is to install an R package. This module downloads the Bioconductor source
tarball and parses that array directly, so no R is involved and the numbers are the published ones
rather than a re-derivation.

WHAT EACH COLUMN IS. `load_data_from_vector` in DNAshapeR's properties.cpp fixes the field order,
and it is not the order the accessor names suggest -- the two Slide values come before the two Roll
values, which come before the two Twist values. Reading it wrong silently swaps roll for slide, so
the mapping is pinned here by index and checked against known values:

    dv[0]  MGW    minor groove width, Angstrom, at the CENTRAL base of the pentamer
    dv[3]  MGrW   major groove width, Angstrom
    dv[6]  ProT   propeller twist, degrees, central base pair
    dv[9]  dv[12] Slide, the two central base-pair steps
    dv[15] dv[18] Roll, the two central steps
    dv[21] dv[24] HelT, helix twist, the two central steps
    dv[27] EP     electrostatic potential in the minor groove, kT/e

THE 512 vs 1024 POINT. A pentamer and its reverse complement are the same physical object read from
the other strand, so the table stores one of each pair -- 512 canonical rows for the 1,024 possible
5-mers. The entries that ARE strand-symmetric (MGW, major width, ProT, EP -- all properties of the
central base pair or of the groove) copy across unchanged. The inter-step quantities (Roll, HelT,
Slide) come in ordered pairs, and reverse-complementing reverses the order, so step 1 and step 2
swap. Getting that swap wrong would put the wrong step's roll at the wrong position on one strand
out of two, which is exactly the kind of half-silent error that shows up as a small unexplained
asymmetry, so it is done explicitly and then verified by re-deriving every canonical row from its
own reverse complement.

The table also carries methylated-cytosine (M) and inosine (Q) pentamers. Those are dropped: the
sequence this project scans is unmethylated reference genome, and keeping them would mean carrying
an alphabet the scanner cannot produce.

VERIFIED ON WRITE, not asserted in prose: all 1,024 codes filled; the AAAAA row reproduces the
published narrow-groove A-tract signature (MGW 3.38 A, ProT -16.51 deg, EP -10.1 kT/e); and the
round trip through reverse complement is exact for every pentamer.

Output: colab/data/dna_shape.npz -- seven float32 arrays of length 1024, indexed by the base-4 code
of the pentamer with A=0, C=1, G=2, T=3, most significant digit first.
"""
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "dna_shape.npz"

VERSION = "1.40.0"
TARBALL = f"DNAshapeR_{VERSION}.tar.gz"
URL = f"https://bioconductor.org/packages/release/bioc/src/contrib/{TARBALL}"
MEMBER = "DNAshapeR/src/TableCompiler.cpp"

BASES = "ACGT"
CODE = {b: i for i, b in enumerate(BASES)}
RC = {"A": "T", "C": "G", "G": "C", "T": "A"}

# name -> index into the 90-value row, from properties.cpp::load_data_from_vector
FIELDS = {"mgw": 0, "mgrw": 3, "prot": 6,
          "slide1": 9, "slide2": 12, "roll1": 15, "roll2": 18,
          "helt1": 21, "helt2": 24, "ep": 27}
# the pairs that swap when the pentamer is reverse complemented
SWAP = [("slide1", "slide2"), ("roll1", "roll2"), ("helt1", "helt2")]
SYMMETRIC = ["mgw", "mgrw", "prot", "ep"]


def kcode(pent):
    """Base-4 index of a 5-mer, most significant digit first. Returns None if it is not over ACGT."""
    c = 0
    for ch in pent:
        if ch not in CODE:
            return None
        c = c * 4 + CODE[ch]
    return c


def rc(pent):
    return "".join(RC[c] for c in reversed(pent))


def fetch():
    p = SP / TARBALL
    if not p.exists():
        SP.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=600) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(p)
    with tarfile.open(p, "r:gz") as t:
        return t.extractfile(MEMBER).read().decode("utf-8", "replace")


def parse(src):
    """Every quoted string in the QueryTable array with 91 whitespace-separated fields."""
    rows = {}
    skipped_alphabet = 0
    for m in re.finditer(r'"([^"]+)"', src):
        parts = m.group(1).split()
        if len(parts) != 91:
            continue
        pent, vals = parts[0], parts[1:]
        if len(pent) != 5:
            continue
        if kcode(pent) is None:
            skipped_alphabet += 1          # methylated (M) or inosine (Q) variants
            continue
        rows[pent] = [float(v) for v in vals]
    return rows, skipped_alphabet


def build():
    src = fetch()
    rows, skipped = parse(src)
    print(f"  DNAshapeR {VERSION}: {len(rows)} ACGT pentamers parsed, "
          f"{skipped} methylated/inosine rows dropped")
    if len(rows) != 512:
        raise SystemExit(f"expected 512 canonical ACGT pentamers, parsed {len(rows)}")

    arr = {k: np.full(1024, np.nan, dtype=np.float64) for k in FIELDS}
    for pent, vals in rows.items():
        c = kcode(pent)
        for k, i in FIELDS.items():
            arr[k][c] = vals[i]

    # fill the reverse complements. Symmetric fields copy; ordered step pairs swap.
    filled = 0
    for c in range(1024):
        if not np.isnan(arr["mgw"][c]):
            continue
        pent = "".join(BASES[(c >> (2 * (4 - i))) & 3] for i in range(5))
        d = kcode(rc(pent))
        if np.isnan(arr["mgw"][d]):
            raise SystemExit(f"neither {pent} nor its reverse complement is in the table")
        for k in SYMMETRIC:
            arr[k][c] = arr[k][d]
        for a, b in SWAP:
            arr[a][c], arr[b][c] = arr[b][d], arr[a][d]
        filled += 1
    print(f"  {filled} reverse-complement entries filled; {1024 - filled} came from the table")

    for k, v in arr.items():
        if np.isnan(v).any():
            raise SystemExit(f"{k}: {int(np.isnan(v).sum())} unfilled codes")

    # the published A-tract signature, as a check that the column mapping is the right one
    a5 = kcode("AAAAA")
    for k, want in (("mgw", 3.38), ("prot", -16.51), ("ep", -10.10), ("mgrw", 12.27)):
        got = float(arr[k][a5])
        if abs(got - want) > 0.02:
            raise SystemExit(f"AAAAA {k} = {got}, expected {want} -- column mapping is wrong")
    print(f"  AAAAA checks out: MGW {arr['mgw'][a5]:.2f} A, ProT {arr['prot'][a5]:.2f} deg, "
          f"EP {arr['ep'][a5]:.2f} kT/e")

    # the round trip: every pentamer must agree with its own reverse complement under the swap
    bad = 0
    for c in range(1024):
        pent = "".join(BASES[(c >> (2 * (4 - i))) & 3] for i in range(5))
        d = kcode(rc(pent))
        for k in SYMMETRIC:
            if abs(arr[k][c] - arr[k][d]) > 1e-9:
                bad += 1
        for a, b in SWAP:
            if abs(arr[a][c] - arr[b][d]) > 1e-9 or abs(arr[b][c] - arr[a][d]) > 1e-9:
                bad += 1
    if bad:
        raise SystemExit(f"{bad} reverse-complement inconsistencies")
    print("  reverse-complement round trip exact for all 1,024 codes")

    for k in FIELDS:
        v = arr[k]
        print(f"    {k:6} mean {v.mean():8.3f}  sd {v.std():7.3f}  "
              f"range [{v.min():8.3f}, {v.max():8.3f}]")

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **{k: v.astype(np.float32) for k, v in arr.items()})
    print(f"  -> {OUT} ({OUT.stat().st_size:,} bytes)")


def load():
    """{field: float32[1024]} indexed by base-4 pentamer code."""
    if not OUT.exists():
        raise SystemExit(f"{OUT} missing -- run `python colab/enh/shape_table.py` first")
    z = np.load(OUT)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    print("=" * 100)
    print("DNA SHAPE + MINOR-GROOVE ELECTROSTATICS, per pentamer, from the DNAshapeR query table")
    print("=" * 100)
    build()
