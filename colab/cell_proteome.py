"""CELL-TYPE-MATCHED PROTEIN ABUNDANCE, WITH PROVENANCE PER GENE.

WHY THIS EXISTS. cell_complete.json's `ppm` layer is not a cell. Measured directly from it:

    ALB   19,929 ppm     RBP4  24,443 ppm     APOA1 20,723 ppm
    ACTB   6,893 ppm     GAPDH  9,978 ppm     VIM    4,542 ppm

Albumin and retinol-binding protein at three times the abundance of actin is a plasma or
whole-organism composite, not any single cell type. Loop 92 found this through a physical closure
rather than a correlation: the ribosome budget came out at 262% utilisation, identically at every
proteome size because the constant cancels, and the top ten codon consumers were ALB, C3, RBP4, CFH
and HPX -- all secreted plasma proteins, made by hepatocytes and exported, so their plasma
concentration is not an intracellular steady state and treating it as one inflates synthesis demand
without bound.

A genuine HeLa dataset has been in the scratchpad since loop 74 -- PaxDb, Geiger 2012, the file loop
74 itself used -- and the model simply is not using it. In it, ALB, RBP4 and APOA1 are all 0.00 while
GAPDH is 300, VIM 249, RAD21 194 and CTCF 159. That is a cell.

WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO. It returns HeLa abundance keyed by gene symbol,
with a per-gene provenance tag, and it returns NOTHING for genes the HeLa measurement does not cover.
It does not fall back to the plasma composite, because a fallback would reintroduce exactly the
proteins that broke the budget while making the coverage look better. A caller that needs whole-genome
coverage has to either accept 7,222 genes or declare that it is mixing sources; it cannot do so by
accident.

THE COVERAGE IS THE POINT AND IS RETURNED, NOT HIDDEN. PaxDb's HeLa file covers about 37% of the
proteome by its own header, and some abundant and important proteins are absent from it -- ACTB and
POLR2A among them, which is a real limitation for any budget that needs them and is reported by
coverage() rather than papered over.
"""
import gzip
import re
from pathlib import Path

SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
HELA = SC / "paxdb_hela.txt"
GTF = SC / "ens_gtf.gz"

_CACHE = {}


def _ensp_to_symbol():
    if "map" in _CACHE:
        return _CACHE["map"]
    m = {}
    with gzip.open(GTF, "rt") as f:
        for ln in f:
            if ln.startswith("#") or "\tCDS\t" not in ln:
                continue
            a = ln.split("\t")[8]
            p = re.search(r'protein_id "([^"]+)"', a)
            g = re.search(r'gene_name "([^"]+)"', a)
            if p and g:
                m[p.group(1)] = g.group(1)
    _CACHE["map"] = m
    return m


def hela_ppm():
    """{gene symbol: ppm} from PaxDb HeLa (Geiger 2012). Zero-abundance entries are kept as 0.0,
    which is a measurement; genes absent from the file are absent from the dict, which is not."""
    if "hela" in _CACHE:
        return _CACHE["hela"]
    m = _ensp_to_symbol()
    out = {}
    with open(HELA) as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            q = ln.rstrip("\n").split("\t")
            if len(q) < 3:
                continue
            sym = m.get(q[1].split(".")[-1])
            if sym is None:
                continue
            try:
                v = float(q[2])
            except ValueError:
                continue
            out[sym] = max(out.get(sym, 0.0), v)
    _CACHE["hela"] = out
    return out


def coverage(gene_names):
    """What fraction of a requested gene list the HeLa measurement actually covers."""
    h = hela_ppm()
    present = [g for g in gene_names if g in h]
    nonzero = [g for g in present if h[g] > 0]
    return {"requested": len(gene_names), "measured": len(present), "nonzero": len(nonzero),
            "fraction_measured": len(present) / max(len(gene_names), 1),
            "total_ppm": sum(h[g] for g in present)}


def copies(gene_names, total_proteins):
    """Molecules per cell, for genes HeLa covers. ppm is normalised over the WHOLE proteome, so
    the conversion uses 1e6 as the denominator rather than the covered subset -- rescaling to the
    subset would silently inflate every abundance by the inverse of the coverage."""
    h = hela_ppm()
    return {g: h[g] / 1e6 * total_proteins for g in gene_names if g in h and h[g] > 0}
