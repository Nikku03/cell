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

*** CORRECTION, ADDED AFTER ADVERSARIAL REVIEW. THIS FILE IS NOT A USABLE ABUNDANCE DISTRIBUTION. ***

The header calls column 3 "abundance" and the values do sum to 999,993, so they are ppm-normalised.
But the SHAPE is impossible for a molar abundance, and every number below was measured from the
file itself:

    median 150.0 EXCEEDS mean 136.7          no molar abundance distribution is left-skewed
    max/median 3.6                           Schwanhausser prot_copies gives 2,822
    top 1% of proteins hold 2.7% of mass     Schwanhausser gives 40.2%
    top 10% hold 17.4%                       Schwanhausser gives 83.8%
    only 918 DISTINCT values for 7,329 proteins, with 73 tied at exactly 190.0
    a hard floor at 4.77 with a large pile-up on it

A real proteome spans five to six orders of magnitude. This spans 112x, and it is rounded to about
three significant figures with both tails truncated.

WHAT THAT MEANS FOR CALLERS. RANK statements survive -- the ordering is still informative, and the
profile correlates with Schwanhausser's at rho +0.9161 across compartments. ABSOLUTE statements
about SPREAD do not, and they fail in the direction that flatters: a compressed distribution makes
everything look more tightly matched than it is. Measured directly on the same 2,039 complexes:

    within-complex log10 spread   this file 0.0766   Schwanhausser 0.5008
    19S proteasome                this file 0.0069   Schwanhausser 0.1305   19x larger
    tighter than random           this file 3.4x     Schwanhausser 2.2x

So loop 97's conclusion that obligate complexes are more matched than chance SURVIVES on real data
at 2.2x, and its headline that proteasome subunits agree "to within 2%" DOES NOT -- on a real
abundance distribution it is +/-35%. That claim was an artifact of this file and is retracted where
it was made.

USE Schwanhausser prot_copies for anything about spread, ratios or limiting subunits. Use this file
only for rank-order questions, and only with the above stated.

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
