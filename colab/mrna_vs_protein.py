"""DOES mRNA RESPONSE PREDICT PROTEIN RESPONSE? Measured on our own perturbation data, not quoted.

THE PREMISE BEING TESTED. The standard claim is that mRNA abundance correlates with protein copy number at
only R^2 ~ 0.30-0.40, so a Perturb-seq readout is a poor proxy for what a cell is actually doing. That is a
statement about STATIC ABUNDANCE across genes. It is not the quantity this project needs.

What this project needs is narrower and more answerable: **when a knockout changes a gene's mRNA, does the
protein change too?** A model that predicts perturbation RESPONSES only fails if the mRNA response and the
protein response disagree. Static abundance correlation could be poor while response correlation is fine
(both driven by the same perturbation), or the reverse (buffering absorbs the mRNA change). Nobody's quoted
R^2 answers it, and it is measurable on data already on disk.

THE DATA. Papalexi et al. ECCITE-seq: 20,729 THP-1 cells, 111 guides, and FOUR antibody-derived tags --
CD86, PD-L1, PD-L2, CD366 -- measured in the same cells as the full transcriptome. Four proteins is thin,
and that is stated rather than glossed: this is a four-point test, not a proteome-wide one. Its value is
that the four proteins have unambiguous genes (CD86, CD274, PDCD1LG2, HAVCR2) so mRNA and protein can be
compared for the SAME molecule under the SAME perturbation in the SAME cells.

THE JOIN, VERIFIED RATHER THAN ASSUMED. The ADT matrix and the h5ad use different barcode conventions and a
naive join silently returns ZERO overlap:

    ADT columns   l1_AAACCTGAGCCAGAAC     lane prefix l1..l8
    h5ad index    AAACCTGAGCCAGAAC-1      10x -N suffix, added to disambiguate barcodes seen in >1 lane

Stripping only the prefix gives 98.61% and looks fine. Normalising BOTH conventions gives **20,729/20,729 =
100.00% positional agreement**, and the guide matrix shares the ADT column order exactly. Checked because a
98.6% join would have produced entirely plausible numbers from 288 mismatched cells.

THE TWO TESTS
  1 DIRECT   knock out gene G, does G's own protein fall? The positive control. If this fails, either the
             guides do not work or the join is wrong, and nothing else is interpretable.
  2 RESPONSE across all guides, does the mRNA log-fold-change predict the protein log-fold-change for the
             same molecule? This is the number the roadmap's premise turns on.

CONTROL: guide labels are permuted, which destroys the perturbation-response link while preserving every
distribution. Reported alongside, because a correlation between two noisy vectors over 100+ guides is not
automatically meaningful.
"""
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
PROT = SP / "protein"
ADT = PROT / "GSM4633615_ECCITE_ADT_counts.tsv.gz"
H5 = SP / "papalexi.h5ad"

# ADT tag -> the gene whose mRNA encodes it. Without this mapping there is nothing to compare.
TAG2GENE = {"CD86": "CD86", "PDL1": "CD274", "PDL2": "PDCD1LG2", "CD366": "HAVCR2"}
MIN_CELLS = 25          # a guide measured in fewer cells is noise, not a measurement


def load_adt():
    with gzip.open(ADT, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        cells = [c.strip('"').split("_", 1)[1] for c in hdr[1:]]
        tags, M = [], []
        for line in fh:
            f = line.rstrip("\n").split("\t")
            tags.append(f[0].strip('"'))
            M.append([float(x) for x in f[1:]])
    return cells, tags, np.array(M)                     # (tags, cells)


def load_rna(want_genes):
    """Per-cell counts for the genes we need, plus total UMI and guide id. CSC, read by column."""
    import h5py
    with h5py.File(H5, "r") as h:
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        if isinstance(ik, bytes):
            ik = ik.decode()
        genes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]])
        oi = h["obs"].attrs.get("_index", "_index")
        if isinstance(oi, bytes):
            oi = oi.decode()
        bc = [re.sub(r"-\d+$", "", x.decode() if isinstance(x, bytes) else str(x))
              for x in h["obs"][oi][:]]
        g = h["obs"]["guide_id"]
        if isinstance(g, h5py.Group):
            cats = [x.decode() for x in g["categories"][:]]
            gid = np.array([cats[c] if 0 <= c < len(cats) else "?" for c in g["codes"][:]])
        else:
            gid = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
        X = h["X"]
        shape = tuple(X.attrs["shape"])
        data, indices, indptr = X["data"], X["indices"], X["indptr"]
        ncell = shape[0]
        gi = {n: i for i, n in enumerate(genes)}
        cols = {}
        for gname in want_genes:
            j = gi.get(gname)
            if j is None:
                continue
            s, e = int(indptr[j]), int(indptr[j + 1])       # CSC: column j is gene j
            col = np.zeros(ncell, np.float32)
            if e > s:
                col[indices[s:e]] = data[s:e]
            cols[gname] = col
        # total UMI needs every column; sum per row via the whole index array once
        tot = np.zeros(ncell, np.float64)
        np.add.at(tot, indices[:], data[:])
    return bc, gid, cols, tot


def lfc(vals, tot, mask_pert, mask_ctrl):
    """log2 fold change of a size-normalised mean. Size normalisation is not optional: guides differ in
    sequencing depth, and an un-normalised mean would read library size as biology."""
    n = vals / np.maximum(tot, 1.0)
    a = n[mask_pert].mean() if mask_pert.sum() else np.nan
    b = n[mask_ctrl].mean() if mask_ctrl.sum() else np.nan
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return np.nan
    return float(np.log2((a + 1e-12) / (b + 1e-12)))


def main():
    from scipy import stats
    print("=" * 100)
    print("mRNA RESPONSE vs PROTEIN RESPONSE -- measured, on Papalexi ECCITE-seq")
    print("=" * 100)
    if not ADT.exists():
        raise SystemExit(f"ADT matrix absent at {ADT}")

    cells, tags, A = load_adt()
    genes_needed = sorted(set(TAG2GENE.values()))
    bc, gid, cols, tot = load_rna(genes_needed)
    assert len(bc) == len(cells), f"cell count mismatch {len(bc)} vs {len(cells)}"
    agree = sum(1 for a, b in zip(cells, bc) if a == b)
    print(f"  join: {agree:,}/{len(bc):,} positional agreement ({100*agree/len(bc):.2f}%)")
    assert agree == len(bc), "barcode join is not exact -- refusing to interpret anything downstream"
    print(f"  {len(bc):,} cells, {len(tags)} ADT tags {tags}, guides {len(set(gid))}")
    missing = [g for g in genes_needed if g not in cols]
    print(f"  genes found in the transcriptome: {sorted(cols)}"
          + (f"   MISSING {missing}" if missing else ""))

    # ADT is size-normalised the same way as RNA: per-cell total ADT, so a cell with more antibody
    # sequencing does not read as more protein.
    adt_tot = A.sum(0)
    ctrl = np.array([g.startswith("NT") for g in gid])
    print(f"  non-targeting control cells: {int(ctrl.sum()):,}")

    guides = sorted({g for g in set(gid) if not g.startswith("NT")})
    R = {"n_cells": int(len(bc)), "tags": tags, "n_guides": len(guides),
         "n_control_cells": int(ctrl.sum()), "direct": {}, "response": {}}

    # ---- TEST 1: DIRECT. knock out G, does G's own protein fall? ----
    print("\n  [1] DIRECT EFFECT -- knock out the gene, does its own protein fall?")
    print(f"      {'tag':7s} {'gene':10s} {'guides':>7s} {'cells':>7s} {'mRNA lfc':>9s} {'prot lfc':>9s}")
    for tag, gene in TAG2GENE.items():
        if tag not in tags or gene not in cols:
            continue
        ti = tags.index(tag)
        sel = np.array([g.startswith(gene) for g in gid])
        ng = len({g for g in gid[sel]})
        if sel.sum() < MIN_CELLS:
            print(f"      {tag:7s} {gene:10s} {ng:7d} {int(sel.sum()):7d}   too few cells")
            continue
        m = lfc(cols[gene], tot, sel, ctrl)
        p = lfc(A[ti], adt_tot, sel, ctrl)
        print(f"      {tag:7s} {gene:10s} {ng:7d} {int(sel.sum()):7d} {m:+9.4f} {p:+9.4f}")
        R["direct"][tag] = {"gene": gene, "n_guides": ng, "n_cells": int(sel.sum()),
                            "mrna_lfc": m, "protein_lfc": p}

    # ---- TEST 2: RESPONSE. across guides, does mRNA lfc track protein lfc? ----
    print("\n  [2] RESPONSE CORRELATION -- across all guides, per molecule")
    print(f"      {'tag':7s} {'gene':10s} {'n guides':>9s} {'Pearson r':>10s} {'R2':>8s} "
          f"{'Spearman':>9s}   {'shuffled r':>11s}")
    rng = np.random.default_rng(0)
    for tag, gene in TAG2GENE.items():
        if tag not in tags or gene not in cols:
            continue
        ti = tags.index(tag)
        mv, pv = [], []
        for gu in guides:
            sel = gid == gu
            if sel.sum() < MIN_CELLS:
                continue
            a = lfc(cols[gene], tot, sel, ctrl)
            b = lfc(A[ti], adt_tot, sel, ctrl)
            if np.isfinite(a) and np.isfinite(b):
                mv.append(a); pv.append(b)
        mv, pv = np.array(mv), np.array(pv)
        if len(mv) < 10:
            print(f"      {tag:7s} {gene:10s} {len(mv):9d}   too few guides")
            continue
        r = float(stats.pearsonr(mv, pv)[0])
        rho = float(stats.spearmanr(mv, pv)[0])
        # CONTROL: permute the protein vector, destroying the guide pairing
        sh = [abs(float(stats.pearsonr(mv, pv[rng.permutation(len(pv))])[0])) for _ in range(200)]
        print(f"      {tag:7s} {gene:10s} {len(mv):9d} {r:+10.4f} {r*r:8.4f} {rho:+9.4f}   "
              f"{np.mean(sh):+7.4f}+/-{np.std(sh):.4f}")
        R["response"][tag] = {"gene": gene, "n_guides": int(len(mv)), "pearson_r": r, "r2": r * r,
                              "spearman": rho, "shuffled_abs_r_mean": float(np.mean(sh)),
                              "shuffled_abs_r_sd": float(np.std(sh))}

    if R["response"]:
        r2s = [v["r2"] for v in R["response"].values()]
        R["mean_r2"] = float(np.mean(r2s))
        print(f"\n  mean R2 of mRNA response vs protein response: {np.mean(r2s):.4f} "
              f"over {len(r2s)} molecules")
        print("  (the literature figure often quoted, ~0.30-0.40, is for STATIC ABUNDANCE across genes;")
        print("   this is the RESPONSE correlation for the same molecule under the same perturbation)")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "mrna_vs_protein.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'mrna_vs_protein.json'}")


if __name__ == "__main__":
    main()
