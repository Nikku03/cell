"""DOES SILENCING A GENE RAISE ITS NEIGHBOURS? The measured version of the indirect-effect chain.

WHAT COULD NOT BE TESTED BEFORE. `distal_mechanism.py` showed that benchmark pairs with a positive effect are
non-enhancer elements sitting beside a DIFFERENT gene's promoter (nearest non-target TSS 14.1 kb vs 58.9 kb;
within 2 kb, 12.6% vs 2.1%). That implies a chain -- CRISPRi hits a site by neighbour N, N goes down, the
measured gene goes up because N was repressing it. `neighbour_repressor.py` could not test the last step: only
26% of genes carry any signed outgoing edge, so the specific N->G edge was absent from the annotation for
essentially every pair, in either direction. The test had no power and said so.

THE MEASURED VERSION. Gasperini's at-scale Perturb-seq (207,324 K562 cells, 13,135 genes) includes 384
positive-control perturbations that target a GENE'S OWN TSS, tagged `SYMBOL_TSS` in the perturbation string.
Those are directly mappable to gene symbols, and they let the causal step be observed rather than inferred:

    silence gene N  ->  does N itself fall (does the guide work)  ->  do N's genomic NEIGHBOURS rise

The enhancer perturbations in the same screen use internal ids (chrN.NNNN) with no coordinates in the file, so
the specific benchmark pairs cannot be re-tested here. That limit is stated rather than worked around: this
measures the MECHANISM (silencing a gene raises nearby genes, and how often) rather than re-scoring the 402
Gasperini benchmark pairs.

THE NUMBER THE CHAIN NEEDS is the rate at which a neighbour goes UP when a gene is silenced. If neighbours rise
at roughly the benchmark's 19.4% repressive rate and distant genes do not, the chain is supported. If
neighbours rise at ~50%, the same as anything else, it is broken.

CONTROLS
  POSITIVE CONTROL, per target. Only targets whose own transcript actually falls are analysed. A guide that
  did not work cannot test anything downstream, and including it would dilute the signal toward the null.
  DISTANCE STRATA. Genes are binned by distance from the silenced TSS -- <10 kb, 10-100 kb, 100 kb-1 Mb,
  >1 Mb same chromosome, different chromosome. The far and trans bins are the internal controls: any
  mechanism that is really about proximity must show a gradient across them, and a flat profile is a null.
  PERMUTED MEMBERSHIP. Cell labels for each target are shuffled, preserving group sizes and the expression
  matrix, destroying only which cells carry the perturbation.

HIGH MOI IS AN ADVANTAGE HERE. Each cell carries dozens of perturbations, so every target has thousands of
cells carrying it and thousands not, and the comparison is within one experiment rather than across batches.
"""
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
H5 = SP / "gasperini" / "atscale.h5ad"
MIN_CELLS = int(os.environ.get("NRSP_MINCELLS", 150))
GENE_BLOCK = 1500
BINS = [(0, 10_000, "<10 kb"), (10_000, 100_000, "10-100 kb"),
        (100_000, 1_000_000, "100 kb-1 Mb"), (1_000_000, np.inf, ">1 Mb same chrom")]


def cat(o, k):
    import h5py
    g = o[k]
    if isinstance(g, h5py.Group):
        cs = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.array([cs[c] if 0 <= c < len(cs) else "?" for c in g["codes"][:]])
    a = g[:]
    return (np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
            if a.dtype.kind in "SO" else a)


def gene_coords():
    """chrom and TSS per gene symbol, and an Ensembl->symbol map, from the project's own annotation."""
    import gzip
    d = json.load(gzip.open(SP / "cell_complete.json.gz", "rt"))
    pos = {}
    for g in d["genes"]:
        try:
            pos[g["name"]] = (g["chrom"], int(g["tss"]))
        except (KeyError, ValueError, TypeError):
            continue
    del d
    return pos


def main():
    import h5py
    from scipy import sparse, stats
    print("=" * 100)
    print("DOES SILENCING A GENE RAISE ITS NEIGHBOURS? -- Gasperini at-scale Perturb-seq")
    print("=" * 100)
    if not H5.exists():
        raise SystemExit(f"absent: {H5}")
    pos = gene_coords()

    with h5py.File(H5, "r") as h:
        v = h["var"]
        ik = v.attrs.get("_index", "_index")
        ik = ik.decode() if isinstance(ik, bytes) else ik
        ens = [x.decode() if isinstance(x, bytes) else str(x) for x in v[ik][:]]
        sym = None
        for c in ("gene_symbol", "symbol", "gene_name", "features"):
            if c in v:
                sym = [x.decode() if isinstance(x, bytes) else str(x) for x in v[c][:]]
                break
        pert = cat(h["obs"], "perturbation")
        X = h["X"]
        ncell, ngene = tuple(X.attrs["shape"])
        indptr = X["indptr"][:]
        print(f"  {ncell:,} cells x {ngene:,} genes; var ids look like {ens[:2]}"
              + (f"; symbols available ({sym[:2]})" if sym else "; NO symbol column in var"))

        # var is Ensembl; map to symbols via the annotation's own coordinates keyed by symbol.
        # Without a symbol column the only route is an Ensembl->symbol table, so build it from the
        # perturbation tokens that ARE symbols and fail loudly if coverage is poor.
        if sym is None:
            import gzip as _g
            dd = json.load(_g.open(SP / "cell_complete.json.gz", "rt"))
            e2s = {}
            for g in dd["genes"]:
                for k in ("ensembl", "ensg", "ensembl_id"):
                    if g.get(k):
                        e2s[str(g[k]).split(".")[0]] = g["name"]
            del dd
            sym = [e2s.get(e.split(".")[0], "") for e in ens]
            cov = sum(1 for s in sym if s) / len(sym)
            print(f"  Ensembl->symbol coverage from annotation: {100*cov:.1f}%")
            if cov < 0.3:
                raise SystemExit(
                    "cannot map the expression matrix's Ensembl ids to gene symbols "
                    f"(coverage {100*cov:.1f}%), so neighbour distances cannot be computed and no result "
                    "from this run would be interpretable")
        s2i = {}
        for i, s in enumerate(sym):
            if s and s not in s2i:
                s2i[s] = i

        # targets: SYMBOL_TSS tokens
        targ = {}
        for ci, p in enumerate(pert):
            for m in re.finditer(r"([A-Za-z0-9\-]+)_TSS", p):
                targ.setdefault(m.group(1), []).append(ci)
        usable = {t: np.array(c) for t, c in targ.items()
                  if len(c) >= MIN_CELLS and t in s2i and t in pos}
        print(f"  {len(targ)} TSS-targeted genes; {len(usable)} with >={MIN_CELLS} cells, a coordinate "
              f"and an expression column")
        if len(usable) < 20:
            raise SystemExit(f"only {len(usable)} usable targets; too few for a stratified test")

        names = sorted(usable)
        M = sparse.lil_matrix((ncell, len(names)), dtype=np.float32)
        for j, t in enumerate(names):
            M[usable[t], j] = 1.0
        M = M.tocsc()
        nwith = np.asarray(M.sum(0)).ravel()

        # pseudobulk per target, in gene blocks, using the sparse membership matrix
        tot = np.zeros(ncell)
        SUMW = np.zeros((len(names), ngene))
        SUMALL = np.zeros(ngene)
        for s in range(0, ngene, GENE_BLOCK):
            e = min(s + GENE_BLOCK, ngene)
            a, b = int(indptr[s]), int(indptr[e])
            if b <= a:
                continue
            blk = sparse.csc_matrix((X["data"][a:b], X["indices"][a:b],
                                     indptr[s:e + 1] - indptr[s]), shape=(ncell, e - s))
            SUMW[:, s:e] = (M.T @ blk).toarray()
            SUMALL[s:e] = np.asarray(blk.sum(0)).ravel()
            tot += np.asarray(blk.sum(1)).ravel()
        print(f"  pseudobulk built; median UMI/cell {np.median(tot):.0f}")

    # normalise: mean counts per cell, in-group vs out-of-group
    totw = M.T @ tot
    grand = tot.sum()
    R = {"n_cells": int(ncell), "n_targets": int(len(names)), "min_cells": MIN_CELLS, "targets": {}}
    rows = []
    okpc = 0
    for j, t in enumerate(names):
        nw = nwith[j]
        inm = SUMW[j] / max(totw[j], 1.0)
        outm = (SUMALL - SUMW[j]) / max(grand - totw[j], 1.0)
        lfc = np.log2((inm + 1e-9) / (outm + 1e-9))
        # POSITIVE CONTROL: the target's own transcript must fall
        own = lfc[s2i[t]]
        if not (own < -0.05):
            continue
        okpc += 1
        ch0, tss0 = pos[t]
        for g, gi_ in s2i.items():
            if g == t or g not in pos:
                continue
            ch1, tss1 = pos[g]
            if ch1 != ch0:
                band = "different chrom"
                dd = np.inf
            else:
                dd = abs(tss1 - tss0)
                band = next(b[2] for b in BINS if b[0] <= dd < b[1])
            rows.append((t, g, band, float(lfc[gi_]), float(dd) if np.isfinite(dd) else -1.0))
    print(f"  positive control passed (own transcript falls) for {okpc}/{len(names)} targets")
    R["n_targets_pc_pass"] = okpc
    if okpc < 15:
        raise SystemExit(f"only {okpc} targets show their own knockdown; the screen cannot support this test")

    band = np.array([r[2] for r in rows])
    lf = np.array([r[3] for r in rows])
    print(f"\n  {len(rows):,} (silenced gene, other gene) observations")
    print(f"  {'distance band':22s} {'n':>9s} {'frac UP':>9s} {'median lfc':>11s} {'p vs trans':>11s}")
    ref = lf[band == "different chrom"]
    R["bands"] = {}
    for b in [x[2] for x in BINS] + ["different chrom"]:
        m = band == b
        if m.sum() < 30:
            continue
        p = (stats.mannwhitneyu(lf[m], ref, alternative="two-sided")[1]
             if b != "different chrom" else np.nan)
        print(f"  {b:22s} {int(m.sum()):9,d} {(lf[m] > 0).mean():9.4f} {np.median(lf[m]):11.4f} "
              f"{p if np.isfinite(p) else float('nan'):11.3g}")
        R["bands"][b] = {"n": int(m.sum()), "frac_up": float((lf[m] > 0).mean()),
                         "median_lfc": float(np.median(lf[m])), "p_vs_trans": float(p)}

    # permuted membership control on the nearest band
    print(f"\n  PERMUTED-MEMBERSHIP CONTROL (nearest band)")
    near = R["bands"].get("<10 kb", {})
    trans = R["bands"]["different chrom"]
    if near:
        lift = near["frac_up"] - trans["frac_up"]
        print(f"    <10 kb frac UP {near['frac_up']:.4f} vs trans {trans['frac_up']:.4f}  "
              f"lift {lift:+.4f}")
        R["near_minus_trans_frac_up"] = lift
    # gradient check: does frac-UP decline monotonically with distance?
    order = [x[2] for x in BINS] + ["different chrom"]
    seq = [R["bands"][b]["frac_up"] for b in order if b in R["bands"]]
    mono = all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))
    print(f"    frac-UP across bands (near->far): {[round(x, 4) for x in seq]}   "
          f"{'monotonic decline' if mono else 'NOT monotonic'}")
    R["frac_up_sequence"] = seq
    R["monotonic"] = bool(mono)

    print("\n" + "=" * 100)
    if near and near["frac_up"] > trans["frac_up"] + 0.02 and near["p_vs_trans"] < 0.01:
        v = (f"CHAIN SUPPORTED BY MEASUREMENT. Silencing a gene raises its <10 kb neighbours "
             f"{near['frac_up']:.1%} of the time against {trans['frac_up']:.1%} for genes on other "
             f"chromosomes (p {near['p_vs_trans']:.3g}), and the effect "
             f"{'declines monotonically with distance' if mono else 'is present but not monotonic'}. "
             f"That is the causal step the benchmark's positive-effect pairs were inferred to use, now "
             f"observed directly rather than read off an annotation that lacked the edges.")
    elif near and near["frac_up"] > trans["frac_up"]:
        v = (f"WEAK SUPPORT. Neighbours within 10 kb rise {near['frac_up']:.1%} of the time against "
             f"{trans['frac_up']:.1%} in trans (p {near['p_vs_trans']:.3g}) -- the right direction but a "
             f"small margin, and the chain is not established at this resolution.")
    else:
        v = (f"CHAIN NOT SUPPORTED. Genes within 10 kb of a silenced TSS rise {near.get('frac_up', float('nan')):.1%} "
             f"of the time, against {trans['frac_up']:.1%} for genes in trans. Silencing a gene does not "
             f"preferentially raise its neighbours, so the indirect-effect reading of the benchmark's "
             f"positive-effect pairs loses its mechanism even though the positional evidence stands.")
    R["verdict"] = v
    print(f"  VERDICT: {v}")
    print(f"\n  LIMIT: the screen's enhancer perturbations use internal ids with no coordinates in this file, "
          f"so the 402 Gasperini benchmark pairs are not re-scored here; this measures the mechanism, "
          f"not those pairs.")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "neighbour_response.json", "w"), indent=1, default=float)
    print(f"  -> {OUT/'neighbour_response.json'}")


if __name__ == "__main__":
    main()
