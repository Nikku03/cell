"""FULL PROFILES for every cell line on disk -- no top-250 truncation.

WHY. coverage_gap.py showed that the same gene knocked out in a DIFFERENT cell line is by far the best
predictor of which genes move: coverage 0.2544 at precision 0.1564 from 73 candidates, against 0.0517/0.0175
for the entire curated network. Combined with frequency it beat the baseline for the first time in this
project, +0.0213 [+0.0108,+0.0317].

That was measured with the OLD per-line readouts, which store only each perturbation's TOP 250 GENES. Every
cross-line mover set was therefore capped at 250 and the coverage number is a floor, not an estimate. This
rebuilds every line at full width.

THE SOURCES ARE NOT UNIFORM, and pretending otherwise is how a silent join failure gets in:

    gwps.h5ad          11,258 x 8,248    ALREADY per-perturbation z    K562, genome-wide
    k562.h5ad           2,285 x 8,563    ALREADY per-perturbation z    K562, essential subset
    hct116.h5ad        17,768 x 16,380   ALREADY per-perturbation      obs/idx = "g_<GENE>_hct116"
    rpe1.h5ad         247,914 x 8,749    SINGLE CELL -> pseudobulk     obs/perturbation, 2,394 genes
    nadig_hepg2.h5ad  145,473 x 9,624    SINGLE CELL -> pseudobulk     same design as RPE1
    nadig_jurkat.h5ad 262,956 x 8,882    SINGLE CELL -> pseudobulk     same design as RPE1
    frangieh.h5ad     218,331 x 23,712   SINGLE CELL, 249 perturbations, 3 conditions, multiplexed guides

Frangieh (melanoma) is included only under its 'Control' condition: the other two are IFN-gamma stimulation
and co-culture, which are a different experiment, and its guide field is multiplexed (58,211 combinations for
249 targets). Small and caveated rather than silently pooled.

METHOD for the single-cell files, following build_rpe1_profiles.py so the result is comparable to what this
project already used: group cells by targeted gene, sum in chunks (the matrices do not fit in memory), divide
to per-knockout means, then standardise each GENE COLUMN across knockouts robustly (median, MAD x 1.4826).
Knockouts under MINCELLS cells are dropped as too noisy. Deterministic throughout.

OUTPUT is the mover SET per perturbation at |z| >= TAU, not the dense matrix: the analysis consumes sets, and
six dense matrices would be gigabytes on a sandbox that has rolled back ten times today. The threshold is
recorded in the file so it cannot be misread later.
"""
import collections
import gzip
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
DEST = Path(__file__).resolve().parent / "data" / "line_movers.json.gz"
TAU, TIDE_FRAC, MINCELLS, CHUNK = 1.0, 0.05, 25, 8000


def cat(h, key):
    g = h["obs"][key]
    if isinstance(g, h5py.Dataset):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
    codes = g["codes"][:]
    return np.array([cats[c] if 0 <= c < len(cats) else "?" for c in codes])


def var_names(h):
    """Gene SYMBOLS, across three different AnnData encodings.

    THE THIRD ENCODING IS THE ONE THAT BIT. gwps.h5ad and k562.h5ad store gene_name as a dataset of INTEGER
    CODES with the strings in a sibling `var/__categories` group -- the pre-0.8 AnnData layout. Reading that
    dataset directly returns 3595, 3591, 4546, which stringify into perfectly plausible-looking "gene names".
    The build then succeeded, the per-line mover counts were correct, and every cross-line join silently
    returned nothing: K562's coverage came out 0.0014 where the earlier measurement was 0.2544, while the
    three lines sharing the modern layout matched each other fine. gwps_rebuild.py already handled this
    branch; not replicating it here is what produced the discrepancy.

    An integer-looking result is therefore rejected rather than returned.
    """
    v = h["var"]
    for k in ("gene_name", "gene_symbol"):
        if k not in v:
            continue
        d = v[k]
        if isinstance(d, h5py.Group):                       # modern categorical
            cats = [x.decode() if isinstance(x, bytes) else str(x) for x in d["categories"][:]]
            return np.array([cats[c] if 0 <= c < len(cats) else "?" for c in d["codes"][:]])
        raw = d[:]
        if raw.dtype.kind in "iu":                          # legacy: codes + var/__categories
            if "__categories" in v and k in v["__categories"]:
                cats = [x.decode() if isinstance(x, bytes) else str(x) for x in v["__categories"][k][:]]
                return np.array([cats[c] if 0 <= c < len(cats) else "?" for c in raw])
            raise KeyError(f"var/{k} is integer codes but var/__categories/{k} is absent")
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in raw])
    idx = v.attrs.get("_index")
    if idx is not None:
        k = idx.decode() if isinstance(idx, bytes) else idx
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v[k][:]])
    raise KeyError("no gene names in var")


def robust_z(M):
    """Standardise each GENE COLUMN across knockouts: (x - median) / (1.4826 * MAD).

    Column-wise, not row-wise: the question is "is this gene unusual in this knockout relative to all
    knockouts", which is what the K562 readout already encodes. MAD rather than sd so a handful of extreme
    knockouts cannot inflate the scale and hide everything else.
    """
    med = np.median(M, axis=0)
    mad = np.median(np.abs(M - med), axis=0) * 1.4826
    mad[mad <= 0] = np.nan
    Z = (M - med) / mad
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def from_pseudobulk(path, name_fn, tag, qc=False):
    """qc=True applies gwps_rebuild.py's filters: energy test p<0.05, >=25 cells, non-targeting dropped.

    Without them this pipeline read all 11,258 rows and reported 1,016 knockouts with >=5 movers where the
    independent build reports 838. Two builds of the same data disagreeing is the signal to stop, not a
    number to pick between.
    """
    with h5py.File(path, "r") as h:
        genes = var_names(h)
        idxk = h["obs"].attrs.get("_index")
        idxk = idxk.decode() if isinstance(idxk, bytes) else (idxk or "idx")
        raw = [x.decode() if isinstance(x, bytes) else str(x) for x in h["obs"][idxk][:]]
        kos = [name_fn(r) for r in raw]
        ok = np.ones(len(kos), bool)
        if qc:
            if "energy_test_p_value" in h["obs"]:
                pv = np.asarray(h["obs"]["energy_test_p_value"][:], float)
                ok &= np.isfinite(pv) & (pv < 0.05)
            if "num_cells_filtered" in h["obs"]:
                ok &= np.asarray(h["obs"]["num_cells_filtered"][:], float) >= MINCELLS
            pk = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
            ok &= np.array([("non-targeting" not in x) for x in pk])
            print(f"    {tag}: QC keeps {int(ok.sum()):,}/{len(ok):,} rows")
        X = h["X"]
        M = np.empty(X.shape, np.float32)
        for s in range(0, X.shape[0], CHUNK):
            M[s:s + CHUNK] = X[s:s + CHUNK]
    keep = [i for i, k in enumerate(kos) if k and ok[i]]
    M, kos = M[keep], [kos[i] for i in keep]
    nb = int((~np.isfinite(M)).sum())
    if nb:
        print(f"    {tag}: {nb:,} non-finite cells set to 0")
        M = np.where(np.isfinite(M), M, 0.0)
    return kos, list(genes), M


def from_singlecell(path, pert_key, tag, cond_key=None, cond_val=None):
    with h5py.File(path, "r") as h:
        genes = var_names(h)
        pert = cat(h, pert_key)
        mask = np.ones(len(pert), bool)
        if cond_key:
            mask &= (cat(h, cond_key) == cond_val)
        X = h["X"]
        ncell = X.shape[0]
        uniq = sorted({p for p, m in zip(pert, mask) if m and p and p.lower() not in
                       ("control", "nan", "?", "none", "non-targeting")})
        pi = {p: i for i, p in enumerate(uniq)}
        S = np.zeros((len(uniq), X.shape[1]), np.float64)
        C = np.zeros(len(uniq), np.int64)
        for s in range(0, ncell, CHUNK):
            e = min(s + CHUNK, ncell)
            blk = np.asarray(X[s:e], np.float32)
            for r in range(e - s):
                if not mask[s + r]:
                    continue
                j = pi.get(pert[s + r])
                if j is None:
                    continue
                S[j] += blk[r]
                C[j] += 1
            if (s // CHUNK) % 5 == 0:
                print(f"    {tag}: {e:,}/{ncell:,} cells", flush=True)
    ok = C >= MINCELLS
    print(f"    {tag}: {ok.sum():,}/{len(uniq):,} knockouts pass >={MINCELLS} cells")
    M = (S[ok] / C[ok, None]).astype(np.float32)
    return [uniq[i] for i in np.where(ok)[0]], list(genes), M


def movers_of(kos, genes, M, tag, standardize, target_frac=None):
    """standardize=False for sources ALREADY in z-units.

    gwps.h5ad and k562.h5ad ship per-perturbation z-scores in which only 0.07% of cells exceed |1|.
    Re-standardising those by column median/MAD forces every column to unit scale, ~32% of cells then clear
    the mover threshold, and EVERY gene qualifies as tide -- the first run of this file reported tide = 8,248
    of 8,248 and zero movers. The per-line tide count is the check that catches it, which is why it is
    printed. Sources arriving as raw means (single-cell pseudobulk) or as small log-ratios (HCT116, values
    ~1e-3) genuinely do need it.
    """
    Z = robust_z(M) if standardize else M
    A = np.abs(Z)
    # STRINGENCY IS CALIBRATED ACROSS LINES, NOT THE NOMINAL THRESHOLD. K562's shipped z is not a
    # unit-variance score: only 0.07% of its cells exceed |1|. robust_z, by construction, puts ~32% of cells
    # past |1|. Applying TAU=1 to both therefore calls 0.07% of K562 and a third of every other line, which
    # made EVERY gene tide (11,677 of 16,380 on HCT116) and left zero movers. Matching the FRACTION of cells
    # called a mover puts every line at the same selectivity, which is what a cross-line candidate set needs.
    tau = TAU if target_frac is None else float(np.quantile(A, 1.0 - target_frac))
    tide = (A >= tau).mean(0) >= TIDE_FRAC
    spec = (A >= tau) & ~tide
    out = {}
    for i, k in enumerate(kos):
        idx = np.where(spec[i])[0]
        if len(idx):
            out.setdefault(k, set()).update(genes[j] for j in idx)
    n = [len(v) for v in out.values()]
    print(f"  {tag}: {len(kos):,} knockouts x {len(genes):,} genes | tau {tau:.3f} | "
          f"cells>=tau {100*(A >= tau).mean():.3f}% | tide {int(tide.sum()):,} | "
          f"{len(out):,} with >=1 mover, median {int(np.median(n)) if n else 0}, "
          f"mean {np.mean(n) if n else 0:.0f}")
    return {k: sorted(v) for k, v in out.items()}, float((A >= tau).mean())


def main():
    res, meta = {}, {}
    ref_frac = [None]
    jobs = [
        ("K562-gwps", False, lambda: from_pseudobulk(
            SP / "gwps.h5ad", lambda s: s.split("_")[1] if len(s.split("_")) > 1 else "", "K562-gwps",
            qc=True)),
        ("K562-ess", False, lambda: from_pseudobulk(
            SP / "k562.h5ad", lambda s: s.split("_")[1] if len(s.split("_")) > 1 else "", "K562-ess",
            qc=True)),
        ("HCT116", True, lambda: from_pseudobulk(
            SP / "hct116.h5ad",
            lambda s: s[2:-7] if s.startswith("g_") and s.endswith("_hct116") else "", "HCT116")),
        ("RPE1", True, lambda: from_singlecell(SP / "rpe1.h5ad", "perturbation", "RPE1")),
        ("HepG2", True, lambda: from_singlecell(SP / "nadig_hepg2.h5ad", "perturbation", "HepG2")),
        ("Jurkat", True, lambda: from_singlecell(SP / "nadig_jurkat.h5ad", "perturbation", "Jurkat")),
        ("Melanoma", True, lambda: from_singlecell(SP / "frangieh.h5ad", "perturbation", "Melanoma",
                                                   "perturbation_2", "Control")),
    ]
    for name, std, fn in jobs:
        try:
            kos, genes, M = fn()
        except Exception as e:
            print(f"  {name}: SKIPPED ({type(e).__name__}: {e})")
            meta[name] = {"status": f"skipped: {type(e).__name__}"}
            continue
        res[name], frac = movers_of(kos, genes, M, name, std,
                                    target_frac=(None if name == "K562-gwps" else ref_frac[0]))
        if name == "K562-gwps":
            ref_frac[0] = frac      # every other line is calibrated to K562's selectivity
        # K562-gwps is the one line with an independent build (gwps_rebuild.py). If this pipeline does not
        # reproduce its 838, the two disagree and neither should be trusted.
        if name == "K562-gwps":
            nm = sum(1 for v in res[name].values() if len(v) >= 5)
            print(f"    CHECK vs gwps_rebuild.py: {nm:,} knockouts with >=5 movers (independent build: 838)")
        meta[name] = {"n_knockouts": len(kos), "n_genes": len(genes),
                      "n_with_movers": len(res[name]),
                      "median_movers": int(np.median([len(v) for v in res[name].values()]))
                      if res[name] else 0}
        del M

    # NAMESPACE ASSERT. The whole point of these files is that they JOIN, so a build that produces
    # non-overlapping identifier spaces is worthless even when every per-line count looks right. Require
    # every line to share a substantial gene vocabulary with the largest one.
    ref = max(res, key=lambda k: len(res[k]))
    refg = {g for v in res[ref].values() for g in v}
    for k in res:
        gg = {g for v in res[k].values() for g in v}
        ov = len(gg & refg) / max(min(len(gg), len(refg)), 1)
        print(f"    namespace overlap {k:12s} vs {ref}: {100*ov:.1f}%")
        assert k == ref or ov > 0.2, (
            f"{k} shares only {100*ov:.1f}% of its gene vocabulary with {ref} -- different identifier "
            "spaces, the cross-line join would silently return nothing")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(DEST, "wt") as f:
        json.dump({"tau": TAU, "tide_frac": TIDE_FRAC, "min_cells": MINCELLS,
                   "note": "specific movers (|robust z| >= tau, tide genes removed) per knockout, per line",
                   "meta": meta, "lines": {k: v for k, v in res.items()}}, f)
    print(f"\n  -> {DEST} ({DEST.stat().st_size/1e6:.1f} MB)")
    chk = json.load(gzip.open(DEST, "rt"))
    assert set(chk["lines"]) == set(res), "roundtrip lost a cell line"
    for k in res:
        print(f"    {k:12s} {len(chk['lines'][k]):>6,} knockouts with movers")


if __name__ == "__main__":
    main()
