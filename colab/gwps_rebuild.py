"""REBUILD THE READOUT ON THE GENOME-WIDE SCREEN: full profiles, ~4x the perturbations, and a real replicate.

WHAT THE CURRENT READOUT ACTUALLY IS, MEASURED RATHER THAN ASSUMED. `nlz_K562.pkl` holds 1,400 knockouts, and for
each one only its TOP 250 GENES -- 3.0% of the transcriptome. Everything else is stored as exactly zero, so the
matrix every result in this project was computed on is truncated by construction, not sparse by biology.

That truncation has one consequence that invalidates a published conclusion. `recall_depth.py` reports "not one
knockout has 300 specific movers, the maximum is 246", and concludes recall@300 and @500 are unanswerable as asked.
The maximum is 246 because the file stores 250 genes. Nine knockouts have EVERY stored gene clearing the threshold
and ten have >=240 -- they are pressed against the cap. A knockout with 400 genuine movers would still read <=250.
The cap does not distort the median (1,186 of 1,400 knockouts have fewer than 25 movers, nowhere near it), so the
saturation argument itself stands; what does not stand is the claim that the ceiling is biological.

WHAT THIS BUILD CHANGES

  full profiles      8,248 genes per perturbation instead of a top-250 slice, so mover counts have no artificial
                     ceiling and recall at depth becomes answerable as originally asked
  more perturbations 11,258 rows over 9,867 distinct genes. NOT all usable: Replogle's own `energy_test_p_value`
                     asks whether a perturbation produced ANY detectable transcriptional change, and 5,401 rows
                     (5,192 genes) clear p<0.05. So the honest gain over the current 1,400 is ~3.9x, not the 8x a
                     raw row count suggests. The rest would add rows that are mostly noise, and are excluded here
                     rather than counted.
  a built-in replicate  RETRACTED -- the premise was wrong, and the measurement is reported as unavailable
                     rather than as a number. See "the replicate that isn't" below.

WHY THIS IS THE HIGHEST-VALUE ITEM OPEN. Every method comparison this project ran came back "indistinguishable" --
the sealed cohorts score 39 and 37 knockouts and every confidence interval straddles zero. That is a power problem,
not a modelling problem. It is also why this rebuild INVALIDATES rather than extends the existing comparison
numbers: the incumbent's 0.0786/0.0685 was measured against a 1,400-gene universe and a truncated readout, so every
method has to be re-scored to get a comparable figure. Doing that is the point.

THE REPLICATE THAT ISN'T -- A RETRACTED CLAIM, KEPT VISIBLE. This file was built partly to measure a
reproducibility ceiling from genes targeted by two independent protospacer sets. That measurement cannot be made
here, and the reason is structural rather than statistical:

    P1P2   8,866 rows   the two guides are ALREADY POOLED into one measurement -- not a replicate
    P1/P2  1,487 rows   genuinely separate, covering 714 genes
    non-targeting 585   negative controls

The 714 split-guide genes are the WEAK tail. Median -log10(energy p) is 1.10 for P1 and 0.84 for P2 against 1.40
for the pooled rows, with fewer cells each. Same-gene profile correlation is +0.0047 against a population-MATCHED
null of +0.0038 -- separation +0.0009 -- and stratifying by phenotype strength shows no trend
(-0.0033, -0.0005, +0.0094, +0.0016, -0.0052 across five strata). The energy p-value is also floored at 1e-4 for
every group, so there is no dynamic range at the strong end to stratify into.

The honest reading is NOT "the screen does not agree with itself". It is that the only genes with two independent
measurements are the ones with nothing to reproduce: every strongly-responding perturbation was pooled into a
single P1P2 row and has no separate replicate at all. The ceiling remains unmeasured.

Two errors were made and corrected while establishing this, both of which would have produced a dramatic false
claim. The null was first drawn from the FILTERED strong perturbations while the same-gene pairs came from the
unfiltered split rows; strong perturbations share the generic stress response, so the null scored HIGHER than the
replicates. And the replicate pairs were first required to BOTH clear the energy test, which cut them to 90 genes
and conditioned the sample on the two halves agreeing.

STORAGE. A dict-of-dicts over 5,192 x 8,248 would be tens of millions of Python floats. The matrix is written as a
compressed npz (kos, genes, M) and `ko_predict.load()` prefers it when present, falling back to the old pickle.
"""
import collections
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

OUT = Path("outputs/orphan")
SP = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
GWPS = SP / "gwps.h5ad"
DEST = SP / "nlz_K562_gwps.npz"

PCUT = 0.05        # Replogle's energy test: did this perturbation change anything at all
MINCELLS = 25      # a perturbation measured in a handful of cells is noise; q10 of the screen is 65 cells
TAU, TIDE_FRAC = 1.0, 0.05     # unchanged from the existing harness, so the comparison is like-for-like


def load_gwps():
    f = h5py.File(GWPS, "r")
    obs = f["obs"]
    var = f["var"]
    oi = [x.decode() for x in obs[obs.attrs.get("_index", "_index")][:]]
    # "0_A1BG_P1_ENSG00000121410" -> gene A1BG, protospacer set P1
    sym = [s.split("_")[1] for s in oi]
    pset = [s.split("_")[2] if len(s.split("_")) > 2 else "" for s in oi]
    gn = var["gene_name"]
    if isinstance(gn, h5py.Group):          # categorical encoding
        cats = [x.decode() for x in gn["categories"][:]]
        codes = gn["codes"][:]
        vg = [cats[c] if 0 <= c < len(cats) else "" for c in codes]
    else:
        raw = gn[:]
        if raw.dtype.kind in "iu" and "__categories" in var:
            cats = [x.decode() for x in var["__categories"]["gene_name"][:]]
            vg = [cats[c] if 0 <= c < len(cats) else "" for c in raw]
        else:
            vg = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    X = f["X"][:]
    # NON-FINITE VALUES IN THE SOURCE MATRIX. The shipped X contains infinities -- 3,437 of them across 1,157 of
    # 5,120 perturbations and 73 genes, almost certainly a divide-by-near-zero in the upstream normalisation.
    # They are NOT harmless: the mover test is |z| >= 1, and inf passes it, so 10.4% of all specific-mover calls
    # and 10.7% of cohort 1's truth entries were infinities being scored as real biology. Set to zero (i.e. "no
    # measured change") rather than dropped, so the gene and perturbation are kept and only the unusable cell is
    # neutralised. Counted and printed, because a silent sanitisation is how the original problem got in.
    nbad = int((~np.isfinite(X)).sum())
    if nbad:
        rows = int((~np.isfinite(X)).any(1).sum())
        print(f"  NON-FINITE in source X: {nbad:,} cells across {rows:,} rows -- set to 0 (unmeasurable, "
              f"not zero-effect); they would otherwise pass the |z|>=1 mover test")
        X = np.where(np.isfinite(X), X, 0.0)
    p = obs["energy_test_p_value"][:]
    nc = obs["num_cells_filtered"][:]
    return X, np.array(sym), np.array(pset), np.array(vg), p, nc


def main():
    X, sym, pset, vg, p, nc = load_gwps()
    print(f"gwps: X {X.shape}, {len(set(sym)):,} distinct perturbed genes, {len(set(vg)):,} readout genes")

    # NON-TARGETING CONTROLS ARE NOT A KNOCKOUT. 585 rows are labelled non-targeting, and 82 of them CLEAR the
    # energy test at p<0.05 -- so an unfiltered build ingests them, collapses them into a single row under the
    # "gene" name non-targeting, and scores predictions against a negative control. Excluded here, and their pass
    # rate is reported instead, because it is a free calibration of the test: 18.5% of true nulls clear p<0.05,
    # which is the false-positive rate the perturbation filter actually operates at.
    isctrl = pset == "non-targeting"
    ctrl_fp = float((np.isfinite(p[isctrl]) & (p[isctrl] < PCUT)).mean()) if isctrl.any() else float("nan")
    print(f"\nnon-targeting controls: {int(isctrl.sum()):,} rows; {ctrl_fp:.1%} clear p<{PCUT} "
          f"-- the energy test's empirical false-positive rate, and the reason they are excluded")
    keep = np.isfinite(p) & (p < PCUT) & (nc >= MINCELLS) & ~isctrl
    print(f"\nquality filter (energy test p<{PCUT}, >={MINCELLS} cells, controls dropped)")
    print(f"  rows kept {keep.sum():,}/{len(keep):,}; distinct genes {len(set(sym[keep])):,}")
    print(f"  dropped for no detectable phenotype : {int((np.isfinite(p) & (p >= PCUT)).sum()):,}")
    print(f"  dropped for too few cells           : {int((nc < MINCELLS).sum()):,}")

    # ---- the built-in replicate: genes whose two protospacer sets were measured SEPARATELY ----
    # Most rows are labelled P1P2, i.e. both guides already pooled into one measurement -- those are not
    # replicates. Only the 1,487 rows split into P1 and P2 are, covering 738 genes.
    #
    # The pairs are taken PRE-FILTER, deliberately. Requiring both members to clear the energy test leaves only 90
    # genes AND conditions the sample on agreement: a gene is far more likely to have both halves significant when
    # the two halves agree, so the surviving pairs would overstate reproducibility. The first version of this
    # check made that mistake and reported a median correlation of +0.0017 over n=62 -- below its own
    # different-gene control, which is the signature of a broken comparison rather than a finding.
    split = np.isin(pset, ["P1", "P2"]) & (nc >= MINCELLS) & ~isctrl
    pairsym = collections.defaultdict(list)
    for i in np.where(split)[0]:
        pairsym[sym[i]].append(i)
    rep = {g: idx for g, idx in pairsym.items() if len(idx) > 1}
    print(f"\nREPLICATE CHECK: {len(rep):,} genes have their two protospacer sets measured separately "
          f"(pre-filter, so the sample is not conditioned on the two halves agreeing)")

    bysym = collections.defaultdict(list)
    for i in np.where(keep)[0]:
        bysym[sym[i]].append(i)
    if rep:
        cors, jac = [], []
        rng = np.random.default_rng(0)
        pick = sorted(rep)
        for g in pick:
            a, b = rep[g][0], rep[g][1]
            x, y = X[a], X[b]
            if np.isfinite(x).all() and np.isfinite(y).all() and x.std() > 0 and y.std() > 0:
                cors.append(float(np.corrcoef(x, y)[0, 1]))
            sa, sb = set(np.where(np.abs(x) >= TAU)[0]), set(np.where(np.abs(y) >= TAU)[0])
            if sa or sb:
                jac.append(len(sa & sb) / len(sa | sb))
        # THE CONTROL MUST COME FROM THE SAME POPULATION AS THE COMPARISON. The first version drew its null from
        # the FILTERED rows (perturbations that cleared the energy test) while drawing same-gene pairs from the
        # unfiltered P1/P2 rows. Strong perturbations share the generic stress response and so correlate with each
        # other, which made the null score HIGHER than the same-gene pairs -- and would have supported the dramatic
        # and false claim that the screen does not agree with itself. The null here pairs a P1 row of one gene with
        # a P2 row of a DIFFERENT gene, drawn from exactly the same rows the replicates come from.
        p1 = [rep[g][0] for g in pick]
        p2 = [rep[g][1] for g in pick]
        null = []
        for t, a in enumerate(p1):
            b = p2[(t + 1 + rng.integers(0, max(len(p2) - 1, 1))) % len(p2)]   # a different gene's second guide
            x, y = X[a], X[b]
            if np.isfinite(x).all() and np.isfinite(y).all() and x.std() > 0 and y.std() > 0:
                null.append(float(np.corrcoef(x, y)[0, 1]))
        c = np.array(cors); n_ = np.array(null); j = np.array(jac)
        print(f"  same-gene, two guide sets     : median r {np.median(c):+.4f}  (n={len(c):,})")
        print(f"  DIFFERENT genes, same rows    : median r {np.median(n_):+.4f}  (n={len(n_):,})  <- matched null")
        print(f"  separation                    : {np.median(c) - np.median(n_):+.4f}")
        print(f"  same-gene mover-set Jaccard   : median {np.median(j):.4f}")
        print("  -> NOT a usable reproducibility ceiling. See the retraction in this file's docstring.")

    # ---- collapse replicates: cell-count-weighted mean, which is the right weighting for a mean of means ----
    genes = sorted({g for g in vg if g})
    gi = {g: i for i, g in enumerate(genes)}
    cols = np.array([gi.get(g, -1) for g in vg])
    ok = cols >= 0
    # TWO SYMBOLS OCCUPY TWO VAR COLUMNS EACH (TBCE, HSPA14): 8,248 columns carry 8,246 distinct names. A plain
    # fancy-index assignment M[r, cols] = prof lets the LAST duplicate win and silently discards the other column.
    # Averaging them is the defensible collapse, and the count vector below does it in one pass.
    ncol = np.zeros(len(genes), np.float32)
    np.add.at(ncol, cols[ok], 1.0)
    ncol[ncol == 0] = 1.0
    dupes = sorted({g for g, n_ in zip(genes, ncol) if n_ > 1})
    if dupes:
        print(f"  {len(dupes)} symbol(s) span multiple var columns and are averaged, not overwritten: "
              f"{', '.join(dupes[:6])}")
    kos = sorted(bysym)
    M = np.zeros((len(kos), len(genes)), np.float32)
    for r, g in enumerate(kos):
        idx = bysym[g]
        w = nc[idx].astype(np.float64)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
        prof = (X[idx] * w[:, None]).sum(0)
        row = np.zeros(len(genes), np.float32)
        np.add.at(row, cols[ok], prof[ok])
        M[r] = row / ncol
    print(f"\nmatrix {M.shape[0]:,} perturbations x {M.shape[1]:,} genes "
          f"(previous readout: 1,400 x 7,223, truncated to the top 250 per knockout)")

    A = np.abs(M)
    tide = (A >= TAU).mean(0) >= TIDE_FRAC
    spec = ((A >= TAU) & ~tide).sum(1)
    print(f"\nTIDE: {int(tide.sum()):,} genes move in >={TIDE_FRAC:.0%} of perturbations "
          f"(old readout: 43 genes, all ribosomal + mtDNA-encoded OXPHOS)")
    print(f"specific movers per perturbation: median {int(np.median(spec))}, p90 {int(np.percentile(spec,90))}, "
          f"p99 {int(np.percentile(spec,99))}, MAX {int(spec.max())}")
    print(f"  perturbations with >=300 specific movers: {(spec >= 300).sum():,}  "
          f"(old readout: 0, but its storage capped every profile at 250)")
    print(f"  perturbations with >=500                : {(spec >= 500).sum():,}")
    print(f"  scorable (>=5 specific movers)          : {(spec >= 5).sum():,}  (old readout: 378)")

    np.savez_compressed(DEST, kos=np.array(kos), genes=np.array(genes), M=M)
    print(f"\n  -> {DEST} ({DEST.stat().st_size/1e6:.1f} MB)")
    json.dump({"n_perturbations": len(kos), "n_genes": len(genes), "control_fp_rate": ctrl_fp,
               "n_controls_dropped": int(isctrl.sum()),
               "n_rows_total": int(len(keep)), "n_rows_kept": int(keep.sum()),
               "n_tide": int(tide.sum()), "n_scorable": int((spec >= 5).sum()),
               "max_specific": int(spec.max()), "n_ge300": int((spec >= 300).sum()),
               "n_ge500": int((spec >= 500).sum()),
               "replicate_genes": len(rep),
               "replicate_median_corr": float(np.median(cors)) if rep and cors else None,
               "control_median_corr": float(np.median(null)) if rep and null else None,
               "replicate_median_jaccard": float(np.median(jac)) if rep and jac else None},
              open(OUT / "gwps_rebuild.json", "w"))
    print(f"  -> {OUT/'gwps_rebuild.json'}")


if __name__ == "__main__":
    main()
