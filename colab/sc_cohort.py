"""STEP 1 of the single-cell programme: choose the 100-200 perturbation cohort, on measured criteria not vibes.

WHY A COHORT AT ALL.  The full single-cell resource is ~656,000 cells over 2,394 perturbations in three lines.
Training anything on all of it, on 4 CPU cores with 4 GB of free disk, is not possible tonight. A cohort chosen
on explicit criteria is not a shortcut -- it is the only way to get a measured answer before committing compute
that does not exist here.

THE FIVE CRITERIA, each turned into a number rather than a judgement call:

    cells        cells present in ALL THREE lines, so every cohort member supports a cross-line comparison.
    guides       distinct guide constructs. Reported honestly: this library is ~1.1 constructs per gene, so
                 "multiple guides" is mostly NOT available and the criterion cannot be met by selection. Saying
                 so beats quietly dropping it.
    function     spread across functional classes, using the k-means partition of the chan2a channel matrix that
                 the LOCO test already showed is the axis the model lives or dies on. Round-robin over classes,
                 so no class dominates the cohort.
    heterogeneity within-perturbation dispersion on high-fano genes MINUS the dispersion of non-targeting cells
                 in the same batches. Raw dispersion would just rank by depth and cell count; the control
                 subtraction is what makes it "response heterogeneity" rather than "noise".
    ntc_batch    fraction of the perturbation's cells sitting in batches that also contain non-targeting cells.
                 Without this a perturbation cannot be compared to a control at all, so it is a hard filter,
                 not a score.

OUTPUT: a ranked cohort with every criterion recorded per perturbation, so the selection can be audited and the
scale-up to 1,000 reuses the same instrument rather than a new set of choices.

NOTE ON HVGs: var already carries precomputed `fano`, `cv` and `mean` per gene, so the dispersion genes are
chosen without reading the expression matrix at all.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C

OUT, SP = A.OUT, A.SP
LINES = {"RPE1": "rpe1.h5ad", "Jurkat": "nadig_jurkat.h5ad", "HepG2": "nadig_hepg2.h5ad"}
import os
N_HVG, MIN_CELLS, BLOCK = 400, 40, 8192
TARGET = int(os.environ.get('SC_TARGET', '200'))
NTC_NAMES = ("non-targeting", "control", "nt", "ntc")


def _dec(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 1 -- COHORT SELECTION on measured criteria")
    report("=" * 100)

    per_line = {}
    for line, fn in LINES.items():
        h = h5py.File(SP / fn, "r")
        gn = _dec(h["var"]["gene_name"][:])
        fano = np.asarray(h["var"]["fano"][:], np.float64)
        mean = np.asarray(h["var"]["mean"][:], np.float64)
        ok = (mean > 0.05) & np.isfinite(fano)
        hvg = np.argsort(-np.where(ok, fano, -np.inf))[:N_HVG]
        hvg.sort()
        pc = h["obs"]["gene"]
        cats = _dec(pc["categories"][:])
        codes = pc["codes"][:]
        gid = h["obs"]["guide_id"]
        gcodes = gid["codes"][:]
        batch = np.asarray(h["obs"]["batch"][:], np.int64)
        umi = np.asarray(h["obs"]["UMI_count"][:], np.float64)
        ntc = {i for i, c in enumerate(cats) if c.strip().lower() in NTC_NAMES}
        if not ntc:
            ntc = {i for i, c in enumerate(cats) if "non-target" in c.lower() or "control" in c.lower()}
        is_ntc = np.isin(codes, list(ntc))
        ntc_batches = set(np.unique(batch[is_ntc]).tolist())
        report(f"\n  {line}: {len(codes):,} cells, {len(cats)} labels, NTC={[cats[i] for i in ntc][:2]} "
               f"({int(is_ntc.sum()):,} cells) in {len(ntc_batches)} batches")

        # per-cell HVG expression, CPM-normalised, read once in blocks
        E = np.zeros((len(codes), N_HVG), np.float32)
        for i0 in range(0, len(codes), BLOCK):
            blk = h["X"][i0:i0 + BLOCK, :]
            E[i0:i0 + blk.shape[0]] = (blk[:, hvg] / np.maximum(umi[i0:i0 + blk.shape[0], None], 1) * 1e4)
        h.close()
        E = np.log1p(E)

        # control dispersion per batch, so heterogeneity is measured against matched noise
        ctrl_disp = {}
        for b in ntc_batches:
            m = is_ntc & (batch == b)
            if m.sum() >= 20:
                ctrl_disp[b] = float(np.mean(E[m].std(0)))
        base_disp = float(np.mean(list(ctrl_disp.values()))) if ctrl_disp else float(np.mean(E[is_ntc].std(0)))

        rec = {}
        for gi, gname in enumerate(cats):
            if gi in ntc:
                continue
            cells = np.where(codes == gi)[0]
            if len(cells) < MIN_CELLS:
                continue
            b = batch[cells]
            frac_ntc_batch = float(np.isin(b, list(ntc_batches)).mean())
            disp = float(np.mean(E[cells].std(0)))
            exp_ctrl = float(np.mean([ctrl_disp.get(x, base_disp) for x in b])) if ctrl_disp else base_disp
            rec[gname] = {"cells": int(len(cells)), "guides": int(len(np.unique(gcodes[cells]))),
                          "batches": int(len(np.unique(b))), "frac_ntc_batch": frac_ntc_batch,
                          "disp": disp, "disp_excess": disp - exp_ctrl}
        per_line[line] = rec
        report(f"    {len(rec):,} perturbations with >= {MIN_CELLS} cells; "
               f"median cells {int(np.median([r['cells'] for r in rec.values()]))}; "
               f"guides/gene median {int(np.median([r['guides'] for r in rec.values()]))}; "
               f"control dispersion {base_disp:.4f}")

    shared = set.intersection(*[set(r) for r in per_line.values()])
    report(f"\n  perturbations passing in ALL THREE lines: {len(shared):,}")

    # functional classes from the channel matrix -- the axis LOCO showed the model lives on
    z = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in z["kos"]]
    universe = sorted(set(kos) | shared)
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    from sklearn.cluster import KMeans
    cand = sorted(shared)
    km = KMeans(n_clusters=20, n_init=4, random_state=0).fit(chan[[urow[g] for g in cand]])
    lab = km.labels_

    rows = []
    for i, g in enumerate(cand):
        r = {k: per_line[k][g] for k in LINES}
        rows.append({"gene": g, "cls": int(lab[i]),
                     "min_cells": min(r[k]["cells"] for k in LINES),
                     "tot_cells": sum(r[k]["cells"] for k in LINES),
                     "max_guides": max(r[k]["guides"] for k in LINES),
                     "min_frac_ntc": min(r[k]["frac_ntc_batch"] for k in LINES),
                     "mean_disp_excess": float(np.mean([r[k]["disp_excess"] for k in LINES])),
                     "per_line": r})
    hard = [r for r in rows if r["min_frac_ntc"] >= 0.5]
    report(f"  after the NTC-batch filter (>=50% of cells in NTC-containing batches): {len(hard):,}")
    gmax = max(r["max_guides"] for r in rows)
    report(f"  guides per gene: max observed {gmax} -- 'multiple guides' is {'AVAILABLE' if gmax > 1 else 'NOT AVAILABLE'} "
           f"in this library; the criterion cannot be met by selection and is reported, not silently dropped")

    # round-robin over functional classes, ranked within class by heterogeneity x cells
    by_cls = {}
    for r in hard:
        by_cls.setdefault(r["cls"], []).append(r)
    for c in by_cls:
        by_cls[c].sort(key=lambda r: -(r["mean_disp_excess"] * np.log10(max(r["min_cells"], 2))))
    cohort, ci = [], 0
    while len(cohort) < TARGET and any(by_cls.values()):
        for c in sorted(by_cls):
            if by_cls[c] and len(cohort) < TARGET:
                cohort.append(by_cls[c].pop(0))
        ci += 1
        if ci > TARGET:
            break

    report(f"\n  COHORT: {len(cohort)} perturbations over {len({r['cls'] for r in cohort})} functional classes")
    report(f"    total cells across 3 lines: {sum(r['tot_cells'] for r in cohort):,}")
    report(f"    min cells per line: median {int(np.median([r['min_cells'] for r in cohort]))}, "
           f"range {min(r['min_cells'] for r in cohort)}-{max(r['min_cells'] for r in cohort)}")
    report(f"    heterogeneity excess: median {np.median([r['mean_disp_excess'] for r in cohort]):+.4f}, "
           f"{sum(1 for r in cohort if r['mean_disp_excess'] > 0)} of {len(cohort)} above their control")
    report(f"    first 12: {', '.join(r['gene'] for r in cohort[:12])}")

    json.dump({"test": "sc_cohort", "target": TARGET, "n_hvg": N_HVG, "min_cells": MIN_CELLS,
               "n_shared": len(shared), "n_after_ntc_filter": len(hard), "max_guides_observed": int(gmax),
               "cohort": [{k: v for k, v in r.items() if k != "per_line"} for r in cohort],
               "cohort_detail": {r["gene"]: r["per_line"] for r in cohort},
               "log": log}, open(OUT / (f"sc_cohort{TARGET}.json" if TARGET != 200 else "sc_cohort.json"), "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_cohort.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
