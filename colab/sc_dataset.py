"""STEP 2: materialise the cohort as a per-cell tensor with every covariate the models are allowed to use.

WHAT GOES IN, and why each field exists rather than "because we have it":

    X          cells x HVG log-CPM expression. The readout.
    pert       which gene was knocked out. The thing to predict from.
    guide      which guide construct -- lets a model separate guide-specific artefacts (seed effects, cutting
               toxicity) from the gene's real biology. The cohort has up to 5 constructs per gene.
    batch      experimental batch. Every comparison must be within-batch or it measures batch.
    line       RPE1 / Jurkat / HepG2. The context axis the cross-line tests said the model has no input for.
    s_score
    g2m_score  cell-cycle position from S and G2/M marker sets, scored separately rather than collapsed, since
               a knockout can arrest in one phase without touching the other.
    umi        sequencing depth. A technical covariate, included so a model can be shown NOT to be using it.
    is_ntc     non-targeting control flag.

CONTROLS ARE CARRIED, NOT SUMMARISED.  Every non-targeting cell in a batch that contains a cohort perturbation is
kept as a cell, not folded into a mean. Pseudobulking the controls would destroy exactly the within-batch,
within-state matching that makes a per-cell comparison worth doing.

GENE SPACE: the top HVGs by mean fano across all three lines, intersected so every line has every gene. A shared
space is what makes cross-line prediction expressible at all; the earlier cross-line test had to rebuild a
different common space per line and the numbers were not comparable between them.

DISK: 4.1 GB free at build time, so the tensor is float32 on ~2,000 genes and the size is asserted before write.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
from adrn_cellstate_gate import S_GENES, G2M_GENES, _dec

OUT, SP = A.OUT, A.SP
LINES = {"RPE1": "rpe1.h5ad", "Jurkat": "nadig_jurkat.h5ad", "HepG2": "nadig_hepg2.h5ad"}
N_HVG, BLOCK, MAX_GB = 2000, 8192, 2.5
NTC_NAMES = ("non-targeting", "control", "nt", "ntc")


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 2 -- build the per-cell cohort tensor")
    report("=" * 100)

    cohort = [r["gene"] for r in json.load(open(OUT / "sc_cohort.json"))["cohort"]]
    cset = set(cohort)
    report(f"  cohort: {len(cohort)} perturbations")

    # shared HVG space across all three lines
    meta = {}
    for line, fn in LINES.items():
        h = h5py.File(SP / fn, "r")
        meta[line] = {"gn": _dec(h["var"]["gene_name"][:]),
                      "fano": np.asarray(h["var"]["fano"][:], np.float64),
                      "mean": np.asarray(h["var"]["mean"][:], np.float64)}
        h.close()
    common = set(meta["RPE1"]["gn"])
    for line in LINES:
        common &= set(meta[line]["gn"])
    score = {}
    for g in common:
        f, m = [], []
        for line in LINES:
            i = meta[line]["gn"].index(g)
            f.append(meta[line]["fano"][i])
            m.append(meta[line]["mean"][i])
        if min(m) > 0.05 and all(np.isfinite(f)):
            score[g] = float(np.mean(f))
    hvg = [g for g, _ in sorted(score.items(), key=lambda kv: -kv[1])[:N_HVG]]
    hvg.sort()
    report(f"  shared genes {len(common):,}; HVG space {len(hvg):,} (mean fano, min mean expr > 0.05)")

    Xs, meta_rows = [], []
    for line, fn in LINES.items():
        h = h5py.File(SP / fn, "r")
        gn = meta[line]["gn"]
        gidx = {g: i for i, g in enumerate(gn)}
        cols = np.array([gidx[g] for g in hvg])
        s_cols = sorted({gidx[g] for g in S_GENES if g in gidx})
        g2_cols = sorted({gidx[g] for g in G2M_GENES if g in gidx})
        pc = h["obs"]["gene"]
        cats = _dec(pc["categories"][:])
        codes = pc["codes"][:]
        gcodes = h["obs"]["guide_id"]["codes"][:]
        gcats = _dec(h["obs"]["guide_id"]["categories"][:])
        batch = np.asarray(h["obs"]["batch"][:], np.int64)
        umi = np.asarray(h["obs"]["UMI_count"][:], np.float64)
        ntc_ids = {i for i, c in enumerate(cats) if c.strip().lower() in NTC_NAMES}
        if not ntc_ids:
            ntc_ids = {i for i, c in enumerate(cats) if "non-target" in c.lower()}
        is_ntc = np.isin(codes, list(ntc_ids))
        want_p = {i for i, c in enumerate(cats) if c in cset}
        sel_p = np.isin(codes, list(want_p))
        keep_batches = set(np.unique(batch[sel_p]).tolist())
        sel = sel_p | (is_ntc & np.isin(batch, list(keep_batches)))
        idx = np.where(sel)[0]
        report(f"  {line}: {len(idx):,} cells kept ({int(sel_p.sum()):,} perturbed, "
               f"{int((is_ntc & np.isin(batch, list(keep_batches))).sum()):,} NTC in shared batches)")

        Xl = np.zeros((len(idx), len(hvg)), np.float32)
        s_sc = np.zeros(len(idx))
        g_sc = np.zeros(len(idx))
        pos = 0
        for i0 in range(0, len(codes), BLOCK):
            i1 = min(i0 + BLOCK, len(codes))
            m = sel[i0:i1]
            if not m.any():
                continue
            blk = h["X"][i0:i1, :][m]
            u = np.maximum(umi[i0:i1][m][:, None], 1)
            Xl[pos:pos + len(blk)] = np.log1p(blk[:, cols] / u * 1e4)
            s_sc[pos:pos + len(blk)] = (blk[:, s_cols] / u * 1e4).sum(1)
            g_sc[pos:pos + len(blk)] = (blk[:, g2_cols] / u * 1e4).sum(1)
            pos += len(blk)
        h.close()
        Xs.append(Xl)
        for j, i in enumerate(idx):
            meta_rows.append((line, cats[codes[i]], gcats[gcodes[i]], int(batch[i]),
                              float(s_sc[j]), float(g_sc[j]), float(umi[i]), bool(is_ntc[i])))

    X = np.concatenate(Xs, 0)
    del Xs
    gb = X.nbytes / 2 ** 30
    report(f"\n  tensor {X.shape} = {gb:.2f} GB")
    assert gb < MAX_GB, f"tensor {gb:.2f} GB exceeds the {MAX_GB} GB budget -- reduce N_HVG"

    line_a = np.array([m[0] for m in meta_rows])
    pert_a = np.array([m[1] for m in meta_rows])
    guide_a = np.array([m[2] for m in meta_rows])
    batch_a = np.array([m[3] for m in meta_rows], np.int64)
    s_a = np.array([m[4] for m in meta_rows], np.float32)
    g2_a = np.array([m[5] for m in meta_rows], np.float32)
    umi_a = np.array([m[6] for m in meta_rows], np.float32)
    ntc_a = np.array([m[7] for m in meta_rows], bool)

    report(f"  cells {len(X):,} | perturbed {int((~ntc_a).sum()):,} | NTC {int(ntc_a.sum()):,}")
    report(f"  distinct perturbations {len(set(pert_a[~ntc_a])):,} | guides {len(set(guide_a[~ntc_a])):,} "
           f"| batches {len(set(batch_a.tolist())):,}")
    for line in LINES:
        m = line_a == line
        report(f"    {line:<7} {int(m.sum()):>7,} cells, {int((m & ~ntc_a).sum()):>6,} perturbed, "
               f"{int((m & ntc_a).sum()):>6,} NTC")

    np.savez(SP / "sc_cohort_cells.npz", X=X, genes=np.array(hvg), line=line_a, pert=pert_a,
             guide=guide_a, batch=batch_a, s_score=s_a, g2m_score=g2_a, umi=umi_a, is_ntc=ntc_a)
    json.dump({"test": "sc_dataset", "n_cells": int(len(X)), "n_genes": len(hvg),
               "n_perts": len(set(pert_a[~ntc_a])), "n_guides": len(set(guide_a[~ntc_a])),
               "n_batches": len(set(batch_a.tolist())), "gb": gb, "log": log},
              open(OUT / "sc_dataset.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {SP / 'sc_cohort_cells.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
