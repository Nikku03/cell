"""CELL CYCLE as a continuous phase angle, inferred and VALIDATED on real single-cell data.

THE PROPOSAL, and the one part of it that needed changing. Represent the cycle as theta in [0, 2pi) rather
than three discrete bins; let theta drive a global capacity term C_vol(theta) = 2^(theta/2pi) and a per-gene
cyclical driver f_i(theta) fitted as a low-order Fourier series. That is right, and it is testable on data
already on disk: RPE1, Jurkat and HepG2 are single-cell, so every cell has its own phase.

    theta_c = atan2(G2M_c, S_c)

is the proposed assignment and it has a defect worth fixing rather than inheriting. S and G2M scores are both
NON-NEGATIVE means over marker sets, so atan2 of two non-negative numbers spans only the FIRST QUADRANT --
theta in [0, pi/2), a quarter of the circle, with G1 cells (low on both) piling up at an arbitrary angle
determined by noise. Centring each score before the atan2 restores the full circle and puts G1 opposite
S/G2M where it belongs. The fix is one line and the difference is visible in the printed occupancy.

WHAT IS ACTUALLY VALIDATED HERE, rather than asserted:

  1 theta recovers known biology     cyclical markers must peak at DIFFERENT theta, in the right order
                                     (S markers before G2/M markers), not all at once
  2 C_vol(theta) is real             total UMI per cell must rise with theta -- cells do grow
  3 f_i(theta) is fittable           a 2-harmonic Fourier fit must beat a flat mean on HELD-OUT cells,
                                     and must do so for known cyclical genes and NOT for housekeeping genes
  4 the confound is real             how much of a knockout's apparent response is explained by a shift in
                                     the phase distribution rather than by regulation

Point 4 is the reason to build this at all. If knocking out a checkpoint gene arrests cells, the pseudobulk
response is partly "these cells are in a different phase", not "these genes were regulated". That is a
confound in every Perturb-seq number this project has produced, and it can be measured.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
SP = Path(os.environ.get("CELL_SCRATCH",
                         "/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

S_MARK = ["PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "TYMS", "RRM1", "RRM2", "SLBP",
          "GINS2", "CDC45", "CDC6", "UNG", "FEN1", "DHFR", "CLSPN", "GMNN", "CHAF1B"]
G2M_MARK = ["MKI67", "TOP2A", "CCNB1", "CCNB2", "CDK1", "BUB1", "PLK1", "AURKA", "AURKB",
            "CENPF", "TPX2", "UBE2C", "NUSAP1", "ANLN", "KIF11", "CDC20", "CKS2", "HMMR"]
HOUSE = ["ACTB", "GAPDH", "TUBB", "RPL13A", "RPS18", "PPIA", "B2M", "TBP", "HPRT1", "YWHAZ"]
NH = 2                    # Fourier harmonics
R = {}


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100)


def load_cells(name, max_cells=60000):
    """Per-cell expression restricted to the marker genes we need, plus total UMI. Chunked."""
    import h5py
    p = SP / name
    with h5py.File(p, "r") as h:
        v = h["var"]
        gn = v["gene_name"] if "gene_name" in v else None
        if isinstance(gn, h5py.Group):
            cats = [x.decode() for x in gn["categories"][:]]
            genes = np.array([cats[c] for c in gn["codes"][:]])
        else:
            genes = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in gn[:]])
        want = sorted(set(S_MARK + G2M_MARK + HOUSE))
        gi = {g: i for i, g in enumerate(genes)}
        cols = [gi[g] for g in want if g in gi]
        names = [g for g in want if g in gi]
        X = h["X"]
        n = min(X.shape[0], max_cells)
        M = np.zeros((n, len(cols)), np.float32)
        tot = np.zeros(n, np.float32)
        for s in range(0, n, 8000):
            e = min(s + 8000, n)
            blk = np.asarray(X[s:e], np.float32)
            M[s:e] = blk[:, cols]
            tot[s:e] = blk.sum(1)
        pert = None
        if "perturbation" in h["obs"]:
            o = h["obs"]["perturbation"]
            if isinstance(o, h5py.Group):
                pc = [x.decode() for x in o["categories"][:]]
                pert = np.array([pc[c] if 0 <= c < len(pc) else "?" for c in o["codes"][:n]])
    return names, M, tot, pert


def phase_angle(names, M):
    """theta from CENTRED S and G2M scores.

    Centring is the whole correction. Raw scores are non-negative means, so atan2(G2M, S) lives in the first
    quadrant only and G1 cells -- low on both axes -- land at whatever angle their noise ratio happens to
    give. Subtracting each score's median puts the origin inside the cloud, so low-low cells sit opposite
    high-high ones and the phase spans the full circle.
    """
    idx = {g: i for i, g in enumerate(names)}
    si = [idx[g] for g in S_MARK if g in idx]
    gi = [idx[g] for g in G2M_MARK if g in idx]
    S = M[:, si].mean(1) if si else np.zeros(len(M))
    G = M[:, gi].mean(1) if gi else np.zeros(len(M))
    Sc, Gc = S - np.median(S), G - np.median(G)
    raw = np.arctan2(G, S) % (2 * np.pi)
    th = np.arctan2(Gc, Sc) % (2 * np.pi)
    return th, raw, S, G, len(si), len(gi)


def fourier_design(th, nh=NH):
    cols = [np.ones_like(th)]
    for k in range(1, nh + 1):
        cols += [np.sin(k * th), np.cos(k * th)]
    return np.stack(cols, 1)


def fit_eval(th, y, tr, te, nh=NH):
    """Held-out R^2 of a Fourier fit against the flat-mean null."""
    A = fourier_design(th, nh)
    w, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
    p = A[te] @ w
    ss = ((y[te] - p) ** 2).sum()
    s0 = ((y[te] - y[tr].mean()) ** 2).sum()
    return 1 - ss / max(s0, 1e-12)


def main():
    hdr("CELL CYCLE -- continuous phase, inferred and validated on real single cells")
    src = [("rpe1.h5ad", "RPE1"), ("nadig_jurkat.h5ad", "Jurkat"), ("nadig_hepg2.h5ad", "HepG2")]
    out = {}
    for fn, tag in src:
        if not (SP / fn).exists():
            print(f"  {tag}: source missing, skipped")
            continue
        names, M, tot, pert = load_cells(fn)
        th, raw, S, G, ns, ng = phase_angle(names, M)
        rng = np.random.default_rng(0)
        m = rng.random(len(th)) < 0.5
        tr, te = np.where(m)[0], np.where(~m)[0]

        # 1. does theta span the circle, and did centring matter?
        occ = np.histogram(th, bins=8, range=(0, 2 * np.pi))[0] / len(th)
        occ_raw = np.histogram(raw, bins=8, range=(0, 2 * np.pi))[0] / len(raw)
        print(f"\n  {tag}: {len(th):,} cells, {ns} S markers, {ng} G2/M markers")
        print(f"    octant occupancy  centred {np.round(occ,3).tolist()}")
        print(f"    octant occupancy  RAW     {np.round(occ_raw,3).tolist()}   <- collapses to one quadrant")

        # 2. C_vol: total UMI must rise with theta
        r2_vol = fit_eval(th, np.log1p(tot), tr, te)
        b = np.digitize(th, np.linspace(0, 2 * np.pi, 9)[1:-1])
        prof = [float(np.median(tot[b == k])) for k in range(8)]
        print(f"    total-UMI vs theta: held-out R2 {r2_vol:+.4f}; median UMI by octant "
              f"{[int(x) for x in prof]}")

        # 3. f_i(theta): Fourier must beat a flat mean for cyclical genes and NOT for housekeeping
        # SIZE-NORMALISE BEFORE FITTING f_i. The architecture is alpha = C_vol(theta) * [alpha_3D + f_i(theta)],
        # so C_vol must be divided out before f_i is fitted or the two terms are conflated. Fitting RAW counts
        # gave housekeeping genes held-out R2 of +0.27 and +0.16 -- they look cyclical because the cell is
        # growing, not because they cycle -- and the control that is supposed to collapse did not. On
        # counts-per-10k the volume term is gone and only genuine cycling remains.
        idx = {g: i for i, g in enumerate(names)}
        cp10k = M / np.maximum(tot[:, None], 1.0) * 1e4
        def grp(gs, mat):
            v = [fit_eval(th, np.log1p(mat[:, idx[g]]), tr, te) for g in gs if g in idx]
            return float(np.median(v)) if v else float("nan"), len(v)
        rawS, _ = grp(S_MARK, M)
        rawH, _ = grp(HOUSE, M)
        r2s, n1 = grp(S_MARK, cp10k)
        r2g, n2 = grp(G2M_MARK, cp10k)
        r2h, n3 = grp(HOUSE, cp10k)
        print(f"    f_i(theta) held-out R2, SIZE-NORMALISED:  S {r2s:+.4f} (n={n1}) | "
              f"G2/M {r2g:+.4f} (n={n2}) | housekeeping {r2h:+.4f} (n={n3})")
        print(f"      on RAW counts these were S {rawS:+.4f}, housekeeping {rawH:+.4f} -- the housekeeping")
        print("      control only collapses once C_vol is divided out, which is why it must be")

        # peak ordering: S markers must peak BEFORE G2/M markers
        def peak(gs):
            ps = []
            for g in gs:
                if g not in idx:
                    continue
                A = fourier_design(th)
                w, *_ = np.linalg.lstsq(A, np.log1p(M[:, idx[g]]), rcond=None)
                grid = np.linspace(0, 2 * np.pi, 200)
                ps.append(grid[np.argmax(fourier_design(grid) @ w)])
            return float(np.median(ps)) if ps else float("nan")
        pS, pG = peak(S_MARK), peak(G2M_MARK)
        gap = (pG - pS) % (2 * np.pi)
        print(f"    peak theta: S {pS:.2f} rad, G2/M {pG:.2f} rad, separation {gap:.2f} rad "
              f"({'ordered' if 0.2 < gap < np.pi else 'NOT resolved'})")

        out[tag] = {"n_cells": int(len(th)), "r2_volume": r2_vol, "r2_S": r2s, "r2_G2M": r2g,
                    "r2_house": r2h, "r2_S_raw": rawS, "r2_house_raw": rawH, "peak_S": pS, "peak_G2M": pG, "separation": gap,
                    "octant_centred": occ.tolist(), "octant_raw": occ_raw.tolist()}

        # 4. the confound: does a knockout shift the phase distribution?
        if pert is not None:
            base = np.array([np.mean(np.cos(th)), np.mean(np.sin(th))])
            shifts = []
            for p in set(pert.tolist()):
                sel = pert == p
                if sel.sum() < 60:
                    continue
                v = np.array([np.mean(np.cos(th[sel])), np.mean(np.sin(th[sel]))])
                shifts.append((float(np.linalg.norm(v - base)), p, int(sel.sum())))
            shifts.sort(reverse=True)
            if shifts:
                med = float(np.median([s for s, _, _ in shifts]))
                print(f"    phase-distribution shift vs all-cells: median {med:.4f}; largest:")
                for s, p, n in shifts[:5]:
                    print(f"      {p:12s} shift {s:.4f}  ({n:,} cells)")
                out[tag]["phase_shift_median"] = med
                out[tag]["phase_shift_top"] = [(p, s, n) for s, p, n in shifts[:10]]
    R["lines"] = out
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "cellcycle.json", "w"), indent=1, default=float)
    print(f"\n  -> {OUT/'cellcycle.json'}")


if __name__ == "__main__":
    main()
