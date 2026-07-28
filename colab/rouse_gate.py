"""ANALYTIC ROUSE-WITH-LOOPS: real polymer physics, no simulation, tested against the 0.663 bar.

WHAT THIS IS. For a Gaussian (Rouse) chain with harmonic crosslinks, the mean-square spatial distance between
any two loci is available in CLOSED FORM from the network's Kirchhoff matrix -- no integrator, no timestep, no
sampling. This is the Gaussian Network Model applied to chromatin:

    L = graph Laplacian of (backbone bonds + cohesin loop bonds)
    <R_ij^2> = (3kT/k) * (L+_ii + L+_jj - 2 L+_ij)        L+ = pseudo-inverse
    P_contact ~ <R_ij^2>^(-3/2)                            Gaussian chain end-to-end density at r->0

That is Stage 3 of the 4D plan -- the Langevin polymer dynamics costed at ~1 GPU-day per region -- reduced to
one sparse linear solve per pair, because the readout only ever needs the equilibrium contact probability and
never the trajectory.

WHY IT IS NOT THE SAME AS "COUNT CTCF BETWEEN". The previous gate's winning feature counts boundaries on the
segment E..P. That is a local, additive quantity. This is TOPOLOGICAL: a loop whose anchors straddle both E
and P pulls them together even though its anchors lie outside the segment, while a loop anchored between them
pushes them apart. Counting cannot express either. If the polymer physics buys anything beyond bookkeeping,
this is where it shows up.

    L x = e_i - e_j  solved sparsely rather than inverting: the Laplacian is tridiagonal plus a handful of
    loop bonds, and only ONE pair is needed per window, so the O(N^3) pseudo-inverse is unnecessary. The
    right-hand side is orthogonal to the constant null vector, so grounding one bead is a valid gauge.

CONTROLS. Same shuffled-CTCF twin as the previous gate (peaks relocated within their own chromosome, count and
signal preserved) plus the label-shuffle. A physics model that cannot beat its own shuffled anchors is fitting
genomic density, and the elaborate derivation above would be decoration.
"""
import bisect
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path("outputs/orphan")
INV = OUT / "invivo"

BEAD = 5_000          # bp per bead; 5 kb keeps a 4 Mb window at 800 beads
FLANK = 300_000       # context beyond the E..P span so loops straddling the pair are represented
MAXSPAN = 4_000_000   # cap so a handful of very distant pairs cannot dominate runtime
K_LOOP = 3.0          # loop-bond stiffness relative to the backbone (cohesin bridge vs one Kuhn segment)
LP = 1_000            # chromatin persistence length in bp (~1-3 kb, ~30-50 nm); 0 disables bending


def loop_bonds(anchors, sigs, n_beads, off):
    """Cohesin loops between CONSECUTIVE CTCF anchors, stiffness scaled by the weaker of the two peaks.

    Consecutive pairing is the minimal topology consistent with extrusion: a ring loads, extrudes both ways,
    and stalls at the first barrier it meets on each side, so the loops that actually form join neighbouring
    anchors. Nested and skipped loops exist, and are approximated by also joining next-nearest anchors at
    reduced weight rather than being ignored.
    """
    out = []
    b = [(a - off) // BEAD for a in anchors]
    for gap, w in ((1, 1.0), (2, 0.35)):
        for i in range(len(b) - gap):
            u, v = b[i], b[i + gap]
            if 0 <= u < n_beads and 0 <= v < n_beads and u != v:
                out.append((u, v, K_LOOP * w * min(sigs[i], sigs[i + gap])))
    return out


def bending(n, kappa):
    """Quadratic curvature penalty -- the SEMIFLEXIBLE GAUSSIAN CHAIN, which keeps the closed form.

    A true worm-like chain uses U = K(1 - cos theta), a three-body NON-quadratic term. Add it and the
    Boltzmann distribution stops being multivariate Gaussian, <R^2> no longer follows from an inverse
    Laplacian, and the only route left is GPU Langevin -- i.e. the expensive path.

    The quadratic curvature penalty (kappa/2) * sum |r_{i+1} - 2 r_i + r_{i-1}|^2 is the harmonic
    approximation to that bending energy. It penalises the SAME thing -- sharp local kinks -- while staying
    quadratic in the coordinates, so the whole model remains Gaussian and solvable in closed form. The
    operator is D2^T D2, the discrete biharmonic, pentadiagonal with stencil [1,-4,6,-4,1]: one extra
    diagonal, no extra asymptotic cost.

    kappa = L_p / a reproduces the standard bending modulus K_bend = kT * L_p / a. It also self-disables
    correctly: at a = 5 kb with L_p = 1 kb, kappa = 0.2 and the chain is nearly free, which is right --
    beads longer than the persistence length ARE freely jointed. The stiffness only switches on when the
    resolution goes below the Kuhn length, which is exactly where the naive model breaks.
    """
    if kappa <= 0 or n < 3:
        return sparse.csr_matrix((n, n))
    r, c, v = [], [], []
    for i in range(1, n - 1):
        for dj, w in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            r.append(i - 1)
            c.append(i + dj)
            v.append(w)
    D2 = sparse.coo_matrix((v, (r, c)), shape=(n - 2, n)).tocsr()
    return (D2.T @ D2) * kappa


def r2_pair(chrom, e, t, ctcf, norm):
    """<R^2> between the element bead and the TSS bead, in units of the backbone Kuhn length."""
    lo, hi = (e, t) if e <= t else (t, e)
    if hi - lo > MAXSPAN:
        return None
    off = max(lo - FLANK, 0)
    end = hi + FLANK
    n = max((end - off) // BEAD + 1, 4)
    a, v = ctcf.get(chrom, (np.array([], np.int64), np.array([], np.float64)))
    l = bisect.bisect_left(a, off)
    r = bisect.bisect_right(a, end)
    anchors, sigs = a[l:r], v[l:r] / norm

    rows, cols, vals = [], [], []
    for i in range(n - 1):                       # backbone
        rows += [i, i + 1]
        cols += [i + 1, i]
        vals += [-1.0, -1.0]
    for u, w, k in loop_bonds(anchors, sigs, n, off):
        rows += [u, w]
        cols += [w, u]
        vals += [-k, -k]
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    deg = -np.asarray(A.sum(1)).ravel()
    L = (sparse.diags(deg) + A + bending(n, LP / BEAD if LP else 0.0)).tolil()
    L[0, 0] += 1.0                               # ground one bead: the RHS is orthogonal to constants

    i = min(max((e - off) // BEAD, 0), n - 1)
    j = min(max((t - off) // BEAD, 0), n - 1)
    if i == j:
        return 0.0
    b = np.zeros(n)
    b[i], b[j] = 1.0, -1.0
    try:
        x = spsolve(L.tocsc(), b)
    except Exception:
        return None
    r2 = float(x[i] - x[j])
    return r2 if np.isfinite(r2) and r2 > 0 else None


NAMES = ["rouse_logr2", "rouse_logP", "rouse_vs_ideal"]


def features(rows, ctcf):
    """log<R^2>, log contact probability, and the ratio to the loop-free ideal chain.

    rouse_vs_ideal is the one that isolates the LOOPS: an ideal chain gives <R^2> ~ s, so dividing out the
    genomic separation leaves only what the loop topology contributed. Without it the model could score well
    on a quantity that is mostly distance in disguise -- the trap the previous gate had to rule out.
    """
    norm = np.median(np.concatenate([v for _, v in ctcf.values()])) or 1.0
    F = np.zeros((len(rows), 3))
    bad = 0
    for k, (c, e, t) in enumerate(rows):
        r2 = r2_pair(c, e, t, ctcf, norm)
        s = max(abs(e - t) / BEAD, 1.0)
        if r2 is None or r2 <= 0:
            bad += 1
            r2 = s                                # loop-free ideal chain as the fallback
        F[k] = (np.log10(r2), -1.5 * np.log10(r2), np.log10(r2 / s))
    return F, bad


def main():
    import pandas as pd
    from crispr_gate import _cv_auprc, _seeded_folds
    from contact_gate import load_ctcf, contact_features, NAMES as CNAMES

    print("=" * 100)
    print("ANALYTIC ROUSE-WITH-LOOPS vs the 0.663 bar")
    print("=" * 100)
    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]

    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}", p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2, int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]

    real, shuf = load_ctcf(), load_ctcf(shuffle_seed=7)
    print(f"  {len(df):,} pairs, bead {BEAD//1000} kb, flank {FLANK//1000} kb, k_loop {K_LOOP}")
    Rr, badr = features(rows, real)
    Rs, _ = features(rows, shuf)
    print(f"  solved analytically for {len(df)-badr:,}/{len(df):,} pairs "
          f"({badr:,} fell back to the ideal chain: span > {MAXSPAN//10**6} Mb or singular)")
    print(f"  log10(<R2>/ideal): median {np.median(Rr[:,2]):+.3f} real vs {np.median(Rs[:,2]):+.3f} shuffled"
          "   <- negative = loops COMPACT the pair below a loop-free chain")

    Cr = contact_features(rows, real)
    tfmat = np.zeros((len(df), len(tfl)), np.int8)
    for i, ek in enumerate(df["element"].values):
        for ti in et.get(ek, []):
            tfmat[i, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    y = df["crispr_hit"].values
    ch = df["chromosome"].values
    Xa = np.hstack([df[base].values, tfmat])
    Xc = np.hstack([Xa, Cr])

    def sc(X):
        return float(np.mean([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in (0, 1, 2)]))

    res = {}
    arms = [("epi + TF identity (311)", Xa),
            ("+ CTCF-count contact  [the 0.663 bar]", Xc),
            ("+ ROUSE only", np.hstack([Xa, Rr])),
            ("+ ROUSE + CTCF-count", np.hstack([Xc, Rr])),
            ("+ ROUSE [shuffled CTCF]", np.hstack([Xa, Rs]))]
    for nm, X in arms:
        a = sc(X)
        res[nm] = a
        print(f"    {nm:40s} AUPRC {a:.4f}")

    bar = res["+ CTCF-count contact  [the 0.663 bar]"]
    both = res["+ ROUSE + CTCF-count"]
    ronly = res["+ ROUSE only"]
    rsh = res["+ ROUSE [shuffled CTCF]"]
    print("\n  " + "-" * 96)
    print(f"  ROUSE+count MINUS the 0.663 bar : {both - bar:+.4f}")
    print(f"  ROUSE-only  MINUS the 0.663 bar : {ronly - bar:+.4f}")
    print(f"  ROUSE-only  MINUS its shuffled  : {ronly - rsh:+.4f}")
    if both - bar >= 0.01 and ronly - rsh >= 0.005:
        v = ("GO -- the polymer topology adds signal the boundary COUNT cannot express. A better contact "
             "model is worth building, and the analytic form already captures part of it.")
    elif ronly - rsh >= 0.005:
        v = ("PARTIAL -- Rouse contact is real (beats its shuffled anchors) but adds nothing beyond the "
             "CTCF-count feature. The topology is already captured by counting; the physics is redundant.")
    else:
        v = ("NO-GO -- analytic Rouse-with-loops does not beat a bisect over a BED file. Stage 3 of the 4D "
             "plan has nothing to add that counting boundaries does not already provide.")
    print(f"\n  VERDICT: {v}")
    res["verdict"] = v
    json.dump(res, open(OUT / "rouse_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'rouse_gate.json'}")


if __name__ == "__main__":
    main()
