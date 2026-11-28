"""WHAT SPHERE COVERAGE OF CHROMATIN WOULD COST, with every number measured rather than assumed.

The companion file `nexus_dna_spheres.py` asks whether a small sphere can represent the physics. This one
asks what it would cost if it could -- and the two questions are independent, so both have to be answered
before the proposal can be accepted or rejected.

EVERY INPUT HERE IS MEASURED FROM A REAL STRUCTURE, NOT ASSUMED.  The usual way to do this arithmetic is to
assume an atomic density for "biomolecular matter" and multiply. That hides the whole answer in one guessed
constant. Instead the neighbour count inside radius R is counted directly in the 1KX5 nucleosome -- the real
packing, including the DNA-histone interface where it is densest and the surface where it is not -- and the
distribution is reported, not just the mean.

WHAT THE COST ACTUALLY IS.  A sphere's work is its PAIR COUNT, ~m^2/2 for m atoms inside, because the physics
is a pairwise sum. Tiling N atoms with spheres of radius R at centre spacing s puts each atom in several
spheres, so the total is (number of spheres) x (pairs per sphere), and the overlap multiplicity is a cost
multiplier, not a free lunch.

THE COMPARISON THAT MATTERS, and it is not "sphere tiling vs nothing".  A plain neighbour list with the same
cutoff computes every pair ONCE and is exact within that cutoff. Sphere tiling computes the same pairs
several times over. So the honest baseline is the cell list, and the tiling has to justify itself against
that -- which it can only do if the per-sphere operation is a nonlinear local SOLVE (a minimisation, an SCF)
rather than a sum. NEXUS's per-sphere operation is a sum. That comparison is computed here explicitly.

-> outputs/orphan/nexus_dna_scale.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexus_dna_spheres import fetch_pdb, load  # noqa: E402  -- same structure, same parser, no drift

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
RADII = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0)
BP_HAPLOID = 3.1e9
NUC_REPEAT = 200          # bp per nucleosome: 147 core + ~53 linker
SEED = 23


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("COST OF SPHERE COVERAGE FOR CHROMATIN -- measured density, not an assumed one")
    report("=" * 100)

    t = load(fetch_pdb(), keep_water=True)
    co = t["co"]
    n = len(co)
    is_w = t["rs"] == "HOH"
    nd = int(t["dna"].sum())
    nh = int((~t["dna"] & ~is_w).sum())
    report(f"  1KX5: {n} atoms | {nd} DNA (147 bp) | {nh} histone+ion | {int(is_w.sum())} ordered water")
    report(f"  DNA heavy atoms per base pair, measured: {nd/147:.1f}")

    # --- measured neighbour counts -------------------------------------------------------------------
    report("\n  ATOMS INSIDE A SPHERE OF RADIUS R, counted in the real structure.")
    report("  The mean is not enough: the DNA-histone core is denser than the surface, and cost goes as the")
    report("  SQUARE of the count, so the expensive spheres dominate. p90 is reported for that reason.")
    report(f"    {'R':>5}{'mean':>9}{'median':>9}{'p90':>9}{'max':>8}")
    counts = {}
    rng = np.random.default_rng(SEED)
    probe = rng.choice(n, size=min(3000, n), replace=False)   # sampling the probe set, not the neighbours
    for R in RADII:
        c = []
        for s in range(0, len(probe), 256):
            idx = probe[s:s + 256]
            d = np.linalg.norm(co[idx][:, None, :] - co[None, :, :], axis=2)
            c.append(((d < R) & (d > 1e-6)).sum(1))
        c = np.concatenate(c)
        counts[R] = c
        report(f"    {R:>5.0f}{c.mean():>9.1f}{np.median(c):>9.1f}{np.percentile(c,90):>9.1f}{c.max():>8.0f}")

    # --- genome-scale atom count ----------------------------------------------------------------------
    report("\n  ATOMS IN CHROMATIN, scaled from the measured nucleosome.")
    dna_per_bp = nd / 147.0
    hist_per_nuc = nh
    n_nuc_hap = BP_HAPLOID / NUC_REPEAT
    dna_hap = BP_HAPLOID * dna_per_bp
    hist_hap = n_nuc_hap * hist_per_nuc
    tot_hap = dna_hap + hist_hap
    report(f"    DNA, haploid          {BP_HAPLOID:.2g} bp x {dna_per_bp:.1f} atoms/bp   = {dna_hap:.3g}")
    report(f"    histones, haploid     {n_nuc_hap:.3g} nucleosomes x {hist_per_nuc}      = {hist_hap:.3g}")
    report(f"    total heavy, haploid                                      = {tot_hap:.3g}")
    report(f"    total heavy, DIPLOID                                      = {2*tot_hap:.3g}")
    report("    (heavy atoms only, and dry: hydrogens roughly +60% and the hydration shell more again,")
    report("     so this is a LOWER bound on the system that would actually have to be covered)")
    N = 2 * tot_hap

    # --- sphere counts and pair work ------------------------------------------------------------------
    report("\n  THE TILING. Sphere centres spaced R apart (not 2R) so neighbours overlap and share atoms --")
    report("  which is what the proposal requires for a change in one sphere to be felt in the next.")
    report("  MULT is how many spheres each atom lands in; it multiplies the work, it is not free.")
    report(f"    {'R':>5}{'atoms/sph':>11}{'spheres':>12}{'mult':>7}{'pairs/sph':>11}{'TOTAL pairs':>14}"
           f"{'vs cell list':>14}")
    rows = {}
    for R in RADII:
        m = counts[R].mean() + 1                    # +1: the sphere's own centre atom
        # centres on a cubic lattice of spacing R covering the same volume: an atom at a generic point lies
        # inside every sphere whose centre is within R, i.e. (4/3)pi R^3 / R^3 = 4.19 of them
        mult = 4.0 / 3.0 * np.pi
        n_sph = N * mult / m                        # every atom in `mult` spheres, m atoms per sphere
        pairs_per = m * (m - 1) / 2
        total = n_sph * pairs_per
        cell = N * counts[R].mean() / 2             # each pair once, exact within the same cutoff
        rows[R] = {"atoms_per_sphere": float(m), "spheres": float(n_sph), "mult": float(mult),
                   "pairs_per_sphere": float(pairs_per), "total_pairs": float(total),
                   "cell_list_pairs": float(cell), "ratio": float(total / cell)}
        report(f"    {R:>5.0f}{m:>11.1f}{n_sph:>12.2e}{mult:>7.1f}{pairs_per:>11.1f}{total:>14.2e}"
               f"{total/cell:>13.1f}x")

    report("\n  THE COMPARISON THAT MATTERS. The last column is sphere tiling against a plain cell list at")
    report("  the SAME cutoff, which computes each pair once and is exact within it. Tiling is strictly more")
    report("  work for the same physics, because overlap re-evaluates shared pairs. A decomposition only")
    report("  repays that when each sphere does a nonlinear local SOLVE whose cost is superlinear in m --")
    report("  a minimisation or an SCF. NEXUS's per-sphere operation is a pairwise SUM, so there is nothing")
    report("  to amortise and the overlap is pure overhead.")
    report(f"  THAT COLUMN IS THE SAME {mult:.1f}x AT EVERY RADIUS, AND THAT IS NOT A COINCIDENCE IN THE DATA:")
    report("  total tiled pairs = N*mult*(m-1)/2 and cell-list pairs = N*(m-1)/2, so the ratio is exactly the")
    report("  overlap multiplicity (4/3)pi and the measured density cancels out. It is a geometric fact about")
    report("  covering space with overlapping balls, not a measurement -- reported as such so it is not read")
    report("  as an empirical finding that happened to come out constant.")

    # --- wall clock -----------------------------------------------------------------------------------
    report("\n  WALL CLOCK, from a measured pair rate on this machine.")
    a = co[:4000]
    tm = time.time()
    for _ in range(3):
        d = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        _ = (332.06 / np.maximum(d, 0.1) ** 2).sum()
    rate = 3 * len(a) ** 2 / (time.time() - tm)
    report(f"    measured: {rate:.2e} pair-evaluations/s in numpy on this box (distance + one r^-2 term)")
    report(f"    {'R':>5}{'total pairs':>14}{'core-seconds':>15}{'core-years':>13}")
    for R in RADII:
        tp = rows[R]["total_pairs"]
        sec = tp / rate
        report(f"    {R:>5.0f}{tp:>14.2e}{sec:>15.2e}{sec/3.15e7:>13.2e}")
        rows[R]["seconds"] = float(sec)

    report("\n  AND THAT IS FOR ONE ENERGY EVALUATION OF ONE STATIC STRUCTURE.")
    report("  It is worth being exact about what it does and does not buy, because the proposal's stated")
    report("  goal was to avoid brute-force MD. MD's cost is not the per-step spatial sum -- it is the")
    report("  10^6-10^9 STEPS. Sphere tiling is a way of organising one step. It does not touch the")
    report("  timescale, so it cannot substitute for dynamics; it can only make a single snapshot cheaper,")
    report("  and the column above shows it makes it dearer than a cell list.")

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"test": "nexus_dna_scale", "n_atoms_1kx5": n, "dna_atoms_per_bp": float(dna_per_bp),
           "genome_heavy_atoms_diploid": float(N), "pair_rate": float(rate),
           "radii": {str(R): rows[R] for R in RADII}, "log": log}
    json.dump(res, open(OUT / "nexus_dna_scale.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_dna_scale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
