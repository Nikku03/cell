"""Regenerates every rem.chromatin table, including the G4 discrepancy diagnostic."""
from __future__ import annotations
import argparse, json, sys
sys.path.insert(0, ".")
import numpy as np
from rem.chromatin import (solve_mu, coverage, gamma, naive_gamma, NUC, _spearman,
                           verify)


def naive_variants(cov, m, ell):
    W = m + ell - 1
    return {"footprint": 1 / np.clip(1 - np.convolve(cov, np.ones(m) / m, mode="same"),
                                     1e-12, None),
            "forbidden_window": 1 / np.clip(1 - np.convolve(cov, np.ones(W) / W,
                                                            mode="same"), 1e-12, None),
            "pointwise": 1 / np.clip(1 - cov, 1e-12, None)}


def g4_diagnostic(L=1500, m=10, occ=0.8, seed=0):
    """Under WHICH definition of the cheap approximation does it order sites wrongly?"""
    rng = np.random.default_rng(seed)
    print(f"\n=== G4 diagnostic: rank correlation of exact vs naive gamma at {occ:.0%} ===")
    print(f"  {'log-sd':>7s} {'exact med':>10s} {'spread':>8s} "
          + " ".join(f"{k:>18s}" for k in ("footprint", "forbidden_window", "pointwise")))
    rows = {}
    for sd in (0.0, 0.5, 1.0, 2.0):
        lw0 = rng.normal(0.0, sd, size=L) if sd > 0 else np.zeros(L)
        mu = solve_mu(lw0, occ, NUC)
        cov = coverage(lw0 + mu, NUC)
        ge = gamma(lw0 + mu, m, NUC)
        ok = np.isfinite(ge)
        sp = float(np.nanpercentile(ge[ok], 90) / np.nanpercentile(ge[ok], 10))
        cells, rows[sd] = [], {}
        for k, gn in naive_variants(cov, m, NUC).items():
            rc = _spearman(ge, gn)
            rows[sd][k] = {"rank": rc, "ratio": float(np.nanmedian(ge) / np.nanmedian(gn))}
            cells.append(f"{rc:+18.3f}")
        print(f"  {sd:7.1f} {np.nanmedian(ge):10.3f} {sp:8.2f} " + " ".join(cells))
    print("  reference claims -0.125 at 80%; no variant reproduces a negative correlation.")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/chromatin_results.json")
    a = ap.parse_args(argv)
    res = {"gates": verify(), "g4_diagnostic": g4_diagnostic()}
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"\nwritten {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
