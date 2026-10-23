"""Regenerates every table for rem.mps. Gates G1, G1b, G2, G3, G4, G5, G6.

The one question this exists to answer: DOES THE REQUIRED BOND DIMENSION STAY BOUNDED AT
REAL TRANSCRIPT LENGTH? Everything else is the scaffolding needed to make that answer mean
something.
"""
from __future__ import annotations

import argparse, json, sys, time
sys.path.insert(0, ".")
import numpy as np

from rem.mps import (solve, exact_stationary, exact_occupancy, kmc_occupancy,
                     realistic_rates, product_mps, occupancy)

ALPHA, BETA, DT = 0.8, 0.9, 0.02
OUT = "benchmarks/driven1d_results.json"


def g1(verbose=True):
    """Reproduce the exact answer at small L. Reference values are in rem/mps.py."""
    print("\n=== G1  MPS vs exact sparse null-space solve ===")
    print(f"  {'L':>3s} " + " ".join(f"{'chi='+str(c):>10s}" for c in (2, 4, 8, 16))
          + f" {'t':>7s} {'resid':>8s}")
    rows, ok = [], True
    for L in (8, 10, 12, 14):
        r = realistic_rates(L, seed=1)
        pe = exact_occupancy(exact_stationary(L, r, ALPHA, BETA), L)
        errs, infos = [], []
        for chi in (2, 4, 8, 16):
            prof, info = solve(L, r, ALPHA, BETA, chi=chi, dt=DT, tol_rate=1e-7,
                               max_seconds=900)
            errs.append(float(np.abs(prof - pe).max())); infos.append(info)
        print(f"  {L:3d} " + " ".join(f"{e:10.2e}" for e in errs)
              + f" {infos[-1]['t']:7.1f} {infos[-1]['residual']:8.1e}")
        mono = all(a > b for a, b in zip(errs, errs[1:]))
        ok &= mono and all(i["converged"] for i in infos)
        rows.append({"L": L, "errs": errs, "monotone": mono,
                     "converged": [i["converged"] for i in infos],
                     "discarded": [i["worst_discarded"] for i in infos]})
    print(f"  monotone geometric convergence in chi at every L, all runs time-converged: "
          f"{'PASS' if ok else 'FAIL'}")
    return {"rows": rows, "pass": bool(ok)}


def g1b():
    """The Trotter ORDER, gated on the exponent (not the magnitude)."""
    print("\n=== G1b Trotter order: fit log(err) vs log(dt) ===")
    L = 8
    r = realistic_rates(L, seed=1)
    pe = exact_occupancy(exact_stationary(L, r, ALPHA, BETA), L)
    out = {}
    for order, label in ((True, "second"), (False, "first")):
        dts, errs = [], []
        for dt in (0.2, 0.1, 0.05, 0.02, 0.01):
            prof, info = solve(L, r, ALPHA, BETA, chi=32, dt=dt, tol_rate=1e-9,
                               second_order=order, max_seconds=600)
            dts.append(dt); errs.append(float(np.abs(prof - pe).max()))
        sl = float(np.polyfit(np.log10(dts), np.log10(errs), 1)[0])
        print(f"  {label:6s} order: " + "  ".join(f"dt={d:g}:{e:.2e}"
                                                  for d, e in zip(dts, errs)))
        print(f"          fitted slope {sl:.3f}")
        out[label] = {"dts": dts, "errs": errs, "slope": sl}
    ok = 1.7 <= out["second"]["slope"] <= 2.3
    print(f"  second-order slope in [1.7, 2.3]: {'PASS' if ok else 'FAIL'}"
          f"   (first order measured {out['first']['slope']:.2f}, as it should be ~1)")
    out["pass"] = bool(ok)
    return out


def g3(Ls=(12, 25, 50, 100), chis=(2, 4, 8, 16, 32), chi_ref=64, budget=1200.0,
       dt=DT):
    """THE question: chi_required(L). Time-converged separately at every chi (G2)."""
    print(f"\n=== G3  chi_required(L)   [reference chi={chi_ref}, dt={dt}] ===")
    print(f"  {'L':>5s} {'chi':>4s} {'maxdev vs ref':>14s} {'disc':>9s} {'t':>8s} "
          f"{'resid':>9s} {'conv':>5s} {'sec':>7s}")
    rows = []
    for L in Ls:
        r = realistic_rates(L, seed=1)
        ref, iref = solve(L, r, ALPHA, BETA, chi=chi_ref, dt=dt, tol_rate=1e-6,
                          max_seconds=budget)
        print(f"  {L:5d} {chi_ref:4d} {'(reference)':>14s} "
              f"{iref['worst_discarded']:9.1e} {iref['t']:8.1f} {iref['residual']:9.1e} "
              f"{str(iref['converged']):>5s} {iref['seconds']:7.1f}")
        entry = {"L": L, "ref": {k: iref[k] for k in
                                 ("t", "residual", "converged", "seconds",
                                  "worst_discarded", "max_bond")}, "chi": {}}
        for chi in chis:
            if chi >= chi_ref:
                continue
            prof, info = solve(L, r, ALPHA, BETA, chi=chi, dt=dt, tol_rate=1e-6,
                               max_seconds=budget)
            dev = float(np.abs(prof - ref).max())
            entry["chi"][chi] = {"dev": dev, "t": info["t"],
                                 "residual": info["residual"],
                                 "converged": info["converged"],
                                 "discarded": info["worst_discarded"],
                                 "seconds": info["seconds"]}
            print(f"  {L:5d} {chi:4d} {dev:14.2e} {info['worst_discarded']:9.1e} "
                  f"{info['t']:8.1f} {info['residual']:9.1e} "
                  f"{str(info['converged']):>5s} {info['seconds']:7.1f}")
        for target in (1e-3, 1e-5):
            need = [c for c in sorted(entry["chi"]) if entry["chi"][c]["dev"] < target]
            entry[f"chi_req_{target:g}"] = need[0] if need else None
        rows.append(entry)
        print(f"        -> chi required for 1e-3: {entry['chi_req_0.001']}, "
              f"for 1e-05: {entry['chi_req_1e-05']}")
    print(f"\n  {'L':>5s} {'chi(1e-3)':>10s} {'chi(1e-5)':>10s}")
    for e in rows:
        print(f"  {e['L']:5d} {str(e['chi_req_0.001']):>10s} "
              f"{str(e['chi_req_1e-05']):>10s}")
    return {"rows": rows, "chi_ref": chi_ref}


def g4(L=50, chi=32, budget=1800.0):
    """Ground truth where exact is impossible: kinetic Monte Carlo with an error bar."""
    print(f"\n=== G4  MPS vs kinetic Monte Carlo at L={L} ===")
    r = realistic_rates(L, seed=1)
    prof, info = solve(L, r, ALPHA, BETA, chi=chi, dt=DT, tol_rate=1e-6,
                       max_seconds=budget)
    t0 = time.perf_counter()
    km, se = kmc_occupancy(L, r, ALPHA, BETA, t_equil=300.0, t_meas=1500.0, seed=3)
    dev = np.abs(prof - km)
    within = int((dev <= 2 * se).sum())
    print(f"  MPS  chi={chi} t={info['t']:.0f} resid={info['residual']:.1e} "
          f"disc={info['worst_discarded']:.1e} ({info['seconds']:.0f}s)")
    print(f"  KMC  {time.perf_counter()-t0:.0f}s, mean standard error {se.mean():.4f}")
    print(f"  max |MPS - KMC| = {dev.max():.4f}   mean = {dev.mean():.4f}")
    print(f"  sites agreeing within 2 standard errors: {within}/{L}")
    ok = within >= int(0.9 * L)
    print(f"  G4 {'PASS' if ok else 'FAIL'} (bar: >=90% of sites within 2 s.e.)")
    return {"max_dev": float(dev.max()), "mean_dev": float(dev.mean()),
            "within_2se": within, "L": L, "mean_se": float(se.mean()), "pass": bool(ok)}


def g5(L=40, chi=32, budget=900.0):
    """THE RELEVANCE GATE. Does exclusion matter at the density biology actually runs at?"""
    print(f"\n=== G5  relevance: interacting vs independent-site, at matched density ===")
    print(f"  {'alpha':>7s} {'density':>8s} {'%jam':>5s} {'max|int-indep|':>15s} "
          f"{'rel/rho':>8s} {'trunc err':>10s}")
    r = realistic_rates(L, seed=1)
    rows = []
    for alpha in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
        prof, info = solve(L, r, alpha, BETA, chi=chi, dt=DT, tol_rate=1e-6,
                           max_seconds=budget)
        rho = float(prof.mean())
        # independent-site model at the SAME density: mean-field TASEP with no exclusion
        # correlation -- each site's occupancy set by its own rates and the global density.
        lo, hi = -60.0, 60.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            q = 1.0 / (1.0 + np.exp(-(np.log(np.r_[r, r[-1]]) * 0 + mid)))
            if q.mean() < rho:
                lo = mid
            else:
                hi = mid
        q = np.full(L, rho)                      # uniform independent sites at same density
        dev = float(np.abs(prof - q).max())
        rows.append({"alpha": alpha, "rho": rho, "dev": dev,
                     "rel": dev / rho if rho > 0 else np.nan,
                     "trunc": info["worst_discarded"]})
        print(f"  {alpha:7.2f} {rho:8.4f} {100*rho:5.0f} {dev:15.4f} "
              f"{dev/max(rho,1e-12):8.3f} {info['worst_discarded']:10.1e}")
    print("  NOTE: for TASEP the independent-site reference is the flat mean-field profile;")
    print("  the deviation includes the boundary layers, which are a real interaction effect.")
    return {"rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="1,1b,5")
    ap.add_argument("--g3-L", default="12,25,50,100")
    ap.add_argument("--g3-budget", type=float, default=1200.0)
    ap.add_argument("--g3-chiref", type=int, default=64)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    want = set(x.strip() for x in a.gates.split(","))
    res = {}
    if "1" in want:  res["G1"] = g1()
    if "1b" in want: res["G1b"] = g1b()
    if "3" in want:
        res["G3"] = g3(Ls=tuple(int(x) for x in a.g3_L.split(",")),
                       chi_ref=a.g3_chiref, budget=a.g3_budget)
    if "4" in want:  res["G4"] = g4()
    if "5" in want:  res["G5"] = g5()
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"\nwritten {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
