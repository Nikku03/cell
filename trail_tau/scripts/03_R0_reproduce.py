#!/usr/bin/env python3
"""PHASE 1 / GATE R0. Reproduce Figure 2 of the paper from the deposited raw files.

Re-implements the paper's own ODE (transcribed from Figure_2.m), integrates it with the paper's
own deposited per-cell fitted parameters, and compares against the paper's own deposited
experimental FRET traces. If I have mis-ordered the parameter vector, mis-scaled time by K1,
used the wrong TRAIL0, or transposed an array, the correlation collapses. This gate can fail.
"""
import os, json
import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp

ROOT = os.path.join(os.path.dirname(__file__), "..")
F2 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 2")

# Constants transcribed from Figure_2.m (lines cited in INVENTORY.md)
K1 = 0.007325300696406          # 1/min, time rescaling
R0_INIT, C80_INIT = 32000.0, 30.0
TRAIL0 = {"25ng": 750.0, "50ng": 1500.0, "100ng": 3000.0}
T_OBS = np.linspace(5, 600, 120)   # minutes

cp = sio.loadmat(os.path.join(F2, "common_parameters.mat"))["common_parameters"].ravel()
rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET = cp

def rhs(t, y, p):
    T, R, Z0, Z3, pC8, Z1, Z2, FLIP, C8, FRET = y
    p1, p2, a0, a1, Kdeg = p
    bind = (T * R**3) / (R**3 + alphaR_3)
    return [
        -bind + rK1bK1*Z0,
        -3*bind + 3*rK1bK1*Z0,
        +bind - rK1bK1*Z0 - rK3K1*Z0*FLIP**3 + rK3bK3*rK3K1*Z3
              - rK2K1*Z0*pC8**2 + rK2bK2*rK2K1*Z1 + a0*Z1,
        +rK3K1*Z0*FLIP**3 - rK3bK3*rK3K1*Z3,
        -2*rK2K1*Z0*pC8**2 + 2*rK2bK2*rK2K1*Z1,
        +rK2K1*Z0*pC8**2 - rK2bK2*rK2K1*Z1 - rK2K1*Z1*FLIP + rK2bK2*rK2K1*Z2 - a0*Z1,
        +rK2K1*Z1*FLIP - rK2bK2*rK2K1*Z2 - a1*Z2,
        -3*rK3K1*Z0*FLIP**3 + 3*rK3bK3*rK3K1*Z3 - rK2K1*Z1*FLIP + rK2bK2*rK2K1*Z2,
        +a0*Z1 + a1*Z2 - Kdeg*(C8/(alphaC8 + C8)) - K_FRET*C8,
        +K_FRET*C8,
    ]

rows, excl = [], []
for grp in ("R", "S"):
    for dose in ("25ng", "50ng", "100ng"):
        fret = sio.loadmat(os.path.join(F2, f"FRET_{grp}_{dose}.mat"))[f"FRET_{grp}_{dose}"]
        par  = sio.loadmat(os.path.join(F2, f"{grp}_{dose}_ind_par.mat"))[f"{grp}_{dose}_ind_par"]
        tend = sio.loadmat(os.path.join(F2, f"Tend_{grp}_{dose}.mat"))[f"Tend_{grp}_{dose}"].ravel()
        for k in range(fret.shape[0]):
            p = par[k, :].astype(float)
            y0 = [TRAIL0[dose], R0_INIT, 0, 0, p[0], 0, 0, p[1], C80_INIT, 0]
            tmax = float(tend[k]) * K1
            sol = solve_ivp(rhs, [0, tmax], y0, args=(p,), method="LSODA",
                            rtol=1e-12, atol=1e-12, dense_output=True)
            if not sol.success:
                excl.append((grp, dose, k, "integration failed")); continue
            # their plot uses ode_solution.x/K1 as minutes; sample at the observed frames
            tm = T_OBS[T_OBS <= float(tend[k]) + 1e-9]
            sim = sol.sol(tm * K1)[9]
            obs = fret[k, :len(tm)].astype(float)
            m = np.isfinite(obs) & np.isfinite(sim)
            n_drop = int((~m).sum())
            if m.sum() < 10:
                excl.append((grp, dose, k, f"only {int(m.sum())} finite points")); continue
            o, s = obs[m], sim[m]
            r = float(np.corrcoef(o, s)[0, 1]) if o.std() > 0 and s.std() > 0 else np.nan
            nrmse = float(np.sqrt(np.mean((o - s)**2)) / (o.max() - o.min())) if o.max() > o.min() else np.nan
            rows.append(dict(group=grp, dose=dose, cell=k, n=int(m.sum()), n_nonfinite=n_drop,
                             pearson_r=r, nrmse=nrmse, Tend_min=float(tend[k])))

print(f"{'grp':4s}{'dose':7s}{'cell':5s}{'n':5s}{'r':>9s}{'nRMSE':>9s}{'Tend':>7s}")
for d in rows:
    print(f"{d['group']:4s}{d['dose']:7s}{d['cell']:<5d}{d['n']:<5d}"
          f"{d['pearson_r']:9.4f}{d['nrmse']:9.4f}{d['Tend_min']:7.0f}")
rr = np.array([d["pearson_r"] for d in rows], float)
print(f"\ncells compared: {len(rows)}   excluded: {len(excl)} {excl}")
print(f"median Pearson r = {np.nanmedian(rr):.4f}   min r = {np.nanmin(rr):.4f}")
print(f"R0 (median r >= 0.95 AND min r >= 0.50): "
      f"{'PASS' if (np.nanmedian(rr) >= 0.95 and np.nanmin(rr) >= 0.50) else 'FAIL'}")

# R0b -- tolerant fraction per dose from the full deposited per-cell fits
F3 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 3")
print("\nR0b  tolerant fraction per dose, counted from deposited per-cell fitted values")
counts = {}
for dose in ("25ng", "50ng", "100ng"):
    nR = sio.loadmat(os.path.join(F3, f"FLIP0_values_R_{dose}.mat"))[f"FLIP0_values_R_{dose}"].size
    nS = sio.loadmat(os.path.join(F3, f"FLIP0_values_S_{dose}.mat"))[f"FLIP0_values_S_{dose}"].size
    counts[dose] = (nR, nS)
    print(f"  {dose:6s} tolerant(R)={nR:4d}  sensitive(S)={nS:4d}  total={nR+nS:4d}  "
          f"tolerant fraction={nR/(nR+nS):.4f}")
# consistency: pC80 arrays must have identical lengths to FLIP0 arrays (same cells)
print("\n  consistency check pC80 vs FLIP0 array lengths (same cells, two parameters):")
ok = True
for dose in ("25ng", "50ng", "100ng"):
    for grp in ("R", "S"):
        a = sio.loadmat(os.path.join(F3, f"FLIP0_values_{grp}_{dose}.mat"))[f"FLIP0_values_{grp}_{dose}"].size
        b = sio.loadmat(os.path.join(F3, f"pC80_values_{grp}_{dose}.mat"))[f"pC80_values_{grp}_{dose}"].size
        ok &= (a == b)
        print(f"    {grp} {dose:6s} FLIP0 n={a:4d}  pC80 n={b:4d}  {'match' if a==b else 'MISMATCH'}")
print(f"  all match: {ok}")
json.dump(dict(rows=rows, counts={k: list(v) for k, v in counts.items()}),
          open(os.path.join(ROOT, "data", "R0_results.json"), "w"), indent=1)
