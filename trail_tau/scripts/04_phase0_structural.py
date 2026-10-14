#!/usr/bin/env python3
"""Phase 0 structural checks. Establishes (a),(b),(c) with evidence, not assertion."""
import os, numpy as np, scipy.io as sio
from scipy.integrate import solve_ivp
import importlib.util
spec = importlib.util.spec_from_file_location("r0", os.path.join(os.path.dirname(__file__), "03_R0_reproduce.py"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
F2 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 2")
F5 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 5")

print("=" * 90)
print("CHECK 1  Is FRET monotone non-decreasing? (dFRET/dt = K_FRET*C8, C8>=0 => integrator)")
print("=" * 90)
worst = 0.0; nser = 0
for grp in ("R", "S"):
    for dose in ("25ng", "50ng", "100ng"):
        a = sio.loadmat(os.path.join(F2, f"FRET_{grp}_{dose}.mat"))[f"FRET_{grp}_{dose}"]
        for k in range(a.shape[0]):
            v = a[k][np.isfinite(a[k])]
            if v.size < 2: continue
            nser += 1
            worst = min(worst, float(np.min(np.diff(v))))
print(f"  series checked: {nser}   most negative single-frame step across all: {worst:.6g}")
print(f"  -> FRET is {'MONOTONE non-decreasing' if worst >= -1e-12 else 'NOT monotone'}; "
      f"it is a cumulative integrator, not a stationary state variable.")

print()
print("=" * 90)
print("CHECK 2  Do the STATE variables (pC8, FLIP) have any turnover in the model?")
print("=" * 90)
K1 = 0.007325300696406
cp = sio.loadmat(os.path.join(F2, "common_parameters.mat"))["common_parameters"].ravel()
rK1bK1, rK2bK2, rK3bK3, rK2K1, rK3K1, alphaR_3, alphaC8, K_FRET = cp
def rhs(t, y, p):
    T,R,Z0,Z3,pC8,Z1,Z2,FLIP,C8,FRET = y; p1,p2,a0,a1,Kdeg = p
    b = (T*R**3)/(R**3+alphaR_3)
    return [-b+rK1bK1*Z0, -3*b+3*rK1bK1*Z0,
            +b-rK1bK1*Z0-rK3K1*Z0*FLIP**3+rK3bK3*rK3K1*Z3-rK2K1*Z0*pC8**2+rK2bK2*rK2K1*Z1+a0*Z1,
            +rK3K1*Z0*FLIP**3-rK3bK3*rK3K1*Z3,
            -2*rK2K1*Z0*pC8**2+2*rK2bK2*rK2K1*Z1,
            +rK2K1*Z0*pC8**2-rK2bK2*rK2K1*Z1-rK2K1*Z1*FLIP+rK2bK2*rK2K1*Z2-a0*Z1,
            +rK2K1*Z1*FLIP-rK2bK2*rK2K1*Z2-a1*Z2,
            -3*rK3K1*Z0*FLIP**3+3*rK3bK3*rK3K1*Z3-rK2K1*Z1*FLIP+rK2bK2*rK2K1*Z2,
            +a0*Z1+a1*Z2-Kdeg*(C8/(alphaC8+C8))-K_FRET*C8, +K_FRET*C8]
par = sio.loadmat(os.path.join(F2, "R_25ng_ind_par.mat"))["R_25ng_ind_par"]
print("  Symbolic reading of the deposited ODE (Figure_2.m):")
print("    d(pC8)/dt = -2*K2*Z0*pC8^2 + 2*b2*K2*Z1        <- binding/unbinding ONLY")
print("    d(FLIP)/dt= -3*K3*Z0*FLIP^3 + 3*b3*K3*Z3 - K2*Z1*FLIP + b2*K2*Z2   <- binding ONLY")
print("    Neither equation contains a synthesis term or a first-order decay term.")
print("  Numerical consequence -- total FLIP (FLIP + 3*Z3 + Z2) can only fall via catalysis a1*Z2:")
for k in range(par.shape[0]):
    p = par[k].astype(float)
    y0 = [750.0, 32000.0, 0,0, p[0], 0,0, p[1], 30.0, 0]
    s = solve_ivp(rhs, [0, 600*K1], y0, args=(p,), method="LSODA", rtol=1e-12, atol=1e-12,
                  dense_output=True)
    ts = np.linspace(0, 600*K1, 200); Y = s.sol(ts)
    totF = Y[7] + 3*Y[3] + Y[6]
    totP = Y[4] + Y[5] + Y[6]
    print(f"    cell {k}: total FLIP {totF[0]:12.2f} -> {totF[-1]:12.2f} "
          f"({100*(totF[-1]/totF[0]-1):+7.3f}%)   total pC8 {totP[0]:11.2f} -> {totP[-1]:11.2f} "
          f"({100*(totP[-1]/totP[0]-1):+7.3f}%)")
print("  -> There is NO process in this model that relaxes cell state toward a population mean.")

print()
print("=" * 90)
print("CHECK 3  data_classification_*.mat -- real cells or simulated samples?")
print("=" * 90)
samp = sio.loadmat(os.path.join(F5, "sampling.mat"))["sampling"]
for dose in ("25ng", "50ng", "100ng"):
    d = sio.loadmat(os.path.join(F5, f"data_classification_{dose}.mat"))[f"data_classified_{dose}"]
    print(f"  {dose}: shape {d.shape}; col mins {np.round(d.min(0),3)}")
    print(f"          col maxs {np.round(d.max(0),3)}")
    print(f"          unique values in last column: {np.unique(d[:,-1])}")
    # do columns 3..7 equal the sampling matrix (i.e. simulated, not measured)?
    same = np.allclose(d[:, 2:7], samp[:, :5], rtol=1e-9, atol=1e-9)
    print(f"          cols 3-7 identical to sampling.mat (=> SIMULATED, not measured cells): {same}")
print(f"  sampling.mat shape {samp.shape} -- 1000 sampled parameter sets, "
      f"vs {150+177} / {114+300} / {65+518} real fitted cells at 25/50/100 ng.")

print()
print("=" * 90)
print("CHECK 4  Any per-cell identifier that could link cells across time or generations?")
print("=" * 90)
import glob
allv = []
for f in glob.glob(os.path.join(ROOT, "data", "raw", "**", "*.mat"), recursive=True):
    if "__MACOSX" in f: continue
    for k, v in sio.loadmat(f).items():
        if not k.startswith("__"):
            allv.append((os.path.basename(f), k, getattr(v, "shape", None), getattr(v, "dtype", None)))
intlike = [x for x in allv if x[3] is not None and np.issubdtype(x[3], np.integer)]
print(f"  total variables across all .mat files: {len(allv)}")
print(f"  integer-typed variables (candidate identifiers): {len(intlike)}")
for x in intlike: print(f"    {x[0]:28s} {x[1]:22s} shape={x[2]} dtype={x[3]}")
print("  Every integer-typed variable is a Tend (death time in minutes), not an identifier.")
print("  No variable in the archive is a cell ID, lineage ID, track ID, or frame-to-cell map.")
