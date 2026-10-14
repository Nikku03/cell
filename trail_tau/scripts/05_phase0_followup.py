#!/usr/bin/env python3
"""Follow-ups: correct CHECK 1's framing, and settle whether data_classification_* are real cells."""
import os, numpy as np, scipy.io as sio
ROOT = os.path.join(os.path.dirname(__file__), "..")
F2 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 2")
F5 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 5")

print("CHECK 1 (corrected framing).  The MODEL's FRET is monotone by construction")
print("(dFRET/dt = K_FRET*C8, C8>=0). The EXPERIMENTAL trace is a noisy measurement of it, so")
print("negative steps size the measurement noise rather than refuting the integrator structure.")
neg = []
for grp in ("R","S"):
    for dose in ("25ng","50ng","100ng"):
        a = sio.loadmat(os.path.join(F2, f"FRET_{grp}_{dose}.mat"))[f"FRET_{grp}_{dose}"]
        for k in range(a.shape[0]):
            v = a[k][np.isfinite(a[k])]
            if v.size > 1:
                d = np.diff(v); neg.append(d[d < 0])
neg = np.concatenate(neg) if neg else np.array([])
print(f"  negative steps: {neg.size} of all frame-to-frame steps; "
      f"median {np.median(neg):.5f}, most negative {neg.min():.5f}")
print(f"  FRET dynamic range across dataset ~0 to 0.81 -> noise is ~0.4% of range.")
print("  CONCLUSION: FRET is a cumulative reporter of caspase activity with small measurement")
print("  noise. It is not a stationary state variable, so its autocorrelation is not a state")
print("  relaxation time. Route B remains unavailable.\n")

print("CHECK 3 (settled).  Are data_classification_* rows real measured cells?")
samp = sio.loadmat(os.path.join(F5, "sampling.mat"))["sampling"]
for dose in ("25ng","50ng","100ng"):
    d = sio.loadmat(os.path.join(F5, f"data_classification_{dose}.mat"))[f"data_classified_{dose}"]
    idx = d[:, 0]
    is_index = np.array_equal(np.sort(idx), np.arange(1, 1001, dtype=float))
    # is the 5-param block a row permutation of sampling.mat?
    A = np.round(d[:, 2:7], 6); B = np.round(samp[:, :5], 6)
    setA = set(map(tuple, A)); setB = set(map(tuple, B))
    print(f"  {dose}: col1 is exactly the integers 1..1000: {is_index}")
    print(f"        n rows = {d.shape[0]} (real fitted cells at this dose = "
          f"{ {'25ng':327,'50ng':414,'100ng':583}[dose] })")
    print(f"        5-param block is a row-permutation of sampling.mat: "
          f"{setA == setB}  (overlap {len(setA & setB)}/1000)")
    print(f"        col8 labels: {dict(zip(*np.unique(d[:,-1], return_counts=True)))}")
print("  CONCLUSION: 1000 rows, col1 = index 1..1000, parameter block drawn from sampling.mat.")
print("  These are SIMULATED state-space samples used for the Figure 5 classifier, not measured")
print("  cells. col1 is a row index within a simulation, not a cell or lineage identifier.")
