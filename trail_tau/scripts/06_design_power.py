#!/usr/bin/env python3
"""What it would take to MEASURE tau. Sample size computed, not asserted.

The point of this script: the required n can be stated WITHOUT knowing tau, provided the
observation design is expressed in units of tau. Gate P3 tests exactly that invariance -- if it
fails, no tau-free sample size may be quoted.

State scale is taken from the REAL deposited per-cell fits (sd of log FLIP0 / log pC80), not
invented. Measurement noise is unknown, so it is SWEPT and reported, never assumed.
"""
import os, numpy as np, scipy.io as sio
from scipy.optimize import curve_fit

ROOT = os.path.join(os.path.dirname(__file__), "..")
F3 = os.path.join(ROOT, "data", "raw", "Codes for Figures", "Figure 3")
rng_global = np.random.default_rng(20260903)

# ---- real state spread, from the deposited fits -------------------------------------------
vals = {}
for var in ("FLIP0", "pC80"):
    allv = []
    for dose in ("25ng", "50ng", "100ng"):
        for grp in ("R", "S"):
            k = f"{var}_values_{grp}_{dose}"
            allv.append(sio.loadmat(os.path.join(F3, k + ".mat"))[k].ravel())
    v = np.concatenate(allv); vals[var] = v
    print(f"  {var}: n={v.size}, sd(log10)={np.std(np.log10(v)):.4f}, "
          f"range {v.min():.4g}-{v.max():.4g}")
print()

def fit_tau(t, r, p0):
    """rho(t) = rho0*exp(-t/tau). Returns tau or nan."""
    try:
        f = lambda x, rho0, tau: rho0 * np.exp(-x / tau)
        popt, _ = curve_fit(f, t, r, p0=p0, maxfev=20000,
                            bounds=([0.0, 1e-3], [1.5, 1e4]))
        return float(popt[1])
    except Exception:
        return np.nan

def routeA_once(n_pairs, tau_true, noise_frac, rng, n_bins=6, window_mult=2.0):
    """n_pairs sister pairs, separations uniform on [0, window_mult*tau]. One estimate."""
    t = rng.uniform(0, window_mult * tau_true, n_pairs)
    rho = np.exp(-t / tau_true)
    z1 = rng.normal(size=n_pairs)
    z2 = rho * z1 + np.sqrt(np.maximum(1 - rho**2, 0)) * rng.normal(size=n_pairs)
    x1 = z1 + noise_frac * rng.normal(size=n_pairs)
    x2 = z2 + noise_frac * rng.normal(size=n_pairs)
    edges = np.linspace(0, window_mult * tau_true, n_bins + 1)
    tb, rb = [], []
    for i in range(n_bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() >= 8 and x1[m].std() > 0 and x2[m].std() > 0:
            tb.append(t[m].mean()); rb.append(np.corrcoef(x1[m], x2[m])[0, 1])
    if len(tb) < 3:
        return np.nan
    return fit_tau(np.array(tb), np.array(rb), p0=[1.0, tau_true])

def ci_factor(n_pairs, tau_true, noise_frac, reps, rng):
    est = np.array([routeA_once(n_pairs, tau_true, noise_frac, rng) for _ in range(reps)])
    est = est[np.isfinite(est)]
    if est.size < reps * 0.5:
        return np.nan, np.nan, est.size
    lo, hi = np.percentile(est, [2.5, 97.5])
    return hi / lo, float(np.median(est)), est.size

print("=" * 92)
print("ROUTE A DESIGN  (sister pairs; each cell's state measured ONCE, destructively, by dosing)")
print("=" * 92)
print("Separations drawn uniformly on [0, 2*tau]. 'CI factor' = ratio of 97.5th to 2.5th")
print("percentile of the tau estimate across independent simulated experiments.\n")

REPS = 400
print(f"{'n pairs':>8s} | " + " | ".join(f"noise {f:.0%}".rjust(22) for f in (0.0, 0.10, 0.25)))
print(f"{'':>8s} | " + " | ".join("CI factor   median tau".rjust(22) for _ in range(3)))
results = {}
for n in (25, 50, 100, 200, 400, 800, 1600):
    row = []
    for nf in (0.0, 0.10, 0.25):
        rng = np.random.default_rng(1000 + n * 7 + int(nf * 100))
        cf, med, k = ci_factor(n, 3.0, nf, REPS, rng)
        results[(n, nf)] = (cf, med, k)
        row.append(f"{cf:9.3f} {med:12.3f}".rjust(22))
    print(f"{n:8d} | " + " | ".join(row))

print("\nGATE P1  estimator unbiased at large n (median within 5% of true tau=3.0, noise 0):")
cf, med, k = results[(1600, 0.0)]
print(f"  median tau at n=1600 = {med:.4f} vs true 3.0 -> "
      f"{'PASS' if abs(med/3.0 - 1) < 0.05 else 'FAIL'} (rel dev {abs(med/3.0-1):.4f})")

print("\nGATE P2  CI width must shrink as 1/sqrt(n) (log-log slope -0.5 +/- 0.1):")
ns = np.array([25, 50, 100, 200, 400, 800, 1600], float)
w = np.array([results[(int(n), 0.0)][0] - 1.0 for n in ns])   # width above 1
good = np.isfinite(w) & (w > 0)
slope = np.polyfit(np.log(ns[good]), np.log(w[good]), 1)[0]
print(f"  slope = {slope:.4f} -> {'PASS' if abs(slope + 0.5) < 0.1 else 'FAIL'} (bar -0.5 +/- 0.1)")
if abs(slope + 0.5) >= 0.1:
    print("  NOTE: a power curve that does not scale as 1/sqrt(n) is a suspected BUG, not biology.")

print("\nGATE P3  required n invariant to the true tau when the design is scaled in units of tau.")
print("  (This is what licenses quoting a sample size without knowing the answer.)")
print(f"  {'true tau (h)':>13s} {'CI factor at n=400':>20s}")
p3 = []
for tt in (1.0, 3.0, 10.0):
    rng = np.random.default_rng(555 + int(tt * 10))
    cf, med, k = ci_factor(400, tt, 0.10, REPS, rng)
    p3.append(cf)
    print(f"  {tt:13.1f} {cf:20.3f}")
sp = (max(p3) - min(p3)) / np.mean(p3)
print(f"  relative spread across tau = {sp:.4f} -> "
      f"{'PASS' if sp < 0.15 else 'FAIL'} (bar 15%)")

print("\nREQUIRED SAMPLE SIZE (interpolated from the sweep, noise 10% of state sd):")
ns2 = np.array([25, 50, 100, 200, 400, 800, 1600], float)
cfs = np.array([results[(int(n), 0.10)][0] for n in ns2])
m = np.isfinite(cfs)
for target in (5.0, 2.0, 1.5):
    lg = np.interp(np.log(target - 1.0), np.log(cfs[m] - 1.0)[::-1], np.log(ns2[m])[::-1])
    print(f"  CI spanning a factor of {target:.1f}  ->  n ~= {int(np.exp(lg)):5d} sister pairs")
