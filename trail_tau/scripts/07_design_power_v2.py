#!/usr/bin/env python3
"""Route A sample size, corrected.

CORRECTION 1 (gate statistic was wrong). P2 measured CI width as (CI_factor - 1). tau is a
SCALE parameter; its CI width on the natural (log) scale is log(CI_factor). Using the wrong
one made the log-log slope read -2.46 against a -0.5 bar. The bar was right; the statistic
was wrong.

CORRECTION 2 (estimator was binning-limited). The binned correlation estimator needs >=8 pairs
per bin, so at n=25 fewer than 3 bins survive and it returns nan -- a property of my binning
choice, not of the information in the data. An unbinned maximum-likelihood estimator is added
and both are reported, because the required sample size depends on the estimator and that is a
modelling choice that must be recorded (rule 7).
"""
import os, numpy as np, scipy.io as sio
from scipy.optimize import curve_fit, minimize

def loglik(params, t, x1, x2):
    rho0, tau = params
    if not (0 < rho0 <= 1.5) or tau <= 0: return 1e12
    r = np.clip(rho0 * np.exp(-t / tau), -0.999, 0.999)
    om = 1 - r**2
    return float(np.sum(0.5*np.log(om) + (x1**2 - 2*r*x1*x2 + x2**2)/(2*om)))

def simulate(n_pairs, tau_true, noise_frac, rng, window_mult=2.0):
    t = rng.uniform(0, window_mult*tau_true, n_pairs)
    rho = np.exp(-t/tau_true)
    z1 = rng.normal(size=n_pairs)
    z2 = rho*z1 + np.sqrt(np.maximum(1-rho**2,0))*rng.normal(size=n_pairs)
    x1 = z1 + noise_frac*rng.normal(size=n_pairs)
    x2 = z2 + noise_frac*rng.normal(size=n_pairs)
    x1 = (x1-x1.mean())/x1.std(); x2 = (x2-x2.mean())/x2.std()   # standardise, as one must
    return t, x1, x2

def est_ml(t, x1, x2, tau0):
    best, bv = np.nan, np.inf
    for s in (0.3*tau0, tau0, 3*tau0):
        r = minimize(loglik, [0.9, s], args=(t,x1,x2), method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
        if r.fun < bv: bv, best = r.fun, r.x[1]
    return float(best)

def est_binned(t, x1, x2, tau_true, n_bins=6, window_mult=2.0):
    edges = np.linspace(0, window_mult*tau_true, n_bins+1); tb, rb = [], []
    for i in range(n_bins):
        m = (t>=edges[i]) & (t<edges[i+1])
        if m.sum() >= 8 and x1[m].std()>0 and x2[m].std()>0:
            tb.append(t[m].mean()); rb.append(np.corrcoef(x1[m],x2[m])[0,1])
    if len(tb) < 3: return np.nan
    try:
        f = lambda x, r0, tau: r0*np.exp(-x/tau)
        p,_ = curve_fit(f, np.array(tb), np.array(rb), p0=[1.0,tau_true],
                        maxfev=20000, bounds=([0,1e-3],[1.5,1e4]))
        return float(p[1])
    except Exception:
        return np.nan

def sweep(estimator, n, tau_true, nf, reps, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        t,x1,x2 = simulate(n, tau_true, nf, rng)
        out.append(estimator(t,x1,x2))
    e = np.array(out); e = e[np.isfinite(e) & (e>0)]
    if e.size < reps*0.5: return np.nan, np.nan, e.size
    lo,hi = np.percentile(e,[2.5,97.5])
    return hi/lo, float(np.median(e)), e.size

REPS, TAU = 400, 3.0
NS = [25, 50, 100, 200, 400, 800, 1600]
print("Route A required sample size. CI factor = 97.5th/2.5th percentile of the tau estimate.")
print("Measurement noise expressed as a fraction of the state's standard deviation.\n")
print(f"{'n pairs':>8s} {'ML CI':>9s} {'ML med':>9s} {'binned CI':>10s} {'bin med':>9s} {'n_ok':>6s}")
res_ml, res_bin = {}, {}
for n in NS:
    a = sweep(lambda t,x,y: est_ml(t,x,y,TAU), n, TAU, 0.10, REPS, 900+n)
    b = sweep(lambda t,x,y: est_binned(t,x,y,TAU), n, TAU, 0.10, REPS, 900+n)
    res_ml[n], res_bin[n] = a, b
    print(f"{n:8d} {a[0]:9.3f} {a[1]:9.3f} {b[0]:10.3f} {b[1]:9.3f} {a[2]:6d}")

print("\nGATE P1  ML estimator unbiased at large n (median within 5% of true tau = 3.0):")
print(f"  n=1600 median {res_ml[1600][1]:.4f} -> "
      f"{'PASS' if abs(res_ml[1600][1]/TAU-1)<0.05 else 'FAIL'} "
      f"(rel dev {abs(res_ml[1600][1]/TAU-1):.4f})")

print("\nGATE P2 (corrected statistic)  log(CI factor) must scale as n^-0.5:")
for name, res in (("ML", res_ml), ("binned", res_bin)):
    ns = np.array([n for n in NS if np.isfinite(res[n][0])], float)
    w  = np.log(np.array([res[n][0] for n in ns.astype(int)]))
    m = w > 0
    sl = np.polyfit(np.log(ns[m]), np.log(w[m]), 1)[0]
    print(f"  {name:7s} slope = {sl:+.4f}  over n = {ns[m].astype(int).tolist()}  -> "
          f"{'PASS' if abs(sl+0.5)<0.1 else 'FAIL'} (bar -0.5 +/- 0.1)")
    if name=="binned":
        drop=[n for n in NS if not np.isfinite(res[n][0])]
        print(f"          excluded (estimator returned nan, too few pairs per bin): {drop}")

print("\nGATE P3  invariance of required n to the true tau, design scaled in units of tau:")
p3=[]
for tt in (1.0,3.0,10.0):
    cf,med,k = sweep(lambda t,x,y: est_ml(t,x,y,tt), 400, tt, 0.10, REPS, 77+int(tt*10))
    p3.append(cf); print(f"  true tau {tt:5.1f} h -> CI factor {cf:.3f} at n=400, median {med:.3f}")
sp=(max(p3)-min(p3))/np.mean(p3)
print(f"  relative spread {sp:.4f} -> {'PASS' if sp<0.15 else 'FAIL'} (bar 15%)")

print("\nGATE P4  does measurement noise bias tau, or only rho_0?")
for nf in (0.0,0.10,0.25,0.50):
    cf,med,k = sweep(lambda t,x,y: est_ml(t,x,y,TAU), 800, TAU, nf, REPS, 313+int(nf*100))
    print(f"  noise {nf:4.0%} of state sd -> median tau {med:.4f} (true {TAU}), CI factor {cf:.3f}")
print("  Expectation stated in advance: noise attenuates rho_0 but leaves tau unbiased,")
print("  because it is independent between the two sisters and does not depend on t.")

print("\nREQUIRED SAMPLE SIZE (ML estimator, noise 10% of state sd):")
ns = np.array(NS,float); cf = np.array([res_ml[n][0] for n in NS])
m=np.isfinite(cf)&(cf>1)
for target in (5.0,2.0,1.5):
    lg=np.interp(np.log(np.log(target)), np.log(np.log(cf[m]))[::-1], np.log(ns[m])[::-1])
    print(f"  CI spanning a factor of {target:.1f}  ->  n ~= {int(round(np.exp(lg))):5d} sister pairs")
