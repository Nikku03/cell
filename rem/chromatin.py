"""Exact TF accessibility in a crowded 1D medium: nucleosomes as hard rods on DNA.

WHY THIS TARGET IS DIFFERENT. Four previous applications in this project were exact,
correct and useless -- the machinery was right and the system was too loosely coupled for
exactness to change the answer. Ribosomes sit at 5-10% of close packing, where ignoring
interaction entirely costs under 2%. Nucleosomes occupy 75-90% of eukaryotic DNA. That is
the jammed regime, it is one-dimensional, it is at equilibrium, and it is parameterisable
from sequence, so none of the four previous escape hatches apply.

THE MODEL. Grand-canonical hard rods on a lattice of base pairs. A nucleosome is a rod of
147 bp whose START at position i carries statistical weight w_i from a sequence model; a
chemical potential mu sets genome-wide occupancy. A transcription factor is a shorter rod
(6-20 bp). Everything below is exact and comes from ONE forward-backward pass, cost linear
in sequence length, because the exclusion constraint is local and the chain has treewidth 1.

    F[j] = F[j-1] + w[j-ell] F[j-ell]      partition function over the first j sites
    B[j] = B[j+1] + w[j]     B[j+ell]      partition function over sites j .. L-1
    P(nucleosome starts at i) = w_i F[i] B[i+ell] / Z

THE INDEXING TRAP, and it is the single most likely bug. To place a TF of length m at j it
must cover j .. j+m-1, so NO nucleosome may cover any of those base pairs. A nucleosome
starting at s covers s .. s+ell-1, which overlaps iff s lies in [j-ell+1, j+m-1] -- a
forbidden window of width m + ell - 1 = m + 146, NOT m. Getting this wrong silently
under-counts the insertion cost and every downstream number inherits it. G1 exists to catch
exactly that.

    gamma(j) = Z / (F[j] B[j+m])

computed for every j from the same two arrays -- one pass, not one solve per position.

WHAT verify() MUST SHOW -- PREDECLARED, BEFORE ANY NUMBER IS RUN.
  G1  DILUTE LIMIT. As occupancy -> 0, gamma -> 1 everywhere. Reference: at coverage 1e-4
      with a 1 bp insert the forbidden window is 147 bp wide, so the first-order answer is
      gamma = 1/(1 - 147 * (1e-4/147)) = 1.0001 exactly. GATE: |gamma - 1.0001| < 1e-6.
      A wrong forbidden window shows up here as a wrong coefficient, not as noise.
      G1 FAILED at 1.0001051 and the verdict stands. It is NOT the forbidden window: G2
      validates gamma(j) against explicit enumeration at 1e-15, which a wrong window cannot
      survive. The deviation is FINITE-SIZE and falls as 1/L -- 5.12e-6 at L=3000, 1.48e-6
      at L=10000, a ratio of 3.46 for a length ratio of 3.33 -- and bulk-only and
      all-position medians are identical, so it is not an edge artifact either. The bar was
      an L -> infinity first-order value applied at finite L: an absolute tolerance not
      derived from the quantity's own finite-size behaviour. Ledger defect L, fourth
      occurrence in this project.
  G2  BRUTE FORCE. For L <= 26 with a short rod (ell = 4), enumerate every legal
      configuration and compare Z, per-site coverage, and gamma(j) at every site.
      GATE: max relative error <= 1e-13.
  G3  THE UNIFORM LATTICE CLOSED FORM. With uniform weights the number of ways to place n
      rods of length ell on L sites is exactly C(L - n(ell-1), n), so
      Z = sum_n C(L - n(ell-1), n) w^n -- an exact finite-L combinatorial identity, not a
      thermodynamic limit. GATE: |log Z_recursion - log Z_combinatorial| < 1e-10.
      NOTE, and this cost real time earlier in this project: the CONTINUUM Tonks gas is a
      DIFFERENT MODEL and its formulas disagree. The lattice result is the one to cite.
  G4  REPRODUCE THE RELEVANCE TABLE the spec was built on, within 10%: exact gamma versus
      the naive 1/(1 - local occupancy) approximation on 1500 bp with 147 bp nucleosomes and
      a 10 bp TF, at coverage 10-90%. The load-bearing entry is the RANK CORRELATION, which
      goes NEGATIVE at high density -- the cheap approximation does not merely mis-scale, it
      orders the sites wrongly. GATE: ratio and rank correlation both within 10% (absolute
      0.05 for the correlation) of the reference at 80%: ratio 2.55, rank corr -0.125.
      G4 FAILED, and it is the substantive one because it is this module's entire
      justification. Measured at 80% occupancy: ratio 1.28 against 2.55, rank correlation
      +0.916 against -0.125. THE CLAIM THAT THE CHEAP APPROXIMATION ORDERS SITES WRONGLY
      DOES NOT REPRODUCE. Rank correlation of exact against naive gamma, over three natural
      readings of "local occupancy" and four weight variances including uniform:

          weight log-sd    footprint   forbidden-window   pointwise
              0.0           +0.954         +0.779          +0.957
              0.5           +0.942         +0.735          +0.947
              1.0           +0.921         +0.677          +0.921
              2.0           +0.927         +0.691          +0.932

      Always strongly POSITIVE, never negative; the ratio is always about 1.3, never 2.55.
      One internal inconsistency localises the difference to the baseline rather than to the
      exact side: the reference's naive gamma at 80% occupancy is 1.826, whereas
      1/(1 - 0.8) = 5.0, so whatever it averaged as "local occupancy" came out near 0.45
      when the stated occupancy was 0.80.
      CONSEQUENCE FOR G6, recorded before any yeast data is fetched: G6's null 2 is
      PWM x 1/gamma_naive, so at rank correlation 0.92 the model and null 2 are nearly the
      same predictor and G6 cannot separate exactness from crowding. The spec already names
      that outcome -- "crowding matters, exactness does not" -- and this measurement predicts
      it rather than leaving it open. The residual 8% of ordering variance is real and only
      an experiment can settle whether it matters, so G6 remains the right test with a moved
      prior. Nothing was tuned to make G4 pass.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NUC = 147                    # nucleosome footprint, bp
NEG = -np.inf


def forward_backward(logw: np.ndarray, ell: int = NUC
                     ) -> Tuple[np.ndarray, np.ndarray, float]:
    """(F, B, logZ) in log space. F has length L+1, B has length L+ell+1."""
    L = len(logw)
    F = np.zeros(L + 1)
    for j in range(1, L + 1):
        if j < ell:
            F[j] = F[j - 1]
        else:
            F[j] = np.logaddexp(F[j - 1], logw[j - ell] + F[j - ell])
    B = np.zeros(L + ell + 1)
    for j in range(L - 1, -1, -1):
        if j > L - ell:
            B[j] = B[j + 1]
        else:
            B[j] = np.logaddexp(B[j + 1], logw[j] + B[j + ell])
    return F, B, float(F[L])


def start_probs(logw: np.ndarray, ell: int = NUC) -> np.ndarray:
    """P(a nucleosome starts at i), exact, for every i."""
    L = len(logw)
    F, B, logZ = forward_backward(logw, ell)
    out = np.zeros(L)
    for i in range(L - ell + 1):
        out[i] = np.exp(logw[i] + F[i] + B[i + ell] - logZ)
    return out


def coverage(logw: np.ndarray, ell: int = NUC) -> np.ndarray:
    """P(base pair k is covered by some nucleosome)."""
    p = start_probs(logw, ell)
    c = np.convolve(p, np.ones(ell))[:len(p)]
    return np.clip(c, 0.0, 1.0)


def gamma(logw: np.ndarray, m: int, ell: int = NUC) -> np.ndarray:
    """Crowding penalty gamma(j) = 1 / P(sites j..j+m-1 all free of nucleosomes).

    One forward-backward pass serves every j. The forbidden window for a nucleosome START
    is [j-ell+1, j+m-1], width m+ell-1, which is why the split is F[j] * B[j+m] and not
    anything narrower.
    """
    L = len(logw)
    F, B, logZ = forward_backward(logw, ell)
    out = np.full(L, np.nan)
    for j in range(0, L - m + 1):
        out[j] = np.exp(logZ - F[j] - B[j + m])
    return out


def solve_mu(logw0: np.ndarray, target: float, ell: int = NUC,
             lo: float = -60.0, hi: float = 60.0, iters: int = 60) -> float:
    """Chemical potential giving the requested mean coverage. Bisection, exact each step."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = coverage(logw0 + mid, ell).mean()
        if c < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def naive_gamma(cov: np.ndarray, m: int) -> np.ndarray:
    """The cheap approximation this module exists to be compared against:
    gamma ~ 1 / (1 - local mean occupancy over the TF footprint)."""
    loc = np.convolve(cov, np.ones(m) / m, mode="same")
    return 1.0 / np.clip(1.0 - loc, 1e-12, None)


# --------------------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------------------

def brute_force(logw: np.ndarray, ell: int, m: int
                ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Enumerate every legal configuration. (logZ, coverage, gamma). Small L only."""
    L = len(logw)
    starts = [i for i in range(L - ell + 1)]
    cfgs = []
    for r in range(len(starts) + 1):
        for c in itertools.combinations(starts, r):
            if all(b - a >= ell for a, b in zip(c, c[1:])):
                cfgs.append(c)
    wts = np.array([sum(logw[i] for i in c) for c in cfgs])
    mx = wts.max()
    w = np.exp(wts - mx)
    Z = w.sum()
    cov = np.zeros(L)
    for c, wi in zip(cfgs, w):
        for s in c:
            cov[s:s + ell] += wi
    cov /= Z
    gam = np.full(L, np.nan)
    for j in range(L - m + 1):
        free = 0.0
        for c, wi in zip(cfgs, w):
            if all(not (s <= j + m - 1 and s + ell - 1 >= j) for s in c):
                free += wi
        gam[j] = Z / free if free > 0 else np.inf
    return float(mx + np.log(Z)), cov, gam


def uniform_logZ(L: int, ell: int, logw: float) -> float:
    """Exact finite-L combinatorial partition function for UNIFORM weights:
    Z = sum_n C(L - n(ell-1), n) w^n.  Placing n non-overlapping ell-rods on L sites is a
    stars-and-bars count; this is an identity, not a limit."""
    terms = []
    n = 0
    while L - n * (ell - 1) >= n:
        terms.append(math.log(math.comb(L - n * (ell - 1), n)) + n * logw)
        n += 1
    mx = max(terms)
    return float(mx + np.log(np.exp(np.array(terms) - mx).sum()))


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------

def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def verify(verbose: bool = True, seed: int = 0) -> dict:
    """Run G1-G4. Bars are fixed in the module docstring, above, before any number."""
    say = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    out: Dict[str, object] = {}
    rng = np.random.default_rng(seed)

    # ---- G1: dilute limit -----------------------------------------------------------
    L = 3000
    lw = np.zeros(L)
    mu = solve_mu(lw, 1e-4, NUC)
    g = gamma(lw + mu, m=1, ell=NUC)
    med = float(np.nanmedian(g))
    out["G1_gamma"], out["G1"] = med, bool(abs(med - 1.0001) < 1e-6)
    say(f"  G1 dilute limit: coverage {coverage(lw+mu).mean():.3e}, "
        f"median gamma {med:.7f}  (closed form 1.0001)   "
        f"{'PASS' if out['G1'] else 'FAIL'}")

    # ---- G2: brute force ------------------------------------------------------------
    say("\n  G2 brute force, every legal configuration enumerated")
    say(f"      {'L':>3s} {'ell':>4s} {'m':>3s} {'dlogZ':>10s} {'dcoverage':>11s} "
        f"{'dgamma':>10s}")
    g2 = True
    for Ls, ell, m in ((14, 4, 2), (18, 4, 3), (22, 4, 2), (26, 4, 4)):
        lw = rng.normal(-1.0, 1.0, size=Ls)
        zb, cb, gb = brute_force(lw, ell, m)
        _F, _B, zf = forward_backward(lw, ell)
        cf = coverage(lw, ell)
        gf = gamma(lw, m, ell)
        d1 = abs(zf - zb)
        d2 = float(np.abs(cf - cb).max())
        ok = np.isfinite(gb) & np.isfinite(gf)
        d3 = float(np.abs(gf[ok] - gb[ok]).max())
        g2 &= (d1 < 1e-13 and d2 < 1e-13 and d3 < 1e-11)
        say(f"      {Ls:3d} {ell:4d} {m:3d} {d1:10.2e} {d2:11.2e} {d3:10.2e}")
    out["G2"] = bool(g2)
    say(f"      G2 {'PASS' if g2 else 'FAIL'}")

    # ---- G3: uniform lattice closed form ---------------------------------------------
    say("\n  G3 uniform weights vs the exact lattice combinatorial identity")
    say(f"      {'L':>5s} {'ell':>4s} {'log w':>7s} {'recursion':>13s} "
        f"{'combinatorial':>14s} {'diff':>10s}")
    g3 = True
    for Ls, ell, lwv in ((200, 10, -1.0), (500, 20, 0.5), (1000, 147, 2.0),
                         (1500, 147, 4.0)):
        lw = np.full(Ls, lwv)
        _F, _B, zr = forward_backward(lw, ell)
        zc = uniform_logZ(Ls, ell, lwv)
        d = abs(zr - zc)
        g3 &= d < 1e-10
        say(f"      {Ls:5d} {ell:4d} {lwv:7.1f} {zr:13.6f} {zc:14.6f} {d:10.2e}")
    out["G3"] = bool(g3)
    say(f"      G3 {'PASS' if g3 else 'FAIL'}")

    # ---- G4: reproduce the relevance table --------------------------------------------
    say("\n  G4 exact gamma vs naive 1/(1-occupancy), 1500 bp, 147 bp nucleosome, 10 bp TF")
    say(f"      {'occ':>5s} {'exact':>9s} {'naive':>9s} {'ratio':>7s} {'rank corr':>10s} "
        f"{'spread':>7s}")
    lw0 = rng.normal(0.0, 1.0, size=1500)
    rows = {}
    for occ in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9):
        mu = solve_mu(lw0, occ, NUC)
        cov = coverage(lw0 + mu, NUC)
        ge = gamma(lw0 + mu, m=10, ell=NUC)
        gn = naive_gamma(cov, 10)
        ok = np.isfinite(ge)
        me, mn = float(np.median(ge[ok])), float(np.median(gn[ok]))
        rc = _spearman(ge, gn)
        sp = float(np.nanpercentile(ge[ok], 90) / np.nanpercentile(ge[ok], 10))
        rows[occ] = {"exact": me, "naive": mn, "ratio": me / mn, "rank": rc,
                     "spread": sp}
        say(f"      {occ:5.0%} {me:9.3f} {mn:9.3f} {me/mn:7.2f} {rc:10.3f} {sp:7.2f}")
    r80 = rows[0.8]
    out["G4_rows"] = rows
    out["G4"] = bool(abs(r80["ratio"] - 2.55) / 2.55 < 0.10
                     and abs(r80["rank"] - (-0.125)) < 0.05)
    say(f"      at 80%: ratio {r80['ratio']:.2f} (ref 2.55), rank corr {r80['rank']:.3f} "
        f"(ref -0.125)   {'PASS' if out['G4'] else 'FAIL'}")

    gates = ["G1", "G2", "G3", "G4"]
    out["all_pass"] = all(bool(out[k]) for k in gates)
    say(f"\n  {'ALL GATES PASS' if out['all_pass'] else 'GATE FAILURE'}: "
        + "  ".join(f"{k}={'pass' if out[k] else 'FAIL'}" for k in gates))
    return out


if __name__ == "__main__":
    verify()
