"""FOLDSIM -- a simulator of chromosome folding built from the rules tested in G19 / G20: put motors (cohesin) and stop
signs (CTCF sites with a direction) on a line of DNA, let the motors reel loops, and read off which sites loop and how
often any two places touch. No measured contact is used to make a prediction: the inputs are the stop-sign map, four
numbers fitted once (foldsimtest.py, on DEV regions) and, optionally, where activity is (for loading motors) and which
zone each stretch is in.

The model (1D loop extrusion; Fudenberg et al. 2016 and Sanborn et al. 2015 in spirit, written from scratch):
  * DNA = a row of 5 kb sites. A motor has a left and a right leg on neighbouring sites when it lands.
  * Every step each leg moves one site outward (the loop grows 10 kb) unless the next site holds another leg (legs do
    not pass each other) or the leg is stalled.
  * A leg moving LEFT stalls when it enters a site with a + stop sign, a leg moving RIGHT at a - stop sign, each with
    the site's capture probability (so a loop ends at two signs facing each other). A stalled leg stays until its
    motor falls off.
  * A motor falls off with probability 1 / lifetime per step and lands again at a random free place (optionally more
    often where activity is high).
  * Contact map: in each snapshot the DNA is a chain where each motor ties its two legs together; the 'effective
    distance' between two bins is the shortest path along chain and ties; contact = (distance + 1) ^ -alpha, averaged
    over snapshots. Zones: optional factor exp(kappa x zone_i x zone_j) (like with like).
Parameters: processivity (kb a motor reels before falling off, = 2 x 5 kb x lifetime), separation (kb of DNA per
motor), capture (probability scale of a stop sign: capture x strength rank), alpha.

Use as a library:  S = foldsim.Sim(n, p_left, p_right, n_motors, lifetime, rng=...); S.run(...); or foldsim.chromosome(...)
Demo:              python3 -I rem/atlas/foldsim.py demo
"""

from __future__ import annotations
import sys

import numpy as np

SITE = 5000


class Sim:
    """R independent copies of a lattice of n sites with motors; vectorised over copies."""
    def __init__(self, n, p_left, p_right, n_motors, lifetime, load=None, R=10, rng=None, release=0.0):
        self.n, self.R, self.nc, self.tau = int(n), int(R), int(max(n_motors, 0)), float(lifetime); self.release = float(release)
        self.pl, self.pr = np.asarray(p_left, float), np.asarray(p_right, float); self.rng = rng or np.random.default_rng(0)
        w = np.ones(self.n - 1) if load is None else np.maximum(np.asarray(load, float)[:-1], 0) + 1e-12
        self.cum = np.cumsum(w / w.sum())
        self.occ = np.zeros((self.R, self.n), bool); self.rep = np.repeat(np.arange(self.R)[:, None], self.nc, 1)
        self.l = np.zeros((self.R, self.nc), np.int64); self.r = np.ones((self.R, self.nc), np.int64)
        self.sl = np.zeros((self.R, self.nc), bool); self.sr = np.zeros((self.R, self.nc), bool); self.on = np.zeros((self.R, self.nc), bool)
        self._land(np.ones((self.R, self.nc), bool))

    def _land(self, todo):
        for _ in range(20):                                    # all motors that need a place try at once; clashes try again
            rr, kk = np.nonzero(todo)
            if not len(rr):
                break
            x = np.minimum(np.searchsorted(self.cum, self.rng.random(len(rr))), self.n - 2)
            ok = ~self.occ[rr, x] & ~self.occ[rr, x + 1]; o = np.lexsort((x, rr)); rs, xs = rr[o], x[o]
            clash = np.zeros(len(o), bool); clash[1:] = (rs[1:] == rs[:-1]) & (xs[1:] - xs[:-1] <= 1); ok[o[clash]] = False
            rr, kk, x = rr[ok], kk[ok], x[ok]
            self.occ[rr, x] = True; self.occ[rr, x + 1] = True; self.l[rr, kk] = x; self.r[rr, kk] = x + 1
            self.sl[rr, kk] = False; self.sr[rr, kk] = False; self.on[rr, kk] = True; todo[rr, kk] = False

    def step(self):
        if not self.nc:
            return
        off = self.on & (self.rng.random((self.R, self.nc)) < 1.0 / self.tau)
        if off.any():
            self.occ[self.rep[off], self.l[off]] = False; self.occ[self.rep[off], self.r[off]] = False; self.on[off] = False
        todo = ~self.on
        if todo.any():
            self._land(todo)
        if self.release > 0:                                  # leaky stop signs: a stalled leg may start moving again
            self.sl &= self.rng.random((self.R, self.nc)) >= self.release; self.sr &= self.rng.random((self.R, self.nc)) >= self.release
        t = np.minimum(self.r + 1, self.n - 1); can = self.on & ~self.sr & (self.r + 1 < self.n) & ~self.occ[self.rep, t]
        self.occ[self.rep[can], self.r[can]] = False; self.r[can] += 1; self.occ[self.rep[can], self.r[can]] = True
        self.sr[can] = self.rng.random(int(can.sum())) < self.pr[self.r[can]]
        t = np.maximum(self.l - 1, 0); can = self.on & ~self.sl & (self.l - 1 >= 0) & ~self.occ[self.rep, t]
        self.occ[self.rep[can], self.l[can]] = False; self.l[can] -= 1; self.occ[self.rep[can], self.l[can]] = True
        self.sl[can] = self.rng.random(int(can.sum())) < self.pl[self.l[can]]

    def run(self, burn, snaps, every):
        """returns legs of every snapshot: arrays (snaps x R, n_motors) of left and right positions (-1 where no motor)."""
        for _ in range(int(burn)):
            self.step()
        L, Rr = [], []
        for _ in range(int(snaps)):
            for _ in range(int(every)):
                self.step()
            L.append(np.where(self.on, self.l, -1)); Rr.append(np.where(self.on, self.r, -1))
        return (np.concatenate(L), np.concatenate(Rr)) if L else (np.zeros((0, self.nc), int), np.zeros((0, self.nc), int))


def capture_p(n, pos, strand, p):
    """capture probabilities per lattice site: pos (bp), strand (+1, -1, or 0 = holds legs from both sides), probability p."""
    pl, pr = np.zeros(n), np.zeros(n); x = np.minimum(np.asarray(pos) // SITE, n - 1); p = np.clip(np.asarray(p, float), 0, 1); st = np.asarray(strand)
    for arr, m in ((pl, st >= 0), (pr, st <= 0)):
        keep = np.ones(n); np.multiply.at(keep, x[m], 1 - p[m]); arr[:] = 1 - keep
    return pl, pr


def capture(n, pos, strand, q, c):
    """as capture_p with probability = c x strength rank q."""
    return capture_p(n, pos, strand, c * np.asarray(q, float))


def chromosome(length_bp, pos, strand, q, processivity_kb=200, separation_kb=150, c=0.9, load=None, R=10, snaps=100, every=5, rng=None, release=0.0, p=None):
    """simulate one chromosome (or region of length_bp); returns (legs_left, legs_right) of all snapshots, lattice size.
    p (optional): capture probability per site, used instead of c x q."""
    n = int(length_bp // SITE) + 1; pl, pr = capture_p(n, pos, strand, p) if p is not None else capture(n, pos, strand, q, c); tau = processivity_kb * 1000 / (2 * SITE)
    S = Sim(n, pl, pr, int(round(length_bp / (separation_kb * 1000))), tau, load=load, R=R, rng=rng, release=release)
    return S.run(burn=int(4 * tau) + 20, snaps=snaps, every=every) + (n,)


def loop_table(L, Rr, n):
    """unique (left site, right site) pairs of all motors in all snapshots with their share of snapshots."""
    m = L >= 0; key, cnt = np.unique(L[m] * n + Rr[m], return_counts=True)
    return key // n, key % n, cnt / max(len(L), 1)


def occupancy(tab, la, lb, ra, rb):
    """share of snapshots with a motor whose left leg is in sites la..lb and right leg in ra..rb (arrays of queries)."""
    tl, tr, tc = tab; o = np.argsort(tl, kind="stable"); tl, tr, tc = tl[o], tr[o], tc[o]; out = np.zeros(len(la))
    i0, i1 = np.searchsorted(tl, la, side="left"), np.searchsorted(tl, lb, side="right")
    for k in range(len(la)):
        if i1[k] > i0[k]:
            rr = tr[i0[k]:i1[k]]; out[k] = tc[i0[k]:i1[k]][(rr >= ra[k]) & (rr <= rb[k])].sum()
    return out


def contact_map(L, Rr, lo, hi, bin_sites=5, alphas=(1.0,), max_snaps=1000, rng=None, tie=0.3):
    """mean contact map of lattice sites lo..hi-1 at bins of `bin_sites` sites, for each alpha; from snapshot legs."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    N = (hi - lo) // bin_sites; rows = np.arange(len(L))
    if len(rows) > max_snaps:
        rows = (rng or np.random.default_rng(0)).choice(len(L), max_snaps, replace=False)
    bi, bj = np.arange(N - 1), np.arange(1, N); out = [np.zeros((N, N)) for _ in alphas]
    for s in rows:
        m = (L[s] >= lo) & (Rr[s] < lo + N * bin_sites); a, b = (L[s][m] - lo) // bin_sites, (Rr[s][m] - lo) // bin_sites; k = a != b
        G = csr_matrix((np.concatenate([np.ones(N - 1), np.full(int(k.sum()), tie)]), (np.concatenate([bi, a[k]]), np.concatenate([bj, b[k]]))), shape=(N, N))
        D = shortest_path(G, method="D", directed=False)
        for q, al in enumerate(alphas):
            out[q] += (D + 1.0) ** (-al)
    return [o / max(len(rows), 1) for o in out]


def obs_over_exp(M, valid=None):
    """divide each diagonal by its mean over valid pixels (distance-normalised map); invalid pixels -> nan."""
    N = len(M); v = np.ones(N, bool) if valid is None else valid; O = np.full((N, N), np.nan); r = np.arange(N)
    for k in range(N):
        i, j = r[:N - k], r[:N - k] + k; ok = v[i] & v[j]
        if ok.any():
            mu = M[i[ok], j[ok]].mean()
            if mu > 0:
                O[i[ok], j[ok]] = M[i[ok], j[ok]] / mu; O[j[ok], i[ok]] = O[i[ok], j[ok]]
    return O


def balance(M, iters=200):
    """iterative correction of a raw region map; bins with < 20% of the median coverage are dropped. Returns (map, valid)."""
    cov = M.sum(1); valid = cov >= 0.2 * np.median(cov[cov > 0]) if (cov > 0).any() else np.zeros(len(M), bool)
    W = M[np.ix_(valid, valid)].astype(float).copy()
    for _ in range(iters):
        s = W.sum(1); s = s / s[s > 0].mean(); s[s == 0] = 1; W /= s[:, None]; W /= s[None, :]
        if np.abs(s - 1).max() < 1e-5:
            break
    out = np.zeros(M.shape); out[np.ix_(valid, valid)] = W; return out, valid


# ---------------------------------------------------------------- 3D chain model
def _pcontact(var, radius):
    """probability that a 3D Gaussian separation with per-axis variance var lies within `radius`."""
    from scipy.special import erf
    x = radius / np.sqrt(np.maximum(var, 1e-12)); return erf(x / np.sqrt(2)) - np.sqrt(2 / np.pi) * x * np.exp(-x * x / 2)


def chain3d(L, Rr, lo, hi, bin_sites=5, k_loop=1.0, radius=1.0, zone=None, k_zone=0.0, max_snaps=1000, rng=None, sample=False):
    """3D chain model of lattice sites lo..hi-1: bins are beads joined by springs (neighbours: 1; each motor ties the beads
    of its two legs with k_loop; optional weak springs k_zone between beads of the same zone). In every snapshot the bead
    positions then follow a Gaussian law that is solved exactly (no volume exclusion). Returns the mean contact map
    (probability that two beads are within `radius`), the mean per-axis variance of their separation, and, if sample,
    one sampled conformation per snapshot (snapshots, beads, 3), in units where a free neighbour spring has variance 1."""
    rng = rng or np.random.default_rng(0); N = (hi - lo) // bin_sites; rows = np.arange(len(L))
    if len(rows) > max_snaps:
        rows = rng.choice(len(L), max_snaps, replace=False)
    base = np.zeros((N, N)); i = np.arange(N - 1); base[i, i] += 1; base[i + 1, i + 1] += 1; base[i, i + 1] -= 1; base[i + 1, i] -= 1
    if zone is not None and k_zone > 0:
        same = (np.sign(zone)[:, None] == np.sign(zone)[None, :]) & ~np.eye(N, dtype=bool); base += k_zone * (np.diag(same.sum(1)) - same)
    J = np.full((N, N), 1.0 / N); P = np.zeros((N, N)); V = np.zeros((N, N)); X = []
    for s in rows:
        m = (L[s] >= lo) & (Rr[s] < lo + N * bin_sites); a, b = (L[s][m] - lo) // bin_sites, (Rr[s][m] - lo) // bin_sites; k = a != b; a, b = a[k], b[k]
        Lm = base.copy(); np.add.at(Lm, (a, a), k_loop); np.add.at(Lm, (b, b), k_loop); np.add.at(Lm, (a, b), -k_loop); np.add.at(Lm, (b, a), -k_loop)
        A = Lm + J; C = np.linalg.inv(A) - J; d = np.diag(C); var = d[:, None] + d[None, :] - 2 * C; P += _pcontact(var, radius); V += var
        if sample:
            x = np.linalg.solve(np.linalg.cholesky(A).T, rng.normal(size=(N, 3))); X.append(x - x.mean(0))
    n = max(len(rows), 1)
    return P / n, V / n, (np.array(X) if sample else None)


def border_calls(X, w=4, span=3):
    """domain borders in single chromosomes from 3D positions X (chromosomes, segments, 3; nan = not seen).
    Score at position p (between segments p-1 and p) = mean distance between the w segments left of p and the w right of
    p / mean distance within each side. A border is called where the score is the highest within +- span positions.
    Returns (share of chromosomes with a border at each position; nan where it cannot be scored, chromosomes used)."""
    import warnings
    n = X.shape[1]; S = np.full((X.shape[0], n), np.nan); iu = np.triu_indices(w, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c0 in range(0, len(X), 1000):
            x = X[c0:c0 + 1000]; D = np.sqrt(((x[:, :, None, :] - x[:, None, :, :]) ** 2).sum(3))
            for p in range(w, n - w + 1):
                inter = np.nanmean(D[:, p - w:p, p:p + w], axis=(1, 2))
                intra = (np.nanmean(D[:, p - w:p, p - w:p][:, iu[0], iu[1]], 1) + np.nanmean(D[:, p:p + w, p:p + w][:, iu[0], iu[1]], 1)) / 2
                S[c0:c0 + 1000, p] = inter / intra
        ok = np.isfinite(S[:, w:n - w + 1]).mean(1) >= 0.8; Sv = S[ok]; best = np.zeros(Sv.shape, bool)
        for p in range(w, n - w + 1):
            best[:, p] = Sv[:, p] >= np.nanmax(Sv[:, max(w, p - span):min(n - w, p + span) + 1], 1)
    best &= np.isfinite(Sv); share = np.full(n, np.nan); share[w:n - w + 1] = best[:, w:n - w + 1].mean(0)
    return share, int(ok.sum())


def demo():
    rng = np.random.default_rng(1); n = 400                               # 2 Mb; two domains: > ... < > ... <
    pos = np.array([300_000, 1_000_000, 1_020_000, 1_700_000]); strand = np.array([1, -1, 1, -1]); q = np.ones(4)
    for name, sep in (("with motors", 150), ("no motors", 1e12)):
        L, Rr, n_ = chromosome(n * SITE, pos, strand, q, processivity_kb=800, separation_kb=sep, R=20, snaps=100, rng=rng)
        P = contact_map(L, Rr, 0, n_ - 1, bin_sites=5, rng=rng)[0]; O = obs_over_exp(P); b = lambda x: x // (SITE * 5)
        tab = loop_table(L, Rr, n_); occ = occupancy(tab, np.array([60 - 1, 204 - 1]), np.array([60 + 1, 204 + 1]), np.array([200 - 1, 340 - 1]), np.array([200 + 1, 340 + 1]))
        print(f"{name:12s} loop occupancy (two domains): {occ.round(2)}; O/E at the two corners {O[b(300_000), b(1_000_000)]:.2f} {O[b(1_020_000), b(1_700_000)]:.2f};"
              f" across the boundary {O[b(650_000), b(1_350_000)]:.2f}")


if __name__ == "__main__":
    {"demo": demo}[sys.argv[1]]()
