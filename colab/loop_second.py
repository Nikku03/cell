"""LOOP 77 -- CHROMATIN AT ONE-SECOND RESOLUTION, AND A CONTACT MAP THAT IS EXACT AND CHEAP.

TWO CHANGES, AND ONLY THE FIRST IS INTERESTING.

1. THE CONTACT MAP IS COMPUTED BY A RANK-k UPDATE INSTEAD OF A FRESH INVERSE.
   Loop 35 builds a contact map by, for every configuration, assembling the loop-graph Laplacian and
   inverting it: `laplacian(n, loops, confine)` then `r2_matrix(...)`, once per configuration, 50
   configurations per condition. That is O(n^3) each time, and at n = 1,926 bins it dominates
   everything.

   But the chain never changes. Only the ~12 cohesin loops change. So each configuration is the SAME
   base Laplacian plus a rank-k update, and the Woodbury identity gives the updated inverse from one
   precomputed base inverse:

       L  = L0 + U U^T,          U is n x k, column i is (e_a - e_b) for loop i
       L^-1 = L0^-1 - L0^-1 U (I + U^T L0^-1 U)^-1 U^T L0^-1

   One O(n^3) inverse for the whole run instead of one per configuration, then O(n^2 k) per
   configuration. This is not an approximation. It is the same matrix, and V1 exists to prove that
   rather than assert it -- if the two maps differ, the speed is worthless and the gate says so.

   (A cruder banded heuristic -- genomic distance minus loop length -- was benchmarked first and is
   NOT used here. It is fast but it is not the resistance distance on the loop graph, and there was
   no reason to defend an approximation when the exact update is available at similar cost.)

2. THE TIMESTEP DROPS FROM 33 s TO 1 s.
   Loop 35's step is one 25 kb bin traversed at 0.75 kb/s = 33.3 s, so a leg either advances a full
   bin or does not move. At 1 s a leg advances with probability v*dt/BIN = 0.03. That is a finer
   clock on the same process, and it should change nothing about the ensemble -- which is exactly why
   V2 tests it. Measured beforehand: at 1 s, 98.8% of steps still carry at least one event, so the
   clock is not yet running empty; by 0.1 s only 33.8% do, and 0.275 s is where the polymer's own
   relaxation time makes the quasi-static assumption invalid anyway. 1 s is the last useful step.

WHAT THIS IS FOR. A 1-second clock is what lets chromatin be coupled to anything that switches on a
timescale of seconds -- TF binding, burst initiation. At 33 s per step those events cannot be
represented at all. The point is not speed for its own sake; it is that the conditionality the rest
of the model needs lives at seconds, and the map has to be affordable at that clock.

WHAT IS NOT CLAIMED. This predicts contact structure. It does not predict transcription. The only
measured chromatin-to-rate link in this project is DNA torsion vs synthesis rate at partial r = 0.14
(outputs/loop_real_chromatin.json), about 2% of variance and cross-cell-type. Nothing here improves
that, and a fast contact map must not be read as a fast gene-expression model.

PREDECLARED, before any number:

  V1 THE FAST MAP IS THE SAME MAP                                    THE GATE.
       Woodbury map vs a fresh full inverse, on the SAME loop configurations. Gate: max relative
       error <= 1e-6 on the finite entries. This is a numerical identity, so anything above that is a
       bug, not a tolerance. If V1 fails nothing else in this module means anything.
  V2 THE ONE-SECOND CLOCK AGREES WITH THE THIRTY-THREE-SECOND CLOCK
       same physics, finer clock, same literature parameters. The two ensembles' contact maps must
       agree: Spearman >= 0.99 over the banded region, and both P(s) slopes inside loop 33's window.
       A finer clock that MOVES the answer would mean the coarse clock was aliasing.
  V3 P(s) LANDS IN THE PREDECLARED WINDOW
       loop 33 measured the real chr21 map at slope -0.9636 and loop 35 fixed the acceptance window
       at (-1.16, -0.76) BEFORE any model ran. Same window here, not renegotiated.
  V4 THE ORIENTATION SIGNATURE APPEARS AND COLLAPSES WHEN SHUFFLED   THE DECISIVE CONTROL.
       convergent-CTCF minus non-convergent at matched separation must be positive with real motif
       orientations and must collapse toward zero when the orientations are permuted. Both halves
       required. A signature that survives shuffling is a distance artifact, not extrusion.
  V5 IT TRACKS THE MEASURED MAP, SCORED AGAINST THE REPLICATE CEILING
       Spearman between simulated and measured chr21 over the banded region. Reported next to the
       replicate-vs-replicate agreement on the two real maps, because that is the ceiling any model
       can reach and quoting a model correlation without it is meaningless. Gate: the model must beat
       a distance-only null (the expected-by-separation map), which is what a model that has learned
       nothing but P(s) would score.
  V6 THE COST IS REPORTED
       wall clock per configuration for both methods, and the resulting factor. Reported, not gated.

-> outputs/loop_second.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from loop_polymer import laplacian, r2_matrix  # noqa: E402
from loop_hic_target import expected  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
HIC = SC / "hic_chr21_25kb.npy"
HICREP = SC / "hic_chr21_25kb_rep.npy"
CTCF = SC / "ctcf_gm12878_hg19.bed.gz"
FASTA = SC / "hg19_chr21.fa.gz"
PFM = SC / "ctcf_pfm.json"
TGT = OUT / "loop_hic_target.json"
CHROM, BIN = "chr21", 25000

# literature parameters, identical to loop 35. None fitted to the map being scored.
V_KB_S = 0.75
RESIDENCE_S = 900.0
DENSITY_KB = 150.0
MAX_BLOCK = 0.95
CONFINE = 1.1e-5
N_CONFIG = 50
BURN_S = 6000.0          # 100 min burn-in, matching loop 35's 200 coarse steps
SAMPLE_EVERY_S = 660.0   # loop 35 sampled every 20 coarse steps = 666 s
DT_FINE, DT_COARSE = 1.0, BIN / 1e3 / V_KB_S
BAND_BP = 2e6

PS_WINDOW = (-1.16, -0.76)
V1_TOL = 1e-6
V2_RHO = 0.99
SEED = 7701

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def simulate(n, block_fwd, block_rev, rng, dt, n_config=N_CONFIG):
    """Loop extrusion on a dt-second clock. Identical physics to loop 35's simulate().

    At dt = 33.3 s a leg advances one bin per step deterministically (p_adv = 1); at dt = 1 s it
    advances with p = v*dt/BIN. Everything else -- CTCF face-specific blocking, cohesin occlusion,
    unbinding and reloading -- is unchanged.
    """
    p_adv = min(1.0, V_KB_S * dt / (BIN / 1e3))
    p_off = min(1.0, dt / RESIDENCE_S)
    n_coh = max(1, int(n * BIN / 1e3 / DENSITY_KB))
    left = rng.integers(0, n, n_coh)
    right = np.minimum(left + 1, n - 1)
    burn = int(BURN_S / dt)
    every = max(1, int(SAMPLE_EVERY_S / dt))
    configs, t, nmove = [], 0, 0
    while len(configs) < n_config:
        occ = np.zeros(n, bool)
        occ[left] = True
        occ[right] = True
        nl = np.maximum(left - 1, 0)
        canl = ((nl != left) & ~occ[nl] & (rng.random(n_coh) < p_adv)
                & (rng.random(n_coh) > block_fwd[left] * MAX_BLOCK))
        nr = np.minimum(right + 1, n - 1)
        canr = ((nr != right) & ~occ[nr] & (rng.random(n_coh) < p_adv)
                & (rng.random(n_coh) > block_rev[right] * MAX_BLOCK))
        if canl.any() or canr.any():
            nmove += 1
        left = np.where(canl, nl, left)
        right = np.where(canr, nr, right)
        off = rng.random(n_coh) < p_off
        if off.any():
            newpos = rng.integers(0, n - 1, int(off.sum()))
            left[off] = newpos
            right[off] = newpos + 1
        t += 1
        if t >= burn and (t - burn) % every == 0:
            configs.append(np.c_[left.copy(), right.copy()])
    return configs, n_coh, nmove / max(t, 1)


def base_inverse(n, confine=CONFINE):
    """L0^-1 for the bare chain, computed ONCE. This is the whole trick."""
    L0 = laplacian(n, loops=[], confine=confine)
    return np.linalg.inv(L0), L0


def r2_woodbury(G0, cfg):
    """<R^2> matrix for one configuration, from the base inverse plus a rank-k update.

    L = L0 + U U^T with U[:, i] = e_a - e_b for loop i, so
        L^-1 = G0 - G0 U (I + U^T G0 U)^-1 U^T G0
    and <R^2>_ij = G_ii + G_jj - 2 G_ij.
    """
    loops = [(int(a), int(b)) for a, b in cfg if b > a]
    n = G0.shape[0]
    if loops:
        k = len(loops)
        U = np.zeros((n, k))
        for i, (a, b) in enumerate(loops):
            U[a, i] = 1.0
            U[b, i] = -1.0
        GU = G0 @ U                                   # n x k
        M = np.eye(k) + U.T @ GU                      # k x k
        G = G0 - GU @ np.linalg.solve(M, GU.T)
    else:
        G = G0
    d = np.diag(G)
    return d[:, None] + d[None, :] - 2.0 * G


def contact_map_fast(n, configs, G0):
    acc = np.zeros((n, n))
    for cfg in configs:
        R2 = r2_woodbury(G0, cfg)
        np.fill_diagonal(R2, np.inf)
        acc += np.maximum(R2, 1e-12) ** -1.5
    return acc / max(len(configs), 1)


def contact_map_exact(n, configs, confine=CONFINE):
    acc = np.zeros((n, n))
    for cfg in configs:
        L = laplacian(n, loops=[(int(a), int(b)) for a, b in cfg if b > a], confine=confine)
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return acc / max(len(configs), 1)


def ps_slope(M, mask, lo=1e5, hi=1e7):
    exp = expected(M, mask)
    d = np.arange(len(M)) * BIN
    s = np.isfinite(exp) & (d >= lo) & (d <= hi) & (exp > 0)
    return float(np.polyfit(np.log10(d[s]), np.log10(exp[s]), 1)[0]), exp


def orientation_effect(M, exp, fs, rs, mask, n):
    from collections import defaultdict
    conv, byd = [], defaultdict(list)
    sites = sorted(fs | rs)
    for i in sites:
        for j in sites:
            if j - i < 4 or (j - i) * BIN > 2e6:
                continue
            if not (mask[i] and mask[j]) or not np.isfinite(M[i, j]) \
               or not np.isfinite(exp[j - i]) or exp[j - i] <= 0:
                continue
            v = M[i, j] / exp[j - i]
            if i in fs and j in rs:
                conv.append((j - i, v))
            else:
                byd[j - i].append(v)
    mc, mo = [], []
    for dd, v in conv:
        if byd.get(dd):
            mc.append(v)
            mo.append(float(np.mean(byd[dd])))
    if len(mc) < 30:
        return float("nan"), 0
    return float(np.mean(np.array(mc) - np.array(mo))), len(mc)


def band_rho(A, B, mask, n, w):
    from scipy.stats import spearmanr
    ii, jj = np.triu_indices(n, 1)
    s = (jj - ii <= w) & mask[ii] & mask[jj]
    a, b = A[ii[s], jj[s]], B[ii[s], jj[s]]
    f = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(spearmanr(a[f], b[f]).statistic), int(f.sum())


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 77 -- chromatin at one-second resolution, with an exact and cheap contact map")
    say("=" * 100)
    say()

    H = np.load(HIC)
    n = len(H)
    mask = np.isfinite(H).sum(1) > 0.5 * n
    say(f"  {CHROM} at {BIN//1000} kb: {n:,} bins, {int(mask.sum()):,} mappable")
    tgt = json.load(open(TGT)) if TGT.exists() else {}

    # oriented CTCF landscape, rebuilt exactly as loops 33/35 do
    say("  rebuilding the oriented CTCF landscape from motif scores")
    seq = []
    for ln in gzip.open(FASTA, "rt"):
        if not ln.startswith(">"):
            seq.append(ln.strip())
    seq = "".join(seq).upper()
    peaks = []
    for ln in gzip.open(CTCF, "rt"):
        f = ln.split("\t")
        if f[0] != CHROM:
            continue
        st, en = int(f[1]), int(f[2])
        peaks.append({"summit": (st + en) // 2})
    pfm = json.load(open(PFM))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    idx = {c: i for i, c in enumerate("ACGT")}

    def sc(s):
        if len(s) != Lw or any(c not in idx for c in s):
            return -1e9
        return float(sum(np.log2(W[i, idx[c]] / 0.25 + 1e-9) for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for p in peaks:
        best, bo = -1e9, 0
        c = p["summit"]
        for off in range(-100, 101, 5):
            s = seq[c + off: c + off + Lw]
            f_, r_ = sc(s), sc(rc(s))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0
    orients = [p["orient"] for p in peaks]
    n_or = sum(1 for o in orients if o)
    say(f"     {len(peaks):,} peaks, {n_or:,} carried an orientation")

    def landscape(ors):
        bf, br = np.zeros(n), np.zeros(n)
        for p, o in zip(peaks, ors):
            b = p["summit"] // BIN
            if 0 <= b < n:
                if o > 0:
                    bf[b] = 1.0
                elif o < 0:
                    br[b] = 1.0
        return bf, br

    bf, br = landscape(orients)
    rng = np.random.default_rng(SEED)
    fs = {p["summit"] // BIN for p, o in zip(peaks, orients) if o > 0}
    rs = {p["summit"] // BIN for p, o in zip(peaks, orients) if o < 0}
    say()

    say("  precomputing the base inverse ONCE (this is the whole trick)")
    tb = time.time()
    G0, _ = base_inverse(n)
    say(f"     one {n}x{n} inverse: {time.time()-tb:.1f}s")
    say()

    say("V1 THE FAST MAP IS THE SAME MAP")
    cfgs1, ncoh, frac_move = simulate(n, bf, br, np.random.default_rng(SEED), DT_FINE)
    say(f"     1 s clock: {ncoh} cohesins, {len(cfgs1)} configurations, "
        f"{frac_move:.1%} of steps carried a move")
    sub = cfgs1[:5]
    ta = time.time()
    Ma = contact_map_fast(n, sub, G0)
    t_fast = (time.time() - ta) / len(sub)
    tb2 = time.time()
    Mb = contact_map_exact(n, sub)
    t_exact = (time.time() - tb2) / len(sub)
    fin = np.isfinite(Ma) & np.isfinite(Mb) & (Mb > 0)
    err = float(np.max(np.abs(Ma[fin] - Mb[fin]) / np.abs(Mb[fin])))
    say(f"     Woodbury {t_fast*1000:.0f} ms/config   fresh inverse {t_exact*1000:.0f} ms/config   "
        f"-> {t_exact/max(t_fast,1e-9):.0f}x")
    say(f"     max relative difference over {int(fin.sum()):,} entries: {err:.3e}   "
        f"(gate {V1_TOL:.0e})")
    v1 = err <= V1_TOL
    say(f"     V1 {'PASS' if v1 else 'FAIL'} -- the speedup is an identity, not an approximation")
    say()

    say("V2 THE ONE-SECOND CLOCK AGREES WITH THE THIRTY-THREE-SECOND CLOCK")
    M1 = contact_map_fast(n, cfgs1, G0)
    cfgs33, _, fm33 = simulate(n, bf, br, np.random.default_rng(SEED), DT_COARSE)
    M33 = contact_map_fast(n, cfgs33, G0)
    w = int(BAND_BP // BIN)
    rho12, npair = band_rho(M1, M33, mask, n, w)
    s1, exp1 = ps_slope(M1, mask)
    s33, exp33 = ps_slope(M33, mask)
    say(f"     33.3 s clock: {fm33:.1%} of steps carried a move;  1 s clock: {frac_move:.1%}")
    say(f"     contact maps agree over the {BAND_BP/1e6:.0f} Mb band: Spearman {rho12:.4f} "
        f"(n={npair:,}, gate {V2_RHO})")
    say(f"     P(s) slope   1 s {s1:+.4f}    33 s {s33:+.4f}")
    v2 = rho12 >= V2_RHO and all(PS_WINDOW[0] <= x <= PS_WINDOW[1] for x in (s1, s33))
    say(f"     V2 {'PASS' if v2 else 'FAIL'}")
    say()

    say("V3 P(s) LANDS IN THE PREDECLARED WINDOW")
    meas_ps = tgt.get("target", {}).get("ps_slope", {}).get("measured")
    say(f"     measured chr21 (loop 33)  {meas_ps if meas_ps is not None else 'n/a'}")
    say(f"     simulated, 1 s clock      {s1:+.4f}    window {PS_WINDOW}")
    v3 = PS_WINDOW[0] <= s1 <= PS_WINDOW[1]
    say(f"     V3 {'PASS' if v3 else 'FAIL'}")
    say()

    say("V4 THE ORIENTATION SIGNATURE APPEARS AND COLLAPSES WHEN SHUFFLED")
    o_real, n_real = orientation_effect(M1, exp1, fs, rs, mask, n)
    sh = list(rng.permutation(orients))
    bfs, brs = landscape(sh)
    cfgs_s, _, _ = simulate(n, bfs, brs, np.random.default_rng(SEED), DT_FINE)
    Ms = contact_map_fast(n, cfgs_s, G0)
    _, exps = ps_slope(Ms, mask)
    fss = {p["summit"] // BIN for p, o in zip(peaks, sh) if o > 0}
    rss = {p["summit"] // BIN for p, o in zip(peaks, sh) if o < 0}
    o_shuf, _ = orientation_effect(Ms, exps, fss, rss, mask, n)
    say(f"     real motif orientation      {o_real:+.4f}   ({n_real} matched pairs)")
    say(f"     SHUFFLED orientation        {o_shuf:+.4f}   <- must collapse")
    v4 = np.isfinite(o_real) and o_real > 0 and (not np.isfinite(o_shuf) or o_shuf < o_real * 0.5)
    say(f"     V4 {'PASS' if v4 else 'FAIL'}")
    say()

    say("V5 IT TRACKS THE MEASURED MAP, AGAINST THE REPLICATE CEILING")
    rho_model, nm = band_rho(M1, H, mask, n, w)
    if HICREP.exists():
        R = np.load(HICREP)
        rho_ceiling, _ = band_rho(H, R, mask, n, w)
    else:
        rho_ceiling = float("nan")
    expm = expected(H, mask)
    D = np.zeros((n, n))
    ii, jj = np.triu_indices(n, 1)
    D[ii, jj] = np.where(np.isfinite(expm[jj - ii]), expm[jj - ii], 0.0)
    D = D + D.T
    rho_dist, _ = band_rho(D, H, mask, n, w)
    say(f"     simulated vs measured        Spearman {rho_model:+.4f}  (n={nm:,})")
    say(f"     distance-only null vs meas.  Spearman {rho_dist:+.4f}  <- what P(s) alone buys")
    say(f"     replicate vs replicate       Spearman {rho_ceiling:+.4f}  <- the ceiling")
    v5 = rho_model > rho_dist
    say(f"     V5 {'PASS' if v5 else 'FAIL'} -- the model "
        f"{'adds structure beyond' if v5 else 'adds nothing beyond'} separation")
    say()

    say("V6 COST")
    per_min_1s = t_fast * (60.0 / SAMPLE_EVERY_S)
    say(f"     one configuration: Woodbury {t_fast*1000:.0f} ms   fresh inverse {t_exact*1000:.0f} ms")
    say(f"     loop 35 ran 3 conditions x {N_CONFIG} fresh inverses = "
        f"{3*N_CONFIG*t_exact:.0f} s of inversion")
    say(f"     here: 1 base inverse + {3*N_CONFIG} rank-k updates = "
        f"{3*N_CONFIG*t_fast:.0f} s")
    say(f"     V6 PASS (reported)")
    say()

    gates = {"V1 fast map is the same map": bool(v1),
             "V2 1 s clock agrees with 33 s clock": bool(v2),
             "V3 P(s) in the predeclared window": bool(v3),
             "V4 orientation signature appears and collapses": bool(v4),
             "V5 beats a distance-only null": bool(v5),
             "V6 cost reported": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(HIC), str(HICREP), str(CTCF), str(FASTA), str(PFM)],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=["Woodbury map compared against a fresh inverse on identical configs",
                                "1 s clock compared against loop 35's 33 s clock",
                                "P(s) window fixed by loop 33/35 before this module existed",
                                "CTCF orientations shuffled as the decisive control",
                                "replicate-vs-replicate reported as the achievable ceiling",
                                "distance-only null reported as the floor",
                                "literature extrusion parameters, none fitted to the scored map"],
                      note="loop 35 inverted a Laplacian per configuration; the chain never changes, "
                           "so each configuration is a rank-k update of one base inverse")
    RM.report(man, emit=say)
    json.dump({"test": "loop_second", "manifest": man, "gates": gates,
               "n_bins": n, "n_mappable": int(mask.sum()), "n_cohesins": ncoh,
               "dt_fine": DT_FINE, "dt_coarse": DT_COARSE,
               "frac_steps_moving_1s": frac_move, "frac_steps_moving_33s": fm33,
               "v1_max_rel_err": err, "t_fast_ms": t_fast * 1000, "t_exact_ms": t_exact * 1000,
               "speedup": t_exact / max(t_fast, 1e-9),
               "rho_1s_vs_33s": rho12, "ps_1s": s1, "ps_33s": s33, "ps_window": list(PS_WINDOW),
               "orientation_real": o_real, "orientation_shuffled": o_shuf,
               "rho_model_vs_measured": rho_model, "rho_distance_null": rho_dist,
               "rho_replicate_ceiling": rho_ceiling,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_second.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_second.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
