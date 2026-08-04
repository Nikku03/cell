"""EXHAUSTIVE RIGID DOCKING BY FFT -- and the only question that matters: does the TRUE pose rank near the top?

THE IDEA, which is the missing stage of the proposed pipeline: represent both partners on a grid, try every
relative rotation and every relative position, and find where they interlock without clashing.

WHY IT IS TRACTABLE.  Scanning translations naively is hopeless -- thousands of rotations x a 96^3 grid is 10^9
evaluations. But translation-scanning IS a correlation, so one FFT evaluates ALL translations at once. Two
96^3 transforms per rotation is ~80 ms, so several hundred rotations is minutes, not years. That single fact is
the difference between "impossible" and "an afternoon".

THE SCORE, Katchalski-Katzir, written as two REAL correlations rather than the usual complex-grid trick:

    interlock[k] = ( surface_R  correlate  solid_L )[k]      surface shell of receptor touched by ligand
    clash[k]     = ( core_R     correlate  solid_L )[k]      ligand atoms buried inside the receptor
    score[k]     = interlock[k] - CLASH_W * clash[k]

THE COMPLEX TRICK IS WRONG THE WAY IT IS USUALLY WRITTEN DOWN, and I had it wrong here first. Putting the core
on the imaginary axis (surface=1, core=-15j) and taking Re(ifftn(F_R * conj(F_L))) does NOT penalise clash:
with R = R_s + i R_c and L = L_s + i L_c,

    R * conj(L) = (R_s L_s + R_c L_c) + i(R_c L_s - R_s L_c)

so the real part is surface-surface PLUS core-core -- burial is REWARDED. The sign only works without the
conjugate (a convolution, not a correlation). Two explicit real correlations cost the same and cannot be got
backwards, and by linearity they collapse to one transform pair anyway:

    score = irfftn( ( F[surf_R] - CLASH_W * F[core_R] ) * conj( F[solid_L] ) )

THE TEST, and it is the whole point.  Exhaustive search is easy to demonstrate and proves nothing on its own --
it always returns a best pose. The question is whether the TRUE pose is anywhere near the top of what the search
finds, because the proposed accept/reject loop assumes the score can recognise the right answer when it sees it.

    hit rate       is a NEAR-NATIVE pose (ligand RMSD <= 5 A, the standard acceptable-pose criterion) in the
                   top 1 / 10 / 100 of the ranked pose list?
    CONTROL        what would a random ranker score? That is not a guess: every retained pose has a known RMSD,
                   so the near-native FRACTION f of the pose set gives the exact chance rate 1-(1-f)^k. A hit
                   rate that only matches 1-(1-f)^k means the SEARCH found the answer and the SCORE did not
                   rank it -- the pose list would work just as well shuffled.

DIAGNOSTIC, clearly separated because it uses the answer: the native rotation is added to the rotation set as
index -1. That decomposes any failure. If near-native is missing even at the native rotation, TRANSLATION
scoring is broken. If it appears at the native rotation but not in the blind search, ROTATION SAMPLING is too
coarse -- a compute problem, not a concept problem.

PREDECLARED, before any number:
    near-native in top-10 for a majority AND above the chance rate  -> search AND score work together. The
                                                                       accept/reject loop has something real.
    near-native present in the pose set, hit rate ~ chance rate     -> the SEARCH works and the SCORE does not.
                                                                       Ranking is the open problem.
    no near-native pose anywhere in the set                         -> rotation sampling or grid resolution,
                                                                       not a refutation of the concept. The
                                                                       native-rotation diagnostic says which.

SAMPLING FLOOR, stated before the run so a null cannot be re-read afterwards. N random rotations cover SO(3)
with P(some rotation within angle t of native) = 1-(1-t^3/(6*pi))^N. A ligand of radius of gyration Rg suffers
ligand RMSD ~ t*Rg from a rotation error t, so RMSD <= 5 A needs t <= 5/Rg. Both numbers are printed per
complex. If P is low, a miss is expected and means nothing.

HONEST LIMITS, up front: rigid bodies only -- sidechains move 1-3 A on binding, so rigid docking finds the right
NEIGHBOURHOOD and the last angstrom needs flexible refinement. Shape complementarity only, no electrostatics.
Every one of these makes the numbers here a LOWER bound.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import flex_physics as fp

OUT = A.OUT
N_PAIRS = 6
GRID, SPACING = 96, 1.4          # cells, Angstrom per cell -> 134 A box
N_ROT = 600                      # rotations sampled; coarse, and the covering probability is printed
CLASH_W = 15.0                   # core-core penalty weight relative to surface-surface reward
NEAR_NATIVE = 5.0                # ligand RMSD threshold for an "acceptable" pose
TOPK = (1, 10, 100)
PEAKS_PER_ROT = 4                # retained per rotation AFTER non-maximum suppression
NMS_CELLS = 4                    # minimum separation between retained peaks, in grid cells
MIN_AT, MAX_AT = 250, 2600       # atom-count window; upper bound is set by the box, checked again per complex


def rotations(n, seed=0):
    """quasi-uniform rotations via random quaternions -- coarse but unbiased"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        out.append(np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]))
    return out


def _dilate(b):
    """6-neighbour dilation. Atom CENTRES at 1.4 A spacing leave gaps inside a protein, so an un-dilated
    occupancy has a pitted, mostly-hollow interior and the 'core' erosion below returns almost nothing --
    the clash term would then be near-dead without ever being exactly zero."""
    o = b.copy()
    for ax in range(3):
        for s in (1, -1):
            o |= np.roll(b, s, axis=ax)
    return o


def _erode(b):
    o = b.copy()
    for ax in range(3):
        for s in (1, -1):
            o &= np.roll(b, s, axis=ax)
    return o


def gridify(co, centre):
    """returns (solid, surf, core) boolean grids. surface shell is one cell = 1.4 A thick."""
    idx = np.floor((co - centre) / SPACING + GRID / 2).astype(int)
    ok = ((idx >= 1) & (idx < GRID - 1)).all(1)
    idx = idx[ok]
    if len(idx) < MIN_AT:
        return None
    occ = np.zeros((GRID, GRID, GRID), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    solid = _dilate(occ)                       # fill inter-atom gaps so the body is solid, not pitted
    core = _erode(solid)
    surf = solid & ~core
    return solid, surf, core


def nms_peaks(sc, n, sep):
    """top n peaks at least `sep` cells apart. Without this the top-12 cells of a smooth correlation surface
    are all neighbours of ONE peak, so a 'top-10 pose list' would be one pose listed ten times and the rank
    test would be meaningless."""
    flat = np.argpartition(sc, -400, axis=None)[-400:]
    flat = flat[np.argsort(sc.reshape(-1)[flat])[::-1]]
    coords = np.array(np.unravel_index(flat, sc.shape)).T
    keep = []
    for c in coords:
        if keep:
            d = np.abs(np.array(keep) - c)
            d = np.minimum(d, GRID - d)        # wrapped distance: the correlation grid is periodic
            if (d.max(1) < sep).any():
                continue
        keep.append(c)
        if len(keep) >= n:
            break
    return [tuple(int(x) for x in c) for c in keep]


def shift_of(pk):
    """irfftn(FA * conj(FL))[k] = sum_x A[x+k] L[x], so the INDEX is the shift in cells, with wraparound
    meaning negative. Subtracting GRID/2 is the convention for a centred FFT, not a correlation index."""
    kk = np.array(pk, float)
    kk[kk > GRID / 2] -= GRID
    return kk * SPACING


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("EXHAUSTIVE RIGID DOCKING BY FFT -- does the TRUE pose rank near the top?")
    report("=" * 100)
    report(f"  grid {GRID}^3 at {SPACING} A ({GRID*SPACING:.0f} A box) | {N_ROT} blind rotations + the native "
           f"rotation as a diagnostic | near-native = ligand RMSD <= {NEAR_NATIVE} A")
    report(f"  score = surf_R*solid_L - {CLASH_W}*core_R*solid_L, two real correlations, one transform pair")
    report("  PREDECLARED: near-native in top-10 for a majority AND above the chance rate 1-(1-f)^10 => search")
    report("  AND score work. Hit rate ~ chance => search works, score does not. Absent entirely => sampling.")

    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb"))
    rots = rotations(N_ROT)
    box = GRID * SPACING
    rows, skipped = [], {"one_chain": 0, "size": 0, "box": 0, "grid": 0}
    for pdb in have:
        if len(rows) >= N_PAIRS:
            break
        try:
            t = fp.table(pdb)
        except Exception:
            continue
        if t is None:
            continue
        chains = sorted(set(t["ch"].tolist()))
        if len(chains) < 2:
            skipped["one_chain"] += 1
            continue
        ca, cb = chains[0], chains[1]
        R = t["co"][t["ch"] == ca]
        L = t["co"][t["ch"] == cb]
        if not (MIN_AT <= len(R) <= MAX_AT and MIN_AT <= len(L) <= MAX_AT):
            skipped["size"] += 1
            continue
        # the correlation is CIRCULAR, so a pose can wrap around the box and score as if it were in contact.
        # requiring D_R + D_L <= box makes every wrapped placement a non-contact one.
        dR = float(np.linalg.norm(R - R.mean(0), axis=1).max() * 2)
        dL = float(np.linalg.norm(L - L.mean(0), axis=1).max() * 2)
        if dR + dL > box:
            skipped["box"] += 1
            continue

        centre = R.mean(0)
        gr = gridify(R, centre)
        if gr is None:
            skipped["grid"] += 1
            continue
        _solid_R, surf_R, core_R = gr
        FA = np.fft.rfftn(surf_R.astype(np.float32)) - CLASH_W * np.fft.rfftn(core_R.astype(np.float32))
        F_surf_only = np.fft.rfftn(surf_R.astype(np.float32))

        lcen = L.mean(0)
        Lc = L - lcen
        rg = float(np.sqrt((Lc ** 2).sum(1).mean()))
        t_need = NEAR_NATIVE / rg                                  # rotation error tolerable, radians
        p_one = min(1.0, t_need ** 3 / (6 * np.pi))
        p_cov = 1 - (1 - p_one) ** N_ROT

        t1 = time.time()
        best, live_checked, live_diff = [], False, None
        # rotation index -1 is the NATIVE orientation -- diagnostic only, kept out of the blind pose list
        for ri, Rm in [(-1, np.eye(3))] + list(enumerate(rots)):
            g = gridify(Lc @ Rm.T + centre, centre)
            if g is None:
                continue
            FL = np.conj(np.fft.rfftn(g[0].astype(np.float32)))
            sc = np.fft.irfftn(FA * FL, surf_R.shape)
            if not live_checked:
                # LIVENESS: the clash term must actually change the answer, or this is surface overlap with a
                # decorative penalty. Compared on the same rotation, same grids.
                sc0 = np.fft.irfftn(F_surf_only * FL, surf_R.shape)
                live_diff = float(np.abs(sc - sc0).max())
                assert live_diff > 1e-3, "clash term is dead -- score is pure surface overlap"
                assert np.unravel_index(np.argmax(sc), sc.shape) != np.unravel_index(np.argmax(sc0), sc0.shape) \
                    or live_diff > 1.0, "clash term does not move the optimum"
                live_checked = True
            for pk in nms_peaks(sc, PEAKS_PER_ROT, NMS_CELLS):
                if sc[pk] <= 0:
                    continue          # net-clashing or non-contacting; not a pose the search "found"
                best.append((float(sc[pk]), ri, shift_of(pk)))

        nat_poses = [b for b in best if b[1] == -1]
        blind = sorted([b for b in best if b[1] != -1], key=lambda x: -x[0])

        def rmsd_of(ri, shift):
            Rm = np.eye(3) if ri == -1 else rots[ri]
            posed = Lc @ Rm.T + centre + shift
            return float(np.sqrt(((posed - L) ** 2).sum(1).mean()))

        rm = np.array([rmsd_of(ri, s) for _v, ri, s in blind]) if blind else np.array([])
        if not len(rm):
            skipped["grid"] += 1
            continue
        hits = {k: bool((rm[:k] <= NEAR_NATIVE).any()) for k in TOPK}
        frac = float((rm <= NEAR_NATIVE).mean())
        chance = {k: 1 - (1 - frac) ** k for k in TOPK}
        nat_rm = [rmsd_of(-1, s) for _v, _r, s in nat_poses]

        rows.append({"pdb": pdb, "chains": [ca, cb], "n_R": int(len(R)), "n_L": int(len(L)),
                     "rg_ligand": rg, "rot_tol_deg": float(np.degrees(t_need)), "p_coverage": float(p_cov),
                     "n_poses": len(blind), "best_rmsd": float(rm.min()),
                     "rank_of_best_rmsd": int(np.argmin(rm)), "near_native_frac": frac,
                     "hits": hits, "chance": chance,
                     "native_rot_best_rmsd": float(min(nat_rm)) if nat_rm else None,
                     "clash_liveness_delta": live_diff, "secs": time.time() - t1})
        report(f"    {pdb} {ca}/{cb} ({len(R)}+{len(L)} atoms, Rg_L {rg:.0f} A): {len(blind)} poses in "
               f"{time.time()-t1:.0f}s | best RMSD {rm.min():.1f} A at rank {int(np.argmin(rm))} | "
               f"top10 {hits[10]} (chance {chance[10]:.3f}) | native-rot best {min(nat_rm):.1f} A | "
               f"P(cover {np.degrees(t_need):.0f} deg) {p_cov:.2f}")

    if not rows:
        report(f"\n  no usable complexes  {skipped}")
        return 1

    report(f"\n  {len(rows)} complexes docked   (skipped: {skipped})")
    report(f"  clash-term liveness: max |score - score_without_clash| = "
           f"{min(r['clash_liveness_delta'] for r in rows):.1f} (min over complexes)")
    report(f"\n  {'top-k':>8} {'hits':>8} {'chance':>10}")
    tab = {}
    for k in TOPK:
        n = sum(r["hits"][k] for r in rows)
        exp = float(np.mean([r["chance"][k] for r in rows])) * len(rows)
        tab[str(k)] = {"hits": n, "expected_by_chance": exp}
        report(f"  {k:>8} {n:>4}/{len(rows):<3} {exp:>10.2f}")
    bm = float(np.median([r["best_rmsd"] for r in rows]))
    nat_ok = [r for r in rows if r["native_rot_best_rmsd"] is not None]
    nbm = float(np.median([r["native_rot_best_rmsd"] for r in nat_ok])) if nat_ok else float("nan")
    report(f"\n    median best RMSD anywhere in the blind pose set : {bm:.1f} A")
    report(f"    median best RMSD at the NATIVE rotation          : {nbm:.1f} A   <- translation scoring alone")
    report(f"    median P(rotation set covers the native)         : "
           f"{np.median([r['p_coverage'] for r in rows]):.2f}")

    report("\n  READING")
    top10, exp10 = sum(r["hits"][10] for r in rows), float(np.mean([r["chance"][10] for r in rows])) * len(rows)
    found = sum(r["best_rmsd"] <= NEAR_NATIVE for r in rows)
    if top10 > len(rows) / 2 and top10 > exp10 + 1:
        report(f"  Near-native poses rank in the top 10 for {top10}/{len(rows)} complexes against {exp10:.1f}")
        report("  expected from ranking the same pose set at random. The exhaustive search AND the shape score")
        report("  work together, and the accept/reject loop has something real to iterate on.")
    elif found:
        report(f"  The search REACHES near-native ({found}/{len(rows)} complexes have a pose <= {NEAR_NATIVE} A,")
        report(f"  median best {bm:.1f} A) but the score does not rank it: top-10 hits {top10} vs {exp10:.1f}")
        report("  expected by chance. Sampling is solved; RANKING is the open problem, and more rotations")
        report("  cannot fix a score that cannot tell the right pose from the wrong one.")
    elif nat_ok and nbm <= NEAR_NATIVE:
        report(f"  No near-native pose in the blind set, but at the NATIVE rotation the score does find the")
        report(f"  right translation ({nbm:.1f} A). So translation scoring works and ROTATION SAMPLING is the")
        report(f"  binding constraint at N={N_ROT} -- a compute problem with a known fix, not a refutation.")
    else:
        report(f"  No near-native pose even at the native rotation (median {nbm:.1f} A). Translation scoring")
        report(f"  itself is wrong -- the grid resolution or the surface definition, not the sampling.")

    json.dump({"test": "dock_fft", "grid": GRID, "spacing": SPACING, "n_rot": N_ROT, "clash_w": CLASH_W,
               "peaks_per_rot": PEAKS_PER_ROT, "nms_cells": NMS_CELLS, "near_native_A": NEAR_NATIVE,
               "complexes": rows, "top_k": tab, "n_complexes": len(rows), "skipped": skipped,
               "median_best_rmsd": bm, "median_native_rot_best_rmsd": nbm, "log": log},
              open(OUT / "dock_fft.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_fft.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
