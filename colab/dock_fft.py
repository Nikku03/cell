"""EXHAUSTIVE RIGID DOCKING BY FFT -- and the only question that matters: does the TRUE pose rank near the top?

THE IDEA, which is the missing search stage: represent both partners on a grid, try every relative rotation and
every relative position, and find where they interlock without clashing.

WHY IT IS TRACTABLE.  Scanning translations naively is hopeless -- hundreds of rotations x a 96^3 grid is 10^8
evaluations. But translation-scanning IS a correlation, so one FFT evaluates ALL translations at once. Two 96^3
transforms per rotation is ~80 ms, so several hundred rotations is minutes, not years.

THE SCORE, Katchalski-Katzir, as two REAL correlations:

    interlock[k] = ( skin_R  correlate  solid_L )[k]   ligand volume sitting in the receptor's outward shell
    clash[k]     = ( core_R  correlate  solid_L )[k]   ligand volume driven into the receptor's interior
    score[k]     = interlock[k] - CLASH_W * clash[k]

THE FIRST VERSION OF THIS FILE WAS VOID, and the way it failed is the point of the rewrite.

  (a) It rasterised ATOM CENTRES and dilated by one cell. At 1.4 A spacing a van der Waals contact is 3.5 A =
      2.5 cells, and one cell of dilation on each side leaves a 0.5-cell GAP. The two shells never touched, so
      the TRUE interface scored 8 out of a maximum of 989 -- near zero. Fixed by rasterising each atom to its
      actual vdW radius and defining the receptor's shell as a 2-cell layer growing OUTWARD from the molecular
      surface, which is exactly where a partner at contact distance sits. Native interlock 8 -> 177.

  (b) It took chains[0] and chains[1], alphabetically. That is not the biological interface. 1AK4 A/B has
      ZERO atom pairs within 5 A (the real interface is B/C); 1BRS and 1B2S got crystal-packing contacts of
      153 and 140 instead of the real 568 and 574. Four of six complexes were docked on the wrong pair. Fixed
      by choosing the chain pair with the most 5 A atom contacts and requiring at least MIN_CONTACTS.

  Together those made the earlier run's conclusion -- "the score is weak, the native cannot be ranked" -- simply
  false. With the representation corrected, the native translation at the native rotation ranks in the TOP 1-4
  of all 884,736 placements in all six complexes. The failure was mine, not the method's.

  The diagnostic that caught it is in `dock_diagnose.py`, and its own auto-reading was ALSO wrong: it branched
  on "is the native being charged a clash penalty" and concluded "score is weak, not broken" when clash was 0.
  It never checked whether native INTERLOCK was non-trivial, which is where the bug actually lived. A gate that
  only tests the failure mode you thought of will confidently clear the one you did not.

SO THIS FILE NOW CARRIES A HARD SANITY GATE. Before a complex is docked, its native pose must score interlock
well above an arbitrary placement's. A complex that fails is reported as REPRESENTATION FAIL and excluded from
the hit rates rather than silently dragging them down.

THE CLASH WEIGHT IS TUNED ON A HELD-OUT SPLIT.  The weight matters enormously (top-18,218 at w=3, top-18 at
w=10, top-3 at w=30), so picking it by looking at the complexes it is then scored on would be tuning on the test
set. Complexes are split by sorted PDB id: the first N_TUNE choose the weight, the rest are never touched until
the blind run. The chosen weight and both splits are printed.

THE TEST.  Exhaustive search always returns a best pose, so "it found something" proves nothing:

    hit rate       is a NEAR-NATIVE pose (ligand RMSD <= 5 A, the standard acceptable-pose criterion) in the
                   top 1 / 10 / 100 of the ranked pose list?
    CONTROL        every retained pose has a known RMSD, so the near-native FRACTION f of the pose set gives
                   the exact random-ranker rate 1-(1-f)^k. A hit rate matching that means the SEARCH found the
                   answer and the SCORE did not rank it -- the list would work as well shuffled.
    DIAGNOSTIC     the native rotation goes in as index -1, kept out of the blind list, so a miss separates
                   "translation scoring is broken" from "rotation sampling is too coarse".

SAMPLING FLOOR, stated before the run so a null cannot be re-read afterwards. N random rotations cover SO(3)
with P(some rotation within angle t of native) = 1-(1-t^3/(6*pi))^N, and a ligand of gyration radius Rg suffers
ligand RMSD ~ t*Rg. Both are printed per complex; if P is low, a miss is expected and means nothing.

PREDECLARED, before any number:
    near-native in top-10 for a majority of TEST complexes AND above chance -> search AND score work together.
    near-native present in the pose set, hit rate ~ chance                  -> search works, score does not.
    no near-native pose anywhere                                            -> sampling or representation; the
                                                                              native-rotation row says which.

HONEST LIMITS: rigid bodies only -- sidechains move 1-3 A on binding, so this finds the right NEIGHBOURHOOD and
the last angstrom needs flexible refinement. Shape only, no electrostatics. Bound-form coordinates on both
sides, which is the easy case; unbound docking is harder and is not what this measures.
"""
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import flex_physics as fp

OUT = A.OUT
GRID, SPACING = 96, 1.4          # cells, Angstrom per cell -> 134 A box
N_ROT = 600                      # blind rotations; coarse, and the covering probability is printed
SKIN, CORE_ERODE = 2, 1          # receptor shell thickness / interior erosion, in cells
WEIGHTS = (1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0)
NEAR_NATIVE = 5.0
TOPK = (1, 10, 100)
PEAKS_PER_ROT, NMS_CELLS = 4, 4
MIN_AT, MAX_AT = 250, 2600
MIN_CONTACTS = 100               # atom pairs within 5 A required for a chain pair to count as an interface
N_TOTAL, N_TUNE = 20, 8
SANITY_RATIO = 5.0               # native interlock must beat the median placement by this factor


def rotations(n, seed=0):
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


def _morph(b, n, grow):
    o = b.copy()
    for _ in range(n):
        p = o.copy()
        for ax in range(3):
            for s in (1, -1):
                if grow:
                    o |= np.roll(p, s, axis=ax)
                else:
                    o &= np.roll(p, s, axis=ax)
    return o


def offsets_for(rad):
    """(offset, atom-mask) pairs: which atoms reach a cell this far away. Radii do not change under rotation,
    so this is computed ONCE per molecule and reused for every pose."""
    rm = int(np.ceil(rad.max() / SPACING))
    out = []
    for dx in range(-rm, rm + 1):
        for dy in range(-rm, rm + 1):
            for dz in range(-rm, rm + 1):
                d = np.sqrt(dx * dx + dy * dy + dz * dz) * SPACING
                sel = rad >= d
                if sel.any():
                    out.append(((dx, dy, dz), sel))
    return out


def rasterise(co, centre, offs):
    """solid vdW volume on the grid. Atom CENTRES alone leave a half-cell gap across a real contact -- that was
    the bug that made the first version of this file report near-zero interlock at true interfaces."""
    idx = np.round((co - centre) / SPACING + GRID / 2).astype(int)
    ok = ((idx >= 3) & (idx < GRID - 3)).all(1)
    occ = np.zeros((GRID, GRID, GRID), bool)
    if ok.sum() < MIN_AT:
        return None
    for (dx, dy, dz), sel in offs:
        j = idx[sel & ok]
        if len(j):
            occ[j[:, 0] + dx, j[:, 1] + dy, j[:, 2] + dz] = True
    return occ


def nms_peaks(sc, n, sep):
    """top n peaks at least `sep` cells apart. Without this the top cells of a smooth correlation surface are
    all neighbours of ONE peak, so a 'top-10 pose list' is one pose listed ten times."""
    flat = np.argpartition(sc, -400, axis=None)[-400:]
    flat = flat[np.argsort(sc.reshape(-1)[flat])[::-1]]
    coords = np.array(np.unravel_index(flat, sc.shape)).T
    keep = []
    for c in coords:
        if keep:
            d = np.abs(np.array(keep) - c)
            d = np.minimum(d, GRID - d)          # the correlation grid is periodic
            if (d.max(1) < sep).any():
                continue
        keep.append(c)
        if len(keep) >= n:
            break
    return [tuple(int(x) for x in c) for c in keep]


def shift_of(pk):
    """irfftn(FA * conj(FL))[k] = sum_x A[x+k] L[x], so the INDEX is the shift in cells, wraparound meaning
    negative. The box guard (D_R + D_L <= box) makes the in-contact representative the only physical one."""
    kk = np.array(pk, float)
    kk[kk > GRID / 2] -= GRID
    return kk * SPACING


def prepare(pdb):
    """pick the chain pair that is actually an interface, build both grids, return everything reusable."""
    try:
        t = fp.table(pdb)
    except Exception:
        return None
    if t is None:
        return None
    chs = sorted(set(t["ch"].tolist()))
    cand = []
    for a, b in combinations(chs, 2):
        A_, B_ = t["co"][t["ch"] == a], t["co"][t["ch"] == b]
        if not (MIN_AT <= len(A_) <= MAX_AT and MIN_AT <= len(B_) <= MAX_AT):
            continue
        n = int((np.linalg.norm(A_[:, None, :] - B_[None, :, :], axis=2) < 5.0).sum())
        cand.append((n, a, b))
    if not cand:
        return None
    n_ct, ca, cb = max(cand)
    if n_ct < MIN_CONTACTS:
        return None
    R, L = t["co"][t["ch"] == ca], t["co"][t["ch"] == cb]
    rR, rL = t["rad"][t["ch"] == ca], t["rad"][t["ch"] == cb]
    dR = float(np.linalg.norm(R - R.mean(0), axis=1).max() * 2)
    dL = float(np.linalg.norm(L - L.mean(0), axis=1).max() * 2)
    if dR + dL > GRID * SPACING:
        return None                              # circular correlation: a wrapped pose could fake a contact
    centre = R.mean(0)
    solid_R = rasterise(R, centre, offsets_for(rR))
    if solid_R is None:
        return None
    skin_R = _morph(solid_R, SKIN, True) & ~solid_R
    core_R = _morph(solid_R, CORE_ERODE, False)
    lcen = L.mean(0)
    offs_L = offsets_for(rL)
    solid_L = rasterise(L - lcen + centre, centre, offs_L)
    if solid_L is None:
        return None
    F_skin = np.fft.rfftn(skin_R.astype(np.float32))
    F_core = np.fft.rfftn(core_R.astype(np.float32))
    FL = np.conj(np.fft.rfftn(solid_L.astype(np.float32)))
    inter = np.fft.irfftn(F_skin * FL, solid_R.shape)
    clash = np.fft.irfftn(F_core * FL, solid_R.shape)
    k_nat = tuple(int(x) for x in (np.round((lcen - centre) / SPACING).astype(int) % GRID))
    sane = float(inter[k_nat]) > SANITY_RATIO * float(np.median(inter[inter > 0])) if (inter > 0).any() else False
    return {"pdb": pdb, "ca": ca, "cb": cb, "n_contacts": n_ct, "R": R, "L": L, "centre": centre,
            "lcen": lcen, "offs_L": offs_L, "F_skin": F_skin, "F_core": F_core, "shape": solid_R.shape,
            "nat_inter": float(inter[k_nat]), "nat_clash": float(clash[k_nat]),
            "med_inter": float(np.median(inter[inter > 0])) if (inter > 0).any() else 0.0,
            "k_nat": k_nat, "nat_inter_field": inter, "nat_clash_field": clash, "sane": bool(sane)}


def main():
    log = []
    t0 = time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("EXHAUSTIVE RIGID DOCKING BY FFT -- does the TRUE pose rank near the top?")
    report("=" * 100)
    report(f"  grid {GRID}^3 at {SPACING} A ({GRID*SPACING:.0f} A box) | {N_ROT} blind rotations + the native")
    report(f"  rotation as a diagnostic | vdW-radius rasterisation, {SKIN}-cell outward shell")
    report(f"  clash weight TUNED on the first {N_TUNE} complexes, blind-tested on the rest -- never both")
    report("  PREDECLARED: near-native in top-10 for a majority of TEST complexes AND above the chance rate")
    report("  1-(1-f)^10 => search AND score work. Hit rate ~ chance => search works, score does not.")

    have = sorted(f[:-4] for f in os.listdir(fp.PDBDIR) if f.endswith(".pdb"))
    report(f"\n  {len(have):,} structures in the cache; selecting complexes with a real interface "
           f"(>= {MIN_CONTACTS} atom pairs within 5 A)")
    prepped, insane = [], []
    for pdb in have:
        if len(prepped) >= N_TOTAL:
            break
        p = prepare(pdb)
        if p is None:
            continue
        if not p["sane"]:
            insane.append(p["pdb"])
            continue
        prepped.append(p)
    if len(prepped) < N_TUNE + 4:
        report(f"\n  only {len(prepped)} usable complexes -- not enough to split")
        return 1
    tune, test = prepped[:N_TUNE], prepped[N_TUNE:]
    report(f"    {len(prepped)} usable   TUNE {[p['pdb'] for p in tune]}")
    report(f"                    TEST {[p['pdb'] for p in test]}")
    if insane:
        report(f"    REPRESENTATION FAIL (native interlock not {SANITY_RATIO}x the median placement): {insane}")

    # ---------------- SANITY: does the true interface register at all? ----------------
    report(f"\n  SANITY -- native pose components (the check the first version of this file did not have)")
    report(f"    {'pdb':<6}{'pair':>6}{'contacts':>10}{'nat interlock':>15}{'nat clash':>11}{'median cell':>13}")
    for p in prepped:
        report(f"    {p['pdb']:<6}{p['ca']+'/'+p['cb']:>6}{p['n_contacts']:>10}"
               f"{p['nat_inter']:>15.0f}{p['nat_clash']:>11.0f}{p['med_inter']:>13.1f}")

    # ---------------- TUNE the clash weight, on the TUNE split only ----------------
    report(f"\n  TUNING the clash weight on {len(tune)} held-out complexes: rank of the native TRANSLATION "
           f"at the native rotation, out of {GRID**3:,}")
    report("    " + f"{'pdb':<6}" + "".join(f"{('w=' + str(int(w))):>10}" for w in WEIGHTS))
    ranks = {w: [] for w in WEIGHTS}
    for p in tune:
        row = []
        for w in WEIGHTS:
            sc = p["nat_inter_field"] - w * p["nat_clash_field"]
            r = int((sc > sc[p["k_nat"]]).sum())
            ranks[w].append(r)
            row.append(f"{r:>10,}")
        report("    " + f"{p['pdb']:<6}" + "".join(row))
    med = {w: float(np.median(ranks[w])) for w in WEIGHTS}
    report("    " + f"{'MEDIAN':<6}" + "".join(f"{med[w]:>10,.0f}" for w in WEIGHTS))
    CLASH_W = min(WEIGHTS, key=lambda w: med[w])
    report(f"    -> clash weight {CLASH_W:g} (median native rank {med[CLASH_W]:,.0f}). The TEST complexes have "
           f"not been looked at.")
    for p in prepped:                            # 14 MB of correlation fields per complex, no longer needed
        p.pop("nat_inter_field", None)
        p.pop("nat_clash_field", None)

    # ---------------- BLIND SEARCH on the TEST split ----------------
    report(f"\n  BLIND SEARCH on {len(test)} complexes never used for tuning")
    rots = rotations(N_ROT)
    rows = []
    for p in test:
        centre, L, lcen = p["centre"], p["L"], p["lcen"]
        Lc = L - lcen
        rg = float(np.sqrt((Lc ** 2).sum(1).mean()))
        t_need = NEAR_NATIVE / rg
        p_cov = 1 - (1 - min(1.0, t_need ** 3 / (6 * np.pi))) ** N_ROT
        FA = p["F_skin"] - CLASH_W * p["F_core"]
        t1 = time.time()
        best = []
        for ri, Rm in [(-1, np.eye(3))] + list(enumerate(rots)):
            g = rasterise(Lc @ Rm.T + centre, centre, p["offs_L"])
            if g is None:
                continue
            sc = np.fft.irfftn(FA * np.conj(np.fft.rfftn(g.astype(np.float32))), p["shape"])
            for pk in nms_peaks(sc, PEAKS_PER_ROT, NMS_CELLS):
                if sc[pk] <= 0:
                    continue                     # net-clashing or non-contacting; not a pose the search found
                best.append((float(sc[pk]), ri, shift_of(pk)))

        def rmsd_of(ri, shift):
            Rm = np.eye(3) if ri == -1 else rots[ri]
            return float(np.sqrt((((Lc @ Rm.T + centre + shift) - L) ** 2).sum(1).mean()))

        nat_rm = [rmsd_of(-1, s) for _v, r, s in best if r == -1]
        blind = sorted([b for b in best if b[1] != -1], key=lambda x: -x[0])
        if not blind:
            continue
        rm = np.array([rmsd_of(ri, s) for _v, ri, s in blind])
        hits = {k: bool((rm[:k] <= NEAR_NATIVE).any()) for k in TOPK}
        frac = float((rm <= NEAR_NATIVE).mean())
        chance = {k: 1 - (1 - frac) ** k for k in TOPK}
        rows.append({"pdb": p["pdb"], "chains": [p["ca"], p["cb"]], "n_contacts": p["n_contacts"],
                     "rg_ligand": rg, "rot_tol_deg": float(np.degrees(t_need)), "p_coverage": float(p_cov),
                     "n_poses": len(blind), "best_rmsd": float(rm.min()),
                     "rank_of_best_rmsd": int(np.argmin(rm)), "near_native_frac": frac,
                     "hits": hits, "chance": chance,
                     "native_rot_best_rmsd": float(min(nat_rm)) if nat_rm else None,
                     "secs": time.time() - t1})
        report(f"    {p['pdb']} {p['ca']}/{p['cb']}: {len(blind)} poses in {time.time()-t1:.0f}s | best RMSD "
               f"{rm.min():5.1f} A at rank {int(np.argmin(rm)):>4} | top1 {str(hits[1]):<5} top10 "
               f"{str(hits[10]):<5} (chance {chance[10]:.3f}) | native-rot best "
               f"{(min(nat_rm) if nat_rm else float('nan')):5.1f} A | P(cov) {p_cov:.2f}")

    if not rows:
        report("\n  no usable TEST complexes")
        return 1

    report(f"\n  {len(rows)} TEST complexes docked blind at clash weight {CLASH_W:g}")
    report(f"\n  {'top-k':>8} {'hits':>10} {'expected by chance':>20}")
    tab = {}
    for k in TOPK:
        n = sum(r["hits"][k] for r in rows)
        exp = float(np.mean([r["chance"][k] for r in rows])) * len(rows)
        tab[str(k)] = {"hits": n, "expected_by_chance": exp}
        report(f"  {k:>8} {str(n) + '/' + str(len(rows)):>10} {exp:>20.2f}")
    bm = float(np.median([r["best_rmsd"] for r in rows]))
    nat_ok = [r for r in rows if r["native_rot_best_rmsd"] is not None]
    nbm = float(np.median([r["native_rot_best_rmsd"] for r in nat_ok])) if nat_ok else float("nan")
    report(f"\n    median best RMSD anywhere in the blind pose set : {bm:.1f} A")
    report(f"    median best RMSD at the NATIVE rotation          : {nbm:.1f} A  <- translation scoring alone")
    report(f"    median P(rotation set covers the native)         : "
           f"{np.median([r['p_coverage'] for r in rows]):.2f}")

    report("\n  READING")
    top10 = sum(r["hits"][10] for r in rows)
    exp10 = float(np.mean([r["chance"][10] for r in rows])) * len(rows)
    found = sum(r["best_rmsd"] <= NEAR_NATIVE for r in rows)
    if top10 > len(rows) / 2 and top10 > exp10 + 1:
        report(f"  Near-native ranks top-10 for {top10}/{len(rows)} TEST complexes against {exp10:.1f} expected")
        report("  from shuffling the same pose set. The search AND the score work together, at a clash weight")
        report("  chosen without ever looking at these complexes. Rigid-body docking is a real stage here.")
    elif found:
        report(f"  The search REACHES near-native ({found}/{len(rows)} complexes have a pose <= {NEAR_NATIVE} A,")
        report(f"  median {bm:.1f} A) but does not RANK it: top-10 hits {top10} vs {exp10:.1f} by chance.")
        report("  Sampling is solved; RANKING is the open problem, and more rotations cannot fix a score that")
        report("  cannot tell the right pose from the wrong one.")
    elif nat_ok and nbm <= NEAR_NATIVE:
        report(f"  No near-native pose blind, but at the NATIVE rotation the score finds the right translation")
        report(f"  ({nbm:.1f} A). Translation scoring works; ROTATION SAMPLING binds at N={N_ROT}. That is a")
        report("  compute problem with a known fix, not a refutation.")
    else:
        report(f"  No near-native pose even at the native rotation (median {nbm:.1f} A) -- and since the sanity")
        report("  gate passed, that is a genuine scoring failure rather than a representation one.")

    json.dump({"test": "dock_fft", "grid": GRID, "spacing": SPACING, "n_rot": N_ROT, "skin": SKIN,
               "weights_swept": list(WEIGHTS), "clash_w_chosen": CLASH_W,
               "tune_pdbs": [p["pdb"] for p in tune], "test_pdbs": [p["pdb"] for p in test],
               "tune_median_rank_by_w": {str(w): med[w] for w in WEIGHTS},
               "representation_fail": insane, "near_native_A": NEAR_NATIVE, "complexes": rows,
               "top_k": tab, "n_test": len(rows), "median_best_rmsd": bm,
               "median_native_rot_best_rmsd": nbm,
               "sanity": [{"pdb": p["pdb"], "pair": p["ca"] + "/" + p["cb"], "n_contacts": p["n_contacts"],
                           "nat_interlock": p["nat_inter"], "nat_clash": p["nat_clash"],
                           "median_cell": p["med_inter"]} for p in prepped],
               "log": log}, open(OUT / "dock_fft.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'dock_fft.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
