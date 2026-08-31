"""Is interface conformational entropy a usable ranking signal, tested OFF the collider?

WHY THIS RUN EXISTS. The previous basin study computed exact side-chain partition functions
on 20-pose shortlists and reported two things: that basin breadth T*S_conf tracks nativeness
at Spearman -0.45, and that reranking by free energy achieves 0/58 against a ceiling that was
ALSO 0/58 (ledger defect N -- a bar above the achievable ceiling measures nothing).

Both numbers were computed on poses SELECTED BY THE GRID SCORE, and that selection invalidates
them in two different ways:

  THE CEILING. A shortlist chosen by a scorer, whose ceiling is then used to characterise the
  SEARCH, is circular. "No acceptable pose in the top 20" is a fact about the scorer that built
  the top 20.

  THE CORRELATION. Selecting on grid score and then correlating two other quantities inside the
  selected set is CONDITIONING ON A COLLIDER. Grid score is a shape-complementarity term; so,
  plausibly, are both T*S_conf and I_rmsd. Conditioning on a common effect of two variables
  induces an association between them even when none exists in the population. The -0.45 is
  therefore not yet evidence about protein interfaces; it is evidence about interfaces THAT
  SCORE WELL ON A SHAPE TERM. This run can withdraw it.

AND THE DEFECT IS ONE LEVEL DEEPER THAN IT LOOKS. The obvious repair -- use all ~6000
candidates run_arm evaluates instead of the top 20 -- DOES NOT WORK, and an adversarial review
of this design caught it before the run. Those 6000 are 2000 rotations x 3 translations, and
the three come from fftcorr.top_translations(S, k=3), which returns THE THREE HIGHEST-SCORING
VOXELS OF THE WHOLE FFT VOLUME. The translation axis is still selected by the scorer under
test, so a ceiling computed on them is still a fact about the score. There is no "unselected
search output" sitting in the existing pipeline waiting to be read off.

So this module does not read a candidate list. It reconstructs the search's REACHABLE SET --
every rotation crossed with every grid translation the FFT ranks -- and interrogates it
directly, in two ways that are both free of the score's ranking.

=================================================================================================
THE GATES, FIXED HERE BEFORE ANY NUMBER IS RUN. Analysis lives in db5_unselected_analysis.py.
=================================================================================================

STEP 0  THE REACHABLE-SET CEILING, computed analytically and WITHOUT EVER CONSULTING THE SCORE.
        For a fixed rotation R the ligand is (x - c) @ R.T + c + t, so displacement from native
        is affine in t and every RMSD is a QUADRATIC in t:
            rmsd(t)^2 = rmsd_min^2 + ||t - t*||^2,   t* = -mean(displacement at t = 0).
        The translations achieving rmsd <= b are therefore exactly a BALL of radius
        sqrt(b^2 - rmsd_min^2) about t*, empty when rmsd_min > b. So the translations that could
        possibly give a CAPRI-acceptable pose can be ENUMERATED IN CLOSED FORM rather than
        searched for, for all 2000 rotations, at the cost of one small matmul per rotation.
        CAPRI acceptable requires f_nat >= 0.1 AND (L_rmsd <= 10 OR I_rmsd <= 4).
          - The L_rmsd <= 10 branch is handled EXACTLY by the ball above on backbone atoms.
          - The I_rmsd <= 4 branch is a SUPERIMPOSED rmsd, which is not affine in t, so it is
            screened by the ball on the DIRECT (unsuperimposed) interface rmsd at a deliberately
            loose 12 A. Superposition can only lower an rmsd, so this is a screen and not a
            proof; it is therefore VALIDATED, NOT ASSUMED (ledger defect M). Every pose
            evaluated exactly in this run records both quantities, and the run reports the
            largest direct interface rmsd ever seen on a pose whose exact I_rmsd was <= 4. If
            that approaches 12 the screen is not safe and the run says so instead of a ceiling.
        Exact capri_metrics is then computed on every enumerated translation.
        CEILING := complexes with >= 1 CAPRI-acceptable pose anywhere in the reachable set.
        THE ENUMERATION IS CAPPED at MAX_ENUM per complex and the cap is REPORTED, because a
        silent cap reads as "we looked everywhere" when it is not.
        IF THE CEILING IS 0, that is the result and it is a positive finding, not a failure:
        this rotation set and this translation grid CANNOT produce an acceptable pose, so no
        ranking of their output could ever have succeeded, the docking line closes for good,
        and every gate below is VOID -- not NULL, not FAILED.

STEP 1  THE SAMPLE, drawn from poses the score ADMITS but does not RANK. A pose needs an
        interface for T*S_conf to be defined at all, so the universe cannot be all of space;
        but "has an interface" must not become "scores well". The admissibility criterion is
        S(t) > 0 -- the Katchalski-Katzir correlation is net favourable, i.e. the pose touches
        without gross clash. Within that set translations are drawn UNIFORMLY AT RANDOM, never
        by rank: R_SAMPLE rotations uniformly from the same 2000, T_PER_ROT translations
        uniformly from each rotation's admissible set. The pool is then stratified into N_BINS
        equal-width I_rmsd bins so the near-native end is represented if it exists at all.
        Every acceptable pose from step 0 is force-included.
        THE UPPER BIN EDGE IS THE 99TH PERCENTILE, NOT THE MAXIMUM, so one far outlier cannot
        set the bin width for everything else (ledger defect L: a scale taken from an extreme
        order statistic is not a scale).
        THIS SAMPLE IS DELIBERATELY ENRICHED IN NEAR-NATIVE POSES. Every retrieval number below
        is a RETRIEVAL RATE ON AN ENRICHED SET, NOT a docking success rate, and must never be
        read against the prior 0/58. The enrichment factor is reported beside it.

STEP 2  Q1 RETEST -- DOES THE CORRELATION SURVIVE OFF THE COLLIDER? The bars are the ORIGINAL
        ones, so this is a replication and not a new hypothesis: Spearman(T*S_conf, I_rmsd)
        <= -0.10, negative on a majority of complexes, and surviving the size control.
        THREE THINGS THE ORIGINAL GOT WRONG ARE FIXED HERE, ALL FOUND BY REVIEW BEFORE THE RUN:
          (a) DEGENERATE POSES ARE EXCLUDED, NOT SCORED AS ZERO. score_pose returned TS = 0.0
              when fewer than two interface residues are repackable. That is a NOT-APPLICABLE
              written as the extreme value of the entropy axis, in the exact direction the
              hypothesis predicts, on exactly the poses (tiny interfaces) that also sit at high
              I_rmsd. Coding a missing value as the most confirmatory possible number
              manufactures the correlation being tested. TS is nan here, and Q1 is computed on
              non-degenerate poses only, with the count reported.
          (b) THE SIZE CONTROL USES A GEOMETRIC CONTACT COUNT, NOT THE GRID SCORE. The prior run
              controlled for "interface size" with abs(grid score) -- the scorer under test --
              which conditions on the very collider the redesign exists to escape. Here size is
              the number of receptor-ligand heavy-atom pairs within 5 A, computed from
              coordinates, plus n_repack, the number of residues that actually became variables,
              which is what makes T*S_conf extensive. Both are partialled out.
          (c) THE INFERENCE UNIT IS THE COMPLEX, NOT THE POSE. Pooling tens of thousands of
              poses across 58 complexes and applying a normal approximation treats clustered
              poses as independent; at that n, p < 1e-3 is passed by |rho| >= 0.02 and carries
              no information. The reported test is a sign test over the per-complex Spearmans.
        AND THE COLLIDER IS MEASURED, NOT ARGUED. Within this one sample, holding the pose
        universe fixed, the same correlation is recomputed on the top 20 poses BY GRID SCORE per
        complex, reproducing the prior study's selection rule on the new data. If it is much
        stronger in that slice than in the whole sample, the -0.45 was selection. That is the
        matched comparison the prior design could not make.

STEP 3  THE RETRIEVAL TEST, with its power checked BEFORE its verdict. Rank each complex's
        sample and ask how often an acceptable pose reaches the top 20:
          (a) energy      E_min ascending
          (b) entropy     T*S_conf descending
          (c) 50/50       mean of the within-complex rank-normalised (a) and (b)
          (d) size        contact count descending          <- THE CONTROL ARM
        (d) exists because acceptable poses ARE large-interface poses, so a ranking that merely
        prefers big interfaces beats random without carrying any nativeness information.
        Beating random is not the interesting bar; beating (d) is.
        CHANCE BASELINE, per complex, since complexes differ in how many acceptable poses they
        hold and a fixed count would be trivially easier for some (ledger defect K): with k
        acceptable of N sampled, a random top-20 hits with p = 1 - C(N-k,20)/C(N,20). The null
        over complexes is the POISSON-BINOMIAL of those p, and its tail is computed EXACTLY by
        dynamic programming, not by a normal approximation, which misstates the tail of a
        small-mean skewed discrete distribution. Signal is declared at one-sided exact p < 0.05.
        THE POWER CHECK IS PART OF THE GATE, NOT AN AFTERTHOUGHT (ledger defect N, recorded one
        commit ago, which this design nearly recommitted). Force-including acceptable poses
        raises k, which raises chance, which can push the bar above the maximum achievable
        score. So before any ranking is evaluated the run computes what a PERFECT ORACLE would
        score -- the number of complexes with >= 1 acceptable pose in the sample -- and if the
        oracle itself cannot clear the bar, STEP 3 IS VOID and reports that instead of a verdict.
        THE HYPOTHESIS IS THAT ENTROPY ADDS TO ENERGY, so the headline is the PAIRED comparison
        of (c) against (a) on the same complexes, not each against chance separately.
        The lambda sweep of E + lambda*(-T*S_conf) is a DIAGNOSTIC. 50/50 is the predeclared
        point and the sweep is not to be mined for a winning lambda afterwards.

WHAT THIS RUN CANNOT DO. It holds the rotation set and the translation grid fixed. It says
nothing about whether a finer or smarter search would find acceptable poses -- only whether
THIS reachable set contains any, and whether entropy can retrieve them from what it admits.
"""
from __future__ import annotations

import argparse, json, sys, time, zlib
sys.path.insert(0, ".")
import numpy as np

from rem import fftcorr
from rem.docking import capri, score
from rem.docking.data import load_case, superimposed_rmsd
from rem.docking.repack import build_from_case
from rem.docking.freeenergy import free_energy
from rem.docking.rigid import RigidSearch, apply_pose, randomize_pose, rotation_set
from benchmarks.db5_dock import _as_struct
from benchmarks.db5_basin import subset_ids

ROTATIONS = 2000
SPACING = 1.5
L_BAR, I_BAR = 10.0, 4.0                 # the CAPRI acceptable rmsd bars themselves
I_DIRECT_SCREEN = 12.0                   # loose screen for the non-affine I_rmsd branch
MAX_ENUM = 40000                         # cap on enumerated translations; reported if it binds
R_SAMPLE, T_PER_ROT = 80, 40             # sampling budget: rotations, translations per rotation
N_BINS, PER_BIN, N_SAMPLE = 25, 20, 500
REPACK_RES, N_CHI1, N_CHI2 = 6, 3, 2
CONTACT_CUT = 5.0
OK = ("high", "medium", "acceptable")


def quad_ball(disp0, bar):
    """Translations with rmsd(t) <= bar, in closed form.

    rmsd(t)^2 = mean(||d_i + t||^2) = ||t||^2 + 2 t.mean(d) + mean(||d||^2)
              = rmsd_min^2 + ||t - t*||^2,   t* = -mean(d).
    Returns (t*, rmsd_min, radius); radius is nan when the ball is empty.
    """
    m = disp0.mean(axis=0)
    rmin2 = max(float((disp0 * disp0).sum(1).mean() - float((m * m).sum())), 0.0)
    rad2 = bar * bar - rmin2
    return -m, float(np.sqrt(rmin2)), (float(np.sqrt(rad2)) if rad2 > 0 else float("nan"))


def shifts_in_ball(tstar, radius, spacing, shape):
    """Integer voxel shifts whose world translation lies in the ball AND that the FFT fold can
    represent -- so this enumerates only poses the search could really produce."""
    if not np.isfinite(radius):
        return np.empty((0, 3), dtype=int)
    n = np.asarray(shape, dtype=int)
    off = n // 2
    c = np.asarray(tstar, float) / spacing
    r = radius / spacing
    rngs = []
    for a in range(3):
        lo = max(int(np.ceil(c[a] - r)), -int(off[a]))
        hi = min(int(np.floor(c[a] + r)), int(n[a] - off[a] - 1))
        if hi < lo:
            return np.empty((0, 3), dtype=int)
        rngs.append(np.arange(lo, hi + 1))
    G = np.stack(np.meshgrid(*rngs, indexing="ij"), axis=-1).reshape(-1, 3)
    d = G - c
    return G[(d * d).sum(1) <= r * r]


def contact_count(rec_coords, lig_coords, cut=CONTACT_CUT):
    """Geometric interface size. NOT the grid score -- that is the scorer under test."""
    from scipy.spatial import cKDTree
    return int(cKDTree(rec_coords).query_ball_point(lig_coords, r=cut,
                                                    return_length=True).sum())


class FastCapri:
    """capri_metrics for many poses of ONE complex, with the per-complex work hoisted out.

    capri.f_nat recomputes the NATIVE contact set on every call and forms a full
    (n_rec x n_lig) distance matrix twice; both are constants of the complex, and they are
    what makes an exhaustive ceiling scan unaffordable. Only pairs already in the native
    contact set can contribute to |nat & dock|, so per pose the work is a contact check over
    the atoms of those residues alone.

    This is an OPTIMISATION, NOT A REDEFINITION, and the test suite asserts it agrees with
    capri.capri_metrics exactly on real poses.
    """

    def __init__(self, rec, lig, native, masks, cutoff=capri.CONTACT_CUTOFF):
        rr, lr = capri._res_index(rec), capri._res_index(lig)
        nat = sorted(capri.contact_set(rec.coords, rr, native, lr, cutoff))
        self.n_nat = len(nat)
        self.cut2 = float(cutoff) ** 2
        rres = sorted({a for a, _b in nat})
        lres = sorted({b for _a, b in nat})
        self.rsel = np.where(np.isin(rr, rres))[0]
        self.lsel = np.where(np.isin(lr, lres))[0]
        rmap = {r: i for i, r in enumerate(rres)}
        lmap = {l: i for i, l in enumerate(lres)}
        self.r_of = np.array([rmap[x] for x in rr[self.rsel]], dtype=int)
        self.l_of = np.array([lmap[x] for x in lr[self.lsel]], dtype=int)
        self.pairid = np.full((len(rres), len(lres)), -1, dtype=int)
        for k, (a, b) in enumerate(nat):
            self.pairid[rmap[a], lmap[b]] = k
        self.rec_sel = rec.coords[self.rsel]
        rmask, lmask = masks
        self.lmask = lmask
        self.lig_bb = np.isin(lig.atom_names, capri.BACKBONE)
        self.nat_bb = native[self.lig_bb]
        self.rec_if = rec.coords[rmask]
        self.Q = np.vstack([self.rec_if, native[lmask]])

    def metrics(self, docked):
        lrms = (float(np.sqrt(((docked[self.lig_bb] - self.nat_bb) ** 2).sum(1).mean()))
                if self.lig_bb.any() else float("nan"))
        P = np.vstack([self.rec_if, docked[self.lmask]])
        irms = float(superimposed_rmsd(P, self.Q)) if len(P) >= 3 else float("nan")
        if self.n_nat == 0:
            fn = float("nan")
        else:
            L = docked[self.lsel]
            d2 = ((self.rec_sel[:, None, :] - L[None, :, :]) ** 2).sum(-1)
            i, j = np.nonzero(d2 <= self.cut2)
            ids = self.pairid[self.r_of[i], self.l_of[j]] if len(i) else np.empty(0, int)
            fn = float(np.unique(ids[ids >= 0]).size) / self.n_nat
        return {"f_nat": fn, "L_rmsd": lrms, "I_rmsd": irms,
                "quality": capri.capri_quality(fn, lrms, irms)}


def score_pose_ext(rec, lig_at_pose, grid_score):
    """As db5_basin.score_pose, plus what the size control actually needs: n_repack (the number
    of residues that became variables, which is what makes T*S_conf extensive) and a geometric
    contact count."""
    rq = score.charges(rec.res_names, rec.atom_names)
    pair = score.pair_energy(rec.coords, rec.elements, rq, lig_at_pose.coords,
                             lig_at_pose.elements,
                             score.charges(lig_at_pose.res_names,
                                           lig_at_pose.atom_names))["total"]
    nc = contact_count(rec.coords, lig_at_pose.coords)
    prob = build_from_case({"r_b": rec, "l_b": lig_at_pose}, side="r", bound=True,
                           max_residues=REPACK_RES, n_chi1=N_CHI1, n_chi2=N_CHI2)
    base = {"grid": -grid_score, "pair": pair, "contacts": nc}
    if len(prob.res_keys) < 2:
        # NOT-APPLICABLE. TS is nan, never 0.0: zero is the extreme of the entropy axis in the
        # direction the hypothesis predicts, so coding a missing value as zero manufactures the
        # correlation. Downstream code excludes these from Q1.
        return {**base, "ve": pair, "greedy": pair, "F": pair, "TS": float("nan"),
                "treewidth": 0, "n_repack": int(len(prob.res_keys)), "degenerate": True}
    g, _e = prob.to_factorgraph()
    ex = prob.solve_exact(g)
    gr = prob.solve_greedy(g, restarts=20)
    fe = free_energy(prob, energy_graph=g)
    return {**base, "ve": ex["energy"], "greedy": gr["energy"], "F": fe["F"],
            "TS": fe["TS_conf"], "treewidth": int(ex["treewidth"]),
            "n_repack": int(len(prob.res_keys)), "degenerate": False}


def run_complex(cid, cls, rots, n_sample):
    case = load_case(cid)
    rec, lig = case["r_u"], case["l_u"]
    seed = zlib.crc32(cid.encode()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    native = lig.coords.copy()
    masks = capri.interface_mask(rec, lig, native)
    rmask, lmask = masks
    moved, _R, _t = randomize_pose(native, seed=seed, max_shift=20.0)
    srch = RigidSearch(rec, _as_struct(lig, moved), spacing=SPACING)
    centre, shape = srch.lig_centre, srch.shape

    lig_bb = np.isin(lig.atom_names, capri.BACKBONE)
    nat_bb, sub_bb = native[lig_bb], moved[lig_bb]
    nat_if, sub_if = native[lmask], moved[lmask]
    rec_if = rec.coords[rmask]
    Q = np.vstack([rec_if, nat_if])

    fast = FastCapri(rec, lig, native, masks)

    def metrics_at(R, shift):
        t = fftcorr.shift_to_world(shift, SPACING)
        c = apply_pose(moved, R, t, centre=centre)
        return c, fast.metrics(c)

    def direct_if(R, shift):
        t = fftcorr.shift_to_world(shift, SPACING)
        sub = (sub_if - centre) @ R.T + centre + t
        return float(np.sqrt(((sub - nat_if) ** 2).sum(1).mean()))

    def cheap_irmsd(R, shift):
        t = fftcorr.shift_to_world(shift, SPACING)
        sub = (sub_if - centre) @ R.T + centre + t
        return float(superimposed_rmsd(np.vstack([rec_if, sub]), Q))

    # ---------------- STEP 0: reachable-set ceiling, no score consulted ----------------
    t0 = time.perf_counter()
    enum, capped, lmins, imins = [], False, [], []
    for ri, R in enumerate(rots):
        bb0 = (sub_bb - centre) @ R.T + centre
        if0 = (sub_if - centre) @ R.T + centre
        tL, Lmin, radL = quad_ball(bb0 - nat_bb, L_BAR)
        tI, Imin, radI = quad_ball(if0 - nat_if, I_DIRECT_SCREEN)
        lmins.append(Lmin); imins.append(Imin)
        if not (np.isfinite(radL) or np.isfinite(radI)):
            continue
        s1 = shifts_in_ball(tL, radL, SPACING, shape)
        s2 = shifts_in_ball(tI, radI, SPACING, shape)
        if len(s1) and len(s2):
            allsh = np.unique(np.vstack([s1, s2]), axis=0)
        else:
            allsh = s1 if len(s1) else s2
        for sh in allsh:
            if len(enum) >= MAX_ENUM:
                capped = True
                break
            enum.append((ri, tuple(int(x) for x in sh)))
        if capped:
            break
    accept, probe = [], []
    for ri, sh in enum:
        shv = np.array(sh)
        _c, m = metrics_at(rots[ri], shv)
        probe.append((m["I_rmsd"], direct_if(rots[ri], shv)))
        if m["quality"] in OK:
            accept.append({"rot": int(ri), "shift": list(sh), **m})
    t_ceil = time.perf_counter() - t0

    # ---------------- STEP 1: poses the score ADMITS but does not RANK ----------------
    ridx = rng.choice(len(rots), size=min(R_SAMPLE, len(rots)), replace=False)
    pool = []
    for ri in ridx:
        S = srch.score_rotation(rots[int(ri)])
        adm = np.argwhere(S > 0.0)
        if len(adm) == 0:
            continue
        pick = adm[rng.choice(len(adm), size=min(T_PER_ROT, len(adm)), replace=False)]
        for raw in pick:
            sh = fftcorr._fold(np.asarray(raw, int), S.shape, None)
            pool.append((int(ri), tuple(int(x) for x in sh),
                         float(S[tuple(int(x) for x in raw)])))
    if not pool:
        raise RuntimeError("no admissible poses")
    pool_I = np.array([cheap_irmsd(rots[ri], np.array(sh)) for ri, sh, _s in pool])

    lo, hi = float(pool_I.min()), float(np.percentile(pool_I, 99))
    edges = np.linspace(lo, max(hi, lo + 1e-6), N_BINS + 1)
    which = np.clip(np.digitize(pool_I, edges) - 1, 0, N_BINS - 1)
    chosen = []
    for b in range(N_BINS):
        m = np.where(which == b)[0]
        if len(m):
            chosen += [int(x) for x in rng.choice(m, size=min(PER_BIN, len(m)), replace=False)]
    chosen = sorted(set(chosen))
    if len(chosen) < n_sample:
        rest = np.setdiff1d(np.arange(len(pool)), np.array(chosen, int))
        if len(rest):
            chosen = sorted(set(chosen) | {int(x) for x in rng.choice(
                rest, size=min(n_sample - len(chosen), len(rest)), replace=False)})

    # ---------------- STEP 2: exact metrics + partition function on the sample ----------
    recs = []
    for j in chosen:
        ri, sh, sc = pool[j]
        c, m = metrics_at(rots[ri], np.array(sh))
        try:
            s = score_pose_ext(rec, _as_struct(lig, c), sc)
        except Exception:                                          # noqa: BLE001
            continue
        recs.append({**s, "I_rmsd": m["I_rmsd"], "L_rmsd": m["L_rmsd"], "f_nat": m["f_nat"],
                     "quality": m["quality"], "rot": int(ri), "forced": False})
    for a in accept:                       # force-include every acceptable pose from step 0
        S = srch.score_rotation(rots[a["rot"]])
        raw = tuple(int(x) % int(n) for x, n in zip(a["shift"], S.shape))
        c = apply_pose(moved, rots[a["rot"]],
                       fftcorr.shift_to_world(np.array(a["shift"]), SPACING), centre=centre)
        try:
            s = score_pose_ext(rec, _as_struct(lig, c), float(S[raw]))
        except Exception:                                          # noqa: BLE001
            continue
        recs.append({**s, "I_rmsd": a["I_rmsd"], "L_rmsd": a["L_rmsd"], "f_nat": a["f_nat"],
                     "quality": a["quality"], "rot": int(a["rot"]), "forced": True})

    sp = np.array(probe) if probe else np.zeros((0, 2))
    near = sp[sp[:, 0] <= I_BAR] if len(sp) else sp
    return {
        "id": cid, "class": cls,
        "ceiling": {"n_enumerated": len(enum), "capped": bool(capped),
                    "n_acceptable": len(accept),
                    "min_L_rmsd_over_rotations": float(np.min(lmins)),
                    "min_direct_I_over_rotations": float(np.min(imins)),
                    "seconds": t_ceil},
        # Screen validation: the largest DIRECT interface rmsd seen on a pose whose exact
        # I_rmsd was already inside the CAPRI bar. If this approaches I_DIRECT_SCREEN the
        # screen is not safe, and this is how the run finds that out rather than assuming.
        "screen": {"n_probe": int(len(sp)), "n_within_I_bar": int(len(near)),
                   "max_direct_given_I_ok": (float(near[:, 1].max()) if len(near) else None),
                   "limit": I_DIRECT_SCREEN},
        "pool": {"n": len(pool), "I_lo": lo, "I_hi99": hi, "I_max": float(pool_I.max())},
        "poses": recs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--out", default="benchmarks/unsel_w0.json")
    a = ap.parse_args(argv)

    ids = subset_ids()
    if a.limit:
        ids = ids[:a.limit]
    if a.nworkers > 1:
        ids = [x for i, x in enumerate(ids) if i % a.nworkers == a.worker]
    rots = rotation_set(ROTATIONS, seed=1)
    print(f"  {len(ids)} complexes, {ROTATIONS} rotations, sample {a.sample}", flush=True)

    out, t0 = [], time.perf_counter()
    for n, (cid, cls) in enumerate(ids, 1):
        try:
            r = run_complex(cid, cls, rots, a.sample)
        except Exception as e:                                     # noqa: BLE001
            print(f"  {cid:6s} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        out.append(r)
        c = r["ceiling"]
        nd = sum(1 for p in r["poses"] if p["degenerate"])
        print(f"  {cid:6s} {cls:10s} enum={c['n_enumerated']:6d}"
              f"{'!' if c['capped'] else ' '} acc={c['n_acceptable']:3d} "
              f"Lmin={c['min_L_rmsd_over_rotations']:6.2f} pool={r['pool']['n']:5d} "
              f"n={len(r['poses']):4d} deg={nd:3d} "
              f"bestI={min(p['I_rmsd'] for p in r['poses']):6.2f} "
              f"{time.perf_counter()-t0:6.0f}s", flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}: {len(out)} complexes, "
          f"{sum(len(c['poses']) for c in out)} poses, {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
