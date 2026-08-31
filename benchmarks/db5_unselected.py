"""Is interface conformational entropy a usable ranking signal, tested OFF the collider?

WHY THIS RUN EXISTS. The previous study computed exact side-chain partition functions on
20-pose shortlists and reported two things: that basin breadth T*S_conf tracks nativeness at
Spearman -0.45, and that reranking by free energy achieves 0/58 against a ceiling that was
ALSO 0/58 (ledger defect N -- a bar above the achievable ceiling measures nothing).

Both of those results were computed on poses selected BY THE GRID SCORE, and that selection
invalidates them in two different ways:

  THE CEILING. A shortlist chosen by a scorer, whose ceiling is then used to characterise the
  SEARCH, is circular. "No acceptable pose in the top 20" is a fact about the scorer that
  built the top 20. The honest question -- did the search ever generate an acceptable pose at
  all? -- has to be asked of the unselected candidate set.

  THE CORRELATION. Selecting on grid score and then correlating two other quantities within
  the selected set is CONDITIONING ON A COLLIDER. Grid score depends on shape
  complementarity; so, plausibly, do both T*S_conf and I_rmsd. Conditioning on a common
  effect of two variables induces an association between them even when none exists in the
  population. The -0.45 is therefore not yet evidence about protein interfaces; it is
  evidence about protein interfaces THAT SCORE WELL ON A SHAPE TERM. This run is the
  load-bearing test, and it can withdraw the earlier number.

The search generates 6000 candidates per complex (2000 rotations x 3 translations) and
run_arm already evaluates EVERY one against native. Nothing here re-runs the docking
differently; it only stops throwing 5980 of them away before looking.

=================================================================================================
THE GATES, FIXED HERE BEFORE ANY NUMBER IS RUN. Analysis lives in db5_unselected_analysis.py.
=================================================================================================

STEP 0  THE UNSELECTED CEILING. No sampling, no score. For each complex scan all 6000
        candidates. CAPRI acceptable requires f_nat >= 0.1 AND (L_rmsd <= 10 OR I_rmsd <= 4),
        so a candidate can only qualify if it first passes an RMSD screen; the screen used
        here is I_rmsd <= 6 OR L_rmsd <= 15, looser than the real bar, and exact CAPRI
        metrics are then computed on every survivor.
        WHY THAT SCREEN IS SAFE, by construction rather than by hope. run_arm's cheap
        per-candidate RMSDs are not an approximation of capri_metrics -- they are the SAME
        ARITHMETIC. L_rmsd is sqrt(mean(||x - x_native||^2)) over backbone atoms in both;
        I_rmsd is superimposed_rmsd over the stacked receptor-plus-ligand interface atoms in
        both, with the receptor half identical in the two arrangements; and the pose
        transform is (x - centre) @ R.T + centre + t in both, run_arm writing it inline and
        capri_metrics receiving it from apply_pose, which is that same expression. The only
        thing capri_metrics adds is f_nat, which needs the full docked coordinates. So the
        screen's 2 A and 5 A slack over the CAPRI bars is pure margin.
        IT IS STILL VALIDATED, NOT ASSUMED (ledger defect M -- a bound must be checked
        against the data it bounds, because an argument that two code paths agree is not a
        measurement that they do). Exact capri_metrics is computed on a random subset of
        candidates per complex and compared against the cheap values; the maximum
        discrepancy is recorded, and if it exceeds the margin (2 A on I_rmsd, 5 A on L_rmsd)
        the screen is not safe and the run reports THAT instead of a ceiling.
        CEILING := complexes with >= 1 acceptable candidate anywhere in the 6000.
        IF THE CEILING IS 0/58, that is the result, and it is a positive finding rather than
        a failure: the search never generates an acceptable pose, so no ranking of its output
        can succeed, the docking line closes for good, and every gate below is VOID -- not
        NULL, not FAILED -- because no ranking could have passed them.

STEP 1  THE STRATIFIED SAMPLE, ~500 per complex. 25 equal-width I_rmsd bins spanning the
        observed range, up to 20 candidates per bin, drawn with an RNG seeded from the
        complex id; topped up to 500 uniformly at random if the bins do not fill; and every
        acceptable candidate from step 0 force-included.
        EQUAL-WIDTH, NOT EQUAL-COUNT, ON PURPOSE: the I_rmsd distribution is heavily skewed
        toward bad poses, so quantile bins would reproduce that skew and defeat the point.
        THIS SAMPLE IS DELIBERATELY ENRICHED IN NEAR-NATIVE POSES. Every retrieval number
        below is therefore a RETRIEVAL RATE ON AN ENRICHED SET and is NOT a docking success
        rate. It must never be compared against the prior 0/58, and the enrichment factor is
        reported alongside it so the comparison cannot be made by accident.

STEP 2  Q1 RETEST -- DOES THE CORRELATION SURVIVE OFF THE COLLIDER? Compute T*S_conf for
        every sampled pose. Bars are the ORIGINAL ones, unchanged, so this is a replication
        and not a new hypothesis:
          pooled Spearman(T*S_conf, I_rmsd) <= -0.10 at p < 1e-3, AND negative per-complex
          median, AND the partial Spearman holding interface contact count fixed <= -0.10.
        If Q1 fails here having passed on the shortlist, THE ORIGINAL -0.45 WAS A SELECTION
        ARTIFACT and is withdrawn in the repo record rather than quietly left standing.

STEP 3  THE RETRIEVAL TEST. Rank each complex's sample three ways and ask how often an
        acceptable pose reaches the top 20:
          (a) energy      E_min ascending
          (b) entropy     T*S_conf descending          <- the signal on its own
          (c) 50/50       mean of the within-complex rank-normalised (a) and (b)
        (c) is the real test. The natural weight is what killed the previous study --
        T*S_conf is 1% of the energy spread, so adding it changes nothing -- so the question
        is whether the signal is USABLE when given weight, not merely PRESENT.
        THE BAR IS NOT ZERO, AND IT IS NOT A FIXED COUNT (ledger defect K -- a ranking test
        whose items differ in size). Complexes differ in how many acceptable poses their
        sample contains, so "an acceptable pose in the top 20" is trivially easier for some
        than others, and a ranking that does nothing at all would still score above zero.
        Each ranking is compared to ITS OWN random-chance expectation: for a complex with k
        acceptable poses among N sampled, a random ranking hits with probability
        p = 1 - C(N-k, 20)/C(N, 20). Summing p over complexes gives the expected hits under
        chance, and the variance of that Poisson-binomial gives the spread. A ranking counts
        as signal only if it exceeds chance by >= 2 sd. This bar is scale-free and computed
        from the data's own composition, which is what defect L keeps asking for.
        The lambda-sweep of E + lambda*(-T*S_conf) is reported as a DIAGNOSTIC ONLY. 50/50 is
        the predeclared point; the sweep exists to show the shape, not to be mined for a
        winning lambda after the fact.

STEP 4  DEGENERATE POSES. The partition function returns T*S_conf = 0 when fewer than two
        interface residues are repackable. These all tie at the bottom of the entropy
        ranking. Their count is reported, and if they exceed 10% of the sample the entropy
        ranking is re-run with them excluded to show what they were doing.

WHAT THIS RUN CANNOT DO. It holds the search fixed. It says nothing about whether a better
search would find acceptable poses, only whether THIS one did, and whether entropy can
retrieve them from what it generated.
"""
from __future__ import annotations

import argparse, glob, json, sys, time, zlib
sys.path.insert(0, ".")
import numpy as np

from rem.docking import capri, score
from rem.docking.data import Structure, load_case
from rem.docking.repack import build_from_case
from rem.docking.freeenergy import free_energy
from rem.docking.rigid import rotation_set
from benchmarks.db5_dock import run_arm, _as_struct
from benchmarks.db5_basin import subset_ids, score_pose

ROTATIONS, TOP_PER_ROT = 2000, 3
N_BINS, PER_BIN, N_SAMPLE = 25, 20, 500
SCREEN_I, SCREEN_L = 6.0, 15.0          # looser than acceptable's 4.0 / 10.0
MARGIN_I, MARGIN_L = 2.0, 5.0           # the slack the screen leaves; validated, not assumed
N_VALIDATE = 40                         # candidates per complex used to validate the screen
OK = ("high", "medium", "acceptable")


def exact_metrics(rec, lig, arm, idx):
    """Full CAPRI metrics for candidate `idx` of the unselected set."""
    p = arm["_full"](int(idx))
    return p, p["metrics"]


def validate_screen(rec, lig, arm, rng):
    """Ledger defect M: the screen is a claimed bound, so check it against the data.

    run_arm's per-candidate I_rmsd/L_rmsd are computed on interface-backbone atoms as the
    search runs. The screen assumes they agree with exact capri_metrics to within the margin
    it leaves. That assumption is measured here rather than trusted.
    """
    n = len(arm["_all"]["I_rmsd"])
    idx = rng.choice(n, size=min(N_VALIDATE, n), replace=False)
    di, dl = [], []
    for i in idx:
        _p, m = exact_metrics(rec, lig, arm, i)
        di.append(abs(m["I_rmsd"] - float(arm["_all"]["I_rmsd"][i])))
        dl.append(abs(m["L_rmsd"] - float(arm["_all"]["L_rmsd"][i])))
    return {"max_dI": float(max(di)), "max_dL": float(max(dl)),
            "safe": bool(max(di) <= MARGIN_I and max(dl) <= MARGIN_L), "n": int(len(idx))}


def ceiling_scan(rec, lig, arm):
    """Every candidate that could possibly be CAPRI-acceptable, checked exactly."""
    I = np.asarray(arm["_all"]["I_rmsd"], float)
    L = np.asarray(arm["_all"]["L_rmsd"], float)
    cand = np.where((I <= SCREEN_I) | (L <= SCREEN_L))[0]
    acc, best_q = [], None
    for i in cand:
        _p, m = exact_metrics(rec, lig, arm, i)
        if m["quality"] in OK:
            acc.append(int(i))
    return {"n_screened": int(len(cand)), "acceptable_idx": acc,
            "n_acceptable": len(acc), "n_candidates": int(len(I))}


def stratified_sample(I, acceptable_idx, rng):
    """Equal-width I_rmsd bins so near-native poses are represented if they exist at all."""
    lo, hi = float(I.min()), float(I.max())
    edges = np.linspace(lo, hi + 1e-9, N_BINS + 1)
    which = np.clip(np.digitize(I, edges) - 1, 0, N_BINS - 1)
    picked = []
    for b in range(N_BINS):
        m = np.where(which == b)[0]
        if len(m) == 0:
            continue
        take = min(PER_BIN, len(m))
        picked += [int(x) for x in rng.choice(m, size=take, replace=False)]
    picked = set(picked) | set(int(i) for i in acceptable_idx)
    if len(picked) < N_SAMPLE:                       # top up uniformly from what is left
        rest = np.setdiff1d(np.arange(len(I)), np.fromiter(picked, int, len(picked)))
        if len(rest):
            extra = rng.choice(rest, size=min(N_SAMPLE - len(picked), len(rest)),
                               replace=False)
            picked |= {int(x) for x in extra}
    return sorted(picked), {"bin_lo": lo, "bin_hi": hi}


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
    print(f"  {len(ids)} complexes, {ROTATIONS} rotations x {TOP_PER_ROT}, "
          f"sample {a.sample}/complex", flush=True)

    out, t0 = [], time.perf_counter()
    for n, (cid, cls) in enumerate(ids, 1):
        try:
            case = load_case(cid)
            rec, lig = case["r_u"], case["l_u"]
            seed = zlib.crc32(cid.encode()) & 0x7FFFFFFF
            arm = run_arm(rec, lig, rots, seed, spacing=1.5, keep=20,
                          top_per_rot=TOP_PER_ROT)
        except Exception as e:                                    # noqa: BLE001
            print(f"  {cid:6s} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        rng = np.random.default_rng(seed)
        val = validate_screen(rec, lig, arm, rng)
        ceil = ceiling_scan(rec, lig, arm)
        I = np.asarray(arm["_all"]["I_rmsd"], float)
        idx, binfo = stratified_sample(I, ceil["acceptable_idx"], rng)

        recs = []
        accset = set(ceil["acceptable_idx"])
        for i in idx:
            p, m = exact_metrics(rec, lig, arm, i)
            lg = _as_struct(lig, p["coords"])
            try:
                sc = score_pose(rec, lg, p["grid_score"])
            except Exception:                                     # noqa: BLE001
                continue
            recs.append({**sc, "idx": int(i), "I_rmsd": m["I_rmsd"], "L_rmsd": m["L_rmsd"],
                         "f_nat": m["f_nat"], "quality": m["quality"],
                         "forced": bool(i in accset)})
        if len(recs) < 20:
            print(f"  {cid:6s} too few scored ({len(recs)})", flush=True)
            continue
        out.append({"id": cid, "class": cls, "validate": val, "ceiling": {
            k: v for k, v in ceil.items() if k != "acceptable_idx"},
            "n_acceptable_full": ceil["n_acceptable"], "bins": binfo, "poses": recs})
        na = sum(r["quality"] in OK for r in recs)
        print(f"  {cid:6s} {cls:10s} cand={ceil['n_candidates']} "
              f"screen={ceil['n_screened']} accept_full={ceil['n_acceptable']} "
              f"sample={len(recs)} accept_sample={na} bestI={min(r['I_rmsd'] for r in recs):.2f} "
              f"dI={val['max_dI']:.2f} {'OK' if val['safe'] else 'SCREEN-UNSAFE'} "
              f"{time.perf_counter()-t0:6.0f}s", flush=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n  wrote {a.out}: {len(out)} complexes, "
          f"{sum(len(c['poses']) for c in out)} poses, {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
