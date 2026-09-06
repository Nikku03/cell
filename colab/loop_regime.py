"""LOOP 78 -- IS THE CONTACT-DECAY WINDOW REACHABLE AT ALL, OR IS IT A MECHANISM PROBLEM?

WHAT LOOP 77 LEFT. With the parallel-update bug fixed -- legs now step one at a time in random order
against live occupancy -- the extrusion model converges cleanly across timesteps (P(s) spread 0.0164
over dt = 33.3 s down to 1 s, inside the same-seed replicate scatter of 0.9692). And it converges to
P(s) = -1.21, OUTSIDE the (-1.16, -0.76) window that loops 33/35 fixed before any model existed,
against a measured -0.9636. Before the fix the coarse clock gave -1.0732, comfortably inside. So the
old agreement was manufactured by the bug.

That leaves one question, and it is the only one worth asking next: is -1.21 a PARAMETER problem or a
MECHANISM problem? Loop 77 ran a single parameter point -- the literature defaults loop 35 chose. If
some other literature-admissible point reaches the window, the mechanism is fine and the defaults
were wrong. If no admissible point reaches it, the mechanism is incomplete and no amount of tuning
will save it.

THE PHYSICAL SUSPICION, stated before the sweep so it can be wrong. The regime is set by the ratio of
PROCESSIVITY (how far a cohesin extrudes before falling off, v * residence) to SEPARATION (how far
apart cohesins are, the density). At loop 35's defaults that is 0.75 kb/s * 900 s = 675 kb against a
150 kb separation, a ratio of 4.5 -- deep in the dense regime where cohesins collide long before
reaching their processivity limit, so loop size is set by traffic rather than by biology. Fudenberg
2016 fitted processivity and separation of the same order as each other, a ratio near 1-2. If the
ratio is the controlling variable, points near 1 should behave differently from points near 5, and
the sweep will show it as structure rather than as scatter.

THE CIRCULARITY, AND THE GUARD AGAINST IT. Sweeping parameters until P(s) lands in a window IS
fitting, and this project has spent seventy-odd loops refusing to do that quietly. Two things keep it
honest:

  - The grid is LITERATURE-ADMISSIBLE ONLY, declared below with its sources, and fixed before the
    run. A point outside those ranges does not count even if it fits perfectly.
  - P(s) is what is being fitted, so P(s) cannot also be the evidence. The HELD-OUT observable is the
    convergent-CTCF orientation effect, which is a different measurement on a different feature of
    the map, and which loop 33 measured independently at +0.3788. A point selected on P(s) and then
    passing the orientation control has earned something; a point that hits the window and fails the
    control has hit it for the wrong reason, and S3 exists to say so.

Also reported without gating: how MANY grid points reach the window. If most of them do, P(s) barely
constrains the model and hitting it is not evidence of anything.

PREDECLARED, before any number:

  S1 THE GRID IS LITERATURE-ADMISSIBLE AND FIXED HERE
       separation 100-400 kb, residence 600-1500 s (10-25 min, FRAP), speed 0.5-1.0 kb/s
       (single-molecule and in-vivo inference). 45 points. Loop 35's defaults (150 kb, 900 s,
       0.75 kb/s) are one point in the grid, not the centre of a fitted search. Reported, not judged.
  S2 CAN ANY ADMISSIBLE POINT REACH THE WINDOW                       THE GATE.
       at least one grid point with P(s) inside (-1.16, -0.76). If none does, the corrected mechanism
       cannot reproduce the measured contact decay anywhere in its literature-admissible parameter
       space, and that is a mechanism result, not a tuning result.
  S3 THE BEST POINT SURVIVES THE HELD-OUT CONTROL           THE GUARD. THE REAL TEST.
       take the admissible point closest to the measured -0.9636, re-run it at dt = 1 s with the full
       configuration count, and require the convergent-CTCF signature to be positive AND to collapse
       to below half its value when motif orientations are shuffled. P(s) was fitted; this was not.
  S4 IT BEATS THE DISTANCE-ONLY NULL BY MORE THAN LOOP 77's POINT DID
       simulated-vs-measured Spearman at the best point, against the distance-only null (0.8280) and
       the replicate ceiling (0.9441). Loop 77's default point reached 0.8424, which is 12.4% of the
       available headroom. Gate: the best point must do better than that.
  S5 THE REGIME IS DIAGNOSED, NOT JUST THE PARAMETERS
       processivity/separation ratio, bin occupancy and mean loop length at every grid point, so the
       result is reported as physics rather than as a lookup table. If P(s) tracks the ratio rather
       than the three parameters separately, that is the finding and it should be visible.

-> outputs/loop_regime.json
"""
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
import loop_second as L77  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
BIN = L77.BIN
PS_WINDOW = L77.PS_WINDOW
MEASURED_PS = -0.9636
L77_RHO = 0.8424          # loop 77's default point vs measured
DIST_NULL = 0.8280        # distance-only null vs measured
CEILING = 0.9441          # replicate vs replicate

# ---- S1 LITERATURE-ADMISSIBLE GRID, fixed here before the run --------------------------------
SEPARATION_KB = [100.0, 150.0, 200.0, 300.0, 400.0]   # cohesin spacing
RESIDENCE_S = [600.0, 900.0, 1500.0]                  # 10 / 15 / 25 min, FRAP
SPEED_KB_S = [0.5, 0.75, 1.0]                         # single-molecule and in-vivo inference

DT_SWEEP = 3.0        # loop 77 showed 3 s and 1 s agree inside replicate noise
DT_FINAL = 1.0
NCFG_SWEEP = 20
NCFG_FINAL = 50
SEED = 7801

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def run_point(n, bf, br, G0, mask, sep, res, spd, dt, ncfg, seed):
    """One grid point. Patches loop 77's module constants, which is why they are restored after."""
    old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
    L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = sep, res, spd
    try:
        cfgs, ncoh, _ = L77.simulate(n, bf, br, np.random.default_rng(seed), dt, n_config=ncfg)
        M = L77.contact_map_fast(n, cfgs, G0)
        ps, exp = L77.ps_slope(M, mask)
    finally:
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    loops = [b - a for cfg in cfgs for a, b in cfg if b > a]
    return {"M": M, "exp": exp, "ps": ps, "n_coh": ncoh, "cfgs": cfgs,
            "mean_loop_kb": float(np.mean(loops) * BIN / 1e3) if loops else 0.0,
            "occupancy": float(2 * ncoh / n),
            "proc_kb": spd * res, "ratio": spd * res / sep}


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 78 -- is the contact-decay window reachable at all, or is it a mechanism problem?")
    say("=" * 100)
    say()

    H = np.load(L77.HIC)
    n = len(H)
    mask = np.isfinite(H).sum(1) > 0.5 * n
    say(f"  {L77.CHROM} at {BIN//1000} kb: {n:,} bins, {int(mask.sum()):,} mappable")
    say(f"  measured P(s) {MEASURED_PS:+.4f};  window {PS_WINDOW};  loop 77's corrected default "
        f"-1.2134")
    say()

    # rebuild the oriented CTCF landscape (same construction as loops 33/35/77)
    import gzip
    seq = "".join(l.strip() for l in gzip.open(L77.FASTA, "rt") if not l.startswith(">")).upper()
    pk = []
    for ln in gzip.open(L77.CTCF, "rt"):
        f = ln.split("\t")
        if f[0] == L77.CHROM:
            pk.append({"summit": (int(f[1]) + int(f[2])) // 2})
    pfm = json.load(open(L77.PFM))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    idx = {c: i for i, c in enumerate("ACGT")}

    def sc(s):
        if len(s) != Lw or any(c not in idx for c in s):
            return -1e9
        return float(sum(np.log2(W[i, idx[c]] / 0.25 + 1e-9) for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for p in pk:
        best, bo = -1e9, 0
        c = p["summit"]
        for off in range(-100, 101, 5):
            s = seq[c + off: c + off + Lw]
            f_, r_ = sc(s), sc(rc(s))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0
    ors = [p["orient"] for p in pk]

    def landscape(o_list):
        bfm, brm = np.zeros(n), np.zeros(n)
        for p, o in zip(pk, o_list):
            b = p["summit"] // BIN
            if 0 <= b < n:
                if o > 0:
                    bfm[b] = 1.0
                elif o < 0:
                    brm[b] = 1.0
        return bfm, brm

    bf, br = landscape(ors)
    fs = {p["summit"] // BIN for p, o in zip(pk, ors) if o > 0}
    rs = {p["summit"] // BIN for p, o in zip(pk, ors) if o < 0}
    say(f"  {len(pk):,} CTCF peaks, {sum(1 for o in ors if o):,} oriented")
    G0, _ = L77.base_inverse(n)
    say()

    say("S1 THE LITERATURE-ADMISSIBLE GRID")
    say(f"     separation {SEPARATION_KB} kb")
    say(f"     residence  {RESIDENCE_S} s")
    say(f"     speed      {SPEED_KB_S} kb/s")
    grid = [(a, b, c) for a in SEPARATION_KB for b in RESIDENCE_S for c in SPEED_KB_S]
    say(f"     {len(grid)} points, swept at dt = {DT_SWEEP} s with {NCFG_SWEEP} configurations")
    say(f"     S1 PASS (declared)")
    say()

    say("S2 CAN ANY ADMISSIBLE POINT REACH THE WINDOW")
    rows = []
    for k, (sep, res, spd) in enumerate(grid, 1):
        r = run_point(n, bf, br, G0, mask, sep, res, spd, DT_SWEEP, NCFG_SWEEP, SEED)
        inw = PS_WINDOW[0] <= r["ps"] <= PS_WINDOW[1]
        rows.append({"sep_kb": sep, "res_s": res, "v_kb_s": spd, "ps": r["ps"],
                     "ratio": r["ratio"], "occupancy": r["occupancy"],
                     "mean_loop_kb": r["mean_loop_kb"], "n_coh": r["n_coh"], "in_window": inw})
        if k % 5 == 0 or inw:
            say(f"       sep {sep:5.0f} kb  res {res:6.0f} s  v {spd:4.2f}  ->  P(s) {r['ps']:+.4f}  "
                f"ratio {r['ratio']:5.2f}  occ {r['occupancy']:4.1%}  "
                f"loop {r['mean_loop_kb']:6.1f} kb {'  <- IN WINDOW' if inw else ''}")
    nin = sum(1 for x in rows if x["in_window"])
    say(f"     {nin} of {len(rows)} admissible points land inside {PS_WINDOW}")
    s2 = nin > 0
    say(f"     S2 {'PASS' if s2 else 'FAIL'} -- the corrected mechanism "
        f"{'CAN' if s2 else 'CANNOT'} reach the measured contact decay within literature ranges")
    if nin > len(rows) * 0.5:
        say(f"     NOTE: more than half the grid lands in the window, so P(s) barely constrains "
            f"this model and hitting it is weak evidence.")
    say()

    say("S3 THE BEST POINT SURVIVES THE HELD-OUT ORIENTATION CONTROL")
    cand = [x for x in rows if x["in_window"]] or rows
    best = min(cand, key=lambda x: abs(x["ps"] - MEASURED_PS))
    say(f"     closest admissible point to the measured {MEASURED_PS:+.4f}: "
        f"sep {best['sep_kb']:.0f} kb, res {best['res_s']:.0f} s, v {best['v_kb_s']:.2f} kb/s  "
        f"(P(s) {best['ps']:+.4f} at dt={DT_SWEEP} s)")
    R = run_point(n, bf, br, G0, mask, best["sep_kb"], best["res_s"], best["v_kb_s"],
                  DT_FINAL, NCFG_FINAL, SEED)
    say(f"     re-run at dt = {DT_FINAL} s, {NCFG_FINAL} configurations: P(s) {R['ps']:+.4f}")
    o_real, n_pairs = L77.orientation_effect(R["M"], R["exp"], fs, rs, mask, n)
    rng = np.random.default_rng(SEED)
    sh = list(rng.permutation(ors))
    bfs, brs = landscape(sh)
    Rs = run_point(n, bfs, brs, G0, mask, best["sep_kb"], best["res_s"], best["v_kb_s"],
                   DT_FINAL, NCFG_FINAL, SEED)
    fss = {p["summit"] // BIN for p, o in zip(pk, sh) if o > 0}
    rss = {p["summit"] // BIN for p, o in zip(pk, sh) if o < 0}
    o_shuf, _ = L77.orientation_effect(Rs["M"], Rs["exp"], fss, rss, mask, n)
    say(f"     real motif orientation   {o_real:+.4f}   ({n_pairs} matched pairs)")
    say(f"     SHUFFLED orientation     {o_shuf:+.4f}   <- must fall below half")
    say(f"     measured (loop 33)       +0.3788")
    s3 = np.isfinite(o_real) and o_real > 0 and (not np.isfinite(o_shuf) or o_shuf < 0.5 * o_real)
    say(f"     S3 {'PASS' if s3 else 'FAIL'} -- P(s) was fitted, this was not")
    say()

    say("S4 IT BEATS THE DISTANCE-ONLY NULL BY MORE THAN LOOP 77's POINT DID")
    w = int(L77.BAND_BP // BIN)
    rho_best, nb = L77.band_rho(R["M"], H, mask, n, w)
    head_77 = (L77_RHO - DIST_NULL) / (CEILING - DIST_NULL)
    head_now = (rho_best - DIST_NULL) / (CEILING - DIST_NULL)
    say(f"     best point vs measured    Spearman {rho_best:+.4f}  (n={nb:,})")
    say(f"     loop 77 default point     Spearman {L77_RHO:+.4f}")
    say(f"     distance-only null        Spearman {DIST_NULL:+.4f}   replicate ceiling {CEILING:+.4f}")
    say(f"     headroom captured: loop 77 {head_77:.1%}  ->  here {head_now:.1%}")
    s4 = rho_best > L77_RHO
    say(f"     S4 {'PASS' if s4 else 'FAIL'}")
    say()

    say("S5 THE REGIME, NOT JUST THE PARAMETERS")
    from scipy.stats import spearmanr
    ps_all = np.array([x["ps"] for x in rows])
    for nm, v in (("processivity/separation ratio", [x["ratio"] for x in rows]),
                  ("separation (kb)", [x["sep_kb"] for x in rows]),
                  ("residence (s)", [x["res_s"] for x in rows]),
                  ("speed (kb/s)", [x["v_kb_s"] for x in rows]),
                  ("bin occupancy", [x["occupancy"] for x in rows]),
                  ("mean loop (kb)", [x["mean_loop_kb"] for x in rows])):
        say(f"       rho(P(s), {nm:30s}) = {spearmanr(ps_all, v).statistic:+.4f}")
    say(f"     if the ratio dominates the three parameters separately, the model is in a regime, "
        f"not at a point")
    say(f"     S5 PASS (reported)")
    say()

    gates = {"S1 grid is literature-admissible and fixed": True,
             "S2 an admissible point reaches the window": bool(s2),
             "S3 best point survives the held-out orientation control": bool(s3),
             "S4 beats the distance null by more than loop 77": bool(s4),
             "S5 regime diagnosed": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L77.HIC), str(L77.CTCF), str(L77.FASTA), str(L77.PFM)],
                      available=len(grid), used=len(rows), selection="all", seed=SEED,
                      controls=["grid restricted to literature ranges declared before the run",
                                "P(s) is the fitted quantity so it is not the evidence",
                                "convergent-CTCF orientation held out as the selection-free test",
                                "orientation shuffling required to collapse the signature",
                                "count of grid points in the window reported, so a loose "
                                "constraint cannot masquerade as a fit",
                                "distance-only null and replicate ceiling both reported"],
                      note="loop 77's corrected model converged to P(s) -1.21 outside the window; "
                           "this asks whether that is a parameter or a mechanism failure")
    RM.report(man, emit=say)
    json.dump({"test": "loop_regime", "manifest": man, "gates": gates,
               "measured_ps": MEASURED_PS, "window": list(PS_WINDOW),
               "grid": rows, "n_in_window": nin, "n_points": len(rows),
               "best": best, "best_ps_dt1": R["ps"],
               "orientation_real": o_real, "orientation_shuffled": o_shuf,
               "rho_best": rho_best, "rho_loop77": L77_RHO, "dist_null": DIST_NULL,
               "ceiling": CEILING, "headroom_loop77": head_77, "headroom_best": head_now,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_regime.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_regime.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
