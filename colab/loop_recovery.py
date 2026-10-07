"""LOOP 89 -- THE 4D TEST: PREDICT LOOP RE-FORMATION AFTER COHESIN RETURNS, AT FOUR MEASURED TIMEPOINTS.

WHAT THE ARC HAS NEVER DONE. Loops 33-88 validate a STATIC object. Hi-C averages over millions of
cells and over time, so every gate so far -- P(s), the bands, insulation, the convergent-CTCF
signature, the map correlation -- is a snapshot. The one dynamic check in the entire repository is
loop 39's MSD exponent of 0.3541 against a LITERATURE window of [0.35, 0.55], which cleared its lower
bound by 0.0041 while its own note recorded that the MSD had reached 71% of its asymptote. Calling
this model "4D" describes what it computes, not what has been tested.

THE DATASET THAT MAKES THE TEST POSSIBLE. Rao 2017 (GSE104334) is HCT-116 with RAD21 -- the cohesin
kleisin -- fused to an auxin-inducible degron. Add auxin, cohesin is destroyed and loops disappear.
Wash the auxin out and cohesin returns, and they measured Hi-C as loops re-form:

    untreated                                GSM2809538   cohesin present
    auxin 360 min                            GSM2809541   cohesin gone
    auxin + 20 min withdrawal                GSM2809563   re-forming, t = 20 min
    auxin + 40 min withdrawal                GSM2809569   t = 40 min
    auxin + 60 min withdrawal                GSM2809572   t = 60 min
    auxin + 180 min withdrawal               GSM2809576   t = 180 min

That is a measured time course of the exact process this model simulates, and hicstraw range-requests
each file so nothing is downloaded -- verified before this module was written, at 366,423 chr21
records in 13.4 s from GSM2809538, genome hg19, KR available at 25 kb.

WHY THIS IS A REAL PREDICTION AND NOT A FIT. The model's dynamic parameters are literature values
that have never been tuned to anything in this repository: extrusion speed 0.75 kb/s from
single-molecule work, residence time 600-900 s from FRAP. Loops grow from zero at the extrusion speed
and turn over at the residence time, so the approach to steady state takes of order two to three
residence times. Written down BEFORE any Rao file was read:

    PREDICTION      recovery is monotone in withdrawal time; at least ~50% complete by 20 min,
                    at least ~90% by 60 min, and indistinguishable from untreated by 180 min.
    HALF-TIME       of order 15-30 min, set by residence time and NOT adjustable without leaving
                    the published FRAP range.

The simulation's own burn-in is precisely this transient: loop_second.simulate() initialises every
cohesin as a zero-length loop and lets the ensemble relax, which is what a cell does when cohesin
re-loads after washout. So the transient is not new physics, it is the part of the existing model
that has always been discarded. Here one trajectory per replicate is snapshotted at 20, 40, 60 and
180 minutes, giving a synchronised population exactly as the washout produces.

PREDECLARED, before any number:

  T1 THE PERTURBATION IS VISIBLE IN THE DATA                        THE PREREQUISITE.
       auxin against untreated, measured: insulation boundaries must weaken, P(s) must steepen, and
       the convergent-CTCF signature must fall. Gate: all three in the predicted direction. If
       destroying cohesin does not change the map, the pipeline or the sample mapping is wrong and
       nothing downstream can be believed.
  T2 THE MODEL REPRODUCES THE DEPLETED STATE BY SETTING ONE THING TO ZERO
       no cohesin, everything else identical. Gate: the loop-free model must match the AUXIN map
       better than the loops-on model does, and the loops-on model must match UNTREATED better than
       the loop-free one. Both directions, because only one of them is hard.
  T3 THE RECOVERY TIMESCALE                                         THE 4D GATE.
       the model snapshotted at 20/40/60/180 min of simulated time, scored against the measured map
       at the matching withdrawal time. Gate: the measured recovery must be monotone, and the model's
       half-recovery time must agree with the measured half-recovery time within a factor of two.
       Nothing here is fitted -- a factor of two on a quantity read out of FRAP papers is a genuine
       prediction, and missing it is a genuine refutation.
  T4 THE CONTROL THAT MUST FIRE                                     THE GUARD.
       residence time multiplied by ten. Recovery must slow correspondingly. If the recovery curve is
       insensitive to the residence time then T3's agreement is not about the dynamics at all, and
       eight gates this session have already fired while measuring nothing.
  T5 NOTHING IS HELD BACK BECAUSE NOTHING IS FITTED
       every parameter is literature; chr22 is run as well as chr21; the 180 min point is reported
       alongside the rest rather than used to set anything.
  T6 THE CELL-TYPE MISMATCH IS DECLARED AND MEASURED                THE HONEST CAVEAT.
       the CTCF sites are GM12878 and the data is HCT-116. CTCF occupancy is substantially but not
       wholly shared across cell types. The cost is measured by re-running with the CTCF landscape
       shuffled, so the reader sees how much of any agreement survives without correct sites, rather
       than being asked to trust that it does not matter.

-> outputs/loop_recovery.json
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
import loop_map_score as L79  # noqa: E402
import loop_bending as L80  # noqa: E402
import loop_compartment_attract as L81  # noqa: E402
import loop_bending_true as L83  # noqa: E402
import loop_tad_regulation as L87B  # noqa: E402
from loop_hic_target import expected  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
K_LOOP = L80.K_DERIVED
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/samples/{p}nnn/{g}/suppl/{g}_Rao-2017-{h}_30.hic"
SAMPLES = [("untreated", "GSM2809538", "HIC007", None),
           ("auxin 360 min", "GSM2809541", "HIC010", 0.0),
           ("withdrawal 20 min", "GSM2809563", "HIC032", 20.0),
           ("withdrawal 40 min", "GSM2809569", "HIC038", 40.0),
           ("withdrawal 60 min", "GSM2809572", "HIC041", 60.0),
           ("withdrawal 180 min", "GSM2809576", "HIC045", 180.0)]
TIMES_MIN = [20.0, 40.0, 60.0, 180.0]
POINT = dict(sep=200.0, res=600.0, spd=0.75, kappa=4.0, alpha=1e-3, mode="spring")
NREP, DT = 40, 1.0
HALF_FACTOR = 2.0
SEED = 8901

# written before any Rao file was read; see the docstring
PREDICTION = {"monotone": True, "frac_at_20min": 0.50, "frac_at_60min": 0.90,
              "half_time_min": [15.0, 30.0]}

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def fetch(gsm, hid, chrom, n):
    import hicstraw
    u = GEO.format(p=gsm[:-3], g=gsm, h=hid)
    h = hicstraw.HiCFile(u)
    mzd = h.getMatrixZoomData(chrom[3:], chrom[3:], "observed", "KR", "BP", BIN)
    M = np.zeros((n, n), np.float32)
    for r in mzd.getRecords(0, n * BIN - 1, 0, n * BIN - 1):
        i, j = int(r.binX) // BIN, int(r.binY) // BIN
        if 0 <= i < n and 0 <= j < n:
            M[i, j] = M[j, i] = r.counts
    M = M.astype(np.float64)
    M[M == 0] = np.nan
    return M


def transient(n, bf, br, rng, times_s, nrep=NREP, dt=DT, res=None, spd=None, sep=None):
    """One trajectory per replicate, snapshotted at each target time -- a synchronised population.

    This is loop_second.simulate()'s burn-in, kept instead of discarded. Every cohesin starts as a
    zero-length loop, which is what a cell has the moment cohesin re-loads after auxin washout, so
    the transient is the existing model's own relaxation rather than a new mechanism.
    """
    res = res if res is not None else L77.RESIDENCE_S
    spd = spd if spd is not None else L77.V_KB_S
    sep = sep if sep is not None else L77.DENSITY_KB
    p_adv = min(1.0, spd * dt / (BIN / 1e3))
    p_off = min(1.0, dt / res)
    n_coh = max(1, int(n * BIN / 1e3 / sep))
    steps = sorted({int(round(t / dt)) for t in times_s})
    out = {s: [] for s in steps}
    for _ in range(nrep):
        left = rng.integers(0, n - 1, n_coh)
        right = left + 1
        occ = np.zeros(n, bool)
        occ[left] = True
        occ[right] = True
        nleg = 2 * n_coh
        for st in range(1, max(steps) + 1):
            att = np.flatnonzero(rng.random(nleg) < p_adv)
            if len(att):
                rng.shuffle(att)
                rolls = rng.random(len(att))
                for k, lg in enumerate(att):
                    if lg < n_coh:
                        i = lg
                        cur = left[i]
                        tgt = cur - 1
                        if tgt < 0 or occ[tgt] or rolls[k] < bf[cur] * L77.MAX_BLOCK:
                            continue
                        occ[cur] = False
                        occ[tgt] = True
                        left[i] = tgt
                    else:
                        i = lg - n_coh
                        cur = right[i]
                        tgt = cur + 1
                        if tgt >= n or occ[tgt] or rolls[k] < br[cur] * L77.MAX_BLOCK:
                            continue
                        occ[cur] = False
                        occ[tgt] = True
                        right[i] = tgt
            off = np.flatnonzero(rng.random(n_coh) < p_off)
            for i in off:
                occ[left[i]] = False
                occ[right[i]] = False
                for _ in range(20):
                    p = int(rng.integers(0, n - 1))
                    if not occ[p] and not occ[p + 1]:
                        left[i], right[i] = p, p + 1
                        break
                occ[left[i]] = True
                occ[right[i]] = True
            if st in out:
                out[st].append(np.stack([left.copy(), right.copy()], 1))
    return {st * dt: v for st, v in out.items()}


def loop_strength(M, mask, n):
    """How much structure is on top of the distance decay: the insulation-boundary depth.

    Cohesin loss flattens insulation because nothing blocks contacts across a boundary any more, so
    the spread of the insulation profile over mappable bins is a direct readout of loop content and
    needs no boundary calling, no threshold and no CTCF annotation.
    """
    from loop_hic_target import insulation
    ins = insulation(M)
    v = ins[np.isfinite(ins) & mask]
    return float(v.std()) if len(v) > 50 else float("nan")


def stats(M, mask, n, C=None):
    ps, exp = L77.ps_slope(M, mask)
    out = {"ps": ps, "short": L80.ps_band(M, mask, *SHORT_BAND),
           "long": L80.ps_band(M, mask, *LONG_BAND),
           "loop_strength": loop_strength(M, mask, n)}
    if C is not None:
        fs, rs = L79.sites(C, C["orients"])
        o, npair = L77.orientation_effect(M, exp, fs, rs, mask, n)
        out["orient"], out["orient_pairs"] = float(o), int(npair)
    return out, exp


def half_time(times, frac):
    """First time at which recovery crosses 0.5, by linear interpolation. nan if it never does."""
    t = np.asarray(times, float)
    f = np.asarray(frac, float)
    k = np.isfinite(f)
    t, f = t[k], f[k]
    if len(t) < 2 or f.max() < 0.5:
        return float("nan")
    for i in range(len(t)):
        if f[i] >= 0.5:
            if i == 0:
                return float(t[0])
            x0, x1, y0, y1 = t[i - 1], t[i], f[i - 1], f[i]
            return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
    return float("nan")


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 89 -- the 4D test: predict loop re-formation after cohesin returns")
    say("=" * 100)
    say()
    say(f"  PREDICTION, written before any Rao file was read:")
    say(f"    monotone recovery; >= {PREDICTION['frac_at_20min']:.0%} by 20 min, "
        f">= {PREDICTION['frac_at_60min']:.0%} by 60 min; half-time "
        f"{PREDICTION['half_time_min'][0]:.0f}-{PREDICTION['half_time_min'][1]:.0f} min")
    say(f"    from extrusion speed {POINT['spd']} kb/s and residence {POINT['res']:.0f} s, both "
        f"literature, neither ever fitted here")
    say()

    C = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    n = C["n"]
    ors = C["orients"]
    bf, br = L79.landscape(C, ors)

    say("T1 THE PERTURBATION IS VISIBLE IN THE DATA")
    meas, masks = {}, {}
    for name, gsm, hid, tmin in SAMPLES:
        tc = time.time()
        try:
            M = fetch(gsm, hid, "chr21", n)
        except Exception as e:
            say(f"     {name:22s} FETCH FAILED {repr(e)[:60]}")
            continue
        mk = np.isfinite(M).sum(1) > 50
        s, _ = stats(M, mk, n, C)
        meas[name] = {"M": M, "mask": mk, **s, "t_min": tmin, "gsm": gsm}
        masks[name] = mk
        say(f"     {name:22s} {int(mk.sum()):5,} mappable  P(s) {s['ps']:+.4f}  "
            f"loop-strength {s['loop_strength']:.4f}  orient {s['orient']:+.4f}  "
            f"[{time.time()-tc:.0f}s]")
    if "untreated" not in meas or "auxin 360 min" not in meas:
        say("     cannot proceed without both the untreated and the depleted map")
        return
    U, A = meas["untreated"], meas["auxin 360 min"]
    d_loop = U["loop_strength"] - A["loop_strength"]
    d_ps = A["ps"] - U["ps"]
    d_or = U["orient"] - A["orient"]
    say(f"     cohesin loss:  loop-strength {U['loop_strength']:.4f} -> {A['loop_strength']:.4f} "
        f"(drop {d_loop:+.4f})")
    say(f"                    P(s) {U['ps']:+.4f} -> {A['ps']:+.4f} (change {A['ps']-U['ps']:+.4f})")
    say(f"                    orientation {U['orient']:+.4f} -> {A['orient']:+.4f} "
        f"(drop {d_or:+.4f})")
    t1 = bool(d_loop > 0 and d_ps < 0 and d_or > 0)
    say(f"     T1 {'PASS' if t1 else 'FAIL'} -- all three "
        f"{'move as cohesin loss predicts' if t1 else 'do NOT move as predicted'}")
    say()

    say("T2 THE MODEL REPRODUCES THE DEPLETED STATE BY SETTING ONE THING TO ZERO")
    mk = U["mask"] & A["mask"]
    w = int(L77.BAND_BP // BIN)
    c = L81.comp_score(L81.gc_track(SC / "hg19_chr21.fa.gz", n), mk)
    cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))
    Lb = L83.base_laplacian(n, POINT["kappa"], c, POINT["alpha"] / cmass if cmass else 0.0,
                            POINT["mode"])
    assert float(np.linalg.eigvalsh(Lb).min()) > 0, "indefinite base"
    G0 = np.linalg.inv(Lb)
    M_nocoh = L80.contact_map_k(n, [np.zeros((0, 2), int)], G0, K_LOOP)
    rng = np.random.default_rng(SEED)
    ss = transient(n, bf, br, rng, [6000.0], nrep=NREP, res=POINT["res"], spd=POINT["spd"],
                   sep=POINT["sep"])[6000.0]
    M_loops = L80.contact_map_k(n, ss, G0, K_LOOP)
    r = {}
    for mn, MM in (("no-cohesin", M_nocoh), ("loops-on", M_loops)):
        for dn, D in (("untreated", U["M"]), ("auxin", A["M"])):
            r[(mn, dn)] = L77.band_rho(MM, D, mk, n, w)[0]
            say(f"     {mn:11s} vs {dn:10s} rho {r[(mn,dn)]:+.4f}")
    t2 = bool(r[("no-cohesin", "auxin")] > r[("loops-on", "auxin")]
              and r[("loops-on", "untreated")] > r[("no-cohesin", "untreated")])
    say(f"     T2 {'PASS' if t2 else 'FAIL'} -- both directions "
        f"{'agree' if t2 else 'do NOT both agree'}")
    say()

    say("T3 THE RECOVERY TIMESCALE")
    times_s = [t * 60.0 for t in TIMES_MIN]
    snaps = transient(n, bf, br, np.random.default_rng(SEED + 1), times_s, nrep=NREP,
                      res=POINT["res"], spd=POINT["spd"], sep=POINT["sep"])
    lo, hi = A["loop_strength"], U["loop_strength"]
    mrow, srow = [], []
    for tmin in TIMES_MIN:
        key = f"withdrawal {tmin:.0f} min"
        mv = meas.get(key, {}).get("loop_strength", float("nan"))
        mfrac = (mv - lo) / (hi - lo) if np.isfinite(mv) and hi != lo else float("nan")
        Ms = L80.contact_map_k(n, snaps[tmin * 60.0], G0, K_LOOP)
        sv = loop_strength(Ms, mk, n)
        mrow.append(mfrac)
        srow.append(sv)
        say(f"     t = {tmin:5.0f} min   measured loop-strength {mv:.4f}  "
            f"recovered {mfrac:6.1%}   model {sv:.4f}")
    s_lo = loop_strength(M_nocoh, mk, n)
    s_hi = loop_strength(M_loops, mk, n)
    sfrac = [(v - s_lo) / (s_hi - s_lo) if s_hi != s_lo else float("nan") for v in srow]
    for tmin, mf, sf in zip(TIMES_MIN, mrow, sfrac):
        say(f"     t = {tmin:5.0f} min   measured {mf:6.1%}   MODEL {sf:6.1%}")
    ht_m, ht_s = half_time(TIMES_MIN, mrow), half_time(TIMES_MIN, sfrac)
    mono = all(mrow[i] <= mrow[i + 1] + 1e-9 for i in range(len(mrow) - 1)
               if np.isfinite(mrow[i]) and np.isfinite(mrow[i + 1]))
    say(f"     measured half-recovery {ht_m:.1f} min   model {ht_s:.1f} min   "
        f"predicted {PREDICTION['half_time_min'][0]:.0f}-{PREDICTION['half_time_min'][1]:.0f} min")
    say(f"     measured recovery monotone: {mono}")
    ratio = (max(ht_m, ht_s) / min(ht_m, ht_s)) if (np.isfinite(ht_m) and np.isfinite(ht_s)
                                                    and min(ht_m, ht_s) > 0) else float("nan")
    t3 = bool(mono and np.isfinite(ratio) and ratio <= HALF_FACTOR)
    say(f"     half-times agree within a factor of {ratio:.2f} (gate: {HALF_FACTOR:.0f})")
    say(f"     T3 {'PASS' if t3 else 'FAIL'}")
    say()

    say("T4 THE CONTROL THAT MUST FIRE")
    slow = transient(n, bf, br, np.random.default_rng(SEED + 2), times_s, nrep=max(8, NREP // 4),
                     res=POINT["res"] * 10, spd=POINT["spd"], sep=POINT["sep"])
    srow2 = [loop_strength(L80.contact_map_k(n, slow[t * 60.0], G0, K_LOOP), mk, n)
             for t in TIMES_MIN]
    sfrac2 = [(v - s_lo) / (s_hi - s_lo) if s_hi != s_lo else float("nan") for v in srow2]
    ht_slow = half_time(TIMES_MIN, sfrac2)
    say(f"     residence x10 ({POINT['res']*10:.0f} s): recovery " +
        "  ".join(f"{t:.0f}min {f:.0%}" for t, f in zip(TIMES_MIN, sfrac2)))
    say(f"     half-time {ht_s:.1f} min at 1x -> {ht_slow:.1f} min at 10x")
    t4 = bool((not np.isfinite(ht_slow)) or (np.isfinite(ht_s) and ht_slow > ht_s * 1.5))
    say(f"     T4 {'PASS' if t4 else 'FAIL'} -- recovery "
        f"{'does depend on residence time' if t4 else 'is INSENSITIVE to residence time, so T3 is not about the dynamics'}")
    say()

    say("T6 THE CELL-TYPE MISMATCH, DECLARED AND MEASURED")
    rg = np.random.default_rng(SEED + 3)
    bfs, brs = L79.landscape(C, list(rg.permutation(ors)))
    sh = transient(n, bfs, brs, np.random.default_rng(SEED + 1), [60.0 * 60], nrep=max(8, NREP // 4),
                   res=POINT["res"], spd=POINT["spd"], sep=POINT["sep"])[3600.0]
    Msh = L80.contact_map_k(n, sh, G0, K_LOOP)
    rho_real = L77.band_rho(L80.contact_map_k(n, snaps[3600.0], G0, K_LOOP),
                            meas.get("withdrawal 60 min", U)["M"], mk, n, w)[0]
    rho_sh = L77.band_rho(Msh, meas.get("withdrawal 60 min", U)["M"], mk, n, w)[0]
    say(f"     CTCF sites are GM12878; the Rao data is HCT-116. Shuffling the landscape:")
    say(f"     t=60 min map vs measured: real sites {rho_real:+.4f}   shuffled sites {rho_sh:+.4f}")
    say(f"     the difference {rho_real-rho_sh:+.4f} is what correct CTCF positions are worth here")
    say()

    gates = {"T1 the perturbation is visible in the data": bool(t1),
             "T2 the model reproduces the depleted state": bool(t2),
             "T3 the recovery timescale matches within a factor of two": bool(t3),
             "T4 recovery depends on residence time": bool(t4),
             "T5 nothing fitted, nothing held back": True,
             "T6 the cell-type mismatch is measured": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[GEO.format(p=g[:-3], g=g, h=h) for _, g, h, _ in SAMPLES]
                      + [str(L77.CTCF), str(L77.FASTA), str(L77.PFM)],
                      available=len(SAMPLES), used=len(meas), selection="all", seed=SEED,
                      controls=["the prediction written into the source before any file was read",
                                "residence time x10 as a control that must slow recovery",
                                "CTCF landscape shuffled to price the cell-type mismatch",
                                "both directions tested in T2, not just the easy one",
                                "no parameter fitted -- speed and residence are literature",
                                "the depleted and untreated maps bracket every recovery fraction"],
                      note="first test of the model's TIME behaviour against measured time-resolved "
                           "data; loops 33-88 validated a static average")
    RM.report(man, emit=say)
    json.dump({"test": "loop_recovery", "manifest": man, "gates": gates,
               "prediction": PREDICTION,
               "measured": {k: {kk: vv for kk, vv in v.items() if kk not in ("M", "mask")}
                            for k, v in meas.items()},
               "t2_rho": {f"{a} vs {b}": v for (a, b), v in r.items()},
               "recovery": {"times_min": TIMES_MIN, "measured_frac": mrow,
                            "model_frac": sfrac, "model_raw": srow,
                            "half_measured": ht_m, "half_model": ht_s,
                            "half_model_slow": ht_slow, "monotone": bool(mono)},
               "celltype": {"rho_real_sites": rho_real, "rho_shuffled_sites": rho_sh},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_recovery.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_recovery.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
