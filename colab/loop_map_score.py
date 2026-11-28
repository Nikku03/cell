"""LOOP 79 -- SCORE ON THE MAP, NOT ON P(s), AND TEST THE ANSWER ON A CHROMOSOME IT NEVER SAW.

WHY. Loop 78 swept 45 literature-admissible parameter points, selected the one whose P(s) best
matched the measured -0.9636, and found that this point correlates with the actual contact map at
Spearman 0.7564 -- WORSE than a distance-only null (0.8280) and worse than loop 77's untuned default
(0.8424). Optimising the one-dimensional contact-decay summary actively damaged agreement with the
two-dimensional object it summarises.

That is a verdict on the ACCEPTANCE CRITERION. P(s) is a projection: it averages the map over every
pair at a given separation and throws away where the contacts are. Loops 33 and 35 adopted it as the
gate, and loop 78 showed it can be satisfied by configurations that place contacts worse than knowing
nothing but genomic distance. So this loop scores on the map itself.

WHICH CREATES A SHARPER CIRCULARITY, AND A BETTER ANSWER TO IT. If the map is the selection target,
the map cannot also be the evidence. Loop 78 held out the orientation effect; here THREE things are
held out, and the third is the one that matters:

    P(s)             now a held-out observable rather than the target -- the roles swap
    orientation      the convergent-CTCF signature, never used in any selection
    CHROMOSOME 22    parameters chosen on chr21 are applied unchanged to chr22 and scored against
                     chr22's measured map

chr22 is the real test. It is a different chromosome, 2,053 bins against chr21's 1,926, with its own
823 CTCF peaks against chr21's 404 and its own contact map. Nothing about it enters the selection. A
parameter set that is genuinely capturing extrusion physics should transfer; one that has been tuned
to chr21's particular contact pattern should not.

PREDECLARED, before any number:

  W1 THE SAME GRID, RESCORED ON THE MAP                              REPORTED, NOT JUDGED.
       the identical 45-point literature-admissible grid from loop 78 -- separation 100-400 kb,
       residence 600-1500 s, speed 0.5-1.0 kb/s -- swept on chr21 and ranked by Spearman against the
       measured map over the 2 Mb band, instead of by P(s) proximity.
  W2 THE BEST-BY-MAP POINT BEATS THE UNTUNED DEFAULT                 THE GATE.
       must exceed loop 77's default point (0.8424), which is itself only 12.4% of the headroom
       between the distance-only null (0.8280) and the replicate ceiling (0.9441). Beating the
       distance null is necessary and nowhere near sufficient.
  W3 HELD OUT: WHERE DOES P(s) LAND                                  ROLES SWAPPED.
       P(s) at the best-by-map point, against the (-1.16, -0.76) window. Not gated -- it is now an
       observable, not a target -- but reported, because if selecting on the map ALSO lands P(s) in
       the window then the two criteria agree after all and loop 78's tension was a selection
       artifact. If it does not, the two criteria are genuinely incompatible and that is the result.
  W4 HELD OUT: THE ORIENTATION CONTROL                               THE SAME GUARD AS LOOP 78.
       convergent-CTCF signature must be positive and collapse below half when motif orientations
       are shuffled. Loop 78's best-by-P(s) point managed +0.1966 -> +0.0690.
  W5 HELD OUT: CHROMOSOME 22                                         THE REAL TEST.
       chr21-selected parameters applied unchanged to chr22 and scored against chr22's measured map,
       versus chr22's OWN distance-only null computed from its own expected-by-separation curve.
       Gate: must beat that null. This is the only gate here that no amount of tuning on chr21 can
       reach for free.
  W6 THE TWO CRITERIA ARE COMPARED HONESTLY
       best-by-map and best-by-P(s) reported side by side on both scores, plus how many grid points
       satisfy BOTH. If the intersection is empty, the arc has been calibrated against a criterion
       that cannot be reconciled with the data it was meant to reproduce, and that needs saying
       plainly rather than being split.

CORRECTION, ADDED AT LOOP 85. build_chrom below has been fixed. As shipped it differed from
loop_hic_target.py -- loop 33, which defined every measured target this arc is scored against -- in
four places, three of which mattered: it never NaN-filled the zero entries before masking, so all
1,926 chr21 bins passed where loop 33 kept 1,377; it fed an unnormalised count matrix to the
log-odds, so all 404 peaks cleared the motif threshold instead of 359; and it scanned the motif on a
5 bp grid over +/-100 bp instead of every position over +/-150 bp, finding only 101 in register. The
fourth -- midpoints instead of narrowPeak summits -- was measured in loop 85's P2 and changes
nothing; it is fixed for agreement, not because it was a cause. Together the three real defects turn
loop 33's measured convergent-CTCF signature of +0.3788 into -0.0358 and its P(s) of -0.9636 into
-1.0388. outputs/loop_map_score.json, and the same fields in loops 80-84, were produced with the
defective version and their measured-map comparisons are superseded. The corrected code reproduces
all ten of loop 33's recorded quantities exactly; see loop_preprocess.py.

-> outputs/loop_map_score.json
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
import loop_second as L77  # noqa: E402
from loop_hic_target import expected  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
PS_WINDOW = L77.PS_WINDOW
MEASURED_PS = -0.9636
L77_RHO = 0.8424
DIST_NULL = 0.8280
CEILING = 0.9441

SEPARATION_KB = [100.0, 150.0, 200.0, 300.0, 400.0]
RESIDENCE_S = [600.0, 900.0, 1500.0]
SPEED_KB_S = [0.5, 0.75, 1.0]
DT_SWEEP, DT_FINAL = 3.0, 1.0
NCFG_SWEEP, NCFG_FINAL = 20, 50
SEED = 7901

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_chrom(chrom, fasta):
    """Hi-C map, mask, oriented CTCF landscape and forward/reverse site sets for one chromosome.

    CORRECTED AT LOOP 85. The version this function shipped with, and which loops 79-84 all ran,
    differed from loop_hic_target.py (loop 33, which defined every target this arc is scored
    against) in four places, and every one of them degraded the comparison:

        unmappable bins   loop 33 sets M[M == 0] = nan then keeps rows with >50 finite entries.
                          This function skipped the nan fill, so nothing was ever non-finite and
                          the >0.5n test passed all 1,926 chr21 bins where loop 33 kept 1,377.
        peak position     loop 33 reads the narrowPeak summit offset (column 9); this function
                          used the interval midpoint. Loop 85's P2 measured this one and it does
                          NOTHING -- identical bins, identical orientations, identical signature.
                          Changed anyway so the two implementations agree, but it was not a cause.
        PWM               loop 33 row-normalises the count matrix before the log-odds; this
                          function fed raw counts in, so the score was column-depth, not affinity.
        motif scan        loop 33 scans EVERY position in +/-150 bp; this function stepped 5 bp
                          over +/-100 bp, which usually misses the true register of a 19 bp motif.

    With all four corrected the function reproduces loop 33's recorded numbers exactly: 1,377
    mappable chr21 bins, P(s) -0.9636, 359/404 peaks oriented, 1,923 convergent against 3,970
    non-convergent pairs, 1.353 vs 0.974, difference +0.3788. See loop_preprocess.py.
    """
    H = np.load(SC / f"hic_{chrom}_25kb.npy").astype(np.float64)
    n = len(H)
    H[H == 0] = np.nan                              # loop_hic_target.py:162
    mask = np.isfinite(H).sum(1) > 50               # loop_hic_target.py:163
    seq = "".join(l.strip() for l in gzip.open(SC / fasta, "rt") if not l.startswith(">")).upper()
    pk = []
    for ln in gzip.open(L77.CTCF, "rt"):
        f = ln.split("\t")
        if f[0] != chrom:
            continue
        st, en = int(f[1]), int(f[2])               # narrowPeak summit is column 9, off start
        off = (int(f[9]) if len(f) > 9 and f[9].strip().lstrip("-").isdigit() and int(f[9]) >= 0
               else (en - st) // 2)
        pk.append({"start": st, "end": en, "summit": st + off})
    pfm = json.load(open(L77.PFM))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    W = W / W.sum(1, keepdims=True)                 # loop_hic_target.py:209 -- was missing
    W = np.log2((W + 1e-3) / 0.25)
    idx = {c: i for i, c in enumerate("ACGT")}

    def sc(s):
        if len(s) != Lw or any(c not in idx for c in s):
            return -1e9
        return float(sum(W[i, idx[c]] for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for p in pk:
        a, b = max(0, p["summit"] - 150), min(len(seq), p["summit"] + 150)
        win = seq[a:b]
        best, bo = -1e9, 0
        for i in range(len(win) - Lw + 1):          # every position, not a 5 bp grid
            s = win[i:i + Lw]
            f_, r_ = sc(s), sc(rc(s))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["motif"], p["orient"] = best, (bo if best > 6.0 else 0)
    ors = [p["orient"] for p in pk]
    return {"H": H, "n": n, "mask": mask, "peaks": pk, "orients": ors,
            "G0": L77.base_inverse(n)[0]}


def landscape(C, o_list):
    n = C["n"]
    bf, br = np.zeros(n), np.zeros(n)
    for p, o in zip(C["peaks"], o_list):
        b = p["summit"] // BIN
        if 0 <= b < n:
            if o > 0:
                bf[b] = 1.0
            elif o < 0:
                br[b] = 1.0
    return bf, br


def sites(C, o_list):
    fs = {p["summit"] // BIN for p, o in zip(C["peaks"], o_list) if o > 0}
    rs = {p["summit"] // BIN for p, o in zip(C["peaks"], o_list) if o < 0}
    return fs, rs


def run_point(C, bf, br, sep, res, spd, dt, ncfg, seed):
    old = (L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S)
    L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = sep, res, spd
    try:
        cfgs, ncoh, _ = L77.simulate(C["n"], bf, br, np.random.default_rng(seed), dt, n_config=ncfg)
        M = L77.contact_map_fast(C["n"], cfgs, C["G0"])
        ps, exp = L77.ps_slope(M, C["mask"])
    finally:
        L77.DENSITY_KB, L77.RESIDENCE_S, L77.V_KB_S = old
    loops = [b - a for cfg in cfgs for a, b in cfg if b > a]
    return {"M": M, "exp": exp, "ps": ps, "n_coh": ncoh,
            "mean_loop_kb": float(np.mean(loops) * BIN / 1e3) if loops else 0.0}


def distance_null(C):
    """The map a model that knows only genomic separation would produce."""
    n, H, mask = C["n"], C["H"], C["mask"]
    e = expected(H, mask)
    D = np.zeros((n, n))
    ii, jj = np.triu_indices(n, 1)
    D[ii, jj] = np.where(np.isfinite(e[jj - ii]), e[jj - ii], 0.0)
    return D + D.T


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 79 -- score on the map, not on P(s), and test on a chromosome it never saw")
    say("=" * 100)
    say()

    C21 = build_chrom("chr21", "hg19_chr21.fa.gz")
    w = int(L77.BAND_BP // BIN)
    bf, br = landscape(C21, C21["orients"])
    fs, rs = sites(C21, C21["orients"])
    say(f"  chr21: {C21['n']:,} bins, {int(C21['mask'].sum()):,} mappable, "
        f"{len(C21['peaks']):,} CTCF peaks ({sum(1 for o in C21['orients'] if o):,} oriented)")
    d21 = distance_null(C21)
    rho_dist21 = L77.band_rho(d21, C21["H"], C21["mask"], C21["n"], w)[0]
    say(f"  chr21 distance-only null vs measured: {rho_dist21:+.4f}   "
        f"(loop 78 recorded {DIST_NULL:+.4f})")
    say()

    say("W1 THE SAME GRID, RESCORED ON THE MAP")
    grid = [(a, b, c) for a in SEPARATION_KB for b in RESIDENCE_S for c in SPEED_KB_S]
    rows = []
    for k, (sep, res, spd) in enumerate(grid, 1):
        R = run_point(C21, bf, br, sep, res, spd, DT_SWEEP, NCFG_SWEEP, SEED)
        rho = L77.band_rho(R["M"], C21["H"], C21["mask"], C21["n"], w)[0]
        rows.append({"sep_kb": sep, "res_s": res, "v_kb_s": spd, "ps": R["ps"], "rho_map": rho,
                     "mean_loop_kb": R["mean_loop_kb"], "n_coh": R["n_coh"],
                     "ps_in_window": PS_WINDOW[0] <= R["ps"] <= PS_WINDOW[1],
                     "beats_dist": rho > rho_dist21})
        if k % 9 == 0:
            say(f"       {k:2d}/45  sep {sep:5.0f} res {res:6.0f} v {spd:4.2f}  "
                f"rho_map {rho:+.4f}  P(s) {R['ps']:+.4f}")
    rows.sort(key=lambda x: -x["rho_map"])
    say(f"     top 5 by map correlation:")
    for x in rows[:5]:
        say(f"       sep {x['sep_kb']:5.0f} kb  res {x['res_s']:6.0f} s  v {x['v_kb_s']:4.2f}  "
            f"rho_map {x['rho_map']:+.4f}   P(s) {x['ps']:+.4f}"
            f"{'  [P(s) in window]' if x['ps_in_window'] else ''}")
    say(f"     W1 PASS (reported)")
    say()

    say("W2 THE BEST-BY-MAP POINT BEATS THE UNTUNED DEFAULT")
    best = rows[0]
    B = run_point(C21, bf, br, best["sep_kb"], best["res_s"], best["v_kb_s"],
                  DT_FINAL, NCFG_FINAL, SEED)
    rho_best = L77.band_rho(B["M"], C21["H"], C21["mask"], C21["n"], w)[0]
    head = (rho_best - rho_dist21) / (CEILING - rho_dist21)
    say(f"     best by map: sep {best['sep_kb']:.0f} kb, res {best['res_s']:.0f} s, "
        f"v {best['v_kb_s']:.2f} kb/s")
    say(f"     re-run at dt={DT_FINAL} s, {NCFG_FINAL} configs: rho_map {rho_best:+.4f}")
    say(f"     loop 77 untuned default {L77_RHO:+.4f}   loop 78 best-by-P(s) 0.7564")
    say(f"     distance null {rho_dist21:+.4f}   replicate ceiling {CEILING:+.4f}   "
        f"headroom captured {head:.1%}")
    w2 = rho_best > L77_RHO
    say(f"     W2 {'PASS' if w2 else 'FAIL'}")
    say()

    say("W3 HELD OUT: WHERE DOES P(s) LAND -- ROLES SWAPPED")
    inw = PS_WINDOW[0] <= B["ps"] <= PS_WINDOW[1]
    say(f"     P(s) at the best-by-map point {B['ps']:+.4f}   window {PS_WINDOW}   "
        f"measured {MEASURED_PS:+.4f}")
    say(f"     {'INSIDE' if inw else 'OUTSIDE'} the window -- the two criteria "
        f"{'agree after all' if inw else 'are genuinely incompatible'}")
    say(f"     W3 PASS (reported, not gated -- P(s) is an observable here, not a target)")
    say()

    say("W4 HELD OUT: THE ORIENTATION CONTROL")
    o_real, npair = L77.orientation_effect(B["M"], B["exp"], fs, rs, C21["mask"], C21["n"])
    rng = np.random.default_rng(SEED)
    sh = list(rng.permutation(C21["orients"]))
    bfs, brs = landscape(C21, sh)
    fss, rss = sites(C21, sh)
    Bs = run_point(C21, bfs, brs, best["sep_kb"], best["res_s"], best["v_kb_s"],
                   DT_FINAL, NCFG_FINAL, SEED)
    o_shuf, _ = L77.orientation_effect(Bs["M"], Bs["exp"], fss, rss, C21["mask"], C21["n"])
    say(f"     real motif orientation   {o_real:+.4f}  ({npair} matched pairs)")
    say(f"     SHUFFLED orientation     {o_shuf:+.4f}  <- must fall below half")
    say(f"     loop 78 best-by-P(s) managed +0.1966 -> +0.0690;  measured (loop 33) +0.3788")
    w4 = np.isfinite(o_real) and o_real > 0 and (not np.isfinite(o_shuf) or o_shuf < 0.5 * o_real)
    say(f"     W4 {'PASS' if w4 else 'FAIL'}")
    say()

    say("W5 HELD OUT: CHROMOSOME 22 -- PARAMETERS APPLIED UNCHANGED")
    C22 = build_chrom("chr22", "hg19_chr22.fa.gz")
    w22 = int(L77.BAND_BP // BIN)
    bf22, br22 = landscape(C22, C22["orients"])
    fs22, rs22 = sites(C22, C22["orients"])
    d22 = distance_null(C22)
    rho_dist22 = L77.band_rho(d22, C22["H"], C22["mask"], C22["n"], w22)[0]
    say(f"     chr22: {C22['n']:,} bins, {int(C22['mask'].sum()):,} mappable, "
        f"{len(C22['peaks']):,} CTCF peaks ({sum(1 for o in C22['orients'] if o):,} oriented)")
    T = run_point(C22, bf22, br22, best["sep_kb"], best["res_s"], best["v_kb_s"],
                  DT_FINAL, NCFG_FINAL, SEED)
    rho22 = L77.band_rho(T["M"], C22["H"], C22["mask"], C22["n"], w22)[0]
    o22, np22 = L77.orientation_effect(T["M"], T["exp"], fs22, rs22, C22["mask"], C22["n"])
    say(f"     model on chr22 vs measured chr22   {rho22:+.4f}")
    say(f"     chr22 distance-only null           {rho_dist22:+.4f}")
    say(f"     P(s) on chr22 {T['ps']:+.4f};  orientation effect {o22:+.4f} ({np22} pairs)")
    w5 = rho22 > rho_dist22
    say(f"     W5 {'PASS' if w5 else 'FAIL'} -- the parameters "
        f"{'TRANSFER to a chromosome they never saw' if w5 else 'do NOT transfer'}")
    say()

    say("W6 THE TWO CRITERIA COMPARED HONESTLY")
    both = [x for x in rows if x["ps_in_window"] and x["beats_dist"]]
    bymap = rows[0]
    byps = min(rows, key=lambda x: abs(x["ps"] - MEASURED_PS))
    say(f"     best by MAP  : sep {bymap['sep_kb']:5.0f} res {bymap['res_s']:6.0f} "
        f"v {bymap['v_kb_s']:4.2f}   rho {bymap['rho_map']:+.4f}  P(s) {bymap['ps']:+.4f}")
    say(f"     best by P(s) : sep {byps['sep_kb']:5.0f} res {byps['res_s']:6.0f} "
        f"v {byps['v_kb_s']:4.2f}   rho {byps['rho_map']:+.4f}  P(s) {byps['ps']:+.4f}")
    say(f"     grid points satisfying BOTH (P(s) in window AND beating the distance null): "
        f"{len(both)} of {len(rows)}")
    if not both:
        say(f"     THE INTERSECTION IS EMPTY. No literature-admissible point reproduces the contact")
        say(f"     decay AND places contacts better than genomic distance alone. The arc's")
        say(f"     acceptance criterion and its data cannot both be satisfied by this mechanism.")
    say(f"     W6 PASS (reported)")
    say()

    gates = {"W1 grid rescored on the map": True,
             "W2 best-by-map beats the untuned default": bool(w2),
             "W3 P(s) reported as a held-out observable": True,
             "W4 orientation control survives": bool(w4),
             "W5 parameters transfer to chr22": bool(w5),
             "W6 the two criteria compared": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy"), str(SC / "hic_chr22_25kb.npy"),
                              str(L77.CTCF), str(SC / "hg19_chr21.fa.gz"),
                              str(SC / "hg19_chr22.fa.gz"), str(L77.PFM)],
                      available=len(grid), used=len(rows), selection="all", seed=SEED,
                      controls=["selection target is the 2D map, so the map is not the evidence",
                                "P(s) held out -- roles swapped from loop 78",
                                "convergent-CTCF orientation held out and shuffle-controlled",
                                "chr22 held out entirely: different bins, different CTCF, "
                                "different map, parameters applied unchanged",
                                "chr22 scored against its OWN distance-only null",
                                "intersection of the two criteria counted and reported"],
                      note="loop 78's best-by-P(s) point scored 0.7564 against the map, below the "
                           "0.8280 distance null; this asks what happens when the map is the target")
    RM.report(man, emit=say)
    json.dump({"test": "loop_map_score", "manifest": man, "gates": gates,
               "grid": rows, "best_by_map": bymap, "best_by_ps": byps,
               "rho_best_chr21": rho_best, "ps_best_chr21": B["ps"], "ps_in_window": bool(inw),
               "rho_dist_chr21": rho_dist21, "headroom": head,
               "orientation_real": o_real, "orientation_shuffled": o_shuf,
               "chr22": {"n_bins": C22["n"], "rho_model": rho22, "rho_dist_null": rho_dist22,
                         "ps": T["ps"], "orientation": o22},
               "n_satisfying_both": len(both),
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_map_score.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_map_score.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
