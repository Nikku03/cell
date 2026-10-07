"""LOOP 85 -- THE PREPROCESSING WAS WRONG FOR SIX LOOPS. FIX IT, ATTRIBUTE IT, RE-SCORE THE ARC.

WHAT LOOP 84 TRIPPED OVER. Loop 84 fixed the orientation CONTROL and then, with the correct control
in hand, measured the convergent-CTCF signature on the real chr21 Hi-C map and got -0.0358 at z -0.68.
On the same file loop 33 measured +0.3788. A measured map cannot mislay its own signature, so the
fault was not the control and not the model: it was the code that turns the file into a map.

FOUR DEFECTS, ALL MINE, ALL IN loop_map_score.build_chrom, which loops 79-84 every one imported.
Compared line by line against loop_hic_target.py, the module that DEFINED every target this arc is
scored against:

    nan fill      loop 33 does M[M == 0] = nan and then keeps rows with more than 50 finite entries
                  (loop_hic_target.py:162-163). build_chrom skipped the fill, so nothing was ever
                  non-finite, the >0.5n test passed all 1,926 chr21 bins, and 549 unmappable bins --
                  centromere, gaps, low-coverage arms -- were carried into every expected(), every
                  P(s), every band slope and every orientation pair.
    peak position loop 33 reads the narrowPeak summit offset in column 9; build_chrom used the
                  interval midpoint. RETRACTED BY P2 BELOW: this one changes nothing at all --
                  not a bin, not an orientation, not a digit of the signature. It is left in the
                  code because matching the reference is still right, and left in this docstring
                  because P2 was written to catch me over-claiming and it did.
    PWM           loop 33 row-normalises the count matrix before taking log-odds
                  (loop_hic_target.py:209); build_chrom fed raw counts to the log, so the score
                  tracked column read depth rather than base preference.
    motif scan    loop 33 scans EVERY offset in +/-150 bp; build_chrom stepped 5 bp over +/-100 bp,
                  and a 19 bp motif on a 5 bp grid is usually read out of register.

The files themselves are innocent: the hashes in loop 84's manifest match loop 33's, byte for byte.

THIS IS THE FIFTH TIME IN THIS SESSION a gate has fired while measuring nothing, and the first time
the flaw was upstream of the gate rather than inside it. The other four -- loop 76's null shuffling
signs within a single-sign arm, loop 77's V2 comparing a map against itself, loop 81's C3 differencing
two collapsed bands, loop 82's D3 returning one bin for every input -- were all self-comparisons. This
one is a silent disagreement with the reference implementation, which no amount of internal
consistency could have caught. Only reading loop 33's source next to mine did.

PREDECLARED, before any number:

  P1 THE CORRECTED CODE REPRODUCES LOOP 33 WITH NO FITTING                THE GATE.
       six independently recorded quantities, from outputs/loop_hic_target.json and loop 33's log:
       1,377 mappable bins of 1,926; P(s) -0.96363; 359 of 404 peaks oriented; 1,923 convergent
       against 3,970 non-convergent pairs; 1.353 vs 0.974; difference +0.3788. Gate: all six, to the
       precision each was recorded at. This is the strongest available check because loop 33 is a
       different implementation written for a different purpose, and nothing here is tuned.
  P2 EACH DEFECT IS ATTRIBUTED SEPARATELY, NOT LUMPED                     HONESTY CHECK.
       four variants, each re-introducing exactly ONE defect into the corrected code, plus the
       all-four variant which is what loops 79-84 actually ran. Reported, not gated. If a defect I
       named turns out to move nothing, my diagnosis over-claimed and the record must say which.
  P3 THE MEASURED MAP CARRIES ITS OWN SIGNATURE UNDER CONTROL (B)         LOOP 84's O6, RETESTED.
       loop 84 got -0.0358 at z -0.68 on measured chr21 and -0.0454 at z -1.1 on chr22, and called
       its own control into question. Gate: with preprocessing corrected, both measured maps must
       give a POSITIVE signature at z >= 4 with the scoring-label null below 25% of it. Passing this
       validates loop 84's control on data where the answer is known independently; failing it means
       the control is broken after all and loop 84's retraction of loops 77-83 is itself wrong.
  P4 EVERY TARGET THE ARC WAS CALIBRATED AGAINST, RECOMPUTED              THE DAMAGE REPORT.
       measured P(s), the short and long band slopes loops 81-83 aimed at, the distance-only null and
       the replicate ceiling that every rho in loops 79-83 was scored between. Defective and
       corrected side by side. Reported, not gated -- there is no pass here, only the true numbers.
  P5 THE ARC'S HEADLINE CLAIM, RE-SCORED                                  THE GATE THAT MATTERS.
       loops 82 and 83's best points, re-run and re-scored against the CORRECTED measured map and the
       CORRECTED distance null. Gate: the best point must still beat the distance-only null. Loops
       79-83 spent five loops establishing that it does; if that survives correction the mechanisms
       stand and only the numbers move, and if it does not then the arc's central result falls and
       this file is where that gets recorded.
  P6 CHROMOSOME 22, WHICH LOOP 33 NEVER TOUCHED                           HELD OUT.
       the same corrected pipeline on chr22: mappable count, oriented peaks, and its measured
       orientation signature with the scoring-label control. Loop 33 only ever ran chr21, so nothing
       about chr22 can be a reproduction -- it is a genuine prediction of the corrected code.

-> outputs/loop_preprocess.json
"""
import gzip
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
import loop_second as L77  # noqa: E402
import loop_map_score as L79  # noqa: E402
import loop_bending as L80  # noqa: E402
import loop_compartment_attract as L81  # noqa: E402
import loop_persistence as L82  # noqa: E402
import loop_bending_true as L83  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = L77.SC
BIN = L77.BIN
K_LOOP = L80.K_DERIVED
SHORT_BAND, LONG_BAND = (1e5, 5e5), (1e6, 1e7)
NPERM = 20
Z_MIN = 4.0
COLLAPSE_MAX = 0.25
SEED = 8501

# loop 33, outputs/loop_hic_target.json and its log -- the reproduction target for P1
LOOP33 = {"n_bins": 1926, "n_mappable": 1377, "ps": -0.9636271223546161,
          "n_peaks": 404, "n_oriented": 359, "n_conv": 1923, "n_other": 3970,
          "conv_mean": 1.353, "other_mean": 0.974, "diff": 0.37879493573816586}

# what loops 79-83 recorded under the defective preprocessing, for the P4 damage report
OLD = {"ps": -0.9636, "short": -0.8666, "long": -0.9721, "dist_null": 0.8280, "ceiling": 0.9441}

# the arc's best points, re-scored in P5
BEST = [("loop 82 best  (spring)", dict(sep=200.0, res=600.0, spd=0.75, kappa=4.0,
                                        alpha=1e-3, mode="spring")),
        ("loop 83 best  (bend)", dict(sep=200.0, res=600.0, spd=0.75, kappa=0.0,
                                      alpha=3e-4, mode="bend"))]

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def build_variant(chrom, fasta, nan_fill=True, use_summit=True, pwm_norm=True, full_scan=True):
    """build_chrom with each of the four defects independently switchable, for P2.

    All four flags True is loop_map_score.build_chrom as corrected, and is asserted identical to it
    in P2 rather than assumed. All four False is what loops 79-84 ran.
    """
    H = np.load(SC / f"hic_{chrom}_25kb.npy").astype(np.float64)
    n = len(H)
    if nan_fill:
        H[H == 0] = np.nan
        mask = np.isfinite(H).sum(1) > 50
    else:
        mask = np.isfinite(H).sum(1) > 0.5 * n
    seq = "".join(l.strip() for l in gzip.open(SC / fasta, "rt") if not l.startswith(">")).upper()
    pk = []
    for ln in gzip.open(L77.CTCF, "rt"):
        f = ln.split("\t")
        if f[0] != chrom:
            continue
        st, en = int(f[1]), int(f[2])
        if use_summit:
            off = (int(f[9]) if len(f) > 9 and f[9].strip().lstrip("-").isdigit() and int(f[9]) >= 0
                   else (en - st) // 2)
        else:
            off = (en - st) // 2
        pk.append({"summit": st + off})
    pfm = json.load(open(L77.PFM))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    if pwm_norm:
        W = W / W.sum(1, keepdims=True)
        W = np.log2((W + 1e-3) / 0.25)
    else:
        W = np.log2(W / 0.25 + 1e-9)
    idx = {c: i for i, c in enumerate("ACGT")}

    def sc(s):
        if len(s) != Lw or any(c not in idx for c in s):
            return -1e9
        return float(sum(W[i, idx[c]] for i, c in enumerate(s)))

    def rc(s):
        return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for p in pk:
        best, bo = -1e9, 0
        if full_scan:
            a, b = max(0, p["summit"] - 150), min(len(seq), p["summit"] + 150)
            win = seq[a:b]
            cand = (win[i:i + Lw] for i in range(len(win) - Lw + 1))
        else:
            c = p["summit"]
            cand = (seq[c + o: c + o + Lw] for o in range(-100, 101, 5))
        for s in cand:
            f_, r_ = sc(s), sc(rc(s))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0
    return {"H": H, "n": n, "mask": mask, "peaks": pk,
            "orients": [p["orient"] for p in pk]}


def d4_counts(M, exp, fs, rs, mask, n):
    """loop 33's D4 estimator verbatim (loop_hic_target.py:291-325), returning its intermediates."""
    conv, other = [], []
    for i in sorted(fs | rs):
        for j in sorted(fs | rs):
            if j - i < 4 or (j - i) * BIN > 2e6:
                continue
            if not (mask[i] and mask[j]) or not np.isfinite(M[i, j]) or not np.isfinite(exp[j - i]):
                continue
            (conv if (i in fs and j in rs) else other).append((j - i, M[i, j] / exp[j - i]))
    byd = defaultdict(list)
    for dd, v in other:
        byd[dd].append(v)
    mc, mo = [], []
    for dd, v in conv:
        if byd.get(dd):
            mc.append(v)
            mo.append(float(np.mean(byd[dd])))
    if len(mc) < 30:
        return {"n_conv": len(conv), "n_other": len(other), "n_matched": len(mc),
                "conv_mean": float("nan"), "other_mean": float("nan"), "diff": float("nan")}
    mc, mo = np.array(mc), np.array(mo)
    return {"n_conv": len(conv), "n_other": len(other), "n_matched": len(mc),
            "conv_mean": float(mc.mean()), "other_mean": float(mo.mean()),
            "diff": float(np.mean(mc - mo))}


def scoring_null(M, exp, ors, C, mask, n, nperm=NPERM, seed=SEED):
    """Loop 84's control (B): one map, permute only the labels used to score it."""
    fs, rs = L79.sites(C, ors)
    real, npair = L77.orientation_effect(M, exp, fs, rs, mask, n)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(nperm):
        f2, r2 = L79.sites(C, list(rng.permutation(ors)))
        v, _ = L77.orientation_effect(M, exp, f2, r2, mask, n)
        if np.isfinite(v):
            null.append(v)
    null = np.array(null) if null else np.array([np.nan])
    sd = null.std()
    z = (real - null.mean()) / sd if sd > 1e-12 else float("inf")
    return {"real": float(real), "n_pairs": int(npair), "null_mean": float(null.mean()),
            "null_sd": float(sd), "z": float(z),
            "p_emp": (int((null >= real).sum()) + 1) / (len(null) + 1),
            "frac": float(null.mean() / real) if np.isfinite(real) and abs(real) > 1e-12
            else float("nan")}


def build_G0(n, c, cmass, kappa, alpha, mode):
    L = L83.base_laplacian(n, kappa, c, alpha / cmass if cmass else 0.0, mode)
    lam = float(np.linalg.eigvalsh(L).min())
    assert lam > 0, f"indefinite base: kappa={kappa} alpha={alpha} mode={mode} lam={lam}"
    return np.linalg.inv(L)


def main():
    t0 = time.time()
    say("=" * 100)
    say("  LOOP 85 -- the preprocessing was wrong for six loops. Fix it, attribute it, re-score.")
    say("=" * 100)
    say()

    C = L79.build_chrom("chr21", "hg19_chr21.fa.gz")
    n, mask, H = C["n"], C["mask"], C["H"]
    ors = C["orients"]
    w = int(L77.BAND_BP // BIN)
    ps, exp = L77.ps_slope(H, mask)
    fs, rs = L79.sites(C, ors)
    d4 = d4_counts(H, exp, fs, rs, mask, n)

    say("P1 THE CORRECTED CODE REPRODUCES LOOP 33 WITH NO FITTING")
    checks = [("bins", n, LOOP33["n_bins"], 0),
              ("mappable bins", int(mask.sum()), LOOP33["n_mappable"], 0),
              ("P(s) slope", ps, LOOP33["ps"], 1e-4),
              ("peaks", len(C["peaks"]), LOOP33["n_peaks"], 0),
              ("oriented peaks", sum(1 for o in ors if o), LOOP33["n_oriented"], 0),
              ("convergent pairs", d4["n_conv"], LOOP33["n_conv"], 0),
              ("non-convergent pairs", d4["n_other"], LOOP33["n_other"], 0),
              ("convergent mean", d4["conv_mean"], LOOP33["conv_mean"], 5e-4),
              ("non-conv mean", d4["other_mean"], LOOP33["other_mean"], 5e-4),
              ("orientation difference", d4["diff"], LOOP33["diff"], 1e-4)]
    ok = []
    for name, got, want, tol in checks:
        hit = abs(got - want) <= tol
        ok.append(hit)
        gs = f"{got:.5f}" if isinstance(got, float) else f"{got:,}"
        ws = f"{want:.5f}" if isinstance(want, float) else f"{want:,}"
        say(f"     {name:24s} corrected {gs:>12s}   loop 33 {ws:>12s}   "
            f"{'match' if hit else 'DIFFERS'}")
    p1 = all(ok)
    say(f"     P1 {'PASS' if p1 else 'FAIL'} -- {sum(ok)}/{len(ok)} independently recorded "
        f"quantities reproduced")
    say()

    say("P2 EACH DEFECT ATTRIBUTED SEPARATELY")
    ref = build_variant("chr21", "hg19_chr21.fa.gz")
    assert int(ref["mask"].sum()) == int(mask.sum()) and ref["orients"] == ors, \
        "build_variant(all True) must equal the corrected build_chrom"
    say(f"     build_variant(all corrections on) == build_chrom: asserted, not assumed")
    variants = [("corrected", {}),
                ("no nan fill", dict(nan_fill=False)),
                ("midpoint not summit", dict(use_summit=False)),
                ("unnormalised PWM", dict(pwm_norm=False)),
                ("5 bp grid, +/-100", dict(full_scan=False)),
                ("all four (loops 79-84)", dict(nan_fill=False, use_summit=False,
                                                pwm_norm=False, full_scan=False))]
    attrib = []
    for name, kw in variants:
        V = ref if not kw else build_variant("chr21", "hg19_chr21.fa.gz", **kw)
        vm, vo = V["mask"], V["orients"]
        vps, vexp = L77.ps_slope(V["H"], vm)
        vfs, vrs = L79.sites(V, vo)
        vd = d4_counts(V["H"], vexp, vfs, vrs, vm, V["n"])
        row = {"variant": name, "mappable": int(vm.sum()),
               "oriented": sum(1 for o in vo if o), "ps": vps, "diff": vd["diff"],
               "n_matched": vd["n_matched"]}
        attrib.append(row)
        say(f"     {name:24s} mappable {row['mappable']:5,}   oriented {row['oriented']:4d}   "
            f"P(s) {vps:+.4f}   signature {vd['diff']:+.4f}")
    base = attrib[0]
    inert = [r["variant"] for r in attrib[1:5]
             if r["mappable"] == base["mappable"] and r["oriented"] == base["oriented"]
             and abs(r["diff"] - base["diff"]) < 1e-6]
    say(f"     defects that changed nothing on their own: "
        f"{', '.join(inert) if inert else 'none -- all four are real'}")
    say(f"     P2 reported (not gated){'; DIAGNOSIS OVER-CLAIMED on ' + ', '.join(inert) if inert else ''}")
    say()

    say("P3 THE MEASURED MAP CARRIES ITS OWN SIGNATURE UNDER CONTROL (B)")
    S21 = scoring_null(H, exp, ors, C, mask, n)
    say(f"     chr21 measured  real {S21['real']:+.4f}   null {S21['null_mean']:+.4f} "
        f"+/- {S21['null_sd']:.4f}   z {S21['z']:+.1f}   survives {S21['frac']:.0%}")
    say(f"     loop 84 got {-0.0358:+.4f} at z {-0.68:+.2f} here, with the same control and the "
        f"same file")
    C22 = L79.build_chrom("chr22", "hg19_chr22.fa.gz")
    n22, m22, H22, o22 = C22["n"], C22["mask"], C22["H"], C22["orients"]
    ps22, exp22 = L77.ps_slope(H22, m22)
    S22 = scoring_null(H22, exp22, o22, C22, m22, n22)
    say(f"     chr22 measured  real {S22['real']:+.4f}   null {S22['null_mean']:+.4f} "
        f"+/- {S22['null_sd']:.4f}   z {S22['z']:+.1f}   survives {S22['frac']:.0%}")
    p3 = all(s["real"] > 0 and s["z"] >= Z_MIN and s["frac"] < COLLAPSE_MAX for s in (S21, S22))
    say(f"     P3 {'PASS' if p3 else 'FAIL'} -- loop 84's control "
        f"{'is validated on data' if p3 else 'does NOT reproduce a known effect, so it is broken'}")
    say()

    say("P4 EVERY TARGET THE ARC WAS CALIBRATED AGAINST, RECOMPUTED")
    D = build_variant("chr21", "hg19_chr21.fa.gz", nan_fill=False, use_summit=False,
                      pwm_norm=False, full_scan=False)
    dm = D["mask"]
    dps = L77.ps_slope(D["H"], dm)[0]
    new = {"ps": ps, "short": L80.ps_band(H, mask, *SHORT_BAND),
           "long": L80.ps_band(H, mask, *LONG_BAND),
           "dist_null": L77.band_rho(L79.distance_null(C), H, mask, n, w)[0]}
    old = {"ps": dps, "short": L80.ps_band(D["H"], dm, *SHORT_BAND),
           "long": L80.ps_band(D["H"], dm, *LONG_BAND),
           "dist_null": L77.band_rho(L79.distance_null(D), D["H"], dm, D["n"], w)[0]}
    c = L81.comp_score(L81.gc_track(L77.FASTA, n), mask)
    cmass = max(float(np.maximum(c, 0).sum()), float(np.maximum(-c, 0).sum()))
    P0 = BEST[1][1]
    G0 = build_G0(n, c, cmass, P0["kappa"], P0["alpha"], P0["mode"])
    bf, br = L79.landscape(C, ors)
    RA = L82.run_point(C, bf, br, P0["sep"], P0["res"], P0["spd"], G0, 1.0, 50, SEED)
    RB = L82.run_point(C, bf, br, P0["sep"], P0["res"], P0["spd"], G0, 1.0, 50, SEED + 1)
    new["ceiling"] = L77.band_rho(RA["M"], RB["M"], mask, n, w)[0]
    old["ceiling"] = L77.band_rho(RA["M"], RB["M"], dm, n, w)[0]
    for k, label in [("ps", "measured P(s)"), ("short", "short band 0.1-0.5 Mb"),
                     ("long", "long band 1-10 Mb"), ("dist_null", "distance-only null rho"),
                     ("ceiling", "replicate ceiling rho")]:
        say(f"     {label:24s} loops 79-83 used {OLD[k]:+.4f}   recomputed defective {old[k]:+.4f}"
            f"   CORRECTED {new[k]:+.4f}")
    say(f"     the band targets loops 81-83 were tuned toward move by "
        f"{abs(new['short']-old['short']):.4f} (short) and {abs(new['long']-old['long']):.4f} (long)")
    say(f"     P4 reported (not gated)")
    say()

    say("P5 THE ARC'S HEADLINE CLAIM, RE-SCORED ON THE CORRECTED MAP")
    say(f"     corrected distance-only null {new['dist_null']:+.4f}   "
        f"replicate ceiling {new['ceiling']:+.4f}")
    rescored = []
    for name, p in BEST:
        g = G0 if p is P0 else build_G0(n, c, cmass, p["kappa"], p["alpha"], p["mode"])
        R = RA if p is P0 else L82.run_point(C, bf, br, p["sep"], p["res"], p["spd"], g,
                                             1.0, 50, SEED)
        rho = L77.band_rho(R["M"], H, mask, n, w)[0]
        rho_old = L77.band_rho(R["M"], D["H"], dm, n, w)[0]
        head = ((rho - new["dist_null"]) / (new["ceiling"] - new["dist_null"])
                if new["ceiling"] > new["dist_null"] else float("nan"))
        sb = L80.ps_band(R["M"], mask, *SHORT_BAND)
        lb = L80.ps_band(R["M"], mask, *LONG_BAND)
        S = scoring_null(R["M"], R["exp"], ors, C, mask, n, nperm=10)
        rescored.append({"point": name, "rho_corrected": rho, "rho_defective": rho_old,
                         "headroom": head, "short": sb, "long": lb,
                         "orient": S["real"], "orient_z": S["z"]})
        say(f"     {name:22s} rho {rho:+.4f} (defective map gave {rho_old:+.4f})   "
            f"headroom {head:+.0%}")
        say(f"     {'':22s} bands {sb:+.4f} / {lb:+.4f} vs target {new['short']:+.4f} / "
            f"{new['long']:+.4f}   orientation {S['real']:+.4f} z {S['z']:+.1f}")
    p5 = all(r["rho_corrected"] > new["dist_null"] for r in rescored)
    say(f"     P5 {'PASS' if p5 else 'FAIL'} -- the best points "
        f"{'still beat' if p5 else 'DO NOT beat'} the distance-only null after correction")
    say()

    say("P6 CHROMOSOME 22, WHICH LOOP 33 NEVER TOUCHED")
    d22 = d4_counts(H22, exp22, *L79.sites(C22, o22), m22, n22)
    say(f"     {n22:,} bins, {int(m22.sum()):,} mappable, {len(C22['peaks']):,} CTCF peaks, "
        f"{sum(1 for o in o22 if o):,} oriented")
    say(f"     P(s) {ps22:+.4f}   {d22['n_conv']:,} convergent vs {d22['n_other']:,} "
        f"non-convergent")
    say(f"     convergent {d22['conv_mean']:.3f} vs non-convergent {d22['other_mean']:.3f}, "
        f"difference {d22['diff']:+.4f}")
    say(f"     loop 33 never ran chr22, so this is a prediction of the corrected code, not a "
        f"reproduction")
    say(f"     P6 reported (not gated)")
    say()

    gates = {"P1 the corrected code reproduces loop 33": bool(p1),
             "P2 each defect attributed separately": True,
             "P3 the measured map carries its signature under control (B)": bool(p3),
             "P4 targets recomputed": True,
             "P5 the arc's headline claim survives correction": bool(p5),
             "P6 chr22 held out": True}
    for k, v in gates.items():
        say(f"  {'PASS' if v else 'FAIL'}  {k}")

    man = RM.manifest(inputs=[str(L77.HIC), str(SC / "hic_chr22_25kb.npy"), str(L77.CTCF),
                              str(L77.FASTA), str(SC / "hg19_chr22.fa.gz"), str(L77.PFM)],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=["ten quantities reproduced against a different implementation",
                                "each defect switched on alone and attributed separately",
                                "build_variant(all on) asserted identical to build_chrom",
                                "the corrected control applied to the MEASURED map as validation",
                                "defective and corrected targets reported side by side",
                                "chr22 held out -- loop 33 never ran it"],
                      note="loop_map_score.build_chrom disagreed with loop_hic_target.py in four "
                           "places; loops 79-84 ran the defective version and their measured-map "
                           "comparisons are superseded by this file")
    RM.report(man, emit=say)
    json.dump({"test": "loop_preprocess", "manifest": man, "gates": gates,
               "loop33_target": LOOP33,
               "reproduction": {name: got for name, got, _, _ in checks},
               "attribution": attrib,
               "measured_chr21": S21, "measured_chr22": S22,
               "targets": {"loops_79_83_used": OLD, "recomputed_defective": old,
                           "corrected": new},
               "rescored": rescored,
               "chr22": {"n": n22, "mappable": int(m22.sum()), "peaks": len(C22["peaks"]),
                         "oriented": sum(1 for o in o22 if o), "ps": ps22, **d22},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_preprocess.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_preprocess.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
