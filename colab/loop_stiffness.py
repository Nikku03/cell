"""THE TWO UNSTATED DEFAULTS -- calibrate them from physics, then stop.

WHAT loop 35 LEFT UNJUSTIFIED.  It got the direction right and the magnitude badly wrong: the
orientation effect came out +0.0272 against a measured +0.3788, and insulation correlated at 0.19
against a gate of 0.40. Two choices in that run were never argued for, and both are calculable rather
than tunable.

    1  LOOP BOND STIFFNESS was set equal to backbone stiffness, silently.
       This is the big one and the arithmetic is immediate. In a resistor network, a loop bond of
       conductance k in parallel with the backbone path it shortcuts gives effective resistance
       s/(1+ks) for a separation of s bonds. With k = 1 and s = 50 that is 50/51 -- a TWO PER CENT
       reduction in anchor separation. A cohesin that brings its anchors 2% closer is not a cohesin.
       Cohesin is not a soft spring; it is a ring holding two loci at its own diameter, ~40 nm, while a
       25 kb bead is ~79 nm across. Requiring the anchor separation to equal the ring diameter,
           s/(1+ks) = (40/79)^2 = 0.256   at s = 50   ->   k = 3.8
       and the independent bead-geometry estimate (79/40)^2 = 3.9 agrees, which is the reason to
       believe the number rather than fit it.

    2  CTCF BLOCKING was normalised by the STRONGEST peak on the chromosome.
       Peak signal is heavy-tailed, so dividing by the maximum makes the typical IDR peak nearly
       transparent -- a site at the median blocks with probability 0.06 where the top site blocks at
       0.95. But every peak here already passed IDR thresholding; they are all real, occupied CTCF
       sites. Rank within the peak set is the honest scale, not distance from the single largest.

THE STOPPING RULE, DECLARED NOW SO THIS CANNOT BECOME TUNING.  Both values above are computed from
geometry that has nothing to do with the Hi-C map. If the gates below still fail at those values, the
conclusion recorded is that the GAUSSIAN MEAN-FIELD TREATMENT is the limiting approximation -- a
Gaussian chain has no excluded volume and its contact probability is an ensemble average of an ensemble
average -- and NOT that the parameters need another pass. No third calibration will be attempted.

PREDECLARED, before any number:

    B1 THE STIFFNESS CALIBRATION IS SELF-CONSISTENT   an isolated loop of 50 bonds with k = 3.8 must
                put its anchors at 0.256 b^2, the ring-diameter target the constant was derived from.
                An identity check on the arithmetic, so a wrong constant fails here and not silently
                three gates later.
    B2 INSULATION CLEARS THE GATE   simulated vs measured insulation Pearson >= 0.40, still beating the
                shuffled-orientation twin.
    B3 THE ORIENTATION EFFECT REACHES THE RIGHT ORDER   simulated convergent-minus-other within a
                factor of 3 of the measured +0.3788, and still collapsing under orientation shuffling.
                Both halves again: magnitude without the collapse is fitting, collapse without
                magnitude is loop 35.
    B4 STILL AFFORDABLE   under ten minutes on four CPUs.

CONTROLS: the same shuffled-orientation twin as loop 35; the stiffness identity as a positive control
on the constant itself; loop 35's own numbers as the direct before/after comparison; and the stopping
rule above, which is the control on ME.

-> outputs/loop_stiffness.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from loop_polymer import laplacian, r2_matrix  # noqa: E402
from loop_hic_target import insulation, expected  # noqa: E402
import loop_extrusion as LE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
RING_NM, BEAD_NM = 40.0, 79.0
K_LOOP = 3.8                     # derived above, agreed by two independent routes
GATE_INSUL = 0.40
GATE_ORIENT_FACTOR = 3.0
GATE_MINUTES = 10.0


def laplacian_k(n, loops, k_loop, confine):
    """Backbone at unit stiffness, loop bonds at k_loop, harmonic tether at confine."""
    L = laplacian(n, loops=(), confine=confine)
    for i, j in loops:
        if i == j:
            continue
        L[i, j] -= k_loop
        L[j, i] -= k_loop
        L[i, i] += k_loop
        L[j, j] += k_loop
    return L


def contact_map_k(n, configs, k_loop, confine):
    acc = np.zeros((n, n))
    for cfg in configs:
        L = laplacian_k(n, [(int(a), int(b)) for a, b in cfg if b > a], k_loop, confine)
        R2 = r2_matrix(L, confined=True)
        np.fill_diagonal(R2, np.inf)
        acc += R2 ** -1.5
    return acc / max(len(configs), 1)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    t_start = time.time()
    say("=" * 100)
    say("THE TWO UNSTATED DEFAULTS -- calibrated from geometry, then the stopping rule applies")
    say("=" * 100)
    say("  loop 35 had the direction right and the magnitude 14x too small. Two choices in it were")
    say("  never argued for: loop bonds as stiff as backbone bonds, and CTCF blocking normalised by")
    say("  the single strongest peak. Both are calculable.")

    # ---- B1 the stiffness identity -----------------------------------------------------------------
    s = 50
    target = (RING_NM / BEAD_NM) ** 2
    Lt = laplacian_k(s + 1, [(0, s)], K_LOOP, 0.0)
    R2t = r2_matrix(Lt, confined=False)
    got = float(R2t[0, s])
    b1 = bool(abs(got - target) / target < 0.05)
    say(f"\n  B1 THE STIFFNESS CALIBRATION IS SELF-CONSISTENT")
    say(f"     a cohesin ring is {RING_NM:.0f} nm across, a 25 kb bead {BEAD_NM:.0f} nm, so anchors")
    say(f"     should sit at ({RING_NM:.0f}/{BEAD_NM:.0f})^2 = {target:.4f} b^2")
    say(f"     isolated {s}-bond loop at k = {K_LOOP}: anchors at {got:.4f} b^2")
    say(f"     for comparison, k = 1 (loop 35's silent default) gives "
        f"{float(r2_matrix(laplacian_k(s+1, [(0, s)], 1.0, 0.0), confined=False)[0, s]):.4f} b^2 -- "
        f"a 2% pull on a {s}-bond span")
    say(f"     B1 {'PASS' if b1 else 'FAIL'}")

    # ---- rebuild the landscape, now ranked rather than max-normalised -------------------------------
    import gzip
    peaks = []
    with gzip.open(LE.CTCF if hasattr(LE, "CTCF") else SC / "ctcf_gm12878_hg19.bed.gz", "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] != "chr21":
                continue
            st, en = int(p[1]), int(p[2])
            off = int(p[9]) if len(p) > 9 and p[9].strip().lstrip("-").isdigit() else -1
            peaks.append({"summit": st + (off if off >= 0 else (en - st) // 2),
                          "signal": float(p[6])})
    seq = "".join(l.strip() for l in gzip.open(SC / "hg19_chr21.fa.gz", "rt") if l[0] != ">").upper()
    pfm = json.load(open(SC / "ctcf_pfm.json"))
    Lw = len(pfm["A"])
    W = np.array([pfm[b] for b in "ACGT"], float).T
    W = np.log2((W / W.sum(1, keepdims=True) + 1e-3) / 0.25)
    ix = {c: i for i, c in enumerate("ACGT")}

    def scr(x):
        return -1e9 if len(x) != Lw or any(c not in ix for c in x) else \
            float(sum(W[i, ix[c]] for i, c in enumerate(x)))

    def rc(x):
        return x.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    for p in peaks:
        w = seq[max(0, p["summit"] - 150):p["summit"] + 150]
        best, bo = -1e9, 0
        for i in range(len(w) - Lw + 1):
            sub = w[i:i + Lw]
            f_, r_ = scr(sub), scr(rc(sub))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0

    sig = np.array([p["signal"] for p in peaks])
    rank = sig.argsort().argsort() / max(len(sig) - 1, 1)      # 0..1 by RANK, not by max
    say(f"\n  CTCF blocking rescaled by RANK: the median IDR peak now blocks at "
        f"{0.5*LE.MAX_BLOCK:.2f} where max-normalisation gave "
        f"{float(np.median(sig)/sig.max())*LE.MAX_BLOCK:.2f}")

    H = np.load(SC / "hic_chr21_25kb.npy").astype(np.float64)
    H[H == 0] = np.nan
    n = len(H)
    mask = np.isfinite(H).sum(1) > 50
    ins_h = insulation(H)
    tgt = json.load(open(OUT / "loop_hic_target.json"))
    measured_orient = tgt["target"]["convergent_minus_other"]["measured"]

    def landscape(orients):
        bf, br = np.zeros(n), np.zeros(n)
        for p, o, rk in zip(peaks, orients, rank):
            b = p["summit"] // 25000
            if 0 <= b < n and o != 0:
                arr = bf if o > 0 else br
                arr[b] = max(arr[b], rk)
        return bf, br

    orients = [p["orient"] for p in peaks]
    fs = {p["summit"] // 25000 for p, o in zip(peaks, orients) if o > 0}
    rs = {p["summit"] // 25000 for p, o in zip(peaks, orients) if o < 0}

    res = {}
    for name, ors in (("real CTCF", orients),
                      ("SHUFFLED orientation", list(rng.permutation(orients)))):
        bf, br = landscape(ors)
        cfgs, n_coh, _, _ = LE.simulate(n, bf, br, np.random.default_rng(SEED))
        S = contact_map_k(n, cfgs, K_LOOP, LE.CONFINE)
        S[~np.isfinite(S)] = np.nan
        e = expected(S, mask)
        ok = np.isfinite(insulation(S)) & np.isfinite(ins_h) & mask
        r = float(np.corrcoef(insulation(S)[ok], ins_h[ok])[0, 1])
        o, npair = LE.orientation_effect(S, e, fs, rs, mask, n)
        ps = LE.ps_slope_map(S, mask)
        res[name] = {"insul_r": r, "orient": o, "ps": ps, "n_pairs": npair}
        say(f"    {name:<24} P(s) {ps:+.4f}   insulation r {r:+.4f}   orientation {o:+.4f}")

    real, shuf = res["real CTCF"], res["SHUFFLED orientation"]
    b2 = bool(real["insul_r"] >= GATE_INSUL and real["insul_r"] > shuf["insul_r"])
    say(f"\n  B2 INSULATION CLEARS THE GATE -- {real['insul_r']:+.4f} vs shuffled "
        f"{shuf['insul_r']:+.4f}   (loop 35 got +0.1878)")
    say(f"     B2 {'PASS' if b2 else 'FAIL'}  (gate {GATE_INSUL})")

    ratio = measured_orient / real["orient"] if real["orient"] > 0 else float("inf")
    b3 = bool(real["orient"] > 0 and ratio <= GATE_ORIENT_FACTOR
              and real["orient"] > 2 * max(shuf["orient"], 0))
    say(f"\n  B3 THE ORIENTATION EFFECT REACHES THE RIGHT ORDER")
    say(f"     measured {measured_orient:+.4f}   simulated {real['orient']:+.4f}   "
        f"shuffled {shuf['orient']:+.4f}")
    say(f"     off by a factor of {ratio:.1f}   (loop 35 was off by 13.9)")
    say(f"     B3 {'PASS' if b3 else 'FAIL'}  (gate: within {GATE_ORIENT_FACTOR}x AND collapses)")

    mins = (time.time() - t_start) / 60.0
    b4 = bool(mins <= GATE_MINUTES)
    say(f"\n  B4 STILL AFFORDABLE -- {mins:.1f} min   B4 {'PASS' if b4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"B1 stiffness calibration self-consistent": b1, "B2 insulation clears the gate": b2,
             "B3 orientation effect right order": b3, "B4 still affordable": b4}
    for k, v in gates.items():
        say(f"  {k:<44}{'PASS' if v else 'FAIL'}")
    if b2 and b3:
        say("  THE DEFAULTS WERE THE PROBLEM. Both constants came from geometry rather than from the")
        say("  map, and fixing them brought the model onto the measurement. The mechanism was right in")
        say("  loop 35; the cohesin was simply too soft to act like one.")
    else:
        say("  THE STOPPING RULE APPLIES, AND IT WAS DECLARED BEFORE THE RUN. Both constants were")
        say("  computed from geometry that has nothing to do with this map, and the gates still do not")
        say("  clear. The recorded conclusion is that the GAUSSIAN MEAN-FIELD TREATMENT is the limit --")
        say("  no excluded volume, and a contact probability that is an average of an average, which")
        say("  smooths exactly the sharp anchor-to-anchor structure being measured. A third calibration")
        say("  is not attempted, because at that point it would be fitting and not physics.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy")], available=n, used=int(mask.sum()),
                      selection="filtered", seed=SEED,
                      controls=["shuffled-orientation twin", "stiffness identity as a positive control",
                                "loop 35's numbers as the direct before/after",
                                "a stopping rule declared before the run"],
                      note="both constants derived from geometry independent of the Hi-C map; if the "
                           "gates fail the limit is recorded as the Gaussian approximation, not as a "
                           "need for more tuning")
    RM.report(man, emit=say)
    json.dump({"test": "loop_stiffness", "manifest": man, "gates": gates,
               "k_loop": K_LOOP, "anchor_r2": got, "anchor_target": target,
               "results": res, "measured_orient": measured_orient,
               "orient_factor_off": ratio, "minutes": mins, "log": log},
              open(OUT / "loop_stiffness.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_stiffness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
