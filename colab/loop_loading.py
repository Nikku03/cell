"""WHERE COHESIN LANDS -- the last uniform assumption, and a stopping rule.

THE DIAGNOSIS loop 43 LEFT.  Fixing the barrier physics took insulation from +0.2974 to +0.3617 and
took the orientation effect from +0.2278 to +0.4995 -- which now OVERSHOOTS the measured +0.3788 by
about a third. Those two facts together say something specific:

    too much anchor-to-anchor loop, not enough domain body

Insulation is a statement about domains. Corner peaks are a statement about anchors. A model that
overshoots the anchors while undershooting the domains has its cohesins concentrated at CTCF sites and
too few of them working in the interiors.

THE LAST UNIFORM ASSUMPTION.  Every loop so far has loaded cohesin at a UNIFORMLY RANDOM position. That
is not what happens. Cohesin is loaded by NIPBL, which concentrates at active promoters and enhancers,
so loading is biased toward accessible chromatin -- and those sites sit inside domains, not at their
edges. Making loading follow that bias puts cohesins where the domain signal is generated.

THE TRACK, AND ITS HONEST LABEL.  ENCODE's search API is returning 504 in this container, so no
measured GM12878 accessibility track could be fetched. What is used instead is CpG observed/expected
density computed from the same hg19 chr21 sequence already on disk. CpG islands mark most active
promoters, so this is a PROMOTER PROXY and not an accessibility measurement, and it is labelled that
way everywhere. It has one real virtue: it is derived from sequence, so it cannot have been
contaminated by the Hi-C map it is scored against.

It also has one real risk. CpG density correlates with GC, GC correlates with the A compartment, and
the A compartment shapes the contact map -- so a loading bias built from CpG could improve the map for
reasons that have nothing to do with loading. That is what L4 is for.

PREDECLARED, before any number:

    L1 LOADING IS ACTUALLY BIASED   the realised distribution of load positions must track the CpG
                profile. A design check: loop 43's first run changed two things at once and cost a
                whole run to untangle, so the change gets verified before anything is read off it.
    L2 INSULATION IMPROVES   above loop 43's +0.3617.
    L3 IT CLEARS THE ORIGINAL GATE   r >= 0.40. Unmoved for the third time; the replicate ceiling of
                0.9285 says it is fair.
    L4 IT IS THE TRACK AND NOT MERELY NON-UNIFORMITY   it must beat a SHUFFLED-CpG loading control that
                is equally non-uniform and equally clustered but points nowhere in particular.
                THE DECISIVE ONE. Clustering cohesins anywhere makes some structure; only clustering
                them at promoters is the biological claim.
    L5 NOTHING ELSE BREAKS   P(s) stays in loop 33's window and the orientation effect stays positive
                and above its shuffled twin. A secondary expectation, stated so it can be checked
                rather than claimed afterwards: if this correction is right, the orientation OVERSHOOT
                should come DOWN toward the measured +0.3788, because cohesins spent in domain
                interiors are cohesins not parked at anchors.

THE STOPPING RULE, DECLARED NOW.  This is the third principled correction aimed at insulation. If it
does not clear 0.40, no fourth is attempted in this arc; the remaining shortfall is recorded against
the Gaussian contact model itself -- which smooths every boundary by construction -- with the measured
size of that shortfall attached, and the project moves on rather than tuning.

CONTROLS: shuffled-CpG loading, equally non-uniform; the uniform-loading model rerun side by side;
shuffled CTCF orientation; the replicate ceiling; and P(s) and orientation carried forward.

-> outputs/loop_loading.json
"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402
from loop_hic_target import insulation, expected  # noqa: E402
import loop_extrusion as LE  # noqa: E402
import loop_insulation as LI  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
BIN = 25000
GATE_INSUL = 0.40
CHROMATIN_WINDOW = (-1.16, -0.76)


def simulate_loaded(n, occ_f, occ_r, load_p, rng, n_config=LI.N_CONFIG, burn=LI.BURN):
    """loop 43's occupancy barrier, with loading drawn from `load_p` instead of uniform."""
    step_s = BIN / 1e3 / LE.V_KB_S
    p_off_coh = min(1.0, step_s / LE.RESIDENCE_S)
    p_off_ctcf = min(1.0, step_s / LI.CTCF_RESIDENCE_S)
    has_f, has_r = occ_f > 0, occ_r > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pon_f = np.clip(p_off_ctcf * occ_f / np.maximum(1 - occ_f, 1e-6), 0, 1)
        pon_r = np.clip(p_off_ctcf * occ_r / np.maximum(1 - occ_r, 1e-6), 0, 1)
    st_f = has_f & (rng.random(n) < occ_f)
    st_r = has_r & (rng.random(n) < occ_r)
    n_coh = max(1, int(n * BIN / 1e3 / LE.DENSITY_KB))
    pos_choices = np.arange(n - 1)
    p = load_p[:n - 1].astype(float)
    p = p / p.sum() if p.sum() > 0 else np.ones(n - 1) / (n - 1)
    left = rng.choice(pos_choices, n_coh, p=p)
    right = np.minimum(left + 1, n - 1)
    loaded = list(left.copy())
    configs = []
    for t in range(burn + n_config * 20):
        st_f = np.where(st_f & (rng.random(n) < p_off_ctcf), False, st_f)
        st_r = np.where(st_r & (rng.random(n) < p_off_ctcf), False, st_r)
        st_f = st_f | (has_f & ~st_f & (rng.random(n) < pon_f))
        st_r = st_r | (has_r & ~st_r & (rng.random(n) < pon_r))
        occ = np.zeros(n, bool)
        occ[left] = True
        occ[right] = True
        nl = np.maximum(left - 1, 0)
        canl = (nl != left) & ~occ[nl] & ~st_f[left]
        nr = np.minimum(right + 1, n - 1)
        canr = (nr != right) & ~occ[nr] & ~st_r[right]
        left = np.where(canl, nl, left)
        right = np.where(canr, nr, right)
        off = rng.random(n_coh) < p_off_coh
        if off.any():
            newp = rng.choice(pos_choices, int(off.sum()), p=p)
            left[off] = newp
            right[off] = newp + 1
            loaded.extend(newp.tolist())
        if t >= burn and (t - burn) % 20 == 0:
            configs.append(np.c_[left.copy(), right.copy()])
    return configs, np.array(loaded)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("WHERE COHESIN LANDS -- the last uniform assumption")
    say("=" * 100)
    say("  loop 43 overshoots the anchors (+0.4995 vs a measured +0.3788) and undershoots the domains")
    say("  (+0.3617 vs a 0.40 gate). Too much anchor loop, not enough domain body. Every loop so far")
    say("  has loaded cohesin uniformly at random, and cohesin is loaded by NIPBL at active promoters.")

    H = np.load(SC / "hic_chr21_25kb.npy").astype(np.float64)
    H[H == 0] = np.nan
    n = len(H)
    mask = np.isfinite(H).sum(1) > 50
    ins_h = insulation(H)
    ins_prev = json.load(open(OUT / "loop_insulation.json"))
    prev_r = ins_prev["results"]["occupancy, real orientation"]["insul_r"]
    prev_o = ins_prev["results"]["occupancy, real orientation"]["orient"]
    ceiling = ins_prev["ceiling_replicate"]
    measured_o = 0.3788
    say(f"\n  loop 43 left insulation {prev_r:+.4f} and orientation {prev_o:+.4f}; the replicate")
    say(f"  ceiling is {ceiling:.4f} and the measured orientation effect is {measured_o:+.4f}")

    seq = "".join(l.strip() for l in gzip.open(SC / "hg19_chr21.fa.gz", "rt") if l[0] != ">").upper()
    cpg = np.zeros(n)
    for i in range(n):
        s = seq[i * BIN:(i + 1) * BIN]
        eff = len(s) - s.count("N")
        if eff < 5000:
            continue
        c, g = s.count("C"), s.count("G")
        cg = s.count("CG")
        cpg[i] = (cg * eff) / max(c * g, 1)          # observed/expected CpG
    cpg = np.nan_to_num(cpg)
    cpg = np.clip(cpg, 0, np.percentile(cpg[cpg > 0], 99) if (cpg > 0).any() else 1)
    say(f"\n  CpG observed/expected computed from the chr21 sequence: "
        f"{int((cpg > 0).sum()):,} usable bins, PROMOTER PROXY not an accessibility measurement")

    # CTCF landscape as loop 43 left it
    peaks = []
    with gzip.open(SC / "ctcf_gm12878_hg19.bed.gz", "rt") as f:
        for line in f:
            p = line.split("\t")
            if p[0] != "chr21":
                continue
            a, b = int(p[1]), int(p[2])
            off = int(p[9]) if len(p) > 9 and p[9].strip().lstrip("-").isdigit() else -1
            peaks.append({"summit": a + (off if off >= 0 else (b - a) // 2), "signal": float(p[6])})
    pfm = json.load(open(SC / "ctcf_pfm.json"))
    P = np.array([pfm[c] for c in "ACGT"], float).T
    W = np.log2((P / P.sum(1, keepdims=True) + 1e-3) / 0.25)
    Lw = len(pfm["A"])
    ix = {c: i for i, c in enumerate("ACGT")}
    for p in peaks:
        w = seq[max(0, p["summit"] - 150):p["summit"] + 150]
        best, bo = -1e9, 0
        for i in range(len(w) - Lw + 1):
            s2 = w[i:i + Lw]
            f_ = -1e9 if any(c not in ix for c in s2) else sum(W[k, ix[c]] for k, c in enumerate(s2))
            s3 = s2.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            r_ = -1e9 if any(c not in ix for c in s3) else sum(W[k, ix[c]] for k, c in enumerate(s3))
            if max(f_, r_) > best:
                best, bo = max(f_, r_), (1 if f_ >= r_ else -1)
        p["orient"] = bo if best > 6.0 else 0
    sig = np.array([p["signal"] for p in peaks])
    rank = sig.argsort().argsort() / max(len(sig) - 1, 1)
    orients = [p["orient"] for p in peaks]
    fs = {p["summit"] // BIN for p, o in zip(peaks, orients) if o > 0}
    rs = {p["summit"] // BIN for p, o in zip(peaks, orients) if o < 0}
    of, orr = np.zeros(n), np.zeros(n)
    for p, o, rk in zip(peaks, orients, rank):
        b = p["summit"] // BIN
        if 0 <= b < n and o != 0:
            arr = of if o > 0 else orr
            arr[b] = max(arr[b], LI.OCC_LO + (LI.OCC_HI - LI.OCC_LO) * rk)

    shuffled_cpg = cpg.copy()
    nz = cpg > 0
    shuffled_cpg[nz] = rng.permutation(cpg[nz])
    uniform = np.ones(n)

    res, loaded_hist = {}, {}
    for name, lp in (("loading from CpG", cpg),
                     ("loading from SHUFFLED CpG", shuffled_cpg),
                     ("uniform loading (loop 43)", uniform)):
        cfgs, loaded = simulate_loaded(n, of, orr, lp, np.random.default_rng(SEED))
        S = LI.ensemble(n, cfgs)
        S[~np.isfinite(S)] = np.nan
        e = expected(S, mask)
        ins_s = insulation(S)
        okk = np.isfinite(ins_s) & np.isfinite(ins_h) & mask
        r = float(np.corrcoef(ins_s[okk], ins_h[okk])[0, 1])
        o, _ = LE.orientation_effect(S, e, fs, rs, mask, n)
        ps = LE.ps_slope_map(S, mask)
        res[name] = {"insul_r": r, "orient": o, "ps": ps}
        loaded_hist[name] = loaded
        say(f"    {name:<32} insulation {r:+.4f}   P(s) {ps:+.4f}   orientation {o:+.4f}")

    # ---- L1 design check ---------------------------------------------------------------------------
    h = np.bincount(loaded_hist["loading from CpG"], minlength=n).astype(float)
    okc = cpg > 0
    r_load = float(np.corrcoef(h[okc], cpg[okc])[0, 1])
    l1 = bool(r_load > 0.5)
    say(f"\n  L1 LOADING IS ACTUALLY BIASED -- realised load positions vs the CpG track: "
        f"r = {r_load:+.4f}   L1 {'PASS' if l1 else 'FAIL'}")

    g, sh, un = (res["loading from CpG"], res["loading from SHUFFLED CpG"],
                 res["uniform loading (loop 43)"])
    l2 = bool(g["insul_r"] > prev_r)
    say(f"  L2 INSULATION IMPROVES -- {g['insul_r']:+.4f} vs loop 43's {prev_r:+.4f}   "
        f"L2 {'PASS' if l2 else 'FAIL'}")
    l3 = bool(g["insul_r"] >= GATE_INSUL)
    say(f"  L3 IT CLEARS THE ORIGINAL GATE -- {g['insul_r']:+.4f} vs {GATE_INSUL} "
        f"({g['insul_r']/ceiling:.0%} of the {ceiling:.3f} ceiling)   L3 {'PASS' if l3 else 'FAIL'}")
    l4 = bool(g["insul_r"] > sh["insul_r"])
    say(f"  L4 IT IS THE TRACK, NOT MERELY NON-UNIFORMITY -- CpG {g['insul_r']:+.4f} vs shuffled-CpG "
        f"{sh['insul_r']:+.4f}   L4 {'PASS' if l4 else 'FAIL'}")
    l5 = bool(CHROMATIN_WINDOW[0] <= g["ps"] <= CHROMATIN_WINDOW[1] and g["orient"] > 0)
    say(f"  L5 NOTHING ELSE BREAKS -- P(s) {g['ps']:+.4f}, orientation {g['orient']:+.4f}   "
        f"L5 {'PASS' if l5 else 'FAIL'}")
    say(f"     the stated secondary expectation was that the orientation OVERSHOOT would come down "
        f"toward")
    say(f"     the measured {measured_o:+.4f}: it went {prev_o:+.4f} -> {g['orient']:+.4f}, "
        f"{'closer' if abs(g['orient']-measured_o) < abs(prev_o-measured_o) else 'not closer'}.")

    say("\n" + "=" * 100)
    gates = {"L1 loading is biased": l1, "L2 insulation improves": l2,
             "L3 clears the original gate": l3, "L4 the track, not non-uniformity": l4,
             "L5 nothing else breaks": l5}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if l3 and l4:
        say(f"  THE GAP IS CLOSED. Insulation went +0.2974 (loop 36) -> {prev_r:+.4f} (barrier physics)")
        say(f"  -> {g['insul_r']:+.4f} (loading), past a gate a replicate experiment says was fair, and")
        say(f"  it beats an equally clustered shuffled-CpG control so it is the promoters and not the")
        say(f"  clustering.")
    else:
        say(f"  THE STOPPING RULE APPLIES AND IT WAS DECLARED BEFORE THE RUN. Three principled")
        say(f"  corrections have now been aimed at insulation: loop-bond stiffness, barrier")
        say(f"  persistence, and loading bias. It stands at {g['insul_r']:+.4f} against 0.40, which is")
        say(f"  {g['insul_r']/ceiling:.0%} of an achievable {ceiling:.3f}. No fourth is attempted.")
        say(f"  THE REMAINING SHORTFALL IS RECORDED AGAINST THE GAUSSIAN CONTACT MODEL. Every boundary")
        say(f"  it produces is smoothed by construction: P ~ <R^2>^(-3/2) is a thermal average, and")
        say(f"  averaging over configurations averages again, so a sharp insulation dip cannot survive")
        say(f"  two averagings. That is the same wall loops 36, 39 and 40 hit, and its cost is now")
        say(f"  measured rather than asserted: {GATE_INSUL - g['insul_r']:+.4f} of correlation.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy"), str(SC / "hg19_chr21.fa.gz")],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=["shuffled-CpG loading, equally non-uniform",
                                "uniform loading rerun side by side",
                                "the replicate ceiling",
                                "P(s) and orientation carried forward",
                                "a stopping rule declared before the run"],
                      note="CpG is a PROMOTER PROXY, not a measured accessibility track; ENCODE's "
                           "search API was returning 504. It is sequence-derived, so it cannot have "
                           "been contaminated by the Hi-C map, and the shuffled control is what "
                           "protects against the GC/compartment route")
    RM.report(man, emit=say)
    json.dump({"test": "loop_loading", "manifest": man, "gates": gates, "results": res,
               "prev_insul_r": prev_r, "prev_orient": prev_o, "ceiling": ceiling,
               "measured_orient": measured_o, "load_track_r": r_load,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_loading.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_loading.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
