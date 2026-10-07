"""COMPARTMENTS -- the half of chromosome structure extrusion cannot make, and whether adding it helps.

THE SEPARATION OF MECHANISM.  Chromosomes carry two structures that look similar in a Hi-C map and are
made by completely different processes:

    TADs           local insulated squares along the diagonal, made by cohesin extrusion stopped by
                   CTCF. loops 35-36 built this and it works.
    COMPARTMENTS   a long-range checkerboard: active regions contact other active regions megabases
                   away, inactive with inactive. Extrusion cannot produce this. Extrusion is local and
                   sequential; it has no way to bring two distant like-labelled regions together.
                   Compartments come from homotypic affinity -- similar chromatin sticking to similar.

So this loop starts with a negative that MUST hold. If the extrusion model already shows compartments,
something is leaking -- most likely the observed/expected normalisation or the GC track contaminating a
correlation -- and every compartment number after it would be an artefact.

WHAT IS ADDED.  Weak homotypic bonds. Each bin is labelled A or B by GC content computed from the hg19
chr21 sequence -- not from the Hi-C map -- and each bin gets a few extra edges to randomly chosen bins
of its OWN type, at low weight. In a Gaussian network that is exactly what an affinity is: more
conductance between like regions, so they sit closer. No new machinery, one more term in the same
Laplacian, and the cost is unchanged.

THE GATE THAT MATTERS, AND WHY IT IS NOT "DOES A CHECKERBOARD APPEAR".  The model is DRIVEN by GC, and
loop 33 measured that GC alone already correlates with the real compartment eigenvector at 0.485. A
model fed GC will therefore produce a GC-shaped answer whatever the physics does. The only honest
question is whether the polymer adds anything to the track it was handed:

    does the MODEL's eigenvector predict the MEASURED eigenvector better than GC does by itself?

If not, the physics is a pass-through and the right report is that GC was doing the work.

PREDECLARED, before any number:

    A1 EXTRUSION ALONE MAKES NO COMPARTMENTS   the loop-36 model's first eigenvector must NOT track the
                measured one beyond shuffles.
                A NEGATIVE THAT MUST HOLD. If it fails, the pipeline is leaking and A2-A4 are void.
    A2 HOMOTYPIC AFFINITY MAKES THEM   with the affinity term, the model's eigenvector correlates with
                the measured eigenvector above the shuffled band.
    A3 IT BEATS ITS OWN INPUT TRACK   that correlation must exceed GC's own 0.485 correlation with the
                measured eigenvector.
                THIS IS THE REAL GATE. Anything less means the polymer re-emitted what it was fed.
    A4 IT DOES NOT BREAK WHAT ALREADY WORKED   P(s) stays inside loop 33's window and the orientation
                effect stays positive and above its shuffled twin.
                A compartment term that buys a checkerboard by destroying TADs has not improved the
                model, it has traded one structure for another.

CONTROLS: the extrusion-only model as the must-fail negative; GC shuffled along the chromosome as the
affinity control; GC's own correlation as the direct comparator for A3; and loops 35-36's P(s) and
orientation numbers carried forward so a regression is visible immediately.

-> outputs/loop_compartment.json
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
from loop_polymer import r2_matrix  # noqa: E402
from loop_hic_target import expected  # noqa: E402
import loop_extrusion as LE  # noqa: E402
from loop_stiffness import laplacian_k, K_LOOP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
N_SHUFFLE = 200
K_AFFINITY = 0.06        # weak, and the same for A-A and B-B
N_PARTNERS = 6           # extra edges per bin, to same-type bins
N_CONFIG = 30
CHROMATIN_WINDOW = (-1.16, -0.76)


def eigenvector(M, mask):
    """First eigenvector of the observed/expected correlation matrix -- the compartment call."""
    n = len(M)
    exp = expected(M, mask)
    O = np.ones_like(M) * np.nan
    for d in range(n):
        if np.isfinite(exp[d]) and exp[d] > 0:
            i = np.arange(0, n - d)
            O[i, i + d] = M[i, i + d] / exp[d]
            O[i + d, i] = O[i, i + d]
    sub = np.nan_to_num(O[np.ix_(mask, mask)], nan=1.0)
    C = np.nan_to_num(np.corrcoef(np.log(np.clip(sub, 1e-9, None))))
    w, v = np.linalg.eigh(C)
    return v[:, -1]


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("COMPARTMENTS -- the half of chromosome structure extrusion cannot make")
    say("=" * 100)
    say("  Extrusion is local and sequential, so it has no way to bring two distant like-labelled")
    say("  regions together. Compartments need homotypic affinity. This starts from a negative that")
    say("  MUST hold, then adds the missing term and asks whether it beats the track that drove it.")

    H = np.load(SC / "hic_chr21_25kb.npy").astype(np.float64)
    H[H == 0] = np.nan
    n = len(H)
    mask = np.isfinite(H).sum(1) > 50
    pc_h = eigenvector(H, mask)

    seq = "".join(l.strip() for l in gzip.open(SC / "hg19_chr21.fa.gz", "rt") if l[0] != ">").upper()
    gc = np.full(n, np.nan)
    for i in range(n):
        s = seq[i * 25000:(i + 1) * 25000]
        eff = len(s) - s.count("N")
        if eff > 1000:
            gc[i] = (s.count("G") + s.count("C")) / eff
    gsub = gc[mask]
    good = np.isfinite(gsub)
    if np.corrcoef(pc_h[good], gsub[good])[0, 1] < 0:
        pc_h = -pc_h
    r_gc = float(np.corrcoef(pc_h[good], gsub[good])[0, 1])
    say(f"\n  measured eigenvector vs GC: {r_gc:+.4f}  (loop 33 got +0.4848; GC is the input track and")
    say(f"  therefore the bar the physics has to clear)")

    # rebuild the loop-36 landscape
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
    Lw = len(pfm["A"])
    W = np.log2((np.array([pfm[c] for c in "ACGT"], float).T /
                 np.array([pfm[c] for c in "ACGT"], float).T.sum(1, keepdims=True) + 1e-3) / 0.25)
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
    bf, br = np.zeros(n), np.zeros(n)
    for p, rk in zip(peaks, rank):
        b = p["summit"] // 25000
        if 0 <= b < n and p["orient"] != 0:
            (bf if p["orient"] > 0 else br)[b] = max((bf if p["orient"] > 0 else br)[b], rk)
    fs = {p["summit"] // 25000 for p in peaks if p["orient"] > 0}
    rs = {p["summit"] // 25000 for p in peaks if p["orient"] < 0}
    cfgs, n_coh, _, _ = LE.simulate(n, bf, br, np.random.default_rng(SEED))
    cfgs = cfgs[:N_CONFIG]

    def affinity_edges(track, r):
        """Extra edges between bins of the SAME type. This is what a homotypic affinity IS in a
        Gaussian network: more conductance between like regions, so they sit closer."""
        ok = np.where(np.isfinite(track))[0]
        med = np.nanmedian(track)
        A = ok[track[ok] >= med]
        B = ok[track[ok] < med]
        e = []
        for grp in (A, B):
            if len(grp) < 2:
                continue
            for i in grp:
                for j in r.choice(grp, size=min(N_PARTNERS, len(grp) - 1), replace=False):
                    if j != i:
                        e.append((int(min(i, j)), int(max(i, j))))
        return e

    def build(track, label):
        edges = affinity_edges(track, np.random.default_rng(SEED)) if track is not None else []
        acc = np.zeros((n, n))
        for cfg in cfgs:
            L = laplacian_k(n, [(int(a), int(b)) for a, b in cfg if b > a], K_LOOP, LE.CONFINE)
            for i, j in edges:
                L[i, j] -= K_AFFINITY
                L[j, i] -= K_AFFINITY
                L[i, i] += K_AFFINITY
                L[j, j] += K_AFFINITY
            R2 = r2_matrix(L, confined=True)
            np.fill_diagonal(R2, np.inf)
            acc += R2 ** -1.5
        S = acc / len(cfgs)
        pc = eigenvector(S, mask)
        if np.corrcoef(pc[good], pc_h[good])[0, 1] < 0:
            pc = -pc
        r = float(np.corrcoef(pc[good], pc_h[good])[0, 1])
        e = expected(S, mask)
        o, _ = LE.orientation_effect(S, e, fs, rs, mask, n)
        ps = LE.ps_slope_map(S, mask)
        say(f"    {label:<32} eigenvector r {r:+.4f}   P(s) {ps:+.4f}   orientation {o:+.4f}   "
            f"({len(edges)} affinity edges)")
        return {"r": r, "ps": ps, "orient": o, "n_edges": len(edges)}

    say("")
    res = {"extrusion only": build(None, "extrusion only (must fail A1)"),
           "affinity from GC": build(gc, "+ homotypic affinity from GC"),
           "affinity from SHUFFLED GC": build(
               np.where(np.isfinite(gc), rng.permutation(np.where(np.isfinite(gc), gc, np.nan)), np.nan),
               "+ affinity from SHUFFLED GC")}

    # shuffle band for the eigenvector correlation
    nullr = np.array([abs(np.corrcoef(rng.permutation(pc_h[good]), pc_h[good])[0, 1])
                      for _ in range(N_SHUFFLE)])
    band = float(np.percentile(nullr, 95))

    a1 = bool(abs(res["extrusion only"]["r"]) <= band)
    say(f"\n  A1 EXTRUSION ALONE MAKES NO COMPARTMENTS -- r {res['extrusion only']['r']:+.4f} against a")
    say(f"     shuffled band of {band:.4f}. This is a negative that MUST hold; if extrusion already")
    say(f"     produced a checkerboard, the pipeline would be leaking and nothing below could be read.")
    say(f"     A1 {'PASS' if a1 else 'FAIL'}")

    a2 = bool(res["affinity from GC"]["r"] > band)
    say(f"\n  A2 HOMOTYPIC AFFINITY MAKES THEM -- r {res['affinity from GC']['r']:+.4f} vs band "
        f"{band:.4f}")
    say(f"     shuffled-GC control: {res['affinity from SHUFFLED GC']['r']:+.4f}")
    say(f"     A2 {'PASS' if a2 else 'FAIL'}")

    a3 = bool(res["affinity from GC"]["r"] > r_gc)
    say(f"\n  A3 IT BEATS ITS OWN INPUT TRACK -- model {res['affinity from GC']['r']:+.4f} vs GC alone "
        f"{r_gc:+.4f}")
    say(f"     THE REAL GATE. The model was handed GC, so anything at or below GC's own correlation")
    say(f"     means the polymer re-emitted what it was fed.")
    say(f"     A3 {'PASS' if a3 else 'FAIL'}")

    st = json.load(open(OUT / "loop_stiffness.json"))
    prev_o = st["results"]["real CTCF"]["orient"]
    g = res["affinity from GC"]
    a4 = bool(CHROMATIN_WINDOW[0] <= g["ps"] <= CHROMATIN_WINDOW[1] and g["orient"] > 0
              and g["orient"] > 0.5 * prev_o)
    say(f"\n  A4 IT DOES NOT BREAK WHAT ALREADY WORKED")
    say(f"     P(s) {g['ps']:+.4f}  (window [{CHROMATIN_WINDOW[0]}, {CHROMATIN_WINDOW[1]}]; "
        f"loop 36 had {st['results']['real CTCF']['ps']:+.4f})")
    say(f"     orientation {g['orient']:+.4f}  (loop 36 had {prev_o:+.4f})")
    say(f"     A4 {'PASS' if a4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"A1 extrusion alone makes none": a1, "A2 affinity makes them": a2,
             "A3 beats its own input track": a3, "A4 does not break TADs": a4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if a1 and a2 and a3 and a4:
        say("  BOTH STRUCTURES, FROM TWO MECHANISMS, IN ONE MODEL. Extrusion makes the TADs and cannot")
        say("  make the compartments; affinity makes the compartments and does not cost the TADs; and")
        say("  the combination predicts the measured eigenvector better than the track it was given.")
    elif a1 and a2 and not a3:
        say("  THE COMPARTMENTS ARE THE INPUT TRACK COMING BACK OUT. Affinity produces a checkerboard,")
        say(f"  but at {g['r']:+.4f} against GC's own {r_gc:+.4f} the polymer added nothing to what it")
        say("  was handed. Honest report: GC was doing the work, and the physics was a pass-through.")
    elif not a1:
        say("  THE NEGATIVE CONTROL FAILED. Extrusion alone produced a compartment signal, which means")
        say("  something upstream is leaking and no number in this loop can be believed.")
    else:
        say("  THE AFFINITY TERM DOES NOT DELIVER. Recorded as measured.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy"), str(SC / "hg19_chr21.fa.gz")],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=["extrusion-only model as a must-fail negative",
                                "GC shuffled along the chromosome",
                                "GC's own correlation as the direct comparator",
                                f"{N_SHUFFLE} eigenvector shuffles",
                                "loops 35-36 P(s) and orientation carried forward"],
                      note="the model is driven by GC, so the only honest gate is whether it beats GC")
    RM.report(man, emit=say)
    json.dump({"test": "loop_compartment", "manifest": man, "gates": gates,
               "r_gc_vs_measured": r_gc, "band": band, "results": res,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_compartment.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_compartment.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
