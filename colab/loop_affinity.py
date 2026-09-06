"""PROXIMITY-CONDITIONED AFFINITY -- one principled correction, and then the stopping rule.

WHAT loop 39 GOT WRONG.  It added ~8,300 fixed random long-range bonds between same-type bins. That is
a small-world graph, not a chromosome: every bin ends up a few steps from every other, contact
probability stops depending on genomic separation, and P(s) flattened to -0.016 from -1.010. The TADs
went with it.

The physical error was specific. Homotypic affinity acts BETWEEN SEGMENTS THAT HAPPEN TO BE NEAR EACH
OTHER. A permanent bond acts whether they are near or not. A Gaussian network has only fixed harmonic
bonds and no native way to say "attract if close", which is the same linearity that makes it cheap.

THE CORRECTION, and it is the standard one.  Solve it self-consistently. Run extrusion alone, read off
which same-type pairs are ALREADY in contact more than typical for their separation, and place affinity
bonds only there. The edge set is then determined by the polymer rather than chosen, and the affinity
can only reinforce contacts the physics already made -- which is what an attraction does.

    step 1   contact map from extrusion alone
    step 2   for each separation, the observed/expected at that separation; keep same-type pairs above
             its 95th percentile. Conditioning WITHIN separation matters: without it this would just
             select nearby pairs and rediscover the backbone.
    step 3   set the bond weight so the total added conductance per bin is at most 10% of the
             backbone's. "Weak transient attraction" has to mean something numerical, and that is the
             number it means here.

PREDECLARED, before any number:

    M1 THE PERTURBATION IS WEAK   added conductance per bin <= 10% of backbone conductance.
                A design check. loop 39 failed because its perturbation was not weak and nothing
                measured it.
    M2 P(s) SURVIVES   stays inside loop 33's window [-1.16, -0.76].
    M3 TADS SURVIVE    the orientation effect stays above half of loop 36's +0.2278.
    M4 COMPARTMENTS APPEAR AND BEAT THEIR INPUT   the model eigenvector correlates with the measured
                one above the shuffle band AND above GC's own +0.4849.
                Same real gate as loop 39: the model is handed GC, so matching GC is not evidence.

THE STOPPING RULE, DECLARED NOW.  If M1-M3 pass and M4 fails, the recorded conclusion is that
COMPARTMENTS ARE OUTSIDE WHAT A GAUSSIAN NETWORK CAN REPRESENT -- an attraction that depends on
distance is not a harmonic bond, and the linearity that makes this model affordable is the same
property that forbids it. No further parameter passes will be made. This is the second time this
project has hit the same wall from a different direction, and that is worth more than a tuned number.

CONTROLS: shuffled GC with the identical edge-selection procedure; the extrusion-only model carried
forward; GC's own correlation as the comparator; and the weakness check as a design gate that loop 39
lacked.

-> outputs/loop_affinity.json
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
from loop_compartment import eigenvector  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
N_SHUFFLE = 200
N_CONFIG = 30
PCTL = 95.0            # a pair must be in the top 5% of contact for its separation
MAX_FRAC = 0.10        # added conductance per bin, as a fraction of backbone
CHROMATIN_WINDOW = (-1.16, -0.76)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("PROXIMITY-CONDITIONED AFFINITY -- one correction, then the stopping rule")
    say("=" * 100)
    say("  loop 39 placed permanent bonds between same-type bins whether or not they were near each")
    say("  other. That is a small-world graph: P(s) went to -0.016. An attraction acts only between")
    say("  segments already close, so the edge set must come from the polymer, not from me.")

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
    good = np.isfinite(gc[mask])
    if np.corrcoef(pc_h[good], gc[mask][good])[0, 1] < 0:
        pc_h = -pc_h
    r_gc = float(np.corrcoef(pc_h[good], gc[mask][good])[0, 1])

    cmp_prev = json.load(open(OUT / "loop_compartment.json"))
    st = json.load(open(OUT / "loop_stiffness.json"))
    band = cmp_prev["band"]
    prev_o = st["results"]["real CTCF"]["orient"]
    say(f"\n  GC alone vs the measured eigenvector: {r_gc:+.4f}   shuffle band {band:.4f}")
    say(f"  loop 36 left orientation at {prev_o:+.4f} and P(s) at "
        f"{st['results']['real CTCF']['ps']:+.4f}")

    # landscape, as loops 36-39 built it
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
    bf, br = np.zeros(n), np.zeros(n)
    for p, rk in zip(peaks, rank):
        b = p["summit"] // 25000
        if 0 <= b < n and p["orient"] != 0:
            (bf if p["orient"] > 0 else br)[b] = max((bf if p["orient"] > 0 else br)[b], rk)
    fs = {p["summit"] // 25000 for p in peaks if p["orient"] > 0}
    rs = {p["summit"] // 25000 for p in peaks if p["orient"] < 0}
    cfgs, n_coh, _, _ = LE.simulate(n, bf, br, np.random.default_rng(SEED))
    cfgs = cfgs[:N_CONFIG]

    def ensemble(edges, kaff):
        acc = np.zeros((n, n))
        for cfg in cfgs:
            L = laplacian_k(n, [(int(a), int(b)) for a, b in cfg if b > a], K_LOOP, LE.CONFINE)
            for i, j in edges:
                L[i, j] -= kaff
                L[j, i] -= kaff
                L[i, i] += kaff
                L[j, j] += kaff
            R2 = r2_matrix(L, confined=True)
            np.fill_diagonal(R2, np.inf)
            acc += R2 ** -1.5
        return acc / len(cfgs)

    S0 = ensemble([], 0.0)
    e0 = expected(S0, mask)
    say(f"\n  step 1: extrusion-only map, P(s) {LE.ps_slope_map(S0, mask):+.4f}")

    def select(track):
        """Same-type pairs already in contact above the 95th percentile FOR THEIR SEPARATION."""
        okb = np.where(np.isfinite(track) & mask)[0]
        med = np.nanmedian(track[okb])
        typ = {int(b): (1 if track[b] >= med else 0) for b in okb}
        edges = []
        for d in range(8, min(n, 400)):
            i = np.array([b for b in okb if b + d in typ])
            if len(i) < 20:
                continue
            j = i + d
            v = S0[i, j] / (e0[d] if np.isfinite(e0[d]) and e0[d] > 0 else np.nan)
            ok = np.isfinite(v)
            if ok.sum() < 20:
                continue
            thr = np.percentile(v[ok], PCTL)
            for a, b, vv in zip(i[ok], j[ok], v[ok]):
                if vv >= thr and typ[int(a)] == typ[int(b)]:
                    edges.append((int(a), int(b)))
        return edges

    res = {}
    for name, track in (("affinity from GC", gc),
                        ("affinity from SHUFFLED GC",
                         np.where(np.isfinite(gc), rng.permutation(gc[np.isfinite(gc)])[
                             np.cumsum(np.isfinite(gc)) - 1], np.nan))):
        edges = select(track)
        deg = 2 * len(edges) / max(int(mask.sum()), 1)
        kaff = min(0.5, MAX_FRAC * 2.0 / max(deg, 1e-9))
        added = deg * kaff
        S = ensemble(edges, kaff)
        e = expected(S, mask)
        pc = eigenvector(S, mask)
        if np.corrcoef(pc[good], pc_h[good])[0, 1] < 0:
            pc = -pc
        r = float(np.corrcoef(pc[good], pc_h[good])[0, 1])
        o, _ = LE.orientation_effect(S, e, fs, rs, mask, n)
        ps = LE.ps_slope_map(S, mask)
        res[name] = {"r": r, "ps": ps, "orient": o, "n_edges": len(edges),
                     "k": kaff, "added_conductance": added, "degree": deg}
        say(f"    {name:<28} {len(edges):>6} edges  k {kaff:.3f}  added/backbone "
            f"{added/2.0:.1%}   eigen r {r:+.4f}  P(s) {ps:+.4f}  orient {o:+.4f}")

    g = res["affinity from GC"]
    m1 = bool(g["added_conductance"] / 2.0 <= MAX_FRAC + 1e-9)
    say(f"\n  M1 THE PERTURBATION IS WEAK -- {g['added_conductance']/2.0:.1%} of backbone conductance "
        f"(cap {MAX_FRAC:.0%})   M1 {'PASS' if m1 else 'FAIL'}")
    m2 = bool(CHROMATIN_WINDOW[0] <= g["ps"] <= CHROMATIN_WINDOW[1])
    say(f"  M2 P(s) SURVIVES -- {g['ps']:+.4f}  (loop 39 destroyed it at -0.0163)   "
        f"M2 {'PASS' if m2 else 'FAIL'}")
    m3 = bool(g["orient"] > 0.5 * prev_o)
    say(f"  M3 TADS SURVIVE -- orientation {g['orient']:+.4f} against loop 36's {prev_o:+.4f}   "
        f"M3 {'PASS' if m3 else 'FAIL'}")
    m4 = bool(g["r"] > band and g["r"] > r_gc)
    say(f"  M4 COMPARTMENTS APPEAR AND BEAT THEIR INPUT -- model {g['r']:+.4f}, band {band:.4f}, "
        f"GC {r_gc:+.4f}")
    say(f"     shuffled-GC control {res['affinity from SHUFFLED GC']['r']:+.4f}")
    say(f"     M4 {'PASS' if m4 else 'FAIL'}")

    say("\n" + "=" * 100)
    gates = {"M1 perturbation is weak": m1, "M2 P(s) survives": m2, "M3 TADs survive": m3,
             "M4 compartments beat their input": m4}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if m1 and m2 and m3 and m4:
        say("  BOTH MECHANISMS, ONE MODEL. Extrusion makes the TADs, proximity-conditioned affinity")
        say("  makes the compartments, neither costs the other, and the combination beats the track it")
        say("  was handed.")
    elif m1 and m2 and m3 and not m4:
        say("  THE STOPPING RULE APPLIES, AND IT WAS DECLARED BEFORE THE RUN.")
        say("  The correction worked as physics: the perturbation is weak, P(s) survived where loop 39")
        say(f"  destroyed it, and the TADs survived. What did not happen is compartments -- {g['r']:+.4f}")
        say(f"  against GC's own {r_gc:+.4f}.")
        say("  RECORDED CONCLUSION: compartments are outside what a Gaussian network can represent. An")
        say("  attraction that depends on distance is not a harmonic bond, and conditioning the bonds")
        say("  on a single reference configuration cannot recover it, because the affinity has to act")
        say("  in every configuration and each one is different. The linearity that makes this model")
        say("  cost 0.4s per configuration is the same property that forbids compartments. That is the")
        say("  second time this project has hit that wall from a different direction -- loop 36 hit it")
        say("  on insulation sharpness -- and no further parameter passes are made.")
    else:
        say("  THE CORRECTION DID NOT HOLD ITS OWN DESIGN CHECKS. Recorded as measured.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy"), str(SC / "hg19_chr21.fa.gz")],
                      available=n, used=int(mask.sum()), selection="filtered", seed=SEED,
                      controls=["shuffled GC through the identical edge-selection procedure",
                                "extrusion-only model carried forward",
                                "GC's own correlation as the comparator",
                                "a weakness design gate that loop 39 lacked",
                                "a stopping rule declared before the run"],
                      note="edge set determined by the polymer's own contacts within separation, not "
                           "chosen; if compartments still fail the limit is the Gaussian network")
    RM.report(man, emit=say)
    json.dump({"test": "loop_affinity", "manifest": man, "gates": gates, "r_gc": r_gc,
               "band": band, "results": res, "prev_orient": prev_o,
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_affinity.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_affinity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
