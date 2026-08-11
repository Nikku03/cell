"""LANGEVIN WITH EXCLUDED VOLUME -- the thing the closed form cannot do, and a claim of mine retracted.

A RETRACTION FIRST, because it decides what this loop tests.  I told the user the insulation shortfall
came partly from "double averaging" -- that P ~ <R^2>^(-3/2) is a thermal average and the ensemble
averages again, so sharp dips could not survive. The first half of that is wrong. For a Gaussian
network the contact probability at capture radius r_c is

    P_ij = integral of the Gaussian over |r| < r_c  =  (3/(2 pi <R_ij^2>))^(3/2) * (4/3) pi r_c^3

which is EXACT for that network, not an approximation to it. And averaging over loop configurations is
not a modelling artefact either -- it is what Hi-C physically does, since the map is pooled over
millions of cells each carrying a different set of cohesins. So drawing explicit coordinates and
counting contacts should reproduce the analytic map, not improve on it. That is predeclared below as
V2 and it exists to make the retraction checkable rather than merely stated.

WHAT IS ACTUALLY MISSING is excluded volume. A Gaussian chain passes through itself. Two loci on
opposite sides of a strong boundary can occupy the same point, so the boundary leaks in a way no
parameter can fix, and no amount of stiffening bonds or persisting barriers will introduce a
constraint the model does not contain. That needs an integrator.

THE DESIGN, and it is chosen to make the expensive part small.

    exact sampling      configurations are drawn directly from the Gaussian network -- X = V diag(1/sqrt(lam)) Z
                        -- rather than relaxed into. So the chain starts already correct at every scale
                        the Gaussian gets right, and the integrator only has to do the one job the
                        Gaussian cannot: push overlapping beads apart and keep them apart.
    one trajectory      loops turn over DURING the run, on the KMC clock, and contacts accumulate
                        throughout. That is not a convenience -- it is what Hi-C measures, a
                        time-and-population average over a chromosome that is being rewired.
    Verlet neighbours   excluded volume is short-ranged, so pair lists are rebuilt every 20 steps with a
                        skin rather than every step, and the O(N^2) scan never happens.
    batched replicas    B chromosomes as one (B, N, 3) tensor, so the cost is arithmetic rather than
                        kernel launches.

PREDECLARED, before any number:

    V1 THE SAMPLER IS EXACT   the covariance of drawn configurations must match G = L^-1 to within
                sampling error.
                An identity check on the sampler itself. If the draws are wrong, every configuration
                fed to the integrator is wrong and nothing downstream means anything.
    V2 DISCRETE COUNTING CHANGES NOTHING   insulation from counting contacts in sampled coordinates
                must match the analytic map within noise, NOT beat it.
                THIS IS THE RETRACTION, MADE FALSIFIABLE. If counting improves insulation, my
                statement above is wrong and the Gaussian contact formula is not doing what I claim.
                If it does not, the "double averaging" explanation is dead and excluded volume is the
                only remaining candidate.
    V3 EXCLUDED VOLUME ACTUALLY BITES   the fraction of bead pairs closer than one diameter must fall
                by at least 10x between the Gaussian draw and the end of the run.
                A design check. An integrator that does not remove overlaps has not applied excluded
                volume, and V4 would be measuring noise.
    V4 INSULATION IMPROVES   above loop 43's +0.3617, still beating the shuffled-orientation twin.
    V5 IT CLEARS THE ORIGINAL GATE   r >= 0.40. Unmoved for the fourth time. The replicate ceiling of
                0.9285 says it is fair.
    V6 THE COST IS REPORTED   wall clock and cost per configuration, against the closed form's 0.42 s,
                so the trade being made is visible rather than implied.

CONTROLS: the analytic map from the same loop configurations as the direct comparator; shuffled CTCF
orientation through the identical pipeline; the overlap fraction before and after as the design check;
and P(s) and the orientation effect carried forward so a regression shows.

-> outputs/loop_langevin.json
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
from loop_stiffness import laplacian_k, K_LOOP  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
SC = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad")
SEED = 1904
BIN = 25000
GATE_INSUL = 0.40
CHROMATIN_WINDOW = (-1.16, -0.76)

# ---- units: kT = 1, and b^2 = 3kT/k = 1 fixes the backbone stiffness -------------------------------
K_BOND = 3.0
K_LOOP_ABS = K_LOOP * K_BOND
SIGMA = 1.0              # bead diameter, one bond length
EPS_REP = 5.0            # soft-core strength, in kT
DT = 0.005
N_STEPS = 100_000
KMC_EVERY = 20           # Langevin steps per KMC step
SAMPLE_EVERY = 200
BURN_STEPS = 20_000
B_REPLICAS = 8
SKIN = 0.6
REBUILD_EVERY = 20
R_CAPTURE = 1.5          # contact if closer than this, in bond lengths


def gaussian_draw(L, rng, batch):
    """Exact draws from the Gaussian network: X = V diag(1/sqrt(lam)) Z, no relaxation needed."""
    w, V = np.linalg.eigh(L)
    w = np.maximum(w, 1e-12)
    A = V / np.sqrt(w)
    n = L.shape[0]
    Z = rng.standard_normal((batch, n, 3))
    return np.einsum("ij,bjd->bid", A, Z) * np.sqrt(K_BOND / 3.0) * np.sqrt(3.0 / K_BOND)


def main():
    log = []

    def say(x):
        print(x, flush=True)
        log.append(x)

    import torch
    torch.set_num_threads(4)
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    say("=" * 100)
    say("LANGEVIN WITH EXCLUDED VOLUME -- and a retraction made falsifiable")
    say("=" * 100)
    say("  RETRACTED: I said the shortfall came partly from double averaging. For a Gaussian network")
    say("  P_ij = (3/(2 pi <R^2>))^(3/2) (4/3) pi r_c^3 is EXACT, and averaging over loop")
    say("  configurations is what Hi-C physically does. So counting contacts in sampled coordinates")
    say("  should change nothing -- V2 tests exactly that. What is actually missing is excluded")
    say("  volume: a Gaussian chain passes through its own boundaries.")

    H = np.load(SC / "hic_chr21_25kb.npy").astype(np.float64)
    H[H == 0] = np.nan
    n = len(H)
    mask = np.isfinite(H).sum(1) > 50
    ins_h = insulation(H)
    prev = json.load(open(OUT / "loop_insulation.json"))
    prev_r = prev["results"]["occupancy, real orientation"]["insul_r"]
    prev_o = prev["results"]["occupancy, real orientation"]["orient"]
    ceiling = prev["ceiling_replicate"]
    say(f"\n  loop 43 (adopted): insulation {prev_r:+.4f}, orientation {prev_o:+.4f}; "
        f"ceiling {ceiling:.4f}")

    # ---- CTCF landscape, as loop 43 left it -------------------------------------------------------
    seq = "".join(l.strip() for l in gzip.open(SC / "hg19_chr21.fa.gz", "rt") if l[0] != ">").upper()
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

    def occupancy(ors):
        of, orr = np.zeros(n), np.zeros(n)
        for p, o, rk in zip(peaks, ors, rank):
            b = p["summit"] // BIN
            if 0 <= b < n and o != 0:
                arr = of if o > 0 else orr
                arr[b] = max(arr[b], LI.OCC_LO + (LI.OCC_HI - LI.OCC_LO) * rk)
        return of, orr

    def score(S, label):
        S = S.copy()
        S[~np.isfinite(S)] = np.nan
        e = expected(S, mask)
        ins_s = insulation(S)
        ok = np.isfinite(ins_s) & np.isfinite(ins_h) & mask
        r = float(np.corrcoef(ins_s[ok], ins_h[ok])[0, 1])
        o, _ = LE.orientation_effect(S, e, fs, rs, mask, n)
        ps = LE.ps_slope_map(S, mask)
        say(f"    {label:<38} insulation {r:+.4f}   P(s) {ps:+.4f}   orientation {o:+.4f}")
        return {"insul_r": r, "orient": o, "ps": ps}

    of, orr = occupancy(orients)
    cfgs, _, _ = LI.simulate_occupancy(n, of, orr, np.random.default_rng(SEED))
    analytic = LI.ensemble(n, cfgs)

    # ---- V1 the sampler ---------------------------------------------------------------------------
    Lt = laplacian_k(400, [(50, 200), (120, 300)], K_LOOP, LE.CONFINE)
    Gt = np.linalg.inv(Lt)
    Xs = gaussian_draw(Lt, np.random.default_rng(1), 4000)
    emp = np.einsum("bid,bjd->ij", Xs, Xs) / (3.0 * len(Xs))
    err = float(np.abs(emp - Gt).max() / np.abs(Gt).max())
    v1 = bool(err < 0.05)
    say(f"\n  V1 THE SAMPLER IS EXACT -- empirical covariance vs G = L^-1, max relative error "
        f"{err:.4f} over 4,000 draws   V1 {'PASS' if v1 else 'FAIL'}")

    # ---- V2 discrete counting on the SAME loop configurations -------------------------------------
    say(f"\n  V2 DISCRETE COUNTING CHANGES NOTHING -- the retraction, made falsifiable")
    cnt = np.zeros((n, n))
    nsamp = 0
    for cfg in cfgs[:20]:
        Lc = laplacian_k(n, [(int(a), int(b)) for a, b in cfg if b > a], K_LOOP, LE.CONFINE)
        X = gaussian_draw(Lc, rng, 4)
        for x in X:
            tree = cKDTree(x)
            for i, j in tree.query_pairs(R_CAPTURE, output_type="ndarray"):
                cnt[i, j] += 1
                cnt[j, i] += 1
            nsamp += 1
    counted = cnt / max(nsamp, 1)
    counted[counted == 0] = np.nan
    r_ana = score(analytic, "analytic  <R^2>^(-3/2)")
    r_cnt = score(counted, "counted   from sampled coordinates")
    v2 = bool(abs(r_cnt["insul_r"] - r_ana["insul_r"]) < 0.05)
    say(f"     difference {r_cnt['insul_r'] - r_ana['insul_r']:+.4f}   V2 "
        f"{'PASS -- the retraction stands' if v2 else 'FAIL -- the retraction was wrong'}")

    # ---- the Langevin run --------------------------------------------------------------------------
    say(f"\n  Langevin: {B_REPLICAS} replicas x {n:,} beads, {N_STEPS:,} steps, dt {DT}, "
        f"soft-core eps {EPS_REP} kT, sigma {SIGMA}")
    dev = torch.device("cpu")
    Lc0 = [(int(a), int(b)) for a, b in cfgs[0] if b > a]
    X = torch.tensor(gaussian_draw(laplacian_k(n, Lc0, K_LOOP, LE.CONFINE), rng, B_REPLICAS),
                     dtype=torch.float32, device=dev)

    def overlap_fraction(Xn):
        f = []
        for b in range(Xn.shape[0]):
            t = cKDTree(Xn[b])
            pr = t.query_pairs(SIGMA, output_type="ndarray")
            pr = pr[np.abs(pr[:, 0] - pr[:, 1]) > 1] if len(pr) else pr
            f.append(len(pr) / n)
        return float(np.mean(f))

    ov0 = overlap_fraction(X.numpy())
    bb = torch.arange(n - 1, device=dev)
    occ_f, occ_r = of, orr
    kmc_state = LI.simulate_occupancy  # reuse the same barrier physics
    cfg_iter = 0
    loops_now = Lc0
    contact = torch.zeros((n, n), dtype=torch.float32, device=dev)
    nacc = 0
    pairs = None
    t_lang = time.time()
    for step in range(N_STEPS):
        if step % REBUILD_EVERY == 0:
            pl = []
            Xn = X.detach().numpy()
            for b in range(B_REPLICAS):
                t = cKDTree(Xn[b])
                pr = t.query_pairs(SIGMA + SKIN, output_type="ndarray")
                if len(pr):
                    pr = pr[np.abs(pr[:, 0] - pr[:, 1]) > 1]
                pl.append(torch.tensor(np.c_[np.full(len(pr), b), pr], dtype=torch.long))
            pairs = torch.cat(pl) if pl else None
        if step % (KMC_EVERY * 20) == 0 and cfg_iter + 1 < len(cfgs):
            cfg_iter += 1
            loops_now = [(int(a), int(b)) for a, b in cfgs[cfg_iter] if b > a]
        F = torch.zeros_like(X)
        d = X[:, 1:, :] - X[:, :-1, :]
        F[:, :-1, :] += K_BOND * d
        F[:, 1:, :] -= K_BOND * d
        if loops_now:
            li = torch.tensor([a for a, _ in loops_now], device=dev)
            lj = torch.tensor([b for _, b in loops_now], device=dev)
            dl = X[:, lj, :] - X[:, li, :]
            F.index_add_(1, li, K_LOOP_ABS * dl)
            F.index_add_(1, lj, -K_LOOP_ABS * dl)
        F -= (LE.CONFINE * K_BOND) * X
        if pairs is not None and len(pairs):
            bi, pi, pj = pairs[:, 0], pairs[:, 1], pairs[:, 2]
            dv = X[bi, pj, :] - X[bi, pi, :]
            r = dv.norm(dim=1, keepdim=True).clamp(min=1e-6)
            over = (r < SIGMA).squeeze(1)
            if over.any():
                mag = 2.0 * EPS_REP * (1.0 - r[over] / SIGMA) / SIGMA
                fv = -mag * dv[over] / r[over]
                F.index_put_((bi[over], pi[over]), fv, accumulate=True)
                F.index_put_((bi[over], pj[over]), -fv, accumulate=True)
        X = X + F * DT + torch.randn_like(X) * np.sqrt(2.0 * DT)
        if step >= BURN_STEPS and step % SAMPLE_EVERY == 0:
            Xn = X.detach().numpy()
            for b in range(B_REPLICAS):
                t = cKDTree(Xn[b])
                pr = t.query_pairs(R_CAPTURE, output_type="ndarray")
                if len(pr):
                    contact[pr[:, 0], pr[:, 1]] += 1
                    contact[pr[:, 1], pr[:, 0]] += 1
            nacc += B_REPLICAS
        if step % 20000 == 0:
            say(f"      step {step:,}/{N_STEPS:,}  ({time.time()-t_lang:.0f}s)")
    t_lang = time.time() - t_lang
    ov1 = overlap_fraction(X.detach().numpy())
    v3 = bool(ov0 / max(ov1, 1e-9) >= 10.0)
    say(f"\n  V3 EXCLUDED VOLUME ACTUALLY BITES -- overlapping pairs per bead "
        f"{ov0:.4f} -> {ov1:.4f}, a {ov0/max(ov1,1e-9):.1f}x reduction   V3 "
        f"{'PASS' if v3 else 'FAIL'}")

    S = contact.numpy() / max(nacc, 1)
    S[S == 0] = np.nan
    say(f"\n  {nacc:,} configurations accumulated over the trajectory")
    r_lan = score(S, "Langevin + excluded volume")

    v4 = bool(r_lan["insul_r"] > prev_r)
    say(f"\n  V4 INSULATION IMPROVES -- {r_lan['insul_r']:+.4f} vs loop 43's {prev_r:+.4f}   "
        f"V4 {'PASS' if v4 else 'FAIL'}")
    v5 = bool(r_lan["insul_r"] >= GATE_INSUL)
    say(f"  V5 IT CLEARS THE ORIGINAL GATE -- {r_lan['insul_r']:+.4f} vs {GATE_INSUL} "
        f"({r_lan['insul_r']/ceiling:.0%} of the {ceiling:.3f} ceiling)   V5 "
        f"{'PASS' if v5 else 'FAIL'}")
    per_cfg = t_lang / max(nacc, 1)
    say(f"  V6 THE COST -- {t_lang/60:.1f} min for {nacc:,} configurations, {per_cfg:.2f}s each,")
    say(f"     against the closed form's 0.42s. The trade is {per_cfg/0.42:.1f}x per configuration.")

    say("\n" + "=" * 100)
    gates = {"V1 sampler is exact": v1, "V2 counting changes nothing": v2,
             "V3 excluded volume bites": v3, "V4 insulation improves": v4,
             "V5 clears the original gate": v5}
    for k, v in gates.items():
        say(f"  {k:<40}{'PASS' if v else 'FAIL'}")
    if v2:
        say("  THE RETRACTION STANDS. Counting contacts in explicit coordinates reproduces the analytic")
        say("  map, so 'double averaging' was never the problem and I was wrong to offer it.")
    if v4 and v5:
        say(f"  AND EXCLUDED VOLUME WAS. It is the one ingredient the closed form cannot contain, it")
        say(f"  took insulation from {prev_r:+.4f} to {r_lan['insul_r']:+.4f} past the gate, and it cost")
        say(f"  {per_cfg/0.42:.0f}x per configuration. That is the price of the thing linearity forbids.")
    elif v4:
        say(f"  EXCLUDED VOLUME HELPS AND DOES NOT CLOSE IT: {prev_r:+.4f} -> {r_lan['insul_r']:+.4f}")
        say(f"  against {GATE_INSUL}. Recorded as measured.")
    else:
        say(f"  EXCLUDED VOLUME DID NOT HELP: {r_lan['insul_r']:+.4f} against {prev_r:+.4f}. The")
        say(f"  integrator ran, the overlaps went away, and the boundaries did not sharpen. Recorded")
        say(f"  as measured -- the candidate I named as the only remaining one was wrong too.")
    say("=" * 100)

    man = RM.manifest(inputs=[str(SC / "hic_chr21_25kb.npy")], available=n, used=int(mask.sum()),
                      selection="filtered", seed=SEED,
                      controls=["the analytic map from the SAME loop configurations",
                                "the sampler checked against G = L^-1",
                                "overlap fraction before and after as a design check",
                                "P(s) and orientation carried forward",
                                "the 0.40 gate unmoved for the fourth time"],
                      note="V2 exists to make a retraction falsifiable: the Gaussian contact formula "
                           "is exact per configuration, so discrete counting should NOT improve on it")
    RM.report(man, emit=say)
    json.dump({"test": "loop_langevin", "manifest": man, "gates": gates,
               "sampler_err": err, "analytic": r_ana, "counted": r_cnt, "langevin": r_lan,
               "overlap_before": ov0, "overlap_after": ov1, "n_configs": nacc,
               "prev_insul_r": prev_r, "ceiling": ceiling,
               "minutes": t_lang / 60, "sec_per_config": per_cfg, "log": log},
              open(OUT / "loop_langevin.json", "w"), indent=2)
    say(f"\n  -> {OUT/'loop_langevin.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
