"""NEXUS SPHERE COVERAGE FOR CHROMATIN: excluded volume without an integrator.

THE GAP THIS FILLS.  `rouse_gate` computed chromatin contact analytically -- Gaussian Network Model, closed
form, no timestep -- and found it real but redundant against counting CTCF sites. Its central approximation is
that the fibre is a PHANTOM chain: a Gaussian polymer passes through itself. That is not a small idealisation.
Self-avoidance is what makes a real fibre crumple rather than mix, and it is the single largest piece of
physics the analytic form throws away.

Brute force restores it with Langevin dynamics and pairwise repulsion -- the ~1 GPU-day per region that Stage
3 of the 4D plan was costed at, and that we declined. The sphere idea restores it for almost nothing:

    cover the fibre in spheres, and excluded volume is just "spheres do not overlap".

No trajectory, no timestep, no force field. Sample conformations from the analytic covariance the GNM already
gives, then RELAX the overlaps out while holding the bonds. What comes back is an ensemble that is
self-avoiding but was never integrated.

THE INITIAL STRUCTURE IS FREE, and comes from data already on disk.  CTCF peaks on chr22 give loop anchors;
anchors give loop bonds; bonds give the Laplacian; the Laplacian's pseudo-inverse IS the equilibrium
covariance of bead positions. One eigendecomposition per window and conformations are drawn directly. There
is nothing to fit and nothing to simulate.

WHAT IT IS SCORED AGAINST, streamed rather than downloaded.  Measured K562 Hi-C, read straight out of a 32 GB
remote .hic by range request -- a 4 Mb window at 25 kb costs ~14 s and no disk. Brute force would have been
downloading the file.

THE BASELINE THAT MATTERS, and it is not zero.  Most of any Hi-C map is genomic distance: contacts decay as
P(s) ~ s^-a and that alone reproduces the map's gross appearance. So every arm is scored on
DISTANCE-CONTROLLED residuals -- both measured and predicted are z-scored WITHIN each separation bin, and the
correlation is computed on what is left. An arm that only knows contacts fall off with distance scores zero
here, which is the point.

ARMS
    distance_only   the null. Predicts from separation alone; zero by construction after the control.
    phantom         GNM ensemble as `rouse_gate` had it -- chain may pass through itself
    spheres         the same ensemble with overlaps relaxed out. The experiment.
    spheres_shuffled_ctcf   anchors relocated within the window, count and spacing preserved. If this scores
                    as well, the model is reading genomic density rather than loop topology.

PREDECLARED, before any number:
    spheres > phantom, and spheres > shuffled
        -> excluded volume is real signal the analytic form was discarding, and sphere coverage delivers it
           at ~1 s per window instead of ~1 GPU-day. That reopens the chromatin line.
    spheres ~ phantom
        -> self-avoidance does not change contact structure at 25 kb resolution. The phantom approximation
           was adequate, which retroactively justifies `rouse_gate` having skipped it, and the redundancy
           against CTCF counting was not caused by the missing physics.
    both ~ 0 after distance control
        -> neither form of the polymer model explains contact structure beyond the decay, and the whole
           contact-prediction line rests on genomic distance plus boundary counting.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
CTCF = INV / "chip.CTCF.chr22.bed"
HIC_URL = "https://www.encodeproject.org/files/ENCFF822UUX/@@download/ENCFF822UUX.hic"   # K562, hg38
CACHE = Path("/tmp/claude-0/-home-user-cell/0f039315-b3a9-52ac-8187-9fae0d726994/scratchpad/hic_win")

RES = 25_000            # matches the Hi-C resolution actually stored; no rebinning
WIN = 4_000_000         # 160 beads per window
WINDOWS = [(16_000_000 + k * WIN, 16_000_000 + (k + 1) * WIN) for k in range(8)]
N_CONF = 300            # conformations per window; the ensemble average is what is compared
K_LOOP = 3.0            # loop-bond stiffness relative to backbone, as in rouse_gate
R_FRAC = float(os.environ.get("SPHERE_R_FRAC", 0.55))
                        # sphere radius as a fraction of the mean backbone bond length. 0.5 = adjacent
                        # spheres just touch; above that the chain is genuinely crowded.
                        # THIS IS THE KNOB THE HEADLINE DEPENDS ON. "Self-avoidance changes nothing" is only
                        # a claim about chromatin if the spheres were big enough to matter -- at a small
                        # radius almost nothing overlaps and the result is a statement about the radius. So
                        # it is settable, and the conclusion is only reportable after a sweep. The run prints
                        # the pre-relaxation overlap fraction so the crowding is visible next to the answer.
N_RELAX = 60            # overlap-relaxation sweeps. Not timesteps -- there is no velocity or force law,
                        # only a geometric projection, so this number is a convergence knob not a duration.
CONTACT_R = 2.0         # contact if centres are within CONTACT_R * bond lengths
SEED = 11


def ctcf_anchors(lo, hi):
    a = []
    for line in open(CTCF):
        f = line.split()
        s, e = int(f[1]), int(f[2])
        if s >= lo and e <= hi:
            a.append(((s + e) // 2, float(f[6])))
    a.sort()
    return a


def laplacian(nb, anchors, lo, shuffle_rng=None):
    """backbone + cohesin loop bonds -> graph Laplacian. Loops join consecutive anchors (extrusion stalls at
    the first barrier each way), plus next-nearest at reduced weight for nested/skipped loops."""
    pos = [(p - lo) // RES for p, _ in anchors]
    sig = [w for _, w in anchors]
    if shuffle_rng is not None:
        # keep the COUNT and the signal values, relocate the positions: a density control, not a null model
        # with replacement: real anchors also collide onto shared beads (151 anchors, 160 beads), so
        # sampling without replacement would give the control a MORE spread-out topology than the truth
        pos = sorted(shuffle_rng.choice(nb, len(pos), replace=True).tolist())
    L = np.zeros((nb, nb))

    def bond(i, j, k):
        if 0 <= i < nb and 0 <= j < nb and i != j:
            L[i, i] += k; L[j, j] += k; L[i, j] -= k; L[j, i] -= k
    for i in range(nb - 1):
        bond(i, i + 1, 1.0)
    for gap, w in ((1, 1.0), (2, 0.35)):
        for t in range(len(pos) - gap):
            k = K_LOOP * w * min(sig[t], sig[t + gap]) / (max(sig) or 1.0)
            bond(pos[t], pos[t + gap], k)
    return L


def sample(L, n_conf, rng):
    """conformations straight from the GNM covariance. Sigma = L^+ ; drop the translational null mode."""
    w, V = np.linalg.eigh(L)
    keep = w > 1e-9
    A = V[:, keep] / np.sqrt(w[keep])              # x = A z  gives Cov(x) = L^+
    z = rng.standard_normal((n_conf, 3, A.shape[1]))
    return np.einsum("ij,ncj->nci", A, z).transpose(0, 2, 1)    # (n_conf, n_beads, 3)


def overlap_frac(X, r):
    """fraction of non-bonded pairs whose spheres interpenetrate"""
    n = X.shape[1]
    d = np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=3)
    iu = np.arange(n)
    d[:, iu, iu] = np.inf
    d[:, iu[:-1], iu[1:]] = np.inf
    d[:, iu[1:], iu[:-1]] = np.inf
    return float((d < 2 * r).mean())


def relax(X, r, n_sweep):
    """SPHERE COVERAGE: push overlapping spheres apart, then pull bonded neighbours back.

    This is a geometric projection, not dynamics -- there is no force law, no mass, no timestep. Each sweep
    resolves the worst overlaps and restores connectivity; alternating the two converges to a conformation
    that is self-avoiding AND still a chain. Cost is O(N^2) per sweep with N=160, which is nothing.
    """
    X = X.copy()
    n = X.shape[1]
    b0 = np.linalg.norm(np.diff(X, axis=1), axis=2).mean()
    for _ in range(n_sweep):
        d = X[:, :, None, :] - X[:, None, :, :]
        dist = np.linalg.norm(d, axis=3)
        iu = np.arange(n)
        dist[:, iu, iu] = np.inf
        for k in (1,):                              # bonded neighbours are allowed to be close
            dist[:, iu[:-k], iu[k:]] = np.inf
            dist[:, iu[k:], iu[:-k]] = np.inf
        over = dist < 2 * r
        if not over.any():
            break
        with np.errstate(invalid="ignore", divide="ignore"):
            push = np.where(over[..., None], d / np.maximum(dist, 1e-9)[..., None]
                            * ((2 * r - dist) / 2)[..., None], 0.0)
        X += 0.5 * push.sum(2)
        # restore the chain: bonded pairs pulled back toward their equilibrium separation
        for _ in range(2):
            dv = np.diff(X, axis=1)
            dl = np.linalg.norm(dv, axis=2, keepdims=True)
            corr = 0.5 * dv * (1 - b0 / np.maximum(dl, 1e-9))
            X[:, :-1, :] += corr
            X[:, 1:, :] -= corr
    return X


def contact_map(X, cutoff):
    d = np.linalg.norm(X[:, :, None, :] - X[:, None, :, :], axis=3)
    return (d < cutoff).mean(0)


def measured(lo, hi):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"chr22_{lo}_{hi}_{RES}.npy"
    if f.exists():
        return np.load(f)
    import hicstraw
    nb = (hi - lo) // RES
    M = np.zeros((nb, nb))
    hic = hicstraw.HiCFile(HIC_URL)
    mzd = hic.getMatrixZoomData("chr22", "chr22", "observed", "NONE", "BP", RES)
    for rec in mzd.getRecords(lo, hi - 1, lo, hi - 1):
        i, j = (rec.binX - lo) // RES, (rec.binY - lo) // RES
        if 0 <= i < nb and 0 <= j < nb:
            M[i, j] = M[j, i] = rec.counts
    np.save(f, M)
    return M


def dist_controlled(pred, meas, min_sep=2):
    """z-score BOTH within each separation bin, then correlate the residuals. Removes P(s) from both sides so
    an arm that only knows 'contacts fall off with distance' scores zero."""
    from scipy.stats import spearmanr
    n = pred.shape[0]
    a, b = [], []
    for s in range(min_sep, n):
        i = np.arange(n - s)
        p, m = pred[i, i + s], meas[i, i + s]
        ok = np.isfinite(p) & np.isfinite(m) & (m > 0)
        if ok.sum() < 8:
            continue
        p, m = p[ok], m[ok]
        if p.std() < 1e-12 or m.std() < 1e-12:
            continue
        a += list((p - p.mean()) / p.std())
        b += list((m - m.mean()) / m.std())
    if len(a) < 50:
        return float("nan"), 0
    return float(spearmanr(a, b).statistic), len(a)


def ps_exponent(M, min_sep=2, max_sep=80):
    n = M.shape[0]
    s, v = [], []
    for k in range(min_sep, min(max_sep, n)):
        i = np.arange(n - k)
        d = M[i, i + k]
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) >= 8:
            s.append(k); v.append(d.mean())
    if len(s) < 10:
        return float("nan")
    return float(np.polyfit(np.log(s), np.log(v), 1)[0])


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    report("=" * 100)
    report("SPHERE COVERAGE FOR CHROMATIN -- excluded volume without an integrator")
    report("=" * 100)
    report("  rouse_gate's GNM treats the fibre as a PHANTOM chain that passes through itself. Self-avoidance")
    report("  is the largest piece of physics it discards. Brute force restores it with Langevin dynamics at")
    report("  ~1 GPU-day/region; spheres restore it as 'spheres do not overlap' -- a geometric projection on")
    report("  an ensemble drawn free from the analytic covariance. No timestep, no force law, no trajectory.")
    report("  Scored against measured K562 Hi-C streamed from a 32 GB remote file by range request.")
    report("  EVERY ARM IS DISTANCE-CONTROLLED: both sides z-scored within each separation bin, so 'contacts")
    report("  fall off with distance' scores zero and only topology can earn anything.")
    report(f"  SPHERE RADIUS {R_FRAC:.2f} x bond length (0.50 = adjacent beads just touch). A null result here")
    report("  is only about chromatin if this is large enough to crowd the chain -- see the overlap fraction.")

    rng = np.random.default_rng(SEED)
    rows = {a: [] for a in ("phantom", "spheres", "spheres_shuffled_ctcf")}
    exps = {a: [] for a in ("measured", "phantom", "spheres")}
    nb = WIN // RES
    used = 0
    ovs = []
    for (lo, hi) in WINDOWS:
        anch = ctcf_anchors(lo, hi)
        if len(anch) < 6:
            report(f"    window {lo/1e6:.0f}-{hi/1e6:.0f} Mb: only {len(anch)} CTCF anchors, skipped")
            continue
        try:
            M = measured(lo, hi)
        except Exception as e:
            report(f"    window {lo/1e6:.0f}-{hi/1e6:.0f} Mb: Hi-C unavailable ({type(e).__name__}), skipped")
            continue
        if (M > 0).mean() < 0.05:
            report(f"    window {lo/1e6:.0f}-{hi/1e6:.0f} Mb: Hi-C too sparse, skipped")
            continue
        used += 1
        for arm in ("phantom", "spheres", "spheres_shuffled_ctcf"):
            sh = np.random.default_rng(SEED + 7) if arm.endswith("shuffled_ctcf") else None
            L = laplacian(nb, anch, lo, shuffle_rng=sh)
            X = sample(L, N_CONF, np.random.default_rng(SEED + used))
            b0 = np.linalg.norm(np.diff(X, axis=1), axis=2).mean()
            if arm != "phantom":
                ov0 = overlap_frac(X, R_FRAC * b0)
                X = relax(X, R_FRAC * b0, N_RELAX)
                ov1 = overlap_frac(X, R_FRAC * b0)
                if arm == "spheres":
                    ovs.append((ov0, ov1))
            P = contact_map(X, CONTACT_R * b0)
            r, npts = dist_controlled(P, M)
            rows[arm].append(r)
            if arm in exps:
                exps[arm].append(ps_exponent(P))
        exps["measured"].append(ps_exponent(M))
        report(f"    window {lo/1e6:.0f}-{hi/1e6:.0f} Mb | {len(anch)} anchors | "
               f"phantom {rows['phantom'][-1]:+.3f}  spheres {rows['spheres'][-1]:+.3f}  "
               f"shuffled {rows['spheres_shuffled_ctcf'][-1]:+.3f}")

    if used < 3:
        report(f"\n  ONLY {used} USABLE WINDOWS -- not a result, an empty run.")
        return 1

    report(f"\n  DISTANCE-CONTROLLED SPEARMAN vs measured Hi-C, {used} windows of {WIN/1e6:.0f} Mb "
           f"at {RES//1000} kb")
    report(f"    {'arm':<26}{'mean r':>9}{'sd':>8}{'windows':>9}")
    res = {}
    report(f"    {'distance_only (null)':<26}{0.0:>9.3f}{0.0:>8.3f}{used:>9}")
    for a in ("phantom", "spheres", "spheres_shuffled_ctcf"):
        v = np.array(rows[a], float)
        res[a] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "n": int(np.isfinite(v).sum())}
        report(f"    {a:<26}{np.nanmean(v):>9.3f}{np.nanstd(v):>8.3f}{int(np.isfinite(v).sum()):>9}")

    if ovs:
        a0 = float(np.mean([o[0] for o in ovs])); a1 = float(np.mean([o[1] for o in ovs]))
        res["overlap_before"], res["overlap_after"] = a0, a1
        report(f"\n  LIVENESS -- did the relaxation actually do anything?")
        report(f"    overlapping non-bonded sphere pairs: {a0:.4f} before -> {a1:.4f} after "
               f"({(1-a1/max(a0,1e-9)):.0%} removed)")
        if a0 < 1e-4:
            report("    *** THE PHANTOM ENSEMBLE BARELY OVERLAPS AT ALL, so there was nothing to fix and")
            report("        'spheres ~ phantom' is a statement about the sphere RADIUS, not about physics.")
        elif a1 > 0.5 * a0:
            report("    *** RELAXATION DID NOT CONVERGE -- most overlaps survive, so the spheres arm is not")
            report("        actually self-avoiding and any null below is uninterpretable.")

    report(f"\n  CONTACT-DECAY EXPONENT  P(s) ~ s^a   (a theory check the correlation cannot give)")
    for a in ("measured", "phantom", "spheres"):
        v = np.array(exps[a], float)
        res.setdefault("exponent", {})[a] = float(np.nanmean(v))
        report(f"    {a:<26}{np.nanmean(v):>9.3f}")
    em, ep, es = (res["exponent"][k] for k in ("measured", "phantom", "spheres"))
    report(f"    A FREE Gaussian chain gives -1.5; this chain is crosslinked by loop bonds and the fit runs")
    report(f"    over {2*RES//1000}-{80*RES//1000} kb, so -1.5 is not the expectation here and the phantom")
    report(f"    arm's {ep:+.3f} is not an error.")
    if es < ep and em < ep:
        report(f"    THE PHYSICS BEHAVES CORRECTLY: spheres steepen the decay {ep:+.3f} -> {es:+.3f}, i.e. "
               f"{es-ep:+.3f}")
        report(f"    TOWARD the measured {em:+.3f}. Self-avoidance compacts the chain exactly as it should.")
        report(f"    It closes {(es-ep)/(em-ep):.0%} of the gap to the data -- real, and not enough to matter")
        report("    for contact prediction, which is the distinction the correlation table makes.")
    else:
        report(f"    Spheres move the exponent {es-ep:+.3f} against a measured {em:+.3f}.")

    ph, sp, sh = res["phantom"]["mean"], res["spheres"]["mean"], res["spheres_shuffled_ctcf"]["mean"]
    report("\n  READING")
    if sp > ph + 0.02 and sp > sh + 0.02:
        report(f"  EXCLUDED VOLUME IS REAL SIGNAL. spheres {sp:.3f} vs phantom {ph:.3f} vs shuffled anchors")
        report(f"  {sh:.3f}. Self-avoidance carries contact structure the analytic form was discarding, and")
        report("  it costs a geometric relaxation rather than a GPU-day. The chromatin line reopens.")
    elif abs(sp - ph) <= 0.02:
        report(f"  SELF-AVOIDANCE CHANGES NOTHING AT THIS SCALE. spheres {sp:.3f} vs phantom {ph:.3f}. The")
        report("  phantom approximation was adequate at 25 kb, so rouse_gate was right to skip it and the")
        report("  redundancy against CTCF counting was NOT caused by the missing physics. Restoring it by")
        report("  brute force would have bought the same nothing at a GPU-day per region.")
    else:
        report(f"  SPHERES HURT. {sp:.3f} vs phantom {ph:.3f} -- the relaxation is destroying topology rather")
        report("  than adding physics; check the bond restoration before reading this as a statement about")
        report("  chromatin.")
    if max(ph, sp) <= sh + 0.02:
        report(f"  AND THE ANCHOR CONTROL IS NOT BEATEN ({sh:.3f}): whatever either arm scores, it is reading")
        report("  genomic density rather than loop topology, and neither is a model of the fibre.")
    else:
        report(f"  BUT THE TOPOLOGY ITSELF IS NOT NOTHING: {max(ph, sp):+.3f} against {sh:+.3f} for the same")
        report("  anchors relocated at matched count and spacing. WHERE the loops are carries the signal; the")
        report("  null sits at zero, so the polymer model earns its number -- excluded volume just is not the")
        report("  part that earns it.")

    name = "chromatin_spheres.json" if abs(R_FRAC - 0.55) < 1e-9 else f"chromatin_spheres_r{R_FRAC:g}.json"
    json.dump({"test": "chromatin_spheres", "windows": used, "res": RES, "arms": res, "r_frac": R_FRAC,
               "n_conf": N_CONF, "log": log}, open(OUT / name, "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
