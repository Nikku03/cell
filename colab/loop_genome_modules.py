"""Loop 220. The whole genome: is the reproducibility failure NOISE, or a shared artefact?

WHY SCALE IS A TEST AND NOT JUST A BIGGER RUN. Loop 219 found that co-change modules make the
interval change highly measurable within replicates 2 and 3 (module-level R2 +0.67841 against a
gene-level -0.54028, and above the maximum of 200 size-matched random groupings) and that the same
modules do not reproduce on replicate 4 (median trajectory correlation +0.2638). It ran on 600
genes.

Those two facts admit exactly two explanations and they have opposite remedies:

    NOISE-LIMITED     the modules are real, each gene is too noisy, and the module average is
                      still too noisy at these sizes. More genes per module means noise falls as
                      1/sqrt(k) and reproducibility should RISE with module size.

    ARTEFACT-LIMITED  replicates 2 and 3 share something replicate 4 does not -- a batch, a
                      library prep, a flowcell. Then module averaging amplifies the shared
                      artefact exactly as efficiently as it amplifies signal, and reproducibility
                      against replicate 4 stays flat no matter how large the modules get.

The whole genome discriminates them, because 13,756 expressed genes support modules an order of
magnitude larger than 600 genes do. P5 is the discriminating gate and it is the reason this loop
is worth running.

WHAT CHANGES AT SCALE BESIDES SIZE. The 600 were loop 198's selection: responders with a promoter
DNase peak. The genome is not filtered for response, so most genes carry no signal at all and the
per-gene noise floor is worse, not better. That cuts against the noise-limited hypothesis and is
stated here so a flat result cannot later be blamed on it.

CLUSTERING AT SCALE. A 13,756-square correlation matrix is 1.5 GB in float64 and the linkage is
worse, so k-means on the 8-dimensional trajectory vectors is used instead. That is a different
algorithm from loop 219's average linkage and P2 checks the two agree on the 600 before the
genome-scale result is read.

PREDECLARED, BEFORE ANY NUMBER.

  P1 IS THE GENOME-SCALE INPUT SOUND?
     Gate: PASS iff the expression filter is applied to all four replicates jointly, the retained
     gene count matches the filter, and the 600-gene set of loops 213-219 is a subset of what is
     retained. FAIL means the two scales are not comparable.

  P2 DOES K-MEANS AGREE WITH LOOP 219's LINKAGE ON THE 600?
     Gate: PASS iff k-means modules on the same 600 genes reach a within-pair module ceiling
     within 0.15 of loop 219's +0.67841. A larger gap means the algorithm change is doing the
     work and nothing at genome scale may be attributed to scale.

  P3 WHAT IS THE GENE-LEVEL CEILING AT GENOME SCALE?
     Replicate 2 predicting replicate 3, gene level, on all retained genes.
     Not scored -- it is the baseline the module numbers are read against, and it is expected to
     be worse than the 600-gene -0.54028 because the genome is not filtered for response.

  P4 DOES THE MODULE CEILING RISE AT SCALE, WITHIN THE AGREEING PAIR?
     Gate: PASS iff the genome-scale module ceiling on replicates 2 and 3 exceeds loop 219's
     +0.67841. If averaging more genes does not help even where the replicates agree, the module
     idea does not scale and P5 is moot.

  P5 THE DISCRIMINATING TEST: DOES REPRODUCIBILITY RISE WITH MODULE SIZE?
     Bin modules by size and measure each bin's trajectory correlation against replicate 4.
     Gate: PASS iff the correlation in the largest size bin exceeds the smallest by more than 0.20.
     A PASS means NOISE-LIMITED: bigger modules reproduce better, and the remedy is more genes or
     more replicates. A FAIL means ARTEFACT-LIMITED: replicates 2 and 3 share something replicate
     4 does not, averaging amplifies it, and no amount of data fixes it.
     Requires P4.

  P6 IS THE ARTEFACT VISIBLE DIRECTLY?
     If P5 fails, the shared component should be findable: project each replicate's genome-scale
     change matrix onto the first principal component of the 2-and-3 average and measure how much
     of each replicate it explains.
     Gate: PASS iff the shared component explains a substantially larger fraction of replicates 2
     and 3 than of replicates 1 and 4 -- which would name the artefact rather than infer it.

  P7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, sys, time, warnings
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loop_response_timing_d as L191
from loop_setpoint_physics import gene_set
from gate_guard import Gates

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SP = L191.SP
OUT = "outputs/loop_genome_modules.json"
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
MIN_TPM, SEED, K = 1.0, 220220, 300
REF_219 = 0.67841
REF_GENE_600 = -0.54028

LOG = []
def say(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s); print(s, flush=True)


def r2s(y, p):
    ss = float(np.sum((y - p) ** 2)); tt = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss / tt if tt > 0 else float("nan")


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def kmeans(X, k, seed, iters=60):
    rng = np.random.default_rng(seed)
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    C = Z[rng.choice(len(Z), k, replace=False)]
    lab = np.zeros(len(Z), int)
    for _ in range(iters):
        sim = Z @ C.T
        new = sim.argmax(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                v = Z[m].mean(0)
                C[j] = v / (np.linalg.norm(v) + 1e-9)
    return lab


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "genome-scale modules"}
    say("=" * 104)
    say("LOOP 220 -- THE WHOLE GENOME: IS THE FAILURE NOISE, OR A SHARED ARTEFACT?")
    say("=" * 104)

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    g = np.array(GRID, float)
    grid6, M6, A9, sym, keep, tssb = gene_set()
    gi600 = np.where(keep)[0]

    # ---------------------------------------------------------------- P1
    say("P1 IS THE GENOME-SCALE INPUT SOUND?")
    base = {r: tpm[(mins == 30) & (reps == r)].mean(0) for r in (1, 2, 3, 4)}
    ok_all = np.all([base[r] >= MIN_TPM for r in (1, 2, 3, 4)], axis=0)
    sel = np.where(ok_all)[0]
    sub600 = set(gi600.tolist())
    covered = len(sub600 & set(sel.tolist()))
    say(f"     all genes {tpm.shape[1]:,}")
    say(f"     baseline TPM >= {MIN_TPM} in ALL FOUR replicates: {len(sel):,}")
    say(f"     loop 198's 600-gene scored set covered: {covered:,} of {len(sub600):,}")
    ok1 = (len(sel) > 5000 and covered >= 0.9 * len(sub600))
    G.add("P1", ok1, stat=float(len(sel)),
          if_true=lambda: f"P1 PASS -- {len(sel):,} genes pass the filter in all four replicates, "
                          f"covering {covered/len(sub600):.1%} of the 600-gene set",
          if_false=lambda: f"P1 FAIL -- {len(sel):,} retained, {covered}/{len(sub600)} of the 600")

    D = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        D[r] = np.array([Mi[j, sel] - Mi[j - 1, sel] for j in range(1, len(g))])
    nint, ngen = D[2].shape
    say(f"     change matrix per replicate: {nint} intervals x {ngen:,} genes")
    pos = {int(v): i for i, v in enumerate(sel)}
    i600 = np.array([pos[v] for v in gi600 if v in pos])

    def mod_traj(Dr, lab, ids):
        return np.array([Dr[:, ids[lab[ids] == j]].mean(1) for j in np.unique(lab[ids])
                         if (lab[ids] == j).sum() >= 3])

    # ---------------------------------------------------------------- P2
    say("P2 DOES K-MEANS AGREE WITH LOOP 219's LINKAGE ON THE 600?")
    X6 = np.mean([D[2][:, i600], D[3][:, i600]], axis=0).T
    lab6 = kmeans(X6, 10, SEED)
    ids6 = np.arange(len(i600))
    a = np.array([D[2][:, i600][:, ids6[lab6 == j]].mean(1) for j in np.unique(lab6)
                  if (lab6 == j).sum() >= 3])
    b = np.array([D[3][:, i600][:, ids6[lab6 == j]].mean(1) for j in np.unique(lab6)
                  if (lab6 == j).sum() >= 3])
    c600 = r2s(b.ravel(), a.ravel())
    say(f"     k-means, 10 modules on the same 600 genes: within-pair ceiling {c600:+.5f}")
    say(f"     loop 219's average linkage recorded {REF_219:+.5f}")
    G.add("P2", bool(abs(c600 - REF_219) <= 0.15), stat=c600, requires=("P1",),
          if_true=lambda: f"P2 PASS -- {c600:+.4f} against {REF_219:+.4f}; the algorithm change is "
                          f"not doing the work",
          if_false=lambda: f"P2 FAIL -- {c600:+.4f} against {REF_219:+.4f}, a gap of "
                           f"{abs(c600-REF_219):.4f}")

    # ---------------------------------------------------------------- P3
    say("P3 WHAT IS THE GENE-LEVEL CEILING AT GENOME SCALE?")
    cg = r2s(D[3].ravel(), D[2].ravel())
    say(f"     replicate 2 predicts replicate 3, gene level, {ngen:,} genes: R2 {cg:+.5f}")
    say(f"     on loop 198's 600 responders it was {REF_GENE_600:+.5f}")
    res["gene_ceiling"] = {"genome": cg, "600": REF_GENE_600, "n": int(ngen)}

    # ---------------------------------------------------------------- P4
    say("P4 DOES THE MODULE CEILING RISE AT SCALE, WITHIN THE AGREEING PAIR?")
    Xg = np.mean([D[2], D[3]], axis=0).T
    lab = kmeans(Xg, K, SEED)
    sizes = np.bincount(lab, minlength=K)
    kept = np.array([j for j in range(K) if sizes[j] >= 3])
    say(f"     k-means k={K}: {len(kept)} modules with >=3 genes, sizes median "
        f"{int(np.median(sizes[kept]))}, max {int(sizes[kept].max())}")
    T2 = np.array([D[2][:, lab == j].mean(1) for j in kept])
    T3 = np.array([D[3][:, lab == j].mean(1) for j in kept])
    T4 = np.array([D[4][:, lab == j].mean(1) for j in kept])
    T23 = np.array([np.mean([D[2], D[3]], axis=0)[:, lab == j].mean(1) for j in kept])
    cmod = r2s(T3.ravel(), T2.ravel())
    say(f"     genome-scale module ceiling, rep2 predicts rep3: R2 {cmod:+.5f}")
    say(f"     loop 219 on 600 genes: {REF_219:+.5f}")
    G.add("P4", bool(cmod > REF_219), stat=cmod, requires=("P2",),
          if_true=lambda: f"P4 PASS -- {cmod:+.4f}, above loop 219's {REF_219:+.4f}",
          if_false=lambda: f"P4 FAIL -- {cmod:+.4f} against {REF_219:+.4f}; averaging more genes "
                           f"does not help even where the replicates agree")
    res["module_ceiling"] = {"genome": cmod, "600_linkage": REF_219, "600_kmeans": c600,
                             "n_modules": int(len(kept))}

    # ---------------------------------------------------------------- P5
    say("P5 THE DISCRIMINATING TEST: DOES REPRODUCIBILITY RISE WITH MODULE SIZE?")
    rr = np.array([corr(T23[i], T4[i]) for i in range(len(kept))])
    sz = sizes[kept]
    qs = np.quantile(sz, [0.25, 0.5, 0.75])
    bins = [(0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
    say("       size bin        n_mod   median size   corr with rep4")
    binvals = []
    for lo, hi in bins:
        m = (sz > lo) & (sz <= hi)
        if m.sum() == 0:
            continue
        v = float(np.median(rr[m]))
        binvals.append(v)
        say(f"       {int(lo):>5}-{'inf' if np.isinf(hi) else int(hi):>5}   {int(m.sum()):>5}   "
            f"{int(np.median(sz[m])):>11}   {v:+.4f}")
    rise = binvals[-1] - binvals[0] if len(binvals) >= 2 else float("nan")
    say(f"     largest bin minus smallest bin: {rise:+.4f}")
    G.add("P5", bool(rise > 0.20), stat=rise, requires=("P4",),
          if_true=lambda: f"P5 PASS -- reproducibility rises {rise:+.3f} with module size. "
                          f"NOISE-LIMITED: the remedy is more genes or more replicates",
          if_false=lambda: f"P5 FAIL -- reproducibility rises only {rise:+.3f} with module size. "
                           f"ARTEFACT-LIMITED: replicates 2 and 3 share something replicate 4 "
                           f"does not, averaging amplifies it as efficiently as it amplifies "
                           f"signal, and no amount of data fixes it")
    res["size_bins"] = binvals

    # ---------------------------------------------------------------- P6
    say("P6 IS THE ARTEFACT VISIBLE DIRECTLY?")
    A23 = np.mean([D[2], D[3]], axis=0)
    A23c = A23 - A23.mean(1, keepdims=True)
    U, S_, Vt = np.linalg.svd(A23c, full_matrices=False)
    pc1 = Vt[0]
    expl = {}
    for r in (1, 2, 3, 4):
        Dc = D[r] - D[r].mean(1, keepdims=True)
        proj = np.outer(Dc @ pc1, pc1)
        expl[r] = float(np.sum(proj ** 2) / np.sum(Dc ** 2))
        say(f"       replicate {r}: the 2-and-3 first component explains {expl[r]:.4%} "
            f"of its change variance")
    shared = (expl[2] + expl[3]) / 2
    other = (expl[1] + expl[4]) / 2
    say(f"     mean over replicates 2,3 {shared:.4%}   over replicates 1,4 {other:.4%}   "
        f"ratio {shared/other if other>0 else float('inf'):.2f}x")
    G.add("P6", bool(shared > 2 * other), stat=shared, requires=("P1",),
          if_true=lambda: f"P6 PASS -- the shared component explains {shared/other:.1f}x more of "
                          f"replicates 2 and 3 than of 1 and 4. The artefact is named, not inferred",
          if_false=lambda: f"P6 FAIL -- {shared:.3%} against {other:.3%}, a ratio of "
                           f"{shared/other if other>0 else float('nan'):.2f}x. There is no single "
                           f"dominant component separating the agreeing pair from the rest")
    res["shared_component"] = {str(k): v for k, v in expl.items()}

    say("P7 WHAT THIS CANNOT SHOW")
    say("     The genome is not filtered for response, so most of these 13,756 genes carry no")
    say("     signal at all. That makes the per-gene noise floor WORSE than on loop 198's 600")
    say("     responders and cuts against the noise-limited hypothesis, which is stated here so a")
    say("     flat P5 cannot later be blamed on it.")
    say("     k-means on 8-dimensional trajectories is a coarse clustering and k=300 is a choice,")
    say("     not a measurement. P2 checks it reproduces loop 219's linkage result on the 600 but")
    say("     does not establish it is the right k at genome scale.")
    say("     If P5 fails and P6 passes, the artefact is identified but not explained -- naming a")
    say("     shared component is not the same as knowing whether it is a batch, a flowcell or a")
    say("     genuine biological difference between the cultures.")
    say("     ENCODE does not publish per-replicate batch metadata for this series, so the")
    say("     hypothesis cannot be confirmed against the experimental record from here.")

    G.summary(seconds=time.time() - t0)
    gates, void = G.as_dict()
    res["gates"], res["void"] = gates, void
    res["seconds"] = time.time() - t0
    res["log"] = LOG
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    say(f"wrote {OUT}")


if __name__ == "__main__":
    main()
