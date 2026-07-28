"""STAGE 2 + STAGE 3: KMC loop extrusion and batched GPU Langevin dynamics, scored against the 0.663 bar.

WHY THIS EXISTS DESPITE THE ANALYTIC RESULT. The closed-form Rouse model said the polymer physics is real
(+0.0131 over shuffled anchors) and redundant (+0.005 over counting CTCF peaks). That is a statement about the
GAUSSIAN equilibrium. It is not a statement about what a real simulation gives, because three things the
analytic form cannot represent might matter:

    excluded volume    the Gaussian chain passes through itself; a real one does not
    non-equilibrium    cohesin is an ATP-driven motor, so the loop ensemble is NOT the Boltzmann one
    contact frequency  the observable is P(d < d_c), not <R^2>; those differ when the distribution is skewed

So this builds the real thing and lets it answer. If it also lands near 0.663 the question is closed with a
simulation rather than an approximation, and that is worth one A100 hour.

STAGE 2 -- KINETIC MONTE CARLO. Cohesin rings load at rate proportional to accessibility, extrude both legs at
v_extrude, stall at CTCF with probability set by peak strength, and unbind at k_off. Run to steady state, then
sample loop configurations. Vectorised over rings; the whole 1D layer is cheap.

STAGE 3 -- LANGEVIN, BATCHED. Written in PyTorch rather than OpenMM for one reason that matters more than
features: a 2,000-bead system uses a few percent of an A100, so the cost is kernel launches, not physics.
Simulating B independent replicas as one (B, N, 3) tensor runs in nearly the same wall-clock as one replica and
returns B samples. That is the 10-50x that makes this affordable, and it is why the ensemble is free.

    U = U_bond + U_bend + U_repulsion + U_loops
    dr = -grad U * dt/gamma + sqrt(2 D dt) * noise          (overdamped Langevin, the relevant limit)

THE BAR IS 0.663, NOT 0.608. The CTCF-count feature already moved the production model there for the cost of a
bisect over a BED file. A simulation that lands below that has been beaten by bookkeeping, and the honest
report is that it was beaten -- not that it beat 0.608.
"""
import bisect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"

# --- Stage 2 (KMC) ---
V_EXTRUDE = 1.0        # kb/s, within the measured 0.5-1.0 range
K_OFF = 1.0 / 900.0    # 1/s -> ~15 min residence
K_LOAD = 2e-4          # per bead per second
# --- Stage 3 (Langevin) ---
BEAD_KB = 2.0          # 2 kb/bead: at chromatin's ~1-3 kb persistence length this is the regime where the
                       # Gaussian assumption is defensible AND the sweep found its optimum
D_CONTACT = 2.0        # contact threshold in bead radii (~45 nm at 2 kb/bead)
# Overridable so the run can be sized to the machine. The defaults are the intended production values;
# a smoke test drops them, and dropping them is a real loss of sampling, not a free speed-up -- the
# contact frequency is estimated from N_REPLICA x (N_STEPS/20) samples, so halving either widens its error.
N_REPLICA = int(os.environ.get("EXT_REPLICA", "64"))
N_STEPS = int(os.environ.get("EXT_STEPS", "3000"))
DT = 0.01


def kmc_loops(n_beads, barrier, loadp, n_conf, seconds=1800.0, dt=1.0, seed=0):
    """Stage 2: steady-state loop configurations from ATP-driven extrusion.

    Returns n_conf snapshots, each a list of (left, right) bead pairs. The loop ensemble is NOT the Boltzmann
    one -- extrusion is motor-driven, so anchors pile up against barriers in a way equilibrium cannot produce.
    That asymmetry is the main thing the analytic Rouse model structurally cannot represent.
    """
    rng = np.random.default_rng(seed)
    steps = max(int(seconds / dt), 1)
    burn = steps // 2
    vb = V_EXTRUDE * dt / BEAD_KB                    # beads per step
    left = np.zeros(0, np.int64)
    right = np.zeros(0, np.int64)
    snaps, take = [], set(np.linspace(burn, steps - 1, n_conf).astype(int).tolist())
    for s in range(steps):
        if len(left):
            mv = rng.random(len(left)) < vb if vb < 1 else np.ones(len(left), bool)
            stepn = max(int(vb), 1)
            passL = rng.random(len(left)) > barrier[np.clip(left - 1, 0, n_beads - 1)]
            passR = rng.random(len(right)) > barrier[np.clip(right + 1, 0, n_beads - 1)]
            left = np.where(mv & passL, np.clip(left - stepn, 0, n_beads - 1), left)
            right = np.where(mv & passR, np.clip(right + stepn, 0, n_beads - 1), right)
            keep = rng.random(len(left)) > K_OFF * dt
            left, right = left[keep], right[keep]
        nnew = rng.poisson(K_LOAD * dt * loadp.sum())
        if nnew:
            p = loadp / loadp.sum()
            pos = rng.choice(n_beads, size=nnew, p=p)
            left = np.concatenate([left, pos])
            right = np.concatenate([right, np.clip(pos + 1, 0, n_beads - 1)])
        if s in take:
            snaps.append(np.stack([left, right], 1).copy() if len(left) else np.zeros((0, 2), np.int64))
    while len(snaps) < n_conf:
        snaps.append(snaps[-1] if snaps else np.zeros((0, 2), np.int64))
    return snaps[:n_conf]


def langevin_contact(n_beads, confs, i, j, torch, device, n_steps=N_STEPS, kappa=0.5, seed=0):
    """Stage 3: batched overdamped Langevin; returns P(d_ij < D_CONTACT) over replicas and time.

    One (B, N, 3) tensor holds every replica. Excluded volume is a soft cosine-capped repulsion rather than
    Lennard-Jones: LJ needs a timestep small enough for its singular core, which costs more than the physics
    is worth at 2 kb resolution where a "bead" is already thousands of nucleosomes.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    B = len(confs)
    # INITIALISE AS AN EQUILIBRATED GAUSSIAN COIL, NOT A ROD.
    # A straight rod of N beads needs the slowest Rouse mode, tau ~ N^2/pi^2 in these units, to collapse:
    # for N=500 that is ~25,000 time units, and 3000 steps at DT=0.01 is 30. The simulation would never
    # leave its initial condition, every pair would sit at its full contour separation, and the contact
    # frequency would be exactly zero -- which is what a smoke test produced. A random walk is already at
    # the Gaussian chain's equilibrium size (<R_ij^2> = |i-j|), so only local structure and the loop
    # constraints have to relax, and those are fast modes.
    r = torch.randn(B, n_beads, 3, device=device, generator=g).cumsum(1)
    r = r - r.mean(1, keepdim=True)
    mx = max((len(c) for c in confs), default=0)
    if mx:
        li = torch.zeros(B, mx, dtype=torch.long, device=device)
        lj = torch.zeros(B, mx, dtype=torch.long, device=device)
        lm = torch.zeros(B, mx, device=device)
        for b, c in enumerate(confs):
            if len(c):
                li[b, :len(c)] = torch.as_tensor(c[:, 0], device=device)
                lj[b, :len(c)] = torch.as_tensor(c[:, 1], device=device)
                lm[b, :len(c)] = 1.0
    hits = 0.0
    inv3 = 0.0
    dsum = 0.0
    nsamp = 0
    for s in range(n_steps):
        F = torch.zeros_like(r)
        d = r[:, 1:] - r[:, :-1]                                  # backbone springs
        F[:, :-1] += d
        F[:, 1:] -= d
        if kappa > 0:                                             # curvature penalty (semiflexible)
            c2 = r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]
            F[:, :-2] -= kappa * c2
            F[:, 1:-1] += 2 * kappa * c2
            F[:, 2:] -= kappa * c2
        if mx:                                                    # cohesin loop bonds
            a = torch.gather(r, 1, li[:, :, None].expand(-1, -1, 3))
            bb = torch.gather(r, 1, lj[:, :, None].expand(-1, -1, 3))
            fv = (bb - a) * lm[:, :, None]
            F.scatter_add_(1, li[:, :, None].expand(-1, -1, 3), fv)
            F.scatter_add_(1, lj[:, :, None].expand(-1, -1, 3), -fv)
        # soft excluded volume against a random subset -- O(N k) instead of O(N^2)
        k = min(32, n_beads - 1)
        idx = torch.randint(0, n_beads, (B, n_beads, k), device=device, generator=g)
        dv = r[:, :, None, :] - torch.gather(r[:, :, None, :].expand(-1, -1, k, -1), 1,
                                             idx[:, :, :, None].expand(-1, -1, -1, 3))
        dist = dv.norm(dim=-1).clamp_min(1e-3)
        rep = torch.where(dist < 1.0, (1.0 - dist), torch.zeros_like(dist))
        F += (dv / dist[..., None] * rep[..., None]).sum(2) * 2.0
        r = r + F * DT + (2 * DT) ** 0.5 * torch.randn(r.shape, device=device, generator=g)
        if s >= n_steps // 2 and s % 10 == 0:
            dij = (r[:, i] - r[:, j]).norm(dim=-1).clamp_min(1e-3)
            hits += float((dij < D_CONTACT).float().sum())
            inv3 += float((dij ** -3).sum())          # smooth contact proxy, see below
            dsum += float(dij.sum())
            nsamp += B
    # TWO observables, because the hard one is a rare event. P(d < d_c) for a pair 500 beads apart is
    # ~1e-3, so 64 replicas x 150 samples yields ~10 hits and an estimate dominated by shot noise. The
    # standard polymer contact proxy <d^-3> uses every sample, is dominated by the same close
    # configurations, and -- unlike the analytic <R^2>^(-3/2) this is meant to improve on -- is sensitive
    # to the SHAPE of the distance distribution rather than just its second moment. That sensitivity is
    # the entire reason to run a simulation instead of solving the Gaussian model, so it is the feature.
    return {"p_hard": hits / max(nsamp, 1),
            "inv3": inv3 / max(nsamp, 1),
            "mean_d": dsum / max(nsamp, 1)}


def main():
    import pandas as pd
    import torch
    from crispr_gate import _cv_auprc, _seeded_folds
    from contact_gate import load_ctcf, contact_features

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f"STAGE 2 + 3 -- KMC extrusion + batched Langevin on {dev}")
    if dev.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    df = pd.read_csv(OUT / "crispr_features_compendium.csv")
    comp = json.load(open(INV / "compendium_tf.json"))
    et, tfl = comp["element_tfs"], comp["tf_list"]
    tss = {}
    with open(INV / "crispr_egpairs.tsv") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ix = {k: hdr.index(k) for k in ("chrom", "chromStart", "chromEnd", "startTSS", "measuredGeneSymbol")}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(hdr):
                continue
            try:
                tss[(f"{p[ix['chrom']]}:{p[ix['chromStart']]}-{p[ix['chromEnd']]}", p[ix["measuredGeneSymbol"]])] = (
                    p[ix["chrom"]], (int(p[ix["chromStart"]]) + int(p[ix["chromEnd"]])) // 2, int(p[ix["startTSS"]]))
            except ValueError:
                continue
    rows = [tss[(e, g)] for e, g in zip(df["element"], df["gene"])]
    ctcf = load_ctcf()
    norm = np.median(np.concatenate([v for _, v in ctcf.values()])) or 1.0

    NSUB = int(os.environ.get("EXT_NSUB", "1200"))
    sel = np.random.default_rng(0).choice(len(rows), min(NSUB, len(rows)), replace=False)
    print(f"  simulating {len(sel):,} of {len(rows):,} pairs ({N_REPLICA} replicas x {N_STEPS} steps each)")

    sim = np.full(len(rows), np.nan)          # <d^-3>, the feature
    hard = np.full(len(rows), np.nan)         # P(d < d_c), reported for honesty about the rare event
    mdist = np.full(len(rows), np.nan)
    sepb = np.full(len(rows), np.nan)         # contour separation in beads, for the relaxation check
    t0 = time.time()
    every = max(1, len(sel) // 40)          # ~40 progress lines whatever EXT_NSUB is
    skipped = 0
    for c, k in enumerate(sel):
        chrom, e, t = rows[k]
        lo, hi = (e, t) if e <= t else (t, e)
        off = max(lo - 200_000, 0)
        n = int((hi + 200_000 - off) / (BEAD_KB * 1000)) + 1
        if n < 8 or n > 1200:
            # >1200 beads = pairs separated by more than ~2 Mb. Skipped for cost, which narrows the
            # simulated subset toward closer pairs. Every arm is scored on that same subset, so the
            # comparison stays fair -- but the count is reported rather than swallowed.
            skipped += 1
            continue
        a, v = ctcf.get(chrom, (np.array([]), np.array([])))
        l, rr = bisect.bisect_left(a, off), bisect.bisect_right(a, off + n * BEAD_KB * 1000)
        barrier = np.zeros(n)
        loadp = np.ones(n) * 0.1
        for pos, sg in zip(a[l:rr], v[l:rr]):
            b = int((pos - off) / (BEAD_KB * 1000))
            if 0 <= b < n:
                barrier[b] = min(0.95, sg / (norm * 3))
                loadp[b] += 1.0
        confs = kmc_loops(n, barrier, loadp, N_REPLICA, seed=int(k))
        i = min(max(int((e - off) / (BEAD_KB * 1000)), 0), n - 1)
        j = min(max(int((t - off) / (BEAD_KB * 1000)), 0), n - 1)
        if i == j:
            continue
        o = langevin_contact(n, confs, i, j, torch, dev, seed=int(k))
        sim[k], hard[k], mdist[k] = o["inv3"], o["p_hard"], o["mean_d"]
        sepb[k] = abs(i - j)
        if c % every == 0:
            dt = time.time() - t0
            eta = dt / max(c, 1) * (len(sel) - c)
            print(f"    {c}/{len(sel)}  {dt:.0f}s elapsed, ~{eta/60:.1f} min left  "
                  f"({n} beads, mean loops/conf {np.mean([len(x) for x in confs]):.1f})", flush=True)
    el = time.time() - t0
    ok = np.isfinite(sim)
    print(f"  simulated {ok.sum():,} pairs in {el/60:.1f} min ({el/max(ok.sum(),1)*1000:.0f} ms/pair)")
    print(f"  skipped {skipped:,} pairs as out of bead range (<8 or >1200 beads, i.e. >~2 Mb apart)")
    print(f"  median <d^-3> {np.nanmedian(sim):.3e} | median P(d<{D_CONTACT}) {np.nanmedian(hard):.5f} "
          f"| median <d> {np.nanmedian(mdist):.2f} beads")
    # THE CHAIN MUST BE A COIL, NOT A ROD. For a Gaussian chain <d> = sqrt(8/3pi)*sqrt(s) ~ 0.92*sqrt(s)
    # for a contour separation of s beads; loops and excluded volume push the ratio down and up
    # respectively, but not by an order of magnitude. An unrelaxed rod gives <d> ~ s, i.e. a ratio of
    # sqrt(s) -- which for typical s here is 15-30, unmistakable. This is the check that would have
    # caught the straight-line initialisation immediately.
    ratio = np.nanmedian(mdist[ok] / np.sqrt(np.maximum(sepb[ok], 1)))
    print(f"  <d>/sqrt(separation) = {ratio:.2f}  (Gaussian coil ~0.92; an unrelaxed rod gives sqrt(s), "
          f"tens) -- {'coil, relaxed' if ratio < 3 else 'NOT RELAXED'}")
    if ratio >= 3:
        raise SystemExit("chain never relaxed to a coil -- the contact numbers would be the initial "
                         "condition, not the physics. Raise EXT_STEPS or check initialisation.")
    if not np.isfinite(np.nanmedian(sim)) or np.nanmedian(sim) <= 0:
        raise SystemExit("every pair returned zero contact proxy -- the chain is not relaxed; do not "
                         "interpret the arms below")

    # score on the simulated subset only -- comparing a subset arm to a full-set number would be meaningless
    sub = np.where(ok)[0]
    Cr = contact_features([rows[i] for i in sub], ctcf)
    tfmat = np.zeros((len(sub), len(tfl)), np.int8)
    for a_, ek in enumerate(df["element"].values[sub]):
        for ti in et.get(ek, []):
            tfmat[a_, ti] = 1
    base = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh",
            "promoter_atac", "promoter_polii", "gene_expr"]
    y = df["crispr_hit"].values[sub]
    ch = df["chromosome"].values[sub]
    Xa = np.hstack([df[base].values[sub], tfmat])
    S = np.log10(np.clip(sim[sub], 1e-12, None))[:, None]
    # <d> is the simulation's version of the quantity the ANALYTIC model already gave (a second moment).
    # Scoring it alongside <d^-3> is what separates "simulation beats the closed form" from "shape
    # sensitivity beats the second moment" -- without this arm the two are confounded.
    Dm = np.log10(np.clip(mdist[sub], 1e-6, None))[:, None]

    SEEDS = (0, 1, 2, 3, 4)

    def per_seed(X):
        """AUPRC for EACH fold seed, kept separate so deltas can be paired seed-by-seed.

        Averaging first and subtracting after throws away the pairing and with it the only cheap
        uncertainty available here. A +0.03 mean that flips sign across fold assignments is noise, and
        collapsing to one number makes that indistinguishable from a real effect.
        """
        v = [_cv_auprc(X, y, _seeded_folds(ch, s)) for s in SEEDS]
        v = [x for x in v if np.isfinite(x)]
        if not v:
            raise SystemExit(f"AUPRC undefined on {len(y)} pairs with {int(y.sum())} positives -- every "
                             "chromosome fold had fewer than 3 positives. Raise EXT_NSUB; do NOT read a "
                             "verdict off this run.")
        return np.array(v)

    A_id, A_cnt = per_seed(Xa), per_seed(np.hstack([Xa, Cr]))
    A_dm, A_sim = per_seed(np.hstack([Xa, Dm])), per_seed(np.hstack([Xa, S]))
    A_both = per_seed(np.hstack([Xa, Cr, S]))

    # THE CONTROL, drawn MORE THAN ONCE. Shuffling the simulated values across pairs destroys the
    # pair-specific physics while keeping the column's marginal distribution and whatever extra capacity
    # a 10th numeric feature gives XGBoost. A SINGLE shuffle is one realisation of a noisy quantity --
    # the identical mistake that once put a GPR control on 2-5 hits and produced "lifts" of 0.5x/3x/5x
    # that were pure sampling noise. N_SHUF independent draws x len(SEEDS) fold assignments gives the
    # control a mean AND a standard error, so the real gain can be required to exceed the control's own
    # scatter rather than merely its point estimate.
    N_SHUF = 5
    shuf_deltas = []
    for sd in range(N_SHUF):
        Ssh = S[np.random.default_rng(sd).permutation(len(S))]
        shuf_deltas.append(per_seed(np.hstack([Xa, Cr, Ssh])) - A_cnt)
    shuf_deltas = np.concatenate(shuf_deltas)
    A_shuf = A_cnt + shuf_deltas.mean()

    def sc(a):
        return float(np.mean(a))

    res = {"n_simulated": int(ok.sum()), "minutes": el / 60, "device": str(dev),
           "median_inv3": float(np.nanmedian(sim)), "median_p_hard": float(np.nanmedian(hard)),
           "coil_ratio": float(ratio), "n_positives": int(y.sum()),
           "identity": sc(A_id), "count_contact": sc(A_cnt), "mean_dist_only": sc(A_dm),
           "SIM_only": sc(A_sim), "SIM_plus_count": sc(A_both),
           "SIM_shuffled_plus_count": sc(A_shuf)}
    print("\n  (all arms scored on the SAME simulated subset, so they are comparable to each other;")
    print("   they are NOT comparable to the full-set 0.663 -- count_contact below IS that bar, resc"
          "ored here)")
    for k_ in ("identity", "count_contact", "mean_dist_only", "SIM_only",
               "SIM_plus_count", "SIM_shuffled_plus_count"):
        print(f"    {k_:24s} AUPRC {res[k_]:.4f}")

    def paired(a, b, label, note):
        d = a - b
        print(f"  {label:34s} {d.mean():+.4f}  [per-seed {' '.join(f'{x:+.3f}' for x in d)}]  {note}")
        return d

    print(f"\n  paired seed-by-seed deltas over {len(SEEDS)} chromosome-fold assignments:")
    d1 = paired(A_both, A_cnt, "SIM + count MINUS count", "<- does simulating beat counting")
    d2 = paired(A_sim, A_dm, "<d^-3> MINUS <d>", "<- shape vs 2nd moment")
    print(f"  {'SHUFFLED-SIM + count MINUS count':34s} {shuf_deltas.mean():+.4f}  "
          f"[{N_SHUF} draws x {len(SEEDS)} seeds, sd {shuf_deltas.std(ddof=1):.4f}, "
          f"range {shuf_deltas.min():+.3f}..{shuf_deltas.max():+.3f}]  <- CONTROL")

    net = d1.mean() - shuf_deltas.mean()
    # Standard error of the DIFFERENCE of two means, which is what "net" is.
    se = float(np.sqrt(d1.var(ddof=1) / len(d1) + shuf_deltas.var(ddof=1) / len(shuf_deltas)))
    consistent = bool(np.all(np.sign(d1) == np.sign(d1.mean())))
    print(f"\n  real MINUS shuffled control : {net:+.4f} +/- {se:.4f} (1 se)   z = {net/max(se,1e-9):+.2f}")
    print(f"  sign consistent across all {len(d1)} fold assignments: {consistent}")
    res.update({"delta_vs_count": float(d1.mean()), "delta_vs_count_per_seed": d1.tolist(),
                "delta_shuffled_control": float(shuf_deltas.mean()),
                "delta_shuffled_control_sd": float(shuf_deltas.std(ddof=1)),
                "delta_net_of_control": float(net), "delta_net_se": se,
                "delta_shape_vs_moment": float(d2.mean()), "sign_consistent": consistent})
    # A GO needs all of: a gain worth having, a gain that SURVIVES the shuffled control, a gain larger
    # than the noise in that comparison, and a sign that does not flip with the fold assignment. Every
    # one of those has, on its own, produced a false positive somewhere in this project.
    if net >= 0.01 and net > 2 * se and consistent:
        res["verdict"] = ("GO -- the simulation adds signal beyond counting CTCF peaks, and the gain "
                          "survives a shuffled-simulation control by more than 2 se.")
    elif d1.mean() >= 0.01 and net < 0.01:
        res["verdict"] = (f"NO-GO -- the +{d1.mean():.4f} gain is largely reproduced by the SHUFFLED "
                          f"control (+{shuf_deltas.mean():.4f}), so most of it is an extra-column "
                          "effect, not physics.")
    elif net >= 0.01 and net <= 2 * se:
        res["verdict"] = (f"UNDERPOWERED -- net {net:+.4f} but 1 se is {se:.4f}, so this run cannot "
                          "tell the effect from the control's own scatter. Raise EXT_NSUB and re-run.")
    elif not consistent:
        res["verdict"] = (f"INCONCLUSIVE -- mean gain {d1.mean():+.4f} but the sign flips across fold "
                          "assignments. Raise EXT_NSUB and re-run.")
    else:
        res["verdict"] = "NO-GO -- full extrusion + Langevin does not beat a bisect over a BED file."
    print(f"\n  VERDICT: {res['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "extrude_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'extrude_gate.json'}")


if __name__ == "__main__":
    main()
