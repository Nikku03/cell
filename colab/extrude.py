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
N_REPLICA = 64
N_STEPS = 3000
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
    r = torch.zeros(B, n_beads, 3, device=device)
    r[:, :, 0] = torch.arange(n_beads, device=device, dtype=torch.float32)[None, :]
    r += 0.1 * torch.randn(r.shape, device=device, generator=g)
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
            dij = (r[:, i] - r[:, j]).norm(dim=-1)
            hits += float((dij < D_CONTACT).float().sum())
            nsamp += B
    return hits / max(nsamp, 1)


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

    sim = np.full(len(rows), np.nan)
    t0 = time.time()
    for c, k in enumerate(sel):
        chrom, e, t = rows[k]
        lo, hi = (e, t) if e <= t else (t, e)
        off = max(lo - 200_000, 0)
        n = int((hi + 200_000 - off) / (BEAD_KB * 1000)) + 1
        if n < 8 or n > 1200:
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
        sim[k] = langevin_contact(n, confs, i, j, torch, dev, seed=int(k))
        if c % 100 == 0:
            print(f"    {c}/{len(sel)}  {time.time()-t0:.0f}s  mean loops/conf "
                  f"{np.mean([len(x) for x in confs]):.1f}", flush=True)
    el = time.time() - t0
    ok = np.isfinite(sim)
    print(f"  simulated {ok.sum():,} pairs in {el/60:.1f} min "
          f"({el/max(ok.sum(),1)*1000:.0f} ms/pair); contact freq median {np.nanmedian(sim):.4f}")

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
    S = np.log10(np.clip(sim[sub], 1e-4, None))[:, None]

    def sc(X):
        return float(np.mean([_cv_auprc(X, y, _seeded_folds(ch, s)) for s in (0, 1, 2)]))

    res = {"n_simulated": int(ok.sum()), "minutes": el / 60, "device": str(dev),
           "identity": sc(Xa), "count_contact": sc(np.hstack([Xa, Cr])),
           "SIM_only": sc(np.hstack([Xa, S])), "SIM_plus_count": sc(np.hstack([Xa, Cr, S]))}
    print("\n  (all arms scored on the SAME simulated subset, so they are comparable to each other)")
    for k_, v_ in res.items():
        if isinstance(v_, float) and k_ not in ("minutes",):
            print(f"    {k_:22s} AUPRC {v_:.4f}")
    d1 = res["SIM_plus_count"] - res["count_contact"]
    print(f"\n  SIM + count MINUS count-only : {d1:+.4f}")
    res["delta_vs_count"] = d1
    res["verdict"] = ("GO -- the simulation adds signal beyond counting CTCF peaks." if d1 >= 0.01 else
                      "NO-GO -- full extrusion + Langevin does not beat a bisect over a BED file.")
    print(f"  VERDICT: {res['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "extrude_gate.json", "w"), indent=1)
    print(f"\n  -> {OUT/'extrude_gate.json'}")


if __name__ == "__main__":
    main()
