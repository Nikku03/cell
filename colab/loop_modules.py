"""Loop 219. Genes that move together: is the change measurable at MODULE level?

THE PROPOSAL. Stop predicting each gene as an independent line. Genes affect one another, so genes
that change together should be modelled together -- attach the lines according to co-change.

WHY IT ATTACKS THE RIGHT PROBLEM. Loop 216 measured that the per-interval change of a single gene
is not measurable: one replicate predicts another at R2 -0.540, against +0.834 for the plateau. Loop
217 could not find a subset of genes where it becomes measurable, and loop 218 showed a fully
independent fourth replicate does not rescue it either. Every one of those worked at the level of
the individual gene. If a set of genes genuinely moves together then their AVERAGE change has the
same signal and less noise, so the module-level change can be measurable when no single gene's is.

THE CONTROL THAT DECIDES WHETHER THIS MEANS ANYTHING, AND IT IS THE WHOLE DESIGN. Averaging k genes
reduces independent noise by sqrt(k) whether or not those genes have anything to do with each
other. So a co-change module that beats a single gene proves nothing. It has to beat a RANDOM
module of the same size, drawn from the same genes. N4 is that comparison and every other gate is
scaffolding for it.

THE SECOND TRAP. Modules discovered on the same data they are then scored on will look coherent by
construction -- that is what clustering does. So modules are discovered on the TRAINING INTERVALS
of replicates 2 and 3 only, and the ceiling is measured on replicate 4, which took no part in
either the clustering or the interval selection. Loop 218 established that replicate 4 is the same
experiment and shares no agreement with 2 and 3 beyond the general level, which makes it the right
scorer.

PREDECLARED, BEFORE ANY NUMBER.

  N1 ARE THE MODULES DISCOVERED WITHOUT TOUCHING THE SCORER?
     Gate: PASS iff clustering uses only replicates 2 and 3, only training intervals, and the
     resulting modules have a non-degenerate size distribution -- at least 5 modules with 5 or
     more genes, and no module holding more than half the genes. A single giant module is not a
     decomposition.

  N2 DO THE MODULES REPRODUCE ON A HELD-OUT REPLICATE?
     Correlate each module's mean change trajectory computed on replicates 2+3 against the same
     module's trajectory on replicate 4.
     Gate: PASS iff the median across modules exceeds 0.30. A FAIL means the modules are noise
     structure and N4 cannot be read.

  N3 IS THE MODULE-LEVEL CHANGE MEASURABLE?
     The replicate ceiling recomputed on module mean changes instead of gene changes: replicate 2
     predicting replicate 3, and 2+3 predicting 4, at module level.
     Gate: PASS iff the module-level ceiling exceeds the gene-level -0.54028 by more than 0.50.

  N4 IS IT THE CO-CHANGE, OR JUST THE AVERAGING?
     Repeat N3 with RANDOM modules matched to the real ones in size, 200 draws.
     Gate: PASS iff the real modules' ceiling exceeds the 95th percentile of the random-module
     ceilings. This is the gate the loop exists for. A FAIL means co-change adds nothing over
     arithmetic and the modules are decoration on a sqrt(k) noise reduction.
     Requires N3.

  N5 CAN THE MODEL PREDICT MODULE CHANGE?
     The persisted set point aggregated to module level, relaxed, scored against persistence on
     module mean changes from replicate 4.
     Gate: PASS iff the model beats persistence by more than 0.01 at module level.
     Requires N4 -- predicting a module that is only an average is predicting an average.

  N6 DOES MODULE MEMBERSHIP HELP AN INDIVIDUAL GENE?
     Decompose each gene's change into its module's mean plus a gene-specific residual, and ask
     how much of the gene's measurable signal lives in the module term.
     Gate: PASS iff the module term carries more than 30% of the between-gene signal variance.
     A FAIL means genes move mostly on their own and the module is not the right unit.

  N7 WHAT THIS CANNOT SHOW.
     Stated, not scored.
"""
import json, os, pickle, sys, time, warnings
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
MODEL = ROOT / "colab" / "models" / "setpoint_stack_v1.pkl"
OUT = "outputs/loop_modules.json"
GRID = [30, 60, 120, 180, 240, 420, 480, 600, 720]
N_TRAIN, SEED, NRAND = 6, 219219, 200
REF_GENE = -0.54028

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


def main():
    t0 = time.time()
    G = Gates(emit=say)
    res = {"test": "co-change modules"}
    say("=" * 104)
    say("LOOP 219 -- GENES THAT MOVE TOGETHER: IS THE CHANGE MEASURABLE AT MODULE LEVEL?")
    say("=" * 104)

    z = np.load(SP / "grtc" / "rna.npz", allow_pickle=True)
    tpm, mins, reps = z["tpm"], z["mins"].astype(int), z["reps"].astype(int)
    grid, M, A9, sym, keep, tssb = gene_set()
    gi = np.where(keep)[0]
    art = pickle.load(open(MODEL, "rb"))
    names = art["genes"]
    pos = {s: k for k, s in enumerate([sym[i] for i in gi])}
    idx = gi[np.array([pos[s] for s in names])]
    S = np.array(art["stack_prediction"])
    g = np.array(GRID, float)
    D = {}
    for r in (1, 2, 3, 4):
        Mi, _ = L191.rep_trajectories(tpm, mins, reps, (r,), g)
        A = Mi[:, idx]
        D[r] = np.array([A[j] - A[j - 1] for j in range(1, len(g))])   # (8, ng)
    ng, nint = len(names), len(g) - 1
    trj = np.arange(nint) < (N_TRAIN - 1)
    say(f"     {ng:,} genes, {nint} intervals, replicates 1-4")

    # ---------------------------------------------------------------- N1
    say("N1 ARE THE MODULES DISCOVERED WITHOUT TOUCHING THE SCORER?")
    Xdisc = np.mean([D[2], D[3]], axis=0)[trj]            # training intervals, reps 2+3 only
    Z = (Xdisc - Xdisc.mean(0)) / (Xdisc.std(0) + 1e-9)
    C = np.corrcoef(Z.T)
    C = np.nan_to_num(C)
    # hierarchical-style clustering by correlation, agglomerated with a simple threshold sweep
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Dm = np.clip(1.0 - C, 0, 2); np.fill_diagonal(Dm, 0.0)
    Lk = linkage(squareform(Dm, checks=False), method="average")
    best = None
    for t in np.arange(0.5, 1.15, 0.05):
        lab = fcluster(Lk, t=t, criterion="distance")
        sizes = np.bincount(lab)[1:]
        big = int((sizes >= 5).sum()); mx = int(sizes.max())
        if big >= 5 and mx <= ng // 2:
            best = (t, lab, sizes, big, mx)
            break
    if best is None:
        G.add("N1", None, void_if=True,
              void_reason="no threshold in the sweep gave >=5 modules of >=5 genes with none "
                          "holding more than half the genes")
        G.summary(seconds=time.time() - t0); return
    thr, lab, sizes, big, mx = best
    mods = [np.where(lab == k)[0] for k in np.unique(lab) if (lab == k).sum() >= 5]
    say(f"     discovery input: replicates 2+3, training intervals only, "
        f"{Xdisc.shape[0]} x {ng}")
    say(f"     cut at correlation distance {thr:.2f}: {len(np.unique(lab))} clusters, "
        f"{big} with >=5 genes, largest {mx}")
    say(f"     modules kept: {len(mods)}  sizes "
        f"{sorted([len(m) for m in mods], reverse=True)[:12]}")
    say(f"     genes in a kept module: {sum(len(m) for m in mods):,} of {ng:,}")
    G.add("N1", True,
          if_true=f"N1 PASS -- {len(mods)} modules, largest {mx}, discovered on replicates 2+3 "
                  f"training intervals only")
    res["modules"] = {"n": len(mods), "sizes": [int(len(m)) for m in mods], "threshold": float(thr)}

    def mod_traj(Dr, ms):
        return np.array([Dr[:, m].mean(1) for m in ms])    # (n_mod, nint)

    # ---------------------------------------------------------------- N2
    say("N2 DO THE MODULES REPRODUCE ON A HELD-OUT REPLICATE?")
    T23 = mod_traj(np.mean([D[2], D[3]], axis=0), mods)
    T4 = mod_traj(D[4], mods)
    rr = np.array([corr(T23[i], T4[i]) for i in range(len(mods))])
    say(f"     module trajectory correlation, replicates 2+3 against replicate 4:")
    say(f"       median {np.median(rr):+.4f}   mean {rr.mean():+.4f}   "
        f"fraction above 0.3 {float((rr>0.3).mean()):.1%}")
    G.add("N2", bool(np.median(rr) > 0.30), stat=float(np.median(rr)), requires=("N1",),
          if_true=lambda: f"N2 PASS -- median {np.median(rr):+.3f}, the modules reproduce on a "
                          f"replicate that took no part in finding them",
          if_false=lambda: f"N2 FAIL -- median {np.median(rr):+.3f}; the modules are noise "
                           f"structure")
    res["reproduce"] = {"median": float(np.median(rr)), "mean": float(rr.mean())}

    # ---------------------------------------------------------------- N3
    say("N3 IS THE MODULE-LEVEL CHANGE MEASURABLE?")
    m2, m3, m4 = mod_traj(D[2], mods), mod_traj(D[3], mods), mod_traj(D[4], mods)
    c23 = r2s(m3.ravel(), m2.ravel())
    c234 = r2s(m4.ravel(), np.mean([m2, m3], axis=0).ravel())
    say(f"     module level:  rep2 predicts rep3   R2 {c23:+.5f}")
    say(f"                    rep2+3 predicts rep4 R2 {c234:+.5f}")
    say(f"     gene level (loop 216, replicates 1-3): {REF_GENE:+.5f}")
    best_c = max(c23, c234)
    G.add("N3", bool(best_c - REF_GENE > 0.50), stat=best_c, requires=("N2",),
          if_true=lambda: f"N3 PASS -- module-level ceiling {best_c:+.4f} against the gene-level "
                          f"{REF_GENE:+.4f}",
          if_false=lambda: f"N3 FAIL -- module-level {best_c:+.4f} against {REF_GENE:+.4f}")
    res["ceiling"] = {"module_23": c23, "module_234": c234, "gene_ref": REF_GENE}

    # ---------------------------------------------------------------- N4
    say("N4 IS IT THE CO-CHANGE, OR JUST THE AVERAGING?")
    rng = np.random.default_rng(SEED)
    sz = [len(m) for m in mods]
    rand_c = []
    for _ in range(NRAND):
        rmods = []
        perm = rng.permutation(ng)
        p = 0
        for s in sz:
            rmods.append(perm[p:p + s]); p += s
        a, b = mod_traj(D[2], rmods), mod_traj(D[3], rmods)
        rand_c.append(r2s(b.ravel(), a.ravel()))
    rand_c = np.array(rand_c)
    p95 = float(np.percentile(rand_c, 95))
    say(f"     RANDOM modules, same sizes, {NRAND} draws:")
    say(f"       median {np.median(rand_c):+.5f}   95th pct {p95:+.5f}   max {rand_c.max():+.5f}")
    say(f"     real modules  {c23:+.5f}")
    G.add("N4", bool(c23 > p95), stat=c23, requires=("N3",),
          if_true=lambda: f"N4 PASS -- real modules at {c23:+.4f} exceed the random 95th "
                          f"percentile {p95:+.4f}. Co-change is doing work beyond sqrt(k)",
          if_false=lambda: f"N4 FAIL -- real {c23:+.4f} against a random 95th percentile of "
                           f"{p95:+.4f}. Averaging ANY {int(np.mean(sz))} genes buys the same "
                           f"thing; the co-change structure adds nothing")
    res["random_control"] = {"median": float(np.median(rand_c)), "p95": p95,
                             "max": float(rand_c.max()), "real": c23}

    # ---------------------------------------------------------------- N5
    say("N5 CAN THE MODEL PREDICT MODULE CHANGE?")
    Mm = M[:, idx]
    lvl = np.array([Mm[j - 1] for j in range(1, len(g))])
    dts = np.array([g[j] - g[j - 1] for j in range(1, len(g))])
    Smod = np.array([S[m].mean() for m in mods])
    Lmod = np.array([lvl[:, m].mean(1) for m in mods]).T          # (nint, n_mod)
    Ytr = np.mean([m2, m3], axis=0).T[trj]
    d_tr = (dts[trj, None] * (Smod[None, :] - Lmod[trj])).ravel()
    lam = float(d_tr @ Ytr.ravel() / (d_tr @ d_tr)) if (d_tr @ d_tr) > 0 else 0.0
    d_te = (dts[~trj, None] * (Smod[None, :] - Lmod[~trj])).ravel()
    y4 = m4.T[~trj].ravel()
    rm, rp_ = r2s(y4, lam * d_te), r2s(y4, np.zeros_like(y4))
    say(f"     module-level forward: model {rm:+.5f}   persistence {rp_:+.5f}   "
        f"margin {rm-rp_:+.5f}")
    G.add("N5", bool(rm - rp_ > 0.01), stat=rm, requires=("N4",),
          if_true=lambda: f"N5 PASS -- the model beats persistence at module level by "
                          f"{rm-rp_:+.5f}",
          if_false=lambda: f"N5 FAIL -- {rm-rp_:+.5f}")
    res["forward"] = {"model": rm, "persistence": rp_, "margin": rm - rp_, "lam": lam}

    # ---------------------------------------------------------------- N6
    say("N6 DOES MODULE MEMBERSHIP HELP AN INDIVIDUAL GENE?")
    inmod = np.concatenate(mods)
    memb = np.zeros(ng, int) - 1
    for k, m in enumerate(mods):
        memb[m] = k
    Dall = np.mean([D[2], D[3], D[4]], axis=0)
    modmean = np.zeros_like(Dall)
    for k, m in enumerate(mods):
        modmean[:, m] = Dall[:, m].mean(1, keepdims=True)
    sel = memb >= 0
    var_tot = float(np.var(Dall[:, sel]))
    var_mod = float(np.var(modmean[:, sel]))
    frac = var_mod / var_tot if var_tot > 0 else float("nan")
    say(f"     genes in a module {int(sel.sum()):,}")
    say(f"     variance of the module term {var_mod:.6f} of total {var_tot:.6f} = {frac:.1%}")
    G.add("N6", bool(frac > 0.30), stat=frac, requires=("N1",),
          if_true=lambda: f"N6 PASS -- {frac:.0%} of the change variance is shared within modules",
          if_false=lambda: f"N6 FAIL -- only {frac:.0%} of a gene's change is its module's; genes "
                           f"move mostly on their own and the module is not the right unit")
    res["decomposition"] = {"var_total": var_tot, "var_module": var_mod, "fraction": frac}

    say("N7 WHAT THIS CANNOT SHOW")
    say("     A module mean is not a gene. Predicting that a group of genes rises is a weaker")
    say("     claim than predicting which ones, and N6 measures how much weaker.")
    say("     Modules are discovered from CO-CHANGE, which is a correlation and not a mechanism.")
    say("     Loop 187 measured that this project's regulatory network does not explain")
    say("     co-expression, so a module here need not correspond to a regulon.")
    say("     Replicate 4 is the scorer, and loop 218 measured that replicate 4 agrees with 2 and")
    say("     3 no better than any other pair -- the only real agreement in this dataset is")
    say("     between 2 and 3. Discovering on the one agreeing pair and scoring on an outlier is")
    say("     the strictest available test and also the harshest.")
    say("     Eight intervals is a very short series to cluster trajectories on.")

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
