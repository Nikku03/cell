"""STEP 4: the STATE-CONDITIONED ADRN. Does knowing the cell's state improve prediction of a held-out knockout?

THE ARCHITECTURAL CLAIM BEING TESTED.  Every ADRN so far maps a gene descriptor to ONE response profile. It has
no input for the state the cell was in, so it must predict the same thing for a G1 cell and a mitotic one. Today's
cross-line and LOCO failures both trace to that: a model with no context input cannot be context-sensitive. The
single-cell data supplies context that pseudobulk destroys.

THE EVALUATION UNIT is the (held-out perturbation x state tertile x line) group, not the individual cell. A single
cell's profile is mostly Poisson noise; the quantity conditioning is supposed to improve is the group's mean
deviation, and that is what is scored. 40 of the 200 perturbations are held out ENTIRELY -- the model never sees
that gene in any line or any state -- so this is the same cold-start question the sealed cohorts ask, now with a
state axis.

TARGET.  For a cell group, the mean expression minus the mean of NON-TARGETING cells in the SAME BATCHES and the
SAME state tertile. Matching the controls on both batch and state is what makes the target the perturbation's
state-specific effect rather than the cell cycle or the batch.

ARMS -- each adds one thing, so the gain of each is attributable:

    mean_train    the average training deviation profile. Knows nothing. The floor.
    state_only    state features only, no gene identity. Captures whatever remains of the cycle main effect
                  after control matching -- and if this scores well, the task is measuring the cell cycle.
    gene_only     gene channels only. This is the existing pseudobulk ADRN, re-scored on this task.
    gene_state    gene channels AND state features, additively.
    gene_x_state  the interaction: gene descriptor (SVD-reduced) outer-product state features. THE CONDITIONED
                  MODEL -- the only arm that can predict a DIFFERENT profile for the same gene in different states.

PREDECLARED GATES, both required on the robustness harness:

    G1  gene_x_state > gene_state    conditioning buys something beyond adding state additively. This is the
                                     architectural claim; without it, "conditioning" is just extra features.
    G2  gene_state   > gene_only     state carries any information at all about a held-out knockout's response.

If G2 passes and G1 fails, state is a main effect and pseudobulk was losing little. If both fail, the single-cell
route adds nothing to prediction whatever the state gate said about the raw interaction being real.

SCORED TWO WAYS because they fail differently: Pearson r against the measured group profile (continuous, sensitive
to the whole vector) and precision@20 on the group's strongest movers (comparable to every other number in this
project).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A
import adrn_ko_channels2 as C
from robustness import Sweeper

OUT, SP = A.OUT, A.SP
BINS_SWEEP, RANK_SWEEP = (2, 3), (16, 32, 64)
N_HOLDOUT, MIN_GROUP, PENALTIES = 40, 25, (1.0, 10.0, 100.0, 1000.0)
NPRED = 20


def main():
    log = []
    t0 = time.time()

    def report(t):
        print(t, flush=True)
        log.append(t)

    report("=" * 100)
    report("STEP 4 -- STATE-CONDITIONED ADRN")
    report("=" * 100)
    report("  PREDECLARED  G1 gene_x_state > gene_state | G2 gene_state > gene_only")

    z = np.load(SP / "sc_cohort_cells.npz", allow_pickle=True)
    X = z["X"]
    line = np.array([str(x) for x in z["line"]])
    pert = np.array([str(x) for x in z["pert"]])
    batch = z["batch"]
    ntc = z["is_ntc"]
    s_sc, g2_sc, umi = z["s_score"].astype(np.float64), z["g2m_score"].astype(np.float64), z["umi"].astype(np.float64)
    genes_hvg = [str(g) for g in z["genes"]]
    perts = sorted(set(pert[~ntc]))
    lines = sorted(set(line))
    report(f"  {X.shape[0]:,} cells, {len(perts)} perturbations, {len(lines)} lines, {X.shape[1]:,} genes")

    # gene descriptors: the same chan2a channels used everywhere else in this project
    zk = np.load(SP / "nlz_K562_gwps.npz", allow_pickle=True)
    kos = [str(k) for k in zk["kos"]]
    universe = sorted(set(kos) | set(perts))
    base, _ = A.build_channels(universe, set(kos), report)
    extra, _, tb = C.extra_channels(universe, set(kos), report)
    chan = np.concatenate([base, extra[:, :tb]], axis=1)
    urow = {g: i for i, g in enumerate(universe)}
    Cg = chan[[urow[g] for g in perts]]
    report(f"  gene descriptors {Cg.shape}")

    rng = np.random.default_rng(A.SEED)
    hold = set(np.array(perts)[rng.choice(len(perts), N_HOLDOUT, replace=False)].tolist())
    train_p = [p for p in perts if p not in hold]
    report(f"  held out {len(hold)} perturbations entirely; {len(train_p)} for training")

    sw1 = Sweeper("gene_x_state - gene_state", axes={"bins": list(BINS_SWEEP), "rank": list(RANK_SWEEP)})
    sw2 = Sweeper("gene_state - gene_only", axes={"bins": list(BINS_SWEEP), "rank": list(RANK_SWEEP)})
    means_all = {}

    for nbins in BINS_SWEEP:
        # state tertiles/halves computed WITHIN line, so a line's overall cycle distribution is not the signal
        binid = np.zeros(len(X), np.int64)
        for ln in lines:
            m = line == ln
            q = np.quantile(g2_sc[m], np.linspace(0, 1, nbins + 1)[1:-1])
            binid[m] = np.digitize(g2_sc[m], q)

        groups, targets = [], []
        for ln in lines:
            for p in perts:
                for b in range(nbins):
                    m = (line == ln) & (pert == p) & ~ntc & (binid == b)
                    if m.sum() < MIN_GROUP:
                        continue
                    bs = set(batch[m].tolist())
                    cm = (line == ln) & ntc & np.isin(batch, list(bs)) & (binid == b)
                    if cm.sum() < 15:
                        continue
                    groups.append((ln, p, b, int(m.sum())))
                    targets.append(X[m].mean(0) - X[cm].mean(0))
        Y = np.array(targets, np.float32)
        report(f"\n  bins={nbins}: {len(groups):,} (line x pert x state) groups with >= {MIN_GROUP} cells")

        gp = np.array([g[1] for g in groups])
        gl = np.array([g[0] for g in groups])
        gb = np.array([g[2] for g in groups])
        is_hold = np.isin(gp, list(hold))
        report(f"    train groups {int((~is_hold).sum()):,} | held-out groups {int(is_hold.sum()):,}")

        # state feature block: bin one-hot, line one-hot, and the group's mean scaled scores
        st_rows = []
        for (ln, p, b, n) in groups:
            m = (line == ln) & (pert == p) & ~ntc & (binid == b)
            st_rows.append([float(g2_sc[m].mean()), float(s_sc[m].mean()), float(np.log10(umi[m].mean()))]
                           + [1.0 if b == k else 0.0 for k in range(nbins)]
                           + [1.0 if ln == q else 0.0 for q in lines])
        St = np.array(st_rows, np.float32)
        St[:, :3] = (St[:, :3] - St[~is_hold, :3].mean(0)) / (St[~is_hold, :3].std(0) + 1e-6)

        pidx = {p: i for i, p in enumerate(perts)}
        Gfull = Cg[[pidx[p] for p in gp]]

        from sklearn.decomposition import TruncatedSVD

        def fit_predict(F):
            Ftr, Fte = F[~is_hold], F[is_hold]
            best, bw = -1e18, None
            r2 = np.random.default_rng(A.SEED + 1).permutation(int((~is_hold).sum()))
            nv = max(20, int(0.2 * len(r2)))
            for pen in PENALTIES:
                w = A.ridge_fit(Ftr[r2[nv:]], Y[~is_hold][r2[nv:]], pen)
                e = -float(np.mean((A.ridge_apply(Ftr[r2[:nv]], w) - Y[~is_hold][r2[:nv]]) ** 2))
                if e > best:
                    best, bw, bp = e, w, pen
            w = A.ridge_fit(Ftr, Y[~is_hold], bp)
            return A.ridge_apply(Fte, w)

        Yte = Y[is_hold]

        def score(P):
            r = np.array([float(np.corrcoef(P[i], Yte[i])[0, 1]) if np.std(P[i]) > 0 else 0.0
                          for i in range(len(Yte))])
            pr = []
            for i in range(len(Yte)):
                truth = set(np.argsort(-np.abs(Yte[i]))[:NPRED].tolist())
                pr.append(len(set(np.argsort(-np.abs(P[i]))[:NPRED].tolist()) & truth) / float(NPRED))
            return np.nan_to_num(r), np.array(pr)

        ones = np.ones((len(groups), 1), np.float32)
        P_mean = np.repeat(Y[~is_hold].mean(0, keepdims=True), int(is_hold.sum()), 0)
        arms = {"mean_train": P_mean,
                "state_only": fit_predict(np.concatenate([ones, St], 1)),
                "gene_only": fit_predict(np.concatenate([ones, Gfull], 1))}
        for rank in RANK_SWEEP:
            Gs = TruncatedSVD(n_components=rank, random_state=0).fit_transform(Cg)
            Gr = Gs[[pidx[p] for p in gp]]
            Gr = (Gr - Gr[~is_hold].mean(0)) / (Gr[~is_hold].std(0) + 1e-6)
            inter = (Gr[:, :, None] * St[:, None, :]).reshape(len(groups), -1)
            a_gs = fit_predict(np.concatenate([ones, Gfull, St], 1))
            a_gx = fit_predict(np.concatenate([ones, Gfull, St, inter], 1))
            r_gs, p_gs = score(a_gs)
            r_gx, p_gx = score(a_gx)
            r_go, p_go = score(arms["gene_only"])
            sw1.add({"bins": nbins, "rank": rank}, f"r|{nbins}", r_gx - r_gs)
            sw1.add({"bins": nbins, "rank": rank}, f"p@20|{nbins}", p_gx - p_gs)
            sw2.add({"bins": nbins, "rank": rank}, f"r|{nbins}", r_gs - r_go)
            sw2.add({"bins": nbins, "rank": rank}, f"p@20|{nbins}", p_gs - p_go)
            means_all[f"bins={nbins}|rank={rank}"] = {
                "gene_x_state_r": float(r_gx.mean()), "gene_state_r": float(r_gs.mean()),
                "gene_only_r": float(r_go.mean()),
                "gene_x_state_p20": float(p_gx.mean()), "gene_state_p20": float(p_gs.mean()),
                "gene_only_p20": float(p_go.mean())}
            report(f"    rank {rank:>3}  r: gene {r_go.mean():.4f} -> +state {r_gs.mean():.4f} "
                   f"-> xstate {r_gx.mean():.4f}   p@20: {p_go.mean():.4f} / {p_gs.mean():.4f} / {p_gx.mean():.4f}")
        for nm in ("mean_train", "state_only"):
            r, p = score(arms[nm])
            means_all[f"bins={nbins}|{nm}"] = {"r": float(r.mean()), "p20": float(p.mean())}
            report(f"    {nm:<11} r {r.mean():.4f}  p@20 {p.mean():.4f}")

    for s in (sw1, sw2):
        report(s.report())
    v1, v2 = sw1.verdict(), sw2.verdict()
    report("\n  READING")
    if v1["verdict"] in ("ROBUST", "DIRECTIONAL"):
        report("  G1 PASSES: conditioning buys something beyond adding state additively -- the interaction is")
        report("  real and the architecture change is justified.")
    elif v2["verdict"] in ("ROBUST", "DIRECTIONAL"):
        report("  G1 fails, G2 passes: state is a MAIN EFFECT, not an interaction. Pseudobulk was losing little,")
        report("  and 'conditioning' here is just extra features.")
    else:
        report("  Both gates fail: state adds nothing to predicting a held-out knockout, however real the raw")
        report("  interaction is. The single-cell route does not improve prediction at this scale.")

    json.dump({"test": "sc_adrn", "n_holdout": N_HOLDOUT, "bins": list(BINS_SWEEP),
               "ranks": list(RANK_SWEEP), "means": means_all,
               "G1_gene_x_state_vs_gene_state": v1, "G2_gene_state_vs_gene_only": v2, "log": log},
              open(OUT / "sc_adrn.json", "w"), indent=2)
    report(f"\n  total {time.time() - t0:.0f}s  -> {OUT / 'sc_adrn.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
