"""cellsim — top-down simulation that DOESN'T drift, because measured data checkpoints hold it on the rails.

Pure forward simulation of a cell drifts into fantasy (we proved it: the free-running GRN lands in arbitrary
attractors, and knockout-fragility predicts essentiality at chance). The fix isn't more physics — it's what weather
models do: run the dynamics forward but ASSIMILATE measured observations at checkpoints so the trajectory can't
wander off the truth. This asks, honestly: given a few measured "checkpoint" genes of a cell's perturbation
response, can we reconstruct the REST — and does the mechanistic SIMULATION earn its keep, or do the data anchors
do all the work?

Task: hold out a perturbation (a knockout), reveal a fraction K of its response genes as CHECKPOINTS (measured
anchors), predict the unrevealed genes. Three predictors, same checkpoints, same held-out genes:
  MECH  — a linear regulatory-graph SIMULATION: seed the knocked-out gene, propagate the perturbation through the
          signed edges, CLAMP the checkpoint genes to their measured values every step (data assimilation), read
          off the rest. This is the top-down simulation, anchored.
  DATA  — low-rank completion from the measured co-response structure (SVD basis learned on OTHER perturbations),
          fit to the checkpoints, reconstruct the rest. No mechanism — pure "the data helps."
  HYBRID— average of the two.
Baseline: predict the training mean response (no checkpoints used). Metric: Pearson r(pred, truth) on the
UNREVEALED genes, averaged over held-out perturbations, swept over K.

-> outputs/orphan/cellsim.json + verdict. Honest by construction: if MECH never beats DATA, we say the simulation
   is a scaffold and the checkpoints are the signal.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "orphan")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def main(n_holdout=150, rank=60, iters=30, Ks=(0.0, 0.1, 0.25, 0.5), seed=0):
    import cellos
    from grn import GRN
    from complete_cell import CompleteCell
    C = CompleteCell()
    g = GRN(C=C)
    k = cellos.CellKernel(quiet=True); dbg = k._load_debugger()
    M, pgenes, syms = dbg["M"], dbg["pgenes"], dbg["syms"]

    # restrict to response genes that live in the regulatory core (so MECH and DATA compete on the SAME genes)
    core_name = {str(C.name[int(x)]): int(x) for x in g.nodes}
    cols = [j for j, s in enumerate(syms) if s in core_name]
    Mc = M[:, cols].astype(np.float32)                       # perturbations x core-genes
    cgenes = [syms[j] for j in cols]
    gi = {s: j for j, s in enumerate(cgenes)}                # core-gene name -> local col
    nG = len(cgenes)

    # signed row-normalized regulatory operator among the core response genes (target x regulator), for MECH
    import scipy.sparse as sp
    gl = {int(x): loc for loc, x in enumerate(g.nodes)}      # global gene idx -> GRN local
    keep = np.array([gl[core_name[s]] for s in cgenes])      # GRN-local index of each core response gene
    W = g.W[keep][:, keep]                                   # sub-operator on the response-gene subspace
    W = W.tocsr()
    has_reg = np.asarray((W != 0).sum(1)).ravel() > 0        # genes the mechanism can actually predict (have a regulator)

    # perturbations we can hold out: KO gene must be a core response gene (so MECH can seed + self-check)
    ho_pool = [i for i, p in enumerate(pgenes) if p in gi]
    rng = np.random.RandomState(seed)
    holdout = list(rng.choice(ho_pool, min(n_holdout, len(ho_pool)), replace=False))
    train_mask = np.ones(len(pgenes), bool); train_mask[holdout] = False
    Mtr = Mc[train_mask]
    mu = Mtr.mean(0)                                          # training column means (center)
    U, S, Vt = np.linalg.svd(Mtr - mu, full_matrices=False)
    V = Vt[:rank].T                                          # genes x rank  (measured co-response basis)

    def data_complete(obs_idx, obs_val):
        """low-rank fit to the checkpoints, reconstruct all genes (the DATA helper)."""
        A = V[obs_idx]                                        # (n_obs x rank)
        z, *_ = np.linalg.lstsq(A, obs_val - mu[obs_idx], rcond=None)
        return mu + V @ z

    def mech_sim(p_local, obs_idx, obs_val):
        """linear regulatory SIMULATION with data assimilation: seed the KO, propagate, clamp checkpoints."""
        r = np.zeros(nG, np.float32)
        obs = np.zeros(nG, bool); obs[obs_idx] = True
        r[obs_idx] = obs_val
        if not obs[p_local]:
            r[p_local] = -1.0                                # knocked-out gene is down
        for _ in range(iters):
            prop = W @ r                                      # one propagation step through signed edges
            r = 0.5 * r + 0.5 * prop
            r[obs_idx] = obs_val                             # re-clamp checkpoints (assimilation)
            if not obs[p_local]:
                r[p_local] = -1.0
        return r

    rows = []
    for K in Ks:
        rd, rm, rmf, rh, rb = [], [], [], [], []
        for i in holdout:
            truth = Mc[i]
            p_local = gi[pgenes[i]]
            nobs = int(K * nG)
            perm = np.random.RandomState(1000 + i).permutation(nG)
            # never "reveal" the perturbed gene itself; keep it as something to predict-through
            perm = perm[perm != p_local]
            obs_idx = perm[:nobs]
            hid = perm[nobs:]                                # evaluate on unrevealed genes only
            if len(hid) < 5:
                continue
            ov = truth[obs_idx]
            pd = data_complete(obs_idx, ov)
            pm = mech_sim(p_local, obs_idx, ov)
            ph = 0.5 * (pd + pm)
            rd.append(pearson(pd[hid], truth[hid]))
            rm.append(pearson(pm[hid], truth[hid]))
            hid_reg = hid[has_reg[hid]]                       # FAIR: score MECH only where it has a regulator to use
            if len(hid_reg) >= 5:
                rmf.append(pearson(pm[hid_reg], truth[hid_reg]))
            rh.append(pearson(ph[hid], truth[hid]))
            rb.append(pearson(mu[hid], truth[hid]))          # baseline: training mean response
        rows.append(dict(K=K, n_checkpoints=int(K * nG), r_data=float(np.nanmean(rd)),
                         r_mech=float(np.nanmean(rm)), r_mech_fair=float(np.nanmean(rmf)) if rmf else float("nan"),
                         r_hybrid=float(np.nanmean(rh)), r_baseline=float(np.nanmean(rb))))

    print(f"CHECKPOINT-ANCHORED CELL SIMULATION — reconstruct a held-out knockout's response ({len(holdout)} held out,"
          f" {nG} core genes, rank {rank})\n")
    print(f"  {'checkpoints':>12} {'MECH sim':>9} {'MECH fair':>10} {'DATA fill':>10} {'HYBRID':>8} {'baseline':>9}")
    print(f"  {'-'*62}")
    for r in rows:
        print(f"  {r['K']*100:>4.0f}% ({r['n_checkpoints']:>4}) {r['r_mech']:>9.3f} {r['r_mech_fair']:>10.3f} "
              f"{r['r_data']:>10.3f} {r['r_hybrid']:>8.3f} {r['r_baseline']:>9.3f}")

    # verdict: does assimilating checkpoints help, and does MECH ever beat DATA?
    withcp = [r for r in rows if r["K"] > 0]
    best_data = max(r["r_data"] for r in withcp)
    best_mech = max(r["r_mech"] for r in withcp)
    base0 = rows[0]["r_baseline"]
    mech_wins = any(r["r_mech"] > r["r_data"] + 0.02 for r in withcp)
    hybrid_wins = any(r["r_hybrid"] > max(r["r_data"], r["r_mech"]) + 0.01 for r in withcp)
    cp_helps = best_data > base0 + 0.05 or best_mech > base0 + 0.05
    if cp_helps and hybrid_wins:
        verdict = (f"CHECKPOINTS RESCUE THE SIM, AND MECHANISM ADDS SIGNAL: data anchors lift reconstruction to "
                   f"r={best_data:.2f}, and the mechanistic simulation contributes on top (hybrid beats both). "
                   f"Assimilating measured checkpoints into a forward sim genuinely works.")
    elif cp_helps and mech_wins:
        verdict = (f"CHECKPOINTS RESCUE THE SIM, AND MECHANISM IS COMPETITIVE: anchored simulation reaches "
                   f"r={best_mech:.2f}, matching/beating pure data completion — the regulatory dynamics carry the "
                   f"checkpoint signal usefully.")
    elif cp_helps:
        verdict = (f"CHECKPOINTS ARE THE SIGNAL, MECHANISM IS SCAFFOLD: data anchors lift reconstruction to "
                   f"r={best_data:.2f} (from baseline {base0:.2f}), but the regulatory simulation does NOT beat pure "
                   f"data completion (mech {best_mech:.2f}). The sim runs on the rails the data lays — honest.")
    else:
        verdict = (f"LIMITED: even with checkpoints, reconstruction stays near baseline (data {best_data:.2f}, "
                   f"mech {best_mech:.2f}, base {base0:.2f}).")
    print("\n" + "=" * 78 + f"\nVERDICT: {verdict}\n" + "=" * 78)
    json.dump({"rows": rows, "n_holdout": len(holdout), "n_core_genes": nG, "rank": rank, "verdict": verdict},
              open(f"{OUT}/cellsim.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
