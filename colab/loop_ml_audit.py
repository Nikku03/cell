"""LOOP 132 -- AUDIT LOOP 131: is the gain real, was the MLP mistreated, and what did the model learn?

WHY THIS EXISTS. Loop 131 reported RMSE 1.3663 against a constant's 1.504 and called it a pass. It
has no error bar. This repository has spent a session learning that a number without its own noise
attached is not a result -- twelve gates fired while measuring nothing, and loop 120 passed two
gates on margins smaller than their own spread. Reporting a 9% improvement with no test of whether
9% is inside the noise would be the same mistake with a neural network attached to it.

Three things are unresolved and two more become obvious once you look at the numbers.

  THE UNIT OF RESAMPLING IS THE CLUSTER, NOT THE RECORD. 17,004 records sit on 7,856 sequences in
  3,006 clusters, so records are not independent -- a bootstrap over records would treat 17,004
  correlated observations as 17,004 independent ones and produce a confidence interval several
  times too narrow. Everything below resamples CLUSTERS.

  AND NESTED CV, OR THE MLP COMPARISON IS RIGGED. Tuning a network on the same folds it is scored
  on is leakage, and it is how the retested model always wins. Early stopping and configuration
  choice happen on an INNER split of the training folds only.

PREDECLARED:

  A1 IS THE GAIN SIGNIFICANT AT ALL?                                THE MISSING ERROR BAR.
       per-fold RMSE for the model and the constant, then a cluster bootstrap on the difference.
       Gate: the 95% interval on (constant - model) must exclude zero. If it does not, loop 131's
       headline is noise and the record says so.
  A2 AND AGAINST THE BEST BASELINE?                                 THE HARDER VERSION.
       the same test against EC median at 1.5015, which is the strongest thing that is not a model.
       Gate: 95% interval excludes zero.
  A3 THE MLP, TRAINED PROPERLY                                      THE FAIR RETEST.
       loop 131 gave it 40 fixed epochs, no early stopping and no schedule, then reported it losing.
       Here it gets an inner validation split, early stopping, a cosine schedule and a three-point
       configuration sweep -- all chosen inside the training folds. Gate: report whether a properly
       trained MLP closes the gap to gradient boosting. Passing means the comparison was made
       fairly, not that the MLP won.
  A4 DOES THE SPLIT'S DRIFT CHANGE THE ANSWER?                      LOOP 131's M1 FAILURE.
       fold 0 drifted +0.207 against a 0.200 gate. Clusters are re-assigned to folds by greedy
       balancing on their target sums, and the best model re-run. Gate: drift under 0.2 AND the
       RMSE moves by less than the A1 bootstrap interval -- the conclusion must not depend on which
       clusters landed where.
  A5 DID IT LEARN CHEMISTRY OR A PER-PROTEIN AVERAGE?               THE QUESTION NOBODY ASKED.
       shuffle the substrate fingerprints WITHIN each protein, so every record keeps its enzyme and
       gets a different substrate from the same enzyme's own set. If performance survives, the model
       is predicting a per-protein constant and the substrate channel is decoration. Gate: shuffling
       must degrade RMSE beyond the bootstrap interval.
  A6 THE CEILING NOBODY CAN BEAT                                    THE VARIANCE DECOMPOSITION.
       how much of the variance in log10 k_cat is BETWEEN proteins and how much is WITHIN one
       protein across its substrates. A model that predicts one number per protein is capped at the
       between-protein share, and that cap is a property of the data rather than of any method.
       Gate: report both, and the implied ceiling for a per-protein predictor.

-> outputs/loop_ml_audit.json
"""
import collections
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(1 << 30)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest as RM  # noqa: E402

OUT = Path(os.environ.get("CELL_OUT", "outputs"))
ML = Path("colab/data/ml")
SEED = 13200
N_FOLDS = 5
N_BOOT = 2000
CONST_RMSE = 1.504
FLOOR = float(np.log10(1.15))

log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cluster_boot(y, pa, pb, clu, rng, n=N_BOOT):
    """95% interval on RMSE(pa) - RMSE(pb), resampling CLUSTERS because records are not independent."""
    idx_by = collections.defaultdict(list)
    for i, c in enumerate(clu):
        idx_by[c].append(i)
    keys = list(idx_by)
    d = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        ii = np.concatenate([idx_by[keys[k]] for k in pick])
        d.append(rmse(y[ii], pa[ii]) - rmse(y[ii], pb[ii]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    say("=" * 100)
    say("  LOOP 132 -- audit loop 131: is the gain real, was the MLP mistreated, what was learned?")
    say("=" * 100)
    say()

    rows = list(csv.DictReader(open(ML / "kcat_records.tsv"), delimiter="\t"))
    y = np.array([float(r["log10_kcat"]) for r in rows])
    seq_id = np.array([int(r["seq_id"]) for r in rows])
    smi_id = np.array([int(r["smiles_id"]) for r in rows])
    fold = np.array([int(r["fold"]) for r in rows])
    clu = np.array([int(r["cluster_id"]) for r in rows])
    SF = np.load(ML / "seq_features.npy")
    FP = np.unpackbits(np.load(ML / "substrate_ecfp.npy"), axis=1).astype(np.float32)
    E = np.load(ML / "esm2_8M_mean.npy").astype(np.float32)
    X = np.hstack([SF[seq_id], E[seq_id], FP[smi_id]])
    say(f"  {len(rows):,} records | {len(set(seq_id.tolist())):,} sequences | "
        f"{len(set(clu.tolist())):,} clusters | X {X.shape}")
    say(f"  records per cluster: median {np.median(list(collections.Counter(clu.tolist()).values())):.0f}, "
        f"max {max(collections.Counter(clu.tolist()).values())}")
    say()

    import xgboost as xgb

    def mk():
        return xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8,
                                colsample_bytree=0.5, reg_lambda=2.0, n_jobs=4,
                                random_state=SEED, verbosity=0)

    def cv(Xm, f=fold):
        p = np.zeros(len(y))
        for k in range(N_FOLDS):
            te, tr = f == k, f != k
            m = mk()
            m.fit(Xm[tr], y[tr])
            p[te] = m.predict(Xm[te])
        return p

    gates = {}
    say("  refitting the best model (XGB all) under the committed split")
    pm = cv(X)
    pc = np.zeros(len(y))
    pe = np.zeros(len(y))
    ec = np.array([r["ec"] for r in rows])
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        pc[te] = np.median(y[tr])
        med = collections.defaultdict(list)
        for i in np.flatnonzero(tr):
            if ec[i]:
                med[ec[i]].append(y[i])
        gm = np.median(y[tr])
        pe[te] = [np.median(med[ec[i]]) if ec[i] in med else gm for i in np.flatnonzero(te)]
    say(f"    model {rmse(y, pm):.4f} | constant {rmse(y, pc):.4f} | EC median {rmse(y, pe):.4f}")
    say()

    # ---------------------------------------------------------------- A1
    say("A1 IS THE GAIN SIGNIFICANT AT ALL?")
    for k in range(N_FOLDS):
        m = fold == k
        say(f"     fold {k}: n={int(m.sum()):>6,}  model {rmse(y[m], pm[m]):.4f}  "
            f"constant {rmse(y[m], pc[m]):.4f}  delta {rmse(y[m], pc[m]) - rmse(y[m], pm[m]):+.4f}")
    per = [rmse(y[fold == k], pc[fold == k]) - rmse(y[fold == k], pm[fold == k])
           for k in range(N_FOLDS)]
    say(f"     per-fold gain: mean {np.mean(per):+.4f}, sd {np.std(per):.4f}, "
        f"all five positive: {all(p > 0 for p in per)}")
    d, lo, hi = cluster_boot(y, pc, pm, clu, rng)
    say(f"     CLUSTER bootstrap on (constant - model): {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]")
    say(f"     resampling clusters, not records -- 17,004 records on 3,006 clusters are not "
        f"17,004 independent observations")
    gates["A1"] = bool(lo > 0)
    say(f"     A1 {'PASS' if gates['A1'] else 'FAIL'} -- the interval "
        f"{'excludes zero' if gates['A1'] else 'INCLUDES ZERO; the headline is noise'}")
    say()

    # ---------------------------------------------------------------- A2
    say("A2 AND AGAINST THE BEST BASELINE?")
    d2, lo2, hi2 = cluster_boot(y, pe, pm, clu, rng)
    say(f"     (EC median - model): {d2:+.4f}  95% [{lo2:+.4f}, {hi2:+.4f}]")
    gates["A2"] = bool(lo2 > 0)
    say(f"     A2 {'PASS' if gates['A2'] else 'FAIL'}")
    say()

    # ---------------------------------------------------------------- A3
    say("A3 THE MLP, TRAINED PROPERLY")
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    CFG = [(512, 128, 1e-3, 0.3), (1024, 256, 5e-4, 0.4), (256, 64, 2e-3, 0.2)]

    def train_mlp(Xtr, ytr, Xva, yva, h1, h2, lr, dr, max_ep=120, patience=12):
        net = nn.Sequential(nn.Linear(Xtr.shape[1], h1), nn.ReLU(), nn.Dropout(dr),
                            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dr / 2),
                            nn.Linear(h2, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
        Xt = torch.tensor(Xtr)
        yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(Xva)
        best, best_state, bad = 1e9, None, 0
        for ep in range(max_ep):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                b = perm[i:i + 256]
                opt.zero_grad()
                nn.functional.mse_loss(net(Xt[b]), yt[b]).backward()
                opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                v = rmse(yva, net(Xv).squeeze(1).numpy())
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        return net, best, ep + 1

    pn = np.zeros(len(y))
    chosen = []
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        trc = np.array(sorted(set(clu[tr].tolist())))
        rng.shuffle(trc)
        vac = set(trc[:max(1, len(trc) // 5)].tolist())     # inner split BY CLUSTER
        inv = np.array([c in vac for c in clu]) & tr
        ini = tr & ~inv
        best_cfg, best_v = None, 1e9
        for cfg in CFG:
            _, v, ep = train_mlp(Xn[ini], y[ini], Xn[inv], y[inv], *cfg)
            if v < best_v:
                best_v, best_cfg = v, cfg
        net, v, ep = train_mlp(Xn[ini], y[ini], Xn[inv], y[inv], *best_cfg)
        chosen.append((best_cfg, ep, v))
        with torch.no_grad():
            pn[te] = net(torch.tensor(Xn[te])).squeeze(1).numpy()
        say(f"     fold {k}: cfg h1={best_cfg[0]} lr={best_cfg[2]} dr={best_cfg[3]}, "
            f"stopped ep {ep}, inner RMSE {v:.4f}")
    say(f"     tuned MLP  RMSE {rmse(y, pn):.4f}  R2 {r2(y, pn):+.4f}")
    say(f"     loop 131's untuned MLP 1.4374 | gradient boosting {rmse(y, pm):.4f}")
    d3, lo3, hi3 = cluster_boot(y, pn, pm, clu, rng)
    say(f"     (MLP - XGB): {d3:+.4f}  95% [{lo3:+.4f}, {hi3:+.4f}]  "
        f"{'XGB still ahead' if lo3 > 0 else 'they are indistinguishable' if lo3 <= 0 <= hi3 else 'MLP ahead'}")
    say(f"     configuration and stopping epoch chosen on an INNER cluster split of the training "
        f"folds only -- never on the fold being scored")
    gates["A3"] = True
    say(f"     A3 PASS -- the comparison is now fair; whether the MLP won is the finding, "
        f"not the gate")
    say()

    # ---------------------------------------------------------------- A4
    say("A4 DOES THE SPLIT'S DRIFT CHANGE THE ANSWER?")
    # THE FIRST VERSION OF THIS WAS BROKEN AND THE GATE CAUGHT IT. Greedy longest-processing-time
    # on cluster target SUMS balances sums, not means: a fold that collects many small clusters and
    # one that collects few large ones can match on total while their means diverge wildly. It
    # produced fold means of -0.778 and +0.642, four times WORSE than the 0.207 it was meant to fix,
    # and the model then scored 1.6040 because fold 0 held all the slow enzymes. Fixed to stratified
    # group k-fold: order clusters by their own mean target and deal them out in a snake pattern, so
    # each fold receives a matched slice of the target distribution.
    cmean = {c: float(np.mean(y[clu == c])) for c in set(clu.tolist())}
    order = sorted(cmean, key=lambda c: cmean[c])
    assign = {}
    for i, c in enumerate(order):
        blk, pos = divmod(i, N_FOLDS)
        assign[c] = pos if blk % 2 == 0 else N_FOLDS - 1 - pos      # snake, not round-robin
    f2 = np.array([assign[c] for c in clu])
    dr2 = [float(y[f2 == k].mean() - y.mean()) for k in range(N_FOLDS)]
    say(f"     rebalanced fold means: " + "  ".join(f"{v:+.3f}" for v in dr2))
    say(f"     worst drift {max(abs(v) for v in dr2):.3f} against the committed split's 0.207")
    pm2 = cv(X, f2)
    say(f"     model under the rebalanced split: RMSE {rmse(y, pm2):.4f} "
        f"against {rmse(y, pm):.4f}")
    shift = abs(rmse(y, pm2) - rmse(y, pm))
    say(f"     shift {shift:.4f}; the A1 interval is {hi - lo:.4f} wide")
    gates["A4"] = bool(max(abs(v) for v in dr2) < 0.2 and shift < (hi - lo))
    say(f"     A4 {'PASS' if gates['A4'] else 'FAIL'} -- the conclusion "
        f"{'does not depend on which clusters landed where' if gates['A4'] else 'DOES depend on the split'}")
    say()

    # ---------------------------------------------------------------- A5
    say("A5 DID IT LEARN CHEMISTRY OR A PER-PROTEIN AVERAGE?")
    by_seq = collections.defaultdict(list)
    for i, s in enumerate(seq_id):
        by_seq[s].append(i)
    perm_idx = np.arange(len(y))
    multi = 0
    for s, ii in by_seq.items():
        if len(ii) > 1:
            multi += len(ii)
            sh = rng.permutation(ii)
            perm_idx[np.array(ii)] = sh
    Xs = np.hstack([SF[seq_id], E[seq_id], FP[smi_id[perm_idx]]])
    say(f"     {multi:,} records sit on a protein with more than one substrate and can be shuffled")
    ps = cv(Xs)
    say(f"     substrate shuffled WITHIN protein: RMSE {rmse(y, ps):.4f} "
        f"against the real {rmse(y, pm):.4f}")
    d5, lo5, hi5 = cluster_boot(y, ps, pm, clu, rng)
    say(f"     (shuffled - real): {d5:+.4f}  95% [{lo5:+.4f}, {hi5:+.4f}]")
    gates["A5"] = bool(lo5 > 0)
    say(f"     A5 {'PASS' if gates['A5'] else 'FAIL'} -- the substrate channel "
        f"{'carries real enzyme-substrate information' if gates['A5'] else 'IS DECORATION: the model predicts a per-protein constant'}")
    say()

    # ---------------------------------------------------------------- A6
    say("A6 THE CEILING NOBODY CAN BEAT")
    means = {s: float(np.mean(y[ii])) for s, ii in by_seq.items()}
    within = np.array([y[i] - means[seq_id[i]] for i in range(len(y))])
    between = np.array([means[s] - y.mean() for s in seq_id])
    vw, vb = float(within.var()), float(between.var())
    say(f"     total variance {y.var():.4f} = between-protein {vb:.4f} + within-protein {vw:.4f}")
    say(f"     between-protein share {vb / (vb + vw):.1%}, within-protein {vw / (vb + vw):.1%}")
    say(f"     a PERFECT per-protein predictor -- one number per enzyme, ignoring the substrate --")
    say(f"     would score RMSE {np.sqrt(vw):.4f}, and no such model can do better")
    say(f"     the experimental floor is {FLOOR:.4f}, so most of the within-protein spread is real")
    say(f"     biology across substrates, not measurement noise")
    say(f"     for scale: this model is at {rmse(y, pm):.4f} and a constant at {CONST_RMSE:.3f}")
    gates["A6"] = bool(np.isfinite(vw) and vw > 0)
    say(f"     A6 {'PASS' if gates['A6'] else 'FAIL'}")
    say()

    say("=" * 100)
    for k in ("A1", "A2", "A3", "A4", "A5", "A6"):
        say(f"  {k}  {'PASS' if gates[k] else 'FAIL'}")
    say(f"  {sum(gates.values())}/6")
    say("=" * 100)

    man = RM.manifest(inputs=[ML / "kcat_records.tsv", ML / "esm2_8M_mean.npy",
                              ML / "substrate_ecfp.npy", ML / "seq_features.npy"],
                      available=len(rows), used=len(rows), selection="all", seed=SEED,
                      controls=["bootstrap over CLUSTERS, because records are not independent",
                                "nested CV: MLP configuration and early stopping on an inner "
                                "cluster split of the training folds only",
                                "a target-balanced re-split to test whether the answer moves",
                                "substrate shuffled WITHIN protein to separate chemistry from a "
                                "per-protein average",
                                "the between/within variance decomposition as a method-free ceiling",
                                "per-fold results reported, not only the pooled number"],
                      note="loop 131 reported a 9% gain with no error bar; this attaches one")
    RM.report(man, emit=say)
    json.dump({"test": "loop_ml_audit", "manifest": man, "gates": gates,
               "model_rmse": rmse(y, pm), "constant_rmse": rmse(y, pc),
               "ec_rmse": rmse(y, pe), "mlp_tuned_rmse": rmse(y, pn),
               "a1": {"per_fold_gain": per, "boot_mean": d, "boot_lo": lo, "boot_hi": hi},
               "a2": {"boot_mean": d2, "boot_lo": lo2, "boot_hi": hi2},
               "a3": {"tuned_rmse": rmse(y, pn), "untuned_rmse": 1.4374,
                      "vs_xgb": [d3, lo3, hi3],
                      "configs": [[list(c), e, v] for c, e, v in chosen]},
               "a4": {"drift": dr2, "rmse_rebalanced": rmse(y, pm2), "shift": shift},
               "a5": {"shuffled_rmse": rmse(y, ps), "delta": [d5, lo5, hi5], "n_shufflable": multi},
               "a6": {"var_total": float(y.var()), "var_between": vb, "var_within": vw,
                      "between_share": vb / (vb + vw),
                      "per_protein_ceiling_rmse": float(np.sqrt(vw))},
               "seconds": time.time() - t0, "log": log},
              open(OUT / "loop_ml_audit.json", "w"), indent=1)
    say(f"\n  -> {OUT / 'loop_ml_audit.json'}   [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
