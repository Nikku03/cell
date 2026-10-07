"""DOES A LEARNED MODEL EXTRACT SIGNAL THE SINGLE FEATURES MISSED?  Neural re-rankers on the docked pairs.

WHAT THIS ANSWERS.  `nexus_catalyst_pilot` scored eight docking statistics ONE AT A TIME and every one landed
between 0.450 and 0.549, with a clean size control at 0.532. But "no single feature separates them" is not the
same claim as "no function of these features separates them". Networks have earned real margins elsewhere in
this project -- GeoConv added +0.07 over the best hand-built feature on interface point pairs -- so the
combination deserves its own test rather than an assumption.

THE TASK IS LISTWISE, which is why a set encoder is the right shape.  Each reaction gives ten candidates of
which exactly one is the true catalyst. Scoring candidates independently cannot express "the most
shape-complementary one AMONG THESE TEN"; attention over the group can. That is the same argument that
motivated the set transformer in `cell_orphan_ties_nn`, where it turned out not to help -- the argument being
sound does not make the result positive.

THE THING THAT WILL KILL A NAIVE READ HERE: n = 60 GROUPS.  Sixty reactions, eight features, and a flexible
model is a recipe for a confident-looking AUC that is pure fit. Three defences, all mandatory:

    GROUPED CV     5-fold by REACTION, so no reaction's candidates span train and test. Splitting by row
                   would let the model memorise a reaction's score scale and read the answer off it.
    UNTRAINED ARM  identical architecture, random init. It has decided several results in this project, twice
                   by exposing a below-chance floor that revealed a bug.
    PERMUTATION    labels shuffled WITHIN each group, model refit end to end, 50 times. This is the actual
                   null for "a flexible model on 60 groups", and it is not 0.5 -- it is whatever this much
                   capacity can extract from noise. Any arm must clear the permutation 95th percentile, not
                   clear 0.5.

ARMS
    best_single        the strongest of the eight, carried from the pilot -- what a learned model must beat
    size_only          the artefact control; if this moves, the decoy matching broke again
    logistic           all eight features, fitted. Separates "a weighted combination helps" from "capacity"
    mlp                per-candidate MLP, still scoring candidates independently
    set_transformer    attention across the ten candidates -- the only arm that can use group context
    *_untrained        random-init controls for both networks

PREDECLARED, before any number:
    every learned arm inside the permutation band
        -> the limit is the FEATURES, not the model. Eight statistics off a rigid-body scan do not contain
           which of ten enzymes is the catalyst, and no architecture conjures it. The structural route stays
           closed and the honest next move is a different representation, not a bigger head.
    set_transformer clears the band and beats logistic
        -> group context is the missing ingredient and the tie-break composition is live again.
    logistic clears the band but set_transformer does not
        -> a weighted combination helps; the extra capacity does not.
"""
import json
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adrn_ko_conjunctions as A

OUT = A.OUT
KEYS = ("best", "top10", "mean", "top50cell", "n_z2", "skew", "std", "clash")
SEEDS = (0, 1, 2, 3, 4)
N_PERM = 50
N_FOLD = 5
EPOCHS = 60
D_MODEL = 16
BEST_SINGLE = 0.498          # 'best' feature, from nexus_catalyst_pilot
SIZE_ONLY_REF = 0.532


def load():
    """groups of ten candidates, z-scored WITHIN each group. Absolute scale tracks protein size, which the
    pilot already showed is the artefact; the shape of the within-group spread is the only usable part."""
    d = {}
    for f in sorted(glob(str(OUT / "catalyst_pilot_features*.json"))):
        d.update(json.load(open(f)))
    X, Y, G, S = [], [], [], []
    for gi, (rx, rec) in enumerate(sorted(d.items())):
        cat, F = rec["catalyst"], rec["feats"]
        gs = sorted(F)
        if cat not in gs or len(gs) < 5:
            continue
        M = np.array([[F[g][k] for k in KEYS] for g in gs], float)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        sz = np.array([F[g]["n_atoms"] for g in gs], float)
        sz = (sz - sz.mean()) / (sz.std() + 1e-9)
        X.append(M)
        Y.append(np.array([1 if g == cat else 0 for g in gs]))
        G.append(gi)
        S.append(sz)
    return X, Y, np.array(G), S


def main():
    log, t0 = [], time.time()

    def report(x):
        print(x, flush=True)
        log.append(x)

    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    torch.set_num_threads(4)

    report("=" * 100)
    report("DOES A LEARNED MODEL BEAT THE SINGLE DOCKING FEATURES?")
    report("=" * 100)
    report(f"  The pilot scored eight statistics one at a time; all landed in [0.450, 0.549] with the size")
    report(f"  control at {SIZE_ONLY_REF:.3f}. 'No single feature works' is not 'no function of them works'.")
    report("  n = 60 GROUPS, so the null for a flexible model is NOT 0.5. Every arm is judged against a")
    report("  within-group label permutation refit end to end 50 times, plus an untrained twin.")

    X, Y, G, S = load()
    ng = len(X)
    report(f"\n  {ng} reactions x {len(X[0])} candidates | {len(KEYS)} features, z-scored within group")
    if ng < 20:
        report("  TOO FEW GROUPS to fit anything. Not a negative result -- an empty one.")
        return 1

    class SetRank(nn.Module):
        """attention over the ten candidates: the only arm that can express 'best AMONG THESE'."""
        def __init__(s, fd, d=D_MODEL, attend=True):
            super().__init__()
            s.attend = attend
            s.proj = nn.Linear(fd, d)
            s.ln = nn.LayerNorm(d)
            s.attn = nn.MultiheadAttention(d, 2, batch_first=True)
            s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            s.out = nn.Linear(d, 1)

        def forward(s, x):
            h = s.proj(x).unsqueeze(0)
            if s.attend:
                a, _ = s.attn(s.ln(h), s.ln(h), s.ln(h), need_weights=False)
                h = h + a
            h = h + s.ff(s.ln(h))
            return s.out(h.squeeze(0)).squeeze(-1)

    def fit_predict(tr, te, ys, seed, attend, train=True):
        torch.manual_seed(seed)
        m = SetRank(len(KEYS), attend=attend)
        if train:
            opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
            order = list(tr)
            rs = np.random.default_rng(seed)
            for _ in range(EPOCHS):
                rs.shuffle(order)
                m.train()
                for g in order:
                    if ys[g].sum() == 0:
                        continue
                    opt.zero_grad()
                    sc = m(torch.tensor(X[g], dtype=torch.float32))
                    loss = -torch.log_softmax(sc, 0)[torch.tensor(ys[g]).bool()].mean()
                    loss.backward()
                    opt.step()
        m.eval()
        out = {}
        with torch.no_grad():
            for g in te:
                out[g] = m(torch.tensor(X[g], dtype=torch.float32)).numpy()
        return out

    def run_cv(ys, seed, arm):
        """grouped CV; returns pooled scores aligned to pooled labels"""
        gk = GroupKFold(n_splits=N_FOLD)
        idx = np.arange(ng)
        pred = [None] * ng
        for tr, te in gk.split(idx, groups=G):
            if arm == "logistic":
                Xtr = np.vstack([X[g] for g in tr])
                ytr = np.concatenate([ys[g] for g in tr])
                lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
                for g in te:
                    pred[g] = lr.predict_proba(X[g])[:, 1]
            else:
                o = fit_predict(tr, te, ys, seed,
                                attend=arm.startswith("set"), train="untrained" not in arm)
                for g in te:
                    pred[g] = o[g]
        y = np.concatenate([ys[g] for g in range(ng)])
        p = np.concatenate([pred[g] for g in range(ng)])
        top1 = float(np.mean([ys[g][int(np.argmax(pred[g]))] == 1 for g in range(ng)]))
        return float(roc_auc_score(y, p)), top1

    # THE ONE-RULE BASELINE. The pilot noticed the true catalyst's within-group z has sd 0.731 rather than
    # ~1.0 -- it sits closer to mid-pack than a random member. A network handed within-group z-scores can
    # learn exactly that rule and nothing else. If "prefer the least extreme candidate" alone matches the
    # networks, then that is what they found, and it is one post-hoc regularity on 60 groups rather than
    # chemistry. Parameter-free, so it cannot overfit and needs no CV.
    y_all = np.concatenate([Y[g] for g in range(ng)])
    mid = np.concatenate([-np.abs(X[g][:, KEYS.index("best")]) for g in range(ng)])
    auc_mid = float(roc_auc_score(y_all, mid))
    top1_mid = float(np.mean([Y[g][int(np.argmax(-np.abs(X[g][:, KEYS.index("best")])))] == 1
                              for g in range(ng)]))
    midall = np.concatenate([-np.abs(X[g]).mean(1) for g in range(ng)])
    auc_mid_all = float(roc_auc_score(y_all, midall))

    ARMS = ("logistic", "mlp", "mlp_untrained", "set_transformer", "set_transformer_untrained")
    report(f"\n  {'arm':<28}{'AUC':>8}{'sd':>7}{'top-1':>8}   ({N_FOLD}-fold grouped CV x {len(SEEDS)} seeds)")
    res = {}
    for arm in ARMS:
        a, t = zip(*[run_cv(Y, s, arm) for s in SEEDS])
        res[arm] = {"auc": float(np.mean(a)), "sd": float(np.std(a)), "top1": float(np.mean(t))}
        report(f"  {arm:<28}{np.mean(a):>8.3f}{np.std(a):>7.3f}{np.mean(t):>8.3f}")
    res["midpack_best"] = {"auc": auc_mid, "sd": 0.0, "top1": top1_mid}
    res["midpack_allfeat"] = {"auc": auc_mid_all, "sd": 0.0, "top1": 0.0}
    report(f"  {'midpack (-|z| of best)':<28}{auc_mid:>8.3f}{'-':>7}{top1_mid:>8.3f}   <- ONE RULE, 0 params")
    report(f"  {'midpack (-mean|z| all feats)':<28}{auc_mid_all:>8.3f}{'-':>7}{'-':>8}   <- ONE RULE, 0 params")
    report(f"  {'best_single (pilot)':<28}{BEST_SINGLE:>8.3f}{'-':>7}{'-':>8}")
    report(f"  {'size_only (pilot control)':<28}{SIZE_ONLY_REF:>8.3f}{'-':>7}{'-':>8}   <- artefact control")

    # ---------------- the null that actually applies ----------------
    report(f"\n  PERMUTATION NULL -- labels shuffled WITHIN each group, model refit end to end, "
           f"{N_PERM} times.")
    report("  This is the null for THIS much capacity on 60 groups. It is not 0.5, and 0.5 is not the bar.")
    rs = np.random.default_rng(0)
    perm = {}
    for arm in ("logistic", "set_transformer"):
        vals = []
        for k in range(N_PERM):
            ysh = []
            for g in range(ng):
                v = Y[g].copy()
                rs.shuffle(v)
                ysh.append(v)
            vals.append(run_cv(ysh, int(rs.integers(10 ** 6)), arm)[0])
        v = np.array(vals)
        perm[arm] = {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                     "p_value": float(np.mean(v >= res[arm]["auc"]))}
        report(f"    {arm:<24} median {np.median(v):.3f}  95th {np.percentile(v, 95):.3f}  "
               f"| observed {res[arm]['auc']:.3f}  p={perm[arm]['p_value']:.3f}")

    report("\n  READING")
    st, lg = res["set_transformer"], res["logistic"]
    mx_mid = max(auc_mid, auc_mid_all)
    clears = [a for a in ("logistic", "set_transformer")
              if res[a]["auc"] > perm[a]["p95"] and perm[a]["p_value"] < 0.05]
    if not clears:
        report("  THE LIMIT IS THE FEATURES, NOT THE MODEL. No learned arm clears its own permutation band:")
        report(f"    logistic {lg['auc']:.3f} vs 95th {perm['logistic']['p95']:.3f} (p="
               f"{perm['logistic']['p_value']:.3f})")
        report(f"    set transformer {st['auc']:.3f} vs 95th {perm['set_transformer']['p95']:.3f} (p="
               f"{perm['set_transformer']['p_value']:.3f})")
        report("  Eight statistics off a rigid-body scan do not contain which of ten enzymes is the catalyst,")
        report("  and neither a weighted combination nor attention across the group conjures it. Capacity was")
        report("  never the bottleneck here -- the representation is. A bigger head on the same features is")
        report("  not the next move; a different measurement is.")
    else:
        report(f"  {clears} clear the permutation band.")
        if st["auc"] <= mx_mid + 0.02:
            report(f"  BUT IT IS THE ONE RULE, NOT CHEMISTRY. A parameter-free 'prefer the least extreme")
            report(f"  candidate' scores {mx_mid:.3f} against the network's {st['auc']:.3f}. The network")
            report("  rediscovered the mid-pack regularity the pilot already flagged as a post-hoc oddity on")
            report("  60 groups -- it is a property of how these decoys were drawn, not of which enzyme")
            report("  catalyses the reaction, and it would not survive a fresh decoy set.")
        else:
            report(f"  And it beats the one-rule baseline ({mx_mid:.3f}), so it is not merely the mid-pack")
            report("  regularity. Read against the untrained twins, then re-run the composition step.")
    # ---------------- THE DECISIVE CONTROL: does the rule survive a DIFFERENT decoy draw? ----------------
    # Run 1 of the pilot drew a different decoy set for the same 60 reactions (matched on sequence length
    # rather than trimmed atoms). It is size-confounded and unusable as a biology result, but it is a
    # genuinely independent sample of decoys, which is exactly what is needed to ask whether the mid-pack
    # regularity is a property of CATALYSTS or of one draw.
    alt = sorted(glob(str(OUT / "decoy_draw1" / "catalyst_pilot_features*.json")))
    if alt:
        da = {}
        for f in alt:
            da.update(json.load(open(f)))
        ya, ma, zv = [], [], []
        for rx, rec in da.items():
            cat, F = rec["catalyst"], rec["feats"]
            gs = sorted(F)
            if cat not in gs or len(gs) < 5:
                continue
            b = np.array([F[g]["best"] for g in gs], float)
            z = (b - b.mean()) / (b.std() + 1e-9)
            ya += [1 if g == cat else 0 for g in gs]
            ma += list(-np.abs(z))
            zv.append(z[gs.index(cat)])
        auc_alt = float(roc_auc_score(np.array(ya), np.array(ma)))
        sd_alt = float(np.std(zv))
        sd_this = float(np.std([X[g][int(np.argmax(Y[g])), KEYS.index("best")] for g in range(ng)]))
        report(f"\n  DOES THE MID-PACK RULE SURVIVE A DIFFERENT DECOY DRAW? (run 1 of the pilot, same 60")
        report(f"  reactions, decoys drawn on a different criterion -- an independent sample of decoys)")
        report(f"    {'draw':<34}{'midpack AUC':>13}{'true-catalyst z sd':>22}")
        report(f"    {'this run (atom-matched)':<34}{mx_mid:>13.3f}{sd_this:>22.3f}")
        report(f"    {'run 1 (length-matched)':<34}{auc_alt:>13.3f}{sd_alt:>22.3f}")
        report(f"    A random member of a z-scored set of 10 has sd ~1.0.")
        if auc_alt < 0.55 and sd_alt > 0.9:
            report(f"    IT DOES NOT SURVIVE. On the other draw the rule is worth {auc_alt:.3f} and the true")
            report(f"    catalyst's z has sd {sd_alt:.3f} -- indistinguishable from a random member. So the")
            report("    mid-pack regularity is a property of WHICH DECOYS WERE SAMPLED, not of catalysts, and")
            report("    the network's apparent margin is fitted to one draw. It would not transfer.")
        else:
            report(f"    It reproduces ({auc_alt:.3f}, sd {sd_alt:.3f}), so the regularity is not draw-specific")
            report("    and deserves a pre-registered test of its own.")
        res["midpack_alt_draw"] = {"auc": auc_alt, "z_sd": sd_alt}

    report("\n  For contrast, where a network DID earn its margin in this project: GeoConv reached 0.617 on")
    report("  interface point pairs against a 0.566 best single feature -- but that task is scored on a")
    report("  complex that already exists. The model was not the difference; having the true interface was.")

    json.dump({"test": "nexus_catalyst_nn", "n_groups": ng, "arms": res, "permutation": perm,
               "best_single": BEST_SINGLE, "size_only": SIZE_ONLY_REF, "log": log},
              open(OUT / "nexus_catalyst_nn.json", "w"), indent=2)
    report(f"\n  total {time.time()-t0:.0f}s  -> {OUT/'nexus_catalyst_nn.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
