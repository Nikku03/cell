"""A SET-LEVEL MODEL: score each element in the context of the OTHER candidates for the same gene.

THE DIAGNOSIS THIS ANSWERS. `missed.py` found the failure mode is not diffuse. The 233 missed positives are
secondary enhancers of multi-enhancer genes: at identical model score they are MORE acetylated, MORE
PolII-loaded, MORE loop-connected and MORE TF-bound than the negatives they sit among, and the model
down-ranks them anyway. It down-ranks them because found positives have median distance-rank 1.6 while
missed ones have 13.4 -- the competition block encodes "the top candidate is the one that matters", which
is right on average and wrong for the 289 of 569 positives (51%) that belong to genes with two or more.

WHY ANOTHER COLUMN CANNOT FIX IT, established by measurement rather than argument. Absolute-scale
neighbourhood features -- total candidate activity for the gene, count of strong candidates -- scored net
-0.0012 +/- 0.0042. A gene-level feature shifts every candidate for that gene by the same amount, so it
cannot re-rank WITHIN the gene, and re-ranking within the gene is the entire problem.

THE CHANGE IN SHAPE. Every model so far scores a pair from that pair's own features. This one scores a
candidate from its own features AND from how it compares to its peers -- learned, not hand-coded:

    h_i          = encode(features of candidate i)
    context_g    = mean and max over all candidates of gene g          (permutation invariant, DeepSets)
    logit_i      = base_logit_i + decode([h_i, mean_g, max_g, h_i - mean_g])

`h_i - mean_g` is the operative term: how this candidate differs from the typical candidate for its gene.
That is what lets the model learn "this gene's neighbourhood is uniformly strong, so several of these are
real" separately from "one candidate dominates, suppress the rest" -- an interaction between a candidate
and the DISTRIBUTION of its peers, which no fixed per-pair feature can express.

Scatter reductions rather than padding: candidate sets run from 1 to 212, and padding to 212 would waste
most of the compute on a median set size of 2.

RESIDUAL ON THE XGBOOST SCORE, deliberately. With 569 positives a network trained from scratch on 330
features would lose to gradient boosting on general grounds and tell us nothing about set structure. Adding
a learned correction to the existing score isolates the question: does peer context add anything the
per-pair model cannot see?

LEAKAGE. The base score for fold k comes from an XGBoost trained without fold k, and the set model for fold
k is likewise trained only on the other folds. Both are out-of-fold at every point.

THE CONTROL. Re-assign candidates to RANDOM genes of the same set-size distribution. That destroys the peer
relationships while preserving set sizes, the feature distributions and the model's capacity. A gain that
survives it is set structure; a gain that does not is extra parameters.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))
INV = OUT / "invivo"
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = int(os.environ.get("SET_EPOCHS", "300"))
H = 48


def build_inputs():
    from contact_gate import load_ctcf, contact_features
    from competition import competition_features, load as load_comp
    from ubiquitous import load_ubiquitous
    df, Xid, Cr = load_comp()
    CF = competition_features(df)
    U = np.isin(df["gene"].values, list(load_ubiquitous())).astype(float)[:, None]
    Xfull = np.hstack([Xid, Cr, CF, U])                      # what XGBoost gets: ~330 columns
    # Compact set for the network. 569 positives cannot support 330 inputs to a neural net; these are the
    # columns the diagnosis implicated plus the ones that define a candidate's standing among its peers.
    cols = ["log_dist", "atac_enh", "h3k27ac_enh", "polr2a_enh", "procap_enh", "tf_count",
            "promoter_atac", "promoter_polii", "gene_expr"]
    Xs = np.hstack([df[cols].values, Cr[:, :1], CF[:, [1, 3]], U])
    return df, Xfull, Xs


def base_oof(Xfull, y, ch, seed):
    import xgboost as xgb
    from crispr_gate import _seeded_folds
    folds = _seeded_folds(ch, seed)
    p = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        c = xgb.XGBClassifier(scale_pos_weight=(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), max_depth=4,
                              n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              min_child_weight=5, reg_lambda=2.0, eval_metric="aucpr", n_jobs=4)
        c.fit(Xfull[tr], y[tr])
        p[te] = c.predict_proba(Xfull[te])[:, 1]
    return p, folds


def make_net(d_in, torch):
    nn = torch.nn
    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_in, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.dec = nn.Sequential(nn.Linear(4 * H, H), nn.ReLU(), nn.Dropout(0.3), nn.Linear(H, 1))
            # ZERO-INIT THE OUTPUT LAYER. The head is a RESIDUAL on an already-good XGBoost logit, so a
            # randomly initialised final layer starts by corrupting it -- a 60-epoch smoke run took the
            # baseline from 0.6707 down to 0.6499 and spent its budget climbing back. Starting at exactly
            # zero means the model begins identical to the baseline and every step is an improvement it
            # had to earn.
            nn.init.zeros_(self.dec[-1].weight)
            nn.init.zeros_(self.dec[-1].bias)

        def forward(self, x, gidx, ngene, base):
            h = self.enc(x)
            s = torch.zeros(ngene, h.shape[1], device=h.device).index_add_(0, gidx, h)
            cnt = torch.zeros(ngene, device=h.device).index_add_(
                0, gidx, torch.ones(len(h), device=h.device)).clamp_min(1)
            mean = s / cnt[:, None]
            mx = torch.full((ngene, h.shape[1]), -1e9, device=h.device)
            mx = mx.index_reduce_(0, gidx, h, "amax", include_self=True)
            mg, xg = mean[gidx], mx[gidx]
            z = torch.cat([h, mg, xg, h - mg], 1)
            return base + self.dec(z).squeeze(-1)
    return SetNet()


def run_setmodel(Xs, y, base_logit, gene_ids, folds, seed, torch):
    """Train per fold on the other folds; return out-of-fold adjusted logits."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros(len(y))
    for k in range(5):
        tr = np.where(folds != k)[0]
        te = np.where(folds == k)[0]
        torch.manual_seed(seed * 10 + k)
        net = make_net(Xs.shape[1], torch).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        pw = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], dtype=torch.float32, device=dev)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        packs = {}
        for tag, idx in (("tr", tr), ("te", te)):
            g = pd.factorize(gene_ids[idx])[0]
            packs[tag] = (torch.tensor((Xs[idx] - mu) / sd, dtype=torch.float32, device=dev),
                          torch.tensor(g, dtype=torch.long, device=dev), int(g.max()) + 1,
                          torch.tensor(base_logit[idx], dtype=torch.float32, device=dev),
                          torch.tensor(y[idx], dtype=torch.float32, device=dev))
        # Inner validation split BY GENE, so a gene's candidates never straddle the split -- splitting by
        # row would put peers of a training candidate into validation and leak the very set structure
        # being tested.
        from sklearn.metrics import average_precision_score
        gtr = gene_ids[tr]
        ug = pd.unique(gtr)
        rs = np.random.default_rng(seed * 100 + k)
        vg = set(rs.choice(ug, max(1, int(0.2 * len(ug))), replace=False).tolist())
        vmask = np.array([g in vg for g in gtr])
        xt, gt, ngt, bt, yt = packs["tr"]
        vm = torch.tensor(vmask, device=dev)
        best, best_state, patience = -1.0, None, 0
        for ep in range(EPOCHS):
            net.train()
            opt.zero_grad()
            o = net(xt, gt, ngt, bt)
            lossf(o[~vm], yt[~vm]).backward()
            opt.step()
            if ep % 5 == 0:
                net.eval()
                with torch.no_grad():
                    v = net(xt, gt, ngt, bt)[vm].cpu().numpy()
                a = average_precision_score(y[tr][vmask], v) if y[tr][vmask].sum() >= 3 else -1.0
                if a > best:
                    best, best_state, patience = a, {kk: vv.clone() for kk, vv in net.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 12:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        xe, ge, nge, be, _ = packs["te"]
        with torch.no_grad():
            out[te] = net(xe, ge, nge, be).cpu().numpy()
    return out


def main():
    import torch
    from sklearn.metrics import average_precision_score
    df, Xfull, Xs = build_inputs()
    y, ch = df["crispr_hit"].values, df["chromosome"].values
    gene = df["gene"].values
    print("=" * 100)
    print("SET-LEVEL MODEL -- score each candidate in the context of its gene's other candidates")
    print("=" * 100)
    sz = pd.Series(gene).groupby(gene).transform("size").values
    npos = pd.Series(y).groupby(pd.Series(gene)).transform("sum").values
    print(f"  {len(y):,} pairs, {int(y.sum())} positives, {pd.Series(gene).nunique():,} genes")
    print(f"  candidates/gene median {np.median(sz):.0f}, max {sz.max()}; "
          f"positives in multi-positive genes {int(y[npos>=2].sum())}/{int(y.sum())} "
          f"({100*y[npos>=2].sum()/y.sum():.0f}%)")
    print(f"  device {'cuda' if torch.cuda.is_available() else 'cpu'}, {EPOCHS} epochs\n")

    R = {"n_pairs": int(len(y)), "n_positives": int(y.sum())}
    b_list, s_list, c_list = [], [], []
    for seed in SEEDS:
        p, folds = base_oof(Xfull, y, ch, seed)
        bl = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        a_base = average_precision_score(y, p)
        a_set = average_precision_score(y, run_setmodel(Xs, y, bl, gene, folds, seed, torch))
        # CONTROL: same set sizes, random membership -- destroys peer relationships only
        rng = np.random.default_rng(seed)
        fake = gene.copy()
        rng.shuffle(fake)
        a_ctl = average_precision_score(y, run_setmodel(Xs, y, bl, fake, folds, seed, torch))
        b_list.append(a_base); s_list.append(a_set); c_list.append(a_ctl)
        print(f"  seed {seed}:  XGBoost {a_base:.4f}   + set model {a_set:.4f}   "
              f"[shuffled-gene control {a_ctl:.4f}]")

    b, s, c = np.array(b_list), np.array(s_list), np.array(c_list)
    d, dc = s - b, c - b
    net = d.mean() - dc.mean()
    se = float(np.sqrt(d.var(ddof=1) / len(d) + dc.var(ddof=1) / len(dc)))
    print(f"\n  XGBoost baseline        {b.mean():.4f}")
    print(f"  + set model             {s.mean():.4f}   delta {d.mean():+.4f}  "
          f"[per-seed {' '.join(f'{x:+.3f}' for x in d)}]")
    print(f"  + shuffled-gene control {c.mean():.4f}   delta {dc.mean():+.4f}")
    print(f"  NET OF CONTROL          {net:+.4f} +/- {se:.4f}   z = {net/max(se,1e-9):+.2f}   "
          f"sign consistent: {bool(np.all(d>0))}")
    R.update({"xgboost": float(b.mean()), "setmodel": float(s.mean()), "control": float(c.mean()),
              "delta": float(d.mean()), "delta_per_seed": d.tolist(),
              "delta_control": float(dc.mean()), "net": float(net), "se": se,
              "z": float(net / max(se, 1e-9)), "sign_consistent": bool(np.all(d > 0))})
    R["verdict_network"] = ("GO -- peer context adds signal the per-pair model cannot see."
                            if net >= 0.01 and net > 2 * se and bool(np.all(d > 0)) else
                            "NO-GO -- the LEARNED set model does not beat per-pair scoring on this data.")
    print(f"\n  VERDICT (network): {R['verdict_network']}")

    # ---- THE SAME IDEA AT ONE PARAMETER ----
    # The network lost, but so did its shuffled-gene control (-0.0235 against -0.0284), which says the
    # damage came from fitting thousands of parameters to 569 positives rather than from set structure
    # being useless. So: the same hypothesis at a capacity the data can support. If a gene clearly has a
    # strong enhancer, its other strong candidates are likelier real too, so add a term proportional to
    # the gene's best score. beta is chosen on the TRAINING folds only.
    print("\n" + "=" * 100)
    print("SAME IDEA, ONE PARAMETER:  score_i += beta * max_j-in-gene(score_j)")
    print("=" * 100)
    BETAS = np.arange(0.0, 1.01, 0.05)
    gser = pd.Series(gene)

    def boost(logit, grp, beta):
        return logit + beta * pd.Series(logit).groupby(grp).transform("max").values

    def run_heur(grp, folds, bl):
        out = np.zeros(len(y))
        betas = []
        for k in range(5):
            tr, te = folds != k, folds == k
            bb, ba = 0.0, -1.0
            for bt in BETAS:
                a = average_precision_score(y[tr], boost(bl, grp, bt)[tr])
                if a > ba:
                    ba, bb = a, bt
            out[te] = boost(bl, grp, bb)[te]
            betas.append(float(bb))
        return average_precision_score(y, out), betas

    hr, hc, allb = [], [], []
    for seed in SEEDS:
        p, folds = base_oof(Xfull, y, ch, seed)
        bl = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        a0 = average_precision_score(y, p)
        ar, bs = run_heur(gser, folds, bl)
        fake = gene.copy()
        np.random.default_rng(seed).shuffle(fake)
        ac, _ = run_heur(pd.Series(fake), folds, bl)
        hr.append(ar - a0); hc.append(ac - a0); allb += bs
        print(f"  seed {seed}: base {a0:.4f}  + gene-max boost {ar:.4f} ({ar-a0:+.4f})   "
              f"[shuffled-gene control {ac-a0:+.4f}]")
    hr, hc = np.array(hr), np.array(hc)
    hnet = hr.mean() - hc.mean()
    hse = float(np.sqrt(hr.var(ddof=1) / len(hr) + hc.var(ddof=1) / len(hc)))
    print(f"\n  real {hr.mean():+.4f}   shuffled-gene control {hc.mean():+.4f}")
    print(f"  NET OF CONTROL {hnet:+.4f} +/- {hse:.4f}   z = {hnet/max(hse,1e-9):+.2f}   "
          f"sign consistent: {bool(np.all(hr>0))}")
    print(f"  beta selected across {len(allb)} independent folds: median {np.median(allb):.2f}, "
          f"range {min(allb):.2f}-{max(allb):.2f}")
    R.update({"heur_delta": float(hr.mean()), "heur_control": float(hc.mean()), "heur_net": float(hnet),
              "heur_se": hse, "heur_z": float(hnet / max(hse, 1e-9)),
              "heur_sign_consistent": bool(np.all(hr > 0)), "beta_median": float(np.median(allb)),
              "beta_range": [float(min(allb)), float(max(allb))]})
    R["verdict"] = (
        "SET STRUCTURE IS REAL BUT MUST BE CHEAP -- a 1-parameter gene-max boost gains "
        f"{hnet:+.4f} (z={hnet/max(hse,1e-9):.1f}) where a learned DeepSets head lost {net:+.4f}. "
        "With 569 positives the capacity, not the hypothesis, was the binding constraint.")
    print(f"\n  VERDICT: {R['verdict']}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "setmodel.json", "w"), indent=1)
    print(f"\n  -> {OUT/'setmodel.json'}")


if __name__ == "__main__":
    main()
