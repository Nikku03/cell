"""transformer_ko2 -- improved knockout-context transformer. Same set-of-tokens input as transformer_ko (perturbed gene + its post-KO
partners, each token = [DepMap | network-SVD | essentiality | log-degree | role]), but four principled upgrades aimed at closing the gap
to the retrieval ORACLE (0.607); v1 sat at 0.481:

  1. CLS cross-attention pooling  -- a learnable query token decides WHICH partners matter, instead of mean-pooling all tokens equally.
  2. Distillation objective        -- row-wise softmax cross-entropy that makes each KO's EMBEDDING-neighbour distribution match its true
                                      tide-removed RESPONSE-neighbour distribution (InfoNCE-style; optimises retrieval ranking directly,
                                      vs v1's pairwise cosine-MSE).
  3. Similarity-weighted top-K transfer -- softmax(sim/tau)-weighted mean of the K nearest train KOs (K, tau tuned ON VAL), vs v1's uniform
                                      mean of 10.
  4. Hybrid similarity             -- val-tuned blend of the learned metric with raw DepMap cosine, to recover neighbours the learned
                                      metric misses (the wall_combine lesson, inside retrieval).

Scored on the same held-out K562 bench (tide-removed specific-mover recall@50), 3 splits, mean+/-sd, WITH the v1 transformer re-trained in
the SAME split as the in-run baseline (apples-to-apples gap) + the wrong-KO gene-identity control. Deterministic per split. CPU.
"""
import json
import os
from pathlib import Path
import numpy as np
from eval_harness import Harness
from neural_ko import build_xy, mean_recall
from transformer_ko import build_context
OUT = Path(os.environ.get("CELL_OUT", "outputs/orphan"))


def run(SEED, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask):
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(4)
    rng = np.random.RandomState(SEED)
    scor = [k for k in H.scorable if k in hidx]
    perm = rng.permutation(len(scor)); n_test = int(0.30 * len(scor))
    test = sorted(scor[i] for i in perm[:n_test]); test_set = set(test)
    train_all = [k for k in have if k not in test_set]
    tperm = rng.permutation(len(train_all)); n_val = int(0.12 * len(train_all))
    val = sorted(train_all[i] for i in tperm[:n_val] if train_all[i] in H.scorable); val_set = set(val)
    train = [k for k in train_all if k not in val_set]
    tr_arr = np.array([hidx[k] for k in train])

    Ft = torch.tensor(Fg); TI = torch.tensor(tok_idx); TR = torch.tensor(tok_role); TM = torch.tensor(tok_mask)
    Yspec = (Y * H.nontide.astype(np.float32)); Yn = Yspec / (np.linalg.norm(Yspec, axis=1, keepdims=True) + 1e-9)
    Ynt = torch.tensor(Yn)
    FEAT = Fg.shape[1]; DM = 128
    # DepMap KO-cosine for the hybrid readout
    dv = np.load(OUT / "depmap_vecs.npz", allow_pickle=True)
    syms = list(dv["syms"]); Z = dv["Z"].astype(np.float32); srow = {s: i for i, s in enumerate(syms)}
    XK = np.stack([Z[srow[k]] for k in have]); XKn = XK / (np.linalg.norm(XK, axis=1, keepdims=True) + 1e-9)

    def gather(idx):
        f = Ft[TI[idx]]; roleoh = torch.nn.functional.one_hot(TR[idx], 4).float()
        return torch.cat([f, roleoh], -1), TM[idx]

    def enc_layer():
        return torch.nn.TransformerEncoderLayer(DM, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")

    class V1(torch.nn.Module):                                # baseline: mean-pool, cosine-MSE, mean transfer
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEAT + 4, DM); s.enc = torch.nn.TransformerEncoder(enc_layer(), 2)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, x, mask):
            h = s.enc(s.proj(x), src_key_padding_mask=(mask < 0.5))
            pooled = (h * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
            e = s.head(pooled); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    class V2(torch.nn.Module):                                # CLS cross-attention pooling
        def __init__(s):
            super().__init__()
            s.proj = torch.nn.Linear(FEAT + 4, DM); s.cls = torch.nn.Parameter(torch.randn(1, 1, DM) * 0.02)
            s.enc = torch.nn.TransformerEncoder(enc_layer(), 3)
            s.head = torch.nn.Sequential(torch.nn.LayerNorm(DM), torch.nn.Linear(DM, 64))
        def forward(s, x, mask):
            h = s.proj(x); B = h.size(0)
            h = torch.cat([s.cls.expand(B, 1, DM), h], 1)
            m = torch.cat([torch.ones(B, 1), mask], 1)
            h = s.enc(h, src_key_padding_mask=(m < 0.5))
            e = s.head(h[:, 0]); return e / (e.norm(dim=1, keepdim=True) + 1e-9)

    def embed_all(model):
        model.eval()
        with torch.no_grad():
            xs, ms = gather(torch.arange(len(have)))
            E = model(xs, ms).numpy()
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def mean_transfer(E):                                     # v1 readout: uniform mean of top-10
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            top = tr_arr[np.argsort(-(E[tr_arr] @ E[i]))[:10]]
            return Y[top].mean(0)
        return f

    def weighted_transfer(E, K, tau, alpha):                 # v2 readout: hybrid sim, softmax-weighted top-K
        def f(ko):
            i = hidx.get(ko)
            if i is None:
                return None
            s = (1 - alpha) * (E[tr_arr] @ E[i]) + alpha * (XKn[tr_arr] @ XKn[i])
            top = np.argsort(-s)[:K]; w = np.exp((s[top] - s[top].max()) / tau); w /= w.sum() + 1e-9
            return (w[:, None] * Y[tr_arr[top]]).sum(0)
        return f

    def train_v1(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(220):
            model.train(); pi = np.random.RandomState(SEED * 100 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]]); xs, ms = gather(idx); e = model(xs, ms)
                loss = ((e @ e.T - Ynt[idx] @ Ynt[idx].T) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, mean_transfer(embed_all(model)))
                if vr > best:
                    best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                if wait >= 18:
                    break
        if bstate:
            model.load_state_dict(bstate)
        return model

    def train_v2(model, te=0.1, tt=0.1):                      # distillation objective
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, wait = -1.0, None, 0
        for ep in range(240):
            model.train(); pi = np.random.RandomState(SEED * 100 + 7 + ep).permutation(len(tr_arr))
            for bs in range(0, len(tr_arr), 256):
                idx = torch.tensor(tr_arr[pi[bs:bs + 256]])
                if len(idx) < 8:
                    continue
                xs, ms = gather(idx); e = model(xs, ms)
                Se = e @ e.T; St = Ynt[idx] @ Ynt[idx].T
                dmask = torch.eye(len(idx)) * (-1e9)
                logP = torch.log_softmax(Se / te + dmask, dim=1)
                Q = torch.softmax(St / tt + dmask, dim=1)
                loss = -(Q * logP).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                vr, _ = mean_recall(H, val, weighted_transfer(embed_all(model), 10, 0.1, 0.0))
                if vr > best:
                    best, bstate, wait = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    wait += 1
                if wait >= 20:
                    break
        if bstate:
            model.load_state_dict(bstate)
        return model

    # train both
    m1 = train_v1(V1()); E1 = embed_all(m1)
    m2 = train_v2(V2()); E2 = embed_all(m2)

    # tune the v2 readout (K, tau, alpha) ON VAL, then apply to test
    best_cfg, best_vr = (10, 0.1, 0.0), -1.0
    for K in (5, 10, 20):
        for tau in (0.05, 0.1, 0.3):
            for alpha in (0.0, 0.3):
                vr, _ = mean_recall(H, val, weighted_transfer(E2, K, tau, alpha))
                if vr > best_vr:
                    best_vr, best_cfg = vr, (K, tau, alpha)
    K, tau, alpha = best_cfg

    def tide_f(ko): return H.mover_freq
    def oracle_f(ko): return H._references(ko, 50)["ORACLE*"]
    sh = rng.permutation(len(test)); smap = {test[i]: test[sh[i]] for i in range(len(test))}
    def v2_shuf(ko):
        w = smap.get(ko); j = hidx.get(w)
        if j is None:
            return None
        s = (1 - alpha) * (E2[tr_arr] @ E2[j]) + alpha * (XKn[tr_arr] @ XKn[j])
        top = np.argsort(-s)[:K]; ww = np.exp((s[top] - s[top].max()) / tau); ww /= ww.sum() + 1e-9
        return (ww[:, None] * Y[tr_arr[top]]).sum(0)

    Kk = 50; res = {}
    res["TIDE-null"], _ = mean_recall(H, test, tide_f, Kk)
    res["v1 transformer (baseline)"], _ = mean_recall(H, test, mean_transfer(E1), Kk)
    res["v2 CLS+distill+weighted"], _ = mean_recall(H, test, weighted_transfer(E2, K, tau, alpha), Kk)
    res["v2 shuffle (control)"], _ = mean_recall(H, test, v2_shuf, Kk)
    res["ORACLE* (ceiling)"], _ = mean_recall(H, test, oracle_f, Kk)
    print(f"    [seed {SEED}] readout tuned on val -> K={K} tau={tau} alpha={alpha}")
    return res


def main():
    H = Harness("K562")
    have, X, Y = build_xy(H)
    hidx = {k: i for i, k in enumerate(have)}
    Fg, tok_idx, tok_role, tok_mask = build_context(H, have)
    seeds = [0, 1, 2]; all_res = []
    for s in seeds:
        print(f"\n{'=' * 60}\n[SPLIT SEED {s}]")
        res = run(s, H, have, Y, hidx, Fg, tok_idx, tok_role, tok_mask)
        for k, v in res.items():
            print(f"    {k:32s} {v:.3f}")
        all_res.append(res)
    keys = list(all_res[0].keys())
    agg = {k: (float(np.mean([r[k] for r in all_res])), float(np.std([r[k] for r in all_res]))) for k in keys}
    print(f"\n{'=' * 60}\nSTABILITY across {len(seeds)} splits (mean +/- sd):")
    for k in keys:
        print(f"    {k:32s} {agg[k][0]:.3f} +/- {agg[k][1]:.3f}")
    V2 = "v2 CLS+distill+weighted"; V1 = "v1 transformer (baseline)"; Tid = "TIDE-null"; Orc = "ORACLE* (ceiling)"
    d = [r[V2] - r[V1] for r in all_res]
    print(f"\n    v2 - v1: {np.mean(d):+.3f} +/- {np.std(d):.3f}  ({[round(x,3) for x in d]})")
    v2m = agg[V2][0]; v1m = agg[V1][0]; tide_m = agg[Tid][0]; orc_m = agg[Orc][0]; shuf_m = agg["v2 shuffle (control)"][0]
    improves = np.mean(d) > 0.01 and all(x > 0 for x in d)
    shuffle_ok = shuf_m < v2m - 0.02
    head_v2 = (v2m - tide_m) / max(orc_m - tide_m, 1e-9); head_v1 = (v1m - tide_m) / max(orc_m - tide_m, 1e-9)
    verdict = (
        "IMPROVED knockout-context transformer (transformer_ko2.py): v1's set-of-tokens transformer + four upgrades (CLS cross-attention "
        "pooling, a distillation/InfoNCE objective distilling the response-neighbour distribution into the embedding, val-tuned "
        "similarity-weighted top-K transfer, and a val-tuned hybrid with raw DepMap cosine). 3-split held-out K562 (tide-removed specific "
        f"recall@50): v1 baseline {v1m:.3f} -> v2 {v2m:.3f} (oracle {orc_m:.3f}, tide {tide_m:.3f}). v2 - v1 = {np.mean(d):+.3f}+/-{np.std(d):.3f} "
        + ("(positive every split) -- the upgrades genuinely improve neighbour selection/transfer; head-room used rises "
           f"{head_v1:.0%} -> {head_v2:.0%} of tide->oracle."
           if improves else
           "-- NO stable improvement: the upgrades do not beat v1 by a reliable margin, so v1's mean-pool + cosine-MSE + mean-transfer was "
           "already near the best this representation supports; the remaining gap to the oracle is representation/data-limited, not a "
           "readout/objective problem.") + " "
        + (f"Gene-identity control PASSES (wrong-KO {shuf_m:.3f} < v2)." if shuffle_ok else
           f"Gene-identity control WARNING ({shuf_m:.3f} ~ v2).") +
        " Single cell type, steady-state mRNA, magnitude target, CPU; deterministic per split.")
    print(f"\nVERDICT: {verdict}")
    out = {"mean_sd": {k: [round(agg[k][0], 4), round(agg[k][1], 4)] for k in keys},
           "v2_minus_v1": [round(x, 4) for x in d], "v2_improves_all_splits": bool(improves),
           "shuffle_control_passes": bool(shuffle_ok), "headroom_v1": round(float(head_v1), 3),
           "headroom_v2": round(float(head_v2), 3), "verdict": verdict, "note": verdict}
    json.dump(out, open(OUT / "transformer_ko2.json", "w"), indent=1)
    print("\n  -> outputs/orphan/transformer_ko2.json")


if __name__ == "__main__":
    main()
